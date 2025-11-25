# DFT/scf/xc/hybrid.py
# -*- coding: utf-8 -*-
"""
B3LYP-lite (RKS) exchange-correlation functional
- Exchange : (1-a0) * [ (1-aX) * LDAx + aX * B88x ]
  ※ HF 交換の a0 は Fock の -a0*K 側で扱うので、ここでは (1-a0) を掛ける。
- Correlation : aC * LYP   （VWN は未実装／将来拡張）

方針：
- evaluate_exchange_correlation() は (Exc: float, Vxc: ndarray) を返す（互換）
- チャンク有効時：交換は assemble_local_potential_chunked、相関は
  LDC_LYP.evaluate_chunked() を使う（未定義メソッド呼び出しを廃止）
- validate_streaming=True なら、ベクトル版の参照値と Exc / Vxc を突き合わせ、
  不一致なら丸ごと参照値へフォールバック
"""

from __future__ import annotations

import numpy as np

from .utils import (
    dprint,
    assemble_local_potential_chunked,
)
from .utils import compute_rho_grad_phi  # ベクトル版参照
from .lda import (
    slater_exchange_density,         # eps_x^LDA(ρ)
    slater_exchange_potential,       # v_x^LDA(ρ) ≈ 4/3 * eps_x^LDA
)
from .gga import (
    b88_enhancement_factor,
    LDC_LYP,
)


class B3LYP:
    """
    B3LYP-lite hybrid XC（RKS）
    """

    def __init__(
        self,
        grid,
        basis_functions,
        density_matrix,
        *,
        chunk_size=None,
        d_thresh: float = 0.0,
        validate_streaming: bool = True,
        use_vwn: bool = False,
    ):
        self.grid = grid
        self.basis = basis_functions
        self.D = np.asarray(density_matrix, dtype=np.float64)

        # 既定パラメータ（dft.py から上書きされ得る）
        self.a0 = 0.20  # HF混合率（Fock 側）
        self.aX = 0.72  # B88 交換寄与の混合
        self.aC = 0.81  # LYP 相関の混合（VWN は現状未実装）
        self.use_vwn = bool(use_vwn)

        self.chunk_size = int(chunk_size) if chunk_size else None
        self.d_thresh = float(d_thresh)
        self.validate_streaming = bool(validate_streaming)

        # 診断
        self.last = {
            "Ex": None, "Ec": None, "Exc": None,
            "norm_Vx_F": None, "norm_Vc_F": None,
            "validate": {"used_streaming": bool(self.chunk_size), "fallback": False, "mismatch": {}},
        }

    # ========= 交換：簡易B88（LDA×Fx） =========
    def _ex_energy_density(self, rho: np.ndarray, grad: np.ndarray):
        """
        eps_x ≈ (1-a0) * [ (1-aX) * eps_x^LDA + aX * (eps_x^LDA * Fx) ]
        v_x   ≈ (1-a0) * v_x^LDA * [ (1-aX) + aX * Fx ]
        """
        eps = 1e-14
        rho_safe = np.clip(rho, eps, None)

        eps_x_lda = slater_exchange_density(rho_safe)   # per-electron
        Fx = b88_enhancement_factor(rho_safe, grad)     # 増強因子（簡易）
        eps_x = (1.0 - self.a0) * ((1.0 - self.aX) * eps_x_lda + self.aX * (eps_x_lda * Fx))

        v_x_lda = slater_exchange_potential(eps_x_lda)
        v_x = (1.0 - self.a0) * v_x_lda * ((1.0 - self.aX) + self.aX * Fx)

        return eps_x, v_x

    # ========= メイン入口 =========
    def evaluate_exchange_correlation(self):
        if not self.chunk_size:
            return self._evaluate_vectorized()

        # --- ストリーミング ---
        Vx, Ex = self._exchange_chunked()
        Vc, Ec = self._correlation_chunked()  # ← 修正済：戻り (Vc, Ec) を保証

        Exc = float(Ex + self.aC * Ec)
        Vxc = 0.5 * ((Vx + self.aC * Vc) + (Vx + self.aC * Vc).T)

        # 参照（ベクトル版）と突合・フォールバック
        if self.validate_streaming:
            Exc_ref, Vxc_ref, diag = self._evaluate_reference_vectorized()
            ok_E = np.isclose(Exc, Exc_ref, rtol=1e-7, atol=1e-8)
            diff_V = np.linalg.norm(Vxc - Vxc_ref)

            self.last["validate"]["mismatch"] = {"|Exc-Exc_ref|": abs(Exc - Exc_ref), "||Vxc-Vref||_F": diff_V}
            if (not ok_E) or (diff_V > 1e-6):
                dprint(f"[xc] streaming/vectorized mismatch -> fallback "
                       f"(dE={Exc-Exc_ref:+.3e}, ||dV||_F={diff_V:.3e})")
                self.last["validate"]["fallback"] = True
                Exc, Vxc = Exc_ref, Vxc_ref

        self.last.update({"Ex": float(Ex), "Ec": float(Ec), "Exc": float(Exc),
                          "norm_Vx_F": float(np.linalg.norm(Vx)), "norm_Vc_F": float(np.linalg.norm(Vc))})
        return Exc, Vxc

    # ---- ベクトル版 ----
    def _evaluate_vectorized(self):
        pts, w = self.grid.points, self.grid.weights
        rho, grad, phi, _ = compute_rho_grad_phi(self.basis, self.D, pts)

        # 交換
        eps_x, v_x = self._ex_energy_density(rho, grad)
        Ex = float(np.dot(w, rho * eps_x))
        Vx = phi @ (((w * v_x)[None, :]) * phi).T
        Vx = 0.5 * (Vx + Vx.T)

        # 相関（LYP）：(Ec, Vc) の順
        lyp = LDC_LYP(self.grid, self.basis, self.D)
        Ec, Vc = lyp.evaluate()
        Vc = 0.5 * (Vc + Vc.T)

        Exc = float(Ex + self.aC * Ec)
        Vxc = 0.5 * ((Vx + self.aC * Vc) + (Vx + self.aC * Vc).T)

        self.last.update({"Ex": Ex, "Ec": Ec, "Exc": Exc,
                          "norm_Vx_F": float(np.linalg.norm(Vx)), "norm_Vc_F": float(np.linalg.norm(Vc)),
                          "validate": {"used_streaming": False, "fallback": False, "mismatch": {}}})
        return Exc, Vxc

    # ---- 交換：チャンク ----
    def _exchange_chunked(self):
        def v_x_func(rho_chunk, grad_chunk):
            _, v = self._ex_energy_density(rho_chunk, grad_chunk)
            return v

        def eps_x_func(rho_chunk, grad_chunk):
            e, _ = self._ex_energy_density(rho_chunk, grad_chunk)
            return e

        Vx, Ex = assemble_local_potential_chunked(
            self.grid, self.basis, self.D,
            v_local_func=v_x_func,
            chunk_size=self.chunk_size,
            d_thresh=self.d_thresh,
            return_energy_density=True,
            energy_density_func=eps_x_func,
        )
        Vx = 0.5 * (Vx + Vx.T)
        return Vx, float(Ex)

    # ---- 相関：チャンク ----
    def _correlation_chunked(self):
        """
        LDC_LYP.evaluate_chunked() は (Ec, Vc) を返す点に注意。
        ここでは (Vc, Ec) で返す一貫 API を提供する。
        """
        lyp = LDC_LYP(self.grid, self.basis, self.D)
        Ec, Vc = lyp.evaluate_chunked(chunk_size=self.chunk_size, d_thresh=self.d_thresh)  # ← 修正点
        Vc = 0.5 * (Vc + Vc.T)
        return Vc, float(Ec)  # ← (Vc, Ec) で返す

    # ---- 参照：ベクトル版（検証用）----
    def _evaluate_reference_vectorized(self):
        pts, w = self.grid.points, self.grid.weights
        rho, grad, phi, _ = compute_rho_grad_phi(self.basis, self.D, pts)

        eps_x_ref, v_x_ref = self._ex_energy_density(rho, grad)
        Ex_ref = float(np.dot(w, rho * eps_x_ref))
        Vx_ref = phi @ (((w * v_x_ref)[None, :]) * phi).T
        Vx_ref = 0.5 * (Vx_ref + Vx_ref.T)

        lyp_ref = LDC_LYP(self.grid, self.basis, self.D)
        Ec_ref, Vc_ref = lyp_ref.evaluate()
        Vc_ref = 0.5 * (Vc_ref + Vc_ref.T)

        Exc_ref = float(Ex_ref + self.aC * Ec_ref)
        Vxc_ref = 0.5 * ((Vx_ref + self.aC * Vc_ref) + (Vx_ref + self.aC * Vc_ref).T)

        return Exc_ref, Vxc_ref, {"Ex": Ex_ref, "Ec": Ec_ref}