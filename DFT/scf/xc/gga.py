# DFT/scf/xc/gga.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Tuple
from .utils import compute_rho_grad_phi, assemble_local_potential, safe_pow
from .utils import assemble_local_potential_chunked  # 追加
# －－ 上の import は同一ファイル内関数の2行を残しつつ、チャンク版も使う －－

# －－－－ Becke'88 exchange enhancement factor (簡易・RKS) －－－－
def b88_enhancement_factor(rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
    """
    Minimal, numerically safe B88 enhancement factor for closed-shell:
    F_x ≈ 1 + β * s / (1 + s), s ~ |∇ρ| / ρ^{4/3}
    """
    eps = 1e-14
    rho_safe = np.clip(rho, eps, None)
    grad_norm = np.linalg.norm(grad_rho, axis=1)
    s = grad_norm / (rho_safe**(4.0 / 3.0) + eps)
    beta = 0.0042
    Fx = 1.0 + beta * s / (1.0 + s)
    return Fx

# －－－－ LYP correlation (簡易・RKS・安定化) －－－－
class LDC_LYP:
    """
    Spin-unpolarized LYP correlation (restricted).
    evaluate() ＝ 従来の全点ベクトル化
    evaluate_chunked() ＝ チャンクストリーミング
    """
    def __init__(self, grid, basis, Dmat, params=None):
        self.grid = grid
        self.basis = basis
        self.D = np.asarray(Dmat, dtype=np.float64)
        default_params = {
            "A": 0.04918,
            "b": 0.132,
            "c": 0.2533,
            "d": 0.349,
        }
        if params is None:
            params = default_params
        else:
            for k, v in default_params.items():
                params.setdefault(k, v)
        self.params = params

    def _eps_vc(self, rho, grad_rho):
        A = self.params["A"]
        b = self.params["b"]
        c = self.params["c"]
        d_c = self.params["d"]
        eps = 1e-14
        rho_safe = np.clip(rho, eps, None)
        if grad_rho is not None and grad_rho.ndim == 2 and grad_rho.shape[1] == 3:
            grad_norm = np.linalg.norm(grad_rho, axis=1)
        else:
            grad_norm = np.zeros_like(rho_safe)
        rs = (3.0 / (4.0 * np.pi * rho_safe)) ** (1.0 / 3.0)
        y = 1.0 + d_c * rs
        rho_43 = rho_safe ** (4.0 / 3.0)
        denom = y**4 + eps
        eps_c = -A * rho_43 * (1.0 + b * rs) / denom
        # local potential (簡易)
        depsc_drho = (-A * (4.0 / 3.0) * rho_safe ** (1.0 / 3.0) * (1.0 + b * rs) / denom)
        v_c = depsc_drho
        return eps_c, v_c

    # －－ 従来の全点版（互換維持） －－
    def evaluate(self) -> Tuple[float, np.ndarray]:
        rho, grad_rho, phi, _ = compute_rho_grad_phi(self.basis, self.D, self.grid.points)
        eps_c, v_c = self._eps_vc(rho, grad_rho)
        Ec = float(np.dot(self.grid.weights, rho * eps_c))
        Vc = assemble_local_potential(self.grid, phi, v_c)
        return Ec, Vc

    # －－ 新規：チャンク・ストリーミング版 －－
    def evaluate_chunked(self, *, chunk_size: int = 20000, d_thresh: float = 0.0) -> Tuple[float, np.ndarray]:
        def v_c_func(rho_chunk, grad_chunk):
            _, v_c = self._eps_vc(rho_chunk, grad_chunk)
            return v_c

        def eps_c_func(rho_chunk, grad_chunk):
            eps_c, _ = self._eps_vc(rho_chunk, grad_chunk)
            return eps_c

        Vc, Ec = assemble_local_potential_chunked(
            self.grid, self.basis, self.D,
            v_local_func=v_c_func,
            chunk_size=chunk_size, d_thresh=d_thresh,
            return_energy_density=True, energy_density_func=eps_c_func
        )
        return Ec, Vc
