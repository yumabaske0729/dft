# DFT/scf/xc/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np
from typing import Tuple, Iterable, Dict, Any, Callable

# －－－－ diagnostics flag －－－－
_DEBUG = bool(int(os.getenv("DEBUG_DIAG", "0") != "0"))
def dprint(msg: str):
    if _DEBUG:
        print(msg)

# －－－－ numerics －－－－
def safe_pow(x: np.ndarray, p: float) -> np.ndarray:
    return np.power(np.clip(x, 0.0, None), p)

def kahan_sum_increment(total: float, c: float, term: float) -> Tuple[float, float]:
    y = term - c
    t = total + y
    c_new = (t - total) - y
    return t, c_new

# －－－－ 既存：全点版（後方互換のため残す） －－－－
def compute_rho_grad_phi(basis_functions, D: np.ndarray, points: np.ndarray):
    """
    Compute closed-shell rho, grad_rho, AO phi and AO gradients at grid points.
    ※ 全点を一括処理（大規模ではメモリ負荷が高い）。チャンク版は本ファイル下部を使用。

    Returns
    -------
    rho : (N,)
    grad_rho : (N,3)
    phi : (nbf,N)
    grad_phi : (nbf,N,3)
    """
    pts = points
    N = pts.shape[0]
    nbf = len(basis_functions)
    rho = np.zeros(N, dtype=np.float64)
    grad_rho = np.zeros((N, 3), dtype=np.float64)
    phi = np.zeros((nbf, N), dtype=np.float64)
    grad_phi = np.zeros((nbf, N, 3), dtype=np.float64)

    from ...math.gaussian_math import norm_prefactor
    for mu, gto in enumerate(basis_functions):
        l, m, n = gto.l, gto.m, gto.n
        A = gto.center
        R = pts - A
        x, y, z = R[:, 0], R[:, 1], R[:, 2]
        r2 = x * x + y * y + z * z

        val = np.zeros(N, dtype=np.float64)
        gx = np.zeros(N, dtype=np.float64)
        gy = np.zeros(N, dtype=np.float64)
        gz = np.zeros(N, dtype=np.float64)

        for alpha, coef in zip(gto.exponents, gto.coefficients):
            Nfac = norm_prefactor(l, m, n, alpha)
            poly = (x ** l) * (y ** m) * (z ** n)
            e = np.exp(-alpha * r2)
            prim = coef * Nfac * e * poly
            val += prim

            if l > 0: lx = l * (x ** (l - 1)) * (y ** m) * (z ** n)
            else:     lx = 0.0
            if m > 0: ly = m * (x ** l) * (y ** (m - 1)) * (z ** n)
            else:     ly = 0.0
            if n > 0: lz = n * (x ** l) * (y ** m) * (z ** (n - 1))
            else:     lz = 0.0

            gx += coef * Nfac * (e * lx - 2.0 * alpha * x * e * poly)
            gy += coef * Nfac * (e * ly - 2.0 * alpha * y * e * poly)
            gz += coef * Nfac * (e * lz - 2.0 * alpha * z * e * poly)

        phi[mu, :] = val
        grad_phi[mu, :, 0] = gx
        grad_phi[mu, :, 1] = gy
        grad_phi[mu, :, 2] = gz

    # Build rho and grad rho
    # diagonal
    for mu in range(nbf):
        p_mu = phi[mu]
        d_mm = float(D[mu, mu])
        if d_mm != 0.0:
            rho += d_mm * p_mu * p_mu
            grad_rho += d_mm * (p_mu[:, None] * grad_phi[mu])

    # off-diagonal (mu<nu)
    for mu in range(nbf):
        p_mu = phi[mu]
        for nu in range(mu + 1, nbf):
            d_mn = float(D[mu, nu])
            if d_mn == 0.0:
                continue
            c2 = 2.0 * d_mn
            p_nu = phi[nu]
            rho += c2 * p_mu * p_nu
            grad_rho += c2 * (p_nu[:, None] * grad_phi[mu] + p_mu[:, None] * grad_phi[nu])

    return rho, grad_rho, phi, grad_phi

def assemble_local_potential(grid, phi: np.ndarray, v_local: np.ndarray) -> np.ndarray:
    """
    Assemble AO potential matrix from local scalar field v(r):
    V_{μν} = ∫ v(r) φ_μ(r) φ_ν(r) dr ≈ Σ_g w_g v_g φ_μ(g) φ_ν(g)
    """
    w = grid.weights
    proj = (w * v_local)[None, :] * phi  # (nbf, N)
    V = phi @ proj.T                     # (nbf, nbf)
    V = 0.5 * (V + V.T)
    return V

# －－－－ ここから新規：チャンク化実装 －－－－

def _select_density_pairs(D: np.ndarray, d_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dの小要素スクリーニング。
    Returns
    -------
    diag_idx : (K,)  対角で |D_mm|>τ の m の配列
    pairs    : (L,3) offdiagで |D_mn|>τ の (m,n,d_mn) を格納
    """
    nbf = D.shape[0]
    diag_idx = np.array([m for m in range(nbf) if abs(float(D[m, m])) > d_thresh], dtype=int)

    pairs_list = []
    # 上三角のみ列挙
    for m in range(nbf):
        row = D[m, m + 1:]
        if d_thresh > 0.0:
            nz = np.where(np.abs(row) > d_thresh)[0]
            for k in nz:
                n = m + 1 + int(k)
                pairs_list.append((m, n, float(D[m, n])))
        else:
            for n in range(m + 1, nbf):
                dmn = float(D[m, n])
                if dmn != 0.0:
                    pairs_list.append((m, n, dmn))

    pairs = np.array(pairs_list, dtype=float) if pairs_list else np.zeros((0, 3), dtype=float)
    # dtype=float にしたので m,n は整数にキャストして使う
    return diag_idx, pairs

def _compute_phi_grad_for_chunk(basis_functions, pts_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    あるチャンク（N_c点）に対する φ と ∇φ を (nbf,N_c)/(nbf,N_c,3) で返す。
    """
    from ...math.gaussian_math import norm_prefactor

    Nc = pts_chunk.shape[0]
    nbf = len(basis_functions)
    phi = np.zeros((nbf, Nc), dtype=np.float64)
    grad = np.zeros((nbf, Nc, 3), dtype=np.float64)

    for mu, gto in enumerate(basis_functions):
        l, m, n = gto.l, gto.m, gto.n
        A = gto.center
        R = pts_chunk - A[None, :]
        x, y, z = R[:, 0], R[:, 1], R[:, 2]
        r2 = x * x + y * y + z * z

        val = np.zeros(Nc, dtype=np.float64)
        gx = np.zeros(Nc, dtype=np.float64)
        gy = np.zeros(Nc, dtype=np.float64)
        gz = np.zeros(Nc, dtype=np.float64)

        # 収縮和
        for alpha, coef in zip(gto.exponents, gto.coefficients):
            Nfac = norm_prefactor(l, m, n, alpha)
            poly = (x ** l) * (y ** m) * (z ** n)
            e = np.exp(-alpha * r2)
            prim = coef * Nfac * e * poly
            val += prim

            if l > 0: lx = l * (x ** (l - 1)) * (y ** m) * (z ** n)
            else:     lx = 0.0
            if m > 0: ly = m * (x ** l) * (y ** (m - 1)) * (z ** n)
            else:     ly = 0.0
            if n > 0: lz = n * (x ** l) * (y ** m) * (z ** (n - 1))
            else:     lz = 0.0

            gx += coef * Nfac * (e * lx - 2.0 * alpha * x * e * poly)
            gy += coef * Nfac * (e * ly - 2.0 * alpha * y * e * poly)
            gz += coef * Nfac * (e * lz - 2.0 * alpha * z * e * poly)

        phi[mu, :] = val
        grad[mu, :, 0] = gx
        grad[mu, :, 1] = gy
        grad[mu, :, 2] = gz

    return phi, grad

def _accumulate_rho_grad_from_phi(
    D: np.ndarray,
    phi: np.ndarray,            # (nbf,Nc)
    grad_phi: np.ndarray,       # (nbf,Nc,3)
    diag_idx: np.ndarray,       # (K,)
    pairs: np.ndarray           # (L,3) -> (m,n, d_mn)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    φ/∇φ から ρ, ∇ρ を構築（チャンク内）。
    """
    Nc = phi.shape[1]
    rho = np.zeros(Nc, dtype=np.float64)
    grad_rho = np.zeros((Nc, 3), dtype=np.float64)

    # 対角
    for m in diag_idx:
        m = int(m)
        d_mm = float(D[m, m])
        if d_mm == 0.0:
            continue
        p = phi[m]
        rho += d_mm * p * p
        grad_rho += d_mm * (p[:, None] * grad_phi[m])

    # 非対角（m<n）
    for rec in pairs:
        m = int(rec[0]); n = int(rec[1]); d_mn = float(rec[2])
        c2 = 2.0 * d_mn
        p_m = phi[m]; p_n = phi[n]
        rho += c2 * p_m * p_n
        grad_rho += c2 * (p_n[:, None] * grad_phi[m] + p_m[:, None] * grad_phi[n])

    return rho, grad_rho

def compute_rho_grad_phi_chunked(
    basis_functions,
    D: np.ndarray,
    points: np.ndarray,
    *,
    chunk_size: int = 20000,
    d_thresh: float = 0.0,
    return_phi: bool = False,
    return_grad_phi: bool = False,
) -> Iterable[Dict[str, Any]]:
    """
    ρ/∇ρ をチャンクに分けて逐次生成するジェネレータ。

    Parameters
    ----------
    basis_functions : list[GTO]
    D : (nbf,nbf)  スピン和密度
    points : (N,3)
    chunk_size : int   1チャンク当たりのグリッド点数
    d_thresh : float   |D_{mn}| < d_thresh を無視（スクリーニング）
    return_phi : bool  チャンクの φ を同時に返す（V組立で必要ならTrue）
    return_grad_phi : bool 同上（∇φが必要ならTrue）

    Yields
    ------
    dict with keys:
      "slice": slice(start, end),
      "rho": (Nc,), "grad_rho": (Nc,3),
      ["phi": (nbf,Nc)], ["grad_phi": (nbf,Nc,3)]
    """
    pts = np.asarray(points, dtype=np.float64)
    N = pts.shape[0]
    diag_idx, pairs = _select_density_pairs(D, d_thresh)

    if chunk_size <= 0:
        chunk_size = N

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        pts_chunk = pts[start:end]
        phi_chunk, grad_chunk = _compute_phi_grad_for_chunk(basis_functions, pts_chunk)
        rho_chunk, grad_rho_chunk = _accumulate_rho_grad_from_phi(D, phi_chunk, grad_chunk, diag_idx, pairs)

        out: Dict[str, Any] = {
            "slice": slice(start, end),
            "rho": rho_chunk,
            "grad_rho": grad_rho_chunk,
        }
        if return_phi:
            out["phi"] = phi_chunk
        if return_grad_phi:
            out["grad_phi"] = grad_chunk
        yield out

def assemble_local_potential_from_chunk(
    V_accum: np.ndarray,          # (nbf,nbf) 蓄積先
    w_chunk: np.ndarray,          # (Nc,)     重み
    v_local_chunk: np.ndarray,    # (Nc,)     ローカルポテンシャル
    phi_chunk: np.ndarray         # (nbf,Nc)
) -> None:
    """
    チャンク一回分の V_{μν} ≈ Σ_g w_g v_g φ_μ(g) φ_ν(g) を蓄積する。
    """
    proj = (w_chunk * v_local_chunk)[None, :] * phi_chunk    # (nbf,Nc)
    V_accum += phi_chunk @ proj.T

def assemble_local_potential_chunked(
    grid,
    basis_functions,
    D: np.ndarray,
    v_local_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    chunk_size: int = 20000,
    d_thresh: float = 0.0,
    return_energy_density: bool = False,
    energy_density_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Tuple[np.ndarray, float] | np.ndarray:
    """
    v(r) が ρ/∇ρ の局所関数 v_local_func(rho, grad_rho) で与えられるとき、
    V_{μν} をチャンク積分で構築。必要なら ρ·ε(r) のエネルギーも同時に積分。

    Parameters
    ----------
    grid : IntegrationGrid
    basis_functions : list[GTO]
    D : (nbf,nbf)
    v_local_func : (rho_chunk, grad_chunk) -> v_local_chunk
    chunk_size, d_thresh : see compute_rho_grad_phi_chunked
    return_energy_density : bool
        True のとき energy_density_func を使って Ep = ∫ ρ ε_p(r) dr も返す。
    energy_density_func : (rho_chunk, grad_chunk) -> eps_chunk
        例: LDA交換の ε_x(ρ) や LYP相関の ε_c(ρ,∇ρ)

    Returns
    -------
    V : (nbf,nbf)  または  (V, Energy)
    """
    nbf = len(basis_functions)
    V = np.zeros((nbf, nbf), dtype=np.float64)
    E_total = 0.0
    for blk in compute_rho_grad_phi_chunked(
        basis_functions, D, grid.points,
        chunk_size=chunk_size, d_thresh=d_thresh, return_phi=True
    ):
        sl = blk["slice"]
        w = grid.weights[sl]
        rho = blk["rho"]; grad = blk["grad_rho"]; phi = blk["phi"]

        v_local = v_local_func(rho, grad)
        assemble_local_potential_from_chunk(V, w, v_local, phi)

        if return_energy_density:
            if energy_density_func is None:
                raise ValueError("energy_density_func is required when return_energy_density=True")
            eps = energy_density_func(rho, grad)   # per-electron などの定義は呼び出し側に合わせる
            E_total += float(np.dot(w, rho * eps))

    V = 0.5 * (V + V.T)
    return (V, E_total) if return_energy_density else V

        
def integrate_density_over_grid(grid, basis_functions, D, *, chunk_size=20000, d_thresh=0.0):
    Ne_grid = 0.0
    sum_w   = 0.0
    for blk in compute_rho_grad_phi_chunked(
        basis_functions, np.asarray(D, float), grid.points,
        chunk_size=chunk_size, d_thresh=d_thresh, return_phi=False
    ):
        sl = blk["slice"]; w = grid.weights[sl]
        rho = blk["rho"]
        Ne_grid += float(np.dot(w, rho))
        sum_w   += float(np.sum(w))
    return Ne_grid, sum_w

