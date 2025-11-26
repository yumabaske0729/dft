# DFT/scf/grid.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math
import numpy as np
from numpy.polynomial.legendre import leggauss
from . import lebedev_data

# ---- Helpers ----
def _default_lebedev_degree() -> int:
    """
    既定の角点数（degree）。環境変数 LEBEDEV_ORDER を優先（= degree で解釈）。
    未設定なら 110 を返す（実務上の最小安定ラインを想定）。
    """
    try:
        val = os.getenv("LEBEDEV_ORDER", None)
        if val is not None and val.strip() != "":
            return int(val)
    except Exception:
        pass
    return 110  # safe default

def _radial_linear_grid(n_radial: int, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre on [0,Rmax] with r^2 Jacobian included in weights."""
    x, w = leggauss(n_radial)  # [-1,1]
    t = 0.5 * (x + 1.0)        # [0,1]
    dt = 0.5 * w
    r = rmax * t
    dr = rmax * dt
    w_r = (r * r) * dr
    return r, w_r

def _smooth_poly(s: np.ndarray) -> np.ndarray:
    # Becke の連続化: 0.5*(3s - s^3)
    return 0.5 * (3.0 * s - s * s * s)

def _becke_switch(s: np.ndarray, m: int = 3) -> np.ndarray:
    out = np.clip(s, -1.0, 1.0)
    for _ in range(m):
        out = _smooth_poly(out)
    return out

def _becke_weights_all(points: np.ndarray, centers: np.ndarray, m: int = 3) -> np.ndarray:
    """
    全グリッド点について、全原子 A の Becke “生”重み w_A^raw(r) を計算し、
    点ごとに正規化して Σ_A w_A(r) = 1 を満たす行列を返す。
    Returns
    ------
    w_norm : (Npts, Natoms) ndarray
      各点で Σ_A w_norm[g, A] = 1
    """
    pts = np.asarray(points, float)  # (N,3)
    ctr = np.asarray(centers, float) # (A,3)
    N, A = pts.shape[0], ctr.shape[0]
    r_gA = np.linalg.norm(pts[:, None, :] - ctr[None, :, :], axis=2)  # (N, A)
    w_raw = np.ones((N, A), dtype=float)
    for a in range(A):
        Ra = ctr[a]
        for b in range(A):
            if a == b:
                continue
            Rb = ctr[b]
            Rab = float(np.linalg.norm(Ra - Rb))
            if Rab == 0.0:
                continue
            s = (r_gA[:, a] - r_gA[:, b]) / Rab
            s = _becke_switch(s, m=m)
            w_raw[:, a] *= 0.5 * (1.0 - s)
    sumW = np.sum(w_raw, axis=1, keepdims=True)  # (N,1)
    sumW[sumW == 0.0] = 1.0
    w_norm = w_raw / sumW
    return w_norm

# ---- Public API ----
class IntegrationGrid:
    def __init__(self, points: np.ndarray, weights: np.ndarray, *, lebedev_degree: int, n_radial: int, rmax: float):
        self.points = np.asarray(points, dtype=float)   # (N,3)
        self.weights = np.asarray(weights, dtype=float) # (N,)
        # 主：lebedev_degree（角点数）。互換：lebedev_order（旧名; 将来非推奨）
        deg = int(lebedev_degree)
        ord_ = int(lebedev_data.DEG2ORD.get(deg, 0))  # 対応する“order”が分かる場合のみ併記
        self.meta = {
            "lebedev_degree": deg,
            "lebedev_order": ord_,   # 互換キー（旧コードを壊さない）
            "n_radial": int(n_radial),
            "Rmax": float(rmax),
        }

def get_lebedev_grid(degree: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Lebedev 角格子（degree=角点数）を返す。degree が None の場合:
    - 環境変数 LEBEDEV_ORDER を degree として採用
    - 未設定なら degree=110
    戻り値: (points[N,3], weights[N], actual_degree)
    """
    # 1) resolve request
    req_deg = _default_lebedev_degree() if degree is None else int(degree)
    # 2) load via lebedev_data (NPY / embedded / nearest fallback)
    try:
        data = lebedev_data.load_grid_by_degree(req_deg, allow_nearest=True)
        pts = data["points"]; wts = data["weights"]
        actual_degree = int(pts.shape[0])
        print(f"[grid] Lebedev degree request={req_deg}, actual={actual_degree}, N_ang={pts.shape[0]}")
        return pts, wts, actual_degree
    except Exception as e:
        # 最低限の埋め込み (degree=6) にフォールバック
        emb = lebedev_data._embedded_deg6()
        pts, wts = emb["points"], emb["weights"]
        print(f"[grid] Lebedev degree request={req_deg} not available ({e}); falling back to degree=6, N_ang=6")
        return pts, wts, 6

def generate_integration_grid(basis_functions, level: int = 3) -> IntegrationGrid:
    """
    多中心 Becke 分割グリッド（原子ごとに点を生成し、その点の重みに「正規化済み Becke 重み」を適用）。
    - Angular : Lebedev（degree=角点数）
    - Radial  : Gauss–Legendre on [0, Rmax], Nr = max(16, level*12)
    - Rmax    : env DFT_RMAX (default 8.0)
    """
    # centers
    centers = np.unique(np.vstack([g.center for g in basis_functions]), axis=0)
    n_atoms = centers.shape[0]

    # angular / radial parameters
    ang_pts, ang_wts, actual_degree = get_lebedev_grid(None)
    n_ang = int(ang_pts.shape[0])
    n_radial = max(16, int(level) * 12)
    RMAX = float(os.getenv("DFT_RMAX", "8.0"))

    # atom shells -> concatenate
    all_pts, all_wts, atom_idx = [], [], []
    for a_idx in range(n_atoms):
        ctr = centers[a_idx]
        r_pts, r_wts = _radial_linear_grid(n_radial, RMAX)
        for r, wr in zip(r_pts, r_wts):
            shell_pts = ctr + r * ang_pts  # (n_ang,3)
            shell_wts = wr * ang_wts       # (n_ang,)
            all_pts.append(shell_pts)
            all_wts.append(shell_wts)
            atom_idx.append(np.full(n_ang, a_idx, dtype=int))

    points = np.vstack(all_pts)   # (Npts,3)
    weights = np.hstack(all_wts)  # (Npts,)
    atom_idx = np.concatenate(atom_idx)

    # Becke weights (Σ_A=1) を由来原子に適用
    W_all = _becke_weights_all(points, centers, m=3)   # (Npts, n_atoms)
    wA = W_all[np.arange(points.shape[0]), atom_idx]   # (Npts,)
    weights = weights * wA

    # QA 出力
    sphere = (4.0 / 3.0) * math.pi * (RMAX ** 3)
    ratio = float(np.sum(weights)) / sphere
    print(f"[grid] Lebedev degree={actual_degree}, Nr={n_radial}, Rmax={RMAX}, Npts={points.shape[0]}")
    print(f"[grid] QA: weights_sum={np.sum(weights):.6f}, ratio={ratio:.3f}")

    return IntegrationGrid(points=points, weights=weights,
                           lebedev_degree=actual_degree, n_radial=n_radial, rmax=RMAX)