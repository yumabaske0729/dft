# -*- coding: utf-8 -*-
"""
Two-electron (electron repulsion) integrals over Cartesian GTOs (AO basis).
Obara–Saika (OS) implementation with 3D Hermite Coulomb R-tensor R^{(n)}_{t,u,v}(T).

改稿ポイント（非退化保障 & 安全化）
--------------------------------
- R^{(n)}_{t,u,v}(T) の VRR を s = t+u+v の層ごとに「旧層 → 新層」の二段バッファで更新。
  → 同じ層で write した値を read しない（累積/退化を完全排除）。
- n 次（Boys の n）を明示保持し、n+1 スライス参照は R_old[1:, ...] で安全にシフト。
- ERI_R_DIAG=1 時の診断ログは、回数制限（既定 3 回）＋重複抑止＋T≈0 スキップで安全に出力。
- ERI 4 指標テンソルは 8 重対称で充填。微小値クリーニング/スケール/バナー出力は従来通り。

参考:
- Obara & Saika, J. Chem. Phys. 84, 3963 (1986)
- Obara & Saika, J. Chem. Phys. 89, 1540 (1988)
- Fermann & Valeev, arXiv:2007.12057
"""
from __future__ import annotations
import os
import numpy as np
from math import pi, sqrt
from typing import Tuple

from ..gaussian_math.gaussian_math import norm_prefactor
from .one_electron.boys import boys_function

# ---- R-DIAG のグローバル状態（回数制限・重複抑止用） ----
_R_DIAG_COUNT = 0
_R_DIAG_SEEN = set()


# ========= 1D Hermite 係数（OS）：E[i,j,t] =========
def _E1d_os(i_max: int, j_max: int, alpha: float, beta: float, Ax: float, Bx: float) -> np.ndarray:
    """
    OS-style 1D Hermite coefficients E[i,j,t] for a pair (A,B).
      p = a+b,  Px = (a*Ax + b*Bx)/p
      E(i+1,j,t) = (Px-Ax) E(i,j,t) + (t+1)/(2p) E(i,j,t+1) + i/(2p) E(i-1,j,t)
      E(i,j+1,t) = (Px-Bx) E(i,j,t) + (t+1)/(2p) E(i,j,t+1) + j/(2p) E(i,j-1,t)
    returns: E shape = (i_max+1, j_max+1, tmax+2)
    """
    p = alpha + beta
    Px = (alpha * Ax + beta * Bx) / p
    tmax = i_max + j_max

    E = np.zeros((i_max + 1, j_max + 1, tmax + 2), dtype=float)
    E[0, 0, 0] = 1.0  # base

    if i_max >= 1:
        for i in range(1, i_max + 1):
            for t in range(0, tmax + 1):
                term1 = (Px - Ax) * E[i - 1, 0, t]
                term2 = ((t + 1) / (2.0 * p)) * E[i - 1, 0, t + 1]
                term3 = ((i - 1) / (2.0 * p)) * (E[i - 2, 0, t] if i - 2 >= 0 else 0.0)
                E[i, 0, t] = term1 + term2 + term3

    if j_max >= 1:
        for j in range(1, j_max + 1):
            for i in range(0, i_max + 1):
                for t in range(0, tmax + 1):
                    term1 = (Px - Bx) * E[i, j - 1, t]
                    term2 = ((t + 1) / (2.0 * p)) * E[i, j - 1, t + 1]
                    term3 = ((j - 1) / (2.0 * p)) * (E[i, j - 2, t] if j - 2 >= 0 else 0.0)
                    E[i, j, t] = term1 + term2 + term3

    return E  # (i_max+1, j_max+1, tmax+2)


# ========= 3D R^{(n)}_{t,u,v}(T)（二段バッファVRR） =========
def _build_R_tensor(
    T: float, Wx: float, Wy: float, Wz: float, eta: float,
    tx_max: int, ty_max: int, tz_max: int
) -> np.ndarray:
    """
    Build R^{(n)}_{t,u,v}(T) for n=0..(tx_max+ty_max+tz_max), 0<=t<=tx_max, etc.
      Base: R^{(n)}_{0,0,0}(T) = F_n(T)
      VRR (OS):
        R^{(n)}_{t+1,u,v} = Wx R^{(n)}_{t,u,v} + (t)/(2η) R^{(n+1)}_{t-1,u,v}
        R^{(n)}_{t,u+1,v} = Wy R^{(n)}_{t,u,v} + (u)/(2η) R^{(n+1)}_{t,u-1,v}
        R^{(n)}_{t,u,v+1} = Wz R^{(n)}_{t,u,v} + (v)/(2η) R^{(n+1)}_{t,u,v-1}
    returns:
      R0 : n=0 スライスのみ（shape=(tx_max+1, ty_max+1, tz_max+1)）
    """
    Nmax = tx_max + ty_max + tz_max
    # Boys F_n(T), n=0..Nmax
    F = np.empty(Nmax + 1, dtype=float)
    for n in range(Nmax + 1):
        F[n] = boys_function(n, T)

    # R_old / R_new の二段バッファ（n,t,u,v）
    R_old = np.zeros((Nmax + 1, tx_max + 1, ty_max + 1, tz_max + 1), dtype=float)
    R_new = np.zeros_like(R_old)
    # base
    R_old[:, 0, 0, 0] = F

    Smax = tx_max + ty_max + tz_max
    inv_2eta = 0.0 if eta == 0.0 else 1.0 / (2.0 * eta)

    for s in range(0, Smax):
        # 次層へ進む準備：まず R_new を R_old コピーで初期化
        # （層 s の値は保持しつつ、層 s+1 で上書き代入）
        R_new[...] = R_old

        for t in range(0, min(tx_max, s) + 1):
            for u in range(0, min(ty_max, s - t) + 1):
                v = s - (t + u)
                if v < 0 or v > tz_max:
                    continue

                Rtuv = R_old[:, t, u, v]  # 読み取りは「旧層」から

                # (t+1, u, v)
                if t + 1 <= tx_max:
                    if t >= 1:
                        dest = Wx * Rtuv
                        dest[:Nmax] += (t * inv_2eta) * R_old[1:, t - 1, u, v]
                        R_new[:, t + 1, u, v] = dest
                    else:
                        R_new[:, t + 1, u, v] = Wx * Rtuv

                # (t, u+1, v)
                if u + 1 <= ty_max:
                    if u >= 1:
                        dest = Wy * Rtuv
                        dest[:Nmax] += (u * inv_2eta) * R_old[1:, t, u - 1, v]
                        R_new[:, t, u + 1, v] = dest
                    else:
                        R_new[:, t, u + 1, v] = Wy * Rtuv

                # (t, u, v+1)
                if v + 1 <= tz_max:
                    if v >= 1:
                        dest = Wz * Rtuv
                        dest[:Nmax] += (v * inv_2eta) * R_old[1:, t, u, v - 1]
                        R_new[:, t, u, v + 1] = dest
                    else:
                        R_new[:, t, u, v + 1] = Wz * Rtuv

        # s 層終了：新しい層を確定
        R_old, R_new = R_new, R_old

    # --- DEBUG: 安全化した R 診断（要求時のみ） ---
    if os.getenv("ERI_R_DIAG", "0") != "0":
        # 回数制限（既定3回）＋重複抑止＋T≈0スキップ
        try:
            diag_max = int(os.getenv("ERI_R_DIAG_MAX", "3"))
        except Exception:
            diag_max = 3
        global _R_DIAG_COUNT, _R_DIAG_SEEN
        key = (tx_max, ty_max, tz_max)
        if (_R_DIAG_COUNT < diag_max) and (key not in _R_DIAG_SEEN) and (T > 1.0e-12):
            def boys(n, TT): return boys_function(n, TT)
            print(f"[R-DIAG] T={T:.6f}, Wx={Wx:.6f}, Wy={Wy:.6f}, Wz={Wz:.6f}, eta={eta:.6f}")
            if tx_max >= 1:
                print(f"[R-DIAG] R(1,0,0)={R_old[0,1,0,0]:.12e}, Wx*F0={Wx*boys(0,T):.12e}, F1={boys(1,T):.12e}")
            if ty_max >= 1:
                print(f"[R-DIAG] R(0,1,0)={R_old[0,0,1,0]:.12e}, Wy*F0={Wy*boys(0,T):.12e}, F1={boys(1,T):.12e}")
            if tz_max >= 1:
                print(f"[R-DIAG] R(0,0,1)={R_old[0,0,0,1]:.12e}, Wz*F0={Wz*boys(0,T):.12e}, F1={boys(1,T):.12e}")
            _R_DIAG_COUNT += 1
            _R_DIAG_SEEN.add(key)

    # n=0 スライス（R^{(0)}_{t,u,v}）だけ返す
    return R_old[0, :, :, :]


# ========= primitive / contracted ERI =========
def primitive_eri(
    l1, m1, n1, a1, A,
    l2, m2, n2, a2, B,
    l3, m3, n3, a3, C,
    l4, m4, n4, a4, D,
) -> float:
    """
    Primitive ERI for four Cartesian GTOs.
    Prefactor includes: exp(-a1 a2/p |A-B|^2 - a3 a4/q |C-D|^2).
    OS 正式：E_x/y/z(AB, CD) と R^{(0)}_{t,u,v}(T) を結合。
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    C = np.asarray(C, float); D = np.asarray(D, float)

    p = a1 + a2
    q = a3 + a4
    P = (a1 * A + a2 * B) / p
    Q = (a3 * C + a4 * D) / q

    diffAB2 = float(np.dot(A - B, A - B))
    diffCD2 = float(np.dot(C - D, C - D))

    W = P - Q
    Wx, Wy, Wz = float(W[0]), float(W[1]), float(W[2])

    pre = 2.0 * (pi ** 2.5) / (p * q * sqrt(p + q))
    pre *= np.exp(-(a1 * a2 / p) * diffAB2 - (a3 * a4 / q) * diffCD2)

    # 1D Hermite（必要スライス）
    ExAB = _E1d_os(l1, l2, a1, a2, A[0], B[0])[l1, l2, :l1 + l2 + 1]
    EyAB = _E1d_os(m1, m2, a1, a2, A[1], B[1])[m1, m2, :m1 + m2 + 1]
    EzAB = _E1d_os(n1, n2, a1, a2, A[2], B[2])[n1, n2, :n1 + n2 + 1]

    ExCD = _E1d_os(l3, l4, a3, a4, C[0], D[0])[l3, l4, :l3 + l4 + 1]
    EyCD = _E1d_os(m3, m4, a3, a4, C[1], D[1])[m3, m4, :m3 + m4 + 1]
    EzCD = _E1d_os(n3, n4, a3, a4, C[2], D[2])[n3, n4, :n3 + n4 + 1]

    # R^{(0)}_{t,u,v}(T)
    eta = (p * q) / (p + q)
    T = eta * (Wx * Wx + Wy * Wy + Wz * Wz)

    tx_max = (l1 + l2) + (l3 + l4)
    ty_max = (m1 + m2) + (m3 + m4)
    tz_max = (n1 + n2) + (n3 + n4)

    R0 = _build_R_tensor(T, Wx, Wy, Wz, eta, tx_max, ty_max, tz_max)  # (tx_max+1, ty_max+1, tz_max+1)

    # 3D 畳み込み：t=i+j, u=k+l, v=m+n
    val = 0.0
    for i in range(l1 + l2 + 1):
        Ei = ExAB[i]
        if Ei == 0.0: continue
        for j in range(l3 + l4 + 1):
            Ej = ExCD[j]
            if Ej == 0.0: continue
            tx = i + j
            Exx = Ei * Ej

            for k in range(m1 + m2 + 1):
                Ek = EyAB[k]
                if Ek == 0.0: continue
                for l in range(m3 + m4 + 1):
                    El = EyCD[l]
                    if El == 0.0: continue
                    ty = k + l
                    Exy = Exx * Ek * El

                    for m in range(n1 + n2 + 1):
                        Em = EzAB[m]
                        if Em == 0.0: continue
                        for n in range(n3 + n4 + 1):
                            En = EzCD[n]
                            if En == 0.0: continue
                            tz = m + n
                            val += Exy * Em * En * R0[tx, ty, tz]

    return pre * val


def contracted_eri(g1, g2, g3, g4) -> float:
    """
    Contracted ERI:
      sum_{a1..a4} (c1 c2 c3 c4) (N1 N2 N3 N4) * primitive_eri(...)
    """
    eri_val = 0.0
    for a1, c1 in zip(g1.exponents, g1.coefficients):
        N1 = norm_prefactor(g1.l, g1.m, g1.n, a1)
        for a2, c2 in zip(g2.exponents, g2.coefficients):
            N2 = norm_prefactor(g2.l, g2.m, g2.n, a2)
            for a3, c3 in zip(g3.exponents, g3.coefficients):
                N3 = norm_prefactor(g3.l, g3.m, g3.n, a3)
                for a4, c4 in zip(g4.exponents, g4.coefficients):
                    N4 = norm_prefactor(g4.l, g4.m, g4.n, a4)
                    eri_val += (
                        (c1 * c2 * c3 * c4)
                        * (N1 * N2 * N3 * N4)
                        * primitive_eri(
                            g1.l, g1.m, g1.n, a1, g1.center,
                            g2.l, g2.m, g2.n, a2, g2.center,
                            g3.l, g3.m, g3.n, a3, g3.center,
                            g4.l, g4.m, g4.n, a4, g4.center,
                        )
                    )
    return float(eri_val)


def build_eri_tensor(basis_functions):
    """
    Build the full 4-index AO ERI tensor with permutational symmetry.
    eri[m,n,r,s] = (mn|rs)

    env:
      ERI_SCALE        : float (default 1.0)  global scale (debug)
      ERI_CLEAN_CUTOFF : float (default 0.0)  |eri| < cutoff -> 0
      ERI_DIAG         : "0/1"                simple diagnostics
      ERI_BANNER       : "0/1"                banner on (default 1)
    """
    if os.getenv("ERI_BANNER", "1") != "0":
        print("[ERI] OS-R build_eri_tensor active (mnrs) ...")

    nbf = len(basis_functions)
    eri = np.zeros((nbf, nbf, nbf, nbf), dtype=float)

    for i in range(nbf):
        for j in range(i + 1):
            for k in range(nbf):
                for l in range(k + 1):
                    if (i, j) < (k, l):
                        continue

                    val = contracted_eri(
                        basis_functions[i], basis_functions[j],
                        basis_functions[k], basis_functions[l]
                    )

                    eri[i, j, k, l] = val
                    eri[j, i, k, l] = val
                    eri[i, j, l, k] = val
                    eri[j, i, l, k] = val
                    eri[k, l, i, j] = val
                    eri[l, k, i, j] = val
                    eri[k, l, j, i] = val
                    eri[l, k, j, i] = val

    scale = float(os.getenv("ERI_SCALE", "1.0"))
    if scale != 1.0:
        eri *= scale
        print(f"[ERI] scaled by ERI_SCALE={scale:.3f}")

    cutoff = float(os.getenv("ERI_CLEAN_CUTOFF", "0.0"))
    if cutoff > 0.0:
        mask = np.abs(eri) < cutoff
        if mask.any():
            eri[mask] = 0.0

    if os.getenv("ERI_DIAG", "0") != "0":
        fn = float(np.linalg.norm(eri))
        asym_mn = float(np.linalg.norm(eri - eri.transpose(1, 0, 2, 3)))
        asym_rs = float(np.linalg.norm(eri - eri.transpose(0, 1, 3, 2)))
        asym_pair = float(np.linalg.norm(eri - eri.transpose(2, 3, 0, 1)))
        print(
            "[eri] diag:",
            f"\neri_F={fn:.6e}, asym_mn={asym_mn:.3e}, asym_rs={asym_rs:.3e}, asym_pair={asym_pair:.3e}"
        )

    return eri


if __name__ == "__main__":
    alpha = 0.5
    A = np.array([0.0, 0.0, 0.0])
    from types import SimpleNamespace
    s = SimpleNamespace(l=0, m=0, n=0, exponents=[alpha], coefficients=[1.0], center=A)
    val = contracted_eri(s, s, s, s)
    print("(ss|ss) same-center =", val)