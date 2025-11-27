# DFT/integrals/one_electron/kinetic_os.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from ...gaussian_math.gaussian_math import norm_prefactor


def _os_1d_overlap_table(i_max: int, j_max: int,
                         Ax: float, Bx: float,
                         alpha: float, beta: float):
    """
    1D OS 再帰で未正規化のプリミティブ重なり S[i,j] を構築。
    S00 = sqrt(pi/p) * exp( -mu * (Ax-Bx)^2 ), p=alpha+beta, mu=alpha*beta/p
    再帰:
      S_{i+1,j} = PA*S_{i,j} + (i/2p) S_{i-1,j} + (j/2p) S_{i,j-1}
      S_{i,j+1} = PB*S_{i,j} + (i/2p) S_{i-1,j} + (j/2p) S_{i,j-1}
    """
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * Ax + beta * Bx) / p
    PA = P - Ax
    PB = P - Bx

    S = np.zeros((i_max + 1, j_max + 1), dtype=float)
    S[0, 0] = np.sqrt(np.pi / p) * np.exp(-mu * (Ax - Bx) * (Ax - Bx))

    # i 方向 (j=0)
    for i in range(1, i_max + 1):
        if i == 1:
            S[i, 0] = PA * S[i - 1, 0]
        else:
            S[i, 0] = PA * S[i - 1, 0] + (i - 1) / (2.0 * p) * S[i - 2, 0]

    # j 方向 (i=0)
    for j in range(1, j_max + 1):
        if j == 1:
            S[0, j] = PB * S[0, j - 1]
        else:
            S[0, j] = PB * S[0, j - 1] + (j - 1) / (2.0 * p) * S[0, j - 2]

    # 内部
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            v = PA * S[i - 1, j]
            if i - 2 >= 0:
                v += (i - 1) / (2.0 * p) * S[i - 2, j]
            v += (j) / (2.0 * p) * S[i - 1, j - 1]
            S[i, j] = v
    return S, p


def _S_lookup(S: np.ndarray, i: int, j: int) -> float:
    """範囲外インデックスは 0 とみなす安全ルックアップ"""
    if i < 0 or j < 0:
        return 0.0
    if i >= S.shape[0] or j >= S.shape[1]:
        return 0.0
    return float(S[i, j])


def _primitive_kinetic_os(l1, m1, n1, alpha, A,
                          l2, m2, n2, beta, B) -> float:
    """
    一般角運動量 (s/p/d/...) に対応したプリミティブ運動エネルギー。
    身近な安定式:
      T = 1/2 * sum_{u=x,y,z} ∫ (∂_u φ_a)(∂_u φ_b) d r
    を 1D OS 重なりテーブル S_u の線形結合で構成する。
    ※ S テーブルは i±1, j±1 を参照するため (l+2, j+2) まで作る。
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ax, Ay, Az = float(A[0]), float(A[1]), float(A[2])
    Bx, By, Bz = float(B[0]), float(B[1]), float(B[2])

    # 1D OS overlap tables (+2 マージン)
    Sx, _ = _os_1d_overlap_table(l1 + 2, l2 + 2, Ax, Bx, alpha, beta)
    Sy, _ = _os_1d_overlap_table(m1 + 2, m2 + 2, Ay, By, alpha, beta)
    Sz, _ = _os_1d_overlap_table(n1 + 2, n2 + 2, Az, Bz, alpha, beta)

    # shorthand
    i, j = l1, l2
    k, l = m1, m2
    u, v = n1, n2

    # 各軸の勾配×勾配
    Ix = (i * j * _S_lookup(Sx, i - 1, j - 1)
          - 2.0 * beta * i * _S_lookup(Sx, i - 1, j + 1)
          - 2.0 * alpha * j * _S_lookup(Sx, i + 1, j - 1)
          + 4.0 * alpha * beta * _S_lookup(Sx, i + 1, j + 1))

    Iy = (k * l * _S_lookup(Sy, k - 1, l - 1)
          - 2.0 * beta * k * _S_lookup(Sy, k - 1, l + 1)
          - 2.0 * alpha * l * _S_lookup(Sy, k + 1, l - 1)
          + 4.0 * alpha * beta * _S_lookup(Sy, k + 1, l + 1))

    Iz = (u * v * _S_lookup(Sz, u - 1, v - 1)
          - 2.0 * beta * u * _S_lookup(Sz, u - 1, v + 1)
          - 2.0 * alpha * v * _S_lookup(Sz, u + 1, v - 1)
          + 4.0 * alpha * beta * _S_lookup(Sz, u + 1, v + 1))

    # 残り 2 軸は通常の 1D 重なり
    S_y, S_z = _S_lookup(Sy, k, l), _S_lookup(Sz, u, v)
    S_x = _S_lookup(Sx, i, j)

    T = 0.5 * (Ix * S_y * S_z + Iy * S_x * S_z + Iz * S_x * S_y)
    return float(T)


def contracted_kinetic(gto1, gto2) -> float:
    """
    収縮 GTO の運動エネルギー（一般 l 対応）。
    T = Σ_{a1,a2} (c1 c2) (N1 N2) * T_primitive
    """
    Tval = 0.0
    for a1, c1 in zip(gto1.exponents, gto1.coefficients):
        N1 = norm_prefactor(gto1.l, gto1.m, gto1.n, a1)
        for a2, c2 in zip(gto2.exponents, gto2.coefficients):
            N2 = norm_prefactor(gto2.l, gto2.m, gto2.n, a2)
            T_prim = _primitive_kinetic_os(
                gto1.l, gto1.m, gto1.n, a1, gto1.center,
                gto2.l, gto2.m, gto2.n, a2, gto2.center
            )
            Tval += (c1 * c2) * (N1 * N2) * T_prim
    return float(Tval)


# --- Optional: quick self-test ---
if __name__ == "__main__":
    import numpy as _np
    from types import SimpleNamespace
    A = _np.array([0.0, 0.0, 0.0])
    B = _np.array([0.1, -0.2, 0.3])
    # s|s
    sA = SimpleNamespace(l=0, m=0, n=0, exponents=[0.5], coefficients=[1.0], center=A)
    sB = SimpleNamespace(l=0, m=0, n=0, exponents=[0.5], coefficients=[1.0], center=B)
    print("T(s|s) =", contracted_kinetic(sA, sB))
    # d_xx|d_xx
    dA = SimpleNamespace(l=2, m=0, n=0, exponents=[0.4], coefficients=[1.0], center=A)
    dB = SimpleNamespace(l=2, m=0, n=0, exponents=[0.5], coefficients=[1.0], center=B)
    print("T(dxx|dxx) =", contracted_kinetic(dA, dB))
