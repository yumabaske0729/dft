# DFT/integrals/one_electron/kinetic.py
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt, pi
from ...gaussian_math.gaussian_math import norm_prefactor

def _os_1d_overlap_table(i_max: int, j_max: int, Ax: float, Bx: float, alpha: float, beta: float):
    """
    1D OS 再帰による未正規化プリミティブ重なり S[i,j] を構築。
    S00 = sqrt(pi/p) * exp(-mu * (Ax-Bx)^2), p=alpha+beta, mu=alpha*beta/p
    再帰:
      S_{i+1,j} = PA * S_{i,j} + (i/(2p)) S_{i-1,j} + (j/(2p)) S_{i,j-1}
    """
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * Ax + beta * Bx) / p
    PA = P - Ax
    PB = P - Bx
    S = np.zeros((i_max + 1, j_max + 1), dtype=float)
    S[0, 0] = sqrt(pi / p) * np.exp(-mu * (Ax - Bx) ** 2)

    # i 方向
    for i in range(1, i_max + 1):
        if i == 1:
            S[i, 0] = PA * S[i - 1, 0]
        else:
            S[i, 0] = PA * S[i - 1, 0] + (i - 1) / (2 * p) * S[i - 2, 0]

    # j 方向
    for j in range(1, j_max + 1):
        if j == 1:
            S[0, j] = PB * S[0, j - 1]
        else:
            S[0, j] = PB * S[0, j - 1] + (j - 1) / (2 * p) * S[0, j - 2]

    # 内部
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            v = PA * S[i - 1, j]
            if i - 2 >= 0:
                v += (i - 1) / (2 * p) * S[i - 2, j]
            v += (j) / (2 * p) * S[i - 1, j - 1]
            S[i, j] = v
    return S, p

def _S_lookup(S: np.ndarray, i: int, j: int) -> float:
    """範囲外アクセスは 0 とする安全ルックアップ"""
    if i < 0 or j < 0:
        return 0.0
    if i >= S.shape[0] or j >= S.shape[1]:
        return 0.0
    return float(S[i, j])

def _primitive_kinetic_general(l1, m1, n1, alpha, A, l2, m2, n2, beta, B) -> float:
    """
    一般 (l,m,n) に対するプリミティブ運動エネルギー。
    部分積分より T = 0.5 * sum_u ∫(∂u φ_a)(∂u φ_b) d r を
    1D OS テーブルの線形結合で構成（d 以上も対応）。
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ax, Ay, Az = float(A[0]), float(A[1]), float(A[2])
    Bx, By, Bz = float(B[0]), float(B[1]), float(B[2])

    # 参照最大次数（微分で +1 / 交差項で +2 を参照するため余裕を持たせる）
    Sx, _ = _os_1d_overlap_table(l1 + 2, l2 + 2, Ax, Bx, alpha, beta)
    Sy, _ = _os_1d_overlap_table(m1 + 2, m2 + 2, Ay, By, alpha, beta)
    Sz, _ = _os_1d_overlap_table(n1 + 2, n2 + 2, Az, Bz, alpha, beta)

    i, j = l1, l2
    k, l = m1, m2
    u, v = n1, n2

    # x 成分（勾配×勾配の 1D 線形結合；一般式）
    Ix = (
        i * j * _S_lookup(Sx, i - 1, j - 1)
        - 2.0 * beta * i * _S_lookup(Sx, i - 1, j + 1)
        - 2.0 * alpha * j * _S_lookup(Sx, i + 1, j - 1)
        + 4.0 * alpha * beta * _S_lookup(Sx, i + 1, j + 1)
    )
    Iy = (
        k * l * _S_lookup(Sy, k - 1, l - 1)
        - 2.0 * beta * k * _S_lookup(Sy, k - 1, l + 1)
        - 2.0 * alpha * l * _S_lookup(Sy, k + 1, l - 1)
        + 4.0 * alpha * beta * _S_lookup(Sy, k + 1, l + 1)
    )
    Iz = (
        u * v * _S_lookup(Sz, u - 1, v - 1)
        - 2.0 * beta * u * _S_lookup(Sz, u - 1, v + 1)
        - 2.0 * alpha * v * _S_lookup(Sz, u + 1, v - 1)
        + 4.0 * alpha * beta * _S_lookup(Sz, u + 1, v + 1)
    )

    # 他 2 軸は 1D 重なり
    S_y, S_z = _S_lookup(Sy, k, l), _S_lookup(Sz, u, v)
    S_x = _S_lookup(Sx, i, j)

    T = 0.5 * (Ix * S_y * S_z + Iy * S_x * S_z + Iz * S_x * S_y)
    return float(T)

def contracted_kinetic(gto1, gto2) -> float:
    """
    収縮 GTO の運動エネルギー（一般 l に対応：s/p/d ...）。
    以前の s/p 限定ガードを撤廃し、_primitive_kinetic_general を用いる。
    """
    Tval = 0.0
    for a1, c1 in zip(gto1.exponents, gto1.coefficients):
        for a2, c2 in zip(gto2.exponents, gto2.coefficients):
            N1 = norm_prefactor(gto1.l, gto1.m, gto1.n, a1)
            N2 = norm_prefactor(gto2.l, gto2.m, gto2.n, a2)
            T_prim = _primitive_kinetic_general(
                gto1.l, gto1.m, gto1.n, a1, gto1.center,
                gto2.l, gto2.m, gto2.n, a2, gto2.center
            )
            Tval += (c1 * c2) * (N1 * N2) * T_prim
    return float(Tval)
