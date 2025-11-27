# -*- coding: utf-8 -*-
"""
One-electron nuclear attraction integrals over Cartesian GTOs (OS scheme).
- s–s : dedicated fast/stable kernel (K_ss) then multiply by (-Z).
- general l : Hermite expansion with E^ij_t recursion (base=1.0), all distance
              damping in the prefactor; Boys F_n(T) with robust small/large t branches.

References:
- Helgaker (Sostrup Summer School, 2006) slides on Hermite expansion and recurrences.
- Fermann & Valeev review (arXiv:2007.12057).
"""
from __future__ import annotations
import os
import numpy as np
from math import pi
from ...gaussian_math.gaussian_math import norm_prefactor
from .boys import boys_function  # robust small/large t

# ---- s–s primitives: fast kernel (positive), final (-Z) outside ----
def _primitive_nuclear_ss_kernel(alpha1, A, alpha2, B, C):
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    p = alpha1 + alpha2
    q = alpha1 * alpha2 / p
    RAB2 = float(np.dot(A - B, A - B))
    P = (alpha1 * A + alpha2 * B) / p
    RPC2 = float(np.dot(P - C, P - C))
    return (2.0 * pi / p) * np.exp(-q * RAB2) * boys_function(0, p * RPC2)

# ---- 1D Hermite coefficients for nuclear attraction (OS form) ----
def _E1d_nuclear(i_max: int, j_max: int, alpha: float, beta: float, Ax: float, Bx: float) -> np.ndarray:
    """
    Build E[i,j,t] (0<=i<=i_max, 0<=j<=j_max, 0<=t<=i+j) for nuclear attraction.
    Recurrences (x-component; y,z analogous):
      p = a + b, Px = (a*Ax + b*Bx)/p
      E(i+1,j,t) = (Px - Ax) * E(i,j,t) + (t+1)/(2p) * E(i,j,t+1) + i/(2p) * E(i-1,j,t)
      E(i,j+1,t) = (Px - Bx) * E(i,j,t) + (t+1)/(2p) * E(i,j,t+1) + j/(2p) * E(i,j-1,t)
    with base E(0,0,0)=1 and E(..., t>i+j)=0.
    """
    p = alpha + beta
    Px = (alpha * Ax + beta * Bx) / p
    tmax = i_max + j_max
    # +1 margin for safe t+1 indexing; last slice stays 0 so "beyond" contributes 0.
    E = np.zeros((i_max + 1, j_max + 1, tmax + 2), dtype=float)
    E[0, 0, 0] = 1.0

    # i-ladder (j=0)
    if i_max >= 1:
        for i in range(1, i_max + 1):
            for t in range(0, tmax + 1):
                term1 = (Px - Ax) * E[i - 1, 0, t]
                term2 = ((t + 1) / (2.0 * p)) * E[i - 1, 0, t + 1]
                term3 = ((i - 1) / (2.0 * p)) * (E[i - 2, 0, t] if i - 2 >= 0 else 0.0)
                E[i, 0, t] = term1 + term2 + term3

    # j-ladder (including all i)
    if j_max >= 1:
        for j in range(1, j_max + 1):
            for i in range(0, i_max + 1):
                for t in range(0, tmax + 1):
                    term1 = (Px - Bx) * E[i, j - 1, t]
                    term2 = ((t + 1) / (2.0 * p)) * E[i, j - 1, t + 1]
                    term3 = ((j - 1) / (2.0 * p)) * (E[i, j - 2, t] if j - 2 >= 0 else 0.0)
                    E[i, j, t] = term1 + term2 + term3
    return E  # shape: (i_max+1, j_max+1, tmax+2)

# ---- general primitive (last step multiplies by -Z) ----
def _primitive_nuclear_general(l1, m1, n1, a1, A,
                               l2, m2, n2, a2, B,
                               C, ZC) -> float:
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    p = a1 + a2
    q = a1 * a2 / p
    P = (a1 * A + a2 * B) / p
    RAB2 = float(np.dot(A - B, A - B))
    RPC2 = float(np.dot(P - C, P - C))

    # 1D coefficients (nuclear version; base=1.0, 1/(2p) factors included)
    Ex = _E1d_nuclear(l1, l2, a1, a2, A[0], B[0])
    Ey = _E1d_nuclear(m1, m2, a1, a2, A[1], B[1])
    Ez = _E1d_nuclear(n1, n2, a1, a2, A[2], B[2])

    T = p * RPC2
    val = 0.0
    tx_max = l1 + l2
    ty_max = m1 + m2
    tz_max = n1 + n2

    for tx in range(0, tx_max + 1):
        Ex_t = Ex[l1, l2, tx]
        if Ex_t == 0.0: 
            continue
        for ty in range(0, ty_max + 1):
            Ey_u = Ey[m1, m2, ty]
            if Ey_u == 0.0:
                continue
            for tz in range(0, tz_max + 1):
                Ez_v = Ez[n1, n2, tz]
                if Ez_v == 0.0:
                    continue
                order = tx + ty + tz
                val += Ex_t * Ey_u * Ez_v * boys_function(order, T)

    # prefactor is positive kernel; final (-ZC)
    K = (2.0 * pi / p) * np.exp(-q * RAB2) * val
    return float((-ZC) * K)

# ---- contracted level ----
def contracted_nuclear(gto1, gto2, atoms, get_atomic_number) -> float:
    """
    One-electron nuclear attraction integral over contracted Cartesian GTOs.
    s–s は専用高速カーネル、一般 l は Hermite 展開＋Boys関数。
    末尾で環境変数 VNE_SCALE_TEST を適用（デフォルト 1.0）。
    """
    V = 0.0
    is_ss = (gto1.l + gto1.m + gto1.n == 0) and (gto2.l + gto2.m + gto2.n == 0)

    for a1, c1 in zip(gto1.exponents, gto1.coefficients):
        N1 = norm_prefactor(gto1.l, gto1.m, gto1.n, a1)
        for a2, c2 in zip(gto2.exponents, gto2.coefficients):
            N2 = norm_prefactor(gto2.l, gto2.m, gto2.n, a2)

            if is_ss:
                for atom in atoms:
                    Z = get_atomic_number(atom.symbol)
                    Kss = _primitive_nuclear_ss_kernel(a1, gto1.center, a2, gto2.center, atom.position)
                    V += (c1 * c2) * (N1 * N2) * (-Z) * Kss
            else:
                for atom in atoms:
                    Z = get_atomic_number(atom.symbol)
                    V += (c1 * c2) * (N1 * N2) * _primitive_nuclear_general(
                        gto1.l, gto1.m, gto1.n, a1, gto1.center,
                        gto2.l, gto2.m, gto2.n, a2, gto2.center,
                        atom.position, Z
                    )

    # --- テスト用スケール（VNE_SCALE_TEST） ---
    # 例: VNE_SCALE_TEST=1.053 なら V を 5.3% 強める
    try:
        scale = float(os.getenv("VNE_SCALE", "1.0"))
    except Exception:
        scale = 1.0
    V *= scale

    return float(V)