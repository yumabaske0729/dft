# DFT/integrals/one_electron/overlap.py  ← 全文置換
import numpy as np
from ...gaussian_math.gaussian_math import norm_prefactor
from .kinetic import _os_1d_overlap_table

def primitive_overlap(l1, m1, n1, alpha1, A, l2, m2, n2, alpha2, B):
    """
    正規化済みプリミティブ重なり S_ab（OS 1D テーブルで安定に構築）
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    # 1D OS テーブル（未正規化1D重なり）
    Sx, _ = _os_1d_overlap_table(l1, l2, A[0], B[0], alpha1, alpha2)
    Sy, _ = _os_1d_overlap_table(m1, m2, A[1], B[1], alpha1, alpha2)
    Sz, _ = _os_1d_overlap_table(n1, n2, A[2], B[2], alpha1, alpha2)
    S_prim = float(Sx[l1, l2] * Sy[m1, m2] * Sz[n1, n2])

    # 正規化（Cartesian GTO）
    N1 = norm_prefactor(l1, m1, n1, alpha1)
    N2 = norm_prefactor(l2, m2, n2, alpha2)
    return N1 * N2 * S_prim

def contracted_overlap(gto1, gto2):
    """
    収縮 GTO の重なり：Σ c1 c2 * (正規化済みプリミティブ S)
    """
    assert len(gto1.exponents) == len(gto1.coefficients), "gto1 exponents/coeffs length mismatch"
    assert len(gto2.exponents) == len(gto2.coefficients), "gto2 exponents/coeffs length mismatch"
    Sval = 0.0
    for a1, c1 in zip(gto1.exponents, gto1.coefficients):
        for a2, c2 in zip(gto2.exponents, gto2.coefficients):
            Sval += (c1 * c2) * primitive_overlap(
                gto1.l, gto1.m, gto1.n, a1, gto1.center,
                gto2.l, gto2.m, gto2.n, a2, gto2.center
            )
    return float(Sval)
