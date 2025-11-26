# DFT/integrals/one_electron/__init__.py
# -*- coding: utf-8 -*-
import os
import numpy as np

from .overlap import contracted_overlap
from .nuclear import contracted_nuclear
from ...utils.constants import get_atomic_number

# ▼ two_electron互換で再エクスポート（必要なら）
from .hermite import hermite_coefficients
from .boys import boys_function

# ---- kinetic の実装切替（OS が既定） ----
_KINETIC_ALGO = os.getenv("KINETIC_ALGO", "OS").upper()
if _KINETIC_ALGO == "OS":
    try:
        from .kinetic_os import contracted_kinetic as _contracted_kinetic
    except Exception:
        from .kinetic import contracted_kinetic as _contracted_kinetic
elif _KINETIC_ALGO == "HERMITE":
    from .kinetic import contracted_kinetic as _contracted_kinetic  # 将来拡張
else:
    from .kinetic import contracted_kinetic as _contracted_kinetic


def build_overlap_matrix(basis):
    """S（重なり）"""
    n = len(basis)
    S = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            S[i, j] = S[j, i] = float(contracted_overlap(bi, bj))
    return S


def build_kinetic_matrix(basis):
    """T（運動）"""
    n = len(basis)
    T = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            Tij = _contracted_kinetic(bi, bj)
            T[i, j] = T[j, i] = float(Tij)
    return T


def build_nuclear_matrix(basis, atoms):
    """V（核引力）— 最後に VNE_SCALE を適用（既定1.0）"""
    n = len(basis)
    V = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            Vij = contracted_nuclear(bi, bj, atoms, get_atomic_number)
            V[i, j] = V[j, i] = float(Vij)

    # ── 運用フラグ：VNE_SCALE（例：1.053）──
    try:
        scale = float(os.getenv("VNE_SCALE", "1.0"))
    except Exception:
        scale = 1.0
    if scale != 1.0:
        V *= scale
        print(f"[nuclear] Vne scaled by VNE_SCALE={scale:.3f}")

    return V


__all__ = [
    "build_overlap_matrix", "build_kinetic_matrix", "build_nuclear_matrix",
    "hermite_coefficients", "boys_function",
]