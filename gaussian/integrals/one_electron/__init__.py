# DFT/integrals/one_electron/__init__.py
import os
import numpy as np
from .overlap import contracted_overlap
from .nuclear import contracted_nuclear
from ...utils.constants import get_atomic_number

# ▼ 追加：two_electron.py 互換のために再エクスポート
from .hermite import hermite_coefficients
from .boys import boys_function

# --- kinetic の実装切替（将来 GUI から設定・保存できる前提の戦略切替） ---
_KINETIC_ALGO = os.getenv("KINETIC_ALGO", "OS").upper()
if _KINETIC_ALGO == "OS":
    try:
        from .kinetic_os import contracted_kinetic as _contracted_kinetic
    except Exception:
        from .kinetic import contracted_kinetic as _contracted_kinetic  # フォールバック（s のみ）
elif _KINETIC_ALGO == "HERMITE":
    from .kinetic_hermite import contracted_kinetic as _contracted_kinetic  # 将来追加
else:
    from .kinetic import contracted_kinetic as _contracted_kinetic

def build_overlap_matrix(basis):
    n = len(basis)
    S = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            S[i, j] = S[j, i] = float(contracted_overlap(bi, bj))
    return S

def build_kinetic_matrix(basis):
    n = len(basis)
    T = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            Tij = _contracted_kinetic(bi, bj)
            T[i, j] = T[j, i] = float(Tij)
    return T

def build_nuclear_matrix(basis, atoms):
    n = len(basis)
    V = np.zeros((n, n), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[:i+1]):
            V[i, j] = V[j, i] = float(contracted_nuclear(bi, bj, atoms, get_atomic_number))
    return V

# （任意）公開シンボルを明示
__all__ = [
    "build_overlap_matrix", "build_kinetic_matrix", "build_nuclear_matrix",
    "hermite_coefficients", "boys_function",
]
