# File: DFT/energy.py
# -*- coding: utf-8 -*-
"""
DFT/energy.py
汎用エネルギー集計ルーチン（RHF / KS-DFT / ハイブリッド対応）
compute_energy を使えば SCF や出力生成で正しい E_2e/E_elec/E_total を得られます。

使い方（例）:
    from DFT.energy import compute_energy
    E_total, diag = compute_energy(D, H_core, J, K, E_nuc,
                                    method='RHF', exch_fraction=1.0,
                                    V_xc_matrix=None, E_xc_grid=0.0, debug=True)
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np


def _safe_einsum(a: np.ndarray, b: np.ndarray) -> float:
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs must be numpy.ndarray")
    if a.shape != b.shape:
        raise ValueError(f"Matrix shape mismatch: {a.shape} vs {b.shape}")
    return float(np.einsum('ij,ij->', a, b))


def compute_energy(
    D: np.ndarray,
    H_core: np.ndarray,
    J: np.ndarray,
    K: np.ndarray,
    E_nuc: float,
    *,
    method: str = 'RHF',
    exch_fraction: float = 1.0,
    V_xc_matrix: Optional[np.ndarray] = None,
    E_xc_grid: float = 0.0,
    debug: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    電子・全エネルギーを返す。

    引数:
      D: 密度行列（スピン和 D、閉殻なら trace(D·S) = 電子数）
      H_core: コアハミルトニアン（T + V_ne）
      J, K: Coulomb / Exchange 行列
      E_nuc: 核反発エネルギー（スカラー）

    キーワード:
      method: 'RHF' または 'DFT'（表示用）
      exch_fraction: exact exchange の割合（RHF=1.0, 純DFT=0.0）
      V_xc_matrix: 基底上 V_xc 行列（DFT のとき）
      E_xc_grid: グリッドで積分した E_xc = ∫ ρ ε_xc d r（DFT のとき）
      debug: True なら詳細 diagnostics をプリント

    返り値:
      E_total: float
      diagnostics: dict
    """
    # sanity checks
    for name, mat in (('D', D), ('H_core', H_core), ('J', J), ('K', K)):
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"{name} must be numpy.ndarray, got {type(mat)}")
    n = D.shape[0]
    if D.shape != (n, n) or H_core.shape != (n, n) or J.shape != (n, n) or K.shape != (n, n):
        raise ValueError("All matrices (D, H_core, J, K) must be square and same shape")

    # 基本和
    sum_DJ = _safe_einsum(D, J)  # EJ
    sum_DK = _safe_einsum(D, K)  # EK

    # G の定義（ハイブリッド対応）
    G = 2.0 * J - float(exch_fraction) * K

    # 1電子寄与
    E_one = _safe_einsum(D, H_core)

    # 2電子寄与（正しい式）
    # E_2e = 0.5 * Tr[D * G] = sum_DJ - 0.5 * exch_fraction * sum_DK
    E_2e = 0.5 * _safe_einsum(D, G)

    # DFT の XC の扱い（注意：E_xc_grid と ∑ D V_xc は両方扱う）
    E_xc_grid_val = float(E_xc_grid) if E_xc_grid is not None else 0.0
    E_xc_mat_term = 0.0
    if V_xc_matrix is not None:
        if not isinstance(V_xc_matrix, np.ndarray):
            raise TypeError("V_xc_matrix must be numpy.ndarray or None")
        if V_xc_matrix.shape != (n, n):
            raise ValueError("V_xc_matrix shape mismatch")
        E_xc_mat_term = _safe_einsum(D, V_xc_matrix)

    # Electronic energy (HF/KS unified)
    E_elec = E_one + E_2e + E_xc_grid_val - E_xc_mat_term

    # Total
    E_total = E_elec + float(E_nuc)

    diagnostics = {
        'method': method,
        'n_basis': n,
        'exch_fraction': float(exch_fraction),
        'sum_DJ': sum_DJ,
        'sum_DK': sum_DK,
        'E_one': E_one,
        'E_2e': E_2e,
        'E_xc_grid': E_xc_grid_val,
        'E_xc_mat_term': E_xc_mat_term,
        'E_elec': E_elec,
        'E_nuc': float(E_nuc),
        'E_total': E_total
    }

    if debug:
        print(
            "energy_diag:",
            f"method={method}",
            f"n_basis={n}",
            f"sum_DJ={sum_DJ:.12f}",
            f"sum_DK={sum_DK:.12f}",
            f"E_one={E_one:.12f}",
            f"E_2e={E_2e:.12f}",
            f"E_xc_grid={E_xc_grid_val:.12f}",
            f"E_xc_mat_term={E_xc_mat_term:.12f}",
            f"E_elec={E_elec:.12f}",
            f"E_nuc={E_nuc:.12f}",
            f"E_total={E_total:.12f}"
        )

    return E_total, diagnostics


# quick sanity check if run directly
if __name__ == "__main__":
    import numpy as _np
    _np.random.seed(1)
    n = 7
    D = _np.random.rand(n, n); D = 0.5 * (D + D.T)
    H_core = _np.random.rand(n, n); H_core = 0.5 * (H_core + H_core.T)
    J = _np.random.rand(n, n); J = 0.5 * (J + J.T)
    K = _np.random.rand(n, n); K = 0.5 * (K + K.T)
    E_nuc = 10.0
    E_total, diag = compute_energy(D, H_core, J, K, E_nuc, debug=True)
    print("Sanity check E_total:", E_total)
