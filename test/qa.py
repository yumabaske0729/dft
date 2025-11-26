# File: qa_jk_energy.py
# -*- coding: utf-8 -*-
"""
qa_jk_energy.py
検査用スクリプト（改良版）
- Jupyter が渡す -f / --profile 等のフラグを無視して、最初に見つかった存在するファイル引数を使います。
- 指定ファイルが無ければデモ（ランダム行列）を実行します。
使い方:
    python qa_jk_energy.py path/to/matrices.npz
    # or (in jupyter)
    !python qa_jk_energy.py path/to/matrices.npz
npz は keys: D, J, K, (H_core optional), (E_nuc optional)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
try:
    # relative import assuming package layout; if not available, import will fail and we fallback below
    from DFT.energy import compute_energy
except Exception:
    # fallback simple implementation if module not importable
    def compute_energy(D, H_core, J, K, E_nuc, *, method='RHF', exch_fraction=1.0, V_xc_matrix=None, E_xc_grid=0.0, debug=False):
        # minimal compatible fallback (not intended for production)
        if H_core is None:
            H_core = np.zeros_like(D)
        sum_DJ = float(np.einsum('ij,ij->', D, J))
        sum_DK = float(np.einsum('ij,ij->', D, K))
        G = 2.0 * J - float(exch_fraction) * K
        E_one = float(np.einsum('ij,ij->', D, H_core))
        E_2e = 0.5 * float(np.einsum('ij,ij->', D, G))
        E_xc_grid_val = float(E_xc_grid) if E_xc_grid is not None else 0.0
        E_xc_mat_term = 0.0
        if V_xc_matrix is not None:
            E_xc_mat_term = float(np.einsum('ij,ij->', D, V_xc_matrix))
        E_elec = E_one + E_2e + E_xc_grid_val - E_xc_mat_term
        E_total = E_elec + float(E_nuc)
        diag = {
            'method': method,
            'n_basis': D.shape[0],
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
            print("fallback energy_diag:", diag)
        return E_total, diag


def load_matrices_from_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, float]:
    arr = np.load(path)
    D = arr['D']
    H_core = arr['H_core'] if 'H_core' in arr else None
    J = arr['J']
    K = arr['K']
    E_nuc = float(arr['E_nuc']) if 'E_nuc' in arr else 0.0
    return D, H_core, J, K, E_nuc


def find_valid_path_from_argv(argv) -> Optional[str]:
    """
    argv を走査して、'-' で始まるフラグを無視し、
    最初に見つかった実在ファイルパスを返す。なければ None を返す。
    This avoids Jupyter's '-f' causing FileNotFoundError.
    """
    for token in argv[1:]:
        if not token:
            continue
        if token.startswith('-'):
            # skip flags like -f, --profile, etc.
            continue
        p = Path(token)
        if p.exists():
            return str(p)
    return None


def main(argv):
    path = find_valid_path_from_argv(argv)
    if path is None:
        print("No valid matrices file provided on command line (ignored flags like -f). Running demo with random matrices.")
        n = 5
        np.random.seed(0)
        D = np.random.rand(n, n); D = 0.5 * (D + D.T)
        H_core = np.random.rand(n, n); H_core = 0.5 * (H_core + H_core.T)
        J = np.random.rand(n, n); J = 0.5 * (J + J.T)
        K = np.random.rand(n, n); K = 0.5 * (K + K.T)
        E_nuc = 1.234
    else:
        print(f"Loading matrices from: {path}")
        D, H_core, J, K, E_nuc = load_matrices_from_npz(Path(path))

    # compute using canonical routine; debug True for diagnostic prints
    E_total, diag = compute_energy(D, H_core, J, K, E_nuc, method='RHF', exch_fraction=1.0, debug=True)

    print("\nResult summary (selected):")
    for k in ('sum_DJ', 'sum_DK', 'E_2e', 'E_elec', 'E_total'):
        if k in diag:
            print(f"  {k}: {diag[k]}")

    # print full diag if desired
    # print(json.dumps(diag, indent=2))

if __name__ == "__main__":
    main(sys.argv)
