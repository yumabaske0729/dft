# DFT/main.py
# -*- coding: utf-8 -*-
import argparse
import sys
import os
import numpy as np
from .input.parser import parse_xyz
from .basis.assign_basis import assign_basis_functions
from .scf.rhf import run_rhf
from .scf.dft import run_dft
from .utils.constants import get_atomic_number
from .exporter import Exporter, ExportOptions

# 公式 one-electron ビルダ（S/T/V）を優先
try:
    from .integrals.one_electron import (
        build_overlap_matrix,
        build_kinetic_matrix,
        build_nuclear_matrix,
    )  # noqa: F401
    print("[one-electron] Using OFFICIAL builders (S/T/V).")
except Exception as e:
    print(f"[one-electron] Fallback in use (T=V=0). Reason: {e}", file=sys.stderr)
    from .integrals.one_electron.overlap import contracted_overlap
    def build_overlap_matrix(basis):
        n = len(basis)
        S = np.zeros((n, n))
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis[: i + 1]):
                val = contracted_overlap(bi, bj)
                S[i, j] = S[j, i] = val
        return S
    def build_kinetic_matrix(basis):
        n = len(basis)
        return np.zeros((n, n))
    def build_nuclear_matrix(basis, atoms):
        n = len(basis)
        return np.zeros((n, n))

def main():
    parser = argparse.ArgumentParser(
        description="Gaussian-like quantum chemistry calculation (RHF/DFT)"
    )
    parser.add_argument("-i", "--input", required=True, help="Input structure file (XYZ format)")
    parser.add_argument("-b", "--basis", default="sto-3g", help="Basis set name (e.g., sto-3g, 6-31g)")
    parser.add_argument("-c", "--charge", type=int, default=0, help="Total molecular charge")
    parser.add_argument("-m", "--multiplicity", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("--method", choices=["RHF", "DFT"], default="RHF", help="Method: RHF or DFT")

    # ERI spot check
    parser.add_argument(
        "--debug-eri",
        action="store_true",
        help="Print selected AO ERIs ((11|11),(11|12),(11|22),(12|12)) for the first two AOs",
    )
    # DFT grid level
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="DFT integration grid level (controls radial points). Example: 5 or 8",
    )
    # ★ 追加: XCストリーミング制御
    parser.add_argument(
        "--xc-chunk-size",
        type=int,
        default=None,
        help="Chunk size for streaming XC integration (set 0 or omit to disable).",
    )
    parser.add_argument(
        "--xc-d-thresh",
        type=float,
        default=0.0,
        help="Density matrix screening threshold |D_mn| < tau ignored (default 0.0).",
    )
    parser.add_argument(
        "--xc-validate",
        type=int,
        default=1,
        help="1: validate streaming vs vectorized and fallback if mismatch; 0: no validation.",
    )

    args = parser.parse_args()

    # --- 入力ファイル探索処理を追加 ---
    input_path = args.input
    if not os.path.isfile(input_path):
        # カレントにファイルがなければ DFT/molecule/ 下を探す
        candidate = os.path.join("DFT", "molecule", os.path.basename(args.input))
        if os.path.isfile(candidate):
            input_path = candidate
            print(f"[info] Input file found in DFT/molecule/: {input_path}")
        else:
            print(f"Error: Input file '{args.input}' not found in current directory or DFT/molecule/", file=sys.stderr)
            sys.exit(1)

    # Load molecule
    try:
        mol = parse_xyz(input_path, charge=args.charge, multiplicity=args.multiplicity)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Assign basis
    basis_funcs = assign_basis_functions(mol, args.basis)
    print(f"Assigned {len(basis_funcs)} basis functions.")

    # -- ERI probe (optional)
    if args.debug_eri and len(basis_funcs) >= 2:
        try:
            from .integrals.two_electron import contracted_eri
            b0, b1 = basis_funcs[0], basis_funcs[1]
            eri_1111 = contracted_eri(b0, b0, b0, b0)
            eri_1112 = contracted_eri(b0, b0, b0, b1)
            eri_1122 = contracted_eri(b0, b0, b1, b1)
            eri_1212 = contracted_eri(b0, b1, b0, b1)
            print(
                "[ERI-probe] "
                f"(11|11)={eri_1111:.12f}, (11|12)={eri_1112:.12f}, "
                f"(11|22)={eri_1122:.12f}, (12|12)={eri_1212:.12f}"
            )
        except Exception as _e:
            print(f"[ERI-probe] failed: {_e}", file=sys.stderr)

    # One-electron matrices
    S = build_overlap_matrix(basis_funcs)
    T = build_kinetic_matrix(basis_funcs)
    V = build_nuclear_matrix(basis_funcs, mol.atoms)

    import numpy as _np
    print(
        f"[diagnose]\n"
        f"S||_F={_np.linalg.norm(S):.6e},\n"
        f"T||_F={_np.linalg.norm(T):.6e},\n"
        f"V||_F={_np.linalg.norm(V):.6e}"
    )

    # Run SCF
    if args.method == "RHF":
        E_scf, C, eps, details = run_rhf(S, T, V, basis_funcs, mol)
        print(f"Final RHF Energy: {E_scf:.8f} Hartree")
    else:
        # DFT: ストリーミング設定を run_dft に渡す
        xc_chunk = None if (args.xc_chunk_size is None or args.xc_chunk_size == 0) else int(args.xc_chunk_size)
        xc_vldt  = bool(int(args.xc_validate))
        E_scf, C, eps, details = run_dft(
            S, T, V, basis_funcs, mol,
            grid_level=args.grid_level,
            xc_chunk_size=xc_chunk,
            xc_d_thresh=float(args.xc_d_thresh),
            xc_validate=xc_vldt,
        )
        print(f"Final DFT(B3LYP-lite) Energy: {E_scf:.8f} Hartree")

    # Export（出力先：DFT/output/）
    exporter = Exporter(
        ExportOptions(
            base_out_dir=os.path.join("DFT", "output"),
            include_subdir="",
        )
    )
    exporter.export_final(
        input_path=input_path,
        method=args.method,
        basis=args.basis,
        mol=mol,
        S=S, T=T, V=V, C=C, eps=eps, E_scf=E_scf,
        get_atomic_number=get_atomic_number,
        details=details,
    )

if __name__ == "__main__":
    main()
