
# DFT/main.py
# -*- coding: utf-8 -*-
import os
import json
import time
import re
import argparse
import numpy as np
from typing import Optional, Dict, Any

from DFT.input.parser import parse_xyz
from DFT.basis.assign_basis import assign_basis_functions
from DFT.scf.rhf import run_rhf, compute_nuclear_repulsion
from DFT.scf.dft import run_dft
from DFT.utils.constants import get_atomic_number
from DFT.exporter.base import Exporter, ExportOptions

# --- one-electron integrals (with safe fallback) ---
try:
    from DFT.integrals.one_electron import (
        build_overlap_matrix,
        build_kinetic_matrix,
        build_nuclear_matrix,
    )
    print("[one-electron] Using OFFICIAL builders (S/T/V).")
except Exception as e:
    print(f"[one-electron] Fallback in use (T=V=0). Reason: {e}")
    from DFT.integrals.one_electron.overlap import contracted_overlap

    def build_overlap_matrix(basis):
        n = len(basis)
        S = np.zeros((n, n))
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis[: i + 1]):
                val = contracted_overlap(bi, bj)
                S[i, j] = S[j, i] = val
        return S

    def build_kinetic_matrix(basis):
        return np.zeros((len(basis), len(basis)))

    def build_nuclear_matrix(basis, atoms):
        return np.zeros((len(basis), len(basis)))

# --- helpers (tags/run_id) ---
def _hill_formula(symbols):
    from collections import Counter
    cnt = Counter(symbols)
    if "C" in cnt:
        order = ["C"] + (["H"] if "H" in cnt else []) + [k for k in sorted(cnt) if k not in ("C", "H")]
    else:
        order = sorted(cnt)
    parts = []
    for k in order:
        n = cnt[k]
        parts.append(k if n == 1 else f"{k}{n}")
    return "".join(parts)

def _safe_tag_from_path(path_str: str, maxlen: int = 64) -> str:
    base = os.path.basename(path_str)
    stem = os.path.splitext(base)[0]
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", stem)
    s = re.sub(r"^[._\-]+", "", s)
    s = re.sub(r"[._\-]+$", "", s)
    return (s or "input")[:maxlen]

def _timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return time.strftime(fmt)

# --- JK safe rebuild (tries candidates and picks non-J one) ---
def _jk_rebuild_safe(basis_funcs, D: np.ndarray):
    """
    Build ERI, J and a 'safe' K:
    J = einsum('rs,mnrs->mn', D, eri)
    K candidates:
    STD : eri_perm = eri ; einsum('rs,mrns->mn', D, eri_perm)
    ALT1: eri_perm = eri.transpose(0,3,2,1); einsum('rs,msrn->mn', D, eri_perm)
    ALT2: eri_perm = eri.transpose(0,1,3,2); einsum('rs,mnsr->mn', D, eri_perm)
    Choose the candidate with the largest J-K_F and hermitize.
    """
    from DFT.integrals.two_electron import build_eri_tensor
    eri = build_eri_tensor(basis_funcs)  # (mnrs)
    J = np.einsum('rs,mnrs->mn', D, eri, optimize=True)
    K_std  = np.einsum('rs,mrns->mn', D, eri, optimize=True)
    K_alt1 = np.einsum('rs,msrn->mn', D, eri.transpose(0, 3, 2, 1), optimize=True)
    K_alt2 = np.einsum('rs,mnsr->mn', D, eri.transpose(0, 1, 3, 2), optimize=True)
    cands = {"STD": K_std, "ALT1": K_alt1, "ALT2": K_alt2}
    diffs = {name: float(np.linalg.norm(J - K)) for name, K in cands.items()}
    best_name = max(diffs, key=diffs.get)
    K_best = cands[best_name]
    J = 0.5 * (J + J.T)
    K_best = 0.5 * (K_best + K_best.T)
    print(f"[jk.safe] picked {best_name}, J-K_F={diffs[best_name]:.3e}")
    return J, K_best

# --- ensure E_nuc is present in details JSON (robust to both layouts) ---
def _inject_E_nuc_to_details(details_path: str, E_nuc_val: float) -> None:
    """
    details.json の構造が
    A) フラット（トップレベルに各フィールド）
    B) ネスト（{"details": {"components": {...}}}）
    のいずれでも、E_nuc を必ず書き込みます（上書き可）。
    """
    try:
        with open(details_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception as e:
        print(f"[export] E_nuc patch skipped (read failed): {e}")
        return
    patched = False
    # パターンA: トップレベルに入れる
    if blob.get("E_nuc") in (None, 0.0):
        blob["E_nuc"] = float(E_nuc_val)
        patched = True
    # パターンB: ネストにも可能なら入れる
    details_obj = blob.setdefault("details", {})
    comps_obj = details_obj.setdefault("components", {})
    if comps_obj.get("E_nuc") in (None, 0.0):
        comps_obj["E_nuc"] = float(E_nuc_val)
        patched = True
    if patched:
        try:
            with open(details_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            print(f"[export] Patched details JSON with E_nuc={E_nuc_val:.12f} Ha -> {details_path}")
        except Exception as e:
            print(f"[export] E_nuc patch skipped (write failed): {e}")

def run_calculation(
    input_path: str,
    basis: str = "sto-3g",
    method: str = "RHF",
    grid_level: int = 3,
    xc_chunk_size: Optional[int] = None,
    xc_d_thresh: float = 0.0,
    xc_validate: bool = True,
    charge: int = 0,
    multiplicity: int = 1,
    debug_eri: bool = False,
) -> Dict[str, Any]:
    """
    RHF または DFT 計算を実行し、成果物を Exporter で
    C:\\Users\\<USER>\\Desktop\\dft\\output\\<run_id>\\ に一元出力します（summary は Exporter に一本化）。
    """
    # --- parse & basis assign ---
    mol = parse_xyz(input_path, charge=charge, multiplicity=multiplicity)
    basis_funcs = assign_basis_functions(mol, basis)
    print(f"Assigned {len(basis_funcs)} basis functions.")

    # --- one-electron matrices ---
    S = build_overlap_matrix(basis_funcs)
    T = build_kinetic_matrix(basis_funcs)
    V = build_nuclear_matrix(basis_funcs, mol.atoms)
    print("[diagnose]")
    print(f"S_F={np.linalg.norm(S):.6e}, T_F={np.linalg.norm(T):.6e}, V_F={np.linalg.norm(V):.6e}")

    # --- SCF ---
    method_up = method.upper()
    if method_up == "RHF":
        E_scf, C, eps, details = run_rhf(S, T, V, basis_funcs, mol)
        print(f"Final RHF Energy: {E_scf:.8f} Hartree")
    elif method_up == "DFT":
        xc_chunk = None if (xc_chunk_size is None or xc_chunk_size == 0) else int(xc_chunk_size)
        E_scf, C, eps, details = run_dft(
            S, T, V, basis_funcs, mol,
            grid_level=grid_level,
            xc_chunk_size=xc_chunk,
            xc_d_thresh=float(xc_d_thresh),
            xc_validate=xc_validate,
        )
        print(f"Final DFT(B3LYP-lite) Energy: {E_scf:.8f} Hartree")
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- run_id 用のタグ類（Exporter 側と整合）---
    symbols   = [a.symbol for a in mol.atoms]
    formula   = _hill_formula(symbols)
    input_tag = _safe_tag_from_path(input_path)
    ts        = _timestamp("%Y%m%d_%H%M%S")

    # --- Exporter init（summary を Exporter に一本化／出力先は固定） ---
    opt = ExportOptions(
        save_matrices=True,
        save_matrices_json=True,
        save_summary=True,
        save_energies=False,    # main 側は CSV energies を書かない
        save_scf_history=False, # main 側は SCF CSV を書かない
        base_out_dir="DFT",
        include_subdir="output",
    )
    exporter = Exporter(opt)

    # --- spin-summed density & H_core ---
    n_electrons = sum(get_atomic_number(a.symbol) for a in mol.atoms) - int(charge)
    n_occ = int(n_electrons // 2)
    D_final = np.zeros_like(S, dtype=np.float64)
    for m in range(n_occ):
        v = C[:, m]
        D_final += 2.0 * np.outer(v, v)
    H_core = T + V

    # --- take J/K/F/Vxc from details if present; otherwise rebuild ---
    mats_details = details.get("matrices", {}) if isinstance(details, dict) else {}
    J_final   = mats_details.get("J_final")
    K_final   = mats_details.get("K_final")
    F_final   = mats_details.get("F_final")
    Vxc_final = mats_details.get("Vxc_final")

    need_rebuild = (J_final is None) or (K_final is None)
    if (not need_rebuild):
        sep = float(np.linalg.norm(J_final - K_final))
        if sep < 1e-10 and np.linalg.norm(J_final) > 1e-10:
            print(f"[jk.guard] J≈K detected (J-K={sep:.3e}); rebuilding K safely.")
            need_rebuild = True
    if need_rebuild:
        try:
            J_final, K_final = _jk_rebuild_safe(basis_funcs, D_final)
        except Exception as e:
            print(f"[jk.safe] rebuild failed: {e}")

    # RHF の場合のみ、F を整合的に再構成（DFT は run_dft 側が F/Vxc を管理）
    if F_final is None and method_up == "RHF" and (J_final is not None) and (K_final is not None):
        F_final = H_core + J_final - 0.5 * K_final
        F_final = 0.5 * (F_final + F_final.T)

    # --- Export via Exporter（run_id の timestamp を main 側 ts に合わせる） ---
    artifacts = exporter.export_final(
        input_path=input_path, method=method_up, basis=basis, mol=mol,
        S=S, T=T, V=V, C=C, eps=eps, E_scf=E_scf,
        get_atomic_number=get_atomic_number,
        details=details,
        timestamp=ts,  # ← 重要：Exporter 側 run_id と完全一致
    )

    # --- ensure E_nuc in details（details JSON 側は保険でパッチ） ---
    try:
        E_nuc_val = compute_nuclear_repulsion(mol.atoms)
    except Exception:
        E_nuc_val = 0.0
    details_path = os.path.join(artifacts["outdir"], f"details_{input_tag}.json")  # ★ outdir を同期
    _inject_E_nuc_to_details(details_path, float(E_nuc_val))

    print("[export] artifacts root:", artifacts["outdir"])
    return {"energy": E_scf, "coeff": C, "eps": eps, "details": details}

# --- CLI エントリポイント（ショートオプション対応・outdir は表示だけ固定文言） ---
def _build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal RHF/DFT runner (outdir fixed to Desktop\\dft\\output)"
    )
    # ★ ショートオプション：-i / -b / -m
    p.add_argument("-i", "--xyz",    required=True,
                   help="座標ファイル（.xyz）パス。短い名前（例: H2.xyz）指定時は DFT/molecure を含む既知パスを自動探索")
    p.add_argument("-b", "--basis",  default="sto-3g", help="基底関数（例：sto-3g, 6-31G）")
    p.add_argument("-m", "--method", default="RHF",    help="手法（例：RHF, DFT）")

    p.add_argument("--charge", type=int, default=0, help="全電荷")
    p.add_argument("--multiplicity", type=int, default=1, help="多重度")
    # DFT オプション
    p.add_argument("--grid-level", type=int, default=3, help="DFT 積分グリッドレベル")
    p.add_argument("--xc-chunk-size", type=int, default=0, help="Vxc チャンクサイズ（0/未指定で自動）")
    p.add_argument("--xc-d-thresh", type=float, default=0.0, help="密度差のしきい値（検証用）")
    p.add_argument("--no-xc-validate", action="store_true", help="XC 検証を無効化")
    return p.parse_args()

if __name__ == "__main__":
    args = _build_cli()
    run_calculation(
        input_path=args.xyz,
        basis=args.basis,
        method=args.method,
        grid_level=args.grid_level,
        xc_chunk_size=(None if args.xc_chunk_size in (None, 0) else args.xc_chunk_size),
        xc_d_thresh=args.xc_d_thresh,
        xc_validate=(not args.no_xc_validate),
        charge=args.charge,
        multiplicity=args.multiplicity,
    )
