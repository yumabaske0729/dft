# -*- coding: utf-8 -*-
"""
check_npz_diagnostics.py — AO npz/JSON の整合性診断（D=2P / P 両対応）
Usage:
  python check_npz_diagnostics.py [--npz PATH] [--details PATH] [--d2p [0|1]]

Notes:
  - D=2P（既定）:
      Tr[DS] = N_elec
      idempotency:  D S D = 2 D   → 残差 ||DSD - 2D||_F
      eigenvals of S^{+1/2} D S^{+1/2} ≈ 2 or 0
  - P（片スピン密度）:
      Tr[PS] = N_elec/2
      idempotency:  P S P = P     → 残差 ||PSP - P||_F
      eigenvals of S^{+1/2} P S^{+1/2} ≈ 1 or 0
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np


def _latest(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat))
    if not cands:
        raise FileNotFoundError("対象ファイルが見つかりません。先に計算を実行してください。")
    return max(cands, key=os.path.getmtime)


def pick_latest_npz():
    return _latest([os.path.join("DFT", "output", "*_*_*_RHF", "matrices_*.npz")])


def pick_latest_details(method="RHF"):
    method = method.upper()
    return _latest([os.path.join("DFT", "output", f"*_*_*_{method}", "details_*.json")])


def _safe_read_details(details_path: str):
    with open(details_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    det = data.get("details", {})
    comps = det.get("components", {})
    diags = det.get("diagnostics", {})
    counts = det.get("counts", {})
    n_elec = counts.get("n_electrons", None)
    if n_elec is None:
        base = os.path.dirname(details_path)
        stem = os.path.basename(details_path).replace("details_", "summary_")
        summary_path = os.path.join(base, stem)
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                n_elec = s.get("results", {}).get("n_electrons", None)
            except Exception:
                pass
    return comps, diags, counts, n_elec


def _S_possqrt(S: np.ndarray) -> np.ndarray:
    # S = U diag(lam) U^T, S^{+1/2} = U diag(sqrt(lam)) U^T
    evals, U = np.linalg.eigh(S)
    sqrt = np.where(evals > 1e-12, np.sqrt(evals), 0.0)
    return U @ np.diag(sqrt) @ U.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="matrices_*.npz のパス")
    ap.add_argument("--details", default=None, help="details_*.json のパス")
    # 値省略可: 省略時は const=1（D=2P）、明示 0 で P
    ap.add_argument("--d2p", nargs="?", const=1, type=int, choices=[0, 1], default=1,
                    help="1: D=2P（既定; 値省略でも1） , 0: P（片スピン密度）")
    args = ap.parse_args()

    npz_path = args.npz or pick_latest_npz()
    z = np.load(npz_path)
    print(f"[npz] {npz_path}\n")

    # 必須配列
    D = z["D_final"]
    S = z["S"]
    H = z["H_core"] if "H_core" in z.files else (z["T"] + z["V"])
    J = z["J_final"]
    K = z["K_final"]
    F = z["F_final"]

    # エネルギー（常に D=2P を基準に評価）
    EJ = float(np.einsum("mn,mn->", D, J))
    EK = float(np.einsum("mn,mn->", D, K))
    EH = float(np.einsum("mn,mn->", D, H))
    E2e = 0.5 * (EJ - 0.5 * EK)
    Eele_JK = EH + E2e
    Eele_F = 0.5 * float(np.einsum("mn,mn->", D, (H + F)))

    print("-- energy components (from npz) --")
    print(f"Tr[D H_core]   = {EH: .12f} Ha")
    print(f"sum D·J        = {EJ: .12f} Ha")
    print(f"sum D·K        = {EK: .12f} Ha")
    print(f"E_2e(recalc)   = {E2e: .12f} Ha")
    print(f"E_elec(J/K)    = {Eele_JK: .12f} Ha")
    print(f"E_elec(½Tr[D(H+F)]) = {Eele_F: .12f} Ha")
    print(f"ΔE_elec (J/K - Fock) = {Eele_JK - Eele_F: .12e} Ha\n")

    # K の相対振幅
    R_KJ = (EK / EJ) if EJ != 0.0 else float("nan")
    print(f"R_KJ = (sum D·K)/(sum D·J) = {R_KJ:.6f}\n")

    # details
    details_path = args.details or pick_latest_details("RHF")
    print(f"[details] {details_path}")
    comps, diags, counts, n_elec = _safe_read_details(details_path)
    E_nuc = comps.get("E_nuc", None)
    print(f"E_nuc (from details) = {E_nuc}")
    print(f"N_elec (from details/summary) = {n_elec}\n")

    # Overlap diagnostics
    print("-- Overlap S -- shape:", S.shape)
    evals_S, _ = np.linalg.eigh(S)
    condS = float(np.max(evals_S) / np.min(evals_S))
    print(f"S diag min/max  = {np.min(np.diag(S)):.6e} / {np.max(np.diag(S)):.6e}")
    print(f"S condition number = {condS:.6e}")

    # D=2P or P での検査行列 M を決定
    d2p_flag = int(args.d2p)
    M = D if d2p_flag == 1 else 0.5 * D
    TrMS = float(np.einsum("mn,mn->", M, S))
    if d2p_flag == 1:
        print(f"Tr[D S] = {TrMS:.12f}  (D=2P なら #electrons と一致が正)")
    else:
        print(f"Tr[P S] = {TrMS:.12f}  (P なら #electrons/2 と一致が正)")
    print()

    # Idempotency diagnostics
    print("-- Density diagnostics --")
    anti = np.linalg.norm(M - M.T)
    print(f"||M - M.T||_F = {anti:.6e}")
    if d2p_flag == 1:
        resid = np.linalg.norm(M @ S @ M - 2.0 * M)
        print(f"||D S D - 2D||_F = {resid:.6e}  (D=2P の冪等性残差；small is good)")
    else:
        resid = np.linalg.norm(M @ S @ M - M)
        print(f"||P S P - P||_F = {resid:.6e}  (P の冪等性残差；small is good)")

    # generalized eigenvalues via S^{+1/2} M S^{+1/2}
    S_sqrt = _S_possqrt(S)
    A = S_sqrt @ M @ S_sqrt
    w, _ = np.linalg.eigh(0.5 * (A + A.T))
    w_sorted = np.sort(w)[::-1]
    top = np.array2string(w_sorted[:min(10, w_sorted.size)], precision=8, separator=' ')
    lab = "D=2P ⇒ ~2/0" if d2p_flag == 1 else "P ⇒ ~1/0"
    print(f"gen. eigvals of S^{1/2} M S^{1/2} (top 10) [{lab}]")
    print(top)
    print(f"# basis = {M.shape[0]}")
    print("\n-- end diagnostics --")


if __name__ == "__main__": main()
