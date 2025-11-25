# -*- coding: utf-8 -*-
"""
check4.py — HEAVY diagnostics (revised, robust)
- Robust S/S_half checks (correct formulas)
- Natural occupations via S^{-1/2} orthogonalization
- Energy/Fock identity + recompose F from (H,J,K)
- NEW: If F_final is missing in NPZ, reconstruct F = H + J - 0.5*K
- NEW: K_est = 2*(H + J - F) to cross-check saved K (no ERI needed)
- Optional: ERI rebuild & K-axes probe (if available)
"""
from __future__ import annotations
import os, sys, glob, json, math, traceback
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

try:
    import scipy.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def info(s): print(f"[INFO] {s}")
def ok(s):   print(f"[ OK ] {s}")
def warn(s): print(f"[WARN] {s}")
def err(s):  print(f"[ERR ] {s}")

# -------- file pickers --------
def pick_latest(patterns: List[str]) -> Optional[str]:
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands: return None
    return max(cands, key=os.path.getmtime)

def pick_npz_default() -> Optional[str]:
    return pick_latest([
        os.path.join("DFT","output","*_*_*_RHF","matrices_*.npz"),
        os.path.join("output","*_*_*_RHF","matrices_*.npz"),
        os.path.join("*_*_*_RHF","matrices_*.npz"),
    ])

def pick_details_default() -> Optional[str]:
    return pick_latest([
        os.path.join("DFT","output","*_*_*_RHF","details_*.json"),
        os.path.join("output","*_*_*_RHF","details_*.json"),
        os.path.join("*_*_*_RHF","details_*.json"),
    ])

# -------- linear algebra helpers --------
def build_S_inv_sqrt(S: np.ndarray, thresh: float = 1e-10) -> Tuple[np.ndarray, Dict[str,float]]:
    evals, U = np.linalg.eigh(S)
    inv_sqrt = np.where(evals > thresh, evals**-0.5, 0.0)
    X = U @ np.diag(inv_sqrt) @ U.T
    stats = {
        "S_eig_min": float(evals.min()),
        "S_eig_max": float(evals.max()),
        "S_cond_est": float(evals.max()/evals.min()) if evals.min()>0 else float('inf')
    }
    return X, stats

def einsum_trace(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.einsum('mn,mn->', A, B, optimize=True))

# -------- energy identities --------
def fock_identities(D: np.ndarray, H: np.ndarray, J: np.ndarray, K: np.ndarray, F: np.ndarray) -> Dict[str, float]:
    EJ, EK, EH = einsum_trace(D,J), einsum_trace(D,K), einsum_trace(D,H)
    E2e = 0.5*(EJ - 0.5*EK)
    Eele_JK = EH + E2e
    Eele_F  = 0.5*einsum_trace(D, H + F)
    return {"TrD_H":EH, "TrD_J":EJ, "TrD_K":EK, "E_2e":E2e, "E_elec(JK)":Eele_JK, "E_elec(0.5Tr)":Eele_F,
            "dE_elec":Eele_JK - Eele_F}

# -------- main --------
def run(args):
    # files
    npz_path = args.npz or pick_npz_default()
    details_path = args.details or pick_details_default()
    if npz_path is None:
        err("matrices_*.npz が見つかりません。"); sys.exit(2)
    info(f"Using npz: {npz_path}")
    if details_path:
        info(f"Using details: {details_path}")

    # env / module origin
    env_echo = {k: os.getenv(k) for k in ["K_EINSUM","K_SCALE_TEST","ERI_SYM_8","JK_DIAG","PYTHONDONTWRITEBYTECODE","PYTHONPATH"]}
    info("Env snapshot: " + ", ".join(f"{k}={v}" for k,v in env_echo.items()))
    try:
        import DFT, DFT.scf.jk as _jk, DFT.scf.rhf as _rhf, DFT.integrals.two_electron as _tei
        info("Module files: " + ", ".join([
            f"DFT={DFT.__file__}", f"jk={_jk.__file__}", f"rhf={_rhf.__file__}", f"two_electron={_tei.__file__}"
        ]))
    except Exception as e:
        warn("DFT modules import issue: " + str(e))

    # --- load NPZ robustly ---
    z = np.load(npz_path, allow_pickle=True)
    npz_keys = set(z.files)
    info(f"NPZ keys: {sorted(npz_keys)}")

    # required
    D = np.asarray(z["D_final"])
    S = np.asarray(z["S"])

    # H = H_core if present, else T+V if available
    T = np.asarray(z["T"]) if "T" in npz_keys else None
    V = np.asarray(z["V"]) if "V" in npz_keys else None
    if "H_core" in npz_keys:
        H = np.asarray(z["H_core"])
    elif T is not None and V is not None:
        H = T + V
        warn("H_core missing in NPZ; using H = T + V")
    else:
        raise KeyError("H_core is not available and T/V are insufficient to rebuild it.")

    # J/K are needed for identities and (if needed) F reconstruction
    if "J_final" not in npz_keys or "K_final" not in npz_keys:
        raise KeyError("J_final or K_final is missing in NPZ — exporter側の保存対象を確認してください。")
    J = np.asarray(z["J_final"])
    K = np.asarray(z["K_final"])

    # F: prefer saved F_final; if absent, reconstruct for RHF: F = H + J - 0.5*K
    if "F_final" in npz_keys:
        F = np.asarray(z["F_final"])
    else:
        warn("F_final is not a file in the archive — reconstructing F = H + J - 0.5*K for RHF.")
        F = H + J - 0.5 * K

    # optional
    C = np.asarray(z["C"]) if "C" in npz_keys else (np.asarray(z["C_final"]) if "C_final" in npz_keys else None)
    eps = np.asarray(z["eps"]) if "eps" in npz_keys else None
    S_half = np.asarray(z["S_half"]) if "S_half" in npz_keys else None

    # --- S / S_half ---
    info("=== Overlap S / S_half (correct checks) ===")
    X_re, Sstats = build_S_inv_sqrt(S)
    ok(f"S eig min/max = {Sstats['S_eig_min']:.6e} / {Sstats['S_eig_max']:.6e}, cond≈{Sstats['S_cond_est']:.3e}")
    if S_half is not None:
        n1 = np.linalg.norm(S_half.T @ S @ S_half - np.eye(S.shape[0]))
        n2 = np.linalg.norm(X_re.T   @ S @ X_re   - np.eye(S.shape[0]))
        print(f"[S_half] \nS_half^T S S_half - I\n_F (saved) = {n1:.6e}")
        print(f"[S_half] \nX_re^T S X_re - I\n_F (recalc)= {n2:.6e}")
        print(f"[S_half] \nS_half_saved - X_re\n_F = {np.linalg.norm(S_half - X_re):.6e}")
    else:
        warn("S_half not present")

    # --- C / D ---
    info("=== MO(C) normalization / D reconstruction ===")
    if C is not None:
        dev = np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1])))
        print(f"[C] max\nC^T S C - I\n = {dev:.6e}")
    print(f"[D] \nD - D^T\n_F = {np.linalg.norm(D - D.T):.6e}")
    TrDS = einsum_trace(D, S)
    print(f"[D] Tr[D S] = {TrDS:.12f}")
    print(f"[D] \nD S D - 2D\n_F = {np.linalg.norm(D @ S @ D - 2.0*D):.6e}")
    D_orth = X_re.T @ D @ X_re
    P_orth = 0.5 * D_orth
    print(f"[D] \nP_orth^2 - P_orth\n_F = {np.linalg.norm(P_orth @ P_orth - P_orth):.6e}")
    occ = np.linalg.eigvalsh(D_orth)
    occ = np.sort(np.real_if_close(occ))[::-1]
    print("[D] natural occ (orth basis, top 12): " + ", ".join(f"{v:.6e}" for v in occ[:12]))
    if np.max(occ) > 2.0001 or np.min(occ) < -1e-6:
        warn("natural occ (orth) out of [0,2] -> investigate D/C saving")
    if C is not None:
        nocc = int(round(TrDS))//2
        D_from_C = 2.0*(C[:, :nocc] @ C[:, :nocc].T)
        print(f"[D] \nD - 2*C_occ C_occ^T\n_F = {np.linalg.norm(D - D_from_C):.6e}")

    # --- Energy / Fock identity ---
    info("=== Energy splits & Fock identity ===")
    res = fock_identities(D, H, J, K, F)
    for k,v in res.items():
        if k in ("E_elec(JK)","E_elec(0.5Tr)","dE_elec"):
            print(f"{k:16s} = {v:+.12f}")
        else:
            print(f"{k:16s} = {v:.12f}")
    R = res["TrD_K"]/res["TrD_J"] if res["TrD_J"]!=0 else float('nan')
    print(f"R_KJ = {R:.6f}")
    print(f"\nJ-K\n_F = {np.linalg.norm(J-K):.6e}, \nJ-J^T\n_F = {np.linalg.norm(J-J.T):.6e}, \nK-K^T\n_F = {np.linalg.norm(K-K.T):.6e}")
    if args.strict and np.allclose(J, K):
        raise RuntimeError("J and K are IDENTICAL; K contraction or saving is wrong.")

    # --- Cross-check via recomposition (no ERI needed) ---
    info("=== Recompose checks (F_fromJK, K_est, J_est) ===")
    F_fromJK = H + J - 0.5*K
    print(f"\nF_fromJK - F\n_F = {np.linalg.norm(F_fromJK - F):.6e}")
    K_est = 2.0*(H + J - F)
    print(f"[K_est] \nK_est - K\n_F = {np.linalg.norm(K_est - K):.6e}")
    EJ_est, EK_est = einsum_trace(D, J), einsum_trace(D, K_est)
    print(f"[K_est] Tr[D·J] = {EJ_est:.12f}, Tr[D·K_est] = {EK_est:.12f}, R_KJ_est = {EK_est/EJ_est if EJ_est!=0 else float('nan'):.6f}")
    J_est = F - H + 0.5*K
    print(f"[J_est] \nJ_est - J\n_F = {np.linalg.norm(J_est - J):.6e}")
    EJ_est2 = einsum_trace(D, J_est)
    print(f"[J_est] Tr[D·J_est] = {EJ_est2:.12f}")

    # final hints
    print("\n=== Hints ===")
    if np.allclose(J, K):
        warn("J and K are identical -> very likely K contraction is using the same pattern as J OR exporter saved J into K.")
    print(" - Inspect DFT/scf/jk.py: ensure eri_perm = eri.transpose(0,2,1,3); einsum 'rs,mrns->mn'")
    print(" - In Exporter: ensure K_final is saved from K (not J), and include F_final in NPZ when available.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="HEAVY diagnostics (revised, robust)")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--details", default=None)
    ap.add_argument("--strict", type=int, default=0)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
