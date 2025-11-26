# DFT/scf/dft.py
# -*- coding: utf-8 -*-
# Restricted KS-DFT (B3LYP-lite). JK is delegated to jk.py.
from __future__ import annotations
import os
import time
import numpy as np
from scipy.linalg import eigh
from .diis import diis_extrapolate
from .jk import form_jk_matrices   # JK builder（物理指標選択版）
from .rhf import compute_nuclear_repulsion, run_rhf
from .grid import generate_integration_grid
from .functional import B3LYP
from ..utils.constants import get_atomic_number

def _get_int_env(name: str, default: int | None = None) -> int | None:
    val = os.getenv(name, None)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default

def _get_float_env(name: str, default: float | None = None) -> float | None:
    val = os.getenv(name, None)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

def _jk_energy_sums(D: np.ndarray, J: np.ndarray, K: np.ndarray) -> tuple[float, float, float]:
    EJ = float(np.einsum('mn,mn->', D, J))
    EK = float(np.einsum('mn,mn->', D, K))
    R = (EK / EJ) if EJ != 0.0 else np.nan
    return EJ, EK, R

def run_dft(
    S,
    T,
    V,
    basis_functions,
    mol,
    *,
    grid_level: int = 3,
    max_iter: int = 60,
    conv_thresh: float = 1e-8,
    # streaming XC options
    xc_chunk_size: int | None = None,
    xc_d_thresh: float = 0.0,
    xc_validate: int = 1,
):
    """Restricted KS-DFT (hybrid: B3LYP-lite)."""
    # env overrides
    gl = _get_int_env("DFT_GRID_LEVEL", None)
    if gl and gl > 0:
        grid_level = gl
        print(f"[dft] grid_level overridden by env: DFT_GRID_LEVEL={grid_level}")
    mi = _get_int_env("DFT_MAX_ITER", None)
    if mi and mi > 0:
        max_iter = mi
        print(f"[dft] max_iter overridden by env: DFT_MAX_ITER={max_iter}")
    ct = _get_float_env("DFT_CONV_THRESH", None)
    if ct and ct > 0.0:
        conv_thresh = ct
        print(f"[dft] conv_thresh overridden by env: DFT_CONV_THRESH={conv_thresh:g}")

    # XC mixing override (optional)
    a0_env = _get_float_env("DFT_A0", None)
    aX_env = _get_float_env("DFT_AX", None)
    aC_env = _get_float_env("DFT_AC", None)
    override_xc = (a0_env is not None) or (aX_env is not None) or (aC_env is not None)

    # streaming options
    chunk_sz = None if (xc_chunk_size is None or int(xc_chunk_size) == 0) else int(xc_chunk_size)
    d_thresh = float(xc_d_thresh)
    vldt = bool(int(xc_validate))
    if chunk_sz:
        print(f"[dft] XC streaming enabled: chunk_size={chunk_sz}, d_thresh={d_thresh}, validate={int(vldt)}")

    t0 = time.perf_counter()
    # arrays
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    nbf = int(S.shape[0])
    H_core = T + V

    # S^{-1/2}
    evalS, U = eigh(S)
    thresh = 1e-8
    inv_sqrt = np.where(evalS > thresh, evalS ** -0.5, 0.0)
    S_half = U @ np.diag(inv_sqrt) @ U.T
    min_pos = float(np.min(evalS[evalS > thresh])) if np.any(evalS > thresh) else np.inf
    max_ev  = float(np.max(evalS)) if evalS.size else np.inf
    S_cond_est = (max_ev / min_pos) if np.isfinite(min_pos) and (min_pos > 0) else np.inf

    # ERI & grid
    from ..integrals.two_electron import build_eri_tensor
    t_eri0 = time.perf_counter()
    eri = build_eri_tensor(basis_functions)
    t_eri1 = time.perf_counter()
    grid = generate_integration_grid(basis_functions, level=grid_level)
    n_points = int(grid.points.shape[0])
    weights_sum = float(np.sum(grid.weights))

    # RHF warm start
    E_rhf, C0, eps0, _ = run_rhf(S, T, V, basis_functions, mol)
    n_electrons = sum(get_atomic_number(a.symbol) for a in mol.atoms) - mol.charge
    nocc = int(n_electrons // 2)

    def density_from_C(C: np.ndarray) -> np.ndarray:
        D = np.zeros((nbf, nbf), dtype=np.float64)
        for m in range(nocc):
            D += 2.0 * np.outer(C[:, m], C[:, m])  # spin-summed D = 2P
        return D

    # initial density
    D = density_from_C(C0)
    E_old = None
    fock_list: list[np.ndarray] = []
    error_list: list[np.ndarray] = []
    scf_hist: list[dict] = []

    # initial B3LYP object (based on D)
    b3 = B3LYP(grid, basis_functions, D, chunk_size=chunk_sz, d_thresh=d_thresh, validate_streaming=vldt)
    if override_xc:
        if a0_env is not None: b3.a0 = float(a0_env)
        if aX_env is not None: b3.aX = float(aX_env)
        if aC_env is not None: b3.aC = float(aC_env)
    a0 = b3.a0  # used in -a0 * K

    # ============================
    # SCF loop
    # ============================
    for iteration in range(1, max_iter + 1):
        # JK at current D  — α = a0（DFT）
        J, K = form_jk_matrices(eri, D, H_core=H_core, a0=a0)

        # XC at current D
        b3 = B3LYP(grid, basis_functions, D, chunk_size=chunk_sz, d_thresh=d_thresh, validate_streaming=vldt)
        if override_xc:
            if a0_env is not None: b3.a0 = float(a0_env)
            if aX_env is not None: b3.aX = float(aX_env)
            if aC_env is not None: b3.aC = float(aC_env)
        Exc, Vxc = b3.evaluate_exchange_correlation()

        # Fock
        F = H_core + J - a0 * K + Vxc
        F = 0.5 * (F + F.T)

        # commutator error
        err = F @ D @ S - S @ D @ F
        err_norm = float(np.linalg.norm(err))

        # DIIS buffers
        fock_list.append(F.copy())
        error_list.append(err.copy())
        if len(fock_list) > 8:
            fock_list.pop(0)
            error_list.pop(0)

        # DIIS use
        F_use = diis_extrapolate(fock_list, error_list) if iteration >= 2 else F

        # diagonalize in orthogonal metric
        Fp = S_half.T @ F_use @ S_half
        eps, Cp = eigh(Fp)
        C = S_half @ Cp

        # new density
        D_new = density_from_C(C)

        # --- energy with D_new (consistent accounting) ---
        Jn, Kn = form_jk_matrices(eri, D_new, H_core=H_core, a0=a0)

        EJ_sum, EK_sum, R_KJ = _jk_energy_sums(D_new, Jn, Kn)
        print(f"[dft] ΣD·J={EJ_sum:.12f}, ΣD·K={EK_sum:.12f}, R_KJ={R_KJ:.6f}")
        if os.getenv("DFT_JK_GUARD", "1") != "0":
            if np.allclose(Jn, Kn, rtol=1e-9, atol=1e-12) and np.linalg.norm(Jn) > 1e-10:
                raise RuntimeError("[dft] Detected J==K at DFT step; check JK builder and ERI permutation.")

        b3_energy = B3LYP(grid, basis_functions, D_new, chunk_size=chunk_sz, d_thresh=d_thresh, validate_streaming=vldt)
        if override_xc:
            if a0_env is not None: b3_energy.a0 = float(a0_env)
            if aX_env is not None: b3_energy.aX = float(aX_env)
            if aC_env is not None: b3_energy.aC = float(aC_env)

        Exc_new, _Vxc_dummy = b3_energy.evaluate_exchange_correlation()

        E_one = float(np.einsum('mn,mn->', D_new, H_core))        
        E_jk = float(0.5 * np.einsum('mn,mn->', D_new, Jn) - 0.25 * a0 * np.einsum('mn,mn->', D_new, Kn))        
        E_elec = E_one + E_jk + float(Exc_new)
        E_nuc  = compute_nuclear_repulsion(mol.atoms)
        E_total = E_elec + E_nuc

        dE   = np.inf if E_old is None else abs(E_total - E_old)
        rmsD = float(np.sqrt(np.mean((D_new - D) ** 2)))
        TrDS = float(np.einsum('mn,mn->', D_new, S))

        print(f"DFT Iter {iteration}: E = {E_total:.12f} Ha, ΔE = {dE:.2e}")

        scf_hist.append({
            "iter": iteration, "E_total": E_total, "dE": dE if np.isfinite(dE) else "",
            "RMS_D": rmsD, "TrDS": TrDS, "E_Hcore": E_one, "E_JK": E_jk,
            "Exc": float(Exc_new), "err_norm": err_norm,
            "sum_DJ": EJ_sum, "sum_DK": EK_sum, "R_KJ": R_KJ,
            "converged": (dE < conv_thresh),
        })

        if dE < conv_thresh:
            print(f"DFT SCF converged in {iteration} cycles. Final Energy = {E_total:.12f} Ha")
            # finalize on D_new
            D = D_new
            # Vxc_final for report consistency（収束時も D_new）
            b3_final = B3LYP(grid, basis_functions, D, chunk_size=chunk_sz, d_thresh=d_thresh, validate_streaming=vldt)
            if override_xc:
                if a0_env is not None: b3_final.a0 = float(a0_env)
                if aX_env is not None: b3_final.aX = float(aX_env)
                if aC_env is not None: b3_final.aC = float(aC_env)
            _Exc_fin, Vxc_final = b3_final.evaluate_exchange_correlation()
            F_final = H_core + Jn - a0 * Kn + Vxc_final
            break

        # 非収束でも最新密度で
        D, E_old = D_new, E_total
        b3_last = B3LYP(grid, basis_functions, D_new, chunk_size=chunk_sz, d_thresh=d_thresh, validate_streaming=vldt)
        if override_xc:
            if a0_env is not None: b3_last.a0 = float(a0_env)
            if aX_env is not None: b3_last.aX = float(aX_env)
            if aC_env is not None: b3_last.aC = float(aC_env)
        _Exc_last, Vxc_last = b3_last.evaluate_exchange_correlation()
        F_final = H_core + Jn - a0 * Kn + Vxc_last
        Vxc_final = Vxc_last
    else:
        print(f"Warning: DFT SCF did not converge in {max_iter} iterations; final ΔE = {dE:.2e}")

    t1 = time.perf_counter()
    mo_energies = [float(x) for x in np.asarray(eps, dtype=float).tolist()]
    mo_occ = [2.0 if i < nocc else 0.0 for i in range(nbf)]

    # --- diagnostics へ vne_scale を記録（記録パッチ） ---
    vne_scale = float(os.getenv("VNE_SCALE", "1.0"))

    details = {
        "components": {
            "E_Hcore": float(np.einsum('mn,mn->', D, H_core)),
            "E_JK": float(0.5 * np.einsum('mn,mn->', D, (Jn - a0 * Kn))),
            "Exc": float(Exc_new),
            "E_elec": float(np.einsum('mn,mn->', D, H_core)) + float(0.5 * np.einsum('mn,mn->', D, (Jn - a0 * Kn))) + float(Exc_new),
            "E_nuc": float(E_nuc),
            "E_total": float(E_total),
            "E_RHF_warmstart": float(E_rhf),
        },
        "diagnostics": {
            "S_eval_min": float(np.min(evalS)) if evalS.size else None,
            "S_eval_max": float(np.max(evalS)) if evalS.size else None,
            "S_cond_est": float(S_cond_est) if np.isfinite(S_cond_est) else None,
            "trace_DS": float(TrDS),
            "eri_build_sec": float(t_eri1 - t_eri0),
            "total_scf_sec": float(t1 - t0),
            "S_norm_F": float(np.linalg.norm(S)),
            "T_norm_F": float(np.linalg.norm(T)),
            "V_norm_F": float(np.linalg.norm(V)),
            "sum_DJ": EJ_sum, "sum_DK": EK_sum, "R_KJ": R_KJ,
            "vne_scale": vne_scale,                  # ← ★ 追加
        },
        "counts": { "n_basis": int(nbf), "n_electrons": int(n_electrons), "n_occ_closed_shell": int(nocc) },
        "grid_info": {
            "level": int(grid_level),
            "lebedev_degree": int(grid.meta.get("lebedev_degree", grid.meta.get("lebedev_order", 6))),
            "lebedev_order": int(grid.meta.get("lebedev_order", 0)),
            "n_radial": int(grid.meta.get("n_radial", 0)),
            "Rmax": float(grid.meta.get("Rmax", 0.0)),
            "n_points": int(n_points),
            "weights_sum": float(weights_sum),
        },
        "xc_params": {"a0": float(b3.a0), "aX": float(b3.aX), "aC": float(b3.aC)},
        "scf_history": scf_hist,
        "matrices": {
            "F_final": F_final, "J_final": Jn, "K_final": Kn,
            "Vxc_final": Vxc_final, "S_half": S_half, "S_evals": evalS,
        },
        "mo": {"energies_hartree": mo_energies, "occupations": mo_occ},
    }
    return float(E_total), C, eps, details
