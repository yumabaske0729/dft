# DFT/scf/rhf.py — RHF (closed-shell; D=2P)
from __future__ import annotations
import os
import time
import numpy as np
from scipy.linalg import eigh
from ..integrals.two_electron import build_eri_tensor
from .diis import diis_extrapolate
from ..utils.constants import get_atomic_number
from .jk import form_jk_matrices  # JK formation is centralized here

def compute_nuclear_repulsion(atoms) -> float:
    E_nuc = 0.0
    for i in range(len(atoms)):
        Zi = get_atomic_number(atoms[i].symbol)
        Ri = atoms[i].position
        for j in range(i + 1, len(atoms)):
            Zj = get_atomic_number(atoms[j].symbol)
            Rj = atoms[j].position
            R = np.linalg.norm(Ri - Rj)
            E_nuc += Zi * Zj / R
    return E_nuc

def _jk_energy_sums(D: np.ndarray, J: np.ndarray, K: np.ndarray) -> tuple[float, float, float]:
    EJ = float(np.einsum('mn,mn->', D, J))
    EK = float(np.einsum('mn,mn->', D, K))
    R = (EK / EJ) if EJ != 0.0 else np.nan
    return EJ, EK, R

def run_rhf(
    S: np.ndarray,
    T: np.ndarray,
    V: np.ndarray,
    basis_functions,
    mol,
    max_iter: int = 60,
    conv_thresh: float = 1e-9,
    rms_thresh: float = 1e-6,
    eri_prebuilt: np.ndarray | None = None,
):
    """Restricted Hartree–Fock (closed shell). Uses spin-summed D = 2P."""
    t0 = time.perf_counter()
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    nbf = S.shape[0]
    H_core = T + V

    # Electron count
    n_elec = sum(get_atomic_number(atom.symbol) for atom in mol.atoms) - mol.charge
    if (n_elec % 2) == 1:
        raise NotImplementedError(
            f"RHF requires even electrons. Detected odd count: {n_elec}. Use UHF/ROHF."
        )
    nocc = n_elec // 2

    # Orthogonalizer S^(-1/2)
    S_sym = 0.5 * (S + S.T)
    evals_S, evecs_S = eigh(S_sym)
    thresh = 1e-7
    inv_sqrt = np.where(evals_S > thresh, 1.0 / np.sqrt(evals_S), 0.0)
    S_half = evecs_S @ np.diag(inv_sqrt) @ evecs_S.T
    print(f"[orth] min(eval_S)={evals_S.min():.3e}, small={(evals_S <= thresh).sum()}, neg={(evals_S < -1e-12).sum()}")

    # ERI
    t_eri0 = time.perf_counter()
    eri = build_eri_tensor(basis_functions) if eri_prebuilt is None else eri_prebuilt
    t_eri1 = time.perf_counter()

    def build_density(C: np.ndarray) -> np.ndarray:
        D = np.zeros((nbf, nbf), dtype=np.float64)
        for m in range(nocc):
            D += 2.0 * np.outer(C[:, m], C[:, m])
        return D

    # Initial guess
    F_guess = H_core.copy()
    Fp = S_half.T @ F_guess @ S_half
    eps, Cp = eigh(Fp)
    order = np.argsort(eps)
    eps = eps[order]
    Cp = Cp[:, order]
    C = S_half @ Cp
    D = build_density(C)

    E_old = None
    fock_list, error_list = [], []
    MAX_DIIS = 8
    E_nuc = compute_nuclear_repulsion(mol.atoms)
    scf_hist = []

    # SCF loop
    for it in range(1, max_iter + 1):
        # Build Fock (DIIS or direct)
        if len(fock_list) >= 2:
            F_d = diis_extrapolate(fock_list, error_list)
        else:
            J_d, K_d = form_jk_matrices(eri, D)
            F_d = H_core + J_d - 0.5 * K_d
            F_d = 0.5 * (F_d + F_d.T)

        Fp = S_half.T @ F_d @ S_half
        eps, Cp = eigh(Fp)
        order = np.argsort(eps)
        eps = eps[order]
        Cp = Cp[:, order]
        C = S_half @ Cp
        D_new = build_density(C)

        # Consistent energy with new density
        Jn, Kn = form_jk_matrices(eri, D_new)
        EJ_sum, EK_sum, R_KJ = _jk_energy_sums(D_new, Jn, Kn)
        diff_norm_JK = np.linalg.norm(Jn - Kn)
        print(f"[rhf] ΣD·J={EJ_sum:.12f}, ΣD·K={EK_sum:.12f}, R_KJ={R_KJ:.6f}, ||J-K||_F={diff_norm_JK:.3e}")

        # Relaxed J==K guard
        if diff_norm_JK < 1e-8:
            print(f"[rhf] WARNING: J and K nearly identical at iter {it}, continuing SCF.")

        F_cons = H_core + Jn - 0.5 * Kn
        F_cons = 0.5 * (F_cons + F_cons.T)

        # Energy components
        E_Hcore = float(np.einsum('mn,mn->', D_new, H_core))
        E_2e = float(0.5 * np.einsum('mn,mn->', D_new, (Jn - 0.5 * Kn)))
        E_elec = E_Hcore + E_2e
        E_scf = E_elec + E_nuc
        Ne = float(np.einsum('mn,mn->', D_new, S))

        delta_E = np.inf if E_old is None else abs(E_scf - E_old)
        rms_D = float(np.sqrt(np.mean((D_new - D) ** 2)))
        err = F_cons @ D_new @ S - S @ D_new @ F_cons
        err_norm = float(np.linalg.norm(err))

        print(f"SCF Iter {it:2d}: E = {E_scf:.12f} Ha, ΔE = {delta_E:.3e}, RMS(D) = {rms_D:.3e}, Tr[DS] = {Ne:.8f}, err_F = {err_norm:.3e}")

        scf_hist.append({
            "iter": it, "E_total": E_scf, "dE": delta_E, "RMS_D": rms_D,
            "sum_DJ": EJ_sum, "sum_DK": EK_sum, "R_KJ": R_KJ, "diff_norm_JK": diff_norm_JK
        })

        if (E_old is not None and delta_E < conv_thresh) and (rms_D < rms_thresh):
            print(f"SCF converged in {it} cycles. Final Energy = {E_scf:.12f} Ha")
            D = D_new
            break

        fock_list.append(F_cons.copy()); error_list.append(err.copy())
        if len(fock_list) > MAX_DIIS:
            fock_list.pop(0); error_list.pop(0)

        D = D_new
        E_old = E_scf
    else:
        print(f"Warning: SCF did not converge in {max_iter} iterations; final ΔE = {delta_E:.3e}, RMS(D) = {rms_D:.3e}")

    return E_scf, C, eps, {
        "scf_history": scf_hist,
        "final_energy": E_scf,
        "mo_energies": eps.tolist(),
        "occupations": [2.0 if i < nocc else 0.0 for i in range(nbf)]
    }