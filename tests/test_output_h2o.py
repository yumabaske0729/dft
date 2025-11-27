
# tests/test_output_h2o.py
# -*- coding: utf-8 -*-
import numpy as np

def test_matrices_symmetry(matrices_h2o, scf_tools):
    for name in ['S','T','V','H_core','D_final']:
        A = matrices_h2o[name]
        err = scf_tools['symmetry_maxabs']
        assert err == 0.0 or err < 1e-12, f"{name} not symmetric: maxabs={err}"

def test_cts_orthogonality(matrices_h2o, scf_tools):
    S = matrices_h2o['S']
    C = matrices_h2o['C']
    err = scf_tools['orthogonality_error']
    assert err < 1e-10, f"C^T S C deviates from I: Fro={err:.3e}"

def test_density_and_idempotency(matrices_h2o, details_h2o, scf_tools):
    S = matrices_h2o['S']
    D = matrices_h2o['D_final']
    occ = np.array(details_h2o['details']['occupations'], dtype=float)
    Ne_target = float(np.sum(occ))     # 10.0 のはず（RHF閉殻）
    Ne = scf_tools['electron_count']
    idem = scf_tools['idempotency_error']
    assert abs(Ne - Ne_target) < 1e-8, f"electron count mismatch: Tr(D S)={Ne} vs {Ne_target}"
    assert idem < 1e-6, f"idempotency error too large: ||DSD-2D||={idem}"

def test_gap_from_details(details_h2o, matrices_h2o):
    eps = matrices_h2o['eps']
    occ = np.array(details_h2o['details']['occupations'], dtype=float)
    occ_idx = np.where(occ > 1e-8)[0]
    HOMO_i = int(len(occ_idx) - 1)
    LUMO_i = HOMO_i + 1
    HOMO, LUMO = float(eps[HOMO_i]), float(eps[LUMO_i])
    gap_Ha = LUMO - HOMO
    gap_eV = gap_Ha * 27.211386245988
    assert gap_eV > 5.0, f"gap too small? {gap_eV:.2f} eV"
