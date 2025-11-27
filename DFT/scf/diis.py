#diis.py
from __future__ import annotations
import numpy as np
from typing import List


def diis_extrapolate(
    fock_list: List[np.ndarray],
    error_list: List[np.ndarray],
    regularization: float = 1e-12,
    max_condition: float = 1e12,
) -> np.ndarray:
    """
    Pulay DIIS extrapolation of the Fock matrix.

    Args:
        fock_list: List of previous Fock matrices, each (nbf, nbf).
        error_list: Corresponding commutator error matrices (FDS - SDF), each (nbf, nbf).
        regularization: Small diagonal shift added to B for numerical stability.
        max_condition: If the condition number of B is larger than this, fall back.

    Returns:
        Extrapolated Fock matrix with the same shape/dtype as the last Fock in fock_list.

    Notes:
        - Requires at least 2 stored iterations; otherwise returns the last Fock.
        - The (n+1)x(n+1) DIIS system is:
              [ B   -1 ] [ c ]   [ 0 ]
              [ -1^T 0 ] [ λ ] = [ -1 ]
          with B_ij = <e_i | e_j> (Frobenius inner product), c are coefficients.
    """
    n = len(fock_list)
    if n < 2 or n != len(error_list):
        return fock_list[-1]

    # Build B matrix
    B = np.empty((n + 1, n + 1), dtype=float)
    B[:-1, :-1] = 0.0
    for i in range(n):
        ei = error_list[i]
        for j in range(i, n):
            ej = error_list[j]
            vij = float(np.vdot(ei, ej))
            B[i, j] = vij
            B[j, i] = vij
        B[i, i] += regularization  # regularize diagonal

    B[:-1, -1] = -1.0
    B[-1, :-1] = -1.0
    B[-1, -1] = 0.0

    rhs = np.zeros(n + 1, dtype=float)
    rhs[-1] = -1.0

    # Safety: condition number check
    try:
        cond = np.linalg.cond(B)
        if not np.isfinite(cond) or cond > max_condition:
            return fock_list[-1]
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return fock_list[-1]

    coeffs = sol[:-1]  # last is Lagrange multiplier

    # Extrapolate Fock
    F_extrap = np.zeros_like(fock_list[0], dtype=fock_list[0].dtype)
    for c, F in zip(coeffs, fock_list):
        F_extrap += c * F
    # Ensure Hermiticity (symmetrize)
    F_extrap = 0.5 * (F_extrap + F_extrap.T)

    return F_extrap
