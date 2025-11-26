# check_mo.py
import numpy as np
from scipy.linalg import eigh, inv, sqrtm
import sys, json

npz_path = sys.argv[1]  # e.g. DFT/output/.../matrices_H2O.npz
details_path = sys.argv[2]  # e.g. DFT/output/.../details_H2O.json

z = np.load(npz_path)
D = z.get('D') or z.get('D_final')
J = z.get('J') or z.get('J_final')
K = z.get('K') or z.get('K_final')
H = z.get('H_core') if 'H_core' in z else (z.get('T') + z.get('V'))
S = z.get('S') or z.get('overlap')  # adjust key name if necessary

F = H + J - 0.5 * K

# generalized eigenproblem F c = S c eps
# compute S^-1/2 then diagonalize S^-1/2 F S^-1/2
eigSvals, eigSvecs = eigh(S)
if np.any(eigSvals <= 1e-8):
    print("WARNING: small or nonpositive S eigenvalues:", eigSvals[:5])
S_inv_sqrt = eigSvecs @ np.diag(1.0/np.sqrt(eigSvals)) @ eigSvecs.T
F_orth = S_inv_sqrt @ F @ S_inv_sqrt
eps, C_orth = eigh(F_orth)
# transform back to AO coeffs
C = S_inv_sqrt @ C_orth
# sort eps ascending
idx = np.argsort(eps)
eps = eps[idx]
C = C[:, idx]

# occupancy: closed-shell 2 per occupied orbital
nocc = int(round(np.trace(D @ S) / 2.0))
sum_occ_eps = 2.0 * np.sum(eps[:nocc])  # = Tr[D F]
tr_DF = np.einsum('ij,ij->', D, F)

print("nocc =", nocc)
print("sum(2*eps_occ) =", sum_occ_eps)
print("Tr[D F]         =", tr_DF)
print("difference Tr[D F] - sum(2 eps occ) =", tr_DF - sum_occ_eps)

# now check energy relation:
tr_DH = float(np.einsum('ij,ij->', D, H))
Eele_recalc = 0.5 * (tr_DH + tr_DF)  # by def
print("Tr[D H] =", tr_DH)
print("Tr[D F] =", tr_DF)
print("E_elec via 0.5 Tr[D(H+F)] =", Eele_recalc)

# compare with details JSON
with open(details_path,'r',encoding='utf-8') as f:
    det = json.load(f)
Eele_details = det.get('details',{}).get('components', {}).get('E_elec', None) or det.get('E_elec', None)
print("E_elec from details JSON:", Eele_details)
