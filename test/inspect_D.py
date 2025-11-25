import numpy as np, glob, os, json
p = max(glob.glob(os.path.join("DFT","output","*_*_*_RHF","matrices_*.npz")), key=os.path.getmtime)
print("npz:", p)
z = np.load(p, allow_pickle=True)
print("keys:", sorted(z.files))

D = z['D_final']
S = z['S'] if 'S' in z.files else None
C = z['C_final'] if 'C_final' in z.files else None
print("D shape, S shape:", D.shape, None if S is None else S.shape)
print("||D||_F =", np.linalg.norm(D))
print("D max/min =", D.max(), D.min())
if S is not None:
    print("||S||_F =", np.linalg.norm(S))
    print("S diag (min/max) =", np.diag(S).min(), np.diag(S).max())
    try:
        # generalized eigenvalues (robust)
        from scipy.linalg import eigh
        vals = eigh(D, S, eigvals_only=True)
        vals_sorted = np.sort(vals)[::-1]
        print("generalized eigvals (eigh) top 10:", vals_sorted[:10])
    except Exception as e:
        # fallback
        vals = np.linalg.eig(np.linalg.solve(S, D))[0]
        print("generalized eigvals (np solve) top 10:", np.sort(vals)[::-1][:10])
    TrDS = float(np.einsum('mn,mn->', D, S))
    print("Tr[D S] =", TrDS)
print("||D - D.T||_F =", np.linalg.norm(D - D.T))
# If C exists, check orthonormality and reconstruct D
if C is not None:
    C = np.asarray(C)
    print("C shape:", C.shape)
    if S is not None:
        CSTC = C.T @ S @ C
        print("max|C.T S C - I| =", np.max(np.abs(CSTC - np.eye(CSTC.shape[0]))))
    # attempt D_from_C assuming closed-shell D = 2*Cocc*Cocc.T
    # try using first nocc = 5 columns (H2O, 10 electrons -> 5 occ)
    nocc = 5
    Cocc = C[:, :nocc]
    D_from_C = 2.0 * (Cocc @ Cocc.T)
    print("||D - 2*Cocc*Cocc.T||_F =", np.linalg.norm(D - D_from_C))
