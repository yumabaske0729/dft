import numpy as np, glob, os
p = max(glob.glob(os.path.join("DFT","output","*_*_*_RHF","matrices_*.npz")), key=os.path.getmtime)
z = np.load(p, allow_pickle=True)
eps = z['eps']
C = z['C']
nelec = 10
nocc = nelec//2
order = np.argsort(eps)
print("eps (sorted)[:10] =", np.sort(eps)[:10])
print("eps (first columns) 0..4 =", eps[:5])
print("indices of first columns in sorted order:", [np.where(order==i)[0][0] for i in range(nocc)])
# If indices are not 0..(nocc-1), the first columns are not the lowest-energy MOs.
