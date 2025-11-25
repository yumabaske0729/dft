# probe_eri.py
# -*- coding: utf-8 -*-
import os, numpy as np

# --- プロジェクトのモジュールをロード ---
from DFT.input.parser import parse_xyz
from DFT.basis.assign_basis import assign_basis_functions
from DFT.integrals.two_electron import build_eri_tensor

def naive_JK(eri, D):
    """遅いが確実な参照実装（O(n^4)）"""
    n = D.shape[0]
    J = np.zeros((n,n), float)
    K = np.zeros((n,n), float)
    # J_mn = sum_rs D_rs (mn|rs)
    for m in range(n):
        for n_ in range(n):
            s = 0.0
            for r in range(n):
                for s_ in range(n):
                    s += D[r, s_] * eri[m, n_, r, s_]
            J[m, n_] = s
    # K_mn = sum_rs D_rs (mr|ns)
    for m in range(n):
        for n_ in range(n):
            s = 0.0
            for r in range(n):
                for s_ in range(n):
                    s += D[r, s_] * eri[m, r, n_, s_]
            K[m, n_] = s
    return J, K

def main():
    # 1) 入力と基底（H2O/6-31G）をロード
    xyz = os.path.join("DFT", "H2O.xyz")
    mol = parse_xyz(xyz, charge=0, multiplicity=1)
    basis = assign_basis_functions(mol, "6-31g")

    print(f"[probe] nbf = {len(basis)} basis functions")

    # 2) ERIテンソル（mnrs）を構築
    eri = build_eri_tensor(basis)  # eri[m,n,r,s] = (mn|rs)
    eri = np.asarray(eri, float, order="C")

    # 3) ERIの「第2軸↔第3軸」対称性の検査
    diff_std = np.linalg.norm(eri - eri.transpose(0,2,1,3))
    diff_alt1 = np.linalg.norm(eri - eri.transpose(0,3,2,1))
    diff_alt2 = np.linalg.norm(eri - eri.transpose(0,1,3,2))
    fn_eri = np.linalg.norm(eri)

    print(f"[probe] ||eri||_F              = {fn_eri:.12e}")
    print(f"[probe] ||eri - perm(0,2,1,3)||= {diff_std:.12e}  (STD: (mr|ns))")
    print(f"[probe] ||eri - perm(0,3,2,1)||= {diff_alt1:.12e} (ALT1:(ms|rn))")
    print(f"[probe] ||eri - perm(0,1,3,2)||= {diff_alt2:.12e} (ALT2:(mn|sr))")

    # 4) ランダム対称Dで参照J/Kを作る（ERI→J/Kが本当に区別できるか）
    rng = np.random.default_rng(1234)
    n = eri.shape[0]
    A = rng.normal(size=(n,n))
    D = 0.5*(A + A.T)  # 対称化（RHFのスピン和Dの形に寄せる）
    J_ref, K_ref = naive_JK(eri, D)
    dJK = np.linalg.norm(J_ref - K_ref)
    EJ = float(np.einsum("mn,mn->", D, J_ref))
    EK = float(np.einsum("mn,mn->", D, K_ref))

    print(f"[probe] ||J_ref - K_ref||_F     = {dJK:.12e}")
    print(f"[probe] ΣD·J_ref = {EJ:.12f} , ΣD·K_ref = {EK:.12f} , R = {EK/EJ if EJ!=0 else np.nan:.6f}")

if __name__ == "__main__":
    main()