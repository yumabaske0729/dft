# -*- coding: utf-8 -*-
"""
full_diagnostics.py
DFT 出力 (DFT/output/*) を総合点検するワンファイル診断ツール。

使い方:
  python full_diagnostics.py

何をやるか:
 - matrices_*.npz と details_*.json を自動検出して複数検査を実行
 - S, S_half, C, eps, D_final, J_final, K_final, H_core/T/V をチェック
 - Fock 恒等式、D の冪等性、MO 正規化、J/K の整合性、ERI 単体テスト など
 - PySCF が利用可能なら PySCF ベンチも自動で実行（任意）
"""
from __future__ import annotations
import os, glob, json, numpy as np, sys
from math import isclose
from typing import Optional

# optional dependencies
try:
    import scipy.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from DFT.integrals.two_electron import primitive_eri, _E1d_os
    from DFT.integrals.one_electron.boys import boys_function
    _HAS_INT_IMPL = True
except Exception:
    _HAS_INT_IMPL = False

# optional PySCF
try:
    from pyscf import gto, scf
    from pyscf.scf import hf
    _HAS_PYSCF = True
except Exception:
    _HAS_PYSCF = False

# helper printing
def ok(s): print("[ OK ]", s)
def warn(s): print("[WARN]", s)
def err(s): print("[ERR ]", s)
def info(s): print("[INFO]", s)

# find latest files
def pick_latest(patterns):
    cands=[]
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)

def pick_npz():
    return pick_latest([os.path.join("DFT","output","*_*_*_RHF","matrices_*.npz"),
                        os.path.join("output","*_*_*_RHF","matrices_*.npz"),
                        os.path.join("*_*_*_RHF","matrices_*.npz")])

def pick_details():
    return pick_latest([os.path.join("DFT","output","*_*_*_RHF","details_*.json"),
                        os.path.join("output","*_*_*_RHF","details_*.json"),
                        os.path.join("*_*_*_RHF","details_*.json")])

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z

def general_eigh(A, B=None):
    if _HAS_SCIPY:
        if B is None:
            w = spla.eigvalsh(A)
        else:
            w = spla.eigvalsh(A, b=B)
        return np.sort(w)[::-1]
    else:
        if B is None:
            w = np.linalg.eigvalsh(A)
            return np.sort(w)[::-1]
        else:
            # fallback: solve inv(B)A x = w x  (may be unstable)
            w = np.linalg.eigvals(np.linalg.solve(B, A))
            w = np.real_if_close(w)
            return np.sort(w)[::-1]

# === main diagnostics ===
def run_all():
    npz = pick_npz()
    details = pick_details()
    if npz is None:
        err("matrices_*.npz が見つかりません。DFT/output を確認してください。")
        return
    info(f"Using npz: {npz}")
    z = load_npz(npz)
    keys = sorted(z.files)
    info("npz keys: " + ", ".join(keys))

    # required arrays
    C = z['C'] if 'C' in keys else (z['C_final'] if 'C_final' in keys else None)
    eps = z['eps'] if 'eps' in keys else None
    D = z['D_final'] if 'D_final' in keys else (z['D'] if 'D' in keys else None)
    J = z['J_final'] if 'J_final' in keys else None
    K = z['K_final'] if 'K_final' in keys else None
    F = z['F_final'] if 'F_final' in keys else None
    H = z['H_core'] if 'H_core' in keys else None
    T = z['T'] if 'T' in keys else None
    V = z['V'] if 'V' in keys else None
    S = z['S'] if 'S' in keys else None
    S_half = z['S_half'] if 'S_half' in keys else None

    # details.json
    details_data = None
    if details:
        info(f"Using details: {details}")
        try:
            with open(details,'r',encoding='utf-8') as f:
                details_data = json.load(f)
        except Exception as e:
            warn("details json 読み込み失敗: " + str(e))

    # 1) basic shapes
    info("=== basic shapes and presence ===")
    def shape_report(name, arr):
        if arr is None:
            warn(f"{name}: MISSING")
        else:
            ok(f"{name}: present shape={np.asarray(arr).shape}")
    shape_report("C", C)
    shape_report("eps", eps)
    shape_report("D", D)
    shape_report("S", S)
    shape_report("S_half", S_half)
    shape_report("J", J)
    shape_report("K", K)
    shape_report("H_core", H)
    shape_report("T", T)
    shape_report("V", V)
    shape_report("F", F)

    # 2) S checks
    if S is not None:
        info("=== Overlap S checks ===")
        s = np.asarray(S)
        diag = np.diag(s)
        ok(f"S diag min/max = {diag.min():.6e} / {diag.max():.6e}")
        cond = np.linalg.cond(s)
        info(f"S cond number = {cond:.6e}")
        if not np.all(diag > 0):
            warn("S diag に非正の値がある")
        if S_half is not None:
            # check S_half @ S_half ≈ S
            diff = np.linalg.norm(S_half.dot(S_half) - s)
            ok(f"||S_half^2 - S||_F = {diff:.6e}")
        # check eigenvalues
        sevals = general_eigh(s)
        info("S eigenvalues (top 6): " + ", ".join(f"{v:.6e}" for v in sevals[:6]))

    # 3) C normalization
    if C is not None:
        info("=== MO (C) normalization checks ===")
        C = np.asarray(C)
        if S is not None:
            CTSC = C.T.dot(S).dot(C)
            maxdev = np.max(np.abs(CTSC - np.eye(CTSC.shape[0])))
            ok(f"max|C.T S C - I| = {maxdev:.6e}")
        CTC = C.T.dot(C)
        maxdev2 = np.max(np.abs(CTC - np.eye(CTC.shape[0])))
        info(f"max|C.T C - I| = {maxdev2:.6e}  (if small, C may be orthonormal already)")

    # 4) D checks
    if D is not None:
        info("=== Density D checks ===")
        D = np.asarray(D)
        sym = np.linalg.norm(D - D.T)
        ok(f"||D - D.T||_F = {sym:.6e}")
        if S is not None:
            TrDS = float(np.einsum('mn,mn->', D, S))
            info(f"Tr[D S] = {TrDS:.12f} (should equal #electrons or #electrons/?)")
            # idempotency: D S D ≈ D
            idemp = np.linalg.norm(D.dot(S).dot(D) - D)
            info(f"||D S D - D||_F = {idemp:.6e} (small is good)")
            # natural occupations
            try:
                nevals = general_eigh(D, S)
                info("natural occ (top 10): " + ", ".join(f"{v:.6e}" for v in nevals[:10]))
            except Exception as e:
                warn("generalized eigvals fail: " + str(e))
        # compare to 2*Cocc*Cocc.T if C exists
        if C is not None and eps is not None:
            nelec = None
            if details_data:
                nelec = details_data.get("meta",{}).get("nelec") or details_data.get("meta",{}).get("Nelec")
            if nelec is None:
                # guess from trace: try even integer nearest
                if S is not None:
                    TrDS_val = float(np.einsum('mn,mn->', D, S))
                    nelec = int(round(TrDS_val))
                    info(f"guessed Nelec = {nelec} from Tr[D S]")
                else:
                    warn("Nelec 不明 (details 未取得, S も missing)")
            if nelec is not None:
                nocc = int(nelec // 2)
                # assume C columns are ordered; try constructing
                Cocc = C[:, :nocc]
                D_from_C = 2.0 * (Cocc @ Cocc.T)
                diff = np.linalg.norm(D - D_from_C)
                info(f"||D - 2*C[:,:{nocc}]C[:,:{nocc}]^T||_F = {diff:.6e}")

    # 5) Fock eigencheck: F C ≈ S C eps
    if F is not None and C is not None and eps is not None and S is not None:
        info("=== Fock eigenproblem check ===")
        F = np.asarray(F)
        C = np.asarray(C)
        eps = np.asarray(eps)
        nbf = C.shape[0]
        # compute residual R = F C - S C eps
        # form S C eps by columns
        SCe = S.dot(C) * eps[np.newaxis,:]
        FC = F.dot(C)
        R = FC - SCe
        res_norm = np.linalg.norm(R)
        info(f"||F C - S C eps||_F = {res_norm:.6e} (small is good)")

    # 6) energy components and Fock identity
    info("=== energy and Fock identity ===")
    if D is not None:
        if H is None and T is not None and V is not None:
            H = T + V
        if H is not None:
            EJ = float(np.einsum('mn,mn->', D, J)) if J is not None else None
            EK = float(np.einsum('mn,mn->', D, K)) if K is not None else None
            EH = float(np.einsum('mn,mn->', D, H))
            E2e = 0.5 * (EJ - 0.5 * EK) if (EJ is not None and EK is not None) else None
            Eele_JK = EH + E2e if E2e is not None else None
            Eele_Fock = 0.5 * float(np.einsum('mn,mn->', D, H + F)) if F is not None else None
            info(f"Tr[D H_core] = {EH:.12f}" if EH is not None else "H missing")
            info(f"Tr[D J] = {EJ:.12f}" if EJ is not None else "J missing")
            info(f"Tr[D K] = {EK:.12f}" if EK is not None else "K missing")
            if E2e is not None:
                info(f"E_2e(recalc) = {E2e:.12f}")
            if Eele_JK is not None and Eele_Fock is not None:
                diff = Eele_JK - Eele_Fock
                ok(f"ΔE_elec (J/K - Fock) = {diff:+.6e} Ha (should be ~0)")
        else:
            warn("H_core 未定。T/V を npz に含めるか H_core を保存してください。")

    # 7) J/K vs PySCF (optional)
    if _HAS_PYSCF:
        info("=== PySCF benchmark of J/K (optional) ===")
        try:
            # need geometry and basis from details
            if details_data is None:
                warn("details json missing -> skip PySCF bench")
            else:
                geo = details_data.get("meta",{}).get("geometry") or details_data.get("geometry")
                if geo is None:
                    warn("geometry not found in details -> skip PySCF bench")
                else:
                    atom_lines = []
                    if isinstance(geo, str):
                        atom_lines = geo.strip().splitlines()
                    else:
                        for item in geo:
                            if isinstance(item, (list,tuple)) and len(item)>=4:
                                atom_lines.append(f"{item[0]} {item[1]} {item[2]} {item[3]}")
                            elif isinstance(item, dict) and 'element' in item:
                                atom_lines.append(f"{item['element']} {item['x']} {item['y']} {item['z']}")
                    atom_str = "; ".join(atom_lines)
                    basis = details_data.get("meta",{}).get("basis","6-31G")
                    mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', charge=details_data.get("meta",{}).get("charge",0))
                    dm = D.copy()
                    vj, vk = hf.get_jk(mol, dm)
                    TrDJ = float(np.einsum('mn,mn->', D, vj))
                    TrDK = float(np.einsum('mn,mn->', D, vk))
                    info(f"PySCF Tr[D·J] = {TrDJ:.12f}, Tr[D·K] = {TrDK:.12f}")
                    if J is not None and K is not None:
                        myJ = float(np.einsum('mn,mn->', D, J))
                        myK = float(np.einsum('mn,mn->', D, K))
                        info(f"Your Tr[D·J] = {myJ:.12f}, Tr[D·K] = {myK:.12f}")
                        info(f"Ratios: J(yours)/J(pyscf) = {myJ/TrDJ:.6f}, K(yours)/K(pyscf) = {myK/TrDK:.6f}")
        except Exception as e:
            warn("PySCF bench failed: " + str(e))

    # 8) ERI symmetries (if stored)
    if 'eri' in keys:
        info("=== ERI symmetry checks ===")
        eri = z['eri']  # assume shape (nbf,nbf,nbf,nbf) or (nbf*(nbf+1)/2,...)
        eri = np.asarray(eri)
        # only proceed if 4D
        if eri.ndim == 4:
            n = eri.shape[0]
            # sample random indices and test permutations
            import random
            for _ in range(20):
                p,q,r,s = [random.randrange(n) for _ in range(4)]
                v = eri[p,q,r,s]
                ok_sym = True
                if not np.allclose(v, eri[q,p,r,s]):
                    warn(f"eri[p,q,r,s] != eri[q,p,r,s] for {p,q,r,s}")
                    ok_sym = False
                if not np.allclose(v, eri[p,q,s,r]):
                    warn(f"eri[p,q,r,s] != eri[p,q,s,r] for {p,q,r,s}")
                    ok_sym = False
                if not np.allclose(v, eri[r,s,p,q]):
                    warn(f"eri[p,q,r,s] != eri[r,s,p,q] for {p,q,r,s}")
                    ok_sym = False
            info("ERI symmetry spot checks done.")
        else:
            warn("ERI は 4D 配列ではありません。変換形式を確認してください。")

    # 9) primitive ERI sanity (if integrals module available)
    if _HAS_INT_IMPL:
        info("=== primitive ERI comparison (OS-R vs Boys-old) ===")
        try:
            # use the same positions as check2 defaults
            A = np.array([0.00, 0.00, 0.00])
            B = np.array([0.20, 0.00, 0.00])
            Cpos = np.array([0.15, 0.10, -0.05])
            Dpos = np.array([0.35, -0.10, 0.10])
            a = b = c = d = 1.0
            val_R = primitive_eri(1,0,0,a,A, 0,0,0,b,B, 1,0,0,c,Cpos, 0,0,0,d,Dpos)
            # reuse small function from check2: primitive_old_boys_total_order
            def primitive_old_boys_total_order_wrapper():
                p = a+b; q = c+d
                P = (a*A + b*B)/p
                Q = (c*Cpos + d*Dpos)/q
                from math import pi, sqrt
                pre = 2.0*(pi**2.5)/(p*q*sqrt(p+q))
                pre *= np.exp(-(a*b/p)*np.dot(A-B,A-B) - (c*d/q)*np.dot(Cpos-Dpos,Cpos-Dpos))
                T = (p*q/(p+q))*np.dot(P-Q,P-Q)
                ExAB=_E1d_os(1,0,a,b,A[0],B[0])[1,0,:1+0+1]
                EyAB=_E1d_os(0,0,a,b,A[1],B[1])[0,0,:0+0+1]
                EzAB=_E1d_os(0,0,a,b,A[2],B[2])[0,0,:0+0+1]
                ExCD=_E1d_os(1,0,c,d,Cpos[0],Dpos[0])[1,0,:1+0+1]
                EyCD=_E1d_os(0,0,c,d,Cpos[1],Dpos[1])[0,0,:0+0+1]
                EzCD=_E1d_os(0,0,c,d,Cpos[2],Dpos[2])[0,0,:0+0+1]
                val=0.0
                for i in range(1+0+1):
                    Ei=ExAB[i]
                    for j in range(1+0+1):
                        Ej=ExCD[j]
                        for k in range(0+0+1):
                            Ek=EyAB[k]
                            for l in range(0+0+1):
                                El=EyCD[l]
                                for m in range(0+0+1):
                                    Em=EzAB[m]
                                    for n in range(0+0+1):
                                        En=EzCD[n]
                                        order=i+j+k+l+m+n
                                        val += Ei*Ej*Ek*El*Em*En*boys_function(order, T)
                return pre*val
            val_B = primitive_old_boys_total_order_wrapper()
            info(f"primitive OS-R = {val_R:.12e}, Boys-old = {val_B:.12e}, ratio={val_B/val_R:.6f}")
        except Exception as e:
            warn("primitive ERI test failed: " + str(e))
    else:
        warn("integrals モジュール未検出 -> primitive ERI テストをスキップ")

    # final suggestions
    print("\n=== Summary suggestions ===")
    # quick heuristics
    if S is None:
        warn("Overlap S が missing -> 多くのチェックができません。S を npz に保存してください。")
    if D is None or C is None:
        warn("D または C が missing -> density 再構築/保存部分を確認してください。")
    if J is None or K is None:
        warn("J/K が missing -> JK ビルダを確認してください（保存も）。")
    if _HAS_PYSCF:
        ok("PySCF available -> recommended to run the PySCF bench for J/K comparison.")
    else:
        info("PySCF not available -> consider installing for independent bench: pip install pyscf")

if __name__ == "__main__":
    run_all()
