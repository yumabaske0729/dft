# qa_jk_energy.py
import numpy as np, glob, os

def pick_latest_npz():
    cands = glob.glob(os.path.join('DFT','output','*_*_*_RHF','matrices_*.npz'))
    if not cands:
        raise FileNotFoundError("matrices_*.npz が見つかりません。まず RHF を走らせてください。")
    return max(cands, key=os.path.getmtime)

def main():
    p = pick_latest_npz()
    z = np.load(p)
    D  = z['D_final']
    J  = z['J_final']
    K  = z['K_final']
    Hc = z['H_core'] if 'H_core' in z else z['T'] + z['V']  # 保険
    EJ = float(np.einsum('mn,mn->', D, J))
    EK = float(np.einsum('mn,mn->', D, K))
    EH = float(np.einsum('mn,mn->', D, Hc))
    # スピン和密度 D（=2P）を使う RHF のエネルギー式:
    # E_elec = EH + 0.5 * (EJ - 0.5*EK)
    E2 = 0.5 * (EJ - 0.5 * EK)
    Eele = EH + E2
    print(f"[file] {p}")
    print(f"Tr[D H_core]   = {EH: .12f} Ha")
    print(f"sum D·J        = {EJ: .12f} Ha")
    print(f"sum D·K        = {EK: .12f} Ha")
    print(f"E_2e(recalc)   = {E2: .12f} Ha   <- 0.5*(EJ - 0.5*EK)")
    print(f"E_elec(recalc) = {Eele: .12f} Ha")
    # K の暫定スケール診断（s を上げると E_2e は下がる）
    for s in (1.25, 1.50, 1.75, 2.00):
        E2s = 0.5 * (EJ - 0.5 * (s * EK))
        print(f"[test] K_scale={s:.2f}  ->  E_2e={E2s: .12f} Ha,  ΔE_2e={E2s - E2:+.3f} Ha")

if __name__ == "__main__":
    main()