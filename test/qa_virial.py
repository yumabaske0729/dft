# qa_jk_energy_fullcheck.py
import numpy as np, glob, os, json

def pick_latest_run():
    cands = glob.glob(os.path.join('DFT', 'output', '*_*_*_RHF'))
    if not cands:
        raise FileNotFoundError("DFT/output/*_*_*_RHF が見つかりません。まず RHF を走らせてください。")
    return max(cands, key=os.path.getmtime)

def main():
    run_dir = pick_latest_run()
    # 文件名タグ（H2O など）
    tag = None
    for name in os.listdir(run_dir):
        if name.startswith('matrices_') and name.endswith('.npz'):
            tag = name[len('matrices_'):-len('.npz')]
            break
    if tag is None:
        raise FileNotFoundError("matrices_*.npz が見つかりません。")

    npz_path = os.path.join(run_dir, f"matrices_{tag}.npz")
    det_path = os.path.join(run_dir, f"details_{tag}.json")

    z = np.load(npz_path)
    D  = z['D_final']
    J  = z['J_final']
    K  = z['K_final']
    Hc = z['H_core'] if 'H_core' in z else (z['T'] + z['V'])

    EJ = float(np.einsum('mn,mn->', D, J))
    EK = float(np.einsum('mn,mn->', D, K))
    EH = float(np.einsum('mn,mn->', D, Hc))
    E2 = 0.5 * (EJ - 0.5 * EK)
    Eele = EH + E2

    # 総エネルギー：details から E_nuc を読む
    with open(det_path, 'r', encoding='utf-8') as f:
        det = json.load(f)
    comps = det.get('details', {}).get('components', {})
    Enuc = float(comps.get('E_nuc', 0.0))
    Etot = Eele + Enuc

    # J-K の差分診断
    diffF = float(np.linalg.norm(J - K))
    maxAbs = float(np.max(np.abs(J - K)))

    print(f"[file] {npz_path}")
    print(f"Tr[D H_core]   = {EH: .12f} Ha")
    print(f"sum D·J        = {EJ: .12f} Ha")
    print(f"sum D·K        = {EK: .12f} Ha")
    print(f"E_2e(recalc)   = {E2: .12f} Ha   <- 0.5*(EJ - 0.5*EK)")
    print(f"E_elec(recalc) = {Eele: .12f} Ha")
    print(f"E_nuc(details) = {Enuc: .12f} Ha")
    print(f"E_total(recalc)= {Etot: .12f} Ha")
    print(f"||J-K||_F      = {diffF:.6e}")
    print(f"max|J-K|       = {maxAbs:.6e}")

    # 閾値チェック（健全性の自動検知）
    if diffF < 1e-10 and np.linalg.norm(J) > 1e-10:
        print("[WARN] J≈K（縮約が崩れている可能性）；jk.py を確認してください。")

if __name__ == "__main__":
    main()