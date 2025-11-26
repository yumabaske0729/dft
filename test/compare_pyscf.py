#!/usr/bin/env python3
"""
Compare our AO matrices and traces against PySCF RHF.

優先順位（入力ファイルの解決）:
1) --npz と --json が指定されていれば、それを使用
2) いずれか未指定の場合、DFT/output/ から最新 run を自動選択して npz/json/summary を補完
   - --method RHF|DFT / --filter-basis / --filter-tag で絞り込み可能
3) PySCF の幾何は --xyz > --summary(input.path) > details.json の atoms の順で決定

Usage examples:
  # 明示パス指定
  python compare_pyscf.py --npz DFT/output/.../matrices_XYZ.npz ^
                          --json DFT/output/.../details_XYZ.json ^
                          --summary DFT/output/.../summary_XYZ.json ^
                          --basis 6-31g

  # 指定なし -> 最新 run を自動選択（RHF を優先して比較）
  python compare_pyscf.py --method RHF --basis 6-31g

  # 基底・タグで絞り込み（DFTの最新）
  python compare_pyscf.py --method DFT --filter-basis 6-31G --filter-tag H2O

  # PySCF を使わず自前値のみ確認
  python compare_pyscf.py --no-pyscf
"""

import argparse, json, sys, os, glob, numpy as np

# ---------- ユーティリティ ----------
def _pick_key(z, *names):
    """np.load(..., allow_pickle=True) が返す NpzFile から、最初に見つかったキーを返す。"""
    for k in names:
        if k in z:
            return z[k]
    return None

def build_geom_from_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        det = json.load(f)
    geom = det.get('details',{}).get('molecule',{}).get('atoms') \
         or det.get('geometry') or det.get('atoms')
    parts = []
    if isinstance(geom, list) and len(geom) > 0:
        for at in geom:
            if isinstance(at, dict):
                sym = at.get('element') or at.get('symbol') or at.get('atom')
                xyz = at.get('xyz') or at.get('position') or at.get('coord')
                if sym and xyz:
                    parts.append(f"{sym} {float(xyz[0])} {float(xyz[1])} {float(xyz[2])}")
    if not parts:
        raise RuntimeError("No atoms in JSON; use --xyz or --summary to locate the XYZ file.")
    return '; '.join(parts)

def load_xyz_path_from_summary(summary_path):
    with open(summary_path,'r',encoding='utf-8') as f:
        s = json.load(f)
    ip = (s.get('input') or {}).get('path')
    if not ip or not os.path.exists(ip):
        raise RuntimeError(f"Could not get existing XYZ path from summary: {ip}")
    return ip

def safe_load_npz(npz_path):
    """
    NPZ から D/H/J/K/S/C/eps を安全にロード。
    配列の選択に or を使わず、順次キー探索で評価します。
    """
    z = np.load(npz_path, allow_pickle=True)

    D = _pick_key(z, 'D_final', 'D', 'density')
    J = _pick_key(z, 'J_final', 'J')
    K = _pick_key(z, 'K_final', 'K')
    S = _pick_key(z, 'S', 'overlap')
    C = _pick_key(z, 'C', 'C_final')
    eps = _pick_key(z, 'eps', 'eigs')

    H = _pick_key(z, 'H_core')
    if H is None:
        T = _pick_key(z, 'T')
        V = _pick_key(z, 'V')
        if T is not None and V is not None:
            H = T + V

    return dict(z=z, D=D, H=H, J=J, K=K, S=S, C=C, eps=eps)

# ---------- 追加: 最新 run を検出 ----------
def pick_latest_run(method=None, filter_basis=None, filter_tag=None, root="DFT/output"):
    """
    root 配下の run ディレクトリから最新を返す。
    run_id 形式: <ts>_<input_tag>_<formula>_<BASIS>_<METHOD>
    例: 20251023_131238_H2O_H2O_6-31G_RHF
    """
    if not os.path.isdir(root):
        return None
    candidates = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not candidates:
        return None

    def ok(d):
        base = os.path.basename(d)
        if method and not base.upper().endswith(f"_{method.upper()}"):
            return False
        if filter_basis and filter_basis.upper() not in base.upper():
            return False
        if filter_tag and filter_tag.upper() not in base.upper():
            return False
        return True

    filtered = [d for d in candidates if ok(d)] or candidates
    return max(filtered, key=os.path.getmtime) if filtered else None

def find_artifacts_in_run(run_dir):
    """run ディレクトリ内の matrices_*.npz, details_*.json, summary_*.json を拾い、各々最新を返す。"""
    def newest(globpat):
        files = glob.glob(os.path.join(run_dir, globpat))
        return max(files, key=os.path.getmtime) if files else None
    return newest("matrices_*.npz"), newest("details_*.json"), newest("summary_*.json")

# ---------- メイン ----------
def main():
    p = argparse.ArgumentParser()
    # 明示パス（指定があれば最優先）
    p.add_argument('--npz', default=None, help='path to matrices_*.npz')
    p.add_argument('--json', default=None, help='path to details_*.json (for E_nuc and fallback geometry)')
    p.add_argument('--summary', default=None, help='path to summary_*.json (to locate input.xyz)')
    p.add_argument('--xyz', default=None, help='explicit XYZ to feed PySCF (overrides --summary/--json)')

    # 自動選択の絞り込み（任意）
    p.add_argument('--method', choices=['RHF','DFT'], default=None,
                   help='prefer runs ending with _RHF or _DFT when auto-picking latest')
    p.add_argument('--filter-basis', default=None, help='limit latest-run search to BASIS substring (e.g., "6-31G")')
    p.add_argument('--filter-tag',   default=None, help='limit latest-run search to runs containing this TAG (e.g., "H2O")')

    # その他
    p.add_argument('--basis', default='6-31g', help='basis name for PySCF (default: 6-31g)')
    p.add_argument('--no-pyscf', action='store_true', help='do not run pyscf (just print our values)')

    args = p.parse_args()

    # ---- 1) 指定されていればそれを使う（不足分のみ補完）----
    want_auto = (not args.npz) or (not args.json)
    if want_auto:
        run_dir = pick_latest_run(method=args.method,
                                  filter_basis=args.filter_basis,
                                  filter_tag=args.filter_tag,
                                  root="DFT/output")
        if not run_dir:
            print("[ERROR] No run directory found under DFT/output.", file=sys.stderr)
            if not args.npz or not args.json:
                print("       Provide --npz and --json explicitly.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"[latest] picked run: {run_dir}")
            npz_auto, json_auto, summary_auto = find_artifacts_in_run(run_dir)
            if not args.npz:    args.npz = npz_auto
            if not args.json:   args.json = json_auto
            if not args.summary:args.summary = summary_auto

    # ---- 入力チェック（最終）----
    if not args.npz or not os.path.exists(args.npz):
        print("npz not found:", args.npz); sys.exit(1)
    if not args.json or not os.path.exists(args.json):
        print("json not found:", args.json); sys.exit(1)

    # どのファイルを使うか明示
    print(f"[use] npz    = {args.npz}")
    print(f"[use] json   = {args.json}")
    print(f"[use] summary= {args.summary or '(none)'}")

    # ---- 自前実装の値を表示 ----
    data = safe_load_npz(args.npz)
    D_our, H_our, J_our, K_our, S_our = data['D'], data['H'], data['J'], data['K'], data['S']
    if D_our is None or H_our is None or J_our is None or K_our is None:
        print("Required arrays not found in npz (need D/H/J/K). Keys:", list(data['z'].files))
        sys.exit(1)

    trDH_our = float(np.einsum('ij,ij->', D_our, H_our))
    trDJ_our = float(np.einsum('ij,ij->', D_our, J_our))
    trDK_our = float(np.einsum('ij,ij->', D_our, K_our))
    E2_our   = 0.5 * (trDJ_our - 0.5 * trDK_our)
    Eele_our = trDH_our + E2_our

    print("=== Our implementation (from npz) ===")
    print(f"Tr[D H]_our = {trDH_our:.12f} Ha")
    print(f"Tr[D J]_our = {trDJ_our:.12f} Ha")
    print(f"Tr[D K]_our = {trDK_our:.12f} Ha")
    print(f"E_2e_our    = {E2_our:.12f} Ha")
    print(f"E_elec_our  = {Eele_our:.12f} Ha")

    with open(args.json,'r',encoding='utf-8') as f:
        det = json.load(f)
    Enuc = det.get('details',{}).get('components',{}).get('E_nuc') \
        or det.get('E_nuc') or det.get('details',{}).get('E_nuclear')
    if Enuc is not None:
        try:
            Enuc = float(Enuc)
            print(f"E_nuc (details JSON) = {Enuc:.12f} Ha")
            print(f"E_total_our          = {Eele_our + Enuc:.12f} Ha")
        except Exception:
            pass

    if args.no_pyscf:
        print("Skipping PySCF run (--no-pyscf).")
        return
    # ---- PySCF ----
    try:
        from pyscf import gto, scf
    except Exception as e:
        print("PySCF import error:", e)
        print("Install PySCF (pip install pyscf / conda install -c conda-forge pyscf) or run with --no-pyscf")
        sys.exit(1)

    # geometry source priority: --xyz > --summary(input.path) > JSON atoms
    if args.xyz:
        if not os.path.exists(args.xyz):
            raise RuntimeError(f"--xyz path not found: {args.xyz}")
        geom_src = args.xyz
        mol = gto.M(atom=args.xyz, unit='Ang', basis=args.basis)
    elif args.summary:
        try:
            xyz_path = load_xyz_path_from_summary(args.summary)
            geom_src = xyz_path
            mol = gto.M(atom=xyz_path, unit='Ang', basis=args.basis)
        except Exception as e:
            print(f"[warn] summary-based XYZ failed: {e} — falling back to details.json atoms if available")
            geom = build_geom_from_json(args.json)
            geom_src = f"[atoms from {os.path.basename(args.json)}]"
            mol = gto.M(atom=geom, unit='Ang', basis=args.basis)
    else:
        geom = build_geom_from_json(args.json)
        geom_src = f"[atoms from {os.path.basename(args.json)}]"
        mol = gto.M(atom=geom, unit='Ang', basis=args.basis)

    print("\n=== Running PySCF RHF on geometry:", geom_src, "basis:", args.basis, "===")
    print("PySCF: nao_nr =", mol.nao_nr())
    mf = scf.RHF(mol)
    e_tot = mf.kernel()
    dm    = mf.make_rdm1()           # spin-summed AO density
    vj, vk = mf.get_jk(mol, dm)      # Coulomb, Exchange
    hcore  = mf.get_hcore()
    S_py   = mf.get_ovlp()

    trDH_py = float(np.einsum('ij,ij->', dm, hcore))
    trDJ_py = float(np.einsum('ij,ij->', dm, vj))
    trDK_py = float(np.einsum('ij,ij->', dm, vk))
    E2_py   = 0.5 * (trDJ_py - 0.5 * trDK_py)
    Eele_py = trDH_py + E2_py

    print("\n=== PySCF results ===")
    print(f"E_total_pyscf = {e_tot:.12f} Ha")
    try:
        E_nuc_py = mol.energy_nuc()
        print(f"E_nuc_pyscf   = {E_nuc_py:.12f} Ha")
    except Exception:
        E_nuc_py = None
    print(f"E_elec_pyscf  = {Eele_py:.12f} Ha")
    print(f"Tr[D H]_py    = {trDH_py:.12f} Ha")
    print(f"Tr[D J]_py    = {trDJ_py:.12f} Ha")
    print(f"Tr[D K]_py    = {trDK_py:.12f} Ha")
    print(f"E_2e_py       = {E2_py:.12f} Ha")

    # ---- Differences (our - pyscf) ----
    def show(name, ours, pyscfv):
        diff = ours - pyscfv
        print(f"{name:15s}: our = {ours:.12f} pyscf = {pyscfv:.12f} diff = {diff:.12e}")

    print("\n=== Differences (our - pyscf) ===")
    show("Tr[D H]", trDH_our, trDH_py)
    show("Tr[D J]", trDJ_our, trDJ_py)
    show("Tr[D K]", trDK_our, trDK_py)
    show("E_2e",   E2_our,   E2_py)
    show("E_elec", Eele_our, Eele_py)
    if Enuc is not None and E_nuc_py is not None:
        print(f"\nE_nuc difference (our - pyscf): {Enuc - E_nuc_py:.12e} Ha")

    # 行列レベルの差
    J_py, K_py = vj, vk
    if J_py.shape == J_our.shape:
        Jdiff = J_our - J_py
        Kdiff = K_our - K_py
        print("\nJ matrix: ||J_our - J_py||_F = {:.6e}, max|diff| = {:.6e}".format(
            np.linalg.norm(Jdiff), np.max(np.abs(Jdiff))))
        print("K matrix: ||K_our - K_py||_F = {:.6e}, max|diff| = {:.6e}".format(
            np.linalg.norm(Kdiff), np.max(np.abs(Kdiff))))
    else:
        print("\nCannot compare J/K shapes: our {}, pyscf {}".format(J_our.shape, J_py.shape))

    # S（重なり）と H の簡易チェック（任意）
    if S_our is not None and S_py is not None and S_our.shape == S_py.shape:
        Sdiff = S_our - S_py
        print("S matrix:  ||S_our - S_py||_F = {:.6e}, max|diff| = {:.6e}".format(
            np.linalg.norm(Sdiff), np.max(np.abs(Sdiff))))
    else:
        print("S matrix: shape mismatch or S not present in npz.")

    # MO energies quick look
    eps_our = np.asarray(data['eps']).flatten() if data.get('eps') is not None else None
    if eps_our is not None:
        eps_py = np.sort(mf.mo_energy)
        eps_our_sorted = np.sort(eps_our)
        nshow = min(len(eps_our_sorted), len(eps_py), 10)
        print("\nMO energies (lowest ->):")
        for i in range(nshow):
            print(f" i={i:2d}: our_eps = {eps_our_sorted[i]: .9f}  py_eps = {eps_py[i]: .9f}  diff = {eps_our_sorted[i]-eps_py[i]: .3e}")

    print("\nDone.")

if __name__ == '__main__':
    main()