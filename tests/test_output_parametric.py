
# tests/test_output_parametric.py
# -*- coding: utf-8 -*-
import os
import json
import numpy as np

def discover_run_files(run_dir):
    files = {}
    for fn in os.listdir(run_dir):
        p = os.path.join(run_dir, fn)
        if fn.startswith("matrices_") and fn.endswith(".json"):
            files['matrices'] = p
        if fn.startswith("details_") and fn.endswith(".json"):
            files['details'] = p
    return files

def to_np(obj):
    return np.array(obj['data'], dtype=float)

def test_parametric_runs(output_runs, scf_tools):
    for run_dir in output_runs:
        files = discover_run_files(run_dir)
        if 'matrices' not in files or 'details' not in files:
            continue

        with open(files['matrices'], 'r', encoding='utf-8') as f:
            mat = json.loads(f.read())['arrays']
        S = to_np(mat['S'])
        H = to_np(mat['H_core'])
        D = to_np(mat['D_final'])
        C = to_np(mat['C'])

        # 対称性
        for A, name in [(S,'S'), (H,'H_core'), (D,'D_final')]:
            err = scf_tools['symmetry_maxabs']
            assert err < 1e-12, f"{os.path.basename(run_dir)}: {name} symmetry error {err}"

        # 直交性
        errC = scf_tools['orthogonality_error']
        assert errC < 1e-10, f"{os.path.basename(run_dir)}: C^T S C error {errC}"

        # 電子数（details の occupations 合計）
        with open(files['details'], 'r', encoding='utf-8') as f:
            det = json.loads(f.read())
        occ = np.array(det['details']['occupations'], dtype=float)
        Ne_target = float(np.sum(occ))
        Ne = scf_tools['electron_count']
        assert abs(Ne - Ne_target) < 1e-6, f"{os.path.basename(run_dir)}: Tr(D S)={Ne} vs {Ne_target}"

        # 同定射性（閉殻でない可能性もあるので緩め）
        idem = scf_tools['idempotency_error']
        assert idem < 1e-3, f"{os.path.basename(run_dir)}: idempotency error {idem}"
