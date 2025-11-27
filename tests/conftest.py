
# tests/conftest.py
# -*- coding: utf-8 -*-
import json
import os
import glob
import pytest
import numpy as np

def load_matrices_json(path: str):
    """matrices_*.json を読み込んで numpy 配列に変換"""
    with open(path, 'r', encoding='utf-8') as f:
        mat = json.loads(f.read())
    arrays = mat['arrays']

    def to_np(obj):
        return np.array(obj['data'], dtype=float)

    return {
        'S': to_np(arrays['S']),
        'T': to_np(arrays['T']),
        'V': to_np(arrays['V']),
        'H_core': to_np(arrays['H_core']),
        'C': to_np(arrays['C']),
        'eps': np.array(arrays['eps']['data'], dtype=float),
        'D_final': to_np(arrays['D_final']),
    }

def load_details_json(path: str):
    """details_*.json を読み込む"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

@pytest.fixture(scope="session")
def output_root():
    """output/ ディレクトリの存在確認"""
    root = os.path.join(os.getcwd(), "output")
    assert os.path.isdir(root), f"not found: {root}"
    return root

@pytest.fixture(scope="session")
def output_runs(output_root):
    """output/ 以下の計算 run ディレクトリ一覧"""
    runs = sorted(
        [d for d in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(d)]
    )
    assert runs, "no runs under output/"
    return runs

@pytest.fixture(scope="session")
def latest_run(output_runs):
    """最新 run ディレクトリ"""
    return output_runs[-1]

@pytest.fixture(scope="session")
def h2o_run(output_runs):
    """H2O を含む最新 run ディレクトリ"""
    candidates = [d for d in output_runs if "H2O" in os.path.basename(d)]
    if not candidates:
        pytest.skip("H2O run not found under output/")
    return sorted(candidates)[-1]

@pytest.fixture(scope="session")
def matrices_h2o(h2o_run):
    """H2O の matrices_*.json を読み込み"""
    p = os.path.join(h2o_run, "matrices_H2O.json")
    assert os.path.isfile(p), f"missing {p}"
    return load_matrices_json(p)

@pytest.fixture(scope="session")
def details_h2o(h2o_run):
    """H2O の details_*.json を読み込み"""
    p = os.path.join(h2o_run, "details_H2O.json")
    assert os.path.isfile(p), f"missing {p}"
    return load_details_json(p)

@pytest.fixture(scope="session")
def scf_tools():
    """
    scf_tools/tools.py の関数群を提供。
    見つからない場合は import 時に pytest が自動 skip する。
    """
    mod = pytest.importorskip("scf_tools.tools", reason="scf_tools/tools.py not available")
    return {
        'sanitize_de': mod.sanitize_de,
        'mad_outliers': mod.mad_outliers,
        'lowdin': mod.lowdin,
        'electron_count': mod.electron_count,
        'idempotency_error': mod.idempotency_error,
        'orthogonality_error': mod.orthogonality_error,
        'symmetry_maxabs': mod.symmetry_maxabs,
    }
