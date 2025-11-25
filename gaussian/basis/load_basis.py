# -*- coding: utf-8 -*-
"""
DFT/basis/load_basis.py
Gaussian形式（.gbs/.bas/.txt）専用の基底関数ローダ。
- JSON読み込みは完全削除
- SP殻対応（SとPに分割）
- D記法→E記法変換
"""

import os
from typing import Dict, Any, List

# 受理する拡張子
_GAUSS_EXTS = {".gbs", ".bas", ".gaussian", ".txt"}

class BasisLoader:
    def __init__(self, basis_dir: str | None = None):
        if basis_dir is None:
            basis_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(basis_dir):
            raise NotADirectoryError(f"basis_dir is not a directory: {basis_dir}")
        self.basis_dir = basis_dir
        self.available_sets = self._discover_basis_sets()

    def _discover_basis_sets(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for fn in os.listdir(self.basis_dir):
            path = os.path.join(self.basis_dir, fn)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(fn)
            if ext.lower() in _GAUSS_EXTS:
                out[stem.lower()] = path
        return out

    def list_basis_sets(self) -> List[str]:
        return sorted(self.available_sets.keys())

    def load(self, basis_name: str) -> Dict[str, Any]:
        key = basis_name.lower()
        if key not in self.available_sets:
            raise ValueError(f"Basis set '{basis_name}' not found. Available: {self.list_basis_sets()}")
        path = self.available_sets[key]
        return _parse_gaussian_basis_file(path)


def _parse_gaussian_basis_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines: List[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("!"):
            continue
        lines.append(s.replace("D", "E").replace("d", "e"))

    lmap = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}
    data: Dict[str, Any] = {}
    i = 0
    cur_sym: str | None = None

    def ensure_elem(sym: str):
        if sym not in data:
            data[sym] = {"shells": []}

    while i < len(lines):
        toks = lines[i].split()
        i += 1
        if not toks:
            continue
        if toks[0] == "****":
            cur_sym = None
            continue
        if len(toks) >= 2 and toks[1] == "0":
            cur_sym = toks[0]
            ensure_elem(cur_sym)
            continue

        if cur_sym is None:
            continue

        shell_type = toks[0].upper()
        if shell_type == "SP":
            nprim = int(toks[1])
            s_scale = 1.0
            p_scale = 1.0
            if len(toks) >= 3:
                s_scale = float(toks[2])
            if len(toks) >= 4:
                p_scale = float(toks[3])

            s_exps, s_coefs, p_exps, p_coefs = [], [], [], []
            for _ in range(nprim):
                vals = lines[i].split()
                i += 1
                exp = float(vals[0])
                cS = float(vals[1])
                cP = float(vals[2])
                s_exps.append(exp * s_scale)
                s_coefs.append(cS)
                p_exps.append(exp * p_scale)
                p_coefs.append(cP)

            data[cur_sym]["shells"].append({
                "angular_momentum": [lmap["S"]],
                "exponents": s_exps,
                "coefficients": [s_coefs],
            })
            data[cur_sym]["shells"].append({
                "angular_momentum": [lmap["P"]],
                "exponents": p_exps,
                "coefficients": [p_coefs],
            })
            continue

        if shell_type not in lmap:
            continue
        nprim = int(toks[1])
        scale = 1.0
        if len(toks) >= 3:
            try:
                scale = float(toks[2])
            except:
                scale = 1.0

        exps, coefs = [], []
        for _ in range(nprim):
            vals = lines[i].split()
            i += 1
            exp = float(vals[0]) * scale
            c = float(vals[1])
            exps.append(exp)
            coefs.append(c)

        data[cur_sym]["shells"].append({
            "angular_momentum": [lmap[shell_type]],
            "exponents": exps,
            "coefficients": [coefs],
        })

    return data
