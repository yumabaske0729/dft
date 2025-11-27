# DFT/basis/load_basis.py  — gbs専用ローダ（JSON関連コードは撤去）
# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, List

# 受理する拡張子
_GAUSS_EXTS = {".gbs", ".bas", ".gaussian", ".txt"}

class BasisLoader:
    """
    Gaussian形式（.gbs/.bas/.txt）専用の基底関数ローダ。
    指定ディレクトリ内の基底関数セットを自動検出して管理します。
    """
    def __init__(self, basis_dir: str | None = None):
        basis_dir = basis_dir or os.path.dirname(os.path.abspath(__file__))
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
    """
    Gaussian形式のテキスト基底をパース。
    - SP殻は SとPの殻へ分割して格納
    - D記法→E記法変換
    出力形式:
      { "H": {"shells": [ { "angular_momentum":[0], "exponents":[...], "coefficients":[[...]] }, ... ] }, ... }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # コメントと空行を除去、D記法→E記法
    lines: List[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("!"):
            continue
        lines.append(s.replace("D", "E").replace("d", "e"))

    # 角運動量の文字→整数マップ
    lmap = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}

    data: Dict[str, Any] = {}
    i = 0
    cur_sym: str | None = None

    def ensure_elem(sym: str):
        if sym not in data:
            data[sym] = {"shells": []}

    # 本体パース
    while i < len(lines):
        toks = lines[i].split()
        i += 1
        if not toks:
            continue
        # 区切り
        if toks[0] == "****":
            cur_sym = None
            continue
        # 元素開始（"H 0" など）
        if len(toks) >= 2 and toks[1] == "0":
            cur_sym = toks[0]
            ensure_elem(cur_sym)
            continue

        if cur_sym is None:
            continue

        shell_type = toks[0].upper()
        # --- SP 共有殻 ---
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

            # S殻として追加
            data[cur_sym]["shells"].append({
                "angular_momentum": [lmap["S"]],
                "exponents": s_exps,
                "coefficients": [s_coefs],
            })
            # P殻として追加
            data[cur_sym]["shells"].append({
                "angular_momentum": [lmap["P"]],
                "exponents": p_exps,
                "coefficients": [p_coefs],
            })
            continue

        # --- 通常殻（S/P/D/…） ---
        if shell_type not in lmap:
            # 未知行はスキップ
            continue

        nprim = int(toks[1])
        scale = 1.0
        if len(toks) >= 3:
            try:
                scale = float(toks[2])
            except Exception:
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