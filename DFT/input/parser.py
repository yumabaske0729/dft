
# DFT/input/parser.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np


class Atom:
    """最小要件: symbol と position(np.ndarray(3,)) を持つ"""
    def __init__(self, symbol: str, xyz: List[float]):
        arr = np.asarray(xyz, dtype=float)
        self.symbol = symbol
        self.position = arr  # assign_basis_functions が参照


class Molecule:
    """最小要件: atoms / charge / multiplicity を持つ"""
    def __init__(self, atoms: List[Atom], charge: int, multiplicity: int):
        self.atoms = list(atoms)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)


def _resolve_xyz_path(filename: str) -> Path:
    """
    .xyz の所在を既知の候補から解決する（先に見つかったものを採用）。
    検索順:
      1) そのまま（絶対 or CWD 相対）
      2) ./DFT/molecure/<filename>   ← ユーザー要望
      3) ./DFT/input/<filename>
      4) ./DFT/<filename>
      5) ./gaussian/<filename>       ← 旧構成互換（任意）
    """
    p = Path(filename).expanduser()
    candidates = [p]
    if not p.is_absolute():
        root = Path.cwd()
        candidates += [
            root / "DFT" / "molecure" / filename,
            root / "DFT" / "input" / filename,
            root / "DFT" / filename,
            root / "gaussian" / filename,
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "XYZ not found. Tried:\n  " + "\n  ".join(str(x) for x in candidates)
    )


def parse_xyz(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    """
    標準的な .xyz を読み込み、Molecule を返す（必ず返す／None は返さない）。
    .xyz は以下を想定：
      - 先頭2行（原子数／コメント）はあってもなくても可
      - 以降は行ごとに: Symbol  x  y  z
    """
    path = _resolve_xyz_path(filename)
    lines = path.read_text(encoding="utf-8").splitlines()

    # 先頭2行が原子数+コメントならスキップ
    start = 0
    if lines:
        try:
            int(lines[0].strip())
            start = 2
        except Exception:
            start = 0

    atoms: List[Atom] = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 4:
            # 空行やコメントはスキップ
            continue
        sym = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except Exception as e:
            raise ValueError(f"Invalid XYZ numeric fields in line: {line}") from e
        atoms.append(Atom(sym, [x, y, z]))

    if not atoms:
        raise ValueError("No atom records parsed from XYZ.")

    return Molecule(atoms=atoms, charge=charge, multiplicity=multiplicity)
