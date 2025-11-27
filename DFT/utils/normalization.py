# DFT/utils/normalization.py
# -*- coding: utf-8 -*-
"""
Contracted GTO（収縮ガウス型軌道）の規格化ユーティリティ。
各 AO が ⟨χ|χ⟩ = 1 となるように、収縮係数をスケールします。

ポイント:
- 正規化の評価は、1電子重なりの "収縮" 版と厳密に同じ式を用いるのが最も安全。
  既に DFT.integrals.one_electron.overlap.contracted_overlap(g1,g2) があり、
  これを g1=g2=g として使えば、二重正規化や式の取り違えを避けられます。
"""

from __future__ import annotations
import math
from DFT.integrals.one_electron.overlap import contracted_overlap


def contracted_norm(gto) -> float:
    """
    収縮 GTO g について、norm2 = ⟨g|g⟩ を contracted_overlap(g,g) で評価し、
    規格化係数 scale = 1/sqrt(norm2) を返す。

    注意:
    - contracted_overlap 側は「プリミティブ正規化（N）」を内部で掛けた
      primitive_overlap を用いているため、ここで N を重ね掛けする必要はない。
    - norm2 <= 0 の場合は異常（指数や係数の誤り等）なので例外を投げる。
    """
    norm2 = float(contracted_overlap(gto, gto))
    if not math.isfinite(norm2) or norm2 <= 0.0:
        raise ValueError(f"[normalization] invalid norm^2 for {gto}: {norm2}")
    return 1.0 / math.sqrt(norm2)
