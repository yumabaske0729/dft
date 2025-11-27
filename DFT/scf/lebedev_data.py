# DFT/scf/lebedev_data.py
# -*- coding: utf-8 -*-
"""
Lebedev grid loader (NPY/embedded/fallback).

改訂点:
- DFT/scf/lebedev/ 配下の NPY を自動探索。
- degree（角点数）で指定し、最寄りの degree へのフォールバックも可能。
- 従来の "order"（3,5,7,...,131）指定もサポート（内部で degree に変換）。
- 埋め込みは最小の degree=6 のみ（等方6点, 和=4π）。

保存ファイル命名規約:
  lebedev_<degree>_points.npy : shape (N,3)
  lebedev_<degree>_weights.npy: shape (N,)  ※ 4π 正規化（sum≈4π）を前提

公開API:
  - load_grid(order: int) -> dict{"points":(N,3), "weights":(N,)}
  - load_grid_by_degree(degree: int, allow_nearest: bool=True) -> dict
  - discover_available_degrees() -> List[int]
"""

from __future__ import annotations
import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# ---- mapping between lebedev "order" and degree (points) ----
DEG2ORD = {
      6:   3,   14:   5,   26:   7,   38:   9,   50:  11,   74:  13,   86:  15,  110:  17,
    146:  19,  170:  21,  194:  23,  230:  25,  266:  27,  302:  29,  350:  31,  434:  35,
    590:  41,  770:  47,  974:  53, 1202:  59, 1454:  65, 1730:  71, 2030:  77, 2354:  83,
   2702:  89, 3074:  95, 3470: 101, 3890: 107, 4334: 113, 4802: 119, 5294: 125, 5810: 131,
}
ORD2DEG = {v: k for k, v in DEG2ORD.items()}

FOUR_PI = 4.0 * math.pi


# ---- embedded minimal grid: degree=6 ----
def _embedded_deg6() -> Dict[str, np.ndarray]:
    pts = np.array(
        [
            [ 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0,-1.0, 0.0],
            [ 0.0, 0.0, 1.0],
            [ 0.0, 0.0,-1.0],
        ],
        dtype=float,
    )
    w = np.full(6, FOUR_PI / 6.0, dtype=float)  # weights sum = 4π
    return {"points": pts, "weights": w}


def _base_dir() -> str:
    """このファイルのディレクトリを返す（DFT/scf）。"""
    return os.path.dirname(os.path.abspath(__file__))


def _lebedev_dir() -> str:
    """サブフォルダ DFT/scf/lebedev/ を返す。"""
    return os.path.join(_base_dir(), "lebedev")


def _npy_paths_for_degree(degree: int) -> Tuple[str, str]:
    """候補となる NPY のパス（base と subdir）を順番に返す。"""
    candidates = []
    for root in (_base_dir(), _lebedev_dir()):
        candidates.append(os.path.join(root, f"lebedev_{degree}_points.npy"))
        candidates.append(os.path.join(root, f"lebedev_{degree}_weights.npy"))
    return tuple(candidates)


def _load_from_npy_degree_exact(degree: int) -> Optional[Dict[str, np.ndarray]]:
    """指定 degree の NPY を base/lebedev の順で探索してロード。"""
    for root in (_base_dir(), _lebedev_dir()):
        pfile = os.path.join(root, f"lebedev_{degree}_points.npy")
        wfile = os.path.join(root, f"lebedev_{degree}_weights.npy")
        if os.path.isfile(pfile) and os.path.isfile(wfile):
            pts = np.asarray(np.load(pfile), float)
            w = np.asarray(np.load(wfile), float)
            if pts.ndim != 2 or pts.shape[1] != 3 or w.ndim != 1 or w.shape[0] != pts.shape[0]:
                raise ValueError(
                    f"Invalid shapes in {pfile}/{wfile}: points {pts.shape}, weights {w.shape}"
                )
            s = float(np.sum(w))
            # normalize to 4π if wildly off (safety; normally不要)
            if not (5.0 < s < 20.0):
                scale = FOUR_PI / (s if s != 0.0 else 1.0)
                w = w * scale
            return {"points": pts, "weights": w}
    return None


def discover_available_degrees() -> List[int]:
    """DFT/scf と DFT/scf/lebedev を走査し、利用可能な degree を列挙。"""
    found = set()
    for root in (_base_dir(), _lebedev_dir()):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.startswith("lebedev_") and name.endswith("_points.npy"):
                try:
                    deg = int(name.split("_")[1])
                    # weights も存在するか
                    if os.path.isfile(os.path.join(root, f"lebedev_{deg}_weights.npy")):
                        found.add(deg)
                except Exception:
                    pass
    return sorted(found)


def _choose_nearest_degree(request_deg: int, available: List[int]) -> Optional[int]:
    """利用可能な degree から request_deg に最も近いものを選ぶ（同距離なら大きい方）。"""
    if not available:
        return None
    best = None
    best_key = None
    for deg in available:
        key = (abs(deg - request_deg), -deg)  # 近さ優先、タイは大きい方
        if (best_key is None) or (key < best_key):
            best_key = key
            best = deg
    return best


def load_grid_by_degree(degree: int, *, allow_nearest: bool = True) -> Dict[str, np.ndarray]:
    """
    degree（角点数）でレベデフ格子をロード。見つからなければ最近傍 degree にフォールバック。
    戻り値 dict: {"points":(N,3), "weights":(N,)}
    """
    if int(degree) == 6:
        # 常に存在する埋め込み
        data = _embedded_deg6()
        print("[lebedev] using embedded degree=6 (N=6)")
        return data

    exact = _load_from_npy_degree_exact(int(degree))
    if exact is not None:
        return exact

    # 近傍にフォールバック
    if allow_nearest:
        avail = discover_available_degrees()
        alt = _choose_nearest_degree(int(degree), avail)
        if alt is not None and alt != int(degree):
            print(f"[lebedev] degree {degree} not found; falling back to nearest degree {alt}.")
            alt_data = _load_from_npy_degree_exact(alt)
            if alt_data is not None:
                return alt_data

    # どうしても見つからない
    req = int(degree)
    msg = [
        f"[lebedev] lebedev_{req}_points.npy / lebedev_{req}_weights.npy が見つかりません。",
        "探索場所:",
        f"  - " + os.path.join(_base_dir(), f"lebedev_{req}_points.npy"),
        f"  - " + os.path.join(_base_dir(), f"lebedev_{req}_weights.npy"),
        f"  - " + os.path.join(_lebedev_dir(), f"lebedev_{req}_points.npy"),
        f"  - " + os.path.join(_lebedev_dir(), f"lebedev_{req}_weights.npy"),
        "対処: SciPy / PySCF で生成した NPY を上記のいずれかに配置してください。",
    ]
    raise FileNotFoundError("\n".join(msg))


def load_grid(order: int) -> Dict[str, np.ndarray]:
    """
    従来API: order（3,5,...,131）指定でロード。内部で degree に変換。
    """
    if int(order) in ORD2DEG:
        deg = ORD2DEG[int(order)]
    else:
        # 未知の order の場合は degree=6 にフォールバック
        print(f"[lebedev] unknown order={order}; using embedded degree=6.")
        deg = 6
    return load_grid_by_degree(deg, allow_nearest=True)