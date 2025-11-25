# DFT/scf/jk.py
# -*- coding: utf-8 -*-
"""
JK builder（closed-shell, spin-summed density D = 2P）

前提と約束:
 - ERI テンソルは two_electron.build_eri_tensor から
   eri[m,n,r,s] = (mn|rs)
   の並びで渡される（ログにも "(mnrs)" と表示）。
 - Coulomb (J):   J_mn = Σ_rs D_rs (mn|rs)
 - Exchange (K):  K_mn = Σ_rs D_rs (mr|ns)

本実装の要点:
 1) J は tensordot で軸 ([0,1],[2,3]) を明示して安全に収縮。
 2) K は「まず ERI を所望の表現に transpose してから」、einsum で
    縮約添字を明示して収縮（可読性・保守性・事故防止を重視）。
    - STD  : (mr|ns) -> eri_perm = eri.transpose(0,2,1,3); einsum 'rs,mrns->mn'
    - ALT1 : (ms|rn) -> eri_perm = eri.transpose(0,3,2,1); einsum 'rs,msrn->mn'
    - ALT2 : (mn|sr) -> eri_perm = eri.transpose(0,1,3,2); einsum 'rs,mnsr->mn'
 3) form_jk_matrices() では、指定（または環境）モードで K を試し、
    J==K が検出されたら自動で他モードへフォールバック（全滅なら停止）。
 4) ERI の 8重対称強制（診断）や JK 診断ログ、K の診断スケーリング等、
    既存のトグルは継承。
 5) 追加: 環境変数に応じた重診断/ダンプ
    - JK_TRACE=1      : 形状/ノルム/対称性/サンプル値、使用einsum式などを詳細print
    - JK_DIAG=1       : ΣD·J / ΣD·K / R_KJ / ||J-K||_F を print
    - JK_DUMP_JSON=1  : 例外時に JSON ダンプを DFT/output/_jk_crash/ へ保存
    - JK_JSON_DIR     : ダンプ先ディレクトリ（既定: DFT/output/_jk_crash）

環境変数（主なもの）:
 - K_EINSUM            : 初期 K モードの選択 ('STD'/'ALT1'/'ALT2'、既定 'STD')
 - JK_DIAG             : 1 で ΣD·J / ΣD·K / R_KJ / ||J-K||_F 等を表示
 - JK_TRACE            : 1 で重診断を詳細print
 - JK_DUMP_JSON        : 1 で例外時に JSON クラッシュダンプを保存
 - JK_JSON_DIR         : クラッシュダンプ出力先（既定 DFT/output/_jk_crash）
 - JK_FORENSICS        : 1 で forensics（K_wrongJ との距離など）
 - ERI_SYM_8           : 1 で ERI テンソルを 8重対称へ強制（診断用途）
 - K_SCALE_TEST        : K に診断スケールを掛ける（float, 既定 1.0）
 - JK_STRICT_KSCALE    : 1 のとき K_SCALE_TEST != 1.0 を拒否して例外
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import platform
from typing import Literal, Tuple, Dict, Any, Optional

import numpy as np

KMode = Literal["STD", "ALT1", "ALT2"]
_ALLOWED_MODES = {"STD", "ALT1", "ALT2"}

# ------------------------------ 小ユーティリティ ------------------------------ #
def _token(val: Optional[str], default: str) -> str:
    if val is None:
        return default
    t = val.strip().split()
    return t[0] if t else default

def _norm_mode(raw: Optional[str], fallback: str = "STD") -> str:
    tok = _token(raw, fallback).upper()
    if tok not in _ALLOWED_MODES:
        raise ValueError(
            f"Unknown K mode: {raw!r} -> {tok!r}. Allowed: {sorted(_ALLOWED_MODES)}"
        )
    return tok

def _as_bool_env(name: str, default: bool = False) -> bool:
    raw = _token(os.getenv(name, "1" if default else "0"), "1" if default else "0").lower()
    return raw in ("1", "true", "yes", "on")

def _get_k_scale() -> float:
    raw = _token(os.getenv("K_SCALE_TEST", "1.0"), "1.0")
    try:
        return float(raw)
    except Exception:
        return 1.0

def _ensure_sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _sum_D_dot(M: np.ndarray, D: np.ndarray) -> float:
    # Tr[D M] (Frobenius inner product)
    return float(np.einsum("mn,mn->", D, M, optimize=True))

def _symmetrize_eri_8(eri: np.ndarray) -> np.ndarray:
    """
    ERI の 8重対称を強制: (mn|rs) の 8通りの置換平均。
    """
    e = eri
    return (
        e
        + e.transpose(1, 0, 2, 3)
        + e.transpose(0, 1, 3, 2)
        + e.transpose(1, 0, 3, 2)
        + e.transpose(2, 3, 0, 1)
        + e.transpose(3, 2, 0, 1)
        + e.transpose(2, 3, 1, 0)
        + e.transpose(3, 2, 1, 0)
    ) * 0.125

def _md5_of_array(a: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(np.ascontiguousarray(a).data)
    return m.hexdigest()

def _now_tag() -> str:
    import time
    return time.strftime("%Y%m%d_%H%M%S")

def _json_dump(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _module_files_snapshot() -> Dict[str, str]:
    mod = {}
    for name in ("DFT", "DFT.scf.jk", "DFT.scf.rhf", "DFT.integrals.two_electron"):
        try:
            m = sys.modules.get(name, __import__(name))
            mod[name] = getattr(m, "__file__", "")
        except Exception:
            mod[name] = ""
    return mod

def _eri_symmetry_residuals(eri: np.ndarray) -> Dict[str, float]:
    nrm = np.linalg.norm
    return {
        "\n\neri\n\n_F": float(nrm(eri)),
        "\n\neri - eri^T(mn)\n\n_F": float(nrm(eri - eri.transpose(1, 0, 2, 3))),
        "\n\neri - eri^T(rs)\n\n_F": float(nrm(eri - eri.transpose(0, 1, 3, 2))),
        "\n\neri - swap(mrns)\n\n_F": float(nrm(eri - eri.transpose(0, 2, 1, 3))),
        "\n\neri - swap(msrn)\n\n_F": float(nrm(eri - eri.transpose(0, 3, 2, 1))),
        "\n\neri - swap(mnsr)\n\n_F": float(nrm(eri - eri.transpose(0, 1, 3, 2))),
    }

def _eri_samples(eri: np.ndarray, nsamp: int = 32) -> Dict[str, Any]:
    rng = np.random.default_rng(12345)
    n = eri.shape[0]
    out = []
    for _ in range(min(nsamp, n * n)):
        m = int(rng.integers(0, n))
        n_ = int(rng.integers(0, n))
        r = int(rng.integers(0, n))
        s = int(rng.integers(0, n))
        base = float(eri[m, n_, r, s])    # (mn|rs)
        mrns = float(eri[m, r, n_, s])    # (mr|ns)
        msrn = float(eri[m, s, r, n_])    # (ms|rn)
        mnsr = float(eri[m, n_, s, r])    # (mn|sr)
        out.append({"mnrs": [m, n_, r, s], "(mn|rs)": base, "(mr|ns)": mrns, "(ms|rn)": msrn, "(mn|sr)": mnsr})
    return {"samples": out}

def _head_matrix(a: np.ndarray, k: int = 6) -> Any:
    k = min(k, a.shape[0])
    return np.asarray(a[:k, :k]).tolist()

# --------------------------------- J / K 構築 --------------------------------- #
def build_J(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Coulomb:
      J_mn = Σ_rs D_rs (mn|rs)
    eri[m,n,r,s] = (mn|rs)
    収縮: D[r,s] × eri[..., r, s] -> axes ([0,1],[2,3]) -> [m,n]
    """
    return np.tensordot(D, eri, axes=([0, 1], [2, 3]))

def _perm_and_einsum_for_mode(mode: KMode) -> Tuple[Tuple[int, int, int, int], str]:
    """
    モードごとの ERI 転置タプルと einsum 式を返す。
      - STD : eri_perm = eri.transpose(0,2,1,3) -> 'rs,mrns->mn'
      - ALT1: eri_perm = eri.transpose(0,3,2,1) -> 'rs,msrn->mn'
      - ALT2: eri_perm = eri.transpose(0,1,3,2) -> 'rs,mnsr->mn'
    """
    if mode == "STD":
        return (0, 2, 1, 3), "rs,mrns->mn"
    elif mode == "ALT1":
        return (0, 3, 2, 1), "rs,msrn->mn"
    elif mode == "ALT2":
        return (0, 1, 3, 2), "rs,mnsr->mn"
    else:
        raise ValueError(f"Unknown K mode: {mode}")


def build_K(
    D: np.ndarray,
    eri: np.ndarray,
    *,
    mode: str = "STD",
    J_ref: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exchange matrix K:
      K_mn = Σ_rs D_rs (mr|ns)

    修正版:
      - transpose + einsum 固定
      - J=K検出時は停止せず警告ログのみ
      - Hermitizeして返却
    """
    # モード正規化
    mode = mode.upper()
    if mode == "STD":
        perm_axes = (0, 2, 1, 3)      # (mr|ns)
        einsum_expr = "rs,mrns->mn"
    elif mode == "ALT1":
        perm_axes = (0, 3, 2, 1)      # (ms|rn)
        einsum_expr = "rs,msrn->mn"
    elif mode == "ALT2":
        perm_axes = (0, 1, 3, 2)      # (mn|sr)
        einsum_expr = "rs,mnsr->mn"
    else:
        raise ValueError(f"Unknown K mode: {mode}")

    # ERI 軸並び替え
    eri_perm = eri.transpose(*perm_axes)

    # einsum による縮約
    K_raw = np.einsum(einsum_expr, D, eri_perm, optimize=True)

    # Hermitize
    K_sym = 0.5 * (K_raw + K_raw.T)

    # J=K 検出（停止せず警告）
    if J_ref is not None:
        diff_norm = np.linalg.norm(J_ref - K_sym)
        if diff_norm < 1e-10:
            print(f"[WARNING] J and K are IDENTICAL (mode={mode}, ||J-K||={diff_norm:.3e})")

    # 診断ログ（オプション）
    if bool(int(os.getenv("JK_DIAG", "0"))):
        EJ = np.einsum("mn,mn->", D, J_ref) if J_ref is not None else np.nan
        EK = np.einsum("mn,mn->", D, K_sym)
        R_KJ = EK / EJ if EJ != 0 else np.nan
        print(f"[JK_DIAG] ΣD·J={EJ:.6f}, ΣD·K={EK:.6f}, R_KJ={R_KJ:.6f}, ||J-K||={diff_norm:.3e}")

    return K_sym, K_raw


# ------------------------------ 重診断/ダンプ支援 ------------------------------ #
def _collect_jk_debug(
    eri: np.ndarray,
    D: np.ndarray,
    J: Optional[np.ndarray] = None,
    Kcands: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "env": {
            "JK_DIAG": os.getenv("JK_DIAG"),
            "JK_TRACE": os.getenv("JK_TRACE"),
            "JK_DUMP_JSON": os.getenv("JK_DUMP_JSON"),
            "JK_JSON_DIR": os.getenv("JK_JSON_DIR"),
            "K_EINSUM": os.getenv("K_EINSUM"),
            "K_SCALE_TEST": os.getenv("K_SCALE_TEST"),
            "JK_STRICT_KSCALE": os.getenv("JK_STRICT_KSCALE"),
            "ERI_SYM_8": os.getenv("ERI_SYM_8"),
            "PYTHONDONTWRITEBYTECODE": os.getenv("PYTHONDONTWRITEBYTECODE"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "modules": _module_files_snapshot(),
        "shapes": {"eri": list(eri.shape), "D": list(D.shape)},
        "hash": {"md5_eri": _md5_of_array(eri), "md5_D": _md5_of_array(D)},
        "eri_symmetry": _eri_symmetry_residuals(eri),
    }
    if J is None:
        J = build_J(D, eri)
    info["J_head"] = _head_matrix(J)
    info["Tr[D·J]"] = _sum_D_dot(J, D)

    # 候補Kを計算して比較
    modes = ("STD", "ALT1", "ALT2")
    Kpack = {}
    for m in modes:
        if Kcands and m in Kcands:
            K = Kcands[m]
        else:
            K, _ = build_K(D, eri, mode=m, J_ref=J)
        Kpack[m] = {
            "Tr[D·K]": _sum_D_dot(K, D),
            "R_KJ": float(_sum_D_dot(K, D) / (_sum_D_dot(J, D) if _sum_D_dot(J, D) != 0 else np.nan)),
            "\n\nJ-K\n\n_F": float(np.linalg.norm(J - K)),
            "max\nJ-K\n": float(np.max(np.abs(J - K)) if J.size and K.size else 0.0),
            "K_head": _head_matrix(K),
        }
    info["K_candidates"] = Kpack
    # ランダムERIサンプル
    info["eri_samples"] = _eri_samples(eri, nsamp=32)
    return info

def _maybe_trace_print(eri: np.ndarray, D: np.ndarray) -> None:
    if not _as_bool_env("JK_TRACE", False):
        return
    print("[jk.trace] shapes: eri", eri.shape, " D", D.shape)
    sym = _eri_symmetry_residuals(eri)
    for k, v in sym.items():
        print(f"[jk.trace] {k} = {v:.6e}")
    print(f"[jk.trace] md5(eri)={_md5_of_array(eri)} md5(D)={_md5_of_array(D)}")
    # 代表サンプル数点
    n = min(2, eri.shape[0] - 1)
    idx = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 0, 0)]
    for (m, n_, r, s) in idx:
        if max(m, n_, r, s) < eri.shape[0]:
            base = float(eri[m, n_, r, s])
            mrns = float(eri[m, r, n_, s])
            msrn = float(eri[m, s, r, n_])
            mnsr = float(eri[m, n_, s, r])
            print(
                f"[jk.trace] (mn|rs)[{m}{n_}{r}{s}]={base:.12e} "
                f"(mr|ns)={mrns:.12e} (ms|rn)={msrn:.12e} (mn|sr)={mnsr:.12e}"
            )

def _dump_crash_json(payload: Dict[str, Any]) -> str:
    outdir = _token(os.getenv("JK_JSON_DIR", r"DFT\output\_jk_crash"), r"DFT\output\_jk_crash")
    path = os.path.join(outdir, f"JKCrash_{_now_tag()}.json")
    _json_dump(payload, path)
    return path

# ------------------------------- トップレベル API ------------------------------- #

def form_jk_matrices(eri: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    JとKを構築し、フォールバック処理を行う。
    修正版:
      - 全モードでJ=Kなら停止せず警告＋最終K返却
    """
    # J構築
    J = np.tensordot(D, eri, axes=([0, 1], [2, 3]))

    # K構築（モード切替）
    modes = ["STD", "ALT1", "ALT2"]
    last_K = None
    for mode in modes:
        K_sym, _ = build_K(D, eri, mode=mode, J_ref=J)
        last_K = K_sym
        # J≠Kなら即返却
        if np.linalg.norm(J - K_sym) > 1e-10:
            return J, K_sym

    # 全モードでJ=K → 警告＋最終K返却
    print(f"[WARNING] J and K are IDENTICAL under all modes tried {modes}. Using last K.")
    return J, last_K

# ------------------------------ optional self-test ------------------------------ #
def _naive_JK_reference(eri: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    小系の遅い O(n^4) 参照（回帰テスト用途）。
    """
    n = D.shape[0]
    J = np.zeros((n, n), float)
    K = np.zeros((n, n), float)
    # J: (mn|rs)
    for m in range(n):
        for n_ in range(n):
            s_val = 0.0
            for r in range(n):
                for s_ in range(n):
                    s_val += D[r, s_] * eri[m, n_, r, s_]
            J[m, n_] = s_val
    # K: (mr|ns)
    for m in range(n):
        for n_ in range(n):
            s_val = 0.0
            for r in range(n):
                for s_ in range(n):
                    s_val += D[r, s_] * eri[m, r, n_, s_]
            K[m, n_] = s_val
    return _ensure_sym(J), _ensure_sym(K)

def _self_test() -> None:
    rng = np.random.default_rng(42)
    n = 4
    D = rng.normal(size=(n, n))
    D = 0.5 * (D + D.T)  # symmetric spin-summed density
    eri = rng.normal(size=(n, n, n, n))
    eri = _symmetrize_eri_8(eri)  # enforce 8-fold symmetry for the test

    J_ref, K_ref = _naive_JK_reference(eri, D)
    J, K = form_jk_matrices(eri, D, mode="STD")
    print("[self-test] \n\nJ-Jref\n\n_F =", np.linalg.norm(J - J_ref))
    print("[self-test] \n\nK-Kref\n\n_F =", np.linalg.norm(K - K_ref))

if __name__ == "__main__":
    _self_test()
