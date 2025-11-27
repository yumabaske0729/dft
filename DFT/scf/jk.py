# DFT/scf/jk.py
# -*- coding: utf-8 -*-
"""
JK builder（closed-shell, spin-summed density D = 2P）
- 軸順序の不一致に対して、RAW/STD/ALT1/ALT2 の候補を試す。
- 物理指標（E_2e = 0.5 Tr[D (J - αK)] の最小化）を優先して K を選択。
  * RHF: α = 0.5
  * DFT: α = a0（Fock側で -a0*K を引くため）
- H_core / a0 が未提供なら、従来の「||J-K||_F が最大」をフォールバック選択。
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple

def build_J(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    # eri[m,n,r,s] = (mn|rs) を前提にした Coulomb
    J = np.tensordot(D, eri, axes=([0,1],[2,3]))
    return 0.5 * (J + J.T)

def _k_raw(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    K = np.einsum('rs,mrns->mn', D, eri, optimize=True)
    return 0.5 * (K + K.T)

def _k_std(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    eri_perm = eri.transpose(0,2,1,3)  # (m,r,n,s)
    K = np.einsum('rs,mrns->mn', D, eri_perm, optimize=True)
    return 0.5 * (K + K.T)

def _k_alt1(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    eri_perm = eri.transpose(0,3,2,1)  # (m,s,r,n)
    K = np.einsum('rs,msrn->mn', D, eri_perm, optimize=True)
    return 0.5 * (K + K.T)

def _k_alt2(D: np.ndarray, eri: np.ndarray) -> np.ndarray:
    eri_perm = eri.transpose(0,1,3,2)  # (m,n,s,r)
    K = np.einsum('rs,mnsr->mn', D, eri_perm, optimize=True)
    return 0.5 * (K + K.T)

def form_jk_matrices(
    eri: np.ndarray,
    D: np.ndarray,
    H_core: Optional[np.ndarray] = None,
    a0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    軸順序に依存しない安全な J/K 構築。
    優先選択: E_2e = 0.5 * Tr[D (J - αK)] 最小（RHF: α=0.5、DFT: α=a0）
    フォールバック: ||J - K||_F 最大
    """
    J = build_J(D, eri)
    candidates = {
        "RAW" : _k_raw(D, eri),
        "STD" : _k_std(D, eri),
        "ALT1": _k_alt1(D, eri),
        "ALT2": _k_alt2(D, eri),
    }

    # J==K ガード用
    def _is_bad(Jm: np.ndarray, Km: np.ndarray) -> bool:
        sep = float(np.linalg.norm(Jm - Km))
        return (sep < 1e-10) and (np.linalg.norm(Jm) > 1e-10)

    # 物理指標が使えるならそれを優先して選択
    alpha: Optional[float] = None
    if H_core is not None:
        # RHF: α=0.5, DFT: α=a0（a0が None の場合はRHFと同様に0.5）
        alpha = 0.5 if (a0 is None) else float(a0)

    if alpha is not None:
        best_name: Optional[str] = None
        best_score = +np.inf  # E_2e の最小化
        for name, Kcand in candidates.items():
            if _is_bad(J, Kcand):
                # J==K に近い候補は避ける
                continue
            # E_2e only（E_one は候補間で不変なので比較不要）
            E2e = float(0.5 * np.einsum('mn,mn->', D, (J - alpha * Kcand)))
            if E2e < best_score:
                best_score = E2e
                best_name = name

        if best_name is None:
            # すべて bad ならフォールバック
            diffs = {n: float(np.linalg.norm(J - K)) for n, K in candidates.items()}
            best_name = max(diffs, key=diffs.get)

        K = candidates[best_name]
        if bool(int(os.getenv("JK_DIAG", "0"))):
            EJ = float(np.einsum('mn,mn->', D, J))
            EK = float(np.einsum('mn,mn->', D, K))
            print(f"[JK] pick={best_name} (alpha={alpha:.3f}), ΣD·J={EJ:.12f}, ΣD·K={EK:.12f}")
        return J, K

    # ---- フォールバック（旧来の選択） ----
    diffs = {name: float(np.linalg.norm(J - K)) for name, K in candidates.items()}
    best = max(diffs, key=diffs.get)
    K = candidates[best]
    if bool(int(os.getenv("JK_DIAG", "0"))):
        EJ = float(np.einsum("mn,mn->", D, J))
        EK = float(np.einsum("mn,mn->", D, K))
        print(f"[JK] pick={best}, ||J-K||_F={diffs[best]:.3e}, ΣD·J={EJ:.12f}, ΣD·K={EK:.12f}")
    return J, K