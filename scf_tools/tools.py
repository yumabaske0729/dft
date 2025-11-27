
# -*- coding: utf-8 -*-
# scf_tools/tools.py
# 異常検出（MAD）、Infinity/NaN健全化、SのSPDクリップ＋Löwdin直交化、
# ロバストDIIS（Huber重み）、電子数/同定射性/直交性チェック

import numpy as np
from dataclasses import dataclass
from typing import List

# --- 異常検出ユーティリティ ---

def sanitize_de(values: List[float]) -> List[float]:
    """dEのInfinity/非有限値をNaNにして統計から除外できるようにする"""
    out = []
    for x in values:
        if isinstance(x, str) and x.lower() == 'infinity':
            out.append(np.nan)
        elif isinstance(x, float) and not np.isfinite(x):
            out.append(np.nan)
        else:
            out.append(x)
    return out

def mad_outliers(series: List[float], thresh: float = 3.5) -> List[int]:
    """MADベース外れ値インデックス（非有限値は除外）"""
    y = np.array([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    if y.size == 0: return []
    m  = np.median(y)
    mad = np.median(np.abs(y - m))
    if mad == 0: return []
    z  = 0.6745 * (y - m) / mad
    return list(np.where(np.abs(z) > thresh)[0])

# --- Sの直交化（Löwdin）＋SPD化 ---

@dataclass
class OrthoResult:
    X: np.ndarray        # S^{-1/2}
    S_spd: np.ndarray    # 数値的にSPD化したS
    evals: np.ndarray    # クリップ後の固有値
    U: np.ndarray        # 固有ベクトル

def lowdin(S: np.ndarray, eps_clip: float = 1e-12) -> OrthoResult:
    """Löwdin対称直交化。SをSPD化し、X=S^{-1/2}を返す"""
    S_sym = 0.5 * (S + S.T)
    evals, U = np.linalg.eigh(S_sym)
    w = np.clip(evals, eps_clip, None)
    S_spd = U @ np.diag(w) @ U.T
    X     = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    return OrthoResult(X=X, S_spd=S_spd, evals=w, U=U)

# --- ロバストDIIS（Huber重み付き） ---

def huber_weight(residual: np.ndarray, delta: float) -> float:
    norm = np.linalg.norm(residual)
    if norm <= delta or norm == 0.0:
        return 1.0
    return float(delta / norm)

@dataclass
class DIISState:
    focks: List[np.ndarray]
    errors: List[np.ndarray]
    max_store: int = 8
    delta: float = 1e-2  # Huber折れ点

    def push(self, F: np.ndarray, E: np.ndarray):
        self.focks.append(F.copy())
        self.errors.append(E.copy())
        if len(self.focks) > self.max_store:
            self.focks.pop(0); self.errors.pop(0)

    def extrapolate(self) -> np.ndarray:
        m = len(self.errors)
        if m == 0:
            raise RuntimeError("DIIS: errors are empty")
        B = np.zeros((m+1, m+1))
        b = np.zeros(m+1); b[-1] = 1.0
        w = np.array([huber_weight(e, self.delta) for e in self.errors], dtype=float)
        for i in range(m):
            for j in range(m):
                B[i,j] = w[i] * float(np.vdot(self.errors[i], self.errors[j])) * w[j]
        for i in range(m):
            B[i, m] = 1.0; B[m, i] = 1.0
        coeff = np.linalg.solve(B, b)[:-1]
        coeff = coeff * w  # 重み再スケール
        F_new = np.zeros_like(self.focks[0])
        for c, F in zip(coeff, self.focks):
            F_new += c * F
        return F_new

# --- 物理整合チェック ---

def electron_count(D: np.ndarray, S: np.ndarray) -> float:
    return float(np.trace(D @ S))  # Tr(D S) = Ne（Mulliken）

def idempotency_error(D: np.ndarray, S: np.ndarray) -> float:
    return float(np.linalg.norm(D @ S @ D - 2.0 * D))  # 閉殻RHF

def orthogonality_error(C: np.ndarray, S: np.ndarray) -> float:
    dev = C.T @ S @ C - np.eye(C.shape[1])
    return float(np.linalg.norm(dev))

def symmetry_maxabs(A: np.ndarray) -> float:
    return float(np.max(np.abs(A - A.T)))
