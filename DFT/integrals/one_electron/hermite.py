#integrals/one_electron/hermite.py
import numpy as np

def hermite_coefficients(l1, l2, PAx, PBx, alpha1, alpha2, max_t):
    """
    1D Hermite係数（オーバーラップ用）の簡易再帰。
    ※ PBx は本再帰では使用しません（将来拡張用に引数は維持）
    """
    mu = alpha1 + alpha2
    diff = PAx - PBx  # 将来拡張用（現行式では未使用）
    E = [0.0] * (max_t + 1)

    # base
    E[0] = np.exp(-alpha1 * alpha2 * (0.0) / mu)  # diffは0扱い（オーバーラップ用の簡略化）
    if max_t >= 1:
        E[1] = PAx * E[0]

    for t in range(2, max_t + 1):
        E[t] = PAx * E[t - 1] + (t - 1) / (2.0 * mu) * E[t - 2]

    return E
