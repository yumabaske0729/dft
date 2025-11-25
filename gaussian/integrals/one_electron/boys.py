# DFT/integrals/one_electron/boys.py
# -*- coding: utf-8 -*-
import numpy as np
from math import erf, sqrt, pi

def boys_function(n: int, t):
    """
    Boys function F_n(t)
    - small t: Taylor expansion around 0
    - big t: upward recursion from F0
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    ta = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(ta)
    small = ta < 1.0e-8
    if np.any(small):
        x = ta[small]
        out[small] = (
            1.0 / (2.0 * n + 1.0)
            - x / (2.0 * n + 3.0)
            + 0.5 * x * x / (2.0 * n + 5.0)
        )
    big = ~small
    if np.any(big):
        tb = ta[big]
        rt = np.sqrt(tb)
        Fk = 0.5 * np.sqrt(pi) * erf(rt) / np.where(rt == 0.0, 1.0, rt)
        if n > 0:
            et = np.exp(-tb)
            for k in range(1, n + 1):
                Fk = (((2.0 * k - 1.0) * Fk) - et) / (2.0 * tb)
        out[big] = Fk
    if np.isscalar(t):
        return float(out[0])
    return out
