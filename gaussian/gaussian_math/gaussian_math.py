#gaussian_math.py

import math
from functools import lru_cache
from typing import Union

@lru_cache(maxsize=None)
def double_factorial(n: int) -> int:
    """
    Compute the double factorial n!!.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    int
        n!! = n * (n-2) * (n-4) * ... * (1 or 2).
    """
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result

def norm_prefactor(l: int, m: int, n: int, alpha: float) -> float:
    """
    Normalization constant for a primitive Gaussian orbital with
    angular momentum (l, m, n) and exponent alpha.

    N = (2α/π)^(3/4) * [ (4α)^(l+m+n) / ( (2l-1)!! (2m-1)!! (2n-1)!! ) ]^1/2

    Parameters
    ----------
    l, m, n : int
    alpha : float

    Returns
    -------
    float
    """
    if any(q < 0 for q in (l, m, n)):
        raise ValueError("Angular momentum quantum numbers must be non-negative.")
    if alpha <= 0.0:
        raise ValueError("Exponent alpha must be positive.")

    pre = (2.0 * alpha / math.pi) ** 0.75
    denom = double_factorial(2*l - 1) * double_factorial(2*m - 1) * double_factorial(2*n - 1)
    lmn = ((4.0 * alpha) ** (l + m + n) / denom) ** 0.5
    return pre * lmn
