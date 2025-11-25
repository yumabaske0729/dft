# DFT/scf/xc/lda.py
# -*- coding: utf-8 -*-
import numpy as np
from .utils import safe_pow

# Dirac–Slater exchange (per electron & potential) — closed shell
def slater_exchange_density(rho: np.ndarray) -> np.ndarray:
    Cx = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    return Cx * safe_pow(rho, 1.0 / 3.0)

def slater_exchange_potential(eps_x: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * eps_x

# VWN correlation (LDA) — will be added in next patch (for full B3LYP correlation mixing).
# def vwn_c_density(rho: np.ndarray) -> np.ndarray: ...
# def vwn_c_potential(...): ...
