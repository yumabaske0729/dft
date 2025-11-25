# DFT/scf/xc/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import numpy as np

class XCFunctional:
    """
    Base interface for grid-based XC functionals.
    """

    def evaluate_exchange_correlation(self) -> Tuple[float, np.ndarray]:
        """
        Returns
        -------
        Exc : float
        Vxc : (nbf,nbf) ndarray
        """
        raise NotImplementedError
