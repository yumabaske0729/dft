# DFT/scf/functional.py
# -*- coding: utf-8 -*-
"""
Thin wrapper to expose XC functionals with backward-compatible names.
"""

from .xc.hybrid import B3LYP  # keep old import path working
__all__ = ["B3LYP"]
