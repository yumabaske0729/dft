import numpy as np
from math import sqrt

class ContractedGTO:
    """
    Contracted Gaussian Type Orbital (Cartesian).
    Attributes
    ---------
    shell : str
        Shell label (e.g., 'S', 'PX', 'PY', 'PZ', 'DXX', ...)
    exponents : list[float]
        Primitive Gaussian exponents (α).
    coefficients : list[float]
        Contraction coefficients (c).
    atom_index : int
        Index of the atom this GTO belongs to.
    center : np.ndarray
        3D coordinates of the GTO center (in Bohr).
    l, m, n : int
        Cartesian angular momentum exponents for x^l y^m z^n.
    """

    def __init__(self, shell: str, exponents: list, coefficients: list,
                 atom_index: int, center: np.ndarray):
        self.shell = shell.upper()
        self.exponents = list(map(float, exponents))
        self.coefficients = list(map(float, coefficients))
        self.atom_index = atom_index
        self.center = np.array(center, dtype=float)
        self.l, self.m, self.n = self._angular_momentum_from_shell(self.shell)

    def _angular_momentum_from_shell(self, shell: str):
        mapping = {
            'S':  (0, 0, 0),
            'PX': (1, 0, 0), 'PY': (0, 1, 0), 'PZ': (0, 0, 1),
            'DXX': (2, 0, 0), 'DYY': (0, 2, 0), 'DZZ': (0, 0, 2),
            'DXY': (1, 1, 0), 'DXZ': (1, 0, 1), 'DYZ': (0, 1, 1),
        }
        if shell in mapping:
            return mapping[shell]
        raise ValueError(f"Unsupported shell type: {shell}")

    def __repr__(self):
        return (
            f"GTO(shell={self.shell}, atom={self.atom_index}, "
            f"center={self.center.tolist()})"
        )
