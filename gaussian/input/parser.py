# DFT/input/parser.py

import numpy as np
from ..utils.constants import BOHR_TO_ANGSTROM

class Atom:
    def __init__(self, symbol: str, position: np.ndarray):
        self.symbol = symbol
        self.position = position

class Molecule:
    def __init__(self, atoms: list, charge: int = 0, multiplicity: int = 1):
        self.atoms = atoms
        self.charge = charge
        self.multiplicity = multiplicity
    
    def get_positions(self) -> np.ndarray:
        """Returns an (N,3) array of atomic positions."""
        return np.array([atom.position for atom in self.atoms])

    def get_symbols(self) -> list:
        """Returns a list of atomic symbols."""
        return [atom.symbol for atom in self.atoms]

def parse_xyz(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    atoms = []
    with open(filename) as f:
        lines = f.readlines()

    try:
        num_atoms = int(lines[0].strip())
    except:
        raise ValueError("First line of XYZ must be number of atoms.")

    atom_lines = lines[2:2 + num_atoms]
    if len(atom_lines) < num_atoms:
        raise ValueError("Not enough atom lines.")

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line: {line!r}")
        symbol = parts[0]
        # Å → Bohr 変換
        coords_ang = [float(v) for v in parts[1:4]]
        coords_bohr = [x / BOHR_TO_ANGSTROM for x in coords_ang]
        x, y, z = coords_bohr
        atoms.append(Atom(symbol, np.array([x, y, z])))

    return Molecule(atoms, charge, multiplicity)
