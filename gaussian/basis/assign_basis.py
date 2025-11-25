# DFT/basis/assign_basis.py
# -*- coding: utf-8 -*-
"""
基底関数の割当て：
- Pople系に典型的な SP 共有殻（angular_momentum = [0,1] かつ coefficients = [S_coeffs, P_coeffs]）
  に“正しく”対応。
- 収縮GTO（ContractedGTO）を 1 関数ずつ ⟨χ|χ⟩=1 になるように厳密規格化。
  （基底データ側が「正規化済み／未正規化」のどちらであっても、最終的に AO の自己重なりが 1 になります）
"""

from typing import List
from .load_basis import BasisLoader
from .contracted_gto import ContractedGTO
from ..utils.normalization import contracted_norm

def _labels_for_l(l: int) -> List[str]:
    """
    角運動量 l に対する Cartesian AO ラベルを返す。
    l=0: S
    l=1: P (PX,PY,PZ)
    l=2: D (6 個: DXX, DYY, DZZ, DXY, DXZ, DYZ)
    """
    if l == 0:
        return ["S"]
    if l == 1:
        return ["PX", "PY", "PZ"]
    if l == 2:
        return ["DXX", "DYY", "DZZ", "DXY", "DXZ", "DYZ"]
    raise ValueError(f"Unsupported angular momentum l={l} (only S/P/D(cartesian-6) supported).")

def assign_basis_functions(molecule, basis_name: str) -> List[ContractedGTO]:
    """
    分子の各原子に対し、指定の基底（JSON）から収縮GTOを構築して割当てる。

    対応仕様
    --------
    1) 単一角運動量殻（例: {"angular_momentum":[0], "exponents":[...], "coefficients":[[...]]}）
       -> l に応じて S / P(3本) / D(6本) を展開
    2) SP共有殻（例: {"angular_momentum":[0,1], "exponents":[...], "coefficients":[ [S_coeffs],[P_coeffs] ]}）
       -> S を1本、P を3本（PX,PY,PZ）で展開（SとPで係数リストを使い分け）
    3) 最終的に全AOを “自己重なり=1” に規格化（⟨χ|χ⟩=1）

    Parameters
    ----------
    molecule : Molecule
        input.parser.Molecule（atoms: List[Atom]）
    basis_name : str
        "sto-3g", "6-31g", "6-31g(d)" 等（拡張子や大小文字は Loader 側で処理）

    Returns
    -------
    List[ContractedGTO]
        収縮GTO（Cartesian）のリスト（規格化済み）
    """
    atoms = molecule.atoms
    loader = BasisLoader()
    basis_data = loader.load(basis_name)

    basis_functions: List[ContractedGTO] = []

    for atom_index, atom in enumerate(atoms):
        symbol = atom.symbol
        if symbol not in basis_data:
            raise ValueError(
                f"Basis data for element '{symbol}' not found in '{basis_name}' basis set."
            )

        elem = basis_data[symbol]
        shells = elem.get("shells", [])

        for sh in shells:
            # angular_momentum は [0] / [1] / [2] など、あるいは [0,1] (SP共有殻) のケース
            ang_list = sh.get("angular_momentum", [0])
            # 指数リスト
            exps = list(map(float, sh.get("exponents", [])))
            # 係数は [[...]] か [...]、SP共有殻では [S_coeffs, P_coeffs] の二重配列
            coeffs_nested = sh.get("coefficients", [])

            # --- 1) 単一角運動量殻 ---
            if isinstance(ang_list, list) and len(ang_list) == 1:
                l = int(ang_list[0])

                # 係数のフラット化（[[...]] or [...] の両対応）
                if len(coeffs_nested) > 0 and isinstance(coeffs_nested[0], list):
                    coeffs = list(map(float, coeffs_nested[0]))
                else:
                    coeffs = list(map(float, coeffs_nested))

                labels = _labels_for_l(l)
                for lab in labels:
                    basis_functions.append(
                        ContractedGTO(
                            shell=lab,
                            exponents=exps,
                            coefficients=coeffs,
                            atom_index=atom_index,
                            center=atom.position,
                        )
                    )

            # --- 2) SP 共有殻（S と P が同一指数で係数だけ別）---
            elif isinstance(ang_list, list) and sorted(ang_list) == [0, 1]:
                # 係数は [S_coeffs, P_coeffs] の二重配列が必須
                if not (
                    len(coeffs_nested) >= 2
                    and isinstance(coeffs_nested[0], list)
                    and isinstance(coeffs_nested[1], list)
                ):
                    raise ValueError(
                        f"SP shell for {symbol} must provide two coefficient lists (S and P)."
                    )

                s_coeffs = list(map(float, coeffs_nested[0]))
                p_coeffs = list(map(float, coeffs_nested[1]))

                # --- S 成分 ---
                basis_functions.append(
                    ContractedGTO(
                        shell="S",
                        exponents=exps,
                        coefficients=s_coeffs,
                        atom_index=atom_index,
                        center=atom.position,
                    )
                )
                # --- P 成分（PX, PY, PZ の3本）---
                for lab in ["PX", "PY", "PZ"]:
                    basis_functions.append(
                        ContractedGTO(
                            shell=lab,
                            exponents=exps,
                            coefficients=p_coeffs,
                            atom_index=atom_index,
                            center=atom.position,
                        )
                    )

            else:
                # 未対応の角運動量指定
                raise ValueError(
                    f"Unsupported angular_momentum spec {ang_list} for atom {symbol} (basis {basis_name})."
                )

    # === 3) 収縮GTOの厳密規格化（⟨χ|χ⟩=1 に強制）========================
    for g in basis_functions:
        scale = contracted_norm(g)  # = 1 / sqrt(<g|g>)
        g.coefficients = [c * scale for c in g.coefficients]

    return basis_functions
