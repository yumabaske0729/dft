# DFT/utils/constants.py

# 長さ変換定数
ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

# 元素記号から原子番号へのマッピング
ATOMIC_NUMBERS = {
    'H': 1,  'He': 2, 'Li': 3,  'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,  'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36
    # 必要に応じて追加可能
}

def get_atomic_number(symbol: str) -> int:
    """
    原子記号から原子番号を取得する関数。
    
    Args:
        symbol (str): 原子記号（例: 'H', 'C'）
    
    Returns:
        int: 原子番号

    Raises:
        ValueError: 未知の原子記号の場合
    """
    try:
        return ATOMIC_NUMBERS[symbol]
    except KeyError:
        raise ValueError(f"Unknown atomic symbol: {symbol}")
