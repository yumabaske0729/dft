"""
math パッケージ: Gaussian積分・数値手法用の数値ライブラリ群
"""

# ガウス関数の正規化・双階乗など
from .gaussian_math import norm_prefactor, double_factorial


# 他の数値ツールを追加する場合はここにまとめる（例）
# from .grid_utils import generate_grid_points

# 必要なら __all__ で明示的に制限することも可能（任意）
__all__ = [
    "norm_prefactor",
    "double_factorial",
    "DIISManager",
    # "generate_grid_points",  # 追加時
]
