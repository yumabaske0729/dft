# -*- coding: utf-8 -*-
"""
Unit tests for DFT.scf.jk (JK builders for spin-summed density D = 2P).

What we check:
  1) STD と ALT1 の完全一致（J/K ともに数値誤差内で同一）
  2) ALT2 は通常は一致しない（検証用モード）
  3) J/K のエルミート性（対称性）
  4) E_2e の一貫性:
       E_2e = 0.5 * Tr[D * (J - 0.5 K)]
            = 0.5 * (sum(D·J) - 0.5 * sum(D·K))
     また STD と ALT1 で E_2e が等しいこと
  5) 環境変数の挙動:
       - K_SCALE_TEST: K がスケールされる
       - K_EINSUM    : env 指定と mode 明示が一致する

注: ERI はランダムテンソルを 8 重対称化して模擬します。
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from DFT.scf.jk import form_jk_matrices, build_J, build_K


def _symmetrize_eri(eri: np.ndarray) -> np.ndarray:
    """
    8-fold symmetry:
      (mn|rs) = (nm|rs) = (mn|sr) = (nm|sr) = (rs|mn) = ...
    """
    e = eri
    e_sym = (
        e
        + e.transpose(1, 0, 2, 3)
        + e.transpose(0, 1, 3, 2)
        + e.transpose(1, 0, 3, 2)
        + e.transpose(2, 3, 0, 1)
        + e.transpose(3, 2, 0, 1)
        + e.transpose(2, 3, 1, 0)
        + e.transpose(3, 2, 1, 0)
    ) / 8.0
    return e_sym


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20251002)


def _make_random_D_eri(rng, nbf: int = 5):
    # 対称 D（スピン和密度候補）
    D = rng.standard_normal((nbf, nbf))
    D = 0.5 * (D + D.T)

    # ランダム ERI を 8 重対称化
    eri = rng.standard_normal((nbf, nbf, nbf, nbf))
    eri = _symmetrize_eri(eri)
    return D, eri


def test_std_alt1_equivalence_random(rng):
    D, eri = _make_random_D_eri(rng, nbf=6)

    J_std, K_std = form_jk_matrices(eri, D, mode="STD")
    J_a1 , K_a1  = form_jk_matrices(eri, D, mode="ALT1")

    np.testing.assert_allclose(J_std, J_a1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(K_std, K_a1, rtol=1e-12, atol=1e-12)


def test_alt2_is_different_in_general(rng):
    D, eri = _make_random_D_eri(rng, nbf=5)

    K_std  = build_K(D, eri, mode="STD")
    K_alt2 = build_K(D, eri, mode="ALT2")

    diffF = np.linalg.norm(K_std - K_alt2)
    assert diffF > 1e-6  # 検証用モード ALT2 は一般には一致しない


def test_hermiticity_of_JK(rng):
    D, eri = _make_random_D_eri(rng, nbf=5)

    J, K = form_jk_matrices(eri, D, mode="STD")
    np.testing.assert_allclose(J, J.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-12)


def test_two_e_energy_identity_and_mode_equivalence(rng):
    """
    E_2e = 0.5 * Tr[D * (J - 0.5 K)]
         = 0.5 * ( Tr[D·J] - 0.5 * Tr[D·K] )
    また、STD と ALT1 で同じ値になる。
    """
    D, eri = _make_random_D_eri(rng, nbf=6)

    # STD
    J_s, K_s = form_jk_matrices(eri, D, mode="STD")
    E2e_s_1 = 0.5 * float(np.einsum("mn,mn->", D, (J_s - 0.5 * K_s)))
    EJ_s = float(np.einsum("mn,mn->", D, J_s))
    EK_s = float(np.einsum("mn,mn->", D, K_s))
    E2e_s_2 = 0.5 * (EJ_s - 0.5 * EK_s)
    np.testing.assert_allclose(E2e_s_1, E2e_s_2, rtol=1e-12, atol=1e-12)

    # ALT1 でも同一
    J_a, K_a = form_jk_matrices(eri, D, mode="ALT1")
    E2e_a = 0.5 * float(np.einsum("mn,mn->", D, (J_a - 0.5 * K_a)))
    np.testing.assert_allclose(E2e_s_1, E2e_a, rtol=1e-12, atol=1e-12)


def test_k_scale_env_applies(monkeypatch, rng):
    """
    K_SCALE_TEST が設定されると K がスケールされる。
    """
    D, eri = _make_random_D_eri(rng, nbf=4)

    # ベースライン（env 未設定）
    monkeypatch.delenv("K_SCALE_TEST", raising=False)
    K_base = build_K(D, eri, mode="STD")

    # 1.50 倍に設定
    monkeypatch.setenv("K_SCALE_TEST", "1.50")
    K_scaled = build_K(D, eri, mode="STD")

    # スケール検証（ノルム比で十分）
    ratio = np.linalg.norm(K_scaled) / np.linalg.norm(K_base)
    assert np.isclose(ratio, 1.5, rtol=1e-12, atol=1e-12)


def test_mode_env_equivalence(monkeypatch, rng):
    """
    K_EINSUM=ALT1 を環境変数で与えた場合と、
    明示 mode="ALT1" の form_jk_matrices が一致すること。
    """
    D, eri = _make_random_D_eri(rng, nbf=5)

    # env 指定 ALT1
    monkeypatch.setenv("K_EINSUM", "ALT1")
    J_env, K_env = form_jk_matrices(eri, D, mode=None)  # env を読む

    # 明示 ALT1
    J_a1, K_a1 = form_jk_matrices(eri, D, mode="ALT1")

    np.testing.assert_allclose(J_env, J_a1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(K_env, K_a1, rtol=1e-12, atol=1e-12)