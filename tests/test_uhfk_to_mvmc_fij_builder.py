"""Tests for fij_builder (spec section 5.1 / 3.4)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pytest

from tools._uhfk_to_mvmc.fij_builder import (
    build_amplitudes,
    build_fij_phys,
)


def _klist(n):
    return np.roll(np.arange(n) - (n // 2), -(n // 2))


def test_build_amplitudes_pbc_l4_single_orbital_freeparticle():
    """Free particles, PBC L=4, single orbital, half filling (Ne_up = 2).
    Expect A_up to be 2 orthonormal plane waves at occupied k's."""
    L = 4
    norb = 1
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array(
        [[i, 0, 0] for i in range(L)], dtype=np.float64
    )
    theta = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)
    k_vals = 2.0 * np.pi * wv[:, 0] / L
    ws = -2.0 * np.cos(k_vals)
    eigenvalue = np.stack([ws, ws], axis=1)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0

    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)
    pytest.skip(
        "L=4 half filling is degenerate; covered by quarter-filling test "
        "in Step 5"
    )


def test_build_amplitudes_pbc_l8_quarter_filling_freeparticle():
    """Free particles, PBC L=8, single orbital, Ne=1 (only k=0 occupied)."""
    L = 8
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array(
        [[i, 0, 0] for i in range(L)], dtype=np.float64
    )
    theta = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)

    k_vals = 2.0 * np.pi * wv[:, 0] / L
    ws = -2.0 * np.cos(k_vals)
    eigenvalue = np.stack([ws, ws], axis=1)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0
    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)
    stepped_occupation = np.zeros((L, 2), dtype=np.float64)
    n_k0 = list(wv[:, 0]).index(0)
    stepped_occupation[n_k0, 0] = 1.0
    stepped_occupation[n_k0, 1] = 1.0

    A_up, A_down = build_amplitudes(
        wavevector_index=wv,
        eigenvector=eigenvector,
        stepped_occupation=stepped_occupation,
        column_spin=column_spin,
        column_mu_group=column_mu_group,
        site_positions=site_positions,
        norb_orig=1,
        theta=theta,
        L=L_vec,
    )

    assert A_up.shape == (L, 1)
    assert A_down.shape == (L, 1)
    expected = np.full(L, 1.0 / np.sqrt(L), dtype=np.complex128)
    np.testing.assert_allclose(A_up[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(A_down[:, 0], expected, atol=1e-12)


def test_build_fij_phys_pbc_l8_k0_only():
    """For Ne=1 (k=0 only) PBC, F_{ij} = 1/L for all (i, j)."""
    L = 8
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array(
        [[i, 0, 0] for i in range(L)], dtype=np.float64
    )
    theta = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0
    stepped_occupation = np.zeros((L, 2), dtype=np.float64)
    n_k0 = list(wv[:, 0]).index(0)
    stepped_occupation[n_k0, 0] = 1.0
    stepped_occupation[n_k0, 1] = 1.0
    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)

    A_up, A_down = build_amplitudes(
        wavevector_index=wv, eigenvector=eigenvector,
        stepped_occupation=stepped_occupation, column_spin=column_spin,
        column_mu_group=column_mu_group,
        site_positions=site_positions, norb_orig=1, theta=theta, L=L_vec,
    )
    F = build_fij_phys(A_up, A_down)

    np.testing.assert_allclose(
        F, np.full((L, L), 1.0 / L, dtype=np.complex128), atol=1e-12
    )


def test_build_amplitudes_rejects_open_pair_closure():
    """If the up occupied set's (k, -k) partner does not equal the down
    occupied set, build_amplitudes must raise (Codex spec review fix 3:
    magnetic / spin-dependent occupations cannot be silently encoded by
    the v1 (k, -k) construction)."""
    import pytest

    L = 8
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array(
        [[i, 0, 0] for i in range(L)], dtype=np.float64
    )
    theta = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0
    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)

    # Up occupies tilde_k = 0 (row 0); partner under PBC is row 0 itself
    # (self-pair). Down occupies tilde_k = 1 (row 1) instead of the
    # self-pair: this breaks (k, -k) closure.
    stepped_occupation = np.zeros((L, 2), dtype=np.float64)
    n_k0 = list(wv[:, 0]).index(0)
    n_k1 = list(wv[:, 0]).index(1)
    stepped_occupation[n_k0, 0] = 1.0
    stepped_occupation[n_k1, 1] = 1.0

    with pytest.raises(ValueError, match="pair-closure"):
        build_amplitudes(
            wavevector_index=wv, eigenvector=eigenvector,
            stepped_occupation=stepped_occupation, column_spin=column_spin,
            column_mu_group=column_mu_group,
            site_positions=site_positions, norb_orig=1,
            theta=theta, L=L_vec,
        )
