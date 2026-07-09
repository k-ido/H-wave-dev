"""Verify F has subtract (r_j - r_i) phase under APBC and the density check
sees the matching positive-theta phase for G - these must NOT be confused.

Background: Codex's APBC review (post-Task-11) showed that the original
add-add (r_i + r_j) formulation in spec section 3.4 came from the
operator-level F coefficient transformation, but the F that mVMC must
read is the amplitude product F = A_up @ A_down.T of the physical
(k, -k) pair, which carries SUBTRACT structure exp(+i k_phys (r_j - r_i))
once theta and tilde_k are combined consistently. The bridge implementation
follows that subtract convention so the per-spin density matches H-wave's
``_save_greenone`` output (G_phys = exp(+i theta (r_i - r_j) / L) G_tilde).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from tools._uhfk_to_mvmc.fij_builder import build_amplitudes, build_fij_phys


def _klist(n):
    return np.roll(np.arange(n) - (n // 2), -(n // 2))


def test_apbc_l8_fij_carries_subtract_phase():
    """For APBC L=8, single occupied (k, -k) pair with v=1, F_phys carries
    exp(+i k_phys (r_j - r_i)) and reproduces the physical density. With
    tilde_k=0 occupied (k_phys = theta/L = pi/L for APBC), F[i, j]
    = (1/L) exp(+i pi (r_j - r_i) / L)."""
    L = 8
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array([[i, 0, 0] for i in range(L)], dtype=np.float64)
    theta = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0
    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)

    stepped_occupation = np.zeros((L, 2), dtype=np.float64)
    # Occupy tilde_k = 0 on the up side; partner is tilde_k = -1 row in _klist.
    n0 = list(wv[:, 0]).index(0)
    stepped_occupation[n0, 0] = 1.0
    n_partner = list(wv[:, 0]).index(-1)
    stepped_occupation[n_partner, 1] = 1.0

    A_up, A_down = build_amplitudes(
        wv, eigenvector, stepped_occupation, column_spin, column_mu_group,
        site_positions, norb_orig=1, theta=theta, L=L_vec,
    )
    F = build_fij_phys(A_up, A_down)

    for i in range(L):
        for j in range(L):
            # k_phys for tilde_k=0 is theta/L = pi/L; F = (1/L) e^{+i k_phys (r_j - r_i)}
            expected = (1.0 / L) * np.exp(1j * np.pi * (j - i) / L)
            assert abs(F[i, j] - expected) < 1e-12, (
                f"F[{i},{j}]={F[i,j]} expected={expected}"
            )
            # And it must NOT match the old (wrong) add-add convention.
            wrong = (1.0 / L) * np.exp(-1j * np.pi * (i + j) / L)
            if abs(expected - wrong) > 1e-10:
                assert abs(F[i, j] - wrong) > 1e-10, (
                    f"F[{i},{j}] matches the OLD add-add convention!"
                )


def test_apbc_l8_density_matches_hwave_negative_bloch_for_n1():
    """For APBC L=8 with only tilde_k = 1 occupied (Codex's test case),
    the per-spin density G_up[0, 1] from the bridge must equal H-wave's
    physical Green (negative-Bloch convention).

    H-wave: G_phys[i, j] = exp(+i theta (r_i - r_j) / L) * (1/N) sum_k
    |v(k)|^2 exp(-i tilde_k (r_j - r_i))
          = (for N=8, theta=pi, tilde_k=2*pi/8, i=0, j=1, |v|=1)
          = (1/8) exp(-i pi/8 - i 2pi/8) = (1/8) exp(-i 3pi/8).
    """
    L = 8
    wv = np.array([[v, 0, 0] for v in _klist(L)], dtype=np.int64)
    site_positions = np.array([[i, 0, 0] for i in range(L)], dtype=np.float64)
    theta = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    L_vec = np.array([L, 1, 1], dtype=np.int64)
    eigenvector = np.zeros((L, 2, 2), dtype=np.complex128)
    for k in range(L):
        eigenvector[k, 0, 0] = 1.0
        eigenvector[k, 1, 1] = 1.0
    column_spin = np.array([0, 1], dtype=np.int64)
    column_mu_group = np.array([0, 1], dtype=np.int64)

    stepped_occupation = np.zeros((L, 2), dtype=np.float64)
    n1 = list(wv[:, 0]).index(1)
    stepped_occupation[n1, 0] = 1.0
    # APBC partner residue: -1 - 1 = -2 mod 8 = 6. _klist value at row 6 is -2.
    n_partner = list(wv[:, 0]).index(-2)
    stepped_occupation[n_partner, 1] = 1.0

    A_up, _A_down = build_amplitudes(
        wv, eigenvector, stepped_occupation, column_spin, column_mu_group,
        site_positions, norb_orig=1, theta=theta, L=L_vec,
    )
    G_up = np.conj(A_up) @ A_up.T

    expected = (1.0 / L) * np.exp(-1j * 3 * np.pi / L)
    assert abs(G_up[0, 1] - expected) < 1e-12, (
        f"G_up[0,1] = {G_up[0, 1]}, expected H-wave value = {expected}"
    )
