"""Analytic-spectrum tests for free-fermion APBC eigenvalues.

Same __new__ stub pattern as tests/test_uhfk_apbc_consistency.py — we drive
_make_ham_trans() directly and diagonalize each k-block.
"""
import numpy as np
import pytest

from hwave.solver.uhfk import UHFk


def make_trans_stub(cellshape, transfer_dict,
                    boundary_theta=(0.0, 0.0, 0.0), norb=1):
    s = object.__new__(UHFk)
    s.shape = tuple(cellshape)
    s.cellshape = tuple(cellshape)
    s.nvol = cellshape[0] * cellshape[1] * cellshape[2]
    s.norb = norb
    s.ns = 2
    s.nd = 2 * norb
    s.enable_spin_orbital = False
    s.param_ham = {"Transfer": dict(transfer_dict)}
    s.boundary_theta = tuple(boundary_theta)
    s.boundary_periodic = all(t == 0.0 for t in boundary_theta)
    return s


def nn_1d(t=1.0):
    return {((1, 0, 0), (0, 0)): -t, ((-1, 0, 0), (0, 0)): -t}


def nn_2d(t=1.0):
    return {
        ((1, 0, 0), (0, 0)): -t, ((-1, 0, 0), (0, 0)): -t,
        ((0, 1, 0), (0, 0)): -t, ((0, -1, 0), (0, 0)): -t,
    }


def nn_3d(t=1.0):
    return {
        ((1, 0, 0), (0, 0)): -t, ((-1, 0, 0), (0, 0)): -t,
        ((0, 1, 0), (0, 0)): -t, ((0, -1, 0), (0, 0)): -t,
        ((0, 0, 1), (0, 0)): -t, ((0, 0, -1), (0, 0)): -t,
    }


def spectrum_from_ham_trans(ham_trans):
    eigs = []
    for hk in ham_trans:
        eigs.extend(np.linalg.eigvalsh(hk).tolist())
    return np.sort(np.asarray(eigs, dtype=np.float64))


def expected_1d_apbc(L, t=1.0):
    spec = []
    for n in range(L):
        eps = -2.0 * t * np.cos((2 * n + 1) * np.pi / L)
        spec.extend([eps, eps])  # spin-degenerate
    return np.sort(np.asarray(spec))


def expected_2d(Lx, Ly, theta_x, theta_y, t=1.0):
    spec = []
    for nx in range(Lx):
        for ny in range(Ly):
            kx = (2 * np.pi * nx + theta_x) / Lx
            ky = (2 * np.pi * ny + theta_y) / Ly
            eps = -2.0 * t * (np.cos(kx) + np.cos(ky))
            spec.extend([eps, eps])
    return np.sort(np.asarray(spec))


def expected_3d(Lx, Ly, Lz, tx, ty, tz, t=1.0):
    spec = []
    for nx in range(Lx):
        for ny in range(Ly):
            for nz in range(Lz):
                kx = (2 * np.pi * nx + tx) / Lx
                ky = (2 * np.pi * ny + ty) / Ly
                kz = (2 * np.pi * nz + tz) / Lz
                eps = -2.0 * t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
                spec.extend([eps, eps])
    return np.sort(np.asarray(spec))


def hamk(cellshape, transfer, theta):
    s = make_trans_stub(cellshape, transfer, boundary_theta=theta)
    s._make_ham_trans()
    return s.ham_trans


@pytest.mark.parametrize("L", [4, 6, 8])
def test_1d_apbc_spectrum(L):
    got = spectrum_from_ham_trans(hamk([L, 1, 1], nn_1d(), (np.pi, 0.0, 0.0)))
    want = expected_1d_apbc(L)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_2d_mixed_apbc_spectrum():
    Lx, Ly = 4, 6
    got = spectrum_from_ham_trans(
        hamk([Lx, Ly, 1], nn_2d(), (np.pi, 0.0, 0.0))
    )
    want = expected_2d(Lx, Ly, np.pi, 0.0)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_3d_full_apbc_spectrum():
    Lx, Ly, Lz = 4, 4, 4
    got = spectrum_from_ham_trans(
        hamk([Lx, Ly, Lz], nn_3d(), (np.pi, np.pi, np.pi))
    )
    want = expected_3d(Lx, Ly, Lz, np.pi, np.pi, np.pi)
    np.testing.assert_allclose(got, want, atol=1e-12)
