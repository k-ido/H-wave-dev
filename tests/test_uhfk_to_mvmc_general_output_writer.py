"""Tests for general_output_writer — v3 InOrbitalGeneral writer + aggregator."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from tools._uhfk_to_mvmc.general_output_writer import (
    ClassInconsistencyError,
    aggregate_general_orbital_params,
    write_zqp_orbital_general,
)
from tools._uhfk_to_mvmc.density_check import (
    DensityMismatchError,
    compare_against_onebodyg_uhf_general,
)


def _antisym_F(nsite):
    """Return a random-but-reproducible antisymmetric (2Ns, 2Ns) F."""
    rng = np.random.default_rng(1234)
    n = 2 * nsite
    R = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (R - R.T) / 2  # antisymmetric


def test_aggregate_general_orbital_params_consistent_class_averages_signed_value():
    """Two rows mapped to the same idx with consistent signed values →
    params[idx] = the shared value (average of equal numbers)."""
    F = _antisym_F(nsite=2)
    # Ns = 2, so 2Ns = 4, upper triangle has 2*Ns*(2Ns-1)/2 = 6 rows.
    # Put (all_i=0, all_j=1) and (all_i=2, all_j=3) into the same class 0,
    # both with sign=+1. F[0,1] and F[2,3] must be equal for the class-
    # consistency check to pass; force them equal here.
    F[0, 1] = 0.42 + 0.13j
    F[1, 0] = -F[0, 1]
    F[2, 3] = 0.42 + 0.13j
    F[3, 2] = -F[2, 3]
    mapping = {
        (0, 1): (0, 1),
        (2, 3): (0, 1),
        (0, 2): (1, 1),
        (0, 3): (2, 1),
        (1, 2): (3, 1),
        (1, 3): (4, 1),
    }
    params = aggregate_general_orbital_params(
        F, mapping=mapping, n_orbital_idx=5,
        epsilon_noise=0.0,
    )
    assert params[0] == pytest.approx(0.42 + 0.13j, abs=1e-14)


def test_aggregate_general_raises_on_inconsistent_class_entries():
    """Two rows mapped to the same idx with unequal signed values →
    ClassInconsistencyError including idx and max residual."""
    F = _antisym_F(nsite=2)
    F[0, 1] = 0.5
    F[1, 0] = -F[0, 1]
    F[2, 3] = 0.5 + 0.1j  # differs from F[0, 1] by 0.1j
    F[3, 2] = -F[2, 3]
    mapping = {
        (0, 1): (0, 1),
        (2, 3): (0, 1),
        (0, 2): (1, 1),
        (0, 3): (2, 1),
        (1, 2): (3, 1),
        (1, 3): (4, 1),
    }
    with pytest.raises(ClassInconsistencyError) as excinfo:
        aggregate_general_orbital_params(
            F, mapping=mapping, n_orbital_idx=5,
            epsilon_noise=0.0,
        )
    msg = str(excinfo.value)
    assert "idx" in msg and "0" in msg
    assert "residual" in msg


def test_write_zqp_orbital_general_roundtrip_matches_input_params():
    """Round-trip: write params to file then re-parse; expect exact match
    on real/imag parts."""
    params = np.array([0.1 + 0.0j, -0.2 + 0.5j, 0.0 + 0.0j], dtype=np.complex128)
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
        write_zqp_orbital_general(tmp.name, params)
        with open(tmp.name) as fp:
            body = fp.read()
    # Header assertions
    assert "NOrbitalIdx" in body
    # Row body: "<idx> <real> <imag>" per line
    data_lines = [
        ln for ln in body.splitlines()
        if ln.strip() and not ln.startswith("==") and not ln.startswith("N")
        and not ln.startswith("i") and not ln.startswith("=")
    ]
    parsed = []
    for ln in data_lines:
        toks = ln.strip().split()
        if len(toks) == 3:
            parsed.append(complex(float(toks[1]), float(toks[2])))
    assert len(parsed) == len(params)
    for p_got, p_want in zip(parsed, params):
        assert p_got == pytest.approx(p_want, abs=1e-15)


def test_compare_against_onebodyg_uhf_general_accepts_matching_g():
    """Synthetic G matching greenone.dat entries → no exception."""
    G_all = np.zeros((4, 4), dtype=np.complex128)
    G_all[0, 1] = 0.25
    G_all[2, 3] = 0.10
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
        # greenone.dat format: i s j t re im
        tmp.write("0 0 1 0  0.25 0.0\n")
        tmp.write("0 1 1 1  0.10 0.0\n")  # spin-down entry: G[0+Ns, 1+Ns]=G[2, 3]=0.10
        tmp.write("0 0 0 1  0.0 0.0\n")  # spin off-diagonal: expected 0 in v3 scope
        tmp.name
        path = tmp.name
    compare_against_onebodyg_uhf_general(G_all, path, tol=1e-10)


def test_compare_against_onebodyg_uhf_general_flags_mismatch():
    G_all = np.zeros((4, 4), dtype=np.complex128)
    G_all[0, 1] = 0.25
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
        tmp.write("0 0 1 0  0.10 0.0\n")  # want 0.10 but bridge gives 0.25
        path = tmp.name
    with pytest.raises(DensityMismatchError, match="differ"):
        compare_against_onebodyg_uhf_general(G_all, path, tol=1e-10)
