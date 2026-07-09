"""Tests for density_check (spec section 5.1)."""
from __future__ import annotations

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pytest

from tools._uhfk_to_mvmc.density_check import (
    density_from_amplitudes,
    compare_against_onebodyg_uhf,
    DensityMismatchError,
    parse_uhf_cisajs_dat,
)


def test_density_from_amplitudes_pbc_k0_only():
    """Single occupied k=0 plane wave on L=4 → G^sigma_{ij} = 1/L."""
    L = 4
    A = np.full((L, 1), 1.0 / np.sqrt(L), dtype=np.complex128)
    G = density_from_amplitudes(A)
    expected = np.full((L, L), 1.0 / L, dtype=np.complex128)
    np.testing.assert_allclose(G, expected, atol=1e-12)
