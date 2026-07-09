"""Tests for density_check (spec section 5.1)."""
from __future__ import annotations

import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pytest

from tools._uhfk_to_mvmc.density_check import (
    density_from_amplitudes,
    compare_against_onebodyg_uhf,
    compare_against_onebodyg_uhf_general,
    DensityMismatchError,
    parse_uhf_cisajs_dat,
)


_DATA_DIR = Path(__file__).parent / "data"


def test_density_from_amplitudes_pbc_k0_only():
    """Single occupied k=0 plane wave on L=4 → G^sigma_{ij} = 1/L."""
    L = 4
    A = np.full((L, 1), 1.0 / np.sqrt(L), dtype=np.complex128)
    G = density_from_amplitudes(A)
    expected = np.full((L, L), 1.0 / L, dtype=np.complex128)
    np.testing.assert_allclose(G, expected, atol=1e-12)


def test_density_check_soc_convention_asymmetric():
    """Under ``is_soc_mode=True``, density_check reads greenone.dat rows
    on mVMC spin-block order (``all_i = i + s * Ns``) and passes when the
    bridge output matches. If the bridge used interleaved order, the
    asymmetric golden values would force a numerical mismatch above
    tolerance."""
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 3] = 0.3
    G_all[1, 2] = 0.2 + 0.1j
    G_all[2, 1] = 0.2 - 0.1j
    G_all[3, 0] = 0.3

    golden = str(_DATA_DIR / "soc_greenone_asymmetric_golden.dat")

    # Should pass without raising.
    compare_against_onebodyg_uhf_general(
        G_all, golden, tol=1e-10, is_soc_mode=True,
    )


def test_density_check_soc_rejects_beyond_tol():
    """SOC mode with an s != t entry mismatched beyond tolerance -> raise
    DensityMismatchError."""
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 3] = 0.3
    G_all[1, 2] = 0.2 + 0.1j + 3e-10  # deliberate error above 1e-10 tol
    G_all[2, 1] = 0.2 - 0.1j
    G_all[3, 0] = 0.3

    golden = str(_DATA_DIR / "soc_greenone_asymmetric_golden.dat")

    with pytest.raises(DensityMismatchError, match="differ"):
        compare_against_onebodyg_uhf_general(
            G_all, golden, tol=1e-10, is_soc_mode=True,
        )


def test_density_check_v3_still_rejects_s_neq_t_when_not_soc_mode():
    """v3 behavior preserved under ``is_soc_mode=False``: if the reference
    greenone.dat has a non-zero s != t entry while the bridge's G_all is
    zero in that off-diagonal block (which is the natural v3 Sz-diagonal
    output), the comparison must still flag the mismatch.

    This exercises the implicit scope-violation guard that used to be
    described in the v3 docstring: v3 bridge builds a spin-block-diagonal
    G_all, so any non-zero s != t entry in the reference will differ from
    the bridge's zero and trigger DensityMismatchError.
    """
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 0] = 0.5
    G_all[1, 1] = 0.5
    G_all[2, 2] = 0.5
    G_all[3, 3] = 0.5  # spin-block-diagonal only, off-diagonal blocks all 0

    # Reference has a non-zero s != t entry, incompatible with v3 scope.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".dat", delete=False
    ) as tmp:
        tmp.write("0 0 0 0  0.5 0.0\n")
        tmp.write("1 0 1 0  0.5 0.0\n")
        tmp.write("0 1 0 1  0.5 0.0\n")
        tmp.write("1 1 1 1  0.5 0.0\n")
        tmp.write("0 0 1 1  0.3 0.0\n")  # off-diagonal block: not zero
        path = tmp.name

    try:
        with pytest.raises(DensityMismatchError, match="differ"):
            compare_against_onebodyg_uhf_general(
                G_all, path, tol=1e-10, is_soc_mode=False,
            )
    finally:
        os.unlink(path)
