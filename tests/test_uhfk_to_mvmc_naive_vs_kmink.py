"""Same Slater determinant via naive same-k pairing or (k, -k) pairing
must give the same G^sigma; sublattice OrbitalIdx averaging breaks the
naive route but preserves the (k, -k) route. Spec section 3.2 / 5.3.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from tools._uhfk_to_mvmc.density_check import density_from_amplitudes


def _klist(n):
    return np.roll(np.arange(n) - (n // 2), -(n // 2))


def test_naive_and_kmink_pairing_give_same_density():
    """Skipped pending non-degenerate Hubbard fixture from Task 11."""
    import pytest
    pytest.skip(
        "Free-particle cos band has degeneracies that make naive vs "
        "(k, -k) Slater representations differ. Replace with the e2e "
        "case_1d_hubbard fixture once it exists (Task 11)."
    )
