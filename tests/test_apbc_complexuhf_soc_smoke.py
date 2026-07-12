"""Phase 5 smoke tests for the apbc_complexuhf SOC fixture and parser.

Phase 5 is HARNESS-ONLY per plan Task 5c: no live ComplexUHF SCF is
launched here. These tests confirm:

1. The v3.6 SOC fixture at
   `tests/validation/apbc_complexuhf/case_soc_rashba_2d_sub_apbc/` has
   the eight files the run.sh case switch will consume.
2. `compare.py::_read_greenone` still parses the 4-index `(i, s, j, t)`
   SOC layout end-to-end on a real H-wave greenone.dat snapshot from
   the shipping v3.6 fixture, including non-zero cross-spin `s != t`
   entries which the Phase 6 G2a/G2b gates rely on.
3. The Phase 3 committed snapshots exist and can be loaded (proves the
   Phase 5 harness can consume them if Phase 6 needs to compare against
   frozen truth).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[1]
_APBC_CU_CASE = _REPO / "tests" / "validation" / "apbc_complexuhf" / \
    "case_soc_rashba_2d_sub_apbc"
_APBC_CU_COMPARE = _REPO / "tests" / "validation" / "apbc_complexuhf" / \
    "compare.py"
_SHIPPING_CASE = _REPO / "tests" / "validation" / "uhfk_mvmc_pairproduct" / \
    "case_soc_rashba_2d_sub_apbc"
_DATA_DIR = _REPO / "tests" / "data"


def test_apbc_complexuhf_soc_fixture_has_required_files():
    """Phase 5a: the fixture skeleton committed by this branch has the
    eight files run.sh case_soc_rashba_2d_sub_apbc (Phase 6) will read.
    Rate: files exist; do not assert on their contents (regeneration
    against StdFace is Phase 6's job)."""
    required = [
        "Geometry.dat", "Transfer.dat", "CoulombIntra.dat",
        "OneBodyG.dat", "geometry_uhf.dat", "input.toml",
        "stan.in", "complexuhf_modpara_override.txt", "README.md",
    ]
    for name in required:
        p = _APBC_CU_CASE / name
        assert p.is_file(), (
            f"Phase 5a fixture missing {name} under {_APBC_CU_CASE}"
        )


def test_apbc_complexuhf_soc_fixture_matches_hwave_hamiltonian():
    """The Hamiltonian input files (Geometry, Transfer, CoulombIntra,
    OneBodyG, geometry_uhf, input.toml) MUST be bit-identical between
    the apbc_complexuhf fixture and the H-wave shipping fixture so the
    Phase 6 cross-check compares equal SCF solvers rather than different
    discretizations. StdFace vs H-wave already disagree by finite VMC
    stderr; on top of that, feeding differently-quantized Hamiltonians
    would make the G2a/G2b diagnosis useless."""
    mirrored = [
        "Geometry.dat", "Transfer.dat", "CoulombIntra.dat",
        "OneBodyG.dat", "geometry_uhf.dat", "input.toml",
    ]
    for name in mirrored:
        a = (_APBC_CU_CASE / name).read_bytes()
        b = (_SHIPPING_CASE / name).read_bytes()
        assert a == b, (
            f"{name}: apbc_complexuhf fixture and H-wave shipping "
            "fixture must have bit-identical Hamiltonian files; "
            f"lengths {len(a)} vs {len(b)}."
        )


def test_apbc_complexuhf_soc_stan_in_declares_phase0_180():
    """The APBC realization on the ComplexUHF side goes through
    phase0 = 180.0 in stan.in (StdFace bakes the -1 wrap sign into
    Trans.def). If the fixture is copied from the non-APBC template
    and phase0 is missing / not 180.0, ComplexUHF silently produces a
    PBC ground truth and the Phase 6 G2a/G2b gates would compare
    APBC-H-wave against PBC-ComplexUHF -> false positives on every
    cross-spin entry.
    """
    text = (_APBC_CU_CASE / "stan.in").read_text()
    # Loose match; ComplexUHF stan.in is line-oriented with '='.
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        if "phase0" in stripped:
            eq_val = stripped.split("=", 1)[1].strip()
            # Allow "180" / "180.0" / "180.00" etc.
            assert float(eq_val) == 180.0, (
                f"stan.in phase0 must be 180.0 (APBC in x); got {eq_val!r}"
            )
            found = True
            break
    assert found, (
        "stan.in does NOT declare phase0; APBC in x would collapse to "
        "PBC on the ComplexUHF side and the G2a/G2b gates would "
        "produce false positives against the APBC H-wave ground truth."
    )


def _load_apbc_compare_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_apbc_compare_under_test", str(_APBC_CU_COMPARE),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_greenone_parses_soc_cross_spin_rows_on_hwave_snapshot():
    """Phase 5b: the parser must round-trip the 4-index (i, s, j, t)
    SOC layout on a real H-wave greenone.dat under APBC + SubShape.

    Uses the shipping fixture's greenone.dat (produced by Phase 2 SCF).
    Asserts:
      - parser returns > 0 rows
      - at least one non-zero cross-spin (s != t) row exists (this is
        what makes the SOC gate non-trivial; a pure-Sz-diagonal parser
        would silently drop these)
      - the parsed dict's values are complex128-compatible.
    """
    greenone = _SHIPPING_CASE / "output" / "greenone.dat"
    assert greenone.is_file(), (
        f"missing shipping greenone.dat at {greenone}; Phase 2 SCF "
        "outputs must be committed for Phase 5 smoke coverage."
    )
    mod = _load_apbc_compare_module()
    parsed = mod._read_greenone(greenone)
    assert len(parsed) > 0, "greenone parser returned empty dict"
    # Non-zero cross-spin sanity: pick any (i, s, j, t) with s != t and
    # magnitude > 1e-6. Rashba SOC on this fixture guarantees several.
    cross_spin_nonzero = [
        v for (i, s, j, t), v in parsed.items()
        if s != t and abs(v) > 1e-6
    ]
    assert cross_spin_nonzero, (
        "no non-zero cross-spin (s != t) rows parsed from greenone; "
        "either the fixture lost its Rashba SOC or the parser silently "
        "dropped 4-index SOC rows. Either way, the Phase 6 G2a-in-"
        "memory-A / G2b cross-check would produce a trivial pass."
    )
    # Sanity on dtype.
    _ = np.complex128(next(iter(parsed.values())))


def test_phase3_snapshots_loadable_from_apbc_harness_side():
    """Phase 5c: the persisted Phase 3 snapshots that Phase 6 G2a-in-
    memory-A / G2b will consume (via the H-wave uhfk_mvmc_pairproduct
    harness on the shipping side, mirrored on the apbc_complexuhf side)
    load without error. This is a low-cost harness smoke test."""
    for name in (
        "v36_case_soc_rashba_2d_sub_apbc_green.npz",
        "v36_case_soc_rashba_2d_sub_apbc_eigen.npz",
        "v36_case_soc_rashba_2d_sub_apbc_occupation.npz",
    ):
        p = _DATA_DIR / name
        assert p.is_file(), f"Phase 3 snapshot missing: {p}"
        with np.load(str(p)) as npz:
            files = list(npz.files)
            assert files, f"Phase 3 snapshot {name} is empty"
