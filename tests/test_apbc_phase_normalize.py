# tests/test_apbc_phase_normalize.py
import math

import pytest

from hwave.solver._apbc_phase import normalize_boundary_condition


def test_accepts_periodic_variants():
    for bc in (
        ["periodic", "periodic", "periodic"],
        ["p", "P", "Periodic"],
        [" periodic ", "p", "P"],
    ):
        theta = normalize_boundary_condition(bc)
        assert theta == (0.0, 0.0, 0.0)


def test_accepts_antiperiodic_variants():
    theta = normalize_boundary_condition(["antiperiodic", "AP", "ap"])
    assert theta == pytest.approx((math.pi, math.pi, math.pi))


def test_accepts_mixed_pbc_apbc():
    theta = normalize_boundary_condition(["ap", "periodic", "p"])
    assert theta == pytest.approx((math.pi, 0.0, 0.0))


def test_rejects_open_and_ambiguous():
    for bc in (
        ["open", "periodic", "periodic"],
        ["aperiodic", "periodic", "periodic"],
        ["", "periodic", "periodic"],
        ["antiperiodicX", "periodic", "periodic"],
        ["peri odic", "periodic", "periodic"],
    ):
        with pytest.raises(ValueError):
            normalize_boundary_condition(bc)


def test_rejects_wrong_length():
    with pytest.raises(ValueError):
        normalize_boundary_condition(["periodic", "periodic"])
    with pytest.raises(ValueError):
        normalize_boundary_condition(["p", "p", "p", "p"])
