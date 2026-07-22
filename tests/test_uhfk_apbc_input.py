"""Verify BoundaryCondition arrives at the solver via the normal input path."""
import numpy as np
import pytest

from hwave.solver._apbc_phase import normalize_boundary_condition


def test_default_when_key_absent_is_all_periodic():
    # The solver-side default is all-periodic; the helper itself always
    # requires an explicit list. Document that the loader must
    # default to ["periodic"]*3 when the TOML key is absent.
    assert normalize_boundary_condition(["periodic"] * 3) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "raw, expected_theta",
    [
        (["periodic", "periodic", "periodic"], (0.0, 0.0, 0.0)),
        (["antiperiodic", "periodic", "periodic"], (np.pi, 0.0, 0.0)),
        (["AP", "p", "AP"], (np.pi, 0.0, np.pi)),
        (["Antiperiodic", "Periodic", "ANTIPERIODIC"], (np.pi, 0.0, np.pi)),
    ],
)
def test_accepted_forms(raw, expected_theta):
    assert normalize_boundary_condition(raw) == expected_theta


@pytest.mark.parametrize(
    "bad",
    [
        ["periodic"],
        ["periodic", "periodic"],
        ["periodic", "periodic", "periodic", "periodic"],
    ],
)
def test_rejects_wrong_length(bad):
    with pytest.raises(ValueError, match="length"):
        normalize_boundary_condition(bad)


def test_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown"):
        normalize_boundary_condition(["periodic", "twisted", "periodic"])


def test_rejects_non_string_entry():
    with pytest.raises(TypeError):
        normalize_boundary_condition(["periodic", 1, "periodic"])
