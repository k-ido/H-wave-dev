"""Unit tests for the APBC gauge-phase helper."""
import numpy as np
import pytest

from hwave.solver._apbc_phase import (
    normalize_boundary_condition,
    transfer_phase,
    inverse_gauge_phase,
    twist_offset,
)


def test_normalize_lowercase_full_words():
    assert normalize_boundary_condition(["periodic", "antiperiodic", "periodic"]) == (0.0, np.pi, 0.0)


def test_normalize_short_and_case():
    assert normalize_boundary_condition(["P", "AP", "p"]) == (0.0, np.pi, 0.0)
    assert normalize_boundary_condition(["Periodic", "AntiPeriodic", "ap"]) == (0.0, np.pi, np.pi)


def test_normalize_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        normalize_boundary_condition(["periodic", "periodic"])


def test_normalize_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown"):
        normalize_boundary_condition(["periodic", "foo", "periodic"])


def test_normalize_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_boundary_condition([0, 1, 0])


def test_transfer_phase_all_periodic_is_one():
    L = np.array([4, 4, 1])
    theta = np.array([0.0, 0.0, 0.0])
    assert transfer_phase(np.array([1, 2, 0]), theta, L) == pytest.approx(1.0)


def test_transfer_phase_antiperiodic_unit_displacement():
    L = np.array([4, 1, 1])
    theta = np.array([np.pi, 0.0, 0.0])
    expected = np.exp(1j * np.pi / 4)
    assert transfer_phase(np.array([1, 0, 0]), theta, L) == pytest.approx(expected)


def test_transfer_phase_signed_displacement():
    L = np.array([4, 1, 1])
    theta = np.array([np.pi, 0.0, 0.0])
    pos = transfer_phase(np.array([1, 0, 0]), theta, L)
    neg = transfer_phase(np.array([-1, 0, 0]), theta, L)
    assert neg == pytest.approx(np.conj(pos))


def test_transfer_phase_multi_direction():
    L = np.array([4, 4, 1])
    theta = np.array([np.pi, np.pi, 0.0])
    expected = np.exp(1j * np.pi * (1.0 / 4.0 + 2.0 / 4.0))
    assert transfer_phase(np.array([1, 2, 0]), theta, L) == pytest.approx(expected)


def test_inverse_gauge_phase_diagonal_is_one():
    L = np.array([4, 4, 1])
    theta = np.array([np.pi, np.pi, 0.0])
    r = np.array([2, 3, 0])
    assert inverse_gauge_phase(r, r, theta, L) == pytest.approx(1.0)


def test_inverse_gauge_phase_value_and_sign():
    L = np.array([4, 4, 1])
    theta = np.array([np.pi, 0.0, 0.0])
    r_i = np.array([3, 1, 0])
    r_j = np.array([1, 1, 0])
    expected = np.exp(1j * np.pi * (3 - 1) / 4)
    assert inverse_gauge_phase(r_i, r_j, theta, L) == pytest.approx(expected)
    assert inverse_gauge_phase(r_j, r_i, theta, L) == pytest.approx(np.conj(expected))


def test_twist_offset_periodic_zero():
    assert tuple(twist_offset([0.0, 0.0, 0.0])) == (0.0, 0.0, 0.0)


def test_twist_offset_antiperiodic_half():
    assert tuple(twist_offset([np.pi, 0.0, np.pi])) == (0.5, 0.0, 0.5)
