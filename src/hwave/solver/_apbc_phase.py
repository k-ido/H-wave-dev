"""APBC gauge-phase helpers for the k-space UHF solver.

All functions are pure and take explicit arrays so they can be tested in
isolation. Conventions follow spec sections 3.2 / 3.3:

* Forward gauge:    \\tilde c_x = exp(i sum_d theta_d * x_d / L_d) * c_x
* Transfer phase:   T(R) -> T(R) * exp(i sum_d theta_d * R_d / L_d)
* Inverse Green:    G_ij = exp(i sum_d theta_d * (r_i - r_j)_d / L_d) * \\tilde G_ij
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

_PERIODIC_ALIASES = {"p", "periodic"}
_ANTIPERIODIC_ALIASES = {"ap", "antiperiodic"}


def normalize_boundary_condition(values: Sequence[object]) -> tuple[float, float, float]:
    """Convert a BoundaryCondition list to (theta_x, theta_y, theta_z) in {0, pi}.

    Accepts case-insensitive ``"periodic"``/``"P"`` and
    ``"antiperiodic"``/``"AP"``. Raises ``TypeError`` for non-strings,
    ``ValueError`` for length mismatch or unknown values.
    """
    if len(values) != 3:
        raise ValueError(
            f"BoundaryCondition length must be 3 (got {len(values)})"
        )
    out: list[float] = []
    for v in values:
        if not isinstance(v, str):
            raise TypeError(
                f"BoundaryCondition entries must be strings (got {type(v).__name__})"
            )
        key = v.strip().lower()
        if key in _PERIODIC_ALIASES:
            out.append(0.0)
        elif key in _ANTIPERIODIC_ALIASES:
            out.append(np.pi)
        else:
            raise ValueError(
                f"unknown BoundaryCondition value {v!r}; "
                "expected 'periodic'/'P' or 'antiperiodic'/'AP'"
            )
    return (out[0], out[1], out[2])


def transfer_phase(displacement: np.ndarray, theta: np.ndarray, L: np.ndarray) -> complex:
    """Return exp(i * sum_d theta_d * displacement_d / L_d).

    ``displacement`` must be the SIGNED real-space displacement in lattice
    units (use the raw ``irvec`` from ``param_ham['Transfer']`` keys, NOT
    the modulo-wrapped FFT index).
    """
    arg = float(np.sum(theta * displacement / L))
    return complex(np.exp(1j * arg))


def inverse_gauge_phase(
    r_i: np.ndarray, r_j: np.ndarray, theta: np.ndarray, L: np.ndarray
) -> complex:
    """Return exp(i * sum_d theta_d * (r_i_d - r_j_d) / L_d).

    Applies to G_ij = <c^dag_i c_j>; diagonal (r_i = r_j) is 1.
    """
    return transfer_phase(r_i - r_j, theta, L)


def twist_offset(theta: Iterable[float]) -> np.ndarray:
    """Return [theta_d / (2*pi)] per direction (values in {0, 1/2})."""
    return np.array([float(t) / (2.0 * np.pi) for t in theta], dtype=np.float64)
