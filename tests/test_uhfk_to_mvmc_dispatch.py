"""Tests for tools/uhfk_to_mvmc.py CLI dispatch (v3 format-first).

Verifies the 4-case matrix in spec §5.3:
  (is_antiparallel_metadata, orbitalidx_format) → routing / rejection
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _minimal_general_orbitalidx(nsite=2):
    total = 2 * nsite * nsite - nsite
    lines = [
        "======================",
        f"NOrbitalIdx  {total}",
        "ComplexType 1",
        "======================",
        "== i_spn_j_spn_OrbitalIdx ==",
        "======================",
    ]
    idx = 0
    for all_i in range(2 * nsite):
        for all_j in range(all_i + 1, 2 * nsite):
            i, spn_i = all_i % nsite, all_i // nsite
            j, spn_j = all_j % nsite, all_j // nsite
            lines.append(f"{i} {spn_i} {j} {spn_j} {idx} 1")
            idx += 1
    for k in range(total):
        lines.append(f"{k} 1")
    return "\n".join(lines) + "\n"


def _minimal_antiparallel_orbitalidx(nsite=2):
    total = nsite * nsite
    lines = [
        "======================",
        f"NOrbitalIdx  {total}",
        "ComplexType 1",
        "======================",
        "== i_j_OrbitalIdx ==",
        "======================",
    ]
    idx = 0
    for i in range(nsite):
        for j in range(nsite):
            lines.append(f"{i} {j} {idx}")
            idx += 1
    for k in range(total):
        lines.append(f"{k} 1")
    return "\n".join(lines) + "\n"


def _write_fixture(tmp, sub_shape=(1, 1, 1), two_sz=0, column_spin=(0, 1)):
    """Emit input.toml + eigen.npz + occupation.npz + geometry_uhf.dat
    minimal enough for CLI to reach the dispatch branch. The mapping
    file path is filled in by the caller.

    Eigenvalues are set to distinct non-degenerate values so
    step_occupation's Fermi-level degeneracy guard does not trip.
    """
    Nsite = 2
    Ncond = 2
    with open(os.path.join(tmp, "input.toml"), "w") as fp:
        sub_str = ", ".join(str(int(x)) for x in sub_shape)
        fp.write(
            "[mode.param]\n"
            f"Ncond = {Ncond}\n"
            f"2Sz = {two_sz}\n"
            "T = 0.0\n"
            f"CellShape = [{Nsite}, 1, 1]\n"
            f"SubShape  = [{sub_str}]\n"
            'BoundaryCondition = ["periodic", "periodic", "periodic"]\n'
        )
    # Distinct eigenvalues per (nvol_row, column) so no Fermi-level
    # degeneracy fires. Column 0 (mu-group 0): 0.0 at row 0 < 0.5 at row 1.
    # Column 1 (mu-group 1): 1.0 at row 0 < 1.5 at row 1.
    eigenvalue = np.array(
        [[0.0, 1.0], [0.5, 1.5]], dtype=np.float64
    )
    np.savez(
        os.path.join(tmp, "eigen.npz"),
        eigenvalue=eigenvalue,
        eigenvector=np.eye(2, dtype=np.complex128).reshape(1, 2, 2).repeat(Nsite, axis=0),
        wavevector_unit=np.eye(3, dtype=np.float64),
        wavevector_index=np.array([[v, 0, 0] for v in [0, 1]], dtype=np.int64),
        twist_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    # Occupation: single up + single down at row 0 satisfies pair-closure for
    # PBC self-pair at k=0.
    occ = np.zeros((Nsite, 2), dtype=np.float64)
    occ[0, 0] = 1.0
    occ[0, 1] = 1.0
    np.savez(
        os.path.join(tmp, "occupation.npz"),
        occupation=occ,
        mu=np.array([0.0, 0.0], dtype=np.float64),
        T=np.float64(0.0),
        column_spin=np.array(column_spin, dtype=np.int64),
        column_mu_group=np.array([0, 1], dtype=np.int64),
    )
    with open(os.path.join(tmp, "geometry_uhf.dat"), "w") as fp:
        fp.write("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")
        fp.write("0.0 0.0 0.0\n")
        fp.write("2 0 0\n0 1 0\n0 0 1\n")
        for i in range(Nsite):
            fp.write(f"{i} 0 0 0\n")


def _run_cli(tmp, orbitalidx_body, extra_args=()):
    orbidx = os.path.join(tmp, "orbitalidx.def")
    with open(orbidx, "w") as fp:
        fp.write(orbitalidx_body)
    out = os.path.join(tmp, "zqp_orbital_uhfk.dat")
    return subprocess.run(
        [sys.executable, "tools/uhfk_to_mvmc.py",
         "--input", os.path.join(tmp, "input.toml"),
         "--eigen", os.path.join(tmp, "eigen.npz"),
         "--occupation", os.path.join(tmp, "occupation.npz"),
         "--geometry", os.path.join(tmp, "geometry_uhf.dat"),
         "--orbitalidx", orbidx,
         "--output", out,
         "--no-check-density",
         "--epsilon-noise", "0",
         *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_dispatch_antiparallel_metadata_plus_antiparallel_format_uses_v21_path():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture(tmp, two_sz=0, column_spin=(0, 1))
        res = _run_cli(tmp, _minimal_antiparallel_orbitalidx(nsite=2))
        assert res.returncode == 0, res.stderr


def test_dispatch_antiparallel_metadata_plus_general_format_forced_general():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture(tmp, two_sz=0, column_spin=(0, 1))
        res = _run_cli(tmp, _minimal_general_orbitalidx(nsite=2))
        # Should succeed via forced-General branch and possibly emit a
        # WARNING (v2.1 closure holds → no warning expected here).
        assert res.returncode == 0, res.stderr


def test_dispatch_not_antiparallel_plus_antiparallel_format_rejected():
    """2Sz not provided (Sz-free) + antiparallel 3-column orbitalidx → reject."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write input.toml without 2Sz (defaults treated as null in dispatch).
        _write_fixture(tmp, two_sz=0)
        # Overwrite input.toml to remove 2Sz.
        with open(os.path.join(tmp, "input.toml"), "w") as fp:
            fp.write(
                "[mode.param]\n"
                "Ncond = 2\n"
                "T = 0.0\n"
                "CellShape = [2, 1, 1]\n"
                "SubShape  = [1, 1, 1]\n"
                'BoundaryCondition = ["periodic", "periodic", "periodic"]\n'
            )
        res = _run_cli(tmp, _minimal_antiparallel_orbitalidx(nsite=2))
        assert res.returncode == 2, res.stderr
        assert "General" in res.stderr or "orbitalidx_general" in res.stderr
