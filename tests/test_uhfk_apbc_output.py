"""APBC output-field tests for eigen.npz."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hwave.solver.uhfk import UHFk


L = 4
NORB = 1
NS = 2
ND = NS * NORB  # 2


def _make_stub_with_eigen_state(boundary_theta):
    """UHFk instance ready for the eigen branch of save_results()."""
    s = UHFk.__new__(UHFk)
    s.enable_spin_orbital = False
    s.has_sublattice = False
    s.cellshape = (L, 1, 1)
    s.cellvol = L
    s.shape = (L, 1, 1)
    s.norb = NORB
    s.ns = NS
    s.nd = ND
    s.nvol = L

    # block_info: one block covering all nd states per k -> [[0, 1]]
    s.block_info = [list(range(ND))]
    # _green_list: per-block lists of arrays. One block -> length-1 list.
    # eigenvalue[b] has shape (nvol, blk_size); eigenvector[b] has shape
    # (nvol, blk_size, blk_size).
    eigvals = np.zeros((L, ND), dtype=np.float64)
    eigvecs = np.zeros((L, ND, ND), dtype=np.complex128)
    s._green_list = {"eigenvalue": [eigvals], "eigenvector": [eigvecs]}

    # k-grid metadata: 1D reciprocal basis along x, integer (n_x, 0, 0) indices.
    s.kvec = np.diag([2 * np.pi / L, 2 * np.pi, 2 * np.pi]).astype(np.float64)
    s.wavenum_table = np.array([[n, 0, 0] for n in range(L)], dtype=np.int64)

    # APBC bookkeeping (added by Task 3).
    s.boundary_theta = tuple(boundary_theta)
    s.boundary_periodic = all(t == 0.0 for t in boundary_theta)

    # save_results also touches self.Green when "green" is in info_outputfile;
    # we pass info_outputfile WITHOUT "green" to avoid that path.
    return s


def _save_and_load_eigen(boundary):
    from hwave.solver._apbc_phase import normalize_boundary_condition
    theta = normalize_boundary_condition(boundary)
    s = _make_stub_with_eigen_state(theta)
    with tempfile.TemporaryDirectory() as tmp:
        info_outputfile = {
            "path_to_output": tmp,
            "eigen": "eigen.npz",
            "green": "",          # empty string suppresses green save
        }
        s.save_results(info_outputfile, green_info={})
        return dict(np.load(Path(tmp) / "eigen.npz", allow_pickle=False))


def test_twist_offset_present_for_apbc():
    data = _save_and_load_eigen(["antiperiodic", "periodic", "antiperiodic"])
    assert "twist_offset" in data
    np.testing.assert_allclose(data["twist_offset"], np.array([0.5, 0.0, 0.5]))


def test_twist_offset_all_zero_for_full_pbc():
    data = _save_and_load_eigen(["periodic", "periodic", "periodic"])
    assert "twist_offset" in data
    np.testing.assert_allclose(data["twist_offset"], np.zeros(3))


def test_existing_eigen_keys_preserved():
    data = _save_and_load_eigen(["antiperiodic", "periodic", "periodic"])
    for k in ("eigenvalue", "eigenvector", "wavevector_unit", "wavevector_index"):
        assert k in data


def test_physical_momentum_reconstruction():
    data = _save_and_load_eigen(["antiperiodic", "periodic", "periodic"])
    kvec = data["wavevector_unit"]      # (3, 3)
    idx = data["wavevector_index"]      # (nvol, 3)
    off = data["twist_offset"]          # (3,)
    k_phys = (idx + off) @ kvec
    # APBC in x with L=4 -> k_x in {pi/4, 3pi/4, 5pi/4, 7pi/4}
    k_x_set = np.sort(np.unique(np.round(k_phys[:, 0], 10)))
    expected = np.sort(np.array(
        [(2 * n + 1) * np.pi / L for n in range(L)]
    ).round(10))
    np.testing.assert_allclose(k_x_set, expected, atol=1e-10)


def _make_stub_with_export_state(boundary_theta):
    """UHFk stub minimal for _export_hamiltonian to run."""
    s = UHFk.__new__(UHFk)
    s.norb = NORB
    # Minimal param_ham so the for-loop over keys yields exactly one
    # exportable interaction; we only care about the BoundaryCondition file
    # for this test, so use a tiny Transfer table.
    s.param_ham = {
        "Transfer": {((1, 0, 0), (0, 0)): -1.0 + 0.0j},
    }
    s.boundary_theta = tuple(boundary_theta)
    s.boundary_periodic = all(t == 0.0 for t in boundary_theta)
    return s


def _read_boundary_file(path: Path):
    lines = [ln.strip() for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    return lines


def test_export_boundary_condition_apbc():
    from hwave.solver._apbc_phase import normalize_boundary_condition
    theta = normalize_boundary_condition(["antiperiodic", "periodic", "antiperiodic"])
    s = _make_stub_with_export_state(theta)
    with tempfile.TemporaryDirectory() as tmp:
        s._export_hamiltonian(tmp, "test_")
        bc_path = Path(tmp) / "test_BoundaryCondition.dat"
        assert bc_path.exists()
        assert _read_boundary_file(bc_path) == ["antiperiodic", "periodic", "antiperiodic"]


def test_export_boundary_condition_pbc():
    from hwave.solver._apbc_phase import normalize_boundary_condition
    theta = normalize_boundary_condition(["periodic"] * 3)
    s = _make_stub_with_export_state(theta)
    with tempfile.TemporaryDirectory() as tmp:
        s._export_hamiltonian(tmp, "pbc_")
        bc_path = Path(tmp) / "pbc_BoundaryCondition.dat"
        assert bc_path.exists()
        assert _read_boundary_file(bc_path) == ["periodic", "periodic", "periodic"]


def test_export_transfer_is_unchanged_under_apbc():
    """The exported Transfer.dat must contain the raw (un-phased) values.

    Boundary info is referenced separately via BoundaryCondition.dat.
    """
    from hwave.solver._apbc_phase import normalize_boundary_condition
    theta = normalize_boundary_condition(["antiperiodic", "periodic", "periodic"])
    s = _make_stub_with_export_state(theta)
    with tempfile.TemporaryDirectory() as tmp:
        s._export_hamiltonian(tmp, "raw_")
        transfer_path = Path(tmp) / "raw_Transfer.dat"
        assert transfer_path.exists()
        # The exported value for R=(1,0,0), orb (1,1) must be -1.0 (NOT
        # multiplied by exp(i theta * 1/L) phase).
        lines = transfer_path.read_text().splitlines()
        data_lines = [ln for ln in lines if ln.strip() and not ln[0].isalpha()]
        # Body rows have exactly 7 columns: rx ry rz orb1 orb2 re im.
        # Header rows (count, orb-multiplicity blocks) have fewer columns and
        # must be filtered out before matching on rx.
        body = [ln.split() for ln in data_lines]
        body_rx1 = [row for row in body if len(row) == 7 and row[0] == "1"]
        assert body_rx1, "Transfer.dat is missing the R=(1,0,0) entry"
        re = float(body_rx1[0][5])
        im = float(body_rx1[0][6])
        np.testing.assert_allclose([re, im], [-1.0, 0.0], atol=1e-12)
