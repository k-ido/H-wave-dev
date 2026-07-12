"""Phase 2g mandatory negative tests for the semantic G4 topology guard.

Spec §4.3 (composite condition) + plan Task 2g (six mandatory negative
tests). Each test constructs a small stub manifest / workspace and
asserts the guard rejects with exit code 2 + empty stdout on the
targeted invariant violation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest


_GUARD_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "validation",
        "uhfk_mvmc_pairproduct",
        "scripts",
        "soc_apbc_topology_guard.py",
    )
)
_REAL_CASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "validation",
        "uhfk_mvmc_pairproduct",
        "case_soc_rashba_2d_sub_apbc",
    )
)


def _load_guard_module():
    spec = importlib.util.spec_from_file_location(
        "_v36_topology_guard_under_test", _GUARD_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def guard_mod():
    return _load_guard_module()


def _load_real_manifest():
    manifest_path = os.path.join(_REAL_CASE_DIR, "composite_element.json")
    with open(manifest_path) as fp:
        return json.load(fp)


def _run_cli(workspace, manifest_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    return subprocess.run(
        [
            sys.executable, _GUARD_PATH,
            "--workspace", workspace,
            "--mode", "g4",
            "--composite-manifest", manifest_path,
        ],
        capture_output=True, text=True, env=env,
    )


def _write_manifest(path, manifest):
    with open(path, "w") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
        fp.write("\n")


def _make_stub_workspace(tmp_path, green_source):
    """Create a workspace with only hwave/green.npz linked from a real
    SCF output. Callers can then override the manifest or delete files
    to trigger specific negative paths."""
    hwave = tmp_path / "hwave"
    hwave.mkdir(parents=True, exist_ok=True)
    dst = hwave / "green.npz"
    dst.write_bytes(green_source.read_bytes())
    return str(tmp_path)


# ---------------------------------------------------------------------
# 1. Missing composite (magnitude falls below 0.8 * G_c_abs)
# ---------------------------------------------------------------------


def test_g4_rejects_missing_composite_element(tmp_path):
    """Manifest points at a composite element whose magnitude on the
    fresh workspace falls below 0.8 * G_c_abs (in this test, we bump
    G_c_abs 10x higher than the real value so no current-run element
    meets the floor)."""
    ws = _make_stub_workspace(
        tmp_path, green_source=(
            _load_real_manifest_dir_green_file()
        ),
    )
    manifest = _load_real_manifest()
    manifest["G_c_abs"] = 10.0 * manifest["G_c_abs"]
    m_path = os.path.join(ws, "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(ws, m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "composite element on current workspace" in res.stderr


# ---------------------------------------------------------------------
# 2. Same-spin composite in manifest.
# ---------------------------------------------------------------------


def test_g4_rejects_same_spin_composite(tmp_path):
    ws = _make_stub_workspace(
        tmp_path, green_source=_load_real_manifest_dir_green_file(),
    )
    manifest = _load_real_manifest()
    manifest["s_c"] = manifest["t_c"] = 0
    m_path = os.path.join(ws, "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(ws, m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "cross-spin" in res.stderr


# ---------------------------------------------------------------------
# 3. Manifest sub_offset_x does not differ between i and j.
# ---------------------------------------------------------------------


def test_g4_rejects_no_sub_offset_diff(tmp_path):
    ws = _make_stub_workspace(
        tmp_path, green_source=_load_real_manifest_dir_green_file(),
    )
    manifest = _load_real_manifest()
    manifest["sub_offset_i"] = [0, 0, 0]
    manifest["sub_offset_j"] = [0, 1, 0]  # same x, different y
    m_path = os.path.join(ws, "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(ws, m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "sub_offset_x differs" in res.stderr


# ---------------------------------------------------------------------
# 4. Manifest G_c_abs below the 1e-3 magnitude floor.
# ---------------------------------------------------------------------


def test_g4_rejects_low_magnitude_composite(tmp_path):
    ws = _make_stub_workspace(
        tmp_path, green_source=_load_real_manifest_dir_green_file(),
    )
    manifest = _load_real_manifest()
    manifest["G_c_abs"] = 5e-4  # below 1e-3
    m_path = os.path.join(ws, "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(ws, m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "magnitude floor" in res.stderr


# ---------------------------------------------------------------------
# 5. Mutation delta below T_M (T_M artificially inflated in manifest).
# ---------------------------------------------------------------------


def test_g4_rejects_mutation_below_T_M(tmp_path):
    ws = _make_stub_workspace(
        tmp_path, green_source=_load_real_manifest_dir_green_file(),
    )
    manifest = _load_real_manifest()
    # Inflate M-gauge-1's T_M so no reasonable current-run delta can
    # meet it.
    manifest["T_M_per_mutation"]["M-gauge-1"] = 10.0
    m_path = os.path.join(ws, "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(ws, m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "M-gauge-1" in res.stderr
    assert "T_M" in res.stderr


# ---------------------------------------------------------------------
# 6. Guard reads CURRENT workspace, NOT manifest-cached values.
# ---------------------------------------------------------------------


def test_g4_reads_current_workspace_not_manifest_alone(tmp_path):
    """Swap the workspace's green.npz for a random-noise green_sublattice
    of the same shape. The guard MUST reject because the recomputed
    G_current at the manifest coordinates no longer matches the
    manifest's cached G_c_abs; this proves the guard is reading the
    CURRENT workspace and not trusting the manifest values alone."""
    manifest = _load_real_manifest()
    real_green_path = os.path.join(_REAL_CASE_DIR, "output", "green.npz")
    with np.load(real_green_path) as gz:
        real_gs = gz["green_sublattice"]
    # Random-noise green_sublattice of the same shape.
    rng = np.random.default_rng(31337)
    noise = (
        rng.standard_normal(real_gs.shape).astype(np.complex128)
        + 1j * rng.standard_normal(real_gs.shape).astype(np.complex128)
    ) * 1e-4  # small so the composite magnitude collapses
    hwave = tmp_path / "hwave"
    hwave.mkdir(parents=True, exist_ok=True)
    np.savez(hwave / "green.npz", green_sublattice=noise)
    m_path = str(tmp_path / "manifest.json")
    _write_manifest(m_path, manifest)
    res = _run_cli(str(tmp_path), m_path)
    assert res.returncode == 2, res.stderr
    assert res.stdout == "", res.stdout
    assert "composite element on current workspace" in res.stderr


# ---------------------------------------------------------------------
# Positive path (baseline): guard passes on the real fixture.
# ---------------------------------------------------------------------


def test_g4_passes_on_real_case_soc_rashba_2d_sub_apbc(tmp_path):
    """Sanity: the guard MUST PASS on the committed fixture with its own
    committed composite_element.json. If this fails, all six negative
    tests above are trivially satisfied for the wrong reason."""
    manifest_path = os.path.join(_REAL_CASE_DIR, "composite_element.json")
    res = _run_cli(_REAL_CASE_DIR, manifest_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.startswith("G4 PASS mode=g4 "), res.stdout
    assert "artifact_source=hwave+bridge+composite-manifest" in res.stdout
    assert "helper=soc_apbc_topology_guard" in res.stdout


# ---------------------------------------------------------------------
# Helper: load the real fixture's green.npz for cloning.
# ---------------------------------------------------------------------


def _load_real_manifest_dir_green_file():
    from pathlib import Path

    return Path(_REAL_CASE_DIR) / "output" / "green.npz"
