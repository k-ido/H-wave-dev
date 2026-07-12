"""SOC + APBC + SubShape topology + mutation guard for v3.6 G4 gate.

Spec §4.3 (topology composite element) + §6.2 (G4 wiring) + §5.3b
(PASS record metadata) + plan Task 2g (semantic upgrade).

Semantic contract (Phase 2g):

  1. Load ``composite_element.json`` via ``--composite-manifest``.
  2. Load CURRENT-run ``${WORK_DIR}/hwave/green.npz`` (NEVER trust
     manifest values alone).
  3. Assert the manifest's ``(i_c, s_c, j_c, t_c)`` element exists on
     the current run with ``abs(G_current) >= 0.8 * G_c_abs``.
  4. Assert ``s_c != t_c``, ``sub_offset_x(i_c) != sub_offset_x(j_c)``,
     ``abs(G_current) >= 1e-3``.
  5. Run all 10 mutations (M-gauge-1..5 + M-ship-1..5) on the CURRENT
     run at the composite element; assert each ``delta_M >= T_M`` per
     the manifest's ``T_M_per_mutation``.
  6. On success emit the anchored PASS record per §5.3b. On any
     failure exit code 2 + empty stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# v3.6 spec §5.3b single source of truth: G4 metadata.
G4_MODE = "g4"
G4_ARTIFACT_SOURCE = "hwave+bridge+composite-manifest"
G4_HELPER = "soc_apbc_topology_guard"


def _load_manifest(path):
    with open(path) as fp:
        return json.load(fp)


def _folded_cell(r_phys, subshape):
    return r_phys // subshape


def _gauge_lift_element(green_sublattice, i, s, j, t, subshape, cell_shape,
                        site_positions, boundary_theta, mutator="baseline"):
    """Element-level gauge_lift with optional M-gauge mutation.

    See scripts/phase2_produce_manifest.py for the derivation of the
    per-mutation phase surgery; this guard reproduces those formulas so
    the workspace-run deltas can be compared directly with the manifest
    T_M values.
    """
    gs_soc = green_sublattice[:, 0, :, 0, :]
    L_folded = cell_shape // subshape
    LFX, LFY, LFZ = int(L_folded[0]), int(L_folded[1]), int(L_folded[2])
    gs = gs_soc.reshape(LFX, LFY, LFZ, gs_soc.shape[1], gs_soc.shape[2])
    G_k = np.fft.ifftn(gs, axes=(0, 1, 2), norm="forward")

    r_phys_i = site_positions[i].astype(np.int64)
    r_phys_j = site_positions[j].astype(np.int64)
    fc_i = _folded_cell(r_phys_i, subshape)
    fc_j = _folded_cell(r_phys_j, subshape)
    so_i = r_phys_i - fc_i * subshape
    so_j = r_phys_j - fc_j * subshape
    folded_orb_i = int(
        so_i[0] + subshape[0] * (so_i[1] + subshape[1] * so_i[2])
    )
    folded_orb_j = int(
        so_j[0] + subshape[0] * (so_j[1] + subshape[1] * so_j[2])
    )
    aa = 2 * folded_orb_i + int(s)
    bb = 2 * folded_orb_j + int(t)

    dr_folded_base = (fc_j.astype(np.float64) + so_j.astype(np.float64)) \
                     - (fc_i.astype(np.float64) + so_i.astype(np.float64))
    dr_full = r_phys_j.astype(np.float64) - r_phys_i.astype(np.float64)
    L_full = (subshape * L_folded).astype(np.float64)

    dr_folded = dr_folded_base.copy()
    theta = np.asarray(boundary_theta, dtype=np.float64)
    twist_L = L_full.copy()
    twist_dr = dr_full.copy()
    twist_sign = -1.0

    if mutator == "M-gauge-1":
        twist_sign = +1.0
    elif mutator == "M-gauge-2":
        theta = theta / (2.0 * np.pi)
    elif mutator == "M-gauge-3":
        twist_L = L_folded.astype(np.float64)
    elif mutator == "M-gauge-4":
        so_diff = so_j.astype(np.float64) - so_i.astype(np.float64)
        dr_folded = (
            fc_j.astype(np.float64) - fc_i.astype(np.float64) - so_diff
        )
    elif mutator == "M-gauge-5":
        twist_dr = dr_folded_base

    phase_twist = np.exp(twist_sign * 1j * np.dot(theta, twist_dr / twist_L))
    accum = 0.0j
    for kx in range(LFX):
        for ky in range(LFY):
            for kz in range(LFZ):
                k_vec = 2.0 * np.pi * np.array(
                    [kx, ky, kz], dtype=np.float64
                ) / L_folded
                phase_folded = np.exp(-1j * np.dot(k_vec, dr_folded))
                accum += G_k[kx, ky, kz, aa, bb] * phase_folded
    accum *= phase_twist
    return accum / float(LFX * LFY * LFZ)


def _build_A_ship_mutated(mutator, eigen, occ, site_positions, cell_shape,
                          subshape, ncond, boundary_theta):
    """Real M-ship mutation: rebuild the shipping A matrix with the
    mutation applied to the SOC branch of build_slater_orbitals so the
    G4 gate exercises the exact Phase 4 mutation formulas, not a
    closed-form surrogate that can disagree on fixture-specific phase
    cancellations."""
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows
    from tools._uhfk_to_mvmc.sublattice_unfold import decode_physical_site

    L_folded = (cell_shape // subshape).astype(np.float64)
    L_phys = cell_shape.astype(np.float64)
    theta = np.asarray(boundary_theta, dtype=np.float64)
    Ns = site_positions.shape[0]

    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[int(ncond)], is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        eigen["wavevector_index"], theta, cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(
        partner_rows, eigen["wavevector_index"],
    )
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows,
        is_soc_mode=True,
    )
    wvi = eigen["wavevector_index"].astype(np.int64)
    ev = eigen["eigenvector"].astype(np.complex128)
    nvol_folded = wvi.shape[0]
    inv_sqrt = 1.0 / np.sqrt(float(nvol_folded))

    folded_cell = np.empty((Ns, 3), dtype=np.int64)
    sub_offset = np.empty((Ns, 3), dtype=np.int64)
    folded_orb = np.empty(Ns, dtype=np.int64)
    for i in range(Ns):
        fc, so = decode_physical_site(site_positions[i], subshape)
        folded_cell[i] = fc
        sub_offset[i] = so
        folded_orb[i] = (
            so[0] + subshape[0] * (so[1] + subshape[1] * so[2])
        )

    if mutator == "M-ship-1":
        phys_arg = np.einsum(
            "d,id->i", theta / L_phys, site_positions.astype(np.float64),
        )
        phys_dn = np.exp(+1j * phys_arg)  # sign flipped
    elif mutator == "M-ship-2":
        theta_wrong = theta / (2.0 * np.pi)
        phys_arg = np.einsum(
            "d,id->i", theta_wrong / L_phys,
            site_positions.astype(np.float64),
        )
        phys_dn = np.exp(-1j * phys_arg)
    elif mutator == "M-ship-3":
        phys_arg = np.einsum(
            "d,id->i", theta / L_folded,
            site_positions.astype(np.float64),
        )
        phys_dn = np.exp(-1j * phys_arg)
    else:
        phys_arg = np.einsum(
            "d,id->i", theta / L_phys, site_positions.astype(np.float64),
        )
        phys_dn = np.exp(-1j * phys_arg)

    k_folded_all = 2.0 * np.pi * wvi.astype(np.float64) / L_folded
    if mutator == "M-ship-4":
        kf_dot_r = np.einsum(
            "kd,id->ki", k_folded_all,
            (folded_cell - sub_offset).astype(np.float64),
        )
    elif mutator == "M-ship-5":
        kf_dot_r = np.einsum(
            "kd,id->ki", k_folded_all, folded_cell.astype(np.float64),
        )
    else:
        kf_dot_r = np.einsum(
            "kd,id->ki", k_folded_all,
            (folded_cell + sub_offset).astype(np.float64),
        )

    plane_wave = (
        np.exp(-1j * kf_dot_r) * inv_sqrt * phys_dn[np.newaxis, :]
    )
    A = np.zeros(
        (2 * Ns, 2 * len(pair_list)), dtype=np.complex128,
    )
    for p, pair in enumerate(pair_list):
        for col_idx, member in enumerate(pair):
            k_row, alpha = int(member[0]), int(member[1])
            slater_col = 2 * p + col_idx
            pw_m = plane_wave[k_row]
            for spin in (0, 1):
                hw = (2 * folded_orb + spin).astype(np.int64)
                v = np.array(
                    [ev[k_row, int(hw[i]), alpha] for i in range(Ns)],
                    dtype=np.complex128,
                )
                A[spin * Ns:(spin + 1) * Ns, slater_col] = pw_m * v
    return A


def _resolve_workspace(workspace):
    """Plan Task 3e: canonicalize the workspace path AND every consumed
    artifact; reject any resolved artifact under ``tests/data``. The
    shared guard covers the workspace root, subdir-symlink cases, and
    single-artifact-symlink cases."""
    from pathlib import Path

    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from _snapshot_guard import (
        reject_snapshot_workspace,
        SnapshotWorkspaceRejected,
    )

    try:
        reject_snapshot_workspace(
            workspace, helper_name="soc_apbc_topology_guard",
        )
    except SnapshotWorkspaceRejected as exc:
        raise ValueError(str(exc)) from exc
    ws = Path(workspace).resolve(strict=True)
    return str(ws)


def soc_apbc_topology_guard(workspace, composite_manifest_path):
    """Run the semantic composite + mutation check on the CURRENT-run
    workspace. Returns a dict with ``max_abs_delta`` and ``tol`` fields
    for the PASS record; raises on any semantic failure so the CLI
    entry point can convert to exit code 2 + empty stdout per §5.3b."""
    _resolve_workspace(workspace)
    manifest = _load_manifest(composite_manifest_path)
    i_c = int(manifest["i_c"])
    s_c = int(manifest["s_c"])
    j_c = int(manifest["j_c"])
    t_c = int(manifest["t_c"])
    G_c_abs_manifest = float(manifest["G_c_abs"])
    T_M_manifest = manifest["T_M_per_mutation"]

    # Cross-spin invariant (from manifest — defensively re-checked).
    if s_c == t_c:
        raise ValueError(
            f"composite manifest requires cross-spin; got s_c=t_c={s_c}"
        )

    cell_shape = np.asarray(manifest["cell_shape"], dtype=np.int64)
    subshape = np.asarray(manifest["sub_shape"], dtype=np.int64)
    theta = np.asarray(manifest["theta_radians"], dtype=np.float64)
    Nsite = int(np.prod(cell_shape))

    # sub_offset_x differs (from manifest — defensively re-checked).
    so_i_x = int(manifest["sub_offset_i"][0])
    so_j_x = int(manifest["sub_offset_j"][0])
    if so_i_x == so_j_x:
        raise ValueError(
            f"composite manifest requires sub_offset_x differs; "
            f"got so_i_x=so_j_x={so_i_x}"
        )

    # Magnitude floor from manifest.
    if G_c_abs_manifest < 1e-3:
        raise ValueError(
            f"composite manifest G_c_abs = {G_c_abs_manifest:.3e} is below "
            "the §4.3 magnitude floor of 1e-3."
        )

    # Load CURRENT-run green_sublattice — NEVER trust manifest values.
    green_npz_path = os.path.join(workspace, "hwave", "green.npz")
    if not os.path.isfile(green_npz_path):
        # Try the SCF standard layout (case_dir/output/green.npz).
        alt = os.path.join(workspace, "output", "green.npz")
        if not os.path.isfile(alt):
            raise FileNotFoundError(
                f"soc_apbc_topology_guard: missing {green_npz_path} "
                f"and {alt}"
            )
        green_npz_path = alt
    green_sublattice = np.load(green_npz_path)["green_sublattice"]

    site_positions = np.array(
        [
            (ix, iy, iz)
            for iz in range(int(cell_shape[2]))
            for iy in range(int(cell_shape[1]))
            for ix in range(int(cell_shape[0]))
        ],
        dtype=np.int64,
    )

    G_current = _gauge_lift_element(
        green_sublattice, i_c, s_c, j_c, t_c,
        subshape=subshape, cell_shape=cell_shape,
        site_positions=site_positions, boundary_theta=theta,
        mutator="baseline",
    )
    G_current_abs = float(abs(G_current))
    magnitude_ratio = G_current_abs / max(G_c_abs_manifest, 1e-16)
    if G_current_abs < 0.8 * G_c_abs_manifest:
        raise ValueError(
            f"composite element on current workspace: "
            f"|G_current[i_c,s_c,j_c,t_c]| = {G_current_abs:.3e} < "
            f"0.8 * G_c_abs_manifest = {0.8 * G_c_abs_manifest:.3e} "
            f"(magnitude_ratio = {magnitude_ratio:.3f}); composite "
            "did not survive; regenerate manifest with a fresh SCF run."
        )

    # M-gauge mutations run at element level on the current-run
    # green_sublattice.
    max_shortfall = 0.0
    for m_gauge in ("M-gauge-1", "M-gauge-2", "M-gauge-3", "M-gauge-4",
                    "M-gauge-5"):
        G_mut = _gauge_lift_element(
            green_sublattice, i_c, s_c, j_c, t_c,
            subshape=subshape, cell_shape=cell_shape,
            site_positions=site_positions, boundary_theta=theta,
            mutator=m_gauge,
        )
        delta = float(abs(G_mut - G_current))
        T = float(T_M_manifest[m_gauge])
        if delta < T:
            raise ValueError(
                f"{m_gauge}: delta = {delta:.3e} < T_M = {T:.3e} "
                "on current workspace; composite lost mutation "
                "sensitivity vs manifest."
            )
        shortfall = T / max(delta, 1e-16)
        max_shortfall = max(max_shortfall, shortfall)

    # M-ship mutations require the shipping A on the current run. Rebuild
    # from eigen/occupation NPZ files under the workspace (SCF standard
    # layout ``output/`` or bridge-linked ``hwave/``); the baseline A and
    # 5 mutated A matrices are constructed once and the composite element
    # extracted from each.
    eigen_path = None
    occ_path = None
    for base in ("hwave", "output"):
        e = os.path.join(workspace, base, "eigen.npz")
        o = os.path.join(workspace, base, "occupation.npz")
        if os.path.isfile(e) and os.path.isfile(o):
            eigen_path, occ_path = e, o
            break
    if eigen_path is None:
        raise FileNotFoundError(
            "soc_apbc_topology_guard: missing eigen.npz + occupation.npz "
            "under workspace's hwave/ or output/ directory; cannot run "
            "M-ship mutations."
        )
    ncond = int(manifest.get("ncond", 2 * int(manifest["N_pairs"])))
    eigen = np.load(eigen_path)
    occ = np.load(occ_path)
    A_base = _build_A_ship_mutated(
        "baseline", eigen, occ, site_positions, cell_shape, subshape,
        ncond, theta,
    )

    # Codex adversarial-review 2026-07-12 hardening: verify the
    # shadow-copy baseline equals the actual shipping A produced by
    # build_slater_orbitals BEFORE running the mutation matrix. If the
    # shadow copy has drifted from the canonical kernel, a drift bug
    # in build_slater_orbitals is invisible to G4 otherwise.
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_pair_list, build_slater_orbitals, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows
    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[ncond], is_soc_mode=True,
    )
    partner_rows_ship, _ = find_partner_rows(
        eigen["wavevector_index"], theta, cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(
        partner_rows_ship, eigen["wavevector_index"],
    )
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows_ship,
        is_soc_mode=True,
    )
    A_ship_canonical = build_slater_orbitals(
        wavevector_index=eigen["wavevector_index"],
        eigenvector=eigen["eigenvector"],
        column_spin=occ["column_spin"],
        site_positions=site_positions,
        cell_shape=cell_shape,
        subshape=subshape,
        theta=theta,
        pair_list=pair_list,
        is_soc_mode=True,
    )
    shadow_drift = float(np.max(np.abs(A_base - A_ship_canonical)))
    if shadow_drift > 1e-10:
        raise ValueError(
            f"G4 shadow-copy baseline drifted from shipping "
            f"build_slater_orbitals: |A_shadow - A_ship|_max = "
            f"{shadow_drift:.3e} > 1e-10. Either the shadow copy or the "
            "shipping kernel has changed; re-align "
            "_build_A_ship_mutated with the shipping SOC branch or "
            "extract a shared kernel."
        )
    G_ship_base = np.conj(A_base) @ A_base.T
    for m_ship in ("M-ship-1", "M-ship-2", "M-ship-3", "M-ship-4",
                   "M-ship-5"):
        A_m = _build_A_ship_mutated(
            m_ship, eigen, occ, site_positions, cell_shape, subshape,
            ncond, theta,
        )
        G_ship_m = np.conj(A_m) @ A_m.T
        all_i = i_c + s_c * Nsite
        all_j = j_c + t_c * Nsite
        delta = float(abs(
            G_ship_m[all_i, all_j] - G_ship_base[all_i, all_j]
        ))
        T = float(T_M_manifest[m_ship])
        if delta < T:
            raise ValueError(
                f"{m_ship}: delta = {delta:.3e} < T_M = {T:.3e} "
                "on current workspace."
            )

    return {
        "max_abs_delta": max(0.0, 1.0 - magnitude_ratio),
        "tol": 0.2,  # 20% magnitude drift allowed per §4.3 addendum
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "v3.6 G4 topology + mutation guard (spec §4.3, §5.3b, §6.2). "
            "Semantic guard verifies the pinned composite element on the "
            "current-run workspace and runs the 10-mutation matrix."
        )
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument(
        "--composite-manifest", dest="composite_manifest",
        required=True,
    )
    args = parser.parse_args(argv)

    if args.mode != G4_MODE:
        print(
            f"soc_apbc_topology_guard: unsupported mode={args.mode}; "
            f"only {G4_MODE!r} is defined.",
            file=sys.stderr,
        )
        return 2

    try:
        result = soc_apbc_topology_guard(
            args.workspace, args.composite_manifest,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"soc_apbc_topology_guard: {e}", file=sys.stderr)
        return 2

    max_abs_delta = float(result.get("max_abs_delta", 0.0))
    tol = float(result.get("tol", 0.2))
    print(
        f"G4 PASS mode={G4_MODE} artifact_source={G4_ARTIFACT_SOURCE} "
        f"helper={G4_HELPER} max_abs_delta={max_abs_delta:.6e} "
        f"tol={tol:.6e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
