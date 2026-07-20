"""Phase 2d-f driver: pick composite element on the SCF-backed workspace,
run the mutation matrix, write ``composite_element.json``.

Spec §4.3 composite condition + §4.4 mutation matrix + §6.2 manifest
(docs/superpowers/specs/2026-07-09-uhfk-mvmc-pairproduct-general-v36-design.md).
v3.7 extends this with the §4 addendum (per-direction dr requirement)
and the §4b 30-entry per-direction mutation matrix
(docs/superpowers/specs/2026-07-12-uhfk-mvmc-pairproduct-general-v37-design.md).

This script is a Phase 2 producer, NOT a v3.6/v3.7 gate: it runs once
when the fixture is prepared, and the JSON it emits is committed
alongside the fixture. Phase 6 G4 (``soc_apbc_topology_guard.py``)
reads that JSON verbatim and re-verifies the composite + mutations on
the fresh workspace.

Schema selection is automatic from the fixture's own ``input.toml``:
a fixture with 0 or 1 active APBC direction (``theta_d != 0`` for at
most one ``d``) gets the v3.6 10-entry whole-vector mutation schema,
selected and scored by the BYTE-FOR-BYTE original v3.6 algorithm
(``_pick_composite_v36`` below is a mechanical parameterization of the
original code -- same formulas, same iteration order, same tie break).
A fixture with 2 or 3 active directions gets the v3.7 30-entry
per-direction schema (``_pick_composite_v37``). No CLI flag is needed:
the shape of ``BoundaryCondition`` alone determines the schema, so the
shipped v3.6 fixture (single-direction APBC) is unaffected by the
v3.7 extension.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

from tools._uhfk_to_mvmc.transfer_hermiticity import (
    check_transfer_dat_hermiticity,
    TransferHermiticityError,
)


# ---------------------------------------------------------------------
# Load soc_apbc_topology_guard.py as a module so the SAME mutator
# kernels the Phase 6 G4 gate uses (_gauge_lift_element,
# _build_A_ship_mutated, and the mutation-id tuples) drive composite
# selection and manifest T_M / delta computation here. A single source
# of truth for the mutation math eliminates the risk of the producer's
# formulas silently drifting from the guard's -- exactly the failure
# mode the guard's own shadow-copy-drift check exists to catch on the
# shipping build_slater_orbitals side.
# ---------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_GUARD_PATH = os.path.join(_SCRIPTS_DIR, "soc_apbc_topology_guard.py")
_guard_spec = importlib.util.spec_from_file_location(
    "_phase2_topology_guard", _GUARD_PATH,
)
guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(guard)


_V36_SPEC_REF = (
    "docs/superpowers/specs/"
    "2026-07-09-uhfk-mvmc-pairproduct-general-v36-design.md "
    "§4.3 §4.4 §6.2"
)
_V37_SPEC_REF = (
    "docs/superpowers/specs/"
    "2026-07-12-uhfk-mvmc-pairproduct-general-v37-design.md "
    "§4 §4b §6.2 (extends v3.6 §4.3 §4.4 §6.2)"
)
_AXIS_CHARS = "xyz"


def _make_site_positions(cell):
    """Enumerate (x, y, z) integer coordinates in the SAME order the
    bridge's green/eigen/occupation NPZ arrays index physical sites
    (x fastest, then y, then z). MUST match
    ``soc_apbc_topology_guard.soc_apbc_topology_guard``'s construction
    exactly -- both scripts must agree on which physical site a given
    integer index refers to, since the producer's ``i_c``/``j_c`` pins
    are consumed verbatim by the guard.
    """
    return np.array(
        [
            (ix, iy, iz)
            for iz in range(int(cell[2]))
            for iy in range(int(cell[1]))
            for ix in range(int(cell[0]))
        ],
        dtype=np.int64,
    )


def _sub_offset(i, site_positions, sub):
    r = site_positions[i]
    return int(r[0] % sub[0]), int(r[1] % sub[1]), int(r[2] % sub[2])


def _gauge_lift_full_G(green_sublattice, nsite, cell, sub, theta, site_positions):
    """Compute G_phys[all_i, all_j] over the full (2*nsite, 2*nsite)
    mVMC spin-block index using the shipping v3.6 gauge_lift. Used
    only to screen composite candidates by magnitude; §4 / §4.3
    filters and the strong-mutation gate are applied by the caller.
    """
    from tools._uhfk_to_mvmc.density_check import gauge_lift

    def _folded_cell_of(r_phys):
        return r_phys // sub

    G = np.zeros((2 * nsite, 2 * nsite), dtype=np.complex128)
    for i in range(nsite):
        for s in (0, 1):
            for j in range(nsite):
                for t in (0, 1):
                    val = gauge_lift(
                        green_sublattice, i, s, j, t,
                        subshape=sub, cell_shape=cell,
                        site_positions=site_positions,
                        folded_cell_of=_folded_cell_of,
                        boundary_theta=theta,
                    )
                    G[i + s * nsite, j + t * nsite] = val
    return G


def _is_structurally_degenerate_mutator(base_id, axis, L_folded, sub):
    """True iff mutator ``base_id`` (``"M-gauge-4"`` or ``"M-ship-4"``)
    is PROVABLY unable to perturb ANY composite candidate on axis
    ``axis`` for this fixture's lattice, regardless of ``theta``.

    The v3.7 per-direction M-gauge-4 omits ``sub_offset[axis]`` from
    ``dr_folded``. Relative to baseline this shifts the folded-BZ phase
    argument by ``-so_diff[axis]``. M-ship-4 instead changes the per-site
    ``kf_dot_r`` position coefficient from 1 to 1/2; relative to
    baseline its shift is ``-sub_offset[axis] / 2``, and relative to
    M-ship-5's coefficient 0 it is ``+sub_offset[axis] / 2``. Every
    folded k-component is an integer multiple of
    ``2*pi / L_folded[axis]``. When ``SUB[axis] >= 2``, reachable
    offsets include 1, so each M-4 phase change can be nontrivial unless
    ``L_folded[axis] == 1``. When ``SUB[axis] <= 1``, no sublattice
    structure exists on that axis and every offset difference is zero,
    which is degenerate for a more basic reason.

    In particular, the shipped v3.7 lattice has ``L_folded=2`` and
    ``SUB=2`` on every axis. For M-gauge-4, a reachable
    ``so_diff=+/-1`` changes the folded phase by
    ``exp(+/- i*pi) = -1``. For M-ship-4, ``sub_offset=1`` changes it
    by ``exp(+/- i*pi/2)`` relative to baseline or M-ship-5. Neither
    v3.7 M-4 is therefore structurally degenerate or duplicated.
    The v3.6 whole-vector ids retain their frozen sub_offset-sign-flip
    semantics and never call this v3.7-only predicate.

    M-gauge-1/2/3/5 and M-ship-1/2/3/5 are unaffected. Mutations 1/2/3
    are theta-mediated and inactive axes are handled separately by the
    spec §4b policy. Mutation 5 is not covered by this M-4 predicate.
    """
    if base_id not in ("M-gauge-4", "M-ship-4"):
        return False
    if int(sub[axis]) <= 1:
        return True
    return int(L_folded[axis]) == 1


def _mutation_threshold(mutator_id, *, axis_char, is_active, is_degenerate,
                        threshold_floor):
    """Return the v3.7 threshold for one per-direction mutator.

    Inactive axes are policy-trivially-satisfied with threshold zero.
    An active-axis threshold must be strictly positive. Structural
    degeneracy therefore means the mutation set is inadequate for the
    lattice and is an authoring error, never a reason to emit a vacuous
    gate in a committed manifest.
    """
    if not is_active:
        return 0.0
    if is_degenerate:
        raise RuntimeError(
            f"{mutator_id} is structurally degenerate on active APBC "
            f"axis {axis_char}; inadequate mutation set for this lattice"
        )
    threshold = float(threshold_floor)
    if not np.isfinite(threshold):
        raise RuntimeError(
            f"{mutator_id} would receive non-finite T_M={threshold} on "
            f"active APBC axis {axis_char}; thresholds must be finite"
        )
    if threshold <= 0.0:
        raise RuntimeError(
            f"{mutator_id} would receive T_M={threshold} on active APBC "
            f"axis {axis_char}; active-axis thresholds must be positive"
        )
    return threshold


def _write_manifest_after_self_check(manifest, output_path):
    """Validate, self-check, and write one producer manifest.

    Returns zero only after policy validation, finite delta validation,
    and the mutation self-check all pass. A failure returns one without
    creating or replacing ``output_path``.
    """
    try:
        mutation_ids = guard._validate_manifest_numeric_policy(manifest)
        deltas = manifest.get("delta_M_per_mutation_at_producer_time")
        if not isinstance(deltas, dict):
            raise RuntimeError(
                "delta_M_per_mutation_at_producer_time must be a dict"
            )
        if set(deltas) != set(mutation_ids):
            raise RuntimeError(
                "delta_M_per_mutation_at_producer_time keys must match "
                "T_M_per_mutation exactly"
            )
        for field_name in ("G_c_real", "G_c_imag"):
            guard._require_finite_float(manifest[field_name], field_name)
        for mutator_id in mutation_ids:
            delta = guard._require_finite_float(
                deltas[mutator_id],
                f"delta_M_per_mutation_at_producer_time[{mutator_id!r}]",
            )
            threshold = guard._require_finite_float(
                manifest["T_M_per_mutation"][mutator_id],
                f"T_M_per_mutation[{mutator_id!r}]",
            )
            if delta < threshold:
                raise RuntimeError(
                    f"{mutator_id}: delta={delta!r} is below "
                    f"threshold={threshold!r}"
                )
        text = json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Manifest self-check FAILED: {exc}", file=sys.stderr)
        return 1

    with open(output_path, "w") as fp:
        fp.write(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {output_path}")
    print(f"sha256 = {sha}")
    print("Manifest PASSES self-check")
    return 0


# ---------------------------------------------------------------------
# v3.6 path: byte-for-byte port of the original single-direction-APBC
# selector + manifest builder. Used whenever a fixture has <= 1 active
# APBC direction. Formulas are unchanged; only module-level constants
# became explicit parameters and the mutator kernels are now imported
# from soc_apbc_topology_guard.py instead of being duplicated locally.
# ---------------------------------------------------------------------


def _pick_composite_v36(G_gauge, G_ship_base, G_ship_mut, green_sublattice,
                         nsite, cell, sub, theta, site_positions):
    """§4.3 composite gate + strong-mutation gate (v3.6 formulas).

    Requires every M-gauge-1..5 delta AND every M-ship-1..5 delta to
    exceed the §4.4 10% floor at the candidate composite. Otherwise
    Phase 4's mutation matrix trips on the fixture-specific composite
    even though the mutation IS being exercised -- it just happens to
    shift the composite by less than 10% |G_base|. This selector
    guarantees the committed composite is strong enough for every
    mutation.
    """
    best = None
    for i in range(nsite):
        so_i_x = _sub_offset(i, site_positions, sub)[0]
        for s in (0, 1):
            for j in range(nsite):
                so_j_x = _sub_offset(j, site_positions, sub)[0]
                for t in (0, 1):
                    if s == t:
                        continue
                    if so_i_x == so_j_x:
                        continue
                    all_i = i + s * nsite
                    all_j = j + t * nsite
                    G_c = G_gauge[all_i, all_j]
                    mag = float(abs(G_c))
                    if mag < 1e-3:
                        continue
                    T_floor = max(1e-5, 0.10 * mag)
                    G_base_gl = guard._gauge_lift_element(
                        green_sublattice, i, s, j, t,
                        subshape=sub, cell_shape=cell,
                        site_positions=site_positions,
                        boundary_theta=theta, mutator="baseline",
                    )
                    m_ok = True
                    for m in (
                        "M-gauge-1", "M-gauge-2", "M-gauge-3",
                        "M-gauge-4", "M-gauge-5",
                    ):
                        G_mut = guard._gauge_lift_element(
                            green_sublattice, i, s, j, t,
                            subshape=sub, cell_shape=cell,
                            site_positions=site_positions,
                            boundary_theta=theta, mutator=m,
                        )
                        if float(abs(G_mut - G_base_gl)) < T_floor:
                            m_ok = False
                            break
                    if not m_ok:
                        continue
                    for m in G_ship_mut:
                        d = float(abs(
                            G_ship_mut[m][all_i, all_j]
                            - G_ship_base[all_i, all_j]
                        ))
                        if d < T_floor:
                            m_ok = False
                            break
                    if not m_ok:
                        continue
                    if best is None or mag > best[5]:
                        best = (i, s, j, t, G_c, mag)
    if best is None:
        raise RuntimeError(
            "no composite element satisfies §4.3 + strong-mutation gate"
        )
    return best


# ---------------------------------------------------------------------
# v3.7 path: §4 addendum selector (per-direction dr requirement) +
# strong-mutation gate restricted to active-axis per-direction mutators.
# ---------------------------------------------------------------------


def _pick_composite_v37(G_gauge, G_ship_base, G_ship_mut, green_sublattice,
                         nsite, cell, sub, theta, site_positions,
                         active_axes, gating_mutator_ids):
    """v3.7 §4 addendum selector: for EACH active APBC direction d,
    require ``abs(r_j[d] - r_i[d]) > 0`` (spec §4). Strong-mutation
    gate restricted to ``gating_mutator_ids`` (all active-axis
    per-direction mutators). Inactive-axis mutations are
    policy-trivially-satisfied (§4b). A structurally degenerate mutator
    on an active axis is rejected as an inadequate mutation set before
    composite selection.

    v3.7 addendum extension (Phase 3c blocker fix): the composite must
    also survive a whole-vector twist gauge drop (theta -> 0). This
    rejects degenerate compositions where individual axis contributions
    cancel arithmetically (e.g. dr_x = -dr_y with theta_x = theta_y).
    Physically: gauge_lift's twist phase exp(-i theta . dr / L_full)
    must differ from 1 at this composite; otherwise a negative-control
    test that drops the twist gauge has zero statistical power there,
    even though the per-direction mutators above all fire individually.
    """
    L_full = (sub * (cell // sub)).astype(np.float64)
    best = None
    for i in range(nsite):
        r_i = site_positions[i]
        for s in (0, 1):
            for j in range(nsite):
                r_j = site_positions[j]
                if any(int(r_j[d]) == int(r_i[d]) for d in active_axes):
                    continue
                r_diff_full = (r_j - r_i).astype(np.float64)
                twist_arg = float(np.dot(theta, r_diff_full / L_full))
                if abs(np.angle(np.exp(-1j * twist_arg))) < 0.1:
                    # Whole-vector twist gauge drop (theta -> 0) would
                    # leave this composite's phase unperturbed (within
                    # 0.1 rad of 1) -- reject per the addendum above.
                    # Threshold note: achievable non-zero twist angles
                    # are integer multiples of pi/N for cubic side N;
                    # on N=4 that is ~0.785 rad, ~7x margin over 0.1.
                    # Re-derive relative to pi/N before reusing on
                    # much larger lattices (v3.8+ CellShape).
                    continue
                for t in (0, 1):
                    if s == t:
                        continue
                    all_i = i + s * nsite
                    all_j = j + t * nsite
                    G_c = G_gauge[all_i, all_j]
                    mag = float(abs(G_c))
                    if mag < 1e-3:
                        continue
                    T_floor = max(1e-5, 0.10 * mag)
                    G_base_gl = guard._gauge_lift_element(
                        green_sublattice, i, s, j, t,
                        subshape=sub, cell_shape=cell,
                        site_positions=site_positions,
                        boundary_theta=theta, mutator="baseline",
                    )
                    m_ok = True
                    for m in gating_mutator_ids:
                        if m.startswith("M-gauge-"):
                            G_mut = guard._gauge_lift_element(
                                green_sublattice, i, s, j, t,
                                subshape=sub, cell_shape=cell,
                                site_positions=site_positions,
                                boundary_theta=theta, mutator=m,
                            )
                            d_val = float(abs(G_mut - G_base_gl))
                        else:
                            d_val = float(abs(
                                G_ship_mut[m][all_i, all_j]
                                - G_ship_base[all_i, all_j]
                            ))
                        if d_val < T_floor:
                            m_ok = False
                            break
                    if not m_ok:
                        continue
                    if best is None or mag > best[5]:
                        best = (i, s, j, t, G_c, mag)
    if best is None:
        raise RuntimeError(
            "no composite element satisfies §4 v3.7 addendum + "
            "strong-mutation gate"
        )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True, help="composite_element.json output")
    args = ap.parse_args()

    case = args.case
    transfer_path = os.path.join(case, "Transfer.dat")
    try:
        check_transfer_dat_hermiticity(transfer_path)
    except (FileNotFoundError, TransferHermiticityError) as exc:
        print(
            f"phase2 producer: Transfer.dat Hermiticity check failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(f"Transfer.dat Hermiticity check: PASS ({transfer_path})")

    from tools._uhfk_to_mvmc.boundary_input import normalize_boundary_condition_list
    from tools._uhfk_to_mvmc.input_loader import load_input_toml

    merged = load_input_toml(os.path.join(case, "input.toml"))
    CELL = np.asarray(merged["CellShape"], dtype=np.int64)
    SUB = np.asarray(merged["SubShape"], dtype=np.int64)
    if CELL.shape != (3,) or SUB.shape != (3,):
        raise ValueError(
            f"{case}/input.toml: CellShape/SubShape must both be "
            f"length-3; got CellShape={CELL.tolist()} "
            f"SubShape={SUB.tolist()}"
        )
    if np.any(CELL % SUB != 0):
        raise ValueError(
            f"{case}/input.toml: CellShape {CELL.tolist()} is not "
            f"exactly divisible by SubShape {SUB.tolist()}"
        )
    NSITE = int(np.prod(CELL))
    L_FOLDED = CELL // SUB
    THETA = np.asarray(
        normalize_boundary_condition_list(merged.get("BoundaryCondition")),
        dtype=np.float64,
    )
    NCOND = int(merged["Ncond"])
    SITE_POSITIONS = _make_site_positions(CELL)

    active_axes = tuple(d for d in range(3) if abs(THETA[d]) > 1e-12)
    is_v37 = len(active_axes) >= 2
    schema_name = "v3.7-30entry" if is_v37 else "v3.6-10entry"
    print(
        f"Fixture geometry: CellShape={CELL.tolist()} "
        f"SubShape={SUB.tolist()} L_folded={L_FOLDED.tolist()} "
        f"Ncond={NCOND} theta={THETA.tolist()}"
    )
    print(
        f"Active APBC axes: {[_AXIS_CHARS[d] for d in active_axes]} "
        f"-> manifest schema = {schema_name}"
    )

    print(f"Loading SCF outputs from {case}/output/...")
    green_sublattice = np.load(f"{case}/output/green.npz")["green_sublattice"]
    eigen = np.load(f"{case}/output/eigen.npz")
    occ = np.load(f"{case}/output/occupation.npz")

    print("Building G_gauge from green_sublattice (v3.6 gauge_lift APBC)...")
    G_gauge = _gauge_lift_full_G(
        green_sublattice, NSITE, CELL, SUB, THETA, SITE_POSITIONS,
    )
    print(
        f"  |G_gauge|_max = {np.max(np.abs(G_gauge)):.3e}; "
        f"trace = {np.trace(G_gauge).real:.4f}"
    )

    print("Building baseline shipping A matrix (real, not a surrogate)...")
    A_base = guard._build_A_ship_mutated(
        "baseline", eigen, occ, SITE_POSITIONS, CELL, SUB, NCOND, THETA,
    )
    N_PAIRS = A_base.shape[1] // 2
    if N_PAIRS != NCOND // 2:
        raise RuntimeError(
            f"derived N_pairs={N_PAIRS} from the shipping A matrix "
            f"width does not match Ncond // 2 = {NCOND // 2}; the "
            "closed-shell assumption behind the manifest's 'ncond' "
            "field is violated for this fixture"
        )
    G_ship_base = np.conj(A_base) @ A_base.T

    if is_v37:
        axis_chars = tuple(_AXIS_CHARS[d] for d in active_axes)
        needed_ship_ids = tuple(
            f"M-ship-{n}-{c}" for n in range(1, 6) for c in axis_chars
        )
        gating_mutator_ids = tuple(
            f"M-gauge-{n}-{c}" for n in range(1, 6) for c in axis_chars
        ) + needed_ship_ids
        for mid in gating_mutator_ids:
            base_id, axis = guard._split_mutator_id(mid)
            _mutation_threshold(
                mid,
                axis_char=_AXIS_CHARS[axis],
                is_active=True,
                is_degenerate=_is_structurally_degenerate_mutator(
                    base_id, axis, L_FOLDED, SUB,
                ),
                threshold_floor=1e-5,
            )
    else:
        needed_ship_ids = tuple(f"M-ship-{n}" for n in range(1, 6))
        gating_mutator_ids = None  # v3.6 path checks all 5 unconditionally.

    print(
        f"Building {len(needed_ship_ids)} mutated shipping A matrices "
        "(real mutations, not surrogates)..."
    )
    G_ship_mut = {}
    for m in needed_ship_ids:
        A_m = guard._build_A_ship_mutated(
            m, eigen, occ, SITE_POSITIONS, CELL, SUB, NCOND, THETA,
        )
        G_ship_mut[m] = np.conj(A_m) @ A_m.T
        print(f"  {m} A built.")

    if is_v37:
        print(
            "Selecting composite element (§4 v3.7 addendum + "
            f"strong-mutation gate over {len(gating_mutator_ids)} "
            "active per-direction mutators)..."
        )
        i_c, s_c, j_c, t_c, gv, mag = _pick_composite_v37(
            G_gauge, G_ship_base, G_ship_mut, green_sublattice,
            NSITE, CELL, SUB, THETA, SITE_POSITIONS,
            active_axes, gating_mutator_ids,
        )
    else:
        print(
            "Selecting composite element (§4.3 + strong-mutation gate: "
            "every M-gauge/M-ship delta >= 10% |G_base|)..."
        )
        i_c, s_c, j_c, t_c, gv, mag = _pick_composite_v36(
            G_gauge, G_ship_base, G_ship_mut, green_sublattice,
            NSITE, CELL, SUB, THETA, SITE_POSITIONS,
        )
    print(f"  composite: i={i_c} s={s_c} j={j_c} t={t_c}")
    print(f"    |G| = {mag:.4e}, G = {gv}")
    print(
        f"    sub_offset(i) = {_sub_offset(i_c, SITE_POSITIONS, SUB)}, "
        f"sub_offset(j) = {_sub_offset(j_c, SITE_POSITIONS, SUB)}"
    )

    all_i = i_c + s_c * NSITE
    all_j = j_c + t_c * NSITE
    G_c_base = complex(G_gauge[all_i, all_j])

    # T_M policy (spec §4.4): T_M = max(1e-5, 0.10 * |G_base|) on every
    # mutation the fixture actually exercises. The strong-mutation
    # selector above guarantees every gated mutation's delta clears
    # T_M at manifest write time; the committed manifest records the
    # actual per-mutation delta so a Phase 6 workspace regression can
    # be diagnosed by comparing.
    T_floor = max(1e-5, 0.10 * float(abs(G_c_base)))
    T_M = {}
    delta_M = {}
    notes = {}

    if is_v37:
        print("Running the 30-entry v3.7 per-direction mutation matrix...")
        for mid in guard._V37_MUTATION_IDS:
            base_id, axis = guard._split_mutator_id(mid)
            axis_char = _AXIS_CHARS[axis]
            is_active = axis in active_axes
            degenerate = _is_structurally_degenerate_mutator(
                base_id, axis, L_FOLDED, SUB,
            )
            T_M[mid] = _mutation_threshold(
                mid,
                axis_char=axis_char,
                is_active=is_active,
                is_degenerate=degenerate,
                threshold_floor=T_floor,
            )
            if not is_active:
                delta_M[mid] = 0.0
                notes[mid] = (
                    f"trivially satisfied under theta_{axis_char} = 0 "
                    "(spec v3.7 §4b): not gated, delta not computed."
                )
                continue
            if mid.startswith("M-gauge-"):
                G_mut = guard._gauge_lift_element(
                    green_sublattice, i_c, s_c, j_c, t_c,
                    subshape=SUB, cell_shape=CELL,
                    site_positions=SITE_POSITIONS, boundary_theta=THETA,
                    mutator=mid,
                )
                d = float(abs(G_mut - G_c_base))
            else:
                d = float(abs(
                    G_ship_mut[mid][all_i, all_j]
                    - G_ship_base[all_i, all_j]
                ))
            delta_M[mid] = d
            notes[mid] = (
                f"active APBC direction theta_{axis_char} = "
                f"{float(THETA[axis]):.6f} rad"
            )
            print(f"  {mid}: delta = {d:.3e}, T_M = {T_M[mid]:.3e}")
    else:
        print("Running M-gauge mutations at composite element (real formulas)...")
        for m in (
            "M-gauge-1", "M-gauge-2", "M-gauge-3", "M-gauge-4", "M-gauge-5",
        ):
            G_mut = guard._gauge_lift_element(
                green_sublattice, i_c, s_c, j_c, t_c,
                subshape=SUB, cell_shape=CELL,
                site_positions=SITE_POSITIONS, boundary_theta=THETA,
                mutator=m,
            )
            d = float(abs(G_mut - G_c_base))
            T_M[m] = T_floor
            delta_M[m] = d
            print(f"  {m}: delta = {d:.3e}, T_M = {T_floor:.3e}")

        print(
            "Recording M-ship mutations at composite element from "
            "precomputed A_mut matrices (real mutations)..."
        )
        for m in (
            "M-ship-1", "M-ship-2", "M-ship-3", "M-ship-4", "M-ship-5",
        ):
            d = float(abs(
                G_ship_mut[m][all_i, all_j] - G_ship_base[all_i, all_j]
            ))
            T_M[m] = T_floor
            delta_M[m] = d
            print(f"  {m}: delta = {d:.3e}, T_M = {T_floor:.3e}")

    manifest = {
        "_spec_ref": _V37_SPEC_REF if is_v37 else _V36_SPEC_REF,
        "_fixture": os.path.relpath(case),
        "manifest_schema": schema_name,
        "N_pairs": N_PAIRS,
        "ncond": NCOND,
        "theta_radians": [float(x) for x in THETA],
        "active_apbc_axes": [_AXIS_CHARS[d] for d in active_axes],
        "cell_shape": [int(x) for x in CELL],
        "sub_shape": [int(x) for x in SUB],
        "i_c": int(i_c),
        "s_c": int(s_c),
        "j_c": int(j_c),
        "t_c": int(t_c),
        "G_c_abs": float(abs(G_c_base)),
        "G_c_real": float(G_c_base.real),
        "G_c_imag": float(G_c_base.imag),
        "T_M_per_mutation": T_M,
        "delta_M_per_mutation_at_producer_time": delta_M,
        "sub_offset_i": list(_sub_offset(i_c, SITE_POSITIONS, SUB)),
        "sub_offset_j": list(_sub_offset(j_c, SITE_POSITIONS, SUB)),
        "site_position_i": [int(x) for x in SITE_POSITIONS[i_c]],
        "site_position_j": [int(x) for x in SITE_POSITIONS[j_c]],
    }
    if is_v37:
        manifest["T_M_notes"] = notes
    return _write_manifest_after_self_check(manifest, args.out)


if __name__ == "__main__":
    sys.exit(main())
