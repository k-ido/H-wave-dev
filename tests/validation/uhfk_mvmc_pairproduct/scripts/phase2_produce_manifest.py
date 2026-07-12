"""Phase 2d-f driver: pick composite element on the SCF-backed workspace,
run the 10 mutation matrix, write ``composite_element.json``.

Spec §4.3 composite condition + §4.4 mutation matrix + §6.2 manifest.
This script is a Phase 2 producer, NOT a v3.6 gate: it runs once when
the fixture is prepared, and the JSON it emits is committed alongside
the fixture. Phase 6 G4 reads that JSON verbatim and re-verifies the
composite + mutations on the fresh workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from tools._uhfk_to_mvmc.pair_product_density import (
    pair_product_density_from_F,
)


# Geometry constants for case_soc_rashba_2d_sub_apbc.
NSITE = 24
LX, LY, LZ = 6, 4, 1
BX, BY, BZ = 2, 2, 1
LFX, LFY, LFZ = LX // BX, LY // BY, LZ // BZ  # 3, 2, 1
N_PAIRS = 4  # Ncond = 8 -> N_pairs = 4 for closed-shell SOC.
THETA = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
CELL = np.array([LX, LY, LZ], dtype=np.int64)
SUB = np.array([BX, BY, BZ], dtype=np.int64)
LFULL = (SUB * np.array([LFX, LFY, LFZ], dtype=np.int64)).astype(np.float64)

SITE_POSITIONS = np.array(
    [(ix, iy, iz) for iz in range(LZ) for iy in range(LY) for ix in range(LX)],
    dtype=np.int64,
)


def _site_pos(i):
    return SITE_POSITIONS[i]


def _sub_offset(i):
    r = _site_pos(i)
    return int(r[0] % BX), int(r[1] % BY), int(r[2] % BZ)


def _folded_cell(r_phys):
    return r_phys // SUB


def _gauge_lift_full_G(green_sublattice):
    """Compute G_phys[all_i, all_j] over the full (48, 48) mVMC spin-block
    index using the shipping v3.6 gauge_lift with boundary_theta = (pi, 0, 0)."""
    from tools._uhfk_to_mvmc.density_check import gauge_lift

    G = np.zeros((2 * NSITE, 2 * NSITE), dtype=np.complex128)
    folded_cell_of = _folded_cell
    for i in range(NSITE):
        for s in (0, 1):
            for j in range(NSITE):
                for t in (0, 1):
                    val = gauge_lift(
                        green_sublattice, i, s, j, t,
                        subshape=SUB, cell_shape=CELL,
                        site_positions=SITE_POSITIONS,
                        folded_cell_of=folded_cell_of,
                        boundary_theta=THETA,
                    )
                    G[i + s * NSITE, j + t * NSITE] = val
    return G


def _pick_composite(G):
    best = None
    for i in range(NSITE):
        for s in (0, 1):
            for j in range(NSITE):
                for t in (0, 1):
                    if s == t:
                        continue
                    if _sub_offset(i)[0] == _sub_offset(j)[0]:
                        continue
                    all_i = i + s * NSITE
                    all_j = j + t * NSITE
                    mag = float(abs(G[all_i, all_j]))
                    if mag < 1e-3:
                        continue
                    if best is None or mag > best[5]:
                        best = (i, s, j, t, G[all_i, all_j], mag)
    if best is None:
        raise RuntimeError("no composite element satisfies §4.3 conditions")
    return best


def _gauge_lift_element(green_sublattice, i, s, j, t, mutator="baseline"):
    """Element-level gauge_lift with optional M-gauge mutation."""
    gs_soc = green_sublattice[:, 0, :, 0, :]
    L_folded = CELL // SUB
    gs = gs_soc.reshape(LFX, LFY, LFZ, gs_soc.shape[1], gs_soc.shape[2])
    G_k = np.fft.ifftn(gs, axes=(0, 1, 2), norm="forward")

    r_phys_i = _site_pos(i).astype(np.int64)
    r_phys_j = _site_pos(j).astype(np.int64)
    fc_i = _folded_cell(r_phys_i)
    fc_j = _folded_cell(r_phys_j)
    so_i = r_phys_i - fc_i * SUB
    so_j = r_phys_j - fc_j * SUB
    folded_orb_i = int(so_i[0] + SUB[0] * (so_i[1] + SUB[1] * so_i[2]))
    folded_orb_j = int(so_j[0] + SUB[0] * (so_j[1] + SUB[1] * so_j[2]))
    aa = 2 * folded_orb_i + int(s)
    bb = 2 * folded_orb_j + int(t)

    dr_folded_base = (fc_j.astype(np.float64) + so_j.astype(np.float64)) \
                     - (fc_i.astype(np.float64) + so_i.astype(np.float64))
    dr_full = r_phys_j.astype(np.float64) - r_phys_i.astype(np.float64)

    dr_folded = dr_folded_base.copy()
    theta = THETA.copy()
    twist_L = LFULL.copy()
    twist_dr = dr_full.copy()
    twist_sign = -1.0

    if mutator == "M-gauge-1":
        twist_sign = +1.0
    elif mutator == "M-gauge-2":
        theta = THETA / (2.0 * np.pi)
    elif mutator == "M-gauge-3":
        twist_L = (CELL // SUB).astype(np.float64)
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
                k_vec = 2.0 * np.pi * np.array([kx, ky, kz], dtype=np.float64) / (
                    CELL // SUB
                )
                phase_folded = np.exp(-1j * np.dot(k_vec, dr_folded))
                accum += G_k[kx, ky, kz, aa, bb] * phase_folded
    accum *= phase_twist
    return accum / float(LFX * LFY * LFZ)


def _build_A_ship_mutated(mutator, eigen, occ):
    """Real M-ship mutation: rebuild the shipping A matrix with the
    mutation applied to the SOC branch of build_slater_orbitals. The
    Phase 2 producer feeds these into the composite selector so the
    committed manifest's T_M thresholds match the REAL mutations Phase 4
    and Phase 6 will trip against, not a closed-form surrogate that can
    disagree on fixture-specific phase cancellations."""
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows
    from tools._uhfk_to_mvmc.sublattice_unfold import decode_physical_site

    L_folded = (CELL // SUB).astype(np.float64)
    L_phys = CELL.astype(np.float64)

    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[8], is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        eigen["wavevector_index"], THETA, CELL // SUB,
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

    folded_cell = np.empty((NSITE, 3), dtype=np.int64)
    sub_offset = np.empty((NSITE, 3), dtype=np.int64)
    folded_orb = np.empty(NSITE, dtype=np.int64)
    for i in range(NSITE):
        fc, so = decode_physical_site(SITE_POSITIONS[i], SUB)
        folded_cell[i] = fc
        sub_offset[i] = so
        folded_orb[i] = so[0] + SUB[0] * (so[1] + SUB[1] * so[2])

    if mutator == "M-ship-1":
        phys_arg = np.einsum(
            "d,id->i", THETA / L_phys, SITE_POSITIONS.astype(np.float64),
        )
        phys_dn = np.exp(+1j * phys_arg)  # sign flipped
    elif mutator == "M-ship-2":
        theta_wrong = THETA / (2.0 * np.pi)
        phys_arg = np.einsum(
            "d,id->i", theta_wrong / L_phys,
            SITE_POSITIONS.astype(np.float64),
        )
        phys_dn = np.exp(-1j * phys_arg)
    elif mutator == "M-ship-3":
        phys_arg = np.einsum(
            "d,id->i", THETA / L_folded,
            SITE_POSITIONS.astype(np.float64),
        )
        phys_dn = np.exp(-1j * phys_arg)
    else:
        phys_arg = np.einsum(
            "d,id->i", THETA / L_phys, SITE_POSITIONS.astype(np.float64),
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
        (2 * NSITE, 2 * len(pair_list)), dtype=np.complex128,
    )
    for p, pair in enumerate(pair_list):
        for col_idx, member in enumerate(pair):
            k_row, alpha = int(member[0]), int(member[1])
            slater_col = 2 * p + col_idx
            pw_m = plane_wave[k_row]
            for spin in (0, 1):
                hw = (2 * folded_orb + spin).astype(np.int64)
                v = np.array(
                    [ev[k_row, int(hw[i]), alpha] for i in range(NSITE)],
                    dtype=np.complex128,
                )
                A[spin * NSITE:(spin + 1) * NSITE, slater_col] = pw_m * v
    return A


def _pick_composite_strong(G_gauge, G_ship_base, G_ship_mut, green_sublattice):
    """§4.3 composite gate + strong-mutation gate.

    Requires every M-gauge-1..5 delta AND every M-ship-1..5 delta to
    exceed the §4.4 10% floor at the candidate composite. Otherwise
    Phase 4 mutation matrix trips on the fixture-specific composite even
    though the mutation IS being exercised — it just happens to shift
    the composite by less than 10% |G_base|. This selector guarantees
    the committed composite is strong enough for every mutation."""
    best = None
    for i in range(NSITE):
        so_i_x = _sub_offset(i)[0]
        for s in (0, 1):
            for j in range(NSITE):
                so_j_x = _sub_offset(j)[0]
                for t in (0, 1):
                    if s == t:
                        continue
                    if so_i_x == so_j_x:
                        continue
                    all_i = i + s * NSITE
                    all_j = j + t * NSITE
                    G_c = G_gauge[all_i, all_j]
                    mag = float(abs(G_c))
                    if mag < 1e-3:
                        continue
                    T_floor = max(1e-5, 0.10 * mag)
                    # M-gauge mutations (real element-level formulas).
                    G_base_gl = _gauge_lift_element(
                        green_sublattice, i, s, j, t, "baseline",
                    )
                    m_ok = True
                    for m in (
                        "M-gauge-1", "M-gauge-2", "M-gauge-3",
                        "M-gauge-4", "M-gauge-5",
                    ):
                        G_mut = _gauge_lift_element(
                            green_sublattice, i, s, j, t, m,
                        )
                        if float(abs(G_mut - G_base_gl)) < T_floor:
                            m_ok = False
                            break
                    if not m_ok:
                        continue
                    # M-ship mutations (from precomputed A_mut matrices).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True, help="composite_element.json output")
    args = ap.parse_args()

    case = args.case
    print(f"Loading SCF outputs from {case}/output/...")
    green_sublattice = np.load(f"{case}/output/green.npz")["green_sublattice"]
    eigen = np.load(f"{case}/output/eigen.npz")
    occ = np.load(f"{case}/output/occupation.npz")

    print("Building G_gauge from green_sublattice (v3.6 gauge_lift APBC)...")
    G_gauge = _gauge_lift_full_G(green_sublattice)
    print(
        f"  |G_gauge|_max = {np.max(np.abs(G_gauge)):.3e}; "
        f"trace = {np.trace(G_gauge).real:.4f}"
    )

    print(
        "Building baseline + M-ship-1..5 mutated shipping A matrices "
        "(real mutations, not surrogates)..."
    )
    A_base = _build_A_ship_mutated("baseline", eigen, occ)
    G_ship_base = np.conj(A_base) @ A_base.T
    G_ship_mut = {}
    for m in (
        "M-ship-1", "M-ship-2", "M-ship-3", "M-ship-4", "M-ship-5",
    ):
        A_m = _build_A_ship_mutated(m, eigen, occ)
        G_ship_mut[m] = np.conj(A_m) @ A_m.T
        print(f"  {m} A built.")

    print(
        "Selecting composite element (§4.3 + strong-mutation gate: "
        "every M-gauge/M-ship delta >= 10% |G_base|)..."
    )
    i_c, s_c, j_c, t_c, gv, mag = _pick_composite_strong(
        G_gauge, G_ship_base, G_ship_mut, green_sublattice,
    )
    print(f"  composite: i={i_c} s={s_c} j={j_c} t={t_c}")
    print(f"    |G| = {mag:.4e}, G = {gv}")
    print(
        f"    sub_offset(i) = {_sub_offset(i_c)}, "
        f"sub_offset(j) = {_sub_offset(j_c)}"
    )

    all_i = i_c + s_c * NSITE
    all_j = j_c + t_c * NSITE
    G_c_base = complex(G_gauge[all_i, all_j])

    # T_M policy (spec §4.4): T_M = max(1e-5, 0.10 * |G_base|). The
    # Phase 4 unit tests and Phase 6 workspace G4 gate both consume this
    # threshold verbatim. The strong-mutation selector above guarantees
    # every real mutation delta clears T_M at manifest write time; the
    # committed manifest merely records the actual per-mutation delta so
    # a Phase 6 workspace regression can be diagnosed by comparing.
    T_floor = max(1e-5, 0.10 * float(abs(G_c_base)))
    T_M = {}
    delta_M = {}
    print("Running M-gauge mutations at composite element (real formulas)...")
    for m in (
        "M-gauge-1", "M-gauge-2", "M-gauge-3", "M-gauge-4", "M-gauge-5",
    ):
        G_mut = _gauge_lift_element(
            green_sublattice, i_c, s_c, j_c, t_c, m
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
        "_spec_ref": (
            "docs/superpowers/specs/"
            "2026-07-09-uhfk-mvmc-pairproduct-general-v36-design.md "
            "§4.3 §4.4 §6.2"
        ),
        "_fixture": os.path.relpath(case),
        "N_pairs": N_PAIRS,
        "ncond": 2 * N_PAIRS,
        "theta_radians": [float(x) for x in THETA],
        "cell_shape": [LX, LY, LZ],
        "sub_shape": [BX, BY, BZ],
        "i_c": int(i_c),
        "s_c": int(s_c),
        "j_c": int(j_c),
        "t_c": int(t_c),
        "G_c_abs": float(abs(G_c_base)),
        "G_c_real": float(G_c_base.real),
        "G_c_imag": float(G_c_base.imag),
        "T_M_per_mutation": T_M,
        "delta_M_per_mutation_at_producer_time": delta_M,
        "sub_offset_i": list(_sub_offset(i_c)),
        "sub_offset_j": list(_sub_offset(j_c)),
        "site_position_i": [int(x) for x in _site_pos(i_c)],
        "site_position_j": [int(x) for x in _site_pos(j_c)],
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(args.out, "w") as fp:
        fp.write(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {args.out}")
    print(f"sha256 = {sha}")
    all_ok = all(delta_M[m] >= T_M[m] for m in delta_M)
    print(f"Manifest {'PASSES self-check' if all_ok else 'has SUB-THRESHOLD entries'}")


if __name__ == "__main__":
    main()
