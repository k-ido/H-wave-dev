"""Tests for density_check (spec section 5.1)."""
from __future__ import annotations

import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pytest

from tools._uhfk_to_mvmc.density_check import (
    density_from_amplitudes,
    compare_against_onebodyg_uhf,
    compare_against_onebodyg_uhf_general,
    DensityMismatchError,
    parse_uhf_cisajs_dat,
)


_DATA_DIR = Path(__file__).parent / "data"

# Tracked snapshots of H-wave outputs for case_soc_rashba_2d_sub, so v3.5
# gauge tests do not depend on the gitignored
# tests/validation/**/work/ tree (Codex Rev.1 finding).
_V35_GREEN_PATH = str(_DATA_DIR / "v35_case_soc_rashba_2d_sub_green.npz")
_V35_EIGEN_PATH = str(_DATA_DIR / "v35_case_soc_rashba_2d_sub_eigen.npz")
_V35_OCC_PATH = str(_DATA_DIR / "v35_case_soc_rashba_2d_sub_occupation.npz")


def test_density_from_amplitudes_pbc_k0_only():
    """Single occupied k=0 plane wave on L=4 → G^sigma_{ij} = 1/L."""
    L = 4
    A = np.full((L, 1), 1.0 / np.sqrt(L), dtype=np.complex128)
    G = density_from_amplitudes(A)
    expected = np.full((L, L), 1.0 / L, dtype=np.complex128)
    np.testing.assert_allclose(G, expected, atol=1e-12)


def test_density_check_soc_convention_asymmetric():
    """Under ``is_soc_mode=True``, density_check reads greenone.dat rows
    on mVMC spin-block order (``all_i = i + s * Ns``) and passes when the
    bridge output matches. If the bridge used interleaved order, the
    asymmetric golden values would force a numerical mismatch above
    tolerance."""
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 3] = 0.3
    G_all[1, 2] = 0.2 + 0.1j
    G_all[2, 1] = 0.2 - 0.1j
    G_all[3, 0] = 0.3

    golden = str(_DATA_DIR / "soc_greenone_asymmetric_golden.dat")

    # Should pass without raising.
    compare_against_onebodyg_uhf_general(
        G_all, golden, tol=1e-10, is_soc_mode=True,
    )


def test_density_check_soc_rejects_beyond_tol():
    """SOC mode with an s != t entry mismatched beyond tolerance -> raise
    DensityMismatchError."""
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 3] = 0.3
    G_all[1, 2] = 0.2 + 0.1j + 3e-10  # deliberate error above 1e-10 tol
    G_all[2, 1] = 0.2 - 0.1j
    G_all[3, 0] = 0.3

    golden = str(_DATA_DIR / "soc_greenone_asymmetric_golden.dat")

    with pytest.raises(DensityMismatchError, match="differ"):
        compare_against_onebodyg_uhf_general(
            G_all, golden, tol=1e-10, is_soc_mode=True,
        )


def test_density_check_v3_still_rejects_s_neq_t_when_not_soc_mode():
    """v3 behavior preserved under ``is_soc_mode=False``: if the reference
    greenone.dat has a non-zero s != t entry while the bridge's G_all is
    zero in that off-diagonal block (which is the natural v3 Sz-diagonal
    output), the comparison must still flag the mismatch.

    This exercises the implicit scope-violation guard that used to be
    described in the v3 docstring: v3 bridge builds a spin-block-diagonal
    G_all, so any non-zero s != t entry in the reference will differ from
    the bridge's zero and trigger DensityMismatchError.
    """
    Ns = 2
    G_all = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    G_all[0, 0] = 0.5
    G_all[1, 1] = 0.5
    G_all[2, 2] = 0.5
    G_all[3, 3] = 0.5  # spin-block-diagonal only, off-diagonal blocks all 0

    # Reference has a non-zero s != t entry, incompatible with v3 scope.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".dat", delete=False
    ) as tmp:
        tmp.write("0 0 0 0  0.5 0.0\n")
        tmp.write("1 0 1 0  0.5 0.0\n")
        tmp.write("0 1 0 1  0.5 0.0\n")
        tmp.write("1 1 1 1  0.5 0.0\n")
        tmp.write("0 0 1 1  0.3 0.0\n")  # off-diagonal block: not zero
        path = tmp.name

    try:
        with pytest.raises(DensityMismatchError, match="differ"):
            compare_against_onebodyg_uhf_general(
                G_all, path, tol=1e-10, is_soc_mode=False,
            )
    finally:
        os.unlink(path)


def test_gauge_lift_fft_round_trip():
    """Codex v3.5 spec §9 Phase A step 1: fftn(ifftn(...)) must recover
    the original green_sublattice at 1e-14 element-wise, matching H-wave's
    forward-normalized FFT convention (_save_green at uhfk.py:2500-2510).
    """
    import numpy as np
    # Load the tracked H-wave green snapshot for case_soc_rashba_2d_sub.
    gp = np.load(_V35_GREEN_PATH)
    green_sublattice = gp["green_sublattice"]
    assert green_sublattice.ndim == 5, (
        f"expected raw H-wave 5D green shape (nvol, ns, norb, ns, norb); "
        f"got {green_sublattice.shape}"
    )
    # Extract the SOC block: under enable_spin_orbital=True, ns=1 and
    # norb=2*norb_phys; the physical density lives in
    # green_sub[:, 0, :, 0, :].
    gs_soc = green_sublattice[:, 0, :, 0, :]
    # Reshape onto the 3 spatial axes for 3D FFT.
    # case_soc_rashba_2d_sub: CellShape=[6,4,1], SubShape=[2,2,1] -> L_folded=[3,2,1]
    L_folded = np.array([3, 2, 1], dtype=np.int64)
    gs_reshaped = gs_soc.reshape(
        L_folded[0], L_folded[1], L_folded[2],
        gs_soc.shape[1], gs_soc.shape[2],
    )
    G_k = np.fft.ifftn(gs_reshaped, axes=(0, 1, 2), norm="forward")
    gs_round = np.fft.fftn(G_k, axes=(0, 1, 2), norm="forward")
    assert np.allclose(gs_round, gs_reshaped, atol=1e-14), (
        "FFT round-trip residual exceeds 1e-14: "
        f"max abs diff = {np.max(np.abs(gs_round - gs_reshaped)):.3e}"
    )


def test_gauge_lift_sub_offset_sign_audit():
    """Codex v3.5 spec §9 Phase A step 2: swapping the sign of sub_offset
    in gauge_lift's phase must make the reconstructed G disagree with the
    correct shipping A density by a Rashba-scale amount.
    """
    import numpy as np
    from tools._uhfk_to_mvmc.density_check import gauge_lift

    # Load tracked case_soc_rashba_2d_sub H-wave snapshot.
    gp = np.load(_V35_GREEN_PATH)
    green_sublattice = gp["green_sublattice"]

    # Fixed inputs derived from input.toml.
    cell_shape = np.array([6, 4, 1], dtype=np.int64)
    subshape = np.array([2, 2, 1], dtype=np.int64)
    Ns = int(np.prod(cell_shape))  # 24
    site_positions = np.array(
        [[x, y, z] for z in range(1) for y in range(4) for x in range(6)],
        dtype=np.int64,
    )
    assert site_positions.shape == (Ns, 3)
    folded_cell_of = lambda r_phys: r_phys // subshape

    # Sanity: gauge_lift on a site-diagonal, spin-diagonal element must be
    # a real positive occupation number close to Ncond/2Ns = 8/48 = 1/6.
    diag = gauge_lift(
        green_sublattice, site_i=0, spin_i=0, site_j=0, spin_j=0,
        subshape=subshape, cell_shape=cell_shape,
        site_positions=site_positions, folded_cell_of=folded_cell_of,
    )
    assert abs(diag.imag) < 1e-10, f"diagonal G[0,0] must be real; got {diag}"
    assert 0.05 < diag.real < 0.3, (
        f"expected diagonal occupation ~1/6; got {diag.real}"
    )

    # Mutation: gauge_lift with sub_offset sign flipped. The mutation is
    # only visible on off-diagonal-in-sub_offset pairs. Pick two sites
    # with distinct sub_offset.
    # site 0 -> r=(0,0,0), fc=(0,0,0), so=(0,0,0)
    # site 1 -> r=(1,0,0), fc=(0,0,0), so=(1,0,0)  <-- distinct sub_offset
    def gauge_lift_flipped(site_i, spin_i, site_j, spin_j):
        # Same as gauge_lift but so_i and so_j are negated in the phase.
        gs_soc = green_sublattice[:, 0, :, 0, :]
        L_folded = cell_shape // subshape
        gs = gs_soc.reshape(
            L_folded[0], L_folded[1], L_folded[2],
            gs_soc.shape[1], gs_soc.shape[2],
        )
        G_k = np.fft.ifftn(gs, axes=(0, 1, 2), norm="forward")
        r_i = site_positions[site_i]
        r_j = site_positions[site_j]
        fc_i = r_i // subshape
        fc_j = r_j // subshape
        so_i = r_i - fc_i * subshape
        so_j = r_j - fc_j * subshape
        folded_orb_i = int(so_i[0] + subshape[0] * (so_i[1] + subshape[1] * so_i[2]))
        folded_orb_j = int(so_j[0] + subshape[0] * (so_j[1] + subshape[1] * so_j[2]))
        aa = 2 * folded_orb_i + spin_i
        bb = 2 * folded_orb_j + spin_j
        accum = 0.0j
        for kx in range(L_folded[0]):
            for ky in range(L_folded[1]):
                for kz in range(L_folded[2]):
                    k_vec = 2.0 * np.pi * np.array([kx, ky, kz]) / L_folded
                    # FLIPPED sub_offset sign:
                    r_diff = (fc_j.astype(np.float64) - so_j.astype(np.float64)) \
                             - (fc_i.astype(np.float64) - so_i.astype(np.float64))
                    phase = np.exp(-1j * np.dot(k_vec, r_diff))
                    accum += G_k[kx, ky, kz, aa, bb] * phase
        return accum / float(np.prod(L_folded))

    # Compare on site (0, up) vs site (1, up) which have different sub_offset.
    correct = gauge_lift(
        green_sublattice, site_i=0, spin_i=0, site_j=1, spin_j=0,
        subshape=subshape, cell_shape=cell_shape,
        site_positions=site_positions, folded_cell_of=folded_cell_of,
    )
    mutated = gauge_lift_flipped(0, 0, 1, 0)
    delta = abs(correct - mutated)
    assert delta > 1e-3, (
        f"sub_offset sign flip should perturb the G element by >1e-3 "
        f"(Rashba scale). Got delta = {delta:.3e}, correct = {correct}, "
        f"mutated = {mutated}"
    )


def test_gauge_lift_absolute_scale_and_full_2Ns_2Ns_coverage():
    """Codex v3.5 spec §9 Phase A step 3: build the full 2Ns × 2Ns
    G_ref matrix from gauge_lift and compare to G_direct =
    np.conj(A_ship) @ A_ship.T over every element, including s != t
    cross-spin blocks. Asserts 5e-13 element-wise agreement (FP roundoff
    floor for the 24-site 3D FFT + 8-column A @ A.T sum; still 200x
    tighter than the 1e-10 ship gate in
    compare_against_green_sublattice), 1% absolute scale, and > 0.01
    cross-spin magnitude (proves the test actually exercises Rashba SOC
    off-diagonals).
    """
    import numpy as np
    from tools._uhfk_to_mvmc.density_check import gauge_lift
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_slater_orbitals, build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows

    gp = np.load(_V35_GREEN_PATH)
    eigen = np.load(_V35_EIGEN_PATH)
    occ = np.load(_V35_OCC_PATH)

    green_sublattice = gp["green_sublattice"]
    eigenvector = eigen["eigenvector"]
    eigenvalue = eigen["eigenvalue"]
    wavevector_index = eigen["wavevector_index"]

    cell_shape = np.array([6, 4, 1], dtype=np.int64)
    subshape = np.array([2, 2, 1], dtype=np.int64)
    Ns = int(np.prod(cell_shape))  # 24
    Ncond = 8
    site_positions = np.array(
        [[x, y, z] for z in range(1) for y in range(4) for x in range(6)],
        dtype=np.int64,
    )
    folded_cell_of = lambda r_phys: r_phys // subshape

    # Step 1: build the shipping A via build_slater_orbitals SOC branch.
    stepped, _ = step_occupation(
        occ["occupation"], eigenvalue, occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[Ncond],
        is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        wavevector_index, np.zeros(3, dtype=np.float64),
        cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(partner_rows, wavevector_index)
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows, is_soc_mode=True,
    )
    A_ship = build_slater_orbitals(
        wavevector_index=wavevector_index,
        eigenvector=eigenvector,
        column_spin=occ["column_spin"],
        site_positions=site_positions.astype(np.int64),
        cell_shape=cell_shape,
        subshape=subshape,
        theta=np.zeros(3, dtype=np.float64),
        pair_list=pair_list,
        is_soc_mode=True,
    )
    G_direct = np.conj(A_ship) @ A_ship.T  # (2Ns, 2Ns)

    # Step 2: build G_ref via gauge_lift for all (site_i, spin_i, site_j, spin_j).
    G_ref = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    for site_i in range(Ns):
        for spin_i in (0, 1):
            for site_j in range(Ns):
                for spin_j in (0, 1):
                    all_i = site_i + spin_i * Ns
                    all_j = site_j + spin_j * Ns
                    G_ref[all_i, all_j] = gauge_lift(
                        green_sublattice, site_i, spin_i, site_j, spin_j,
                        subshape=subshape, cell_shape=cell_shape,
                        site_positions=site_positions,
                        folded_cell_of=folded_cell_of,
                    )

    # Codex v3.5 spec §9 Phase A step 3 assertions:
    max_diff = np.max(np.abs(G_ref - G_direct))
    assert max_diff < 5e-13, (
        f"|G_ref - G_direct|_max = {max_diff:.3e} > 5e-13; the gauge_lift "
        "does not match the shipping-A density."
    )

    scale_ref = np.max(np.abs(G_ref))
    scale_direct = np.max(np.abs(G_direct))
    assert abs(scale_ref - scale_direct) / max(scale_direct, 1e-30) < 0.01, (
        f"|G_ref|_max = {scale_ref}, |G_direct|_max = {scale_direct}; "
        "absolute-scale mismatch > 1% suggests a 1/N_folded scale bug."
    )

    # Cross-spin coverage: pick site 0 (up) x site 0 (dn) etc., cross-spin
    # block G[0..Ns, Ns..2Ns].
    cross_spin_block = G_ref[:Ns, Ns:]
    cross_max = np.max(np.abs(cross_spin_block))
    assert cross_max > 0.01, (
        f"|G_ref[s!=t]|_max = {cross_max:.3e} <= 0.01; the fixture does not "
        "exercise Rashba SOC off-diagonals as expected. Verify fixture "
        "parameters (Ncond, alpha_R) or the pack aa/bb formula."
    )


def test_gauge_lift_catches_orientation_swap_mutation():
    """Codex v3.5 spec §9 Phase A step 4: build G with the WRONG
    orientation A_ship @ A_ship.conj().T (transposed/conjugated) and
    verify gauge_lift disagrees by > 1e-3 element-wise. This proves the
    gate catches the exact Codex v3.5 spec Rev.4 finding.
    """
    import numpy as np
    from tools._uhfk_to_mvmc.density_check import gauge_lift
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_slater_orbitals, build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows

    gp = np.load(_V35_GREEN_PATH)
    eigen = np.load(_V35_EIGEN_PATH)
    occ = np.load(_V35_OCC_PATH)

    green_sublattice = gp["green_sublattice"]

    cell_shape = np.array([6, 4, 1], dtype=np.int64)
    subshape = np.array([2, 2, 1], dtype=np.int64)
    Ns = int(np.prod(cell_shape))
    Ncond = 8
    site_positions = np.array(
        [[x, y, z] for z in range(1) for y in range(4) for x in range(6)],
        dtype=np.int64,
    )
    folded_cell_of = lambda r_phys: r_phys // subshape

    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[Ncond],
        is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        eigen["wavevector_index"], np.zeros(3, dtype=np.float64),
        cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(partner_rows, eigen["wavevector_index"])
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows, is_soc_mode=True,
    )
    A_ship = build_slater_orbitals(
        wavevector_index=eigen["wavevector_index"],
        eigenvector=eigen["eigenvector"],
        column_spin=occ["column_spin"],
        site_positions=site_positions.astype(np.int64),
        cell_shape=cell_shape,
        subshape=subshape,
        theta=np.zeros(3, dtype=np.float64),
        pair_list=pair_list,
        is_soc_mode=True,
    )
    G_wrong = A_ship @ np.conj(A_ship).T  # WRONG orientation

    # Compare G_wrong to gauge_lift on a couple of cross-spin elements
    # where the orientation matters.
    max_disagree = 0.0
    for site_i in range(0, Ns, 4):  # sample every 4th site to keep it fast
        for site_j in range(0, Ns, 4):
            for spin_i, spin_j in [(0, 1), (1, 0)]:
                all_i = site_i + spin_i * Ns
                all_j = site_j + spin_j * Ns
                gref = gauge_lift(
                    green_sublattice, site_i, spin_i, site_j, spin_j,
                    subshape=subshape, cell_shape=cell_shape,
                    site_positions=site_positions,
                    folded_cell_of=folded_cell_of,
                )
                disagree = abs(G_wrong[all_i, all_j] - gref)
                if disagree > max_disagree:
                    max_disagree = disagree
    assert max_disagree > 1e-3, (
        f"orientation-swap mutation must disagree with gauge_lift by "
        f">1e-3 on some cross-spin element; got max disagreement "
        f"{max_disagree:.3e}. If this passes trivially, either A_ship "
        "or gauge_lift is real-only for the fixture (Rashba should give "
        "complex off-diagonals)."
    )


def test_compare_against_green_sublattice_soc_mode_passes_on_correct_A_ship():
    """Codex v3.5 spec §3: compare_against_green_sublattice with
    is_soc_sublattice_mode=True must PASS at 1e-10 tolerance when the
    input G is the correct shipping-A density.
    """
    import numpy as np
    from tools._uhfk_to_mvmc.density_check import (
        compare_against_green_sublattice,
    )
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_slater_orbitals, build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows

    eigen = np.load(_V35_EIGEN_PATH)
    occ = np.load(_V35_OCC_PATH)

    cell_shape = np.array([6, 4, 1], dtype=np.int64)
    subshape = np.array([2, 2, 1], dtype=np.int64)
    Ns = int(np.prod(cell_shape))
    Ncond = 8
    site_positions = np.array(
        [[x, y, z] for z in range(1) for y in range(4) for x in range(6)],
        dtype=np.int64,
    )

    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[Ncond],
        is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        eigen["wavevector_index"], np.zeros(3, dtype=np.float64),
        cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(partner_rows, eigen["wavevector_index"])
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows, is_soc_mode=True,
    )
    A_ship = build_slater_orbitals(
        wavevector_index=eigen["wavevector_index"],
        eigenvector=eigen["eigenvector"],
        column_spin=occ["column_spin"],
        site_positions=site_positions.astype(np.int64),
        cell_shape=cell_shape,
        subshape=subshape,
        theta=np.zeros(3, dtype=np.float64),
        pair_list=pair_list,
        is_soc_mode=True,
    )
    G_ship = np.conj(A_ship) @ A_ship.T

    # Must pass at 1e-10 without raising.
    compare_against_green_sublattice(
        G_ship,
        green_path=_V35_GREEN_PATH,
        Ns=Ns,
        tol=1e-10,
        is_soc_sublattice_mode=True,
        cell_shape=cell_shape,
        subshape=subshape,
        site_positions=site_positions,
    )


def test_compare_against_green_sublattice_soc_mode_fires_on_orientation_swap():
    """Codex v3.5 spec §3 + §6.2b: compare_against_green_sublattice with
    is_soc_sublattice_mode=True must RAISE DensityMismatchError when
    the input G is transposed/conjugated (orientation swap mutation).
    """
    import numpy as np
    from tools._uhfk_to_mvmc.density_check import (
        compare_against_green_sublattice, DensityMismatchError,
    )
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_slater_orbitals, build_pair_list, compute_canonical_reps,
    )
    from tools._uhfk_to_mvmc.occupation_step import step_occupation
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows

    eigen = np.load(_V35_EIGEN_PATH)
    occ = np.load(_V35_OCC_PATH)

    cell_shape = np.array([6, 4, 1], dtype=np.int64)
    subshape = np.array([2, 2, 1], dtype=np.int64)
    Ns = int(np.prod(cell_shape))
    Ncond = 8
    site_positions = np.array(
        [[x, y, z] for z in range(1) for y in range(4) for x in range(6)],
        dtype=np.int64,
    )

    stepped, _ = step_occupation(
        occ["occupation"], eigen["eigenvalue"], occ["column_spin"],
        occ["column_mu_group"], float(occ["T"]),
        ncond_per_group=[Ncond],
        is_soc_mode=True,
    )
    partner_rows, _ = find_partner_rows(
        eigen["wavevector_index"], np.zeros(3, dtype=np.float64),
        cell_shape // subshape,
    )
    canonical, _ = compute_canonical_reps(partner_rows, eigen["wavevector_index"])
    pair_list = build_pair_list(
        stepped, occ["column_spin"], canonical, partner_rows, is_soc_mode=True,
    )
    A_ship = build_slater_orbitals(
        wavevector_index=eigen["wavevector_index"],
        eigenvector=eigen["eigenvector"],
        column_spin=occ["column_spin"],
        site_positions=site_positions.astype(np.int64),
        cell_shape=cell_shape,
        subshape=subshape,
        theta=np.zeros(3, dtype=np.float64),
        pair_list=pair_list,
        is_soc_mode=True,
    )
    G_wrong = A_ship @ np.conj(A_ship).T  # WRONG orientation.
    with pytest.raises(DensityMismatchError):
        compare_against_green_sublattice(
            G_wrong,
            green_path=_V35_GREEN_PATH,
            Ns=Ns,
            tol=1e-10,
            is_soc_sublattice_mode=True,
            cell_shape=cell_shape,
            subshape=subshape,
            site_positions=site_positions,
        )
