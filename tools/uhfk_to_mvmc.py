"""H-wave UHFk → mVMC PairProduct (AntiParallel Slater) bridge CLI.

Spec: docs/superpowers/specs/2026-06-30-uhfk-mvmc-pairproduct-bridge-design.md
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

# Allow running from repo root: tools/ is a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools._uhfk_to_mvmc.input_loader import (
    load_input_toml, derive_ne_per_group, load_geometry_uhf,
)
from tools._uhfk_to_mvmc.occupation_step import (
    step_occupation, OccupationGuardError,
)
from tools._uhfk_to_mvmc.orbitalidx_reader import (
    parse_orbitalidx_def, OrbitalidxFormatError,
)
from tools._uhfk_to_mvmc.fij_builder import (
    build_amplitudes, build_fij_phys,
)
from tools._uhfk_to_mvmc.output_writer import (
    aggregate_orbital_params, write_zqp_orbital,
)
from tools._uhfk_to_mvmc.density_check import (
    density_from_amplitudes, compare_against_onebodyg_uhf,
    DensityMismatchError,
)
from tools._uhfk_to_mvmc.boundary_input import (
    BoundaryInputError,
    check_eigen_twist_consistency,
    normalize_boundary_condition_list,
)


def _column_spin_to_mu_group_is_bijective(column_spin, column_mu_group):
    """True iff column_spin=0 cols share one mu_group and column_spin=1
    cols share a different mu_group (bijective spin<->mu_group)."""
    ups = set(int(g) for g, s in zip(column_mu_group, column_spin) if int(s) == 0)
    downs = set(int(g) for g, s in zip(column_mu_group, column_spin) if int(s) == 1)
    return len(ups) == 1 and len(downs) == 1 and ups.isdisjoint(downs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="H-wave UHFk → mVMC PairProduct bridge"
    )
    parser.add_argument("--input", required=True, help="input.toml path")
    parser.add_argument("--eigen", required=True, help="eigen.npz path")
    parser.add_argument("--occupation", required=True,
                        help="occupation.npz path")
    parser.add_argument("--geometry", required=True,
                        help="geometry_uhf.dat path")
    parser.add_argument("--orbitalidx", required=True,
                        help="orbitalidx.def path")
    parser.add_argument("--output", required=True,
                        help="zqp_orbital_uhfk.dat path")
    parser.add_argument("--check-density", dest="check_density",
                        action="store_true", default=True)
    parser.add_argument("--no-check-density", dest="check_density",
                        action="store_false")
    parser.add_argument("--onebodyg-uhf", default=None,
                        help="_UHF_cisajs.dat for --check-density")
    parser.add_argument(
        "--epsilon-noise", type=float, default=1.0e-8,
        help=(
            "Amplitude of uniform noise added to each averaged f_{idx} "
            "param before writing to lift rank-deficient F. Required for "
            "mVMC's Pfaffian Slater evaluation when F is built from a "
            "single (k, -k) shell; see ComplexUHF/output.c:274 for the "
            "same workaround. Default 1e-8 sits in the stable plateau "
            "noise <= 1e-7 where the residual mVMC-vs-UHF bias is "
            "smaller than 1 VMC stderr at NVMCSample=10000. Set to 0 "
            "to disable. (default: 1e-8)"
        ),
    )
    parser.add_argument(
        "--rng-seed", type=int, default=7919,
        help="Seed for the noise RNG (default: 7919, reproducible).",
    )
    # v3.1 spec §3.8: SOC path additionally emits mVMC trans.def from
    # H-wave Transfer.dat because StdFace's HubbardGC generator drops
    # Rashba s != t entries. Non-SOC path leaves vmcdry's trans.def
    # untouched, so both flags default to None and are ignored unless
    # is_soc_mode = True.
    parser.add_argument(
        "--transfer", default=None,
        help=(
            "H-wave Transfer.dat path. Required under SOC "
            "(enable_spin_orbital = true); ignored otherwise."
        ),
    )
    parser.add_argument(
        "--emit-trans", dest="emit_trans", default=None,
        help=(
            "Output mVMC trans.def path. Required under SOC; ignored "
            "otherwise. The emitter overwrites the existing trans.def "
            "so vmcdry.out must have already produced its Hamiltonian "
            "file layout at this location."
        ),
    )
    # v3.2 spec §1: under SOC + SubShape > [1, 1, 1] StdFace's
    # orbitalidxgen.def over-groups pair classes (assumes full-lattice
    # translation invariance, which SOC breaks under sublattice folding),
    # so the class-consistency guard (spec §4.3) fires with residuals of
    # order 1e-1. The bridge emits its own all-unique-classes
    # orbitalidxgen.def to this path and uses it in place of the caller-
    # supplied --orbitalidx for the F aggregation. Required under
    # SOC + SubShape > [1, 1, 1]; ignored otherwise.
    parser.add_argument(
        "--emit-orbitalidx", dest="emit_orbitalidx", default=None,
        help=(
            "Output all-unique-classes orbitalidxgen.def path. Required "
            "under SOC + SubShape > [1, 1, 1] (bypasses StdFace's class "
            "over-grouping); ignored otherwise."
        ),
    )
    args = parser.parse_args(argv)

    toml_param = load_input_toml(args.input)
    cell_shape = list(toml_param["CellShape"])
    sub_shape = list(toml_param.get("SubShape", cell_shape))
    try:
        boundary_theta_tuple = normalize_boundary_condition_list(
            toml_param.get("BoundaryCondition")
        )
    except (BoundaryInputError, ValueError) as e:
        print(f"ERROR (boundary): {e}", file=sys.stderr)
        return 2
    # v3.1 (spec §3.1): SOC dispatch flag. The v1 blanket reject is
    # replaced by the 6-case dispatch matrix below. SOC+APBC is fail-fast
    # right after boundary/eigen-twist canonicalization; SOC+antiparallel
    # is fail-fast at the dispatch decision. The --transfer / --emit-trans
    # required-args check for is_soc_mode lives inside the General branch
    # (spec §3.8): SOC+APBC and SOC+antiparallel reject before the emitter
    # is ever invoked, so those pre-dispatch rejects fire on their own
    # dedicated messages rather than a shared "missing --transfer" one.
    is_soc_mode = bool(toml_param.get("enable_spin_orbital", False))

    cell_shape_arr = np.array(cell_shape, dtype=np.int64)
    subshape_arr = np.array(sub_shape, dtype=np.int64)
    if np.any(cell_shape_arr % subshape_arr != 0):
        print(
            f"ERROR: SubShape = {sub_shape} does not divide "
            f"CellShape = {cell_shape} in every direction",
            file=sys.stderr,
        )
        return 2
    L_folded_arr = cell_shape_arr // subshape_arr
    subvol = int(np.prod(subshape_arr))

    ne_per_group = derive_ne_per_group(toml_param)
    # v3: N_up != N_down is legal for the General path (2Sz>0 A case).
    # The strict AntiParallel check is deferred to the AntiParallel
    # branch below so the General dispatch can service Sz-imbalanced UHF.

    theta = np.array(boundary_theta_tuple, dtype=np.float64)
    has_apbc = bool(np.any(np.abs(theta - np.pi) < 1e-12))
    # L passed to build_amplitudes / partner_index is the folded lattice
    # size (partner rows are computed on the folded BZ). CellShape is
    # preserved separately for the unfold path (spec §3.3).
    L = L_folded_arr

    eigen = np.load(args.eigen, allow_pickle=False)
    eigen_twist = (
        eigen["twist_offset"] if "twist_offset" in eigen.files else None
    )
    try:
        check_eigen_twist_consistency(boundary_theta_tuple, eigen_twist)
    except BoundaryInputError as e:
        print(f"ERROR (eigen twist): {e}", file=sys.stderr)
        return 2
    # v3.2 (spec §3.8): SOC + APBC is validated end-to-end via
    # ``emit_trans_def(boundary_theta=...)`` which applies the physical
    # wrap-phase to boundary-crossing rows so mVMC's ``H = -sum trans``
    # recovers the physical Hamiltonian on the periodic-site frame.
    # The prior v3.1 deferred reject is removed here.
    is_soc_sublattice_mode = is_soc_mode and any(
        int(s) != 1 for s in sub_shape
    )
    # v3.5 Phase D lift (2026-07-05): the v3.4 Rev.2 pre-dispatch reject
    # for SOC + SubShape > [1, 1, 1] has been removed. The v3.5 density
    # gate below now invokes
    # ``compare_against_green_sublattice(is_soc_sublattice_mode=True)``,
    # which lifts H-wave's ``green_sublattice`` to the physical basis via
    # ``gauge_lift`` and validates the SHIPPING ``conj(A) @ A.T`` directly
    # at 1e-10. This closes the "dual-A" gap flagged by Codex v3.4 Rev.2
    # (the previous gate validated a "reference A" that dropped
    # ``sub_offset``, not the shipping A). See
    # docs/superpowers/specs/2026-07-05-uhfk-mvmc-pairproduct-general-v35-design.md
    # §2-3 for the gauge derivation and §6.2b for the adversarial
    # negative-regression test that pins the gate's independence from the
    # shipping-A convention.
    #
    # v3.4 Rev.1 finding 2 defer (still in effect at v3.5): SOC + APBC +
    # SubShape > [1, 1, 1] is the composed triple combination and remains
    # UNVALIDATED. Only its two-way subsets are covered by fixtures:
    #   - case_soc_rashba_2d_nosub_apbc: SOC + APBC (SubShape = [1,1,1]),
    #     0.03% mVMC vs H-wave delta.
    #   - case_soc_rashba_2d_sub: SOC + SubShape (PBC), 0.22% delta.
    # The composed SOC+APBC+SubShape phase path has no fixture and no
    # end-to-end validation, so we fail-fast pre-dispatch. When a future
    # spec adds a SOC+APBC+SubShape fixture and empirical <H> validation,
    # remove this guard (v3.6+, see §8 non-goals).
    if is_soc_mode and has_apbc and any(int(s) != 1 for s in sub_shape):
        print(
            "ERROR: enable_spin_orbital = true + antiperiodic BC + "
            "SubShape > [1, 1, 1] is not yet validated. "
            "case_soc_rashba_2d_nosub_apbc covers SOC+APBC (0.03% delta); "
            "case_soc_rashba_2d_sub covers SOC+SubShape (0.22% delta); "
            "the composed SOC+APBC+SubShape phase path has no fixture. "
            "Deferred to a future spec.",
            file=sys.stderr,
        )
        return 2
    eigenvalue = eigen["eigenvalue"]
    eigenvector = eigen["eigenvector"]
    wavevector_index = eigen["wavevector_index"]

    occ = np.load(args.occupation, allow_pickle=False)
    occupation = occ["occupation"]
    column_spin = occ["column_spin"]
    column_mu_group = occ["column_mu_group"]
    T_scf = float(occ["T"])

    # v3 dispatch (spec §5.3): format-first + is_antiparallel_metadata tuple.
    from tools._uhfk_to_mvmc.orbitalidx_general_reader import (
        detect_orbitalidx_format,
        parse_orbitalidx_general_def,
        OrbitalidxFormatError as OrbitalidxGeneralFormatError,
    )
    from tools._uhfk_to_mvmc.general_fij_builder import (
        build_fij_general,
        build_pair_list,
        build_slater_orbitals,
        compute_canonical_reps,
        validate_general_prerequisites,
    )
    from tools._uhfk_to_mvmc.general_output_writer import (
        aggregate_general_orbital_params,
        write_zqp_orbital_general,
    )
    from tools._uhfk_to_mvmc.density_check import (
        compare_against_onebodyg_uhf_general,
    )
    from tools._uhfk_to_mvmc.partner_index import find_partner_rows

    try:
        orbitalidx_format = detect_orbitalidx_format(args.orbitalidx)
    except OrbitalidxGeneralFormatError as e:
        print(f"ERROR (orbitalidx): {e}", file=sys.stderr)
        return 2
    two_sz_raw = toml_param.get("2Sz")
    is_antiparallel_metadata = (
        two_sz_raw is not None
        and int(two_sz_raw) == 0
        and bool(np.all(column_spin >= 0))
        and set(np.unique(column_spin).tolist()) <= {0, 1}
        and len(np.unique(column_mu_group)) == 2
        and _column_spin_to_mu_group_is_bijective(column_spin, column_mu_group)
    )

    # v3.1 dispatch (spec §3.1): 6-case matrix over
    #   (is_antiparallel_metadata, orbitalidx_format, is_soc_mode).
    #
    # | is_ap_meta | orbitalidx_format | is_soc_mode | Path             |
    # |------------|-------------------|-------------|------------------|
    # | True       | antiparallel      | False       | v2.1 AntiParallel|
    # | True       | general           | False       | v3 forced-General|
    # | False      | general           | False       | v3 General (A/B) |
    # | False      | antiparallel      | False       | reject           |
    # | *          | general           | True        | v3.1 General-SOC |
    # | *          | antiparallel      | True        | reject (6-col)   |
    if is_soc_mode and orbitalidx_format == "antiparallel":
        print(
            "ERROR: enable_spin_orbital = true requires 6-column "
            "orbitalidx_general.def (StdFace: set "
            'model="FermionHubbardGC" in stan.in). The 3/4-column '
            "antiparallel format does not carry spin-off-diagonal "
            "classes.",
            file=sys.stderr,
        )
        return 2

    if is_antiparallel_metadata and orbitalidx_format == "antiparallel":
        # Fall through to the legacy v2.1 path below (unchanged). The
        # v1 spec §7 AntiParallel Sz-fixed sector constraint is enforced
        # here (it was pre-dispatch before v3 dispatch was introduced).
        if ne_per_group[0] != ne_per_group[1]:
            print(
                f"ERROR: N_up = {ne_per_group[0]} != N_down = "
                f"{ne_per_group[1]}; v1 spec section 7 requires "
                "AntiParallel Sz-fixed sector",
                file=sys.stderr,
            )
            return 2
    elif orbitalidx_format == "general":
        # v3 General path (also serves the forced-General branch when
        # is_antiparallel_metadata is True).
        # v3.1 spec §3.8: SOC path requires --transfer / --emit-trans
        # (the emitter runs after zqp_orbital_uhfk.dat is written; check
        # up-front so the caller catches missing args before any expensive
        # I/O and before the output file exists on disk).
        if is_soc_mode and (
            args.transfer is None or args.emit_trans is None
        ):
            print(
                "ERROR: enable_spin_orbital = true requires --transfer "
                "and --emit-trans (v3.1 spec §3.8: vmcdry.out's "
                "FermionHubbardGC generator drops Rashba s != t transfer "
                "entries; the bridge emits the SOC-preserving trans.def "
                "from H-wave's Transfer.dat).",
                file=sys.stderr,
            )
            return 2
        # v3.2 spec §1: SOC + SubShape > [1, 1, 1] requires the bridge to
        # emit an all-unique-classes orbitalidxgen.def in place of
        # StdFace's over-grouped one; fail up-front when the caller
        # forgot the flag so the error surfaces before any I/O.
        if is_soc_sublattice_mode and args.emit_orbitalidx is None:
            print(
                "ERROR: enable_spin_orbital = true + SubShape > [1, 1, 1] "
                "requires --emit-orbitalidx (v3.2 spec §1: StdFace's "
                "orbitalidxgen.def over-groups pair classes under SOC + "
                "sublattice folding; the bridge emits an all-unique-"
                "classes replacement).",
                file=sys.stderr,
            )
            return 2
        # SOC path is no longer experimental (v3.2): the trans.def sign
        # convention is empirically pinned via ComplexUHF verification at
        # 4.4e-8% precision on case_soc_rashba_2d_nosub. An earlier
        # attempt to derive it from H-wave's sc.py epsilon_k swap does
        # NOT apply (uhfk.py:1143-1144 does not perform that swap). See
        # tools/_uhfk_to_mvmc/trans_emit.py module docstring for the
        # empirical basis.
        _, site_R_int, norb = load_geometry_uhf(args.geometry)
        if norb != 1:
            print(
                f"ERROR: geometry has norb={norb} orbitals per cell; "
                "v3 general path requires single-orbital (norb_orig == 1)",
                file=sys.stderr,
            )
            return 2
        Ns = site_R_int.shape[0]
        if Ns != int(np.prod(cell_shape)):
            print(
                f"ERROR: geometry has {Ns} sites but CellShape implies "
                f"{int(np.prod(cell_shape))} (v3 single orbital)",
                file=sys.stderr,
            )
            return 2
        norb_orig = 1
        nd_expected = 2 * norb_orig * subvol
        nvol_folded_expected = int(np.prod(L_folded_arr))
        if eigenvector.shape != (
            nvol_folded_expected, nd_expected, nd_expected
        ):
            print(
                f"ERROR: eigen.npz eigenvector shape {eigenvector.shape} "
                f"does not match expected (nvol_folded="
                f"{nvol_folded_expected}, nd={nd_expected}, "
                f"nd={nd_expected}) derived from CellShape={cell_shape} "
                f"and SubShape={sub_shape}",
                file=sys.stderr,
            )
            return 2
        # v3 General path: derive ne_per_group from the actual
        # column_mu_group shape rather than from the input toml's
        # Ncond/2Sz splitting. Rationale: under H-wave Sz-free
        # (no 2Sz key), the SCF uses a single global chemical potential,
        # so column_mu_group has only one unique value. The AntiParallel
        # derivation ``[N_up, N_down] = [(Ncond+2Sz)/2, (Ncond-2Sz)/2]``
        # would fabricate a non-existent mu-group 1 and trip
        # ``step_occupation``'s "mu-group N has no eigenvector columns"
        # guard. For a single-group Sz-free case the semantically
        # correct target is ``[Ncond]`` (single group, fill lowest Ncond
        # eigenvalues regardless of spin).
        n_mu_groups = int(len(np.unique(column_mu_group)))
        if n_mu_groups == 1:
            ne_per_group_general = [int(toml_param["Ncond"])]
        else:
            ne_per_group_general = ne_per_group
        try:
            stepped_occupation, _ = step_occupation(
                occupation, eigenvalue, column_spin, column_mu_group,
                T_scf, ne_per_group_general,
                is_soc_mode=is_soc_mode,
            )
        except OccupationGuardError as e:
            print(f"ERROR (occupation guard): {e}", file=sys.stderr)
            return 2
        partner_rows, _ = find_partner_rows(wavevector_index, theta, L)
        try:
            validate_general_prerequisites(
                Ncond=int(toml_param["Ncond"]),
                stepped_occupation=stepped_occupation,
                column_spin=column_spin,
                partner_rows=partner_rows,
                wavevector_index=wavevector_index,
                is_soc_mode=is_soc_mode,
            )
        except ValueError as e:
            print(f"ERROR (general prerequisites): {e}", file=sys.stderr)
            return 2
        # v3.2 spec §1: under SOC + SubShape > [1, 1, 1] the bridge emits
        # its own all-unique-classes orbitalidxgen.def and uses it as the
        # effective mapping source (bypasses StdFace's class over-grouping
        # that would trip class-consistency in
        # aggregate_general_orbital_params). The E2E harness is responsible
        # for moving the emitted file over StdFace's version so mVMC also
        # consumes the same all-unique classes at InOrbitalGeneral read
        # time.
        if is_soc_sublattice_mode:
            from tools._uhfk_to_mvmc.orbitalidx_general_emitter import (
                emit_orbitalidx_all_unique,
            )
            emit_orbitalidx_all_unique(
                Ns, args.emit_orbitalidx, complex_type=1,
            )
            effective_orbitalidx_path = args.emit_orbitalidx
        else:
            effective_orbitalidx_path = args.orbitalidx
        try:
            info_general = parse_orbitalidx_general_def(
                effective_orbitalidx_path
            )
        except OrbitalidxGeneralFormatError as e:
            print(f"ERROR (orbitalidx_general): {e}", file=sys.stderr)
            return 2
        if info_general["complex_type"] != 1:
            print(
                f"ERROR: orbitalidx_general.def ComplexType = "
                f"{info_general['complex_type']}; bridge writes complex "
                "Fij values, ComplexType 1 required",
                file=sys.stderr,
            )
            return 2
        if info_general["nsite"] != Ns:
            print(
                f"ERROR: orbitalidx_general.def nsite = "
                f"{info_general['nsite']} != geometry nsite = {Ns}",
                file=sys.stderr,
            )
            return 2
        canonical, _self = compute_canonical_reps(
            partner_rows, wavevector_index
        )
        pair_list = build_pair_list(
            stepped_occupation, column_spin, canonical, partner_rows,
            is_soc_mode=is_soc_mode,
        )
        A = build_slater_orbitals(
            wavevector_index=wavevector_index,
            eigenvector=eigenvector,
            column_spin=column_spin,
            site_positions=site_R_int.astype(np.int64),
            cell_shape=cell_shape_arr,
            subshape=subshape_arr,
            theta=theta,
            pair_list=pair_list,
            is_soc_mode=is_soc_mode,
        )
        F_general = build_fij_general(A)
        params = aggregate_general_orbital_params(
            F_general, info_general["mapping"],
            info_general["n_orbital_idx"],
            epsilon_noise=args.epsilon_noise,
            complex_type=info_general["complex_type"],
            rng=np.random.default_rng(args.rng_seed),
        )
        if args.check_density:
            if args.onebodyg_uhf is None:
                print(
                    "ERROR: --check-density requires --onebodyg-uhf",
                    file=sys.stderr,
                )
                return 2
            G_all = np.conj(A) @ A.T
            # v3.2 (spec §1 note): H-wave's greenone.dat has a known bug
            # under SOC + SubShape > [1, 1, 1] (a downstream fold path
            # returns wrong values while ``green.npz['green_sublattice']``
            # remains correct — being fixed on a separate H-wave branch,
            # see memory/feedback_hwave_sublattice_green_testing.md).
            # For this case route the density check through the folded
            # green_sublattice instead of greenone.dat. The green.npz
            # path is derived from the --eigen path (H-wave writes both
            # to the same output/ directory).
            if is_soc_sublattice_mode:
                # v3.5 Phase C density gate for SOC + SubShape > [1, 1, 1].
                # The v3.4 dual-A workaround (build a reference A without
                # ``sub_offset`` for the density check while shipping a
                # distinct A with ``sub_offset`` to mVMC) is retired: the
                # v3.5 branch of ``compare_against_green_sublattice``
                # LIFTS ``green_sublattice`` to the physical basis via
                # ``gauge_lift`` and compares element-wise to the shipping
                # ``conj(A) @ A.T`` (spec §3 v3.5 design). This gauge-
                # invariant lift restores a strict full-element density
                # gate at 1e-10 directly on the shipping A/F. Codex Rev.2
                # concern (dual-A pattern hides shipping-A regressions)
                # is resolved.
                from tools._uhfk_to_mvmc.density_check import (
                    compare_against_green_sublattice,
                )
                green_npz_path = os.path.join(
                    os.path.dirname(args.eigen), "green.npz"
                )
                if not os.path.isfile(green_npz_path):
                    print(
                        f"ERROR: --check-density under SOC + SubShape > "
                        f"[1, 1, 1] requires green.npz at "
                        f"{green_npz_path!r} (derived from --eigen "
                        f"parent dir); H-wave writes it when "
                        f"has_sublattice is True.",
                        file=sys.stderr,
                    )
                    return 2
                try:
                    compare_against_green_sublattice(
                        G_all, green_npz_path,
                        site_positions=site_R_int.astype(np.int64),
                        cell_shape=cell_shape_arr,
                        subshape=subshape_arr,
                        Ns=Ns,
                        tol=1e-10,
                        is_soc_sublattice_mode=True,
                    )
                except DensityMismatchError as e:
                    print(
                        f"ERROR (density check, green_sublattice): {e}",
                        file=sys.stderr,
                    )
                    return 3
                print(
                    "density check OK (tol 1e-10; SOC+SubShape gauge-"
                    "lifted green_sublattice, v3.5)"
                )
            else:
                try:
                    compare_against_onebodyg_uhf_general(
                        G_all, args.onebodyg_uhf, tol=1e-10,
                        is_soc_mode=is_soc_mode,
                    )
                except DensityMismatchError as e:
                    print(f"ERROR (density check): {e}", file=sys.stderr)
                    return 3
                print("density check OK (tol 1e-10)")
        # Codex adversarial review (Rev.1, finding 3): under SOC, both
        # zqp_orbital_uhfk.dat and trans.def must land atomically as a
        # pair. A failure in emit_trans_def used to leave a stale zqp on
        # disk with the pre-existing trans.def next to it, misleading a
        # subsequent mVMC run. Both temps are committed via os.replace
        # only after write_zqp_orbital_general AND emit_trans_def both
        # succeed. Non-SOC path still goes through the same block; the
        # trans temp is skipped because is_soc_mode is False.
        #
        # Codex Rev.2 finding 2: two os.replace calls are not truly atomic
        # (a failure of the second commit leaves the first landed and
        # would silently pair a fresh zqp with the previous trans.def).
        # Preflight the SOC outputs so os.replace only fires against
        # already-validated destinations: reject same-path collisions
        # (would silently overwrite each other), reject directory targets
        # (os.replace can't atomically overwrite a directory with a
        # file), and reject unwritable parent directories. With these
        # guards the two os.replace calls are best-effort atomic; a
        # residual filesystem-level failure (e.g. mid-syscall power loss)
        # can still split the pair but has no in-process recovery path.
        if is_soc_mode:
            out_real = os.path.realpath(args.output)
            trans_real = os.path.realpath(args.emit_trans)
            if out_real == trans_real:
                print(
                    f"ERROR: --output and --emit-trans resolve to the "
                    f"same path ({out_real}); they must be distinct.",
                    file=sys.stderr,
                )
                return 2
            for label, path in (
                ("output", args.output),
                ("emit-trans", args.emit_trans),
            ):
                if os.path.isdir(path):
                    print(
                        f"ERROR: --{label} target {path!r} is an "
                        "existing directory; must be a file path.",
                        file=sys.stderr,
                    )
                    return 2
                parent = os.path.dirname(os.path.abspath(path)) or "."
                if not os.path.isdir(parent):
                    print(
                        f"ERROR: --{label} parent directory "
                        f"{parent!r} does not exist.",
                        file=sys.stderr,
                    )
                    return 2
                if not os.access(parent, os.W_OK):
                    print(
                        f"ERROR: --{label} parent directory "
                        f"{parent!r} is not writable.",
                        file=sys.stderr,
                    )
                    return 2
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        trans_out_path = None
        trans_tmp_path = None
        if is_soc_mode:
            trans_out_path = os.path.abspath(args.emit_trans)
            trans_out_dir = os.path.dirname(trans_out_path) or "."
            os.makedirs(trans_out_dir, exist_ok=True)
            trans_fd, trans_tmp_path = tempfile.mkstemp(
                prefix=".uhfk_to_mvmc.trans.", suffix=".tmp",
                dir=trans_out_dir,
            )
            os.close(trans_fd)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".uhfk_to_mvmc.", suffix=".tmp", dir=out_dir,
        )
        os.close(tmp_fd)

        def _cleanup_tmp_outputs():
            for p in (tmp_path, trans_tmp_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        try:
            write_zqp_orbital_general(tmp_path, params)
            if is_soc_mode:
                # v3.1 spec §3.8: emit mVMC trans.def from H-wave
                # Transfer.dat so SOC (Rashba s != t) entries survive
                # into mVMC's Hamiltonian. Runs only under is_soc_mode
                # because vmcdry.out already produces a correct
                # spin-diagonal trans.def for non-SOC v3 A/B fixtures.
                # v3.2: boundary_theta is threaded so APBC rows acquire
                # the wrap-phase sign flip at boundary crossings.
                from tools._uhfk_to_mvmc.trans_emit import (
                    emit_trans_def, TransEmitError,
                )
                try:
                    emit_trans_def(
                        args.transfer, cell_shape, trans_tmp_path,
                        boundary_theta=boundary_theta_tuple,
                    )
                except TransEmitError as e:
                    print(f"ERROR (trans_emit): {e}", file=sys.stderr)
                    _cleanup_tmp_outputs()
                    return 2
            # Both writes succeeded; commit atomically.
            os.replace(tmp_path, out_path)
            if is_soc_mode:
                os.replace(trans_tmp_path, trans_out_path)
        except Exception:
            _cleanup_tmp_outputs()
            raise
        print(
            f"wrote {args.output} "
            f"({info_general['n_orbital_idx']} General params)"
        )
        if is_soc_mode:
            print(
                f"wrote {args.emit_trans} "
                f"(SOC trans.def with Rashba entries preserved)"
            )
        return 0
    else:
        print(
            "ERROR: input is Sz-imbalanced or Sz-free (not "
            "AntiParallel-compatible), but orbitalidx.def is 3- or "
            "4-column (AntiParallel) format. Please regenerate with "
            "orbitalidx_general.def (6-column) via StdFace (e.g. "
            "'OrbitalGeneral 1' in stan.in).",
            file=sys.stderr,
        )
        return 2

    _, site_R_int, norb = load_geometry_uhf(args.geometry)
    if norb != 1:
        print(
            f"ERROR: geometry has norb={norb} orbitals per cell; "
            "v1 spec section 7 requires single-orbital (norb_orig == 1)",
            file=sys.stderr,
        )
        return 2
    norb_orig = 1
    Ns = site_R_int.shape[0]
    if Ns != int(np.prod(cell_shape)):
        print(
            f"ERROR: geometry has {Ns} sites but CellShape implies "
            f"{int(np.prod(cell_shape))} (v1 single orbital)",
            file=sys.stderr,
        )
        return 2

    # Codex finding 4 (v2): eigen.npz must match the folded BZ shape
    # derived from CellShape / SubShape. If a caller passes a stale
    # SubShape that disagrees with H-wave's actual fold, refuse before
    # building amplitudes.
    nd_expected = 2 * norb_orig * subvol
    nvol_folded_expected = int(np.prod(L_folded_arr))
    if eigenvector.shape != (nvol_folded_expected, nd_expected, nd_expected):
        print(
            f"ERROR: eigen.npz eigenvector shape {eigenvector.shape} does "
            f"not match expected (nvol_folded={nvol_folded_expected}, "
            f"nd={nd_expected}, nd={nd_expected}) derived from "
            f"CellShape={cell_shape} and SubShape={sub_shape}",
            file=sys.stderr,
        )
        return 2

    # Parse static def files before expensive occupation/amplitude work so
    # fail-fast guards on the static schema fire first.
    try:
        info = parse_orbitalidx_def(args.orbitalidx)
    except OrbitalidxFormatError as e:
        print(f"ERROR (orbitalidx): {e}", file=sys.stderr)
        return 2
    if info["complex_type"] != 1:
        print(
            f"ERROR: orbitalidx.def ComplexType = {info['complex_type']}; "
            "bridge writes complex Fij values, ComplexType 1 required",
            file=sys.stderr,
        )
        return 2
    if info["nsite"] != Ns:
        print(
            f"ERROR: orbitalidx.def nsite = {info['nsite']} != "
            f"geometry nsite = {Ns}",
            file=sys.stderr,
        )
        return 2
    if has_apbc and not info["has_sign_column"]:
        print(
            "ERROR: APBC in BoundaryCondition but orbitalidx.def has no "
            "4th sign column; rerun StdFace with phase0 = 180.0",
            file=sys.stderr,
        )
        return 2

    try:
        stepped_occupation, _ = step_occupation(
            occupation, eigenvalue, column_spin, column_mu_group,
            T_scf, ne_per_group,
        )
    except OccupationGuardError as e:
        print(f"ERROR (occupation guard): {e}", file=sys.stderr)
        return 2

    A_up, A_down = build_amplitudes(
        wavevector_index=wavevector_index,
        eigenvector=eigenvector,
        stepped_occupation=stepped_occupation,
        column_spin=column_spin,
        column_mu_group=column_mu_group,
        site_positions=site_R_int,
        norb_orig=norb_orig,
        theta=theta,
        L=L,  # L_folded (partner_rows lives on folded BZ)
        cell_shape=cell_shape_arr,
        subshape=subshape_arr,
    )
    F_phys = build_fij_phys(A_up, A_down)

    params = aggregate_orbital_params(
        F_phys, info["mapping"], info["n_orbital_idx"],
        epsilon_noise=args.epsilon_noise,
        complex_type=info["complex_type"],
        rng=np.random.default_rng(args.rng_seed),
    )

    # Density validation must run BEFORE writing the output file so a
    # failed check never leaves a mVMC-readable artifact on disk (Codex
    # adversarial review fix 2).
    if args.check_density:
        if args.onebodyg_uhf is None:
            print(
                "ERROR: --check-density requires --onebodyg-uhf "
                "(path to _UHF_cisajs.dat)",
                file=sys.stderr,
            )
            return 2
        G_up = density_from_amplitudes(A_up)
        G_down = density_from_amplitudes(A_down)
        try:
            compare_against_onebodyg_uhf(
                G_up, G_down, args.onebodyg_uhf, tol=1e-10
            )
        except DensityMismatchError as e:
            print(f"ERROR (density check): {e}", file=sys.stderr)
            return 3
        print("density check OK (tol 1e-10)")

    # Atomic write: emit to a sibling temp file and rename only after the
    # validation above passes. If the rename fails (cross-device, etc.),
    # clean up the temp before propagating.
    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".uhfk_to_mvmc.", suffix=".tmp", dir=out_dir
    )
    os.close(tmp_fd)
    try:
        write_zqp_orbital(tmp_path, params)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"wrote {args.output} ({info['n_orbital_idx']} params)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
