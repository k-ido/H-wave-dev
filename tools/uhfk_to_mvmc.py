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
    args = parser.parse_args(argv)

    toml_param = load_input_toml(args.input)
    cell_shape = list(toml_param["CellShape"])
    sub_shape = list(toml_param.get("SubShape", cell_shape))
    boundary = list(toml_param.get(
        "BoundaryCondition", ["periodic", "periodic", "periodic"]
    ))
    enable_spin_orbital = bool(toml_param.get("enable_spin_orbital", False))
    if enable_spin_orbital:
        print("ERROR: enable_spin_orbital is not supported in v1",
              file=sys.stderr)
        return 2

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

    theta = np.array([
        np.pi if b.lower() in {"ap", "antiperiodic"} else 0.0
        for b in boundary
    ], dtype=np.float64)
    has_apbc = bool(np.any(theta > 0))
    # L passed to build_amplitudes / partner_index is the folded lattice
    # size (partner rows are computed on the folded BZ). CellShape is
    # preserved separately for the unfold path (spec §3.3).
    L = L_folded_arr

    eigen = np.load(args.eigen, allow_pickle=False)
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
            )
        except ValueError as e:
            print(f"ERROR (general prerequisites): {e}", file=sys.stderr)
            return 2
        try:
            info_general = parse_orbitalidx_general_def(args.orbitalidx)
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
            try:
                compare_against_onebodyg_uhf_general(
                    G_all, args.onebodyg_uhf, tol=1e-10,
                )
            except DensityMismatchError as e:
                print(f"ERROR (density check): {e}", file=sys.stderr)
                return 3
            print("density check OK (tol 1e-10)")
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".uhfk_to_mvmc.", suffix=".tmp", dir=out_dir,
        )
        os.close(tmp_fd)
        try:
            write_zqp_orbital_general(tmp_path, params)
            os.replace(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        print(
            f"wrote {args.output} "
            f"({info_general['n_orbital_idx']} General params)"
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
