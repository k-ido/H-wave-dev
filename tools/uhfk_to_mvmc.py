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
    if ne_per_group[0] != ne_per_group[1]:
        print(
            f"ERROR: N_up = {ne_per_group[0]} != N_down = "
            f"{ne_per_group[1]}; v1 spec section 7 requires AntiParallel "
            "Sz-fixed sector",
            file=sys.stderr,
        )
        return 2

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

    if np.any(column_spin < 0):
        print(
            "ERROR: occupation.npz has column_spin = -1 (Sz-free / mixed "
            "block); v1 spec section 7 requires Sz-fixed",
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
