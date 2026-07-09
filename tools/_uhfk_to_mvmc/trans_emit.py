"""H-wave Transfer.dat -> mVMC trans.def emitter for SOC (v3.1 spec §3.8).

Reads H-wave's Wannier90-like ``Transfer.dat`` (SOI-packed ``iWan/jWan``
indices ``2 * a_phys + spin + 1``) and emits mVMC's real-space
``trans.def`` with ``(i, s, j, t)`` rows for the full physical lattice,
preserving Rashba ``s != t`` off-diagonal spin entries that
``vmcdry.out``'s ``FermionHubbardGC`` generator drops silently.

Behaviour in one line: for each Transfer.dat entry we emit
``s == t: -val_hwave`` and ``s != t: +val_hwave`` (see
``DEFAULT_SIGN_DIAG`` / ``DEFAULT_SIGN_OFFDIAG``).

Sign convention (empirically pinned)
------------------------------------
The mixed convention ``(sign_diag = -1, sign_offdiag = +1)`` is
empirically pinned against ComplexUHF on
``case_soc_rashba_2d_nosub`` (SubShape = [1, 1, 1] + SOC): ComplexUHF
started from a random Green function converges on the emitted
``trans.def`` to H-wave's ``<H>`` at 4.4e-8% precision, i.e. the sign
convention is correct to machine round-off in the H-Hamiltonian sense
for that fixture. Any residual delta on the mVMC side (currently
~0.20% at NVMCSample=10000) is VMC statistical noise, not a sign bug.

An earlier version of this docstring claimed the sign convention was
derivable from ``src/hwave/sc.py:250-255`` (an ``epsilon_k[orb2, orb1]``
swap). That swap is real in ``sc.py`` (a separate solver), but
``uhfk.py:1143-1144`` does NOT perform the same swap — it stores
``tab_r[(*irvec, *orbvec)] = v`` directly, no index reversal. So the
"derived from H-wave's internal swap" story does not apply to the
UHFk path this bridge consumes. The sign convention that ships here
is the one that empirically produces the ComplexUHF match on the
non-sublattice SOC fixture; the derivation is deferred until an
independent audit of the full UHFk Fourier + folded-orbital
convention chain nails down which composition of signs is at play.

Verification scope
------------------
- ``case_soc_rashba_2d_nosub`` (SubShape [1, 1, 1] + SOC): the emitted
  ``trans.def`` matches ComplexUHF's SCF at 4.4e-8%. Verified.
- ``case_soc_rashba_2d_nosub_apbc`` (SubShape [1, 1, 1] + SOC + APBC):
  end-to-end match at VMC precision. Verified.
- ``case_soc_rashba_2d_sub`` (SubShape [2, 2, 1] + SOC): the emitted
  ``trans.def`` STILL matches ComplexUHF at ~0.15% (ComplexUHF from
  random init converges to E=-25.14 vs H-wave -25.10 on the same
  trans.def), so the Hamiltonian coefficients emitted by this module
  are correct for that fixture too. The reason the E2E energy compare
  is skipped for this case is a separate bug on the
  ``build_slater_orbitals`` (WF construction) path under SOC +
  SubShape > [1, 1, 1] — the emitted ``zqp_orbital_uhfk.dat`` encodes
  a Slater whose diagonal density matches H-wave's ``green_sublattice``
  but whose off-diagonal cross-spin entries are wrong, so mVMC's
  ``<H>`` on that WF disagrees with H-wave's ``Energy_Total`` by ~17.
  See ``docs/superpowers/specs/2026-07-01-uhfk-mvmc-pairproduct-general-v31-design.md``
  §3.8 and the SOC + SubShape > 1 note in
  ``tests/validation/uhfk_mvmc_pairproduct/run.sh``.
"""
from __future__ import annotations

import numpy as np

from hwave.solver._apbc_phase import inverse_gauge_phase

# Sign multipliers applied per emitted trans.def row. Empirically pinned:
# ComplexUHF SCF on the emitted trans.def matches H-wave's <H> to
# 4.4e-8% on case_soc_rashba_2d_nosub (SubShape [1, 1, 1] + SOC) and to
# ~0.15% on case_soc_rashba_2d_sub (SubShape [2, 2, 1] + SOC) at
# random-init SCF from the emitted Hamiltonian. See the module docstring
# for verification scope and the aborted "derived from sc.py swap" story.
DEFAULT_SIGN_DIAG = -1.0
DEFAULT_SIGN_OFFDIAG = +1.0


class TransEmitError(ValueError):
    """Raised on malformed Transfer.dat or trans.def emission failure."""


def _unpack_soi(iWan: int) -> tuple[int, int]:
    """Inverse of ``emit_rashba_transfer._pack_index``: ``iWan = 2 * a + s + 1``.

    Returns ``(a_phys, spin)``. Under v3.1 scope (``norb_orig = 1``),
    ``a_phys`` is always ``0`` and ``spin`` is ``{0, 1}``.
    """
    if iWan < 1:
        raise TransEmitError(
            f"invalid 1-based orbital index iWan={iWan} (must be >= 1)"
        )
    idx0 = iWan - 1
    return (idx0 // 2, idx0 % 2)


def parse_hwave_transfer(path):
    """Parse H-wave ``Transfer.dat`` (Wannier90-like format).

    File layout (see ``src/hwave/qlmsio/wan90.py::read_w90``)::

        <header line>                               # skipped
        <num_wann>                                  # skipped (unused)
        <nr>                                        # number of entries
        <ndegen block: nr ints, 15 per line>        # applied as divisor
        <data rows: rx ry rz iWan jWan re im>

    Applies the R-point degeneracy division from the Wannier90 convention
    ``H(k) = sum_R exp(i k.R) H(R) / ndegen(R)``. Mirrors the semantics of
    ``hwave.qlmsio.wan90.read_w90``: if every ndegen entry is ``1`` the
    division is a no-op (sparse H-wave format); otherwise each entry's
    coefficient is divided by ``ndegen[i]``, where ``i`` is the position of
    the entry's R-vector in file order (dense listing required for the
    non-unit case).

    Parameters
    ----------
    path : str
        Path to ``Transfer.dat``.

    Returns
    -------
    entries : list of tuples
        Each entry is ``(rx, ry, rz, iWan, jWan, val_complex)``. ``rx``,
        ``ry``, ``rz`` are signed real-space displacement components,
        ``iWan``/``jWan`` are 1-based SOI-packed orbital indices, and
        ``val_complex`` is ``(re + 1j * im) / ndegen``.

    Raises
    ------
    TransEmitError
        On file layout errors, truncated files, ndegen count mismatch, or
        a non-unit ndegen block paired with a sparse file listing (fewer
        distinct R-vectors than ``nr``).
    """
    try:
        with open(path) as fp:
            lines = fp.read().splitlines()
    except OSError as e:
        raise TransEmitError(f"cannot open Transfer.dat: {path}: {e}") from e

    if len(lines) < 3:
        raise TransEmitError(
            f"{path}: too short for Wannier90-like header "
            f"(got {len(lines)} lines, need >= 3)"
        )

    # Skip header (line 0), num_wann (line 1). Read nr (line 2).
    try:
        nr = int(lines[2].strip())
    except ValueError as e:
        raise TransEmitError(
            f"{path}: line 3 is not an integer nr (got {lines[2]!r})"
        ) from e
    if nr < 0:
        raise TransEmitError(f"{path}: negative nr = {nr}")

    # Parse ndegen block: nr integers, 15 per line -> ceil(nr / 15) lines.
    ndegen_rows = (nr + 14) // 15
    data_start = 3 + ndegen_rows
    if len(lines) < data_start:
        raise TransEmitError(
            f"{path}: truncated ndegen block (need {ndegen_rows} rows for "
            f"nr={nr}, only {len(lines) - 3} available)"
        )
    ndegen = []
    for i, line in enumerate(lines[3:data_start], start=4):
        try:
            ndegen.extend(int(x) for x in line.split())
        except ValueError as e:
            raise TransEmitError(
                f"{path}:{i}: cannot parse ndegen row: {line!r}"
            ) from e
    if len(ndegen) != nr:
        raise TransEmitError(
            f"{path}: ndegen count mismatch (declared nr={nr}, found "
            f"{len(ndegen)})"
        )
    # When every degeneracy is one, division is a no-op and the sparse
    # H-wave listing is legal (nr may be a placeholder). Only non-unit
    # ndegen requires the positional R -> ndegen[i] mapping.
    all_unit = all(d == 1 for d in ndegen)
    deg_of_r = {}  # irvec -> ndegen, keyed by first appearance in file.
    seen = set()  # (rx, ry, rz, iWan, jWan) tuples already emitted.

    entries = []
    for line_num, line in enumerate(lines[data_start:], start=data_start + 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            continue
        toks = stripped.split()
        if len(toks) < 7:
            raise TransEmitError(
                f"{path}:{line_num}: expected 7 columns "
                f"(rx ry rz iWan jWan re im), got {len(toks)}: {stripped!r}"
            )
        try:
            rx, ry, rz = int(toks[0]), int(toks[1]), int(toks[2])
            iWan, jWan = int(toks[3]), int(toks[4])
            re, im = float(toks[5]), float(toks[6])
        except ValueError as e:
            raise TransEmitError(
                f"{path}:{line_num}: cannot parse row: {stripped!r}"
            ) from e
        # Mirror hwave.qlmsio.wan90.read_w90's duplicate-entry rejection.
        # Two rows with the same (R, iWan, jWan) key would be silently
        # summed into trans.def by emit_trans_def below, double-counting
        # the hopping. H-wave's reader treats this as fatal; we match.
        key = (rx, ry, rz, iWan, jWan)
        if key in seen:
            raise TransEmitError(
                f"{path}:{line_num}: duplicate Transfer.dat entry "
                f"(rx, ry, rz, iWan, jWan) = {key}; H-wave's read_w90 "
                "rejects duplicate hopping entries -- matching that "
                "contract here."
            )
        seen.add(key)
        # Mirror hwave.qlmsio.wan90.read_w90 ndegen lookup.
        if all_unit:
            deg = 1
        else:
            irvec = (rx, ry, rz)
            if irvec not in deg_of_r:
                idx_r = len(deg_of_r)
                if idx_r >= len(ndegen):
                    raise TransEmitError(
                        f"{path}:{line_num}: more distinct R-points than "
                        f"declared (nr={nr})"
                    )
                deg_of_r[irvec] = ndegen[idx_r]
            deg = deg_of_r[irvec]
        entries.append((rx, ry, rz, iWan, jWan, complex(re, im) / deg))

    # Reject non-unit + sparse listing (fewer distinct Rs than declared),
    # matching hwave.qlmsio.wan90.read_w90.
    if not all_unit and len(deg_of_r) != nr:
        raise TransEmitError(
            f"{path}: non-unit ndegen requires a dense file listing all "
            f"{nr} R-points (found {len(deg_of_r)})"
        )
    return entries


def emit_trans_def(
    transfer_path, cell_shape, out_path,
    sign_diag=DEFAULT_SIGN_DIAG, sign_offdiag=DEFAULT_SIGN_OFFDIAG,
    boundary_theta=None,
):
    """Emit mVMC ``trans.def`` from H-wave ``Transfer.dat``.

    For each ``(R = (rx, ry, rz), iWan, jWan, val)`` entry in
    ``Transfer.dat`` and each source site ``i_src = (ix, iy, iz)`` in
    the physical lattice, emits one row::

        i_src_flat  s_src  j_tgt_flat  s_tgt  sign*re  sign*im

    where ``j_tgt = ((ix + rx) mod Lx, (iy + ry) mod Ly, (iz + rz) mod Lz)``
    (PBC unfold), ``s_src, s_tgt = unpack_soi(iWan, jWan)``, and the
    flat index is site-major (``i = ix + Lx * (iy + Ly * iz)``, matching
    H-wave's ``Geometry.dat`` layout).

    Parameters
    ----------
    transfer_path : str
        H-wave ``Transfer.dat`` path.
    cell_shape : list-like of length 3
        Physical lattice shape ``[Lx, Ly, Lz]``.
    out_path : str
        Output ``trans.def`` path.
    sign_diag : float, optional
        Multiplier applied to spin-diagonal (``s_src == s_tgt``) entries.
        Defaults to ``DEFAULT_SIGN_DIAG`` (``-1.0``); this converts
        H-wave's Hamiltonian coefficient to mVMC's ``H = -sum trans``
        convention and is byte-verified against vmcdry.out's
        spin-diagonal trans.def output. Change only for controlled unit
        tests.
    sign_offdiag : float, optional
        Multiplier applied to spin-off-diagonal (``s_src != s_tgt``,
        Rashba) entries. Defaults to ``DEFAULT_SIGN_OFFDIAG`` (``+1.0``);
        this value is **empirically pinned** via ComplexUHF verification
        at 4.4e-8% precision on ``case_soc_rashba_2d_nosub``. An earlier
        attempt to derive it from H-wave's ``sc.py`` ``epsilon_k[orb2,
        orb1]`` index swap does NOT apply: ``uhfk.py`` does not perform
        that swap (see uhfk.py:1143-1144). Re-verify against the E2E
        harness after any mVMC or H-wave version bump. See the module
        docstring for the empirical basis.
    boundary_theta : array-like of length 3 or None, optional
        Twist ``(theta_x, theta_y, theta_z)`` in radians. ``None``
        (default) means all-PBC and no boundary phase is applied.
        Non-zero components mean twisted / antiperiodic BC in the
        corresponding direction. For a bond whose displacement takes
        the target site out of the primary cell in direction ``d``,
        the emitted row acquires the physical wrap phase
        ``exp(i * theta_d * wraps_d)`` (``= (-1)^wraps_d`` for
        ``theta_d = pi`` / APBC). Non-wrapping bonds are unchanged.
        This mirrors the sign flip StdFace bakes into vmcdry.out's
        ``trans.def`` under ``phase0 = 180`` so mVMC's ``H = -sum trans``
        recovers the physical Hamiltonian on the periodic-site frame.

    Raises
    ------
    TransEmitError
        On malformed input or unsupported SOI unpacking (e.g. a physical
        orbital index != 0 under v3.1's single-orbital scope).
    """
    entries = parse_hwave_transfer(transfer_path)
    if len(cell_shape) != 3:
        raise TransEmitError(
            f"cell_shape must have length 3, got {list(cell_shape)}"
        )
    Lx, Ly, Lz = (int(c) for c in cell_shape)
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise TransEmitError(
            f"cell_shape components must be positive: {[Lx, Ly, Lz]}"
        )

    # v3.2 spec §3.8 (SOC + APBC): compose per-row wrap phase for
    # boundary-crossing bonds. ``inverse_gauge_phase(r_j_wrapped,
    # r_j_unwrapped, theta, L)`` evaluates to
    # ``exp(i * theta . (r_j_wrapped - r_j_unwrapped) / L)`` which
    # equals ``(-1)^wraps_d`` in each AP direction (theta_d = pi and
    # ``r_j_unwrapped - r_j_wrapped`` a multiple of L_d in that
    # direction, with the wrap count as the multiplier). Non-wrapping
    # bonds land on ``exp(0) = 1`` and are unchanged. The formula
    # generalizes to arbitrary twist without special-casing APBC.
    if boundary_theta is not None and np.any(
        np.abs(np.asarray(boundary_theta, dtype=np.float64)) > 1e-12
    ):
        theta_arr = np.asarray(boundary_theta, dtype=np.float64)
        L_arr = np.asarray(cell_shape, dtype=np.float64)
        apply_gauge = True
    else:
        theta_arr = None
        L_arr = None
        apply_gauge = False

    def site_index(ix, iy, iz):
        return ix + Lx * (iy + Ly * iz)

    rows = []
    for rx, ry, rz, iWan, jWan, val in entries:
        a_src, s_src = _unpack_soi(iWan)
        a_tgt, s_tgt = _unpack_soi(jWan)
        # v3.1 spec §1 scope: single physical orbital per site.
        if a_src != 0 or a_tgt != 0:
            raise TransEmitError(
                f"trans_emit assumes norb_orig == 1 (v3.1); got "
                f"iWan={iWan} -> a_src={a_src}, jWan={jWan} -> "
                f"a_tgt={a_tgt}"
            )
        sign = sign_diag if s_src == s_tgt else sign_offdiag
        for iz in range(Lz):
            for iy in range(Ly):
                for ix in range(Lx):
                    i_site = site_index(ix, iy, iz)
                    jx = (ix + rx) % Lx
                    jy = (iy + ry) % Ly
                    jz = (iz + rz) % Lz
                    j_site = site_index(jx, jy, jz)
                    v_val = val
                    if apply_gauge:
                        r_j_wrapped = np.array(
                            [jx, jy, jz], dtype=np.float64
                        )
                        r_j_unwrapped = np.array(
                            [ix + rx, iy + ry, iz + rz], dtype=np.float64
                        )
                        wrap_phase = inverse_gauge_phase(
                            r_j_wrapped, r_j_unwrapped, theta_arr, L_arr
                        )
                        v_val = v_val * wrap_phase
                    v = sign * v_val
                    rows.append(
                        (i_site, s_src, j_site, s_tgt, v.real, v.imag)
                    )

    # Mirror vmcdry.out's trans.def header layout so mVMC's parser
    # accepts the file interchangeably with the vmcdry-generated one.
    with open(out_path, "w") as fw:
        fw.write("======================== \n")
        fw.write("NTransfer      {}  \n".format(len(rows)))
        fw.write("======================== \n")
        fw.write("========i_j_s_tijs====== \n")
        fw.write("======================== \n")
        for (i, s, j, t, re, im) in rows:
            fw.write(
                "{:5d}{:6d}{:6d}{:6d}{:26.15f}{:26.15f}\n".format(
                    i, s, j, t, re, im
                )
            )
