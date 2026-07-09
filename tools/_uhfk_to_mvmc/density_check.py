"""Density matrix consistency check against H-wave's _UHF_cisajs.dat.

Spec section 5.1. G_{ij} = <c^dag_i c_j> in H-wave's convention is
   G^sigma_{ij} = sum_alpha (A^sigma)^*_{i, alpha} A^sigma_{j, alpha}
              = (A^sigma.conj() @ A^sigma.T)_{ij}
"""
from __future__ import annotations

import numpy as np


class DensityMismatchError(RuntimeError):
    """Raised when bridge-built G^sigma_{ij} disagrees with H-wave's
    _UHF_cisajs.dat beyond the given tolerance."""


def density_from_amplitudes(A):
    """G_{ij} = sum_alpha conj(A_{i, alpha}) A_{j, alpha}."""
    A = np.asarray(A, dtype=np.complex128)
    return np.conj(A) @ A.T


def parse_uhf_cisajs_dat(path):
    """Parse a _UHF_cisajs.dat text file. Returns list of
    (i, s, j, t, complex value)."""
    out = []
    with open(path) as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            i, s, j, t = (int(toks[0]), int(toks[1]),
                          int(toks[2]), int(toks[3]))
            v = float(toks[4]) + 1j * float(toks[5])
            out.append((i, s, j, t, v))
    return out


def compare_against_onebodyg_uhf(
    G_up, G_down, onebodyg_uhf_path, tol=1e-10
):
    """Compare bridge-built G^up, G^down against H-wave's _UHF_cisajs.dat.

    Spec section 5.1: line by line iterate (i, s, j, t) entries; bridge
    side returns G^s_{ij} (s == t required, otherwise 0 for Sz-fixed).
    """
    entries = parse_uhf_cisajs_dat(onebodyg_uhf_path)
    diffs = []
    for (i, s, j, t, v_uhf) in entries:
        if s != t:
            v_bridge = 0.0 + 0.0j
        elif s == 0:
            v_bridge = complex(G_up[i, j])
        elif s == 1:
            v_bridge = complex(G_down[i, j])
        else:
            raise DensityMismatchError(
                f"unexpected spin index s={s} in {onebodyg_uhf_path}"
            )
        if abs(v_bridge - v_uhf) > tol:
            diffs.append((i, s, j, t, v_bridge, v_uhf))
    if diffs:
        head = diffs[:3]
        raise DensityMismatchError(
            f"{len(diffs)} (i, s, j, t) entries differ beyond tol={tol}; "
            f"first 3: {head}"
        )


def compare_against_onebodyg_uhf_general(
    G_all: np.ndarray, onebodyg_uhf_path: str, tol: float = 1e-10,
    is_soc_mode: bool = False,
) -> None:
    """Compare bridge-built 2Ns × 2Ns G against H-wave's greenone.dat
    for the General path (spec §4.5, §3.7).

    G_all[iσ, jσ'] = <c^†_{i,σ} c_{j,σ'}> in the physical basis with
    the mVMC spin-block index ``all_i = i + σ * Ns`` (site-major,
    spin-minor). greenone.dat lines are ``i s j t re im`` and are
    compared element-wise to ``G_all[i + s*Ns, j + t*Ns]``.

    Behavior by mode (spec §3.7):

    ``is_soc_mode=False`` (v3 Sz-diagonal path)
        The bridge produces a spin-block-diagonal ``G_all`` because
        amplitudes are Sz-fixed. Reference greenone.dat rows with
        s == t compare element-wise. Rows with s != t are also
        compared element-wise; because ``G_all`` is zero on
        off-diagonal blocks, any non-zero s != t reference value
        raises ``DensityMismatchError``. This element-wise comparison
        is the mixed-block scope-violation guard: it catches the case
        where a caller feeds SOC-tainted reference data into the v3
        path.

    ``is_soc_mode=True`` (v3.1 SOC path)
        Both G_all and the reference greenone.dat carry physically
        non-zero s != t entries. Every row is compared under the same
        `tol` regardless of (s, t); the scope-violation framing does
        not apply because the SOC bridge deliberately populates the
        off-diagonal spin blocks.

    Both modes share the same comparison kernel: uniform element-wise
    match under `tol`. The flag documents intent and gates any future
    mode-specific diagnostics without changing v3 behavior.
    """
    G_all = np.asarray(G_all, dtype=np.complex128)
    two_ns = G_all.shape[0]
    assert two_ns % 2 == 0, "G_all must have even row count"
    Ns = two_ns // 2

    entries = parse_uhf_cisajs_dat(onebodyg_uhf_path)
    diffs = []
    for (i, s, j, t, v_uhf) in entries:
        all_i = i + s * Ns
        all_j = j + t * Ns
        v_bridge = complex(G_all[all_i, all_j])
        if abs(v_bridge - v_uhf) > tol:
            diffs.append((i, s, j, t, v_bridge, v_uhf))
    if diffs:
        head = diffs[:3]
        mode_label = "SOC" if is_soc_mode else "v3 Sz-diagonal"
        raise DensityMismatchError(
            f"{len(diffs)} (i, s, j, t) entries differ beyond tol={tol} "
            f"[{mode_label} mode]; first 3: {head}"
        )
