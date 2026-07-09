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
