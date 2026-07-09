"""v3 InOrbitalGeneral F builder (Sz-fixed 2Sz≠0 A + Sz-free non-mixed B).

Spec: docs/superpowers/specs/2026-07-01-uhfk-mvmc-pairproduct-general-v3-design.md
      §3.1-§3.5, §4.1
"""
from __future__ import annotations

import numpy as np


def compute_canonical_reps(
    partner_rows: np.ndarray,
    wavevector_index: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Pick one canonical row per unordered {k, partner(k)} pair.

    Rules (spec §3.3):
      - self-pair k (partner_rows[k] == k): k is canonical.
      - non-self-pair {k, partner(k)}: canonical = row whose
        wavevector_index tuple is lexicographically smaller
        (fallback tie-break: smaller row index).

    Returns
    -------
    canonical_rows : list[int]
        Sorted list of canonical row indices; length =
        (#self-pairs) + (#non-self-pairs / 2).
    self_pair_rows : list[int]
        Subset of canonical_rows whose partner is themselves.
    """
    partner_rows = np.asarray(partner_rows, dtype=np.int64)
    wavevector_index = np.asarray(wavevector_index, dtype=np.int64)
    nvol = partner_rows.shape[0]

    canonical = []
    self_pairs = []
    seen = set()
    for k_row in range(nvol):
        if k_row in seen:
            continue
        partner = int(partner_rows[k_row])
        if partner == k_row:
            canonical.append(k_row)
            self_pairs.append(k_row)
            seen.add(k_row)
            continue
        # Non-self: pick lex-smaller wavevector_index tuple.
        wv_k = tuple(int(v) for v in wavevector_index[k_row])
        wv_p = tuple(int(v) for v in wavevector_index[partner])
        if wv_k < wv_p:
            chosen = k_row
        elif wv_p < wv_k:
            chosen = partner
        else:  # exact tie → smaller row index
            chosen = min(k_row, partner)
        canonical.append(chosen)
        seen.add(k_row)
        seen.add(partner)
    canonical.sort()
    return canonical, self_pairs


def _occ_by_spin(stepped_occupation, column_spin, spin_value):
    """Return dict[k_row] = ordered list of columns occupied at that k
    with the specified spin (0 = up, 1 = down)."""
    nvol, nd = stepped_occupation.shape
    columns = [c for c in range(nd) if int(column_spin[c]) == spin_value]
    out = {}
    for k_row in range(nvol):
        out[k_row] = [c for c in columns if stepped_occupation[k_row, c] >= 0.5]
    return out


def validate_general_prerequisites(
    Ncond: int,
    stepped_occupation: np.ndarray,
    column_spin: np.ndarray,
    partner_rows: np.ndarray,
    wavevector_index: np.ndarray,
) -> None:
    """Fail-fast validation before General F construction.

    Raises ValueError with a diagnostic message on:
      - Any column_spin value not in {0, 1} (mixed block; v3 rejects → v3.1)
      - Ncond odd (mVMC PairProduct requires even Ne)
      - sum(stepped_occupation) != Ncond
      - Non-self canonical (k, partner): n_excess_up_k != n_excess_up_p
        OR n_excess_down_k != n_excess_down_p (same-spin excess imbalance)
      - Self-pair canonical k: n_excess_up_k odd OR n_excess_down_k odd
    """
    column_spin = np.asarray(column_spin, dtype=np.int64)
    stepped_occupation = np.asarray(stepped_occupation, dtype=np.float64)

    if np.any((column_spin != 0) & (column_spin != 1)):
        offending = np.unique(column_spin[(column_spin != 0) & (column_spin != 1)])
        raise ValueError(
            f"v3 scope: column_spin contains mixed block value(s) {offending.tolist()}; "
            "mixed block / SOC support is deferred to v3.1 (see spec §1, §7)"
        )
    if Ncond % 2 != 0:
        raise ValueError(
            f"Ncond={Ncond} is odd; mVMC PairProduct requires even Ne"
        )
    total_occ = float(np.sum(stepped_occupation))
    if abs(total_occ - Ncond) > 0.5:
        raise ValueError(
            f"sum(stepped_occupation)={total_occ} != Ncond={Ncond}"
        )

    occ_up = _occ_by_spin(stepped_occupation, column_spin, 0)
    occ_dn = _occ_by_spin(stepped_occupation, column_spin, 1)
    partner_rows = np.asarray(partner_rows, dtype=np.int64)
    canonical, self_pairs = compute_canonical_reps(partner_rows, wavevector_index)

    for k in canonical:
        partner = int(partner_rows[k])
        nu_k, nd_k = len(occ_up[k]), len(occ_dn[k])
        if partner == k:
            n_cross = min(nu_k, nd_k)
            excess_up = nu_k - n_cross
            excess_dn = nd_k - n_cross
            if excess_up % 2 != 0:
                raise ValueError(
                    f"self-pair canonical row {k}: excess up-spin count "
                    f"{excess_up} is odd (must be even for same-spin pairing)"
                )
            if excess_dn % 2 != 0:
                raise ValueError(
                    f"self-pair canonical row {k}: excess down-spin count "
                    f"{excess_dn} is odd (must be even)"
                )
        else:
            nu_p, nd_p = len(occ_up[partner]), len(occ_dn[partner])
            n_cross_kd = min(nu_k, nd_p)
            n_cross_dk = min(nu_p, nd_k)
            excess_up_k = nu_k - n_cross_kd
            excess_up_p = nu_p - n_cross_dk
            excess_dn_k = nd_k - n_cross_dk
            excess_dn_p = nd_p - n_cross_kd
            if excess_up_k != excess_up_p:
                raise ValueError(
                    f"canonical block (k={k}, partner={partner}): "
                    f"n_excess_up_k={excess_up_k} != n_excess_up_p={excess_up_p} "
                    f"(#up@k={nu_k}, #down@partner={nd_p}, "
                    f"#up@partner={nu_p}, #down@k={nd_k}); "
                    "spin-cross + same-spin excess pair-closure violated"
                )
            if excess_dn_k != excess_dn_p:
                raise ValueError(
                    f"canonical block (k={k}, partner={partner}): "
                    f"n_excess_down_k={excess_dn_k} != n_excess_down_p={excess_dn_p} "
                    "pair-closure violated"
                )


def build_pair_list(
    stepped_occupation: np.ndarray,
    column_spin: np.ndarray,
    canonical_rows: list[int],
    partner_rows: np.ndarray,
) -> list[dict]:
    """Emit the ordered pair list for General F construction (spec §3.3).

    Each pair is emitted from a canonical (k, partner(k)) block. For a
    non-self canonical k:
      1. (up@k, down@partner) — cross_kd pairs
      2. (up@partner, down@k) — cross_dk pairs
      3. (up@k, up@partner) — same-spin up excess
      4. (down@k, down@partner) — same-spin down excess

    For a self-pair canonical k:
      1. (up@k, down@k) — cross
      2. (up@k, up@k) — same-spin up excess (paired 2-at-a-time)
      3. (down@k, down@k) — same-spin down excess (paired 2-at-a-time)

    Each entry is a dict:
        {"alpha": (k_row, col, spin_label),
         "beta":  (k_row, col, spin_label)}
    where spin_label is "up" or "down".
    """
    occ_up = _occ_by_spin(stepped_occupation, column_spin, 0)
    occ_dn = _occ_by_spin(stepped_occupation, column_spin, 1)
    partner_rows = np.asarray(partner_rows, dtype=np.int64)
    pairs = []
    for k in canonical_rows:
        partner = int(partner_rows[k])
        if partner == k:
            u = occ_up[k]
            d = occ_dn[k]
            n_cross = min(len(u), len(d))
            for i in range(n_cross):
                pairs.append({
                    "alpha": (k, u[i], "up"),
                    "beta":  (k, d[i], "down"),
                })
            excess_u = u[n_cross:]
            excess_d = d[n_cross:]
            for i in range(len(excess_u) // 2):
                pairs.append({
                    "alpha": (k, excess_u[2 * i], "up"),
                    "beta":  (k, excess_u[2 * i + 1], "up"),
                })
            for i in range(len(excess_d) // 2):
                pairs.append({
                    "alpha": (k, excess_d[2 * i], "down"),
                    "beta":  (k, excess_d[2 * i + 1], "down"),
                })
        else:
            uk, dk = occ_up[k], occ_dn[k]
            up_, dp = occ_up[partner], occ_dn[partner]
            n_cross_kd = min(len(uk), len(dp))
            n_cross_dk = min(len(up_), len(dk))
            for i in range(n_cross_kd):
                pairs.append({
                    "alpha": (k, uk[i], "up"),
                    "beta":  (partner, dp[i], "down"),
                })
            for i in range(n_cross_dk):
                pairs.append({
                    "alpha": (partner, up_[i], "up"),
                    "beta":  (k, dk[i], "down"),
                })
            excess_uk = uk[n_cross_kd:]
            excess_up_ = up_[n_cross_dk:]
            excess_dk = dk[n_cross_dk:]
            excess_dp = dp[n_cross_kd:]
            for i in range(len(excess_uk)):
                pairs.append({
                    "alpha": (k, excess_uk[i], "up"),
                    "beta":  (partner, excess_up_[i], "up"),
                })
            for i in range(len(excess_dk)):
                pairs.append({
                    "alpha": (k, excess_dk[i], "down"),
                    "beta":  (partner, excess_dp[i], "down"),
                })
    return pairs


def _spin_row_offset(spin_label: str, Ns_phys: int) -> int:
    """Return 0 for 'up' rows (top half of 2Ns A matrix) or Ns_phys for
    'down' rows (bottom half). Used to place the folded orbital
    amplitude at the correct spin block."""
    if spin_label == "up":
        return 0
    if spin_label == "down":
        return Ns_phys
    raise ValueError(f"unknown spin label: {spin_label!r}")


def build_slater_orbitals(
    wavevector_index: np.ndarray,
    eigenvector: np.ndarray,
    column_spin: np.ndarray,
    site_positions: np.ndarray,
    cell_shape: np.ndarray,
    subshape: np.ndarray,
    theta: np.ndarray,
    pair_list: list[dict],
) -> np.ndarray:
    """Extract physical Slater orbitals for every pair member into a
    (2*Ns_phys, 2*len(pair_list)) complex matrix A (spec §3.1, §4.1).

    Column layout: for pair index p in [0, len(pair_list)):
      - A[:, 2*p]     = ψ_alpha(i, spin_alpha)  (top half if alpha is up)
      - A[:, 2*p+1]   = ψ_beta (i, spin_beta )  (top half if beta is up)

    Amplitudes follow the v2.1 positive-Bloch convention (spec §3.1):
      ψ_alpha(i, sigma) = v[k_alpha, row(sigma, sub_offset(r_i)), col_alpha]
                       * exp(+i k_folded · folded_cell(r_i))
                       * exp(+i theta · r_i / L_phys)
                       / sqrt(nvol_folded)

    Non-zero only when sigma == column_spin[col_alpha] (mixed block is
    out of scope in v3).
    """
    from .sublattice_unfold import (
        decode_physical_site,
        encode_folded_orbital,
        folded_row_indices,
    )

    wavevector_index = np.asarray(wavevector_index, dtype=np.int64)
    eigenvector = np.asarray(eigenvector, dtype=np.complex128)
    column_spin = np.asarray(column_spin, dtype=np.int64)
    site_positions = np.asarray(site_positions, dtype=np.int64)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    subshape = np.asarray(subshape, dtype=np.int64)
    theta = np.asarray(theta, dtype=np.float64)

    Ns_phys = site_positions.shape[0]
    nvol_folded = wavevector_index.shape[0]
    subvol = int(np.prod(subshape))
    norb_folded = subvol  # v3 requires norb_orig == 1
    L_folded = (cell_shape // subshape).astype(np.float64)
    L_phys = cell_shape.astype(np.float64)

    inv_sqrt = 1.0 / np.sqrt(float(nvol_folded))
    theta_over_L = theta / L_phys

    # Pre-compute per-site (folded_cell, folded_orb) for row lookup.
    folded_cell = np.empty((Ns_phys, 3), dtype=np.int64)
    folded_orb_per_site = np.empty(Ns_phys, dtype=np.int64)
    for i in range(Ns_phys):
        fc, so = decode_physical_site(site_positions[i], subshape)
        folded_cell[i] = fc
        folded_orb_per_site[i] = encode_folded_orbital(0, so, 1, subshape)

    # Pre-compute per-site physical gauge factor exp(+i theta r_i / L_phys).
    phys_arg = np.einsum(
        "d,id->i", theta_over_L, site_positions.astype(np.float64)
    )
    phys_up = np.exp(+1j * phys_arg)

    A = np.zeros((2 * Ns_phys, 2 * len(pair_list)), dtype=np.complex128)

    for p, pair in enumerate(pair_list):
        for offset, member in enumerate(("alpha", "beta")):
            k_row, col, spin_label = pair[member]
            # k_folded_phys per-direction = 2 pi n_tilde / L_folded
            k_folded = (
                2.0 * np.pi
                * wavevector_index[k_row].astype(np.float64) / L_folded
            )
            # exp(+i k_folded · folded_cell_i)
            kf_dot_fc = np.einsum(
                "d,id->i", k_folded, folded_cell.astype(np.float64)
            )
            plane_wave = np.exp(+1j * kf_dot_fc) * inv_sqrt * phys_up
            # Which nd rows to read: pure-spin per v3 scope.
            row_up, row_dn = np.empty(Ns_phys, dtype=np.int64), np.empty(
                Ns_phys, dtype=np.int64
            )
            for i in range(Ns_phys):
                row_up[i], row_dn[i] = folded_row_indices(
                    int(folded_orb_per_site[i]), norb_folded
                )
            if spin_label == "up":
                v_row = row_up
            else:
                v_row = row_dn
            v_vals = np.array(
                [eigenvector[k_row, int(v_row[i]), col] for i in range(Ns_phys)],
                dtype=np.complex128,
            )
            spin_offset = _spin_row_offset(spin_label, Ns_phys)
            A[spin_offset:spin_offset + Ns_phys, 2 * p + offset] = (
                plane_wave * v_vals
            )

    return A


def build_fij_general(A: np.ndarray) -> np.ndarray:
    """Assemble the antisymmetric F matrix from the 2-column-per-pair
    Slater amplitude matrix A (spec §3.3).

    F[iσ, jσ'] = sum_{p} [ A[iσ, 2p] * A[jσ', 2p+1]
                          - A[iσ, 2p+1] * A[jσ', 2p] ]

    F.shape = (A.shape[0], A.shape[0]) and F.T = -F elementwise.
    """
    A = np.asarray(A, dtype=np.complex128)
    n_row = A.shape[0]
    n_col = A.shape[1]
    assert n_col % 2 == 0, "A must have even column count (pairs of orbitals)"
    F = np.zeros((n_row, n_row), dtype=np.complex128)
    for p in range(n_col // 2):
        alpha = A[:, 2 * p]
        beta = A[:, 2 * p + 1]
        F += np.outer(alpha, beta) - np.outer(beta, alpha)
    return F
