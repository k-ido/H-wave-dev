"""(k, -k) time-reversal pair Fij builder.

Spec sections 3.2 / 3.3 / 3.4 / 5.1.

Physical-basis amplitude convention (negative-Bloch, matches H-wave's
``np.fft.fftn(..., norm='forward')`` k->r kernel ``exp(-i k r)``):

    psi_phys(r, k) = v(k) * exp(-i k_phys * r) / sqrt(N_k)

where k_phys = tilde_k + theta/L. For the (k, -k) time-reversal partner,
the down side carries the OPPOSITE-sign theta correction:

    A_up_{i, alpha(k, n)}   = (1/sqrt N) e^{-i tilde_k r_i} v_{k up n}   e^{-i theta r_i / L}
    A_down_{j, alpha(k, n)} = (1/sqrt N) e^{+i tilde_k r_j} v_{k_p down n} e^{+i theta r_j / L}

(``k_p`` is the partner row whose tilde index residue matches
``-n - 2*twist_offset`` mod L; for self-pair k the partner equals the
row itself and the same eigenvector column is reused.)

This convention satisfies F_phys[i, j] = A_up[i, alpha] * A_down[j, alpha]
summed over occupied pairs, so the density check
G^sigma_{ij} = sum_alpha conj(A^sigma_{i, alpha}) A^sigma_{j, alpha}
matches H-wave's physical Green function from ``_save_greenone``
(`G_phys = e^{+i theta (r_i - r_j) / L} G_tilde`).
"""
from __future__ import annotations

import numpy as np

from .partner_index import find_partner_rows


def _phys_phase_up(site_positions, theta, L):
    """e^{-i sum_d theta_d r_d / L_d} per site (NEGATIVE exponent), shape (Ns,)."""
    arg = -np.einsum("d,id->i", theta / L, site_positions)
    return np.exp(1j * arg)


def _phys_phase_down(site_positions, theta, L):
    """e^{+i sum_d theta_d r_d / L_d} per site (POSITIVE exponent), shape (Ns,).

    A^down carries the time-reversal partner's negative-Bloch factor
    exp(-i k_phys' r_j) = exp(+i k_phys r_j) = exp(+i (tilde_k + theta/L) r_j).
    The tilde_k part lives in plane_wave_down; this helper supplies the
    theta/L part with the matching POSITIVE sign.
    """
    arg = np.einsum("d,id->i", theta / L, site_positions)
    return np.exp(1j * arg)


def build_amplitudes(
    wavevector_index,
    eigenvector,
    stepped_occupation,
    column_spin,
    column_mu_group,
    site_positions,
    norb_orig,
    theta,
    L,
):
    """Build the physical-basis A^up, A^down occupied-orbital matrices.

    Returns
    -------
    A_up, A_down : (Ns, N_occ) complex arrays
        Each column is one occupied pair alpha(k, n) built from the
        physical-basis eigenstates. ``A_up @ A_down.T`` yields F_phys.

    The down orbital column index aligns with the up column index via the
    (k, -k) partner lookup; for self-pair k the partner equals the up's
    own row.

    Raises
    ------
    ValueError
        If the partner-mapped (k, -k) occupation set is not consistent
        with the actual occupied down set (Codex finding 3: in magnetic
        UHF or spin-dependent fillings, the up-and-down occupied sets
        need not coincide through the time-reversal partner map).
    """
    wavevector_index = np.asarray(wavevector_index, dtype=np.int64)
    eigenvector = np.asarray(eigenvector, dtype=np.complex128)
    stepped_occupation = np.asarray(stepped_occupation, dtype=np.float64)
    column_spin = np.asarray(column_spin, dtype=np.int64)
    column_mu_group = np.asarray(column_mu_group, dtype=np.int64)
    site_positions = np.asarray(site_positions, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    L = np.asarray(L, dtype=np.int64)

    nvol, nd = stepped_occupation.shape
    Ns = site_positions.shape[0]
    if norb_orig != 1:
        raise NotImplementedError(
            "v1 spec section 7 restricts to norb_orig == 1; got "
            f"{norb_orig}. Multi-orbital is a v2 extension."
        )

    partner_rows, is_self_pair = find_partner_rows(wavevector_index, theta, L)

    k_per_row = 2.0 * np.pi * wavevector_index.astype(np.float64) / L
    # Negative-Bloch convention matching H-wave's np.fft.fftn(norm='forward'):
    # k -> r kernel is exp(-i k r), so the real-space eigenstate at momentum k
    # carries exp(-i k r) (Codex finding 1).
    plane_wave_up = np.exp(-1j * np.einsum("kd,id->ki", k_per_row, site_positions))
    plane_wave_down = np.conj(plane_wave_up)  # exp(+i tilde_k r) for partner k' = -k
    phys_up = _phys_phase_up(site_positions, theta, L)
    phys_down = _phys_phase_down(site_positions, theta, L)

    sqrt_Nk = np.sqrt(float(nvol))

    up_cols = np.where(column_spin == 0)[0]
    down_cols = np.where(column_spin == 1)[0]
    if len(up_cols) == 0 or len(down_cols) == 0:
        raise ValueError(
            "Sz-fixed mode requires both up-only and down-only column "
            "blocks; got column_spin = " + str(column_spin.tolist())
        )

    # Pair-closure validation (Codex finding 3): the (k, -k) construction
    # assumes every occupied up state has its time-reversal partner occupied
    # on the down side, and vice versa. Magnetic UHF or asymmetric fillings
    # break this. Refuse rather than silently produce the wrong Slater state.
    occ_up_rows = set()
    occ_down_rows = set()
    for n_row in range(nvol):
        for col_up in up_cols:
            if stepped_occupation[n_row, col_up] >= 0.5:
                occ_up_rows.add(n_row)
                break
        for col_down in down_cols:
            if stepped_occupation[n_row, col_down] >= 0.5:
                occ_down_rows.add(n_row)
                break
    partner_of_occ_up = {int(partner_rows[n]) for n in occ_up_rows}
    if partner_of_occ_up != occ_down_rows:
        missing_in_down = sorted(partner_of_occ_up - occ_down_rows)
        extra_in_down = sorted(occ_down_rows - partner_of_occ_up)
        raise ValueError(
            "(k, -k) pair-closure violated: the time-reversal partner of "
            "the occupied up set does not equal the occupied down set. "
            f"partner(occ_up) has {len(partner_of_occ_up)} entries, "
            f"occ_down has {len(occ_down_rows)}; "
            f"missing in down: {missing_in_down[:10]}; "
            f"extra in down: {extra_in_down[:10]}. "
            "v1 bridge requires paramagnetic (k, -k) closure; magnetic / "
            "asymmetric occupations are not supported."
        )

    A_up_list = []
    A_down_list = []
    for n_row in range(nvol):
        for col_up in up_cols:
            if stepped_occupation[n_row, col_up] < 0.5:
                continue
            partner_n = int(partner_rows[n_row])
            u_amp_up = eigenvector[n_row, 0, col_up]
            col_down = down_cols[0]
            u_amp_down = eigenvector[partner_n, 1, col_down]

            A_up_list.append(
                (1.0 / sqrt_Nk) * plane_wave_up[n_row] * u_amp_up * phys_up
            )
            A_down_list.append(
                (1.0 / sqrt_Nk) * plane_wave_down[n_row] * u_amp_down * phys_down
            )

    if not A_up_list:
        raise ValueError("no occupied up-spin states found")

    A_up = np.stack(A_up_list, axis=1)
    A_down = np.stack(A_down_list, axis=1)
    return A_up, A_down


def build_fij_phys(A_up, A_down):
    """Return F^phys_{ij} = (A_up @ A_down.T)_{ij}, shape (Ns, Ns) complex.

    F is built from c^dag c^dag coefficients, both of which are
    pre-conjugated by the (k, -k) negative-Bloch construction. The result
    has translation-invariant ``(r_j - r_i)`` structure (Codex finding 1).
    """
    A_up = np.asarray(A_up, dtype=np.complex128)
    A_down = np.asarray(A_down, dtype=np.complex128)
    return A_up @ A_down.T
