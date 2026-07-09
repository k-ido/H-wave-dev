"""Shared target-occupation assertion helper (spec §4.1, §6.1, §6.2).

Both pytest and the mVMC E2E validation harness (`run.sh`) invoke this
helper immediately after H-wave SCF, BEFORE the bridge runs. Target
values live only in this module (module-level `_CASE_TARGETS` constant)
so a single edit propagates everywhere.

CLI usage:
    python3 scripts/assert_occupation.py <case_dir> <work_dir>

Programmatic usage:
    from assert_occupation import assert_case_occupation
    assert_case_occupation(case_dir_path, work_dir_path)
"""
from __future__ import annotations

import os
import sys

import numpy as np


# Single source of truth for every case's target (spin -> list of
# tuple(wavevector_index rows)). Keys must match the fixture directory
# names under tests/validation/uhfk_mvmc_pairproduct/.
_CASE_TARGETS: dict[str, dict] = {
    "case_pbc_sz2": {
        # Target FM UHF ground state at PBC L=8, U small enough to
        # remain paramagnetic-like: 3 up (at k=0, ±2π/8=±π/4) + 1 down
        # (at k=0). Spec §6.2 justification: cross pair at self-pair k=0
        # + same-spin up-up excess at non-self (+π/4, -π/4) block.
        "up": [(0, 0, 0), (1, 0, 0), (-1, 0, 0)],
        "down": [(0, 0, 0)],
    },
    "case_zeeman_sz_free": {
        # 1D L=8 PBC Hubbard with Ne=4, no 2Sz constraint, flag_fock=false.
        # Plan intent was an on-site Zeeman-like Transfer entry to force a
        # definite state, but H-wave's uhfk drops spin-block Transfer
        # indices when enable_spin_orbital=false (uhfk.py:1136-1146) and
        # the Hermite check hard-errors on them (uhfk.py:879). The
        # bridge in turn rejects enable_spin_orbital=true, so the Zeeman
        # entries could not survive in the fixture; the file carries only
        # the standard NN hopping. H-wave then converges to a
        # symmetry-broken FM state selected by the initial random Green
        # seed: Ne_up=3 at k=0 and ±π/4, Ne_down=1 at k=0 — the same
        # target occupation as case_pbc_sz2 but reached via the Sz-free
        # (single-mu-group, no 2Sz) branch, which is exactly the B ケース
        # that Task 10 exercises. §3.2 walkthrough (canonical blocks):
        #   self k=0:    NN_up=1, NN_down=1 → n_cross=1, excess=0/0 ✓
        #   self k=π:    empty ✓
        #   (+1,-1):     NN_up_k=NN_up_p=1, NN_down=0 → excess_up_k=
        #                 excess_up_p=1 (same-spin up-up pair) ✓
        #   (+2,-2):     empty ✓
        #   (+3,-3):     empty ✓
        # Ne = 4 (even).
        "up": [(0, 0, 0), (1, 0, 0), (-1, 0, 0)],
        "down": [(0, 0, 0)],
    },
}


def _occupied_by_spin(occupation, column_spin, wavevector_index):
    """Return dict[spin] = set of tuple(wavevector_index) for occupied
    columns."""
    out = {"up": [], "down": []}
    nvol, nd = occupation.shape
    for k_row in range(nvol):
        for col in range(nd):
            if occupation[k_row, col] < 0.5:
                continue
            spn = int(column_spin[col])
            if spn == 0:
                out["up"].append(tuple(int(v) for v in wavevector_index[k_row]))
            elif spn == 1:
                out["down"].append(tuple(int(v) for v in wavevector_index[k_row]))
    return out


def assert_case_occupation(case_dir, work_dir):
    """Compare H-wave SCF occupation in ``<work_dir>/output/`` against the
    helper's target for ``case_dir.name``. Raises KeyError on unknown
    case, AssertionError on divergence with observed vs target dump."""
    case_dir = os.fspath(case_dir)
    work_dir = os.fspath(work_dir)
    case_name = os.path.basename(os.path.normpath(case_dir))
    if case_name not in _CASE_TARGETS:
        raise KeyError(
            f"unknown case {case_name!r}; add its target to _CASE_TARGETS "
            f"in {__file__} before running the validation harness"
        )
    target = _CASE_TARGETS[case_name]

    occ_path = os.path.join(work_dir, "output", "occupation.npz")
    eig_path = os.path.join(work_dir, "output", "eigen.npz")
    if not os.path.isfile(occ_path):
        raise AssertionError(
            f"occupation.npz missing at {occ_path}; run H-wave SCF first "
            f"(case {case_name})"
        )
    if not os.path.isfile(eig_path):
        raise AssertionError(
            f"eigen.npz missing at {eig_path}; ensure input.toml requests "
            f"eigen output (case {case_name})"
        )
    occ_data = np.load(occ_path, allow_pickle=False)
    eig_data = np.load(eig_path, allow_pickle=False)
    occupation = occ_data["occupation"]
    column_spin = occ_data["column_spin"]
    wavevector_index = eig_data["wavevector_index"]

    observed = _occupied_by_spin(occupation, column_spin, wavevector_index)
    obs_sets = {k: sorted(v) for k, v in observed.items()}
    tgt_sets = {k: sorted(v) for k, v in target.items()}
    if obs_sets != tgt_sets:
        raise AssertionError(
            f"occupation mismatch for case {case_name}:\n"
            f"  observed = {obs_sets}\n"
            f"  target   = {tgt_sets}\n"
            f"If the fixture Transfer.dat was intentionally changed "
            f"(e.g. Zeeman stabilization), update "
            f"_CASE_TARGETS[{case_name!r}] in {__file__} so pytest and "
            f"run.sh stay in sync."
        )


def _cli(argv):
    if len(argv) != 2:
        print(
            "usage: python3 assert_occupation.py <case_dir> <work_dir>",
            file=sys.stderr,
        )
        return 2
    try:
        assert_case_occupation(argv[0], argv[1])
    except (KeyError, AssertionError) as e:
        print(f"OCCUPATION ASSERTION FAILED: {e}", file=sys.stderr)
        return 1
    print(f"occupation check OK for {os.path.basename(argv[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
