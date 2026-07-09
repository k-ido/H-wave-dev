"""Compare H-wave UHFk total energy against mVMC initial energy from
``zvo_out_*.dat`` produced with the bridge's zqp_orbital_uhfk.dat.

Usage: compare.py <hwave_energy_dat> <mvmc_zvo_out_dat> <case_name>

H-wave's ``energy.dat`` lists one quantity per line (Total, NumElec,
NumDouble, ...). We grab the "Total" line.

mVMC's ``zvo_out_xxx.dat`` lists per-bin samples; each line has the
expectation values <H>, <H^2>, ... For NSROptItrStep=0 only the initial
WF is measured. We average the first column (<H>) over the bins and
report mean + standard error.
"""
from __future__ import annotations

import os
import re
import sys
from statistics import mean, stdev
from typing import List


def parse_hwave_energy(path: str) -> float:
    """Parse H-wave UHFk's ``energy.dat`` and return Energy_Total.

    Format (uhfk.py:_save_results writes one ``KEY = VALUE`` per line):
      Energy_Total        = -3.7500000000001137
      Energy_Band         = ...
      ...
    """
    with open(path) as fp:
        for line in fp:
            m = re.match(
                r"\s*Energy_Total\s*=\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)",
                line,
            )
            if m:
                return float(m.group(1))
    raise RuntimeError(f"could not find Energy_Total in {path}")


def parse_mvmc_zvo_out(path: str) -> List[float]:
    """Return the list of per-bin <H> values from a zvo_out_*.dat file.

    Format (mVMC 1.4.0): each line is "<H> <H^2> <doublon> <singleon> ...".
    Comment lines start with '#'. We collect column 0.
    """
    samples: List[float] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if not toks:
                continue
            try:
                samples.append(float(toks[0]))
            except ValueError:
                continue
    if not samples:
        raise RuntimeError(f"no numeric rows parsed from {path}")
    return samples


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    hwave_path = sys.argv[1]
    mvmc_path = sys.argv[2]
    case = sys.argv[3]

    e_uhf = parse_hwave_energy(hwave_path)
    bins = parse_mvmc_zvo_out(mvmc_path)
    e_mvmc = mean(bins)
    se_mvmc = stdev(bins) / (len(bins) ** 0.5) if len(bins) > 1 else 0.0

    delta = e_mvmc - e_uhf
    rel = abs(delta) / max(abs(e_uhf), 1e-12)
    # When n_bins >= 2 the standard error of the mean is well-defined;
    # otherwise (single bin, see mVMC NSROptItrStep=0 + NDataQtySmp=1)
    # fall back to a pure relative tolerance.
    if len(bins) >= 2 and se_mvmc > 0:
        sigma_sigmas = abs(delta) / se_mvmc
        print(f"[{case}] H-wave UHFk total = {e_uhf:.10e}")
        print(f"[{case}] mVMC initial mean = {e_mvmc:.10e}  "
              f"stderr = {se_mvmc:.2e}  (n_bins={len(bins)})")
        print(f"[{case}] delta = {delta:+.4e}  ({sigma_sigmas:.2f} sigma, "
              f"{rel * 100:.2f}% rel)")
        if sigma_sigmas <= 3.0 and rel < 0.05:
            print(f"[{case}] OK  (within 3 sigma and 5% rel)")
            return 0
        print(f"[{case}] FAIL  (delta exceeds 3 sigma or 5% rel)")
        return 1
    # Single-bin path.
    print(f"[{case}] H-wave UHFk total = {e_uhf:.10e}")
    print(f"[{case}] mVMC initial mean = {e_mvmc:.10e}  "
          f"(n_bins={len(bins)}; stderr undefined)")
    print(f"[{case}] delta = {delta:+.4e}  ({rel * 100:.2f}% rel)")
    if rel < 0.01:  # 1% sanity for single-bin case
        print(f"[{case}] OK  (within 1% rel; single-bin VMC sample)")
        return 0
    print(f"[{case}] FAIL  (delta exceeds 1% rel)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
