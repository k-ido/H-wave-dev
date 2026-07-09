#!/usr/bin/env bash
# Orchestrate the H-wave UHFk -> bridge -> mVMC PairProduct E2E for one case.
# Usage: ./run.sh case_pbc   (or case_apbc)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE="${1:-case_pbc}"
ROOT="${HERE}/../../.."
CASE_DIR="${HERE}/${CASE}"
[[ -d "${CASE_DIR}" ]] || { echo "case not found: ${CASE_DIR}" >&2; exit 1; }

# We reuse the mVMC build under apbc_complexuhf/build/mvmc to avoid two copies.
MVMC_BUILD="${HERE}/../apbc_complexuhf/build/mvmc/build"
VMCDRY="${MVMC_BUILD}/src/mVMC/vmcdry.out"
VMC="${MVMC_BUILD}/src/mVMC/vmc.out"
[[ -x "${VMCDRY}" ]] || { echo "vmcdry.out missing; run ../apbc_complexuhf/build_complexuhf.sh first" >&2; exit 1; }
[[ -x "${VMC}" ]] || { echo "vmc.out missing; rerun build_complexuhf.sh after pulling the updated script" >&2; exit 1; }

WORK="${CASE_DIR}/work"
rm -rf "${WORK}"
mkdir -p "${WORK}"

# ---- step 1: H-wave UHFk SCF ----
HWAVE_WORK="${WORK}/hwave"
mkdir -p "${HWAVE_WORK}"
for f in CoulombIntra.dat Geometry.dat OneBodyG.dat Transfer.dat geometry_uhf.dat input.toml; do
  cp "${CASE_DIR}/${f}" "${HWAVE_WORK}/${f}"
done

(
  cd "${HWAVE_WORK}"
  # Shim the deprecated np.float alias before importing hwave (matches
  # apbc_complexuhf/run.sh). Use PYTHONPATH so this works whether the
  # package is editable-installed or not.
  PYTHONPATH="${ROOT}/src" python3 -c "
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import sys
sys.argv = ['hwave', 'input.toml']
from hwave.qlms import main
main()
"
)

[[ -f "${HWAVE_WORK}/output/eigen.npz" ]]
[[ -f "${HWAVE_WORK}/output/occupation.npz" ]]
[[ -f "${HWAVE_WORK}/output/greenone.dat" ]]
echo "  H-wave UHFk SCF: ${HWAVE_WORK}/output/*"

# ---- step 2: vmcdry to produce mVMC .def files ----
MVMC_WORK="${WORK}/mvmc"
mkdir -p "${MVMC_WORK}"
cp "${CASE_DIR}/stan.in" "${MVMC_WORK}/stan.in"
cd "${MVMC_WORK}"
"${VMCDRY}" stan.in </dev/null > vmcdry.log 2>&1
[[ -f "namelist.def" ]]
# StdFace's FermionHubbard generator writes ``ComplexType 0`` (real
# orbitalidx.def). The bridge produces complex F values (APBC carries a
# nontrivial phase, even under PBC the (k,-k) construction is complex
# off-diagonally), so flip the header to ``ComplexType 1`` here. The
# numerical content of the value rows remains real-zero in PBC and
# real+imag in APBC; mVMC accepts both interchangeably under
# ComplexType=1.
sed -i -E 's/^ComplexType +0/ComplexType 1/' orbitalidx.def
echo "  vmcdry.out: ${MVMC_WORK}/{namelist,modpara,orbitalidx,trans,coulombintra,...}.def"

# ---- step 3: bridge ----
# Rank-lift noise amplitude. Default 1e-8 matches the bridge's own CLI
# default (stable plateau where mVMC <H> reaches UHF within VMC stderr
# at NVMCSample = 10000). Override per run via:
#     EPSILON_NOISE=1e-7 ./run.sh case_apbc
EPSILON_NOISE="${EPSILON_NOISE:-1.0e-8}"
RNG_SEED="${RNG_SEED:-7919}"
cd "${ROOT}"
python3 tools/uhfk_to_mvmc.py \
    --input        "${HWAVE_WORK}/input.toml" \
    --eigen        "${HWAVE_WORK}/output/eigen.npz" \
    --occupation   "${HWAVE_WORK}/output/occupation.npz" \
    --geometry     "${HWAVE_WORK}/geometry_uhf.dat" \
    --orbitalidx   "${MVMC_WORK}/orbitalidx.def" \
    --output       "${MVMC_WORK}/zqp_orbital_uhfk.dat" \
    --check-density \
    --onebodyg-uhf "${HWAVE_WORK}/output/greenone.dat" \
    --epsilon-noise "${EPSILON_NOISE}" \
    --rng-seed "${RNG_SEED}"
echo "  bridge wrote: ${MVMC_WORK}/zqp_orbital_uhfk.dat (density check OK, epsilon=${EPSILON_NOISE})"

# ---- step 4: register InOrbital in namelist.def ----
NAMELIST="${MVMC_WORK}/namelist.def"
# Remove any prior entry for InOrbital* lines, then append the bridge file.
sed -i '/^InOrbital/d' "${NAMELIST}"
sed -i '/^InOrbitalAntiParallel/d' "${NAMELIST}"
echo "InOrbital ${MVMC_WORK}/zqp_orbital_uhfk.dat" >> "${NAMELIST}"
# Disable Gutzwiller / Jastrow / GeneralRBM projection lines: the bridge
# produces a pure Slater initial WF, and StdFace defaults to nonzero
# projection parameters that would multiply the Slater by a non-trivial
# correlator. Without InGutzwiller / InJastrow files mVMC starts those
# parameters from random values, giving an energy that does NOT match
# the UHF Slater. Commenting these lines makes mVMC ignore the
# correlators entirely.
sed -i 's/^Gutzwiller/#Gutzwiller/' "${NAMELIST}"
sed -i 's/^Jastrow/#Jastrow/' "${NAMELIST}"
echo "  namelist.def updated to read InOrbital ${MVMC_WORK}/zqp_orbital_uhfk.dat"
echo "  Gutzwiller / Jastrow projection disabled (pure Slater test)"

# ---- step 5: vmc.out (NVMCCalMode=1, NSROptItrStep=0 -> measurement only) ----
cd "${MVMC_WORK}"
"${VMC}" namelist.def > vmc.log 2>&1 \
    || { echo "vmc.out failed; see ${MVMC_WORK}/vmc.log" >&2; tail -50 vmc.log >&2; exit 1; }

# zvo_out_xxx.dat holds bin-by-bin <H>, <H^2>, ... lines.
OUT_FILE="$(ls output/zvo_out_*.dat 2>/dev/null | head -n1)"
[[ -n "${OUT_FILE}" ]] || { echo "mVMC produced no zvo_out output" >&2; exit 1; }
echo "  vmc.out -> ${MVMC_WORK}/${OUT_FILE}"

# ---- step 6: compare energies ----
cd "${HERE}"
python3 compare.py "${HWAVE_WORK}/output/energy.dat" "${MVMC_WORK}/${OUT_FILE}" "${CASE}"
