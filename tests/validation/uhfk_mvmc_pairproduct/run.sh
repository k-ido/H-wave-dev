#!/usr/bin/env bash
# Orchestrate the H-wave UHFk -> bridge -> mVMC PairProduct E2E for one case.
# Usage: ./run.sh case_pbc   (or case_apbc)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE="${1:-case_pbc}"
ROOT="${HERE}/../../.."
CASE_DIR="${HERE}/${CASE}"
[[ -d "${CASE_DIR}" ]] || { echo "case not found: ${CASE_DIR}" >&2; exit 1; }

# v3.1 spec §1 out-of-scope: SubShape > 1 + SOC has an additional
# class-consistency issue (independent of the trans.def emitter added
# in Task 15) that is deferred to v3.2. Reject the fixture up-front
# so the docker E2E loop can't accidentally exercise the deferred path.
if [[ "${CASE}" == "case_soc_rashba_2d_sub" ]]; then
  echo "case_soc_rashba_2d_sub: deferred to v3.2 (v3.1 spec §1 out-of-scope)" >&2
  exit 1
fi

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

# ---- step 1.5: harness-gate — assert target occupation per case ----
if grep -qE "case_(pbc_sz2|zeeman_sz_free|soc_rashba_2d_nosub|soc_rashba_2d_nosub_apbc|soc_rashba_2d_sub)" <<< "${CASE}"; then
  python3 "${HERE}/scripts/assert_occupation.py" "${CASE_DIR}" "${HWAVE_WORK}" || {
    echo "harness-gate: occupation assertion failed for ${CASE}" >&2
    exit 1
  }
fi

# ---- step 1.6: SOC canonical-mix gate for case_soc_rashba_2d_sub ----
# Spec §4.2.3 criterion 5: SubShape [2,2,1] on CellShape [6,4,1] must
# yield a folded BZ with at least one non-self canonical block. If a
# future refactor of find_partner_rows / compute_canonical_reps or of
# the fixture's SubShape choice silently degenerates every folded k to a
# self-pair, the block-mix contract is broken and this fixture no longer
# exercises the non-self canonical path — fail loud.
if [[ "${CASE}" == "case_soc_rashba_2d_sub" ]]; then
  HWAVE_WORK="${HWAVE_WORK}" PYTHONPATH="${ROOT}/src:${ROOT}" python3 - <<'PYEOF' || exit 1
import os, sys
import numpy as np
from tools._uhfk_to_mvmc.general_fij_builder import compute_canonical_reps
from tools._uhfk_to_mvmc.partner_index import find_partner_rows

hwave_work = os.environ["HWAVE_WORK"]
eig = np.load(os.path.join(hwave_work, "output", "eigen.npz"), allow_pickle=False)
# Fixture pins BoundaryCondition = periodic in all directions -> theta = 0.
theta = np.zeros(3)
# CellShape=[6,4,1] / SubShape=[2,2,1] -> folded BZ [3,2,1].
L_folded = np.array([3, 2, 1], dtype=np.int64)
partner_rows, _ = find_partner_rows(eig["wavevector_index"], theta, L_folded)
canonical, self_pairs = compute_canonical_reps(partner_rows, eig["wavevector_index"])
n_non_self = len(canonical) - len(self_pairs)
if n_non_self < 1:
    print(
        f"harness-gate: case_soc_rashba_2d_sub reported {n_non_self} non-self "
        f"canonical blocks; SubShape choice degenerated",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"  canonical mix: {n_non_self} non-self, {len(self_pairs)} self")
PYEOF
fi

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
# v3 A/B cases: StdFace's FermionHubbard also emits orbitalidxgen.def
# (6-column General) when 2Sz != 0. Flip its ComplexType too.
if [[ -f orbitalidxgen.def ]]; then
  sed -i -E 's/^ComplexType +0/ComplexType 1/' orbitalidxgen.def
fi
echo "  vmcdry.out: ${MVMC_WORK}/{namelist,modpara,orbitalidx,trans,coulombintra,...}.def"

# ---- step 3: bridge ----
# Rank-lift noise amplitude. Default 1e-8 matches the bridge's own CLI
# default (stable plateau where mVMC <H> reaches UHF within VMC stderr
# at NVMCSample = 10000). Override per run via:
#     EPSILON_NOISE=1e-7 ./run.sh case_apbc
EPSILON_NOISE="${EPSILON_NOISE:-1.0e-8}"
RNG_SEED="${RNG_SEED:-7919}"

# v3 A/B routing: if StdFace produced orbitalidxgen.def (fires when
# 2Sz != 0 or lGC=1), prefer it and route the bridge through the
# General path. Otherwise stay on the v1/v2 AntiParallel-only path.
if [[ -f "${MVMC_WORK}/orbitalidxgen.def" ]]; then
  BRIDGE_ORBITALIDX="${MVMC_WORK}/orbitalidxgen.def"
else
  BRIDGE_ORBITALIDX="${MVMC_WORK}/orbitalidx.def"
fi

cd "${ROOT}"
# v3.1 spec §3.8: SOC path additionally emits mVMC trans.def because
# StdFace's FermionHubbardGC generator drops Rashba s != t transfer
# entries. Non-SOC fixtures leave vmcdry's trans.def untouched.
BRIDGE_SOC_ARGS=()
if [[ "${CASE}" == case_soc_* ]]; then
  BRIDGE_SOC_ARGS=(
    --transfer   "${HWAVE_WORK}/Transfer.dat"
    --emit-trans "${MVMC_WORK}/trans.def"
  )
fi
python3 tools/uhfk_to_mvmc.py \
    --input        "${HWAVE_WORK}/input.toml" \
    --eigen        "${HWAVE_WORK}/output/eigen.npz" \
    --occupation   "${HWAVE_WORK}/output/occupation.npz" \
    --geometry     "${HWAVE_WORK}/geometry_uhf.dat" \
    --orbitalidx   "${BRIDGE_ORBITALIDX}" \
    --output       "${MVMC_WORK}/zqp_orbital_uhfk.dat" \
    --check-density \
    --onebodyg-uhf "${HWAVE_WORK}/output/greenone.dat" \
    --epsilon-noise "${EPSILON_NOISE}" \
    --rng-seed "${RNG_SEED}" \
    "${BRIDGE_SOC_ARGS[@]}"
echo "  bridge wrote: ${MVMC_WORK}/zqp_orbital_uhfk.dat (density check OK, epsilon=${EPSILON_NOISE})"
if [[ "${CASE}" == case_soc_* ]]; then
  echo "  bridge wrote: ${MVMC_WORK}/trans.def (SOC: Rashba entries preserved)"
fi

# ---- step 4: register InOrbital(General) in namelist.def ----
NAMELIST="${MVMC_WORK}/namelist.def"
# Remove any prior entry for InOrbital* lines, then append the bridge file
# under the appropriate keyword.
sed -i '/^InOrbital/d' "${NAMELIST}"
sed -i '/^InOrbitalAntiParallel/d' "${NAMELIST}"
sed -i '/^InOrbitalGeneral/d' "${NAMELIST}"
if [[ "${BRIDGE_ORBITALIDX}" == *orbitalidxgen.def ]]; then
  # v3 General path: tell mVMC to consume the 6-column class table and
  # bridge params via InOrbitalGeneral. Uncomment the commented
  # "OrbitalGeneral" line StdFace emitted and comment the AntiParallel
  # "Orbital" line so mVMC does NOT try to parse orbitalidx.def as the
  # main pair table.
  sed -i -E 's|^# OrbitalGeneral|  OrbitalGeneral|' "${NAMELIST}"
  sed -i -E 's|^         Orbital  |#        Orbital  |' "${NAMELIST}"
  # StdFace also emits OrbitalParallel orbitalidxpara.def for 2Sz != 0
  # (grand-canonical-style triplet pair). mVMC rejects when more than one
  # OrbitalX* keyword is active alongside OrbitalGeneral, so comment it.
  sed -i -E 's|^ OrbitalParallel|# OrbitalParallel|' "${NAMELIST}"
  echo "InOrbitalGeneral ${MVMC_WORK}/zqp_orbital_uhfk.dat" >> "${NAMELIST}"
else
  echo "InOrbital ${MVMC_WORK}/zqp_orbital_uhfk.dat" >> "${NAMELIST}"
fi
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
