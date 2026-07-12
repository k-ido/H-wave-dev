# case_soc_rashba_2d_sub_apbc (v3.6 shipping fixture)

Second SOC + SubShape > [1, 1, 1] fixture, extending v3.5's
`case_soc_rashba_2d_sub` with antiperiodic boundary condition in the
`x` direction. Ships in v3.6 (spec
`docs/superpowers/specs/2026-07-09-uhfk-mvmc-pairproduct-general-v36-design.md`,
plan `docs/superpowers/plans/2026-07-09-uhfk-mvmc-pairproduct-general-v36.md`
Task 2).

## Geometry / interaction

- `CellShape` = `[6, 4, 1]`
- `SubShape` = `[2, 2, 1]` (folded BZ `L_folded = [3, 2, 1]`)
- `BoundaryCondition` = `["antiperiodic", "periodic", "periodic"]` (single-direction APBC in `x`)
- Rashba SOC: `t = 1`, `alpha = 0.5`, `U = 2` (identical to v3.5 `case_soc_rashba_2d_sub`)
- Total physical sites: 24 (spin-orbital dim 48)

## Ncond gap check

Phase 2 step 2b: `soc_pilot_gap.py` with the fixture's own
`BoundaryCondition = ["antiperiodic", "periodic", "periodic"]` (twist
`theta = (pi, 0, 0)`, folded BZ `L_folded = [3, 2, 1]`):

```
PYTHONPATH=src python tests/validation/uhfk_mvmc_pairproduct/scripts/soc_pilot_gap.py \
    --case tests/validation/uhfk_mvmc_pairproduct/case_soc_rashba_2d_sub_apbc \
    --ncond 4,6,8,10,12,14
```

- `Ncond= 4`: gap = 2.333e-01 (OK)
- `Ncond= 6`: gap = 7.389e-02 (OK; close to `>= 5e-2` boundary)
- `Ncond= 8`: gap = 1.532e-01 (OK)  ← fixture choice for physics
  consistency with v3.5 `case_soc_rashba_2d_sub` (same 1/6 filling on
  24 sites).
- `Ncond=10`: gap = 1.298e+00 (OK)
- `Ncond=12`: gap = 1.211e-01 (OK)
- `Ncond=14`: gap = 3.513e-01 (OK)

`Ncond = 8` matches v3.5's PBC fixture at the same physical filling, so
the mVMC `<H>` delta between the PBC and APBC runs stays directly
comparable at the same Ncond. The plan (Task 2b) allows the smallest
gap-passing choice; `Ncond = 4` also satisfies `>= 5e-2` and is
reserved as a fallback if the composite-element check in Phase 2 step
2d fails to find a cross-spin sub_offset-differing element at
`Ncond = 8`.

## Composite element manifest

`composite_element.json` (spec §6.2 addendum), produced by
`scripts/phase2_produce_manifest.py`. Selected composite:

- `(i_c, s_c, j_c, t_c) = (5, 1, 18, 0)` (cross-spin, sub_offset_x
  differs, `|G| = 8.09e-02` >= 1e-3)
- Physical positions: `r_i = (5, 0, 0)`, `r_j = (0, 3, 0)`
- Sub-offsets: `sub_offset(i) = (1, 0, 0)`, `sub_offset(j) = (0, 1, 0)`
  (differ in x AND y; the x-differ carries the APBC signature and the
  y-differ ensures the mutation matrix's k-space partial sums do not
  degenerate to unit phases at k = (pi, pi, 0).)

Per-mutation sensitivity at the composite element (Phase 2 producer, real
mutation formulas — no surrogate):

| Mutation | delta_M | T_M | Status |
|----------|---------|-----|--------|
| M-gauge-1 (twist sign flip) | 8.09e-02 | 8.09e-03 | OK |
| M-gauge-2 (twist unit)      | 1.44e-01 | 8.09e-03 | OK |
| M-gauge-3 (twist L)         | 1.56e-01 | 8.09e-03 | OK |
| M-gauge-4 (sub_offset sign) | 1.42e-01 | 8.09e-03 | OK |
| M-gauge-5 (dr swap in twist)| 8.09e-02 | 8.09e-03 | OK |
| M-ship-1 (phys_dn sign)     | 8.09e-02 | 8.09e-03 | OK |
| M-ship-2 (phys_dn unit)     | 1.44e-01 | 8.09e-03 | OK |
| M-ship-3 (phys_dn L)        | 1.56e-01 | 8.09e-03 | OK |
| M-ship-4 (kf offset sign)   | 1.42e-01 | 8.09e-03 | OK |
| M-ship-5 (drop sub_offset)  | 7.13e-02 | 8.09e-03 | OK |

The Phase 2 producer's composite selector requires every M-gauge-1..5
and M-ship-1..5 delta to exceed the §4.4 10% floor at the candidate
composite BEFORE committing. This aligns the manifest with the Phase 4
mutation matrix (which runs the real formulas on the persisted
snapshot) and the Phase 6 G4 gate (which runs the real formulas on the
current-run workspace via `soc_apbc_topology_guard.py`). No mutation
uses a closed-form surrogate; both Phase 2 producer and Phase 6 gate
rebuild the shipping A matrix per M-ship-* mutation.

`T_M_per_mutation` is `max(1e-5, 0.10 * |G_base|)` per spec §4.4. Every
mutation delta at this composite exceeds T_M by at least 8x, so the
manifest is stable against small SCF re-convergence noise between
Phase 2 producer and Phase 6 G4 runs.

## v3.6 gate coverage

Phase 6 `run.sh case_soc_rashba_2d_sub_apbc` produces:

- `${WORK_DIR}/hwave/{eigen,green,occupation,energy}.dat`
- `${WORK_DIR}/bridge_zeronoise/{...}` via `--epsilon-noise 0`
  (G0-writer-check consumes this)
- `${WORK_DIR}/bridge/{...}` via `--epsilon-noise 1e-8` (shipping)
- `${WORK_DIR}/complexuhf/zvo_UHF_cisajs.dat` (G2a/G2b external check)
- `${WORK_DIR}/mvmc/zvo_out_selected.dat` (G3)

The seven-gate contract (G0-writer-check, G1, G2a-emitted-F,
G2a-in-memory-A, G2b, G3, G4) is validated per §6.2 of the v3.6 design
spec.

## Deferred (v3.7+)

- Multi-direction APBC (`AP-AP-P`, `AP-AP-AP`) + SubShape > 1 is NOT
  covered here; the v3.6 CLI (post-Phase 6) rejects that triple with a
  clear "deferred to v3.7" message per spec §8.
