.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script ``tools/uhfk_to_mvmc.py`` converts an H-wave UHFk SCF result
into an mVMC ``InOrbital`` / ``InOrbitalAntiParallel`` initial wave
function file (``zqp_orbital_uhfk.dat``), so that mVMC's PairProduct
state can be initialized from the H-wave Slater determinant.

Scope (v3):

- Single orbital (``norb_orig = 1``). Sublattice folding is supported
  for arbitrary ``SubShape`` that divides ``CellShape`` in every
  direction; ``SubShape`` unspecified defaults to ``CellShape`` (i.e.
  fully folded), matching ``uhfk.py:_init_lattice``.
- **Sz-fixed UHF (2Sz = 0)** → AntiParallel path (v1/v2/v2.1 behavior
  unchanged; consumes ``orbitalidx.def`` 3- or 4-column format).
- **Sz-fixed 2Sz ≠ 0 (A) and Sz-free non-mixed (B)** → General path
  (v3, consumes ``orbitalidx_general.def`` 6-column format). The CLI
  auto-dispatches on ``(is_antiparallel_metadata, orbitalidx_format)``;
  the user must generate ``orbitalidx_general.def`` via StdFace (set
  ``2Sz`` to a non-zero value in ``stan.in`` for the A case).
- Same-spin pair components (``F[up, up]``, ``F[down, down]``) are
  supported for spin-imbalanced Slater states via same-spin excess pair
  emission at canonical ``(k, partner(k))`` blocks.
- Sz-free with mixed block (``column_spin = -1``, SOC / spin-orbital
  mode) is **out of scope in v3**; deferred to v3.1.
- PBC and APBC both supported deterministically. The bridge uses the
  positive-Bloch amplitude convention consistent with H-wave's
  negative-gauge APBC transformation (``tilde_c_r =
  exp(-i theta r / L_phys) c_r``) plus the tilde-side positive-Bloch
  Fourier ``c_R = (1/sqrt(N_folded)) sum_k c_k exp(+i k R)`` implied by
  ``np.fft.ifftn(..., norm='forward')`` on the Hamiltonian side. For
  ``SubShape = [1, 1, 1]`` the (k, -k) time-reversal pair sum
  symmetrises the plane-wave factor and the numerical density is
  independent of the sign choice; for ``SubShape > [1, 1, 1]`` only the
  positive-Bloch form here reproduces H-wave's ``greenone.dat`` (verified
  to 1e-14 for the SubShape=[2,1,1] APBC L=8 fixture).
- AntiParallel pair form: (k, -k) time-reversal pairing over
  ``(k_row, local_band)``. General path (v3): canonical
  ``(k, partner(k))`` blocks with cross + same-spin excess pair
  emission; occupation distributions that violate the §3.2
  pair-closure conditions (same-spin excess imbalance across a canonical
  block, or self-pair with odd excess) are rejected with a clear error
  from ``validate_general_prerequisites``.
- T=0 Slater projection; finite-T SCF with fractional occupations near
  the Fermi level is rejected (rerun with smaller ``T``).
- Aggregated ``params[idx]`` carry a small uniform rank-lift noise
  (default amplitude ``1e-8``, ``--epsilon-noise`` to override) so that
  mVMC's Pfaffian Slater evaluation does not become singular when F is
  built from a single (k, -k) shell. The noise is complex for
  ``ComplexType 1`` orbitalidx and real-only for ``ComplexType 0``.
  Mirrors the same trick mVMC's own ComplexUHF uses
  (``mVMC-1.4.0/src/ComplexUHF/output.c:274``).

Scope (v3.1)
^^^^^^^^^^^^

- **Sz-free mixed block (SOC, ``enable_spin_orbital = true``)** →
  General-SOC path. Consumes ``orbitalidx_general.def`` (6-column) and
  emits Sz-non-conserving ``F[up_i, down_j]`` / ``F[down_i, up_j]``
  entries. Supports Zeeman, Rashba, Dresselhaus and general
  σ_x/σ_y 1-body couplings.
- Bridge validates ``eigen.npz["twist_offset"]`` against the
  canonicalized ``BoundaryCondition`` from ``input.toml`` to catch
  stale input/eigen pairings.
- Bridge additionally emits ``trans.def`` from H-wave's ``Transfer.dat``
  under SOC (see below) — mVMC's ``vmcdry.out`` cannot preserve
  spin-off-diagonal Rashba transfer entries.

Scope (v3.2)
^^^^^^^^^^^^

- **SOC + APBC** is now supported end-to-end. The trans.def emitter
  threads ``boundary_theta`` from ``BoundaryCondition``: rows whose
  ``(R_x, R_y, R_z)`` crosses a boundary along an APBC direction pick
  up the physical wrap-phase (``exp(i theta_d)`` for a positive-``R``
  crossing, ``exp(-i theta_d)`` for a negative-``R`` crossing; ``theta_d
  = pi`` in APBC directions). Like the base SOC sign convention
  (see the "Sign convention" note below), the wrap-phase convention
  is **empirically pinned** via E2E verification in
  ``tests/validation/uhfk_mvmc_pairproduct/case_soc_rashba_2d_nosub_apbc``
  (mVMC ⟨H⟩ within 0.03% of H-wave UHF); a first-principles derivation
  from H-wave's internal conventions does not go through cleanly on the
  UHFk path — see ``tools/_uhfk_to_mvmc/trans_emit.py`` module docstring.
- **SOC + SubShape > [1, 1, 1]** was deferred at v3.4 (Codex v3.4
  Rev.2 finding: the dual-A density gate could not validate the
  shipping A/F because the reference A dropped ``sub_offset`` from
  the plane-wave phase). v3.5 restores this path via a gauge-lifted
  single-A density check that validates the SHIPPING A directly —
  see "Scope (v3.5)" below.
- **SOC + APBC + SubShape > [1, 1, 1]** triple combination remains
  deferred beyond v3.5 (Rev.1 finding 2: no E2E fixture; the composed
  phase path from SOC + APBC and SOC + SubShape has not been
  independently validated). The CLI fail-fasts pre-dispatch when
  ``enable_spin_orbital = true``, an antiperiodic ``BoundaryCondition``
  entry, and ``SubShape > [1, 1, 1]`` are all set.
- The v3.1 SOC ``trans.def`` sign convention remains **empirically
  pinned**: an earlier attempt to derive it from H-wave's ``epsilon_k``
  swap composed with mVMC's ``H = -sum trans`` was found not to apply
  to the UHFk path (``uhfk.py`` does not perform the same
  ``epsilon_k[orb2, orb1]`` swap that ``sc.py`` does). The convention
  that ships is re-verified against ComplexUHF at 4.4e-8% agreement on
  ``case_soc_rashba_2d_nosub``. See ``tools/_uhfk_to_mvmc/trans_emit.py``
  module docstring for the empirical basis and the aborted derivation
  attempt.

Scope (v3.5)
^^^^^^^^^^^^

- **SOC + SubShape > [1, 1, 1]** is now shippable in v3.5 via the
  gauge-lifted density check
  ``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
  at 1e-10 element-wise tolerance. The check LIFTS H-wave's
  ``green_sublattice`` (folded-Bloch basis) into the physical basis via
  ``gauge_lift`` and compares element-wise to the SHIPPING
  ``conj(A) @ A.T`` — the same A that is emitted to mVMC — closing the
  v3.4 dual-A hole. The gauge transform relates the shipping A's phase
  ``exp(-i k · (folded_cell + sub_offset))`` to
  ``green_sublattice``'s folded-Bloch storage; the derivation is in
  ``docs/superpowers/specs/2026-07-05-uhfk-mvmc-pairproduct-general-v35-design.md``
  §2-3 (note: ``docs/superpowers/`` is gitignored and ships out-of-tree,
  so the spec is not vendored with the release). The v3.4
  ``_soc_reference_convention`` escape hatch has been removed; only the
  shipping-convention A remains in the codebase.
- **SOC + APBC + SubShape > [1, 1, 1]** triple combination remains
  rejected (unvalidated composed phase path; no E2E fixture — see
  "Scope (v3.2)" above).
- Required CLI flags under SOC + SubShape > [1, 1, 1]: ``--transfer``
  (to read H-wave's ``Transfer.dat``), ``--emit-trans`` (to write
  mVMC's ``trans.def`` with Rashba off-diagonal spin preserved), and
  ``--emit-orbitalidx`` (to bypass StdFace's class merging, which
  cannot express Sz-non-conserving classes under a folded lattice —
  see the v3.2 spec §1 rationale).
- Empirical E2E: ``case_soc_rashba_2d_sub`` mVMC ⟨H⟩ vs H-wave
  ``Energy_Total`` at 0.22% delta (delta -0.055 out of 25.10) with the
  gauge-lifted density gate clean at 1e-10.

Scope (v3.6)
^^^^^^^^^^^^

- **SOC + single-direction APBC + SubShape > [1, 1, 1]** is now shippable
  in v3.6. The bridge's ``build_slater_orbitals`` composes the sub_offset
  gauge (``exp(-i k_folded · (folded_cell + sub_offset))``) with the APBC
  twist (``exp(-i theta · r_phys / L_phys)``) so the shipping A carries
  both phases consistently; ``gauge_lift`` in
  ``tools/_uhfk_to_mvmc/density_check.py`` receives ``boundary_theta``
  and applies the same composed transform when lifting
  ``green_sublattice`` back to the physical basis. The shipping density
  gate ``compare_against_green_sublattice(...,
  is_soc_sublattice_mode=True)`` passes at 1e-10 on the v3.6 shipping
  fixture ``case_soc_rashba_2d_sub_apbc``
  (``CellShape = [6, 4, 1]``, ``SubShape = [2, 2, 1]``,
  ``BoundaryCondition = ["antiperiodic", "periodic", "periodic"]``,
  ``enable_spin_orbital = true``, Rashba ``alpha = 0.5``, ``U = 2``,
  ``Ncond = 8``).

- **Multi-direction APBC (n_apbc_dirs >= 2) + SOC + SubShape > [1, 1, 1]**
  remains rejected pre-dispatch. The exact CLI reject message is::

      ERROR: enable_spin_orbital = true + multi-direction APBC
      (n_apbc_dirs=<count>) + SubShape > [1, 1, 1] is deferred to v3.7.
      Single-direction APBC + SOC + SubShape > 1 is supported in v3.6 as
      of case_soc_rashba_2d_sub_apbc.

  The Phase 6 dispatch predicate that fires this reject::

      n_apbc_dirs = sum(1 for t in theta if abs(t - pi) < 1e-12
                                       or abs(t + pi) < 1e-12)
      if is_soc_mode and n_apbc_dirs > 1 and any(s != 1 for s in sub_shape):
          return 2

  A first-principles derivation of the composed twist gauge on
  multi-direction APBC + SubShape > 1 has not been validated against an
  E2E fixture (no ``case_soc_rashba_2d_sub_apbc_apbc`` exists in the v3.6
  tree); the reject exists to fail-fast rather than silently produce a
  wrong ``trans.def`` / shipping A.

- **v3.6 seven-gate contract** (all seven records PASS on fresh workspace
  for ``case_soc_rashba_2d_sub_apbc``):

  1. **G0-writer-check**: emitted-F rank-lift-noise-off writer path
     matches the aggregated (mapping, params) at 1e-10.
  2. **G1**: shipping ``build_slater_orbitals`` density matches the
     ``gauge_lift``-lifted green_sublattice at 1e-10.
  3. **G2a-emitted-F**: emitted-F projector density matches ComplexUHF
     one-body Green at 1e-6.
  4. **G2a-in-memory-A**: in-memory shipping A density matches
     ComplexUHF at 1e-6.
  5. **G2b**: gauge-lifted green_sublattice matches ComplexUHF at 1e-6.
  6. **G3**: mVMC ⟨H⟩ vs H-wave ``Energy_Total`` relative delta ≤ 1 %.
  7. **G4**: composite element ``(i_c, s_c, j_c, t_c)`` from
     ``composite_element.json`` survives the current SCF run and every
     mutation in the M-gauge-1..5 + M-ship-1..5 matrix trips above
     ``T_M = max(1e-5, 0.10 * |G_base|)`` per spec §4.4.

  The gates run from
  ``tests/validation/uhfk_mvmc_pairproduct/run.sh case_soc_rashba_2d_sub_apbc``;
  each gate emits an anchored PASS line (``^GATE_NAME PASS mode=... ``)
  verified by ``awk 'index($0, p) == 1'``.

- **v3.6 hardening pass (Codex adversarial-review 2026-07-12)**. After
  the initial Phase 6 seven-gate PASS, four rounds of adversarial
  review surfaced a chain of findings that were subsequently addressed:

  * **G1 / G2a / G2b real comparisons**: the initial dispatchers were
    stubs that emitted PASS with ``max_abs_delta=0.0`` without
    performing the claimed numeric comparison. Replaced by real
    dispatchers in ``compare.py`` that read the workspace's H-wave
    outputs, build the shipping A via ``build_slater_orbitals``, lift
    ``green_sublattice`` via ``gauge_lift`` with the workspace's own
    ``boundary_theta``, and compare against a strictly-parsed
    ComplexUHF ``zvo_UHF_cisajs.dat``. Missing / truncated / malformed
    / duplicate / out-of-range / non-finite ComplexUHF entries all
    fail closed via a new ``ComplexUHFParseError``.

  * **Cross-solver ComplexUHF seeding**: the H-wave broken-symmetry
    minimum is not the same as ComplexUHF's random-init default under
    Rashba + APBC; two independent SCFs converged to different valid
    minima with 4.76e-2 element-wise density disagreement. Fix:
    ``scripts/seed_complexuhf_from_hwave.py`` writes H-wave's shipping
    A density into ComplexUHF's ``initial.def`` (5-line header for the
    ``IgnoreLinesInDef=5`` convention) with a small Hermitian
    ``perturb-scale`` (1e-6 for this fixture) so ComplexUHF actually
    iterates (``run.sh`` asserts the SCF ran >= 1 step via awk-parse
    of ``uhf.log``) but stays inside H-wave's basin. Post-seeding
    ComplexUHF runs 9 SCF steps and converges to
    ``Energy_Total = -25.3717166`` matching H-wave to 10 digits.

  * **Snapshot-workspace rejection**: the snapshot guard resolved
    ``tests/data`` relative to the process CWD, so invocation from
    outside the repo silently bypassed the check. Fixed by anchoring
    the tests/data root at the module's own directory
    (``Path(__file__).resolve().parents[3]``); the guard now fails
    closed if the anchored root is missing.

  * **G4 shadow-copy drift**: the topology guard's
    ``_build_A_ship_mutated`` reimplemented the SOC branch of
    ``build_slater_orbitals`` inline and never proved its baseline
    matched the canonical kernel. A drift in ``build_slater_orbitals``
    could bypass G4. Fixed with a mandatory 1e-10 equality assertion
    against the canonical kernel BEFORE running the mutation matrix.

  * **Loader-injection env sanitization**: ``run.sh`` inherited the
    caller's ``LD_LIBRARY_PATH`` / ``LD_PRELOAD`` / ``LD_AUDIT``
    unchanged into ``vmcdry.out`` / ``vmc.out`` / ``UHF`` child
    processes. Fixed by (a) top-level ``unset`` of those vars +
    ``BASH_ENV`` / ``ENV``, (b) ``trap - DEBUG ERR RETURN EXIT`` to
    clear inherited traps, (c) ``run_native()`` wrapper using
    ``env -u LD_PRELOAD -u LD_AUDIT [-u LD_LIBRARY_PATH|
    LD_LIBRARY_PATH=<validated>]`` for per-command sanitization
    around every native solver invocation, and (d)
    ``MVMC_LD_LIBRARY_PATH`` validation (absolute path, owned by
    current user, not world-writable, single directory only).

  Regression tests pin each finding: 5 tests for loader-env
  sanitization (`tests/test_run_sh_loader_env_sanitize.py`), 6 tests
  for the strict ComplexUHF parser
  (`tests/test_uhfk_mvmc_pairproduct_compare_wiring.py`), 3 tests for
  the CWD-independent snapshot guard
  (`tests/test_snapshot_rejection_guard_v36.py`).

Workflow
^^^^^^^^

1. Generate mVMC inputs (``orbitalidx.def`` etc.) via StdFace, using
   ``CalcMode = 2`` and ``Lsub`` for the desired sublattice translation
   symmetry. For APBC set ``phase0 = 180.0``.
2. Run H-wave UHFk SCF, requesting both ``eigen.npz`` and the new
   ``occupation.npz`` under ``[file.output]``::

       [file.output]
         path_to_output = "output"
         eigen          = "eigen.npz"
         green          = "green.npz"
         occupation     = "occupation.npz"
         onebodyg       = "greenone.dat"

3. Run the bridge::

       python tools/uhfk_to_mvmc.py \
           --input        input.toml \
           --eigen        output/eigen.npz \
           --occupation   output/occupation.npz \
           --geometry     geometry_uhf.dat \
           --orbitalidx   mvmc_inputs/orbitalidx.def \
           --output       mvmc_inputs/zqp_orbital_uhfk.dat \
           --check-density \
           --onebodyg-uhf output/greenone.dat

4. Add ``InOrbital zqp_orbital_uhfk.dat`` (or
   ``InOrbitalAntiParallel zqp_orbital_uhfk.dat``, or
   ``InOrbitalGeneral zqp_orbital_uhfk.dat`` for the v3 General path)
   to mVMC's ``namelist.def``; mVMC will initialize PairProduct
   parameters from this file.

Dispatch (v3)
^^^^^^^^^^^^^

The CLI parses ``--orbitalidx`` first and picks the code path from
``(is_antiparallel_metadata, orbitalidx_format)``:

- ``(True, antiparallel)`` — v2.1 AntiParallel path (v1/v2/v2.1
  behavior, unchanged).
- ``(True, general)`` — forced-General branch. If the occupied set
  also satisfies the v2.1 ``(k_row, local_band)`` pair-closure,
  ``F[up, down]`` reproduces v2.1's F within 1e-12; otherwise a
  ``WARNING`` is logged and the emitted General Slater is a valid
  ``InOrbitalGeneral`` state but not v2.1-reproducible.
- ``(False, general)`` — v3 General path (A + B scope).
- ``(False, antiparallel)`` — rejected; regenerate
  ``orbitalidx_general.def`` via StdFace with a non-zero ``2Sz`` (A) or
  Zeeman-driven Sz-free (B) setup.

``is_antiparallel_metadata`` requires ALL of: ``2Sz`` explicitly set to
0 in ``input.toml``, ``N_up == N_down``, ``column_spin ∈ {0, 1}``,
``column_mu_group`` has exactly 2 unique values, and column_spin↔mu_group
is bijective.

Dispatch (v3.1, 6-case)
^^^^^^^^^^^^^^^^^^^^^^^

The CLI computes ``is_soc_mode = toml_param.get("enable_spin_orbital",
False)`` after parsing ``input.toml``, then routes on
``(is_antiparallel_metadata, orbitalidx_format, is_soc_mode)``:

- ``(True, antiparallel, False)`` — v2.1 AntiParallel (unchanged).
- ``(True, general, False)`` — v3 forced-General.
- ``(False, general, False)`` — v3 General (A/B).
- ``(False, antiparallel, False)`` — rejected.
- ``(\*, general, True)`` — **v3.1 General-SOC**.
- ``(\*, antiparallel, True)`` — rejected (SOC requires 6-column).

BoundaryCondition contract (v3.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bridge canonicalizes ``BoundaryCondition`` through H-wave's shared
``normalize_boundary_condition`` helper. Accepted forms:

- PBC: ``"p"``, ``"periodic"`` (case-insensitive, whitespace-stripped)
- APBC: ``"ap"``, ``"antiperiodic"`` (same)

Every other string raises a ``ValueError`` before dispatch; no raw
fallback exists. Omitting the key defaults to all-PBC.

Bridge trans.def emitter (v3.1 SOC only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under SOC, mVMC's ``vmcdry.out`` builds ``trans.def`` via
``StdFace_Hopping`` which is strictly spin-diagonal and drops Rashba
``s != t`` transfer entries. To close the gap, the bridge reads
H-wave's ``Transfer.dat`` (Wannier90-like format,
``iWan = 2 * a_phys + spin + 1``) and emits mVMC's ``trans.def`` with
``(i, s, j, t, re, im)`` rows preserving Rashba off-diagonal spin.

Sign convention (empirically pinned):

- ``s == t`` (NN hopping): ``trans = -val_hwave`` (matches vmcdry's
  own sign flip).
- ``s != t`` (Rashba): ``trans = +val_hwave``.

The mixed convention ``(sign_diag = -1, sign_offdiag = +1)`` is
**empirically pinned** via ComplexUHF verification at 4.4e-8% agreement
on ``case_soc_rashba_2d_nosub``. The mVMC convention is
``H = -Σ trans c†c``, but the exact interaction between the
source-target index convention and H-wave's k-space Hamiltonian is
**not derivable from first principles alone** in this bridge — see
``tools/_uhfk_to_mvmc/trans_emit.py`` module docstring for the empirical
basis and the aborted "derived from H-wave's ``sc.py`` swap" story
(``sc.py`` performs an ``epsilon_k[orb2, orb1]`` swap that ``uhfk.py``
does not perform, so the swap-derivation does not compose through).
A uniform flip (``-val`` on both) produces a 42.58% ⟨H⟩ delta on the
same fixture. **Re-verify against ``case_soc_rashba_2d_nosub`` E2E after
any mVMC or H-wave version bump** — the mVMC-side mechanism cites
``locgrn_fsz.c:128`` which mVMC's own author marked ``//TBC``.

New CLI flags (SOC-only): ``--transfer <path>`` and
``--emit-trans <path>``.

Out of scope for v3.5
^^^^^^^^^^^^^^^^^^^^^

- SOC + APBC + ``SubShape > [1, 1, 1]`` triple combination is deferred
  beyond v3.5 (Rev.1 finding 2: no E2E fixture; the composed phase path
  from SOC + APBC and SOC + SubShape has not been independently
  validated). The two-way subsets are covered:
  ``case_soc_rashba_2d_nosub_apbc`` for SOC + APBC (0.03% delta) and
  ``case_soc_rashba_2d_sub`` for SOC + SubShape (0.22% delta).
- 2-body Sz-non-conserving interactions (spin-flip Coulomb,
  Hund coupling, pair hopping). CoulombIntra (on-site U) remains
  the only supported 2-body term.

Class-consistency check (v3 General)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the General path, ``aggregate_general_orbital_params`` verifies
that every ``orbitalidx_general.def`` class's signed F entries agree
within ``class_consistency_tol`` (default 1e-8) **before** averaging.
Otherwise ``ClassInconsistencyError`` is raised with the offending
class idx and observed max residual — this guards against
StdFace-generated classes that assume symmetries the Slater state does
not respect (e.g., a symmetric Hamiltonian yielding a symmetry-broken
UHF ground state).

Density-check (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--check-density`` reconstructs the 1-body density matrix from the
(k, -k) pair construction and compares element-wise against H-wave's
physical-basis ``greenone.dat`` output (tolerance 1e-10). A mismatch is
fatal and indicates a bug in either H-wave's APBC handling, the bridge,
or the geometry assumption.

Under SOC + SubShape > [1, 1, 1] the density check switches to
``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
at 1e-10 (H-wave's ``greenone.dat`` fold path is known-buggy on this
combination; ``green_sublattice`` is the source of truth). The check
LIFTS H-wave's ``green_sublattice`` into the physical basis via
``gauge_lift`` and compares element-wise against the SHIPPING
``conj(A) @ A.T`` — the same A that is emitted to mVMC — so a
regression in the shipping-A path cannot silently pass. This restores
the density gate that was closed in v3.4 (dual-A hole) via the gauge
lift derived in the v3.5 spec (§2-3); the v3.4
``_soc_reference_convention`` escape hatch has been removed.

Exit codes
^^^^^^^^^^

- ``0`` — success.
- ``2`` — a fail-fast guard rejected the input (out-of-scope mode,
  Sz-free SCF, fractional finite-T occupation, ``orbitalidx.def`` /
  geometry / boundary mismatch, etc.). The stderr message names the
  failed check.
- ``3`` — ``--check-density`` detected a density mismatch beyond
  tolerance.
