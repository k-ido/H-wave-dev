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

Exit codes
^^^^^^^^^^

- ``0`` — success.
- ``2`` — a fail-fast guard rejected the input (out-of-scope mode,
  Sz-free SCF, fractional finite-T occupation, ``orbitalidx.def`` /
  geometry / boundary mismatch, etc.). The stderr message names the
  failed check.
- ``3`` — ``--check-density`` detected a density mismatch beyond
  tolerance.
