.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script ``tools/uhfk_to_mvmc.py`` converts an H-wave UHFk SCF result
into an mVMC ``InOrbital`` / ``InOrbitalAntiParallel`` initial wave
function file (``zqp_orbital_uhfk.dat``), so that mVMC's PairProduct
state can be initialized from the H-wave Slater determinant.

Scope (v1):

- Single orbital (``norb_orig = 1``, ``SubShape = [1, 1, 1]``).
- Sz-fixed UHF only (``2Sz = 0`` with ``N_up = N_down``).
- PBC and APBC both supported deterministically. The bridge uses the
  negative-Bloch convention so the per-spin density reconstruction
  matches H-wave's ``np.fft.fftn(norm='forward')`` k-to-r kernel under
  both boundary conditions.
- AntiParallel pair form only. Magnetic / asymmetric occupations that
  violate (k, -k) pair-closure are rejected with a clear error.
- T=0 Slater projection; finite-T SCF with fractional occupations near
  the Fermi level is rejected (rerun with smaller ``T``).
- Aggregated ``params[idx]`` carry a small uniform rank-lift noise
  (default amplitude ``1e-6``, ``--epsilon-noise`` to override) so that
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
   ``InOrbitalAntiParallel zqp_orbital_uhfk.dat``) to mVMC's
   ``namelist.def``; mVMC will initialize PairProduct parameters from
   this file.

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
