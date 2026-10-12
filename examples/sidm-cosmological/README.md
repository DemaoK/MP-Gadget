# Cosmological SIDM validation

This port starts from the isolated solver at `MP-Gadget-SIDM:a5474a7f`.
The cosmological `SIDM` branch incorporates `origin/master:4e055379` through
merge `a45b2156`; the branches had diverged, so a fast-forward was not possible.

The direct solver scatters live type 1 particles with one fixed equal DM mass.
Gas and other species retain their native gravity and hydrodynamic evolution.
Unequal type 1 masses in zoom initial conditions are unsupported. Start from a
fresh IC or a synchronized, flushed checkpoint written by this implementation;
these tests make no compatibility claim for legacy SIDM checkpoints.

Positions and neighbor support are comoving. Stored velocities are canonical
`u = a v_pec`. The cross section per unit mass converts as
`sigma_code = sigma_cgs h UnitMass / UnitLength^2`. A source accumulates exposure
`E = integral(dt/a^4)` and relative displacement uses
`D = integral(dt/a^2)`. Velocity-dependent scattering evaluates the physical
pair speed at the collision endpoint: `w_internal = a w_physical` in code
velocity units. Cached cosmological integrals preserve short integer intervals.

Use `SplitGravityTimestepsOn=1`, `TreeGravOn=1` and `ForceEqualTimesteps=0`.
Gravity tolerances and default softening are unchanged. The module defaults are
directed probability `p=.1` and transport fraction `C=.1`; the smoke harness
explicitly selects the previously tested optional `p=.5`, `C=.2` values.
The transport envelope uses one global DM bulk frame and twice the global
maximum canonical speed. Separate bulk flows or a fast population can therefore
shorten source intervals in otherwise quiet regions.
Off-grid visits also require re-alignment to short power-of-two bins before
longer intervals become available. Deadline selection scans the DM population
and uses global reductions; these costs remain even when few sources are due.

The validation Python environment requires NumPy and bigfile. From the repository
root, build the three variants and run the focused suite:

```bash
bash examples/sidm-cosmological/build.sh /path/to/checkout/with/installed/depends 4
python3 examples/sidm-cosmological/validate.py \
  --output "$PWD/build/sidm-cosmological/smoke" \
  --genic "$PWD/build/sidm-cosmological/MP-GenIC" \
  --release "$PWD/build/sidm-cosmological/MP-Gadget-release" \
  --debug "$PWD/build/sidm-cosmological/MP-Gadget-debug" \
  --without-sidm "$PWD/build/sidm-cosmological/MP-Gadget-without-sidm"
```

The build helper copies installed dependency outputs so it cannot rebuild into
the source checkout. It uses the normal `Options.mk.example` settings, including
`-O3 -ffast-math`. Release adds only `-DSIDM`; debug adds
`-DSIDM -DSIDMDEBUG`. SIDM does not override the compiler's floating-point options.
The helper leaves the local build configured for release SIDM; generated
artifacts live under ignored `build/sidm-cosmological`.

The harness generates 4,096 DM particles, then 4,096 DM plus 4,096 gas, with
native MP-GenIC and the included tabulated spectrum. The smoke uses the cubic
SPH kernel so the native initial smoothing-length guess fits the small gas
population. It uses two MPI ranks and
two OpenMP threads, so four physical cores suffice. `--oversubscribe` accommodates
an allocation advertised as one MPI task without increasing the requested
thread count. Each solver invocation has a five minute timeout and allocates
1,000 MB per node. A deliberately large `sigma0=10000000` and smoke-specific `MaxSizeTimestep=.01`
exercise collisions and source-only events quickly; these settings are validation inputs.

Checks cover finite positive physical fields, species counts and identities,
debug collision probability bounds, release/debug agreement, a restart from the
middle flushed snapshot, and agreement between runtime-disabled and compiled-out
SIDM. Two binaries restarting from the same flushed checkpoint must agree;
their initial resumed collision solve must have zero exposure. Uninterrupted
and resumed stochastic trajectories can differ: source limits accumulated
during sequential collisions are recomputed from the saved endpoint, and native
velocity/thermodynamic snapshot fields also round to float. The report records
their aggregate event counts, DM kinetic energy/momentum and maximum support
difference without requiring particlewise trajectory agreement. These tiny cases establish integration behavior;
they do not establish scientific convergence or realistic computational cost.

DM floating fields use `2e-12` relative/absolute comparison tolerances, with exact
IDs, masses and clock integers. Repeated execution of the same two-thread hydro
binary showed native gas differences of `1.9e-7` in position, `7.6e-6` in velocity,
`5.6e-5` in internal energy (maximum scaled difference `3.9e-5`), and one float ULP
in density. The harness therefore uses separate gas bounds: position
`rtol=2e-10, atol=2e-7`, velocity `rtol=1e-5, atol=1e-6`, internal energy
`rtol=1e-4, atol=1e-7`, and density `rtol=2e-7`. It records observed differences
and bounds in the report; masses and clocks retain exact checks.
The compiled-out native hydro repeat also varied (`1.3e-9` in gas position and
`6.0e-8` in internal energy), with DM fields unchanged; its report is
[`native-hydro-repeat.json`](../../build/sidm-cosmological/native-hydro-repeat.json).

The harness also compares `SIDMOn=1, sigma0=0` against native disabled output.
It requires zero collisions and integrated collision probability, and verifies debug counters
`source_transport`, `gravity_transport` and `deadline_tightened` remain zero, so
transport changes cannot hide native-phase sequencing errors. The zero-cross-section
case has no source-only epochs; the collision smoke separately exercises off-grid
phases. To run only this check
against an existing successful suite, add `--zero-only /path/to/previous/suite`
to the command above and select a new output directory. This reuses its ICs and
disabled outputs without repeating the other simulations.

Additional production-module regressions can be run after building SIDM:

```bash
make -C libgadget -j4 .objs/test_sidm_direct .objs/test_sidm_rate \
  .objs/test_sidm_distributed .objs/test_sidm_local .objs/test_sidm_hierarchical
OMP_NUM_THREADS=2 SIDM_TEST_THREADS=2 mpirun --oversubscribe --bind-to none -np 2 \
  libgadget/.objs/test_sidm_distributed
SIDM_TEST_EMPTY_OWNER=1 OMP_NUM_THREADS=2 SIDM_TEST_THREADS=2 \
  mpirun --oversubscribe --bind-to none -np 2 libgadget/.objs/test_sidm_distributed
OMP_NUM_THREADS=2 mpirun --oversubscribe --bind-to none -np 2 libgadget/.objs/test_sidm_local
OMP_NUM_THREADS=2 mpirun --oversubscribe --bind-to none -np 2 libgadget/.objs/test_sidm_hierarchical
```

Distributed tests independently check periodic nearest neighbors, nonunity
little-h unit conversion, endpoint velocity-dependent cross sections, inactive
receivers, conservation and export retries with interspersed gas. The empty-owner
variant tests a genuinely empty MPI rank. Local tests compare matter-only
cosmological integrals with analytic expressions and check mixed-species clocks,
transport limits, common velocity boosts and PM flushes. A transport follow-up
exercises an actual gravity-bin shortening and receiver deadline tightening,
and an opt-in `--expect-resolution-guard` mode verifies that a genuinely
insufficient minimum-step budget aborts. Hierarchy tests check
native SIDM-only activity, canonical free drift around a prescribed off-grid
elastic collision, and the highest legal gravity bin with empty ranks. The
standalone direct test additionally compares the full MPI stochastic collision
schedule against an independent serial oracle.

## Normal compiler validation on 2026-10-12

All three binaries were rebuilt from `Options.mk.example` with its normal
`-O3 -ffast-math` settings. Only the SIDM feature macros differ between variants.
The full current harness passed for DM and gas/DM, including restarts,
release/debug agreement, disabled/compiled-out agreement and zero scattering.
It recorded 248 source-only epochs and 85,490 collisions for DM, and 159 epochs
and 65,583 collisions for gas/DM. Existing comparison tolerances were unchanged.

Focused rate, direct-collision, distributed, empty-owner, transport, hierarchy
and native timeline/tree tests passed. The rate test checks NaN/infinity behavior
only when `__FINITE_MATH_ONLY__` is disabled; all finite-input quadrature and
normalization checks run with the selected compiler settings.

Current evidence: [full smoke](../../build/sidm-cosmological/normal-smoke/validation.json),
[focused tests](../../build/sidm-cosmological/normal-tests/tests.json),
[debug direct oracle](../../build/sidm-cosmological/normal-tests/direct-debug.json),
[binary hashes](../../build/sidm-cosmological/binaries.json), and
[build record](../../build/sidm-cosmological/build-record.json).

## Initial integration validation on 2026-10-12

These results and the two audit reports describe the original `-O2` builds
without fast math, before the compiler override was removed. They are retained
as historical evidence; the normal-compiler validation is recorded separately.

The focused suite passed with two MPI ranks and two OpenMP threads. The native
force-tree unit test used its intended single-rank fixture, also with two threads.

| Case | Source-only epochs | Elastic collisions | Full and resumed release/debug | Runtime-disabled/compiled-out |
| --- | ---: | ---: | --- | --- |
| 4,096 DM | 248 | 85,490 | Passed | Passed |
| 4,096 DM + 4,096 gas | 159 | 65,583 | Passed within the documented gas bounds | Passed within the documented gas bounds |

Both cases also rejected enabling SIDM in a compiled-out binary and passed
flushed-clock, mass, identity and finite/positive-field checks. Resumed runs
started with zero collision exposure. The DM maximum relative support
difference between uninterrupted and restarted runs was `0.00396`; the mixed
case was `0.00375`. These are informational differences under the restart
contract described above.

The additional zero-cross-section check passed for DM and mixed hydro using the
same binaries, establishing native-phase equivalence. Both reported zero events, zero source-only epochs, and zero
source/gravity transport limits or tightened deadlines. DM endpoint fields
matched native disabled output exactly. Detailed gas differences and the
comparison bounds are recorded in
[`zero-cross-section/validation.json`](../../build/sidm-cosmological/zero-cross-section/validation.json)
and [`zero-cross-section.log`](../../build/sidm-cosmological/zero-cross-section.log).

The initial smoke above used the harness before this extra comparison was added.
The follow-up ran through `--zero-only`, reusing the saved native-disabled outputs
and unchanged binaries. At that stage the combined default harness path had not
been rerun end to end; the normal-compiler validation above now covers it.
The archived build record preserves the original build hashes and records that
follow-up invocation separately under `post_build_validation`.

Generated evidence is in
[`build/sidm-cosmological/validated/validation.json`](../../build/sidm-cosmological/validated/validation.json),
[`validated.log`](../../build/sidm-cosmological/validated.log),
[`tests.json`](../../build/sidm-cosmological/tests.json),
[`transport-followup.json`](../../build/sidm-cosmological/transport-followup.json),
[`initial-build/binaries.json`](../../build/sidm-cosmological/initial-build/binaries.json), and
[`initial-build/build-record.json`](../../build/sidm-cosmological/initial-build/build-record.json).
The independent integral checks are recorded in
[`timefactor-check.log`](../../build/sidm-cosmological/timefactor-check.log).
Read-only audit reports are in
[`audits/claude-round1.md`](../../audits/claude-round1.md) and
[`audits/claude-round2.md`](../../audits/claude-round2.md).
