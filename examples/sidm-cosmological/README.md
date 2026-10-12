# Cosmological SIDM

SIDM scatters live type 1 particles with one fixed equal DM mass while gas and
other species retain native gravity and hydrodynamics. Enable it at compile
time with `OPT += -DSIDM` in `Options.mk`, then set `SIDMOn=1` in the runtime
parameter file. Add `-DSIDMDEBUG` for ownership/conservation audits, counters
and detailed timers. Release and debug use the same collision operator.

## Build and smoke test

The normal `Options.mk.example` compiler settings apply, including
`-fopenmp -O3 -g -Wall -ffast-math`. SIDM does not override floating-point
options. The helper below builds compiled-out, debug and release variants;
only `-DSIDM` and `-DSIDMDEBUG` differ between them. It copies installed
`depends/include` and `depends/lib` from the supplied checkout and leaves the
local `Options.mk` configured for release SIDM.

From the repository root, with Python 3, NumPy and bigfile available:

```bash
bash examples/sidm-cosmological/build.sh /path/to/checkout/with/installed/depends 4
python3 examples/sidm-cosmological/validate.py \
  --output "$PWD/build/sidm-cosmological/smoke" \
  --genic "$PWD/build/sidm-cosmological/MP-GenIC" \
  --release "$PWD/build/sidm-cosmological/MP-Gadget-release" \
  --debug "$PWD/build/sidm-cosmological/MP-Gadget-debug" \
  --without-sidm "$PWD/build/sidm-cosmological/MP-Gadget-without-sidm"
```

The harness uses native MP-GenIC and the included spectrum for 4,096 DM particles
and for 4,096 DM plus 4,096 gas. Runs use two MPI ranks, two OpenMP threads per
rank, a five-minute invocation timeout and `MaxMemSizePerNode=1000` MB. Four
physical cores suffice; `--oversubscribe` accommodates allocations advertised
as one MPI task. The small gas fixture uses the cubic SPH kernel so its native
initial smoothing-length estimate fits the available population.

The deliberately large `sigma0=10000000`, `MaxSizeTimestep=.01`, directed
probability `.5` and transport fraction `.2` exercise collisions and source-only
events quickly. These are smoke inputs; production defaults remain `.1` for
both probability and transport fraction. Generated binaries, configs, hashes
and reports live under ignored `build/sidm-cosmological`.

## Units and supported configurations

Positions and neighbor support are comoving; stored velocities are canonical
`u = a v_pec`. Cross section per unit mass converts as
`sigma_code = sigma_cgs h UnitMass / UnitLength^2`. Collision exposure is
`E = integral(dt/a^4)` and relative displacement uses `D = integral(dt/a^2)`.
Velocity-dependent scattering evaluates the endpoint speed using
`w_internal = a w_physical` in code velocity units; runtime `wTurn` is in km/s.
Cached cosmological integrals preserve short integer intervals.

Use `SplitGravityTimestepsOn=1`, `TreeGravOn=1`, and `ForceEqualTimesteps=0`.
SIDM requires double velocities, equal positive live type 1 masses, more DM
particles than `SIDMNeighbors` (default 38, maximum 128), and periodic neighbor
support smaller than half the box. Unequal-mass DM zoom ICs are unsupported.
Gravity tolerances and default softening are unchanged. Native
`MaxSizeTimestep` is honored; SIDM adds no separate global time cap.

| Runtime parameter | Default | Allowed range and purpose |
| --- | ---: | --- |
| `SIDMDirectedProbability` | `.1` | `(0,1]`; bounds each directed pair trial and sets the forward source exposure bound. |
| `SIDMTransportFraction` | `.1` | `(0,.25]`; source relative-displacement budget as a fraction of support, also used in the DM gravity transport criterion. |
| `SIDMMaxSupportChange` | `.25` | Greater than `SIDMTransportFraction` and less than `1`; maximum fractional support change over a nonzero source interval. |

Exceeding the support-change limit aborts. Restart the last synchronized
checkpoint with a smaller transport fraction; endpoint retries cannot recover
geometry skipped during the interval.

The transport envelope uses one global DM bulk frame and twice the global
maximum canonical speed. Distinct bulk flows or a fast population can shorten
source intervals in unrelated quiet regions. Off-grid visits can temporarily
require short power-of-two bins during re-alignment. Deadline selection also
scans the DM population and uses global reductions, even when few sources are
due. These tiny tests establish integration behavior, without establishing
scientific convergence or realistic computational cost.

## Code responsibilities

| Module | Responsibility |
| --- | --- |
| `sidm-local.c` | Own source clocks, select events, accumulate exposure, limit transport, and commit or flush visits. |
| `sidm-distributed.c` | Find periodic DM neighbors, convert physical units, and pass source intervals to the collision operator. |
| `sidm-direct.c` | Apply ordered equal-mass elastic collisions across MPI owners, using current endpoint velocities. |
| `sidm-rate.h`, `sidm-random.h` | Evaluate the kernel/cross section and deterministic random streams. |
| `timefac.c` | Supply `sidm_drift_factor` and `sidm_scatter_factor` cosmological integrals. |
| `run.c`, `timestep.c` | Integrate SIDM events with native split gravity; hydro advances on native events. |

The run loop initializes the local scheduler, obtains a visit with
`sidm_local_begin_active`, calls `sidm_distributed_force_intervals`, and commits
and ends the visit. The distributed adapter and standalone tests call
`sidm_direct_scatter` with a strided particle view, avoiding a second forwarding
interface.
Zero-exposure visits initialize support and rates without scattering. Inactive
receivers may change velocity while retaining their source clocks. Pairs above
the configured probability bound use multiple Bernoulli trials; this bounds each
trial without establishing converged interleaving with other neighbors.

## Restart and comparison contracts

Start from fresh ICs or a synchronized PM checkpoint written by this
implementation, with every source interval flushed before output. Legacy SIDM
checkpoint compatibility is untested. Release and debug restarting from the
same checkpoint must agree, and the first resumed solve has zero exposure.

An uninterrupted trajectory can differ from its restart. Forward limits retain
rates encountered during sequential collisions; restart recomputes rates from
the saved endpoint. Native velocity and thermodynamic snapshot fields also
round to float. The harness records event counts, DM kinetic energy/momentum,
and support differences as information rather than requiring particlewise
agreement between these trajectories.

The harness checks identities, masses, finite physical fields, flushed clocks,
debug probability bounds, release/debug agreement, and runtime-disabled versus
compiled-out agreement. DM floating fields use `rtol=atol=2e-12`; IDs, masses
and clock integers are exact. Repeated executions of the same threaded hydro
binary established a native gas reproducibility floor, so gas bounds are:

| Gas field | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| Position | `2e-10` | `2e-7` |
| Velocity | `1e-5` | `1e-6` |
| Internal energy | `1e-4` | `1e-7` |
| Density | `2e-7` | `0` |

Observed differences and bounds are saved in each report. The `sigma0=0` check
requires zero collisions, integrated collision probability, transport limits
and tightened deadlines. It establishes native-phase equivalence with no
source-only epochs; the collision smoke separately exercises off-grid phases.
To reuse an existing successful suite's ICs and disabled outputs for this check,
add `--zero-only /path/to/previous/suite` and select a new output directory.

## Validated normal compiler builds: 2026-10-12

The full harness passed with the normal `Options.mk.example` flags. Full and
restarted release/debug comparisons, runtime-disabled/compiled-out comparisons,
zero-scattering comparisons and compiled-out parameter rejection all passed.
After the API/readability cleanup, all seven run modes for both DM and mixed
hydro also matched the previous normal-compiler outputs within the same
comparison tolerances. Source-only epoch and collision counts were unchanged.

| Case | Source-only epochs | Elastic collisions |
| --- | ---: | ---: |
| 4,096 DM | 248 | 85,490 |
| 4,096 DM + 4,096 gas | 159 | 65,583 |

Focused tests passed for rate quadrature, the serial collision oracle,
distributed neighbors/unit conversion/mixed species, empty MPI owners, local
transport and receiver deadline tightening, the minimum-step resolution guard,
off-grid hierarchy, and native timeline/tree behavior. The tree fixture uses
one MPI rank; other MPI regressions use two. Under `__FINITE_MATH_ONLY__`, the
rate test skips NaN/infinity checks while retaining finite-input quadrature and
normalization checks. All 16 direct-operator state fingerprints were identical
across release/debug and before/after the cleanup.

Current evidence: [full smoke](../../build/sidm-cosmological/maintenance-smoke/validation.json),
[before/after comparisons](../../build/sidm-cosmological/maintenance-smoke/before-after.json),
[focused tests](../../build/sidm-cosmological/maintenance-tests/tests.json),
[debug direct oracle](../../build/sidm-cosmological/maintenance-tests/direct-debug.json),
[binary hashes](../../build/sidm-cosmological/binaries.json), and
[build record](../../build/sidm-cosmological/build-record.json).
The earlier [first audit](../../audits/claude-round1.md) and
[second audit](../../audits/claude-round2.md) are historical records of the initial
integration, which used different compiler settings; their strict-math build
advice does not describe the current build configuration.
