# Claude audit record — round 1

Read-only Claude CLI 2.1.296 audit, 2026-10-12. Model reported by the CLI: `claude-fable-5-1`. The report below is preserved as returned; maintainer disposition follows it.

## SIDM cosmological integration — audit round 1

**Bottom line:** I found no confirmed material correctness defect in the unit conversions, exposure/displacement integrals, clock bookkeeping, split-kick phases, live-DM filtering, periodic ownership, collectives, threading, checkpoint flush, zero-cross-section path, or compiler gating. One reachable hard-abort path is a hypothesis that needs a reproduction, and several of the stated claims are not actually exercised by the tests or the (incomplete) smoke run.

### Verified correct (by reading)

- **Units.** `sidm-distributed.c:261` gives `sigma_code = sigma_cgs·h·UnitMass/UnitLength²`, which is right for code mass `UnitMass/h` and code length `UnitLength/h`. `w_code = a·w_phys` at `:266` so `u/w_code = v_pec/w_phys`. Pair probability `½·m·W(r_com,h_com)·σ·u·E` with `E=∫dt/a⁴` (`timefac.c:2103-2104` in the diff) is the comoving form of `ρ_phys(σ/m)v_phys dt`. Elastic exchange in `u = a·v_pec` at fixed `a` is exactly physical elastic scattering.
- **Integrals.** Quartic-cell fit and cell integral (`timefac.c` `sidm_cell_integral`, coefficient recovery) are algebraically correct; multi-cell path (first partial, suffix sums, tail) is consistent. Matches the native `drift_integ` convention. Independent GSL check reports 1e-15 relative agreement.
- **Clocks and boundaries.** Deadlines are always `now + 2^bin` with `now % 2^bin == 0` (`prepare_forward` gates on `is_timebin_active`), so SIDM-only events never land on an occupied native bin boundary; hence `sidm_local_kick_phase` only rewrites `Ti_kick` for empty bins at off-grid times, and at native times writes exactly the absolute values the incremental `update_kick_times` would produce. `find_next_kick` change is identity for aligned native times. PM end is always a native event, so flush/`assert_flushed` invariants hold.
- **Live filtering / hydro.** SIDM-only events skip density/hydro/SF/cooling correctly; `nactivesidm` matches `begin_active` on both branches; gas, stars, BHs never enter the DM tree or the operator (test asserts this).
- **Geometry.** Local walk shrinks `Hsml` only after K local candidates; exports use the full radius; ghost reduction merges into the K-heap, so the retained K are globally exact. Owner/index come from the remote rank's `work`. Radius capped below `BoxSize/2`.
- **Collectives/empty ranks.** Every MPI call in `sidm-local.c`, `sidm-distributed.c`, `sidm-direct.c` is unconditional on all ranks; `collective_error` fences every `goto cleanup`. Round-robin partner schedule checked for 2, 3, 4 ranks.
- **Threading.** Prediction is read-only into disjoint cache slots; application is serial with epoch-marked invalidation of both endpoints; `qsort` on disjoint arrays.
- **Zero σ.** `rate=0 → limits=DBL_MAX` (finite) → `interval_bin` caps at `MaxSizeTimestep`; operator returns without drawing.
- **Gating.** Non-SIDM build has stubs for every symbol used outside `#ifdef SIDM`; parameter file is read on every rank, so `Local.enabled` is rank-consistent.

### Findings

| # | Severity | Location | Status | Mechanism | Minimal remedy |
|---|---|---|---|---|---|
| 1 | Medium | `libgadget/sidm-local.c:164-165`, `:250-251` | Hypothesis (needs reproduction) | Transport budget is a hard `endrun`. A receiver-only particle whose deadline was tightened to bin 1 has `remaining ≈ 2·v₀·D_min`. If the global envelope `max_speed` rises before that deadline (an opening gravity kick or a single elastic scatter can raise the global max by up to 2×), `remaining + tol < 2·v₁·D_min` trips `:165` and aborts a legitimate run. With `sigma0` small the deadlines are transport-dominated, so many particles sit near exhaustion. The smoke never entered this branch (`deadline_tightened=0`, `gravity_transport=0` at every PM report), so it is untested. | At `:165` set `dt = minimum` (deadline `now+dti(1)`) instead of aborting, and widen the `:250` check by one minimum step `2·max_speed·forward_drift[1]`; the 0.25 support-change margin still covers the overshoot. |
| 2 | Low | `Makefile.rules:24-26` | Confirmed behaviour | `-fno-fast-math` is appended to the whole build when `-DSIDM` appears in `OPT`, so native gravity/hydro objects differ numerically from an upstream `-ffast-math` build even with `SIDMOn=0`. Not a correctness bug, but the disabled-vs-compiled-out comparison in `validate.py` only passes because `build.sh` omits fast-math in both variants. | Document, or apply the flag per-object to `sidm-*.o` only. |
| 3 | Low | `libgadget/sidm-local.c:90-97`, `:387` | Confirmed cost effect | Sources committed at an off-grid event only get bins active at that `now`, so they are forced onto short bins until they re-align (visible in the smoke as 1–5 source events between each bulk event). Correct, but each such event is a full `O(N)` scan plus Allreduce in `sidm_local_next_event` and `after_open`. | None required; note under cost limitations. |

### Test coverage versus claims

- **Exercised well:** direct operator vs serial oracle (2 ranks, empty owner, subdivision, boost invariance, strided view, negative inputs); exact periodic K-NN vs brute force; little-h and `a·w` conversion; inactive receivers; conservation; export-retry equivalence; gas exclusion; `Hsml` seed variants including the `Father` leaf guess; EdS analytic factors; LCDM factors vs GSL; mixed-species clocks; PM flush.
- **Not exercised by any test:** deadline tightening and the exhaustion guard (finding 1); gravity-bin shortening in `hierarchical_gravity_and_timesteps:379-391`; the support-change abort; `sidm_local_kick_phase` sequencing with real gravity kicks (`test_sidm_hierarchical` swaps velocities by hand and never calls the kick-phase or timestep code with SIDM enabled); restart from a flushed checkpoint under the current harness.
- **Smoke status:** `smoke-v4/validation.json` is `passed=false` with no cases. The DM case ran through 3 PM steps with release/debug agreement, then failed on a restart expectation that has since been revised in `validate.py`; the hydro case, the disabled/compiled-out comparison and the reject-missing-build check were never reached. No run of the current `validate.py` exists. The evidence file's "all focused tests pass" is supported by the logs; "final smoke validation in progress" is accurate.

### Remaining limitations

- A decisive, cheap check for the kick-phase bookkeeping is missing: with `sigma0=0` and transport never binding (true in the smoke), an SIDM-enabled run should reproduce the native run to roundoff. Adding that comparison to `validate.py` would turn the phase analysis above from reading into evidence.
- The smoke uses `sigma0=1e7` at z≈9 on 4096 particles. It demonstrates integration plumbing only; it says nothing about convergence, realistic cost, the global-envelope penalty in simulations with bulk flows, or behaviour over many PM steps where `Local.travel` and the envelope evolve.
- Release-mode `audit_population` runs once per process; equal-mass and tree-count invariants are re-checked only in `SIDMDEBUG`.

## Maintainer disposition

The audit began while smoke validation was still running. Its statements about
`smoke-v4` describe an earlier attempt, not the completed validation at
`build/sidm-cosmological/validated/validation.json` (`passed: true`). The final
suite exercised both DM and gas/DM, full and restarted release/debug runs,
runtime-disabled/compiled-out comparisons, and missing-build rejection. See
`examples/sidm-cosmological/README.md` for outcomes and the restart contract.
The report's `timefac.c:2103-2104` refers to diff positions, not source line numbers.

The whole-build floating-point setting is intentional and is now explicitly
documented, including its effect when SIDM is disabled at runtime. Comparisons
use matching compiler arithmetic flags. Temporary short source bins during
hierarchical re-alignment and their scan/reduction costs are also documented.
The proposed relaxation of the minimum-step transport guard was not adopted.
A source already scheduled at `now+2` (bin 1) is processed at the next possible
event; there is no intervening event that can raise the speed envelope. A larger
pending bin can encounter an intervening speed increase, but if its remaining
budget is then smaller than one minimum-tick displacement, no future integer
clock value can satisfy that budget. Aborting is the intended resolution limit.
Allowing an extra tick would silently violate the displacement invariant; the
separate support-change tolerance does not bound that skipped geometry.


Follow-up regressions passed with two MPI ranks and two OpenMP threads:
`test_sidm_local` now exercises receiver deadline tightening after a speed
increase and actual transport-driven bin assignment in
`hierarchical_gravity_and_timesteps`. The opt-in `--expect-resolution-guard`
fixture reproduces the intended abort only after asserting that the remaining
budget is smaller than the shortest representable drift. Commands, hashes and
results are in `build/sidm-cosmological/transport-followup.json`; logs are
`transport-regression-release.log` and `transport-resolution-guard.log` in the
same directory. No production guard was relaxed.

The additional zero-cross-section DM and gas/DM runs passed. They have no
source-only epochs and no transport-limited sources, gravity steps or tightened
deadlines, so this establishes native-phase equivalence. DM endpoint positions,
velocities and potentials match native-disabled output exactly. Gas differences
are within the previously measured native threading variation. See
`build/sidm-cosmological/zero-cross-section/validation.json`. The nonzero-scattering
smoke separately exercises independent source events; this zero-scattering check
does not claim to test such events.

No confirmed material correctness defect remains from round 1. The documented
cost limitations and untested larger-scale physical configurations remain open
validation work, not evidence of convergence or production performance.
