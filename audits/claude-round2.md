# Claude audit record — round 2

Read-only Claude CLI 2.1.296 audit, 2026-10-12. Model reported by the CLI:
`claude-fable-5-1`. The report below is preserved as returned; maintainer
disposition follows it.

# SIDM cosmological integration — audit round 2

**Bottom line:** No unresolved material correctness defect was found. The round‑1 hypothesis about the minimum-step transport guard is not a defect; the maintainer disposition is correct. The only new confirmed problems are evidence-record gaps, not code.

## Status of round‑1 findings

| # | Round‑1 item | Round‑2 status |
|---|---|---|
| 1 | Relax transport guard at `sidm-local.c:164-165`, widen `:250` | **Hypothesis rejected; disposition verified.** Loop order in `run.c:369-371` and `:835`: the envelope used to tighten a deadline in `sidm_local_next_event` is the same `max_speed` later used by `sidm_local_begin_active` (`sidm-local.c:205`) to charge travel over the interval, so a bin‑1 deadline is consumed at the next event with no intervening envelope change and `:250` cannot trip. For a larger pending bin, the abort needs `v_new/v_old > D(S,M)/D(S,S+2)`; a single elastic scatter raises the envelope by at most 2×, so the deadline must lie within about 4 ticks of the envelope-raising event. With `TIMEBINS 46` (`timebinmgr.h:13`) a tick is `Δloga_output/2^46`, so this is a genuine resolution limit. The proposed widening would admit relative displacement beyond `transport·h` with no bound; `SIDMMaxSupportChange` bounds the change of `h`, not pair geometry. `test_sidm_local.c:117-180` now covers alignment-limited tightening to bin 1 and the opt‑in guard abort (logs confirm 4 tests pass; guard aborts with the intended message on both ranks). |
| 2 | Whole-build `-fno-fast-math` | Confirmed behaviour, documented at `README.md:47-52`. Accepted. |
| 3 | Short-bin realignment cost | Confirmed, documented at `README.md:28-30`. Accepted. |

## New findings

**Confirmed, low severity (evidence record, not code):**

1. `transport-followup.json` does not exist in `build/sidm-cosmological/`, although `README.md:156` and `audits/claude-round1.md:71` cite it for commands and hashes. Only `transport-regression-release.log` and `transport-resolution-guard.log` exist. `tests.json` and `test-sidm_local.log` record the earlier two-test `test_sidm_local`, so no JSON record covers the follow-up binary. Remedy: regenerate the JSON or correct both citations.
2. `validated/validation.json` was produced by an earlier `validate.py` revision: it lacks the `binary_sha256` field and the in-loop `*_zero_cross_section` sections the current script writes (`validate.py:218-220`, `:265-266`). The current full path has not run end to end; the zero-cross-section check used `--zero-only`. Binary identity is still established: `binaries.json` hashes equal those in `zero-cross-section/validation.json`. Remedy: note this in the README or rerun the full harness once.

**Hypotheses / observations (no defect):**

- Native `get_exact_factor` (`timefac.c:43-58`) integrates between `exp(loga_from_ti())` endpoints, so drifts over SIDM-only intervals shorter than roughly 10³ ticks lose relative precision, while the SIDM ledger uses exact integer widths (`timefac.c:171-181`). The resulting position error is at the double ulp level; immaterial. No action.

## Rechecked hooks (verified by reading)

- **Phase and scheduling.** `sidm_local_kick_phase` (`sidm-local.c:192-198`) writes absolute values. For every bin native code later reads, these equal the incremental `update_kick_times` result: active occupied bins get `T` at closing and `T+dti/2` at opening; off-grid SIDM-only events touch only bins below `mintimebin`, which native does not read until the next native closing resets them. `mintimebin` is never modified at SIDM-only events because `hierarchical_gravity_and_timesteps` and `find_hydro_timesteps` are gated by `totgravactive` and `native_event` (`run.c:802-810`). `find_next_kick` (`timestep.c:1353-1359`) is the identity at native times. PM ends and sync points are always native events. The exact DM match in the zero‑σ run corroborates this.
- **Units and factors.** `sigma_code = σ_cgs·h·UnitMass/UnitLength²` and `w_code = a·w_phys` (`sidm-distributed.c:261-266`); pair probability `½·m·W_com·σ·u·E` with `E=∫dt/a⁴` equals `ρ_phys σ v_phys dt`. `sidm_interval_width` splits at segment edges and uses `dloga_from_dti` (`timebinmgr.c:420-432`), so 2‑tick intervals are exact.
- **Live type‑1 filtering and receivers.** `live()` and `DMMASK` everywhere; gas, stars, BHs never enter the DM tree or operator. Inactive receivers are drifted every event (`domain.c:309-313`, `drift.c:83-102`), so neighbour positions are current.
- **Hierarchy boundaries.** The gravity limit only shortens bins (`timestep.c:379-391`); `sidm_drift[b]` endpoints stay within the PM step and `final_ti`. Deadlines never cross a sync point (`sidm-local.c:92`).
- **Ownership, migration, empty ranks.** All DM is kept on its top-leaf owner (`domain.c:300-323`); per-particle clocks and `SIDMTravelLast` live in `P` and migrate; `Local.travel` is global. Empty-rank tree build fixed (`forcetree.c` early return). Toptree export pass never visits particles, so exports use the unshrunk query radius; the primary runs once before ghost merges (`treewalk.c:851-873`); ghost reduction is serial (`treewalk.c:767-793`), so `retain` merges are race-free, including multiple node-list entries per task.
- **Restart flush.** `sidm_local_assert_flushed` precedes `write_checkpoint` (`run.c:732`); `sidm_local_initialize` resets all clocks and travel on restart; `SIDMSupport` is the only read SIDM field (`petaio.c:1029`), and `SmoothingLength` is registered only for types 0/4/5, so no double registration.
- **Direct operator contract.** Caller supplies validated equal mass (`audit_population`, once per process in release), `validated_mass>0`, disjoint per-source heaps, `dt≥0` (zero-exposure PM flush measures rates), rank-consistent parameters (`MPI_Bcast` at `sidm-distributed.c:36`) and seeds, and fixed ownership within the call. `max_sigma_v` feeds `limits[i]` consistently with the calibrated solver.

## Tests actually evidenced

- `tests.json`: rate, direct (2 ranks), local (2 tests at that time), distributed, distributed empty-owner, hierarchical, native timefac/timebinmgr/forcetree, all `returncode 0`.
- Follow-up logs: `test_sidm_local` with 4 tests passing (release build, 2 ranks); guard fixture aborts as intended.
- `validated/validation.json` (`passed: true`): DM and gas/DM full and resumed release/debug agreement, runtime-disabled vs compiled-out agreement, missing-build rejection, flushed clocks, resumed zero exposure.
- `zero-cross-section/validation.json` (`passed: true`): zero events, zero source-only epochs, zero transport counters, exact DM match to native-disabled output for both cases.

## Remaining test limitations

- In the collision smoke, every PM report shows `source_transport=0`, `gravity_transport=0`, `deadline_tightened=0` (`validated/dm/debug/run.log`). Transport-limited scheduling, gravity-bin shortening, and the guard are exercised only by two-particle and four-particle unit fixtures, never by an integrated run.
- Smoke runs cover three PM intervals at z≈9 with 4,096 particles and `sigma0=1e7`. They establish plumbing and native-phase equivalence only, not convergence, realistic cost, or long-run envelope behaviour.
- Source hashes in `build-record.json` could not be checked against the working tree without running commands.
- The current `validate.py` full path has not been executed (finding 2 above).

## Maintainer disposition

No production correction was requested or made after this audit.

1. The transport manifest was finalized while this audit was running and now
   exists at `build/sidm-cosmological/transport-followup.json`. It records the
   exact commands, test-source hash, binary hashes, successful four-test release
   and debug runs, and the expected nonzero guard run. The source hash was
   checked against the final working tree. This complements the earlier focused
   suite in `tests.json`; the earlier record is retained without relabeling it.
2. The full smoke in `validated/` ran before the zero-cross-section comparison
   was added to the harness. The additional `--zero-only` path ran both DM and
   gas/DM against those saved native-disabled outputs using the same binaries.
   The combined current default harness path has not been rerun end to end.
   README now states this explicitly. `build-record.json` retains the build
   source hashes and records the later harness invocation and its hashes under
   `post_build_validation`; no rebuild or full rerun is implied.
3. A final hash comparison found no production C/header change since the tested
   builds. The direct collision operator also matches the calibrated isolated
   source byte for byte. The follow-up changes are tests and documentation.

The transport-limited integration, longer cosmological evolution, star
formation/feedback, larger MPI layouts and realistic performance limitations
remain as described above. Small smoke tests and focused fixtures do not
establish physical convergence or production cost.
