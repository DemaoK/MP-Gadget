#ifndef SUBHALO_COLLECTIVE_H
#define SUBHALO_COLLECTIVE_H

/* Oversized candidates unbound in place with the global tree (design 7.8, Phase 2b plan step 9,
 * halofinder_phase2b_plan.md).
 *
 * The plan (subhalo_plan_build) has decided the central of every oversized group, cut the
 * satellites' sub-trees into serial keys (processed with provisional tags by
 * subhalo_process_serial) and listed the in-place candidates with their rounds (deepest first).
 * After the domain restore, subhalo_collective_run:
 *  1. gathers the serial sub-tree results into the member table (states, cores, Keep);
 *  2. assigns every particle of an oversized group to the source of one candidate (a per-call
 *     scratch label, the tags are not rewritten during the rounds): own tag of an in-place
 *     candidate; the provisional remainder of a serial sub-tree (BoundTrackId -1) to the
 *     candidate above its root; untagged and foreign-tagged group particles to the central;
 *     particles bound by a serial member are excluded;
 *  3. for every round: the HBT loop with one global tree (force_tree_active_moments over the
 *     bound potential-type particles of the round) and one potential-only walk per iteration,
 *     the per-candidate frames, counts and tracer counts by one MPI_Allreduce, kill, convergence
 *     and the iteration cap; the sinking of the candidate's descendants (serial and in-place)
 *     tested on the cores, a sunk survivor's bound set joining the candidate, which is unbound
 *     again; the final pass; the cores, the most-bound tracer and the reserved tracers from a
 *     per-owner selection; the list positions of the unbound particles by mpsort_mpi on (label,
 *     E, ID) for the relaxed sources (claims); the properties through a distributed radial
 *     ordered set (mpsort_mpi on (label, r, ID) and a vector MPI_Exscan of mass); the unbound
 *     particles pass to the parent candidate;
 *  4. the hierarchy of every oversized group (the serial recipe: ranks by mass, the most massive
 *     survivor promoted to the central, nearest surviving ancestors), the records of the in-place
 *     candidates on their owners and the hierarchy columns of the serial sub-tree results;
 *  5. the final tag pass: reservation, deepest valid claim (serial claims first, then the in-place
 *     ones) with the final-tree check of the binding owner (J2), BoundTrackId, dense own-tag ranks
 *     of the in-place candidates from one more mpsort_mpi, the serial ranks and counts corrected
 *     for the particles the in-place ancestors took over.
 *
 * Walk contract (the potential-only visit): query = position, round label, own particle index;
 * result = sum_j m_j W(r_ij, h) with W = grav_spline_potential and h = FORCE_SOFTENING(); the
 * geometric opening len^2 > r^2 theta^2 against the node centre of mass (theta =
 * SubhaloTreeOpenAngle, theta^2 capped at 0.33 as in the serial kernel); node labels in
 * NODE.mom.hmax: the dense round label of the candidate owning every particle below, -1 mixed,
 * -2 empty; a node with another label or -2 is skipped, a mixed node is opened, a node with the
 * query's label is used as a monopole when it passes the opening test; particle leaves are
 * filtered on the label and the query particle itself (local queries only); no PM window, no
 * Rcut; exports through the treewalk's top-tree path; NoNgblist = 1; the reduce writes the sum
 * into a per-particle scratch array, phi = -(G / a) sum. */

#include <mpi.h>
#include <stddef.h>
#include "subhalo_internal.h"
#include "domain.h"

/* Bytes treewalk.c's ev_begin reserves for the potential-only walk: 40960 x (24 + (1 +
 * ImportBufferBoost) (query + result)) plus 100 buffers per thread (boost 2 assumed). */
size_t subhalo_collective_walk_bytes(void);
/* The ImportBufferBoost the estimate uses (the run's parameter, set by set_subhalo_params). */
void subhalo_collective_set_import_buffer_boost(double boost);

/* Diagnostics of the last call (global). */
struct SubhaloCollectiveStats {
    int NRounds;
    int64_t NCands;
    int64_t NCandsRound[16];      /* candidates per round (the first 16 rounds) */
    int Iterations[16];           /* walks per round, the final passes included */
    double Seconds[16];           /* wall time per round */
    int64_t NSunk;                /* descendants sunk into in-place candidates */
    int64_t NDied;                /* in-place candidates that died */
    int64_t NAlive;
    int64_t NStolen;              /* serial own-tag particles taken over by an in-place ancestor */
    int64_t NJ2;                  /* claims across the final tree (J2) given to the binding owner */
};
const struct SubhaloCollectiveStats * subhalo_collective_last_stats(void);

/* Steps 2-5 above. Needs the restored domain (TopLeaf valid), GrNr restored
 * (subhalo_restore_field_grnr), the serial results in ctx->Res and the in-place slots
 * ctx->Res[IpResStart, IpResStart + NIpRes). Writes the in-place records, patches the serial
 * sub-tree records, writes the final tags of every particle of an oversized key. Collective over
 * MPI_COMM_WORLD (the treewalk); every rank calls it, also without in-place candidates (it then
 * returns at once). */
void subhalo_collective_run(struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, DomainDecomp * ddecomp,
                            struct SubTrackTable * tab, MPI_Comm Comm);

#endif
