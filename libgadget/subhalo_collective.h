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
 *     particles bound by a serial member are excluded; the central's own particles outside the
 *     group are released (its source is the group only, HBT FeedCentrals);
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
 *     of the in-place candidates from one more mpsort_mpi; the serial ranks and counts repaired
 *     (audit Q2) for the particles the in-place candidates took over and for the tails returned to
 *     their surviving serial binder (merged into its own-tag sequence by its final-pass energy
 *     order, from the binder ordinals the serial keys recorded, audit Q3): each change goes to the
 *     rank holding its track's result, which sorts the track's removed ranks and additions, corrects
 *     the record and answers the rank queries of the track's remaining particles (binary searches);
 *     only the list of the affected TrackIds is replicated.
 *
 * Memory: the plan admits the call (subhalo_coll_estimate, audit Q1) before the exchange; every
 * count and rank is checked against the catalogue's int32 columns on every rank before it is
 * narrowed (audit Q4).
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
    /* the final tag pass (audit Q2) */
    int64_t NAdded;               /* serial tails given to their surviving serial binder (Q3) */
    int64_t NRepairTracks;        /* serial tracks whose ranks were repaired */
    int64_t NRepairQueries;       /* own-tag particles of those tracks that asked their new rank */
    double TagSortSeconds;        /* the decisions and the in-place own-tag ranks */
    double RepairSeconds;         /* the distributed serial repair */
    double TagSeconds;            /* the whole final tag pass */
    /* memory (audit Q1): this call's admission estimate and the measured high water of the
     * collective call, the largest over the ranks (bytes) */
    double EstPeakBytes;          /* persistent + largest stage, as admitted by the plan */
    double MeasPeakBytes;         /* free bytes at the start of the call minus the least free bytes during it */
    double EstAvailBytes;         /* free bytes the plan expected at the start of the call */
    double FreeAtStartBytes;      /* measured free bytes at the start of the call */
    double MaxMeasOverEst;        /* largest measured / estimated over the ranks */
    double WalkElasticBytes;      /* this rank: the most a walk took from the free memory (elastic export tables, not in MeasPeakBytes beyond the reserve) */
    int EstPeakRank, MeasPeakRank;
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

/* ---- test hooks ---- */
/* The potential-only walk alone: for every local particle with label[i] in [0, nlabels) (dense
 * labels; -1 outside every candidate), out[i] = sum_j m_j W(r_ij, h) over the other live particles
 * j of its label whose type is in PotentialTypes, with the opening angle theta (theta^2 capped at
 * 0.33 as in the rounds); out[i] is not written for the other particles. Needs a decomposed
 * domain. Collective. */
void subhalo_collective_test_walk(DomainDecomp * ddecomp, const int32_t * label, int64_t nlabels, double theta, double h,
                                  int PotentialTypes, double * out);
/* The properties of one round (round_props: the radial profile and the Phase 5 recipes) for the
 * local particles with label[i] in [0, nlabels) taken as the bound sets of nlabels alive
 * candidates with the prescribed cores core[6 c .. 6 c + 5] (CorePos, CoreVel in Vel units) and
 * the thresholds rho: dfr[8 c ..] receives Rs, RockstarCoreVel 3, BulkVel 3, Vrms (km/s; -1 where
 * undefined), mvir[c] (if not NULL) the bound virial mass (-1 without a crossing). up needs atime,
 * h, BoxSize and TracerTypes. Collective (Phase 5 audit R4: the serial and the distributed recipes
 * on identical inputs). */
struct SubhaloUnbindParams;
void subhalo_collective_test_props(const int32_t * label, int64_t nlabels, const double * core, const struct SubhaloUnbindParams * up,
                                   const double rho[SUBHALO_NSO], double * dfr, double * mvir);
/* A function called on every rank right before the final tag pass of the next calls (NULL: none),
 * with the call context (ctx->BindOrd: the binder ordinals of this rank) and the plan; P then holds
 * the serial keys' (provisional) tags. */
typedef void (*subhalo_collective_hook)(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan);
void subhalo_collective_set_pretag_hook(subhalo_collective_hook fn);

#endif
