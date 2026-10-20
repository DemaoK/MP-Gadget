#ifndef SUBHALO_COLLECTIVE_H
#define SUBHALO_COLLECTIVE_H

/* Oversized candidates unbound in place with the global tree (design 7.8). The plan has decided
 * every oversized group's central, cut the satellites' sub-trees into serial keys and listed the
 * in-place candidates by round, deepest first. After the domain restore, subhalo_collective_run:
 *  1. gathers the serial sub-tree results into the member table;
 *  2. assigns every particle of an oversized group to one candidate's source (a scratch label, the
 *     tags are not rewritten during the rounds); a central's source is its FOF group only (decision 52);
 *  3. per round: HBT's loop with one global tree and one potential-only walk per iteration, the
 *     sinking, the final pass, the cores and reserved tracers, the relaxed-source claims and the
 *     properties from distributed sorts; the unbound particles pass to the parent candidate;
 *  4. the hierarchy of every oversized group (the serial recipe) and the records;
 *  5. the final tag pass (D10), repairing the serial sub-tree ranks and counts for the particles the
 *     in-place candidates took over or returned to a surviving serial binder.
 * The plan admits the call's memory (subhalo_coll_estimate) before the exchange.
 *
 * Walk contract: result = sum_j m_j W(r_ij, h) over the query's own candidate (W =
 * grav_spline_potential, h = FORCE_SOFTENING()), geometric opening len^2 > r^2 theta^2 (theta^2 capped
 * at 0.33 as in the serial kernel); node labels in NODE.mom.hmax (round label, -1 mixed, -2 empty)
 * skip other candidates' nodes; no PM window; the query skips itself; phi = -(G / a) sum. */

#include <mpi.h>
#include <stddef.h>
#include "subhalo_internal.h"
#include "domain.h"

/* Bytes treewalk.c's ev_begin reserves for the potential-only walk: 40960 + max(4097, 100 x threads)
 * buffers of index, query and result plus ImportBufferBoost x (query + result). */
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
    /* the final tag pass */
    int64_t NAdded;               /* serial tails given to their surviving serial binder */
    int64_t NRepairTracks;        /* serial tracks whose ranks were repaired */
    int64_t NRepairQueries;       /* own-tag particles of those tracks that asked their new rank */
    double TagSortSeconds;        /* the decisions and the in-place own-tag ranks */
    double RepairSeconds;         /* the distributed serial repair */
    double TagSeconds;            /* the whole final tag pass */
    /* memory: the admission estimate and the measured high water, the largest over the ranks (bytes) */
    double EstPeakBytes;          /* persistent + largest stage, as admitted by the plan */
    double MeasPeakBytes;         /* free bytes at the start of the call minus the least free bytes during it */
    double EstAvailBytes;         /* free bytes the plan expected at the start of the call */
    double FreeAtStartBytes;      /* measured free bytes at the start of the call */
    double MaxMeasOverEst;        /* largest measured / estimated over the ranks */
    double WalkElasticBytes;      /* this rank: the most a walk took from the free memory (elastic export tables) */
    int EstPeakRank, MeasPeakRank;
};
const struct SubhaloCollectiveStats * subhalo_collective_last_stats(void);

/* Steps 1-5 above. Needs the restored domain, GrNr restored (subhalo_restore_field_grnr), the serial
 * results in ctx->Res and the in-place slots ctx->Res[IpResStart, IpResStart + NIpRes). Collective
 * over MPI_COMM_WORLD (the treewalk): every rank calls it, also without in-place candidates. */
void subhalo_collective_run(struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, DomainDecomp * ddecomp,
                            struct SubTrackTable * tab, MPI_Comm Comm);

/* ---- test hooks ---- */
/* The walk alone: out[i] = sum_j m_j W(r_ij, h) over the other live PotentialTypes particles of
 * label[i] in [0, nlabels), out[i] untouched for the others. Needs a decomposed domain. Collective. */
void subhalo_collective_test_walk(DomainDecomp * ddecomp, const int32_t * label, int64_t nlabels, double theta, double h,
                                  int PotentialTypes, double * out);
/* round_props on the labelled particles as the bound sets of nlabels candidates with cores core[6 c ..]
 * (CorePos translated comoving, CoreVel in Vel units): dfr[8 c ..] gets Rs (comoving),
 * RockstarCoreVel 3, BulkVel 3, Vrms (km/s; -1 undefined),
 * mvir[c] (if not NULL) the bound virial mass, -1. up needs atime, h, BoxSize, TracerTypes. Collective. */
struct SubhaloUnbindParams;
void subhalo_collective_test_props(const int32_t * label, int64_t nlabels, const double * core, const struct SubhaloUnbindParams * up,
                                   const double rho[SUBHALO_NSO], double * dfr, double * mvir);
/* Called on every rank right before the final tag pass (NULL: none), when P holds the serial keys'
 * provisional tags and ctx->BindOrd this rank's binder ordinals. */
typedef void (*subhalo_collective_hook)(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan);
void subhalo_collective_set_pretag_hook(subhalo_collective_hook fn);

#endif
