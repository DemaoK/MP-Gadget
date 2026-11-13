#ifndef SUBHALO_COLLECTIVE_H
#define SUBHALO_COLLECTIVE_H

/* Oversized candidates unbound in place with the global tree (design 7.8). After the domain restore,
 * subhalo_collective_run gathers the serial sub-tree results, gives every particle of an oversized
 * group to one candidate's source (a central's is its FOF group, as HBT), runs the rounds deepest
 * first (HBT's loop with one global tree and one potential-only walk per iteration, sinking, final
 * pass, cores, claims, distributed properties), then the hierarchy, the records and the final tag
 * pass (design D10). The plan checks the call's memory (subhalo_coll_estimate) before the exchange.
 * Walk: result = sum_j m_j W(r_ij, h) over the query's own candidate (W = grav_spline_potential, h =
 * FORCE_SOFTENING()), opening len^2 > r^2 theta^2 (theta^2 capped at 0.33 as in the serial kernel),
 * node labels in NODE.mom.hmax (round label, -1 mixed, -2 empty) skip other candidates' nodes; no PM
 * window; the query skips itself; phi = -(G / a) sum. */

#include <mpi.h>
#include <stddef.h>
#include "subhalo_internal.h"
#include "domain.h"

/* Bytes treewalk.c's ev_begin reserves for the potential-only walk: 40960 + max(4097, 100 x threads)
 * buffers of index, query and result plus ImportBufferBoost x (query + result). */
size_t subhalo_collective_walk_bytes(void);
/* The inputs of subhalo_coll_estimate: this rank's particle counts (NUniv in oversized groups,
 * NSubPart in its sub-tree keys), owned candidates and their core records (OwnedRecv); the global
 * counts; the domain's top tree, this rank's top leaves and the tree's sizing. */
struct SubhaloCollSizes {
    int64_t NumPart, MaxPart, NUniv, NSubPart, NOwned, OwnedRecv;
    int64_t NCands, NMembers, NSubResults;
    int64_t NTopNodes, NTopLeaves, NLocalLeaves;
    double TreeAllocFactor;
    int nthreads;
};
/* Memory estimate of subhalo_collective_run (particle-free; subhalo_collective.c), made before the
 * exchange: this rank's persistent bytes and per-stage peaks; fills a->persist, stage[] and need. */
void subhalo_coll_estimate(struct SubhaloMemCheck * a, const struct SubhaloCollSizes * z);

/* Diagnostics of the last call (global). */
struct SubhaloCollectiveStats {
    int NRounds;
    int64_t NCands;
    int64_t NCandsRound[16];      /* candidates per round (the first 16 rounds) */
    int Iterations[16];           /* walks per round, the final passes included */
    double Seconds[16];           /* wall time per round */
    int64_t NSunk;                /* descendants sunk into in-place candidates */
    int64_t NDied;                /* in-place candidates that died */
    int64_t NDiedMeanFrame;       /* of which resolved tracks dead at their first unbinding, from the previous-bound frame (no rescue in place) */
    int64_t NAlive;
    int64_t NStolen;              /* serial own-tag particles taken over by an in-place ancestor */
    int64_t NJ2;                  /* claims across the final tree given to the binding owner */
    /* the final tag pass */
    int64_t NAdded;               /* serial tails given to their surviving serial binder */
    int64_t NRepairTracks;        /* serial tracks whose ranks were repaired */
    int64_t NRepairQueries;       /* own-tag particles of those tracks that asked their new rank */
    double RepairSeconds;         /* the distributed serial repair */
    double TagSeconds;            /* the whole final tag pass */
    /* memory: the plan's estimate and the measured high water, the largest over the ranks (bytes) */
    double EstPeakBytes;          /* persistent + largest stage, as checked by the plan */
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

/* One neighbour of a k-nearest search: squared distance (subhalo_knn_r2), ID and previous tag. */
struct SubhaloKnnHit {
    double d2;
    MyIDType ID;
    int64_t BoundTrackId;
};
/* The k nearest particles of tl[0..nt) with the query's label to each query ql[0..nq) (labels in
 * label[], dense, below 2^24), the query itself excluded: hits[k q ..] sorted by (d2, ID), nhit[q]
 * of them (fewer only when the label has fewer). Exact whenever it completes: the radius of a query
 * doubles per pass until its k-th neighbour lies strictly inside. Limits of the builder and walker it
 * shares: more than eight particles at one position abort the tree build, and a query whose exports
 * exceed a thread's export buffer aborts the walk. Needs a decomposed domain with every particle on
 * its domain rank. Collective over MPI_COMM_WORLD; returns the number of passes. */
int subhalo_collective_knn(DomainDecomp * ddecomp, const int32_t * label, const int * tl, int64_t nt,
                           const int * ql, int64_t nq, int k, struct SubhaloKnnHit * hits, int32_t * nhit);
/* Export rounds of the last subhalo_collective_knn on this rank, summed over its passes. */
int64_t subhalo_collective_knn_export_rounds(void);
/* Bytes treewalk.c reserves for one pass of that walk with k neighbours. */
size_t subhalo_collective_knn_walk_bytes(int k);

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
