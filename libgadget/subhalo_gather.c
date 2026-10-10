/* Runtime subhalo finder: host vote routing, the plan (budgets, oversized keys, packing),
 * the whole-key particle exchange, the local sort, the member fetch, the domain restore and
 * the per-thread scratch arenas.
 *
 * Phase 1 plan (halofinder_phase1_plan.md) section 1 steps 2-5, 7a and 9; design 7.1, 7.3.
 * The exchange reuses the mechanics of fof_distribute_particles (fofpetaio.c): target task in
 * P[i].TopLeaf, domain_exchange on the main particle table, local sort; slots_gc_sorted and
 * domain_maintain restore the domain afterwards. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/openmpsort.h"
#include "utils/system.h"

#include "partmanager.h"
#include "slotsmanager.h"
#include "exchange.h"
#include "domain.h"
#include "walltime.h"
#include "subhalo.h"
#include "subhalo_internal.h"

/* Number of the largest serial keys packed exactly by the greedy on rank 0; the rest (the
 * tail) are spread over the remaining budgets in proportion to them (what longest-first
 * packing reduces to once the keys are small compared with the budgets), see subhalo_plan_pack. */
#define SUBHALO_HEAD_PER_TASK 32
#define SUBHALO_HEAD_MAX 4096
/* Classification and packing passes are repeated until a pass moves no key in place, at most
 * this many times; the census rounds of subhalo_plan_pack then settle the plan either way. */
#define SUBHALO_PLAN_MAXITER 16
/* The truly constant part of the byte budget (H1): headroom for the allocator's per-block
 * headers and alignment (domain_exchange itself assumes up to 2 x 4096 bytes per array) over
 * the few dozen allocations of one call, qsort_openmp's per-thread merge state, and the small
 * fixed-size scratch of the driver. Everything that scales with the work is itemised in
 * subhalo_plan_default_cfg. */
#define SUBHALO_RESERVE_BYTES (8L * 1024L * 1024L)
/* Main-stack bytes per rank of the communicator (NTask-sized count/budget vectors). */
#define SUBHALO_RESERVE_PER_TASK 256
/* Per-key scratch besides the result, segment and output record: the assigned-key entry, the
 * thread schedule (KeyOrder + owner), the newborn numbering entries and the in-place fetch id. */
#define SUBHALO_KEY_MISC_BYTES 128
/* domain_exchange never moves more than this per batch (exchange.c) */
#define SUBHALO_EXCHANGE_BATCH_BYTES (2040L * 1024L * 1024L)

/* ------------------------------------------------------------------ step 2: host vote */

struct VotePair {
    int64_t TrackId;
    int64_t GrNr;
    int64_t count;
};

static int
cmp_votepair(const void * a, const void * b)
{
    const struct VotePair * x = (const struct VotePair *) a, * y = (const struct VotePair *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    return (x->GrNr > y->GrNr) - (x->GrNr < y->GrNr);
}

/* Sort and merge equal (TrackId, GrNr) pairs; returns the new count. */
static int64_t
merge_votepairs(struct VotePair * v, int64_t n)
{
    int64_t i, m = 0;
    qsort_openmp(v, n, sizeof(struct VotePair), cmp_votepair);
    for(i = 0; i < n; i++) {
        if(m > 0 && v[m-1].TrackId == v[i].TrackId && v[m-1].GrNr == v[i].GrNr)
            v[m-1].count += v[i].count;
        else
            v[m++] = v[i];
    }
    return m;
}

void
subhalo_host_vote(struct SubTrackTable * tab, MPI_Comm Comm)
{
    int64_t i, n = 0;
    const int NTask = tab->NTask;
    /* live particles only (H6): a swallowed BH keeps its tags but does not vote */
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && !P[i].Swallowed && P[i].TrackId >= 0 && P[i].BoundTrackId == P[i].TrackId)
            n++;
    struct VotePair * v = (struct VotePair *) mymalloc("SubVote", sizeof(struct VotePair) * (n + 1));
    n = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && !P[i].Swallowed && P[i].TrackId >= 0 && P[i].BoundTrackId == P[i].TrackId) {
            v[n].TrackId = P[i].TrackId;
            v[n].GrNr = P[i].GrNr >= 0 ? P[i].GrNr : -1;
            v[n].count = 1;
            n++;
        }
    n = merge_votepairs(v, n);

    /* sorted by TrackId, hence by home rank */
    int64_t * sendcounts = ta_malloc("SubVoteCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < n; i++)
        sendcounts[subhalo_table_home_rank(tab, v[i].TrackId)]++;
    struct VotePair * in = NULL;
    int64_t nin = subhalo_alltoallv(v, sendcounts, sizeof(struct VotePair), (void **) &in, 0, NULL, Comm);
    nin = merge_votepairs(in, nin);

    /* Reduce per track: the GrNr with the most votes, ties to the smaller GrNr; "no group"
     * only with a strict majority of the track's bound particles. */
    int64_t a = 0, nhosted = 0, nfield = 0;
    int missing = 0;
    while(a < nin) {
        int64_t b = a, total = 0, nogroup = 0, best = -1, bestcount = 0;
        while(b < nin && in[b].TrackId == in[a].TrackId) {
            total += in[b].count;
            if(in[b].GrNr < 0)
                nogroup += in[b].count;
            else if(in[b].count > bestcount) {
                /* GrNr ascending within the track: strict > keeps the smaller GrNr on ties */
                bestcount = in[b].count;
                best = in[b].GrNr;
            }
            b++;
        }
        struct SubTrack * rec = subhalo_table_get(tab, in[a].TrackId);
        if(!rec)
            missing = 1;
        else {
            rec->Host = (2 * nogroup > total || best < 0) ? -1 : best;
            if(rec->Host >= 0) nhosted++; else nfield++;
        }
        a = b;
    }
    if(missing)
        endrun(5, "Host vote for a track not homed on rank %d.\n", tab->ThisTask);
    myfree(in);
    ta_free(sendcounts);
    myfree(v);

    int64_t loc[2] = {nhosted, nfield}, tot[2];
    MPI_Allreduce(loc, tot, 2, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo host vote: %ld tracks hosted by a FOF group, %ld with no host group.\n", tot[0], tot[1]);
}

/* ------------------------------------------------------------------ step 3: plan */

static int
cmp_dirmsg(const void * a, const void * b)
{
    const struct SubhaloDirMsg * x = (const struct SubhaloDirMsg *) a, * y = (const struct SubhaloDirMsg *) b;
    if(x->g != y->g)
        return (x->g > y->g) - (x->g < y->g);
    if(x->kind != y->kind)
        return (x->kind > y->kind) - (x->kind < y->kind);
    return (x->rank > y->rank) - (x->rank < y->rank);
}

/* Number of capacity dimensions: particles, the six types, bytes. */
#define PLAN_NDIM 8
/* Per-rank oversized census: six types, total, member tags, local in-place segments. */
#define PLAN_NOV 9

/* Per-rank budget of one iteration (design 7.3). cap[0] = B (particles the rank may receive),
 * cap[1 + t] = free slots of type t (1e30 if the type has no slots), cap[7] = bytes left for
 * the received keys (arenas, member records, per-key buffers). */
struct RankBudget {
    double cap[PLAN_NDIM];
    double freebytes;
};

/* Head key sent to rank 0 for the greedy. */
struct HeadKey {
    int64_t g;
    double dims[PLAN_NDIM];
};
struct HeadResult {
    int64_t g;
    int64_t target;     /* -1: cannot be placed */
};

static int
cmp_localkey_g(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloLocalKey *) a)->g, y = ((const struct SubhaloLocalKey *) b)->g;
    return (x > y) - (x < y);
}
static int
cmp_assigned_g(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloAssignedKey *) a)->g, y = ((const struct SubhaloAssignedKey *) b)->g;
    return (x > y) - (x < y);
}
static int
cmp_inplace_g(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloInplaceKey *) a)->g, y = ((const struct SubhaloInplaceKey *) b)->g;
    return (x > y) - (x < y);
}

void
subhalo_plan_default_cfg(struct SubhaloPlanCfg * cfg, int64_t TotNgroups, int NTask)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    memset(cfg, 0, sizeof(*cfg));
    cfg->GatherFraction = sp->SubhaloMaxGatherFraction;
    cfg->CollectiveMinPart = sp->SubhaloCollectiveMinPart;
    cfg->ForceRankZero = sp->SubhaloDebugForceExchangeFailure;
    cfg->TotNgroups = TotNgroups;
    cfg->KChunk = subhalo_key_chunk(TotNgroups, NTask);
    cfg->MaxIter = SUBHALO_PLAN_MAXITER;
    /* Byte costs of the post-exchange peaks (H1). The serial key loop holds, simultaneously:
     *  - per particle of a received key: its arena share (the thread arenas sum the per-thread
     *    maxima, at most the sum of the key lengths; the "alone" arena is freed before them);
     *    the restore exchange that follows sizes its batches from the free memory
     *    (exchange.c, domain_find_iter_space) and its per-particle lists are covered by the
     *    SortBytesPerPart term, so it adds nothing per particle here;
     *  - per member track (distinct tag in a key, bounded by the summed per-rank tag counts):
     *    the fetched record (SubMembers), the output copy (SubOutRecs) and one spare record,
     *    plus the id list, the fetch/store permutation, the sink-winner slot and one spare;
     *  - per key: its result (record + bookkeeping), its segment, its output record and the
     *    small per-key scratch (SUBHALO_KEY_MISC_BYTES).
     * The fetch and store move records in batches bounded by subhalo_table_batch_peak_bytes,
     * which is in the reserve together with the constant SUBHALO_RESERVE_BYTES and the
     * NTask-sized vectors. */
    cfg->PartBytes = SUBHALO_ARENA_BYTES_PER_PART;
    cfg->MemberBytes = 3. * sizeof(struct SubTrack) + 4. * sizeof(int64_t);
    cfg->KeyBytes = sizeof(struct SubhaloKeyResult) + sizeof(struct SubhaloKeySeg) + sizeof(struct SubTrack) + SUBHALO_KEY_MISC_BYTES;
    cfg->InplaceKeyBytes = SUBHALO_INPLACE_KEY_BYTES;
    /* exchange.c per departing particle: the particle and slot copies, the exchange list entry
     * and the layout cache (the slot element size is added per rank) */
    cfg->ExchPartBytes = sizeof(struct particle_data) + sizeof(int) + 8;
    /* the local sort (qsort_openmp sorts particle_data indirectly: two pointers per element),
     * reused later by the newborn request list and the restore's exchange lists */
    cfg->SortBytesPerPart = 16;
    /* constant per call: SUBHALO_RESERVE_BYTES, one batch of the record fetch/store, the
     * NTask-sized vectors, and the newborn numbering's directory entries and lookups for the
     * KChunk keys this rank's directory covers (about four 16-byte entries per key) */
    cfg->ReserveBytes = SUBHALO_RESERVE_BYTES + subhalo_table_batch_peak_bytes(NTask) + (double) NTask * SUBHALO_RESERVE_PER_TASK
        + 64. * (double) cfg->KChunk;
}

/* Budget of this rank given the census of its particles in oversized keys (ov, PLAN_NOV) and
 * the global number of in-place keys. */
static void
rank_budget(struct RankBudget * rb, const struct SubhaloPlanRes * res, const int64_t * ov, int64_t NInplace, const struct SubhaloPlanCfg * cfg)
{
    int t;
    const int64_t dep = res->grp[6] - ov[6];             /* serial-key particles leaving */
    const int64_t stay = res->NumLive - dep;
    double B = cfg->GatherFraction * (double) (res->MaxPart - stay);
    rb->cap[0] = B > 0 ? B : 0;
    double maxelsize = 0;
    double slotgrowth = 0;
    for(t = 0; t < 6; t++) {
        if(res->slotelsize[t] <= 0) {
            rb->cap[1 + t] = 1e30;
            continue;
        }
        const int64_t dept = res->grp[t] - ov[t];
        rb->cap[1 + t] = (double) (res->slotfree[t] + dept);
        if(res->slotelsize[t] > maxelsize)
            maxelsize = res->slotelsize[t];
        slotgrowth += (double) dept * res->slotelsize[t];
    }
    /* exchange buffers (at most one batch), slot growth, the local sort, the constant reserve,
     * the in-place vectors of every in-place key and this rank's local part of the in-place
     * keys (their members and per-key buffers); the remainder is for the received keys. */
    double exch = (double) dep * (cfg->ExchPartBytes + maxelsize);
    if(exch > SUBHALO_EXCHANGE_BATCH_BYTES)
        exch = SUBHALO_EXCHANGE_BATCH_BYTES;
    rb->freebytes = res->freebytes;
    const double avail = res->freebytes - exch - slotgrowth - cfg->SortBytesPerPart * (double) res->MaxPart
        - cfg->ReserveBytes - cfg->InplaceKeyBytes * (double) NInplace
        - cfg->MemberBytes * (double) ov[7] - cfg->KeyBytes * (double) ov[8];
    rb->cap[7] = avail > 0 ? avail : 0;
}

static void
key_dims(const struct SubhaloPlanCfg * cfg, const struct SubhaloDirKey * k, double dims[PLAN_NDIM])
{
    int t;
    dims[0] = (double) k->Length;
    for(t = 0; t < 6; t++)
        dims[1 + t] = (double) k->LenType[t];
    dims[7] = subhalo_plan_key_bytes(cfg, k->Length, k->Tags);
}

/* Greedy longest-first on rank 0 for the head keys (already in GrNr = decreasing length order):
 * the rank with the most free particle budget among those where the key fits in every dimension. */
static void
pack_head(const struct HeadKey * hk, int64_t nh, const struct RankBudget * rb, int NTask, double * asg, struct HeadResult * hr)
{
    int64_t k;
    int r, d;
    for(k = 0; k < nh; k++) {
        int best = -1;
        double bestfree = -1;
        for(r = 0; r < NTask; r++) {
            int fits = 1;
            for(d = 0; d < PLAN_NDIM; d++)
                if(hk[k].dims[d] > rb[r].cap[d] - asg[PLAN_NDIM * r + d])
                    fits = 0;
            if(!fits)
                continue;
            const double freer = rb[r].cap[0] - asg[PLAN_NDIM * r];
            if(freer > bestfree) {
                bestfree = freer;
                best = r;
            }
        }
        hr[k].g = hk[k].g;
        hr[k].target = best;
        if(best >= 0)
            for(d = 0; d < PLAN_NDIM; d++)
                asg[PLAN_NDIM * best + d] += hk[k].dims[d];
    }
}

/* Oversized census of every rank and the global number of in-place keys. */
static int64_t
plan_ov_census(const struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloDirMsg * in, int NTask, int64_t * ovloc, int64_t * ovtot, MPI_Comm Comm)
{
    int64_t k, i;
    int t;
    int64_t nov = 0, NInplace = 0;
    memset(ovloc, 0, sizeof(int64_t) * PLAN_NOV * NTask);
    for(k = 0; k < ndir; k++) {
        if(!dk[k].oversized) continue;
        nov++;
        for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++) {
            const int rr = in[i].rank;
            for(t = 0; t < 6; t++) {
                ovloc[PLAN_NOV * rr + t] += in[i].cnt[t];
                ovloc[PLAN_NOV * rr + 6] += in[i].cnt[t];
            }
            ovloc[PLAN_NOV * rr + 7] += in[i].ntags;
            ovloc[PLAN_NOV * rr + 8] += 1;
        }
    }
    MPI_Allreduce(ovloc, ovtot, PLAN_NOV * NTask, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&nov, &NInplace, 1, MPI_INT64, MPI_SUM, Comm);
    return NInplace;
}

/* Budgets of every rank for the current oversized set. Collective. */
static void
plan_budgets(const struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloDirMsg * in, const struct SubhaloPlanRes * res,
             const struct SubhaloPlanCfg * cfg, int NTask, int ThisTask, int64_t * ovloc, int64_t * ovtot, struct RankBudget * rb, MPI_Comm Comm)
{
    const int64_t NInplace = plan_ov_census(dk, ndir, in, NTask, ovloc, ovtot, Comm);
    struct RankBudget mine;
    rank_budget(&mine, res, ovtot + PLAN_NOV * ThisTask, NInplace, cfg);
    MPI_Allgather(&mine, sizeof(struct RankBudget), MPI_BYTE, rb, sizeof(struct RankBudget), MPI_BYTE, Comm);
}

/* Per-destination loads of the current serial assignment against the budgets (H2). Fills
 * excess[PLAN_NDIM * r + d] (> 0 where over budget) and returns the number of destinations
 * over budget (identical on every rank). Collective. */
static int
plan_census(const struct SubhaloDirKey * dk, int64_t ndir, const struct RankBudget * rb, const struct SubhaloPlanCfg * cfg,
            int NTask, double * loadloc, double * excess, MPI_Comm Comm)
{
    int64_t k;
    int r, d;
    double dims[PLAN_NDIM];
    memset(loadloc, 0, sizeof(double) * PLAN_NDIM * NTask);
    for(k = 0; k < ndir; k++) {
        if(dk[k].oversized || dk[k].target < 0) continue;
        key_dims(cfg, &dk[k], dims);
        for(d = 0; d < PLAN_NDIM; d++)
            loadloc[PLAN_NDIM * dk[k].target + d] += dims[d];
    }
    MPI_Allreduce(loadloc, excess, PLAN_NDIM * NTask, MPI_DOUBLE, MPI_SUM, Comm);
    int nover = 0;
    for(r = 0; r < NTask; r++) {
        int over = 0;
        for(d = 0; d < PLAN_NDIM; d++) {
            const double e = excess[PLAN_NDIM * r + d] - rb[r].cap[d];
            excess[PLAN_NDIM * r + d] = e > 0 ? e : 0;
            if(e > 0)
                over = 1;
        }
        nover += over;
    }
    return nover;
}

static int
plan_covered(const double * removed, const double * excess)
{
    int d;
    for(d = 0; d < PLAN_NDIM; d++)
        if(excess[d] > 0 && removed[d] < excess[d])
            return 0;
    return 1;
}

static int
plan_contributes(const double * dims, const double * excess)
{
    int d;
    for(d = 0; d < PLAN_NDIM; d++)
        if(excess[d] > 0 && dims[d] > 0)
            return 1;
    return 0;
}

/* Repair of the over-budget destinations: for each, the serial keys assigned to it that
 * contribute to an exceeded dimension are moved in place smallest first (largest GrNr first:
 * FOF numbers groups by decreasing length) until every exceeded dimension is covered. The
 * walk is the sequential one in descending g: directory rank q starts from the totals of the
 * candidates on the ranks above it (all of them are removed unless the excess was already
 * covered there, in which case q removes nothing). Returns the number of keys moved on this
 * rank. Collective. */
static int64_t
plan_repair(struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloPlanCfg * cfg, int NTask, int ThisTask,
            const double * excess, MPI_Comm Comm)
{
    int64_t k;
    int d;
    double dims[PLAN_NDIM];
    double * cand = (double *) mymalloc("SubPlanCand", sizeof(double) * PLAN_NDIM * NTask);
    double * pre = (double *) mymalloc("SubPlanCandPre", sizeof(double) * PLAN_NDIM * NTask);
    double * tot = (double *) mymalloc("SubPlanCandTot", sizeof(double) * PLAN_NDIM * NTask);
    memset(cand, 0, sizeof(double) * PLAN_NDIM * NTask);
    memset(pre, 0, sizeof(double) * PLAN_NDIM * NTask);
    for(k = 0; k < ndir; k++) {
        if(dk[k].oversized || dk[k].target < 0) continue;
        const double * ex = excess + PLAN_NDIM * dk[k].target;
        key_dims(cfg, &dk[k], dims);
        if(!plan_contributes(dims, ex)) continue;
        for(d = 0; d < PLAN_NDIM; d++)
            cand[PLAN_NDIM * dk[k].target + d] += dims[d];
    }
    MPI_Exscan(cand, pre, PLAN_NDIM * NTask, MPI_DOUBLE, MPI_SUM, Comm);
    if(ThisTask == 0)
        memset(pre, 0, sizeof(double) * PLAN_NDIM * NTask);
    MPI_Allreduce(cand, tot, PLAN_NDIM * NTask, MPI_DOUBLE, MPI_SUM, Comm);
    /* removed so far when the walk reaches this rank: the candidates of the ranks above */
    int64_t i;
    for(i = 0; i < PLAN_NDIM * NTask; i++)
        tot[i] = tot[i] - pre[i] - cand[i];
    int64_t nmoved = 0;
    for(k = ndir - 1; k >= 0; k--) {
        if(dk[k].oversized || dk[k].target < 0) continue;
        const int r = dk[k].target;
        const double * ex = excess + PLAN_NDIM * r;
        key_dims(cfg, &dk[k], dims);
        if(!plan_contributes(dims, ex)) continue;
        double * removed = tot + PLAN_NDIM * r;
        if(plan_covered(removed, ex)) continue;
        dk[k].oversized = 1;
        dk[k].target = -1;
        nmoved++;
        for(d = 0; d < PLAN_NDIM; d++)
            removed[d] += dims[d];
    }
    myfree(tot);
    myfree(pre);
    myfree(cand);
    return nmoved;
}

void
subhalo_plan_pack(struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloDirMsg * in,
                  const struct SubhaloPlanRes * res, const struct SubhaloPlanCfg * cfg,
                  struct SubhaloPlanStats * stats, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t k, i;
    int r, t, d;
    const int64_t TotNgroups = cfg->TotNgroups;
    memset(stats, 0, sizeof(*stats));

    for(k = 0; k < ndir; k++) {
        dk[k].oversized = (cfg->CollectiveMinPart > 0 && dk[k].Length > cfg->CollectiveMinPart);
        dk[k].target = -1;
    }

    int64_t * ovloc = (int64_t *) mymalloc("SubOvLoc", sizeof(int64_t) * PLAN_NOV * NTask);
    int64_t * ovtot = (int64_t *) mymalloc("SubOvTot", sizeof(int64_t) * PLAN_NOV * NTask);
    struct RankBudget * rb = (struct RankBudget *) mymalloc("SubBudgets", sizeof(struct RankBudget) * NTask);
    double * asg = (double *) mymalloc("SubAssigned", sizeof(double) * PLAN_NDIM * NTask);
    double * loadloc = (double *) mymalloc("SubLoadLoc", sizeof(double) * PLAN_NDIM * NTask);
    double * excess = (double *) mymalloc("SubExcess", sizeof(double) * PLAN_NDIM * NTask);
    double * CE = (double *) mymalloc("SubCumCap", sizeof(double) * NTask);
    const int64_t HeadCap = (int64_t) SUBHALO_HEAD_PER_TASK * NTask < SUBHALO_HEAD_MAX ? (int64_t) SUBHALO_HEAD_PER_TASK * NTask : SUBHALO_HEAD_MAX;
    const int64_t H = TotNgroups < HeadCap ? TotNgroups : HeadCap;
    double dims[PLAN_NDIM];

    int converged = 0;
    int iter = 0;
    while(iter < cfg->MaxIter) {
        iter++;
        plan_budgets(dk, ndir, in, res, cfg, NTask, ThisTask, ovloc, ovtot, rb, Comm);

        /* keys that fit no rank in some dimension */
        double maxcap[PLAN_NDIM];
        for(d = 0; d < PLAN_NDIM; d++) {
            maxcap[d] = 0;
            for(r = 0; r < NTask; r++)
                if(rb[r].cap[d] > maxcap[d]) maxcap[d] = rb[r].cap[d];
        }
        int64_t newov = 0;
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized) continue;
            key_dims(cfg, &dk[k], dims);
            int over = 0;
            for(d = 0; d < PLAN_NDIM; d++)
                if(dims[d] > maxcap[d])
                    over = 1;
            if(over) {
                dk[k].oversized = 1;
                newov++;
            }
        }
        if(MPIU_Any(newov > 0, Comm))
            continue;

        for(k = 0; k < ndir; k++)
            dk[k].target = -1;

        if(cfg->ForceRankZero) {
            /* Debug: every serial key to rank 0, so that the exchange over-commits. Deliberately
             * not checked against the budgets. */
            for(k = 0; k < ndir; k++)
                if(!dk[k].oversized)
                    dk[k].target = 0;
            converged = 1;
            break;
        }

        /* head: exact greedy on rank 0 */
        int64_t nhloc = 0;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].g <= H)
                nhloc++;
        struct HeadKey * hloc = (struct HeadKey *) mymalloc("SubHeadLoc", sizeof(struct HeadKey) * (nhloc + 1));
        nhloc = 0;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].g <= H) {
                hloc[nhloc].g = dk[k].g;
                key_dims(cfg, &dk[k], hloc[nhloc].dims);
                nhloc++;
            }
        int64_t hbytes64 = nhloc * (int64_t) sizeof(struct HeadKey);
        int64_t htot = 0;
        MPI_Allreduce(&hbytes64, &htot, 1, MPI_INT64, MPI_SUM, Comm);
        subhalo_check_mpi_count(htot, "the plan's head-key gather", Comm);
        int * hcounts = ta_malloc("SubHeadCounts", int, NTask);
        int * hdispl = ta_malloc("SubHeadDispl", int, NTask);
        int hbytes = (int) hbytes64;
        MPI_Allgather(&hbytes, 1, MPI_INT, hcounts, 1, MPI_INT, Comm);
        int64_t hsum = 0;
        for(r = 0; r < NTask; r++) {
            hdispl[r] = (int) hsum;
            hsum += hcounts[r];
        }
        const int64_t nh = htot / sizeof(struct HeadKey);
        struct HeadKey * hall = (struct HeadKey *) mymalloc("SubHeadAll", htot + 1);
        MPI_Gatherv(hloc, hbytes, MPI_BYTE, hall, hcounts, hdispl, MPI_BYTE, 0, Comm);
        struct HeadResult * hr = (struct HeadResult *) mymalloc("SubHeadRes", sizeof(struct HeadResult) * (nh + 1));
        memset(asg, 0, sizeof(double) * PLAN_NDIM * NTask);
        if(ThisTask == 0)
            pack_head(hall, nh, rb, NTask, asg, hr);
        MPI_Bcast(hr, sizeof(struct HeadResult) * nh, MPI_BYTE, 0, Comm);
        MPI_Bcast(asg, PLAN_NDIM * NTask, MPI_DOUBLE, 0, Comm);
        int64_t j = 0;
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized || dk[k].g > H) continue;
            /* hr is in g order; the directory keys too */
            while(j < nh && hr[j].g < dk[k].g) j++;
            if(j >= nh || hr[j].g != dk[k].g)
                endrun(5, "Subhalo plan: head key %ld lost.\n", dk[k].g);
            dk[k].target = hr[j].target;
        }
        myfree(hr);
        myfree(hall);
        ta_free(hdispl);
        ta_free(hcounts);
        myfree(hloc);

        /* tail: the remaining budgets E_r, with the tail's type and byte mix taken as uniform
         * (the census below checks the real loads) */
        double tl[PLAN_NDIM], tmax[PLAN_NDIM];
        for(d = 0; d < PLAN_NDIM; d++)
            tl[d] = tmax[d] = 0;
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized || dk[k].g <= H) continue;
            key_dims(cfg, &dk[k], dims);
            for(d = 0; d < PLAN_NDIM; d++) {
                tl[d] += dims[d];
                if(dims[d] > tmax[d]) tmax[d] = dims[d];
            }
        }
        double TailN[PLAN_NDIM], TailMax[PLAN_NDIM];
        MPI_Allreduce(tl, TailN, PLAN_NDIM, MPI_DOUBLE, MPI_SUM, Comm);
        MPI_Allreduce(tmax, TailMax, PLAN_NDIM, MPI_DOUBLE, MPI_MAX, Comm);
        double cum = 0;
        for(r = 0; r < NTask; r++) {
            double E = 0;
            if(TailN[0] > 0) {
                E = 1e300;
                for(d = 0; d < PLAN_NDIM; d++) {
                    if(TailN[d] <= 0 || rb[r].cap[d] >= 1e29) continue;
                    const double Ed = (rb[r].cap[d] - asg[PLAN_NDIM * r + d] - TailMax[d]) * TailN[0] / TailN[d];
                    if(Ed < E) E = Ed;
                }
            }
            if(E < 0) E = 0;
            cum += E;
            CE[r] = cum;
        }
        /* The keys whose start (prefix of lengths in GrNr order) lies below the total capacity
         * are placed; each goes to the rank whose share of the cumulative capacity contains a
         * low-discrepancy point of its tail index (golden-ratio Weyl sequence). Every rank then
         * receives close to its share E_r of the tail, and the keys it receives are spread over
         * the whole tail rather than being one contiguous GrNr range, so a type mix that
         * correlates with GrNr (gas groups first, say) reaches every rank in the tail's average
         * proportion, which is what E_r assumed. The census below checks the residuals. */
        double pfx = 0;
        MPI_Exscan(&tl[0], &pfx, 1, MPI_DOUBLE, MPI_SUM, Comm);
        if(ThisTask == 0)
            pfx = 0;
        int64_t ntl = 0, jt = 0;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].g > H)
                ntl++;
        MPI_Exscan(&ntl, &jt, 1, MPI_INT64, MPI_SUM, Comm);
        if(ThisTask == 0)
            jt = 0;
        const double CEtot = CE[NTask - 1];
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized || dk[k].g <= H) continue;
            if(pfx >= CEtot)
                dk[k].target = -1;
            else {
                double w = ((double) jt + 0.5) * 0.6180339887498949;
                w = (w - floor(w)) * CEtot;
                int lo_r = 0, hi_r = NTask;
                while(lo_r < hi_r) {
                    int mid = (lo_r + hi_r) / 2;
                    if(CE[mid] > w) hi_r = mid;
                    else lo_r = mid + 1;
                }
                dk[k].target = lo_r < NTask ? lo_r : NTask - 1;
            }
            pfx += dk[k].Length;
            jt++;
        }

        /* keys that could not be placed become oversized; repeat if any */
        newov = 0;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].target < 0) {
                dk[k].oversized = 1;
                newov++;
            }
        if(MPIU_Any(newov > 0, Comm))
            continue;

        /* the proposal is stable: no key became oversized with the budgets it was packed with */
        converged = 1;
        break;
    }
    if(!converged) {
        /* Iteration cap: keep the last targets, move what could not be placed in place; the
         * census below checks everything against the final budgets. */
        stats->Fallback = 1;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && !cfg->ForceRankZero && dk[k].target < 0)
                dk[k].oversized = 1;
    }
    /* The census of the proposal against the budgets of the final staying particles (H2): the
     * tail's uniform mix and the passes' budget changes are only estimates. Over-budget
     * destinations shed keys in place (smallest first), which changes the holders' budgets, so
     * the census is repeated until it passes. Each round moves at least one key, so this
     * terminates; the targets of the other keys are kept (no repacking). The debug
     * ForceRankZero plan is deliberately left unchecked. */
    while(!cfg->ForceRankZero) {
        plan_budgets(dk, ndir, in, res, cfg, NTask, ThisTask, ovloc, ovtot, rb, Comm);
        if(plan_census(dk, ndir, rb, cfg, NTask, loadloc, excess, Comm) == 0)
            break;
        int64_t nmoved = plan_repair(dk, ndir, cfg, NTask, ThisTask, excess, Comm), nmovedtot = 0;
        MPI_Allreduce(&nmoved, &nmovedtot, 1, MPI_INT64, MPI_SUM, Comm);
        if(nmovedtot == 0)
            endrun(5, "Subhalo plan: a destination is over budget but no key could be moved.\n");
        stats->NRepaired += nmovedtot;
        stats->CensusRounds++;
    }
    stats->Iterations = iter;
    for(k = 0; k < ndir; k++)
        if(dk[k].oversized)
            dk[k].target = -1;

    /* diagnostics */
    {
        int64_t loc[3] = {0, 0, 0}, tot[3];
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized) loc[1]++;
            else {
                loc[2]++;
                if(dk[k].Length > loc[0]) loc[0] = dk[k].Length;
            }
        }
        MPI_Allreduce(&loc[0], &tot[0], 1, MPI_INT64, MPI_MAX, Comm);
        MPI_Allreduce(&loc[1], &tot[1], 2, MPI_INT64, MPI_SUM, Comm);
        stats->LargestSerial = tot[0];
        stats->NOversized = tot[1];
        stats->NSerial = tot[2];
        stats->MinBudget = rb[0].cap[0];
        stats->MinByteBudget = rb[0].cap[7];
        stats->MinFreeBytes = rb[0].freebytes;
        for(r = 1; r < NTask; r++) {
            if(rb[r].cap[0] < stats->MinBudget) stats->MinBudget = rb[r].cap[0];
            if(rb[r].cap[7] < stats->MinByteBudget) stats->MinByteBudget = rb[r].cap[7];
            if(rb[r].freebytes < stats->MinFreeBytes) stats->MinFreeBytes = rb[r].freebytes;
        }
    }
    (void) i; (void) t;
    myfree(CE);
    myfree(excess);
    myfree(loadloc);
    myfree(asg);
    myfree(rb);
    myfree(ovtot);
    myfree(ovloc);
}

/* A particle of a FOF group for the census: key, type and tag. */
struct CensusPart {
    int64_t g;
    int64_t TrackId;
    int64_t type;
};

static int
cmp_censuspart(const void * a, const void * b)
{
    const struct CensusPart * x = (const struct CensusPart *) a, * y = (const struct CensusPart *) b;
    if(x->g != y->g)
        return (x->g > y->g) - (x->g < y->g);
    return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
}

void
subhalo_plan_build(struct SubhaloPlan * plan, const FOFGroups * fof, MPI_Comm Comm)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t i, k;
    int r, t;

    memset(plan, 0, sizeof(struct SubhaloPlan));
    plan->NTask = NTask;
    plan->ThisTask = ThisTask;
    plan->TotNgroups = fof->TotNgroups;
    plan->KChunk = subhalo_key_chunk(fof->TotNgroups, NTask);
    const int64_t KChunk = plan->KChunk;
    const int64_t TotNgroups = fof->TotNgroups;

    /* ---- local census: (GrNr, type) counts and distinct tags of the particles in groups ---- */
    struct SubhaloPlanRes res;
    memset(&res, 0, sizeof(res));
    int64_t ngrp = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        res.NumLive++;
        if(P[i].GrNr >= 0) ngrp++;
    }
    /* On the top so that it can be freed while the census messages (bottom) live on. */
    struct CensusPart * gt = (struct CensusPart *) mymalloc2("SubCensusKeys", sizeof(struct CensusPart) * (ngrp + 1));
    ngrp = 0;
    int badgrnr = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage || P[i].GrNr < 0) continue;
        if(P[i].GrNr < 1 || P[i].GrNr > TotNgroups)
            badgrnr = 1;
        gt[ngrp].g = P[i].GrNr;
        gt[ngrp].TrackId = P[i].TrackId;
        gt[ngrp].type = P[i].Type;
        ngrp++;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: particle GrNr outside [1, %ld].\n", TotNgroups);
    qsort_openmp(gt, ngrp, sizeof(struct CensusPart), cmp_censuspart);
    int64_t ncensus = 0;
    for(i = 0; i < ngrp; i++)
        if(i == 0 || gt[i].g != gt[i-1].g)
            ncensus++;
    const int64_t nmsg = ncensus + fof->Ngroups;
    struct SubhaloDirMsg * msg = (struct SubhaloDirMsg *) mymalloc("SubDirMsg", sizeof(struct SubhaloDirMsg) * (nmsg + 1));
    int64_t m = -1;
    for(i = 0; i < ngrp; i++) {
        const int64_t g = gt[i].g;
        if(m < 0 || msg[m].g != g) {
            m++;
            memset(&msg[m], 0, sizeof(struct SubhaloDirMsg));
            msg[m].g = g;
            msg[m].rank = ThisTask;
            msg[m].kind = 0;
        }
        msg[m].cnt[gt[i].type]++;
        /* distinct tags: the particles of g are sorted by TrackId */
        if(gt[i].TrackId >= 0 && (i == 0 || gt[i-1].g != g || gt[i-1].TrackId != gt[i].TrackId))
            msg[m].ntags++;
        res.grp[gt[i].type]++;
        res.grp[6]++;
    }
    myfree(gt);
    for(i = 0; i < fof->Ngroups; i++) {
        const struct Group * G = &fof->Group[i];
        struct SubhaloDirMsg * d = &msg[ncensus + i];
        memset(d, 0, sizeof(struct SubhaloDirMsg));
        d->g = G->base.GrNr;
        for(t = 0; t < 6; t++)
            d->cnt[t] = G->LenType[t];
        for(t = 0; t < 3; t++)
            d->ref[t] = G->base.FirstPos[t];
        d->rank = ThisTask;
        d->kind = 1;
        if(d->g < 1 || d->g > TotNgroups)
            badgrnr = 1;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: group GrNr outside [1, %ld].\n", TotNgroups);
    qsort(msg, nmsg, sizeof(struct SubhaloDirMsg), cmp_dirmsg);
    int64_t * sendcounts = ta_malloc("SubPlanCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nmsg; i++)
        sendcounts[subhalo_key_dir_rank(msg[i].g, KChunk)]++;
    struct SubhaloDirMsg * in = NULL;
    int64_t nin = subhalo_alltoallv(msg, sendcounts, sizeof(struct SubhaloDirMsg), (void **) &in, 0, NULL, Comm);
    qsort(in, nin, sizeof(struct SubhaloDirMsg), cmp_dirmsg);

    /* ---- the directory: keys g in [lo, lo + ndir) ---- */
    const int64_t lo = (int64_t) ThisTask * KChunk + 1;
    int64_t ndir = TotNgroups - lo + 1;
    if(ndir < 0) ndir = 0;
    if(ndir > KChunk) ndir = KChunk;
    struct SubhaloDirKey * dk = (struct SubhaloDirKey *) mymalloc("SubDirKeys", sizeof(struct SubhaloDirKey) * (ndir + 1));
    for(k = 0; k < ndir; k++) {
        memset(&dk[k], 0, sizeof(struct SubhaloDirKey));
        dk[k].g = lo + k;
        dk[k].cstart = -1;
        dk[k].decider = -1;
        dk[k].target = -1;
    }
    int baddir = 0;
    for(i = 0; i < nin; i++) {
        const int64_t kk = in[i].g - lo;
        if(kk < 0 || kk >= ndir) {
            baddir = 1;
            continue;
        }
        struct SubhaloDirKey * d = &dk[kk];
        if(in[i].kind == 1) {
            if(d->has_group)
                baddir = 1;
            d->has_group = 1;
            d->Length = 0;
            for(t = 0; t < 6; t++) {
                d->LenType[t] = in[i].cnt[t];
                d->Length += in[i].cnt[t];
            }
            for(t = 0; t < 3; t++)
                d->ref[t] = in[i].ref[t];
        }
        else {
            /* census entries sort before the group record (kind 0 < 1), by rank */
            if(d->cstart < 0) {
                d->cstart = i;
                d->decider = in[i].rank;
            }
            d->ccount++;
            d->Tags += in[i].ntags;
        }
    }
    for(k = 0; k < ndir; k++) {
        if(!dk[k].has_group) {
            baddir = 1;
            continue;
        }
        int64_t sum = 0;
        for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++)
            for(t = 0; t < 6; t++)
                sum += in[i].cnt[t];
        if(sum != dk[k].Length)
            baddir = 1;
    }
    if(baddir)
        endrun(5, "Subhalo plan: the key directory on rank %d is inconsistent with the FOF groups.\n", ThisTask);

    /* ---- classification and packing (particle-free, subhalo_plan_pack) ---- */
    res.MaxPart = PartManager->MaxPart;
    for(t = 0; t < 6; t++) {
        const struct slot_info * info = &SlotsManager->info[t];
        res.slotelsize[t] = info->enabled ? (int64_t) info->elsize : 0;
        res.slotfree[t] = info->enabled ? info->maxsize - info->size : 0;
    }
    res.freebytes = (double) mymalloc_freebytes();
    struct SubhaloPlanCfg cfg;
    subhalo_plan_default_cfg(&cfg, TotNgroups, NTask);
    subhalo_plan_pack(dk, ndir, in, &res, &cfg, &plan->Stats, Comm);
    message(0, "Subhalo plan: %ld serial keys (largest %ld), %ld oversized keys in place (%ld moved by %d census round(s)); "
               "min budget %g particles, min byte budget %g MB, min free bytes %g MB; %d pass(es)%s%s.\n",
            plan->Stats.NSerial, plan->Stats.LargestSerial, plan->Stats.NOversized, plan->Stats.NRepaired, plan->Stats.CensusRounds,
            plan->Stats.MinBudget, plan->Stats.MinByteBudget / (1024. * 1024.), plan->Stats.MinFreeBytes / (1024. * 1024.), plan->Stats.Iterations,
            plan->Stats.Fallback ? " [pass cap reached]" : "",
            sp->SubhaloDebugForceExchangeFailure ? " [debug: every serial key sent to rank 0]" : "");

    /* ---- replies (top of the stack, in this order: LocalKeys, Assigned, Inplace) ---- */
    /* 1. every holder learns the fate of the keys it holds */
    int64_t nrep = 0;
    for(k = 0; k < ndir; k++)
        nrep += dk[k].ccount;
    struct SubhaloLocalKey * rep = (struct SubhaloLocalKey *) mymalloc("SubPlanRep", sizeof(struct SubhaloLocalKey) * (nrep + 1));
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(k = 0; k < ndir; k++)
        for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++)
            sendcounts[in[i].rank]++;
    int64_t * off = ta_malloc("SubPlanOff", int64_t, NTask);
    off[0] = 0;
    for(r = 1; r < NTask; r++)
        off[r] = off[r-1] + sendcounts[r-1];
    for(k = 0; k < ndir; k++)
        for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++) {
            struct SubhaloLocalKey * lk = &rep[off[in[i].rank]++];
            memset(lk, 0, sizeof(*lk));
            lk->g = dk[k].g;
            lk->Length = dk[k].Length;
            lk->oversized = dk[k].oversized;
            lk->target = dk[k].oversized ? in[i].rank : dk[k].target;
            lk->decider = dk[k].decider;
            for(t = 0; t < 3; t++)
                lk->ref[t] = dk[k].ref[t];
        }
    plan->NLocalKeys = subhalo_alltoallv(rep, sendcounts, sizeof(struct SubhaloLocalKey), (void **) &plan->LocalKeys, 1, NULL, Comm);
    /* from directory ranks in rank order: already sorted by g; make sure */
    qsort(plan->LocalKeys, plan->NLocalKeys, sizeof(struct SubhaloLocalKey), cmp_localkey_g);

    /* 2. every processing rank learns its serial keys */
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    int64_t nser = 0;
    for(k = 0; k < ndir; k++)
        if(!dk[k].oversized) {
            sendcounts[dk[k].target]++;
            nser++;
        }
    struct SubhaloAssignedKey * as = (struct SubhaloAssignedKey *) mymalloc("SubPlanAs", sizeof(struct SubhaloAssignedKey) * (nser + 1));
    off[0] = 0;
    for(r = 1; r < NTask; r++)
        off[r] = off[r-1] + sendcounts[r-1];
    for(k = 0; k < ndir; k++)
        if(!dk[k].oversized) {
            struct SubhaloAssignedKey * a = &as[off[dk[k].target]++];
            a->g = dk[k].g;
            a->Length = dk[k].Length;
        }
    plan->NAssigned = subhalo_alltoallv(as, sendcounts, sizeof(struct SubhaloAssignedKey), (void **) &plan->Assigned, 1, NULL, Comm);
    qsort(plan->Assigned, plan->NAssigned, sizeof(struct SubhaloAssignedKey), cmp_assigned_g);

    /* 3. the global list of oversized keys */
    int64_t novloc = 0;
    for(k = 0; k < ndir; k++)
        if(dk[k].oversized)
            novloc++;
    struct SubhaloInplaceKey * ipl = (struct SubhaloInplaceKey *) mymalloc("SubPlanIpl", sizeof(struct SubhaloInplaceKey) * (novloc + 1));
    novloc = 0;
    for(k = 0; k < ndir; k++)
        if(dk[k].oversized) {
            struct SubhaloInplaceKey * e = &ipl[novloc++];
            memset(e, 0, sizeof(*e));
            e->g = dk[k].g;
            e->Length = dk[k].Length;
            e->decider = dk[k].decider;
            for(t = 0; t < 3; t++)
                e->ref[t] = dk[k].ref[t];
        }
    /* byte counts of the Allgatherv: the global total is checked before any narrowing (H3) */
    const int64_t ibytes64 = novloc * (int64_t) sizeof(struct SubhaloInplaceKey);
    int64_t itot = 0;
    MPI_Allreduce(&ibytes64, &itot, 1, MPI_INT64, MPI_SUM, Comm);
    subhalo_check_mpi_count(itot, "the in-place key list (bytes)", Comm);
    int * icounts = ta_malloc("SubIplCounts", int, NTask);
    int * idispl = ta_malloc("SubIplDispl", int, NTask);
    int ibytes = (int) ibytes64;
    MPI_Allgather(&ibytes, 1, MPI_INT, icounts, 1, MPI_INT, Comm);
    int64_t isum = 0;
    for(r = 0; r < NTask; r++) {
        idispl[r] = (int) isum;
        isum += icounts[r];
    }
    plan->Inplace = (struct SubhaloInplaceKey *) mymalloc2("SubInplaceKeys", itot + 1);
    MPI_Allgatherv(ipl, ibytes, MPI_BYTE, plan->Inplace, icounts, idispl, MPI_BYTE, Comm);
    plan->NInplace = itot / sizeof(struct SubhaloInplaceKey);
    qsort(plan->Inplace, plan->NInplace, sizeof(struct SubhaloInplaceKey), cmp_inplace_g);
    ta_free(idispl);
    ta_free(icounts);
    myfree(ipl);
    myfree(as);
    ta_free(off);
    myfree(rep);

    myfree(dk);
    myfree(in);
    ta_free(sendcounts);
    myfree(msg);

    /* ---- targets into P[i].TopLeaf (the layout of the exchange) ---- */
    int64_t lost = 0;
    #pragma omp parallel for reduction(+: lost)
    for(i = 0; i < PartManager->NumPart; i++) {
        P[i].TopLeaf = ThisTask;
        if(P[i].IsGarbage || P[i].GrNr < 0)
            continue;
        struct SubhaloLocalKey key;
        key.g = P[i].GrNr;
        const struct SubhaloLocalKey * lk = (const struct SubhaloLocalKey *)
            bsearch(&key, plan->LocalKeys, plan->NLocalKeys, sizeof(struct SubhaloLocalKey), cmp_localkey_g);
        if(!lk) {
            lost++;
            continue;
        }
        P[i].TopLeaf = lk->oversized ? ThisTask : lk->target;
    }
    if(lost)
        endrun(5, "Subhalo plan: %ld particles on rank %d have no plan entry.\n", lost, ThisTask);
}

void
subhalo_plan_free(struct SubhaloPlan * plan)
{
    if(plan->Inplace)
        myfree(plan->Inplace);
    if(plan->Assigned)
        myfree(plan->Assigned);
    if(plan->LocalKeys)
        myfree(plan->LocalKeys);
    plan->Inplace = NULL;
    plan->Assigned = NULL;
    plan->LocalKeys = NULL;
}

/* ------------------------------------------------------------------ step 4: exchange */

static int
subhalo_layout(int i, const void * userdata)
{
    const struct part_manager_type * pman = (const struct part_manager_type *) userdata;
    /* Reuse the topleaf to hold the target task, as fof_sorted_layout does. The domain is
     * restored with domain_maintain afterwards. */
    return pman->Base[i].TopLeaf;
}

int
subhalo_plan_exchange(struct SubhaloPlan * plan, MPI_Comm Comm)
{
    int failure = domain_exchange(subhalo_layout, PartManager, NULL, PartManager, SlotsManager, 10000, Comm);
    if(failure)
        message(1, "Subhalo exchange did not complete on this rank.\n");
    if(subhalo_get_params()->SubhaloDebugForceExchangeFailure) {
        message(0, "Subhalo debug: forcing the exchange failure path.\n");
        failure = 1;
    }
    return failure;
}

/* ------------------------------------------------------------------ local sort and segments */

static int
order_by_key_track_rank(const void * a, const void * b)
{
    const struct particle_data * pa = (const struct particle_data *) a;
    const struct particle_data * pb = (const struct particle_data *) b;
    if(pa->IsGarbage != pb->IsGarbage)
        return (pa->IsGarbage > pb->IsGarbage) - (pa->IsGarbage < pb->IsGarbage);
    if(pa->GrNr != pb->GrNr)
        return (pa->GrNr > pb->GrNr) - (pa->GrNr < pb->GrNr);
    if(pa->TrackId != pb->TrackId)
        return (pa->TrackId > pb->TrackId) - (pa->TrackId < pb->TrackId);
    if(pa->BindRank != pb->BindRank)
        return (pa->BindRank > pb->BindRank) - (pa->BindRank < pb->BindRank);
    /* the ID makes the order, and every sum over it, independent of the arrival order */
    return (pa->ID > pb->ID) - (pa->ID < pb->ID);
}

void
subhalo_local_sort(void)
{
    qsort_openmp(PartManager->Base, PartManager->NumPart, sizeof(struct particle_data), order_by_key_track_rank);
}

int64_t
subhalo_build_segments(const struct SubhaloPlan * plan, struct SubhaloKeySeg ** segs)
{
    int64_t i, n = 0;
    const int64_t NumPart = PartManager->NumPart;
    for(i = 0; i < NumPart && !P[i].IsGarbage; i++)
        if(P[i].GrNr >= 0 && (i == 0 || P[i].GrNr != P[i-1].GrNr))
            n++;
    struct SubhaloKeySeg * s = (struct SubhaloKeySeg *) mymalloc2("SubKeySegs", sizeof(struct SubhaloKeySeg) * (n + 1));
    int64_t k = 0, nserial = 0;
    int bad = 0;
    for(i = 0; i < NumPart && !P[i].IsGarbage; ) {
        if(P[i].GrNr < 0) {
            i++;
            continue;
        }
        int64_t j = i;
        while(j < NumPart && !P[j].IsGarbage && P[j].GrNr == P[i].GrNr)
            j++;
        struct SubhaloKeySeg * seg = &s[k++];
        memset(seg, 0, sizeof(*seg));
        seg->g = P[i].GrNr;
        seg->start = i;
        seg->n = j - i;
        seg->ipindex = -1;
        struct SubhaloAssignedKey akey = {seg->g, 0};
        const struct SubhaloAssignedKey * a = (const struct SubhaloAssignedKey *)
            bsearch(&akey, plan->Assigned, plan->NAssigned, sizeof(akey), cmp_assigned_g);
        if(a) {
            seg->Length = a->Length;
            seg->inplace = 0;
            seg->decider = plan->ThisTask;
            if(seg->n != seg->Length)
                bad = 1;
            nserial++;
        }
        else {
            struct SubhaloInplaceKey ikey;
            ikey.g = seg->g;
            const struct SubhaloInplaceKey * ip = (const struct SubhaloInplaceKey *)
                bsearch(&ikey, plan->Inplace, plan->NInplace, sizeof(ikey), cmp_inplace_g);
            if(!ip)
                bad = 1;
            else {
                seg->Length = ip->Length;
                seg->inplace = 1;
                seg->decider = ip->decider;
                seg->ipindex = ip - plan->Inplace;
                int d;
                for(d = 0; d < 3; d++)
                    seg->ref[d] = ip->ref[d];
            }
        }
        i = j;
    }
    if(nserial != plan->NAssigned)
        bad = 1;
    if(bad)
        endrun(5, "Subhalo exchange left rank %d with incomplete keys (%ld serial runs for %ld assigned keys).\n",
               plan->ThisTask, nserial, plan->NAssigned);
    *segs = s;
    return n;
}

/* ------------------------------------------------------------------ step 7a: member records */

static int
cmp_int64(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

int64_t
subhalo_fetch_members(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubTrackTable * tab, struct SubTrack ** members, MPI_Comm Comm)
{
    int64_t k, i, n = 0;
    for(k = 0; k < nsegs; k++)
        for(i = segs[k].start; i < segs[k].start + segs[k].n; i++)
            if(P[i].TrackId >= 0 && (i == segs[k].start || P[i].TrackId != P[i-1].TrackId))
                n++;
    int64_t * ids = (int64_t *) mymalloc("SubMemberIds", sizeof(int64_t) * (n + 1));
    n = 0;
    for(k = 0; k < nsegs; k++)
        for(i = segs[k].start; i < segs[k].start + segs[k].n; i++)
            if(P[i].TrackId >= 0 && (i == segs[k].start || P[i].TrackId != P[i-1].TrackId))
                ids[n++] = P[i].TrackId;
    qsort(ids, n, sizeof(int64_t), cmp_int64);
    int64_t nu = 0;
    for(i = 0; i < n; i++)
        if(nu == 0 || ids[i] != ids[nu-1])
            ids[nu++] = ids[i];
    struct SubTrack * mem = (struct SubTrack *) mymalloc2("SubMembers", sizeof(struct SubTrack) * (nu + 1));
    subhalo_table_fetch(tab, ids, nu, mem, Comm);
    myfree(ids);
    *members = mem;
    return nu;
}

/* ------------------------------------------------------------------ step 9: domain restore */

void
subhalo_restore_domain(DomainDecomp * ddecomp)
{
    /* Peano-sort first so that one exchange iteration does not overfill a rank. */
    slots_gc_sorted(PartManager, SlotsManager);
    if(domain_maintain(ddecomp, NULL))
        endrun(0, "Domain exchange after the subhalo finder did not complete!\n");
    slots_gc_sorted(PartManager, SlotsManager);
}

/* ------------------------------------------------------------------ per-thread arenas (design 7.1) */

struct KeyOrder {
    int64_t n;
    int64_t g;
    int64_t k;
};

static int
cmp_keyorder(const void * a, const void * b)
{
    const struct KeyOrder * x = (const struct KeyOrder *) a, * y = (const struct KeyOrder *) b;
    /* longest first, ties by GrNr */
    if(x->n != y->n)
        return (x->n < y->n) - (x->n > y->n);
    return (x->g > y->g) - (x->g < y->g);
}

void
subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, subhalo_key_fn fn, void * userdata)
{
    int64_t k, ns = 0, total = 0;
    int t;
    const int nthreads = omp_get_max_threads();
    for(k = 0; k < nsegs; k++)
        if(!segs[k].inplace) {
            ns++;
            total += segs[k].n;
        }
    struct KeyOrder * ord = (struct KeyOrder *) mymalloc("SubKeyOrder", sizeof(struct KeyOrder) * (ns + 1));
    ns = 0;
    for(k = 0; k < nsegs; k++)
        if(!segs[k].inplace) {
            ord[ns].n = segs[k].n;
            ord[ns].g = segs[k].g;
            ord[ns].k = k;
            ns++;
        }
    qsort(ord, ns, sizeof(struct KeyOrder), cmp_keyorder);

    /* keys above one thread's share run alone, with every thread inside the key */
    const int64_t share = (total + nthreads - 1) / nthreads;
    int64_t nalone = 0;
    while(nthreads > 1 && nalone < ns && ord[nalone].n > share)
        nalone++;
    if(nalone > 0) {
        struct SubhaloArena arena;
        arena.size = bytes_per_part * ord[0].n + 64;
        arena.base = (char *) mymalloc2("SubArenaAlone", arena.size);
        for(k = 0; k < nalone; k++) {
            arena.used = 0;
            fn(&segs[ord[k].k], ord[k].k, &arena, nthreads, userdata);
        }
        myfree(arena.base);
    }

    /* the rest: static longest-processing-time-first assignment to threads */
    int64_t * load = (int64_t *) mymalloc("SubThreadLoad", sizeof(int64_t) * nthreads);
    int64_t * maxn = (int64_t *) mymalloc("SubThreadMaxN", sizeof(int64_t) * nthreads);
    int64_t * count = (int64_t *) mymalloc("SubThreadCount", sizeof(int64_t) * nthreads);
    int * owner = (int *) mymalloc("SubKeyOwner", sizeof(int) * (ns + 1));
    for(t = 0; t < nthreads; t++)
        load[t] = maxn[t] = count[t] = 0;
    for(k = nalone; k < ns; k++) {
        int best = 0;
        for(t = 1; t < nthreads; t++)
            if(load[t] < load[best])
                best = t;
        owner[k] = best;
        load[best] += ord[k].n;
        count[best]++;
        if(ord[k].n > maxn[best])
            maxn[best] = ord[k].n;
    }
    struct SubhaloArena * arenas = (struct SubhaloArena *) mymalloc("SubArenas", sizeof(struct SubhaloArena) * nthreads);
    for(t = 0; t < nthreads; t++) {
        arenas[t].size = bytes_per_part * maxn[t] + 64;
        arenas[t].used = 0;
        arenas[t].base = (char *) mymalloc2("SubArena", arenas[t].size);
    }
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        int tt;
        /* if the runtime gives fewer threads, each covers several queues */
        for(tt = tid; tt < nthreads; tt += nt) {
            int64_t kk;
            for(kk = nalone; kk < ns; kk++) {
                if(owner[kk] != tt) continue;
                arenas[tt].used = 0;
                fn(&segs[ord[kk].k], ord[kk].k, &arenas[tt], 1, userdata);
            }
        }
    }
    for(t = nthreads - 1; t >= 0; t--)
        myfree(arenas[t].base);
    myfree(arenas);
    myfree(owner);
    myfree(count);
    myfree(maxn);
    myfree(load);
    myfree(ord);
}
