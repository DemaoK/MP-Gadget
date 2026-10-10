/* Runtime subhalo finder: tracer snapshot and host vote, previous-bound frames, track keys,
 * the plan (field-particle keys, pseudo-keys, budgets, oversized keys, packing, orphan
 * routing), the whole-key particle exchange, the local sort, the member fetch, the domain
 * restore and the per-thread scratch arenas.
 *
 * Phase 1 plan (halofinder_phase1_plan.md) section 1 steps 2-5, 7a and 9; Phase 2a plan
 * (halofinder_phase2a_plan.md) section 1 steps 0, 2 and 3; design 7.1, 7.2, 7.3.
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

static int
cmp_int64(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

/* ------------------------------------------------------------------ steps 0 and 2: tracers and host vote */

/* The live-particle predicate of the finder (H6): garbage and swallowed particles neither vote
 * nor count, but their tags are left untouched. */
static inline int
part_live(const struct particle_data * p)
{
    return !p->IsGarbage && !p->Swallowed;
}

/* A tracer tuple of step 0. */
struct TracerTuple {
    int64_t TrackId;
    int64_t GrNr;
    MyIDType ID;
    int32_t BindRank;
    int32_t pad;
};

static int
cmp_tracertuple(const void * a, const void * b)
{
    const struct TracerTuple * x = (const struct TracerTuple *) a, * y = (const struct TracerTuple *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    if(x->BindRank != y->BindRank)
        return (x->BindRank > y->BindRank) - (x->BindRank < y->BindRank);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* Keep the first K tuples of every track of a (TrackId, BindRank, ID)-sorted array. */
static int64_t
keep_first_tracers(struct TracerTuple * v, int64_t n, int K)
{
    int64_t i, m = 0, run = 0;
    for(i = 0; i < n; i++) {
        if(i == 0 || v[i].TrackId != v[i-1].TrackId)
            run = 0;
        if(run++ < K)
            v[m++] = v[i];
    }
    return m;
}

void
subhalo_host_vote(struct SubTrackTable * tab, MPI_Comm Comm)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    const int K = sp->SubhaloNumTracerHostFinding;
    const int TracerTypes = sp->SubhaloTracerTypes;
    int64_t i, n = 0;
    const int NTask = tab->NTask;
    if(K < 1 || K > SUBHALO_MAXTRACERS)
        endrun(1, "SubhaloNumTracerHostFinding = %d outside [1, %d].\n", K, SUBHALO_MAXTRACERS);
    /* own-bound live tracer-type particles (H6: a swallowed BH neither votes nor counts) */
    for(i = 0; i < PartManager->NumPart; i++) {
        const struct particle_data * p = &P[i];
        if(part_live(p) && p->TrackId >= 0 && p->BoundTrackId == p->TrackId && ((TracerTypes >> p->Type) & 1))
            n++;
    }
    struct TracerTuple * v = (struct TracerTuple *) mymalloc("SubTracers", sizeof(struct TracerTuple) * (n + 1));
    n = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        const struct particle_data * p = &P[i];
        if(part_live(p) && p->TrackId >= 0 && p->BoundTrackId == p->TrackId && ((TracerTypes >> p->Type) & 1)) {
            v[n].TrackId = p->TrackId;
            v[n].GrNr = p->GrNr >= 0 ? p->GrNr : -1;
            v[n].ID = p->ID;
            v[n].BindRank = p->BindRank;
            v[n].pad = 0;
            n++;
        }
    }
    qsort_openmp(v, n, sizeof(struct TracerTuple), cmp_tracertuple);
    n = keep_first_tracers(v, n, K);

    /* sorted by TrackId, hence by home rank */
    int64_t * sendcounts = ta_malloc("SubVoteCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < n; i++)
        sendcounts[subhalo_table_home_rank(tab, v[i].TrackId)]++;
    struct TracerTuple * in = NULL;
    int64_t nin = subhalo_alltoallv(v, sendcounts, sizeof(struct TracerTuple), (void **) &in, 0, NULL, Comm);
    qsort_openmp(in, nin, sizeof(struct TracerTuple), cmp_tracertuple);
    nin = keep_first_tracers(in, nin, K);

    /* HBT DecideLocalHostId (subhalo_tracking.cpp:294-325): weight 1 / (1 + sqrt(i)) for the
     * i-th tracer in rank order; the GrNr (-1: no group) with the largest sum wins, ties to the
     * smaller GrNr. A track without any tuple keeps Host = -1 (hostless, or dead if it has no
     * particle left). */
    int64_t a = 0, nhosted = 0, nfield = 0;
    int missing = 0;
    while(a < nin) {
        int64_t b = a;
        while(b < nin && in[b].TrackId == in[a].TrackId)
            b++;
        int64_t cand[SUBHALO_MAXTRACERS];
        double w[SUBHALO_MAXTRACERS];
        int nc = 0, j, c;
        struct SubTrack * rec = subhalo_table_get(tab, in[a].TrackId);
        for(j = 0; j < b - a; j++) {
            const int64_t g = in[a + j].GrNr;
            for(c = 0; c < nc && cand[c] != g; c++) {}
            if(c == nc) {
                cand[nc] = g;
                w[nc] = 0;
                nc++;
            }
            w[c] += 1.0 / (1.0 + sqrt((double) j));
            if(rec)
                rec->TracerIDs[j] = in[a + j].ID;
        }
        int best = 0;
        for(c = 1; c < nc; c++)
            if(w[c] > w[best] || (w[c] == w[best] && cand[c] < cand[best]))
                best = c;
        if(!rec)
            missing = 1;
        else {
            rec->NTracers = (int32_t) (b - a);
            rec->Host = cand[best];
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
    message(0, "Subhalo host vote: %ld tracks hosted by a FOF group, %ld with no host group (tracer vote).\n", tot[0], tot[1]);
}

/* ------------------------------------------------------------------ step 3: previous-bound frames */

/* Stable merge sort of fixed-size records by the int64 at their start: contributions of one
 * track keep the source-rank order of an Alltoallv receive buffer, so a home rank sums them in
 * rank order. */
static int
cmp_lead_int64(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

static void
stable_sort_lead(void * base, int64_t n, size_t elsize)
{
    char * in = (char *) base;
    char * tmp = (char *) mymalloc("SubStableTmp", elsize * (n + 1));
    int64_t width;
    for(width = 1; width < n; width *= 2) {
        int64_t lo;
        for(lo = 0; lo < n; lo += 2 * width) {
            int64_t mid = lo + width < n ? lo + width : n;
            int64_t hi = lo + 2 * width < n ? lo + 2 * width : n;
            int64_t x = lo, y = mid, o = lo;
            while(x < mid && y < hi) {
                if(cmp_lead_int64(in + y * elsize, in + x * elsize) < 0) memcpy(tmp + (o++) * elsize, in + (y++) * elsize, elsize);
                else memcpy(tmp + (o++) * elsize, in + (x++) * elsize, elsize);
            }
            while(x < mid) memcpy(tmp + (o++) * elsize, in + (x++) * elsize, elsize);
            while(y < hi) memcpy(tmp + (o++) * elsize, in + (y++) * elsize, elsize);
        }
        memcpy(in, tmp, elsize * n);
    }
    myfree(tmp);
}

/* The reference candidate of a track on one rank: the lowest (not own tag, BindRank, ID). */
struct FrameRef {
    int64_t TrackId;
    int64_t notown;
    MyIDType ID;
    int64_t BindRank;
    double Pos[3];
};

static int
framereef_less(const struct FrameRef * x, const struct FrameRef * y)
{
    if(x->notown != y->notown) return x->notown < y->notown;
    if(x->BindRank != y->BindRank) return x->BindRank < y->BindRank;
    return x->ID < y->ID;
}

/* Partial sums of one track's previous bound particles on one rank. */
struct FrameSum {
    int64_t TrackId;
    int64_t count;
    double mass;
    double ref[3];
    double mdx[3];
    double mv[3];
};

struct BoundIndex {
    int64_t TrackId;
    MyIDType ID;
    int64_t i;
};

static int
cmp_boundindex(const void * a, const void * b)
{
    const struct BoundIndex * x = (const struct BoundIndex *) a, * y = (const struct BoundIndex *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

void
subhalo_previous_frames(struct SubTrackTable * tab, MPI_Comm Comm)
{
    const int NTask = tab->NTask;
    const double BoxSize = PartManager->BoxSize;
    int64_t i, j;
    int k;

    int64_t nb = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(part_live(&P[i]) && P[i].BoundTrackId >= 0)
            nb++;
    struct BoundIndex * idx = (struct BoundIndex *) mymalloc("SubFrameIdx", sizeof(struct BoundIndex) * (nb + 1));
    nb = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(part_live(&P[i]) && P[i].BoundTrackId >= 0) {
            idx[nb].TrackId = P[i].BoundTrackId;
            idx[nb].ID = P[i].ID;
            idx[nb].i = i;
            nb++;
        }
    /* ID order makes the local sums independent of the particle order */
    qsort_openmp(idx, nb, sizeof(struct BoundIndex), cmp_boundindex);
    int64_t ns = 0;
    for(i = 0; i < nb; i++)
        if(i == 0 || idx[i].TrackId != idx[i-1].TrackId)
            ns++;

    /* 1. the common reference of every track (H7): local candidates to the home rank, the
     *    minimum back to every holder in request order */
    struct FrameRef * myref = (struct FrameRef *) mymalloc("SubFrameRef", sizeof(struct FrameRef) * (ns + 1));
    int64_t s = -1;
    for(i = 0; i < nb; i++) {
        const struct particle_data * p = &P[idx[i].i];
        struct FrameRef c;
        c.TrackId = idx[i].TrackId;
        c.notown = p->TrackId != idx[i].TrackId;
        c.ID = p->ID;
        c.BindRank = p->BindRank;
        for(k = 0; k < 3; k++)
            c.Pos[k] = p->Pos[k];
        if(i == 0 || idx[i].TrackId != idx[i-1].TrackId)
            myref[++s] = c;
        else if(framereef_less(&c, &myref[s]))
            myref[s] = c;
    }
    int64_t * sendcounts = ta_malloc("SubFrameCounts", int64_t, NTask);
    int64_t * recvcounts = ta_malloc("SubFrameRecvCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(j = 0; j < ns; j++)
        sendcounts[subhalo_table_home_rank(tab, myref[j].TrackId)]++;
    {
        struct FrameRef * inref = NULL;
        int64_t nin = subhalo_alltoallv(myref, sendcounts, sizeof(struct FrameRef), (void **) &inref, 0, recvcounts, Comm);
        /* the minimum per track: one pass over a TrackId-sorted copy */
        struct FrameRef * minref = (struct FrameRef *) mymalloc("SubFrameMin", sizeof(struct FrameRef) * (nin + 1));
        memcpy(minref, inref, sizeof(struct FrameRef) * nin);
        stable_sort_lead(minref, nin, sizeof(struct FrameRef));
        int64_t nmin = 0;
        for(i = 0; i < nin; i++) {
            if(nmin == 0 || minref[i].TrackId != minref[nmin-1].TrackId)
                minref[nmin++] = minref[i];
            else if(framereef_less(&minref[i], &minref[nmin-1]))
                minref[nmin-1] = minref[i];
        }
        for(i = 0; i < nin; i++) {
            int64_t lo = 0, hi = nmin;
            while(lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if(minref[mid].TrackId < inref[i].TrackId) lo = mid + 1;
                else hi = mid;
            }
            if(lo >= nmin || minref[lo].TrackId != inref[i].TrackId)
                endrun(5, "Frame reference of track %ld lost on its home rank.\n", inref[i].TrackId);
            inref[i] = minref[lo];
        }
        myfree(minref);
        struct FrameRef * back = NULL;
        int64_t nback = subhalo_alltoallv(inref, recvcounts, sizeof(struct FrameRef), (void **) &back, 0, NULL, Comm);
        if(nback != ns)
            endrun(5, "Frame reference exchange returned %ld entries for %ld tracks.\n", nback, ns);
        for(j = 0; j < ns; j++) {
            if(back[j].TrackId != myref[j].TrackId)
                endrun(5, "Frame reference exchange out of order (%ld for %ld).\n", back[j].TrackId, myref[j].TrackId);
            myref[j] = back[j];
        }
        myfree(back);
        myfree(inref);
    }

    /* 2. local sums relative to the common reference, in ID order */
    struct FrameSum * sums = (struct FrameSum *) mymalloc("SubFrameSums", sizeof(struct FrameSum) * (ns + 1));
    s = -1;
    for(i = 0; i < nb; i++) {
        if(i == 0 || idx[i].TrackId != idx[i-1].TrackId) {
            s++;
            memset(&sums[s], 0, sizeof(struct FrameSum));
            sums[s].TrackId = idx[i].TrackId;
            for(k = 0; k < 3; k++)
                sums[s].ref[k] = myref[s].Pos[k];
        }
        const struct particle_data * p = &P[idx[i].i];
        const double m = p->Mass;
        sums[s].count++;
        sums[s].mass += m;
        for(k = 0; k < 3; k++) {
            sums[s].mdx[k] += m * NEAREST(p->Pos[k] - sums[s].ref[k], BoxSize);
            sums[s].mv[k] += m * p->Vel[k];
        }
    }
    struct FrameSum * in = NULL;
    int64_t nin = subhalo_alltoallv(sums, sendcounts, sizeof(struct FrameSum), (void **) &in, 0, NULL, Comm);
    stable_sort_lead(in, nin, sizeof(struct FrameSum));

    /* 3. the home ranks reduce in rank order */
    int64_t a = 0, nframes = 0;
    int missing = 0;
    while(a < nin) {
        int64_t b = a;
        int64_t count = 0;
        double mass = 0, mdx[3] = {0, 0, 0}, mv[3] = {0, 0, 0};
        while(b < nin && in[b].TrackId == in[a].TrackId) {
            count += in[b].count;
            mass += in[b].mass;
            for(k = 0; k < 3; k++) {
                mdx[k] += in[b].mdx[k];
                mv[k] += in[b].mv[k];
            }
            b++;
        }
        struct SubTrack * rec = subhalo_table_get(tab, in[a].TrackId);
        if(!rec)
            missing = 1;
        else {
            rec->NPrev = count;
            rec->MPrev = mass;
            for(k = 0; k < 3; k++) {
                double x = in[a].ref[k] + (mass > 0 ? mdx[k] / mass : 0);
                while(x >= BoxSize) x -= BoxSize;
                while(x < 0) x += BoxSize;
                rec->RefPos[k] = x;
                rec->RefVel[k] = mass > 0 ? mv[k] / mass : 0;
            }
            nframes++;
        }
        a = b;
    }
    if(missing)
        endrun(5, "Previous-bound frame of a track not homed on rank %d.\n", tab->ThisTask);
    myfree(in);
    myfree(sums);
    ta_free(recvcounts);
    ta_free(sendcounts);
    myfree(myref);
    myfree(idx);
    int64_t tot = 0;
    MPI_Allreduce(&nframes, &tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo frames: previous-bound frames of %ld tracks.\n", tot);
}

/* ------------------------------------------------------------------ step 3: keys of the tracks */

int64_t
subhalo_assign_keys(struct SubTrackTable * tab, int64_t TotNgroups, MPI_Comm Comm)
{
    int64_t i, nh = 0;
    for(i = 0; i < tab->NLocal; i++) {
        struct SubTrack * t = &tab->Track[i];
        t->Key = -1;
        if(t->NboundPrev > 0 && t->Host < 0)
            nh++;
    }
    int64_t off = 0, tot = 0;
    MPI_Exscan(&nh, &off, 1, MPI_INT64, MPI_SUM, Comm);
    if(tab->ThisTask == 0)
        off = 0;
    MPI_Allreduce(&nh, &tot, 1, MPI_INT64, MPI_SUM, Comm);
    int64_t o = 0;
    for(i = 0; i < tab->NLocal; i++) {
        struct SubTrack * t = &tab->Track[i];
        if(t->NboundPrev <= 0)
            continue;
        if(t->Host >= 0)
            t->Key = t->Host;
        else
            t->Key = TotNgroups + 1 + off + (o++);
    }
    return tot;
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
    if(x->rank != y->rank)
        return (x->rank > y->rank) - (x->rank < y->rank);
    return (x->track > y->track) - (x->track < y->track);
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
     *  - per member track (distinct tag in a key and routed orphan, bounded by the summed
     *    per-rank tag counts plus the orphans): the fetched record (SubMembers), the output copy
     *    (SubOutRecs), its result slot and candidate node, the id list, the fetch/store
     *    permutation, the sink-winner slot, the hosted index and spares, and the tracer queue of
     *    the remote descendant lookup;
     *  - per key: its newborn result slot and node, an in-place winner slot, its segment, its
     *    output record and the small per-key scratch (SUBHALO_KEY_MISC_BYTES).
     * The fetch and store move records in batches bounded by subhalo_table_batch_peak_bytes,
     * which is in the reserve together with the constant SUBHALO_RESERVE_BYTES and the
     * NTask-sized vectors. */
    /* the arena allowance of subhalo_run_serial_keys, from the validated SubhaloSourceRelaxFactor (J7) */
    cfg->PartBytes = (double) subhalo_arena_bytes_per_part();
    /* Phase 2a: per member also its result slot and candidate node (a member hosted by the key),
     * its arena node allowance, and its share of the remote descendant lookup
     * (SUBHALO_MAXTRACERS ids and hits) */
    cfg->MemberBytes = 2. * sizeof(struct SubTrack) + sizeof(struct SubhaloResult) + SUBHALO_NODE_BYTES + SUBHALO_ARENA_NODE_BYTES + 6. * sizeof(int64_t)
        + SUBHALO_MAXTRACERS * (sizeof(MyIDType) + sizeof(struct SubhaloIdHit) + sizeof(int64_t));
    cfg->KeyBytes = 2. * sizeof(struct SubhaloResult) + SUBHALO_NODE_BYTES + SUBHALO_ARENA_NODE_BYTES + sizeof(struct SubhaloKeySeg) + sizeof(struct SubTrack) + SUBHALO_KEY_MISC_BYTES;
    cfg->InplaceKeyBytes = SUBHALO_INPLACE_KEY_BYTES;
    /* exchange.c per departing particle: the particle and slot copies, the exchange list entry
     * and the layout cache (the slot element size is added per rank) */
    cfg->ExchPartBytes = sizeof(struct particle_data) + sizeof(int) + 8;
    /* the local sort (qsort_openmp sorts particle_data indirectly: two pointers per element),
     * reused later by the newborn request list and the restore's exchange lists, and by the
     * sorted (ID, index) index of the remote descendant lookup with its sort copy (32 bytes per
     * particle, the larger of the two; J4) */
    cfg->SortBytesPerPart = 32;
    /* constant per call: SUBHALO_RESERVE_BYTES, one batch of the record fetch/store, the
     * NTask-sized vectors, and the newborn numbering's directory entries and lookups for the
     * KChunk keys this rank's directory covers (about four 16-byte entries per key) */
    cfg->ReserveBytes = SUBHALO_RESERVE_BYTES + subhalo_table_batch_peak_bytes(NTask) + (double) NTask * SUBHALO_RESERVE_PER_TASK
        + 64. * (double) cfg->KChunk;
    /* the batch buffers of the remote descendant lookup (J4); the record fetch/store batches and
     * the lookup batches never coexist, but both are reserved */
    cfg->ReserveBytes += subhalo_lookup_batch_peak_bytes(NTask);
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

/* A particle of a key for the census: key, tag, type and whether it is a field particle. */
struct CensusPart {
    int64_t g;
    int64_t TrackId;
    int32_t type;
    int32_t field;
};

static int
cmp_censuspart(const void * a, const void * b)
{
    const struct CensusPart * x = (const struct CensusPart *) a, * y = (const struct CensusPart *) b;
    if(x->g != y->g)
        return (x->g > y->g) - (x->g < y->g);
    return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
}


static int
cmp_trackkey(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloTrackKey *) a)->TrackId, y = ((const struct SubhaloTrackKey *) b)->TrackId;
    return (x > y) - (x < y);
}

static int64_t
field_key(const struct SubhaloTrackKey * map, int64_t n, int64_t T)
{
    int64_t lo = 0, hi = n;
    while(lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if(map[mid].TrackId < T) lo = mid + 1;
        else hi = mid;
    }
    return (lo < n && map[lo].TrackId == T) ? map[lo].Key : -1;
}

/* The key of a live particle before the exchange: its group, or its track's key if it is a
 * tagged field particle, -1 otherwise. */
static inline int64_t
census_key(const struct particle_data * p, const struct SubhaloTrackKey * map, int64_t nmap)
{
    if(p->GrNr >= 0)
        return p->GrNr;
    if(p->TrackId >= 0)
        return field_key(map, nmap, p->TrackId);
    return -1;
}

/* The keys of the tracks tagging this rank's live field particles, from their home ranks
 * (request/reply). Allocates the map on the bottom of the stack. Collective. */
static int64_t
query_field_keys(struct SubTrackTable * tab, struct SubhaloTrackKey ** mapout, int home_only, MPI_Comm Comm)
{
    const int NTask = tab->NTask;
    int64_t i, n = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(part_live(&P[i]) && (home_only ? P[i].GrNr == -1 : P[i].GrNr < 0) && P[i].TrackId >= 0)
            n++;
    struct SubhaloTrackKey * map = (struct SubhaloTrackKey *) mymalloc("SubFieldKeys", sizeof(struct SubhaloTrackKey) * (n + 1));
    n = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(part_live(&P[i]) && (home_only ? P[i].GrNr == -1 : P[i].GrNr < 0) && P[i].TrackId >= 0) {
            map[n].TrackId = P[i].TrackId;
            map[n].Key = -1;
            n++;
        }
    qsort_openmp(map, n, sizeof(struct SubhaloTrackKey), cmp_trackkey);
    int64_t nu = 0;
    for(i = 0; i < n; i++)
        if(nu == 0 || map[i].TrackId != map[nu-1].TrackId)
            map[nu++] = map[i];
    int64_t * sendcounts = ta_malloc("SubFKCounts", int64_t, NTask);
    int64_t * recvcounts = ta_malloc("SubFKRecv", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nu; i++)
        sendcounts[subhalo_table_home_rank(tab, map[i].TrackId)]++;
    struct SubhaloTrackKey * in = NULL;
    int64_t nin = subhalo_alltoallv(map, sendcounts, sizeof(struct SubhaloTrackKey), (void **) &in, 0, recvcounts, Comm);
    int missing = 0;
    for(i = 0; i < nin; i++) {
        const struct SubTrack * rec = subhalo_table_get(tab, in[i].TrackId);
        if(!rec)
            missing = 1;
        in[i].Key = rec ? rec->Key : -1;
    }
    if(missing)
        endrun(5, "Field key query for a track not homed on rank %d.\n", tab->ThisTask);
    struct SubhaloTrackKey * back = NULL;
    int64_t nback = subhalo_alltoallv(in, recvcounts, sizeof(struct SubhaloTrackKey), (void **) &back, 0, NULL, Comm);
    if(nback != nu)
        endrun(5, "Field key query returned %ld answers for %ld tracks.\n", nback, nu);
    /* answers come back in request order */
    for(i = 0; i < nu; i++)
        map[i].Key = back[i].Key;
    myfree(back);
    myfree(in);
    ta_free(recvcounts);
    ta_free(sendcounts);
    *mapout = map;
    return nu;
}

int64_t
subhalo_home_field_keys(struct SubTrackTable * tab, struct SubhaloTrackKey ** map, MPI_Comm Comm)
{
    return query_field_keys(tab, map, 1, Comm);
}

int64_t
subhalo_track_key_lookup(const struct SubhaloTrackKey * map, int64_t n, int64_t TrackId)
{
    return field_key(map, n, TrackId);
}

void
subhalo_plan_build(struct SubhaloPlan * plan, const FOFGroups * fof, struct SubTrackTable * tab, int64_t NHostless, MPI_Comm Comm)
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
    plan->TotKeys = fof->TotNgroups + NHostless;
    plan->KChunk = subhalo_key_chunk(plan->TotKeys, NTask);
    const int64_t KChunk = plan->KChunk;
    const int64_t TotNgroups = fof->TotNgroups;
    const int64_t TotKeys = plan->TotKeys;

    /* ---- the keys of the field particles (Phase 2a plan step 3) ---- */
    struct SubhaloTrackKey * fkmap = NULL;
    const int64_t nfk = query_field_keys(tab, &fkmap, 0, Comm);

    /* ---- local census: (key, type) counts and distinct tags of the particles of the keys ---- */
    struct SubhaloPlanRes res;
    memset(&res, 0, sizeof(res));
    int64_t ngrp = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        res.NumLive++;
        if(part_live(&P[i]) ? census_key(&P[i], fkmap, nfk) >= 0 : P[i].GrNr >= 0) ngrp++;
    }
    /* On the top so that it can be freed while the census messages (bottom) live on. */
    struct CensusPart * gt = (struct CensusPart *) mymalloc2("SubCensusKeys", sizeof(struct CensusPart) * (ngrp + 1));
    ngrp = 0;
    int badgrnr = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        /* a swallowed BH in a group stays in its group key (FOF counted it); a swallowed
         * tagged field BH stays home */
        const int64_t key = part_live(&P[i]) ? census_key(&P[i], fkmap, nfk) : P[i].GrNr;
        if(key < 0) continue;
        if(P[i].GrNr >= 0 && P[i].GrNr > TotNgroups)
            badgrnr = 1;
        if(key < 1 || key > TotKeys)
            badgrnr = 1;
        gt[ngrp].g = key;
        gt[ngrp].TrackId = P[i].TrackId;
        gt[ngrp].type = P[i].Type;
        gt[ngrp].field = P[i].GrNr < 0;
        ngrp++;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: particle key outside [1, %ld] (%ld groups).\n", TotKeys, TotNgroups);
    qsort_openmp(gt, ngrp, sizeof(struct CensusPart), cmp_censuspart);
    int64_t ncensus = 0;
    for(i = 0; i < ngrp; i++)
        if(i == 0 || gt[i].g != gt[i-1].g)
            ncensus++;
    /* orphans hosted by a group (step 1), from their home ranks */
    int64_t norph = 0;
    for(i = 0; i < tab->NLocal; i++)
        if(tab->Track[i].NboundPrev <= 0 && tab->Track[i].Host >= 0)
            norph++;
    const int64_t nmsg = ncensus + fof->Ngroups + norph;
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
            msg[m].track = -1;
        }
        msg[m].cnt[gt[i].type]++;
        if(gt[i].field)
            msg[m].nfield++;
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
        d->track = -1;
        if(d->g < 1 || d->g > TotNgroups)
            badgrnr = 1;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: group GrNr outside [1, %ld].\n", TotNgroups);
    m = ncensus + fof->Ngroups;
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * tr = &tab->Track[i];
        if(!(tr->NboundPrev <= 0 && tr->Host >= 0))
            continue;
        struct SubhaloDirMsg * d = &msg[m++];
        memset(d, 0, sizeof(struct SubhaloDirMsg));
        d->g = tr->Host;
        d->ntags = 1;
        d->rank = ThisTask;
        d->kind = 2;
        d->track = tr->TrackId;
        if(d->g < 1 || d->g > TotNgroups)
            badgrnr = 1;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: an orphan's host outside [1, %ld].\n", TotNgroups);
    qsort(msg, nmsg, sizeof(struct SubhaloDirMsg), cmp_dirmsg);
    int64_t * sendcounts = ta_malloc("SubPlanCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nmsg; i++)
        sendcounts[subhalo_key_dir_rank(msg[i].g, KChunk)]++;
    struct SubhaloDirMsg * in = NULL;
    int64_t nin = subhalo_alltoallv(msg, sendcounts, sizeof(struct SubhaloDirMsg), (void **) &in, 0, NULL, Comm);
    /* (g, kind, rank) and, for the orphans, the TrackId: a total order */
    qsort(in, nin, sizeof(struct SubhaloDirMsg), cmp_dirmsg);

    /* ---- the directory: keys g in [lo, lo + nall) ---- */
    const int64_t lo = (int64_t) ThisTask * KChunk + 1;
    int64_t nall = TotKeys - lo + 1;
    if(nall < 0) nall = 0;
    if(nall > KChunk) nall = KChunk;
    struct SubhaloDirKey * dk = (struct SubhaloDirKey *) mymalloc("SubDirKeys", sizeof(struct SubhaloDirKey) * (nall + 1));
    for(k = 0; k < nall; k++) {
        memset(&dk[k], 0, sizeof(struct SubhaloDirKey));
        dk[k].g = lo + k;
        dk[k].cstart = -1;
        dk[k].ostart = -1;
        dk[k].decider = -1;
        dk[k].target = -1;
    }
    int baddir = 0;
    int64_t * foflen = (int64_t *) mymalloc("SubDirFofLen", sizeof(int64_t) * (nall + 1));
    for(k = 0; k < nall; k++)
        foflen[k] = 0;
    for(i = 0; i < nin; i++) {
        const int64_t kk = in[i].g - lo;
        if(kk < 0 || kk >= nall) {
            baddir = 1;
            continue;
        }
        struct SubhaloDirKey * d = &dk[kk];
        if(in[i].kind == 1) {
            if(d->has_group || d->g > TotNgroups)
                baddir = 1;
            d->has_group = 1;
            for(t = 0; t < 6; t++)
                foflen[kk] += in[i].cnt[t];
            for(t = 0; t < 3; t++)
                d->ref[t] = in[i].ref[t];
        }
        else if(in[i].kind == 2) {
            if(d->ostart < 0)
                d->ostart = i;
            d->ocount++;
            d->Tags += in[i].ntags;
        }
        else {
            /* census entries sort first (kind 0), by rank */
            if(d->cstart < 0)
                d->cstart = i;
            d->ccount++;
            d->Tags += in[i].ntags;
            d->NField += in[i].nfield;
            int64_t c = 0;
            for(t = 0; t < 6; t++) {
                d->LenType[t] += in[i].cnt[t];
                c += in[i].cnt[t];
            }
            d->Length += c;
            /* the deciding rank of an in-place key holds part of its FOF particles (the field
             * particles of an oversized key stay home outside its segments) */
            if(d->decider < 0 && (d->g > TotNgroups || c > in[i].nfield))
                d->decider = in[i].rank;
        }
    }
    for(k = 0; k < nall; k++) {
        if(dk[k].g <= TotNgroups) {
            if(!dk[k].has_group || dk[k].Length - dk[k].NField != foflen[k])
                baddir = 1;
        }
        else if(dk[k].ocount > 0)
            baddir = 1;
    }
    myfree(foflen);
    if(baddir)
        endrun(5, "Subhalo plan: the key directory on rank %d is inconsistent with the FOF groups.\n", ThisTask);
    /* the keys that exist: every FOF group, and the pseudo-keys with particles (a hostless track
     * without live particles gets no key and dies in subhalo_update_unhosted) */
    int64_t ndir = 0;
    for(k = 0; k < nall; k++)
        if(dk[k].g <= TotNgroups || dk[k].Length > 0)
            dk[ndir++] = dk[k];

    /* ---- classification and packing (particle-free, subhalo_plan_pack) ---- */
    res.MaxPart = PartManager->MaxPart;
    for(t = 0; t < 6; t++) {
        const struct slot_info * info = &SlotsManager->info[t];
        res.slotelsize[t] = info->enabled ? (int64_t) info->elsize : 0;
        res.slotfree[t] = info->enabled ? info->maxsize - info->size : 0;
    }
    res.freebytes = (double) mymalloc_freebytes();
    struct SubhaloPlanCfg cfg;
    subhalo_plan_default_cfg(&cfg, TotKeys, NTask);
    subhalo_plan_pack(dk, ndir, in, &res, &cfg, &plan->Stats, Comm);
    message(0, "Subhalo plan: %ld serial keys (largest %ld), %ld oversized keys in place (%ld moved by %d census round(s)); "
               "%ld pseudo-keys of hostless tracks; min budget %g particles, min byte budget %g MB, min free bytes %g MB; %d pass(es)%s%s.\n",
            plan->Stats.NSerial, plan->Stats.LargestSerial, plan->Stats.NOversized, plan->Stats.NRepaired, plan->Stats.CensusRounds,
            NHostless, plan->Stats.MinBudget, plan->Stats.MinByteBudget / (1024. * 1024.), plan->Stats.MinFreeBytes / (1024. * 1024.), plan->Stats.Iterations,
            plan->Stats.Fallback ? " [pass cap reached]" : "",
            sp->SubhaloDebugForceExchangeFailure ? " [debug: every serial key sent to rank 0]" : "");

    /* ---- replies (top of the stack, in this order: LocalKeys, Assigned, Inplace, OrphanIds) ---- */
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

    /* 3. the global list of oversized keys (their FOF particles only) */
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
            e->Length = dk[k].g <= TotNgroups ? dk[k].Length - dk[k].NField : 0;
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

    /* 4. the orphans hosted by a serial key go to its processing rank (an orphan hosted by an
     *    oversized key is not processed: the Phase 1 in-place rule ignores orphans) */
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    int64_t nofw = 0;
    for(k = 0; k < ndir; k++)
        if(!dk[k].oversized && dk[k].target >= 0)
            for(i = dk[k].ostart; i >= 0 && i < dk[k].ostart + dk[k].ocount; i++) {
                sendcounts[dk[k].target]++;
                nofw++;
            }
    int64_t * ofw = (int64_t *) mymalloc("SubPlanOrph", sizeof(int64_t) * (nofw + 1));
    off[0] = 0;
    for(r = 1; r < NTask; r++)
        off[r] = off[r-1] + sendcounts[r-1];
    for(k = 0; k < ndir; k++)
        if(!dk[k].oversized && dk[k].target >= 0)
            for(i = dk[k].ostart; i >= 0 && i < dk[k].ostart + dk[k].ocount; i++)
                ofw[off[dk[k].target]++] = in[i].track;
    plan->NOrphanIds = subhalo_alltoallv(ofw, sendcounts, sizeof(int64_t), (void **) &plan->OrphanIds, 1, NULL, Comm);
    qsort(plan->OrphanIds, plan->NOrphanIds, sizeof(int64_t), cmp_int64);
    myfree(ofw);
    myfree(as);
    ta_free(off);
    myfree(rep);

    myfree(dk);
    myfree(in);
    ta_free(sendcounts);
    myfree(msg);

    /* ---- targets into P[i].TopLeaf (the layout of the exchange); the key of a travelling
     *      field particle into its GrNr ---- */
    int64_t lost = 0, nfieldtravel = 0;
    #pragma omp parallel for reduction(+: lost, nfieldtravel)
    for(i = 0; i < PartManager->NumPart; i++) {
        P[i].TopLeaf = ThisTask;
        if(P[i].IsGarbage)
            continue;
        const int64_t key = part_live(&P[i]) ? census_key(&P[i], fkmap, nfk) : P[i].GrNr;
        if(key < 0)
            continue;
        struct SubhaloLocalKey lkey;
        lkey.g = key;
        const struct SubhaloLocalKey * lk = (const struct SubhaloLocalKey *)
            bsearch(&lkey, plan->LocalKeys, plan->NLocalKeys, sizeof(struct SubhaloLocalKey), cmp_localkey_g);
        if(!lk) {
            lost++;
            continue;
        }
        P[i].TopLeaf = lk->oversized ? ThisTask : lk->target;
        if(P[i].GrNr < 0 && !lk->oversized) {
            P[i].GrNr = -(2 + key);
            nfieldtravel++;
        }
    }
    if(lost)
        endrun(5, "Subhalo plan: %ld particles on rank %d have no plan entry.\n", lost, ThisTask);
    myfree(fkmap);
    int64_t nft = 0;
    MPI_Allreduce(&nfieldtravel, &nft, 1, MPI_INT64, MPI_SUM, Comm);
    int64_t no = 0;
    MPI_Allreduce(&plan->NOrphanIds, &no, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo plan: %ld field particles travel with their tracks' keys; %ld hosted orphans routed to their keys.\n", nft, no);
}

void
subhalo_restore_field_grnr(void)
{
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        if(P[i].GrNr < -1)
            P[i].GrNr = -1;
}

void
subhalo_plan_free(struct SubhaloPlan * plan)
{
    if(plan->OrphanIds)
        myfree(plan->OrphanIds);
    if(plan->Inplace)
        myfree(plan->Inplace);
    if(plan->Assigned)
        myfree(plan->Assigned);
    if(plan->LocalKeys)
        myfree(plan->LocalKeys);
    plan->OrphanIds = NULL;
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
    /* the processing key: the group, or the encoded key of a travelling field particle */
    const int64_t ka = subhalo_particle_key(pa), kb = subhalo_particle_key(pb);
    if(ka != kb)
        return (ka > kb) - (ka < kb);
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
        if(subhalo_particle_key(&P[i]) >= 0 && (i == 0 || subhalo_particle_key(&P[i]) != subhalo_particle_key(&P[i-1])))
            n++;
    struct SubhaloKeySeg * s = (struct SubhaloKeySeg *) mymalloc2("SubKeySegs", sizeof(struct SubhaloKeySeg) * (n + 1));
    int64_t k = 0, nserial = 0;
    int bad = 0;
    for(i = 0; i < NumPart && !P[i].IsGarbage; ) {
        const int64_t key = subhalo_particle_key(&P[i]);
        if(key < 0) {
            i++;
            continue;
        }
        int64_t j = i;
        while(j < NumPart && !P[j].IsGarbage && subhalo_particle_key(&P[j]) == key)
            j++;
        struct SubhaloKeySeg * seg = &s[k++];
        memset(seg, 0, sizeof(*seg));
        seg->g = key;
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

int64_t
subhalo_fetch_members(const struct SubhaloKeySeg * segs, int64_t nsegs, const int64_t * extra, int64_t nextra,
                      struct SubTrackTable * tab, struct SubTrack ** members, MPI_Comm Comm)
{
    int64_t k, i, n = nextra;
    for(k = 0; k < nsegs; k++)
        for(i = segs[k].start; i < segs[k].start + segs[k].n; i++)
            if(P[i].TrackId >= 0 && (i == segs[k].start || P[i].TrackId != P[i-1].TrackId))
                n++;
    int64_t * ids = (int64_t *) mymalloc("SubMemberIds", sizeof(int64_t) * (n + 1));
    for(n = 0; n < nextra; n++)
        ids[n] = extra[n];
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

double
subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, const int64_t * extra,
                        subhalo_key_fn fn, void * userdata)
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
#define KEY_ARENA_BYTES(kk) ((size_t) bytes_per_part * (size_t) segs[kk].n + (size_t) (extra ? extra[kk] : 0) + 64)
    double peakfrac = 0;

    /* keys above one thread's share run alone, with every thread inside the key */
    const int64_t share = (total + nthreads - 1) / nthreads;
    int64_t nalone = 0;
    while(nthreads > 1 && nalone < ns && ord[nalone].n > share)
        nalone++;
    if(nalone > 0) {
        struct SubhaloArena arena;
        arena.size = 0;
        for(k = 0; k < nalone; k++)
            if(KEY_ARENA_BYTES(ord[k].k) > arena.size)
                arena.size = KEY_ARENA_BYTES(ord[k].k);
        arena.base = (char *) mymalloc2("SubArenaAlone", arena.size);
        for(k = 0; k < nalone; k++) {
            arena.used = 0;
            arena.peak = 0;
            fn(&segs[ord[k].k], ord[k].k, &arena, nthreads, userdata);
            const double f = (double) arena.peak / (double) KEY_ARENA_BYTES(ord[k].k);
            if(f > peakfrac)
                peakfrac = f;
        }
        myfree(arena.base);
    }

    /* the rest: static longest-processing-time-first assignment to threads */
    int64_t * load = (int64_t *) mymalloc("SubThreadLoad", sizeof(int64_t) * nthreads);
    int64_t * maxb = (int64_t *) mymalloc("SubThreadMaxB", sizeof(int64_t) * nthreads);
    int64_t * count = (int64_t *) mymalloc("SubThreadCount", sizeof(int64_t) * nthreads);
    double * tpeak = (double *) mymalloc("SubThreadPeak", sizeof(double) * nthreads);
    int * owner = (int *) mymalloc("SubKeyOwner", sizeof(int) * (ns + 1));
    for(t = 0; t < nthreads; t++) {
        load[t] = maxb[t] = count[t] = 0;
        tpeak[t] = 0;
    }
    for(k = nalone; k < ns; k++) {
        int best = 0;
        for(t = 1; t < nthreads; t++)
            if(load[t] < load[best])
                best = t;
        owner[k] = best;
        load[best] += ord[k].n;
        count[best]++;
        if((int64_t) KEY_ARENA_BYTES(ord[k].k) > maxb[best])
            maxb[best] = KEY_ARENA_BYTES(ord[k].k);
    }
    struct SubhaloArena * arenas = (struct SubhaloArena *) mymalloc("SubArenas", sizeof(struct SubhaloArena) * nthreads);
    for(t = 0; t < nthreads; t++) {
        arenas[t].size = maxb[t] > 0 ? maxb[t] : 64;
        arenas[t].used = 0;
        arenas[t].peak = 0;
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
                arenas[tt].peak = 0;
                fn(&segs[ord[kk].k], ord[kk].k, &arenas[tt], 1, userdata);
                /* relative to what the key itself was allowed, not the thread's larger arena */
                const double f = (double) arenas[tt].peak / (double) KEY_ARENA_BYTES(ord[kk].k);
                if(f > tpeak[tt])
                    tpeak[tt] = f;
            }
        }
    }
#undef KEY_ARENA_BYTES
    for(t = 0; t < nthreads; t++)
        if(tpeak[t] > peakfrac)
            peakfrac = tpeak[t];
    for(t = nthreads - 1; t >= 0; t--)
        myfree(arenas[t].base);
    myfree(arenas);
    myfree(owner);
    myfree(tpeak);
    myfree(count);
    myfree(maxb);
    myfree(load);
    myfree(ord);
    return peakfrac;
}
