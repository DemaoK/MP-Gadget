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

/* Number of the largest serial keys packed exactly by the greedy on rank 0; the rest are
 * laid onto the remaining budgets in GrNr order (a water-fill, which is what longest-first
 * packing reduces to once the keys are small compared with the budgets). */
#define SUBHALO_HEAD_PER_TASK 32
#define SUBHALO_HEAD_MAX 4096
/* Classification and packing are iterated until no new key becomes oversized. */
#define SUBHALO_PLAN_MAXITER 8
/* Bytes kept free for everything the byte check does not itemise (records, MPI buffers). */
#define SUBHALO_RESERVE_BYTES (8L * 1024L * 1024L)
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
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && P[i].TrackId >= 0 && P[i].BoundTrackId == P[i].TrackId)
            n++;
    struct VotePair * v = (struct VotePair *) mymalloc("SubVote", sizeof(struct VotePair) * (n + 1));
    n = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && P[i].TrackId >= 0 && P[i].BoundTrackId == P[i].TrackId) {
            v[n].TrackId = P[i].TrackId;
            v[n].GrNr = P[i].GrNr >= 0 ? P[i].GrNr : -1;
            v[n].count = 1;
            n++;
        }
    n = merge_votepairs(v, n);

    /* sorted by TrackId, hence by home rank */
    int * sendcounts = ta_malloc("SubVoteCounts", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
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

/* Message to the key directory: a census of the particles of key g on a rank (kind 0), or
 * the group record of g from the rank that holds it (kind 1). */
struct DirMsg {
    int64_t g;
    int64_t cnt[6];
    double ref[3];
    int32_t rank;
    int32_t kind;
};

static int
cmp_dirmsg(const void * a, const void * b)
{
    const struct DirMsg * x = (const struct DirMsg *) a, * y = (const struct DirMsg *) b;
    if(x->g != y->g)
        return (x->g > y->g) - (x->g < y->g);
    if(x->kind != y->kind)
        return (x->kind > y->kind) - (x->kind < y->kind);
    return (x->rank > y->rank) - (x->rank < y->rank);
}

static int
cmp_uint64(const void * a, const void * b)
{
    const uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return (x > y) - (x < y);
}

/* A key on its directory rank. */
struct DirKey {
    int64_t g;
    int64_t Length;
    int64_t LenType[6];
    double ref[3];
    int64_t cstart, ccount;     /* census contributions, sorted by rank */
    int target;
    int oversized;
    int decider;
    int has_group;
};

/* Per-rank budget of this iteration (design 7.3). */
struct RankBudget {
    double B;           /* particles the rank may receive */
    double KeyCap;      /* largest key the rank's scratch bytes allow */
    double cap[6];      /* free slots per type */
    double freebytes;
};

/* Head key sent to rank 0 for the greedy. */
struct HeadKey {
    int64_t g;
    int64_t Length;
    int64_t LenType[6];
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

/* Compute this rank's budget given the particles of oversized keys staying on it. */
static void
rank_budget(struct RankBudget * rb, const int64_t grp[7], const int64_t ov[7], int64_t NumLive)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    int t;
    const int64_t dep = grp[6] - ov[6];             /* serial-key particles leaving */
    const int64_t stay = NumLive - dep;
    rb->B = sp->SubhaloMaxGatherFraction * (double) (PartManager->MaxPart - stay);
    if(rb->B < 0)
        rb->B = 0;
    size_t maxelsize = 0;
    double slotgrowth = 0;
    for(t = 0; t < 6; t++) {
        const struct slot_info * info = &SlotsManager->info[t];
        if(!info->enabled) {
            rb->cap[t] = 1e30;
            continue;
        }
        const int64_t dept = grp[t] - ov[t];
        rb->cap[t] = (double) (info->maxsize - info->size + dept);
        if(info->elsize > maxelsize)
            maxelsize = info->elsize;
        slotgrowth += (double) dept * info->elsize;
    }
    /* exchange buffers of exchange.c (particle + slot copy, list and layout cache per
     * departing particle, at most one batch), slot growth, the local sort's index (16 bytes
     * per particle) and a reserve; the remainder is for the per-thread arenas. */
    double exch = (double) dep * (sizeof(struct particle_data) + maxelsize + sizeof(int) + 8);
    if(exch > SUBHALO_EXCHANGE_BATCH_BYTES)
        exch = SUBHALO_EXCHANGE_BATCH_BYTES;
    rb->freebytes = (double) mymalloc_freebytes();
    const double avail = rb->freebytes - exch - slotgrowth - 16. * PartManager->MaxPart - SUBHALO_RESERVE_BYTES;
    rb->KeyCap = avail > 0 ? avail / SUBHALO_ARENA_BYTES_PER_PART : 0;
}

/* Greedy longest-first on rank 0 for the head keys (already in GrNr = decreasing length order). */
static void
pack_head(const struct HeadKey * hk, int64_t nh, const struct RankBudget * rb, int NTask, double * asg, double * asgT, struct HeadResult * hr)
{
    int64_t k;
    int r, t;
    for(k = 0; k < nh; k++) {
        int best = -1;
        double bestfree = -1;
        for(r = 0; r < NTask; r++) {
            const double freer = rb[r].B - asg[r];
            if(hk[k].Length > freer || hk[k].Length > rb[r].KeyCap)
                continue;
            int fits = 1;
            for(t = 0; t < 6; t++)
                if(hk[k].LenType[t] > rb[r].cap[t] - asgT[6*r + t])
                    fits = 0;
            if(!fits)
                continue;
            if(freer > bestfree) {
                bestfree = freer;
                best = r;
            }
        }
        hr[k].g = hk[k].g;
        hr[k].target = best;
        if(best >= 0) {
            asg[best] += hk[k].Length;
            for(t = 0; t < 6; t++)
                asgT[6*best + t] += hk[k].LenType[t];
        }
    }
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

    /* ---- local census: (GrNr, type) counts of the particles in groups ---- */
    int64_t NumLive = 0, ngrp = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        NumLive++;
        if(P[i].GrNr >= 0) ngrp++;
    }
    /* On the top so that it can be freed while the census messages (bottom) live on. */
    uint64_t * gt = (uint64_t *) mymalloc2("SubCensusKeys", sizeof(uint64_t) * (ngrp + 1));
    ngrp = 0;
    int badgrnr = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage || P[i].GrNr < 0) continue;
        if(P[i].GrNr < 1 || P[i].GrNr > TotNgroups)
            badgrnr = 1;
        gt[ngrp++] = ((uint64_t) P[i].GrNr) * 8 + P[i].Type;
    }
    if(badgrnr)
        endrun(5, "Subhalo plan: particle GrNr outside [1, %ld].\n", TotNgroups);
    qsort_openmp(gt, ngrp, sizeof(uint64_t), cmp_uint64);
    int64_t ncensus = 0;
    for(i = 0; i < ngrp; i++)
        if(i == 0 || gt[i] / 8 != gt[i-1] / 8)
            ncensus++;
    const int64_t nmsg = ncensus + fof->Ngroups;
    struct DirMsg * msg = (struct DirMsg *) mymalloc("SubDirMsg", sizeof(struct DirMsg) * (nmsg + 1));
    int64_t grp[7] = {0};
    int64_t m = -1;
    for(i = 0; i < ngrp; i++) {
        const int64_t g = gt[i] / 8;
        const int type = gt[i] % 8;
        if(m < 0 || msg[m].g != g) {
            m++;
            memset(&msg[m], 0, sizeof(struct DirMsg));
            msg[m].g = g;
            msg[m].rank = ThisTask;
            msg[m].kind = 0;
        }
        msg[m].cnt[type]++;
        grp[type]++;
        grp[6]++;
    }
    myfree(gt);
    for(i = 0; i < fof->Ngroups; i++) {
        const struct Group * G = &fof->Group[i];
        struct DirMsg * d = &msg[ncensus + i];
        memset(d, 0, sizeof(struct DirMsg));
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
    qsort(msg, nmsg, sizeof(struct DirMsg), cmp_dirmsg);
    int * sendcounts = ta_malloc("SubPlanCounts", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(i = 0; i < nmsg; i++)
        sendcounts[subhalo_key_dir_rank(msg[i].g, KChunk)]++;
    struct DirMsg * in = NULL;
    int64_t nin = subhalo_alltoallv(msg, sendcounts, sizeof(struct DirMsg), (void **) &in, 0, NULL, Comm);
    qsort(in, nin, sizeof(struct DirMsg), cmp_dirmsg);

    /* ---- the directory: keys g in [lo, lo + ndir) ---- */
    const int64_t lo = (int64_t) ThisTask * KChunk + 1;
    int64_t ndir = TotNgroups - lo + 1;
    if(ndir < 0) ndir = 0;
    if(ndir > KChunk) ndir = KChunk;
    struct DirKey * dk = (struct DirKey *) mymalloc("SubDirKeys", sizeof(struct DirKey) * (ndir + 1));
    for(k = 0; k < ndir; k++) {
        memset(&dk[k], 0, sizeof(struct DirKey));
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
        struct DirKey * d = &dk[kk];
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

    /* ---- classification and packing, iterated until stable ---- */
    for(k = 0; k < ndir; k++)
        dk[k].oversized = (sp->SubhaloCollectiveMinPart > 0 && dk[k].Length > sp->SubhaloCollectiveMinPart);

    int64_t * ovloc = (int64_t *) mymalloc("SubOvLoc", sizeof(int64_t) * 7 * NTask);
    int64_t * ovtot = (int64_t *) mymalloc("SubOvTot", sizeof(int64_t) * 7 * NTask);
    struct RankBudget * rb = (struct RankBudget *) mymalloc("SubBudgets", sizeof(struct RankBudget) * NTask);
    double * asg = (double *) mymalloc("SubAssigned", sizeof(double) * 7 * NTask);
    double * asgT = asg + NTask;
    double * CE = (double *) mymalloc("SubCumCap", sizeof(double) * NTask);
    const int64_t H = TotNgroups < SUBHALO_HEAD_MAX && TotNgroups < (int64_t) SUBHALO_HEAD_PER_TASK * NTask ? TotNgroups :
                      ((int64_t) SUBHALO_HEAD_PER_TASK * NTask < SUBHALO_HEAD_MAX ? (int64_t) SUBHALO_HEAD_PER_TASK * NTask : SUBHALO_HEAD_MAX);

    int iter;
    for(iter = 0; iter < SUBHALO_PLAN_MAXITER; iter++) {
        /* particles of oversized keys stay on their ranks */
        memset(ovloc, 0, sizeof(int64_t) * 7 * NTask);
        for(k = 0; k < ndir; k++) {
            if(!dk[k].oversized) continue;
            for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++) {
                const int rr = in[i].rank;
                for(t = 0; t < 6; t++) {
                    ovloc[7*rr + t] += in[i].cnt[t];
                    ovloc[7*rr + 6] += in[i].cnt[t];
                }
            }
        }
        MPI_Allreduce(ovloc, ovtot, 7 * NTask, MPI_INT64, MPI_SUM, Comm);
        struct RankBudget mine;
        rank_budget(&mine, grp, ovtot + 7 * ThisTask, NumLive);
        MPI_Allgather(&mine, sizeof(struct RankBudget), MPI_BYTE, rb, sizeof(struct RankBudget), MPI_BYTE, Comm);

        double maxfit = 0, maxcap[6] = {0};
        for(r = 0; r < NTask; r++) {
            const double f = rb[r].B < rb[r].KeyCap ? rb[r].B : rb[r].KeyCap;
            if(f > maxfit) maxfit = f;
            for(t = 0; t < 6; t++)
                if(rb[r].cap[t] > maxcap[t]) maxcap[t] = rb[r].cap[t];
        }
        /* keys that fit no rank */
        int64_t newov = 0;
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized) continue;
            int over = dk[k].Length > maxfit;
            for(t = 0; t < 6; t++)
                if(dk[k].LenType[t] > maxcap[t])
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

        if(sp->SubhaloDebugForceExchangeFailure) {
            /* Debug: every serial key to rank 0, so that the exchange over-commits. */
            for(k = 0; k < ndir; k++)
                if(!dk[k].oversized)
                    dk[k].target = 0;
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
                hloc[nhloc].Length = dk[k].Length;
                memcpy(hloc[nhloc].LenType, dk[k].LenType, sizeof(int64_t) * 6);
                nhloc++;
            }
        int * hcounts = ta_malloc("SubHeadCounts", int, NTask);
        int * hdispl = ta_malloc("SubHeadDispl", int, NTask);
        int hbytes = nhloc * sizeof(struct HeadKey);
        MPI_Allgather(&hbytes, 1, MPI_INT, hcounts, 1, MPI_INT, Comm);
        int64_t htot = 0;
        for(r = 0; r < NTask; r++) {
            hdispl[r] = htot;
            htot += hcounts[r];
        }
        const int64_t nh = htot / sizeof(struct HeadKey);
        struct HeadKey * hall = (struct HeadKey *) mymalloc("SubHeadAll", htot + 1);
        MPI_Gatherv(hloc, hbytes, MPI_BYTE, hall, hcounts, hdispl, MPI_BYTE, 0, Comm);
        struct HeadResult * hr = (struct HeadResult *) mymalloc("SubHeadRes", sizeof(struct HeadResult) * (nh + 1));
        memset(asg, 0, sizeof(double) * 7 * NTask);
        if(ThisTask == 0)
            pack_head(hall, nh, rb, NTask, asg, asgT, hr);
        MPI_Bcast(hr, sizeof(struct HeadResult) * nh, MPI_BYTE, 0, Comm);
        MPI_Bcast(asg, 7 * NTask, MPI_DOUBLE, 0, Comm);
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

        /* tail: water-fill of the remaining budgets in GrNr order */
        double tl[7] = {0}, tmax[7] = {0};
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized || dk[k].g <= H) continue;
            tl[6] += dk[k].Length;
            if(dk[k].Length > tmax[6]) tmax[6] = dk[k].Length;
            for(t = 0; t < 6; t++) {
                tl[t] += dk[k].LenType[t];
                if(dk[k].LenType[t] > tmax[t]) tmax[t] = dk[k].LenType[t];
            }
        }
        double TailN[7], TailMax[7];
        MPI_Allreduce(tl, TailN, 7, MPI_DOUBLE, MPI_SUM, Comm);
        MPI_Allreduce(tmax, TailMax, 7, MPI_DOUBLE, MPI_MAX, Comm);
        double cum = 0;
        for(r = 0; r < NTask; r++) {
            double E = 0;
            if(rb[r].KeyCap >= TailMax[6]) {
                E = rb[r].B - asg[r] - TailMax[6];
                /* per type: approximate, the tail's type mix is taken as uniform */
                for(t = 0; t < 6; t++) {
                    if(TailN[t] <= 0 || rb[r].cap[t] >= 1e29) continue;
                    const double Et = (rb[r].cap[t] - asgT[6*r + t] - TailMax[t]) * TailN[6] / TailN[t];
                    if(Et < E) E = Et;
                }
            }
            if(E < 0) E = 0;
            cum += E;
            CE[r] = cum;
        }
        double pfx = 0;
        MPI_Exscan(&tl[6], &pfx, 1, MPI_DOUBLE, MPI_SUM, Comm);
        if(ThisTask == 0)
            pfx = 0;
        for(k = 0; k < ndir; k++) {
            if(dk[k].oversized || dk[k].g <= H) continue;
            /* first rank whose cumulative capacity exceeds the start of this key */
            int lo_r = 0, hi_r = NTask;
            while(lo_r < hi_r) {
                int mid = (lo_r + hi_r) / 2;
                if(CE[mid] > pfx) hi_r = mid;
                else lo_r = mid + 1;
            }
            dk[k].target = lo_r < NTask ? lo_r : -1;
            pfx += dk[k].Length;
        }

        /* keys that could not be placed become oversized; repeat if any */
        newov = 0;
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].target < 0) {
                dk[k].oversized = 1;
                newov++;
            }
        if(!MPIU_Any(newov > 0, Comm))
            break;
    }
    if(iter >= SUBHALO_PLAN_MAXITER) {
        message(0, "Subhalo plan: classification not stable after %d iterations; keys that could not be placed are processed in place.\n", iter);
        for(k = 0; k < ndir; k++)
            if(!dk[k].oversized && dk[k].target < 0)
                dk[k].oversized = 1;
        iter = SUBHALO_PLAN_MAXITER - 1;
    }
    plan->Iterations = iter + 1;
    for(k = 0; k < ndir; k++) {
        if(dk[k].oversized)
            dk[k].target = -1;
    }

    /* ---- diagnostics ---- */
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
        plan->LargestSerial = tot[0];
        plan->NOversized = tot[1];
        plan->NSerial = tot[2];
        plan->MinBudget = rb[0].B;
        plan->MinFreeBytes = rb[0].freebytes;
        for(r = 1; r < NTask; r++) {
            if(rb[r].B < plan->MinBudget) plan->MinBudget = rb[r].B;
            if(rb[r].freebytes < plan->MinFreeBytes) plan->MinFreeBytes = rb[r].freebytes;
        }
        message(0, "Subhalo plan: %ld serial keys (largest %ld), %ld oversized keys in place; "
                   "min budget %g particles, min free bytes %g MB; %d iteration(s)%s.\n",
                plan->NSerial, plan->LargestSerial, plan->NOversized, plan->MinBudget,
                plan->MinFreeBytes / (1024. * 1024.), plan->Iterations,
                sp->SubhaloDebugForceExchangeFailure ? " [debug: every serial key sent to rank 0]" : "");
    }

    /* ---- replies (top of the stack, in this order: LocalKeys, Assigned, Inplace) ---- */
    /* 1. every holder learns the fate of the keys it holds */
    int64_t nrep = 0;
    for(k = 0; k < ndir; k++)
        nrep += dk[k].ccount;
    struct SubhaloLocalKey * rep = (struct SubhaloLocalKey *) mymalloc("SubPlanRep", sizeof(struct SubhaloLocalKey) * (nrep + 1));
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(k = 0; k < ndir; k++)
        for(i = dk[k].cstart; i >= 0 && i < dk[k].cstart + dk[k].ccount; i++)
            sendcounts[in[i].rank]++;
    int * off = ta_malloc("SubPlanOff", int, NTask);
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
    memset(sendcounts, 0, sizeof(int) * NTask);
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
    int novloc = 0;
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
    int * icounts = ta_malloc("SubIplCounts", int, NTask);
    int * idispl = ta_malloc("SubIplDispl", int, NTask);
    int ibytes = novloc * sizeof(struct SubhaloInplaceKey);
    MPI_Allgather(&ibytes, 1, MPI_INT, icounts, 1, MPI_INT, Comm);
    int64_t itot = 0;
    for(r = 0; r < NTask; r++) {
        idispl[r] = itot;
        itot += icounts[r];
    }
    if(itot >= INT32_MAX)
        endrun(5, "Too many oversized keys (%ld bytes) for the in-place list.\n", itot);
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

    myfree(CE);
    myfree(asg);
    myfree(rb);
    myfree(ovtot);
    myfree(ovloc);
    myfree(dk);
    myfree(in);
    ta_free(sendcounts);
    myfree(msg);

    /* ---- targets into P[i].TopLeaf (the layout of the exchange) ---- */
    int lost = 0;
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
        endrun(5, "Subhalo plan: %d particles on rank %d have no plan entry.\n", lost, ThisTask);
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
