/* Runtime subhalo finder: per-key processing of the Phase 1 membership rule ("every FOF group
 * is one subhalo, no unbinding"), serial and in place.
 *
 * Phase 1 plan (halofinder_phase1_plan.md) section 1 steps 7a-7e and the in-place part of
 * step 9. Step 7f (hostless and dead tracks) is done on the home ranks in subhalo.c. */

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
#include "subhalo.h"
#include "subhalo_internal.h"

static int
cmp_member_trackid(const void * a, const void * b)
{
    const int64_t x = ((const struct SubTrack *) a)->TrackId, y = ((const struct SubTrack *) b)->TrackId;
    return (x > y) - (x < y);
}

struct SubTrack *
subhalo_find_member(const struct SubhaloCallCtx * ctx, int64_t TrackId)
{
    struct SubTrack key;
    key.TrackId = TrackId;
    return (struct SubTrack *) bsearch(&key, ctx->Members, ctx->NMembers, sizeof(struct SubTrack), cmp_member_trackid);
}

/* A member may win or sink only in the key its host vote chose, and only if it is not
 * already an orphan (Phase 1 carries orphans but never resurrects them). */
static int
member_is_candidate(const struct SubTrack * m, int64_t g)
{
    return m->Host == g && m->SnapDeath < 0;
}

static double
wrap_box(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* Sums over the particles of a key (or of its local part). Positions relative to ref. */
struct KeySums {
    double count;
    double mass;
    double ntracer;
    double LenType[6];
    double MassType[6];
    double mdx[3];
    double mv[3];
};
#define KEYSUMS_NDOUBLE (sizeof(struct KeySums) / sizeof(double))

static void
key_sums(struct KeySums * s, int64_t start, int64_t n, const double ref[3], int TracerTypes, double BoxSize)
{
    int64_t i;
    int d;
    memset(s, 0, sizeof(struct KeySums));
    /* sequential: the order (and the result) is that of the local sort */
    for(i = start; i < start + n; i++) {
        const struct particle_data * p = &P[i];
        const double m = p->Mass;
        s->count += 1;
        s->mass += m;
        s->LenType[p->Type] += 1;
        s->MassType[p->Type] += m;
        if(TracerTypes & (1 << p->Type))
            s->ntracer += 1;
        for(d = 0; d < 3; d++) {
            s->mdx[d] += m * NEAREST(p->Pos[d] - ref[d], BoxSize);
            s->mv[d] += m * p->Vel[d];
        }
    }
}

/* Step 7e: the winner's new record. old is the winner's previous record (NULL for a newborn).
 * cen is the mass-weighted centre in the translated frame; mb the index of the rank-0
 * particle in P (or -1 if it is not on this rank, not used in Phase 1). */
static void
fill_winner_record(struct SubTrack * rec, const struct SubTrack * old, int64_t winner, int64_t g, const struct KeySums * s,
                   const double cen[3], int64_t mb, int inplace, const struct SubhaloCallCtx * ctx)
{
    int d, t;
    subhalo_track_clear(rec, winner);
    if(old) {
        /* identity and history carry over; every property column is recomputed or -1 */
        rec->SnapBirth = old->SnapBirth;
        rec->SnapLastIsolation = old->SnapLastIsolation;
        rec->MboundPeak = old->MboundPeak;
        rec->VmaxAtMboundPeak = old->VmaxAtMboundPeak;
        rec->VmaxPeak = old->VmaxPeak;
        rec->MboundAcc = old->MboundAcc;
        rec->VmaxAcc = old->VmaxAcc;
        rec->MboundFirstAcc = old->MboundFirstAcc;
        rec->VmaxFirstAcc = old->VmaxFirstAcc;
        rec->SnapMboundPeak = old->SnapMboundPeak;
        rec->SnapVmaxPeak = old->SnapVmaxPeak;
        rec->SnapAcc = old->SnapAcc;
        rec->SnapFirstAcc = old->SnapFirstAcc;
        rec->SnapLastMajorMerger = old->SnapLastMajorMerger;
        rec->GeomParentTrackId = old->GeomParentTrackId;
        rec->GeomUpParentTrackId = old->GeomUpParentTrackId;
        for(d = 0; d < 3; d++) {
            rec->TracerPos[d] = old->TracerPos[d];
            rec->TracerVel[d] = old->TracerVel[d];
        }
        rec->SnapTracerLost = old->SnapTracerLost;
        rec->NboundPrev = old->NboundPrev;
    }
    else {
        rec->SnapBirth = ctx->SnapNum;
        rec->NboundPrev = 0;
    }
    rec->Host = g;
    rec->HostGroup = g;
    rec->Rank = rec->Depth = 0;
    rec->NestedParentTrackId = -1;
    rec->CentralTrackId = winner;
    rec->SinkTrackId = -1;
    rec->DescendantTrackId = winner;
    rec->SnapDeath = rec->SnapSink = -1;
    rec->Nbound = rec->NboundExclusive = (int32_t) s->count;
    rec->NboundTracer = (int32_t) s->ntracer;
    rec->Flags = inplace ? SUBHALO_FLAG_INPLACE : 0;
    rec->Mbound = s->mass;
    for(t = 0; t < 6; t++) {
        rec->NboundType[t] = rec->NboundExclusiveType[t] = (int32_t) s->LenType[t];
        rec->NtailType[t] = 0;
        rec->MboundType[t] = s->MassType[t];
    }
    for(d = 0; d < 3; d++) {
        const double pos = wrap_box(cen[d] - ctx->Offset[d], ctx->BoxSize);
        rec->CorePos[d] = rec->CoMPos[d] = pos;
        /* internal Vel = a * v_pec whatever UsePeculiarVelocity: physical peculiar km/s is Vel / a */
        const double v = (s->mass > 0 ? s->mv[d] / s->mass : 0) / ctx->atime;
        rec->CoreVel[d] = rec->CoMVel[d] = v;
    }
    if(mb >= 0) {
        rec->MostBoundID = P[mb].ID;
        for(d = 0; d < 3; d++) {
            rec->MostBoundPos[d] = wrap_box(P[mb].Pos[d] - ctx->Offset[d], ctx->BoxSize);
            rec->MostBoundVel[d] = P[mb].Vel[d] / ctx->atime;
        }
    }
    rec->Alive = 1;
}

/* ------------------------------------------------------------------ serial keys (steps 7a-7e) */

struct DistRec {
    double d;
    MyIDType id;
    int64_t idx;
};

static int
cmp_distrec(const void * a, const void * b)
{
    const struct DistRec * x = (const struct DistRec *) a, * y = (const struct DistRec *) b;
    if(x->d != y->d)
        return (x->d > y->d) - (x->d < y->d);
    return (x->id > y->id) - (x->id < y->id);
}

static void
process_serial_key(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata)
{
    struct SubhaloCallCtx * ctx = (struct SubhaloCallCtx *) userdata;
    const int64_t g = seg->g, start = seg->start, n = seg->n;
    int64_t i;
    int d;

    /* a, b: members and the winner (largest previous Nbound among the members voted into g,
     * ties to the smaller TrackId); the key is sorted by TrackId, so members are runs. */
    int64_t winner = SUBHALO_NOTRACK;
    int64_t bestN = -1;
    const struct SubTrack * wrec = NULL;
    for(i = start; i < start + n; i++) {
        const int64_t T = P[i].TrackId;
        if(T < 0 || (i > start && T == P[i-1].TrackId))
            continue;
        const struct SubTrack * m = subhalo_find_member(ctx, T);
        if(!m)
            endrun(5, "Member %ld of key %ld has no fetched record.\n", T, g);
        if(!member_is_candidate(m, g))
            continue;
        if(m->NboundPrev > bestN || (m->NboundPrev == bestN && T < winner)) {
            bestN = m->NboundPrev;
            winner = T;
            wrec = m;
        }
    }
    const int newborn = (winner == SUBHALO_NOTRACK);
    if(newborn)
        winner = SUBHALO_PROVISIONAL(g);

    /* d: the other members voted into g sink into the winner */
    double maxsink = 0;
    for(i = start; i < start + n; i++) {
        const int64_t T = P[i].TrackId;
        if(T < 0 || T == winner || (i > start && T == P[i-1].TrackId))
            continue;
        const struct SubTrack * m = subhalo_find_member(ctx, T);
        if(!member_is_candidate(m, g))
            continue;
        ctx->SinkWinner[m - ctx->Members] = winner;
        if(m->Mbound > maxsink)
            maxsink = m->Mbound;
    }

    /* c: centre relative to the first particle (deterministic order), then ranks by distance */
    double ref[3];
    for(d = 0; d < 3; d++)
        ref[d] = P[start].Pos[d];
    struct KeySums s;
    key_sums(&s, start, n, ref, ctx->TracerTypes, ctx->BoxSize);
    double cen[3];
    for(d = 0; d < 3; d++)
        cen[d] = ref[d] + (s.mass > 0 ? s.mdx[d] / s.mass : 0);

    struct DistRec * dr = (struct DistRec *) subhalo_arena_get(arena, sizeof(struct DistRec) * n);
    if(!dr)
        endrun(5, "Subhalo arena too small for key %ld (%ld particles).\n", g, n);
    for(i = 0; i < n; i++) {
        const struct particle_data * p = &P[start + i];
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(p->Pos[d] - cen[d], ctx->BoxSize);
            r2 += dx * dx;
        }
        dr[i].d = sqrt(r2);
        dr[i].id = p->ID;
        dr[i].idx = start + i;
    }
    if(nthreads > 1)
        qsort_openmp(dr, n, sizeof(struct DistRec), cmp_distrec);
    else
        qsort(dr, n, sizeof(struct DistRec), cmp_distrec);

    /* 8: tags written straight into P */
    for(i = 0; i < n; i++) {
        struct particle_data * p = &P[dr[i].idx];
        p->TrackId = winner;
        p->BoundTrackId = winner;
        p->BindRank = i;
    }

    /* e: the winner's record */
    struct SubhaloKeyResult * res = &ctx->Res[k];
    fill_winner_record(&res->rec, newborn ? NULL : wrec, winner, g, &s, cen, dr[0].idx, 0, ctx);
    res->MaxSinkMbound = maxsink;
    res->newborn = newborn;
    res->valid = 1;
}

void
subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx)
{
    subhalo_run_serial_keys(segs, nsegs, sizeof(struct DistRec), process_serial_key, ctx);
}

/* ------------------------------------------------------------------ in place (step 9) */

void
subhalo_process_inplace(const struct SubhaloKeySeg * segs, int64_t nsegs, const struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, struct SubTrackTable * tab, MPI_Comm Comm)
{
    const int64_t K = plan->NInplace;
    int64_t i, k, s;
    int d;
    /* every rank has the same list, so this test is collective-safe */
    if(K == 0)
        return;
    const int ThisTask = plan->ThisTask;

    struct KeySums * ls = (struct KeySums *) mymalloc("SubIplSums", sizeof(struct KeySums) * K);
    struct KeySums * gs = (struct KeySums *) mymalloc("SubIplSumsG", sizeof(struct KeySums) * K);
    int64_t * cnt = (int64_t *) mymalloc("SubIplCnt", sizeof(int64_t) * K);
    int64_t * off = (int64_t *) mymalloc("SubIplOff", sizeof(int64_t) * K);
    int64_t * best = (int64_t *) mymalloc("SubIplBest", sizeof(int64_t) * K);
    int64_t * gbest = (int64_t *) mymalloc("SubIplBestG", sizeof(int64_t) * K);
    double * msink = (double *) mymalloc("SubIplSink", sizeof(double) * K);
    double * gmsink = (double *) mymalloc("SubIplSinkG", sizeof(double) * K);
    memset(ls, 0, sizeof(struct KeySums) * K);
    for(k = 0; k < K; k++) {
        cnt[k] = 0;
        best[k] = -1;
        msink[k] = 0;
    }

    /* local sums and the local best candidate (largest previous Nbound) */
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace) continue;
        k = seg->ipindex;
        cnt[k] = seg->n;
        key_sums(&ls[k], seg->start, seg->n, seg->ref, ctx->TracerTypes, ctx->BoxSize);
        for(i = seg->start; i < seg->start + seg->n; i++) {
            const int64_t T = P[i].TrackId;
            if(T < 0 || (i > seg->start && T == P[i-1].TrackId))
                continue;
            const struct SubTrack * m = subhalo_find_member(ctx, T);
            if(!m)
                endrun(5, "Member %ld of in-place key %ld has no fetched record.\n", T, seg->g);
            if(member_is_candidate(m, seg->g) && m->NboundPrev > best[k])
                best[k] = m->NboundPrev;
        }
    }
    MPI_Allreduce(best, gbest, K, MPI_INT64, MPI_MAX, Comm);
    /* among the candidates with the largest previous Nbound, the smallest TrackId */
    for(k = 0; k < K; k++)
        best[k] = INT64_MAX;
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace) continue;
        k = seg->ipindex;
        for(i = seg->start; i < seg->start + seg->n; i++) {
            const int64_t T = P[i].TrackId;
            if(T < 0 || (i > seg->start && T == P[i-1].TrackId))
                continue;
            const struct SubTrack * m = subhalo_find_member(ctx, T);
            if(member_is_candidate(m, seg->g) && m->NboundPrev == gbest[k] && T < best[k])
                best[k] = T;
        }
    }
    int64_t * winner = (int64_t *) mymalloc("SubIplWinner", sizeof(int64_t) * K);
    MPI_Allreduce(best, winner, K, MPI_INT64, MPI_MIN, Comm);
    for(k = 0; k < K; k++)
        if(gbest[k] < 0)
            winner[k] = SUBHALO_PROVISIONAL(plan->Inplace[k].g);

    /* sinking: every holder marks its copies; the copies are identical */
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace) continue;
        k = seg->ipindex;
        for(i = seg->start; i < seg->start + seg->n; i++) {
            const int64_t T = P[i].TrackId;
            if(T < 0 || T == winner[k] || (i > seg->start && T == P[i-1].TrackId))
                continue;
            const struct SubTrack * m = subhalo_find_member(ctx, T);
            if(!member_is_candidate(m, seg->g))
                continue;
            ctx->SinkWinner[m - ctx->Members] = winner[k];
            if(m->Mbound > msink[k])
                msink[k] = m->Mbound;
        }
    }
    MPI_Allreduce(msink, gmsink, K, MPI_DOUBLE, MPI_MAX, Comm);

    /* two-stage centre and the global rank prefix (vector exclusive scan in rank order) */
    MPI_Allreduce(ls, gs, K * KEYSUMS_NDOUBLE, MPI_DOUBLE, MPI_SUM, Comm);
    MPI_Exscan(cnt, off, K, MPI_INT64, MPI_SUM, Comm);
    if(ThisTask == 0)
        for(k = 0; k < K; k++)
            off[k] = 0;

    /* tags */
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace) continue;
        k = seg->ipindex;
        if(off[k] + seg->n > INT32_MAX)
            endrun(5, "In-place key %ld too large for 32-bit ranks.\n", seg->g);
        for(i = 0; i < seg->n; i++) {
            struct particle_data * p = &P[seg->start + i];
            p->TrackId = winner[k];
            p->BoundTrackId = winner[k];
            p->BindRank = off[k] + i;
        }
    }

    /* the winner's record on the deciding (lowest holding) rank; fetch the previous record of
     * a winner whose copy this rank does not hold (collective) */
    int64_t nneed = 0;
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace || seg->decider != ThisTask) continue;
        k = seg->ipindex;
        if(!SUBHALO_IS_PROVISIONAL(winner[k]) && !subhalo_find_member(ctx, winner[k]))
            nneed++;
    }
    int64_t * needids = (int64_t *) mymalloc("SubIplNeed", sizeof(int64_t) * (nneed + 1));
    struct SubTrack * needrec = (struct SubTrack *) mymalloc("SubIplNeedRec", sizeof(struct SubTrack) * (nneed + 1));
    nneed = 0;
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace || seg->decider != ThisTask) continue;
        k = seg->ipindex;
        if(!SUBHALO_IS_PROVISIONAL(winner[k]) && !subhalo_find_member(ctx, winner[k]))
            needids[nneed++] = winner[k];
    }
    subhalo_table_fetch(tab, needids, nneed, needrec, Comm);

    int64_t j = 0;
    for(s = 0; s < nsegs; s++) {
        const struct SubhaloKeySeg * seg = &segs[s];
        if(!seg->inplace || seg->decider != ThisTask) continue;
        k = seg->ipindex;
        const int newborn = SUBHALO_IS_PROVISIONAL(winner[k]);
        const struct SubTrack * old = NULL;
        if(!newborn) {
            old = subhalo_find_member(ctx, winner[k]);
            if(!old)
                old = &needrec[j++];
        }
        double cen[3];
        for(d = 0; d < 3; d++)
            cen[d] = seg->ref[d] + (gs[k].mass > 0 ? gs[k].mdx[d] / gs[k].mass : 0);
        if((int64_t) gs[k].count != seg->Length)
            endrun(5, "In-place key %ld: %ld particles counted, length %ld.\n", seg->g, (int64_t) gs[k].count, seg->Length);
        struct SubhaloKeyResult * res = &ctx->Res[s];
        /* the rank-0 particle is the first one on the lowest holding rank, which is this one */
        fill_winner_record(&res->rec, old, winner[k], seg->g, &gs[k], cen, seg->start, 1, ctx);
        res->MaxSinkMbound = gmsink[k];
        res->newborn = newborn;
        res->valid = 1;
    }

    myfree(needrec);
    myfree(needids);
    myfree(winner);
    myfree(gmsink);
    myfree(msink);
    myfree(gbest);
    myfree(best);
    myfree(off);
    myfree(cnt);
    myfree(gs);
    myfree(ls);
}
