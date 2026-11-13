/* Runtime subhalo finder: the HBT-HERONS recipe per serial key, on one rank, by one worker or by all
 * threads when the key runs alone (subhalo_unbind.h, intra-key threading) (design 7.2 steps 7a-7h
 * and 8, D10 tags), including the sub-tree keys of oversized groups (7.8). The oversized
 * candidates themselves are in subhalo_collective.c, unclaimed tracks in subhalo_orphan.c. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/openmpsort.h"
#include "utils/system.h"

#include "partmanager.h"
#include "subhalo.h"
#include "subhalo_internal.h"
#include "subhalo_unbind.h"

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


/* ------------------------------------------------------------------ result slots */

static int
cmp_seg_g(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloKeySeg *) a)->g, y = ((const struct SubhaloKeySeg *) b)->g;
    return (x > y) - (x < y);
}

/* The key a member is hosted by on this call: a resolved track's Key, an orphan's Host. */
static inline int64_t
hosted_key(const struct SubTrack * m)
{
    return m->NboundPrev > 0 ? m->Key : m->Host;
}

int64_t
subhalo_result_slots(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx)
{
    int64_t i, s;
    ctx->HostedStart = (int64_t *) mymalloc("SubHostedStart", sizeof(int64_t) * (nsegs + 1));
    ctx->ResStart = (int64_t *) mymalloc("SubResStart", sizeof(int64_t) * (nsegs + 1));
    for(s = 0; s <= nsegs; s++)
        ctx->HostedStart[s] = 0;
    /* the segments are sorted by key (the local sort) */
    for(i = 0; i < ctx->NMembers; i++) {
        struct SubhaloKeySeg key;
        key.g = hosted_key(&ctx->Members[i]);
        if(key.g < 0)
            continue;
        const struct SubhaloKeySeg * seg = (const struct SubhaloKeySeg *) bsearch(&key, segs, nsegs, sizeof(key), cmp_seg_g);
        if(seg && !seg->inplace)
            ctx->HostedStart[seg - segs + 1]++;
    }
    for(s = 0; s < nsegs; s++)
        ctx->HostedStart[s + 1] += ctx->HostedStart[s];
    ctx->Hosted = (int64_t *) mymalloc("SubHosted", sizeof(int64_t) * (ctx->HostedStart[nsegs] + 1));
    int64_t * fill = (int64_t *) mymalloc("SubHostedFill", sizeof(int64_t) * (nsegs + 1));
    for(s = 0; s < nsegs; s++)
        fill[s] = ctx->HostedStart[s];
    /* member order (TrackId) within a segment */
    for(i = 0; i < ctx->NMembers; i++) {
        struct SubhaloKeySeg key;
        key.g = hosted_key(&ctx->Members[i]);
        if(key.g < 0)
            continue;
        const struct SubhaloKeySeg * seg = (const struct SubhaloKeySeg *) bsearch(&key, segs, nsegs, sizeof(key), cmp_seg_g);
        if(seg && !seg->inplace)
            ctx->Hosted[fill[seg - segs]++] = i;
    }
    myfree(fill);
    int64_t nres = 0;
    for(s = 0; s < nsegs; s++) {
        ctx->ResStart[s] = nres;
        nres += segs[s].inplace ? 0 : ctx->HostedStart[s + 1] - ctx->HostedStart[s] + 1;
    }
    ctx->ResStart[nsegs] = nres;
    /* one slot per in-place candidate whose record this rank writes */
    ctx->IpResStart = nres;
    ctx->NIpRes = 0;
    if(ctx->plan)
        for(i = 0; i < ctx->plan->NCands; i++)
            if(ctx->plan->Cands[i].owner == ctx->ThisTask)
                ctx->NIpRes++;
    nres += ctx->NIpRes;
    return nres;
}

void
subhalo_free_slots(struct SubhaloCallCtx * ctx)
{
    myfree(ctx->Hosted);
    myfree(ctx->ResStart);
    myfree(ctx->HostedStart);
    ctx->Hosted = ctx->ResStart = ctx->HostedStart = NULL;
}

/* ------------------------------------------------------------------ serial keys (design 7.2 steps 7a-7h, 8) */

_Static_assert(SUBHALO_ARENA_FIXED_BYTES_PER_PART >= SUBHALO_UNBIND_BYTES_PER_PART + 16,
               "SUBHALO_ARENA_FIXED_BYTES_PER_PART must cover the kernel, the owner map and one more list entry per particle");
/* The bound properties run after the final pass inside the kernel's transient allowance. */
_Static_assert(SUBHALO_PROPS_BYTES_PER_PART <= SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART,
               "SUBHALO_PROPS_BYTES_PER_PART must fit in the kernel's transient scratch");

#define TAG_RESERVED 0x40000000

/* One candidate of a key (design 7.1): a member hosted here or the newborn. */
struct SubNode {
    int64_t TrackId;
    const struct SubTrack * old;    /* the member's previous record, NULL for the newborn */
    int kind, state;                /* SUBHALO_MEM_*, SUBHALO_ST_* */
    int sinkinto;                   /* node it sank into, -1 */
    int parent, child, sib, depth;  /* processing tree (children in TrackId order) */
    int visited;                    /* traversal: children entered */
    int fparent, fdepth, rank;      /* final hierarchy over the survivors */
    int seq, firstseq, byseq;       /* list allocation order: own number, first of the subtree, inverse map */
    int has_core;
    int64_t own_off, own_n;         /* own source in the grouped list */
    int admit_recv, admit_src;      /* admission: a receiver; a proper ancestor of one */
    int64_t adm_off, adm_n;         /* admitted particles in the grouped admission list */
    int64_t keep;                   /* list entries kept by the last compaction */
    size_t region;                  /* arena offset where the lists of the subtree start */
    struct SubCandidate c;
    struct SubhaloCore core;
    MyIDType mbID;                  /* the most-bound tracer when it died or sank */
    double mbPos[3], mbVel[3];
    int has_mb;
};
_Static_assert(sizeof(struct SubNode) <= SUBHALO_NODE_BYTES, "SUBHALO_NODE_BYTES does not cover struct SubNode");

struct SerialCtx {
    struct SubhaloCallCtx * ctx;
    struct SubNode * Nodes;          /* parallel to ctx->Res */
    const int64_t * BindStart;       /* per segment: its slice of ctx->BindOrd */
    int64_t * BindCount;             /* per segment: the entries written */
};

static inline int
is_tracer(int64_t i, int TracerTypes)
{
    return (TracerTypes >> P[i].Type) & 1;
}

static void *
arena_need(struct SubhaloArena * a, size_t bytes, int64_t g, const char * what)
{
    void * p = subhalo_arena_get(a, bytes);
    if(!p)
        endrun(1, "Subhalo arena exhausted in key %ld allocating %s (%lu bytes, %lu of %lu used); "
                  "the nest tree needs more list space than subhalo_arena_bytes_per_part() provides.\n",
               g, what, (unsigned long) bytes, (unsigned long) a->used, (unsigned long) a->size);
    return p;
}

static inline size_t
align16(size_t b)
{
    return (b + 15) & ~((size_t) 15);
}

/* Seed of the subsampling RNG from (TrackId, SnapNum): rank- and thread-independent. */
static uint64_t
node_seed(int64_t TrackId, int SnapNum)
{
    uint64_t z = (uint64_t) TrackId * 0x9E3779B97F4A7C15ULL + (uint64_t) SnapNum * 0xD1B54A32D192ED03ULL + 0x2545F4914F6CDD1DULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int
node_index(const struct SubNode * nd, int nn, int64_t TrackId)
{
    int lo = 0, hi = nn;
    while(lo < hi) {
        int mid = (lo + hi) / 2;
        if(nd[mid].TrackId < TrackId) lo = mid + 1;
        else hi = mid;
    }
    return (lo < nn && nd[lo].TrackId == TrackId) ? lo : -1;
}

/* |v1 - v2 + H a dx| with v in internal Vel units (a v_pec), physical km/s-like units
 * (HBT KineticDistance / RelativeVelocity, subhalo_tracking.cpp:33-39). */
static double
kinetic_distance(const double x1[3], const double v1[3], const double x2[3], const double v2[3], const struct SubhaloUnbindParams * up)
{
    double s = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double dx = NEAREST(x1[d] - x2[d], up->BoxSize);
        const double dv = (v1[d] - v2[d]) / up->atime + up->Hubble * up->atime * dx;
        s += dv * dv;
    }
    return sqrt(s);
}

/* An orphan's tracer in the translated frame and internal Vel units; 0 if its tracer is lost. */
static int
orphan_tracer(const struct SubTrack * m, const struct SubhaloCallCtx * ctx, double pos[3], double vel[3])
{
    if(m->SnapTracerLost >= 0)
        return 0;
    const double vfac = subhalo_vel_to_kms(ctx->atime);
    int d;
    for(d = 0; d < 3; d++) {
        double x = m->TracerPos[d] + ctx->Offset[d];
        while(x >= ctx->BoxSize) x -= ctx->BoxSize;
        while(x < 0) x += ctx->BoxSize;
        pos[d] = x;
        vel[d] = m->TracerVel[d] / vfac;
    }
    return 1;
}

static void
set_mb_from(struct SubNode * v, int64_t p)
{
    int d;
    v->mbID = P[p].ID;
    for(d = 0; d < 3; d++) {
        v->mbPos[d] = P[p].Pos[d];
        v->mbVel[d] = P[p].Vel[d];
    }
    v->has_mb = 1;
}

/* The state of x after either of its unbindings; a resolved track that died takes its previous
 * most-bound tracer, the first own-tag tracer of its own source in BindRank order (as dead_mb). */
static void
node_unbound(struct SubNode * x, int alive, const int32_t * ownlist, int64_t start, int TracerTypes)
{
    int64_t j;
    if(alive) {
        x->state = SUBHALO_ST_ALIVE;
        return;
    }
    if(x->kind != SUBHALO_MEM_RESOLVED) {
        x->state = SUBHALO_ST_FAILED;
        return;
    }
    x->state = SUBHALO_ST_DIED;
    for(j = 0; j < x->own_n; j++) {
        const int64_t pi = start + ownlist[x->own_off + j];
        if(P[pi].TrackId == x->TrackId && is_tracer(pi, TracerTypes)) {
            set_mb_from(x, pi);
            return;
        }
    }
}

/* The core used by the sinking test: of the bound set for a survivor (subhalo_core); for a
 * track that just died, its previous most-bound tracer with zero dispersion (HBT
 * GetCorePhaseSpaceProperties for Nbound == 0, subhalo_merge.cpp:93-104). */
static void
node_core(struct SubNode * v, const struct SubhaloUnbindParams * up, int CoreSize)
{
    int d;
    if(v->state == SUBHALO_ST_ALIVE) {
        subhalo_core(&v->c, up, CoreSize, v->core.Pos, v->core.Vel, &v->core.SigmaX, &v->core.SigmaV);
        v->has_core = 1;
    }
    else if(v->state == SUBHALO_ST_DIED && v->has_mb) {
        for(d = 0; d < 3; d++) {
            v->core.Pos[d] = v->mbPos[d];
            v->core.Vel[d] = v->mbVel[d];
        }
        v->core.SigmaX = v->core.SigmaV = 0;
        v->has_core = 1;
    }
    else
        v->has_core = 0;
}

/* 1 if b is v or an ancestor of v in the final hierarchy (fparent over the survivors). */
static int
final_ancestor(const struct SubNode * nd, int v, int b, int nn)
{
    int u = v, steps = 0;
    while(u >= 0 && steps++ <= nn) {
        if(u == b)
            return 1;
        u = nd[u].fparent;
    }
    return 0;
}

/* One unbinding of track T in the diagnostics: counted, with its iteration cap and convergence ratio. */
static void
count_unbinding(int itercap, double ratio, int64_t T, struct SubhaloCallStats * ks)
{
    ks->NCandidates++;
    if(itercap) {
        ks->NItercap++;
        if(ks->WorstTrackId == -1 || ratio < ks->WorstConvRatio) {
            ks->WorstConvRatio = ratio;
            ks->WorstTrackId = T;
        }
    }
}

/* subhalo_unbind with the diagnostics: every unbinding counted with its own iteration cap (the ITERCAP flag
 * of this run only, ORed back afterwards). A resolved track's candidate (old) dead from its mean frame, or
 * alive below SUBHALO_RESCUE_LOSS of NPrev, gets the second attempt of subhalo_unbind_rescue. */
static int
unbind_counted(struct SubCandidate * c, const struct SubTrack * old, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena,
               struct SubhaloCallStats * ks)
{
    const int f0 = c->Flags;
    c->Flags = 0;
    int alive = subhalo_unbind(c, up, arena);
    count_unbinding((c->Flags & SUBHALO_FLAG_ITERCAP) != 0, c->ConvergenceRatio, c->TrackId, ks);
    if(old && old->NPrev > 0) {
        const int first = alive;
        struct SubhaloSecondAttempt sa;
        alive = subhalo_unbind_rescue(c, first, old->NPrev, subhalo_rescue_core_size(old->NboundPrev, subhalo_get_params()->SubhaloCoreSize),
                                      up, arena, &sa);
        if(sa.tried) {
            count_unbinding(sa.Itercap, sa.ConvergenceRatio, c->TrackId, ks);
            if(first) {
                ks->NLossTried++;
                ks->NLossRescued += sa.stood;
            }
            else {
                ks->NRescueTried++;
                ks->NRescued += sa.stood;
            }
        }
    }
    c->Flags |= f0;
    return alive;
}

/* Add a key's (or a rank's) diagnostics to the totals; thread-safe. */
static void
stats_merge(struct SubhaloCallStats * to, const struct SubhaloCallStats * ks)
{
    #pragma omp critical(subhalo_stats)
    {
        to->NCandidates += ks->NCandidates;
        to->NItercap += ks->NItercap;
        to->NFinalPositive += ks->NFinalPositive;
        to->NFinalBound += ks->NFinalBound;
        to->NTailRejected += ks->NTailRejected;
        to->NFallbackField += ks->NFallbackField;
        to->NFallbackReleased += ks->NFallbackReleased;
        to->NRescueTried += ks->NRescueTried;
        to->NRescued += ks->NRescued;
        to->NLossTried += ks->NLossTried;
        to->NLossRescued += ks->NLossRescued;
        if(ks->NItercap > 0 && (to->WorstTrackId == -1 || ks->WorstConvRatio < to->WorstConvRatio
                                || (ks->WorstConvRatio == to->WorstConvRatio && ks->WorstTrackId < to->WorstTrackId))) {
            to->WorstConvRatio = ks->WorstConvRatio;
            to->WorstTrackId = ks->WorstTrackId;
        }
        to->NAdmitted += ks->NAdmitted;
        to->NAdmitTaken += ks->NAdmitTaken;
        to->NAdmitReceivers += ks->NAdmitReceivers;
        to->NAdmitWithheld += ks->NAdmitWithheld;
    }
}

/* HBT MergeRecursively (subhalo_merge.cpp:41-91): sink the subtree below x into ref, children
 * first. Only tracks resolved at the call's start take part (an older orphan sunk here would get
 * SnapSink != SnapDeath). Returns 1 if a survivor sank (its bound set appended to ref's list). */
static int
merge_subtree(struct SubNode * nd, int x, int ref, const struct SubhaloUnbindParams * up, int symmetric, int TracerTypes)
{
    int merged = 0, y;
    for(y = nd[x].child; y >= 0; y = nd[y].sib)
        merged |= merge_subtree(nd, y, ref, up, symmetric, TracerTypes);
    if(x == ref)
        return merged;
    struct SubNode * u = &nd[x];
    if(u->kind != SUBHALO_MEM_RESOLVED || !(u->state == SUBHALO_ST_ALIVE || u->state == SUBHALO_ST_DIED) || !u->has_core)
        return merged;
    const struct SubNode * r = &nd[ref];
    if(!subhalo_sink_test(&u->core, &r->core, symmetric, up->BoxSize))
        return merged;
    if(u->state == SUBHALO_ST_ALIVE) {
        /* the most-bound tracer of the bound set becomes the orphan's tracer (MergeTo,
         * subhalo_merge.cpp:213-221), then the bound set joins the reference */
        int64_t j;
        for(j = 0; j < u->c.Nbound; j++)
            if(is_tracer(u->c.Index[j], TracerTypes)) {
                set_mb_from(u, u->c.Index[j]);
                break;
            }
        if(!u->has_mb && u->c.Nbound > 0)
            set_mb_from(u, u->c.Index[0]);
        struct SubNode * rr = &nd[ref];
        for(j = 0; j < u->c.Nbound; j++)
            rr->c.Index[rr->c.N++] = u->c.Index[j];
        u->c.Nbound = 0;
        merged = 1;
    }
    u->state = SUBHALO_ST_SUNK;
    u->sinkinto = ref;
    return merged;
}

/* HBT CompareMass (subhalo_tracking.cpp:112-139): Mbound, Vmax, MostBoundID, TrackId, all
 * descending; returns 1 if a ranks before b. */
static int
mass_before(const struct SubTrack * a, const struct SubTrack * b)
{
    if(a->Mbound != b->Mbound) return a->Mbound > b->Mbound;
    if(a->Vmax != b->Vmax) return a->Vmax > b->Vmax;
    if(a->MostBoundID != b->MostBoundID) return a->MostBoundID > b->MostBoundID;
    return a->TrackId > b->TrackId;
}

/* The identity and history columns that a new record takes over from the previous one. */
void
subhalo_track_carry(struct SubTrack * rec, const struct SubTrack * old, int64_t TrackId, int SnapNum)
{
    int d;
    subhalo_track_clear(rec, TrackId);
    if(!old) {
        rec->SnapBirth = SnapNum;
        return;
    }
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
    /* the followed set stays cleared: a carried record is resolved */
    rec->NboundPrev = old->NboundPrev;
    rec->Host = old->Host;
    rec->Key = old->Key;
}

/* MostBound* of a record from a particle (physical frame, km/s). */
static void
record_mb(struct SubTrack * rec, const struct SubNode * v, const struct SubhaloCallCtx * ctx)
{
    if(!v->has_mb)
        return;
    const double vfac = subhalo_vel_to_kms(ctx->atime);
    int d;
    rec->MostBoundID = v->mbID;
    for(d = 0; d < 3; d++) {
        double x = v->mbPos[d] - ctx->Offset[d];
        while(x >= ctx->BoxSize) x -= ctx->BoxSize;
        while(x < 0) x += ctx->BoxSize;
        rec->MostBoundPos[d] = x;
        rec->MostBoundVel[d] = v->mbVel[d] * vfac;
    }
}

struct IdLocal {
    MyIDType ID;
    int64_t i;
};

static int
cmp_idlocal(const void * a, const void * b)
{
    const struct IdLocal * x = (const struct IdLocal *) a, * y = (const struct IdLocal *) b;
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->i > y->i) - (x->i < y->i);
}

/* An orphan that survived as its key's central this call (HBT resets its death and sink,
 * subhalo_unbind.cpp:697-707). */
static inline int
x_is_resurrected(const struct SubNode * x)
{
    return x->kind == SUBHALO_MEM_ORPHAN && x->state == SUBHALO_ST_ALIVE;
}

/* The key centre of an ordinary or pseudo-key: mass-weighted over its live FOF particles (all
 * particles of a pseudo-key), periodic about the first one; xg, vg untouched if there are none. */
static void
key_fof_centre(int64_t start, int64_t n, int64_t g, int pseudo, const struct SubhaloCallCtx * ctx, double xg[3], double vg[3])
{
    int64_t i;
    int d;
    int64_t first = -1;
    double M = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0};
    for(i = start; i < start + n; i++) {
        if(!subhalo_part_live(&P[i]) || !(pseudo || P[i].GrNr == g))
            continue;
        if(first < 0)
            first = i;
        const double m = P[i].Mass;
        M += m;
        for(d = 0; d < 3; d++) {
            mx[d] += m * NEAREST(P[i].Pos[d] - P[first].Pos[d], ctx->BoxSize);
            mv[d] += m * P[i].Vel[d];
        }
    }
    if(first >= 0 && M > 0)
        for(d = 0; d < 3; d++) {
            double x = P[first].Pos[d] + mx[d] / M;
            while(x >= ctx->BoxSize) x -= ctx->BoxSize;
            while(x < 0) x += ctx->BoxSize;
            xg[d] = x;
            vg[d] = mv[d] / M;
        }
}

/* ReassignParticles (subhalo_reassign_gas.cpp) for the non-tracers of the owner map: a particle
 * none of whose k nearest tracers shares its owner moves to the nearest tracer's owner. */
static void
reassign_particles(int32_t * owner, int64_t start, int64_t n, int TracerTypes, const struct SubhaloParams * sp,
                   const struct SubhaloCallCtx * ctx, struct SubhaloArena * arena, int64_t g)
{
    int64_t i;
    int64_t nt = 0, nq = 0;
    for(i = 0; i < n; i++) {
        if(owner[i] < 0) continue;
        if(is_tracer(start + i, TracerTypes)) nt++;
        else nq++;
    }
    if(nt > 0 && nq > 0) {
        const size_t mark = arena->used;
        const int kk = sp->SubhaloNumNeighboursForReassignment;
        int64_t * tr = (int64_t *) arena_need(arena, sizeof(int64_t) * nt, g, "reassignment tracers");
        nt = 0;
        for(i = 0; i < n; i++)
            if(owner[i] >= 0 && is_tracer(start + i, TracerTypes))
                tr[nt++] = start + i;
        /* query batches of at most 4096, buffers from a quarter of the space left, taken
         * before the tree, whose build may use all the rest for its nodes */
        int64_t B = (int64_t) ((arena->size - arena->used) / 4 / (8 + 8 * (size_t) kk + 32));
        if(B > 4096) B = 4096;
        if(B > nq) B = nq;
        if(B < 1) B = 1;
        int64_t * q = (int64_t *) arena_need(arena, sizeof(int64_t) * B, g, "reassignment queries");
        int64_t * out = (int64_t *) arena_need(arena, sizeof(int64_t) * B * kk, g, "reassignment neighbours");
        /* one tree per key (HBT builds one per group), queried in batches */
        struct SubhaloKnnTree * kt = subhalo_knn_build(tr, nt, ctx->BoxSize, arena);
        int64_t nb = 0;
        for(i = 0; i <= n; i++) {
            if(i < n && owner[i] >= 0 && !is_tracer(start + i, TracerTypes))
                q[nb++] = start + i;
            if(nb == B || (i == n && nb > 0)) {
                subhalo_knn_query(kt, q, nb, kk, out);
                int64_t a;
                for(a = 0; a < nb; a++) {
                    const int own = owner[q[a] - start];
                    int found = 0, c;
                    for(c = 0; c < kk; c++)
                        if(out[a * kk + c] >= 0 && owner[out[a * kk + c] - start] == own)
                            found = 1;
                    /* the nearest tracer (first of the row) decides */
                    if(!found && out[a * kk] >= 0) {
                        const int o2 = owner[out[a * kk] - start];
                        if(o2 >= 0 && o2 != own)
                            owner[q[a] - start] = o2;
                    }
                }
                nb = 0;
            }
        }
        arena->used = mark;
    }
}

struct SerialAdmitTree {
    const struct SubNode * nd;
    int nn;
    double MinD, BoxSize;   /* SubhaloAdmitMinCoreDistance and the box of the withheld test */
};

/* The admission map (the owner map's memory once the own sources are grouped): the receiver's node
 * of a particle admitted and not yet decided, ADM_TAKEN once its receiver bound it, -1 otherwise. */
#define ADM_TAKEN INT32_MAX

static subhalo_admit_hook AdmitHook = NULL;

void
subhalo_set_admit_hook(subhalo_admit_hook fn)
{
    AdmitHook = fn;
}

void
subhalo_admit_report(MyIDType ID, int64_t Receiver, int source)
{
    if(!AdmitHook)
        return;
    /* keys run concurrently on threads */
    #pragma omp critical(subhalo_admit_hook)
    AdmitHook(ID, Receiver, source);
}

/* 1 if a is a proper ancestor of v in the processing tree. */
static int
admit_ancestor(const struct SubhaloAdmitTree * at, int64_t a, int64_t v)
{
    const int da = at->depth(at->t, a);
    int64_t u;
    for(u = at->parent(at->t, v); u >= 0 && at->depth(at->t, u) >= da; u = at->parent(at->t, u))
        if(u == a)
            return 1;
    return 0;
}

int __attribute__((noinline))
subhalo_admit_withheld(const double sPos[3], const float sVel[3], const double aPos[3], const float aVel[3],
                       float aSigmaX, float aSigmaV, double MinD, double BoxSize)
{
    /* a core without a positive dispersion has no scale to be near in */
    if(!(aSigmaX > 0 && aSigmaV > 0))
        return 0;
    const double sv[3] = {sVel[0], sVel[1], sVel[2]}, av[3] = {aVel[0], aVel[1], aVel[2]};
    return subhalo_core_distance(sPos, sv, aPos, av, aSigmaX, aSigmaV, BoxSize) < MinD;
}

int64_t
subhalo_admit_choice(const int64_t * btid, int kk, int64_t A, int need, const struct SubhaloAdmitTree * at)
{
    int64_t lab[SUBHALO_KNN_MAXK];
    int nl = 0, c, a, b;
    for(c = 0; c < kk; c++) {
        const int64_t r = btid[c] >= 0 ? at->receiver(at->t, btid[c]) : -1;
        if(r >= 0)
            lab[nl++] = r;
    }
    int64_t best = -1;
    int bestv = 0;
    for(a = 0; a < nl; a++) {
        const int64_t r = lab[a];
        int votes = 0, seen = 0;
        for(b = 0; b < nl; b++)
            if(lab[b] == r) {
                votes++;
                seen |= b < a;
            }
        if(seen || !admit_ancestor(at, A, r))
            continue;
        if(best >= 0 && votes == bestv) {
            const int dr = at->depth(at->t, r), db = at->depth(at->t, best);
            if(dr < db || (dr == db && at->trackid(at->t, r) > at->trackid(at->t, best)))
                continue;
        }
        else if(best >= 0 && votes < bestv)
            continue;
        best = r;
        bestv = votes;
    }
    if(bestv < need)
        return -1;
    return at->withheld && at->withheld(at->t, best, A) ? SUBHALO_ADMIT_WITHHELD : best;
}

/* The admission tree of a serial key: its nodes, receivers flagged by admit_recv. */
static int64_t
key_receiver(const void * t, int64_t TrackId)
{
    const struct SerialAdmitTree * kt = (const struct SerialAdmitTree *) t;
    const int r = node_index(kt->nd, kt->nn, TrackId);
    return r >= 0 && kt->nd[r].admit_recv ? r : -1;
}
static int64_t
key_parent(const void * t, int64_t v)
{
    return ((const struct SerialAdmitTree *) t)->nd[v].parent;
}
static int
key_depth(const void * t, int64_t v)
{
    return ((const struct SerialAdmitTree *) t)->nd[v].depth;
}
static int64_t
key_trackid(const void * t, int64_t v)
{
    return ((const struct SerialAdmitTree *) t)->nd[v].TrackId;
}
/* the stored records of the previous call, as the plan's member table holds them */
static int
key_withheld(const void * t, int64_t r, int64_t A)
{
    const struct SerialAdmitTree * kt = (const struct SerialAdmitTree *) t;
    const struct SubTrack * s = kt->nd[r].old, * a = kt->nd[A].old;
    return subhalo_admit_withheld(s->CorePos, s->CoreVel, a->CorePos, a->CoreVel, a->CoreSigmaX, a->CoreSigmaV, kt->MinD, kt->BoxSize);
}

/* The choice of a serial key from a row of the k + 1 nearest tracers (self dropped by value
 * wherever its tie order puts it): the k others' previous tags. */
static int
admit_choice(const int64_t * row, int64_t self, int kk, int A, const struct SubhaloAdmitTree * at, int need)
{
    int64_t btid[SUBHALO_KNN_MAXK];
    int used = 0, c;
    for(c = 0; c <= kk && used < kk; c++) {
        const int64_t p = row[c];
        if(p < 0 || p == self)
            continue;
        btid[used++] = P[p].BoundTrackId;
    }
    for(c = used; c < kk; c++)
        btid[c] = -1;
    return (int) subhalo_admit_choice(btid, kk, A, need, at);
}

/* A serial key as its admission sees it: the owner map of its particles P[start, start + n), its
 * nodes and its central's node; filled by the admission: the admitted particles grouped by receiver
 * (admlist, at the receivers' adm_off, adm_n) and the votes withheld near their owner's core. */
struct AdmitKey {
    int32_t * owner;
    struct SubNode * nd;
    int nn, ci;
    int64_t g, start, n;
    int32_t * admlist;
    int64_t nwithheld;
};

/* The admitted particles (owner map entries -2 - receiver) grouped by receiver in key order into
 * ak->admlist (adm_off, adm_n); the owner map becomes the admission map (receiver, -1 otherwise). */
static void
admit_lists(struct AdmitKey * ak, int64_t nadm, struct SubhaloArena * arena)
{
    int32_t * owner = ak->owner;
    struct SubNode * nd = ak->nd;
    const int nn = ak->nn;
    const int64_t n = ak->n;
    int64_t i, o = 0;
    int v;
    for(i = 0; i < n; i++)
        if(owner[i] <= -2)
            nd[-2 - owner[i]].adm_n++;
    for(v = 0; v < nn; v++) {
        nd[v].adm_off = o;
        o += nd[v].adm_n;
        nd[v].adm_n = 0;
    }
    int32_t * al = (int32_t *) arena_need(arena, sizeof(int32_t) * (nadm + 1), ak->g, "admission lists");
    for(i = 0; i < n; i++) {
        if(owner[i] <= -2) {
            struct SubNode * r = &nd[-2 - owner[i]];
            al[r->adm_off + r->adm_n++] = (int32_t) i;
            owner[i] = -2 - owner[i];
        }
        else
            owner[i] = -1;
    }
    ak->admlist = al;
}

static int
cmp_admitrec_id(const void * a, const void * b)
{
    const struct SubhaloAdmitRec * x = (const struct SubhaloAdmitRec *) a, * y = (const struct SubhaloAdmitRec *) b;
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* Admission by neighbour majority (SubhaloAdmitShare = s): a tracer owned by node A joins the source of
 * the receiver below A to which at least the share s of its k nearest owned tracers were bound at the
 * previous call (BoundTrackId, not yet rewritten). Receivers are the resolved members with a previous
 * bound set below the key's root. The vote of a whole key (a group or pseudo-key processed serially);
 * an oversized group's is taken by the plan across ranks (plan_vote). Returns the count; the owner map
 * becomes the admission map and ak->admlist holds the admitted particles grouped by receiver
 * (adm_off, adm_n), in key order. */
static int64_t
admit_particles(struct AdmitKey * ak, const struct SubhaloParams * sp, const struct SubhaloCallCtx * ctx, struct SubhaloArena * arena)
{
    int32_t * owner = ak->owner;
    struct SubNode * nd = ak->nd;
    const int nn = ak->nn, ci = ak->ci, TracerTypes = ctx->TracerTypes;
    const int64_t g = ak->g, start = ak->start, n = ak->n;
    const int kk = sp->SubhaloNumNeighboursForReassignment;
    const int need = subhalo_admit_need(sp->SubhaloAdmitShare, kk);
    const struct SerialAdmitTree st = {nd, nn, sp->SubhaloAdmitMinCoreDistance, ctx->BoxSize};
    const struct SubhaloAdmitTree at = {key_receiver, key_parent, key_depth, key_trackid, &st,
                                        sp->SubhaloAdmitMinCoreDistance > 0 ? key_withheld : NULL};
    int64_t i, nt = 0, nc = 0, nadm = 0;
    int v, u, nrecv = 0;
    for(v = 0; v < nn; v++)
        if(v != ci && nd[v].kind == SUBHALO_MEM_RESOLVED && nd[v].old->NPrev > 0) {
            nd[v].admit_recv = 1;
            nrecv++;
            for(u = nd[v].parent; u >= 0; u = nd[u].parent)
                nd[u].admit_src = 1;
        }
    if(nrecv == 0)
        return 0;
    for(i = 0; i < n; i++)
        if(owner[i] >= 0 && is_tracer(start + i, TracerTypes)) {
            nt++;
            nc += nd[owner[i]].admit_src;
        }
    /* fewer than k other tracers in the key: no admission */
    if(nc == 0 || nt <= kk)
        return 0;
    const size_t mark = arena->used;
    const int kq = kk + 1;
    int64_t * tr = (int64_t *) arena_need(arena, sizeof(int64_t) * nt, g, "admission tracers");
    nt = 0;
    for(i = 0; i < n; i++)
        if(owner[i] >= 0 && is_tracer(start + i, TracerTypes))
            tr[nt++] = start + i;
    /* query batches from a quarter of the space left, taken before the tree (as reassign_particles) */
    int64_t B = (int64_t) ((arena->size - arena->used) / 4 / (8 + 8 * (size_t) kq + 4 + 48));
    if(B > 65536) B = 65536;
    if(B > nc) B = nc;
    if(B < 1) B = 1;
    int64_t * q = (int64_t *) arena_need(arena, sizeof(int64_t) * B, g, "admission queries");
    int64_t * out = (int64_t *) arena_need(arena, sizeof(int64_t) * B * kq, g, "admission neighbours");
    int32_t * got = (int32_t *) arena_need(arena, sizeof(int32_t) * B, g, "admission choices");
    const struct SubhaloKnnTree * kt = subhalo_knn_build(tr, nt, ctx->BoxSize, arena);
    int64_t nb = 0, a;
    for(i = 0; i <= n; i++) {
        if(i < n && owner[i] >= 0 && nd[owner[i]].admit_src && is_tracer(start + i, TracerTypes))
            q[nb++] = start + i;
        if(nb == B || (i == n && nb > 0)) {
            /* the queries and the votes split over the threads in a key run alone */
            subhalo_knn_query(kt, q, nb, kq, out);
            if(nb > subhalo_unbind_get_par_min() && !omp_in_parallel() && omp_get_max_threads() > 1) {
                #pragma omp parallel for schedule(static)
                for(a = 0; a < nb; a++)
                    got[a] = admit_choice(out + a * kq, q[a], kk, owner[q[a] - start], &at, need);
            }
            else
                for(a = 0; a < nb; a++)
                    got[a] = admit_choice(out + a * kq, q[a], kk, owner[q[a] - start], &at, need);
            /* a decided candidate is never read again: its entry holds -2 - receiver */
            for(a = 0; a < nb; a++)
                if(got[a] >= 0) {
                    owner[q[a] - start] = -2 - got[a];
                    subhalo_admit_report(P[q[a]].ID, nd[got[a]].TrackId, SUBHALO_ADMIT_KEY);
                    nadm++;
                }
                else if(got[a] == SUBHALO_ADMIT_WITHHELD)
                    ak->nwithheld++;
            nb = 0;
        }
    }
    arena->used = mark;
    if(nadm == 0)
        return 0;
    admit_lists(ak, nadm, arena);
    return nadm;
}

/* The admission of a serial sub-tree key: the offers of its oversized group's vote, found by the
 * mark in GrNr and the record of the particle's ID (the key's root receives too, from above it).
 * Marks the owner map as admit_particles does and returns the count. */
static int64_t
admit_records(struct AdmitKey * ak, const struct SubhaloPlan * plan, struct SubhaloArena * arena)
{
    const int64_t g = ak->g, start = ak->start, n = ak->n;
    int64_t i, nadm = 0;
    for(i = 0; i < n; i++) {
        const struct particle_data * p = &P[start + i];
        if(!(p->GrNr < -1 && ((-(p->GrNr + 2)) & SUBHALO_KEY_OFFERBIT)))
            continue;
        struct SubhaloAdmitRec key;
        key.ID = p->ID;
        const struct SubhaloAdmitRec * rec = (const struct SubhaloAdmitRec *)
            bsearch(&key, plan->AdmitRecs, plan->NAdmitRecs, sizeof(struct SubhaloAdmitRec), cmp_admitrec_id);
        const int r = rec ? node_index(ak->nd, ak->nn, rec->Receiver) : -1;
        if(r < 0)
            endrun(5, "Sub-tree key %ld: offered particle %ld has no record or a receiver %ld outside the key.\n",
                   g, (long) p->ID, rec ? rec->Receiver : -1L);
        ak->owner[i] = -2 - r;
        nadm++;
    }
    if(nadm == 0)
        return 0;
    admit_lists(ak, nadm, arena);
    return nadm;
}

/* One serial key while it runs (design 7.2): its slice P[start, start + n), its candidates'
 * nodes and result slots, its centre, and the arena tables that its steps hand on. The steps
 * with floating-point code are always_inline: they keep the key's -ffast-math code generation. */
struct KeyRun {
    struct SerialCtx * sc;
    struct SubhaloCallCtx * ctx;
    const struct SubhaloUnbindParams * up;
    const struct SubhaloParams * sp;
    struct SubhaloArena * arena;
    int64_t k, g, start, n;
    int subkey;                     /* a serial sub-tree key of the oversized group gfof (its particles' GrNr is encoded) */
    int pseudo;
    const struct SubhaloSubKey * skey;
    int64_t gfof;
    int TracerTypes;
    struct SubNode * nd;            /* nn nodes in TrackId order; ci the central, a sub-tree key's root */
    struct SubhaloResult * res;
    int nn, ci;
    double xg[3], vg[3];            /* the key centre */
    int32_t * owner;                /* the owner map, then the tag map of step 8 */
    int32_t * ownlist;              /* the own sources grouped by owner, then the binder map of step 8 */
    int32_t * adm;                  /* the admission map, NULL without admitted particles */
    const int32_t * admlist;
    int nseq;                       /* nodes unbound so far (step d) */
    int nalive, X;                  /* survivors and the most massive one (-1) */
    int32_t rootkeep;               /* a sub-tree root's tail inside its relaxed source */
    struct SubhaloCallStats ks;     /* this key's diagnostics, added to ctx->Stats at the end */
};

static void
key_open(struct KeyRun * K, const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, struct SerialCtx * sc)
{
    struct SubhaloCallCtx * ctx = sc->ctx;
    memset(K, 0, sizeof(*K));
    K->sc = sc;
    K->ctx = ctx;
    K->up = ctx->up;
    K->sp = subhalo_get_params();
    K->arena = arena;
    K->k = k;
    K->g = seg->g;
    K->start = seg->start;
    K->n = seg->n;
    K->subkey = ctx->plan && K->g >= ctx->plan->SubKeyBase;
    K->skey = K->subkey ? &ctx->plan->SubKeys[K->g - ctx->plan->SubKeyBase] : NULL;
    K->gfof = K->subkey ? K->skey->g : K->g;
    K->pseudo = !K->subkey && K->g > ctx->TotNgroups;
    K->TracerTypes = ctx->TracerTypes;
    K->nd = sc->Nodes + ctx->ResStart[k];
    K->res = ctx->Res + ctx->ResStart[k];
    K->X = -1;
    K->ks.WorstConvRatio = 1;
    K->ks.WorstTrackId = -1;
    if(K->n > INT32_MAX - 1)
        endrun(1, "Serial key %ld has %ld particles, more than the int32 bookkeeping of a key allows.\n", K->g, K->n);
}

/* Step a: the candidates, the members hosted by this key (sorted by TrackId), else the newborn. */
static void
key_nodes(struct KeyRun * K)
{
    const struct SubhaloCallCtx * ctx = K->ctx;
    const int64_t h0 = ctx->HostedStart[K->k], nh = ctx->HostedStart[K->k + 1] - h0;
    struct SubNode * nd = K->nd;
    int64_t j;
    int v, nn = 0;
    for(j = 0; j < nh; j++) {
        const struct SubTrack * m = &ctx->Members[ctx->Hosted[h0 + j]];
        struct SubNode * u = &nd[nn++];
        memset(u, 0, sizeof(*u));
        u->TrackId = m->TrackId;
        u->old = m;
        u->kind = m->NboundPrev > 0 ? SUBHALO_MEM_RESOLVED : SUBHALO_MEM_ORPHAN;
    }
    if(nn == 0) {
        if(K->pseudo || K->subkey)
            endrun(5, "%s %ld has no member track on rank %d.\n", K->pseudo ? "Pseudo-key" : "Sub-tree key", K->g, ctx->ThisTask);
        memset(&nd[0], 0, sizeof(nd[0]));
        nd[0].TrackId = SUBHALO_PROVISIONAL(K->g);
        nd[0].kind = SUBHALO_MEM_NEWBORN;
        nn = 1;
    }
    for(v = 0; v < nn; v++) {
        nd[v].parent = nd[v].child = nd[v].sib = nd[v].fparent = -1;
        nd[v].sinkinto = -1;
        nd[v].state = SUBHALO_ST_NONE;
        K->res[v].valid = 0;
    }
    K->nn = nn;
}

/* The key's centre: a sub-tree key takes its whole group's from the plan. */
static inline __attribute__((always_inline)) void
key_centre(struct KeyRun * K)
{
    int d;
    if(K->subkey) {
        const struct SubhaloInplaceKey * ik = &K->ctx->plan->Inplace[K->skey->gidx];
        for(d = 0; d < 3; d++) {
            K->xg[d] = ik->cen[d];
            K->vg[d] = ik->vel[d];
        }
    }
    else
        key_fof_centre(K->start, K->n, K->g, K->pseudo, K->ctx, K->xg, K->vg);
}

/* Step b: the central (DecideCentrals, subhalo_tracking.cpp:767-800); a sub-tree key's root is the
 * top of its hierarchy (not a central). */
static inline __attribute__((always_inline)) int
key_central(const struct KeyRun * K)
{
    const struct SubNode * nd = K->nd;
    const int nn = K->nn;
    int v;
    if(K->subkey) {
        const int64_t root = K->ctx->plan->Members[K->skey->root].TrackId;
        const int ci = node_index(nd, nn, root);
        if(ci < 0)
            endrun(5, "Sub-tree key %ld on rank %d does not hold its root %ld.\n", K->g, K->ctx->ThisTask, root);
        return ci;
    }
    double Mmax = -1;
    int nres = 0;
    for(v = 0; v < nn; v++)
        if(nd[v].kind == SUBHALO_MEM_RESOLVED) {
            nres++;
            if(nd[v].old->Mbound > Mmax)
                Mmax = nd[v].old->Mbound;
        }
    double best = 1e300;
    int bi = -1;
    for(v = 0; v < nn; v++) {
        const struct SubTrack * m = nd[v].old;
        double dist;
        if(nres > 0) {
            if(nd[v].kind != SUBHALO_MEM_RESOLVED || m->Mbound < K->sp->SubhaloMajorProgenitorMassRatio * Mmax)
                continue;
            dist = m->NPrev > 0 ? kinetic_distance(m->RefPos, m->RefVel, K->xg, K->vg, K->up) : 1e299;
            /* nodes are in TrackId order: strict < keeps the smaller TrackId on ties */
            if(bi < 0 || dist < best) {
                best = dist;
                bi = v;
            }
        }
        else if(nd[v].kind == SUBHALO_MEM_ORPHAN) {
            /* only orphans: every one is a candidate (their Mbound is 0); the largest weight of markers in
             * the group wins, evidence of membership a stray marker cannot fake, then the kinetic distance */
            double tp[3], tv[3];
            dist = orphan_tracer(m, K->ctx, tp, tv) ? kinetic_distance(tp, tv, K->xg, K->vg, K->up) : 1e299;
            if(bi < 0 || subhalo_orphan_ranks_before(m->MarkerWeight, dist, m->TrackId, nd[bi].old->MarkerWeight, best, nd[bi].TrackId)) {
                best = dist;
                bi = v;
            }
        }
    }
    return bi >= 0 ? bi : 0;
}

/* The processing tree: nesting from NestedParentTrackId, detached members attach to the central,
 * a cycle of stale parents is broken; child lists in TrackId order and depths. */
static void
key_tree(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    const int nn = K->nn, ci = K->ci;
    int v;
    for(v = 0; v < nn; v++) {
        if(v == ci)
            continue;
        const int p = nd[v].old ? node_index(nd, nn, nd[v].old->NestedParentTrackId) : -1;
        nd[v].parent = (p >= 0 && p != v && nd[p].kind == SUBHALO_MEM_RESOLVED) ? p : ci;
    }
    for(v = 0; v < nn; v++) {
        int u = v, steps = 0;
        while(u != ci && steps <= nn) {
            u = nd[u].parent;
            steps++;
        }
        if(u != ci)
            nd[v].parent = ci;      /* a cycle of stale parents: break it at v */
    }
    for(v = nn - 1; v >= 0; v--)
        if(v != ci) {
            nd[v].sib = nd[nd[v].parent].child;
            nd[nd[v].parent].child = v;
        }
    for(v = 0; v < nn; v++) {
        int u = v, dd = 0;
        while(u != ci) {
            u = nd[u].parent;
            dd++;
        }
        nd[v].depth = dd;
    }
}

/* The sources: the owner map (ConstrainToSingleHost, FeedCentrals) and the own source lists, both
 * in the arena in this order. */
static void
key_owner_map(struct KeyRun * K)
{
    const struct SubNode * nd = K->nd;
    const int nn = K->nn, ci = K->ci, pseudo = K->pseudo, subkey = K->subkey;
    const int64_t g = K->g;
    int64_t i;
    int32_t * owner = (int32_t *) arena_need(K->arena, sizeof(int32_t) * (K->n + 1), g, "owner map");
    K->ownlist = (int32_t *) arena_need(K->arena, sizeof(int32_t) * (K->n + 1), g, "source lists");
    for(i = 0; i < K->n; i++) {
        const struct particle_data * p = &P[K->start + i];
        int o = -1;
        if(subhalo_part_live(p)) {
            const int t = p->TrackId >= 0 ? node_index(nd, nn, p->TrackId) : -1;
            /* the central's source is its FOF group only, as HBT's FeedCentrals: its own
             * FOF-less particles would otherwise keep accumulating, call after call, in its bound set.
             * Pseudo-keys and sub-tree keys (whose root is never the central) keep theirs. */
            if(t >= 0 && nd[t].kind == SUBHALO_MEM_RESOLVED && (pseudo || subkey || t != ci || p->GrNr == g))
                o = t;                      /* a resolved member's own particles (a satellite's also in the field) */
            else if(pseudo || p->GrNr == g)
                o = ci;                     /* every other FOF particle feeds the central */
        }
        owner[i] = o;
    }
    K->owner = owner;
}

/* The own sources grouped by owner, key order within each. */
static void
key_own_lists(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    const int32_t * owner = K->owner;
    int64_t i, o = 0;
    int v;
    for(v = 0; v < K->nn; v++)
        nd[v].own_n = 0;
    for(i = 0; i < K->n; i++)
        if(owner[i] >= 0)
            nd[owner[i]].own_n++;
    for(v = 0; v < K->nn; v++) {
        nd[v].own_off = o;
        o += nd[v].own_n;
        nd[v].own_n = 0;
    }
    for(i = 0; i < K->n; i++)
        if(owner[i] >= 0) {
            struct SubNode * u = &nd[owner[i]];
            K->ownlist[u->own_off + u->own_n++] = (int32_t) i;
        }
}

/* The owner map, ReassignParticles (step c, subhalo_reassign_gas.cpp) for the non-tracers, the own
 * lists, and the admission by neighbour majority; adm then marks the admitted and taken particles
 * (the owner map is not read again). */
static inline __attribute__((always_inline)) void
key_sources(struct KeyRun * K)
{
    const struct SubhaloParams * sp = K->sp;
    key_owner_map(K);
    if(sp->SubhaloReassignGas)
        reassign_particles(K->owner, K->start, K->n, K->TracerTypes, sp, K->ctx, K->arena, K->g);
    key_own_lists(K);
    struct AdmitKey ak = {K->owner, K->nd, K->nn, K->ci, K->g, K->start, K->n, NULL, 0};
    K->adm = NULL;
    if(sp->SubhaloAdmitShare > 0) {
        /* a sub-tree key takes its group's vote, taken across ranks by the plan */
        if(K->subkey)
            K->ks.NAdmitted = admit_records(&ak, K->ctx->plan, K->arena);
        else
            K->ks.NAdmitted = admit_particles(&ak, sp, K->ctx, K->arena);
        K->ks.NAdmitWithheld += ak.nwithheld;
        if(K->ks.NAdmitted > 0)
            K->adm = K->owner;
    }
    K->admlist = ak.admlist;
}

/* Step d, x's list: its own particles not taken by a descendant, the remainders of its children
 * (a dead child's whole list) and its admitted particles. Returns the capacity, which also leaves
 * room for the bound sets of the survivors below x that may sink into it. */
static int64_t
node_gather(struct KeyRun * K, struct SubNode * x)
{
    const struct SubNode * nd = K->nd;
    const int32_t * adm = K->adm, * ownlist = K->ownlist;
    int64_t input = x->own_n + x->adm_n, reserve = 0, s, j;
    int c;
    for(c = x->child; c >= 0; c = nd[c].sib) {
        if(nd[c].state == SUBHALO_ST_ALIVE) input += nd[c].c.N - nd[c].c.Nbound;
        else if(nd[c].state == SUBHALO_ST_DIED || nd[c].state == SUBHALO_ST_FAILED) input += nd[c].c.N;
    }
    if(K->sp->SubhaloMergeTrapped)
        for(s = x->firstseq; s < K->nseq; s++)
            if(nd[nd[s].byseq].state == SUBHALO_ST_ALIVE)
                reserve += nd[nd[s].byseq].c.Nbound;
    const int64_t cap = input + reserve;
    x->c.Index = (int64_t *) arena_need(K->arena, sizeof(int64_t) * (cap + 1), K->g, "candidate list");
    x->c.N = 0;
    /* own particles taken by a descendant belong to it */
    for(j = 0; j < x->own_n; j++)
        if(!adm || adm[ownlist[x->own_off + j]] != ADM_TAKEN)
            x->c.Index[x->c.N++] = K->start + ownlist[x->own_off + j];
    for(c = x->child; c >= 0; c = nd[c].sib) {
        int64_t a0 = -1;
        if(nd[c].state == SUBHALO_ST_ALIVE) a0 = nd[c].c.Nbound;
        else if(nd[c].state == SUBHALO_ST_DIED || nd[c].state == SUBHALO_ST_FAILED) a0 = 0;
        if(a0 >= 0)
            for(j = a0; j < nd[c].c.N; j++)
                x->c.Index[x->c.N++] = nd[c].c.Index[j];
    }
    for(j = 0; j < x->adm_n; j++)
        x->c.Index[x->c.N++] = K->start + K->admlist[x->adm_off + j];
    return cap;
}

/* Step d, x's frame: the previous-bound frame of a resolved track, the tracer of a resurrected
 * orphan, the key centre for a newborn (or when undefined). */
static inline __attribute__((always_inline)) void
node_frame(const struct KeyRun * K, struct SubNode * x)
{
    int d;
    for(d = 0; d < 3; d++) {
        x->c.RefPos[d] = K->xg[d];
        x->c.RefVel[d] = K->vg[d];
    }
    if(x->kind == SUBHALO_MEM_RESOLVED && x->old->NPrev > 0)
        for(d = 0; d < 3; d++) {
            x->c.RefPos[d] = x->old->RefPos[d];
            x->c.RefVel[d] = x->old->RefVel[d];
        }
    else if(x->kind == SUBHALO_MEM_ORPHAN)
        orphan_tracer(x->old, K->ctx, x->c.RefPos, x->c.RefVel);
    x->c.TrackId = x->TrackId;
    x->c.Seed = node_seed(x->TrackId, K->ctx->SnapNum);
    x->c.Flags = 0;
}

/* Step d, x unbound; then the sinking of its subtree into it and x unbound again
 * (subhalo_unbind.cpp:828-835). The energies live above the list only while they are used. */
static inline __attribute__((always_inline)) void
node_unbind(struct KeyRun * K, struct SubNode * x, int u, int64_t cap)
{
    const struct SubhaloParams * sp = K->sp;
    struct SubhaloArena * arena = K->arena;
    const size_t mark = arena->used;
    x->c.Energy = (double *) arena_need(arena, sizeof(double) * (cap + 1), K->g, "candidate energies");
    int alive = unbind_counted(&x->c, x->kind == SUBHALO_MEM_RESOLVED ? x->old : NULL, K->up, arena, &K->ks);
    arena->used = mark;
    node_unbound(x, alive, K->ownlist, K->start, K->TracerTypes);
    node_core(x, K->up, sp->SubhaloCoreSize);
    if(x->state == SUBHALO_ST_ALIVE && sp->SubhaloMergeTrapped
       && merge_subtree(K->nd, u, u, K->up, sp->SubhaloSinkSymmetric, K->TracerTypes)) {
        x->c.Energy = (double *) arena_need(arena, sizeof(double) * (cap + 1), K->g, "candidate energies");
        alive = unbind_counted(&x->c, NULL, K->up, arena, &K->ks);
        arena->used = mark;
        node_unbound(x, alive, K->ownlist, K->start, K->TracerTypes);
        node_core(x, K->up, sp->SubhaloCoreSize);
    }
}

/* Step d, x's admitted particles: the bound ones are taken, the others leave its list before its
 * tail is cut and its parent gathers the remainder (their owner still lists them). */
static void
node_admitted(struct KeyRun * K, struct SubNode * x, int u)
{
    int32_t * adm = K->adm;
    const int64_t start = K->start;
    int64_t w = x->c.Nbound, ntaken = 0, j;
    for(j = 0; j < x->c.Nbound; j++)
        if(adm[x->c.Index[j] - start] == u) {
            adm[x->c.Index[j] - start] = ADM_TAKEN;
            ntaken++;
        }
    for(j = x->c.Nbound; j < x->c.N; j++) {
        const int64_t li = x->c.Index[j] - start;
        if(adm[li] == u)
            adm[li] = -1;
        else
            x->c.Index[w++] = x->c.Index[j];
    }
    x->c.N = w;
    K->ks.NAdmitTaken += ntaken;
    K->ks.NAdmitReceivers += ntaken > 0;
}

/* Step d, after x: its sequence number, then the compaction of its subtree's lists: a survivor
 * keeps its relaxed source (bound set and tail, TruncateSource), everything else nothing; x keeps
 * its whole list until its parent has gathered the remainder. */
static inline __attribute__((always_inline)) void
node_compact(struct KeyRun * K, struct SubNode * x, int u)
{
    struct SubNode * nd = K->nd;
    struct SubhaloArena * arena = K->arena;
    int64_t s;
    x->seq = K->nseq;
    nd[K->nseq].byseq = u;
    K->nseq++;
    char * dst = arena->base + x->region;
    for(s = x->firstseq; s < x->seq; s++) {
        struct SubNode * w = &nd[nd[s].byseq];
        int64_t keep = 0;
        if(w->state == SUBHALO_ST_ALIVE)
            keep = w->keep;
        if(keep > 0)
            memmove(dst, w->c.Index, sizeof(int64_t) * keep);
        w->c.Index = (int64_t *) dst;
        w->c.N = keep;
        w->keep = keep;
        dst += align16(sizeof(int64_t) * (keep + 1));
    }
    memmove(dst, x->c.Index, sizeof(int64_t) * x->c.N);
    x->c.Index = (int64_t *) dst;
    dst += align16(sizeof(int64_t) * (x->c.N + 1));
    arena->used = dst - arena->base;
    /* the relaxed source size applied at the parent's compaction */
    if(x->state == SUBHALO_ST_ALIVE) {
        int64_t nkeep = (int64_t) (x->c.Nbound * K->sp->SubhaloSourceRelaxFactor);
        if(nkeep < x->c.Nbound) nkeep = x->c.Nbound;
        if(nkeep > x->c.N) nkeep = x->c.N;
        x->keep = nkeep;
    }
    else
        x->keep = 0;
}

/* Step d for node u once its children are done; an orphan below the root takes no part. */
static inline __attribute__((always_inline)) void
key_node(struct KeyRun * K, int u)
{
    struct SubNode * x = &K->nd[u];
    if(x->kind == SUBHALO_MEM_ORPHAN && u != K->ci) {
        x->state = SUBHALO_ST_NONE;
        x->seq = -1;
        return;
    }
    const int64_t cap = node_gather(K, x);
    node_frame(K, x);
    node_unbind(K, x, u, cap);
    if(x->adm_n > 0)
        node_admitted(K, x, u);
    x->c.Energy = NULL;
    node_compact(K, x, u);
}

/* Step d: post-order unbinding with the remainder passed up, sinking, re-unbinding (RecursiveUnbind,
 * subhalo_unbind.cpp:810-842); a subtree's lists start at its region of the arena. Then the root's
 * own list to its relaxed size too (nothing above it gathers its remainder); a sub-tree key's root
 * keeps its whole list (its tags are provisional, design 7.8). */
static inline __attribute__((always_inline)) void
key_unbind(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    struct SubhaloArena * arena = K->arena;
    const int ci = K->ci;
    int u = ci, v;
    nd[u].region = arena->used;
    nd[u].firstseq = K->nseq;
    for(;;) {
        if(nd[u].child >= 0 && !nd[u].visited) {
            nd[u].visited = 1;
            u = nd[u].child;
            nd[u].region = arena->used;
            nd[u].firstseq = K->nseq;
            continue;
        }
        nd[u].visited = 1;
        key_node(K, u);
        if(u == ci)
            break;
        if(nd[u].sib >= 0) {
            u = nd[u].sib;
            nd[u].region = arena->used;
            nd[u].firstseq = K->nseq;
            continue;
        }
        u = nd[u].parent;
    }
    if(nd[ci].state == SUBHALO_ST_ALIVE && !K->subkey)
        nd[ci].c.N = nd[ci].keep;
    for(v = 0; v < K->nn; v++)
        if(nd[v].state == SUBHALO_ST_ALIVE && v != ci)
            nd[v].c.N = nd[v].keep;
}

/* Step g: the final pass, the HBT and Rockstar bound properties and the baryonic sums of every
 * survivor. */
static inline __attribute__((always_inline)) void
key_final_pass(struct KeyRun * K)
{
    const struct SubhaloCallCtx * ctx = K->ctx;
    const struct SubhaloUnbindParams * up = K->up;
    struct SubhaloArena * arena = K->arena;
    const int CoreSize = K->sp->SubhaloCoreSize;
    const double vu = subhalo_vel_to_kms(1.0);   /* the unit factor: subhalo_props_hbt divides by a only */
    int v, d, nalive = 0;
    for(v = 0; v < K->nn; v++) {
        struct SubNode * x = &K->nd[v];
        if(x->state != SUBHALO_ST_ALIVE)
            continue;
        nalive++;
        struct SubTrack * rec = &K->res[v].rec;
        subhalo_track_carry(rec, x->old, x->TrackId, ctx->SnapNum);
        const size_t mark = arena->used;
        x->c.Energy = (double *) arena_need(arena, sizeof(double) * (x->c.Nbound + 1), K->g, "final-pass energies");
        const int64_t npos = subhalo_final_pass(&x->c, up, arena);
        K->ks.NFinalPositive += npos;
        K->ks.NFinalBound += x->c.Nbound;
        subhalo_props_hbt(rec, &x->c, up, CoreSize, ctx->Offset, arena);
        /* Rockstar's recipes need the final-pass energies; they convert their velocities
         * themselves (km/s), so the unit factor below is not applied to them */
        subhalo_props_rockstar(rec, &x->c, up, ctx->rho, CoreSize, ctx->Offset, arena);
        subhalo_props_baryons(rec, &x->c, ctx->MetalReturnOn);
        x->c.Energy = NULL;
        arena->used = mark;
        for(d = 0; d < 3; d++) {
            rec->CoreVel[d] *= vu;
            rec->MostBoundVel[d] *= vu;
            rec->CoMVel[d] *= vu;
        }
        rec->CoreSigmaV *= vu;
        rec->Vmax *= vu;
        rec->Ekin *= vu * vu;
        rec->Epot *= vu * vu;
    }
    K->nalive = nalive;
}

/* Step f of a sub-tree key: the processing tree as it stands (no promotion: the root's parent is an
 * in-place candidate); Rank, Depth, NestedParentTrackId and CentralTrackId are written by the rounds
 * from the group's hierarchy. A dead root still claims its whole list, below every survivor. */
static void
subkey_hierarchy(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    const int nn = K->nn;
    int v;
    if(K->nalive > 0) {
        for(v = 0; v < nn; v++) {
            if(nd[v].state != SUBHALO_ST_ALIVE)
                continue;
            int u = nd[v].parent;
            while(u >= 0 && nd[u].state != SUBHALO_ST_ALIVE)
                u = nd[u].parent;
            nd[v].fparent = u;
        }
        for(v = 0; v < nn; v++) {
            if(nd[v].state != SUBHALO_ST_ALIVE)
                continue;
            int u = v, dd = 0;
            while(nd[u].fparent >= 0 && dd <= nn) {
                u = nd[u].fparent;
                dd++;
            }
            nd[v].fdepth = dd;
            nd[v].rank = -1;
            struct SubTrack * rec = &K->res[v].rec;
            rec->HostGroup = K->gfof;
            rec->Host = K->gfof;
            rec->Rank = 0;
            rec->Depth = dd;
            rec->NestedParentTrackId = nd[v].fparent >= 0 ? nd[nd[v].fparent].TrackId : -1;
            rec->CentralTrackId = -1;
            rec->SinkTrackId = -1;
            rec->DescendantTrackId = nd[v].TrackId;
            rec->SnapDeath = rec->SnapSink = -1;
            rec->Alive = 1;
        }
    }
    if(nd[K->ci].state != SUBHALO_ST_ALIVE)
        nd[K->ci].fdepth = -1;
}

/* Step f, ranks by CompareMass (insertion: a key has few members); returns the most massive survivor. */
static inline __attribute__((always_inline)) int
key_ranks(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    const struct SubhaloResult * res = K->res;
    int v, r, X = -1;
    for(v = 0; v < K->nn; v++)
        nd[v].rank = -1;
    for(r = 0; r < K->nalive; r++) {
        int b = -1;
        for(v = 0; v < K->nn; v++)
            if(nd[v].state == SUBHALO_ST_ALIVE && nd[v].rank < 0 && (b < 0 || mass_before(&res[v].rec, &res[b].rec)))
                b = v;
        nd[b].rank = r;
    }
    for(v = 0; v < K->nn; v++)
        if(nd[v].rank == 0)
            X = v;
    return X;
}

/* Step f, the final tree: a survivor's parent is its nearest surviving ancestor in the processing
 * tree; the most massive survivor X is the central: it leaves its parent and every other top
 * survivor (the decided central among them) is nested under it. */
static void
key_final_tree(struct KeyRun * K)
{
    struct SubNode * nd = K->nd;
    const int nn = K->nn, X = K->X;
    int v;
    for(v = 0; v < nn; v++) {
        if(nd[v].state != SUBHALO_ST_ALIVE)
            continue;
        int u = nd[v].parent;
        while(u >= 0 && nd[u].state != SUBHALO_ST_ALIVE)
            u = nd[u].parent;
        nd[v].fparent = u;
    }
    nd[X].fparent = -1;
    for(v = 0; v < nn; v++) {
        if(nd[v].state != SUBHALO_ST_ALIVE || v == X)
            continue;
        int u = nd[v].parent;
        while(u >= 0 && nd[u].state != SUBHALO_ST_ALIVE)
            u = nd[u].parent;
        if(u < 0)
            nd[v].fparent = X;
    }
    for(v = 0; v < nn; v++) {
        if(nd[v].state != SUBHALO_ST_ALIVE)
            continue;
        int u = v, dd = 0;
        while(nd[u].fparent >= 0 && dd <= nn) {
            u = nd[u].fparent;
            dd++;
        }
        nd[v].fdepth = dd;
    }
}

/* Step f, the hierarchy columns of every survivor's record. */
static void
key_headers(struct KeyRun * K)
{
    const struct SubNode * nd = K->nd;
    const int X = K->X;
    int v, d;
    for(v = 0; v < K->nn; v++) {
        if(nd[v].state != SUBHALO_ST_ALIVE)
            continue;
        struct SubTrack * rec = &K->res[v].rec;
        rec->HostGroup = K->pseudo ? -1 : K->g;
        rec->Host = rec->HostGroup;
        rec->Rank = nd[v].rank;
        rec->Depth = nd[v].fdepth;
        rec->NestedParentTrackId = nd[v].fparent >= 0 ? nd[nd[v].fparent].TrackId : -1;
        rec->CentralTrackId = nd[X].TrackId;
        rec->SinkTrackId = -1;
        rec->DescendantTrackId = nd[v].TrackId;
        rec->SnapDeath = rec->SnapSink = -1;
        rec->Alive = 1;
        if(x_is_resurrected(&nd[v])) {
            /* the orphan tracer state is not a Subhalo column: cleared, as a restart reads it */
            rec->SnapTracerLost = -1;
            subhalo_track_clear_orphan_set(rec);
            for(d = 0; d < 3; d++) {
                rec->TracerPos[d] = -1;
                rec->TracerVel[d] = -1;
            }
        }
    }
}

/* Step f: ranks by mass and the final hierarchy (UpdateTracks: SortMemberLists, ExtendCentralNest,
 * AssignRanks, FillDepth, subhalo_tracking.cpp:1489-1492). */
static inline __attribute__((always_inline)) void
key_hierarchy(struct KeyRun * K)
{
    if(K->subkey) {
        subkey_hierarchy(K);
        return;
    }
    if(K->nalive == 0)
        return;
    K->X = key_ranks(K);
    key_final_tree(K);
    key_headers(K);
}

/* Step 8, in the owner map's memory: every survivor's bound particles mark their binder in the
 * binder map, its first SubhaloMinNumTracerPart tracers are reserved in the tag map. */
static void
tags_reserve(struct KeyRun * K)
{
    int32_t * tagnode = K->owner, * bnode = K->ownlist;
    int64_t i, j;
    int v;
    for(i = 0; i < K->n; i++) {
        tagnode[i] = -1;
        bnode[i] = -1;
    }
    for(v = 0; v < K->nn; v++) {
        const struct SubNode * x = &K->nd[v];
        if(x->state != SUBHALO_ST_ALIVE)
            continue;
        int nr = 0;
        for(j = 0; j < x->c.Nbound; j++) {
            const int64_t li = x->c.Index[j] - K->start;
            bnode[li] = v;
            if(nr < K->sp->SubhaloMinNumTracerPart && is_tracer(x->c.Index[j], K->TracerTypes)) {
                tagnode[li] = v | TAG_RESERVED;
                nr++;
            }
        }
    }
}

/* Step 8, the claims: the deepest source list takes an unreserved particle; a tail claim needs the
 * binder to be the claimant or its ancestor in the FINAL tree (changed by the promotion). */
static void
tags_claim(struct KeyRun * K)
{
    const struct SubNode * nd = K->nd;
    int32_t * tagnode = K->owner;
    const int32_t * bnode = K->ownlist;
    int64_t j, nrejected = 0;
    int v;
    for(v = 0; v < K->nn; v++) {
        const struct SubNode * x = &nd[v];
        if(x->state != SUBHALO_ST_ALIVE && !(K->subkey && v == K->ci))
            continue;
        for(j = 0; j < x->c.N; j++) {
            const int64_t li = x->c.Index[j] - K->start;
            const int t = tagnode[li];
            if(t >= 0 && (t & TAG_RESERVED))
                continue;
            if(!(t < 0 || nd[v].fdepth > nd[t].fdepth || (nd[v].fdepth == nd[t].fdepth && nd[v].TrackId < nd[t].TrackId)))
                continue;
            const int b = bnode[li];
            if(b >= 0 && b != v && !final_ancestor(nd, v, b, K->nn)) {
                nrejected++;
                continue;
            }
            tagnode[li] = v;
        }
    }
    K->ks.NTailRejected = nrejected;
}

/* Step 8, the tags of node v's claims: BindRank in own-tag energy order (Index[0, Nbound) in
 * final-pass energy order, Index[Nbound, N) the tail in that of its unbinding), BoundTrackId the
 * binder; the exclusive and tail counts into rec. Returns the tail count inside relaxed. */
static int64_t
node_tags(const struct KeyRun * K, int v, struct SubTrack * rec, int64_t relaxed)
{
    const struct SubNode * nd = K->nd, * x = &nd[v];
    const int32_t * tagnode = K->owner, * bnode = K->ownlist;
    int64_t r = 0, nex = 0, ntl = 0, j;
    for(j = 0; j < x->c.N; j++) {
        const int64_t pi = x->c.Index[j];
        const int64_t li = pi - K->start;
        if(tagnode[li] < 0 || (tagnode[li] & ~TAG_RESERVED) != v)
            continue;
        if(r > INT32_MAX)
            endrun(1, "Track %ld: BindRank beyond the int32 range.\n", x->TrackId);
        P[pi].TrackId = x->TrackId;
        P[pi].BindRank = (int32_t) (r++);
        P[pi].BoundTrackId = bnode[li] >= 0 ? nd[bnode[li]].TrackId : -1;
        if(bnode[li] == v) {
            rec->NboundExclusiveType[P[pi].Type]++;
            nex++;
        }
        else if(j < relaxed) {
            rec->NtailType[P[pi].Type]++;
            ntl++;
        }
    }
    rec->NboundExclusive = (int32_t) nex;
    return ntl;
}

/* Step 8, the tags written for every survivor and a sub-tree key's root, whose tail counts only
 * the part inside its relaxed source (a dead root writes its counts nowhere). */
static void
tags_write(struct KeyRun * K)
{
    int v;
    for(v = 0; v < K->nn; v++) {
        const struct SubNode * x = &K->nd[v];
        const int alive = x->state == SUBHALO_ST_ALIVE;
        const int root = K->subkey && v == K->ci;
        if(!alive && !root)
            continue;
        struct SubTrack dummy;
        struct SubTrack * rec = alive ? &K->res[v].rec : &dummy;
        if(!alive)
            memset(&dummy, 0, sizeof(dummy));
        const int64_t ntl = node_tags(K, v, rec, root ? (alive ? x->keep : 0) : x->c.N);
        if(root)
            K->rootkeep = (int32_t) ntl;
    }
}

/* Step 8, everything else in the key is untagged (swallowed particles keep their tags); a guest
 * its receiver did not take leaves with the tags it came with, its owner's. */
static void
tags_clear_rest(const struct KeyRun * K)
{
    const int32_t * tagnode = K->owner;
    int64_t i;
    for(i = 0; i < K->n; i++) {
        struct particle_data * p = &P[K->start + i];
        if(tagnode[i] >= 0 || !subhalo_part_live(p))
            continue;
        if(p->GrNr < -1 && ((-(p->GrNr + 2)) & SUBHALO_KEY_GUESTBIT))
            continue;
        p->TrackId = -1;
        p->BindRank = -1;
        p->BoundTrackId = -1;
    }
}

/* Step 8 of a sub-tree key: a particle bound by one survivor but tagged by another records its
 * binder ordinals, so that a claim voided in the rounds merges into the binder's sequence by energy. */
static void
subkey_bind_ords(struct KeyRun * K)
{
    struct SubhaloBindOrd * bo = K->ctx->BindOrd + K->sc->BindStart[K->k];
    const int32_t * tagnode = K->owner;
    int64_t nbo = 0, j;
    int v;
    for(v = 0; v < K->nn; v++) {
        const struct SubNode * x = &K->nd[v];
        if(x->state != SUBHALO_ST_ALIVE)
            continue;
        int64_t own = 0;
        for(j = 0; j < x->c.Nbound; j++) {
            const int64_t pi = x->c.Index[j];
            const int t = tagnode[pi - K->start];
            if(t < 0)
                continue;
            if((t & ~TAG_RESERVED) == v) {
                own++;
                continue;
            }
            if(nbo >= K->n)
                endrun(5, "Sub-tree key %ld: more binder ordinals than particles.\n", K->g);
            bo[nbo].ID = P[pi].ID;
            bo[nbo].g = own;
            bo[nbo].f = j;
            nbo++;
        }
    }
    K->sc->BindCount[K->k] = nbo;
}

/* Step 8: the tags (design D10): the most-bound tracers reserved, the deepest source list for the
 * rest, BindRank in own-tag energy order, BoundTrackId the binder. */
static void
key_tags(struct KeyRun * K)
{
    tags_reserve(K);
    tags_claim(K);
    tags_write(K);
    tags_clear_rest(K);
    if(K->subkey && K->sc->BindStart)
        subkey_bind_ords(K);
}

/* Step h: the arena is cleared; with a track resolved at the previous call that died or sank here,
 * the key's particles indexed by ID for the descendant votes, else NULL. */
static struct IdLocal *
key_desc_index(struct KeyRun * K)
{
    const struct SubNode * nd = K->nd;
    int64_t i;
    int v, ndead = 0;
    K->arena->used = 0;
    for(v = 0; v < K->nn; v++)
        if(!K->subkey && nd[v].kind == SUBHALO_MEM_RESOLVED && (nd[v].state == SUBHALO_ST_DIED || nd[v].state == SUBHALO_ST_SUNK) && nd[v].old->NTracers > 0)
            ndead++;
    if(ndead == 0)
        return NULL;
    struct IdLocal * ids = (struct IdLocal *) arena_need(K->arena, sizeof(struct IdLocal) * (K->n + 1), K->g, "descendant index");
    for(i = 0; i < K->n; i++) {
        ids[i].ID = P[K->start + i].ID;
        ids[i].i = K->start + i;
    }
    qsort(ids, K->n, sizeof(struct IdLocal), cmp_idlocal);
    return ids;
}

/* Step h, the descendant votes of a track that died or sank: its step-0 tracers in this key, the
 * others pending (every one in a sub-tree key: the provisional tags are final only after the rounds). */
static inline __attribute__((always_inline)) void
desc_votes(const struct KeyRun * K, struct SubhaloResult * rs, const struct SubNode * x, const struct IdLocal * ids)
{
    const int64_t n = K->n;
    int t;
    for(t = 0; t < x->old->NTracers && t < SUBHALO_MAXTRACERS; t++) {
        if(K->subkey) {
            rs->DescOwner[t] = SUBHALO_DESC_PENDING;
            continue;
        }
        struct IdLocal key = {x->old->TracerIDs[t], -1};
        int64_t lo = 0, hi = n;
        while(lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if(ids[mid].ID < key.ID) lo = mid + 1;
            else hi = mid;
        }
        if(lo < n && ids[lo].ID == key.ID) {
            const struct particle_data * tp = &P[ids[lo].i];
            rs->DescOwner[t] = subhalo_part_live(tp) ? tp->BoundTrackId : -1;
            rs->DescWeight[t] = subhalo_part_live(tp) ? subhalo_tracer_weight(tp->TracerPenalty) : 1;
        }
        else
            rs->DescOwner[t] = SUBHALO_DESC_PENDING;
    }
    rs->ndesc = t;
}

/* Step h, the record of a member that did not survive, from its previous one; a track resolved at
 * the previous call that died or sank becomes an orphan with its descendant votes. An orphan that
 * stays one (or failed resurrection) keeps its record and step-1 state. */
static inline __attribute__((always_inline)) void
node_lost(const struct KeyRun * K, int v, const struct IdLocal * ids, double * maxmerge)
{
    const struct SubNode * nd = K->nd, * x = &nd[v];
    const int X = K->X;
    struct SubhaloResult * rs = &K->res[v];
    struct SubTrack * rec = &rs->rec;
    *rec = *x->old;
    rs->valid = 1;
    rec->HostGroup = K->pseudo ? -1 : K->gfof;
    /* the central of the current host, -1 when nothing in the key survived; a sub-tree
     * key's members get theirs from the rounds */
    rec->CentralTrackId = X >= 0 ? nd[X].TrackId : SUBHALO_NOTRACK;
    /* a track dying or sinking in a group with another central is a satellite at its death
     * (SnapLastIsolation, as HBT ranks the new orphan); older orphans keep their history */
    if(!K->pseudo && X >= 0 && X != v && x->kind == SUBHALO_MEM_RESOLVED && (x->state == SUBHALO_ST_DIED || x->state == SUBHALO_ST_SUNK)
       && rec->SnapLastIsolation < 0)
        rec->SnapLastIsolation = K->ctx->SnapNum - 1;
    if(!(x->kind == SUBHALO_MEM_RESOLVED && (x->state == SUBHALO_ST_DIED || x->state == SUBHALO_ST_SUNK)))
        return;
    record_mb(rec, x, K->ctx);
    const int64_t sink = x->state == SUBHALO_ST_SUNK ? nd[x->sinkinto].TrackId : SUBHALO_NOTRACK;
    subhalo_track_make_orphan(rec, sink, K->ctx->SnapNum);
    /* the nearest surviving ancestor */
    int u = x->parent;
    while(u >= 0 && nd[u].state != SUBHALO_ST_ALIVE)
        u = nd[u].parent;
    rec->NestedParentTrackId = u >= 0 ? nd[u].TrackId : (X >= 0 && X != v ? nd[X].TrackId : -1);
    if(x->state == SUBHALO_ST_SUNK && x->sinkinto == X && x->old->Mbound > *maxmerge)
        *maxmerge = x->old->Mbound;
    desc_votes(K, rs, x, ids);
}

/* Step h: the result slot of every node, with its state for the rounds; the central's largest
 * merger is the most massive track that arrived as its satellite or sank into it this call. */
static inline __attribute__((always_inline)) void
key_results(struct KeyRun * K, const struct IdLocal * ids)
{
    const struct SubNode * nd = K->nd;
    const int X = K->X;
    double maxmerge = 0;
    int v;
    for(v = 0; v < K->nn; v++) {
        const struct SubNode * x = &nd[v];
        struct SubhaloResult * rs = &K->res[v];
        rs->ndesc = 0;
        rs->history = 0;
        rs->newborn = 0;
        rs->MaxMergerMbound = 0;
        rs->subkey = K->subkey ? K->g : -1;
        rs->state = x->state;
        rs->Keep = (K->subkey && v == K->ci) ? K->rootkeep : 0;
        rs->has_core = x->has_core;
        if(x->has_core)
            rs->core = x->core;
        else
            memset(&rs->core, 0, sizeof(rs->core));
        rs->SinkTo = x->state == SUBHALO_ST_SUNK ? nd[x->sinkinto].TrackId : -1;
        if(x->state == SUBHALO_ST_ALIVE) {
            rs->valid = 1;
            rs->history = 1;
            rs->newborn = x->kind == SUBHALO_MEM_NEWBORN;
            /* a track that arrived as a satellite of the central this call */
            if(X >= 0 && v != X && x->old && (x->old->Rank <= 0 || x->old->CentralTrackId != nd[X].TrackId)
               && x->old->Mbound > maxmerge)
                maxmerge = x->old->Mbound;
        }
        else if(x->kind == SUBHALO_MEM_NEWBORN)
            rs->valid = 0;
        else
            node_lost(K, v, ids, &maxmerge);
    }
    if(X >= 0)
        K->res[X].MaxMergerMbound = maxmerge;
}

/* One serial key, design 7.2 steps 7a-7h and 8 in the order the data needs: d, g, f, 8, h. */
static void
process_serial_key(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata)
{
    struct KeyRun K;
    (void) nthreads;
    key_open(&K, seg, k, arena, (struct SerialCtx *) userdata);
    key_nodes(&K);
    key_centre(&K);
    K.ci = key_central(&K);
    key_tree(&K);
    key_sources(&K);
    key_unbind(&K);
    key_final_pass(&K);
    key_hierarchy(&K);
    key_tags(&K);
    const struct IdLocal * ids = key_desc_index(&K);
    key_results(&K, ids);
    arena->used = 0;
    stats_merge(&K.ctx->Stats, &K.ks);
}

static int
cmp_bindord_id(const void * a, const void * b)
{
    const MyIDType x = ((const struct SubhaloBindOrd *) a)->ID, y = ((const struct SubhaloBindOrd *) b)->ID;
    return (x > y) - (x < y);
}

void
subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx)
{
    struct SerialCtx sc;
    sc.ctx = ctx;
    sc.Nodes = (struct SubNode *) mymalloc("SubNodes", sizeof(struct SubNode) * (ctx->NRes + 1));
    /* per key: the alignment allowance of its candidate nodes' lists */
    int64_t * extra = (int64_t *) mymalloc("SubArenaExtra", sizeof(int64_t) * (nsegs + 1));
    int64_t k;
    for(k = 0; k < nsegs; k++)
        extra[k] = (ctx->HostedStart[k + 1] - ctx->HostedStart[k] + 1) * (int64_t) SUBHALO_ARENA_NODE_BYTES;
    /* one binder-ordinal slot per particle of the sub-tree keys here, on the top so that it
     * survives the restore (the plan charges SubKeyPartBytes per such particle) */
    int64_t * bstart = (int64_t *) mymalloc("SubBindStart", sizeof(int64_t) * (nsegs + 1));
    int64_t * bcount = (int64_t *) mymalloc("SubBindCount", sizeof(int64_t) * (nsegs + 1));
    int64_t nbo = 0;
    for(k = 0; k < nsegs; k++) {
        bstart[k] = nbo;
        bcount[k] = 0;
        if(ctx->plan && segs[k].g >= ctx->plan->SubKeyBase)
            nbo += segs[k].n;
    }
    ctx->BindOrd = (struct SubhaloBindOrd *) mymalloc2("SubBindOrd", sizeof(struct SubhaloBindOrd) * (nbo + 1));
    ctx->NBindOrd = 0;
    sc.BindStart = bstart;
    sc.BindCount = bcount;
    const double peak = subhalo_run_serial_keys(segs, nsegs, subhalo_arena_bytes_per_part(), extra, process_serial_key, &sc);
    if(peak > ctx->Stats.ArenaPeakFrac)
        ctx->Stats.ArenaPeakFrac = peak;
    /* compact and sort by ID */
    for(k = 0; k < nsegs; k++) {
        if(bcount[k] > 0 && bstart[k] != ctx->NBindOrd)
            memmove(ctx->BindOrd + ctx->NBindOrd, ctx->BindOrd + bstart[k], sizeof(struct SubhaloBindOrd) * bcount[k]);
        ctx->NBindOrd += bcount[k];
    }
    qsort(ctx->BindOrd, ctx->NBindOrd, sizeof(struct SubhaloBindOrd), cmp_bindord_id);
    myfree(bcount);
    myfree(bstart);
    myfree(extra);
    myfree(sc.Nodes);
}

void
subhalo_free_bind_ord(struct SubhaloCallCtx * ctx)
{
    if(ctx->BindOrd)
        myfree(ctx->BindOrd);
    ctx->BindOrd = NULL;
    ctx->NBindOrd = 0;
}
