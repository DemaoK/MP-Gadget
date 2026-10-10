/* Runtime subhalo finder: per-key processing.
 *
 * Serial keys (Phase 2a plan, halofinder_phase2a_plan.md, section 1 steps 7a-7h and 8): the
 * HBT-HERONS recipe per key on one rank and one thread: candidates, central decision, nesting,
 * sources (ConstrainToSingleHost, FeedCentrals), ReassignParticles, post-order unbinding with the
 * remainder passed up, sinking and re-unbinding, the final pass and the HBT properties, ranks
 * and hierarchy, the D10 tag rule written into P, and the local part of the descendants.
 * Oversized keys keep the Phase 1 in-place rule ("every FOF particle to one track, no
 * unbinding", Phase 1 plan step 9) until Phase 2b. Unclaimed tracks are handled on the home
 * ranks in subhalo.c. */

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
/* subhalo_process_inplace holds, on every rank and for every in-place key, two KeySums, seven
 * 8-byte vectors and the plan's SubhaloInplaceKey: the plan charges SUBHALO_INPLACE_KEY_BYTES. */
_Static_assert(2 * sizeof(struct KeySums) + 7 * sizeof(int64_t) + sizeof(struct SubhaloInplaceKey) <= SUBHALO_INPLACE_KEY_BYTES,
               "SUBHALO_INPLACE_KEY_BYTES does not cover the in-place vectors");

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
    /* the i4 count columns: abort rather than wrap (H4) */
    if(s->count > INT32_MAX || s->ntracer > INT32_MAX)
        endrun(1, "Key %ld has %g particles, more than the i4 count columns of the catalogue hold.\n", g, s->count);
    rec->Nbound = rec->NboundExclusive = (int32_t) s->count;
    rec->NboundTracer = (int32_t) s->ntracer;
    rec->Flags = inplace ? SUBHALO_FLAG_INPLACE : 0;
    rec->Mbound = s->mass;
    for(t = 0; t < 6; t++) {
        rec->NboundType[t] = rec->NboundExclusiveType[t] = (int32_t) s->LenType[t];
        rec->NtailType[t] = 0;
        rec->MboundType[t] = s->MassType[t];
    }
    /* physical peculiar km/s (design 7.10): Vel * UnitVelocity_in_cm_per_s / (1e5 a) */
    const double vfac = subhalo_vel_to_kms(ctx->atime);
    for(d = 0; d < 3; d++) {
        const double pos = wrap_box(cen[d] - ctx->Offset[d], ctx->BoxSize);
        rec->CorePos[d] = rec->CoMPos[d] = pos;
        const double v = (s->mass > 0 ? s->mv[d] / s->mass : 0) * vfac;
        rec->CoreVel[d] = rec->CoMVel[d] = v;
    }
    if(mb >= 0) {
        rec->MostBoundID = P[mb].ID;
        for(d = 0; d < 3; d++) {
            rec->MostBoundPos[d] = wrap_box(P[mb].Pos[d] - ctx->Offset[d], ctx->BoxSize);
            rec->MostBoundVel[d] = P[mb].Vel[d] * vfac;
        }
    }
    rec->Alive = 1;
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
        nres += segs[s].inplace ? 1 : ctx->HostedStart[s + 1] - ctx->HostedStart[s] + 1;
    }
    ctx->ResStart[nsegs] = nres;
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

/* ------------------------------------------------------------------ serial keys (steps 7a-7h, 8) */

_Static_assert(SUBHALO_ARENA_FIXED_BYTES_PER_PART >= SUBHALO_UNBIND_BYTES_PER_PART + 16,
               "SUBHALO_ARENA_FIXED_BYTES_PER_PART must cover the kernel, the owner map and one more list entry per particle");
/* The Phase 3 properties run after the final pass inside the kernel's transient allowance. */
_Static_assert(SUBHALO_PROPS_BYTES_PER_PART <= SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART,
               "SUBHALO_PROPS_BYTES_PER_PART must fit in the kernel's transient scratch");

enum { NODE_RESOLVED = 0, NODE_ORPHAN = 1, NODE_NEWBORN = 2 };
/* ST_NONE: not unbound (an orphan that stays one); ST_FAILED: an orphan central that was not
 * resurrected, or a newborn that did not survive */
enum { ST_NONE = 0, ST_ALIVE = 1, ST_DIED = 2, ST_SUNK = 3, ST_FAILED = 4 };
#define TAG_RESERVED 0x40000000

/* One candidate of a key (design 7.1): a member hosted here or the newborn. */
struct SubNode {
    int64_t TrackId;
    const struct SubTrack * old;    /* the member's previous record, NULL for the newborn */
    int kind, state;
    int was_alive;                  /* sunk after surviving its own unbinding */
    int sinkinto;                   /* node it sank into, -1 */
    int parent, child, sib, depth;  /* processing tree (children in TrackId order) */
    int visited;                    /* traversal: children entered */
    int fparent, fdepth, rank;      /* final hierarchy over the survivors */
    int seq, firstseq, byseq;       /* list allocation order: own number, first of the subtree, inverse map */
    int has_core;
    int64_t own_off, own_n;         /* own source in the grouped list */
    int64_t keep;                   /* list entries kept by the last compaction */
    size_t region;                  /* arena offset where the lists of the subtree start */
    struct SubCandidate c;
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;
    MyIDType mbID;                  /* the most-bound tracer when it died or sank */
    double mbPos[3], mbVel[3];
    int has_mb;
};
_Static_assert(sizeof(struct SubNode) <= SUBHALO_NODE_BYTES, "SUBHALO_NODE_BYTES does not cover struct SubNode");

struct SerialCtx {
    struct SubhaloCallCtx * ctx;
    struct SubNode * Nodes;          /* parallel to ctx->Res */
};

static inline int
part_live(const struct particle_data * p)
{
    return !p->IsGarbage && !p->Swallowed;
}

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

/* The core used by the sinking test: of the bound set for a survivor (subhalo_core); for a
 * track that just died, its previous most-bound tracer with zero dispersion (HBT
 * GetCorePhaseSpaceProperties for Nbound == 0, subhalo_merge.cpp:93-104). */
static void
node_core(struct SubNode * v, const struct SubhaloUnbindParams * up, int CoreSize)
{
    int d;
    if(v->state == ST_ALIVE) {
        subhalo_core(&v->c, up, CoreSize, v->CorePos, v->CoreVel, &v->SigmaX, &v->SigmaV);
        v->has_core = 1;
    }
    else if(v->state == ST_DIED && v->has_mb) {
        for(d = 0; d < 3; d++) {
            v->CorePos[d] = v->mbPos[d];
            v->CoreVel[d] = v->mbVel[d];
        }
        v->SigmaX = v->SigmaV = 0;
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

/* subhalo_unbind with the J11 diagnostics: every unbinding counted, the iteration cap and the
 * convergence ratio Nbound / Nlast recorded (the flag of this run only, ORed back afterwards). */
static int
unbind_counted(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena, struct SubhaloCallStats * ks)
{
    const int f0 = c->Flags;
    c->Flags = 0;
    const int alive = subhalo_unbind(c, up, arena);
    ks->NCandidates++;
    if(c->Flags & SUBHALO_FLAG_ITERCAP) {
        ks->NItercap++;
        if(ks->WorstTrackId == -1 || c->ConvergenceRatio < ks->WorstConvRatio) {
            ks->WorstConvRatio = c->ConvergenceRatio;
            ks->WorstTrackId = c->TrackId;
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
        if(ks->NItercap > 0 && (to->WorstTrackId == -1 || ks->WorstConvRatio < to->WorstConvRatio
                                || (ks->WorstConvRatio == to->WorstConvRatio && ks->WorstTrackId < to->WorstTrackId))) {
            to->WorstConvRatio = ks->WorstConvRatio;
            to->WorstTrackId = ks->WorstTrackId;
        }
    }
}

/* HBT MergeRecursively (subhalo_merge.cpp:41-91) for the subtree below x against the
 * reference ref, children before their parent. Only tracks resolved at the call's start take
 * part (survivors and tracks that died this call): an orphan from an earlier call is not sunk
 * here (it is not trapped in HBT, but sinking it would give SnapSink != SnapDeath). Returns 1
 * if a survivor sank (its bound set is appended to ref's list, whose capacity reserves it). */
static int
merge_subtree(struct SubNode * nd, int x, int ref, const struct SubhaloUnbindParams * up, int symmetric, int TracerTypes)
{
    int merged = 0, y;
    for(y = nd[x].child; y >= 0; y = nd[y].sib)
        merged |= merge_subtree(nd, y, ref, up, symmetric, TracerTypes);
    if(x == ref)
        return merged;
    struct SubNode * u = &nd[x];
    if(u->kind != NODE_RESOLVED || !(u->state == ST_ALIVE || u->state == ST_DIED) || !u->has_core)
        return merged;
    const struct SubNode * r = &nd[ref];
    if(!subhalo_sink_test(u->CorePos, u->CoreVel, r->CorePos, r->CoreVel, r->SigmaX, r->SigmaV, u->SigmaX, u->SigmaV, symmetric, up->BoxSize))
        return merged;
    if(u->state == ST_ALIVE) {
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
        u->was_alive = 1;
        merged = 1;
    }
    u->state = ST_SUNK;
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
    return x->kind == NODE_ORPHAN && x->state == ST_ALIVE;
}

static void
process_serial_key(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata)
{
    struct SerialCtx * sc = (struct SerialCtx *) userdata;
    struct SubhaloCallCtx * ctx = sc->ctx;
    const struct SubhaloUnbindParams * up = ctx->up;
    const struct SubhaloParams * sp = subhalo_get_params();
    const int64_t g = seg->g, start = seg->start, n = seg->n;
    const int pseudo = g > ctx->TotNgroups;
    const int TracerTypes = ctx->TracerTypes;
    const int CoreSize = sp->SubhaloCoreSize;
    const int MergeTrapped = sp->SubhaloMergeTrapped;
    const double R = sp->SubhaloSourceRelaxFactor;
    const int64_t h0 = ctx->HostedStart[k], nh = ctx->HostedStart[k + 1] - h0;
    struct SubNode * nd = sc->Nodes + ctx->ResStart[k];
    struct SubhaloResult * res = ctx->Res + ctx->ResStart[k];
    int64_t i, j;
    int d, v, nn = 0;
    struct SubhaloCallStats ks;     /* this key's diagnostics, added to ctx->Stats at the end */
    memset(&ks, 0, sizeof(ks));
    ks.WorstConvRatio = 1;
    ks.WorstTrackId = -1;
    (void) nthreads;

    if(n > INT32_MAX - 1)
        endrun(1, "Serial key %ld has %ld particles, more than the int32 bookkeeping of a key allows.\n", g, n);

    /* ---- a: the candidates: the members hosted by this key (sorted by TrackId), else the newborn ---- */
    for(j = 0; j < nh; j++) {
        const struct SubTrack * m = &ctx->Members[ctx->Hosted[h0 + j]];
        struct SubNode * u = &nd[nn++];
        memset(u, 0, sizeof(*u));
        u->TrackId = m->TrackId;
        u->old = m;
        u->kind = m->NboundPrev > 0 ? NODE_RESOLVED : NODE_ORPHAN;
    }
    if(nn == 0) {
        if(pseudo)
            endrun(5, "Pseudo-key %ld has no hostless track on rank %d.\n", g, ctx->ThisTask);
        memset(&nd[0], 0, sizeof(nd[0]));
        nd[0].TrackId = SUBHALO_PROVISIONAL(g);
        nd[0].kind = NODE_NEWBORN;
        nn = 1;
    }
    for(v = 0; v < nn; v++) {
        nd[v].parent = nd[v].child = nd[v].sib = nd[v].fparent = -1;
        nd[v].sinkinto = -1;
        nd[v].state = ST_NONE;
        res[v].valid = 0;
    }

    /* ---- the key's centre: mass-weighted over its FOF particles (all particles of a
     *      pseudo-key), periodic about the first one; computed locally (plan section 8) ---- */
    double xg[3] = {0, 0, 0}, vg[3] = {0, 0, 0};
    {
        int64_t first = -1;
        double M = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0};
        for(i = start; i < start + n; i++) {
            if(!part_live(&P[i]) || !(pseudo || P[i].GrNr == g))
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

    /* ---- b: the central (DecideCentrals, subhalo_tracking.cpp:767-800) ---- */
    int ci = 0;
    {
        double Mmax = -1;
        int nres = 0;
        for(v = 0; v < nn; v++)
            if(nd[v].kind == NODE_RESOLVED) {
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
                if(nd[v].kind != NODE_RESOLVED || m->Mbound < sp->SubhaloMajorProgenitorMassRatio * Mmax)
                    continue;
                dist = m->NPrev > 0 ? kinetic_distance(m->RefPos, m->RefVel, xg, vg, up) : 1e299;
            }
            else if(nd[v].kind == NODE_ORPHAN) {
                /* only orphans: every one is a candidate (their Mbound is 0), HBT resurrects the
                 * one closest in the kinetic distance (FeedCentrals on an orphan-only list) */
                double tp[3], tv[3];
                dist = orphan_tracer(m, ctx, tp, tv) ? kinetic_distance(tp, tv, xg, vg, up) : 1e299;
            }
            else
                continue;
            /* nodes are in TrackId order: strict < keeps the smaller TrackId on ties */
            if(bi < 0 || dist < best) {
                best = dist;
                bi = v;
            }
        }
        if(bi >= 0)
            ci = bi;
    }

    /* ---- nesting from NestedParentTrackId; detached members attach to the central ---- */
    for(v = 0; v < nn; v++) {
        if(v == ci)
            continue;
        const int p = nd[v].old ? node_index(nd, nn, nd[v].old->NestedParentTrackId) : -1;
        nd[v].parent = (p >= 0 && p != v && nd[p].kind == NODE_RESOLVED) ? p : ci;
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

    /* ---- sources: the owner map (ConstrainToSingleHost, FeedCentrals) ---- */
    int32_t * owner = (int32_t *) arena_need(arena, sizeof(int32_t) * (n + 1), g, "owner map");
    int32_t * ownlist = (int32_t *) arena_need(arena, sizeof(int32_t) * (n + 1), g, "source lists");
    for(i = 0; i < n; i++) {
        const struct particle_data * p = &P[start + i];
        int o = -1;
        if(part_live(p)) {
            const int t = p->TrackId >= 0 ? node_index(nd, nn, p->TrackId) : -1;
            if(t >= 0 && nd[t].kind == NODE_RESOLVED)
                o = t;                      /* a resolved member's own particles (both in g and in the field) */
            else if(pseudo || p->GrNr == g)
                o = ci;                     /* every other FOF particle feeds the central */
        }
        owner[i] = o;
    }

    /* ---- c: ReassignParticles (subhalo_reassign_gas.cpp) for the non-tracers ---- */
    if(sp->SubhaloReassignGas) {
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
            /* query batches of at most 4096, their buffers taken from a quarter of the space
             * left BEFORE the tree, whose build may use all the rest for its nodes */
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

    /* the own sources grouped by owner, key order within each */
    for(v = 0; v < nn; v++)
        nd[v].own_n = 0;
    for(i = 0; i < n; i++)
        if(owner[i] >= 0)
            nd[owner[i]].own_n++;
    {
        int64_t o = 0;
        for(v = 0; v < nn; v++) {
            nd[v].own_off = o;
            o += nd[v].own_n;
            nd[v].own_n = 0;
        }
        for(i = 0; i < n; i++)
            if(owner[i] >= 0) {
                struct SubNode * u = &nd[owner[i]];
                ownlist[u->own_off + u->own_n++] = (int32_t) i;
            }
    }

    /* ---- d: post-order unbinding with the remainder passed up, sinking, re-unbinding
     *      (RecursiveUnbind, subhalo_unbind.cpp:810-842) ---- */
    int nseq = 0;
    {
        int u = ci;
        nd[u].region = arena->used;
        nd[u].firstseq = nseq;
        for(;;) {
            if(nd[u].child >= 0 && !nd[u].visited) {
                nd[u].visited = 1;
                u = nd[u].child;
                nd[u].region = arena->used;
                nd[u].firstseq = nseq;
                continue;
            }
            nd[u].visited = 1;
            /* post: process u */
            struct SubNode * x = &nd[u];
            if(x->kind == NODE_ORPHAN && u != ci) {
                x->state = ST_NONE;
                x->seq = -1;
            }
            else {
                int c;
                int64_t input = x->own_n, reserve = 0, s;
                for(c = x->child; c >= 0; c = nd[c].sib) {
                    if(nd[c].state == ST_ALIVE) input += nd[c].c.N - nd[c].c.Nbound;
                    else if(nd[c].state == ST_DIED || nd[c].state == ST_FAILED) input += nd[c].c.N;
                }
                if(MergeTrapped)
                    for(s = x->firstseq; s < nseq; s++)
                        if(nd[nd[s].byseq].state == ST_ALIVE)
                            reserve += nd[nd[s].byseq].c.Nbound;
                const int64_t cap = input + reserve;
                x->c.Index = (int64_t *) arena_need(arena, sizeof(int64_t) * (cap + 1), g, "candidate list");
                x->c.N = 0;
                for(j = 0; j < x->own_n; j++)
                    x->c.Index[x->c.N++] = start + ownlist[x->own_off + j];
                for(c = x->child; c >= 0; c = nd[c].sib) {
                    int64_t a0 = -1;
                    if(nd[c].state == ST_ALIVE) a0 = nd[c].c.Nbound;
                    else if(nd[c].state == ST_DIED || nd[c].state == ST_FAILED) a0 = 0;
                    if(a0 >= 0)
                        for(j = a0; j < nd[c].c.N; j++)
                            x->c.Index[x->c.N++] = nd[c].c.Index[j];
                }
                /* the frame: the previous-bound frame of a resolved track, the tracer of a
                 * resurrected orphan, the key centre for a newborn (or when undefined) */
                for(d = 0; d < 3; d++) {
                    x->c.RefPos[d] = xg[d];
                    x->c.RefVel[d] = vg[d];
                }
                if(x->kind == NODE_RESOLVED && x->old->NPrev > 0)
                    for(d = 0; d < 3; d++) {
                        x->c.RefPos[d] = x->old->RefPos[d];
                        x->c.RefVel[d] = x->old->RefVel[d];
                    }
                else if(x->kind == NODE_ORPHAN)
                    orphan_tracer(x->old, ctx, x->c.RefPos, x->c.RefVel);
                x->c.TrackId = x->TrackId;
                x->c.Seed = node_seed(x->TrackId, ctx->SnapNum);
                x->c.Flags = 0;
                const size_t mark = arena->used;
                x->c.Energy = (double *) arena_need(arena, sizeof(double) * (cap + 1), g, "candidate energies");
                int alive = unbind_counted(&x->c, up, arena, &ks);
                arena->used = mark;
                if(alive)
                    x->state = ST_ALIVE;
                else if(x->kind == NODE_RESOLVED) {
                    x->state = ST_DIED;
                    /* its previous most-bound tracer (first own-tag tracer in BindRank order) */
                    for(j = 0; j < x->own_n; j++) {
                        const int64_t pi = start + ownlist[x->own_off + j];
                        if(P[pi].TrackId == x->TrackId && is_tracer(pi, TracerTypes)) {
                            set_mb_from(x, pi);
                            break;
                        }
                    }
                }
                else
                    x->state = ST_FAILED;
                node_core(x, up, CoreSize);
                /* sinking of the subtree into x, then x unbound again (subhalo_unbind.cpp:828-835) */
                if(x->state == ST_ALIVE && MergeTrapped
                   && merge_subtree(nd, u, u, up, sp->SubhaloSinkSymmetric, TracerTypes)) {
                    x->c.Energy = (double *) arena_need(arena, sizeof(double) * (cap + 1), g, "candidate energies");
                    alive = unbind_counted(&x->c, up, arena, &ks);
                    arena->used = mark;
                    x->state = alive ? ST_ALIVE : (x->kind == NODE_RESOLVED ? ST_DIED : ST_FAILED);
                    node_core(x, up, CoreSize);
                }
                x->c.Energy = NULL;
                x->seq = nseq;
                nd[nseq].byseq = u;
                nseq++;
                /* compaction of the subtree's lists: a survivor keeps its relaxed source (bound
                 * set and tail, TruncateSource), everything else nothing; x keeps its whole
                 * list until its parent has gathered the remainder */
                char * dst = arena->base + x->region;
                for(s = x->firstseq; s < x->seq; s++) {
                    struct SubNode * w = &nd[nd[s].byseq];
                    int64_t keep = 0;
                    if(w->state == ST_ALIVE)
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
                if(x->state == ST_ALIVE) {
                    int64_t ks = (int64_t) (x->c.Nbound * R);
                    if(ks < x->c.Nbound) ks = x->c.Nbound;
                    if(ks > x->c.N) ks = x->c.N;
                    x->keep = ks;
                }
                else
                    x->keep = 0;
            }
            if(u == ci)
                break;
            if(nd[u].sib >= 0) {
                u = nd[u].sib;
                nd[u].region = arena->used;
                nd[u].firstseq = nseq;
                continue;
            }
            u = nd[u].parent;
        }
    }
    /* the root's own list to its relaxed size too (nothing above it gathers its remainder) */
    if(nd[ci].state == ST_ALIVE)
        nd[ci].c.N = nd[ci].keep;
    for(v = 0; v < nn; v++)
        if(nd[v].state == ST_ALIVE && v != ci)
            nd[v].c.N = nd[v].keep;

    /* ---- the final pass, the HBT properties (step 7g) and the Phase 3 bound properties and
     *      baryonic sums of every survivor ---- */
    const double vu = subhalo_vel_to_kms(1.0);   /* the unit factor: subhalo_props_hbt divides by a only */
    int nalive = 0;
    for(v = 0; v < nn; v++) {
        struct SubNode * x = &nd[v];
        if(x->state != ST_ALIVE)
            continue;
        nalive++;
        struct SubTrack * rec = &res[v].rec;
        subhalo_track_carry(rec, x->old, x->TrackId, ctx->SnapNum);
        const size_t mark = arena->used;
        x->c.Energy = (double *) arena_need(arena, sizeof(double) * (x->c.Nbound + 1), g, "final-pass energies");
        const int64_t npos = subhalo_final_pass(&x->c, up, arena);
        ks.NFinalPositive += npos;
        ks.NFinalBound += x->c.Nbound;
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

    /* ---- f: ranks by mass and the final hierarchy (UpdateTracks: SortMemberLists,
     *      ExtendCentralNest, AssignRanks, FillDepth, subhalo_tracking.cpp:1489-1492) ---- */
    int X = -1;
    if(nalive > 0) {
        /* ranks: insertion by CompareMass (a key has few members) */
        int r = 0;
        for(v = 0; v < nn; v++)
            nd[v].rank = -1;
        for(r = 0; r < nalive; r++) {
            int b = -1;
            for(v = 0; v < nn; v++)
                if(nd[v].state == ST_ALIVE && nd[v].rank < 0 && (b < 0 || mass_before(&res[v].rec, &res[b].rec)))
                    b = v;
            nd[b].rank = r;
        }
        for(v = 0; v < nn; v++)
            if(nd[v].rank == 0)
                X = v;
        /* surviving parent: the nearest surviving ancestor in the processing tree */
        for(v = 0; v < nn; v++) {
            if(nd[v].state != ST_ALIVE)
                continue;
            int u = nd[v].parent;
            while(u >= 0 && nd[u].state != ST_ALIVE)
                u = nd[u].parent;
            nd[v].fparent = u;
        }
        /* the most massive survivor is the central: it leaves its parent and every other top
         * survivor (the decided central among them) is nested under it */
        nd[X].fparent = -1;
        for(v = 0; v < nn; v++) {
            if(nd[v].state != ST_ALIVE || v == X)
                continue;
            int u = nd[v].parent;
            while(u >= 0 && nd[u].state != ST_ALIVE)
                u = nd[u].parent;
            if(u < 0)
                nd[v].fparent = X;
        }
        for(v = 0; v < nn; v++) {
            if(nd[v].state != ST_ALIVE)
                continue;
            int u = v, dd = 0;
            while(nd[u].fparent >= 0 && dd <= nn) {
                u = nd[u].fparent;
                dd++;
            }
            nd[v].fdepth = dd;
        }
        for(v = 0; v < nn; v++) {
            if(nd[v].state != ST_ALIVE)
                continue;
            struct SubTrack * rec = &res[v].rec;
            rec->HostGroup = pseudo ? -1 : g;
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
                for(d = 0; d < 3; d++) {
                    rec->TracerPos[d] = -1;
                    rec->TracerVel[d] = -1;
                }
            }
        }
    }

    /* ---- 8: the tags (D10): top-down reservation of the most-bound tracers, deepest source
     *      list for the rest, BindRank in own-tag energy order (bound prefix first, then the
     *      tail; the reservation decides ownership only, J12), BoundTrackId the survivor that
     *      bound the particle. A tail claim is valid only if the particle's binding owner is the
     *      claimant or one of its ancestors in the FINAL tree (D1, J2): after the central
     *      promotion above a processing ancestor can sit on another branch, and its bound
     *      particles then stay in the deepest claimant whose branch contains it (at least the
     *      binding owner itself, whose bound set claims them). Bound sets are unchanged. The
     *      owner map's memory is reused. ---- */
    int32_t * tagnode = owner;
    int32_t * bnode = ownlist;
    for(i = 0; i < n; i++) {
        tagnode[i] = -1;
        bnode[i] = -1;
    }
    for(v = 0; v < nn; v++) {
        struct SubNode * x = &nd[v];
        if(x->state != ST_ALIVE)
            continue;
        int nr = 0;
        for(j = 0; j < x->c.Nbound; j++) {
            const int64_t li = x->c.Index[j] - start;
            bnode[li] = v;
            if(nr < sp->SubhaloMinNumTracerPart && is_tracer(x->c.Index[j], TracerTypes)) {
                tagnode[li] = v | TAG_RESERVED;
                nr++;
            }
        }
    }
    int64_t nrejected = 0;
    for(v = 0; v < nn; v++) {
        struct SubNode * x = &nd[v];
        if(x->state != ST_ALIVE)
            continue;
        for(j = 0; j < x->c.N; j++) {
            const int64_t li = x->c.Index[j] - start;
            const int t = tagnode[li];
            if(t >= 0 && (t & TAG_RESERVED))
                continue;
            if(!(t < 0 || nd[v].fdepth > nd[t].fdepth || (nd[v].fdepth == nd[t].fdepth && nd[v].TrackId < nd[t].TrackId)))
                continue;
            const int b = bnode[li];
            if(b >= 0 && b != v && !final_ancestor(nd, v, b, nn)) {
                nrejected++;
                continue;
            }
            tagnode[li] = v;
        }
    }
    for(v = 0; v < nn; v++) {
        struct SubNode * x = &nd[v];
        if(x->state != ST_ALIVE)
            continue;
        struct SubTrack * rec = &res[v].rec;
        int64_t r = 0, nex = 0, ntl = 0;
        /* Index[0, Nbound) is in final-pass energy order, Index[Nbound, N) the tail in the energy
         * order of its unbinding: one pass gives the own-tag energy order */
        for(j = 0; j < x->c.N; j++) {
            const int64_t pi = x->c.Index[j];
            const int64_t li = pi - start;
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
            else {
                rec->NtailType[P[pi].Type]++;
                ntl++;
            }
        }
        rec->NboundExclusive = (int32_t) nex;
    }
    /* everything else in the key is untagged (swallowed particles keep their tags, H6) */
    for(i = 0; i < n; i++) {
        if(tagnode[i] >= 0 || !part_live(&P[start + i]))
            continue;
        P[start + i].TrackId = -1;
        P[start + i].BindRank = -1;
        P[start + i].BoundTrackId = -1;
    }

    /* ---- h: descendants of the tracks resolved at the previous call that died or sank: their
     *      step-0 tracers in the key (by ID), the rest queued for step 11 ---- */
    arena->used = 0;
    int ndead = 0;
    for(v = 0; v < nn; v++)
        if(nd[v].kind == NODE_RESOLVED && (nd[v].state == ST_DIED || nd[v].state == ST_SUNK) && nd[v].old->NTracers > 0)
            ndead++;
    struct IdLocal * ids = NULL;
    if(ndead > 0) {
        ids = (struct IdLocal *) arena_need(arena, sizeof(struct IdLocal) * (n + 1), g, "descendant index");
        for(i = 0; i < n; i++) {
            ids[i].ID = P[start + i].ID;
            ids[i].i = start + i;
        }
        qsort(ids, n, sizeof(struct IdLocal), cmp_idlocal);
    }

    /* ---- results ---- */
    double maxmerge = 0;
    for(v = 0; v < nn; v++) {
        struct SubNode * x = &nd[v];
        struct SubhaloResult * rs = &res[v];
        rs->ndesc = 0;
        rs->history = 0;
        rs->newborn = 0;
        rs->MaxMergerMbound = 0;
        if(x->state == ST_ALIVE) {
            rs->valid = 1;
            rs->history = 1;
            rs->newborn = x->kind == NODE_NEWBORN;
            /* a track that arrived as a satellite of the central this call */
            if(X >= 0 && v != X && x->old && (x->old->Rank <= 0 || x->old->CentralTrackId != nd[X].TrackId)
               && x->old->Mbound > maxmerge)
                maxmerge = x->old->Mbound;
            continue;
        }
        if(x->kind == NODE_NEWBORN) {
            rs->valid = 0;
            continue;
        }
        struct SubTrack * rec = &rs->rec;
        *rec = *x->old;
        rs->valid = 1;
        rec->HostGroup = pseudo ? -1 : g;
        if(X >= 0)
            rec->CentralTrackId = nd[X].TrackId;
        /* a track that dies or sinks inside a group with another central is a satellite at its
         * death: the satellite branch of the SnapLastIsolation rule (subhalo_history_update; HBT
         * ranks the new orphan as a satellite). Orphans of earlier calls keep their history. */
        if(!pseudo && X >= 0 && X != v && x->kind == NODE_RESOLVED && (x->state == ST_DIED || x->state == ST_SUNK)
           && rec->SnapLastIsolation < 0)
            rec->SnapLastIsolation = ctx->SnapNum - 1;
        if(x->kind == NODE_RESOLVED && (x->state == ST_DIED || x->state == ST_SUNK)) {
            record_mb(rec, x, ctx);
            const int64_t sink = x->state == ST_SUNK ? nd[x->sinkinto].TrackId : SUBHALO_NOTRACK;
            subhalo_track_make_orphan(rec, sink, ctx->SnapNum);
            /* the nearest surviving ancestor */
            int u = x->parent;
            while(u >= 0 && nd[u].state != ST_ALIVE)
                u = nd[u].parent;
            rec->NestedParentTrackId = u >= 0 ? nd[u].TrackId : (X >= 0 && X != v ? nd[X].TrackId : -1);
            if(x->state == ST_SUNK && x->sinkinto == X && x->old->Mbound > maxmerge)
                maxmerge = x->old->Mbound;
            /* descendant votes: the tracers in this key, the others pending */
            int t;
            for(t = 0; t < x->old->NTracers && t < SUBHALO_MAXTRACERS; t++) {
                struct IdLocal key = {x->old->TracerIDs[t], -1};
                int64_t lo = 0, hi = n;
                while(lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if(ids[mid].ID < key.ID) lo = mid + 1;
                    else hi = mid;
                }
                if(lo < n && ids[lo].ID == key.ID)
                    rs->DescOwner[t] = part_live(&P[ids[lo].i]) ? P[ids[lo].i].BoundTrackId : -1;
                else
                    rs->DescOwner[t] = SUBHALO_DESC_PENDING;
            }
            rs->ndesc = t;
        }
        /* an orphan that stays one (or failed resurrection) keeps its record and step-1 state */
    }
    if(X >= 0)
        res[X].MaxMergerMbound = maxmerge;
    arena->used = 0;
    ks.NTailRejected = nrejected;
    stats_merge(&ctx->Stats, &ks);
}

void
subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx)
{
    struct SerialCtx sc;
    sc.ctx = ctx;
    sc.Nodes = (struct SubNode *) mymalloc("SubNodes", sizeof(struct SubNode) * (ctx->NRes + 1));
    /* per key: the alignment allowance of its candidate nodes' lists (J7) */
    int64_t * extra = (int64_t *) mymalloc("SubArenaExtra", sizeof(int64_t) * (nsegs + 1));
    int64_t k;
    for(k = 0; k < nsegs; k++)
        extra[k] = (ctx->HostedStart[k + 1] - ctx->HostedStart[k] + 1) * (int64_t) SUBHALO_ARENA_NODE_BYTES;
    const double peak = subhalo_run_serial_keys(segs, nsegs, subhalo_arena_bytes_per_part(), extra, process_serial_key, &sc);
    if(peak > ctx->Stats.ArenaPeakFrac)
        ctx->Stats.ArenaPeakFrac = peak;
    myfree(extra);
    myfree(sc.Nodes);
}

/* ------------------------------------------------------------------ in place (step 9) */

static void
keysum_add(struct KeySums * s, const struct particle_data * p, const double ref[3], int TracerTypes, double BoxSize)
{
    int d;
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

static int
cmp_inplace_key_g(const void * a, const void * b)
{
    const int64_t x = ((const struct SubhaloInplaceKey *) a)->g, y = ((const struct SubhaloInplaceKey *) b)->g;
    return (x > y) - (x < y);
}

/* A home field particle of the fallback: in-place key index, new tag, old rank and ID. */
struct HomeField {
    int64_t k;
    int64_t tag;
    int64_t oldrank;
    MyIDType ID;
    int64_t i;
};

static int
cmp_homefield(const void * a, const void * b)
{
    const struct HomeField * x = (const struct HomeField *) a, * y = (const struct HomeField *) b;
    if(x->k != y->k) return (x->k > y->k) - (x->k < y->k);
    if(x->oldrank != y->oldrank) return (x->oldrank > y->oldrank) - (x->oldrank < y->oldrank);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* The field half of the Phase 1 fallback (audit J3). A live field particle that did not travel
 * (GrNr == -1 after the serial keys) is tagged by a track whose key is oversized, or by an
 * orphan (Key -1, a stale tag of an earlier fallback). Under the Phase 1 rule the processed
 * record and the particle tags must describe the same set:
 *  - oversized FOF key g with winner W: every such particle of a track keyed to g (W's own,
 *    those of the members that sink into W, those of a non-member keyed to g) goes to W as
 *    bound (TrackId = BoundTrackId = W), ranked after W's FOF particles; fsum[k] returns their
 *    sums for W's record (in the frame of plan->Inplace[k].ref);
 *  - oversized pseudo-key of a hostless track T: T keeps its field particles, all bound to T,
 *    dense ranks from 0 (subhalo_update_unhosted then counts exactly this set);
 *  - Key -1 (an orphan or dead track): released (untagged).
 * Ranks are dense per key over the ranks in rank order (MPI_Exscan), locally by (old BindRank,
 * ID). Collective. */
static void
normalise_home_field(const struct SubhaloPlan * plan, const int64_t * winner, struct KeySums * fsum,
                     struct SubhaloCallCtx * ctx, struct SubTrackTable * tab, struct SubhaloCallStats * ks, MPI_Comm Comm)
{
    const int64_t K = plan->NInplace;
    int64_t i, k;
    if(K > 0)
        memset(fsum, 0, sizeof(struct KeySums) * K);
    int64_t nh = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(part_live(&P[i]) && P[i].GrNr == -1 && P[i].TrackId >= 0)
            nh++;
    if(!MPIU_Any(nh > 0, Comm))
        return;
    struct SubhaloTrackKey * map = NULL;
    const int64_t nmap = subhalo_home_field_keys(tab, &map, Comm);
    struct HomeField * hf = (struct HomeField *) mymalloc("SubHomeField", sizeof(struct HomeField) * (nh + 1));
    int64_t n = 0, nrel = 0;
    int bad = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        struct particle_data * p = &P[i];
        if(!(part_live(p) && p->GrNr == -1 && p->TrackId >= 0))
            continue;
        const int64_t key = subhalo_track_key_lookup(map, nmap, p->TrackId);
        if(key < 0) {
            p->TrackId = -1;
            p->BindRank = -1;
            p->BoundTrackId = -1;
            nrel++;
            continue;
        }
        struct SubhaloInplaceKey ikey;
        ikey.g = key;
        const struct SubhaloInplaceKey * ip = (const struct SubhaloInplaceKey *)
            bsearch(&ikey, plan->Inplace, K, sizeof(ikey), cmp_inplace_key_g);
        if(!ip) {
            bad = 1;
            continue;
        }
        struct HomeField * e = &hf[n++];
        e->k = ip - plan->Inplace;
        e->tag = key <= ctx->TotNgroups ? winner[e->k] : p->TrackId;
        e->oldrank = p->BindRank;
        e->ID = p->ID;
        e->i = i;
    }
    if(bad)
        endrun(5, "A field particle on rank %d stayed home although its track's key is serial.\n", ctx->ThisTask);
    qsort(hf, n, sizeof(struct HomeField), cmp_homefield);
    int64_t * cnt = (int64_t *) mymalloc("SubHomeCnt", sizeof(int64_t) * (K + 1));
    int64_t * off = (int64_t *) mymalloc("SubHomeOff", sizeof(int64_t) * (K + 1));
    for(k = 0; k < K; k++)
        cnt[k] = off[k] = 0;
    for(i = 0; i < n; i++)
        cnt[hf[i].k]++;
    MPI_Exscan(cnt, off, K, MPI_INT64, MPI_SUM, Comm);
    if(ctx->ThisTask == 0)
        for(k = 0; k < K; k++)
            off[k] = 0;
    struct KeySums * ls = (struct KeySums *) mymalloc("SubHomeSums", sizeof(struct KeySums) * (K + 1));
    memset(ls, 0, sizeof(struct KeySums) * (K + 1));
    int64_t a = 0;
    int toolarge = 0;
    for(i = 0; i < n; i++) {
        k = hf[i].k;
        if(i == 0 || hf[i].k != hf[i-1].k)
            a = 0;
        struct particle_data * p = &P[hf[i].i];
        const int64_t base = plan->Inplace[k].g <= ctx->TotNgroups ? plan->Inplace[k].Length : 0;
        const int64_t r = base + off[k] + (a++);
        if(r > INT32_MAX)
            toolarge = 1;
        p->TrackId = hf[i].tag;
        p->BoundTrackId = hf[i].tag;
        p->BindRank = (int32_t) r;
        if(plan->Inplace[k].g <= ctx->TotNgroups)
            keysum_add(&ls[k], p, plan->Inplace[k].ref, ctx->TracerTypes, ctx->BoxSize);
    }
    if(toolarge)
        endrun(5, "In-place key on rank %d: BindRank beyond the int32 range.\n", ctx->ThisTask);
    if(K > 0)
        MPI_Allreduce(ls, fsum, K * KEYSUMS_NDOUBLE, MPI_DOUBLE, MPI_SUM, Comm);
    ks->NFallbackField += n;
    ks->NFallbackReleased += nrel;
    myfree(ls);
    myfree(off);
    myfree(cnt);
    myfree(hf);
    myfree(map);
}

void
subhalo_process_inplace(const struct SubhaloKeySeg * segs, int64_t nsegs, const struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, struct SubTrackTable * tab, MPI_Comm Comm)
{
    const int64_t K = plan->NInplace;
    int64_t i, k, s;
    int d;
    const int ThisTask = plan->ThisTask;
    struct SubhaloCallStats ks;
    memset(&ks, 0, sizeof(ks));
    ks.WorstConvRatio = 1;
    ks.WorstTrackId = -1;
    /* every rank has the same list, so these branches are collective-safe */
    if(K > 0)
        message(0, "WARNING: Subhalo finder: %ld oversized key(s) processed in place with the Phase 1 rule "
                   "(every FOF particle and the field particles of their tracks to one track, no unbinding) until Phase 2b.\n", K);

    struct KeySums * ls = (struct KeySums *) mymalloc("SubIplSums", sizeof(struct KeySums) * (K + 1));
    struct KeySums * gs = (struct KeySums *) mymalloc("SubIplSumsG", sizeof(struct KeySums) * (K + 1));
    struct KeySums * fs = (struct KeySums *) mymalloc("SubIplSumsF", sizeof(struct KeySums) * (K + 1));
    int64_t * cnt = (int64_t *) mymalloc("SubIplCnt", sizeof(int64_t) * (K + 1));
    int64_t * off = (int64_t *) mymalloc("SubIplOff", sizeof(int64_t) * (K + 1));
    int64_t * best = (int64_t *) mymalloc("SubIplBest", sizeof(int64_t) * (K + 1));
    int64_t * gbest = (int64_t *) mymalloc("SubIplBestG", sizeof(int64_t) * (K + 1));
    double * msink = (double *) mymalloc("SubIplSink", sizeof(double) * (K + 1));
    double * gmsink = (double *) mymalloc("SubIplSinkG", sizeof(double) * (K + 1));
    int64_t * winner = (int64_t *) mymalloc("SubIplWinner", sizeof(int64_t) * (K + 1));
    memset(ls, 0, sizeof(struct KeySums) * (K + 1));
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
    if(K > 0)
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
    if(K > 0)
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
    if(K > 0)
        MPI_Allreduce(msink, gmsink, K, MPI_DOUBLE, MPI_MAX, Comm);

    /* two-stage centre and the global rank prefix (vector exclusive scan in rank order) */
    if(K > 0) {
        MPI_Allreduce(ls, gs, K * KEYSUMS_NDOUBLE, MPI_DOUBLE, MPI_SUM, Comm);
        MPI_Exscan(cnt, off, K, MPI_INT64, MPI_SUM, Comm);
    }
    if(ThisTask == 0)
        for(k = 0; k < K; k++)
            off[k] = 0;

    /* tags of the FOF particles: all bound to the winner, ranks in the local sort order */
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

    /* the field particles of the oversized keys' tracks, and stale orphan tags (J3); every rank */
    normalise_home_field(plan, winner, fs, ctx, tab, &ks, Comm);
    stats_merge(&ctx->Stats, &ks);
    if(K == 0) {
        myfree(winner); myfree(gmsink); myfree(msink); myfree(gbest); myfree(best); myfree(off); myfree(cnt);
        myfree(fs); myfree(gs); myfree(ls);
        return;
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
        if((int64_t) gs[k].count != seg->Length)
            endrun(5, "In-place key %ld: %ld particles counted, length %ld.\n", seg->g, (int64_t) gs[k].count, seg->Length);
        /* the record counts exactly the particles tagged above: FOF and field (J3) */
        struct KeySums all = gs[k];
        double * a = (double *) &all;
        const double * b = (const double *) &fs[k];
        size_t q;
        for(q = 0; q < KEYSUMS_NDOUBLE; q++)
            a[q] += b[q];
        double cen[3];
        for(d = 0; d < 3; d++)
            cen[d] = seg->ref[d] + (all.mass > 0 ? all.mdx[d] / all.mass : 0);
        struct SubhaloResult * res = &ctx->Res[ctx->ResStart[s]];
        /* the most-bound particle of the Phase 1 rule: the first tracer-type particle in rank
         * order (BindRank 0.. are on this, the lowest holding rank), else the rank-0 particle */
        int64_t mb = seg->start;
        for(i = seg->start; i < seg->start + seg->n; i++)
            if(part_live(&P[i]) && is_tracer(i, ctx->TracerTypes)) {
                mb = i;
                break;
            }
        fill_winner_record(&res->rec, old, winner[k], seg->g, &all, cen, mb, 1, ctx);
        res->MaxMergerMbound = gmsink[k];
        res->newborn = newborn;
        res->valid = 1;
        res->history = 1;
        res->ndesc = 0;
        res->ipk = k;
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
    myfree(fs);
    myfree(gs);
    myfree(ls);
}
