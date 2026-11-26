#ifndef SUBHALO_COLLECTIVE_INTERNAL_H
#define SUBHALO_COLLECTIVE_INTERNAL_H

/* State shared by the files of the in-place path: the rounds (subhalo_collective.c), the walks
 * (subhalo_collective_walk.c) and the distributed properties (subhalo_collective_props.c). */

#include <stdint.h>
#include <string.h>
#include "partmanager.h"
#include "subhalo_unbind.h"
#include "subhalo_collective.h"

/* per-particle flags */
#define CF_BOUND   1     /* bound by its source candidate (Cand) */
#define CF_RES     2     /* one of the reserved most-bound tracers of Cand */
#define CF_SCLAIM  4     /* a serial claim: the tag of a serial sub-tree member (a tail, or a root's
                          * remainder inside its relaxed source), valid while that member is alive */
#define CF_SBOUND  8     /* bound by a serial sub-tree member (BoundTrackId), excluded from the sources */
#define CF_UNIV    16    /* a particle of an oversized key: the final tag pass writes it */
#define CF_RELEASE 32    /* a field particle tagged by an orphan or the central: untagged by the final pass */

/* The diagnostics of the call (subhalo_collective_last_stats). */
extern struct SubhaloCollectiveStats CollStats;

/* The particle count a tree build of the in-place walks is told on a nearly empty rank, to hold the
 * top-level grid and the per-thread node caches (the builds and the memory estimate). */
static inline int64_t
coll_tree_min_part(int64_t nleaf, int nthr)
{
    return 2048 + 512 * (int64_t) nthr + 4 * nleaf * nthr;
}

/* The state of a candidate during the call, identical on every rank. */
struct CandState {
    int state;               /* SUBHALO_ST_* */
    int iterating;           /* in the current unbinding loop */
    int ran;                 /* unbound by the current call of unbind_loop */
    int capped;              /* that unbinding stopped at the iteration cap */
    int flags;               /* SUBHALO_FLAG_ITERCAP */
    int iterations;          /* of the last unbinding */
    int absorbed;            /* a survivor sank into it this round */
    int has_core;
    int has_mb;
    int64_t N, Nbound, Nlast;
    double ConvRatio;
    double Ref0[3];          /* the reference of the nearest-image sums */
    double RefPos[3], RefVel[3];
    struct SubhaloCore core; /* the sinking core (loop order) */
    MyIDType mbID;           /* most-bound tracer at the sinking (alive) or the previous one (died) */
    double mbPos[3], mbVel[3];
    int64_t keep;            /* relaxed source: max(Nbound, min(N, R Nbound)) */
    int64_t SinkTrackId;
    /* the current record quantities the hierarchy needs */
    float Mbound_rec, Vmax_rec;
    MyIDType MostBoundID_rec;
};

/* A local particle admitted to an in-place receiver's source: its owner's and its receiver's
 * candidates. */
struct CollAdm {
    int64_t index;
    int32_t owner, recv;
};
_Static_assert(sizeof(struct CollAdm) <= SUBHALO_INPLACE_ADM_BYTES, "SUBHALO_INPLACE_ADM_BYTES does not cover struct CollAdm");

struct Coll {
    struct SubhaloPlan * plan;
    struct SubhaloCallCtx * ctx;
    DomainDecomp * ddecomp;
    MPI_Comm Comm;
    int NTask, ThisTask;
    const struct SubhaloUnbindParams * up;
    const struct SubhaloParams * sp;
    int64_t NumPart;
    int32_t * Cand;
    int32_t * Claim;
    uint8_t * Flag;
    double * En;
    double * ClaimE;
    struct CandState * cs;
    int64_t c0, c1;          /* the current round */
    struct CollAdm * Adm;    /* this rank's particles admitted to in-place receivers, by index */
    int64_t NAdm, NAdmTaken;
    int AdmAny;              /* 1: some rank holds admitted particles (the receivers are then counted) */
    int64_t NAdmRecv;        /* in-place receivers that took at least one admitted particle (global) */
    int nwalks;
    struct SubhaloCallStats st;
};

static inline int
is_tracer(int64_t i, int TracerTypes)
{
    return (TracerTypes >> P[i].Type) & 1;
}

static inline uint64_t
double_key(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
}

static inline int
in_round(const struct Coll * C, int32_t c)
{
    return c >= C->c0 && c < C->c1;
}

/* What the owner tells every rank about a candidate's bound set in energy order: one row per
 * candidate, zero on the other ranks and summed as CORE_ND doubles. */
#define CORE_ND 15
struct CoreRow {
    struct SubhaloCore core;
    double mbPos[3], mbVel[3];   /* the most-bound tracer */
    double has;                  /* 1: the candidate has bound particles */
};
_Static_assert(sizeof(struct CoreRow) == CORE_ND * sizeof(double), "struct CoreRow is reduced as CORE_ND doubles");
struct CoreOut {
    struct CoreRow * d;      /* [nc] */
    MyIDType * mbID;         /* [nc] */
    int64_t * resv;          /* [nc][2 K]: (rank, index) of the reserved tracers, -1 */
};

/* Per-candidate results of a round, on every rank (the owner writes the record from them). */
#define HS_N 0
#define HS_NTR 1
#define HS_NT 2
#define HS_MT 8
#define HS_M 14
#define HS_K 15
#define HS_U 16
#define HS_J 17
#define HS_I 20
#define HS_SFR 26
#define HS_HE 27
#define HS_BHM 28
#define HS_MDOT 29
#define HS_GZ 30
#define HS_SZ 31
#define HS_GZE 32
#define HS_SZE (32 + NMETALS)
#define NHS (32 + 2 * NMETALS)

struct RoundProps {
    double * hs;             /* [nc][NHS] holder-side sums */
    double * bhm;            /* [nc] most massive BH subgrid mass, -1 */
    MyIDType * bhid;         /* [nc] its ID, (MyIDType) -1 */
    /* the radial profile */
    double * Mso;            /* [nc][NSO] */
    int64_t * kc;            /* [nc][NSO] crossing positions, -1 */
    double * prof;           /* [nc][8]: hbt best M/r, Rmax, vir best M/r, RmaxVir, rhalf, Renclose, (unused) x2 */
    double * jv;             /* [nc][9]: JVir sums 3, x sums 3, v sums 3 */
    float * shape;           /* [nc][2][5]: BtoA, CtoA, AxisA[3] (vir, 500c), -1 */
    double * dfr;            /* [nc][8]: Rs, RockstarCoreVel 3, BulkVel 3, Vrms (km/s), -1 */
};

struct RadRec {
    uint64_t key[3];         /* radix: ID, r, label (most significant last) */
    double r, m;
    double dx[3];            /* NEAREST(x - core) */
    double vel[3];
};

/* Bytes per radial record kept through the properties (the record, its position, M(<= r)) and the
 * most allocations round_props holds at once (the round's vectors, the radial set, the deferred
 * recipe's), each costing the allocator up to two alignment blocks (utils/memory.c). */
#define COLL_PROPS_REC_BYTES (sizeof(struct RadRec) + 8 + 8)
#define COLL_PROPS_NALLOC 32
#define COLL_ALLOC_OVERHEAD (2. * 4096.)

/* The round's potential-only walk, Sum[i] = sum_j m_j W (subhalo_collective_walk.c). Collective. */
void coll_walk(struct Coll * C, double * Sum);

/* The properties of the round's alive candidates into rp, allocated on the bottom of the stack and
 * freed by free_round_props (subhalo_collective_props.c). Collective. */
void round_props(const struct Coll * C, const struct CoreOut * core, struct RoundProps * rp);
void free_round_props(struct RoundProps * rp);

#endif
