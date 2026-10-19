#ifndef SUBHALO_INTERNAL_H
#define SUBHALO_INTERNAL_H

/* Private interface of the subhalo finder, shared by its sources and the unit tests (design 7.1).
 * Memory rule of a call: what must survive the exchange or the domain restore lives on the top of
 * the main stack (mymalloc2), since domain_exchange may myrealloc the slots, the last bottom
 * allocation; bottom allocations are scratch freed by the function that made them. */

#include <math.h>
#include "subhalo.h"

/* Per-thread arena bytes per particle of a key (design 7.1): the kernel's 112
 * (SUBHALO_UNBIND_BYTES_PER_PART, whose scratch the properties reuse), 8 for the owner map and
 * grouped sources, 8 for completed nodes' lists awaiting their parent, 8 R for the survivors'
 * relaxed sources (R = SubhaloSourceRelaxFactor rounded up to 1/8; bound sets are disjoint), plus
 * SUBHALO_ARENA_NODE_BYTES per candidate node: subhalo_arena_bytes_per_part() = 128 + 8 R. */
#define SUBHALO_ARENA_FIXED_BYTES_PER_PART 128
#define SUBHALO_ARENA_NODE_BYTES 64
#define SUBHALO_MAX_RELAX_FACTOR 10.
size_t subhalo_arena_bytes_per_part(void);
/* Main-stack bytes per candidate node of a serial key (struct SubNode in subhalo_track.c;
 * checked there by a static assertion), charged by the plan per member track. */
#define SUBHALO_NODE_BYTES 512
/* Bytes every rank holds per in-place key, per in-place candidate and per member track of an
 * oversized group (static assertions in subhalo_collective.c); the plan charges the actual counts. */
#define SUBHALO_INPLACE_KEY_BYTES 512
#define SUBHALO_INPLACE_CAND_BYTES 4096
#define SUBHALO_INPLACE_MEMBER_BYTES 768
/* Transient per-candidate scratch of a round's properties on every rank (static assertions in
 * subhalo_collective.c); charged by the admission and, with SUBHALO_INPLACE_CAND_BYTES, the packing. */
#define SUBHALO_INPLACE_CAND_PROPS_BYTES 6144

/* Estimated cost of a serial key of N particles and M member tracks (only ratios matter): N log2 N
 * for the unbinding trees, a term per member candidate and a per-key constant, fitted to measured
 * dm-128 key times (one unit about 1e-7 s; the per-key constant kept low so the many small keys do
 * not dominate the packing; decision 41). The planner uses M = 0, as the directory's Tags counts a
 * track once per holding rank; subhalo_run_serial_keys uses the segment's distinct tags. */
#define SUBHALO_COST_ITER 10.
#define SUBHALO_COST_MEMBER 1500.
#define SUBHALO_COST_KEY 2000.
static inline double subhalo_key_cost(int64_t N, int64_t M) {
    const double n = N > 2 ? (double) N : 2.;
    return SUBHALO_COST_ITER * (double) (N > 0 ? N : 0) * log2(n) + SUBHALO_COST_MEMBER * (double) (M > 0 ? M : 0) + SUBHALO_COST_KEY;
}

/* Catalogue int32 guards (the schema columns are i4): a count of n particles fits if
 * 0 <= n <= INT32_MAX; a dense 0-based rank r fits only if r < INT32_MAX (the rank INT32_MAX
 * stands for INT32_MAX + 1 particles). n * elsize bytes fit an MPI int count if <= INT_MAX. */
static inline int subhalo_count_fits_i32(double n) { return n >= 0 && n <= (double) INT32_MAX; }
static inline int subhalo_rank_fits_i32(int64_t r) { return r >= 0 && r < (int64_t) INT32_MAX; }
static inline int subhalo_mpi_bytes_fit(int64_t n, size_t elsize) {
    return n >= 0 && (elsize == 0 || n <= (int64_t) (2147483647 / elsize));
}

/* ------------------------------------------------------------------ shared helpers (subhalo.c) */

/* Finite test that survives -ffast-math (which lets the compiler assume isfinite() is true):
 * the exponent bits of an IEEE double. */
int subhalo_isfinite(double x);

/* x wrapped into [0, BoxSize) by repeated whole-box shifts (the finder's one periodic wrap). */
static inline double
subhalo_wrap_box(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* Compensated (Neumaier) sum s[0] + s[1], used by the serial and distributed property recipes (the
 * latter merges per-rank pairs in rank order) so that equal masses sum exactly whatever the grouping
 * and other inputs agree to about one rounding. volatile keeps -ffast-math from dropping the errors. */
static inline void
subhalo_ksum_add(double s[2], double x)
{
    volatile double t = s[0] + x;
    volatile double d, e, c;
    if(fabs(s[0]) >= fabs(x)) {
        d = s[0] - t;
        e = d + x;
    }
    else {
        d = x - t;
        e = d + s[0];
    }
    c = s[1] + e;
    s[0] = t;
    s[1] = c;
}
/* a += b (b to the right of a in the summation order) */
static inline void
subhalo_ksum_merge(double a[2], const double b[2])
{
    subhalo_ksum_add(a, b[0]);
    volatile double c = a[1] + b[1];
    a[1] = c;
}
static inline double
subhalo_ksum_value(const double s[2])
{
    volatile double v = s[0] + s[1];
    return v;
}

/* The velocity trio (subhalo_props.c) from the compensated aperture sums
 * about the core velocity cvel (Vel units): Mb the aperture mass, Sb and S2 its first and second
 * moments, Mc and Sc those of the Rockstar core; vkms converts Vel to km/s. One function for the
 * serial and the distributed recipe, so that equal sums give equal columns. */
void subhalo_deferred_velocities(float BulkVel[3], float CoreVel[3], float * Vrms, const double cvel[3], double Mb, double Mc,
                                 double Sb[3][2], double S2[3][2], double Sc[3][2], double vkms);

/* Velocity conversion of the catalogue (design 7.10): internal Vel -> physical peculiar km/s,
 * Vel * UnitVelocity_in_cm_per_s / (1e5 * atime). The units come from subhalo_set_units. */
double subhalo_vel_to_kms(double atime);

/* Collective range check before narrowing a 64-bit count to an MPI int: endrun(1) on every rank,
 * naming what, if value < 0 or > INT_MAX on any rank (exchanges are not batched past the limit). */
void subhalo_check_mpi_count(int64_t value, const char * what, MPI_Comm Comm);

/* Keyed MPI_Alltoallv of fixed-size elements, sendcounts[r] for rank r in rank order (64-bit counts,
 * checked against the MPI int limit before narrowing). *recvbuf is allocated on the main stack (top
 * if top != 0) and freed by the caller; recvcounts (NTask) if not NULL. Returns the received count.
 * Collective. */
int64_t subhalo_alltoallv(const void * sendbuf, const int64_t * sendcounts, size_t elsize, void ** recvbuf, int top, int64_t * recvcounts, MPI_Comm Comm);

/* Bytes one batch of subhalo_table_fetch / subhalo_table_store may hold on any rank (its
 * request and reply buffers), whatever the request pattern. */
double subhalo_table_batch_peak_bytes(int NTask);

/* The FOF keys g = GrNr live in [1, TotNgroups] (fof_assign_grnr numbers them from 1).
 * Their directory (the plan, the newborn numbering) is a static partition: key g on rank
 * (g - 1) / KChunk, KChunk = max(1, ceil(TotNgroups / NTask)). */
static inline int64_t subhalo_key_chunk(int64_t TotNgroups, int NTask) {
    int64_t c = (TotNgroups + NTask - 1) / NTask;
    return c > 0 ? c : 1;
}
static inline int subhalo_key_dir_rank(int64_t g, int64_t KChunk) {
    return (int) ((g - 1) / KChunk);
}

/* Step 13: history columns of a record resolved this call (design 7.7). MaxMergerMbound is, for
 * a central, the largest previous Mbound among the tracks that sank into it or became its
 * satellites this call (0 if none). Rank > 0 marks a satellite, Rank 0 a central, a hostless
 * track (HostGroup -1) counts as a central. */
void subhalo_history_update(struct SubTrack * rec, int SnapNum, double MaxMergerMbound);

/* A cleared record of TrackId that takes over the identity, history and orphan-tracer columns
 * and the transient NboundPrev, Host and Key of old (NULL: a newborn, SnapBirth = SnapNum). */
void subhalo_track_carry(struct SubTrack * rec, const struct SubTrack * old, int64_t TrackId, int SnapNum);

/* Set the orphan state of a record that sank into winner (SinkTrackId) or died (winner = -1). */
void subhalo_track_make_orphan(struct SubTrack * t, int64_t winner, int SnapNum);

/* The tracks no key processed this call, on their home ranks: one without live particles dies (not
 * sunk); one with own-tag field particles gets the hostless rule (all bound, SUBHALO_FLAG_INPLACE,
 * properties cleared), which only the unit test reaches: every pseudo-key is processed.
 * Returns the global counts. Collective. */
void subhalo_update_unhosted(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nhostless, int64_t * ndead, MPI_Comm Comm);

/* subhalo_read_state without the abort on a validation failure (duplicate or missing TrackIds,
 * out-of-range ids, inconsistent Subhalo/Orphan classification): returns 1 (collectively) and
 * leaves the table in an unspecified state; IO failures are still fatal. 0 on success, with the
 * state time in *StateTime. Collective. Used by subhalo_read_state and the unit tests. */
int subhalo_read_state_checked(const char * pigname, struct SubTrackTable * tab, double * StateTime, MPI_Comm Comm);

/* ------------------------------------------------------------------ plan (subhalo_gather.c) */

/* Plan entry for a key present on this rank before the exchange. */
struct SubhaloLocalKey {
    int64_t g;
    int64_t Length;         /* global length (fof->Group[].Length) */
    int target;             /* processing rank of a serial key; ThisTask for an oversized key */
    int oversized;          /* 1: processed in place */
    int decider;            /* lowest rank holding particles of the key */
    int pad;
    double ref[3];          /* FirstPos of the group, translated frame */
};

/* An oversized key processed in place (FOF group or pseudo-key), global, sorted by g. Length: its
 * FOF particles left after the sub-tree cut (informative). cen, vel: mass-weighted centre and mean Vel
 * of its FOF particles (a pseudo-key's all), the frame of a candidate without a previous-bound one. */
struct SubhaloInplaceKey {
    int64_t g;
    int64_t Length;
    int decider;            /* the owner of the key's candidate records: lowest rank holding its particles */
    int central;            /* index of its depth-0 candidate in plan->Cands */
    double ref[3];
    double cen[3], vel[3];
    double mass;
    int64_t NMembers;       /* member tracks of the key in plan->Members */
};

/* ---- oversized groups, design 7.8 (plan in subhalo_gather.c, rounds in subhalo_collective.c) ---- */

enum { SUBHALO_MEM_RESOLVED = 0, SUBHALO_MEM_ORPHAN = 1, SUBHALO_MEM_NEWBORN = 2 };
/* Processing states, shared by the serial key nodes (subhalo_track.c) and the in-place
 * candidates: NONE not unbound (an orphan that stays one), FAILED an orphan central that was not
 * resurrected or a newborn that did not survive. */
enum { SUBHALO_ST_NONE = 0, SUBHALO_ST_ALIVE = 1, SUBHALO_ST_DIED = 2, SUBHALO_ST_SUNK = 3, SUBHALO_ST_FAILED = 4 };

/* A member track of an oversized key (the design 7.8 member table), identical on every rank, sorted
 * by TrackId: resolved tracks keyed to it, orphans hosted by it, or a newborn central. Parent spans
 * the serial nest tree of the whole group (central decided as the serial key does); a resolved member
 * is an in-place candidate (cand >= 0) or in a serial sub-tree key (SubKey >= 0). */
struct SubhaloMember {
    int64_t TrackId;
    int64_t g;
    int64_t PrevParent;      /* NestedParentTrackId of the previous record */
    int64_t PrevCentral;     /* CentralTrackId of the previous record */
    int64_t SubKey;          /* serial sub-tree key holding it, -1 */
    int32_t Parent;          /* member index of the processing parent, -1 */
    int32_t kind;            /* SUBHALO_MEM_* */
    int32_t cand;            /* in-place candidate index, -1 */
    int32_t depth;           /* depth in the processing tree (the central 0) */
    int32_t root;            /* 1: the root of SubKey (its processing parent is an in-place candidate) */
    int32_t PrevRank;
    int32_t NboundPrev;
    int32_t SnapTracerLost;
    float MboundPrev, VmaxPrev;
    int64_t NPrev;
    double RefPos[3], RefVel[3];         /* previous-bound frame (translated frame, internal Vel) */
    double TracerPos[3], TracerVel[3];   /* an orphan's tracer as in the record (output frame, km/s) */
    /* this call: the serial sub-tree results (gathered after the serial keys) and the rounds */
    int32_t state;           /* SUBHALO_ST_* */
    int32_t Nbound;
    int32_t NboundExclusive;
    int32_t Keep;            /* a sub-tree root: own-tag remainder particles inside its relaxed source */
    int32_t has_core;
    int32_t frank, fdepth;   /* the group's final hierarchy (survivors), -1 otherwise */
    int32_t resrank;         /* a sub-tree member: the rank holding its serial result (gather_serial), -1 */
    float Mbound, Vmax;
    MyIDType MostBoundID;
    int64_t SinkTrackId;     /* the member it sank into, -1 */
    int64_t FinalParent;     /* NestedParentTrackId after the group's hierarchy (nearest surviving ancestor) */
    int64_t Central;         /* the group's central after the hierarchy, -1 if nothing survived */
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;   /* the sinking core (translated frame, Vel units) */
};

/* An in-place candidate, identical on every rank, sorted by (round, g, TrackId): the rounds are
 * contiguous ranges plan->RoundStart[r] .. RoundStart[r + 1], and the dense label of a candidate
 * in its round is its index minus RoundStart[round]. */
struct SubhaloInplaceCand {
    int64_t TrackId;
    int64_t g;               /* its oversized key */
    int32_t member;          /* index in plan->Members */
    int32_t parent;          /* the parent candidate, -1 for a depth-0 candidate */
    int32_t depth, round;
    int32_t owner;           /* rank that writes its record (its key's decider) */
    int32_t gidx;            /* its key in plan->Inplace */
    double RefPos[3], RefVel[3];   /* initial frame (translated, Vel units) */
};

/* A serial sub-tree key: keys g in [SubKeyBase, SubKeyBase + NSubKeys). */
struct SubhaloSubKey {
    int64_t g;               /* its oversized group */
    int64_t gidx;            /* the group in plan->Inplace (final plan) */
    int64_t root;            /* member index of its root */
};

/* A serial key assigned to this rank. */
struct SubhaloAssignedKey {
    int64_t g;
    int64_t Length;
};

/* Diagnostics of the packing, identical on every rank. */
struct SubhaloPlanStats {
    int64_t LargestSerial;
    int64_t NSerial;
    int64_t NOversized;
    int64_t NRepaired;      /* serial keys moved in place by the post-packing census */
    double MinBudget;       /* min_r B_r (particles) */
    double MinByteBudget;   /* min_r of the byte budget left for the received keys */
    double MinFreeBytes;
    int Iterations;         /* classification and packing passes */
    int CensusRounds;       /* census rounds that moved keys in place */
    int Fallback;           /* 1 if the pass cap was reached (the census then settles the plan) */
};

enum { SUBHALO_COLL_SETUP = 0, SUBHALO_COLL_TREE, SUBHALO_COLL_CORE, SUBHALO_COLL_RADIAL, SUBHALO_COLL_PROPS,
       SUBHALO_COLL_SOURCE, SUBHALO_COLL_TAGS, SUBHALO_COLL_REPAIR, SUBHALO_COLL_NSTAGE };
extern const char * const SubhaloCollStageName[SUBHALO_COLL_NSTAGE];
struct SubhaloCollAdmit {
    int checked;             /* 1: the plan has in-place candidates and was checked */
    int failed;              /* 1: some rank cannot fit the collective stage */
    int failrank, failstage; /* the rank with the largest shortfall and its largest stage */
    double persist;          /* this rank: scratch held through the whole collective call */
    double stage[SUBHALO_COLL_NSTAGE];   /* this rank: the transient peak of each stage */
    double need;             /* this rank: persist + max(stage) */
    double avail;            /* this rank: free bytes left for the collective call */
    double worst_short;      /* largest need - avail over the ranks (> 0: failed) */
    int64_t NumPart, NUniv, NSubPart;    /* this rank's inputs: particles, particles of oversized groups, of its sub-tree keys */
    /* the reserve admission of every plan: the free bytes at pack time against cfg.ReserveBytes +
     * cfg.SortBytesPerPart * MaxPart */
    int reserve_failed;      /* 1: some rank is short of the reserve */
    int reserve_rank;        /* the rank with the largest shortfall */
    double reserve_short;    /* its shortfall (bytes; <= 0 when the check passed) */
    double reserve_need;     /* this rank: the reserve */
    double pack_free;        /* this rank: the free bytes at pack time */
};
struct SubhaloPlan {
    int NTask, ThisTask;
    /* Keys: FOF groups [1, TotNgroups], pseudo-keys of hostless resolved tracks up to TotKeys0, serial
     * sub-tree keys [SubKeyBase, TotKeys]; KChunk partitions [1, TotKeys] for the directory. */
    int64_t TotNgroups, TotKeys, KChunk;
    int64_t TotKeys0, SubKeyBase;
    /* on the top of the main stack before the four below, in this order, identical on every rank */
    struct SubhaloMember * Members;          /* sorted by TrackId */
    int64_t NMembers;
    struct SubhaloInplaceCand * Cands;       /* sorted by (round, g, TrackId) */
    int64_t NCands;
    int64_t * RoundStart;                    /* [NRounds + 1] */
    int NRounds;
    int SplitPasses;                         /* census passes that cut sub-tree keys */
    struct SubhaloSubKey * SubKeys;          /* [NSubKeys] */
    int64_t NSubKeys;
    /* all four on the top of the main stack, allocated in this order */
    struct SubhaloLocalKey * LocalKeys;     /* sorted by g */
    int64_t NLocalKeys;
    struct SubhaloAssignedKey * Assigned;   /* sorted by g */
    int64_t NAssigned;
    struct SubhaloInplaceKey * Inplace;     /* sorted by g, global */
    int64_t NInplace;
    int64_t * OrphanIds;                    /* orphans hosted by a serial key processed here, sorted */
    int64_t NOrphanIds;
    struct SubhaloPlanStats Stats;
    struct SubhaloCollAdmit Admit;          /* the admission of the collective stage */
    double PackFreeBytes;                   /* this rank: mymalloc_freebytes() at the last packing */
    int64_t ExchOut, ExchIn;                /* this rank: live particles sent and received by subhalo_plan_exchange */
};

/* ---- the packing as a particle-free function (unit-testable with synthetic inputs) ---- */

/* A census message to the key directory: the particles of key g held by one rank (kind 0), the FOF
 * group record of g (kind 1), or an orphan hosted by g from its home rank (kind 2, track = TrackId). */
struct SubhaloDirMsg {
    int64_t g;
    int64_t cnt[6];
    int64_t ntags;          /* kind 0: distinct TrackIds >= 0 among this rank's particles of g */
    double ref[3];
    int32_t rank;
    int32_t kind;
    int64_t nfield;         /* kind 0: of cnt, field particles (GrNr < 0) keyed to g by their track */
    int64_t track;          /* kind 2 */
};

/* A key on its directory rank (keys g in [ThisTask * KChunk + 1, ...], ascending). */
struct SubhaloDirKey {
    int64_t g;
    int64_t Length;
    int64_t LenType[6];
    int64_t Tags;           /* upper bound on the distinct member tracks: the sum of the ntags */
    double ref[3];
    int64_t cstart, ccount; /* census contributions in the directory's message array, by rank */
    int64_t NField;         /* field particles in Length */
    int64_t ostart, ocount; /* kind-2 (hosted orphan) messages in the message array */
    int target;             /* processing rank of a serial key, -1 if none */
    int oversized;          /* 1: processed in place */
    int decider;            /* lowest rank holding particles of the key */
    int has_group;
    int forced;             /* in place whatever its size (a split group, an earlier pass's in-place key) */
    int pad;
};

/* This rank's resources at plan time (inputs of the budget, design 7.3). */
struct SubhaloPlanRes {
    int64_t MaxPart;
    int64_t NumLive;        /* non-garbage particles on the rank */
    int64_t grp[7];         /* of which in FOF groups, by type; [6] the total */
    int64_t slotfree[6];    /* maxsize - size of an enabled slot type; ignored if elsize is 0 */
    int64_t slotelsize[6];  /* 0 for a disabled slot type */
    double freebytes;       /* mymalloc_freebytes() */
};

/* Configuration of the packing; the byte costs are those of subhalo_plan_default_cfg. */
struct SubhaloPlanCfg {
    double GatherFraction;      /* SubhaloMaxGatherFraction */
    int64_t CollectiveMinPart;  /* 0 = off */
    int ForceRankZero;          /* SubhaloDebugForceExchangeFailure: every serial key to rank 0, unchecked */
    int64_t TotNgroups, KChunk;
    int MaxIter;                /* classification passes before the conservative fallback */
    /* byte costs (see subhalo_plan_default_cfg for what each one covers) */
    double PartBytes;           /* per particle of a received key */
    double MemberBytes;         /* per member track of a key */
    double KeyBytes;            /* per key processed on the rank (serial or local part in place) */
    double InplaceKeyBytes;     /* per in-place key, on every rank (global vectors) */
    double InplaceCandBytes;    /* per in-place candidate, on every rank */
    double InplaceMemberBytes;  /* per member track of the in-place keys, on every rank */
    /* in-place candidates and members the plan pass knows; a key made oversized while packing adds
     * one candidate and its census tags */
    int64_t NCandsKnown, NMembersKnown;
    int64_t SubKeyBase;         /* first serial sub-tree key (INT64_MAX: none) */
    double SubKeyPartBytes;     /* per particle of a sub-tree key, kept to the final tag pass (struct SubhaloBindOrd) */
    double ExchPartBytes;       /* per departing particle, exchange buffers (capped at one batch) */
    double SortBytesPerPart;    /* per MaxPart, the local sort and the restore's per-particle lists */
    double ReserveBytes;        /* the constant terms */
};

/* Fill cfg from the parameters and the communicator size. */
void subhalo_plan_default_cfg(struct SubhaloPlanCfg * cfg, int64_t TotNgroups, int NTask);

/* Classification and packing (design 7.3): passes of budgets, head greedy and proportional tail until
 * none moves a key in place (at most MaxIter), then census rounds moving keys of over-budget
 * destinations in place until all fit (ForceRankZero unchecked). dk: this rank's directory slice; in:
 * its census messages sorted by (g, kind, rank). Particle-free (res describes the rank). Collective. */
void subhalo_plan_pack(struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloDirMsg * in,
                       const struct SubhaloPlanRes * res, const struct SubhaloPlanCfg * cfg,
                       struct SubhaloPlanStats * stats, MPI_Comm Comm);
/* The bytes a serial key costs its processing rank (arena, member records, per-key buffers). */
static inline double subhalo_plan_key_bytes(const struct SubhaloPlanCfg * cfg, int64_t Length, int64_t Tags) {
    return cfg->PartBytes * (double) Length + cfg->MemberBytes * (double) Tags + cfg->KeyBytes;
}

/* A contiguous run of one key's particles in P after the local sort. */
struct SubhaloKeySeg {
    int64_t g;
    int64_t start, n;       /* particles P[start .. start + n) */
    int64_t Length;         /* global length of the key */
    int inplace;            /* 1 for an oversized key (n may be < Length) */
    int decider;            /* in place: lowest holding rank */
    int64_t ipindex;        /* in place: index into plan->Inplace, else -1 */
    double ref[3];          /* in place: common reference position */
    int64_t Tracks;         /* serial: the distinct tracks tagging its particles (Phase 5d cost model) */
};

/* Steps 0 and 2: tracer snapshot and HBT's weighted host vote. Every rank sends, per track T,
 * the (BindRank, ID, GrNr) of the SubhaloNumTracerHostFinding lowest-BindRank live tracer-type
 * particles it holds with TrackId == BoundTrackId == T to T's home rank, which keeps the first
 * SubhaloNumTracerHostFinding by (BindRank, ID) as TracerIDs and sets Host to the GrNr (-1: no
 * group) with the largest weight sum exp(-TracerPenalty) / (1 + sqrt(i)), ties to the smaller
 * GrNr. Collective. */
void subhalo_host_vote(struct SubTrackTable * tab, MPI_Comm Comm);
/* A tracer's vote weight exp(-TracerPenalty); exactly 1 for a zero (or negative) penalty, so
 * runs without a scattering module vote bit for bit as before. */
static inline double
subhalo_tracer_weight(float penalty)
{
    return penalty > 0 ? exp(-(double) penalty) : 1.0;
}
/* Weighted plurality of n ballots: the candidate >= 0 with the largest summed weight (sums in
 * ballot order), ties to the smaller; SUBHALO_NOTRACK if there is none. top and second (NULL
 * allowed) get the winning and the runner-up sums (0 if absent). n <= SUBHALO_MAXTRACERS. */
int64_t subhalo_weighted_plurality(const int64_t * cand, const double * w, int n, double * top, double * second);
/* Step 3, previous-bound frames: sums over the live particles with BoundTrackId == T, as
 * nearest-image displacements from one common reference per track (the lowest (own tag, BindRank,
 * ID)), reduced on the home rank in rank order into RefPos, RefVel, NPrev, MPrev. Collective. */
void subhalo_previous_frames(struct SubTrackTable * tab, MPI_Comm Comm);
/* Step 3, keys of the tracks on the home ranks: Key = Host for a hosted resolved track, the
 * pseudo-key TotNgroups + 1 + ordinal for a hostless resolved track (ordinals in TrackId order
 * by an exclusive scan), -1 otherwise. Returns the global number of pseudo-keys. Collective. */
int64_t subhalo_assign_keys(struct SubTrackTable * tab, int64_t TotNgroups, MPI_Comm Comm);
/* Step 3, the plan: census, budgets and packing of the FOF, pseudo- and sub-tree keys; hosted orphans
 * routed to their serial key's rank (OrphanIds). Allocates the tables on the top, sets P[i].TopLeaf
 * to each particle's target and encodes a travelling particle's key in GrNr (undone by
 * subhalo_restore_field_grnr). ddecomp (NULL in tests) sizes the admission's tree. Returns 0, or 1 on
 * every rank if the admission fails (P untouched; the caller frees the plan). Collective. */
int subhalo_plan_build(struct SubhaloPlan * plan, const FOFGroups * fof, struct SubTrackTable * tab, int64_t NHostless,
                       const DomainDecomp * ddecomp, MPI_Comm Comm);
/* The kinematic constants of the plan-step central decision (the serial key's kinetic distance):
 * set by the driver before subhalo_plan_build (a, H(a) in internal units). */
void subhalo_plan_set_frame(double atime, double Hubble);

/* Admission estimate of subhalo_collective_run (particle-free; subhalo_collective.c), made before the
 * exchange: this rank's persistent bytes and per-stage peaks from its particle counts (NUniv in
 * oversized groups, NSubPart in its sub-tree keys), owned candidates and their core records
 * (OwnedRecv), the global counts and the top tree; fills a->persist, stage[] and need. */
void subhalo_coll_estimate(struct SubhaloCollAdmit * a, int64_t NumPart, int64_t MaxPart, int64_t NUniv, int64_t NSubPart,
                           int64_t NOwned, int64_t OwnedRecv, int64_t NCands, int64_t NMembers, int64_t NSubResults,
                           int64_t NTopNodes, int64_t NTopLeaves, int64_t NLocalLeaves, double TreeAllocFactor, int nthreads);
/* The admission of the last subhalo_plan_build (this rank; the failure fields are global). */
const struct SubhaloCollAdmit * subhalo_plan_last_admission(void);
/* What the last subhalo_plan_build made (global; tests and logs). */
struct SubhaloPlanSummary {
    int64_t NSerial, NOversized;     /* keys packed serially, in place */
    int64_t NSubKeys, NCands;        /* serial sub-tree keys, in-place candidates */
    int64_t NEncFof, NEncField;      /* group particles of sub-tree keys and field particles that travel encoded */
    int NRounds;
    int Admitted;                    /* 0: the collective admission failed (nothing encoded) */
};
const struct SubhaloPlanSummary * subhalo_plan_last_summary(void);
/* Binary search of the member table, NULL if absent. */
const struct SubhaloMember * subhalo_plan_member(const struct SubhaloPlan * plan, int64_t TrackId);
/* The in-place key g in plan->Inplace, -1 if g is not in place. */
int64_t subhalo_plan_inplace_index(const struct SubhaloPlan * plan, int64_t g);
/* The processing key of a particle after subhalo_plan_build: GrNr for a group particle, the
 * encoded key of a travelling field particle, -1 otherwise. */
#define SUBHALO_KEY_FOFBIT (1LL << 52)
static inline int64_t subhalo_particle_key(const struct particle_data * p) {
    if(p->GrNr >= 0) return p->GrNr;
    if(p->GrNr < -1) return (-(p->GrNr + 2)) & (SUBHALO_KEY_FOFBIT - 1);
    return -1;
}
/* The processing key of a track (its record's Key) for the tracks tagging this rank's live
 * field particles: (TrackId, Key) pairs sorted by TrackId. */
struct SubhaloTrackKey {
    int64_t TrackId;
    int64_t Key;
};
/* GrNr = -1 again for every field particle and the group number again for the group particles
 * of the serial sub-tree keys (after the serial keys, and on the failure path). */
void subhalo_restore_field_grnr(const struct SubhaloPlan * plan);
/* Step 4: the exchange. Returns this rank's failure flag (not reduced). Collective. */
int subhalo_plan_exchange(struct SubhaloPlan * plan, MPI_Comm Comm);
/* Frees what subhalo_plan_build allocated (reverse order). */
void subhalo_plan_free(struct SubhaloPlan * plan);
/* Local sort of P by (garbage last, processing key, TrackId, BindRank, ID). */
void subhalo_local_sort(void);
/* An order-preserving threaded selection over P[0 .. N): COUNT_CHUNKS fills off[c + 1]
 * with the number of selected particles of chunk c and turns off into offsets, FILL_CHUNKS runs
 * BODY(i, m) for the m-th selected particle i in P order. The result does not depend on the
 * thread count. */
#define SUBHALO_COUNT_CHUNKS(N, nc, off, PRED) do { \
        int c_; \
        (off)[0] = 0; \
        _Pragma("omp parallel for") \
        for(c_ = 0; c_ < (nc); c_++) { \
            int64_t cnt_ = 0, i; \
            for(i = c_ * (int64_t) (N) / (nc); i < (c_ + 1) * (int64_t) (N) / (nc); i++) \
                if(PRED) cnt_++; \
            (off)[c_ + 1] = cnt_; \
        } \
        for(c_ = 0; c_ < (nc); c_++) \
            (off)[c_ + 1] += (off)[c_]; \
    } while(0)
#define SUBHALO_FILL_CHUNKS(N, nc, off, PRED, BODY) do { \
        int c_; \
        _Pragma("omp parallel for") \
        for(c_ = 0; c_ < (nc); c_++) { \
            int64_t m = (off)[c_], i; \
            for(i = c_ * (int64_t) (N) / (nc); i < (c_ + 1) * (int64_t) (N) / (nc); i++) \
                if(PRED) { BODY; m++; } \
        } \
    } while(0)
/* Test knob: 0 the record sort when it fits (default), 1 always the indirect sort of P. */
void subhalo_local_sort_set_mode(int mode);
/* Chunks of an order-preserving threaded pass over n items (about four per thread, at least 4096
 * items each); the results of such passes do not depend on it. */
int subhalo_par_nchunks(int64_t n);
/* The key runs on this rank after the sort: serial keys assigned here and the local part of
 * oversized keys. Allocates *segs on the top of the stack. endrun on a broken plan. */
int64_t subhalo_build_segments(const struct SubhaloPlan * plan, struct SubhaloKeySeg ** segs);
/* Step 7a: distinct TrackIds of the particles in the segments and the extra ids (the routed
 * orphans), sorted, and copies of their records (with the transient fields). Allocates
 * *members on the top of the stack. Collective. */
int64_t subhalo_fetch_members(const struct SubhaloKeySeg * segs, int64_t nsegs, const int64_t * extra, int64_t nextra,
                              struct SubTrackTable * tab, struct SubTrack ** members, MPI_Comm Comm);
/* Step 9: slots_gc_sorted, domain_maintain (endrun on failure), slots_gc_sorted. */
void subhalo_restore_domain(DomainDecomp * ddecomp);

/* Per-thread scratch arena (design 7.1). */
struct SubhaloArena {
    char * base;
    size_t size;
    size_t used;
    size_t peak;            /* high-water mark of used (diagnostic; the driver zeroes it) */
};
static inline void * subhalo_arena_get(struct SubhaloArena * a, size_t bytes) {
    /* keep 16-byte alignment */
    bytes = (bytes + 15) & ~((size_t) 15);
    if(a->used + bytes > a->size)
        return NULL;
    void * p = a->base + a->used;
    a->used += bytes;
    if(a->used > a->peak)
        a->peak = a->used;
    return p;
}

/* Callback processing serial key k with one arena; nthreads > 1 only for a key processed alone. */
typedef void (*subhalo_key_fn)(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata);
/* Static cost-ordered schedule of the serial segments: keys costing more than total / nthreads run
 * alone first with every thread inside, the rest go longest-first to the least loaded thread (one
 * arena each). fn() sees exactly its key's allowance, bytes_per_part * n + extra[k] + 64 (extra may be
 * NULL), so results do not depend on the thread sharing. Returns the largest arena high-water fraction. */
double subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, const int64_t * extra,
                               subhalo_key_fn fn, void * userdata);
/* This rank's measured key times, summed over the subhalo_run_serial_keys calls since the last
 * reset (the driver resets before the serial stage). */
struct SubhaloSerialStats {
    int64_t NKeys;            /* keys processed */
    int64_t NAlone;           /* of which alone (every thread inside the key) */
    int64_t LargestN;         /* particles of the largest key */
    double KeyTimeMax;        /* the longest measured key time (s) */
    int64_t KeyTimeMaxN;      /* its particles */
    double AloneWall;         /* wall time of the keys processed alone (s) */
    double ThreadedWall;      /* wall time of the threaded part (s) */
    /* the threaded part, per actual worker (a team smaller than the queues runs several per worker) */
    double ThreadMax;         /* sum over calls of the largest per-worker summed key time */
    double ThreadSum;         /* sum over calls and workers of the per-worker summed key times */
    double ThreadMeanSum;     /* sum over calls of the mean per-worker summed key time */
    int NThreads;             /* workers of the team of the last call */
    int NQueues;              /* queues of the last call (omp_get_max_threads()) */
    double KeyTimeSum;        /* summed key times (alone keys counted once, at their wall time) */
    /* least-squares sums of the single-threaded key times t against x = N log2 max(N, 2) (the fit
     * t = a x + b calibrates SUBHALO_COST_KEY / SUBHALO_COST_ITER = b / a) */
    double FitN, FitX, FitXX, FitY, FitXY;
};
const struct SubhaloSerialStats * subhalo_serial_last_stats(void);
void subhalo_serial_reset_stats(void);

/* ------------------------------------------------------------------ batched ID lookup (subhalo.c) */

/* What the rank holding a live particle reports about it (steps 1 and 11). */
struct SubhaloIdHit {
    int64_t GrNr;
    int64_t BoundTrackId;
    double Pos[3];           /* translated comoving frame */
    double Vel[3];           /* internal Vel */
    int found;
    float Penalty;           /* its TracerPenalty */
};
/* For ids[0..n) (any rank, duplicates allowed) find the live particle with that exact ID on any
 * rank: a per-rank sorted (ID, index) index, the ids gathered in batches bounded by
 * SubhaloLookupBatchBytes and searched by every rank, hits returned to the asking rank.
 * out[i].found = 0 for a miss. Collective. */
void subhalo_lookup_ids(const MyIDType * ids, int64_t n, struct SubhaloIdHit * out, MPI_Comm Comm);
/* Bytes the batch buffers of one subhalo_lookup_ids batch can hold on any rank (gathered ids,
 * outgoing and returning replies): at most SubhaloLookupBatchBytes, or the one-id-per-rank
 * floor NTask * (8 + 80) + 80 bytes when that is larger. */
double subhalo_lookup_batch_peak_bytes(int NTask);
/* Bytes of the sorted (ID, index) index and its sort scratch for NumPart particles. */
double subhalo_lookup_index_bytes(int64_t NumPart);
/* Measured high-water mark of the batch buffers of the last lookup on this rank (tests). */
double subhalo_lookup_last_peak_bytes(void);
/* Test knob: at most maxbatch ids per rank and batch (0 restores the byte cap alone). */
void subhalo_lookup_set_max_batch(int64_t maxbatch);
/* Test knob: 0 auto (scan when all queries of the call are at most 1/16 of the rank's
 * live particles, else the sorted index), 1 always the index, 2 always the scan. */
void subhalo_lookup_set_mode(int mode);
/* Step 1: every orphan record homed here (Nbound == 0, SnapTracerLost < 0) looks up its
 * MostBoundID: a hit sets Host to the particle's GrNr (-1 outside groups) and TracerPos
 * (physical frame) / TracerVel (km/s), a miss sets SnapTracerLost = SnapNum. Returns the
 * global (looked up, found) counts. Collective. */
void subhalo_orphan_lookup(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nlook, int64_t * nfound, MPI_Comm Comm);

/* ------------------------------------------------------------------ per-key processing (subhalo_track.c) */

/* One record emitted by the key processing (one per member hosted by a serial key, resolved or
 * orphan, at most one newborn per key, one winner per in-place key on its deciding rank). */
struct SubhaloResult {
    struct SubTrack rec;     /* the new record, history not yet updated */
    double MaxMergerMbound;  /* central: largest previous Mbound of the tracks that sank into it or arrived as its satellites */
    int valid;               /* 1 if rec is to be stored */
    int newborn;             /* 1 if rec.TrackId is provisional */
    int history;             /* 1: resolved this call, apply subhalo_history_update */
    int ndesc;               /* descendant votes (a track resolved at the previous call that died or sank) */
    int64_t DescOwner[SUBHALO_MAXTRACERS];  /* per tracer: its BoundTrackId, -1 if unbound or lost, SUBHALO_DESC_PENDING still to be looked up (step 11) */
    float DescWeight[SUBHALO_MAXTRACERS];   /* and its vote weight exp(-TracerPenalty), set with a resolved DescOwner */
    int64_t ipk;             /* the central of an oversized key (in place or a sub-tree survivor promoted to
                              * Rank 0): index of its key in plan->Inplace, else -1 */
    /* the node's state for the rounds (serial sub-tree keys) */
    int64_t subkey;          /* the sub-tree key it was processed in, -1 */
    int32_t state;           /* SUBHALO_ST_* */
    int32_t Keep;            /* a sub-tree root: own-tag remainder particles inside its relaxed source */
    int32_t has_core;
    int32_t pad;
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;   /* the sinking core (translated frame, Vel units) */
    int64_t SinkTo;          /* the node it sank into in the key (TrackId), -1 */
};

/* The stages of one call, timed by the driver (wall time and allocator high water of each);
 * SubhaloStageName in subhalo.c. */
enum { SUBHALO_STAGE_SETUP = 0, SUBHALO_STAGE_VOTE, SUBHALO_STAGE_FRAMES, SUBHALO_STAGE_LOOKUP, SUBHALO_STAGE_PLAN,
       SUBHALO_STAGE_EXCHANGE, SUBHALO_STAGE_SORT, SUBHALO_STAGE_FETCH, SUBHALO_STAGE_SERIAL, SUBHALO_STAGE_RESTORE,
       SUBHALO_STAGE_ROUNDS, SUBHALO_STAGE_NUMBER, SUBHALO_STAGE_DESCENDANTS, SUBHALO_STAGE_HISTORY, SUBHALO_STAGE_STORE,
       SUBHALO_STAGE_UNHOSTED, SUBHALO_STAGE_ORPHANSYNC, SUBHALO_STAGE_FIELDSO, SUBHALO_NSTAGE };
extern const char * const SubhaloStageName[SUBHALO_NSTAGE];

/* Diagnostics of one call (local during the call, global totals after subhalo_find). */
struct SubhaloCallStats {
    int64_t NCandidates;     /* candidates unbound in serial keys (each unbinding counted once) */
    int64_t NItercap;        /* of which stopped at SubhaloMaxUnbindIterations (SUBHALO_FLAG_ITERCAP) */
    double WorstConvRatio;   /* smallest Nbound / Nlast of a capped candidate (1 if none) */
    int64_t WorstTrackId;    /* its TrackId (-1 if none; provisional for a newborn) */
    int64_t NFinalPositive;  /* final-pass bound particles with E >= 0, summed over the survivors */
    int64_t NFinalBound;     /* bound particles of the final passes */
    int64_t NTailRejected;   /* tail claims crossing the binding owner's final branch */
    int64_t NFallbackField;  /* particles of the oversized keys whose tags the final pass of the rounds wrote */
    int64_t NFallbackReleased; /* field particles of the oversized keys untagged instead (stale orphan tags, the central's); not in NFallbackField */
    double ArenaPeakFrac;    /* largest arena high-water mark of a key / its arena size */
    /* ---- instrumentation (global, set on every return path; a failed call keeps the wall times
     * and peaks of the stages that ran, zero counters and Complete 0) ---- */
    int Complete;            /* 1: the call completed */
    int PeakRank, PeakStage; /* where the used high water was reached (-1 if not measured) */
    double WallTime;         /* wall time of the call, max over the ranks (s) */
    double WallTimeMean;     /* mean over the ranks */
    int64_t PeakBytes;       /* main-allocator used high water of the call (bytes), max over the ranks; 0 under use_malloc */
    double StageMax[SUBHALO_NSTAGE];        /* wall time per stage (s), max over the ranks */
    double StageMean[SUBHALO_NSTAGE];       /* mean over the ranks */
    double StagePeakBytes[SUBHALO_NSTAGE];  /* used high water within the stage, max over the ranks */
    /* the finder's own increment per stage: its high water without the field SO's elastic export
     * tables, minus the usage at entry (max over the ranks); the call's is the largest */
    double StageIncrBytes[SUBHALO_NSTAGE];
    int64_t PeakIncrementBytes;
    int PeakIncrementRank, PeakIncrementStage;
    double EntryBytes;                      /* main-allocator usage at the call's entry, max over the ranks */
    double FieldSOElasticBytes;             /* the largest export tables of a field SO walk, max over the ranks (excluded above) */
    double ExchOutMax, ExchOutMean;         /* particles sent by the exchange, max and mean over the ranks */
    double ExchInMax, ExchInMean;           /* received */
    int64_t NSerial, LargestSerial;         /* serial keys of the plan, the largest's particles */
    int64_t NSerialKeys, NAloneKeys;        /* keys run by subhalo_run_serial_keys (sum over the ranks), of which alone */
    double KeyTimeMax;                      /* longest measured key time (s) */
    int64_t KeyTimeMaxN;                    /* its particles */
    double ThreadImbalance;                 /* largest over the ranks of max / mean of the per-thread summed key times */
    double ThreadImbalanceMean;             /* mean over the ranks (ranks with threaded keys) */
    double RankImbalance;                   /* max / mean over the ranks of the serial key time (alone wall + slowest thread) */
    double KeyFitA, KeyFitB;                /* t = A N log2 N + B fitted to the single-threaded key times of every rank */
    int64_t NCands;                         /* in-place candidates */
    int NRounds;                            /* their rounds */
    double TreeBytes;                       /* the admission's tree-stage estimate of the rounds, max over the ranks */
    int FieldSOPass1Walks, FieldSOPass2Walks, FieldSOBins;
    int64_t FieldSOCentrals;
};
/* The statistics of the last completed subhalo_find (global; for the tests and the log). */
const struct SubhaloCallStats * subhalo_last_call_stats(void);

struct SubhaloUnbindParams;

struct SubhaloCallCtx {
    int SnapNum;
    double atime;
    double BoxSize;
    double Offset[3];        /* PartManager->CurrentParticleOffset */
    int TracerTypes;
    int ThisTask;
    int64_t TotNgroups;      /* keys above are pseudo-keys */
    const struct SubhaloPlan * plan;   /* keys above plan->TotKeys0 are serial sub-tree keys */
    int64_t IpResStart;      /* first result slot of the in-place candidates owned by this rank */
    int64_t NIpRes;
    const struct SubhaloUnbindParams * up;
    struct SubTrack * Members;   /* sorted by TrackId, read-only during the key loop */
    int64_t NMembers;
    struct SubhaloResult * Res;  /* [NRes] */
    int64_t NRes;
    int64_t * ResStart;          /* [nsegs]: first result slot of each segment */
    int64_t * Hosted;            /* member indices grouped by segment (serial: the members hosted by the key) */
    int64_t * HostedStart;       /* [nsegs + 1] */
    /* properties */
    double rho[SUBHALO_NSO];     /* subhalo_so_thresholds at atime */
    int MetalReturnOn;           /* All.MetalReturnOn: metal columns of subhalo_props_baryons */
    struct SubhaloCallStats Stats;   /* this rank's diagnostics (threads add atomically) */
    /* the binder ordinals of the sub-tree keys processed here, sorted by ID (top of the stack,
     * from subhalo_process_serial to subhalo_free_bind_ord) */
    struct SubhaloBindOrd * BindOrd;
    int64_t NBindOrd;
};

/* A tail claim in a serial sub-tree key (bound by member B, tagged by another): its place in B's
 * final-pass energy order (f) and B's exclusive particles before it (g), to return it to B if the
 * claim becomes void in the rounds (decision 19). */
struct SubhaloBindOrd {
    MyIDType ID;
    int64_t g;
    int64_t f;
};
/* Frees ctx->BindOrd (after subhalo_collective_run; the collective allocations are gone). */
void subhalo_free_bind_ord(struct SubhaloCallCtx * ctx);

/* Binary search of a member record; NULL if absent. */
struct SubTrack * subhalo_find_member(const struct SubhaloCallCtx * ctx, int64_t TrackId);
/* The members hosted by every serial segment (resolved: Key == g; orphan: Host == g), the
 * result slots of every segment (hosted members + 1 for a serial key, none for an in-place
 * segment), then one slot per in-place candidate owned by this rank (ctx->IpResStart, NIpRes;
 * ctx->plan must be set). Allocates ctx->Hosted, HostedStart, ResStart on the bottom of the stack
 * (freed by subhalo_free_slots) and returns the number of result slots. */
int64_t subhalo_result_slots(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx);
void subhalo_free_slots(struct SubhaloCallCtx * ctx);
/* Steps 7a-7h and 8 for every serial segment (threaded by subhalo_run_serial_keys). A sub-tree key
 * (g > plan->TotKeys0) has its root on top (no central decision, no FeedCentrals, the group's key
 * centre) and provisional tags: the root keeps its whole untruncated list; the result carries Keep,
 * state and core for the rounds, which write the hierarchy; descendant votes go to the remote lookup. */
void subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx);

#endif
