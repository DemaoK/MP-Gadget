#ifndef SUBHALO_INTERNAL_H
#define SUBHALO_INTERNAL_H

/* Private interface shared by subhalo.c, subhalo_gather.c, subhalo_track.c and subhalo_io.c
 * (and the unit tests). Not part of the public contract in subhalo.h.
 *
 * Memory discipline of one finder call (subhalo_find): everything that must survive the
 * particle exchange or the domain restore lives on the TOP of the main stack (mymalloc2),
 * because domain_exchange may grow the slot block with myrealloc, which needs SlotsBase to be
 * the last allocation on the bottom. Bottom allocations (mymalloc) are scratch that is freed
 * within the function that made it. */

#include "subhalo.h"

/* Per-thread scratch per particle of a key (design 7.1). Phase 1 uses 24 bytes of it; the plan
 * reserves the design budget so the byte check does not change when unbinding lands. */
#define SUBHALO_ARENA_BYTES_PER_PART 60
/* Bytes every rank holds per in-place key (global vectors of subhalo_process_inplace and the
 * plan's in-place list; checked by a static assertion in subhalo_track.c). */
#define SUBHALO_INPLACE_KEY_BYTES 512

/* ------------------------------------------------------------------ shared helpers (subhalo.c) */

/* Finite test that survives -ffast-math (which lets the compiler assume isfinite() is true):
 * the exponent bits of an IEEE double. */
int subhalo_isfinite(double x);

/* Velocity conversion of the catalogue (design 7.10): internal Vel -> physical peculiar km/s,
 * Vel * UnitVelocity_in_cm_per_s / (1e5 * atime). The units come from subhalo_set_units. */
double subhalo_vel_to_kms(double atime);

/* Collective range check before narrowing a local 64-bit count to an MPI int: if value < 0 or
 * value > INT_MAX on any rank, every rank aborts with endrun(1) naming what. Phase 1 does not
 * batch exchanges beyond the MPI int limit (except the record fetch and store, which batch by
 * bytes); it fails cleanly instead. */
void subhalo_check_mpi_count(int64_t value, const char * what, MPI_Comm Comm);

/* Keyed MPI_Alltoallv of fixed-size elements. sendbuf holds sendcounts[r] elements for rank r,
 * in rank order (64-bit counts; offsets are computed in 64 bits). Every local count and the
 * local send and receive totals are checked collectively against the MPI int limit before any
 * narrowing (endrun(1) on every rank otherwise). The receive buffer is allocated on the main
 * stack (top if top != 0, else bottom) and returned in *recvbuf; the caller frees it. If
 * recvcounts is not NULL it receives the per-source counts (NTask int64). Returns the number of
 * received elements. Collective. */
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

/* Step 13: history columns of a record that won a key or is hostless this call. MaxSinkMbound
 * is the largest previous Mbound among the tracks that sank into it this call (0 if none). */
void subhalo_history_update(struct SubTrack * rec, int SnapNum, double MaxSinkMbound);

/* Set the orphan state of a record that sank into winner (SinkTrackId) or died (winner = -1). */
void subhalo_track_make_orphan(struct SubTrack * t, int64_t winner, int SnapNum);

/* Step 7f on the home ranks (hostless resolved and dead tracks, H6/H7); returns the global
 * counts. Collective. Exposed for the unit tests. */
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

/* Global list of the oversized keys, identical on every rank, sorted by g. */
struct SubhaloInplaceKey {
    int64_t g;
    int64_t Length;
    int decider;
    int pad;
    double ref[3];
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

struct SubhaloPlan {
    int NTask, ThisTask;
    int64_t TotNgroups, KChunk;
    /* all three on the top of the main stack, allocated in this order */
    struct SubhaloLocalKey * LocalKeys;     /* sorted by g */
    int64_t NLocalKeys;
    struct SubhaloAssignedKey * Assigned;   /* sorted by g */
    int64_t NAssigned;
    struct SubhaloInplaceKey * Inplace;     /* sorted by g, global */
    int64_t NInplace;
    struct SubhaloPlanStats Stats;
};

/* ---- the packing as a particle-free function (unit-testable with synthetic inputs) ---- */

/* A census message to the key directory: the particles of key g held by one rank (kind 0),
 * or the FOF group record of g from the rank that holds it (kind 1). */
struct SubhaloDirMsg {
    int64_t g;
    int64_t cnt[6];
    int64_t ntags;          /* kind 0: distinct TrackIds >= 0 among this rank's particles of g */
    double ref[3];
    int32_t rank;
    int32_t kind;
};

/* A key on its directory rank (keys g in [ThisTask * KChunk + 1, ...], ascending). */
struct SubhaloDirKey {
    int64_t g;
    int64_t Length;
    int64_t LenType[6];
    int64_t Tags;           /* upper bound on the distinct member tracks: the sum of the ntags */
    double ref[3];
    int64_t cstart, ccount; /* census contributions in the directory's message array, by rank */
    int target;             /* processing rank of a serial key, -1 if none */
    int oversized;          /* 1: processed in place */
    int decider;            /* lowest rank holding particles of the key */
    int has_group;
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
    double ExchPartBytes;       /* per departing particle, exchange buffers (capped at one batch) */
    double SortBytesPerPart;    /* per MaxPart, the local sort and the restore's per-particle lists */
    double ReserveBytes;        /* the constant terms */
};

/* Fill cfg from the parameters and the communicator size. */
void subhalo_plan_default_cfg(struct SubhaloPlanCfg * cfg, int64_t TotNgroups, int NTask);

/* Classification and packing (design 7.3): passes of (budgets, keys that fit no rank in place,
 * head greedy + proportional tail, unplaced keys in place) until a pass moves nothing in place
 * (or MaxIter passes), then census rounds that check every destination's assigned particles,
 * per-type particles and bytes against the budgets of the final staying particles and move
 * keys of over-budget destinations in place, smallest first, until the census passes. On
 * return every key of dk has oversized set or a target and the plan fits (except with
 * ForceRankZero, which is deliberately unchecked). dk is this rank's directory slice and in its
 * census messages (sorted by g, then kind, then rank; cstart/ccount index into it). Uses no
 * particle data: res describes the rank. Collective. */
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
};

/* Step 2: host vote. Sets tab->Track[].Host on the home ranks. Collective. */
void subhalo_host_vote(struct SubTrackTable * tab, MPI_Comm Comm);
/* Step 3: plan. Allocates plan->LocalKeys, Assigned, Inplace on the top of the stack and sets
 * P[i].TopLeaf to the target rank of every particle. Collective. */
void subhalo_plan_build(struct SubhaloPlan * plan, const FOFGroups * fof, MPI_Comm Comm);
/* Step 4: the exchange. Returns this rank's failure flag (not reduced). Collective. */
int subhalo_plan_exchange(struct SubhaloPlan * plan, MPI_Comm Comm);
/* Frees what subhalo_plan_build allocated (reverse order). */
void subhalo_plan_free(struct SubhaloPlan * plan);
/* Local sort of P by (garbage last, GrNr, TrackId, BindRank, ID). */
void subhalo_local_sort(void);
/* The key runs on this rank after the sort: serial keys assigned here and the local part of
 * oversized keys. Allocates *segs on the top of the stack. endrun on a broken plan. */
int64_t subhalo_build_segments(const struct SubhaloPlan * plan, struct SubhaloKeySeg ** segs);
/* Step 7a: distinct TrackIds of the particles in the segments, sorted, and copies of their
 * records (with Host). Allocates *members on the top of the stack. Collective. */
int64_t subhalo_fetch_members(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubTrackTable * tab, struct SubTrack ** members, MPI_Comm Comm);
/* Step 9: slots_gc_sorted, domain_maintain (endrun on failure), slots_gc_sorted. */
void subhalo_restore_domain(DomainDecomp * ddecomp);

/* Per-thread scratch arena (design 7.1). */
struct SubhaloArena {
    char * base;
    size_t size;
    size_t used;
};
static inline void * subhalo_arena_get(struct SubhaloArena * a, size_t bytes) {
    /* keep 16-byte alignment */
    bytes = (bytes + 15) & ~((size_t) 15);
    if(a->used + bytes > a->size)
        return NULL;
    void * p = a->base + a->used;
    a->used += bytes;
    return p;
}

/* Callback processing serial key k with one arena; nthreads > 1 only for a key processed alone. */
typedef void (*subhalo_key_fn)(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata);
/* Static longest-first schedule of the serial segments over the OpenMP threads, one arena per
 * thread (mymalloc2 in thread order, freed in reverse); keys above one thread's share are
 * processed alone first. bytes_per_part sizes the arenas. */
void subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, subhalo_key_fn fn, void * userdata);

/* ------------------------------------------------------------------ per-key processing (subhalo_track.c) */

/* Result of one key on the rank that emits the winner's record. */
struct SubhaloKeyResult {
    struct SubTrack rec;     /* the winner's new record, history not yet updated */
    double MaxSinkMbound;    /* largest previous Mbound of the members that sank into it */
    int valid;               /* 1 if this rank emits rec */
    int newborn;             /* 1 if rec.TrackId is provisional */
};

struct SubhaloCallCtx {
    int SnapNum;
    double atime;
    double BoxSize;
    double Offset[3];        /* PartManager->CurrentParticleOffset */
    int TracerTypes;
    int ThisTask;
    struct SubTrack * Members;   /* sorted by TrackId, read-only during the key loop */
    int64_t NMembers;
    int64_t * SinkWinner;        /* [NMembers]: winner a member sinks into, -1 if none */
    struct SubhaloKeyResult * Res;   /* [nsegs] */
};

/* Binary search of a member record; NULL if absent. */
struct SubTrack * subhalo_find_member(const struct SubhaloCallCtx * ctx, int64_t TrackId);
/* Steps 7a-7e for every serial segment (OpenMP through subhalo_run_serial_keys). */
void subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx);
/* Step 9 in-place variant for every oversized key. Collective (every rank calls it, with or
 * without in-place segments). */
void subhalo_process_inplace(const struct SubhaloKeySeg * segs, int64_t nsegs, const struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, struct SubTrackTable * tab, MPI_Comm Comm);

#endif
