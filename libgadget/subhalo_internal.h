#ifndef SUBHALO_INTERNAL_H
#define SUBHALO_INTERNAL_H

/* Private interface of the subhalo finder, included by subhalo.c, subhalo_gather.c,
 * subhalo_track.c and subhalo_io.c, by subhalo_unbind.c and subhalo_props.c through
 * subhalo_unbind.h, and by the unit tests. Not part of the public contract in subhalo.h.
 *
 * Memory discipline of one finder call (subhalo_find): everything that must survive the
 * particle exchange or the domain restore lives on the TOP of the main stack (mymalloc2),
 * because domain_exchange may grow the slot block with myrealloc, which needs SlotsBase to be
 * the last allocation on the bottom. Bottom allocations (mymalloc) are scratch that is freed
 * within the function that made it. */

#include "subhalo.h"

/* Per-thread scratch per particle of a key (design 7.1), Phase 2a (audit J7):
 *  - the unbinding kernel, SUBHALO_UNBIND_BYTES_PER_PART (112, subhalo_unbind.h): the Index and
 *    Energy pair of the candidate being unbound and the kernel's transient sources and nodes;
 *    the Phase 3 properties take SUBHALO_PROPS_BYTES_PER_PART (16) of that transient scratch
 *    after the final pass (static assertion in subhalo_track.c);
 *  - 8: the per-particle owner map and the grouped source list (two int32), reused by the tags;
 *  - 8: the whole lists of the completed nodes whose parent has not gathered their remainder
 *    yet (their subtrees are disjoint, so one entry per particle);
 *  - 8 R: the relaxed sources kept by the completed survivors, each at most R times its bound
 *    set (TruncateSource); the bound sets of distinct survivors are disjoint, so at most R
 *    entries per particle, R = SubhaloSourceRelaxFactor rounded up to 1/8;
 *  - per candidate node SUBHALO_ARENA_NODE_BYTES for the alignment of its lists.
 * subhalo_arena_bytes_per_part() = 128 + 8 R is computed once when the parameters are
 * validated (1 <= R <= SUBHALO_MAX_RELAX_FACTOR) and used by the planner and the key loop. */
#define SUBHALO_ARENA_FIXED_BYTES_PER_PART 128
#define SUBHALO_ARENA_NODE_BYTES 64
#define SUBHALO_MAX_RELAX_FACTOR 10.
size_t subhalo_arena_bytes_per_part(void);
/* Main-stack bytes per candidate node of a serial key (struct SubNode in subhalo_track.c;
 * checked there by a static assertion), charged by the plan per member track. */
#define SUBHALO_NODE_BYTES 512
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

/* The tracks no key processed this call, on the home ranks (H6/H7): a resolved track without
 * live particles dies (not sunk); a resolved track with own-tag field particles but no
 * processed key (its pseudo-key was oversized) keeps the Phase 1 hostless rule: Nbound = its
 * field particles, which subhalo_process_inplace has normalised to it (all bound, dense ranks,
 * J3), periodic centre about a common reference, MostBound* the first tracer-type particle in
 * rank order, SUBHALO_FLAG_INPLACE, every property of the previous call cleared. Returns the
 * global counts. Collective. Exposed for the unit tests. */
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

/* Global list of the oversized keys, identical on every rank, sorted by g. Length counts the
 * FOF particles only: field particles of an oversized key stay home with their tags (Phase 1
 * behaviour until Phase 2b), so they are not part of its in-place segments. */
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
    /* Keys: the FOF groups g in [1, TotNgroups], then the pseudo-keys of the hostless resolved
     * tracks, g in [TotNgroups + 1, TotKeys] (Phase 2a plan step 3). KChunk partitions
     * [1, TotKeys] for the key directory. */
    int64_t TotNgroups, TotKeys, KChunk;
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
};

/* ---- the packing as a particle-free function (unit-testable with synthetic inputs) ---- */

/* A census message to the key directory: the particles of key g held by one rank (kind 0),
 * the FOF group record of g from the rank that holds it (kind 1), or an orphan track hosted
 * by g from its home rank (kind 2, ntags = 1, track = its TrackId; Phase 2a step 3). */
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
    int64_t NField;         /* field particles in Length (Phase 2a) */
    int64_t ostart, ocount; /* kind-2 (hosted orphan) messages in the message array */
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

/* Steps 0 and 2: tracer snapshot and HBT's weighted host vote. Every rank sends, per track T,
 * the (BindRank, ID, GrNr) of the SubhaloNumTracerHostFinding lowest-BindRank live tracer-type
 * particles it holds with TrackId == BoundTrackId == T to T's home rank, which keeps the first
 * SubhaloNumTracerHostFinding by (BindRank, ID) as TracerIDs and sets Host to the GrNr (-1: no
 * group) with the largest weight sum 1 / (1 + sqrt(i)), ties to the smaller GrNr. Collective. */
void subhalo_host_vote(struct SubTrackTable * tab, MPI_Comm Comm);
/* Step 3, previous-bound frames: the sums over the live particles with BoundTrackId == T,
 * positions as nearest-image displacements from one common reference per track (the
 * lowest-(own tag, BindRank, ID) such particle, found through the home rank, H7), reduced on
 * the home rank in rank order into RefPos, RefVel, NPrev, MPrev. Collective. */
void subhalo_previous_frames(struct SubTrackTable * tab, MPI_Comm Comm);
/* Step 3, keys of the tracks on the home ranks: Key = Host for a hosted resolved track, the
 * pseudo-key TotNgroups + 1 + ordinal for a hostless resolved track (ordinals in TrackId order
 * by an exclusive scan), -1 otherwise. Returns the global number of pseudo-keys. Collective. */
int64_t subhalo_assign_keys(struct SubTrackTable * tab, int64_t TotNgroups, MPI_Comm Comm);
/* Step 3: plan. The keys of the field particles come from their tracks' Key (request/reply
 * to the home ranks); the census, budgets and packing cover FOF and pseudo-keys; the orphans
 * hosted by a serial key are routed to its processing rank (plan->OrphanIds). Allocates
 * plan->LocalKeys, Assigned, Inplace, OrphanIds on the top of the stack, sets P[i].TopLeaf to
 * the target rank of every particle and encodes the key of every field particle that travels
 * with a serial key as GrNr = -(2 + key) (subhalo_restore_field_grnr undoes it). Collective. */
void subhalo_plan_build(struct SubhaloPlan * plan, const FOFGroups * fof, struct SubTrackTable * tab, int64_t NHostless, MPI_Comm Comm);
/* The processing key of a particle after subhalo_plan_build: GrNr for a group particle, the
 * encoded key of a travelling field particle, -1 otherwise. */
static inline int64_t subhalo_particle_key(const struct particle_data * p) {
    if(p->GrNr >= 0) return p->GrNr;
    if(p->GrNr < -1) return -(p->GrNr + 2);
    return -1;
}
/* The processing key of a track (its record's Key) for the tracks tagging this rank's live
 * field particles: (TrackId, Key) pairs sorted by TrackId. */
struct SubhaloTrackKey {
    int64_t TrackId;
    int64_t Key;
};
/* After the exchange and before subhalo_restore_field_grnr: the Key of every track tagging a
 * live field particle that stayed home (GrNr == -1, not encoded: its track's key is oversized
 * or -1), from the home ranks. Allocates *map on the bottom of the stack. Collective. */
int64_t subhalo_home_field_keys(struct SubTrackTable * tab, struct SubhaloTrackKey ** map, MPI_Comm Comm);
/* The Key of TrackId in such a map, -1 if absent. */
int64_t subhalo_track_key_lookup(const struct SubhaloTrackKey * map, int64_t n, int64_t TrackId);
/* GrNr = -1 again for every field particle (after the serial keys, and on the failure path). */
void subhalo_restore_field_grnr(void);
/* Step 4: the exchange. Returns this rank's failure flag (not reduced). Collective. */
int subhalo_plan_exchange(struct SubhaloPlan * plan, MPI_Comm Comm);
/* Frees what subhalo_plan_build allocated (reverse order). */
void subhalo_plan_free(struct SubhaloPlan * plan);
/* Local sort of P by (garbage last, processing key, TrackId, BindRank, ID). */
void subhalo_local_sort(void);
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
/* Static longest-first schedule of the serial segments over the OpenMP threads, one arena per
 * thread (mymalloc2 in thread order, freed in reverse); keys above one thread's share are
 * processed alone first. Key k needs an arena of bytes_per_part * segs[k].n + extra[k] + 64
 * bytes (extra may be NULL). Returns the largest arena high-water mark of a key as a fraction
 * of its arena (diagnostic). */
double subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, const int64_t * extra,
                               subhalo_key_fn fn, void * userdata);

/* ------------------------------------------------------------------ batched ID lookup (subhalo.c) */

/* What the rank holding a live particle reports about it (steps 1 and 11). */
struct SubhaloIdHit {
    int64_t GrNr;
    int64_t BoundTrackId;
    double Pos[3];           /* translated comoving frame */
    double Vel[3];           /* internal Vel */
    int found;
    int pad;
};
/* For ids[0..n) (any rank, duplicates allowed) find the live particle with that exact ID on any
 * rank: a per-rank sorted (ID, index) index, the ids gathered in batches bounded by
 * SubhaloLookupBatchBytes and searched by every rank, hits returned to the asking rank.
 * out[i].found = 0 for a miss. Collective. */
void subhalo_lookup_ids(const MyIDType * ids, int64_t n, struct SubhaloIdHit * out, MPI_Comm Comm);
/* Bytes the batch buffers of one subhalo_lookup_ids batch can hold on any rank (gathered ids,
 * outgoing and returning replies; J4): at most SubhaloLookupBatchBytes, or the one-id-per-rank
 * floor NTask * (8 + 80) + 80 bytes when that is larger. */
double subhalo_lookup_batch_peak_bytes(int NTask);
/* Bytes of the sorted (ID, index) index and its sort scratch for NumPart particles. */
double subhalo_lookup_index_bytes(int64_t NumPart);
/* Measured high-water mark of the batch buffers of the last lookup on this rank (tests). */
double subhalo_lookup_last_peak_bytes(void);
/* Test knob: at most maxbatch ids per rank and batch (0 restores the byte cap alone). */
void subhalo_lookup_set_max_batch(int64_t maxbatch);
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
    int64_t ipk;             /* in-place winner: index of its key in plan->Inplace, else -1 */
};

/* Diagnostics of one call (local during the call, global totals after subhalo_find). */
struct SubhaloCallStats {
    int64_t NCandidates;     /* candidates unbound in serial keys (each unbinding counted once) */
    int64_t NItercap;        /* of which stopped at SubhaloMaxUnbindIterations (SUBHALO_FLAG_ITERCAP) */
    double WorstConvRatio;   /* smallest Nbound / Nlast of a capped candidate (1 if none) */
    int64_t WorstTrackId;    /* its TrackId (-1 if none; provisional for a newborn) */
    int64_t NFinalPositive;  /* final-pass bound particles with E >= 0, summed over the survivors */
    int64_t NFinalBound;     /* bound particles of the final passes */
    int64_t NTailRejected;   /* tail claims crossing the binding owner's final branch (J2) */
    int64_t NFallbackField;  /* field particles normalised by the Phase 1 fallback (J3) */
    int64_t NFallbackReleased; /* of which released (stale tags of orphans, J3) */
    double ArenaPeakFrac;    /* largest arena high-water mark of a key / its arena size */
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
    const struct SubhaloUnbindParams * up;
    struct SubTrack * Members;   /* sorted by TrackId, read-only during the key loop */
    int64_t NMembers;
    int64_t * SinkWinner;        /* [NMembers]: winner a member sinks into in an in-place key, -1 if none */
    struct SubhaloResult * Res;  /* [NRes] */
    int64_t NRes;
    int64_t * ResStart;          /* [nsegs]: first result slot of each segment */
    int64_t * Hosted;            /* member indices grouped by segment (serial: the members hosted by the key) */
    int64_t * HostedStart;       /* [nsegs + 1] */
    int64_t NInplaceWarn;        /* in-place keys processed with the Phase 1 rule (diagnostic) */
    /* Phase 3 */
    double rho[SUBHALO_NSO];     /* subhalo_so_thresholds at atime */
    int MetalReturnOn;           /* All.MetalReturnOn: metal columns of subhalo_props_baryons */
    struct SubhaloCallStats Stats;   /* this rank's diagnostics (threads add atomically) */
};

/* Binary search of a member record; NULL if absent. */
struct SubTrack * subhalo_find_member(const struct SubhaloCallCtx * ctx, int64_t TrackId);
/* The members hosted by every serial segment (resolved: Key == g; orphan: Host == g), the
 * result slots of every segment (hosted members + 1 for a serial key, 1 for an in-place
 * segment). Allocates ctx->Hosted, HostedStart, ResStart on the bottom of the stack (freed by
 * subhalo_free_slots) and returns the number of result slots. */
int64_t subhalo_result_slots(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx);
void subhalo_free_slots(struct SubhaloCallCtx * ctx);
/* Steps 7a-7h and 8 for every serial segment (OpenMP through subhalo_run_serial_keys). */
void subhalo_process_serial(const struct SubhaloKeySeg * segs, int64_t nsegs, struct SubhaloCallCtx * ctx);
/* Step 9 in-place variant for every oversized key (the Phase 1 rule: every FOF particle bound
 * to the winner, the largest previous member, the other candidates sink into it), and the field
 * particles that stayed home (J3): those of the tracks keyed to an oversized FOF key go to its
 * winner (counted in its record), those of an oversized pseudo-key stay with their hostless
 * track (all bound, dense ranks), stale tags of orphans are released. Must run after the serial
 * keys and before subhalo_restore_field_grnr (the travelled field particles still carry their
 * encoded GrNr). Collective (every rank calls it, with or without in-place segments). */
void subhalo_process_inplace(const struct SubhaloKeySeg * segs, int64_t nsegs, const struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, struct SubTrackTable * tab, MPI_Comm Comm);

#endif
