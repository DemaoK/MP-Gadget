#ifndef SUBHALO_H
#define SUBHALO_H

/* Runtime subhalo finder: public interface.
 *
 * Design: halofinder_design.md (sections referenced below as "design 7.2" etc).
 * Phase 1 plan: halofinder_phase1_plan.md.
 *
 * Phase 1 implemented the plumbing and a "FOF-as-subhalo" membership rule. Phase 2a
 * (halofinder_phase2a_plan.md) replaces the rule for every serial key by the HBT-HERONS recipe:
 * tracer host vote, orphan lookup, previous-bound frames, pseudo-keys for hostless tracks, the
 * post-order unbinding with sinking, the D10 tag rule, descendants and the HBT properties.
 * Oversized keys keep the Phase 1 in-place rule until Phase 2b.
 *
 * Conventions:
 *  - every communicating function takes the communicator explicitly;
 *  - scratch comes from mymalloc/mymalloc2 and is freed in reverse order within the function;
 *  - the track table alone lives in its own allocator (SubTrackTable.alloc);
 *  - endrun() on broken invariants, message() for diagnostics.
 */

#include <mpi.h>
#include <stdint.h>
#include <bigfile-mpi.h>

#include "types.h"
#include "utils/paramset.h"
#include "utils/memory.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "fof.h"
#include "domain.h"
#include "cosmology.h"
#include "petaio.h"

/* ------------------------------------------------------------------ parameters (design 10) */

struct SubhaloParams {
    int    SubhaloFinderOn;              /* master switch; requires SnapshotWithFOF=1 and FOFSaveParticles=1 */
    double SubhaloMaxGatherFraction;     /* 0.5: fraction of (MaxPart - staying) a rank may receive */
    int64_t SubhaloCollectiveMinPart;    /* 0 = off; keys above this are processed in place (testing) */
    int    SubhaloMaxUnbindIterations;   /* 50 (Phase 2a) */
    int    SubhaloRestartFresh;          /* 1 allows an empty table when no complete state is found */
    int    SubhaloTracerTypes;           /* bitmask, default 2+16 */
    int    SubhaloPotentialTypes;        /* bitmask, default 1+2+16+32 */
    int    SubhaloMinNumPart;            /* 20 */
    int    SubhaloMinNumTracerPart;      /* 10 */
    int    SubhaloNumTracerHostFinding;  /* 10 */
    int64_t SubhaloLookupBatchBytes;     /* 64 MB */
    double SubhaloMajorProgenitorMassRatio; /* 0.8 */
    double SubhaloBoundMassPrecision;    /* 0.995 */
    double SubhaloSourceRelaxFactor;     /* 3 */
    int    SubhaloPotentialSampleSize;   /* 1000 */
    double SubhaloTreeOpenAngle;         /* 0.45 */
    int    SubhaloMergeTrapped;          /* 1 */
    int    SubhaloSinkSymmetric;         /* 0 */
    int    SubhaloCoreSize;              /* 20 */
    int    SubhaloReassignGas;           /* 1 */
    int    SubhaloNumNeighboursForReassignment; /* 10 (HBT NumNeighboursForReassignment) */
    int    SubhaloUnbindWithThermal;     /* 0 */
    int    SubhaloFieldSO;               /* 1 */
    int    SubhaloSOBins;                /* 64 */
    double SubhaloSOSearchFactor;        /* 2 */
    int    SubhaloSOMaxDoublings;        /* 3 */
    int    SubhaloSORefineMinPart;       /* 1000 */
    double SubhaloMajorMergerRatio;      /* 0.3 */
    int    SubhaloGeometricParents;      /* 0 */
    /* Debug knob (not in the design doc): 1 sends every serial key to rank 0 so that the
     * exchange over-commits and the failure path of design 7.3 can be exercised. */
    int    SubhaloDebugForceExchangeFailure;
};

/* Read the parameters on rank 0 and broadcast. Validates SubhaloFinderOn against
 * SnapshotWithFOF and FOFSaveParticles (endrun on an invalid combination). */
void set_subhalo_params(ParameterSet * ps);
/* Test helper: set the parameters without a ParameterSet (validated like the parameter file;
 * endrun(1) on an invalid value). */
void subhalo_set_testpar(struct SubhaloParams params);
/* The run's unit system (the snapshot header values, design 7.10): catalogue velocities are
 * Vel * UnitVelocity_in_cm_per_s / (1e5 * a) km/s; positions stay in the run's length unit.
 * The three values are also written to the PIG Header. Called from begrun; the defaults are
 * MP-Gadget's (kpc/h, 1e10 Msun/h, km/s). endrun(1) on a non-finite or non-positive value. */
void subhalo_set_units(double UnitLength_in_cm, double UnitMass_in_g, double UnitVelocity_in_cm_per_s);
void subhalo_get_units(double * UnitLength_in_cm, double * UnitMass_in_g, double * UnitVelocity_in_cm_per_s);
int subhalo_finder_on(void);
const struct SubhaloParams * subhalo_get_params(void);

/* ------------------------------------------------------------------ constants */

#define SUBHALO_SCHEMA_VERSION 1
#define SUBHALO_NSO 5                 /* vir, 200b, 200c, 500c, 2500c */
#define SUBHALO_NOTRACK (-1L)
/* Upper bound on SubhaloNumTracerHostFinding: the tracer IDs of step 0 live in the record. */
#define SUBHALO_MAXTRACERS 16
/* Provisional TrackId of a newborn candidate with global ordinal g (design 7.2 step 3). */
#define SUBHALO_PROVISIONAL(g) (-(2L + (int64_t)(g)))
#define SUBHALO_IS_PROVISIONAL(t) ((t) < -1L)
#define SUBHALO_PROVISIONAL_ORDINAL(t) (-(t) - 2L)

/* Flags bits (design 8) */
#define SUBHALO_FLAG_ITERCAP      (1 << 0)
#define SUBHALO_FLAG_INPLACE      (1 << 1)
#define SUBHALO_FLAG_PLANCENTRAL  (1 << 2)
#define SUBHALO_FLAG_SOCUTOFF     (1 << 3)
#define SUBHALO_FLAG_SONOCROSS    (1 << 4)

/* Particle block types for the finder's own IOTable entries (PTYPE_FOF_GROUP is 1024). */
#define PTYPE_SUBHALO 1025
#define PTYPE_ORPHAN  1026

/* ------------------------------------------------------------------ track record (Appendix B) */

/* One record per track ever created. Field widths equal the on-disk dtypes so the restart
 * round trip is exact. Columns marked (later) are written as -1 until their phase lands.
 * Group F (orphan) columns are a subset of these; see subhalo_io.c for the two column lists. */
struct SubTrack {
    /* group A: identity and links */
    int64_t TrackId;
    int64_t HostGroup;               /* FOF GroupID at this call, -1 field/hostless */
    int64_t NestedParentTrackId;     /* pid, -1 for a central */
    int64_t CentralTrackId;          /* upid; a central's own TrackId */
    int64_t SinkTrackId;             /* -1 unless sunk */
    int64_t DescendantTrackId;       /* itself while alive; -1 or the descendant when dead/sunk */
    int32_t Rank, Depth;
    int32_t SnapBirth, SnapDeath, SnapSink, SnapLastIsolation;
    MyIDType MostBoundID;
    /* counts (D10) */
    int32_t Nbound, NboundExclusive, NboundTracer;
    int32_t Flags;
    int32_t NboundType[6], NboundExclusiveType[6], NtailType[6];
    /* particle slices in PIG_xxx/<t>/ (design 9): in-group (section 0) and field (section 1) */
    int64_t SliceOffsetByType[6];
    int32_t SliceLengthByType[6], SliceNboundByType[6];
    int64_t FieldSliceOffsetByType[6];
    int32_t FieldSliceLengthByType[6], FieldSliceNboundByType[6];
    /* group B: bound kinematics and structure */
    float   Mbound, MboundType[6];
    double  CorePos[3], MostBoundPos[3], CoMPos[3];     /* comoving, run length unit, physical frame (offset removed) */
    float   CoreVel[3], MostBoundVel[3], CoMVel[3];     /* physical peculiar km/s */
    float   CoreSigmaX, CoreSigmaV;
    float   Vmax, Rmax, VmaxVir, RmaxVir, Rhalf, Renclose;
    float   MBoundSO[SUBHALO_NSO], RBoundSO[SUBHALO_NSO];
    float   J[3], JVir[3];
    float   SpinPeebles, SpinBullock, Ekin, Epot, KinToPot, Xoff, Voff, RsKlypin;
    float   BtoA, CtoA, BtoA500, CtoA500;
    float   AxisA[3], AxisA500[3];
    float   InertiaTensor[6];
    float   Rs, RockstarCoreVel[3], BulkVel[3], Vrms;   /* (later) */
    /* group C: baryonic sums */
    float   StarFormationRate, GasMetalMass, StellarMetalMass, BlackholeMass, BlackholeAccretionRate, MassHeIonized;
    float   GasMetalElemMass[NMETALS], StellarMetalElemMass[NMETALS];
    MyIDType MostMassiveBHID;
    /* group D: field SO, centrals only */
    float   MSO[SUBHALO_NSO], RSO[SUBHALO_NSO];
    float   MTypeSO200c[6], MTypeSOVir[6];
    /* group E: history (design 7.7) */
    float   MboundPeak, VmaxAtMboundPeak, VmaxPeak, MboundAcc, VmaxAcc, MboundFirstAcc, VmaxFirstAcc;
    int32_t SnapMboundPeak, SnapVmaxPeak, SnapAcc, SnapFirstAcc, SnapLastMajorMerger;
    int64_t GeomParentTrackId, GeomUpParentTrackId;
    /* group F: orphan tracer state */
    double  TracerPos[3];
    float   TracerVel[3];
    int32_t SnapTracerLost;
    /* ---- transient, never written: valid only inside one subhalo_find call ---- */
    int64_t Host;                    /* host vote result this call (GrNr or -1); an orphan's tracer GrNr */
    int32_t NboundPrev;              /* Nbound at the previous call, frozen for the central choice */
    int32_t Alive;                   /* 1 if this track won a key or is hostless-resolved this call */
    /* Phase 2a (plan section 1, steps 0-3): previous-bound frame from the BoundTrackId sums, in the
     * translated comoving frame and internal Vel units (the unbinding kernel's frame) */
    double RefPos[3], RefVel[3];
    int64_t NPrev;                   /* live particles with BoundTrackId == TrackId */
    double MPrev;                    /* and their mass */
    MyIDType TracerIDs[SUBHALO_MAXTRACERS];  /* step 0: the most-bound own tracers, rank order */
    int32_t NTracers;
    int32_t pad2a;
    int64_t Key;                     /* processing key of the track's field particles: Host for a hosted
                                      * resolved track, the pseudo-key of a hostless resolved track,
                                      * -1 for orphans and dead tracks */
};

/* Initialise every column of a record to "absent" (-1 / 0 as appropriate) with the given TrackId. */
void subhalo_track_clear(struct SubTrack * t, int64_t TrackId);

/* ------------------------------------------------------------------ track table (design 7.1) */

/* Distributed with a static home: record T lives on rank T / Chunk at local index T % Chunk
 * (every rank holds the same Chunk; the last rank may hold fewer records). Chunk is
 * ceil(NTotal / NTask), recomputed after every call; when it changes the table is
 * redistributed so the rule holds exactly. The table is a single block in its own allocator,
 * grown by doubling with a copy. */
struct SubTrackTable {
    Allocator alloc[1];
    struct SubTrack * Track;
    int64_t NLocal;                  /* records held by this rank */
    int64_t NAlloc;                  /* capacity of Track */
    int64_t NTotal;                  /* tracks ever created, == next TrackId */
    int64_t Chunk;
    int NTask, ThisTask;
    int initialised;
};

/* Create the allocator and an empty table. bytes is the initial allocator size; it grows. */
void subhalo_table_init(struct SubTrackTable * tab, size_t bytes, MPI_Comm Comm);
void subhalo_table_free(struct SubTrackTable * tab);
static inline int subhalo_table_home_rank(const struct SubTrackTable * tab, int64_t TrackId) {
    return (int) (TrackId / tab->Chunk);
}
static inline int64_t subhalo_table_local_index(const struct SubTrackTable * tab, int64_t TrackId) {
    return TrackId % tab->Chunk;
}
/* Returns the local record of TrackId or NULL if it is not homed here. */
struct SubTrack * subhalo_table_get(struct SubTrackTable * tab, int64_t TrackId);
/* Grow the local block to hold at least n records (collective-free). */
void subhalo_table_reserve(struct SubTrackTable * tab, int64_t n);
/* After NTotal changed: recompute Chunk = ceil(NTotal / NTask) and, if any record is not at its
 * home under it, move every record to its new home (collective; one MPI_Alltoallv, skipped when
 * nothing moves). After return every rank holds exactly the records homed on it for TrackIds in
 * [0, NTotal), in local-index order (NLocal = their count), cleared (subhalo_track_clear) where
 * new; fill them through subhalo_table_get. There is no append: a record exists iff its TrackId
 * is below NTotal. */
void subhalo_table_redistribute(struct SubTrackTable * tab, MPI_Comm Comm);
/* Fetch copies of records for the requested TrackIds (any ranks, duplicates allowed) from their
 * home ranks: out[i] is the record of ids[i]. Collective (request/reply MPI_Alltoallv pairs,
 * batched by SubhaloLookupBatchBytes so that one batch holds a bounded number of records on
 * every rank; the batches loop while any rank has requests left). */
void subhalo_table_fetch(struct SubTrackTable * tab, const int64_t * ids, int64_t n, struct SubTrack * out, MPI_Comm Comm);
/* Write modified records back to their home ranks (collective; batched like the fetch). The
 * transient fields are copied too. Duplicate stores of one TrackId are allowed (the last one
 * received wins; the driver only stores identical duplicates). */
void subhalo_table_store(struct SubTrackTable * tab, const struct SubTrack * recs, int64_t n, MPI_Comm Comm);
/* The global table of the running finder (subhalo.c). */
struct SubTrackTable * subhalo_get_table(void);

/* ------------------------------------------------------------------ driver (design 7.2, 11) */

/* Called once after init() has read the particles, at the start of run() and runfof().
 * Sets every particle's tags to -1; if the finder is on, finds the latest complete state in
 * PIG_m with m <= MaxStateSnapNum (design 11 step 1), reads the tables, joins the tags onto
 * the particles by lineage base ID, and sets the global counters. Aborts if no state exists
 * and SubhaloRestartFresh is 0, unless RestartSnapNum < 0 (a fresh run from ICs).
 * runfof passes MaxStateSnapNum = RestartSnapNum - 1 (offline mode). Collective. */
void subhalo_init(int RestartSnapNum, int MaxStateSnapNum, const char * OutputDir, MPI_Comm Comm);

/* One finder call, between write_checkpoint and fof_save_groups. fof->Group[] and P[i].GrNr
 * are those of fof_fof(ddecomp, 1, Comm). On return the domain is restored (the function does
 * its own slots_gc_sorted + domain_maintain), the tags describe this call, and the table
 * holds the new records. Returns 1 if the call completed, 0 if the exchange failed (tags and
 * table untouched). Collective. */
int subhalo_find(FOFGroups * fof, DomainDecomp * ddecomp, int SnapNum, double atime, Cosmology * CP, MPI_Comm Comm);

/* Status accessors used by the writer. */
int subhalo_last_call_complete(void);     /* 1 if the most recent subhalo_find completed */
int subhalo_last_call_snapshot(void);     /* snapshot number of the most recent completed call, -1 if none */
int subhalo_state_snapshot(void);         /* snapshot whose state the tags currently describe */
double subhalo_state_time(void);          /* and its time */
double subhalo_time_since_previous_call(void);
/* Count tagged particles per type on this rank (debug/diagnostic). */
void subhalo_count_tagged(const struct part_manager_type * pman, int64_t ntagged[6]);

/* ------------------------------------------------------------------ IO (design 9, subhalo_io.c) */

/* Selection predicate for the PIG particle write: in a group or tagged, and not swallowed.
 * With the finder off this reduces to GrNr >= 0 && !Swallowed.
 * Size limit: the host IO path (petaio_build_selection's int selection, the int index of the
 * IOTable getters) indexes local particles with int, so the PIG writer with the finder on
 * checks collectively that every rank holds at most INT_MAX particles (endrun(1) otherwise).
 * The finder's own table IO calls the int getters/setters on rebased single rows and has no
 * such limit; the int32 count columns (Nbound, NboundType, slice lengths) are range-checked
 * before they are stored (endrun(1) rather than wrap). */
int subhalo_save_select(int64_t i, const struct particle_data * Parts);

/* The 16-byte radix key of design 9 for the PIG exchange sort, 112 bits used:
 * section (1) | key (39) | TrackId + 1 (40) | BindRank (32), most significant first.
 * mpsort compares a radix as a little-endian integer (fof.c's 24-byte keys put the primary
 * key in u[2]), so out[1] is the HIGH word and out[0] the LOW word:
 *   out[1] = (section << 63) | (key << 24) | ((TrackId + 1) >> 16)        (key: bits 24..62)
 *   out[0] = (((TrackId + 1) & 0xFFFF) << 48) | ((uint64_t) BindRank << 16)
 * Section 0 and key = GrNr for GrNr >= 0; section 1 and key = 0 for a tagged particle with
 * GrNr < 0. TrackId + 1 and BindRank are 0 for an untagged particle. endrun on a range
 * overflow (GrNr >= 2^39, TrackId + 1 >= 2^40, BindRank < 0). */
void subhalo_sort_key(const struct particle_data * p, uint64_t out[2]);
/* qsort comparator on the same key (Type first, then the 16-byte key). */
int subhalo_order_by_type_and_key(const void * a, const void * b);

/* Register TrackId (i8), BindRank (i4), BoundTrackId (i8) for all six types in a PIG IOTable. */
void subhalo_register_particle_blocks(struct IOTable * IOTable);

/* After the PIG exchange and local sort: compute the in-group and field slices of every track
 * on this rank (global offsets via MPI_Exscan of the per-type selected counts; runs straddling
 * rank boundaries reconciled with the nearest non-empty neighbours) and store them into the
 * home-rank records. selection/ptype_offset/ptype_count are those of petaio_build_selection on
 * the (possibly temporary) particle table pman. Collective. */
void subhalo_record_slices(const struct part_manager_type * pman, const int * selection, const int64_t * ptype_offset, const int64_t * ptype_count, MPI_Comm Comm);

/* Header attributes of design 9 / Appendix B, with SubhaloStateComplete = 0. Called inside
 * fof_write_header while the Header block is open. Collective (reductions for the totals). */
void subhalo_write_header_attrs(BigBlock * bh, const struct part_manager_type * pman, double atime, MPI_Comm Comm);

/* Write PIG_xxx/Subhalo/<column> (Nbound > 0) and PIG_xxx/Orphan/<column> (Nbound == 0) from the home-rank
 * records, each in TrackId order (globally sorted by the static partition). Collective. */
void subhalo_write_tables(BigFile * bf, MPI_Comm Comm);

/* Reopen Header, set SubhaloStateComplete = complete, SubhaloStateSnapshot and
 * SubhaloStateTime, close. Called after the last particle block. Collective. */
void subhalo_finalize_output(BigFile * bf, int complete, MPI_Comm Comm);

/* Record the snapshot number of the PIG being written (SnapshotIndex, and the check that a
 * complete state describes this snapshot). Called by fof_save_particles before the header is
 * written, with the number parsed from the PIG name; -1 if unknown. Not collective. */
void subhalo_set_output_snapshot(int SnapNum);

/* ------------------------------------------------------------------ restart (design 11, subhalo_io.c) */

/* Search PIG_m, m = MaxStateSnapNum down to 0, for a complete, readable state (marker,
 * schema, SubhaloStateSnapshot == m, finite state time equal to the file's, counts match row
 * counts, every required block present with the schema's dtype and with every payload file its
 * header declares present at the declared size). Returns m or -1. Collective (rank 0 probes,
 * broadcasts). FOFFileBase is All.FOFFileBase. */
int subhalo_find_state(const char * OutputDir, const char * FOFFileBase, int MaxStateSnapNum, MPI_Comm Comm);

/* Read Subhalo/<column> and Orphan/<column> of that PIG into the table (contiguous row slices per rank, then
 * routed to the home ranks), set NTotal from NumTracksTotal, return the state time. Requires
 * exactly one row per TrackId in [0, NumTracksTotal) (no duplicate, no hole), Subhalo rows
 * resolved (Nbound > 0, SnapDeath < 0) and Orphan rows dead or sunk (SnapDeath >= 0). Any read
 * failure or violation is fatal (endrun(1)) with a message naming the PIG and the remedies
 * (clear its SubhaloStateComplete marker, or SubhaloRestartFresh = 1). Collective. */
double subhalo_read_state(const char * pigname, struct SubTrackTable * tab, MPI_Comm Comm);

/* Join the saved per-particle tags onto pman by lineage base ID (low 56 bits of the ID), the
 * joint mpsort_mpi of saved and local records with nearest-non-empty-rank boundary
 * reconciliation. Every saved row takes part, untagged rows included, and the saved tags are
 * validated first (TrackId and BoundTrackId in [-1, NumTracksTotal), BindRank >= 0 for a tagged
 * row; fatal otherwise, like a read failure). A local particle with an exact saved ID match
 * takes its tag, -1 if it was saved untagged; otherwise, if
 * its own generation byte (ID >> 56) is > 0, it inherits the tag of the saved record whose ID
 * equals its lineage base (generation 0, always the spawning parent in MP-Gadget:
 * slots_split_particle gives children base + (g << 56) and stars never spawn), if there is
 * one; everything else gets -1 (in particular a generation-0 particle never inherits from a
 * saved child). Returns the number of inherited (non-exact) tags on this rank. Collective. */
int64_t subhalo_join_tags(const char * pigname, struct part_manager_type * pman, MPI_Comm Comm);

#endif
