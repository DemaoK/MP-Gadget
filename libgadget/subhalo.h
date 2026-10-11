#ifndef SUBHALO_H
#define SUBHALO_H

/* Runtime subhalo finder (HBT-HERONS membership, Rockstar properties): public interface.
 * Parameters (design 10), the track record (Appendix B), the track table (7.1), the driver
 * (7.2, 11), PIG output (9) and restart (11). "design X" refers to halofinder_design.md.
 * Conventions: communicating functions take the communicator; scratch is mymalloc'd and freed
 * in reverse order; the track table has its own allocator; endrun() on broken invariants. */

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
    int64_t SubhaloCollectiveMinPart;    /* 0 = off; testing: keys above this are processed in place */
    int    SubhaloMaxUnbindIterations;   /* 50 */
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
    int    SubhaloFinalPass;             /* 0 (default) = sampled with an exact head, as HBT (decision 42.4); 1 = exact */
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
    /* Debug: 1 sends every serial key to rank 0, to exercise the exchange failure path (design 7.3). */
    int    SubhaloDebugForceExchangeFailure;
};

/* Read the parameters on rank 0 and broadcast; endrun if SubhaloFinderOn lacks SnapshotWithFOF
 * or FOFSaveParticles. */
void set_subhalo_params(ParameterSet * ps);
/* Test helper: set the parameters without a ParameterSet, validated like the parameter file. */
void subhalo_set_testpar(struct SubhaloParams params);
/* The run's units (design 7.10), set from begrun and written to the PIG Header: catalogue
 * velocities are Vel * UnitVelocity_in_cm_per_s / (1e5 a) km/s. Defaults: kpc/h, 1e10 Msun/h, km/s. */
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
/* Provisional TrackId of a newborn candidate with FOF ordinal g >= 1 (design 7.2 step 3): <= -3,
 * since -1 is "no track" and -2 is SUBHALO_DESC_PENDING. */
#define SUBHALO_PROVISIONAL(g) (-(2L + (int64_t)(g)))
#define SUBHALO_IS_PROVISIONAL(t) ((t) <= -3L)
#define SUBHALO_PROVISIONAL_ORDINAL(t) (-(t) - 2L)
/* A descendant vote whose tracer is outside the dead track's key, looked up after the domain
 * restore (step 11). */
#define SUBHALO_DESC_PENDING (-2L)

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
 * round trip is exact; subhalo_io.c lists the Subhalo and Orphan columns. */
struct SubTrack {
    /* group A: identity and links */
    int64_t TrackId;
    int64_t HostGroup;               /* FOF GroupID at this call, -1 field/hostless; for an orphan its
                                      * tracer's group (-1 in the field or lost) */
    int64_t NestedParentTrackId;     /* pid, -1 for a central */
    int64_t CentralTrackId;          /* upid; a central's own TrackId; for an orphan the central of its
                                      * current host group, -1 if none (its other links stay as at death) */
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
    float   Rs, RockstarCoreVel[3], BulkVel[3], Vrms;
    /* group C: baryonic sums */
    float   StarFormationRate, GasMetalMass, StellarMetalMass, BlackholeMass, BlackholeAccretionRate, MassHeIonized;
    float   GasMetalElemMass[NMETALS], StellarMetalElemMass[NMETALS];
    MyIDType MostMassiveBHID;
    /* group D: field SO, centrals only */
    float   MSO[SUBHALO_NSO], RSO[SUBHALO_NSO];
    float   MTypeSO200c[6], MTypeSOVir[6];
    /* Rockstar-comparable profile of the virial sphere (decision 44, subhalo_so.h pass 3) */
    float   RsSO, RsKlypinSO, VmaxSO, RmaxSO;
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
    /* previous-bound frame from the BoundTrackId sums, in the kernel's frame (translated comoving,
     * internal Vel units) */
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

/* Static home: record T lives on rank T / Chunk at local index T % Chunk, Chunk = ceil(NTotal /
 * NTask), redistributed when Chunk changes. One block in its own allocator, grown by doubling. */
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
/* After NTotal changed: recompute Chunk and move records to their homes (one Alltoallv, skipped
 * when nothing moves); new records are cleared. No append: a record exists iff TrackId < NTotal. Collective. */
void subhalo_table_redistribute(struct SubTrackTable * tab, MPI_Comm Comm);
/* Fetch copies of the records of ids[] (duplicates allowed) from their home ranks into out[].
 * Collective; batched by SubhaloLookupBatchBytes until no rank has requests left. */
void subhalo_table_fetch(struct SubTrackTable * tab, const int64_t * ids, int64_t n, struct SubTrack * out, MPI_Comm Comm);
/* Write records back to their home ranks, transient fields included (collective, batched like
 * the fetch). Duplicates allowed: the last received wins (the driver stores identical ones only). */
void subhalo_table_store(struct SubTrackTable * tab, const struct SubTrack * recs, int64_t n, MPI_Comm Comm);
/* The global table of the running finder (subhalo.c). */
struct SubTrackTable * subhalo_get_table(void);

/* ------------------------------------------------------------------ driver (design 7.2, 11) */

/* After init(), in run() and runfof(): clear the tags, then, finder on, load the latest complete
 * state in PIG_m, m <= MaxStateSnapNum (design 11), and join its tags. No state is fatal unless
 * SubhaloRestartFresh or RestartSnapNum < 0 (fresh run). runfof passes RestartSnapNum - 1. Collective. */
void subhalo_init(int RestartSnapNum, int MaxStateSnapNum, const char * OutputDir, MPI_Comm Comm);

/* One finder call (design 7.2), between write_checkpoint and fof_save_groups, on the groups of
 * fof_fof. On return the domain is restored and the tags and table describe this call. Returns 0,
 * tags and table untouched (orphan tracer state excepted), if the exchange or orphan lookup did
 * not fit. Collective over MPI_COMM_WORLD (the field SO's treewalk). */
int subhalo_find(FOFGroups * fof, DomainDecomp * ddecomp, int SnapNum, double atime, Cosmology * CP, int MetalReturnOn, MPI_Comm Comm);

/* Status accessors used by the writer. */
int subhalo_last_call_complete(void);     /* 1 if the most recent subhalo_find completed */
int subhalo_last_call_snapshot(void);     /* snapshot number of the most recent completed call, -1 if none */
int subhalo_state_snapshot(void);         /* snapshot whose state the tags currently describe */
double subhalo_state_time(void);          /* and its time */
double subhalo_time_since_previous_call(void);
/* Comoving SO thresholds (internal units) of the call whose records the table holds, or of the
 * loaded state's Header; -1 if unknown. */
void subhalo_state_so_thresholds(double rho[SUBHALO_NSO]);
void subhalo_set_state_so_thresholds(const double rho[SUBHALO_NSO]);
/* Count tagged particles per type on this rank (debug/diagnostic). */
void subhalo_count_tagged(const struct part_manager_type * pman, int64_t ntagged[6]);

/* ------------------------------------------------------------------ IO (design 9, subhalo_io.c) */

/* PIG particle selection: in a group or tagged, and not swallowed (finder off: GrNr >= 0).
 * The host IO indexes particles with int, so the writer checks every rank holds <= INT_MAX. */
int subhalo_save_select(int64_t i, const struct particle_data * Parts);

/* 16-byte radix key of the PIG exchange sort (design 9): section (1) | GrNr or 0 (39) |
 * TrackId + 1 (40) | BindRank (32); out[1] is the high word (mpsort compares little-endian).
 * Untagged: TrackId + 1 = BindRank = 0. endrun on a range overflow. */
void subhalo_sort_key(const struct particle_data * p, uint64_t out[2]);
/* qsort comparator on the same key (Type first, then the 16-byte key). */
int subhalo_order_by_type_and_key(const void * a, const void * b);

/* Register TrackId (i8), BindRank (i4), BoundTrackId (i8) for all six types in a PIG IOTable. */
void subhalo_register_particle_blocks(struct IOTable * IOTable);

/* After the PIG exchange and local sort: the in-group and field slices of every track, runs
 * across rank boundaries joined, stored to the home records. Arguments as petaio_build_selection's. Collective. */
void subhalo_record_slices(const struct part_manager_type * pman, const int * selection, const int64_t * ptype_offset, const int64_t * ptype_count, MPI_Comm Comm);

/* Header attributes (design 9, Appendix B) with SubhaloStateComplete = 0, inside fof_write_header. Collective. */
void subhalo_write_header_attrs(BigBlock * bh, const struct part_manager_type * pman, double atime, MPI_Comm Comm);
/* The header's SubhaloParameters string ("Name = value", doubles %.17g); provenance only. */
void subhalo_format_params(char * buf, size_t size);

/* Write Subhalo/<column> (Nbound > 0) and Orphan/<column> (Nbound == 0) in TrackId order. Collective. */
void subhalo_write_tables(BigFile * bf, MPI_Comm Comm);

/* After the last particle block: set SubhaloStateComplete, StateSnapshot and StateTime. Collective. */
void subhalo_finalize_output(BigFile * bf, int complete, MPI_Comm Comm);

/* The snapshot number of the PIG being written (-1 if unknown), set by fof_save_particles
 * before the header; a complete state must describe it. Not collective. */
void subhalo_set_output_snapshot(int SnapNum);

/* ------------------------------------------------------------------ restart (design 11, subhalo_io.c) */

/* Search PIG_m, m = MaxStateSnapNum down to 0, for a complete and readable state (marker, schema,
 * snapshot, time, row counts, every block and payload file). Returns m or -1. Collective (rank 0 probes). */
int subhalo_find_state(const char * OutputDir, const char * FOFFileBase, int MaxStateSnapNum, MPI_Comm Comm);

/* Read the Subhalo and Orphan tables into the track table and return the state time. Requires one
 * row per TrackId in [0, NumTracksTotal), Subhalo rows alive and Orphan rows dead; else endrun(1). Collective. */
double subhalo_read_state(const char * pigname, struct SubTrackTable * tab, MPI_Comm Comm);

/* Join the saved particle tags onto pman (joint sort of saved and local records by ID): an exact
 * ID match takes its tag; a spawned particle (ID >> 56 > 0) inherits its generation-0 parent's;
 * else -1. Saved tags are validated (fatal). Returns the inherited count on this rank. Collective. */
int64_t subhalo_join_tags(const char * pigname, struct part_manager_type * pman, MPI_Comm Comm);

#endif
