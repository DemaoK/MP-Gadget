/* Runtime subhalo finder: parameters, the track table, the batched ID lookups (orphan tracers,
 * remote descendants), the driver and the track history.
 *
 * Design: halofinder_design.md; Phase 1 plan: halofinder_phase1_plan.md; Phase 2a plan:
 * halofinder_phase2a_plan.md (section 1 is the pipeline implemented by subhalo_find below,
 * step by step).
 * The exchange, plan and domain restore are in subhalo_gather.c, the per-key processing in
 * subhalo_track.c, the catalogue output and the restart state in subhalo_io.c. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/openmpsort.h"
#include "utils/system.h"
#include "utils/string.h"

#include "walltime.h"
#include "gravity.h"
#include "subhalo.h"
#include "subhalo_internal.h"
#include "subhalo_collective.h"
#include "subhalo_unbind.h"
#include "subhalo_so.h"

/* ------------------------------------------------------------------ parameters */

static struct SubhaloParams SubParams;
/* PIG base name (FOFFileBase), read with the parameters so subhalo_init can find the state. */
static char SubhaloFOFFileBase[100] = "PIG";

/* The track table of the running finder. */
static struct SubTrackTable SubTable;

/* Status of the finder, read by the writer (subhalo_io.c). */
static struct {
    int LastCallComplete;       /* 1 if the most recent subhalo_find completed */
    int LastCallSnapshot;       /* PIG number of the most recent completed call, -1 if none */
    int StateSnapshot;          /* PIG number whose membership the tags describe, -1 if none */
    double StateTime;           /* scale factor of that state, -1 if none */
    double TimeSincePrevious;   /* scale-factor interval between the last two completed calls, -1 if none */
    double SOThresholds[SUBHALO_NSO];   /* comoving SO thresholds of the call the records describe, -1 if unknown */
} SubStatus = {0, -1, -1, -1., -1., {-1., -1., -1., -1., -1.}};

#define SUBHALO_TABLE_INITIAL_BYTES (64L * 1024L * 1024L)

/* The run's units (design 7.10); MP-Gadget's defaults until subhalo_set_units is called. */
static struct {
    double UnitLength_in_cm;
    double UnitMass_in_g;
    double UnitVelocity_in_cm_per_s;
} SubUnits = {3.085678e21, 1.989e43, 1e5};

int
subhalo_isfinite(double x)
{
    uint64_t u;
    memcpy(&u, &x, sizeof(u));
    return ((u >> 52) & 0x7FF) != 0x7FF;
}

/* Largest double below 2^62, the range in which the integer-valued parameters are converted. */
#define SUBHALO_INT_PARAM_MAX 4.6e18

/* Validation of the values Phase 1 uses (H14). Every failure is endrun(1) on every rank: the
 * caller runs it on every rank with the broadcast values. */
static void
subhalo_validate_params(const struct SubhaloParams * sp)
{
    if(!subhalo_isfinite(sp->SubhaloMaxGatherFraction) || sp->SubhaloMaxGatherFraction <= 0)
        endrun(1, "SubhaloMaxGatherFraction = %g must be finite and positive (values above 1 are allowed for testing).\n", sp->SubhaloMaxGatherFraction);
    if(sp->SubhaloCollectiveMinPart < 0)
        endrun(1, "SubhaloCollectiveMinPart = %ld must not be negative.\n", sp->SubhaloCollectiveMinPart);
    if(sp->SubhaloLookupBatchBytes < 1024L * 1024L)
        endrun(1, "SubhaloLookupBatchBytes = %ld must be at least 1 MB.\n", sp->SubhaloLookupBatchBytes);
    if(sp->SubhaloTracerTypes < 0 || sp->SubhaloTracerTypes > 63)
        endrun(1, "SubhaloTracerTypes = %d must be a bitmask of the six particle types (0-63).\n", sp->SubhaloTracerTypes);
    if(sp->SubhaloPotentialTypes < 0 || sp->SubhaloPotentialTypes > 63)
        endrun(1, "SubhaloPotentialTypes = %d must be a bitmask of the six particle types (0-63).\n", sp->SubhaloPotentialTypes);
    if(sp->SubhaloMinNumTracerPart < 1 || sp->SubhaloNumTracerHostFinding < 1)
        endrun(1, "SubhaloMinNumTracerPart = %d and SubhaloNumTracerHostFinding = %d must be at least 1.\n",
               sp->SubhaloMinNumTracerPart, sp->SubhaloNumTracerHostFinding);
    if(sp->SubhaloNumTracerHostFinding > sp->SubhaloMinNumTracerPart)
        endrun(1, "SubhaloNumTracerHostFinding = %d must not exceed SubhaloMinNumTracerPart = %d.\n",
               sp->SubhaloNumTracerHostFinding, sp->SubhaloMinNumTracerPart);
    if(sp->SubhaloNumTracerHostFinding > SUBHALO_MAXTRACERS)
        endrun(1, "SubhaloNumTracerHostFinding = %d must not exceed %d (the tracer IDs live in the track record).\n",
               sp->SubhaloNumTracerHostFinding, SUBHALO_MAXTRACERS);
    if(sp->SubhaloNumNeighboursForReassignment < 1 || sp->SubhaloNumNeighboursForReassignment > SUBHALO_KNN_MAXK)
        endrun(1, "SubhaloNumNeighboursForReassignment = %d must be in [1, %d].\n",
               sp->SubhaloNumNeighboursForReassignment, SUBHALO_KNN_MAXK);
    if(sp->SubhaloCoreSize < 1 || sp->SubhaloPotentialSampleSize < 0 || sp->SubhaloMaxUnbindIterations < 1)
        endrun(1, "SubhaloCoreSize = %d and SubhaloMaxUnbindIterations = %d must be at least 1, SubhaloPotentialSampleSize = %d not negative.\n",
               sp->SubhaloCoreSize, sp->SubhaloMaxUnbindIterations, sp->SubhaloPotentialSampleSize);
    if(sp->SubhaloFinalPass != 0 && sp->SubhaloFinalPass != 1)
        endrun(1, "SubhaloFinalPass = %d must be 1 (exact final pass) or 0 (sampled final pass).\n", sp->SubhaloFinalPass);
    if(!(sp->SubhaloSourceRelaxFactor >= 1 && sp->SubhaloSourceRelaxFactor <= SUBHALO_MAX_RELAX_FACTOR))
        endrun(1, "SubhaloSourceRelaxFactor = %g must be in [1, %g]: the per-thread arena reserves 8 R bytes per particle "
                  "for the relaxed sources (J7; HBT finds no change above 3).\n", sp->SubhaloSourceRelaxFactor, SUBHALO_MAX_RELAX_FACTOR);
    if(!(sp->SubhaloSourceRelaxFactor >= 1) || !(sp->SubhaloTreeOpenAngle > 0) || !(sp->SubhaloBoundMassPrecision > 0 && sp->SubhaloBoundMassPrecision <= 1)
       || !(sp->SubhaloMajorProgenitorMassRatio >= 0 && sp->SubhaloMajorProgenitorMassRatio <= 1))
        endrun(1, "SubhaloSourceRelaxFactor = %g must be >= 1, SubhaloTreeOpenAngle = %g positive, SubhaloBoundMassPrecision = %g in (0, 1], "
                  "SubhaloMajorProgenitorMassRatio = %g in [0, 1].\n", sp->SubhaloSourceRelaxFactor, sp->SubhaloTreeOpenAngle,
               sp->SubhaloBoundMassPrecision, sp->SubhaloMajorProgenitorMassRatio);
    if(sp->SubhaloMinNumPart < 1)
        endrun(1, "SubhaloMinNumPart = %d must be at least 1.\n", sp->SubhaloMinNumPart);
    if(!subhalo_isfinite(sp->SubhaloMajorMergerRatio) || sp->SubhaloMajorMergerRatio < 0)
        endrun(1, "SubhaloMajorMergerRatio = %g must be finite and non-negative.\n", sp->SubhaloMajorMergerRatio);
    /* Later-phase knobs: only their finiteness, so that a typo cannot reach any arithmetic. */
    const double later[5] = {sp->SubhaloMajorProgenitorMassRatio, sp->SubhaloBoundMassPrecision,
                             sp->SubhaloSourceRelaxFactor, sp->SubhaloTreeOpenAngle, sp->SubhaloSOSearchFactor};
    int k;
    for(k = 0; k < 5; k++)
        if(!subhalo_isfinite(later[k]))
            endrun(1, "A floating-point Subhalo* parameter is not finite (%g).\n", later[k]);
}

/* Arena bytes per particle of a serial key (subhalo_internal.h, J7), from the validated
 * SubhaloSourceRelaxFactor; set with the parameters. */
static size_t SubArenaBytesPerPart = SUBHALO_ARENA_FIXED_BYTES_PER_PART + 24;

size_t
subhalo_arena_bytes_per_part(void)
{
    return SubArenaBytesPerPart;
}

static void
subhalo_set_arena_bytes(const struct SubhaloParams * sp)
{
    double R = sp->SubhaloSourceRelaxFactor;
    if(!(R >= 1)) R = 1;
    if(R > SUBHALO_MAX_RELAX_FACTOR) R = SUBHALO_MAX_RELAX_FACTOR;
    SubArenaBytesPerPart = SUBHALO_ARENA_FIXED_BYTES_PER_PART + (size_t) ceil(8. * R);
}

/* A double parameter that must hold a non-negative integer, checked before the conversion. */
static int64_t
subhalo_int_param(double v, const char * name)
{
    if(!subhalo_isfinite(v) || v < 0 || v > SUBHALO_INT_PARAM_MAX || v != floor(v))
        endrun(1, "%s = %g must be a non-negative integer below %g.\n", name, v, SUBHALO_INT_PARAM_MAX);
    return (int64_t) v;
}

void
set_subhalo_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    int checks[2] = {0, 0};
    /* the two integer-valued doubles, converted on every rank after validation */
    double raw[2] = {0, 0};
    /* the treewalk's import buffer factor (the memory checks of the field SO and the in-place walk) */
    double boost = 2;
    if(ThisTask == 0) {
        boost = param_get_double(ps, "ImportBufferBoost");
        SubParams.SubhaloFinderOn = param_get_int(ps, "SubhaloFinderOn");
        SubParams.SubhaloMaxGatherFraction = param_get_double(ps, "SubhaloMaxGatherFraction");
        raw[0] = param_get_double(ps, "SubhaloCollectiveMinPart");
        SubParams.SubhaloMaxUnbindIterations = param_get_int(ps, "SubhaloMaxUnbindIterations");
        SubParams.SubhaloRestartFresh = param_get_int(ps, "SubhaloRestartFresh");
        SubParams.SubhaloTracerTypes = param_get_int(ps, "SubhaloTracerTypes");
        SubParams.SubhaloPotentialTypes = param_get_int(ps, "SubhaloPotentialTypes");
        SubParams.SubhaloMinNumPart = param_get_int(ps, "SubhaloMinNumPart");
        SubParams.SubhaloMinNumTracerPart = param_get_int(ps, "SubhaloMinNumTracerPart");
        SubParams.SubhaloNumTracerHostFinding = param_get_int(ps, "SubhaloNumTracerHostFinding");
        raw[1] = param_get_double(ps, "SubhaloLookupBatchBytes");
        SubParams.SubhaloMajorProgenitorMassRatio = param_get_double(ps, "SubhaloMajorProgenitorMassRatio");
        SubParams.SubhaloBoundMassPrecision = param_get_double(ps, "SubhaloBoundMassPrecision");
        SubParams.SubhaloSourceRelaxFactor = param_get_double(ps, "SubhaloSourceRelaxFactor");
        SubParams.SubhaloPotentialSampleSize = param_get_int(ps, "SubhaloPotentialSampleSize");
        SubParams.SubhaloFinalPass = param_get_int(ps, "SubhaloFinalPass");
        SubParams.SubhaloTreeOpenAngle = param_get_double(ps, "SubhaloTreeOpenAngle");
        SubParams.SubhaloMergeTrapped = param_get_int(ps, "SubhaloMergeTrapped");
        SubParams.SubhaloSinkSymmetric = param_get_int(ps, "SubhaloSinkSymmetric");
        SubParams.SubhaloCoreSize = param_get_int(ps, "SubhaloCoreSize");
        SubParams.SubhaloReassignGas = param_get_int(ps, "SubhaloReassignGas");
        SubParams.SubhaloNumNeighboursForReassignment = param_get_int(ps, "SubhaloNumNeighboursForReassignment");
        SubParams.SubhaloUnbindWithThermal = param_get_int(ps, "SubhaloUnbindWithThermal");
        SubParams.SubhaloFieldSO = param_get_int(ps, "SubhaloFieldSO");
        SubParams.SubhaloSOBins = param_get_int(ps, "SubhaloSOBins");
        SubParams.SubhaloSOSearchFactor = param_get_double(ps, "SubhaloSOSearchFactor");
        SubParams.SubhaloSOMaxDoublings = param_get_int(ps, "SubhaloSOMaxDoublings");
        SubParams.SubhaloSORefineMinPart = param_get_int(ps, "SubhaloSORefineMinPart");
        SubParams.SubhaloMajorMergerRatio = param_get_double(ps, "SubhaloMajorMergerRatio");
        SubParams.SubhaloGeometricParents = param_get_int(ps, "SubhaloGeometricParents");
        SubParams.SubhaloDebugForceExchangeFailure = param_get_int(ps, "SubhaloDebugForceExchangeFailure");
        checks[0] = param_get_int(ps, "SnapshotWithFOF");
        checks[1] = param_get_int(ps, "FOFSaveParticles");
        param_get_string2(ps, "FOFFileBase", SubhaloFOFFileBase, sizeof(SubhaloFOFFileBase));
    }
    MPI_Bcast(&SubParams, sizeof(struct SubhaloParams), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(checks, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(raw, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&boost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    subhalo_so_set_import_buffer_boost(boost);
    subhalo_collective_set_import_buffer_boost(boost);
    MPI_Bcast(SubhaloFOFFileBase, sizeof(SubhaloFOFFileBase), MPI_CHAR, 0, MPI_COMM_WORLD);

    if(!SubParams.SubhaloFinderOn) {
        /* unused while off; keep the conversion defined */
        SubParams.SubhaloCollectiveMinPart = 0;
        SubParams.SubhaloLookupBatchBytes = 64L * 1024L * 1024L;
        return;
    }
    if(checks[0] != 1)
        endrun(1, "SubhaloFinderOn = 1 requires SnapshotWithFOF = 1 (the finder runs at every PART/PIG output).\n");
    if(checks[1] != 1)
        endrun(1, "SubhaloFinderOn = 1 requires FOFSaveParticles = 1 (the PIG particle blocks are the finder's restart state).\n");
    SubParams.SubhaloCollectiveMinPart = subhalo_int_param(raw[0], "SubhaloCollectiveMinPart");
    SubParams.SubhaloLookupBatchBytes = subhalo_int_param(raw[1], "SubhaloLookupBatchBytes");
    subhalo_validate_params(&SubParams);
    subhalo_set_arena_bytes(&SubParams);
}

void
subhalo_set_testpar(struct SubhaloParams params)
{
    if(params.SubhaloFinderOn)
        subhalo_validate_params(&params);
    SubParams = params;
    subhalo_set_arena_bytes(&SubParams);
}

void
subhalo_set_units(double UnitLength_in_cm, double UnitMass_in_g, double UnitVelocity_in_cm_per_s)
{
    if(!subhalo_isfinite(UnitLength_in_cm) || UnitLength_in_cm <= 0 ||
       !subhalo_isfinite(UnitMass_in_g) || UnitMass_in_g <= 0 ||
       !subhalo_isfinite(UnitVelocity_in_cm_per_s) || UnitVelocity_in_cm_per_s <= 0)
        endrun(1, "Subhalo finder: invalid unit system (%g cm, %g g, %g cm/s).\n",
               UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s);
    SubUnits.UnitLength_in_cm = UnitLength_in_cm;
    SubUnits.UnitMass_in_g = UnitMass_in_g;
    SubUnits.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s;
}

void
subhalo_get_units(double * UnitLength_in_cm, double * UnitMass_in_g, double * UnitVelocity_in_cm_per_s)
{
    *UnitLength_in_cm = SubUnits.UnitLength_in_cm;
    *UnitMass_in_g = SubUnits.UnitMass_in_g;
    *UnitVelocity_in_cm_per_s = SubUnits.UnitVelocity_in_cm_per_s;
}

double
subhalo_vel_to_kms(double atime)
{
    /* internal Vel = a * v_pec (whatever UsePeculiarVelocity, which only affects the snapshot
     * convention): physical peculiar velocity in cm/s is Vel * UnitVelocity / a */
    return SubUnits.UnitVelocity_in_cm_per_s / (1e5 * atime);
}

int
subhalo_finder_on(void)
{
    return SubParams.SubhaloFinderOn;
}

const struct SubhaloParams *
subhalo_get_params(void)
{
    return &SubParams;
}

/* ------------------------------------------------------------------ records */

static void
fill_float(float * x, int n, float v)
{
    int i;
    for(i = 0; i < n; i++)
        x[i] = v;
}

static void
fill_double(double * x, int n, double v)
{
    int i;
    for(i = 0; i < n; i++)
        x[i] = v;
}

void
subhalo_track_clear(struct SubTrack * t, int64_t TrackId)
{
    int k;
    memset(t, 0, sizeof(struct SubTrack));
    /* group A */
    t->TrackId = TrackId;
    t->HostGroup = -1;
    t->NestedParentTrackId = -1;
    t->CentralTrackId = -1;
    t->SinkTrackId = -1;
    t->DescendantTrackId = -1;
    t->Rank = t->Depth = -1;
    t->SnapBirth = t->SnapDeath = t->SnapSink = t->SnapLastIsolation = -1;
    t->MostBoundID = (MyIDType) -1;
    /* counts: zero means "no bound particles" */
    t->Nbound = t->NboundExclusive = t->NboundTracer = 0;
    t->Flags = 0;
    for(k = 0; k < 6; k++) {
        t->NboundType[k] = t->NboundExclusiveType[k] = t->NtailType[k] = 0;
        /* slices: no particles */
        t->SliceOffsetByType[k] = -1;
        t->SliceLengthByType[k] = t->SliceNboundByType[k] = 0;
        t->FieldSliceOffsetByType[k] = -1;
        t->FieldSliceLengthByType[k] = t->FieldSliceNboundByType[k] = 0;
    }
    /* group B: masses are counts-like (0), every other property absent (-1) */
    t->Mbound = 0;
    fill_float(t->MboundType, 6, 0);
    fill_double(t->CorePos, 3, -1);
    fill_double(t->MostBoundPos, 3, -1);
    fill_double(t->CoMPos, 3, -1);
    fill_float(t->CoreVel, 3, -1);
    fill_float(t->MostBoundVel, 3, -1);
    fill_float(t->CoMVel, 3, -1);
    t->CoreSigmaX = t->CoreSigmaV = -1;
    t->Vmax = t->Rmax = t->VmaxVir = t->RmaxVir = t->Rhalf = t->Renclose = -1;
    fill_float(t->MBoundSO, SUBHALO_NSO, -1);
    fill_float(t->RBoundSO, SUBHALO_NSO, -1);
    fill_float(t->J, 3, -1);
    fill_float(t->JVir, 3, -1);
    t->SpinPeebles = t->SpinBullock = t->Ekin = t->Epot = t->KinToPot = t->Xoff = t->Voff = t->RsKlypin = -1;
    t->BtoA = t->CtoA = t->BtoA500 = t->CtoA500 = -1;
    fill_float(t->AxisA, 3, -1);
    fill_float(t->AxisA500, 3, -1);
    fill_float(t->InertiaTensor, 6, -1);
    t->Rs = -1;
    fill_float(t->RockstarCoreVel, 3, -1);
    fill_float(t->BulkVel, 3, -1);
    t->Vrms = -1;
    /* group C */
    t->StarFormationRate = t->GasMetalMass = t->StellarMetalMass = -1;
    t->BlackholeMass = t->BlackholeAccretionRate = t->MassHeIonized = -1;
    fill_float(t->GasMetalElemMass, NMETALS, -1);
    fill_float(t->StellarMetalElemMass, NMETALS, -1);
    t->MostMassiveBHID = (MyIDType) -1;
    /* group D */
    fill_float(t->MSO, SUBHALO_NSO, -1);
    fill_float(t->RSO, SUBHALO_NSO, -1);
    fill_float(t->MTypeSO200c, 6, -1);
    fill_float(t->MTypeSOVir, 6, -1);
    t->RsSO = t->RsKlypinSO = t->VmaxSO = t->RmaxSO = -1;
    /* group E */
    t->MboundPeak = t->VmaxAtMboundPeak = t->VmaxPeak = -1;
    t->MboundAcc = t->VmaxAcc = t->MboundFirstAcc = t->VmaxFirstAcc = -1;
    t->SnapMboundPeak = t->SnapVmaxPeak = t->SnapAcc = t->SnapFirstAcc = t->SnapLastMajorMerger = -1;
    t->GeomParentTrackId = t->GeomUpParentTrackId = -1;
    /* group F */
    fill_double(t->TracerPos, 3, -1);
    fill_float(t->TracerVel, 3, -1);
    t->SnapTracerLost = -1;
    /* transient */
    t->Host = -1;
    t->NboundPrev = 0;
    t->Alive = 0;
    t->NPrev = 0;
    t->MPrev = 0;
    t->NTracers = 0;
    t->Key = -1;
}

void
subhalo_track_make_orphan(struct SubTrack * t, int64_t winner, int SnapNum)
{
    int k;
    /* The tracer starts at the last most-bound particle (HBT keeps following it). */
    for(k = 0; k < 3; k++) {
        t->TracerPos[k] = t->MostBoundPos[k];
        t->TracerVel[k] = t->MostBoundVel[k];
    }
    if(winner >= 0 || SUBHALO_IS_PROVISIONAL(winner)) {
        t->SinkTrackId = winner;
        t->SnapSink = SnapNum;
    }
    t->DescendantTrackId = winner;
    t->SnapDeath = SnapNum;
    t->Nbound = t->NboundExclusive = t->NboundTracer = 0;
    t->Mbound = 0;
    for(k = 0; k < 6; k++) {
        t->NboundType[k] = t->NboundExclusiveType[k] = t->NtailType[k] = 0;
        t->MboundType[k] = 0;
    }
    t->Alive = 0;
}

/* Step 13 (design 7.7), the part that needs only the previous record (rec carries the previous
 * history columns and the new state). */
void
subhalo_history_update(struct SubTrack * rec, int SnapNum, double MaxMergerMbound)
{
    if(rec->Mbound > rec->MboundPeak) {
        rec->MboundPeak = rec->Mbound;
        rec->SnapMboundPeak = SnapNum;
        rec->VmaxAtMboundPeak = rec->Vmax;
    }
    if(rec->Vmax > rec->VmaxPeak) {
        rec->VmaxPeak = rec->Vmax;
        rec->SnapVmaxPeak = SnapNum;
    }
    const int satellite = rec->Rank > 0 && rec->HostGroup >= 0;
    if(satellite) {
        /* the first time the track is a satellite: its state at the last output as a central
         * (the accretion columns of the previous record), never reset */
        if(rec->SnapFirstAcc < 0) {
            rec->SnapFirstAcc = rec->SnapAcc;
            rec->MboundFirstAcc = rec->MboundAcc;
            rec->VmaxFirstAcc = rec->VmaxAcc;
        }
    }
    else {
        /* central or hostless: state at the last output as a central */
        rec->MboundAcc = rec->Mbound;
        rec->VmaxAcc = rec->Vmax;
        rec->SnapAcc = SnapNum;
    }
    /* SnapLastIsolation follows HBT (subhalo_tracking.cpp, SnapshotOfLastIsolation): it stays -1
     * while the track has only ever been a central; when the track first becomes a satellite it
     * is set to the previous snapshot; afterwards it moves to the current snapshot whenever the
     * track is a central again. Rank 0 is a central, a hostless track counts as a central. */
    if(satellite) {
        if(rec->SnapLastIsolation < 0)
            rec->SnapLastIsolation = SnapNum - 1;
    } else if(rec->SnapLastIsolation >= 0)
        rec->SnapLastIsolation = SnapNum;
    /* a central that a track with Mbound > M * MM / (1 + MM) joined (as a new satellite or by
     * sinking) this call */
    const double MM = SubParams.SubhaloMajorMergerRatio;
    if(!satellite && MaxMergerMbound > 0 && MaxMergerMbound > rec->Mbound * MM / (1 + MM))
        rec->SnapLastMajorMerger = SnapNum;
}

/* ------------------------------------------------------------------ shared MPI helper */

void
subhalo_check_mpi_count(int64_t value, const char * what, MPI_Comm Comm)
{
    if(MPIU_Any(value < 0 || value > INT_MAX, Comm))
        endrun(1, "Subhalo finder: %s exceeds the MPI int limit on some rank (here %ld). "
                  "Phase 1 does not batch this exchange; run with more ranks.\n", what, value);
}

int64_t
subhalo_alltoallv(const void * sendbuf, const int64_t * sendcounts, size_t elsize, void ** recvbuf, int top, int64_t * recvcounts_out, MPI_Comm Comm)
{
    int NTask, i;
    MPI_Comm_size(Comm, &NTask);
    int64_t * recvcounts64 = ta_malloc("SubRecvCounts64", int64_t, NTask);
    MPI_Alltoall((void *) sendcounts, 1, MPI_INT64, recvcounts64, 1, MPI_INT64, Comm);
    int64_t nsend = 0, nrecv = 0;
    int bad = 0;
    for(i = 0; i < NTask; i++) {
        if(sendcounts[i] < 0 || recvcounts64[i] < 0)
            bad = 1;
        nsend += sendcounts[i];
        nrecv += recvcounts64[i];
    }
    /* every count and displacement is bounded by the totals */
    if(nsend > INT_MAX || nrecv > INT_MAX)
        bad = 1;
    if(MPIU_Any(bad, Comm))
        endrun(1, "Subhalo exchange of %ld / %ld elements of %lu bytes exceeds the MPI int counts on some rank. "
                  "Phase 1 does not batch this exchange; run with more ranks.\n", nsend, nrecv, (unsigned long) elsize);

    int * scounts = ta_malloc("SubSendCounts", int, NTask);
    int * recvcounts = ta_malloc("SubRecvCounts", int, NTask);
    int * sdispls = ta_malloc("SubSendDispls", int, NTask);
    int * rdispls = ta_malloc("SubRecvDispls", int, NTask);
    int64_t so = 0, ro = 0;
    for(i = 0; i < NTask; i++) {
        scounts[i] = (int) sendcounts[i];
        recvcounts[i] = (int) recvcounts64[i];
        sdispls[i] = (int) so;
        rdispls[i] = (int) ro;
        so += sendcounts[i];
        ro += recvcounts64[i];
    }

    char * recv;
    if(top)
        recv = (char *) mymalloc2("SubhaloRecv", elsize * nrecv + 1);
    else
        recv = (char *) mymalloc("SubhaloRecv", elsize * nrecv + 1);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);
    MPI_Alltoallv_sparse((void *) sendbuf, scounts, sdispls, dtype, recv, recvcounts, rdispls, dtype, Comm);
    MPI_Type_free(&dtype);

    if(recvcounts_out)
        memcpy(recvcounts_out, recvcounts64, sizeof(int64_t) * NTask);
    ta_free(rdispls);
    ta_free(sdispls);
    ta_free(recvcounts);
    ta_free(scounts);
    ta_free(recvcounts64);
    *recvbuf = recv;
    return nrecv;
}

/* ------------------------------------------------------------------ batched ID lookup (steps 1, 11) */

struct IdIndex {
    MyIDType ID;
    int64_t i;
};

static int
cmp_idindex(const void * a, const void * b)
{
    const struct IdIndex * x = (const struct IdIndex *) a, * y = (const struct IdIndex *) b;
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->i > y->i) - (x->i < y->i);
}

/* A hit returned to the asking rank: the position of the id in that rank's batch. */
struct IdReply {
    int64_t q;
    struct SubhaloIdHit h;
};

/* Test knob: at most this many ids per rank and batch (0: the byte cap alone). */
static int64_t SubLookupMaxBatch = 0;
/* High-water mark of the batch buffers (gathered ids, outgoing and returning replies) of the
 * last subhalo_lookup_ids call on this rank. */
static double SubLookupLastPeak = 0;

void
subhalo_lookup_set_max_batch(int64_t maxbatch)
{
    SubLookupMaxBatch = maxbatch > 0 ? maxbatch : 0;
}

double
subhalo_lookup_last_peak_bytes(void)
{
    return SubLookupLastPeak;
}

/* Ids per rank and batch (J4): one batch gathers at most NTask * cap ids on every rank, a rank
 * owning every one of them replies to all (NTask * cap replies out) and receives the replies to
 * its own cap queries, so the batch buffers peak at
 *   NTask * cap * (sizeof(MyIDType) + sizeof(struct IdReply)) + cap * sizeof(struct IdReply)
 * on any rank whatever the pattern (live IDs are unique, init.c). cap is the largest value for
 * which this stays within SubhaloLookupBatchBytes, at least one id per rank (the floor: a batch
 * then holds NTask * 88 + 80 bytes with the 64-bit layout). */
static int64_t
lookup_cap(int NTask)
{
    const double per = (double) NTask * (double) (sizeof(MyIDType) + sizeof(struct IdReply)) + (double) sizeof(struct IdReply);
    const int64_t bb = SubParams.SubhaloLookupBatchBytes > 0 ? SubParams.SubhaloLookupBatchBytes : 64L * 1024L * 1024L;
    int64_t cap = (int64_t) ((double) bb / per);
    if(cap > INT_MAX / NTask)
        cap = INT_MAX / NTask;
    if(SubLookupMaxBatch > 0 && cap > SubLookupMaxBatch)
        cap = SubLookupMaxBatch;
    return cap > 0 ? cap : 1;
}

double
subhalo_lookup_batch_peak_bytes(int NTask)
{
    const int64_t cap = lookup_cap(NTask);
    return (double) NTask * cap * (double) (sizeof(MyIDType) + sizeof(struct IdReply)) + (double) cap * sizeof(struct IdReply)
        + 3 * 4096.;
}

double
subhalo_lookup_index_bytes(int64_t NumPart)
{
    /* the (ID, index) index and the direct merge-sort copy of qsort_openmp */
    return 2. * sizeof(struct IdIndex) * (double) (NumPart + 1) + 4096.;
}

/* Phase 5d: how a rank answers a batch: 0 by the batch size (a scan of the particles against
 * the sorted batch when the batch is small against the particles, else the sorted ID index),
 * 1 always the index, 2 always the scan (test knob). The answers are the same: the live
 * particle with the ID, the lowest index in P if an ID were duplicated. */
static int SubLookupMode = 0;

void
subhalo_lookup_set_mode(int mode)
{
    SubLookupMode = mode;
}

struct LookupQuery {
    MyIDType ID;
    int64_t pos;
};

static int
cmp_lookupquery(const void * a, const void * b)
{
    const struct LookupQuery * x = (const struct LookupQuery *) a, * y = (const struct LookupQuery *) b;
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->pos > y->pos) - (x->pos < y->pos);
}

/* The sorted (ID, index) index over the local live particles, in P order before the sort
 * (threaded, order-preserving). */
static struct IdIndex *
lookup_build_index(int64_t * nlout)
{
    const int64_t N = PartManager->NumPart;
    const int nc = subhalo_par_nchunks(N);
    int64_t * off = ta_malloc("SubLookupOff", int64_t, nc + 1);
    int c;
    off[0] = 0;
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t cnt = 0, i;
        for(i = c * N / nc; i < (c + 1) * N / nc; i++)
            if(!P[i].IsGarbage && !P[i].Swallowed)
                cnt++;
        off[c + 1] = cnt;
    }
    for(c = 0; c < nc; c++)
        off[c + 1] += off[c];
    const int64_t nl = off[nc];
    struct IdIndex * idx = (struct IdIndex *) mymalloc("SubIdIndex", sizeof(struct IdIndex) * (nl + 1));
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t m = off[c], i;
        for(i = c * N / nc; i < (c + 1) * N / nc; i++)
            if(!P[i].IsGarbage && !P[i].Swallowed) {
                idx[m].ID = P[i].ID;
                idx[m].i = i;
                m++;
            }
    }
    ta_free(off);
    qsort_openmp(idx, nl, sizeof(struct IdIndex), cmp_idindex);
    *nlout = nl;
    return idx;
}

/* all[0 .. tot) on entry: the gathered ids; on exit: the index in P of the live particle with
 * that id (the lowest if duplicated), or (MyIDType) -1. */
static void
lookup_answer_scan(MyIDType * all, int64_t tot)
{
    struct LookupQuery * qs = (struct LookupQuery *) mymalloc("SubLookupQs", sizeof(struct LookupQuery) * (tot + 1));
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < tot; i++) {
        qs[i].ID = all[i];
        qs[i].pos = i;
        all[i] = (MyIDType) -1;
    }
    qsort_openmp(qs, tot, sizeof(struct LookupQuery), cmp_lookupquery);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        const MyIDType id = P[i].ID;
        int64_t lo = 0, hi = tot;
        while(lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if(qs[mid].ID < id) lo = mid + 1;
            else hi = mid;
        }
        for(; lo < tot && qs[lo].ID == id; lo++) {
            MyIDType * slot = &all[qs[lo].pos];
            MyIDType old = __atomic_load_n(slot, __ATOMIC_RELAXED);
            while((MyIDType) i < old && !__atomic_compare_exchange_n(slot, &old, (MyIDType) i, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                ;
        }
    }
    myfree(qs);
}

void
subhalo_lookup_ids(const MyIDType * ids, int64_t n, struct SubhaloIdHit * out, MPI_Comm Comm)
{
    int NTask, r;
    MPI_Comm_size(Comm, &NTask);
    int64_t i;
    memset(out, 0, sizeof(struct SubhaloIdHit) * n);

    /* the live particles; the sorted (ID, index) index is built at the first batch that uses it */
    int64_t nlive = 0;
    #pragma omp parallel for reduction(+: nlive)
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && !P[i].Swallowed)
            nlive++;
    /* the batches of this call answer by scan when all queries together are few against the
     * live particles (every batch then is); otherwise the index is built once, here */
    int64_t ntot = 0;
    MPI_Allreduce(&n, &ntot, 1, MPI_INT64, MPI_SUM, Comm);
    const int scan = SubLookupMode == 2 || (SubLookupMode == 0 && ntot <= nlive / 16);
    struct IdIndex * idx = NULL;
    int64_t nl = 0;
    if(!scan)
        idx = lookup_build_index(&nl);

    /* batches: every rank contributes at most cap ids, so that the ids and both reply buffers
     * of a batch stay within SubhaloLookupBatchBytes on every rank (lookup_cap) */
    const int64_t cap = lookup_cap(NTask);
    SubLookupLastPeak = 0;
    int * counts = ta_malloc("SubLookupCounts", int, NTask);
    int * displs = ta_malloc("SubLookupDispls", int, NTask);
    int64_t * sendcounts = ta_malloc("SubLookupSend", int64_t, NTask);
    MPI_Datatype idtype;
    MPI_Type_contiguous(sizeof(MyIDType), MPI_BYTE, &idtype);
    MPI_Type_commit(&idtype);
    int64_t done = 0;
    while(1) {
        const int64_t nb = n - done < cap ? n - done : cap;
        if(!MPIU_Any(nb > 0, Comm))
            break;
        int nbi = (int) nb;
        MPI_Allgather(&nbi, 1, MPI_INT, counts, 1, MPI_INT, Comm);
        int64_t tot = 0;
        for(r = 0; r < NTask; r++) {
            displs[r] = (int) tot;
            tot += counts[r];
        }
        MyIDType * all = (MyIDType *) mymalloc("SubLookupIds", sizeof(MyIDType) * (tot + 1));
        MPI_Allgatherv((void *) (ids + done), nbi, idtype, all, counts, displs, idtype, Comm);
        /* the answering particle of every gathered id: a scan of the particles against the
         * sorted batch (O(N log tot), no index), or the index */
        if(scan)
            lookup_answer_scan(all, tot);
        else {
            #pragma omp parallel for
            for(i = 0; i < tot; i++) {
                const MyIDType id = all[i];
                int64_t lo = 0, hi = nl;
                while(lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if(idx[mid].ID < id) lo = mid + 1;
                    else hi = mid;
                }
                all[i] = (lo < nl && idx[lo].ID == id) ? (MyIDType) idx[lo].i : (MyIDType) -1;
            }
        }
        /* hits for every asking rank, in rank order */
        int64_t nhit = 0;
        memset(sendcounts, 0, sizeof(int64_t) * NTask);
        for(r = 0; r < NTask; r++)
            for(i = displs[r]; i < displs[r] + counts[r]; i++)
                if(all[i] != (MyIDType) -1) {
                    sendcounts[r]++;
                    nhit++;
                }
        struct IdReply * rep = (struct IdReply *) mymalloc("SubLookupRep", sizeof(struct IdReply) * (nhit + 1));
        nhit = 0;
        for(r = 0; r < NTask; r++)
            for(i = displs[r]; i < displs[r] + counts[r]; i++) {
                if(all[i] == (MyIDType) -1)
                    continue;
                const struct particle_data * p = &P[all[i]];
                struct IdReply * e = &rep[nhit++];
                memset(e, 0, sizeof(*e));
                e->q = i - displs[r];
                e->h.GrNr = p->GrNr >= 0 ? p->GrNr : -1;
                e->h.BoundTrackId = p->BoundTrackId;
                int d;
                for(d = 0; d < 3; d++) {
                    e->h.Pos[d] = p->Pos[d];
                    e->h.Vel[d] = p->Vel[d];
                }
                e->h.found = 1;
            }
        struct IdReply * back = NULL;
        int64_t nback = subhalo_alltoallv(rep, sendcounts, sizeof(struct IdReply), (void **) &back, 0, NULL, Comm);
        const double peak = (double) (tot + 1) * sizeof(MyIDType) + (double) (nhit + 1) * sizeof(struct IdReply)
            + (double) (nback + 1) * sizeof(struct IdReply);
        if(peak > SubLookupLastPeak)
            SubLookupLastPeak = peak;
        /* an ID lives on one rank; if it were duplicated the last answer wins */
        for(i = 0; i < nback; i++)
            out[done + back[i].q] = back[i].h;
        myfree(back);
        myfree(rep);
        myfree(all);
        done += nb;
    }
    MPI_Type_free(&idtype);
    ta_free(sendcounts);
    ta_free(displs);
    ta_free(counts);
    if(idx)
        myfree(idx);
}

static double
wrap_pos(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

void
subhalo_orphan_lookup(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nlook, int64_t * nfound, MPI_Comm Comm)
{
    int64_t i, n = 0;
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * t = &tab->Track[i];
        if(t->NboundPrev <= 0 && t->SnapTracerLost < 0 && t->MostBoundID != (MyIDType) -1)
            n++;
    }
    MyIDType * ids = (MyIDType *) mymalloc("SubOrphanIds", sizeof(MyIDType) * (n + 1));
    int64_t * which = (int64_t *) mymalloc("SubOrphanWhich", sizeof(int64_t) * (n + 1));
    struct SubhaloIdHit * hit = (struct SubhaloIdHit *) mymalloc("SubOrphanHits", sizeof(struct SubhaloIdHit) * (n + 1));
    n = 0;
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * t = &tab->Track[i];
        if(t->NboundPrev <= 0 && t->SnapTracerLost < 0 && t->MostBoundID != (MyIDType) -1) {
            ids[n] = t->MostBoundID;
            which[n] = i;
            n++;
        }
    }
    subhalo_lookup_ids(ids, n, hit, Comm);
    const double vfac = subhalo_vel_to_kms(atime);
    const double BoxSize = PartManager->BoxSize;
    int64_t nf = 0;
    for(i = 0; i < n; i++) {
        struct SubTrack * t = &tab->Track[which[i]];
        int d;
        if(hit[i].found) {
            /* HBT's one-tracer vote (subhalo_tracking.cpp:231-236) and the tracer state
             * (subhalo_update_mostbound.cpp:20-70), physical frame and km/s */
            t->Host = hit[i].GrNr;
            for(d = 0; d < 3; d++) {
                t->TracerPos[d] = wrap_pos(hit[i].Pos[d] - PartManager->CurrentParticleOffset[d], BoxSize);
                t->TracerVel[d] = hit[i].Vel[d] * vfac;
            }
            nf++;
        }
        else {
            /* the particle no longer exists: the record keeps its last position */
            t->Host = -1;
            t->SnapTracerLost = SnapNum;
        }
    }
    myfree(hit);
    myfree(which);
    myfree(ids);
    int64_t loc[2] = {n, nf}, tot[2];
    MPI_Allreduce(loc, tot, 2, MPI_INT64, MPI_SUM, Comm);
    *nlook = tot[0];
    *nfound = tot[1];
}

/* ------------------------------------------------------------------ the track table */

/* The table block alternates between the bottom and the top of its allocator, so that a growth
 * (fresh block, copy, free the old one) always frees the last block of its direction. */
static int
table_block_on_bottom(const struct SubTrackTable * tab)
{
    return tab->alloc->bottom > 0;
}

static struct SubTrack *
table_alloc_other_side(struct SubTrackTable * tab, int64_t n)
{
    size_t bytes = sizeof(struct SubTrack) * (n > 0 ? n : 1);
    if(table_block_on_bottom(tab))
        return (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTracks", bytes);
    return (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTracks", bytes);
}

void
subhalo_table_init(struct SubTrackTable * tab, size_t bytes, MPI_Comm Comm)
{
    memset(tab, 0, sizeof(struct SubTrackTable));
    MPI_Comm_size(Comm, &tab->NTask);
    MPI_Comm_rank(Comm, &tab->ThisTask);
    if(0 != allocator_malloc_init(tab->alloc, "SUBTRACK", bytes, 0, NULL))
        endrun(1, "Failed to create the subhalo track allocator.\n");
    tab->NAlloc = bytes / sizeof(struct SubTrack);
    if(tab->NAlloc < 1)
        tab->NAlloc = 1;
    tab->Track = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTracks", sizeof(struct SubTrack) * tab->NAlloc);
    tab->NLocal = 0;
    tab->NTotal = 0;
    tab->Chunk = 1;
    tab->initialised = 1;
}

void
subhalo_table_free(struct SubTrackTable * tab)
{
    if(!tab->initialised)
        return;
    allocator_free(tab->Track);
    allocator_destroy(tab->alloc);
    tab->Track = NULL;
    tab->NLocal = tab->NAlloc = tab->NTotal = 0;
    tab->Chunk = 1;
    tab->initialised = 0;
}

struct SubTrack *
subhalo_table_get(struct SubTrackTable * tab, int64_t TrackId)
{
    if(TrackId < 0 || TrackId >= tab->NTotal)
        return NULL;
    if(subhalo_table_home_rank(tab, TrackId) != tab->ThisTask)
        return NULL;
    int64_t i = subhalo_table_local_index(tab, TrackId);
    if(i >= tab->NLocal)
        return NULL;
    return &tab->Track[i];
}

void
subhalo_table_reserve(struct SubTrackTable * tab, int64_t n)
{
    if(n <= tab->NAlloc)
        return;
    int64_t newalloc = tab->NAlloc;
    while(newalloc < n)
        newalloc *= 2;
    struct SubTrack * newtrack = table_alloc_other_side(tab, newalloc);
    memcpy(newtrack, tab->Track, sizeof(struct SubTrack) * tab->NLocal);
    allocator_free(tab->Track);
    tab->Track = newtrack;
    tab->NAlloc = newalloc;
}

static int64_t
table_chunk(int64_t NTotal, int NTask)
{
    int64_t c = (NTotal + NTask - 1) / NTask;
    return c > 0 ? c : 1;
}

static int64_t
table_expected_local(int64_t NTotal, int64_t Chunk, int ThisTask)
{
    int64_t n = NTotal - (int64_t) ThisTask * Chunk;
    if(n < 0) n = 0;
    if(n > Chunk) n = Chunk;
    return n;
}

/* Generic: whatever the current placement (home ranks under an older Chunk, or the contiguous
 * row slices of a restart read), every record moves to its home under the new Chunk and every
 * homed TrackId without a record gets a cleared one. The exchange is skipped when every rank
 * already holds exactly its home records. */
void
subhalo_table_redistribute(struct SubTrackTable * tab, MPI_Comm Comm)
{
    int64_t i;
    const int64_t NewChunk = table_chunk(tab->NTotal, tab->NTask);
    const int64_t expected = table_expected_local(tab->NTotal, NewChunk, tab->ThisTask);
    const int64_t first = (int64_t) tab->ThisTask * NewChunk;

    /* Fast path: the records here are exactly a prefix of this rank's home range; the new
     * TrackIds (created this call) are appended as cleared records. */
    int ok = tab->NLocal <= expected;
    for(i = 0; ok && i < tab->NLocal; i++)
        if(tab->Track[i].TrackId != first + i)
            ok = 0;
    /* Every existing record is checked, so a passing check means no record has to move;
     * the homed TrackIds not held yet are the tracks created since the last call. */
    if(!MPIU_Any(!ok, Comm)) {
        subhalo_table_reserve(tab, expected);
        for(i = tab->NLocal; i < expected; i++)
            subhalo_track_clear(&tab->Track[i], first + i);
        tab->NLocal = expected;
        tab->Chunk = NewChunk;
        return;
    }

    /* Full exchange of the records to their new homes. Counts and offsets in 64 bits; the
     * MPI ints are checked collectively before narrowing. */
    int64_t * sendcounts = ta_malloc("SubTabSendCounts", int64_t, tab->NTask);
    memset(sendcounts, 0, sizeof(int64_t) * tab->NTask);
    int bad = 0;
    for(i = 0; i < tab->NLocal; i++) {
        const int64_t T = tab->Track[i].TrackId;
        if(T < 0 || T >= tab->NTotal) {
            bad = 1;
            continue;
        }
        sendcounts[T / NewChunk]++;
    }
    if(bad)
        endrun(5, "Track table on rank %d holds a TrackId outside [0, %ld).\n", tab->ThisTask, tab->NTotal);
    int64_t * recvcounts = ta_malloc("SubTabRecvCounts", int64_t, tab->NTask);
    MPI_Alltoall(sendcounts, 1, MPI_INT64, recvcounts, 1, MPI_INT64, Comm);
    int64_t nrecv = 0, nsend = 0;
    int r;
    for(r = 0; r < tab->NTask; r++) {
        nsend += sendcounts[r];
        nrecv += recvcounts[r];
    }
    subhalo_check_mpi_count(nsend > nrecv ? nsend : nrecv, "the track table redistribution", Comm);

    /* Send buffer in the table allocator, on the same side as the current block. */
    const int onbottom = table_block_on_bottom(tab);
    struct SubTrack * sendbuf;
    if(onbottom)
        sendbuf = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTrackSend", sizeof(struct SubTrack) * (tab->NLocal + 1));
    else
        sendbuf = (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTrackSend", sizeof(struct SubTrack) * (tab->NLocal + 1));
    int64_t * offset = ta_malloc("SubTabOffset", int64_t, tab->NTask);
    offset[0] = 0;
    for(r = 1; r < tab->NTask; r++)
        offset[r] = offset[r-1] + sendcounts[r-1];
    for(i = 0; i < tab->NLocal; i++) {
        const int64_t T = tab->Track[i].TrackId;
        sendbuf[offset[T / NewChunk]++] = tab->Track[i];
    }
    ta_free(offset);

    /* The new block on the other side. */
    int64_t newalloc = tab->NAlloc;
    while(newalloc < expected)
        newalloc *= 2;
    struct SubTrack * newtrack;
    if(onbottom)
        newtrack = (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTracks", sizeof(struct SubTrack) * newalloc);
    else
        newtrack = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTracks", sizeof(struct SubTrack) * newalloc);
    for(i = 0; i < expected; i++)
        subhalo_track_clear(&newtrack[i], first + i);

    /* Receive buffer on the same side as the new block, after it (freed first). */
    int * scounts = ta_malloc("SubTabSendCounts32", int, tab->NTask);
    int * rcounts = ta_malloc("SubTabRecvCounts32", int, tab->NTask);
    int * sdispls = ta_malloc("SubTabSendDispls", int, tab->NTask);
    int * rdispls = ta_malloc("SubTabRecvDispls", int, tab->NTask);
    int64_t so = 0, ro = 0;
    for(r = 0; r < tab->NTask; r++) {
        scounts[r] = (int) sendcounts[r];
        rcounts[r] = (int) recvcounts[r];
        sdispls[r] = (int) so;
        rdispls[r] = (int) ro;
        so += sendcounts[r];
        ro += recvcounts[r];
    }
    struct SubTrack * recvbuf;
    if(onbottom)
        recvbuf = (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTrackRecv", sizeof(struct SubTrack) * (nrecv + 1));
    else
        recvbuf = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTrackRecv", sizeof(struct SubTrack) * (nrecv + 1));
    MPI_Datatype dtype;
    MPI_Type_contiguous(sizeof(struct SubTrack), MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);
    MPI_Alltoallv_sparse(sendbuf, scounts, sdispls, dtype, recvbuf, rcounts, rdispls, dtype, Comm);
    MPI_Type_free(&dtype);

    int64_t nplaced = 0;
    for(i = 0; i < nrecv; i++) {
        const int64_t li = recvbuf[i].TrackId - first;
        if(li < 0 || li >= expected)
            endrun(5, "Track %ld received on rank %d outside its home range.\n", recvbuf[i].TrackId, tab->ThisTask);
        newtrack[li] = recvbuf[i];
        nplaced++;
    }
    allocator_free(recvbuf);
    ta_free(rdispls);
    ta_free(sdispls);
    ta_free(rcounts);
    ta_free(scounts);
    allocator_free(sendbuf);
    allocator_free(tab->Track);
    ta_free(recvcounts);
    ta_free(sendcounts);

    tab->Track = newtrack;
    tab->NAlloc = newalloc;
    tab->NLocal = expected;
    tab->Chunk = NewChunk;
    int64_t nplaced_tot = 0;
    MPI_Allreduce(&nplaced, &nplaced_tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Track table redistributed: %ld tracks (%ld records moved or kept), chunk %ld.\n", tab->NTotal, nplaced_tot, tab->Chunk);
}

/* ---- batched record fetch and store (H1) ----
 * Requests are grouped by home rank through a permutation (8 bytes per request, the only
 * allocation proportional to n). Each batch sends at most PairCap requests to every home
 * rank, PairCap = max(1, SubhaloLookupBatchBytes / (NTask * (sizeof(SubTrack) + 8))), so that
 * one batch moves at most max(SubhaloLookupBatchBytes, NTask * (sizeof(SubTrack) + 8)) bytes
 * into and out of any rank whatever the request pattern. The batches are collective: the loop
 * runs while any rank has requests left. subhalo_table_batch_peak_bytes is the bound the plan
 * reserves for the buffers of one batch. */
static int64_t
table_pair_cap(int NTask)
{
    const int64_t per = (int64_t) sizeof(struct SubTrack) + 8;
    const int64_t bb = subhalo_get_params()->SubhaloLookupBatchBytes > 0 ? subhalo_get_params()->SubhaloLookupBatchBytes : 64L * 1024L * 1024L;
    int64_t cap = bb / ((int64_t) NTask * per);
    return cap > 0 ? cap : 1;
}

double
subhalo_table_batch_peak_bytes(int NTask)
{
    const double per = (double) sizeof(struct SubTrack) + 8;
    /* requests and replies of one batch: at most PairCap * NTask of each in and out */
    return 2. * (double) table_pair_cap(NTask) * NTask * per + 4096. * 4;
}

/* Group n items by home rank: perm[j] is the item at grouped position j; cnt/off per rank. */
static void
table_group_by_home(struct SubTrackTable * tab, const int64_t * ids, const struct SubTrack * recs, int64_t n,
                    int64_t * perm, int64_t * cnt, int64_t * off)
{
    int64_t i;
    int r;
    memset(cnt, 0, sizeof(int64_t) * tab->NTask);
    for(i = 0; i < n; i++)
        cnt[subhalo_table_home_rank(tab, ids ? ids[i] : recs[i].TrackId)]++;
    off[0] = 0;
    for(r = 1; r < tab->NTask; r++)
        off[r] = off[r-1] + cnt[r-1];
    int64_t * fill = ta_malloc("SubTabFill", int64_t, tab->NTask);
    memcpy(fill, off, sizeof(int64_t) * tab->NTask);
    for(i = 0; i < n; i++)
        perm[fill[subhalo_table_home_rank(tab, ids ? ids[i] : recs[i].TrackId)]++] = i;
    ta_free(fill);
}

void
subhalo_table_fetch(struct SubTrackTable * tab, const int64_t * ids, int64_t n, struct SubTrack * out, MPI_Comm Comm)
{
    int64_t i;
    int r;
    const int NTask = tab->NTask;
    int bad = 0;
    for(i = 0; i < n; i++)
        if(ids[i] < 0 || ids[i] >= tab->NTotal)
            bad = 1;
    if(bad)
        endrun(5, "subhalo_table_fetch: TrackId outside [0, %ld) on rank %d.\n", tab->NTotal, tab->ThisTask);

    int64_t * cnt = ta_malloc("SubFetchCnt", int64_t, NTask);
    int64_t * off = ta_malloc("SubFetchOff", int64_t, NTask);
    int64_t * done = ta_malloc("SubFetchDone", int64_t, NTask);
    int64_t * sendcounts = ta_malloc("SubFetchSend", int64_t, NTask);
    int64_t * recvcounts = ta_malloc("SubFetchRecv", int64_t, NTask);
    int64_t * perm = (int64_t *) mymalloc("SubFetchPerm", sizeof(int64_t) * (n + 1));
    table_group_by_home(tab, ids, NULL, n, perm, cnt, off);
    memset(done, 0, sizeof(int64_t) * NTask);
    const int64_t cap = table_pair_cap(NTask);

    while(1) {
        int64_t nb = 0;
        for(r = 0; r < NTask; r++) {
            sendcounts[r] = cnt[r] - done[r] < cap ? cnt[r] - done[r] : cap;
            nb += sendcounts[r];
        }
        if(!MPIU_Any(nb > 0, Comm))
            break;
        int64_t * reqids = (int64_t *) mymalloc("SubFetchIds", sizeof(int64_t) * (nb + 1));
        int64_t j = 0;
        for(r = 0; r < NTask; r++) {
            int64_t k;
            for(k = 0; k < sendcounts[r]; k++)
                reqids[j++] = ids[perm[off[r] + done[r] + k]];
        }
        /* request */
        int64_t * inids = NULL;
        int64_t nin = subhalo_alltoallv(reqids, sendcounts, sizeof(int64_t), (void **) &inids, 0, recvcounts, Comm);
        struct SubTrack * reply = (struct SubTrack *) mymalloc("SubFetchReply", sizeof(struct SubTrack) * (nin + 1));
        int missing = 0;
        for(i = 0; i < nin; i++) {
            struct SubTrack * rec = subhalo_table_get(tab, inids[i]);
            if(!rec) {
                missing = 1;
                subhalo_track_clear(&reply[i], inids[i]);
                continue;
            }
            reply[i] = *rec;
        }
        if(missing)
            endrun(5, "subhalo_table_fetch: a requested track is not homed on rank %d.\n", tab->ThisTask);
        /* reply: what we received from r goes back to r */
        struct SubTrack * got = NULL;
        int64_t ngot = subhalo_alltoallv(reply, recvcounts, sizeof(struct SubTrack), (void **) &got, 0, NULL, Comm);
        if(ngot != nb)
            endrun(5, "subhalo_table_fetch: got %ld records for %ld requests.\n", ngot, nb);
        j = 0;
        for(r = 0; r < NTask; r++) {
            int64_t k;
            for(k = 0; k < sendcounts[r]; k++)
                out[perm[off[r] + done[r] + k]] = got[j++];
            done[r] += sendcounts[r];
        }
        myfree(got);
        myfree(reply);
        myfree(inids);
        myfree(reqids);
    }
    myfree(perm);
    ta_free(recvcounts);
    ta_free(sendcounts);
    ta_free(done);
    ta_free(off);
    ta_free(cnt);
}

void
subhalo_table_store(struct SubTrackTable * tab, const struct SubTrack * recs, int64_t n, MPI_Comm Comm)
{
    int64_t i;
    int r;
    const int NTask = tab->NTask;
    int bad = 0;
    for(i = 0; i < n; i++)
        if(recs[i].TrackId < 0 || recs[i].TrackId >= tab->NTotal)
            bad = 1;
    if(bad)
        endrun(5, "subhalo_table_store: TrackId outside [0, %ld) on rank %d.\n", tab->NTotal, tab->ThisTask);

    int64_t * cnt = ta_malloc("SubStoreCnt", int64_t, NTask);
    int64_t * off = ta_malloc("SubStoreOff", int64_t, NTask);
    int64_t * done = ta_malloc("SubStoreDone", int64_t, NTask);
    int64_t * sendcounts = ta_malloc("SubStoreSend", int64_t, NTask);
    int64_t * perm = (int64_t *) mymalloc("SubStorePerm", sizeof(int64_t) * (n + 1));
    table_group_by_home(tab, NULL, recs, n, perm, cnt, off);
    memset(done, 0, sizeof(int64_t) * NTask);
    const int64_t cap = table_pair_cap(NTask);
    int missing = 0;
    while(1) {
        int64_t nb = 0;
        for(r = 0; r < NTask; r++) {
            sendcounts[r] = cnt[r] - done[r] < cap ? cnt[r] - done[r] : cap;
            nb += sendcounts[r];
        }
        if(!MPIU_Any(nb > 0, Comm))
            break;
        struct SubTrack * sendbuf = (struct SubTrack *) mymalloc("SubStoreSend", sizeof(struct SubTrack) * (nb + 1));
        int64_t j = 0;
        for(r = 0; r < NTask; r++) {
            int64_t k;
            for(k = 0; k < sendcounts[r]; k++)
                sendbuf[j++] = recs[perm[off[r] + done[r] + k]];
            done[r] += sendcounts[r];
        }
        struct SubTrack * in = NULL;
        int64_t nin = subhalo_alltoallv(sendbuf, sendcounts, sizeof(struct SubTrack), (void **) &in, 0, NULL, Comm);
        /* duplicates (identical in-place copies of a member) overwrite each other */
        for(i = 0; i < nin; i++) {
            struct SubTrack * rec = subhalo_table_get(tab, in[i].TrackId);
            if(!rec) {
                missing = 1;
                continue;
            }
            *rec = in[i];
        }
        myfree(in);
        myfree(sendbuf);
    }
    if(missing)
        endrun(5, "subhalo_table_store: a stored track is not homed on rank %d (redistribute first).\n", tab->ThisTask);
    myfree(perm);
    ta_free(sendcounts);
    ta_free(done);
    ta_free(off);
    ta_free(cnt);
}

struct SubTrackTable *
subhalo_get_table(void)
{
    return &SubTable;
}

/* ------------------------------------------------------------------ status */

int subhalo_last_call_complete(void) { return SubStatus.LastCallComplete; }
int subhalo_last_call_snapshot(void) { return SubStatus.LastCallSnapshot; }
int subhalo_state_snapshot(void) { return SubStatus.StateSnapshot; }
double subhalo_state_time(void) { return SubStatus.StateTime; }
/* In units of the scale factor: a(this call) - a(previous completed call or loaded state). */
double subhalo_time_since_previous_call(void) { return SubStatus.TimeSincePrevious; }

void
subhalo_state_so_thresholds(double rho[SUBHALO_NSO])
{
    memcpy(rho, SubStatus.SOThresholds, sizeof(SubStatus.SOThresholds));
}

void
subhalo_set_state_so_thresholds(const double rho[SUBHALO_NSO])
{
    int d;
    for(d = 0; d < SUBHALO_NSO; d++)
        SubStatus.SOThresholds[d] = rho ? rho[d] : -1.;
}

void
subhalo_count_tagged(const struct part_manager_type * pman, int64_t ntagged[6])
{
    int64_t i;
    int64_t n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0;
    #pragma omp parallel for reduction(+: n0, n1, n2, n3, n4, n5)
    for(i = 0; i < pman->NumPart; i++) {
        const struct particle_data * p = &pman->Base[i];
        /* the live-particle predicate of the finder (H6): swallowed BHs keep their tags but
         * no longer count */
        if(p->IsGarbage || p->Swallowed || p->TrackId < 0)
            continue;
        switch(p->Type) {
            case 0: n0++; break;
            case 1: n1++; break;
            case 2: n2++; break;
            case 3: n3++; break;
            case 4: n4++; break;
            default: n5++; break;
        }
    }
    ntagged[0] = n0; ntagged[1] = n1; ntagged[2] = n2;
    ntagged[3] = n3; ntagged[4] = n4; ntagged[5] = n5;
}

static void
log_tagged(const char * when, MPI_Comm Comm)
{
    int64_t loc[6], tot[6];
    subhalo_count_tagged(PartManager, loc);
    MPI_Allreduce(loc, tot, 6, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo: tagged particles %s by type: %ld %ld %ld %ld %ld %ld\n", when,
            tot[0], tot[1], tot[2], tot[3], tot[4], tot[5]);
}

/* ------------------------------------------------------------------ init (restart, design 11) */

void
subhalo_init(int RestartSnapNum, int MaxStateSnapNum, const char * OutputDir, MPI_Comm Comm)
{
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        P[i].TrackId = -1;
        P[i].BindRank = -1;
        P[i].BoundTrackId = -1;
    }
    SubStatus.LastCallComplete = 0;
    SubStatus.LastCallSnapshot = -1;
    SubStatus.StateSnapshot = -1;
    SubStatus.StateTime = -1;
    SubStatus.TimeSincePrevious = -1;
    subhalo_set_state_so_thresholds(NULL);

    if(!subhalo_finder_on())
        return;

    if(SubTable.initialised)
        subhalo_table_free(&SubTable);
    subhalo_table_init(&SubTable, SUBHALO_TABLE_INITIAL_BYTES, Comm);

    if(RestartSnapNum < 0) {
        message(0, "Subhalo finder: fresh start from the initial conditions, empty track table.\n");
        return;
    }
    if(MaxStateSnapNum < 0) {
        /* Offline mode at snapshot 0: there is no earlier PIG, every group is newborn. */
        message(0, "Subhalo finder: no snapshot before %d can hold a state; starting with an empty track table.\n", RestartSnapNum);
        return;
    }

    const int m = subhalo_find_state(OutputDir, SubhaloFOFFileBase, MaxStateSnapNum, Comm);
    if(m < 0) {
        /* Nonzero code: endrun passes it to MPI_Abort, so mpirun exits nonzero and scripts can
         * detect the abort (every rank has the same m, so every rank aborts here). */
        if(!SubParams.SubhaloRestartFresh)
            endrun(1, "Subhalo finder: no complete state in %s/%s_xxx with number <= %d. "
                      "Set SubhaloRestartFresh = 1 to restart with an empty track table (TrackIds restart at 0).\n",
                      OutputDir, SubhaloFOFFileBase, MaxStateSnapNum);
        message(0, "WARNING: Subhalo finder: no complete state with number <= %d; SubhaloRestartFresh = 1, "
                   "starting with an empty track table. TrackIds restart at 0.\n", MaxStateSnapNum);
        return;
    }
    if(m < MaxStateSnapNum)
        message(0, "WARNING: Subhalo finder: the latest complete state is PIG %d, older than %d. "
                   "The tags describe the membership at %d and the catalogues in between are missing.\n",
                   m, MaxStateSnapNum, m);

    char * pigname = fastpm_strdup_printf("%s/%s_%03d", OutputDir, SubhaloFOFFileBase, m);
    SubStatus.StateTime = subhalo_read_state(pigname, &SubTable, Comm);
    int64_t inherited = subhalo_join_tags(pigname, PartManager, Comm);
    myfree(pigname);
    SubStatus.StateSnapshot = m;
    SubStatus.LastCallSnapshot = m;

    int64_t inherited_tot = 0;
    MPI_Allreduce(&inherited, &inherited_tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo finder: state of PIG %d loaded (%ld tracks, a = %g); %ld particles inherited a tag from their lineage.\n",
            m, SubTable.NTotal, SubStatus.StateTime, inherited_tot);
    log_tagged("after the restart join", Comm);
}

/* ------------------------------------------------------------------ step 10: new TrackIds */

struct NewbornEntry {
    int64_t g;
    int64_t final;
};

static void
radix_newborn_g(const void * a, void * radix, void * arg)
{
    ((uint64_t *) radix)[0] = ((const struct NewbornEntry *) a)->g;
}

static int
cmp_int64(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

static int
cmp_newborn_g(const void * a, const void * b)
{
    const int64_t x = ((const struct NewbornEntry *) a)->g, y = ((const struct NewbornEntry *) b)->g;
    return (x > y) - (x < y);
}

static int64_t
newborn_lookup(const struct NewbornEntry * map, int64_t n, int64_t ordinal)
{
    struct NewbornEntry key = {ordinal, -1};
    const struct NewbornEntry * hit = (const struct NewbornEntry *) bsearch(&key, map, n, sizeof(map[0]), cmp_newborn_g);
    return hit ? hit->final : SUBHALO_NOTRACK;
}

static int64_t
rewrite_id(int64_t T, const struct NewbornEntry * map, int64_t n)
{
    if(!SUBHALO_IS_PROVISIONAL(T))
        return T;
    int64_t f = newborn_lookup(map, n, SUBHALO_PROVISIONAL_ORDINAL(T));
    return f >= 0 ? f : T;
}

/* Numbers the newborn keys in host (GrNr) order and rewrites every provisional tag and record
 * reference. Returns the global number of newborns. Collective. */
static int64_t
subhalo_number_newborns(struct SubhaloResult * res, int64_t nres, int64_t TotNgroups, int64_t NTotalBefore, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t i;

    int64_t nlocal = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid && res[i].newborn)
            nlocal++;
    int64_t ntot = 0;
    MPI_Allreduce(&nlocal, &ntot, 1, MPI_INT64, MPI_SUM, Comm);
    if(ntot == 0)
        return 0;

    /* 1. global host order: mpsort on g, then an exclusive scan of the sorted counts */
    struct NewbornEntry * ent = (struct NewbornEntry *) mymalloc("SubNewborn", sizeof(struct NewbornEntry) * (nlocal + 1));
    int64_t n = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid && res[i].newborn) {
            ent[n].g = SUBHALO_PROVISIONAL_ORDINAL(res[i].rec.TrackId);
            ent[n].final = -1;
            n++;
        }
    mpsort_mpi(ent, nlocal, sizeof(struct NewbornEntry), radix_newborn_g, 8, NULL, Comm);
    int64_t offset = 0;
    MPI_Exscan(&nlocal, &offset, 1, MPI_INT64, MPI_SUM, Comm);
    if(ThisTask == 0)
        offset = 0;
    for(i = 0; i < nlocal; i++) {
        ent[i].final = NTotalBefore + offset + i;
        if(i > 0 && ent[i].g <= ent[i-1].g)
            endrun(5, "Newborn key %ld appears twice.\n", ent[i].g);
    }

    /* 2. store the (ordinal -> final) pairs on the key directory rank of g */
    const int64_t KChunk = subhalo_key_chunk(TotNgroups, NTask);
    int64_t * sendcounts = ta_malloc("SubNbCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nlocal; i++)
        sendcounts[subhalo_key_dir_rank(ent[i].g, KChunk)]++;
    /* ent is sorted by g, hence by directory rank */
    struct NewbornEntry * dir = NULL;
    int64_t ndir = subhalo_alltoallv(ent, sendcounts, sizeof(struct NewbornEntry), (void **) &dir, 0, NULL, Comm);
    qsort(dir, ndir, sizeof(struct NewbornEntry), cmp_newborn_g);

    /* 3. every rank asks for the ordinals it holds: particle tags and its records */
    /* the particles' provisional tags: threaded, order-preserving (chunk counts, offsets, fills) */
    const int64_t NumPart = PartManager->NumPart;
    const int nc = subhalo_par_nchunks(NumPart);
    int64_t * off = ta_malloc("SubNbOff", int64_t, nc + 1);
    int c;
    off[0] = 0;
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t cnt = 0, k;
        for(k = c * NumPart / nc; k < (c + 1) * NumPart / nc; k++) {
            if(P[k].IsGarbage) continue;
            if(SUBHALO_IS_PROVISIONAL(P[k].TrackId)) cnt++;
            if(SUBHALO_IS_PROVISIONAL(P[k].BoundTrackId)) cnt++;
        }
        off[c + 1] = cnt;
    }
    for(c = 0; c < nc; c++)
        off[c + 1] += off[c];
    int64_t nreq = off[nc];
    int j;
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        if(SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            nreq++;
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                nreq++;
    }
    int64_t * req = (int64_t *) mymalloc("SubNbReq", sizeof(int64_t) * (nreq + 1));
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t m = off[c], k;
        for(k = c * NumPart / nc; k < (c + 1) * NumPart / nc; k++) {
            if(P[k].IsGarbage) continue;
            if(SUBHALO_IS_PROVISIONAL(P[k].TrackId)) req[m++] = SUBHALO_PROVISIONAL_ORDINAL(P[k].TrackId);
            if(SUBHALO_IS_PROVISIONAL(P[k].BoundTrackId)) req[m++] = SUBHALO_PROVISIONAL_ORDINAL(P[k].BoundTrackId);
        }
    }
    nreq = off[nc];
    ta_free(off);
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        if(SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(res[i].rec.TrackId);
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(res[i].DescOwner[j]);
    }
    qsort_openmp(req, nreq, sizeof(int64_t), cmp_int64);
    int64_t nuniq = 0;
    for(i = 0; i < nreq; i++)
        if(nuniq == 0 || req[i] != req[nuniq-1])
            req[nuniq++] = req[i];
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    int badord = 0;
    for(i = 0; i < nuniq; i++) {
        if(req[i] < 1 || req[i] > TotNgroups) {
            badord = 1;
            continue;
        }
        sendcounts[subhalo_key_dir_rank(req[i], KChunk)]++;
    }
    if(badord)
        endrun(5, "A provisional TrackId names a key outside [1, %ld].\n", TotNgroups);
    int64_t * recvcounts = ta_malloc("SubNbRecvCounts", int64_t, NTask);
    int64_t * asked = NULL;
    int64_t nasked = subhalo_alltoallv(req, sendcounts, sizeof(int64_t), (void **) &asked, 0, recvcounts, Comm);
    struct NewbornEntry * answer = (struct NewbornEntry *) mymalloc("SubNbAnswer", sizeof(struct NewbornEntry) * (nasked + 1));
    for(i = 0; i < nasked; i++) {
        answer[i].g = asked[i];
        answer[i].final = newborn_lookup(dir, ndir, asked[i]);
    }
    struct NewbornEntry * map = NULL;
    int64_t nmap = subhalo_alltoallv(answer, recvcounts, sizeof(struct NewbornEntry), (void **) &map, 0, NULL, Comm);
    if(nmap != nuniq)
        endrun(5, "Newborn lookup returned %ld answers for %ld requests.\n", nmap, nuniq);
    /* answers come back in request order, which is sorted by g */

    /* 4. rewrite */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        P[i].TrackId = rewrite_id(P[i].TrackId, map, nmap);
        P[i].BoundTrackId = rewrite_id(P[i].BoundTrackId, map, nmap);
    }
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        struct SubTrack * t = &res[i].rec;
        t->TrackId = rewrite_id(t->TrackId, map, nmap);
        t->NestedParentTrackId = rewrite_id(t->NestedParentTrackId, map, nmap);
        t->CentralTrackId = rewrite_id(t->CentralTrackId, map, nmap);
        t->SinkTrackId = rewrite_id(t->SinkTrackId, map, nmap);
        t->DescendantTrackId = rewrite_id(t->DescendantTrackId, map, nmap);
        for(j = 0; j < res[i].ndesc; j++)
            res[i].DescOwner[j] = rewrite_id(res[i].DescOwner[j], map, nmap);
        res[i].newborn = 0;
    }

    /* 5. nothing provisional may survive */
    int left = 0;
    #pragma omp parallel for reduction(|: left)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        if(SUBHALO_IS_PROVISIONAL(P[i].TrackId) || SUBHALO_IS_PROVISIONAL(P[i].BoundTrackId))
            left = 1;
    }
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        const struct SubTrack * t = &res[i].rec;
        if(SUBHALO_IS_PROVISIONAL(t->TrackId) || SUBHALO_IS_PROVISIONAL(t->NestedParentTrackId) ||
           SUBHALO_IS_PROVISIONAL(t->CentralTrackId) || SUBHALO_IS_PROVISIONAL(t->SinkTrackId) ||
           SUBHALO_IS_PROVISIONAL(t->DescendantTrackId))
            left = 1;
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                left = 1;
    }
    if(MPIU_Any(left, Comm))
        endrun(5, "Provisional TrackIds left after the newborn numbering.\n");

    myfree(map);
    myfree(answer);
    myfree(asked);
    ta_free(recvcounts);
    myfree(req);
    myfree(dir);
    ta_free(sendcounts);
    myfree(ent);
    return ntot;
}

/* ------------------------------------------------------------------ step 7f: hostless and dead tracks */

/* The live-particle predicate of the finder (H6): garbage and swallowed particles neither vote
 * nor count, but their tags are left untouched. */
static inline int
subhalo_live(const struct particle_data * p)
{
    return !p->IsGarbage && !p->Swallowed;
}

/* The common periodic reference of a hostless track (H7): the position of its globally
 * minimum-ID live field particle, found through the home rank. */
struct FieldRef {
    int64_t TrackId;
    MyIDType ID;
    double Pos[3];
};

/* Partial sums of a track's own-tag field particles on one rank, positions as nearest-image
 * displacements from the track's common reference. */
struct FieldSum {
    int64_t TrackId;
    int64_t count;
    int64_t ntracer;
    int64_t LenType[6];
    double MassType[6];
    double mass;
    double ref[3];      /* the common reference (identical in every contribution) */
    double mdx[3];      /* sum m * NEAREST(x - ref) */
    double mv[3];       /* sum m * Vel */
    /* the most-bound particle of the fallback (J3): the lowest (not a tracer type, BindRank, ID) */
    int64_t mbkey[3];
    MyIDType mbID;
    double mbPos[3], mbVel[3];
};

static int
fieldsum_mb_less(const int64_t a[3], const int64_t b[3])
{
    if(a[0] != b[0]) return a[0] < b[0];
    if(a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

static int
cmp_fieldsum(const void * a, const void * b)
{
    const int64_t x = ((const struct FieldSum *) a)->TrackId, y = ((const struct FieldSum *) b)->TrackId;
    return (x > y) - (x < y);
}

static int
cmp_fieldref(const void * a, const void * b)
{
    const struct FieldRef * x = (const struct FieldRef *) a, * y = (const struct FieldRef *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

struct TagIndex {
    int64_t TrackId;
    int64_t i;
};

static int
cmp_tagindex(const void * a, const void * b)
{
    const struct TagIndex * x = (const struct TagIndex *) a, * y = (const struct TagIndex *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    /* ID order makes the local sums independent of the particle order */
    const MyIDType ix = P[x->i].ID, iy = P[y->i].ID;
    return (ix > iy) - (ix < iy);
}

static double
wrap_box(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* Stable merge sort by TrackId: contributions of one track keep the source-rank order of the
 * Alltoallv receive buffer, so the home rank sums them in rank order. */
static void
fieldsum_stable_sort(struct FieldSum * in, int64_t nin)
{
    struct FieldSum * tmp = (struct FieldSum *) mymalloc("SubFieldTmp", sizeof(struct FieldSum) * (nin + 1));
    int64_t width;
    for(width = 1; width < nin; width *= 2) {
        int64_t lo;
        for(lo = 0; lo < nin; lo += 2 * width) {
            int64_t mid = lo + width < nin ? lo + width : nin;
            int64_t hi = lo + 2 * width < nin ? lo + 2 * width : nin;
            int64_t a = lo, b = mid, o = lo;
            while(a < mid && b < hi) {
                if(cmp_fieldsum(&in[b], &in[a]) < 0) tmp[o++] = in[b++];
                else tmp[o++] = in[a++];
            }
            while(a < mid) tmp[o++] = in[a++];
            while(b < hi) tmp[o++] = in[b++];
        }
        memcpy(in, tmp, sizeof(struct FieldSum) * nin);
    }
    myfree(tmp);
}

/* Collective. Returns the global numbers of hostless and dead tracks.
 * Centre of a hostless track: one common reference per track (the minimum-ID live field
 * particle, H7), mass-weighted nearest-image displacements from it summed per rank in ID order,
 * then over ranks in rank order on the home rank. The result does not depend on the rank
 * partition beyond floating-point summation order: bit-level rank-count independence is not
 * claimed for these sums. */
void
subhalo_update_unhosted(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nhostless, int64_t * ndead, MPI_Comm Comm)
{
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i, j;
    int k;
    const double BoxSize = PartManager->BoxSize;
    const int TracerTypes = SubParams.SubhaloTracerTypes;
    const double vfac = subhalo_vel_to_kms(atime);

    /* own-tag live field particles, grouped by track */
#define UNH_PRED (subhalo_live(&P[i]) && P[i].GrNr < 0 && P[i].TrackId >= 0)
    const int nc = subhalo_par_nchunks(PartManager->NumPart);
    int64_t * off = ta_malloc("SubFieldOff", int64_t, nc + 1);
    SUBHALO_COUNT_CHUNKS(PartManager->NumPart, nc, off, UNH_PRED);
    int64_t nf = off[nc];
    struct TagIndex * idx = (struct TagIndex *) mymalloc("SubFieldIdx", sizeof(struct TagIndex) * (nf + 1));
    SUBHALO_FILL_CHUNKS(PartManager->NumPart, nc, off, UNH_PRED, {
            idx[m].TrackId = P[i].TrackId;
            idx[m].i = i;
        });
#undef UNH_PRED
    ta_free(off);
    qsort_openmp(idx, nf, sizeof(struct TagIndex), cmp_tagindex);
    /* the first entry of every track (threaded, order-preserving) */
#define TRACK_START(i) ((i) == 0 || idx[i].TrackId != idx[(i)-1].TrackId)
    const int ncb = subhalo_par_nchunks(nf);
    int64_t * offb = ta_malloc("SubFieldOffB", int64_t, ncb + 1);
    SUBHALO_COUNT_CHUNKS(nf, ncb, offb, TRACK_START(i));
    const int64_t ns = offb[ncb];
    int64_t * tstart = (int64_t *) mymalloc("SubFieldStart", sizeof(int64_t) * (ns + 1));
    SUBHALO_FILL_CHUNKS(nf, ncb, offb, TRACK_START(i), tstart[m] = i);
#undef TRACK_START
    tstart[ns] = nf;
    ta_free(offb);

    /* 1. the common reference: local minimum-ID particle of each track to the home rank, the
     *    global minimum back to every holder (request/reply in the same order) */
    struct FieldRef * myref = (struct FieldRef *) mymalloc("SubFieldRef", sizeof(struct FieldRef) * (ns + 1));
    int64_t s;
    #pragma omp parallel for private(k)
    for(s = 0; s < ns; s++) {
        const int64_t i0 = tstart[s];
        myref[s].TrackId = idx[i0].TrackId;
        myref[s].ID = P[idx[i0].i].ID;
        for(k = 0; k < 3; k++)
            myref[s].Pos[k] = P[idx[i0].i].Pos[k];
    }
    int64_t * sendcounts = ta_malloc("SubFieldCounts", int64_t, NTask);
    int64_t * recvcounts = ta_malloc("SubFieldRecvCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    /* sorted by TrackId, hence by home rank */
    for(j = 0; j < ns; j++)
        sendcounts[subhalo_table_home_rank(tab, myref[j].TrackId)]++;
    {
        struct FieldRef * inref = NULL;
        int64_t nin = subhalo_alltoallv(myref, sendcounts, sizeof(struct FieldRef), (void **) &inref, 0, recvcounts, Comm);
        struct FieldRef * minref = (struct FieldRef *) mymalloc("SubFieldMinRef", sizeof(struct FieldRef) * (nin + 1));
        memcpy(minref, inref, sizeof(struct FieldRef) * nin);
        qsort(minref, nin, sizeof(struct FieldRef), cmp_fieldref);
        /* keep the first (minimum-ID) entry of each track */
        int64_t nmin = 0;
        for(i = 0; i < nin; i++)
            if(nmin == 0 || minref[i].TrackId != minref[nmin-1].TrackId)
                minref[nmin++] = minref[i];
        for(i = 0; i < nin; i++) {
            struct FieldRef key = {inref[i].TrackId, 0, {0, 0, 0}};
            /* binary search on TrackId alone */
            int64_t lo = 0, hi = nmin;
            while(lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if(minref[mid].TrackId < key.TrackId) lo = mid + 1;
                else hi = mid;
            }
            if(lo >= nmin || minref[lo].TrackId != key.TrackId)
                endrun(5, "Hostless reference of track %ld lost on its home rank.\n", key.TrackId);
            inref[i] = minref[lo];
        }
        myfree(minref);
        struct FieldRef * back = NULL;
        int64_t nback = subhalo_alltoallv(inref, recvcounts, sizeof(struct FieldRef), (void **) &back, 0, NULL, Comm);
        if(nback != ns)
            endrun(5, "Hostless reference exchange returned %ld entries for %ld tracks.\n", nback, ns);
        for(j = 0; j < ns; j++) {
            if(back[j].TrackId != myref[j].TrackId)
                endrun(5, "Hostless reference exchange out of order (%ld for %ld).\n", back[j].TrackId, myref[j].TrackId);
            myref[j] = back[j];
        }
        myfree(back);
        myfree(inref);
    }

    /* 2. local sums relative to the common reference, in ID order */
    struct FieldSum * sums = (struct FieldSum *) mymalloc("SubFieldSums", sizeof(struct FieldSum) * (ns + 1));
    /* per track in ID order, as before; the tracks in parallel */
    #pragma omp parallel for private(i, k)
    for(s = 0; s < ns; s++) {
      memset(&sums[s], 0, sizeof(struct FieldSum));
      sums[s].TrackId = idx[tstart[s]].TrackId;
      for(k = 0; k < 3; k++)
          sums[s].ref[k] = myref[s].Pos[k];
      for(i = tstart[s]; i < tstart[s + 1]; i++) {
        const struct particle_data * p = &P[idx[i].i];
        const double m = p->Mass;
        sums[s].count++;
        sums[s].LenType[p->Type]++;
        sums[s].MassType[p->Type] += m;
        sums[s].mass += m;
        if(TracerTypes & (1 << p->Type))
            sums[s].ntracer++;
        for(k = 0; k < 3; k++) {
            sums[s].mdx[k] += m * NEAREST(p->Pos[k] - sums[s].ref[k], BoxSize);
            sums[s].mv[k] += m * p->Vel[k];
        }
        const int64_t mk[3] = {(TracerTypes & (1 << p->Type)) ? 0 : 1, p->BindRank, (int64_t) p->ID};
        if(sums[s].count == 1 || fieldsum_mb_less(mk, sums[s].mbkey)) {
            for(k = 0; k < 3; k++) {
                sums[s].mbkey[k] = mk[k];
                sums[s].mbPos[k] = p->Pos[k];
                sums[s].mbVel[k] = p->Vel[k];
            }
            sums[s].mbID = p->ID;
        }
      }
    }
    struct FieldSum * in = NULL;
    int64_t nin = subhalo_alltoallv(sums, sendcounts, sizeof(struct FieldSum), (void **) &in, 0, NULL, Comm);
    fieldsum_stable_sort(in, nin);

    /* 3. the home ranks: hostless resolved tracks and dead tracks */
    int64_t nh = 0, nd = 0;
    int64_t a = 0;
    int toolarge = 0;
    for(j = 0; j < tab->NLocal; j++) {
        struct SubTrack * t = &tab->Track[j];
        while(a < nin && in[a].TrackId < t->TrackId)
            a++;
        int64_t b = a;
        while(b < nin && in[b].TrackId == t->TrackId)
            b++;
        if(t->Alive || t->SnapDeath >= 0) {
            a = b;
            continue;
        }
        if(b > a) {
            /* hostless resolved track whose pseudo-key was not processed (unreachable from
             * subhalo_find since Phase 2b: oversized pseudo-keys are unbound in place): keeps its
             * field particles, all bound; sums over ranks in rank order */
            int64_t count = 0, ntracer = 0, LenType[6] = {0};
            double MassType[6] = {0}, mass = 0, mdx[3] = {0}, mv[3] = {0};
            int64_t c, cmb = a;
            for(c = a; c < b; c++) {
                if(fieldsum_mb_less(in[c].mbkey, in[cmb].mbkey))
                    cmb = c;
                count += in[c].count;
                ntracer += in[c].ntracer;
                for(k = 0; k < 6; k++) {
                    LenType[k] += in[c].LenType[k];
                    MassType[k] += in[c].MassType[k];
                }
                mass += in[c].mass;
                for(k = 0; k < 3; k++) {
                    mdx[k] += in[c].mdx[k];
                    mv[k] += in[c].mv[k];
                }
            }
            /* the i4 count columns: abort rather than wrap (H4) */
            if(count > INT32_MAX || ntracer > INT32_MAX)
                toolarge = 1;
            /* identity and history carried, every property of the previous call cleared */
            {
                const struct SubTrack old = *t;
                subhalo_track_carry(t, &old, old.TrackId, SnapNum);
            }
            t->Flags = SUBHALO_FLAG_INPLACE;
            t->HostGroup = -1;
            t->Rank = t->Depth = 0;
            t->NestedParentTrackId = -1;
            t->CentralTrackId = t->TrackId;
            t->SinkTrackId = -1;
            t->DescendantTrackId = t->TrackId;
            t->Nbound = t->NboundExclusive = (int32_t) count;
            t->NboundTracer = (int32_t) ntracer;
            t->Mbound = mass;
            for(k = 0; k < 6; k++) {
                t->NboundType[k] = t->NboundExclusiveType[k] = (int32_t) LenType[k];
                t->NtailType[k] = 0;
                t->MboundType[k] = MassType[k];
            }
            for(k = 0; k < 3; k++) {
                const double cen = in[a].ref[k] + (mass > 0 ? mdx[k] / mass : 0);
                t->CorePos[k] = t->CoMPos[k] = wrap_box(cen - PartManager->CurrentParticleOffset[k], BoxSize);
                t->CoreVel[k] = t->CoMVel[k] = (mass > 0 ? mv[k] / mass : 0) * vfac;
            }
            /* MostBound*: the first tracer-type particle in rank order (the Phase 1 rule has no
             * binding energies) */
            t->MostBoundID = in[cmb].mbID;
            for(k = 0; k < 3; k++) {
                t->MostBoundPos[k] = wrap_box(in[cmb].mbPos[k] - PartManager->CurrentParticleOffset[k], BoxSize);
                t->MostBoundVel[k] = in[cmb].mbVel[k] * vfac;
            }
            t->Alive = 1;
            subhalo_history_update(t, SnapNum, 0);
            nh++;
            if(toolarge)
                endrun(1, "Hostless track %ld has %ld particles, more than the i4 count columns hold.\n", t->TrackId, count);
        }
        else if(t->NboundPrev > 0) {
            /* resolved at the previous call, no live particles anywhere now and not sunk: dead */
            subhalo_track_make_orphan(t, SUBHALO_NOTRACK, SnapNum);
            nd++;
        }
        a = b;
    }
    myfree(in);
    myfree(sums);
    ta_free(recvcounts);
    ta_free(sendcounts);
    myfree(myref);
    myfree(tstart);
    myfree(idx);

    MPI_Allreduce(&nh, nhostless, 1, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&nd, ndead, 1, MPI_INT64, MPI_SUM, Comm);
}

/* ------------------------------------------------------------------ step 11: remote descendants */

/* The tracers of dead or sunk tracks that were not in their key are looked up on the restored
 * layout; then every such track takes the majority of its tracers' BoundTrackId (ties to the
 * smaller TrackId; -1 if none is bound), merger_tree.cpp:143-181. Collective. */
static void
subhalo_finish_descendants(struct SubhaloResult * res, int64_t nres, MPI_Comm Comm)
{
    int64_t i, n = 0;
    int j;
    for(i = 0; i < nres; i++)
        if(res[i].valid)
            for(j = 0; j < res[i].ndesc; j++)
                if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING)
                    n++;
    MyIDType * ids = (MyIDType *) mymalloc("SubDescIds", sizeof(MyIDType) * (n + 1));
    struct SubhaloIdHit * hit = (struct SubhaloIdHit *) mymalloc("SubDescHits", sizeof(struct SubhaloIdHit) * (n + 1));
    n = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid)
            for(j = 0; j < res[i].ndesc; j++)
                if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING)
                    ids[n++] = res[i].rec.TracerIDs[j];
    subhalo_lookup_ids(ids, n, hit, Comm);
    n = 0;
    int64_t nremote = 0, nfound = 0, ndecided = 0;
    for(i = 0; i < nres; i++) {
        if(!res[i].valid || res[i].ndesc <= 0)
            continue;
        for(j = 0; j < res[i].ndesc; j++)
            if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING) {
                res[i].DescOwner[j] = hit[n].found ? hit[n].BoundTrackId : -1;
                nremote++;
                nfound += hit[n].found;
                n++;
            }
        int64_t best = -1, bestc = 0;
        int a, b;
        for(a = 0; a < res[i].ndesc; a++) {
            const int64_t o = res[i].DescOwner[a];
            if(o < 0)
                continue;
            int64_t c = 0;
            for(b = 0; b < res[i].ndesc; b++)
                if(res[i].DescOwner[b] == o)
                    c++;
            if(c > bestc || (c == bestc && o < best)) {
                bestc = c;
                best = o;
            }
        }
        res[i].rec.DescendantTrackId = best;
        ndecided++;
    }
    myfree(hit);
    myfree(ids);
    int64_t loc[3] = {ndecided, nremote, nfound}, tot[3];
    MPI_Allreduce(loc, tot, 3, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo descendants: %ld dead or sunk tracks; %ld tracers looked up outside their keys, %ld found.\n",
            tot[0], tot[1], tot[2]);
}

/* ------------------------------------------------------------------ orphan links (J8) */

/* After the store and the unhosted update, on the home records: every orphan's HostGroup is the
 * group of its tracer at this call (Host: the step-1 lookup, or the vote for a track that died
 * this call; -1 in the field or for a lost tracer), and its CentralTrackId the central of that
 * group: the winner of an in-place key (ipl, sorted by g), the central the serial key already
 * set, -1 in the field. Links fixed at death (parent, sink, descendant) are historical and
 * kept. Not collective. */
struct InplaceWinnerEntry {
    int64_t g;
    int64_t winner;
};

static int
cmp_iplwinner(const void * a, const void * b)
{
    const int64_t x = ((const struct InplaceWinnerEntry *) a)->g, y = ((const struct InplaceWinnerEntry *) b)->g;
    return (x > y) - (x < y);
}

static int64_t
subhalo_sync_orphan_links(struct SubTrackTable * tab, const struct InplaceWinnerEntry * ipl, int64_t nipl)
{
    int64_t i, nchanged = 0;
    for(i = 0; i < tab->NLocal; i++) {
        struct SubTrack * t = &tab->Track[i];
        if(t->Nbound > 0 || t->SnapDeath < 0)
            continue;
        const int64_t host = t->Host >= 0 ? t->Host : -1;
        if(t->HostGroup != host)
            nchanged++;
        t->HostGroup = host;
        if(host < 0) {
            t->CentralTrackId = -1;
            continue;
        }
        struct InplaceWinnerEntry key = {host, -1};
        const struct InplaceWinnerEntry * w = (const struct InplaceWinnerEntry *)
            bsearch(&key, ipl, nipl, sizeof(key), cmp_iplwinner);
        if(w)
            t->CentralTrackId = w->winner;
    }
    return nchanged;
}

/* ------------------------------------------------------------------ diagnostics */

static struct SubhaloCallStats SubLastStats;

const struct SubhaloCallStats *
subhalo_last_call_stats(void)
{
    return &SubLastStats;
}

/* Global totals of the per-rank statistics (the worst ratio with its TrackId). Collective. */
static void
stats_reduce(const struct SubhaloCallStats * loc, struct SubhaloCallStats * tot, MPI_Comm Comm)
{
    int64_t a[7] = {loc->NCandidates, loc->NItercap, loc->NFinalPositive, loc->NFinalBound, loc->NTailRejected,
                    loc->NFallbackField, loc->NFallbackReleased}, b[7];
    MPI_Allreduce(a, b, 7, MPI_INT64, MPI_SUM, Comm);
    memset(tot, 0, sizeof(*tot));
    tot->NCandidates = b[0];
    tot->NItercap = b[1];
    tot->NFinalPositive = b[2];
    tot->NFinalBound = b[3];
    tot->NTailRejected = b[4];
    tot->NFallbackField = b[5];
    tot->NFallbackReleased = b[6];
    MPI_Allreduce(&loc->ArenaPeakFrac, &tot->ArenaPeakFrac, 1, MPI_DOUBLE, MPI_MAX, Comm);
    struct { double r; int rank; } in, out;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    in.r = loc->NItercap > 0 ? loc->WorstConvRatio : 2.;
    in.rank = ThisTask;
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, Comm);
    tot->WorstConvRatio = out.r < 2. ? out.r : 1.;
    int64_t wt = loc->WorstTrackId;
    MPI_Bcast(&wt, 1, MPI_INT64, out.rank, Comm);
    tot->WorstTrackId = out.r < 2. ? wt : -1;
}

/* ------------------------------------------------------------------ per-stage instrumentation (Phase 5c) */

const char * const SubhaloStageName[SUBHALO_NSTAGE] = {
    "setup", "vote", "frames", "lookup", "plan", "exchange", "sort", "fetch", "serial", "restore",
    "rounds", "number", "descendants", "history", "store", "unhosted", "orphansync", "fieldso"};
/* The rounds stage is split by subhalo_collective_run into /Subhalo/Rounds/{Setup,Rounds,Tree,
 * Walk,Tags} (Tree: the tree builds of the potential-only walks, redirected from forcetree.c's own
 * /Misc and /Tree/Build/{Nodes,Moments} clocks; Walk: the walks); the driver's end of the stage
 * measures the rest as /Subhalo/Rounds/Cleanup (all of the stage when there is no in-place
 * candidate), so that the parent /Subhalo/Rounds, the sum of its children, is this rank's rounds
 * time of the summary (Phase 5 audit R3, decision 47.5) at any thread count. Likewise the field SO
 * stage: subhalo_field_so_full measures /Subhalo/FieldSO/{Setup,Tree} (Tree: its tree build,
 * redirected from forcetree.c's clocks) and the driver's end of the stage the rest, the walks, as
 * /Subhalo/FieldSO/Walks. */
static const char * const SubhaloStageClock[SUBHALO_NSTAGE] = {
    "/Subhalo/Setup", "/Subhalo/Vote", "/Subhalo/Frames", "/Subhalo/Lookup", "/Subhalo/Plan", "/Subhalo/Exchange",
    "/Subhalo/Sort", "/Subhalo/Fetch", "/Subhalo/Serial", "/Subhalo/Restore", "/Subhalo/Rounds/Cleanup", "/Subhalo/Number",
    "/Subhalo/Descendants", "/Subhalo/History", "/Subhalo/Store", "/Subhalo/Unhosted", "/Subhalo/OrphanSync",
    "/Subhalo/FieldSO/Walks"};

/* This rank's stage clock of the current call: wall time and main-allocator used high water per
 * stage (the high water from the allocator's low-water mark of the free bytes, reset at every
 * stage end; the collective resets it once more at its own start, inside the rounds stage). */
static struct {
    double tstart, tlast;
    double time[SUBHALO_NSTAGE];
    double peak[SUBHALO_NSTAGE];
    double own[SUBHALO_NSTAGE];     /* Phase 5d: the high water without the field SO's elastic export tables */
    double entry;                   /* usage at the call's entry */
    double elastic;                 /* the largest field SO export tables */
} SubClock;

static double
stage_used_high_water(void)
{
    if(A_MAIN->use_malloc)
        return 0;
    return (double) (A_MAIN->size - allocator_get_min_free(A_MAIN));
}

static void
stage_begin_call(void)
{
    memset(&SubClock, 0, sizeof(SubClock));
    SubClock.tstart = SubClock.tlast = MPI_Wtime();
    allocator_reset_min_free(A_MAIN);
    subhalo_so_elastic_reset();
    SubClock.entry = A_MAIN->use_malloc ? 0 : (double) (A_MAIN->size - mymalloc_freebytes());
}

static void
stage_end(int s)
{
    const double t = MPI_Wtime();
    SubClock.time[s] += t - SubClock.tlast;
    SubClock.tlast = t;
    /* the field SO walks take their elastic export tables out of the low-water mark; the
     * absolute high water puts them back */
    const double own = stage_used_high_water();
    double u = own;
    const size_t wmin = subhalo_so_elastic_min_free();
    if(!A_MAIN->use_malloc && wmin != (size_t) -1 && (double) (A_MAIN->size - wmin) > u)
        u = (double) (A_MAIN->size - wmin);
    if(u > SubClock.peak[s])
        SubClock.peak[s] = u;
    if(own > SubClock.own[s])
        SubClock.own[s] = own;
    if(subhalo_so_elastic_bytes() > SubClock.elastic)
        SubClock.elastic = subhalo_so_elastic_bytes();
    allocator_reset_min_free(A_MAIN);
    subhalo_so_elastic_reset();
    walltime_measure(SubhaloStageClock[s]);
}

/* Finishes every call, on every return path (collective): reduces the stage clock and the call's
 * diagnostics into SubLastStats and logs the per-call summary block. locstats (this rank's
 * diagnostics), plan and so may be NULL; on a failed call (complete = 0) every counter is zero
 * and only the wall times and memory peaks of the stages that ran are kept. */
static void
subhalo_finish_call(int SnapNum, int complete, const struct SubhaloCallStats * locstats, const struct SubhaloPlan * plan,
                    const struct SubhaloSOStats * so, MPI_Comm Comm)
{
    int NTask, ThisTask, s;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    struct SubhaloCallStats * st = &SubLastStats;
    if(complete && locstats)
        stats_reduce(locstats, st, Comm);
    else
        memset(st, 0, sizeof(*st));
    st->Complete = complete;

    /* wall time and the used high water per stage */
    const double wall = MPI_Wtime() - SubClock.tstart;
    double loc[3 * SUBHALO_NSTAGE + 3], mx[3 * SUBHALO_NSTAGE + 3], sum[SUBHALO_NSTAGE + 1];
    double incr[SUBHALO_NSTAGE];
    for(s = 0; s < SUBHALO_NSTAGE; s++) {
        loc[s] = SubClock.time[s];
        loc[SUBHALO_NSTAGE + s] = SubClock.peak[s];
        /* a stage that did not run (or freed below the entry) has no increment */
        incr[s] = SubClock.own[s] > SubClock.entry ? SubClock.own[s] - SubClock.entry : 0;
        loc[2 * SUBHALO_NSTAGE + s] = incr[s];
    }
    loc[3 * SUBHALO_NSTAGE] = wall;
    loc[3 * SUBHALO_NSTAGE + 1] = SubClock.entry;
    loc[3 * SUBHALO_NSTAGE + 2] = SubClock.elastic;
    MPI_Allreduce(loc, mx, 3 * SUBHALO_NSTAGE + 3, MPI_DOUBLE, MPI_MAX, Comm);
    double tl[SUBHALO_NSTAGE + 1];
    for(s = 0; s < SUBHALO_NSTAGE; s++)
        tl[s] = SubClock.time[s];
    tl[SUBHALO_NSTAGE] = wall;
    MPI_Allreduce(tl, sum, SUBHALO_NSTAGE + 1, MPI_DOUBLE, MPI_SUM, Comm);
    for(s = 0; s < SUBHALO_NSTAGE; s++) {
        st->StageMax[s] = mx[s];
        st->StageMean[s] = sum[s] / NTask;
        st->StagePeakBytes[s] = mx[SUBHALO_NSTAGE + s];
        st->StageIncrBytes[s] = mx[2 * SUBHALO_NSTAGE + s];
    }
    st->EntryBytes = mx[3 * SUBHALO_NSTAGE + 1];
    st->FieldSOElasticBytes = mx[3 * SUBHALO_NSTAGE + 2];
    st->WallTime = mx[3 * SUBHALO_NSTAGE];
    st->WallTimeMean = sum[SUBHALO_NSTAGE] / NTask;
    {
        int ps = 0;
        for(s = 1; s < SUBHALO_NSTAGE; s++)
            if(SubClock.peak[s] > SubClock.peak[ps])
                ps = s;
        struct { double v; int r; } in = {SubClock.peak[ps], ThisTask}, out;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, Comm);
        MPI_Bcast(&ps, 1, MPI_INT, out.r, Comm);
        st->PeakBytes = (int64_t) out.v;
        st->PeakRank = out.v > 0 ? out.r : -1;
        st->PeakStage = out.v > 0 ? ps : -1;
    }
    {
        /* the finder's own increment: the largest stage increment, its rank and stage */
        int is = 0;
        for(s = 1; s < SUBHALO_NSTAGE; s++)
            if(incr[s] > incr[is])
                is = s;
        struct { double v; int r; } in = {incr[is], ThisTask}, out;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, Comm);
        MPI_Bcast(&is, 1, MPI_INT, out.r, Comm);
        st->PeakIncrementBytes = (int64_t) out.v;
        st->PeakIncrementRank = out.v > 0 ? out.r : -1;
        st->PeakIncrementStage = out.v > 0 ? is : -1;
    }

    if(complete) {
        /* the exchange volume */
        if(plan) {
            double ex[2] = {(double) plan->ExchOut, (double) plan->ExchIn}, exm[2], exs[2];
            MPI_Allreduce(ex, exm, 2, MPI_DOUBLE, MPI_MAX, Comm);
            MPI_Allreduce(ex, exs, 2, MPI_DOUBLE, MPI_SUM, Comm);
            st->ExchOutMax = exm[0];
            st->ExchInMax = exm[1];
            st->ExchOutMean = exs[0] / NTask;
            st->ExchInMean = exs[1] / NTask;
            st->NSerial = plan->Stats.NSerial;
            st->LargestSerial = plan->Stats.LargestSerial;
            st->NCands = plan->NCands;
            st->NRounds = plan->NRounds;
            double tb = plan->Admit.checked ? plan->Admit.stage[SUBHALO_COLL_TREE] : 0;
            MPI_Allreduce(&tb, &st->TreeBytes, 1, MPI_DOUBLE, MPI_MAX, Comm);
        }
        /* the serial keys: counts, the longest key, the thread and rank balance, the cost fit */
        const struct SubhaloSerialStats * ss = subhalo_serial_last_stats();
        int64_t cnt[2] = {ss->NKeys, ss->NAlone}, cntg[2];
        MPI_Allreduce(cnt, cntg, 2, MPI_INT64, MPI_SUM, Comm);
        st->NSerialKeys = cntg[0];
        st->NAloneKeys = cntg[1];
        struct { double v; int r; } kin = {ss->KeyTimeMax, ThisTask}, kout;
        MPI_Allreduce(&kin, &kout, 1, MPI_DOUBLE_INT, MPI_MAXLOC, Comm);
        int64_t kn = ss->KeyTimeMaxN;
        MPI_Bcast(&kn, 1, MPI_INT64, kout.r, Comm);
        st->KeyTimeMax = kout.v;
        st->KeyTimeMaxN = kn;
        const double timb = ss->ThreadMeanSum > 0 ? ss->ThreadMax / ss->ThreadMeanSum : 0;
        const double rload = ss->AloneWall + ss->ThreadMax;
        double bl[8] = {timb, timb > 0 ? 1 : 0, rload, ss->FitN, ss->FitX, ss->FitXX, ss->FitY, ss->FitXY}, bs[8], bm = 0, rm = 0;
        MPI_Allreduce(bl, bs, 8, MPI_DOUBLE, MPI_SUM, Comm);
        MPI_Allreduce(&timb, &bm, 1, MPI_DOUBLE, MPI_MAX, Comm);
        MPI_Allreduce(&rload, &rm, 1, MPI_DOUBLE, MPI_MAX, Comm);
        st->ThreadImbalance = bm;
        st->ThreadImbalanceMean = bs[1] > 0 ? bs[0] / bs[1] : 0;
        st->RankImbalance = bs[2] > 0 ? rm / (bs[2] / NTask) : 0;
        const double den = bs[3] * bs[5] - bs[4] * bs[4];
        if(bs[3] >= 2 && den > 0) {
            st->KeyFitA = (bs[3] * bs[7] - bs[4] * bs[6]) / den;
            st->KeyFitB = (bs[6] - st->KeyFitA * bs[4]) / bs[3];
        }
        if(so) {
            st->FieldSOPass1Walks = so->Npass1Walks;
            st->FieldSOPass2Walks = so->Npass2Walks;
            st->FieldSOBins = so->NBins;
            st->FieldSOCentrals = so->Ncentral;
        }
    }

    /* the summary block */
    message(0, "Subhalo call summary, snapshot %d (%s): wall %.3f s (max over %d ranks), %.3f s (mean); main allocator used high water "
               "%.1f MB on rank %d in stage %s.\n", SnapNum, complete ? "complete" : "FAILED", st->WallTime, NTask, st->WallTimeMean,
            st->PeakBytes / 1048576., st->PeakRank, st->PeakStage >= 0 ? SubhaloStageName[st->PeakStage] : "-");
    message(0, "Subhalo call summary: finder's own increment %.1f MB on rank %d in stage %s (used high water without the field SO "
               "walks' elastic export tables, %.1f MB at most, minus the usage at entry, %.1f MB at most).\n",
            st->PeakIncrementBytes / 1048576., st->PeakIncrementRank,
            st->PeakIncrementStage >= 0 ? SubhaloStageName[st->PeakIncrementStage] : "-", st->FieldSOElasticBytes / 1048576.,
            st->EntryBytes / 1048576.);
    message(0, "Subhalo call summary:   %-12s | %9s | %9s | %9s | %9s\n", "stage", "max s", "mean s", "peak MB", "incr MB");
    for(s = 0; s < SUBHALO_NSTAGE; s++)
        message(0, "Subhalo call summary:   %-12s | %9.4f | %9.4f | %9.1f | %9.1f\n", SubhaloStageName[s], st->StageMax[s], st->StageMean[s],
                st->StagePeakBytes[s] / 1048576., st->StageIncrBytes[s] / 1048576.);
    if(!complete)
        return;
    message(0, "Subhalo call summary: exchange: particles out %.0f max, %.1f mean; in %.0f max, %.1f mean per rank.\n",
            st->ExchOutMax, st->ExchOutMean, st->ExchInMax, st->ExchInMean);
    message(0, "Subhalo call summary: serial keys: %ld run (%ld alone), plan %ld serial keys, largest %ld particles; longest key %.4f s "
               "(%ld particles); thread imbalance max/mean %.3f (worst rank), %.3f (mean over ranks); rank imbalance max/mean %.3f.\n",
            st->NSerialKeys, st->NAloneKeys, st->NSerial, st->LargestSerial, st->KeyTimeMax, st->KeyTimeMaxN, st->ThreadImbalance,
            st->ThreadImbalanceMean, st->RankImbalance);
    message(0, "Subhalo call summary: key time fit t = %g s x N log2 N + %g s (B/A = %g particle-log units; the cost model uses "
               "SUBHALO_COST_KEY / SUBHALO_COST_ITER = %g).\n", st->KeyFitA, st->KeyFitB,
            st->KeyFitA > 0 ? st->KeyFitB / st->KeyFitA : 0., SUBHALO_COST_KEY / SUBHALO_COST_ITER);
    message(0, "Subhalo call summary: rounds: %ld in-place candidates in %d round(s), tree stage estimate %.1f MB (max over ranks).\n",
            st->NCands, st->NRounds, st->TreeBytes / 1048576.);
    if(so)
        message(0, "Subhalo call summary: field SO: %ld centrals, %d pass-1 walks, %d pass-2 walks, %d bins.\n", st->FieldSOCentrals,
                st->FieldSOPass1Walks, st->FieldSOPass2Walks, st->FieldSOBins);
}

/* ------------------------------------------------------------------ the driver */

int
subhalo_find(FOFGroups * fof, DomainDecomp * ddecomp, int SnapNum, double atime, Cosmology * CP, int MetalReturnOn, MPI_Comm Comm)
{
    int64_t i;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    struct SubTrackTable * tab = &SubTable;
    const struct SubhaloParams * sp = &SubParams;

    walltime_measure("/Misc");
    stage_begin_call();
    if(!tab->initialised)
        subhalo_table_init(tab, SUBHALO_TABLE_INITIAL_BYTES, Comm);
    /* The static home rule must hold at the start of every call. */
    subhalo_table_redistribute(tab, Comm);

    message(0, "Subhalo finder: call for snapshot %d at a = %g, %ld tracks, %ld FOF groups.\n",
            SnapNum, atime, tab->NTotal, fof->TotNgroups);

    /* Tags must reference existing tracks. */
    int badtag = 0;
    #pragma omp parallel for reduction(+: badtag)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        if(P[i].TrackId >= tab->NTotal || P[i].BoundTrackId >= tab->NTotal || P[i].TrackId < -1 || P[i].BoundTrackId < -1)
            badtag++;
    }
    if(MPIU_Any(badtag > 0, Comm))
        endrun(5, "Particle tags reference tracks outside [0, %ld).\n", tab->NTotal);
    log_tagged("before the call", Comm);

    /* per-call transient fields of the home records */
    for(i = 0; i < tab->NLocal; i++) {
        struct SubTrack * t = &tab->Track[i];
        t->Host = -1;
        t->NboundPrev = t->Nbound;
        t->Alive = 0;
        t->NPrev = 0;
        t->MPrev = 0;
        t->NTracers = 0;
        t->Key = -1;
        int d;
        for(d = 0; d < 3; d++)
            t->RefPos[d] = t->RefVel[d] = 0;
    }

    stage_end(SUBHALO_STAGE_SETUP);

    /* Steps 0 and 2: tracer snapshot and weighted host vote, on the home ranks. */
    subhalo_host_vote(tab, Comm);
    stage_end(SUBHALO_STAGE_VOTE);
    /* Step 3 (frames): previous-bound frames from the BoundTrackId sums. */
    subhalo_previous_frames(tab, Comm);
    stage_end(SUBHALO_STAGE_FRAMES);

    /* Step 1: orphan tracer lookup (Host, TracerPos, TracerVel, SnapTracerLost). It runs before
     * the plan can budget anything, so its peak (the ID index with its sort copy, one batch, the
     * per-orphan ids and hits) is checked against the free memory first (J4); a rank short of it
     * fails the call cleanly, before anything is modified. */
    {
        int64_t norph = 0;
        for(i = 0; i < tab->NLocal; i++)
            if(tab->Track[i].NboundPrev <= 0)
                norph++;
        const double need = subhalo_lookup_index_bytes(PartManager->NumPart) + subhalo_lookup_batch_peak_bytes(tab->NTask)
            + (double) (norph + 1) * (sizeof(MyIDType) + sizeof(int64_t) + sizeof(struct SubhaloIdHit)) + 4. * 4096;
        const double avail = (double) mymalloc_freebytes();
        if(MPIU_Any(need > avail, Comm)) {
            message(1, "Subhalo finder: the orphan lookup needs %g bytes, %g free on this rank.\n", need, avail);
            message(0, "Subhalo finder: not enough memory for the orphan lookup of snapshot %d; the call is skipped. "
                       "Tags and track table are unchanged; this PIG will be marked incomplete. "
                       "Lower SubhaloLookupBatchBytes or run with more memory per rank.\n", SnapNum);
            SubStatus.LastCallComplete = 0;
            stage_end(SUBHALO_STAGE_LOOKUP);
            subhalo_finish_call(SnapNum, 0, NULL, NULL, NULL, Comm);
            return 0;
        }
    }
    int64_t nlook = 0, nfound = 0;
    subhalo_orphan_lookup(tab, SnapNum, atime, &nlook, &nfound, Comm);
    message(0, "Subhalo orphans: %ld tracers looked up, %ld found.\n", nlook, nfound);
    stage_end(SUBHALO_STAGE_LOOKUP);

    /* Step 3: keys (pseudo-keys of the hostless tracks) and the plan. */
    const int64_t NHostless = subhalo_assign_keys(tab, fof->TotNgroups, Comm);
    struct SubhaloPlan plan = {0};
    subhalo_plan_set_frame(atime, hubble_function(CP, atime));
    if(subhalo_plan_build(&plan, fof, tab, NHostless, ddecomp, Comm)) {
        /* audit Q1: the in-place stage of this plan cannot fit on some rank, or (Phase 5c) some
         * rank is short of the plan's reserve. Nothing in P has changed yet (no target, no
         * encoded GrNr), so the domain needs no restore. */
        if(plan.Admit.failed)
            message(0, "Subhalo finder: the in-place stage of snapshot %d does not fit in memory on rank %d (stage: %s)%s; the call is "
                       "skipped before the exchange. Tags and track table are unchanged; this PIG will be marked incomplete. "
                       "Run with more memory per rank or more ranks.\n", SnapNum, plan.Admit.failrank,
                    plan.Admit.failstage >= 0 ? SubhaloCollStageName[plan.Admit.failstage] : "?",
                    plan.Admit.reserve_failed ? " and the reserve admission failed too" : "");
        else
            message(0, "Subhalo finder: the reserve admission of snapshot %d failed: rank %d is %g MB short of the plan's reserve; the "
                       "call is skipped before the exchange. Tags and track table are unchanged; this PIG will be marked incomplete. "
                       "Run with more memory per rank or more ranks.\n", SnapNum, plan.Admit.reserve_rank,
                    plan.Admit.reserve_short / 1048576.);
        subhalo_restore_field_grnr(&plan);
        subhalo_plan_free(&plan);
        SubStatus.LastCallComplete = 0;
        stage_end(SUBHALO_STAGE_PLAN);
        subhalo_finish_call(SnapNum, 0, NULL, NULL, NULL, Comm);
        return 0;
    }
    stage_end(SUBHALO_STAGE_PLAN);

    /* Step 4: exchange whole serial keys, tags travel unchanged. */
    int failure = subhalo_plan_exchange(&plan, Comm);
    stage_end(SUBHALO_STAGE_EXCHANGE);

    /* Step 5: commit point. Nothing persistent has been modified yet (the step-1 tracer state
     * of the orphans is stored on the home records, which the next complete call redoes). */
    if(MPIU_Any(failure, Comm)) {
        message(0, "Subhalo finder: the particle exchange failed for snapshot %d; restoring the domain. "
                   "Tags and track table are unchanged; this PIG will be marked incomplete.\n", SnapNum);
        subhalo_restore_field_grnr(&plan);
        subhalo_plan_free(&plan);
        subhalo_restore_domain(ddecomp);
        SubStatus.LastCallComplete = 0;
        stage_end(SUBHALO_STAGE_RESTORE);
        subhalo_finish_call(SnapNum, 0, NULL, NULL, NULL, Comm);
        return 0;
    }

    subhalo_local_sort();
    struct SubhaloKeySeg * segs = NULL;
    const int64_t nsegs = subhalo_build_segments(&plan, &segs);
    stage_end(SUBHALO_STAGE_SORT);

    /* Step 7a: member records (with the transient fields), fetched from the home ranks. */
    struct SubTrack * members = NULL;
    const int64_t nmembers = subhalo_fetch_members(segs, nsegs, plan.OrphanIds, plan.NOrphanIds, tab, &members, Comm);
    stage_end(SUBHALO_STAGE_FETCH);

    struct SubhaloUnbindParams up;
    memset(&up, 0, sizeof(up));
    up.atime = atime;
    up.Hubble = hubble_function(CP, atime);
    up.G = CP->GravInternal;
    up.h = FORCE_SOFTENING();
    up.BoxSize = PartManager->BoxSize;
    up.PotentialTypes = sp->SubhaloPotentialTypes;
    up.TracerTypes = sp->SubhaloTracerTypes;
    up.MinNumPart = sp->SubhaloMinNumPart;
    up.MinNumTracerPart = sp->SubhaloMinNumTracerPart;
    up.SampleSize = sp->SubhaloPotentialSampleSize;
    up.SampledFinalPass = !sp->SubhaloFinalPass;
    /* the core and the reserved tracers (SubhaloMinNumTracerPart >= SubhaloNumTracerHostFinding) */
    up.FinalRefineTracers = sp->SubhaloCoreSize > sp->SubhaloMinNumTracerPart ? sp->SubhaloCoreSize : sp->SubhaloMinNumTracerPart;
    up.MaxIter = sp->SubhaloMaxUnbindIterations;
    up.OpenAngle = sp->SubhaloTreeOpenAngle;
    up.BoundMassPrecision = sp->SubhaloBoundMassPrecision;
    up.WithThermal = sp->SubhaloUnbindWithThermal;

    struct SubhaloCallCtx ctx = {0};
    ctx.SnapNum = SnapNum;
    ctx.atime = atime;
    ctx.BoxSize = PartManager->BoxSize;
    for(i = 0; i < 3; i++)
        ctx.Offset[i] = PartManager->CurrentParticleOffset[i];
    ctx.TracerTypes = sp->SubhaloTracerTypes;
    ctx.ThisTask = ThisTask;
    ctx.TotNgroups = fof->TotNgroups;
    ctx.plan = &plan;
    ctx.up = &up;
    ctx.Members = members;
    ctx.NMembers = nmembers;
    /* Phase 3: the SO thresholds of this call and the metal switch of the baryonic sums */
    subhalo_so_thresholds(CP, atime, ctx.rho);
    ctx.MetalReturnOn = MetalReturnOn;
    ctx.Stats.WorstConvRatio = 1;
    ctx.Stats.WorstTrackId = -1;

    /* The result slots: the members hosted by every serial key, its newborn, the in-place
     * candidates owned here. The results live on the top (they survive the domain restore); the
     * slot index on the bottom is freed before the restore. */
    const int64_t nres = subhalo_result_slots(segs, nsegs, &ctx);
    struct SubhaloResult * res = (struct SubhaloResult *) mymalloc2("SubResults", sizeof(struct SubhaloResult) * (nres + 1));
    memset(res, 0, sizeof(struct SubhaloResult) * (nres + 1));
    for(i = 0; i <= nres; i++) {
        res[i].ipk = -1;
        res[i].subkey = -1;
    }
    ctx.Res = res;
    ctx.NRes = nres;

    /* Steps 7-8: serial keys, tags written straight into P. */
    subhalo_serial_reset_stats();
    subhalo_process_serial(segs, nsegs, &ctx);
    stage_end(SUBHALO_STAGE_SERIAL);

    /* The travelled field particles and the group particles of the sub-tree keys get their GrNr
     * back; the slot index is freed before the restore. */
    subhalo_restore_field_grnr(&plan);
    subhalo_free_slots(&ctx);

    /* Step 9: restore the domain, then the oversized candidates in place, deepest round first
     * (Phase 2b): their records, the hierarchy of the sub-tree results, the final tags of every
     * particle of an oversized key. */
    subhalo_restore_domain(ddecomp);
    stage_end(SUBHALO_STAGE_RESTORE);
    subhalo_collective_run(&plan, &ctx, ddecomp, tab, Comm);
    subhalo_free_bind_ord(&ctx);
    stage_end(SUBHALO_STAGE_ROUNDS);

    /* Step 10: new TrackIds in host order. */
    const int64_t NTotalBefore = tab->NTotal;
    const int64_t nnew = subhalo_number_newborns(res, nres, fof->TotNgroups, NTotalBefore, Comm);
    tab->NTotal = NTotalBefore + nnew;
    subhalo_table_redistribute(tab, Comm);

    /* The final TrackId of every in-place winner, on every rank (orphan links, J8). */
    const int64_t K = plan.NInplace;
    struct InplaceWinnerEntry * ipl = (struct InplaceWinnerEntry *) mymalloc2("SubIplWinners", sizeof(struct InplaceWinnerEntry) * (K + 1));
    {
        int64_t * w = (int64_t *) mymalloc("SubIplW", sizeof(int64_t) * (K + 1));
        int64_t * gw = (int64_t *) mymalloc("SubIplGW", sizeof(int64_t) * (K + 1));
        for(i = 0; i < K; i++)
            w[i] = -1;
        for(i = 0; i < nres; i++)
            if(res[i].valid && res[i].ipk >= 0)
                w[res[i].ipk] = res[i].rec.TrackId;
        if(K > 0)
            MPI_Allreduce(w, gw, K, MPI_INT64, MPI_MAX, Comm);
        for(i = 0; i < K; i++) {
            ipl[i].g = plan.Inplace[i].g;
            ipl[i].winner = gw[i];
        }
        myfree(gw);
        myfree(w);
    }
    stage_end(SUBHALO_STAGE_NUMBER);

    /* Step 11: descendants of the dead and sunk tracks (remote part and the majority). */
    subhalo_finish_descendants(res, nres, Comm);
    stage_end(SUBHALO_STAGE_DESCENDANTS);

    /* Step 13 for the records resolved this call, then everything back to the home ranks. */
    int64_t nout = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid)
            nout++;
    struct SubTrack * out = (struct SubTrack *) mymalloc("SubOutRecs", sizeof(struct SubTrack) * (nout + 1));
    nout = 0;
    for(i = 0; i < nres; i++) {
        if(!res[i].valid)
            continue;
        if(res[i].history)
            subhalo_history_update(&res[i].rec, SnapNum, res[i].MaxMergerMbound);
        out[nout++] = res[i].rec;
    }
    stage_end(SUBHALO_STAGE_HISTORY);
    subhalo_table_store(tab, out, nout, Comm);
    myfree(out);
    /* The per-key buffers are no longer needed: release them (reverse order) before the
     * update of the unclaimed tracks. The in-place winners (ipl) were allocated after them. */
    struct SubhaloCallStats locstats = ctx.Stats;
    const int64_t nipl = K;
    struct InplaceWinnerEntry * iplcopy = (struct InplaceWinnerEntry *) mymalloc("SubIplCopy", sizeof(struct InplaceWinnerEntry) * (nipl + 1));
    memcpy(iplcopy, ipl, sizeof(struct InplaceWinnerEntry) * nipl);
    myfree(ipl);
    myfree(res);
    myfree(members);
    myfree(segs);
    subhalo_plan_free(&plan);
    qsort(iplcopy, nipl, sizeof(struct InplaceWinnerEntry), cmp_iplwinner);
    stage_end(SUBHALO_STAGE_STORE);

    /* Unclaimed tracks on the home ranks: resolved tracks without a key die (the hostless rule
     * of an unprocessed pseudo-key is unreachable since Phase 2b processes oversized pseudo-keys
     * in place). */
    int64_t nhostless = 0, ndead = 0;
    subhalo_update_unhosted(tab, SnapNum, atime, &nhostless, &ndead, Comm);
    stage_end(SUBHALO_STAGE_UNHOSTED);
    /* The orphans' current host and central (J8). */
    int64_t norphsync = subhalo_sync_orphan_links(tab, iplcopy, nipl), norphsync_tot = 0;
    MPI_Allreduce(&norphsync, &norphsync_tot, 1, MPI_INT64, MPI_SUM, Comm);
    myfree(iplcopy);
    stage_end(SUBHALO_STAGE_ORPHANSYNC);

    /* Step 12: field SO of the centrals of this call, on the restored domain, from the home
     * records (every record now describes this call; the PIG writer reads them afterwards). */
    struct SubhaloSOStats sostats;
    memset(&sostats, 0, sizeof(sostats));
    if(sp->SubhaloFieldSO) {
        subhalo_field_so_full(tab->Track, tab->NLocal, ddecomp, atime, CP, sp, NULL, NULL, &sostats, Comm);
        stage_end(SUBHALO_STAGE_FIELDSO);
    }

    /* Counts of this call on the home ranks. */
    int64_t loc[4] = {0, 0, 0, 0}, tot[4];
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * t = &tab->Track[i];
        if(t->SnapSink == SnapNum) loc[0]++;
        if(t->SnapDeath == SnapNum && t->SnapSink != SnapNum) loc[1]++;
        if(t->Nbound > 0) loc[2]++;
        if(t->Nbound > 0 && t->HostGroup < 0) loc[3]++;
    }
    MPI_Allreduce(loc, tot, 4, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo finder: snapshot %d done: %ld newborn, %ld sunk, %ld died (%ld without particles), %ld resolved (%ld hostless, %ld by the unprocessed-pseudo-key rule); %ld tracks in total.\n",
            SnapNum, nnew, tot[0], tot[1], ndead, tot[2], tot[3], nhostless, tab->NTotal);
    subhalo_finish_call(SnapNum, 1, &locstats, &plan, sp->SubhaloFieldSO ? &sostats : NULL, Comm);
    message(0, "Subhalo diagnostics: %ld unbindings, %ld at the iteration cap (worst Nbound/Nlast %g, track %ld); "
               "%ld of %ld final-pass bound particles with E >= 0; %ld tail claims across the final branch dropped; "
               "in-place tags written %ld (%ld stale field tags released); %ld orphan hosts updated; arena high water %.3f.\n",
            SubLastStats.NCandidates, SubLastStats.NItercap, SubLastStats.WorstConvRatio, SubLastStats.WorstTrackId,
            SubLastStats.NFinalPositive, SubLastStats.NFinalBound, SubLastStats.NTailRejected,
            SubLastStats.NFallbackField, SubLastStats.NFallbackReleased, norphsync_tot, SubLastStats.ArenaPeakFrac);
    log_tagged("after the call", Comm);

    if(SubStatus.StateSnapshot >= 0 && SubStatus.StateTime > 0)
        SubStatus.TimeSincePrevious = atime - SubStatus.StateTime;
    else
        SubStatus.TimeSincePrevious = -1;
    SubStatus.LastCallComplete = 1;
    SubStatus.LastCallSnapshot = SnapNum;
    SubStatus.StateSnapshot = SnapNum;
    SubStatus.StateTime = atime;
    subhalo_set_state_so_thresholds(ctx.rho);
    message(0, "Subhalo finder: time since the previous call (scale factor) %g.\n", SubStatus.TimeSincePrevious);
    return 1;
}
