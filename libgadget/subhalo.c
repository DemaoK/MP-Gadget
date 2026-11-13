/* Finder core: parameters, units, status, stage clocks, the driver subhalo_find (design 7.2, 10)
 * and the restart entry subhalo_init (11); services in subhalo_{table,lookup,newborn,orphan}.c. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <float.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/openmpsort.h"
#include "utils/system.h"
#include "utils/string.h"

#include "walltime.h"
#include "treewalk.h"
#include "gravity.h"
#include "subhalo.h"
#include "subhalo_internal.h"
#include "subhalo_collective.h"
#include "subhalo_unbind.h"
#include "subhalo_so.h"

/* TracerPenalty sits in the padding after BindRank, so struct particle_data keeps its size. */
_Static_assert(offsetof(struct particle_data, TracerPenalty) == offsetof(struct particle_data, BindRank) + sizeof(int32_t)
               && offsetof(struct particle_data, BoundTrackId) == offsetof(struct particle_data, TracerPenalty) + sizeof(float),
               "TracerPenalty must fill the padding between BindRank and BoundTrackId");

/* ------------------------------------------------------------------ parameters */

struct SubhaloParams SubParams;
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

/* How a parameter is declared, stored and printed: SUBPAR_COUNT is an int64_t declared as a double
 * so that a parameter file can write it in exponent form; it must hold a non-negative integer. */
enum { SUBPAR_INT, SUBPAR_DOUBLE, SUBPAR_COUNT };
/* Flags: lower bound excluded; range checked only with the field SO on (its only reader); written last
 * and only while it and SubhaloAdmitShare are positive, so a run without admission keeps its header. */
#define SUBPAR_LO_OPEN 1
#define SUBPAR_FIELDSO 2
#define SUBPAR_ADMIT 4

/* One Subhalo* parameter: declaration (name, kind, default, help), its member of struct
 * SubhaloParams and the range [lo, hi] the start-up check enforces while the finder is on. */
struct SubhaloParamDesc {
    const char * name;
    int kind;
    size_t offset;
    double def;
    double lo, hi;
    int flags;
    const char * help;
};

#define SUBPAR(name, kind, def, lo, hi, flags, help) {#name, SUBPAR_##kind, offsetof(struct SubhaloParams, name), def, lo, hi, flags, help}

/* Every Subhalo* parameter, in the order of the parameter dump and the header string (design 10).
 * Doubles must be finite; rules between parameters are in subhalo_validate_params. */
static const struct SubhaloParamDesc SubhaloParamTable[] = {
    SUBPAR(SubhaloFinderOn, INT, 0, -DBL_MAX, DBL_MAX, 0, "Run the subhalo finder at every PART/PIG output. Requires SnapshotWithFOF = 1 and FOFSaveParticles = 1."),
    SUBPAR(SubhaloMaxGatherFraction, DOUBLE, 0.5, 0, DBL_MAX, SUBPAR_LO_OPEN, "Fraction of the particle slots left free on a rank after its own group particles have left that may be filled by whole groups."),
    SUBPAR(SubhaloCollectiveMinPart, COUNT, 0, 0, DBL_MAX, 0, "Testing knob: candidate keys above this particle count are processed in place even if they would fit a rank. 0 = off."),
    SUBPAR(SubhaloMaxUnbindIterations, INT, 50, 1, DBL_MAX, 0, "Iteration cap of the unbinding loop."),
    SUBPAR(SubhaloRestartFresh, INT, 0, -DBL_MAX, DBL_MAX, 0, "If 1, start with an empty track table when no complete finder state is found at restart; otherwise abort."),
    SUBPAR(SubhaloTracerTypes, INT, 2+16, 0, 63, 0, "Bitmask of tracer particle types."),
    SUBPAR(SubhaloPotentialTypes, INT, 1+2+16+32, 0, 63, 0, "Bitmask of particle types that source the potential and are unbound."),
    SUBPAR(SubhaloMinNumPart, INT, 20, 1, DBL_MAX, 0, "Minimum number of bound particles of a resolved subhalo."),
    SUBPAR(SubhaloMinNumTracerPart, INT, 10, 1, DBL_MAX, 0, "Minimum number of bound tracer particles of a resolved subhalo."),
    /* at most SUBHALO_MAXTRACERS: the tracer IDs live in the track record */
    SUBPAR(SubhaloNumTracerHostFinding, INT, 10, 1, SUBHALO_MAXTRACERS, 0, "Tracers used in the host vote and for descendants. Must not exceed SubhaloMinNumTracerPart."),
    SUBPAR(SubhaloOrphanTracers, INT, 10, 1, SUBHALO_MAXTRACERS, 0, "Particles an orphan follows: its MostBoundID and the first host-vote tracers at death; position and host from their clipped, penalty-weighted set. 1 = the MostBoundID alone (HBT). At most 16."),
    SUBPAR(SubhaloLookupBatchBytes, COUNT, 64.*1024*1024, 1024.*1024, DBL_MAX, 0, "Per-rank bound in bytes on one batch of the tracer ID lookups: the gathered IDs plus the outgoing and returning replies (at least one ID per rank and batch)."),
    SUBPAR(SubhaloMajorProgenitorMassRatio, DOUBLE, 0.8, 0, 1, 0, "Central candidacy mass ratio."),
    SUBPAR(SubhaloBoundMassPrecision, DOUBLE, 0.995, 0, 1, SUBPAR_LO_OPEN, "Unbinding convergence criterion."),
    SUBPAR(SubhaloSourceRelaxFactor, DOUBLE, 3, 1, SUBHALO_MAX_RELAX_FACTOR, 0, "Source size as a multiple of Nbound, in [1, 10]; the per-thread arena reserves 8 bytes per particle per unit."),
    SUBPAR(SubhaloPotentialSampleSize, INT, 1000, 0, DBL_MAX, 0, "Potential subsampling size; 0 disables subsampling."),
    SUBPAR(SubhaloFinalPass, INT, 0, 0, 1, 0, "0 = sampled final pass (default; HBT-HERONS's rule: the last unbinding iteration's subsampled order, the head of the most bound re-ranked by exact energies over all bound particles; subhalos above SubhaloPotentialSampleSize only); 1 = exact final pass (energies of the bound set over all its particles set the whole binding order)."),
    SUBPAR(SubhaloTreeOpenAngle, DOUBLE, 0.45, 0, DBL_MAX, SUBPAR_LO_OPEN, "Geometric opening angle of the serial octree and of the in-place walk."),
    SUBPAR(SubhaloMergeTrapped, INT, 1, -DBL_MAX, DBL_MAX, 0, "Enable sinking of trapped satellites."),
    SUBPAR(SubhaloSinkSymmetric, INT, 0, -DBL_MAX, DBL_MAX, 0, "0 = HBT code behaviour, 1 = documented symmetric sinking test."),
    SUBPAR(SubhaloCoreSize, INT, 20, 1, DBL_MAX, 0, "Core particles for the centre and sinking."),
    SUBPAR(SubhaloReassignGas, INT, 1, -DBL_MAX, DBL_MAX, 0, "HBT ReassignParticles for non-tracers (serial groups only)."),
    SUBPAR(SubhaloNumNeighboursForReassignment, INT, 10, 1, SUBHALO_KNN_MAXK, 0, "Nearest tracers searched by SubhaloReassignGas (HBT NumNeighboursForReassignment)."),
    SUBPAR(SubhaloAdmitShare, DOUBLE, 0, 0, 1, SUBPAR_ADMIT, "In [0, 1], 0 = off. A tracer owned by an ancestor of a satellite joins the satellite's source when at least this share of its SubhaloNumNeighboursForReassignment nearest tracers were bound to the satellite at the previous call; it stays only if the unbinding binds it."),
    SUBPAR(SubhaloAdmitMinCoreDistance, DOUBLE, 12, 0, DBL_MAX, SUBPAR_ADMIT, "Not negative, 0 = off. With SubhaloAdmitShare > 0, a particle is not admitted to a satellite whose previous core lies nearer its owner's previous core than this phase-space distance of the sinking test (in the owner's dispersions; a satellite sinks below 2)."),
    SUBPAR(SubhaloUnbindWithThermal, INT, 0, -DBL_MAX, DBL_MAX, 0, "Add the gas internal energy to the kinetic term when unbinding."),
    SUBPAR(SubhaloFieldSO, INT, 1, -DBL_MAX, DBL_MAX, 0, "Run the field spherical overdensity tree walk for centrals."),
    SUBPAR(SubhaloSOBins, INT, 64, 8, 256, SUBPAR_FIELDSO, "Logarithmic radial bins of the field SO pass 1."),
    SUBPAR(SubhaloSOSearchFactor, DOUBLE, 2, 0, DBL_MAX, SUBPAR_LO_OPEN | SUBPAR_FIELDSO, "Field SO search radius in units of the radius implied by the inclusive bound mass."),
    SUBPAR(SubhaloSOMaxDoublings, INT, 3, 0, 60, SUBPAR_FIELDSO, "Maximum number of field SO search-radius doublings per central."),
    SUBPAR(SubhaloSORefineMinPart, INT, 1000, -DBL_MAX, DBL_MAX, 0, "Centrals at least this large get the field SO pass-2 refinement."),
    SUBPAR(SubhaloMajorMergerRatio, DOUBLE, 0.3, 0, DBL_MAX, 0, "Mass ratio for SnapLastMajorMerger."),
    SUBPAR(SubhaloGeometricParents, INT, 0, -DBL_MAX, DBL_MAX, 0, "Also compute consistent-trees-style geometric pid/upid."),
    SUBPAR(SubhaloDebugForceExchangeFailure, INT, 0, -DBL_MAX, DBL_MAX, 0, "Debug: send every serial key to rank 0 and take the exchange failure path."),
};
#define SUBHALO_NPARAMS ((int) (sizeof(SubhaloParamTable) / sizeof(SubhaloParamTable[0])))

void
subhalo_declare_params(ParameterSet * ps)
{
    int k;
    for(k = 0; k < SUBHALO_NPARAMS; k++) {
        const struct SubhaloParamDesc * d = &SubhaloParamTable[k];
        if(d->kind == SUBPAR_INT)
            param_declare_int(ps, d->name, OPTIONAL, (int) d->def, d->help);
        else
            param_declare_double(ps, d->name, OPTIONAL, d->def, d->help);
    }
}

/* The value of a parameter as a double (exact for every int and for counts below 2^53). */
static double
subhalo_param_value(const struct SubhaloParamDesc * d, const struct SubhaloParams * sp)
{
    const char * m = (const char *) sp + d->offset;
    if(d->kind == SUBPAR_INT)
        return *(const int *) m;
    if(d->kind == SUBPAR_COUNT)
        return (double) *(const int64_t *) m;
    return *(const double *) m;
}

/* 1 if v is finite and inside the range of d; the range of a SUBPAR_FIELDSO parameter applies only
 * with fieldso set. Finiteness first: under -ffast-math a NaN comparison proves nothing. */
static int
subhalo_param_ok(const struct SubhaloParamDesc * d, double v, int fieldso)
{
    if(!subhalo_isfinite(v))
        return 0;
    if((d->flags & SUBPAR_FIELDSO) && !fieldso)
        return 1;
    const int lo_ok = (d->flags & SUBPAR_LO_OPEN) ? v > d->lo : v >= d->lo;
    return lo_ok && v <= d->hi;
}

static const struct SubhaloParamDesc *
subhalo_param_desc(const char * name)
{
    int k;
    for(k = 0; k < SUBHALO_NPARAMS; k++)
        if(0 == strcmp(SubhaloParamTable[k].name, name))
            return &SubhaloParamTable[k];
    endrun(5, "%s is not a subhalo parameter\n", name);
}

int
subhalo_admit_distance_valid(double MinD)
{
    return subhalo_param_ok(subhalo_param_desc("SubhaloAdmitMinCoreDistance"), MinD, 0);
}

/* Validation of the parameters: endrun(1) on every rank, which all hold the broadcast values. */
static void
subhalo_validate_params(const struct SubhaloParams * sp)
{
    int k;
    for(k = 0; k < SUBHALO_NPARAMS; k++) {
        const struct SubhaloParamDesc * d = &SubhaloParamTable[k];
        const double v = subhalo_param_value(d, sp);
        if(!subhalo_param_ok(d, v, sp->SubhaloFieldSO)) {
            char lo[32] = "-inf", hi[32] = "inf";
            if(d->lo > -DBL_MAX)
                snprintf(lo, sizeof(lo), "%g", d->lo);
            if(d->hi < DBL_MAX)
                snprintf(hi, sizeof(hi), "%g", d->hi);
            endrun(1, "%s = %.17g must be finite and in %c%s, %s]. %s\n", d->name, v,
                   (d->flags & SUBPAR_LO_OPEN) ? '(' : '[', lo, hi, d->help);
        }
    }
    if(sp->SubhaloNumTracerHostFinding > sp->SubhaloMinNumTracerPart)
        endrun(1, "SubhaloNumTracerHostFinding = %d must not exceed SubhaloMinNumTracerPart = %d.\n",
               sp->SubhaloNumTracerHostFinding, sp->SubhaloMinNumTracerPart);
}

/* Arena bytes per particle of a serial key (subhalo_internal.h), set from SubhaloSourceRelaxFactor. */
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
    SubArenaBytesPerPart = SUBHALO_ARENA_FIXED_BYTES_PER_PART + (size_t) ceil(8. * R)
                           + (sp->SubhaloAdmitShare > 0 ? SUBHALO_ADMIT_BYTES_PER_PART : 0);
}

/* A double parameter that must hold a non-negative integer, checked before the conversion. */
static int64_t
subhalo_int_param(double v, const char * name)
{
    if(!subhalo_isfinite(v) || v < 0 || v > SUBHALO_INT_PARAM_MAX || v != floor(v))
        endrun(1, "%s = %g must be a non-negative integer below %g.\n", name, v, SUBHALO_INT_PARAM_MAX);
    return (int64_t) v;
}

/* The run's ImportBufferBoost: the default of gadget/params.c until set_subhalo_params reads it. */
static double SubhaloImportBufferBoost = 2;

double
subhalo_import_buffer_boost(void)
{
    return SubhaloImportBufferBoost;
}

void
subhalo_set_import_buffer_boost(double boost)
{
    SubhaloImportBufferBoost = boost;
}

size_t
subhalo_walk_reserve_bytes(size_t q, size_t r, double boost)
{
    const size_t bpb = sizeof(struct data_index) + q + r + (size_t) ceil(boost * (q + r));
    const int64_t nthr = omp_get_max_threads();
    const int64_t nbuf = 4096 * 10 + (4096 + 1 > 100 * nthr ? 4096 + 1 : 100 * nthr);
    return (size_t) nbuf * bpb;
}

void
set_subhalo_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    int checks[2] = {0, 0};
    /* the SUBPAR_COUNT parameters as read, converted on every rank after validation */
    double raw[SUBHALO_NPARAMS] = {0};
    /* the treewalk's import buffer factor (the memory checks of the field SO and the in-place walk) */
    double boost = 2;
    int k;
    if(ThisTask == 0) {
        boost = param_get_double(ps, "ImportBufferBoost");
        for(k = 0; k < SUBHALO_NPARAMS; k++) {
            const struct SubhaloParamDesc * d = &SubhaloParamTable[k];
            char * m = (char *) &SubParams + d->offset;
            if(d->kind == SUBPAR_INT)
                *(int *) m = param_get_int(ps, d->name);
            else if(d->kind == SUBPAR_DOUBLE)
                *(double *) m = param_get_double(ps, d->name);
            else
                raw[k] = param_get_double(ps, d->name);
        }
        checks[0] = param_get_int(ps, "SnapshotWithFOF");
        checks[1] = param_get_int(ps, "FOFSaveParticles");
        param_get_string2(ps, "FOFFileBase", SubhaloFOFFileBase, sizeof(SubhaloFOFFileBase));
    }
    MPI_Bcast(&SubParams, sizeof(struct SubhaloParams), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(checks, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(raw, SUBHALO_NPARAMS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&boost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    subhalo_set_import_buffer_boost(boost);
    MPI_Bcast(SubhaloFOFFileBase, sizeof(SubhaloFOFFileBase), MPI_CHAR, 0, MPI_COMM_WORLD);

    const int on = SubParams.SubhaloFinderOn;
    if(on) {
        if(checks[0] != 1)
            endrun(1, "SubhaloFinderOn = 1 requires SnapshotWithFOF = 1 (the finder runs at every PART/PIG output).\n");
        if(checks[1] != 1)
            endrun(1, "SubhaloFinderOn = 1 requires FOFSaveParticles = 1 (the PIG particle blocks are the finder's restart state).\n");
    }
    /* unused while off: the defaults keep the conversion defined */
    for(k = 0; k < SUBHALO_NPARAMS; k++) {
        const struct SubhaloParamDesc * d = &SubhaloParamTable[k];
        if(d->kind == SUBPAR_COUNT)
            *(int64_t *) ((char *) &SubParams + d->offset) = on ? subhalo_int_param(raw[k], d->name) : (int64_t) d->def;
    }
    if(!on)
        return;
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

/* Provenance only (a restart uses the supplied parameter file): every parameter in table order,
 * doubles %.17g, the SUBPAR_ADMIT ones last. */
void
subhalo_format_params(char * buf, size_t size)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    int last, k;
    buf[0] = '\0';
    for(last = 0; last < 2; last++)
        for(k = 0; k < SUBHALO_NPARAMS; k++) {
            const struct SubhaloParamDesc * d = &SubhaloParamTable[k];
            const int admit = (d->flags & SUBPAR_ADMIT) != 0;
            if(admit != last || (admit && !(sp->SubhaloAdmitShare > 0 && subhalo_param_value(d, sp) > 0)))
                continue;
            const size_t len = strlen(buf);
            const char * m = (const char *) sp + d->offset;
            if(d->kind == SUBPAR_INT)
                snprintf(buf + len, size - len, "%s = %d\n", d->name, *(const int *) m);
            else if(d->kind == SUBPAR_COUNT)
                snprintf(buf + len, size - len, "%s = %ld\n", d->name, *(const int64_t *) m);
            else
                snprintf(buf + len, size - len, "%s = %.17g\n", d->name, *(const double *) m);
        }
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

struct SubTrackTable *
subhalo_get_table(void)
{
    return &SubTable;
}

/* ------------------------------------------------------------------ status */

int subhalo_last_call_complete(void) { return SubStatus.LastCallComplete; }

/* The PIG of a complete call carries the interval's scatterings, consumed once it is written (also after a failed
 * particle write: the continuing run starts from this call); a skipped or incomplete call keeps them. */
void
subhalo_consume_penalties(int SnapNum)
{
    if(!SubStatus.LastCallComplete || SubStatus.LastCallSnapshot != SnapNum)
        return;
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].TracerPenalty = 0;
}
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
        /* the finder's live-particle predicate: swallowed BHs keep their tags but do not count */
        if(!subhalo_part_live(p) || p->TrackId < 0)
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
        /* nonzero code, so mpirun exits nonzero; m is the same on every rank */
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
    /* The snapshot's TracerPenalty column (zero when absent) holds the scatterings since the previous
     * call; a complete call at the snapshot itself consumed them, any other state still needs them. */
    if(m == RestartSnapNum) {
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
            P[i].TracerPenalty = 0;
    }

    int64_t inherited_tot = 0;
    MPI_Allreduce(&inherited, &inherited_tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo finder: state of PIG %d loaded (%ld tracks, a = %g); %ld particles inherited a tag from their lineage.\n",
            m, SubTable.NTotal, SubStatus.StateTime, inherited_tot);
    log_tagged("after the restart join", Comm);
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
    int64_t a[15] = {loc->NCandidates, loc->NItercap, loc->NFinalPositive, loc->NFinalBound, loc->NTailRejected,
                     loc->NFallbackField, loc->NFallbackReleased, loc->NRescueTried, loc->NRescued, loc->NLossTried,
                     loc->NLossRescued, loc->NAdmitted, loc->NAdmitTaken, loc->NAdmitReceivers, loc->NAdmitWithheld}, b[15];
    MPI_Allreduce(a, b, 15, MPI_INT64, MPI_SUM, Comm);
    memset(tot, 0, sizeof(*tot));
    tot->NCandidates = b[0];
    tot->NItercap = b[1];
    tot->NFinalPositive = b[2];
    tot->NFinalBound = b[3];
    tot->NTailRejected = b[4];
    tot->NFallbackField = b[5];
    tot->NFallbackReleased = b[6];
    tot->NRescueTried = b[7];
    tot->NRescued = b[8];
    tot->NLossTried = b[9];
    tot->NLossRescued = b[10];
    tot->NAdmitted = b[11];
    tot->NAdmitTaken = b[12];
    tot->NAdmitReceivers = b[13];
    tot->NAdmitWithheld = b[14];
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

/* ------------------------------------------------------------------ per-stage instrumentation */

const char * const SubhaloStageName[SUBHALO_NSTAGE] = {
    "setup", "vote", "frames", "lookup", "plan", "exchange", "sort", "fetch", "serial", "restore",
    "rounds", "number", "descendants", "history", "store", "unhosted", "orphansync", "fieldso"};
/* The rounds and field SO stages are subdivided by their callees (tree builds redirected to
 * .../Tree); the driver books the remainder as Rounds/Cleanup and FieldSO/Walks. */
static const char * const SubhaloStageClock[SUBHALO_NSTAGE] = {
    "/Subhalo/Setup", "/Subhalo/Vote", "/Subhalo/Frames", "/Subhalo/Lookup", "/Subhalo/Plan", "/Subhalo/Exchange",
    "/Subhalo/Sort", "/Subhalo/Fetch", "/Subhalo/Serial", "/Subhalo/Restore", "/Subhalo/Rounds/Cleanup", "/Subhalo/Number",
    "/Subhalo/Descendants", "/Subhalo/History", "/Subhalo/Store", "/Subhalo/Unhosted", "/Subhalo/OrphanSync",
    "/Subhalo/FieldSO/Walks"};

/* This rank's stage clock: wall time and main-allocator used high water per stage (from the
 * low-water mark of the free bytes, reset at every stage end). */
static struct {
    double tstart, tlast;
    double time[SUBHALO_NSTAGE];
    double peak[SUBHALO_NSTAGE];
    double own[SUBHALO_NSTAGE];     /* the high water without the field SO's elastic export tables */
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

/* End of every call, on every return path: reduce the clocks and diagnostics into SubLastStats
 * and log the summary. A failed call keeps only times and memory. Collective. */
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
            double tb = plan->MemCheck.checked ? plan->MemCheck.stage[SUBHALO_COLL_TREE] : 0;
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
        t->MarkerWeight = 0;
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

    /* Step 1: orphan tracer lookup. It runs before the plan budgets memory, so its peak is checked
     * first; a rank short of it fails the call before anything is modified. */
    {
        int64_t norph = 0, nids = 0;
        for(i = 0; i < tab->NLocal; i++)
            if(tab->Track[i].NboundPrev <= 0) {
                const MyIDType * mk;
                const int nm = orphan_markers(&tab->Track[i], &mk);
                norph++;
                nids += nm > 1 ? nm : 1;
            }
        /* the ids and hits of every marker, the record index of every orphan */
        const double need = subhalo_lookup_index_bytes(PartManager->NumPart) + subhalo_lookup_batch_peak_bytes(tab->NTask)
            + (double) (nids + 1) * (sizeof(MyIDType) + sizeof(struct SubhaloIdHit)) + (double) (norph + 1) * sizeof(int64_t) + 4. * 4096;
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
        /* the admission vote, the in-place stage or the plan's reserve does not fit on some rank; P
         * is unchanged so far, so the domain needs no restore. The track table keeps what the orphan
         * lookup wrote (the markers' positions and velocities, SnapTracerLost of lost markers). */
        if(plan.VoteFailed)
            message(0, "Subhalo finder: the admission vote of the oversized keys of snapshot %d does not fit in memory on rank %d "
                       "(%g MB short); the call is skipped before the exchange. Particle tags and GrNr are unchanged; the track "
                       "table keeps this call's orphan lookup (marker positions and velocities, lost markers); this PIG will be "
                       "marked incomplete. Run with more memory per rank or more ranks.\n", SnapNum,
                    subhalo_plan_last_vote()->WorstRank, subhalo_plan_last_vote()->WorstShort / 1048576.);
        else if(plan.MemCheck.failed)
            message(0, "Subhalo finder: the in-place stage of snapshot %d does not fit in memory on rank %d (stage: %s)%s; the call is "
                       "skipped before the exchange. Particle tags and GrNr are unchanged; the track table keeps this call's "
                       "orphan lookup; this PIG will be marked incomplete. Run with more memory per rank or more ranks.\n",
                    SnapNum, plan.MemCheck.failrank,
                    plan.MemCheck.failstage >= 0 ? SubhaloCollStageName[plan.MemCheck.failstage] : "?",
                    plan.MemCheck.reserve_failed ? " and the reserve admission failed too" : "");
        else
            message(0, "Subhalo finder: the reserve admission of snapshot %d failed: rank %d is %g MB short of the plan's reserve; the "
                       "call is skipped before the exchange. Particle tags and GrNr are unchanged; the track table keeps this "
                       "call's orphan lookup; this PIG will be marked incomplete. Run with more memory per rank or more ranks.\n",
                    SnapNum, plan.MemCheck.reserve_rank,
                    plan.MemCheck.reserve_short / 1048576.);
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

    /* Step 5: commit point; particle tags are unchanged so far, the travelling GrNr is restored on
     * failure, and the track table holds this call's orphan lookup (marker positions and
     * velocities, lost markers). */
    if(MPIU_Any(failure, Comm)) {
        message(0, "Subhalo finder: the particle exchange failed for snapshot %d; restoring the domain. Particle tags and "
                   "GrNr are unchanged; the track table keeps this call's orphan lookup; this PIG will be marked incomplete.\n",
                SnapNum);
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
    /* the SO thresholds of this call and the metal switch of the baryonic sums */
    subhalo_so_thresholds(CP, atime, ctx.rho);
    ctx.MetalReturnOn = MetalReturnOn;
    ctx.Stats.WorstConvRatio = 1;
    ctx.Stats.WorstTrackId = -1;
    /* the plan's vote is global: its withheld count enters the reduction once */
    ctx.Stats.NAdmitWithheld = ThisTask == 0 ? subhalo_plan_last_vote()->NWithheld : 0;

    /* Result slots (serial keys' members and newborns, in-place candidates owned here), on the
     * top so that they survive the domain restore. */
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
     * (design 7.8). */
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

    /* The final TrackId of every in-place winner, on every rank (for the orphan links). */
    const int64_t K = plan.NInplace;
    struct InplaceWinnerEntry * ipl = (struct InplaceWinnerEntry *) mymalloc2("SubIplWinners", sizeof(struct InplaceWinnerEntry) * (K + 1));
    gather_inplace_winners(&plan, res, nres, ipl, Comm);
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
    /* Free the per-key buffers before the unclaimed-track update; ipl is copied out first since
     * it was allocated after them. */
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

    /* Unclaimed tracks on the home ranks: resolved tracks without a key die. */
    int64_t nhostless = 0, ndead = 0;
    subhalo_update_unhosted(tab, SnapNum, atime, &nhostless, &ndead, Comm);
    stage_end(SUBHALO_STAGE_UNHOSTED);
    /* The orphans' current host and central. */
    int64_t norphsync = subhalo_sync_orphan_links(tab, iplcopy, nipl), norphsync_tot = 0;
    MPI_Allreduce(&norphsync, &norphsync_tot, 1, MPI_INT64, MPI_SUM, Comm);
    myfree(iplcopy);
    stage_end(SUBHALO_STAGE_ORPHANSYNC);

    /* Step 12: field SO of this call's centrals on the restored domain, from the home records. */
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
               "%ld of %ld tracks dead from the mean frame rescued from the clipped core frame; "
               "%ld of %ld tracks alive below half their previous bound number from the mean frame bound more from the clipped core frame; "
               "%ld of %ld final-pass bound particles with E >= 0; %ld tail claims across the final branch dropped; "
               "in-place tags written %ld, field tags released %ld; %ld orphan hosts updated; arena high water %.3f.\n",
            SubLastStats.NCandidates, SubLastStats.NItercap, SubLastStats.WorstConvRatio, SubLastStats.WorstTrackId,
            SubLastStats.NRescued, SubLastStats.NRescueTried, SubLastStats.NLossRescued, SubLastStats.NLossTried,
            SubLastStats.NFinalPositive, SubLastStats.NFinalBound, SubLastStats.NTailRejected,
            SubLastStats.NFallbackField, SubLastStats.NFallbackReleased, norphsync_tot, SubLastStats.ArenaPeakFrac);
    /* the withheld counts appear only with the withholding on */
    char wh[64] = "", vwh[64] = "";
    if(sp->SubhaloAdmitMinCoreDistance > 0) {
        snprintf(wh, sizeof(wh), ", %ld withheld near their owner's core", SubLastStats.NAdmitWithheld);
        snprintf(vwh, sizeof(vwh), ", %ld withheld near their owner's core", subhalo_plan_last_vote()->NWithheld);
    }
    if(sp->SubhaloAdmitShare > 0)
        message(0, "Subhalo admission: %ld particles of ancestors admitted to satellites by neighbour majority, %ld taken, by %ld satellites%s.\n",
                SubLastStats.NAdmitted, SubLastStats.NAdmitTaken, SubLastStats.NAdmitReceivers, wh);
    if(sp->SubhaloAdmitShare > 0 && subhalo_plan_last_vote()->NQueries > 0) {
        const struct SubhaloVoteStats * vs = subhalo_plan_last_vote();
        message(0, "Subhalo admission vote of the oversized keys across ranks: %ld tracers queried, %ld offered (%ld by in-place owners)%s, "
                   "%ld taken to their receivers' sub-tree keys (%ld guests), %ld kept home for in-place receivers; %d radius pass(es), "
                   "%.3g s.\n", vs->NQueries, vs->NOffered, vs->NOfferedInplace, vwh, vs->NServed, vs->NGuests, vs->NHome, vs->Passes, vs->Seconds);
    }
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
