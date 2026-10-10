/* Runtime subhalo finder: parameters, the track table, the driver and the track history.
 *
 * Design: halofinder_design.md; Phase 1 plan: halofinder_phase1_plan.md (section 1 is the
 * algorithm implemented by subhalo_find below, step by step).
 * The exchange, plan and domain restore are in subhalo_gather.c, the per-key processing in
 * subhalo_track.c, the catalogue output and the restart state in subhalo_io.c. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/system.h"
#include "utils/string.h"

#include "walltime.h"
#include "subhalo.h"
#include "subhalo_internal.h"

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
} SubStatus = {0, -1, -1, -1., -1.};

#define SUBHALO_TABLE_INITIAL_BYTES (64L * 1024L * 1024L)

void
set_subhalo_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    int checks[2] = {0, 0};
    if(ThisTask == 0) {
        SubParams.SubhaloFinderOn = param_get_int(ps, "SubhaloFinderOn");
        SubParams.SubhaloMaxGatherFraction = param_get_double(ps, "SubhaloMaxGatherFraction");
        SubParams.SubhaloCollectiveMinPart = (int64_t) param_get_double(ps, "SubhaloCollectiveMinPart");
        SubParams.SubhaloMaxUnbindIterations = param_get_int(ps, "SubhaloMaxUnbindIterations");
        SubParams.SubhaloRestartFresh = param_get_int(ps, "SubhaloRestartFresh");
        SubParams.SubhaloTracerTypes = param_get_int(ps, "SubhaloTracerTypes");
        SubParams.SubhaloPotentialTypes = param_get_int(ps, "SubhaloPotentialTypes");
        SubParams.SubhaloMinNumPart = param_get_int(ps, "SubhaloMinNumPart");
        SubParams.SubhaloMinNumTracerPart = param_get_int(ps, "SubhaloMinNumTracerPart");
        SubParams.SubhaloNumTracerHostFinding = param_get_int(ps, "SubhaloNumTracerHostFinding");
        SubParams.SubhaloLookupBatchBytes = (int64_t) param_get_double(ps, "SubhaloLookupBatchBytes");
        SubParams.SubhaloMajorProgenitorMassRatio = param_get_double(ps, "SubhaloMajorProgenitorMassRatio");
        SubParams.SubhaloBoundMassPrecision = param_get_double(ps, "SubhaloBoundMassPrecision");
        SubParams.SubhaloSourceRelaxFactor = param_get_double(ps, "SubhaloSourceRelaxFactor");
        SubParams.SubhaloPotentialSampleSize = param_get_int(ps, "SubhaloPotentialSampleSize");
        SubParams.SubhaloTreeOpenAngle = param_get_double(ps, "SubhaloTreeOpenAngle");
        SubParams.SubhaloMergeTrapped = param_get_int(ps, "SubhaloMergeTrapped");
        SubParams.SubhaloSinkSymmetric = param_get_int(ps, "SubhaloSinkSymmetric");
        SubParams.SubhaloCoreSize = param_get_int(ps, "SubhaloCoreSize");
        SubParams.SubhaloReassignGas = param_get_int(ps, "SubhaloReassignGas");
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
    MPI_Bcast(SubhaloFOFFileBase, sizeof(SubhaloFOFFileBase), MPI_CHAR, 0, MPI_COMM_WORLD);

    if(!SubParams.SubhaloFinderOn)
        return;
    if(checks[0] != 1)
        endrun(0, "SubhaloFinderOn = 1 requires SnapshotWithFOF = 1 (the finder runs at every PART/PIG output).\n");
    if(checks[1] != 1)
        endrun(0, "SubhaloFinderOn = 1 requires FOFSaveParticles = 1 (the PIG particle blocks are the finder's restart state).\n");
    if(SubParams.SubhaloMaxGatherFraction <= 0)
        endrun(0, "SubhaloMaxGatherFraction = %g must be positive.\n", SubParams.SubhaloMaxGatherFraction);
    if(SubParams.SubhaloCollectiveMinPart < 0)
        endrun(0, "SubhaloCollectiveMinPart = %ld must not be negative.\n", SubParams.SubhaloCollectiveMinPart);
    if(SubParams.SubhaloNumTracerHostFinding > SubParams.SubhaloMinNumTracerPart)
        endrun(0, "SubhaloNumTracerHostFinding = %d must not exceed SubhaloMinNumTracerPart = %d.\n",
               SubParams.SubhaloNumTracerHostFinding, SubParams.SubhaloMinNumTracerPart);
}

void
subhalo_set_testpar(struct SubhaloParams params)
{
    SubParams = params;
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

/* Step 13 (design 7.7), the part that needs only the previous record. Every Phase 1 track is a
 * central or hostless, so the accretion state and the isolation snapshot are set every call. */
void
subhalo_history_update(struct SubTrack * rec, int SnapNum, double MaxSinkMbound)
{
    if(rec->Mbound > rec->MboundPeak) {
        rec->MboundPeak = rec->Mbound;
        rec->SnapMboundPeak = SnapNum;
        rec->VmaxAtMboundPeak = rec->Vmax;
    }
    /* Vmax is -1 in Phase 1, so VmaxPeak stays -1. */
    if(rec->Vmax > rec->VmaxPeak) {
        rec->VmaxPeak = rec->Vmax;
        rec->SnapVmaxPeak = SnapNum;
    }
    /* central or hostless: state at the last output as a central */
    rec->MboundAcc = rec->Mbound;
    rec->VmaxAcc = rec->Vmax;
    rec->SnapAcc = SnapNum;
    /* SnapLastIsolation follows HBT (subhalo_tracking.cpp, SnapshotOfLastIsolation): it stays -1
     * while the track has only ever been a central; when the track first becomes a satellite it
     * is set to the previous snapshot; afterwards it moves to the current snapshot whenever the
     * track is a central again. Rank 0 is a central, -1 a hostless track (treated as central).
     * Phase 1 has no satellites, so it stays -1. */
    if(rec->Rank > 0) {
        if(rec->SnapLastIsolation < 0)
            rec->SnapLastIsolation = SnapNum - 1;
    } else if(rec->SnapLastIsolation >= 0)
        rec->SnapLastIsolation = SnapNum;
    /* SnapFirstAcc / MboundFirstAcc / VmaxFirstAcc: untouched (set when a track first becomes a satellite). */
    const double MM = SubParams.SubhaloMajorMergerRatio;
    if(MaxSinkMbound > 0 && MaxSinkMbound > rec->Mbound * MM / (1 + MM))
        rec->SnapLastMajorMerger = SnapNum;
}

/* ------------------------------------------------------------------ shared MPI helper */

int64_t
subhalo_alltoallv(const void * sendbuf, const int * sendcounts, size_t elsize, void ** recvbuf, int top, int * recvcounts_out, MPI_Comm Comm)
{
    int NTask, i;
    MPI_Comm_size(Comm, &NTask);
    int * recvcounts = ta_malloc("SubRecvCounts", int, NTask);
    int * sdispls = ta_malloc("SubSendDispls", int, NTask);
    int * rdispls = ta_malloc("SubRecvDispls", int, NTask);

    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, Comm);
    int64_t nsend = 0, nrecv = 0;
    for(i = 0; i < NTask; i++) {
        sdispls[i] = nsend;
        rdispls[i] = nrecv;
        nsend += sendcounts[i];
        nrecv += recvcounts[i];
    }
    if(nsend >= INT32_MAX || nrecv >= INT32_MAX)
        endrun(5, "Subhalo exchange of %ld / %ld elements overflows the MPI counts.\n", nsend, nrecv);

    char * recv;
    if(top)
        recv = (char *) mymalloc2("SubhaloRecv", elsize * nrecv + 1);
    else
        recv = (char *) mymalloc("SubhaloRecv", elsize * nrecv + 1);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);
    MPI_Alltoallv_sparse((void *) sendbuf, (int *) sendcounts, sdispls, dtype, recv, recvcounts, rdispls, dtype, Comm);
    MPI_Type_free(&dtype);

    if(recvcounts_out)
        memcpy(recvcounts_out, recvcounts, sizeof(int) * NTask);
    ta_free(rdispls);
    ta_free(sdispls);
    ta_free(recvcounts);
    *recvbuf = recv;
    return nrecv;
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

    /* Full exchange of the records to their new homes. */
    int * sendcounts = ta_malloc("SubTabSendCounts", int, tab->NTask);
    memset(sendcounts, 0, sizeof(int) * tab->NTask);
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

    /* Send buffer in the table allocator, on the same side as the current block. */
    const int onbottom = table_block_on_bottom(tab);
    struct SubTrack * sendbuf;
    if(onbottom)
        sendbuf = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTrackSend", sizeof(struct SubTrack) * (tab->NLocal + 1));
    else
        sendbuf = (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTrackSend", sizeof(struct SubTrack) * (tab->NLocal + 1));
    int * offset = ta_malloc("SubTabOffset", int, tab->NTask);
    int r;
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
    int * recvcounts = ta_malloc("SubTabRecvCounts", int, tab->NTask);
    int * sdispls = ta_malloc("SubTabSendDispls", int, tab->NTask);
    int * rdispls = ta_malloc("SubTabRecvDispls", int, tab->NTask);
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, Comm);
    int64_t nrecv = 0, nsend = 0;
    for(r = 0; r < tab->NTask; r++) {
        sdispls[r] = nsend;
        rdispls[r] = nrecv;
        nsend += sendcounts[r];
        nrecv += recvcounts[r];
    }
    if(nrecv >= INT32_MAX || nsend >= INT32_MAX)
        endrun(5, "Track table redistribution overflows the MPI counts.\n");
    struct SubTrack * recvbuf;
    if(onbottom)
        recvbuf = (struct SubTrack *) allocator_alloc_top(tab->alloc, "SubTrackRecv", sizeof(struct SubTrack) * (nrecv + 1));
    else
        recvbuf = (struct SubTrack *) allocator_alloc_bot(tab->alloc, "SubTrackRecv", sizeof(struct SubTrack) * (nrecv + 1));
    MPI_Datatype dtype;
    MPI_Type_contiguous(sizeof(struct SubTrack), MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);
    MPI_Alltoallv_sparse(sendbuf, sendcounts, sdispls, dtype, recvbuf, recvcounts, rdispls, dtype, Comm);
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
    ta_free(recvcounts);
    allocator_free(sendbuf);
    allocator_free(tab->Track);
    ta_free(sendcounts);

    tab->Track = newtrack;
    tab->NAlloc = newalloc;
    tab->NLocal = expected;
    tab->Chunk = NewChunk;
    int64_t nplaced_tot = 0;
    MPI_Allreduce(&nplaced, &nplaced_tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Track table redistributed: %ld tracks (%ld records moved or kept), chunk %ld.\n", tab->NTotal, nplaced_tot, tab->Chunk);
}

/* Sort helper: request index ordered by destination. */
struct TableReq {
    int64_t TrackId;
    int64_t index;
};

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

    int * sendcounts = ta_malloc("SubFetchCounts", int, NTask);
    int * recvcounts = ta_malloc("SubFetchRecvCounts", int, NTask);
    int * offset = ta_malloc("SubFetchOffset", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(i = 0; i < n; i++)
        sendcounts[subhalo_table_home_rank(tab, ids[i])]++;
    offset[0] = 0;
    for(r = 1; r < NTask; r++)
        offset[r] = offset[r-1] + sendcounts[r-1];

    int64_t * reqids = (int64_t *) mymalloc("SubFetchIds", sizeof(int64_t) * (n + 1));
    int64_t * perm = (int64_t *) mymalloc("SubFetchPerm", sizeof(int64_t) * (n + 1));
    for(i = 0; i < n; i++) {
        const int dest = subhalo_table_home_rank(tab, ids[i]);
        perm[offset[dest]] = i;
        reqids[offset[dest]] = ids[i];
        offset[dest]++;
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
    if(ngot != n)
        endrun(5, "subhalo_table_fetch: got %ld records for %ld requests.\n", ngot, n);
    for(i = 0; i < n; i++)
        out[perm[i]] = got[i];
    myfree(got);
    myfree(reply);
    myfree(inids);
    myfree(perm);
    myfree(reqids);
    ta_free(offset);
    ta_free(recvcounts);
    ta_free(sendcounts);
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

    int * sendcounts = ta_malloc("SubStoreCounts", int, NTask);
    int * offset = ta_malloc("SubStoreOffset", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(i = 0; i < n; i++)
        sendcounts[subhalo_table_home_rank(tab, recs[i].TrackId)]++;
    offset[0] = 0;
    for(r = 1; r < NTask; r++)
        offset[r] = offset[r-1] + sendcounts[r-1];
    struct SubTrack * sendbuf = (struct SubTrack *) mymalloc("SubStoreSend", sizeof(struct SubTrack) * (n + 1));
    for(i = 0; i < n; i++) {
        const int dest = subhalo_table_home_rank(tab, recs[i].TrackId);
        sendbuf[offset[dest]++] = recs[i];
    }
    struct SubTrack * in = NULL;
    int64_t nin = subhalo_alltoallv(sendbuf, sendcounts, sizeof(struct SubTrack), (void **) &in, 0, NULL, Comm);
    int missing = 0;
    for(i = 0; i < nin; i++) {
        struct SubTrack * rec = subhalo_table_get(tab, in[i].TrackId);
        if(!rec) {
            missing = 1;
            continue;
        }
        *rec = in[i];
    }
    if(missing)
        endrun(5, "subhalo_table_store: a stored track is not homed on rank %d (redistribute first).\n", tab->ThisTask);
    myfree(in);
    myfree(sendbuf);
    ta_free(offset);
    ta_free(sendcounts);
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
subhalo_count_tagged(const struct part_manager_type * pman, int64_t ntagged[6])
{
    int64_t i;
    int64_t n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0;
    #pragma omp parallel for reduction(+: n0, n1, n2, n3, n4, n5)
    for(i = 0; i < pman->NumPart; i++) {
        const struct particle_data * p = &pman->Base[i];
        if(p->IsGarbage || p->TrackId < 0)
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
subhalo_number_newborns(struct SubhaloKeyResult * res, int64_t nres, int64_t TotNgroups, int64_t NTotalBefore, MPI_Comm Comm)
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
    int * sendcounts = ta_malloc("SubNbCounts", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(i = 0; i < nlocal; i++)
        sendcounts[subhalo_key_dir_rank(ent[i].g, KChunk)]++;
    /* ent is sorted by g, hence by directory rank */
    struct NewbornEntry * dir = NULL;
    int64_t ndir = subhalo_alltoallv(ent, sendcounts, sizeof(struct NewbornEntry), (void **) &dir, 0, NULL, Comm);
    qsort(dir, ndir, sizeof(struct NewbornEntry), cmp_newborn_g);

    /* 3. every rank asks for the ordinals it holds: particle tags and its records */
    int64_t nreq = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        if(SUBHALO_IS_PROVISIONAL(P[i].TrackId)) nreq++;
        if(SUBHALO_IS_PROVISIONAL(P[i].BoundTrackId)) nreq++;
    }
    for(i = 0; i < nres; i++)
        if(res[i].valid && SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            nreq++;
    int64_t * req = (int64_t *) mymalloc("SubNbReq", sizeof(int64_t) * (nreq + 1));
    nreq = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        if(SUBHALO_IS_PROVISIONAL(P[i].TrackId)) req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(P[i].TrackId);
        if(SUBHALO_IS_PROVISIONAL(P[i].BoundTrackId)) req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(P[i].BoundTrackId);
    }
    for(i = 0; i < nres; i++)
        if(res[i].valid && SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(res[i].rec.TrackId);
    qsort(req, nreq, sizeof(int64_t), cmp_int64);
    int64_t nuniq = 0;
    for(i = 0; i < nreq; i++)
        if(nuniq == 0 || req[i] != req[nuniq-1])
            req[nuniq++] = req[i];
    memset(sendcounts, 0, sizeof(int) * NTask);
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
    int * recvcounts = ta_malloc("SubNbRecvCounts", int, NTask);
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
        res[i].newborn = 0;
    }

    /* 5. nothing provisional may survive */
    int left = 0;
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

/* Partial sums of a track's own-tag field particles on one rank. The centre is carried as a
 * local mass-weighted mean relative to the first local particle, combined at the home rank
 * with periodic wrapping relative to the first contribution. */
struct FieldSum {
    int64_t TrackId;
    int64_t count;
    int64_t ntracer;
    int64_t LenType[6];
    double MassType[6];
    double mass;
    double cen[3];      /* local centre of mass, translated frame */
    double mv[3];       /* sum m * Vel */
};

static int
cmp_fieldsum(const void * a, const void * b)
{
    const int64_t x = ((const struct FieldSum *) a)->TrackId, y = ((const struct FieldSum *) b)->TrackId;
    return (x > y) - (x < y);
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

/* Collective. Returns the global numbers of hostless and dead tracks. */
static void
subhalo_update_unhosted(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nhostless, int64_t * ndead, MPI_Comm Comm)
{
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i, j;
    const double BoxSize = PartManager->BoxSize;
    const int TracerTypes = SubParams.SubhaloTracerTypes;

    /* own-tag field particles, grouped by track (Phase 1 never touches their tags) */
    int64_t nf = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && P[i].GrNr < 0 && P[i].TrackId >= 0)
            nf++;
    struct TagIndex * idx = (struct TagIndex *) mymalloc("SubFieldIdx", sizeof(struct TagIndex) * (nf + 1));
    nf = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage && P[i].GrNr < 0 && P[i].TrackId >= 0) {
            idx[nf].TrackId = P[i].TrackId;
            idx[nf].i = i;
            nf++;
        }
    qsort(idx, nf, sizeof(struct TagIndex), cmp_tagindex);
    int64_t ns = 0;
    for(i = 0; i < nf; i++)
        if(i == 0 || idx[i].TrackId != idx[i-1].TrackId)
            ns++;
    struct FieldSum * sums = (struct FieldSum *) mymalloc("SubFieldSums", sizeof(struct FieldSum) * (ns + 1));
    int64_t s = -1;
    double ref[3] = {0, 0, 0};
    double mdx[3] = {0, 0, 0};
    for(i = 0; i <= nf; i++) {
        if(i == nf || i == 0 || idx[i].TrackId != idx[i-1].TrackId) {
            /* close the previous sum */
            if(s >= 0) {
                int k;
                for(k = 0; k < 3; k++)
                    sums[s].cen[k] = ref[k] + (sums[s].mass > 0 ? mdx[k] / sums[s].mass : 0);
            }
            if(i == nf)
                break;
            s++;
            memset(&sums[s], 0, sizeof(struct FieldSum));
            sums[s].TrackId = idx[i].TrackId;
            int k;
            for(k = 0; k < 3; k++) {
                ref[k] = P[idx[i].i].Pos[k];
                mdx[k] = 0;
            }
        }
        const struct particle_data * p = &P[idx[i].i];
        const double m = p->Mass;
        int k;
        sums[s].count++;
        sums[s].LenType[p->Type]++;
        sums[s].MassType[p->Type] += m;
        sums[s].mass += m;
        if(TracerTypes & (1 << p->Type))
            sums[s].ntracer++;
        for(k = 0; k < 3; k++) {
            mdx[k] += m * NEAREST(p->Pos[k] - ref[k], BoxSize);
            sums[s].mv[k] += m * p->Vel[k];
        }
    }
    /* sums are sorted by TrackId, hence by home rank */
    int * sendcounts = ta_malloc("SubFieldCounts", int, NTask);
    memset(sendcounts, 0, sizeof(int) * NTask);
    for(j = 0; j < ns; j++)
        sendcounts[subhalo_table_home_rank(tab, sums[j].TrackId)]++;
    struct FieldSum * in = NULL;
    int64_t nin = subhalo_alltoallv(sums, sendcounts, sizeof(struct FieldSum), (void **) &in, 0, NULL, Comm);
    /* stable merge so contributions combine in rank order */
    {
        /* insertion into TrackId order keeping the source order: use a stable merge sort */
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

    int64_t nh = 0, nd = 0;
    int64_t a = 0;
    for(j = 0; j < tab->NLocal; j++) {
        struct SubTrack * t = &tab->Track[j];
        /* contributions for this track */
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
            /* hostless resolved track: keeps its field particles */
            int64_t count = 0, ntracer = 0, LenType[6] = {0};
            double MassType[6] = {0}, mass = 0, mdxw[3] = {0}, mv[3] = {0};
            int64_t c;
            int k;
            const double * c0 = in[a].cen;
            for(c = a; c < b; c++) {
                count += in[c].count;
                ntracer += in[c].ntracer;
                for(k = 0; k < 6; k++) {
                    LenType[k] += in[c].LenType[k];
                    MassType[k] += in[c].MassType[k];
                }
                mass += in[c].mass;
                for(k = 0; k < 3; k++) {
                    mdxw[k] += in[c].mass * NEAREST(in[c].cen[k] - c0[k], BoxSize);
                    mv[k] += in[c].mv[k];
                }
            }
            t->HostGroup = -1;
            t->Rank = t->Depth = 0;
            t->NestedParentTrackId = -1;
            t->CentralTrackId = t->TrackId;
            t->SinkTrackId = -1;
            t->DescendantTrackId = t->TrackId;
            t->Nbound = t->NboundExclusive = count;
            t->NboundTracer = ntracer;
            t->Mbound = mass;
            for(k = 0; k < 6; k++) {
                t->NboundType[k] = t->NboundExclusiveType[k] = LenType[k];
                t->NtailType[k] = 0;
                t->MboundType[k] = MassType[k];
            }
            for(k = 0; k < 3; k++) {
                const double cen = c0[k] + (mass > 0 ? mdxw[k] / mass : 0);
                t->CorePos[k] = t->CoMPos[k] = wrap_box(cen - PartManager->CurrentParticleOffset[k], BoxSize);
                t->CoreVel[k] = t->CoMVel[k] = (mass > 0 ? mv[k] / mass : 0) / atime;
            }
            /* MostBound*: no binding order for hostless particles in Phase 1; the previous values stay. */
            t->Alive = 1;
            subhalo_history_update(t, SnapNum, 0);
            nh++;
        }
        else if(t->NboundPrev > 0) {
            /* resolved at the previous call, no particles anywhere now and not sunk: dead */
            subhalo_track_make_orphan(t, SUBHALO_NOTRACK, SnapNum);
            nd++;
        }
        a = b;
    }
    myfree(in);
    ta_free(sendcounts);
    myfree(sums);
    myfree(idx);

    MPI_Allreduce(&nh, nhostless, 1, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&nd, ndead, 1, MPI_INT64, MPI_SUM, Comm);
}

/* ------------------------------------------------------------------ the driver */

int
subhalo_find(FOFGroups * fof, DomainDecomp * ddecomp, int SnapNum, double atime, Cosmology * CP, MPI_Comm Comm)
{
    int64_t i;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    struct SubTrackTable * tab = &SubTable;

    walltime_measure("/Misc");
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
        tab->Track[i].Host = -1;
        tab->Track[i].NboundPrev = tab->Track[i].Nbound;
        tab->Track[i].Alive = 0;
    }

    /* Step 2: host vote, reduced on the home ranks. */
    subhalo_host_vote(tab, Comm);
    walltime_measure("/Subhalo/Vote");

    /* Step 3: plan (budgets, oversized keys, targets in P[i].TopLeaf). */
    struct SubhaloPlan plan = {0};
    subhalo_plan_build(&plan, fof, Comm);
    walltime_measure("/Subhalo/Plan");

    /* Step 4: exchange whole serial keys, tags travel unchanged. */
    int failure = subhalo_plan_exchange(&plan, Comm);
    walltime_measure("/Subhalo/Exchange");

    /* Step 5: commit point. Nothing persistent has been modified yet. */
    if(MPIU_Any(failure, Comm)) {
        message(0, "Subhalo finder: the particle exchange failed for snapshot %d; restoring the domain. "
                   "Tags and track table are unchanged; this PIG will be marked incomplete.\n", SnapNum);
        subhalo_plan_free(&plan);
        subhalo_restore_domain(ddecomp);
        walltime_measure("/Subhalo/Restore");
        SubStatus.LastCallComplete = 0;
        return 0;
    }

    subhalo_local_sort();
    struct SubhaloKeySeg * segs = NULL;
    const int64_t nsegs = subhalo_build_segments(&plan, &segs);
    walltime_measure("/Subhalo/Sort");

    /* Step 7a: member records with their host votes, fetched from the home ranks. */
    struct SubTrack * members = NULL;
    const int64_t nmembers = subhalo_fetch_members(segs, nsegs, tab, &members, Comm);
    int64_t * sinkwinner = (int64_t *) mymalloc2("SubSinkWinner", sizeof(int64_t) * (nmembers + 1));
    for(i = 0; i < nmembers; i++)
        sinkwinner[i] = SUBHALO_NOTRACK;
    struct SubhaloKeyResult * res = (struct SubhaloKeyResult *) mymalloc2("SubKeyResults", sizeof(struct SubhaloKeyResult) * (nsegs + 1));
    memset(res, 0, sizeof(struct SubhaloKeyResult) * (nsegs + 1));
    walltime_measure("/Subhalo/Members");

    struct SubhaloCallCtx ctx = {0};
    ctx.SnapNum = SnapNum;
    ctx.atime = atime;
    ctx.BoxSize = PartManager->BoxSize;
    for(i = 0; i < 3; i++)
        ctx.Offset[i] = PartManager->CurrentParticleOffset[i];
    ctx.TracerTypes = SubParams.SubhaloTracerTypes;
    ctx.ThisTask = ThisTask;
    ctx.Members = members;
    ctx.NMembers = nmembers;
    ctx.SinkWinner = sinkwinner;
    ctx.Res = res;

    /* Steps 7-8: serial keys, tags written straight into P. */
    subhalo_process_serial(segs, nsegs, &ctx);
    walltime_measure("/Subhalo/Serial");

    /* Step 9 (in-place part, before the restore in Phase 1). */
    subhalo_process_inplace(segs, nsegs, &plan, &ctx, tab, Comm);
    walltime_measure("/Subhalo/InPlace");

    /* Step 9: restore the domain. */
    subhalo_restore_domain(ddecomp);
    walltime_measure("/Subhalo/Restore");

    /* Step 7d: sunk members become orphans. Copies of the same member on several ranks (in
     * place) receive the same update, so duplicate stores are identical. */
    int64_t nsunk_local = 0;
    for(i = 0; i < nmembers; i++) {
        if(sinkwinner[i] == SUBHALO_NOTRACK)
            continue;
        subhalo_track_make_orphan(&members[i], sinkwinner[i], SnapNum);
        members[i].HostGroup = members[i].Host;
        nsunk_local++;
    }

    /* Step 10: new TrackIds in host order. */
    const int64_t NTotalBefore = tab->NTotal;
    const int64_t nnew = subhalo_number_newborns(res, nsegs, fof->TotNgroups, NTotalBefore, Comm);
    tab->NTotal = NTotalBefore + nnew;
    subhalo_table_redistribute(tab, Comm);
    walltime_measure("/Subhalo/Number");

    /* Step 13 for the winners, then write winners and sunk members back to their homes. */
    int64_t nout = 0;
    for(i = 0; i < nsegs; i++)
        if(res[i].valid)
            nout++;
    struct SubTrack * out = (struct SubTrack *) mymalloc("SubOutRecs", sizeof(struct SubTrack) * (nout + nsunk_local + 1));
    nout = 0;
    for(i = 0; i < nsegs; i++) {
        if(!res[i].valid)
            continue;
        subhalo_history_update(&res[i].rec, SnapNum, res[i].MaxSinkMbound);
        out[nout++] = res[i].rec;
    }
    for(i = 0; i < nmembers; i++)
        if(sinkwinner[i] != SUBHALO_NOTRACK)
            out[nout++] = members[i];
    subhalo_table_store(tab, out, nout, Comm);
    myfree(out);

    /* Step 7f: hostless and dead tracks, on the home ranks. */
    int64_t nhostless = 0, ndead = 0;
    subhalo_update_unhosted(tab, SnapNum, atime, &nhostless, &ndead, Comm);
    walltime_measure("/Subhalo/Update");

    /* Sunk count: only the deciding copy counts (duplicates in place are identical). */
    int64_t nsunk = 0, nsunk_home = 0;
    for(i = 0; i < tab->NLocal; i++)
        if(tab->Track[i].SnapSink == SnapNum)
            nsunk_home++;
    MPI_Allreduce(&nsunk_home, &nsunk, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo finder: snapshot %d done: %ld newborn, %ld sunk, %ld hostless, %ld dead; %ld tracks in total.\n",
            SnapNum, nnew, nsunk, nhostless, ndead, tab->NTotal);
    log_tagged("after the call", Comm);

    myfree(res);
    myfree(sinkwinner);
    myfree(members);
    myfree(segs);
    subhalo_plan_free(&plan);

    if(SubStatus.StateSnapshot >= 0 && SubStatus.StateTime > 0)
        SubStatus.TimeSincePrevious = atime - SubStatus.StateTime;
    else
        SubStatus.TimeSincePrevious = -1;
    SubStatus.LastCallComplete = 1;
    SubStatus.LastCallSnapshot = SnapNum;
    SubStatus.StateSnapshot = SnapNum;
    SubStatus.StateTime = atime;
    message(0, "Subhalo finder: time since the previous call (scale factor) %g.\n", SubStatus.TimeSincePrevious);
    (void) CP;
    return 1;
}
