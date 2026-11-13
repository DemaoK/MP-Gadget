/* The finder's records and the distributed track table (design 7.1): record reset, orphan and
 * history columns, the keyed all-to-all, and the table's storage, fetch and store by TrackId. */

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

#include "subhalo.h"
#include "subhalo_internal.h"

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
    subhalo_track_clear_orphan_set(t);
    /* transient */
    t->Host = -1;
    t->NboundPrev = 0;
    t->Alive = 0;
    t->NPrev = 0;
    t->MPrev = 0;
    t->MarkerWeight = 0;
    t->NTracers = 0;
    t->Key = -1;
}

void
subhalo_track_clear_orphan_set(struct SubTrack * t)
{
    int k;
    for(k = 0; k < SUBHALO_MAXTRACERS; k++)
        t->OrphanIDs[k] = (MyIDType) -1;
    t->NOrphanTracers = t->NOrphanFound = 0;
    t->OrphanSigmaX = t->OrphanScale = -1;
}

void
subhalo_track_make_orphan(struct SubTrack * t, int64_t winner, int SnapNum)
{
    int k, j;
    /* The tracer starts at the last most-bound particle (HBT keeps following it). */
    for(k = 0; k < 3; k++) {
        t->TracerPos[k] = t->MostBoundPos[k];
        t->TracerVel[k] = t->MostBoundVel[k];
    }
    /* The followed set: MostBoundID, then the first distinct step-0 tracers. */
    const int K = SubParams.SubhaloOrphanTracers > 1 ? SubParams.SubhaloOrphanTracers : 1;
    subhalo_track_clear_orphan_set(t);
    if(t->MostBoundID != (MyIDType) -1) {
        int n = 0;
        t->OrphanIDs[n++] = t->MostBoundID;
        for(j = 0; j < t->NTracers && j < SUBHALO_MAXTRACERS && n < K; j++) {
            const MyIDType id = t->TracerIDs[j];
            for(k = 0; k < n && t->OrphanIDs[k] != id; k++) {}
            if(k == n && id != (MyIDType) -1)
                t->OrphanIDs[n++] = id;
        }
        t->NOrphanTracers = n;
        t->NOrphanFound = 1;
    }
    /* the clip scale: the core size at the last resolved record */
    t->OrphanScale = t->CoreSigmaX > 0 ? t->CoreSigmaX : (t->Rhalf > 0 ? t->Rhalf : -1);
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
    /* HBT's SnapshotOfLastIsolation: -1 while always a central, the previous snapshot when it first
     * becomes a satellite, then the current one whenever it is a central (or hostless) again. */
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
                  "The finder does not batch this exchange; run with more ranks.\n", what, value);
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
                  "The finder does not batch this exchange; run with more ranks.\n", nsend, nrecv, (unsigned long) elsize);

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

/* Works from any placement (an older Chunk, or a restart read's row slices); the exchange is
 * skipped when every rank already holds a prefix of its home range. */
void
subhalo_table_redistribute(struct SubTrackTable * tab, MPI_Comm Comm)
{
    int64_t i;
    const int64_t NewChunk = table_chunk(tab->NTotal, tab->NTask);
    const int64_t expected = table_expected_local(tab->NTotal, NewChunk, tab->ThisTask);
    const int64_t first = (int64_t) tab->ThisTask * NewChunk;

    /* Fast path: the records are a prefix of the home range; new TrackIds get cleared records. */
    int ok = tab->NLocal <= expected;
    for(i = 0; ok && i < tab->NLocal; i++)
        if(tab->Track[i].TrackId != first + i)
            ok = 0;
    if(!MPIU_Any(!ok, Comm)) {
        subhalo_table_reserve(tab, expected);
        for(i = tab->NLocal; i < expected; i++)
            subhalo_track_clear(&tab->Track[i], first + i);
        tab->NLocal = expected;
        tab->Chunk = NewChunk;
        return;
    }

    /* Full exchange of the records to their new homes. */
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

/* Batched record fetch and store: at most PairCap requests per home rank and batch, so one
 * batch moves O(SubhaloLookupBatchBytes) bytes on any rank (subhalo_table_batch_peak_bytes). */
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
