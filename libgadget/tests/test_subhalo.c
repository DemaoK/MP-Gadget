/* Driver and plumbing tests of the runtime subhalo finder (MPI).
 * Run as: mpirun -np 4 .objs/test_subhalo with OMP_NUM_THREADS 1 or 2; above 2 threads the
 * MP-Gadget tree build itself fails on these 125-particle-per-rank fixtures (NODECACHE_SIZE).
 * Tests (the test_subhalo_ prefix omitted):
 *  1. table            track table: growth, home rank and local index, redistribution, round trips.
 *  2. sort_key         the 16-byte PIG sort key and the local comparator.
 *  3. exchange_tags    TrackId/BindRank/BoundTrackId survive domain_exchange.
 *  4. join             restart tag join from a synthetic PIG.
 *  5. fof_as_subhalo   the driver over five calls on two clumps.
 *  6. forced_failure   in-place (oversized) keys and the exchange failure path.
 *  7. plan_pack        the head/tail packing as a particle-free function, incl. cost balance.
 *  8. unhosted_direct  hostless centre across a periodic wrap, swallowed BHs, dead tracks, units.
 *  9. lifecycle_pig    driver, PIG writer and restart over four calls with an empty rank;
 *                      corrupted states rejected; header wall time and peak memory.
 * 10. many_keys        200 keys (> 32 per rank): tail packing on real particles.
 * Then, in registration order: twohalo, overlap, central_source, sink_orphan (persistent tracks,
 * a satellite with a tail, decision 52: a central's FOF-less particles untagged, a satellite's
 * kept; sinking, orphan tracer, resurrection; all but central_source log a tag/record checksum),
 * nest_tree, deep_chain, mixed_call, nocentral, fallback (serial/in-place transitions), itercap,
 * lookup (three modes), serial_keys (scheduler at 1-4 threads), reserve_admission,
 * local_sort (thread-count independent), header_params, sampled_default (SubhaloFinalPass = 0),
 * weighted_plurality and penalty_descendant (collision-penalty weights of the tracer votes).
 *
 * SUBHALO_TEST_ONLY=<substring> runs only the matching tests. The sort-key range checks abort
 * through endrun, so they run alone: SUBHALO_TEST_SORTKEY_OVERFLOW=grnr|trackid|bindrank
 * mpirun -np 1 .objs/test_subhalo must exit non-zero. */

/* nftw() for the temporary PIG cleanup */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <ftw.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#include <bigfile-mpi.h>

#include <libgadget/subhalo.h>
#include <libgadget/subhalo_internal.h>
#include <libgadget/petaio.h>
#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/exchange.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/cosmology.h>
#include <libgadget/gravity.h>
#include "stub.h"

static struct ClockTable CT;

/* ------------------------------------------------------------------ helpers */

/* Uniform assertion with a diagnostic: every rank evaluates the same data. */
#define CHECK(cond, ...) do { if(!(cond)) { message(1, __VA_ARGS__); } assert_true(cond); } while(0)

/* Count local failures, print the first few, then assert collectively with assert_all_true. */
#define LOCAL_CHECK(bad, cond, ...) do { if(!(cond)) { if((bad) < 10) message(1, __VA_ARGS__); (bad)++; } } while(0)

static int
get_ntask(void)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    return NTask;
}

static int
get_thistask(void)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    return ThisTask;
}

static struct SubhaloParams
default_subhalo_params(void)
{
    /* Every field of struct SubhaloParams, at the defaults of design section 10. */
    struct SubhaloParams sp;
    memset(&sp, 0, sizeof(sp));
    sp.SubhaloFinderOn = 1;
    sp.SubhaloMaxGatherFraction = 0.5;
    sp.SubhaloCollectiveMinPart = 0;
    sp.SubhaloMaxUnbindIterations = 50;
    sp.SubhaloRestartFresh = 1;
    sp.SubhaloTracerTypes = 2 + 16;
    sp.SubhaloPotentialTypes = 1 + 2 + 16 + 32;
    sp.SubhaloMinNumPart = 20;
    sp.SubhaloMinNumTracerPart = 10;
    sp.SubhaloNumTracerHostFinding = 10;
    sp.SubhaloLookupBatchBytes = 64L * 1024 * 1024;
    sp.SubhaloMajorProgenitorMassRatio = 0.8;
    sp.SubhaloBoundMassPrecision = 0.995;
    sp.SubhaloSourceRelaxFactor = 3;
    sp.SubhaloPotentialSampleSize = 1000;
    sp.SubhaloFinalPass = 1;
    sp.SubhaloTreeOpenAngle = 0.45;
    sp.SubhaloMergeTrapped = 1;
    sp.SubhaloSinkSymmetric = 0;
    sp.SubhaloCoreSize = 20;
    sp.SubhaloReassignGas = 1;
    sp.SubhaloNumNeighboursForReassignment = 10;
    sp.SubhaloUnbindWithThermal = 0;
    /* off in the Phase 1 fixtures: their sparse particle sets give the full tree fewer nodes
     * than the top-level grid; the Phase 2a/3 fixtures (p2_setup) switch it on */
    sp.SubhaloFieldSO = 0;
    sp.SubhaloSOBins = 64;
    sp.SubhaloSOSearchFactor = 2;
    sp.SubhaloSOMaxDoublings = 3;
    sp.SubhaloSORefineMinPart = 1000;
    sp.SubhaloMajorMergerRatio = 0.3;
    sp.SubhaloGeometricParents = 0;
    sp.SubhaloDebugForceExchangeFailure = 0;
    return sp;
}

static Cosmology
test_cosmology(void)
{
    Cosmology CP;
    memset(&CP, 0, sizeof(CP));
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 0.3;
    CP.OmegaBaryon = 0.045;
    CP.OmegaLambda = 0.7;
    CP.HubbleParam = 0.7;
    CP.RadiationOn = 0;
    CP.w0_fld = -1;
    CP.MNu[0] = CP.MNu[1] = CP.MNu[2] = 0;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.01, units);
    return CP;
}

/* Deterministic per-ID random numbers, identical on every rank. */
static uint64_t
splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static double
urand(uint64_t id, uint64_t salt, uint64_t k)
{
    uint64_t h = splitmix64(splitmix64(id * 0x100000001B3ULL + salt) ^ (k * 0x9E3779B97F4A7C15ULL));
    return (h >> 11) * (1.0 / 9007199254740992.0);
}

/* A point uniform in the ball of radius R around c, a deterministic function of (id, salt). */
static void
rand_in_ball(uint64_t id, uint64_t salt, double R, const double c[3], double out[3])
{
    uint64_t k = 0;
    double u[3], r2;
    do {
        int d;
        r2 = 0;
        for(d = 0; d < 3; d++) {
            u[d] = 2 * urand(id, salt, k++) - 1;
            r2 += u[d] * u[d];
        }
    } while(r2 > 1);
    int d;
    for(d = 0; d < 3; d++)
        out[d] = c[d] + R * u[d];
}

static int
cmp_double(const void * a, const void * b)
{
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

/* Allocate the particle table and the slots the way test_fof.c does. */
static void
setup_pman(int64_t MaxPart, double BoxSize)
{
    particle_alloc_memory(PartManager, BoxSize, MaxPart);
    PartManager->NumPart = 0;
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
}

static void
teardown_pman(void)
{
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
    MPI_Barrier(MPI_COMM_WORLD);
}

static void
add_dm_particle(MyIDType id, const double pos[3])
{
    if(PartManager->NumPart >= PartManager->MaxPart)
        endrun(1, "test setup: MaxPart %ld too small\n", PartManager->MaxPart);
    struct particle_data * p = &P[PartManager->NumPart++];
    memset(p, 0, sizeof(*p));
    p->ID = id;
    p->Type = 1;
    p->Mass = 1;
    p->IsGarbage = 0;
    p->GrNr = -1;
    p->TrackId = -1;
    p->BindRank = -1;
    p->BoundTrackId = -1;
    int d;
    for(d = 0; d < 3; d++)
        p->Pos[d] = pos[d];
}

/* Per-particle snapshot, gathered to every rank and sorted by ID, so assertions on it are
 * identical everywhere (collective-safe). */
struct tagrec {
    MyIDType ID;
    int64_t GrNr;
    int64_t TrackId;
    int64_t BoundTrackId;
    int32_t BindRank;
    int32_t Type;
    double Pos[3];
};

static int
cmp_tagrec_id(const void * a, const void * b)
{
    const struct tagrec * x = a, * y = b;
    return (x->ID > y->ID) - (x->ID < y->ID);
}

static struct tagrec *
gather_tags(int64_t * ntot)
{
    int NTask = get_ntask();
    int64_t i;
    int nlocal = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage)
            nlocal++;
    struct tagrec * loc = malloc((nlocal + 1) * sizeof(struct tagrec));
    int n = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage)
            continue;
        struct tagrec * r = &loc[n++];
        memset(r, 0, sizeof(*r));
        r->ID = P[i].ID;
        r->GrNr = P[i].GrNr;
        r->TrackId = P[i].TrackId;
        r->BoundTrackId = P[i].BoundTrackId;
        r->BindRank = P[i].BindRank;
        r->Type = P[i].Type;
        int d;
        for(d = 0; d < 3; d++)
            r->Pos[d] = P[i].Pos[d];
    }
    int nbytes = nlocal * sizeof(struct tagrec);
    int * counts = malloc(NTask * sizeof(int));
    int * displs = malloc(NTask * sizeof(int));
    MPI_Allgather(&nbytes, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t total = 0;
    int t;
    for(t = 0; t < NTask; t++) {
        displs[t] = total;
        total += counts[t];
    }
    struct tagrec * all = malloc(total + sizeof(struct tagrec));
    MPI_Allgatherv(loc, nbytes, MPI_BYTE, all, counts, displs, MPI_BYTE, MPI_COMM_WORLD);
    *ntot = total / sizeof(struct tagrec);
    qsort(all, *ntot, sizeof(struct tagrec), cmp_tagrec_id);
    free(displs);
    free(counts);
    free(loc);
    return all;
}

/* Copies of records 0..n-1 of the running finder's table, identical on every rank. Collective. */
static struct SubTrack *
fetch_all_tracks(int64_t n)
{
    int64_t * ids = malloc((n + 1) * sizeof(int64_t));
    struct SubTrack * out = malloc((n + 1) * sizeof(struct SubTrack));
    int64_t i;
    for(i = 0; i < n; i++)
        ids[i] = i;
    subhalo_table_fetch(subhalo_get_table(), ids, n, out, MPI_COMM_WORLD);
    free(ids);
    return out;
}

/* One finder call exactly as run.c makes it, on the current particle positions. */
static int
run_finder_call(int snap, Cosmology * CP, int64_t * TotNgroups)
{
    double atime = 0.5 + 0.05 * snap;
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    *TotNgroups = fof.TotNgroups;
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, 0, MPI_COMM_WORLD);
    fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
}

/* The summed time of this rank's walltime clocks whose names start with prefix (decision 47.5). */
static double
clock_sum(const char * prefix)
{
    double t = 0;
    int i;
    for(i = 0; i < CT.N; i++)
        if(strncmp(CT.C[i].name, prefix, strlen(prefix)) == 0)
            t += CT.C[i].time;
    return t;
}

/* Phase 5c: the consistency of the per-call statistics of the last subhalo_find. */
static void
check_call_stats(const char * what, int complete)
{
    const struct SubhaloCallStats * st = subhalo_last_call_stats();
    const int NTask = get_ntask();
    int s;
    double summean = 0;
    CHECK(st->Complete == complete, "%s: stats Complete %d, expected %d\n", what, st->Complete, complete);
    CHECK(st->WallTime > 0 && st->WallTimeMean > 0 && st->WallTimeMean <= st->WallTime * (1 + 1e-12),
          "%s: wall time max %g mean %g\n", what, st->WallTime, st->WallTimeMean);
    for(s = 0; s < SUBHALO_NSTAGE; s++) {
        CHECK(st->StageMax[s] >= 0 && st->StageMean[s] <= st->StageMax[s] * (1 + 1e-12) + 1e-15 && st->StageMax[s] <= st->WallTime
              && st->StagePeakBytes[s] <= (double) st->PeakBytes + 1,
              "%s: stage %s: max %g s mean %g s (wall %g s), peak %g of %ld bytes\n", what, SubhaloStageName[s], st->StageMax[s],
              st->StageMean[s], st->WallTime, st->StagePeakBytes[s], st->PeakBytes);
        summean += st->StageMean[s];
    }
    CHECK(summean <= st->WallTimeMean * (1 + 1e-9) + 1e-12, "%s: the stages add up to %g s, the call took %g s\n", what, summean,
          st->WallTimeMean);
    CHECK(st->PeakBytes > 0 && st->PeakStage >= 0 && st->PeakStage < SUBHALO_NSTAGE && st->PeakRank >= 0 && st->PeakRank < NTask
          && (int64_t) st->StagePeakBytes[st->PeakStage] == st->PeakBytes,
          "%s: peak %ld bytes on rank %d in stage %d\n", what, st->PeakBytes, st->PeakRank, st->PeakStage);
    /* Phase 5d: the finder's own increment: per stage at most the stage's high water less the
     * entry usage of some rank, the call's the largest of them */
    double imax = 0;
    for(s = 0; s < SUBHALO_NSTAGE; s++) {
        CHECK(st->StageIncrBytes[s] >= 0 && st->StageIncrBytes[s] <= st->StagePeakBytes[s] + 1,
              "%s: stage %s: increment %g bytes, high water %g\n", what, SubhaloStageName[s], st->StageIncrBytes[s], st->StagePeakBytes[s]);
        if(st->StageIncrBytes[s] > imax)
            imax = st->StageIncrBytes[s];
    }
    CHECK(st->PeakIncrementBytes == (int64_t) imax && st->PeakIncrementBytes > 0 && st->PeakIncrementBytes < st->PeakBytes
          && st->PeakIncrementStage >= 0 && st->PeakIncrementStage < SUBHALO_NSTAGE && st->PeakIncrementRank >= 0
          && st->PeakIncrementRank < NTask && st->EntryBytes > 0 && st->EntryBytes < st->PeakBytes,
          "%s: own increment %ld bytes (largest stage %g) on rank %d in stage %d; entry %g, high water %ld\n", what,
          st->PeakIncrementBytes, imax, st->PeakIncrementRank, st->PeakIncrementStage, st->EntryBytes, st->PeakBytes);
    /* the tables count with their allocator charge (audit 5d R2): whole 4096-byte allocator blocks */
    if(complete && st->FieldSOPass1Walks > 0)
        CHECK(st->FieldSOElasticBytes > 0 && fmod(st->FieldSOElasticBytes, 4096.) == 0
              && st->StageIncrBytes[SUBHALO_STAGE_FIELDSO] < st->StagePeakBytes[SUBHALO_STAGE_FIELDSO],
              "%s: field SO elastic export tables %g bytes, stage increment %g, high water %g\n", what, st->FieldSOElasticBytes,
              st->StageIncrBytes[SUBHALO_STAGE_FIELDSO], st->StagePeakBytes[SUBHALO_STAGE_FIELDSO]);
    if(complete) {
        CHECK(fabs(st->ExchOutMean - st->ExchInMean) < 1e-6 && st->ExchOutMax >= st->ExchOutMean && st->ExchInMax >= st->ExchInMean,
              "%s: exchange out %g/%g in %g/%g\n", what, st->ExchOutMax, st->ExchOutMean, st->ExchInMax, st->ExchInMean);
        CHECK(st->NSerialKeys == st->NSerial && st->NAloneKeys <= st->NSerialKeys,
              "%s: %ld serial keys run (%ld alone), the plan has %ld\n", what, st->NSerialKeys, st->NAloneKeys, st->NSerial);
        CHECK(st->NSerialKeys == 0 || (st->KeyTimeMax > 0 && st->KeyTimeMaxN > 0 && st->KeyTimeMaxN <= st->LargestSerial),
              "%s: longest key %g s with %ld particles, largest serial key %ld\n", what, st->KeyTimeMax, st->KeyTimeMaxN, st->LargestSerial);
        CHECK((st->RankImbalance == 0 || st->RankImbalance >= 1 - 1e-9) && (st->ThreadImbalance == 0 || st->ThreadImbalance >= 1 - 1e-9)
              && st->ThreadImbalanceMean <= st->ThreadImbalance * (1 + 1e-12),
              "%s: imbalance rank %g thread %g (mean %g)\n", what, st->RankImbalance, st->ThreadImbalance, st->ThreadImbalanceMean);
        CHECK(st->NCands == 0 || (st->NRounds > 0 && st->TreeBytes > 0), "%s: %ld candidates in %d rounds, tree %g bytes\n", what,
              st->NCands, st->NRounds, st->TreeBytes);
    }
    else
        CHECK(st->NCandidates == 0 && st->NSerialKeys == 0 && st->NSerial == 0 && st->ExchOutMax == 0 && st->ExchInMax == 0
              && st->NCands == 0 && st->FieldSOPass1Walks == 0,
              "%s: a failed call left counters (%ld unbindings, %ld keys, exchange %g)\n", what, st->NCandidates, st->NSerialKeys,
              st->ExchOutMax);
}

/* Phase 5c: the PIG header's SubhaloWallTime and SubhaloPeakBytes are those of the last call. */
static void
check_header_cost(const char * dir, int snap, const char * what)
{
    double w = -1;
    uint64_t pb = 0, pi = 0;
    int ok = 1;
    if(get_thistask() == 0) {
        char * f = fastpm_strdup_printf("%s/PIG_%03d", dir, snap);
        BigFile bf;
        BigBlock bh;
        if(0 != big_file_open(&bf, f) || 0 != big_file_open_block(&bf, &bh, "Header"))
            endrun(1, "test: cannot open %s/Header\n", f);
        if(0 != big_block_get_attr(&bh, "SubhaloWallTime", &w, "f8", 1) || 0 != big_block_get_attr(&bh, "SubhaloPeakBytes", &pb, "u8", 1)
           || 0 != big_block_get_attr(&bh, "SubhaloPeakIncrementBytes", &pi, "u8", 1))
            ok = 0;
        /* R3: the written SubhaloParameters is the formatter's string for the run's parameters */
        char want[4096];
        subhalo_format_params(want, sizeof(want));
        BigAttr * a = big_block_lookup_attr(&bh, "SubhaloParameters");
        if(!a || a->nmemb != (int) strlen(want) || 0 != memcmp(a->data, want, a->nmemb))
            ok = 0;
        big_block_close(&bh);
        big_file_close(&bf);
        myfree(f);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pb, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pi, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const struct SubhaloCallStats * st = subhalo_last_call_stats();
    CHECK(ok && w == st->WallTime && w > 0 && pb == (uint64_t) st->PeakBytes && pb > 0,
          "%s: header SubhaloWallTime %g SubhaloPeakBytes %lu (read %d), the call's %g s and %ld bytes\n", what, w, (unsigned long) pb, ok,
          st->WallTime, st->PeakBytes);
    /* Phase 5d: the finder's own increment round-trips too */
    CHECK(ok && pi == (uint64_t) st->PeakIncrementBytes && pi > 0 && pi < pb,
          "%s: header SubhaloPeakIncrementBytes %lu, the call's %ld bytes (high water %lu)\n", what, (unsigned long) pi,
          st->PeakIncrementBytes, (unsigned long) pb);
}

static void
common_physics_setup(void)
{
    walltime_init(&CT);
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    /* FOFSaveParticles = 1 (required by the finder), b = 0.2, min length 5. */
    set_fof_testpar(1, 0.2, 5);
    /* Mean separation 50 => comoving linking length 10 (kpc/h); the clumps below have mean
     * separations of 2-4 and the field lattice spacing is >= 100. */
    fof_init(50.0);
    init_forcetree_params(0.7);
}

/* ------------------------------------------------------------------ test 1: track table */

static void
payload_fill(struct SubTrack * t, int64_t T)
{
    subhalo_track_clear(t, T);
    t->Nbound = 7 * T + 1;
    t->Mbound = 0.5f * T;
    t->CorePos[0] = T + 0.25;
    t->SnapBirth = (int32_t)(T % 5);
    t->HostGroup = 1000 + T;
}

static int
payload_ok(const struct SubTrack * t, int64_t T, int32_t nbound)
{
    return t->TrackId == T && t->Nbound == nbound && t->Mbound == 0.5f * T
        && t->CorePos[0] == T + 0.25 && t->SnapBirth == (int32_t)(T % 5) && t->HostGroup == 1000 + T;
}

/* Fill the record of every TrackId in [from, to) homed on this rank. subhalo_table_redistribute
 * has already materialised a cleared record for each of them (the table rule of subhalo.h):
 * check that it is there, cleared, at its local index, then write the payload through
 * subhalo_table_get, which is how the driver fills new records. */
static void
fill_homed(struct SubTrackTable * tab, int64_t from, int64_t to, int * bad)
{
    int64_t T;
    for(T = from; T < to; T++) {
        if(subhalo_table_home_rank(tab, T) != tab->ThisTask)
            continue;
        struct SubTrack * g = subhalo_table_get(tab, T);
        LOCAL_CHECK(*bad, g != NULL, "fill: T=%ld homed on %d has no local record (NLocal %ld)\n", T, tab->ThisTask, tab->NLocal);
        if(!g)
            continue;
        LOCAL_CHECK(*bad, g == &tab->Track[subhalo_table_local_index(tab, T)],
                "fill: T=%ld is not at its local index %ld\n", T, subhalo_table_local_index(tab, T));
        LOCAL_CHECK(*bad, g->TrackId == T && g->Nbound == 0 && g->HostGroup == -1 && g->SnapBirth == -1,
                "fill: new record T=%ld is not cleared (TrackId %ld Nbound %d HostGroup %ld)\n", T, g->TrackId, g->Nbound, g->HostGroup);
        payload_fill(g, T);
    }
}

/* Every local record is in local-index order and carries its payload; NLocal matches the rule. */
static void
check_table_layout(struct SubTrackTable * tab, int64_t NTotal, const int32_t * nbound_of, int * bad)
{
    int64_t chunk = tab->Chunk;
    int64_t first = tab->ThisTask * chunk;
    int64_t last = first + chunk;
    if(last > NTotal) last = NTotal;
    int64_t expect = last > first ? last - first : 0;
    LOCAL_CHECK(*bad, tab->NLocal == expect, "rank %d NLocal %ld expected %ld (Chunk %ld NTotal %ld)\n",
            tab->ThisTask, tab->NLocal, expect, chunk, NTotal);
    int64_t i;
    for(i = 0; i < tab->NLocal && i < expect; i++) {
        int64_t T = first + i;
        LOCAL_CHECK(*bad, payload_ok(&tab->Track[i], T, nbound_of[T]), "rank %d local %ld: TrackId %ld expected %ld or payload wrong\n",
                tab->ThisTask, i, tab->Track[i].TrackId, T);
        struct SubTrack * g = subhalo_table_get(tab, T);
        LOCAL_CHECK(*bad, g == &tab->Track[i], "subhalo_table_get(%ld) did not return the local record\n", T);
    }
    /* Not-homed TrackIds give NULL. */
    int64_t T;
    for(T = 0; T < NTotal; T++)
        if(subhalo_table_home_rank(tab, T) != tab->ThisTask)
            LOCAL_CHECK(*bad, subhalo_table_get(tab, T) == NULL, "subhalo_table_get(%ld) non-NULL on rank %d\n", T, tab->ThisTask);
}

static void
check_fetch(struct SubTrackTable * tab, int64_t NTotal, const int32_t * nbound_of, int * bad)
{
    /* Every TrackId in reverse order plus duplicates. */
    int64_t n = NTotal + 3;
    int64_t * ids = malloc(n * sizeof(int64_t));
    struct SubTrack * out = malloc(n * sizeof(struct SubTrack));
    int64_t i;
    for(i = 0; i < NTotal; i++)
        ids[i] = NTotal - 1 - i;
    ids[NTotal] = 3 % NTotal;
    ids[NTotal + 1] = 3 % NTotal;
    ids[NTotal + 2] = NTotal - 1;
    subhalo_table_fetch(tab, ids, n, out, MPI_COMM_WORLD);
    for(i = 0; i < n; i++)
        LOCAL_CHECK(*bad, payload_ok(&out[i], ids[i], nbound_of[ids[i]]), "fetch[%ld]: TrackId %ld (wanted %ld) Nbound %d (wanted %d)\n",
                i, out[i].TrackId, ids[i], out[i].Nbound, nbound_of[ids[i]]);
    free(out);
    free(ids);
}

/* R3: every parameter appears once in SubhaloParameters and parses back bit for bit, including
 * a non-default neighbour count and doubles that %g would round. */
static void
test_subhalo_header_params(void ** state)
{
    const struct SubhaloParams saved = *subhalo_get_params();
    struct SubhaloParams p = default_subhalo_params();
    p.SubhaloNumNeighboursForReassignment = 7;
    p.SubhaloBoundMassPrecision = 0.9950001;
    p.SubhaloMaxGatherFraction = 0.1 + 0.2;
    p.SubhaloMajorProgenitorMassRatio = 0.80000000000000004 - 1e-16;
    p.SubhaloSourceRelaxFactor = 2.718281828459045;
    p.SubhaloTreeOpenAngle = 0.45000000000000001 + 1e-12;
    p.SubhaloSOSearchFactor = 1.0 / 3.0 + 1;
    p.SubhaloMajorMergerRatio = 0.30000001;
    p.SubhaloCollectiveMinPart = 123456789012L;
    p.SubhaloLookupBatchBytes = 3L * 1024 * 1024 + 1;
    subhalo_set_testpar(p);
    char buf[4096];
    subhalo_format_params(buf, sizeof(buf));
    CHECK(strlen(buf) < sizeof(buf) - 1, "SubhaloParameters fills its buffer (%zu bytes)\n", strlen(buf));
    char g[32];
    snprintf(g, sizeof(g), "%g", p.SubhaloBoundMassPrecision);
    CHECK(strtod(g, NULL) != p.SubhaloBoundMassPrecision, "%%g keeps 0.9950001 (%s): the fixture tests nothing\n", g);
#define D(f) {#f, 1, 0, &p.f}
#define I(f) {#f, 0, 0, &p.f}
#define L(f) {#f, 2, 0, &p.f}
    struct { const char * name; int kind; int seen; void * val; } f[] = {
        I(SubhaloFinderOn), D(SubhaloMaxGatherFraction), L(SubhaloCollectiveMinPart), I(SubhaloMaxUnbindIterations),
        I(SubhaloRestartFresh), I(SubhaloTracerTypes), I(SubhaloPotentialTypes), I(SubhaloMinNumPart),
        I(SubhaloMinNumTracerPart), I(SubhaloNumTracerHostFinding), L(SubhaloLookupBatchBytes),
        D(SubhaloMajorProgenitorMassRatio), D(SubhaloBoundMassPrecision), D(SubhaloSourceRelaxFactor),
        I(SubhaloPotentialSampleSize), I(SubhaloFinalPass), D(SubhaloTreeOpenAngle), I(SubhaloMergeTrapped),
        I(SubhaloSinkSymmetric), I(SubhaloCoreSize), I(SubhaloReassignGas), I(SubhaloNumNeighboursForReassignment),
        I(SubhaloUnbindWithThermal), I(SubhaloFieldSO), I(SubhaloSOBins), D(SubhaloSOSearchFactor),
        I(SubhaloSOMaxDoublings), I(SubhaloSORefineMinPart), D(SubhaloMajorMergerRatio), I(SubhaloGeometricParents),
        I(SubhaloDebugForceExchangeFailure),
    };
#undef D
#undef I
#undef L
    const int nf = sizeof(f) / sizeof(f[0]);
    int nlines = 0, bad = 0, k;
    char * save = NULL, * line;
    for(line = strtok_r(buf, "\n", &save); line; line = strtok_r(NULL, "\n", &save), nlines++) {
        char name[128], val[64];
        if(2 != sscanf(line, "%127s = %63s", name, val)) {
            LOCAL_CHECK(bad, 0, "unparsable line '%s'\n", line);
            continue;
        }
        for(k = 0; k < nf; k++)
            if(0 == strcmp(name, f[k].name))
                break;
        LOCAL_CHECK(bad, k < nf, "unknown parameter '%s'\n", name);
        if(k == nf)
            continue;
        f[k].seen++;
        int same = f[k].kind == 1 ? strtod(val, NULL) == *(double *) f[k].val
                 : f[k].kind == 2 ? strtoll(val, NULL, 10) == *(int64_t *) f[k].val
                 : strtol(val, NULL, 10) == *(int *) f[k].val;
        LOCAL_CHECK(bad, same, "%s = %s does not round-trip\n", name, val);
    }
    for(k = 0; k < nf; k++)
        LOCAL_CHECK(bad, f[k].seen == 1, "%s appears %d times\n", f[k].name, f[k].seen);
    LOCAL_CHECK(bad, nlines == nf, "%d lines for %d parameters\n", nlines, nf);
    assert_all_true(bad == 0);
    subhalo_set_testpar(saved);
}

static void
test_subhalo_table(void ** state)
{
    int NTask = get_ntask();
    int ThisTask = get_thistask();
    struct SubTrackTable tab;
    memset(&tab, 0, sizeof(tab));
    /* Room for two records: everything below grows past it. */
    subhalo_table_init(&tab, 2 * sizeof(struct SubTrack), MPI_COMM_WORLD);
    assert_true(tab.initialised);
    assert_int_equal(tab.NTask, NTask);
    assert_int_equal(tab.ThisTask, ThisTask);
    assert_int_equal(tab.NLocal, 0);
    assert_int_equal(tab.NTotal, 0);
    int64_t nalloc_initial = tab.NAlloc;

    const int64_t N1 = 5, N2 = 37, N3 = 41;
    int32_t nbound_of[64];
    int64_t T;
    for(T = 0; T < 64; T++)
        nbound_of[T] = 7 * T + 1;
    int bad = 0;

    /* Stage 1: 5 tracks. With 4 ranks Chunk = 2: ranks hold 2, 2, 1, 0 records (an empty rank). */
    tab.NTotal = N1;
    subhalo_table_redistribute(&tab, MPI_COMM_WORLD);
    assert_int_equal(tab.Chunk, (N1 + NTask - 1) / NTask);
    if(NTask == 4) {
        assert_int_equal(tab.Chunk, 2);
        assert_int_equal(subhalo_table_home_rank(&tab, 0), 0);
        assert_int_equal(subhalo_table_home_rank(&tab, 1), 0);
        assert_int_equal(subhalo_table_home_rank(&tab, 2), 1);
        assert_int_equal(subhalo_table_home_rank(&tab, 4), 2);
        assert_int_equal(subhalo_table_local_index(&tab, 3), 1);
        assert_int_equal(subhalo_table_local_index(&tab, 4), 0);
    }
    for(T = 0; T < N1; T++) {
        assert_int_equal(subhalo_table_home_rank(&tab, T), T / tab.Chunk);
        assert_int_equal(subhalo_table_local_index(&tab, T), T % tab.Chunk);
    }
    fill_homed(&tab, 0, N1, &bad);
    check_table_layout(&tab, N1, nbound_of, &bad);
    assert_all_true((bad == 0));
    check_fetch(&tab, N1, nbound_of, &bad);
    assert_all_true((bad == 0));

    /* Stage 2: 37 tracks, Chunk = 10: every old record moves to rank 0 and every rank grows past
     * the initial allocation to hold its new (cleared) records, which are then filled. */
    tab.NTotal = N2;
    subhalo_table_redistribute(&tab, MPI_COMM_WORLD);
    assert_int_equal(tab.Chunk, (N2 + NTask - 1) / NTask);
    if(NTask == 4) {
        assert_int_equal(tab.Chunk, 10);
        assert_int_equal(subhalo_table_home_rank(&tab, 36), 3);
        assert_int_equal(subhalo_table_local_index(&tab, 36), 6);
    }
    /* After the redistribution each rank holds exactly the records homed on it for TrackIds in
     * [0, N2): the old ones moved with their payload, the new ones cleared. */
    {
        int64_t expect = 0;
        for(T = 0; T < N2; T++)
            if(subhalo_table_home_rank(&tab, T) == ThisTask)
                expect++;
        LOCAL_CHECK(bad, tab.NLocal == expect, "after redistribute: NLocal %ld expected %ld\n", tab.NLocal, expect);
        for(T = 0; T < N1; T++) {
            struct SubTrack * g = subhalo_table_get(&tab, T);
            if(subhalo_table_home_rank(&tab, T) == ThisTask)
                LOCAL_CHECK(bad, g && payload_ok(g, T, nbound_of[T]), "after redistribute: old record %ld lost its payload\n", T);
        }
    }
    fill_homed(&tab, N1, N2, &bad);
    check_table_layout(&tab, N2, nbound_of, &bad);
    assert_all_true((bad == 0));
    /* Some rank must have grown past the initial allocation. */
    int64_t grew = tab.NAlloc > nalloc_initial, anygrew = 0;
    MPI_Allreduce(&grew, &anygrew, 1, MPI_INT64, MPI_MAX, MPI_COMM_WORLD);
    assert_true(anygrew);
    int64_t nsum = 0;
    MPI_Allreduce(&tab.NLocal, &nsum, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(nsum, N2);
    check_fetch(&tab, N2, nbound_of, &bad);
    assert_all_true((bad == 0));

    /* Store: rank r rewrites the records with T % NTask == r (records homed elsewhere). */
    {
        int64_t nmine = 0;
        struct SubTrack * mine = malloc((N2 + 1) * sizeof(struct SubTrack));
        for(T = 0; T < N2; T++) {
            if(T % NTask != ThisTask)
                continue;
            payload_fill(&mine[nmine], T);
            mine[nmine].Nbound = -(T + 100);
            nmine++;
        }
        subhalo_table_store(&tab, mine, nmine, MPI_COMM_WORLD);
        free(mine);
        for(T = 0; T < N2; T++)
            nbound_of[T] = -(T + 100);
    }
    check_table_layout(&tab, N2, nbound_of, &bad);
    check_fetch(&tab, N2, nbound_of, &bad);
    assert_all_true((bad == 0));

    /* Stage 3: 41 tracks, Chunk = 11: records move between non-zero ranks. */
    tab.NTotal = N3;
    subhalo_table_redistribute(&tab, MPI_COMM_WORLD);
    assert_int_equal(tab.Chunk, (N3 + NTask - 1) / NTask);
    for(T = N2; T < N3; T++)
        nbound_of[T] = 7 * T + 1;
    fill_homed(&tab, N2, N3, &bad);
    check_table_layout(&tab, N3, nbound_of, &bad);
    check_fetch(&tab, N3, nbound_of, &bad);
    assert_all_true((bad == 0));
    MPI_Allreduce(&tab.NLocal, &nsum, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(nsum, N3);

    subhalo_table_free(&tab);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* ------------------------------------------------------------------ test 2: sort key */

static void
make_keyed(struct particle_data * p, int type, int64_t grnr, int64_t track, int32_t bindrank)
{
    memset(p, 0, sizeof(*p));
    p->Type = type;
    p->GrNr = grnr;
    p->TrackId = track;
    p->BindRank = bindrank;
    p->BoundTrackId = track;
    p->ID = 1;
}

static int
cmp_key128(const uint64_t a[2], const uint64_t b[2])
{
    if(a[1] != b[1])
        return a[1] < b[1] ? -1 : 1;
    if(a[0] != b[0])
        return a[0] < b[0] ? -1 : 1;
    return 0;
}

struct keycase { int64_t GrNr; int64_t TrackId; int32_t BindRank; int same_as_prev; };

#define GRNR_MAX ((1LL << 39) - 1)
#define TRACK_MAX ((1LL << 40) - 2)

static const struct keycase key_order[] = {
    /* section 0, group 0: untagged first (BindRank ignored), then by TrackId, then BindRank */
    {0, -1, 0, 0},
    {0, -1, 17, 1},
    {0, 0, 0, 0},
    {0, 0, 5, 0},
    {0, 3, 0, 0},
    {0, 3, INT32_MAX, 0},
    /* group 1, TrackId + 1 crossing the 16-bit split of the two words */
    {1, -1, 3, 0},
    {1, 0, 0, 0},
    {1, 65534, 9, 0},
    {1, 65535, 0, 0},
    {1, 70000, 2, 0},
    {1, 70000, 3, 0},
    {1, 70001, 0, 0},
    {GRNR_MAX, -1, 0, 0},
    {GRNR_MAX, TRACK_MAX, INT32_MAX, 0},
    /* section 1: tagged particles outside groups, by TrackId then BindRank */
    {-1, 0, 0, 0},
    {-1, 0, 1, 0},
    {-1, 1, 0, 0},
    {-1, 65535, 9, 0},
    {-1, 65536, 0, 0},
    {-1, TRACK_MAX, INT32_MAX, 0},
};

static void
test_subhalo_sort_key(void ** state)
{
    const int n = sizeof(key_order) / sizeof(key_order[0]);
    struct particle_data p;
    uint64_t prev[2] = {0, 0}, cur[2];
    int i;
    /* Ordering of hand-built particles. */
    for(i = 0; i < n; i++) {
        make_keyed(&p, 1, key_order[i].GrNr, key_order[i].TrackId, key_order[i].BindRank);
        subhalo_sort_key(&p, cur);
        if(i > 0) {
            int c = cmp_key128(prev, cur);
            if(key_order[i].same_as_prev)
                CHECK(c == 0, "sort key case %d (GrNr %ld TrackId %ld BindRank %d) should equal the previous\n",
                        i, key_order[i].GrNr, key_order[i].TrackId, key_order[i].BindRank);
            else
                CHECK(c < 0, "sort key case %d (GrNr %ld TrackId %ld BindRank %d) not after the previous: %016lx %016lx <= %016lx %016lx\n",
                        i, key_order[i].GrNr, key_order[i].TrackId, key_order[i].BindRank,
                        (unsigned long) cur[1], (unsigned long) cur[0], (unsigned long) prev[1], (unsigned long) prev[0]);
        }
        prev[0] = cur[0];
        prev[1] = cur[1];
    }

    /* Exact bit layout of subhalo.h: out[1] = section<<63 | key<<24 | (T+1)>>16,
     * out[0] = ((T+1) & 0xFFFF)<<48 | BindRank<<16. */
    make_keyed(&p, 1, 5, 0x123456789ALL, 0x0BCDEF01);
    subhalo_sort_key(&p, cur);
    CHECK(cur[1] == 0x0000000005123456ULL && cur[0] == 0x789B0BCDEF010000ULL,
            "layout (5, 0x123456789A, 0x0BCDEF01): got %016lx %016lx\n", (unsigned long) cur[1], (unsigned long) cur[0]);
    make_keyed(&p, 1, -1, 2, 7);
    subhalo_sort_key(&p, cur);
    CHECK(cur[1] == 0x8000000000000000ULL && cur[0] == 0x0003000000070000ULL,
            "layout section 1 (T=2, BR=7): got %016lx %016lx\n", (unsigned long) cur[1], (unsigned long) cur[0]);
    make_keyed(&p, 1, 9, -1, 123);
    subhalo_sort_key(&p, cur);
    CHECK(cur[1] == (9ULL << 24) && cur[0] == 0, "layout untagged in group 9: got %016lx %016lx\n",
            (unsigned long) cur[1], (unsigned long) cur[0]);
    make_keyed(&p, 1, GRNR_MAX, TRACK_MAX, INT32_MAX);
    subhalo_sort_key(&p, cur);
    CHECK(cur[1] == 0x7FFFFFFFFFFFFFFFULL && cur[0] == 0xFFFF7FFFFFFF0000ULL,
            "layout at the range limits: got %016lx %016lx\n", (unsigned long) cur[1], (unsigned long) cur[0]);

    /* The local comparator: Type first, then the same key. Two types, shuffled. */
    const int m = 2 * n;
    struct particle_data * parts = malloc(m * sizeof(struct particle_data));
    for(i = 0; i < m; i++) {
        /* deterministic shuffle (5 is coprime with m = 42): position (i * 5) % m holds case i % n, type 4 or 1 */
        int c = i % n;
        make_keyed(&parts[(i * 5) % m], i < n ? 4 : 1, key_order[c].GrNr, key_order[c].TrackId, key_order[c].BindRank);
        parts[(i * 5) % m].ID = i;
    }
    qsort(parts, m, sizeof(struct particle_data), subhalo_order_by_type_and_key);
    int ntype1 = 0;
    for(i = 0; i < m; i++) {
        if(parts[i].Type == 1)
            ntype1++;
        if(i == 0)
            continue;
        CHECK(parts[i - 1].Type <= parts[i].Type, "comparator: Type out of order at %d\n", i);
        if(parts[i - 1].Type == parts[i].Type) {
            uint64_t a[2], b[2];
            subhalo_sort_key(&parts[i - 1], a);
            subhalo_sort_key(&parts[i], b);
            CHECK(cmp_key128(a, b) <= 0, "comparator: key out of order at %d\n", i);
        }
        CHECK(subhalo_order_by_type_and_key(&parts[i - 1], &parts[i]) <= 0, "comparator not consistent at %d\n", i);
    }
    assert_int_equal(ntype1, n);
    /* Antisymmetry on a strictly ordered pair. */
    {
        struct particle_data a, b;
        make_keyed(&a, 1, 0, 0, 0);
        make_keyed(&b, 1, 0, 0, 1);
        assert_true(subhalo_order_by_type_and_key(&a, &b) < 0);
        assert_true(subhalo_order_by_type_and_key(&b, &a) > 0);
        make_keyed(&b, 0, 7, 9, 9);
        assert_true(subhalo_order_by_type_and_key(&b, &a) < 0);
    }
    free(parts);
}

/* Opt-in (see the top of the file): must abort through endrun. */
static void
test_subhalo_sort_key_overflow(void ** state)
{
    const char * which = getenv("SUBHALO_TEST_SORTKEY_OVERFLOW");
    struct particle_data p;
    uint64_t key[2];
    if(!strcmp(which, "grnr"))
        make_keyed(&p, 1, GRNR_MAX + 1, 0, 0);
    else if(!strcmp(which, "trackid"))
        make_keyed(&p, 1, 0, TRACK_MAX + 1, 0);
    else
        make_keyed(&p, 1, 0, 0, -5);
    message(1, "Expecting subhalo_sort_key to abort on the '%s' range check.\n", which);
    subhalo_sort_key(&p, key);
    message(1, "subhalo_sort_key returned %016lx %016lx without aborting on '%s'\n",
            (unsigned long) key[1], (unsigned long) key[0], which);
    assert_true(0);
}

/* ------------------------------------------------------------------ test 3: tags through the exchange */

static int64_t tagT(MyIDType id) { return (int64_t) id * 1000003LL + (1LL << 35); }
static int32_t tagB(MyIDType id) { return (int32_t)(((uint32_t) id * 7919u) ^ 0x5A5A5A5Au) & 0x7fffffff; }
static int64_t tagBT(MyIDType id) { return -(int64_t) id - 2; }
static float tagP(MyIDType id) { return 0.125f * (float) (id % 11); }

static int
scatter_layout(int i, const void * userdata)
{
    int NTask = *(const int *) userdata;
    return (P[i].ID * 3 + 1) % NTask;
}

static int
home_layout(int i, const void * userdata)
{
    int NTask = *(const int *) userdata;
    return P[i].ID % NTask;
}

static void
check_exchanged_tags(int (*layout)(int, const void *), int NTask, int64_t TotNumPart)
{
    int ThisTask = get_thistask();
    int bad = 0;
    int64_t i, nongarbage = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage)
            continue;
        nongarbage++;
        LOCAL_CHECK(bad, layout(i, &NTask) == ThisTask, "particle ID %lu on rank %d, layout says %d\n",
                (unsigned long) P[i].ID, ThisTask, layout(i, &NTask));
        LOCAL_CHECK(bad, P[i].TrackId == tagT(P[i].ID) && P[i].BindRank == tagB(P[i].ID) && P[i].BoundTrackId == tagBT(P[i].ID)
                && P[i].TracerPenalty == tagP(P[i].ID),
                "particle ID %lu: tags (%ld %d %ld %g) expected (%ld %d %ld %g)\n", (unsigned long) P[i].ID,
                P[i].TrackId, P[i].BindRank, P[i].BoundTrackId, P[i].TracerPenalty, tagT(P[i].ID), tagB(P[i].ID), tagBT(P[i].ID),
                tagP(P[i].ID));
    }
    assert_all_true((bad == 0));
    int64_t tot = 0;
    MPI_Allreduce(&nongarbage, &tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(tot, TotNumPart);
}

static void
test_subhalo_exchange_tags(void ** state)
{
    walltime_init(&CT);
    int NTask = get_ntask();
    int ThisTask = get_thistask();
    const int npertype = 8;
    int64_t NType[6] = {npertype, npertype, npertype, npertype, npertype, npertype};
    /* TracerPenalty fills the padding after BindRank: the particle size is unchanged */
    assert_int_equal(offsetof(struct particle_data, TracerPenalty), offsetof(struct particle_data, BindRank) + 4);
    assert_int_equal(offsetof(struct particle_data, BoundTrackId), offsetof(struct particle_data, BindRank) + 8);
    assert_int_equal(sizeof(struct SubhaloIdHit), 72);

    particle_alloc_memory(PartManager, 1000., 1024);
    PartManager->NumPart = 6 * npertype;
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    slots_reserve(1, NType, SlotsManager);
    slots_setup_topology(PartManager, NType, SlotsManager);
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        P[i].Mass = 1;
        P[i].Pos[0] = P[i].Pos[1] = P[i].Pos[2] = 1.0 + (P[i].ID % 997);
        P[i].GrNr = -1;
        P[i].TrackId = tagT(P[i].ID);
        P[i].BindRank = tagB(P[i].ID);
        P[i].BoundTrackId = tagBT(P[i].ID);
        P[i].TracerPenalty = tagP(P[i].ID);
    }
    slots_setup_id(PartManager, SlotsManager);
    int64_t TotNumPart = 0;
    MPI_Allreduce(&PartManager->NumPart, &TotNumPart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    int fail = domain_exchange(scatter_layout, &NTask, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);
    assert_all_true((!fail));
    domain_test_id_uniqueness(PartManager);
    check_exchanged_tags(scatter_layout, NTask, TotNumPart);

    /* and back, with a garbage-collect in between */
    slots_gc_sorted(PartManager, SlotsManager);
    fail = domain_exchange(home_layout, &NTask, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);
    assert_all_true((!fail));
    check_exchanged_tags(home_layout, NTask, TotNumPart);

    slots_free(SlotsManager);
    myfree(P);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* ------------------------------------------------------------------ test 4: restart join */

#define GEN(g, base) ((((MyIDType)(g)) << 56) | (MyIDType)(base))

struct savedrec { MyIDType ID; int64_t TrackId; int32_t BindRank; int64_t BoundTrackId; };
struct localrec { MyIDType ID; int64_t TrackId; int32_t BindRank; int64_t BoundTrackId; int inherited; };

/* The synthetic state. Saved rows (as in PIG_xxx/1/) and the local particles, with the tag
 * each local particle must receive. */
static int
join_build(struct savedrec * s, struct localrec * l, int * nsaved, int * nlocal)
{
    int ns = 0, nl = 0, ninh = 0;
    int b, g;
#define SAVE(id, t, r, bt) do { s[ns].ID = (id); s[ns].TrackId = (t); s[ns].BindRank = (r); s[ns].BoundTrackId = (bt); ns++; } while(0)
#define LOCAL(id, t, r, bt, inh) do { l[nl].ID = (id); l[nl].TrackId = (t); l[nl].BindRank = (r); l[nl].BoundTrackId = (bt); l[nl].inherited = (inh); ninh += (inh); nl++; } while(0)
    /* Exact matches, 64-bit TrackIds, and tail particles whose BoundTrackId differs. */
    for(b = 100; b < 106; b++) {
        int64_t T = 5000000000LL + b;
        int64_t BT = (b % 2) ? T : 1;
        SAVE(GEN(0, b), T, b - 100, BT);
        LOCAL(GEN(0, b), T, b - 100, BT, 0);
    }
    /* Saved particles that no longer exist. */
    for(b = 200; b < 203; b++)
        SAVE(GEN(0, b), 7, 0, 7);
    /* Local particles with no saved record. */
    for(b = 300; b < 303; b++)
        LOCAL(GEN(0, b), -1, -1, -1, 0);
    /* Children spawned after the state inherit the saved parent's tag. */
    SAVE(GEN(0, 500), 50, 11, 50);
    LOCAL(GEN(0, 500), 50, 11, 50, 0);
    LOCAL(GEN(1, 500), 50, 11, 50, 1);
    LOCAL(GEN(2, 500), 50, 11, 50, 1);
    /* Two saved generations: exact matches keep their own, a new child takes the lineage
     * base's (the gen-0 parent's) tag, not its gen-1 sibling's. */
    SAVE(GEN(0, 600), 60, 1, 60);
    SAVE(GEN(1, 600), 61, 2, 61);
    LOCAL(GEN(0, 600), 60, 1, 60, 0);
    LOCAL(GEN(1, 600), 61, 2, 61, 0);
    LOCAL(GEN(3, 600), 60, 1, 60, 1);
    /* A long run of one base (1 saved + 16 local) that cannot fit on one rank's share of the
     * joint sort: it straddles a rank boundary. */
    SAVE(GEN(0, 700), 70, 3, 70);
    LOCAL(GEN(0, 700), 70, 3, 70, 0);
    for(g = 1; g < 16; g++)
        LOCAL(GEN(g, 700), 70, 3, 70, 1);
    /* Parent gone (or never local), child present: inherits, BoundTrackId included. */
    SAVE(GEN(0, 800), 80, 4, 81);
    LOCAL(GEN(5, 800), 80, 4, 81, 1);
    /* A saved untagged group member: neither it nor its child gets a tag. */
    SAVE(GEN(0, 900), -1, -1, -1);
    LOCAL(GEN(0, 900), -1, -1, -1, 0);
    LOCAL(GEN(1, 900), -1, -1, -1, 0);
    /* Only a child was saved (tagged): the gen-0 parent never inherits from its child, and a
     * new gen-2 sibling inherits only from the lineage base (gen 0), which is untagged here. */
    SAVE(GEN(1, 1000), 100, 5, 100);
    LOCAL(GEN(0, 1000), -1, -1, -1, 0);
    LOCAL(GEN(1, 1000), 100, 5, 100, 0);
    LOCAL(GEN(2, 1000), -1, -1, -1, 0);
    /* H11: a tagged parent with a child saved untagged. The exact match keeps the child's saved
     * (absent) tag instead of inheriting the parent's; a new sibling does inherit the parent's. */
    SAVE(GEN(0, 1100), 110, 0, 110);
    SAVE(GEN(1, 1100), -1, -1, -1);
    LOCAL(GEN(0, 1100), 110, 0, 110, 0);
    LOCAL(GEN(1, 1100), -1, -1, -1, 0);
    LOCAL(GEN(2, 1100), 110, 0, 110, 1);
    /* The same over a run long enough to straddle rank boundaries of the joint sort: exact
     * untagged matches after the boundary still win over the generation-0 parent. */
    SAVE(GEN(0, 1200), 120, 7, 120);
    for(g = 1; g < 15; g++) {
        SAVE(GEN(g, 1200), -1, -1, -1);
        LOCAL(GEN(g, 1200), -1, -1, -1, 0);
    }
    LOCAL(GEN(15, 1200), 120, 7, 120, 1);
#undef SAVE
#undef LOCAL
    *nsaved = ns;
    *nlocal = nl;
    return ninh;
}

static void
write_test_block(BigFile * bf, const char * name, const char * dtype, void * buf, int64_t nlocal, int64_t ntot)
{
    BigBlock bb;
    BigBlockPtr ptr;
    int NTask = get_ntask();
    int nfile = ntot > 0 ? 1 : 0;
    if(0 != big_file_mpi_create_block(bf, &bb, name, dtype, 1, nfile, ntot, MPI_COMM_WORLD))
        endrun(1, "test: failed to create block %s: %s\n", name, big_file_get_error_message());
    if(ntot > 0) {
        BigArray array;
        size_t dims[2] = {nlocal, 1};
        big_array_init(&array, buf, dtype, 2, dims, NULL);
        if(0 != big_block_seek(&bb, &ptr, 0))
            endrun(1, "test: failed to seek %s: %s\n", name, big_file_get_error_message());
        if(0 != big_block_mpi_write(&bb, &ptr, &array, NTask, MPI_COMM_WORLD))
            endrun(1, "test: failed to write %s: %s\n", name, big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD))
        endrun(1, "test: failed to close block %s: %s\n", name, big_file_get_error_message());
}

static int
rm_entry(const char * path, const struct stat * sb, int flag, struct FTW * ftwbuf)
{
    return remove(path);
}

static void
rm_tree(const char * path)
{
    if(get_thistask() == 0)
        nftw(path, rm_entry, 16, FTW_DEPTH | FTW_PHYS);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* Write a minimal PIG with the per-particle tag blocks for type 1 (empty blocks for the other
 * types) and the Header attributes a state reader may look at. Rows are spread over every rank
 * but the last. */
static void
write_test_pig(const char * pigname, const struct savedrec * s, int nsaved, int64_t NumTracksTotal)
{
    int NTask = get_ntask(), ThisTask = get_thistask();
    int writers = NTask > 1 ? NTask - 1 : 1;
    int j, n = 0;
    MyIDType * id = malloc((nsaved + 1) * sizeof(MyIDType));
    int64_t * tr = malloc((nsaved + 1) * sizeof(int64_t));
    int32_t * br = malloc((nsaved + 1) * sizeof(int32_t));
    int64_t * bt = malloc((nsaved + 1) * sizeof(int64_t));
    for(j = 0; j < nsaved; j++) {
        if(j % writers != ThisTask)
            continue;
        id[n] = s[j].ID;
        tr[n] = s[j].TrackId;
        br[n] = s[j].BindRank;
        bt[n] = s[j].BoundTrackId;
        n++;
    }
    BigFile bf;
    if(0 != big_file_mpi_create(&bf, pigname, MPI_COMM_WORLD))
        endrun(1, "test: failed to create %s: %s\n", pigname, big_file_get_error_message());

    BigBlock bh;
    if(0 != big_file_mpi_create_block(&bf, &bh, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD))
        endrun(1, "test: failed to create Header: %s\n", big_file_get_error_message());
    uint64_t saved[6] = {0, (uint64_t) nsaved, 0, 0, 0, 0};
    uint64_t ingroup[6] = {0, 0, 0, 0, 0, 0};
    uint64_t zero = 0, ntracks = NumTracksTotal;
    int32_t snap = 1, complete = 1, schema = SUBHALO_SCHEMA_VERSION;
    double atime = 0.5, box = 1000., dt = 0.;
    big_block_set_attr(&bh, "NumPartSavedTotal", saved, "u8", 6);
    big_block_set_attr(&bh, "NumPartInGroupTotal", ingroup, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &zero, "u8", 1);
    big_block_set_attr(&bh, "NumSubhalosTotal", &zero, "u8", 1);
    big_block_set_attr(&bh, "NumOrphansTotal", &zero, "u8", 1);
    big_block_set_attr(&bh, "NumTracksTotal", &ntracks, "u8", 1);
    big_block_set_attr(&bh, "SnapshotIndex", &snap, "i4", 1);
    big_block_set_attr(&bh, "SubhaloLastCallSnapshot", &snap, "i4", 1);
    big_block_set_attr(&bh, "SubhaloStateSnapshot", &snap, "i4", 1);
    big_block_set_attr(&bh, "SubhaloSchemaVersion", &schema, "i4", 1);
    big_block_set_attr(&bh, "SubhaloStateTime", &atime, "f8", 1);
    big_block_set_attr(&bh, "TimeSincePreviousFinderCall", &dt, "f8", 1);
    big_block_set_attr(&bh, "Time", &atime, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &box, "f8", 1);
    big_block_set_attr(&bh, "SubhaloStateComplete", &complete, "i4", 1);
    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD))
        endrun(1, "test: failed to close Header: %s\n", big_file_get_error_message());

    int t;
    char name[64];
    for(t = 0; t < 6; t++) {
        int64_t nl = (t == 1) ? n : 0;
        int64_t nt = (t == 1) ? nsaved : 0;
        snprintf(name, sizeof(name), "%d/ID", t);
        write_test_block(&bf, name, "u8", id, nl, nt);
        snprintf(name, sizeof(name), "%d/TrackId", t);
        write_test_block(&bf, name, "i8", tr, nl, nt);
        snprintf(name, sizeof(name), "%d/BindRank", t);
        write_test_block(&bf, name, "i4", br, nl, nt);
        snprintf(name, sizeof(name), "%d/BoundTrackId", t);
        write_test_block(&bf, name, "i8", bt, nl, nt);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD))
        endrun(1, "test: failed to close %s: %s\n", pigname, big_file_get_error_message());
    free(bt);
    free(br);
    free(tr);
    free(id);
}

static void
test_subhalo_join(void ** state)
{
    walltime_init(&CT);
    int NTask = get_ntask(), ThisTask = get_thistask();
    struct savedrec s[128];
    struct localrec l[128];
    int nsaved, nlocal;
    int ninherited = join_build(s, l, &nsaved, &nlocal);

    const char * dir = getenv("SUBHALO_TEST_DIR") ? getenv("SUBHALO_TEST_DIR") : "test_subhalo_tmp";
    char pigname[256];
    snprintf(pigname, sizeof(pigname), "%s/PIG_001", dir);
    rm_tree(dir);
    if(ThisTask == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);
    write_test_pig(pigname, s, nsaved, 5000000106LL);

    /* Local particles on every rank but the last (an empty rank). Tags pre-filled with junk:
     * the join must overwrite every one. */
    setup_pman(128, 1000.);
    int owners = NTask > 1 ? NTask - 1 : 1;
    int j;
    for(j = 0; j < nlocal; j++) {
        if(j % owners != ThisTask)
            continue;
        double pos[3] = {1.0 + j, 2.0, 3.0};
        add_dm_particle(l[j].ID, pos);
        P[PartManager->NumPart - 1].TrackId = 999;
        P[PartManager->NumPart - 1].BindRank = 999;
        P[PartManager->NumPart - 1].BoundTrackId = 999;
    }
    if(NTask > 1)
        assert_all_true((ThisTask != NTask - 1 || PartManager->NumPart == 0));

    int64_t ninh_local = subhalo_join_tags(pigname, PartManager, MPI_COMM_WORLD);

    int bad = 0;
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        const struct localrec * e = NULL;
        for(j = 0; j < nlocal; j++)
            if(l[j].ID == P[i].ID)
                e = &l[j];
        LOCAL_CHECK(bad, e != NULL, "join: unknown local ID %lx\n", (unsigned long) P[i].ID);
        if(!e)
            continue;
        if(e->TrackId < 0) {
            /* design 11: everything else gets TrackId = -1 and BoundTrackId = -1 */
            LOCAL_CHECK(bad, P[i].TrackId == -1 && P[i].BoundTrackId == -1,
                    "join: ID gen %lu base %lu expected untagged, got (%ld %d %ld)\n",
                    (unsigned long)(P[i].ID >> 56), (unsigned long)(P[i].ID & 0x00ffffffffffffffUL),
                    P[i].TrackId, P[i].BindRank, P[i].BoundTrackId);
        } else {
            LOCAL_CHECK(bad, P[i].TrackId == e->TrackId && P[i].BindRank == e->BindRank && P[i].BoundTrackId == e->BoundTrackId,
                    "join: ID gen %lu base %lu (%s) expected (%ld %d %ld), got (%ld %d %ld)\n",
                    (unsigned long)(P[i].ID >> 56), (unsigned long)(P[i].ID & 0x00ffffffffffffffUL),
                    e->inherited ? "inherited" : "exact",
                    e->TrackId, e->BindRank, e->BoundTrackId, P[i].TrackId, P[i].BindRank, P[i].BoundTrackId);
        }
    }
    assert_all_true((bad == 0));
    int64_t ninh_tot = 0;
    MPI_Allreduce(&ninh_local, &ninh_tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    CHECK(ninh_tot == ninherited, "join: %ld inherited tags reported, expected %d\n", ninh_tot, ninherited);
    /* The join neither adds nor removes particles. */
    int64_t ntot = 0;
    MPI_Allreduce(&PartManager->NumPart, &ntot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(ntot, nlocal);

    teardown_pman();
    rm_tree(dir);
}

/* ------------------------------------------------------------------ test 5: two clumps, five calls */

/* Box 1000 kpc/h, linking length 10. Clump A: 600 particles in a ball of radius 20 at 250;
 * clump B: 360 in radius 17 at 750; field: a 10^3 lattice of spacing 100 (singletons). IDs:
 * A [0, 600), B [600, 960), field [960, 1960).
 *  stage 0: initial;                       groups A (600, track 0) and B (360, track 1)
 *  stage 1: unchanged;                     identical result
 *  stage 2: the third of B with the largest x moves into A;   A 720, B 240
 *  stage 3: 600 field particles move into B, 30 A particles are dispersed into the field;
 *           B (840) is now the larger group, so the GrNr order swaps; A 690; the 30 dispersed
 *           particles lose tag 0: a central's source is its FOF group only (HBT FeedCentrals)
 *  stage 4: the 30 come back to A and B is translated onto A: one group of 1560; both are major
 *           members (840, 690), the kinetic distance picks track 0 and track 1 sinks into it. */
#define BOXSIZE 1000.
#define T5_NA 600
#define T5_NB 360
#define T5_NF 1000
#define T5_NTOT (T5_NA + T5_NB + T5_NF)
#define T5_NCAP (T5_NB / 3)
#define T5_NFMOVE 600
#define T5_NDISP 30
#define T5_NSTAGE 5

static const double T5_cA[3] = {250, 250, 250};
static const double T5_cB[3] = {750, 750, 750};
static const double T5_RA = 20, T5_RB = 17;
static double T5_xthr;

static void
t5_init(void)
{
    double xs[T5_NB];
    int i;
    for(i = 0; i < T5_NB; i++) {
        double p[3];
        rand_in_ball(T5_NA + i, 1, T5_RB, T5_cB, p);
        xs[i] = p[0];
    }
    qsort(xs, T5_NB, sizeof(double), cmp_double);
    T5_xthr = xs[T5_NB - T5_NCAP - 1];
}

static int
t5_is_cap(MyIDType id)
{
    if(id < T5_NA || id >= T5_NA + T5_NB)
        return 0;
    double p[3];
    rand_in_ball(id, 1, T5_RB, T5_cB, p);
    return p[0] > T5_xthr;
}

static void
t5_field_lattice(int64_t k, double out[3])
{
    out[0] = 100. * (k % 10);
    out[1] = 100. * ((k / 10) % 10);
    out[2] = 100. * (k / 100);
}

static void
t5_field_in_B(MyIDType id, double out[3])
{
    uint64_t salt = 3;
    do {
        rand_in_ball(id, salt, 0.9 * T5_RB, T5_cB, out);
        salt += 1000;
    } while(out[0] > T5_xthr - 2.0);
}

static void
t5_position(MyIDType id, int stage, double out[3])
{
    if(id < T5_NA) {
        if(stage == 3 && id < T5_NDISP) {
            out[0] = 50. + 100. * (id % 10);
            out[1] = 50. + 100. * (id / 10);
            out[2] = 550.;
        } else if(stage == 4 && id < T5_NDISP)
            rand_in_ball(id, 4, 0.8 * T5_RA, T5_cA, out);
        else
            rand_in_ball(id, 1, T5_RA, T5_cA, out);
        return;
    }
    if(id < T5_NA + T5_NB) {
        if(stage >= 2 && t5_is_cap(id)) {
            rand_in_ball(id, 2, 0.8 * T5_RA, T5_cA, out);
            return;
        }
        if(stage == 4) {
            /* Phase 2a: B lands on A as a full ball, so that the cores coincide and A sinks */
            rand_in_ball(id, 7, T5_RB, T5_cA, out);
            return;
        }
        rand_in_ball(id, 1, T5_RB, T5_cB, out);
        return;
    }
    int64_t k = id - T5_NA - T5_NB;
    if(stage >= 3 && k < T5_NFMOVE) {
        if(stage == 4)
            rand_in_ball(id, 7, T5_RB, T5_cA, out);
        else
            t5_field_in_B(id, out);
        return;
    }
    t5_field_lattice(k, out);
}

/* Expected FOF label (0: the group containing A's core, 1: B's, -1: none) and TrackId. */
static void
t5_expect(MyIDType id, int stage, int * label, int64_t * track)
{
    /* Phase 2a: at stage 4 the central decision (kinetic distance among the major members)
     * picks track 0 and track 1 sinks into it */
    if(id < T5_NA) {
        *label = (stage == 3 && id < T5_NDISP) ? -1 : 0;
        *track = (stage == 3 && id < T5_NDISP) ? -1 : 0;
        return;
    }
    if(id < T5_NA + T5_NB) {
        int cap = t5_is_cap(id);
        if(stage <= 1) { *label = 1; *track = 1; }
        else if(stage <= 3) { *label = cap ? 0 : 1; *track = cap ? 0 : 1; }
        else { *label = 0; *track = 0; }
        return;
    }
    int64_t k = id - T5_NA - T5_NB;
    int moved = k < T5_NFMOVE;
    if(stage < 3 || !moved) { *label = -1; *track = -1; }
    else if(stage == 3) { *label = 1; *track = 1; }
    else { *label = 0; *track = 0; }
}

static void
t5_setup_particles(int64_t MaxPart)
{
    int NTask = get_ntask(), ThisTask = get_thistask();
    t5_init();
    setup_pman(MaxPart, BOXSIZE);
    MyIDType id;
    for(id = 0; id < T5_NTOT; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3];
        t5_position(id, 0, pos);
        add_dm_particle(id, pos);
    }
}

static void
t5_move_to_stage(int stage)
{
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        t5_position(P[i].ID, stage, P[i].Pos);
        /* the particles displaced from A at stage 3 are cold (bound to A were they in its
         * source): they are dropped as the central's FOF-less particles, whatever their energy */
        P[i].Vel[0] = P[i].Vel[1] = P[i].Vel[2] = 0;
    }
}

static int
cmp_grnr_rank(const void * a, const void * b)
{
    const struct tagrec * x = a, * y = b;
    if(x->GrNr != y->GrNr)
        return (x->GrNr > y->GrNr) - (x->GrNr < y->GrNr);
    return (x->BindRank > y->BindRank) - (x->BindRank < y->BindRank);
}

/* Check that within every FOF group (GrNr >= 0) all particles carry one TrackId >= 0 with
 * BoundTrackId == TrackId and BindRank a permutation of 0..L-1. Returns 0 on success. */
static int
check_group_invariants(const struct tagrec * r, int64_t n)
{
    /* sort a copy by (GrNr, BindRank) */
    struct tagrec * c = malloc((n + 1) * sizeof(struct tagrec));
    memcpy(c, r, n * sizeof(struct tagrec));
    int64_t i, j;
    qsort(c, n, sizeof(struct tagrec), cmp_grnr_rank);
    int bad = 0;
    for(i = 0; i < n; i = j) {
        for(j = i; j < n && c[j].GrNr == c[i].GrNr; j++) {}
        if(c[i].GrNr < 0)
            continue;
        int64_t k;
        for(k = i; k < j; k++) {
            if(c[k].TrackId < 0 || c[k].TrackId != c[i].TrackId || c[k].BoundTrackId != c[k].TrackId) {
                if(bad < 10)
                    message(0, "group %ld: particle %lu has TrackId %ld BoundTrackId %ld (first member: %ld)\n",
                            c[i].GrNr, (unsigned long) c[k].ID, c[k].TrackId, c[k].BoundTrackId, c[i].TrackId);
                bad++;
            }
            if(c[k].BindRank != k - i) {
                if(bad < 10)
                    message(0, "group %ld: BindRank not dense: position %ld has BindRank %d\n", c[i].GrNr, k - i, c[k].BindRank);
                bad++;
            }
        }
    }
    free(c);
    return bad;
}

static double
dist2(const double a[3], const double b[3])
{
    double s = 0;
    int d;
    for(d = 0; d < 3; d++) {
        double x = NEAREST(a[d] - b[d], BOXSIZE);
        s += x * x;
    }
    return s;
}

/* The record of a track that holds a FOF group (serial or in place). members: the group's
 * particles (all of GrNr), n of them. */
static void
check_hosted_record(const struct SubTrack * rec, int64_t t, int64_t grnr, const struct tagrec * all, int64_t ntot, int stage, int inplace)
{
    /* members of the group */
    int64_t i, n = 0;
    double mean[3] = {0, 0, 0};
    const struct tagrec * mb = NULL;
    for(i = 0; i < ntot; i++) {
        if(all[i].GrNr != grnr)
            continue;
        n++;
        int d;
        for(d = 0; d < 3; d++)
            mean[d] += all[i].Pos[d];
        if(all[i].BindRank == 0)
            mb = &all[i];
    }
    int d;
    for(d = 0; d < 3; d++)
        mean[d] /= n;
    CHECK(rec->TrackId == t, "record %ld has TrackId %ld\n", t, rec->TrackId);
    CHECK(rec->HostGroup == grnr, "stage %d track %ld: HostGroup %ld, expected GrNr %ld\n", stage, t, rec->HostGroup, grnr);
    CHECK(rec->Nbound == n && rec->NboundExclusive == n, "stage %d track %ld: Nbound %d NboundExclusive %d, group length %ld\n",
            stage, t, rec->Nbound, rec->NboundExclusive, n);
    CHECK(rec->NboundType[1] == n, "stage %d track %ld: NboundType[1] %d != %ld\n", stage, t, rec->NboundType[1], n);
    for(d = 0; d < 6; d++)
        if(d != 1)
            CHECK(rec->NboundType[d] == 0, "stage %d track %ld: NboundType[%d] = %d\n", stage, t, d, rec->NboundType[d]);
    CHECK(fabs(rec->Mbound - n) < 1e-3 * n, "stage %d track %ld: Mbound %g != %ld\n", stage, t, rec->Mbound, n);
    CHECK(fabs(rec->MboundType[1] - n) < 1e-3 * n, "stage %d track %ld: MboundType[1] %g\n", stage, t, rec->MboundType[1]);
    CHECK(rec->Rank == 0 && rec->Depth == 0, "stage %d track %ld: Rank %d Depth %d\n", stage, t, rec->Rank, rec->Depth);
    CHECK(rec->NestedParentTrackId == -1 && rec->CentralTrackId == t, "stage %d track %ld: pid %ld upid %ld\n",
            stage, t, rec->NestedParentTrackId, rec->CentralTrackId);
    CHECK(rec->SinkTrackId == -1, "stage %d track %ld: SinkTrackId %ld\n", stage, t, rec->SinkTrackId);
    CHECK(rec->DescendantTrackId == t, "stage %d track %ld: DescendantTrackId %ld (itself while alive)\n", stage, t, rec->DescendantTrackId);
    /* HBT's SnapshotOfLastIsolation: -1 until the track first becomes a satellite, which never
     * happens in Phase 1. */
    CHECK(rec->SnapLastIsolation == -1, "stage %d track %ld: SnapLastIsolation %d, expected -1 (never a satellite)\n", stage, t, rec->SnapLastIsolation);
    for(d = 0; d < 3; d++) {
        CHECK(fabs(rec->CoMPos[d] - mean[d]) < 1e-4, "stage %d track %ld: CoMPos[%d] %.8g, mean %.8g\n", stage, t, d, rec->CoMPos[d], mean[d]);
        /* Phase 2a: the core of the 20 most bound lies near the centre of a cold clump (Phase 2b:
         * in place too, the same HBT recipe) */
        CHECK(fabs(rec->CorePos[d] - rec->CoMPos[d]) < 5, "stage %d track %ld: CorePos[%d] %.8g far from CoMPos %.8g\n", stage, t, d, rec->CorePos[d], rec->CoMPos[d]);
        /* the particles have zero velocity */
        CHECK(fabs(rec->CoMVel[d]) < 1e-6, "stage %d track %ld: CoMVel[%d] = %g\n", stage, t, d, rec->CoMVel[d]);
    }
    CHECK(mb != NULL, "stage %d track %ld: no particle with BindRank 0\n", stage, t);
    if(inplace)
        CHECK(rec->Flags & SUBHALO_FLAG_INPLACE, "stage %d track %ld: in-place key without SUBHALO_FLAG_INPLACE (Flags %d)\n", stage, t, rec->Flags);
    else
        CHECK(!(rec->Flags & SUBHALO_FLAG_INPLACE), "stage %d track %ld: serial key flagged in place (Flags %d)\n", stage, t, rec->Flags);
    CHECK(rec->MostBoundID == mb->ID, "stage %d track %ld: MostBoundID %lu, BindRank-0 particle %lu\n",
            stage, t, (unsigned long) rec->MostBoundID, (unsigned long) mb->ID);
    for(d = 0; d < 3; d++)
        CHECK(fabs(rec->MostBoundPos[d] - mb->Pos[d]) < 1e-4, "stage %d track %ld: MostBoundPos[%d]\n", stage, t, d);
    /* Phase 2a: BindRank is the energy order; in a cold uniform clump the most bound tenth lies
     * closer to the core than the least bound tenth. */
    double near = 0, far = 0;
    int64_t nnear = 0, nfar = 0;
    for(i = 0; i < ntot; i++) {
        if(all[i].GrNr != grnr || all[i].BindRank < 0)
            continue;
        if(all[i].BindRank < n / 10) {
            near += sqrt(dist2(all[i].Pos, rec->CorePos));
            nnear++;
        }
        else if(all[i].BindRank >= n - n / 10) {
            far += sqrt(dist2(all[i].Pos, rec->CorePos));
            nfar++;
        }
    }
    CHECK(nnear > 0 && nfar > 0 && near / nnear < 0.5 * far / nfar,
          "stage %d track %ld: the most bound tenth is not concentrated (mean distance %g vs %g)\n", stage, t,
          nnear ? near / nnear : -1, nfar ? far / nfar : -1);
}

/* Full check of one stage of the two-clump scenario. prev: the stage before (or NULL). */
static void
t5_check_stage(int stage, int ret, int64_t TotNgroups, const struct tagrec * r, int64_t n,
        const struct tagrec * prev, int inplace)
{
    int64_t i;
    CHECK(ret == 1, "stage %d: subhalo_find returned %d\n", stage, ret);
    CHECK(subhalo_last_call_complete() == 1, "stage %d: last call not complete\n", stage);
    CHECK(subhalo_last_call_snapshot() == stage, "stage %d: last call snapshot %d\n", stage, subhalo_last_call_snapshot());
    CHECK(n == T5_NTOT, "stage %d: %ld particles, expected %d\n", stage, n, T5_NTOT);
    for(i = 0; i < n; i++)
        CHECK(r[i].ID == (MyIDType) i, "stage %d: particle %ld missing\n", stage, i);

    /* The FOF result is as designed. */
    int64_t grnr[2] = {-2, -2}, size[2] = {0, 0};
    int bad = 0;
    for(i = 0; i < n; i++) {
        int label;
        int64_t track;
        t5_expect(r[i].ID, stage, &label, &track);
        if(label < 0) {
            LOCAL_CHECK(bad, r[i].GrNr == -1, "stage %d: field particle %ld in group %ld\n", stage, i, r[i].GrNr);
            continue;
        }
        if(grnr[label] == -2)
            grnr[label] = r[i].GrNr;
        LOCAL_CHECK(bad, r[i].GrNr == grnr[label] && r[i].GrNr >= 0, "stage %d: particle %ld (label %d) in group %ld, label group %ld\n",
                stage, i, label, r[i].GrNr, grnr[label]);
        size[label]++;
    }
    CHECK(bad == 0, "stage %d: the FOF groups are not the designed clumps\n", stage);
    int ngroups = stage == 4 ? 1 : 2;
    CHECK(TotNgroups == ngroups, "stage %d: %ld FOF groups, expected %d\n", stage, TotNgroups, ngroups);
    if(ngroups == 2) {
        CHECK(grnr[0] != grnr[1], "stage %d: both labels in one group\n", stage);
        /* FOF numbers groups by decreasing length */
        CHECK((size[0] > size[1]) == (grnr[0] < grnr[1]), "stage %d: GrNr order (%ld, %ld) does not follow length (%ld, %ld)\n",
                stage, grnr[0], grnr[1], size[0], size[1]);
    }

    /* Per-particle tags. */
    for(i = 0; i < n; i++) {
        int label;
        int64_t track;
        t5_expect(r[i].ID, stage, &label, &track);
        LOCAL_CHECK(bad, r[i].TrackId == track, "stage %d: particle %ld (label %d) TrackId %ld, expected %ld\n",
                stage, i, label, r[i].TrackId, track);
        if(track < 0)
            LOCAL_CHECK(bad, r[i].BoundTrackId == -1 && r[i].BindRank == -1, "stage %d: untagged particle %ld has BindRank %d BoundTrackId %ld\n",
                    stage, i, r[i].BindRank, r[i].BoundTrackId);
        else
            LOCAL_CHECK(bad, r[i].BoundTrackId == track, "stage %d: particle %ld BoundTrackId %ld != TrackId %ld\n", stage, i, r[i].BoundTrackId, track);
    }
    CHECK(bad == 0, "stage %d: particle tags wrong\n", stage);
    CHECK(check_group_invariants(r, n) == 0, "stage %d: group invariants violated\n", stage);

    /* Records. */
    struct SubTrackTable * tab = subhalo_get_table();
    CHECK(tab->NTotal == 2, "stage %d: NTotal %ld, expected 2\n", stage, tab->NTotal);
    struct SubTrack * rec = fetch_all_tracks(2);
    int t;
    for(t = 0; t < 2; t++) {
        CHECK(rec[t].SnapBirth == 0, "stage %d track %d: SnapBirth %d\n", stage, t, rec[t].SnapBirth);
        if(stage == 4 && t == 1)
            continue;
        /* the label whose group this track holds */
        int lab = (stage == 4) ? 0 : t;
        check_hosted_record(&rec[t], t, grnr[lab], r, n, stage, inplace);
    }
    if(stage == 4) {
        CHECK(rec[1].Nbound == 0, "stage 4: sunk track 1 has Nbound %d\n", rec[1].Nbound);
        CHECK(rec[1].SinkTrackId == 0, "stage 4: track 1 SinkTrackId %ld, expected 0\n", rec[1].SinkTrackId);
        CHECK(rec[1].SnapSink == 4 && rec[1].SnapDeath == 4, "stage 4: track 1 SnapSink %d SnapDeath %d\n", rec[1].SnapSink, rec[1].SnapDeath);
        CHECK(rec[1].DescendantTrackId == 0, "stage 4: track 1 DescendantTrackId %ld, expected 0\n", rec[1].DescendantTrackId);
    } else {
        CHECK(rec[0].SnapSink == -1 && rec[0].SnapDeath == -1 && rec[1].SnapSink == -1 && rec[1].SnapDeath == -1,
                "stage %d: a live track has SnapSink/SnapDeath set\n", stage);
    }
    /* History (plan 1 step 13): Mbound peaks. Track 0: 600, 600, 720, 720 (690 now), 1560;
     * track 1: 360, 360, 360 (240 now), 840, 840 (sunk). SnapMboundPeak is not checked where
     * two calls tie. */
    static const float peak0[5] = {600, 600, 720, 720, 1560};
    static const float peak1[5] = {360, 360, 360, 840, 840};
    static const int snap0[5] = {0, -9, 2, 2, 4};
    static const int snap1[5] = {0, -9, 0, 3, 3};
    CHECK(fabs(rec[0].MboundPeak - peak0[stage]) < 1e-2, "stage %d track 0: MboundPeak %g, expected %g\n", stage, rec[0].MboundPeak, peak0[stage]);
    CHECK(fabs(rec[1].MboundPeak - peak1[stage]) < 1e-2, "stage %d track 1: MboundPeak %g, expected %g\n", stage, rec[1].MboundPeak, peak1[stage]);
    if(snap0[stage] >= 0)
        CHECK(rec[0].SnapMboundPeak == snap0[stage], "stage %d track 0: SnapMboundPeak %d, expected %d\n", stage, rec[0].SnapMboundPeak, snap0[stage]);
    if(snap1[stage] >= 0)
        CHECK(rec[1].SnapMboundPeak == snap1[stage], "stage %d track 1: SnapMboundPeak %d, expected %d\n", stage, rec[1].SnapMboundPeak, snap1[stage]);
    free(rec);
}

static void
test_subhalo_fof_as_subhalo(void ** state)
{
    common_physics_setup();
    struct SubhaloParams sp = default_subhalo_params();
    subhalo_set_testpar(sp);
    Cosmology CP = test_cosmology();
    /* Every rank could hold every particle: the budget never forces a key in place. */
    t5_setup_particles(4 * T5_NTOT);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    CHECK(subhalo_get_table()->NTotal == 0, "fresh init: NTotal %ld\n", subhalo_get_table()->NTotal);
    {
        int bad = 0;
        int64_t i;
        for(i = 0; i < PartManager->NumPart; i++)
            LOCAL_CHECK(bad, P[i].TrackId == -1 && P[i].BoundTrackId == -1, "subhalo_init left a tag on %lu\n", (unsigned long) P[i].ID);
        assert_all_true((bad == 0));
    }

    struct tagrec * prev = NULL;
    int stage;
    for(stage = 0; stage < T5_NSTAGE; stage++) {
        if(stage > 0)
            t5_move_to_stage(stage);
        int64_t ngroups;
        int ret = run_finder_call(stage, &CP, &ngroups);
        int64_t n;
        struct tagrec * r = gather_tags(&n);
        t5_check_stage(stage, ret, ngroups, r, n, prev, 0);
        if(stage == 1) {
            /* identical to stage 0 */
            int64_t i;
            int bad = 0;
            for(i = 0; i < n; i++)
                LOCAL_CHECK(bad, r[i].TrackId == prev[i].TrackId && r[i].BindRank == prev[i].BindRank
                        && r[i].BoundTrackId == prev[i].BoundTrackId && r[i].GrNr == prev[i].GrNr,
                        "second call: particle %ld changed from (%ld %d %ld) to (%ld %d %ld)\n", i,
                        prev[i].TrackId, prev[i].BindRank, prev[i].BoundTrackId, r[i].TrackId, r[i].BindRank, r[i].BoundTrackId);
            CHECK(bad == 0, "the second call with the same particles changed the tags\n");
        }
        free(prev);
        prev = r;
    }
    free(prev);
    teardown_pman();
}

/* ------------------------------------------------------------------ test 6: in place and failure */

/* 16 small clumps (40..55 particles, radius 6) on a 4x4 grid, plus an 8^3 field lattice. */
#define T6_NCLUMP 16
#define T6_NFIELD 512

static int64_t
t6_clump_start(int c)
{
    return 40L * c + (int64_t) c * (c - 1) / 2;
}

static int64_t
t6_ntot(void)
{
    return t6_clump_start(T6_NCLUMP) + T6_NFIELD;
}

static int
t6_label(MyIDType id)
{
    int c;
    for(c = 0; c < T6_NCLUMP; c++)
        if((int64_t) id >= t6_clump_start(c) && (int64_t) id < t6_clump_start(c + 1))
            return c;
    return -1;
}

static void
t6_position(MyIDType id, double out[3])
{
    int c = t6_label(id);
    if(c >= 0) {
        double cen[3] = {125. + 250. * (c % 4), 125. + 250. * (c / 4), 125.};
        rand_in_ball(id, 6, 6.0, cen, out);
        return;
    }
    int64_t k = id - t6_clump_start(T6_NCLUMP);
    out[0] = 62.5 + 125. * (k % 8);
    out[1] = 62.5 + 125. * ((k / 8) % 8);
    out[2] = 62.5 + 125. * (k / 64);
}

/* Every clump is one group; newborn TrackIds are in GrNr order. */
static void
t6_check_newborn(const struct tagrec * r, int64_t n)
{
    int64_t grnr[T6_NCLUMP], track[T6_NCLUMP];
    int c, bad = 0;
    int64_t i;
    for(c = 0; c < T6_NCLUMP; c++)
        grnr[c] = track[c] = -2;
    for(i = 0; i < n; i++) {
        c = t6_label(r[i].ID);
        if(c < 0) {
            LOCAL_CHECK(bad, r[i].GrNr == -1 && r[i].TrackId == -1 && r[i].BoundTrackId == -1,
                    "field particle %lu: GrNr %ld TrackId %ld\n", (unsigned long) r[i].ID, r[i].GrNr, r[i].TrackId);
            continue;
        }
        if(grnr[c] == -2) {
            grnr[c] = r[i].GrNr;
            track[c] = r[i].TrackId;
        }
        LOCAL_CHECK(bad, r[i].GrNr == grnr[c] && r[i].GrNr >= 0 && r[i].TrackId == track[c],
                "clump %d particle %lu: GrNr %ld TrackId %ld (clump: %ld %ld)\n", c, (unsigned long) r[i].ID, r[i].GrNr, r[i].TrackId, grnr[c], track[c]);
    }
    CHECK(bad == 0, "the small clumps are not one group and one track each\n");
    CHECK(check_group_invariants(r, n) == 0, "group invariants violated\n");
    /* TrackId = position of the clump's GrNr among all clumps' GrNr */
    for(c = 0; c < T6_NCLUMP; c++) {
        int64_t below = 0;
        int c2;
        for(c2 = 0; c2 < T6_NCLUMP; c2++)
            if(grnr[c2] < grnr[c])
                below++;
        CHECK(track[c] == below, "clump %d (GrNr %ld) has TrackId %ld, expected %ld (host order)\n", c, grnr[c], track[c], below);
    }
}

/* In-place keys: the two-clump configuration with every key oversized. */
static void
t6_inplace_run(struct SubhaloParams sp, Cosmology * CP, int ncalls)
{
    subhalo_set_testpar(sp);
    t5_setup_particles(4 * T5_NTOT);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    CHECK(subhalo_get_table()->NTotal == 0, "re-init: NTotal %ld, the table was not reset\n", subhalo_get_table()->NTotal);
    struct tagrec * prev = NULL;
    int call;
    for(call = 0; call < ncalls; call++) {
        /* stage 0 positions every call; the stage number only sets the snapshot */
        int64_t ngroups, n;
        int ret = run_finder_call(call, CP, &ngroups);
        struct tagrec * r = gather_tags(&n);
        /* stage-1 expectations are the stage-0 ones with snapshot 1 */
        t5_check_stage(call, ret, ngroups, r, n, prev, 1);
        free(prev);
        prev = r;
    }
    free(prev);
    teardown_pman();
}

static void
test_subhalo_forced_failure(void ** state)
{
    common_physics_setup();
    Cosmology CP = test_cosmology();
    int NTask = get_ntask(), ThisTask = get_thistask();

    /* (a) SubhaloCollectiveMinPart = 1: every key with more than one particle is processed in
     * place (Exscan BindRank, two-stage centre), twice so tracks persist through that path. */
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloCollectiveMinPart = 1;
    t6_inplace_run(sp, &CP, 2);

    /* (b) SubhaloMaxGatherFraction so small that no rank can take any key: oversized by the
     * budget rule rather than by the testing knob. */
    sp = default_subhalo_params();
    sp.SubhaloMaxGatherFraction = 1e-9;
    t6_inplace_run(sp, &CP, 1);

    /* (c) The exchange failure path. 16 keys of 40-55 particles (760) and 512 field particles
     * (1272 in all, 318 per rank on 4 ranks) with MaxPart 560: the normal plan fits, but with
     * SubhaloDebugForceExchangeFailure every serial key goes to rank 0, which cannot hold 760. */
    sp = default_subhalo_params();
    subhalo_set_testpar(sp);
    const int64_t ntot = t6_ntot();
    const int64_t maxpart = NTask > 1 ? 560 : 2 * ntot;
    setup_pman(maxpart, BOXSIZE);
    MyIDType id;
    for(id = 0; id < (MyIDType) ntot; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3];
        t6_position(id, pos);
        add_dm_particle(id, pos);
    }
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);

    int64_t ngroups, n0, n1, n2;
    int ret = run_finder_call(0, &CP, &ngroups);
    CHECK(ret == 1, "failure test, first call: subhalo_find returned %d\n", ret);
    CHECK(ngroups == T6_NCLUMP, "failure test: %ld groups, expected %d\n", ngroups, T6_NCLUMP);
    struct tagrec * r0 = gather_tags(&n0);
    CHECK(n0 == ntot, "failure test: %ld particles, expected %ld\n", n0, ntot);
    t6_check_newborn(r0, n0);
    struct SubTrackTable * tab = subhalo_get_table();
    CHECK(tab->NTotal == T6_NCLUMP, "failure test: NTotal %ld\n", tab->NTotal);
    struct SubTrack * rec0 = fetch_all_tracks(T6_NCLUMP);
    CHECK(subhalo_last_call_snapshot() == 0, "failure test: last call snapshot %d\n", subhalo_last_call_snapshot());

    int64_t i;
    int bad = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        LOCAL_CHECK(bad, P[i].TracerPenalty == 0, "complete call: particle %lu keeps TracerPenalty %g\n",
                    (unsigned long) P[i].ID, P[i].TracerPenalty);
    assert_all_true((bad == 0));
    if(NTask > 1) {
        /* small unanimous penalties: a skipped call keeps them, the next complete call zeroes them */
        for(i = 0; i < PartManager->NumPart; i++)
            P[i].TracerPenalty = 1e-3f * (float) (1 + P[i].ID % 4);
        sp.SubhaloDebugForceExchangeFailure = 1;
        subhalo_set_testpar(sp);
        ret = run_finder_call(1, &CP, &ngroups);
        CHECK(ret == 0, "forced failure: subhalo_find returned %d, expected 0\n", ret);
        CHECK(subhalo_last_call_complete() == 0, "forced failure: last call reported complete\n");
        CHECK(subhalo_last_call_snapshot() == 0, "forced failure: last call snapshot moved to %d\n", subhalo_last_call_snapshot());
        CHECK(tab->NTotal == T6_NCLUMP, "forced failure: NTotal changed to %ld\n", tab->NTotal);
        struct tagrec * r1 = gather_tags(&n1);
        CHECK(n1 == ntot, "forced failure: %ld particles after the restore, expected %ld\n", n1, ntot);
        for(i = 0; i < PartManager->NumPart; i++)
            LOCAL_CHECK(bad, P[i].TracerPenalty == 1e-3f * (float) (1 + P[i].ID % 4),
                        "forced failure: particle %lu TracerPenalty %g changed\n", (unsigned long) P[i].ID, P[i].TracerPenalty);
        for(i = 0; i < n1 && i < n0; i++)
            LOCAL_CHECK(bad, r1[i].ID == r0[i].ID && r1[i].TrackId == r0[i].TrackId && r1[i].BindRank == r0[i].BindRank
                    && r1[i].BoundTrackId == r0[i].BoundTrackId,
                    "forced failure: particle %lu tags changed (%ld %d %ld) -> (%ld %d %ld)\n", (unsigned long) r0[i].ID,
                    r0[i].TrackId, r0[i].BindRank, r0[i].BoundTrackId, r1[i].TrackId, r1[i].BindRank, r1[i].BoundTrackId);
        CHECK(bad == 0, "forced failure modified tags\n");
        struct SubTrack * rec1 = fetch_all_tracks(T6_NCLUMP);
        int t;
        for(t = 0; t < T6_NCLUMP; t++) {
            CHECK(rec1[t].TrackId == rec0[t].TrackId && rec1[t].HostGroup == rec0[t].HostGroup && rec1[t].Nbound == rec0[t].Nbound
                    && rec1[t].SnapBirth == rec0[t].SnapBirth && rec1[t].SnapLastIsolation == rec0[t].SnapLastIsolation
                    && rec1[t].SinkTrackId == rec0[t].SinkTrackId && rec1[t].Mbound == rec0[t].Mbound
                    && rec1[t].CoMPos[0] == rec0[t].CoMPos[0] && rec1[t].MostBoundID == rec0[t].MostBoundID,
                    "forced failure: record %d changed (HostGroup %ld -> %ld, Nbound %d -> %d)\n", t,
                    rec0[t].HostGroup, rec1[t].HostGroup, rec0[t].Nbound, rec1[t].Nbound);
        }
        free(rec1);
        free(r1);

        /* The next call proceeds from the unchanged tags: same tracks as the first call. */
        sp.SubhaloDebugForceExchangeFailure = 0;
        subhalo_set_testpar(sp);
        ret = run_finder_call(2, &CP, &ngroups);
        CHECK(ret == 1, "after the failure: subhalo_find returned %d\n", ret);
        CHECK(subhalo_last_call_complete() == 1 && subhalo_last_call_snapshot() == 2, "after the failure: status %d %d\n",
                subhalo_last_call_complete(), subhalo_last_call_snapshot());
        CHECK(tab->NTotal == T6_NCLUMP, "after the failure: NTotal %ld (new tracks were created)\n", tab->NTotal);
        struct tagrec * r2 = gather_tags(&n2);
        CHECK(n2 == ntot, "after the failure: %ld particles\n", n2);
        bad = 0;
        for(i = 0; i < n2 && i < n0; i++)
            LOCAL_CHECK(bad, r2[i].TrackId == r0[i].TrackId && r2[i].BoundTrackId == r0[i].BoundTrackId,
                    "after the failure: particle %lu TrackId %ld, first call %ld\n", (unsigned long) r2[i].ID, r2[i].TrackId, r0[i].TrackId);
        CHECK(bad == 0, "the call after a failure did not continue the tracks\n");
        for(i = 0; i < PartManager->NumPart; i++)
            LOCAL_CHECK(bad, P[i].TracerPenalty == 0, "after the failure: particle %lu keeps TracerPenalty %g\n",
                        (unsigned long) P[i].ID, P[i].TracerPenalty);
        assert_all_true((bad == 0));
        CHECK(check_group_invariants(r2, n2) == 0, "after the failure: group invariants violated\n");
        free(r2);
    }
    free(rec0);
    free(r0);
    teardown_pman();
}

/* ------------------------------------------------------------------ test 7: the packing (H2) */

/* A synthetic key for subhalo_plan_pack: its particles by holder rank and type. */
struct synkey {
    int64_t cnt[4][6];
    int64_t tags[4];
};

/* Build this rank's directory slice and census messages for nkeys synthetic keys (g = 1..nkeys,
 * in decreasing length order), run the packing, and gather (target, oversized) of every key
 * to every rank. */
static void
synthetic_pack(const struct synkey * keys, int64_t nkeys, const struct SubhaloPlanRes * res, struct SubhaloPlanCfg * cfg,
               int * target, int * oversized, struct SubhaloPlanStats * stats)
{
    int NTask = get_ntask(), ThisTask = get_thistask();
    const int64_t KChunk = subhalo_key_chunk(nkeys, NTask);
    cfg->TotNgroups = nkeys;
    cfg->KChunk = KChunk;
    const int64_t lo = (int64_t) ThisTask * KChunk + 1;
    int64_t ndir = nkeys - lo + 1;
    if(ndir < 0) ndir = 0;
    if(ndir > KChunk) ndir = KChunk;
    struct SubhaloDirKey * dk = calloc(ndir + 1, sizeof(struct SubhaloDirKey));
    struct SubhaloDirMsg * in = calloc(4 * ndir + 1, sizeof(struct SubhaloDirMsg));
    int64_t k, nin = 0;
    int r, t;
    for(k = 0; k < ndir; k++) {
        const struct synkey * K = &keys[lo - 1 + k];
        dk[k].g = lo + k;
        dk[k].cstart = nin;
        dk[k].decider = -1;
        dk[k].target = -1;
        dk[k].has_group = 1;
        for(r = 0; r < 4 && r < NTask; r++) {
            int64_t sum = 0;
            for(t = 0; t < 6; t++)
                sum += K->cnt[r][t];
            if(sum == 0)
                continue;
            struct SubhaloDirMsg * m = &in[nin++];
            m->g = dk[k].g;
            m->rank = r;
            m->kind = 0;
            m->ntags = K->tags[r];
            for(t = 0; t < 6; t++) {
                m->cnt[t] = K->cnt[r][t];
                dk[k].LenType[t] += K->cnt[r][t];
                dk[k].Length += K->cnt[r][t];
            }
            dk[k].Tags += K->tags[r];
            if(dk[k].decider < 0)
                dk[k].decider = r;
        }
        dk[k].ccount = nin - dk[k].cstart;
    }
    subhalo_plan_pack(dk, ndir, in, res, cfg, stats, MPI_COMM_WORLD);
    /* every key's fate, everywhere */
    int * loc = calloc(2 * nkeys, sizeof(int));
    for(k = 0; k < ndir; k++) {
        loc[2 * (lo - 1 + k)] = dk[k].target + 1;
        loc[2 * (lo - 1 + k) + 1] = dk[k].oversized;
    }
    int * all = calloc(2 * nkeys, sizeof(int));
    MPI_Allreduce(loc, all, 2 * nkeys, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for(k = 0; k < nkeys; k++) {
        target[k] = all[2 * k] - 1;
        oversized[k] = all[2 * k + 1];
    }
    free(all);
    free(loc);
    free(in);
    free(dk);
}

/* Independent check of a plan: every key is placed or in place, and every destination's
 * particles and per-type particles fit the budgets computed from the final staying particles. */
static void
check_synthetic_plan(const struct synkey * keys, int64_t nkeys, const struct SubhaloPlanRes * allres, double frac,
                     const int * target, const int * oversized, const char * what)
{
    int NTask = get_ntask();
    int64_t k;
    int r, t, q;
    double load[4][7], cap[4][7];
    memset(load, 0, sizeof(load));
    for(r = 0; r < NTask && r < 4; r++) {
        int64_t dep[7] = {0};
        for(k = 0; k < nkeys; k++) {
            if(oversized[k]) continue;
            for(t = 0; t < 6; t++) {
                dep[t] += keys[k].cnt[r][t];
                dep[6] += keys[k].cnt[r][t];
            }
        }
        const double stay = allres[r].NumLive - dep[6];
        cap[r][0] = frac * (allres[r].MaxPart - stay);
        for(t = 0; t < 6; t++)
            cap[r][1 + t] = allres[r].slotelsize[t] > 0 ? allres[r].slotfree[t] + dep[t] : 1e30;
    }
    int bad = 0;
    for(k = 0; k < nkeys; k++) {
        if(oversized[k]) {
            CHECK(target[k] == -1, "%s: in-place key %ld has target %d\n", what, k + 1, target[k]);
            continue;
        }
        CHECK(target[k] >= 0 && target[k] < NTask, "%s: serial key %ld has target %d\n", what, k + 1, target[k]);
        for(q = 0; q < NTask && q < 4; q++)
            for(t = 0; t < 6; t++) {
                load[target[k]][0] += keys[k].cnt[q][t];
                load[target[k]][1 + t] += keys[k].cnt[q][t];
            }
    }
    for(r = 0; r < NTask && r < 4; r++)
        for(t = 0; t < 7; t++)
            if(load[r][t] > cap[r][t]) {
                message(0, "%s: destination %d dimension %d: load %g > capacity %g\n", what, r, t, load[r][t], cap[r][t]);
                bad++;
            }
    CHECK(bad == 0, "%s: the plan violates its capacities\n", what);
}

/* The eight-pass input of the audit appendix (four ranks, MaxPart 16871, fraction 0.5, DM only). */
static const int64_t appendix_rows[30][4] = {
    {102, 862, 758, 1116}, {1849, 289, 626, 23}, {1711, 159, 48, 748}, {96, 664, 256, 1573},
    {502, 49, 707, 1203}, {43, 54, 1064, 1275}, {1432, 1, 117, 861}, {618, 0, 700, 1028},
    {582, 1066, 319, 326}, {938, 55, 638, 608}, {62, 2092, 24, 15}, {16, 761, 643, 730},
    {1742, 20, 187, 3}, {263, 587, 73, 767}, {505, 166, 0, 993}, {33, 10, 319, 1249},
    {73, 960, 85, 271}, {686, 577, 113, 2}, {214, 90, 876, 182}, {224, 384, 407, 79},
    {257, 88, 207, 355}, {45, 0, 5, 814}, {30, 243, 311, 76}, {218, 37, 108, 70},
    {253, 116, 0, 1}, {70, 43, 82, 119}, {144, 0, 74, 54}, {0, 20, 112, 30},
    {34, 73, 36, 6}, {8, 8, 8, 13},
};

static void
test_subhalo_plan_pack(void ** state)
{
    int NTask = get_ntask(), ThisTask = get_thistask();
    if(NTask != 4) {
        message(0, "test_subhalo_plan_pack needs 4 ranks; skipped.\n");
        return;
    }
    struct SubhaloParams sp = default_subhalo_params();
    subhalo_set_testpar(sp);
    struct SubhaloPlanStats stats;
    int r, t;
    int64_t k;

    /* (a) the audit appendix: head-only, eight classification passes in the old code */
    {
        const int64_t nkeys = 30;
        struct synkey * keys = calloc(nkeys, sizeof(struct synkey));
        struct SubhaloPlanRes allres[4];
        memset(allres, 0, sizeof(allres));
        for(k = 0; k < nkeys; k++)
            for(r = 0; r < 4; r++)
                keys[k].cnt[r][1] = appendix_rows[k][r];
        for(r = 0; r < 4; r++) {
            allres[r].MaxPart = 16871;
            for(k = 0; k < nkeys; k++)
                allres[r].grp[1] += appendix_rows[k][r];
            allres[r].grp[6] = allres[r].grp[1];
            allres[r].NumLive = allres[r].grp[6];
            allres[r].freebytes = 1e15;
        }
        struct SubhaloPlanCfg cfg;
        subhalo_plan_default_cfg(&cfg, nkeys, NTask);
        cfg.GatherFraction = 0.5;
        cfg.CollectiveMinPart = 0;
        int target[30], oversized[30];
        synthetic_pack(keys, nkeys, &allres[ThisTask], &cfg, target, oversized, &stats);
        message(0, "appendix input: %ld serial, %ld in place, %d passes, %ld repaired in %d census rounds, fallback %d\n",
                stats.NSerial, stats.NOversized, stats.Iterations, stats.NRepaired, stats.CensusRounds, stats.Fallback);
        check_synthetic_plan(keys, nkeys, allres, 0.5, target, oversized, "appendix input");
        CHECK(stats.NSerial > 0, "appendix input: no serial key left\n");
        /* the same input with a cap of 2 passes runs the conservative fallback, which must
         * also publish a checked plan */
        cfg.MaxIter = 2;
        synthetic_pack(keys, nkeys, &allres[ThisTask], &cfg, target, oversized, &stats);
        message(0, "appendix input, 2 passes: %ld serial, %ld in place, %ld repaired in %d census rounds, fallback %d\n",
                stats.NSerial, stats.NOversized, stats.NRepaired, stats.CensusRounds, stats.Fallback);
        CHECK(stats.Fallback == 1 && stats.NRepaired > 0, "appendix input with 2 passes: fallback %d, %ld keys repaired\n", stats.Fallback, stats.NRepaired);
        check_synthetic_plan(keys, nkeys, allres, 0.5, target, oversized, "appendix input, fallback");
        free(keys);
    }

    /* (b) the type-mix input: 128 DM head keys, then a tail of 190 gas keys followed by 190
     * star keys, gas and star slots limited. The uniform-mix water-fill puts most of the gas
     * on the first ranks; the census must move keys in place until every type fits. */
    {
        const int64_t nhead = 128, ngas = 190, nstar = 190, nkeys = nhead + ngas + nstar;
        struct synkey * keys = calloc(nkeys, sizeof(struct synkey));
        for(k = 0; k < nkeys; k++) {
            const int holder = k % 4;
            if(k < nhead)
                keys[k].cnt[holder][1] = 100;
            else if(k < nhead + ngas)
                keys[k].cnt[holder][0] = 99;
            else
                keys[k].cnt[holder][4] = 99;
            keys[k].tags[holder] = 1;
        }
        struct SubhaloPlanRes allres[4];
        memset(allres, 0, sizeof(allres));
        for(r = 0; r < 4; r++) {
            allres[r].MaxPart = 100000;
            for(k = 0; k < nkeys; k++)
                for(t = 0; t < 6; t++) {
                    allres[r].grp[t] += keys[k].cnt[r][t];
                    allres[r].grp[6] += keys[k].cnt[r][t];
                }
            allres[r].NumLive = allres[r].grp[6];
            allres[r].slotelsize[0] = 100;
            allres[r].slotelsize[4] = 100;
            allres[r].slotfree[0] = 500;
            allres[r].slotfree[4] = 500;
            allres[r].freebytes = 1e15;
        }
        struct SubhaloPlanCfg cfg;
        subhalo_plan_default_cfg(&cfg, nkeys, NTask);
        cfg.GatherFraction = 0.5;
        cfg.CollectiveMinPart = 0;
        int * target = calloc(nkeys, sizeof(int)), * oversized = calloc(nkeys, sizeof(int));
        synthetic_pack(keys, nkeys, &allres[ThisTask], &cfg, target, oversized, &stats);
        message(0, "type-mix input: %ld serial, %ld in place, %d passes, %ld repaired in %d census rounds, fallback %d\n",
                stats.NSerial, stats.NOversized, stats.Iterations, stats.NRepaired, stats.CensusRounds, stats.Fallback);
        CHECK(stats.Fallback == 0, "type-mix input: the pass cap was reached\n");
        check_synthetic_plan(keys, nkeys, allres, 0.5, target, oversized, "type-mix input");
        /* the proportional tail spreads gas and stars over every rank: almost nothing goes in place */
        CHECK(stats.NSerial > nhead + (ngas + nstar) / 2, "type-mix input: only %ld serial keys left\n", stats.NSerial);
        /* tighter slots (no free slot beyond the departing particles): whatever the tail
         * proposes, the census must leave a plan that fits */
        for(r = 0; r < 4; r++)
            allres[r].slotfree[0] = allres[r].slotfree[4] = 0;
        synthetic_pack(keys, nkeys, &allres[ThisTask], &cfg, target, oversized, &stats);
        message(0, "type-mix input, no free slots: %ld serial, %ld in place, %d passes, %ld repaired in %d census rounds\n",
                stats.NSerial, stats.NOversized, stats.Iterations, stats.NRepaired, stats.CensusRounds);
        check_synthetic_plan(keys, nkeys, allres, 0.5, target, oversized, "type-mix input, no free slots");
        free(oversized);
        free(target);
        free(keys);
    }

    /* (c) Phase 5c, cost balance: 8 head keys of 1000 particles, 120 of 100, a tail of 300 keys of
     * 20, held round-robin; rank 0 has ten times the particle budget of the others. The old greedy
     * (most free budget) put the whole head on rank 0; the cost-aware head and the water-filled
     * tail must leave every rank within a few per cent of the mean estimated cost, inside the
     * capacities. */
    {
        const int64_t nbig = 8, nmid = 120, nsmall = 300, nkeys = nbig + nmid + nsmall;
        struct synkey * keys = calloc(nkeys, sizeof(struct synkey));
        for(k = 0; k < nkeys; k++) {
            const int holder = k % 4;
            keys[k].cnt[holder][1] = k < nbig ? 1000 : (k < nbig + nmid ? 100 : 20);
            keys[k].tags[holder] = 1;
        }
        struct SubhaloPlanRes allres[4];
        memset(allres, 0, sizeof(allres));
        for(r = 0; r < 4; r++) {
            allres[r].MaxPart = r == 0 ? 200000 : 20000;
            for(k = 0; k < nkeys; k++)
                allres[r].grp[1] += keys[k].cnt[r][1];
            allres[r].grp[6] = allres[r].grp[1];
            allres[r].NumLive = allres[r].grp[6];
            allres[r].freebytes = 1e15;
        }
        struct SubhaloPlanCfg cfg;
        subhalo_plan_default_cfg(&cfg, nkeys, NTask);
        cfg.GatherFraction = 0.5;
        cfg.CollectiveMinPart = 0;
        int * target = calloc(nkeys, sizeof(int)), * oversized = calloc(nkeys, sizeof(int));
        synthetic_pack(keys, nkeys, &allres[ThisTask], &cfg, target, oversized, &stats);
        check_synthetic_plan(keys, nkeys, allres, 0.5, target, oversized, "cost balance");
        double cost[4] = {0, 0, 0, 0}, headcost[4] = {0, 0, 0, 0}, tot = 0, mx = 0;
        int64_t nhead[4] = {0, 0, 0, 0};
        for(k = 0; k < nkeys; k++) {
            if(oversized[k] || target[k] < 0)
                continue;
            const int64_t len = keys[k].cnt[k % 4][1];
            cost[target[k]] += subhalo_key_cost(len, 0);
            if(k < nbig + nmid) {
                headcost[target[k]] += subhalo_key_cost(len, 0);
                nhead[target[k]]++;
            }
            tot += subhalo_key_cost(len, 0);
        }
        for(r = 0; r < 4; r++)
            if(cost[r] > mx)
                mx = cost[r];
        message(0, "cost balance: %ld serial, %ld in place; cost per rank %g %g %g %g (head %g %g %g %g, head keys %ld %ld %ld %ld), max/mean %.4f\n",
                stats.NSerial, stats.NOversized, cost[0], cost[1], cost[2], cost[3], headcost[0], headcost[1], headcost[2], headcost[3],
                nhead[0], nhead[1], nhead[2], nhead[3], mx / (tot / 4));
        CHECK(stats.NSerial == nkeys && stats.NOversized == 0, "cost balance: %ld serial, %ld in place\n", stats.NSerial, stats.NOversized);
        CHECK(mx / (tot / 4) < 1.03, "cost balance: the most loaded rank has %.4f x the mean cost\n", mx / (tot / 4));
        /* every rank gets head keys (the old greedy gave all of them to rank 0) */
        CHECK(nhead[1] > 0 && nhead[2] > 0 && nhead[3] > 0, "cost balance: head keys per rank %ld %ld %ld %ld\n", nhead[0], nhead[1],
              nhead[2], nhead[3]);
        free(oversized);
        free(target);
        free(keys);
    }
}

/* ------------------------------------------------------------------ Phase 5d: the local sort */

/* subhalo_local_sort (the record sort with the threaded permutation, and the indirect sort of P
 * it falls back to) leaves the same particle table at 1 to 4 threads: garbage last, then by key
 * (group, encoded travelling key, -1 first), TrackId, BindRank, ID, and the input order for full
 * ties (duplicated records). The reference is built independently of the code under test (audit
 * 5d R4): libc qsort of the input indices on the comparator below, made a total order by the
 * input index, then an out-of-place gather; every byte of the table is compared. */
static const struct particle_data * LocalSortIn;

static int
cmp_localsort_ref(const void * a, const void * b)
{
    const int64_t ia = *(const int64_t *) a, ib = *(const int64_t *) b;
    const struct particle_data * x = &LocalSortIn[ia], * y = &LocalSortIn[ib];
    const int64_t kx = subhalo_particle_key(x), ky = subhalo_particle_key(y);
    int c = (x->IsGarbage > y->IsGarbage) - (x->IsGarbage < y->IsGarbage);
    if(!c) c = (kx > ky) - (kx < ky);
    if(!c) c = (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    if(!c) c = (x->BindRank > y->BindRank) - (x->BindRank < y->BindRank);
    if(!c) c = (x->ID > y->ID) - (x->ID < y->ID);
    if(!c) c = (ia > ib) - (ia < ib);
    return c;
}

static void
test_subhalo_local_sort(void ** state)
{
    const int64_t N = 60000;
    setup_pman(N + 10, BOXSIZE);
    int64_t i;
    for(i = 0; i < N; i++) {
        double pos[3] = {urand(i, 91, 0) * BOXSIZE, urand(i, 91, 1) * BOXSIZE, urand(i, 91, 2) * BOXSIZE};
        add_dm_particle((MyIDType) (1000000 + splitmix64(i) % 5000000), pos);
        struct particle_data * p = &P[i];
        const double u = urand(i, 92, 0);
        if(u < 0.3)
            p->GrNr = -1;
        else if(u < 0.8)
            p->GrNr = 1 + (int64_t) (urand(i, 92, 1) * 300);
        else if(u < 0.95)
            p->GrNr = -(2 + (1 + (int64_t) (urand(i, 92, 2) * 300)));
        else
            p->GrNr = -(2 + ((1 + (int64_t) (urand(i, 92, 3) * 300)) | SUBHALO_KEY_FOFBIT));
        p->TrackId = urand(i, 93, 0) < 0.3 ? -1 : (int64_t) (urand(i, 93, 1) * 50);
        p->BindRank = urand(i, 94, 0) < 0.3 ? -1 : (int32_t) (urand(i, 94, 1) * 20);
        p->IsGarbage = urand(i, 95, 0) < 0.15;
        p->Hsml = (MyFloat) i;      /* the input position, to check the stability */
        /* full duplicates of an earlier record (same keys and ID): their order is the input order */
        if(i > 10 && urand(i, 96, 0) < 0.02) {
            const int64_t j = (int64_t) (urand(i, 96, 1) * (i - 1));
            const MyFloat h = p->Hsml;
            *p = P[j];
            p->Hsml = h;
        }
    }
    struct particle_data * in = malloc(sizeof(struct particle_data) * N);
    struct particle_data * ref = malloc(sizeof(struct particle_data) * N);
    memcpy(in, P, sizeof(struct particle_data) * N);
    const int saved = omp_get_max_threads();
    int64_t * idx = malloc(sizeof(int64_t) * N);
    for(i = 0; i < N; i++)
        idx[i] = i;
    LocalSortIn = in;
    qsort(idx, N, sizeof(int64_t), cmp_localsort_ref);
    for(i = 0; i < N; i++)
        memcpy(&ref[i], &in[idx[i]], sizeof(struct particle_data));
    int64_t bad = 0, ndup = 0;
    /* the fixture has full ties (resolved by the input index only) */
    for(i = 1; i < N; i++)
        if(ref[i - 1].ID == ref[i].ID && ref[i - 1].TrackId == ref[i].TrackId
           && ref[i - 1].BindRank == ref[i].BindRank && ref[i - 1].IsGarbage == ref[i].IsGarbage
           && subhalo_particle_key(&ref[i - 1]) == subhalo_particle_key(&ref[i]))
            ndup++;
    LOCAL_CHECK(bad, ndup > 100, "local sort reference: %ld full ties\n", ndup);
    free(idx);
    int mode, T;
    for(mode = 0; mode < 2; mode++)
        for(T = 1; T <= 4; T++) {
            omp_set_num_threads(T);
            subhalo_local_sort_set_mode(mode);
            memcpy(P, in, sizeof(struct particle_data) * N);
            subhalo_local_sort();
            /* every byte of the whole table */
            int64_t first = -1;
            if(memcmp(P, ref, sizeof(struct particle_data) * N))
                for(i = 0; i < N && first < 0; i++)
                    if(memcmp(&P[i], &ref[i], sizeof(struct particle_data)))
                        first = i;
            LOCAL_CHECK(bad, first < 0, "local sort, mode %d, %d threads: the table differs from the reference at %ld\n", mode, T, first);
        }
    subhalo_local_sort_set_mode(0);
    omp_set_num_threads(saved);
    message(0, "local sort: %ld particles (%ld full ties), record and indirect sorts at 1-4 threads equal the independent reference\n", N, ndup);
    assert_all_true(bad == 0);
    free(ref);
    free(in);
    teardown_pman();
}

/* ------------------------------------------------------------------ Phase 5c: the serial key scheduler */

struct sk_log {
    const char * base;
    int64_t k;
    int nthreads;
};
struct sk_ud {
    struct sk_log * log;
    int64_t nlog;
    int64_t * done;
    uint64_t * result;
    size_t * asize;         /* the logical arena size key k saw */
    const int64_t * extra;
    int64_t bad;
};

/* Writes a per-key sequence into the arena (bytes_per_part 8) and records which queue (arena,
 * alone or not) ran key k, in the order of the calls. Then takes the key's extra allowance and
 * the rest of the arena, writing all of it, and checks that the arena refuses anything beyond
 * (audit 5d R1: the key sees exactly its own allowance, not the larger buffer of its queue). */
static void
sk_fn(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata)
{
    struct sk_ud * u = (struct sk_ud *) userdata;
    int64_t slot;
    #pragma omp atomic capture
    slot = u->nlog++;
    u->log[slot].base = arena->base;
    u->log[slot].k = k;
    u->log[slot].nthreads = nthreads;
    u->asize[k] = arena->size;
    #pragma omp atomic
    u->done[k]++;
    uint64_t * buf = (uint64_t *) subhalo_arena_get(arena, sizeof(uint64_t) * seg->n);
    char * xb = (char *) subhalo_arena_get(arena, (size_t) u->extra[k]);
    if(!buf || !xb) {
        #pragma omp atomic
        u->bad++;
        return;
    }
    memset(xb, 0x5a, (size_t) u->extra[k]);
    const size_t rest = (arena->size - arena->used) & ~((size_t) 15);
    char * rb = (char *) subhalo_arena_get(arena, rest);
    if(!rb || arena->used + 16 <= arena->size || subhalo_arena_get(arena, 16)) {
        #pragma omp atomic
        u->bad++;
        return;
    }
    memset(rb, 0xa5, rest);
    uint64_t h = 0;
    int64_t i;
    for(i = 0; i < seg->n; i++) {
        buf[i] = splitmix64((uint64_t) seg->g * 1000003ULL + (uint64_t) i);
        h = splitmix64(h ^ buf[i]);
    }
    u->result[k] = h;
}

/* The reduced-team case (Phase 5 audit R2): each key spins for a fixed time and records the
 * worker that ran it. */
struct sk_timed {
    int * worker;
    double * dt;
    size_t * asize;
};

static void
sk_fn_timed(const struct SubhaloKeySeg * seg, int64_t k, struct SubhaloArena * arena, int nthreads, void * userdata)
{
    struct sk_timed * u = (struct sk_timed *) userdata;
    const double t0 = omp_get_wtime();
    (void) nthreads;
    u->asize[k] = arena->size;
    while(omp_get_wtime() - t0 < 2e-4 + 1e-8 * seg->n)
        ;
    u->worker[k] = omp_get_thread_num();
    u->dt[k] = omp_get_wtime() - t0;
}

static void
test_subhalo_serial_keys(void ** state)
{
    const int64_t nsegs = 90;
    struct SubhaloKeySeg * segs = calloc(nsegs, sizeof(struct SubhaloKeySeg));
    int64_t k, nserial = 0, largest = 0;
    for(k = 0; k < nsegs; k++) {
        segs[k].g = k + 1;
        /* two large keys, ties in n (equal cost, broken by g), a spread of small ones */
        segs[k].n = k == 3 ? 150000 : (k == 50 ? 25000 : (k % 5 == 0 ? 500 : 1 + (int64_t) (urand(k, 5, 0) * 3000)));
        segs[k].Length = segs[k].n;
        /* member counts (Phase 5d cost term); equal for the n = 500 keys so that their costs tie */
        segs[k].Tracks = k % 5 == 0 ? 3 : 1 + k % 13;
        segs[k].inplace = (k % 7 == 6);
        segs[k].ipindex = -1;
        if(!segs[k].inplace) {
            nserial++;
            if(segs[k].n > largest)
                largest = segs[k].n;
        }
    }
    /* a non-trivial extra allowance per key (the hosted-candidate records in the finder) */
    int64_t * extra = calloc(nsegs, sizeof(int64_t));
    for(k = 0; k < nsegs; k++)
        extra[k] = (k % 4) * 1000 + 64 * (segs[k].Tracks + 1) + k % 3;
#define SK_KEY_BYTES(kk) (sizeof(uint64_t) * (size_t) segs[kk].n + (size_t) extra[kk] + 64)
    uint64_t * ref = calloc(nsegs, sizeof(uint64_t));
    const int saved = omp_get_max_threads();
    int T;
    for(T = 1; T <= 4; T++) {
        omp_set_num_threads(T);
        struct sk_ud u;
        memset(&u, 0, sizeof(u));
        u.log = calloc(nsegs, sizeof(struct sk_log));
        u.done = calloc(nsegs, sizeof(int64_t));
        u.result = calloc(nsegs, sizeof(uint64_t));
        u.asize = calloc(nsegs, sizeof(size_t));
        u.extra = extra;
        subhalo_serial_reset_stats();
        const double peakfrac = subhalo_run_serial_keys(segs, nsegs, sizeof(uint64_t), extra, sk_fn, &u);
        const struct SubhaloSerialStats * ss = subhalo_serial_last_stats();
        int64_t bad = u.bad, i, j;
        LOCAL_CHECK(bad, u.bad == 0, "serial keys, %d threads: %ld keys could not take their allowance or got more\n", T, u.bad);
        /* each key saw exactly its own allowance, whichever keys shared its queue (audit 5d R1);
         * every key filled its arena, so the high-water fraction is 1 up to the alignment */
        for(k = 0; k < nsegs; k++)
            if(!segs[k].inplace)
                LOCAL_CHECK(bad, u.asize[k] == SK_KEY_BYTES(k), "serial keys, %d threads: key %ld saw an arena of %lu bytes, its allowance %lu\n", T,
                            k, (unsigned long) u.asize[k], (unsigned long) SK_KEY_BYTES(k));
        LOCAL_CHECK(bad, peakfrac <= 1 && peakfrac > 0.99, "serial keys, %d threads: peak fraction %g\n", T, peakfrac);
        /* the threaded queues' buffers (one per queue, live together) do not overlap, each as long
         * as the largest allowance of its keys */
        for(i = 0; i < u.nlog; i++)
            for(j = 0; j < u.nlog; j++) {
                if(u.log[i].nthreads > 1 || u.log[j].nthreads > 1 || u.log[i].base >= u.log[j].base)
                    continue;
                LOCAL_CHECK(bad, u.log[i].base + SK_KEY_BYTES(u.log[i].k) <= u.log[j].base,
                            "serial keys, %d threads: the arena of key %ld overlaps the queue of key %ld\n", T, u.log[i].k, u.log[j].k);
            }
        /* every serial key once, no in-place segment */
        for(k = 0; k < nsegs; k++)
            LOCAL_CHECK(bad, u.done[k] == (segs[k].inplace ? 0 : 1), "serial keys, %d threads: key %ld run %ld times\n", T, k, u.done[k]);
        /* the keys run alone are those above the thread's share of the cost */
        double total = 0;
        for(k = 0; k < nsegs; k++)
            if(!segs[k].inplace)
                total += subhalo_key_cost(segs[k].n, segs[k].Tracks);
        int64_t nalone = 0;
        for(i = 0; i < u.nlog; i++) {
            const int64_t kk = u.log[i].k;
            const int alone = T > 1 && subhalo_key_cost(segs[kk].n, segs[kk].Tracks) > total / T;
            nalone += alone;
            LOCAL_CHECK(bad, (u.log[i].nthreads > 1) == alone && (alone ? u.log[i].nthreads == T : u.log[i].nthreads == 1),
                        "serial keys, %d threads: key %ld (n %ld) ran with %d threads\n", T, kk, segs[kk].n, u.log[i].nthreads);
        }
        /* each queue (arena; alone or not) in non-increasing cost, ties by g; the threaded queues
         * LPT-balanced: their estimated costs differ by at most the largest threaded key */
        double qcost[8], maxjob = 0;
        const char * qbase[8];
        int nq = 0, q;
        for(i = 0; i < u.nlog; i++) {
            for(j = i + 1; j < u.nlog; j++) {
                if(u.log[j].base != u.log[i].base || (u.log[j].nthreads > 1) != (u.log[i].nthreads > 1))
                    continue;
                const int64_t a = u.log[i].k, b = u.log[j].k;
                const double ca = subhalo_key_cost(segs[a].n, segs[a].Tracks), cb = subhalo_key_cost(segs[b].n, segs[b].Tracks);
                LOCAL_CHECK(bad, ca > cb || (ca == cb && segs[a].g < segs[b].g),
                            "serial keys, %d threads: key %ld (n %ld) ran before key %ld (n %ld) on one queue\n", T, b, segs[b].n, a, segs[a].n);
                break;
            }
            if(u.log[i].nthreads > 1)
                continue;
            const double c = subhalo_key_cost(segs[u.log[i].k].n, segs[u.log[i].k].Tracks);
            if(c > maxjob)
                maxjob = c;
            for(q = 0; q < nq; q++)
                if(qbase[q] == u.log[i].base)
                    break;
            if(q == nq && nq < 8) {
                qbase[nq] = u.log[i].base;
                qcost[nq++] = 0;
            }
            if(q < 8)
                qcost[q] += c;
        }
        if(nq == T) {
            double lo = qcost[0], hi = qcost[0];
            for(q = 1; q < nq; q++) {
                if(qcost[q] < lo) lo = qcost[q];
                if(qcost[q] > hi) hi = qcost[q];
            }
            LOCAL_CHECK(bad, hi - lo <= maxjob * (1 + 1e-12), "serial keys, %d threads: queue costs %g to %g, largest key %g\n", T, lo, hi, maxjob);
        }
        else
            LOCAL_CHECK(bad, 0, "serial keys, %d threads: %d threaded queues\n", T, nq);
        /* the outcome does not depend on the thread count */
        for(k = 0; k < nsegs; k++) {
            if(T == 1)
                ref[k] = u.result[k];
            else
                LOCAL_CHECK(bad, u.result[k] == ref[k], "serial keys, %d threads: key %ld result differs\n", T, k);
        }
        LOCAL_CHECK(bad, ss->NKeys == nserial && ss->NAlone == nalone && ss->LargestN == largest && ss->NThreads == T
                    && ss->KeyTimeMax >= 0 && ss->KeyTimeMaxN > 0 && (nalone == 0 || ss->AloneWall > 0),
                    "serial keys, %d threads: stats %ld keys (%ld alone), largest %ld, %d threads\n", T, ss->NKeys, ss->NAlone, ss->LargestN,
                    ss->NThreads);
        message(0, "serial keys, %d threads: %ld keys, %ld alone, %d threaded queues, longest key %g s (%ld particles)\n", T, ss->NKeys,
                ss->NAlone, nq, ss->KeyTimeMax, ss->KeyTimeMaxN);
        assert_all_true(bad == 0);
        free(u.asize);
        free(u.result);
        free(u.done);
        free(u.log);
    }
    /* a deliberately reduced team (audit R2): 4 queues, but the scheduler called from inside an
     * active parallel region with one active level, so its own region runs on 1 worker, which
     * covers every queue. The reported workers, maximum and mean are the worker's: its maximum
     * busy time is the sum of every threaded key (not a quarter of it, as by queue). */
    {
        const int T = 4;
        const int savedlev = omp_get_max_active_levels();
        const int saveddyn = omp_get_dynamic();
        omp_set_dynamic(0);
        omp_set_num_threads(T);
        omp_set_max_active_levels(1);
        struct sk_timed u;
        u.worker = calloc(nsegs, sizeof(int));
        u.dt = calloc(nsegs, sizeof(double));
        u.asize = calloc(nsegs, sizeof(size_t));
        for(k = 0; k < nsegs; k++)
            u.worker[k] = -1;
        int outer = 0, inner_max = 0;
        subhalo_serial_reset_stats();
        #pragma omp parallel num_threads(2)
        {
            #pragma omp master
            {
                outer = omp_get_num_threads();
                inner_max = omp_get_max_threads();
                subhalo_run_serial_keys(segs, nsegs, sizeof(uint64_t), extra, sk_fn_timed, &u);
            }
        }
        omp_set_max_active_levels(savedlev);
        omp_set_dynamic(saveddyn);
        const struct SubhaloSerialStats * ss = subhalo_serial_last_stats();
        /* the keys of the threaded part: the alone keys are above the share of 4 queues */
        double total = 0, wsum = 0;
        for(k = 0; k < nsegs; k++)
            if(!segs[k].inplace)
                total += subhalo_key_cost(segs[k].n, segs[k].Tracks);
        int64_t bad = 0, nthr = 0;
        for(k = 0; k < nsegs; k++) {
            if(segs[k].inplace || subhalo_key_cost(segs[k].n, segs[k].Tracks) > total / T)
                continue;
            nthr++;
            wsum += u.dt[k];
            LOCAL_CHECK(bad, u.worker[k] == 0, "serial keys, reduced team: key %ld ran on worker %d\n", k, u.worker[k]);
        }
        /* one worker covering every queue: each key still sees exactly its own allowance */
        for(k = 0; k < nsegs; k++)
            if(!segs[k].inplace)
                LOCAL_CHECK(bad, u.asize[k] == SK_KEY_BYTES(k), "serial keys, reduced team: key %ld saw an arena of %lu bytes, its allowance %lu\n",
                            k, (unsigned long) u.asize[k], (unsigned long) SK_KEY_BYTES(k));
        message(0, "serial keys, reduced team: outer team %d, %d queues requested, %d worker(s); %ld threaded keys, their summed time %g s; "
                   "reported max %g s, sum %g s, mean %g s, threaded wall %g s\n", outer, inner_max, ss->NThreads, nthr, wsum,
                ss->ThreadMax, ss->ThreadSum, ss->ThreadMeanSum, ss->ThreadedWall);
        LOCAL_CHECK(bad, outer == 2 && inner_max == T && ss->NQueues == T && ss->NThreads == 1,
                    "serial keys, reduced team: outer %d, queues %d / %d, workers %d\n", outer, inner_max, ss->NQueues, ss->NThreads);
        /* one worker: its maximum is the sum and the mean; it covers every threaded key's time */
        LOCAL_CHECK(bad, nthr > 10 && ss->ThreadMax == ss->ThreadSum && ss->ThreadMeanSum == ss->ThreadSum && ss->ThreadMax >= wsum
                    && ss->ThreadMax <= ss->ThreadedWall * (1 + 1e-9) + 1e-6,
                    "serial keys, reduced team: max %g sum %g mean %g, keys %g s, wall %g s\n", ss->ThreadMax, ss->ThreadSum,
                    ss->ThreadMeanSum, wsum, ss->ThreadedWall);
        assert_all_true(bad == 0);
        free(u.asize);
        free(u.dt);
        free(u.worker);
    }
#undef SK_KEY_BYTES
    omp_set_num_threads(saved);
    free(extra);
    free(ref);
    free(segs);
}

/* ------------------------------------------------------------------ test 8: hostless, swallowed, units (direct) */

/* The hostless update and the host vote called directly on hand-placed particles, so that the
 * rank partition is chosen by the test (H6, H7, H10). */
static void
place_direct(MyIDType id, int type, double x, double y, double z, double mass, int64_t track, int swallowed, int64_t grnr)
{
    double pos[3] = {x, y, z};
    add_dm_particle(id, pos);
    struct particle_data * p = &P[PartManager->NumPart - 1];
    p->Type = type;
    p->Mass = mass;
    p->Swallowed = swallowed;
    p->GrNr = grnr;
    p->TrackId = track;
    p->BoundTrackId = track;
    p->BindRank = (int32_t)(id % 7);
    p->Vel[0] = 1;
    p->Vel[1] = -2;
    p->Vel[2] = 0.5;
}

static void
reset_direct_records(struct SubTrackTable * tab)
{
    int64_t T;
    for(T = 0; T < tab->NTotal; T++) {
        struct SubTrack * rec = subhalo_table_get(tab, T);
        if(!rec) continue;
        subhalo_track_clear(rec, T);
        rec->Nbound = rec->NboundPrev = 3;
        rec->Mbound = 3;
        rec->SnapBirth = 0;
        rec->HostGroup = 1;
        rec->Alive = 0;
    }
}

static void
test_subhalo_unhosted_direct(void ** state)
{
    walltime_init(&CT);
    int NTask = get_ntask(), ThisTask = get_thistask();
    struct SubhaloParams sp = default_subhalo_params();
    subhalo_set_testpar(sp);
    setup_pman(64, 1000.);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    struct SubTrackTable * tab = subhalo_get_table();
    tab->NTotal = 4;
    subhalo_table_redistribute(tab, MPI_COMM_WORLD);
    /* non-default velocity unit: 1e7 cm/s, so 1 internal unit at a = 0.5 is 200 km/s */
    subhalo_set_units(3.085678e21, 1.989e43, 1e7);
    const double atime = 0.5, vkms = 1e7 / (1e5 * atime);

    /* Three partitions of the periodic hostless track 0: particles at x = 0, 400, 700 in a box
     * of 1000 (the audit's 0, 4, 7 in a box of 10). The common reference is the minimum-ID
     * particle (x = 0); the nearest-image displacements 0, 400, -300 give 100/3. */
    int part[3][3] = {{0, 0, 0}, {0, 1, 1}, {3, 1, 2}};
    int p;
    for(p = 0; p < 3; p++) {
        PartManager->NumPart = 0;
        reset_direct_records(tab);
        const double xs[3] = {0, 400, 700};
        int j;
        for(j = 0; j < 3; j++)
            if(part[p][j] % NTask == ThisTask)
                place_direct(10 + j, 1, xs[j], 500, 500, 1, 0, 0, -1);
        /* track 1: two live DM particles and a swallowed tagged BH of mass 10 (H6) */
        if(ThisTask == 0) {
            place_direct(20, 1, 100, 100, 100, 1, 1, 0, -1);
            place_direct(22, 5, 300, 100, 100, 10, 1, 1, -1);
        }
        if(ThisTask == NTask - 1)
            place_direct(21, 1, 104, 100, 100, 1, 1, 0, -1);
        /* track 2: only a swallowed BH: no live particle, so it dies (H6) */
        if(ThisTask == NTask / 2)
            place_direct(30, 5, 600, 600, 600, 10, 2, 1, -1);
        /* track 3: no particle at all: a non-sunk dead track */
        int64_t nh = 0, nd = 0;
        subhalo_update_unhosted(tab, 7, atime, &nh, &nd, MPI_COMM_WORLD);
        CHECK(nh == 2 && nd == 2, "partition %d: %ld hostless, %ld dead; expected 2 and 2\n", p, nh, nd);
        struct SubTrack * rec = fetch_all_tracks(4);
        /* track 0: the periodic centre, independent of the partition */
        CHECK(fabs(rec[0].CoMPos[0] - 100. / 3.) < 1e-9 && fabs(rec[0].CoMPos[1] - 500) < 1e-9 && fabs(rec[0].CoMPos[2] - 500) < 1e-9,
              "partition %d: hostless centre (%.10g %.10g %.10g), expected (%.10g 500 500)\n", p,
              rec[0].CoMPos[0], rec[0].CoMPos[1], rec[0].CoMPos[2], 100. / 3.);
        CHECK(rec[0].Nbound == 3 && rec[0].HostGroup == -1 && rec[0].SnapDeath == -1 && rec[0].CentralTrackId == 0,
              "partition %d: track 0 Nbound %d HostGroup %ld SnapDeath %d\n", p, rec[0].Nbound, rec[0].HostGroup, rec[0].SnapDeath);
        /* H10: velocities in km/s with the run's unit */
        CHECK(fabs(rec[0].CoMVel[0] - vkms) < 1e-3 && fabs(rec[0].CoMVel[1] + 2 * vkms) < 1e-3 && fabs(rec[0].CoMVel[2] - 0.5 * vkms) < 1e-3,
              "partition %d: hostless CoMVel (%g %g %g), expected (%g %g %g)\n", p,
              rec[0].CoMVel[0], rec[0].CoMVel[1], rec[0].CoMVel[2], vkms, -2 * vkms, 0.5 * vkms);
        /* track 1: the swallowed BH counts neither particles nor mass */
        CHECK(rec[1].Nbound == 2 && rec[1].NboundType[1] == 2 && rec[1].NboundType[5] == 0 && fabs(rec[1].Mbound - 2) < 1e-6
              && rec[1].MboundType[5] == 0,
              "partition %d: track 1 with a swallowed BH: Nbound %d (type 5: %d), Mbound %g\n", p, rec[1].Nbound, rec[1].NboundType[5], rec[1].Mbound);
        CHECK(fabs(rec[1].CoMPos[0] - 102) < 1e-9, "partition %d: track 1 centre %g, expected 102 (the BH excluded)\n", p, rec[1].CoMPos[0]);
        /* tracks 2 and 3: dead, not sunk */
        int t;
        for(t = 2; t < 4; t++)
            CHECK(rec[t].Nbound == 0 && rec[t].SnapDeath == 7 && rec[t].SinkTrackId == -1 && rec[t].SnapSink == -1
                  && rec[t].DescendantTrackId == -1,
                  "partition %d: track %d should be dead and not sunk: Nbound %d SnapDeath %d SinkTrackId %ld\n",
                  p, t, rec[t].Nbound, rec[t].SnapDeath, rec[t].SinkTrackId);
        free(rec);
        /* the swallowed particles keep their tags */
        int bad = 0;
        int64_t i;
        for(i = 0; i < PartManager->NumPart; i++)
            if(P[i].Swallowed)
                LOCAL_CHECK(bad, P[i].TrackId == (P[i].ID == 22 ? 1 : 2), "swallowed particle %lu lost its tag\n", (unsigned long) P[i].ID);
        assert_all_true((bad == 0));
    }

    /* The host vote (H6): track 1 has two live particles in group 5 and three swallowed BHs
     * outside groups. Counting the BHs would give "no group" the strict majority. */
    PartManager->NumPart = 0;
    reset_direct_records(tab);
    for(p = 0; p < 5; p++) {
        if(p % NTask != ThisTask)
            continue;
        if(p < 2)
            place_direct(40 + p, 1, 100 + p, 100, 100, 1, 1, 0, 5);
        else
            place_direct(40 + p, 5, 300 + p, 100, 100, 1, 1, 1, -1);
    }
    subhalo_host_vote(tab, MPI_COMM_WORLD);
    struct SubTrack * rec = fetch_all_tracks(4);
    CHECK(rec[1].Host == 5, "host vote with swallowed BHs: Host %ld, expected group 5\n", rec[1].Host);
    free(rec);
    /* and the live-particle count of the diagnostics */
    int64_t ntag[6], ntagtot[6];
    subhalo_count_tagged(PartManager, ntag);
    MPI_Allreduce(ntag, ntagtot, 6, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    CHECK(ntagtot[1] == 2 && ntagtot[5] == 0, "tagged count: type 1 %ld type 5 %ld, expected 2 and 0\n", ntagtot[1], ntagtot[5]);

    /* Collision penalties (SIDM): track 2's tracers of rank 0-4 are in group 7, 5-9 in group 8.
     * Unpenalised, 7 wins (2.61 to 1.38); a penalty 0.1 on the first five keeps 7 (2.36), a
     * penalty 5 (weight 0.0067) hands the vote to 8. */
    const float pen[3] = {0, 0.1f, 5};
    const int64_t expect[3] = {7, 7, 8};
    int c;
    for(c = 0; c < 3; c++) {
        PartManager->NumPart = 0;
        reset_direct_records(tab);
        for(p = 0; p < 10; p++) {
            if((p * 3 + 1) % NTask != ThisTask)
                continue;
            place_direct(60 + p, 1, 200 + p, 100, 100, 1, 2, 0, p < 5 ? 7 : 8);
            P[PartManager->NumPart - 1].BindRank = p;
            P[PartManager->NumPart - 1].TracerPenalty = p < 5 ? pen[c] : 0;
        }
        subhalo_host_vote(tab, MPI_COMM_WORLD);
        rec = fetch_all_tracks(4);
        CHECK(rec[2].Host == expect[c] && rec[2].NTracers == 10 && rec[2].TracerIDs[0] == 60 && rec[2].TracerIDs[9] == 69,
              "penalised host vote %d: Host %ld (expected %ld), %d tracers, first %lu last %lu\n", c, rec[2].Host, expect[c],
              rec[2].NTracers, (unsigned long) rec[2].TracerIDs[0], (unsigned long) rec[2].TracerIDs[9]);
        free(rec);
    }

    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    teardown_pman();
}

/* The shared weighted plurality of the host and descendant votes. */
static void
test_subhalo_weighted_plurality(void ** state)
{
    double top, second;
    const int64_t none[1] = {0};
    const double w1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    /* no ballot, and only negative candidates */
    CHECK(subhalo_weighted_plurality(none, w1, 0, &top, &second) == SUBHALO_NOTRACK && top == 0 && second == 0,
          "plurality of nothing: top %g second %g\n", top, second);
    const int64_t neg[3] = {-1, -2, -1};
    CHECK(subhalo_weighted_plurality(neg, w1, 3, &top, &second) == SUBHALO_NOTRACK && top == 0 && second == 0,
          "plurality of negatives: top %g second %g\n", top, second);
    CHECK(subhalo_weighted_plurality(neg, w1, 3, NULL, NULL) == SUBHALO_NOTRACK, "plurality without the sums\n");
    /* equal sums go to the smaller candidate, whatever the ballot order */
    const int64_t tie[4] = {7, 3, 7, 3}, tie2[4] = {3, 7, 3, 7};
    CHECK(subhalo_weighted_plurality(tie, w1, 4, &top, &second) == 3 && top == 2 && second == 2, "tie: top %g second %g\n", top, second);
    CHECK(subhalo_weighted_plurality(tie2, w1, 4, NULL, NULL) == 3, "tie in the other order\n");
    /* unit weights reproduce the counts exactly (negative ballots skipped) */
    const int64_t cnt[8] = {5, 2, 5, 9, 2, 5, -1, 9};
    CHECK(subhalo_weighted_plurality(cnt, w1, 8, &top, &second) == 5 && top == 3 && second == 2,
          "counts: top %g second %g\n", top, second);
    const int64_t cnt2[6] = {9, 2, -1, 9, 2, -1};
    CHECK(subhalo_weighted_plurality(cnt2, w1, 6, &top, &second) == 2 && top == 2 && second == 2, "count tie: top %g\n", top);
    /* exp weights: two unpenalised ballots beat three with penalty 5 */
    const int64_t mix[5] = {4, 4, 4, 8, 8};
    const double e5 = subhalo_tracer_weight(5);
    const double wm[5] = {e5, e5, e5, 1, 1};
    CHECK(subhalo_weighted_plurality(mix, wm, 5, &top, &second) == 8 && top == 2 && fabs(second - 3 * exp(-5.)) < 1e-12,
          "exp weights: top %g second %g\n", top, second);
    /* a heavy negative ballot does not count */
    const int64_t nb[2] = {-1, 6};
    const double wn[2] = {10, 0.5};
    CHECK(subhalo_weighted_plurality(nb, wn, 2, &top, &second) == 6 && top == 0.5 && second == 0, "negative ballot: top %g\n", top);
    /* the weight of a zero or negative penalty is exactly 1 */
    CHECK(subhalo_tracer_weight(0) == 1.0 && subhalo_tracer_weight(-1) == 1.0 && fabs(subhalo_tracer_weight(1) - exp(-1.0)) < 1e-15,
          "tracer weights %g %g %g\n", subhalo_tracer_weight(0), subhalo_tracer_weight(-1), subhalo_tracer_weight(1));
}

/* ------------------------------------------------------------------ test 9: life cycle with the PIG */

/* Move every particle of rank `er` to rank er - 1 by giving its top leaves to that rank (the
 * leaves of a task are a contiguous range), so that `er` is an empty interior rank. */
static void
empty_rank_domain(DomainDecomp * dd, int er)
{
    int NTask = get_ntask();
    if(er <= 0 || er >= NTask)
        return;
    int j;
    for(j = dd->Tasks[er].StartLeaf; j < dd->Tasks[er].EndLeaf; j++)
        dd->TopLeaves[j].Task = er - 1;
    dd->Tasks[er - 1].EndLeaf = dd->Tasks[er].EndLeaf;
    dd->Tasks[er].StartLeaf = dd->Tasks[er].EndLeaf;
    if(domain_maintain(dd, NULL))
        endrun(1, "test: could not empty rank %d\n", er);
}

/* One finder call as in run.c, with rank `er` empty (if er > 0) and the PIG written to dir. */
static int
lifecycle_call(int snap, double atime, Cosmology * CP, int er, const char * dir, int64_t * TotNgroups, int64_t * NumPartEmpty)
{
    /* every rank holds particles before the decomposition (the host's domain_decompose_full
     * does not cope with an empty rank); the emptied rank is made after it */
    int NTask = get_ntask();
    slots_gc_sorted(PartManager, SlotsManager);
    if(domain_exchange(home_layout, &NTask, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD))
        endrun(1, "test: redistribution before the decomposition failed\n");
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    empty_rank_domain(&ddecomp, er);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    *TotNgroups = fof.TotNgroups;
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, 0, MPI_COMM_WORLD);
    int64_t nloc = (get_thistask() == er) ? PartManager->NumPart : 0;
    MPI_Allreduce(&nloc, NumPartEmpty, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(dir) {
        double MassTable[6] = {0};
        int domain_needed = fof_save_groups(&fof, dir, "PIG", snap, CP, atime, MassTable, 0, MPI_COMM_WORLD);
        fof_finish(&fof);
        if(domain_needed) {
            slots_gc_sorted(PartManager, SlotsManager);
            if(domain_maintain(&ddecomp, NULL))
                endrun(1, "test: domain exchange after the PIG write failed\n");
            slots_gc_sorted(PartManager, SlotsManager);
        }
    }
    else
        fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
}

/* Sparse configuration (14% of the particles in groups: the new-partmanager PIG branch):
 * clump A 100 particles at 250, clump B 60 at 750, a 10^3 field lattice. */
#define LC_NA 100
#define LC_NB 60
#define LC_NF 1000
#define LC_NTOT (LC_NA + LC_NB + LC_NF)

static void
lc_position(MyIDType id, int stage, double out[3])
{
    const double cA[3] = {250, 250, 250}, cB[3] = {750, 750, 750};
    if(id < LC_NA) {
        if(stage >= 3) {
            /* A dispersed: no group left anywhere (Phase 2a: a 40 kpc lattice, still bound as a
             * hostless track and smaller than half the box, so its centre is unambiguous) */
            out[0] = 30. + 40. * (id % 10);
            out[1] = 30. + 40. * (id / 10);
            out[2] = 150.;
            return;
        }
        rand_in_ball(id, 11, 6.0, cA, out);
        return;
    }
    if(id < LC_NA + LC_NB) {
        if(stage >= 1) {
            const int64_t k = id - LC_NA;
            out[0] = 50. + 40. * (k % 10);
            out[1] = 50. + 40. * (k / 10);
            out[2] = 850.;
            return;
        }
        rand_in_ball(id, 12, 5.0, cB, out);
        return;
    }
    const int64_t k = id - LC_NA - LC_NB;
    out[0] = 100. * (k % 10);
    out[1] = 100. * ((k / 10) % 10);
    out[2] = 100. * (k / 100);
}

static int
cmp_id_pos(const void * a, const void * b)
{
    const struct tagrec * x = a, * y = b;
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* Expected hostless centre: the minimum-ID particle as the reference, nearest images, equal masses. */
static void
expected_hostless_centre(const struct tagrec * r, int64_t n, int64_t track, double cen[3])
{
    int64_t i;
    int d;
    double ref[3] = {0, 0, 0}, sum[3] = {0, 0, 0};
    int64_t cnt = 0;
    for(i = 0; i < n; i++) {
        if(r[i].TrackId != track || r[i].GrNr >= 0) continue;
        if(cnt == 0)
            for(d = 0; d < 3; d++) ref[d] = r[i].Pos[d];
        for(d = 0; d < 3; d++)
            sum[d] += NEAREST(r[i].Pos[d] - ref[d], BOXSIZE);
        cnt++;
    }
    for(d = 0; d < 3; d++) {
        cen[d] = ref[d] + (cnt > 0 ? sum[d] / cnt : 0);
        while(cen[d] >= BOXSIZE) cen[d] -= BOXSIZE;
        while(cen[d] < 0) cen[d] += BOXSIZE;
    }
}

/* Rewrite one i8 block of a PIG on rank 0 (corrupted-state fixtures). */
static void
rewrite_i8_block(const char * pigname, const char * block, const int64_t * vals, int64_t n)
{
    if(get_thistask() == 0) {
        BigFile bf;
        BigBlock bb;
        BigBlockPtr ptr;
        if(0 != big_file_open(&bf, pigname))
            endrun(1, "test: cannot open %s\n", pigname);
        size_t fsize[1] = {n};
        if(0 != big_file_create_block(&bf, &bb, block, "i8", 1, 1, fsize))
            endrun(1, "test: cannot recreate %s/%s: %s\n", pigname, block, big_file_get_error_message());
        BigArray array;
        size_t dims[2] = {n, 1};
        big_array_init(&array, (void *) vals, "i8", 2, dims, NULL);
        big_block_seek(&bb, &ptr, 0);
        if(0 != big_block_write(&bb, &ptr, &array))
            endrun(1, "test: cannot write %s/%s\n", pigname, block);
        big_block_close(&bb);
        big_file_close(&bf);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/* Restart from PIG_snap of dir and compare the tags and persistent record fields with the
 * in-memory state they were written from. */
static void
check_restart_roundtrip(const char * dir, int snap, const char * what)
{
    int64_t n0, n1;
    struct tagrec * r0 = gather_tags(&n0);
    const int64_t NTotal = subhalo_get_table()->NTotal;
    struct SubTrack * rec0 = fetch_all_tracks(NTotal);
    subhalo_init(snap, snap, dir, MPI_COMM_WORLD);
    CHECK(subhalo_state_snapshot() == snap, "%s: restart loaded state %d, expected %d\n", what, subhalo_state_snapshot(), snap);
    CHECK(subhalo_get_table()->NTotal == NTotal, "%s: restart NTotal %ld, expected %ld\n", what, subhalo_get_table()->NTotal, NTotal);
    struct tagrec * r1 = gather_tags(&n1);
    CHECK(n1 == n0, "%s: %ld particles after the restart, %ld before\n", what, n1, n0);
    int64_t i;
    int bad = 0;
    for(i = 0; i < n0 && i < n1; i++)
        LOCAL_CHECK(bad, r0[i].ID == r1[i].ID && r0[i].TrackId == r1[i].TrackId && r0[i].BindRank == r1[i].BindRank
                    && r0[i].BoundTrackId == r1[i].BoundTrackId,
                    "%s: particle %lu tags (%ld %d %ld) became (%ld %d %ld) through the PIG\n", what, (unsigned long) r0[i].ID,
                    r0[i].TrackId, r0[i].BindRank, r0[i].BoundTrackId, r1[i].TrackId, r1[i].BindRank, r1[i].BoundTrackId);
    CHECK(bad == 0, "%s: the restart join did not reproduce the tags\n", what);
    struct SubTrack * rec1 = fetch_all_tracks(NTotal);
    for(i = 0; i < NTotal; i++) {
        if(rec0[i].Nbound > 0) {
            /* a Subhalo row holds every persistent column but the orphan tracer state, which
             * a track that never sank keeps at its cleared value */
            const size_t persistent = offsetof(struct SubTrack, Host);
            CHECK(0 == memcmp(&rec0[i], &rec1[i], persistent),
                  "%s: record %ld differs after the restart (Nbound %d/%d SnapDeath %d/%d HostGroup %ld/%ld)\n", what, i,
                  rec0[i].Nbound, rec1[i].Nbound, rec0[i].SnapDeath, rec1[i].SnapDeath, rec0[i].HostGroup, rec1[i].HostGroup);
            continue;
        }
        /* an Orphan row: the link, history and tracer columns */
        const struct SubTrack * a = &rec0[i], * b = &rec1[i];
        int same = a->TrackId == b->TrackId && a->HostGroup == b->HostGroup && a->NestedParentTrackId == b->NestedParentTrackId
            && a->CentralTrackId == b->CentralTrackId && a->SinkTrackId == b->SinkTrackId && a->DescendantTrackId == b->DescendantTrackId
            && a->SnapBirth == b->SnapBirth && a->SnapDeath == b->SnapDeath && a->SnapSink == b->SnapSink
            && a->SnapLastIsolation == b->SnapLastIsolation && a->MostBoundID == b->MostBoundID && b->Nbound == 0
            && a->MboundPeak == b->MboundPeak && a->SnapMboundPeak == b->SnapMboundPeak && a->MboundAcc == b->MboundAcc
            && a->SnapAcc == b->SnapAcc && a->SnapLastMajorMerger == b->SnapLastMajorMerger && a->SnapTracerLost == b->SnapTracerLost;
        int d;
        for(d = 0; d < 3; d++)
            same = same && a->TracerPos[d] == b->TracerPos[d] && a->TracerVel[d] == b->TracerVel[d];
        CHECK(same, "%s: orphan record %ld differs after the restart (SnapDeath %d/%d SinkTrackId %ld/%ld)\n", what, i,
              a->SnapDeath, b->SnapDeath, a->SinkTrackId, b->SinkTrackId);
    }
    free(rec1);
    free(r1);
    free(rec0);
    free(r0);
}

static void
test_subhalo_lifecycle_pig(void ** state)
{
    common_physics_setup();
    petaio_init();
    int NTask = get_ntask(), ThisTask = get_thistask();
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloRestartFresh = 0;
    subhalo_set_testpar(sp);
    Cosmology CP = test_cosmology();
    /* velocities in units of 100 km/s (H10) */
    subhalo_set_units(3.085678e21, 1.989e43, 1e7);
    const int er = NTask >= 3 ? 1 : 0;

    /* SUBHALO_TEST_DIR moves the PIGs to a faster file system */
    const char * dir = getenv("SUBHALO_TEST_DIR") ? getenv("SUBHALO_TEST_DIR") : "test_subhalo_lc";
    rm_tree(dir);
    if(ThisTask == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);

    setup_pman(4 * LC_NTOT, BOXSIZE);
    MyIDType id;
    for(id = 0; id < LC_NTOT; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3];
        lc_position(id, 0, pos);
        add_dm_particle(id, pos);
        P[PartManager->NumPart - 1].Vel[0] = 1;
        P[PartManager->NumPart - 1].Vel[1] = -2;
        P[PartManager->NumPart - 1].Vel[2] = 0.5;
    }
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);

    int stage;
    for(stage = 0; stage < 4; stage++) {
        const double atime = 0.4 + 0.1 * stage;
        if(stage > 0) {
            int64_t i;
            for(i = 0; i < PartManager->NumPart; i++)
                lc_position(P[i].ID, stage, P[i].Pos);
        }
        if(stage == 2) {
            /* delete B's particles: its track has no particle left anywhere */
            int64_t i = 0;
            while(i < PartManager->NumPart) {
                if(P[i].ID >= LC_NA && P[i].ID < LC_NA + LC_NB) {
                    P[i] = P[PartManager->NumPart - 1];
                    PartManager->NumPart--;
                }
                else
                    i++;
            }
        }
        int64_t ngroups, nempty;
        int ret = lifecycle_call(stage, atime, &CP, er, dir, &ngroups, &nempty);
        CHECK(ret == 1, "life cycle stage %d: subhalo_find returned %d\n", stage, ret);
        check_call_stats("life cycle", 1);
        check_header_cost(dir, stage, "life cycle");
        if(er > 0)
            CHECK(nempty == 0, "life cycle stage %d: the emptied rank %d holds %ld particles\n", stage, er, nempty);
        const int expgroups[4] = {2, 1, 1, 0};
        CHECK(ngroups == expgroups[stage], "life cycle stage %d: %ld FOF groups, expected %d\n", stage, ngroups, expgroups[stage]);
        int64_t n;
        struct tagrec * r = gather_tags(&n);
        qsort(r, n, sizeof(struct tagrec), cmp_id_pos);
        struct SubTrack * rec = fetch_all_tracks(2);
        CHECK(subhalo_get_table()->NTotal == 2, "life cycle stage %d: NTotal %ld\n", stage, subhalo_get_table()->NTotal);
        const double vkms = 1e7 / (1e5 * atime);
        if(stage == 0) {
            int t;
            for(t = 0; t < 2; t++)
                CHECK(fabs(rec[t].CoMVel[0] - vkms) < 1e-3 * vkms && fabs(rec[t].CoMVel[1] + 2 * vkms) < 1e-3 * vkms
                      && fabs(rec[t].MostBoundVel[2] - 0.5 * vkms) < 1e-3 * vkms,
                      "stage 0 track %d: CoMVel (%g %g) MostBoundVel[2] %g, expected (%g %g) %g km/s\n", t,
                      rec[t].CoMVel[0], rec[t].CoMVel[1], rec[t].MostBoundVel[2], vkms, -2 * vkms, 0.5 * vkms);
        }
        if(stage == 1 || stage == 3) {
            /* the dispersed track is hostless and resolved; its tags are untouched */
            const int64_t t = (stage == 1) ? 1 : 0;
            const int64_t nexp = (stage == 1) ? LC_NB : LC_NA;
            double cen[3];
            expected_hostless_centre(r, n, t, cen);
            CHECK(rec[t].Nbound == nexp && rec[t].HostGroup == -1 && rec[t].SnapDeath == -1 && rec[t].Rank == 0,
                  "stage %d: hostless track %ld: Nbound %d (expected %ld) HostGroup %ld SnapDeath %d\n",
                  stage, t, rec[t].Nbound, nexp, rec[t].HostGroup, rec[t].SnapDeath);
            int d;
            for(d = 0; d < 3; d++)
                CHECK(fabs(rec[t].CoMPos[d] - cen[d]) < 1e-6, "stage %d: hostless centre[%d] %.10g, expected %.10g\n", stage, d, rec[t].CoMPos[d], cen[d]);
            CHECK(fabs(rec[t].CoMVel[0] - vkms) < 1e-3 * vkms, "stage %d: hostless CoMVel %g, expected %g km/s\n", stage, rec[t].CoMVel[0], vkms);
            int64_t i, ntag = 0;
            for(i = 0; i < n; i++)
                if(r[i].TrackId == t) {
                    ntag++;
                    CHECK(r[i].GrNr < 0 && r[i].BoundTrackId == t, "stage %d: particle %lu of the hostless track\n", stage, (unsigned long) r[i].ID);
                }
            CHECK(ntag == nexp, "stage %d: %ld particles tagged %ld, expected %ld\n", stage, ntag, t, nexp);
        }
        if(stage >= 2) {
            /* B's particles are gone: dead, not sunk */
            CHECK(rec[1].Nbound == 0 && rec[1].SnapDeath == 2 && rec[1].SinkTrackId == -1 && rec[1].SnapSink == -1,
                  "stage %d: track 1 Nbound %d SnapDeath %d SinkTrackId %ld (expected a non-sunk death at 2)\n",
                  stage, rec[1].Nbound, rec[1].SnapDeath, rec[1].SinkTrackId);
        }
        if(stage == 2)
            CHECK(rec[0].Nbound == LC_NA && rec[0].HostGroup >= 0, "stage 2: track 0 Nbound %d HostGroup %ld\n", rec[0].Nbound, rec[0].HostGroup);
        free(rec);
        free(r);
    }

    /* the PIG of the last stage (zero groups, an empty interior rank) restores the state */
    check_restart_roundtrip(dir, 3, "life cycle PIG_003");

    /* (v) corrupted states. A truncated payload under an intact block header: PIG_003 is not a
     * state any more and the search falls back to PIG_002. */
    if(ThisTask == 0) {
        char * f = fastpm_strdup_printf("%s/PIG_003/1/TrackId/000000", dir);
        struct stat st;
        if(0 != stat(f, &st) || 0 != truncate(f, st.st_size / 2))
            endrun(1, "test: cannot truncate %s\n", f);
        myfree(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int m = subhalo_find_state(dir, "PIG", 3, MPI_COMM_WORLD);
    CHECK(m == 2, "truncated payload: the state search returned %d, expected 2\n", m);
    /* A duplicate TrackId shared by Subhalo and Orphan (PIG_002: track 0 resolved, track 1 an
     * orphan): the shape checks pass, the identity check must reject it. */
    {
        struct SubTrackTable t2;
        double st = -1;
        subhalo_table_init(&t2, 1024 * sizeof(struct SubTrack), MPI_COMM_WORLD);
        char * pig2 = fastpm_strdup_printf("%s/PIG_002", dir);
        int rc = subhalo_read_state_checked(pig2, &t2, &st, MPI_COMM_WORLD);
        CHECK(rc == 0, "PIG_002 should be a valid state before the corruption\n");
        subhalo_table_free(&t2);
        const int64_t dup[1] = {0};
        rewrite_i8_block(pig2, "Orphan/TrackId", dup, 1);
        m = subhalo_find_state(dir, "PIG", 2, MPI_COMM_WORLD);
        CHECK(m == 2, "duplicate TrackIds: the shape probe should still pass\n");
        subhalo_table_init(&t2, 1024 * sizeof(struct SubTrack), MPI_COMM_WORLD);
        rc = subhalo_read_state_checked(pig2, &t2, &st, MPI_COMM_WORLD);
        CHECK(rc == 1, "PIG_002 with a duplicate TrackId was accepted\n");
        subhalo_table_free(&t2);
        myfree(pig2);
        /* a duplicate inside the Subhalo table (PIG_000 has tracks 0 and 1 resolved) */
        char * pig0 = fastpm_strdup_printf("%s/PIG_000", dir);
        const int64_t dup0[2] = {0, 0};
        rewrite_i8_block(pig0, "Subhalo/TrackId", dup0, 2);
        subhalo_table_init(&t2, 1024 * sizeof(struct SubTrack), MPI_COMM_WORLD);
        rc = subhalo_read_state_checked(pig0, &t2, &st, MPI_COMM_WORLD);
        CHECK(rc == 1, "PIG_000 with duplicate TrackIds was accepted\n");
        subhalo_table_free(&t2);
        myfree(pig0);
    }
    /* a payload with the wrong dtype (u4 for i8) is not a state either */
    {
        char * pig1 = fastpm_strdup_printf("%s/PIG_001", dir);
        if(ThisTask == 0) {
            BigFile bf;
            BigBlock bb;
            if(0 != big_file_open(&bf, pig1) || 0 != big_file_open_block(&bf, &bb, "Subhalo/HostGroup"))
                endrun(1, "test: cannot open PIG_001/Subhalo/HostGroup\n");
            const size_t nrow = bb.size;
            big_block_close(&bb);
            int32_t * v = calloc(nrow + 1, sizeof(int32_t));
            size_t fsize[1] = {nrow};
            BigBlockPtr ptr;
            BigArray array;
            size_t dims[2] = {nrow, 1};
            big_file_create_block(&bf, &bb, "Subhalo/HostGroup", "i4", 1, 1, fsize);
            big_array_init(&array, v, "i4", 2, dims, NULL);
            big_block_seek(&bb, &ptr, 0);
            big_block_write(&bb, &ptr, &array);
            big_block_close(&bb);
            big_file_close(&bf);
            free(v);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        m = subhalo_find_state(dir, "PIG", 1, MPI_COMM_WORLD);
        CHECK(m == 0, "wrong dtype: the state search returned %d, expected 0\n", m);
        myfree(pig1);
    }
    teardown_pman();
    rm_tree(dir);

    /* The dense configuration of test 5 (49% in groups: the PIG reuses the particle table)
     * with the same empty interior rank, written and restored. */
    rm_tree(dir);
    if(ThisTask == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);
    t5_setup_particles(4 * T5_NTOT);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    int64_t ngroups, nempty;
    int ret = lifecycle_call(0, 0.5, &CP, er, dir, &ngroups, &nempty);
    CHECK(ret == 1 && ngroups == 2, "dense configuration: ret %d, %ld groups\n", ret, ngroups);
    if(er > 0)
        CHECK(nempty == 0, "dense configuration: the emptied rank holds %ld particles\n", nempty);
    check_restart_roundtrip(dir, 0, "dense PIG_000");
    teardown_pman();
    rm_tree(dir);
    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
}

/* ------------------------------------------------------------------ test 10: many keys (tail packing) */

/* 200 clumps of 6..10 particles on a 20 x 10 grid: more keys than SUBHALO_HEAD_PER_TASK * NTask,
 * so the tail water-fill and the census run on real particles. */
#define T10_NCLUMP 200

static int
t10_len(int c)
{
    return 6 + (c % 5);
}

static int64_t
t10_first(int c)
{
    int64_t s = 0;
    int k;
    for(k = 0; k < c; k++)
        s += t10_len(k);
    return s;
}

static int
t10_label(MyIDType id, int64_t * firsts)
{
    int lo = 0, hi = T10_NCLUMP;
    if((int64_t) id >= firsts[T10_NCLUMP])
        return -1;
    while(hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if(firsts[mid] <= (int64_t) id) lo = mid;
        else hi = mid;
    }
    return lo;
}

static void
test_subhalo_many_keys(void ** state)
{
    common_physics_setup();
    int NTask = get_ntask(), ThisTask = get_thistask();
    struct SubhaloParams sp = default_subhalo_params();
    /* Phase 2a: clumps of 6..10 particles survive the unbinding with these thresholds */
    sp.SubhaloMinNumPart = 5;
    sp.SubhaloMinNumTracerPart = 5;
    sp.SubhaloNumTracerHostFinding = 5;
    subhalo_set_testpar(sp);
    Cosmology CP = test_cosmology();
    int64_t firsts[T10_NCLUMP + 1];
    int c;
    for(c = 0; c <= T10_NCLUMP; c++)
        firsts[c] = t10_first(c);
    const int64_t ntot = firsts[T10_NCLUMP];
    setup_pman(4 * ntot + 64, BOXSIZE);
    MyIDType id;
    for(id = 0; id < (MyIDType) ntot; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        c = t10_label(id, firsts);
        double cen[3] = {25. + 50. * (c % 20), 50. + 100. * (c / 20), 500.};
        double pos[3];
        rand_in_ball(id, 21, 2.0, cen, pos);
        add_dm_particle(id, pos);
    }
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    int call;
    struct tagrec * prev = NULL;
    for(call = 0; call < 2; call++) {
        int64_t ngroups, n;
        int ret = run_finder_call(call, &CP, &ngroups);
        CHECK(ret == 1 && ngroups == T10_NCLUMP, "many keys, call %d: ret %d, %ld groups\n", call, ret, ngroups);
        struct tagrec * r = gather_tags(&n);
        CHECK(n == ntot, "many keys: %ld particles\n", n);
        int64_t * grnr = malloc(sizeof(int64_t) * T10_NCLUMP), * track = malloc(sizeof(int64_t) * T10_NCLUMP);
        for(c = 0; c < T10_NCLUMP; c++)
            grnr[c] = track[c] = -2;
        int bad = 0;
        int64_t i;
        for(i = 0; i < n; i++) {
            c = t10_label(r[i].ID, firsts);
            if(grnr[c] == -2) {
                grnr[c] = r[i].GrNr;
                track[c] = r[i].TrackId;
            }
            LOCAL_CHECK(bad, r[i].GrNr == grnr[c] && r[i].GrNr >= 0 && r[i].TrackId == track[c] && r[i].TrackId >= 0,
                        "many keys: clump %d particle %lu: GrNr %ld TrackId %ld (clump %ld %ld)\n", c, (unsigned long) r[i].ID,
                        r[i].GrNr, r[i].TrackId, grnr[c], track[c]);
            if(prev)
                LOCAL_CHECK(bad, r[i].TrackId == prev[i].TrackId && r[i].BindRank == prev[i].BindRank,
                            "many keys: particle %lu changed between identical calls\n", (unsigned long) r[i].ID);
        }
        CHECK(bad == 0, "many keys: the clumps are not one group and one track each\n");
        CHECK(check_group_invariants(r, n) == 0, "many keys: group invariants violated\n");
        /* TrackIds in host (GrNr) order */
        for(c = 0; c < T10_NCLUMP; c++) {
            int64_t below = 0;
            int c2;
            for(c2 = 0; c2 < T10_NCLUMP; c2++)
                if(grnr[c2] < grnr[c])
                    below++;
            CHECK(track[c] == below, "many keys: clump %d (GrNr %ld) has TrackId %ld, expected %ld\n", c, grnr[c], track[c], below);
        }
        free(track);
        free(grnr);
        free(prev);
        prev = r;
    }
    free(prev);
    teardown_pman();
}

/* ------------------------------------------------------------------ tests 11-13: Phase 2a (plan section 5, tests 5-7) */

/* Two Hernquist clumps in equilibrium (sampled from the isotropic distribution function as in
 * tests/test_subhalo_unbind.c, with this file's deterministic per-ID random numbers instead of
 * gsl), truncated at 4 scale radii, plus a field lattice. A: 2000 particles, M = 1e12 Msun/h,
 * scale 5 kpc/h (physical); B: 600, M = 2e11, scale 1.5 kpc/h; every seventh clump particle is gas (a non-tracer); the first P2_HOT
 * particles of each clump sit in its core at three times the escape speed (an unbound tail).
 * Positions are physical offsets divided by the call's scale factor (comoving), velocities
 * Vel = a v_pec. */
#define P2_NA 2000
#define P2_NB 600
#define P2_NF 4096
#define P2_NTOT (P2_NA + P2_NB + P2_NF)
#define P2_HOT 30
#define P2_MA 100.
#define P2_SA 5.
#define P2_MB 20.
#define P2_SB 1.5
static const double P2_cA[3] = {375., 375., 375.};
static const double P2_cB[3] = {750., 750., 750.};

struct p2rng {
    uint64_t s;
};

static double
p2_uniform(struct p2rng * r)
{
    r->s = splitmix64(r->s);
    return (r->s >> 11) * (1.0 / 9007199254740992.0);
}

/* Hernquist DF in units G = M = a = 1 up to a constant, q2 = -E in (0, 1). */
static double
p2_hq_df(double q2)
{
    if(q2 <= 0)
        return 0;
    if(q2 >= 1)
        q2 = 1 - 1e-15;
    const double q = sqrt(q2), omq = 1 - q2;
    const double br = 3 * asin(q) + q * sqrt(omq) * (1 - 2 * q2) * (8 * q2 * q2 - 8 * q2 - 3);
    return br > 0 ? pow(omq, -2.5) * br : 0;
}

static double
p2_hq_speed(double r, struct p2rng * rng)
{
    const double psi = 1. / (1. + r);
    const double vesc = sqrt(2 * psi);
    double gmax = 0;
    int k;
    for(k = 0; k <= 600; k++) {
        const double v2 = vesc * vesc * pow(10., -12. + 12. * k / 600.);
        const double g = v2 * p2_hq_df(psi - 0.5 * v2);
        if(g > gmax) gmax = g;
    }
    gmax *= 1.5;
    for(;;) {
        const double v = vesc * p2_uniform(rng);
        if(p2_uniform(rng) * gmax < v * v * p2_hq_df(psi - 0.5 * v * v))
            return v;
    }
}

static void
p2_direction(struct p2rng * rng, double u[3])
{
    const double ct = 2 * p2_uniform(rng) - 1, st = sqrt(1 - ct * ct);
    const double ph = 2 * M_PI * p2_uniform(rng);
    u[0] = st * cos(ph);
    u[1] = st * sin(ph);
    u[2] = ct;
}

/* 0: clump A, 1: clump B, -1: field */
static int
p2_clump(MyIDType id)
{
    if(id < P2_NA) return 0;
    if(id < P2_NA + P2_NB) return 1;
    return -1;
}

static int
p2_type(MyIDType id)
{
    return (p2_clump(id) >= 0 && id % 7 == 3) ? 0 : 1;
}

/* Physical offset, peculiar velocity and mass of a clump particle relative to its clump. */
static void
p2_clump_particle(MyIDType id, double G, double x[3], double v[3], double * mass)
{
    const int c = p2_clump(id);
    const double M = c == 0 ? P2_MA : P2_MB, a = c == 0 ? P2_SA : P2_SB;
    const int64_t N = c == 0 ? P2_NA : P2_NB;
    const int64_t k = c == 0 ? (int64_t) id : (int64_t) id - P2_NA;
    const double rmax = 4 * a, Xmax = pow(rmax / (rmax + a), 2);
    struct p2rng rng = {splitmix64(id * 0x100000001B3ULL + 77)};
    double ux[3], uv[3];
    int d;
    p2_direction(&rng, ux);
    p2_direction(&rng, uv);
    *mass = M * Xmax / N;
    if(k < P2_HOT) {
        /* hot core particle: three times the central escape speed */
        const double r = 0.5 * a * p2_uniform(&rng);
        for(d = 0; d < 3; d++) {
            x[d] = r * ux[d];
            v[d] = 3 * sqrt(2 * G * M / a) * uv[d];
        }
        return;
    }
    /* the cusp is cut at a few hundredths of the scale radius: the host's FOF tree (two
     * threads) runs out of nodes on near-coincident particles */
    const double u = 0.003 + 0.997 * p2_uniform(&rng);
    const double s = sqrt(Xmax * u);
    const double r = s / (1 - s);
    const double sp = p2_hq_speed(r, &rng) * sqrt(G * M / a);
    for(d = 0; d < 3; d++) {
        x[d] = a * r * ux[d];
        v[d] = sp * uv[d];
    }
}

static double
p2_wrap(double x)
{
    while(x >= BOXSIZE) x -= BOXSIZE;
    while(x < 0) x += BOXSIZE;
    return x;
}

/* Place every local particle for a call at scale factor atime: clump c centred at cen[c]
 * (comoving) with bulk peculiar velocity bulk[c]; the field on a 62.5 kpc/h lattice at rest, off the domain cell boundaries (enough
 * field particles on every rank for the host's threaded FOF tree). */
static void
p2_place(double atime, double G, const double cen[2][3], const double bulk[2][3])
{
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        struct particle_data * p = &P[i];
        const int c = p2_clump(p->ID);
        int d;
        if(c < 0) {
            const int64_t k = (int64_t) p->ID - P2_NA - P2_NB;
            p->Pos[0] = 31.25 + 62.5 * (k % 16);
            p->Pos[1] = 31.25 + 62.5 * ((k / 16) % 16);
            p->Pos[2] = 31.25 + 62.5 * (k / 256);
            p->Vel[0] = p->Vel[1] = p->Vel[2] = 0;
            continue;
        }
        double x[3], v[3], m;
        p2_clump_particle(p->ID, G, x, v, &m);
        p->Mass = m;
        for(d = 0; d < 3; d++) {
            p->Pos[d] = p2_wrap(cen[c][d] + x[d] / atime);
            p->Vel[d] = atime * (bulk[c][d] + v[d]);
        }
    }
}

static struct gravshort_tree_params P2_SavedTree;

/* Physics, parameters (defaults; softening 0.3 kpc/h) and the particles with their gas slots. */
static void
p2_setup(Cosmology * CP)
{
    common_physics_setup();
    petaio_init();
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    subhalo_set_testpar(sp);
    *CP = test_cosmology();
    P2_SavedTree = get_gravshort_treepar();
    struct gravshort_tree_params tp = P2_SavedTree;
    tp.FractionalGravitySoftening = 0.006;
    set_gravshort_treepar(tp);
    gravshort_set_softenings(50.);
    int NTask = get_ntask(), ThisTask = get_thistask();
    setup_pman(4 * P2_NTOT, BOXSIZE);
    /* room for every gas particle on one rank (a whole key may land anywhere) */
    int64_t atleast[6] = {P2_NTOT, 0, 0, 0, 128, 128};
    slots_reserve(1, atleast, SlotsManager);
    MyIDType id;
    int64_t ngas = 0;
    for(id = 0; id < P2_NTOT; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3] = {0, 0, 0};
        add_dm_particle(id, pos);
        struct particle_data * p = &P[PartManager->NumPart - 1];
        p->Type = p2_type(id);
        if(p->Type == 0)
            p->PI = ngas++;
    }
    SlotsManager->info[0].size = ngas;
    slots_setup_id(PartManager, SlotsManager);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
}

static void
p2_teardown(void)
{
    set_gravshort_treepar(P2_SavedTree);
    gravshort_set_softenings(50.);
    teardown_pman();
}

/* The D10 invariants on the gathered particles against the records: own-tag BindRank dense with
 * the exclusive bound rows first; BindRank is the energy order of all own-tag particles (J12),
 * so the reservation shows only as: the first min(10, own-tag tracers) tracer-type (type 1)
 * particles in rank order are bound to the track, and MostBoundID is the first of them;
 * BoundTrackId the tag, an ancestor of it or -1, bound owners resolved; Nbound, NboundTracer,
 * NboundExclusive, NtailType from the particles. Records processed in place
 * (SUBHALO_FLAG_INPLACE, Phase 2b) obey the same rules. Returns the number of violations. */
static int
p2_check_tags(const struct tagrec * r, int64_t n, const struct SubTrack * rec, int64_t nt, const char * what)
{
    int bad = 0;
    int64_t t, i;
    int64_t * order = malloc((n + 1) * sizeof(int64_t));
    for(t = 0; t < nt; t++) {
        int64_t nown = 0, nb = 0, nbt = 0, nex = 0, ntl = 0;
        for(i = 0; i < n; i++) {
            if(r[i].BoundTrackId == t) {
                nb++;
                if(r[i].Type == 1) nbt++;
            }
            if(r[i].TrackId == t) {
                if(r[i].BindRank < 0 || r[i].BindRank >= n) {
                    LOCAL_CHECK(bad, 0, "%s: track %ld particle %lu BindRank %d\n", what, t, (unsigned long) r[i].ID, r[i].BindRank);
                    continue;
                }
                order[nown++] = i;
                if(r[i].BoundTrackId == t) nex++;
                else ntl++;
            }
        }
        if(rec[t].Nbound == 0) {
            LOCAL_CHECK(bad, nown == 0 && nb == 0, "%s: orphan %ld still has %ld own-tag and %ld bound rows\n", what, t, nown, nb);
            continue;
        }
        LOCAL_CHECK(bad, nb == rec[t].Nbound && nbt == rec[t].NboundTracer, "%s: track %ld: %ld bound rows (%ld tracers), Nbound %d NboundTracer %d\n",
                    what, t, nb, nbt, rec[t].Nbound, rec[t].NboundTracer);
        LOCAL_CHECK(bad, nex == rec[t].NboundExclusive, "%s: track %ld: %ld exclusive rows, NboundExclusive %d\n", what, t, nex, rec[t].NboundExclusive);
        int64_t tl = 0;
        int k;
        for(k = 0; k < 6; k++)
            tl += rec[t].NtailType[k];
        LOCAL_CHECK(bad, ntl == tl, "%s: track %ld: %ld tail rows, NtailType sum %ld\n", what, t, ntl, tl);
        /* dense ranks: place by BindRank */
        int64_t * byrank = malloc((nown + 1) * sizeof(int64_t));
        for(i = 0; i < nown; i++)
            byrank[i] = -1;
        for(i = 0; i < nown; i++) {
            const int32_t b = r[order[i]].BindRank;
            if(b < nown && byrank[b] < 0) byrank[b] = order[i];
            else LOCAL_CHECK(bad, 0, "%s: track %ld: BindRank %d duplicated or beyond %ld\n", what, t, b, nown);
        }
        int64_t ntr = 0, first_tracer = -1;
        for(i = 0; i < nown; i++) {
            if(byrank[i] < 0) continue;
            const struct tagrec * q = &r[byrank[i]];
            LOCAL_CHECK(bad, (q->BoundTrackId == t) == (i < nex), "%s: track %ld: BindRank %ld bound %d, exclusive prefix %ld\n",
                        what, t, i, q->BoundTrackId == t, nex);
            if(q->Type == 1) {
                if(ntr == 0)
                    first_tracer = byrank[i];
                if(ntr < 10)
                    LOCAL_CHECK(bad, q->BoundTrackId == t, "%s: track %ld: tracer %ld in rank order (BindRank %ld) is not bound to it\n",
                                what, t, ntr, i);
                ntr++;
            }
        }
        LOCAL_CHECK(bad, first_tracer >= 0 && r[first_tracer].ID == rec[t].MostBoundID,
                    "%s: track %ld: MostBoundID %lu is not the first tracer in rank order\n", what, t, (unsigned long) rec[t].MostBoundID);
        free(byrank);
    }
    for(i = 0; i < n; i++) {
        const int64_t T = r[i].TrackId, B = r[i].BoundTrackId;
        if(T < 0) {
            LOCAL_CHECK(bad, B == -1, "%s: untagged particle %lu bound to %ld\n", what, (unsigned long) r[i].ID, B);
            continue;
        }
        if(B < 0 || B == T)
            continue;
        LOCAL_CHECK(bad, B < nt && rec[B].Nbound > 0, "%s: particle %lu bound to unresolved %ld\n", what, (unsigned long) r[i].ID, B);
        int64_t u = T;
        int steps = 0;
        while(u >= 0 && u != B && steps++ <= nt)
            u = rec[u].NestedParentTrackId;
        LOCAL_CHECK(bad, u == B, "%s: particle %lu tagged %ld bound to %ld, not an ancestor\n", what, (unsigned long) r[i].ID, T, B);
    }
    free(order);
    return bad;
}

/* A checksum of the tags and the main record columns, logged so that runs with different thread
 * counts can be compared. */
static void
p2_log_checksum(const char * what, const struct tagrec * r, int64_t n, const struct SubTrack * rec, int64_t nt)
{
    uint64_t h = 0;
    int64_t i;
    for(i = 0; i < n; i++)
        h = splitmix64(h ^ (uint64_t) r[i].ID ^ ((uint64_t) r[i].TrackId << 20) ^ ((uint64_t) r[i].BindRank << 40) ^ ((uint64_t) r[i].BoundTrackId << 8));
    for(i = 0; i < nt; i++) {
        uint64_t m;
        double x = rec[i].Mbound + 1e-3 * rec[i].Vmax + rec[i].CorePos[0];
        memcpy(&m, &x, sizeof(m));
        h = splitmix64(h ^ m ^ (uint64_t) rec[i].Nbound ^ ((uint64_t) rec[i].Rank << 32));
    }
    message(0, "Phase 2a test checksum %s: %016lx\n", what, (unsigned long) h);
}

/* A record of a resolved central hosted by its own group, with HBT properties set. */
static void
p2_check_central(const struct SubTrack * rec, int64_t t, int64_t grnr, int64_t glen, const char * what)
{
    CHECK(rec->TrackId == t && rec->HostGroup == grnr && rec->Rank == 0 && rec->Depth == 0 && rec->NestedParentTrackId == -1
          && rec->CentralTrackId == t && rec->SnapDeath == -1 && rec->DescendantTrackId == t,
          "%s: track %ld: HostGroup %ld (group %ld) Rank %d Depth %d pid %ld upid %ld SnapDeath %d\n", what, t, rec->HostGroup, grnr,
          rec->Rank, rec->Depth, rec->NestedParentTrackId, rec->CentralTrackId, rec->SnapDeath);
    CHECK(rec->Nbound > 0.8 * glen && rec->Nbound <= glen, "%s: track %ld: Nbound %d for a group of %ld\n", what, t, rec->Nbound, glen);
    CHECK(rec->Vmax > 0 && rec->Rmax > 0 && rec->CoreSigmaX > 0 && rec->CoreSigmaV > 0 && rec->Ekin > 0 && rec->Epot < 0
          && rec->KinToPot > 0.2 && rec->KinToPot < 1.5,
          "%s: track %ld: Vmax %g Rmax %g SigmaX %g SigmaV %g Ekin %g Epot %g KinToPot %g\n", what, t, rec->Vmax, rec->Rmax,
          rec->CoreSigmaX, rec->CoreSigmaV, rec->Ekin, rec->Epot, rec->KinToPot);
}

static int64_t
p2_group_of(const struct tagrec * r, int64_t n, int clump, int64_t * len)
{
    /* the group holding the clump's most common GrNr; its length */
    int64_t i, g = -1;
    for(i = 0; i < n && g < 0; i++)
        if(p2_clump(r[i].ID) == clump && r[i].GrNr >= 0 && (int64_t) r[i].ID == (clump == 0 ? 0 : P2_NA))
            g = r[i].GrNr;
    *len = 0;
    for(i = 0; i < n; i++)
        if(r[i].GrNr == g)
            (*len)++;
    return g;
}

/* Test 5 of the plan: two persistent tracks over three calls. */
static void
test_subhalo_twohalo(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int call;
    /* a cold gas particle at rest at A's centre: the most bound particle of A, but not a tracer */
    const MyIDType coldgas = 31;
    /* decision 47.5: the field SO stage's walltime children /Subhalo/FieldSO/{Setup,Tree,Walks}
     * cover the stage (this rank; new clocks would inherit the times of earlier tests' slots:
     * cleared here) */
    {
        int i;
        for(i = 0; i < CT.Nmax; i++)
            CT.C[i].time = CT.AC[i].time = 0;
    }
    for(call = 0; call < 3; call++) {
        p2_place(0.5 + 0.05 * call, CP.GravInternal, cen, bulk);
        {
            int64_t i;
            int d;
            for(i = 0; i < PartManager->NumPart; i++)
                if(P[i].ID == coldgas)
                    for(d = 0; d < 3; d++) {
                        P[i].Pos[d] = cen[0][d] + 0.01;
                        P[i].Vel[d] = 0;
                    }
        }
        int64_t ng, n;
        const double fso0 = clock_sum("/Subhalo/FieldSO/"), ftr0 = clock_sum("/Subhalo/FieldSO/Tree");
        int ret = run_finder_call(call, &CP, &ng);
        CHECK(ret == 1 && ng == 2, "two haloes, call %d: ret %d, %ld groups\n", call, ret, ng);
        check_call_stats("two haloes", 1);
        {
            const struct SubhaloCallStats * st = subhalo_last_call_stats();
            const double fso = clock_sum("/Subhalo/FieldSO/") - fso0, ftr = clock_sum("/Subhalo/FieldSO/Tree") - ftr0;
            /* mean over the ranks against the mean stage time: a rank's own stage time is not exposed */
            int NTask;
            double fsomean = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &NTask);
            MPI_Allreduce(&fso, &fsomean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            fsomean /= NTask;
            const double smean = st->StageMean[SUBHALO_STAGE_FIELDSO];
            message(0, "two haloes, call %d: walltime /Subhalo/FieldSO children %.5f s (tree build %.5f s; mean over the ranks %.5f s), "
                       "the summary's field SO %.5f s (max), %.5f s (mean)\n", call, fso, ftr, fsomean, st->StageMax[SUBHALO_STAGE_FIELDSO], smean);
            CHECK(ftr > 0, "two haloes, call %d: no time on /Subhalo/FieldSO/Tree (%g s)\n", call, ftr);
            CHECK(fsomean <= smean * (1 + 1e-6) + 1e-4 && fsomean >= 0.8 * smean - 1e-4,
                  "two haloes, call %d: walltime field SO %g s (mean %g s, tree %g s), summary mean %g s\n", call, fso, fsomean, ftr, smean);
        }
        {
            const struct SubhaloCallStats * st = subhalo_last_call_stats();
            CHECK(st->NSerialKeys == 2 && st->LargestSerial > P2_NB && st->NCands == 0 && st->FieldSOCentrals == 2
                  && st->FieldSOPass1Walks > 0 && st->FieldSOBins > 0,
                  "two haloes, call %d: stats %ld keys (largest %ld), %ld candidates, field SO %ld centrals %d walks %d bins\n", call,
                  st->NSerialKeys, st->LargestSerial, st->NCands, st->FieldSOCentrals, st->FieldSOPass1Walks, st->FieldSOBins);
        }
        struct tagrec * r = gather_tags(&n);
        CHECK(subhalo_get_table()->NTotal == 2, "two haloes, call %d: NTotal %ld\n", call, subhalo_get_table()->NTotal);
        struct SubTrack * rec = fetch_all_tracks(2);
        int64_t lenA, lenB;
        const int64_t gA = p2_group_of(r, n, 0, &lenA), gB = p2_group_of(r, n, 1, &lenB);
        p2_check_central(&rec[0], 0, gA, lenA, "two haloes");
        p2_check_central(&rec[1], 1, gB, lenB, "two haloes");
        /* the hot core particles are each track's unbound tail */
        int64_t i, ntail[2] = {0, 0};
        int bad = 0;
        for(i = 0; i < n; i++) {
            const int c = p2_clump(r[i].ID);
            if(c < 0) {
                LOCAL_CHECK(bad, r[i].TrackId == -1, "two haloes: field particle %lu tagged %ld\n", (unsigned long) r[i].ID, r[i].TrackId);
                continue;
            }
            const int64_t k = c == 0 ? (int64_t) r[i].ID : (int64_t) r[i].ID - P2_NA;
            if(k < P2_HOT)
                LOCAL_CHECK(bad, r[i].TrackId == c && r[i].BoundTrackId == -1, "two haloes: hot particle %lu: TrackId %ld BoundTrackId %ld\n",
                            (unsigned long) r[i].ID, r[i].TrackId, r[i].BoundTrackId);
            if(r[i].TrackId == c && r[i].BoundTrackId == -1)
                ntail[c]++;
        }
        CHECK(bad == 0, "two haloes, call %d: particle tags\n", call);
        CHECK(ntail[0] >= P2_HOT && ntail[1] >= P2_HOT, "two haloes, call %d: tails %ld %ld\n", call, ntail[0], ntail[1]);
        /* J12: BindRank is the energy order of every own-tag particle, so the cold gas particle
         * ranks among the first ten (formerly reserved to tracers) while MostBoundID stays a tracer */
        for(i = 0; i < n; i++)
            if(r[i].ID == coldgas)
                CHECK(r[i].Type == 0 && r[i].TrackId == 0 && r[i].BoundTrackId == 0 && r[i].BindRank < 10 && rec[0].MostBoundID != coldgas,
                      "two haloes, call %d: cold gas particle type %d tagged %ld bound %ld BindRank %d (MostBoundID %lu)\n", call,
                      r[i].Type, r[i].TrackId, r[i].BoundTrackId, r[i].BindRank, (unsigned long) rec[0].MostBoundID);
        CHECK(p2_check_tags(r, n, rec, 2, "two haloes") == 0, "two haloes, call %d: tag invariants\n", call);
        char what[64];
        snprintf(what, sizeof(what), "two haloes call %d", call);
        p2_log_checksum(what, r, n, rec, 2);
        free(rec);
        free(r);
    }
    p2_teardown();
}

/* R1: one call of the default-pass sequence on the two-halo fixture; the PIG saved when dir is set. */
static int
sd_call(int snap, Cosmology * CP, const char * dir)
{
    const double atime = 0.5 + 0.05 * snap;
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {30. * snap, 0, 0}};
    p2_place(atime, CP->GravInternal, cen, bulk);
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    const int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, 0, MPI_COMM_WORLD);
    if(dir) {
        double MassTable[6] = {0};
        if(fof_save_groups(&fof, dir, "PIG", snap, CP, atime, MassTable, 0, MPI_COMM_WORLD)) {
            slots_gc_sorted(PartManager, SlotsManager);
            if(domain_maintain(&ddecomp, NULL))
                endrun(1, "test: domain exchange after the PIG write failed\n");
            slots_gc_sorted(PartManager, SlotsManager);
        }
    }
    fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
}

/* R1: non-default core size and reserved tracer count for sd_sequence (0 = defaults) */
static int SdCoreSize = 0, SdMinTracers = 0;

struct sd_state {
    struct tagrec * r[2];
    struct SubTrack * rec[2];
    int64_t n[2], nt[2];
};

/* Calls 0, 1 (PIGs saved), then call 2 in memory or after reloading PIG_001; the tags and records
 * after calls 1 and 2. T threads; finalpass the SubhaloFinalPass option. */
static void
sd_sequence(int T, int reload, int finalpass, struct sd_state * out)
{
    Cosmology CP;
    const int saved = omp_get_max_threads();
    omp_set_num_threads(T);
    p2_setup(&CP);
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloFinalPass = finalpass;
    if(SdCoreSize > 0)
        sp.SubhaloCoreSize = SdCoreSize;
    if(SdMinTracers > 0)
        sp.SubhaloMinNumTracerPart = SdMinTracers;
    subhalo_set_testpar(sp);
    const char * base = getenv("SUBHALO_TEST_DIR") ? getenv("SUBHALO_TEST_DIR") : "test_subhalo_lc";
    char * dir = fastpm_strdup_printf("%s/sd_T%d_r%d_f%d", base, T, reload, finalpass);
    rm_tree(dir);
    if(get_thistask() == 0 && mkdir(base, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", base);
    if(get_thistask() == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);
    int snap;
    for(snap = 0; snap < 3; snap++) {
        if(snap == 2 && reload) {
            subhalo_init(1, 1, dir, MPI_COMM_WORLD);
            CHECK(subhalo_state_snapshot() == 1, "sampled default: reload gave state %d\n", subhalo_state_snapshot());
        }
        const int ret = sd_call(snap, &CP, snap < 2 ? dir : NULL);
        const struct SubhaloCallStats * st = subhalo_last_call_stats();
        CHECK(ret == 1 && st->Complete && st->NSerialKeys == 2 && st->NCands == 0,
              "sampled default, %d threads, call %d: ret %d, %ld keys, %ld collective candidates\n", T, snap, ret, st->NSerialKeys, st->NCands);
        /* at 2 threads clump A (2000 particles, above SubhaloPotentialSampleSize) runs alone, threaded */
        CHECK(T == 1 || st->NAloneKeys >= 1, "sampled default, %d threads, call %d: %ld keys alone\n", T, snap, st->NAloneKeys);
        if(snap == 0)
            continue;
        const int k = snap - 1;
        out->r[k] = gather_tags(&out->n[k]);
        out->nt[k] = subhalo_get_table()->NTotal;
        out->rec[k] = fetch_all_tracks(out->nt[k]);
    }
    rm_tree(dir);
    myfree(dir);
    p2_teardown();
    omp_set_num_threads(saved);
}

/* Tags and persistent record columns of two sequences equal; returns the differences. */
static int
sd_compare(const struct sd_state * a, const struct sd_state * b, const char * what)
{
    int bad = 0, k;
    int64_t i;
    for(k = 0; k < 2; k++) {
        LOCAL_CHECK(bad, a->n[k] == b->n[k] && a->nt[k] == b->nt[k], "%s, call %d: %ld/%ld particles, %ld/%ld tracks\n", what, k + 1,
                    a->n[k], b->n[k], a->nt[k], b->nt[k]);
        if(a->n[k] != b->n[k] || a->nt[k] != b->nt[k])
            continue;
        for(i = 0; i < a->n[k]; i++) {
            const struct tagrec * x = &a->r[k][i], * y = &b->r[k][i];
            LOCAL_CHECK(bad, x->ID == y->ID && x->GrNr == y->GrNr && x->TrackId == y->TrackId && x->BoundTrackId == y->BoundTrackId
                        && x->BindRank == y->BindRank && x->Type == y->Type,
                        "%s, call %d: particle %lu: (%ld %ld %ld %d) vs (%ld %ld %ld %d)\n", what, k + 1, (unsigned long) x->ID, x->GrNr,
                        x->TrackId, x->BoundTrackId, x->BindRank, y->GrNr, y->TrackId, y->BoundTrackId, y->BindRank);
        }
        /* every persisted column: the record up to the per-call scratch (check_restart_roundtrip) */
        for(i = 0; i < a->nt[k]; i++)
            LOCAL_CHECK(bad, 0 == memcmp(&a->rec[k][i], &b->rec[k][i], offsetof(struct SubTrack, Host)),
                        "%s, call %d: record %ld differs (Nbound %d/%d Vmax %g/%g Epot %g/%g RsSO %g/%g)\n", what, k + 1, i,
                        a->rec[k][i].Nbound, b->rec[k][i].Nbound, a->rec[k][i].Vmax, b->rec[k][i].Vmax, a->rec[k][i].Epot,
                        b->rec[k][i].Epot, a->rec[k][i].RsSO, b->rec[k][i].RsSO);
    }
    return bad;
}

static void
sd_free(struct sd_state * s)
{
    int k;
    for(k = 0; k < 2; k++) {
        free(s->r[k]);
        free(s->rec[k]);
    }
}

/* R1: the real driver with the production default (SubhaloFinalPass = 0, sampled final pass) on
 * the two-halo fixture (clump A above SubhaloPotentialSampleSize): tags and every persisted record
 * column after calls 1 and 2 are the same at 1 and 2 threads (A runs alone, threaded), and with
 * call 2 continued in memory or after reloading PIG_001. The exact pass (SubhaloFinalPass = 1)
 * gives a different binding order, so the default path is the one exercised. */
static void
test_subhalo_sampled_default(void ** state)
{
    struct sd_state ref, x;
    int bad = 0;
    sd_sequence(1, 0, 0, &ref);
    const int Ts[3] = {2, 1, 2}, Rl[3] = {0, 1, 1};
    int c;
    for(c = 0; c < 3; c++) {
        char what[96];
        snprintf(what, sizeof(what), "sampled default, %d threads%s vs 1 thread", Ts[c], Rl[c] ? " with a reload" : "");
        sd_sequence(Ts[c], Rl[c], 0, &x);
        const int b = sd_compare(&ref, &x, what);
        message(0, "%s: %d differences\n", what, b);
        bad += b;
        sd_free(&x);
    }
    assert_all_true(bad == 0);
    /* the default differs from the exact pass in clump A's order (the sampled pass ran) */
    sd_sequence(1, 0, 1, &x);
    int64_t i, ndiff = 0;
    for(i = 0; i < ref.n[0] && i < x.n[0]; i++)
        ndiff += ref.r[0][i].ID == x.r[0][i].ID && ref.r[0][i].BindRank != x.r[0][i].BindRank;
    message(0, "sampled default vs exact pass, call 1: %ld of %ld BindRank differ\n", ndiff, ref.n[0]);
    CHECK(ndiff > 0, "sampled default: the binding order equals the exact pass's (%ld differences)\n", ndiff);
    sd_free(&x);
    sd_free(&ref);
    /* a non-default core (37) and reserved tracer count (45): a 225-tracer exact head, 1 vs 2 threads */
    SdCoreSize = 37;
    SdMinTracers = 45;
    sd_sequence(1, 0, 0, &ref);
    sd_sequence(2, 1, 0, &x);
    bad = sd_compare(&ref, &x, "sampled default, core 37, 2 threads with a reload vs 1 thread");
    message(0, "sampled default, core 37 / 45 reserved tracers, 2 threads with a reload vs 1 thread: %d differences\n", bad);
    SdCoreSize = SdMinTracers = 0;
    assert_all_true(bad == 0);
    sd_free(&x);
    sd_free(&ref);
}

/* Phase 5c: the reserve admission. The two haloes with A (2000 particles) in place: a first call
 * completes and gives the plan's reserve on every rank; the second call runs with the free memory
 * squeezed to 2 MB below that reserve, where the collective stage alone still fits: the call
 * fails before the exchange (reserve_failed, the collective admission passed), the tags are
 * unchanged, the statistics are those of a failed call; the third call, unsqueezed, completes. */
static void
test_subhalo_reserve_admission(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloCollectiveMinPart = 1000;
    subhalo_set_testpar(sp);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int64_t ng, n, i;
    int ret = run_finder_call(0, &CP, &ng);
    const struct SubhaloCollAdmit * ad = subhalo_plan_last_admission();
    CHECK(ret == 1 && ad->checked && !ad->failed && !ad->reserve_failed && ad->reserve_short < 0,
          "reserve: the first call returned %d (checked %d failed %d reserve failed %d short %g)\n", ret, ad->checked, ad->failed,
          ad->reserve_failed, ad->reserve_short);
    check_call_stats("reserve, first call", 1);
    CHECK(subhalo_last_call_stats()->NCands > 0, "reserve: no in-place candidate in the first call\n");
    double need = ad->reserve_need;
    MPI_Allreduce(MPI_IN_PLACE, &need, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double target = need - 2. * 1024 * 1024;

    /* the squeezed call, at a = 0.55 */
    p2_place(0.55, CP.GravInternal, cen, bulk);
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    struct tagrec * pre = gather_tags(&n);
    const double fr = (double) mymalloc_freebytes();
    void * block = fr > target ? mymalloc2("TestReserveSqueeze", (size_t) (fr - target)) : NULL;
    message(0, "reserve: the plan's reserve is %g MB; free %g MB squeezed to %g MB\n", need / 1048576., fr / 1048576.,
            (double) mymalloc_freebytes() / 1048576.);
    ret = subhalo_find(&fof, &ddecomp, 1, 0.55, &CP, 0, MPI_COMM_WORLD);
    if(block)
        myfree(block);
    fof_finish(&fof);
    domain_free(&ddecomp);
    ad = subhalo_plan_last_admission();
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    message(0, "reserve: ret %d; reserve failed %d on rank %d, short %g MB (this rank needs %g MB, had %g MB); collective checked %d "
               "failed %d (needs %g MB of %g MB); plan %ld serial, %ld in place, %ld candidates\n", ret, ad->reserve_failed,
            ad->reserve_rank, ad->reserve_short / 1048576., ad->reserve_need / 1048576., ad->pack_free / 1048576., ad->checked,
            ad->failed, ad->need / 1048576., ad->avail / 1048576., ps->NSerial, ps->NOversized, ps->NCands);
    CHECK(ret == 0 && !subhalo_last_call_complete() && !ps->Admitted, "reserve: the squeezed call returned %d\n", ret);
    CHECK(ad->reserve_failed && ad->reserve_short >= 2. * 1024 * 1024 && ad->reserve_short < need,
          "reserve: reserve failed %d, short %g bytes\n", ad->reserve_failed, ad->reserve_short);
    CHECK(ad->checked && !ad->failed && ad->need <= ad->avail, "reserve: the collective admission (checked %d) failed %d: %g of %g bytes\n",
          ad->checked, ad->failed, ad->need, ad->avail);
    check_call_stats("reserve, squeezed call", 0);
    CHECK(subhalo_last_call_stats()->StageMax[SUBHALO_STAGE_EXCHANGE] == 0 && subhalo_last_call_stats()->StageMax[SUBHALO_STAGE_PLAN] > 0,
          "reserve: the failed call timed the exchange (%g s)\n", subhalo_last_call_stats()->StageMax[SUBHALO_STAGE_EXCHANGE]);
    int64_t n2, bad = 0;
    struct tagrec * post = gather_tags(&n2);
    for(i = 0; i < n && i < n2; i++)
        if(post[i].ID != pre[i].ID || post[i].GrNr != pre[i].GrNr || post[i].TrackId != pre[i].TrackId
           || post[i].BindRank != pre[i].BindRank || post[i].BoundTrackId != pre[i].BoundTrackId)
            bad++;
    CHECK(n2 == n && bad == 0, "reserve: %ld particles changed by the failed call\n", bad);
    free(post);
    free(pre);

    /* the memory back: the call completes */
    ret = run_finder_call(1, &CP, &ng);
    ad = subhalo_plan_last_admission();
    CHECK(ret == 1 && subhalo_last_call_complete() && !ad->reserve_failed, "reserve: the next call returned %d\n", ret);
    check_call_stats("reserve, next call", 1);
    p2_teardown();
}

/* The CoM of the particles bound to track t (mass-weighted, nearest images about the first). */
static void
p2_bound_com(const struct tagrec * r, const double * mass, int64_t n, int64_t t, double com[3], int64_t * cnt)
{
    int64_t i;
    int d;
    double ref[3] = {0, 0, 0}, s[3] = {0, 0, 0}, M = 0;
    *cnt = 0;
    for(i = 0; i < n; i++) {
        if(r[i].BoundTrackId != t) continue;
        if(*cnt == 0)
            for(d = 0; d < 3; d++) ref[d] = r[i].Pos[d];
        for(d = 0; d < 3; d++)
            s[d] += mass[i] * NEAREST(r[i].Pos[d] - ref[d], BOXSIZE);
        M += mass[i];
        (*cnt)++;
    }
    for(d = 0; d < 3; d++)
        com[d] = p2_wrap(ref[d] + (M > 0 ? s[d] / M : 0));
}

/* B at 5 kpc/h (physical) from A's centre on a circular orbit, 60 of its particles pulled to
 * A's centre (the overlap scenario of plan test 6), positions and velocities for atime. */
static void
p2_infall(Cosmology * CP, double atime)
{
    double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    /* B at 5 kpc/h (physical) from A's centre on a circular orbit: its tidal radius (about 3
     * kpc/h) is inside its truncation radius (6 kpc/h), so its outskirts are stripped */
    const double dphys = 5.;
    const double MinA = P2_MA * pow(dphys / (dphys + P2_SA), 2);
    const double vc = sqrt(CP->GravInternal * MinA / dphys);
    cen[1][0] = P2_cA[0] + dphys / atime;
    cen[1][1] = P2_cA[1];
    cen[1][2] = P2_cA[2];
    bulk[1][1] = vc;
    p2_place(atime, CP->GravInternal, cen, bulk);
    /* 60 of B's particles have been pulled out towards A's centre and move against B: unbound
     * to B, bound to A (what the satellite passes up and the central binds) */
    {
        int64_t i;
        for(i = 0; i < PartManager->NumPart; i++) {
            const int64_t k = (int64_t) P[i].ID - P2_NA;
            if(p2_clump(P[i].ID) != 1 || k < P2_HOT || k >= P2_HOT + 60)
                continue;
            double x[3];
            const double c0[3] = {0, 0, 0};
            rand_in_ball(P[i].ID, 41, 1.0, c0, x);
            int d;
            for(d = 0; d < 3; d++) {
                P[i].Pos[d] = p2_wrap(P2_cA[d] + (x[d] + (d == 0 ? 1. : 0)) / atime);
                P[i].Vel[d] = atime * (d == 1 ? -vc : 0);
            }
        }
    }
}

/* Test 6 of the plan: B falls into A (one FOF group): a satellite with a stripped tail, part of
 * it bound by the central under the satellite's tag; the frames of the next call from the
 * BoundTrackId sums; the restart join keeps BoundTrackId. */
static void
test_subhalo_overlap(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    const char * dir = getenv("SUBHALO_TEST_DIR") ? getenv("SUBHALO_TEST_DIR") : "test_subhalo_p2";
    rm_tree(dir);
    if(get_thistask() == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);
    double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int64_t ng, nempty;
    int ret = lifecycle_call(0, 0.5, &CP, 0, dir, &ng, &nempty);
    CHECK(ret == 1 && ng == 2, "overlap: first call\n");
    const double atime = 0.55;
    p2_infall(&CP, atime);
    ret = lifecycle_call(1, atime, &CP, 0, dir, &ng, &nempty);
    CHECK(ret == 1 && ng == 1, "overlap: %ld groups after the infall, expected 1\n", ng);
    int64_t n, i;
    struct tagrec * r = gather_tags(&n);
    struct SubTrack * rec = fetch_all_tracks(2);
    int64_t lenA;
    const int64_t g = p2_group_of(r, n, 0, &lenA);
    CHECK(rec[0].Rank == 0 && rec[0].Depth == 0 && rec[0].HostGroup == g && rec[0].CentralTrackId == 0,
          "overlap: track 0 Rank %d Depth %d HostGroup %ld\n", rec[0].Rank, rec[0].Depth, rec[0].HostGroup);
    CHECK(rec[1].Nbound > 0 && rec[1].Rank == 1 && rec[1].Depth == 1 && rec[1].NestedParentTrackId == 0 && rec[1].CentralTrackId == 0
          && rec[1].HostGroup == g && rec[1].SnapDeath == -1,
          "overlap: track 1 Nbound %d Rank %d Depth %d pid %ld upid %ld HostGroup %ld (group %ld)\n", rec[1].Nbound, rec[1].Rank,
          rec[1].Depth, rec[1].NestedParentTrackId, rec[1].CentralTrackId, rec[1].HostGroup, g);
    int64_t ntail1 = 0, nover = 0;
    for(i = 0; i < n; i++) {
        if(r[i].TrackId == 1 && r[i].BoundTrackId != 1) ntail1++;
        if(r[i].TrackId == 1 && r[i].BoundTrackId == 0) nover++;
    }
    message(0, "Overlap: satellite Nbound %d (exclusive %d), tail %ld of which %ld bound by the central; central Nbound %d exclusive %d\n",
            rec[1].Nbound, rec[1].NboundExclusive, ntail1, nover, rec[0].Nbound, rec[0].NboundExclusive);
    CHECK(ntail1 > P2_HOT, "overlap: the satellite has no stripped tail (%ld)\n", ntail1);
    CHECK(nover >= 10, "overlap: only %ld satellite tail particles are bound by the central\n", nover);
    CHECK(rec[0].Nbound > rec[0].NboundExclusive, "overlap: central Nbound %d not above NboundExclusive %d\n", rec[0].Nbound, rec[0].NboundExclusive);
    CHECK(p2_check_tags(r, n, rec, 2, "overlap") == 0, "overlap: tag invariants\n");
    p2_log_checksum("overlap", r, n, rec, 2);

    /* the next call's frame of the central from the BoundTrackId sums, against the CoM of its
     * full bound list computed here (the particles have not moved) */
    {
        double * mass = malloc((n + 1) * sizeof(double));
        /* particle masses: the clump masses are those of p2_clump_particle */
        for(i = 0; i < n; i++) {
            double x[3], v[3], m = 1;
            if(p2_clump(r[i].ID) >= 0)
                p2_clump_particle(r[i].ID, CP.GravInternal, x, v, &m);
            mass[i] = m;
        }
        double com[3];
        int64_t cnt;
        p2_bound_com(r, mass, n, 0, com, &cnt);
        struct SubTrackTable * tab = subhalo_get_table();
        subhalo_previous_frames(tab, MPI_COMM_WORLD);
        struct SubTrack * fr = fetch_all_tracks(2);
        int d;
        for(d = 0; d < 3; d++)
            CHECK(fabs(NEAREST(fr[0].RefPos[d] - com[d], BOXSIZE)) < 1e-6 && fabs(NEAREST(fr[0].RefPos[d] - PartManager->CurrentParticleOffset[d] - rec[0].CoMPos[d], BOXSIZE)) < 1e-3,
                  "overlap: frame[%d] %.9g, bound CoM %.9g, CoMPos %.9g\n", d, fr[0].RefPos[d], com[d], rec[0].CoMPos[d]);
        CHECK(fr[0].NPrev == cnt && cnt == rec[0].Nbound, "overlap: NPrev %ld, bound rows %ld, Nbound %d\n", fr[0].NPrev, cnt, rec[0].Nbound);
        free(fr);
        free(mass);
    }
    free(rec);
    free(r);
    /* restart join on the PIG written by the call */
    check_restart_roundtrip(dir, 1, "overlap PIG_001");
    p2_teardown();
    rm_tree(dir);
}

/* The j-th tracer id of clump c counted from its end (gas ids are skipped: no reassignment by type). */
static MyIDType
cs_moved_id(int c, int j)
{
    MyIDType id = c == 0 ? P2_NA : P2_NA + P2_NB;
    int k = -1;
    while(k < j)
        if(p2_type(--id) == 1)
            k++;
    return id;
}

/* Decision 52 (HBT FeedCentrals): a central's source is its FOF group only. B falls into A as in
 * the overlap test, and 20 cold tracers of each clump, tagged at the first call, are moved into the
 * field far from the group: A's (the central's) are untagged, B's (the satellite's) keep tag 1.
 * test_subhalo_lifecycle_pig checks that a hostless pseudo-key keeps all its field particles. */
#define CS_NMOVE 20
static void
test_subhalo_central_source(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int64_t ng, n, i;
    int ret = run_finder_call(0, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "central source: first call returned %d with %ld groups\n", ret, ng);
    struct tagrec * r = gather_tags(&n);
    int c, j, bad = 0;
    for(c = 0; c < 2; c++)
        for(j = 0; j < CS_NMOVE; j++) {
            const MyIDType id = cs_moved_id(c, j);
            CHECK(id < (MyIDType) n && r[id].ID == id && r[id].TrackId == c && r[id].GrNr >= 0,
                  "central source: particle %lu of clump %d not tagged by it at the first call\n", (unsigned long) id, c);
        }
    free(r);
    p2_infall(&CP, 0.55);
    /* the moved tracers at rest on a 62.5 kpc/h line, 37 kpc/h from the field lattice and far from the group */
    for(i = 0; i < PartManager->NumPart; i++)
        for(c = 0; c < 2; c++)
            for(j = 0; j < CS_NMOVE; j++)
                if(P[i].ID == cs_moved_id(c, j)) {
                    P[i].Pos[0] = 72.5 + 62.5 * (j % 16);
                    P[i].Pos[1] = 135. + 250. * c + 62.5 * (j / 16);
                    P[i].Pos[2] = 135.;
                    P[i].Vel[0] = P[i].Vel[1] = P[i].Vel[2] = 0;
                }
    ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "central source: second call returned %d with %ld groups\n", ret, ng);
    r = gather_tags(&n);
    struct SubTrack * rec = fetch_all_tracks(2);
    CHECK(rec[0].Rank == 0 && rec[1].Rank == 1 && rec[1].NestedParentTrackId == 0 && rec[1].Nbound > 0,
          "central source: track 0 Rank %d, track 1 Rank %d parent %ld Nbound %d\n", rec[0].Rank, rec[1].Rank, rec[1].NestedParentTrackId,
          rec[1].Nbound);
    int64_t nA = 0, nB = 0;
    for(c = 0; c < 2; c++)
        for(j = 0; j < CS_NMOVE; j++) {
            const struct tagrec * q = &r[cs_moved_id(c, j)];
            LOCAL_CHECK(bad, q->GrNr == -1, "central source: moved particle %lu in group %ld\n", (unsigned long) q->ID, q->GrNr);
            if(c == 0) {
                nA += q->TrackId == -1 && q->BindRank == -1 && q->BoundTrackId == -1;
                LOCAL_CHECK(bad, q->TrackId == -1 && q->BindRank == -1 && q->BoundTrackId == -1,
                            "central source: the central's FOF-less particle %lu has (%ld %d %ld), expected untagged\n", (unsigned long) q->ID,
                            q->TrackId, q->BindRank, q->BoundTrackId);
            }
            else {
                nB += q->TrackId == 1 && q->BindRank >= 0;
                LOCAL_CHECK(bad, q->TrackId == 1 && q->BindRank >= 0,
                            "central source: the satellite's FOF-less particle %lu has (%ld %d %ld), expected tag 1\n", (unsigned long) q->ID,
                            q->TrackId, q->BindRank, q->BoundTrackId);
            }
        }
    message(0, "Central source: of %d FOF-less particles each, %ld of the central untagged, %ld of the satellite kept; Nbound %d / %d\n",
            CS_NMOVE, nA, nB, rec[0].Nbound, rec[1].Nbound);
    CHECK(bad == 0, "central source: FOF-less particles\n");
    CHECK(p2_check_tags(r, n, rec, 2, "central source") == 0, "central source: tag invariants\n");
    free(rec);
    free(r);
    p2_teardown();
}

/* Test 7 of the plan: B placed on A's core with its velocity sinks; its tracer is followed at
 * the next call; isolated with 30 field particles in a new group it is resurrected. */
static void
test_subhalo_sink_orphan(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int64_t ng, n, i;
    int ret = run_finder_call(0, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "sink: first call\n");
    /* call 1: B on A's centre at rest */
    cen[1][0] = P2_cA[0]; cen[1][1] = P2_cA[1]; cen[1][2] = P2_cA[2];
    p2_place(0.55, CP.GravInternal, cen, bulk);
    ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sink: one group expected (%ld)\n", ng);
    struct tagrec * r = gather_tags(&n);
    struct SubTrack * rec = fetch_all_tracks(2);
    CHECK(subhalo_get_table()->NTotal == 2, "sink: NTotal %ld\n", subhalo_get_table()->NTotal);
    CHECK(rec[1].Nbound == 0 && rec[1].SinkTrackId == 0 && rec[1].SnapSink == 1 && rec[1].SnapDeath == 1 && rec[1].DescendantTrackId == 0,
          "sink: track 1 Nbound %d SinkTrackId %ld SnapSink %d SnapDeath %d DescendantTrackId %ld\n", rec[1].Nbound, rec[1].SinkTrackId,
          rec[1].SnapSink, rec[1].SnapDeath, rec[1].DescendantTrackId);
    /* B has a fifth of A's mass: not a major merger at SubhaloMajorMergerRatio 0.3 */
    CHECK(rec[0].Nbound > P2_NA && rec[0].SnapLastMajorMerger == -1, "sink: central Nbound %d SnapLastMajorMerger %d\n", rec[0].Nbound, rec[0].SnapLastMajorMerger);
    CHECK(p2_check_tags(r, n, rec, 2, "sink") == 0, "sink: tag invariants\n");
    const MyIDType tracer = rec[1].MostBoundID;
    int64_t ti = -1;
    for(i = 0; i < n; i++)
        if(r[i].ID == tracer) ti = i;
    CHECK(ti >= 0 && r[ti].Type == 1, "sink: the orphan's tracer %lu is not a live DM particle\n", (unsigned long) tracer);
    int d;
    for(d = 0; d < 3; d++)
        CHECK(fabs(rec[1].TracerPos[d] - p2_wrap(r[ti].Pos[d] - PartManager->CurrentParticleOffset[d])) < 1e-3,
              "sink: TracerPos[%d] %g, tracer at %g\n", d, rec[1].TracerPos[d], r[ti].Pos[d]);
    p2_log_checksum("sink", r, n, rec, 2);
    free(rec);
    free(r);

    /* call 2: everything shifted by 5 kpc/h; the orphan follows its tracer */
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].Pos[0] = p2_wrap(P[i].Pos[0] + 5.);
    ret = run_finder_call(2, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "orphan: call 2\n");
    r = gather_tags(&n);
    rec = fetch_all_tracks(2);
    int64_t lenA;
    const int64_t g = p2_group_of(r, n, 0, &lenA);
    for(i = 0; i < n; i++)
        if(r[i].ID == tracer) ti = i;
    CHECK(rec[1].Nbound == 0 && rec[1].SnapTracerLost == -1 && rec[1].HostGroup == g, "orphan: Nbound %d SnapTracerLost %d HostGroup %ld (group %ld)\n",
          rec[1].Nbound, rec[1].SnapTracerLost, rec[1].HostGroup, g);
    for(d = 0; d < 3; d++)
        CHECK(fabs(rec[1].TracerPos[d] - p2_wrap(r[ti].Pos[d] - PartManager->CurrentParticleOffset[d])) < 1e-3,
              "orphan: TracerPos[%d] %g, tracer now at %g\n", d, rec[1].TracerPos[d], r[ti].Pos[d]);
    free(rec);
    free(r);

    /* call 3: the tracer and 30 field particles in a cold ball far away: a new group whose only
     * member is the orphan, which is resurrected as its central (no newborn) */
    const double ball[3] = {500., 100., 900.};
    for(i = 0; i < PartManager->NumPart; i++) {
        const int64_t k = (int64_t) P[i].ID - P2_NA - P2_NB;
        const int isfield = p2_clump(P[i].ID) < 0 && k < 30;
        if(P[i].ID != tracer && !isfield)
            continue;
        double x[3];
        const double c0[3] = {0, 0, 0};
        rand_in_ball(P[i].ID, 31, 1.0, c0, x);
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = ball[d] + x[d];
            P[i].Vel[d] = 0;
        }
        P[i].Mass = 0.05;
    }
    ret = run_finder_call(3, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "resurrection: call 3, %ld groups\n", ng);
    r = gather_tags(&n);
    CHECK(subhalo_get_table()->NTotal == 2, "resurrection: NTotal %ld (a newborn was created)\n", subhalo_get_table()->NTotal);
    rec = fetch_all_tracks(2);
    for(i = 0; i < n; i++)
        if(r[i].ID == tracer) ti = i;
    CHECK(rec[1].Nbound >= 20 && rec[1].Nbound <= 31 && rec[1].SnapDeath == -1 && rec[1].SinkTrackId == -1 && rec[1].Rank == 0
          && rec[1].HostGroup == r[ti].GrNr && rec[1].CentralTrackId == 1 && rec[1].SnapBirth == 0,
          "resurrection: track 1 Nbound %d SnapDeath %d SinkTrackId %ld Rank %d HostGroup %ld (tracer group %ld)\n", rec[1].Nbound,
          rec[1].SnapDeath, rec[1].SinkTrackId, rec[1].Rank, rec[1].HostGroup, r[ti].GrNr);
    CHECK(r[ti].TrackId == 1 && r[ti].BoundTrackId == 1, "resurrection: tracer tagged %ld bound %ld\n", r[ti].TrackId, r[ti].BoundTrackId);
    CHECK(rec[0].Nbound > 0 && rec[0].HostGroup != rec[1].HostGroup, "resurrection: track 0 Nbound %d HostGroup %ld\n", rec[0].Nbound, rec[0].HostGroup);
    CHECK(p2_check_tags(r, n, rec, 2, "resurrection") == 0, "resurrection: tag invariants\n");
    p2_log_checksum("resurrection", r, n, rec, 2);
    free(rec);
    free(r);
    p2_teardown();
}

/* ------------------------------------------------------------------ audit regression fixtures (J1-J12) */

/* Synthetic tracking state (halofinder_phase2a_audit_codex.md, regression items 1-5): particles
 * with chosen positions, velocities and tags, and the records of a previous call injected into
 * the table, so that one call exercises a chosen hierarchy. Every particle is a DM tracer of
 * mass SYN_M (1e6 Msun/h): a 100-particle ball of 1 kpc has v_circ ~ 20 km/s. The ids
 * [SynNTot - SYN_NFIELD, SynNTot) are a 62.5 kpc/h field lattice at rest. */
#define SYN_M 1e-4
#define SYN_NFIELD 4096
static int64_t SynNTot;

struct SynPart {
    double pos[3];      /* comoving, output frame (syn_place adds CurrentParticleOffset) */
    double vel[3];      /* internal Vel */
    double mass;
    int64_t tag, bound;
    int32_t rank;
    int swallowed;
};
typedef void (*syn_fn)(MyIDType id, double atime, struct SynPart * s, void * ud);

/* A point of a ball of physical radius Rphys about cen (comoving) with velocity bulk plus an
 * isotropic offset of at most sigma (physical km/s), as Pos and Vel at atime. */
static void
syn_ball(MyIDType id, uint64_t salt, const double cen[3], double Rphys, const double bulk[3], double sigma, double atime, struct SynPart * s)
{
    double x[3], v[3];
    const double c0[3] = {0, 0, 0};
    rand_in_ball(id, salt, Rphys, c0, x);
    rand_in_ball(id, salt + 1000, sigma, c0, v);
    int d;
    for(d = 0; d < 3; d++) {
        s->pos[d] = cen[d] + x[d] / atime;
        s->vel[d] = atime * (bulk[d] + v[d]);
    }
}

static void
syn_setup(Cosmology * CP, int64_t ntot, struct SubhaloParams sp)
{
    common_physics_setup();
    petaio_init();
    subhalo_set_testpar(sp);
    *CP = test_cosmology();
    P2_SavedTree = get_gravshort_treepar();
    struct gravshort_tree_params tp = P2_SavedTree;
    tp.FractionalGravitySoftening = 0.006;
    set_gravshort_treepar(tp);
    gravshort_set_softenings(50.);
    int NTask = get_ntask(), ThisTask = get_thistask();
    setup_pman(4 * ntot, BOXSIZE);
    MyIDType id;
    for(id = 0; id < (MyIDType) ntot; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3] = {0, 0, 0};
        add_dm_particle(id, pos);
    }
    slots_setup_id(PartManager, SlotsManager);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    SynNTot = ntot;
}

static void
syn_teardown(void)
{
    int d;
    for(d = 0; d < 3; d++)
        PartManager->CurrentParticleOffset[d] = 0;
    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    p2_teardown();
}

/* Place every local particle (and, with settags, set its tags) from fn; the lattice ids are
 * handled here. */
static void
syn_place(syn_fn fn, void * ud, double atime, int settags)
{
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        struct particle_data * p = &P[i];
        struct SynPart s;
        memset(&s, 0, sizeof(s));
        s.mass = SYN_M;
        s.tag = s.bound = -1;
        s.rank = -1;
        const int64_t k = (int64_t) p->ID - (SynNTot - SYN_NFIELD);
        if(k >= 0) {
            s.pos[0] = 31.25 + 62.5 * (k % 16);
            s.pos[1] = 31.25 + 62.5 * ((k / 16) % 16);
            s.pos[2] = 31.25 + 62.5 * (k / 256);
        }
        fn(p->ID, atime, &s, ud);
        int d;
        for(d = 0; d < 3; d++) {
            p->Pos[d] = p2_wrap(s.pos[d] + PartManager->CurrentParticleOffset[d]);
            p->Vel[d] = s.vel[d];
        }
        p->Mass = s.mass;
        p->Swallowed = s.swallowed;
        if(settags) {
            p->TrackId = s.tag;
            p->BoundTrackId = s.bound;
            p->BindRank = s.rank;
        }
    }
}

/* A resolved record of the previous call (snapshot 0). */
static struct SubTrack
syn_rec(int64_t T, int nbound, double mbound, int64_t parent, int64_t central, int rank, int depth)
{
    struct SubTrack r;
    subhalo_track_clear(&r, T);
    r.Nbound = r.NboundExclusive = r.NboundTracer = nbound;
    r.NboundType[1] = r.NboundExclusiveType[1] = nbound;
    r.Mbound = r.MboundType[1] = mbound;
    r.NestedParentTrackId = parent;
    r.CentralTrackId = central;
    r.Rank = rank;
    r.Depth = depth;
    r.HostGroup = 1;
    r.SnapBirth = 0;
    r.DescendantTrackId = T;
    r.MboundPeak = mbound;
    r.SnapMboundPeak = 0;
    return r;
}

/* An orphan record with a stale host (HostGroup 99, CentralTrackId 0) and its tracer. */
static struct SubTrack
syn_orphan(int64_t T, MyIDType tracer)
{
    struct SubTrack r;
    subhalo_track_clear(&r, T);
    r.SnapBirth = 0;
    r.SnapDeath = 0;
    r.MostBoundID = tracer;
    r.HostGroup = 99;
    r.CentralTrackId = 0;
    r.TracerPos[0] = 11;
    r.TracerPos[1] = 22;
    r.TracerPos[2] = 33;
    r.TracerVel[0] = r.TracerVel[1] = r.TracerVel[2] = 0;
    r.SnapTracerLost = -1;
    r.MboundPeak = 0.01;
    return r;
}

/* The records of a previous call, into the running finder's table (every rank passes all). */
static void
syn_records(const struct SubTrack * recs, int64_t n)
{
    struct SubTrackTable * tab = subhalo_get_table();
    tab->NTotal = n;
    subhalo_table_redistribute(tab, MPI_COMM_WORLD);
    int64_t T;
    for(T = 0; T < n; T++) {
        struct SubTrack * t = subhalo_table_get(tab, T);
        if(t)
            *t = recs[T];
    }
}

static int64_t
tagrec_index(const struct tagrec * r, int64_t n, MyIDType id)
{
    int64_t lo = 0, hi = n;
    while(lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if(r[mid].ID < id) lo = mid + 1;
        else hi = mid;
    }
    return (lo < n && r[lo].ID == id) ? lo : -1;
}

/* ---- regression item 2 and J2: a real nest tree in two groups ----
 * Group 1, the audit's chain C(0) -> X(1) -> A(2) -> D(3) along x, 5 kpc apart: C is the decided
 * central (largest previous Mbound) and moves at 1000 km/s; X (2000 particles) becomes the most
 * massive survivor and is promoted; the 50 particles p, tagged D and unbound to it, sit at rest
 * in C's frame: D, A and X keep them in their tails, C binds them. The deepest claim (D) is on
 * another branch than the binding owner C after the promotion, so p must end tagged C.
 * Group 2: R(4) -> P(5) -> G(6) with G on R's core (it sinks into its grandparent while its
 * parent survives, and R is re-unbound with G's bound set), and R -> F(7) -> E(8) with F hot (it
 * dies, E is detached to R). */
#define NT_C0 0
#define NT_X0 200
#define NT_A0 2200
#define NT_D0 2400
#define NT_P0 2500
#define NT_R0 2550
#define NT_PP0 2850
#define NT_G0 3000
#define NT_F0 3060
#define NT_E0 3100
#define NT_END 3200

static void
nest_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    if(id >= NT_END)
        return;
    const double s5 = 5. / atime;
    const double ca[3] = {375, 375, 375}, cb[3] = {750, 750, 750};
    const double zero[3] = {0, 0, 0}, vC[3] = {0, 1000, 0};
    double c[3];
    int d;
    for(d = 0; d < 3; d++)
        c[d] = ca[d];
    if(id < NT_X0) {                 /* C */
        c[0] -= 1.5 * s5;
        syn_ball(id, 1, c, 1.0, vC, 5, atime, s);
        s->tag = s->bound = 0; s->rank = id - NT_C0;
    } else if(id < NT_A0) {          /* X */
        c[0] -= 0.5 * s5;
        syn_ball(id, 2, c, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 1; s->rank = id - NT_X0;
    } else if(id < NT_D0) {          /* A */
        c[0] += 0.5 * s5;
        syn_ball(id, 3, c, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 2; s->rank = id - NT_A0;
    } else if(id < NT_P0) {          /* D */
        c[0] += 1.5 * s5;
        syn_ball(id, 4, c, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 3; s->rank = id - NT_D0;
    } else if(id < NT_R0) {          /* p: D's unbound tail, at rest in C's frame */
        c[0] -= 1.5 * s5;
        syn_ball(id, 5, c, 0.5, vC, 5, atime, s);
        s->tag = 3; s->bound = -1; s->rank = 100 + (id - NT_P0);
    } else if(id < NT_PP0) {         /* R */
        syn_ball(id, 6, cb, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 4; s->rank = id - NT_R0;
    } else if(id < NT_G0) {          /* P */
        double cp[3] = {cb[0] + s5, cb[1], cb[2]};
        syn_ball(id, 7, cp, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 5; s->rank = id - NT_PP0;
    } else if(id < NT_F0) {          /* G, on R's core */
        syn_ball(id, 8, cb, 0.5, zero, 5, atime, s);
        s->tag = s->bound = 6; s->rank = id - NT_G0;
    } else if(id < NT_E0) {          /* F, hot */
        double cf[3] = {cb[0] - s5, cb[1], cb[2]};
        syn_ball(id, 9, cf, 1.0, zero, 3000, atime, s);
        s->tag = s->bound = 7; s->rank = id - NT_F0;
    } else {                         /* E */
        double ce[3] = {cb[0] - s5, cb[1], cb[2]};
        syn_ball(id, 10, ce, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 8; s->rank = id - NT_E0;
    }
}

static void
test_subhalo_nest_tree(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    syn_setup(&CP, NT_END + SYN_NFIELD, sp);
    const double atime = 0.55;
    struct SubTrack recs[9];
    recs[0] = syn_rec(0, 250, 1.0, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 2000, 0.2, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 1, 0, 2, 2);
    recs[3] = syn_rec(3, 100, 0.01, 2, 0, 3, 3);
    recs[4] = syn_rec(4, 300, 0.03, -1, 4, 0, 0);
    recs[5] = syn_rec(5, 150, 0.015, 4, 4, 1, 1);
    recs[6] = syn_rec(6, 60, 0.006, 5, 4, 3, 2);
    recs[7] = syn_rec(7, 40, 0.004, 4, 4, 4, 1);
    recs[8] = syn_rec(8, 100, 0.01, 7, 4, 2, 2);
    syn_records(recs, 9);
    syn_place(nest_particle, NULL, atime, 1);
    int64_t ng, nempty, n, i;
    int ret = lifecycle_call(1, atime, &CP, 0, NULL, &ng, &nempty);
    CHECK(ret == 1 && ng == 2, "nest: ret %d, %ld groups (expected 2)\n", ret, ng);
    CHECK(subhalo_get_table()->NTotal == 9, "nest: NTotal %ld\n", subhalo_get_table()->NTotal);
    struct tagrec * r = gather_tags(&n);
    struct SubTrack * rec = fetch_all_tracks(9);
    /* group 1: X promoted, C under X, the chain below X unchanged */
    CHECK(rec[1].Rank == 0 && rec[1].Depth == 0 && rec[1].NestedParentTrackId == -1 && rec[1].CentralTrackId == 1,
          "nest: X Rank %d Depth %d pid %ld upid %ld\n", rec[1].Rank, rec[1].Depth, rec[1].NestedParentTrackId, rec[1].CentralTrackId);
    CHECK(rec[0].Nbound >= 240 && rec[0].NestedParentTrackId == 1 && rec[0].Depth == 1 && rec[0].CentralTrackId == 1,
          "nest: C Nbound %d pid %ld Depth %d upid %ld\n", rec[0].Nbound, rec[0].NestedParentTrackId, rec[0].Depth, rec[0].CentralTrackId);
    CHECK(rec[2].NestedParentTrackId == 1 && rec[2].Depth == 1 && rec[3].NestedParentTrackId == 2 && rec[3].Depth == 2,
          "nest: A pid %ld Depth %d, D pid %ld Depth %d\n", rec[2].NestedParentTrackId, rec[2].Depth, rec[3].NestedParentTrackId, rec[3].Depth);
    int bad = 0;
    for(i = 0; i < n; i++) {
        const MyIDType id = r[i].ID;
        if(id >= NT_P0 && id < NT_R0)
            LOCAL_CHECK(bad, r[i].TrackId == 0 && r[i].BoundTrackId == 0, "nest: tail particle %lu tagged %ld bound %ld, expected C (0)\n",
                        (unsigned long) id, r[i].TrackId, r[i].BoundTrackId);
        if(id >= NT_G0 && id < NT_F0)
            LOCAL_CHECK(bad, r[i].TrackId == 4 && r[i].BoundTrackId == 4, "nest: G's particle %lu tagged %ld bound %ld, expected R (4)\n",
                        (unsigned long) id, r[i].TrackId, r[i].BoundTrackId);
        if(id >= NT_PP0 && id < NT_G0)
            LOCAL_CHECK(bad, r[i].BoundTrackId == 5, "nest: P's particle %lu bound %ld\n", (unsigned long) id, r[i].BoundTrackId);
        if(id >= NT_E0 && id < NT_END)
            LOCAL_CHECK(bad, r[i].BoundTrackId == 8, "nest: E's particle %lu bound %ld\n", (unsigned long) id, r[i].BoundTrackId);
        if(id >= NT_F0 && id < NT_E0)
            LOCAL_CHECK(bad, r[i].BoundTrackId == -1 && r[i].TrackId != 7, "nest: F's hot particle %lu tagged %ld bound %ld\n",
                        (unsigned long) id, r[i].TrackId, r[i].BoundTrackId);
    }
    CHECK(bad == 0, "nest: per-particle binding owners\n");
    /* (up to ten of p are among C's reserved most-bound tracers, which no tail claims) */
    CHECK(subhalo_last_call_stats()->NTailRejected >= 40, "nest: %ld tail claims across the final branch rejected, expected >= 40\n",
          subhalo_last_call_stats()->NTailRejected);
    /* group 2: G sank into its grandparent R and R took its bound set; F died, E detached to R */
    CHECK(rec[4].Rank == 0 && rec[4].Nbound >= 350, "nest: R Rank %d Nbound %d\n", rec[4].Rank, rec[4].Nbound);
    CHECK(rec[5].Nbound > 0 && rec[5].NestedParentTrackId == 4, "nest: P Nbound %d pid %ld\n", rec[5].Nbound, rec[5].NestedParentTrackId);
    CHECK(rec[6].Nbound == 0 && rec[6].SinkTrackId == 4 && rec[6].SnapSink == 1 && rec[6].DescendantTrackId == 4,
          "nest: G Nbound %d SinkTrackId %ld SnapSink %d Desc %ld\n", rec[6].Nbound, rec[6].SinkTrackId, rec[6].SnapSink, rec[6].DescendantTrackId);
    CHECK(rec[7].Nbound == 0 && rec[7].SnapDeath == 1 && rec[7].SinkTrackId == -1 && rec[7].NestedParentTrackId == 4
          && rec[7].SnapLastIsolation == 0 && rec[7].HostGroup == rec[4].HostGroup && rec[7].CentralTrackId == 4,
          "nest: F Nbound %d SnapDeath %d Sink %ld pid %ld SnapLastIsolation %d HostGroup %ld upid %ld\n", rec[7].Nbound, rec[7].SnapDeath,
          rec[7].SinkTrackId, rec[7].NestedParentTrackId, rec[7].SnapLastIsolation, rec[7].HostGroup, rec[7].CentralTrackId);
    CHECK(rec[8].Nbound > 0 && rec[8].NestedParentTrackId == 4 && rec[8].Depth == 1, "nest: E Nbound %d pid %ld Depth %d\n",
          rec[8].Nbound, rec[8].NestedParentTrackId, rec[8].Depth);
    CHECK(p2_check_tags(r, n, rec, 9, "nest") == 0, "nest: tag invariants\n");
    /* the Phase 3 hooks ran: bound SO, Rockstar Vmax, field SO of the two centrals */
    CHECK(rec[1].MBoundSO[0] > 0 && rec[1].VmaxVir > 0 && rec[1].MSO[2] > 0 && rec[4].MSO[2] > 0 && rec[0].MSO[2] == -1,
          "nest: Phase 3 columns MBoundSO %g VmaxVir %g MSO200c %g / %g (satellite %g)\n", rec[1].MBoundSO[0], rec[1].VmaxVir,
          rec[1].MSO[2], rec[4].MSO[2], rec[0].MSO[2]);
    free(rec);
    free(r);
    syn_teardown();
}

/* ---- J7: deep chains, the arena high-water mark at R = 3 and R = 10, sinking on and off ----
 * A 30-node chain T0 -> T1 -> ... -> T29 (5 kpc apart, 25 bound particles each, 40 for T0) and
 * 300 hot particles in T29's tail that every node rejects: each survivor keeps up to R * 25 of
 * them, the audit's worst case for the persistent lists. */
#define DC_L 30
#define DC_N0 40
#define DC_N 25
#define DC_HOT 300
#define DC_END (DC_N0 + (DC_L - 1) * DC_N + DC_HOT)

static int
dc_node(MyIDType id)
{
    if(id < DC_N0) return 0;
    if(id < DC_N0 + (DC_L - 1) * DC_N) return 1 + (id - DC_N0) / DC_N;
    return -1;
}

static void
dc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    if(id >= DC_END)
        return;
    const double s5 = 5. / atime;
    const double zero[3] = {0, 0, 0};
    const int v = dc_node(id);
    if(v >= 0) {
        double c[3] = {375 - 14.5 * s5 + v * s5, 375, 375};
        syn_ball(id, 21, c, 0.5, zero, 2, atime, s);
        s->tag = s->bound = v;
        s->rank = v == 0 ? id : (id - DC_N0) % DC_N;
        return;
    }
    const int64_t j = id - (DC_N0 + (DC_L - 1) * DC_N);
    double c[3] = {375 - 14.5 * s5 + (j % DC_L) * s5, 375, 375};
    syn_ball(id, 22, c, 0.5, zero, 3000, atime, s);
    s->tag = DC_L - 1;
    s->bound = -1;
    s->rank = DC_N + j;
}

static void
test_subhalo_deep_chain(void ** state)
{
    const double Rs[2] = {3, 10};
    int ir, merge;
    for(ir = 0; ir < 2; ir++)
        for(merge = 0; merge < 2; merge++) {
            Cosmology CP;
            struct SubhaloParams sp = default_subhalo_params();
            sp.SubhaloSourceRelaxFactor = Rs[ir];
            sp.SubhaloMergeTrapped = merge;
            syn_setup(&CP, DC_END + SYN_NFIELD, sp);
            CHECK(subhalo_arena_bytes_per_part() == (size_t) (SUBHALO_ARENA_FIXED_BYTES_PER_PART + 8 * Rs[ir]),
                  "deep chain: arena %lu bytes per particle at R = %g\n", (unsigned long) subhalo_arena_bytes_per_part(), Rs[ir]);
            struct SubTrack recs[DC_L];
            int v;
            recs[0] = syn_rec(0, DC_N0, 1.0, -1, 0, 0, 0);
            for(v = 1; v < DC_L; v++)
                recs[v] = syn_rec(v, DC_N, 0.0025, v - 1, 0, v, v);
            syn_records(recs, DC_L);
            const double atime = 0.55;
            syn_place(dc_particle, NULL, atime, 1);
            int64_t ng, nempty, n, i;
            int ret = lifecycle_call(1, atime, &CP, 0, NULL, &ng, &nempty);
            CHECK(ret == 1 && ng == 1, "deep chain R %g merge %d: ret %d, %ld groups\n", Rs[ir], merge, ret, ng);
            const struct SubhaloCallStats * st = subhalo_last_call_stats();
            message(0, "Deep chain R = %g, sinking %d: arena high water %.3f of the key's allowance\n", Rs[ir], merge, st->ArenaPeakFrac);
            CHECK(st->ArenaPeakFrac > 0 && st->ArenaPeakFrac <= 1, "deep chain: arena high water %g\n", st->ArenaPeakFrac);
            struct tagrec * r = gather_tags(&n);
            struct SubTrack * rec = fetch_all_tracks(DC_L);
            int nalive = 0;
            for(i = 0; i < DC_L; i++)
                if(rec[i].Nbound > 0) nalive++;
            CHECK(nalive == DC_L && rec[0].Rank == 0 && rec[DC_L - 1].Depth == DC_L - 1,
                  "deep chain: %d survivors of %d, T0 Rank %d, T29 Depth %d\n", nalive, DC_L, rec[0].Rank, rec[DC_L - 1].Depth);
            int64_t ntail = 0;
            for(i = 0; i < n; i++)
                if(r[i].ID >= DC_N0 + (DC_L - 1) * DC_N && r[i].ID < DC_END && r[i].TrackId == DC_L - 1)
                    ntail++;
            CHECK(ntail == (int64_t) (Rs[ir] * DC_N) - DC_N, "deep chain: T29 keeps %ld hot particles, expected %g\n", ntail, Rs[ir] * DC_N - DC_N);
            CHECK(p2_check_tags(r, n, rec, DC_L, "deep chain") == 0, "deep chain: tag invariants\n");
            free(rec);
            free(r);
            syn_teardown();
        }
}

/* ---- regression item 1, J1 and J8: lifecycle events in one call, with an empty rank, a nonzero
 *      particle offset and a non-default velocity unit ----
 * S (0) hosts the dying D (1): D's ten tracers are six cold particles at S's centre (S binds
 * them), one cold and one hot particle in U's group (2; found remotely: bound to U, unbound) and
 * two hot ones; a newborn N in its own group numbers while those votes are pending (J1). V (3)
 * is an oversized group processed in place. Orphans: O1 (4) follows an S particle, O2 (5) a
 * swallowed field particle, O3 (6) an ID that does not exist, O4 (7) a moving field particle,
 * O5 (8) a V particle. Calls 2 and 3 move O1's tracer to the field, then into V. */
#define MX_S0 0
#define MX_D0 300
#define MX_U0 340
#define MX_V0 490
#define MX_N0 1190
#define MX_END 1290
#define MX_O2 (MX_END + 5)
#define MX_O4 (MX_END + 77)
#define MX_O3 ((MyIDType) 999999999)

struct MixUd {
    int call;
};

static void
mix_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const struct MixUd * m = (const struct MixUd *) ud;
    const double cS[3] = {750, 250, 250}, cU[3] = {250, 750, 250}, cV[3] = {750, 750, 750}, cN[3] = {250, 250, 250};
    const double zero[3] = {0, 0, 0}, vU[3] = {0, 0, 300};
    if(id == MX_O2) {
        s->swallowed = 1;
        return;
    }
    if(id == MX_O4) {
        s->vel[0] = 10 * atime;
        s->vel[1] = -20 * atime;
        s->vel[2] = 5 * atime;
        return;
    }
    if(id >= MX_END)
        return;
    if(id == 5 && m->call >= 2) {
        /* O1's tracer: into the field at call 2, into V at call 3 */
        if(m->call == 2) {
            const double far[3] = {500, 500, 875};
            syn_ball(id, 39, far, 0.1, zero, 0, atime, s);
        }
        else
            syn_ball(id, 39, cV, 0.5, zero, 5, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
        return;
    }
    if(id < MX_D0) {
        syn_ball(id, 31, cS, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 0; s->rank = id - MX_S0;
    } else if(id < MX_U0) {
        const int r = id - MX_D0;
        s->tag = s->bound = 1;
        s->rank = r;
        if(r < 6)
            syn_ball(id, 32, cS, 0.5, zero, 5, atime, s);
        else if(r == 6)
            syn_ball(id, 32, cU, 0.5, vU, 5, atime, s);
        else if(r == 7)
            syn_ball(id, 32, cU, 0.5, zero, 3000, atime, s);
        else
            syn_ball(id, 32, cS, 1.0, zero, 3000, atime, s);
    } else if(id < MX_V0) {
        syn_ball(id, 33, cU, 1.0, vU, 5, atime, s);
        s->tag = s->bound = 2; s->rank = id - MX_U0;
    } else if(id < MX_N0) {
        syn_ball(id, 34, cV, 1.0, zero, 5, atime, s);
        s->tag = s->bound = 3; s->rank = id - MX_V0;
    } else
        syn_ball(id, 35, cN, 1.0, zero, 5, atime, s);
}

static void
test_subhalo_mixed_call(void ** state)
{
    Cosmology CP;
    const int NTask = get_ntask();
    struct SubhaloParams sp = default_subhalo_params();
    /* V (700 particles) is the only key above the threshold */
    sp.SubhaloCollectiveMinPart = 500;
    /* the field SO routes the centrals of the empty rank to a rank with particles, but its full
     * tree cannot be built on a rank without particles (forcetree.c sizes the node block by
     * NumPart: "Not enough force nodes to topnode grid"), so it stays off with an empty rank */
    sp.SubhaloFieldSO = 0;
    syn_setup(&CP, MX_END + SYN_NFIELD, sp);
    const double UV = 2e5;
    subhalo_set_units(3.085678e21, 1.989e43, UV);
    const double off[3] = {3.5, -2.25, 7.0};
    int d;
    for(d = 0; d < 3; d++)
        PartManager->CurrentParticleOffset[d] = off[d];
    struct SubTrack recs[9];
    recs[0] = syn_rec(0, 300, 0.03, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 40, 0.004, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 150, 0.015, -1, 2, 0, 0);
    recs[3] = syn_rec(3, 700, 0.07, -1, 3, 0, 0);
    recs[4] = syn_orphan(4, 5);
    recs[5] = syn_orphan(5, MX_O2);
    recs[6] = syn_orphan(6, MX_O3);
    recs[7] = syn_orphan(7, MX_O4);
    recs[8] = syn_orphan(8, MX_V0 + 5);
    syn_records(recs, 9);
    const int er = NTask > 2 ? 2 : 0;
    struct MixUd ud = {1};
    int call;
    for(call = 1; call <= 3; call++) {
        const double atime = 0.5 + 0.05 * call;
        ud.call = call;
        syn_place(mix_particle, &ud, atime, call == 1);
        int64_t ng, nempty, n, i;
        int ret = lifecycle_call(call, atime, &CP, er, NULL, &ng, &nempty);
        CHECK(ret == 1 && ng == 4, "mixed call %d: ret %d, %ld groups\n", call, ret, ng);
        check_call_stats("mixed call", 1);
        CHECK(subhalo_last_call_stats()->NCands > 0 && subhalo_last_call_stats()->NRounds > 0,
              "mixed call %d: stats %ld in-place candidates\n", call, subhalo_last_call_stats()->NCands);
        CHECK(er == 0 || nempty == 0, "mixed call %d: the emptied rank holds %ld particles\n", call, nempty);
        CHECK(subhalo_get_table()->NTotal == 10, "mixed call %d: NTotal %ld\n", call, subhalo_get_table()->NTotal);
        struct tagrec * r = gather_tags(&n);
        struct SubTrack * rec = fetch_all_tracks(10);
        const int64_t gS = r[tagrec_index(r, n, MX_S0 + 1)].GrNr, gV = r[tagrec_index(r, n, MX_V0 + 1)].GrNr;
        const int64_t gN = r[tagrec_index(r, n, MX_N0 + 1)].GrNr;
        /* the newborn: final TrackId 9 on its particles and record */
        int bad = 0;
        for(i = 0; i < n; i++) {
            LOCAL_CHECK(bad, r[i].TrackId >= -1 && r[i].BoundTrackId >= -1, "mixed: particle %lu keeps a provisional tag %ld/%ld\n",
                        (unsigned long) r[i].ID, r[i].TrackId, r[i].BoundTrackId);
            if(r[i].ID >= MX_N0 && r[i].ID < MX_END)
                LOCAL_CHECK(bad, r[i].TrackId == 9 && r[i].BoundTrackId == 9, "mixed: newborn particle %lu tagged %ld bound %ld\n",
                            (unsigned long) r[i].ID, r[i].TrackId, r[i].BoundTrackId);
        }
        CHECK(bad == 0, "mixed call %d: tags\n", call);
        CHECK(rec[9].Nbound >= 95 && rec[9].HostGroup == gN && rec[9].CentralTrackId == 9 && rec[9].SnapBirth == 1,
              "mixed call %d: newborn Nbound %d HostGroup %ld (group %ld) upid %ld SnapBirth %d\n", call, rec[9].Nbound,
              rec[9].HostGroup, gN, rec[9].CentralTrackId, rec[9].SnapBirth);
        /* D died at call 1 inside S's group: six local votes for S, one remote for U, three unbound */
        CHECK(rec[1].Nbound == 0 && rec[1].SnapDeath == 1 && rec[1].DescendantTrackId == 0 && rec[1].SinkTrackId == -1
              && rec[1].SnapLastIsolation == 0,
              "mixed call %d: D Nbound %d SnapDeath %d Desc %ld Sink %ld SnapLastIsolation %d\n", call, rec[1].Nbound,
              rec[1].SnapDeath, rec[1].DescendantTrackId, rec[1].SinkTrackId, rec[1].SnapLastIsolation);
        /* V in place */
        CHECK(rec[3].Nbound > 0 && (rec[3].Flags & SUBHALO_FLAG_INPLACE) && rec[3].HostGroup == gV,
              "mixed call %d: V Nbound %d Flags %d HostGroup %ld\n", call, rec[3].Nbound, rec[3].Flags, rec[3].HostGroup);
        /* orphans (J8): current host and central, misses sticky with the last position */
        const int64_t t1 = tagrec_index(r, n, 5);
        const int64_t h1 = call == 1 ? gS : (call == 2 ? -1 : gV);
        const int64_t c1 = call == 1 ? 0 : (call == 2 ? -1 : 3);
        CHECK(rec[4].Nbound == 0 && r[t1].GrNr == h1 && rec[4].HostGroup == h1 && rec[4].CentralTrackId == c1 && rec[4].SnapTracerLost == -1,
              "mixed call %d: O1 HostGroup %ld (expected %ld, tracer in %ld) upid %ld (expected %ld) lost %d\n", call,
              rec[4].HostGroup, h1, r[t1].GrNr, rec[4].CentralTrackId, c1, rec[4].SnapTracerLost);
        int o;
        for(o = 5; o <= 6; o++)
            CHECK(rec[o].SnapTracerLost == 1 && rec[o].HostGroup == -1 && rec[o].CentralTrackId == -1 && rec[o].TracerPos[0] == 11
                  && rec[o].TracerPos[1] == 22 && rec[o].TracerPos[2] == 33,
                  "mixed call %d: lost orphan %d SnapTracerLost %d HostGroup %ld upid %ld TracerPos %g %g %g\n", call, o,
                  rec[o].SnapTracerLost, rec[o].HostGroup, rec[o].CentralTrackId, rec[o].TracerPos[0], rec[o].TracerPos[1], rec[o].TracerPos[2]);
        const int64_t t4 = tagrec_index(r, n, MX_O4);
        CHECK(rec[7].HostGroup == -1 && rec[7].CentralTrackId == -1 && rec[7].SnapTracerLost == -1,
              "mixed call %d: field orphan HostGroup %ld upid %ld lost %d\n", call, rec[7].HostGroup, rec[7].CentralTrackId, rec[7].SnapTracerLost);
        for(d = 0; d < 3; d++) {
            const double phys = p2_wrap(r[t4].Pos[d] - off[d]);
            const double v = (d == 0 ? 10 : (d == 1 ? -20 : 5)) * UV / 1e5;
            CHECK(fabs(rec[7].TracerPos[d] - phys) < 1e-9 && fabs(rec[7].TracerVel[d] - v) < 1e-4,
                  "mixed call %d: field orphan TracerPos[%d] %.12g (expected %.12g) TracerVel %g (expected %g)\n", call, d,
                  rec[7].TracerPos[d], phys, rec[7].TracerVel[d], v);
        }
        CHECK(rec[8].HostGroup == gV && rec[8].CentralTrackId == 3, "mixed call %d: O5 HostGroup %ld (V's group %ld) upid %ld\n", call,
              rec[8].HostGroup, gV, rec[8].CentralTrackId);
        CHECK(p2_check_tags(r, n, rec, 10, "mixed") == 0, "mixed call %d: tag invariants\n", call);
        free(rec);
        free(r);
    }
    syn_teardown();
}

/* ---- collision penalties in the descendant vote: call 1 of mixed_call, where the dying D has
 *      six tracers bound to S in its key and one bound to U, found remotely ----
 * Unpenalised S wins 6 to 1; with penalty 2 on the six (6 e^-2 = 0.81) the unpenalised remote
 * tracer wins; with penalty 1 on that one too (e^-1 = 0.37) S wins again. D's host stays S's group
 * (tracers 8 and 9 there are unpenalised). Every penalty is zero after the call. */
static void
test_subhalo_penalty_descendant(void ** state)
{
    const float pS[3] = {0, 2, 2}, pU[3] = {0, 0, 1};
    const int64_t expect[3] = {0, 2, 0};
    int c;
    for(c = 0; c < 3; c++) {
        Cosmology CP;
        struct SubhaloParams sp = default_subhalo_params();
        sp.SubhaloCollectiveMinPart = 500;
        sp.SubhaloFieldSO = 0;
        syn_setup(&CP, MX_END + SYN_NFIELD, sp);
        struct SubTrack recs[4];
        recs[0] = syn_rec(0, 300, 0.03, -1, 0, 0, 0);
        recs[1] = syn_rec(1, 40, 0.004, 0, 0, 1, 1);
        recs[2] = syn_rec(2, 150, 0.015, -1, 2, 0, 0);
        recs[3] = syn_rec(3, 700, 0.07, -1, 3, 0, 0);
        syn_records(recs, 4);
        struct MixUd ud = {1};
        const double atime = 0.55;
        syn_place(mix_particle, &ud, atime, 1);
        int64_t i, ng, nempty;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].ID >= MX_D0 && P[i].ID < MX_D0 + 6)
                P[i].TracerPenalty = pS[c];
            else if(P[i].ID == MX_D0 + 6)
                P[i].TracerPenalty = pU[c];
        }
        int ret = lifecycle_call(1, atime, &CP, 0, NULL, &ng, &nempty);
        CHECK(ret == 1 && ng == 4, "penalty case %d: ret %d, %ld groups\n", c, ret, ng);
        int64_t n;
        struct tagrec * r = gather_tags(&n);
        struct SubTrack * rec = fetch_all_tracks(4);
        const int64_t gS = r[tagrec_index(r, n, MX_S0 + 1)].GrNr;
        CHECK(rec[1].Nbound == 0 && rec[1].SnapDeath == 1 && rec[1].HostGroup == gS && rec[1].DescendantTrackId == expect[c],
              "penalty case %d: D Nbound %d SnapDeath %d HostGroup %ld (S's %ld) Desc %ld, expected %ld\n", c, rec[1].Nbound,
              rec[1].SnapDeath, rec[1].HostGroup, gS, rec[1].DescendantTrackId, expect[c]);
        int bad = 0;
        for(i = 0; i < PartManager->NumPart; i++)
            LOCAL_CHECK(bad, P[i].TracerPenalty == 0, "penalty case %d: particle %lu keeps TracerPenalty %g\n", c,
                        (unsigned long) P[i].ID, P[i].TracerPenalty);
        assert_all_true((bad == 0));
        free(rec);
        free(r);
        syn_teardown();
    }
}

/* ---- Phase 3 audit P2: the current central of a group in which nothing survives ----
 * K (0) held its own group at the previous call; its 100 particles are now hot (3000 km/s) and
 * K dies in its serial key with no survivor; the orphan O (1, stale host 99 and central 0)
 * follows one of K's particles into the same group. After the store and the orphan link
 * synchronisation both have HostGroup = the group and CentralTrackId = -1 (no central). */
#define NC_END 100

static void
nc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double c[3] = {500, 500, 500}, zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= NC_END)
        return;
    syn_ball(id, 81, c, 1.0, zero, 3000, atime, s);
    s->tag = s->bound = 0;
    s->rank = id;
}

static void
test_subhalo_nocentral(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    syn_setup(&CP, NC_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 100, 0.01, -1, 0, 0, 0);
    recs[1] = syn_orphan(1, 7);
    syn_records(recs, 2);
    syn_place(nc_particle, NULL, 0.55, 1);
    int64_t ng, nempty, n;
    const int ret = lifecycle_call(1, 0.55, &CP, 0, NULL, &ng, &nempty);
    CHECK(ret == 1 && ng == 1, "no central: ret %d, %ld groups\n", ret, ng);
    struct tagrec * r = gather_tags(&n);
    struct SubTrack * rec = fetch_all_tracks(2);
    const int64_t g = r[tagrec_index(r, n, 7)].GrNr;
    message(0, "no central: K Nbound %d SnapDeath %d HostGroup %ld CentralTrackId %ld; O HostGroup %ld CentralTrackId %ld (group %ld)\n",
            rec[0].Nbound, rec[0].SnapDeath, rec[0].HostGroup, rec[0].CentralTrackId, rec[1].HostGroup, rec[1].CentralTrackId, g);
    CHECK(g >= 1 && rec[0].Nbound == 0 && rec[0].SnapDeath == 1 && rec[0].HostGroup == g && rec[0].CentralTrackId == -1,
          "no central: K Nbound %d SnapDeath %d HostGroup %ld (group %ld) CentralTrackId %ld\n", rec[0].Nbound, rec[0].SnapDeath,
          rec[0].HostGroup, g, rec[0].CentralTrackId);
    CHECK(rec[1].Nbound == 0 && rec[1].HostGroup == g && rec[1].CentralTrackId == -1,
          "no central: O HostGroup %ld (group %ld) CentralTrackId %ld\n", rec[1].HostGroup, g, rec[1].CentralTrackId);
    CHECK(p2_check_tags(r, n, rec, 2, "no central") == 0, "no central: tag invariants\n");
    free(rec);
    free(r);
    syn_teardown();
}

/* ---- regression item 3 and J3: the serial/in-place boundary with populated state ----
 * Stage 0-1 serial (the overlap scenario: B a satellite of A with a tail partly bound by A);
 * stage 2 the group is oversized (SubhaloCollectiveMinPart 100), 40 of A's and 20 of B's
 * particles are in the field: Phase 2b unbinds B's sub-tree key in place (round 0) and then A
 * (round 1), B stays A's satellite, B's moved particles stay in its in-place source and A's are
 * released (decision 52);
 * stage 3 the same; stage 4 every clump particle dispersed: A and B are hostless with oversized
 * pseudo-keys, unbound in place; stage 5 serial again. Tag invariants, the restart
 * round trip of the PIG and the orphan links at every stage. SUBHALO_TEST_KEEP keeps the PIGs
 * for tools/subhalo_check.py. */
static void
fb_disperse(int stage)
{
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++) {
        struct particle_data * p = &P[i];
        const int c = p2_clump(p->ID);
        if(c < 0)
            continue;
        int64_t k = -1;
        if(stage == 4)
            k = (int64_t) p->ID;
        else if(c == 0 && p->ID % 50 == 7)
            k = 3000 + p->ID / 50;
        else if(c == 1 && (p->ID - P2_NA) % 30 == 11)
            k = 3100 + (p->ID - P2_NA) / 30;
        if(k < 0)
            continue;
        if(k >= 3000) {
            k -= 3000;
            p->Pos[0] = 100 + 40 * (k % 20);
            p->Pos[1] = 100 + 40 * (k / 20);
            p->Pos[2] = 875;
        }
        else {
            p->Pos[0] = 20 + 40 * (k % 14);
            p->Pos[1] = 20 + 40 * ((k / 14) % 14);
            p->Pos[2] = 20 + 40 * (k / 196);
        }
        p->Vel[0] = p->Vel[1] = p->Vel[2] = 0;
    }
}

static void
test_subhalo_fallback(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    const char * dir = getenv("SUBHALO_TEST_DIR") ? getenv("SUBHALO_TEST_DIR") : "test_subhalo_fb";
    rm_tree(dir);
    if(get_thistask() == 0 && mkdir(dir, 0755) != 0 && errno != EEXIST)
        endrun(1, "test: cannot create %s\n", dir);
    MPI_Barrier(MPI_COMM_WORLD);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int stage;
    for(stage = 0; stage <= 5; stage++) {
        const double atime = 0.5 + 0.05 * stage;
        struct SubhaloParams sp = default_subhalo_params();
        sp.SubhaloFieldSO = 1;
        sp.SubhaloCollectiveMinPart = (stage >= 2 && stage <= 4) ? 100 : 0;
        subhalo_set_testpar(sp);
        if(stage == 0 || stage == 5)
            p2_place(atime, CP.GravInternal, cen, bulk);
        else
            p2_infall(&CP, atime);
        if(stage >= 2 && stage <= 4)
            fb_disperse(stage == 4 ? 4 : 2);
        int64_t ng, nempty, n, i;
        int ret = lifecycle_call(stage, atime, &CP, 0, dir, &ng, &nempty);
        CHECK(ret == 1, "fallback stage %d: ret %d\n", stage, ret);
        const int64_t NT = subhalo_get_table()->NTotal;
        struct tagrec * r = gather_tags(&n);
        struct SubTrack * rec = fetch_all_tracks(NT);
        const struct SubhaloCallStats * st = subhalo_last_call_stats();
        message(0, "Fallback stage %d: %ld groups, %ld tracks; A Nbound %d Flags %d HostGroup %ld; B Nbound %d Sink %ld HostGroup %ld upid %ld; "
                   "fallback field %ld (released %ld)\n", stage, ng, NT, rec[0].Nbound, rec[0].Flags, rec[0].HostGroup, rec[1].Nbound,
                rec[1].SinkTrackId, rec[1].HostGroup, rec[1].CentralTrackId, st->NFallbackField, st->NFallbackReleased);
        char what[64];
        snprintf(what, sizeof(what), "fallback stage %d", stage);
        if(stage == 1)
            CHECK(rec[1].Nbound > 0 && rec[1].Rank == 1 && rec[0].Nbound > rec[0].NboundExclusive,
                  "fallback stage 1: no satellite with overlap (B Nbound %d Rank %d, A %d/%d)\n", rec[1].Nbound, rec[1].Rank,
                  rec[0].Nbound, rec[0].NboundExclusive);
        if(stage == 2 || stage == 3) {
            int bad = 0;
            for(i = 0; i < n; i++) {
                const int c = p2_clump(r[i].ID);
                const int moved = (c == 0 && r[i].ID % 50 == 7) || (c == 1 && (r[i].ID - P2_NA) % 30 == 11);
                /* Phase 2b: B's moved particles stay in its in-place source (its tail, bound by A or
                 * not); decision 52: A's, the central's FOF-less particles, are released untagged */
                if(moved && r[i].GrNr < 0 && c == 1)
                    LOCAL_CHECK(bad, r[i].TrackId == 1 && (r[i].BoundTrackId == -1 || r[i].BoundTrackId == 0 || r[i].BoundTrackId == 1),
                                "%s: field particle %lu of B tagged %ld bound %ld\n", what, (unsigned long) r[i].ID, r[i].TrackId, r[i].BoundTrackId);
                if(moved && r[i].GrNr < 0 && c == 0)
                    LOCAL_CHECK(bad, r[i].TrackId == -1 && r[i].BoundTrackId == -1 && r[i].BindRank == -1,
                                "%s: field particle %lu of A tagged %ld bound %ld\n", what, (unsigned long) r[i].ID, r[i].TrackId, r[i].BoundTrackId);
            }
            CHECK(bad == 0, "%s: field particles of the oversized key\n", what);
            CHECK(st->NFallbackReleased == (stage == 2 ? 40 : 0), "%s: %ld field tags released\n", what, st->NFallbackReleased);
            /* both A and B's sub-tree key are above SubhaloCollectiveMinPart: two rounds, B (depth 1)
             * then A, B stays A's satellite as in the serial stage 1 */
            CHECK((rec[0].Flags & SUBHALO_FLAG_INPLACE) && rec[0].Rank == 0 && rec[0].Nbound > rec[0].NboundExclusive
                  && rec[1].Nbound > 0 && (rec[1].Flags & SUBHALO_FLAG_INPLACE) && rec[1].Rank == 1 && rec[1].NestedParentTrackId == 0
                  && rec[1].CentralTrackId == 0,
                  "%s: A Flags %d Rank %d Nbound %d/%d, B Nbound %d Flags %d Rank %d pid %ld upid %ld\n", what, rec[0].Flags, rec[0].Rank,
                  rec[0].Nbound, rec[0].NboundExclusive, rec[1].Nbound, rec[1].Flags, rec[1].Rank, rec[1].NestedParentTrackId,
                  rec[1].CentralTrackId);
            CHECK(st->NFallbackField >= 2000, "%s: %ld particle tags written by the in-place pass\n", what, st->NFallbackField);
        }
        if(stage == 4)
            /* Phase 2b: the hostless pseudo-key is unbound in place with the HBT recipe */
            CHECK(rec[0].Nbound > 0 && rec[0].HostGroup == -1 && rec[0].Rank == 0 && (rec[0].Flags & SUBHALO_FLAG_INPLACE)
                  && rec[0].Vmax > 0,
                  "%s: hostless A Nbound %d/%d HostGroup %ld Rank %d Flags %d Vmax %g\n", what, rec[0].Nbound, rec[0].NboundExclusive,
                  rec[0].HostGroup, rec[0].Rank, rec[0].Flags, rec[0].Vmax);
        /* the orphan B: its current host from its tracer */
        if(stage >= 2 && rec[1].Nbound == 0) {
            const int64_t t = tagrec_index(r, n, rec[1].MostBoundID);
            const int64_t h = t >= 0 ? r[t].GrNr : -1;
            CHECK(rec[1].HostGroup == h && (h < 0 ? rec[1].CentralTrackId == -1 : rec[1].CentralTrackId >= 0),
                  "%s: orphan B HostGroup %ld (tracer in %ld) upid %ld\n", what, rec[1].HostGroup, h, rec[1].CentralTrackId);
            if(stage == 3 && h >= 0)
                CHECK(rec[1].CentralTrackId == 0, "%s: orphan B in the oversized key has central %ld\n", what, rec[1].CentralTrackId);
        }
        CHECK(p2_check_tags(r, n, rec, NT, what) == 0, "%s: tag invariants\n", what);
        free(rec);
        free(r);
        check_restart_roundtrip(dir, stage, what);
    }
    p2_teardown();
    if(!getenv("SUBHALO_TEST_KEEP"))
        rm_tree(dir);
}

/* ---- J11: the iteration cap and the final-pass diagnostics; J4: the orphan lookup's memory
 *      check fails the call cleanly ---- */
static void
test_subhalo_itercap(void ** state)
{
    Cosmology CP;
    p2_setup(&CP);
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloMaxUnbindIterations = 1;
    subhalo_set_testpar(sp);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int call;
    for(call = 0; call < 2; call++) {
        p2_place(0.5 + 0.05 * call, CP.GravInternal, cen, bulk);
        int64_t ng;
        int ret = run_finder_call(call, &CP, &ng);
        CHECK(ret == 1 && ng == 2, "itercap call %d: ret %d groups %ld\n", call, ret, ng);
        const struct SubhaloCallStats * st = subhalo_last_call_stats();
        struct SubTrack * rec = fetch_all_tracks(2);
        const int nflag = !!(rec[0].Flags & SUBHALO_FLAG_ITERCAP) + !!(rec[1].Flags & SUBHALO_FLAG_ITERCAP);
        message(0, "Itercap call %d: %ld unbindings, %ld capped (worst %g, track %ld), %ld of %ld final-pass E >= 0\n", call,
                st->NCandidates, st->NItercap, st->WorstConvRatio, st->WorstTrackId, st->NFinalPositive, st->NFinalBound);
        CHECK(st->NCandidates == 2 && st->NItercap == 2 && nflag == 2 && st->WorstConvRatio < sp.SubhaloBoundMassPrecision
              && st->NFinalBound == rec[0].Nbound + rec[1].Nbound && st->NFinalPositive >= 0,
              "itercap call %d: %ld unbindings, %ld capped, %d flagged records, worst %g, final bound %ld (records %d + %d)\n", call,
              st->NCandidates, st->NItercap, nflag, st->WorstConvRatio, st->NFinalBound, rec[0].Nbound, rec[1].Nbound);
        if(call == 1)
            CHECK(st->WorstTrackId == 0 || st->WorstTrackId == 1, "itercap: worst track %ld\n", st->WorstTrackId);
        else
            CHECK(SUBHALO_IS_PROVISIONAL(st->WorstTrackId), "itercap: worst track %ld at the first call (a newborn)\n", st->WorstTrackId);
        free(rec);
    }
    /* J4: a lookup batch that cannot fit: the call is skipped before anything changes */
    int64_t n0, n1, i;
    struct tagrec * r0 = gather_tags(&n0);
    sp.SubhaloLookupBatchBytes = 10000000000000L;
    subhalo_set_testpar(sp);
    int64_t ng;
    int ret = run_finder_call(2, &CP, &ng);
    CHECK(ret == 0 && !subhalo_last_call_complete() && subhalo_state_snapshot() == 1,
          "lookup memory check: ret %d complete %d state %d\n", ret, subhalo_last_call_complete(), subhalo_state_snapshot());
    struct tagrec * r1 = gather_tags(&n1);
    int bad = 0;
    for(i = 0; i < n0 && i < n1; i++)
        LOCAL_CHECK(bad, r0[i].ID == r1[i].ID && r0[i].TrackId == r1[i].TrackId && r0[i].BindRank == r1[i].BindRank
                    && r0[i].BoundTrackId == r1[i].BoundTrackId, "lookup memory check: particle %lu changed\n", (unsigned long) r0[i].ID);
    CHECK(bad == 0 && n0 == n1, "lookup memory check: tags changed\n");
    free(r1);
    free(r0);
    p2_teardown();
}

/* ---- J4: the batched ID lookup under skewed hits, repeated ids, zero-query ranks, tiny
 *      batches and the byte cap ---- */
#define LK_N0 20000
static void
lk_expect(MyIDType id, double pos[3], int64_t * bt)
{
    pos[0] = 0.001 * id;
    pos[1] = 0.5 + 0.002 * id;
    pos[2] = 7;
    *bt = (int64_t) (id % 5) - 1;
}

/* A particle with id but answers that are not lk_expect's (a later duplicate), or a garbage /
 * swallowed one (Phase 5d audit R4). */
static void
lk_add_other(MyIDType id, int garbage, int swallowed)
{
    double pos[3];
    int64_t bt;
    lk_expect(id, pos, &bt);
    pos[0] += 1;
    add_dm_particle(id, pos);
    P[PartManager->NumPart - 1].BoundTrackId = bt + 1000;
    P[PartManager->NumPart - 1].IsGarbage = garbage;
    P[PartManager->NumPart - 1].Swallowed = swallowed;
}

/* Every query that exists is answered with lk_expect's values (the lowest live P index of a
 * duplicated ID; garbage and swallowed particles never answer), the others are misses. */
static int
lk_verify(const MyIDType * q, int64_t nq, const struct SubhaloIdHit * out, int NTask, const char * what, int mode)
{
    int bad = 0;
    int64_t i;
    for(i = 0; i < nq; i++) {
        const int exists = q[i] < LK_N0 || (q[i] >= LK_N0 + 100 && q[i] < LK_N0 + 100 * NTask);
        LOCAL_CHECK(bad, out[i].found == exists, "%s mode %d: id %lu found %d\n", what, mode, (unsigned long) q[i], out[i].found);
        if(!exists || !out[i].found) continue;
        double pos[3];
        int64_t bt;
        lk_expect(q[i], pos, &bt);
        LOCAL_CHECK(bad, out[i].Pos[0] == pos[0] && out[i].Pos[1] == pos[1] && out[i].BoundTrackId == bt && out[i].GrNr == -1,
                    "%s mode %d: id %lu answered Pos %g %g bound %ld GrNr %ld\n", what, mode, (unsigned long) q[i], out[i].Pos[0],
                    out[i].Pos[1], out[i].BoundTrackId, out[i].GrNr);
    }
    return bad;
}

static void
test_subhalo_lookup(void ** state)
{
    common_physics_setup();
    const int NTask = get_ntask(), ThisTask = get_thistask();
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloLookupBatchBytes = 1024L * 1024L;
    subhalo_set_testpar(sp);
    setup_pman(LK_N0 + 1000, BOXSIZE);
    /* rank 0 holds every id below LK_N0 (skew); rank r > 0 holds 100 ids of its own */
    MyIDType id;
    if(ThisTask == 0)
        for(id = 0; id < LK_N0; id++) {
            double pos[3];
            int64_t bt;
            lk_expect(id, pos, &bt);
            add_dm_particle(id, pos);
            P[PartManager->NumPart - 1].BoundTrackId = bt;
        }
    else
        for(id = LK_N0 + 100 * ThisTask; id < LK_N0 + 100 * ThisTask + 100; id++) {
            double pos[3];
            int64_t bt;
            lk_expect(id, pos, &bt);
            add_dm_particle(id, pos);
            P[PartManager->NumPart - 1].BoundTrackId = bt;
        }
    /* Phase 5d audit R4: duplicated live IDs, answered by the lowest P index in every mode (7 on
     * rank 0, one id per other rank); ids whose lowest copies are garbage (11) or swallowed (13),
     * answered by their first live copy; ids held only by garbage (5000100) or swallowed
     * (5000101) particles, which are misses */
    if(ThisTask == 0) {
        lk_add_other(7, 0, 0);
        lk_add_other(7, 0, 0);
        P[11].IsGarbage = 1;
        P[13].Swallowed = 1;
        double pos[3];
        int64_t bt;
        for(id = 11; id <= 13; id += 2) {
            lk_expect(id, pos, &bt);
            add_dm_particle(id, pos);
            P[PartManager->NumPart - 1].BoundTrackId = bt;
            lk_add_other(id, 0, 0);
        }
        lk_add_other(5000100, 1, 0);
    }
    else {
        lk_add_other(LK_N0 + 100 * ThisTask + 5, 0, 0);
        lk_add_other(5000100, 1, 0);
        lk_add_other(5000101, 0, 1);
    }
    /* queries: rank 0 none; rank 1 15000 ids of rank 0, 300 copies of one id and 50 missing;
     * rank 2 the ids of every other rank twice; every rank > 0 thousands of rank 0's ids, so that
     * full batches concentrate their replies on rank 0 */
    int64_t nq = 0, i;
    MyIDType * q = malloc(sizeof(MyIDType) * (20000 + 200 * NTask));
    if(ThisTask == 1) {
        for(i = 0; i < 15000; i++) q[nq++] = (MyIDType) ((i * 7) % LK_N0);
        for(i = 0; i < 300; i++) q[nq++] = 42;
        for(i = 0; i < 50; i++) q[nq++] = 5000000 + i;
        for(i = 0; i < 3; i++) {
            q[nq++] = 7; q[nq++] = 11; q[nq++] = 13; q[nq++] = 5000100; q[nq++] = 5000101;
        }
    }
    else if(ThisTask == 2) {
        int rr;
        for(rr = 1; rr < NTask; rr++)
            for(i = 0; i < 200; i++) q[nq++] = LK_N0 + 100 * rr + (i % 100);
        for(i = 0; i < 5000; i++) q[nq++] = (MyIDType) ((i * 13) % LK_N0);
    }
    else if(ThisTask >= 3) {
        for(i = 0; i < 3; i++) q[nq++] = 17 + i;
        for(i = 0; i < 5000; i++) q[nq++] = (MyIDType) ((i * 11) % LK_N0);
    }
    struct SubhaloIdHit * out = malloc(sizeof(struct SubhaloIdHit) * (nq + 1));
    struct SubhaloIdHit * out0 = malloc(sizeof(struct SubhaloIdHit) * (nq + 1));
    int mode;
    int lmode;
    for(lmode = 0; lmode < 3; lmode++)
    for(mode = 0; mode < 2; mode++) {
        /* mode 0: three ids per rank and batch; mode 1: the 1 MB byte cap alone; lmode (Phase
         * 5d): how the answers are found, auto / the sorted index / the scan of the particles */
        subhalo_lookup_set_max_batch(mode == 0 ? 3 : 0);
        subhalo_lookup_set_mode(lmode);
        subhalo_lookup_ids(q, nq, out, MPI_COMM_WORLD);
        int bad = lk_verify(q, nq, out, NTask, "lookup", mode);
        /* the three answer modes and both batchings agree exactly */
        if(lmode == 0 && mode == 0)
            memcpy(out0, out, sizeof(struct SubhaloIdHit) * nq);
        else
            LOCAL_CHECK(bad, memcmp(out0, out, sizeof(struct SubhaloIdHit) * nq) == 0, "lookup mode %d / %d: answers differ from auto\n", lmode, mode);
        assert_all_true(bad == 0);
        double peak = subhalo_lookup_last_peak_bytes(), gpeak = 0;
        MPI_Allreduce(&peak, &gpeak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        message(0, "Lookup mode %d: batch high water %g bytes (bound %g, SubhaloLookupBatchBytes %ld)\n", mode, gpeak,
                subhalo_lookup_batch_peak_bytes(NTask), sp.SubhaloLookupBatchBytes);
        CHECK(gpeak <= subhalo_lookup_batch_peak_bytes(NTask), "lookup mode %d: batch high water %g above the bound %g\n", mode, gpeak,
              subhalo_lookup_batch_peak_bytes(NTask));
        if(mode == 1)
            /* with three querying ranks the batches nearly fill the cap */
            CHECK(gpeak <= sp.SubhaloLookupBatchBytes + 16384 && (NTask < 4 || gpeak > 0.5 * sp.SubhaloLookupBatchBytes),
                  "lookup: byte cap %ld, high water %g\n", sp.SubhaloLookupBatchBytes, gpeak);
    }
    /* a call so small that the auto mode differs between the ranks (audit 5d R4): rank 0's
     * ~20000 live particles answer by the scan (ntot <= nlive / 16), the ~100 of the other ranks
     * by the index; the answers equal those of the forced modes */
    {
        int64_t n2 = 0;
        if(ThisTask == 1) {
            q[n2++] = 7; q[n2++] = 11; q[n2++] = 13; q[n2++] = 5000100; q[n2++] = 5000101; q[n2++] = 42;
            if(NTask > 2) q[n2++] = LK_N0 + 100 * 2 + 5;
        }
        else if(ThisTask == 2) {
            q[n2++] = LK_N0 + 100 + 5; q[n2++] = 7; q[n2++] = 99999999;
        }
        else
            q[n2++] = 13;
        int64_t ntot = 0, nlive = 0, lo, hi;
        for(i = 0; i < PartManager->NumPart; i++)
            nlive += !P[i].IsGarbage && !P[i].Swallowed;
        MPI_Allreduce(&n2, &ntot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
        const int64_t scan = ntot <= nlive / 16;
        MPI_Allreduce(&scan, &lo, 1, MPI_INT64, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&scan, &hi, 1, MPI_INT64, MPI_MAX, MPI_COMM_WORLD);
        CHECK(NTask < 2 || (lo == 0 && hi == 1), "lookup mixed modes: %ld queries, auto scan on rank %d: %ld (ranks min %ld max %ld)\n", ntot,
              ThisTask, scan, lo, hi);
        for(lmode = 0; lmode < 3; lmode++)
        for(mode = 0; mode < 2; mode++) {
            subhalo_lookup_set_max_batch(mode == 0 ? 2 : 0);
            subhalo_lookup_set_mode(lmode);
            subhalo_lookup_ids(q, n2, out, MPI_COMM_WORLD);
            int bad = lk_verify(q, n2, out, NTask, "lookup mixed", mode);
            if(lmode == 0 && mode == 0)
                memcpy(out0, out, sizeof(struct SubhaloIdHit) * n2);
            else
                LOCAL_CHECK(bad, memcmp(out0, out, sizeof(struct SubhaloIdHit) * n2) == 0, "lookup mixed %d / %d: answers differ from auto\n", lmode,
                            mode);
            assert_all_true(bad == 0);
        }
        message(0, "Lookup mixed modes: %ld queries, auto mode scan on some ranks and index on others; the three modes agree\n", ntot);
    }
    subhalo_lookup_set_max_batch(0);
    subhalo_lookup_set_mode(0);
    free(out0);
    free(out);
    free(q);
    teardown_pman();
}

int main(void) {
    if(getenv("SUBHALO_TEST_SORTKEY_OVERFLOW")) {
        const struct CMUnitTest overflow[] = {
            cmocka_unit_test(test_subhalo_sort_key_overflow),
        };
        return cmocka_run_group_tests_mpi(overflow, NULL, NULL);
    }
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_subhalo_table),
        cmocka_unit_test(test_subhalo_sort_key),
        cmocka_unit_test(test_subhalo_exchange_tags),
        cmocka_unit_test(test_subhalo_join),
        cmocka_unit_test(test_subhalo_fof_as_subhalo),
        cmocka_unit_test(test_subhalo_forced_failure),
        cmocka_unit_test(test_subhalo_plan_pack),
        cmocka_unit_test(test_subhalo_unhosted_direct),
        cmocka_unit_test(test_subhalo_lifecycle_pig),
        cmocka_unit_test(test_subhalo_many_keys),
        cmocka_unit_test(test_subhalo_twohalo),
        cmocka_unit_test(test_subhalo_overlap),
        cmocka_unit_test(test_subhalo_central_source),
        cmocka_unit_test(test_subhalo_sink_orphan),
        cmocka_unit_test(test_subhalo_nest_tree),
        cmocka_unit_test(test_subhalo_deep_chain),
        cmocka_unit_test(test_subhalo_mixed_call),
        cmocka_unit_test(test_subhalo_nocentral),
        cmocka_unit_test(test_subhalo_fallback),
        cmocka_unit_test(test_subhalo_itercap),
        cmocka_unit_test(test_subhalo_lookup),
        cmocka_unit_test(test_subhalo_serial_keys),
        cmocka_unit_test(test_subhalo_reserve_admission),
        cmocka_unit_test(test_subhalo_local_sort),
        cmocka_unit_test(test_subhalo_header_params),
        cmocka_unit_test(test_subhalo_sampled_default),
        cmocka_unit_test(test_subhalo_weighted_plurality),
        cmocka_unit_test(test_subhalo_penalty_descendant),
    };
    /* SUBHALO_TEST_ONLY=<substring> runs only the tests whose name contains it */
    const char * only = getenv("SUBHALO_TEST_ONLY");
    if(only && only[0]) {
        struct CMUnitTest sel[sizeof(tests) / sizeof(tests[0])];
        size_t i, ns = 0;
        for(i = 0; i < sizeof(tests) / sizeof(tests[0]); i++)
            if(strstr(tests[i].name, only))
                sel[ns++] = tests[i];
        return _cmocka_run_group_tests_mpi("selected", sel, ns, NULL, NULL);
    }
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
