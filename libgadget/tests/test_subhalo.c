/* Unit tests for the runtime subhalo finder, Phase 1 (halofinder_phase1_plan.md section 5).
 *
 * Run as: mpirun -np 4 .objs/test_subhalo
 *
 *  1. test_subhalo_table          track table: append past the initial allocation, home-rank and
 *                                 local-index arithmetic, redistribution when Chunk changes,
 *                                 fetch/store round trips, free.
 *  2. test_subhalo_sort_key       the 16-byte PIG sort key and the local comparator.
 *  3. test_subhalo_exchange_tags  TrackId/BindRank/BoundTrackId survive domain_exchange.
 *  4. test_subhalo_join           restart tag join from a synthetic PIG written in this test.
 *  5. test_subhalo_fof_as_subhalo the Phase 1 driver over five calls on two clumps.
 *  6. test_subhalo_forced_failure in-place (oversized) keys, and the exchange failure path.
 *  7. test_subhalo_plan_pack      the packing as a particle-free function on the audit's two
 *                                 inputs (the eight-pass appendix table, the tail type mix),
 *                                 checked against the budgets of the final staying particles.
 *  8. test_subhalo_unhosted_direct hostless centre with a periodic wrap under three rank
 *                                 partitions, swallowed BHs in the vote/counts/mass, dead
 *                                 tracks, non-default velocity units.
 *  9. test_subhalo_lifecycle_pig  the driver plus the PIG writer and restart over four calls
 *                                 with an empty interior rank: hostless, dead (not sunk) and
 *                                 zero-group calls, km/s velocities, the restart round trip on
 *                                 both PIG branches, and corrupted states (truncated payload,
 *                                 duplicate TrackIds, wrong dtype) rejected.
 * 10. test_subhalo_many_keys      200 keys (> 32 per rank): the tail packing on real particles.
 *
 * The range checks of subhalo_sort_key abort through endrun (MPI_Abort), so they cannot run
 * in-process. Set SUBHALO_TEST_SORTKEY_OVERFLOW=grnr|trackid|bindrank to run only a test that
 * feeds the overflowing value: the expected outcome is an abort with the endrun message, e.g.
 *   SUBHALO_TEST_SORTKEY_OVERFLOW=grnr mpirun -np 1 .objs/test_subhalo ; echo $?   (non-zero)
 */

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
    sp.SubhaloTreeOpenAngle = 0.45;
    sp.SubhaloMergeTrapped = 1;
    sp.SubhaloSinkSymmetric = 0;
    sp.SubhaloCoreSize = 20;
    sp.SubhaloReassignGas = 1;
    sp.SubhaloUnbindWithThermal = 0;
    sp.SubhaloFieldSO = 1;
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
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, MPI_COMM_WORLD);
    fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
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
        LOCAL_CHECK(bad, P[i].TrackId == tagT(P[i].ID) && P[i].BindRank == tagB(P[i].ID) && P[i].BoundTrackId == tagBT(P[i].ID),
                "particle ID %lu: tags (%ld %d %ld) expected (%ld %d %ld)\n", (unsigned long) P[i].ID,
                P[i].TrackId, P[i].BindRank, P[i].BoundTrackId, tagT(P[i].ID), tagB(P[i].ID), tagBT(P[i].ID));
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
 *           particles keep tag 0 outside any group
 *  stage 4: the 30 come back to A and B is translated onto A: one group of 1560; track 1 wins
 *           by previous Nbound (840 > 690) although its TrackId is larger, track 0 sinks into 1. */
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
    int d;
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
        rand_in_ball(id, 1, T5_RB, T5_cB, out);
        if(stage == 4)
            for(d = 0; d < 3; d++)
                out[d] += T5_cA[d] - T5_cB[d];
        return;
    }
    int64_t k = id - T5_NA - T5_NB;
    if(stage >= 3 && k < T5_NFMOVE) {
        t5_field_in_B(id, out);
        if(stage == 4)
            for(d = 0; d < 3; d++)
                out[d] += T5_cA[d] - T5_cB[d];
        return;
    }
    t5_field_lattice(k, out);
}

/* Expected FOF label (0: the group containing A's core, 1: B's, -1: none) and TrackId. */
static void
t5_expect(MyIDType id, int stage, int * label, int64_t * track)
{
    if(id < T5_NA) {
        *label = (stage == 3 && id < T5_NDISP) ? -1 : 0;
        *track = stage == 4 ? 1 : 0;
        return;
    }
    if(id < T5_NA + T5_NB) {
        int cap = t5_is_cap(id);
        if(stage <= 1) { *label = 1; *track = 1; }
        else if(stage <= 3) { *label = cap ? 0 : 1; *track = cap ? 0 : 1; }
        else { *label = 0; *track = 1; }
        return;
    }
    int64_t k = id - T5_NA - T5_NB;
    int moved = k < T5_NFMOVE;
    if(stage < 3 || !moved) { *label = -1; *track = -1; }
    else if(stage == 3) { *label = 1; *track = 1; }
    else { *label = 0; *track = 1; }
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
    for(i = 0; i < PartManager->NumPart; i++)
        t5_position(P[i].ID, stage, P[i].Pos);
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
        CHECK(fabs(rec->CorePos[d] - rec->CoMPos[d]) < 1e-4, "stage %d track %ld: CorePos[%d] %.8g != CoMPos %.8g\n", stage, t, d, rec->CorePos[d], rec->CoMPos[d]);
        /* the particles have zero velocity */
        CHECK(fabs(rec->CoMVel[d]) < 1e-6, "stage %d track %ld: CoMVel[%d] = %g\n", stage, t, d, rec->CoMVel[d]);
    }
    CHECK(mb != NULL, "stage %d track %ld: no particle with BindRank 0\n", stage, t);
    if(inplace) {
        CHECK(rec->Flags & SUBHALO_FLAG_INPLACE, "stage %d track %ld: in-place key without SUBHALO_FLAG_INPLACE (Flags %d)\n", stage, t, rec->Flags);
        return;
    }
    CHECK(!(rec->Flags & SUBHALO_FLAG_INPLACE), "stage %d track %ld: serial key flagged in place (Flags %d)\n", stage, t, rec->Flags);
    CHECK(rec->MostBoundID == mb->ID, "stage %d track %ld: MostBoundID %lu, BindRank-0 particle %lu\n",
            stage, t, (unsigned long) rec->MostBoundID, (unsigned long) mb->ID);
    for(d = 0; d < 3; d++)
        CHECK(fabs(rec->MostBoundPos[d] - mb->Pos[d]) < 1e-4, "stage %d track %ld: MostBoundPos[%d]\n", stage, t, d);
    /* BindRank is the order of increasing distance from the centre. */
    double * dk = malloc(n * sizeof(double));
    for(i = 0; i < n; i++)
        dk[i] = -1;
    for(i = 0; i < ntot; i++)
        if(all[i].GrNr == grnr && all[i].BindRank >= 0 && all[i].BindRank < n)
            dk[all[i].BindRank] = dist2(all[i].Pos, rec->CorePos);
    int bad = 0;
    for(i = 1; i < n; i++)
        if(dk[i] < dk[i - 1] - 1e-9 * (1 + dk[i - 1]))
            bad++;
    free(dk);
    CHECK(bad == 0, "stage %d track %ld: %d BindRank inversions against the distance from CorePos\n", stage, t, bad);
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
            LOCAL_CHECK(bad, r[i].BoundTrackId == -1, "stage %d: untagged particle %ld has BoundTrackId %ld\n", stage, i, r[i].BoundTrackId);
        else if(label >= 0)
            LOCAL_CHECK(bad, r[i].BoundTrackId == track, "stage %d: particle %ld BoundTrackId %ld != TrackId %ld\n", stage, i, r[i].BoundTrackId, track);
        else if(prev) {
            /* tagged, outside any group: GrNr = -1 particles keep their tags untouched (plan 1 step 3) */
            LOCAL_CHECK(bad, r[i].TrackId == prev[i].TrackId && r[i].BindRank == prev[i].BindRank && r[i].BoundTrackId == prev[i].BoundTrackId,
                    "stage %d: field particle %ld tags changed from (%ld %d %ld) to (%ld %d %ld)\n", stage, i,
                    prev[i].TrackId, prev[i].BindRank, prev[i].BoundTrackId, r[i].TrackId, r[i].BindRank, r[i].BoundTrackId);
        }
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
        if(stage == 4 && t == 0)
            continue;
        /* the label whose group this track holds */
        int lab = (stage == 4) ? 0 : t;
        check_hosted_record(&rec[t], t, grnr[lab], r, n, stage, inplace);
    }
    if(stage == 4) {
        CHECK(rec[0].Nbound == 0, "stage 4: sunk track 0 has Nbound %d\n", rec[0].Nbound);
        CHECK(rec[0].SinkTrackId == 1, "stage 4: track 0 SinkTrackId %ld, expected 1\n", rec[0].SinkTrackId);
        CHECK(rec[0].SnapSink == 4 && rec[0].SnapDeath == 4, "stage 4: track 0 SnapSink %d SnapDeath %d\n", rec[0].SnapSink, rec[0].SnapDeath);
    } else {
        CHECK(rec[0].SnapSink == -1 && rec[0].SnapDeath == -1 && rec[1].SnapSink == -1 && rec[1].SnapDeath == -1,
                "stage %d: a live track has SnapSink/SnapDeath set\n", stage);
    }
    /* History (plan 1 step 13): Mbound peaks. Track 0: 600, 600, 720, 720 (690 now); track 1:
     * 360, 360, 360 (240 now), 840, 1560. SnapMboundPeak is not checked where two calls tie. */
    static const float peak0[5] = {600, 600, 720, 720, 720};
    static const float peak1[5] = {360, 360, 360, 840, 1560};
    static const int snap0[5] = {0, -9, 2, 2, 2};
    static const int snap1[5] = {0, -9, 0, 3, 4};
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

    if(NTask > 1) {
        sp.SubhaloDebugForceExchangeFailure = 1;
        subhalo_set_testpar(sp);
        ret = run_finder_call(1, &CP, &ngroups);
        CHECK(ret == 0, "forced failure: subhalo_find returned %d, expected 0\n", ret);
        CHECK(subhalo_last_call_complete() == 0, "forced failure: last call reported complete\n");
        CHECK(subhalo_last_call_snapshot() == 0, "forced failure: last call snapshot moved to %d\n", subhalo_last_call_snapshot());
        CHECK(tab->NTotal == T6_NCLUMP, "forced failure: NTotal changed to %ld\n", tab->NTotal);
        struct tagrec * r1 = gather_tags(&n1);
        CHECK(n1 == ntot, "forced failure: %ld particles after the restore, expected %ld\n", n1, ntot);
        int64_t i;
        int bad = 0;
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

    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    teardown_pman();
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
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, MPI_COMM_WORLD);
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
            /* A dispersed: no group left anywhere */
            out[0] = 30. + 90. * (id % 10);
            out[1] = 30. + 90. * (id / 10);
            out[2] = 150.;
            return;
        }
        rand_in_ball(id, 11, 6.0, cA, out);
        return;
    }
    if(id < LC_NA + LC_NB) {
        if(stage >= 1) {
            const int64_t k = id - LC_NA;
            out[0] = 50. + 90. * (k % 10);
            out[1] = 50. + 90. * (k / 10);
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
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
