/* Tests of the oversized candidates unbound in place (subhalo_collective.c, design 7.8), MPI.
 * Run as: mpirun -np 4 .objs/test_subhalo_collective (green at 1 to 4 threads).
 * Tests (the test_collective_ prefix omitted):
 *  1. serial_vs_inplace   a clump with an embedded satellite, serial against fully in place.
 *  2. subtree_keys        the central in place, its satellites' sub-trees as serial keys; D10.
 *  3. merger              two oversized groups: the loser's sub-tree in round 0, then the central.
 *  4. kill_passup         a dying in-place satellite passes its particles to its parent.
 *  5. binder              tails returned to a surviving serial binder, in its energy order.
 *  6. serial_repairs      returned tails and reserved tracers against a reference from the old tags.
 *  7. relaxed_boundary    sub-tree remainder tags on both sides of the relaxed-source boundary.
 *  8. dead_field          a dead in-place candidate with field particles on other ranks.
 *  9. sunk_root_field     a serial root sinks into the in-place central, its descendant survives.
 * 10. failure_after_split exchange failure after a real split: tags and records restored.
 * 11. voided_claim        the voided-claim rule over three rounds and the central promotion.
 * 12. long_chain          four in-place rounds, two groups of unequal depth.
 * 13. admission           a memory squeeze refuses the collective stage before the exchange.
 * 14. admission_props     the deferred-properties stage as the largest admission need.
 * 15. guards              int32 and MPI byte-count guards.
 * 16. walk_theta          the potential-only walk at two opening angles against a direct sum.
 * 17. props_identity      serial and distributed deferred properties agree on four layouts.
 * 18. central_source      decision 52: a central's FOF-less particles untagged, a satellite's kept.
 * SUBHALO_TEST_ONLY=<substring> runs only the matching tests. */

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
#include <libgadget/subhalo_collective.h>
#include <libgadget/subhalo_unbind.h>
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
/* gravshort.h defines static treewalk callbacks this test does not use */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <libgadget/gravshort.h>
#pragma GCC diagnostic pop

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
    int32_t Task;            /* the rank holding it */
    int32_t pad;
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
        r->Task = get_thistask();
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

/* One finder call exactly as run.c makes it, on the current particle positions; pre (after the
 * FOF, before the finder) and post (after the finder) are called on every rank if not NULL. */
typedef void (*call_hook)(void * ud);
static int
run_finder_call_ex(int snap, Cosmology * CP, int64_t * TotNgroups, call_hook pre, call_hook post, void * ud)
{
    double atime = 0.5 + 0.05 * snap;
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    *TotNgroups = fof.TotNgroups;
    if(pre)
        pre(ud);
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, 0, MPI_COMM_WORLD);
    if(post)
        post(ud);
    fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
}

static int
run_finder_call(int snap, Cosmology * CP, int64_t * TotNgroups)
{
    return run_finder_call_ex(snap, CP, TotNgroups, NULL, NULL, NULL);
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


#define BOXSIZE 1000.

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

/* The particle type of a synthetic id (NULL: every particle a DM tracer of type 1); a type-0
 * particle gets an SPH slot (zeroed), a non-tracer that FOF attaches as a secondary. */
static int (*SynTypeFn)(MyIDType id) = NULL;

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
    if(SynTypeFn) {
        /* room for every gas particle on one rank (a whole key may land anywhere) */
        int64_t atleast[6] = {ntot, 0, 0, 0, 128, 128};
        slots_reserve(1, atleast, SlotsManager);
    }
    MyIDType id;
    int64_t ngas = 0;
    for(id = 0; id < (MyIDType) ntot; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3] = {0, 0, 0};
        add_dm_particle(id, pos);
        struct particle_data * p = &P[PartManager->NumPart - 1];
        if(SynTypeFn && SynTypeFn(id) == 0) {
            p->Type = 0;
            p->PI = ngas++;
        }
    }
    if(SynTypeFn) {
        SlotsManager->info[0].size = ngas;
        int64_t i;
        for(i = 0; i < PartManager->NumPart; i++)
            if(P[i].Type == 0)
                memset(&SPHP(i), 0, sizeof(struct sph_particle_data));
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

/* ------------------------------------------------------------------ comparison helpers */

/* Opening angle of both the serial octree and the in-place walk in these tests (the default 0.45
 * leaves 1e-4 relative potential errors that can swap near-degenerate energies). */
#define COLL_TEST_THETA 0.15

/* One finished call: the particles and the records, identical on every rank. */
struct Outcome {
    struct tagrec * r;
    int64_t n;
    struct SubTrack * rec;
    int64_t nt;
    struct SubhaloCollectiveStats st;
};

static void
outcome_take(struct Outcome * o)
{
    o->r = gather_tags(&o->n);
    o->nt = subhalo_get_table()->NTotal;
    o->rec = fetch_all_tracks(o->nt);
    o->st = *subhalo_collective_last_stats();
}

static void
outcome_free(struct Outcome * o)
{
    free(o->rec);
    free(o->r);
}

/* The first ten tracer-type (type 1) own-tag particles of track t in BindRank order. */
static int
ten_tracers(const struct tagrec * r, int64_t n, int64_t t, MyIDType out[10])
{
    int64_t i;
    int k = 0;
    int32_t best[10];
    for(k = 0; k < 10; k++) {
        best[k] = INT32_MAX;
        out[k] = (MyIDType) -1;
    }
    for(i = 0; i < n; i++) {
        if(r[i].TrackId != t || r[i].Type != 1)
            continue;
        /* insertion into the ten smallest ranks */
        int32_t b = r[i].BindRank;
        MyIDType id = r[i].ID;
        for(k = 0; k < 10; k++)
            if(b < best[k]) {
                int32_t tb = best[k];
                MyIDType ti = out[k];
                best[k] = b;
                out[k] = id;
                b = tb;
                id = ti;
            }
    }
    int nf = 0;
    for(k = 0; k < 10; k++)
        if(best[k] < INT32_MAX)
            nf++;
    return nf;
}

static double
reldiff(double a, double b)
{
    const double s = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return s > 0 ? fabs(a - b) / s : 0;
}

/* Serial versus in place for the resolved tracks of a call: Nbound within nbtol, identical ten
 * most-bound tracers, the properties within ptol; returns the largest relative Nbound
 * difference. Logs the numbers. */
static double
compare_outcomes(const struct Outcome * a, const struct Outcome * b, double nbtol, double ptol, int strict, const char * what)
{
    CHECK(a->nt == b->nt && a->n == b->n, "%s: %ld/%ld tracks, %ld/%ld particles\n", what, a->nt, b->nt, a->n, b->n);
    int64_t t, i;
    double worst = 0;
    for(t = 0; t < a->nt; t++) {
        const struct SubTrack * x = &a->rec[t], * y = &b->rec[t];
        CHECK((x->Nbound > 0) == (y->Nbound > 0), "%s: track %ld alive serially %d, in place %d\n", what, t, x->Nbound > 0, y->Nbound > 0);
        if(x->Nbound <= 0)
            continue;
        const double dn = reldiff(x->Nbound, y->Nbound);
        if(dn > worst)
            worst = dn;
        MyIDType ta[10], tb[10];
        const int na = ten_tracers(a->r, a->n, t, ta), nb = ten_tracers(b->r, b->n, t, tb);
        int same = na == nb, k;
        for(k = 0; k < na && same; k++)
            same = ta[k] == tb[k];
        message(0, "%s: track %ld: Nbound %d / %d (%.2e), NboundExclusive %d / %d, Rank %d / %d, parent %ld / %ld, ten tracers %s, "
                   "MostBoundID %lu / %lu, Vmax %g / %g, Rhalf %g / %g, MBoundSO[vir] %g / %g, SpinPeebles %g / %g, BtoA %g / %g, "
                   "CoreSigmaX %g / %g, Ekin %g / %g, Epot %g / %g\n", what, t,
                x->Nbound, y->Nbound, dn, x->NboundExclusive, y->NboundExclusive, x->Rank, y->Rank, x->NestedParentTrackId,
                y->NestedParentTrackId, same ? "identical" : "DIFFERENT", (unsigned long) x->MostBoundID, (unsigned long) y->MostBoundID,
                x->Vmax, y->Vmax, x->Rhalf, y->Rhalf, x->MBoundSO[0], y->MBoundSO[0], x->SpinPeebles, y->SpinPeebles,
                x->BtoA, y->BtoA, x->CoreSigmaX, y->CoreSigmaX, x->Ekin, y->Ekin, x->Epot, y->Epot);
        if(!same)
            message(0, "%s: track %ld ten tracers serially %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu, in place %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n",
                    what, t, ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6], ta[7], ta[8], ta[9], tb[0], tb[1], tb[2], tb[3], tb[4], tb[5],
                    tb[6], tb[7], tb[8], tb[9]);
        CHECK(dn <= nbtol, "%s: track %ld Nbound %d serially, %d in place\n", what, t, x->Nbound, y->Nbound);
        /* strict: the same ten in the same order; else the same set up to one exchanged at the
         * edge (near-degenerate energies of a cold ball swap under the tree's 1e-4 errors) */
        int common = 0, k2;
        for(k = 0; k < na; k++)
            for(k2 = 0; k2 < nb; k2++)
                if(ta[k] == tb[k2])
                    common++;
        CHECK(strict ? same : (na == nb && common >= na - 1), "%s: track %ld: the ten most-bound tracers differ (%d in common)\n", what, t, common);
        CHECK(x->MostBoundID == y->MostBoundID, "%s: track %ld MostBoundID %lu / %lu\n", what, t, (unsigned long) x->MostBoundID, (unsigned long) y->MostBoundID);
        CHECK(x->Rank == y->Rank && x->Depth == y->Depth && x->NestedParentTrackId == y->NestedParentTrackId
              && x->CentralTrackId == y->CentralTrackId && x->HostGroup == y->HostGroup,
              "%s: track %ld hierarchy (%d %d %ld %ld %ld) / (%d %d %ld %ld %ld)\n", what, t, x->Rank, x->Depth, x->NestedParentTrackId,
              x->CentralTrackId, x->HostGroup, y->Rank, y->Depth, y->NestedParentTrackId, y->CentralTrackId, y->HostGroup);
        /* properties within the tree tolerance */
        const double props[][2] = {
            {x->Mbound, y->Mbound}, {x->Vmax, y->Vmax}, {x->Rhalf, y->Rhalf}, {x->MBoundSO[0], y->MBoundSO[0]},
            {x->RBoundSO[0], y->RBoundSO[0]}, {x->MBoundSO[2], y->MBoundSO[2]}, {x->VmaxVir, y->VmaxVir}, {x->Ekin, y->Ekin},
            {x->Epot, y->Epot}, {x->SpinPeebles, y->SpinPeebles}, {x->CoreSigmaX, y->CoreSigmaX}, {x->CoreSigmaV, y->CoreSigmaV},
            {x->Renclose, y->Renclose},
        };
        const char * pname[] = {"Mbound", "Vmax", "Rhalf", "MBoundSO[vir]", "RBoundSO[vir]", "MBoundSO[200c]", "VmaxVir", "Ekin", "Epot",
                                "SpinPeebles", "CoreSigmaX", "CoreSigmaV", "Renclose"};
        /* the core of 20 particles: one exchanged particle of near-degenerate energy moves its
         * dispersions by a few per cent (cold synthetic balls) */
        for(k = 0; k < (int) (sizeof(props) / sizeof(props[0])); k++) {
            const double tol = (k == 10 || k == 11) && !strict ? 8 * ptol : ptol;
            CHECK(reldiff(props[k][0], props[k][1]) <= tol, "%s: track %ld %s %g serially, %g in place\n", what, t, pname[k], props[k][0], props[k][1]);
        }
        int d;
        for(d = 0; d < 3; d++)
            CHECK(fabs(x->CorePos[d] - y->CorePos[d]) < 0.1 * x->CoreSigmaX + 0.01 && fabs(x->CoMPos[d] - y->CoMPos[d]) < 0.05,
                  "%s: track %ld CorePos[%d] %g / %g CoMPos %g / %g\n", what, t, d, x->CorePos[d], y->CorePos[d], x->CoMPos[d], y->CoMPos[d]);
        CHECK(fabs(x->BtoA - y->BtoA) < 0.02 && fabs(x->CtoA - y->CtoA) < 0.02, "%s: track %ld BtoA %g / %g CtoA %g / %g\n", what, t,
              x->BtoA, y->BtoA, x->CtoA, y->CtoA);
        double jx = 0, jy = 0;
        for(d = 0; d < 3; d++) {
            jx += x->J[d] * x->J[d];
            jy += y->J[d] * y->J[d];
        }
        CHECK(reldiff(sqrt(jx), sqrt(jy)) <= 2 * ptol, "%s: track %ld |J| %g / %g\n", what, t, sqrt(jx), sqrt(jy));
        /* the Phase 5 columns: Rs fitted in both modes or -1 in both, Vrms within ptol, the two
         * mean velocities within ptol Vrms (plus 0.01 km/s for a cold ball) */
        message(0, "%s: track %ld Rs %g / %g (RsKlypin %g / %g), Vrms %g / %g, BulkVel (%g %g %g) / (%g %g %g), "
                   "RockstarCoreVel (%g %g %g) / (%g %g %g)\n", what, t, x->Rs, y->Rs, x->RsKlypin, y->RsKlypin, x->Vrms, y->Vrms,
                x->BulkVel[0], x->BulkVel[1], x->BulkVel[2], y->BulkVel[0], y->BulkVel[1], y->BulkVel[2],
                x->RockstarCoreVel[0], x->RockstarCoreVel[1], x->RockstarCoreVel[2],
                y->RockstarCoreVel[0], y->RockstarCoreVel[1], y->RockstarCoreVel[2]);
        CHECK((x->Rs == -1 && y->Rs == -1) || (x->Rs > 0 && y->Rs > 0 && reldiff(x->Rs, y->Rs) <= ptol),
              "%s: track %ld Rs %g serially, %g in place\n", what, t, x->Rs, y->Rs);
        CHECK((x->Vrms == -1 && y->Vrms == -1) || reldiff(x->Vrms, y->Vrms) <= ptol, "%s: track %ld Vrms %g serially, %g in place\n",
              what, t, x->Vrms, y->Vrms);
        for(d = 0; d < 3; d++) {
            const double vtol = ptol * fmax(x->Vrms, 0) + 0.01;
            CHECK(fabs(x->BulkVel[d] - y->BulkVel[d]) <= vtol && fabs(x->RockstarCoreVel[d] - y->RockstarCoreVel[d]) <= vtol,
                  "%s: track %ld BulkVel[%d] %g / %g RockstarCoreVel %g / %g\n", what, t, d, x->BulkVel[d], y->BulkVel[d],
                  x->RockstarCoreVel[d], y->RockstarCoreVel[d]);
        }
        CHECK(!(x->Flags & SUBHALO_FLAG_INPLACE), "%s: track %ld flagged in place in the serial run\n", what, t);
    }
    /* BoundTrackId and TrackId of every particle */
    int64_t nbd = 0, ntd = 0;
    for(i = 0; i < a->n; i++) {
        CHECK(a->r[i].ID == b->r[i].ID, "%s: particle %ld IDs %lu / %lu\n", what, i, (unsigned long) a->r[i].ID, (unsigned long) b->r[i].ID);
        if(a->r[i].BoundTrackId != b->r[i].BoundTrackId) nbd++;
        if(a->r[i].TrackId != b->r[i].TrackId) ntd++;
    }
    message(0, "%s: largest Nbound difference %.3e; %ld of %ld particles with another BoundTrackId, %ld with another TrackId\n", what,
            worst, nbd, a->n, ntd);
    CHECK(nbd <= 0.002 * a->n, "%s: %ld particles with another BoundTrackId\n", what, nbd);
    CHECK(ntd <= 0.005 * a->n, "%s: %ld particles with another TrackId\n", what, ntd);
    return worst;
}

/* ------------------------------------------------------------------ test 1 */

/* The overlap scenario of test_subhalo.c: call 0 two clumps, call 1 B on a circular orbit at 5
 * kpc/h in A with 60 particles pulled to A's centre. Subsampling off; the field SO on. */
static void
t1_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    p2_setup(&CP);
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    subhalo_set_testpar(sp);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int64_t ng;
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int ret = run_finder_call(0, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "serial/in place: call 0 has %ld groups\n", ng);
    sp.SubhaloCollectiveMinPart = collective;
    subhalo_set_testpar(sp);
    p2_infall(&CP, 0.55);
    ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "serial/in place: call 1 has %ld groups\n", ng);
    outcome_take(o);
    p2_teardown();
}

static void
test_collective_serial_vs_inplace(void ** state)
{
    struct Outcome a, b;
    t1_run(0, &a);
    CHECK(a.st.NCands == 0, "serial run: %ld in-place candidates\n", a.st.NCands);
    t1_run(1, &b);
    message(0, "serial/in place: %d rounds, %ld candidates (%ld + %ld), %d + %d walks, %g + %g s\n", b.st.NRounds, b.st.NCands,
            b.st.NCandsRound[0], b.st.NCandsRound[1], b.st.Iterations[0], b.st.Iterations[1], b.st.Seconds[0], b.st.Seconds[1]);
    CHECK(b.st.NRounds == 2 && b.st.NCands == 2 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1,
          "in place: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    CHECK(a.nt == 2 && a.rec[1].Nbound > 0 && a.rec[1].Rank == 1 && a.rec[1].NestedParentTrackId == 0,
          "serial: B Nbound %d Rank %d parent %ld\n", a.rec[1].Nbound, a.rec[1].Rank, a.rec[1].NestedParentTrackId);
    CHECK((b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && (b.rec[1].Flags & SUBHALO_FLAG_INPLACE), "in place: Flags %d %d\n",
          b.rec[0].Flags, b.rec[1].Flags);
    compare_outcomes(&a, &b, 1e-3, 1e-2, 1, "serial/in place");
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "in place") == 0, "in place: tag invariants\n");
    /* the field SO of the in-place central */
    CHECK(reldiff(a.rec[0].MSO[0], b.rec[0].MSO[0]) < 1e-2, "serial/in place: MSO[vir] %g / %g\n", a.rec[0].MSO[0], b.rec[0].MSO[0]);
    outcome_free(&b);
    outcome_free(&a);
}

/* ------------------------------------------------------------------ synthetic hierarchies (tests 2-4) */

/* Test 2: A (0, 2000) the central, B (1, 300) and C (2, 200) its satellites 5 kpc/h (physical)
 * either side, D (3, 100) B's satellite, and S, 60 of B's previous particles at A's centre moving
 * against B: B's unbound remainder, bound by A. */
#define ST_A0 0
#define ST_B0 2000
#define ST_C0 2300
#define ST_D0 2500
#define ST_S0 2600
#define ST_END 2660

static void
st_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    double cB[3] = {500 + 5 / atime, 500, 500}, cC[3] = {500 - 5 / atime, 500, 500}, cD[3] = {500 + 6.2 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 40, 0}, vC[3] = {0, -40, 0}, vD[3] = {0, 44, 0}, vS[3] = {0, -40, 0};
    (void) ud;
    if(id >= ST_END)
        return;
    if(id < ST_B0) {
        syn_ball(id, 51, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id - ST_A0;
    }
    else if(id < ST_C0) {
        syn_ball(id, 52, cB, 1.0, vB, 8, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - ST_B0;
    }
    else if(id < ST_D0) {
        syn_ball(id, 53, cC, 0.8, vC, 8, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - ST_C0;
    }
    else if(id < ST_S0) {
        syn_ball(id, 54, cD, 0.3, vD, 3, atime, s);
        s->tag = s->bound = 3;
        s->rank = id - ST_D0;
    }
    else {
        syn_ball(id, 55, cA, 0.5, vS, 5, atime, s);
        s->tag = s->bound = 1;
        s->rank = 300 + id - ST_S0;
    }
    (void) cD;
}

static void
st_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, ST_END + SYN_NFIELD, sp);
    struct SubTrack recs[4];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 360, 0.036, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 0, 0, 2, 1);
    recs[3] = syn_rec(3, 100, 0.01, 1, 0, 3, 2);
    syn_records(recs, 4);
    syn_place(st_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sub-tree keys: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_subtree_keys(void ** state)
{
    struct Outcome a, b;
    st_run(0, &a);
    st_run(1000, &b);
    CHECK(b.st.NRounds == 1 && b.st.NCands == 1, "sub-tree keys: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    CHECK((b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[1].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[2].Flags & SUBHALO_FLAG_INPLACE)
          && !(b.rec[3].Flags & SUBHALO_FLAG_INPLACE), "sub-tree keys: Flags %d %d %d %d\n", b.rec[0].Flags, b.rec[1].Flags,
          b.rec[2].Flags, b.rec[3].Flags);
    int t;
    for(t = 1; t < 4; t++)
        CHECK(b.rec[t].Nbound > 0, "sub-tree keys: track %d Nbound %d\n", t, b.rec[t].Nbound);
    CHECK(b.rec[1].NestedParentTrackId == 0 && b.rec[1].Depth == 1 && b.rec[2].NestedParentTrackId == 0 && b.rec[2].Depth == 1
          && b.rec[3].NestedParentTrackId == 1 && b.rec[3].Depth == 2 && b.rec[1].CentralTrackId == 0 && b.rec[3].CentralTrackId == 0,
          "sub-tree keys: parents %ld %ld %ld depths %d %d %d\n", b.rec[1].NestedParentTrackId, b.rec[2].NestedParentTrackId,
          b.rec[3].NestedParentTrackId, b.rec[1].Depth, b.rec[2].Depth, b.rec[3].Depth);
    /* the provisional remainder of B's sub-tree reaches the central: S bound by A under B's tail tag */
    int64_t i, nS = 0, nSA = 0;
    for(i = 0; i < b.n; i++)
        if(b.r[i].ID >= ST_S0 && b.r[i].ID < ST_END) {
            nS++;
            if(b.r[i].BoundTrackId == 0 && (b.r[i].TrackId == 1 || b.r[i].TrackId == 0))
                nSA++;
        }
    {
        int64_t h[3][4];
        memset(h, 0, sizeof(h));
        for(i = 0; i < b.n; i++)
            if(b.r[i].ID >= ST_S0 && b.r[i].ID < ST_END) {
                h[0][b.r[i].TrackId + 1 < 4 ? b.r[i].TrackId + 1 : 3]++;
                h[1][b.r[i].BoundTrackId + 1 < 4 ? b.r[i].BoundTrackId + 1 : 3]++;
            }
        for(i = 0; i < a.n; i++)
            if(a.r[i].ID >= ST_S0 && a.r[i].ID < ST_END)
                h[2][a.r[i].BoundTrackId + 1 < 4 ? a.r[i].BoundTrackId + 1 : 3]++;
        message(0, "sub-tree keys: stripped particles: tags (-1, 0, 1, >1) %ld %ld %ld %ld, bound %ld %ld %ld %ld; serial bound %ld %ld %ld %ld\n",
                h[0][0], h[0][1], h[0][2], h[0][3], h[1][0], h[1][1], h[1][2], h[1][3], h[2][0], h[2][1], h[2][2], h[2][3]);
    }
    message(0, "sub-tree keys: %ld of %ld stripped particles bound by the central\n", nSA, nS);
    CHECK(nS == 60 && nSA >= 55, "sub-tree keys: %ld of %ld stripped particles bound by A\n", nSA, nS);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "sub-tree keys") == 0, "sub-tree keys: tag invariants\n");
    compare_outcomes(&a, &b, 1e-3, 1e-2, 0, "sub-tree keys");
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 3: A (0, 1500 particles, central of group 1 at the previous call) and B (1, 1000, central
 * of group 2) merge into one group: B, 6 kpc/h from A at 50 km/s, is too light to compete for
 * the centre (0.67 < SubhaloMajorProgenitorMassRatio) and becomes A's satellite. */
#define MG_B0 1500
#define MG_END 2500

static void
mg_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    double cB[3] = {500, 500 + 6 / atime, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {50, 0, 0};
    (void) ud;
    if(id >= MG_END)
        return;
    if(id < MG_B0) {
        syn_ball(id, 61, cA, 2.0, zero, 15, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 62, cB, 1.5, vB, 15, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - MG_B0;
    }
}

static void
mg_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, MG_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 1500, 0.15, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 1000, 0.10, -1, 1, 0, 0);
    recs[1].HostGroup = 2;
    syn_records(recs, 2);
    syn_place(mg_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "merger: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_merger(void ** state)
{
    struct Outcome a, b;
    mg_run(0, &a);
    mg_run(800, &b);
    message(0, "merger: %d rounds (%ld + %ld candidates, %d + %d walks, %g + %g s)\n", b.st.NRounds, b.st.NCandsRound[0],
            b.st.NCandsRound[1], b.st.Iterations[0], b.st.Iterations[1], b.st.Seconds[0], b.st.Seconds[1]);
    CHECK(b.st.NRounds == 2 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1, "merger: %d rounds\n", b.st.NRounds);
    CHECK(b.rec[1].Nbound > 0 && b.rec[1].Rank == 1 && b.rec[1].NestedParentTrackId == 0 && b.rec[1].CentralTrackId == 0
          && (b.rec[1].Flags & SUBHALO_FLAG_INPLACE) && (b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && b.rec[0].Rank == 0,
          "merger: B Nbound %d Rank %d parent %ld Flags %d / %d\n", b.rec[1].Nbound, b.rec[1].Rank, b.rec[1].NestedParentTrackId,
          b.rec[1].Flags, b.rec[0].Flags);
    CHECK(fabs(b.rec[1].Mbound - a.rec[1].Mbound) <= 2e-3 * a.rec[1].Mbound, "merger: B Mbound %g in place, %g serially\n",
          b.rec[1].Mbound, a.rec[1].Mbound);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "merger") == 0, "merger: tag invariants\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "merger");
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 4: A (0, 3000 particles in 1 kpc/h, cold) and its satellite B (1, 500) whose particles now
 * fill A's centre with a 150 km/s dispersion: B (in place at depth 1) dies in round 0, its
 * particles pass to A in round 1, which binds them. */
#define KP_B0 3000
#define KP_END 3500

static void
kp_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= KP_END)
        return;
    if(id < KP_B0) {
        syn_ball(id, 71, cA, 1.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 72, cA, 1.0, zero, 150, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - KP_B0;
    }
}

static void
kp_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, KP_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 500, 0.05, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(kp_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "kill: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_kill_passup(void ** state)
{
    struct Outcome a, b;
    kp_run(0, &a);
    kp_run(300, &b);
    message(0, "kill: %d rounds (%ld + %ld candidates), %ld died, %ld sunk; serial B Nbound %d SnapDeath %d Sink %ld; "
               "in place B Nbound %d SnapDeath %d Sink %ld; A Nbound %d / %d\n", b.st.NRounds, b.st.NCandsRound[0], b.st.NCandsRound[1],
            b.st.NDied, b.st.NSunk, a.rec[1].Nbound, a.rec[1].SnapDeath, a.rec[1].SinkTrackId, b.rec[1].Nbound, b.rec[1].SnapDeath,
            b.rec[1].SinkTrackId, a.rec[0].Nbound, b.rec[0].Nbound);
    CHECK(b.st.NRounds == 2 && b.st.NDied == 1, "kill: %d rounds, %ld died\n", b.st.NRounds, b.st.NDied);
    CHECK(b.rec[1].Nbound == 0 && b.rec[1].SnapDeath == 1 && a.rec[1].Nbound == 0 && b.rec[1].SinkTrackId == a.rec[1].SinkTrackId,
          "kill: B Nbound %d SnapDeath %d Sink %ld (serial Sink %ld)\n", b.rec[1].Nbound, b.rec[1].SnapDeath, b.rec[1].SinkTrackId,
          a.rec[1].SinkTrackId);
    int64_t i, nB = 0, nBA = 0;
    for(i = 0; i < b.n; i++)
        if(b.r[i].ID >= KP_B0 && b.r[i].ID < KP_END) {
            nB++;
            if(b.r[i].BoundTrackId == 0)
                nBA++;
        }
    message(0, "kill: %ld of %ld of B's particles bound by A\n", nBA, nB);
    CHECK(nBA >= 0.9 * nB && b.rec[0].Nbound >= 3000 + 0.9 * 500, "kill: %ld of %ld of B's particles bound by A (A Nbound %d)\n",
          nBA, nB, b.rec[0].Nbound);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "kill") == 0, "kill: tag invariants\n");
    compare_outcomes(&a, &b, 1e-3, 1e-2, 0, "kill");
    outcome_free(&b);
    outcome_free(&a);
}

/* ------------------------------------------------------------------ Phase 2b audit fixes (Q1-Q5) */

/* A point at physical distance rphys from cen (comoving) in a direction drawn from (id, salt). */
static void
syn_at_radius(MyIDType id, uint64_t salt, const double cen[3], double rphys, double atime, double out[3])
{
    double u[3];
    const double c0[3] = {0, 0, 0};
    double n2 = 0;
    int d;
    do {
        rand_in_ball(id, salt + (uint64_t) (n2 * 1e6), 1.0, c0, u);
        n2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    } while(n2 < 1e-4);
    for(d = 0; d < 3; d++)
        out[d] = cen[d] + rphys * u[d] / sqrt(n2) / atime;
}

/* ---- test 5: the binder takes the particle (audit Q3) and test 6: many serial repairs (Q2) ----
 * Per group g (centre 500 + 125 g): A (3 g, 2000 particles, the in-place central), B (3 g + 1,
 * 300 particles, 5 kpc/h away, a serial satellite) and C (3 g + 2, 100 particles at A's centre,
 * B's satellite: it sinks into A in the rounds while B survives). T: 60 previous tails of C
 * bound by B, spread through B's ball at radii decreasing with the ID (the more bound, the higher
 * the ID: ID order opposed to the energy order), every fourth one gas (a non-tracer). R (optional,
 * 30): previous tails of B at A's very centre and cold, which A binds and reserves. */
#define BD_NA 2000
#define BD_NB 300
#define BD_NC 100
#define BD_NT 60
#define BD_NR 30
static int BdGroups = 1, BdWithR = 0;

static int64_t
bd_npg(void)
{
    return BD_NA + BD_NB + BD_NC + BD_NT + (BdWithR ? BD_NR : 0);
}

/* the part of an id: 0 A, 1 B, 2 C, 3 T, 4 R; -1 lattice */
static int
bd_part(MyIDType id, int * g, int64_t * j)
{
    const int64_t npg = bd_npg();
    if(id >= (MyIDType) (BdGroups * npg))
        return -1;
    *g = (int) (id / npg);
    int64_t k = (int64_t) (id % npg);
    if(k < BD_NA) { *j = k; return 0; }
    k -= BD_NA;
    if(k < BD_NB) { *j = k; return 1; }
    k -= BD_NB;
    if(k < BD_NC) { *j = k; return 2; }
    k -= BD_NC;
    if(k < BD_NT) { *j = k; return 3; }
    *j = k - BD_NT;
    return 4;
}

static int
bd_type(MyIDType id)
{
    int g;
    int64_t j;
    return bd_part(id, &g, &j) == 3 && j % 4 == 1 ? 0 : 1;
}

static void
bd_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    int g;
    int64_t j;
    const int part = bd_part(id, &g, &j);
    (void) ud;
    if(part < 0)
        return;
    const double cA[3] = {500 + 125 * g, 500, 500};
    const double cB[3] = {cA[0] + 5 / atime, cA[1], cA[2]};
    /* B fast enough that the cold knots at A's centre (C, R) are not bound to it in its key */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    const int64_t TA = 3 * g, TB = 3 * g + 1, TC = 3 * g + 2;
    int d;
    switch(part) {
        case 0:
            syn_ball(id, 81, cA, 3.0, zero, 20, atime, s);
            s->tag = s->bound = TA;
            s->rank = j;
            break;
        case 1:
            syn_ball(id, 82, cB, 1.0, vB, 8, atime, s);
            s->tag = s->bound = TB;
            s->rank = j;
            break;
        case 2:
            syn_ball(id, 83, cA, 0.1, zero, 0.5, atime, s);
            s->tag = s->bound = TC;
            s->rank = j;
            break;
        case 3: {
            /* radius decreasing with j: the last ones are the most bound in B */
            syn_at_radius(id, 84, cB, 0.02 + 0.9 * (1 - (double) j / BD_NT), atime, s->pos);
            double v[3];
            rand_in_ball(id, 85, 1.0, zero, v);
            for(d = 0; d < 3; d++)
                s->vel[d] = atime * (vB[d] + v[d]);
            s->tag = TC;
            s->bound = TB;
            s->rank = BD_NC + j;
            break;
        }
        default:
            /* as cold as C at A's centre: among A's most-bound tracers once C has sunk */
            syn_ball(id, 86, cA, 0.1, zero, 0.5, atime, s);
            s->tag = TB;
            s->bound = TA;
            s->rank = BD_NB + j;
            break;
    }
}

/* the pre-final-pass state, captured by the hook (test 6) */
static struct tagrec * BdPre = NULL;
static int64_t BdNPre = 0;
static struct SubhaloBindOrd * BdBo = NULL;
static int64_t BdNBo = 0;

static int
cmp_bindord_test(const void * a, const void * b)
{
    const MyIDType x = ((const struct SubhaloBindOrd *) a)->ID, y = ((const struct SubhaloBindOrd *) b)->ID;
    return (x > y) - (x < y);
}

static void
bd_hook(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan)
{
    (void) plan;
    BdPre = gather_tags(&BdNPre);
    const int NTask = get_ntask();
    int nb = (int) (ctx->NBindOrd * sizeof(struct SubhaloBindOrd));
    int * counts = malloc(NTask * sizeof(int));
    int * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    BdBo = malloc(tot + sizeof(struct SubhaloBindOrd));
    MPI_Allgatherv(ctx->BindOrd, nb, MPI_BYTE, BdBo, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    BdNBo = tot / sizeof(struct SubhaloBindOrd);
    qsort(BdBo, BdNBo, sizeof(struct SubhaloBindOrd), cmp_bindord_test);
    free(displ);
    free(counts);
}

static void
bd_run(int64_t collective, int ngroups, int withR, int hook, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    BdGroups = ngroups;
    BdWithR = withR;
    SynTypeFn = bd_type;
    syn_setup(&CP, ngroups * bd_npg() + SYN_NFIELD, sp);
    SynTypeFn = NULL;
    struct SubTrack * recs = malloc(sizeof(struct SubTrack) * 3 * ngroups);
    int g;
    for(g = 0; g < ngroups; g++) {
        recs[3 * g] = syn_rec(3 * g, BD_NA, BD_NA * SYN_M, -1, 3 * g, 0, 0);
        recs[3 * g + 1] = syn_rec(3 * g + 1, BD_NB + BD_NT, (BD_NB + BD_NT) * SYN_M, 3 * g, 3 * g, 1, 1);
        recs[3 * g + 2] = syn_rec(3 * g + 2, BD_NC, BD_NC * SYN_M, 3 * g + 1, 3 * g, 2, 2);
        int t;
        for(t = 0; t < 3; t++)
            recs[3 * g + t].HostGroup = g + 1;
    }
    syn_records(recs, 3 * ngroups);
    free(recs);
    syn_place(bd_particle, NULL, 0.55, 1);
    if(hook)
        subhalo_collective_set_pretag_hook(bd_hook);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    subhalo_collective_set_pretag_hook(NULL);
    CHECK(ret == 1 && ng == ngroups, "binder: %ld groups, ret %d\n", ng, ret);
    outcome_take(o);
    syn_teardown();
}

/* Test 5 (audit Q3): A in place, B a serial survivor, C (B's satellite) sinks into A while B
 * survives: the tails of C that B binds return to B, merged into B's own-tag sequence by B's
 * final-pass energy order. The serial run (every track in one key) ranks B's own-tag particles
 * by one pass over B's energy-ordered list: it is the oracle, rank by rank. */
static void
test_collective_binder(void ** state)
{
    struct Outcome a, b;
    bd_run(0, 1, 0, 0, &a);
    bd_run(1000, 1, 0, 0, &b);
    const int64_t TA = 0, TB = 1, TC = 2;
    message(0, "binder: in place %ld candidates, %ld sunk, %ld tails given to their binder, %ld taken; C Nbound %d / %d Sink %ld / %ld; "
               "B Nbound %d / %d NboundExclusive %d / %d\n", b.st.NCands, b.st.NSunk, b.st.NAdded, b.st.NStolen, a.rec[TC].Nbound,
            b.rec[TC].Nbound, a.rec[TC].SinkTrackId, b.rec[TC].SinkTrackId, a.rec[TB].Nbound, b.rec[TB].Nbound,
            a.rec[TB].NboundExclusive, b.rec[TB].NboundExclusive);
    CHECK(b.st.NCands == 1 && (b.rec[TA].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[TB].Flags & SUBHALO_FLAG_INPLACE),
          "binder: %ld candidates, Flags %d %d\n", b.st.NCands, b.rec[TA].Flags, b.rec[TB].Flags);
    CHECK(b.rec[TC].Nbound == 0 && b.rec[TC].SinkTrackId == TA && a.rec[TC].Nbound == 0 && a.rec[TC].SinkTrackId == TA,
          "binder: C should sink into A in both runs\n");
    CHECK(b.rec[TB].Nbound > 0 && b.rec[TB].NestedParentTrackId == TA, "binder: B Nbound %d parent %ld\n", b.rec[TB].Nbound,
          b.rec[TB].NestedParentTrackId);
    /* B's own reservation (its ten most-bound tracers, J12) already takes the innermost T in the key */
    CHECK(b.st.NAdded >= BD_NT - 12, "binder: %ld tails given to the binder\n", b.st.NAdded);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "binder in place") == 0, "binder: tag invariants\n");
    CHECK(p2_check_tags(a.r, a.n, a.rec, a.nt, "binder serial") == 0, "binder: tag invariants (serial)\n");
    CHECK(a.rec[TB].NboundExclusive == b.rec[TB].NboundExclusive && a.rec[TB].Nbound == b.rec[TB].Nbound,
          "binder: B NboundExclusive %d / %d\n", a.rec[TB].NboundExclusive, b.rec[TB].NboundExclusive);
    /* B's own-tag particles, rank by rank, against the serial run; T bound and tagged by B */
    int64_t i, nB = 0, ndiff = 0, nT = 0, nTgas = 0, nTtr = 0, maxown = -1, minT = INT32_MAX, agree = 0, pairs = 0;
    for(i = 0; i < b.n; i++) {
        int g;
        int64_t j;
        const int part = bd_part(b.r[i].ID, &g, &j);
        if(b.r[i].TrackId == TB || a.r[i].TrackId == TB) {
            nB++;
            if(b.r[i].TrackId != a.r[i].TrackId || b.r[i].BindRank != a.r[i].BindRank || b.r[i].BoundTrackId != a.r[i].BoundTrackId) {
                if(ndiff < 10)
                    message(0, "binder: particle %lu (part %d): serial (%ld, %d, %ld), in place (%ld, %d, %ld)\n", (unsigned long) b.r[i].ID,
                            part, a.r[i].TrackId, a.r[i].BindRank, a.r[i].BoundTrackId, b.r[i].TrackId, b.r[i].BindRank, b.r[i].BoundTrackId);
                ndiff++;
            }
        }
        if(part == 1 && b.r[i].TrackId == TB && b.r[i].BoundTrackId == TB && b.r[i].BindRank > maxown)
            maxown = b.r[i].BindRank;
        if(part == 3) {
            if(b.r[i].TrackId == TB && b.r[i].BoundTrackId == TB) {
                nT++;
                if(b.r[i].Type == 0) nTgas++;
                else nTtr++;
                if(b.r[i].BindRank < minT)
                    minT = b.r[i].BindRank;
            }
        }
    }
    /* the ID order of T against its rank order (the old code appended by ID) */
    for(i = 0; i < b.n; i++) {
        int g, g2;
        int64_t j, j2, k;
        if(bd_part(b.r[i].ID, &g, &j) != 3 || b.r[i].TrackId != TB)
            continue;
        for(k = i + 1; k < b.n; k++)
            if(bd_part(b.r[k].ID, &g2, &j2) == 3 && b.r[k].TrackId == TB) {
                pairs++;
                if(b.r[k].BindRank > b.r[i].BindRank)
                    agree++;
            }
    }
    message(0, "binder: %ld own-tag rows of B, %ld differ from the serial run; %ld of T given to B (%ld gas, %ld tracers), "
               "first T rank %ld, last own exclusive rank %ld; %ld of %ld T pairs in ID order\n", nB, ndiff, nT, nTgas, nTtr, minT, maxown,
            agree, pairs);
    CHECK(ndiff == 0, "binder: %ld of B's own-tag rows differ from the serial run\n", ndiff);
    CHECK(nT >= BD_NT - 2 && nTgas > 0 && nTtr > 0, "binder: %ld T given to B (%ld gas, %ld tracers)\n", nT, nTgas, nTtr);
    CHECK(minT < maxown, "binder: the returned tails are not merged into B's exclusive sequence (first %ld, last own %ld)\n", minT, maxown);
    CHECK(agree < pairs / 4, "binder: %ld of %ld T pairs in ID order: the energy order was meant to oppose it\n", agree, pairs);
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 6 (audit Q2): three groups, each with tails losing their claimant (C sinks: the tails
 * return to the binder B) and B's tails at A's centre that A binds and reserves: the serial ranks
 * are repaired on the tracks' result ranks. The reference is computed here from the gathered
 * pre-pass tags (the serial keys' provisional ranks, captured by the hook) and binder ordinals:
 * for every serial track, its final own-tag particles sorted by (kept exclusive: old rank;
 * returned: binder ordinal g before the old exclusive rank g, then its energy position; kept
 * tail: after every exclusive one, by old rank) must carry exactly that dense rank. */
static void
test_collective_serial_repairs(void ** state)
{
    struct Outcome b;
    bd_run(1000, 3, 1, 1, &b);
    CHECK(BdPre && BdNPre == b.n, "repairs: hook captured %ld particles of %ld\n", BdNPre, b.n);
    message(0, "repairs: %ld candidates, %ld sunk, %ld returned to their binder, %ld taken by in-place candidates, %ld tracks repaired, "
               "%ld rank queries, %g s\n", b.st.NCands, b.st.NSunk, b.st.NAdded, b.st.NStolen, b.st.NRepairTracks, b.st.NRepairQueries,
            b.st.RepairSeconds);
    CHECK(b.st.NCands == 3 && b.st.NAdded >= 3 * (BD_NT - 12) && b.st.NStolen >= 3, "repairs: %ld candidates, %ld added, %ld taken\n",
          b.st.NCands, b.st.NAdded, b.st.NStolen);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "repairs") == 0, "repairs: tag invariants\n");
    int64_t T, i, nbad = 0, nchk = 0, nrem = 0, nadd = 0;
    int64_t * key = malloc(sizeof(int64_t) * 3 * (b.n + 1));
    int64_t * idx = malloc(sizeof(int64_t) * (b.n + 1));
    for(T = 0; T < b.nt; T++) {
        if(b.rec[T].Nbound <= 0 || (b.rec[T].Flags & SUBHALO_FLAG_INPLACE))
            continue;
        int64_t nbex = 0, nown = 0;
        for(i = 0; i < b.n; i++)
            if(BdPre[i].TrackId == T && BdPre[i].BoundTrackId == T)
                nbex++;
        for(i = 0; i < b.n; i++) {
            if(BdPre[i].TrackId == T && b.r[i].TrackId != T)
                nrem++;
            if(b.r[i].TrackId != T)
                continue;
            int64_t * k = key + 3 * nown;
            if(BdPre[i].TrackId == T) {
                if(BdPre[i].BindRank < nbex) {
                    k[0] = BdPre[i].BindRank;
                    k[1] = 1;
                }
                else {
                    k[0] = INT32_MAX + (int64_t) BdPre[i].BindRank;
                    k[1] = 1;
                }
                k[2] = 0;
            }
            else {
                struct SubhaloBindOrd q = {b.r[i].ID, 0, 0};
                const struct SubhaloBindOrd * e = bsearch(&q, BdBo, BdNBo, sizeof(struct SubhaloBindOrd), cmp_bindord_test);
                CHECK(e && BdPre[i].BoundTrackId == T, "repairs: particle %lu returned to %ld without a binder ordinal (pre bound %ld)\n",
                      (unsigned long) b.r[i].ID, T, BdPre[i].BoundTrackId);
                k[0] = e->g;
                k[1] = 0;
                k[2] = e->f;
                nadd++;
            }
            idx[nown++] = i;
        }
        /* insertion sort of the keys (small tracks) */
        int64_t p, q2;
        for(p = 1; p < nown; p++)
            for(q2 = p; q2 > 0; q2--) {
                int64_t * x = key + 3 * (q2 - 1), * y = key + 3 * q2;
                if(x[0] < y[0] || (x[0] == y[0] && (x[1] < y[1] || (x[1] == y[1] && x[2] <= y[2]))))
                    break;
                int64_t t3[3] = {x[0], x[1], x[2]}, ti = idx[q2 - 1];
                x[0] = y[0]; x[1] = y[1]; x[2] = y[2];
                y[0] = t3[0]; y[1] = t3[1]; y[2] = t3[2];
                idx[q2 - 1] = idx[q2];
                idx[q2] = ti;
            }
        for(p = 0; p < nown; p++) {
            nchk++;
            if(b.r[idx[p]].BindRank != p) {
                if(nbad < 10)
                    message(0, "repairs: track %ld particle %lu: BindRank %d, reference %ld (pre tag %ld rank %d bound %ld)\n", T,
                            (unsigned long) b.r[idx[p]].ID, b.r[idx[p]].BindRank, p, BdPre[idx[p]].TrackId, BdPre[idx[p]].BindRank,
                            BdPre[idx[p]].BoundTrackId);
                nbad++;
            }
        }
    }
    message(0, "repairs: %ld serial own-tag rows checked against the reference, %ld differ; %ld removed, %ld returned\n", nchk, nbad, nrem, nadd);
    CHECK(nbad == 0 && nrem >= 3 && nadd >= 3 * (BD_NT - 12), "repairs: %ld rows differ (%ld removed, %ld returned)\n", nbad, nrem, nadd);
    free(idx);
    free(key);
    free(BdBo);
    free(BdPre);
    BdBo = NULL;
    BdPre = NULL;
    outcome_free(&b);
}

/* ---- test 7: the sub-tree remainder on both sides of the relaxed-source boundary ----
 * A (0, 2000) the central in place; B (1, 100 particles, 5 kpc/h away) a serial satellite whose
 * 350 previous tails S sit at A's centre moving against B: B's relaxed source keeps R Nbound =
 * 300 list entries, so about 200 of S stay B's tails (bound by A) and the rest, beyond the
 * boundary, go to A. B's list is the serial kernel's in both runs, so every S particle must have
 * the same tag (and B's tails the same rank) serially and in place. */
#define RB_B0 2000
#define RB_S0 2100
#define RB_END 2450

static void
rb_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 5 / atime, 500, 500};
    /* S spread over 2 kpc/h at rest and B fast: S is bound neither to itself nor to B */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    (void) ud;
    if(id >= RB_END)
        return;
    if(id < RB_B0) {
        syn_ball(id, 91, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < RB_S0) {
        syn_ball(id, 92, cB, 0.5, vB, 5, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - RB_B0;
    }
    else {
        syn_ball(id, 93, cA, 2.0, zero, 5, atime, s);
        s->tag = 1;
        s->bound = -1;
        s->rank = 100 + id - RB_S0;
    }
}

static void
rb_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, RB_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 100, 0.01, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(rb_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "relaxed boundary: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_relaxed_boundary(void ** state)
{
    struct Outcome a, b;
    rb_run(0, &a);
    rb_run(1000, &b);
    CHECK(b.st.NCands == 1 && !(b.rec[1].Flags & SUBHALO_FLAG_INPLACE), "relaxed boundary: %ld candidates\n", b.st.NCands);
    int64_t i, nB = 0, nA = 0, nother = 0, ndiff = 0, nbA = 0;
    for(i = 0; i < b.n; i++) {
        if(b.r[i].ID < RB_S0 || b.r[i].ID >= RB_END)
            continue;
        if(b.r[i].TrackId == 1) nB++;
        else if(b.r[i].TrackId == 0) nA++;
        else nother++;
        if(b.r[i].BoundTrackId == 0)
            nbA++;
        if(b.r[i].TrackId != a.r[i].TrackId || (b.r[i].TrackId == 1 && b.r[i].BindRank != a.r[i].BindRank)) {
            if(ndiff < 10)
                message(0, "relaxed boundary: particle %lu serial (%ld, %d), in place (%ld, %d)\n", (unsigned long) b.r[i].ID,
                        a.r[i].TrackId, a.r[i].BindRank, b.r[i].TrackId, b.r[i].BindRank);
            ndiff++;
        }
    }
    int64_t tl = 0;
    int t;
    for(t = 0; t < 6; t++)
        tl += b.rec[1].NtailType[t];
    message(0, "relaxed boundary: B Nbound %d NboundExclusive %d tails %ld; of S: %ld B's tails, %ld A's, %ld else, %ld bound by A; "
               "%ld differ from the serial run\n", b.rec[1].Nbound, b.rec[1].NboundExclusive, tl, nB, nA, nother, nbA, ndiff);
    CHECK(ndiff == 0, "relaxed boundary: %ld particles of S with another tag than serially\n", ndiff);
    CHECK(nB > 0 && nA > 0 && nother == 0 && nB + nA == RB_END - RB_S0, "relaxed boundary: S split %ld (B) / %ld (A) / %ld\n", nB, nA, nother);
    /* inside: B's whole relaxed tail is S (B's own ball is bound); beyond: A's */
    CHECK(nB == tl && b.rec[1].NboundExclusive + tl <= (int64_t) (3 * b.rec[1].NboundExclusive) + 1,
          "relaxed boundary: %ld of S tagged B, B's tail count %ld, NboundExclusive %d\n", nB, tl, b.rec[1].NboundExclusive);
    CHECK(nbA >= 0.95 * (RB_END - RB_S0), "relaxed boundary: %ld of S bound by A\n", nbA);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "relaxed boundary") == 0, "relaxed boundary: tag invariants\n");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- test 8: a dead in-place candidate with field particles ----
 * A (0, 3000 cold particles in 1 kpc/h) and its satellite B (1) at A's centre: 100 cold gas
 * particles (non-tracers), 10 tracers of which one is hot, and 20 tracers in the field far away
 * (outside every FOF group, on other ranks) that move fast. B, in place (depth 1), keeps 9 bound
 * tracers after its first iteration and dies; its whole source passes to A, whose relaxed source
 * holds all of it; B's previous tracers vote remotely and give A as its descendant. */
#define DK_B0 3000
#define DK_BT 3100
#define DK_F0 3110
#define DK_END 3130

static int
dk_type(MyIDType id)
{
    return id >= DK_B0 && id < DK_BT ? 0 : 1;
}

static void
dk_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    /* the group in another part of the box than the field particles (other ranks) */
    const double cA[3] = {750, 750, 750};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= DK_END)
        return;
    if(id < DK_B0) {
        syn_ball(id, 101, cA, 1.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < DK_F0) {
        /* B's tracers first in its previous rank order (its step-0 tracers), then its gas */
        const int tracer = id >= DK_BT;
        syn_ball(id, 102, cA, 0.3, zero, tracer && id == DK_F0 - 1 ? 300 : 3, atime, s);
        s->tag = s->bound = 1;
        s->rank = tracer ? id - DK_BT : 10 + id - DK_B0;
    }
    else {
        /* field: a sparse line far from the group, 15 kpc/h apart, moving fast */
        const int64_t j = id - DK_F0;
        s->pos[0] = 125 + 15 * j;
        s->pos[1] = 125;
        s->pos[2] = 125;
        s->vel[0] = atime * 150;
        s->tag = s->bound = 1;
        s->rank = 110 + j;
    }
}

static void
test_collective_dead_field(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = 100;
    SynTypeFn = dk_type;
    syn_setup(&CP, DK_END + SYN_NFIELD, sp);
    SynTypeFn = NULL;
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 130, 0.013, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(dk_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "dead field: %ld groups\n", ng);
    struct Outcome o;
    outcome_take(&o);
    syn_teardown();
    int64_t i, nB = 0, nA = 0, nbA = 0, nF = 0, nFA = 0, nFfield = 0;
    int ftask = -1, gtask_mask = 0;
    int64_t fother = 0;
    for(i = 0; i < o.n; i++)
        if(o.r[i].ID < DK_F0 && o.r[i].GrNr == 1)
            gtask_mask |= 1 << o.r[i].Task;
    for(i = 0; i < o.n; i++) {
        if(o.r[i].ID < DK_B0 || o.r[i].ID >= DK_END)
            continue;
        nB++;
        if(o.r[i].TrackId == 0) nA++;
        if(o.r[i].BoundTrackId == 0) nbA++;
        if(o.r[i].ID >= DK_F0) {
            nF++;
            if(o.r[i].TrackId == 0 && o.r[i].BoundTrackId == -1) nFA++;
            if(o.r[i].GrNr == -1) nFfield++;
            if(!(gtask_mask & (1 << o.r[i].Task))) fother++;
            ftask = o.r[i].Task;
        }
    }
    message(0, "dead field: %d rounds, %ld died; B Nbound %d SnapDeath %d Sink %ld Descendant %ld; of B's %ld particles %ld tagged A, "
               "%ld bound by A; field: %ld of %ld A's tails, %ld outside every group, %ld on ranks without a group particle (e.g. %d)\n",
            o.st.NRounds, o.st.NDied, o.rec[1].Nbound, o.rec[1].SnapDeath, o.rec[1].SinkTrackId, o.rec[1].DescendantTrackId, nB, nA, nbA,
            nFA, nF, nFfield, fother, ftask);
    CHECK(o.st.NRounds == 2 && o.st.NDied == 1, "dead field: %d rounds, %ld died\n", o.st.NRounds, o.st.NDied);
    CHECK(o.rec[1].Nbound == 0 && o.rec[1].SnapDeath == 1 && o.rec[1].SinkTrackId == -1 && o.rec[1].DescendantTrackId == 0,
          "dead field: B Nbound %d SnapDeath %d Sink %ld Descendant %ld\n", o.rec[1].Nbound, o.rec[1].SnapDeath, o.rec[1].SinkTrackId,
          o.rec[1].DescendantTrackId);
    CHECK(nA == nB && nFA == nF && nFfield == nF && nbA >= 105, "dead field: %ld of %ld reached A (%ld field as A's tails), %ld bound by A\n",
          nA, nB, nFA, nbA);
    CHECK(get_ntask() == 1 || fother > 0, "dead field: no field particle on a rank without group particles\n");
    CHECK(p2_check_tags(o.r, o.n, o.rec, o.nt, "dead field") == 0, "dead field: tag invariants\n");
    outcome_free(&o);
}

/* ---- test 9: a serial root with field particles sinks into the in-place central ----
 * A (0, 3000) in place; R (1, 150 particles at A's centre, A's satellite) and D (2, 200 particles
 * 5 kpc/h away, R's satellite) form a serial sub-tree key. FR: 10 field particles of R 30 kpc/h
 * (comoving) out, bound to R; FD: 10 field particles of D, bound to D; DT: 20 tails of D at A's
 * centre that R binds. R sinks into A in the round: its whole bound set (FR, DT) joins A; D
 * survives under A and keeps its claims (FD bound by D, DT bound by A). */
#define SR_R0 3000
#define SR_FR0 3150
#define SR_D0 3160
#define SR_FD0 3360
#define SR_DT0 3370
#define SR_END 3390

static void
sr_dir(int64_t j, int sgn, double u[3])
{
    const double al = M_PI / 3, ph = 2 * M_PI * j / 10.;
    u[0] = sgn * cos(al);
    u[1] = sin(al) * cos(ph);
    u[2] = sin(al) * sin(ph);
}

static void
sr_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cD[3] = {500 + 5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vD[3] = {0, 40, 0};
    double u[3], v[3];
    int d;
    (void) ud;
    if(id >= SR_END)
        return;
    if(id < SR_R0) {
        syn_ball(id, 111, cA, 1.5, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < SR_FR0) {
        syn_ball(id, 112, cA, 0.3, zero, 3, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - SR_R0;
    }
    else if(id < SR_D0) {
        sr_dir(id - SR_FR0, -1, u);
        rand_in_ball(id, 113, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cA[d] + 30 * u[d];
            s->vel[d] = atime * v[d];
        }
        s->tag = s->bound = 1;
        s->rank = 150 + id - SR_FR0;
    }
    else if(id < SR_FD0) {
        syn_ball(id, 114, cD, 0.8, vD, 6, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - SR_D0;
    }
    else if(id < SR_DT0) {
        sr_dir(id - SR_FD0, 1, u);
        rand_in_ball(id, 115, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cD[d] + 30 * u[d];
            s->vel[d] = atime * (vD[d] + v[d]);
        }
        s->tag = s->bound = 2;
        s->rank = 200 + id - SR_FD0;
    }
    else {
        syn_ball(id, 116, cA, 0.3, zero, 2, atime, s);
        s->tag = 2;
        s->bound = 1;
        s->rank = 210 + id - SR_DT0;
    }
}

static void
sr_setup(Cosmology * CP, int64_t collective, int failure)
{
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloDebugForceExchangeFailure = failure;
    syn_setup(CP, SR_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 160, 0.016, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 210, 0.021, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(sr_particle, NULL, 0.55, 1);
}

static void
sr_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    sr_setup(&CP, collective, 0);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sunk root: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_sunk_root_field(void ** state)
{
    struct Outcome a, b;
    sr_run(0, &a);
    sr_run(1000, &b);
    const struct SubTrack * R = &b.rec[1], * D = &b.rec[2];
    message(0, "sunk root: %ld candidates, %ld sunk, %ld taken, %ld returned; R Nbound %d Sink %ld SnapSink %d Descendant %ld; D Nbound %d "
               "parent %ld Depth %d central %ld\n", b.st.NCands, b.st.NSunk, b.st.NStolen, b.st.NAdded, R->Nbound, R->SinkTrackId, R->SnapSink,
            R->DescendantTrackId, D->Nbound, D->NestedParentTrackId, D->Depth, D->CentralTrackId);
    CHECK(b.st.NCands == 1 && b.st.NSunk >= 1, "sunk root: %ld candidates, %ld sunk\n", b.st.NCands, b.st.NSunk);
    CHECK(R->Nbound == 0 && R->SinkTrackId == 0 && R->SnapSink == 1 && R->SnapDeath == 1 && R->DescendantTrackId == 0,
          "sunk root: R's sink links\n");
    CHECK(D->Nbound > 0 && D->NestedParentTrackId == 0 && D->Depth == 1 && D->CentralTrackId == 0 && D->Rank == 1, "sunk root: D's links\n");
    int64_t i, nR = 0, nRA = 0, nFR = 0, nFRA = 0, nFD = 0, nFDD = 0, nDT = 0, nDTD = 0, nDTA = 0, nfield = 0;
    for(i = 0; i < b.n; i++) {
        const MyIDType id = b.r[i].ID;
        if(id >= SR_R0 && id < SR_D0) {
            nR++;
            if(b.r[i].TrackId == 0 && b.r[i].BoundTrackId == 0) nRA++;
            if(id >= SR_FR0) {
                nFR++;
                if(b.r[i].TrackId == 0 && b.r[i].BoundTrackId == 0) nFRA++;
            }
        }
        if(id >= SR_FD0 && id < SR_DT0) {
            nFD++;
            if(b.r[i].TrackId == 2 && b.r[i].BoundTrackId == 2) nFDD++;
        }
        if(id >= SR_DT0 && id < SR_END) {
            nDT++;
            if(b.r[i].TrackId == 2) nDTD++;
            if(b.r[i].BoundTrackId == 0) nDTA++;
        }
        if(((id >= SR_FR0 && id < SR_D0) || (id >= SR_FD0 && id < SR_DT0)) && b.r[i].GrNr == -1)
            nfield++;
    }
    message(0, "sunk root: R's bound set %ld/%ld to A (field %ld/%ld); D's field %ld/%ld kept; D's tails at A's centre %ld/%ld tagged D, %ld "
               "bound by A; %ld field particles outside the group\n", nRA, nR, nFRA, nFR, nFDD, nFD, nDTD, nDT, nDTA, nfield);
    CHECK(nRA == nR && nFRA == nFR && nFR == 10, "sunk root: R's bound set not transferred to A (%ld/%ld)\n", nRA, nR);
    CHECK(nFDD == nFD && nFD == 10 && nfield == 20, "sunk root: D's field particles (%ld/%ld kept, %ld outside)\n", nFDD, nFD, nfield);
    CHECK(nDTA == nDT && nDTD >= nDT - 10, "sunk root: D's claims bound by A (%ld tagged D, %ld bound by A of %ld)\n", nDTD, nDTA, nDT);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "sunk root") == 0, "sunk root: tag invariants\n");
    CHECK(a.rec[1].Nbound == 0 && a.rec[1].SinkTrackId == 0, "sunk root: the serial run sinks R into A too\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "sunk root");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- test 18: the central's source is its FOF group only (decision 52, HBT FeedCentrals) ----
 * A (0, 3000) the central with FA, 20 of its previous particles at rest in the field far away
 * (bound to A were they in its source); B (1, 200) its satellite 5 kpc/h out with FB, 10 field
 * particles bound to B. Serially, with A in place and B a sub-tree key, and with both in place:
 * FA untagged (the in-place final pass releases them), FB kept by B. */
#define CS_FA0 3000
#define CS_B0 3020
#define CS_FB0 3220
#define CS_END 3230

static void
cs_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 40, 0};
    double u[3], v[3];
    int d;
    (void) ud;
    if(id >= CS_END)
        return;
    if(id < CS_FA0) {
        syn_ball(id, 121, cA, 1.5, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < CS_B0) {
        /* a sparse line far from the group, 15 kpc/h apart, at rest */
        s->pos[0] = 125 + 15 * (id - CS_FA0);
        s->pos[1] = 125;
        s->pos[2] = 125;
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < CS_FB0) {
        syn_ball(id, 122, cB, 0.8, vB, 6, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - CS_B0;
    }
    else {
        sr_dir(id - CS_FB0, 1, u);
        rand_in_ball(id, 123, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cB[d] + 30 * u[d];
            s->vel[d] = atime * (vB[d] + v[d]);
        }
        s->tag = s->bound = 1;
        s->rank = 200 + id - CS_FB0;
    }
}

static void
cs_run(int64_t collective, struct Outcome * o, int64_t * nreleased)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, CS_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3020, 0.302, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 210, 0.021, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(cs_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "central source (%ld): %ld groups\n", collective, ng);
    *nreleased = subhalo_last_call_stats()->NFallbackReleased;
    outcome_take(o);
    syn_teardown();
}

/* The FOF-less particles of the central untagged, the satellite's kept and bound by it. */
static void
cs_check(const struct Outcome * o, const char * what)
{
    int64_t i, nFA = 0, nFAu = 0, nFB = 0, nFBk = 0, nfield = 0;
    for(i = 0; i < o->n; i++) {
        const struct tagrec * q = &o->r[i];
        if(q->ID >= CS_FA0 && q->ID < CS_B0) {
            nFA++;
            nFAu += q->TrackId == -1 && q->BindRank == -1 && q->BoundTrackId == -1;
            nfield += q->GrNr == -1;
        }
        if(q->ID >= CS_FB0 && q->ID < CS_END) {
            nFB++;
            nFBk += q->TrackId == 1 && q->BoundTrackId == 1 && q->BindRank >= 0;
            nfield += q->GrNr == -1;
        }
    }
    message(0, "%s: A Nbound %d Rank %d, B Nbound %d Rank %d parent %ld; the central's FOF-less %ld/%ld untagged, the satellite's %ld/%ld "
               "kept, %ld outside the group\n", what, o->rec[0].Nbound, o->rec[0].Rank, o->rec[1].Nbound, o->rec[1].Rank,
            o->rec[1].NestedParentTrackId, nFAu, nFA, nFBk, nFB, nfield);
    CHECK(o->rec[0].Rank == 0 && o->rec[1].Rank == 1 && o->rec[1].NestedParentTrackId == 0 && o->rec[1].Nbound > 0,
          "%s: the hierarchy\n", what);
    CHECK(nFA == CS_B0 - CS_FA0 && nFAu == nFA && nFB == CS_END - CS_FB0 && nFBk == nFB && nfield == nFA + nFB,
          "%s: FOF-less particles (%ld/%ld untagged, %ld/%ld kept)\n", what, nFAu, nFA, nFBk, nFB);
    CHECK(o->rec[0].Nbound <= CS_FA0, "%s: A binds %d particles, more than its group part\n", what, o->rec[0].Nbound);
    CHECK(p2_check_tags(o->r, o->n, o->rec, o->nt, what) == 0, "%s: tag invariants\n", what);
}

static void
test_collective_central_source(void ** state)
{
    struct Outcome a, b, c;
    int64_t ra, rb, rc;
    cs_run(0, &a, &ra);
    cs_run(1000, &b, &rb);
    cs_run(100, &c, &rc);
    message(0, "central source: candidates %ld / %ld / %ld, released %ld / %ld / %ld (serial, A in place, A and B in place)\n",
            a.st.NCands, b.st.NCands, c.st.NCands, ra, rb, rc);
    CHECK(a.st.NCands == 0 && b.st.NCands == 1 && c.st.NCands == 2, "central source: candidates %ld / %ld / %ld\n", a.st.NCands,
          b.st.NCands, c.st.NCands);
    CHECK(ra == 0 && rb == CS_B0 - CS_FA0 && rc == CS_B0 - CS_FA0, "central source: released %ld / %ld / %ld\n", ra, rb, rc);
    cs_check(&a, "central source, serial");
    cs_check(&b, "central source, A in place");
    cs_check(&c, "central source, A and B in place");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "central source, A in place");
    compare_outcomes(&a, &c, 2e-3, 1e-2, 0, "central source, A and B in place");
    outcome_free(&c);
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- test 10: a failed exchange after a real split ----
 * The sunk-root fixture with A in place (sub-tree key R + D, its group particles and the field
 * particles FR, FD travel encoded) and SubhaloDebugForceExchangeFailure: GrNr, the tags and the
 * records are those before the call and the call is incomplete. */
struct FailSnap {
    struct tagrec * r;
    int64_t n;
};

static void
fail_pre(void * ud)
{
    struct FailSnap * s = (struct FailSnap *) ud;
    s->r = gather_tags(&s->n);
}

static void
test_collective_failure_after_split(void ** state)
{
    Cosmology CP;
    sr_setup(&CP, 1000, 1);
    struct SubTrack * before = fetch_all_tracks(3);
    struct FailSnap pre = {NULL, 0};
    int64_t ng;
    const int ret = run_finder_call_ex(1, &CP, &ng, fail_pre, NULL, &pre);
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    message(0, "failure after split: ret %d complete %d; plan: %ld sub-tree keys, %ld candidates, %ld group and %ld field particles encoded\n",
            ret, subhalo_last_call_complete(), ps->NSubKeys, ps->NCands, ps->NEncFof, ps->NEncField);
    CHECK(ret == 0 && !subhalo_last_call_complete(), "failure after split: ret %d\n", ret);
    CHECK(ps->NSubKeys >= 1 && ps->NCands >= 1 && ps->NEncFof > 0 && ps->NEncField > 0, "failure after split: no real split\n");
    int64_t n;
    struct tagrec * post = gather_tags(&n);
    CHECK(n == pre.n, "failure after split: %ld / %ld particles\n", n, pre.n);
    int64_t i, bad = 0;
    for(i = 0; i < n; i++)
        if(post[i].ID != pre.r[i].ID || post[i].GrNr != pre.r[i].GrNr || post[i].TrackId != pre.r[i].TrackId
           || post[i].BindRank != pre.r[i].BindRank || post[i].BoundTrackId != pre.r[i].BoundTrackId) {
            if(bad < 5)
                message(0, "failure after split: particle %lu GrNr %ld -> %ld tag (%ld, %d, %ld) -> (%ld, %d, %ld)\n", (unsigned long) post[i].ID,
                        pre.r[i].GrNr, post[i].GrNr, pre.r[i].TrackId, pre.r[i].BindRank, pre.r[i].BoundTrackId, post[i].TrackId,
                        post[i].BindRank, post[i].BoundTrackId);
            bad++;
        }
    CHECK(bad == 0, "failure after split: %ld particles changed\n", bad);
    struct SubTrack * after = fetch_all_tracks(3);
    int t;
    for(t = 0; t < 3; t++)
        CHECK(after[t].TrackId == before[t].TrackId && after[t].Nbound == before[t].Nbound && after[t].NboundExclusive == before[t].NboundExclusive
              && after[t].Mbound == before[t].Mbound && after[t].NestedParentTrackId == before[t].NestedParentTrackId
              && after[t].CentralTrackId == before[t].CentralTrackId && after[t].Rank == before[t].Rank && after[t].Depth == before[t].Depth
              && after[t].HostGroup == before[t].HostGroup && after[t].SinkTrackId == before[t].SinkTrackId && after[t].Flags == before[t].Flags
              && after[t].SnapDeath == before[t].SnapDeath && after[t].MostBoundID == before[t].MostBoundID
              && after[t].DescendantTrackId == before[t].DescendantTrackId && subhalo_get_table()->NTotal == 3,
              "failure after split: record %d changed\n", t);
    free(after);
    free(post);
    free(pre.r);
    free(before);
    syn_teardown();
}

/* ---- test 11: a voided claim across three rounds, the central promotion and J2 ----
 * A (0) the previous central: 1000 cold and 1000 hot particles; B (1, its satellite, 1500 cold
 * particles 4 kpc/h away: too light at the previous call to compete for the centre, the most
 * massive survivor now); C (2, B's satellite, 150 cold particles at A's centre). All three in
 * place: C in round 0, B in round 1, A in round 2. P: 40 tails of C at A's centre moving fast:
 * C claims them (relaxed source), B passes them on without claiming (C's claim is alive), C sinks
 * into A in round 2 and its claim becomes void: the accepted rule gives them to A, not back to B.
 * Q: 40 previous tails of B at A's centre at rest: B claims them in round 1, A binds them in round
 * 2; after the promotion of B to the central, A is B's satellite and no longer an ancestor of B:
 * the claim crosses the final tree and goes to the binding owner A (J2). */
#define VC_AH 1000
#define VC_B0 2000
#define VC_C0 3500
#define VC_P0 3650
#define VC_Q0 3690
#define VC_END 3730

static void
vc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 4 / atime, 500, 500};
    /* B fast: the knots at A's centre are not bound to it */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    (void) ud;
    if(id >= VC_END)
        return;
    if(id < VC_B0) {
        syn_ball(id, 121, cA, 2.0, zero, id < VC_AH ? 15 : 400, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < VC_C0) {
        syn_ball(id, 122, cB, 1.5, vB, 15, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - VC_B0;
    }
    else if(id < VC_P0) {
        syn_ball(id, 123, cA, 0.3, zero, 3, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - VC_C0;
    }
    else if(id < VC_Q0) {
        /* moving at 90 km/s through A's centre against B: bound to neither C nor B */
        const double vP[3] = {0, -90, 0};
        syn_ball(id, 124, cA, 0.5, vP, 3, atime, s);
        s->tag = 2;
        s->bound = -1;
        s->rank = 150 + id - VC_P0;
    }
    else {
        syn_ball(id, 126, cA, 0.5, zero, 3, atime, s);
        s->tag = 1;
        s->bound = -1;
        s->rank = 1500 + id - VC_Q0;
    }
}

static void
test_collective_voided_claim(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = 100;
    syn_setup(&CP, VC_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 1500, 0.15, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 150, 0.015, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(vc_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "voided claim: %ld groups\n", ng);
    struct Outcome o;
    outcome_take(&o);
    syn_teardown();
    const struct SubTrack * A = &o.rec[0], * B = &o.rec[1], * C = &o.rec[2];
    message(0, "voided claim: %d rounds (%ld %ld %ld), %ld sunk, %ld J2; A Nbound %d Rank %d parent %ld; B Nbound %d Rank %d central %ld; "
               "C Nbound %d Sink %ld\n", o.st.NRounds, o.st.NCandsRound[0], o.st.NCandsRound[1], o.st.NCandsRound[2], o.st.NSunk, o.st.NJ2,
            A->Nbound, A->Rank, A->NestedParentTrackId, B->Nbound, B->Rank, B->CentralTrackId, C->Nbound, C->SinkTrackId);
    CHECK(o.st.NRounds == 3 && o.st.NCandsRound[0] == 1 && o.st.NCandsRound[1] == 1 && o.st.NCandsRound[2] == 1,
          "voided claim: %d rounds\n", o.st.NRounds);
    CHECK(C->Nbound == 0 && C->SinkTrackId == 0 && C->SnapSink == 1, "voided claim: C should sink into A\n");
    /* the promotion: B the central, A its satellite */
    CHECK(B->Rank == 0 && B->NestedParentTrackId == -1 && B->CentralTrackId == 1 && A->Rank == 1 && A->NestedParentTrackId == 1
          && A->CentralTrackId == 1 && A->Depth == 1 && (B->Flags & SUBHALO_FLAG_PLANCENTRAL) && !(A->Flags & SUBHALO_FLAG_PLANCENTRAL),
          "voided claim: promotion (B Rank %d, A Rank %d parent %ld, Flags %d %d)\n", B->Rank, A->Rank, A->NestedParentTrackId, B->Flags, A->Flags);
    int64_t i, nP = 0, nPA = 0, nPB = 0, nQ = 0, nQA = 0;
    for(i = 0; i < o.n; i++) {
        const MyIDType id = o.r[i].ID;
        if(id >= VC_P0 && id < VC_Q0) {
            nP++;
            if(o.r[i].TrackId == 0) nPA++;
            if(o.r[i].TrackId == 1) nPB++;
        }
        if(id >= VC_Q0 && id < VC_END) {
            nQ++;
            if(o.r[i].TrackId == 0 && o.r[i].BoundTrackId == 0) nQA++;
        }
    }
    message(0, "voided claim: C's former claims %ld to A, %ld to B of %ld; B's claims bound by A %ld of %ld given to A (J2)\n", nPA, nPB, nP,
            nQA, nQ);
    CHECK(nPA == nP && nPB == 0, "voided claim: the void claims went %ld to A, %ld to B (no retroactive B claim)\n", nPA, nPB);
    CHECK(nQA >= nQ - 2 && o.st.NJ2 >= nQ - 2, "voided claim: %ld of %ld J2 claims given to A (%ld counted)\n", nQA, nQ, o.st.NJ2);
    CHECK(p2_check_tags(o.r, o.n, o.rec, o.nt, "voided claim") == 0, "voided claim: tag invariants\n");
    outcome_free(&o);
}

/* ---- test 12: a long in-place chain and groups at unequal depths in one call ----
 * Group 1: A0 > A1 > A2 > A3 (tracks 0-3, depths 0-3); group 2: B0 > B1 (4, 5). Everything in
 * place: four rounds (A3; A2; A1 and B1; A0 and B0), against the serial run. */
#define LC_N0 2000
#define LC_N1 600
#define LC_N2 300
#define LC_N3 150
#define LC_M0 1500
#define LC_M1 400
static const int64_t LcStart[7] = {0, LC_N0, LC_N0 + LC_N1, LC_N0 + LC_N1 + LC_N2, LC_N0 + LC_N1 + LC_N2 + LC_N3,
                                   LC_N0 + LC_N1 + LC_N2 + LC_N3 + LC_M0, LC_N0 + LC_N1 + LC_N2 + LC_N3 + LC_M0 + LC_M1};

static void
lc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double c1[3] = {500, 500, 500}, c2[3] = {750, 500, 500};
    const double R[6] = {3.0, 1.0, 0.6, 0.4, 2.5, 0.8}, sig[6] = {20, 8, 6, 4, 18, 7};
    const double off[6][3] = {{0, 0, 0}, {5, 0, 0}, {-5, 0, 0}, {0, 5, 0}, {0, 0, 0}, {5, 0, 0}};
    const double vel[6][3] = {{0, 0, 0}, {0, 40, 0}, {0, -40, 0}, {40, 0, 0}, {0, 0, 0}, {0, 40, 0}};
    (void) ud;
    int t;
    for(t = 0; t < 6; t++)
        if((int64_t) id < LcStart[t + 1])
            break;
    if(t >= 6)
        return;
    const double * c = t < 4 ? c1 : c2;
    double cen[3];
    int d;
    for(d = 0; d < 3; d++)
        cen[d] = c[d] + off[t][d] / atime;
    syn_ball(id, 131 + t, cen, R[t], vel[t], sig[t], atime, s);
    s->tag = s->bound = t;
    s->rank = id - LcStart[t];
}

static void
lc_setup(Cosmology * CP, struct SubhaloParams sp)
{
    syn_setup(CP, LcStart[6] + SYN_NFIELD, sp);
    struct SubTrack recs[6];
    recs[0] = syn_rec(0, LC_N0, LC_N0 * SYN_M, -1, 0, 0, 0);
    recs[1] = syn_rec(1, LC_N1, LC_N1 * SYN_M, 0, 0, 1, 1);
    recs[2] = syn_rec(2, LC_N2, LC_N2 * SYN_M, 1, 0, 2, 2);
    recs[3] = syn_rec(3, LC_N3, LC_N3 * SYN_M, 2, 0, 3, 3);
    recs[4] = syn_rec(4, LC_M0, LC_M0 * SYN_M, -1, 4, 0, 0);
    recs[5] = syn_rec(5, LC_M1, LC_M1 * SYN_M, 4, 4, 1, 1);
    recs[4].HostGroup = recs[5].HostGroup = 2;
    syn_records(recs, 6);
    syn_place(lc_particle, NULL, 0.55, 1);
}

static void
lc_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    lc_setup(&CP, sp);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "long chain: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_long_chain(void ** state)
{
    struct Outcome a, b;
    lc_run(0, &a);
    lc_run(100, &b);
    message(0, "long chain: %d rounds, candidates per round %ld %ld %ld %ld, walks %d %d %d %d\n", b.st.NRounds, b.st.NCandsRound[0],
            b.st.NCandsRound[1], b.st.NCandsRound[2], b.st.NCandsRound[3], b.st.Iterations[0], b.st.Iterations[1], b.st.Iterations[2],
            b.st.Iterations[3]);
    CHECK(b.st.NRounds == 4 && b.st.NCands == 6 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1 && b.st.NCandsRound[2] == 2
          && b.st.NCandsRound[3] == 2, "long chain: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    const int depth[6] = {0, 1, 2, 3, 0, 1};
    const int64_t parent[6] = {-1, 0, 1, 2, -1, 4};
    int t;
    for(t = 0; t < 6; t++)
        CHECK(b.rec[t].Nbound > 0 && b.rec[t].Depth == depth[t] && b.rec[t].NestedParentTrackId == parent[t]
              && (b.rec[t].Flags & SUBHALO_FLAG_INPLACE) && (!(depth[t] == 0) || (b.rec[t].Flags & SUBHALO_FLAG_PLANCENTRAL)),
              "long chain: track %d Nbound %d Depth %d parent %ld Flags %d\n", t, b.rec[t].Nbound, b.rec[t].Depth,
              b.rec[t].NestedParentTrackId, b.rec[t].Flags);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "long chain") == 0, "long chain: tag invariants\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "long chain");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- test 13: the admission of the collective stage (audit Q1) ----
 * The long-chain fixture without forcing, the free memory squeezed (a top block allocated after
 * the FOF) so that no serial key fits and every key is promoted in place, while the collective
 * stage itself (its walk reserve alone) does not fit either: the call fails before the exchange
 * with the domain, GrNr and the tags untouched. The next call with the memory back is serial. */
struct LmCtx {
    void * block;
    double target;
    struct tagrec * pre;
    int64_t npre;
};

static void
lm_pre(void * ud)
{
    struct LmCtx * c = (struct LmCtx *) ud;
    c->pre = gather_tags(&c->npre);
    const double fr = (double) mymalloc_freebytes();
    c->block = fr > c->target ? mymalloc2("TestSqueeze", (size_t) (fr - c->target)) : NULL;
    message(0, "admission: free %g MB squeezed to %g MB\n", fr / 1048576., (double) mymalloc_freebytes() / 1048576.);
}

static void
lm_post(void * ud)
{
    struct LmCtx * c = (struct LmCtx *) ud;
    if(c->block)
        myfree(c->block);
    c->block = NULL;
}

static void
test_collective_admission(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloLookupBatchBytes = 1024 * 1024;
    lc_setup(&CP, sp);
    struct LmCtx c = {NULL, 0.5 * (double) subhalo_collective_walk_bytes(), NULL, 0};
    int64_t ng;
    int ret = run_finder_call_ex(1, &CP, &ng, lm_pre, lm_post, &c);
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    const struct SubhaloCollAdmit * ad = subhalo_plan_last_admission();
    message(0, "admission: ret %d; plan %ld serial, %ld in place, %ld candidates; admitted %d, failed %d on rank %d at the %s, "
               "this rank needs %g MB (%g held + stages %g %g %g %g %g %g %g %g), %g MB available\n", ret, ps->NSerial, ps->NOversized, ps->NCands,
            ps->Admitted, ad->failed, ad->failrank, ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-", ad->need / 1048576.,
            ad->persist / 1048576., ad->stage[0] / 1048576., ad->stage[1] / 1048576., ad->stage[2] / 1048576., ad->stage[3] / 1048576.,
            ad->stage[4] / 1048576., ad->stage[5] / 1048576., ad->stage[6] / 1048576., ad->stage[7] / 1048576., ad->avail / 1048576.);
    CHECK(ret == 0 && !subhalo_last_call_complete(), "admission: the squeezed call returned %d\n", ret);
    CHECK(ps->NSerial == 0 && ps->NOversized >= 2 && ps->NCands == 6 && !ps->Admitted && ad->failed
          && ps->NEncFof == 0 && ps->NEncField == 0, "admission: every key promoted and the admission failed\n");
    CHECK(ad->failstage == SUBHALO_COLL_TREE, "admission: the largest stage is the %s\n", SubhaloCollStageName[ad->failstage]);
    int64_t n, i, bad = 0;
    struct tagrec * post = gather_tags(&n);
    for(i = 0; i < n; i++)
        if(post[i].ID != c.pre[i].ID || post[i].GrNr != c.pre[i].GrNr || post[i].TrackId != c.pre[i].TrackId
           || post[i].BindRank != c.pre[i].BindRank || post[i].BoundTrackId != c.pre[i].BoundTrackId)
            bad++;
    CHECK(n == c.npre && bad == 0, "admission: %ld particles changed by the failed call\n", bad);
    free(post);
    free(c.pre);
    /* the memory back: the same call is serial and completes */
    ret = run_finder_call(1, &CP, &ng);
    ps = subhalo_plan_last_summary();
    CHECK(ret == 1 && subhalo_last_call_complete() && ps->NSerial == 2 && ps->NCands == 0, "admission: the next call returned %d (%ld serial)\n",
          ret, ps->NSerial);
    syn_teardown();
}

/* ---- test 14: the int32 and MPI byte-count guards (audit Q4), without allocating ---- */
static void
test_collective_guards(void ** state)
{
    CHECK(subhalo_count_fits_i32((double) INT32_MAX) && !subhalo_count_fits_i32((double) INT32_MAX + 1.) && subhalo_count_fits_i32(0)
          && !subhalo_count_fits_i32(-1.) && !subhalo_count_fits_i32(1e300), "guards: counts\n");
    /* a rank r stands for r + 1 particles: INT32_MAX is out */
    CHECK(subhalo_rank_fits_i32(INT32_MAX - 1L) && !subhalo_rank_fits_i32(INT32_MAX) && !subhalo_rank_fits_i32(-1)
          && subhalo_rank_fits_i32(0), "guards: ranks\n");
    /* 24-byte records: 89478485 fit an MPI int byte count, 89478486 do not (the audit's boundary) */
    CHECK(subhalo_mpi_bytes_fit(89478485, 24) && !subhalo_mpi_bytes_fit(89478486, 24) && subhalo_mpi_bytes_fit(2147483647L, 1)
          && !subhalo_mpi_bytes_fit(2147483648L, 1) && !subhalo_mpi_bytes_fit(-1, 8) && subhalo_mpi_bytes_fit(0, 56),
          "guards: MPI byte counts\n");
    /* the admission estimate: the persistent scratch is 25 bytes per particle, every stage at least
     * its records, the largest stage chosen */
    struct SubhaloCollAdmit a;
    memset(&a, 0, sizeof(a));
    subhalo_coll_estimate(&a, 1000000, 2000000, 100000, 50000, 2, 2 * 20 * 4, 10, 100, 100, 4097, 3585, 1000, 0.7, 1);
    double mx = 0;
    int s;
    for(s = 0; s < SUBHALO_COLL_NSTAGE; s++)
        if(a.stage[s] > mx)
            mx = a.stage[s];
    const double persist = 25. * 1000001 + 11. * SUBHALO_INPLACE_CAND_BYTES + 3. * (sizeof(struct SubTrack) + 8.);
    CHECK(fabs(a.persist - persist) < 1 && a.need == a.persist + mx
          && a.stage[SUBHALO_COLL_RADIAL] >= 176. * 100000 && a.stage[SUBHALO_COLL_TREE] >= 0.7 * 1000000 * sizeof(struct NODE)
          /* Phase 5 audit R1: the properties keep 104 bytes per record and every candidate's scratch */
          && a.stage[SUBHALO_COLL_PROPS] >= 104. * 100000 + 11. * SUBHALO_INPLACE_CAND_PROPS_BYTES,
          "guards: estimate persist %g need %g\n", a.persist, a.need);
}

/* ---- test 15: the potential-only walk at the default opening angle against a direct sum ----
 * Two interpenetrating clumps (labels 0 and 1) straddling the periodic boundary at x = 0, a knot
 * of label-0 particles closer than the softening across the boundary, a field outside every
 * label: on 4 ranks the clumps span several domains (remote monopoles, mixed-label top nodes).
 * At theta = 0.45 the monopole-only geometric opening (theta^2 <= 0.33) bounds the relative
 * error of a node's contribution by about theta^2 times its quadrupole fraction; the sums are
 * required within 1e-2 per particle and 2e-3 in the mean (the measured values are logged), and
 * within 1e-3 / 2e-4 at theta = 0.15. */
#define WK_N0 1500
#define WK_N1 800
#define WK_NK 20
#define WK_END (WK_N0 + WK_N1 + WK_NK)

static int
wk_label(MyIDType id)
{
    if(id < WK_N0 + WK_N1 + WK_NK)
        return id >= WK_N0 && id < WK_N0 + WK_N1 ? 1 : 0;
    return -1;
}

static void
wk_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double c0[3] = {0, 500, 500}, c1[3] = {1.0, 501, 500}, ck[3] = {0, 500.2, 500};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= WK_END)
        return;
    if(id < WK_N0)
        syn_ball(id, 141, c0, 3.0, zero, 0, atime, s);
    else if(id < WK_N0 + WK_N1)
        syn_ball(id, 142, c1, 2.0, zero, 0, atime, s);
    else
        syn_ball(id, 143, ck, 0.05, zero, 0, atime, s);
    /* masses within a factor 2 */
    s->mass = SYN_M * (1 + urand(id, 144, 0));
}

struct WkPart {
    MyIDType ID;
    double Pos[3];
    double Mass;
    int32_t label, type;
};

static void
wk_check(double theta, double maxtol, double meantol)
{
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    const int64_t n = PartManager->NumPart;
    int32_t * label = malloc(sizeof(int32_t) * (n + 1));
    double * out = malloc(sizeof(double) * (n + 1));
    int64_t i, j;
    int nloc = 0;
    for(i = 0; i < n; i++) {
        label[i] = wk_label(P[i].ID);
        out[i] = -1;
        if(label[i] >= 0)
            nloc++;
    }
    const double h = FORCE_SOFTENING();
    const int PotentialTypes = 1 + 2 + 16 + 32;
    subhalo_collective_test_walk(&ddecomp, label, 2, theta, h, PotentialTypes, out);
    /* the direct sum over every labelled particle, nearest images */
    const int NTask = get_ntask();
    struct WkPart * loc = malloc(sizeof(struct WkPart) * (nloc + 1));
    int k = 0;
    for(i = 0; i < n; i++)
        if(label[i] >= 0) {
            loc[k].ID = P[i].ID;
            int d;
            for(d = 0; d < 3; d++)
                loc[k].Pos[d] = P[i].Pos[d];
            loc[k].Mass = P[i].Mass;
            loc[k].label = label[i];
            loc[k].type = P[i].Type;
            k++;
        }
    int nb = nloc * sizeof(struct WkPart);
    int * counts = malloc(NTask * sizeof(int)), * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    struct WkPart * all = malloc(tot + sizeof(struct WkPart));
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    const int64_t nall = tot / sizeof(struct WkPart);
    double maxerr = 0, sumerr = 0, nerr = 0;
    int64_t nsoft = 0, nwrap = 0;
    for(i = 0; i < n; i++) {
        if(label[i] < 0)
            continue;
        double exact = 0;
        for(j = 0; j < nall; j++) {
            if(all[j].label != label[i] || all[j].ID == P[i].ID || !((PotentialTypes >> all[j].type) & 1))
                continue;
            double r2 = 0, raw = 0;
            int d;
            for(d = 0; d < 3; d++) {
                const double x = NEAREST(all[j].Pos[d] - P[i].Pos[d], BOXSIZE);
                r2 += x * x;
                raw += fabs(all[j].Pos[d] - P[i].Pos[d]);
            }
            const double rr = sqrt(r2);
            if(rr < h)
                nsoft++;
            if(raw > 0.5 * BOXSIZE)
                nwrap++;
            exact += all[j].Mass * grav_spline_potential(rr, h);
        }
        const double err = fabs(out[i] - exact) / exact;
        if(err > maxerr)
            maxerr = err;
        sumerr += err;
        nerr++;
    }
    double g[3] = {maxerr, sumerr, nerr}, gs[3];
    MPI_Allreduce(&g[0], &gs[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&g[1], &gs[1], 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int64_t pairs[2] = {nsoft, nwrap}, gp[2];
    MPI_Allreduce(pairs, gp, 2, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "walk theta %g: %g queries, largest relative error %.3e, mean %.3e; %ld softened and %ld periodic-image pairs in the direct sum\n",
            theta, gs[2], gs[0], gs[1] / gs[2], gp[0], gp[1]);
    CHECK(gs[2] == WK_END && gs[0] <= maxtol && gs[1] / gs[2] <= meantol && gp[0] > 0 && gp[1] > 0,
          "walk theta %g: %g queries, largest relative error %g, mean %g\n", theta, gs[2], gs[0], gs[1] / gs[2]);
    free(all);
    free(displ);
    free(counts);
    free(loc);
    free(out);
    free(label);
    domain_free(&ddecomp);
}

static void
test_collective_walk_theta(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    syn_setup(&CP, WK_END + SYN_NFIELD, sp);
    syn_place(wk_particle, NULL, 0.55, 0);
    wk_check(0.45, 1e-2, 2e-3);
    wk_check(0.15, 1e-3, 2e-4);
    syn_teardown();
}

/* ---- test 16: the deferred recipes of the two modes on identical inputs (Phase 5 audit R4) ----
 * Seven power-law balls (density ~ r^-2, every particle bound, one across the periodic boundary,
 * one below the NFW fit's 100 particles, one of 137 whose equal-mass virial crossing is at kc =
 * 100, the fit's boundary of decision 47.1: Rs defined with the kc particles inside the crossing,
 * -1 on nfw.c's literal kc - 1), large bulk velocities, prescribed cores (subhalo_core of
 * the ID order) and thresholds with crossings inside every ball. The serial recipes
 * (subhalo_props_rockstar and subhalo_props_deferred on rank 0, which holds every particle) against
 * the distributed ones (round_props through subhalo_collective_test_props) with the particles on
 * rank 0 only, round-robin over the 4 ranks, and unevenly split (1:2:4:6): with equal masses
 * the four Phase 5 columns and the virial mass are bitwise identical in every layout (the
 * compensated cumulative masses are the correctly rounded k m in both modes); with random unequal
 * masses they agree to 1e-6; with masses 1 and 2^-50 (one in 50 heavy, inside the light ones:
 * exact multiples of 2^-50, whose plain sums lose the light masses in an order-dependent way, so
 * that the plain prefix sums of the uneven layout differ from the others) bitwise again, the
 * virial masses included. Also the compensated sum itself survives -ffast-math. */
#define PI_NL 7
static const int64_t PiN[PI_NL] = {3000, 1200, 450, 260, 99, 2500, 137};
#define PI_CORESIZE 20

static int64_t
pi_total(void)
{
    int64_t t = 0;
    int l;
    for(l = 0; l < PI_NL; l++)
        t += PiN[l];
    return t;
}

static int
pi_label(MyIDType id, int64_t * j)
{
    int64_t s = 0;
    int l;
    *j = -1;
    for(l = 0; l < PI_NL; l++) {
        if((int64_t) id < s + PiN[l]) {
            *j = (int64_t) id - s;
            return l;
        }
        s += PiN[l];
    }
    return -1;
}

/* Particle id of ball l: r = R u (M(< r) ~ r), R = 20 (N / 3000)^(1/3), mean density inside R the
 * same for every ball; velocities bulk + 200 x uniform in a ball. */
enum { PI_EQUAL = 0, PI_UNEQUAL, PI_WIDE };
static const char * const PiMode[3] = {"equal", "unequal", "wide"};

static void
pi_particle(MyIDType id, int mode, double pos[3], double vel[3], double * mass)
{
    int64_t j;
    const int l = pi_label(id, &j);
    const double cen[PI_NL][3] = {{500, 500, 500}, {0.5, 300, 700}, {800, 150, 220}, {250, 820, 400}, {650, 650, 90}, {120, 480, 930}, {380, 100, 600}};
    const double bulk[3] = {3000. - 1000 * l, -2000. + 700 * l, 50. * l};
    const double R = 20. * cbrt((double) PiN[l] / 3000.);
    double u[3];
    const double zero[3] = {0, 0, 0};
    rand_in_ball(id, 161, 1.0, zero, u);
    const double un = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    /* wide masses: the heavy particles (j % 50 == 0) in the inner 5 % of the radius, so that a
     * plain running sum drops the light masses beyond them, but not a later rank's partial sum */
    const double r = (mode == PI_WIDE && j % 50 == 0 ? 0.05 : 1.0) * R * urand(id, 162, 0);
    int d;
    for(d = 0; d < 3; d++) {
        double x = cen[l][d] + (un > 0 ? r * u[d] / un : 0);
        while(x >= BOXSIZE) x -= BOXSIZE;
        while(x < 0) x += BOXSIZE;
        pos[d] = x;
    }
    double v[3];
    rand_in_ball(id, 163, 200., zero, v);
    for(d = 0; d < 3; d++)
        vel[d] = bulk[d] + v[d];
    if(mode == PI_EQUAL)
        *mass = SYN_M;
    else if(mode == PI_UNEQUAL)
        *mass = SYN_M * (0.5 + urand(id, 164, 0));
    else
        *mass = j % 50 == 0 ? 1.0 : ldexp(1.0, -50);
}

/* layout 3: rank 0 holds PiSplit particles, so that the radial order (sorted by label and radius,
 * the ranks' counts kept) changes rank between kc - 1 and kc of the last ball */
static int64_t PiSplit = 0;

/* layout 0: every particle on rank 0; 1: round robin; 2: 1/13, 2/13, 4/13 and 6/13 on ranks 0-3
 * (the sorted records then change rank inside the inner part of the first balls); 3: above */
static int
pi_rank(MyIDType id, int layout, int NTask)
{
    if(layout == 0)
        return 0;
    if(layout == 1)
        return (int) (id % NTask);
    if(layout == 3)
        return (int64_t) id < PiSplit || NTask == 1 ? 0 : 1 + (int) (id % (NTask - 1));
    const int k = (int) (splitmix64(id ^ 0x5151) % 13);
    return (k < 1 ? 0 : (k < 3 ? 1 : (k < 7 ? 2 : 3))) % NTask;
}

static void
pi_place(int layout, int mode)
{
    const int NTask = get_ntask(), ThisTask = get_thistask();
    const int64_t n = pi_total();
    setup_pman(n + 1, BOXSIZE);
    MyIDType id;
    for(id = 0; id < (MyIDType) n; id++) {
        if(pi_rank(id, layout, NTask) != ThisTask)
            continue;
        double pos[3], vel[3], m;
        pi_particle(id, mode, pos, vel, &m);
        add_dm_particle(id, pos);
        struct particle_data * p = &P[PartManager->NumPart - 1];
        int d;
        for(d = 0; d < 3; d++)
            p->Vel[d] = vel[d];
        p->Mass = m;
    }
}

static void
pi_distributed(const double * core, const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO], double * dfr, double * mvir)
{
    int32_t * label = (int32_t *) malloc(sizeof(int32_t) * (PartManager->NumPart + 1));
    int64_t i, j;
    for(i = 0; i < PartManager->NumPart; i++)
        label[i] = pi_label(P[i].ID, &j);
    subhalo_collective_test_props(label, PI_NL, core, up, rho, dfr, mvir);
    free(label);
}

struct PiRad { double r, m; MyIDType id; };

static int
pi_rad_cmp(const void * a, const void * b)
{
    const struct PiRad * x = (const struct PiRad *) a, * y = (const struct PiRad *) b;
    if(x->r != y->r)
        return x->r < y->r ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

/* The outermost virial crossing kc of the n particles Index about core (the recipe's radial
 * order and compensated M(<= r)); *Mkc the mass at it. */
static int64_t
pi_kc(const int64_t * Index, int64_t n, const double * core, double rfloor, double rho0, double * Mkc)
{
    struct PiRad * pr = (struct PiRad *) malloc(sizeof(struct PiRad) * (n + 1));
    int64_t k, kc = -1;
    int d;
    for(k = 0; k < n; k++) {
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[Index[k]].Pos[d] - core[d], BOXSIZE);
            r2 += dx * dx;
        }
        pr[k].r = sqrt(r2) > rfloor ? sqrt(r2) : rfloor;
        pr[k].m = P[Index[k]].Mass;
        pr[k].id = P[Index[k]].ID;
    }
    qsort(pr, n, sizeof(struct PiRad), pi_rad_cmp);
    double Mk[2] = {0, 0};
    *Mkc = 0;
    for(k = 0; k < n; k++) {
        subhalo_ksum_add(Mk, pr[k].m);
        const double M = subhalo_ksum_value(Mk);
        if(M / (4.0 / 3.0 * M_PI * pr[k].r * pr[k].r * pr[k].r) > rho0) {
            kc = k;
            *Mkc = M;
        }
    }
    free(pr);
    return kc;
}

static void
pi_run(int mode)
{
    const int exact = mode != PI_UNEQUAL;
    const int ThisTask = get_thistask();
    struct SubhaloUnbindParams up;
    memset(&up, 0, sizeof(up));
    up.atime = 0.7;
    up.h = 2.8 * 0.3;
    up.BoxSize = BOXSIZE;
    up.TracerTypes = 2;
    up.PotentialTypes = 2;
    up.G = 43007.1;
    /* the thresholds: the mean density inside R of ball 0 times 1.5, 2.5, 6, 15, 60 (crossings at
     * R / 1.2, R / 1.6, R / 2.4, R / 3.9, R / 7.7 for equal masses) */
    double Mball = 0;
    {
        MyIDType id;
        for(id = 0; id < (MyIDType) PiN[0]; id++) {
            double pos[3], vel[3], m;
            pi_particle(id, mode, pos, vel, &m);
            Mball += m;
        }
    }
    const double rhoR = 3 * Mball / (4 * M_PI * 20. * 20. * 20.);
    const double f[SUBHALO_NSO] = {1.5, 2.5, 6, 15, 60};
    double rho[SUBHALO_NSO];
    int s, l, d, k;
    for(s = 0; s < SUBHALO_NSO; s++)
        rho[s] = f[s] * rhoR;
    /* the serial recipes on rank 0, which holds everything (layout 0) */
    double core[6 * PI_NL];
    struct SubTrack ser[PI_NL];
    memset(core, 0, sizeof(core));
    memset(ser, 0, sizeof(ser));
    int64_t kcsplit = -1;
    double Mkcsplit = 0;
    pi_place(0, mode);
    if(ThisTask == 0) {
        int64_t * Index = (int64_t *) mymalloc("PiIndex", sizeof(int64_t) * (PartManager->NumPart + 1));
        int64_t start = 0, i;
        for(l = 0; l < PI_NL; l++) {
            int64_t n = 0;
            for(i = 0; i < PartManager->NumPart; i++) {
                int64_t j;
                if(pi_label(P[i].ID, &j) == l)
                    Index[start + n++] = i;
            }
            struct SubCandidate c;
            memset(&c, 0, sizeof(c));
            c.Index = Index + start;
            c.N = c.Nbound = n;
            double sx, sv;
            subhalo_core(&c, &up, PI_CORESIZE, core + 6 * l, core + 6 * l + 3, &sx, &sv);
            if(l == PI_NL - 1) {
                /* layout 3 splits the last ball between kc - 1 and kc */
                double Mkc;
                const int64_t kc = pi_kc(c.Index, n, core + 6 * l, up.h / 2.8, rho[0], &Mkc);
                assert_true(kc > 0);
                PiSplit = start + kc;
                kcsplit = kc;
                Mkcsplit = Mkc;
            }
            for(d = 0; d < 3; d++)
                c.RefPos[d] = core[6 * l + d];
            struct SubhaloArena arena;
            arena.size = SUBHALO_PROPS_BYTES_PER_PART * n + 64;
            arena.used = 0;
            arena.base = (char *) mymalloc("PiArena", arena.size);
            const double Offset[3] = {0, 0, 0};
            subhalo_track_clear(&ser[l], l);
            subhalo_props_rockstar(&ser[l], &c, &up, rho, PI_CORESIZE, Offset, &arena);
            struct SubTrack r2 = ser[l];
            subhalo_props_deferred(&r2, &c, &up, rho, PI_CORESIZE, &arena);
            assert_true(r2.Rs == ser[l].Rs && r2.Vrms == ser[l].Vrms);
            for(d = 0; d < 3; d++)
                assert_true(r2.BulkVel[d] == ser[l].BulkVel[d] && r2.RockstarCoreVel[d] == ser[l].RockstarCoreVel[d]);
            myfree(arena.base);
            start += n;
        }
        myfree(Index);
    }
    MPI_Bcast(core, 6 * PI_NL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ser, sizeof(ser), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PiSplit, 1, MPI_INT64, 0, MPI_COMM_WORLD);
    /* the oracle's kc is the recipe's: its mass is MBoundSO[vir] */
    message(0, "props identity (%s masses): the last ball's kc %ld (M %.9g, serial MBoundSO[vir] %.9g), layout 3 splits at %ld\n",
            PiMode[mode], (long) kcsplit, Mkcsplit, ser[PI_NL - 1].MBoundSO[0], (long) PiSplit);
    if(ThisTask == 0)
        assert_true((float) Mkcsplit == ser[PI_NL - 1].MBoundSO[0]);
    /* the distributed recipes in four layouts */
    double dfr[4][8 * PI_NL], mvir[4][PI_NL];
    int lay;
    for(lay = 0; lay < 4; lay++) {
        if(lay > 0)
            pi_place(lay, mode);
        message(0, "props identity (%s masses), layout %d: rank 0 holds %ld particles\n", PiMode[mode], lay,
                (long) PartManager->NumPart);
        if(lay == 3 && ThisTask == 0 && get_ntask() > 1)
            assert_true(PartManager->NumPart == PiSplit);
        pi_distributed(core, &up, rho, dfr[lay], mvir[lay]);
        teardown_pman();
    }
    int64_t bad = 0;
    double worst = 0;
    for(l = 0; l < PI_NL; l++) {
        const struct SubTrack * x = &ser[l];
        const float sv[8] = {x->Rs, x->RockstarCoreVel[0], x->RockstarCoreVel[1], x->RockstarCoreVel[2],
                             x->BulkVel[0], x->BulkVel[1], x->BulkVel[2], x->Vrms};
        message(0, "props identity (%s masses), ball %d (%ld particles): serial Rs %.9g Vrms %.9g BulkVel (%.9g %.9g %.9g) "
                   "RockstarCoreVel (%.9g %.9g %.9g) MBoundSO[vir] %.9g\n", PiMode[mode], l, (long) PiN[l],
                sv[0], sv[7], sv[4], sv[5], sv[6], sv[1], sv[2], sv[3], x->MBoundSO[0]);
        /* every ball crosses; the fit runs in the balls of 200 or more particles (aperture about 0.8 N), never below 100 */
        LOCAL_CHECK(bad, mode == PI_WIDE || (x->Vrms > 0 && x->MBoundSO[0] > 0 && (PiN[l] < 100 ? x->Rs == -1 : (PiN[l] < 200 || x->Rs > 0))),
                    "props identity: ball %d Vrms %g MBoundSO %g Rs %g\n", l, x->Vrms, x->MBoundSO[0], x->Rs);
        /* the fit's aperture (decision 47.1): the kc particles inside the crossing particle, Rs
         * defined iff kc >= 100; with equal masses kc = MBoundSO[vir] / m - 1 */
        if(mode == PI_EQUAL) {
            const int64_t kcv = llround(x->MBoundSO[0] / SYN_M) - 1;
            message(0, "props identity: ball %d kc %ld\n", l, (long) kcv);
            LOCAL_CHECK(bad, (kcv >= SUBHALO_NFW_MINPART) == (x->Rs > 0), "props identity: ball %d kc %ld Rs %g\n", l, (long) kcv, x->Rs);
            /* the last ball sits on the boundary: kc = 100 fits (nfw.c's literal aperture would not) */
            LOCAL_CHECK(bad, l != PI_NL - 1 || (kcv == SUBHALO_NFW_MINPART && x->Rs > 0),
                        "props identity: boundary ball %d kc %ld (expected %d) Rs %g\n", l, (long) kcv, SUBHALO_NFW_MINPART, x->Rs);
        }
        for(lay = 0; lay < 4; lay++) {
            const double * o = dfr[lay] + 8 * l;
            for(k = 0; k < 8; k++) {
                const float v = (float) o[k];
                if(exact)
                    LOCAL_CHECK(bad, v == sv[k], "props identity (%s masses): ball %d layout %d column %d serial %.9g distributed %.9g\n",
                                PiMode[mode], l, lay, k, sv[k], v);
                else {
                    const double scale = k == 0 ? fabs(sv[0]) : (k == 7 ? sv[7] : fabs(sv[k]) + sv[7]);
                    const double dd = fabs(v - sv[k]) / (scale > 0 ? scale : 1);
                    if(dd > worst)
                        worst = dd;
                    LOCAL_CHECK(bad, dd <= 1e-6, "props identity (unequal masses): ball %d layout %d column %d serial %.9g distributed %.9g\n",
                                l, lay, k, sv[k], v);
                }
            }
            if(exact)
                LOCAL_CHECK(bad, (float) mvir[lay][l] == x->MBoundSO[0] && mvir[lay][l] == mvir[0][l],
                            "props identity: ball %d layout %d Mvir %.17g / %.17g, serial %.9g\n", l, lay, mvir[lay][l], mvir[0][l],
                            x->MBoundSO[0]);
        }
    }
    if(!exact)
        message(0, "props identity (unequal masses): largest relative difference %.3g\n", worst);
    assert_all_true(bad == 0);
}

static void
test_collective_props_identity(void ** state)
{
    /* the compensated sum under the build's -ffast-math: 1 + 10 x 1e-16 - 1 */
    double ks[2] = {0, 0};
    int i;
    subhalo_ksum_add(ks, 1.0);
    for(i = 0; i < 10; i++)
        subhalo_ksum_add(ks, 1e-16);
    subhalo_ksum_add(ks, -1.0);
    CHECK(fabs(subhalo_ksum_value(ks) - 1e-15) < 1e-30, "compensated sum: %.17g\n", subhalo_ksum_value(ks));
    common_physics_setup();
    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    pi_run(PI_EQUAL);
    pi_run(PI_UNEQUAL);
    pi_run(PI_WIDE);
}

/* ---- test 17: many in-place candidates in one round: the properties stage (Phase 5 audit R1) ----
 * 2304 isolated 12-particle groups, each with one resolved track (SubhaloMinNumPart 10), all
 * forced in place (SubhaloCollectiveMinPart 10): one round of 2304 candidates, whose per-candidate property
 * scratch (the deferred recipes' reductions, the profile vectors: SUBHALO_INPLACE_CAND_PROPS_BYTES
 * each on every rank) is the largest stage. (a) Unsqueezed: admitted with the properties as the
 * largest stage, the call completes and the measured high water stays within the estimate. (b) The
 * same call with the free memory squeezed on every rank to halfway between the need without the
 * properties stage (the admission before the fix) and the need with it: the admission fails at
 * the properties stage, before the exchange, where the old estimate would have admitted it. Also
 * (audit R3, decision 47.5) the walltime children /Subhalo/Rounds/{Setup,Rounds,Tree,Walk,Tags,
 * Cleanup} add up to the summary's rounds time. */
#define ML_NX 16
#define ML_NYZ 12
#define ML_NG (ML_NX * ML_NYZ * ML_NYZ)
#define ML_NP 12

static void
ml_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    (void) ud;
    if((int64_t) id >= ML_NG * ML_NP)
        return;
    const int64_t g = (int64_t) id / ML_NP;
    /* x on multiples of 62.5 (31.25 from every lattice coordinate: no field particle links) */
    const double cen[3] = {62.5 * (g % ML_NX), 40. + 80. * ((g / ML_NX) % ML_NYZ), 40. + 80. * (g / (ML_NX * ML_NYZ))};
    const double zero[3] = {0, 0, 0};
    syn_ball(id, 171, cen, 0.5, zero, 2., atime, s);
    s->tag = s->bound = g;
    s->rank = (int32_t) ((int64_t) id - g * ML_NP);
}

static void
ml_setup(Cosmology * CP)
{
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloCollectiveMinPart = 10;
    sp.SubhaloMinNumPart = 10;
    sp.SubhaloLookupBatchBytes = 1024 * 1024;
    syn_setup(CP, ML_NG * ML_NP + SYN_NFIELD, sp);
    struct SubTrack * recs = (struct SubTrack *) malloc(sizeof(struct SubTrack) * ML_NG);
    int64_t g;
    for(g = 0; g < ML_NG; g++) {
        recs[g] = syn_rec(g, ML_NP, ML_NP * SYN_M, -1, g, 0, 0);
        recs[g].HostGroup = g + 1;
    }
    syn_records(recs, ML_NG);
    free(recs);
    syn_place(ml_particle, NULL, 0.55, 1);
}

struct MlCtx {
    double freepre;          /* free bytes at the pre hook */
    double target;           /* > 0: squeeze the free bytes to this */
    void * block;
};

static void
ml_pre(void * ud)
{
    struct MlCtx * c = (struct MlCtx *) ud;
    c->freepre = (double) mymalloc_freebytes();
    c->block = c->target > 0 && c->freepre > c->target ? mymalloc2("TestSqueeze", (size_t) (c->freepre - c->target)) : NULL;
}

static void
ml_post(void * ud)
{
    struct MlCtx * c = (struct MlCtx *) ud;
    if(c->block)
        myfree(c->block);
    c->block = NULL;
}

/* persist + the largest stage other than the properties: the admission's need before the fix */
static double
ml_need_without_props(const struct SubhaloCollAdmit * ad)
{
    double mx = 0;
    int s;
    for(s = 0; s < SUBHALO_COLL_NSTAGE; s++)
        if(s != SUBHALO_COLL_PROPS && ad->stage[s] > mx)
            mx = ad->stage[s];
    return ad->persist + mx;
}

/* The summed time of this rank's walltime clocks whose names start with prefix. */
static double
ml_clock_sum(const char * prefix)
{
    double t = 0;
    int i;
    for(i = 0; i < CT.N; i++)
        if(strncmp(CT.C[i].name, prefix, strlen(prefix)) == 0)
            t += CT.C[i].time;
    return t;
}

static void
test_collective_admission_props(void ** state)
{
    Cosmology CP;
    int64_t ng;
    struct MlCtx c = {0, 0, NULL};
    /* (a) */
    ml_setup(&CP);
    /* Phase 5 audit R3: the rounds stage's walltime children (this rank; the clock names carry
     * "@file:line"). walltime_init (in ml_setup) resets the table but not the times of its slots,
     * which new clocks would inherit from the earlier tests: cleared here */
    {
        int i;
        for(i = 0; i < CT.Nmax; i++)
            CT.C[i].time = CT.AC[i].time = 0;
    }
    double wt0 = ml_clock_sum("/Subhalo/Rounds/"), wcl = ml_clock_sum("/Subhalo/Rounds/Cleanup"), wtr = ml_clock_sum("/Subhalo/Rounds/Tree");
    double t0 = MPI_Wtime();
    int ret = run_finder_call_ex(1, &CP, &ng, ml_pre, ml_post, &c);
    const double wt1 = ml_clock_sum("/Subhalo/Rounds/");
    wcl = ml_clock_sum("/Subhalo/Rounds/Cleanup") - wcl;
    wtr = ml_clock_sum("/Subhalo/Rounds/Tree") - wtr;
    {
        const struct SubhaloCallStats * st = subhalo_last_call_stats();
        const double smean = st->StageMean[SUBHALO_STAGE_ROUNDS];
        /* mean over the ranks against the mean stage time: a rank's own stage time is not exposed */
        double loc[2] = {wt1 - wt0, wcl}, sum[2];
        MPI_Allreduce(loc, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double wmean = sum[0] / get_ntask(), clmean = sum[1] / get_ntask();
        message(0, "many labels (a): walltime /Subhalo/Rounds children %.4f s (cleanup %.4f s, tree builds %.4f s; mean over the ranks "
                   "%.4f s, cleanup %.4f s), the summary's rounds %.4f s (max), %.4f s (mean): coverage %.3f\n", wt1 - wt0, wcl, wtr,
                wmean, clmean, st->StageMax[SUBHALO_STAGE_ROUNDS], smean, wmean / smean);
        CHECK(wtr > 0, "many labels (a): no time on /Subhalo/Rounds/Tree (%g s)\n", wtr);
        /* the cleanup share is printed only: a descheduled rank can make it any size */
        CHECK(wmean <= smean * (1 + 1e-6) + 1e-4 && wmean >= 0.8 * smean - 1e-4,
              "many labels (a): walltime rounds %g s (mean %g s, cleanup %g s), summary mean %g s\n", wt1 - wt0, wmean, wcl, smean);
    }
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    const struct SubhaloCollAdmit * ad = subhalo_plan_last_admission();
    const struct SubhaloCollectiveStats * cs = subhalo_collective_last_stats();
    const double need = ad->need, noprops = ml_need_without_props(ad), delta = c.freepre - ad->avail;
    message(0, "many labels (a): ret %d in %.2f s, %ld groups, %ld candidates in %d round(s), %ld alive; this rank needs %g MB (%g held, "
               "properties %g MB, tree %g MB; %g MB without the properties stage), measured high water %g MB of the estimate %g MB "
               "(largest measured/estimated %.3f)\n", ret, MPI_Wtime() - t0, ng, ps->NCands, ps->NRounds, cs->NAlive, need / 1048576.,
            ad->persist / 1048576., ad->stage[SUBHALO_COLL_PROPS] / 1048576., ad->stage[SUBHALO_COLL_TREE] / 1048576.,
            noprops / 1048576., cs->MeasPeakBytes / 1048576., cs->EstPeakBytes / 1048576., cs->MaxMeasOverEst);
    CHECK(ret == 1 && ng == ML_NG && subhalo_last_call_complete() && ps->Admitted && ps->NCands == ML_NG && ps->NRounds == 1
          && cs->NAlive == ML_NG, "many labels (a): ret %d, %ld groups, %ld candidates, %ld alive\n", ret, ng, ps->NCands, cs->NAlive);
    CHECK(ad->checked && !ad->failed && ad->failstage == SUBHALO_COLL_PROPS, "many labels (a): the largest stage is the %s\n",
          ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-");
    CHECK(cs->MaxMeasOverEst > 0 && cs->MaxMeasOverEst <= 1, "many labels (a): measured / estimated %g\n", cs->MaxMeasOverEst);
    CHECK(need - noprops > 2 * 1048576., "many labels (a): the properties stage adds %g MB\n", (need - noprops) / 1048576.);
    syn_teardown();
    /* (b) */
    ml_setup(&CP);
    c.target = delta + 0.5 * (need + noprops);
    struct tagrec * pre = NULL;
    int64_t npre = 0, n, i, bad = 0;
    pre = gather_tags(&npre);
    ret = run_finder_call_ex(1, &CP, &ng, ml_pre, ml_post, &c);
    ps = subhalo_plan_last_summary();
    ad = subhalo_plan_last_admission();
    message(0, "many labels (b): free %g MB at the call squeezed to %g MB; ret %d, admitted %d, failed %d on rank %d at the %s; this rank needs "
               "%g MB (%g MB without the properties stage), %g MB available\n", c.freepre / 1048576., c.target / 1048576., ret, ps->Admitted,
            ad->failed, ad->failrank, ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-", ad->need / 1048576.,
            ml_need_without_props(ad) / 1048576., ad->avail / 1048576.);
    CHECK(ret == 0 && !subhalo_last_call_complete() && !ps->Admitted && ad->failed && ad->failstage == SUBHALO_COLL_PROPS,
          "many labels (b): ret %d, admitted %d, failed %d at stage %d\n", ret, ps->Admitted, ad->failed, ad->failstage);
    /* on every rank the need without the properties stage fits: the old estimate admitted the call */
    LOCAL_CHECK(bad, ml_need_without_props(ad) < ad->avail && ad->need > ad->avail, "many labels (b): need %g / %g MB without the "
                "properties, %g MB available\n", ad->need / 1048576., ml_need_without_props(ad) / 1048576., ad->avail / 1048576.);
    struct tagrec * post = gather_tags(&n);
    for(i = 0; i < n && n == npre; i++)
        if(post[i].ID != pre[i].ID || post[i].TrackId != pre[i].TrackId || post[i].BoundTrackId != pre[i].BoundTrackId
           || post[i].BindRank != pre[i].BindRank)
            bad++;
    CHECK(n == npre, "many labels (b): %ld / %ld particles\n", n, npre);
    free(post);
    free(pre);
    assert_all_true(bad == 0);
    syn_teardown();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_collective_serial_vs_inplace),
        cmocka_unit_test(test_collective_subtree_keys),
        cmocka_unit_test(test_collective_merger),
        cmocka_unit_test(test_collective_kill_passup),
        cmocka_unit_test(test_collective_binder),
        cmocka_unit_test(test_collective_serial_repairs),
        cmocka_unit_test(test_collective_relaxed_boundary),
        cmocka_unit_test(test_collective_dead_field),
        cmocka_unit_test(test_collective_sunk_root_field),
        cmocka_unit_test(test_collective_failure_after_split),
        cmocka_unit_test(test_collective_voided_claim),
        cmocka_unit_test(test_collective_long_chain),
        cmocka_unit_test(test_collective_admission),
        cmocka_unit_test(test_collective_admission_props),
        cmocka_unit_test(test_collective_guards),
        cmocka_unit_test(test_collective_walk_theta),
        cmocka_unit_test(test_collective_props_identity),
        cmocka_unit_test(test_collective_central_source),
    };
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
