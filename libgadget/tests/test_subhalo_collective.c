/* Unit tests of the oversized candidates unbound in place (Phase 2b plan section 5, tests 1-4,
 * halofinder_phase2b_plan.md). Run as: mpirun -np 4 .objs/test_subhalo_collective
 *
 *  1. test_collective_serial_vs_inplace  a Hernquist clump with an embedded satellite (the overlap
 *     scenario of test_subhalo.c), processed (a) serially and (b) with SubhaloCollectiveMinPart =
 *     1 (both the central and the satellite's sub-tree in place, two rounds), subsampling off:
 *     Nbound within 0.1 %, the ten most-bound tracers identical, BoundTrackId of every particle
 *     agreeing to 0.2 %, the properties within the tree tolerance.
 *  2. test_collective_subtree_keys       a central with two satellites, one carrying its own
 *     satellite, forced in place for the central only: the sub-trees are serial keys, the
 *     satellite's stripped particles (its provisional remainder) are bound by the in-place
 *     central under the satellite's tail tag, the D10 invariants, agreement with the serial run.
 *  3. test_collective_merger             two groups above the forced threshold merge: the loser's
 *     sub-tree in place at depth 1 (round 0), then the central (round 1); the loser is a
 *     satellite with its mass within tolerance of the serial run.
 *  4. test_collective_kill_passup        an in-place satellite that dies passes its particles to
 *     its parent candidate in the next round, which binds them.
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

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_collective_serial_vs_inplace),
        cmocka_unit_test(test_collective_subtree_keys),
        cmocka_unit_test(test_collective_merger),
        cmocka_unit_test(test_collective_kill_passup),
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
