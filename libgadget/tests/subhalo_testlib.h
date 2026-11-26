#ifndef SUBHALO_TESTLIB_H
#define SUBHALO_TESTLIB_H

/* Helpers shared by test_subhalo.c and test_subhalo_collective.c: rank and assertion helpers, the
 * default parameters and cosmology, the per-ID random numbers, particle tables, and the two-clump
 * (p2_) and synthetic (syn_) fixtures. Included after the test file's includes and its
 * struct tagrec, which differs between the two files. */

static struct ClockTable CT;

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
    /* the fixtures follow an orphan through its MostBoundID alone (default 10; orphan_set tests K > 1) */
    sp.SubhaloOrphanTracers = 1;
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
    sp.SubhaloAdmitMinCoreDistance = 12;
    sp.SubhaloUnbindWithThermal = 0;
    /* off by default: sparse particle sets give the full tree fewer nodes than the
     * top-level grid; the fixtures that need it (p2_setup and others) switch it on */
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

static int
cmp_tagrec_id(const void * a, const void * b)
{
    const struct tagrec * x = a, * y = b;
    return (x->ID > y->ID) - (x->ID < y->ID);
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

/* The invariants of design D10 on the gathered particles against the records: own-tag BindRank dense with
 * the exclusive bound rows first; BindRank is the energy order of all own-tag particles,
 * so the reservation shows only as: the first min(10, own-tag tracers) tracer-type (type 1)
 * particles in rank order are bound to the track, and MostBoundID is the first of them;
 * BoundTrackId the tag, an ancestor of it or -1, bound owners resolved; Nbound, NboundTracer,
 * NboundExclusive, NtailType from the particles. Records processed in place
 * (SUBHALO_FLAG_INPLACE) obey the same rules. Returns the number of violations. */
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
 * A's centre (the overlap scenario: a stripped tail re-bound by the parent), positions and
 * velocities for atime. */
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

/* Synthetic tracking state: particles
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

#endif
