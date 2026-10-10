/* Unit test of the field spherical overdensity (subhalo_so.c; Phase 3 plan section 5, test 5).
 *
 * Run as: mpirun -np 4 .objs/test_subhalo_so   (also -np 1, 2, 3)
 *
 * A periodic box of 1000 kpc/h with a uniform background (type 2) and three clumps:
 *   A  a Hernquist clump (type 1) with a live black hole at its centre; refined by pass 2;
 *   B  a Hernquist clump, half gas (type 0), with a compact second concentration 100 kpc/h
 *      from its centre, inside the search sphere, which moves the outermost 200b crossing
 *      outwards; refined;
 *   C  a small clump straddling the periodic boundary in x and z; pass-1 interpolation only;
 * plus a central in the most empty point of the box (no crossing), a satellite, a hostless
 * track and a dead track (not queried), a swallowed black hole of mass 1024 at B's centre and
 * a garbage particle of mass 1024 at A's centre (both must be ignored), and a non-zero
 * CurrentParticleOffset (CorePos is in the output frame, P[].Pos in the translated one).
 * Particle masses are dyadic, so every mass sum is exact whatever its order.
 *
 * Checks:
 *  1. Against a brute-force particle-level count on the gathered particles inside the final
 *     search radius (Rockstar's rule: the outermost particle k with M(<=r_k) / (4/3 pi r_k^3)
 *     > rho): MSO within the mass of one pass-1 bin around the crossing (log width
 *     ln(R / Rlo) / SubhaloSOBins on either side) for unrefined centrals and within that of two
 *     pass-2 sub-bins for refined ones; RSO = (3 MSO / (4 pi rho))^(1/3); the per-type masses
 *     sum to MSO and match the brute-force per-type masses within the same tolerance; the
 *     flags; the relative errors are logged.
 *  2. Identical results (bit for bit) with the records spread over the ranks, all on one rank,
 *     and with batches of 1 and 2 queries per walk; a checksum is logged ("SO test checksum")
 *     that must be identical with any number of ranks.
 *  3. The doubling cap: SubhaloSOSearchFactor = 0.02 and SubhaloSOMaxDoublings = 0 set
 *     SUBHALO_FLAG_SOCUTOFF on A and B with MSO the exact mass inside the search radius; with
 *     SubhaloSOSearchFactor = 0.3 the radius is doubled and the results again match the brute
 *     force.
 *  4. subhalo_field_so (thresholds from subhalo_so_thresholds) equals the explicit-threshold
 *     call when the thresholds agree with this test's formulas.
 *  5. The memory check: with a large mymalloc block leaving just the treewalk reserve of pass 1
 *     at 16 bins free after the tree build, the call halves its bins to 16, skips pass 2 and
 *     still matches the brute force (with the tolerance of 16 pass-1 bins); with half of that
 *     free it skips the field SO (every central at -1, SO flags clear, others untouched).
 *  6. A rank without particles (an interior rank emptied after the domain decomposition, as in
 *     test_subhalo.c): its centrals are routed to another rank and the results are identical,
 *     with the records spread over the ranks and with all of them on the empty rank.
 *  7. Non-dyadic, mixed-mass and mixed-type refined shells (test_subhalo_field_so_mixed):
 *     refined masses within one f4 ulp of the long-double particle sums, the per-type sum
 *     invariant, and agreement between particle partitions (see the comment before that test).
 *     The dyadic fixture above keeps the exact, bit-for-bit mapping checks. */

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

#include <libgadget/subhalo.h>
#include <libgadget/subhalo_unbind.h>
#include <libgadget/subhalo_so.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/cosmology.h>
#include <libgadget/gravity.h>
#include <libgadget/utils/mymalloc.h>
#include "stub.h"

static struct ClockTable CT;

#define CHECK(cond, ...) do { if(!(cond)) { message(1, __VA_ARGS__); } assert_true(cond); } while(0)

#define BOX 1000.
#define MPART (1. / 1024)          /* DM and gas */
#define MBG (1. / 512)             /* background, type 2 */
#define MBH (1. / 64)              /* the live black hole */
#define NA 16384
#define NB1 8192
#define NB2 4096
#define NC 512
#define NBG 4096
#define NREC 7

static const double CenA[3] = {300, 300, 300};
static const double CenB[3] = {700, 300, 600};
static const double CenB2[3] = {800, 300, 600};
static const double CenC[3] = {995, 500, 3};
static const double Offset[3] = {37.5, -12.25, 501};

/* ------------------------------------------------------------------ helpers */

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

static double
wrap(double x)
{
    while(x >= BOX) x -= BOX;
    while(x < 0) x += BOX;
    return x;
}

static double
nearest(double dx)
{
    if(dx > 0.5 * BOX) return dx - BOX;
    if(dx < -0.5 * BOX) return dx + BOX;
    return dx;
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
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.01, units);
    return CP;
}

/* The thresholds of design 7.10, written independently of subhalo_props.c. */
static void
test_thresholds(const Cosmology * CP, double a, double rho[SUBHALO_NSO])
{
    const double H = hubble_function(CP, a);
    const double rhoc = 3 * H * H * a * a * a / (8 * M_PI * CP->GravInternal);
    const double rhom = CP->Omega0 * 3 * CP->Hubble * CP->Hubble / (8 * M_PI * CP->GravInternal);
    const double x = CP->Omega0 / (a * a * a) * CP->Hubble * CP->Hubble / (H * H) - 1;
    rho[0] = (18 * M_PI * M_PI + 82 * x - 39 * x * x) * rhoc;
    rho[1] = 200 * rhom;
    rho[2] = 200 * rhoc;
    rho[3] = 500 * rhoc;
    rho[4] = 2500 * rhoc;
}

/* Particle id of the global set: position (output frame), type, mass. Deterministic. */
static void
make_particle(int64_t id, double pos[3], int * type, double * mass, int * swallowed)
{
    *swallowed = 0;
    const double * cen = NULL;
    double a = 0;
    int64_t k = id;
    *type = 1;
    *mass = MPART;
    if(k < NA) { cen = CenA; a = 20; }
    else if((k -= NA) < NB1) { cen = CenB; a = 15; *type = (k % 2) ? 0 : 1; }
    else if((k -= NB1) < NB2) { cen = CenB2; a = 4; }
    else if((k -= NB2) < NC) { cen = CenC; a = 8; }
    else if((k -= NC) < NBG) {
        int d;
        for(d = 0; d < 3; d++)
            pos[d] = BOX * urand(id, 7, d);
        *type = 2;
        *mass = MBG;
        return;
    }
    else {
        k -= NBG;
        int d;
        *type = 5;
        if(k == 0) {             /* live black hole at A */
            for(d = 0; d < 3; d++) pos[d] = CenA[d] + 0.1;
            *mass = MBH;
        }
        else {                   /* swallowed black hole at B */
            for(d = 0; d < 3; d++) pos[d] = CenB[d] + 0.2;
            *mass = 1024;
            *swallowed = 1;
        }
        return;
    }
    /* Hernquist truncated at 10 a: M(<r) / M = r^2 / (r + a)^2 */
    const double umax = (10. / 11.) * (10. / 11.);
    const double s = sqrt(umax * urand(id, 1, 0));
    const double r = a * s / (1 - s);
    const double ct = 2 * urand(id, 2, 0) - 1, st = sqrt(1 - ct * ct), ph = 2 * M_PI * urand(id, 3, 0);
    pos[0] = wrap(cen[0] + r * st * cos(ph));
    pos[1] = wrap(cen[1] + r * st * sin(ph));
    pos[2] = wrap(cen[2] + r * ct);
}

#define NTOT (NA + NB1 + NB2 + NC + NBG + 2)
#define ID_GARBAGE 5      /* an A particle turned into garbage of mass 1024 at A's centre */

static void
setup_particles(void)
{
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    particle_alloc_memory(PartManager, BOX, NTOT + 1000);
    PartManager->NumPart = 0;
    int d;
    for(d = 0; d < 3; d++)
        PartManager->CurrentParticleOffset[d] = Offset[d];
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t atleast[6] = {NB1, 0, 0, 0, 16, 16};
    slots_reserve(1, atleast, SlotsManager);
    int64_t id, ngas = 0, nbh = 0;
    for(id = 0; id < NTOT; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        struct particle_data * p = &P[PartManager->NumPart++];
        memset(p, 0, sizeof(*p));
        double pos[3], mass;
        int type, swallowed;
        make_particle(id, pos, &type, &mass, &swallowed);
        p->ID = id;
        p->Type = type;
        p->Mass = mass;
        p->Swallowed = swallowed;
        p->GrNr = -1;
        p->TrackId = p->BoundTrackId = -1;
        p->BindRank = -1;
        for(d = 0; d < 3; d++)
            p->Pos[d] = wrap(pos[d] + Offset[d]);
        if(type == 0)
            p->PI = ngas++;
        if(type == 5)
            p->PI = nbh++;
    }
    SlotsManager->info[0].size = ngas;
    SlotsManager->info[5].size = nbh;
    slots_setup_id(PartManager, SlotsManager);
}

/* Gathered live and dead particles: output-frame position, mass, type, excluded flag. */
struct gpart {
    double x[3];
    double m;
    int type;
    int excluded;
};

static struct gpart *
gather_particles(int64_t * ntot)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const int64_t n = PartManager->NumPart;
    struct gpart * loc = malloc((n + 1) * sizeof(struct gpart));
    int64_t i;
    int d;
    for(i = 0; i < n; i++) {
        for(d = 0; d < 3; d++)
            loc[i].x[d] = wrap(P[i].Pos[d] - Offset[d]);
        loc[i].m = P[i].Mass;
        loc[i].type = P[i].Type;
        loc[i].excluded = P[i].IsGarbage || P[i].Swallowed;
    }
    int nb = n * sizeof(struct gpart);
    int * cnt = malloc(NTask * sizeof(int)), * off = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    for(d = 0; d < NTask; d++) {
        off[d] = tot;
        tot += cnt[d];
    }
    struct gpart * all = malloc(tot + sizeof(struct gpart));
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, cnt, off, MPI_BYTE, MPI_COMM_WORLD);
    free(off);
    free(cnt);
    free(loc);
    *ntot = tot / sizeof(struct gpart);
    return all;
}

/* ------------------------------------------------------------------ brute force */

struct rpair {
    double r;
    int64_t i;
};

static int
cmp_rpair(const void * a, const void * b)
{
    const struct rpair * x = a, * y = b;
    if(x->r != y->r)
        return (x->r > y->r) - (x->r < y->r);
    return (x->i > y->i) - (x->i < y->i);
}

struct brute {
    double M[SUBHALO_NSO], R[SUBHALO_NSO];     /* -1 when no particle qualifies */
    double Minside;                            /* mass inside Rsearch */
    int above;                                 /* density at Rsearch above min(rho) */
};

/* Mass within r of cen (sorted pairs; inclusive). */
static double
mass_within(const struct rpair * rp, int64_t n, const struct gpart * g, double r, int type)
{
    double M = 0;
    int64_t k;
    for(k = 0; k < n && rp[k].r <= r; k++)
        if(type < 0 || g[rp[k].i].type == type)
            M += g[rp[k].i].m;
    return M;
}

static struct rpair *
radial_order(const struct gpart * g, int64_t ng, const double cen[3], double Rmax, int64_t * n)
{
    struct rpair * rp = malloc((ng + 1) * sizeof(struct rpair));
    int64_t i, m = 0;
    for(i = 0; i < ng; i++) {
        if(g[i].excluded)
            continue;
        double r2 = 0;
        int d;
        for(d = 0; d < 3; d++) {
            const double dx = nearest(g[i].x[d] - cen[d]);
            r2 += dx * dx;
        }
        const double r = sqrt(r2);
        if(r > Rmax)
            continue;
        rp[m].r = r;
        rp[m].i = i;
        m++;
    }
    qsort(rp, m, sizeof(rp[0]), cmp_rpair);
    *n = m;
    return rp;
}

static struct brute
brute_force(const struct gpart * g, int64_t ng, const double cen[3], double Rsearch, const double rho[SUBHALO_NSO])
{
    struct brute b;
    int64_t n, k;
    int d;
    struct rpair * rp = radial_order(g, ng, cen, Rsearch, &n);
    for(d = 0; d < SUBHALO_NSO; d++)
        b.M[d] = b.R[d] = -1;
    double M = 0;
    for(k = 0; k < n; k++) {
        M += g[rp[k].i].m;
        const double r = rp[k].r;
        for(d = 0; d < SUBHALO_NSO; d++)
            if(M > 4. / 3. * M_PI * rho[d] * r * r * r) {
                b.M[d] = M;
                b.R[d] = r;
            }
    }
    b.Minside = M;
    double rhomin = rho[0];
    for(d = 1; d < SUBHALO_NSO; d++)
        rhomin = rho[d] < rhomin ? rho[d] : rhomin;
    b.above = M > 4. / 3. * M_PI * rhomin * Rsearch * Rsearch * Rsearch;
    free(rp);
    return b;
}

/* ------------------------------------------------------------------ records */

struct result {
    double MSO[SUBHALO_NSO], RSO[SUBHALO_NSO], MT200c[6], MTVir[6];
    double Flags, Rsearch;
};

static void
make_records(struct SubTrack * G, const double voidpos[3])
{
    int k, d;
    for(k = 0; k < NREC; k++) {
        subhalo_track_clear(&G[k], k);
        G[k].Rank = 0;
        G[k].HostGroup = k;
        G[k].Flags = SUBHALO_FLAG_ITERCAP | SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS;
        for(d = 0; d < SUBHALO_NSO; d++)
            G[k].MSO[d] = G[k].RSO[d] = -7;
        for(d = 0; d < 6; d++)
            G[k].MTypeSO200c[d] = G[k].MTypeSOVir[d] = -7;
    }
    const double * cen[NREC] = {CenA, CenB, CenC, voidpos, CenA, CenB, CenC};
    const int nb[NREC] = {NA, NB1, NC, 30, 100, 100, 0};
    for(k = 0; k < NREC; k++) {
        for(d = 0; d < 3; d++)
            G[k].CorePos[d] = cen[k][d];
        G[k].Nbound = nb[k];
        G[k].Mbound = nb[k] * MPART;
    }
    G[4].Rank = 1;               /* satellite */
    G[5].HostGroup = -1;         /* hostless */
}

static int
is_central(int k)
{
    return k < 4;
}

/* Run one call with the records placed on ranks by owner(k) and gather the results. */
static void
run_case(DomainDecomp * ddecomp, const struct SubTrack * G, int (*owner)(int, int), const Cosmology * CP,
         const struct SubhaloParams * par, const double * rho, struct result * res, struct SubhaloSOStats * stats)
{
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    struct SubTrack * loc = malloc(NREC * sizeof(struct SubTrack));
    int idx[NREC];
    int n = 0, k, d;
    for(k = 0; k < NREC; k++)
        if(owner(k, NTask) == ThisTask) {
            idx[n] = k;
            loc[n++] = G[k];
        }
    double Rs[NREC];
    if(rho)
        subhalo_field_so_full(loc, n, ddecomp, 1.0, CP, par, rho, Rs, stats, MPI_COMM_WORLD);
    else {
        subhalo_field_so(loc, n, ddecomp, 1.0, CP, par, MPI_COMM_WORLD);
        for(k = 0; k < n; k++)
            Rs[k] = 0;
    }
    struct result * mine = calloc(NREC, sizeof(struct result));
    for(k = 0; k < n; k++) {
        struct result * r = &mine[idx[k]];
        const struct SubTrack * t = &loc[k];
        for(d = 0; d < SUBHALO_NSO; d++) {
            r->MSO[d] = t->MSO[d];
            r->RSO[d] = t->RSO[d];
        }
        for(d = 0; d < 6; d++) {
            r->MT200c[d] = t->MTypeSO200c[d];
            r->MTVir[d] = t->MTypeSOVir[d];
        }
        r->Flags = t->Flags;
        r->Rsearch = Rs[k];
    }
    MPI_Allreduce(mine, res, NREC * sizeof(struct result) / sizeof(double), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(mine);
    free(loc);
}

static int owner_spread(int k, int NTask) { return k % NTask; }
static int owner_last(int k, int NTask) { return NTask - 1; }
static int owner_first(int k, int NTask) { return 0; }
static int EmptyRank = -1;
static int owner_empty(int k, int NTask) { return EmptyRank; }

/* Move every particle of rank `er` to rank er - 1 by giving its top leaves to that rank (the
 * leaves of a task are a contiguous range), so that `er` is an empty rank (test_subhalo.c). */
static void
empty_rank_domain(DomainDecomp * dd, int er)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
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

static uint64_t
checksum(const struct result * res)
{
    const unsigned char * b = (const unsigned char *) res;
    uint64_t h = 1469598103934665603ULL;
    size_t i;
    for(i = 0; i < NREC * sizeof(struct result); i++)
        h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

static const char * DefName[SUBHALO_NSO] = {"vir", "200b", "200c", "500c", "2500c"};

/* Compare a call's results with the brute force. Returns the number of failures. */
static int
compare_brute(const struct result * res, const struct gpart * g, int64_t ng, const double voidpos[3],
              const struct SubhaloParams * par, const double rho[SUBHALO_NSO], const char * what)
{
    int bad = 0, k, d, t;
    const double * cen[4] = {CenA, CenB, CenC, voidpos};
    const double soft = FORCE_SOFTENING() / 2.8;
    for(k = 0; k < NREC; k++) {
        const struct result * r = &res[k];
        if(!is_central(k)) {
            /* not queried: untouched */
            for(d = 0; d < SUBHALO_NSO; d++)
                if(r->MSO[d] != -7 || r->RSO[d] != -7) bad++;
            for(t = 0; t < 6; t++)
                if(r->MT200c[t] != -7 || r->MTVir[t] != -7) bad++;
            if((int) r->Flags != (SUBHALO_FLAG_ITERCAP | SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS) || r->Rsearch != -1) {
                message(0, "%s: record %d (not a central) was modified\n", what, k);
                bad++;
            }
            continue;
        }
        const double R = r->Rsearch;
        const double Rlo = fmin(soft, 0.25 * R);
        const double dlog = log(R / Rlo) / par->SubhaloSOBins;
        const int refined = (k == 0 ? NA : (k == 1 ? NB1 : (k == 2 ? NC : 30))) >= par->SubhaloSORefineMinPart;
        struct brute b = brute_force(g, ng, cen[k], R, rho);
        int64_t n;
        struct rpair * rp = radial_order(g, ng, cen[k], R, &n);
        const int flags = (int) r->Flags;
        if(!(flags & SUBHALO_FLAG_ITERCAP)) {
            message(0, "%s: record %d lost an unrelated flag\n", what, k);
            bad++;
        }
        if(!!(flags & SUBHALO_FLAG_SOCUTOFF) != b.above) {
            message(0, "%s: record %d cutoff flag %d, brute force above at R = %g: %d\n", what, k,
                    !!(flags & SUBHALO_FLAG_SOCUTOFF), R, b.above);
            bad++;
        }
        int nocross = 0;
        for(d = 0; d < SUBHALO_NSO; d++) {
            const double K = 4. / 3. * M_PI * rho[d];
            if(r->MSO[d] < 0) {
                nocross = 1;
                if(r->MSO[d] != -1 || r->RSO[d] != -1) bad++;
                if(b.M[d] >= 0) {
                    message(0, "%s: record %d %s: no crossing found, brute force M = %g at r = %g\n", what, k, DefName[d], b.M[d], b.R[d]);
                    bad++;
                }
                continue;
            }
            if(b.M[d] < 0) {
                message(0, "%s: record %d %s: M = %g but the brute force has no crossing\n", what, k, DefName[d], r->MSO[d]);
                bad++;
                continue;
            }
            /* RSO consistent with MSO */
            const double Rc = cbrt(r->MSO[d] / K);
            if(fabs(r->RSO[d] - Rc) > 1e-5 * Rc) {
                message(0, "%s: record %d %s: RSO %g != (3 M / 4 pi rho)^1/3 = %g\n", what, k, DefName[d], r->RSO[d], Rc);
                bad++;
            }
            double tol, Mref = b.M[d];
            if((flags & SUBHALO_FLAG_SOCUTOFF) && b.Minside > K * R * R * R) {
                /* still above at R: the exact mass inside R */
                Mref = b.Minside;
                tol = 1e-6 * Mref;
            }
            else {
                /* the mass of one pass-1 bin (two sub-bins) on either side of the crossing */
                const double w = refined ? 2 * (exp(dlog) - 1) / SUBHALO_SO_NSUB : (exp(dlog) - 1);
                tol = mass_within(rp, n, g, b.R[d] * (1 + w), -1) - mass_within(rp, n, g, b.R[d] / (1 + w), -1);
                tol = fmax(tol, 1e-6 * Mref);
            }
            const double err = r->MSO[d] - Mref;
            message(0, "%s: record %d %s: MSO %.6g brute %.6g (r %.4g) rel err %+.2e tol %.2e (%s)\n", what, k, DefName[d],
                    r->MSO[d], Mref, b.R[d], err / Mref, tol / Mref, refined ? "pass 2" : "pass 1");
            if(fabs(err) > tol) {
                message(0, "%s: record %d %s: |MSO - brute| = %g above the tolerance %g\n", what, k, DefName[d], fabs(err), tol);
                bad++;
            }
            /* per-type masses: sum to MSO and match the brute force per type */
            if(d == 0 || d == 2) {
                const double * mt = d == 0 ? r->MTVir : r->MT200c;
                double sum = 0;
                for(t = 0; t < 6; t++) {
                    sum += mt[t];
                    const double ref = mass_within(rp, n, g, Mref == b.Minside ? R : b.R[d], t);
                    if(fabs(mt[t] - ref) > tol + 1e-6 * Mref) {
                        message(0, "%s: record %d %s type %d: %g vs brute %g (tol %g)\n", what, k, DefName[d], t, mt[t], ref, tol);
                        bad++;
                    }
                }
                if(fabs(sum - r->MSO[d]) > 1e-5 * r->MSO[d]) {
                    message(0, "%s: record %d %s: per-type sum %g != MSO %g\n", what, k, DefName[d], sum, r->MSO[d]);
                    bad++;
                }
            }
        }
        if(!!(flags & SUBHALO_FLAG_SONOCROSS) != nocross) {
            message(0, "%s: record %d no-crossing flag %d but %d definitions without a crossing\n", what, k,
                    !!(flags & SUBHALO_FLAG_SONOCROSS), nocross);
            bad++;
        }
        free(rp);
    }
    return bad;
}

static int
same_results(const struct result * a, const struct result * b)
{
    return memcmp(a, b, NREC * sizeof(struct result)) == 0;
}

/* ------------------------------------------------------------------ the test */

static void
test_subhalo_field_so(void ** state)
{
    walltime_init(&CT);
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    init_forcetree_params(0.7);
    struct gravshort_tree_params tp = get_gravshort_treepar();
    tp.FractionalGravitySoftening = 1. / 50;
    set_gravshort_treepar(tp);
    gravshort_set_softenings(50.);       /* GravitySoftening = 1 kpc/h */
    /* the treewalk reserve with the parameter's default boost (logged for subhalo_so.c's
     * comment), then the boost of these tests: ImportBufferBoost is never set here, so
     * treewalk.c uses 0, and the 650 MB test allocator would not hold pass 2's reserve at 2 */
    subhalo_so_set_import_buffer_boost(2);
    message(0, "treewalk reserve with boost 2, 64 bins: NT = 1 pass 1 %zu pass 2 %zu; NT = 6 pass 1 %zu pass 2 %zu bytes\n",
            subhalo_so_walk_bytes(1, 64, 1), subhalo_so_walk_bytes(2, 64, 1), subhalo_so_walk_bytes(1, 64, 6),
            subhalo_so_walk_bytes(2, 64, 6));
    subhalo_so_set_import_buffer_boost(0);

    Cosmology CP = test_cosmology();
    double rho[SUBHALO_NSO], rhoA3[SUBHALO_NSO];
    test_thresholds(&CP, 1.0, rho);
    subhalo_so_thresholds(&CP, 1.0, rhoA3);
    int d, k;
    for(d = 0; d < SUBHALO_NSO; d++)
        message(0, "threshold %s: %g (subhalo_so_thresholds %g)\n", DefName[d], rho[d], rhoA3[d]);

    setup_particles();
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    /* the garbage particle: wherever it landed, at A's centre with mass 1024 */
    int64_t i;
    for(i = 0; i < PartManager->NumPart; i++)
        if(P[i].ID == ID_GARBAGE) {
            P[i].IsGarbage = 1;
            P[i].Mass = 1024;
            for(d = 0; d < 3; d++)
                P[i].Pos[d] = wrap(CenA[d] + Offset[d]);
        }

    int64_t ng;
    struct gpart * g = gather_particles(&ng);
    CHECK(ng == NTOT, "gathered %ld particles, expected %d\n", ng, NTOT);

    /* the emptiest point of a 20^3 grid away from the clumps */
    double voidpos[3] = {0, 0, 0}, best = -1;
    int ix, iy, iz;
    for(ix = 0; ix < 20; ix++)
    for(iy = 0; iy < 20; iy++)
    for(iz = 0; iz < 20; iz++) {
        const double c[3] = {25 + 50 * ix, 25 + 50 * iy, 25 + 50 * iz};
        double dmin = 1e30;
        for(i = 0; i < ng; i++) {
            double r2 = 0;
            for(d = 0; d < 3; d++) {
                const double dx = nearest(g[i].x[d] - c[d]);
                r2 += dx * dx;
            }
            if(r2 < dmin)
                dmin = r2;
        }
        if(dmin > best) {
            best = dmin;
            memcpy(voidpos, c, sizeof(c));
        }
    }
    message(0, "void central at %g %g %g, nearest particle %g\n", voidpos[0], voidpos[1], voidpos[2], sqrt(best));

    struct SubTrack * G = malloc(NREC * sizeof(struct SubTrack));
    make_records(G, voidpos);

    struct SubhaloParams par;
    memset(&par, 0, sizeof(par));
    par.SubhaloSOBins = 64;
    par.SubhaloSOSearchFactor = 2;
    par.SubhaloSOMaxDoublings = 3;
    par.SubhaloSORefineMinPart = 1000;

    /* 1. defaults against the brute force */
    struct result res1[NREC], res2[NREC], res3[NREC], res4[NREC];
    struct SubhaloSOStats st;
    run_case(&ddecomp, G, owner_spread, &CP, &par, rho, res1, &st);
    CHECK(st.Ncentral == 4 && st.Nrefined == 2 && st.Ncutoff == 0 && st.Nnocross == 1,
          "stats: %ld centrals %ld refined %ld cut off %ld without crossing\n", st.Ncentral, st.Nrefined, st.Ncutoff, st.Nnocross);
    int bad = compare_brute(res1, g, ng, voidpos, &par, rho, "defaults");
    CHECK(bad == 0, "defaults: %d failures against the brute force\n", bad);
    /* the void central has no crossing at all; B's 200b crossing lies beyond its main clump */
    for(d = 0; d < SUBHALO_NSO; d++)
        CHECK(res1[3].MSO[d] == -1, "void central %s: %g\n", DefName[d], res1[3].MSO[d]);
    CHECK(res1[1].RSO[1] > 100, "B: R200b = %g, not beyond the second concentration's centre\n", res1[1].RSO[1]);

    /* 2. rank placement and batching do not change a bit */
    run_case(&ddecomp, G, owner_last, &CP, &par, rho, res2, NULL);
    CHECK(same_results(res1, res2), "results differ with all records on the last rank\n");
    subhalo_so_set_max_batch(1);
    run_case(&ddecomp, G, owner_spread, &CP, &par, rho, res3, &st);
    CHECK(same_results(res1, res3), "results differ with batches of one query\n");
    subhalo_so_set_max_batch(2);
    run_case(&ddecomp, G, owner_first, &CP, &par, rho, res4, &st);
    CHECK(same_results(res1, res4), "results differ with batches of two queries on rank 0\n");
    CHECK(st.Npass1Walks >= 2, "batches of two on rank 0: %d pass-1 walks\n", st.Npass1Walks);
    subhalo_so_set_max_batch(0);
    message(0, "SO test checksum: %016lx\n", checksum(res1));

    /* 3. the doubling cap, then doubling within the cap */
    struct SubhaloParams cap = par;
    cap.SubhaloSOSearchFactor = 0.02;
    cap.SubhaloSOMaxDoublings = 0;
    run_case(&ddecomp, G, owner_spread, &CP, &cap, rho, res2, &st);
    CHECK((int) res2[0].Flags & SUBHALO_FLAG_SOCUTOFF, "A not cut off with a tiny search radius\n");
    CHECK((int) res2[1].Flags & SUBHALO_FLAG_SOCUTOFF, "B not cut off with a tiny search radius\n");
    CHECK(st.Ncutoff >= 2 && st.Ndoubled == 0, "cap: %ld cut off, %ld doubled\n", st.Ncutoff, st.Ndoubled);
    bad = compare_brute(res2, g, ng, voidpos, &cap, rho, "cap");
    CHECK(bad == 0, "cap: %d failures against the brute force\n", bad);
    for(k = 0; k < 2; k++) {
        /* every definition still above at R: the exact mass inside R, the radius beyond R */
        const double * cen = k == 0 ? CenA : CenB;
        struct brute b = brute_force(g, ng, cen, res2[k].Rsearch, rho);
        for(d = 0; d < SUBHALO_NSO; d++)
            CHECK(res2[k].MSO[d] == (float) b.Minside && res2[k].RSO[d] > res2[k].Rsearch,
                  "cap: record %d %s: MSO %g inside-R mass %g RSO %g R %g\n", k, DefName[d], res2[k].MSO[d], b.Minside,
                  res2[k].RSO[d], res2[k].Rsearch);
    }

    struct SubhaloParams dbl = par;
    dbl.SubhaloSOSearchFactor = 0.3;
    dbl.SubhaloSOMaxDoublings = 4;
    run_case(&ddecomp, G, owner_spread, &CP, &dbl, rho, res3, &st);
    CHECK(st.Ndoubled >= 3 && st.Ncutoff == 0, "doubling: %ld doubled, %ld cut off\n", st.Ndoubled, st.Ncutoff);
    for(k = 0; k < 3; k++)
        CHECK(res3[k].Rsearch > 0.3 * cbrt(3 * G[k].Mbound / (4 * M_PI * rho[1])) * 1.5,
              "doubling: record %d final R %g not doubled\n", k, res3[k].Rsearch);
    bad = compare_brute(res3, g, ng, voidpos, &dbl, rho, "doubling");
    CHECK(bad == 0, "doubling: %d failures against the brute force\n", bad);

    /* pass 1 only for every central */
    struct SubhaloParams p1 = par;
    p1.SubhaloSORefineMinPart = 1 << 30;
    run_case(&ddecomp, G, owner_spread, &CP, &p1, rho, res4, &st);
    CHECK(st.Nrefined == 0, "pass 1 only: %ld refined\n", st.Nrefined);
    bad = compare_brute(res4, g, ng, voidpos, &p1, rho, "pass 1 only");
    CHECK(bad == 0, "pass 1 only: %d failures against the brute force\n", bad);

    /* 4. the plain entry point with subhalo_so_thresholds */
    int same_rho = 1;
    for(d = 0; d < SUBHALO_NSO; d++)
        same_rho &= fabs(rhoA3[d] - rho[d]) <= 1e-12 * rho[d];
    CHECK(same_rho, "subhalo_so_thresholds differs from the design 7.10 formulas\n");
    run_case(&ddecomp, G, owner_spread, &CP, &par, NULL, res2, NULL);
    for(k = 0; k < NREC; k++)
        for(d = 0; d < SUBHALO_NSO; d++)
            CHECK(fabs(res2[k].MSO[d] - res1[k].MSO[d]) <= 1e-5 * fabs(res1[k].MSO[d]),
                  "plain call: record %d %s: %g vs %g\n", k, DefName[d], res2[k].MSO[d], res1[k].MSO[d]);

    /* 5. the memory check: the free bytes after the tree build of this configuration, then a
     * block leaving the pass-1 reserve at 16 bins (+ 4 MiB), then half of that */
    run_case(&ddecomp, G, owner_spread, &CP, &par, rho, res1, &st);
    CHECK(st.NBins == 64 && st.Pass2 && !st.Skipped && st.Nrouted == 0,
          "defaults: %d bins pass 2 %d skipped %d routed %ld\n", st.NBins, st.Pass2, st.Skipped, st.Nrouted);
    const int64_t free0 = st.FreeBytes;
    const int NT = st.NTypes;
    CHECK(NT == 4, "%d particle types present, expected 4 (gas, DM, background, black hole)\n", NT);
    const int64_t fit16 = subhalo_so_walk_bytes(1, 16, NT) + (4L << 20);
    message(0, "memory check: %ld bytes free after the tree build; pass 1 needs %zu (64 bins) %zu (32) %zu (16), pass 2 %zu\n",
            free0, subhalo_so_walk_bytes(1, 64, NT), subhalo_so_walk_bytes(1, 32, NT), subhalo_so_walk_bytes(1, 16, NT),
            subhalo_so_walk_bytes(2, 16, NT));
    CHECK(free0 > fit16 + (8L << 20) && (int64_t) subhalo_so_walk_bytes(1, 32, NT) > fit16,
          "memory check: %ld free bytes do not allow the test (16 bins need %ld)\n", free0, fit16);
    {
        void * block = mymalloc("SOtestblock", free0 - fit16);
        struct result resm[NREC];
        run_case(&ddecomp, G, owner_spread, &CP, &par, rho, resm, &st);
        myfree(block);
        CHECK(st.NBins == 16 && !st.Pass2 && !st.Skipped && st.Nrefined == 0 && st.Ncentral == 4,
              "shrink: %d bins pass 2 %d skipped %d refined %ld (free %ld)\n", st.NBins, st.Pass2, st.Skipped, st.Nrefined, st.FreeBytes);
        struct SubhaloParams p16 = par;
        p16.SubhaloSOBins = 16;
        p16.SubhaloSORefineMinPart = 1 << 30;
        bad = compare_brute(resm, g, ng, voidpos, &p16, rho, "shrunk to 16 bins");
        CHECK(bad == 0, "shrunk to 16 bins: %d failures against the brute force\n", bad);
        /* the same as an explicit 16-bin pass-1-only call */
        run_case(&ddecomp, G, owner_spread, &CP, &p16, rho, res2, &st);
        CHECK(same_results(resm, res2), "shrunk call differs from the explicit 16-bin pass-1 call\n");

        block = mymalloc("SOtestblock", free0 - fit16 / 2);
        run_case(&ddecomp, G, owner_spread, &CP, &par, rho, resm, &st);
        myfree(block);
        CHECK(st.Skipped && st.Npass1Walks == 0 && st.Npass2Walks == 0, "skip: skipped %d, %d + %d walks\n",
              st.Skipped, st.Npass1Walks, st.Npass2Walks);
        for(k = 0; k < NREC; k++) {
            const struct result * r = &resm[k];
            const double v = is_central(k) ? -1 : -7;
            int ok = 1;
            for(d = 0; d < SUBHALO_NSO; d++)
                ok &= r->MSO[d] == v && r->RSO[d] == v;
            for(d = 0; d < 6; d++)
                ok &= r->MT200c[d] == v && r->MTVir[d] == v;
            const int fl = SUBHALO_FLAG_ITERCAP | (is_central(k) ? 0 : SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS);
            ok &= (int) r->Flags == fl && r->Rsearch == -1;
            CHECK(ok, "skip: record %d MSO %g flags %d Rsearch %g\n", k, r->MSO[0], (int) r->Flags, r->Rsearch);
        }
    }

    /* 6. a rank without particles */
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(NTask >= 2) {
        EmptyRank = NTask / 2;
        empty_rank_domain(&ddecomp, EmptyRank);
        /* the particles sent away stay as garbage until a gc */
        int64_t nloc = 0, nempty;
        for(i = 0; i < PartManager->NumPart && ThisTask == EmptyRank; i++)
            nloc += !P[i].IsGarbage;
        MPI_Allreduce(&nloc, &nempty, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
        CHECK(nempty == 0, "the emptied rank %d holds %ld live particles\n", EmptyRank, nempty);
        int64_t nown = 0;
        for(k = 0; k < NREC; k++)
            nown += is_central(k) && owner_spread(k, NTask) == EmptyRank;
        run_case(&ddecomp, G, owner_spread, &CP, &par, rho, res2, &st);
        CHECK(st.Nrouted == nown && st.Ncentral == 4, "empty rank %d: %ld routed, expected %ld\n", EmptyRank, st.Nrouted, nown);
        CHECK(same_results(res1, res2), "empty rank %d: results differ with the records spread\n", EmptyRank);
        run_case(&ddecomp, G, owner_empty, &CP, &par, rho, res3, &st);
        CHECK(st.Nrouted == 4, "empty rank %d: %ld routed with every record on it\n", EmptyRank, st.Nrouted);
        CHECK(same_results(res1, res3), "empty rank %d: results differ with every record on it\n", EmptyRank);
        subhalo_so_set_max_batch(1);
        run_case(&ddecomp, G, owner_empty, &CP, &par, rho, res4, &st);
        subhalo_so_set_max_batch(0);
        CHECK(same_results(res1, res4), "empty rank %d: results differ with batches of one\n", EmptyRank);
    }

    free(G);
    free(g);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

/* ------------------------------------------------------------------ non-dyadic refined shells */

/* Test 7 (audit P3). One central with non-dyadic, mixed masses of types 1, 2 and 3:
 *   core    NM_CORE particles uniform in r inside 0.8 of the inner edge of the 200c crossing bin;
 *   shell1  NM_SH1 particles (half type 1, half type 3) in one pass-2 sub-bin (3) of that bin;
 *   shell2a NM_SH2A particles in the middle of the pass-1 bin before the vir crossing bin;
 *   shell2b NM_SH2B particles (types 2 and 3) in sub-bin 3 of the vir crossing bin;
 *   a background (type 2) only beyond the search radius.
 * The masses are scaled so that the 200c crossing lies in sub-bin 40 of its bin (core + shell1
 * = K200c r200c^3) and the vir crossing in sub-bin 40 of its bin, with the cumulative density at
 * each crossing bin's inner edge above the threshold. The sub-bins after the shell are empty, so
 * the interpolation returns the accumulated mass (the audit's trigger): MSO200c = M(core +
 * shell1), MSOvir = MSO200b = the whole clump, per type alike. 200b crosses in an empty bin.
 *
 * Accuracy bound (documented in subhalo_so.c): the sub-bins accumulate in double, so a refined
 * mass carries a relative summation error of at most ~N 2^-53 (N particles, 2e-11 here, ~1e-15
 * in practice) whatever the partition, below the f4 rounding of the record (2^-24). Checks, for
 * the default domain, the local particle order reversed and (NTask >= 2) a rank emptied:
 *   - MSO vir/200c/200b and MTypeSOVir/MTypeSO200c within one f4 ulp (2^-23 relative) of the
 *     long-double sums of the particle masses;
 *   - the per-type masses sum to MSO within 2^-22 relative (the f4 roundings);
 *   - every MSO (also 500c, 2500c, interpolated in the core) within 1e-6 relative of the
 *     default-domain result.
 * The float accumulation of the previous pass 2 misses these by orders of magnitude (100000
 * additions of ~1e-5 of the total each). */

#define NM_CORE 40000
#define NM_SH1 100000
#define NM_SH2A 20000
#define NM_SH2B 40000
#define NM_BG 8192
#define NM_TOT (NM_CORE + NM_SH1 + NM_SH2A + NM_SH2B + NM_BG)
#define NM_MBOUND 10.

static const double CenM[3] = {400, 520, 610};

struct mixed_geom {
    double R, Rlo, Scale;
    double rs1, r200, rs2a, rs2b, rvir, rcore;
    double fcore, fsh1, fsh2a, fsh2b;     /* mass scale of each population */
};

/* Radius of population member k (0..n) of a population: the shells are thin (1e-6 relative). */
static void
mixed_particle(int64_t id, const struct mixed_geom * gm, double pos[3], int * type, double * mass)
{
    int64_t k = id;
    double r;
    const double u = urand(id, 11, 0);
    if(k < NM_CORE) {
        r = 2 + (gm->rcore - 2) * u;
        *type = (k % 3 == 0) ? 3 : 1;
        *mass = gm->fcore * (*type == 3 ? 1.7 : 1.) * (1 + 0.3 * urand(id, 12, 0));
    }
    else if((k -= NM_CORE) < NM_SH1) {
        r = gm->rs1 * (1 + 1e-6 * u);
        *type = (k % 2) ? 3 : 1;
        *mass = gm->fsh1 * (1 + 0.5 * urand(id, 12, 0));
    }
    else if((k -= NM_SH1) < NM_SH2A) {
        r = gm->rs2a * (1 + 1e-6 * u);
        *type = (k % 2) ? 2 : 1;
        *mass = gm->fsh2a * (1 + 0.4 * urand(id, 12, 0));
    }
    else if((k -= NM_SH2A) < NM_SH2B) {
        r = gm->rs2b * (1 + 1e-6 * u);
        *type = (k % 2) ? 3 : 2;
        *mass = gm->fsh2b * (1 + 0.6 * urand(id, 12, 0));
    }
    else {
        /* background beyond the search radius */
        int d;
        *type = 2;
        *mass = 0.013 * (1 + urand(id, 12, 0));
        uint64_t s = 0;
        do {
            double r2 = 0;
            for(d = 0; d < 3; d++) {
                pos[d] = BOX * urand(id, 13 + s, d);
                const double dx = nearest(pos[d] - CenM[d]);
                r2 += dx * dx;
            }
            s++;
            if(r2 > (gm->R + 20) * (gm->R + 20))
                break;
        } while(1);
        return;
    }
    const double ct = 2 * urand(id, 2, 0) - 1, st = sqrt(1 - ct * ct), ph = 2 * M_PI * urand(id, 3, 0);
    pos[0] = wrap(CenM[0] + r * st * cos(ph));
    pos[1] = wrap(CenM[1] + r * st * sin(ph));
    pos[2] = wrap(CenM[2] + r * ct);
}

/* Population of particle id: 0 core, 1 shell1, 2 shell2a, 3 shell2b, 4 background. */
static int
mixed_pop(int64_t id)
{
    if(id < NM_CORE) return 0;
    if((id -= NM_CORE) < NM_SH1) return 1;
    if((id -= NM_SH1) < NM_SH2A) return 2;
    if((id -= NM_SH2A) < NM_SH2B) return 3;
    return 4;
}

/* Sum of the masses (all types if type < 0) of the populations 0..maxpop, in long double, from
 * the generator (the reference; the masses stored in P are these doubles). */
static long double
mixed_sum(const struct mixed_geom * gm, int maxpop, int type)
{
    long double s = 0;
    int64_t id;
    for(id = 0; id < NM_TOT; id++) {
        if(mixed_pop(id) > maxpop)
            break;
        double pos[3], m;
        int t;
        mixed_particle(id, gm, pos, &t, &m);
        if(type < 0 || t == type)
            s += m;
    }
    return s;
}

/* Bin edges as subhalo_so.c: inner edge of pass-1 bin k and the outer edge of sub-bin m of it. */
static double
mixed_edge(const struct mixed_geom * gm, int k)
{
    return gm->Rlo * exp(k / gm->Scale);
}

static double
mixed_sub(const struct mixed_geom * gm, int k, double frac)
{
    const double a = mixed_edge(gm, k), b = mixed_edge(gm, k + 1);
    return a + frac * (b - a);
}

static void
mixed_geometry(struct mixed_geom * gm, const double rho[SUBHALO_NSO], const struct SubhaloParams * par)
{
    double rhomin = rho[0];
    int d;
    for(d = 1; d < SUBHALO_NSO; d++)
        rhomin = fmin(rhomin, rho[d]);
    const double soft = FORCE_SOFTENING() / 2.8;
    const int NB = par->SubhaloSOBins;
    memset(gm, 0, sizeof(*gm));
    gm->R = par->SubhaloSOSearchFactor * cbrt(3. * NM_MBOUND / (4. * M_PI * rhomin));
    gm->Rlo = fmin(soft, 0.25 * gm->R);
    gm->Scale = NB / log(gm->R / gm->Rlo);
    const double K200c = 4. / 3. * M_PI * rho[2], Kvir = 4. / 3. * M_PI * rho[0];
    /* 200c: the bin containing 45 kpc/h; shell in sub-bin 3, crossing in sub-bin 40 */
    const int k1 = (int) floor(log(45. / gm->Rlo) * gm->Scale);
    gm->rs1 = mixed_sub(gm, k1, 3.5 / SUBHALO_SO_NSUB);
    gm->r200 = mixed_sub(gm, k1, 40.5 / SUBHALO_SO_NSUB);
    gm->rcore = 0.8 * mixed_edge(gm, k1);
    const double M1 = K200c * gm->r200 * gm->r200 * gm->r200;
    /* vir: the bin containing 1.6 r200c */
    const int k2 = (int) floor(log(1.6 * gm->r200 / gm->Rlo) * gm->Scale);
    gm->rs2a = mixed_sub(gm, k2 - 1, 0.5);
    gm->rs2b = mixed_sub(gm, k2, 3.5 / SUBHALO_SO_NSUB);
    gm->rvir = mixed_sub(gm, k2, 40.5 / SUBHALO_SO_NSUB);
    const double M2 = Kvir * gm->rvir * gm->rvir * gm->rvir;
    /* core 0.9 M1, shell1 0.1 M1; shell2a up to 0.9 M2, shell2b the rest. The mean raw masses
     * (1 + 0.3 u) (x1.7 for a third), (1 + 0.5 u), (1 + 0.4 u), (1 + 0.6 u) */
    gm->fcore = 0.9 * M1 / (NM_CORE * (1.15 * (2. / 3 + 1.7 / 3)));
    gm->fsh1 = 0.1 * M1 / (NM_SH1 * 1.25);
    gm->fsh2a = (0.9 * M2 - M1) / (NM_SH2A * 1.2);
    gm->fsh2b = 0.1 * M2 / (NM_SH2B * 1.3);
    message(0, "mixed: R %g Rlo %g bins %d; 200c bin %d shell %g crossing %g; vir bin %d shells %g %g crossing %g; "
               "M1 %g M2 %g\n", gm->R, gm->Rlo, NB, k1, gm->rs1, gm->r200, k2, gm->rs2a, gm->rs2b, gm->rvir, M1, M2);
    CHECK(k2 > k1 + 1 && gm->fsh2a > 0 && gm->rvir * 1.25 < gm->R, "mixed: geometry k1 %d k2 %d fsh2a %g\n", k1, k2, gm->fsh2a);
}

static void
mixed_setup_particles(const struct mixed_geom * gm)
{
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    particle_alloc_memory(PartManager, BOX, NM_TOT + 1000);
    PartManager->NumPart = 0;
    int d;
    for(d = 0; d < 3; d++)
        PartManager->CurrentParticleOffset[d] = Offset[d];
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t atleast[6] = {16, 0, 0, 0, 16, 16};
    slots_reserve(1, atleast, SlotsManager);
    int64_t id;
    for(id = 0; id < NM_TOT; id++) {
        /* a blocked initial layout (the domain decomposition redistributes it) */
        if((int)(id * NTask / NM_TOT) != ThisTask)
            continue;
        struct particle_data * p = &P[PartManager->NumPart++];
        memset(p, 0, sizeof(*p));
        double pos[3], mass;
        int type;
        mixed_particle(id, gm, pos, &type, &mass);
        p->ID = id;
        p->Type = type;
        p->Mass = mass;
        p->GrNr = -1;
        p->TrackId = p->BoundTrackId = -1;
        p->BindRank = -1;
        for(d = 0; d < 3; d++)
            p->Pos[d] = wrap(pos[d] + Offset[d]);
    }
    slots_setup_id(PartManager, SlotsManager);
}

struct mixed_result {
    double MSO[SUBHALO_NSO], MTVir[6], MT200c[6], Rsearch;
};

/* One call with the central on rank `owner`; results gathered on every rank. */
static void
mixed_run(DomainDecomp * dd, int owner, const Cosmology * CP, const struct SubhaloParams * par, const double * rho,
          struct mixed_result * res, struct SubhaloSOStats * st)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    struct SubTrack rec;
    subhalo_track_clear(&rec, 0);
    rec.Rank = 0;
    rec.HostGroup = 0;
    rec.Nbound = NM_TOT;
    rec.Mbound = NM_MBOUND;
    int d;
    for(d = 0; d < 3; d++)
        rec.CorePos[d] = CenM[d];
    const int n = ThisTask == owner;
    double Rs = 0;
    subhalo_field_so_full(&rec, n, dd, 1.0, CP, par, rho, &Rs, st, MPI_COMM_WORLD);
    struct mixed_result mine;
    memset(&mine, 0, sizeof(mine));
    if(n) {
        for(d = 0; d < SUBHALO_NSO; d++)
            mine.MSO[d] = rec.MSO[d];
        for(d = 0; d < 6; d++) {
            mine.MTVir[d] = rec.MTypeSOVir[d];
            mine.MT200c[d] = rec.MTypeSO200c[d];
        }
        mine.Rsearch = Rs;
    }
    MPI_Allreduce(&mine, res, sizeof(mine) / sizeof(double), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/* Checks of one partition against the references and the default-domain result (ref0 may be
 * NULL). Returns the number of failures. */
static int
mixed_check(const struct mixed_result * r, const struct mixed_result * ref0, const struct mixed_geom * gm,
            const long double Mref[SUBHALO_NSO], const long double MTref[2][6], const char * what)
{
    const double ulp = ldexp(1., -23);
    int bad = 0, d, t;
    if(fabs(r->Rsearch - gm->R) > 1e-12 * gm->R) {
        message(0, "%s: search radius %g, expected %g (no doubling)\n", what, r->Rsearch, gm->R);
        bad++;
    }
    for(d = 0; d < SUBHALO_NSO; d++) {
        if(Mref[d] > 0) {
            const double err = (double) ((r->MSO[d] - Mref[d]) / Mref[d]);
            message(0, "%s: %s MSO %.9g reference %.12Lg rel err %+.2e (bound %.2e)\n", what, DefName[d], r->MSO[d], Mref[d], err, ulp);
            if(!(fabs(err) <= ulp)) bad++;
        }
        else if(!(r->MSO[d] > 0)) {
            message(0, "%s: %s MSO %g\n", what, DefName[d], r->MSO[d]);
            bad++;
        }
        if(ref0 && !(fabs(r->MSO[d] - ref0->MSO[d]) <= 1e-6 * ref0->MSO[d])) {
            message(0, "%s: %s MSO %.9g differs from the default domain's %.9g\n", what, DefName[d], r->MSO[d], ref0->MSO[d]);
            bad++;
        }
    }
    for(d = 0; d < 2; d++) {
        const double * mt = d == 0 ? r->MTVir : r->MT200c;
        const double mso = r->MSO[d == 0 ? 0 : 2];
        double sum = 0;
        for(t = 0; t < 6; t++) {
            sum += mt[t];
            const long double ref = MTref[d][t];
            const double err = ref > 0 ? (double) fabsl((mt[t] - ref) / ref) : fabs(mt[t]);
            if(!(err <= ulp)) {
                message(0, "%s: %s type %d: %.9g reference %.12Lg rel err %.2e\n", what, d == 0 ? "vir" : "200c", t, mt[t], ref, err);
                bad++;
            }
        }
        if(!(fabs(sum - mso) <= 2 * ulp * mso)) {
            message(0, "%s: %s per-type sum %.9g != MSO %.9g\n", what, d == 0 ? "vir" : "200c", sum, mso);
            bad++;
        }
    }
    return bad;
}

static void
reverse_particles(void)
{
    int64_t i, n = PartManager->NumPart;
    for(i = 0; i < n / 2; i++) {
        struct particle_data tmp = P[i];
        P[i] = P[n - 1 - i];
        P[n - 1 - i] = tmp;
    }
}

static void
test_subhalo_field_so_mixed(void ** state)
{
    walltime_init(&CT);
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    init_forcetree_params(0.7);
    struct gravshort_tree_params tp = get_gravshort_treepar();
    tp.FractionalGravitySoftening = 1. / 50;
    set_gravshort_treepar(tp);
    gravshort_set_softenings(50.);
    subhalo_so_set_import_buffer_boost(0);     /* as in test_subhalo_field_so */

    Cosmology CP = test_cosmology();
    double rho[SUBHALO_NSO];
    test_thresholds(&CP, 1.0, rho);
    struct SubhaloParams par;
    memset(&par, 0, sizeof(par));
    par.SubhaloSOBins = 64;
    par.SubhaloSOSearchFactor = 2;
    par.SubhaloSOMaxDoublings = 3;
    par.SubhaloSORefineMinPart = 1000;

    struct mixed_geom gm;
    mixed_geometry(&gm, rho, &par);
    /* references: 200c the core and shell1, vir and 200b the whole clump */
    long double Mref[SUBHALO_NSO] = {0}, MTref[2][6];
    int t, d;
    Mref[2] = mixed_sum(&gm, 1, -1);
    Mref[0] = Mref[1] = mixed_sum(&gm, 3, -1);
    for(t = 0; t < 6; t++) {
        MTref[0][t] = mixed_sum(&gm, 3, t);
        MTref[1][t] = mixed_sum(&gm, 1, t);
    }

    mixed_setup_particles(&gm);
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    /* how the clump is spread */
    int64_t nclump = 0, i;
    for(i = 0; i < PartManager->NumPart; i++)
        nclump += P[i].ID < NM_TOT - NM_BG;
    message(1, "mixed: rank %d holds %ld of the clump's particles\n", ThisTask, nclump);

    struct mixed_result r0, r1, r2;
    struct SubhaloSOStats st;
    mixed_run(&ddecomp, 0, &CP, &par, rho, &r0, &st);
    CHECK(st.Nrefined == 1 && st.Ndoubled == 0 && st.NTypes == 3 && st.Pass2,
          "mixed: %ld refined %ld doubled %d types pass 2 %d\n", st.Nrefined, st.Ndoubled, st.NTypes, st.Pass2);
    int bad = mixed_check(&r0, NULL, &gm, Mref, MTref, "mixed default");
    CHECK(bad == 0, "mixed default domain: %d failures\n", bad);

    /* the local particle order reversed: other tree, other summation order */
    reverse_particles();
    mixed_run(&ddecomp, NTask - 1, &CP, &par, rho, &r1, &st);
    bad = mixed_check(&r1, &r0, &gm, Mref, MTref, "mixed reversed");
    CHECK(bad == 0, "mixed reversed order: %d failures\n", bad);
    subhalo_so_set_max_batch(1);
    mixed_run(&ddecomp, 0, &CP, &par, rho, &r1, &st);
    subhalo_so_set_max_batch(0);
    bad = mixed_check(&r1, &r0, &gm, Mref, MTref, "mixed reversed batch 1");
    CHECK(bad == 0, "mixed reversed order, batches of one: %d failures\n", bad);

    /* a rank emptied: its particles go to its neighbour */
    if(NTask >= 2) {
        empty_rank_domain(&ddecomp, NTask / 2);
        mixed_run(&ddecomp, NTask / 2, &CP, &par, rho, &r2, &st);
        bad = mixed_check(&r2, &r0, &gm, Mref, MTref, "mixed emptied rank");
        CHECK(bad == 0, "mixed emptied rank: %d failures\n", bad);
    }
    for(d = 0; d < SUBHALO_NSO; d++)
        message(0, "mixed: %s MSO %.9g\n", DefName[d], r0.MSO[d]);

    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_subhalo_field_so),
        cmocka_unit_test(test_subhalo_field_so_mixed),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
