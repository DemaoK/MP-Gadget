/* Unit tests of the serial unbinding kernel (subhalo_unbind.c, subhalo_props.c):
 * Phase 2a plan section 5, tests 1 to 4, plus checks of the tree potential against direct
 * summation, of the k-nearest tracer search against brute force and of the death path.
 *
 * Haloes are Hernquist (1990) spheres sampled from the exact isotropic distribution function
 * (eq. 17), positions from the inverse cumulative mass profile truncated at 100 scale radii, so
 * every particle is bound to the analytic potential; the only losses are the truncation, the
 * softening and the Monte Carlo noise of the N-body potential. a = 1, H = 0, internal units
 * kpc/h, 1e10 Msun/h, km/s (G = CP->GravInternal for those units). The halo is placed across a
 * periodic boundary to exercise the minimum-image arithmetic. Serial: mpirun -np 1. */

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_rng.h>
#include <omp.h>

#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/physconst.h>
#include <libgadget/subhalo_unbind.h>
#include <libgadget/utils/mymalloc.h>
#include "stub.h"

/* GRAVITY / UnitLength^3 * UnitMass * UnitTime^2 for kpc/h, 1e10 Msun/h, km/s (cosmology.c:19) */
#define TEST_G (6.672e-8 / pow(3.085678e21, 3) * 1.989e43 * pow(3.085678e21 / 1e5, 2))
#define BOXSIZE 20000.
#define NHALO 20000
#define HQ_M 100.        /* 1e12 Msun/h */
#define HQ_A 20.         /* kpc/h */
#define HQ_RMAX (100. * HQ_A)
#define EPS 0.5          /* Plummer-equivalent softening; h = 2.8 EPS */

static const double Centre[3] = {3.0, BOXSIZE - 5.0, 0.5 * BOXSIZE};
static const double Bulk[3] = {100., -50., 30.};

static struct SubhaloUnbindParams
default_params(void)
{
    struct SubhaloUnbindParams up;
    memset(&up, 0, sizeof(up));
    up.atime = 1;
    up.Hubble = 0;
    up.G = TEST_G;
    up.h = 2.8 * EPS;
    up.BoxSize = BOXSIZE;
    up.PotentialTypes = 1 + 2 + 16 + 32;
    up.TracerTypes = 2 + 16;
    up.MinNumPart = 20;
    up.MinNumTracerPart = 10;
    up.SampleSize = 0;      /* exact: the tree on every particle */
    up.MaxIter = 50;
    up.OpenAngle = 0.45;
    up.BoundMassPrecision = 0.995;
    up.WithThermal = 0;
    return up;
}

/* ------------------------------------------------------------------ Hernquist sampling */

/* Hernquist DF in units G = M = a = 1 up to a constant, q2 = -E in (0, 1). */
static double
hq_df(double q2)
{
    if(q2 <= 0)
        return 0;
    if(q2 >= 1)
        q2 = 1 - 1e-15;
    const double q = sqrt(q2), omq = 1 - q2;
    const double br = 3 * asin(q) + q * sqrt(omq) * (1 - 2 * q2) * (8 * q2 * q2 - 8 * q2 - 3);
    return br > 0 ? pow(omq, -2.5) * br : 0;
}

/* Speed at radius r (units a) from p(v) dv ~ v^2 f(psi - v^2 / 2) dv on [0, v_esc], by
 * rejection against 1.5 times the maximum of v^2 f, located on a geometric grid in v^2 (near
 * the centre the peak sits at v^2 ~ r). */
static double
hq_speed(double r, gsl_rng * rng)
{
    const double psi = 1. / (1. + r);
    const double vesc = sqrt(2 * psi);
    double gmax = 0;
    int k;
    for(k = 0; k <= 600; k++) {
        const double v2 = vesc * vesc * pow(10., -12. + 12. * k / 600.);
        const double g = v2 * hq_df(psi - 0.5 * v2);
        if(g > gmax) gmax = g;
    }
    gmax *= 1.5;
    for(;;) {
        const double v = vesc * gsl_rng_uniform(rng);
        if(gsl_rng_uniform(rng) * gmax < v * v * hq_df(psi - 0.5 * v * v))
            return v;
    }
}

static void
random_direction(gsl_rng * rng, double u[3])
{
    const double ct = 2 * gsl_rng_uniform(rng) - 1, st = sqrt(1 - ct * ct);
    const double ph = 2 * M_PI * gsl_rng_uniform(rng);
    u[0] = st * cos(ph);
    u[1] = st * sin(ph);
    u[2] = ct;
}

static double
wrapb(double x)
{
    while(x >= BOXSIZE) x -= BOXSIZE;
    while(x < 0) x += BOXSIZE;
    return x;
}

/* Sample n particles of a Hernquist sphere (mass M, scale a, truncated at rmax) into
 * P[first, first + n), centred on cen with bulk velocity vel. Returns the particle mass. */
static double
make_hernquist(int64_t first, int64_t n, double M, double a, double rmax, const double cen[3], const double vel[3],
               unsigned long seed, int64_t id0)
{
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    const double Xmax = pow(rmax / (rmax + a), 2);
    const double m = M * Xmax / n;
    const double vg = sqrt(TEST_G * M / a);
    int64_t i;
    int d;
    for(i = first; i < first + n; i++) {
        const double s = sqrt(Xmax * gsl_rng_uniform_pos(rng));
        const double r = s / (1 - s);
        const double v = hq_speed(r, rng);
        double ux[3], uv[3];
        random_direction(rng, ux);
        random_direction(rng, uv);
        memset(&P[i], 0, sizeof(P[i]));
        P[i].Type = 1;
        P[i].Mass = m;
        P[i].ID = id0 + (i - first);
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(cen[d] + a * r * ux[d]);
            P[i].Vel[d] = vel[d] + vg * v * uv[d];   /* a = 1: Vel = v_pec */
        }
    }
    gsl_rng_free(rng);
    return m;
}

/* ------------------------------------------------------------------ harness */

struct TestCand {
    struct SubCandidate c;
    int64_t * Index;
    double * Energy;
};

static void
cand_init(struct TestCand * t, int64_t first, int64_t n, const double ref[3], const double refvel[3], uint64_t seed)
{
    int64_t i;
    int d;
    memset(&t->c, 0, sizeof(t->c));
    t->Index = (int64_t *) mymalloc("TestIndex", sizeof(int64_t) * (n + 1));
    t->Energy = (double *) mymalloc("TestEnergy", sizeof(double) * (n + 1));
    for(i = 0; i < n; i++)
        t->Index[i] = first + i;
    t->c.Index = t->Index;
    t->c.Energy = t->Energy;
    t->c.N = n;
    t->c.TrackId = 0;
    t->c.Seed = seed;
    for(d = 0; d < 3; d++) {
        t->c.RefPos[d] = ref[d];
        t->c.RefVel[d] = refvel[d];
    }
}

static void
cand_free(struct TestCand * t)
{
    myfree(t->Energy);
    myfree(t->Index);
}

/* An arena with exactly the kernel's per-particle scratch budget: the test checks that the
 * kernel lives within SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART (Index and Energy are the test's). */
static struct SubhaloArena
arena_alloc(int64_t n)
{
    struct SubhaloArena a;
    a.size = SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART * n + 64;
    a.used = 0;
    a.base = (char *) mymalloc("TestArena", a.size);
    return a;
}

static void
setup_pman(int64_t MaxPart)
{
    particle_alloc_memory(PartManager, BOXSIZE, MaxPart);
    PartManager->NumPart = MaxPart;
}

static void
teardown_pman(void)
{
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

/* Index[0, n) is a permutation of [first, first + n). */
static int
is_permutation(const int64_t * Index, int64_t n, int64_t first)
{
    char * seen = (char *) mymalloc("TestSeen", n + 1);
    memset(seen, 0, n + 1);
    int64_t i;
    int ok = 1;
    for(i = 0; i < n; i++) {
        const int64_t k = Index[i] - first;
        if(k < 0 || k >= n || seen[k]) {
            ok = 0;
            break;
        }
        seen[k] = 1;
    }
    myfree(seen);
    return ok;
}

static int
is_sorted(const double * E, const int64_t * I, int64_t n)
{
    int64_t i;
    for(i = 1; i < n; i++)
        if(E[i] < E[i - 1] || (E[i] == E[i - 1] && I[i] < I[i - 1]))
            return 0;
    return 1;
}

static double
dist_periodic(const double x[3], const double y[3])
{
    double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double dx = NEAREST(x[d] - y[d], BOXSIZE);
        r2 += dx * dx;
    }
    return sqrt(r2);
}

/* Unbind + final pass of a fresh candidate over P[first, first + n). */
static int
run_candidate(struct TestCand * t, int64_t first, int64_t n, const double ref[3], const double vel[3], uint64_t seed,
              const struct SubhaloUnbindParams * up, struct SubhaloArena * arena, int64_t * npos)
{
    cand_init(t, first, n, ref, vel, seed);
    const int alive = subhalo_unbind(&t->c, up, arena);
    assert_int_equal(arena->used, 0);
    *npos = 0;
    if(alive) {
        /* the bound prefix is in energy order unless the subsampled loop's centre refinement
         * reordered its head (subhalo_unbind.h) */
        if(!(up->SampleSize > 0 && t->c.Nbound > up->SampleSize))
            assert_true(is_sorted(t->c.Energy, t->c.Index, t->c.Nbound));
        assert_true(is_sorted(t->c.Energy + t->c.Nbound, t->c.Index + t->c.Nbound, t->c.N - t->c.Nbound));
        int64_t i;
        for(i = 0; i < t->c.Nbound; i++)
            assert_true(t->c.Energy[i] < 0);
        for(i = t->c.Nbound; i < t->c.N; i++)
            assert_true(t->c.Energy[i] >= 0);
        const int64_t nb = t->c.Nbound;
        *npos = subhalo_final_pass(&t->c, up, arena);
        assert_int_equal(arena->used, 0);
        assert_int_equal(t->c.Nbound, nb);
        assert_true(is_sorted(t->c.Energy, t->c.Index, t->c.Nbound));
    }
    assert_true(is_permutation(t->c.Index, n, first));
    return alive;
}

/* ------------------------------------------------------------------ test 1 */

static void
test_hernquist(void ** state)
{
    setup_pman(NHALO);
    make_hernquist(0, NHALO, HQ_M, HQ_A, HQ_RMAX, Centre, Bulk, 4242, 0);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(NHALO);
    struct TestCand t;
    int64_t npos;
    const double t0 = MPI_Wtime();
    const int alive = run_candidate(&t, 0, NHALO, Centre, Bulk, 7, &up, &arena, &npos);
    message(0, "Unbind + final pass of %d particles (tree): %.3f s with %d OpenMP threads\n", NHALO, MPI_Wtime() - t0, omp_get_max_threads());
    assert_true(alive);
    const double fb = (double) t.c.Nbound / NHALO;
    message(0, "Hernquist: Nbound %ld / %d = %.4f after %d iterations; final pass E >= 0: %ld; NboundTracer %ld\n",
            (long) t.c.Nbound, NHALO, fb, t.c.Iterations, (long) npos, (long) t.c.NboundTracer);
    assert_true(fb >= 0.9);
    assert_true(npos < 0.01 * t.c.Nbound);
    assert_int_equal(t.c.NboundTracer, t.c.Nbound);
    assert_false(t.c.Flags & SUBHALO_FLAG_ITERCAP);

    /* tree against direct summation on the bound set (SampleSize >= N forces the direct sum) */
    {
        double * Ed = (double *) mymalloc("TestEd", sizeof(double) * t.c.Nbound);
        struct SubhaloUnbindParams ud = up;
        ud.SampleSize = 1 << 30;
        subhalo_energies(t.c.Index, t.c.Nbound, Ed, t.c.Index, t.c.Nbound, t.c.RefPos, t.c.RefVel, &ud, &arena);
        double maxrel = 0, sum2 = 0;
        int64_t i;
        for(i = 0; i < t.c.Nbound; i++) {
            /* compare potentials: E - K is common to both */
            double K = 0;
            int d;
            for(d = 0; d < 3; d++) {
                const double dv = P[t.c.Index[i]].Vel[d] - t.c.RefVel[d];
                K += 0.5 * dv * dv;
            }
            const double rel = fabs((t.c.Energy[i] - K) - (Ed[i] - K)) / fabs(Ed[i] - K);
            sum2 += rel * rel;
            if(rel > maxrel) maxrel = rel;
        }
        message(0, "Tree (theta 0.45) vs direct potential: rms rel err %.3g, max %.3g\n", sqrt(sum2 / t.c.Nbound), maxrel);
        assert_true(sqrt(sum2 / t.c.Nbound) < 5e-3);
        myfree(Ed);
    }

    /* core and properties */
    double cp[3], cv[3], sx, sv;
    subhalo_core(&t.c, &up, 20, cp, cv, &sx, &sv);
    const double off = dist_periodic(cp, Centre);
    double voff = 0;
    int d;
    for(d = 0; d < 3; d++)
        voff += (cv[d] - Bulk[d]) * (cv[d] - Bulk[d]);
    message(0, "Core: offset %.3f kpc/h from the true centre, SigmaX %.3f, |dv| %.2f km/s, SigmaV %.2f\n", off, sx, sqrt(voff), sv);
    assert_true(off < sx);

    struct SubTrack rec;
    memset(&rec, 0, sizeof(rec));
    const double Offset[3] = {0, 0, 0};
    subhalo_props_hbt(&rec, &t.c, &up, 20, Offset);
    const double vmax_true = sqrt(TEST_G * HQ_M / (4 * HQ_A));
    message(0, "Props: Vmax %.2f (analytic %.2f, err %.2f%%), Rmax %.2f (analytic %.1f), Ekin %.4g Epot %.4g T/|U| %.3f, Mbound %.3f\n",
            rec.Vmax, vmax_true, 100 * (rec.Vmax / vmax_true - 1), rec.Rmax, HQ_A, rec.Ekin, rec.Epot, rec.KinToPot, rec.Mbound);
    assert_true(fabs(rec.Vmax / vmax_true - 1) < 0.05);
    assert_int_equal(rec.Nbound, t.c.Nbound);
    assert_true(rec.MostBoundID == P[t.c.Index[0]].ID);
    for(d = 0; d < 3; d++) {
        assert_true(rec.CorePos[d] >= 0 && rec.CorePos[d] < BOXSIZE);
        assert_true(fabs(rec.CoMVel[d] - Bulk[d]) < 10);
    }
    /* virial ratio of an equilibrium sphere (2T ~ |W|, i.e. T/|U| ~ 0.5) */
    assert_true(rec.KinToPot > 0.4 && rec.KinToPot < 0.6);

    /* brute-force Vmax for the exactness of the histogram search */
    {
        double best = 0;
        int64_t i, j;
        const int64_t nb = t.c.Nbound;
        double * r = (double *) mymalloc("TestR", sizeof(double) * nb);
        for(i = 0; i < nb; i++) {
            r[i] = dist_periodic(P[t.c.Index[i]].Pos, cp);
            if(r[i] < EPS) r[i] = EPS;
        }
        /* O(N^2) on a strided subset of evaluation radii plus the reported Rmax */
        for(i = 0; i < nb; i += 7) {
            double M = 0;
            for(j = 0; j < nb; j++)
                if(r[j] <= r[i]) M += P[t.c.Index[j]].Mass;
            if(M / r[i] > best) best = M / r[i];
        }
        const double vb = sqrt(TEST_G * best);
        message(0, "Vmax brute force (subset of radii) %.4f <= histogram search %.4f\n", vb, rec.Vmax);
        assert_true(vb <= rec.Vmax * (1 + 1e-6));
        assert_true(vb >= rec.Vmax * (1 - 2e-3));
        myfree(r);
    }
    /* an offset initial frame (2 kpc/h and 40 km/s off) is recovered by the loop; HBT's loop
     * never re-admits a particle it removed, so a grossly wrong frame loses particles for good
     * (0.5 a and 150 km/s off keeps about 82 %): the caller's frame matters */
    {
        struct TestCand w;
        double badpos[3], badvel[3];
        for(d = 0; d < 3; d++) {
            badpos[d] = wrapb(Centre[d] + (d == 0 ? 2. : 0));
            badvel[d] = Bulk[d] + (d == 2 ? 40. : 0);
        }
        assert_true(run_candidate(&w, 0, NHALO, badpos, badvel, 7, &up, &arena, &npos));
        double vo = 0;
        for(d = 0; d < 3; d++)
            vo += (w.c.RefVel[d] - t.c.RefVel[d]) * (w.c.RefVel[d] - t.c.RefVel[d]);
        message(0, "Bad initial frame: Nbound %ld after %d iterations, final CoM %.3g kpc/h and %.3g km/s from the good run's\n",
                (long) w.c.Nbound, w.c.Iterations, dist_periodic(w.c.RefPos, t.c.RefPos), sqrt(vo));
        assert_true(labs((long) (w.c.Nbound - t.c.Nbound)) <= 0.01 * t.c.Nbound);
        assert_true(dist_periodic(w.c.RefPos, t.c.RefPos) < 0.05 * HQ_A);
        cand_free(&w);
    }
    cand_free(&t);
    myfree(arena.base);
    teardown_pman();
}

/* ------------------------------------------------------------------ test 2 */

#define NSHELL 2000

static void
test_unbound_shell(void ** state)
{
    setup_pman(NHALO + NSHELL);
    const double m = make_hernquist(0, NHALO, HQ_M, HQ_A, HQ_RMAX, Centre, Bulk, 4242, 0);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(NHALO + NSHELL);

    /* reference: the halo alone */
    struct TestCand t1;
    int64_t npos;
    assert_true(run_candidate(&t1, 0, NHALO, Centre, Bulk, 7, &up, &arena, &npos));
    const int64_t nb1 = t1.c.Nbound;
    char * bound1 = (char *) mymalloc("TestBound1", NHALO);
    memset(bound1, 0, NHALO);
    int64_t i;
    for(i = 0; i < nb1; i++)
        bound1[t1.c.Index[i]] = 1;

    /* shell at 3 a, twice the escape speed of halo + shell there, isotropic directions */
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 99);
    const double rs = 3 * HQ_A;
    const double vesc = sqrt(2 * (TEST_G * HQ_M / (rs + HQ_A) + TEST_G * NSHELL * m / rs));
    int d;
    for(i = NHALO; i < NHALO + NSHELL; i++) {
        double ux[3], uv[3];
        random_direction(rng, ux);
        random_direction(rng, uv);
        memset(&P[i], 0, sizeof(P[i]));
        P[i].Type = 1;
        P[i].Mass = m;
        P[i].ID = i;
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(Centre[d] + rs * ux[d]);
            P[i].Vel[d] = Bulk[d] + 2 * vesc * uv[d];
        }
    }
    gsl_rng_free(rng);

    struct TestCand t2;
    assert_true(run_candidate(&t2, 0, NHALO + NSHELL, Centre, Bulk, 7, &up, &arena, &npos));
    int64_t shell_bound = 0, only1 = 0, only2 = 0;
    char * bound2 = (char *) mymalloc("TestBound2", NHALO + NSHELL);
    memset(bound2, 0, NHALO + NSHELL);
    for(i = 0; i < t2.c.Nbound; i++) {
        bound2[t2.c.Index[i]] = 1;
        if(t2.c.Index[i] >= NHALO)
            shell_bound++;
    }
    for(i = 0; i < NHALO; i++) {
        if(bound1[i] && !bound2[i]) only1++;
        if(!bound1[i] && bound2[i]) only2++;
    }
    message(0, "Shell: Nbound %ld (halo alone %ld), shell particles bound %ld, halo particles bound only alone %ld, only with shell %ld, final pass E >= 0: %ld\n",
            (long) t2.c.Nbound, (long) nb1, (long) shell_bound, (long) only1, (long) only2, (long) npos);
    assert_int_equal(shell_bound, 0);
    assert_true(labs((long) (t2.c.Nbound - nb1)) <= 0.01 * nb1);
    assert_true(only1 + only2 <= 0.01 * nb1);
    assert_true(t2.c.Iterations > 1);

    /* the same with the potential subsampled (test 3's setting) */
    {
        struct TestCand t3;
        struct SubhaloUnbindParams us = up;
        us.SampleSize = 1000;
        assert_true(run_candidate(&t3, 0, NHALO + NSHELL, Centre, Bulk, 31, &us, &arena, &npos));
        int64_t sb = 0;
        for(i = 0; i < t3.c.Nbound; i++)
            if(t3.c.Index[i] >= NHALO)
                sb++;
        message(0, "Shell, subsampled 1000: Nbound %ld, shell particles bound %ld, iterations %d, final pass E >= 0: %ld\n",
                (long) t3.c.Nbound, (long) sb, t3.c.Iterations, (long) npos);
        assert_int_equal(sb, 0);
        assert_true(labs((long) (t3.c.Nbound - nb1)) <= 0.02 * nb1);
        cand_free(&t3);
    }
    /* iteration cap: one iteration, flagged, accepted as is */
    {
        struct TestCand t4;
        struct SubhaloUnbindParams uc = up;
        uc.MaxIter = 1;
        cand_init(&t4, 0, NHALO + NSHELL, Centre, Bulk, 7);
        assert_true(subhalo_unbind(&t4.c, &uc, &arena));
        message(0, "Iteration cap 1: Nbound %ld, Flags %d\n", (long) t4.c.Nbound, t4.c.Flags);
        assert_int_equal(t4.c.Iterations, 1);
        assert_true(t4.c.Flags & SUBHALO_FLAG_ITERCAP);
        assert_true(is_sorted(t4.c.Energy, t4.c.Index, t4.c.Nbound));
        cand_free(&t4);
    }
    myfree(bound2);
    cand_free(&t2);
    myfree(bound1);
    cand_free(&t1);
    myfree(arena.base);
    teardown_pman();
}

/* ------------------------------------------------------------------ test 3 */

static void
test_subsampling(void ** state)
{
    setup_pman(NHALO);
    make_hernquist(0, NHALO, HQ_M, HQ_A, HQ_RMAX, Centre, Bulk, 4242, 0);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(NHALO);
    struct TestCand t0;
    int64_t npos;
    assert_true(run_candidate(&t0, 0, NHALO, Centre, Bulk, 7, &up, &arena, &npos));
    const int64_t nb0 = t0.c.Nbound;

    struct SubhaloUnbindParams us = up;
    us.SampleSize = 1000;
    struct TestCand a, b, c;
    cand_init(&a, 0, NHALO, Centre, Bulk, 12345);
    cand_init(&b, 0, NHALO, Centre, Bulk, 12345);
    cand_init(&c, 0, NHALO, Centre, Bulk, 54321);
    assert_true(subhalo_unbind(&a.c, &us, &arena));
    assert_true(subhalo_unbind(&b.c, &us, &arena));
    assert_true(subhalo_unbind(&c.c, &us, &arena));
    assert_int_equal(arena.used, 0);
    assert_int_equal(a.c.Nbound, b.c.Nbound);
    assert_memory_equal(a.Index, b.Index, sizeof(int64_t) * NHALO);
    assert_memory_equal(a.Energy, b.Energy, sizeof(double) * NHALO);
    assert_memory_equal(a.c.RefPos, b.c.RefPos, sizeof(double) * 3);
    const int differs = memcmp(a.Index, c.Index, sizeof(int64_t) * NHALO) != 0;
    assert_true(differs);
    assert_true(is_permutation(a.Index, NHALO, 0));
    message(0, "Subsampled (1000): Nbound %ld (seed 12345, twice identical) and %ld (seed 54321), exact %ld; iterations %d\n",
            (long) a.c.Nbound, (long) c.c.Nbound, (long) nb0, a.c.Iterations);
    assert_true(labs((long) (a.c.Nbound - nb0)) <= 0.02 * nb0);
    assert_true(labs((long) (c.c.Nbound - nb0)) <= 0.02 * nb0);
    /* refinement: the most bound particle agrees with the exact run's */
    const int64_t np = subhalo_final_pass(&a.c, &us, &arena);
    message(0, "Subsampled final pass: E >= 0 for %ld; most bound %ld (exact run %ld)\n", (long) np, (long) a.Index[0], (long) t0.Index[0]);
    assert_true(np < 0.01 * a.c.Nbound);
    cand_free(&c);
    cand_free(&b);
    cand_free(&a);
    cand_free(&t0);
    myfree(arena.base);
    teardown_pman();
}

/* ------------------------------------------------------------------ test 4 */

#define NSAT 2000

static void
test_satellite(void ** state)
{
    setup_pman(NHALO + NSAT);
    make_hernquist(0, NHALO, HQ_M, HQ_A, HQ_RMAX, Centre, Bulk, 4242, 0);
    const double Ms = HQ_M / 10, as = HQ_A / 4;
    double scen[3], svel[3];
    int d;
    const double vc = sqrt(TEST_G * HQ_M * 4. / 9. / (2 * HQ_A));   /* circular speed at 2 a */
    for(d = 0; d < 3; d++) {
        scen[d] = wrapb(Centre[d] + (d == 0 ? 2 * HQ_A : 0));
        svel[d] = Bulk[d] + (d == 1 ? vc : 0);
    }
    make_hernquist(NHALO, NSAT, Ms, as, 30 * as, scen, svel, 777, NHALO);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(NHALO + NSAT);
    int64_t npos, i;

    /* host alone, for reference */
    struct TestCand h0;
    assert_true(run_candidate(&h0, 0, NHALO, Centre, Bulk, 7, &up, &arena, &npos));

    /* the satellite: its own particles plus the host particles within 15 kpc/h of its centre
     * (inside its Jacobi radius, about 17 kpc/h), in its own frame */
    struct TestCand s;
    int64_t nin = 0;
    for(i = 0; i < NHALO; i++)
        if(dist_periodic(P[i].Pos, scen) < 15.)
            nin++;
    cand_init(&s, NHALO, NSAT + nin, scen, svel, 8);
    nin = 0;
    for(i = 0; i < NHALO; i++)
        if(dist_periodic(P[i].Pos, scen) < 15.)
            s.Index[NSAT + nin++] = i;
    assert_true(subhalo_unbind(&s.c, &up, &arena));
    npos = subhalo_final_pass(&s.c, &up, &arena);
    int64_t sat_own = 0;
    for(i = 0; i < s.c.Nbound; i++)
        if(s.c.Index[i] >= NHALO)
            sat_own++;
    const int64_t captured = s.c.Nbound - sat_own;
    const int64_t nrem = s.c.N - s.c.Nbound;
    message(0, "Satellite (+%ld host particles in its source): Nbound %ld, own %ld / %d, captured host particles %ld, remainder %ld, final pass E >= 0: %ld\n",
            (long) nin, (long) s.c.Nbound, (long) sat_own, NSAT, (long) captured, (long) nrem, (long) npos);
    assert_true(sat_own > 0.8 * NSAT);
    assert_true(nrem > 0.5 * nin);

    /* the host: its particles not in the satellite's source, plus the satellite's whole
     * unbound remainder appended (post-order unbinding) */
    char * insat = (char *) mymalloc("TestInSat", NHALO + NSAT);
    memset(insat, 0, NHALO + NSAT);
    for(i = 0; i < s.c.N; i++)
        insat[s.c.Index[i]] = 1;
    struct TestCand h;
    cand_init(&h, 0, NHALO - nin + nrem, Centre, Bulk, 9);
    int64_t k = 0;
    for(i = 0; i < NHALO; i++)
        if(!insat[i])
            h.Index[k++] = i;
    for(i = 0; i < nrem; i++)
        h.Index[k++] = s.c.Index[s.c.Nbound + i];
    assert_int_equal(k, h.c.N);
    assert_true(subhalo_unbind(&h.c, &up, &arena));
    npos = subhalo_final_pass(&h.c, &up, &arena);
    int64_t host_bound = 0, sat_in_host = 0;
    for(i = 0; i < h.c.Nbound; i++) {
        if(h.c.Index[i] < NHALO)
            host_bound++;
        else
            sat_in_host++;
    }
    message(0, "Host + remainder: Nbound %ld (host particles %ld + captured %ld = %ld; host alone %ld; satellite particles %ld), final pass E >= 0: %ld\n",
            (long) h.c.Nbound, (long) host_bound, (long) captured, (long) (host_bound + captured), (long) h0.c.Nbound,
            (long) sat_in_host, (long) npos);
    assert_true(labs((long) (host_bound + captured - h0.c.Nbound)) <= 0.01 * h0.c.Nbound);

    /* sinking: satellite core far from the host core in phase space */
    double sp[3], sv[3], ssx, ssv, hp[3], hv[3], hsx, hsv;
    subhalo_core(&s.c, &up, 20, sp, sv, &ssx, &ssv);
    subhalo_core(&h.c, &up, 20, hp, hv, &hsx, &hsv);
    message(0, "Cores: satellite at %.2f kpc/h from the host core, host SigmaX %.3f SigmaV %.2f, satellite SigmaX %.3f SigmaV %.2f\n",
            dist_periodic(sp, hp), hsx, hsv, ssx, ssv);
    assert_int_equal(subhalo_sink_test(sp, sv, hp, hv, hsx, hsv, ssx, ssv, 0, BOXSIZE), 0);
    assert_int_equal(subhalo_sink_test(sp, sv, hp, hv, hsx, hsv, ssx, ssv, 1, BOXSIZE), 0);

    /* move the satellite rigidly onto the host core with the host core velocity */
    for(i = NHALO; i < NHALO + NSAT; i++)
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(P[i].Pos[d] + NEAREST(hp[d] - sp[d], BOXSIZE));
            P[i].Vel[d] += hv[d] - sv[d];
        }
    subhalo_core(&s.c, &up, 20, sp, sv, &ssx, &ssv);
    message(0, "Moved satellite: core offset %.3g, |dv| %.3g\n", dist_periodic(sp, hp),
            sqrt((sv[0] - hv[0]) * (sv[0] - hv[0]) + (sv[1] - hv[1]) * (sv[1] - hv[1]) + (sv[2] - hv[2]) * (sv[2] - hv[2])));
    assert_int_equal(subhalo_sink_test(sp, sv, hp, hv, hsx, hsv, ssx, ssv, 0, BOXSIZE), 1);
    assert_int_equal(subhalo_sink_test(sp, sv, hp, hv, hsx, hsv, ssx, ssv, 1, BOXSIZE), 1);

    cand_free(&h);
    myfree(insat);
    cand_free(&s);
    cand_free(&h0);
    myfree(arena.base);
    teardown_pman();
}

/* ------------------------------------------------------------------ extra checks */

/* The same physical system at a = 0.5 with H = 0.2 (comoving positions x / a, softening h / a,
 * Vel = a (v - H a dx)) has the same energies: checks the G / a, Vel / a and H a dx terms. */
static void
test_scale_factor(void ** state)
{
    const int64_t n = 3000;
    setup_pman(n);
    make_hernquist(0, n, 10., 5., 100., Centre, Bulk, 21, 0);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(n);
    struct TestCand t;
    cand_init(&t, 0, n, Centre, Bulk, 1);
    double * E1 = (double *) mymalloc("TestE1", sizeof(double) * n);
    subhalo_energies(t.Index, n, E1, t.Index, n, Centre, Bulk, &up, &arena);

    const double a = 0.5, H = 0.2;
    struct SubhaloUnbindParams ua = up;
    ua.atime = a;
    ua.Hubble = H;
    ua.h = up.h / a;
    int64_t i;
    int d;
    double refvel[3];
    for(d = 0; d < 3; d++)
        refvel[d] = a * Bulk[d];
    for(i = 0; i < n; i++)
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[i].Pos[d] - Centre[d], BOXSIZE) / a;   /* comoving */
            P[i].Pos[d] = wrapb(Centre[d] + dx);
            P[i].Vel[d] = a * (P[i].Vel[d] - H * a * dx);
        }
    double * E2 = (double *) mymalloc("TestE2", sizeof(double) * n);
    subhalo_energies(t.Index, n, E2, t.Index, n, Centre, refvel, &ua, &arena);
    double maxrel = 0;
    for(i = 0; i < n; i++) {
        const double rel = fabs(E2[i] - E1[i]) / fabs(E1[i]);
        if(rel > maxrel) maxrel = rel;
    }
    message(0, "a = 0.5, H = 0.2 against a = 1, H = 0: max relative energy difference %.3g\n", maxrel);
    assert_true(maxrel < 1e-9);
    myfree(E2);
    myfree(E1);
    cand_free(&t);
    myfree(arena.base);
    teardown_pman();
}


/* Thermal energy (WithThermal): gas energies gain exactly u = Entropy / (gamma - 1) (rho / a^3)^(gamma - 1)
 * (petaio.c GTInternalEnergy), and hot gas is unbound while cold gas and dark matter stay. */
static void
test_thermal(void ** state)
{
    const int64_t n = 4000;
    particle_alloc_memory(PartManager, BOXSIZE, n);
    PartManager->NumPart = n;
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    int64_t ngas = 0, i;
    for(i = 0; i < n; i++)
        if(i % 4 == 0)
            ngas++;
    int64_t atleast[6] = {ngas, 0, 0, 0, 0, 0};
    slots_reserve(1, atleast, SlotsManager);
    SlotsManager->info[0].size = ngas;
    make_hernquist(0, n, 10., 5., 100., Centre, Bulk, 33, 0);
    int64_t g = 0;
    for(i = 0; i < n; i++) {
        if(i % 4)
            continue;
        P[i].Type = 0;
        P[i].PI = g;
        const double u = (g % 2) ? 1e6 : 1.;   /* hot: sqrt(2 u) = 1414 km/s, far above escape */
        SphP[g].Density = 2.0;
        SphP[g].Entropy = u * GAMMA_MINUS1 / pow(2.0, GAMMA_MINUS1);
        g++;
    }
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(n);
    struct TestCand t;
    cand_init(&t, 0, n, Centre, Bulk, 1);
    double * E0 = (double *) mymalloc("TestE0", sizeof(double) * n);
    double * E1 = (double *) mymalloc("TestE1", sizeof(double) * n);
    subhalo_energies(t.Index, n, E0, t.Index, n, Centre, Bulk, &up, &arena);
    struct SubhaloUnbindParams ut = up;
    ut.WithThermal = 1;
    subhalo_energies(t.Index, n, E1, t.Index, n, Centre, Bulk, &ut, &arena);
    double maxerr = 0;
    for(i = 0; i < n; i++) {
        const double u = P[i].Type == 0 ? ((P[i].PI % 2) ? 1e6 : 1.) : 0;
        const double err = fabs(E1[i] - E0[i] - u) / (u > 0 ? u : 1);
        if(err > maxerr) maxerr = err;
    }
    message(0, "Thermal: max relative error of E(thermal) - E = u: %.3g\n", maxerr);
    assert_true(maxerr < 1e-9);
    assert_true(subhalo_unbind(&t.c, &ut, &arena));
    subhalo_final_pass(&t.c, &ut, &arena);
    int64_t hot = 0, cold = 0, dm = 0;
    for(i = 0; i < t.c.Nbound; i++) {
        const int64_t p = t.c.Index[i];
        if(P[p].Type != 0) dm++;
        else if(P[p].PI % 2) hot++;
        else cold++;
    }
    message(0, "Thermal unbinding: bound DM %ld / %ld, cold gas %ld / %ld, hot gas %ld / %ld\n",
            (long) dm, (long) (n - ngas), (long) cold, (long) ((ngas + 1) / 2), (long) hot, (long) (ngas / 2));
    assert_int_equal(hot, 0);
    assert_true(cold > 0.95 * ((ngas + 1) / 2));
    assert_true(dm > 0.95 * (n - ngas));
    myfree(E1);
    myfree(E0);
    cand_free(&t);
    myfree(arena.base);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

/* k nearest tracers against brute force, across the periodic boundary. */
static void
test_knn(void ** state)
{
    const int64_t nt = 3000, nq = 500;
    const int k = 10;
    setup_pman(nt + nq);
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 5);
    int64_t i, j;
    int d;
    for(i = 0; i < nt + nq; i++) {
        memset(&P[i], 0, sizeof(P[i]));
        P[i].Type = i < nt ? 1 : 0;
        P[i].Mass = 1;
        P[i].ID = i;
        for(d = 0; d < 3; d++)
            P[i].Pos[d] = wrapb(Centre[d] + 200 * (gsl_rng_uniform(rng) - 0.5));
    }
    gsl_rng_free(rng);
    int64_t * tr = (int64_t *) mymalloc("TestTr", sizeof(int64_t) * nt);
    int64_t * q = (int64_t *) mymalloc("TestQ", sizeof(int64_t) * nq);
    int64_t * out = (int64_t *) mymalloc("TestOut", sizeof(int64_t) * nq * k);
    for(i = 0; i < nt; i++)
        tr[i] = i;
    for(i = 0; i < nq; i++)
        q[i] = nt + i;
    /* arena: 48 bytes per tracer plus nodes */
    struct SubhaloArena arena;
    arena.size = 128 * nt + 64;
    arena.used = 0;
    arena.base = (char *) mymalloc("TestArena", arena.size);
    subhalo_knn_tracers(q, nq, tr, nt, k, out, BOXSIZE, &arena);
    assert_int_equal(arena.used, 0);
    int64_t bad = 0;
    for(i = 0; i < nq; i++) {
        /* brute force: the k smallest (d2, index) */
        double bd[10];
        int64_t bi[10];
        int nb = 0, a;
        for(j = 0; j < nt; j++) {
            double r2 = 0;
            for(d = 0; d < 3; d++) {
                const double dx = NEAREST(P[q[i]].Pos[d] - P[j].Pos[d], BOXSIZE);
                r2 += dx * dx;
            }
            if(nb == k && !(r2 < bd[k - 1] || (r2 == bd[k - 1] && j < bi[k - 1])))
                continue;
            a = nb < k ? nb++ : k - 1;
            while(a > 0 && (r2 < bd[a - 1] || (r2 == bd[a - 1] && j < bi[a - 1]))) {
                bd[a] = bd[a - 1];
                bi[a] = bi[a - 1];
                a--;
            }
            bd[a] = r2;
            bi[a] = j;
        }
        for(a = 0; a < k; a++)
            if(out[i * k + a] != bi[a])
                bad++;
    }
    message(0, "kNN: %ld mismatches against brute force over %ld queries x %d\n", (long) bad, (long) nq, k);
    assert_int_equal(bad, 0);
    myfree(arena.base);
    myfree(out);
    myfree(q);
    myfree(tr);
    teardown_pman();
}

/* Death: a hot cloud with no bound core, and a source below MinNumPart. */
static void
test_death(void ** state)
{
    const int64_t n = 500;
    setup_pman(n);
    make_hernquist(0, n, 0.01, HQ_A, HQ_RMAX, Centre, Bulk, 11, 0);
    int64_t i;
    int d;
    for(i = 0; i < n; i++)
        for(d = 0; d < 3; d++)
            P[i].Vel[d] = Bulk[d] + 1000. * (d == i % 3 ? 1 : 0) * (i % 2 ? 1 : -1);
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = arena_alloc(n);
    struct TestCand t;
    cand_init(&t, 0, n, Centre, Bulk, 1);
    assert_int_equal(subhalo_unbind(&t.c, &up, &arena), 0);
    assert_int_equal(t.c.Nbound, 0);
    assert_true(is_permutation(t.Index, n, 0));
    cand_free(&t);
    cand_init(&t, 0, 15, Centre, Bulk, 1);
    assert_int_equal(subhalo_unbind(&t.c, &up, &arena), 0);
    assert_int_equal(arena.used, 0);
    cand_free(&t);
    myfree(arena.base);
    teardown_pman();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_hernquist),
        cmocka_unit_test(test_unbound_shell),
        cmocka_unit_test(test_subsampling),
        cmocka_unit_test(test_satellite),
        cmocka_unit_test(test_scale_factor),
        cmocka_unit_test(test_thermal),
        cmocka_unit_test(test_knn),
        cmocka_unit_test(test_death),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
