/* Unit tests of the Phase 3 subhalo properties (subhalo_props.c): Phase 3 plan section 5,
 * tests 1 to 4, plus the group-C baryonic sums.
 *
 *  1. A sampled NFW sphere (c200c = 10, 50000 particles, truncated at 3 R200c): bound SO masses
 *     and radii of all five definitions, VmaxVir, RsKlypin and Rhalf against the analytic
 *     profile (the expected values are computed in the test from the NFW formulas).
 *  2. A triaxial Gaussian (axis ratios 0.7 and 0.5, rotated axes, 20000 particles): the Allgood
 *     shapes at Rvir and R500c and the inertia tensor against the input ratios and major axis.
 *  3. A thin disc in solid-body rotation in a tilted plane with a dense non-rotating central clump
 *     (the core): J and JVir against Omega R^2 / 2 M_disc / M_total along the normal, both spins
 *     finite and positive (SpinBullock against the formula with the analytic j, SpinPeebles
 *     against the HBT energies), Xoff/Voff near zero; then with the clump displaced in position
 *     and velocity: Xoff and Voff equal to the known shift times M_disc / M_total.
 *  4. subhalo_so_thresholds at a = 1 and a = 0.1 against the formulas of design 7.10.
 *  5. subhalo_props_baryons on a handful of gas, star and BH particles.
 *  7. Units (Phase 3 audit P4): the disc of test 3 at a = 0.5 with H(a) set, in kpc/h and km/s
 *     and again in Mpc/h with a 2 km/s velocity unit (the same physical system): J and JVir in
 *     physical kpc/h km/s, the spins, Vmax and the SO masses identical.
 *  6. The HBT Vmax / Rmax of subhalo_props_hbt against a sorted cumulative-mass oracle: the
 *     multi-peak profile of the Phase 2a audit (J9; the former bounded histogram search lost
 *     its maximum), a flat profile of equal-radius pairs (Rmax is the smallest radius attaining
 *     the maximum, as HBT's max_element), and random masses on a few shared radii.
 *
 *  8. The Phase 5 deferred columns (subhalo_props_deferred): the fitted NFW Rs of the sampled
 *     sphere of test 1 within 15 % of its scale radius; on the disc of test 3 BulkVel against the
 *     sample mean and Bulk, Vrms against the sample dispersion and sqrt(frac / 2) DISC_VEDGE,
 *     RockstarCoreVel against the mean inside 0.1 Rvir about CorePos (centred and displaced
 *     clump); -1 without a crossing (test_absent); unit invariance (test_units).
 *  9. Phase 5 audit fixes: R6, subhalo_nfw_fit returns -1 on repeated edges at the start radius
 *     (zero probe step), nonfinite edges (test_nfw_fit); R5, the bin edge's interpolation in the
 *     run's length unit (the audit's 0.01/0.02/0.03 Mpc/h example: 20 kpc/h, not Rockstar's
 *     literal 24.995) and an overshooting-bin oracle for Rs (test_nfw_bin_edges).
 *
 * Every candidate goes through subhalo_final_pass first (energies and order as in production),
 * then subhalo_props_hbt, subhalo_props_rockstar (which also fills the Phase 5 columns) and
 * subhalo_props_deferred (the standalone entry, bit-identical), all with an arena of exactly
 * SUBHALO_PROPS_BYTES_PER_PART per particle, which checks their scratch budget. Internal units
 * kpc/h, 1e10 Msun/h, km/s, a = 1, H = 0 (the Hubble term does not enter these properties
 * beyond the energies). Haloes are placed across a periodic boundary. Serial: mpirun -np 1. */

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
#include <gsl/gsl_randist.h>
#include <gsl/gsl_eigen.h>

#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/cosmology.h>
#include <libgadget/subhalo.h>
#include <libgadget/subhalo_unbind.h>
#include <libgadget/utils/mymalloc.h>
#include "stub.h"

/* GRAVITY / UnitLength^3 * UnitMass * UnitTime^2 for kpc/h, 1e10 Msun/h, km/s (cosmology.c:19) */
#define TEST_G (6.672e-8 / pow(3.085678e21, 3) * 1.989e43 * pow(3.085678e21 / 1e5, 2))
#define BOXSIZE 20000.
#define EPS 0.2          /* Plummer-equivalent softening; h = 2.8 EPS */
#define CORESIZE 20

static const double Centre[3] = {3.0, BOXSIZE - 5.0, 0.5 * BOXSIZE};
static const double Bulk[3] = {100., -50., 30.};
/* an orthonormal triad (u1 x u2 = -u3) for the tilted test systems */
static const double U1[3] = {1. / 3, 2. / 3, 2. / 3};
static const double U2[3] = {2. / 3, 1. / 3, -2. / 3};
static const double U3[3] = {2. / 3, -2. / 3, 1. / 3};

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
    up.SampleSize = 1000;
    up.MaxIter = 50;
    up.OpenAngle = 0.45;
    up.BoundMassPrecision = 0.995;
    up.WithThermal = 0;
    return up;
}

/* A flat LCDM cosmology without radiation, Omega0 = 0.3, h = 0.7, in the test units. */
static void
setup_cosmology(Cosmology * CP)
{
    memset(CP, 0, sizeof(*CP));
    CP->CMBTemperature = 2.7255;
    CP->Omega0 = 0.3;
    CP->OmegaLambda = 0.7;
    CP->OmegaBaryon = 0.045;
    CP->HubbleParam = 0.7;
    CP->RadiationOn = 0;
    CP->Omega_fld = 0;
    CP->w0_fld = -1;
    CP->wa_fld = 0;
    CP->Omega_ur = 0;
    CP->MNu[0] = CP->MNu[1] = CP->MNu[2] = 0;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(CP, 0.01, units);
}

static double
wrapb(double x)
{
    while(x >= BOXSIZE) x -= BOXSIZE;
    while(x < 0) x += BOXSIZE;
    return x;
}

static void
setup_pman(int64_t MaxPart)
{
    particle_alloc_memory(PartManager, BOXSIZE, MaxPart);
    PartManager->NumPart = MaxPart;
    memset(P, 0, sizeof(P[0]) * MaxPart);
}

static void
teardown_pman(void)
{
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

/* ------------------------------------------------------------------ harness */

/* Final pass, HBT and Rockstar properties of the candidate P[0, n) with every particle bound.
 * The final pass gets the kernel's scratch budget, subhalo_props_rockstar exactly its own. */
static void
run_props_up(int64_t n, const double ref[3], const double rho[SUBHALO_NSO], struct SubTrack * rec, int64_t * Index, double * Energy,
             struct SubhaloUnbindParams up);

static void
run_props(int64_t n, const double ref[3], const double rho[SUBHALO_NSO], struct SubTrack * rec, int64_t * Index, double * Energy)
{
    run_props_up(n, ref, rho, rec, Index, Energy, default_params());
}

static void
run_props_up(int64_t n, const double ref[3], const double rho[SUBHALO_NSO], struct SubTrack * rec, int64_t * Index, double * Energy,
             struct SubhaloUnbindParams up)
{
    struct SubCandidate c;
    memset(&c, 0, sizeof(c));
    int64_t i;
    int d;
    for(i = 0; i < n; i++)
        Index[i] = i;
    c.Index = Index;
    c.Energy = Energy;
    c.N = c.Nbound = n;
    for(d = 0; d < 3; d++) {
        c.RefPos[d] = ref[d];
        c.RefVel[d] = 0;
    }
    struct SubhaloArena arena;
    arena.size = SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART * n + 64;
    arena.used = 0;
    arena.base = (char *) mymalloc("TestArena", arena.size);
    subhalo_final_pass(&c, &up, &arena);
    assert_int_equal(arena.used, 0);
    myfree(arena.base);

    subhalo_track_clear(rec, 0);
    const double Offset[3] = {0, 0, 0};
    arena.size = SUBHALO_PROPS_BYTES_PER_PART * n;
    arena.used = 0;
    arena.base = (char *) mymalloc("TestArena", arena.size);
    subhalo_props_hbt(rec, &c, &up, CORESIZE, Offset, &arena);
    assert_int_equal(arena.used, 0);
    subhalo_props_rockstar(rec, &c, &up, rho, CORESIZE, Offset, &arena);
    assert_int_equal(arena.used, 0);
    /* subhalo_props_rockstar fills the Phase 5 columns on its radial order; the standalone
     * subhalo_props_deferred must give the same bits */
    struct SubTrack r2 = *rec;
    r2.Rs = r2.Vrms = -7;
    subhalo_props_deferred(&r2, &c, &up, rho, CORESIZE, &arena);
    assert_int_equal(arena.used, 0);
    assert_true(r2.Rs == rec->Rs && r2.Vrms == rec->Vrms);
    for(d = 0; d < 3; d++)
        assert_true(r2.BulkVel[d] == rec->BulkVel[d] && r2.RockstarCoreVel[d] == rec->RockstarCoreVel[d]);
    myfree(arena.base);
}

static double
dot3(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* ------------------------------------------------------------------ 1. NFW */

static double
nfw_m(double x)
{
    return log1p(x) - x / (1 + x);
}

#define NFW_C 10.
#define NFW_M200 100.    /* 1e12 Msun/h */
#define NFW_XT 30.       /* truncation radius in units of rs: 3 R200c */
#define NFW_N 50000
#define JEANS_NG 600

struct NFW {
    double M200, R200, rs, Mt;
    double lr[JEANS_NG], ls2[JEANS_NG];   /* log r, log sigma_r^2 of the truncated profile */
};

static double
nfw_mass(const struct NFW * h, double r)
{
    if(r > NFW_XT * h->rs)
        r = NFW_XT * h->rs;
    return h->M200 * nfw_m(r / h->rs) / nfw_m(NFW_C);
}

static double
nfw_rho(const struct NFW * h, double r)
{
    const double x = r / h->rs;
    return h->M200 / (4 * M_PI * h->rs * h->rs * h->rs * nfw_m(NFW_C)) / (x * (1 + x) * (1 + x));
}

/* Isotropic Jeans dispersion of the truncated sphere: rho sigma_r^2 = int_r^rt rho G M / r'^2 dr',
 * tabulated on a log grid by the trapezoid rule from the truncation radius inwards. */
static void
nfw_jeans_table(struct NFW * h)
{
    const double lmin = log(1e-3 * h->rs), lmax = log(NFW_XT * h->rs);
    double integ = 0;
    int k;
    for(k = JEANS_NG - 1; k >= 0; k--) {
        const double lr = lmin + (lmax - lmin) * k / (JEANS_NG - 1);
        const double r = exp(lr);
        if(k < JEANS_NG - 1) {
            const double r1 = exp(h->lr[k + 1]);
            const double f0 = nfw_rho(h, r) * TEST_G * nfw_mass(h, r) / (r * r) * r;
            const double f1 = nfw_rho(h, r1) * TEST_G * nfw_mass(h, r1) / (r1 * r1) * r1;
            integ += 0.5 * (f0 + f1) * (h->lr[k + 1] - lr);
        }
        h->lr[k] = lr;
        const double s2 = integ / nfw_rho(h, r);
        h->ls2[k] = log(s2 > 1e-30 ? s2 : 1e-30);
    }
}

static double
nfw_sigma(const struct NFW * h, double r)
{
    const double lmin = h->lr[0], lmax = h->lr[JEANS_NG - 1];
    double u = (log(r) - lmin) / (lmax - lmin) * (JEANS_NG - 1);
    if(u < 0) u = 0;
    if(u > JEANS_NG - 1.001) u = JEANS_NG - 1.001;
    const int k = (int) u;
    const double f = u - k;
    return sqrt(exp(h->ls2[k] * (1 - f) + h->ls2[k + 1] * f));
}

/* Radius (bisection) where the analytic mean enclosed density equals rho. */
static double
nfw_so_radius(const struct NFW * h, double rho)
{
    double lo = 1e-3 * h->rs, hi = NFW_XT * h->rs;
    int it;
    for(it = 0; it < 200; it++) {
        const double mid = 0.5 * (lo + hi);
        if(nfw_mass(h, mid) / (4. / 3 * M_PI * mid * mid * mid) > rho)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

static void
test_nfw(void ** state)
{
    Cosmology CP;
    setup_cosmology(&CP);
    double rho[SUBHALO_NSO];
    subhalo_so_thresholds(&CP, 1.0, rho);
    struct NFW h;
    h.M200 = NFW_M200;
    h.R200 = cbrt(3 * NFW_M200 / (4 * M_PI * rho[2]));
    h.rs = h.R200 / NFW_C;
    h.Mt = h.M200 * nfw_m(NFW_XT) / nfw_m(NFW_C);
    nfw_jeans_table(&h);

    const int64_t n = NFW_N;
    setup_pman(n);
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 4242);
    const double m = h.Mt / n;
    int64_t i;
    int d, s;
    for(i = 0; i < n; i++) {
        /* inverse cumulative mass by bisection in x = r / rs */
        const double target = gsl_rng_uniform_pos(rng) * nfw_m(NFW_XT);
        double lo = 0, hi = NFW_XT;
        int it;
        for(it = 0; it < 60; it++) {
            const double mid = 0.5 * (lo + hi);
            if(nfw_m(mid) < target) lo = mid;
            else hi = mid;
        }
        const double r = 0.5 * (lo + hi) * h.rs;
        const double sig = nfw_sigma(&h, r);
        const double ct = 2 * gsl_rng_uniform(rng) - 1, st = sqrt(1 - ct * ct), ph = 2 * M_PI * gsl_rng_uniform(rng);
        const double u[3] = {st * cos(ph), st * sin(ph), ct};
        P[i].Type = 1;
        P[i].Mass = m;
        P[i].ID = i + 1;
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(Centre[d] + r * u[d]);
            P[i].Vel[d] = Bulk[d] + gsl_ran_gaussian(rng, sig);
        }
    }
    gsl_rng_free(rng);

    int64_t * Index = (int64_t *) mymalloc("Index", sizeof(int64_t) * n);
    double * Energy = (double *) mymalloc("Energy", sizeof(double) * n);
    struct SubTrack rec;
    run_props(n, Centre, rho, &rec, Index, Energy);

    /* SO masses and radii (all five definitions lie inside the truncation radius) */
    const char * name[SUBHALO_NSO] = {"vir", "200b", "200c", "500c", "2500c"};
    for(s = 0; s < SUBHALO_NSO; s++) {
        const double R = nfw_so_radius(&h, rho[s]);
        const double M = nfw_mass(&h, R);
        message(0, "NFW %-5s: MBoundSO %.4f expected %.4f (%+.2f%%), RBoundSO %.3f expected %.3f (%+.2f%%)\n",
                name[s], rec.MBoundSO[s], M, 100 * (rec.MBoundSO[s] / M - 1), rec.RBoundSO[s], R, 100 * (rec.RBoundSO[s] / R - 1));
        assert_true(fabs(rec.MBoundSO[s] / M - 1) < 0.02);
        assert_true(fabs(rec.RBoundSO[s] / R - 1) < 0.01);
    }
    /* Vmax: max_x m(x) / x at x = 2.16258 */
    double best = 0;
    for(i = 1; i < 200000; i++) {
        const double x = 1e-5 * i;
        if(nfw_m(x) / x > best) best = nfw_m(x) / x;
    }
    const double Vmax = sqrt(TEST_G * h.M200 / (nfw_m(NFW_C) * h.rs) * best);
    const double Mhalf = 0.5 * h.Mt;
    double lo = 0, hi = NFW_XT * h.rs;
    for(i = 0; i < 200; i++) {
        const double mid = 0.5 * (lo + hi);
        if(nfw_mass(&h, mid) < Mhalf) lo = mid;
        else hi = mid;
    }
    const double Rhalf = 0.5 * (lo + hi);
    message(0, "NFW VmaxVir %.2f expected %.2f (%+.2f%%), RmaxVir %.2f (2.1626 rs = %.2f), RsKlypin %.3f rs %.3f (%+.1f%%), "
               "Rhalf %.2f expected %.2f (%+.2f%%), Renclose %.1f (rt %.1f)\n",
            rec.VmaxVir, Vmax, 100 * (rec.VmaxVir / Vmax - 1), rec.RmaxVir, 2.1626 * h.rs, rec.RsKlypin, h.rs,
            100 * (rec.RsKlypin / h.rs - 1), rec.Rhalf, Rhalf, 100 * (rec.Rhalf / Rhalf - 1), rec.Renclose, NFW_XT * h.rs);
    assert_true(fabs(rec.VmaxVir / Vmax - 1) < 0.02);
    assert_true(fabs(rec.RsKlypin / h.rs - 1) < 0.15);
    assert_true(fabs(rec.Rhalf / Rhalf - 1) < 0.02);
    assert_true(rec.Renclose <= NFW_XT * h.rs * 1.001 && rec.Renclose > 0.95 * NFW_XT * h.rs);
    assert_true(rec.VmaxVir <= rec.Vmax * (1 + 1e-6));
    /* a non-rotating sphere: round, little spin, centred */
    message(0, "NFW shapes %.3f %.3f (500c %.3f %.3f), SpinPeebles %.4f SpinBullock %.4f Xoff %.3f Voff %.2f\n",
            rec.BtoA, rec.CtoA, rec.BtoA500, rec.CtoA500, rec.SpinPeebles, rec.SpinBullock, rec.Xoff, rec.Voff);
    assert_true(rec.BtoA > 0.9 && rec.CtoA > 0.85 && rec.BtoA500 > 0.9 && rec.CtoA500 > 0.85);
    assert_true(rec.SpinPeebles >= 0 && rec.SpinPeebles < 0.02);
    assert_true(rec.SpinBullock >= 0 && rec.SpinBullock < 0.02);
    assert_true(rec.Xoff < 0.02 * rec.RBoundSO[0]);
    /* Phase 5: the fitted Rs; BulkVel equal to Bulk within the sampling noise (sigma ~ 150 km/s
     * over ~ 4e4 particles: 0.75 km/s per component), Vrms finite and positive */
    message(0, "NFW Rs (fit) %.3f rs %.3f (%+.2f%%), RsKlypin %.3f; BulkVel (%.2f %.2f %.2f) Bulk (%g %g %g); "
               "RockstarCoreVel (%.2f %.2f %.2f); Vrms %.2f\n", rec.Rs, h.rs, 100 * (rec.Rs / h.rs - 1), rec.RsKlypin,
            rec.BulkVel[0], rec.BulkVel[1], rec.BulkVel[2], Bulk[0], Bulk[1], Bulk[2],
            rec.RockstarCoreVel[0], rec.RockstarCoreVel[1], rec.RockstarCoreVel[2], rec.Vrms);
    assert_true(rec.Rs > 0 && fabs(rec.Rs / h.rs - 1) < 0.15);
    for(d = 0; d < 3; d++) {
        assert_true(fabs(rec.BulkVel[d] - Bulk[d]) < 3.);
        assert_true(fabs(rec.RockstarCoreVel[d] - Bulk[d]) < 10.);
    }
    assert_true(rec.Vrms > 0 && subhalo_isfinite(rec.Vrms));
    myfree(Energy);
    myfree(Index);
    teardown_pman();
}

/* subhalo_nfw_fit alone on exact equal-mass NFW bin edges (no sampling noise) out to c rs, with
 * and without down-weighted inner bins. Rockstar's iteration (nfw.c:64-99, kept verbatim) is a
 * local refinement of the Klypin start, not a global fit: chi^2 is sharply peaked at rs, the
 * Newton step overshoots, the accepted probe step shrinks the next probe tenfold, and a step
 * that improves nothing ends the loop. So: from starts below rs (0.3, 0.6 rs) it lands within
 * 2 % of rs; from starts above rs (1.6, 2.5 rs) it only moves towards rs (by at most 1.11 times
 * the first probe, rs0 / 10); a start outside the edges falls back to the middle edge and
 * returns a radius inside them. On the sampled sphere of test_nfw the Klypin start is already
 * at the chi^2 minimum and the fit keeps it. */
static void
test_nfw_fit(void ** state)
{
    const double rs = 7.5;
    const double conc[3] = {4., 10., 25.};
    const int nbins[3] = {50, 20, 6};
    const double start[6] = {0.3, 0.6, 1.6, 2.5, 0, 1e6};
    int ic, ib, is, b;
    for(ic = 0; ic < 3; ic++)
        for(ib = 0; ib < 3; ib++) {
            const int nb = nbins[ib];
            double edge[SUBHALO_NFW_MAXBINS + 1];
            const double Mt = nfw_m(conc[ic]);
            edge[0] = 0;
            for(b = 1; b <= nb; b++) {
                double lo = 0, hi = conc[ic];
                int it;
                for(it = 0; it < 100; it++) {
                    const double mid = 0.5 * (lo + hi);
                    if(nfw_m(mid) < Mt * b / nb) lo = mid;
                    else hi = mid;
                }
                edge[b] = 0.5 * (lo + hi) * rs;
            }
            for(is = 0; is < 6; is++) {
                const double f0 = subhalo_nfw_fit(start[is] * rs, edge, nb, 0);
                const double f1 = subhalo_nfw_fit(start[is] * rs, edge, nb, edge[nb / 4]);
                message(0, "NFW fit c %g, %d bins, start %g rs: %.4f rs (inner bins down-weighted %.4f rs)\n",
                        conc[ic], nb, start[is], f0 / rs, f1 / rs);
                const double st = start[is] * rs;
                if(!(st > edge[0]) || !(st < edge[nb])) {
                    assert_true(f0 > edge[0] && f0 < edge[nb] && f1 > edge[0] && f1 < edge[nb]);
                }
                else if(start[is] < 1) {
                    assert_true(fabs(f0 / rs - 1) < 0.02);
                    assert_true(fabs(f1 / rs - 1) < 0.02);
                }
                else {
                    assert_true(fabs(f0 - rs) < fabs(st - rs) && f0 >= rs * (1 - 1e-6));
                    assert_true(fabs(f1 - rs) < fabs(st - rs) && f1 >= rs * (1 - 1e-6));
                }
            }
        }
    /* too few bins */
    double e3[3] = {0, 1, 2};
    assert_true(subhalo_nfw_fit(1, e3, 2, 0) == -1);
    /* Phase 5 audit R6: degenerate or nonfinite input is a failed fit, not a radius. Repeated
     * edges at the fallback start (edge[(6 + 1) / 2] = 1 = the outer edge): the probe step is 0
     * and the derivatives 0 / 0; before the fix the loop broke and returned 1. */
    {
        double rep[7] = {0, 1, 1, 1, 1, 1, 1};
        const double bad_start[3] = {-1, 5, NAN};
        int k;
        for(k = 0; k < 3; k++) {
            const double f = subhalo_nfw_fit(bad_start[k], rep, 6, 0);
            message(0, "NFW fit on repeated edges [0,1,1,1,1,1,1], start %g: %g\n", bad_start[k], f);
            assert_true(f == -1);
        }
        double en[7] = {0, 1, 2, NAN, 4, 5, 6};
        assert_true(subhalo_nfw_fit(2, en, 6, 0) == -1);
        double ei[7] = {0, 1, 2, 3, 4, 5, INFINITY};
        assert_true(subhalo_nfw_fit(2, ei, 6, 0) == -1);
        /* a nonfinite start on valid edges falls back to the middle edge as before */
        double eg[7] = {0, 1, 2, 3, 4, 5, 6};
        const double f = subhalo_nfw_fit(NAN, eg, 6, 0);
        assert_true(f > 0 && f < 6);
    }
    /* the bin plan of calc_scale_radius */
    double mp;
    assert_int_equal(subhalo_nfw_bin_plan(99, 99, 1, &mp), 0);
    assert_int_equal(subhalo_nfw_bin_plan(100, 100, 1, &mp), 6);    /* 100 / 15 */
    assert_true(fabs(mp - 100. / 6) < 1e-12);
    assert_int_equal(subhalo_nfw_bin_plan(1000, 1000, 1, &mp), 50); /* ppbin 20 */
    assert_int_equal(subhalo_nfw_bin_plan(5000, 5000, 1, &mp), 50); /* ppbin 100 */
    assert_int_equal(subhalo_nfw_bin_plan(5001, 5001, 1, &mp), 49); /* ppbin 101 */
    assert_int_equal(subhalo_nfw_bin_plan(200, 200, 50, &mp), 13);
    assert_true(mp == 50);                                           /* at least the largest mass */
}

/* ------------------------------------------------------------------ 2. triaxial Gaussian */

static void
test_triaxial(void ** state)
{
    Cosmology CP;
    setup_cosmology(&CP);
    double rho[SUBHALO_NSO];
    subhalo_so_thresholds(&CP, 1.0, rho);
    const int64_t n = 20000;
    const double sig[3] = {40., 28., 20.};   /* along U1, U2, U3: ratios 0.7, 0.5 */
    const double M = 100.;
    setup_pman(n);
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 777);
    int64_t i;
    int d, e;
    for(i = 0; i < n; i++) {
        double y[3];
        for(e = 0; e < 3; e++)
            y[e] = gsl_ran_gaussian(rng, sig[e]);
        P[i].Type = 1;
        P[i].Mass = M / n;
        P[i].ID = i + 1;
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(Centre[d] + y[0] * U1[d] + y[1] * U2[d] + y[2] * U3[d]);
            P[i].Vel[d] = Bulk[d] + gsl_ran_gaussian(rng, 50.);
        }
    }
    gsl_rng_free(rng);
    int64_t * Index = (int64_t *) mymalloc("Index", sizeof(int64_t) * n);
    double * Energy = (double *) mymalloc("Energy", sizeof(double) * n);
    struct SubTrack rec;
    run_props(n, Centre, rho, &rec, Index, Energy);

    const double cosA = fabs(rec.AxisA[0] * U1[0] + rec.AxisA[1] * U1[1] + rec.AxisA[2] * U1[2]);
    const double cosA5 = fabs(rec.AxisA500[0] * U1[0] + rec.AxisA500[1] * U1[1] + rec.AxisA500[2] * U1[2]);
    const double norm = sqrt(rec.AxisA[0] * rec.AxisA[0] + rec.AxisA[1] * rec.AxisA[1] + rec.AxisA[2] * rec.AxisA[2]);
    message(0, "Triaxial: Rvir %.1f R500c %.1f; BtoA %.4f CtoA %.4f (0.7, 0.5), axis off by %.2f deg; "
               "BtoA500 %.4f CtoA500 %.4f, axis off by %.2f deg\n",
            rec.RBoundSO[0], rec.RBoundSO[3], rec.BtoA, rec.CtoA, acos(fmin(cosA, 1.)) * 180 / M_PI,
            rec.BtoA500, rec.CtoA500, acos(fmin(cosA5, 1.)) * 180 / M_PI);
    assert_true(rec.RBoundSO[0] > 0 && rec.RBoundSO[3] > 0);
    assert_true(fabs(rec.BtoA - 0.7) < 0.03 && fabs(rec.CtoA - 0.5) < 0.03);
    assert_true(fabs(rec.BtoA500 - 0.7) < 0.03 && fabs(rec.CtoA500 - 0.5) < 0.03);
    assert_true(cosA > cos(5 * M_PI / 180) && cosA5 > cos(5 * M_PI / 180));
    assert_true(fabs(norm - 1) < 1e-5);

    /* the inertia tensor of the full set: eigenvalues sigma_i^2 */
    double T[9] = {rec.InertiaTensor[0], rec.InertiaTensor[3], rec.InertiaTensor[4],
                   rec.InertiaTensor[3], rec.InertiaTensor[1], rec.InertiaTensor[5],
                   rec.InertiaTensor[4], rec.InertiaTensor[5], rec.InertiaTensor[2]};
    double eval[3], evec[9];
    gsl_matrix_view mv = gsl_matrix_view_array(T, 3, 3);
    gsl_vector_view ev = gsl_vector_view_array(eval, 3);
    gsl_matrix_view em = gsl_matrix_view_array(evec, 3, 3);
    gsl_eigen_symmv_workspace * ws = gsl_eigen_symmv_alloc(3);
    gsl_eigen_symmv(&mv.matrix, &ev.vector, &em.matrix, ws);
    gsl_eigen_symmv_sort(&ev.vector, &em.matrix, GSL_EIGEN_SORT_VAL_DESC);
    gsl_eigen_symmv_free(ws);
    const double v0[3] = {evec[0], evec[3], evec[6]};
    const double cosT = fabs(dot3(v0, U1));
    message(0, "Triaxial inertia tensor: eigenvalues %.2f %.2f %.2f (expected %.0f %.0f %.0f), sqrt ratios %.4f %.4f, "
               "axis off by %.2f deg\n", eval[0], eval[1], eval[2], sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2],
            sqrt(eval[1] / eval[0]), sqrt(eval[2] / eval[0]), acos(fmin(cosT, 1.)) * 180 / M_PI);
    assert_true(fabs(sqrt(eval[1] / eval[0]) - 0.7) < 0.03 && fabs(sqrt(eval[2] / eval[0]) - 0.5) < 0.03);
    assert_true(fabs(eval[0] / (sig[0] * sig[0]) - 1) < 0.05);
    assert_true(cosT > cos(5 * M_PI / 180));
    myfree(Energy);
    myfree(Index);
    teardown_pman();
}

/* ------------------------------------------------------------------ 3. rotating disc */

#define DISC_N 20000
#define DISC_R 30.
#define DISC_M 50.
#define DISC_VEDGE 200.

/* A uniform disc of radius DISC_R in the plane (U1, U2), thickness 0.01 DISC_R, rotating
 * rigidly about n = U1 x U2 with edge speed DISC_VEDGE, centred on Centre with velocity Bulk,
 * into P[0, DISC_N). */
static void
make_disc(gsl_rng * rng)
{
    const double Omega = DISC_VEDGE / DISC_R;
    double nrm[3];
    nrm[0] = U1[1] * U2[2] - U1[2] * U2[1];
    nrm[1] = U1[2] * U2[0] - U1[0] * U2[2];
    nrm[2] = U1[0] * U2[1] - U1[1] * U2[0];
    int64_t i;
    int d;
    for(i = 0; i < DISC_N; i++) {
        const double R = DISC_R * sqrt(gsl_rng_uniform(rng)), ph = 2 * M_PI * gsl_rng_uniform(rng);
        const double z = gsl_ran_gaussian(rng, 0.01 * DISC_R);
        double x[3], v[3];
        for(d = 0; d < 3; d++)
            x[d] = R * cos(ph) * U1[d] + R * sin(ph) * U2[d] + z * nrm[d];
        /* v = Omega n x x */
        v[0] = Omega * (nrm[1] * x[2] - nrm[2] * x[1]);
        v[1] = Omega * (nrm[2] * x[0] - nrm[0] * x[2]);
        v[2] = Omega * (nrm[0] * x[1] - nrm[1] * x[0]);
        P[i].Type = 1;
        P[i].Mass = DISC_M / DISC_N;
        P[i].ID = i + 1;
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(Centre[d] + x[d]);
            P[i].Vel[d] = Bulk[d] + v[d];
        }
    }
}

/* A dense non-rotating clump of n particles (total mass M, Gaussian radius 0.3) at Centre + dX U1
 * moving with Bulk + dV U2, into P[first, first + n). */
static void
make_clump(gsl_rng * rng, int64_t first, int64_t n, double M, double dX, double dV)
{
    int64_t i;
    int d;
    for(i = first; i < first + n; i++) {
        P[i].Type = 1;
        P[i].Mass = M / n;
        P[i].ID = i + 1;
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] = wrapb(Centre[d] + dX * U1[d] + gsl_ran_gaussian(rng, 0.3));
            P[i].Vel[d] = Bulk[d] + dV * U2[d];
        }
    }
}

/* The Phase 5 velocity trio of the disc tests against oracles over P[0, n) (every particle
 * bound and inside the virial aperture there): the mass-weighted mean and dispersion of all
 * particles, and the mean over r < 0.1 RBoundSO[0] about CorePos (radii floored at EPS). */
static void
disc_trio_check(int64_t n, const struct SubTrack * rec, const char * label)
{
    double Mt = 0, Mc = 0, mv[3] = {0, 0, 0}, cv[3] = {0, 0, 0};
    const double rcore = 0.1 * rec->RBoundSO[0];
    int64_t i, ncore = 0;
    int d;
    for(i = 0; i < n; i++) {
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[i].Pos[d] - rec->CorePos[d], BOXSIZE);
            r2 += dx * dx;
        }
        const double r = fmax(sqrt(r2), EPS);
        Mt += P[i].Mass;
        for(d = 0; d < 3; d++)
            mv[d] += P[i].Mass * P[i].Vel[d];
        if(r < rcore) {
            ncore++;
            Mc += P[i].Mass;
            for(d = 0; d < 3; d++)
                cv[d] += P[i].Mass * P[i].Vel[d];
        }
    }
    double var = 0;
    for(d = 0; d < 3; d++) {
        mv[d] /= Mt;
        cv[d] /= Mc;
    }
    for(i = 0; i < n; i++)
        for(d = 0; d < 3; d++)
            var += P[i].Mass * (P[i].Vel[d] - mv[d]) * (P[i].Vel[d] - mv[d]);
    const double vrms = sqrt(var / Mt);
    message(0, "%s: BulkVel (%.3f %.3f %.3f) sample (%.3f %.3f %.3f); RockstarCoreVel (%.3f %.3f %.3f) oracle "
               "(%.3f %.3f %.3f) over %ld inside 0.1 Rvir; Vrms %.3f sample %.3f\n", label,
            rec->BulkVel[0], rec->BulkVel[1], rec->BulkVel[2], mv[0], mv[1], mv[2], rec->RockstarCoreVel[0],
            rec->RockstarCoreVel[1], rec->RockstarCoreVel[2], cv[0], cv[1], cv[2], (long) ncore, rec->Vrms, vrms);
    assert_true(ncore >= SUBHALO_ROCKSTAR_CORE_MINPART);
    assert_true(fabs(rec->Vrms / vrms - 1) < 1e-4);
    for(d = 0; d < 3; d++) {
        assert_true(fabs(rec->BulkVel[d] - mv[d]) < 1e-4 * vrms);
        assert_true(fabs(rec->RockstarCoreVel[d] - cv[d]) < 1e-4 * vrms);
    }
}

static void
test_disc(void ** state)
{
    Cosmology CP;
    setup_cosmology(&CP);
    double rho[SUBHALO_NSO];
    subhalo_so_thresholds(&CP, 1.0, rho);
    /* The disc alone has a flat potential minimum: its 20 most-bound particles are picked by the
     * discreteness noise of the potential over a few kpc/h, which is a poorly defined centre. A
     * dense non-rotating clump of 0.2 M_disc (500 particles, radius 0.3 kpc/h) supplies the core:
     * at the disc centre first (symmetric system), then displaced. */
    const int64_t nclump = 500, n = DISC_N + nclump;
    const double Mclump = 0.2 * DISC_M;
    const double frac = DISC_M / (DISC_M + Mclump);
    setup_pman(n);
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 99);
    make_disc(rng);
    make_clump(rng, DISC_N, nclump, Mclump, 0, 0);
    const double Omega = DISC_VEDGE / DISC_R;
    double nrm[3] = {-U3[0], -U3[1], -U3[2]};   /* U1 x U2 */
    int64_t i;
    int d;
    /* the sample's own Omega <R^2> M_disc / M_total besides the analytic Omega R^2 / 2 M_disc / M_total
     * (the clump at the centre carries no angular momentum) */
    double R2 = 0;
    for(i = 0; i < DISC_N; i++) {
        double x[3];
        for(d = 0; d < 3; d++)
            x[d] = NEAREST(P[i].Pos[d] - Centre[d], BOXSIZE);
        const double z = dot3(x, nrm);
        R2 += dot3(x, x) - z * z;
    }
    R2 /= DISC_N;
    const double jan = Omega * DISC_R * DISC_R / 2 * frac;

    int64_t * Index = (int64_t *) mymalloc("Index", sizeof(int64_t) * n);
    double * Energy = (double *) mymalloc("Energy", sizeof(double) * n);
    struct SubTrack rec;
    run_props(n, Centre, rho, &rec, Index, Energy);
    const double J[3] = {rec.J[0], rec.J[1], rec.J[2]};
    const double jpar = dot3(J, nrm);
    const double jperp = sqrt(fmax(dot3(J, J) - jpar * jpar, 0));
    const double JV[3] = {rec.JVir[0], rec.JVir[1], rec.JVir[2]};
    message(0, "Disc: J.n %.2f analytic Omega R^2/2 Md/M %.2f (%+.2f%%), sample %.2f; |J_perp| %.3f; JVir.n %.2f\n",
            jpar, jan, 100 * (jpar / jan - 1), Omega * R2 * frac, jperp, dot3(JV, nrm));
    assert_true(fabs(jpar / jan - 1) < 0.01);
    assert_true(jperp < 0.01 * jan);
    assert_true(fabs(dot3(JV, nrm) / jan - 1) < 0.01);
    /* spins: Bullock from the analytic j and the bound virial mass and radius (the whole system is
     * inside Rvir); Peebles from the HBT energies of the same candidate */
    const double lB = jan / sqrt(2 * TEST_G * rec.MBoundSO[0] * rec.RBoundSO[0]);
    const double lP = sqrt(dot3(J, J)) * sqrt(fabs(rec.Ekin + rec.Epot)) / (TEST_G * rec.Mbound);
    message(0, "Disc: SpinBullock %.4f (with the analytic j %.4f), SpinPeebles %.4f (from Ekin + Epot %.4f); "
               "Mvir %.3f of %.3f, Xoff %.3f Voff %.3f\n", rec.SpinBullock, lB, rec.SpinPeebles, lP,
            rec.MBoundSO[0], rec.Mbound, rec.Xoff, rec.Voff);
    assert_true(rec.SpinBullock > 0 && subhalo_isfinite(rec.SpinBullock));
    assert_true(rec.SpinPeebles > 0 && subhalo_isfinite(rec.SpinPeebles));
    assert_true(fabs(rec.SpinBullock / lB - 1) < 0.02);
    assert_true(fabs(rec.SpinPeebles / lP - 1) < 1e-4);
    assert_true(fabs(rec.MBoundSO[0] / rec.Mbound - 1) < 1e-6);
    assert_true(rec.Xoff >= 0 && rec.Xoff < 0.01 * DISC_R);
    assert_true(rec.Voff >= 0 && rec.Voff < 0.01 * DISC_VEDGE);
    /* Phase 5: the trio against the sample; the rigid rotation of a uniform disc has <v^2> =
     * VEDGE^2 / 2 about the centre, carried by the fraction frac of the mass, the clump at rest:
     * Vrms = sqrt(frac / 2) VEDGE, BulkVel = Bulk, RockstarCoreVel = Bulk (symmetric core) */
    disc_trio_check(n, &rec, "Disc trio");
    const double vrms_an = sqrt(frac / 2) * DISC_VEDGE;
    message(0, "Disc: Vrms %.3f analytic sqrt(frac/2) VEDGE %.3f (%+.2f%%), Rs %.3f\n", rec.Vrms, vrms_an,
            100 * (rec.Vrms / vrms_an - 1), rec.Rs);
    assert_true(fabs(rec.Vrms / vrms_an - 1) < 0.05);
    for(d = 0; d < 3; d++) {
        assert_true(fabs(rec.BulkVel[d] - Bulk[d]) < 0.01 * DISC_VEDGE);
        assert_true(fabs(rec.RockstarCoreVel[d] - Bulk[d]) < 0.02 * DISC_VEDGE);
    }

    /* the clump displaced by dX along U1 and moving with dV along U2 relative to the disc: the
     * core follows the clump, so Xoff = dX M_disc / M_total and Voff = dV M_disc / M_total */
    const double dX = 6., dV = 80.;
    make_clump(rng, DISC_N, nclump, Mclump, dX, dV);
    gsl_rng_free(rng);
    run_props(n, Centre, rho, &rec, Index, Energy);
    double coff[3];
    for(d = 0; d < 3; d++)
        coff[d] = NEAREST(rec.CorePos[d] - Centre[d], BOXSIZE);
    message(0, "Displaced core: (CorePos - centre) . U1 = %.3f (%.1f), Xoff %.3f expected %.3f (%+.2f%%), "
               "Voff %.3f expected %.3f (%+.2f%%), Mvir %.3f of %.3f\n", dot3(coff, U1), dX,
            rec.Xoff, dX * frac, 100 * (rec.Xoff / (dX * frac) - 1), rec.Voff, dV * frac, 100 * (rec.Voff / (dV * frac) - 1),
            rec.MBoundSO[0], rec.Mbound);
    assert_true(fabs(dot3(coff, U1) - dX) < 0.1);
    assert_true(fabs(rec.Xoff / (dX * frac) - 1) < 0.03);
    assert_true(fabs(rec.Voff / (dV * frac) - 1) < 0.03);
    /* the core moves with the clump: RockstarCoreVel follows it, BulkVel = Bulk + dV U2 (1 - frac) */
    disc_trio_check(n, &rec, "Displaced trio");
    for(d = 0; d < 3; d++)
        assert_true(fabs(rec.BulkVel[d] - (Bulk[d] + dV * U2[d] * (1 - frac))) < 0.01 * DISC_VEDGE);
    myfree(Energy);
    myfree(Index);
    teardown_pman();
}

/* ------------------------------------------------------------------ 7. units (Phase 3 audit P4) */

/* The disc with its central clump at a = 0.5 in the length unit UL and velocity unit UV (the
 * mass unit 1e10 Msun/h): positions, softening and box in UL, Vel = a v_pec in UV, G and H(a)
 * from the unit system. */
static void
units_props(double UL, double UV, struct SubTrack * rec)
{
    const double atime = 0.5, UM = 1.989e43;
    const double L = UL / 3.085678e21, V = UV / 1e5;
    Cosmology CP;
    memset(&CP, 0, sizeof(CP));
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 0.3;
    CP.OmegaLambda = 0.7;
    CP.OmegaBaryon = 0.045;
    CP.HubbleParam = 0.7;
    CP.w0_fld = -1;
    struct UnitSystem units = get_unitsystem(UL, UM, UV);
    init_cosmology(&CP, 0.01, units);
    subhalo_set_units(UL, UM, UV);
    double rho[SUBHALO_NSO];
    subhalo_so_thresholds(&CP, atime, rho);
    struct SubhaloUnbindParams up = default_params();
    up.atime = atime;
    up.Hubble = hubble_function(&CP, atime);
    up.G = CP.GravInternal;
    up.h = 2.8 * EPS / L;
    up.BoxSize = BOXSIZE / L;
    const int64_t nclump = 500, n = DISC_N + nclump;
    setup_pman(n);
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 77);
    make_disc(rng);
    make_clump(rng, DISC_N, nclump, 0.2 * DISC_M, 2., 30.);
    gsl_rng_free(rng);
    int64_t i;
    int d;
    double ref[3];
    for(i = 0; i < n; i++)
        for(d = 0; d < 3; d++) {
            P[i].Pos[d] /= L;
            P[i].Vel[d] = atime * P[i].Vel[d] / V;
        }
    for(d = 0; d < 3; d++)
        ref[d] = Centre[d] / L;
    PartManager->BoxSize = BOXSIZE / L;
    int64_t * Index = (int64_t *) mymalloc("Index", sizeof(int64_t) * n);
    double * Energy = (double *) mymalloc("Energy", sizeof(double) * n);
    run_props_up(n, ref, rho, rec, Index, Energy, up);
    myfree(Energy);
    myfree(Index);
    PartManager->BoxSize = BOXSIZE;
    teardown_pman();
}

static void
test_units(void ** state)
{
    struct SubTrack a, b;
    units_props(3.085678e21, 1e5, &a);
    units_props(3.085678e24, 2e5, &b);
    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    message(0, "Units: J (%g %g %g) / (%g %g %g) kpc/h km/s, JVir.x %g / %g, SpinPeebles %g / %g, SpinBullock %g / %g, "
               "Vmax %g / %g (internal), VmaxVir %g / %g km/s, MBoundSO[vir] %g / %g, RBoundSO[vir] %g kpc/h / %g Mpc/h\n", a.J[0], a.J[1], a.J[2], b.J[0], b.J[1], b.J[2],
            a.JVir[0], b.JVir[0], a.SpinPeebles, b.SpinPeebles, a.SpinBullock, b.SpinBullock, a.Vmax, b.Vmax, a.VmaxVir, b.VmaxVir,
            a.MBoundSO[0], b.MBoundSO[0], a.RBoundSO[0], b.RBoundSO[0]);
    int d;
    for(d = 0; d < 3; d++) {
        assert_true(fabs(a.J[d] - b.J[d]) <= 1e-4 * fabs(a.J[d]) + 1e-6 * sqrt(a.J[0] * a.J[0] + a.J[1] * a.J[1] + a.J[2] * a.J[2]));
        assert_true(fabs(a.JVir[d] - b.JVir[d]) <= 1e-4 * fabs(a.JVir[d]) + 1e-6 * sqrt(a.JVir[0] * a.JVir[0] + a.JVir[1] * a.JVir[1] + a.JVir[2] * a.JVir[2]));
    }
    assert_true(fabs(a.SpinPeebles / b.SpinPeebles - 1) < 1e-4);
    assert_true(fabs(a.SpinBullock / b.SpinBullock - 1) < 1e-4);
    /* the HBT columns are in internal velocity units here (the key loop converts them) */
    assert_true(fabs(a.Vmax / (2 * b.Vmax) - 1) < 1e-4);
    assert_true(fabs(a.VmaxVir / b.VmaxVir - 1) < 1e-4);
    assert_true(fabs(a.MBoundSO[0] / b.MBoundSO[0] - 1) < 1e-4);
    assert_true(fabs(a.RBoundSO[0] / (1000 * b.RBoundSO[0]) - 1) < 1e-4);
    /* Phase 5: physical km/s and the run's length unit */
    message(0, "Units: Vrms %g / %g km/s, BulkVel.x %g / %g, RockstarCoreVel.x %g / %g, Rs %g kpc/h / %g Mpc/h\n",
            a.Vrms, b.Vrms, a.BulkVel[0], b.BulkVel[0], a.RockstarCoreVel[0], b.RockstarCoreVel[0], a.Rs, b.Rs);
    assert_true(a.Vrms > 0 && fabs(a.Vrms / b.Vrms - 1) < 1e-4);
    for(d = 0; d < 3; d++) {
        assert_true(fabs(a.BulkVel[d] - b.BulkVel[d]) < 1e-4 * a.Vrms);
        assert_true(fabs(a.RockstarCoreVel[d] - b.RockstarCoreVel[d]) < 1e-4 * a.Vrms);
    }
    assert_true((a.Rs == -1 && b.Rs == -1) || fabs(a.Rs / (1000 * b.Rs) - 1) < 1e-3);
}

/* ------------------------------------------------------------------ 4. thresholds */

static void
test_thresholds(void ** state)
{
    Cosmology CP;
    setup_cosmology(&CP);
    const double as[2] = {1.0, 0.1};
    int k, s;
    for(k = 0; k < 2; k++) {
        const double a = as[k];
        double rho[SUBHALO_NSO];
        subhalo_so_thresholds(&CP, a, rho);
        /* the formulas of design 7.10, from the definitions */
        const double H = hubble_function(&CP, a);
        const double rc = 3 * H * H * a * a * a / (8 * M_PI * CP.GravInternal);
        const double rm = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * CP.GravInternal);
        const double Om = CP.Omega0 / (a * a * a) * pow(CP.Hubble / H, 2);
        const double x = Om - 1;
        const double Dvir = 18 * M_PI * M_PI + 82 * x - 39 * x * x;
        const double expect[SUBHALO_NSO] = {Dvir * rc, 200 * rm, 200 * rc, 500 * rc, 2500 * rc};
        message(0, "Thresholds at a = %.1f: Omega_m(a) %.5f Delta_vir %.3f (x critical); rho vir %.4e 200b %.4e 200c %.4e "
                   "500c %.4e 2500c %.4e; 200b / rho_crit_c = %.3f\n", a, Om, rho[0] / rc, rho[0], rho[1], rho[2], rho[3],
                rho[4], rho[1] / rc);
        for(s = 0; s < SUBHALO_NSO; s++)
            assert_true(fabs(rho[s] / expect[s] - 1) < 1e-12);
        /* the comoving critical density at a = 1 is 3 H0^2 / (8 pi G) = 27.75 h^2 Msun / kpc^3, i.e.
         * 2.775e-8 in 1e10 Msun/h / (kpc/h)^3 */
        if(a == 1.0) {
            assert_true(fabs(rc / 2.7754e-8 - 1) < 1e-3);
            /* Omega_m = 0.3: Delta_vir = 18 pi^2 - 82 0.7 - 39 0.49 = 101.1 (H(1) differs from H0 by
             * the massless neutrinos hubble_function adds even without radiation: 1e-5) */
            assert_true(fabs(rho[0] / rc - (18 * M_PI * M_PI - 82 * 0.7 - 39 * 0.49)) < 0.01);
            /* 200b = 60 rho_crit < vir < 200c < 500c < 2500c */
            assert_true(rho[1] < rho[0] && rho[0] < rho[2] && rho[2] < rho[3] && rho[3] < rho[4]);
            assert_true(fabs(rho[1] / rc - 60) < 0.01);
        }
        else {
            /* matter domination: Delta_vir -> 18 pi^2 = 177.65, 200b -> 200c from below */
            assert_true(fabs(rho[0] / rc - 18 * M_PI * M_PI) < 0.5);
            assert_true(rho[0] < rho[1] && rho[1] < rho[2] && rho[2] / rho[1] < 1.01);
        }
        /* the mean density is constant in comoving units */
        assert_true(fabs(rho[1] / (200 * CP.Omega0 * 2.7754e-8) - 1) < 1e-3);
    }
}

/* ------------------------------------------------------------------ 5. baryons */

static void
test_baryons(void ** state)
{
    /* 4 gas, 2 DM, 3 stars, 3 BHs */
    int64_t NLocal[6] = {4, 2, 0, 0, 3, 3};
    const int64_t n = 12;
    PartManager->MaxPart = 64;
    PartManager->NumPart = n;
    PartManager->BoxSize = BOXSIZE;
    PartManager->Base = (struct particle_data *) mymalloc("P", PartManager->MaxPart * sizeof(struct particle_data));
    memset(PartManager->Base, 0, sizeof(struct particle_data) * PartManager->MaxPart);
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    slots_reserve(1, NLocal, SlotsManager);
    slots_setup_topology(PartManager, NLocal, SlotsManager);
    int64_t i;
    int j;
    for(i = 0; i < n; i++) {
        P[i].ID = 100 + i;
        P[i].Mass = 1 + 0.1 * i;
    }
    slots_setup_id(PartManager, SlotsManager);
    double esfr = 0, eheiii = 0, egz = 0, esz = 0, ebh = 0, emdot = 0, egze0 = 0, esze8 = 0;
    for(i = 0; i < n; i++) {
        const double m = P[i].Mass;
        if(P[i].Type == 0) {
            SPHP(i).Sfr = 0.5 * i;
            SPHP(i).Metallicity = 0.01 * (i + 1);
            for(j = 0; j < NMETALS; j++)
                SPHP(i).Metals[j] = 0.001 * (j + 1) * (i + 1);
            P[i].HeIIIionized = (i % 2);
            esfr += 0.5 * i;
            eheiii += (i % 2) * m;
            egz += 0.01 * (i + 1) * m;
            egze0 += 0.001 * (i + 1) * m;
        }
        else if(P[i].Type == 4) {
            STARP(i).Metallicity = 0.02 * (i + 1);
            for(j = 0; j < NMETALS; j++)
                STARP(i).Metals[j] = 0.002 * (j + 1) * (i + 1);
            esz += 0.02 * (i + 1) * m;
            esze8 += 0.002 * 9 * (i + 1) * m;
        }
        else if(P[i].Type == 5) {
            /* the second BH is the most massive (tie with the third, which has the larger ID) */
            BHP(i).Mass = (i == 9) ? 0.01 : 0.03;
            BHP(i).Mdot = 0.1 * i;
            ebh += BHP(i).Mass;
            emdot += 0.1 * i;
        }
    }
    int64_t Index[12];
    for(i = 0; i < n; i++)
        Index[i] = n - 1 - i;
    struct SubCandidate c;
    memset(&c, 0, sizeof(c));
    c.Index = Index;
    c.N = c.Nbound = n;
    struct SubTrack rec;
    subhalo_track_clear(&rec, 0);
    subhalo_props_baryons(&rec, &c, 1);
    message(0, "Baryons: SFR %g (%g) HeIII %g (%g) GasZ %g (%g) StarZ %g (%g) BH %g (%g) Mdot %g (%g) MostMassiveBHID %lu\n",
            rec.StarFormationRate, esfr, rec.MassHeIonized, eheiii, rec.GasMetalMass, egz, rec.StellarMetalMass, esz,
            rec.BlackholeMass, ebh, rec.BlackholeAccretionRate, emdot, (unsigned long) rec.MostMassiveBHID);
    assert_true(fabs(rec.StarFormationRate - esfr) < 1e-5 * esfr);
    assert_true(fabs(rec.MassHeIonized - eheiii) < 1e-5 * eheiii);
    assert_true(fabs(rec.GasMetalMass - egz) < 1e-5 * egz);
    assert_true(fabs(rec.StellarMetalMass - esz) < 1e-5 * esz);
    assert_true(fabs(rec.GasMetalElemMass[0] - egze0) < 1e-5 * egze0);
    assert_true(fabs(rec.StellarMetalElemMass[8] - esze8) < 1e-5 * esze8);
    assert_true(fabs(rec.BlackholeMass - ebh) < 1e-5 * ebh);
    assert_true(fabs(rec.BlackholeAccretionRate - emdot) < 1e-5 * emdot);
    assert_true(rec.MostMassiveBHID == 100 + 10);
    /* without metal return the metal columns stay absent */
    subhalo_props_baryons(&rec, &c, 0);
    assert_true(rec.GasMetalMass == -1 && rec.StellarMetalMass == -1 && rec.GasMetalElemMass[3] == -1);
    assert_true(fabs(rec.StarFormationRate - esfr) < 1e-5 * esfr);
    /* only DM bound: zero sums, no BH */
    Index[0] = 4;
    Index[1] = 5;
    c.N = c.Nbound = 2;
    subhalo_props_baryons(&rec, &c, 1);
    assert_true(rec.StarFormationRate == 0 && rec.BlackholeMass == 0 && rec.GasMetalMass == 0);
    assert_true(rec.MostMassiveBHID == (MyIDType) -1);
    /* nothing bound: every column absent */
    c.N = c.Nbound = 0;
    subhalo_props_baryons(&rec, &c, 1);
    assert_true(rec.StarFormationRate == -1 && rec.BlackholeMass == -1 && rec.StellarMetalElemMass[0] == -1);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    PartManager->Base = NULL;
    PartManager->NumPart = 0;
}

/* Nothing bound, and a tiny candidate without crossings: every Rockstar column stays -1. */
static void
test_absent(void ** state)
{
    Cosmology CP;
    setup_cosmology(&CP);
    double rho[SUBHALO_NSO];
    subhalo_so_thresholds(&CP, 1.0, rho);
    setup_pman(30);
    int64_t i;
    int d;
    /* 30 light particles spread over 1 Mpc/h: mean density far below every threshold */
    for(i = 0; i < 30; i++) {
        P[i].Type = 1;
        P[i].Mass = 1e-6;
        P[i].ID = i + 1;
        for(d = 0; d < 3; d++)
            P[i].Pos[d] = wrapb(Centre[d] + 500. * sin(1.7 * i + d));
    }
    int64_t Index[30];
    double Energy[30];
    struct SubTrack rec;
    run_props(30, Centre, rho, &rec, Index, Energy);
    int s;
    for(s = 0; s < SUBHALO_NSO; s++)
        assert_true(rec.MBoundSO[s] == -1 && rec.RBoundSO[s] == -1);
    assert_true(rec.VmaxVir == -1 && rec.RmaxVir == -1 && rec.JVir[0] == -1 && rec.SpinBullock == -1);
    assert_true(rec.Xoff == -1 && rec.Voff == -1 && rec.RsKlypin == -1 && rec.BtoA == -1 && rec.AxisA500[2] == -1);
    /* the Phase 5 columns need the crossing too */
    assert_true(rec.Rs == -1 && rec.Vrms == -1);
    for(d = 0; d < 3; d++)
        assert_true(rec.RockstarCoreVel[d] == -1 && rec.BulkVel[d] == -1);
    /* defined without a crossing */
    assert_true(rec.Rhalf > 0 && rec.Renclose >= rec.Rhalf && rec.InertiaTensor[0] > 0 && rec.SpinPeebles >= 0);
    /* nothing bound */
    struct SubCandidate c;
    memset(&c, 0, sizeof(c));
    struct SubhaloUnbindParams up = default_params();
    struct SubhaloArena arena = {NULL, 0, 0};
    const double Offset[3] = {0, 0, 0};
    rec.Rhalf = 5;
    subhalo_props_rockstar(&rec, &c, &up, rho, CORESIZE, Offset, &arena);
    assert_true(rec.Rhalf == -1 && rec.J[1] == -1 && rec.InertiaTensor[5] == -1);
    assert_true(rec.Rs == -1 && rec.Vrms == -1 && rec.BulkVel[0] == -1 && rec.RockstarCoreVel[2] == -1);
    rec.Rs = rec.Vrms = rec.BulkVel[0] = rec.RockstarCoreVel[2] = 5;
    subhalo_props_deferred(&rec, &c, &up, rho, CORESIZE, &arena);
    assert_true(rec.Rs == -1 && rec.Vrms == -1 && rec.BulkVel[0] == -1 && rec.RockstarCoreVel[2] == -1);
    teardown_pman();
}

/* ------------------------------------------------------------------ 6. HBT Vmax */

/* max over the bound particles i of M(<= r_i) / r_i by brute force, radii from cen floored at
 * rfloor, the smallest radius on ties. */
static double
vmax_brute(int64_t n, const double cen[3], double rfloor, double * rbest)
{
    double best = -1;
    int64_t i, j;
    int d;
    double * r = (double *) mymalloc("TestR", sizeof(double) * n);
    for(i = 0; i < n; i++) {
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[i].Pos[d] - cen[d], BOXSIZE);
            r2 += dx * dx;
        }
        r[i] = sqrt(r2) > rfloor ? sqrt(r2) : rfloor;
    }
    for(i = 0; i < n; i++) {
        double M = 0;
        for(j = 0; j < n; j++)
            if(r[j] <= r[i])
                M += P[j].Mass;
        if(M / r[i] > best * (1 + 1e-13) || (fabs(M / r[i] - best) <= 1e-13 * best && r[i] < *rbest)) {
            best = M / r[i];
            *rbest = r[i];
        }
    }
    myfree(r);
    return best;
}

/* subhalo_props_hbt on P[0, n) in Index order with CoreSize core particles, an arena of exactly
 * the documented budget. */
static void
hbt_props(int64_t n, int coresize, struct SubTrack * rec)
{
    struct SubhaloUnbindParams up = default_params();
    struct SubCandidate c;
    memset(&c, 0, sizeof(c));
    int64_t * Index = (int64_t *) mymalloc("TestIndex", sizeof(int64_t) * n);
    double * Energy = (double *) mymalloc("TestEnergy", sizeof(double) * n);
    int64_t i;
    for(i = 0; i < n; i++) {
        Index[i] = i;
        Energy[i] = -1;
    }
    c.Index = Index;
    c.Energy = Energy;
    c.N = c.Nbound = n;
    struct SubhaloArena arena;
    arena.size = SUBHALO_PROPS_BYTES_PER_PART * n;
    arena.used = 0;
    arena.base = (char *) mymalloc("TestArena", arena.size);
    subhalo_track_clear(rec, 0);
    const double Offset[3] = {0, 0, 0};
    subhalo_props_hbt(rec, &c, &up, coresize, Offset, &arena);
    assert_int_equal(arena.used, 0);
    myfree(arena.base);
    myfree(Energy);
    myfree(Index);
}

/* P[i] at Centre + r along axis (k / 2) % 3, sign by k % 2 */
static void
place(int64_t i, double r, int64_t k, double m, MyIDType id)
{
    int d;
    memset(&P[i], 0, sizeof(P[i]));
    P[i].Type = 1;
    P[i].Mass = m;
    P[i].ID = id;
    for(d = 0; d < 3; d++)
        P[i].Pos[d] = wrapb(Centre[d] + (d == (k / 2) % 3 ? (k % 2 ? -r : r) : 0));
}

static void
test_vmax(void ** state)
{
    const double rfloor = 2.8 * EPS / 2.8;
    struct SubTrack rec;
    double rb = 0, best;
    int64_t i, b;
    setup_pman(600);

    /* (a) the audit's multi-peak profile: rlo(b) = exp((b + 0.2) / 64), rhi(b) = exp((b + 0.8)
     * / 64), M(rhi) = rhi, M(rlo) = 1.00001 rlo except 1.005 rlo at b = 220, endpoints at 1 and
     * exp(4) with 1e-9 each; a central core particle of mass 1e-12 */
    {
        int64_t n = 0;
        place(n, 0, 0, 1e-12, 1 + n);
        n++;
        double Mprev = 1e-12;
        place(n, 1, n, 1e-9, 1 + n);
        Mprev += 1e-9;
        n++;
        for(b = 0; b < 256; b++) {
            const double rlo = exp((b + 0.2) / 64), rhi = exp((b + 0.8) / 64);
            const double Mlo = rlo * (b == 220 ? 1.005 : 1.00001);
            place(n, rlo, n, Mlo - Mprev, 1 + n);
            n++;
            place(n, rhi, n, rhi - Mlo, 1 + n);
            n++;
            Mprev = rhi;
        }
        place(n, exp(4.), n, 1e-9, 1 + n);
        n++;
        for(i = 0; i < n; i++)
            assert_true(P[i].Mass > 0);
        hbt_props(n, 1, &rec);
        best = vmax_brute(n, rec.CorePos, rfloor, &rb);
        message(0, "Vmax multi-peak profile (%ld particles): Rmax %.8g (oracle %.8g, audit 31.20645611), Vmax^2 / G %.8g (oracle %.8g, audit 1.005)\n",
                (long) n, rec.Rmax, rb, rec.Vmax * rec.Vmax / TEST_G, best);
        assert_true(fabs(rb / 31.20645611 - 1) < 1e-8);
        assert_true(fabs(best / 1.005 - 1) < 1e-8);
        assert_true(fabs(rec.Rmax / rb - 1) < 1e-6);
        assert_true(fabs(rec.Vmax / sqrt(TEST_G * best) - 1) < 1e-6);
    }
    /* (b) flat: pairs at radii 1..100 (integers, exact), mass 0.5 each, the radius-1 pair the
     * core (CoreSize 2, centred exactly): M(<= r) / r = 1 at every radius, Rmax the smallest */
    {
        int64_t n = 0, k;
        for(k = 1; k <= 100; k++) {
            place(n, (double) k, 2 * (k % 3), 0.5, 1000 - n);
            n++;
            place(n, (double) k, 2 * (k % 3) + 1, 0.5, 1000 - n);
            n++;
        }
        hbt_props(n, 2, &rec);
        best = vmax_brute(n, rec.CorePos, rfloor, &rb);
        message(0, "Vmax flat profile: Rmax %g (oracle %g), Vmax^2 / G %.12g (oracle %.12g)\n", rec.Rmax, rb, rec.Vmax * rec.Vmax / TEST_G, best);
        assert_true(rec.Rmax == 1 && rb == 1 && best == 1);
        assert_true(fabs(rec.Vmax / sqrt(TEST_G) - 1) < 1e-6);
    }
    /* (c) random masses on 12 shared radii (many equal radii), and particles inside the floor */
    {
        gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng, 3);
        const int64_t n = 500;
        place(0, 0, 0, 0.01, 1);
        for(i = 1; i < n; i++) {
            const int inner = i % 25 == 0;   /* a few light particles inside the floor */
            const double r = inner ? 0.1 * gsl_rng_uniform(rng) : (double) (1 + gsl_rng_uniform_int(rng, 12));
            place(i, r, gsl_rng_uniform_int(rng, 6), (inner ? 0.01 : 1) * (0.1 + gsl_rng_uniform(rng)), 1 + (i * 37) % n);
        }
        gsl_rng_free(rng);
        hbt_props(n, 1, &rec);
        best = vmax_brute(n, rec.CorePos, rfloor, &rb);
        message(0, "Vmax shared radii: Rmax %g (oracle %g), Vmax^2 / G %.10g (oracle %.10g)\n", rec.Rmax, rb, rec.Vmax * rec.Vmax / TEST_G, best);
        assert_true(rb > 1);   /* the maximum lies on a shared radius, not at the floor */
        assert_true(fabs(rec.Rmax / rb - 1) < 1e-6);
        assert_true(fabs(rec.Vmax / sqrt(TEST_G * best) - 1) < 1e-6);
    }
    teardown_pman();
}

/* ------------------------------------------------------------------ 9. NFW bin edges (Phase 5 audit R5) */

/* Rockstar's bin edge (nfw.c:126-132) converts the midpoint to kpc/h and subtracts the
 * interpolation in Mpc/h (an upstream unit bug that effectively drops the correction); the recipe
 * applies the correction in the run's length unit. (a) The audit's example, radii 0.01, 0.02, 0.03
 * Mpc/h and an excess of half the closing mass: 0.020 Mpc/h = 20 kpc/h (Rockstar literally 24.995
 * kpc/h), the same edge in kpc/h units. (b) An overshooting-bin oracle: 132 equal masses at
 * distinct radii 0.5 + 0.1 k from a one-particle core, every particle above the threshold (the
 * crossing is the outermost particle), Rockstar's fit aperture (decision 44) the 130 inside the one
 * before it, 8 bins of 16.25 masses (every bin closes with an excess); Rs of subhalo_props_deferred
 * against the fit on the edges computed here with the intended interpolation, and different from
 * the fit on the former aperture (all 132, crossing included). The crossing particle carries a
 * large velocity: BulkVel and Vrms of the velocity aperture (the 131 inside it, at rest) are 0,
 * with it they would not be. */
static void
test_nfw_bin_edges(void ** state)
{
    /* (a) */
    const double e_mpc = subhalo_nfw_edge(1, 1.0, 0.01, 1, 0.02, 0.03, 1.5, 1.0);
    const double e_kpc = subhalo_nfw_edge(1, 1.0, 10., 1, 20., 30., 1.5, 1.0);
    const double rockstar_literal_kpc = 1e3 * 0.5 * (0.02 + 0.03) - 0.5 * (0.02 - 0.01);
    message(0, "NFW bin edge, radii 0.01/0.02/0.03 Mpc/h, excess/mass 0.5: %.6g Mpc/h = %.6g kpc/h (kpc/h units: %.6g; Rockstar's "
               "literal nfw.c arithmetic %.6g kpc/h)\n", e_mpc, 1e3 * e_mpc, e_kpc, rockstar_literal_kpc);
    assert_true(fabs(e_mpc - 0.02) < 1e-15);
    assert_true(fabs(e_kpc - 20.) < 1e-12);
    assert_true(fabs(rockstar_literal_kpc - 24.995) < 1e-9);
    /* no excess, or the first particle: the midpoint */
    assert_true(fabs(subhalo_nfw_edge(1, 1.0, 10., 1, 20., 30., 1.0, 1.0) - 25.) < 1e-12);
    assert_true(fabs(subhalo_nfw_edge(1, 1.0, 0., 0, 20., 30., 1.5, 1.0) - 25.) < 1e-12);

    /* (b) */
    const int64_t n = 132;
    const double m = 0.01;
    int64_t k;
    int b;
    setup_pman(n);
    place(0, 0, 0, m, 1);
    for(k = 1; k < n; k++)
        place(k, 0.5 + 0.1 * k, k, m, 1 + k);
    P[n - 1].Vel[0] = 5000.;             /* the crossing particle */
    struct SubhaloUnbindParams up = default_params();
    struct SubCandidate c;
    memset(&c, 0, sizeof(c));
    int64_t * Index = (int64_t *) mymalloc("TestIndex", sizeof(int64_t) * n);
    for(k = 0; k < n; k++)
        Index[k] = k;
    c.Index = Index;
    c.N = c.Nbound = n;
    double rho[SUBHALO_NSO];
    for(k = 0; k < SUBHALO_NSO; k++)
        rho[k] = 1e-12;               /* every particle inside every aperture */
    struct SubhaloArena arena;
    arena.size = SUBHALO_PROPS_BYTES_PER_PART * n;
    arena.used = 0;
    arena.base = (char *) mymalloc("TestArena", arena.size);
    struct SubTrack rec;
    subhalo_track_clear(&rec, 0);
    subhalo_props_deferred(&rec, &c, &up, rho, 1, &arena);
    assert_int_equal(arena.used, 0);
    myfree(arena.base);
    /* the oracle: radii from the same core, sorted (distinct by construction) */
    double cpos[3], cvel[3], sx, sv;
    subhalo_core(&c, &up, 1, cpos, cvel, &sx, &sv);
    const double rfloor = up.h / 2.8;
    double r[132];
    for(k = 0; k < n; k++) {
        double r2 = 0;
        int d;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[k].Pos[d] - cpos[d], BOXSIZE);
            r2 += dx * dx;
        }
        r[k] = sqrt(r2) > rfloor ? sqrt(r2) : rfloor;
        assert_true(k == 0 || r[k] > r[k - 1]);
    }
    double mpbin;
    const double Mvir = n * m;
    const int64_t nfit = n - 2;                    /* nfw.c: i < total_p - 1, total_p the crossing */
    const int nb = subhalo_nfw_bin_plan(nfit, nfit * m, m, &mpbin);
    assert_int_equal(nb, 8);                       /* ppbin 130 / (130 / 15) = 16 */
    assert_true(fabs(mpbin / m - 16.25) < 1e-12);
    double edge[SUBHALO_NFW_MAXBINS + 1], lit[SUBHALO_NFW_MAXBINS + 1];
    edge[0] = lit[0] = 0;
    int novershoot = 0;
    for(b = 1, k = 0; b <= nb && k < nfit; k++) {
        const double M = (double) (k + 1) * m;
        if(!(M >= b * mpbin * (1 - 1e-9)))
            continue;
        const double excess = (M - b * mpbin) / m;
        const double rnext = k + 1 < n ? r[k + 1] : r[k];
        edge[b] = 0.5 * (r[k] + rnext) - excess * (r[k] - r[k - 1]);
        lit[b] = 0.5 * (r[k] + rnext);               /* Rockstar's effective edge: the correction dropped */
        if(excess > 0.1)
            novershoot++;
        b++;
    }
    assert_int_equal(b - 1, nb);
    assert_true(novershoot >= nb / 2);
    double best = 0, rbest = -1;
    for(k = 0; k < n; k++)
        if((double) (k + 1) * m / r[k] > best) {
            best = (double) (k + 1) * m / r[k];
            rbest = r[k];
        }
    const double Rvir = cbrt(3 * Mvir / (4 * M_PI * rho[0]));
    const double rs0 = subhalo_klypin_rs(Rvir, Mvir, best, rbest);
    const double rs = subhalo_nfw_fit(rs0, edge, nb, 3 * rfloor);
    const double rs_lit = subhalo_nfw_fit(rs0, lit, nb, 3 * rfloor);
    /* the former aperture (D4: every particle up to and including the crossing) */
    double mpold, eold[SUBHALO_NFW_MAXBINS + 1];
    const int nbold = subhalo_nfw_bin_plan(n, Mvir, m, &mpold);
    eold[0] = 0;
    for(b = 1, k = 0; b <= nbold && k < n; k++) {
        const double M = (double) (k + 1) * m;
        if(!(M >= b * mpold * (1 - 1e-9)))
            continue;
        const double rnext = k + 1 < n ? r[k + 1] : r[k];
        eold[b] = 0.5 * (r[k] + rnext) - (M - b * mpold) / m * (r[k] - r[k - 1]);
        b++;
    }
    const double rs_old = subhalo_nfw_fit(rs0, eold, b - 1, 3 * rfloor);
    message(0, "NFW bin oracle: Rockstar's aperture Rs %.8g, the former aperture (crossing included) %.8g (%d bins of %.4g masses); "
               "BulkVel (%g %g %g) Vrms %g RockstarCoreVel.x %g with the crossing particle at 5000\n", rs, rs_old, nbold, mpold / m,
            rec.BulkVel[0], rec.BulkVel[1], rec.BulkVel[2], rec.Vrms, rec.RockstarCoreVel[0]);
    assert_true(rs_old > 0 && fabs(rs_old / rs - 1) > 1e-4);
    assert_true(rec.BulkVel[0] == 0 && rec.BulkVel[1] == 0 && rec.BulkVel[2] == 0 && rec.Vrms == 0 && rec.RockstarCoreVel[0] == 0);
    message(0, "NFW bin oracle: %d bins of %.4g masses, %d with an excess > 0.1; Rs %.8g, oracle %.8g (Rockstar's effective midpoint "
               "edges: %.8g); edge 1 %.6g (midpoint %.6g)\n", nb, mpbin / m, novershoot, rec.Rs, rs, rs_lit, edge[1], lit[1]);
    assert_true(rs > 0 && fabs(rec.Rs / rs - 1) < 1e-6);
    assert_true(edge[1] < lit[1]);
    myfree(Index);
    teardown_pman();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_thresholds),
        cmocka_unit_test(test_nfw),
        cmocka_unit_test(test_nfw_fit),
        cmocka_unit_test(test_triaxial),
        cmocka_unit_test(test_disc),
        cmocka_unit_test(test_baryons),
        cmocka_unit_test(test_absent),
        cmocka_unit_test(test_vmax),
        cmocka_unit_test(test_units),
        cmocka_unit_test(test_nfw_bin_edges),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
