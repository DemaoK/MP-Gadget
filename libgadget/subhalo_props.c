/* HBT-level subhalo properties (design 7.5, Phase 2a plan section 4 and step 7g).
 *
 * Input: a candidate after subhalo_final_pass, i.e. Index[0, Nbound) in ascending final-pass
 * energy, Energy the final-pass energies in the frame (RefPos, RefVel). The HBT set (Phase 2a)
 * first, then the Rockstar set and the baryonic sums of Phase 3 (below). Declared in
 * subhalo_unbind.h. */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_eigen.h>

#include "utils/endrun.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "cosmology.h"
#include "subhalo_unbind.h"

static inline double
wrap_box(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* Radius of P[p] from cen, comoving, floored at rfloor. */
static inline double
floored_radius(int64_t p, const double cen[3], double rfloor, double BoxSize)
{
    double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double dx = NEAREST(P[p].Pos[d] - cen[d], BoxSize);
        r2 += dx * dx;
    }
    const double r = sqrt(r2);
    return r > rfloor ? r : rfloor;
}

/* Exact max over the bound particles i of M(<= r_i) / r_i, radii from cen floored at rfloor
 * (HBT CalculateProfileProperties, subhalo.cpp:200-231, which sorts the radii). The function
 * has no scratch memory, so instead of a sort it brackets the maximum with mass histograms:
 * a pass over [lo, hi] bins the particles in log r and keeps per bin the mass and the smallest
 * and largest radius; with M_b the mass at r <= the largest radius of bin b, M_b / rmax_b is a
 * value the profile attains and M_b / rmin_b bounds every particle of the bin, so only bins
 * whose bound beats the best attained value are refined by further passes over their own
 * [rmin_b, rmax_b]. A bin holding a single radius is exact. Returns M / r at the maximum and its
 * radius in *Rmax. */
#define VMAX_NBINS 256
#define VMAX_WORK 128
#define VMAX_MAXPASS 64

static double
mass_within(const struct SubCandidate * c, const double cen[3], double rfloor, double BoxSize, double r)
{
    double M = 0;
    int64_t i;
    for(i = 0; i < c->Nbound; i++)
        if(floored_radius(c->Index[i], cen, rfloor, BoxSize) <= r)
            M += P[c->Index[i]].Mass;
    return M;
}

static double
vmax_profile(const struct SubCandidate * c, const double cen[3], double rfloor, double BoxSize, double * Rmax)
{
    const int64_t n = c->Nbound;
    int64_t i;
    double rmin = 1e300, rmax = 0;
    for(i = 0; i < n; i++) {
        const double r = floored_radius(c->Index[i], cen, rfloor, BoxSize);
        if(r < rmin) rmin = r;
        if(r > rmax) rmax = r;
    }
    double wlo[VMAX_WORK], whi[VMAX_WORK], wub[VMAX_WORK];
    int nw = 0, npass = 0;
    double best = 0, bestr = rmin;
    wlo[nw] = rmin;
    whi[nw] = rmax;
    wub[nw] = 1e300;
    nw++;
    while(nw > 0 && npass < VMAX_MAXPASS) {
        nw--;
        const double lo = wlo[nw], hi = whi[nw];
        /* the bound of a pushed range may have been beaten since */
        if(wub[nw] <= best * (1 + 1e-12))
            continue;
        npass++;
        if(!(hi > lo)) {
            const double v = mass_within(c, cen, rfloor, BoxSize, lo) / lo;
            if(v > best) {
                best = v;
                bestr = lo;
            }
            continue;
        }
        double bm[VMAX_NBINS], bmin[VMAX_NBINS], bmax[VMAX_NBINS];
        int b;
        for(b = 0; b < VMAX_NBINS; b++) {
            bm[b] = 0;
            bmin[b] = 1e300;
            bmax[b] = 0;
        }
        const double llo = log(lo);
        const double lscale = VMAX_NBINS / (log(hi) - llo);
        double Mbelow = 0;
        for(i = 0; i < n; i++) {
            const double r = floored_radius(c->Index[i], cen, rfloor, BoxSize);
            const double m = P[c->Index[i]].Mass;
            if(r < lo) {
                Mbelow += m;
                continue;
            }
            if(r > hi)
                continue;
            b = (int) ((log(r) - llo) * lscale);
            if(b < 0) b = 0;
            if(b >= VMAX_NBINS) b = VMAX_NBINS - 1;
            bm[b] += m;
            if(r < bmin[b]) bmin[b] = r;
            if(r > bmax[b]) bmax[b] = r;
        }
        double M = Mbelow, Mle[VMAX_NBINS];
        for(b = 0; b < VMAX_NBINS; b++) {
            M += bm[b];
            Mle[b] = M;
            if(bm[b] > 0 && M / bmax[b] > best) {
                best = M / bmax[b];
                bestr = bmax[b];
            }
        }
        for(b = 0; b < VMAX_NBINS; b++)
            if(bm[b] > 0 && bmin[b] < bmax[b] && Mle[b] / bmin[b] > best * (1 + 1e-12) && nw < VMAX_WORK) {
                wlo[nw] = bmin[b];
                whi[nw] = bmax[b];
                wub[nw] = Mle[b] / bmin[b];
                nw++;
            }
    }
    /* the maximum, recomputed exactly at its radius */
    *Rmax = bestr;
    return mass_within(c, cen, rfloor, BoxSize, bestr) / bestr;
}

void
subhalo_props_hbt(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                  int CoreSize, const double Offset[3])
{
    const int64_t n = c->Nbound;
    const double a = up->atime;
    const double BoxSize = up->BoxSize;
    int64_t i;
    int d, t;

    /* counts and masses */
    double Mt[6] = {0, 0, 0, 0, 0, 0}, M = 0;
    int64_t Nt[6] = {0, 0, 0, 0, 0, 0}, ntr = 0, mbt = -1;
    for(i = 0; i < n; i++) {
        const int64_t p = c->Index[i];
        const int type = P[p].Type;
        Nt[type]++;
        Mt[type] += P[p].Mass;
        M += P[p].Mass;
        if((up->TracerTypes >> type) & 1) {
            ntr++;
            if(mbt < 0)
                mbt = p;
        }
    }
    rec->Nbound = (int32_t) n;
    rec->NboundTracer = (int32_t) ntr;
    rec->Mbound = (float) M;
    for(t = 0; t < 6; t++) {
        rec->NboundType[t] = (int32_t) Nt[t];
        rec->MboundType[t] = (float) Mt[t];
    }
    rec->Flags |= c->Flags;
    if(n <= 0)
        return;

    /* core (subhalo_merge.cpp:93-181) */
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;
    subhalo_core(c, up, CoreSize, CorePos, CoreVel, &SigmaX, &SigmaV);
    for(d = 0; d < 3; d++) {
        rec->CorePos[d] = wrap_box(CorePos[d] - Offset[d], BoxSize);
        rec->CoreVel[d] = (float) (CoreVel[d] / a);
        rec->CoMPos[d] = wrap_box(c->RefPos[d] - Offset[d], BoxSize);
        rec->CoMVel[d] = (float) (c->RefVel[d] / a);
    }
    rec->CoreSigmaX = (float) SigmaX;
    rec->CoreSigmaV = (float) (SigmaV / a);

    /* the most-bound tracer (HBT MostBoundParticleId after CountParticleTypes,
     * subhalo_unbind.cpp:768-775); the most bound particle if no tracer is bound */
    const int64_t mb = mbt >= 0 ? mbt : c->Index[0];
    rec->MostBoundID = P[mb].ID;
    for(d = 0; d < 3; d++) {
        rec->MostBoundPos[d] = wrap_box(P[mb].Pos[d] - Offset[d], BoxSize);
        rec->MostBoundVel[d] = (float) (P[mb].Vel[d] / a);
    }

    /* Vmax, Rmax: whole bound profile from CorePos, radii floored at GravitySoftening = h / 2.8
     * (subhalo.cpp:221-231; physical radius a r) */
    double Rmax;
    const double v2 = vmax_profile(c, CorePos, up->h / 2.8, BoxSize, &Rmax);
    rec->Vmax = (float) sqrt(up->G * v2 / a);
    rec->Rmax = (float) Rmax;

    /* Ekin, Epot per unit mass in the final frame; phi = E - K (- u) of the final pass;
     * Epot counts each pair once */
    double K = 0, U = 0;
    for(i = 0; i < n; i++) {
        const int64_t p = c->Index[i];
        const double m = P[p].Mass;
        double k2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[p].Pos[d] - c->RefPos[d], BoxSize);
            const double dv = (P[p].Vel[d] - c->RefVel[d]) / a + up->Hubble * a * dx;
            k2 += dv * dv;
        }
        double phi = c->Energy[i] - 0.5 * k2;
        if(up->WithThermal)
            phi -= subhalo_thermal_energy(p, a);
        K += m * 0.5 * k2;
        U += m * phi;
    }
    rec->Ekin = (float) (K / M);
    rec->Epot = (float) (0.5 * U / M);
    rec->KinToPot = rec->Epot != 0 ? (float) (K / M / fabs(0.5 * U / M)) : -1;
}

/* ================================================================== Phase 3: Rockstar recipes
 *
 * Rockstar's property recipes on the HBT bound set (design D4, 4.3, 7.5, 7.10; Phase 3 plan
 * sections 1A and 3): bound SO masses and radii, VmaxVir/RmaxVir, Rhalf, Renclose, J/JVir, the
 * Peebles and Bullock spins, Xoff/Voff, the Klypin Rs, the Allgood iterative shapes at Rvir and
 * R500c and the mass-normalised inertia tensor; then the group-C baryonic sums. Sources:
 * Rockstar-BigFile properties.c:246-314 (calc_shape), 383-480 (_calc_additional_halo_props),
 * nfw.c:14-45 (f_to_c, estimate_scale_radius), groupies.c:63-103 (thresholds). */

void
subhalo_so_thresholds(const Cosmology * CP, double atime, double rho[SUBHALO_NSO])
{
    const double a3 = atime * atime * atime;
    const double G = CP->GravInternal;
    const double H = hubble_function(CP, atime);
    /* comoving critical density at a and the comoving mean matter density (constant) */
    const double rhocrit = 3 * H * H * a3 / (8 * M_PI * G);
    const double rhomean = CP->Omega0 * 3 * CP->Hubble * CP->Hubble / (8 * M_PI * G);
    /* Bryan & Norman relative to critical (groupies.c:63-66 divided by Omega_m(a)) */
    const double Oma = CP->Omega0 / a3 * (CP->Hubble / H) * (CP->Hubble / H);
    const double x = Oma - 1;
    const double Dvir = 18 * M_PI * M_PI + 82 * x - 39 * x * x;
    rho[0] = Dvir * rhocrit;
    rho[1] = 200 * rhomean;
    rho[2] = 200 * rhocrit;
    rho[3] = 500 * rhocrit;
    rho[4] = 2500 * rhocrit;
}

/* (radius, P index) pairs of the radial order, ties by P index. 16 bytes per bound particle:
 * the only per-particle scratch of subhalo_props_rockstar. */
struct RPair {
    double r;
    int64_t p;
};
#define RPAIR_LESS(x, y) ((x)->r < (y)->r || ((x)->r == (y)->r && (x)->p < (y)->p))

/* Allocation-free introsort (the algorithm of subhalo_unbind.c's SUBHALO_DEFINE_SORT, which is
 * private to that file): quicksort with median of three, heapsort below the depth limit,
 * insertion sort below 16. The order is a strict total order, so the result is unique. */
static void
rpair_isort(struct RPair * a, int64_t n)
{
    int64_t i, j;
    for(i = 1; i < n; i++) {
        struct RPair x = a[i];
        for(j = i; j > 0 && RPAIR_LESS(&x, &a[j - 1]); j--)
            a[j] = a[j - 1];
        a[j] = x;
    }
}

static void
rpair_sift(struct RPair * a, int64_t root, int64_t n)
{
    for(;;) {
        int64_t child = 2 * root + 1;
        if(child >= n)
            return;
        if(child + 1 < n && RPAIR_LESS(&a[child], &a[child + 1]))
            child++;
        if(!RPAIR_LESS(&a[root], &a[child]))
            return;
        struct RPair t = a[root];
        a[root] = a[child];
        a[child] = t;
        root = child;
    }
}

static void
rpair_intro(struct RPair * a, int64_t n, int depth)
{
    while(n > 16) {
        struct RPair t;
        if(depth-- <= 0) {
            int64_t i;
            for(i = n / 2 - 1; i >= 0; i--)
                rpair_sift(a, i, n);
            for(i = n - 1; i > 0; i--) {
                t = a[0]; a[0] = a[i]; a[i] = t;
                rpair_sift(a, 0, i);
            }
            return;
        }
        const int64_t mid = n / 2;
        if(RPAIR_LESS(&a[mid], &a[0])) { t = a[mid]; a[mid] = a[0]; a[0] = t; }
        if(RPAIR_LESS(&a[n - 1], &a[0])) { t = a[n - 1]; a[n - 1] = a[0]; a[0] = t; }
        if(RPAIR_LESS(&a[n - 1], &a[mid])) { t = a[n - 1]; a[n - 1] = a[mid]; a[mid] = t; }
        const struct RPair pivot = a[mid];
        int64_t i = -1, j = n;
        for(;;) {
            do i++; while(RPAIR_LESS(&a[i], &pivot));
            do j--; while(RPAIR_LESS(&pivot, &a[j]));
            if(i >= j)
                break;
            t = a[i]; a[i] = a[j]; a[j] = t;
        }
        if(j + 1 < n - j - 1) {
            rpair_intro(a, j + 1, depth);
            a += j + 1;
            n -= j + 1;
        }
        else {
            rpair_intro(a + j + 1, n - j - 1, depth);
            n = j + 1;
        }
    }
    rpair_isort(a, n);
}

static void
rpair_sort(struct RPair * a, int64_t n)
{
    int depth = 2;
    int64_t m = n;
    while(m > 1) {
        m >>= 1;
        depth += 2;
    }
    rpair_intro(a, n, depth);
}

/* Eigen-decomposition of a symmetric 3x3 matrix (destroyed), eigenvalues in descending order,
 * eigenvectors as the columns of evec. */
static void
eigen3(double mat[9], double eval[3], double evec[9], gsl_eigen_symmv_workspace * ws)
{
    gsl_matrix_view m = gsl_matrix_view_array(mat, 3, 3);
    gsl_vector_view ev = gsl_vector_view_array(eval, 3);
    gsl_matrix_view em = gsl_matrix_view_array(evec, 3, 3);
    gsl_eigen_symmv(&m.matrix, &ev.vector, &em.matrix, ws);
    gsl_eigen_symmv_sort(&ev.vector, &em.matrix, GSL_EIGEN_SORT_VAL_DESC);
}

/* Smallest aperture for which a shape is reported (plan section 3). */
#define SHAPE_MINPART 20
#define SHAPE_ITERATIONS 10

/* Allgood et al. (2006) iterative reduced inertia tensor, Rockstar calc_shape
 * (properties.c:246-314, WEIGHTED_SHAPES = 1): over the radially sorted bound particles
 * pr[0, np) (Rockstar's aperture, the particles up to the SO crossing), those inside the
 * ellipsoid whose major axis is R contribute m x_i x_j / q2, q2 = sum_k (e_k . x)^2 / ax_k^2
 * (the ellipsoid in units of R, floored at (rfloor / R)^2), normalised by sum m / q2; the axes
 * are rescaled from the eigenvalues so the major axis stays R; at most SHAPE_ITERATIONS, stop
 * when both ratios change by less than 1 %. Positions about cen (translated comoving). Leaves
 * the outputs untouched (-1) when np or the first ellipsoid holds fewer than SHAPE_MINPART
 * particles; if a later iteration does, the previous iteration's result is kept. The major
 * axis is a unit vector with its largest component positive. */
static void
allgood_shape(const struct RPair * pr, int64_t np, const double cen[3], double R, double rfloor, double BoxSize,
              gsl_eigen_symmv_workspace * ws, float * BtoA, float * CtoA, float AxisA[3])
{
    if(np < SHAPE_MINPART || !(R > 0))
        return;
    double orth[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};   /* rows: the axis directions */
    double ax2[3] = {R * R, R * R, R * R};                  /* squared semi-axes */
    const double minq2 = (rfloor / R) * (rfloor / R);
    double ba_prev = 0, ca_prev = 0;
    int it, d, e;
    for(it = 0; it < SHAPE_ITERATIONS; it++) {
        double T[9] = {0}, wsum = 0;
        int64_t j, cnt = 0;
        for(j = 0; j < np; j++) {
            const int64_t p = pr[j].p;
            double dx[3], q2 = 0;
            for(d = 0; d < 3; d++)
                dx[d] = NEAREST(P[p].Pos[d] - cen[d], BoxSize);
            for(e = 0; e < 3; e++) {
                const double pe = orth[e][0] * dx[0] + orth[e][1] * dx[1] + orth[e][2] * dx[2];
                q2 += pe * pe / ax2[e];
            }
            if(q2 < minq2)
                q2 = minq2;
            if(!(q2 <= 1))
                continue;
            const double w = P[p].Mass / q2;
            cnt++;
            wsum += w;
            for(d = 0; d < 3; d++)
                for(e = 0; e <= d; e++)
                    T[3 * d + e] += w * dx[d] * dx[e];
        }
        if(cnt < SHAPE_MINPART || !(wsum > 0))
            return;
        for(d = 0; d < 3; d++)
            for(e = 0; e <= d; e++) {
                T[3 * d + e] /= wsum;
                T[3 * e + d] = T[3 * d + e];
            }
        double eval[3], evec[9];
        eigen3(T, eval, evec, ws);
        if(!(eval[0] > 0 && eval[1] > 0 && eval[2] > 0))
            return;
        const double ba = sqrt(eval[1] / eval[0]), ca = sqrt(eval[2] / eval[0]);
        /* the major axis, sign fixed by its largest component */
        double A[3] = {evec[0], evec[3], evec[6]};
        int big = 0;
        for(d = 1; d < 3; d++)
            if(fabs(A[d]) > fabs(A[big]))
                big = d;
        const double sgn = A[big] < 0 ? -1 : 1;
        *BtoA = (float) ba;
        *CtoA = (float) ca;
        for(d = 0; d < 3; d++)
            AxisA[d] = (float) (sgn * A[d]);
        if(fabs(ba - ba_prev) < 0.01 * ba_prev && fabs(ca - ca_prev) < 0.01 * ca_prev)
            return;
        ba_prev = ba;
        ca_prev = ca;
        /* new ellipsoid: axes along the eigenvectors, semi-axes proportional to sqrt(eval),
         * the major one equal to R (properties.c:307-311) */
        for(e = 0; e < 3; e++) {
            for(d = 0; d < 3; d++)
                orth[e][d] = evec[3 * d + e];
            ax2[e] = R * R * eval[e] / eval[0];
        }
    }
}

/* Concentration from f = c / (ln(1 + c) - c / (1 + c)), Rockstar f_to_c (nfw.c:19-32): Newton
 * with a forward-difference slope from c = f, halving c on a negative step. Rockstar iterates
 * without a cap; here at most 200 iterations, returning -1 on failure. */
static double
klypin_c_to_f(double c)
{
    const double cp1 = 1.0 + c;
    return c * cp1 / (log1p(c) * cp1 - c);
}

static double
klypin_f_to_c(double f)
{
    double c = f, tc = klypin_c_to_f(c);
    int it;
    for(it = 0; it < 200; it++) {
        if(fabs((f - tc) / f) <= 1e-7)
            return c;
        const double slope = (klypin_c_to_f(c + 0.1) - tc) / 0.1;
        const double cn = c + (f - tc) / slope;
        if(!subhalo_isfinite(cn))
            return -1;
        if(cn < 0)
            c /= 2;
        else
            c = cn;
        tc = klypin_c_to_f(c);
    }
    return -1;
}

/* Rockstar's RMAX_TO_RS and RS_CONSTANT (nfw.c, the NFW values Rmax / Rs and
 * Vmax^2 Rvir / (G Mvir) * f(c) / c). */
#define KLYPIN_RMAX_TO_RS 2.1626
#define KLYPIN_RS_CONSTANT 0.216216

void
subhalo_props_rockstar(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       const double rho[SUBHALO_NSO], int CoreSize, const double Offset[3], struct SubhaloArena * arena)
{
    const int64_t n = c->Nbound;
    const double a = up->atime, G = up->G, BoxSize = up->BoxSize;
    const double rfloor = up->h / 2.8;               /* GravitySoftening, Plummer-equivalent */
    const double vu = subhalo_vel_to_kms(1.0);       /* internal velocity unit -> km/s */
    const double vkms = subhalo_vel_to_kms(a);       /* internal Vel -> physical peculiar km/s */
    int64_t i, k;
    int d, s;
    (void) Offset;

    /* every column this function owns is absent until computed */
    for(s = 0; s < SUBHALO_NSO; s++)
        rec->MBoundSO[s] = rec->RBoundSO[s] = -1;
    rec->VmaxVir = rec->RmaxVir = rec->Rhalf = rec->Renclose = -1;
    for(d = 0; d < 3; d++) {
        rec->J[d] = rec->JVir[d] = -1;
        rec->AxisA[d] = rec->AxisA500[d] = -1;
    }
    rec->SpinPeebles = rec->SpinBullock = rec->Xoff = rec->Voff = rec->RsKlypin = -1;
    rec->BtoA = rec->CtoA = rec->BtoA500 = rec->CtoA500 = -1;
    for(d = 0; d < 6; d++)
        rec->InertiaTensor[d] = -1;
    if(n <= 0)
        return;

    /* the core, recomputed in the translated frame and internal units so that the result does
     * not depend on whether the caller has converted rec->CorePos/CoreVel yet (it is the
     * subhalo_props_hbt core: same function, same order of Index) */
    double cpos[3], cvel[3], sx, sv;
    subhalo_core(c, up, CoreSize, cpos, cvel, &sx, &sv);

    /* radial order from the core, radii floored at the softening */
    const size_t mark = arena->used;
    struct RPair * pr = (struct RPair *) subhalo_arena_get(arena, sizeof(struct RPair) * n);
    if(!pr)
        endrun(1, "Subhalo properties: arena exhausted for the radial order of %ld particles (%lu of %lu bytes used)\n",
               (long) n, (unsigned long) arena->used, (unsigned long) arena->size);
    double Mtot = 0;
    for(i = 0; i < n; i++) {
        const int64_t p = c->Index[i];
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[p].Pos[d] - cpos[d], BoxSize);
            r2 += dx * dx;
        }
        const double r = sqrt(r2);
        pr[i].r = r > rfloor ? r : rfloor;
        pr[i].p = p;
        Mtot += P[p].Mass;
    }
    rpair_sort(pr, n);

    /* cumulative profile (properties.c:383-415): outermost crossing per definition, VmaxVir over
     * the particles whose enclosed density exceeds the virial threshold, the half-mass radius */
    int64_t kc[SUBHALO_NSO];
    double Mso[SUBHALO_NSO];
    for(s = 0; s < SUBHALO_NSO; s++) {
        kc[s] = -1;
        Mso[s] = 0;
    }
    double M = 0, best = 0, rbest = -1, rhalf = -1;
    for(k = 0; k < n; k++) {
        const double r = pr[k].r;
        M += P[pr[k].p].Mass;
        const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
        for(s = 0; s < SUBHALO_NSO; s++)
            if(dens > rho[s]) {
                kc[s] = k;
                Mso[s] = M;
            }
        if(dens > rho[0] && M / r > best) {
            best = M / r;
            rbest = r;
        }
        if(rhalf < 0 && M >= 0.5 * Mtot)
            rhalf = r;
    }
    for(s = 0; s < SUBHALO_NSO; s++)
        if(kc[s] >= 0) {
            rec->MBoundSO[s] = (float) Mso[s];
            rec->RBoundSO[s] = (float) cbrt(3 * Mso[s] / (4 * M_PI * rho[s]));
        }
    rec->Rhalf = (float) rhalf;
    rec->Renclose = (float) pr[n - 1].r;
    const double Mvir = Mso[0];
    const double Rvir = kc[0] >= 0 ? cbrt(3 * Mvir / (4 * M_PI * rho[0])) : -1;
    if(kc[0] >= 0 && best > 0) {
        rec->VmaxVir = (float) (sqrt(G * best / a) * vu);
        rec->RmaxVir = (float) rbest;
    }

    /* J (full set, bound-CoM velocity frame), the inertia tensor (full set), and inside the
     * virial aperture (crossing particle included, D4) JVir in the CoreVel frame and the mass
     * weighted mean position and velocity for Xoff/Voff (properties.c:417-458). Positions about
     * the core; dx x (Vel - V) is a dx x (v - V / a), the physical specific angular momentum. */
    double J[3] = {0, 0, 0}, Jv[3] = {0, 0, 0}, xs[3] = {0, 0, 0}, vs[3] = {0, 0, 0}, I[6] = {0, 0, 0, 0, 0, 0};
    for(k = 0; k < n; k++) {
        const int64_t p = pr[k].p;
        const double m = P[p].Mass;
        double dx[3], dvc[3], dvk[3];
        for(d = 0; d < 3; d++) {
            dx[d] = NEAREST(P[p].Pos[d] - cpos[d], BoxSize);
            dvc[d] = P[p].Vel[d] - c->RefVel[d];
            dvk[d] = P[p].Vel[d] - cvel[d];
        }
        J[0] += m * (dx[1] * dvc[2] - dx[2] * dvc[1]);
        J[1] += m * (dx[2] * dvc[0] - dx[0] * dvc[2]);
        J[2] += m * (dx[0] * dvc[1] - dx[1] * dvc[0]);
        I[0] += m * dx[0] * dx[0];
        I[1] += m * dx[1] * dx[1];
        I[2] += m * dx[2] * dx[2];
        I[3] += m * dx[0] * dx[1];
        I[4] += m * dx[0] * dx[2];
        I[5] += m * dx[1] * dx[2];
        if(k <= kc[0]) {
            Jv[0] += m * (dx[1] * dvk[2] - dx[2] * dvk[1]);
            Jv[1] += m * (dx[2] * dvk[0] - dx[0] * dvk[2]);
            Jv[2] += m * (dx[0] * dvk[1] - dx[1] * dvk[0]);
            for(d = 0; d < 3; d++) {
                xs[d] += m * dx[d];
                vs[d] += m * dvk[d];
            }
        }
    }
    double jmag = 0;
    for(d = 0; d < 3; d++) {
        J[d] /= Mtot;
        jmag += J[d] * J[d];
        rec->J[d] = (float) (J[d] * vu);
    }
    jmag = sqrt(jmag);
    for(d = 0; d < 6; d++)
        rec->InertiaTensor[d] = (float) (I[d] / Mtot);

    /* Peebles spin from J and the final-pass specific self-energy of the full set (7.10):
     * Ekin + Epot with each pair once, as subhalo_props_hbt computes them */
    if(c->Energy) {
        double K = 0, U = 0;
        for(i = 0; i < n; i++) {
            const int64_t p = c->Index[i];
            const double m = P[p].Mass;
            double k2 = 0;
            for(d = 0; d < 3; d++) {
                const double dx = NEAREST(P[p].Pos[d] - c->RefPos[d], BoxSize);
                const double dv = (P[p].Vel[d] - c->RefVel[d]) / a + up->Hubble * a * dx;
                k2 += dv * dv;
            }
            double phi = c->Energy[i] - 0.5 * k2;
            if(up->WithThermal)
                phi -= subhalo_thermal_energy(p, a);
            K += m * 0.5 * k2;
            U += m * phi;
        }
        const double eps = (K + 0.5 * U) / Mtot;
        if(Mtot > 0)
            rec->SpinPeebles = (float) (jmag * sqrt(fabs(eps)) / (G * Mtot));
    }

    if(kc[0] >= 0 && Mvir > 0) {
        double jv = 0, xo = 0, vo = 0;
        for(d = 0; d < 3; d++) {
            Jv[d] /= Mvir;
            jv += Jv[d] * Jv[d];
            rec->JVir[d] = (float) (Jv[d] * vu);
            xo += (xs[d] / Mvir) * (xs[d] / Mvir);
            vo += (vs[d] / Mvir) * (vs[d] / Mvir);
        }
        rec->SpinBullock = (float) (sqrt(jv) / sqrt(2 * G * Mvir * a * Rvir));
        rec->Xoff = (float) sqrt(xo);
        rec->Voff = (float) (sqrt(vo) * vkms);
        /* Klypin Rs (nfw.c:34-45): f = Rvir Vmax^2 a / (G Mvir RS_CONSTANT), with
         * Vmax^2 a / G = max M / r (comoving) */
        if(best > 0) {
            const double f = Rvir * best / (Mvir * KLYPIN_RS_CONSTANT);
            double rs = rbest / KLYPIN_RMAX_TO_RS;
            if(f >= 4.625) {
                const double conc = klypin_f_to_c(f);
                if(conc > 0)
                    rs = Rvir / conc;
            }
            rec->RsKlypin = (float) rs;
        }
    }

    /* shapes at Rvir and R500c over the particles up to each crossing */
    gsl_eigen_symmv_workspace * ws = gsl_eigen_symmv_alloc(3);
    if(!ws)
        endrun(1, "Subhalo properties: gsl_eigen_symmv_alloc failed\n");
    if(kc[0] >= 0)
        allgood_shape(pr, kc[0] + 1, cpos, Rvir, rfloor, BoxSize, ws, &rec->BtoA, &rec->CtoA, rec->AxisA);
    if(kc[3] >= 0)
        allgood_shape(pr, kc[3] + 1, cpos, cbrt(3 * Mso[3] / (4 * M_PI * rho[3])), rfloor, BoxSize, ws,
                      &rec->BtoA500, &rec->CtoA500, rec->AxisA500);
    gsl_eigen_symmv_free(ws);

    arena->used = mark;
}

void
subhalo_props_baryons(struct SubTrack * rec, const struct SubCandidate * c, int MetalReturnOn)
{
    const int64_t n = c->Nbound;
    int64_t i;
    int j;
    rec->StarFormationRate = rec->GasMetalMass = rec->StellarMetalMass = -1;
    rec->BlackholeMass = rec->BlackholeAccretionRate = rec->MassHeIonized = -1;
    for(j = 0; j < NMETALS; j++)
        rec->GasMetalElemMass[j] = rec->StellarMetalElemMass[j] = -1;
    rec->MostMassiveBHID = (MyIDType) -1;
    if(n <= 0)
        return;
    /* add_particle_to_group (fof.c:631-667) over the bound set */
    double sfr = 0, gz = 0, sz = 0, heiii = 0, bhm = 0, mdot = 0, gze[NMETALS], sze[NMETALS];
    double bhbest = -1;
    MyIDType bhid = (MyIDType) -1;
    for(j = 0; j < NMETALS; j++)
        gze[j] = sze[j] = 0;
    for(i = 0; i < n; i++) {
        const int64_t p = c->Index[i];
        const double m = P[p].Mass;
        if(P[p].Type == 0) {
            heiii += m * P[p].HeIIIionized;
            sfr += SPHP(p).Sfr;
            gz += SPHP(p).Metallicity * m;
            for(j = 0; j < NMETALS; j++)
                gze[j] += SPHP(p).Metals[j] * m;
        }
        else if(P[p].Type == 4) {
            sz += STARP(p).Metallicity * m;
            for(j = 0; j < NMETALS; j++)
                sze[j] += STARP(p).Metals[j] * m;
        }
        else if(P[p].Type == 5) {
            const double bm = BHP(p).Mass;
            bhm += bm;
            mdot += BHP(p).Mdot;
            if(bm > bhbest || (bm == bhbest && P[p].ID < bhid)) {
                bhbest = bm;
                bhid = P[p].ID;
            }
        }
    }
    rec->StarFormationRate = (float) sfr;
    rec->MassHeIonized = (float) heiii;
    rec->BlackholeMass = (float) bhm;
    rec->BlackholeAccretionRate = (float) mdot;
    rec->MostMassiveBHID = bhid;
    if(MetalReturnOn) {
        rec->GasMetalMass = (float) gz;
        rec->StellarMetalMass = (float) sz;
        for(j = 0; j < NMETALS; j++) {
            rec->GasMetalElemMass[j] = (float) gze[j];
            rec->StellarMetalElemMass[j] = (float) sze[j];
        }
    }
}
