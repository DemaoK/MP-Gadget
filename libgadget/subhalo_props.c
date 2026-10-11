/* Subhalo properties of a bound set (design 7.5, D4, section 8; declared in subhalo_unbind.h): the
 * HBT set, Rockstar's recipes, the deferred NFW fit and velocities, and the baryonic sums.
 * Input: a candidate after subhalo_final_pass, Index[0, Nbound) in final-pass binding order (with the
 * sampled final pass: the loop's order, energies unsorted estimates read only by order-free sums). */

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

/* (radius, P index) pairs of the radial order, ties by particle ID then P index. 16 bytes per
 * bound particle: the only per-particle scratch of the property functions. */
struct RPair {
    double r;
    int64_t p;
};

static inline int
rpair_less(const struct RPair * x, const struct RPair * y)
{
    if(x->r != y->r)
        return x->r < y->r;
    return P[x->p].ID < P[y->p].ID || (P[x->p].ID == P[y->p].ID && x->p < y->p);
}
#define RPAIR_LESS(x, y) rpair_less((x), (y))

static void rpair_sort(struct RPair * a, int64_t n);

/* The radial order of the bound set Index[0, Nbound) about cen (translated comoving), radii
 * NEAREST-periodic and floored at rfloor, in SUBHALO_PROPS_BYTES_PER_PART * Nbound of arena
 * scratch that the caller rewinds; endrun if the arena is short. *Mtot: the bound mass. */
static struct RPair *
radial_order(const struct SubCandidate * c, const double cen[3], double rfloor, double BoxSize,
             struct SubhaloArena * arena, double * Mtot)
{
    const int64_t n = c->Nbound;
    struct RPair * pr = (struct RPair *) (arena ? subhalo_arena_get(arena, sizeof(struct RPair) * n) : NULL);
    if(!pr)
        endrun(1, "Subhalo properties: arena exhausted for the radial order of %ld particles (%lu of %lu bytes used)\n",
               (long) n, (unsigned long) (arena ? arena->used : 0), (unsigned long) (arena ? arena->size : 0));
    double M = 0;
    int64_t i;
    int d;
    for(i = 0; i < n; i++) {
        const int64_t p = c->Index[i];
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[p].Pos[d] - cen[d], BoxSize);
            r2 += dx * dx;
        }
        const double r = sqrt(r2);
        pr[i].r = r > rfloor ? r : rfloor;
        pr[i].p = p;
        M += P[p].Mass;
    }
    rpair_sort(pr, n);
    *Mtot = M;
    return pr;
}

void
subhalo_props_hbt(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                  int CoreSize, const double Offset[3], struct SubhaloArena * arena)
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

    /* Vmax, Rmax over the whole bound profile from CorePos, radii floored at the softening (HBT's
     * CalculateProfileProperties): M(<= r) / r at the last of equal radii, Rmax the first maximum. */
    {
        const size_t mark = arena ? arena->used : 0;
        double Mtot, Mc = 0, best = -1, Rmax = 0;
        const struct RPair * pr = radial_order(c, CorePos, up->h / 2.8, BoxSize, arena, &Mtot);
        for(i = 0; i < n; i++) {
            Mc += P[pr[i].p].Mass;
            if(i + 1 < n && pr[i + 1].r == pr[i].r)
                continue;
            const double v = Mc / pr[i].r;
            if(v > best) {
                best = v;
                Rmax = pr[i].r;
            }
        }
        arena->used = mark;
        rec->Vmax = (float) sqrt(up->G * best / a);
        rec->Rmax = (float) Rmax;
    }

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

/* ================================================================== Rockstar recipes
 * Rockstar's properties on the HBT bound set (D4, 4.3, 7.10), after Rockstar-BigFile properties.c,
 * nfw.c and groupies.c: bound SO, VmaxVir, Rhalf, J, spins, Xoff/Voff, Klypin Rs, shapes, inertia. */

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

/* Allocation-free introsort, the algorithm of subhalo_unbind.c's private SUBHALO_DEFINE_SORT; the
 * order is strict and total, so the result is unique. */
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

/* Smallest aperture for which a shape is reported. */
#define SHAPE_MINPART 20
#define SHAPE_ITERATIONS 10

/* Allgood et al. (2006) reduced inertia tensor, Rockstar calc_shape (WEIGHTED_SHAPES): weights
 * m / q2 over pr[0, np) inside the ellipsoid of major axis R, at most SHAPE_ITERATIONS, until both
 * ratios change by < 1 %. Outputs untouched below SHAPE_MINPART particles (a later short iteration
 * keeps the previous result); AxisA is a unit vector with its largest component positive. */
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

/* f(c) = c / (ln(1 + c) - c / (1 + c)), the NFW function Rockstar's f_to_c inverts. */
static double
klypin_c_to_f(double c)
{
    const double cp1 = 1.0 + c;
    return c * cp1 / (log1p(c) * cp1 - c);
}

/* Rockstar's f_to_c (nfw.c:19-32): Newton from c = f with a forward-difference slope, halving c on
 * a negative step; at most 200 iterations (Rockstar has no cap), -1 on failure. */
double
subhalo_klypin_f_to_c(double f)
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

static void deferred_on_order(struct SubTrack * rec, const struct RPair * pr, int64_t n, const double cvel[3],
                              const double rho[SUBHALO_NSO], double rfloor, double vkms);

void
subhalo_props_rockstar(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       const double rho[SUBHALO_NSO], int CoreSize, const double Offset[3], struct SubhaloArena * arena)
{
    const int64_t n = c->Nbound;
    const double a = up->atime, G = up->G, BoxSize = up->BoxSize;
    const double rfloor = up->h / 2.8;               /* GravitySoftening, Plummer-equivalent */
    const double vu = subhalo_vel_to_kms(1.0);       /* internal velocity unit -> km/s */
    const double vkms = subhalo_vel_to_kms(a);       /* internal Vel -> physical peculiar km/s */
    /* J and JVir in physical kpc/h km/s whatever the run's length unit (design 7.10); the spins use
     * the internal sums */
    double ULen, UMass, UVel;
    subhalo_get_units(&ULen, &UMass, &UVel);
    const double lenfac = ULen / 3.085678e21;
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
    if(n <= 0) {
        deferred_on_order(rec, NULL, 0, NULL, rho, rfloor, vkms);   /* the deferred columns: -1 */
        return;
    }

    /* the core in the translated frame and internal units, independent of the caller's conversion
     * of rec->CorePos/CoreVel (the subhalo_props_hbt core: same function, same Index order) */
    double cpos[3], cvel[3], sx, sv;
    subhalo_core(c, up, CoreSize, cpos, cvel, &sx, &sv);

    /* radial order from the core, radii floored at the softening (as subhalo_props_hbt's; rebuilt so
     * the two calls stay independent) */
    const size_t mark = arena->used;
    double Mtot;
    struct RPair * pr = radial_order(c, cpos, rfloor, BoxSize, arena, &Mtot);

    /* cumulative profile (properties.c:383-415): outermost crossing per definition, VmaxVir over
     * the particles whose enclosed density exceeds the virial threshold, the half-mass radius */
    int64_t kc[SUBHALO_NSO];
    double Mso[SUBHALO_NSO];
    for(s = 0; s < SUBHALO_NSO; s++) {
        kc[s] = -1;
        Mso[s] = 0;
    }
    /* M(<= r) as a compensated sum: the value of the distributed profile of subhalo_collective.c
     * for the same radial order, exactly so for equal masses */
    double Mk[2] = {0, 0}, best = 0, rbest = -1, rhalf = -1;
    for(k = 0; k < n; k++) {
        const double r = pr[k].r;
        subhalo_ksum_add(Mk, P[pr[k].p].Mass);
        const double M = subhalo_ksum_value(Mk);
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

    /* J and the inertia tensor over the full set (J in the bound-CoM velocity frame); JVir and the
     * Xoff/Voff means inside the virial aperture, crossing particle included (D4), in the core frame.
     * Positions about the core; dx x (Vel - V) = a dx x (v - V / a), the physical specific J. */
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
        rec->J[d] = (float) (J[d] * vu * lenfac);
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
            rec->JVir[d] = (float) (Jv[d] * vu * lenfac);
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
                const double conc = subhalo_klypin_f_to_c(f);
                if(conc > 0)
                    rs = Rvir / conc;
            }
            rec->RsKlypin = (float) rs;
        }
    }

    /* the deferred recipes (Rs fit, RockstarCoreVel, BulkVel, Vrms) on the same order and core */
    deferred_on_order(rec, pr, n, cvel, rho, rfloor, vkms);

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

/* ================================================================== deferred recipes
 * Rockstar's NFW fit (nfw.c:51-144), core/bulk velocities and Vrms (properties.c) on the bound virial
 * aperture; the bin and fit helpers are shared with the distributed version in subhalo_collective.c. */

double
subhalo_klypin_rs(double Rvir, double Mvir, double best, double rbest)
{
    if(!(best > 0) || !(Mvir > 0) || !(Rvir > 0))
        return -1;
    const double f = Rvir * best / (Mvir * KLYPIN_RS_CONSTANT);
    double rs = rbest / KLYPIN_RMAX_TO_RS;
    if(f >= 4.625) {
        const double conc = subhalo_klypin_f_to_c(f);
        if(conc > 0)
            rs = Rvir / conc;
    }
    return rs;
}

int
subhalo_nfw_bin_plan(int64_t nfit, double Mfit, double mmax, double * mpbin)
{
    *mpbin = 0;
    if(nfit < SUBHALO_NFW_MINPART || !(Mfit > 0))
        return 0;
    /* nfw.c:115-121 */
    int64_t ppbin = (nfit + SUBHALO_NFW_MAXBINS - 1) / SUBHALO_NFW_MAXBINS;
    if(ppbin < SUBHALO_NFW_MINPERBIN)
        ppbin = nfit / (nfit / SUBHALO_NFW_MINPERBIN);
    int64_t nb = nfit / ppbin;
    if(nb > SUBHALO_NFW_MAXBINS)
        nb = SUBHALO_NFW_MAXBINS;
    double mp = Mfit / (double) nb;
    if(mp < mmax)
        mp = mmax;
    *mpbin = mp;
    return (int) nb;
}

double
subhalo_nfw_threshold(int b, double mpbin)
{
    return b * mpbin * (1 - 1e-9);
}

/* nfw.c:126-132's bin edge without its unit bug (the correction subtracted in Mpc/h from a midpoint
 * in kpc/h, so in effect the midpoint): both terms in the run's unit (deviations in subhalo_unbind.h). */
double
subhalo_nfw_edge(int b, double mpbin, double rprev, int has_prev, double r, double rnext, double Mc, double m)
{
    double e = 0.5 * (r + rnext);
    if(has_prev && m > 0)
        e -= (Mc - b * mpbin) / m * (r - rprev);
    return e;
}

/* NFW enclosed mass in units of 4 pi rho_s rs^3 */
static inline double
nfw_menc(double r, double rs)
{
    const double x = r / rs;
    return log1p(x) - x / (1 + x);
}

/* chi2_scale (nfw.c:51-62) */
static double
nfw_chi2(double rs, const double * edge, const double * w, int nb)
{
    const double bin_mass = nfw_menc(edge[nb], rs) / nb;
    double last = 0, chi2 = 0;
    int i;
    for(i = 0; i < nb; i++) {
        const double menc = nfw_menc(edge[i + 1], rs);
        const double dx = w[i] * ((menc - last) - bin_mass) / bin_mass;
        chi2 += dx * dx;
        last = menc;
    }
    return chi2;
}

double
subhalo_nfw_fit(double rs0, const double * edge, int nb, double wcut)
{
    double w[SUBHALO_NFW_MAXBINS];
    int i;
    if(nb < 3 || nb > SUBHALO_NFW_MAXBINS)
        return -1;
    /* degenerate input: every edge finite, a positive extent */
    for(i = 0; i <= nb; i++)
        if(!subhalo_isfinite(edge[i]))
            return -1;
    for(i = 0; i < nb; i++)
        w[i] = edge[i] < wcut ? 0.1 : 1;
    const double rmin = edge[0], rmax = edge[nb];
    if(!(rmax > rmin))
        return -1;
    /* calc_scale_from_bins (nfw.c:64-99) */
    double rs = rs0;
    if(!subhalo_isfinite(rs) || !(rs > rmin) || !(rs < rmax))
        rs = edge[(nb + 1) / 2];
    if(!(rs > 0))
        return -1;
    double chi2 = nfw_chi2(rs, edge, w, nb);
    if(!subhalo_isfinite(chi2))
        return -1;
    double drs = rs / 10.0, last = 1e30;
    int iter = 0;
    /* Rockstar's stopping rule; a nonfinite objective, step or derivative fails the fit (-1) rather
     * than returning the radius reached (repeated edges at the start radius give drs = 0) */
    while(fabs(chi2 - last) > 0.005 * chi2 && iter < SUBHALO_NFW_MINPERBIN) {
        last = chi2;
        if(rs + drs > rmax)
            drs = 0.1 * (rmax - rs);
        if(rs - drs < rmin)
            drs = 0.1 * (rs - rmin);
        if(!(drs > 0) || !subhalo_isfinite(drs))
            return -1;
        const double cr = nfw_chi2(rs + drs, edge, w, nb);
        const double cl = nfw_chi2(rs - drs, edge, w, nb);
        const double dx = 0.5 * (cr - cl) / drs;
        const double dx2 = (cr + cl - 2.0 * chi2) / (drs * drs);
        if(!subhalo_isfinite(cr) || !subhalo_isfinite(cl) || !subhalo_isfinite(dx) || !subhalo_isfinite(dx2))
            return -1;
        double move = 0;
        if(dx2 != 0)
            move = -dx / dx2;
        if(!subhalo_isfinite(move))
            return -1;
        if(!move)
            break;              /* nfw.c: return rs */
        if(rs + move > 4 * rs)
            move = 3.0 * rs;
        if(rs + move < 0.25 * rs)
            move = -0.75 * rs;
        double nrs = rs + move;
        if(nrs > rmax)
            nrs = rs + 0.8 * (rmax - rs);
        else if(nrs < rmin)
            nrs = rs + 0.8 * (rmin - rs);
        double nchi2 = nfw_chi2(nrs, edge, w, nb);
        if(!subhalo_isfinite(nchi2))
            return -1;
        if(cr < nchi2) {
            nchi2 = cr;
            nrs = rs + drs;
        }
        if(cl < nchi2) {
            nchi2 = cl;
            nrs = rs - drs;
        }
        if(last < nchi2) {
            drs *= 0.5;
            continue;
        }
        drs = fabs(rs - nrs) / 10.0;
        rs = nrs;
        chi2 = nchi2;
        iter++;
    }
    return (subhalo_isfinite(rs) && rs > 0) ? rs : -1;
}

void
subhalo_deferred_velocities(float BulkVel[3], float CoreVel[3], float * Vrms, const double cvel[3], double Mb, double Mc,
                            double Sb[3][2], double S2[3][2], double Sc[3][2], double vkms)
{
    double var = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double mb = subhalo_ksum_value(Sb[d]) / Mb;
        var += subhalo_ksum_value(S2[d]) / Mb - mb * mb;
        BulkVel[d] = (float) ((cvel[d] + mb) * vkms);
        CoreVel[d] = (float) ((cvel[d] + subhalo_ksum_value(Sc[d]) / Mc) * vkms);
    }
    *Vrms = (float) (sqrt(var > 0 ? var : 0) * vkms);
}

/* subhalo_props_deferred's recipe on a built radial order pr[0, n) and core velocity (shared with
 * subhalo_props_rockstar: one sort per subhalo); the four columns are -1 first. Rockstar's aperture
 * (decisions 44, 47.1): with kc the outermost virial crossing, the velocities and the fit use
 * pr[0, kc), the crossing particle excluded, of mass Mfit = M(<= r_(kc-1)). */
static void
deferred_on_order(struct SubTrack * rec, const struct RPair * pr, int64_t n, const double cvel[3], const double rho[SUBHALO_NSO],
                  double rfloor, double vkms)
{
    int64_t k;
    int d;

    rec->Rs = rec->Vrms = -1;
    for(d = 0; d < 3; d++)
        rec->RockstarCoreVel[d] = rec->BulkVel[d] = -1;
    if(n <= 0)
        return;

    /* the virial crossing and the VmaxVir pair (the profile rule of subhalo_props_rockstar, its
     * compensated cumulative mass); Mfit the cumulative mass one particle inside the crossing */
    int64_t kc = -1;
    double Mk[2] = {0, 0}, Mvir = 0, Mfit = 0, best = 0, rbest = -1, Mkm1 = 0;
    for(k = 0; k < n; k++) {
        const double r = pr[k].r;
        subhalo_ksum_add(Mk, P[pr[k].p].Mass);
        const double M = subhalo_ksum_value(Mk);
        const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
        if(dens > rho[0]) {
            kc = k;
            Mvir = M;
            Mfit = Mkm1;
            if(M / r > best) {
                best = M / r;
                rbest = r;
            }
        }
        Mkm1 = M;
    }
    /* no crossing, or the crossing is the innermost particle (an empty aperture): nothing defined */
    if(kc < 1 || !(Mvir > 0))
        return;
    const int64_t nap = kc, nfit = kc;
    const double Rvir = cbrt(3 * Mvir / (4 * M_PI * rho[0]));
    const double rcore = 0.1 * Rvir;

    /* velocities about the core velocity (keeps the one-pass variance accurate), compensated sums as
     * the distributed recipe's; the core is r < 0.1 Rvir, at least SUBHALO_ROCKSTAR_CORE_MINPART
     * particles. mmax: the aperture's largest mass (nfw.c's max_mass). */
    int64_t ncore = 0;
    double mmax = 0;
    for(k = 0; k < nap; k++) {
        if(pr[k].r < rcore)
            ncore++;
        if(P[pr[k].p].Mass > mmax)
            mmax = P[pr[k].p].Mass;
    }
    if(ncore < SUBHALO_ROCKSTAR_CORE_MINPART)
        ncore = nap < SUBHALO_ROCKSTAR_CORE_MINPART ? nap : SUBHALO_ROCKSTAR_CORE_MINPART;
    double Sc[3][2], Sb[3][2], S2[3][2], Mc[2] = {0, 0}, Mb[2] = {0, 0};
    for(d = 0; d < 3; d++)
        Sc[d][0] = Sc[d][1] = Sb[d][0] = Sb[d][1] = S2[d][0] = S2[d][1] = 0;
    for(k = 0; k < nap; k++) {
        const int64_t p = pr[k].p;
        const double m = P[p].Mass;
        subhalo_ksum_add(Mb, m);
        if(k < ncore)
            subhalo_ksum_add(Mc, m);
        for(d = 0; d < 3; d++) {
            const double dv = P[p].Vel[d] - cvel[d];
            subhalo_ksum_add(Sb[d], m * dv);
            subhalo_ksum_add(S2[d], m * dv * dv);
            if(k < ncore)
                subhalo_ksum_add(Sc[d], m * dv);
        }
    }
    subhalo_deferred_velocities(rec->BulkVel, rec->RockstarCoreVel, &rec->Vrms, cvel, subhalo_ksum_value(Mb), subhalo_ksum_value(Mc),
                        Sb, S2, Sc, vkms);

    /* Rs: equal-mass bins of the aperture (calc_scale_radius, nfw.c:101-144), closed by the
     * compensated cumulative mass against Mfit's thresholds, as the distributed recipe */
    double mpbin;
    const int nb = subhalo_nfw_bin_plan(nfit, Mfit, mmax, &mpbin);
    if(nb > 0) {
        double edge[SUBHALO_NFW_MAXBINS + 1];
        int b = 1;
        edge[0] = 0;
        Mk[0] = Mk[1] = 0;
        for(k = 0; k < nfit && b <= nb; k++) {
            const double m = P[pr[k].p].Mass;
            subhalo_ksum_add(Mk, m);
            const double M = subhalo_ksum_value(Mk);
            /* while, as the distributed scan: one particle may close several bins only when
             * mpbin is the largest particle mass, and then both modes close them alike */
            while(b <= nb && M >= subhalo_nfw_threshold(b, mpbin)) {
                const double rnext = k + 1 < n ? pr[k + 1].r : pr[k].r;
                const double rprev = k > 0 ? pr[k - 1].r : 0;
                edge[b] = subhalo_nfw_edge(b, mpbin, rprev, k > 0, pr[k].r, rnext, M, m);
                b++;
            }
        }
        const double rs = subhalo_nfw_fit(subhalo_klypin_rs(Rvir, Mvir, best, rbest), edge, b - 1, 3 * rfloor);
        rec->Rs = (float) (rs > 0 ? rs : -1);
    }
}

void
subhalo_props_deferred(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       const double rho[SUBHALO_NSO], int CoreSize, struct SubhaloArena * arena)
{
    const int64_t n = c->Nbound;
    const double rfloor = up->h / 2.8;               /* GravitySoftening, Plummer-equivalent */
    if(n <= 0) {
        deferred_on_order(rec, NULL, 0, NULL, rho, rfloor, 0);
        return;
    }
    double cpos[3], cvel[3], sx, sv;
    subhalo_core(c, up, CoreSize, cpos, cvel, &sx, &sv);
    const size_t mark = arena->used;
    double Mtot;
    const struct RPair * pr = radial_order(c, cpos, rfloor, up->BoxSize, arena, &Mtot);
    deferred_on_order(rec, pr, n, cvel, rho, rfloor, subhalo_vel_to_kms(up->atime));
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
