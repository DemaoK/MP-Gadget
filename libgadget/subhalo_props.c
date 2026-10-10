/* HBT-level subhalo properties (design 7.5, Phase 2a plan section 4 and step 7g).
 *
 * Input: a candidate after subhalo_final_pass, i.e. Index[0, Nbound) in ascending final-pass
 * energy, Energy the final-pass energies in the frame (RefPos, RefVel). The Rockstar set of
 * Phase 3 is not touched. Declared in subhalo_unbind.h. */

#include <math.h>
#include <string.h>
#include <stdint.h>

#include "utils/endrun.h"
#include "partmanager.h"
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
