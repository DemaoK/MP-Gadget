#ifndef GRAVSHORT_H
#define GRAVSHORT_H

#include "partmanager.h"
#include "treewalk.h"
#include "gravity.h"
#include <math.h>

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterGravShort;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat OldAcc;
} TreeWalkQueryGravShort;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat Potential;
} TreeWalkResultGravShort;

struct GravShortPriv {
    /* Size of a PM cell, in internal units. Box / Nmesh */
    double cellsize;
    /* How many PM cells do we go
     * before we stop calculating the tree?*/
    double Rcut;
    /* Newton's constant in internal units*/
    double G;
    inttime_t Ti_Current;
    /* Matter density in internal units.
     * rho_0 = Omega0 * rho_crit
     * rho_crit = 3 H^2 /(8 pi G).
     * This is (rho_0)^(1/3) ,
     * Note: should account for
     * massive neutrinos, but doesn't. */
    double cbrtrho0;
    /* Pointer to the place to store accelerations*/
    MyFloat (*Accel)[3];
};

#define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

/* Dimensionless potential polynomial wp(u), 0 <= u = r / h < 1, of the Gadget-2 spline kernel
 * with h = FORCE_SOFTENING(): inside h the potential of a mass m is G m wp(u) / h (wp(0) = -2.8,
 * wp(1) = -1). Shared by apply_accn_to_output (gravshort-tree.c), which must stay bit-identical,
 * and the subhalo unbinding kernel (subhalo_unbind.c). */
static inline double
grav_spline_wp(const double u)
{
    if(u < 0.5)
        return -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
    else
        return -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
}

/* W(r, h) >= 0 such that the (Newtonian, no long-range window) potential of a point mass m at
 * comoving distance r is -G m W(r, h) / a in physical units: W = 1 / r for r >= h and
 * -wp(r / h) / h inside the softening (W(0, h) = 2.8 / h). */
static inline double
grav_spline_potential(const double r, const double h)
{
    if(r >= h)
        return 1.0 / r;
    return -grav_spline_wp(r / h) / h;
}

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    double G = GRAV_GET_PRIV(tw)->G;
    GRAV_GET_PRIV(tw)->Accel[i][0] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][1] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][2] *= G;

    if(tw->tree->full_particle_tree_flag) {
        /* On a PM step, update the stored full tree grav accel for the next PM step.
         * Needs to be done here so internal treewalk iterations don't get a partial acceleration.*/
        P[i].FullTreeGravAccel[0] = GRAV_GET_PRIV(tw)->Accel[i][0];
        P[i].FullTreeGravAccel[1] = GRAV_GET_PRIV(tw)->Accel[i][1];
        P[i].FullTreeGravAccel[2] = GRAV_GET_PRIV(tw)->Accel[i][2];
        /* calculate the potential */
        P[i].Potential += P[i].Mass / (FORCE_SOFTENING() / 2.8);
        /* remove self-potential */
        P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) * GRAV_GET_PRIV(tw)->cbrtrho0;
        P[i].Potential *= G;
    }
}

/*Compute the absolute magnitude of the acceleration for a particle.*/
static MyFloat
grav_get_abs_accel(struct particle_data * PP, const double G)
{
    double aold=0;
    int j;
    for(j = 0; j < 3; j++) {
       double ax = PP->FullTreeGravAccel[j] + PP->GravPM[j];
       aold += ax*ax;
    }
    return sqrt(aold) / G;
}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->OldAcc = grav_get_abs_accel(&P[place], GRAV_GET_PRIV(tw)->G);
}

static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][0], result->Acc[0]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][1], result->Acc[1]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][2], result->Acc[2]);
    if(tw->tree->full_particle_tree_flag)
        TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

#endif
