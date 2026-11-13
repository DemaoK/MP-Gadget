#ifndef SUBHALO_PROPS_H
#define SUBHALO_PROPS_H

/* Sums of a bound set to record columns, inlined into the serial key and the in-place rounds; each
 * caller's -ffast-math context fixes the compiled form of its columns (design 7.5). */

#include <math.h>
#include <stdint.h>
#include "subhalo_unbind.h"

#define SUBHALO_FINISH static inline __attribute__((always_inline))

/* Internal units: positions translated comoving, velocities in Vel units (a v_pec). */
struct SubhaloPropSums {
    /* HBT (subhalo_finish_hbt) */
    int64_t N, NTracer, NType[6];
    double M, MType[6];            /* the bound mass */
    int Flags;                     /* or'ed into the record */
    struct SubhaloCore core;
    double CoMPos[3], CoMVel[3];   /* the final frame */
    MyIDType MostBoundID;
    double MostBoundPos[3], MostBoundVel[3];
    double Best, Rmax;             /* max M(<= r) / r of the whole profile from the core, its radius */
    double K, U;                   /* sum m k2 / 2 and sum m phi in the final frame */
    /* Rockstar profile (subhalo_finish_profile) */
    int64_t Cross[SUBHALO_NSO];    /* the outermost crossing of each threshold, -1 none */
    double Mso[SUBHALO_NSO];       /* M(<= r) there */
    double BestVir, RmaxVir;       /* max M / r inside the virial threshold (0 none), its radius */
    double Rhalf, Renclose;
    /* baryons (subhalo_finish_baryons) */
    double Sfr, HeIII, BhMass, BhMdot, GasZ, StarZ, GasZElem[NMETALS], StarZElem[NMETALS];
    MyIDType BhId;                 /* the most massive BH, (MyIDType) -1 none */
};

/* The unit factors of the finishing, taken before it (the calls stay out of the formulas). */
struct SubhaloPropUnits {
    double vu;                     /* internal velocity unit -> km/s */
    double vkms;                   /* internal Vel -> physical peculiar km/s */
    double lenfac;                 /* the run's length unit -> kpc/h: J, JVir in kpc/h km/s for any unit (7.10) */
};

SUBHALO_FINISH void
subhalo_finish_units(struct SubhaloPropUnits * u, double atime)
{
    double ULen, UMass, UVel;
    u->vu = subhalo_vel_to_kms(1.0);
    u->vkms = subhalo_vel_to_kms(atime);
    subhalo_get_units(&ULen, &UMass, &UVel);
    u->lenfac = ULen / 3.085678e21;
}

/* The HBT counts, masses and flags (any N). */
SUBHALO_FINISH void
subhalo_finish_counts(struct SubTrack * rec, const struct SubhaloPropSums * s)
{
    int t;
    rec->Nbound = (int32_t) s->N;
    rec->NboundTracer = (int32_t) s->NTracer;
    rec->Mbound = (float) s->M;
    for(t = 0; t < 6; t++) {
        rec->NboundType[t] = (int32_t) s->NType[t];
        rec->MboundType[t] = (float) s->MType[t];
    }
    rec->Flags |= s->Flags;
}

/* The other HBT columns (N > 0): core, CoM, most-bound tracer, Vmax/Rmax and energies; positions
 * with Offset removed and wrapped, velocities Vel / a. */
SUBHALO_FINISH void
subhalo_finish_hbt(struct SubTrack * rec, const struct SubhaloPropSums * s, const struct SubhaloUnbindParams * up,
                   const double Offset[3])
{
    const double a = up->atime, BoxSize = up->BoxSize;
    int d;
    for(d = 0; d < 3; d++) {
        rec->CorePos[d] = subhalo_wrap_box(s->core.Pos[d] - Offset[d], BoxSize);
        rec->CoreVel[d] = (float) (s->core.Vel[d] / a);
        rec->CoMPos[d] = subhalo_wrap_box(s->CoMPos[d] - Offset[d], BoxSize);
        rec->CoMVel[d] = (float) (s->CoMVel[d] / a);
    }
    rec->CoreSigmaX = (float) s->core.SigmaX;
    rec->CoreSigmaV = (float) (s->core.SigmaV / a);
    rec->MostBoundID = s->MostBoundID;
    for(d = 0; d < 3; d++) {
        rec->MostBoundPos[d] = subhalo_wrap_box(s->MostBoundPos[d] - Offset[d], BoxSize);
        rec->MostBoundVel[d] = (float) (s->MostBoundVel[d] / a);
    }
    rec->Vmax = (float) sqrt(up->G * s->Best / a);
    rec->Rmax = (float) s->Rmax;
    rec->Ekin = (float) (s->K / s->M);
    rec->Epot = (float) (0.5 * s->U / s->M);
    rec->KinToPot = rec->Epot != 0 ? (float) (s->K / s->M / fabs(0.5 * s->U / s->M)) : -1;
}

/* The unit factor of the HBT velocities and energies: Vel / a to km/s, applied to the stored floats. */
SUBHALO_FINISH void
subhalo_finish_kms(struct SubTrack * rec, double vu)
{
    int d;
    for(d = 0; d < 3; d++) {
        rec->CoreVel[d] *= vu;
        rec->MostBoundVel[d] *= vu;
        rec->CoMVel[d] *= vu;
    }
    rec->CoreSigmaV *= vu;
    rec->Vmax *= vu;
    rec->Ekin *= vu * vu;
    rec->Epot *= vu * vu;
}

/* Every Rockstar column (profile, moments, shapes): -1, absent until computed. */
SUBHALO_FINISH void
subhalo_finish_rockstar_undefined(struct SubTrack * rec)
{
    int d, sd;
    for(sd = 0; sd < SUBHALO_NSO; sd++)
        rec->MBoundSO[sd] = rec->RBoundSO[sd] = -1;
    rec->VmaxVir = rec->RmaxVir = rec->Rhalf = rec->Renclose = -1;
    for(d = 0; d < 3; d++) {
        rec->J[d] = rec->JVir[d] = -1;
        rec->AxisA[d] = rec->AxisA500[d] = -1;
    }
    rec->SpinPeebles = rec->SpinBullock = rec->Xoff = rec->Voff = rec->RsKlypin = -1;
    rec->BtoA = rec->CtoA = rec->BtoA500 = rec->CtoA500 = -1;
    for(d = 0; d < 6; d++)
        rec->InertiaTensor[d] = -1;
}

/* The SO masses and radii, Rhalf, Renclose and VmaxVir/RmaxVir (N > 0, after
 * subhalo_finish_rockstar_undefined); J, the inertia tensor, spins and offsets are each caller's. */
SUBHALO_FINISH void
subhalo_finish_profile(struct SubTrack * rec, const struct SubhaloPropSums * s, const struct SubhaloUnbindParams * up,
                       const double rho[SUBHALO_NSO], const struct SubhaloPropUnits * u)
{
    const double a = up->atime, G = up->G, vu = u->vu;
    const int64_t * kc = s->Cross;
    int sd;

    for(sd = 0; sd < SUBHALO_NSO; sd++)
        if(kc[sd] >= 0) {
            rec->MBoundSO[sd] = (float) s->Mso[sd];
            rec->RBoundSO[sd] = (float) cbrt(3 * s->Mso[sd] / (4 * M_PI * rho[sd]));
        }
    rec->Rhalf = (float) s->Rhalf;
    rec->Renclose = (float) s->Renclose;
    const double best = s->BestVir, rbest = s->RmaxVir;
    if(kc[0] >= 0 && best > 0) {
        rec->VmaxVir = (float) (sqrt(G * best / a) * vu);
        rec->RmaxVir = (float) rbest;
    }
}

/* Every baryonic column: -1, absent until computed. */
SUBHALO_FINISH void
subhalo_finish_baryons_undefined(struct SubTrack * rec)
{
    int j;
    rec->StarFormationRate = rec->GasMetalMass = rec->StellarMetalMass = -1;
    rec->BlackholeMass = rec->BlackholeAccretionRate = rec->MassHeIonized = -1;
    for(j = 0; j < NMETALS; j++)
        rec->GasMetalElemMass[j] = rec->StellarMetalElemMass[j] = -1;
    rec->MostMassiveBHID = (MyIDType) -1;
}

/* The baryonic columns of fof.c's sums over the bound set (N > 0, after
 * subhalo_finish_baryons_undefined); the metal masses only with MetalReturnOn. */
SUBHALO_FINISH void
subhalo_finish_baryons(struct SubTrack * rec, const struct SubhaloPropSums * s, int MetalReturnOn)
{
    int j;
    rec->StarFormationRate = (float) s->Sfr;
    rec->MassHeIonized = (float) s->HeIII;
    rec->BlackholeMass = (float) s->BhMass;
    rec->BlackholeAccretionRate = (float) s->BhMdot;
    rec->MostMassiveBHID = s->BhId;
    if(MetalReturnOn) {
        rec->GasMetalMass = (float) s->GasZ;
        rec->StellarMetalMass = (float) s->StarZ;
        for(j = 0; j < NMETALS; j++) {
            rec->GasMetalElemMass[j] = (float) s->GasZElem[j];
            rec->StellarMetalElemMass[j] = (float) s->StarZElem[j];
        }
    }
}

#undef SUBHALO_FINISH

#endif
