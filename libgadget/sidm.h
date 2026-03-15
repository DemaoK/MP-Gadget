#ifndef SIDM_H
#define SIDM_H

#include "forcetree.h"
#include "types.h"
#include "timestep.h"
#include "density.h"
#include "utils/paramset.h"
#include "utils/system.h"
/* Compute SIDM scattering impulses for gravity-active DM particles. */
void sidm_force(const ActiveParticles * act, const double atime, const double hubble, Cosmology * CP, RandTable * rnd, const struct UnitSystem units, DriftKickTimes * times, const ForceTree * const tree);

void set_sidm_params(ParameterSet * ps);

#endif
