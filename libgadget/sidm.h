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

/* Return sigma/m in internal comoving code units for a squared internal
 * relative velocity. This intentionally shares the vdSIDM shape and unit
 * conversion used by sidm_force(). */
double sidm_sigma_over_m_code(double vrel2, double atime, Cosmology * CP, const struct UnitSystem units);

void set_sidm_params(ParameterSet * ps);

#endif
