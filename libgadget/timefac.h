#ifndef __TIMEFAC_H
#define __TIMEFAC_H

#include "types.h"
#include "cosmology.h"

/* Get the exact drift and kick factors at given time by integrating. */
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s);

/* Bounded tables for SIDM drift and collision exposure. Initialize after the
 * integer timeline and cosmology, before querying from parallel loops. */
void sidm_time_factors_init(const Cosmology *CP);
/* D = integral dt/a^2, E = integral dt/a^4. Endpoints may cross sync points.
 * Source intervals use E; coordinate displacement budgets use D. */
double sidm_drift_factor(const Cosmology *CP, inttime_t ti0, inttime_t ti1);
double sidm_scatter_factor(const Cosmology *CP, inttime_t ti0, inttime_t ti1);
#endif
