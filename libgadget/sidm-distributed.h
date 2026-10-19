#ifndef SIDM_DISTRIBUTED_H
#define SIDM_DISTRIBUTED_H
#include "forcetree.h"
#include "timestep.h"
#include "utils/system.h"
void set_sidm_distributed_params(ParameterSet *ps);
/* Collective source-owned, single-kernel SIDM on current split-stage velocities.
 * The live type-1 particles have equal mass and are drifted to Ti_Current.
 * Other species are ignored. Intervals and returned limits are dt/a^4
 * exposure factors; geometry uses comoving positions and support.
 * Motion limits are comoving support divided by measured relative speed,
 * in the simulation velocity convention. ActiveParticle must be an explicit
 * list of indices into P, including on ranks with no sources. Arrays follow
 * that list and have NumActiveParticle entries. Zero exposure
 * initializes rates/support without scattering. Inactive receivers retain
 * their source clocks. Ownership stays fixed until this operation returns.
 * seed is the global base seed; the adapter derives the event stream.
 * The scheduler uses the individual returned limits. */
void sidm_distributed_force_intervals(const ActiveParticles *sources,Cosmology *CP,
    struct UnitSystem units,DriftKickTimes *times,const ForceTree *tree,
    const double *intervals,double *limits,double *motion_limits,uint64_t seed);
#endif
