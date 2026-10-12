#ifndef SIDM_LOCAL_H
#define SIDM_LOCAL_H

#include "timestep.h"

/* Separate cosmological SIDM source clocks. Only synchronized PM checkpoints
 * may be restarted: every source interval is flushed before writing them. */
void sidm_local_configure(int enabled, double transport_fraction,
    double max_support_change, double max_dloga);
int sidm_local_enabled(void);
void sidm_local_initialize(const DriftKickTimes *times, Cosmology *CP);
inttime_t sidm_local_next_event(const DriftKickTimes *times, Cosmology *CP, int first);
/* Collective: call after opening kicks, before selecting the next event. */
void sidm_local_update_speed_bound(void);
void sidm_local_assert_flushed(const DriftKickTimes *times);
void sidm_local_kick_phase(DriftKickTimes *times, int opening);

struct sidm_local_visit {
    ActiveParticles sources;
    /* Collision exposures and forward bounds are integrals dt/a^4. */
    double *intervals, *limits;
    double *old_support; /* Comoving support at the previous source visit. */
};
/* Borrow reusable storage until sidm_local_end. Only source entries of the
 * interval/limit/support arrays are initialized. The native active list must
 * include SIDM-only sources; its live DM indices must be strictly increasing. */
struct sidm_local_visit sidm_local_begin_active(const DriftKickTimes *times, Cosmology *CP,
    int flush, const ActiveParticles *active);
void sidm_local_commit(struct sidm_local_visit *visit, DriftKickTimes *times, Cosmology *CP);
void sidm_local_end(struct sidm_local_visit *visit);
/* Maximum drift factor (integral dt/a^2); non-DM returns DBL_MAX. */
double sidm_local_gravity_limit(int i, const DriftKickTimes *times);
#ifdef SIDMDEBUG
/* Aggregate diagnostic counts outside the OpenMP timestep loop. */
void sidm_local_record_gravity_limits(uint64_t checked, uint64_t shortened);
#endif

#endif
