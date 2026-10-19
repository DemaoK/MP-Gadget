#ifndef SIDM_LOCAL_H
#define SIDM_LOCAL_H

#include "timestep.h"

#ifdef SIDM
static inline inttime_t sidm_source_deadline(const struct particle_data *p)
{
    return p->TiSIDMLast + dti_from_timebin(p->TimeBinSIDM);
}
#endif

/* Separate cosmological SIDM source clocks. Only synchronized PM checkpoints
 * may be restarted: every source interval is flushed before writing them. */
void sidm_local_configure(int enabled, double transport_fraction, double max_dloga);
int sidm_local_enabled(void);
void sidm_local_initialize(const DriftKickTimes *times, Cosmology *CP);
inttime_t sidm_local_next_event(const DriftKickTimes *times, int first);
/* Recounted by native activity construction after ownership changes. */
void sidm_local_set_bin_counts(const int64_t counts[TIMEBINS+1]);
void sidm_local_assert_flushed(const DriftKickTimes *times);
void sidm_local_kick_phase(DriftKickTimes *times, int opening);

struct sidm_local_visit {
    ActiveParticles sources;
    /* All arrays follow source-list order. Exposure and rate limits are dt/a^4. */
    double *intervals, *limits;
    /* Support / sampled maximum canonical relative speed (drift integral). */
    double *motion_limits;
#ifdef SIDMDEBUG
    double *old_support; /* Comoving support at the previous source visit. */
#endif
};
/* Allocate event scratch, released by commit; do not migrate particles or
 * retain nested allocations across commit. The distributed solve fills limits.
 * The native active list must include SIDM-only sources; its live DM indices
 * must be strictly increasing. */
struct sidm_local_visit sidm_local_begin_active(const DriftKickTimes *times, Cosmology *CP,
    int flush, const ActiveParticles *active);
void sidm_local_commit(struct sidm_local_visit *visit, DriftKickTimes *times, Cosmology *CP);

#endif
