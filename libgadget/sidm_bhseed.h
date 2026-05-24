#ifndef SIDM_BHSEED_H
#define SIDM_BHSEED_H

#ifdef SIDM

#include "cosmology.h"
#include "density.h"
#include "domain.h"
#include "fof.h"
#include "timestep.h"
#include "utils/paramset.h"
#include "utils/system.h"

enum SIDMBHSeedTrigger {
    SIDM_BHSEED_TRIGGER_NONE = 0,
    SIDM_BHSEED_TRIGGER_RESOLVED = 1,
    SIDM_BHSEED_TRIGGER_UNRESOLVED_SOFTENING = 2,
    SIDM_BHSEED_TRIGGER_UNRESOLVED_STALL = 3,
};

struct SIDMBHSeedResult {
    int should_seed;
    int trigger;
    double seed_mass;
    double smfp_mass;
    double smfp_radius;
    double reservoir_mass;
    double rho_inf;
    double sound_speed_inf;
    double reservoir_radius;
    double collapse_progress;
    double collapse_time;
    double clock_fof_mass;
    double previous_clock_fof_mass;
    double merger_mass_jump;
    double merger_gamma;
    double knudsen;
    double halo_vmax;
    double halo_vmax_internal;
    double halo_rmax;
    double nfw_scale_radius;
    double nfw_scale_density;
    int num_dm;
    int vmax_profile_bins;
    int major_merger;
};

void set_sidm_bhseed_params(ParameterSet * ps);
int sidm_bhseed_is_enabled(void);
int sidm_bhseed_dynmass_catchup_on(void);
double sidm_bhseed_min_fof_mass(void);
double sidm_bhseed_dark_bondi_lambda(void);

/* First-pass seed diagnostics. The collapse clock uses the FoF-measured Vmax
 * and Rmax to infer equivalent NFW rs and rho_s. The SMFP reservoir is measured
 * from the FoF-reduced cumulative Knudsen profile around the central candidate. */
struct SIDMBHSeedResult sidm_bhseed_evaluate_candidate(
    int index,
    const struct Group * group,
    double atime,
    Cosmology * CP,
    const struct UnitSystem units);

void sidm_bhseed_swallow_dm(int * ActiveBlackHoles, int64_t NumActiveBlackHoles,
    DomainDecomp * ddecomp, double atime, Cosmology * CP, const DriftKickTimes * times,
    RandTable * rnd);
void sidm_bhseed_update_dm_only(const ActiveParticles * act, DomainDecomp * ddecomp,
    double atime, Cosmology * CP, const DriftKickTimes * times, RandTable * rnd);

#endif
#endif
