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
    double knudsen;
    double nfw_scale_radius;
    double nfw_scale_density;
    int num_dm;
    int nfw_fit_used;
    int nfw_fit_bins;
};

void set_sidm_bhseed_params(ParameterSet * ps);
int sidm_bhseed_is_enabled(void);
int sidm_bhseed_dynmass_catchup_on(void);
double sidm_bhseed_min_fof_mass(void);
double sidm_bhseed_dark_bondi_lambda(void);

/* First-pass seed diagnostics. The collapse clock uses the FoF outer NFW fit
 * when available, falling back to SIDMBHDefaultConcentration only if the fit
 * fails. The SMFP reservoir is still measured with a local DM aperture around
 * the central candidate. */
struct SIDMBHSeedResult sidm_bhseed_evaluate_candidate(
    int index,
    const struct Group * group,
    double atime,
    Cosmology * CP,
    const struct UnitSystem units,
    const struct kick_factor_data * kf);

void sidm_bhseed_swallow_dm(int * ActiveBlackHoles, int64_t NumActiveBlackHoles,
    DomainDecomp * ddecomp, double atime, Cosmology * CP, const DriftKickTimes * times,
    RandTable * rnd);

#endif
#endif
