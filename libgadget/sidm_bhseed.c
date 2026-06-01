#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "sidm_bhseed.h"

#ifdef SIDM

#include "density.h"
#include "gravity.h"
#include "partmanager.h"
#include "physconst.h"
#include "sidm.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include "walltime.h"

static struct SIDMBHSeedParams {
    int SeedOn;
    int DynMassCatchupOn;
    double CollapseThreshold;
    double CollapseCoeff;
    double MergerAlpha;
    double MajorMergerMassJump;
    double ReservoirKnudsenThreshold;
    int MinReservoirParticles;
    double SeedSMFPFraction;
    double SeedMassMin;
    double SeedMassMax;
    double DarkBondiLambda;
    double MinFoFMass;
} sidm_bhseed_params;

void
set_sidm_bhseed_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        sidm_bhseed_params.SeedOn = param_get_int(ps, "SIDMBHSeedOn");
        sidm_bhseed_params.DynMassCatchupOn = param_get_int(ps, "SIDMBHDynMassCatchupOn");
        sidm_bhseed_params.CollapseThreshold = param_get_double(ps, "SIDMBHCollapseThreshold");
        sidm_bhseed_params.CollapseCoeff = param_get_double(ps, "SIDMBHCollapseCoeff");
        sidm_bhseed_params.MergerAlpha = param_get_double(ps, "SIDMBHMergerAlpha");
        sidm_bhseed_params.MajorMergerMassJump = param_get_double(ps, "SIDMBHMajorMergerMassJump");
        sidm_bhseed_params.ReservoirKnudsenThreshold = param_get_double(ps, "SIDMBHReservoirKnudsenThreshold");
        sidm_bhseed_params.MinReservoirParticles = param_get_int(ps, "SIDMBHMinReservoirParticles");
        sidm_bhseed_params.SeedSMFPFraction = param_get_double(ps, "SIDMBHSeedSMFPFraction");
        sidm_bhseed_params.SeedMassMin = param_get_double(ps, "SIDMBHSeedMassMin");
        sidm_bhseed_params.SeedMassMax = param_get_double(ps, "SIDMBHSeedMassMax");
        sidm_bhseed_params.DarkBondiLambda = param_get_double(ps, "SIDMBHDarkBondiLambda");
        sidm_bhseed_params.MinFoFMass = param_get_double(ps, "SIDMBHMinFoFMass");

        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.SeedSMFPFraction <= 0)
            endrun(1, "SIDMBHSeedOn requires SIDMBHSeedSMFPFraction > 0.\n");
        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.MinFoFMass < 0)
            endrun(1, "SIDMBHMinFoFMass must be non-negative.\n");
        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.MergerAlpha < 0)
            endrun(1, "SIDMBHMergerAlpha must be non-negative.\n");
        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.MajorMergerMassJump < 0)
            endrun(1, "SIDMBHMajorMergerMassJump must be non-negative.\n");
    }
    MPI_Bcast(&sidm_bhseed_params, sizeof(struct SIDMBHSeedParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int
sidm_bhseed_is_enabled(void)
{
    return sidm_bhseed_params.SeedOn;
}

int
sidm_bhseed_dynmass_catchup_on(void)
{
    return sidm_bhseed_params.DynMassCatchupOn;
}

double
sidm_bhseed_min_fof_mass(void)
{
    return sidm_bhseed_params.MinFoFMass;
}

double
sidm_bhseed_dark_bondi_lambda(void)
{
    return sidm_bhseed_params.DarkBondiLambda;
}

static double
sidm_bhseed_clamp(double x, double xmin, double xmax)
{
    if(x < xmin)
        x = xmin;
    if(xmax > 0 && x > xmax)
        x = xmax;
    return x;
}

static int
sidm_bhseed_clock_is_better(double mass, double progress, double last_check,
    double ref_mass, double ref_progress, double ref_last_check)
{
    if(mass > ref_mass)
        return 1;
    if(mass < ref_mass)
        return 0;
    if(progress > ref_progress)
        return 1;
    if(progress < ref_progress)
        return 0;
    return last_check > ref_last_check;
}

/* Halo-scale collapse time from FoF-measured Vmax and Rmax. FoF stores
 * SIDMVmax as sqrt(G M(<Rmax) / Rmax) with the usual MP-Gadget Msun/h and kpc/h
 * code units. The h factors cancel in G M/R. Since Rmax is comoving,
 * SIDMVmax = sqrt(a) Vmax_phys, and the velocity compared to wTurn is the
 * internal canonical velocity a Vmax_phys = sqrt(a) SIDMVmax. */
static double
sidm_bhseed_estimate_tc_from_group(const struct Group * group, double atime,
    Cosmology * CP, const struct UnitSystem units, struct SIDMBHSeedResult * result)
{
    if(CP == NULL || atime <= 0 || CP->GravInternal <= 0)
        return 0;

    const double rmax = group->SIDMRmax;
    const double vmax = group->SIDMVmax;
    result->halo_vmax = vmax;
    result->halo_rmax = rmax;
    result->vmax_profile_bins = group->SIDMVmaxProfileBins;
    if(rmax <= 0 || vmax <= 0)
        return 0;

    const double vmax_internal = sqrt(atime) * vmax;
    result->halo_vmax_internal = vmax_internal;

    const double rs = rmax / SIDM_BHSEED_NFW_XMAX;
    const double rho_s = vmax * vmax /
        (SIDM_BHSEED_NFW_VMAX_COEFF * SIDM_BHSEED_NFW_VMAX_COEFF *
         CP->GravInternal * rs * rs);
    result->nfw_scale_radius = rs;
    result->nfw_scale_density = rho_s;

    const double sigma1d_eff = 1.1 * vmax_internal / sqrt(3.0);
    const double sigma_code = sidm_sigma_kappa_over_m_code(sigma1d_eff, atime, CP, units);
    const double dyn = sqrt(4.0 * M_PI * CP->GravInternal * rho_s /
        (atime * atime * atime));
    if(sigma_code <= 0 || rho_s <= 0 || dyn <= 0)
        return 0;

    return sidm_bhseed_params.CollapseCoeff / (sigma_code * rho_s * rs) / dyn;
}

static void
sidm_bhseed_set_reservoir_state(struct SIDMBHSeedResult * result, double radius,
    double mass, double rho, double sigma1d, double kn, int numdm, int is_smfp)
{
    result->reservoir_radius = radius;
    result->rho_inf = rho;
    result->sound_speed_inf = sigma1d;
    result->knudsen = kn;
    result->num_dm = numdm;
    if(is_smfp) {
        result->smfp_radius = radius;
        result->smfp_mass = mass;
    }
}

static void
sidm_bhseed_smfp_diagnostic_from_group(const struct Group * group, double atime, Cosmology * CP,
    const struct UnitSystem units,
    struct SIDMBHSeedResult * result)
{
    const double search_radius = group->SIDMSMFPProfileRMax;
    if(search_radius <= 0 || group->SIDMSMFPProfileBins <= 0)
        return;

    double mass = 0;
    double momentum[3] = {0, 0, 0};
    double mv2 = 0;
    double count = 0;
    double best_kn = 1e30;

    for(int i = 0; i < SIDM_SMFP_PROFILE_BINS; i++) {
        mass += group->SIDMSMFPProfileMass[i];
        mv2 += group->SIDMSMFPProfileMV2[i];
        count += group->SIDMSMFPProfileCount[i];
        for(int k = 0; k < 3; k++)
            momentum[k] += group->SIDMSMFPProfileMomentum[i][k];

        const int numdm = (int)(count + 0.5);
        const double radius = search_radius * (i + 1.0) / SIDM_SMFP_PROFILE_BINS;
        if(numdm < sidm_bhseed_params.MinReservoirParticles || mass <= 0 || radius <= 0)
            continue;

        const double volume = 4.0 * M_PI / 3.0 * radius * radius * radius;
        const double rho = mass / volume;
        double v2 = mv2;
        for(int k = 0; k < 3; k++)
            v2 -= momentum[k] * momentum[k] / mass;
        const double sigma1d2 = DMAX(v2 / (3.0 * mass), 0.0);
        const double sigma1d = sqrt(sigma1d2);
        const double sigma_code = sidm_sigma_kappa_over_m_code(sigma1d, atime, CP, units);
        const double hscale = sigma1d > 0 && rho > 0 && atime > 0 ?
            sigma1d / sqrt(4.0 * M_PI * CP->GravInternal * rho * atime) : 0;
        const double mfp = rho > 0 && sigma_code > 0 ? 1.0 / (rho * sigma_code) : 0;
        const double kn = hscale > 0 ? mfp / hscale : 1e30;
        const int is_smfp = kn < sidm_bhseed_params.ReservoirKnudsenThreshold;

        if(is_smfp || kn < best_kn) {
            sidm_bhseed_set_reservoir_state(result, radius, mass, rho, sigma1d, kn, numdm, is_smfp);
            best_kn = kn;
        }
    }
}

struct SIDMBHSeedResult
sidm_bhseed_evaluate_candidate(int index, const struct Group * group, double atime,
    Cosmology * CP, const struct UnitSystem units)
{
    struct SIDMBHSeedResult result;
    memset(&result, 0, sizeof(result));
    result.trigger = SIDM_BHSEED_TRIGGER_NONE;
    result.knudsen = 1e30;

    if(!sidm_bhseed_params.SeedOn || P[index].Type != 1)
        return result;
    if(group->Mass < sidm_bhseed_params.MinFoFMass || group->LenType[5] > 0)
        return result;

    const double tc = sidm_bhseed_estimate_tc_from_group(group, atime, CP, units, &result);
    result.collapse_time = tc;

    if(sidm_bhseed_clock_is_better(group->SIDMBHClockFoFMass, group->SIDMBHCollapseProgress,
        group->SIDMBHLastCheckTime, P[index].SIDMBHClockFoFMass,
        P[index].SIDMBHCollapseProgress, P[index].SIDMBHLastCheckTime)) {
        message(0, "SIDM BH clock inherited: candidate ID=%llu inherited_from=%llu progress=%g last_a=%g Mclock=%g previous_progress=%g previous_last_a=%g previous_Mclock=%g\n",
            (unsigned long long) P[index].ID,
            (unsigned long long) group->SIDMBHClockID,
            group->SIDMBHCollapseProgress,
            group->SIDMBHLastCheckTime,
            group->SIDMBHClockFoFMass,
            P[index].SIDMBHCollapseProgress,
            P[index].SIDMBHLastCheckTime,
            P[index].SIDMBHClockFoFMass);
        P[index].SIDMBHCollapseProgress = group->SIDMBHCollapseProgress;
        P[index].SIDMBHLastCheckTime = group->SIDMBHLastCheckTime;
        P[index].SIDMBHClockFoFMass = group->SIDMBHClockFoFMass;
    }

    const double old_progress = P[index].SIDMBHCollapseProgress;
    const double old_clock_mass = P[index].SIDMBHClockFoFMass;
    result.previous_clock_fof_mass = old_clock_mass;
    if(P[index].SIDMBHLastCheckTime > 0 && atime > P[index].SIDMBHLastCheckTime && tc > 0) {
        const double hubble = hubble_function(CP, atime);
        const double dt = log(atime / P[index].SIDMBHLastCheckTime) / hubble;
        double dprogress = dt / tc;
        if(old_clock_mass > 0 && group->Mass > old_clock_mass && dt > 0) {
            result.merger_mass_jump = group->Mass / old_clock_mass - 1.0;
            if(result.merger_mass_jump > sidm_bhseed_params.MajorMergerMassJump) {
                result.major_merger = 1;
                result.merger_gamma = (group->Mass - old_clock_mass) / (dt * old_clock_mass);
                dprogress = (1.0 / tc - sidm_bhseed_params.MergerAlpha *
                    result.merger_gamma * old_progress) * dt;
                message(0, "SIDM BH clock major merger: candidate ID=%llu old_progress=%g dprogress=%g old_Mclock=%g new_Mfof=%g jump=%g gamma=%g alpha=%g tc=%g dt=%g\n",
                    (unsigned long long) P[index].ID, old_progress, dprogress,
                    old_clock_mass, group->Mass, result.merger_mass_jump,
                    result.merger_gamma, sidm_bhseed_params.MergerAlpha, tc, dt);
            }
        }
        P[index].SIDMBHCollapseProgress = DMAX(P[index].SIDMBHCollapseProgress + dprogress, 0.0);
    }
    P[index].SIDMBHLastCheckTime = atime;
    P[index].SIDMBHClockFoFMass = group->Mass;
    result.collapse_progress = P[index].SIDMBHCollapseProgress;
    result.clock_fof_mass = P[index].SIDMBHClockFoFMass;

    sidm_bhseed_smfp_diagnostic_from_group(group, atime, CP, units, &result);

    if(result.collapse_progress < sidm_bhseed_params.CollapseThreshold)
        return result;
    if(result.smfp_mass <= 0)
        return result;

    double seed_mass = sidm_bhseed_params.SeedSMFPFraction * result.smfp_mass;
    seed_mass = sidm_bhseed_clamp(seed_mass, sidm_bhseed_params.SeedMassMin, sidm_bhseed_params.SeedMassMax);
    if(seed_mass <= 0)
        return result;

    result.seed_mass = seed_mass;
    result.reservoir_mass = DMAX(result.smfp_mass - seed_mass, 0.0);
    result.should_seed = 1;
    result.trigger = SIDM_BHSEED_TRIGGER_RESOLVED;

    message(0, "SIDM BH seed candidate ID %llu: progress=%g threshold=%g tc=%g Mclock=%g prev_Mclock=%g major_merger=%d jump=%g gamma=%g VmaxFoF=%g VmaxInternal=%g RmaxComoving=%g VmaxBins=%d rsComoving=%g rhosComoving=%g Msmfp=%g Rsmfp=%g Kn=%g Ndm=%d Mseed=%g Mres=%g rho_inf_comoving=%g sigma1d_internal=%g\n",
        (unsigned long long) P[index].ID, result.collapse_progress, sidm_bhseed_params.CollapseThreshold,
        result.collapse_time, result.clock_fof_mass, result.previous_clock_fof_mass,
        result.major_merger, result.merger_mass_jump, result.merger_gamma,
        result.halo_vmax, result.halo_vmax_internal, result.halo_rmax, result.vmax_profile_bins,
        result.nfw_scale_radius, result.nfw_scale_density,
        result.smfp_mass, result.smfp_radius, result.knudsen,
        result.num_dm, result.seed_mass, result.reservoir_mass,
        result.rho_inf, result.sound_speed_inf);

    return result;
}

struct SIDMBHDMSwallowPriv {
    struct kick_factor_data * kf;
    RandTable * rnd;
    int64_t * N_dm_swallowed;
    MyIDType * DM_SwallowID;
    MyFloat * AccretedMass;
    MyFloat (*AccretedMomentum)[3];
};

#define SIDM_DM_SWALLOW_GET_PRIV(tw) ((struct SIDMBHDMSwallowPriv *) ((tw)->priv))

static int
sidm_bhseed_origin_bh_haswork(int n, TreeWalk * tw)
{
    (void) tw;
    return P[n].Type == 5 && !P[n].Swallowed && BHP(n).SIDMSeedOrigin;
}

static int64_t
sidm_bhseed_active_origin_bhs(const ActiveParticles * act, int ** ActiveBlackHoles,
    int64_t * NumActiveBlackHoles)
{
    TreeWalk tw_bh[1] = {{0}};
    tw_bh->haswork = sidm_bhseed_origin_bh_haswork;

    treewalk_build_queue(tw_bh, act->ActiveParticle, act->NumActiveParticle, 0);

    int64_t totbh;
    MPI_Allreduce(&tw_bh->WorkSetSize, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    *NumActiveBlackHoles = tw_bh->WorkSetSize;
    if(totbh > 0) {
        const size_t nalloc = tw_bh->WorkSetSize > 0 ? tw_bh->WorkSetSize : 1;
        *ActiveBlackHoles = (int *) mymalloc2("SIDMActiveOriginBH",
            nalloc * sizeof(int));
        if(tw_bh->WorkSetSize > 0)
            memcpy(*ActiveBlackHoles, tw_bh->WorkSet, tw_bh->WorkSetSize * sizeof(int));
    }
    myfree(tw_bh->WorkSet);
    return totbh;
}

static void
sidm_bhseed_apply_dark_reservoir_accretion(int * ActiveBlackHoles,
    int64_t NumActiveBlackHoles, double atime, Cosmology * CP, const DriftKickTimes * times)
{
    (void) times;
    if(atime <= 0)
        return;

    const double hubble = hubble_function(CP, atime);
    const double a3inv = 1.0 / (atime * atime * atime);
    double local_dark_mass = 0;
    int64_t local_updated = 0;

    for(int64_t i = 0; i < NumActiveBlackHoles; i++) {
        const int n = ActiveBlackHoles[i];
        BHP(n).SIDMDarkMdot = 0;
        if(!BHP(n).SIDMSeedOrigin || BHP(n).SIDMDarkReservoirMass <= 0 ||
           BHP(n).SIDMRhoInf <= 0 || BHP(n).SIDMSoundSpeedInf <= 0)
            continue;

        int timebin = P[n].TimeBinHydro > 0 ? P[n].TimeBinHydro : P[n].TimeBinGravity;
        if(timebin <= 0 || hubble <= 0)
            continue;
        const double dtime = get_dloga_for_bin(timebin, P[n].Ti_drift) / hubble;
        if(dtime <= 0)
            continue;

        const double rho_dark_proper = BHP(n).SIDMRhoInf * a3inv;
        const double sound_dark_proper = BHP(n).SIDMSoundSpeedInf / atime;
        const double ainf3 = sound_dark_proper * sound_dark_proper * sound_dark_proper;
        if(ainf3 <= 0)
            continue;

        double mdot_dark = 4.0 * M_PI * sidm_bhseed_dark_bondi_lambda() *
            CP->GravInternal * CP->GravInternal * BHP(n).Mass * BHP(n).Mass *
            rho_dark_proper / ainf3;
        double delta_dark = mdot_dark * dtime;
        if(delta_dark > BHP(n).SIDMDarkReservoirMass)
            delta_dark = BHP(n).SIDMDarkReservoirMass;
        if(delta_dark < 0)
            delta_dark = 0;
        if(delta_dark <= 0)
            continue;

        BHP(n).SIDMDarkMdot = delta_dark / dtime;
        BHP(n).SIDMDarkReservoirMass -= delta_dark;
        if(sidm_bhseed_dynmass_catchup_on()) {
            const double buffer = DMAX(P[n].Mass - BHP(n).Mass, 0);
            const double used = DMIN(buffer, delta_dark);
            BHP(n).SIDMDMDynMassDebt += delta_dark - used;
        }
        BHP(n).Mass += delta_dark;
        local_dark_mass += delta_dark;
        local_updated++;
    }

    double total_dark_mass = local_dark_mass;
    int64_t total_updated = local_updated;
    MPI_Allreduce(MPI_IN_PLACE, &total_dark_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_updated, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(total_updated > 0)
        message(0, "SIDM BH dark-only reservoir accreted %g onto %lld BHs.\n",
            total_dark_mass, (long long) total_updated);
}

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat RhoInf;
    MyFloat DMDynMassDebt;
    MyIDType ID;
} TreeWalkQuerySIDMDMSwallow;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass;
    MyFloat AccretedMomentum[3];
} TreeWalkResultSIDMDMSwallow;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterSIDMDMSwallow;

static int
sidm_bhseed_dm_swallow_haswork(int n, TreeWalk * tw)
{
    return P[n].Type == 5 && !P[n].Swallowed &&
        BHP(n).SIDMSeedOrigin && BHP(n).SIDMDMDynMassDebt > 0 &&
        BHP(n).SIDMRhoInf > 0 &&
        (BHP(n).SIDMReservoirRadius > 0 || P[n].Hsml > 0);
}

static void
sidm_bhseed_dm_swallow_copy(int place, TreeWalkQuerySIDMDMSwallow * I, TreeWalk * tw)
{
    I->Hsml = BHP(place).SIDMReservoirRadius > 0 ? BHP(place).SIDMReservoirRadius : P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->RhoInf = BHP(place).SIDMRhoInf;
    I->DMDynMassDebt = BHP(place).SIDMDMDynMassDebt;
    I->ID = P[place].ID;
    for(int k = 0; k < 3; k++)
        I->Vel[k] = P[place].Vel[k];
}

static void
sidm_bhseed_dm_swallow_ngbiter(TreeWalkQuerySIDMDMSwallow * I,
        TreeWalkResultSIDMDMSwallow * O,
        TreeWalkNgbIterSIDMDMSwallow * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.mask = DMMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    if(P[other].Type != 1 || P[other].IsGarbage || P[other].Swallowed)
        return;
    if(P[other].ID == I->ID)
        return;
    if(iter->base.r2 >= iter->kernel.HH || I->RhoInf <= 0)
        return;

    const double missing = DMIN(I->DMDynMassDebt, DMAX(I->BH_Mass - I->Mass, 0));
    if(missing <= 0)
        return;

    const double u = iter->base.r * iter->kernel.Hinv;
    const double wk = density_kernel_wk(&iter->kernel, u);
    double p = missing * wk / I->RhoInf;
    if(p > 1)
        p = 1;

    const double w = get_random_number(P[other].ID + I->ID + 9191, SIDM_DM_SWALLOW_GET_PRIV(lv->tw)->rnd);
    if(w >= p)
        return;

    MyIDType readid;
    MyIDType newswallowid = I->ID + 1;
    MyIDType * swal = SIDM_DM_SWALLOW_GET_PRIV(lv->tw)->DM_SwallowID + P[other].PI;
#pragma omp atomic read
    readid = *swal;
    do {
        if(readid != 0)
            return;
    } while(!__atomic_compare_exchange_n(swal, &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    O->Mass += P[other].Mass;
    MyFloat VelPred[3];
    DM_VelPred(other, VelPred, SIDM_DM_SWALLOW_GET_PRIV(lv->tw)->kf);
    for(int k = 0; k < 3; k++)
        O->AccretedMomentum[k] += P[other].Mass * VelPred[k];
    slots_mark_garbage(other, PartManager, SlotsManager);

    int tid = omp_get_thread_num();
    SIDM_DM_SWALLOW_GET_PRIV(lv->tw)->N_dm_swallowed[tid]++;
}

static void
sidm_bhseed_dm_swallow_reduce(int place, TreeWalkResultSIDMDMSwallow * remote,
        enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int PI = P[place].PI;
    (void) mode;
    if(remote->Mass <= 0)
        return;
    TREEWALK_REDUCE(SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMass[PI], remote->Mass);
    for(int k = 0; k < 3; k++) {
        TREEWALK_REDUCE(SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMomentum[PI][k], remote->AccretedMomentum[k]);
    }
}

static void
sidm_bhseed_dm_swallow_postprocess(int n, TreeWalk * tw)
{
    const int PI = P[n].PI;
    const double accmass = SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMass[PI];
    if(accmass <= 0)
        return;

    for(int k = 0; k < 3; k++)
        P[n].Vel[k] = (P[n].Vel[k] * P[n].Mass + SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMomentum[PI][k]) / (P[n].Mass + accmass);
    P[n].Mass += accmass;
    if(BHP(n).SIDMDMDynMassDebt > accmass)
        BHP(n).SIDMDMDynMassDebt -= accmass;
    else
        BHP(n).SIDMDMDynMassDebt = 0;

    message(0, "SIDM BH DM catch-up: ID=%llu accreted_dyn_mass=%g new_PMass=%g remaining_dm_debt=%g\n",
        (unsigned long long) P[n].ID, accmass, P[n].Mass, BHP(n).SIDMDMDynMassDebt);
    for(int k = 0; k < 3; k++)
        SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMomentum[PI][k] = 0;
    SIDM_DM_SWALLOW_GET_PRIV(tw)->AccretedMass[PI] = 0;
}

void
sidm_bhseed_swallow_dm(int * ActiveBlackHoles, int64_t NumActiveBlackHoles,
    DomainDecomp * ddecomp, double atime, Cosmology * CP, const DriftKickTimes * times,
    RandTable * rnd)
{
    (void) atime;
    if(!sidm_bhseed_params.SeedOn || !sidm_bhseed_params.DynMassCatchupOn)
        return;

    int64_t LocalCatchupBlackHoles = 0;
    for(int64_t i = 0; i < NumActiveBlackHoles; i++) {
        if(sidm_bhseed_dm_swallow_haswork(ActiveBlackHoles[i], NULL))
            LocalCatchupBlackHoles++;
    }
    int64_t TotCatchupBlackHoles = LocalCatchupBlackHoles;
    MPI_Allreduce(MPI_IN_PLACE, &TotCatchupBlackHoles, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(TotCatchupBlackHoles <= 0)
        return;

    message(0, "SIDM BH DM catch-up treewalk for %lld BHs with dynamical-mass debt.\n",
        (long long) TotCatchupBlackHoles);

    struct kick_factor_data kf;
    init_kick_factor_data(&kf, times, CP);

    ForceTree dmtree = {0};
    force_tree_rebuild_mask(&dmtree, ddecomp, DMMASK, NULL);

    struct SIDMBHDMSwallowPriv priv[1] = {{0}};
    priv->kf = &kf;
    priv->rnd = rnd;
    priv->N_dm_swallowed = ta_malloc("sidm_dm_swallowed", int64_t, omp_get_max_threads());
    memset(priv->N_dm_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());
    priv->DM_SwallowID = mymalloc("SIDMDMSwallowID", SlotsManager->info[1].size * sizeof(MyIDType));
    priv->AccretedMass = mymalloc("SIDMDMAccretedMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->AccretedMomentum = mymalloc("SIDMDMAccretedMomentum", SlotsManager->info[5].size * sizeof(priv->AccretedMomentum[0]));
    memset(priv->DM_SwallowID, 0, SlotsManager->info[1].size * sizeof(MyIDType));
    memset(priv->AccretedMass, 0, SlotsManager->info[5].size * sizeof(MyFloat));
    memset(priv->AccretedMomentum, 0, SlotsManager->info[5].size * sizeof(priv->AccretedMomentum[0]));

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "SIDM_BH_DM_SWALLOW";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSIDMDMSwallow);
    tw->ngbiter = (TreeWalkNgbIterFunction) sidm_bhseed_dm_swallow_ngbiter;
    tw->haswork = sidm_bhseed_dm_swallow_haswork;
    tw->postprocess = (TreeWalkProcessFunction) sidm_bhseed_dm_swallow_postprocess;
    tw->preprocess = NULL;
    tw->fill = (TreeWalkFillQueryFunction) sidm_bhseed_dm_swallow_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sidm_bhseed_dm_swallow_reduce;
    tw->query_type_elsize = sizeof(TreeWalkQuerySIDMDMSwallow);
    tw->result_type_elsize = sizeof(TreeWalkResultSIDMDMSwallow);
    tw->tree = &dmtree;
    tw->priv = priv;

    treewalk_run(tw, ActiveBlackHoles, NumActiveBlackHoles);

    int64_t nlocal = 0;
    for(int i = 0; i < omp_get_max_threads(); i++)
        nlocal += priv->N_dm_swallowed[i];
    int64_t ntot = nlocal;
    MPI_Reduce(&nlocal, &ntot, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "SIDM BH DM catch-up swallowed %lld DM particles.\n", (long long) ntot);

    myfree(priv->AccretedMomentum);
    myfree(priv->AccretedMass);
    myfree(priv->DM_SwallowID);
    ta_free(priv->N_dm_swallowed);
    force_tree_free(&dmtree);
    walltime_measure("/SIDM/BHDMSwallow");
}

void
sidm_bhseed_update_dm_only(const ActiveParticles * act, DomainDecomp * ddecomp,
    double atime, Cosmology * CP, const DriftKickTimes * times, RandTable * rnd)
{
    int * ActiveBlackHoles = NULL;
    int64_t NumActiveBlackHoles = 0;
    const int64_t TotActiveBlackHoles = sidm_bhseed_active_origin_bhs(act,
        &ActiveBlackHoles, &NumActiveBlackHoles);
    if(TotActiveBlackHoles <= 0)
        return;

    sidm_bhseed_apply_dark_reservoir_accretion(ActiveBlackHoles,
        NumActiveBlackHoles, atime, CP, times);
    sidm_bhseed_swallow_dm(ActiveBlackHoles, NumActiveBlackHoles, ddecomp,
        atime, CP, times, rnd);

    if(ActiveBlackHoles)
        myfree(ActiveBlackHoles);
}

#endif
