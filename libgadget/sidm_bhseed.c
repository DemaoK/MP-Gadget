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
    double DefaultConcentration;
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
        sidm_bhseed_params.DefaultConcentration = param_get_double(ps, "SIDMBHDefaultConcentration");
        sidm_bhseed_params.ReservoirKnudsenThreshold = param_get_double(ps, "SIDMBHReservoirKnudsenThreshold");
        sidm_bhseed_params.MinReservoirParticles = param_get_int(ps, "SIDMBHMinReservoirParticles");
        sidm_bhseed_params.SeedSMFPFraction = param_get_double(ps, "SIDMBHSeedSMFPFraction");
        sidm_bhseed_params.SeedMassMin = param_get_double(ps, "SIDMBHSeedMassMin");
        sidm_bhseed_params.SeedMassMax = param_get_double(ps, "SIDMBHSeedMassMax");
        sidm_bhseed_params.DarkBondiLambda = param_get_double(ps, "SIDMBHDarkBondiLambda");
        sidm_bhseed_params.MinFoFMass = param_get_double(ps, "SIDMBHMinFoFMass");

        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.SeedSMFPFraction <= 0)
            endrun(1, "SIDMBHSeedOn requires SIDMBHSeedSMFPFraction > 0.\n");
        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.DefaultConcentration <= 1)
            endrun(1, "SIDMBHDefaultConcentration must be > 1 for the fallback NFW clock.\n");
        if(sidm_bhseed_params.SeedOn && sidm_bhseed_params.MinFoFMass < 0)
            endrun(1, "SIDMBHMinFoFMass must be non-negative.\n");
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

/* Halo-scale collapse time. Prefer the outer NFW fit measured during FoF; the
 * default concentration is only a fallback for poorly resolved/failed fits. All
 * quantities are in MP-Gadget internal comoving units, matching the SIDM
 * scatter module's sigma/m conversion. */
static double
sidm_bhseed_estimate_tc_from_group(const struct Group * group, double atime,
    Cosmology * CP, const struct UnitSystem units, struct SIDMBHSeedResult * result)
{
    const double mhalo = group->MassType[1] > 0 ? group->MassType[1] : group->Mass;
    if(mhalo <= 0)
        return 0;

    const double rho_crit0 = 3.0 * CP->Hubble * CP->Hubble / (8.0 * M_PI * CP->GravInternal);
    const double rho_ref_comoving = 200.0 * CP->Omega0 * rho_crit0;
    if(rho_ref_comoving <= 0)
        return 0;

    const double r200 = cbrt(3.0 * mhalo / (4.0 * M_PI * rho_ref_comoving));
    double rs = group->SIDMNFWScaleRadius;
    double rho_s = group->SIDMNFWScaleDensity;
    result->nfw_fit_bins = group->SIDMNFWFitBins;

    if(rs > 0 && rho_s > 0) {
        result->nfw_fit_used = 1;
    } else {
        const double c = sidm_bhseed_params.DefaultConcentration;
        rs = r200 / c;
        const double fc = log(1.0 + c) - c / (1.0 + c);
        if(rs <= 0 || fc <= 0)
            return 0;
        rho_s = mhalo / (4.0 * M_PI * rs * rs * rs * fc);
        result->nfw_fit_used = 0;
    }

    result->nfw_scale_radius = rs;
    result->nfw_scale_density = rho_s;
    const double v2002 = CP->GravInternal * mhalo / r200;
    const double vrel2 = 6.0 * DMAX(v2002 / 2.0, 0.0);
    const double sigma_code = sidm_sigma_over_m_code(vrel2, atime, CP, units);
    const double dyn = sqrt(4.0 * M_PI * CP->GravInternal * rho_s);
    if(sigma_code <= 0 || rho_s <= 0 || dyn <= 0)
        return 0;

    return sidm_bhseed_params.CollapseCoeff / (sigma_code * rho_s * rs) / dyn;
}

struct SIDMBHReservoirSample {
    double r;
    double mass;
    double vel[3];
    double vel2;
};

static int
sidm_bhseed_compare_reservoir_radius(const void * a, const void * b)
{
    const struct SIDMBHReservoirSample * sa = (const struct SIDMBHReservoirSample *) a;
    const struct SIDMBHReservoirSample * sb = (const struct SIDMBHReservoirSample *) b;
    return (sa->r > sb->r) - (sa->r < sb->r);
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
sidm_bhseed_local_smfp_diagnostic(int index, double atime, Cosmology * CP,
    const struct UnitSystem units, const struct kick_factor_data * kf,
    struct SIDMBHSeedResult * result)
{
    const double soft = FORCE_SOFTENING();
    double search_radius = result->nfw_scale_radius;
    if(search_radius < soft)
        search_radius = soft;
    if(search_radius <= 0)
        return;

    int nsamples = 0;

    for(int i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 1 || P[i].IsGarbage || P[i].Swallowed)
            continue;
        double r2 = 0;
        for(int k = 0; k < 3; k++) {
            const double dx = NEAREST(P[i].Pos[k] - P[index].Pos[k], PartManager->BoxSize);
            r2 += dx * dx;
        }
        if(r2 <= search_radius * search_radius)
            nsamples++;
    }

    if(nsamples <= 0)
        return;

    struct SIDMBHReservoirSample * samples = (struct SIDMBHReservoirSample *)
        mymalloc("SIDMSMFPBoundarySamples", sizeof(samples[0]) * nsamples);

    int n = 0;
    for(int i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 1 || P[i].IsGarbage || P[i].Swallowed)
            continue;
        double r2 = 0;
        for(int k = 0; k < 3; k++) {
            const double dx = NEAREST(P[i].Pos[k] - P[index].Pos[k], PartManager->BoxSize);
            r2 += dx * dx;
        }
        if(r2 > search_radius * search_radius)
            continue;
        MyFloat VelPred[3];
        DM_VelPred(i, VelPred, kf);
        samples[n].r = sqrt(r2);
        samples[n].mass = P[i].Mass;
        samples[n].vel2 = 0;
        for(int k = 0; k < 3; k++) {
            samples[n].vel[k] = VelPred[k];
            samples[n].vel2 += VelPred[k] * VelPred[k];
        }
        n++;
    }
    nsamples = n;
    qsort(samples, nsamples, sizeof(samples[0]), sidm_bhseed_compare_reservoir_radius);

    double mass = 0;
    double momentum[3] = {0, 0, 0};
    double mv2 = 0;
    double best_kn = 1e30;

    for(int i = 0; i < nsamples; i++) {
        mass += samples[i].mass;
        mv2 += samples[i].mass * samples[i].vel2;
        for(int k = 0; k < 3; k++)
            momentum[k] += samples[i].mass * samples[i].vel[k];

        const int numdm = i + 1;
        if(numdm < sidm_bhseed_params.MinReservoirParticles || mass <= 0 || samples[i].r <= 0)
            continue;

        const double volume = 4.0 * M_PI / 3.0 * samples[i].r * samples[i].r * samples[i].r;
        const double rho = mass / volume;
        double v2 = mv2;
        for(int k = 0; k < 3; k++)
            v2 -= momentum[k] * momentum[k] / mass;
        const double sigma1d2 = DMAX(v2 / (3.0 * mass), 0.0);
        const double sigma1d = sqrt(sigma1d2);
        const double vrel2 = 6.0 * sigma1d2;
        const double sigma_code = sidm_sigma_over_m_code(vrel2, atime, CP, units);
        const double hscale = sigma1d > 0 && rho > 0 ? sigma1d / sqrt(4.0 * M_PI * CP->GravInternal * rho) : 0;
        const double mfp = rho > 0 && sigma_code > 0 ? 1.0 / (rho * sigma_code) : 0;
        const double kn = hscale > 0 ? mfp / hscale : 1e30;
        const int is_smfp = kn < sidm_bhseed_params.ReservoirKnudsenThreshold;

        if(is_smfp || kn < best_kn) {
            sidm_bhseed_set_reservoir_state(result, samples[i].r, mass, rho, sigma1d, kn, numdm, is_smfp);
            best_kn = kn;
        }
    }

    myfree(samples);
}

struct SIDMBHSeedResult
sidm_bhseed_evaluate_candidate(int index, const struct Group * group, double atime,
    Cosmology * CP, const struct UnitSystem units, const struct kick_factor_data * kf)
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

    if(group->SIDMBHCollapseProgress > P[index].SIDMBHCollapseProgress ||
       (group->SIDMBHCollapseProgress == P[index].SIDMBHCollapseProgress &&
        group->SIDMBHLastCheckTime > P[index].SIDMBHLastCheckTime)) {
        message(0, "SIDM BH clock inherited: candidate ID=%llu inherited_from=%llu progress=%g last_a=%g previous_progress=%g previous_last_a=%g\n",
            (unsigned long long) P[index].ID,
            (unsigned long long) group->SIDMBHClockID,
            group->SIDMBHCollapseProgress,
            group->SIDMBHLastCheckTime,
            P[index].SIDMBHCollapseProgress,
            P[index].SIDMBHLastCheckTime);
        P[index].SIDMBHCollapseProgress = group->SIDMBHCollapseProgress;
        P[index].SIDMBHLastCheckTime = group->SIDMBHLastCheckTime;
    }

    if(P[index].SIDMBHLastCheckTime > 0 && atime > P[index].SIDMBHLastCheckTime && tc > 0) {
        const double hubble = hubble_function(CP, atime);
        const double dt = log(atime / P[index].SIDMBHLastCheckTime) / hubble;
        P[index].SIDMBHCollapseProgress += dt / tc;
    }
    P[index].SIDMBHLastCheckTime = atime;
    result.collapse_progress = P[index].SIDMBHCollapseProgress;

    sidm_bhseed_local_smfp_diagnostic(index, atime, CP, units, kf, &result);

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

    message(0, "SIDM BH seed candidate ID %llu: progress=%g threshold=%g tc=%g NFWfit=%d NFWbins=%d rs=%g rhos=%g Msmfp=%g Rsmfp=%g Kn=%g Ndm=%d Mseed=%g Mres=%g rho_inf_comoving=%g cs_inf_comoving=%g\n",
        (unsigned long long) P[index].ID, result.collapse_progress, sidm_bhseed_params.CollapseThreshold,
        result.collapse_time, result.nfw_fit_used, result.nfw_fit_bins,
        result.nfw_scale_radius, result.nfw_scale_density,
        result.smfp_mass, result.smfp_radius, result.knudsen,
        result.num_dm, result.seed_mass, result.reservoir_mass,
        result.rho_inf, result.sound_speed_inf);

    walltime_measure("/SIDM/BHSeedDiag");
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
    if(!sidm_bhseed_params.DynMassCatchupOn || NumActiveBlackHoles <= 0)
        return;

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

#endif
