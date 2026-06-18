#include <math.h>
#include <string.h>
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "bhinfo.h"
#include "physconst.h"
#ifdef SIDM
#include "sidm_bhseed.h"
#endif

/* Structure needs to be packed to ensure disc write is the same on all architectures and the record size is correct. */
struct __attribute__((__packed__)) BHinfo{
    /* Stores sizeof(struct BHinfo) - 2 * sizeof(size1) . Allows size of record to be stored in the struct.*/
    int size1;
    MyIDType ID;
    MyFloat Mass;
    MyFloat Mdot;
    MyFloat Density;
    int minTimeBin;
    int encounter;

    double  MinPotPos[3];
    MyFloat MinPot;
    MyFloat BH_Entropy;
    MyFloat BH_SurroundingGasVel[3];
    MyFloat BH_accreted_momentum[3];

    MyFloat BH_accreted_Mass;
    MyFloat BH_accreted_BHMass;
    MyFloat BH_FeedbackWeightSum;

    MyIDType SPH_SwallowID;
    MyIDType SwallowID;

    int CountProgs;
    int Swallowed;

    /****************************************/
    double Pos[3];
    MyFloat BH_SurroundingDensity;
    MyFloat BH_SurroundingParticles;
    MyFloat BH_SurroundingVel[3];
    MyFloat BH_SurroundingRmsVel;

    double BH_DFAccel[3];
    double BH_DragAccel[3];
    double BH_FullTreeGravAccel[3];
    double Velocity[3];
    double Mtrack;
    double Mdyn;

    double KineticFdbkEnergy;
    double NumDM;
    /* Kept for backwards compatibility, not written to*/
    double V1sumDM[3];
    double VDisp;
    double MgasEnc;
    int KEflag;

#ifdef SIDM
    int SIDMSeedOrigin;
    int SIDMSeedTrigger;
    MyFloat SIDMSMFPMassInitial;
    MyFloat SIDMSMFPRadius;
    MyFloat SIDMDarkReservoirMass;
    MyFloat SIDMDarkReservoirInitial;
    MyFloat SIDMDarkMdot;
    MyFloat SIDMRhoInf;
    MyFloat SIDMSoundSpeedInf;
    MyFloat SIDMReservoirRadius;
    MyFloat SIDMGasDynMassDebt;
    MyFloat SIDMDMDynMassDebt;
    MyFloat SIDMCollapseProgress;
    MyFloat SIDMCollapseTime;
    MyFloat SIDMClockFoFMass;
#endif

    double a;
    /* See size1 above*/
    int size2;
};


size_t
collect_BH_info(const int * const ActiveBlackHoles, const int64_t NumActiveBlackHoles, struct BHPriv *priv, const struct part_manager_type * const PartManager, const struct bh_particle_data* const BHManager, FILE * FdBlackholeDetails, const double atime)
{
    int i;

    struct BHinfo * infos = (struct BHinfo *) mymalloc("BHDetailCache", NumActiveBlackHoles * sizeof(struct BHinfo));
    memset(infos, 0, NumActiveBlackHoles*sizeof(struct BHinfo));

    report_memory_usage("BLACKHOLE");

    const int size = sizeof(struct BHinfo) - sizeof(infos[0].size1) - sizeof(infos[0].size2);

    const struct particle_data * const pp = PartManager->Base;
    #pragma omp parallel for
    for(i = 0; i < NumActiveBlackHoles; i++)
    {
        const int p_i = ActiveBlackHoles ? ActiveBlackHoles[i] : i;
        if(p_i < 0 || p_i > PartManager->NumPart)
            endrun(1, "Bad index %d in black hole with %ld active, %ld total\n", p_i, NumActiveBlackHoles, PartManager->NumPart);
        if(pp[p_i].Type != 5)
            endrun(1, "Supposed BH %d of %ld has type %d\n", p_i, NumActiveBlackHoles, pp[p_i].Type);
        const int PI = pp[p_i].PI;

        struct BHinfo * info = &infos[i];
        /* Zero the struct*/
        info->size1 = size;
        info->size2 = size;
        info->ID = pp[p_i].ID;
        info->Mass = BHManager[PI].Mass;
        info->Mdot = BHManager[PI].Mdot;
        info->Density = BHManager[PI].Density;
        info->minTimeBin = BHManager[PI].minTimeBin;
        info->encounter = BHManager[PI].encounter;

        info->BH_Entropy = priv ? priv->BH_Entropy[PI] : 0;
        int k;
        for(k=0; k < 3; k++) {
            info->MinPotPos[k] = BHManager[PI].MinPotPos[k] - PartManager->CurrentParticleOffset[k];
            info->BH_SurroundingGasVel[k] = priv ? priv->BH_SurroundingGasVel[PI][k] : 0;
            info->BH_accreted_momentum[k] = priv ? priv->BH_accreted_momentum[PI][k] : 0;
            info->BH_DragAccel[k] = BHManager[PI].DragAccel[k];
            info->BH_FullTreeGravAccel[k] = pp[p_i].FullTreeGravAccel[k];
            info->Pos[k] = pp[p_i].Pos[k] - PartManager->CurrentParticleOffset[k];
            info->Velocity[k] = pp[p_i].Vel[k];
            info->BH_DFAccel[k] = BHManager[PI].DFAccel[k];
        }

        /****************************************************************************/
        /* Output some DF info for debugging */
        info->MinPot = BHManager[PI].MinPot;
        info->BH_SurroundingDensity = BHManager[PI].DF_SurroundingDensity;
        info->BH_SurroundingRmsVel = BHManager[PI].DF_SurroundingRmsVel;
        info->BH_SurroundingParticles = 0;
        info->BH_SurroundingVel[0] = BHManager[PI].DF_SurroundingVel[0];
        info->BH_SurroundingVel[1] = BHManager[PI].DF_SurroundingVel[1];
        info->BH_SurroundingVel[2] = BHManager[PI].DF_SurroundingVel[2];

        /****************************************************************************/
        info->BH_accreted_BHMass = priv ? priv->BH_accreted_BHMass[PI] : 0;
        info->BH_accreted_Mass = priv ? priv->BH_accreted_Mass[PI] : 0;
        info->BH_FeedbackWeightSum = priv ? priv->BH_FeedbackWeightSum[PI] : 0;

        info->SPH_SwallowID = priv ? priv->SPH_SwallowID[PI] : 0;
        info->SwallowID =  BHManager[PI].SwallowID;
        info->CountProgs = BHManager[PI].CountProgs;
        info->Swallowed =  pp[p_i].Swallowed;
        /************************************************************************************************/
        /* When SeedBHDynMass is larger than gas particle mass, we have three mass tracer of blackhole. */
        /* BHP(p_i).Mass : intrinsic mass of BH, accreted every (active) time step.                     */
        /* P[p_i].Mass :  Dynamic mass of BH, used for gravitational interaction.                       */
        /*                Starts to accrete gas particle when BHP(p_i).Mass > SeedBHDynMass             */
        /* BHP(p_i).Mtrack: Initialized as gas particle mass, and is capped at SeedBHDynMass,           */
        /*                 it traces BHP(p_i).Mass by swallowing gas when BHP(p_i).Mass < SeedBHDynMass */
        /************************************************************************************************/
        info->Mtrack = BHManager[PI].Mtrack;
        info->Mdyn = pp[p_i].Mass;

        info->KineticFdbkEnergy = BHManager[PI].KineticFdbkEnergy;
        info->NumDM = priv ? priv->NumDM[PI] : 0;
        info->VDisp = BHManager[PI].VDisp;
        info->MgasEnc = priv ? priv->MgasEnc[PI] : 0;
        info->KEflag = priv ? priv->KEflag[PI] : 0;

#ifdef SIDM
        info->SIDMSeedOrigin = BHManager[PI].SIDMSeedOrigin;
        info->SIDMSeedTrigger = BHManager[PI].SIDMSeedTrigger;
        info->SIDMSMFPMassInitial = BHManager[PI].SIDMSMFPMassInitial;
        info->SIDMSMFPRadius = BHManager[PI].SIDMSMFPRadius;
        info->SIDMDarkReservoirMass = BHManager[PI].SIDMDarkReservoirMass;
        info->SIDMDarkReservoirInitial = BHManager[PI].SIDMDarkReservoirInitial;
        info->SIDMDarkMdot = BHManager[PI].SIDMDarkMdot;
        info->SIDMRhoInf = BHManager[PI].SIDMRhoInf;
        info->SIDMSoundSpeedInf = BHManager[PI].SIDMSoundSpeedInf;
        info->SIDMReservoirRadius = BHManager[PI].SIDMReservoirRadius;
        info->SIDMGasDynMassDebt = BHManager[PI].SIDMGasDynMassDebt;
        info->SIDMDMDynMassDebt = BHManager[PI].SIDMDMDynMassDebt;
        info->SIDMCollapseProgress = BHManager[PI].SIDMCollapseProgress;
        info->SIDMCollapseTime = BHManager[PI].SIDMCollapseTime;
        info->SIDMClockFoFMass = BHManager[PI].SIDMClockFoFMass;
#endif

        info->a = atime;
    }

    fwrite(infos,sizeof(struct BHinfo),NumActiveBlackHoles,FdBlackholeDetails);
    // fflush(FdBlackholeDetails);
    myfree(infos);
    int64_t totalN;

    MPI_Allreduce(&NumActiveBlackHoles, &totalN, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "Written details of %ld blackholes in %lu bytes each.\n", totalN, sizeof(struct BHinfo));
    return totalN * sizeof(struct BHinfo);
}

void
write_blackhole_txt(FILE * FdBlackHoles, const struct UnitSystem units, const double atime)
{
    int total_bh, i;
    double total_mdoteddington;
    double total_mass_holes, total_mdot;
#ifdef SIDM
    const int write_sidm_columns = sidm_bhseed_is_enabled();
    int total_sidm_bh;
    double total_sidm_dark_mdot;
    double total_sidm_dark_meddington;
    double total_sidm_reservoir;
    double total_sidm_reservoir_initial;
    double total_sidm_gas_dyn_mass_debt;
    double total_sidm_dm_dyn_mass_debt;
#endif

    double Local_BH_mass = 0;
    double Local_BH_Mdot = 0;
    double Local_BH_Medd = 0;
    int Local_BH_num = 0;
#ifdef SIDM
    double Local_SIDM_DarkMdot = 0;
    double Local_SIDM_DarkMedd = 0;
    double Local_SIDM_Reservoir = 0;
    double Local_SIDM_ReservoirInitial = 0;
    double Local_SIDM_GasDynMassDebt = 0;
    double Local_SIDM_DMDynMassDebt = 0;
    int Local_SIDM_BH_num = 0;
#endif
    /* Compute total mass of black holes
     * present by summing contents of black hole array*/
    #pragma omp parallel for reduction(+ : Local_BH_num) reduction(+: Local_BH_mass) reduction(+: Local_BH_Mdot) reduction(+: Local_BH_Medd)
    for(i = 0; i < SlotsManager->info[5].size; i ++)
    {
        if(BhP[i].SwallowID != (MyIDType) -1)
            continue;
        Local_BH_num++;
        Local_BH_mass += BhP[i].Mass;
        Local_BH_Mdot += BhP[i].Mdot;
        if(BhP[i].Mass > 0)
            Local_BH_Medd += BhP[i].Mdot/BhP[i].Mass;
    }
#ifdef SIDM
    if(write_sidm_columns) {
        #pragma omp parallel for reduction(+ : Local_SIDM_BH_num) reduction(+: Local_SIDM_DarkMdot, Local_SIDM_DarkMedd, Local_SIDM_Reservoir, Local_SIDM_ReservoirInitial, Local_SIDM_GasDynMassDebt, Local_SIDM_DMDynMassDebt)
        for(i = 0; i < SlotsManager->info[5].size; i ++)
        {
            if(BhP[i].SwallowID != (MyIDType) -1 || !BhP[i].SIDMSeedOrigin)
                continue;
            Local_SIDM_BH_num++;
            Local_SIDM_DarkMdot += BhP[i].SIDMDarkMdot;
            if(BhP[i].Mass > 0)
                Local_SIDM_DarkMedd += BhP[i].SIDMDarkMdot / BhP[i].Mass;
            Local_SIDM_Reservoir += BhP[i].SIDMDarkReservoirMass;
            Local_SIDM_ReservoirInitial += BhP[i].SIDMDarkReservoirInitial;
            Local_SIDM_GasDynMassDebt += BhP[i].SIDMGasDynMassDebt;
            Local_SIDM_DMDynMassDebt += BhP[i].SIDMDMDynMassDebt;
        }
    }
#endif

    MPI_Reduce(&Local_BH_mass, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_Medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Local_BH_num, &total_bh, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef SIDM
    if(write_sidm_columns) {
        MPI_Reduce(&Local_SIDM_DarkMdot, &total_sidm_dark_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_DarkMedd, &total_sidm_dark_meddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_Reservoir, &total_sidm_reservoir, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_ReservoirInitial, &total_sidm_reservoir_initial, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_GasDynMassDebt, &total_sidm_gas_dyn_mass_debt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_DMDynMassDebt, &total_sidm_dm_dyn_mass_debt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Local_SIDM_BH_num, &total_sidm_bh, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif

    if(FdBlackHoles)
    {
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (units.UnitMass_in_g / SOLAR_MASS) / (units.UnitTime_in_s / SEC_PER_YEAR);

        total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS /
                    (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * units.UnitTime_in_s);

#ifdef SIDM
        if(write_sidm_columns) {
            double sidm_dark_mdot_in_msun_per_year =
                total_sidm_dark_mdot * (units.UnitMass_in_g / SOLAR_MASS) / (units.UnitTime_in_s / SEC_PER_YEAR);
            total_sidm_dark_meddington *= 1.0 / ((4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS /
                        (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * units.UnitTime_in_s);

            fprintf(FdBlackHoles, "%g %d %g %g %g %g %d %g %g %g %g %g %g %g\n",
                    atime, total_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mdoteddington,
                    total_sidm_bh, total_sidm_dark_mdot, sidm_dark_mdot_in_msun_per_year,
                    total_sidm_dark_meddington, total_sidm_reservoir, total_sidm_reservoir_initial,
                    total_sidm_gas_dyn_mass_debt, total_sidm_dm_dyn_mass_debt);
        } else
#endif
        {
        fprintf(FdBlackHoles, "%g %d %g %g %g %g\n",
                atime, total_bh, total_mass_holes, total_mdot, mdot_in_msun_per_year, total_mdoteddington);
        }
        fflush(FdBlackHoles);
    }
}
