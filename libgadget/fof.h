#ifndef FOF_H
#define FOF_H

#include "utils/paramset.h"
#include "cosmology.h"
#include "timestep.h"
#include "slotsmanager.h"
#include "utils/system.h"

void set_fof_params(ParameterSet * ps);

#ifdef SIDM
#define SIDM_VMAX_PROFILE_BINS 24
#define SIDM_BHSEED_NFW_XMAX 2.1625815870646098
#define SIDM_BHSEED_NFW_VMAX_COEFF 1.648
#endif

double fof_get_mean_primary_separation(const struct part_manager_type * PartManager, const Cosmology * CP, int primary_mask);
double fof_get_comoving_linking_length(const struct part_manager_type * PartManager, const Cosmology * CP);
void fof_init(const struct part_manager_type * PartManager, const Cosmology * CP, double Type1MeanSeparation);
/* For the tests*/
void set_fof_testpar(int FOFSaveParticles, double FOFHaloLinkingLength, int FOFHaloMinLength);

struct BaseGroup {
    int OriginalTask;
    int OriginalIndex;
    int Length;
    int GrNr;
    MyIDType MinID;
    int MinIDTask;
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    float FirstPos[3];
};

struct Group
{
    struct BaseGroup base;
    int Length;
    int LenType[6];
    double MassType[6];
    double Mass;
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    double CM[3];
    double Vel[3];

    double Imom[3][3]; /* sum M r_j r_k */
    double Jmom[3]; /* sum M R_i x V_i  */

    double Sfr;
    /* Metal masses. These are the total mass in metals in the gas and stars respectively,
     * and then the species specific breakdowns for stars and gas.
     * You can obtain metallicities by dividing them by the type-specific masses.*/
    double GasMetalMass;
    double StellarMetalMass;
    float StellarMetalElemMass[NMETALS];
    float GasMetalElemMass[NMETALS];

    /* Number of gas particles in the halo which have had helium ionization happen to them*/
    float MassHeIonized;
    /*These are used for storing black hole properties*/
    double BH_Mass;
    double BH_Mdot;
    double MaxDens;

    int seed_index;
    int seed_task;

#ifdef SIDM
    double SIDMSeedPotential;
    double SIDMSeedPos[3];
    int sidm_seed_index;
    int sidm_seed_task;
    double SIDMBHCollapseProgress;
    double SIDMBHLastCheckTime;
    MyIDType SIDMBHClockID;
    double SIDMBHClockFoFMass;
    double SIDMVmaxProfileMass[SIDM_VMAX_PROFILE_BINS];
    double SIDMVmaxProfileCount[SIDM_VMAX_PROFILE_BINS];
    double SIDMVmax; /* sqrt(G M(<Rmax) / Rmax) with comoving Rmax */
    double SIDMRmax;
    double SIDMVmaxProfileRMin;
    double SIDMVmaxProfileRMax;
    int SIDMVmaxProfileBins;
#endif
};

/* Structure to hold all allocated FOF groups*/
typedef struct FOFGroups
{
    struct Group * Group;
    int64_t Ngroups;
    int64_t TotNgroups;
} FOFGroups;

/* Computes the Group structure, saved as a global array below.
 * If StoreGrNr is true, this writes to GrNr in partmanager.h.
 * Note this over-writes PeanoKey and means the tree cannot be rebuilt.*/
FOFGroups fof_fof(DomainDecomp * ddecomp, const int StoreGrNr, Cosmology * CP,
    const DriftKickTimes * times, MPI_Comm Comm);

/*Frees the Group structure*/
void fof_finish(FOFGroups * fof);

/*Uses the Group structure to seed blackholes.
 * The active particle struct is used only because we may need to reallocate it. Randon number seeds the BH mass.*/
void fof_seed(FOFGroups * fof, ActiveParticles * act, double atime, const RandTable * const rnd, MPI_Comm Comm);

#ifdef SIDM
void fof_seed_sidm(FOFGroups * fof, ActiveParticles * act, double atime, Cosmology * CP, const DriftKickTimes * times, const struct UnitSystem units, MPI_Comm Comm);
#endif

/* Saves the Group structure to disc.
 Returns 1 if a domain_exchange is needed afterwards.*/
int fof_save_groups(FOFGroups * fof, const char * OutputDir, const char * FOFFileBase, int num, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, MPI_Comm Comm);

/* Does the actual saving of the particles
 Returns 1 if a domain_exchange is needed afterwards.*/
int fof_save_particles(FOFGroups * fof, char * fname, int SaveParticles, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, MPI_Comm Comm);

#endif
