/*Simple test for the exchange function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#define qsort_openmp qsort

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include "stub.h"

static struct ClockTable CT;

#define NUMPART1 8
static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * NumPart);
    PartManager->NumPart = NumPart;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        /* DM only*/
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        int j;
        for(j=0; j<3; j++) {
            P[i].Pos[j] = BoxSize * (j+1) * P[i].ID / (PartManager->NumPart * NTask);
            while(P[i].Pos[j] > BoxSize)
                P[i].Pos[j] -= BoxSize;
        }
    }
    int64_t num_primary_total = 0;
    MPI_Allreduce(&PartManager->NumPart, &num_primary_total, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    Cosmology CP = {0};
    CP.Omega0 = 1;
    CP.OmegaCDM = 1;
    CP.RhoCrit = num_primary_total / pow(BoxSize, 3);
    const double mean_separation = fof_get_mean_primary_separation(PartManager, &CP, 1 << 1);
    fof_init(PartManager, &CP, mean_separation);
    /* TODO: Here create particles in some halo-like configuration*/
    return 0;
}

static void
test_fof_linking_uses_primary_mass_density(void **state)
{
    (void) state;
    const int NumPrimary = 8;
    const int NumBoundary = 64;
    const double BoxSize = 100;
    const double MeanSeparation = 2;

    set_fof_testpar(1, 0.2, 5);
    particle_alloc_memory(PartManager, BoxSize, NumPrimary + NumBoundary);
    PartManager->NumPart = NumPrimary + NumBoundary;

    int i;
    for(i = 0; i < NumPrimary; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
    }
    for(; i < PartManager->NumPart; i++) {
        P[i].Type = 3;
        P[i].Mass = 8;
        P[i].IsGarbage = 0;
    }

    Cosmology CP = {0};
    CP.Omega0 = 1;
    CP.OmegaCDM = 1;
    CP.RhoCrit = 1 / pow(MeanSeparation, 3);

    const double link_length = fof_get_comoving_linking_length(PartManager, &CP);
    assert_true(fabs(link_length - 0.2 * MeanSeparation) < 1e-12);

    myfree(P);
}

static void
test_fof_linking_handles_massive_neutrino_density(void **state)
{
    (void) state;
    const int NumPrimary = 8;
    const double BoxSize = 100;
    const double MeanSeparation = 2;

    set_fof_testpar(1, 0.2, 5);
    particle_alloc_memory(PartManager, BoxSize, NumPrimary);
    PartManager->NumPart = NumPrimary;

    int i;
    for(i = 0; i < NumPrimary; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
    }

    Cosmology CP = {0};
    CP.Omega0 = 1.0;
    CP.OmegaBaryon = 0.1;
    CP.OmegaCDM = 0.8;
    CP.RhoCrit = 1 / pow(MeanSeparation, 3) / (CP.OmegaCDM + CP.OmegaBaryon);

    const double mean_separation = fof_get_mean_primary_separation(PartManager, &CP, 1 << 1);
    assert_true(fabs(mean_separation - MeanSeparation) < 1e-12);

    myfree(P);
}

static void
test_fof_linking_uses_cdm_density_when_baryons_are_separate(void **state)
{
    (void) state;
    const int NumPrimary = 8;
    const int NumGas = 1;
    const double BoxSize = 100;
    const double MeanSeparation = 2;

    set_fof_testpar(1, 0.2, 5);
    particle_alloc_memory(PartManager, BoxSize, NumPrimary + NumGas);
    PartManager->NumPart = NumPrimary + NumGas;

    int i;
    for(i = 0; i < NumPrimary; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
    }
    P[i].Type = 0;
    P[i].Mass = 1;
    P[i].IsGarbage = 0;

    Cosmology CP = {0};
    CP.Omega0 = 1.0;
    CP.OmegaBaryon = 0.1;
    CP.OmegaCDM = 0.8;
    CP.RhoCrit = 1 / pow(MeanSeparation, 3) / CP.OmegaCDM;

    const double mean_separation = fof_get_mean_primary_separation(PartManager, &CP, 1 << 1);
    assert_true(fabs(mean_separation - MeanSeparation) < 1e-12);

    myfree(P);
}

static void
test_fof_linking_uses_cdm_density_when_bh_baryons_are_separate(void **state)
{
    (void) state;
    const int NumPrimary = 8;
    const int NumBH = 1;
    const double BoxSize = 100;
    const double MeanSeparation = 2;

    set_fof_testpar(1, 0.2, 5);
    particle_alloc_memory(PartManager, BoxSize, NumPrimary + NumBH);
    PartManager->NumPart = NumPrimary + NumBH;

    int i;
    for(i = 0; i < NumPrimary; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
    }
    P[i].Type = 5;
    P[i].Mass = 1;
    P[i].IsGarbage = 0;

    Cosmology CP = {0};
    CP.Omega0 = 1.0;
    CP.OmegaBaryon = 0.1;
    CP.OmegaCDM = 0.8;
    CP.RhoCrit = 1 / pow(MeanSeparation, 3) / CP.OmegaCDM;

    const double mean_separation = fof_get_mean_primary_separation(PartManager, &CP, 1 << 1);
    assert_true(fabs(mean_separation - MeanSeparation) < 1e-12);

    myfree(P);
}

#ifdef SIDM
static void
test_fof_linking_ignores_sidm_type5_when_detecting_separate_baryons(void **state)
{
    (void) state;
    const int NumPrimary = 8;
    const int NumBH = 1;
    const double BoxSize = 100;
    const double MeanSeparation = 2;

    set_fof_testpar(1, 0.2, 5);
    particle_alloc_memory(PartManager, BoxSize, NumPrimary + NumBH);
    PartManager->NumPart = NumPrimary + NumBH;

    int i;
    for(i = 0; i < NumPrimary; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
    }
    P[i].Type = 5;
    P[i].Mass = 1;
    P[i].IsGarbage = 0;
    P[i].PI = 0;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t newSlots[6] = {0, 0, 0, 0, 0, NumBH};
    slots_reserve(0, newSlots, SlotsManager);
    SlotsManager->info[5].size = NumBH;
    BhP[0].SIDMSeedOrigin = 1;

    Cosmology CP = {0};
    CP.Omega0 = 1.0;
    CP.OmegaBaryon = 0.1;
    CP.OmegaCDM = 0.8;
    CP.RhoCrit = 1 / pow(MeanSeparation, 3) / (CP.OmegaCDM + CP.OmegaBaryon);

    const double mean_separation = fof_get_mean_primary_separation(PartManager, &CP, 1 << 1);
    assert_true(fabs(mean_separation - MeanSeparation) < 1e-12);

    slots_free(SlotsManager);
    myfree(P);
}
#endif

static void
test_fof(void **state)
{
    int NTask;
    walltime_init(&CT);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    set_fof_testpar(1, 0.2, 5);
    init_forcetree_params(0.7);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int NumPart = 512*512 / NTask;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    Cosmology CP = {0};
    CP.RhoCrit = 1;
    CP.GravInternal = 1;
    FOFGroups fof = fof_fof(&ddecomp, 1, &CP, NULL, MPI_COMM_WORLD);

    /* Example assertion: this checks that the groups were allocated. */
    assert_all_true(fof.Group);
    assert_true(fof.TotNgroups == 1);
    /* Assert some more things about the particles,
     * maybe checking the halo properties*/

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(P);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fof_linking_uses_primary_mass_density),
        cmocka_unit_test(test_fof_linking_handles_massive_neutrino_density),
        cmocka_unit_test(test_fof_linking_uses_cdm_density_when_baryons_are_separate),
        cmocka_unit_test(test_fof_linking_uses_cdm_density_when_bh_baryons_are_separate),
#ifdef SIDM
        cmocka_unit_test(test_fof_linking_ignores_sidm_type5_when_detecting_separate_baryons),
#endif
        cmocka_unit_test(test_fof),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
