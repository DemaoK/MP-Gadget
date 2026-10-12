/* Native split-gravity activity and canonical drift around an off-grid collision. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "stub.h"
#include <libgadget/drift.h>
#include <libgadget/gravity.h>
#include <libgadget/sidm-local.h>
#include <libgadget/timefac.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/walltime.h>
#ifdef SIDM
static Cosmology cp;
static struct ClockTable clocks;
static int rank;
static void near(double a,double b)
{ if(fabs(a-b)>2e-9*fmax(1,fabs(b))) fail_msg("actual %.17g expected %.17g",a,b); }
static void test_collision_activity_and_drift(void **state)
{
    (void)state;
    particle_alloc_memory(PartManager,1024,3);
    PartManager->NumPart=rank?0:3;
    memset(P,0,3*sizeof(*P));
    const double velocity[]={.5,-.25,.125};
    const int fine=TIMEBINS-4,coarse=TIMEBINS-2;
    const inttime_t collision=(TIMEBASE/3)&~(inttime_t)1;
    sidm_local_configure(1,.2,.25,.1);
    DriftKickTimes times=init_driftkicktime(0);
    times.PM_length=TIMEBASE;times.mintimebin=times.mingravtimebin=fine;
    times.maxtimebin=coarse;
    for(int i=0;i<PartManager->NumPart;i++) {
        P[i].ID=i+1;P[i].Type=1;P[i].Mass=1;P[i].Hsml=100;
        P[i].Pos[0]=P[i].Pos[1]=P[i].Pos[2]=100;
        P[i].Vel[0]=velocity[i];P[i].TimeBinGravity=i<2?fine:coarse;
    }
    sidm_local_initialize(&times,&cp);
    for(int i=0;i<PartManager->NumPart;i++) P[i].TiSIDMNext=i?TIMEBASE:collision;
    drift_all_particles(0,collision,&cp,(double[3]){0});
    times.Ti_Current=collision;
    ActiveParticles act=init_empty_active_particles(PartManager);
    build_active_particles(&act,&times,1,get_atime(collision),PartManager);
    assert_int_equal(act.NumActiveGravity,0);
    assert_int_equal(act.NumActiveSIDM,rank?0:1);
    assert_int_equal(act.NumActiveParticle,rank?0:1);
    free_active_particles(&act);
    if(!rank) {
        const double swap=P[0].Vel[0];P[0].Vel[0]=P[2].Vel[0];P[2].Vel[0]=swap;
        assert_int_equal(P[2].TimeBinGravity,coarse);
    }
    drift_all_particles(collision,TIMEBASE,&cp,(double[3]){0});
    const double before=2/cp.Hubble*(1/sqrt(.1)-1/sqrt(get_atime(collision)));
    const double after=2/cp.Hubble*(1/sqrt(get_atime(collision))-1/sqrt(.2));
    for(int i=0;i<PartManager->NumPart;i++) {
        const int swapped=i==0?2:i==2?0:1;
        near(P[i].Pos[0],100+velocity[i]*before+velocity[swapped]*after);
        near(P[i].Vel[0],velocity[swapped]);
    }
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
    sidm_local_configure(0,.2,.25,.1);
}
static void test_sparse_top_bin(void **state)
{
    (void)state;
    /* Exercise the actual bin assignment, histogram and collective pushdown
     * with the highest legal bin, two OpenMP threads and empty MPI ranks.
     * No lower source subset exists, so no force tree is needed here. */
    particle_alloc_memory(PartManager,8,3);
    PartManager->NumPart=rank?0:2;
    memset(P,0,2*sizeof(*P));
    for(int i=0;i<PartManager->NumPart;i++) {
        P[i].ID=i+1;P[i].Type=1;P[i].Mass=1;
    }
    struct gravshort_tree_params gravity={0};
    gravity.FractionalGravitySoftening=.1;
    set_gravshort_treepar(gravity);gravshort_set_softenings(1);
    ParameterSet *ps=parameter_set_new();
    param_declare_double(ps,"ErrTolIntAccuracy",OPTIONAL,.02,"test");
    param_declare_double(ps,"MaxGasVel",OPTIONAL,1e6,"test");
    param_declare_double(ps,"MaxSizeTimestep",OPTIONAL,1,"test");
    param_declare_double(ps,"MinSizeTimestep",OPTIONAL,1e-12,"test");
    param_declare_int(ps,"ForceEqualTimesteps",OPTIONAL,0,"test");
    param_declare_double(ps,"MaxRMSDisplacementFac",OPTIONAL,.25,"test");
    param_declare_double(ps,"CourantFac",OPTIONAL,.15,"test");
    assert_int_equal(param_parse(ps,""),0);set_timestep_params(ps);
    parameter_set_free(ps);
    DriftKickTimes times=init_driftkicktime(0);times.PM_length=TIMEBASE;
    ActiveParticles act=init_empty_active_particles(PartManager);
    act.NumActiveGravity=PartManager->NumPart;
    PetaPM pm={0};pm.Asmth=1;pm.Nmesh=1;
    cp.GravInternal=1;
    struct grav_accel_store stored={0};
    assert_int_equal(hierarchical_gravity_and_timesteps(&act,&pm,NULL,stored,
        &times,1,0,2,&cp,"."),0);
    assert_int_equal(times.mingravtimebin,TIMEBINS);
    assert_int_equal(times.maxtimebin,TIMEBINS);
    for(int i=0;i<PartManager->NumPart;i++) {
        assert_int_equal(P[i].TimeBinGravity,TIMEBINS);
        near(P[i].Vel[0],0);
    }
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
}
static int setup(void **state)
{
    (void)state;MPI_Comm_rank(MPI_COMM_WORLD,&rank);walltime_init(&clocks);
    cp.Omega0=1;cp.OmegaBaryon=.05;cp.HubbleParam=.7;cp.CMBTemperature=2.7255;
    init_cosmology(&cp,.1,get_unitsystem(3.085678e21,1.989e43,1e5));
    /* Pure matter reference excludes the native massless-neutrino background. */
    memset(cp.ONu.nu_degeneracies,0,sizeof(cp.ONu.nu_degeneracies));
    double outputs[]={.1,.2};set_sync_params_test(2,outputs);
    setup_sync_points(&cp,.1,.2,0,0);sidm_time_factors_init(&cp);
    return 0;
}
int main(void)
{
    omp_set_dynamic(0);omp_set_num_threads(2);
    const struct CMUnitTest tests[]={cmocka_unit_test(test_collision_activity_and_drift),
        cmocka_unit_test(test_sparse_top_bin)};
    return cmocka_run_group_tests_mpi(tests,setup,NULL);
}
#else
int main(void) {puts("SKIP hierarchical SIDM: SIDM disabled");return 0;}
#endif
