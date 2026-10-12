/* Cosmological exposure and transport clocks with mixed particle species. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "stub.h"
#include <libgadget/sidm-local.h>
#include <libgadget/gravity.h>
#include <libgadget/timefac.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/walltime.h>

#ifdef SIDM
static Cosmology cp;
static struct ClockTable clocks;
static int rank;
static void near(double a,double b)
{
    assert_true(isfinite(a));
    if(fabs(a-b)>3e-10*fmax(1,fabs(b)))
        fail_msg("actual %.17g expected %.17g",a,b);
}
static double drift_exact(double a,double b)
{ return 2/cp.Hubble*(1/sqrt(a)-1/sqrt(b)); }
static double scatter_exact(double a,double b)
{ return .4/cp.Hubble*(pow(a,-2.5)-pow(b,-2.5)); }
static void test_cosmological_factors(void **state)
{
    (void)state;
    const inttime_t ticks[]={0,2,TIMEBASE/8,TIMEBASE/2,TIMEBASE};
    for(unsigned i=0;i<sizeof(ticks)/sizeof(*ticks);i++) {
        const double a=.1,b=exp(loga_from_ti(ticks[i]));
        near(sidm_drift_factor(&cp,0,ticks[i]),drift_exact(a,b));
        near(sidm_scatter_factor(&cp,0,ticks[i]),scatter_exact(a,b));
        near(sidm_scatter_factor(&cp,ticks[i],ticks[i]),0);
    }
    near(sidm_scatter_factor(&cp,0,TIMEBASE),
         sidm_scatter_factor(&cp,0,TIMEBASE/2)+sidm_scatter_factor(&cp,TIMEBASE/2,TIMEBASE));
    /* Local differential ratio is a^-2, not a constant endpoint conversion. */
    const inttime_t t=TIMEBASE/4,dt=TIMEBASE/65536;
    const double a=exp(loga_from_ti(t));
    near(sidm_scatter_factor(&cp,t,t+dt)/sidm_drift_factor(&cp,t,t+dt),
         scatter_exact(a,exp(loga_from_ti(t+dt)))/drift_exact(a,exp(loga_from_ti(t+dt))));
}
static void clocks_for_boost(double boost,inttime_t *deadline,double *gravity)
{
    particle_alloc_memory(PartManager,1e9,3);
    PartManager->NumPart=3;
    memset(P,0,3*sizeof(*P));
    for(int i=0;i<3;i++) {
        P[i].ID=3*rank+i+1;P[i].Type=i==2?0:1;P[i].Mass=i==2?1000:1;
        P[i].Hsml=1e6;P[i].TimeBinGravity=TIMEBINS-2;
        P[i].Vel[0]=i==2?1e8:boost+(i?-.5:.5);
    }
    sidm_local_configure(1,.2,.25,.1);
    DriftKickTimes times=init_driftkicktime(0);
    times.PM_length=TIMEBASE;times.mintimebin=times.mingravtimebin=TIMEBINS-2;
    sidm_local_initialize(&times,&cp);
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    assert_int_equal(visit.sources.NumActiveSIDM,2);
    const double bound=1.1*sidm_scatter_factor(&cp,0,TIMEBASE/16);
    for(int a=0;a<visit.sources.NumActiveParticle;a++) {
        const int i=visit.sources.ActiveParticle[a];
        assert_int_equal(P[i].Type,1);near(visit.intervals[i],0);
        visit.limits[i]=bound;
    }
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    sidm_local_update_speed_bound();
    *deadline=sidm_local_next_event(&times,&cp,0);
    assert_true(*deadline>0 && *deadline<=TIMEBASE/16);
    assert_true(sidm_scatter_factor(&cp,0,*deadline)<=bound);
    *gravity=sidm_local_gravity_limit(0,&times);
    near(sidm_local_gravity_limit(2,&times),DBL_MAX);
    assert_true(*gravity>0 && *gravity<DBL_MAX);
    times.Ti_Current=*deadline;
    visit=sidm_local_begin_active(&times,&cp,0,NULL);
    assert_int_equal(visit.sources.NumActiveParticle,2);
    const double atime=exp(loga_from_ti(*deadline));
    for(int a=0;a<visit.sources.NumActiveParticle;a++) {
        const int i=visit.sources.ActiveParticle[a];
        near(visit.intervals[i],scatter_exact(.1,atime));
        visit.limits[i]=bound;
    }
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    /* A PM output flushes the exposure regardless of the previous rate bin. */
    for(int i=0;i<2;i++) P[i].TiSIDMNext=TIMEBASE;
    times.Ti_Current=times.PM_kick=TIMEBASE;
    times.Ti_kick[TIMEBINS-2]=TIMEBASE;
    visit=sidm_local_begin_active(&times,&cp,1,NULL);
    for(int a=0;a<visit.sources.NumActiveParticle;a++) {
        const int i=visit.sources.ActiveParticle[a];
        near(visit.intervals[i],scatter_exact(atime,.2));
        visit.limits[i]=bound;
    }
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    sidm_local_assert_flushed(&times);
    for(int i=0;i<2;i++) assert_int_equal(P[i].TimeBinSIDM,0);
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
    sidm_local_configure(0,.2,.25,.1);
}
static void test_mixed_clocks_and_boost(void **state)
{
    (void)state;
    inttime_t first,boosted;double limit,boost_limit;
    clocks_for_boost(0,&first,&limit);
    clocks_for_boost(1048576,&boosted,&boost_limit);
    assert_int_equal(first,boosted);near(limit,boost_limit);
}
/* A receiver keeps its source epoch when an unrelated source causes the
 * global post-opening velocity envelope to increase. */
static void transport_case(int expect_resolution_guard)
{
    const double speed=1e6,transport=.2;
    const inttime_t initial_deadline=expect_resolution_guard?4:8;
    particle_alloc_memory(PartManager,1e9,2);
    PartManager->NumPart=2;memset(P,0,2*sizeof(*P));
    sidm_local_configure(1,transport,.25,.1);
    DriftKickTimes times=init_driftkicktime(0);
    times.PM_length=TIMEBASE;times.mintimebin=times.mingravtimebin=TIMEBINS-2;
    const double budget=2*speed*1.1*sidm_drift_factor(&cp,0,initial_deadline);
    for(int i=0;i<2;i++) {
        P[i].ID=2*rank+i+1;P[i].Type=1;P[i].Mass=1;
        P[i].Hsml=budget/transport;P[i].TimeBinGravity=TIMEBINS-2;
        P[i].Vel[0]=i?-speed:speed;
    }
    sidm_local_initialize(&times,&cp);
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    visit.limits[0]=1.1*sidm_scatter_factor(&cp,0,2);
    visit.limits[1]=DBL_MAX;
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    sidm_local_update_speed_bound();
    assert_int_equal(sidm_local_next_event(&times,&cp,0),2);
    assert_int_equal(P[1].TiSIDMNext,initial_deadline);
    assert_int_equal(P[1].TiSIDMLast,0);
    times.Ti_Current=2;
    visit=sidm_local_begin_active(&times,&cp,0,NULL);
    assert_int_equal(visit.sources.NumActiveParticle,1);
    assert_int_equal(visit.sources.ActiveParticle[0],0);
    visit.limits[0]=1.1*sidm_scatter_factor(&cp,2,4);
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    const double old_travel=2*speed*sidm_drift_factor(&cp,0,2);
    const double new_speed=(expect_resolution_guard?2:1.5)*speed;
    P[0].Vel[0]=new_speed;P[1].Vel[0]=-new_speed;
    sidm_local_update_speed_bound();
    const double minimum_travel=2*new_speed*sidm_drift_factor(&cp,2,4);
    if(expect_resolution_guard) {
        /* No representable next event can honor the remaining budget. This
         * opt-in fixture must terminate through the production guard. */
        assert_true(old_travel+minimum_travel>budget+128*DBL_EPSILON);
        sidm_local_next_event(&times,&cp,0);
        fail_msg("Expected the SIDM transport resolution guard to terminate");
    }
    assert_int_equal(sidm_local_next_event(&times,&cp,0),4);
    assert_int_equal(P[1].TiSIDMNext,4);
    assert_int_equal(P[1].TimeBinSIDM,1);
    assert_int_equal(P[1].TiSIDMLast,0);
    assert_int_equal(P[1].TimeBinGravity,TIMEBINS-2);
    assert_true(old_travel+minimum_travel<=budget);
    /* A bin-one deadline is the next event itself. It is consumed before any
     * subsequent opening velocity change can enter deadline selection. */
    times.Ti_Current=4;
    visit=sidm_local_begin_active(&times,&cp,0,NULL);
    assert_int_equal(visit.sources.NumActiveParticle,2);
    near(visit.intervals[1]/sidm_scatter_factor(&cp,0,4),1);
    visit.limits[0]=visit.limits[1]=DBL_MAX;
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    assert_int_equal(P[1].TiSIDMLast,4);
    assert_true(P[1].TiSIDMNext>4);
    P[0].Vel[0]=2*speed;P[1].Vel[0]=-2*speed;
    sidm_local_update_speed_bound();
    assert_true(sidm_local_next_event(&times,&cp,0)>4);
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
    sidm_local_configure(0,.2,.25,.1);
}
static void test_transport_deadline_tightening(void **state)
{ (void)state;transport_case(0); }
static void test_transport_resolution_guard(void **state)
{ (void)state;transport_case(1); }

static void test_gravity_transport_shortening(void **state)
{
    (void)state;
    particle_alloc_memory(PartManager,1e9,4);
    PartManager->NumPart=4;memset(P,0,4*sizeof(*P));
    ParameterSet *ps=parameter_set_new();
    param_declare_double(ps,"ErrTolIntAccuracy",OPTIONAL,.02,"test");
    param_declare_double(ps,"MaxGasVel",OPTIONAL,1e6,"test");
    param_declare_double(ps,"MaxSizeTimestep",OPTIONAL,1,"test");
    param_declare_double(ps,"MinSizeTimestep",OPTIONAL,1e-12,"test");
    param_declare_int(ps,"ForceEqualTimesteps",OPTIONAL,0,"test");
    param_declare_double(ps,"MaxRMSDisplacementFac",OPTIONAL,.25,"test");
    param_declare_double(ps,"CourantFac",OPTIONAL,.15,"test");
    assert_int_equal(param_parse(ps,""),0);set_timestep_params(ps);parameter_set_free(ps);
    const int target_bin=TIMEBINS-4;
    DriftKickTimes times=init_driftkicktime(TIMEBASE/2);
    times.PM_length=TIMEBASE;times.mintimebin=times.mingravtimebin=TIMEBINS-1;
    const double speed=.5;
    const double budget=1.1*sidm_drift_factor(&cp,times.Ti_Current,
        times.Ti_Current+dti_from_timebin(target_bin));
    for(int i=0;i<4;i++) {
        P[i].ID=4*rank+i+1;P[i].Type=1;P[i].Mass=1;
        P[i].Hsml=budget*speed/.2;P[i].TimeBinGravity=TIMEBINS-1;
        P[i].Vel[0]=(i&1)?-speed:speed;
    }
    struct gravshort_tree_params gravity={0};gravity.FractionalGravitySoftening=.01;
    set_gravshort_treepar(gravity);gravshort_set_softenings(P[0].Hsml);
    sidm_local_configure(1,.2,.25,.1);sidm_local_initialize(&times,&cp);
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    for(int i=0;i<4;i++) visit.limits[i]=DBL_MAX;
    sidm_local_commit(&visit,&times,&cp);sidm_local_end(&visit);
    near(sidm_local_gravity_limit(0,&times)/budget,1);
    ActiveParticles active=init_empty_active_particles(PartManager);
    active.NumActiveGravity=PartManager->NumPart;
    PetaPM pm={0};pm.Asmth=1;pm.Nmesh=1;
    struct grav_accel_store stored={0};
    assert_int_equal(hierarchical_gravity_and_timesteps(&active,&pm,NULL,stored,
        &times,get_atime(times.Ti_Current),0,2,&cp,"."),0);
    assert_int_equal(times.mingravtimebin,target_bin);
    for(int i=0;i<4;i++) {
        assert_int_equal(P[i].TimeBinGravity,target_bin);
        near(P[i].Vel[0],(i&1)?-speed:speed);
    }
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
    sidm_local_configure(0,.2,.25,.1);
}
static int setup(void **state)
{
    (void)state;MPI_Comm_rank(MPI_COMM_WORLD,&rank);walltime_init(&clocks);
    cp.Omega0=1;cp.OmegaBaryon=.05;cp.HubbleParam=.7;cp.CMBTemperature=2.7255;
    init_cosmology(&cp,.1,get_unitsystem(3.085678e21,1.989e43,1e5));
    /* Pure matter reference excludes the native massless-neutrino background. */
    memset(cp.ONu.nu_degeneracies,0,sizeof(cp.ONu.nu_degeneracies));
    double output[]={.1,.2};set_sync_params_test(2,output);
    setup_sync_points(&cp,.1,.2,0,0);sidm_time_factors_init(&cp);
    return 0;
}
int main(int argc,char **argv)
{
    omp_set_dynamic(0);omp_set_num_threads(2);
    if(argc==2 && !strcmp(argv[1],"--expect-resolution-guard")) {
        const struct CMUnitTest guard[]={cmocka_unit_test(test_transport_resolution_guard)};
        return cmocka_run_group_tests_mpi(guard,setup,NULL);
    }
    const struct CMUnitTest tests[]={cmocka_unit_test(test_cosmological_factors),
        cmocka_unit_test(test_mixed_clocks_and_boost),
        cmocka_unit_test(test_transport_deadline_tightening),
        cmocka_unit_test(test_gravity_transport_shortening)};
    return cmocka_run_group_tests_mpi(tests,setup,NULL);
}
#else
int main(void) {puts("SKIP local SIDM: SIDM disabled");return 0;}
#endif
