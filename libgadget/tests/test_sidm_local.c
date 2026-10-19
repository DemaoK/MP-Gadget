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
static void allocate_particles(int n)
{
    particle_alloc_memory(PartManager,1e9,n);
    PartManager->NumPart=n;
    memset(P,0,(size_t)n*sizeof(*P));
    for(int i=0;i<n;i++) {
        P[i].ID=(uint64_t)n*rank+i+1;P[i].Type=1;P[i].Mass=1;
        P[i].Hsml=1;P[i].TimeBinGravity=TIMEBINS-2;
    }
}
static void release_particles(void)
{
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
    sidm_local_configure(0,.2,.1);
}
static DriftKickTimes initialize_clocks(void)
{
    sidm_local_configure(1,.2,.1);
    DriftKickTimes times=init_driftkicktime(0);
    times.PM_length=TIMEBASE;times.mintimebin=times.mingravtimebin=TIMEBINS-2;
    sidm_local_initialize(&times,&cp);
    return times;
}
static void unlimited(struct sidm_local_visit *visit)
{
    for(int a=0;a<visit->sources.NumActiveParticle;a++)
        visit->limits[a]=visit->motion_limits[a]=DBL_MAX;
}
static void recount(void)
{
    int64_t counts[TIMEBINS+1]={0};
    for(int i=0;i<PartManager->NumPart;i++)
        if(P[i].Type==1 && !P[i].IsGarbage && !P[i].Swallowed)
            counts[P[i].TimeBinSIDM]++;
    sidm_local_set_bin_counts(counts);
}
/* Separate canonical drift and scattering integrals select independent
 * source deadlines. Unrelated velocities cannot set a global motion clock. */
static void clocks_for_boost(double boost,double distant_speed,inttime_t deadline[3])
{
    allocate_particles(4);
    P[0].Vel[0]=boost+.5;P[1].Vel[0]=boost-.5;
    P[2].Vel[0]=distant_speed;P[3].Vel[0]=1e30;P[3].Type=0;
    DriftKickTimes times=initialize_clocks();
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    assert_int_equal(visit.sources.NumActiveSIDM,3);
    unlimited(&visit);
    const double rate=1.1*sidm_scatter_factor(&cp,0,TIMEBASE/16);
    const double motion=1.1*sidm_drift_factor(&cp,0,TIMEBASE/32);
    visit.limits[0]=rate;
    const double relative=fabs(P[0].Vel[0]-P[1].Vel[0]);
    P[1].Hsml=relative*motion/.2;
    visit.motion_limits[1]=P[1].Hsml/relative;
    for(int a=0;a<visit.sources.NumActiveParticle;a++)
        near(visit.intervals[a],0);
    sidm_local_commit(&visit,&times,&cp);
    for(int i=0;i<3;i++) {
        deadline[i]=sidm_source_deadline(&P[i]);
        assert_true(is_timebin_active(P[i].TimeBinSIDM,0));
        assert_int_equal(deadline[i],dti_from_timebin(P[i].TimeBinSIDM));
    }
    assert_int_equal(deadline[0],TIMEBASE/16);
    assert_int_equal(deadline[1],TIMEBASE/32);
    assert_int_equal(deadline[2],TIMEBASE/8); /* max_dloga only */
    assert_true(sidm_scatter_factor(&cp,0,deadline[0])<=rate);
    assert_true(sidm_drift_factor(&cp,0,deadline[1])<=motion);
    assert_true(dloga_from_dti(deadline[2],0)<=.1);
    assert_int_equal(sidm_local_next_event(&times,0),deadline[1]);
    /* Changing a distant velocity does not retroactively shorten clocks. */
    P[2].Vel[0]*=1000;
    assert_int_equal(sidm_local_next_event(&times,0),deadline[1]);
    release_particles();
}
static void test_mixed_clocks_and_boost(void **state)
{
    (void)state;
    inttime_t first[3],boosted[3];
    clocks_for_boost(0,1e6,first);
    clocks_for_boost(1048576,1e20,boosted);
    for(int i=0;i<3;i++) assert_int_equal(first[i],boosted[i]);
}
static void test_pm_due_and_checkpoint_flush(void **state)
{
    (void)state;
    allocate_particles(3);P[2].Type=0;
    DriftKickTimes times=initialize_clocks();
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    unlimited(&visit);
    visit.limits[0]=1.1*sidm_scatter_factor(&cp,0,2);
    visit.motion_limits[1]=1.1*sidm_drift_factor(&cp,0,8)/.2;
    sidm_local_commit(&visit,&times,&cp);
    assert_int_equal(sidm_source_deadline(&P[0]),2);
    assert_int_equal(sidm_source_deadline(&P[1]),8);
    times.Ti_Current=times.PM_length=2;
    assert_true(is_PM_timestep(&times));
    /* PM's native list contains every particle; ordinary SIDM visits still
     * consume only due clocks and preserve an inactive receiver's exposure. */
    ActiveParticles active=init_empty_active_particles(PartManager);
    active.NumActiveSIDM=1;
    visit=sidm_local_begin_active(&times,&cp,0,&active);
    assert_int_equal(visit.sources.NumActiveParticle,1);
    assert_int_equal(visit.sources.ActiveParticle[0],0);
    near(visit.intervals[0],scatter_exact(.1,get_atime(2)));
    unlimited(&visit);
    sidm_local_commit(&visit,&times,&cp);
    assert_int_equal(P[1].TiSIDMLast,0);
    assert_int_equal(sidm_source_deadline(&P[1]),8);
    /* An explicit checkpoint also consumes the undued source at this PM. */
    times.PM_start=2;times.PM_length=2;times.Ti_Current=times.PM_kick=4;
    times.Ti_kick[TIMEBINS-2]=4;
    visit=sidm_local_begin_active(&times,&cp,1,&active);
    assert_int_equal(visit.sources.NumActiveParticle,2);
    near(visit.intervals[0],scatter_exact(get_atime(2),get_atime(4)));
    near(visit.intervals[1],scatter_exact(.1,get_atime(4)));
    unlimited(&visit);sidm_local_commit(&visit,&times,&cp);
    sidm_local_assert_flushed(&times);
    for(int i=0;i<2;i++) {
        assert_int_equal(P[i].TiSIDMLast,4);
        assert_int_equal(sidm_source_deadline(&P[i]),8);
        assert_true(is_timebin_active(P[i].TimeBinSIDM,4));
    }
    times.PM_start=4;times.PM_length=TIMEBASE-4;
    assert_int_equal(sidm_local_next_event(&times,0),8);
    /* Terminal flushing has no pending source bin beyond the timeline. */
    for(int i=0;i<2;i++) {P[i].TiSIDMLast=TIMEBASE/2;P[i].TimeBinSIDM=TIMEBINS-1;}
    recount();
    times.Ti_Current=times.PM_kick=TIMEBASE;
    times.Ti_kick[TIMEBINS-2]=TIMEBASE;
    visit=sidm_local_begin_active(&times,&cp,1,NULL);
    unlimited(&visit);sidm_local_commit(&visit,&times,&cp);
    sidm_local_assert_flushed(&times);
    for(int i=0;i<2;i++) assert_int_equal(P[i].TimeBinSIDM,0);
    release_particles();
}
static void test_threaded_source_selection_and_commit(void **state)
{
    (void)state;
    enum { n=8192 };
    struct { inttime_t last,next; unsigned char bin; } serial[n];
    double exposure[2][n];
    int indices[n];
    inttime_t next_event[2];
    const inttime_t now=TIMEBASE/64;
    for(int threads=1;threads<=2;threads++) {
        omp_set_num_threads(threads);
        allocate_particles(n);
        for(int i=0;i<n;i++) {
            P[i].Type=i%8==0?0:1;
            P[i].IsGarbage=i%8==1;
        }
        DriftKickTimes times=initialize_clocks();
        times.Ti_Current=now;
        ActiveParticles active=init_empty_active_particles(PartManager);
        active.ActiveParticle=indices;
        int at=0;
        /* Reorder non-DM/garbage entries while preserving the native ordered
         * live DM subsequence, so candidate slots differ from particle indices. */
        for(int i=n-1;i>=0;i--) if(i%8<2) indices[at++]=i;
        for(int i=0;i<n;i++) if(i%8>=2) {
            indices[at++]=i;
            const int due=i%17==0;
            P[i].TimeBinSIDM=TIMEBINS-(due?6:5);
            active.NumActiveSIDM+=due;
        }
        recount();
        for(int flush=0;flush<=1;flush++) {
            struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,
                flush,flush?NULL:&active);
            const int expected=flush?3*n/4:active.NumActiveSIDM;
            assert_int_equal(visit.sources.NumActiveParticle,expected);
            unlimited(&visit);
            int a=0;
            for(int i=0;i<n;i++) if(i%8>=2 && (flush || i%17==0)) {
                assert_int_equal(visit.sources.ActiveParticle[a],i);
                const double interval=sidm_scatter_factor(&cp,P[i].TiSIDMLast,now);
                assert_true(visit.intervals[a]==interval);
                if(threads==1) exposure[flush][i]=visit.intervals[a];
                else assert_true(visit.intervals[a]==exposure[flush][i]);
                if(flush && i%3==0)
                    visit.limits[a]=1.1*sidm_scatter_factor(&cp,now,now+now/2);
                if(flush?i%3==1:i%2==1)
                    visit.motion_limits[a]=1.1*sidm_drift_factor(&cp,now,
                        now+(flush?now/4:now/2))/.2;
                a++;
            }
            sidm_local_commit(&visit,&times,&cp);
            for(int i=0;i<n;i++) if(i%8>=2) {
                if(flush || i%17==0) {
                    const inttime_t step=flush?(i%3==0?now/2:i%3==1?now/4:now):
                        (i%2?now/2:now);
                    assert_int_equal(P[i].TiSIDMLast,now);
                    assert_int_equal(sidm_source_deadline(&P[i]),now+step);
                    assert_int_equal(dti_from_timebin(P[i].TimeBinSIDM),step);
                } else {
                    assert_int_equal(P[i].TiSIDMLast,0);
                    assert_int_equal(sidm_source_deadline(&P[i]),2*now);
                }
            }
            const inttime_t next=sidm_local_next_event(&times,0);
            assert_int_equal(next,now+(flush?now/4:now/2));
            if(threads==1) next_event[flush]=next;
            else assert_int_equal(next,next_event[flush]);
        }
        for(int i=0;i<n;i++) {
            if(threads==1) {
                serial[i].last=P[i].TiSIDMLast;serial[i].next=sidm_source_deadline(&P[i]);
                serial[i].bin=P[i].TimeBinSIDM;
            } else {
                assert_int_equal(P[i].TiSIDMLast,serial[i].last);
                assert_int_equal(sidm_source_deadline(&P[i]),serial[i].next);
                assert_int_equal(P[i].TimeBinSIDM,serial[i].bin);
            }
        }
        release_particles();
    }
}
static void test_local_motion_resolution_guard(void **state)
{
    (void)state;
    allocate_particles(1);
    DriftKickTimes times=initialize_clocks();
    struct sidm_local_visit visit=sidm_local_begin_active(&times,&cp,1,NULL);
    unlimited(&visit);
    visit.motion_limits[0]=.5*sidm_drift_factor(&cp,0,2)/.2;
    sidm_local_commit(&visit,&times,&cp);
    fail_msg("Expected the local SIDM Courant resolution guard to terminate");
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
        const struct CMUnitTest guard[]={cmocka_unit_test(test_local_motion_resolution_guard)};
        return cmocka_run_group_tests_mpi(guard,setup,NULL);
    }
    const struct CMUnitTest tests[]={cmocka_unit_test(test_cosmological_factors),
        cmocka_unit_test(test_mixed_clocks_and_boost),
        cmocka_unit_test(test_pm_due_and_checkpoint_flush),
        cmocka_unit_test(test_threaded_source_selection_and_commit)};
    return cmocka_run_group_tests_mpi(tests,setup,NULL);
}
#else
int main(void) {puts("SKIP local SIDM: SIDM disabled");return 0;}
#endif
