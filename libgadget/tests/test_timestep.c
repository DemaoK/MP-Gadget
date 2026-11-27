/* Tests for the hierarchical gravity integrator: which acceleration sets the timestep and the kick of the longest bin.*/
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "stub.h"

#include <libgadget/utils/mymalloc.h>
#include <libgadget/utils/paramset.h>
#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/timefac.h>
#include <libgadget/timestep.h>
#include <libgadget/gravity.h>

#define NPART 3
#define BOXSIZE 1000.
/* Small enough that the gravity timestep of a particle exceeds the PM step at a = 0.2.*/
#define SMALLACC 1e-4

static struct ClockTable CT;
static Cosmology CP;
static const double V0[3] = {1e-3, 2e-3, 3e-3};

/* Three dark matter particles with distinct records along the axes x, y, z and a common velocity V0.*/
static void
make_particles(const int TimeBin)
{
    particle_alloc_memory(PartManager, BOXSIZE, NPART + 1);
    PartManager->NumPart = NPART;
    int i, k;
    for(i = 0; i < NPART; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].ID = i + 1;
        P[i].TimeBinGravity = TimeBin;
        for(k = 0; k < 3; k++) {
            P[i].Pos[k] = BOXSIZE / 4 * (i + 1);
            P[i].Vel[k] = V0[k];
            P[i].GravPM[k] = 0;
            P[i].FullTreeGravAccel[k] = (k == i) ? SMALLACC : 0;
        }
    }
}

static struct grav_accel_store
alloc_store(int64_t nstore)
{
    struct grav_accel_store store;
    store.nstore = nstore;
    store.GravAccel = (MyFloat (*) [3]) mymalloc2("GravAccel", nstore * sizeof(store.GravAccel[0]));
    memset(store.GravAccel, 0, nstore * sizeof(store.GravAccel[0]));
    return store;
}

static int
run_step(DriftKickTimes * times, struct grav_accel_store store)
{
    ActiveParticles act = init_empty_active_particles(PartManager);
    act.NumActiveGravity = act.NumActiveParticle;
    PetaPM pm = {0};
    pm.Asmth = 1.25;
    pm.Nmesh = 16;
    DomainDecomp ddecomp = {0};
    return hierarchical_gravity_and_timesteps(&act, &pm, &ddecomp, store, times, get_atime(times->Ti_Current), 0, 2, &CP, ".");
}

/* Largest bin active at Ti_Current that fits in the PM step: the longest bin of the step.*/
static int
longest_bin(const DriftKickTimes * times)
{
    int bin;
    for(bin = TIMEBINS; bin > 0; bin--)
        if(is_timebin_active(bin, times->Ti_Current) && dti_from_timebin(bin) <= times->PM_length)
            break;
    return bin;
}

/* Each particle's velocity changed by acc[i] times the kick factor of the longest bin.*/
static void
check_kick(const DriftKickTimes * times, const int bin, MyFloat (*acc)[3])
{
    const double gravkick = get_exact_gravkick_factor(&CP, times->Ti_kick[bin], times->Ti_kick[bin] + dti_from_timebin(bin)/2);
    assert_true(gravkick > 0);
    const double tol = 1e-8 * SMALLACC * gravkick;
    int i, k;
    for(i = 0; i < NPART; i++)
        for(k = 0; k < 3; k++)
            assert_true(fabs((P[i].Vel[k] - V0[k]) - acc[i][k] * gravkick) <= tol);
}

/* An output on a PM step sorted the particles after the force computation:
 * the stored array is in the reversed order of the particles. The kick follows each particle's own record.*/
static void
test_pm_kick_follows_record(void ** state)
{
    make_particles(0);
    DriftKickTimes times = init_driftkicktime(TIMEBASE);
    times.PM_length = TIMEBASE / 32;
    times.PM_start = times.Ti_Current - times.PM_length;
    assert_true(is_PM_timestep(&times));

    MyFloat records[NPART][3];
    struct grav_accel_store store = alloc_store(NPART);
    int i, k;
    for(i = 0; i < NPART; i++)
        for(k = 0; k < 3; k++) {
            records[i][k] = P[i].FullTreeGravAccel[k];
            store.GravAccel[NPART - 1 - i][k] = P[i].FullTreeGravAccel[k];
        }
    assert_int_equal(run_step(&times, store), 0);

    const int bin = longest_bin(&times);
    assert_int_equal(times.maxtimebin, bin);
    check_kick(&times, bin, records);
    myfree(P);
}

/* An output on a PM step removed a garbage particle with a large acceleration from index 1 after
 * the force computation, so each later particle sits one index below its entry in the stored array.
 * Every record puts its particle in the longest bin. Read from the array, particle 1 takes the garbage
 * particle's short timestep, and with three particles on a PM step the push-down then moves all of them
 * to that bin, so a stale read shows in every particle's bin without a tree for the shorter bins.*/
static void
test_pm_timestep_follows_record(void ** state)
{
    make_particles(0);
    DriftKickTimes times = init_driftkicktime(TIMEBASE);
    times.PM_length = TIMEBASE / 32;
    times.PM_start = times.Ti_Current - times.PM_length;
    assert_true(is_PM_timestep(&times));

    struct grav_accel_store store = alloc_store(NPART + 1);
    int k;
    for(k = 0; k < 3; k++) {
        store.GravAccel[0][k] = P[0].FullTreeGravAccel[k];
        store.GravAccel[1][k] = (k == 2) ? -10. : 0;
        store.GravAccel[2][k] = P[1].FullTreeGravAccel[k];
        store.GravAccel[3][k] = P[2].FullTreeGravAccel[k];
    }
    assert_int_equal(run_step(&times, store), 0);

    const int bin = longest_bin(&times);
    message(0, "Longest bin %d, step bin %d, particle bins %d %d %d\n", bin, times.maxtimebin, P[0].TimeBinGravity, P[1].TimeBinGravity, P[2].TimeBinGravity);
    assert_int_equal(times.maxtimebin, bin);
    int i;
    for(i = 0; i < NPART; i++)
        assert_int_equal(P[i].TimeBinGravity, bin);
    myfree(P);
}

/* On a step that is not a PM step the stored array holds this step's force and the record
 * is the last PM step's: the timestep and the kick follow the array.*/
static void
test_short_step_follows_store(void ** state)
{
    DriftKickTimes times = init_driftkicktime(TIMEBASE);
    times.PM_start = TIMEBASE;
    times.PM_length = TIMEBASE / 32;
    times.Ti_Current = times.PM_start + times.PM_length / 2;
    int b;
    for(b = 0; b <= TIMEBINS; b++)
        times.Ti_kick[b] = times.Ti_Current;
    assert_false(is_PM_timestep(&times));
    const int bin = longest_bin(&times);
    make_particles(bin);

    MyFloat accels[NPART][3];
    struct grav_accel_store store = alloc_store(NPART);
    int i, k;
    for(i = 0; i < NPART; i++)
        for(k = 0; k < 3; k++) {
            accels[i][k] = (k == (i + 1) % 3) ? SMALLACC : 0;
            store.GravAccel[i][k] = accels[i][k];
        }
    assert_int_equal(run_step(&times, store), 0);

    assert_int_equal(times.maxtimebin, bin);
    for(i = 0; i < NPART; i++)
        assert_int_equal(P[i].TimeBinGravity, bin);
    check_kick(&times, bin, accels);
    myfree(P);
}

static int
setup_timestep(void ** state)
{
    /* The parameter set lives on the temporary heap, which parameter_set_free resets.*/
    ParameterSet * ps = parameter_set_new();
    param_declare_double(ps, "ErrTolIntAccuracy", OPTIONAL, 0.02, NULL);
    param_declare_double(ps, "MaxGasVel", OPTIONAL, 3e5, NULL);
    param_declare_double(ps, "MaxSizeTimestep", OPTIONAL, 0.1, NULL);
    param_declare_double(ps, "MinSizeTimestep", OPTIONAL, 0, NULL);
    param_declare_int(ps, "ForceEqualTimesteps", OPTIONAL, 0, NULL);
    param_declare_double(ps, "MaxRMSDisplacementFac", OPTIONAL, 0.2, NULL);
    param_declare_double(ps, "CourantFac", OPTIONAL, 0.15, NULL);
    char content[1] = "";
    param_parse(ps, content);
    set_timestep_params(ps);
    parameter_set_free(ps);

    walltime_init(&CT);
    /* Sync points at a = 0.1, 0.2, 1: the step at ti = TIMEBASE is the output at a = 0.2.*/
    double outs[3] = {0.1, 0.2, 1.0};
    set_sync_params_test(3, outs);
    setup_sync_points(NULL, 0.1, 1.0, 0.0, 0);

    CP.MNu[0] = CP.MNu[1] = CP.MNu[2] = 0;
    CP.CMBTemperature = 2.7255;
    CP.HubbleParam = 0.7;
    CP.Omega0 = 0.3;
    CP.OmegaCDM = 0.255;
    CP.OmegaBaryon = 0.045;
    CP.OmegaLambda = 0.7;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.1, units);

    struct gravshort_tree_params treeacc = {0};
    treeacc.FractionalGravitySoftening = 1./30.;
    set_gravshort_treepar(treeacc);
    gravshort_set_softenings(30.);
    return 0;
}

static int
teardown_timestep(void ** state)
{
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_pm_kick_follows_record),
        cmocka_unit_test(test_pm_timestep_follows_record),
        cmocka_unit_test(test_short_step_follows_store),
    };
    return cmocka_run_group_tests_mpi(tests, setup_timestep, teardown_timestep);
}
