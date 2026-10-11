/* Tests of the walltime clocks and of walltime_redirect_begin/end on a controlled clock. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <mpi.h>
#include <string.h>

#include <libgadget/walltime.h>
#include "stub.h"

/* The controlled clock: this definition replaces the MPI library's MPI_Wtime (the MPI profiling
 * interface allows it), so walltime.c runs unchanged. Dyadic steps keep every sum exact. */
static double Now = 100.;
double MPI_Wtime(void)
{
    return Now;
}

static struct ClockTable CT;

/* The summed time of the clocks whose names start with prefix. */
static double
clock_sum(const char * prefix)
{
    double t = 0;
    int i;
    for(i = 0; i < CT.N; i++)
        if(strncmp(CT.C[i].name, prefix, strlen(prefix)) == 0)
            t += CT.C[i].time;
    return t;
}

static void
test_walltime_redirect(void ** state)
{
    const double t0 = Now;
    walltime_init(&CT);
    /* a measure returns and books the interval since the last one */
    Now += 1;
    assert_true(walltime_measure("/Test/A") == 1);
    assert_true(clock_sum("/Test/A@") == 1);
    /* an ignored measure returns its interval and books nothing; its adds book it */
    Now += 2;
    assert_true(walltime_measure(WALLTIME_IGNORE) == 2);
    assert_true(clock_sum("/") == 1);
    walltime_add("/Test/B", 1.5);
    walltime_add("/Test/C", 0.5);
    assert_true(clock_sum("/Test/B@") == 1.5 && clock_sum("/Test/C@") == 0.5);
    /* the caller measures before the begin, so the redirect clock gets only the callee's time */
    Now += 0.5;
    walltime_measure("/Test/D");
    walltime_redirect_begin("/Test/R");
    Now += 0.25;
    assert_true(walltime_measure("/Tree/Build/Nodes") == 0.25);
    /* a nested begin keeps the outer destination */
    walltime_redirect_begin("/Test/Inner");
    Now += 0.75;
    assert_true(walltime_measure("/Misc") == 0.75);
    /* an ignored measure stays ignored under a redirect; its adds are redirected */
    Now += 1;
    assert_true(walltime_measure(WALLTIME_IGNORE) == 1);
    walltime_add("/Tree/Walk", 0.625);
    walltime_add("/Tree/Other", 0.375);
    /* the inner end books its interval to the outer clock and keeps redirecting */
    Now += 0.5;
    walltime_redirect_end();
    assert_true(clock_sum("/Test/R@") == 2.5);
    /* the final end books the tail */
    Now += 0.125;
    walltime_redirect_end();
    assert_true(clock_sum("/Test/R@") == 2.625);
    assert_true(clock_sum("/Test/Inner") == 0 && clock_sum("/Tree/") == 0 && clock_sum("/Misc") == 0);
    assert_true(clock_sum("/Test/D@") == 0.5);
    /* an unmatched end does nothing */
    walltime_redirect_end();
    assert_true(clock_sum("/Test/R@") == 2.625);
    /* afterwards /Tree/Build books to its own clock, the interval since the last end */
    Now += 2;
    assert_true(walltime_measure("/Tree/Build/Nodes") == 2);
    assert_true(clock_sum("/Tree/Build/Nodes@") == 2 && clock_sum("/Test/R@") == 2.625);
    /* every interval booked once: the total equals the elapsed controlled time */
    message(0, "walltime: clocks total %g s, elapsed %g s\n", clock_sum("/"), Now - t0);
    assert_true(clock_sum("/") == Now - t0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_walltime_redirect),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
