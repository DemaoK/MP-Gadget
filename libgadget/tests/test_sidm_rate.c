/* Standalone: cc -std=c99 -O2 -Wall -Wextra test_sidm_rate.c -lm. */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "../sidm-rate.h"

static void close_enough(double x, double expected, double tolerance)
{
    if(!isfinite(x) || fabs(x-expected) > tolerance*fmax(1, fabs(expected))) {
        fprintf(stderr, "actual %.17g expected %.17g tolerance %.3g\n", x, expected, tolerance);
        assert(0);
    }
}

static double kernel_integral(double h, double hi, double hj)
{
    const int n = 8192;
    double sum = 0;
    for(int i = 0; i <= n; i++) {
        const double r = h*i/n;
        const double w = 0.5*sidm_rate_cubic_kernel(r, hi)+0.5*sidm_rate_cubic_kernel(r, hj);
        sum += (i == 0 || i == n ? 1 : (i & 1) ? 4 : 2)*4*acos(-1.0)*r*r*w;
    }
    return sum*h/(3*n);
}

static void test_normalized_kernel_and_pair_counting(void)
{
    const double radii[]={.01,.7,12};
    for(unsigned int i=0;i<sizeof(radii)/sizeof(*radii);i++) {
        const double h=radii[i];
        close_enough(kernel_integral(h,h,h),1,2e-13);
        close_enough(sidm_rate_cubic_kernel(0,h)*h*h*h,8/acos(-1.0),2e-15);
        assert(sidm_rate_cubic_kernel(h,h)==0);
    }
    /* Both directed halves integrate to the uniform-bath rate, including
     * unequal support radii. This requires no overlap kernel or rate bound. */
    close_enough(kernel_integral(1,.25,1),1,2e-12);
    assert(isnan(sidm_rate_cubic_kernel(-1,1)));
}

/* Independent integral of the angular kernel, rather than its closed form. */
static double shape_by_quadrature(double y)
{
    const int n = 65536;
    long double sum = 0;
    for(int i = 0; i <= n; i++) {
        const long double t = (long double)i/n, d = 1+y*t;
        sum += (i == 0 || i == n ? 1 : (i & 1) ? 4 : 2)*6*t*(1-t)/(d*d);
    }
    return sum/(3*n);
}

static void test_stable_viscosity_shape(void)
{
    const double y[] = {0, 1e-14, 1e-8, 9.999e-3, 0.01, 0.1, 0.5, 1, 4, 20, 100};
    for(unsigned int i = 0; i < sizeof(y)/sizeof(*y); i++)
        close_enough(sidm_rate_viscosity_shape(y[i]), shape_by_quadrature(y[i]), 2e-12);
    assert(sidm_rate_viscosity_shape(0) == 1);
    assert(sidm_rate_viscosity_shape(INFINITY) == 0);
    assert(isnan(sidm_rate_viscosity_shape(-1)));
    double last = 1;
    for(int i = -120; i <= 120; i++) {
        const double argument = pow(10.0, i/10.0), f = sidm_rate_viscosity_shape(argument);
        assert(isfinite(f) && f >= 0 && f <= last*(1+1e-12));
        last = f;
    }
    assert(sidm_rate_viscosity_shape(DBL_MAX) == 0);
    /* Independently bracket the peak; w/sqrt(3) belongs to another model. */
    double lo = 0.4, hi = 1.5;
    for(int i = 0; i < 100; i++) {
        const double a = lo+(hi-lo)/3, b = hi-(hi-lo)/3;
        if(a*sidm_rate_viscosity_shape(a*a) < b*sidm_rate_viscosity_shape(b*b)) lo = a;
        else hi = b;
    }
    const double peak = 0.5*(lo+hi);
    close_enough(peak, 0.8996219, 2e-6);
    close_enough(peak*sidm_rate_viscosity_shape(peak*peak), 0.4800938931614, 3e-13);
}

int main(void)
{
    test_normalized_kernel_and_pair_counting();
    test_stable_viscosity_shape();
    puts("PASS SIDM rate helpers");
    return 0;
}
