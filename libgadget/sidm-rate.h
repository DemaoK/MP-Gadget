#ifndef SIDM_RATE_H
#define SIDM_RATE_H

/* Full-support single cubic kernel and analytic velocity-dependent cross
 * section. Each source supplies its own directed half-rate; sequential
 * collision updates use the current pair-relative velocity. */
#include <float.h>
#include <math.h>

static inline double
sidm_rate_cubic_kernel(double r, double h)
{
    if(!isfinite(r) || !isfinite(h) || r < 0 || h <= 0)
        return NAN;
    if(r >= h)
        return 0;
    const double q = r/h;
    const double f = q < 0.5 ? 1 - 6*q*q*(1-q) : 2*(1-q)*(1-q)*(1-q);
    const double invh = 1/h;
    return (8/acos(-1.0))*f*invh*invh*invh;
}

/* Normalized viscosity cross-section divided by its zero-speed value.
 * y=(v/w)^2, or the actual regularized argument used by the caller.
 * The normalized integral representation is
 *   f(y) = 6 integral_0^1 t(1-t)/(1+y*t)^2 dt.
 * It proves positivity and monotonic decrease. The series avoids cancellation
 * at the origin; inverse powers avoid overflow in the large-y expression.
 */
static inline double
sidm_rate_viscosity_shape(double y)
{
    if(isnan(y) || y < 0)
        return NAN;
    if(isinf(y))
        return 0;
    if(y < 0.01) {
        /* Coefficient n is 6*(-1)^n*(n+1)/((n+2)*(n+3)). */
        double f = 6.0*9/(10*11);
        for(int n = 7; n >= 0; n--)
            f = 6.0*((n & 1) ? -1 : 1)*(n+1)/((n+2)*(n+3)) + y*f;
        return f;
    }
    if(y < 1)
        return 6*((2+y)*log1p(y)-2*y)/(y*y*y);
    const double invy = 1/y;
    return invy*(invy*(6*((1+2*invy)*log1p(y)-2)));
}

#endif
