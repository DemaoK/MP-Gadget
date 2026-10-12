#include <mpi.h>
#include <math.h>
#include <gsl/gsl_integration.h>

#include "physconst.h"
#include "timefac.h"
#include "timebinmgr.h"
#include "utils/endrun.h"
#include <float.h>

#define WORKSIZE 10000

/* Integrand for the drift table*/
static double drift_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
  double h = hubble_function(CP, a);
  return 1 / (h * a * a * a);
}

/* Integrand for the gravkick table*/
static double gravkick_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
  double h = hubble_function(CP, a);

  return 1 / (h * a * a);
}

/* Integrand for the hydrokick table.
 * Note this is the same function as drift.*/
static double hydrokick_integ(double a, void *param)
{
  double h;

  Cosmology * CP = (Cosmology *) param;
  h = hubble_function(CP, a);

  return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
}

/*Do the integral required to get a factor.*/
static double get_exact_factor(Cosmology * CP, inttime_t t0, inttime_t t1, double (*factor) (double, void *))
{
    double result, abserr;
    if(t0 == t1)
        return 0;
    double a0 = exp(loga_from_ti(t0));
    double a1 = exp(loga_from_ti(t1));
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = factor;
    F.params = CP;
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    return result;
}

/*Get the exact drift factor*/
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &drift_integ);
}

/*Get the exact drift factor*/
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &gravkick_integ);
}

double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &hydrokick_integ);
}

/* Integrand for comoving distance */
static double comoving_distance_integ(double a, void *param)
{
    Cosmology *CP = (Cosmology *) param;
    double h = hubble_function(CP, a);
    return 1. / (h * a * a);
}

/* Function to compute the comoving distance between two scale factors */
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
{
    double result, abserr;
    gsl_function F;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(WORKSIZE);

    F.function = comoving_distance_integ;
    F.params = CP;

    // Using GSL to perform the integration
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);

    return (LIGHTCGS/UnitVelocity_in_cm_per_s) * result;
}

/* Quartic integrands on a fixed log(a) grid. A local polynomial integral
 * preserves short integer intervals; backward long-double sums keep the
 * large early-time exposure out of differences between late-time cells.
 * The storage and query cost are independent of the particle population. */
#define SIDM_FACTOR_CELLS 8192
static struct {
    const Cosmology *cosmology;
    double begin, end, width;
    inttime_t final_ti;
    double polynomial[2][SIDM_FACTOR_CELLS][5];
    long double suffix[2][SIDM_FACTOR_CELLS+1];
} SIDMFactors;

static double sidm_cell_integral(const double c[5], double z, double dz, double width)
{
    const double mid=z+.5*dz, dz2=dz*dz;
    const double value=c[0]+mid*(c[1]+mid*(c[2]+mid*(c[3]+mid*c[4])));
    return width*(value+dz2*(c[2]+3*mid*c[3]+6*mid*mid*c[4])/12+
        c[4]*dz2*dz2/80);
}

void sidm_time_factors_init(const Cosmology *CP)
{
    if(!(CP->Hubble>0) || !isfinite(CP->Hubble))
        endrun(1,"SIDM requires a positive finite Hubble conversion.\n");
    SIDMFactors.cosmology=NULL;
    SIDMFactors.begin=loga_from_ti(0);
    SIDMFactors.end=SIDMFactors.begin;
    SIDMFactors.final_ti=0;
    for(SyncPoint *point=find_next_sync_point(0); point;
        point=find_next_sync_point(point->ti)) {
        SIDMFactors.end=point->loga;
        SIDMFactors.final_ti=point->ti;
    }
    SIDMFactors.width=(SIDMFactors.end-SIDMFactors.begin)/SIDM_FACTOR_CELLS;
    if(!isfinite(SIDMFactors.width) || SIDMFactors.width<=0)
        endrun(1,"SIDM requires an initialized increasing timeline.\n");
    for(int cell=0;cell<SIDM_FACTOR_CELLS;cell++) {
        double samples[2][5];
        for(int k=0;k<5;k++) {
            const double x=SIDMFactors.begin+(cell+.25*k)*SIDMFactors.width;
            const double a=exp(x), h=hubble_function(CP,a);
            samples[0][k]=1/(a*a*h);
            samples[1][k]=samples[0][k]/(a*a);
            if(!(samples[0][k]>0) || !isfinite(samples[0][k]) ||
               !(samples[1][k]>0) || !isfinite(samples[1][k]))
                endrun(1,"Nonfinite SIDM cosmological time integrand.\n");
        }
        for(int kind=0;kind<2;kind++) {
            const double *f=samples[kind];
            double *c=SIDMFactors.polynomial[kind][cell];
            const double even_half=.5*(f[1]+f[3])-f[2], even_end=.5*(f[0]+f[4])-f[2];
            const double odd_half=.5*(f[3]-f[1]), odd_end=.5*(f[4]-f[0]);
            c[0]=f[2];
            c[4]=(4.0/3)*(even_end-4*even_half); c[2]=even_end-c[4];
            c[3]=(4.0/3)*(odd_end-2*odd_half); c[1]=odd_end-c[3];
        }
    }
    for(int kind=0;kind<2;kind++) {
        SIDMFactors.suffix[kind][SIDM_FACTOR_CELLS]=0;
        for(int cell=SIDM_FACTOR_CELLS-1;cell>=0;cell--)
            SIDMFactors.suffix[kind][cell]=SIDMFactors.suffix[kind][cell+1]+
                sidm_cell_integral(SIDMFactors.polynomial[kind][cell],-1,2,SIDMFactors.width);
    }
    SIDMFactors.cosmology=CP;
}

/* Use integer differences within each native timeline segment. Subtracting
 * loga_from_ti endpoints would lose the shortest representable intervals. */
static double sidm_interval_width(inttime_t ti0,inttime_t ti1)
{
    double result=0;
    while(ti0<ti1) {
        const inttime_t edge=((ti0>>TIMEBINS)+1)*TIMEBASE;
        const inttime_t stop=ti1<edge?ti1:edge;
        result+=dloga_from_dti(stop-ti0,ti0);
        ti0=stop;
    }
    return result;
}

static double sidm_time_factor(const Cosmology *CP,inttime_t ti0,inttime_t ti1,int kind)
{
    if(ti0==ti1) return 0;
    if(SIDMFactors.cosmology!=CP || ti0<0 || ti1<ti0 || ti1>SIDMFactors.final_ti)
        endrun(1,"Invalid or uninitialized SIDM cosmological interval [%ld,%ld].\n",ti0,ti1);
    const double x=loga_from_ti(ti0), width=sidm_interval_width(ti0,ti1);
    double coordinate=(x-SIDMFactors.begin)/SIDMFactors.width;
    int first=(int)floor(coordinate);
    if(first<0) first=0;
    if(first>=SIDM_FACTOR_CELLS) first=SIDM_FACTOR_CELLS-1;
    const double fraction=fmin(1,fmax(0,coordinate-first));
    const double first_width=(1-fraction)*SIDMFactors.width;
    if(width<=first_width)
        return sidm_cell_integral(SIDMFactors.polynomial[kind][first],2*fraction-1,
            2*width/SIDMFactors.width,width);
    double result=sidm_cell_integral(SIDMFactors.polynomial[kind][first],2*fraction-1,
        2*(1-fraction),first_width);
    const double remaining=(width-first_width)/SIDMFactors.width;
    int whole=(int)floor(remaining);
    if(whole>SIDM_FACTOR_CELLS-first-1) whole=SIDM_FACTOR_CELLS-first-1;
    const int last=first+1+whole;
    result+=(double)(SIDMFactors.suffix[kind][first+1]-SIDMFactors.suffix[kind][last]);
    if(last<SIDM_FACTOR_CELLS) {
        const double tail=fmax(0,fmin(1,remaining-whole));
        result+=sidm_cell_integral(SIDMFactors.polynomial[kind][last],-1,2*tail,
            tail*SIDMFactors.width);
    }
    if(!(result>0) || !isfinite(result))
        endrun(1,"Invalid SIDM cosmological time factor.\n");
    return result;
}

double sidm_drift_factor(const Cosmology *CP,inttime_t ti0,inttime_t ti1)
{ return sidm_time_factor(CP,ti0,ti1,0); }
double sidm_scatter_factor(const Cosmology *CP,inttime_t ti0,inttime_t ti1)
{ return sidm_time_factor(CP,ti0,ti1,1); }
