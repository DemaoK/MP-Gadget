#include "sidm-local.h"
#include "gravity.h"
#include "timefac.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

static struct {
    int enabled;
    double transport, max_dloga;
    double bulk[3], max_speed; /* Common canonical-velocity frame and speed bound. */
    inttime_t commit_epoch;
    double travel; /* Cumulative comoving separation bound. */
    inttime_t travel_epoch, forward_epoch;
    int forward_maxbin;
    double forward_drift[TIMEBINS+1], forward_scatter[TIMEBINS+1];
    double forward_dloga[TIMEBINS+1];
#ifdef SIDMDEBUG
    /* Rank-local counters, reduced only at PM flushes. Diagnostic bounds are
     * unrounded and measured before the next opening kick. */
    uint64_t deadline_scans, deadline_tightened;
    uint64_t source_rate, source_transport, source_maxstep;
    uint64_t gravity_checked, gravity_transport;
    double min_rate, min_transport;
#endif
} Local;

void sidm_local_configure(int enabled, double transport, double max_dloga)
{
    memset(&Local, 0, sizeof(Local));
    Local.enabled=enabled;
    Local.transport=transport;
    Local.max_dloga=max_dloga;
#ifdef SIDMDEBUG
    Local.min_rate=Local.min_transport=DBL_MAX;
#endif
    if(enabled && (!(transport>0 && transport<=.25) ||
       !isfinite(max_dloga) || max_dloga<=0))
        endrun(1,"Invalid local SIDM transport fraction or maximum timestep.\n");
#ifndef SIDM
    if(enabled) endrun(1,"Local SIDM needs an SIDM build.\n");
#endif
}
int sidm_local_enabled(void) { return Local.enabled; }
#ifdef SIDMDEBUG
void sidm_local_record_gravity_limits(uint64_t checked, uint64_t shortened)
{
    Local.gravity_checked+=checked;
    Local.gravity_transport+=shortened;
}
#endif

#ifdef SIDM
static int live_dm(int i) { return P[i].Type==1 && !P[i].IsGarbage && !P[i].Swallowed; }
static double bulk_speed2(int i)
{
    double norm=0;
    for(int d=0;d<3;d++) norm+=(P[i].Vel[d]-Local.bulk[d])*(P[i].Vel[d]-Local.bulk[d]);
    return norm;
}

/* Forward candidates share an event epoch and are reused for every source.
 * The aligned hierarchy also prevents a source deadline crossing an output. */
static void prepare_forward(inttime_t now, Cosmology *CP)
{
    if(Local.forward_epoch==now) return;
    Local.forward_epoch=now;
    Local.forward_maxbin=0;
    SyncPoint *next=find_next_sync_point(now);
    if(!next) return;
    for(int bin=1;bin<=TIMEBINS;bin++) {
        const inttime_t dti=dti_from_timebin(bin);
        if(!is_timebin_active(bin,now) || dti>next->ti-now) break;
        Local.forward_drift[bin]=sidm_drift_factor(CP,now,now+dti);
        Local.forward_scatter[bin]=sidm_scatter_factor(CP,now,now+dti);
        Local.forward_dloga[bin]=dloga_from_dti(dti,now);
        Local.forward_maxbin=bin;
    }
}

/* The arrays increase with bin. A source lookup takes at most six comparisons
 * regardless of how many integer bins fit below its current deadline. */
static int bounded_bin(double limit,const double *factors,int maxbin)
{
    int low=0,high=maxbin+1;
    while(high-low>1) {
        const int middle=low+(high-low)/2;
        if(factors[middle]<=limit) low=middle;
        else high=middle;
    }
    return low;
}
/* Uses the forward table already prepared for this commit epoch. */
static int interval_bin(double exposure)
{
    int bin=bounded_bin(exposure,Local.forward_scatter,Local.forward_maxbin);
    bin=bounded_bin(Local.max_dloga,Local.forward_dloga,bin);
    if(bin<1) endrun(1,"SIDM source interval is below representable time resolution.\n");
    return bin;
}

void sidm_local_initialize(const DriftKickTimes *times,Cosmology *CP)
{
    if(!Local.enabled) return;
    if(sizeof(P[0].Vel[0])!=sizeof(double)) endrun(1,"Local SIDM requires double velocities.\n");
    sidm_time_factors_init(CP);
    Local.commit_epoch=Local.forward_epoch=-1;
    Local.travel=0; Local.travel_epoch=times->Ti_Current;
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i)) {
        P[i].TiSIDMLast=P[i].TiSIDMNext=times->Ti_Current;
        P[i].TimeBinSIDM=0;
        P[i].SIDMTravelLast=0;
    }
}

inttime_t sidm_local_next_event(const DriftKickTimes *times, Cosmology *CP, int first)
{
    if(first) return times->Ti_Current;
    const inttime_t now=times->Ti_Current, step=dti_from_timebin(times->mintimebin);
    if(step<2) endrun(1,"Uninitialized gravity clock in local SIDM scheduler.\n");
    inttime_t next=find_next_kick(now,times->mintimebin);
    const inttime_t pm=times->PM_start+times->PM_length;
    if(pm<=now) endrun(1,"Local SIDM scheduler found an expired PM deadline.\n");
    if(pm<next) next=pm;
    const double tol=128*DBL_EPSILON*fmax(1,fabs(Local.travel));
    prepare_forward(now,CP);
    const double minimum=Local.max_speed>0 ? Local.forward_drift[1]:0;
#ifdef SIDMDEBUG
    uint64_t scans=0,tightened=0;
#pragma omp parallel for reduction(min:next) reduction(+:scans,tightened)
#else
#pragma omp parallel for reduction(min:next)
#endif
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i)) {
#ifdef SIDMDEBUG
        scans++;
#endif
        if(P[i].TiSIDMNext<=now || P[i].TiSIDMLast>now)
            endrun(1,"Expired or invalid SIDM source clock for ID %lu.\n",P[i].ID);
        if(Local.max_speed>0) {
            const double used=Local.travel-P[i].SIDMTravelLast;
            const double remaining=Local.transport*P[i].Hsml-used;
            if(!isfinite(used) || used<0 || !isfinite(remaining))
                endrun(1,"Invalid local SIDM displacement ledger for ID %lu.\n",P[i].ID);
            if(remaining+tol<2*Local.max_speed*minimum)
                endrun(1,"SIDM transport budget exhausted below representable time resolution for ID %lu.\n",P[i].ID);
            const double dt=fmax(minimum,remaining/(2*Local.max_speed));
            /* Preserve an already-safe absolute deadline. Quantizing a fresh
             * interval at every fine event would align all distant sources to
             * that event's short bin, collapsing the local hierarchy. */
            if(sidm_drift_factor(CP,now,P[i].TiSIDMNext)>dt) {
                const int bin=bounded_bin(dt,Local.forward_drift,Local.forward_maxbin);
                if(bin<1) endrun(1,"SIDM transport interval is below representable time resolution.\n");
                inttime_t motion=now+dti_from_timebin(bin);
                if(motion<P[i].TiSIDMNext) {
                    P[i].TiSIDMNext=motion;
                    P[i].TimeBinSIDM=bin;
#ifdef SIDMDEBUG
                    tightened++;
#endif
                }
            }
        }
        if(P[i].TiSIDMNext<next) next=P[i].TiSIDMNext;
    }
#ifdef SIDMDEBUG
    Local.deadline_scans+=scans; Local.deadline_tightened+=tightened;
#endif
    MPI_Allreduce(MPI_IN_PLACE,&next,1,MPI_INT64_T,MPI_MIN,MPI_COMM_WORLD);
    return next;
}

void sidm_local_kick_phase(DriftKickTimes *times, int opening)
{
    /* Empty bins have a well-defined phase without incrementing any inactive
     * occupied bin. A subsequent gravity step may enter an aligned empty bin. */
    for(int b=0;b<=TIMEBINS;b++) if(is_timebin_active(b,times->Ti_Current))
        times->Ti_kick[b]=times->Ti_Current+(opening?dti_from_timebin(b)/2:0);
}

struct sidm_local_visit sidm_local_begin_active(const DriftKickTimes *times, Cosmology *CP,
    int flush, const ActiveParticles *active)
{
    Local.travel+=2*Local.max_speed*sidm_drift_factor(CP,Local.travel_epoch,times->Ti_Current);
    if(!isfinite(Local.travel) || Local.travel<0)
        endrun(1,"Invalid accumulated local SIDM displacement.\n");
    Local.travel_epoch=times->Ti_Current;
    const int n=PartManager->NumPart;
    struct sidm_local_visit out={0};
    out.sources.ActiveParticle=mymalloc("SIDM active sources",((size_t)n+1)*sizeof(int));
    out.intervals=mymalloc("SIDM intervals",((size_t)n+1)*sizeof(double));
    out.limits=mymalloc("SIDM limits",((size_t)n+1)*sizeof(double));
#ifdef SIDMDEBUG
    out.old_support=mymalloc("SIDM previous support",((size_t)n+1)*sizeof(double));
#endif
    out.sources.Particles=P;
    /* The native split-gravity list includes SIDM-only sources. PM flushes
     * inspect every DM clock regardless of the native activity list. */
    const int indexed=active && !flush;
#ifdef SIDMDEBUG
    if(indexed && (active->NumActiveParticle<0 || active->NumActiveParticle>n))
        endrun(1,"Invalid local SIDM active list.\n");
    int previous=-1;
#endif
    const int candidates=indexed?active->NumActiveParticle:n;
    const double tol=128*DBL_EPSILON*fmax(1,fabs(Local.travel));
    for(int a=0;a<candidates;a++) {
        const int i=indexed && active->ActiveParticle?active->ActiveParticle[a]:a;
#ifdef SIDMDEBUG
        if(i<0 || i>=n) endrun(1,"Invalid local SIDM active particle index.\n");
#endif
        if(!live_dm(i)) continue;
#ifdef SIDMDEBUG
        if(i<=previous) endrun(1,"Local SIDM requires ordered unique DM indices.\n");
        previous=i;
#endif
        if(P[i].TiSIDMLast>times->Ti_Current || P[i].TiSIDMNext<times->Ti_Current)
            endrun(1,"SIDM source interval missed for ID %lu.\n",P[i].ID);
        if(!flush && P[i].TiSIDMNext!=times->Ti_Current) continue;
        out.sources.ActiveParticle[out.sources.NumActiveParticle++]=i;
        out.intervals[i]=sidm_scatter_factor(CP,P[i].TiSIDMLast,times->Ti_Current);
#ifdef SIDMDEBUG
        out.old_support[i]=P[i].Hsml;
#endif
        if(out.intervals[i]>0 && P[i].Hsml>0) {
            const double used=Local.travel-P[i].SIDMTravelLast;
            if(!isfinite(used) || used<0)
                endrun(1,"Invalid local SIDM source displacement for ID %lu.\n",P[i].ID);
            if(used>Local.transport*P[i].Hsml+tol)
                endrun(1,"SIDM source exceeded relative-displacement budget for ID %lu.\n",P[i].ID);
        }
    }
#ifdef SIDMDEBUG
    if(indexed && out.sources.NumActiveParticle!=active->NumActiveSIDM)
        endrun(1,"Local SIDM source count disagrees with native activity.\n");
#endif
    out.sources.NumActiveSIDM=out.sources.NumActiveParticle;
    return out;
}

static void update_bulk_frame(const DriftKickTimes *times)
{
    /* Any common frame bounds relative motion. Refresh at initialization and
     * PM synchronization; speeds are measured after opening kicks each event. */
    if(Local.commit_epoch<0 || is_PM_timestep(times)) {
        double sum[4]={0};
        for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i)) {
            for(int d=0;d<3;d++) sum[d]+=P[i].Mass*P[i].Vel[d];
            sum[3]+=P[i].Mass;
        }
        MPI_Allreduce(MPI_IN_PLACE,sum,4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        if(!(sum[3]>0) || !isfinite(sum[3])) endrun(1,"Invalid total mass in local SIDM simulation.\n");
        for(int d=0;d<3;d++) {
            Local.bulk[d]=sum[d]/sum[3];
            if(!isfinite(Local.bulk[d])) endrun(1,"Nonfinite local SIDM bulk velocity.\n");
        }
    }
    Local.commit_epoch=times->Ti_Current;
#ifdef SIDMDEBUG
    /* Pre-opening speeds classify diagnostic limits only. The next drift
     * envelope is computed independently after every opening kick. */
    sidm_local_update_speed_bound();
#endif
}

void sidm_local_update_speed_bound(void)
{
    /* Relative speed is bounded by twice this maximum in any common frame.
     * The caller selects the kick phase: normally the stored midpoint
     * velocities used for the next drift, or pre-opening for diagnostics. */
    double vmax=0;
#pragma omp parallel for reduction(max:vmax)
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i)) {
        const double norm=bulk_speed2(i);
        if(!isfinite(norm)) endrun(1,"Nonfinite local SIDM canonical speed for ID %lu.\n",P[i].ID);
        vmax=fmax(vmax,sqrt(norm));
    }
    MPI_Allreduce(&vmax,&Local.max_speed,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
}

void sidm_local_assert_flushed(const DriftKickTimes *times)
{
    if(!Local.enabled) return;
    if(!is_PM_timestep(times) || times->PM_kick!=times->Ti_Current)
        endrun(1,"Local SIDM checkpoints require synchronized PM state.\n");
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i))
        if(P[i].TiSIDMLast!=times->Ti_Current || P[i].SIDMTravelLast!=Local.travel ||
           times->Ti_kick[P[i].TimeBinGravity]!=times->Ti_Current)
            endrun(1,"Local SIDM checkpoint has an unflushed source or unsynchronized velocity.\n");
}

#ifdef SIDMDEBUG
static void record_source_limit(double rate,double h)
{
    const double motion=Local.max_speed>0 ? Local.transport*h/(2*Local.max_speed):DBL_MAX;
    const int ratebin=bounded_bin(rate,Local.forward_scatter,Local.forward_maxbin);
    const int motionbin=bounded_bin(motion,Local.forward_drift,Local.forward_maxbin);
    const int capbin=bounded_bin(Local.max_dloga,Local.forward_dloga,Local.forward_maxbin);
    Local.min_rate=fmin(Local.min_rate,rate);
    Local.min_transport=fmin(Local.min_transport,motion);
    /* Quantized-bin ties are assigned rate, transport, then the time cap.
     * Actual deadline tightening is counted separately after opening. */
    if(ratebin<=motionbin && ratebin<=capbin) Local.source_rate++;
    else if(motionbin<=capbin) Local.source_transport++;
    else Local.source_maxstep++;
}

static void report_sources(int64_t count,double maxchange,const DriftKickTimes *times)
{
    MPI_Allreduce(MPI_IN_PLACE,&maxchange,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    message(0,"SIDM local sources: visited=%ld max_support_change=%.9g speed_envelope=%.9g ti=%ld\n",count,maxchange,Local.max_speed,times->Ti_Current);
    if(is_PM_timestep(times)) {
        uint64_t counts[]={Local.source_rate,Local.source_transport,Local.source_maxstep,
            Local.deadline_scans,Local.deadline_tightened,Local.gravity_checked,Local.gravity_transport};
        double minima[]={Local.min_rate,Local.min_transport};
        MPI_Allreduce(MPI_IN_PLACE,counts,7,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,minima,2,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
        message(0,"SIDM local limiters: ti=%ld source_rate=%llu source_transport=%llu source_maxstep=%llu deadline_scans=%llu deadline_tightened=%llu gravity_checked=%llu gravity_transport=%llu min_rate=%.9g min_transport=%.9g\n",
            times->Ti_Current,(unsigned long long)counts[0],(unsigned long long)counts[1],
            (unsigned long long)counts[2],(unsigned long long)counts[3],(unsigned long long)counts[4],
            (unsigned long long)counts[5],(unsigned long long)counts[6],minima[0],minima[1]);
        Local.source_rate=Local.source_transport=Local.source_maxstep=0;
        Local.deadline_scans=Local.deadline_tightened=0;
        Local.gravity_checked=Local.gravity_transport=0;
        Local.min_rate=Local.min_transport=DBL_MAX;
    }
}
#endif

void sidm_local_commit(struct sidm_local_visit *visit, DriftKickTimes *times, Cosmology *CP)
{
    prepare_forward(times->Ti_Current,CP);
    update_bulk_frame(times);
    /* Flush exposure at the final output without assigning a step beyond the
     * defined timeline. Bin zero then denotes no pending source interval. */
    const int terminal=find_next_sync_point(times->Ti_Current)==NULL;
#ifdef SIDMDEBUG
    double maxchange=0;
#endif
    for(int a=0;a<visit->sources.NumActiveParticle;a++) {
        int i=visit->sources.ActiveParticle[a];
#ifdef SIDMDEBUG
        const double h=P[i].Hsml, old=visit->old_support[i];
        record_source_limit(visit->limits[i],h);
        if(old>0 && visit->intervals[i]>0) {
            const double change=fabs(h-old);
            const double used=Local.travel-P[i].SIDMTravelLast;
            /* The Kth-neighbor radius changes by at most the largest relative
             * displacement, even when neighbor identities change. Allow for
             * rounded positions/ledger and the native drift quadrature's
             * requested relative accuracy versus the tabulated integral. */
            const double tol=128*DBL_EPSILON*fmax(PartManager->BoxSize,fabs(Local.travel))+
                1e-8*used;
            maxchange=fmax(maxchange,change/old);
            if(change>used+tol)
                endrun(1,"SIDM support change exceeds the displacement bound for ID %lu.\n",P[i].ID);
        }
#endif
        P[i].TiSIDMLast=times->Ti_Current;
        P[i].SIDMTravelLast=Local.travel;
        P[i].TimeBinSIDM=terminal?0:interval_bin(visit->limits[i]);
        P[i].TiSIDMNext=times->Ti_Current+dti_from_timebin(P[i].TimeBinSIDM);
    }
#ifdef SIDMDEBUG
    report_sources(visit->sources.NumActiveParticle,maxchange,times);
    myfree(visit->old_support);
#endif
    myfree(visit->limits);
    myfree(visit->intervals);
    myfree(visit->sources.ActiveParticle);
}

double sidm_local_gravity_limit(int i, const DriftKickTimes *times)
{
    if(!Local.enabled || !live_dm(i)) return DBL_MAX;
#ifdef SIDMDEBUG
    if(Local.commit_epoch!=times->Ti_Current) endrun(1,"SIDM gravity limit requested before source commit.\n");
#else
    (void)times;
#endif
    const double speed=sqrt(bulk_speed2(i));
    /* The scattering support follows local particle spacing. Using the fixed
     * gravity support alone would impose a tiny step on the entire outer halo. */
    const double scale=fmax(P[i].Hsml,FORCE_SOFTENING());
    return speed>0 ? Local.transport*scale/speed:DBL_MAX;
}

#else
void sidm_local_initialize(const DriftKickTimes *t,Cosmology *c) { (void)t;(void)c; }
inttime_t sidm_local_next_event(const DriftKickTimes *t,Cosmology *c,int first) { (void)c;(void)first; return find_next_kick(t->Ti_Current,t->mintimebin); }
void sidm_local_update_speed_bound(void) { }
void sidm_local_assert_flushed(const DriftKickTimes *t) { (void)t; }
void sidm_local_kick_phase(DriftKickTimes *t,int opening) { (void)opening; update_kick_times(t); }
double sidm_local_gravity_limit(int i,const DriftKickTimes *t) { (void)i;(void)t;return DBL_MAX; }
#endif
