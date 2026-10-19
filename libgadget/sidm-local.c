#include "sidm-local.h"
#include "timefac.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

static struct {
    int enabled;
    double transport, max_dloga;
    int64_t bins[TIMEBINS+1];
    inttime_t forward_epoch;
    int forward_maxbin;
    double forward_drift[TIMEBINS+1], forward_scatter[TIMEBINS+1];
    double forward_dloga[TIMEBINS+1];
#ifdef SIDMDEBUG
    uint64_t source_rate, source_transport, source_maxstep, due, flushed;
#endif
} Local;

void sidm_local_configure(int enabled, double transport, double max_dloga)
{
    memset(&Local, 0, sizeof(Local));
    Local.enabled=enabled;
    Local.transport=transport;
    Local.max_dloga=max_dloga;
    if(enabled && (!(transport>0 && transport<=.25) ||
       !isfinite(max_dloga) || max_dloga<=0))
        endrun(1,"Invalid local SIDM transport fraction or maximum timestep.\n");
#ifndef SIDM
    if(enabled) endrun(1,"Local SIDM needs an SIDM build.\n");
#endif
}
int sidm_local_enabled(void) { return Local.enabled; }

void sidm_local_set_bin_counts(const int64_t counts[TIMEBINS+1])
{
    memcpy(Local.bins,counts,sizeof(Local.bins));
}

#ifdef SIDM
static int live_dm(int i) { return P[i].Type==1 && !P[i].IsGarbage && !P[i].Swallowed; }

/* Forward integrals share an epoch. Aligned bins also stop source intervals
 * at the next output, where every pending exposure is flushed. */
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

static int interval_bin(double exposure,double motion,int *limiter)
{
    int bin=bounded_bin(exposure,Local.forward_scatter,Local.forward_maxbin);
    bin=bounded_bin(Local.transport*motion,Local.forward_drift,bin);
    bin=bounded_bin(Local.max_dloga,Local.forward_dloga,bin);
    if(bin<1) endrun(1,"SIDM source interval is below representable time resolution.\n");
#ifdef SIDMDEBUG
    const int rate=bounded_bin(exposure,Local.forward_scatter,Local.forward_maxbin);
    const int drift=bounded_bin(Local.transport*motion,Local.forward_drift,Local.forward_maxbin);
    const int cap=bounded_bin(Local.max_dloga,Local.forward_dloga,Local.forward_maxbin);
    *limiter=rate<=drift && rate<=cap?0:drift<=cap?1:2;
#else
    (void)limiter;
#endif
    return bin;
}

void sidm_local_initialize(const DriftKickTimes *times,Cosmology *CP)
{
    if(!Local.enabled) return;
    if(sizeof(P[0].Vel[0])!=sizeof(double)) endrun(1,"Local SIDM requires double velocities.\n");
    sidm_time_factors_init(CP);
    Local.forward_epoch=-1;
    memset(Local.bins,0,sizeof(Local.bins));
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i)) {
        P[i].TiSIDMLast=times->Ti_Current;
        P[i].TimeBinSIDM=0;
        Local.bins[0]++;
    }
}

inttime_t sidm_local_next_event(const DriftKickTimes *times,int first)
{
    if(first) return times->Ti_Current;
    const inttime_t now=times->Ti_Current;
    inttime_t next=find_next_kick(now,times->mintimebin);
    const inttime_t pm=times->PM_start+times->PM_length;
    if(pm<=now) endrun(1,"Local SIDM scheduler found an expired PM deadline.\n");
    if(pm<next) next=pm;
    /* Native activity construction recounts these bins after migration. Commit
     * updates only visited sources. No particle scan is needed here. */
    for(int b=1;b<=TIMEBINS;b++) if(Local.bins[b]) {
        const inttime_t due=find_next_kick(now,b);
        if(due<next) next=due;
        break;
    }
    MPI_Allreduce(MPI_IN_PLACE,&next,1,MPI_INT64_T,MPI_MIN,MPI_COMM_WORLD);
    return next;
}

void sidm_local_kick_phase(DriftKickTimes *times,int opening)
{
    for(int b=0;b<=TIMEBINS;b++) if(is_timebin_active(b,times->Ti_Current))
        times->Ti_kick[b]=times->Ti_Current+(opening?dti_from_timebin(b)/2:0);
}

static int select_source(int i,const DriftKickTimes *times,int flush)
{
    if(!live_dm(i)) return 0;
    const inttime_t next=sidm_source_deadline(&P[i]);
    if(P[i].TiSIDMLast>times->Ti_Current || next<times->Ti_Current)
        endrun(1,"SIDM source interval missed for ID %lu.\n",P[i].ID);
    return flush || next==times->Ti_Current;
}

struct sidm_local_visit sidm_local_begin_active(const DriftKickTimes *times,Cosmology *CP,
    int flush,const ActiveParticles *active)
{
    const int n=PartManager->NumPart;
    struct sidm_local_visit out={0};
    out.sources.Particles=P;
    const int indexed=active && !flush;
    const int candidates=indexed?active->NumActiveParticle:n;
    out.sources.ActiveParticle=mymalloc("SIDM active sources",((size_t)candidates+1)*sizeof(int));
    if(omp_get_max_threads()>1 && candidates>=1024*omp_get_max_threads()) {
        gadget_thread_arrays queues={.dest=out.sources.ActiveParticle,.total_size=candidates};
#pragma omp parallel
        {
#pragma omp master
            {
                queues.narrays=omp_get_num_threads();
                queues.schedsz=(size_t)candidates/queues.narrays+1;
                queues.sizes=ta_malloc("SIDM source counts",size_t,queues.narrays);
                queues.srcs=ta_malloc("SIDM source queues",int *,queues.narrays);
                /* Native compaction preserves chunk order; allocate only one
                 * static chunk per queue instead of a full population per thread. */
                queues.srcs[0]=queues.dest;
                for(int t=0;t<queues.narrays;t++) {
                    queues.sizes[t]=0;
                    if(t) queues.srcs[t]=mymalloc("SIDM source chunk",queues.schedsz*sizeof(int));
                }
            }
#pragma omp barrier
            const int tid=omp_get_thread_num();
            size_t count=0;
#pragma omp for schedule(static,queues.schedsz)
            for(int a=0;a<candidates;a++) {
                const int i=indexed && active->ActiveParticle?active->ActiveParticle[a]:a;
                if(!select_source(i,times,flush)) continue;
                queues.srcs[tid][count++]=i;
            }
            queues.sizes[tid]=count;
        }
        out.sources.NumActiveParticle=gadget_compact_thread_arrays(&out.sources.ActiveParticle,&queues);
    } else {
        for(int a=0;a<candidates;a++) {
            const int i=indexed && active->ActiveParticle?active->ActiveParticle[a]:a;
            if(!select_source(i,times,flush)) continue;
            out.sources.ActiveParticle[out.sources.NumActiveParticle++]=i;
        }
    }
#ifdef SIDMDEBUG
    if(indexed && out.sources.NumActiveParticle!=active->NumActiveSIDM)
        endrun(1,"Local SIDM source count disagrees with native activity.\n");
#endif
    const int nsource=out.sources.NumActiveParticle;
    out.sources.NumActiveSIDM=nsource;
    out.sources.ActiveParticle=myrealloc(out.sources.ActiveParticle,((size_t)nsource+1)*sizeof(int));
    out.intervals=mymalloc("SIDM intervals",((size_t)nsource+1)*sizeof(double));
    out.limits=mymalloc("SIDM limits",((size_t)nsource+1)*sizeof(double));
    out.motion_limits=mymalloc("SIDM motion limits",((size_t)nsource+1)*sizeof(double));
#ifdef SIDMDEBUG
    out.old_support=mymalloc("SIDM previous support",((size_t)nsource+1)*sizeof(double));
    uint64_t due=0;
#pragma omp parallel for if(nsource>=1024*omp_get_max_threads()) reduction(+:due)
#else
#pragma omp parallel for if(nsource>=1024*omp_get_max_threads())
#endif
    for(int a=0;a<nsource;a++) {
        const int i=out.sources.ActiveParticle[a];
        out.intervals[a]=sidm_scatter_factor(CP,P[i].TiSIDMLast,times->Ti_Current);
#ifdef SIDMDEBUG
        out.old_support[a]=P[i].Hsml;
        due+=sidm_source_deadline(&P[i])==times->Ti_Current;
#endif
    }
#ifdef SIDMDEBUG
    Local.due+=due;Local.flushed+=nsource-due;
#endif
    return out;
}

void sidm_local_assert_flushed(const DriftKickTimes *times)
{
    if(!Local.enabled) return;
    if(!is_PM_timestep(times) || times->PM_kick!=times->Ti_Current)
        endrun(1,"Local SIDM checkpoints require synchronized PM state.\n");
    for(int i=0;i<PartManager->NumPart;i++) if(live_dm(i))
        if(P[i].TiSIDMLast!=times->Ti_Current ||
           times->Ti_kick[P[i].TimeBinGravity]!=times->Ti_Current)
            endrun(1,"Local SIDM checkpoint has an unflushed source or unsynchronized velocity.\n");
}

void sidm_local_commit(struct sidm_local_visit *visit,DriftKickTimes *times,Cosmology *CP)
{
    prepare_forward(times->Ti_Current,CP);
    const int terminal=find_next_sync_point(times->Ti_Current)==NULL;
    const int nsource=visit->sources.NumActiveParticle;
    int64_t delta[TIMEBINS+1]={0};
#ifdef SIDMDEBUG
    double maxchange=0;
    uint64_t source_limits[3]={0};
#pragma omp parallel for if(nsource>=1024*omp_get_max_threads()) reduction(+:delta[:TIMEBINS+1],source_limits[:3]) reduction(max:maxchange)
#else
#pragma omp parallel for if(nsource>=1024*omp_get_max_threads()) reduction(+:delta[:TIMEBINS+1])
#endif
    for(int a=0;a<nsource;a++) {
        const int i=visit->sources.ActiveParticle[a];
#ifdef SIDMDEBUG
        if(visit->old_support[a]>0 && visit->intervals[a]>0)
            maxchange=fmax(maxchange,fabs(P[i].Hsml/visit->old_support[a]-1));
#endif
        delta[P[i].TimeBinSIDM]--;
        P[i].TiSIDMLast=times->Ti_Current;
        /* Local relative motion predicts a support crossing time. It is not
         * a bound on later entrants or on velocity changes after this visit. */
        int limiter=0;
        P[i].TimeBinSIDM=terminal?0:interval_bin(visit->limits[a],visit->motion_limits[a],&limiter);
        delta[P[i].TimeBinSIDM]++;
#ifdef SIDMDEBUG
        source_limits[limiter]+=!terminal;
#endif
    }
    for(int b=0;b<=TIMEBINS;b++) Local.bins[b]+=delta[b];
#ifdef SIDMDEBUG
    Local.source_rate+=source_limits[0];
    Local.source_transport+=source_limits[1];
    Local.source_maxstep+=source_limits[2];
    MPI_Allreduce(MPI_IN_PLACE,&maxchange,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    message(0,"SIDM local sources: max_support_change=%.9g ti=%ld\n",maxchange,times->Ti_Current);
    if(is_PM_timestep(times)) {
        uint64_t counts[]={Local.source_rate,Local.source_transport,Local.source_maxstep,Local.due,Local.flushed};
        MPI_Allreduce(MPI_IN_PLACE,counts,5,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
        message(0,"SIDM local limiters: ti=%ld source_rate=%llu source_transport=%llu source_maxstep=%llu due=%llu flushed=%llu\n",
            times->Ti_Current,(unsigned long long)counts[0],(unsigned long long)counts[1],
            (unsigned long long)counts[2],(unsigned long long)counts[3],(unsigned long long)counts[4]);
        Local.source_rate=Local.source_transport=Local.source_maxstep=Local.due=Local.flushed=0;
    }
    myfree(visit->old_support);
#endif
    myfree(visit->motion_limits);
    myfree(visit->limits);
    myfree(visit->intervals);
    myfree(visit->sources.ActiveParticle);
}

#else
void sidm_local_initialize(const DriftKickTimes *t,Cosmology *c) { (void)t;(void)c; }
inttime_t sidm_local_next_event(const DriftKickTimes *t,int first) { (void)first;return find_next_kick(t->Ti_Current,t->mintimebin); }
void sidm_local_assert_flushed(const DriftKickTimes *t) { (void)t; }
void sidm_local_kick_phase(DriftKickTimes *t,int opening) { (void)opening;update_kick_times(t); }
#endif
