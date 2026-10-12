#include "sidm-distributed.h"
#include "sidm-direct.h"
#include "sidm-local.h"
#include "sidm-rate.h"
#include "sidm-random.h"
#include "treewalk.h"
#include "utils/endrun.h"
#include "walltime.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct { int neighbors, velocity_dependent; double probability, sigma0, w; } DistParams;
#ifdef SIDM
static void release_work_storage(void);
#endif
void set_sidm_distributed_params(ParameterSet *ps)
{
#ifdef SIDM
    release_work_storage();
#endif
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(!rank) {
        DistParams.neighbors=param_get_int(ps,"SIDMNeighbors");
        DistParams.probability=param_get_double(ps,"SIDMDirectedProbability");
        DistParams.velocity_dependent=param_get_int(ps,"vdSIDMOn");
        DistParams.sigma0=param_get_double(ps,"sigma0");
        DistParams.w=param_get_double(ps,"wTurn");
    }
    MPI_Bcast(&DistParams,sizeof(DistParams),MPI_BYTE,0,MPI_COMM_WORLD);
    if(DistParams.neighbors<1 || DistParams.neighbors>SIDM_DIRECT_MAX_NEIGHBORS ||
       !(DistParams.probability>0 && DistParams.probability<=1) ||
       !isfinite(DistParams.sigma0) || DistParams.sigma0<0 ||
       (DistParams.velocity_dependent && (!(DistParams.w>0) || !isfinite(DistParams.w))))
        endrun(1,"Invalid SIDM neighbor count, probability, or cross-section parameters.\n");
}

#ifdef SIDM
/* Radius seeds affect search work only; the retained K neighbors remain exact. */
#define SUPPORT_REUSE_FACTOR 1.25
#define SUPPORT_LEAF_FACTOR .75
#define SUPPORT_MAX_ITERATIONS 64

struct support_result { TreeWalkResultBase base; int n; struct sidm_direct_neighbor heap[]; };
struct support_work { double radius, h; int done, n; };
struct dist_query { TreeWalkQueryBase base; double h; uint64_t id; };
struct dist_work {
    struct support_work *support;
    struct sidm_direct_neighbor *heaps;
    int *source_index, rank, neighbors;
};
struct dist_scratch { void *data; size_t capacity; };
static struct {
    struct dist_scratch support, source_index, heaps, direct_sources;
#ifdef SIDMDEBUG
    struct dist_scratch entries, marked;
#endif
    /* Fixed equal-mass DM; ownership changes do not change these invariants. */
    double mass;
    int64_t count;
} Scratch;
static void *reserve_scratch(struct dist_scratch *s,size_t count,size_t width)
{
    if(count>SIZE_MAX/width) endrun(1,"SIDM scratch size overflow.\n");
    if(count>s->capacity) {
        void *next=realloc(s->data,count*width);
        if(!next) endrun(1,"SIDM scratch allocation failed.\n");
        s->data=next; s->capacity=count;
    }
    return s->data;
}
static void release_work_storage(void)
{
    free(Scratch.support.data); free(Scratch.source_index.data);
    free(Scratch.heaps.data); free(Scratch.direct_sources.data);
#ifdef SIDMDEBUG
    free(Scratch.entries.data); free(Scratch.marked.data);
#endif
    memset(&Scratch,0,sizeof(Scratch));
}
static int farther(struct sidm_direct_neighbor a, struct sidm_direct_neighbor b)
{ return a.r2 > b.r2 || (a.r2 == b.r2 && a.id > b.id); }

static struct sidm_direct_neighbor *support_heap(struct dist_work *work, int i)
{
    const int slot = work->source_index[i];
    return work->heaps + (size_t)slot*work->neighbors;
}

/* K largest-at-root heap. Local and exported walks cover disjoint particles;
 * the debug reduction checks this invariant across returned partial heaps. */
static void retain(int *n, struct sidm_direct_neighbor *heap, struct sidm_direct_neighbor value, int k)
{
    int at;
    if(*n < k) {
        at = (*n)++;
        while(at && farther(value, heap[(at-1)/2])) {
            heap[at] = heap[(at-1)/2]; at = (at-1)/2;
        }
    } else {
        if(!farther(heap[0], value)) return;
        at = 0;
        while(2*at+1 < k) {
            int child = 2*at+1;
            if(child+1 < k && farther(heap[child+1], heap[child])) child++;
            if(!farther(heap[child], value)) break;
            heap[at] = heap[child]; at = child;
        }
    }
    heap[at] = value;
}
static int support_haswork(int i, TreeWalk *tw)
{
    struct dist_work *work = tw->priv;
    return P[i].Type == 1 && !P[i].IsGarbage && !P[i].Swallowed && !work->support[i].done;
}
static void support_fill(int i, TreeWalkQueryBase *query, TreeWalk *tw)
{
    struct dist_work *work = tw->priv;
    struct dist_query *q = (struct dist_query *)query;
    q->id = P[i].ID; q->h = work->support[i].radius;
}
static void support_reduce(int i, TreeWalkResultBase *result, enum TreeWalkReduceMode mode, TreeWalk *tw)
{
    struct dist_work *work = tw->priv;
    struct support_work *out = &work->support[i];
    struct sidm_direct_neighbor *heap = support_heap(work,i);
    struct support_result *incoming = (struct support_result *)result;
    if(mode == TREEWALK_PRIMARY) {
        out->n=incoming->n;
        memcpy(heap,incoming->heap,(size_t)incoming->n*sizeof(*heap));
        return;
    }
    for(int j = 0; j < incoming->n; j++) {
#ifdef SIDMDEBUG
        for(int t = 0; t < out->n; t++)
            if(heap[t].id == incoming->heap[j].id)
                endrun(1,"Duplicate SIDM neighbor in exported support reduction.\n");
#endif
        retain(&out->n, heap, incoming->heap[j], work->neighbors);
    }
}
static void support_ngbiter(TreeWalkQueryBase *query, TreeWalkResultBase *result,
    TreeWalkNgbIterBase *iter, LocalTreeWalk *lv)
{
    const struct dist_query *q = (const struct dist_query *)query;
    struct support_result *out = (struct support_result *)result;
    struct dist_work *work = lv->tw->priv;
    if(iter->other == -1) {
        iter->Hsml = q->h; iter->mask = DMMASK; iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int j = iter->other;
    if(P[j].Swallowed || P[j].ID == q->id) return;
    retain(&out->n, out->heap, (struct sidm_direct_neighbor){.r2=iter->r2,.id=P[j].ID,.owner=work->rank,.index=j}, work->neighbors);
    if(out->n == work->neighbors) iter->Hsml = nextafter(sqrt(out->heap[0].r2), INFINITY);
}
static TreeWalk make_walk(const ForceTree *tree,struct dist_work *work)
{
    TreeWalk tw={0};tw.tree=tree;tw.priv=work;tw.ev_label="SIDM_Support";
    tw.visit=treewalk_visit_nolist_ngbiter;tw.NoNgblist=1;
    tw.ngbiter=support_ngbiter;tw.ngbiter_type_elsize=sizeof(TreeWalkNgbIterBase);
    tw.haswork=support_haswork;tw.fill=support_fill;tw.reduce=support_reduce;
    tw.query_type_elsize=sizeof(struct dist_query);
    tw.result_type_elsize=offsetof(struct support_result,heap)+(size_t)work->neighbors*sizeof(struct sidm_direct_neighbor);
    return tw;
}

static void audit_population(const ForceTree *tree)
{
    double mmin=DBL_MAX,mmax=0;int64_t count=0;
#pragma omp parallel for reduction(min:mmin) reduction(max:mmax) reduction(+:count)
    for(int i=0;i<PartManager->NumPart;i++) if(P[i].Type==1 && !P[i].IsGarbage && !P[i].Swallowed) {
        if(!isfinite(P[i].Mass) || P[i].Mass<=0 || sizeof(P[i].Vel[0])!=sizeof(double))
            endrun(1,"SIDM requires equal-mass DM with double velocities.\n");
        for(int d=0;d<3;d++) if(!isfinite(P[i].Vel[d]) || !isfinite(P[i].Pos[d]))
            endrun(1,"Nonfinite SIDM phase-space state.\n");
        mmin=fmin(mmin,P[i].Mass);mmax=fmax(mmax,P[i].Mass);count++;
    }
    if(count!=tree->NumParticles) endrun(1,"SIDM tree does not contain all live DM.\n");
    MPI_Allreduce(MPI_IN_PLACE,&mmin,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&mmax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    if(mmin!=mmax || count<=DistParams.neighbors) endrun(1,"SIDM needs equal masses and enough neighbors.\n");
    Scratch.mass=mmin;Scratch.count=count;
}

#ifdef SIDMDEBUG
static void report_support_timers(const double walk_seconds[6])
{
    /* Exclusive TreeWalk stages. TopExport and Results include communication;
     * Secondary includes arrival waits, so these are not network-only timers. */
    const char *walk_labels[]={"TopExport","Primary","Secondary","Wait","Results","Setup"};
    double support_other=walltime_measure(".");
    for(int k=0;k<6;k++) {
        char label[80];snprintf(label,sizeof(label),"/SPH/SIDM_Support/%s",walk_labels[k]);
        walltime_add_internal(label,walk_seconds[k]); support_other-=walk_seconds[k];
    }
    walltime_add_internal("/SPH/SIDM_Support/Other",support_other);
}

static void report_scatter(const struct sidm_direct_stats *stats,int nactive,
    double limit,double hmin,double hmax,int iterations)
{
    /* Exclusive leaves: recording the whole stage beside its children would
     * count the stage twice in the SPH/root totals. Release keeps one timer. */
    const double elapsed=walltime_measure(".");
    walltime_add("/SPH/SIDM_ExchangeScatter/Other",elapsed-stats->prepare_seconds-stats->route_seconds-
        stats->pack_seconds-stats->mpi_seconds-stats->pair_seconds-stats->audit_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Prepare",stats->prepare_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Route",stats->route_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Pack",stats->pack_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/MPI",stats->mpi_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Predict",stats->predict_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Pairs",stats->pair_seconds-stats->predict_seconds);
    walltime_add("/SPH/SIDM_ExchangeScatter/Audit",stats->audit_seconds);
    MPI_Allreduce(MPI_IN_PLACE,&limit,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    int64_t nsources=nactive;
    MPI_Allreduce(MPI_IN_PLACE,&nsources,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&hmin,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&hmax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    message(0,"SIDMDirectIntervals: N=%ld sources=%ld min_source_limit=%.17g h=[%.17g,%.17g] support_iterations=%d directed_pairs=%llu trials=%llu events=%llu subdivided_pairs=%llu source_packets=%llu bytes=%llu journaled=%llu threaded_pairs=%llu recomputed_pairs=%llu maxP=%.17g maxTrialP=%.17g sumP=%.17g deltaK=%.17Lg deltaP=%.17Lg\n",
        (long)Scratch.count,(long)nsources,limit,nsources?hmin:0,hmax,iterations,
        (unsigned long long)stats->pairs,(unsigned long long)stats->trials,(unsigned long long)stats->events,
        (unsigned long long)stats->subdivided_pairs,(unsigned long long)stats->source_packets,(unsigned long long)stats->bytes,
        (unsigned long long)stats->journaled_particles,(unsigned long long)stats->threaded_pairs,
        (unsigned long long)stats->recomputed_pairs,stats->max_probability,stats->max_trial_probability,
        stats->sum_probability,stats->delta_k,stats->delta_p);
}
#endif

void sidm_distributed_force_intervals(const ActiveParticles *act,Cosmology *CP,
    struct UnitSystem units,DriftKickTimes *times,const ForceTree *tree,
    const double *intervals,double *limits,uint64_t seed)
{
    const int npart=PartManager->NumPart, nactive=act->NumActiveParticle;
    struct dist_work work={0};MPI_Comm_rank(MPI_COMM_WORLD,&work.rank);
    work.neighbors=DistParams.neighbors;
    work.support=reserve_scratch(&Scratch.support,(size_t)npart+1,sizeof(*work.support));
    work.source_index=reserve_scratch(&Scratch.source_index,(size_t)npart+1,sizeof(*work.source_index));
    work.heaps=reserve_scratch(&Scratch.heaps,((size_t)nactive+1)*work.neighbors,sizeof(*work.heaps));
    struct sidm_journal journal={.validated_mass=Scratch.mass};
    int audit=Scratch.mass<=0;
#ifdef SIDMDEBUG
    audit=1;
    journal.capacity=npart;
    journal.entries=reserve_scratch(&Scratch.entries,(size_t)npart+1,sizeof(*journal.entries));
    journal.marked=reserve_scratch(&Scratch.marked,(size_t)npart+1,sizeof(*journal.marked));
    memset(journal.marked,0,(size_t)npart);
    for(int i=0;i<npart;i++) work.source_index[i]=-1;
#endif
    if(audit) audit_population(tree);
    journal.validated_mass=Scratch.mass;
    const double mass=Scratch.mass;
    const double sigma0=DistParams.sigma0*CP->HubbleParam*units.UnitMass_in_g/(units.UnitLength_in_cm*units.UnitLength_in_cm);
    /* Stored velocities are a times peculiar velocity. The dt/a^4 source
     * exposure already includes expansion of density and relative speed.
     * The velocity-dependent shape is evaluated at this event endpoint. */
    const double atime=get_atime(times->Ti_Current);
    const double w=DistParams.w*1e5/units.UnitVelocity_in_cm_per_s*atime;
    const double maxradius=nextafter(.5*PartManager->BoxSize,0);
    const double guess=PartManager->BoxSize*cbrt((double)work.neighbors/Scratch.count);
    for(int a=0;a<nactive;a++) {
        const int i=act->ActiveParticle?act->ActiveParticle[a]:a;
#ifdef SIDMDEBUG
        if(i<0 || i>=npart || work.source_index[i]>=0 || P[i].Ti_drift!=times->Ti_Current)
            endrun(1,"Invalid SIDM source list or drift epoch.\n");
#endif
        work.source_index[i]=a;
        memset(&work.support[i],0,sizeof(work.support[i]));
        const int cached=isfinite(P[i].Hsml) && P[i].Hsml>0;
        double h=cached?SUPPORT_REUSE_FACTOR*P[i].Hsml:guess;
        if(!cached && tree->Father) {
            const int no=tree->Father[i];
            if(no>=tree->firstnode && no<tree->firstnode+tree->numnodes) {
                const struct NODE *node=&tree->Nodes[no];
                if(node->f.ChildType==PARTICLE_NODE_TYPE && node->s.noccupied>0) {
                    const double local=SUPPORT_LEAF_FACTOR*node->len*cbrt((double)work.neighbors/node->s.noccupied);
                    if(isfinite(local) && local>0) h=local;
                }
            }
        }
        work.support[i].radius=fmin(h,maxradius);
    }
    int iterations=0;
#ifdef SIDMDEBUG
    double walk_seconds[6]={0};
#endif
    for(;;) {
        TreeWalk tw=make_walk(tree,&work);
        treewalk_run(&tw,act->ActiveParticle,nactive);
#ifdef SIDMDEBUG
        walk_seconds[0]+=tw.timecomp0; walk_seconds[1]+=tw.timecomp1;
        walk_seconds[2]+=tw.timecomp2; walk_seconds[3]+=tw.timewait1;
        walk_seconds[4]+=tw.timecommsumm; walk_seconds[5]+=tw.timecomp3;
#endif
        int64_t left=0;
        for(int a=0;a<nactive;a++) {
            const int i=act->ActiveParticle?act->ActiveParticle[a]:a;
            struct support_work *s=&work.support[i];
            if(s->done) continue;
            if(s->n==work.neighbors) {
                s->h=nextafter(sqrt(support_heap(&work,i)[0].r2),INFINITY);s->done=1;
            } else {
                if(s->radius>=maxradius) endrun(1,"SIDM cannot find enough neighbors within allowed support.\n");
                s->radius=fmin(2*s->radius,maxradius);left++;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,&left,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
        iterations++;
        if(!left) break;
        if(iterations>=SUPPORT_MAX_ITERATIONS) endrun(1,"SIDM support search did not converge.\n");
    }
    struct sidm_direct_source *sources=reserve_scratch(&Scratch.direct_sources,(size_t)nactive+1,sizeof(*sources));
#ifdef SIDMDEBUG
    double hmin=DBL_MAX,hmax=0;
#endif
    for(int a=0;a<nactive;a++) {
        const int i=act->ActiveParticle?act->ActiveParticle[a]:a;
        const double h=work.support[i].h;
        if(!isfinite(h) || h<=DBL_MIN || h>=.5*PartManager->BoxSize)
            endrun(1,"Degenerate or oversized SIDM support.\n");
        P[i].Hsml=h;
#ifdef SIDMDEBUG
        hmin=fmin(hmin,h);hmax=fmax(hmax,h);
#endif
        sources[a]=(struct sidm_direct_source){.index=i,.n=work.neighbors,.neighbors=support_heap(&work,i),
            .h=h,.dt=intervals[i],.seed=sidm_event_seed(seed,(uint64_t)times->Ti_Current)};
    }
#ifdef SIDMDEBUG
    report_support_timers(walk_seconds);
#else
    walltime_measure("/SPH/SIDM_Support");
#endif
    struct sidm_direct_stats stats;char error[256];
    const struct sidm_direct_particles native={P,sizeof(*P),offsetof(struct particle_data,ID),offsetof(struct particle_data,Vel)};
    if(sidm_direct_scatter_view(native,npart,sources,nactive,sigma0,w,DistParams.velocity_dependent,
        DistParams.probability,MPI_COMM_WORLD,&journal,&stats,error)) endrun(1,"SIDM: %s\n",error);
#ifdef SIDMDEBUG
    double limit=DBL_MAX;
#endif
    for(int a=0;a<nactive;a++) {
        const int i=act->ActiveParticle?act->ActiveParticle[a]:a;
        const double rate=.5*mass*sidm_rate_cubic_kernel(0,sources[a].h)*sources[a].max_sigma_v;
        limits[i]=rate>0?DistParams.probability/rate:DBL_MAX;
        if(!isfinite(rate) || rate<0 || !isfinite(limits[i]) || limits[i]<=0)
            endrun(1,"Invalid SIDM measured source rate.\n");
#ifdef SIDMDEBUG
        limit=fmin(limit,limits[i]);
#endif
    }
#ifdef SIDMDEBUG
    report_scatter(&stats,nactive,limit,hmin,hmax,iterations);
#else
    walltime_measure("/SPH/SIDM_ExchangeScatter");
#endif
    walltime_measure("/SPH/SIDM_Finalize");
}
#endif
