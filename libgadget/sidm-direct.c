#include "sidm-direct.h"
#include "sidm-rate.h"
#include "sidm-random.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One descriptor per (source,destination), not a second full edge array. */
struct group { int source, first, count; };
struct packet { uint64_t seed; double v[3]; int n; };
struct target {
    double coefficient;
    int index;
#ifdef SIDMDEBUG
    uint64_t id; /* Owner index is sufficient outside development audits. */
#endif
};
struct reply { double v[3], max_sigma_v; };
#ifdef SIDMDEBUG
#define PROFILE(code) do { code; } while(0)
#else
#define PROFILE(code) ((void)0)
#endif
/* Independent compile-time sizes for transport and read-only evaluation.
 * A source occurs once per destination, so delaying its return until the
 * end of a larger message does not change the ordered collision updates. */
#ifndef SIDM_DIRECT_WIRE_SOURCES
#define SIDM_DIRECT_WIRE_SOURCES 512
#endif
#ifndef SIDM_DIRECT_EVAL_SOURCES
#define SIDM_DIRECT_EVAL_SOURCES 128
#endif
#define WIRE_BYTES (SIDM_DIRECT_WIRE_SOURCES*(sizeof(struct packet)+SIDM_DIRECT_MAX_NEIGHBORS*sizeof(struct target)))
static struct {
    int *routing, ranks;
    unsigned char buffer[WIRE_BYTES];
    struct reply returned[SIDM_DIRECT_WIRE_SOURCES];
} Workspace;

static int reserve_routing(int ntask)
{
    if(ntask>Workspace.ranks) {
        int *next=realloc(Workspace.routing,((size_t)ntask*4+1)*sizeof(*next));
        if(!next) return -1;
        Workspace.routing=next; Workspace.ranks=ntask;
    }
    return 0;
}

/* Only read-only pair evaluation is threaded. Accepted collisions are applied
 * in the original order. A changed endpoint invalidates its cached rates for
 * the rest of this batch, including changes from another source's collision.
 * Predictions use fixed storage; only per-particle marks grow with the local
 * population. No second full edge or velocity array is needed. */
struct pair_evaluation { double u, sv, draw; };
static struct {
    struct pair_evaluation pairs[SIDM_DIRECT_EVAL_SOURCES*SIDM_DIRECT_MAX_NEIGHBORS];
    unsigned *changed, epoch;
    size_t capacity;
} Cache;
static int reserve_cache(int nlocal)
{
    if((size_t)nlocal>Cache.capacity) {
        unsigned *next=realloc(Cache.changed,(size_t)nlocal*sizeof(*next));
        if(!next) return -1;
        memset(next+Cache.capacity,0,((size_t)nlocal-Cache.capacity)*sizeof(*next));
        Cache.changed=next;Cache.capacity=nlocal;
    }
    return 0;
}
static unsigned cache_epoch(void)
{
    if(++Cache.epoch==0) {
        memset(Cache.changed,0,Cache.capacity*sizeof(*Cache.changed));
        Cache.epoch=1;
    }
    return Cache.epoch;
}

static double uniform(uint64_t *s) { *s=sidm_random_mix(*s); return (*s>>11)*0x1.0p-53; }
static void fail(int *failed,char error[256],const char *why)
{ if(!*failed) snprintf(error,256,"%s",why); *failed=1; }
static int collective_error(int failed,char error[256],MPI_Comm comm)
{
    int rank,first; MPI_Comm_rank(comm,&rank); first=failed?rank:INT_MAX;
    MPI_Allreduce(MPI_IN_PLACE,&first,1,MPI_INT,MPI_MIN,comm);
    if(first==INT_MAX) return 0;
    MPI_Bcast(error,256,MPI_CHAR,first,comm); return -1;
}
static int neighbor_order(const void *a,const void *b)
{
    const struct sidm_direct_neighbor *x=a,*y=b;
    if(x->owner!=y->owner) return (x->owner>y->owner)-(x->owner<y->owner);
    return (x->id>y->id)-(x->id<y->id);
}
static uint64_t particle_id(struct sidm_direct_particles p,int i)
{ return *(const uint64_t *)((char *)p.base+(size_t)i*p.stride+p.id_offset); }
static double *velocity(struct sidm_direct_particles p,int i)
{ return (double *)((char *)p.base+(size_t)i*p.stride+p.velocity_offset); }
#ifdef SIDMDEBUG
static void touch(struct sidm_journal *j,struct sidm_direct_particles p,int i,
    int *failed,char error[256])
{
    if(j->marked[i]) return;
    for(int d=0;d<3;d++) if(!isfinite(velocity(p,i)[d])) fail(failed,error,"Nonfinite direct SIDM touched velocity");
    j->entries[j->count].index=i;
    memcpy(j->entries[j->count++].before,velocity(p,i),3*sizeof(double)); j->marked[i]=1;
}

#endif
static struct pair_evaluation evaluate_pair(const double a[3],const double b[3],
    uint64_t bid,uint64_t seed,double sigma0,double w,int vd)
{
    const double u=hypot(hypot(a[0]-b[0],a[1]-b[1]),a[2]-b[2]);
    const double sv=sigma0*u*(vd?sidm_rate_viscosity_shape((u/w)*(u/w)):1);
    seed=sidm_random_mix(seed^sidm_random_mix(bid+UINT64_C(0xd1b54a32d192ed03)));
    return (struct pair_evaluation){u,sv,uniform(&seed)};
}
static int scatter_pair(double a[3],double b[3],uint64_t bid,uint64_t seed,
    double coefficient,double sigma0,double w,int vd,double pmax,
    struct sidm_journal *j,struct sidm_direct_particles p,int ai,int bi,
    double *max_sv,struct sidm_direct_stats *s,int *failed,char error[256],
    const struct pair_evaluation *cached)
{
    if(*failed) return 0;
    const double u=cached?cached->u:hypot(hypot(a[0]-b[0],a[1]-b[1]),a[2]-b[2]);
    const double sv=cached?cached->sv:sigma0*u*(vd?sidm_rate_viscosity_shape((u/w)*(u/w)):1);
    const double q=coefficient*sv;
    if(!isfinite(sv) || sv<0 || !isfinite(q) || q<0) {
        fail(failed,error,"Invalid direct SIDM current pair rate/probability"); return 0;
    }
    if(sv>*max_sv) *max_sv=sv;
#ifdef SIDMDEBUG
    s->pairs++;
    s->max_probability=fmax(s->max_probability,q); s->sum_probability+=q;
#else
    (void)s;
#endif
    if(q==0) return 0;
    const double nc=q<=pmax?1:ceil(q/pmax);
    if(!isfinite(nc) || nc>1000000) {
        fail(failed,error,"Direct SIDM pair needs more than one million exposure subdivisions"); return 0;
    }
    const uint64_t n=(uint64_t)nc;
    const double probability=n==1?q:q/n;
#ifdef SIDMDEBUG
    if(probability>1 || probability>pmax*(1+16*DBL_EPSILON)) {
        fail(failed,error,"Invalid direct SIDM subdivided probability"); return 0;
    }
    s->trials+=n; s->subdivided_pairs+=n>1;
    s->max_trial_probability=fmax(s->max_trial_probability,probability);
#endif
    if(cached && n==1 && cached->draw>=probability) return 0;
    seed=sidm_random_mix(seed^sidm_random_mix(bid+UINT64_C(0xd1b54a32d192ed03)));
    int changed=0;
#ifdef SIDMDEBUG
    int touched=0;
#else
    (void)j;(void)p;(void)ai;(void)bi;
#endif
    for(uint64_t k=0;k<n;k++) {
        if(uniform(&seed)>=probability) continue;
#ifdef SIDMDEBUG
        if(!touched) {
            if(ai>=0) touch(j,p,ai,failed,error);
            touch(j,p,bi,failed,error);
            if(*failed) return 0;
            touched=1;
        }
#endif
        changed=1;
        const double z=2*uniform(&seed)-1,phi=2*acos(-1.0)*uniform(&seed);
        const double r=sqrt(fmax(0,1-z*z)),dir[3]={r*cos(phi),r*sin(phi),z};
        for(int d=0;d<3;d++) {
            const double center=.5*a[d]+.5*b[d],dv=.5*u*dir[d];
            a[d]=center+dv; b[d]=center-dv;
            if(!isfinite(a[d]) || !isfinite(b[d])) fail(failed,error,"Nonfinite direct SIDM scattered velocity");
        }
#ifdef SIDMDEBUG
        s->events++;
#endif
    }
    return changed;
}

static int prepare_sources(struct sidm_direct_particles p,int nlocal,
    struct sidm_direct_source *src,int ns,struct sidm_journal *j,
    int rank,int ntask,MPI_Comm comm,char error[256])
{
#ifdef SIDMDEBUG
    int failed=0;
#else
    (void)p;(void)nlocal;(void)rank;(void)ntask;(void)comm;(void)error;
#endif
    for(int a=0;a<ns;a++) {
        struct sidm_direct_source *s=&src[a]; s->max_sigma_v=0;
#ifdef SIDMDEBUG
        if(s->index<0 || s->index>=nlocal || s->n<0 || s->n>SIDM_DIRECT_MAX_NEIGHBORS ||
           (s->n&&!s->neighbors) || !isfinite(s->h) || s->h<=0 || !isfinite(s->dt) || s->dt<0) {
            fail(&failed,error,"Invalid direct SIDM source"); continue;
        }
        if(j->marked[s->index]) fail(&failed,error,"Duplicate direct SIDM source index");
        j->marked[s->index]=1;
#endif
    }
#ifdef SIDMDEBUG
    /* Validation markers are not journal entries. */
    for(int a=0;a<ns;a++) if(src[a].index>=0&&src[a].index<nlocal) j->marked[src[a].index]=0;
    if(collective_error(failed,error,comm)) return -1;
    int bad_neighbors=0;
#pragma omp parallel for reduction(|:bad_neighbors)
#else
#pragma omp parallel for
#endif
    for(int a=0;a<ns;a++) {
        struct sidm_direct_source *s=&src[a];
        qsort(s->neighbors,(size_t)s->n,sizeof(*s->neighbors),neighbor_order);
        for(int k=0;k<s->n;k++) {
            const struct sidm_direct_neighbor *e=&s->neighbors[k];
#ifdef SIDMDEBUG
            if(e->owner<0||e->owner>=ntask||e->index<0||e->id==particle_id(p,s->index)||
               !isfinite(e->r2)||e->r2<0 || (k&&e->id==s->neighbors[k-1].id&&e->owner==s->neighbors[k-1].owner) ||
               (e->owner==rank&&(e->index>=nlocal||particle_id(p,e->index)!=e->id))) {
                bad_neighbors=1; continue;
            }
#endif
            const double coefficient=.5*j->validated_mass*sidm_rate_cubic_kernel(sqrt(e->r2),s->h)*s->dt;
#ifdef SIDMDEBUG
            if(!isfinite(coefficient) || coefficient<0) bad_neighbors=1;
#endif
            s->neighbors[k].coefficient=coefficient;
        }
    }
#ifdef SIDMDEBUG
    if(bad_neighbors) fail(&failed,error,"Invalid/duplicate direct SIDM neighbor identity or geometry");
    if(collective_error(failed,error,comm)) return -1;
#endif
    return 0;
}

#ifdef SIDMDEBUG
static void audit_journal(struct sidm_direct_particles p,const struct sidm_journal *j,
    const struct sidm_direct_stats *local,struct sidm_direct_stats *stats,
    MPI_Comm comm,int *failed,char error[256])
{
    /* Audit the complete endpoint journal, including source-free receivers. */
    long double moments[6]={0};
    for(int k=0;k<j->count;k++) {
        const struct sidm_journal_entry *e=&j->entries[k];
        for(int d=0;d<3;d++) {
            const long double a=e->before[d],b=velocity(p,e->index)[d],m=j->validated_mass;
            moments[d]+=m*(b-a); moments[3]+=.5L*m*(b-a)*(b+a);
            moments[4]+=.5L*m*a*a;
            if(!isfinite(b)) fail(failed,error,"Nonfinite direct SIDM final velocity");
        }
        moments[5]+=j->validated_mass;
    }
    MPI_Allreduce(MPI_IN_PLACE,moments,6,MPI_LONG_DOUBLE,MPI_SUM,comm);
    uint64_t counts[]={local->pairs,local->trials,local->events,local->subdivided_pairs,local->source_packets,local->bytes,(uint64_t)j->count,
        local->threaded_pairs,local->recomputed_pairs};
    MPI_Allreduce(MPI_IN_PLACE,counts,9,MPI_UINT64_T,MPI_SUM,comm);
    stats->pairs=counts[0]; stats->trials=counts[1]; stats->events=counts[2];
    stats->subdivided_pairs=counts[3]; stats->source_packets=counts[4]; stats->bytes=counts[5]; stats->journaled_particles=counts[6];
    stats->threaded_pairs=counts[7];stats->recomputed_pairs=counts[8];
    double maxima[]={local->max_probability,local->max_trial_probability};
    MPI_Allreduce(MPI_IN_PLACE,maxima,2,MPI_DOUBLE,MPI_MAX,comm);
    stats->max_probability=maxima[0]; stats->max_trial_probability=maxima[1];
    MPI_Allreduce(&local->sum_probability,&stats->sum_probability,1,MPI_DOUBLE,MPI_SUM,comm);
    stats->delta_k=moments[3];
    for(int d=0;d<3;d++) stats->delta_p=fmaxl(stats->delta_p,fabsl(moments[d]));
    if(!isfinite(moments[3])||fabsl(moments[3])>1e-10L*fmaxl(moments[4],LDBL_MIN)||
       stats->delta_p>1e-10L*fmaxl(sqrtl(2*moments[5]*moments[4]),LDBL_MIN))
        fail(failed,error,"Direct SIDM collision conservation check failed");
}
#endif

static int phase_partner(int rank,int nrank,int phase)
{
    /* Round-robin pairing: hold the last slot fixed and rotate the others.
     * Odd rank counts add a dummy slot, giving each real rank one bye. */
    const int ring=nrank+(nrank%2)-1;
    const int position=rank==ring?ring:(rank-phase+ring)%ring;
    const int other=ring-position;
    return other==ring?ring:(other+phase)%ring;
}

int sidm_direct_scatter_view(struct sidm_direct_particles p,int nlocal,
    struct sidm_direct_source *src,int ns,double sigma0,double w,int vd,double pmax,
    MPI_Comm comm,struct sidm_journal *j,struct sidm_direct_stats *stats,char error[256])
{
#ifdef SIDMDEBUG
    double timer=MPI_Wtime();
#endif
    int rank,ntask,failed=0,status=-1;
    int *count=NULL,*start=NULL,*cursor=NULL,*incoming=NULL;
    struct group *groups=NULL;
    unsigned char *buffer=Workspace.buffer;
    struct reply *returned=Workspace.returned;
    struct sidm_direct_stats local={0};
    const int threaded=omp_get_max_threads()>1;
    MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&ntask);
    memset(error,0,256); if(stats) memset(stats,0,sizeof(*stats));
#ifdef SIDMDEBUG
    if(nlocal<0 || ns<0 || (nlocal&&!p.base) || (ns&&!src) || !j || !stats ||
       !isfinite(sigma0) || sigma0<0 || (vd!=0&&vd!=1) || (vd&&(!isfinite(w)||w<=0)) ||
       !(pmax>0&&pmax<=1) || !(j->validated_mass>0) || !isfinite(j->validated_mass) ||
       j->capacity<nlocal || j->count<0 || j->count>nlocal || (nlocal&&(!j->entries||!j->marked)))
        fail(&failed,error,"Invalid direct SIDM input, journal, or parameters");
    if(p.stride<sizeof(uint64_t) || p.stride<3*sizeof(double) ||
       p.id_offset>p.stride-sizeof(uint64_t) || p.velocity_offset>p.stride-3*sizeof(double) ||
       p.stride%sizeof(double) || p.id_offset%sizeof(uint64_t) || p.velocity_offset%sizeof(double) ||
       (uintptr_t)p.base%sizeof(double))
        fail(&failed,error,"Invalid direct SIDM particle view");
    if(collective_error(failed,error,comm)) goto cleanup;
    double params[]={sigma0,vd?w:0,vd,pmax,j->validated_mass},lo[5],hi[5];
    MPI_Allreduce(params,lo,5,MPI_DOUBLE,MPI_MIN,comm);
    MPI_Allreduce(params,hi,5,MPI_DOUBLE,MPI_MAX,comm);
    for(int k=0;k<5;k++) if(lo[k]!=hi[k]) fail(&failed,error,"Inconsistent direct SIDM parameters across ranks");
    for(int k=0;k<j->count;k++) {
        const int i=j->entries[k].index;
        j->marked[i]=0;
    }
    j->count=0;
#endif
    if(reserve_routing(ntask)) fail(&failed,error,"Direct SIDM routing allocation failed");
    if(collective_error(failed,error,comm)) goto cleanup;
    count=Workspace.routing; start=count+ntask; cursor=start+ntask+1; incoming=cursor+ntask;
    memset(count,0,(size_t)ntask*sizeof(*count)); start[0]=0;
    int64_t ng=0;
    if(prepare_sources(p,nlocal,src,ns,j,rank,ntask,comm,error)) goto cleanup;
    PROFILE(stats->prepare_seconds=MPI_Wtime()-timer; timer=MPI_Wtime());
    for(int a=0;a<ns;a++) for(int k=0;k<src[a].n;k++)
        if(!k || src[a].neighbors[k].owner!=src[a].neighbors[k-1].owner) {
            count[src[a].neighbors[k].owner]++; ng++;
        }
    if(ng>INT_MAX) fail(&failed,error,"Direct SIDM group count exceeds index range");
    /* Keep the population-sized groups transient; only bounded communication
     * buffers and small routing arrays persist between visits. Allocation and
     * size failures share one collective before any rank exchanges counts. */
    if(!failed) groups=malloc((size_t)(ng?ng:1)*sizeof(*groups));
    if(!groups) fail(&failed,error,"Direct SIDM group allocation failed");
    if(threaded && reserve_cache(nlocal)) fail(&failed,error,"Direct SIDM evaluation cache allocation failed");
    if(collective_error(failed,error,comm)) goto cleanup;
    for(int t=0;t<ntask;t++) start[t+1]=start[t]+count[t];
    memcpy(cursor,start,(size_t)ntask*sizeof(*cursor));
    MPI_Alltoall(count,1,MPI_INT,incoming,1,MPI_INT,comm);
#ifdef SIDMDEBUG
    local.bytes=(uint64_t)(ntask-1)*sizeof(int);
#endif
    for(int a=0;a<ns;a++) for(int k=0;k<src[a].n;) {
        const int first=k,owner=src[a].neighbors[k].owner;
        while(k<src[a].n&&src[a].neighbors[k].owner==owner) k++;
        groups[cursor[owner]++]=(struct group){a,first,k-first};
    }
    PROFILE(stats->route_seconds=MPI_Wtime()-timer; timer=MPI_Wtime());
    /* Local updates precede remote phases; no MPI or ghost storage needed. */
    for(int offset=start[rank];offset<start[rank+1];offset+=SIDM_DIRECT_EVAL_SOURCES) {
        const int batch=start[rank+1]-offset<SIDM_DIRECT_EVAL_SOURCES?start[rank+1]-offset:SIDM_DIRECT_EVAL_SOURCES;
        const int parallel=threaded && batch>=16;
        const unsigned epoch=parallel?cache_epoch():0;
        if(parallel) {
#ifdef SIDMDEBUG
            const double before=MPI_Wtime();
#endif
#pragma omp parallel for
            for(int b=0;b<batch;b++) {
                const struct group *g=&groups[offset+b];
                const struct sidm_direct_source *s=&src[g->source];
                const uint64_t seed=s->seed^sidm_random_mix(particle_id(p,s->index));
                for(int k=0;k<g->count;k++) {
                    const struct sidm_direct_neighbor *e=&s->neighbors[g->first+k];
                    Cache.pairs[b*SIDM_DIRECT_MAX_NEIGHBORS+k]=evaluate_pair(
                        velocity(p,s->index),velocity(p,e->index),e->id,seed,sigma0,w,vd);
                }
            }
            PROFILE(stats->predict_seconds+=MPI_Wtime()-before);
        }
        for(int b=0;b<batch;b++) {
            const struct group *g=&groups[offset+b];struct sidm_direct_source *s=&src[g->source];
            const uint64_t seed=s->seed^sidm_random_mix(particle_id(p,s->index));
            for(int k=0;k<g->count;k++) {
                const struct sidm_direct_neighbor *e=&s->neighbors[g->first+k];
                const struct pair_evaluation *cached=parallel && Cache.changed[s->index]!=epoch &&
                    Cache.changed[e->index]!=epoch?&Cache.pairs[b*SIDM_DIRECT_MAX_NEIGHBORS+k]:NULL;
#ifdef SIDMDEBUG
                local.threaded_pairs+=parallel;
                local.recomputed_pairs+=parallel && !cached;
#endif
                if(scatter_pair(velocity(p,s->index),velocity(p,e->index),e->id,seed,
                    e->coefficient,sigma0,w,vd,pmax,j,p,s->index,e->index,&s->max_sigma_v,&local,&failed,error,cached) && parallel)
                    Cache.changed[s->index]=Cache.changed[e->index]=epoch;
            }
        }
    }
    PROFILE(stats->pair_seconds=MPI_Wtime()-timer);
    for(int phase=0;phase<ntask+(ntask%2)-1;phase++) {
        const int other=phase_partner(rank,ntask,phase);
        if(other>=ntask || (!count[other]&&!incoming[other])) continue;
        /* Reverse direction is packed only after all returns from forward. */
        for(int direction=0;direction<2;direction++) {
            const int sending=(rank<other)==(direction==0);
            const int total=sending?count[other]:incoming[other];
            for(int offset=0;offset<total;offset+=SIDM_DIRECT_WIRE_SOURCES) {
                const int batch=total-offset<SIDM_DIRECT_WIRE_SOURCES?total-offset:SIDM_DIRECT_WIRE_SOURCES;
                int bytes=0; PROFILE(timer=MPI_Wtime());
                if(sending) for(int b=0;b<batch;b++) {
                    const struct group *g=&groups[start[other]+offset+b];
                    const struct sidm_direct_source *s=&src[g->source];
                    struct packet packet={.seed=s->seed^sidm_random_mix(particle_id(p,s->index)),.n=g->count};
                    memcpy(packet.v,velocity(p,s->index),sizeof(packet.v));
                    memcpy(buffer+bytes,&packet,sizeof(packet)); bytes+=sizeof(packet);
                    for(int k=g->first;k<g->first+g->count;k++) {
                        const struct sidm_direct_neighbor *e=&s->neighbors[k];
                        struct target target={.coefficient=e->coefficient,.index=e->index};
#ifdef SIDMDEBUG
                        target.id=e->id;
#endif
                        memcpy(buffer+bytes,&target,sizeof(target)); bytes+=sizeof(target);
                    }
                }
                PROFILE(stats->pack_seconds+=MPI_Wtime()-timer; timer=MPI_Wtime());
                /* The receiver already has the bounded buffer. Receive up to
                 * its capacity to avoid a separate message-size handshake. */
                MPI_Sendrecv(sending?buffer:NULL,sending?bytes:0,MPI_BYTE,other,19203,
                    sending?NULL:buffer,sending?0:WIRE_BYTES,MPI_BYTE,other,19203,comm,MPI_STATUS_IGNORE);
                PROFILE(stats->mpi_seconds+=MPI_Wtime()-timer; timer=MPI_Wtime());
                if(!sending) {
                    size_t at=0;
                    for(int first=0;first<batch;first+=SIDM_DIRECT_EVAL_SOURCES) {
                        const int n=batch-first<SIDM_DIRECT_EVAL_SOURCES?batch-first:SIDM_DIRECT_EVAL_SOURCES;
                        const int parallel=threaded && n>=16;
                        const unsigned epoch=parallel?cache_epoch():0;
                        if(parallel) {
                            /* Prediction leaves the ordered collision cursor unchanged. */
                            size_t offsets[SIDM_DIRECT_EVAL_SOURCES],next_packet=at;
#ifdef SIDMDEBUG
                            const double before=MPI_Wtime();
#endif
                            for(int b=0;b<n;b++) {
                                struct packet a;
                                memcpy(&a,buffer+next_packet,sizeof(a));
                                offsets[b]=next_packet;
                                next_packet+=sizeof(a)+(size_t)a.n*sizeof(struct target);
                            }
#pragma omp parallel for
                            for(int b=0;b<n;b++) {
                                struct packet a;size_t pos=offsets[b];
                                memcpy(&a,buffer+pos,sizeof(a));pos+=sizeof(a);
                                for(int k=0;k<a.n;k++) {
                                    struct target e;memcpy(&e,buffer+pos,sizeof(e));pos+=sizeof(e);
#ifdef SIDMDEBUG
                                    if(e.index<0||e.index>=nlocal||particle_id(p,e.index)!=e.id) continue;
#endif
                                    Cache.pairs[b*SIDM_DIRECT_MAX_NEIGHBORS+k]=evaluate_pair(
                                        a.v,velocity(p,e.index),particle_id(p,e.index),a.seed,sigma0,w,vd);
                                }
                            }
                            PROFILE(stats->predict_seconds+=MPI_Wtime()-before);
                        }
                        for(int b=0;b<n;b++) {
                            struct packet a; memcpy(&a,buffer+at,sizeof(a)); at+=sizeof(a);
                            double max_sv=0;int source_changed=0;
                            for(int k=0;k<a.n;k++) {
                                struct target e; memcpy(&e,buffer+at,sizeof(e)); at+=sizeof(e);
#ifdef SIDMDEBUG
                                if(e.index<0||e.index>=nlocal||particle_id(p,e.index)!=e.id) {
                                    fail(&failed,error,"Direct SIDM remote index/ID mismatch"); continue;
                                }
#endif
                                const struct pair_evaluation *cached=parallel && !source_changed && Cache.changed[e.index]!=epoch?
                                    &Cache.pairs[b*SIDM_DIRECT_MAX_NEIGHBORS+k]:NULL;
#ifdef SIDMDEBUG
                                local.threaded_pairs+=parallel;
                                local.recomputed_pairs+=parallel && !cached;
#endif
                                if(scatter_pair(a.v,velocity(p,e.index),particle_id(p,e.index),a.seed,e.coefficient,
                                    sigma0,w,vd,pmax,j,p,-1,e.index,&max_sv,&local,&failed,error,cached) && parallel) {
                                    source_changed=1;Cache.changed[e.index]=epoch;
                                }
                            }
                            memcpy(returned[first+b].v,a.v,sizeof(a.v)); returned[first+b].max_sigma_v=max_sv;
                        }
                    }
                }
                PROFILE(stats->pair_seconds+=MPI_Wtime()-timer; timer=MPI_Wtime());
                MPI_Sendrecv(sending?NULL:returned,sending?0:batch*(int)sizeof(*returned),MPI_BYTE,other,19204,
                    sending?returned:NULL,sending?batch*(int)sizeof(*returned):0,MPI_BYTE,other,19204,comm,MPI_STATUS_IGNORE);
                PROFILE(stats->mpi_seconds+=MPI_Wtime()-timer; timer=MPI_Wtime());
                if(sending) {
#ifdef SIDMDEBUG
                    local.source_packets+=batch;
                    local.bytes+=bytes+batch*sizeof(*returned);
#endif
                    for(int b=0;b<batch;b++) {
                        struct sidm_direct_source *s=&src[groups[start[other]+offset+b].source];
                        double *v=velocity(p,s->index);
                        if(memcmp(v,returned[b].v,sizeof(returned[b].v))) {
#ifdef SIDMDEBUG
                            touch(j,p,s->index,&failed,error);
#endif
                            memcpy(v,returned[b].v,sizeof(returned[b].v));
                        }
                        s->max_sigma_v=fmax(s->max_sigma_v,returned[b].max_sigma_v);
                    }
                }
                PROFILE(stats->pack_seconds+=MPI_Wtime()-timer);
            }
        }
    }
#ifdef SIDMDEBUG
    timer=MPI_Wtime();
    audit_journal(p,j,&local,stats,comm,&failed,error);
#endif
    status=collective_error(failed,error,comm);
    PROFILE(stats->audit_seconds=MPI_Wtime()-timer);
cleanup:
    free(groups);
    return status;
}

/* Compact-array compatibility for independent operator tests. Production uses
 * the native particle view; no scratch particle array or copyback is needed. */
int sidm_direct_scatter(struct sidm_particle *p,int nlocal,
    struct sidm_direct_source *src,int ns,double sigma0,double w,int vd,double pmax,
    MPI_Comm comm,struct sidm_journal *j,struct sidm_direct_stats *stats,char error[256])
{
    const struct sidm_direct_particles view={p,sizeof(*p),
        offsetof(struct sidm_particle,id),offsetof(struct sidm_particle,v)};
    return sidm_direct_scatter_view(view,nlocal,src,ns,sigma0,w,vd,pmax,comm,j,stats,error);
}
