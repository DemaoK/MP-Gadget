/* Independent serial execution oracle for the directed rank-pair schedule.
 * This test gathers states; the production module never does. */
#include "../sidm-direct.h"
#include "../sidm-rate.h"
#include "../sidm-random.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SIDM_DIRECT_TEST_N
#define SIDM_DIRECT_TEST_N 401
#endif
#define N SIDM_DIRECT_TEST_N
#define K 7
static int rank,tasks,owners;
static double exposure_scale=1;
static int owner(int id) { return id%owners; }
static int index_of(int id) { return id/owners; }
static int source(int id) { return id%5!=0; }
static uint64_t hash(uint64_t h,const void *data,size_t size)
{
    const unsigned char *p=data;
    for(size_t i=0;i<size;i++) h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
static uint64_t mix(uint64_t x)
{
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
static double draw(uint64_t *x) { *x=mix(*x); return (*x>>11)*0x1.0p-53; }
static int is_target(int i,int j) { int d=(j-i+N)%N; return d>0&&d<=K; }
static double separation(int i,int j) { return .02*((j-i+N)%N); }
static double exposure(int id,int zero) { return zero?0:exposure_scale*1.2*(1+id%3); }
static void initial(int id,double v[3],int boost)
{
    const double shift[]={1024,-512,256};
    for(int d=0;d<3;d++) v[d]=.125*((id*(d+3)+d*7)%31-15)+(boost?shift[d]:0);
}
static void oracle_group(int sender,int receiver,double v[N][3],double rate[N],
    double pmax,int vd,int zero,struct sidm_direct_stats *st)
{
    for(int i=0;i<N;i++) if(source(i)&&owner(i)==sender)
        for(int j=0;j<N;j++) if(owner(j)==receiver&&is_target(i,j)) {
            const double u=hypot(hypot(v[i][0]-v[j][0],v[i][1]-v[j][1]),v[i][2]-v[j][2]);
            const double sv=.3*u*(vd?sidm_rate_viscosity_shape(u*u/4):1);
            const double q=.5*sidm_rate_cubic_kernel(separation(i,j),1)*exposure(i,zero)*sv;
            rate[i]=fmax(rate[i],sv); st->pairs++; st->sum_probability+=q;
            st->max_probability=fmax(st->max_probability,q);
            if(q==0) continue;
            const int n=(int)ceil(q/pmax);
            st->trials+=n; st->subdivided_pairs+=n>1;
            st->max_trial_probability=fmax(st->max_trial_probability,q/n);
            uint64_t seed=mix(17^mix(i+1)^mix(j+1+UINT64_C(0xd1b54a32d192ed03)));
            for(int k=0;k<n;k++) if(draw(&seed)<q/n) {
                const double z=2*draw(&seed)-1,phi=2*acos(-1.0)*draw(&seed);
                const double r=sqrt(fmax(0,1-z*z)),unit[]={r*cos(phi),r*sin(phi),z};
                for(int d=0;d<3;d++) {
                    double c=.5*v[i][d]+.5*v[j][d];
                    v[i][d]=c+.5*u*unit[d]; v[j][d]=c-.5*u*unit[d];
                }
                st->events++;
            }
        }
}
static void oracle(double v[N][3],double rate[N],double pmax,int vd,int zero,struct sidm_direct_stats *st)
{
    for(int r=0;r<tasks;r++) oracle_group(r,r,v,rate,pmax,vd,zero,st);
    /* Circle tournament expressed as a rotating rank array, independent of
     * the production phase_partner expression. Local rank groups commute. */
    const int nt=tasks+(tasks%2);
    int *ring=malloc(nt*sizeof(*ring)); for(int r=0;r<nt;r++) ring[r]=r;
    for(int phase=0;phase<nt-1;phase++) {
        for(int r=0;r<nt/2;r++) {
            int a=ring[r],b=ring[nt-1-r];
            if(a>=tasks||b>=tasks) continue;
            if(a>b) { int t=a;a=b;b=t; }
            oracle_group(a,b,v,rate,pmax,vd,zero,st);
            oracle_group(b,a,v,rate,pmax,vd,zero,st);
        }
        const int first=ring[0];
        for(int r=0;r<nt-2;r++) ring[r]=ring[r+1];
        ring[nt-2]=first;
    }
    free(ring);
}
static void close_number(double a,double b)
{
    if(fabs(a-b)>2e-10*(1+fabs(b))) {
        fprintf(stderr,"rank %d: %.17g != %.17g\n",rank,a,b); MPI_Abort(MPI_COMM_WORLD,2);
    }
}
static void check(double pmax,int vd,int boost,int zero,int negative)
{
    /* Nontrivial native stride and gaps around the fields. All cases use the
     * same view interface as production and preserve unrelated fields. */
    struct native { double unused[7],v[3]; uint64_t guard,id; } p[N];
    struct sidm_direct_source s[N];
    struct sidm_direct_neighbor ngb[N][K];
    struct sidm_journal_entry entries[N]; unsigned char marked[N]={0};
    struct sidm_journal journal={.entries=entries,.marked=marked,.capacity=N,.validated_mass=1};
    int n=0,ns=0;
    for(int id=0;id<N;id++) if(owner(id)==rank) {
        memset(&p[n],0x5a,sizeof(p[n]));
        p[n].id=id+1;
        initial(id,p[n].v,boost);
        assert(index_of(id)==n);
        if(source(id)) {
            s[ns]=(struct sidm_direct_source){.index=n,.n=K,.neighbors=ngb[ns],.h=1,.exposure=exposure(id,zero),.seed=17};
            /* Reverse order forces the production group sort. */
            for(int k=0;k<K;k++) {
                int j=(id+K-k)%N;double r=separation(id,j);
                ngb[ns][k]=(struct sidm_direct_neighbor){.r2=r*r,.id=j+1,.owner=owner(j),.index=index_of(j)};
            }
            ns++;
        }
        n++;
    }
    if(negative==1&&rank==0&&ns) s[0].neighbors[0].index=N+1;
    if(negative==2&&rank==0&&ns) s[0].exposure=-1;
    if(negative==3&&rank==0&&ns) s[0].neighbors[1]=s[0].neighbors[0];
    if(negative==4&&rank==0&&ns) s[0].exposure=DBL_MAX;
    struct sidm_direct_stats actual,expected={0}; char error[256];
    const struct sidm_direct_particles view={p,sizeof(*p),
        offsetof(struct native,id),offsetof(struct native,v)};
    int code=sidm_direct_scatter(view,n,s,ns,.3,2,vd,pmax,MPI_COMM_WORLD,&journal,&actual,error);
    for(int i=0;i<n;i++) assert(p[i].guard==UINT64_C(0x5a5a5a5a5a5a5a5a));
    if(negative) { assert(code!=0);if(!rank)printf("PASS direct negative %d: %s\n",negative,error);return; }
    if(code) { fprintf(stderr,"%s\n",error);MPI_Abort(MPI_COMM_WORLD,1); }
    double ref[N][3],rate[N]={0}; for(int i=0;i<N;i++) initial(i,ref[i],boost);
    oracle(ref,rate,pmax,vd,zero,&expected);
    for(int i=0;i<n;i++) for(int d=0;d<3;d++) close_number(p[i].v[d],ref[p[i].id-1][d]);
    for(int a=0;a<ns;a++) close_number(s[a].max_sigma_v,rate[p[s[a].index].id-1]);
#ifdef SIDMDEBUG
    assert(actual.pairs==expected.pairs&&actual.events==expected.events&&actual.trials==expected.trials);
    assert(actual.subdivided_pairs==expected.subdivided_pairs);
    close_number(actual.max_probability,expected.max_probability);
    close_number(actual.sum_probability,expected.sum_probability);
    assert(actual.max_trial_probability<=pmax*(1+1e-14));
    /* Unchanged candidates are not journaled. Every changed endpoint has its
     * original state, including endpoints only used as remote receivers. */
    int touched=0;
    for(int i=0;i<n;i++) {
        double original[3]; initial(p[i].id-1,original,boost);
        if(memcmp(original,p[i].v,sizeof(original))) assert(marked[i]);
        touched+=marked[i]!=0;
    }
    assert(touched==journal.count);
    assert((uint64_t)journal.count<=2*expected.events);
    if(zero) assert(journal.count==0);
    for(int k=0;k<journal.count;k++) {
        double original[3]; initial(p[entries[k].index].id-1,original,boost);
        assert(!memcmp(entries[k].before,original,sizeof(original)));
    }
    if(!zero && exposure_scale==1) assert(actual.events>0);
    if(!zero&&!vd&&pmax<1&&exposure_scale==1) assert(actual.subdivided_pairs>0&&actual.max_probability>1);
    if(tasks==1) assert(actual.source_packets==0&&actual.bytes==0);
    else assert(actual.source_packets<actual.pairs);
#endif
    /* A second identical initial state with a common boost must retain the
     * same trajectory up to velocity arithmetic roundoff. */
    if(boost) {
        struct sidm_direct_stats unboosted={0};double v[N][3],r[N]={0};
        for(int i=0;i<N;i++) initial(i,v[i],0);
        oracle(v,r,pmax,vd,zero,&unboosted);
        const double shift[]={1024,-512,256};
        assert(unboosted.events==expected.events);
        for(int i=0;i<N;i++) for(int d=0;d<3;d++) close_number(ref[i][d]-shift[d],v[i][d]);
    }
    /* Exact output fingerprints compare builds/thread counts without relaxing
     * the independent oracle's floating tolerance. Includes changed receivers
     * and source rates after shared endpoints and subdivided exposures. */
    uint64_t fingerprint=0;
    for(int i=0;i<n;i++) fingerprint=hash(fingerprint,p[i].v,sizeof(p[i].v));
    for(int a=0;a<ns;a++) {
        fingerprint=hash(fingerprint,&s[a].max_sigma_v,sizeof(double));
        fingerprint=hash(fingerprint,&s[a].max_relative_speed,sizeof(double));
    }
    uint64_t hashes[8];MPI_Gather(&fingerprint,1,MPI_UINT64_T,hashes,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(!rank) {
        printf("STATE owners=%d p=%g vd=%d boost=%d zero=%d",owners,pmax,vd,boost,zero);
        if(exposure_scale!=1) printf(" exposure_scale=%g",exposure_scale);
        for(int r=0;r<tasks;r++) printf(" %016llx",(unsigned long long)hashes[r]);
        puts("");
    }
    /* No sources on ANY rank, reusing a populated journal. */
    assert(!sidm_direct_scatter(view,n,NULL,0,.3,2,vd,pmax,MPI_COMM_WORLD,&journal,&actual,error));
    assert(actual.pairs==0&&actual.events==0&&journal.count==0);
    if(!rank)printf("PASS direct ranks=%d owners=%d p=%g vd=%d boost=%d zero=%d events=%llu subdivisions=%llu\n",
        tasks,owners,pmax,vd,boost,zero,(unsigned long long)expected.events,(unsigned long long)expected.subdivided_pairs);
}
/* Many independent endpoint sets plus short dependency chains exercise the
 * concurrent executor, including nonzero exposure and subdivision. Each rank
 * sends its groups to the next rank, so both local and wire execution are used. */
static void check_waves(void)
{
    enum { GROUPS=128, WIDTH=4, LOCAL=GROUPS*(WIDTH+1) };
    struct particle { uint64_t id; double v[3]; } p[LOCAL];
    struct sidm_direct_source s[GROUPS];
    struct sidm_direct_neighbor neighbors[GROUPS][WIDTH];
    struct sidm_journal_entry entries[LOCAL];unsigned char marked[LOCAL]={0};
    struct sidm_journal journal={.entries=entries,.marked=marked,.capacity=LOCAL,.validated_mass=1};
    double reference[LOCAL][3],reference_sv[GROUPS],reference_u[GROUPS];
    const int threads=omp_get_max_threads();
    uint64_t fingerprint=0;
    for(int run=0;run<4;run++) {
        const int threaded=run%2,zero=run>=2;
        omp_set_num_threads(threaded?(threads>1?threads:4):1);
        for(int i=0;i<LOCAL;i++) {
            p[i].id=(uint64_t)rank*LOCAL+i+1;
            initial(i,p[i].v,0);
        }
        for(int b=0;b<GROUPS;b++) {
            s[b]=(struct sidm_direct_source){.index=b,.n=WIDTH,.neighbors=neighbors[b],.h=1,.exposure=zero?0:2,.seed=13};
            for(int k=0;k<WIDTH;k++) {
                /* Four successive groups share receivers; their dependencies
                 * must be respected while separate families run concurrently. */
                int target=GROUPS+(b/4)*WIDTH+k;
                const int destination=(rank+1)%tasks;
                neighbors[b][k]=(struct sidm_direct_neighbor){.r2=.01,.id=(uint64_t)destination*LOCAL+target+1,.owner=destination,.index=target};
            }
        }
        struct sidm_direct_stats stats;char error[256];
        const struct sidm_direct_particles view={p,sizeof(*p),offsetof(struct particle,id),offsetof(struct particle,v)};
        assert(!sidm_direct_scatter(view,LOCAL,s,GROUPS,.3,2,0,.1,MPI_COMM_WORLD,&journal,&stats,error));
        for(int i=0;i<LOCAL;i++) {
            if(!threaded) memcpy(reference[i],p[i].v,sizeof(reference[i]));
            else assert(!memcmp(reference[i],p[i].v,sizeof(reference[i])));
        }
        for(int b=0;b<GROUPS;b++) {
            assert(s[b].max_relative_speed>0);
            if(!threaded) {reference_sv[b]=s[b].max_sigma_v;reference_u[b]=s[b].max_relative_speed;}
            else {assert(s[b].max_sigma_v==reference_sv[b]);assert(s[b].max_relative_speed==reference_u[b]);}
        }
#ifdef SIDMDEBUG
        if(!zero) assert(stats.events>0 && stats.subdivided_pairs>0);
        else assert(stats.events==0 && journal.count==0);
        if(threaded) assert(stats.threaded_pairs==stats.pairs);
#endif
        if(threaded) {
            for(int i=0;i<LOCAL;i++) fingerprint=hash(fingerprint,p[i].v,sizeof(p[i].v));
            for(int b=0;b<GROUPS;b++) {
                fingerprint=hash(fingerprint,&s[b].max_sigma_v,sizeof(double));
                fingerprint=hash(fingerprint,&s[b].max_relative_speed,sizeof(double));
            }
        }
    }
    omp_set_num_threads(threads);
    uint64_t hashes[8];MPI_Gather(&fingerprint,1,MPI_UINT64_T,hashes,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(!rank) {
        printf("STATE waves");
        for(int r=0;r<tasks;r++) printf(" %016llx",(unsigned long long)hashes[r]);
        puts("\nPASS direct concurrent waves");
    }
}
int main(int argc,char **argv)
{
    MPI_Init(&argc,&argv); MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&tasks);
    assert(tasks<=8);
    const uint64_t epochs[]={0,2,UINT64_C(1)<<46,UINT64_MAX};
    for(unsigned i=0;i<sizeof(epochs)/sizeof(*epochs);i++)
        assert(sidm_event_seed(71,epochs[i])==mix((71^epochs[i])^mix(epochs[i])));
    for(int empty=0;empty<(tasks>1?2:1);empty++) {
        owners=tasks-empty;
        for(int vd=0;vd<2;vd++) {
            check(.1,vd,0,0,0); /* Strong subdivision. */
            check(.5,vd,0,0,0); /* Production probability setting. */
            check(.5,vd,1,0,0); /* Bulk boost. */
        }
        check(1,0,0,0,0); /* Allowed probability endpoint. */
        check(.5,0,0,1,0);
        /* Weak exposures exercise rejecting groups before the first event,
         * and fresh recomputation after another source changes an endpoint. */
        const double weak[]={.0001,.01};
        for(unsigned i=0;i<sizeof(weak)/sizeof(*weak);i++) {
            exposure_scale=weak[i];
            for(int vd=0;vd<2;vd++) check(.5,vd,0,0,0);
        }
        exposure_scale=1;
#ifdef SIDMDEBUG
        for(int bad=1;bad<=3;bad++) check(.5,0,0,0,bad);
#endif
        check(.5,0,0,0,4);
    }
    check_waves();
    if(!rank) puts("PASS direct SIDM");
    MPI_Finalize();return 0;
}
