/* Production support walks and direct collisions: independent geometry/rate
 * oracle, inactive receivers, conservation and export-retry equivalence. */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stub.h"
#ifdef SIDM
#include "../domain.h"
#include "../sidm-local.h"
#include "../treewalk.h"
#include "../walltime.h"
static void capture_walk(TreeWalk *,int *,size_t);
#define treewalk_run capture_walk
#include "../sidm-distributed.c"
#undef treewalk_run
#define TEST_N 512
static struct ClockTable clocks;
static uint64_t exported_batches;
static int velocity_dependent;
static void capture_walk(TreeWalk *tw,int *active,size_t nactive)
{ treewalk_run(tw,active,nactive);exported_batches+=tw->Nexportfull; }
static void fixture(struct particle_data *p, int id)
{
    memset(p,0,sizeof(*p));
    p->ID=id+1; p->Type=1; p->Mass=1;
    p->TimeBinGravity=p->TimeBinHydro=1;
    /* Two distinct IDs coincide; the nearest-neighbor solve excludes self by
     * identity, and the deliberately uneven edge spacing crosses the box. */
    const int k=id==1 ? 0 : id;
    p->Pos[0]=.12+1.10*(k%8)+.025*sin(.9*k);
    p->Pos[1]=.13+1.10*((k/8)%8)+.025*cos(.7*k);
    p->Pos[2]=.14+1.10*(k/64)+.025*sin(.3+.4*k);
    p->Vel[0]=sin(.1+.73*id);
    p->Vel[1]=cos(.4+.51*id);
    p->Vel[2]=.6*sin(.2+.37*id);
    p->Hsml=id%4==0 ? 0 : id%4==1 ? .125 : id%4==2 ? 8 : 2.5;
}

static DomainDecomp octants(int ntask)
{
    const int empty=getenv("SIDM_TEST_EMPTY_OWNER")!=NULL;
    DomainDecomp d={0};
    d.domain_allocated_flag=1; d.DomainComm=MPI_COMM_WORLD;
    d.NTopNodes=9; d.NTopLeaves=8;
    d.TopNodes=mymalloc("distributed_test_nodes",9*sizeof(*d.TopNodes));
    d.TopLeaves=mymalloc("distributed_test_leaves",8*sizeof(*d.TopLeaves));
    d.Tasks=mymalloc("distributed_test_tasks",ntask*sizeof(*d.Tasks));
    memset(d.TopNodes,0,9*sizeof(*d.TopNodes));
    memset(d.TopLeaves,0,8*sizeof(*d.TopLeaves));
    d.TopNodes[0].Daughter=1; d.TopNodes[0].Shift=3*BITS_PER_DIMENSION;
    for(int leaf=0;leaf<8;leaf++) {
        d.TopNodes[leaf+1].StartKey=(peano_t)leaf << (3*(BITS_PER_DIMENSION-1));
        d.TopNodes[leaf+1].Daughter=-1;
        d.TopNodes[leaf+1].Shift=3*(BITS_PER_DIMENSION-1);
        d.TopNodes[leaf+1].Leaf=leaf;
        d.TopLeaves[leaf].Task=empty?0:leaf/(8/ntask); d.TopLeaves[leaf].topnode=leaf+1;
    }
    for(int task=0;task<ntask;task++) {
        d.Tasks[task].StartLeaf=empty?(task?8:0):task*(8/ntask);
        d.Tasks[task].EndLeaf=empty?8:(task+1)*(8/ntask);
    }
    return d;
}

static double distance(int i,int j,int periodic)
{
    struct particle_data a,b;fixture(&a,i);fixture(&b,j);
    double r2=0;
    for(int dim=0;dim<3;dim++) {
        double dx=a.Pos[dim]-b.Pos[dim];
        if(periodic) dx-=8*nearbyint(dx/8);
        r2+=dx*dx;
    }
    return sqrt(r2);
}
static int compare_double(const void *a,const void *b)
{
    const double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
static double support(int i,int periodic,double fixed)
{
    if(fixed>0) return fixed;
    double r[TEST_N-1];int n=0;
    for(int j=0;j<TEST_N;j++) if(i!=j) r[n++]=distance(i,j,periodic);
    qsort(r,n,sizeof(*r),compare_double);
    return nextafter(r[DistParams.neighbors-1],INFINITY);
}

static void moments(long double result[4])
{
    long double local[4]={0};
    for(int i=0;i<PartManager->NumPart;i++) if(P[i].Type==1) for(int d=0;d<3;d++) {
        const long double v=P[i].Vel[d];local[d]+=P[i].Mass*v;
        local[3]+=.5L*P[i].Mass*v*v;
    }
    MPI_Allreduce(local,result,4,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
static void reset_state(void)
{
    for(int i=0;i<PartManager->NumPart;i++) {
        if(P[i].Type!=1) continue;
        const int leaf=P[i].TopLeaf, id=P[i].ID-1;
        fixture(&P[i],id);P[i].TopLeaf=leaf;
    }
}
static void collect(double state[TEST_N][3])
{
    double local[TEST_N][3]={{0}};
    for(int i=0;i<PartManager->NumPart;i++) if(P[i].Type==1)
        for(int d=0;d<3;d++) local[P[i].ID-1][d]=P[i].Vel[d];
    MPI_Allreduce(local,state,TEST_N*3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

static int setup(void **s) { (void)s;walltime_init(&clocks);return 0; }
static void test_direct_hierarchy(void **state)
{
    (void)state;
    init_forcetree_params(2);
    ParameterSet *ps=parameter_set_new();
    param_declare_double(ps,"ImportBufferBoost",OPTIONAL,2,"test");
    assert_int_equal(param_parse(ps,""),0);set_treewalk_params(ps);parameter_set_free(ps);
    int rank,ntask;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ntask);
    sidm_local_configure(1,.2,.25,16);
    Cosmology cp={0};cp.HubbleParam=.5;cp.Hubble=1;cp.Omega0=cp.OmegaCDM=1;cp.ONu.rhocrit=1;
    double outputs[]={.1,.2};set_sync_params_test(2,outputs);
    setup_sync_points(&cp,.1,.2,0,0);
    const struct UnitSystem units=get_unitsystem(2,8,1e5);
    particle_alloc_memory(PartManager,8,2*TEST_N);
    DomainDecomp domain=octants(ntask);
    for(int id=0;id<TEST_N;id++) {
        struct particle_data p;fixture(&p,id);
        int leaf=peano_hilbert_key(p.Pos[0]>=4,p.Pos[1]>=4,p.Pos[2]>=4,1);
        if(domain.TopLeaves[leaf].Task!=rank) continue;
        p.TopLeaf=leaf;P[PartManager->NumPart++]=p;
        /* Coincident massive gas must not enter the DM neighbor/rate solve. */
        p.Type=0;p.ID+=TEST_N;p.Mass=100;p.Vel[0]=1000;
        P[PartManager->NumPart++]=p;
    }
    ForceTree tree={0};force_tree_rebuild_mask(&tree,&domain,DMMASK,NULL);
    DriftKickTimes times={0};times.Ti_Current=2;
    DistParams.neighbors=38;
    DistParams.probability=.5;DistParams.velocity_dependent=velocity_dependent;DistParams.sigma0=20;DistParams.w=20;
    int indices[2*TEST_N],ns=0;
    /* More than 100 exports per thread on two ranks, even with balanced
     * chunks. Non-source ranks still exercise inactive-only receivers. */
    for(int i=0;i<PartManager->NumPart;i++) if(P[i].Type==1&&rank==0&&((ntask>1&&!getenv("SIDM_TEST_EMPTY_OWNER"))||P[i].ID%3)) indices[ns++]=i;
    ActiveParticles sources={.Particles=P,.ActiveParticle=indices,.NumActiveParticle=ns};
    double reference[TEST_N][3];uint64_t large_batches=0;
    for(int tiny=0;tiny<2;tiny++) {
        reset_state();exported_batches=0;
        for(int i=0;i<PartManager->NumPart;i++) P[i].Ti_drift=times.Ti_Current;
        /* Four ranks need room for a complete native 32-source chunk. */
        const size_t slots=ntask==4?256:100;
        treewalk_set_max_export_buffer(tiny?slots*sizeof(struct dist_query)*omp_get_max_threads():4*1024*1024);
        double intervals[2*TEST_N]={0},limits[2*TEST_N];
        for(int i=0;i<PartManager->NumPart;i++) limits[i]=NAN;
        sidm_distributed_force_intervals(&sources,&cp,units,&times,&tree,intervals,limits,123);
        /* Zero exposure still measures the exact initial neighbor rate; it
         * must neither collide nor reset inactive receivers' support. */
        for(int a=0;a<ns;a++) {
            int i=indices[a],id=P[i].ID-1;double h=support(id,1,0),max_sv=0;
            struct particle_data src;fixture(&src,id);
            assert_true(fabs(P[i].Hsml/h-1)<1e-13);
            for(int j=0;j<TEST_N;j++) if(j!=id&&distance(id,j,1)<=h) {
                struct particle_data target;fixture(&target,j);
                double u=hypot(hypot(src.Vel[0]-target.Vel[0],src.Vel[1]-target.Vel[1]),src.Vel[2]-target.Vel[2]);
                const double w=20*get_atime(times.Ti_Current);
                const double shape=velocity_dependent?sidm_rate_viscosity_shape(u*u/(w*w)):1;
                /* h=.5, mass unit=8, length unit=2: sigma_code=20. */
                max_sv=fmax(max_sv,20*u*shape);
            }
            assert_true(fabs(limits[i]/(.5/(.5*sidm_rate_cubic_kernel(0,h)*max_sv))-1)<1e-12);
            intervals[i]=.4;
        }
        for(int i=0;i<PartManager->NumPart;i++) {
            if(P[i].Type!=1) {
                struct particle_data initial;fixture(&initial,P[i].ID-TEST_N-1);
                initial.Vel[0]=1000;
                assert_memory_equal(P[i].Vel,initial.Vel,sizeof(P[i].Vel));
                assert_true(P[i].Hsml==initial.Hsml);continue;
            }
            struct particle_data initial;fixture(&initial,P[i].ID-1);
            assert_memory_equal(P[i].Vel,initial.Vel,sizeof(P[i].Vel));
        }
        long double before[4],after[4];moments(before);
        sidm_distributed_force_intervals(&sources,&cp,units,&times,&tree,intervals,limits,123);
        moments(after);
        for(int d=0;d<4;d++) assert_true(fabsl(after[d]-before[d])<1e-11*(1+fabsl(before[d])));
        double current[TEST_N][3];collect(current);
        int receivers=0;
        for(int i=0;i<PartManager->NumPart;i++) {
            if(P[i].Type!=1) {
                struct particle_data initial;fixture(&initial,P[i].ID-TEST_N-1);
                initial.Vel[0]=1000;
                assert_memory_equal(P[i].Vel,initial.Vel,sizeof(P[i].Vel));
                assert_true(P[i].Hsml==initial.Hsml);continue;
            }
            struct particle_data initial;fixture(&initial,P[i].ID-1);
            receivers+=(memcmp(P[i].Vel,initial.Vel,sizeof(P[i].Vel))!=0)&&(rank!=0||((ntask==1||getenv("SIDM_TEST_EMPTY_OWNER"))&&P[i].ID%3==0));
        }
        MPI_Allreduce(MPI_IN_PLACE,&receivers,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        assert_true(receivers>0);
        uint64_t batches;MPI_Allreduce(&exported_batches,&batches,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
        if(!tiny) { memcpy(reference,current,sizeof(reference));large_batches=batches; }
        else {
            assert_memory_equal(reference,current,sizeof(reference));
            if(ntask>1 && !getenv("SIDM_TEST_EMPTY_OWNER")) assert_true(batches>large_batches);
        }
    }
    treewalk_set_max_export_buffer(3584*1024*1024L);
    sidm_local_configure(0,.2,.25,16);
    force_tree_free(&tree);myfree(domain.Tasks);myfree(domain.TopLeaves);myfree(domain.TopNodes);
    myfree(P);PartManager->Base=NULL;PartManager->NumPart=PartManager->MaxPart=0;
}
static void test_velocity_dependent_units(void **state)
{ velocity_dependent=1;test_direct_hierarchy(state);velocity_dependent=0; }
int main(void)
{
    int requested=1,observed_threads=0;
    const char *value=getenv("SIDM_TEST_THREADS");
    if(value) {
        char *end=NULL;errno=0;
        const long count=strtol(value,&end,10);
        if(errno || end==value || *end || count<1 || count>64) {
            fprintf(stderr,"SIDM_TEST_THREADS must be an integer from 1 to 64.\n");
            return EXIT_FAILURE;
        }
        requested=(int)count;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(requested);
#pragma omp parallel
    {
#pragma omp single
        observed_threads=omp_get_num_threads();
    }
    printf("SIDM distributed test threads: requested=%d observed=%d\n",requested,observed_threads);
    if(observed_threads!=requested) return EXIT_FAILURE;
    const struct CMUnitTest tests[]={cmocka_unit_test(test_direct_hierarchy),
        cmocka_unit_test(test_velocity_dependent_units)};
    return cmocka_run_group_tests_mpi(tests,setup,NULL);
}
#else
int main(void) { puts("SKIP test_sidm_distributed: SIDM disabled.");return 0; }
#endif
