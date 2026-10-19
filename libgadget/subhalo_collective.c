/* Oversized candidates unbound in place with the global tree (design 7.8); the outline and the
 * walk contract are in subhalo_collective.h.
 *
 * Memory per rank (admitted by the plan through subhalo_coll_estimate): for the whole call 25 bytes
 * of scratch per local particle (source candidate, claim, flags, two energies) and the per-candidate
 * vectors; then the stages, each freed before the next: tree and walk, cores, radial set,
 * properties, relaxed sources, final tags, serial repair.
 *
 * Edge cases: a claim of a candidate or serial member that sinks later is void; an in-place survivor
 * that sinks in a later round keeps its final-pass most-bound tracer, a serial one that of its
 * record; a serial tail whose claimant sank while its serial binder survives goes to the binder, in
 * the binder's final-pass energy order; the final-tree check of a claim (J2) applies to in-place
 * binders only (serial ones were checked in their key). */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <gsl/gsl_eigen.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include "utils/mpsort.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "forcetree.h"
#include "treewalk.h"
/* gravshort.h defines static treewalk callbacks this file does not use */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "gravshort.h"
#pragma GCC diagnostic pop
#include "gravity.h"
#include "walltime.h"
#include "subhalo_unbind.h"
#include "subhalo_collective.h"

/* per-particle flags */
#define CF_BOUND   1     /* bound by its source candidate (Cand) */
#define CF_RES     2     /* one of the reserved most-bound tracers of Cand */
#define CF_SCLAIM  4     /* a serial claim: the tag of a serial sub-tree member (a tail, or a root's
                          * remainder inside its relaxed source), valid while that member is alive */
#define CF_SBOUND  8     /* bound by a serial sub-tree member (BoundTrackId), excluded from the sources */
#define CF_UNIV    16    /* a particle of an oversized key: the final tag pass writes it */
#define CF_RELEASE 32    /* a field particle tagged by an orphan or the central: untagged by the final pass */

/* Shape fit as subhalo_props.c */
#define SHAPE_MINPART 20
#define SHAPE_ITERATIONS 10
#define KLYPIN_RMAX_TO_RS 2.1626
#define KLYPIN_RS_CONSTANT 0.216216

static struct SubhaloCollectiveStats CollStats;
static subhalo_collective_hook PretagHook = NULL;

void
subhalo_collective_set_pretag_hook(subhalo_collective_hook fn)
{
    PretagHook = fn;
}

const struct SubhaloCollectiveStats *
subhalo_collective_last_stats(void)
{
    return &CollStats;
}

/* The state of a candidate during the call, identical on every rank. */
struct CandState {
    int state;               /* SUBHALO_ST_* */
    int iterating;           /* in the current unbinding loop */
    int ran;                 /* unbound by the current call of unbind_loop */
    int capped;              /* that unbinding stopped at the iteration cap */
    int flags;               /* SUBHALO_FLAG_ITERCAP */
    int iterations;          /* of the last unbinding */
    int absorbed;            /* a survivor sank into it this round */
    int has_core;
    int has_mb;
    int pad;
    int64_t N, Nbound, Nlast, Ntracer;
    double Mbound, ConvRatio;
    double Ref0[3];          /* the reference of the nearest-image sums */
    double RefPos[3], RefVel[3];
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;   /* the sinking core (loop order) */
    MyIDType mbID;           /* most-bound tracer at the sinking (alive) or the previous one (died) */
    double mbPos[3], mbVel[3];
    int64_t keep;            /* relaxed source: max(Nbound, min(N, R Nbound)) */
    int64_t SinkTrackId;
    /* the current record quantities the hierarchy needs */
    float Mbound_rec, Vmax_rec;
    MyIDType MostBoundID_rec;
};

struct Coll {
    struct SubhaloPlan * plan;
    struct SubhaloCallCtx * ctx;
    DomainDecomp * ddecomp;
    MPI_Comm Comm;
    int NTask, ThisTask;
    const struct SubhaloUnbindParams * up;
    const struct SubhaloParams * sp;
    int64_t NumPart;
    int32_t * Cand;
    int32_t * Claim;
    uint8_t * Flag;
    double * En;
    double * ClaimE;
    struct CandState * cs;
    int64_t c0, c1;          /* the current round */
    int nwalks;
    struct SubhaloCallStats st;
};

static inline int
part_live(const struct particle_data * p)
{
    return !p->IsGarbage && !p->Swallowed;
}

static inline int
is_tracer(int64_t i, int TracerTypes)
{
    return (TracerTypes >> P[i].Type) & 1;
}

static inline uint64_t
double_key(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
}

static const struct SubhaloMember *
member_of(const struct Coll * C, int64_t TrackId)
{
    if(TrackId == SUBHALO_NOTRACK)
        return NULL;
    return subhalo_plan_member(C->plan, TrackId);
}

/* HBT CompareMass (subhalo_tracking.cpp:112-139), as mass_before in subhalo_track.c. */
static int
mass_before_v(float Ma, float Va, MyIDType Ia, int64_t Ta, float Mb, float Vb, MyIDType Ib, int64_t Tb)
{
    if(Ma != Mb) return Ma > Mb;
    if(Va != Vb) return Va > Vb;
    if(Ia != Ib) return Ia > Ib;
    return Ta > Tb;
}

static double CollImportBufferBoost = 2;

void
subhalo_collective_set_import_buffer_boost(double boost)
{
    CollImportBufferBoost = boost;
}

size_t
subhalo_collective_walk_bytes(void)
{
    /* query: TreeWalkQueryBase + label + index; result: TreeWalkResultBase + one double */
    const size_t q = sizeof(TreeWalkQueryBase) + 8, r = sizeof(TreeWalkResultBase) + 8;
    const size_t bpb = sizeof(struct data_index) + q + r + (size_t) ceil(CollImportBufferBoost * (q + r));
    const int64_t nthr = omp_get_max_threads();
    const int64_t nbuf = 4096 * 10 + (4096 + 1 > 100 * nthr ? 4096 + 1 : 100 * nthr);
    return (size_t) nbuf * bpb;
}

/* ------------------------------------------------------------------ the potential-only walk */

typedef struct {
    TreeWalkQueryBase base;
    int32_t Label;           /* dense round label of the query's candidate */
    int32_t Index;           /* the query particle on its home rank */
} TreeWalkQueryColl;

typedef struct {
    TreeWalkResultBase base;
    double Phi;              /* sum_j m_j W(r_ij, h) */
} TreeWalkResultColl;

struct CollWalkPriv {
    const int32_t * Cand;
    double * Sum;
    int64_t c0;
    double theta2, h;
};
#define COLL_PRIV(tw) ((struct CollWalkPriv *) ((tw)->priv))

static inline int
node_label(const struct NODE * n)
{
    return (int) n->mom.hmax;
}

static inline int
label_merge(int a, int b)
{
    if(a == -2) return b;
    if(b == -2) return a;
    return a == b ? a : -1;
}

/* Post-order labels of the local nodes below no (particle and node children; a pseudo node keeps
 * what the top-leaf exchange gave it). */
static int
label_local(const ForceTree * tree, int no, const int32_t * Cand, int64_t c0)
{
    struct NODE * n = &tree->Nodes[no];
    int lab = -2, j;
    if(n->f.ChildType == PARTICLE_NODE_TYPE) {
        for(j = 0; j < n->s.noccupied && j < NMAXCHILD; j++)
            lab = label_merge(lab, (int) (Cand[n->s.suns[j]] - c0));
    }
    else if(n->f.ChildType == NODE_NODE_TYPE) {
        for(j = 0; j < NMAXCHILD; j++) {
            const int c = n->s.suns[j];
            if(c < 0)
                continue;
            if(!node_is_node(c, tree))
                endrun(5, "Subhalo in-place walk: node %d has a child %d that is not a node.\n", no, c);
            lab = label_merge(lab, label_local(tree, c, Cand, c0));
        }
    }
    else
        return node_label(n);
    n->mom.hmax = lab;
    return lab;
}

/* The internal top-level nodes from their eight children (top leaves already labelled). */
static int
label_top(const ForceTree * tree, int no)
{
    struct NODE * n = &tree->Nodes[no];
    if(!n->f.InternalTopLevel)
        return node_label(n);
    int lab = -2, j;
    for(j = 0; j < 8; j++)
        lab = label_merge(lab, label_top(tree, n->s.suns[j]));
    n->mom.hmax = lab;
    return lab;
}

/* Node labels of design 7.8: local post-order pass from every local top leaf, the top-leaf labels
 * exchanged with one MPI_Allgatherv, then propagated up the top-level tree (pseudo nodes
 * included). MyFloat holds the dense round labels exactly (< 2^24 candidates per round). */
static void
label_tree(const ForceTree * tree, const DomainDecomp * ddecomp, const int32_t * Cand, int64_t c0)
{
    int ThisTask, NTask, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int * lab = (int *) mymalloc("SubCollTopLab", sizeof(int) * (ddecomp->NTopLeaves + 1));
    for(i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i++)
        lab[i] = label_local(tree, ddecomp->TopLeaves[i].treenode, Cand, c0);
    int * counts = ta_malloc("SubCollLabCnt", int, NTask);
    int * displ = ta_malloc("SubCollLabDispl", int, NTask);
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = ddecomp->Tasks[r].StartLeaf;
        counts[r] = ddecomp->Tasks[r].EndLeaf - ddecomp->Tasks[r].StartLeaf;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, lab, counts, displ, MPI_INT, MPI_COMM_WORLD);
    ta_free(displ);
    ta_free(counts);
    for(i = 0; i < ddecomp->NTopLeaves; i++)
        tree->Nodes[ddecomp->TopLeaves[i].treenode].mom.hmax = lab[i];
    label_top(tree, tree->firstnode);
    myfree(lab);
}

static void
coll_fill(int i, TreeWalkQueryColl * I, TreeWalk * tw)
{
    const struct CollWalkPriv * pv = COLL_PRIV(tw);
    I->Label = (int32_t) (pv->Cand[i] - pv->c0);
    I->Index = i;
}

static void
coll_reduce(int i, TreeWalkResultColl * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    struct CollWalkPriv * pv = COLL_PRIV(tw);
    TREEWALK_REDUCE(pv->Sum[i], O->Phi);
}

/* The potential-only visit (the structure of force_treeev_shortrange, gravshort-tree.c). */
static int
coll_visit(TreeWalkQueryColl * I, TreeWalkResultColl * O, LocalTreeWalk * lv)
{
    const ForceTree * tree = lv->tw->tree;
    const struct CollWalkPriv * pv = COLL_PRIV(lv->tw);
    const double BoxSize = tree->BoxSize;
    const double theta2 = pv->theta2, h = pv->h;
    const int label = I->Label;
    /* the query meets itself only on its home rank, in the primary walk */
    const int self = lv->mode == TREEWALK_PRIMARY ? I->Index : -1;
    const double * inpos = I->base.Pos;
    double sum = 0;
    int64_t ninter = 0;
    int listindex;
    for(listindex = 0; listindex < NODELISTLENGTH; listindex++) {
        int no = I->base.NodeList[listindex];
        const int startno = no;
        if(no < 0)
            break;
        while(no >= 0) {
            const struct NODE * nop = &tree->Nodes[no];
            if(lv->mode == TREEWALK_GHOSTS && nop->f.TopLevel && no != startno)
                break;
            const int nl = node_label(nop);
            /* empty, or every particle below belongs to another candidate */
            if(nl == -2 || (nl >= 0 && nl != label)) {
                no = nop->sibling;
                continue;
            }
            double dx[3];
            int d;
            for(d = 0; d < 3; d++)
                dx[d] = NEAREST(nop->mom.cofm[d] - inpos[d], BoxSize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            /* geometric opening; a mixed node is always opened */
            const int open = nl == -1 || (double) nop->len * (double) nop->len > r2 * theta2;
            if(!open) {
                no = nop->sibling;
                if(lv->mode != TREEWALK_TOPTREE) {
                    sum += nop->mom.mass * grav_spline_potential(sqrt(r2), h);
                    ninter++;
                }
                continue;
            }
            if(lv->mode == TREEWALK_TOPTREE) {
                if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                    if(-1 == treewalk_export_particle(lv, nop->s.suns[0]))
                        return -1;
                    no = nop->sibling;
                    continue;
                }
                /* a local top leaf is walked by the primary walk */
                if(nop->f.TopLevel && !nop->f.InternalTopLevel) {
                    no = nop->sibling;
                    continue;
                }
                no = nop->s.suns[0];
            }
            else {
                if(nop->f.ChildType == PARTICLE_NODE_TYPE) {
                    int i;
                    for(i = 0; i < nop->s.noccupied; i++) {
                        const int pp = nop->s.suns[i];
                        if(pp == self || pv->Cand[pp] - pv->c0 != label)
                            continue;
                        double r2p = 0;
                        for(d = 0; d < 3; d++) {
                            const double x = NEAREST(P[pp].Pos[d] - inpos[d], BoxSize);
                            r2p += x * x;
                        }
                        sum += P[pp].Mass * grav_spline_potential(sqrt(r2p), h);
                        ninter++;
                    }
                    no = nop->sibling;
                }
                else if(nop->f.ChildType == PSEUDO_NODE_TYPE)
                    no = nop->sibling;
                else
                    no = nop->s.suns[0];
            }
        }
    }
    O->Phi = sum;
    treewalk_add_counters(lv, ninter);
    return 1;
}

/* One global tree over the bound potential-type particles of the iterating candidates of the
 * round and one walk for all their bound particles: Sum[i] = sum_j m_j W. Collective. */
static void
coll_walk(struct Coll * C, double * Sum)
{
    const int64_t NumPart = PartManager->NumPart;
    int64_t i, nq = 0, nt = 0;
    const int PotentialTypes = C->up->PotentialTypes;
    for(i = 0; i < NumPart; i++) {
        const int32_t c = C->Cand[i];
        if(c < C->c0 || c >= C->c1 || !(C->Flag[i] & CF_BOUND) || !C->cs[c].iterating || !part_live(&P[i]))
            continue;
        nq++;
        if((PotentialTypes >> P[i].Type) & 1)
            nt++;
    }
    int * ql = (int *) mymalloc("SubCollQueries", sizeof(int) * (nq + 1));
    int * tl = (int *) mymalloc("SubCollTreeList", sizeof(int) * (nt + 1));
    nq = nt = 0;
    for(i = 0; i < NumPart; i++) {
        const int32_t c = C->Cand[i];
        if(c < C->c0 || c >= C->c1 || !(C->Flag[i] & CF_BOUND) || !C->cs[c].iterating || !part_live(&P[i]))
            continue;
        ql[nq++] = (int) i;
        if((PotentialTypes >> P[i].Type) & 1)
            tl[nt++] = (int) i;
        Sum[i] = 0;
    }
    ActiveParticles act;
    memset(&act, 0, sizeof(act));
    act.ActiveParticle = tl;
    act.NumActiveParticle = nt;
    act.MaxActiveParticle = nt;
    act.Particles = PartManager->Base;
    ForceTree tree;
    memset(&tree, 0, sizeof(tree));
    /* force_tree_build sizes its node block by NumPart: raised for the build only on a nearly empty
     * rank, to hold the top-level grid and the per-thread node caches (only listed particles are inserted) */
    const int64_t NumPartSave = PartManager->NumPart;
    {
        const int nthr = omp_get_max_threads();
        const int64_t nleaf = C->ddecomp->Tasks[C->ThisTask].EndLeaf - C->ddecomp->Tasks[C->ThisTask].StartLeaf;
        int64_t need = 2048 + 512 * (int64_t) nthr + 4 * nleaf * nthr;
        if(need > PartManager->MaxPart)
            need = PartManager->MaxPart;
        if(PartManager->NumPart < need)
            PartManager->NumPart = need;
    }
    /* the whole build booked to /Subhalo/Rounds/Tree, not forcetree.c's own clocks, so that the
     * /Subhalo/Rounds children cover the stage (decision 47.5) */
    walltime_measure("/Subhalo/Rounds/Rounds");
    walltime_redirect_begin("/Subhalo/Rounds/Tree");
    force_tree_active_moments(&tree, C->ddecomp, &act, 0, 0, NULL);
    walltime_redirect_end();
    PartManager->NumPart = NumPartSave;
    label_tree(&tree, C->ddecomp, C->Cand, C->c0);

    struct CollWalkPriv pv;
    pv.Cand = C->Cand;
    pv.Sum = Sum;
    pv.c0 = C->c0;
    pv.theta2 = C->up->OpenAngle * C->up->OpenAngle;
    if(pv.theta2 > 0.33)
        pv.theta2 = 0.33;
    pv.h = C->up->h;
    TreeWalk tw[1];
    memset(tw, 0, sizeof(tw));
    tw->ev_label = "SUBHALO_INPLACE";
    tw->type = TREEWALK_ACTIVE;
    tw->visit = (TreeWalkVisitFunction) coll_visit;
    tw->NoNgblist = 1;
    tw->haswork = NULL;
    tw->fill = (TreeWalkFillQueryFunction) coll_fill;
    tw->reduce = (TreeWalkReduceResultFunction) coll_reduce;
    tw->query_type_elsize = sizeof(TreeWalkQueryColl);
    tw->result_type_elsize = sizeof(TreeWalkResultColl);
    tw->priv = &pv;
    tw->tree = &tree;
    /* the treewalk sizes its export tables from the free memory: that elastic part is measured
     * apart, and the high water charges the walk with the reserve the admission guarantees */
    const size_t lowbefore = allocator_get_min_free(A_MAIN), freebefore = mymalloc_freebytes();
    treewalk_run(tw, ql, nq);
    walltime_measure("/Subhalo/Rounds/Walk");
    {
        const size_t lowwalk = allocator_get_min_free(A_MAIN);
        const double took = (double) freebefore - (double) lowwalk;
        if(took > CollStats.WalkElasticBytes)
            CollStats.WalkElasticBytes = took;
        const size_t reserve = subhalo_collective_walk_bytes();
        size_t low = freebefore > reserve ? freebefore - reserve : 0;
        if(lowwalk > low)
            low = lowwalk;
        A_MAIN->minfree = lowbefore < low ? lowbefore : low;
    }
    force_tree_free(&tree);
    myfree(tl);
    myfree(ql);
    C->nwalks++;
}

void
subhalo_collective_test_walk(DomainDecomp * ddecomp, const int32_t * label, int64_t nlabels, double theta, double h,
                             int PotentialTypes, double * out)
{
    struct Coll C;
    struct SubhaloUnbindParams up;
    memset(&C, 0, sizeof(C));
    memset(&up, 0, sizeof(up));
    up.OpenAngle = theta;
    up.h = h;
    up.PotentialTypes = PotentialTypes;
    C.up = &up;
    C.ddecomp = ddecomp;
    C.Comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &C.NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &C.ThisTask);
    C.NumPart = PartManager->NumPart;
    C.c0 = 0;
    C.c1 = nlabels;
    C.cs = (struct CandState *) mymalloc("SubTestWalkCs", sizeof(struct CandState) * (nlabels + 1));
    memset(C.cs, 0, sizeof(struct CandState) * (nlabels + 1));
    int64_t i;
    for(i = 0; i < nlabels; i++)
        C.cs[i].iterating = 1;
    C.Cand = (int32_t *) mymalloc("SubTestWalkCand", sizeof(int32_t) * (C.NumPart + 1));
    C.Flag = (uint8_t *) mymalloc("SubTestWalkFlag", C.NumPart + 1);
    for(i = 0; i < C.NumPart; i++) {
        C.Cand[i] = label[i] >= 0 && label[i] < nlabels ? label[i] : -1;
        C.Flag[i] = C.Cand[i] >= 0 ? CF_BOUND : 0;
    }
    coll_walk(&C, out);
    myfree(C.Flag);
    myfree(C.Cand);
    myfree(C.cs);
}

/* ------------------------------------------------------------------ the unbinding loop */

static inline int
in_round(const struct Coll * C, int32_t c)
{
    return c >= C->c0 && c < C->c1;
}

/* E_i in the candidate's current frame from the walk sum (design 7.10, subhalo_unbind.h). */
static inline double
coll_energy(const struct Coll * C, int64_t i, double sum, const struct CandState * s)
{
    const struct SubhaloUnbindParams * up = C->up;
    const double a = up->atime, Ha = up->Hubble * a;
    double K = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double dx = NEAREST(P[i].Pos[d] - s->RefPos[d], up->BoxSize);
        const double dv = (P[i].Vel[d] - s->RefVel[d]) / a + Ha * dx;
        K += dv * dv;
    }
    double e = 0.5 * K - up->G / a * sum;
    if(up->WithThermal)
        e += subhalo_thermal_energy(i, a);
    return e;
}

#define NSUM 9
/* Per candidate of the round (bound particles, or the whole source with all = 1): count, mass,
 * sum m dx about Ref0, sum m Vel, tracers. Collective. */
static void
round_sums(const struct Coll * C, int all, double * out)
{
    const int64_t nc = C->c1 - C->c0;
    double * loc = (double *) mymalloc("SubCollSums", sizeof(double) * NSUM * (nc + 1));
    memset(loc, 0, sizeof(double) * NSUM * (nc + 1));
    int64_t i;
    int d;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t c = C->Cand[i];
        if(!in_round(C, c) || !part_live(&P[i]) || (!all && !(C->Flag[i] & CF_BOUND)))
            continue;
        double * s = loc + NSUM * (c - C->c0);
        const double m = P[i].Mass;
        s[0] += 1;
        s[1] += m;
        for(d = 0; d < 3; d++) {
            s[2 + d] += m * NEAREST(P[i].Pos[d] - C->cs[c].Ref0[d], C->up->BoxSize);
            s[5 + d] += m * P[i].Vel[d];
        }
        if(is_tracer(i, C->up->TracerTypes))
            s[8] += 1;
    }
    MPI_Allreduce(loc, out, NSUM * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    myfree(loc);
}

/* The HBT loop (subhalo_unbind without subsampling) for the iterating candidates of the round,
 * every source particle of an iterating candidate starting bound. A dead candidate's particles
 * are left unbound. Collective. */
static void
unbind_loop(struct Coll * C)
{
    const struct SubhaloUnbindParams * up = C->up;
    const int64_t nc = C->c1 - C->c0;
    int64_t i, c;
    int d;
    double * S = (double *) mymalloc("SubCollLoopSums", sizeof(double) * NSUM * (nc + 1));
    /* every source particle bound; the source counts (subhalo_unbind's early return) */
    for(i = 0; i < C->NumPart; i++)
        if(in_round(C, C->Cand[i]) && C->cs[C->Cand[i]].iterating && part_live(&P[i]))
            C->Flag[i] |= CF_BOUND;
    round_sums(C, 0, S);
    int any = 0;
    for(c = C->c0; c < C->c1; c++)
        C->cs[c].ran = 0;
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        if(!s->iterating)
            continue;
        const double * v = S + NSUM * (c - C->c0);
        s->N = s->Nbound = (int64_t) v[0];
        s->iterations = 0;
        s->ConvRatio = 0;
        s->ran = 1;
        s->capped = 0;
        C->st.NCandidates++;
        if(s->N < up->MinNumPart || s->N <= 0 || v[8] < up->MinNumTracerPart) {
            s->iterating = 0;
            s->Nbound = 0;
            s->state = -1;           /* dead, resolved below */
            continue;
        }
        any = 1;
    }
    while(MPIU_Any(any, C->Comm)) {
        for(c = C->c0; c < C->c1; c++)
            if(C->cs[c].iterating) {
                C->cs[c].Nlast = C->cs[c].Nbound;
                C->cs[c].iterations++;
            }
        coll_walk(C, C->En);
        /* energies of the queried particles; the bound ones stay bound */
        for(i = 0; i < C->NumPart; i++) {
            const int32_t cc = C->Cand[i];
            if(!in_round(C, cc) || !C->cs[cc].iterating || !(C->Flag[i] & CF_BOUND) || !part_live(&P[i]))
                continue;
            C->En[i] = coll_energy(C, i, C->En[i], &C->cs[cc]);
            if(!(C->En[i] < 0))
                C->Flag[i] &= ~CF_BOUND;
        }
        round_sums(C, 0, S);
        any = 0;
        for(c = C->c0; c < C->c1; c++) {
            struct CandState * s = &C->cs[c];
            if(!s->iterating)
                continue;
            const double * v = S + NSUM * (c - C->c0);
            s->Nbound = (int64_t) v[0];
            s->ConvRatio = s->Nlast > 0 ? (double) s->Nbound / (double) s->Nlast : 0;
            if(s->Nbound < up->MinNumPart || v[8] < up->MinNumTracerPart) {
                s->iterating = 0;
                s->state = -1;
                s->Nbound = 0;
                continue;
            }
            /* frame of the bound set */
            s->Mbound = v[1];
            for(d = 0; d < 3; d++) {
                s->RefPos[d] = subhalo_wrap_box(s->Ref0[d] + v[2 + d] / v[1], up->BoxSize);
                s->RefVel[d] = v[5 + d] / v[1];
            }
            s->Ntracer = (int64_t) v[8];
            if((double) s->Nbound >= (double) s->Nlast * up->BoundMassPrecision) {
                s->iterating = 0;
                continue;
            }
            if(up->MaxIter > 0 && s->iterations >= up->MaxIter) {
                /* the flag of an earlier unbinding of the round is kept (ORed, as unbind_counted) */
                s->flags |= SUBHALO_FLAG_ITERCAP;
                s->capped = 1;
                s->iterating = 0;
                continue;
            }
            any = 1;
        }
    }
    /* the deaths: the particles are all unbound */
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(in_round(C, cc) && C->cs[cc].state == -1)
            C->Flag[i] &= ~(CF_BOUND | CF_RES);
    }
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        if(s->state == -1) {
            const struct SubhaloMember * m = &C->plan->Members[C->plan->Cands[c].member];
            s->state = m->kind == SUBHALO_MEM_RESOLVED ? SUBHALO_ST_DIED : SUBHALO_ST_FAILED;
            s->Nbound = 0;
            s->Mbound = 0;
        }
        if(s->ran && s->capped) {
            C->st.NItercap++;
            if(C->st.WorstTrackId == -1 || s->ConvRatio < C->st.WorstConvRatio) {
                C->st.WorstConvRatio = s->ConvRatio;
                C->st.WorstTrackId = C->plan->Cands[c].TrackId;
            }
        }
    }
    myfree(S);
}

/* ------------------------------------------------------------------ cores (per-owner selection) */

struct CoreRec {
    int64_t cand;
    double E;
    MyIDType ID;
    double Pos[3], Vel[3], Mass;
    int32_t tracer, rank;
    int64_t index;
    int32_t owner, pad;
};

static int
cmp_corerec_owner(const void * a, const void * b)
{
    const struct CoreRec * x = (const struct CoreRec *) a, * y = (const struct CoreRec *) b;
    if(x->owner != y->owner) return (x->owner > y->owner) - (x->owner < y->owner);
    if(x->cand != y->cand) return (x->cand > y->cand) - (x->cand < y->cand);
    if(x->E != y->E) return (x->E > y->E) - (x->E < y->E);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* (E, ID) order of the core selection */
static inline int
corerec_before(const struct CoreRec * x, const struct CoreRec * y)
{
    if(x->E != y->E)
        return x->E < y->E;
    return x->ID < y->ID;
}

/* A bounded max-heap of the K first records in (E, ID) order: a rank keeps at most K tracers and K
 * non-tracers per candidate, not a record per bound particle. */
static void
core_heap_push(struct CoreRec * h, int64_t * n, int64_t K, const struct CoreRec * e)
{
    int64_t j;
    if(*n < K) {
        j = (*n)++;
        while(j > 0) {
            const int64_t p = (j - 1) / 2;
            if(!corerec_before(&h[p], e))
                break;
            h[j] = h[p];
            j = p;
        }
        h[j] = *e;
        return;
    }
    if(K <= 0 || !corerec_before(e, &h[0]))
        return;
    /* replace the largest and sift down */
    j = 0;
    for(;;) {
        int64_t c = 2 * j + 1;
        if(c >= K)
            break;
        if(c + 1 < K && corerec_before(&h[c], &h[c + 1]))
            c++;
        if(!corerec_before(e, &h[c]))
            break;
        h[j] = h[c];
        j = c;
    }
    h[j] = *e;
}

/* What the owner tells every rank about a candidate's bound set in energy order. */
#define CORE_ND 15
struct CoreOut {
    double * d;              /* [nc][CORE_ND]: CorePos 3, CoreVel 3, SigmaX, SigmaV, mbPos 3, mbVel 3, has */
    MyIDType * mbID;         /* [nc] */
    int64_t * resv;          /* [nc][2 K]: (rank, index) of the reserved tracers, -1 */
};

/* The core (subhalo_core), the most-bound tracer and the MinNumTracerPart reserved tracers of the
 * round's alive candidates in (En, ID) order: every rank sends its first K tracers and K non-tracers
 * per candidate to the owner, which merges them; the results are reduced to every rank. Collective. */
static void
round_cores(const struct Coll * C, struct CoreOut * out)
{
    const int64_t nc = C->c1 - C->c0;
    const int CoreSize = C->sp->SubhaloCoreSize > 0 ? C->sp->SubhaloCoreSize : 1;
    const int NR = C->up->MinNumTracerPart;
    const int K = CoreSize > NR ? CoreSize : NR;
    const int TracerTypes = C->up->TracerTypes;
    const double BoxSize = C->up->BoxSize;
    int64_t i, c0;
    out->d = (double *) mymalloc("SubCollCoreOutD", sizeof(double) * CORE_ND * (nc + 1));
    out->mbID = (MyIDType *) mymalloc("SubCollCoreOutId", sizeof(MyIDType) * (nc + 1));
    out->resv = (int64_t *) mymalloc("SubCollCoreOutRes", sizeof(int64_t) * 2 * NR * (nc + 1));
    /* per candidate and kind (2 c: non-tracers, 2 c + 1: tracers): the heap capacity min(count, K)
     * and its slice of the record array */
    int64_t * hoff = (int64_t *) mymalloc("SubCollCoreHoff", sizeof(int64_t) * (2 * nc + 1));
    int64_t * hn = (int64_t *) mymalloc("SubCollCoreHn", sizeof(int64_t) * (2 * nc + 1));
    memset(hoff, 0, sizeof(int64_t) * (2 * nc + 1));
    memset(hn, 0, sizeof(int64_t) * (2 * nc + 1));
    for(i = 0; i < C->NumPart; i++)
        if(in_round(C, C->Cand[i]) && (C->Flag[i] & CF_BOUND) && C->cs[C->Cand[i]].state == SUBHALO_ST_ALIVE && part_live(&P[i]))
            hoff[2 * (C->Cand[i] - C->c0) + is_tracer(i, TracerTypes)]++;
    int64_t n = 0;
    for(c0 = 0; c0 < 2 * nc; c0++) {
        const int64_t cap = hoff[c0] < K ? hoff[c0] : K;
        hoff[c0] = n;
        n += cap;
    }
    hoff[2 * nc] = n;
    struct CoreRec * r = (struct CoreRec *) mymalloc("SubCollCoreRec", sizeof(struct CoreRec) * (n + 1));
    for(i = 0; i < C->NumPart; i++) {
        const int32_t c = C->Cand[i];
        if(!(in_round(C, c) && (C->Flag[i] & CF_BOUND) && C->cs[c].state == SUBHALO_ST_ALIVE && part_live(&P[i])))
            continue;
        struct CoreRec rec;
        struct CoreRec * e = &rec;
        memset(e, 0, sizeof(*e));
        e->cand = c;
        e->E = C->En[i];
        e->ID = P[i].ID;
        int d;
        for(d = 0; d < 3; d++) {
            e->Pos[d] = P[i].Pos[d];
            e->Vel[d] = P[i].Vel[d];
        }
        e->Mass = P[i].Mass;
        e->tracer = is_tracer(i, TracerTypes);
        e->rank = C->ThisTask;
        e->index = i;
        e->owner = C->plan->Cands[c].owner;
        const int64_t h = 2 * (c - C->c0) + e->tracer;
        core_heap_push(r + hoff[h], &hn[h], hoff[h + 1] - hoff[h], e);
    }
    /* compact (every heap is full unless the kind had fewer than K records) and order by owner */
    int64_t m = 0;
    for(c0 = 0; c0 < 2 * nc; c0++) {
        if(hoff[c0] != m)
            memmove(r + m, r + hoff[c0], sizeof(struct CoreRec) * hn[c0]);
        m += hn[c0];
    }
    qsort(r, m, sizeof(struct CoreRec), cmp_corerec_owner);
    int64_t * sendcounts = ta_malloc("SubCollCoreCnt", int64_t, C->NTask);
    memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
    for(i = 0; i < m; i++)
        sendcounts[r[i].owner]++;
    struct CoreRec * in = NULL;
    const int64_t nin = subhalo_alltoallv(r, sendcounts, sizeof(struct CoreRec), (void **) &in, 0, NULL, C->Comm);
    ta_free(sendcounts);
    qsort(in, nin, sizeof(struct CoreRec), cmp_corerec_owner);
    double * d = (double *) mymalloc("SubCollCoreD", sizeof(double) * CORE_ND * (nc + 1));
    MyIDType * id = (MyIDType *) mymalloc("SubCollCoreId", sizeof(MyIDType) * (nc + 1));
    int64_t * rv = (int64_t *) mymalloc("SubCollCoreRes", sizeof(int64_t) * 2 * NR * (nc + 1));
    memset(d, 0, sizeof(double) * CORE_ND * (nc + 1));
    memset(id, 0, sizeof(MyIDType) * (nc + 1));
    for(i = 0; i < 2 * NR * nc; i++)
        rv[i] = -1;
    int64_t a = 0;
    while(a < nin) {
        int64_t b = a;
        while(b < nin && in[b].cand == in[a].cand)
            b++;
        const int64_t c = in[a].cand - C->c0;
        double * o = d + CORE_ND * c;
        /* subhalo_core: the origin is the most bound particle, tracers first, then fill */
        int64_t num = 0, j;
        int pass, k;
        double msum = 0, pos[3] = {0, 0, 0}, pos2[3] = {0, 0, 0}, vel[3] = {0, 0, 0}, vel2[3] = {0, 0, 0};
        for(pass = 0; pass < 2 && num < CoreSize; pass++)
            for(j = a; j < b && num < CoreSize; j++) {
                if((pass == 0) != (in[j].tracer != 0))
                    continue;
                num++;
                const double mm = in[j].Mass;
                msum += mm;
                for(k = 0; k < 3; k++) {
                    const double dx = NEAREST(in[j].Pos[k] - in[a].Pos[k], BoxSize);
                    pos[k] += mm * dx;
                    pos2[k] += mm * dx * dx;
                    vel[k] += mm * in[j].Vel[k];
                    vel2[k] += mm * in[j].Vel[k] * in[j].Vel[k];
                }
            }
        double sx2 = 0, sv2 = 0;
        for(k = 0; k < 3; k++) {
            pos[k] /= msum;
            pos2[k] /= msum;
            vel[k] /= msum;
            vel2[k] /= msum;
            o[k] = subhalo_wrap_box(in[a].Pos[k] + pos[k], BoxSize);
            o[3 + k] = vel[k];
            const double vx = pos2[k] - pos[k] * pos[k], vv = vel2[k] - vel[k] * vel[k];
            sx2 += vx > 0 ? vx : 0;
            sv2 += vv > 0 ? vv : 0;
        }
        o[6] = sqrt(sx2);
        o[7] = sqrt(sv2);
        int64_t mb = a, nr = 0;
        for(j = a; j < b; j++)
            if(in[j].tracer) {
                mb = j;
                break;
            }
        for(k = 0; k < 3; k++) {
            o[8 + k] = in[mb].Pos[k];
            o[11 + k] = in[mb].Vel[k];
        }
        o[14] = 1;
        id[c] = in[mb].ID;
        for(j = a; j < b && nr < NR; j++)
            if(in[j].tracer) {
                rv[2 * (NR * c + nr)] = in[j].rank;
                rv[2 * (NR * c + nr) + 1] = in[j].index;
                nr++;
            }
        a = b;
    }
    if(nc > 0) {
        MPI_Allreduce(d, out->d, CORE_ND * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        MPI_Allreduce(id, out->mbID, nc, MPI_UINT64_T, MPI_MAX, C->Comm);
        MPI_Allreduce(rv, out->resv, 2 * NR * nc, MPI_INT64, MPI_MAX, C->Comm);
    }
    myfree(rv);
    myfree(id);
    myfree(d);
    myfree(in);
    myfree(r);
    myfree(hn);
    myfree(hoff);
}

static void
free_cores(struct CoreOut * o)
{
    myfree(o->resv);
    myfree(o->mbID);
    myfree(o->d);
}

/* The previous most-bound tracer of the round's dead candidates, as process_serial_key: the first
 * own-tag tracer in (BindRank, ID) order over all ranks. Collective. */
static void
dead_mb(struct Coll * C)
{
    const int64_t nc = C->c1 - C->c0;
    int64_t i, c;
    int d;
    int64_t * br = (int64_t *) mymalloc("SubCollMbBr", sizeof(int64_t) * 2 * (nc + 1));
    MyIDType * ids = (MyIDType *) mymalloc("SubCollMbId", sizeof(MyIDType) * 2 * (nc + 1));
    double * pv = (double *) mymalloc("SubCollMbPv", sizeof(double) * 12 * (nc + 1));
    for(c = 0; c < nc; c++) {
        br[c] = INT64_MAX;
        ids[c] = (MyIDType) -1;
    }
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || C->cs[cc].state != SUBHALO_ST_DIED || !part_live(&P[i]) || !is_tracer(i, C->up->TracerTypes)
           || P[i].TrackId != C->plan->Cands[cc].TrackId)
            continue;
        if(P[i].BindRank < br[cc - C->c0])
            br[cc - C->c0] = P[i].BindRank;
    }
    if(nc > 0)
        MPI_Allreduce(br, br + nc, nc, MPI_INT64, MPI_MIN, C->Comm);
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || C->cs[cc].state != SUBHALO_ST_DIED || !part_live(&P[i]) || !is_tracer(i, C->up->TracerTypes)
           || P[i].TrackId != C->plan->Cands[cc].TrackId || P[i].BindRank != br[nc + cc - C->c0])
            continue;
        if(P[i].ID < ids[cc - C->c0])
            ids[cc - C->c0] = P[i].ID;
    }
    if(nc > 0)
        MPI_Allreduce(ids, ids + nc, nc, MPI_UINT64_T, MPI_MIN, C->Comm);
    memset(pv, 0, sizeof(double) * 12 * (nc + 1));
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || C->cs[cc].state != SUBHALO_ST_DIED || P[i].ID != ids[nc + cc - C->c0] || !part_live(&P[i])
           || P[i].TrackId != C->plan->Cands[cc].TrackId)
            continue;
        for(d = 0; d < 3; d++) {
            pv[6 * (cc - C->c0) + d] = P[i].Pos[d];
            pv[6 * (cc - C->c0) + 3 + d] = P[i].Vel[d];
        }
    }
    if(nc > 0)
        MPI_Allreduce(pv, pv + 6 * nc, 6 * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    for(c = 0; c < nc; c++) {
        struct CandState * s = &C->cs[C->c0 + c];
        if(s->state != SUBHALO_ST_DIED)
            continue;
        s->has_mb = br[nc + c] < INT64_MAX;
        s->has_core = s->has_mb;
        if(!s->has_mb)
            continue;
        s->mbID = ids[nc + c];
        for(d = 0; d < 3; d++) {
            s->mbPos[d] = s->CorePos[d] = pv[6 * nc + 6 * c + d];
            s->mbVel[d] = s->CoreVel[d] = pv[6 * nc + 6 * c + 3 + d];
        }
        s->SigmaX = s->SigmaV = 0;
    }
    myfree(pv);
    myfree(ids);
    myfree(br);
}

/* ------------------------------------------------------------------ properties: the distributed ordered set */

/* Per-candidate results of a round, on every rank (the owner writes the record from them). */
#define HS_N 0
#define HS_NTR 1
#define HS_NT 2
#define HS_MT 8
#define HS_M 14
#define HS_K 15
#define HS_U 16
#define HS_J 17
#define HS_I 20
#define HS_SFR 26
#define HS_HE 27
#define HS_BHM 28
#define HS_MDOT 29
#define HS_GZ 30
#define HS_SZ 31
#define HS_GZE 32
#define HS_SZE (32 + NMETALS)
#define NHS (32 + 2 * NMETALS)

struct RoundProps {
    double * hs;             /* [nc][NHS] holder-side sums */
    double * bhm;            /* [nc] most massive BH subgrid mass, -1 */
    MyIDType * bhid;         /* [nc] its ID, (MyIDType) -1 */
    /* the radial profile */
    double * Mso;            /* [nc][NSO] */
    int64_t * kc;            /* [nc][NSO] crossing positions, -1 */
    double * prof;           /* [nc][8]: hbt best M/r, Rmax, vir best M/r, RmaxVir, rhalf, Renclose, (unused) x2 */
    double * jv;             /* [nc][9]: JVir sums 3, x sums 3, v sums 3 */
    float * shape;           /* [nc][2][5]: BtoA, CtoA, AxisA[3] (vir, 500c), -1 */
    double * dfr;            /* [nc][8]: Rs, RockstarCoreVel 3, BulkVel 3, Vrms (km/s), -1 */
};

struct RadRec {
    uint64_t key[3];         /* radix: ID, r, label (most significant last) */
    double r, m;
    double dx[3];            /* NEAREST(x - core) */
    double vel[3];
};

static void
radrec_radix(const void * a, void * radix, void * arg)
{
    memcpy(radix, ((const struct RadRec *) a)->key, 24);
}

static void
coll_eigen3(double mat[9], double eval[3], double evec[9], gsl_eigen_symmv_workspace * ws)
{
    gsl_matrix_view m = gsl_matrix_view_array(mat, 3, 3);
    gsl_vector_view ev = gsl_vector_view_array(eval, 3);
    gsl_matrix_view em = gsl_matrix_view_array(evec, 3, 3);
    gsl_eigen_symmv(&m.matrix, &ev.vector, &em.matrix, ws);
    gsl_eigen_symmv_sort(&ev.vector, &em.matrix, GSL_EIGEN_SORT_VAL_DESC);
}

/* The Allgood fit of subhalo_props.c (allgood_shape) for every candidate with an aperture,
 * the tensor sums reduced over the ranks each iteration. np[c]: particles of the aperture (the
 * global positions 0 .. np - 1), R[c] its radius; out: BtoA, CtoA, AxisA per candidate. */
static void
shapes_distributed(const struct Coll * C, const struct RadRec * rr, int64_t nr, const int64_t * pos, int64_t nc,
                   const int64_t * np, const double * R, double rfloor, float * out, int stride)
{
    int64_t c, k;
    int d, e, it;
    double * orth = (double *) mymalloc("SubCollShOrth", sizeof(double) * 9 * (nc + 1));
    double * ax2 = (double *) mymalloc("SubCollShAx", sizeof(double) * 3 * (nc + 1));
    double * prev = (double *) mymalloc("SubCollShPrev", sizeof(double) * 2 * (nc + 1));
    int * active = (int *) mymalloc("SubCollShAct", sizeof(int) * (nc + 1));
    double * T = (double *) mymalloc("SubCollShT", sizeof(double) * 8 * (nc + 1));
    double * TG = (double *) mymalloc("SubCollShTG", sizeof(double) * 8 * (nc + 1));
    gsl_eigen_symmv_workspace * ws = gsl_eigen_symmv_alloc(3);
    if(!ws)
        endrun(1, "Subhalo properties: gsl_eigen_symmv_alloc failed\n");
    int nact = 0;
    for(c = 0; c < nc; c++) {
        active[c] = np[c] >= SHAPE_MINPART && R[c] > 0;
        nact += active[c];
        for(d = 0; d < 9; d++)
            orth[9 * c + d] = (d % 4 == 0) ? 1 : 0;
        for(d = 0; d < 3; d++)
            ax2[3 * c + d] = R[c] * R[c];
        prev[2 * c] = prev[2 * c + 1] = 0;
    }
    for(it = 0; it < SHAPE_ITERATIONS && nact > 0; it++) {
        memset(T, 0, sizeof(double) * 8 * (nc + 1));
        for(k = 0; k < nr; k++) {
            c = (int64_t) rr[k].key[2];
            if(!active[c] || pos[k] >= np[c])
                continue;
            const double minq2 = (rfloor / R[c]) * (rfloor / R[c]);
            double q2 = 0;
            for(e = 0; e < 3; e++) {
                const double * o = orth + 9 * c + 3 * e;
                const double pe = o[0] * rr[k].dx[0] + o[1] * rr[k].dx[1] + o[2] * rr[k].dx[2];
                q2 += pe * pe / ax2[3 * c + e];
            }
            if(q2 < minq2)
                q2 = minq2;
            if(!(q2 <= 1))
                continue;
            const double w = rr[k].m / q2;
            double * t = T + 8 * c;
            t[6] += 1;
            t[7] += w;
            int q = 0;
            for(d = 0; d < 3; d++)
                for(e = 0; e <= d; e++)
                    t[q++] += w * rr[k].dx[d] * rr[k].dx[e];
        }
        MPI_Allreduce(T, TG, 8 * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        nact = 0;
        for(c = 0; c < nc; c++) {
            if(!active[c])
                continue;
            const double * t = TG + 8 * c;
            if(t[6] < SHAPE_MINPART || !(t[7] > 0)) {
                active[c] = 0;
                continue;
            }
            double M[9], eval[3], evec[9];
            int q = 0;
            for(d = 0; d < 3; d++)
                for(e = 0; e <= d; e++) {
                    M[3 * d + e] = t[q++] / t[7];
                    M[3 * e + d] = M[3 * d + e];
                }
            coll_eigen3(M, eval, evec, ws);
            if(!(eval[0] > 0 && eval[1] > 0 && eval[2] > 0)) {
                active[c] = 0;
                continue;
            }
            const double ba = sqrt(eval[1] / eval[0]), ca = sqrt(eval[2] / eval[0]);
            double A[3] = {evec[0], evec[3], evec[6]};
            int big = 0;
            for(d = 1; d < 3; d++)
                if(fabs(A[d]) > fabs(A[big]))
                    big = d;
            const double sgn = A[big] < 0 ? -1 : 1;
            float * o = out + stride * c;
            o[0] = (float) ba;
            o[1] = (float) ca;
            for(d = 0; d < 3; d++)
                o[2 + d] = (float) (sgn * A[d]);
            if(fabs(ba - prev[2 * c]) < 0.01 * prev[2 * c] && fabs(ca - prev[2 * c + 1]) < 0.01 * prev[2 * c + 1]) {
                active[c] = 0;
                continue;
            }
            prev[2 * c] = ba;
            prev[2 * c + 1] = ca;
            for(e = 0; e < 3; e++) {
                for(d = 0; d < 3; d++)
                    orth[9 * c + 3 * e + d] = evec[3 * d + e];
                ax2[3 * c + e] = R[c] * R[c] * eval[e] / eval[0];
            }
            nact++;
        }
    }
    gsl_eigen_symmv_free(ws);
    myfree(TG);
    myfree(T);
    myfree(active);
    myfree(prev);
    myfree(ax2);
    myfree(orth);
}

/* The compensated sums (subhalo_ksum_*) as a non-commutative MPI operation on pairs of doubles:
 * MPI applies it in rank order, the order of the radial ordered set. */
static MPI_Datatype KsumType = MPI_DATATYPE_NULL;
static MPI_Op KsumOp = MPI_OP_NULL;

static void
ksum_mpi_op(void * in, void * inout, int * len, MPI_Datatype * dt)
{
    const double * a = (const double *) in;
    double * b = (double *) inout;
    int i;
    (void) dt;
    for(i = 0; i < *len; i++) {
        double r[2] = {a[2 * i], a[2 * i + 1]};
        subhalo_ksum_merge(r, b + 2 * i);
        b[2 * i] = r[0];
        b[2 * i + 1] = r[1];
    }
}

static void
ksum_mpi_init(void)
{
    if(KsumOp != MPI_OP_NULL)
        return;
    MPI_Type_contiguous(2, MPI_DOUBLE, &KsumType);
    MPI_Type_commit(&KsumType);
    MPI_Op_create(ksum_mpi_op, 0, &KsumOp);
}

/* Per-candidate transient scratch of round_props on every rank, charged by subhalo_coll_estimate:
 * the vectors kept to the end plus the largest later stage (asserted below). */
#define COLL_NL (8 + 2 * SUBHALO_NSO)
#define COLL_DFR_NA 8                          /* (1) ncore, M, Sb 3, S2 3: pairs */
#define COLL_DFR_NW 4                          /* (4) the core: Mcore, Sc 3: pairs */
#define COLL_DFR_NB (5 * SUBHALO_NFW_MAXBINS)  /* (4) per bin: r at pb - 1, pb, pb + 1, m, Mc */
#define COLL_DFR_DOUBLES (2 * (2 * COLL_DFR_NA) + 2 * (SUBHALO_NFW_MAXBINS + 1) + 2 * COLL_DFR_NB + 2 * (2 * COLL_DFR_NW) + 2)
#define COLL_PROPS_KEPT_BYTES (2 * 8 + 4 * 8 + 2 * COLL_NL * 8)
#define COLL_PROPS_DFR_BYTES (COLL_DFR_DOUBLES * 8 + 3 * 8)
#define COLL_PROPS_SHAPE_BYTES ((9 + 3 + 2 + 8 + 8) * 8 + 4 + 8 + 8)
#define COLL_PROPS_CAND_BYTES (COLL_PROPS_KEPT_BYTES + COLL_PROPS_DFR_BYTES)
_Static_assert((NHS + 2) * 8 <= COLL_PROPS_CAND_BYTES && COLL_PROPS_KEPT_BYTES + 2 * COLL_NL * 8 <= COLL_PROPS_CAND_BYTES
               && COLL_PROPS_KEPT_BYTES + 9 * 8 <= COLL_PROPS_CAND_BYTES && COLL_PROPS_SHAPE_BYTES <= COLL_PROPS_DFR_BYTES,
               "COLL_PROPS_CAND_BYTES is not the largest per-candidate stage of round_props");
_Static_assert(COLL_PROPS_CAND_BYTES <= SUBHALO_INPLACE_CAND_PROPS_BYTES,
               "SUBHALO_INPLACE_CAND_PROPS_BYTES does not cover the per-candidate scratch of round_props");
/* Bytes per radial record kept through the properties (the record, its position, M(<= r)) and the
 * most allocations round_props holds at once (the round's vectors, the radial set, the deferred
 * recipe's), each costing the allocator up to two alignment blocks (utils/memory.c). */
#define COLL_PROPS_REC_BYTES (sizeof(struct RadRec) + 8 + 8)
#define COLL_PROPS_NALLOC 32
#define COLL_ALLOC_OVERHEAD (2. * 4096.)

/* subhalo_props_deferred's recipes on the round's radial set (rr by (label, r, ID), pos, Mc = M(<= r)):
 * Rockstar's aperture pos < kc[vir] for the velocities and the NFW fit (decisions 44, 47.1), in four
 * reductions (1)-(4). Results in rp->dfr (km/s, run length unit), -1 undefined. Collective. */
static void
deferred_distributed(const struct Coll * C, const struct CoreOut * core, const struct RadRec * rr, int64_t nr,
                     const int64_t * pos, const double * Mc, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const double * rho = C->ctx->rho;
    const double rfloor = C->up->h / 2.8;
    const double vkms = subhalo_vel_to_kms(C->up->atime);
    const int NB = SUBHALO_NFW_MAXBINS;
    const int NA = 2 * COLL_DFR_NA, NW = 2 * COLL_DFR_NW, NV = COLL_DFR_NB;
    int64_t c, k;
    int d, b;
    if(nc <= 0)
        return;
    ksum_mpi_init();
    /* the allocations COLL_PROPS_DFR_BYTES counts */
    double * A = (double *) mymalloc("SubCollDfA", sizeof(double) * NA * nc);
    double * AG = (double *) mymalloc("SubCollDfAG", sizeof(double) * NA * nc);
    double * X = (double *) mymalloc("SubCollDfX", sizeof(double) * (NB + 1) * nc);
    double * XG = (double *) mymalloc("SubCollDfXG", sizeof(double) * (NB + 1) * nc);
    double * V = (double *) mymalloc("SubCollDfV", sizeof(double) * NV * nc);
    double * VG = (double *) mymalloc("SubCollDfVG", sizeof(double) * NV * nc);
    double * W = (double *) mymalloc("SubCollDfW", sizeof(double) * NW * nc);
    double * WG = (double *) mymalloc("SubCollDfWG", sizeof(double) * NW * nc);
    int64_t * kc = (int64_t *) mymalloc("SubCollDfKc", sizeof(int64_t) * 3 * nc);   /* kc, ncore, nb */
    double * Rv = (double *) mymalloc("SubCollDfRv", sizeof(double) * 2 * nc);       /* Rvir, mpbin */
    for(c = 0; c < nc; c++) {
        kc[3 * c] = rp->kc[SUBHALO_NSO * c];
        Rv[2 * c] = kc[3 * c] >= 0 ? cbrt(3 * rp->Mso[SUBHALO_NSO * c] / (4 * M_PI * rho[0])) : -1;
    }

    /* (1) inside the aperture: velocity sums about the core velocity and the count inside 0.1 Rvir
     * (compensated pairs); (2) the largest particle mass and M(<= r) at kc - 1 (one record: a MAX) */
    memset(A, 0, sizeof(double) * NA * nc);
    for(c = 0; c < nc; c++)
        X[(NB + 1) * c] = X[(NB + 1) * c + 1] = 0;
    for(k = 0; k < nr; k++) {
        c = (int64_t) rr[k].key[2];
        if(pos[k] >= kc[3 * c])
            continue;
        const double * cp = core->d + CORE_ND * c;
        const double m = rr[k].m;
        double * a = A + NA * c;
        if(rr[k].r < 0.1 * Rv[2 * c])
            a[0] += 1;
        subhalo_ksum_add(a + 2, m);
        for(d = 0; d < 3; d++) {
            const double dv = rr[k].vel[d] - cp[3 + d];
            subhalo_ksum_add(a + 4 + 2 * d, m * dv);
            subhalo_ksum_add(a + 10 + 2 * d, m * dv * dv);
        }
        if(m > X[(NB + 1) * c])
            X[(NB + 1) * c] = m;
        if(pos[k] == kc[3 * c] - 1)
            X[(NB + 1) * c + 1] = Mc[k];
    }
    MPI_Allreduce(A, AG, COLL_DFR_NA * nc, KsumType, KsumOp, C->Comm);
    for(c = 0; c < nc; c++) {
        XG[2 * c] = X[(NB + 1) * c];
        XG[2 * c + 1] = X[(NB + 1) * c + 1];
    }
    MPI_Allreduce(XG, V, 2 * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    for(c = 0; c < nc; c++) {
        kc[3 * c + 1] = 0;
        kc[3 * c + 2] = 0;
        Rv[2 * c + 1] = 0;
        if(kc[3 * c] < 1)
            continue;
        const int64_t nap = kc[3 * c], nfit = kc[3 * c];
        const double * a = AG + NA * c;
        int64_t ncore = (int64_t) subhalo_ksum_value(a);
        if(ncore < SUBHALO_ROCKSTAR_CORE_MINPART)
            ncore = nap < SUBHALO_ROCKSTAR_CORE_MINPART ? nap : SUBHALO_ROCKSTAR_CORE_MINPART;
        kc[3 * c + 1] = ncore;
        /* the bin plan from the fit aperture's mass Mfit and largest mass, as the serial recipe */
        kc[3 * c + 2] = subhalo_nfw_bin_plan(nfit, V[2 * c + 1], V[2 * c], &Rv[2 * c + 1]);
    }

    /* (3) the closing position of each bin: the first record reaching its threshold, the global
     * minimum by a MAX of the negated positions */
    for(k = 0; k < (NB + 1) * nc; k++)
        X[k] = -1e300;
    {
        int64_t lab = -1;
        int nextb = 1;
        for(k = 0; k < nr; k++) {
            c = (int64_t) rr[k].key[2];
            if(c != lab) {
                lab = c;
                nextb = 1;
            }
            if(pos[k] >= kc[3 * c])
                continue;
            while(nextb <= kc[3 * c + 2] && Mc[k] >= subhalo_nfw_threshold(nextb, Rv[2 * c + 1])) {
                X[(NB + 1) * c + nextb] = -(double) pos[k];
                nextb++;
            }
        }
    }
    MPI_Allreduce(X, XG, (NB + 1) * nc, MPI_DOUBLE, MPI_MAX, C->Comm);

    /* (4) the radii at and around each closing position, the mass and cumulative mass there (one
     * record each: exact sums), and the core velocity sums (compensated pairs) */
    memset(V, 0, sizeof(double) * NV * nc);
    memset(W, 0, sizeof(double) * NW * nc);
    for(k = 0; k < nr; k++) {
        c = (int64_t) rr[k].key[2];
        double * v = V + NV * c;
        const int64_t pk = pos[k];
        if(pk < kc[3 * c + 1]) {
            const double * cp = core->d + CORE_ND * c;
            double * w = W + NW * c;
            subhalo_ksum_add(w, rr[k].m);
            for(d = 0; d < 3; d++)
                subhalo_ksum_add(w + 2 + 2 * d, rr[k].m * (rr[k].vel[d] - cp[3 + d]));
        }
        for(b = 1; b <= kc[3 * c + 2]; b++) {
            const double xb = XG[(NB + 1) * c + b];
            if(xb < -1e299)
                break;
            const int64_t pb = (int64_t) -xb;
            double * e = v + 5 * (b - 1);
            if(pk == pb - 1)
                e[0] += rr[k].r;
            if(pk == pb) {
                e[1] += rr[k].r;
                e[3] += rr[k].m;
                e[4] += Mc[k];
            }
            if(pk == pb + 1)
                e[2] += rr[k].r;
        }
    }
    MPI_Allreduce(V, VG, NV * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    MPI_Allreduce(W, WG, COLL_DFR_NW * nc, KsumType, KsumOp, C->Comm);
    for(c = 0; c < nc; c++) {
        if(kc[3 * c] < 1)
            continue;
        const double * v = VG + NV * c;
        const double * a = AG + NA * c, * w = WG + NW * c;
        const double * cp = core->d + CORE_ND * c;
        double * o = rp->dfr + 8 * c;
        /* the velocity trio: the serial recipe's function on the same sums */
        double Sb[3][2], S2[3][2], Sc[3][2];
        float bv[3], cv[3], vrms;
        for(d = 0; d < 3; d++) {
            Sb[d][0] = a[4 + 2 * d];
            Sb[d][1] = a[5 + 2 * d];
            S2[d][0] = a[10 + 2 * d];
            S2[d][1] = a[11 + 2 * d];
            Sc[d][0] = w[2 + 2 * d];
            Sc[d][1] = w[3 + 2 * d];
        }
        subhalo_deferred_velocities(bv, cv, &vrms, cp + 3, subhalo_ksum_value(a + 2), subhalo_ksum_value(w), Sb, S2, Sc, vkms);
        for(d = 0; d < 3; d++) {
            o[1 + d] = cv[d];
            o[4 + d] = bv[d];
        }
        o[7] = vrms;
        const int nb = (int) kc[3 * c + 2];
        if(nb <= 0)
            continue;
        const double mpbin = Rv[2 * c + 1];
        const int64_t nbound = (int64_t) rp->hs[NHS * c + HS_N];
        double edge[SUBHALO_NFW_MAXBINS + 1];
        edge[0] = 0;
        int nf = 0;
        for(b = 1; b <= nb; b++) {
            const double xb = XG[(NB + 1) * c + b];
            if(xb < -1e299)
                break;
            const int64_t pb = (int64_t) -xb;
            const double * e = v + 5 * (b - 1);
            const double rnext = pb + 1 < nbound ? e[2] : e[1];
            edge[b] = subhalo_nfw_edge(b, mpbin, e[0], pb > 0, e[1], rnext, e[4], e[3]);
            nf = b;
        }
        const double * prf = rp->prof + 8 * c;
        const double rs0 = subhalo_klypin_rs(Rv[2 * c], rp->Mso[SUBHALO_NSO * c], prf[2], prf[3]);
        const double rs = subhalo_nfw_fit(rs0, edge, nf, 3 * rfloor);
        o[0] = rs > 0 ? rs : -1;
    }
    myfree(Rv);
    myfree(kc);
    myfree(WG);
    myfree(W);
    myfree(VG);
    myfree(V);
    myfree(XG);
    myfree(X);
    myfree(AG);
    myfree(A);
}

/* The holder-side sums of the round's alive candidates (counts, masses, energies, J, inertia,
 * baryons) and the most massive BH, reduced into rp->hs/bhm/bhid (allocated by the caller);
 * returns the local bound count. Collective. */
static int64_t
round_holder_sums(const struct Coll * C, const struct CoreOut * core, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const struct SubhaloUnbindParams * up = C->up;
    const double a = up->atime, BoxSize = up->BoxSize;
    int64_t i, c;
    int d;
    double * hs = (double *) mymalloc("SubCollHsLoc", sizeof(double) * NHS * (nc + 1));
    double * bm = (double *) mymalloc("SubCollBhmLoc", sizeof(double) * (nc + 1));
    MyIDType * bid = (MyIDType *) mymalloc("SubCollBhidLoc", sizeof(MyIDType) * (nc + 1));
    memset(hs, 0, sizeof(double) * NHS * (nc + 1));
    for(c = 0; c < nc; c++) {
        bm[c] = -1;
        bid[c] = (MyIDType) -1;
    }
    int64_t nb = 0;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || !(C->Flag[i] & CF_BOUND) || C->cs[cc].state != SUBHALO_ST_ALIVE || !part_live(&P[i]))
            continue;
        nb++;
        const struct CandState * cs = &C->cs[cc];
        const double * cp = core->d + CORE_ND * (cc - C->c0);
        double * h = hs + NHS * (cc - C->c0);
        const double m = P[i].Mass;
        const int type = P[i].Type;
        h[HS_N] += 1;
        if(is_tracer(i, up->TracerTypes))
            h[HS_NTR] += 1;
        h[HS_NT + type] += 1;
        h[HS_MT + type] += m;
        h[HS_M] += m;
        double k2 = 0, dxc[3], dvc[3];
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[i].Pos[d] - cs->RefPos[d], BoxSize);
            const double dv = (P[i].Vel[d] - cs->RefVel[d]) / a + up->Hubble * a * dx;
            k2 += dv * dv;
            dxc[d] = NEAREST(P[i].Pos[d] - cp[d], BoxSize);
            dvc[d] = P[i].Vel[d] - cs->RefVel[d];
        }
        double phi = C->En[i] - 0.5 * k2;
        if(up->WithThermal)
            phi -= subhalo_thermal_energy(i, a);
        h[HS_K] += m * 0.5 * k2;
        h[HS_U] += m * phi;
        h[HS_J + 0] += m * (dxc[1] * dvc[2] - dxc[2] * dvc[1]);
        h[HS_J + 1] += m * (dxc[2] * dvc[0] - dxc[0] * dvc[2]);
        h[HS_J + 2] += m * (dxc[0] * dvc[1] - dxc[1] * dvc[0]);
        h[HS_I + 0] += m * dxc[0] * dxc[0];
        h[HS_I + 1] += m * dxc[1] * dxc[1];
        h[HS_I + 2] += m * dxc[2] * dxc[2];
        h[HS_I + 3] += m * dxc[0] * dxc[1];
        h[HS_I + 4] += m * dxc[0] * dxc[2];
        h[HS_I + 5] += m * dxc[1] * dxc[2];
        if(type == 0) {
            h[HS_HE] += m * P[i].HeIIIionized;
            h[HS_SFR] += SPHP(i).Sfr;
            h[HS_GZ] += SPHP(i).Metallicity * m;
            for(d = 0; d < NMETALS; d++)
                h[HS_GZE + d] += SPHP(i).Metals[d] * m;
        }
        else if(type == 4) {
            h[HS_SZ] += STARP(i).Metallicity * m;
            for(d = 0; d < NMETALS; d++)
                h[HS_SZE + d] += STARP(i).Metals[d] * m;
        }
        else if(type == 5) {
            const double bmass = BHP(i).Mass;
            h[HS_BHM] += bmass;
            h[HS_MDOT] += BHP(i).Mdot;
            const int64_t q = cc - C->c0;
            if(bmass > bm[q] || (bmass == bm[q] && P[i].ID < bid[q])) {
                bm[q] = bmass;
                bid[q] = P[i].ID;
            }
        }
    }
    if(nc > 0) {
        MPI_Allreduce(hs, rp->hs, NHS * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        MPI_Allreduce(bm, rp->bhm, nc, MPI_DOUBLE, MPI_MAX, C->Comm);
        for(c = 0; c < nc; c++)
            if(bm[c] != rp->bhm[c])
                bid[c] = (MyIDType) -1;
        MPI_Allreduce(bid, rp->bhid, nc, MPI_UINT64_T, MPI_MIN, C->Comm);
    }
    myfree(bid);
    myfree(bm);
    myfree(hs);
    return nb;
}

/* The properties of the round's alive candidates with the formulas of subhalo_props_hbt, _rockstar
 * and _baryons, from the final-pass energies, frames and cores. Collective; rp is allocated here
 * (bottom) and freed by the caller (free_round_props). */
static void
round_props(const struct Coll * C, const struct CoreOut * core, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const struct SubhaloUnbindParams * up = C->up;
    const double BoxSize = up->BoxSize;
    const double rfloor = up->h / 2.8;
    const double * rho = C->ctx->rho;
    int64_t i, c, k;
    int d, s;

    rp->hs = (double *) mymalloc("SubCollHs", sizeof(double) * NHS * (nc + 1));
    rp->bhm = (double *) mymalloc("SubCollBhm", sizeof(double) * (nc + 1));
    rp->bhid = (MyIDType *) mymalloc("SubCollBhid", sizeof(MyIDType) * (nc + 1));
    rp->Mso = (double *) mymalloc("SubCollMso", sizeof(double) * SUBHALO_NSO * (nc + 1));
    rp->kc = (int64_t *) mymalloc("SubCollKc", sizeof(int64_t) * SUBHALO_NSO * (nc + 1));
    rp->prof = (double *) mymalloc("SubCollProf", sizeof(double) * 8 * (nc + 1));
    rp->jv = (double *) mymalloc("SubCollJv", sizeof(double) * 9 * (nc + 1));
    rp->shape = (float *) mymalloc("SubCollShape", sizeof(float) * 10 * (nc + 1));
    for(k = 0; k < 10 * nc; k++)
        rp->shape[k] = -1;
    rp->dfr = (double *) mymalloc("SubCollDfr", sizeof(double) * 8 * (nc + 1));
    for(k = 0; k < 8 * nc; k++)
        rp->dfr[k] = -1;

    /* ---- the holder-side sums ---- */
    int64_t nb = round_holder_sums(C, core, rp);

    /* ---- the radial ordered set about the cores: (label, r, ID), M(<= r) by a vector Exscan ---- */
    int64_t ntot = 0;
    MPI_Allreduce(&nb, &ntot, 1, MPI_INT64, MPI_SUM, C->Comm);
    for(c = 0; c < nc; c++) {
        for(s = 0; s < SUBHALO_NSO; s++) {
            rp->kc[SUBHALO_NSO * c + s] = -1;
            rp->Mso[SUBHALO_NSO * c + s] = 0;
        }
        for(d = 0; d < 8; d++)
            rp->prof[8 * c + d] = -1;
        for(d = 0; d < 9; d++)
            rp->jv[9 * c + d] = 0;
    }
    if(ntot == 0)
        return;
    struct RadRec * rr = (struct RadRec *) mymalloc("SubCollRad", sizeof(struct RadRec) * (nb + 1));
    nb = 0;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || !(C->Flag[i] & CF_BOUND) || C->cs[cc].state != SUBHALO_ST_ALIVE || !part_live(&P[i]))
            continue;
        const double * cp = core->d + CORE_ND * (cc - C->c0);
        struct RadRec * e = &rr[nb++];
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            e->dx[d] = NEAREST(P[i].Pos[d] - cp[d], BoxSize);
            e->vel[d] = P[i].Vel[d];
            r2 += e->dx[d] * e->dx[d];
        }
        const double r = sqrt(r2);
        e->r = r > rfloor ? r : rfloor;
        e->m = P[i].Mass;
        e->key[0] = P[i].ID;
        e->key[1] = double_key(e->r);
        e->key[2] = (uint64_t) (cc - C->c0);
    }
    mpsort_mpi(rr, nb, sizeof(struct RadRec), radrec_radix, 24, NULL, C->Comm);
    /* positions and cumulative masses: a local prefix per label plus a vector Exscan, the masses as
     * compensated pairs in rank order, so M(<= r) is the serial recipe's sum */
    ksum_mpi_init();
    int64_t * cnt = (int64_t *) mymalloc("SubCollRadCnt", sizeof(int64_t) * 2 * (nc + 1));
    double * msum = (double *) mymalloc("SubCollRadM", sizeof(double) * 4 * (nc + 1));
    memset(cnt, 0, sizeof(int64_t) * 2 * (nc + 1));
    memset(msum, 0, sizeof(double) * 4 * (nc + 1));
    for(k = 0; k < nb; k++) {
        cnt[rr[k].key[2]]++;
        subhalo_ksum_add(msum + 2 * rr[k].key[2], rr[k].m);
    }
    MPI_Exscan(cnt, cnt + nc, nc, MPI_INT64, MPI_SUM, C->Comm);
    MPI_Exscan(msum, msum + 2 * nc, nc, KsumType, KsumOp, C->Comm);
    if(C->ThisTask == 0)
        for(c = 0; c < nc; c++) {
            cnt[nc + c] = 0;
            msum[2 * nc + 2 * c] = msum[2 * nc + 2 * c + 1] = 0;
        }
    int64_t * pos = (int64_t *) mymalloc("SubCollRadPos", sizeof(int64_t) * (nb + 1));
    /* local candidates of each per-label quantity (the global one is reduced below) */
    const int NL = COLL_NL;
    double * L = (double *) mymalloc("SubCollRadL", sizeof(double) * NL * (nc + 1));
    double * G = (double *) mymalloc("SubCollRadG", sizeof(double) * NL * (nc + 1));
    /* L: 0 hbt best, 1 vir best, 2 rhalf position (min), 3 Renclose (max r), 4.. kc[s] (max), (4 + NSO).. unused */
    for(c = 0; c < nc; c++) {
        double * l = L + NL * c;
        l[0] = -1;
        l[1] = 0;
        l[2] = 1e300;
        l[3] = -1;
        for(s = 0; s < SUBHALO_NSO; s++)
            l[4 + s] = -1;
    }
    double * Mc = (double *) mymalloc("SubCollRadMc", sizeof(double) * (nb + 1));
    {
        int64_t lab = -1, j = 0;
        double Mk[2] = {0, 0};
        for(k = 0; k < nb; k++) {
            c = (int64_t) rr[k].key[2];
            if(c != lab) {
                lab = c;
                j = 0;
                Mk[0] = msum[2 * nc + 2 * c];
                Mk[1] = msum[2 * nc + 2 * c + 1];
            }
            pos[k] = cnt[nc + c] + j++;
            subhalo_ksum_add(Mk, rr[k].m);
            const double M = subhalo_ksum_value(Mk);
            Mc[k] = M;
            const double r = rr[k].r;
            const double Mtot = rp->hs[NHS * c + HS_M];
            const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
            double * l = L + NL * c;
            for(s = 0; s < SUBHALO_NSO; s++)
                if(dens > rho[s])
                    l[4 + s] = (double) pos[k];
            if(M / r > l[0])
                l[0] = M / r;
            if(dens > rho[0] && M / r > l[1])
                l[1] = M / r;
            if(M >= 0.5 * Mtot && (double) pos[k] < l[2])
                l[2] = (double) pos[k];
            if(r > l[3])
                l[3] = r;
        }
    }
    /* maxima of the values and the crossing positions, the minimum of the half-mass position */
    double * MX = (double *) mymalloc("SubCollRadMx", sizeof(double) * NL * (nc + 1));
    for(c = 0; c < nc; c++) {
        double * l = L + NL * c, * g = G + NL * c;
        g[0] = l[0];
        g[1] = l[1];
        g[2] = -l[2];
        g[3] = l[3];
        for(s = 0; s < SUBHALO_NSO; s++)
            g[4 + s] = l[4 + s];
    }
    MPI_Allreduce(G, MX, NL * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    /* second scan: the position attaining each maximum first, the masses at the crossings, the radii */
    for(c = 0; c < nc; c++) {
        double * g = G + NL * c;
        for(d = 0; d < NL; d++)
            g[d] = 0;
        g[0] = g[1] = 1e300;        /* positions of the first maximum: min */
    }
    for(k = 0; k < nb; k++) {
        c = (int64_t) rr[k].key[2];
        const double * mx = MX + NL * c;
        double * g = G + NL * c;
        const double r = rr[k].r, M = Mc[k];
        const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
        if(M / r == mx[0] && (double) pos[k] < g[0])
            g[0] = (double) pos[k];
        if(mx[1] > 0 && dens > rho[0] && M / r == mx[1] && (double) pos[k] < g[1])
            g[1] = (double) pos[k];
    }
    double * G2 = (double *) mymalloc("SubCollRadG2", sizeof(double) * NL * (nc + 1));
    for(c = 0; c < nc; c++) {
        G[NL * c] = -G[NL * c];
        G[NL * c + 1] = -G[NL * c + 1];
    }
    MPI_Allreduce(G, G2, NL * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    for(c = 0; c < nc; c++) {
        double * g = G + NL * c;
        for(d = 0; d < NL; d++)
            g[d] = 0;
    }
    /* sums of the values at the chosen positions (one particle each) */
    for(k = 0; k < nb; k++) {
        c = (int64_t) rr[k].key[2];
        const double * mx = MX + NL * c, * p2 = G2 + NL * c;
        double * g = G + NL * c;
        const double pk = (double) pos[k];
        if(pk == -p2[0]) g[0] += rr[k].r;                 /* Rmax */
        if(pk == -p2[1]) g[1] += rr[k].r;                 /* RmaxVir */
        if(pk == -mx[2]) g[2] += rr[k].r;                 /* Rhalf */
        for(s = 0; s < SUBHALO_NSO; s++)
            if(mx[4 + s] >= 0 && pk == mx[4 + s])
                g[4 + s] += Mc[k];                         /* Mso */
    }
    MPI_Allreduce(G, G2, NL * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    for(c = 0; c < nc; c++) {
        const double * mx = MX + NL * c, * g = G2 + NL * c;
        double * pr = rp->prof + 8 * c;
        pr[0] = mx[0];                  /* best M/r (HBT, all bound) */
        pr[1] = g[0];                   /* Rmax */
        pr[2] = mx[1];                  /* best M/r inside the virial threshold, 0 if none */
        pr[3] = mx[1] > 0 ? g[1] : -1;  /* RmaxVir */
        pr[4] = g[2];                   /* Rhalf */
        pr[5] = mx[3];                  /* Renclose */
        for(s = 0; s < SUBHALO_NSO; s++) {
            rp->kc[SUBHALO_NSO * c + s] = (int64_t) mx[4 + s];
            rp->Mso[SUBHALO_NSO * c + s] = mx[4 + s] >= 0 ? g[4 + s] : 0;
        }
    }
    myfree(G2);
    myfree(MX);
    /* inside the virial aperture (crossing particle included, D4): JVir in the CoreVel frame and the
     * mean offsets (Rockstar properties.c:417-458) */
    {
        double * jl = (double *) mymalloc("SubCollJvLoc", sizeof(double) * 9 * (nc + 1));
        memset(jl, 0, sizeof(double) * 9 * (nc + 1));
        for(k = 0; k < nb; k++) {
            c = (int64_t) rr[k].key[2];
            if(pos[k] > rp->kc[SUBHALO_NSO * c])
                continue;
            const double * cp = core->d + CORE_ND * c;
            double dvk[3];
            for(d = 0; d < 3; d++)
                dvk[d] = rr[k].vel[d] - cp[3 + d];
            const double m = rr[k].m;
            double * j = jl + 9 * c;
            j[0] += m * (rr[k].dx[1] * dvk[2] - rr[k].dx[2] * dvk[1]);
            j[1] += m * (rr[k].dx[2] * dvk[0] - rr[k].dx[0] * dvk[2]);
            j[2] += m * (rr[k].dx[0] * dvk[1] - rr[k].dx[1] * dvk[0]);
            for(d = 0; d < 3; d++) {
                j[3 + d] += m * rr[k].dx[d];
                j[6 + d] += m * dvk[d];
            }
        }
        MPI_Allreduce(jl, rp->jv, 9 * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        myfree(jl);
    }
    /* the deferred recipes: Rs fit, RockstarCoreVel, BulkVel, Vrms */
    deferred_distributed(C, core, rr, nb, pos, Mc, rp);
    /* shapes at Rvir and R500c over the particles up to each crossing */
    {
        int64_t * np = (int64_t *) mymalloc("SubCollShNp", sizeof(int64_t) * (nc + 1));
        double * R = (double *) mymalloc("SubCollShR", sizeof(double) * (nc + 1));
        int ap;
        for(ap = 0; ap < 2; ap++) {
            const int sdef = ap == 0 ? 0 : 3;
            for(c = 0; c < nc; c++) {
                const int64_t kcs = rp->kc[SUBHALO_NSO * c + sdef];
                np[c] = kcs >= 0 ? kcs + 1 : 0;
                R[c] = kcs >= 0 ? cbrt(3 * rp->Mso[SUBHALO_NSO * c + sdef] / (4 * M_PI * rho[sdef])) : -1;
            }
            shapes_distributed(C, rr, nb, pos, nc, np, R, rfloor, rp->shape + 5 * ap, 10);
        }
        myfree(R);
        myfree(np);
    }
    myfree(Mc);
    myfree(G);
    myfree(L);
    myfree(pos);
    myfree(msum);
    myfree(cnt);
    myfree(rr);
}

static void
free_round_props(struct RoundProps * rp)
{
    myfree(rp->dfr);
    myfree(rp->shape);
    myfree(rp->jv);
    myfree(rp->prof);
    myfree(rp->kc);
    myfree(rp->Mso);
    myfree(rp->bhid);
    myfree(rp->bhm);
    myfree(rp->hs);
}

/* ------------------------------------------------------------------ sinking */

static int64_t
member_lookup_c(const struct Coll * C, int64_t TrackId)
{
    const struct SubhaloMember * m = member_of(C, TrackId);
    return m ? m - C->plan->Members : -1;
}

/* 1 if member a is a proper ancestor of member j in the processing tree. */
static int
member_ancestor(const struct SubhaloMember * M, int64_t j, int64_t a, int64_t nmem)
{
    int64_t u = M[j].Parent, steps = 0;
    while(u >= 0 && steps++ <= nmem) {
        if(u == a)
            return 1;
        u = M[u].Parent;
    }
    return 0;
}

/* HBT MergeRecursively for every alive candidate X of the round over its processing subtree: a
 * resolved descendant (alive or dead this call) whose core passes subhalo_sink_test sinks into X, a
 * sunk survivor's bound set joins X's source. Returns the number of absorbing candidates (all ranks). */
static int64_t
round_sinking(struct Coll * C)
{
    if(!C->sp->SubhaloMergeTrapped)
        return 0;
    struct SubhaloMember * M = C->plan->Members;
    const int64_t nmem = C->plan->NMembers;
    const struct SubhaloInplaceCand * Cd = C->plan->Cands;
    int64_t c, j, nabs = 0, nsink = 0;
    /* members that sank alive into a candidate of this round */
    char * sank = (char *) mymalloc("SubCollSank", nmem + 1);
    memset(sank, 0, nmem + 1);
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * x = &C->cs[c];
        if(x->state != SUBHALO_ST_ALIVE || !x->has_core)
            continue;
        const int64_t xm = Cd[c].member;
        for(j = 0; j < nmem; j++) {
            if(j == xm || M[j].g != M[xm].g || M[j].kind != SUBHALO_MEM_RESOLVED || !member_ancestor(M, j, xm, nmem))
                continue;
            int st;
            const double * pos, * vel;
            double sx, sv;
            if(M[j].cand >= 0) {
                const struct CandState * y = &C->cs[M[j].cand];
                st = y->state;
                if(!y->has_core) continue;
                pos = y->CorePos;
                vel = y->CoreVel;
                sx = y->SigmaX;
                sv = y->SigmaV;
            }
            else {
                st = M[j].state;
                if(!M[j].has_core) continue;
                pos = M[j].CorePos;
                vel = M[j].CoreVel;
                sx = M[j].SigmaX;
                sv = M[j].SigmaV;
            }
            if(st != SUBHALO_ST_ALIVE && st != SUBHALO_ST_DIED)
                continue;
            if(!subhalo_sink_test(pos, vel, x->CorePos, x->CoreVel, x->SigmaX, x->SigmaV, sx, sv, C->sp->SubhaloSinkSymmetric, C->up->BoxSize))
                continue;
            nsink++;
            if(M[j].cand >= 0) {
                C->cs[M[j].cand].state = SUBHALO_ST_SUNK;
                C->cs[M[j].cand].SinkTrackId = Cd[c].TrackId;
            }
            M[j].state = SUBHALO_ST_SUNK;
            M[j].SinkTrackId = Cd[c].TrackId;
            /* a survivor's bound set joins x */
            if(st == SUBHALO_ST_ALIVE) {
                sank[j] = 1;
                if(!x->absorbed) {
                    x->absorbed = 1;
                    nabs++;
                }
            }
        }
    }
    CollStats.NSunk += nsink;
    if(nabs == 0) {
        myfree(sank);
        return 0;
    }
    /* the bound sets move: serial ones by their BoundTrackId, in-place ones by their candidate */
    int64_t i;
    for(i = 0; i < C->NumPart; i++) {
        if(!part_live(&P[i]))
            continue;
        if(C->Flag[i] & CF_SBOUND) {
            const int64_t b = member_lookup_c(C, P[i].BoundTrackId);
            if(b >= 0 && sank[b] && M[b].cand < 0) {
                C->Cand[i] = M[member_lookup_c(C, M[b].SinkTrackId)].cand;
                C->Flag[i] &= ~CF_SBOUND;
            }
        }
        else if(C->Cand[i] >= 0 && (C->Flag[i] & CF_BOUND)) {
            const int64_t y = Cd[C->Cand[i]].member;
            if(sank[y]) {
                C->Cand[i] = M[member_lookup_c(C, M[y].SinkTrackId)].cand;
                C->Flag[i] &= ~(CF_BOUND | CF_RES);
            }
        }
    }
    myfree(sank);
    return nabs;
}

/* ------------------------------------------------------------------ claims and the commit of a round */

#define CF_WITHIN 64     /* round scratch: inside its candidate's relaxed source */
#define CF_ADDED 128     /* final pass: given to its serial binder (its serial claimant sank) */

struct UPosRec {
    uint64_t key[3];         /* radix: ID, E, label */
    int32_t rank, pad;
    int64_t index;
};

static void
uposrec_radix(const void * a, void * radix, void * arg)
{
    memcpy(radix, ((const struct UPosRec *) a)->key, 24);
}

/* CF_WITHIN for the unbound particles of each alive candidate inside its relaxed source (list
 * positions Nbound .. keep - 1 in (E, ID) order), by a global sort on (label, E, ID). Collective. */
static void
round_within(struct Coll * C)
{
    const int64_t nc = C->c1 - C->c0;
    int64_t i, n = 0, ntot = 0, k, c;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(in_round(C, cc) && C->cs[cc].state == SUBHALO_ST_ALIVE && !(C->Flag[i] & CF_BOUND) && part_live(&P[i])
           && C->cs[cc].keep > C->cs[cc].Nbound)
            n++;
    }
    MPI_Allreduce(&n, &ntot, 1, MPI_INT64, MPI_SUM, C->Comm);
    if(ntot == 0)
        return;
    struct UPosRec * u = (struct UPosRec *) mymalloc("SubCollUPos", sizeof(struct UPosRec) * (n + 1));
    n = 0;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!(in_round(C, cc) && C->cs[cc].state == SUBHALO_ST_ALIVE && !(C->Flag[i] & CF_BOUND) && part_live(&P[i])
             && C->cs[cc].keep > C->cs[cc].Nbound))
            continue;
        u[n].key[0] = P[i].ID;
        u[n].key[1] = double_key(C->En[i]);
        u[n].key[2] = (uint64_t) (cc - C->c0);
        u[n].rank = C->ThisTask;
        u[n].pad = 0;
        u[n].index = i;
        n++;
    }
    mpsort_mpi(u, n, sizeof(struct UPosRec), uposrec_radix, 24, NULL, C->Comm);
    int64_t * cnt = (int64_t *) mymalloc("SubCollUPosCnt", sizeof(int64_t) * 2 * (nc + 1));
    memset(cnt, 0, sizeof(int64_t) * 2 * (nc + 1));
    for(k = 0; k < n; k++)
        cnt[u[k].key[2]]++;
    MPI_Exscan(cnt, cnt + nc, nc, MPI_INT64, MPI_SUM, C->Comm);
    if(C->ThisTask == 0)
        for(c = 0; c < nc; c++)
            cnt[nc + c] = 0;
    /* the answers grouped by holder */
    int64_t * sendcounts = ta_malloc("SubCollUPosSend", int64_t, C->NTask);
    memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
    int64_t lab = -1, j = 0, nw = 0;
    for(k = 0; k < n; k++) {
        c = (int64_t) u[k].key[2];
        if(c != lab) {
            lab = c;
            j = 0;
        }
        const struct CandState * s = &C->cs[C->c0 + c];
        const int64_t p = cnt[nc + c] + j++;
        u[k].pad = p < s->keep - s->Nbound;
        if(u[k].pad) {
            sendcounts[u[k].rank]++;
            nw++;
        }
    }
    int64_t * off = ta_malloc("SubCollUPosOff", int64_t, C->NTask);
    int r;
    off[0] = 0;
    for(r = 1; r < C->NTask; r++)
        off[r] = off[r - 1] + sendcounts[r - 1];
    int64_t * idx = (int64_t *) mymalloc("SubCollUPosIdx", sizeof(int64_t) * (nw + 1));
    for(k = 0; k < n; k++)
        if(u[k].pad)
            idx[off[u[k].rank]++] = u[k].index;
    int64_t * in = NULL;
    const int64_t nin = subhalo_alltoallv(idx, sendcounts, sizeof(int64_t), (void **) &in, 0, NULL, C->Comm);
    for(k = 0; k < nin; k++)
        C->Flag[in[k]] |= CF_WITHIN;
    myfree(in);
    myfree(idx);
    ta_free(off);
    ta_free(sendcounts);
    myfree(cnt);
    myfree(u);
}

/* No valid deeper claim: neither a serial claim of an alive member nor the claim of an alive
 * in-place candidate (a claim of a candidate or member that has sunk since is void). */
static inline int
claimable(const struct Coll * C, int64_t i)
{
    if(C->Flag[i] & CF_SCLAIM) {
        const struct SubhaloMember * m = member_of(C, P[i].TrackId);
        if(m && m->state == SUBHALO_ST_ALIVE)
            return 0;
    }
    if(C->Claim[i] >= 0 && C->cs[C->Claim[i]].state == SUBHALO_ST_ALIVE)
        return 0;
    return 1;
}

/* The end of a round: the reserved tracers, the claims of the relaxed sources, and the
 * remainders passed to the parent candidates (HBT's append of the child's unbound particles). */
static void
round_commit(struct Coll * C, const struct CoreOut * fin)
{
    const int64_t nc = C->c1 - C->c0;
    const int NR = C->up->MinNumTracerPart;
    int64_t i, c;
    int q;
    for(c = 0; c < nc; c++) {
        if(C->cs[C->c0 + c].state != SUBHALO_ST_ALIVE)
            continue;
        for(q = 0; q < NR; q++) {
            const int64_t rk = fin->resv[2 * (NR * c + q)], ix = fin->resv[2 * (NR * c + q) + 1];
            if(rk == C->ThisTask && ix >= 0 && ix < C->NumPart)
                C->Flag[ix] |= CF_RES;
        }
    }
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc))
            continue;
        const struct CandState * s = &C->cs[cc];
        if(s->state == SUBHALO_ST_ALIVE) {
            if(((C->Flag[i] & CF_BOUND) || (C->Flag[i] & CF_WITHIN)) && claimable(C, i)) {
                C->Claim[i] = cc;
                C->ClaimE[i] = C->En[i];
            }
            if(!(C->Flag[i] & CF_BOUND)) {
                C->Cand[i] = C->plan->Cands[cc].parent;
                C->Flag[i] &= ~CF_RES;
            }
        }
        else {
            C->Cand[i] = C->plan->Cands[cc].parent;
            C->Flag[i] &= ~(CF_BOUND | CF_RES);
        }
        C->Flag[i] &= ~CF_WITHIN;
    }
}

/* ------------------------------------------------------------------ records */

/* Vmax as the serial key stores it: sqrt(G M/r / a) in Vel units, then the unit factor. */
static inline float
rec_vmax(double best, const struct SubhaloUnbindParams * up)
{
    float v = (float) sqrt(up->G * best / up->atime);
    v *= subhalo_vel_to_kms(1.0);
    return v;
}

/* The record of an alive candidate of the current round (owner only), with the formulas of
 * subhalo_props_hbt, subhalo_props_rockstar and subhalo_props_baryons and the unit conversions
 * of process_serial_key; the hierarchy columns and the exclusive/tail counts come later. */
static void
fill_alive_record(const struct Coll * C, int64_t c, const struct CoreOut * fin, const struct RoundProps * rp,
                  struct SubTrack * rec, const struct SubTrack * old)
{
    const struct SubhaloUnbindParams * up = C->up;
    const struct SubhaloCallCtx * ctx = C->ctx;
    const struct CandState * s = &C->cs[c];
    const struct SubhaloInplaceCand * cd = &C->plan->Cands[c];
    const struct SubhaloMember * m = &C->plan->Members[cd->member];
    const int64_t q = c - C->c0;
    const double a = up->atime, G = up->G, BoxSize = up->BoxSize;
    const double vu = subhalo_vel_to_kms(1.0), vkms = subhalo_vel_to_kms(a);
    const double * rho = ctx->rho;
    double ULen, UMass, UVel;
    subhalo_get_units(&ULen, &UMass, &UVel);
    const double lenfac = ULen / 3.085678e21;
    const double * h = rp->hs + NHS * q;
    const double * cp = fin->d + CORE_ND * q;
    const double * pr = rp->prof + 8 * q;
    const double * Mso = rp->Mso + SUBHALO_NSO * q;
    const int64_t * kc = rp->kc + SUBHALO_NSO * q;
    int d, t;

    subhalo_track_carry(rec, old, cd->TrackId, ctx->SnapNum);
    /* HBT (subhalo_props_hbt) */
    rec->Nbound = (int32_t) h[HS_N];
    rec->NboundTracer = (int32_t) h[HS_NTR];
    rec->Mbound = (float) h[HS_M];
    for(t = 0; t < 6; t++) {
        rec->NboundType[t] = (int32_t) h[HS_NT + t];
        rec->MboundType[t] = (float) h[HS_MT + t];
    }
    rec->Flags |= s->flags | SUBHALO_FLAG_INPLACE;
    for(d = 0; d < 3; d++) {
        rec->CorePos[d] = subhalo_wrap_box(cp[d] - ctx->Offset[d], BoxSize);
        rec->CoreVel[d] = (float) (cp[3 + d] / a);
        rec->CoMPos[d] = subhalo_wrap_box(s->RefPos[d] - ctx->Offset[d], BoxSize);
        rec->CoMVel[d] = (float) (s->RefVel[d] / a);
        rec->MostBoundPos[d] = subhalo_wrap_box(cp[8 + d] - ctx->Offset[d], BoxSize);
        rec->MostBoundVel[d] = (float) (cp[11 + d] / a);
    }
    rec->CoreSigmaX = (float) cp[6];
    rec->CoreSigmaV = (float) (cp[7] / a);
    rec->MostBoundID = fin->mbID[q];
    rec->Vmax = (float) sqrt(G * pr[0] / a);
    rec->Rmax = (float) pr[1];
    {
        const double K = h[HS_K], U = h[HS_U], M = h[HS_M];
        rec->Ekin = (float) (K / M);
        rec->Epot = (float) (0.5 * U / M);
        rec->KinToPot = rec->Epot != 0 ? (float) (K / M / fabs(0.5 * U / M)) : -1;
    }
    for(d = 0; d < 3; d++) {
        rec->CoreVel[d] *= vu;
        rec->MostBoundVel[d] *= vu;
        rec->CoMVel[d] *= vu;
    }
    rec->CoreSigmaV *= vu;
    rec->Vmax *= vu;
    rec->Ekin *= vu * vu;
    rec->Epot *= vu * vu;

    /* Rockstar (subhalo_props_rockstar) */
    int sd;
    for(sd = 0; sd < SUBHALO_NSO; sd++)
        rec->MBoundSO[sd] = rec->RBoundSO[sd] = -1;
    rec->VmaxVir = rec->RmaxVir = rec->Rhalf = rec->Renclose = -1;
    for(d = 0; d < 3; d++) {
        rec->J[d] = rec->JVir[d] = -1;
        rec->AxisA[d] = rec->AxisA500[d] = -1;
    }
    rec->SpinPeebles = rec->SpinBullock = rec->Xoff = rec->Voff = rec->RsKlypin = -1;
    rec->BtoA = rec->CtoA = rec->BtoA500 = rec->CtoA500 = -1;
    for(d = 0; d < 6; d++)
        rec->InertiaTensor[d] = -1;
    const double Mtot = h[HS_M];
    for(sd = 0; sd < SUBHALO_NSO; sd++)
        if(kc[sd] >= 0) {
            rec->MBoundSO[sd] = (float) Mso[sd];
            rec->RBoundSO[sd] = (float) cbrt(3 * Mso[sd] / (4 * M_PI * rho[sd]));
        }
    rec->Rhalf = (float) pr[4];
    rec->Renclose = (float) pr[5];
    const double Mvir = Mso[0];
    const double Rvir = kc[0] >= 0 ? cbrt(3 * Mvir / (4 * M_PI * rho[0])) : -1;
    const double best = pr[2], rbest = pr[3];
    if(kc[0] >= 0 && best > 0) {
        rec->VmaxVir = (float) (sqrt(G * best / a) * vu);
        rec->RmaxVir = (float) rbest;
    }
    double J[3], jmag = 0;
    for(d = 0; d < 3; d++) {
        J[d] = h[HS_J + d] / Mtot;
        jmag += J[d] * J[d];
        rec->J[d] = (float) (J[d] * vu * lenfac);
    }
    jmag = sqrt(jmag);
    for(d = 0; d < 6; d++)
        rec->InertiaTensor[d] = (float) (h[HS_I + d] / Mtot);
    {
        const double eps = (h[HS_K] + 0.5 * h[HS_U]) / Mtot;
        if(Mtot > 0)
            rec->SpinPeebles = (float) (jmag * sqrt(fabs(eps)) / (G * Mtot));
    }
    if(kc[0] >= 0 && Mvir > 0) {
        const double * jv = rp->jv + 9 * q;
        double jj = 0, xo = 0, vo = 0, Jv[3];
        for(d = 0; d < 3; d++) {
            Jv[d] = jv[d] / Mvir;
            jj += Jv[d] * Jv[d];
            rec->JVir[d] = (float) (Jv[d] * vu * lenfac);
            xo += (jv[3 + d] / Mvir) * (jv[3 + d] / Mvir);
            vo += (jv[6 + d] / Mvir) * (jv[6 + d] / Mvir);
        }
        rec->SpinBullock = (float) (sqrt(jj) / sqrt(2 * G * Mvir * a * Rvir));
        rec->Xoff = (float) sqrt(xo);
        rec->Voff = (float) (sqrt(vo) * vkms);
        if(best > 0) {
            const double f = Rvir * best / (Mvir * KLYPIN_RS_CONSTANT);
            double rs = rbest / KLYPIN_RMAX_TO_RS;
            if(f >= 4.625) {
                const double conc = subhalo_klypin_f_to_c(f);
                if(conc > 0)
                    rs = Rvir / conc;
            }
            rec->RsKlypin = (float) rs;
        }
    }
    /* the deferred columns (subhalo_props_deferred) */
    {
        const double * df = rp->dfr + 8 * q;
        rec->Rs = (float) df[0];
        for(d = 0; d < 3; d++) {
            rec->RockstarCoreVel[d] = (float) df[1 + d];
            rec->BulkVel[d] = (float) df[4 + d];
        }
        rec->Vrms = (float) df[7];
    }
    const float * sh = rp->shape + 10 * q;
    rec->BtoA = sh[0];
    rec->CtoA = sh[1];
    rec->BtoA500 = sh[5];
    rec->CtoA500 = sh[6];
    for(d = 0; d < 3; d++) {
        rec->AxisA[d] = sh[2 + d];
        rec->AxisA500[d] = sh[7 + d];
    }

    /* baryons (subhalo_props_baryons) */
    rec->StarFormationRate = (float) h[HS_SFR];
    rec->MassHeIonized = (float) h[HS_HE];
    rec->BlackholeMass = (float) h[HS_BHM];
    rec->BlackholeAccretionRate = (float) h[HS_MDOT];
    rec->MostMassiveBHID = rp->bhid[q];
    rec->GasMetalMass = rec->StellarMetalMass = -1;
    for(t = 0; t < NMETALS; t++)
        rec->GasMetalElemMass[t] = rec->StellarMetalElemMass[t] = -1;
    if(ctx->MetalReturnOn) {
        rec->GasMetalMass = (float) h[HS_GZ];
        rec->StellarMetalMass = (float) h[HS_SZ];
        for(t = 0; t < NMETALS; t++) {
            rec->GasMetalElemMass[t] = (float) h[HS_GZE + t];
            rec->StellarMetalElemMass[t] = (float) h[HS_SZE + t];
        }
    }
    /* a resurrected orphan: the orphan tracer state is not a Subhalo column */
    if(m->kind == SUBHALO_MEM_ORPHAN) {
        rec->SnapTracerLost = -1;
        subhalo_track_clear_orphan_set(rec);
        for(d = 0; d < 3; d++) {
            rec->TracerPos[d] = -1;
            rec->TracerVel[d] = -1;
        }
    }
}

/* ------------------------------------------------------------------ the hierarchy of the oversized keys */

struct HierCtx {
    const struct SubhaloMember * M;
    const float * Mb, * Vm;
    const MyIDType * Id;
};
static const struct HierCtx * HierSort;

static int
cmp_hier_mass(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    const struct HierCtx * h = HierSort;
    if(mass_before_v(h->Mb[x], h->Vm[x], h->Id[x], h->M[x].TrackId, h->Mb[y], h->Vm[y], h->Id[y], h->M[y].TrackId))
        return -1;
    if(mass_before_v(h->Mb[y], h->Vm[y], h->Id[y], h->M[y].TrackId, h->Mb[x], h->Vm[x], h->Id[x], h->M[x].TrackId))
        return 1;
    return 0;
}

static int
cmp_member_idx_g(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    const struct SubhaloMember * M = HierSort->M;
    if(M[x].g != M[y].g) return (M[x].g > M[y].g) - (M[x].g < M[y].g);
    return (M[x].TrackId > M[y].TrackId) - (M[x].TrackId < M[y].TrackId);
}

/* UpdateTracks of process_serial_key over every oversized key (CompareMass ranks, the most massive
 * promoted, parents the nearest surviving ancestors). maxmerge[gidx]: the largest previous Mbound of
 * the tracks that sank into the central or arrived as its satellites. Identical on every rank. */
static void
coll_hierarchy(struct Coll * C, double * maxmerge)
{
    struct SubhaloMember * M = C->plan->Members;
    const int64_t nmem = C->plan->NMembers;
    int64_t i, c, j;
    float * Mb = (float *) mymalloc("SubCollHierMb", sizeof(float) * (nmem + 1));
    float * Vm = (float *) mymalloc("SubCollHierVm", sizeof(float) * (nmem + 1));
    MyIDType * Id = (MyIDType *) mymalloc("SubCollHierId", sizeof(MyIDType) * (nmem + 1));
    int64_t * ord = (int64_t *) mymalloc("SubCollHierOrd", sizeof(int64_t) * (nmem + 1));
    int64_t * sv = (int64_t *) mymalloc("SubCollHierSv", sizeof(int64_t) * (nmem + 1));
    for(i = 0; i < nmem; i++) {
        Mb[i] = M[i].Mbound;
        Vm[i] = M[i].Vmax;
        Id[i] = M[i].MostBoundID;
    }
    /* the in-place candidates' states and current values */
    for(c = 0; c < C->plan->NCands; c++) {
        const int64_t m = C->plan->Cands[c].member;
        M[m].state = C->cs[c].state;
        M[m].SinkTrackId = C->cs[c].SinkTrackId;
        Mb[m] = M[m].Mbound = C->cs[c].Mbound_rec;
        Vm[m] = M[m].Vmax = C->cs[c].Vmax_rec;
        Id[m] = M[m].MostBoundID = C->cs[c].MostBoundID_rec;
    }
    struct HierCtx hc = {M, Mb, Vm, Id};
    HierSort = &hc;
    for(i = 0; i < nmem; i++)
        ord[i] = i;
    qsort(ord, nmem, sizeof(int64_t), cmp_member_idx_g);
    for(i = 0; i < C->plan->NInplace; i++)
        maxmerge[i] = 0;
    i = 0;
    while(i < nmem) {
        int64_t e = i;
        while(e < nmem && M[ord[e]].g == M[ord[i]].g)
            e++;
        const int64_t gidx = subhalo_plan_inplace_index(C->plan, M[ord[i]].g);
        int64_t ns = 0;
        for(j = i; j < e; j++) {
            const int64_t v = ord[j];
            M[v].frank = M[v].fdepth = -1;
            M[v].FinalParent = M[v].Central = SUBHALO_NOTRACK;
            if(M[v].state == SUBHALO_ST_ALIVE)
                sv[ns++] = v;
        }
        qsort(sv, ns, sizeof(int64_t), cmp_hier_mass);
        int64_t X = ns > 0 ? sv[0] : -1;
        for(j = 0; j < ns; j++)
            M[sv[j]].frank = (int32_t) j;
        /* the nearest surviving ancestor; top survivors under the promoted central */
        for(j = i; j < e; j++) {
            const int64_t v = ord[j];
            int64_t u = M[v].Parent, steps = 0;
            while(u >= 0 && M[u].state != SUBHALO_ST_ALIVE && steps++ <= nmem)
                u = M[u].Parent;
            if(v == X)
                M[v].FinalParent = SUBHALO_NOTRACK;
            else if(u >= 0 && M[u].state == SUBHALO_ST_ALIVE)
                M[v].FinalParent = M[u].TrackId;
            else
                M[v].FinalParent = X >= 0 ? M[X].TrackId : SUBHALO_NOTRACK;
            M[v].Central = X >= 0 ? M[X].TrackId : SUBHALO_NOTRACK;
        }
        for(j = 0; j < ns; j++) {
            const int64_t v = sv[j];
            int64_t u = v, dd = 0;
            while(M[u].FinalParent != SUBHALO_NOTRACK && dd <= ns) {
                u = member_lookup_c(C, M[u].FinalParent);
                dd++;
            }
            M[v].fdepth = (int32_t) dd;
        }
        /* the central's merger columns */
        if(X >= 0 && gidx >= 0) {
            double mm = 0;
            for(j = i; j < e; j++) {
                const int64_t v = ord[j];
                if(v == X || M[v].kind != SUBHALO_MEM_RESOLVED)
                    continue;
                if(M[v].state == SUBHALO_ST_ALIVE && (M[v].PrevRank <= 0 || M[v].PrevCentral != M[X].TrackId) && M[v].MboundPrev > mm)
                    mm = M[v].MboundPrev;
                if(M[v].state == SUBHALO_ST_SUNK && M[v].SinkTrackId == M[X].TrackId && M[v].MboundPrev > mm)
                    mm = M[v].MboundPrev;
            }
            maxmerge[gidx] = mm;
        }
        i = e;
    }
    HierSort = NULL;
    myfree(sv);
    myfree(ord);
    myfree(Id);
    myfree(Vm);
    myfree(Mb);
}

/* 1 if A is T or one of T's ancestors in the final hierarchy. */
static int
final_ancestor_c(const struct Coll * C, int64_t T, int64_t A)
{
    int steps = 0;
    while(T != SUBHALO_NOTRACK && steps++ <= C->plan->NMembers + 1) {
        if(T == A)
            return 1;
        const struct SubhaloMember * m = member_of(C, T);
        if(!m)
            return 0;
        T = m->FinalParent;
    }
    return 0;
}

/* record_mb of process_serial_key from a stored most-bound particle. */
static void
rec_mb(struct SubTrack * rec, MyIDType id, const double pos[3], const double vel[3], const struct SubhaloCallCtx * ctx)
{
    const double vfac = subhalo_vel_to_kms(ctx->atime);
    int d;
    rec->MostBoundID = id;
    for(d = 0; d < 3; d++) {
        rec->MostBoundPos[d] = subhalo_wrap_box(pos[d] - ctx->Offset[d], ctx->BoxSize);
        rec->MostBoundVel[d] = vel[d] * vfac;
    }
}

/* The hierarchy columns of the in-place records (owner) and of the serial sub-tree results,
 * a serial survivor that sank into an in-place candidate turned into an orphan. */
static void
coll_records(struct Coll * C, const double * maxmerge, const int64_t * slotof, const struct SubTrack * oldrec)
{
    struct SubhaloCallCtx * ctx = C->ctx;
    const int64_t TotNgroups = C->plan->TotNgroups;
    int64_t c, i;
    for(c = 0; c < C->plan->NCands; c++) {
        if(slotof[c] < 0)
            continue;
        struct SubhaloResult * rs = &ctx->Res[slotof[c]];
        struct SubTrack * rec = &rs->rec;
        const struct SubhaloInplaceCand * cd = &C->plan->Cands[c];
        const struct SubhaloMember * m = &C->plan->Members[cd->member];
        const struct CandState * s = &C->cs[c];
        const struct SubTrack * old = m->kind == SUBHALO_MEM_NEWBORN ? NULL : &oldrec[slotof[c] - ctx->IpResStart];
        const int64_t host = cd->g <= TotNgroups ? cd->g : -1;
        rs->ndesc = 0;
        rs->MaxMergerMbound = 0;
        rs->ipk = -1;
        rs->subkey = -1;
        rs->state = s->state;
        if(s->state == SUBHALO_ST_ALIVE) {
            rs->valid = 1;
            rs->history = 1;
            rs->newborn = m->kind == SUBHALO_MEM_NEWBORN;
            rec->HostGroup = host;
            rec->Host = host;
            rec->Rank = m->frank;
            rec->Depth = m->fdepth;
            rec->NestedParentTrackId = m->FinalParent;
            rec->CentralTrackId = m->Central;
            rec->SinkTrackId = -1;
            rec->DescendantTrackId = cd->TrackId;
            rec->SnapDeath = rec->SnapSink = -1;
            rec->Alive = 1;
            if(m->frank == 0) {
                rs->MaxMergerMbound = maxmerge[cd->gidx];
                rs->ipk = cd->gidx;
                /* Flags bit 2: the group's central decision was taken in the plan step */
                if(C->plan->Inplace[cd->gidx].central >= 0)
                    rec->Flags |= SUBHALO_FLAG_PLANCENTRAL;
            }
            continue;
        }
        rs->history = 0;
        rs->newborn = 0;
        if(m->kind == SUBHALO_MEM_NEWBORN) {
            rs->valid = 0;
            continue;
        }
        rs->valid = 1;
        *rec = *old;
        rec->HostGroup = host;
        rec->CentralTrackId = m->Central;
        if(m->kind != SUBHALO_MEM_RESOLVED)
            continue;   /* an orphan central that was not resurrected keeps its record */
        if(host >= 0 && m->Central != SUBHALO_NOTRACK && m->Central != cd->TrackId && rec->SnapLastIsolation < 0)
            rec->SnapLastIsolation = ctx->SnapNum - 1;
        if(s->has_mb)
            rec_mb(rec, s->mbID, s->mbPos, s->mbVel, ctx);
        subhalo_track_make_orphan(rec, s->state == SUBHALO_ST_SUNK ? s->SinkTrackId : SUBHALO_NOTRACK, ctx->SnapNum);
        rec->NestedParentTrackId = m->FinalParent;
        int t;
        for(t = 0; t < old->NTracers && t < SUBHALO_MAXTRACERS; t++)
            rs->DescOwner[t] = SUBHALO_DESC_PENDING;
        rs->ndesc = t;
    }
    /* the serial sub-tree results */
    for(i = 0; i < ctx->IpResStart; i++) {
        struct SubhaloResult * rs = &ctx->Res[i];
        if(rs->subkey < 0 || !rs->valid)
            continue;
        struct SubTrack * rec = &rs->rec;
        const struct SubhaloMember * m = member_of(C, rec->TrackId);
        if(!m)
            endrun(5, "Subhalo in place: sub-tree result %ld is not a member.\n", rec->TrackId);
        if(rs->state == SUBHALO_ST_ALIVE && m->state == SUBHALO_ST_ALIVE) {
            rec->Rank = m->frank;
            rec->Depth = m->fdepth;
            rec->NestedParentTrackId = m->FinalParent;
            rec->CentralTrackId = m->Central;
            if(m->frank == 0) {
                const int64_t gidx = subhalo_plan_inplace_index(C->plan, m->g);
                rs->MaxMergerMbound = maxmerge[gidx];
                rs->ipk = gidx;
                if(C->plan->Inplace[gidx].central >= 0)
                    rec->Flags |= SUBHALO_FLAG_PLANCENTRAL;
            }
            else
                rec->Flags &= ~SUBHALO_FLAG_PLANCENTRAL;
            continue;
        }
        if(rs->state == SUBHALO_ST_ALIVE) {
            /* sank into an in-place candidate: an orphan with the most-bound tracer of its set */
            const struct SubTrack * old = subhalo_find_member(ctx, rec->TrackId);
            if(!old)
                endrun(5, "Subhalo in place: no previous record of sub-tree member %ld.\n", rec->TrackId);
            const MyIDType mbid = rec->MostBoundID;
            double mpos[3], mvel[3];
            int d;
            for(d = 0; d < 3; d++) {
                mpos[d] = rec->MostBoundPos[d];
                mvel[d] = rec->MostBoundVel[d];
            }
            *rec = *old;
            rec->MostBoundID = mbid;
            for(d = 0; d < 3; d++) {
                rec->MostBoundPos[d] = mpos[d];
                rec->MostBoundVel[d] = mvel[d];
            }
            subhalo_track_make_orphan(rec, m->SinkTrackId, ctx->SnapNum);
            rs->history = 0;
            rs->state = SUBHALO_ST_SUNK;
            int t;
            for(t = 0; t < old->NTracers && t < SUBHALO_MAXTRACERS; t++)
                rs->DescOwner[t] = SUBHALO_DESC_PENDING;
            rs->ndesc = t;
            rs->MaxMergerMbound = 0;
        }
        /* a member that died in its key and then sank into an in-place candidate (its core is
         * the previous most-bound tracer, merge_subtree) */
        if(rs->state == SUBHALO_ST_DIED && m->state == SUBHALO_ST_SUNK) {
            rec->SinkTrackId = m->SinkTrackId;
            rec->SnapSink = ctx->SnapNum;
            rec->DescendantTrackId = m->SinkTrackId;
            rs->state = SUBHALO_ST_SUNK;
        }
        rec->HostGroup = m->g;
        rec->NestedParentTrackId = m->FinalParent;
        rec->CentralTrackId = m->Central;
        if(m->Central != SUBHALO_NOTRACK && m->Central != rec->TrackId && rec->SnapLastIsolation < 0)
            rec->SnapLastIsolation = ctx->SnapNum - 1;
    }
}

/* ------------------------------------------------------------------ the final tag pass */

struct TagRec {
    uint64_t key[3];         /* radix: ID, E, (candidate << 1) | not bound by it */
    int32_t rank, type;
    int64_t index;
};

static void
tagrec_radix(const void * a, void * radix, void * arg)
{
    memcpy(radix, ((const struct TagRec *) a)->key, 24);
}

/* The final tag of one particle of an oversized key (design 7.2 step 9, D10): a reserved tracer, else a
 * valid serial claim, else the deepest alive in-place claim; a claim across the final tree goes to the
 * binder (J2); a serial tail whose claimant sank goes to its surviving serial binder. */
struct TagDec {
    int64_t tag;             /* the final TrackId, SUBHALO_NOTRACK */
    int64_t bT;              /* the final BoundTrackId */
    int64_t tc;              /* the in-place candidate whose own-tag sequence ranks it, -1 */
    double E;                /* its energy there */
    int bound;               /* bound by tc */
    int j2;                  /* a claim given to its binding owner */
    int change;              /* serial repair: -1 none, 0 a serial tail taken, 1 a serial exclusive particle taken, 2 given to its serial binder */
    int64_t chT;             /* the serial track of the change */
};

enum { CHG_TAIL = 0, CHG_EXCL = 1, CHG_ADD = 2 };

static void
decide_tag(const struct Coll * C, int64_t i, struct TagDec * d)
{
    const struct SubhaloInplaceCand * Cd = C->plan->Cands;
    const uint8_t f = C->Flag[i];
    const int64_t T0 = P[i].TrackId, B0 = P[i].BoundTrackId;
    const int32_t cc = C->Cand[i];
    int64_t bT = SUBHALO_NOTRACK, bc = -1;
    if(f & CF_SBOUND)
        bT = B0;
    else if(cc >= 0 && (f & CF_BOUND) && C->cs[cc].state == SUBHALO_ST_ALIVE) {
        bT = Cd[cc].TrackId;
        bc = cc;
    }
    d->tag = SUBHALO_NOTRACK;
    d->tc = -1;
    d->bound = 0;
    d->E = 0;
    d->j2 = 0;
    d->change = -1;
    d->chT = SUBHALO_NOTRACK;
    const struct SubhaloMember * m0 = member_of(C, T0);
    const int sclaim = (f & CF_SCLAIM) && m0 && m0->state == SUBHALO_ST_ALIVE && m0->cand < 0;
    if((f & CF_RES) && bc >= 0) {
        d->tag = bT;
        d->tc = bc;
        d->bound = 1;
        d->E = C->En[i];
    }
    else if(sclaim) {
        d->tag = T0;
        if(bc >= 0 && !final_ancestor_c(C, T0, bT)) {
            d->tag = bT;
            d->tc = bc;
            d->bound = 1;
            d->E = C->En[i];
            d->j2 = 1;
        }
    }
    else if(C->Claim[i] >= 0 && C->cs[C->Claim[i]].state == SUBHALO_ST_ALIVE) {
        d->tc = C->Claim[i];
        d->tag = Cd[d->tc].TrackId;
        d->bound = bc == d->tc;
        d->E = d->bound ? C->En[i] : C->ClaimE[i];
        if(bc >= 0 && bc != d->tc && !final_ancestor_c(C, d->tag, bT)) {
            d->tag = bT;
            d->tc = bc;
            d->bound = 1;
            d->E = C->En[i];
            d->j2 = 1;
        }
    }
    else if(bc >= 0) {
        d->tag = bT;
        d->tc = bc;
        d->bound = 1;
        d->E = C->En[i];
    }
    else if(f & CF_SBOUND) {
        /* a serial tail whose claimant sank into an in-place candidate while its serial binder
         * survives: the binder's own, merged into its sequence by the binder's energy order */
        d->tag = bT;
        d->change = CHG_ADD;
        d->chT = bT;
    }
    /* a serial own-tag particle taken over by an in-place candidate */
    if((f & CF_SCLAIM) && m0 && m0->cand < 0 && m0->state == SUBHALO_ST_ALIVE && d->tag != T0) {
        d->change = B0 == T0 ? CHG_EXCL : CHG_TAIL;
        d->chT = T0;
    }
    d->bT = d->tag == SUBHALO_NOTRACK ? SUBHALO_NOTRACK : bT;
}

/* ---- the serial repair: a change of a serial sub-tree track's own-tag sequence, routed to the
 *      rank holding the track's result, which owns its ranks and counts ---- */
struct SerChg {
    int64_t T;               /* the serial track */
    int64_t key;             /* removal: the old BindRank; addition: g, set by the owner from its binder ordinals */
    int64_t f;               /* addition: the position in the binder's final-pass energy order (owner) */
    MyIDType ID;
    int64_t index;           /* the particle on its holder */
    int32_t holder, owner;
    int32_t kind, type;      /* CHG_* */
};

struct SerQuery {
    int64_t T;
    int64_t r;               /* the particle's old BindRank */
    int64_t index;
    int32_t holder, owner;
};

struct SerReply {
    int64_t index;
    int64_t rank;
    int32_t holder, pad;
};

static int
cmp_serchg_owner(const void * a, const void * b)
{
    const struct SerChg * x = (const struct SerChg *) a, * y = (const struct SerChg *) b;
    return (x->owner > y->owner) - (x->owner < y->owner);
}

/* the owner's order: by track, removals (by old rank) before additions (by g, f, ID) */
static int
cmp_serchg_track(const void * a, const void * b)
{
    const struct SerChg * x = (const struct SerChg *) a, * y = (const struct SerChg *) b;
    if(x->T != y->T) return (x->T > y->T) - (x->T < y->T);
    const int ax = x->kind == CHG_ADD, ay = y->kind == CHG_ADD;
    if(ax != ay) return ax - ay;
    if(x->key != y->key) return (x->key > y->key) - (x->key < y->key);
    if(x->f != y->f) return (x->f > y->f) - (x->f < y->f);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

static int
cmp_serquery_owner(const void * a, const void * b)
{
    const struct SerQuery * x = (const struct SerQuery *) a, * y = (const struct SerQuery *) b;
    return (x->owner > y->owner) - (x->owner < y->owner);
}

static int
cmp_serreply_holder(const void * a, const void * b)
{
    const struct SerReply * x = (const struct SerReply *) a, * y = (const struct SerReply *) b;
    return (x->holder > y->holder) - (x->holder < y->holder);
}

static int
cmp_int64_c(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

/* One affected track on its owner: its changes in the sorted list. */
struct RepTrack {
    int64_t T;
    int64_t r0, r1;          /* removals, by old rank */
    int64_t a0, a1;          /* additions, by (g, f, ID) */
    int64_t nbex;            /* its serial NboundExclusive (before the repair) */
};

/* entries of [lo, hi) of the sorted change list with key < x (le = 0) or key <= x (le = 1) */
static int64_t
chg_count_below(const struct SerChg * c, int64_t lo, int64_t hi, int64_t x, int le)
{
    const int64_t base = lo;
    while(lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if(c[mid].key < x || (le && c[mid].key == x)) lo = mid + 1;
        else hi = mid;
    }
    return lo - base;
}

/* The new dense rank of an own-tag particle with old rank r of the track: the removed ranks
 * below r leave, the additions placed before it (all of them for a tail particle) enter. */
static int64_t
rep_new_rank(const struct SerChg * c, const struct RepTrack * t, int64_t r)
{
    const int64_t nrem = chg_count_below(c, t->r0, t->r1, r, 0);
    const int64_t nadd = r < t->nbex ? chg_count_below(c, t->a0, t->a1, r, 1) : t->a1 - t->a0;
    return r - nrem + nadd;
}

/* The serial ranks and counts repaired: each change goes to the rank holding its track's result, which
 * corrects the counts and answers the new ranks of the track's particles; only the affected TrackIds
 * are replicated. chg (bottom) is freed by the caller. Collective. */
static void
serial_repair(struct Coll * C, struct SerChg * chg, int64_t nchg)
{
    struct SubhaloCallCtx * ctx = C->ctx;
    const double t0 = MPI_Wtime();
    int64_t k, i;
    int64_t loc[3] = {0, 0, nchg}, tot3[3];
    for(k = 0; k < nchg; k++) {
        if(chg[k].kind == CHG_ADD) loc[1]++;
        else loc[0]++;
    }
    MPI_Allreduce(loc, tot3, 3, MPI_INT64, MPI_SUM, C->Comm);
    CollStats.NStolen = tot3[0];
    CollStats.NAdded = tot3[1];
    if(tot3[2] == 0)
        return;
    /* 1. the changes to the track owners */
    int64_t * sendcounts = ta_malloc("SubRepSend", int64_t, C->NTask);
    memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
    for(k = 0; k < nchg; k++)
        sendcounts[chg[k].owner]++;
    qsort(chg, nchg, sizeof(struct SerChg), cmp_serchg_owner);
    struct SerChg * in = NULL;
    const int64_t nin = subhalo_alltoallv(chg, sendcounts, sizeof(struct SerChg), (void **) &in, 0, NULL, C->Comm);
    /* 2. the binder ordinals of the additions (this rank processed the binder's key) */
    int bad = 0;
    for(k = 0; k < nin; k++) {
        if(in[k].kind != CHG_ADD)
            continue;
        int64_t lo = 0, hi = ctx->NBindOrd;
        while(lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if(ctx->BindOrd[mid].ID < in[k].ID) lo = mid + 1;
            else hi = mid;
        }
        if(lo >= ctx->NBindOrd || ctx->BindOrd[lo].ID != in[k].ID) {
            if(bad++ < 5)
                message(1, "Subhalo in place: particle %lu given to its binder %ld has no binder ordinal.\n", (unsigned long) in[k].ID, in[k].T);
            continue;
        }
        in[k].key = ctx->BindOrd[lo].g;
        in[k].f = ctx->BindOrd[lo].f;
    }
    if(MPIU_Any(bad > 0, C->Comm))
        endrun(5, "Subhalo in place: tails given to their serial binder without a binder ordinal.\n");
    qsort(in, nin, sizeof(struct SerChg), cmp_serchg_track);
    /* the affected tracks of this rank, their runs, and the corrections of their records */
    int64_t ntr = 0;
    for(k = 0; k < nin; k++)
        if(k == 0 || in[k].T != in[k - 1].T)
            ntr++;
    struct RepTrack * rt = (struct RepTrack *) mymalloc("SubRepTracks", sizeof(struct RepTrack) * (ntr + 1));
    /* this rank's sub-tree results by TrackId */
    int64_t nres = 0;
    for(i = 0; i < ctx->IpResStart; i++)
        if(ctx->Res[i].subkey >= 0 && ctx->Res[i].valid)
            nres++;
    int64_t * rmap = (int64_t *) mymalloc("SubRepResMap", sizeof(int64_t) * 2 * (nres + 1));
    nres = 0;
    for(i = 0; i < ctx->IpResStart; i++)
        if(ctx->Res[i].subkey >= 0 && ctx->Res[i].valid) {
            rmap[2 * nres] = ctx->Res[i].rec.TrackId;
            rmap[2 * nres + 1] = i;
            nres++;
        }
    qsort(rmap, nres, 2 * sizeof(int64_t), cmp_int64_c);
    ntr = 0;
    k = 0;
    while(k < nin) {
        struct RepTrack * t = &rt[ntr++];
        t->T = in[k].T;
        t->r0 = k;
        while(k < nin && in[k].T == t->T && in[k].kind != CHG_ADD)
            k++;
        t->r1 = t->a0 = k;
        while(k < nin && in[k].T == t->T)
            k++;
        t->a1 = k;
        const struct SubhaloMember * m = member_of(C, t->T);
        if(!m || m->cand >= 0 || m->resrank != C->ThisTask)
            endrun(5, "Subhalo in place: repair of track %ld on rank %d, which does not hold its result.\n", t->T, C->ThisTask);
        t->nbex = m->NboundExclusive;
        /* the additions: their new ranks (k-th addition after g own exclusive particles, less
         * the removed ones below g) are answered with the queries below */
        int64_t lo = 0, hi = nres;
        while(lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if(rmap[2 * mid] < t->T) lo = mid + 1;
            else hi = mid;
        }
        if(lo >= nres || rmap[2 * lo] != t->T)
            endrun(5, "Subhalo in place: no result of serial track %ld on its owner %d.\n", t->T, C->ThisTask);
        struct SubTrack * rec = &ctx->Res[rmap[2 * lo + 1]].rec;
        if(rec->Nbound <= 0)
            continue;
        int64_t j;
        int64_t nex = rec->NboundExclusive;
        for(j = t->r0; j < t->r1; j++) {
            if(in[j].kind == CHG_EXCL) {
                nex--;
                rec->NboundExclusiveType[in[j].type]--;
            }
            else
                rec->NtailType[in[j].type]--;
        }
        for(j = t->a0; j < t->a1; j++) {
            nex++;
            rec->NboundExclusiveType[in[j].type]++;
        }
        if(!subhalo_count_fits_i32((double) nex) || nex > rec->Nbound)
            bad = 1;
        rec->NboundExclusive = (int32_t) nex;
    }
    myfree(rmap);
    /* 3. the affected TrackIds to every rank (bounded by the member table) */
    int64_t * aff = (int64_t *) mymalloc("SubRepAffLoc", sizeof(int64_t) * (ntr + 1));
    for(k = 0; k < ntr; k++)
        aff[k] = rt[k].T;
    int64_t naff = 0;
    MPI_Allreduce(&ntr, &naff, 1, MPI_INT64, MPI_SUM, C->Comm);
    subhalo_check_mpi_count(naff, "the serial tracks to repair", C->Comm);
    int * counts = ta_malloc("SubRepAffCnt", int, C->NTask);
    int * displ = ta_malloc("SubRepAffDispl", int, C->NTask);
    int nloc = (int) ntr;
    MPI_Allgather(&nloc, 1, MPI_INT, counts, 1, MPI_INT, C->Comm);
    int64_t s = 0;
    int r;
    for(r = 0; r < C->NTask; r++) {
        displ[r] = (int) s;
        s += counts[r];
    }
    int64_t * allaff = (int64_t *) mymalloc("SubRepAff", sizeof(int64_t) * (naff + 1));
    MPI_Allgatherv(aff, nloc, MPI_INT64, allaff, counts, displ, MPI_INT64, C->Comm);
    ta_free(displ);
    ta_free(counts);
    qsort(allaff, naff, sizeof(int64_t), cmp_int64_c);
    /* 4. the remaining own-tag particles of the affected tracks ask their new rank */
    int64_t nq = 0;
    for(i = 0; i < C->NumPart; i++) {
        if(!(C->Flag[i] & CF_UNIV) || (C->Flag[i] & CF_ADDED) || P[i].TrackId < 0 || !part_live(&P[i]))
            continue;
        if(bsearch(&P[i].TrackId, allaff, naff, sizeof(int64_t), cmp_int64_c))
            nq++;
    }
    struct SerQuery * q = (struct SerQuery *) mymalloc("SubRepQ", sizeof(struct SerQuery) * (nq + 1));
    memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
    nq = 0;
    for(i = 0; i < C->NumPart; i++) {
        if(!(C->Flag[i] & CF_UNIV) || (C->Flag[i] & CF_ADDED) || P[i].TrackId < 0 || !part_live(&P[i]))
            continue;
        if(!bsearch(&P[i].TrackId, allaff, naff, sizeof(int64_t), cmp_int64_c))
            continue;
        const struct SubhaloMember * m = member_of(C, P[i].TrackId);
        if(!m || m->resrank < 0)
            endrun(5, "Subhalo in place: affected serial track %ld without a result rank.\n", P[i].TrackId);
        q[nq].T = P[i].TrackId;
        q[nq].r = P[i].BindRank;
        q[nq].index = i;
        q[nq].holder = C->ThisTask;
        q[nq].owner = m->resrank;
        sendcounts[m->resrank]++;
        nq++;
    }
    qsort(q, nq, sizeof(struct SerQuery), cmp_serquery_owner);
    struct SerQuery * qin = NULL;
    const int64_t nqin = subhalo_alltoallv(q, sendcounts, sizeof(struct SerQuery), (void **) &qin, 0, NULL, C->Comm);
    /* 5. the answers: the additions' ranks and the queries' ranks, back to the holders */
    int64_t nadd = 0;
    for(k = 0; k < nin; k++)
        if(in[k].kind == CHG_ADD)
            nadd++;
    struct SerReply * rep = (struct SerReply *) mymalloc("SubRepReply", sizeof(struct SerReply) * (nadd + nqin + 1));
    memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
    int64_t nrep = 0;
    for(k = 0; k < ntr; k++) {
        const struct RepTrack * t = &rt[k];
        int64_t j;
        for(j = t->a0; j < t->a1; j++) {
            const int64_t g = in[j].key;
            const int64_t nr = g - chg_count_below(in, t->r0, t->r1, g, 0) + (j - t->a0);
            if(!subhalo_rank_fits_i32(nr))
                bad = 1;
            rep[nrep].index = in[j].index;
            rep[nrep].rank = nr;
            rep[nrep].holder = in[j].holder;
            rep[nrep].pad = 0;
            sendcounts[in[j].holder]++;
            nrep++;
        }
    }
    for(k = 0; k < nqin; k++) {
        int64_t lo = 0, hi = ntr;
        while(lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if(rt[mid].T < qin[k].T) lo = mid + 1;
            else hi = mid;
        }
        if(lo >= ntr || rt[lo].T != qin[k].T)
            endrun(5, "Subhalo in place: rank query of track %ld on rank %d, which has no change of it.\n", qin[k].T, C->ThisTask);
        const int64_t nr = rep_new_rank(in, &rt[lo], qin[k].r);
        if(!subhalo_rank_fits_i32(nr))
            bad = 1;
        rep[nrep].index = qin[k].index;
        rep[nrep].rank = nr;
        rep[nrep].holder = qin[k].holder;
        rep[nrep].pad = 0;
        sendcounts[qin[k].holder]++;
        nrep++;
    }
    /* no rank or count beyond the int32 columns, decided on every rank before writing */
    if(MPIU_Any(bad, C->Comm))
        endrun(1, "Subhalo in place: a repaired serial BindRank or count is beyond the catalogue's int32 range.\n");
    qsort(rep, nrep, sizeof(struct SerReply), cmp_serreply_holder);
    struct SerReply * rin = NULL;
    const int64_t nrin = subhalo_alltoallv(rep, sendcounts, sizeof(struct SerReply), (void **) &rin, 0, NULL, C->Comm);
    for(k = 0; k < nrin; k++) {
        const int64_t ix = rin[k].index;
        P[ix].BindRank = (int32_t) rin[k].rank;
        C->Flag[ix] &= ~CF_ADDED;
    }
    int64_t loc2[2] = {nq, nrep}, tot2[2];
    MPI_Allreduce(loc2, tot2, 2, MPI_INT64, MPI_SUM, C->Comm);
    CollStats.NRepairQueries = tot2[0];
    CollStats.NRepairTracks = naff;
    myfree(rin);
    myfree(rep);
    myfree(qin);
    myfree(q);
    myfree(allaff);
    myfree(aff);
    myfree(rt);
    myfree(in);
    ta_free(sendcounts);
    CollStats.RepairSeconds = MPI_Wtime() - t0;
}

/* The final tags of every particle of an oversized key (decide_tag); the in-place own-tag ranks in
 * list order (bound by final-pass energy, then the relaxed tail by removal energy) and counts; the
 * serial ranks and counts repaired (serial_repair). Collective. */
static void
coll_final_tags(struct Coll * C, const int64_t * slotof)
{
    struct SubhaloCallCtx * ctx = C->ctx;
    const struct SubhaloInplaceCand * Cd = C->plan->Cands;
    const int64_t ncand = C->plan->NCands;
    const double t0 = MPI_Wtime();
    int64_t i, k, c;
    int t;
    int64_t nt = 0, nchg = 0, nwritten = 0, nrel = 0, nj2 = 0;
    struct TagDec dec;
    /* the exact sizes first: the in-place own-tag records and the serial changes */
    for(i = 0; i < C->NumPart; i++) {
        if(!(C->Flag[i] & CF_UNIV) || (C->Flag[i] & CF_RELEASE))
            continue;
        decide_tag(C, i, &dec);
        if(dec.tag != SUBHALO_NOTRACK && dec.tc >= 0)
            nt++;
        if(dec.change >= 0)
            nchg++;
    }
    /* per candidate and type: exclusive and tail own-tag counts */
    int64_t * cnt = (int64_t *) mymalloc("SubCollTagCnt", sizeof(int64_t) * 12 * (ncand + 1));
    memset(cnt, 0, sizeof(int64_t) * 12 * (ncand + 1));
    int64_t * g = (int64_t *) mymalloc("SubCollTagCntG", sizeof(int64_t) * 12 * (ncand + 1));
    struct SerChg * chg = (struct SerChg *) mymalloc("SubCollSerChg", sizeof(struct SerChg) * (nchg + 1));
    struct TagRec * tr = (struct TagRec *) mymalloc("SubCollTagRec", sizeof(struct TagRec) * (nt + 1));
    nt = nchg = 0;
    for(i = 0; i < C->NumPart; i++) {
        const uint8_t f = C->Flag[i];
        if(f & CF_RELEASE) {
            P[i].TrackId = -1;
            P[i].BindRank = -1;
            P[i].BoundTrackId = -1;
            nrel++;
            continue;
        }
        if(!(f & CF_UNIV))
            continue;
        nwritten++;
        decide_tag(C, i, &dec);
        nj2 += dec.j2;
        if(dec.change >= 0) {
            struct SerChg * e = &chg[nchg++];
            memset(e, 0, sizeof(*e));
            const struct SubhaloMember * m = member_of(C, dec.chT);
            if(!m || m->resrank < 0)
                endrun(5, "Subhalo in place: serial track %ld changed without a result rank.\n", dec.chT);
            e->T = dec.chT;
            e->key = dec.change == CHG_ADD ? 0 : P[i].BindRank;
            e->f = 0;
            e->ID = P[i].ID;
            e->index = i;
            e->holder = C->ThisTask;
            e->owner = m->resrank;
            e->kind = dec.change;
            e->type = P[i].Type;
            if(dec.change == CHG_ADD)
                C->Flag[i] |= CF_ADDED;
        }
        P[i].TrackId = dec.tag;
        P[i].BoundTrackId = dec.bT;
        if(dec.tag == SUBHALO_NOTRACK) {
            P[i].BindRank = -1;
            continue;
        }
        if(dec.tc < 0)
            continue;           /* a serial tag: its rank is repaired below */
        struct TagRec * e = &tr[nt++];
        e->key[0] = P[i].ID;
        e->key[1] = double_key(dec.E);
        e->key[2] = ((uint64_t) dec.tc << 1) | (uint64_t) (dec.bound ? 0 : 1);
        e->rank = C->ThisTask;
        e->type = P[i].Type;
        e->index = i;
        cnt[12 * dec.tc + (dec.bound ? 0 : 6) + P[i].Type]++;
    }
    /* ---- the in-place exclusive and tail counts, checked against the int32 columns on every rank
     *      before any rank is written ---- */
    if(ncand > 0)
        MPI_Allreduce(cnt, g, 12 * ncand, MPI_INT64, MPI_SUM, C->Comm);
    for(c = 0; c < ncand; c++) {
        int64_t own = 0;
        for(t = 0; t < 12; t++)
            own += g[12 * c + t];
        if(!subhalo_count_fits_i32((double) own))
            endrun(1, "Subhalo in place: candidate %ld has %ld own-tag particles, beyond the catalogue's int32 ranks and counts.\n",
                   Cd[c].TrackId, own);
    }
    /* ---- the in-place own-tag ranks: (candidate, bound first, E, ID) ---- */
    {
        int64_t tot = 0;
        MPI_Allreduce(&nt, &tot, 1, MPI_INT64, MPI_SUM, C->Comm);
        if(tot > 0) {
            mpsort_mpi(tr, nt, sizeof(struct TagRec), tagrec_radix, 24, NULL, C->Comm);
            int64_t * off = (int64_t *) mymalloc("SubCollTagOff", sizeof(int64_t) * 2 * (ncand + 1));
            memset(off, 0, sizeof(int64_t) * 2 * (ncand + 1));
            for(k = 0; k < nt; k++)
                off[tr[k].key[2] >> 1]++;
            MPI_Exscan(off, off + ncand, ncand, MPI_INT64, MPI_SUM, C->Comm);
            if(C->ThisTask == 0)
                for(c = 0; c < ncand; c++)
                    off[ncand + c] = 0;
            int64_t * sendcounts = ta_malloc("SubCollTagSend", int64_t, C->NTask);
            memset(sendcounts, 0, sizeof(int64_t) * C->NTask);
            for(k = 0; k < nt; k++)
                sendcounts[tr[k].rank]++;
            int64_t * o = ta_malloc("SubCollTagO", int64_t, C->NTask);
            int r;
            o[0] = 0;
            for(r = 1; r < C->NTask; r++)
                o[r] = o[r - 1] + sendcounts[r - 1];
            int64_t * ans = (int64_t *) mymalloc("SubCollTagAns", sizeof(int64_t) * 2 * (nt + 1));
            int64_t lab = -1, j = 0;
            for(k = 0; k < nt; k++) {
                const int64_t cc = (int64_t) (tr[k].key[2] >> 1);
                if(cc != lab) {
                    lab = cc;
                    j = 0;
                }
                const int64_t rank = off[ncand + cc] + j++;
                /* guarded collectively by the totals above */
                if(!subhalo_rank_fits_i32(rank))
                    endrun(1, "Subhalo in place: BindRank of candidate %ld beyond the int32 range.\n", Cd[cc].TrackId);
                int64_t * a = ans + 2 * o[tr[k].rank]++;
                a[0] = tr[k].index;
                a[1] = rank;
            }
            int64_t * in = NULL;
            const int64_t nin = subhalo_alltoallv(ans, sendcounts, 2 * sizeof(int64_t), (void **) &in, 0, NULL, C->Comm);
            for(k = 0; k < nin; k++)
                P[in[2 * k]].BindRank = (int32_t) in[2 * k + 1];
            myfree(in);
            myfree(ans);
            ta_free(o);
            ta_free(sendcounts);
            myfree(off);
        }
    }
    myfree(tr);
    CollStats.TagSortSeconds = MPI_Wtime() - t0;
    /* ---- the serial ranks and counts ---- */
    serial_repair(C, chg, nchg);
    /* ---- the exclusive and tail counts of the in-place records ---- */
    for(c = 0; c < ncand; c++) {
        if(slotof[c] < 0 || C->cs[c].state != SUBHALO_ST_ALIVE)
            continue;
        struct SubTrack * rec = &ctx->Res[slotof[c]].rec;
        int64_t nex = 0;
        for(t = 0; t < 6; t++) {
            rec->NboundExclusiveType[t] = (int32_t) g[12 * c + t];
            rec->NtailType[t] = (int32_t) g[12 * c + 6 + t];
            nex += g[12 * c + t];
        }
        rec->NboundExclusive = (int32_t) nex;
    }
    myfree(chg);
    myfree(g);
    myfree(cnt);
    int64_t loc[3] = {nwritten, nrel, nj2}, tot3[3];
    MPI_Allreduce(loc, tot3, 3, MPI_INT64, MPI_SUM, C->Comm);
    ctx->Stats.NFallbackField += nwritten;
    ctx->Stats.NFallbackReleased += nrel;
    CollStats.NJ2 = tot3[2];
    CollStats.TagSeconds = MPI_Wtime() - t0;
}

/* ------------------------------------------------------------------ setup */

/* A serial sub-tree result as gathered on every rank. */
struct SerialInfo {
    int64_t TrackId;
    int64_t SinkTo;
    int32_t state, Nbound, NboundExclusive, Keep;
    int32_t has_core, rank;  /* rank: the rank holding the result (the repair's track owner) */
    float Mbound, Vmax;
    MyIDType MostBoundID;
    double CorePos[3], CoreVel[3], SigmaX, SigmaV;
};

/* The per-key, per-candidate and per-member vectors every rank holds, as charged by the plan's
 * budget (subhalo_plan_default_cfg) and the admission (subhalo_coll_estimate). */
_Static_assert(sizeof(struct SubhaloInplaceKey) + 8 + 4 * 16 <= SUBHALO_INPLACE_KEY_BYTES,
               "SUBHALO_INPLACE_KEY_BYTES does not cover the per-key vectors");
_Static_assert(sizeof(struct SubhaloInplaceCand) + sizeof(struct CandState)
               + 8 * (NHS + 2 + CORE_ND + 1 + 2 * SUBHALO_MAXTRACERS + 2 * SUBHALO_NSO + 8 + 9 + 8 + 2 * NSUM + 12 + 4 + 4 + 3 * 12 + 2 + 1)
               + 4 * 10 <= SUBHALO_INPLACE_CAND_BYTES, "SUBHALO_INPLACE_CAND_BYTES does not cover the per-candidate vectors");
_Static_assert(sizeof(struct SubhaloMember) + sizeof(struct SerialInfo) + 32 <= SUBHALO_INPLACE_MEMBER_BYTES,
               "SUBHALO_INPLACE_MEMBER_BYTES does not cover the member table");

const char * const SubhaloCollStageName[SUBHALO_COLL_NSTAGE] = {
    "setup (serial results, previous records)", "tree build and walk", "core selection", "radial sort",
    "properties (profile, deferred recipes, shapes)", "relaxed-source sort", "final tags", "serial-tag repair"};

/* Fixed slack per stage for the allocator's block headers and alignment and the small NTask- and
 * leaf-sized vectors of the sorts. */
#define COLL_STAGE_SLACK (256. * 1024.)

void
subhalo_coll_estimate(struct SubhaloCollAdmit * a, int64_t NumPart, int64_t MaxPart, int64_t NUniv, int64_t NSubPart,
                      int64_t NOwned, int64_t OwnedRecv, int64_t NCands, int64_t NMembers, int64_t NSubResults,
                      int64_t NTopNodes, int64_t NTopLeaves, int64_t NLocalLeaves, double TreeAllocFactor, int nthreads)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    const int CoreSize = sp->SubhaloCoreSize > 0 ? sp->SubhaloCoreSize : 1;
    const double K = CoreSize > sp->SubhaloMinNumTracerPart ? CoreSize : sp->SubhaloMinNumTracerPart;
    const double N = (double) NumPart + 1, U = (double) NUniv + 1, S = (double) NSubPart + 1;
    const double nc = (double) NCands + 1;
    int s, NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    /* through the whole call: 25 bytes per particle, per candidate its state and vectors, per owned
     * candidate its previous record and slot */
    a->persist = 25. * N + nc * SUBHALO_INPLACE_CAND_BYTES + (double) (NOwned + 1) * (sizeof(struct SubTrack) + 8.);
    /* setup: the serial results gathered on every rank (local copy and the whole list), the
     * fetch of the owned previous records, the hierarchy arrays of the member table */
    {
        /* the fetch's batch buffers hold this rank's requests and the replies it serves: at most
         * its owned candidates and every candidate's record, each way */
        double fetch = 2. * (double) (NOwned + NCands + 2) * (sizeof(struct SubTrack) + 8.) + 4096. * 4;
        if(fetch > subhalo_table_batch_peak_bytes(NTask))
            fetch = subhalo_table_batch_peak_bytes(NTask);
        a->stage[SUBHALO_COLL_SETUP] = 2. * (double) (NSubResults + 1) * sizeof(struct SerialInfo)
            + (double) (NOwned + 1) * (sizeof(struct SubTrack) + 16.) + fetch
            + (double) (NMembers + 1) * (2 * sizeof(float) + sizeof(MyIDType) + 2 * sizeof(int64_t) + 1);
    }
    /* the tree: TreeAllocFactor x NumPart nodes (NumPart raised as in coll_walk) plus the top nodes,
     * no Father array; the query and tree lists, the top-leaf labels, the walk's export reserve */
    {
        double np = (double) NumPart;
        double need = 2048. + 512. * nthreads + 4. * (double) NLocalLeaves * nthreads;
        if(need > (double) MaxPart)
            need = (double) MaxPart;
        if(np < need)
            np = need;
        const double nodes = TreeAllocFactor * np + (double) NTopNodes + 1;
        a->stage[SUBHALO_COLL_TREE] = nodes * sizeof(struct NODE) + 8. * U + 16. * (double) (NTopLeaves + 1) * 8.
            + (double) subhalo_collective_walk_bytes();
    }
    /* the cores: the bounded heaps (at most K tracers and K non-tracers per candidate), the
     * owner's receive buffer */
    {
        double heap = 2. * K * nc;
        if(heap > U)
            heap = U;
        a->stage[SUBHALO_COLL_CORE] = sizeof(struct CoreRec) * (heap + (double) OwnedRecv + 1);
    }
    /* the radial ordered set: the records and mpsort's record-sized exchange copy */
    a->stage[SUBHALO_COLL_RADIAL] = 2. * sizeof(struct RadRec) * U;
    /* the properties (round_props): the sorted records with positions and cumulative masses, every
     * candidate's transient scratch and the allocator's alignment blocks */
    a->stage[SUBHALO_COLL_PROPS] = COLL_PROPS_REC_BYTES * U + nc * SUBHALO_INPLACE_CAND_PROPS_BYTES
        + COLL_PROPS_NALLOC * COLL_ALLOC_OVERHEAD;
    /* the relaxed-source positions: records and mpsort's copy */
    a->stage[SUBHALO_COLL_SOURCE] = 2. * sizeof(struct UPosRec) * U;
    /* the final tags: records and mpsort's copy, then records, answers and their receive buffer;
     * the change list (at most one per particle) lives below them */
    {
        const double sort = 2. * sizeof(struct TagRec) * U, ans = sizeof(struct TagRec) * U + 2. * 16. * U;
        a->stage[SUBHALO_COLL_TAGS] = (sort > ans ? sort : ans) + sizeof(struct SerChg) * U;
    }
    /* the serial repair: changes sent (one per particle at most) and received (one per sub-tree
     * particle processed here), the affected TrackIds, the queries and replies both ways */
    a->stage[SUBHALO_COLL_REPAIR] = sizeof(struct SerChg) * (U + S) + 8. * (double) (NMembers + 1) * 2
        + sizeof(struct SerQuery) * (U + S) + sizeof(struct SerReply) * (U + 2. * S) + sizeof(struct RepTrack) * (double) (NMembers + 1);
    double mx = 0;
    for(s = 0; s < SUBHALO_COLL_NSTAGE; s++) {
        a->stage[s] += COLL_STAGE_SLACK;
        if(a->stage[s] > mx)
            mx = a->stage[s];
    }
    a->need = a->persist + mx;
    a->NumPart = NumPart;
    a->NUniv = NUniv;
    a->NSubPart = NSubPart;
}

/* The serial sub-tree results into the member table on every rank: state, counts, Keep, record
 * values, sinking core and in-key sink. Collective. */
static void
gather_serial(struct Coll * C)
{
    struct SubhaloCallCtx * ctx = C->ctx;
    int64_t i, n = 0;
    for(i = 0; i < ctx->IpResStart; i++)
        if(ctx->Res[i].subkey >= 0 && ctx->Res[i].valid)
            n++;
    int64_t tot = 0;
    MPI_Allreduce(&n, &tot, 1, MPI_INT64, MPI_SUM, C->Comm);
    subhalo_check_mpi_count(tot * (int64_t) sizeof(struct SerialInfo), "the serial sub-tree results (bytes)", C->Comm);
    struct SerialInfo * all = (struct SerialInfo *) mymalloc("SubCollSerAll", sizeof(struct SerialInfo) * (tot + 1));
    struct SerialInfo * loc = (struct SerialInfo *) mymalloc("SubCollSerLoc", sizeof(struct SerialInfo) * (n + 1));
    n = 0;
    for(i = 0; i < ctx->IpResStart; i++) {
        const struct SubhaloResult * rs = &ctx->Res[i];
        if(rs->subkey < 0 || !rs->valid)
            continue;
        struct SerialInfo * e = &loc[n++];
        memset(e, 0, sizeof(*e));
        e->TrackId = rs->rec.TrackId;
        e->SinkTo = rs->SinkTo;
        e->state = rs->state;
        e->Nbound = rs->rec.Nbound;
        e->NboundExclusive = rs->rec.NboundExclusive;
        e->Keep = rs->Keep;
        e->has_core = rs->has_core;
        e->rank = C->ThisTask;
        e->Mbound = rs->rec.Mbound;
        e->Vmax = rs->rec.Vmax;
        e->MostBoundID = rs->rec.MostBoundID;
        int d;
        for(d = 0; d < 3; d++) {
            e->CorePos[d] = rs->CorePos[d];
            e->CoreVel[d] = rs->CoreVel[d];
        }
        e->SigmaX = rs->SigmaX;
        e->SigmaV = rs->SigmaV;
    }
    int * counts = ta_malloc("SubCollSerCnt", int, C->NTask);
    int * displ = ta_malloc("SubCollSerDispl", int, C->NTask);
    int nb = (int) (n * sizeof(struct SerialInfo));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, C->Comm);
    int64_t s = 0;
    int r;
    for(r = 0; r < C->NTask; r++) {
        displ[r] = (int) s;
        s += counts[r];
    }
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, counts, displ, MPI_BYTE, C->Comm);
    ta_free(displ);
    ta_free(counts);
    myfree(loc);
    for(i = 0; i < C->plan->NMembers; i++)
        C->plan->Members[i].resrank = -1;
    for(i = 0; i < tot; i++) {
        const int64_t j = member_lookup_c(C, all[i].TrackId);
        if(j < 0)
            endrun(5, "Subhalo in place: serial result %ld is not a member of an oversized key.\n", all[i].TrackId);
        struct SubhaloMember * m = &C->plan->Members[j];
        m->state = all[i].state;
        m->SinkTrackId = all[i].state == SUBHALO_ST_SUNK ? all[i].SinkTo : SUBHALO_NOTRACK;
        m->Nbound = all[i].state == SUBHALO_ST_ALIVE ? all[i].Nbound : 0;
        m->NboundExclusive = all[i].NboundExclusive;
        m->Keep = all[i].Keep;
        m->has_core = all[i].has_core;
        m->resrank = all[i].rank;
        m->Mbound = all[i].Mbound;
        m->Vmax = all[i].Vmax;
        m->MostBoundID = all[i].MostBoundID;
        int d;
        for(d = 0; d < 3; d++) {
            m->CorePos[d] = all[i].CorePos[d];
            m->CoreVel[d] = all[i].CoreVel[d];
        }
        m->SigmaX = all[i].SigmaX;
        m->SigmaV = all[i].SigmaV;
    }
    myfree(all);
}

/* Every particle of an oversized key to one candidate's source (design 7.8): own tags, serial sub-tree
 * remainders to the candidate above the root (serially bound ones excluded), other group particles to
 * the central; field particles tagged by the central (decision 52) or an orphan are released. */
static void
init_sources(struct Coll * C)
{
    const struct SubhaloPlan * plan = C->plan;
    int64_t i;
    for(i = 0; i < C->NumPart; i++) {
        C->Cand[i] = -1;
        C->Claim[i] = -1;
        C->Flag[i] = 0;
        C->En[i] = 0;
        C->ClaimE[i] = 0;
        const struct particle_data * p = &P[i];
        if(!part_live(p))
            continue;
        int64_t u = -1, ku = -1;
        const struct SubhaloMember * m = p->TrackId >= 0 ? member_of(C, p->TrackId) : NULL;
        if(p->GrNr >= 1 && (ku = subhalo_plan_inplace_index(plan, p->GrNr)) >= 0)
            u = p->GrNr;
        else if(p->GrNr < 0 && m) {
            u = m->g;
            /* a FOF key only, so that its central's field particles are released below (decision 52);
             * a pseudo-key (u > TotNgroups) keeps them (ku stays -1) */
            if(u <= plan->TotNgroups)
                ku = subhalo_plan_inplace_index(plan, u);
        }
        if(u < 0)
            continue;
        if(m && m->g != u)
            m = NULL;
        C->Flag[i] = CF_UNIV;
        const int central = ku >= 0 ? plan->Inplace[ku].central : -1;
        if(m && m->cand >= 0 && (m->kind != SUBHALO_MEM_ORPHAN || p->GrNr == u) && (m->cand != central || p->GrNr == u))
            C->Cand[i] = m->cand;
        else if(m && m->cand < 0 && m->SubKey >= 0) {
            const struct SubhaloSubKey * sk = &plan->SubKeys[m->SubKey - plan->SubKeyBase];
            const struct SubhaloMember * R = &plan->Members[sk->root];
            if(R->Parent < 0 || plan->Members[R->Parent].cand < 0)
                endrun(5, "Subhalo in place: the root %ld of sub-tree key %ld has no in-place parent.\n", R->TrackId, m->SubKey);
            if(p->BoundTrackId >= 0)
                C->Flag[i] |= CF_SBOUND | CF_SCLAIM;
            else {
                C->Cand[i] = plan->Members[R->Parent].cand;
                if(m->state == SUBHALO_ST_ALIVE && (!m->root || p->BindRank < m->NboundExclusive + m->Keep))
                    C->Flag[i] |= CF_SCLAIM;
            }
        }
        else if(p->GrNr == u)
            C->Cand[i] = central;       /* a group particle: ku was found above */
        else {
            /* a field particle tagged by an orphan member (a stale tag) or by the central */
            C->Flag[i] = CF_RELEASE;
        }
    }
}

/* ------------------------------------------------------------------ the rounds */

/* The sinking core and the most-bound tracer (loop order) of the alive candidates of the round. */
static void
set_loop_cores(struct Coll * C)
{
    struct CoreOut lc;
    round_cores(C, &lc);
    int64_t c;
    int d;
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        if(s->state != SUBHALO_ST_ALIVE)
            continue;
        const double * o = lc.d + CORE_ND * (c - C->c0);
        s->has_core = o[14] > 0;
        for(d = 0; d < 3; d++) {
            s->CorePos[d] = o[d];
            s->CoreVel[d] = o[3 + d];
            s->mbPos[d] = o[8 + d];
            s->mbVel[d] = o[11 + d];
        }
        s->SigmaX = o[6];
        s->SigmaV = o[7];
        s->mbID = lc.mbID[c - C->c0];
        s->has_mb = s->has_core;
    }
    free_cores(&lc);
    dead_mb(C);
}

static void
run_round(struct Coll * C, int r, const int64_t * slotof, const struct SubTrack * oldrec)
{
    const struct SubhaloPlan * plan = C->plan;
    const struct SubhaloUnbindParams * up = C->up;
    C->c0 = plan->RoundStart[r];
    C->c1 = plan->RoundStart[r + 1];
    const double t0 = MPI_Wtime();
    const int w0 = C->nwalks;
    int64_t c, i;
    int d;
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        const struct SubhaloInplaceCand * cd = &plan->Cands[c];
        s->state = SUBHALO_ST_ALIVE;
        s->iterating = 1;
        s->absorbed = 0;
        s->flags = 0;
        for(d = 0; d < 3; d++) {
            s->Ref0[d] = s->RefPos[d] = cd->RefPos[d];
            s->RefVel[d] = cd->RefVel[d];
        }
    }
    /* the unbinding loop */
    unbind_loop(C);
    /* sinking of the descendants on the loop-order cores, then the absorbing candidates unbound
     * again from their whole source (as process_serial_key) */
    set_loop_cores(C);
    if(round_sinking(C) > 0) {
        for(c = C->c0; c < C->c1; c++) {
            struct CandState * s = &C->cs[c];
            s->iterating = s->absorbed && s->state == SUBHALO_ST_ALIVE;
        }
        unbind_loop(C);
        set_loop_cores(C);
    }
    /* the final pass on the converged sets with the final frames */
    int64_t npos = 0;
    for(c = C->c0; c < C->c1; c++)
        C->cs[c].iterating = C->cs[c].state == SUBHALO_ST_ALIVE;
    coll_walk(C, C->En);
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || !C->cs[cc].iterating || !(C->Flag[i] & CF_BOUND) || !part_live(&P[i]))
            continue;
        C->En[i] = coll_energy(C, i, C->En[i], &C->cs[cc]);
        if(C->En[i] >= 0)
            npos++;
    }
    C->ctx->Stats.NFinalPositive += npos;
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        s->iterating = 0;
        if(s->state != SUBHALO_ST_ALIVE)
            continue;
        C->st.NFinalBound += s->Nbound;
        int64_t keep = (int64_t) (s->Nbound * C->sp->SubhaloSourceRelaxFactor);
        if(keep < s->Nbound) keep = s->Nbound;
        if(keep > s->N) keep = s->N;
        s->keep = keep;
    }
    /* the final cores, most-bound and reserved tracers, then the properties */
    struct CoreOut fin;
    round_cores(C, &fin);
    struct RoundProps rp;
    round_props(C, &fin, &rp);
    /* the full bound counts against the int32 columns before anything is narrowed; the sums are
     * reduced, so every rank takes the same decision */
    for(c = C->c0; c < C->c1; c++) {
        if(C->cs[c].state != SUBHALO_ST_ALIVE)
            continue;
        const double * h = rp.hs + NHS * (c - C->c0);
        int ok = subhalo_count_fits_i32(h[HS_N]) && subhalo_count_fits_i32(h[HS_NTR]), t;
        for(t = 0; t < 6; t++)
            ok = ok && subhalo_count_fits_i32(h[HS_NT + t]);
        if(!ok)
            endrun(1, "Subhalo in place: candidate %ld has %.0f bound particles, beyond the catalogue's int32 counts.\n",
                   plan->Cands[c].TrackId, h[HS_N]);
    }
    for(c = C->c0; c < C->c1; c++) {
        struct CandState * s = &C->cs[c];
        if(s->state != SUBHALO_ST_ALIVE)
            continue;
        const int64_t q = c - C->c0;
        const double * o = fin.d + CORE_ND * q;
        s->mbID = fin.mbID[q];
        for(d = 0; d < 3; d++) {
            s->mbPos[d] = o[8 + d];
            s->mbVel[d] = o[11 + d];
        }
        s->has_mb = 1;
        s->Mbound_rec = (float) rp.hs[NHS * q + HS_M];
        s->Vmax_rec = rec_vmax(rp.prof[8 * q], up);
        s->MostBoundID_rec = fin.mbID[q];
        if(slotof[c] >= 0) {
            const struct SubhaloMember * m = &plan->Members[plan->Cands[c].member];
            const struct SubTrack * old = m->kind == SUBHALO_MEM_NEWBORN ? NULL : &oldrec[slotof[c] - C->ctx->IpResStart];
            fill_alive_record(C, c, &fin, &rp, &C->ctx->Res[slotof[c]].rec, old);
        }
    }
    free_round_props(&rp);
    /* the relaxed sources (claims) and the remainders to the parents */
    round_within(C);
    round_commit(C, &fin);
    free_cores(&fin);
    int64_t nalive = 0, ndied = 0;
    for(c = C->c0; c < C->c1; c++) {
        if(C->cs[c].state == SUBHALO_ST_ALIVE) nalive++;
        else if(C->cs[c].state == SUBHALO_ST_DIED || C->cs[c].state == SUBHALO_ST_FAILED) ndied++;
    }
    CollStats.NAlive += nalive;
    CollStats.NDied += ndied;
    const double dt = MPI_Wtime() - t0;
    if(r < 16) {
        CollStats.NCandsRound[r] = C->c1 - C->c0;
        CollStats.Iterations[r] = C->nwalks - w0;
        CollStats.Seconds[r] = dt;
    }
    message(0, "Subhalo in place: round %d of %d: %ld candidates (%ld alive, %ld died), %d tree builds and walks, %g s; free %g MB.\n",
            r, plan->NRounds, C->c1 - C->c0, nalive, ndied, C->nwalks - w0, dt, mymalloc_freebytes() / (1024. * 1024.));
}

void
subhalo_collective_run(struct SubhaloPlan * plan, struct SubhaloCallCtx * ctx, DomainDecomp * ddecomp,
                       struct SubTrackTable * tab, MPI_Comm Comm)
{
    memset(&CollStats, 0, sizeof(CollStats));
    if(plan->NCands == 0)
        return;
    /* the measured high water of the call against the plan's admission */
    const double Free0 = (double) mymalloc_freebytes();
    allocator_reset_min_free(A_MAIN);
    struct Coll C;
    memset(&C, 0, sizeof(C));
    C.plan = plan;
    C.ctx = ctx;
    C.ddecomp = ddecomp;
    C.Comm = Comm;
    MPI_Comm_size(Comm, &C.NTask);
    MPI_Comm_rank(Comm, &C.ThisTask);
    C.up = ctx->up;
    C.sp = subhalo_get_params();
    C.NumPart = PartManager->NumPart;
    C.st.WorstConvRatio = 1;
    C.st.WorstTrackId = -1;
    const int64_t ncand = plan->NCands;
    int64_t c, j;
    const double t0 = MPI_Wtime();
    CollStats.NRounds = plan->NRounds;
    CollStats.NCands = ncand;

    /* 1. the serial sub-tree results */
    gather_serial(&C);

    /* the owner slots and the previous records of the owned candidates (top) */
    int64_t * slotof = (int64_t *) mymalloc2("SubCollSlotOf", sizeof(int64_t) * (ncand + 1));
    j = 0;
    for(c = 0; c < ncand; c++)
        slotof[c] = plan->Cands[c].owner == C.ThisTask ? ctx->IpResStart + j++ : -1;
    if(j != ctx->NIpRes)
        endrun(5, "Subhalo in place: %ld owned candidates for %ld result slots.\n", j, ctx->NIpRes);
    struct SubTrack * oldrec = (struct SubTrack *) mymalloc2("SubCollOldRec", sizeof(struct SubTrack) * (j + 1));
    {
        int64_t * ids = (int64_t *) mymalloc("SubCollOldIds", sizeof(int64_t) * (j + 1));
        int64_t * pos = (int64_t *) mymalloc("SubCollOldPos", sizeof(int64_t) * (j + 1));
        int64_t n = 0;
        for(c = 0; c < ncand; c++) {
            if(slotof[c] < 0)
                continue;
            subhalo_track_clear(&oldrec[slotof[c] - ctx->IpResStart], plan->Cands[c].TrackId);
            if(!SUBHALO_IS_PROVISIONAL(plan->Cands[c].TrackId)) {
                ids[n] = plan->Cands[c].TrackId;
                pos[n] = slotof[c] - ctx->IpResStart;
                n++;
            }
        }
        struct SubTrack * got = (struct SubTrack *) mymalloc("SubCollOldGot", sizeof(struct SubTrack) * (n + 1));
        subhalo_table_fetch(tab, ids, n, got, Comm);
        for(c = 0; c < n; c++)
            oldrec[pos[c]] = got[c];
        myfree(got);
        myfree(pos);
        myfree(ids);
        for(c = 0; c < ctx->NIpRes; c++) {
            memset(&ctx->Res[ctx->IpResStart + c], 0, sizeof(struct SubhaloResult));
            ctx->Res[ctx->IpResStart + c].ipk = -1;
            ctx->Res[ctx->IpResStart + c].subkey = -1;
        }
    }

    /* the per-call scratch (top) */
    C.cs = (struct CandState *) mymalloc2("SubCollCands", sizeof(struct CandState) * (ncand + 1));
    memset(C.cs, 0, sizeof(struct CandState) * (ncand + 1));
    for(c = 0; c < ncand; c++) {
        C.cs[c].state = SUBHALO_ST_NONE;
        C.cs[c].SinkTrackId = SUBHALO_NOTRACK;
    }
    C.Cand = (int32_t *) mymalloc2("SubCollCand", sizeof(int32_t) * (C.NumPart + 1));
    C.Claim = (int32_t *) mymalloc2("SubCollClaim", sizeof(int32_t) * (C.NumPart + 1));
    C.Flag = (uint8_t *) mymalloc2("SubCollFlag", C.NumPart + 1);
    C.En = (double *) mymalloc2("SubCollEn", sizeof(double) * (C.NumPart + 1));
    C.ClaimE = (double *) mymalloc2("SubCollClaimE", sizeof(double) * (C.NumPart + 1));

    /* 2. the sources */
    init_sources(&C);
    walltime_measure("/Subhalo/Rounds/Setup");

    /* 3. the rounds, deepest first */
    int r;
    for(r = 0; r < plan->NRounds; r++)
        run_round(&C, r, slotof, oldrec);
    walltime_measure("/Subhalo/Rounds/Rounds");

    /* 4. the hierarchy, the records */
    double * maxmerge = (double *) mymalloc("SubCollMaxMerge", sizeof(double) * (plan->NInplace + 1));
    coll_hierarchy(&C, maxmerge);
    coll_records(&C, maxmerge, slotof, oldrec);
    myfree(maxmerge);

    /* 5. the final tags */
    if(PretagHook)
        PretagHook(ctx, plan);
    coll_final_tags(&C, slotof);
    walltime_measure("/Subhalo/Rounds/Tags");

    myfree(C.ClaimE);
    myfree(C.En);
    myfree(C.Flag);
    myfree(C.Claim);
    myfree(C.Cand);
    myfree(C.cs);
    myfree(oldrec);
    myfree(slotof);
    {
        const double meas = Free0 - (double) allocator_get_min_free(A_MAIN);
        const double est = plan->Admit.checked ? plan->Admit.need : 0;
        struct { double v; int r; } in[3], out[3];
        in[0].v = meas;
        in[1].v = est;
        in[2].v = est > 0 ? meas / est : 0;
        in[0].r = in[1].r = in[2].r = C.ThisTask;
        MPI_Allreduce(in, out, 3, MPI_DOUBLE_INT, MPI_MAXLOC, Comm);
        CollStats.MeasPeakBytes = out[0].v;
        CollStats.MeasPeakRank = out[0].r;
        CollStats.EstPeakBytes = out[1].v;
        CollStats.EstPeakRank = out[1].r;
        CollStats.MaxMeasOverEst = out[2].v;
        double av[2] = {plan->Admit.checked ? -plan->Admit.avail : 0, -Free0}, avg[2];
        MPI_Allreduce(av, avg, 2, MPI_DOUBLE, MPI_MAX, Comm);
        CollStats.EstAvailBytes = -avg[0];
        CollStats.FreeAtStartBytes = -avg[1];
        double el = CollStats.WalkElasticBytes;
        MPI_Allreduce(MPI_IN_PLACE, &el, 1, MPI_DOUBLE, MPI_MAX, Comm);
        CollStats.WalkElasticBytes = el;
        message(0, "Subhalo in place: memory high water %g MB (rank %d; the walks charged their reserve %g MB, they took up to %g MB of "
                   "the free memory for their export tables), admitted estimate %g MB (rank %d), largest measured/estimated %.3f; "
                   "least free at the start %g MB (the plan expected at least %g MB).\n", CollStats.MeasPeakBytes / 1048576.,
                CollStats.MeasPeakRank, subhalo_collective_walk_bytes() / 1048576., el / 1048576., CollStats.EstPeakBytes / 1048576.,
                CollStats.EstPeakRank, CollStats.MaxMeasOverEst, CollStats.FreeAtStartBytes / 1048576., CollStats.EstAvailBytes / 1048576.);
    }

    /* the candidate-level diagnostics are identical on every rank: counted once */
    if(C.ThisTask == 0) {
        ctx->Stats.NCandidates += C.st.NCandidates;
        ctx->Stats.NItercap += C.st.NItercap;
        ctx->Stats.NFinalBound += C.st.NFinalBound;
        if(C.st.NItercap > 0 && (ctx->Stats.WorstTrackId == -1 || C.st.WorstConvRatio < ctx->Stats.WorstConvRatio)) {
            ctx->Stats.WorstConvRatio = C.st.WorstConvRatio;
            ctx->Stats.WorstTrackId = C.st.WorstTrackId;
        }
    }
    int64_t iters = 0;
    for(r = 0; r < plan->NRounds && r < 16; r++)
        iters += CollStats.Iterations[r];
    message(0, "Subhalo in place: %ld candidates in %d round(s), %ld tree builds and walks, %ld alive, %ld died, %ld descendants sunk; "
               "%ld serial own-tag particles taken by in-place candidates, %ld serial tails given to their binder, %ld claims given to the "
               "binding owner (J2); final tags %g s (serial repair of %ld tracks, %ld rank queries: %g s); %g s.\n",
            ncand, plan->NRounds, iters, CollStats.NAlive, CollStats.NDied, CollStats.NSunk, CollStats.NStolen, CollStats.NAdded,
            CollStats.NJ2, CollStats.TagSeconds, CollStats.NRepairTracks, CollStats.NRepairQueries, CollStats.RepairSeconds, MPI_Wtime() - t0);
}

/* ------------------------------------------------------------------ test hook: the round's properties */

void
subhalo_collective_test_props(const int32_t * label, int64_t nlabels, const double * core, const struct SubhaloUnbindParams * up,
                              const double rho[SUBHALO_NSO], double * dfr, double * mvir)
{
    struct Coll C;
    struct SubhaloCallCtx ctx;
    struct CoreOut co;
    struct RoundProps rp;
    int64_t i, c;
    int d;
    memset(&C, 0, sizeof(C));
    memset(&ctx, 0, sizeof(ctx));
    memset(&co, 0, sizeof(co));
    for(d = 0; d < SUBHALO_NSO; d++)
        ctx.rho[d] = rho[d];
    C.ctx = &ctx;
    C.up = up;
    C.Comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &C.NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &C.ThisTask);
    C.NumPart = PartManager->NumPart;
    C.c0 = 0;
    C.c1 = nlabels;
    C.cs = (struct CandState *) mymalloc("SubTestPropsCs", sizeof(struct CandState) * (nlabels + 1));
    memset(C.cs, 0, sizeof(struct CandState) * (nlabels + 1));
    co.d = (double *) mymalloc("SubTestPropsCore", sizeof(double) * CORE_ND * (nlabels + 1));
    memset(co.d, 0, sizeof(double) * CORE_ND * (nlabels + 1));
    for(c = 0; c < nlabels; c++) {
        C.cs[c].state = SUBHALO_ST_ALIVE;
        for(d = 0; d < 3; d++) {
            C.cs[c].RefPos[d] = co.d[CORE_ND * c + d] = core[6 * c + d];
            C.cs[c].RefVel[d] = co.d[CORE_ND * c + 3 + d] = core[6 * c + 3 + d];
        }
    }
    C.Cand = (int32_t *) mymalloc("SubTestPropsCand", sizeof(int32_t) * (C.NumPart + 1));
    C.Flag = (uint8_t *) mymalloc("SubTestPropsFlag", C.NumPart + 1);
    C.En = (double *) mymalloc("SubTestPropsEn", sizeof(double) * (C.NumPart + 1));
    for(i = 0; i < C.NumPart; i++) {
        C.Cand[i] = label[i] >= 0 && label[i] < nlabels ? label[i] : -1;
        C.Flag[i] = C.Cand[i] >= 0 ? CF_BOUND : 0;
        C.En[i] = 0;
    }
    round_props(&C, &co, &rp);
    for(c = 0; c < 8 * nlabels; c++)
        dfr[c] = rp.dfr[c];
    if(mvir)
        for(c = 0; c < nlabels; c++)
            mvir[c] = rp.kc[SUBHALO_NSO * c] >= 0 ? rp.Mso[SUBHALO_NSO * c] : -1;
    free_round_props(&rp);
    myfree(C.En);
    myfree(C.Flag);
    myfree(C.Cand);
    myfree(co.d);
    myfree(C.cs);
}
