/* The labelled walks of the in-place path over the global tree: the potential-only walk of the
 * unbinding (contract in subhalo_collective.h) and the k-nearest search of the admission vote,
 * called from the plan (subhalo_gather.c). */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
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
#include "subhalo_collective_internal.h"

size_t
subhalo_collective_walk_bytes(void)
{
    /* query: TreeWalkQueryBase + label + index; result: TreeWalkResultBase + one double */
    const size_t q = sizeof(TreeWalkQueryBase) + 8, r = sizeof(TreeWalkResultBase) + 8;
    return subhalo_walk_reserve_bytes(q, r, subhalo_import_buffer_boost());
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
void
coll_walk(struct Coll * C, double * Sum)
{
    const int64_t NumPart = PartManager->NumPart;
    int64_t i, nq = 0, nt = 0;
    const int PotentialTypes = C->up->PotentialTypes;
    for(i = 0; i < NumPart; i++) {
        const int32_t c = C->Cand[i];
        if(c < C->c0 || c >= C->c1 || !(C->Flag[i] & CF_BOUND) || !C->cs[c].iterating || !subhalo_part_live(&P[i]))
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
        if(c < C->c0 || c >= C->c1 || !(C->Flag[i] & CF_BOUND) || !C->cs[c].iterating || !subhalo_part_live(&P[i]))
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
        int64_t need = coll_tree_min_part(nleaf, nthr);
        if(need > PartManager->MaxPart)
            need = PartManager->MaxPart;
        if(PartManager->NumPart < need)
            PartManager->NumPart = need;
    }
    /* the whole build booked to /Subhalo/Rounds/Tree, not forcetree.c's own clocks, so that the
     * /Subhalo/Rounds children cover the stage */
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
     * apart, and the high water charges the walk with the reserve the memory check guarantees */
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

/* ------------------------------------------------------------------ the k-nearest walk */

typedef struct {
    TreeWalkQueryBase base;
    double H;                /* the search radius of this pass */
    int32_t Label;           /* the query's label: only particles of the same label are neighbours */
    int32_t Index;           /* the query particle on its home rank */
} TreeWalkQueryKnn;

typedef struct {
    TreeWalkResultBase base;
    int32_t n, pad;
    struct SubhaloKnnHit h[];  /* k entries, sorted by (d2, ID) */
} TreeWalkResultKnn;

struct KnnWalkPriv {
    const int32_t * label;
    const int32_t * slot;    /* query slot of a local particle */
    const double * H;
    struct SubhaloKnnHit * hits;
    int32_t * nhit;
    int k;
    double slack;
};
#define KNN_PRIV(tw) ((struct KnnWalkPriv *) ((tw)->priv))

static inline int
knn_hit_before(const struct SubhaloKnnHit * a, const struct SubhaloKnnHit * b)
{
    return a->d2 < b->d2 || (a->d2 == b->d2 && a->ID < b->ID);
}

/* Inserts e into the sorted list h[0..*n) of at most k entries; a particle met twice counts once. */
static inline void
knn_hit_insert(struct SubhaloKnnHit * h, int32_t * n, int k, const struct SubhaloKnnHit * e)
{
    int j;
    if(*n == k && !knn_hit_before(e, &h[k - 1]))
        return;
    for(j = 0; j < *n; j++)
        if(h[j].ID == e->ID)
            return;
    j = *n < k ? (*n)++ : k - 1;
    while(j > 0 && knn_hit_before(e, &h[j - 1])) {
        h[j] = h[j - 1];
        j--;
    }
    h[j] = *e;
}

/* Lower bound of the periodic distance from q to the cube of node nop, less a slack that covers
 * the rounding of the node geometry. */
static inline double
knn_cube_dist2(const struct NODE * nop, const double q[3], double BoxSize, double slack)
{
    double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double g = fabs(NEAREST(nop->center[d] - q[d], BoxSize)) - 0.5 * nop->len - slack;
        if(g > 0)
            r2 += g * g;
    }
    return r2;
}

static void
knn_fill(int i, TreeWalkQueryKnn * I, TreeWalk * tw)
{
    const struct KnnWalkPriv * pv = KNN_PRIV(tw);
    I->H = pv->H[pv->slot[i]];
    I->Label = pv->label[i];
    I->Index = i;
}

static void
knn_reduce(int i, TreeWalkResultKnn * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    struct KnnWalkPriv * pv = KNN_PRIV(tw);
    const int32_t s = pv->slot[i];
    int j;
    (void) mode;
    for(j = 0; j < O->n; j++)
        knn_hit_insert(pv->hits + (int64_t) s * pv->k, &pv->nhit[s], pv->k, &O->h[j]);
}

/* The neighbours of one query within its radius H on this rank (the structure of coll_visit):
 * nodes of other labels and nodes farther than H, or than the k-th found, are skipped. */
static int
knn_visit(TreeWalkQueryKnn * I, TreeWalkResultKnn * O, LocalTreeWalk * lv)
{
    const ForceTree * tree = lv->tw->tree;
    const struct KnnWalkPriv * pv = KNN_PRIV(lv->tw);
    const double BoxSize = tree->BoxSize;
    const int k = pv->k, label = I->Label;
    /* the query meets itself only on its home rank, in the primary walk */
    const int self = lv->mode == TREEWALK_PRIMARY ? I->Index : -1;
    const double * inpos = I->base.Pos;
    const double H2 = I->H * I->H;
    int64_t ninter = 0;
    int listindex;
    O->n = 0;
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
            double lim2 = H2;
            if(O->n == k && O->h[k - 1].d2 < lim2)
                lim2 = O->h[k - 1].d2;
            if(nl == -2 || (nl >= 0 && nl != label) || knn_cube_dist2(nop, inpos, BoxSize, pv->slack) > lim2) {
                no = nop->sibling;
                continue;
            }
            if(lv->mode == TREEWALK_TOPTREE) {
                if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                    if(-1 == treewalk_export_particle(lv, nop->s.suns[0]))
                        return -1;
                    no = nop->sibling;
                    continue;
                }
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
                        if(pp == self || pv->label[pp] != label)
                            continue;
                        struct SubhaloKnnHit e;
                        e.d2 = subhalo_knn_r2(P[pp].Pos, inpos, BoxSize);
                        e.ID = P[pp].ID;
                        e.BoundTrackId = P[pp].BoundTrackId;
                        knn_hit_insert(O->h, &O->n, k, &e);
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
    treewalk_add_counters(lv, ninter);
    return 1;
}

size_t
subhalo_collective_knn_walk_bytes(int k)
{
    const size_t q = sizeof(TreeWalkQueryKnn), r = sizeof(TreeWalkResultKnn) + sizeof(struct SubhaloKnnHit) * (size_t) k;
    return subhalo_walk_reserve_bytes(q, r, subhalo_import_buffer_boost());
}

static int64_t KnnExportRounds = 0;

int64_t
subhalo_collective_knn_export_rounds(void)
{
    return KnnExportRounds;
}

int
subhalo_collective_knn(DomainDecomp * ddecomp, const int32_t * label, const int * tl, int64_t nt,
                       const int * ql, int64_t nq, int k, struct SubhaloKnnHit * hits, int32_t * nhit)
{
    const int64_t NumPart = PartManager->NumPart;
    const double BoxSize = PartManager->BoxSize;
    int64_t i;
    if(k < 1 || k > SUBHALO_KNN_MAXK + 1)
        endrun(1, "Subhalo k-nearest walk: k = %d outside [1, %d].\n", k, SUBHALO_KNN_MAXK + 1);
    int32_t * slot = (int32_t *) mymalloc("SubKnnSlot", sizeof(int32_t) * (NumPart + 1));
    double * H = (double *) mymalloc("SubKnnH", sizeof(double) * (nq + 1));
    int * act = (int *) mymalloc("SubKnnActive", sizeof(int) * (nq + 1));
    ActiveParticles ap;
    memset(&ap, 0, sizeof(ap));
    ap.ActiveParticle = (int *) tl;
    ap.NumActiveParticle = nt;
    ap.MaxActiveParticle = nt;
    ap.Particles = PartManager->Base;
    ForceTree tree;
    memset(&tree, 0, sizeof(tree));
    /* force_tree_build sizes its node block by NumPart: raised for the build only on a nearly empty
     * rank, as in coll_walk */
    {
        const int nthr = omp_get_max_threads();
        int ThisTask;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        const int64_t nleaf = ddecomp->Tasks[ThisTask].EndLeaf - ddecomp->Tasks[ThisTask].StartLeaf;
        int64_t need = coll_tree_min_part(nleaf, nthr);
        if(need > PartManager->MaxPart)
            need = PartManager->MaxPart;
        if(PartManager->NumPart < need)
            PartManager->NumPart = need;
    }
    /* a build that runs out of nodes raises the global TreeAllocFactor and retries: restored, so
     * that the vote leaves the force tree's sizing as it found it */
    const double taf = force_tree_get_alloc_factor();
    force_tree_active_moments(&tree, ddecomp, &ap, 0, 1, NULL);
    init_forcetree_params(taf);
    PartManager->NumPart = NumPart;
    label_tree(&tree, ddecomp, label, 0);
    /* the first radius is the side of the query's tree leaf, doubled until the k-th neighbour lies
     * strictly inside it, or until it exceeds the box (which then holds every nearest image) */
    for(i = 0; i < nq; i++) {
        const int f = force_get_father(ql[i], &tree);
        H[i] = f >= tree.firstnode ? tree.Nodes[f].len : BoxSize / 64;
        if(!(H[i] > 0))
            H[i] = BoxSize / 64;
        slot[ql[i]] = (int32_t) i;
        act[i] = ql[i];
    }
    struct KnnWalkPriv pv;
    pv.label = label;
    pv.slot = slot;
    pv.H = H;
    pv.hits = hits;
    pv.nhit = nhit;
    pv.k = k;
    pv.slack = 1e-9 * BoxSize;
    TreeWalk tw[1];
    memset(tw, 0, sizeof(tw));
    tw->ev_label = "SUBHALO_KNN";
    tw->type = TREEWALK_ACTIVE;
    tw->visit = (TreeWalkVisitFunction) knn_visit;
    tw->NoNgblist = 1;
    tw->fill = (TreeWalkFillQueryFunction) knn_fill;
    tw->reduce = (TreeWalkReduceResultFunction) knn_reduce;
    tw->query_type_elsize = sizeof(TreeWalkQueryKnn);
    tw->result_type_elsize = sizeof(TreeWalkResultKnn) + sizeof(struct SubhaloKnnHit) * (size_t) k;
    tw->priv = &pv;
    tw->tree = &tree;
    int64_t na = nq, natot;
    int npass = 0;
    MPI_Allreduce(&na, &natot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    KnnExportRounds = 0;
    while(natot > 0) {
        for(i = 0; i < na; i++)
            nhit[slot[act[i]]] = 0;
        treewalk_run(tw, act, na);
        KnnExportRounds += tw->Nexportfull;
        npass++;
        int64_t nn = 0;
        for(i = 0; i < na; i++) {
            const int32_t s = slot[act[i]];
            const int32_t n = nhit[s];
            if(H[s] > BoxSize || (n == k && hits[(int64_t) s * k + k - 1].d2 < H[s] * H[s]))
                continue;
            H[s] *= 2;
            act[nn++] = act[i];
        }
        na = nn;
        MPI_Allreduce(&na, &natot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    }
    force_tree_free(&tree);
    myfree(act);
    myfree(H);
    myfree(slot);
    return npass;
}

