/* Field spherical overdensity of the subhalo finder's centrals (design 7.6, Phase 3 plan
 * section 4). The algorithm and its resolution are described in subhalo_so.h.
 *
 * Mechanics.
 *  - Anchored queries: treewalk_init_query copies P[i].Pos (and, with DEBUG, P[i].ID) of the
 *    query's particle and then calls fill, so every query is issued from a local particle. The
 *    s-th query of a walk is anchored to the s-th local particle that is neither garbage nor
 *    swallowed (the anchors are therefore distinct and ascending); fill overwrites Pos with the
 *    central's core in the translated frame and sets the radii; fill and reduce map the anchor
 *    index back to the query slot by a binary search in the anchor list. The anchor list is the
 *    active set of treewalk_run with haswork NULL, so the treewalk uses it as its queue as is.
 *    A rank with more queries than live particles (or than the batch cap) issues them in
 *    batches; every rank runs the same number of walks (global maximum of the batch counts).
 *  - Visits: treewalk_visit_nolist_ngbiter with NoNgblist = 1, accumulating directly in
 *    ngbiter; garbage (skipped by the visit already) and swallowed particles are excluded.
 *  - Doubling: a manual redo loop. Each round walks every central still pending (in batches),
 *    then each central either is final or has its radius doubled and stays pending; the rounds
 *    end when no rank has a pending central. At most SubhaloSOMaxDoublings + 1 rounds.
 *  - Pass 2 classifies a particle into the crossing bin with the same function and the same
 *    (Rlo, Scale) as pass 1, so the pass-2 particle set of a crossing bin is exactly the pass-1
 *    one, and the cumulative mass at the outer edge agrees.
 *
 * Memory (per rank): 448 bytes per central for the whole call (struct so_row) plus 8 for the
 * pending list, and per query slot of a batch 4 + 8 bytes of index and max(SubhaloSOBins x 48,
 * 4352) bytes of accumulator, the batch capped at 1/8 of the free bytes after the tree build.
 * The treewalk export buffers take the rest of the free memory as for any walk: query 64 / 168
 * bytes, result SubhaloSOBins x 48 / 4352 bytes for pass 1 / pass 2 (+ 8 each with DEBUG).
 * treewalk.c's ev_begin wants 40960 x (24 + (1 + ImportBufferBoost) (query + result)) free
 * bytes: with the defaults (64 bins, boost 2) 386 MB for pass 1 and 556 MB for pass 2. */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include "partmanager.h"
#include "forcetree.h"
#include "treewalk.h"
#include "gravity.h"
#include "subhalo_unbind.h"   /* subhalo_so_thresholds */
#include "subhalo_so.h"

#define SO_NT 6
/* definitions with per-type masses: vir (SO slot 0) and 200c (SO slot 2) */
#define SO_DEF_VIR 0
#define SO_DEF_200C 2
#define SO_MINBINS 8
#define SO_MAXBINS 256

static int64_t SOMaxBatch = 0;

static inline int64_t so_imin(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t so_imax(int64_t a, int64_t b) { return a > b ? a : b; }

void
subhalo_so_set_max_batch(int64_t maxbatch)
{
    SOMaxBatch = maxbatch;
}

/* Per-type slot of a definition: 0 for vir, 1 for 200c, -1 otherwise. */
static inline int
so_typeslot(int d)
{
    return d == SO_DEF_VIR ? 0 : (d == SO_DEF_200C ? 1 : -1);
}

/* One central on this rank, alive for the whole call. */
struct so_row {
    double Pos[3];                 /* core, translated comoving frame */
    double R, Rlo, Scale;          /* search radius, inner log edge, SubhaloSOBins / ln(R / Rlo) */
    /* pass-1 outcome per definition: crossing bin [Rin, Rout], cumulative mass at Rin and the
     * bin's content, total and (vir, 200c) per type */
    double Rin[SUBHALO_NSO], Rout[SUBHALO_NSO], Cin[SUBHALO_NSO], Bin[SUBHALO_NSO];
    double CinT[2][SO_NT], BinT[2][SO_NT];
    int64_t rec;                   /* index into recs */
    int jc[SUBHALO_NSO];           /* crossing bin (j + 1); -1 none; SubhaloSOBins: cut off at R */
    int ndoubl;
    int redo;                      /* pending another pass-1 round */
    int refine;                    /* pass 2 */
    int cutoff;
    int pad;
};

/* Private data of the walks. */
struct so_priv {
    struct so_row * rows;
    const int64_t * batch;         /* row of each query slot */
    const int * anchor;            /* anchor particle of each slot, ascending */
    int64_t nanchor;
    char * acc;                    /* accumulator of each slot */
    size_t accsize;
    int NB;
};

#define SO_PRIV(tw) ((struct so_priv *) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    double R, Rlo, Scale;
} TreeWalkQuerySO1;

typedef struct {
    TreeWalkResultBase base;
    double M[];                    /* [NB][SO_NT] */
} TreeWalkResultSO1;

typedef struct {
    TreeWalkQueryBase base;
    double Hsml, Rlo, Scale;
    double Rin[SUBHALO_NSO], Rout[SUBHALO_NSO];
    int jc[SUBHALO_NSO];           /* pass-1 bin to refine, -1 none */
    int pad;
} TreeWalkQuerySO2;

typedef struct {
    TreeWalkResultBase base;
    float Sub[SUBHALO_NSO][SUBHALO_SO_NSUB];
    float SubT[2][SUBHALO_SO_NSUB][SO_NT];
} TreeWalkResultSO2;

/* The pass-1 bin of radius r. Not inlined, so that pass 1 and pass 2 evaluate it with the same
 * instructions and classify every particle identically. */
static int __attribute__((noinline))
so_bin(double r, double Rlo, double Scale, int NB)
{
    if(r <= Rlo)
        return 0;
    int k = (int) floor(log(r / Rlo) * Scale);
    if(k < 0)
        k = 0;
    if(k >= NB)
        k = NB - 1;
    return k;
}

/* Outer edge of pass-1 bin k. */
static inline double
so_edge(const struct so_row * row, int k, int NB)
{
    if(k >= NB - 1)
        return row->R;
    return row->Rlo * exp((k + 1) / row->Scale);
}

static int64_t
so_slot(const struct so_priv * pv, int place)
{
    int64_t lo = 0, hi = pv->nanchor - 1;
    while(lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if(pv->anchor[mid] == place)
            return mid;
        if(pv->anchor[mid] < place)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    endrun(5, "subhalo_field_so: particle %d is not an anchor of this walk (%ld anchors)\n", place, pv->nanchor);
    return -1;
}

/* ------------------------------------------------------------------ pass 1 */

static void
so1_fill(int place, TreeWalkQuerySO1 * I, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    const struct so_row * row = &pv->rows[pv->batch[so_slot(pv, place)]];
    int d;
    for(d = 0; d < 3; d++)
        I->base.Pos[d] = row->Pos[d];
    I->R = row->R;
    I->Rlo = row->Rlo;
    I->Scale = row->Scale;
}

static void
so1_reduce(int place, TreeWalkResultSO1 * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    double * acc = (double *) (pv->acc + so_slot(pv, place) * pv->accsize);
    const int n = pv->NB * SO_NT;
    int k;
    /* the accumulators are zeroed before the walk, so primary and ghost results both add */
    for(k = 0; k < n; k++)
        acc[k] += O->M[k];
}

static void
so1_ngbiter(TreeWalkQuerySO1 * I, TreeWalkResultSO1 * O, TreeWalkNgbIterBase * iter, LocalTreeWalk * lv)
{
    if(iter->other == -1) {
        iter->Hsml = I->R;
        iter->mask = ALLMASK;
        iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int other = iter->other;
    if(P[other].IsGarbage || P[other].Swallowed)
        return;
    const int k = so_bin(iter->r, I->Rlo, I->Scale, SO_PRIV(lv->tw)->NB);
    O->M[SO_NT * k + P[other].Type] += P[other].Mass;
}

/* ------------------------------------------------------------------ pass 2 */

static void
so2_fill(int place, TreeWalkQuerySO2 * I, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    const struct so_row * row = &pv->rows[pv->batch[so_slot(pv, place)]];
    int d;
    for(d = 0; d < 3; d++)
        I->base.Pos[d] = row->Pos[d];
    I->Rlo = row->Rlo;
    I->Scale = row->Scale;
    double rmax = 0;
    for(d = 0; d < SUBHALO_NSO; d++) {
        const int refined = row->jc[d] >= 1 && row->jc[d] < pv->NB;
        I->jc[d] = refined ? row->jc[d] : -1;
        I->Rin[d] = row->Rin[d];
        I->Rout[d] = row->Rout[d];
        if(refined && row->Rout[d] > rmax)
            rmax = row->Rout[d];
    }
    /* a margin over the outermost crossing bin; so_bin selects the particles */
    I->Hsml = DMIN(row->R, 1.001 * rmax);
    I->pad = 0;
}

static void
so2_reduce(int place, TreeWalkResultSO2 * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    float * acc = (float *) (pv->acc + so_slot(pv, place) * pv->accsize);
    const float * src = &O->Sub[0][0];
    const int n = (sizeof(O->Sub) + sizeof(O->SubT)) / sizeof(float);
    int k;
    for(k = 0; k < n; k++)
        acc[k] += src[k];
}

static void
so2_ngbiter(TreeWalkQuerySO2 * I, TreeWalkResultSO2 * O, TreeWalkNgbIterBase * iter, LocalTreeWalk * lv)
{
    if(iter->other == -1) {
        iter->Hsml = I->Hsml;
        iter->mask = ALLMASK;
        iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int other = iter->other;
    if(P[other].IsGarbage || P[other].Swallowed)
        return;
    const int k = so_bin(iter->r, I->Rlo, I->Scale, SO_PRIV(lv->tw)->NB);
    const int type = P[other].Type;
    const float mass = P[other].Mass;
    int d;
    for(d = 0; d < SUBHALO_NSO; d++) {
        if(I->jc[d] != k)
            continue;
        int m = (int) floor((iter->r - I->Rin[d]) / (I->Rout[d] - I->Rin[d]) * SUBHALO_SO_NSUB);
        if(m < 0)
            m = 0;
        if(m >= SUBHALO_SO_NSUB)
            m = SUBHALO_SO_NSUB - 1;
        O->Sub[d][m] += mass;
        const int ts = so_typeslot(d);
        if(ts >= 0)
            O->SubT[ts][m][type] += mass;
    }
}

/* ------------------------------------------------------------------ the walks */

/* One collective walk over the queries rows[batch[0..nq)) (nq may be 0 on any rank). */
static void
so_walk(int pass, struct so_priv * pv, int64_t nq, ForceTree * tree)
{
    TreeWalk tw[1] = {{0}};
    tw->ev_label = pass == 1 ? "SUBHALO_SO_PASS1" : "SUBHALO_SO_PASS2";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->NoNgblist = 1;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw->haswork = NULL;
    if(pass == 1) {
        tw->ngbiter = (TreeWalkNgbIterFunction) so1_ngbiter;
        tw->fill = (TreeWalkFillQueryFunction) so1_fill;
        tw->reduce = (TreeWalkReduceResultFunction) so1_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQuerySO1);
        tw->result_type_elsize = sizeof(TreeWalkResultSO1) + (size_t) pv->NB * SO_NT * sizeof(double);
    }
    else {
        tw->ngbiter = (TreeWalkNgbIterFunction) so2_ngbiter;
        tw->fill = (TreeWalkFillQueryFunction) so2_fill;
        tw->reduce = (TreeWalkReduceResultFunction) so2_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQuerySO2);
        tw->result_type_elsize = sizeof(TreeWalkResultSO2);
    }
    tw->priv = pv;
    tw->tree = tree;
    pv->nanchor = nq;
    if(nq > 0)
        memset(pv->acc, 0, nq * pv->accsize);
    /* the anchors are live particles, distinct and ascending: with haswork NULL and an active
     * set the treewalk takes the list as its queue */
    treewalk_run(tw, (int *) pv->anchor, nq);
}

/* Log-linear interpolation of M(<r) between (ra, Ma) (density above K = 4/3 pi rho) and
 * (rb, Mb) (not above), solved for M(r) = K r^3. Returns the mass at the crossing. */
static double
so_cross(double ra, double Ma, double rb, double Mb, double K)
{
    if(!(Ma > 0) || !(rb > ra))
        return Ma;
    if(Mb < Ma)
        Mb = Ma;
    const double alpha = log(Mb / Ma) / log(rb / ra);
    if(!(alpha < 3))
        return Mb;
    double r = exp((log(Ma) - alpha * log(ra) - log(K)) / (3 - alpha));
    if(r < ra) r = ra;
    if(r > rb) r = rb;
    double M = K * r * r * r;
    if(M < Ma) M = Ma;
    if(M > Mb) M = Mb;
    return M;
}

/* Pass-1 analysis of one row from its histogram M[NB][SO_NT]. Returns 1 if the row must be
 * walked again with a doubled radius (radius already updated), 0 if its pass-1 outcome is final. */
static int
so_pass1_analyse(struct so_row * row, const double * M, int NB, const double rho[SUBHALO_NSO], double rhomin,
                 int MaxDoublings, double BoxSize, double soft)
{
    int j[SUBHALO_NSO];
    int d, k, t;
    for(d = 0; d < SUBHALO_NSO; d++)
        j[d] = -1;
    double C = 0, Dlast = 0;
    for(k = 0; k < NB; k++) {
        for(t = 0; t < SO_NT; t++)
            C += M[SO_NT * k + t];
        const double e = so_edge(row, k, NB);
        const double D = C / (4. / 3. * M_PI * e * e * e);
        for(d = 0; d < SUBHALO_NSO; d++)
            if(D > rho[d])
                j[d] = k;
        if(k == NB - 1)
            Dlast = D;
    }
    const int above = Dlast > rhomin;
    if(above && row->ndoubl < MaxDoublings && row->R < 0.5 * BoxSize) {
        row->R = DMIN(2 * row->R, 0.5 * BoxSize);
        row->ndoubl++;
        row->Rlo = soft > 0 ? DMIN(soft, 0.25 * row->R) : 1e-3 * row->R;
        row->Scale = NB / log(row->R / row->Rlo);
        return 1;
    }
    row->cutoff = above;
    /* cumulative masses at the inner edge of each crossing bin and the bins' contents */
    double CT[SO_NT] = {0};
    C = 0;
    for(d = 0; d < SUBHALO_NSO; d++) {
        row->jc[d] = j[d] < 0 ? -1 : j[d] + 1;
        row->Cin[d] = row->Bin[d] = 0;
        row->Rin[d] = row->Rout[d] = 0;
        const int ts = so_typeslot(d);
        if(ts >= 0)
            for(t = 0; t < SO_NT; t++)
                row->CinT[ts][t] = row->BinT[ts][t] = 0;
    }
    for(k = 0; k < NB; k++) {
        double binm = 0;
        for(t = 0; t < SO_NT; t++)
            binm += M[SO_NT * k + t];
        for(d = 0; d < SUBHALO_NSO; d++) {
            if(row->jc[d] != k)
                continue;
            /* bin k is the crossing bin of d: C, CT hold the mass inside its inner edge */
            row->Rin[d] = so_edge(row, k - 1, NB);
            row->Rout[d] = so_edge(row, k, NB);
            row->Cin[d] = C;
            row->Bin[d] = binm;
            const int ts = so_typeslot(d);
            if(ts >= 0)
                for(t = 0; t < SO_NT; t++) {
                    row->CinT[ts][t] = CT[t];
                    row->BinT[ts][t] = M[SO_NT * k + t];
                }
        }
        C += binm;
        for(t = 0; t < SO_NT; t++)
            CT[t] += M[SO_NT * k + t];
    }
    /* still above at R (cut off at the cap): the mass inside R */
    for(d = 0; d < SUBHALO_NSO; d++) {
        if(row->jc[d] != NB)
            continue;
        row->Rin[d] = row->Rout[d] = row->R;
        row->Cin[d] = C;
        const int ts = so_typeslot(d);
        if(ts >= 0)
            for(t = 0; t < SO_NT; t++)
                row->CinT[ts][t] = CT[t];
    }
    return 0;
}

/* Final values of definition d of a row; sub/subT are the pass-2 sub-bins or NULL. */
static void
so_finish_def(const struct so_row * row, int d, int NB, double rho, const float * sub, const float * subT,
              struct SubTrack * rec)
{
    const int ts = so_typeslot(d);
    float * mtype = ts == 0 ? rec->MTypeSOVir : (ts == 1 ? rec->MTypeSO200c : NULL);
    int t, m;
    if(row->jc[d] < 0) {
        rec->MSO[d] = rec->RSO[d] = -1;
        if(mtype)
            for(t = 0; t < SO_NT; t++)
                mtype[t] = -1;
        return;
    }
    const double K = 4. / 3. * M_PI * rho;
    double ra = row->Rin[d], Ca = row->Cin[d];
    double CaT[SO_NT], CbT[SO_NT];
    for(t = 0; t < SO_NT; t++)
        CaT[t] = ts >= 0 ? row->CinT[ts][t] : 0;
    double Mfin;
    if(row->jc[d] == NB) {
        /* cut off at the doubling cap: the mass inside R */
        Mfin = Ca;
        for(t = 0; t < SO_NT; t++)
            CbT[t] = CaT[t];
    }
    else {
        double rb, Cb;
        if(sub) {
            const double w = (row->Rout[d] - row->Rin[d]) / SUBHALO_SO_NSUB;
            int mstar = -1;
            double C = Ca;
            for(m = 0; m < SUBHALO_SO_NSUB; m++) {
                C += sub[m];
                const double s = m == SUBHALO_SO_NSUB - 1 ? row->Rout[d] : row->Rin[d] + (m + 1) * w;
                if(C > K * s * s * s)
                    mstar = m;
            }
            for(m = 0; m <= mstar; m++) {
                Ca += sub[m];
                if(ts >= 0)
                    for(t = 0; t < SO_NT; t++)
                        CaT[t] += subT[SO_NT * m + t];
            }
            if(mstar >= 0)
                ra = mstar == SUBHALO_SO_NSUB - 1 ? row->Rout[d] : row->Rin[d] + (mstar + 1) * w;
            if(mstar == SUBHALO_SO_NSUB - 1) {
                /* only by rounding: the pass-1 outer edge was below the threshold */
                rb = ra;
                Cb = Ca;
                for(t = 0; t < SO_NT; t++)
                    CbT[t] = CaT[t];
            }
            else {
                const int mb = mstar + 1;
                rb = mb == SUBHALO_SO_NSUB - 1 ? row->Rout[d] : row->Rin[d] + (mb + 1) * w;
                Cb = Ca + sub[mb];
                for(t = 0; t < SO_NT; t++)
                    CbT[t] = CaT[t] + (ts >= 0 ? subT[SO_NT * mb + t] : 0);
            }
        }
        else {
            rb = row->Rout[d];
            Cb = Ca + row->Bin[d];
            for(t = 0; t < SO_NT; t++)
                CbT[t] = CaT[t] + (ts >= 0 ? row->BinT[ts][t] : 0);
        }
        Mfin = so_cross(ra, Ca, rb, Cb, K);
        const double f = Cb > Ca ? (Mfin - Ca) / (Cb - Ca) : 0;
        for(t = 0; t < SO_NT; t++)
            CbT[t] = CaT[t] + f * (CbT[t] - CaT[t]);
    }
    rec->MSO[d] = Mfin;
    rec->RSO[d] = cbrt(Mfin / K);
    if(mtype)
        for(t = 0; t < SO_NT; t++)
            mtype[t] = CbT[t];
}

static inline double
so_wrap(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

static inline int
so_is_central(const struct SubTrack * r)
{
    return r->Rank == 0 && r->HostGroup >= 0 && r->Nbound > 0;
}

/* Run the walks of one pass over the rows in list[0..nlist) in batches of at most bcap
 * (collective: every rank runs the global maximum number of batches). After each batch the
 * batch's rows are analysed: pass 1 sets row->redo (and doubles R) or the pass-1 outcome,
 * pass 2 writes the final values into the records. Returns the number of walks. */
static int
so_run_batches(int pass, struct so_priv * pv, int64_t * batchbuf, const int64_t * list, int64_t nlist, int64_t bcap,
               ForceTree * tree, const double rho[SUBHALO_NSO], double rhomin, const struct SubhaloParams * par,
               double BoxSize, double soft, struct SubTrack * recs, MPI_Comm Comm)
{
    int64_t nbatch = (nlist > 0 && bcap > 0) ? (nlist + bcap - 1) / bcap : 0;
    if(nlist > 0 && bcap <= 0)
        endrun(5, "subhalo_field_so: %ld queries but no anchor particle on this rank\n", nlist);
    int64_t nbatch_max = 0;
    MPI_Allreduce(&nbatch, &nbatch_max, 1, MPI_INT64, MPI_MAX, Comm);
    int64_t b;
    for(b = 0; b < nbatch_max; b++) {
        const int64_t lo = so_imin(b * bcap, nlist);
        const int64_t hi = so_imin(lo + bcap, nlist);
        const int64_t nq = hi - lo;
        int64_t s;
        for(s = 0; s < nq; s++)
            batchbuf[s] = list[lo + s];
        so_walk(pass, pv, nq, tree);
        const int NB = pv->NB;
        #pragma omp parallel for
        for(s = 0; s < nq; s++) {
            struct so_row * row = &pv->rows[batchbuf[s]];
            if(pass == 1) {
                const double * M = (const double *) (pv->acc + s * pv->accsize);
                row->redo = so_pass1_analyse(row, M, NB, rho, rhomin, par->SubhaloSOMaxDoublings, BoxSize, soft);
            }
            else {
                const float * acc = (const float *) (pv->acc + s * pv->accsize);
                struct SubTrack * rec = &recs[row->rec];
                int d;
                for(d = 0; d < SUBHALO_NSO; d++) {
                    const int ts = so_typeslot(d);
                    const int refined = row->jc[d] >= 1 && row->jc[d] < NB;
                    so_finish_def(row, d, NB, rho[d], refined ? acc + d * SUBHALO_SO_NSUB : NULL,
                                  (refined && ts >= 0) ? acc + SUBHALO_NSO * SUBHALO_SO_NSUB + ts * SUBHALO_SO_NSUB * SO_NT : NULL,
                                  rec);
                }
            }
        }
    }
    return (int) nbatch_max;
}

void
subhalo_field_so_full(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                      const Cosmology * CP, const struct SubhaloParams * par, const double * rho_in,
                      double * Rsearch, struct SubhaloSOStats * stats, MPI_Comm Comm)
{
    const double tstart = MPI_Wtime();
    const int NB = par->SubhaloSOBins;
    if(NB < SO_MINBINS || NB > SO_MAXBINS)
        endrun(1, "SubhaloSOBins = %d must be in [%d, %d].\n", NB, SO_MINBINS, SO_MAXBINS);
    if(!(par->SubhaloSOSearchFactor > 0) || par->SubhaloSOMaxDoublings < 0 || par->SubhaloSOMaxDoublings > 60)
        endrun(1, "SubhaloSOSearchFactor = %g must be positive and SubhaloSOMaxDoublings = %d in [0, 60].\n",
               par->SubhaloSOSearchFactor, par->SubhaloSOMaxDoublings);
    double rho[SUBHALO_NSO];
    int d;
    if(rho_in)
        memcpy(rho, rho_in, sizeof(rho));
    else
        subhalo_so_thresholds(CP, atime, rho);
    double rhomin = rho[0];
    for(d = 0; d < SUBHALO_NSO; d++) {
        if(!(rho[d] > 0))
            endrun(1, "subhalo_field_so: threshold %d is %g, not positive\n", d, rho[d]);
        if(rho[d] < rhomin)
            rhomin = rho[d];
    }
    const double BoxSize = PartManager->BoxSize;
    const double soft = FORCE_SOFTENING() / 2.8;

    int64_t i, ncen = 0;
    for(i = 0; i < n; i++) {
        if(Rsearch)
            Rsearch[i] = -1;
        if(so_is_central(&recs[i]))
            ncen++;
    }

    /* the rows (whole call), then the tree, then the batch buffers: freed in reverse */
    struct so_row * rows = (struct so_row *) mymalloc("SOrows", so_imax(ncen, 1) * sizeof(struct so_row));
    int64_t * list = (int64_t *) mymalloc("SOlist", so_imax(ncen, 1) * sizeof(int64_t));
    int64_t c = 0;
    for(i = 0; i < n; i++) {
        struct SubTrack * r = &recs[i];
        if(!so_is_central(r))
            continue;
        struct so_row * row = &rows[c];
        memset(row, 0, sizeof(*row));
        row->rec = i;
        for(d = 0; d < 3; d++)
            row->Pos[d] = so_wrap(r->CorePos[d] + PartManager->CurrentParticleOffset[d], BoxSize);
        double R = par->SubhaloSOSearchFactor * cbrt(3. * r->Mbound / (4. * M_PI * rhomin));
        if(!(R > soft))
            R = soft > 0 ? soft : 1e-3 * BoxSize;
        if(R > 0.5 * BoxSize)
            R = 0.5 * BoxSize;
        row->R = R;
        row->Rlo = soft > 0 ? DMIN(soft, 0.25 * R) : 1e-3 * R;
        row->Scale = NB / log(row->R / row->Rlo);
        list[c] = c;
        c++;
        r->Flags &= ~(SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS);
    }

    ForceTree tree = {0};
    force_tree_full(&tree, ddecomp, 0, NULL);

    /* batch capacity: live anchors, the test knob and 1/8 of the free memory */
    const size_t acc1 = (size_t) NB * SO_NT * sizeof(double);
    const size_t acc2 = sizeof(((TreeWalkResultSO2 *) 0)->Sub) + sizeof(((TreeWalkResultSO2 *) 0)->SubT);
    const size_t accsize = acc1 > acc2 ? acc1 : acc2;
    const size_t slotbytes = accsize + sizeof(int) + sizeof(int64_t);
    int64_t bcap = ncen;
    if(SOMaxBatch > 0 && bcap > SOMaxBatch)
        bcap = SOMaxBatch;
    const int64_t memcap = so_imax((int64_t) (mymalloc_freebytes() / 8 / slotbytes), 1);
    if(bcap > memcap)
        bcap = memcap;
    int * anchor = (int *) mymalloc("SOanchor", so_imax(bcap, 1) * sizeof(int));
    int64_t nlive = 0;
    for(i = 0; i < PartManager->NumPart && nlive < bcap; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        anchor[nlive++] = (int) i;
    }
    bcap = nlive;
    int64_t * batchbuf = (int64_t *) mymalloc("SObatch", so_imax(bcap, 1) * sizeof(int64_t));
    char * acc = (char *) mymalloc("SOacc", so_imax(bcap, 1) * accsize);

    struct so_priv pv = {0};
    pv.rows = rows;
    pv.batch = batchbuf;
    pv.anchor = anchor;
    pv.acc = acc;
    pv.NB = NB;

    /* pass 1 with the doubling rounds */
    int npass1 = 0, npass2 = 0;
    int64_t npend = ncen;
    pv.accsize = acc1;
    while(1) {
        if(!MPIU_Any(npend > 0, Comm))
            break;
        npass1 += so_run_batches(1, &pv, batchbuf, list, npend, bcap, &tree, rho, rhomin, par, BoxSize, soft, recs, Comm);
        int64_t k, m = 0;
        for(k = 0; k < npend; k++)
            if(rows[list[k]].redo)
                list[m++] = list[k];
        npend = m;
    }

    /* pass 2 for the large centrals with a crossing bin; the others finish here */
    int64_t nref = 0;
    for(c = 0; c < ncen; c++) {
        struct so_row * row = &rows[c];
        struct SubTrack * rec = &recs[row->rec];
        int any = 0;
        for(d = 0; d < SUBHALO_NSO; d++)
            any |= row->jc[d] >= 1 && row->jc[d] < NB;
        row->refine = any && rec->Nbound >= par->SubhaloSORefineMinPart;
        if(row->refine)
            list[nref++] = c;
        else
            for(d = 0; d < SUBHALO_NSO; d++)
                so_finish_def(row, d, NB, rho[d], NULL, NULL, rec);
    }
    pv.accsize = acc2;
    npass2 = so_run_batches(2, &pv, batchbuf, list, nref, bcap, &tree, rho, rhomin, par, BoxSize, soft, recs, Comm);

    myfree(acc);
    myfree(batchbuf);
    myfree(anchor);
    force_tree_free(&tree);

    /* flags and diagnostics */
    int64_t cnt[5] = {ncen, nref, 0, 0, 0};
    for(c = 0; c < ncen; c++) {
        const struct so_row * row = &rows[c];
        struct SubTrack * rec = &recs[row->rec];
        int nocross = 0;
        for(d = 0; d < SUBHALO_NSO; d++)
            nocross |= row->jc[d] < 0;
        if(row->cutoff)
            rec->Flags |= SUBHALO_FLAG_SOCUTOFF;
        if(nocross)
            rec->Flags |= SUBHALO_FLAG_SONOCROSS;
        cnt[2] += row->ndoubl > 0;
        cnt[3] += row->cutoff;
        cnt[4] += nocross;
        if(Rsearch)
            Rsearch[row->rec] = row->R;
    }
    myfree(list);
    myfree(rows);

    int64_t tot[5];
    MPI_Allreduce(cnt, tot, 5, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo field SO: %ld centrals, %ld refined, %ld doubled, %ld cut off at the cap, %ld without a crossing; "
               "%d + %d walks, %.3g s.\n", tot[0], tot[1], tot[2], tot[3], tot[4], npass1, npass2, MPI_Wtime() - tstart);
    if(stats) {
        stats->Ncentral = tot[0];
        stats->Nrefined = tot[1];
        stats->Ndoubled = tot[2];
        stats->Ncutoff = tot[3];
        stats->Nnocross = tot[4];
        stats->Npass1Walks = npass1;
        stats->Npass2Walks = npass2;
    }
}

void
subhalo_field_so(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                 const Cosmology * CP, const struct SubhaloParams * par, MPI_Comm Comm)
{
    subhalo_field_so_full(recs, n, ddecomp, atime, CP, par, NULL, NULL, NULL, Comm);
}
