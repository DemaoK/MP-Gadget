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
 *  - Ranks without a live particle: their centrals are routed (whole records) to the lowest
 *    rank with a live particle, which walks them as its own and sends them back (subhalo_so.h).
 *  - Visits: treewalk_visit_nolist_ngbiter with NoNgblist = 1, accumulating directly in
 *    ngbiter; garbage (skipped by the visit already) and swallowed particles are excluded.
 *  - Types: the results carry the masses of the particle types present among the live
 *    particles of the whole run only (NT <= 6 slots; one for a DM-only run), which loses nothing.
 *  - Doubling: a manual redo loop. Each round walks every central still pending (in batches),
 *    then each central either is final or has its radius doubled and stays pending; the rounds
 *    end when no rank has a pending central. At most SubhaloSOMaxDoublings + 1 rounds.
 *  - Pass 2 classifies a particle into the crossing bin with the same function and the same
 *    (Rlo, Scale) as pass 1, so the pass-2 particle set of a crossing bin is exactly the pass-1
 *    one, and the cumulative mass at the outer edge agrees.
 *  - Accumulation: both passes add particle masses in double, in the walk's results, their
 *    merges and the analysis; only the record's f4 fields round. For vir and 200c pass 2
 *    carries the per-type sub-bins only and the total sub-bins are their sums, so the refined
 *    total and the per-type masses come from the same additions. Accuracy bound: a refined
 *    (or pass-1) mass of N particles carries a relative summation error of at most about
 *    N 2^-53 (in practice ~sqrt(N) 2^-53), which depends on the particle partition and the
 *    traversal order only at that level, far below the f4 rounding of the output (2^-24);
 *    the per-type masses sum to MSO within the f4 roundings. This is not bitwise
 *    reproducibility across partitions for arbitrary masses (it is for dyadic masses, whose
 *    sums are exact); test_subhalo_so.c checks both. The SO columns of the catalogues compare
 *    bitwise across layouts in practice because these double errors almost never move an f4
 *    rounding.
 *  - Pass 3 (decision 44) does not add its own rounding to that: its masses are 128-bit
 *    fixed-point integers (subhalo_so.h, exact for every particle mass in practice), its count an
 *    integer and its largest mass a max, so its profile is bitwise independent of the partition,
 *    the traversal, the threads and the merge order for any masses, GIVEN its inputs: the centre,
 *    the published f4 MSO[vir] / RSO[vir] and the live particle data; the analysis is a serial
 *    function of that profile and of those f4 values. Its NFW fit, a Newton iteration with
 *    comparisons, would not be robust against the last-bit differences that passes 1 and 2
 *    tolerate. Those passes' double sums do set its inputs, though: where their layout-dependent
 *    rounding moves a published f4 MSO[vir] or RSO[vir] (rare, as above; never for dyadic
 *    masses), the pass-3 sphere and its four columns move with it (audit 5e R3).
 *
 * Memory (per rank): 456 bytes per central for the whole call (struct so_row) plus 8 for the
 * pending list, and per query slot of a batch 4 + 8 + 4 NThreads bytes of index (the last for the
 * treewalk's queue copy when black holes exist) and max(NB NT 8, (3 + 2 NT) 64 x 8, 1936) bytes of
 * accumulator (each term only for the passes that run; 1936 = pass 3's 16 + 16 SUBHALO_SO_NPROF),
 * the batch capped at 1/8 of the free bytes after the tree build and at what leaves the treewalk
 * its reserve. Results: pass 1 NB NT 8 bytes, pass 2 (3 + 2 NT) 512 bytes (+ 8 each with DEBUG),
 * pass 3 1936 bytes (1944 with DEBUG); queries 64 / 168 / 64 bytes. treewalk.c's ev_begin wants
 * (40960 + max(4097, 100 NThreads)) x (24 + (1 + ImportBufferBoost) (query + result)) free bytes
 * (subhalo_so_walk_bytes): with the defaults (64 bins, boost 2, <= 40 threads) and NT = 1,
 * 75 MiB for pass 1 and 353 MiB for pass 2; with NT = 6, 405 MiB and 1013 MiB (before the type
 * compaction pass 1 always needed the NT = 6 figure; the float pass 2 needed 254 and 584 MiB);
 * pass 3, 6024 bytes per buffer whatever NB and NT, 259 MiB (271,423,368 bytes; 135 MiB with the
 * former 64-bit masses, audit 5e R1), below pass 2's 353 MiB. Pass 3 is admitted on its own
 * (its columns -1 if it does not fit) and the batch cap keeps the largest reserve of the passes
 * that run.
 * Before the walks the call checks that requirement against the free memory after the tree
 * build (all ranks): pass 1 halves its bins down to 16 until it fits, pass 2 is skipped (pass-1
 * interpolation for every central) if it does not fit, and the whole call is skipped (records
 * at -1) if pass 1 does not fit at 16 bins.
 */

#include <mpi.h>
#include <math.h>
#include <float.h>
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
#include "subhalo_unbind.h"   /* subhalo_so_thresholds, the NFW fit and Klypin helpers */
#include "subhalo_internal.h" /* subhalo_vel_to_kms */
#include "subhalo_so.h"

#define SO_NT 6
/* definitions with per-type masses: vir (SO slot 0) and 200c (SO slot 2) */
#define SO_DEF_VIR 0
#define SO_DEF_200C 2
#define SO_MINBINS 8
#define SO_MAXBINS 256
/* pass 1 halves its bins down to this many when the treewalk reserve does not fit */
#define SO_SHRINKBINS 16
/* allocator slack on top of the computed requirements */
#define SO_SLACK (1L << 20)

static int64_t SOMaxBatch = 0;
/* treewalk.c keeps ImportBufferBoost private: the parameter's default (gadget/params.c) unless
 * subhalo_so_set_import_buffer_boost says otherwise */
static double SOImportBufferBoost = 2.;

static inline int64_t so_imin(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t so_imax(int64_t a, int64_t b) { return a > b ? a : b; }

void
subhalo_so_set_max_batch(int64_t maxbatch)
{
    SOMaxBatch = maxbatch;
}

void
subhalo_so_set_import_buffer_boost(double boost)
{
    SOImportBufferBoost = boost > 0 ? boost : 0;
}

/* Per-type slot of a definition: 0 for vir, 1 for 200c, -1 otherwise. */
static inline int
so_typeslot(int d)
{
    return d == SO_DEF_VIR ? 0 : (d == SO_DEF_200C ? 1 : -1);
}

/* One central walked by this rank (its own or a routed one), alive for the whole call. */
struct so_row {
    double Pos[3];                 /* core, translated comoving frame */
    double R, Rlo, Scale;          /* search radius, inner log edge, NB / ln(R / Rlo) */
    /* pass-1 outcome per definition: crossing bin [Rin, Rout], cumulative mass at Rin and the
     * bin's content, total and (vir, 200c) per type */
    double Rin[SUBHALO_NSO], Rout[SUBHALO_NSO], Cin[SUBHALO_NSO], Bin[SUBHALO_NSO];
    double CinT[2][SO_NT], BinT[2][SO_NT];
    struct SubTrack * rec;         /* the record written */
    double * rsearch;              /* where the final search radius goes, or NULL */
    int jc[SUBHALO_NSO];           /* crossing bin (j + 1); -1 none; NB: cut off at R */
    int ndoubl;
    int redo;                      /* pending another pass-1 round */
    int refine;                    /* pass 2 */
    int cutoff;
    int pad;
};

/* The particle types present (globally) and their result slots. */
struct so_types {
    int NT;                        /* number of slots */
    int slot[SO_NT];               /* slot of each type, -1 absent */
    int type[SO_NT];               /* type of each slot */
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
    struct so_types ty;
    /* pass 3 (decision 44): the fixed-point mass scale (a power of two), the profile's bins per
     * unit ln r, and the constants of the analysis */
    double qscale, pscale;
    double G, atime, vu, soft, rhovir;
    int64_t nprofile;              /* pass-3 rows with >= SUBHALO_NFW_MINPART particles (this rank) */
};

#define SO_PRIV(tw) ((struct so_priv *) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    double R, Rlo, Scale;
} TreeWalkQuerySO1;

typedef struct {
    TreeWalkResultBase base;
    double M[];                    /* [NB][NT] */
} TreeWalkResultSO1;

typedef struct {
    TreeWalkQueryBase base;
    double Hsml, Rlo, Scale;
    double Rin[SUBHALO_NSO], Rout[SUBHALO_NSO];
    int jc[SUBHALO_NSO];           /* pass-1 bin to refine, -1 none */
    int pad;
} TreeWalkQuerySO2;

/* Pass 2 accumulates in double, and a definition with per-type masses (vir, 200c) carries only
 * its per-type sub-bins: its total sub-bins are their sums (so_run_batches), so the refined total
 * and the per-type masses come from the same additions and sum to each other up to the double
 * rounding of the final sum over types. The definitions without per-type masses carry their
 * total sub-bins. */
#define SO_NUNTYPED (SUBHALO_NSO - 2)

typedef struct {
    TreeWalkResultBase base;
    /* Sub[SO_NUNTYPED][SUBHALO_SO_NSUB] (200b, 500c, 2500c) then SubT[2][SUBHALO_SO_NSUB][NT] */
    double S[];
} TreeWalkResultSO2;

/* Pass 3 (decision 44): the total-mass profile of the virial sphere r <= RSO[vir], in
 * SUBHALO_SO_NPROF logarithmic bins between Rin = SUBHALO_SO_PROF_RIN RSO[vir] and RSO[vir]
 * (r <= Rin in bin 0), the masses as 128-bit fixed-point integers (qscale = 2^s units per mass
 * unit, subhalo_so_prof_unit), so that every sum is exact and associative: for the same sphere
 * and particles the profile is the same integer vector whatever the particle partition, the
 * traversal and the merge order. */
struct so3_acc {
    int64_t N;                     /* particles inside RSO[vir] */
    double mmax;                   /* their largest mass (merged by max) */
    subhalo_so_q Q[SUBHALO_SO_NPROF];   /* fixed-point mass per bin */
};

typedef struct {
    TreeWalkQueryBase base;
    double Hsml, R, Rin;
} TreeWalkQuerySO3;

typedef struct {
    TreeWalkResultBase base;
    struct so3_acc A;
} TreeWalkResultSO3;

/* Accumulator (result without its base) bytes of a pass. */
static size_t
so_acc_bytes(int pass, int NB, int NT)
{
    if(pass == 1)
        return (size_t) NB * NT * sizeof(double);
    if(pass == 3)
        return sizeof(struct so3_acc);
    return (size_t) (SO_NUNTYPED + 2 * NT) * SUBHALO_SO_NSUB * sizeof(double);
}

/* Offset (in doubles) of the total sub-bins of definition d (without per-type masses) in a
 * pass-2 accumulator. */
static inline size_t
so2_uoff(int d)
{
    int e, u = 0;
    for(e = 0; e < d; e++)
        u += so_typeslot(e) < 0;
    return (size_t) u * SUBHALO_SO_NSUB;
}

/* Offset (in doubles) of the per-type sub-bins of type slot ts in a pass-2 accumulator. */
static inline size_t
so2_toff(int ts, int NT)
{
    return (size_t) SO_NUNTYPED * SUBHALO_SO_NSUB + (size_t) ts * SUBHALO_SO_NSUB * NT;
}

size_t
subhalo_so_walk_bytes(int pass, int NB, int NT)
{
    const size_t q = pass == 1 ? sizeof(TreeWalkQuerySO1) : (pass == 2 ? sizeof(TreeWalkQuerySO2) : sizeof(TreeWalkQuerySO3));
    /* pass 3's result has no flexible part: its accumulator is inside the struct */
    const size_t r = pass == 3 ? sizeof(TreeWalkResultSO3)
                   : (pass == 1 ? sizeof(TreeWalkResultSO1) : sizeof(TreeWalkResultSO2)) + so_acc_bytes(pass, NB, NT);
    /* as ev_begin: bytes per buffer, the 40960-buffer reserve, then more than 4096 buffers and
     * at least 100 per thread */
    const size_t bpb = sizeof(struct data_index) + q + r + (size_t) ceil(SOImportBufferBoost * (q + r));
    const int64_t nbuf = 4096 * 10 + so_imax(4096 + 1, 100 * (int64_t) omp_get_max_threads());
    return (size_t) nbuf * bpb;
}

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
    const int n = pv->NB * pv->ty.NT;
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
    const struct so_priv * pv = SO_PRIV(lv->tw);
    const int k = so_bin(iter->r, I->Rlo, I->Scale, pv->NB);
    /* every live particle's type has a slot (the types are those of the live particles) */
    O->M[pv->ty.NT * k + pv->ty.slot[P[other].Type]] += P[other].Mass;
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
    double * acc = (double *) (pv->acc + so_slot(pv, place) * pv->accsize);
    const size_t n = so_acc_bytes(2, pv->NB, pv->ty.NT) / sizeof(double);
    size_t k;
    for(k = 0; k < n; k++)
        acc[k] += O->S[k];
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
    const struct so_priv * pv = SO_PRIV(lv->tw);
    const int NT = pv->ty.NT;
    const int k = so_bin(iter->r, I->Rlo, I->Scale, pv->NB);
    const int tslot = pv->ty.slot[P[other].Type];
    const double mass = P[other].Mass;
    int d;
    for(d = 0; d < SUBHALO_NSO; d++) {
        if(I->jc[d] != k)
            continue;
        int m = (int) floor((iter->r - I->Rin[d]) / (I->Rout[d] - I->Rin[d]) * SUBHALO_SO_NSUB);
        if(m < 0)
            m = 0;
        if(m >= SUBHALO_SO_NSUB)
            m = SUBHALO_SO_NSUB - 1;
        const int ts = so_typeslot(d);
        if(ts >= 0)
            O->S[so2_toff(ts, NT) + (size_t) m * NT + tslot] += mass;
        else
            O->S[so2_uoff(d) + m] += mass;
    }
}

/* ------------------------------------------------------------------ pass 3 */

double
subhalo_so_prof_unit(int64_t nlive, double mmax)
{
    if(!(mmax > 0))
        return 1;
    int emax, bits = 0;
    frexp(mmax, &emax);                        /* mmax < 2^emax */
    while(bits < 63 && ((uint64_t) 1 << bits) < (uint64_t) nlive)
        bits++;
    return ldexp(1.0, 126 - emax - bits);
}

/* m qscale is exact (a power-of-two scaling); rint rounds it to the nearest integer (a no-op from
 * 2^52 on), below 2^127, which converts to the 128-bit integer exactly */
static inline subhalo_so_q
so_prof_q(double m, double qscale)
{
    return (subhalo_so_q) rint(m * qscale);
}

subhalo_so_q
subhalo_so_prof_quantise(double m, double qscale)
{
    return so_prof_q(m, qscale);
}

/* The profile bin of radius r for inner edge Rin (bins per unit ln r pscale). Not inlined, so
 * that the walk and any later use classify with the same instructions. */
static int __attribute__((noinline))
so_prof_bin(double r, double Rin, double pscale)
{
    if(r <= Rin)
        return 0;
    int k = (int) floor(log(r / Rin) * pscale);
    if(k < 0)
        k = 0;
    if(k >= SUBHALO_SO_NPROF)
        k = SUBHALO_SO_NPROF - 1;
    return k;
}

static void
so3_fill(int place, TreeWalkQuerySO3 * I, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    const struct so_row * row = &pv->rows[pv->batch[so_slot(pv, place)]];
    int d;
    for(d = 0; d < 3; d++)
        I->base.Pos[d] = row->Pos[d];
    /* the published (f4) virial radius: the sphere is the one the catalogue states */
    I->R = row->rec->RSO[SO_DEF_VIR];
    I->Rin = SUBHALO_SO_PROF_RIN * I->R;
    /* a margin; the ngbiter selects r <= R */
    I->Hsml = 1.001 * I->R;
}

static void
so3_reduce(int place, TreeWalkResultSO3 * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    const struct so_priv * pv = SO_PRIV(tw);
    struct so3_acc * acc = (struct so3_acc *) (pv->acc + so_slot(pv, place) * pv->accsize);
    int k;
    /* integers: exact and associative; the max is order independent */
    acc->N += O->A.N;
    if(O->A.mmax > acc->mmax)
        acc->mmax = O->A.mmax;
    for(k = 0; k < SUBHALO_SO_NPROF; k++)
        acc->Q[k] += O->A.Q[k];
}

static void
so3_ngbiter(TreeWalkQuerySO3 * I, TreeWalkResultSO3 * O, TreeWalkNgbIterBase * iter, LocalTreeWalk * lv)
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
    if(!(iter->r <= I->R))
        return;
    const struct so_priv * pv = SO_PRIV(lv->tw);
    const double m = P[other].Mass;
    O->A.Q[so_prof_bin(iter->r, I->Rin, pv->pscale)] += so_prof_q(m, pv->qscale);
    O->A.N++;
    if(m > O->A.mmax)
        O->A.mmax = m;
}

/* The pass-3 columns of a row from its profile (decision 44): RsSO, RsKlypinSO, VmaxSO, RmaxSO,
 * all -1 first and -1 with fewer than SUBHALO_NFW_MINPART particles inside RSO[vir]. Returns 1
 * when the row has at least SUBHALO_NFW_MINPART particles (the diagnostic count), 0 otherwise.
 *  - Edges e[j], j = 1..NPROF: Rin exp(j / pscale), the last one RSO[vir]; e[0] = 0. Cumulative
 *    mass C[j] = M(<= e[j]) (the exact integer prefix sum, converted once to double). The
 *    columns are a function of this profile and of the f4 MSO[vir] / RSO[vir] only, so bitwise
 *    layout independent given those (audit 5e R3).
 *  - VmaxSO / RmaxSO: the largest C[j] / r over the edges with r = max(e[j], softening) whose
 *    enclosed density C[j] / (4/3 pi r^3) exceeds rho[vir] (Rockstar's vmax rule at the edges;
 *    the first edge attaining it); VmaxSO = sqrt(G C / (a r)) in km/s. Resolution: one bin,
 *    ln(1000) / NPROF in ln r. No such edge leaves all four columns -1 although the sphere has
 *    enough particles (audit 5e R2; subhalo_so.h): the mass is in the last bin, just inside
 *    RSO[vir], and the f4 RSO[vir] is at or above the density crossing.
 *  - RsKlypinSO: subhalo_klypin_rs(RSO[vir], MSO[vir], that maximum, RmaxSO).
 *  - RsSO: subhalo_nfw_fit from RsKlypinSO on the equal-mass bins of subhalo_nfw_bin_plan (N
 *    particles, the mass inside RSO[vir], the largest mass): bin b closes where M(<r) reaches
 *    b mpbin (subhalo_nfw_threshold picks the profile bin), its edge interpolated inside that
 *    profile bin linearly in ln r (in r^3 in bin 0, which reaches r = 0): the histogram form of
 *    nfw.c's excess-mass edge interpolation. */
static int
so_profile_finish(const struct so_row * row, const struct so3_acc * A, const struct so_priv * pv)
{
    struct SubTrack * rec = row->rec;
    const int NP = SUBHALO_SO_NPROF;
    int j, b;
    rec->RsSO = rec->RsKlypinSO = rec->VmaxSO = rec->RmaxSO = -1;
    if(A->N < SUBHALO_NFW_MINPART)
        return 0;
    const double R = rec->RSO[SO_DEF_VIR], Mso = rec->MSO[SO_DEF_VIR];
    const double Rin = SUBHALO_SO_PROF_RIN * R;
    double e[SUBHALO_SO_NPROF + 1], C[SUBHALO_SO_NPROF + 1];
    subhalo_so_q cq = 0;
    e[0] = C[0] = 0;
    for(j = 1; j <= NP; j++) {
        cq += A->Q[j - 1];
        C[j] = (double) cq / pv->qscale;          /* qscale is a power of two */
        e[j] = j < NP ? Rin * exp(j / pv->pscale) : R;
    }
    const double K = 4. / 3. * M_PI * pv->rhovir;
    double best = 0, rbest = -1;
    for(j = 1; j <= NP; j++) {
        const double r = e[j] > pv->soft ? e[j] : pv->soft;
        if(C[j] > K * r * r * r && C[j] / r > best) {
            best = C[j] / r;
            rbest = r;
        }
    }
    if(!(best > 0))
        return 1;
    rec->VmaxSO = (float) (sqrt(pv->G * best / pv->atime) * pv->vu);
    rec->RmaxSO = (float) rbest;
    const double rk = subhalo_klypin_rs(R, Mso, best, rbest);
    rec->RsKlypinSO = (float) (rk > 0 ? rk : -1);
    double mpbin;
    const int nb = subhalo_nfw_bin_plan(A->N, C[NP], A->mmax, &mpbin);
    double edge[SUBHALO_NFW_MAXBINS + 1];
    edge[0] = 0;
    j = 1;
    for(b = 1; b <= nb; b++) {
        const double T = subhalo_nfw_threshold(b, mpbin);
        while(j <= NP && C[j] < T)
            j++;
        if(j > NP)
            break;
        /* C[j - 1] < T <= C[j]: the crossing is inside profile bin j - 1 */
        double f = (b * mpbin - C[j - 1]) / (C[j] - C[j - 1]);
        if(f < 0) f = 0;
        if(f > 1) f = 1;
        edge[b] = j == 1 ? e[1] * cbrt(f) : e[j - 1] * pow(e[j] / e[j - 1], f);
    }
    const double rs = subhalo_nfw_fit(rk, edge, b - 1, 3 * pv->soft);
    rec->RsSO = (float) (rs > 0 ? rs : -1);
    return 1;
}

/* ------------------------------------------------------------------ the walks */

/* Phase 5d: the export tables the treewalk sizes from the free memory (NThread x BunchSize
 * data_index entries) are elastic, not a requirement of the field SO. Every walk replaces them
 * in the main allocator's low-water mark by the free bytes it would have had without them, so
 * that the finder's per-stage high water measures its own needs; the true low-water mark of the
 * walks (the absolute high water, SubhaloPeakBytes) and the largest tables are kept here. A
 * table counts with its whole allocator charge (payload rounded up to the allocator's alignment,
 * plus the block header), so nothing of it stays in the increment (audit 5d R2). */
static size_t SOWalkMinFree = (size_t) -1;
static double SOWalkElastic = 0;

void
subhalo_so_elastic_reset(void)
{
    SOWalkMinFree = (size_t) -1;
    SOWalkElastic = 0;
}

size_t
subhalo_so_elastic_min_free(void)
{
    return SOWalkMinFree;
}

double
subhalo_so_elastic_bytes(void)
{
    return SOWalkElastic;
}

/* One collective walk over the queries rows[batch[0..nq)) (nq may be 0 on any rank). */
static void
so_walk(int pass, struct so_priv * pv, int64_t nq, ForceTree * tree)
{
    TreeWalk tw[1] = {{0}};
    tw->ev_label = pass == 1 ? "SUBHALO_SO_PASS1" : (pass == 2 ? "SUBHALO_SO_PASS2" : "SUBHALO_SO_PASS3");
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->NoNgblist = 1;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw->haswork = NULL;
    if(pass == 1) {
        tw->ngbiter = (TreeWalkNgbIterFunction) so1_ngbiter;
        tw->fill = (TreeWalkFillQueryFunction) so1_fill;
        tw->reduce = (TreeWalkReduceResultFunction) so1_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQuerySO1);
        tw->result_type_elsize = sizeof(TreeWalkResultSO1) + so_acc_bytes(1, pv->NB, pv->ty.NT);
    }
    else if(pass == 3) {
        tw->ngbiter = (TreeWalkNgbIterFunction) so3_ngbiter;
        tw->fill = (TreeWalkFillQueryFunction) so3_fill;
        tw->reduce = (TreeWalkReduceResultFunction) so3_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQuerySO3);
        tw->result_type_elsize = sizeof(TreeWalkResultSO3);
    }
    else {
        tw->ngbiter = (TreeWalkNgbIterFunction) so2_ngbiter;
        tw->fill = (TreeWalkFillQueryFunction) so2_fill;
        tw->reduce = (TreeWalkReduceResultFunction) so2_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQuerySO2);
        tw->result_type_elsize = sizeof(TreeWalkResultSO2) + so_acc_bytes(2, pv->NB, pv->ty.NT);
    }
    tw->priv = pv;
    tw->tree = tree;
    pv->nanchor = nq;
    if(nq > 0)
        memset(pv->acc, 0, nq * pv->accsize);
    /* the anchors are live particles, distinct and ascending: with haswork NULL and an active
     * set the treewalk takes the list as its queue */
    const size_t lowbefore = allocator_get_min_free(A_MAIN);
    treewalk_run(tw, (int *) pv->anchor, nq);
    if(!A_MAIN->use_malloc) {
        const size_t lowwalk = allocator_get_min_free(A_MAIN);
        /* the tables' allocator charge, as treewalk.c's alloc_export_memory allocates them */
        const double elastic = (double) tw->NThread * (double) allocator_charge(A_MAIN, sizeof(data_index) * tw->BunchSize);
        if(lowwalk < SOWalkMinFree)
            SOWalkMinFree = lowwalk;
        if(elastic > SOWalkElastic)
            SOWalkElastic = elastic;
        /* the walk's low point without its export tables (they live through every iteration of
         * the walk, together with the walk's other buffers) */
        size_t adj = lowwalk + (size_t) elastic;
        if(adj > lowbefore)
            adj = lowbefore;
        A_MAIN->minfree = adj;
    }
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

/* Mass of type t in pass-1 bin k of the histogram M[NB][NT] (0 for a type without a slot). */
static inline double
so_m1(const double * M, const struct so_types * ty, int k, int t)
{
    return ty->slot[t] >= 0 ? M[(size_t) ty->NT * k + ty->slot[t]] : 0;
}

/* Mass of type t in pass-2 sub-bin m of the per-type sub-bins subT[NSUB][NT]. */
static inline double
so_m2(const double * subT, const struct so_types * ty, int m, int t)
{
    return ty->slot[t] >= 0 ? subT[(size_t) ty->NT * m + ty->slot[t]] : 0;
}

/* Pass-1 analysis of one row from its histogram M[NB][NT]. Returns 1 if the row must be
 * walked again with a doubled radius (radius already updated), 0 if its pass-1 outcome is final. */
static int
so_pass1_analyse(struct so_row * row, const double * M, int NB, const struct so_types * ty, const double rho[SUBHALO_NSO],
                 double rhomin, int MaxDoublings, double BoxSize, double soft)
{
    int j[SUBHALO_NSO];
    int d, k, t;
    for(d = 0; d < SUBHALO_NSO; d++)
        j[d] = -1;
    double C = 0, Dlast = 0;
    for(k = 0; k < NB; k++) {
        for(t = 0; t < SO_NT; t++)
            C += so_m1(M, ty, k, t);
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
            binm += so_m1(M, ty, k, t);
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
                    row->BinT[ts][t] = so_m1(M, ty, k, t);
                }
        }
        C += binm;
        for(t = 0; t < SO_NT; t++)
            CT[t] += so_m1(M, ty, k, t);
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

/* Final values of definition d of a row; sub/subT are the pass-2 sub-bins ([NSUB], [NSUB][NT]) or NULL. */
static void
so_finish_def(const struct so_row * row, int d, int NB, const struct so_types * ty, double rho, const double * sub,
              const double * subT, struct SubTrack * rec)
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
                        CaT[t] += so_m2(subT, ty, m, t);
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
                    CbT[t] = CaT[t] + (ts >= 0 ? so_m2(subT, ty, mb, t) : 0);
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
 * pass 2 writes the final values into the records, pass 3 the profile columns. Returns the number
 * of walks. */
static int
so_run_batches(int pass, struct so_priv * pv, int64_t * batchbuf, const int64_t * list, int64_t nlist, int64_t bcap,
               ForceTree * tree, const double rho[SUBHALO_NSO], double rhomin, const struct SubhaloParams * par,
               double BoxSize, double soft, MPI_Comm Comm)
{
    int64_t nbatch = (nlist > 0 && bcap > 0) ? (nlist + bcap - 1) / bcap : 0;
    /* cannot happen: rows exist only on ranks with a live particle (the others route theirs) */
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
        const int NT = pv->ty.NT;
        int64_t nprof = 0;
        #pragma omp parallel for reduction(+: nprof)
        for(s = 0; s < nq; s++) {
            struct so_row * row = &pv->rows[batchbuf[s]];
            if(pass == 1) {
                const double * M = (const double *) (pv->acc + s * pv->accsize);
                row->redo = so_pass1_analyse(row, M, NB, &pv->ty, rho, rhomin, par->SubhaloSOMaxDoublings, BoxSize, soft);
            }
            else if(pass == 3)
                nprof += so_profile_finish(row, (const struct so3_acc *) (pv->acc + s * pv->accsize), pv);
            else {
                const double * acc = (const double *) (pv->acc + s * pv->accsize);
                int d;
                for(d = 0; d < SUBHALO_NSO; d++) {
                    const int ts = so_typeslot(d);
                    const int refined = row->jc[d] >= 1 && row->jc[d] < NB;
                    if(!refined) {
                        so_finish_def(row, d, NB, &pv->ty, rho[d], NULL, NULL, row->rec);
                        continue;
                    }
                    if(ts < 0) {
                        so_finish_def(row, d, NB, &pv->ty, rho[d], acc + so2_uoff(d), NULL, row->rec);
                        continue;
                    }
                    /* the total sub-bins are the sums of the per-type ones */
                    const double * subT = acc + so2_toff(ts, NT);
                    double sub[SUBHALO_SO_NSUB];
                    int m, u;
                    for(m = 0; m < SUBHALO_SO_NSUB; m++) {
                        sub[m] = 0;
                        for(u = 0; u < NT; u++)
                            sub[m] += subT[(size_t) m * NT + u];
                    }
                    so_finish_def(row, d, NB, &pv->ty, rho[d], sub, subT, row->rec);
                }
            }
        }
        pv->nprofile += nprof;
    }
    return (int) nbatch_max;
}

/* No SO for this record (skipped call or no live particle anywhere). */
static void
so_clear_record(struct SubTrack * rec)
{
    int d;
    for(d = 0; d < SUBHALO_NSO; d++)
        rec->MSO[d] = rec->RSO[d] = -1;
    for(d = 0; d < SO_NT; d++)
        rec->MTypeSO200c[d] = rec->MTypeSOVir[d] = -1;
    rec->RsSO = rec->RsKlypinSO = rec->VmaxSO = rec->RmaxSO = -1;
}

/* ------------------------------------------------------------------ routing */

/* A central routed from a rank without a live particle to the host rank and back. */
struct so_guest {
    struct SubTrack rec;
    double Rsearch;
    int64_t idx;                   /* index in the origin's recs */
};

struct so_route {
    int host;                      /* lowest rank with a live particle; -1 none */
    int active;                    /* some rank routes its centrals */
    int * cnt;                     /* [4][NTask] bytes: send count, send offset, recv count, recv offset */
    struct so_guest * buf;         /* origin: the records sent; host: the records received */
    int64_t nbuf;
};

/* Collective. Ranks with centrals but no live particle send them to the host. */
static void
so_route_out(struct so_route * rt, struct SubTrack * recs, int64_t n, int64_t ncen, int64_t nlive, MPI_Comm Comm)
{
    int NTask, ThisTask, t;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    memset(rt, 0, sizeof(*rt));
    int64_t mine[2] = {nlive, ncen};
    int64_t * all = ta_malloc("SOrankinfo", int64_t, 2 * NTask);
    MPI_Allgather(mine, 2, MPI_INT64, all, 2, MPI_INT64, Comm);
    rt->host = -1;
    for(t = NTask - 1; t >= 0; t--)
        if(all[2 * t] > 0)
            rt->host = t;
    for(t = 0; t < NTask; t++)
        rt->active |= all[2 * t] == 0 && all[2 * t + 1] > 0;
    int64_t nrecv = 0;
    if(rt->host >= 0 && ThisTask == rt->host)
        for(t = 0; t < NTask; t++)
            if(all[2 * t] == 0)
                nrecv += all[2 * t + 1];
    ta_free(all);
    if(!rt->active || rt->host < 0)
        return;
    const int64_t nsend = (nlive == 0) ? ncen : 0;
    rt->nbuf = nsend + nrecv;
    rt->buf = (struct so_guest *) mymalloc("SOguest", so_imax(rt->nbuf, 1) * sizeof(struct so_guest));
    rt->cnt = (int *) mymalloc("SOroutecnt", 4 * NTask * sizeof(int));
    int * scnt = rt->cnt, * soff = rt->cnt + NTask, * rcnt = rt->cnt + 2 * NTask, * roff = rt->cnt + 3 * NTask;
    int64_t i, c = 0;
    for(i = 0; i < n && nsend > 0; i++) {
        if(!so_is_central(&recs[i]))
            continue;
        rt->buf[c].rec = recs[i];
        rt->buf[c].Rsearch = -1;
        rt->buf[c].idx = i;
        c++;
    }
    for(t = 0; t < NTask; t++)
        scnt[t] = soff[t] = 0;
    if(nsend > 0) {
        if((double) nsend * sizeof(struct so_guest) > 2e9)
            endrun(5, "subhalo_field_so: %ld centrals to route from a rank without particles\n", nsend);
        scnt[rt->host] = nsend * sizeof(struct so_guest);
    }
    MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, Comm);
    int64_t off = 0;
    for(t = 0; t < NTask; t++) {
        roff[t] = off;
        off += rcnt[t];
    }
    if(off != nrecv * (int64_t) sizeof(struct so_guest) || off > 2e9)
        endrun(5, "subhalo_field_so: routing receives %ld bytes, expected %ld\n", off, nrecv * (int64_t) sizeof(struct so_guest));
    /* an origin sends from buf and receives nothing; the host receives into buf */
    MPI_Alltoallv(nsend > 0 ? rt->buf : NULL, scnt, soff, MPI_BYTE, nrecv > 0 ? rt->buf : NULL, rcnt, roff, MPI_BYTE, Comm);
}

/* Collective. The host sends the routed records back; the origins copy the SO fields, the SO
 * flag bits and the search radius into their records. Frees the routing buffers. */
static void
so_route_back(struct so_route * rt, struct SubTrack * recs, double * Rsearch, MPI_Comm Comm)
{
    if(!rt->active || rt->host < 0)
        return;
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int * scnt = rt->cnt, * soff = rt->cnt + NTask, * rcnt = rt->cnt + 2 * NTask, * roff = rt->cnt + 3 * NTask;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    const int ishost = ThisTask == rt->host;
    /* the reverse of so_route_out: the host sends what it received */
    MPI_Alltoallv(ishost ? rt->buf : NULL, rcnt, roff, MPI_BYTE, ishost ? NULL : rt->buf, scnt, soff, MPI_BYTE, Comm);
    if(!ishost) {
        const int bits = SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS;
        int64_t g;
        for(g = 0; g < rt->nbuf; g++) {
            const struct so_guest * gu = &rt->buf[g];
            struct SubTrack * r = &recs[gu->idx];
            memcpy(r->MSO, gu->rec.MSO, sizeof(r->MSO));
            memcpy(r->RSO, gu->rec.RSO, sizeof(r->RSO));
            memcpy(r->MTypeSO200c, gu->rec.MTypeSO200c, sizeof(r->MTypeSO200c));
            memcpy(r->MTypeSOVir, gu->rec.MTypeSOVir, sizeof(r->MTypeSOVir));
            r->RsSO = gu->rec.RsSO;
            r->RsKlypinSO = gu->rec.RsKlypinSO;
            r->VmaxSO = gu->rec.VmaxSO;
            r->RmaxSO = gu->rec.RmaxSO;
            r->Flags = (r->Flags & ~bits) | (gu->rec.Flags & bits);
            if(Rsearch)
                Rsearch[gu->idx] = gu->Rsearch;
        }
    }
    myfree(rt->cnt);
    myfree(rt->buf);
}

/* ------------------------------------------------------------------ the call */

void
subhalo_field_so_full(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                      const Cosmology * CP, const struct SubhaloParams * par, const double * rho_in,
                      double * Rsearch, struct SubhaloSOStats * stats, MPI_Comm Comm)
{
    const double tstart = MPI_Wtime();
    const int NBpar = par->SubhaloSOBins;
    if(NBpar < SO_MINBINS || NBpar > SO_MAXBINS)
        endrun(1, "SubhaloSOBins = %d must be in [%d, %d].\n", NBpar, SO_MINBINS, SO_MAXBINS);
    if(!(par->SubhaloSOSearchFactor > 0) || par->SubhaloSOMaxDoublings < 0 || par->SubhaloSOMaxDoublings > 60)
        endrun(1, "SubhaloSOSearchFactor = %g must be positive and SubhaloSOMaxDoublings = %d in [0, 60].\n",
               par->SubhaloSOSearchFactor, par->SubhaloSOMaxDoublings);
    double rho[SUBHALO_NSO];
    int d, t;
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

    /* live particles and the types present among them (globally) */
    int64_t i, nlive = 0;
    int tmask = 0;
    #pragma omp parallel for reduction(+: nlive) reduction(|: tmask)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        nlive++;
        tmask |= 1 << P[i].Type;
    }
    MPI_Allreduce(MPI_IN_PLACE, &tmask, 1, MPI_INT, MPI_BOR, Comm);
    /* pass 3's fixed-point unit (subhalo_so_prof_unit, audit 5e R1): qscale = 2^s with
     * N_live m_max qscale <= 2^126, so that no 128-bit sum of quantised masses can overflow (N_live
     * and the largest mass are exact global reductions, so every rank and every layout picks the
     * same s). A mass m is rounded by at most 0.5 / (qscale m) relative, and not at all for
     * m >= 2^(bits(N_live) - 102) m_max (subhalo_so.h); the smallest positive mass is checked
     * against 1e-9 */
    double mmmg[2] = {0, -DBL_MAX};                /* the largest mass, minus the smallest positive */
    int64_t nliveg = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        if(P[i].Mass > mmmg[0])
            mmmg[0] = P[i].Mass;
        if(P[i].Mass > 0 && -P[i].Mass > mmmg[1])
            mmmg[1] = -P[i].Mass;
    }
    MPI_Allreduce(MPI_IN_PLACE, mmmg, 2, MPI_DOUBLE, MPI_MAX, Comm);
    MPI_Allreduce(&nlive, &nliveg, 1, MPI_INT64, MPI_SUM, Comm);
    const double mmaxg = mmmg[0], mming = -mmmg[1];
    const double qscale = subhalo_so_prof_unit(nliveg, mmaxg);
    if(mmaxg > 0 && 0.5 / (qscale * mming) > 1e-9)
        message(0, "WARNING: subhalo field SO pass 3: the smallest particle mass %g (largest %g, %ld live particles) is "
                   "quantised to %g relative in the fixed-point profile (unit %g).\n", mming, mmaxg, nliveg,
                0.5 / (qscale * mming), 1 / qscale);
    struct so_types ty = {0};
    for(t = 0; t < SO_NT; t++) {
        ty.slot[t] = -1;
        if(tmask & (1 << t)) {
            ty.type[ty.NT] = t;
            ty.slot[t] = ty.NT++;
        }
    }

    int64_t ncen = 0;
    for(i = 0; i < n; i++) {
        if(Rsearch)
            Rsearch[i] = -1;
        if(so_is_central(&recs[i])) {
            ncen++;
            recs[i].Flags &= ~(SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS);
            /* pass 3's columns: -1 unless pass 3 sets them */
            recs[i].RsSO = recs[i].RsKlypinSO = recs[i].VmaxSO = recs[i].RmaxSO = -1;
        }
    }

    /* the centrals of ranks without a live particle go to the host */
    struct so_route rt;
    so_route_out(&rt, recs, n, ncen, nlive, Comm);
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    const int64_t nguest = (rt.active && ThisTask == rt.host) ? rt.nbuf : 0;
    const int64_t nrows = nlive > 0 ? ncen + nguest : 0;
    if(rt.host < 0 && ncen > 0)
        message(1, "WARNING: subhalo field SO: no live particle on any rank, %ld centrals without SO.\n", ncen);
    if(rt.host < 0)
        for(i = 0; i < n; i++)
            if(so_is_central(&recs[i]))
                so_clear_record(&recs[i]);

    /* the rows (whole call), then the tree, then the batch buffers: freed in reverse */
    struct so_row * rows = (struct so_row *) mymalloc("SOrows", so_imax(nrows, 1) * sizeof(struct so_row));
    int64_t * list = (int64_t *) mymalloc("SOlist", so_imax(nrows, 1) * sizeof(int64_t));
    int64_t c = 0;
    /* own centrals first, then the guests */
    int64_t g;
    for(i = 0; i < n && nrows > 0; i++) {
        if(!so_is_central(&recs[i]))
            continue;
        rows[c].rec = &recs[i];
        rows[c].rsearch = Rsearch ? &Rsearch[i] : NULL;
        c++;
    }
    for(g = 0; g < nguest; g++) {
        rows[c].rec = &rt.buf[g].rec;
        rows[c].rsearch = &rt.buf[g].Rsearch;
        rows[c].rec->Flags &= ~(SUBHALO_FLAG_SOCUTOFF | SUBHALO_FLAG_SONOCROSS);
        c++;
    }
    for(c = 0; c < nrows; c++) {
        struct so_row * row = &rows[c];
        struct SubTrack * r = row->rec;
        double * rs = row->rsearch;
        memset(row, 0, sizeof(*row));
        row->rec = r;
        row->rsearch = rs;
        for(d = 0; d < 3; d++)
            row->Pos[d] = so_wrap(r->CorePos[d] + PartManager->CurrentParticleOffset[d], BoxSize);
        double R = par->SubhaloSOSearchFactor * cbrt(3. * r->Mbound / (4. * M_PI * rhomin));
        if(!(R > soft))
            R = soft > 0 ? soft : 1e-3 * BoxSize;
        if(R > 0.5 * BoxSize)
            R = 0.5 * BoxSize;
        row->R = R;
        row->Rlo = soft > 0 ? DMIN(soft, 0.25 * R) : 1e-3 * R;
        list[c] = c;
    }

    ForceTree tree = {0};
    force_tree_full(&tree, ddecomp, 0, NULL);

    /* the treewalk's reserve (ev_begin) against the free memory after the tree build: pass 1
     * halves its bins down to SO_SHRINKBINS until it fits on every rank, pass 2 runs only if it
     * fits, and nothing runs if pass 1 does not fit at SO_SHRINKBINS bins */
    const int64_t nthr = omp_get_max_threads();
    const int64_t freeb = (int64_t) mymalloc_freebytes();
    int64_t freemin = freeb;
    MPI_Allreduce(MPI_IN_PLACE, &freemin, 1, MPI_INT64, MPI_MIN, Comm);
    /* per query slot: index, anchor, the treewalk's queue copy and the accumulator */
    const int64_t slotfix = sizeof(int) + sizeof(int64_t) + nthr * sizeof(int);
    #define SO_NEED1(nb) ((int64_t) subhalo_so_walk_bytes(1, (nb), ty.NT) + slotfix + (int64_t) so_acc_bytes(1, (nb), ty.NT) + SO_SLACK)
    int NB = NBpar;
    while(NB > SO_SHRINKBINS && MPIU_Any(SO_NEED1(NB) > freeb, Comm))
        NB = (int) so_imax(NB / 2, SO_SHRINKBINS);
    const int skip = MPIU_Any(SO_NEED1(NB) > freeb, Comm);
    const int64_t need2 = (int64_t) subhalo_so_walk_bytes(2, NB, ty.NT) + slotfix
                        + (int64_t) so_imax(so_acc_bytes(1, NB, ty.NT), so_acc_bytes(2, NB, ty.NT)) + SO_SLACK;
    const int dopass2 = !skip && !MPIU_Any(need2 > freeb, Comm);
    if(NB != NBpar)
        message(0, "Subhalo field SO: %d pass-1 bins need %ld free bytes for the treewalk, the least free on a rank is %ld: "
                   "using %d bins for this call.\n", NBpar, SO_NEED1(NBpar), freemin, NB);
    if(skip)
        message(0, "WARNING: subhalo field SO SKIPPED for this call: pass 1 with %d bins needs %ld free bytes for the treewalk, "
                   "the least free on a rank is %ld. MSO, RSO and MTypeSO of every central are -1.\n", NB, SO_NEED1(NB), freemin);
    else if(!dopass2)
        message(0, "WARNING: subhalo field SO: pass 2 needs %ld free bytes for the treewalk, the least free on a rank is %ld: "
                   "no refinement this call (pass-1 interpolation for every central).\n", need2, freemin);
    /* pass 3 (the virial-sphere profile, decision 44) runs only if it fits; otherwise its four
     * columns stay -1 for this call and nothing else changes */
    const int64_t need3 = (int64_t) subhalo_so_walk_bytes(3, NB, ty.NT) + slotfix
                        + (int64_t) so_imax(so_acc_bytes(1, NB, ty.NT), so_acc_bytes(3, NB, ty.NT)) + SO_SLACK;
    const int dopass3 = !skip && !MPIU_Any(need3 > freeb, Comm);
    if(!skip && !dopass3)
        message(0, "WARNING: subhalo field SO: pass 3 needs %ld free bytes for the treewalk, the least free on a rank is %ld: "
                   "RsSO, RsKlypinSO, VmaxSO and RmaxSO are -1 this call.\n", need3, freemin);
    int64_t needmax = dopass2 ? need2 : SO_NEED1(NB);
    if(dopass3 && need3 > needmax)
        needmax = need3;
    #undef SO_NEED1

    for(c = 0; c < nrows; c++)
        rows[c].Scale = NB / log(rows[c].R / rows[c].Rlo);

    /* batch capacity: live anchors, the test knob, 1/8 of the free memory and what leaves the
     * treewalk its reserve */
    const size_t acc1 = so_acc_bytes(1, NB, ty.NT);
    const size_t acc2 = so_acc_bytes(2, NB, ty.NT);
    const size_t acc3 = so_acc_bytes(3, NB, ty.NT);
    size_t accsize = dopass2 && acc2 > acc1 ? acc2 : acc1;
    if(dopass3 && acc3 > accsize)
        accsize = acc3;
    const int64_t slotbytes = accsize + slotfix;
    int64_t bcap = skip ? 0 : nrows;
    if(SOMaxBatch > 0 && bcap > SOMaxBatch)
        bcap = SOMaxBatch;
    const int64_t avail = so_imin(freeb / 8, freeb - needmax);
    const int64_t memcap = so_imax(avail / slotbytes, 1);
    if(bcap > memcap)
        bcap = memcap;
    int * anchor = (int *) mymalloc("SOanchor", so_imax(bcap, 1) * sizeof(int));
    int64_t na = 0;
    for(i = 0; i < PartManager->NumPart && na < bcap; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        anchor[na++] = (int) i;
    }
    bcap = na;
    int64_t * batchbuf = (int64_t *) mymalloc("SObatch", so_imax(bcap, 1) * sizeof(int64_t));
    char * acc = (char *) mymalloc("SOacc", so_imax(bcap, 1) * accsize);

    struct so_priv pv = {0};
    pv.rows = rows;
    pv.batch = batchbuf;
    pv.anchor = anchor;
    pv.acc = acc;
    pv.NB = NB;
    pv.ty = ty;
    pv.qscale = qscale;
    pv.pscale = SUBHALO_SO_NPROF / log(1. / SUBHALO_SO_PROF_RIN);
    pv.G = CP->GravInternal;
    pv.atime = atime;
    pv.vu = subhalo_vel_to_kms(1.0);
    pv.soft = soft;
    pv.rhovir = rho[SO_DEF_VIR];

    /* pass 1 with the doubling rounds */
    int npass1 = 0, npass2 = 0, npass3 = 0;
    int64_t npend = skip ? 0 : nrows;
    pv.accsize = acc1;
    while(1) {
        if(!MPIU_Any(npend > 0, Comm))
            break;
        npass1 += so_run_batches(1, &pv, batchbuf, list, npend, bcap, &tree, rho, rhomin, par, BoxSize, soft, Comm);
        int64_t k, m = 0;
        for(k = 0; k < npend; k++)
            if(rows[list[k]].redo)
                list[m++] = list[k];
        npend = m;
    }

    /* pass 2 for the large centrals with a crossing bin; the others finish here */
    int64_t nref = 0;
    for(c = 0; c < nrows; c++) {
        struct so_row * row = &rows[c];
        struct SubTrack * rec = row->rec;
        if(skip) {
            so_clear_record(rec);
            continue;
        }
        int any = 0;
        for(d = 0; d < SUBHALO_NSO; d++)
            any |= row->jc[d] >= 1 && row->jc[d] < NB;
        row->refine = dopass2 && any && rec->Nbound >= par->SubhaloSORefineMinPart;
        if(row->refine)
            list[nref++] = c;
        else
            for(d = 0; d < SUBHALO_NSO; d++)
                so_finish_def(row, d, NB, &ty, rho[d], NULL, NULL, rec);
    }
    pv.accsize = acc2;
    if(dopass2)
        npass2 = so_run_batches(2, &pv, batchbuf, list, nref, bcap, &tree, rho, rhomin, par, BoxSize, soft, Comm);

    /* pass 3: the profile inside the published RSO[vir] of every row with a virial result */
    int64_t nprof = 0;
    for(c = 0; c < nrows && dopass3; c++)
        if(rows[c].rec->MSO[SO_DEF_VIR] > 0 && rows[c].rec->RSO[SO_DEF_VIR] > 0)
            list[nprof++] = c;
    pv.accsize = acc3;
    if(dopass3)
        npass3 = so_run_batches(3, &pv, batchbuf, list, nprof, bcap, &tree, rho, rhomin, par, BoxSize, soft, Comm);
    /* the rows with an edge above rho[vir] (VmaxSO set); pv.nprofile counts those with enough
     * particles, some of which may have none (audit 5e R2) */
    int64_t nprofdef = 0;
    for(c = 0; c < nprof; c++)
        nprofdef += rows[list[c]].rec->VmaxSO > 0;

    myfree(acc);
    myfree(batchbuf);
    myfree(anchor);
    force_tree_free(&tree);

    /* flags and diagnostics */
    int64_t cnt[5] = {nrows, nref, 0, 0, 0};
    for(c = 0; c < nrows && !skip; c++) {
        const struct so_row * row = &rows[c];
        struct SubTrack * rec = row->rec;
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
        if(row->rsearch)
            *row->rsearch = row->R;
    }
    myfree(list);
    myfree(rows);
    so_route_back(&rt, recs, Rsearch, Comm);

    int64_t tot[9];
    int64_t cnt6[9] = {cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], nguest, nprof, pv.nprofile, nprofdef};
    MPI_Allreduce(cnt6, tot, 9, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo field SO: %ld centrals (%ld routed from ranks without particles), %ld refined, %ld doubled, "
               "%ld cut off at the cap, %ld without a crossing; %d bins, %d types; %d + %d walks, %.3g s.\n",
               tot[0], tot[5], tot[1], tot[2], tot[3], tot[4], NB, ty.NT, npass1, npass2, MPI_Wtime() - tstart);
    if(dopass3)
        message(0, "Subhalo field SO pass 3: %ld centrals profiled inside RSO[vir], %ld with >= %d particles, %ld of them "
                   "with a profile edge above rho[vir] (VmaxSO, RsSO); %d walks.\n", tot[6], tot[7], SUBHALO_NFW_MINPART,
                tot[8], npass3);
    if(stats) {
        stats->Ncentral = tot[0];
        stats->Nrefined = tot[1];
        stats->Ndoubled = tot[2];
        stats->Ncutoff = tot[3];
        stats->Nnocross = tot[4];
        stats->Nrouted = tot[5];
        stats->Npass1Walks = npass1;
        stats->Npass2Walks = npass2;
        stats->NBins = NB;
        stats->NTypes = ty.NT;
        stats->Pass2 = dopass2;
        stats->Pass3 = dopass3;
        stats->Nprofiled = tot[6];
        stats->Nprofile = tot[7];
        stats->NprofileVmax = tot[8];
        stats->ProfUnit = dopass3 ? qscale : 0;
        stats->Npass3Walks = npass3;
        stats->Skipped = skip;
        stats->FreeBytes = freemin;
    }
}

void
subhalo_field_so(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                 const Cosmology * CP, const struct SubhaloParams * par, MPI_Comm Comm)
{
    subhalo_field_so_full(recs, n, ddecomp, atime, CP, par, NULL, NULL, NULL, Comm);
}
