/* Unbinding kernel of the subhalo finder (design 7.4): the octree, energies, the HBT-HERONS
 * unbinding loop, the final pass, core, sinking test and k-nearest tracers, ported from HBT's
 * gravity_tree.cpp, subhalo_unbind.cpp, subhalo_merge.cpp and subhalo_reassign_gas.cpp.
 * Contract and conventions in subhalo_unbind.h. */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>

#include "utils/endrun.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "physconst.h"
/* gravshort.h defines static treewalk callbacks this file does not use */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "gravshort.h"
#pragma GCC diagnostic pop
#include "subhalo_unbind.h"

/* ------------------------------------------------------------------ small utilities */

/* Arena allocation that must succeed: the arena is sized by SUBHALO_UNBIND_BYTES_PER_PART. */
static void *
arena_need(struct SubhaloArena * a, size_t bytes, const char * what)
{
    void * p = subhalo_arena_get(a, bytes);
    if(!p)
        endrun(1, "Subhalo kernel: arena exhausted allocating %s (%lu bytes, %lu of %lu used)\n",
               what, (unsigned long) bytes, (unsigned long) a->used, (unsigned long) a->size);
    return p;
}

/* Bytes still available in the arena (after alignment padding of one more allocation). */
static size_t
arena_left(const struct SubhaloArena * a)
{
    return a->size > a->used + 32 ? a->size - a->used - 32 : 0;
}

/* ------------------------------------------------------------------ intra-key threading */

/* Parallel mode (subhalo_unbind.h): an operation on more than par_min elements, outside a
 * parallel region with several threads available, splits over the threads. The algorithm of each
 * step depends on the size only, so no result depends on the thread count or on par_min. */
static int64_t par_min = SUBHALO_UNBIND_PAR_MIN;

void
subhalo_unbind_set_par_min(int64_t n)
{
    par_min = n > 0 ? n : SUBHALO_UNBIND_PAR_MIN;
}

int64_t
subhalo_unbind_get_par_min(void)
{
    return par_min;
}

static inline int
par_mode(int64_t n)
{
    return n > par_min && !omp_in_parallel() && omp_get_max_threads() > 1;
}

/* The per-element loops (energies, neighbour queries: each element independent, any split
 * gives the same result) thread above min(SUBHALO_LOOP_PAR_MIN, par_min) elements. */
#define SUBHALO_LOOP_PAR_MIN 4096
static inline int
loop_par(int64_t n)
{
    return n > (par_min < SUBHALO_LOOP_PAR_MIN ? par_min : SUBHALO_LOOP_PAR_MIN) && !omp_in_parallel();
}

/* The dynamic chunk of those loops: n / (4 threads) clamped to [16, 256], so a few thousand
 * elements still feed every thread; 256 with loop_chunk_adaptive cleared. Scheduling only. */
static int loop_chunk_adaptive = 1;

void
subhalo_unbind_set_loop_chunk(int adaptive)
{
    loop_chunk_adaptive = adaptive;
}

static inline int
loop_chunk(int64_t n)
{
    if(!loop_chunk_adaptive)
        return 256;
    int64_t c = n / (4 * (int64_t) omp_get_max_threads());
    return c < 16 ? 16 : c > 256 ? 256 : (int) c;
}

/* The fixed chunks of the reductions and order-preserving gathers: max(CHUNK_MIN, ceil(n /
 * CHUNK_MAX)) elements, one chunk for n <= CHUNK_MIN. They define the rounding of the sums. */
#define SUBHALO_CHUNK_MAX 256
static inline int64_t
chunk_len(int64_t n)
{
    const int64_t c = (n + SUBHALO_CHUNK_MAX - 1) / SUBHALO_CHUNK_MAX;
    return c > SUBHALO_UNBIND_CHUNK_MIN ? c : SUBHALO_UNBIND_CHUNK_MIN;
}

static inline int
chunk_count(int64_t n, int64_t C)
{
    return n > 0 ? (int) ((n + C - 1) / C) : 0;
}

/* for(I = 0; I < N; I++) BODY, split statically over the threads when PAR. Two loops rather than
 * an if() clause, so that the serial path opens no parallel region at all. */
#define SUBHALO_PAR_FOR(PAR, I, N, ...)                                                 \
    do {                                                                                \
        if(PAR) {                                                                       \
            _Pragma("omp parallel for schedule(static)")                               \
            for(I = 0; I < (N); I++) { __VA_ARGS__ }                                    \
        }                                                                               \
        else                                                                            \
            for(I = 0; I < (N); I++) { __VA_ARGS__ }                                    \
    } while(0)

/* splitmix64: the seeded generator of design D3 item 3, in place of HBT's std::random_shuffle,
 * so results do not depend on the thread or rank count. */
static inline uint64_t
splitmix64(uint64_t * s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniform integer in [0, n) (multiply-high; bias below 2^-64 n). */
static inline uint64_t
rng_below(uint64_t * s, uint64_t n)
{
    return (uint64_t) (((unsigned __int128) splitmix64(s) * n) >> 64);
}

static inline int
type_in(int mask, int type)
{
    return (mask >> type) & 1;
}

static inline double
wrap_box(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* The tie order of design 7.10: particle ID, then (defensively) the P index. */
static inline int
id_less(int64_t a, int64_t b)
{
    return P[a].ID < P[b].ID || (P[a].ID == P[b].ID && a < b);
}

/* In-place introsort, allocation-free and thread-safe. Every comparator below is a strict total
 * order (ties by ID and P index), so the result is unique whatever the algorithm or order of P. */
#define SUBHALO_DEFINE_SORT(NAME, T, LESS)                                              \
static void NAME##_isort(T * a, int64_t n) {                                            \
    int64_t i, j;                                                                       \
    for(i = 1; i < n; i++) {                                                            \
        T x = a[i];                                                                     \
        for(j = i; j > 0 && LESS((&x), (&a[j - 1])); j--)                               \
            a[j] = a[j - 1];                                                            \
        a[j] = x;                                                                       \
    }                                                                                   \
}                                                                                       \
static void NAME##_sift(T * a, int64_t root, int64_t n) {                               \
    for(;;) {                                                                           \
        int64_t child = 2 * root + 1;                                                   \
        if(child >= n) return;                                                          \
        if(child + 1 < n && LESS((&a[child]), (&a[child + 1]))) child++;                \
        if(!LESS((&a[root]), (&a[child]))) return;                                      \
        T t = a[root]; a[root] = a[child]; a[child] = t;                                \
        root = child;                                                                   \
    }                                                                                   \
}                                                                                       \
static void NAME##_hsort(T * a, int64_t n) {                                            \
    int64_t i;                                                                          \
    for(i = n / 2 - 1; i >= 0; i--) NAME##_sift(a, i, n);                               \
    for(i = n - 1; i > 0; i--) {                                                        \
        T t = a[0]; a[0] = a[i]; a[i] = t;                                              \
        NAME##_sift(a, 0, i);                                                           \
    }                                                                                   \
}                                                                                       \
static void NAME##_intro(T * a, int64_t n, int depth) {                                 \
    while(n > 16) {                                                                     \
        if(depth-- <= 0) { NAME##_hsort(a, n); return; }                                \
        const int64_t mid = n / 2;                                                      \
        T t;                                                                            \
        if(LESS((&a[mid]), (&a[0]))) { t = a[mid]; a[mid] = a[0]; a[0] = t; }           \
        if(LESS((&a[n - 1]), (&a[0]))) { t = a[n - 1]; a[n - 1] = a[0]; a[0] = t; }     \
        if(LESS((&a[n - 1]), (&a[mid]))) { t = a[n - 1]; a[n - 1] = a[mid]; a[mid] = t; } \
        const T pivot = a[mid];                                                         \
        int64_t i = -1, j = n;                                                          \
        for(;;) {                                                                       \
            do i++; while(LESS((&a[i]), (&pivot)));                                     \
            do j--; while(LESS((&pivot), (&a[j])));                                     \
            if(i >= j) break;                                                           \
            t = a[i]; a[i] = a[j]; a[j] = t;                                            \
        }                                                                               \
        /* [0, j] and [j + 1, n), both non-empty; recurse on the smaller */             \
        if(j + 1 < n - j - 1) { NAME##_intro(a, j + 1, depth); a += j + 1; n -= j + 1; } \
        else { NAME##_intro(a + j + 1, n - j - 1, depth); n = j + 1; }                  \
    }                                                                                   \
    NAME##_isort(a, n);                                                                 \
}                                                                                       \
static void NAME(T * a, int64_t n) {                                                    \
    int depth = 2;                                                                      \
    int64_t m = n;                                                                      \
    while(m > 1) { m >>= 1; depth += 2; }                                               \
    NAME##_intro(a, n, depth);                                                          \
}

/* Parallel merge sort with NAME's unique output: one introsorted block per thread, then pairwise
 * merge-path merges. The merge buffer comes from the arena; without room, NAME runs serially. */
#define SUBHALO_DEFINE_PSORT(NAME, T, LESS)                                             \
static int64_t NAME##_corank(const T * A, int64_t m, const T * B, int64_t nb, int64_t k) { \
    /* the number i of A elements among the first k outputs of the merge that takes A first \
     * on ties: the smallest i with A[i] after B[k - i - 1] */                         \
    int64_t lo = k > nb ? k - nb : 0, hi = k < m ? k : m;                               \
    while(lo < hi) {                                                                    \
        const int64_t i = lo + (hi - lo) / 2, j = k - i;                                \
        if(j > 0 && !LESS((&B[j - 1]), (&A[i]))) lo = i + 1;                            \
        else hi = i;                                                                    \
    }                                                                                   \
    return lo;                                                                          \
}                                                                                       \
static void NAME##_merge(const T * A, int64_t m, const T * B, int64_t nb, T * out,      \
                         int64_t k0, int64_t k1) {                                      \
    int64_t i = NAME##_corank(A, m, B, nb, k0), j = k0 - i, k = k0;                     \
    const int64_t i1 = NAME##_corank(A, m, B, nb, k1), j1 = k1 - i1;                    \
    while(i < i1 && j < j1) {                                                           \
        if(LESS((&B[j]), (&A[i]))) out[k++] = B[j++];                                   \
        else out[k++] = A[i++];                                                         \
    }                                                                                   \
    while(i < i1) out[k++] = A[i++];                                                    \
    while(j < j1) out[k++] = B[j++];                                                    \
}                                                                                       \
static void NAME##_par(T * a, int64_t n, struct SubhaloArena * arena) {                 \
    if(!par_mode(n) || arena_left(arena) < sizeof(T) * (size_t) n) {                    \
        NAME(a, n);                                                                     \
        return;                                                                         \
    }                                                                                   \
    const size_t mark = arena->used;                                                    \
    T * buf = (T *) arena_need(arena, sizeof(T) * n, #NAME " merge buffer");            \
    const int nblk = omp_get_max_threads();                                             \
    int blk, w;                                                                         \
    _Pragma("omp parallel for schedule(static, 1)")                                    \
    for(blk = 0; blk < nblk; blk++) {                                                   \
        const int64_t lo = (int64_t) blk * n / nblk, hi = (int64_t) (blk + 1) * n / nblk; \
        NAME(a + lo, hi - lo);                                                          \
    }                                                                                   \
    T * from = a, * to = buf;                                                           \
    for(w = 1; w < nblk; w *= 2) {                                                      \
        _Pragma("omp parallel")                                                         \
        {                                                                               \
            const int tid = omp_get_thread_num(), nth = omp_get_num_threads();          \
            const int64_t g0 = (int64_t) tid * n / nth, g1 = (int64_t) (tid + 1) * n / nth; \
            int k;                                                                      \
            for(k = 0; k < nblk; k += 2 * w) {                                          \
                const int km = k + w < nblk ? k + w : nblk;                             \
                const int ke = k + 2 * w < nblk ? k + 2 * w : nblk;                     \
                const int64_t p0 = (int64_t) k * n / nblk, pm = (int64_t) km * n / nblk; \
                const int64_t p1 = (int64_t) ke * n / nblk;                             \
                const int64_t s0 = g0 > p0 ? g0 : p0, s1 = g1 < p1 ? g1 : p1;           \
                if(s0 < s1)                                                             \
                    NAME##_merge(from + p0, pm - p0, from + pm, p1 - pm, to + p0, s0 - p0, s1 - p0); \
            }                                                                           \
        }                                                                               \
        T * tmp = from; from = to; to = tmp;                                            \
    }                                                                                   \
    if(from != a) {                                                                     \
        int64_t i;                                                                      \
        _Pragma("omp parallel for schedule(static)")                                   \
        for(i = 0; i < n; i++)                                                          \
            a[i] = from[i];                                                             \
    }                                                                                   \
    arena->used = mark;                                                                 \
}

/* (energy, P index) pairs: the energy order of design 7.10, ties by ID. */
struct EPair {
    double E;
    int64_t idx;
};
#define EPAIR_LESS(x, y) ((x)->E < (y)->E || ((x)->E == (y)->E && id_less((x)->idx, (y)->idx)))
SUBHALO_DEFINE_SORT(sort_epair, struct EPair, EPAIR_LESS)
SUBHALO_DEFINE_PSORT(sort_epair, struct EPair, EPAIR_LESS)

/* (energy, P index, list position) triples of the most-bound refinement: ordered as EPair, the
 * position carried along to permute the parallel arrays. */
struct EPairPos {
    double E;
    int64_t idx;
    int64_t pos;
};
SUBHALO_DEFINE_SORT(sort_epairpos, struct EPairPos, EPAIR_LESS)
SUBHALO_DEFINE_PSORT(sort_epairpos, struct EPairPos, EPAIR_LESS)

/* A gravity or neighbour source: Morton key, P index, position (comoving; relative to the
 * tree origin, NEAREST(Pos - origin), and absolute in a built neighbour tree), mass (upscaled
 * when subsampled). 48 bytes. Equal keys are ordered by ID. */
struct SubSrc {
    uint64_t key;
    int64_t p;
    double x[3];
    double m;
};
#define SRC_LESS(x, y) ((x)->key < (y)->key || ((x)->key == (y)->key && id_less((x)->p, (y)->p)))
SUBHALO_DEFINE_SORT(sort_src, struct SubSrc, SRC_LESS)
SUBHALO_DEFINE_PSORT(sort_src, struct SubSrc, SRC_LESS)

/* Sort the parallel arrays (Energy, Index)[0, n) by (Energy, Index), through a pair scratch. */
static void
sort_energy_index(double * Energy, int64_t * Index, int64_t n, struct SubhaloArena * arena)
{
    if(n < 2)
        return;
    const size_t mark = arena->used;
    struct EPair * pr = (struct EPair *) arena_need(arena, sizeof(struct EPair) * n, "energy sort pairs");
    const int par = par_mode(n);
    int64_t i;
    SUBHALO_PAR_FOR(par, i, n, pr[i].E = Energy[i]; pr[i].idx = Index[i];);
    sort_epair_par(pr, n, arena);
    SUBHALO_PAR_FOR(par, i, n, Energy[i] = pr[i].E; Index[i] = pr[i].idx;);
    arena->used = mark;
}

/* ------------------------------------------------------------------ octree */

/* Morton-ordered octree in pre-order: the first child of node n is n + 1, the next sibling its
 * skip, a leaf has skip == n + 1; a node covers sources [first, first + count). Single-child
 * chains are collapsed. HBT's GravityTree_t, built from sorted keys instead of insertion. */
#define MORTON_LEVELS 21

struct SubSkel {
    int32_t skip, first, count;
    float len;               /* cell side, comoving */
};
struct SubMom {
    double com[3];
    double mass;
};
struct SubBox {
    double lo[3], hi[3];
};

struct SubTree {
    struct SubSrc * src;
    int64_t ns;
    struct SubSkel * sk;
    int64_t nn;
    struct SubMom * mom;     /* gravity trees */
    struct SubBox * box;     /* neighbour trees */
    double BoxSize;          /* every displacement is NEAREST(dx, BoxSize) ... */
    int wrap;                /* ... unless 0: no pair can be farther apart than BoxSize / 2 along
                                an axis, so NEAREST is the identity and is skipped */
};

static inline uint64_t
spread3(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}

static inline int
morton_digit(uint64_t key, int level)
{
    return (int) ((key >> (3 * (MORTON_LEVELS - 1 - level))) & 7);
}

/* Recursive build over src[lo, hi) whose keys share the top 3 * level bits; sk == NULL counts
 * nodes only (*nn is the running count). */
static void
tree_build_rec(const struct SubSrc * src, struct SubSkel * sk, int64_t * nn, int64_t lo, int64_t hi,
               int level, int64_t leafsize, double rootlen)
{
    /* collapse the levels where every source shares the digit */
    while(level < MORTON_LEVELS && hi - lo > leafsize
          && morton_digit(src[lo].key, level) == morton_digit(src[hi - 1].key, level))
        level++;
    const int64_t node = (*nn)++;
    if(sk) {
        sk[node].first = (int32_t) lo;
        sk[node].count = (int32_t) (hi - lo);
        sk[node].len = (float) ldexp(rootlen, -level);
    }
    if(hi - lo > leafsize && level < MORTON_LEVELS) {
        int64_t s = lo;
        while(s < hi) {
            const int d = morton_digit(src[s].key, level);
            int64_t e = s + 1;
            while(e < hi && morton_digit(src[e].key, level) == d)
                e++;
            tree_build_rec(src, sk, nn, s, e, level + 1, leafsize, rootlen);
            s = e;
        }
    }
    if(sk)
        sk[node].skip = (int32_t) *nn;
}

/* Parallel build: "top" nodes built serially, subtree tasks of at most T sources by
 * tree_build_rec in parallel, placed by a prefix sum of their node counts, so every node lands
 * where the serial build puts it: the tree is bit-identical whatever T and the thread count. */
struct TopItem {
    int64_t lo, hi;          /* source range */
    int64_t nodes;           /* nodes of a task's subtree; 1 for a top node */
    int64_t off;             /* index of the item's first node */
    int32_t level;           /* task: tree_build_rec's level argument; top node: its collapsed level */
    int32_t end;             /* top node: index of the first item after its subtree; task: -1 */
};
#define SUBHALO_TOP_MAXITEMS 1024
struct TopPlan {
    int n;
    struct TopItem it[SUBHALO_TOP_MAXITEMS];
};

/* First index in [s, hi) whose digit at level exceeds d (the keys of [s, hi) share their digits
 * above level and are sorted, so the digits at level are non-decreasing). */
static int64_t
digit_run_end(const struct SubSrc * src, int64_t s, int64_t hi, int level, int d)
{
    int64_t lo = s + 1;
    while(lo < hi) {
        const int64_t m = lo + (hi - lo) / 2;
        if(morton_digit(src[m].key, level) > d)
            hi = m;
        else
            lo = m + 1;
    }
    return lo;
}

/* Returns 0 if the plan overflows SUBHALO_TOP_MAXITEMS. */
static int
plan_rec(const struct SubSrc * src, struct TopPlan * plan, int64_t lo, int64_t hi, int level, int64_t leafsize, int64_t T)
{
    if(plan->n >= SUBHALO_TOP_MAXITEMS)
        return 0;
    int lev = level;
    if(hi - lo > T && hi - lo > leafsize)
        while(lev < MORTON_LEVELS && morton_digit(src[lo].key, lev) == morton_digit(src[hi - 1].key, lev))
            lev++;
    struct TopItem * it = &plan->it[plan->n];
    it->lo = lo;
    it->hi = hi;
    if(hi - lo <= T || hi - lo <= leafsize || lev >= MORTON_LEVELS) {
        /* a task: tree_build_rec(lo, hi, level) builds this subtree exactly as the serial build */
        it->level = level;
        it->end = -1;
        it->nodes = 0;
        plan->n++;
        return 1;
    }
    const int k = plan->n++;
    it->level = lev;
    it->nodes = 1;
    int64_t s = lo;
    while(s < hi) {
        const int64_t e = digit_run_end(src, s, hi, lev, morton_digit(src[s].key, lev));
        if(!plan_rec(src, plan, s, e, lev + 1, leafsize, T))
            return 0;
        s = e;
    }
    plan->it[k].end = plan->n;
    return 1;
}

/* The plan for this leafsize (T doubled until it fits; T >= ns gives one task) and the node
 * count of the tree, the task counts in parallel. */
static int64_t
plan_make(const struct SubSrc * src, int64_t ns, int64_t leafsize, struct TopPlan * plan)
{
    int64_t T = ns / (8 * (int64_t) omp_get_max_threads());
    if(T < 1024)
        T = 1024;
    for(;;) {
        plan->n = 0;
        if(plan_rec(src, plan, 0, ns, 0, leafsize, T))
            break;
        T *= 2;
    }
    int k;
    #pragma omp parallel for schedule(dynamic, 1)
    for(k = 0; k < plan->n; k++) {
        struct TopItem * it = &plan->it[k];
        if(it->end < 0) {
            int64_t c = 0;
            tree_build_rec(src, NULL, &c, it->lo, it->hi, it->level, leafsize, 0);
            it->nodes = c;
        }
    }
    int64_t nn = 0;
    for(k = 0; k < plan->n; k++) {
        plan->it[k].off = nn;
        nn += plan->it[k].nodes;
    }
    return nn;
}

static void
plan_build(const struct SubSrc * src, struct SubSkel * sk, int64_t nn, int64_t leafsize, double rootlen, const struct TopPlan * plan)
{
    int k;
    for(k = 0; k < plan->n; k++) {
        const struct TopItem * it = &plan->it[k];
        if(it->end < 0)
            continue;
        sk[it->off].first = (int32_t) it->lo;
        sk[it->off].count = (int32_t) (it->hi - it->lo);
        sk[it->off].len = (float) ldexp(rootlen, -it->level);
        sk[it->off].skip = (int32_t) (it->end < plan->n ? plan->it[it->end].off : nn);
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for(k = 0; k < plan->n; k++) {
        const struct TopItem * it = &plan->it[k];
        if(it->end < 0) {
            int64_t c = it->off;
            tree_build_rec(src, sk, &c, it->lo, it->hi, it->level, leafsize, rootlen);
            if(c != it->off + it->nodes)
                endrun(1, "Subhalo kernel: parallel tree task built %ld nodes, counted %ld\n", (long) (c - it->off), (long) it->nodes);
        }
    }
}

/* Componentwise min and max of the source positions (exact in any order). */
static void
src_extent(const struct SubSrc * src, int64_t ns, int par, double lo[3], double hi[3])
{
    int d;
    for(d = 0; d < 3; d++) {
        lo[d] = DBL_MAX;
        hi[d] = -DBL_MAX;
    }
    if(!par) {
        int64_t i;
        for(i = 0; i < ns; i++)
            for(d = 0; d < 3; d++) {
                if(src[i].x[d] < lo[d]) lo[d] = src[i].x[d];
                if(src[i].x[d] > hi[d]) hi[d] = src[i].x[d];
            }
        return;
    }
    #pragma omp parallel
    {
        double l[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, h[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        int64_t i;
        int dd;
        #pragma omp for schedule(static)
        for(i = 0; i < ns; i++)
            for(dd = 0; dd < 3; dd++) {
                if(src[i].x[dd] < l[dd]) l[dd] = src[i].x[dd];
                if(src[i].x[dd] > h[dd]) h[dd] = src[i].x[dd];
            }
        #pragma omp critical(subhalo_src_extent)
        for(dd = 0; dd < 3; dd++) {
            if(l[dd] < lo[dd]) lo[dd] = l[dd];
            if(h[dd] > hi[dd]) hi[dd] = h[dd];
        }
    }
}

/* Skeleton over t->src[0, t->ns) (positions relative to the origin): Morton keys, sort, then the
 * nodes, the leaves widened until they fit in the arena. plan != NULL: built in parallel, the
 * decomposition left in plan for the moments; the same tree. */
static void
tree_build_skeleton(struct SubTree * t, size_t node_extra, struct SubhaloArena * arena, struct TopPlan * plan)
{
    int64_t i;
    int d;
    const int64_t ns = t->ns;
    const int par = plan != NULL;
    if(ns >= INT32_MAX)
        endrun(1, "Subhalo kernel: %ld sources exceed the int32 tree layout\n", (long) ns);
    double lo[3], hi[3];
    if(par)
        src_extent(t->src, ns, 1, lo, hi);
    else {
        for(d = 0; d < 3; d++) {
            lo[d] = DBL_MAX;
            hi[d] = -DBL_MAX;
        }
        for(i = 0; i < ns; i++)
            for(d = 0; d < 3; d++) {
                if(t->src[i].x[d] < lo[d]) lo[d] = t->src[i].x[d];
                if(t->src[i].x[d] > hi[d]) hi[d] = t->src[i].x[d];
            }
    }
    double ext = 0;
    for(d = 0; d < 3; d++)
        if(hi[d] - lo[d] > ext)
            ext = hi[d] - lo[d];
    double rootlen = ext * (1 + 1e-9);
    if(rootlen <= 0)
        rootlen = 1e-10 * (fabs(lo[0]) + fabs(lo[1]) + fabs(lo[2]) + 1);
    const double scale = (double) (1 << MORTON_LEVELS) / rootlen;
    struct SubSrc * src = t->src;
    SUBHALO_PAR_FOR(par, i, ns,
        uint64_t key = 0;
        int dd;
        for(dd = 0; dd < 3; dd++) {
            double q = (src[i].x[dd] - lo[dd]) * scale;
            int64_t iq = (int64_t) q;
            if(iq < 0) iq = 0;
            if(iq > (1 << MORTON_LEVELS) - 1) iq = (1 << MORTON_LEVELS) - 1;
            key |= spread3((uint64_t) iq) << (2 - dd);
        }
        src[i].key = key;
    );
    /* the merge buffer is released before the node budget is taken: the same cap either way */
    sort_src_par(t->src, ns, arena);

    /* node budget from what is left; widen leaves until the tree fits */
    const size_t per_node = sizeof(struct SubSkel) + node_extra;
    int64_t cap = (int64_t) (arena_left(arena) / per_node);
    if(cap > INT32_MAX - 1)
        cap = INT32_MAX - 1;
    if(cap < 1)
        endrun(1, "Subhalo kernel: no arena space left for tree nodes (%ld sources)\n", (long) ns);
    int64_t leafsize = 8, nn;
    for(;;) {
        if(par)
            nn = plan_make(t->src, ns, leafsize, plan);
        else {
            nn = 0;
            tree_build_rec(t->src, NULL, &nn, 0, ns, 0, leafsize, rootlen);
        }
        if(nn <= cap)
            break;
        leafsize *= 2;
    }
    t->sk = (struct SubSkel *) arena_need(arena, sizeof(struct SubSkel) * nn, "tree skeleton");
    if(par) {
        plan_build(t->src, t->sk, nn, leafsize, rootlen, plan);
        t->nn = nn;
    }
    else {
        t->nn = 0;
        tree_build_rec(t->src, t->sk, &t->nn, 0, ns, 0, leafsize, rootlen);
    }
}

/* Children first over a tree built with a plan: each task's nodes in reverse in parallel, then
 * the top nodes serially; each node reads only its children, so the serial pass's results. */
#define SUBHALO_TREE_UPWARD_PAR(t, plan, node_fn)                                       \
    do {                                                                                \
        int k_;                                                                         \
        _Pragma("omp parallel for schedule(dynamic, 1)")                               \
        for(k_ = 0; k_ < (plan)->n; k_++) {                                             \
            const struct TopItem * it_ = &(plan)->it[k_];                               \
            int64_t m_;                                                                 \
            if(it_->end < 0)                                                            \
                for(m_ = it_->off + it_->nodes - 1; m_ >= it_->off; m_--)               \
                    node_fn((t), m_);                                                   \
        }                                                                               \
        for(k_ = (plan)->n - 1; k_ >= 0; k_--)                                          \
            if((plan)->it[k_].end >= 0)                                                 \
                node_fn((t), (plan)->it[k_].off);                                       \
    } while(0)

/* Monopole of node n from its sources (leaf) or its children (HBT UpdateInternalNodes,
 * gravity_tree.cpp:27-90). */
static inline void
grav_moment_node(struct SubTree * t, int64_t n)
{
    double m = 0, mx[3] = {0, 0, 0};
    int d;
    if(t->sk[n].skip == n + 1) {
        int64_t j;
        for(j = t->sk[n].first; j < t->sk[n].first + t->sk[n].count; j++) {
            m += t->src[j].m;
            for(d = 0; d < 3; d++)
                mx[d] += t->src[j].m * t->src[j].x[d];
        }
    }
    else {
        int64_t ch;
        for(ch = n + 1; ch < t->sk[n].skip; ch = t->sk[ch].skip) {
            m += t->mom[ch].mass;
            for(d = 0; d < 3; d++)
                mx[d] += t->mom[ch].mass * t->mom[ch].com[d];
        }
    }
    t->mom[n].mass = m;
    for(d = 0; d < 3; d++)
        t->mom[n].com[d] = m > 0 ? mx[d] / m : 0;
}

/* The parallel-mode build; its own frame holds the plan (outside any parallel region). */
static void __attribute__((noinline))
tree_build_gravity_par(struct SubTree * t, struct SubhaloArena * arena)
{
    struct TopPlan plan;
    tree_build_skeleton(t, sizeof(struct SubMom), arena, &plan);
    t->mom = (struct SubMom *) arena_need(arena, sizeof(struct SubMom) * t->nn, "tree moments");
    SUBHALO_TREE_UPWARD_PAR(t, &plan, grav_moment_node);
}

/* Gravity tree: skeleton plus monopoles (mass, centre of mass), computed children first by a
 * reverse pre-order pass. */
static void
tree_build_gravity(struct SubTree * t, struct SubhaloArena * arena)
{
    if(par_mode(t->ns)) {
        tree_build_gravity_par(t, arena);
        return;
    }
    tree_build_skeleton(t, sizeof(struct SubMom), arena, NULL);
    t->mom = (struct SubMom *) arena_need(arena, sizeof(struct SubMom) * t->nn, "tree moments");
    int64_t n;
    for(n = t->nn - 1; n >= 0; n--)
        grav_moment_node(t, n);
}

/* |NEAREST(x - y)|^2 for coordinates differing by less than 1.5 BoxSize per axis; wrap == 0 when
 * every separation is known to be below BoxSize / 2 (NEAREST is then the identity). */
static inline __attribute__((always_inline)) double
pair_r2(const double x[3], const double y[3], double BoxSize, const int wrap)
{
    double ex = x[0] - y[0], ey = x[1] - y[1], ez = x[2] - y[2];
    if(wrap) {
        ex = NEAREST(ex, BoxSize);
        ey = NEAREST(ey, BoxSize);
        ez = NEAREST(ez, BoxSize);
    }
    return ex * ex + ey * ey + ez * ez;
}

static inline double
periodic_r2(const double x[3], const double y[3], double BoxSize)
{
    return pair_r2(x, y, BoxSize, 1);
}

/* sum_j m_j W(|x_j - y|, h) over the tree, excluding the source self (HBT EvaluatePotential:
 * open when len^2 > r^2 theta^2 against the node's centre of mass). theta2 < 1/3 guarantees that
 * a node containing the target is always opened (its cell diagonal is sqrt(3) len). */
static inline __attribute__((always_inline)) double
tree_potential_impl(const struct SubTree * t, const double y[3], int64_t self, double theta2, double h, const int wrap)
{
    const double BoxSize = t->BoxSize;
    double sum = 0;
    int64_t n = 0;
    while(n < t->nn) {
        const struct SubSkel * s = &t->sk[n];
        const double r2 = pair_r2(t->mom[n].com, y, BoxSize, wrap);
        const double len = s->len;
        if(len * len > r2 * theta2) {
            /* open */
            if(s->skip == n + 1) {
                int64_t j;
                for(j = s->first; j < s->first + s->count; j++) {
                    if(t->src[j].p == self)
                        continue;
                    sum += t->src[j].m * grav_spline_potential(sqrt(pair_r2(t->src[j].x, y, BoxSize, wrap)), h);
                }
                n = s->skip;
            }
            else
                n++;
        }
        else {
            sum += t->mom[n].mass * grav_spline_potential(sqrt(r2), h);
            n = s->skip;
        }
    }
    return sum;
}

static double
tree_potential(const struct SubTree * t, const double y[3], int64_t self, double theta2, double h)
{
    return t->wrap ? tree_potential_impl(t, y, self, theta2, h, 1) : tree_potential_impl(t, y, self, theta2, h, 0);
}

/* The same sum by direct summation. */
static inline __attribute__((always_inline)) double
direct_potential_impl(const struct SubTree * t, const double y[3], int64_t self, double h, const int wrap)
{
    double sum = 0;
    int64_t j;
    for(j = 0; j < t->ns; j++) {
        if(t->src[j].p == self)
            continue;
        sum += t->src[j].m * grav_spline_potential(sqrt(pair_r2(t->src[j].x, y, t->BoxSize, wrap)), h);
    }
    return sum;
}

static double
direct_potential(const struct SubTree * t, const double y[3], int64_t self, double h)
{
    return t->wrap ? direct_potential_impl(t, y, self, h, 1) : direct_potential_impl(t, y, self, h, 0);
}

/* ------------------------------------------------------------------ energies */

double
subhalo_thermal_energy(int64_t i, double atime)
{
    if(P[i].Type != 0)
        return 0;
    /* petaio.c GTInternalEnergy */
    const double a3inv = 1. / (atime * atime * atime);
    return SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPHP(i).Density * a3inv, GAMMA_MINUS1);
}

/* Widen [lo, hi] to the targets' frame coordinates NEAREST(Pos - RefPos). */
static void
tgt_extent(const int64_t * tgt, int64_t nt, const double RefPos[3], double BoxSize, int par, double lo[3], double hi[3])
{
    int d;
    if(!par) {
        int64_t i;
        for(i = 0; i < nt; i++)
            for(d = 0; d < 3; d++) {
                const double y = NEAREST(P[tgt[i]].Pos[d] - RefPos[d], BoxSize);
                if(y < lo[d]) lo[d] = y;
                if(y > hi[d]) hi[d] = y;
            }
        return;
    }
    #pragma omp parallel
    {
        double l[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, h[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        int64_t i;
        int dd;
        #pragma omp for schedule(static)
        for(i = 0; i < nt; i++)
            for(dd = 0; dd < 3; dd++) {
                const double y = NEAREST(P[tgt[i]].Pos[dd] - RefPos[dd], BoxSize);
                if(y < l[dd]) l[dd] = y;
                if(y > h[dd]) h[dd] = y;
            }
        #pragma omp critical(subhalo_tgt_extent)
        for(dd = 0; dd < 3; dd++) {
            if(l[dd] < lo[dd]) lo[dd] = l[dd];
            if(h[dd] > hi[dd]) hi[dd] = h[dd];
        }
    }
}

/* Pass 1 of potential_energies on srcidx[i0, i1): potential-type sources, the subsampleable
 * ones among them and their mass per type (summed in list order). */
static void
source_count_chunk(const int64_t * srcidx, int64_t i0, int64_t i1, int PotentialTypes, int64_t * npot, int64_t * nsub, double M[6])
{
    int64_t i, np = 0, nsb = 0;
    int t;
    for(t = 0; t < 6; t++)
        M[t] = 0;
    for(i = i0; i < i1; i++) {
        const int type = P[srcidx[i]].Type;
        if(!type_in(PotentialTypes, type))
            continue;
        np++;
        if(!type_in(SUBHALO_NOSUBSAMPLE_TYPES, type)) {
            nsb++;
            M[type] += P[srcidx[i]].Mass;
        }
    }
    *npot = np;
    *nsub = nsb;
}

/* The source records of srcidx[i0, i1) in list order from out[0]: every potential-type source,
 * except, when sampled, the subsampleable ones of rank >= sample in the whole list (rank0: the
 * rank of the chunk's first subsampleable source). */
static void
source_gather_chunk(const int64_t * srcidx, int64_t i0, int64_t i1, int PotentialTypes, int sampled, int64_t sample,
                    const double fac[6], const double RefPos[3], double BoxSize, int64_t rank0, struct SubSrc * out)
{
    int64_t i, rank = rank0, k = 0;
    int d;
    for(i = i0; i < i1; i++) {
        const int64_t p = srcidx[i];
        const int type = P[p].Type;
        if(!type_in(PotentialTypes, type))
            continue;
        if(sampled && !type_in(SUBHALO_NOSUBSAMPLE_TYPES, type)) {
            if(rank++ >= sample)
                continue;
        }
        out[k].p = p;
        out[k].m = P[p].Mass * fac[type];
        for(d = 0; d < 3; d++)
            out[k].x[d] = NEAREST(P[p].Pos[d] - RefPos[d], BoxSize);
        k++;
    }
}

/* Energies of tgt[0, nt) in the frame (RefPos, RefVel), sourced by the potential types of
 * srcidx[0, nsrc). If sample > 0 and more can be subsampled, only the never-subsampled ones and
 * the first `sample` others (the caller shuffled the list) source it, masses scaled per type to
 * conserve each type's mass (HBT). Direct sum up to direct_max unsampled sources, else the
 * tree. Returns 1 if subsampled. */
static int
potential_energies(const int64_t * tgt, int64_t nt, double * E, const int64_t * srcidx, int64_t nsrc, int64_t sample,
                   int64_t direct_max, const double RefPos[3], const double RefVel[3],
                   const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    const double BoxSize = up->BoxSize;
    int64_t i;
    int d, t;
    /* pass 1: what can be subsampled, per fixed chunk of the list, partials in chunk order */
    const int64_t C = chunk_len(nsrc);
    const int nch = chunk_count(nsrc, C);
    const int gpar = par_mode(nsrc) && nch > 1;
    int64_t cpot[SUBHALO_CHUNK_MAX], csub[SUBHALO_CHUNK_MAX], cout[SUBHALO_CHUNK_MAX], crank[SUBHALO_CHUNK_MAX];
    double cM[SUBHALO_CHUNK_MAX][6];
    int ch;
    if(gpar) {
        #pragma omp parallel for schedule(dynamic, 1)
        for(ch = 0; ch < nch; ch++)
            source_count_chunk(srcidx, ch * C, (ch + 1) * C < nsrc ? (ch + 1) * C : nsrc, up->PotentialTypes,
                               &cpot[ch], &csub[ch], cM[ch]);
    }
    else
        for(ch = 0; ch < nch; ch++)
            source_count_chunk(srcidx, ch * C, (ch + 1) * C < nsrc ? (ch + 1) * C : nsrc, up->PotentialTypes,
                               &cpot[ch], &csub[ch], cM[ch]);
    int64_t npot = 0, nsubable = 0;
    double Mtot[6] = {0};
    for(ch = 0; ch < nch; ch++) {
        npot += cpot[ch];
        nsubable += csub[ch];
        for(t = 0; t < 6; t++)
            Mtot[t] += cM[ch][t];
    }
    const int sampled = sample > 0 && nsubable > sample;
    double fac[6] = {1, 1, 1, 1, 1, 1};
    int64_t ns = npot;
    if(sampled) {
        /* serial: stops after the first `sample` subsampleable sources */
        double Msamp[6] = {0};
        int64_t taken = 0;
        for(i = 0; i < nsrc && taken < sample; i++) {
            const int type = P[srcidx[i]].Type;
            if(type_in(up->PotentialTypes, type) && !type_in(SUBHALO_NOSUBSAMPLE_TYPES, type)) {
                Msamp[type] += P[srcidx[i]].Mass;
                taken++;
            }
        }
        double TotTrue = 0, TotScaled = 0;
        for(t = 0; t < 6; t++) {
            if(type_in(SUBHALO_NOSUBSAMPLE_TYPES, t))
                continue;
            fac[t] = Msamp[t] > 0 ? Mtot[t] / Msamp[t] : 0;
            TotTrue += Mtot[t];
            TotScaled += Msamp[t] * fac[t];
        }
        /* mass conservation when a type is absent from the sample */
        for(t = 0; t < 6; t++)
            if(!type_in(SUBHALO_NOSUBSAMPLE_TYPES, t) && TotScaled > 0)
                fac[t] *= TotTrue / TotScaled;
        ns = npot - nsubable + sample;
    }

    /* the gather keeps the list order: each chunk's output offset and first subsampleable rank */
    {
        int64_t off = 0, rank = 0;
        for(ch = 0; ch < nch; ch++) {
            cout[ch] = off;
            crank[ch] = rank;
            int64_t tsub = csub[ch];
            if(sampled) {
                tsub = sample - rank;
                if(tsub < 0) tsub = 0;
                if(tsub > csub[ch]) tsub = csub[ch];
            }
            off += cpot[ch] - csub[ch] + tsub;
            rank += csub[ch];
        }
    }
    struct SubTree tree = {0};
    tree.BoxSize = BoxSize;
    tree.src = (struct SubSrc *) arena_need(arena, sizeof(struct SubSrc) * (ns > 0 ? ns : 1), "potential sources");
    struct SubSrc * src = tree.src;
    if(gpar) {
        #pragma omp parallel for schedule(dynamic, 1)
        for(ch = 0; ch < nch; ch++)
            source_gather_chunk(srcidx, ch * C, (ch + 1) * C < nsrc ? (ch + 1) * C : nsrc, up->PotentialTypes,
                                sampled, sample, fac, RefPos, BoxSize, crank[ch], src + cout[ch]);
    }
    else
        for(ch = 0; ch < nch; ch++)
            source_gather_chunk(srcidx, ch * C, (ch + 1) * C < nsrc ? (ch + 1) * C : nsrc, up->PotentialTypes,
                                sampled, sample, fac, RefPos, BoxSize, crank[ch], src + cout[ch]);
    tree.ns = ns;
    /* sources and targets spanning less than BoxSize / 2 (with a margin) on every axis need no
     * wrapping */
    {
        double lo[3], hi[3];
        src_extent(tree.src, tree.ns, par_mode(tree.ns), lo, hi);
        tgt_extent(tgt, nt, RefPos, BoxSize, par_mode(nt), lo, hi);
        tree.wrap = 0;
        for(d = 0; d < 3; d++)
            if(!(hi[d] - lo[d] < 0.49 * BoxSize))
                tree.wrap = 1;
    }
    const int use_tree = sampled || tree.ns > direct_max;
    if(use_tree && tree.ns > 0)
        tree_build_gravity(&tree, arena);

    const double a = up->atime;
    const double Ga = up->G / a;
    const double Ha = up->Hubble * a;
    double theta2 = up->OpenAngle * up->OpenAngle;
    if(theta2 > 0.33)
        theta2 = 0.33;
    const int par = loop_par(nt);
    const int lch = loop_chunk(nt);
    #pragma omp parallel for schedule(dynamic, lch) if(par)
    for(i = 0; i < nt; i++) {
        const int64_t p = tgt[i];
        double y[3], K = 0;
        int dd;
        for(dd = 0; dd < 3; dd++) {
            y[dd] = NEAREST(P[p].Pos[dd] - RefPos[dd], BoxSize);
            const double dv = (P[p].Vel[dd] - RefVel[dd]) / a + Ha * y[dd];
            K += dv * dv;
        }
        double sum = 0;
        if(tree.ns > 0)
            sum = use_tree ? tree_potential(&tree, y, p, theta2, up->h) : direct_potential(&tree, y, p, up->h);
        double e = 0.5 * K - Ga * sum;
        if(up->WithThermal)
            e += subhalo_thermal_energy(p, a);
        E[i] = e;
    }
    arena->used = mark;
    return sampled;
}

void
subhalo_energies(const int64_t * idx, int64_t n, double * E, const int64_t * src, int64_t ns,
                 const double RefPos[3], const double RefVel[3], const struct SubhaloUnbindParams * up,
                 struct SubhaloArena * arena)
{
    potential_energies(idx, n, E, src, ns, 0, up->SampleSize, RefPos, RefVel, up, arena);
}

/* Partial sums of bound_frame over Index[i0, i1), in list order. */
static void
bound_frame_chunk(const int64_t * Index, int64_t i0, int64_t i1, const double ref[3], double BoxSize, double part[7])
{
    double M = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0};
    int64_t i;
    int d;
    for(i = i0; i < i1; i++) {
        const struct particle_data * pp = &P[Index[i]];
        const double m = pp->Mass;
        M += m;
        for(d = 0; d < 3; d++) {
            mx[d] += m * NEAREST(pp->Pos[d] - ref[d], BoxSize);
            mv[d] += m * pp->Vel[d];
        }
    }
    part[0] = M;
    for(d = 0; d < 3; d++) {
        part[1 + d] = mx[d];
        part[4 + d] = mv[d];
    }
}

/* Mass-weighted centre (relative to the first particle, wrapped) and velocity of P[Index[0, n)]
 * (HBT CentreOfMassReferenceFrame), over the fixed chunks. Returns the mass; n == 0: untouched. */
static double
bound_frame(const int64_t * Index, int64_t n, double Pos[3], double Vel[3], double BoxSize)
{
    if(n <= 0)
        return 0;
    const int64_t C = chunk_len(n);
    const int nch = chunk_count(n, C);
    double part[SUBHALO_CHUNK_MAX][7];
    int ch, d;
    double M = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0}, ref[3];
    for(d = 0; d < 3; d++)
        ref[d] = P[Index[0]].Pos[d];
    if(par_mode(n) && nch > 1) {
        #pragma omp parallel for schedule(dynamic, 1)
        for(ch = 0; ch < nch; ch++)
            bound_frame_chunk(Index, ch * C, (ch + 1) * C < n ? (ch + 1) * C : n, ref, BoxSize, part[ch]);
    }
    else
        for(ch = 0; ch < nch; ch++)
            bound_frame_chunk(Index, ch * C, (ch + 1) * C < n ? (ch + 1) * C : n, ref, BoxSize, part[ch]);
    for(ch = 0; ch < nch; ch++) {
        M += part[ch][0];
        for(d = 0; d < 3; d++) {
            mx[d] += part[ch][1 + d];
            mv[d] += part[ch][4 + d];
        }
    }
    if(M <= 0)
        return 0;
    for(d = 0; d < 3; d++) {
        Pos[d] = wrap_box(ref[d] + mx[d] / M, BoxSize);
        Vel[d] = mv[d] / M;
    }
    return M;
}

/* Tracers among Index[0, n), counting stopped at stop (serial: the survival test stops early). */
static int64_t
count_tracers(const int64_t * Index, int64_t n, int TracerTypes, int64_t stop)
{
    int64_t i, nt = 0;
    for(i = 0; i < n && nt < stop; i++)
        if(type_in(TracerTypes, P[Index[i]].Type))
            nt++;
    return nt;
}

/* All tracers among Index[0, n) (an integer sum: any order). */
static int64_t
count_tracers_all(const int64_t * Index, int64_t n, int TracerTypes)
{
    if(!par_mode(n))
        return count_tracers(Index, n, TracerTypes, n);
    int64_t i, nt = 0;
    #pragma omp parallel for schedule(static) reduction(+: nt)
    for(i = 0; i < n; i++)
        if(type_in(TracerTypes, P[Index[i]].Type))
            nt++;
    return nt;
}

/* Bound (Energy < 0) to the front of (Index, Energy)[0, n), returning their number; the bound
 * keep their order. n <= CHUNK_MIN: the swap loop; above, a stable partition (decision 34: the
 * swap loop's unbound order cannot be parallelised, and only a dead candidate's list shows it). */
static int64_t
partition_bound(int64_t * Index, double * Energy, int64_t n, struct SubhaloArena * arena)
{
    int64_t i, j = 0;
    if(n <= SUBHALO_UNBIND_CHUNK_MIN) {
        for(i = 0; i < n; i++) {
            if(Energy[i] < 0) {
                if(i != j) {
                    const int64_t ti = Index[i];
                    const double te = Energy[i];
                    Index[i] = Index[j];
                    Energy[i] = Energy[j];
                    Index[j] = ti;
                    Energy[j] = te;
                }
                j++;
            }
        }
        return j;
    }
    const size_t mark = arena->used;
    int64_t * ti = (int64_t *) arena_need(arena, sizeof(int64_t) * n, "partition indices");
    double * te = (double *) arena_need(arena, sizeof(double) * n, "partition energies");
    const int64_t C = chunk_len(n);
    const int nch = chunk_count(n, C);
    const int par = par_mode(n);
    int64_t cb[SUBHALO_CHUNK_MAX], boff[SUBHALO_CHUNK_MAX], uoff[SUBHALO_CHUNK_MAX];
    int ch;
    #pragma omp parallel for schedule(static) if(par)
    for(ch = 0; ch < nch; ch++) {
        const int64_t i1 = (ch + 1) * C < n ? (ch + 1) * C : n;
        int64_t ii, nb = 0;
        for(ii = ch * C; ii < i1; ii++)
            nb += Energy[ii] < 0;
        cb[ch] = nb;
    }
    int64_t nbound = 0;
    for(ch = 0; ch < nch; ch++) {
        boff[ch] = nbound;
        nbound += cb[ch];
    }
    for(ch = 0; ch < nch; ch++)
        uoff[ch] = nbound + ch * C - boff[ch];
    #pragma omp parallel for schedule(static) if(par)
    for(ch = 0; ch < nch; ch++) {
        const int64_t i1 = (ch + 1) * C < n ? (ch + 1) * C : n;
        int64_t ii, kb = boff[ch], ku = uoff[ch];
        for(ii = ch * C; ii < i1; ii++) {
            const int64_t k = Energy[ii] < 0 ? kb++ : ku++;
            ti[k] = Index[ii];
            te[k] = Energy[ii];
        }
    }
    SUBHALO_PAR_FOR(par, i, n, Index[i] = ti[i]; Energy[i] = te[i];);
    arena->used = mark;
    return nbound;
}

/* ------------------------------------------------------------------ the unbinding loop */

/* HBT RefineBindingEnergyOrder: reorder Index[0, nref) by that subset's exact-mass self-binding
 * energy; the loop energies travel with their particles, as in HBT. */
static void
refine_most_bound(struct SubCandidate * c, int64_t nref, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    int64_t i;
    double * Eref = (double *) arena_need(arena, sizeof(double) * nref, "refinement energies");
    potential_energies(c->Index, nref, Eref, c->Index, nref, 0, up->SampleSize, c->RefPos, c->RefVel, up, arena);
    struct EPairPos * pr = (struct EPairPos *) arena_need(arena, sizeof(struct EPairPos) * nref, "refinement order");
    double * oe = (double *) arena_need(arena, sizeof(double) * nref, "refinement energy copy");
    int64_t * Index = c->Index;
    double * Energy = c->Energy;
    const int par = par_mode(nref);
    /* ties by the particle's ID, not its list position */
    SUBHALO_PAR_FOR(par, i, nref, pr[i].E = Eref[i]; pr[i].idx = Index[i]; pr[i].pos = i; oe[i] = Energy[i];);
    sort_epairpos_par(pr, nref, arena);
    SUBHALO_PAR_FOR(par, i, nref, Index[i] = pr[i].idx; Energy[i] = oe[pr[i].pos];);
    arena->used = mark;
}

int
subhalo_unbind(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    const int64_t N = c->N;
    int64_t * Index = c->Index;
    double * Energy = c->Energy;
    int64_t i;

    c->Iterations = 0;
    c->Nbound = 0;
    c->NboundTracer = 0;
    c->Mbound = 0;
    c->ConvergenceRatio = 0;
    c->NboundPositive = -1;

    /* A source below the thresholds cannot survive: the first iteration would kill it. */
    if(N < up->MinNumPart || N <= 0 || count_tracers(Index, N, up->TracerTypes, up->MinNumTracerPart) < up->MinNumTracerPart)
        return 0;

    /* Shuffle once: each iteration's subsample is the first SampleSize subsampleable particles
     * of the bound prefix, which keeps its shuffled order (HBT PrepareParticlesForSubsampling). */
    const int64_t sample = up->SampleSize > 0 ? up->SampleSize : 0;
    if(sample > 0 && N > sample) {
        uint64_t s = c->Seed;
        for(i = N - 1; i > 0; i--) {
            const int64_t j = (int64_t) rng_below(&s, (uint64_t) (i + 1));
            const int64_t t = Index[i];
            Index[i] = Index[j];
            Index[j] = t;
        }
    }

    double RefPos[3], RefVel[3];
    int d;
    for(d = 0; d < 3; d++) {
        RefPos[d] = c->RefPos[d];
        RefVel[d] = c->RefVel[d];
    }
    int64_t Nbound = N, Nlast;
    double Mbound = 0;
    int sampled_last = 0;
    for(;;) {
        Nlast = Nbound;
        c->Iterations++;
        /* energies of the current set in the current frame, the tree rebuilt on the current
         * set every iteration (no correction tree: design D3 item 5) */
        sampled_last = potential_energies(Index, Nlast, Energy, Index, Nlast, sample, sample, RefPos, RefVel, up, arena);

        /* bound to the front in their (shuffled) order; the unbound are sorted at the end */
        Nbound = partition_bound(Index, Energy, Nlast, arena);
        c->ConvergenceRatio = (double) Nbound / (double) Nlast;

        /* survival thresholds (subhalo_unbind.cpp:629-674) */
        const int64_t ntr = count_tracers(Index, Nbound, up->TracerTypes, up->MinNumTracerPart);
        if(Nbound < up->MinNumPart || ntr < up->MinNumTracerPart) {
            c->Nbound = 0;
            c->NboundTracer = 0;
            c->Mbound = 0;
            for(d = 0; d < 3; d++) {
                c->RefPos[d] = RefPos[d];
                c->RefVel[d] = RefVel[d];
            }
            arena->used = mark;
            return 0;
        }

        /* frame of the bound set (subhalo_unbind.cpp:690-692) */
        Mbound = bound_frame(Index, Nbound, RefPos, RefVel, up->BoxSize);

        /* count convergence (subhalo_unbind.cpp:695) */
        if((double) Nbound >= (double) Nlast * up->BoundMassPrecision)
            break;
        if(up->MaxIter > 0 && c->Iterations >= up->MaxIter) {
            c->Flags |= SUBHALO_FLAG_ITERCAP;
            break;
        }
    }

    sort_energy_index(Energy, Index, Nbound, arena);
    sort_energy_index(Energy + Nbound, Index + Nbound, N - Nbound, arena);

    c->Nbound = Nbound;
    c->Mbound = Mbound;
    for(d = 0; d < 3; d++) {
        c->RefPos[d] = RefPos[d];
        c->RefVel[d] = RefVel[d];
    }
    /* centre refinement after a subsampled loop (subhalo_unbind.cpp:718-733) */
    if(sampled_last && Nbound > sample) {
        int64_t nref = (int64_t) (SUBHALO_REFINE_FRACTION * Nbound);
        if(nref < sample)
            nref = sample;
        if(nref > Nbound)
            nref = Nbound;
        refine_most_bound(c, nref, up, arena);
    }
    c->NboundTracer = count_tracers_all(Index, Nbound, up->TracerTypes);
    arena->used = mark;
    return 1;
}

/* The head K of the sampled final pass: the shortest prefix with T = MARGIN * Kt tracers (all if
 * fewer, then subhalo_core's non-tracer fill up to T) and at least T particles. The margin
 * absorbs the loop order misplacing tracers by a few ranks. */
static int64_t
final_refine_count(const int64_t * Index, int64_t n, const struct SubhaloUnbindParams * up)
{
    const int64_t Kt = up->FinalRefineTracers > SUBHALO_FINAL_REFINE_MIN ? up->FinalRefineTracers : SUBHALO_FINAL_REFINE_MIN;
    int64_t T = SUBHALO_FINAL_REFINE_MARGIN * Kt;
    if(T > n)
        T = n;
    const int64_t NT = count_tracers_all(Index, n, up->TracerTypes);
    const int64_t need_tr = NT < T ? NT : T;
    const int64_t need_nt = T - need_tr;
    int64_t i, nt = 0, nn = 0;
    for(i = 0; i < n; i++) {
        if(i >= T && nt >= need_tr && nn >= need_nt)
            break;
        if(type_in(up->TracerTypes, P[Index[i]].Type))
            nt++;
        else
            nn++;
    }
    return i;
}

int64_t
subhalo_final_pass(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    const int64_t n = c->Nbound;
    c->NboundPositive = 0;
    if(n <= 0)
        return 0;
    c->Mbound = bound_frame(c->Index, n, c->RefPos, c->RefVel, up->BoxSize);
    int64_t i, npos = 0;
    if(up->SampledFinalPass && up->SampleSize > 0 && n > up->SampleSize) {
        /* the sampled pass: the order of subhalo_unbind kept, except its head. */
        const int64_t K = final_refine_count(c->Index, n, up);
        /* 1. the head: the exact pass's call (same arena state, so bitwise the same energies),
         *    then sorted by them */
        potential_energies(c->Index, K, c->Energy, c->Index, n, 0, up->SampleSize, c->RefPos, c->RefVel, up, arena);
        sort_energy_index(c->Energy, c->Index, K, arena);
        /* 2. all energies again with the loop's subsampled estimator on a seeded shuffle, for the
         *    property sums (the head's too, so the sums use one estimator) */
        int64_t * sh = (int64_t *) arena_need(arena, sizeof(int64_t) * n, "final-pass subsample");
        const int64_t * Index = c->Index;
        SUBHALO_PAR_FOR(par_mode(n), i, n, sh[i] = Index[i];);
        uint64_t s = c->Seed ^ 0x5851f42d4c957f2dULL;
        for(i = n - 1; i > 0; i--) {
            const int64_t j = (int64_t) rng_below(&s, (uint64_t) (i + 1));
            const int64_t t = sh[i];
            sh[i] = sh[j];
            sh[j] = t;
        }
        potential_energies(c->Index, n, c->Energy, sh, n, up->SampleSize, up->SampleSize, c->RefPos, c->RefVel, up, arena);
        arena->used = mark;
    }
    else {
        potential_energies(c->Index, n, c->Energy, c->Index, n, 0, up->SampleSize, c->RefPos, c->RefVel, up, arena);
        sort_energy_index(c->Energy, c->Index, n, arena);
    }
    const double * E = c->Energy;
    if(par_mode(n)) {
        #pragma omp parallel for schedule(static) reduction(+: npos)
        for(i = 0; i < n; i++)
            if(E[i] >= 0)
                npos++;
    }
    else
        for(i = 0; i < n; i++)
            if(E[i] >= 0)
                npos++;
    c->NboundTracer = count_tracers_all(c->Index, n, up->TracerTypes);
    c->NboundPositive = npos;
    arena->used = mark;
    return npos;
}

/* ------------------------------------------------------------------ core and sinking */

void
subhalo_core(const struct SubCandidate * c, const struct SubhaloUnbindParams * up, int CoreSize,
             double CorePos[3], double CoreVel[3], double * SigmaX, double * SigmaV)
{
    int d, pass;
    int64_t i, num = 0;
    const int64_t n = c->Nbound;
    const int64_t size = CoreSize > 0 ? CoreSize : 1;
    double msum = 0, pos[3] = {0, 0, 0}, pos2[3] = {0, 0, 0}, vel[3] = {0, 0, 0}, vel2[3] = {0, 0, 0}, origin[3];
    if(n <= 0) {
        for(d = 0; d < 3; d++)
            CorePos[d] = CoreVel[d] = 0;
        *SigmaX = *SigmaV = 0;
        return;
    }
    for(d = 0; d < 3; d++)
        origin[d] = P[c->Index[0]].Pos[d];
    /* tracers first, then non-tracers to fill (GetCorePhaseSpaceProperties,
     * subhalo_merge.cpp:93-181) */
    for(pass = 0; pass < 2 && num < size; pass++) {
        for(i = 0; i < n && num < size; i++) {
            const struct particle_data * pp = &P[c->Index[i]];
            const int tracer = type_in(up->TracerTypes, pp->Type);
            if((pass == 0) != (tracer != 0))
                continue;
            num++;
            const double m = pp->Mass;
            msum += m;
            for(d = 0; d < 3; d++) {
                const double dx = NEAREST(pp->Pos[d] - origin[d], up->BoxSize);
                pos[d] += m * dx;
                pos2[d] += m * dx * dx;
                vel[d] += m * pp->Vel[d];
                vel2[d] += m * pp->Vel[d] * pp->Vel[d];
            }
        }
    }
    double sx2 = 0, sv2 = 0;
    for(d = 0; d < 3; d++) {
        pos[d] /= msum;
        pos2[d] /= msum;
        vel[d] /= msum;
        vel2[d] /= msum;
        CorePos[d] = wrap_box(origin[d] + pos[d], up->BoxSize);
        CoreVel[d] = vel[d];
        const double vx = pos2[d] - pos[d] * pos[d];
        const double vv = vel2[d] - vel[d] * vel[d];
        sx2 += vx > 0 ? vx : 0;
        sv2 += vv > 0 ? vv : 0;
    }
    *SigmaX = sqrt(sx2);
    *SigmaV = sqrt(sv2);
}

/* d / sigma with a zero dispersion treated as an infinitely narrow core. */
static double
scaled_offset(double dist, double sigma)
{
    if(sigma > 0)
        return dist / sigma;
    return dist > 0 ? 1e300 : 0;
}

int
subhalo_sink_test(const double sPos[3], const double sVel[3], const double aPos[3], const double aVel[3],
                  double aSigmaX, double aSigmaV, double sSigmaX, double sSigmaV, int symmetric, double BoxSize)
{
    /* PhaseSpaceDistance / AreOverlappingInPhaseSpace, subhalo_merge.cpp:13-26 */
    double dx2 = 0, dv2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double dx = NEAREST(sPos[d] - aPos[d], BoxSize);
        const double dv = sVel[d] - aVel[d];
        dx2 += dx * dx;
        dv2 += dv * dv;
    }
    const double dx = sqrt(dx2), dv = sqrt(dv2);
    double D = scaled_offset(dx, aSigmaX) + scaled_offset(dv, aSigmaV);
    if(symmetric) {
        const double Ds = scaled_offset(dx, sSigmaX) + scaled_offset(dv, sSigmaV);
        if(Ds < D)
            D = Ds;
    }
    return D < 2.0;
}

/* ------------------------------------------------------------------ k nearest tracers */

/* The read-only neighbour tree: boxes in coordinates NEAREST(Pos - origin), sources holding
 * absolute positions after the build. slack lowers the box bounds so that the rounding of the
 * layout coordinates never prunes a node holding an exact tie. */
struct SubhaloKnnTree {
    struct SubTree t;
    double origin[3];
    double slack;
};

static int64_t knn_builds = 0;

int64_t
subhalo_knn_build_count(void)
{
    int64_t n;
    #pragma omp atomic read
    n = knn_builds;
    return n;
}

/* Neighbour list of one query: sorted ascending by (d2, ID, P index), at most k entries. */
struct KnnList {
    double d2[SUBHALO_KNN_MAXK];
    int64_t p[SUBHALO_KNN_MAXK];
    int n, k;
};

static inline int
knn_better(double d2, int64_t p, double d2w, int64_t pw)
{
    return d2 < d2w || (d2 == d2w && id_less(p, pw));
}

static inline void
knn_insert(struct KnnList * L, double d2, int64_t p)
{
    if(L->n > 0 && L->n == L->k && !knn_better(d2, p, L->d2[L->n - 1], L->p[L->n - 1]))
        return;
    int j = L->n < L->k ? L->n++ : L->k - 1;
    while(j > 0 && knn_better(d2, p, L->d2[j - 1], L->p[j - 1])) {
        L->d2[j] = L->d2[j - 1];
        L->p[j] = L->p[j - 1];
        j--;
    }
    L->d2[j] = d2;
    L->p[j] = p;
}

/* Lower bound on one axis of the periodic distance from q to [lo, hi] (layout coordinates): the
 * gap to the nearer image, less the slack. */
static inline double
knn_axis_gap(double q, double lo, double hi, double BoxSize, double slack)
{
    double g, g2;
    if(q < lo) {
        g = lo - q;
        g2 = q + BoxSize - hi;
    }
    else if(q > hi) {
        g = q - hi;
        g2 = lo + BoxSize - q;
    }
    else
        return 0;
    if(g2 < g)
        g = g2;
    g -= slack;
    return g > 0 ? g : 0;
}

static inline double
knn_box_dist2(const struct SubhaloKnnTree * kt, int64_t n, const double q[3])
{
    const struct SubBox * b = &kt->t.box[n];
    double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        const double e = knn_axis_gap(q[d], b->lo[d], b->hi[d], kt->t.BoxSize, kt->slack);
        r2 += e * e;
    }
    return r2;
}

/* Depth-first, nearest child first, pruned by the k-th distance (HBT's NearestNeighbours); q in
 * layout coordinates for the boxes, qa absolute for the point distances. */
static void
knn_rec(const struct SubhaloKnnTree * kt, int64_t n, const double q[3], const double qa[3], struct KnnList * L)
{
    const struct SubTree * t = &kt->t;
    if(L->n > 0 && L->n == L->k && knn_box_dist2(kt, n, q) > L->d2[L->n - 1])
        return;
    if(t->sk[n].skip == n + 1) {
        int64_t j;
        for(j = t->sk[n].first; j < t->sk[n].first + t->sk[n].count; j++)
            knn_insert(L, periodic_r2(t->src[j].x, qa, t->BoxSize), t->src[j].p);
        return;
    }
    int64_t ch[8];
    double cd[8];
    int nc = 0, a, b;
    int64_t c;
    for(c = n + 1; c < t->sk[n].skip && nc < 8; c = t->sk[c].skip) {
        ch[nc] = c;
        cd[nc] = knn_box_dist2(kt, c, q);
        nc++;
    }
    for(a = 1; a < nc; a++)
        for(b = a; b > 0 && cd[b] < cd[b - 1]; b--) {
            const double td = cd[b]; cd[b] = cd[b - 1]; cd[b - 1] = td;
            const int64_t tc = ch[b]; ch[b] = ch[b - 1]; ch[b - 1] = tc;
        }
    for(a = 0; a < nc; a++)
        knn_rec(kt, ch[a], q, qa, L);
}

/* Bounding box of node n from its sources (leaf) or its children. */
static inline void
knn_box_node(struct SubTree * t, int64_t n)
{
    struct SubBox * b = &t->box[n];
    int d;
    for(d = 0; d < 3; d++) {
        b->lo[d] = DBL_MAX;
        b->hi[d] = -DBL_MAX;
    }
    if(t->sk[n].skip == n + 1) {
        int64_t j;
        for(j = t->sk[n].first; j < t->sk[n].first + t->sk[n].count; j++)
            for(d = 0; d < 3; d++) {
                if(t->src[j].x[d] < b->lo[d]) b->lo[d] = t->src[j].x[d];
                if(t->src[j].x[d] > b->hi[d]) b->hi[d] = t->src[j].x[d];
            }
    }
    else {
        int64_t ch;
        for(ch = n + 1; ch < t->sk[n].skip; ch = t->sk[ch].skip)
            for(d = 0; d < 3; d++) {
                if(t->box[ch].lo[d] < b->lo[d]) b->lo[d] = t->box[ch].lo[d];
                if(t->box[ch].hi[d] > b->hi[d]) b->hi[d] = t->box[ch].hi[d];
            }
    }
}

static void __attribute__((noinline))
knn_build_par(struct SubTree * t, struct SubhaloArena * arena)
{
    struct TopPlan plan;
    tree_build_skeleton(t, sizeof(struct SubBox), arena, &plan);
    t->box = (struct SubBox *) arena_need(arena, sizeof(struct SubBox) * t->nn, "neighbour boxes");
    SUBHALO_TREE_UPWARD_PAR(t, &plan, knn_box_node);
}

struct SubhaloKnnTree *
subhalo_knn_build(const int64_t * tr, int64_t nt, double BoxSize, struct SubhaloArena * arena)
{
    struct SubhaloKnnTree * kt = (struct SubhaloKnnTree *) arena_need(arena, sizeof(struct SubhaloKnnTree), "neighbour tree");
    int64_t i;
    int d;
    memset(kt, 0, sizeof(*kt));
    kt->t.BoxSize = BoxSize;
    kt->t.wrap = 1;
    /* far above the rounding of NEAREST(Pos - origin) (a few ulp of BoxSize), far below any
     * physical separation */
    kt->slack = 1e-12 * BoxSize;
    #pragma omp atomic
    knn_builds++;
    if(nt <= 0)
        return kt;
    for(d = 0; d < 3; d++)
        kt->origin[d] = P[tr[0]].Pos[d];
    struct SubTree * t = &kt->t;
    t->src = (struct SubSrc *) arena_need(arena, sizeof(struct SubSrc) * nt, "neighbour sources");
    struct SubSrc * src = t->src;
    const double * origin = kt->origin;
    const int par = par_mode(nt);
    SUBHALO_PAR_FOR(par, i, nt,
        int dd;
        src[i].p = tr[i];
        src[i].m = 1;
        for(dd = 0; dd < 3; dd++)
            src[i].x[dd] = NEAREST(P[tr[i]].Pos[dd] - origin[dd], BoxSize);
    );
    t->ns = nt;
    if(par)
        knn_build_par(t, arena);
    else {
        tree_build_skeleton(t, sizeof(struct SubBox), arena, NULL);
        t->box = (struct SubBox *) arena_need(arena, sizeof(struct SubBox) * t->nn, "neighbour boxes");
        int64_t n;
        for(n = t->nn - 1; n >= 0; n--)
            knn_box_node(t, n);
    }
    /* the point distances come from the absolute positions */
    SUBHALO_PAR_FOR(par, i, nt,
        int dd;
        for(dd = 0; dd < 3; dd++)
            src[i].x[dd] = P[src[i].p].Pos[dd];
    );
    return kt;
}

void
subhalo_knn_query(const struct SubhaloKnnTree * kt, const int64_t * query, int64_t nq, int k, int64_t * out)
{
    int64_t i;
    if(k > SUBHALO_KNN_MAXK)
        endrun(1, "Subhalo kernel: k = %d nearest neighbours exceeds SUBHALO_KNN_MAXK = %d\n", k, SUBHALO_KNN_MAXK);
    if(k <= 0)
        return;
    if(kt->t.ns <= 0) {
        for(i = 0; i < nq * k; i++)
            out[i] = -1;
        return;
    }
    const double BoxSize = kt->t.BoxSize;
    const int par = loop_par(nq);
    const int lch = loop_chunk(nq);
    #pragma omp parallel for schedule(dynamic, lch) if(par)
    for(i = 0; i < nq; i++) {
        struct KnnList L;
        double q[3], qa[3];
        int dd, j;
        L.n = 0;
        L.k = k;
        for(dd = 0; dd < 3; dd++) {
            qa[dd] = P[query[i]].Pos[dd];
            q[dd] = NEAREST(qa[dd] - kt->origin[dd], BoxSize);
        }
        knn_rec(kt, 0, q, qa, &L);
        for(j = 0; j < k; j++)
            out[i * k + j] = j < L.n ? L.p[j] : -1;
    }
}

void
subhalo_knn_tracers(const int64_t * query, int64_t nq, const int64_t * tr, int64_t nt, int k,
                    int64_t * out, double BoxSize, struct SubhaloArena * arena)
{
    if(k > SUBHALO_KNN_MAXK)
        endrun(1, "Subhalo kernel: k = %d nearest neighbours exceeds SUBHALO_KNN_MAXK = %d\n", k, SUBHALO_KNN_MAXK);
    if(k <= 0)
        return;
    const size_t mark = arena->used;
    const struct SubhaloKnnTree * kt = subhalo_knn_build(tr, nt, BoxSize, arena);
    subhalo_knn_query(kt, query, nq, k, out);
    arena->used = mark;
}
