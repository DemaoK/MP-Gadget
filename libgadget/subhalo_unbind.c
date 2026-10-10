/* Serial unbinding kernel of the subhalo finder (design 7.4, Phase 2a plan section 3).
 *
 * Ported from HBT-HERONS (HBT-BF tree): src/gravity_tree.cpp (monopole octree, geometric
 * opening), src/subhalo_unbind.cpp:463-808 (the loop), src/subhalo_merge.cpp (core and sinking)
 * and src/subhalo_reassign_gas.cpp (k-nearest tracers). The algorithms are ported, not the
 * classes; see subhalo_unbind.h for the contract and the conventions. */

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

/* splitmix64 (Steele, Lea & Flood 2014): the seeded generator of design D3 item 3, replacing
 * HBT's std::random_shuffle (subhalo_unbind.cpp:442), so results do not depend on the thread
 * or rank count. */
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

/* In-place introsort (quicksort, median of three, heapsort fallback, insertion sort below 16),
 * allocation-free and thread-safe, unlike qsort. Every comparator below is a strict total order
 * on distinct elements (ties broken by a P index or a position), so the result is unique and
 * does not depend on the algorithm. */
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

/* (energy, P index) pairs: the energy order of design 7.10, ties by P index. */
struct EPair {
    double E;
    int64_t idx;
};
#define EPAIR_LESS(x, y) ((x)->E < (y)->E || ((x)->E == (y)->E && (x)->idx < (y)->idx))
SUBHALO_DEFINE_SORT(sort_epair, struct EPair, EPAIR_LESS)

/* A gravity or neighbour source: Morton key, P index, position relative to the tree origin
 * (comoving), mass (upscaled when subsampled). 48 bytes. */
struct SubSrc {
    uint64_t key;
    int64_t p;
    double x[3];
    double m;
};
#define SRC_LESS(x, y) ((x)->key < (y)->key || ((x)->key == (y)->key && (x)->p < (y)->p))
SUBHALO_DEFINE_SORT(sort_src, struct SubSrc, SRC_LESS)

/* Sort the parallel arrays (Energy, Index)[0, n) by (Energy, Index), through a pair scratch. */
static void
sort_energy_index(double * Energy, int64_t * Index, int64_t n, struct SubhaloArena * arena)
{
    if(n < 2)
        return;
    const size_t mark = arena->used;
    struct EPair * pr = (struct EPair *) arena_need(arena, sizeof(struct EPair) * n, "energy sort pairs");
    int64_t i;
    for(i = 0; i < n; i++) {
        pr[i].E = Energy[i];
        pr[i].idx = Index[i];
    }
    sort_epair(pr, n);
    for(i = 0; i < n; i++) {
        Energy[i] = pr[i].E;
        Index[i] = pr[i].idx;
    }
    arena->used = mark;
}

/* ------------------------------------------------------------------ octree */

/* The tree is a Morton-ordered octree laid out in pre-order: the first child of node n is
 * n + 1, the next sibling of a node is its skip, and a node is a leaf iff skip == n + 1. Every
 * node covers the contiguous source range [first, first + count). Single-child chains are
 * collapsed (a node's cell is the smallest Morton cell holding its sources), so a node has
 * either >= 2 children or is a leaf of at most leafsize sources (or one Morton cell at the
 * finest level). Port of the tree of HBT's GravityTree_t (gravity_tree.cpp, oct_tree.cpp),
 * built from sorted keys instead of insertion. */
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

/* Recursive build over src[lo, hi) whose keys share the top 3 * level bits. sk == NULL counts
 * nodes only. Returns nothing; *nn is the running node count. Depth <= MORTON_LEVELS + 1. */
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

/* Build the skeleton over t->src[0, t->ns) (positions already relative to the origin):
 * Morton keys, sort, then as many nodes as fit in the arena (each needs node_bytes in all,
 * skeleton included), widening the leaves until the tree fits. */
static void
tree_build_skeleton(struct SubTree * t, size_t node_extra, struct SubhaloArena * arena)
{
    int64_t i;
    int d;
    const int64_t ns = t->ns;
    if(ns >= INT32_MAX)
        endrun(1, "Subhalo kernel: %ld sources exceed the int32 tree layout\n", (long) ns);
    double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    for(i = 0; i < ns; i++)
        for(d = 0; d < 3; d++) {
            if(t->src[i].x[d] < lo[d]) lo[d] = t->src[i].x[d];
            if(t->src[i].x[d] > hi[d]) hi[d] = t->src[i].x[d];
        }
    double ext = 0;
    for(d = 0; d < 3; d++)
        if(hi[d] - lo[d] > ext)
            ext = hi[d] - lo[d];
    double rootlen = ext * (1 + 1e-9);
    if(rootlen <= 0)
        rootlen = 1e-10 * (fabs(lo[0]) + fabs(lo[1]) + fabs(lo[2]) + 1);
    const double scale = (double) (1 << MORTON_LEVELS) / rootlen;
    for(i = 0; i < ns; i++) {
        uint64_t key = 0;
        for(d = 0; d < 3; d++) {
            double q = (t->src[i].x[d] - lo[d]) * scale;
            int64_t iq = (int64_t) q;
            if(iq < 0) iq = 0;
            if(iq > (1 << MORTON_LEVELS) - 1) iq = (1 << MORTON_LEVELS) - 1;
            key |= spread3((uint64_t) iq) << (2 - d);
        }
        t->src[i].key = key;
    }
    sort_src(t->src, ns);

    /* node budget from what is left; widen leaves until the tree fits */
    const size_t per_node = sizeof(struct SubSkel) + node_extra;
    int64_t cap = (int64_t) (arena_left(arena) / per_node);
    if(cap > INT32_MAX - 1)
        cap = INT32_MAX - 1;
    if(cap < 1)
        endrun(1, "Subhalo kernel: no arena space left for tree nodes (%ld sources)\n", (long) ns);
    int64_t leafsize = 8, nn;
    for(;;) {
        nn = 0;
        tree_build_rec(t->src, NULL, &nn, 0, ns, 0, leafsize, rootlen);
        if(nn <= cap)
            break;
        leafsize *= 2;
    }
    t->sk = (struct SubSkel *) arena_need(arena, sizeof(struct SubSkel) * nn, "tree skeleton");
    t->nn = 0;
    tree_build_rec(t->src, t->sk, &t->nn, 0, ns, 0, leafsize, rootlen);
}

/* Gravity tree: skeleton plus monopoles (mass, centre of mass), computed children first by a
 * reverse pre-order pass (HBT UpdateInternalNodes, gravity_tree.cpp:27-90). */
static void
tree_build_gravity(struct SubTree * t, struct SubhaloArena * arena)
{
    tree_build_skeleton(t, sizeof(struct SubMom), arena);
    t->mom = (struct SubMom *) arena_need(arena, sizeof(struct SubMom) * t->nn, "tree moments");
    int64_t n;
    int d;
    for(n = t->nn - 1; n >= 0; n--) {
        double m = 0, mx[3] = {0, 0, 0};
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
}

/* sum_j m_j W(|x_j - y|, h) over the tree, excluding the source with P index self
 * (EvaluatePotential, gravity_tree.cpp:92-171: geometric opening len^2 > r^2 theta^2 against
 * the node centre of mass, monopole otherwise). theta2 is capped below 1/3 so that a node
 * containing the target is always opened (true for HBT's 0.45). */
static double
tree_potential(const struct SubTree * t, const double y[3], int64_t self, double theta2, double h)
{
    double sum = 0;
    int64_t n = 0;
    while(n < t->nn) {
        const struct SubSkel * s = &t->sk[n];
        const double dx = t->mom[n].com[0] - y[0];
        const double dy = t->mom[n].com[1] - y[1];
        const double dz = t->mom[n].com[2] - y[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double len = s->len;
        if(len * len > r2 * theta2) {
            /* open */
            if(s->skip == n + 1) {
                int64_t j;
                for(j = s->first; j < s->first + s->count; j++) {
                    if(t->src[j].p == self)
                        continue;
                    const double ex = t->src[j].x[0] - y[0];
                    const double ey = t->src[j].x[1] - y[1];
                    const double ez = t->src[j].x[2] - y[2];
                    sum += t->src[j].m * grav_spline_potential(sqrt(ex * ex + ey * ey + ez * ez), h);
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

/* The same sum by direct summation. */
static double
direct_potential(const struct SubSrc * src, int64_t ns, const double y[3], int64_t self, double h)
{
    double sum = 0;
    int64_t j;
    for(j = 0; j < ns; j++) {
        if(src[j].p == self)
            continue;
        const double ex = src[j].x[0] - y[0];
        const double ey = src[j].x[1] - y[1];
        const double ez = src[j].x[2] - y[2];
        sum += src[j].m * grav_spline_potential(sqrt(ex * ex + ey * ey + ez * ez), h);
    }
    return sum;
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

/* Energies of tgt[0, nt) in the frame (RefPos, RefVel). The potential is sourced by the
 * potential-type particles of srcidx[0, nsrc). If sample > 0 and more than sample of them can
 * be subsampled, only the never-subsampled ones plus the first `sample` subsampleable ones (in
 * list order: the caller shuffled the list) source it, with masses scaled per type so each
 * type keeps its full mass and the total is conserved
 * (HBT GetMassUpscaleFactorPerParticleType, subhalo_unbind.cpp:345-390). Direct sum if the
 * source count is <= direct_max and the potential is not subsampled, the octree otherwise
 * (HBT always uses its tree). Returns 1 if subsampled. Arena rewound on return. */
static int
potential_energies(const int64_t * tgt, int64_t nt, double * E, const int64_t * srcidx, int64_t nsrc, int64_t sample,
                   int64_t direct_max, const double RefPos[3], const double RefVel[3],
                   const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    const double BoxSize = up->BoxSize;
    int64_t i;
    int d, t;
    /* pass 1: what can be subsampled */
    int64_t npot = 0, nsubable = 0;
    double Mtot[6] = {0};
    for(i = 0; i < nsrc; i++) {
        const int type = P[srcidx[i]].Type;
        if(!type_in(up->PotentialTypes, type))
            continue;
        npot++;
        if(!type_in(SUBHALO_NOSUBSAMPLE_TYPES, type)) {
            nsubable++;
            Mtot[type] += P[srcidx[i]].Mass;
        }
    }
    const int sampled = sample > 0 && nsubable > sample;
    double fac[6] = {1, 1, 1, 1, 1, 1};
    int64_t ns = npot;
    if(sampled) {
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

    struct SubTree tree = {0};
    tree.src = (struct SubSrc *) arena_need(arena, sizeof(struct SubSrc) * (ns > 0 ? ns : 1), "potential sources");
    int64_t taken = 0, k = 0;
    for(i = 0; i < nsrc; i++) {
        const int64_t p = srcidx[i];
        const int type = P[p].Type;
        if(!type_in(up->PotentialTypes, type))
            continue;
        if(sampled && !type_in(SUBHALO_NOSUBSAMPLE_TYPES, type)) {
            if(taken >= sample)
                continue;
            taken++;
        }
        tree.src[k].p = p;
        tree.src[k].m = P[p].Mass * fac[type];
        for(d = 0; d < 3; d++)
            tree.src[k].x[d] = NEAREST(P[p].Pos[d] - RefPos[d], BoxSize);
        k++;
    }
    tree.ns = k;
    const int use_tree = sampled || tree.ns > direct_max;
    if(use_tree && tree.ns > 0)
        tree_build_gravity(&tree, arena);

    const double a = up->atime;
    const double Ga = up->G / a;
    const double Ha = up->Hubble * a;
    double theta2 = up->OpenAngle * up->OpenAngle;
    if(theta2 > 0.33)
        theta2 = 0.33;
    const int par = nt > 4096 && !omp_in_parallel();
    #pragma omp parallel for schedule(dynamic, 256) if(par)
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
            sum = use_tree ? tree_potential(&tree, y, p, theta2, up->h) : direct_potential(tree.src, tree.ns, y, p, up->h);
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

/* Mass-weighted centre and velocity of P[Index[0, n)] (HBT CentreOfMassReferenceFrame,
 * subhalo_unbind.cpp:150-195: positions relative to the first particle, NEAREST). Pos wrapped
 * into [0, BoxSize). Returns the mass; leaves Pos/Vel untouched if n == 0. */
static double
bound_frame(const int64_t * Index, int64_t n, double Pos[3], double Vel[3], double BoxSize)
{
    if(n <= 0)
        return 0;
    int64_t i;
    int d;
    double M = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0}, ref[3];
    for(d = 0; d < 3; d++)
        ref[d] = P[Index[0]].Pos[d];
    for(i = 0; i < n; i++) {
        const struct particle_data * pp = &P[Index[i]];
        const double m = pp->Mass;
        M += m;
        for(d = 0; d < 3; d++) {
            mx[d] += m * NEAREST(pp->Pos[d] - ref[d], BoxSize);
            mv[d] += m * pp->Vel[d];
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

static int64_t
count_tracers(const int64_t * Index, int64_t n, int TracerTypes, int64_t stop)
{
    int64_t i, nt = 0;
    for(i = 0; i < n && nt < stop; i++)
        if(type_in(TracerTypes, P[Index[i]].Type))
            nt++;
    return nt;
}

/* ------------------------------------------------------------------ the unbinding loop */

/* HBT RefineBindingEnergyOrder (subhalo_unbind.cpp:398-428): reorder Index[0, nref) by the
 * self-binding energy of that subset with exact masses; the loop energies travel with their
 * particles (HBT does not update them either, :709-733). */
static void
refine_most_bound(struct SubCandidate * c, int64_t nref, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    int64_t i;
    double * Eref = (double *) arena_need(arena, sizeof(double) * nref, "refinement energies");
    potential_energies(c->Index, nref, Eref, c->Index, nref, 0, up->SampleSize, c->RefPos, c->RefVel, up, arena);
    struct EPair * pr = (struct EPair *) arena_need(arena, sizeof(struct EPair) * nref, "refinement order");
    int64_t * oi = (int64_t *) arena_need(arena, sizeof(int64_t) * nref, "refinement index copy");
    double * oe = (double *) arena_need(arena, sizeof(double) * nref, "refinement energy copy");
    for(i = 0; i < nref; i++) {
        pr[i].E = Eref[i];
        pr[i].idx = i;      /* ties by position, itself in (E, P index) order */
        oi[i] = c->Index[i];
        oe[i] = c->Energy[i];
    }
    sort_epair(pr, nref);
    for(i = 0; i < nref; i++) {
        c->Index[i] = oi[pr[i].idx];
        c->Energy[i] = oe[pr[i].idx];
    }
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

    /* A source below the thresholds cannot survive: the first iteration would kill it. */
    if(N < up->MinNumPart || N <= 0 || count_tracers(Index, N, up->TracerTypes, up->MinNumTracerPart) < up->MinNumTracerPart)
        return 0;

    /* Shuffle once: the random subsample of every iteration is the first SampleSize
     * subsampleable particles of the bound prefix, which the stable partition keeps in
     * shuffled order (HBT PrepareParticlesForSubsampling, subhalo_unbind.cpp:433-445). */
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

        /* bound to the front, keeping their relative (shuffled) order (stable_partition,
         * subhalo_unbind.cpp:35-44); the unbound are sorted at the end */
        int64_t j = 0;
        for(i = 0; i < Nlast; i++) {
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
        Nbound = j;

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
    c->NboundTracer = count_tracers(Index, Nbound, up->TracerTypes, Nbound);
    arena->used = mark;
    return 1;
}

int64_t
subhalo_final_pass(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    const int64_t n = c->Nbound;
    if(n <= 0)
        return 0;
    c->Mbound = bound_frame(c->Index, n, c->RefPos, c->RefVel, up->BoxSize);
    potential_energies(c->Index, n, c->Energy, c->Index, n, 0, up->SampleSize, c->RefPos, c->RefVel, up, arena);
    sort_energy_index(c->Energy, c->Index, n, arena);
    int64_t i, npos = 0;
    for(i = 0; i < n; i++)
        if(c->Energy[i] >= 0)
            npos++;
    c->NboundTracer = count_tracers(c->Index, n, up->TracerTypes, n);
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

/* Neighbour list of one query: sorted ascending by (d2, P index), at most k entries. */
struct KnnList {
    double d2[SUBHALO_KNN_MAXK];
    int64_t p[SUBHALO_KNN_MAXK];
    int n, k;
};

static inline int
knn_better(double d2, int64_t p, double d2w, int64_t pw)
{
    return d2 < d2w || (d2 == d2w && p < pw);
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

static inline double
box_dist2(const struct SubBox * b, const double q[3])
{
    double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        double e = 0;
        if(q[d] < b->lo[d]) e = b->lo[d] - q[d];
        else if(q[d] > b->hi[d]) e = q[d] - b->hi[d];
        r2 += e * e;
    }
    return r2;
}

/* Depth-first, nearest child first, pruned by the current k-th distance (the search of
 * HBT's GeoTree_t::NearestNeighbours used by subhalo_reassign_gas.cpp:140-150). */
static void
knn_rec(const struct SubTree * t, int64_t n, const double q[3], struct KnnList * L)
{
    if(L->n > 0 && L->n == L->k && box_dist2(&t->box[n], q) > L->d2[L->n - 1])
        return;
    if(t->sk[n].skip == n + 1) {
        int64_t j;
        for(j = t->sk[n].first; j < t->sk[n].first + t->sk[n].count; j++) {
            const double ex = t->src[j].x[0] - q[0];
            const double ey = t->src[j].x[1] - q[1];
            const double ez = t->src[j].x[2] - q[2];
            knn_insert(L, ex * ex + ey * ey + ez * ez, t->src[j].p);
        }
        return;
    }
    int64_t ch[8];
    double cd[8];
    int nc = 0, a, b;
    int64_t c;
    for(c = n + 1; c < t->sk[n].skip && nc < 8; c = t->sk[c].skip) {
        ch[nc] = c;
        cd[nc] = box_dist2(&t->box[c], q);
        nc++;
    }
    for(a = 1; a < nc; a++)
        for(b = a; b > 0 && cd[b] < cd[b - 1]; b--) {
            const double td = cd[b]; cd[b] = cd[b - 1]; cd[b - 1] = td;
            const int64_t tc = ch[b]; ch[b] = ch[b - 1]; ch[b - 1] = tc;
        }
    for(a = 0; a < nc; a++)
        knn_rec(t, ch[a], q, L);
}

void
subhalo_knn_tracers(const int64_t * query, int64_t nq, const int64_t * tr, int64_t nt, int k,
                    int64_t * out, double BoxSize, struct SubhaloArena * arena)
{
    const size_t mark = arena->used;
    int64_t i;
    int d;
    if(k > SUBHALO_KNN_MAXK)
        endrun(1, "Subhalo kernel: k = %d nearest neighbours exceeds SUBHALO_KNN_MAXK = %d\n", k, SUBHALO_KNN_MAXK);
    if(k <= 0)
        return;
    if(nt <= 0) {
        for(i = 0; i < nq * k; i++)
            out[i] = -1;
        return;
    }
    double origin[3];
    for(d = 0; d < 3; d++)
        origin[d] = P[tr[0]].Pos[d];
    struct SubTree tree = {0};
    tree.src = (struct SubSrc *) arena_need(arena, sizeof(struct SubSrc) * nt, "neighbour sources");
    for(i = 0; i < nt; i++) {
        tree.src[i].p = tr[i];
        tree.src[i].m = 1;
        for(d = 0; d < 3; d++)
            tree.src[i].x[d] = NEAREST(P[tr[i]].Pos[d] - origin[d], BoxSize);
    }
    tree.ns = nt;
    tree_build_skeleton(&tree, sizeof(struct SubBox), arena);
    tree.box = (struct SubBox *) arena_need(arena, sizeof(struct SubBox) * tree.nn, "neighbour boxes");
    int64_t n;
    for(n = tree.nn - 1; n >= 0; n--) {
        struct SubBox * b = &tree.box[n];
        for(d = 0; d < 3; d++) {
            b->lo[d] = DBL_MAX;
            b->hi[d] = -DBL_MAX;
        }
        if(tree.sk[n].skip == n + 1) {
            int64_t j;
            for(j = tree.sk[n].first; j < tree.sk[n].first + tree.sk[n].count; j++)
                for(d = 0; d < 3; d++) {
                    if(tree.src[j].x[d] < b->lo[d]) b->lo[d] = tree.src[j].x[d];
                    if(tree.src[j].x[d] > b->hi[d]) b->hi[d] = tree.src[j].x[d];
                }
        }
        else {
            int64_t ch;
            for(ch = n + 1; ch < tree.sk[n].skip; ch = tree.sk[ch].skip)
                for(d = 0; d < 3; d++) {
                    if(tree.box[ch].lo[d] < b->lo[d]) b->lo[d] = tree.box[ch].lo[d];
                    if(tree.box[ch].hi[d] > b->hi[d]) b->hi[d] = tree.box[ch].hi[d];
                }
        }
    }
    const int par = nq > 4096 && !omp_in_parallel();
    #pragma omp parallel for schedule(dynamic, 256) if(par)
    for(i = 0; i < nq; i++) {
        struct KnnList L;
        double q[3];
        int dd, j;
        L.n = 0;
        L.k = k;
        for(dd = 0; dd < 3; dd++)
            q[dd] = NEAREST(P[query[i]].Pos[dd] - origin[dd], BoxSize);
        knn_rec(&tree, 0, q, &L);
        for(j = 0; j < k; j++)
            out[i * k + j] = j < L.n ? L.p[j] : -1;
    }
    arena->used = mark;
}
