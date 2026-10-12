#ifndef SIDM_DIRECT_H
#define SIDM_DIRECT_H
#include <mpi.h>
#include <stdint.h>
#include <stddef.h>

/* Compact state for standalone operator tests; production uses a native view. */
struct sidm_particle { uint64_t id; double mass, v[3]; };
/* Debug-only owner journal. Release needs only validated_mass and does not
 * allocate, mark, copy, or scan endpoint audit storage. */
struct sidm_journal_entry { int index; double before[3]; };
struct sidm_journal {
    struct sidm_journal_entry *entries;
    unsigned char *marked;
    int capacity, count;
    double validated_mass;
};
#define SIDM_DIRECT_MAX_NEIGHBORS 128
/* Retained by the read-only support walk. Ownership is valid for this event
 * only; rebuild after migration. No per-edge velocity is stored or sent. */
struct sidm_direct_neighbor {
    /* The collision operator consumes r2 and replaces it with its prepared
     * kernel/exposure coefficient. Rebuild geometry before the next call.
     * This reuses the support heap without another K*N scratch array. */
    union { double r2, coefficient; };
    uint64_t id;
    int owner, index;
};
/* Aligned uint64 IDs and double[3] velocities in stable owner-local storage.
 * A strided view avoids copying the full native particle population. The
 * caller validates mass/velocity and excludes garbage from source/neighbor
 * lists. Offsets and stride must preserve their C types' alignment. */
struct sidm_direct_particles { void *base; size_t stride, id_offset, velocity_offset; };
struct sidm_direct_source {
    int index, n;
    struct sidm_direct_neighbor *neighbors;
    double h, dt, max_sigma_v;
    uint64_t seed;
};
struct sidm_direct_stats {
    /* Diagnostics are populated only with -DSIDMDEBUG. Release leaves zero,
     * avoiding per-pair counters and diagnostic collectives. Source rates
     * and accepted particle updates are identical in both builds. */
    uint64_t pairs, trials, events, subdivided_pairs, source_packets, bytes;
    uint64_t threaded_pairs, recomputed_pairs;
    double max_probability, max_trial_probability, sum_probability;
    long double delta_k, delta_p;
    uint64_t journaled_particles;
    /* Rank-local seconds, exclusive categories. MPI includes waiting; these
     * are not network-only costs and should not be summed over ranks. */
    double prepare_seconds, route_seconds, pack_seconds, mpi_seconds,
        pair_seconds, audit_seconds;
    /* Read-only prediction is included in pair_seconds. */
    double predict_seconds;
};

/* Collective, source-owned directed single kernels. Sorts neighbor lists by
 * owner/ID, groups all neighbors of one source on each destination, and sends
 * one fresh source state there. Disjoint rank pairs execute A->B, return, then
 * B->A. Bounded packets; no global particle gather or probability treewalk.
 * Velocities are changed immediately outside retryable treewalk callbacks.
 * Read-only rates and first random draws are prepared in threaded batches.
 * Applying updates stays ordered; any endpoint changed since preparation
 * forces a fresh pair evaluation. This preserves the serial trajectory.
 * Wire messages may contain several evaluation batches; a source occurs
 * only once per destination. Bounded wire storage and routing scratch are
 * retained between calls; source-group storage is released after each call.
 *
 * Each pair uses its CURRENT relative speed. If its exposure probability is
 * above pmax, split that pair's full exposure into ceil(P/pmax) Bernoulli
 * trials, selected BEFORE any draw. Equal-mass elastic scattering preserves
 * that pair's relative speed. This is a pair-operator split, not a claim of
 * converged interleaving with other neighbors; subdivision is reported.
 * pmax must be in (0,1]. No speed ceiling, clipping, or outcome-dependent retry.
 *
 * Neighbor arrays must be disjoint between sources (sorted in parallel).
 * With SIDMDEBUG, accepted updates are journaled before their first mutation.
 * The caller supplies identical parameters on all ranks and validates all
 * initial particle state and fixed equal mass, and provides a journal with
 * validated_mass > 0. All sources, including
 * dt=0, measure rates. Failure invalidates the entire (possibly partial) step.
 */
int sidm_direct_scatter(struct sidm_particle *p, int nlocal,
    struct sidm_direct_source *sources, int nsource,
    double sigma0, double w, int vd, double pmax, MPI_Comm comm,
    struct sidm_journal *journal, struct sidm_direct_stats *stats,
    char error[256]);
int sidm_direct_scatter_view(struct sidm_direct_particles p, int nlocal,
    struct sidm_direct_source *sources, int nsource,
    double sigma0, double w, int vd, double pmax, MPI_Comm comm,
    struct sidm_journal *journal, struct sidm_direct_stats *stats,
    char error[256]);
#endif
