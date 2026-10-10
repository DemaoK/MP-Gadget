#ifndef SUBHALO_UNBIND_H
#define SUBHALO_UNBIND_H

/* Serial unbinding kernel of the subhalo finder (design 7.4, Phase 2a plan section 3).
 *
 * Port of HBT-HERONS: the monopole octree with geometric opening (src/gravity_tree.cpp), the
 * unbinding loop with subsampling, refinement, survival thresholds, frame update and count
 * convergence (src/subhalo_unbind.cpp:463-808), the core with non-tracer fill and the sinking
 * criterion (src/subhalo_merge.cpp:13-26,93-181) and the k-nearest tracer search of
 * src/subhalo_reassign_gas.cpp. Deliberate deviations: design D3 items 3-6.
 *
 * Every function here is serial and thread-safe (it runs inside OpenMP threads, one key per
 * thread); scratch comes only from the caller's arena (subhalo_arena_get), which is rewound to
 * its entry state on return, so nothing a function allocates survives it. Energy evaluation
 * loops are split over OpenMP threads only when called outside a parallel region (a key that is
 * processed alone); results do not depend on the thread count.
 *
 * Conventions (design 7.10): positions are P[i].Pos (translated comoving frame), relative vectors
 * NEAREST(dx, BoxSize); velocities are the internal P[i].Vel = a v_pec; energies per unit mass in
 * physical (km/s)^2:
 *   E_i   = 1/2 |Vel_i / a - RefVel / a + H a dx_i|^2 + phi_i (+ u_i for type 0 with WithThermal)
 *   dx_i  = NEAREST(Pos_i - RefPos)
 *   phi_i = -(G / a) sum_{j != i, Type_j in PotentialTypes} m_j W(|dx_ij|, h)
 * with W = grav_spline_potential (gravshort.h). The octree works in coordinates relative to the
 * frame (one NEAREST per particle), so pair distances are periodic minimum images for any
 * candidate smaller than half the box. */

#include <stdint.h>
#include "subhalo.h"
#include "subhalo_internal.h"

/* A candidate subhalo during processing: an index list into P (design 7.1). */
struct SubCandidate {
    int64_t TrackId;         /* provisional (< -1) for a newborn */
    int64_t * Index;         /* [N] indices into P; after subhalo_unbind: bound first in
                                ascending energy, then the unbound in ascending energy */
    double * Energy;         /* [N] specific energies parallel to Index (scratch) */
    int64_t N;               /* source size */
    int64_t Nbound;          /* bound prefix length */
    int64_t NboundTracer;
    double RefPos[3];        /* frame: translated comoving position; in: previous bound CoM,
                                out: final bound CoM */
    double RefVel[3];        /* frame: internal Vel units (a v_pec); in/out as above */
    double Mbound;
    int Iterations;
    int Flags;               /* SUBHALO_FLAG_ITERCAP */
    uint64_t Seed;           /* subsampling RNG seed, set by the caller from (TrackId, SnapNum) */
};

struct SubhaloUnbindParams {
    double atime, Hubble;    /* a and H(a) in internal units (hubble_function) */
    double G;                /* CP->GravInternal */
    double h;                /* FORCE_SOFTENING(), comoving */
    double BoxSize;
    int PotentialTypes, TracerTypes;
    int MinNumPart, MinNumTracerPart;
    int SampleSize;          /* SubhaloPotentialSampleSize, 0 = never subsample */
    int MaxIter;
    double OpenAngle, BoundMassPrecision;
    int WithThermal;
};

/* Types never subsampled when the potential is subsampled (HBT DoNotSubsampleParticleTypes,
 * config_parser.h:162: black holes): they always source the potential with their true mass. */
#define SUBHALO_NOSUBSAMPLE_TYPES (1 << 5)
/* Fraction of the bound set whose order is refined with exact masses after a subsampled loop
 * (HBT BoundFractionCenterRefinement, config_parser.h:140). */
#define SUBHALO_REFINE_FRACTION 0.1
/* Largest k accepted by subhalo_knn_tracers (the per-query heap lives on the thread stack). */
#define SUBHALO_KNN_MAXK 64

/* Arena bytes per particle of a candidate. The caller's Index (8) and Energy (8) arrays, plus
 * the kernel's transient scratch: a source record per potential-type particle (position,
 * mass, P index, Morton key: 48 bytes) and octree nodes (48 bytes each) in the space left;
 * the build typically needs 0.3 to 0.6 nodes per source with 8-particle leaves, and the kernel
 * widens the leaves until the tree fits in what remains, so 1 node per source keeps the
 * standard leaf size in all realistic cases. The energy sorts reuse the same scratch (16 bytes
 * per particle). 16 + 48 + 48 = 112. */
#define SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART 96
#define SUBHALO_UNBIND_BYTES_PER_PART (16 + SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART)

/* Unbind c in place. Arena scratch only (subhalo_arena_get; nodes, keys, sample lists).
 * Returns 1 if the candidate survives, 0 if it died (Nbound set to 0, Index order undefined
 * beyond "all particles present"). Deterministic for a given Seed and particle order.
 * On survival: Index[0, Nbound) holds the bound set in ascending Energy (except that, when the
 * potential was subsampled and Nbound > SampleSize, the max(SampleSize, 0.1 Nbound) most bound
 * are reordered by their exact-mass self-energy as HBT's RefineMostBoundParticle does, keeping
 * the loop energies: subhalo_final_pass supersedes both), Index[Nbound, N) the unbound in
 * ascending Energy (each with the energy of the iteration that removed it); RefPos/RefVel the
 * bound CoM frame; Mbound, NboundTracer, Iterations set; Flags |= SUBHALO_FLAG_ITERCAP when
 * MaxIter iterations did not converge. */
int subhalo_unbind(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena);

/* Recompute the energies of the bound prefix with exact masses and no subsampling (direct sum
 * for Nbound <= SampleSize, the candidate's own tree otherwise), in the final bound-CoM frame,
 * and sort the prefix by energy. Membership unchanged. Returns the number of prefix particles
 * with E >= 0 after the pass (logged by the caller).
 * The frame (RefPos, RefVel, Mbound) is recomputed from the prefix first, so the pass is also
 * valid after the caller edited the prefix; NboundTracer is recounted. */
int64_t subhalo_final_pass(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena);

/* Core of the SubhaloCoreSize most-bound tracers, filled with non-tracers when fewer are bound
 * (subhalo_merge.cpp:122-158): mass-weighted CorePos (translated comoving), CoreVel (Vel units),
 * and the dispersions SigmaX (comoving) and SigmaV (Vel units) used by the sinking test.
 * Uses Index[0, Nbound) in its current order. Nbound == 0: zeros. */
void subhalo_core(const struct SubCandidate * c, const struct SubhaloUnbindParams * up, int CoreSize,
                  double CorePos[3], double CoreVel[3], double * SigmaX, double * SigmaV);

/* HBT sinking test of satellite s against ancestor a using the cores above
 * (|dx| / SigmaX_a + |dv| / SigmaV_a < 2, or the symmetric form when symmetric != 0). */
int subhalo_sink_test(const double sPos[3], const double sVel[3], const double aPos[3], const double aVel[3],
                      double aSigmaX, double aSigmaV, double sSigmaX, double sSigmaV, int symmetric, double BoxSize);

/* For every particle in query[0..nq) find the k nearest among the tracer list tr[0..nt)
 * (indices into P), writing k indices per query into out (row-major). Octree-based.
 * Each row is sorted by distance, nearest first (ties by P index); if nt < k the row is padded
 * with -1. A query that is itself in tr finds itself at distance 0 (no self exclusion).
 * k <= SUBHALO_KNN_MAXK. Arena: 48 bytes per tracer plus the nodes in the space left. */
void subhalo_knn_tracers(const int64_t * query, int64_t nq, const int64_t * tr, int64_t nt, int k,
                         int64_t * out, double BoxSize, struct SubhaloArena * arena);

/* Energies: E = 0.5 |Vel/a - RefVel/a + H a dx|^2 + phi (+ u for gas with WithThermal), dx =
 * NEAREST(Pos - RefPos) comoving, phi = -(G / a) sum_j m_j W(|dx_ij|, h) over the potential
 * types, no self term; W = grav_spline_potential(r, h) (gravshort.h). */

/* ------------------------------------------------------------------ additions to the plan's interface */

/* Energies E[0..n) of the particles idx[0..n) in the frame (RefPos, RefVel), the potential
 * sourced by the potential-type particles among src[0..ns) at their true masses (a target that
 * is also a source does not feel itself). Direct sum when the number of potential-type sources
 * is <= up->SampleSize, the octree with up->OpenAngle otherwise. idx and src may be the same
 * array. Used for evaluating energies of arbitrary lists (tails, tests). */
void subhalo_energies(const int64_t * idx, int64_t n, double * E, const int64_t * src, int64_t ns,
                      const double RefPos[3], const double RefVel[3], const struct SubhaloUnbindParams * up,
                      struct SubhaloArena * arena);

/* The specific thermal energy u of particle i that the kernel adds with WithThermal (type 0
 * only, 0 otherwise): the InternalEnergy block's conversion of SPHP(i).Entropy (petaio.c
 * GTInternalEnergy), physical (km/s)^2. */
double subhalo_thermal_energy(int64_t i, double atime);

/* ------------------------------------------------------------------ properties (subhalo_props.c) */

/* Fill the step 7g columns of rec from a candidate after the final pass (plan section 4):
 * Nbound, NboundTracer, NboundType, Mbound, MboundType, Flags |= c->Flags, CorePos/CoreVel/
 * CoreSigmaX/CoreSigmaV, MostBoundID/Pos/Vel (the most-bound tracer, else the most bound),
 * CoMPos/CoMVel (the final frame), Vmax/Rmax (HBT: all bound particles, radii from CorePos
 * floored at h / 2.8, Vmax = max sqrt(G M(<=r) / (a r))), Ekin, Epot, KinToPot. NboundExclusive,
 * NboundExclusiveType and NtailType are step 8's and untouched, as is every other column.
 * Positions in the physical frame (Offset removed, wrapped into [0, BoxSize)), comoving;
 * velocities physical peculiar km/s. Needs no scratch. */
void subhalo_props_hbt(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       int CoreSize, const double Offset[3]);

#endif
