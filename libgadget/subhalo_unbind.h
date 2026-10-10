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
 * with W = grav_spline_potential (gravshort.h). Every pair displacement is a nearest image,
 * NEAREST(x_j - x_i), taken at each interaction as HBT does (gravity_tree.cpp:110-135): direct
 * sums, tree leaves, node monopoles and the opening test, and the k-nearest search (point
 * distances and periodic box bounds). The trees are built in coordinates relative to one origin
 * (NEAREST(Pos - origin)) only as a layout convenience; no distance is taken from those
 * coordinates without re-wrapping, so the results hold for sources of any extent.
 *
 * Ties (design 7.10): every ordering is broken by the particle ID (P[i].ID) and only then by the
 * P index, so results do not depend on the order of P: energy sorts, the most-bound refinement,
 * equal Morton keys in the trees, the radial order of the properties and neighbour distances. */

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
    /* Diagnostics (design 7.4, logged by the driver; written by the kernel, never read by it):
     * ConvergenceRatio is Nbound / Nlast of the last iteration subhalo_unbind ran (>=
     * BoundMassPrecision when it converged, below it when the cap was hit with ITERCAP set; the
     * ratio at the kill for a candidate that died; 0 if no iteration ran). NboundPositive is the
     * number of bound-prefix particles with E >= 0 after subhalo_final_pass (its return value),
     * -1 until a final pass ran. */
    double ConvergenceRatio;
    int64_t NboundPositive;
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
/* Largest k accepted by subhalo_knn_query (the per-query list lives on the thread stack). */
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
 * bound CoM frame; Mbound, NboundTracer, Iterations, ConvergenceRatio set; NboundPositive = -1;
 * Flags |= SUBHALO_FLAG_ITERCAP when MaxIter iterations did not converge (ConvergenceRatio is
 * then the Nbound / Nlast at the cap). */
int subhalo_unbind(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena);

/* Recompute the energies of the bound prefix with exact masses and no subsampling (direct sum
 * for Nbound <= SampleSize, the candidate's own tree otherwise), in the final bound-CoM frame,
 * and sort the prefix by energy. Membership unchanged. Returns the number of prefix particles
 * with E >= 0 after the pass (logged by the caller), also stored in c->NboundPositive.
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

/* k nearest tracers (ReassignParticles, subhalo_reassign_gas.cpp:131-150): one read-only
 * octree over the tracer list, built once per key and queried in as many batches as the caller
 * likes (HBT builds one tree per group).
 *
 * subhalo_knn_build: the tree over tr[0..nt) (indices into P; nt may be 0), allocated in the
 * arena where it lives until the caller rewinds the arena below it. The tree holds its own copy
 * of the tracer indices and positions, so tr may be reused after the build. Arena: 48 bytes per
 * tracer plus the nodes (64 bytes each, about 0.3 to 0.6 per tracer) in the space left, the
 * leaves widened until they fit; endrun if even one node does not fit. A caller that needs
 * more arena after the build (query and output buffers) should take it before the build, which
 * may use the whole remainder for the nodes when the arena is tight. Like every kernel function,
 * one arena per thread.
 *
 * subhalo_knn_query: for every particle in query[0..nq) the k nearest tracers of t, k indices
 * per query into out (row-major). Each row is sorted by periodic distance, nearest first, ties
 * by P[].ID and then P index; if the tree has fewer than k tracers the row is padded with -1.
 * A query that is itself a tracer finds itself at distance 0 (no self exclusion).
 * 1 <= k <= SUBHALO_KNN_MAXK. Read-only on t, no scratch: concurrent queries on one tree are
 * safe; split over OpenMP threads when nq > 4096 and not already in a parallel region.
 *
 * subhalo_knn_build_count: number of trees built by this process since start (all threads),
 * for the tests' one-build-per-key check.
 *
 * subhalo_knn_tracers: build + query + rewind, for one-shot use. */
struct SubhaloKnnTree;
struct SubhaloKnnTree * subhalo_knn_build(const int64_t * tr, int64_t nt, double BoxSize, struct SubhaloArena * arena);
void subhalo_knn_query(const struct SubhaloKnnTree * t, const int64_t * query, int64_t nq, int k, int64_t * out);
int64_t subhalo_knn_build_count(void);
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

/* Arena bytes per bound particle that subhalo_props_hbt and subhalo_props_rockstar take (the
 * radial order: radius and P index). */
#define SUBHALO_PROPS_BYTES_PER_PART 16

/* Fill the step 7g columns of rec from a candidate after the final pass (plan section 4):
 * Nbound, NboundTracer, NboundType, Mbound, MboundType, Flags |= c->Flags, CorePos/CoreVel/
 * CoreSigmaX/CoreSigmaV, MostBoundID/Pos/Vel (the most-bound tracer, else the most bound),
 * CoMPos/CoMVel (the final frame), Vmax/Rmax (HBT CalculateProfileProperties: all bound
 * particles sorted by radius from CorePos floored at h / 2.8; Vmax = max over radii r of
 * sqrt(G M(<=r) / (a r)), exact, Rmax the smallest radius attaining it), Ekin, Epot, KinToPot.
 * NboundExclusive, NboundExclusiveType and NtailType are step 8's and untouched, as is every
 * other column. Positions in the physical frame (Offset removed, wrapped into [0, BoxSize)),
 * comoving; velocities physical peculiar km/s. Scratch: SUBHALO_PROPS_BYTES_PER_PART * Nbound
 * from the arena, rewound on return; endrun if the arena is short. */
void subhalo_props_hbt(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       int CoreSize, const double Offset[3], struct SubhaloArena * arena);

/* ------------------------------------------------------------------ Phase 3 properties (subhalo_props.c) */

/* Comoving mass-density thresholds at a, in the order vir, 200b, 200c, 500c, 2500c (design 7.10,
 * groupies.c:63-103): rho_crit_c(a) = 3 H(a)^2 a^3 / (8 pi G), rho_mean_c = Omega0 3 H0^2 /
 * (8 pi G), x = Omega_m(a) - 1 with Omega_m(a) = Omega0 a^-3 (H0 / H(a))^2, Delta_vir = 18 pi^2 +
 * 82 x - 39 x^2 relative to critical; H(a) = hubble_function, H0 = CP->Hubble, G = GravInternal. */
void subhalo_so_thresholds(const Cosmology * CP, double atime, double rho[SUBHALO_NSO]);

/* The Rockstar columns of rec (Phase 3 plan section 3) from a candidate after the final pass
 * (Index[0, Nbound) bound in final-pass energy order, Energy the final-pass energies, RefPos/
 * RefVel the final bound-CoM frame, as subhalo_props_hbt takes it): MBoundSO, RBoundSO, VmaxVir,
 * RmaxVir, Rhalf, Renclose, J, JVir, SpinPeebles, SpinBullock, Xoff, Voff, RsKlypin, BtoA, CtoA,
 * AxisA, BtoA500, CtoA500, AxisA500, InertiaTensor. Every one of these is set to -1 first and
 * stays -1 when undefined (no crossing, too few particles, c->Energy NULL for SpinPeebles).
 * rho from subhalo_so_thresholds at up->atime. The core is recomputed with subhalo_core (the
 * same CorePos/CoreVel as subhalo_props_hbt), so rec's other columns are not read and the call
 * may come before or after the caller's unit conversion of the HBT columns. Lengths comoving
 * in the run's unit, velocities km/s (subhalo_vel_to_kms), J and JVir physical length unit x
 * km/s. Scratch: SUBHALO_PROPS_BYTES_PER_PART * Nbound from the arena, rewound on return;
 * endrun if the arena is short. Offset is unused (the core is in the translated frame). */
void subhalo_props_rockstar(struct SubTrack * rec, const struct SubCandidate * c,
                            const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO],
                            int CoreSize, const double Offset[3], struct SubhaloArena * arena);

/* Group C baryonic sums over the bound set, add_particle_to_group style (fof.c): SFR, HeIII mass
 * (gas), BH subgrid mass and Mdot, MostMassiveBHID (largest BHP(i).Mass, ties to the smaller ID;
 * (MyIDType) -1 without a bound BH); with MetalReturnOn the gas and stellar metal masses and
 * element masses, else those stay -1. Nbound == 0: every column -1. No scratch. */
void subhalo_props_baryons(struct SubTrack * rec, const struct SubCandidate * c, int MetalReturnOn);

#endif
