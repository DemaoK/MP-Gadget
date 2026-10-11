#ifndef SUBHALO_UNBIND_H
#define SUBHALO_UNBIND_H

/* Unbinding kernel of the subhalo finder (design 7.4, Phase 2a plan section 3).
 *
 * Port of HBT-HERONS: the monopole octree with geometric opening (src/gravity_tree.cpp), the
 * unbinding loop with subsampling, refinement, survival thresholds, frame update and count
 * convergence (src/subhalo_unbind.cpp:463-808), the core with non-tracer fill and the sinking
 * criterion (src/subhalo_merge.cpp:13-26,93-181) and the k-nearest tracer search of
 * src/subhalo_reassign_gas.cpp. Deliberate deviations: design D3 items 3-6.
 *
 * Every function here is thread-safe (it runs inside OpenMP threads, one key per thread);
 * scratch comes only from the caller's arena (subhalo_arena_get), which is rewound to its entry
 * state on return, so nothing a function allocates survives it. A step on more than
 * SUBHALO_UNBIND_PAR_MIN entries that is called outside a parallel region (a key processed alone)
 * splits its work over the OpenMP threads: see the "Phase 5b: intra-key threading" block at the
 * end of this file. Results never depend on the thread count.
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
    /* 1 - SubhaloFinalPass: 0 = the exact final pass, 1 = the sampled one (subhalo_final_pass).
     * Zero initialisation gives the exact pass (historical behaviour); the production default
     * SubhaloFinalPass = 0 (sampled) comes from gadget/params.c. */
    int SampledFinalPass;
    /* Tracers whose order the sampled pass takes from exact energies (the caller's
     * max(SubhaloCoreSize, SubhaloMinNumTracerPart): the core and the reserved tracers, which
     * include the host-finding ones); SUBHALO_FINAL_REFINE_MIN if smaller. */
    int FinalRefineTracers;
};

/* Types never subsampled when the potential is subsampled (HBT DoNotSubsampleParticleTypes,
 * config_parser.h:162: black holes): they always source the potential with their true mass. */
#define SUBHALO_NOSUBSAMPLE_TYPES (1 << 5)
/* Fraction of the bound set whose order is refined with exact masses after a subsampled loop
 * (HBT BoundFractionCenterRefinement, config_parser.h:140). */
#define SUBHALO_REFINE_FRACTION 0.1
/* The exact head of the sampled final pass (subhalo_final_pass): Kt = max(SUBHALO_FINAL_REFINE_MIN,
 * FinalRefineTracers) tracers must come from exact energies; the head holds
 * SUBHALO_FINAL_REFINE_MARGIN * Kt tracers so that they do. */
#define SUBHALO_FINAL_REFINE_MIN 20
#define SUBHALO_FINAL_REFINE_MARGIN 5
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
 * valid after the caller edited the prefix; NboundTracer is recounted.
 *
 * Sampled final pass (up->SampledFinalPass, SubhaloFinalPass = 0; decision 42.4): HBT-HERONS's
 * own rule, which has no exact pass, plus an exact head. When up->SampleSize > 0 and
 * Nbound > SampleSize, the prefix keeps the order subhalo_unbind left it in (the last
 * iteration's subsampled energies, the max(SampleSize, 0.1 Nbound) most bound reordered by their
 * exact self-binding energy: RefineMostBoundParticle, subhalo_unbind.cpp:709-733), except its
 * head Index[0, K): with Kt = max(SUBHALO_FINAL_REFINE_MIN, up->FinalRefineTracers), K is the
 * shortest prefix holding the first T = SUBHALO_FINAL_REFINE_MARGIN * Kt tracers (and, with
 * fewer than T tracers bound, the first non-tracers subhalo_core fills with), at least T
 * particles. The head gets the exact pass's energies (the tree over the whole prefix in the
 * final frame, built exactly as the exact pass builds it, K walks: bitwise the exact pass's
 * values for those particles) and is sorted by them. So the most-bound tracer (MostBoundID),
 * the reserved tracers and the core (the first Kt tracers) equal the exact pass's only if the
 * true ones lie inside the selected head, the shortest prefix of the order after HBT's refinement
 * holding T = 5 x max(20, CoreSize, MinNumTracerPart) tracers (variable particle length); SampleSize
 * = 0 and the in-place path never sample. The rest of the order (BindRank beyond K) is the loop's.
 * Energy[0, Nbound) is then recomputed in the final frame with
 * the loop's estimator (the SampleSize-particle subsample of a seeded shuffle of the prefix,
 * masses upscaled per type), for the energy sums of the properties (Epot, SpinPeebles) and
 * NboundPositive only: it is not sorted and not used for the order. The caller must not have
 * edited the prefix since subhalo_unbind (subhalo_track.c does not). Otherwise (SampleSize == 0
 * or Nbound <= SampleSize) the exact pass runs, so a candidate below the sample size gets
 * bit-identical results under both options. Thread-count independent like the exact pass (the
 * scan and the shuffle are serial, the energies those of potential_energies, the sorts total
 * orders). Arena: the exact pass's tree, then 8 bytes per prefix particle for the shuffled copy
 * plus the subsample's sources and tree, within SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART. */
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

/* Arena bytes per bound particle that subhalo_props_hbt, subhalo_props_rockstar and
 * subhalo_props_deferred take (the radial order: radius and P index). */
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
 * in the run's unit, velocities km/s (subhalo_vel_to_kms), J and JVir in physical kpc/h km/s
 * (the run's length unit converted with subhalo_get_units; the spins are unit-free). Scratch: SUBHALO_PROPS_BYTES_PER_PART * Nbound from the arena, rewound on return;
 * endrun if the arena is short. Offset is unused (the core is in the translated frame). */
void subhalo_props_rockstar(struct SubTrack * rec, const struct SubCandidate * c,
                            const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO],
                            int CoreSize, const double Offset[3], struct SubhaloArena * arena);

/* Rockstar f_to_c (nfw.c:19-32): the NFW concentration with c / (ln(1 + c) - c / (1 + c)) = f,
 * -1 if the Newton iteration fails (subhalo_props_rockstar and the in-place recipes). */
double subhalo_klypin_f_to_c(double f);

/* ------------------------------------------------------------------ Phase 5 properties (subhalo_props.c) */

/* Rockstar's NFW shape fit (nfw.c:51-144) and the velocity trio (properties.c:199-244, 417-458)
 * on the bound set (Phase 5 plan section 1 item 1): Rs, RockstarCoreVel, BulkVel, Vrms of rec.
 * The apertures are Rockstar's (decision 44, reversing decision 36a and design D4's "crossing
 * particle included" for these four columns): with the radially sorted bound particles (radii
 * from the core of subhalo_core, floored at GravitySoftening = h / 2.8, ties by ID) and kc the
 * outermost virial crossing (the rule of subhalo_props_rockstar; Mvir = M(<= r_kc) and Rvir =
 * (3 Mvir / (4 pi rho[0]))^(1/3), as MBoundSO[vir] / RBoundSO[vir], which keep the crossing
 * particle), the aperture of all four is the Nap = kc particles inside the crossing particle
 * (properties.c's j < dens_tot), of mass Mfit = M(<= r_(kc-1)) for the fit (decision 47.1: one
 * consistent off-by-one; nfw.c's i < total_p - 1 with total_p = dens_tot would also drop the
 * particle before the crossing, see the deviations below).
 *  - BulkVel: the mass-weighted mean Vel of the velocity aperture; Vrms: sqrt(sum_d sum m (v_d -
 *    BulkVel_d)^2 / M) over it, M its own mass (no Hubble term, as Rockstar; Rockstar divides
 *    both sums by Mvir, the mass with the crossing particle, a 1 / Nvir bias not copied).
 *  - RockstarCoreVel: the mass-weighted mean Vel of the velocity aperture's particles with r <
 *    0.1 Rvir, or of its innermost min(SUBHALO_ROCKSTAR_CORE_MINPART, Nap) when fewer are inside.
 *  - Rs: the fit of subhalo_nfw_fit on the equal-mass bins of the aperture
 *    (subhalo_nfw_bin_plan with Nap, Mfit and its largest mass, subhalo_nfw_threshold,
 *    subhalo_nfw_edge; the last bin's outer neighbour rnext may be the crossing particle) from
 *    the Klypin start (subhalo_klypin_rs, Mvir and Rvir); -1 when Nap = kc <
 *    SUBHALO_NFW_MINPART, i.e. the crossing is among the innermost 100 bound particles, so
 *    always below 101 bound particles (Rockstar falls back to the Klypin value there; ours is
 *    RsKlypin) or the fit fails.
 * All four stay -1 without a virial crossing, when the crossing is the innermost bound particle
 * (kc = 0: an empty aperture) or Nbound == 0; with kc = 1 the velocity trio is the innermost
 * particle's (Vrms 0) and Rs is -1. Velocities physical peculiar
 * km/s (subhalo_vel_to_kms(a) on Vel), Rs comoving in the run's length unit. The core is
 * recomputed (as subhalo_props_rockstar does), so rec's other columns are not read. Scratch:
 * SUBHALO_PROPS_BYTES_PER_PART * Nbound from the arena, rewound on return; endrun if short.
 * subhalo_props_rockstar fills the same four columns with the same arithmetic on its own radial
 * order (one sort per subhalo; the production path, subhalo_track.c); this standalone entry
 * gives bit-identical values (the unit tests compare the two).
 *
 * Serial and distributed (Phase 5 audit R4): subhalo_collective.c computes the same four columns
 * for an in-place candidate from its distributed radial ordered set. Every accumulation of the
 * recipe is a compensated (Neumaier) double sum in both modes (subhalo_ksum_*): the cumulative
 * mass M(<= r) that sets the aperture, Mvir and the bin closing (the distributed one merges
 * per-rank pairs in rank order through the prefix scan), the aperture's mass and velocity
 * moments and the core's; the bin plan uses the same reduced quantities in both (Mfit, the
 * cumulative mass at the fit aperture's last particle, and the largest mass), the bin edges, the fit and the velocity trio the same functions
 * (subhalo_nfw_edge, subhalo_nfw_fit, subhalo_deferred_velocities). Guarantee: for an identical
 * bound set, core and radial order, the two modes give identical columns whenever the
 * compensated sums are exact, in particular for equal particle masses (DM-only runs), where
 * every cumulative mass is the correctly rounded k m in both modes whatever the rank
 * decomposition, and for any masses that are small multiples of a common quantum; for other
 * masses they agree to about one rounding of the final sums, and the 1e-9 margin of
 * subhalo_nfw_threshold keeps the bin edges where they are unless a cumulative mass lies within
 * 1e-9 of a threshold (test_collective_props_identity checks both cases across three layouts).
 * The bound set and core themselves come from different trees in the two modes and may differ.
 *
 * Deviations from Rockstar's recipe (beyond the bound set of HBT membership and the centre, the
 * 20-most-bound-tracer core instead of Rockstar's 6D-subgroup centre):
 *  - the bin edge interpolation (subhalo_nfw_edge) is applied in the run's length unit; nfw.c
 *    subtracts the excess-mass interpolation computed from the radii in Mpc/h from a midpoint it
 *    has converted to kpc/h, an upstream unit bug that effectively drops the correction (for radii
 *    0.01, 0.02, 0.03 Mpc/h and an excess of half the closing mass: Rockstar 24.995 kpc/h, here
 *    20 kpc/h; audit R5), so part of any remaining Rs difference against Rockstar comes from the
 *    edges, not from membership alone;
 *  - the fit (subhalo_nfw_fit) is in double and returns -1 for degenerate or nonfinite input
 *    instead of Rockstar's start radius or a NaN (audit R6);
 *  - the bound-set radii are floored at the softening in the bin edges (nfw.c uses the unfloored
 *    radii; the bins below 3 softenings weigh 0.1 in both, no measurable effect);
 *  - BulkVel / Vrms normalised by the velocity aperture's own mass (above);
 *  - the fit's aperture excludes only the crossing particle, as the velocity aperture
 *    (properties.c); nfw.c's i < total_p - 1 on total_p = dens_tot also drops the particle
 *    before it (decision 47.1: one consistent off-by-one, one more fit particle than Rockstar,
 *    so Rs is defined from kc = 100 where Rockstar needs 101 inside particles).
 * The SO-sphere versions (RsSO, RsKlypinSO, VmaxSO, RmaxSO of the field SO's pass 3, decision 44,
 * subhalo_so.h) use the same fit and Klypin functions on every particle inside RSO[vir] (Rockstar's
 * host set: satellites, unbound and non-FOF particles included) with these further deviations:
 *  - the aperture is the interpolated field-SO RSO[vir] (all particles with r <= RSO[vir]), not
 *    the particle crossing, and Rockstar's off-by-one exclusions do not apply;
 *  - the profile is a histogram (SUBHALO_SO_NPROF log bins over 3 decades): the equal-mass bin
 *    edges are where the histogram's M(<r), interpolated linearly in ln r inside a profile bin,
 *    reaches each closing mass (the histogram form of nfw.c's excess-mass edge interpolation, in
 *    place of the particle radii); the edges are unfloored;
 *  - VmaxSO / RmaxSO are the maximum over the profile's bin edges (radii floored at the
 *    softening), resolution one bin (0.025 dex in r); RsKlypinSO starts from MSO[vir], RSO[vir]. */
void subhalo_props_deferred(struct SubTrack * rec, const struct SubCandidate * c,
                            const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO],
                            int CoreSize, struct SubhaloArena * arena);

/* The pieces both modes share (subhalo_collective.c replicates the fit per label from reduced
 * bin edges), so that the serial and the distributed Rs come from the same arithmetic. */
#define SUBHALO_NFW_MAXBINS 50        /* MAX_SCALE_BINS */
#define SUBHALO_NFW_MINPERBIN 15      /* MIN_PART_PER_BIN (also Rockstar's iteration cap) */
#define SUBHALO_NFW_MINPART 100       /* MIN_SCALE_PART */
#define SUBHALO_ROCKSTAR_CORE_MINPART 100
/* The bin plan of calc_scale_radius for nfit particles of total mass Mfit and largest particle
 * mass mmax: returns the number of bins (0 when nfit < SUBHALO_NFW_MINPART), *mpbin the mass per
 * bin (at least mmax). */
int subhalo_nfw_bin_plan(int64_t nfit, double Mfit, double mmax, double * mpbin);
/* The cumulative mass at which bin b (1-based) closes: the first particle of the radial order
 * with M(<= r) >= this closes it (Rockstar's running subtraction, with a 1e-9 relative margin
 * against the rounding of the cumulative sums; with the compensated sums of both modes the
 * decision is identical whenever the sums are exact, see above). */
double subhalo_nfw_threshold(int b, double mpbin);
/* The outer edge of bin b closed by the particle at r (mass m, cumulative mass Mc) between its
 * neighbours rprev (ignored when has_prev == 0) and rnext (r when it is the last bound particle):
 * 0.5 (r + rnext) - (Mc - b mpbin) / m (r - rprev) (nfw.c:126-132, both terms in the run's
 * length unit: the intended interpolation, not nfw.c's mixed units; see the deviations above). */
double subhalo_nfw_edge(int b, double mpbin, double rprev, int has_prev, double r, double rnext, double Mc, double m);
/* calc_scale_from_bins (nfw.c:64-99) in double: Newton on the chi^2 of the bin masses from rs0
 * (replaced by the middle edge when outside (edge[0], edge[nbins])), at most
 * SUBHALO_NFW_MINPERBIN steps; bins whose inner edge is below wcut weigh 0.1. edge[0] = 0.
 * Returns -1 when nbins < 3, an edge is not finite, edge[nbins] <= edge[0], the start or a step
 * gives a nonfinite chi^2, a zero or nonfinite probe step or a nonfinite derivative (repeated
 * edges at the start radius), or the result is not a finite positive radius (audit R6);
 * Rockstar's finite stopping rule (a zero Newton move, the 0.5 % chi^2 change, the step cap) is
 * kept. */
double subhalo_nfw_fit(double rs0, const double * edge, int nbins, double wcut);
/* Rockstar estimate_scale_radius (nfw.c:34-45) from the bound virial mass and radius and the
 * VmaxVir pair (best = max M / r, rbest its radius, comoving): the RsKlypin column. */
double subhalo_klypin_rs(double Rvir, double Mvir, double best, double rbest);

/* Group C baryonic sums over the bound set, add_particle_to_group style (fof.c): SFR, HeIII mass
 * (gas), BH subgrid mass and Mdot, MostMassiveBHID (largest BHP(i).Mass, ties to the smaller ID;
 * (MyIDType) -1 without a bound BH); with MetalReturnOn the gas and stellar metal masses and
 * element masses, else those stay -1. Nbound == 0: every column -1. No scratch. */
void subhalo_props_baryons(struct SubTrack * rec, const struct SubCandidate * c, int MetalReturnOn);

/* ------------------------------------------------------------------ Phase 5b: intra-key threading (subhalo_unbind.c) */

/* Parallel mode of the kernel (Phase 5 plan section 4 item 6). A step on more than
 * SUBHALO_UNBIND_PAR_MIN elements (sources, targets, list entries) called outside an OpenMP
 * parallel region with more than one thread available (a key processed alone by
 * subhalo_run_serial_keys) splits over the threads: the source gather, the extents, the Morton
 * keys, the sorts (a parallel merge sort, its merge buffer from the arena; the serial introsort
 * when the arena has no room for it), the octree (subtrees built and their moments or boxes
 * computed in parallel, the top nodes serially), the energies, the bound/unbound partition, the
 * frame and the counts. Every result (Index order, Energy, RefPos/RefVel, Mbound, the
 * subsample, the neighbour lists) is defined independently of the thread count:
 *  - the floating sums (the per-type masses of the subsampling, bound_frame) run over fixed
 *    chunks of max(CHUNK_MIN, ceil(n / 256)) list entries, combined in chunk order;
 *  - sorts and trees produce the unique result of their strict total order (ties by ID);
 *  - the bound/unbound partition of a list of more than CHUNK_MIN entries is stable (both parts
 *    keep their relative order; 16 bytes per entry of arena scratch, within
 *    SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART); at most CHUNK_MIN entries keep the swap loop of
 *    HEAD 9e60f168, whose unbound order differs (decision 34; visible only in a dead
 *    candidate's list).
 * A candidate whose every step is at most CHUNK_MIN elements is bit-identical to HEAD 9e60f168.
 * The seeded shuffle, the subsample mass sum and the early-stopping tracer count of the
 * survival test stay serial.
 *
 * Two thresholds (decision 42.5, second round). SUBHALO_UNBIND_CHUNK_MIN = 16384 defines results
 * (the chunks of the floating sums and of the gathers, the partition's switch) and must not
 * change. SUBHALO_UNBIND_PAR_MIN only decides which steps share their work out; any value gives
 * the same results, because every step it gates gives the serial result at any thread count:
 *  - the merge sorts (energy order, refinement order, Morton order): the unique order of a
 *    strict total order, whatever the blocks and the merges;
 *  - the gravity and neighbour trees: the plan's top nodes and subtree tasks put every node at
 *    the index of the serial build with the same content (any task size); each node's moment or
 *    box is its own serial sum over its children or sources, as in the serial pass;
 *  - the Morton keys, the copies around the sorts and the refinement, the extents (min/max),
 *    the tracer and positive-energy counts (integers): exact in any split;
 *  - the energy and neighbour-query loops (each element independent) thread above
 *    min(4096, PAR_MIN) elements, in dynamic chunks of n / (4 threads) clamped to [16, 256];
 *  - the chunked steps (subsampling masses and source gather, bound_frame) are a single chunk
 *    for n <= CHUNK_MIN, i.e. the serial loop, and then run serially; the partition of at most
 *    CHUNK_MIN entries is the swap loop.
 * PAR_MIN = 512 (was 16384, which left every step but the energy loops of more than 4096
 * targets serial below 16384): measured with the timing harness of test_subhalo_unbind on r30
 * (4 cores x 2 hyperthreads), host + satellite candidates, SampleSize 1000, at 8 threads the
 * candidates of 1000 / 2000 / 4000 / 14174 particles go from 0.97x / 0.94x / 0.99x / 3.9x to
 * 3.3x / 4.1x / 4.2x / 4.5x of their 1-thread time (with the adaptive loop chunks); 256 gains
 * nothing more, 1024 leaves the subsample tree (~1000 sources) and the candidates below 1024
 * serial (halofinder_parmin_handoff.md). subhalo_unbind_set_par_min changes PAR_MIN at run
 * time (tests and the timing harness only; called outside any parallel region; n <= 0
 * restores the default). */
#define SUBHALO_UNBIND_PAR_MIN 512
#define SUBHALO_UNBIND_CHUNK_MIN 16384
void subhalo_unbind_set_par_min(int64_t n);
int64_t subhalo_unbind_get_par_min(void);
/* The dynamic chunk of the energy and neighbour-query loops (decision 42.5, second round):
 * nonzero (the default) n / (4 threads) clamped to [16, 256], zero the fixed 256 of Phase 5b.
 * Scheduling only, results never depend on it; tests and the timing harness only. */
void subhalo_unbind_set_loop_chunk(int adaptive);

#endif
