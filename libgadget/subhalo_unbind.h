#ifndef SUBHALO_UNBIND_H
#define SUBHALO_UNBIND_H

/* Unbinding kernel and property interface of the subhalo finder (design 7.4, 7.5, D3 items 3-6):
 * a port of HBT-HERONS's tree, unbinding loop, core, sinking test and k-nearest tracer search.
 *
 * Thread-safe (one key per thread); scratch comes only from the caller's arena and is rewound on
 * return (except the tree of subhalo_knn_build). A key run alone splits its larger steps over the threads (see the threading block at
 * the end); results never depend on the thread count.
 *
 * Conventions (design 7.10): Pos in the translated comoving frame, Vel = a v_pec, energies per
 * unit mass in physical (km/s)^2:
 *   E_i   = 1/2 |Vel_i / a - RefVel / a + H a dx_i|^2 + phi_i (+ u_i for type 0 with WithThermal)
 *   dx_i  = NEAREST(Pos_i - RefPos)
 *   phi_i = -(G / a) sum_{j != i, Type_j in PotentialTypes} m_j W(|dx_ij|, h)
 * with W = grav_spline_potential. Every pair displacement is a nearest image taken at each
 * interaction, as HBT does, so sources of any extent are handled. Every ordering breaks ties by particle ID, then by P index. */

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
    /* Diagnostics, written by the kernel and logged by the driver (design 7.4) */
    double ConvergenceRatio;  /* Nbound / Nlast of the last iteration (at the kill if it died), 0 if none ran */
    int64_t NboundPositive;   /* bound-prefix particles with E >= 0 after subhalo_final_pass, -1 before */
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
    /* 1 - SubhaloFinalPass: 1 = the sampled final pass, 0 (zero initialisation) = the exact one */
    int SampledFinalPass;
    /* tracers the sampled pass orders by exact energies: max(SubhaloCoreSize, SubhaloMinNumTracerPart) */
    int FinalRefineTracers;
};

/* Types never subsampled (HBT DoNotSubsampleParticleTypes: black holes); always at true mass. */
#define SUBHALO_NOSUBSAMPLE_TYPES (1 << 5)
/* Fraction of the bound set re-ordered with exact masses after a subsampled loop (HBT
 * BoundFractionCenterRefinement). */
#define SUBHALO_REFINE_FRACTION 0.1
/* Exact head of the sampled final pass: Kt = max(MIN, FinalRefineTracers) tracers must come from
 * exact energies, so the head holds MARGIN * Kt tracers. */
#define SUBHALO_FINAL_REFINE_MIN 20
#define SUBHALO_FINAL_REFINE_MARGIN 5
/* Largest k accepted by subhalo_knn_query (the per-query list lives on the thread stack). */
#define SUBHALO_KNN_MAXK 64

/* Arena bytes per candidate particle: the caller's Index and Energy (16), plus the kernel's
 * scratch: a 48-byte source record and room for one 48-byte node per source (a build needs 0.3 to
 * 0.6; the leaves widen until the tree fits). The energy sorts reuse the same scratch. */
#define SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART 96
#define SUBHALO_UNBIND_BYTES_PER_PART (16 + SUBHALO_UNBIND_SCRATCH_BYTES_PER_PART)

/* Unbind c in place. Returns 1 if it survives, 0 if it died (Nbound 0, Index a permutation).
 * On survival Index[0, Nbound) is the bound set in ascending Energy (after a subsampled loop the
 * max(SampleSize, 0.1 Nbound) most bound re-ordered by exact self-energy, as HBT), then the unbound
 * in ascending Energy; RefPos/RefVel the bound CoM frame; Flags |= ITERCAP when MaxIter is hit. */
int subhalo_unbind(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena);

/* Final pass over the bound prefix in its recomputed CoM frame: exact-mass energies, prefix sorted
 * by them; returns the count with E >= 0 (NboundPositive). Membership is unchanged. Sampled
 * (SampledFinalPass and Nbound > SampleSize > 0, decision 42.4): needs the unedited post-unbind
 * prefix; only its head of MARGIN * Kt tracers is sorted by exact energies, the rest keeps HBT's
 * loop order, and Energy is then the loop's estimator (not sorted; for the property sums only). */
int64_t subhalo_final_pass(struct SubCandidate * c, const struct SubhaloUnbindParams * up, struct SubhaloArena * arena);

/* Core of the CoreSize most-bound tracers of Index[0, Nbound), filled with non-tracers when fewer
 * (HBT): mass-weighted CorePos (comoving), CoreVel (Vel units) and the sinking test's SigmaX
 * (comoving), SigmaV (Vel units); zeros when Nbound == 0. */
void subhalo_core(const struct SubCandidate * c, const struct SubhaloUnbindParams * up, int CoreSize,
                  double CorePos[3], double CoreVel[3], double * SigmaX, double * SigmaV);

/* HBT sinking test of satellite s against ancestor a using the cores above
 * (|dx| / SigmaX_a + |dv| / SigmaV_a < 2, or the symmetric form when symmetric != 0). */
int subhalo_sink_test(const double sPos[3], const double sVel[3], const double aPos[3], const double aVel[3],
                      double aSigmaX, double aSigmaV, double sSigmaX, double sSigmaV, int symmetric, double BoxSize);

/* k nearest tracers (ReassignParticles): one read-only tree per key in the arena (its nodes may
 * take the whole remainder: take query buffers first), queried in batches; rows nearest first,
 * ties by ID, padded with -1, a tracer finds itself. Concurrent queries are safe. The built tree
 * owns its arena storage until the caller rewinds past it; tr may be reused after the build.
 * subhalo_knn_tracers builds, queries and rewinds. */
struct SubhaloKnnTree;
struct SubhaloKnnTree * subhalo_knn_build(const int64_t * tr, int64_t nt, double BoxSize, struct SubhaloArena * arena);
void subhalo_knn_query(const struct SubhaloKnnTree * t, const int64_t * query, int64_t nq, int k, int64_t * out);
int64_t subhalo_knn_build_count(void);
void subhalo_knn_tracers(const int64_t * query, int64_t nq, const int64_t * tr, int64_t nt, int k,
                         int64_t * out, double BoxSize, struct SubhaloArena * arena);

/* Energies E[0..n) of idx[0..n) in the frame (RefPos, RefVel), sourced by the potential types
 * among src[0..ns) at true mass, no self term; direct sum up to SampleSize sources, else the tree. */
void subhalo_energies(const int64_t * idx, int64_t n, double * E, const int64_t * src, int64_t ns,
                      const double RefPos[3], const double RefVel[3], const struct SubhaloUnbindParams * up,
                      struct SubhaloArena * arena);

/* Specific thermal energy u of a gas particle (0 otherwise), as petaio.c GTInternalEnergy. */
double subhalo_thermal_energy(int64_t i, double atime);

/* ------------------------------------------------------------------ properties (subhalo_props.c) */

/* Arena bytes per bound particle of the property functions (the radial order). */
#define SUBHALO_PROPS_BYTES_PER_PART 16

/* The HBT columns of rec after the final pass (design 7.2 step 7g): counts, masses, core,
 * MostBound*, CoM*, Vmax/Rmax (HBT's profile from CorePos, radii floored at h / 2.8), energies.
 * Positions comoving, Offset removed and wrapped; velocities Vel / a (the caller converts to km/s). */
void subhalo_props_hbt(struct SubTrack * rec, const struct SubCandidate * c, const struct SubhaloUnbindParams * up,
                       int CoreSize, const double Offset[3], struct SubhaloArena * arena);

/* ------------------------------------------------------------------ Rockstar bound properties (subhalo_props.c) */

/* Comoving density thresholds at a for vir, 200b, 200c, 500c, 2500c (design 7.10, Rockstar's
 * groupies.c; Delta_vir of Bryan and Norman relative to critical). */
void subhalo_so_thresholds(const Cosmology * CP, double atime, double rho[SUBHALO_NSO]);

/* The Rockstar bound-set columns of rec after the final pass (design D4, section 8): SO masses and
 * radii, VmaxVir, Rhalf, J, spins, Xoff/Voff, RsKlypin, shapes, the deferred four (as below);
 * each -1 when undefined. Reads no other column (Offset unused); velocities km/s, J in physical
 * kpc/h km/s. */
void subhalo_props_rockstar(struct SubTrack * rec, const struct SubCandidate * c,
                            const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO],
                            int CoreSize, const double Offset[3], struct SubhaloArena * arena);

/* Rockstar f_to_c: the NFW concentration with c / (ln(1 + c) - c / (1 + c)) = f, -1 on failure. */
double subhalo_klypin_f_to_c(double f);

/* ------------------------------------------------------------------ deferred properties (subhalo_props.c) */

/* Rockstar's NFW fit Rs and RockstarCoreVel, BulkVel, Vrms on the aperture of the kc bound
 * particles inside the outermost virial crossing (design D4, decisions 44, 47.1); -1 when undefined.
 * Deviations from Rockstar: bin edges interpolated in one length unit (nfw.c mixes Mpc/h and kpc/h),
 * a double fit returning -1 on bad input, softening-floored edge radii, Vrms over the aperture's own
 * mass, only the crossing particle left out of the fit. Outputs final: velocities physical peculiar
 * km/s, Rs comoving in the run's length unit. subhalo_props_rockstar gives the same values
 * bitwise; subhalo_collective.c in place, with the same compensated sums (equal if they are exact). */
void subhalo_props_deferred(struct SubTrack * rec, const struct SubCandidate * c,
                            const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO],
                            int CoreSize, struct SubhaloArena * arena);

/* Pieces both modes share, so the serial and the in-place Rs come from the same arithmetic. */
#define SUBHALO_NFW_MAXBINS 50        /* MAX_SCALE_BINS */
#define SUBHALO_NFW_MINPERBIN 15      /* MIN_PART_PER_BIN (also Rockstar's iteration cap) */
#define SUBHALO_NFW_MINPART 100       /* MIN_SCALE_PART */
#define SUBHALO_ROCKSTAR_CORE_MINPART 100
/* Bin plan of nfit particles of mass Mfit, largest mass mmax: the bin count (0 below MINPART),
 * *mpbin the mass per bin (at least mmax). */
int subhalo_nfw_bin_plan(int64_t nfit, double Mfit, double mmax, double * mpbin);
/* Cumulative mass at which bin b (1-based) closes, with a 1e-9 relative margin against the
 * rounding of the cumulative sums. */
double subhalo_nfw_threshold(int b, double mpbin);
/* Outer edge of bin b closed by the particle at r (mass m, cumulative Mc) between rprev and rnext:
 * 0.5 (r + rnext), minus (Mc - b mpbin) / m (r - rprev) only when has_prev and m > 0 (nfw.c's
 * interpolation in one length unit). */
double subhalo_nfw_edge(int b, double mpbin, double rprev, int has_prev, double r, double rnext, double Mc, double m);
/* Rockstar calc_scale_from_bins in double on edge[0..nbins] (edge[0] = 0): Newton on the chi^2 of
 * the bin masses from rs0, bins below wcut weighted 0.1, Rockstar's stopping rule; -1 on degenerate
 * or nonfinite input or result. */
double subhalo_nfw_fit(double rs0, const double * edge, int nbins, double wcut);
/* Rockstar estimate_scale_radius from Mvir, Rvir and the VmaxVir pair (best = max M / r at rbest). */
double subhalo_klypin_rs(double Rvir, double Mvir, double best, double rbest);

/* Baryonic sums over the bound set as fof.c's: SFR, HeIII, BH mass and Mdot, MostMassiveBHID, the
 * metal masses with MetalReturnOn (else -1); every column -1 when Nbound == 0. */
void subhalo_props_baryons(struct SubTrack * rec, const struct SubCandidate * c, int MetalReturnOn);

/* ------------------------------------------------------------------ intra-key threading (subhalo_unbind.c) */

/* A step on more than PAR_MIN elements outside a parallel region (a key run alone) splits over the
 * threads. CHUNK_MIN fixes the chunks of the floating sums and the partition's switch, so it defines
 * results and must not change; PAR_MIN only shares work out (decisions 34, 42.5). set_par_min:
 * tests only, outside any parallel region, n <= 0 restores the default. */
#define SUBHALO_UNBIND_PAR_MIN 512
#define SUBHALO_UNBIND_CHUNK_MIN 16384
void subhalo_unbind_set_par_min(int64_t n);
int64_t subhalo_unbind_get_par_min(void);
/* Dynamic chunk of the energy and query loops: nonzero (default) n / (4 threads) clamped to
 * [16, 256], zero a fixed 256. Scheduling only; tests and the timing harness. */
void subhalo_unbind_set_loop_chunk(int adaptive);

#endif
