#ifndef SUBHALO_SO_H
#define SUBHALO_SO_H

/* Field spherical overdensity of the subhalo finder's centrals (design 7.6, Phase 3 plan
 * section 4): MSO[5], RSO[5], MTypeSO200c[6], MTypeSOVir[6] and Flags bits 3 and 4 of every
 * central, from all particles (all types, garbage and swallowed excluded) in a sphere around the
 * central's CorePos, by two distributed tree walks over a full tree on the restored domain.
 *
 * Definitions, in the order vir, 200b, 200c, 500c, 2500c, with the comoving thresholds rho[d]
 * of subhalo_so_thresholds (subhalo_unbind.h, design 7.10). Radii are comoving, measured from CorePos with
 * periodic minimum images.
 *
 *  - Search radius R = SubhaloSOSearchFactor (3 Mbound / (4 pi min(rho)))^(1/3), Mbound the
 *    central's own bound mass (plan section 8), floored at GravitySoftening and capped at half
 *    the box.
 *  - Pass 1: SubhaloSOBins logarithmic bins between Rlo = min(GravitySoftening, R / 4) and R,
 *    mass per bin and type; particles inside Rlo go to bin 0. For each definition j is the
 *    outermost bin whose outer-edge cumulative density exceeds rho[d]; the crossing is in bin
 *    j + 1. If the outer edge of the last bin is still above min(rho), R is doubled and the
 *    central walked again, at most SubhaloSOMaxDoublings times; a central still above at the cap
 *    gets SUBHALO_FLAG_SOCUTOFF and, for every definition still above at R, the mass inside R.
 *    A definition with no qualifying bin gets -1 and SUBHALO_FLAG_SONOCROSS.
 *  - Pass 2, for Nbound >= SubhaloSORefineMinPart: SUBHALO_SO_NSUB linear sub-bins across each
 *    crossing bin, baseline the pass-1 cumulative mass at its inner edge; the crossing is in the
 *    sub-bin after the outermost sub-bin whose outer-edge density exceeds rho[d].
 *  - Inside the final (sub-)bin (lower edge above the threshold, upper edge not) M(<r) is
 *    interpolated linearly in log M against log r and the crossing solved from
 *    M(r) = 4/3 pi rho r^3; MSO = that mass, RSO = (3 MSO / (4 pi rho))^(1/3); the per-type
 *    masses are those at the lower edge plus the same fraction of the bin's per-type content.
 *
 * Pass 3 (decision 44: Rockstar-comparable profile quantities of the host sphere), for every
 * central with MSO[vir] > 0 and RSO[vir] > 0: one walk about the same centre (CorePos, the same
 * periodic wrap) over all live particles with r <= RSO[vir] (the published f4 value), all types,
 * satellites' and unbound particles included, accumulating their number, their largest mass and
 * their total mass in SUBHALO_SO_NPROF logarithmic bins between Rin = SUBHALO_SO_PROF_RIN RSO[vir]
 * and RSO[vir] (r <= Rin in bin 0). From that profile (subhalo_so.c so_profile_finish):
 *  - VmaxSO, RmaxSO: the maximum of G M(<= e) / (a e) over the bin edges e (radii floored at
 *    GravitySoftening, edges whose enclosed density exceeds rho[vir], Rockstar's vmax rule),
 *    VmaxSO in km/s, RmaxSO comoving; resolution one bin, ln(1000) / SUBHALO_SO_NPROF = 0.0576
 *    in ln r (5.9 % in RmaxSO; VmaxSO is evaluated at the edges and can only be below the
 *    profile's maximum, by a second-order amount at a smooth peak);
 *  - RsKlypinSO: subhalo_klypin_rs(RSO[vir], MSO[vir], max M / r, RmaxSO);
 *  - RsSO: Rockstar's NFW fit (subhalo_nfw_fit from RsKlypinSO) on the equal-mass bins of
 *    subhalo_nfw_bin_plan for the N particles inside RSO[vir] (their mass and largest mass),
 *    each bin edge where the profile's M(<r) reaches the bin's closing mass, interpolated
 *    linearly in ln r inside the profile bin (in r^3 inside bin 0); deviations from nfw.c in
 *    subhalo_unbind.h.
 * All four are -1 for satellites (not queried), for centrals without a virial result, with fewer
 * than SUBHALO_NFW_MINPART particles inside RSO[vir], when the call is skipped or pass 3 does not
 * fit in memory (warning); RsSO alone is -1 when the fit fails. All four are also -1 for a sphere
 * with SUBHALO_NFW_MINPART or more particles but no profile edge whose enclosed density exceeds
 * rho[vir] (audit 5e R2): the mass sits in the last profile bin, just inside RSO[vir] (e.g. a thin
 * shell at 0.99 RSO[vir]), so the only edge holding it is RSO[vir] itself, and the published f4
 * RSO[vir] rounds at or above the density crossing; such a central counts in
 * SubhaloSOStats.Nprofile (a particle count) but not in NprofileVmax.
 * The masses are summed as 128-bit fixed-point integers (subhalo_so_prof_unit: unit 2^-s of the
 * run's mass unit, s fixed per call from the global live particle count N and the largest mass
 * m_max as the largest s with N m_max 2^s <= 2^126, so no sum can overflow). A particle's mass m is
 * rounded to the unit with a relative error of at most 2^-(s+1) / m; it is exact (no rounding at
 * all) for every float mass m >= 2^(bits(N) - 102) m_max, bits(N) = ceil(log2 N): at N = 2^31,
 * every mass down to 2^-71 m_max (4e-22), so in practice every particle mass is exact. Integer sums
 * are exact and associative: for the same centre, the same published f4 MSO[vir] and RSO[vir] and
 * the same live particle data (positions, masses, types), the profile, and hence the four columns,
 * are bitwise the same for every particle partition, rank count, thread count and batch size, for
 * any masses (audit 5e R3). The end-to-end columns are not stronger than passes 1 and 2: those
 * sum doubles, whose layout-dependent last-bit rounding (non-dyadic masses) can, rarely, move a
 * published f4 MSO[vir] or RSO[vir] and with it the pass-3 sphere (see subhalo_so.c).
 *
 * The particle-level rule (outermost particle k with M(<=r_k) / (4/3 pi r_k^3) > rho) is
 * resolved to one sub-bin for refined centrals and to one pass-1 bin for the others; a density
 * excess confined to one bin whose outer edge is below the threshold is not seen (design 7.6,
 * "Resolution limit").
 *
 * Ranks without a live particle. Queries are anchored on local live particles (subhalo_so.c),
 * so a rank holding centrals but no live particle (possible after a domain decomposition that
 * leaves a rank empty) cannot issue them. Such a rank sends its centrals (whole records) to the
 * lowest rank with a live particle, which walks them with its own, and receives them back; the
 * results are those of the rank-local call, bit for bit when the mass sums are exact. If no rank
 * has a live particle every central gets -1 (with a warning). The call never aborts for this.
 *
 * Memory. treewalk.c's ev_begin reserves 40960 buffers of 24 + (1 + ImportBufferBoost)
 * (query + result) bytes from the free memory and needs room for 100 more per thread
 * (subhalo_so_walk_bytes); it aborts when they are not free. After the tree build the call
 * checks that requirement against the free memory of every rank: if pass 1 does not fit,
 * SubhaloSOBins is halved for this call (down to 16) and logged; if pass 1 still does not fit
 * the field SO is skipped for this call with a warning, MSO, RSO, MTypeSO200c and MTypeSOVir
 * of every central -1 and Flags bits 3 and 4 clear; if pass 2 does not fit, no central is
 * refined this call (pass-1 interpolation, with a warning); if pass 3 does not fit, RsSO,
 * RsKlypinSO, VmaxSO and RmaxSO are -1 this call (with a warning), nothing else changes.
 * ImportBufferBoost is private to
 * treewalk.c: the check assumes the parameter's default, 2, unless
 * subhalo_so_set_import_buffer_boost is called (a larger boost than assumed is caught by the
 * treewalk's own check, an abort with the walk's name and both numbers). */

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include "subhalo.h"
#include "domain.h"
#include "cosmology.h"

/* Linear sub-bins of pass 2 across each crossing bin. */
#define SUBHALO_SO_NSUB 64
/* Pass 3 (decision 44): logarithmic profile bins inside RSO[vir] and their inner edge in units of
 * RSO[vir]. 120 bins of 16-byte fixed-point masses plus the count and the largest mass: 1936 bytes
 * per result (1944 with DEBUG; 976 with the former 8-byte masses, audit 5e R1); 3 decades,
 * 0.025 dex per bin. */
#define SUBHALO_SO_NPROF 120
#define SUBHALO_SO_PROF_RIN 1e-3

/* Pass 3's fixed-point mass (audit 5e R1): a 128-bit integer, declared 8-byte aligned so that the
 * walk's results keep the treewalk's 8-byte record layout (sizeof stays 16; GCC and clang emit
 * alignment-safe code for it). */
__extension__ typedef __int128 subhalo_so_q __attribute__((aligned(8)));

/* Pass 3's fixed-point unit for nlive live particles (global) whose largest mass is mmax: 2^s
 * units per mass unit, s = 126 - emax - bits(nlive) with mmax < 2^emax and nlive <= 2^bits(nlive):
 * every quantised mass is at most 2^(126 - bits(nlive)), so the sum of nlive of them is at most
 * 2^126 and cannot overflow; 1 when mmax <= 0. A pure function of its arguments (both exact global
 * reductions), so every rank and layout picks the same unit. */
double subhalo_so_prof_unit(int64_t nlive, double mmax);
/* A mass in that unit, rounded to the nearest integer (exact when m qscale >= 2^52 or m's
 * binary digits end at or above the unit). */
subhalo_so_q subhalo_so_prof_quantise(double m, double qscale);

/* Field SO for the centrals of this call. recs[0..n) are the rank's result records (any order);
 * only those with Rank == 0 && HostGroup >= 0 && Nbound > 0 are queried, the others are not
 * touched. For the queried records the function writes MSO, RSO, MTypeSO200c, MTypeSOVir and
 * sets or clears Flags bits 3 and 4. Needs the restored domain (P[i].TopLeaf valid, no particle
 * outside its domain) and CorePos in the output frame (CurrentParticleOffset removed, wrapped).
 * Builds a full tree (force_tree_full) and frees it. Collective over MPI_COMM_WORLD (the
 * treewalk framework communicates on MPI_COMM_WORLD; Comm must be it or an equivalent copy):
 * every rank must call it, also with n == 0. Pass 3 also writes RsSO, RsKlypinSO, VmaxSO and
 * RmaxSO of the queried records (-1 where undefined). */
void subhalo_field_so(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                      const Cosmology * CP, const struct SubhaloParams * par, MPI_Comm Comm);

/* Diagnostics of one call (global totals). */
struct SubhaloSOStats {
    int64_t Ncentral;      /* centrals queried */
    int64_t Nrefined;      /* centrals refined by pass 2 */
    int64_t Ndoubled;      /* centrals whose search radius was doubled at least once */
    int64_t Ncutoff;       /* centrals with SUBHALO_FLAG_SOCUTOFF */
    int64_t Nnocross;      /* centrals with SUBHALO_FLAG_SONOCROSS */
    int64_t Nrouted;       /* centrals routed from ranks without a live particle */
    int64_t FreeBytes;     /* least free bytes on a rank after the tree build */
    int Npass1Walks;       /* pass-1 treewalk_run calls (doubling rounds x batches) */
    int Npass2Walks;
    int NBins;             /* pass-1 bins used (SubhaloSOBins, or fewer when memory is short) */
    int NTypes;            /* particle types carried by the walks (those present) */
    int Pass2;             /* pass 2 allowed by the memory check */
    int Skipped;           /* the whole call skipped for lack of memory */
    int Pass3;             /* pass 3 allowed by the memory check */
    int Npass3Walks;
    int64_t Nprofiled;     /* centrals walked by pass 3 (those with a virial result) */
    int64_t Nprofile;      /* of which with >= SUBHALO_NFW_MINPART particles inside RSO[vir] (the count) */
    int64_t NprofileVmax;  /* of those, with a profile edge above rho[vir]: VmaxSO, RmaxSO, RsKlypinSO set */
    double ProfUnit;       /* pass 3's fixed-point unit (subhalo_so_prof_unit), 0 without pass 3 */
};

/* The same with the thresholds given (rho == NULL: subhalo_so_thresholds), the final search
 * radius of every record returned in Rsearch (n entries, -1 for records not queried; may be
 * NULL) and the diagnostics in stats (may be NULL). Used by the tests. Collective. */
void subhalo_field_so_full(struct SubTrack * recs, int64_t n, DomainDecomp * ddecomp, double atime,
                           const Cosmology * CP, const struct SubhaloParams * par, const double * rho,
                           double * Rsearch, struct SubhaloSOStats * stats, MPI_Comm Comm);

/* Test knob: at most this many queries per treewalk_run (0 = no limit, the default); forces the
 * batched path that is otherwise taken only when a rank holds more centrals than particles. */
void subhalo_so_set_max_batch(int64_t maxbatch);

/* The ImportBufferBoost the memory check assumes (default 2, the parameter's default). */
void subhalo_so_set_import_buffer_boost(double boost);

/* Free bytes treewalk.c's ev_begin needs on this rank for a walk of the given pass (1, 2 or 3)
 * with NB pass-1 bins and NT particle types (not counting this module's own buffers; pass 3
 * depends on neither). */
size_t subhalo_so_walk_bytes(int pass, int NB, int NT);
/* Phase 5d: the field SO walks' elastic export tables (sized by the treewalk from the free
 * memory) are taken out of the main allocator's low-water mark after every walk; these give the
 * true low-water mark of the walks since the last reset (SIZE_MAX if none) and their largest
 * export tables (bytes of allocator charge: payload, alignment padding and block headers). */
void subhalo_so_elastic_reset(void);
size_t subhalo_so_elastic_min_free(void);
double subhalo_so_elastic_bytes(void);

#endif
