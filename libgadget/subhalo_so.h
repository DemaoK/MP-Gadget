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
 * refined this call (pass-1 interpolation, with a warning). ImportBufferBoost is private to
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

/* Field SO for the centrals of this call. recs[0..n) are the rank's result records (any order);
 * only those with Rank == 0 && HostGroup >= 0 && Nbound > 0 are queried, the others are not
 * touched. For the queried records the function writes MSO, RSO, MTypeSO200c, MTypeSOVir and
 * sets or clears Flags bits 3 and 4. Needs the restored domain (P[i].TopLeaf valid, no particle
 * outside its domain) and CorePos in the output frame (CurrentParticleOffset removed, wrapped).
 * Builds a full tree (force_tree_full) and frees it. Collective over MPI_COMM_WORLD (the
 * treewalk framework communicates on MPI_COMM_WORLD; Comm must be it or an equivalent copy):
 * every rank must call it, also with n == 0. */
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

/* Free bytes treewalk.c's ev_begin needs on this rank for a walk of the given pass (1 or 2)
 * with NB pass-1 bins and NT particle types (not counting this module's own buffers). */
size_t subhalo_so_walk_bytes(int pass, int NB, int NT);
/* Phase 5d: the field SO walks' elastic export tables (sized by the treewalk from the free
 * memory) are taken out of the main allocator's low-water mark after every walk; these give the
 * true low-water mark of the walks since the last reset (SIZE_MAX if none) and their largest
 * export tables (bytes of allocator charge: payload, alignment padding and block headers). */
void subhalo_so_elastic_reset(void);
size_t subhalo_so_elastic_min_free(void);
double subhalo_so_elastic_bytes(void);

#endif
