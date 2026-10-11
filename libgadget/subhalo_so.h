#ifndef SUBHALO_SO_H
#define SUBHALO_SO_H

/* Field spherical overdensity of the centrals (design 7.6): MSO, RSO, MTypeSO200c, MTypeSOVir,
 * Flags bits 3-4 and the pass-3 columns, from all live particles in a sphere about CorePos
 * (satellites', unbound and FOF-less ones included), by tree walks over a full tree.
 *
 * Passes 1-2 (vir, 200b, 200c, 500c, 2500c at the thresholds of subhalo_so_thresholds): log bins out
 * to R = SubhaloSOSearchFactor x the radius of Mbound at the loosest threshold, R doubled while the
 * outer bin is still above it; linear sub-bins across each crossing bin when Nbound >=
 * SubhaloSORefineMinPart; the crossing interpolated in log M - log r. Resolution: one (sub-)bin.
 * Pass 3 (decision 44): the mass profile inside the published f4 RSO[vir] gives VmaxSO, RmaxSO,
 * RsKlypinSO and RsSO; its masses are 128-bit fixed-point, so for the same centre, live particles
 * and published f4 MSO[vir]/RSO[vir] it is bitwise layout independent (passes 1-2 sum doubles).
 *
 * Contracts: queries are anchored on local live particles, so a rank with centrals but no live
 * particle routes them to the lowest rank that has one. The treewalk reserve is checked after the
 * tree build: short memory halves the pass-1 bins (down to 16), then skips pass 2, pass 3 or the
 * whole call (columns -1) with a warning; it never aborts for this. */

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include "subhalo.h"
#include "domain.h"
#include "cosmology.h"

/* Linear sub-bins of pass 2 across each crossing bin. */
#define SUBHALO_SO_NSUB 64
/* Pass 3: log profile bins inside RSO[vir] (3 decades, 0.025 dex per bin) and their inner edge in
 * units of RSO[vir]; 1936 bytes per walk result, 1944 with DEBUG (16-byte masses, count, largest mass). */
#define SUBHALO_SO_NPROF 120
#define SUBHALO_SO_PROF_RIN 1e-3

/* Pass 3's fixed-point mass: a 128-bit integer, 8-byte aligned so that the walk results keep the
 * treewalk's 8-byte record layout. */
__extension__ typedef __int128 subhalo_so_q __attribute__((aligned(8)));

/* Pass 3's fixed-point unit, 2^s per mass unit with s = 126 - emax - bits (mmax < 2^emax, nlive
 * <= 2^bits) so that no sum of nlive masses can overflow; 1 when mmax <= 0. Given global nlive and
 * mmax, every rank and layout picks the same unit. */
double subhalo_so_prof_unit(int64_t nlive, double mmax);
/* A mass in that unit, rounded to the nearest integer (exact when m qscale >= 2^52 or m's
 * binary digits end at or above the unit). */
subhalo_so_q subhalo_so_prof_quantise(double m, double qscale);

/* Field SO of the centrals (Rank 0, HostGroup >= 0, Nbound > 0) among the rank's records recs[0..n):
 * writes their SO and pass-3 columns (-1 where undefined) and Flags bits 3-4; other records untouched.
 * Needs the restored domain and CorePos in the output frame. Collective on MPI_COMM_WORLD (the
 * treewalk's communicator; Comm must be it or a copy): every rank calls it, also with n == 0. */
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
/* The walks' export tables are elastic and kept out of the allocator's low-water mark; these give
 * the walks' true low-water mark since the reset (SIZE_MAX if none) and their largest table charge. */
void subhalo_so_elastic_reset(void);
size_t subhalo_so_elastic_min_free(void);
double subhalo_so_elastic_bytes(void);

#endif
