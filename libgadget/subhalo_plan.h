#ifndef SUBHALO_PLAN_H
#define SUBHALO_PLAN_H

/* The plan's packing as a particle-free function (subhalo_gather.c; unit-testable with synthetic
 * inputs): census messages, directory keys, the rank's resources and the configuration. */

#include "subhalo_internal.h"

/* A census message to the key directory: the particles of key g held by one rank (kind 0), the FOF
 * group record of g (kind 1), or an orphan hosted by g from its home rank (kind 2, track = TrackId). */
struct SubhaloDirMsg {
    int64_t g;
    int64_t cnt[6];
    int64_t ntags;          /* kind 0: distinct TrackIds >= 0 among this rank's particles of g */
    double spare[3];        /* unused: keeps the size of the exchanged record */
    int32_t rank;
    int32_t kind;
    int64_t nfield;         /* kind 0: of cnt, field particles (GrNr < 0) keyed to g by their track */
    int64_t track;          /* kind 2 */
};

/* A key on its directory rank (keys g in [ThisTask * KChunk + 1, ...], ascending). */
struct SubhaloDirKey {
    int64_t g;
    int64_t Length;
    int64_t LenType[6];
    int64_t Tags;           /* upper bound on the distinct member tracks: the sum of the ntags */
    double spare[3];        /* unused: keeps the entry's size */
    int64_t cstart, ccount; /* census contributions in the directory's message array, by rank */
    int64_t NField;         /* field particles in Length */
    int64_t ostart, ocount; /* kind-2 (hosted orphan) messages in the message array */
    int target;             /* processing rank of a serial key, -1 if none */
    int oversized;          /* 1: processed in place */
    int decider;            /* lowest rank holding particles of the key */
    int has_group;
    int forced;             /* in place whatever its size (a split group, an earlier pass's in-place key) */
    int pad;
};

/* This rank's resources at plan time (inputs of the budget, design 7.3). */
struct SubhaloPlanRes {
    int64_t MaxPart;
    int64_t NumLive;        /* non-garbage particles on the rank */
    int64_t grp[7];         /* of which in FOF groups, by type; [6] the total */
    int64_t slotfree[6];    /* maxsize - size of an enabled slot type; ignored if elsize is 0 */
    int64_t slotelsize[6];  /* 0 for a disabled slot type */
    double freebytes;       /* mymalloc_freebytes() */
};

/* Configuration of the packing; the byte costs are those of subhalo_plan_default_cfg. */
struct SubhaloPlanCfg {
    double GatherFraction;      /* SubhaloMaxGatherFraction */
    int64_t CollectiveMinPart;  /* 0 = off */
    int ForceRankZero;          /* SubhaloDebugForceExchangeFailure: every serial key to rank 0, unchecked */
    int64_t TotNgroups, KChunk;
    int MaxIter;                /* classification passes before the conservative fallback */
    /* byte costs (see subhalo_plan_default_cfg for what each one covers) */
    double PartBytes;           /* per particle of a received key */
    double MemberBytes;         /* per member track of a key */
    double KeyBytes;            /* per key processed on the rank (serial or local part in place) */
    double InplaceKeyBytes;     /* per in-place key, on every rank (global vectors) */
    double InplaceCandBytes;    /* per in-place candidate, on every rank */
    double InplaceMemberBytes;  /* per member track of the in-place keys, on every rank */
    /* in-place candidates and members the plan pass knows; a key made oversized while packing adds
     * one candidate and its census tags */
    int64_t NCandsKnown, NMembersKnown;
    int64_t SubKeyBase;         /* first serial sub-tree key (INT64_MAX: none) */
    double SubKeyPartBytes;     /* per particle of a sub-tree key, kept to the final tag pass (struct SubhaloBindOrd) */
    double ExchPartBytes;       /* per departing particle, exchange buffers (capped at one batch) */
    double SortBytesPerPart;    /* per MaxPart, the local sort and the restore's per-particle lists */
    double ReserveBytes;        /* the constant terms */
};

/* Fill cfg from the parameters and the communicator size. */
void subhalo_plan_default_cfg(struct SubhaloPlanCfg * cfg, int64_t TotNgroups, int NTask);

/* Classification and packing (design 7.3): passes of budgets, head greedy and proportional tail until
 * none moves a key in place (at most MaxIter), then census rounds moving keys of over-budget
 * destinations in place until all fit (ForceRankZero unchecked). dk: this rank's directory slice; in:
 * its census messages sorted by (g, kind, rank). Particle-free (res describes the rank). Collective. */
void subhalo_plan_pack(struct SubhaloDirKey * dk, int64_t ndir, const struct SubhaloDirMsg * in,
                       const struct SubhaloPlanRes * res, const struct SubhaloPlanCfg * cfg,
                       struct SubhaloPlanStats * stats, MPI_Comm Comm);

#endif
