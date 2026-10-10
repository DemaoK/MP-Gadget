/* Runtime subhalo finder: output of the PIG tables and particle tags, completion marker,
 * restart state search, table read and tag join.
 *
 * Design: halofinder_design.md sections 9 (output), 11 (restart), D7 (marker), Appendix B
 * (schema). Phase 1 plan: halofinder_phase1_plan.md section 3.
 *
 * Every function here that takes a communicator is collective over it. Scratch memory comes
 * from the mymalloc stack and is freed in reverse order before each function returns. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/stat.h>

#include <bigfile-mpi.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/openmpsort.h"
#include "utils/system.h"
#include "utils/string.h"

#include "subhalo.h"
#include "subhalo_internal.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "petaio.h"

/* Snapshot number of the PIG being written, set by fof_save_particles (-1 if unknown). */
static int OutputSnapNum = -1;

void
subhalo_set_output_snapshot(int SnapNum)
{
    OutputSnapNum = SnapNum;
}

/* ------------------------------------------------------------------ selection and sort key */

int
subhalo_save_select(int64_t i, const struct particle_data * Parts)
{
    if(Parts[i].Swallowed)
        return 0;
    if(Parts[i].GrNr >= 0)
        return 1;
    return Parts[i].TrackId >= 0 && subhalo_finder_on();
}

#define SUBHALO_KEY_LIMIT (1LL << 39)
#define SUBHALO_TRACKP1_LIMIT (1LL << 40)

void
subhalo_sort_key(const struct particle_data * p, uint64_t out[2])
{
    uint64_t section = 1, key = 0, tp1 = 0, br = 0;
    if(p->GrNr >= 0) {
        if(p->GrNr >= SUBHALO_KEY_LIMIT)
            endrun(5, "Subhalo sort key: GrNr %ld does not fit in 39 bits\n", p->GrNr);
        section = 0;
        key = p->GrNr;
    }
    if(p->TrackId >= 0) {
        if(p->TrackId + 1 >= SUBHALO_TRACKP1_LIMIT)
            endrun(5, "Subhalo sort key: TrackId %ld does not fit in 40 bits\n", p->TrackId);
        if(p->BindRank < 0)
            endrun(5, "Subhalo sort key: particle %lu has TrackId %ld and BindRank %d < 0\n", p->ID, p->TrackId, p->BindRank);
        tp1 = p->TrackId + 1;
        br = (uint32_t) p->BindRank;
    }
    out[1] = (section << 63) | (key << 24) | (tp1 >> 16);
    out[0] = ((tp1 & 0xFFFFLu) << 48) | (br << 16);
}

int
subhalo_order_by_type_and_key(const void * a, const void * b)
{
    const struct particle_data * pa = (const struct particle_data *) a;
    const struct particle_data * pb = (const struct particle_data *) b;

    if(pa->Type < pb->Type)
        return -1;
    if(pa->Type > pb->Type)
        return +1;
    uint64_t ka[2], kb[2];
    subhalo_sort_key(pa, ka);
    subhalo_sort_key(pb, kb);
    if(ka[1] != kb[1])
        return (ka[1] < kb[1]) ? -1 : +1;
    if(ka[0] != kb[0])
        return (ka[0] < kb[0]) ? -1 : +1;
    return 0;
}

/* ------------------------------------------------------------------ particle blocks */

SIMPLE_GETTER(GTPartTrackId, TrackId, int64_t, 1, struct particle_data)
SIMPLE_GETTER(GTPartBindRank, BindRank, int32_t, 1, struct particle_data)
SIMPLE_GETTER(GTPartBoundTrackId, BoundTrackId, int64_t, 1, struct particle_data)

void
subhalo_register_particle_blocks(struct IOTable * IOTable)
{
    int ptype;
    /* Explicit names: the IO_REG macros would clash with the table getters of the same name. */
    for(ptype = 0; ptype < 6; ptype++) {
        io_register_io_block("TrackId", "i8", 1, ptype, (property_getter) GTPartTrackId, NULL, 1, IOTable);
        io_register_io_block("BindRank", "i4", 1, ptype, (property_getter) GTPartBindRank, NULL, 1, IOTable);
        io_register_io_block("BoundTrackId", "i8", 1, ptype, (property_getter) GTPartBoundTrackId, NULL, 1, IOTable);
    }
}

/* ------------------------------------------------------------------ table columns (Appendix B) */

#define SUBTRACK_PROPERTY(name, field, type, items) \
    SIMPLE_GETTER(GTSub ## name, field, type, items, struct SubTrack) \
    SIMPLE_SETTER(STSub ## name, field, type, items, struct SubTrack)

#define SUBREG(name, dtype, items, ptype, IOTable) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GTSub ## name, (property_setter) STSub ## name, 1, IOTable)

/* group A */
SUBTRACK_PROPERTY(TrackId, TrackId, int64_t, 1)
SUBTRACK_PROPERTY(HostGroup, HostGroup, int64_t, 1)
SUBTRACK_PROPERTY(NestedParentTrackId, NestedParentTrackId, int64_t, 1)
SUBTRACK_PROPERTY(CentralTrackId, CentralTrackId, int64_t, 1)
SUBTRACK_PROPERTY(SinkTrackId, SinkTrackId, int64_t, 1)
SUBTRACK_PROPERTY(DescendantTrackId, DescendantTrackId, int64_t, 1)
SUBTRACK_PROPERTY(Rank, Rank, int32_t, 1)
SUBTRACK_PROPERTY(Depth, Depth, int32_t, 1)
SUBTRACK_PROPERTY(SnapBirth, SnapBirth, int32_t, 1)
SUBTRACK_PROPERTY(SnapDeath, SnapDeath, int32_t, 1)
SUBTRACK_PROPERTY(SnapSink, SnapSink, int32_t, 1)
SUBTRACK_PROPERTY(SnapLastIsolation, SnapLastIsolation, int32_t, 1)
SUBTRACK_PROPERTY(MostBoundID, MostBoundID, uint64_t, 1)
/* counts */
SUBTRACK_PROPERTY(Nbound, Nbound, int32_t, 1)
SUBTRACK_PROPERTY(NboundExclusive, NboundExclusive, int32_t, 1)
SUBTRACK_PROPERTY(NboundTracer, NboundTracer, int32_t, 1)
SUBTRACK_PROPERTY(Flags, Flags, int32_t, 1)
SUBTRACK_PROPERTY(NboundType, NboundType[0], int32_t, 6)
SUBTRACK_PROPERTY(NboundExclusiveType, NboundExclusiveType[0], int32_t, 6)
SUBTRACK_PROPERTY(NtailType, NtailType[0], int32_t, 6)
/* slices */
SUBTRACK_PROPERTY(SliceOffsetByType, SliceOffsetByType[0], int64_t, 6)
SUBTRACK_PROPERTY(SliceLengthByType, SliceLengthByType[0], int32_t, 6)
SUBTRACK_PROPERTY(SliceNboundByType, SliceNboundByType[0], int32_t, 6)
SUBTRACK_PROPERTY(FieldSliceOffsetByType, FieldSliceOffsetByType[0], int64_t, 6)
SUBTRACK_PROPERTY(FieldSliceLengthByType, FieldSliceLengthByType[0], int32_t, 6)
SUBTRACK_PROPERTY(FieldSliceNboundByType, FieldSliceNboundByType[0], int32_t, 6)
/* group B */
SUBTRACK_PROPERTY(Mbound, Mbound, float, 1)
SUBTRACK_PROPERTY(MboundType, MboundType[0], float, 6)
SUBTRACK_PROPERTY(CorePos, CorePos[0], double, 3)
SUBTRACK_PROPERTY(MostBoundPos, MostBoundPos[0], double, 3)
SUBTRACK_PROPERTY(CoMPos, CoMPos[0], double, 3)
SUBTRACK_PROPERTY(CoreVel, CoreVel[0], float, 3)
SUBTRACK_PROPERTY(MostBoundVel, MostBoundVel[0], float, 3)
SUBTRACK_PROPERTY(CoMVel, CoMVel[0], float, 3)
SUBTRACK_PROPERTY(CoreSigmaX, CoreSigmaX, float, 1)
SUBTRACK_PROPERTY(CoreSigmaV, CoreSigmaV, float, 1)
SUBTRACK_PROPERTY(Vmax, Vmax, float, 1)
SUBTRACK_PROPERTY(Rmax, Rmax, float, 1)
SUBTRACK_PROPERTY(VmaxVir, VmaxVir, float, 1)
SUBTRACK_PROPERTY(RmaxVir, RmaxVir, float, 1)
SUBTRACK_PROPERTY(Rhalf, Rhalf, float, 1)
SUBTRACK_PROPERTY(Renclose, Renclose, float, 1)
SUBTRACK_PROPERTY(MBoundSO, MBoundSO[0], float, SUBHALO_NSO)
SUBTRACK_PROPERTY(RBoundSO, RBoundSO[0], float, SUBHALO_NSO)
SUBTRACK_PROPERTY(J, J[0], float, 3)
SUBTRACK_PROPERTY(JVir, JVir[0], float, 3)
SUBTRACK_PROPERTY(SpinPeebles, SpinPeebles, float, 1)
SUBTRACK_PROPERTY(SpinBullock, SpinBullock, float, 1)
SUBTRACK_PROPERTY(Ekin, Ekin, float, 1)
SUBTRACK_PROPERTY(Epot, Epot, float, 1)
SUBTRACK_PROPERTY(KinToPot, KinToPot, float, 1)
SUBTRACK_PROPERTY(Xoff, Xoff, float, 1)
SUBTRACK_PROPERTY(Voff, Voff, float, 1)
SUBTRACK_PROPERTY(RsKlypin, RsKlypin, float, 1)
SUBTRACK_PROPERTY(BtoA, BtoA, float, 1)
SUBTRACK_PROPERTY(CtoA, CtoA, float, 1)
SUBTRACK_PROPERTY(BtoA500, BtoA500, float, 1)
SUBTRACK_PROPERTY(CtoA500, CtoA500, float, 1)
SUBTRACK_PROPERTY(AxisA, AxisA[0], float, 3)
SUBTRACK_PROPERTY(AxisA500, AxisA500[0], float, 3)
SUBTRACK_PROPERTY(InertiaTensor, InertiaTensor[0], float, 6)
SUBTRACK_PROPERTY(Rs, Rs, float, 1)
SUBTRACK_PROPERTY(RockstarCoreVel, RockstarCoreVel[0], float, 3)
SUBTRACK_PROPERTY(BulkVel, BulkVel[0], float, 3)
SUBTRACK_PROPERTY(Vrms, Vrms, float, 1)
/* group C */
SUBTRACK_PROPERTY(StarFormationRate, StarFormationRate, float, 1)
SUBTRACK_PROPERTY(GasMetalMass, GasMetalMass, float, 1)
SUBTRACK_PROPERTY(StellarMetalMass, StellarMetalMass, float, 1)
SUBTRACK_PROPERTY(BlackholeMass, BlackholeMass, float, 1)
SUBTRACK_PROPERTY(BlackholeAccretionRate, BlackholeAccretionRate, float, 1)
SUBTRACK_PROPERTY(MassHeIonized, MassHeIonized, float, 1)
SUBTRACK_PROPERTY(GasMetalElemMass, GasMetalElemMass[0], float, NMETALS)
SUBTRACK_PROPERTY(StellarMetalElemMass, StellarMetalElemMass[0], float, NMETALS)
SUBTRACK_PROPERTY(MostMassiveBHID, MostMassiveBHID, uint64_t, 1)
/* group D */
SUBTRACK_PROPERTY(MSO, MSO[0], float, SUBHALO_NSO)
SUBTRACK_PROPERTY(RSO, RSO[0], float, SUBHALO_NSO)
SUBTRACK_PROPERTY(MTypeSO200c, MTypeSO200c[0], float, 6)
SUBTRACK_PROPERTY(MTypeSOVir, MTypeSOVir[0], float, 6)
/* group E */
SUBTRACK_PROPERTY(MboundPeak, MboundPeak, float, 1)
SUBTRACK_PROPERTY(VmaxAtMboundPeak, VmaxAtMboundPeak, float, 1)
SUBTRACK_PROPERTY(VmaxPeak, VmaxPeak, float, 1)
SUBTRACK_PROPERTY(MboundAcc, MboundAcc, float, 1)
SUBTRACK_PROPERTY(VmaxAcc, VmaxAcc, float, 1)
SUBTRACK_PROPERTY(MboundFirstAcc, MboundFirstAcc, float, 1)
SUBTRACK_PROPERTY(VmaxFirstAcc, VmaxFirstAcc, float, 1)
SUBTRACK_PROPERTY(SnapMboundPeak, SnapMboundPeak, int32_t, 1)
SUBTRACK_PROPERTY(SnapVmaxPeak, SnapVmaxPeak, int32_t, 1)
SUBTRACK_PROPERTY(SnapAcc, SnapAcc, int32_t, 1)
SUBTRACK_PROPERTY(SnapFirstAcc, SnapFirstAcc, int32_t, 1)
SUBTRACK_PROPERTY(SnapLastMajorMerger, SnapLastMajorMerger, int32_t, 1)
SUBTRACK_PROPERTY(GeomParentTrackId, GeomParentTrackId, int64_t, 1)
SUBTRACK_PROPERTY(GeomUpParentTrackId, GeomUpParentTrackId, int64_t, 1)
/* group F */
SUBTRACK_PROPERTY(TracerPos, TracerPos[0], double, 3)
SUBTRACK_PROPERTY(TracerVel, TracerVel[0], float, 3)
SUBTRACK_PROPERTY(SnapTracerLost, SnapTracerLost, int32_t, 1)

/* Columns common to both tables: group A links and snapshots, and group E. */
static void
subhalo_register_links_and_history(int ptype, struct IOTable * T)
{
    SUBREG(TrackId, "i8", 1, ptype, T);
    SUBREG(HostGroup, "i8", 1, ptype, T);
    SUBREG(NestedParentTrackId, "i8", 1, ptype, T);
    SUBREG(CentralTrackId, "i8", 1, ptype, T);
    SUBREG(SinkTrackId, "i8", 1, ptype, T);
    SUBREG(DescendantTrackId, "i8", 1, ptype, T);
    SUBREG(SnapBirth, "i4", 1, ptype, T);
    SUBREG(SnapDeath, "i4", 1, ptype, T);
    SUBREG(SnapSink, "i4", 1, ptype, T);
    SUBREG(SnapLastIsolation, "i4", 1, ptype, T);
    SUBREG(MostBoundID, "u8", 1, ptype, T);
    SUBREG(MboundPeak, "f4", 1, ptype, T);
    SUBREG(VmaxAtMboundPeak, "f4", 1, ptype, T);
    SUBREG(VmaxPeak, "f4", 1, ptype, T);
    SUBREG(MboundAcc, "f4", 1, ptype, T);
    SUBREG(VmaxAcc, "f4", 1, ptype, T);
    SUBREG(MboundFirstAcc, "f4", 1, ptype, T);
    SUBREG(VmaxFirstAcc, "f4", 1, ptype, T);
    SUBREG(SnapMboundPeak, "i4", 1, ptype, T);
    SUBREG(SnapVmaxPeak, "i4", 1, ptype, T);
    SUBREG(SnapAcc, "i4", 1, ptype, T);
    SUBREG(SnapFirstAcc, "i4", 1, ptype, T);
    SUBREG(SnapLastMajorMerger, "i4", 1, ptype, T);
    SUBREG(GeomParentTrackId, "i8", 1, ptype, T);
    SUBREG(GeomUpParentTrackId, "i8", 1, ptype, T);
}

/* The two column lists of Appendix B: PTYPE_SUBHALO entries for Subhalo/, PTYPE_ORPHAN
 * entries for Orphan/. Allocated on the top of the stack; free with destroy_io_blocks. */
static void
subhalo_register_table_blocks(struct IOTable * T)
{
    T->used = 0;
    T->allocated = 160;
    T->ent = (IOTableEntry *) mymalloc2("SubhaloIOTable", T->allocated * sizeof(IOTableEntry));

    const int s = PTYPE_SUBHALO;
    subhalo_register_links_and_history(s, T);
    SUBREG(Rank, "i4", 1, s, T);
    SUBREG(Depth, "i4", 1, s, T);
    SUBREG(Nbound, "i4", 1, s, T);
    SUBREG(NboundExclusive, "i4", 1, s, T);
    SUBREG(NboundTracer, "i4", 1, s, T);
    SUBREG(Flags, "i4", 1, s, T);
    SUBREG(NboundType, "i4", 6, s, T);
    SUBREG(NboundExclusiveType, "i4", 6, s, T);
    SUBREG(NtailType, "i4", 6, s, T);
    SUBREG(SliceOffsetByType, "i8", 6, s, T);
    SUBREG(SliceLengthByType, "i4", 6, s, T);
    SUBREG(SliceNboundByType, "i4", 6, s, T);
    SUBREG(FieldSliceOffsetByType, "i8", 6, s, T);
    SUBREG(FieldSliceLengthByType, "i4", 6, s, T);
    SUBREG(FieldSliceNboundByType, "i4", 6, s, T);
    SUBREG(Mbound, "f4", 1, s, T);
    SUBREG(MboundType, "f4", 6, s, T);
    SUBREG(CorePos, "f8", 3, s, T);
    SUBREG(MostBoundPos, "f8", 3, s, T);
    SUBREG(CoMPos, "f8", 3, s, T);
    SUBREG(CoreVel, "f4", 3, s, T);
    SUBREG(MostBoundVel, "f4", 3, s, T);
    SUBREG(CoMVel, "f4", 3, s, T);
    SUBREG(CoreSigmaX, "f4", 1, s, T);
    SUBREG(CoreSigmaV, "f4", 1, s, T);
    SUBREG(Vmax, "f4", 1, s, T);
    SUBREG(Rmax, "f4", 1, s, T);
    SUBREG(VmaxVir, "f4", 1, s, T);
    SUBREG(RmaxVir, "f4", 1, s, T);
    SUBREG(Rhalf, "f4", 1, s, T);
    SUBREG(Renclose, "f4", 1, s, T);
    SUBREG(MBoundSO, "f4", SUBHALO_NSO, s, T);
    SUBREG(RBoundSO, "f4", SUBHALO_NSO, s, T);
    SUBREG(J, "f4", 3, s, T);
    SUBREG(JVir, "f4", 3, s, T);
    SUBREG(SpinPeebles, "f4", 1, s, T);
    SUBREG(SpinBullock, "f4", 1, s, T);
    SUBREG(Ekin, "f4", 1, s, T);
    SUBREG(Epot, "f4", 1, s, T);
    SUBREG(KinToPot, "f4", 1, s, T);
    SUBREG(Xoff, "f4", 1, s, T);
    SUBREG(Voff, "f4", 1, s, T);
    SUBREG(RsKlypin, "f4", 1, s, T);
    SUBREG(BtoA, "f4", 1, s, T);
    SUBREG(CtoA, "f4", 1, s, T);
    SUBREG(BtoA500, "f4", 1, s, T);
    SUBREG(CtoA500, "f4", 1, s, T);
    SUBREG(AxisA, "f4", 3, s, T);
    SUBREG(AxisA500, "f4", 3, s, T);
    SUBREG(InertiaTensor, "f4", 6, s, T);
    SUBREG(Rs, "f4", 1, s, T);
    SUBREG(RockstarCoreVel, "f4", 3, s, T);
    SUBREG(BulkVel, "f4", 3, s, T);
    SUBREG(Vrms, "f4", 1, s, T);
    SUBREG(StarFormationRate, "f4", 1, s, T);
    SUBREG(GasMetalMass, "f4", 1, s, T);
    SUBREG(StellarMetalMass, "f4", 1, s, T);
    SUBREG(BlackholeMass, "f4", 1, s, T);
    SUBREG(BlackholeAccretionRate, "f4", 1, s, T);
    SUBREG(MassHeIonized, "f4", 1, s, T);
    SUBREG(GasMetalElemMass, "f4", NMETALS, s, T);
    SUBREG(StellarMetalElemMass, "f4", NMETALS, s, T);
    SUBREG(MostMassiveBHID, "u8", 1, s, T);
    SUBREG(MSO, "f4", SUBHALO_NSO, s, T);
    SUBREG(RSO, "f4", SUBHALO_NSO, s, T);
    SUBREG(MTypeSO200c, "f4", 6, s, T);
    SUBREG(MTypeSOVir, "f4", 6, s, T);

    const int o = PTYPE_ORPHAN;
    subhalo_register_links_and_history(o, T);
    SUBREG(TracerPos, "f8", 3, o, T);
    SUBREG(TracerVel, "f4", 3, o, T);
    SUBREG(SnapTracerLost, "i4", 1, o, T);
}

static const char *
subhalo_table_dir(int ptype)
{
    return (ptype == PTYPE_SUBHALO) ? "Subhalo" : "Orphan";
}

/* Number of records held by this rank, 0 for a table that does not exist yet. */
static int64_t
subhalo_table_nlocal(const struct SubTrackTable * tab)
{
    if(!tab || !tab->initialised || !tab->Track)
        return 0;
    return tab->NLocal;
}

/* ------------------------------------------------------------------ slices (design 9) */

/* One piece of a (TrackId, section, type) run as seen by one rank. A run that straddles rank
 * boundaries arrives at the home rank as several pieces with the same (TrackId, type,
 * section, GrNr); the home rank merges them (min start, summed length and nbound). */
struct SlicePiece {
    int64_t TrackId;
    int64_t GrNr;       /* -1 for section 1 */
    int64_t start;      /* global row index in block <type>/ */
    int64_t length;
    int64_t nbound;     /* particles with BoundTrackId == TrackId */
    int32_t type;
    int32_t section;
};

static int
slice_piece_cmp_track(const void * a, const void * b)
{
    const struct SlicePiece * pa = (const struct SlicePiece *) a;
    const struct SlicePiece * pb = (const struct SlicePiece *) b;
    return (pa->TrackId > pb->TrackId) - (pa->TrackId < pb->TrackId);
}

static int
slice_piece_cmp_full(const void * a, const void * b)
{
    const struct SlicePiece * pa = (const struct SlicePiece *) a;
    const struct SlicePiece * pb = (const struct SlicePiece *) b;
    if(pa->TrackId != pb->TrackId)
        return (pa->TrackId > pb->TrackId) - (pa->TrackId < pb->TrackId);
    if(pa->type != pb->type)
        return (pa->type > pb->type) - (pa->type < pb->type);
    if(pa->section != pb->section)
        return (pa->section > pb->section) - (pa->section < pb->section);
    if(pa->GrNr != pb->GrNr)
        return (pa->GrNr > pb->GrNr) - (pa->GrNr < pb->GrNr);
    return (pa->start > pb->start) - (pa->start < pb->start);
}

static void
subhalo_reset_slices(struct SubTrack * t)
{
    int k;
    for(k = 0; k < 6; k++) {
        t->SliceOffsetByType[k] = -1;
        t->SliceLengthByType[k] = 0;
        t->SliceNboundByType[k] = 0;
        t->FieldSliceOffsetByType[k] = -1;
        t->FieldSliceLengthByType[k] = 0;
        t->FieldSliceNboundByType[k] = 0;
    }
}

/* Walk the selected particles of each type and emit one piece per local run of equal
 * (section, GrNr, TrackId) with TrackId >= 0. If pieces is NULL only counts. */
static int64_t
subhalo_collect_pieces(const struct part_manager_type * pman, const int * selection, const int64_t * ptype_offset, const int64_t * ptype_count, const int64_t * global_offset, struct SlicePiece * pieces)
{
    int64_t npieces = 0;
    int t;
    for(t = 0; t < 6; t++) {
        int64_t j;
        struct SlicePiece cur = {0};
        int open = 0;
        for(j = 0; j < ptype_count[t]; j++) {
            const struct particle_data * p = &pman->Base[selection[ptype_offset[t] + j]];
            const int64_t grnr = (p->GrNr >= 0) ? p->GrNr : -1;
            if(open && (p->TrackId != cur.TrackId || grnr != cur.GrNr)) {
                if(pieces)
                    pieces[npieces] = cur;
                npieces++;
                open = 0;
            }
            if(p->TrackId < 0)
                continue;
            if(!open) {
                cur.TrackId = p->TrackId;
                cur.GrNr = grnr;
                cur.start = global_offset[t] + j;
                cur.length = 0;
                cur.nbound = 0;
                cur.type = t;
                cur.section = (grnr >= 0) ? 0 : 1;
                open = 1;
            }
            cur.length++;
            if(p->BoundTrackId == p->TrackId)
                cur.nbound++;
        }
        if(open) {
            if(pieces)
                pieces[npieces] = cur;
            npieces++;
        }
    }
    return npieces;
}

void
subhalo_record_slices(const struct part_manager_type * pman, const int * selection, const int64_t * ptype_offset, const int64_t * ptype_count, MPI_Comm Comm)
{
    int ThisTask, NTask;
    MPI_Comm_rank(Comm, &ThisTask);
    MPI_Comm_size(Comm, &NTask);

    struct SubTrackTable * tab = subhalo_get_table();
    const int64_t nlocal = subhalo_table_nlocal(tab);
    int64_t i;
    for(i = 0; i < nlocal; i++)
        subhalo_reset_slices(&tab->Track[i]);

    /* Global row index of this rank's first selected particle of each type. */
    int64_t count[6], global_offset[6];
    int t;
    for(t = 0; t < 6; t++)
        count[t] = ptype_count[t];
    MPI_Exscan(count, global_offset, 6, MPI_INT64, MPI_SUM, Comm);
    if(ThisTask == 0)
        for(t = 0; t < 6; t++)
            global_offset[t] = 0;

    const int64_t npieces = subhalo_collect_pieces(pman, selection, ptype_offset, ptype_count, global_offset, NULL);
    const int64_t NTotal = tab ? tab->NTotal : 0;

    struct SlicePiece * pieces = (struct SlicePiece *) mymalloc("SlicePieces", sizeof(struct SlicePiece) * (npieces + 1));
    subhalo_collect_pieces(pman, selection, ptype_offset, ptype_count, global_offset, pieces);

    for(i = 0; i < npieces; i++) {
        if(pieces[i].TrackId >= NTotal || tab == NULL || tab->Chunk <= 0)
            endrun(5, "Particle tag %ld in the PIG write is not a track of the table (NTotal %ld)\n", pieces[i].TrackId, NTotal);
    }
    /* Home ranks are monotonic in TrackId, so a sort by TrackId groups the pieces by target.
     * Counts in 64 bits, checked against the MPI int limit inside subhalo_alltoallv (H3). */
    qsort_openmp(pieces, npieces, sizeof(struct SlicePiece), slice_piece_cmp_track);
    int64_t * sendcounts = ta_malloc("SliceSendCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < npieces; i++) {
        const int home = subhalo_table_home_rank(tab, pieces[i].TrackId);
        if(home < 0 || home >= NTask)
            endrun(5, "TrackId %ld has home rank %d of %d\n", pieces[i].TrackId, home, NTask);
        sendcounts[home]++;
    }
    struct SlicePiece * recv = NULL;
    const int64_t nrecv = subhalo_alltoallv(pieces, sendcounts, sizeof(struct SlicePiece), (void **) &recv, 0, NULL, Comm);
    ta_free(sendcounts);

    /* Merge the pieces of each run, then keep one run per (TrackId, type, section). */
    qsort_openmp(recv, nrecv, sizeof(struct SlicePiece), slice_piece_cmp_full);
    int64_t nsplit = 0, nnoncontig = 0;
    i = 0;
    while(i < nrecv) {
        /* run of equal (TrackId, type, section, GrNr) */
        struct SlicePiece run = recv[i];
        int64_t end = recv[i].start + recv[i].length;
        int64_t j = i + 1;
        while(j < nrecv && recv[j].TrackId == run.TrackId && recv[j].type == run.type
                && recv[j].section == run.section && recv[j].GrNr == run.GrNr) {
            run.length += recv[j].length;
            run.nbound += recv[j].nbound;
            if(recv[j].start + recv[j].length > end)
                end = recv[j].start + recv[j].length;
            j++;
        }
        if(end - run.start != run.length)
            nnoncontig++;
        if(run.length > INT_MAX)
            endrun(5, "Slice of track %ld type %d has %ld particles\n", run.TrackId, run.type, run.length);

        struct SubTrack * rec = subhalo_table_get(tab, run.TrackId);
        if(!rec)
            endrun(5, "Track %ld is not homed on rank %d\n", run.TrackId, ThisTask);
        int64_t * off = (run.section == 0) ? rec->SliceOffsetByType : rec->FieldSliceOffsetByType;
        int32_t * len = (run.section == 0) ? rec->SliceLengthByType : rec->FieldSliceLengthByType;
        int32_t * nb = (run.section == 0) ? rec->SliceNboundByType : rec->FieldSliceNboundByType;
        if(off[run.type] >= 0) {
            /* A second run of the same track in the same section (a track spread over two
             * groups, possible only after a failed finder call): keep the longest. */
            nsplit++;
            if(run.length > len[run.type] || (run.length == len[run.type] && run.start < off[run.type])) {
                off[run.type] = run.start;
                len[run.type] = run.length;
                nb[run.type] = run.nbound;
            }
        }
        else {
            off[run.type] = run.start;
            len[run.type] = run.length;
            nb[run.type] = run.nbound;
        }
        i = j;
    }

    myfree(recv);
    myfree(pieces);

    int64_t diag[2] = {nsplit, nnoncontig}, diagtot[2];
    MPI_Allreduce(diag, diagtot, 2, MPI_INT64, MPI_SUM, Comm);
    if(diagtot[0] > 0)
        message(0, "Warning: %ld track slices are split over several groups; the longest run is recorded.\n", diagtot[0]);
    if(diagtot[1] > 0)
        message(0, "Warning: %ld track slices are not contiguous in the PIG particle order.\n", diagtot[1]);
}

/* ------------------------------------------------------------------ tables */

void
subhalo_write_tables(BigFile * bf, MPI_Comm Comm)
{
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    struct SubTrackTable * tab = subhalo_get_table();
    const int64_t nlocal = subhalo_table_nlocal(tab);

    struct IOTable T = {0};
    subhalo_register_table_blocks(&T);

    /* Row indices of the two tables, each in local (TrackId) order. */
    int64_t * rows = (int64_t *) mymalloc("SubhaloRows", sizeof(int64_t) * (nlocal + 1));
    int64_t nsub = 0, norph = 0, i;
    for(i = 0; i < nlocal; i++) {
        if(i > 0 && tab->Track[i].TrackId <= tab->Track[i-1].TrackId)
            endrun(5, "Track table is not in TrackId order on rank %d: %ld after %ld\n", ThisTask, tab->Track[i].TrackId, tab->Track[i-1].TrackId);
        if(tab->Track[i].Nbound > 0)
            nsub++;
    }
    /* subhalo rows first, orphan rows after them */
    int64_t isub = 0, iorph = nsub;
    for(i = 0; i < nlocal; i++) {
        if(tab->Track[i].Nbound > 0)
            rows[isub++] = i;
        else
            rows[iorph++] = i;
    }
    norph = iorph - nsub;

    struct conversions conv = {0};
    int e;
    for(e = 0; e < T.used; e++) {
        IOTableEntry * ent = &T.ent[e];
        const int64_t n = (ent->ptype == PTYPE_SUBHALO) ? nsub : norph;
        const int64_t * idx = (ent->ptype == PTYPE_SUBHALO) ? rows : rows + nsub;
        char blockname[128];
        snprintf(blockname, sizeof(blockname), "%s/%s", subhalo_table_dir(ent->ptype), ent->name);
        BigArray array = {0};
        petaio_alloc_buffer(&array, ent, n);
        char * p = (char *) array.data;
        int64_t j;
        /* the IOTable getters take an int index: call them on the record itself (index 0 of a
         * rebased base pointer), so no local row count is narrowed (H4) */
        for(j = 0; j < n; j++) {
            ent->getter(0, p, &tab->Track[idx[j]], NULL, &conv);
            p += array.strides[0];
        }
        petaio_save_block(bf, blockname, &array, 0);
        petaio_destroy_buffer(&array);
    }
    myfree(rows);
    destroy_io_blocks(&T);

    int64_t cnt[2] = {nsub, norph}, tot[2];
    MPI_Allreduce(cnt, tot, 2, MPI_INT64, MPI_SUM, Comm);
    message(0, "Wrote subhalo tables: %ld subhalos, %ld orphans.\n", tot[0], tot[1]);
}

/* ------------------------------------------------------------------ header and marker */

static void
subhalo_format_params(char * buf, size_t size)
{
    const struct SubhaloParams * sp = subhalo_get_params();
    snprintf(buf, size,
        "SubhaloFinderOn = %d\n"
        "SubhaloMaxGatherFraction = %g\n"
        "SubhaloCollectiveMinPart = %ld\n"
        "SubhaloMaxUnbindIterations = %d\n"
        "SubhaloRestartFresh = %d\n"
        "SubhaloTracerTypes = %d\n"
        "SubhaloPotentialTypes = %d\n"
        "SubhaloMinNumPart = %d\n"
        "SubhaloMinNumTracerPart = %d\n"
        "SubhaloNumTracerHostFinding = %d\n"
        "SubhaloLookupBatchBytes = %ld\n"
        "SubhaloMajorProgenitorMassRatio = %g\n"
        "SubhaloBoundMassPrecision = %g\n"
        "SubhaloSourceRelaxFactor = %g\n"
        "SubhaloPotentialSampleSize = %d\n"
        "SubhaloFinalPass = %d\n"
        "SubhaloTreeOpenAngle = %g\n"
        "SubhaloMergeTrapped = %d\n"
        "SubhaloSinkSymmetric = %d\n"
        "SubhaloCoreSize = %d\n"
        "SubhaloReassignGas = %d\n"
        "SubhaloUnbindWithThermal = %d\n"
        "SubhaloFieldSO = %d\n"
        "SubhaloSOBins = %d\n"
        "SubhaloSOSearchFactor = %g\n"
        "SubhaloSOMaxDoublings = %d\n"
        "SubhaloSORefineMinPart = %d\n"
        "SubhaloMajorMergerRatio = %g\n"
        "SubhaloGeometricParents = %d\n"
        "SubhaloDebugForceExchangeFailure = %d\n",
        sp->SubhaloFinderOn, sp->SubhaloMaxGatherFraction, sp->SubhaloCollectiveMinPart,
        sp->SubhaloMaxUnbindIterations, sp->SubhaloRestartFresh, sp->SubhaloTracerTypes,
        sp->SubhaloPotentialTypes, sp->SubhaloMinNumPart, sp->SubhaloMinNumTracerPart,
        sp->SubhaloNumTracerHostFinding, sp->SubhaloLookupBatchBytes,
        sp->SubhaloMajorProgenitorMassRatio, sp->SubhaloBoundMassPrecision,
        sp->SubhaloSourceRelaxFactor, sp->SubhaloPotentialSampleSize, sp->SubhaloFinalPass, sp->SubhaloTreeOpenAngle,
        sp->SubhaloMergeTrapped, sp->SubhaloSinkSymmetric, sp->SubhaloCoreSize,
        sp->SubhaloReassignGas, sp->SubhaloUnbindWithThermal, sp->SubhaloFieldSO,
        sp->SubhaloSOBins, sp->SubhaloSOSearchFactor, sp->SubhaloSOMaxDoublings,
        sp->SubhaloSORefineMinPart, sp->SubhaloMajorMergerRatio, sp->SubhaloGeometricParents,
        sp->SubhaloDebugForceExchangeFailure);
}

void
subhalo_write_header_attrs(BigBlock * bh, const struct part_manager_type * pman, double atime, MPI_Comm Comm)
{
    int64_t nsaved[6] = {0}, NumPartSavedTotal[6];
    int64_t i;
    #pragma omp parallel for reduction(+: nsaved[:6])
    for(i = 0; i < pman->NumPart; i++) {
        if(pman->Base[i].IsGarbage)
            continue;
        if(subhalo_save_select(i, pman->Base))
            nsaved[pman->Base[i].Type]++;
    }
    MPI_Allreduce(nsaved, NumPartSavedTotal, 6, MPI_INT64, MPI_SUM, Comm);

    struct SubTrackTable * tab = subhalo_get_table();
    const int64_t nlocal = subhalo_table_nlocal(tab);
    int64_t cnt[2] = {0, 0}, tot[2];
    for(i = 0; i < nlocal; i++) {
        if(tab->Track[i].Nbound > 0)
            cnt[0]++;
        else
            cnt[1]++;
    }
    MPI_Allreduce(cnt, tot, 2, MPI_INT64, MPI_SUM, Comm);
    int64_t NumSubhalosTotal = tot[0], NumOrphansTotal = tot[1];
    int64_t NumTracksTotal = (tab && tab->initialised) ? tab->NTotal : 0;
    if(NumSubhalosTotal + NumOrphansTotal != NumTracksTotal)
        message(0, "Warning: %ld subhalos + %ld orphans != %ld tracks: the state will not be restartable.\n",
                NumSubhalosTotal, NumOrphansTotal, NumTracksTotal);

    int SnapshotIndex = OutputSnapNum;
    int SubhaloLastCallSnapshot = subhalo_last_call_snapshot();
    int SubhaloStateSnapshot = subhalo_state_snapshot();
    int SubhaloSchemaVersion = SUBHALO_SCHEMA_VERSION;
    int SubhaloStateComplete = 0;
    double SubhaloStateTime = subhalo_state_time();
    double TimeSincePreviousFinderCall = subhalo_time_since_previous_call();
    const char * VelocityConvention = "peculiar_physical_kms";
    double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
    subhalo_get_units(&UnitLength_in_cm, &UnitMass_in_g, &UnitVelocity_in_cm_per_s);
    char params[4096];
    subhalo_format_params(params, sizeof(params));
    /* the comoving thresholds (vir, 200b, 200c, 500c, 2500c) of the call the records describe */
    double SOThresholds[SUBHALO_NSO];
    subhalo_state_so_thresholds(SOThresholds);
    /* Phase 5c: the cost of the last finder call (wall time, max over the ranks, in seconds; the
     * main-allocator used high water, max over the ranks, in bytes; 0 under use_malloc) */
    const struct SubhaloCallStats * cs = subhalo_last_call_stats();
    double SubhaloWallTime = cs->WallTime;
    uint64_t SubhaloPeakBytes = cs->PeakBytes > 0 ? (uint64_t) cs->PeakBytes : 0;
    /* Phase 5d: the finder's own increment (bytes, max over the ranks and stages): the used high
     * water without the field SO walks' elastic export tables minus the usage at the call's entry */
    uint64_t SubhaloPeakIncrementBytes = cs->PeakIncrementBytes > 0 ? (uint64_t) cs->PeakIncrementBytes : 0;

    if(
    (0 != big_block_set_attr(bh, "NumPartSavedTotal", NumPartSavedTotal, "u8", 6)) ||
    (0 != big_block_set_attr(bh, "NumSubhalosTotal", &NumSubhalosTotal, "u8", 1)) ||
    (0 != big_block_set_attr(bh, "NumOrphansTotal", &NumOrphansTotal, "u8", 1)) ||
    (0 != big_block_set_attr(bh, "NumTracksTotal", &NumTracksTotal, "u8", 1)) ||
    (0 != big_block_set_attr(bh, "SnapshotIndex", &SnapshotIndex, "i4", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloLastCallSnapshot", &SubhaloLastCallSnapshot, "i4", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloStateSnapshot", &SubhaloStateSnapshot, "i4", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloSchemaVersion", &SubhaloSchemaVersion, "i4", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloStateComplete", &SubhaloStateComplete, "i4", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloStateTime", &SubhaloStateTime, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "TimeSincePreviousFinderCall", &TimeSincePreviousFinderCall, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "VelocityConvention", VelocityConvention, "S1", strlen(VelocityConvention))) ||
    (0 != big_block_set_attr(bh, "UnitLength_in_cm", &UnitLength_in_cm, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "UnitMass_in_g", &UnitMass_in_g, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "UnitVelocity_in_cm_per_s", &UnitVelocity_in_cm_per_s, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "SOThresholds", SOThresholds, "f8", SUBHALO_NSO)) ||
    (0 != big_block_set_attr(bh, "SubhaloWallTime", &SubhaloWallTime, "f8", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloPeakBytes", &SubhaloPeakBytes, "u8", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloPeakIncrementBytes", &SubhaloPeakIncrementBytes, "u8", 1)) ||
    (0 != big_block_set_attr(bh, "SubhaloParameters", params, "S1", strlen(params)))) {
        endrun(0, "Failed to write subhalo header attributes: %s\n", big_file_get_error_message());
    }
    message(0, "Subhalo header at a = %g: %ld subhalos, %ld orphans, %ld tracks, state of snapshot %d.\n",
            atime, NumSubhalosTotal, NumOrphansTotal, NumTracksTotal, SubhaloStateSnapshot);
}

void
subhalo_finalize_output(BigFile * bf, int complete, MPI_Comm Comm)
{
    int SubhaloStateSnapshot = subhalo_state_snapshot();
    double SubhaloStateTime = subhalo_state_time();
    int SubhaloStateComplete = complete ? 1 : 0;
    if(SubhaloStateComplete && OutputSnapNum >= 0 && SubhaloStateSnapshot != OutputSnapNum) {
        message(0, "Warning: the tags describe snapshot %d but this PIG is snapshot %d: marking the state incomplete.\n",
                SubhaloStateSnapshot, OutputSnapNum);
        SubhaloStateComplete = 0;
    }
    /* Every rank must write the same value. */
    int c = SubhaloStateComplete;
    MPI_Allreduce(&c, &SubhaloStateComplete, 1, MPI_INT, MPI_MIN, Comm);

    BigBlock bh;
    if(0 != big_file_mpi_open_block(bf, &bh, "Header", Comm))
        endrun(0, "Failed to reopen the PIG header: %s\n", big_file_get_error_message());
    if(
    (0 != big_block_set_attr(&bh, "SubhaloStateComplete", &SubhaloStateComplete, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "SubhaloStateSnapshot", &SubhaloStateSnapshot, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "SubhaloStateTime", &SubhaloStateTime, "f8", 1))) {
        endrun(0, "Failed to set the subhalo completion marker: %s\n", big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bh, Comm))
        endrun(0, "Failed to close the PIG header: %s\n", big_file_get_error_message());
    message(0, "Subhalo state marker: SubhaloStateComplete = %d (state of snapshot %d).\n", SubhaloStateComplete, SubhaloStateSnapshot);
}

/* ------------------------------------------------------------------ restart: state search */

/* Serial check (H8) that a block exists with the given number of rows and members, that its
 * dtype is the schema's (kind and width; BigFile converts the byte order on read, but not
 * precision), and that every payload file its header declares exists with the declared size
 * (fsize rows x nmemb x itemsize bytes): a block header surviving a missing or truncated
 * payload is not a readable state. Returns the number of problems (0 if the block is good). */
static int
subhalo_probe_block(BigFile * bf, const char * blockname, int64_t size, int nmemb, const char * dtype)
{
    BigBlock bb;
    if(0 != big_file_open_block(bf, &bb, blockname)) {
        message(0, "  block %s missing\n", blockname);
        return 1;
    }
    int bad = 0;
    if((int64_t) bb.size != size || (nmemb > 0 && bb.nmemb != nmemb)) {
        message(0, "  block %s has %ld x %d entries, expected %ld x %d\n", blockname, (int64_t) bb.size, bb.nmemb, size, nmemb);
        bad++;
    }
    if(dtype && (big_file_dtype_kind(bb.dtype) != big_file_dtype_kind(dtype) ||
                 big_file_dtype_itemsize(bb.dtype) != big_file_dtype_itemsize(dtype))) {
        message(0, "  block %s has dtype %s, the schema says %s\n", blockname, bb.dtype, dtype);
        bad++;
    }
    const int64_t rowbytes = (int64_t) bb.nmemb * big_file_dtype_itemsize(bb.dtype);
    int f;
    for(f = 0; f < bb.Nfile && !bad; f++) {
        char * fname = fastpm_strdup_printf("%s/%06X", bb.basename, f);
        struct stat st;
        const int64_t expect = (int64_t) bb.fsize[f] * rowbytes;
        if(0 != stat(fname, &st)) {
            message(0, "  block %s: payload file %s missing (%ld bytes declared)\n", blockname, fname, expect);
            bad++;
        }
        else if((int64_t) st.st_size != expect) {
            message(0, "  block %s: payload file %s has %ld bytes, %ld declared\n", blockname, fname, (int64_t) st.st_size, expect);
            bad++;
        }
        myfree(fname);
    }
    big_block_close(&bb);
    return bad;
}

/* Rank 0 only: is PIG_m a complete state of schema SUBHALO_SCHEMA_VERSION? */
static int
subhalo_state_valid(const char * fname, int m)
{
    BigFile bf = {0};
    if(0 != big_file_open(&bf, fname))
        return 0;

    int valid = 0;
    BigBlock bh;
    if(0 != big_file_open_block(&bf, &bh, "Header")) {
        message(0, "PIG %s has no readable Header: no subhalo state.\n", fname);
        big_file_close(&bf);
        return 0;
    }
    int complete = 0, schema = -1, statesnap = -1;
    int64_t nsub = -1, norph = -1, ntracks = -1;
    int64_t nsaved[6];
    double statetime = -1, time = -1;
    if(0 != big_block_get_attr(&bh, "SubhaloStateComplete", &complete, "i4", 1)) {
        message(0, "PIG %s has no subhalo state.\n", fname);
        goto header_done;
    }
    if(complete != 1) {
        message(0, "PIG %s: SubhaloStateComplete = %d, skipping.\n", fname, complete);
        goto header_done;
    }
    if((0 != big_block_get_attr(&bh, "SubhaloSchemaVersion", &schema, "i4", 1)) ||
       (0 != big_block_get_attr(&bh, "SubhaloStateSnapshot", &statesnap, "i4", 1)) ||
       (0 != big_block_get_attr(&bh, "SubhaloStateTime", &statetime, "f8", 1)) ||
       (0 != big_block_get_attr(&bh, "Time", &time, "f8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumSubhalosTotal", &nsub, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumOrphansTotal", &norph, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumTracksTotal", &ntracks, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumPartSavedTotal", nsaved, "u8", 6))) {
        message(0, "PIG %s: subhalo header attributes missing, skipping.\n", fname);
        goto header_done;
    }
    if(schema != SUBHALO_SCHEMA_VERSION) {
        message(0, "PIG %s: subhalo schema version %d, this build reads %d, skipping.\n", fname, schema, SUBHALO_SCHEMA_VERSION);
        goto header_done;
    }
    /* finite-value checks first: under -ffast-math a NaN comparison proves nothing */
    if(!subhalo_isfinite(statetime) || !subhalo_isfinite(time) || statetime <= 0 || time <= 0) {
        message(0, "PIG %s: non-finite or non-positive state time (%g, file %g), skipping.\n", fname, statetime, time);
        goto header_done;
    }
    if(statesnap != m || fabs(statetime - time) > 1e-6 * fabs(time)) {
        message(0, "PIG %s: state of snapshot %d at a = %g does not match the file (snapshot %d, a = %g), skipping.\n",
                fname, statesnap, statetime, m, time);
        goto header_done;
    }
    if(nsub < 0 || norph < 0 || nsub + norph != ntracks) {
        message(0, "PIG %s: %ld subhalos + %ld orphans != %ld tracks, skipping.\n", fname, nsub, norph, ntracks);
        goto header_done;
    }
    valid = 1;
header_done:
    big_block_close(&bh);
    if(!valid) {
        big_file_close(&bf);
        return 0;
    }

    /* Every column of the schema with the right row count. */
    struct IOTable T = {0};
    subhalo_register_table_blocks(&T);
    int e, bad = 0;
    for(e = 0; e < T.used; e++) {
        char blockname[128];
        snprintf(blockname, sizeof(blockname), "%s/%s", subhalo_table_dir(T.ent[e].ptype), T.ent[e].name);
        bad += subhalo_probe_block(&bf, blockname, (T.ent[e].ptype == PTYPE_SUBHALO) ? nsub : norph, T.ent[e].items, T.ent[e].dtype);
    }
    destroy_io_blocks(&T);
    /* The particle tags and IDs the join reads. */
    int t;
    for(t = 0; t < 6; t++) {
        if(nsaved[t] == 0)
            continue;
        const char * names[4] = {"ID", "TrackId", "BindRank", "BoundTrackId"};
        const char * dtypes[4] = {"u8", "i8", "i4", "i8"};
        int k;
        for(k = 0; k < 4; k++) {
            char blockname[128];
            snprintf(blockname, sizeof(blockname), "%d/%s", t, names[k]);
            bad += subhalo_probe_block(&bf, blockname, nsaved[t], 1, dtypes[k]);
        }
    }
    big_file_close(&bf);
    if(bad) {
        message(0, "PIG %s: %d subhalo state blocks missing or truncated, skipping.\n", fname, bad);
        return 0;
    }
    return 1;
}

int
subhalo_find_state(const char * OutputDir, const char * FOFFileBase, int MaxStateSnapNum, MPI_Comm Comm)
{
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    int found = -1;
    if(ThisTask == 0) {
        int m;
        for(m = MaxStateSnapNum; m >= 0; m--) {
            char * fname = fastpm_strdup_printf("%s/%s_%03d", OutputDir, FOFFileBase, m);
            int valid = subhalo_state_valid(fname, m);
            myfree(fname);
            if(valid) {
                found = m;
                break;
            }
        }
        if(found >= 0 && found < MaxStateSnapNum)
            message(0, "WARNING: the latest complete subhalo state is in snapshot %d, older than snapshot %d. "
                       "The catalogues in between are missing.\n", found, MaxStateSnapNum);
    }
    MPI_Bcast(&found, 1, MPI_INT, 0, Comm);
    return found;
}

/* ------------------------------------------------------------------ restart: table read */

/* First row and row count of this rank in a balanced contiguous split of N rows. */
static void
subhalo_split_rows(int64_t N, int ThisTask, int NTask, int64_t * start, int64_t * n)
{
    const int64_t s = N * ThisTask / NTask;
    const int64_t e = N * (ThisTask + 1) / NTask;
    *start = s;
    *n = e - s;
}

/* A restart-state read failure after the state was probed complete is fatal (H8): the message
 * names the PIG and the two ways out. */
#define SUBHALO_STATE_FATAL(pigname, ...) do { \
        message(1, __VA_ARGS__); \
        endrun(1, "Subhalo restart state %s is unreadable or inconsistent (see above). Clear its marker " \
                  "(set Header/SubhaloStateComplete = 0 in that PIG) so that the previous complete state is used, " \
                  "or set SubhaloRestartFresh = 1 to restart with an empty track table.\n", (pigname)); \
    } while(0)

/* Read this rank's rows of a block (rank order = the balanced split) into buf, which holds
 * nlocal x nmemb items of dtype. big_block_mpi_read reads each rank's dims[0] rows in rank
 * order from ptr, so every rank seeks to 0. Concurrency is the number of ranks (petaio's
 * IO.NumWriters is private to petaio.c). Collective; any failure is fatal (H8). */
static void
subhalo_read_rows(BigFile * bf, const char * pigname, const char * blockname, const char * dtype, int nmemb, int64_t ntotal, int64_t nlocal, void * buf, MPI_Comm Comm)
{
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    BigBlock bb;
    BigBlockPtr ptr;
    if(0 != big_file_mpi_open_block(bf, &bb, blockname, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to open block %s of %s: %s\n", blockname, pigname, big_file_get_error_message());
    if((int64_t) bb.size != ntotal || bb.nmemb != nmemb)
        SUBHALO_STATE_FATAL(pigname, "Block %s of %s has %ld x %d entries, expected %ld x %d\n", blockname, pigname, (int64_t) bb.size, bb.nmemb, ntotal, nmemb);
    size_t dims[2] = {nlocal, nmemb};
    const int elsize = big_file_dtype_itemsize(dtype);
    ptrdiff_t strides[2] = {elsize * nmemb, elsize};
    BigArray array = {0};
    big_array_init(&array, buf, dtype, 2, dims, strides);
    int fail = (0 != big_block_seek(&bb, &ptr, 0));
    if(!fail)
        fail = (0 != big_block_mpi_read(&bb, &ptr, &array, NTask, Comm));
    if(fail)
        SUBHALO_STATE_FATAL(pigname, "Failed to read block %s of %s: %s\n", blockname, pigname, big_file_get_error_message());
    if(0 != big_block_mpi_close(&bb, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to close block %s of %s: %s\n", blockname, pigname, big_file_get_error_message());
}

/* An unsigned 8-byte header count read into an int64_t: reject values a corrupted header could
 * hold that no table can have. */
static int
subhalo_count_ok(int64_t n)
{
    return n >= 0 && n < (1LL << 62);
}

int
subhalo_read_state_checked(const char * pigname, struct SubTrackTable * tab, double * StateTimeOut, MPI_Comm Comm)
{
    int ThisTask, NTask;
    MPI_Comm_rank(Comm, &ThisTask);
    MPI_Comm_size(Comm, &NTask);

    BigFile bf = {0};
    if(0 != big_file_mpi_open(&bf, pigname, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to open subhalo state %s: %s\n", pigname, big_file_get_error_message());
    BigBlock bh;
    if(0 != big_file_mpi_open_block(&bf, &bh, "Header", Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to open the header of %s: %s\n", pigname, big_file_get_error_message());
    int64_t NumSubhalosTotal = -1, NumOrphansTotal = -1, NumTracksTotal = -1;
    int schema = -1;
    double StateTime = -1;
    if((0 != big_block_get_attr(&bh, "NumSubhalosTotal", &NumSubhalosTotal, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumOrphansTotal", &NumOrphansTotal, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "NumTracksTotal", &NumTracksTotal, "u8", 1)) ||
       (0 != big_block_get_attr(&bh, "SubhaloSchemaVersion", &schema, "i4", 1)) ||
       (0 != big_block_get_attr(&bh, "SubhaloStateTime", &StateTime, "f8", 1)))
        SUBHALO_STATE_FATAL(pigname, "Subhalo state attributes missing in %s: %s\n", pigname, big_file_get_error_message());
    /* the SO thresholds of the state's call (optional: absent before they were written) */
    {
        double so[SUBHALO_NSO];
        if(0 == big_block_get_attr(&bh, "SOThresholds", so, "f8", SUBHALO_NSO))
            subhalo_set_state_so_thresholds(so);
        else
            subhalo_set_state_so_thresholds(NULL);
    }
    if(0 != big_block_mpi_close(&bh, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to close the header of %s\n", pigname);
    if(schema != SUBHALO_SCHEMA_VERSION)
        SUBHALO_STATE_FATAL(pigname, "Subhalo schema %d in %s, this build reads %d\n", schema, pigname, SUBHALO_SCHEMA_VERSION);
    if(!subhalo_count_ok(NumSubhalosTotal) || !subhalo_count_ok(NumOrphansTotal) || !subhalo_count_ok(NumTracksTotal)
            || NumSubhalosTotal + NumOrphansTotal != NumTracksTotal || !subhalo_isfinite(StateTime))
        SUBHALO_STATE_FATAL(pigname, "Subhalo state %s: inconsistent header (%ld subhalos + %ld orphans, %ld tracks, a = %g)\n",
                            pigname, NumSubhalosTotal, NumOrphansTotal, NumTracksTotal, StateTime);

    int64_t sub_start, nsub, orph_start, norph;
    subhalo_split_rows(NumSubhalosTotal, ThisTask, NTask, &sub_start, &nsub);
    subhalo_split_rows(NumOrphansTotal, ThisTask, NTask, &orph_start, &norph);

    struct SubTrack * recs = (struct SubTrack *) mymalloc("SubhaloReadRecs", sizeof(struct SubTrack) * (nsub + norph + 1));
    int64_t i;
    for(i = 0; i < nsub + norph; i++)
        subhalo_track_clear(&recs[i], -1);

    struct IOTable T = {0};
    subhalo_register_table_blocks(&T);
    struct conversions conv = {0};
    int e;
    for(e = 0; e < T.used; e++) {
        IOTableEntry * ent = &T.ent[e];
        const int issub = (ent->ptype == PTYPE_SUBHALO);
        const int64_t n = issub ? nsub : norph;
        struct SubTrack * base = issub ? recs : recs + nsub;
        char blockname[128];
        snprintf(blockname, sizeof(blockname), "%s/%s", subhalo_table_dir(ent->ptype), ent->name);
        BigArray array = {0};
        petaio_alloc_buffer(&array, ent, n);
        subhalo_read_rows(&bf, pigname, blockname, ent->dtype, ent->items, issub ? NumSubhalosTotal : NumOrphansTotal, n, array.data, Comm);
        char * p = (char *) array.data;
        int64_t j;
        /* int setters on rebased single records (H4) */
        for(j = 0; j < n; j++) {
            ent->setter(0, p, &base[j], NULL, &conv);
            p += array.strides[0];
        }
        petaio_destroy_buffer(&array);
    }
    destroy_io_blocks(&T);
    if(0 != big_file_mpi_close(&bf, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to close %s\n", pigname);

    /* Orphan rows carry no counts: an orphan has no bound particles. */
    for(i = nsub; i < nsub + norph; i++) {
        int k;
        recs[i].Nbound = recs[i].NboundExclusive = recs[i].NboundTracer = 0;
        recs[i].Mbound = 0;
        for(k = 0; k < 6; k++) {
            recs[i].NboundType[k] = recs[i].NboundExclusiveType[k] = recs[i].NtailType[k] = 0;
            recs[i].MboundType[k] = 0;
        }
    }
    /* Identity and classification (H9): every TrackId in range; a Subhalo row is resolved
     * (Nbound > 0, not dead), an Orphan row is dead or sunk (SnapDeath set). */
    int64_t badrange = 0, badclass = 0;
    for(i = 0; i < nsub + norph; i++) {
        if(recs[i].TrackId < 0 || recs[i].TrackId >= NumTracksTotal)
            badrange++;
        if(i < nsub && (recs[i].Nbound <= 0 || recs[i].SnapDeath >= 0))
            badclass++;
        if(i >= nsub && recs[i].SnapDeath < 0)
            badclass++;
    }
    if(MPIU_Any(badrange > 0 || badclass > 0, Comm)) {
        message(1, "Subhalo state %s on rank %d: %ld TrackIds outside [0, %ld), %ld rows whose Subhalo/Orphan classification "
                   "contradicts their Nbound/SnapDeath\n", pigname, ThisTask, badrange, NumTracksTotal, badclass);
        myfree(recs);
        return 1;
    }

    /* Route every row to its home and require exactly one row per homed TrackId (H9): no
     * duplicate, no hole. subhalo_table_store is not used here because it accepts duplicates. */
    tab->NTotal = NumTracksTotal;
    subhalo_table_redistribute(tab, Comm);
    int64_t * sendcounts = ta_malloc("SubReadCounts", int64_t, NTask);
    int64_t * fill = ta_malloc("SubReadFill", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nsub + norph; i++)
        sendcounts[subhalo_table_home_rank(tab, recs[i].TrackId)]++;
    fill[0] = 0;
    int r;
    for(r = 1; r < NTask; r++)
        fill[r] = fill[r-1] + sendcounts[r-1];
    struct SubTrack * sendbuf = (struct SubTrack *) mymalloc("SubReadSend", sizeof(struct SubTrack) * (nsub + norph + 1));
    for(i = 0; i < nsub + norph; i++)
        sendbuf[fill[subhalo_table_home_rank(tab, recs[i].TrackId)]++] = recs[i];
    struct SubTrack * in = NULL;
    const int64_t nin = subhalo_alltoallv(sendbuf, sendcounts, sizeof(struct SubTrack), (void **) &in, 0, NULL, Comm);
    char * seen = (char *) mymalloc("SubReadSeen", tab->NLocal + 1);
    memset(seen, 0, tab->NLocal + 1);
    const int64_t first = (int64_t) tab->ThisTask * tab->Chunk;
    int64_t ndup = 0, nhole = 0, nmisplaced = 0;
    for(i = 0; i < nin; i++) {
        const int64_t li = in[i].TrackId - first;
        if(li < 0 || li >= tab->NLocal) {
            nmisplaced++;
            continue;
        }
        if(seen[li]) {
            ndup++;
            continue;
        }
        seen[li] = 1;
        tab->Track[li] = in[i];
    }
    for(i = 0; i < tab->NLocal; i++)
        if(!seen[i])
            nhole++;
    myfree(seen);
    myfree(in);
    myfree(sendbuf);
    ta_free(fill);
    ta_free(sendcounts);
    myfree(recs);
    if(MPIU_Any(ndup > 0 || nhole > 0 || nmisplaced > 0, Comm)) {
        message(1, "Subhalo state %s on rank %d: %ld duplicate TrackIds, %ld TrackIds of [0, %ld) without a row, %ld misplaced rows\n",
                pigname, ThisTask, ndup, nhole, NumTracksTotal, nmisplaced);
        return 1;
    }

    message(0, "Read subhalo state from %s: %ld subhalos, %ld orphans, %ld tracks, a = %g.\n",
            pigname, NumSubhalosTotal, NumOrphansTotal, NumTracksTotal, StateTime);
    *StateTimeOut = StateTime;
    return 0;
}

double
subhalo_read_state(const char * pigname, struct SubTrackTable * tab, MPI_Comm Comm)
{
    double StateTime = -1;
    if(subhalo_read_state_checked(pigname, tab, &StateTime, Comm))
        SUBHALO_STATE_FATAL(pigname, "Subhalo state %s failed validation.\n", pigname);
    return StateTime;
}

/* ------------------------------------------------------------------ restart: tag join */

#define SUBHALO_ID_BASE_MASK 0x00ffffffffffffffLu

/* A saved (kind 0) or local (kind 1) record of the joint sort. For a local record TrackId,
 * BoundTrackId, BindRank and inherited hold the result after the walk. */
struct JoinRec {
    MyIDType ID;
    int64_t TrackId;
    int64_t BoundTrackId;
    int64_t index;      /* local: index in pman */
    int32_t BindRank;
    int32_t task;       /* local: owning rank */
    int32_t kind;
    int32_t inherited;
};

struct JoinResult {
    int64_t index;
    int64_t TrackId;
    int64_t BoundTrackId;
    int32_t BindRank;
    int32_t inherited;
};

/* Summary of one rank's sorted records, for the boundary reconciliation. */
struct JoinBoundary {
    int64_t n;
    uint64_t firstBase, lastBase;
    MyIDType lastID;
    int64_t lastTrackId, lastBoundTrackId;
    int64_t lowTrackId, lowBoundTrackId;
    int32_t lastBindRank, lowBindRank;
    int32_t lastIsSaved;
    int32_t hasLow;         /* the saved generation-0 record of lastBase is on this rank */
};

/* primary: base ID (most significant word); then generation; then kind (saved first). */
static void
join_radix(const void * ptr, void * radix, void * arg)
{
    const struct JoinRec * r = (const struct JoinRec *) ptr;
    uint64_t * u = (uint64_t *) radix;
    u[0] = ((((uint64_t) r->ID) >> 56) << 1) | (uint64_t) r->kind;
    u[1] = ((uint64_t) r->ID) & SUBHALO_ID_BASE_MASK;
}

static inline uint64_t
join_base(const struct JoinRec * r)
{
    return ((uint64_t) r->ID) & SUBHALO_ID_BASE_MASK;
}

/* Generation byte of the ID: 0 for an original particle, g for the g-th child spawned from it
 * (slots_split_particle gives children base + (g << 56); stars never spawn). */
static inline int
join_generation(const struct JoinRec * r)
{
    return (int) (((uint64_t) r->ID) >> 56);
}

int64_t
subhalo_join_tags(const char * pigname, struct part_manager_type * pman, MPI_Comm Comm)
{
    int ThisTask, NTask;
    MPI_Comm_rank(Comm, &ThisTask);
    MPI_Comm_size(Comm, &NTask);
    int64_t i;

    BigFile bf = {0};
    if(0 != big_file_mpi_open(&bf, pigname, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to open %s for the tag join: %s\n", pigname, big_file_get_error_message());
    /* the number of tracks of the state, for the range checks of the saved tags (H9) */
    int64_t NumTracksTotal = -1;
    {
        BigBlock bh;
        if(0 != big_file_mpi_open_block(&bf, &bh, "Header", Comm))
            SUBHALO_STATE_FATAL(pigname, "Failed to open the header of %s: %s\n", pigname, big_file_get_error_message());
        if(0 != big_block_get_attr(&bh, "NumTracksTotal", &NumTracksTotal, "u8", 1) || !subhalo_count_ok(NumTracksTotal))
            SUBHALO_STATE_FATAL(pigname, "%s: NumTracksTotal missing or invalid\n", pigname);
        if(0 != big_block_mpi_close(&bh, Comm))
            SUBHALO_STATE_FATAL(pigname, "Failed to close the header of %s\n", pigname);
    }

    /* Row counts of the saved tags per type (-1: block absent). */
    int64_t Nsaved[6];
    int64_t maxsaved = 0;
    int t;
    for(t = 0; t < 6; t++) {
        char blockname[128];
        BigBlock bb;
        snprintf(blockname, sizeof(blockname), "%d/TrackId", t);
        Nsaved[t] = -1;
        if(0 != big_file_mpi_open_block(&bf, &bb, blockname, Comm))
            continue;
        Nsaved[t] = bb.size;
        if(0 != big_block_mpi_close(&bb, Comm))
            SUBHALO_STATE_FATAL(pigname, "Failed to close %s of %s\n", blockname, pigname);
        int64_t start, n;
        subhalo_split_rows(Nsaved[t], ThisTask, NTask, &start, &n);
        maxsaved += n;
    }
    int64_t nlocalpart = 0;
    for(i = 0; i < pman->NumPart; i++)
        if(!pman->Base[i].IsGarbage)
            nlocalpart++;

    struct JoinRec * recs = (struct JoinRec *) mymalloc("JoinRecs", sizeof(struct JoinRec) * (maxsaved + nlocalpart + 1));
    int64_t n = 0;
    int64_t nsavedrecs = 0, badtags = 0;
    for(t = 0; t < 6; t++) {
        if(Nsaved[t] < 0)
            continue;
        int64_t start, nrow;
        subhalo_split_rows(Nsaved[t], ThisTask, NTask, &start, &nrow);
        MyIDType * id = (MyIDType *) mymalloc2("JoinID", sizeof(MyIDType) * (nrow + 1));
        int64_t * trackid = (int64_t *) mymalloc2("JoinTrackId", sizeof(int64_t) * (nrow + 1));
        int32_t * bindrank = (int32_t *) mymalloc2("JoinBindRank", sizeof(int32_t) * (nrow + 1));
        int64_t * boundtrackid = (int64_t *) mymalloc2("JoinBoundTrackId", sizeof(int64_t) * (nrow + 1));
        char blockname[128];
        snprintf(blockname, sizeof(blockname), "%d/ID", t);
        subhalo_read_rows(&bf, pigname, blockname, "u8", 1, Nsaved[t], nrow, id, Comm);
        snprintf(blockname, sizeof(blockname), "%d/TrackId", t);
        subhalo_read_rows(&bf, pigname, blockname, "i8", 1, Nsaved[t], nrow, trackid, Comm);
        snprintf(blockname, sizeof(blockname), "%d/BindRank", t);
        subhalo_read_rows(&bf, pigname, blockname, "i4", 1, Nsaved[t], nrow, bindrank, Comm);
        snprintf(blockname, sizeof(blockname), "%d/BoundTrackId", t);
        subhalo_read_rows(&bf, pigname, blockname, "i8", 1, Nsaved[t], nrow, boundtrackid, Comm);
        int64_t j;
        for(j = 0; j < nrow; j++) {
            /* the saved tags must name tracks of the state, and a tagged row a rank (H9) */
            if(trackid[j] < -1 || trackid[j] >= NumTracksTotal || boundtrackid[j] < -1 || boundtrackid[j] >= NumTracksTotal
                    || (trackid[j] >= 0 && bindrank[j] < 0))
                badtags++;
            /* Every saved row takes part, untagged ones included (H11): an exact match to a row
             * saved untagged gives -1 instead of the generation-0 parent's tag. */
            if(trackid[j] >= 0)
                nsavedrecs++;
            struct JoinRec * r = &recs[n++];
            r->ID = id[j];
            r->TrackId = trackid[j];
            r->BoundTrackId = boundtrackid[j];
            r->BindRank = bindrank[j];
            r->index = -1;
            r->task = -1;
            r->kind = 0;
            r->inherited = 0;
        }
        myfree(boundtrackid);
        myfree(bindrank);
        myfree(trackid);
        myfree(id);
    }
    if(0 != big_file_mpi_close(&bf, Comm))
        SUBHALO_STATE_FATAL(pigname, "Failed to close %s\n", pigname);
    if(MPIU_Any(badtags > 0, Comm))
        SUBHALO_STATE_FATAL(pigname, "Subhalo state %s on rank %d: %ld saved particle rows with a TrackId/BoundTrackId outside [-1, %ld) "
                            "or a tagged row with BindRank < 0\n", pigname, ThisTask, badtags, NumTracksTotal);

    for(i = 0; i < pman->NumPart; i++) {
        if(pman->Base[i].IsGarbage)
            continue;
        struct JoinRec * r = &recs[n++];
        r->ID = pman->Base[i].ID;
        r->TrackId = -1;
        r->BoundTrackId = -1;
        r->BindRank = -1;
        r->index = i;
        r->task = ThisTask;
        r->kind = 1;
        r->inherited = 0;
    }
    recs = (struct JoinRec *) myrealloc(recs, sizeof(struct JoinRec) * (n + 1));

    mpsort_mpi(recs, n, sizeof(struct JoinRec), join_radix, 16, NULL, Comm);

    /* Boundary reconciliation: every rank publishes a summary of its sorted records; a rank
     * looks back through its nearest non-empty predecessors (a run of equal base may span
     * several ranks, and ranks may be empty) for the record just before its first one and
     * for the saved generation-0 (parent) record of its first base. */
    struct JoinBoundary mine = {0};
    mine.n = n;
    mine.lastTrackId = mine.lastBoundTrackId = mine.lowTrackId = mine.lowBoundTrackId = -1;
    mine.lastBindRank = mine.lowBindRank = -1;
    if(n > 0) {
        mine.firstBase = join_base(&recs[0]);
        mine.lastBase = join_base(&recs[n-1]);
        mine.lastID = recs[n-1].ID;
        mine.lastIsSaved = (recs[n-1].kind == 0);
        mine.lastTrackId = recs[n-1].TrackId;
        mine.lastBoundTrackId = recs[n-1].BoundTrackId;
        mine.lastBindRank = recs[n-1].BindRank;
        for(i = n - 1; i >= 0 && join_base(&recs[i]) == mine.lastBase; i--) {
            if(recs[i].kind == 0 && join_generation(&recs[i]) == 0) {
                mine.hasLow = 1;
                mine.lowTrackId = recs[i].TrackId;
                mine.lowBoundTrackId = recs[i].BoundTrackId;
                mine.lowBindRank = recs[i].BindRank;
            }
        }
    }
    struct JoinBoundary * all = (struct JoinBoundary *) mymalloc2("JoinBoundary", sizeof(struct JoinBoundary) * NTask);
    MPI_Allgather(&mine, sizeof(struct JoinBoundary), MPI_BYTE, all, sizeof(struct JoinBoundary), MPI_BYTE, Comm);

    int haveprev = 0, prevIsSaved = 0, haveLow = 0;
    MyIDType prevID = 0;
    int64_t prevTrackId = -1, prevBoundTrackId = -1, lowTrackId = -1, lowBoundTrackId = -1;
    int32_t prevBindRank = -1, lowBindRank = -1;
    uint64_t curBase = 0;
    if(n > 0) {
        curBase = join_base(&recs[0]);
        int q;
        for(q = ThisTask - 1; q >= 0; q--) {
            if(all[q].n == 0)
                continue;
            if(!haveprev) {
                haveprev = 1;
                prevIsSaved = all[q].lastIsSaved;
                prevID = all[q].lastID;
                prevTrackId = all[q].lastTrackId;
                prevBoundTrackId = all[q].lastBoundTrackId;
                prevBindRank = all[q].lastBindRank;
            }
            if(all[q].lastBase != curBase)
                break;
            /* the generation-0 record sorts first in its base run: at most one rank has it */
            if(all[q].hasLow) {
                haveLow = 1;
                lowTrackId = all[q].lowTrackId;
                lowBoundTrackId = all[q].lowBoundTrackId;
                lowBindRank = all[q].lowBindRank;
            }
            if(all[q].firstBase != all[q].lastBase)
                break;
        }
    }
    myfree(all);

    int64_t nlocalrecs = 0;
    for(i = 0; i < n; i++) {
        struct JoinRec * r = &recs[i];
        const uint64_t base = join_base(r);
        if(base != curBase) {
            curBase = base;
            haveLow = 0;
        }
        if(r->kind == 0) {
            if(join_generation(r) == 0) {
                haveLow = 1;
                lowTrackId = r->TrackId;
                lowBoundTrackId = r->BoundTrackId;
                lowBindRank = r->BindRank;
            }
            continue;
        }
        nlocalrecs++;
        /* An exact match sorts immediately before the local record. */
        int exact;
        if(i > 0) {
            exact = (recs[i-1].kind == 0 && recs[i-1].ID == r->ID);
            if(exact) {
                r->TrackId = recs[i-1].TrackId;
                r->BoundTrackId = recs[i-1].BoundTrackId;
                r->BindRank = recs[i-1].BindRank;
            }
        } else {
            exact = (haveprev && prevIsSaved && prevID == r->ID);
            if(exact) {
                r->TrackId = prevTrackId;
                r->BoundTrackId = prevBoundTrackId;
                r->BindRank = prevBindRank;
            }
        }
        /* Inheritance: only a spawned child (generation > 0) without its own saved record, and
         * only from the saved record of its lineage base, which is always the spawning parent.
         * A parent never inherits from a child. */
        if(!exact && haveLow && join_generation(r) > 0) {
            r->TrackId = lowTrackId;
            r->BoundTrackId = lowBoundTrackId;
            r->BindRank = lowBindRank;
            r->inherited = 1;
        }
    }

    /* Return the results to the owners (64-bit counts, checked in subhalo_alltoallv, H3). */
    int64_t * sendcounts = ta_malloc("JoinCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < n; i++)
        if(recs[i].kind == 1)
            sendcounts[recs[i].task]++;
    struct JoinResult * send = (struct JoinResult *) mymalloc2("JoinSend", sizeof(struct JoinResult) * (nlocalrecs + 1));
    int64_t * fill = ta_malloc("JoinFill", int64_t, NTask);
    fill[0] = 0;
    for(t = 1; t < NTask; t++)
        fill[t] = fill[t-1] + sendcounts[t-1];
    for(i = 0; i < n; i++) {
        if(recs[i].kind != 1)
            continue;
        struct JoinResult * s = &send[fill[recs[i].task]++];
        s->index = recs[i].index;
        s->TrackId = recs[i].TrackId;
        s->BoundTrackId = recs[i].BoundTrackId;
        s->BindRank = recs[i].BindRank;
        s->inherited = recs[i].inherited;
    }
    ta_free(fill);
    myfree(recs);

    struct JoinResult * recv = NULL;
    const int64_t nrecv = subhalo_alltoallv(send, sendcounts, sizeof(struct JoinResult), (void **) &recv, 0, NULL, Comm);
    if(nrecv != nlocalpart)
        endrun(5, "Tag join returned %ld results for %ld particles\n", nrecv, nlocalpart);

    #pragma omp parallel for
    for(i = 0; i < pman->NumPart; i++) {
        pman->Base[i].TrackId = -1;
        pman->Base[i].BoundTrackId = -1;
        pman->Base[i].BindRank = -1;
    }
    int64_t ninherited = 0, nexact = 0;
    for(i = 0; i < nrecv; i++) {
        if(recv[i].index < 0 || recv[i].index >= pman->NumPart || pman->Base[recv[i].index].IsGarbage)
            endrun(5, "Tag join result for invalid particle index %ld\n", recv[i].index);
        struct particle_data * p = &pman->Base[recv[i].index];
        p->TrackId = recv[i].TrackId;
        p->BoundTrackId = recv[i].BoundTrackId;
        p->BindRank = recv[i].BindRank;
        if(recv[i].TrackId >= 0) {
            if(recv[i].inherited)
                ninherited++;
            else
                nexact++;
        }
    }
    myfree(recv);
    myfree(send);
    ta_free(sendcounts);

    int64_t cnt[3] = {nsavedrecs, nexact, ninherited}, tot[3];
    MPI_Allreduce(cnt, tot, 3, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo tag join from %s: %ld saved tags, %ld exact matches, %ld inherited by lineage.\n",
            pigname, tot[0], tot[1], tot[2]);
    return ninherited;
}
