#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripted integration checks for the runtime subhalo finder: Phase 1
(halofinder_phase1_plan.md section 6), Phase 2a (halofinder_phase2a_plan.md section 6) and Phase 3
(halofinder_phase3_plan.md section 6).
Every subcommand exits 0 on success and 1 on a failed check, printing each failure; 2 means the
input could not be read.

  offon OFFDIR ONDIR          check 1: PART_000 byte-identical between the finder-off and -on
                              runs (a warning plus a by-ID roundoff check if not, since MP-Gadget
                              is not bitwise reproducible run to run; --strict-bytes to fail);
                              later PART_nnn matched by ID (max displacement report);
                              PIG_000 FOFGroups identical, finder blocks present and complete.
  consistency PIGDIR [...]    check 2: the reader rules on one or more PIGs (groups = subhalos,
                              slices tile the groups, BindRank dense, bound prefixes, counts);
                              --no-phase1 for Phase 2a catalogues.
  tracking PIG_prev PIG_next  check 3: host votes, centrals, newborn numbering, sinks (the target
                              resolved, or an orphan of the same call: same-call sink chains as in
                              tree, without cycles, the first resolved target hosted by the vote),
                              resurrections, recomputed from the two PIGs. Default: the Phase 2a
                              rules (weighted ten-tracer vote); --phase1-vote: the Phase 1 rules
                              (majority vote over every own-bound particle, the most massive
                              candidate wins, the others sink). The vote runs over the current
                              population (PIG_next's saved rows, or --part PART_next) with pre-call
                              tags joined from PIG_prev as the restart join does (exact ID first,
                              else a spawned child inherits its generation-0 base's record);
                              previously tagged particles absent from it are omitted (Phase 1) or
                              make a Phase 2a vote inconclusive (a warning) unless --part is given.
  hierarchy PIGDIR [...]      Phase 2a: Depth(child) = Depth(parent) + 1, parents hosted in the
                              same group, CentralTrackId the Rank 0 member of its HostGroup, Rank
                              dense per group with Ranks 1.. by Mbound descending; hostless tracks
                              are their own centrals.
  tags PIGDIR [...]           Phase 2a (D10, J12): BindRank is the energy order of all own-tag
                              particles of a track (dense 0..n-1, every type), the exclusive bound
                              prefix (BoundTrackId == TrackId) first; the first
                              SubhaloMinNumTracerPart (default 10) particles of a tracer type
                              (SubhaloTracerTypes) in that order are bound to the track and
                              MostBoundID is the first of them (the tracer reservation decides
                              ownership only, not the order). BoundTrackId is the tag, an ancestor
                              of it, or -1; NboundExclusive <= Nbound; the full bound list (every
                              row with BoundTrackId == T, inside T's key) has Nbound members,
                              NboundType per type, NboundTracer tracers, MboundType mass.
                              --phase1 for Phase 1 PIGs (section 1 holds stale tags there).
  orphans PIGDIR [...]        Orphan/*: TracerPos inside the box, SnapTracerLost -1 or <= the
                              snapshot, Snap* ordering, SinkTrackId / DescendantTrackId /
                              NestedParentTrackId / CentralTrackId refer to existing tracks.
  props PIGDIR [...]          Phase 3: the property columns. Bound SO (MBoundSO, RBoundSO) and field
                              SO (MSO, RSO) defined for a looser threshold whenever for a tighter one,
                              non-increasing with the threshold density (Header SOThresholds, else the
                              design 7.10 formulas), R = (3 M / 4 pi rho)^(1/3); MBoundSO <= Mbound,
                              VmaxVir <= Vmax, RmaxVir <= RBoundSO[vir], Rhalf <= Renclose; axis ratios
                              in (0, 1] with c <= b, AxisA unit vectors (all -1 when undefined); MSO,
                              RSO, MTypeSO* -1 off centrals, sum(MTypeSO200c) = MSO[200c],
                              sum(MTypeSOVir) = MSO[vir]; Flags bits 3 and 4 only on centrals, bit 4
                              set exactly when an MSO definition is -1; group C zero for absent types,
                              MostMassiveBHID (UINT64_MAX the only 'none'; 0 is a real ID) set exactly
                              when a black hole is bound; no NaN. Only defined values (!= -1) are
                              tested, so a catalogue written before Phase 3 passes. MSO below MBoundSO
                              is a warning. Phase 5: Vrms, BulkVel, RockstarCoreVel defined exactly
                              with RBoundSO[vir] (Vrms >= 0); Rs -1 below 100 bound particles, else
                              > 0 and below 1.5 RBoundSO[vir] (-1 there is a warning);
                              |RockstarCoreVel - CoreVel| > Vrms a warning; skipped when all -1.
  tree OUTPUTDIR|PIG...       Phase 4: the track history over a PIG sequence. Tracks present at every
                              output after birth; resolved nodes their own descendant; deaths with
                              SnapDeath = the first orphan output, DescendantTrackId resolved there,
                              SinkTrackId present there (resolved, or an orphan that died in the same
                              call: same-call sink chains L -> P -> C, without cycles), both unchanged
                              afterwards; MboundPeak / VmaxPeak running
                              maxima; SnapAcc, SnapFirstAcc, SnapLastIsolation, SnapLastMajorMerger
                              against the Rank history (subhalo_history_update).
  compare PIG_A PIG_B         check 5: every Subhalo/* and Orphan/* column equal after sorting by
                              TrackId, and the per-particle tags equal when matched by ID.
  markincomplete PIGDIR       check 6 helper: set Header SubhaloStateComplete (default 0).
  tagcounts PIGDIR            tagged particles per type (to compare with the run log, check 4).

Examples (from the directory holding the run outputs):
  python3 tools/subhalo_check.py offon off/output on/output
  python3 tools/subhalo_check.py consistency on/output/PIG_00?
  python3 tools/subhalo_check.py tracking --phase1-vote on/output/PIG_000 on/output/PIG_001
  python3 tools/subhalo_check.py compare on/output/PIG_001 offline/output/PIG_001
  python3 tools/subhalo_check.py markincomplete restart/output/PIG_001
  python3 tools/subhalo_check.py tagcounts on/output/PIG_001
  python3 tools/subhalo_check.py consistency --no-phase1 output/PIG_00?      (Phase 2a)
  python3 tools/subhalo_check.py hierarchy output/PIG_00?
  python3 tools/subhalo_check.py tags output/PIG_00?
  python3 tools/subhalo_check.py tags --phase1 on/output/PIG_00?               (Phase 1)
  python3 tools/subhalo_check.py orphans output/PIG_00?
  python3 tools/subhalo_check.py tracking output/PIG_003 output/PIG_004
  python3 tools/subhalo_check.py props output/PIG_00?                         (Phase 3)
  python3 tools/subhalo_check.py tree output                                  (Phase 4)

The Header attribute file format (BigFile attr-v2), used by the markincomplete fallback when the
Python bigfile module cannot rewrite attributes: one attribute per line,
    <name> <dtype> <nmemb> <HEX> #HUMANE [ <values> ]
where <dtype> is a numpy dtype string with byte order (e.g. "<i4"), <HEX> is the raw
little-endian bytes of the values in upper-case hexadecimal (2 * itemsize * nmemb digits), and
the part from "#HUMANE" on is a human-readable copy that the reader ignores. Example:
    SubhaloStateComplete <i4 1 01000000 #HUMANE [ 1 ]
"""

from __future__ import print_function

import argparse
import filecmp
import glob
import os
import re
import sys

import numpy as np
import bigfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloCatalogue, SubhaloFormatError, NTYPES, FLAG_INPLACE, NO_GROUP_U4, header_attrs  # noqa: E402


class Checker(object):
    """Collects failures and notes."""

    def __init__(self, name):
        self.name = name
        self.failures = []
        self.nchecks = 0

    def check(self, cond, msg):
        self.nchecks += 1
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def fail(self, msg):
        self.nchecks += 1
        self.failures.append(msg)

    def note(self, msg):
        print("  note: %s" % msg)

    def warn(self, msg):
        """A rule that is expected but not required: printed, not counted as a failure."""
        print("  WARNING: %s" % msg)

    def report(self):
        if self.failures:
            print("FAILED %s: %d of %d checks" % (self.name, len(self.failures), self.nchecks))
            shown = self.failures[:50]
            for f in shown:
                print("  - %s" % f)
            if len(self.failures) > len(shown):
                print("  ... and %d more" % (len(self.failures) - len(shown)))
            return 1
        print("OK %s: %d checks" % (self.name, self.nchecks))
        return 0


def attr(h, name, default=None):
    v = h.get(name, default)
    if isinstance(v, np.ndarray) and v.shape == (1,):
        return v[0]
    return v


def snapnum_of(path):
    m = re.search(r"_(\d+)/?$", path.rstrip("/"))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------- offon
def _block_dirs(root):
    """Relative paths of every BigFile block directory (those with a 'header' file)."""
    out = set()
    for dirpath, _, files in os.walk(root):
        if "header" in files:
            out.add(os.path.relpath(dirpath, root))
    return out


def _block_is_empty(blockdir):
    """True for a block with no rows (NFILE 0, or every file of size 0)."""
    try:
        with open(os.path.join(blockdir, "header")) as f:
            lines = f.read().splitlines()
    except IOError:
        return False
    nfile, nrows = None, 0
    for l in lines:
        if l.startswith("NFILE:"):
            nfile = int(l.split(":")[1])
        elif re.match(r"^[0-9A-Fa-f]{6}:", l):
            nrows += int(l.split(":")[1])
    return nfile == 0 or nrows == 0


def _data_files(blockdir):
    return sorted(f for f in os.listdir(blockdir) if re.match(r"^[0-9A-Fa-f]{6}$", f))


def compare_part_bytes(ck, off, on):
    """Every block's 'header' (dtype, nfile, sizes, checksums) and data files byte-identical;
    attr files only reported. Byte differences are recorded in ck (the caller decides whether
    they are failures)."""
    boff, bon = _block_dirs(off), _block_dirs(on)
    ck.check(boff == bon, "%s vs %s: block sets differ: only off %s, only on %s"
             % (off, on, sorted(boff - bon)[:10], sorted(bon - boff)[:10]))
    ndiff_attr = 0
    for b in sorted(boff & bon):
        doff, don = os.path.join(off, b), os.path.join(on, b)
        foff, fon = _data_files(doff), _data_files(don)
        if not ck.check(foff == fon, "%s: data file lists differ (%s vs %s)" % (b, foff, fon)):
            continue
        if not foff and _block_is_empty(doff) and _block_is_empty(don):
            # NFILE 0 on both sides: nothing to compare (the header differs only in layout
            # details that carry no data)
            continue
        ck.check(filecmp.cmp(os.path.join(doff, "header"), os.path.join(don, "header"), shallow=False),
                 "%s: block header (dtype/size/checksum) differs" % b)
        for f in foff:
            ck.check(filecmp.cmp(os.path.join(doff, f), os.path.join(don, f), shallow=False),
                     "%s/%s: data bytes differ" % (b, f))
        for a in ("attr-v2", "attr"):
            pa, pb = os.path.join(doff, a), os.path.join(don, a)
            if os.path.exists(pa) and os.path.exists(pb) and not filecmp.cmp(pa, pb, shallow=False):
                ndiff_attr += 1
                ck.note("%s/%s differs (attributes may legitimately differ)" % (b, a))
    if ndiff_attr == 0:
        ck.note("all attribute files identical too")


def _periodic(d, box):
    return d - box * np.round(d / box)


def compare_part_by_id(ck, off, on, tol):
    foff, fon = bigfile.File(off), bigfile.File(on)
    box = float(attr(header_attrs(foff), "BoxSize"))
    ntot = np.atleast_1d(header_attrs(foff)["TotNumPart"]).astype(np.int64)
    meansep = box / np.cbrt(max(ntot[1], 1))
    if tol is None:
        # "far below the softening": softening is ~ meansep / 30, tolerance a tenth of that
        tol = meansep / 300.
    worst = 0.0
    for t in range(NTYPES):
        if ntot[t] == 0:
            continue
        bo, bn = "%d/ID" % t, "%d/Position" % t
        if bo not in foff.blocks or bo not in fon.blocks:
            ck.fail("%s or %s: missing %s" % (off, on, bo))
            continue
        ida, idb = foff[bo][:], fon[bo][:]
        if not ck.check(ida.shape == idb.shape, "%s type %d: %d vs %d particles" % (os.path.basename(off), t, len(ida), len(idb))):
            continue
        sa, sb = np.argsort(ida), np.argsort(idb)
        if not ck.check(np.array_equal(ida[sa], idb[sb]), "%s type %d: ID sets differ" % (os.path.basename(off), t)):
            continue
        pa, pb = foff[bn][:][sa], fon[bn][:][sb]
        disp = np.sqrt(np.sum(_periodic(pa - pb, box) ** 2, axis=1))
        dmax = float(disp.max()) if disp.size else 0.0
        worst = max(worst, dmax)
        msg = "%s type %d: max displacement %.3e (median %.3e) of %d particles" % (
            os.path.basename(off), t, dmax, float(np.median(disp)) if disp.size else 0.0, len(disp))
        if "%d/Velocity" % t in foff.blocks and "%d/Velocity" % t in fon.blocks:
            va, vb = foff["%d/Velocity" % t][:][sa], fon["%d/Velocity" % t][:][sb]
            msg += ", max |dv| %.3e" % float(np.abs(va - vb).max() if va.size else 0.0)
        print("  " + msg)
        ck.check(dmax <= tol, msg + " exceeds tolerance %.3e" % tol)
    return worst


def compare_pig000(ck, off, on):
    foff, fon = bigfile.File(off), bigfile.File(on)
    boff, bon = set(foff.blocks), set(fon.blocks)
    for b in sorted(x for x in boff if x.startswith("FOFGroups/")):
        if not ck.check(b in bon, "%s missing in the finder-on PIG" % b):
            continue
        ck.check(np.array_equal(foff[b][:], fon[b][:]), "%s differs between the off and on runs" % b)
    ck.check(not any(b.startswith("Subhalo/") or b.startswith("Orphan/") for b in boff),
             "the finder-off PIG has Subhalo/Orphan blocks")
    ck.check("Subhalo/TrackId" in bon, "the finder-on PIG has no Subhalo/* table")
    ck.check("Orphan/TrackId" in bon, "the finder-on PIG has no Orphan/* table")
    ck.check(int(attr(header_attrs(fon), "SubhaloStateComplete", -1)) == 1,
             "the finder-on PIG_000 has SubhaloStateComplete = %s" % attr(header_attrs(fon), "SubhaloStateComplete"))
    lbt = foff["FOFGroups/LengthByType"][:].astype(np.int64) if "FOFGroups/LengthByType" in boff else np.zeros((0, 6), np.int64)
    for t in range(NTYPES):
        ng = int(lbt[:, t].sum()) if lbt.shape[0] else 0
        if ng == 0:
            continue
        for name in ("TrackId", "BindRank", "BoundTrackId"):
            ck.check("%d/%s" % (t, name) in bon, "finder-on PIG_000 lacks %d/%s" % (t, name))
        if "%d/ID" % t not in boff or "%d/GroupID" % t not in boff:
            continue
        # section 0 holds the same particles per group; order inside a group may differ
        ida, ga = foff["%d/ID" % t][:][:ng], foff["%d/GroupID" % t][:][:ng]
        idb, gb = fon["%d/ID" % t][:][:ng], fon["%d/GroupID" % t][:][:ng]
        oa, ob = np.lexsort((ida, ga)), np.lexsort((idb, gb))
        ck.check(np.array_equal(ida[oa], idb[ob]) and np.array_equal(ga[oa], gb[ob]),
                 "PIG_000 type %d: group membership (GroupID, ID) differs between the off and on runs" % t)


def cmd_offon(args):
    ck = Checker("offon %s %s" % (args.offdir, args.ondir))
    p0off, p0on = os.path.join(args.offdir, "PART_000"), os.path.join(args.ondir, "PART_000")
    if not (os.path.isdir(p0off) and os.path.isdir(p0on)):
        print("error: PART_000 missing under %s or %s" % (args.offdir, args.ondir))
        return 2
    print("PART_000 byte comparison")
    # MP-Gadget itself is not bitwise reproducible run to run with OpenMP threads (two runs of
    # the same binary and parameters can differ at the last bit in a few positions), so a byte
    # difference is reported, then PART_000 is matched by ID like the later outputs. With
    # --strict-bytes a byte difference is a failure.
    bck = Checker("PART_000 bytes")
    compare_part_bytes(bck, p0off, p0on)
    ck.nchecks += bck.nchecks
    if bck.failures:
        if args.strict_bytes:
            ck.failures.extend(bck.failures)
        else:
            for f in bck.failures[:10]:
                ck.warn("PART_000 not byte-identical: %s" % f)
            print("PART_000 matched by ID (byte comparison failed)")
            compare_part_by_id(ck, p0off, p0on, args.tol)
    else:
        print("  PART_000 byte-identical")
    parts = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(args.offdir, "PART_[0-9][0-9][0-9]")))
                   & set(os.path.basename(p) for p in glob.glob(os.path.join(args.ondir, "PART_[0-9][0-9][0-9]"))))
    for p in parts:
        if p == "PART_000":
            continue
        print("%s matched by ID" % p)
        compare_part_by_id(ck, os.path.join(args.offdir, p), os.path.join(args.ondir, p), args.tol)
    g0off, g0on = os.path.join(args.offdir, "PIG_000"), os.path.join(args.ondir, "PIG_000")
    if os.path.isdir(g0off) and os.path.isdir(g0on):
        print("PIG_000 FOF tables and finder blocks")
        compare_pig000(ck, g0off, g0on)
    else:
        ck.fail("PIG_000 missing under %s or %s" % (args.offdir, args.ondir))
    return ck.report()


# ---------------------------------------------------------------------------------- consistency
def _runs(values, starts_extra=None):
    """Boundaries of runs of equal values; starts_extra forces extra run starts (group starts).
    Returns (start, length) arrays."""
    n = values.shape[0]
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    change = np.ones(n, dtype=bool)
    change[1:] = values[1:] != values[:-1]
    if starts_extra is not None:
        se = starts_extra[(starts_extra >= 0) & (starts_extra < n)]
        change[se] = True
    starts = np.nonzero(change)[0].astype(np.int64)
    lengths = np.diff(np.append(starts, n)).astype(np.int64)
    return starts, lengths


def check_consistency(pigdir, ck, phase1=True, expect_field="auto", allow_incomplete=False):
    try:
        cat = SubhaloCatalogue(pigdir)
    except SubhaloFormatError as e:
        ck.fail(str(e))
        return
    h = cat.header
    tag = os.path.basename(pigdir.rstrip("/"))
    if not allow_incomplete:
        ck.check(cat.complete(), "%s: SubhaloStateComplete = %s" % (tag, attr(h, "SubhaloStateComplete")))
    ck.check(int(attr(h, "SubhaloSchemaVersion", -1)) == 1, "%s: SubhaloSchemaVersion %s" % (tag, attr(h, "SubhaloSchemaVersion")))
    snap = snapnum_of(pigdir)
    if snap is not None and "SnapshotIndex" in h:
        ck.check(int(attr(h, "SnapshotIndex")) == snap, "%s: SnapshotIndex %s" % (tag, attr(h, "SnapshotIndex")))
    if cat.complete() and "SubhaloStateSnapshot" in h and snap is not None:
        ck.check(int(attr(h, "SubhaloStateSnapshot")) == snap, "%s: complete but SubhaloStateSnapshot %s"
                 % (tag, attr(h, "SubhaloStateSnapshot")))

    S, O = cat.subhalos, cat.orphans
    nsub, norph = cat.nsubhalos, cat.norphans
    ck.check(int(attr(h, "NumSubhalosTotal", -1)) == nsub, "%s: NumSubhalosTotal %s, %d rows" % (tag, attr(h, "NumSubhalosTotal"), nsub))
    ck.check(int(attr(h, "NumOrphansTotal", -1)) == norph, "%s: NumOrphansTotal %s, %d rows" % (tag, attr(h, "NumOrphansTotal"), norph))
    ntracks = int(attr(h, "NumTracksTotal", -1))
    st, ot = S["TrackId"], O["TrackId"]
    ck.check(np.all(np.diff(st) > 0), "%s: Subhalo/TrackId not strictly increasing" % tag)
    ck.check(np.all(np.diff(ot) > 0), "%s: Orphan/TrackId not strictly increasing" % tag)
    alltid = np.sort(np.concatenate([st, ot]))
    ck.check(np.array_equal(alltid, np.arange(ntracks)),
             "%s: Subhalo and Orphan TrackIds are not exactly 0..NumTracksTotal-1 (%d tracks, %d rows, %d distinct)"
             % (tag, ntracks, len(alltid), len(np.unique(alltid))))
    ck.check(np.all(S["Nbound"] > 0), "%s: %d Subhalo rows with Nbound <= 0" % (tag, np.count_nonzero(S["Nbound"] <= 0)))
    ck.check(np.array_equal(S["Nbound"], S["NboundType"].sum(axis=1)), "%s: Nbound != sum(NboundType)" % tag)

    saved = np.atleast_1d(attr(h, "NumPartSavedTotal", np.zeros(6))).astype(np.int64)
    ingroup_attr = np.atleast_1d(attr(h, "NumPartInGroupTotal", np.full(6, -1))).astype(np.int64)
    gid = cat.groups["GroupID"]
    lbt = cat.groups["LengthByType"]
    length = lbt.sum(axis=1) if lbt.shape[0] else np.zeros(0, np.int64)
    ck.check(np.all(np.diff(gid) > 0), "%s: FOFGroups/GroupID not increasing" % tag)

    # per-track accumulations for Phase 1 checks
    sec0_tracks, sec0_ranks, sec0_ids = [], [], []
    nfield_rows = 0
    field_rows_of_orphans = 0
    host = S["HostGroup"]

    for t in range(NTYPES):
        ns = cat.nsaved(t)
        ng = cat.ningroup(t)
        ck.check(saved[t] == ns, "%s type %d: NumPartSavedTotal %d, %d rows" % (tag, t, saved[t], ns))
        if ingroup_attr[t] >= 0:
            ck.check(ingroup_attr[t] == ng, "%s type %d: NumPartInGroupTotal %d, LengthByType sum %d" % (tag, t, ingroup_attr[t], ng))
        if ns == 0:
            ck.check(np.all(S["SliceLengthByType"][:, t] == 0) and np.all(S["FieldSliceLengthByType"][:, t] == 0),
                     "%s type %d: slices recorded but no rows saved" % (tag, t))
            continue
        if not ck.check(cat.has_particle_tags(t), "%s: missing %d/TrackId, BindRank or BoundTrackId" % (tag, t)):
            continue
        tr, br, bt = cat.read(t, "TrackId"), cat.read(t, "BindRank"), cat.read(t, "BoundTrackId")
        ids = cat.read(t, "ID")
        for name, a in (("TrackId", tr), ("BindRank", br), ("BoundTrackId", bt), ("ID", ids)):
            ck.check(a.shape[0] == ns, "%s type %d: %s has %d rows, expected %d" % (tag, t, name, a.shape[0], ns))
        if not ck.check(ng <= ns, "%s type %d: %d rows in groups but %d saved" % (tag, t, ng, ns)):
            continue

        # LengthByType cumulation still addresses the FOF slices
        if "%d/GroupID" % t in cat.blocks:
            pg = cat.read(t, "GroupID")
            expect = np.repeat(gid, lbt[:, t]).astype(pg.dtype)
            ck.check(np.array_equal(pg[:ng], expect), "%s type %d: GroupID of section 0 does not follow LengthByType" % (tag, t))
            ck.check(np.all(pg[ng:] == NO_GROUP_U4) or not np.any(np.isin(pg[ng:], gid)),
                     "%s type %d: section 1 rows carry a valid GroupID" % (tag, t))

        # section 0: runs per group
        gstarts = cat.group_offsets[:-1, t]
        nonempty = lbt[:, t] > 0
        rstart, rlen = _runs(tr[:ng], gstarts[nonempty])
        rtrack = tr[:ng][rstart] if ng else np.zeros(0, np.int64)
        rgroup = np.searchsorted(cat.group_offsets[1:, t], rstart, side="right")  # row of FOFGroups
        tagged = rtrack >= 0
        # untagged runs only at a group start
        bad = (~tagged) & (rstart != cat.group_offsets[np.minimum(rgroup, len(gid) - 1), t]) if len(gid) else ~tagged
        ck.check(not np.any(bad), "%s type %d: %d untagged runs not at the start of their group" % (tag, t, np.count_nonzero(bad)))
        # a track appears in at most one run per group
        pair = np.stack([rgroup[tagged], rtrack[tagged]], axis=1)
        if pair.shape[0]:
            upair = np.unique(pair, axis=0)
            ck.check(upair.shape[0] == pair.shape[0], "%s type %d: a track's in-group rows are not contiguous (%d runs, %d tracks)"
                     % (tag, t, pair.shape[0], upair.shape[0]))
        # each tagged run is the track's in-group slice
        rows = np.searchsorted(st, rtrack[tagged])
        known = (rows < nsub) & (st[np.minimum(rows, max(nsub - 1, 0))] == rtrack[tagged]) if nsub else np.zeros(np.count_nonzero(tagged), bool)
        ck.check(np.all(known), "%s type %d: %d in-group runs of tracks that are not resolved subhalos (e.g. %s)"
                 % (tag, t, np.count_nonzero(~known), rtrack[tagged][~known][:5]))
        ts, tl, tt, tg, trow = rstart[tagged][known], rlen[tagged][known], rtrack[tagged][known], rgroup[tagged][known], rows[known]
        ck.check(np.array_equal(S["SliceOffsetByType"][trow, t], ts) and np.array_equal(S["SliceLengthByType"][trow, t], tl),
                 "%s type %d: SliceOffsetByType/SliceLengthByType do not match the in-group runs (%d mismatches)"
                 % (tag, t, np.count_nonzero((S["SliceOffsetByType"][trow, t] != ts) | (S["SliceLengthByType"][trow, t] != tl))))
        # After a failed call (SubhaloStateComplete = 0) the table is that of the last complete
        # call, so HostGroup names the groups of that call while the slices are computed on this
        # output's groups: only a warning then.
        hg_ok = np.array_equal(host[trow], gid[tg])
        if cat.complete():
            ck.check(hg_ok, "%s type %d: %d tracks whose in-group slice is not in their HostGroup"
                     % (tag, t, np.count_nonzero(host[trow] != gid[tg])))
        elif not hg_ok:
            ck.warn("%s type %d: %d tracks whose in-group slice is not in their HostGroup (expected in an "
                    "incomplete PIG: the table is from the last complete call)" % (tag, t, np.count_nonzero(host[trow] != gid[tg])))
        nslices = np.count_nonzero(S["SliceLengthByType"][:, t] > 0)
        ck.check(nslices == len(ts), "%s type %d: %d tracks claim an in-group slice, %d found" % (tag, t, nslices, len(ts)))
        _check_bound_prefix(ck, tag, t, "Slice", S, trow, ts, tl, tt, bt, br)

        # section 1: field slices by (TrackId, BindRank)
        f_tr, f_br = tr[ng:], br[ng:]
        nfield_rows += f_tr.shape[0]
        ck.check(np.all(f_tr >= 0), "%s type %d: %d untagged rows in section 1" % (tag, t, np.count_nonzero(f_tr < 0)))
        if f_tr.shape[0] > 1:
            order_ok = (f_tr[1:] > f_tr[:-1]) | ((f_tr[1:] == f_tr[:-1]) & (f_br[1:] >= f_br[:-1]))
            ck.check(np.all(order_ok), "%s type %d: section 1 not ordered by (TrackId, BindRank)" % (tag, t))
        fstart, flen = _runs(f_tr)
        fstart = fstart + ng
        ftrack = tr[fstart] if fstart.size else np.zeros(0, np.int64)
        frows = np.searchsorted(st, ftrack)
        fknown = (frows < nsub) & (st[np.minimum(frows, max(nsub - 1, 0))] == ftrack) if nsub else np.zeros(len(ftrack), bool)
        if np.any(~fknown):
            orph = np.isin(ftrack[~fknown], ot)
            field_rows_of_orphans += int(flen[~fknown][orph].sum())
            ck.check(np.all(orph), "%s type %d: section 1 rows of unknown TrackIds %s" % (tag, t, ftrack[~fknown][~orph][:5]))
        fs, fl, ft, frow = fstart[fknown], flen[fknown], ftrack[fknown], frows[fknown]
        ck.check(np.array_equal(S["FieldSliceOffsetByType"][frow, t], fs) and np.array_equal(S["FieldSliceLengthByType"][frow, t], fl),
                 "%s type %d: FieldSliceOffsetByType/FieldSliceLengthByType do not match section 1 (%d mismatches)"
                 % (tag, t, np.count_nonzero((S["FieldSliceOffsetByType"][frow, t] != fs) | (S["FieldSliceLengthByType"][frow, t] != fl))))
        nfslices = np.count_nonzero(S["FieldSliceLengthByType"][:, t] > 0)
        ck.check(nfslices == len(fs), "%s type %d: %d tracks claim a field slice, %d found" % (tag, t, nfslices, len(fs)))
        _check_bound_prefix(ck, tag, t, "FieldSlice", S, frow, fs, fl, ft, bt, None)

        if phase1:
            ck.check(np.all(tr[:ng] >= 0), "%s type %d: %d untagged rows inside groups (Phase 1 tags every group member)"
                     % (tag, t, np.count_nonzero(tr[:ng] < 0)))
            ck.check(np.array_equal(bt[:ng], tr[:ng]), "%s type %d: %d in-group rows with BoundTrackId != TrackId"
                     % (tag, t, np.count_nonzero(bt[:ng] != tr[:ng])))
            sec0_tracks.append(tr[:ng])
            sec0_ranks.append(br[:ng])
            sec0_ids.append(ids[:ng])

    # expected emptiness of section 1. With "auto", PIG_000 must be empty (no call before it)
    # and a later empty section 1 is only a warning: it is legitimately empty at the first call
    # after which tracks exist (every track was just born in a group, so no tagged particle is
    # outside one yet). --expect-field nonempty makes it a hard rule.
    mode = expect_field
    if mode == "auto":
        mode = "empty" if snap == 0 else "nonempty-warn"
    if mode == "empty":
        ck.check(nfield_rows == 0, "%s: section 1 has %d rows, expected empty" % (tag, nfield_rows))
    elif mode == "nonempty":
        ck.check(nfield_rows > 0, "%s: section 1 is empty, expected tagged particles outside groups" % tag)
    elif mode == "nonempty-warn" and nfield_rows == 0:
        ck.warn("%s: section 1 is empty (no tagged particle outside groups); expected after the first "
                "call that had tracks, use --expect-field nonempty to require it" % tag)
    print("  %s: %d groups, %d subhalos, %d orphans, %d field rows (%d of orphan tracks)"
          % (tag, len(gid), nsub, norph, nfield_rows, field_rows_of_orphans))

    if not phase1:
        return
    # every FOF group is one subhalo with Nbound = Length
    hosted = host >= 0
    hg, hc = np.unique(host[hosted], return_counts=True)
    ck.check(np.array_equal(hg, gid), "%s: hosted subhalos cover %d groups of %d" % (tag, len(hg), len(gid)))
    ck.check(np.all(hc == 1), "%s: %d groups host more than one subhalo" % (tag, np.count_nonzero(hc > 1)))
    hrows = np.nonzero(hosted)[0]
    gr = np.searchsorted(gid, host[hrows])
    okg = gr < len(gid)
    gr, hrows = gr[okg], hrows[okg]
    ck.check(np.array_equal(S["Nbound"][hrows], length[gr]), "%s: %d hosted subhalos with Nbound != group Length"
             % (tag, np.count_nonzero(S["Nbound"][hrows] != length[gr])))
    ck.check(np.array_equal(S["NboundType"][hrows], lbt[gr]), "%s: NboundType != LengthByType for %d subhalos"
             % (tag, np.count_nonzero(np.any(S["NboundType"][hrows] != lbt[gr], axis=1))))
    ck.check(np.array_equal(S["SliceLengthByType"][hrows], lbt[gr].astype(S["SliceLengthByType"].dtype)),
             "%s: in-group slices do not tile the group ranges" % tag)
    ck.check(np.all(S["Rank"][hrows] == 0) and np.all(S["CentralTrackId"][hrows] == st[hrows]),
             "%s: a hosted Phase 1 subhalo is not a central" % tag)
    # hostless: Nbound = own-tag field particles, no in-group slice
    hl = np.nonzero(~hosted)[0]
    if hl.size:
        ck.check(np.all(S["SliceLengthByType"][hl] == 0), "%s: hostless subhalos with an in-group slice" % tag)
        ck.check(np.array_equal(S["Nbound"][hl], S["FieldSliceLengthByType"][hl].sum(axis=1)),
                 "%s: %d hostless subhalos with Nbound != own-tag field rows"
                 % (tag, np.count_nonzero(S["Nbound"][hl] != S["FieldSliceLengthByType"][hl].sum(axis=1))))
    # BindRank dense over all types of a key; MostBoundID is the BindRank-0 particle
    if sec0_tracks:
        T = np.concatenate(sec0_tracks)
        R = np.concatenate(sec0_ranks).astype(np.int64)
        I = np.concatenate(sec0_ids)
        if T.size:
            o = np.lexsort((R, T))
            T, R, I = T[o], R[o], I[o]
            ut, first, cnt = np.unique(T, return_index=True, return_counts=True)
            pos = np.arange(T.size) - np.repeat(first, cnt)
            dense = R == pos
            ck.check(np.all(dense), "%s: BindRank not dense 0..n-1 within %d tracks"
                     % (tag, len(np.unique(T[~dense]))))
            rows = np.searchsorted(st, ut)
            okr = (rows < nsub) & (st[np.minimum(rows, max(nsub - 1, 0))] == ut)
            mb = I[first][okr]
            ck.check(np.array_equal(S["MostBoundID"][rows[okr]], mb), "%s: MostBoundID is not the BindRank-0 particle for %d tracks"
                     % (tag, np.count_nonzero(S["MostBoundID"][rows[okr]] != mb)))


def _check_bound_prefix(ck, tag, t, prefix, S, trow, starts, lengths, tracks, bt, br):
    """SliceNbound/FieldSliceNbound (audit H13): nb equals the number of rows of the slice bound to
    the track (BoundTrackId == TrackId), those rows are exactly the first nb (prefix all bound,
    suffix none), and BindRank (if given) is non-decreasing within the slice."""
    nb = S[prefix + "NboundByType"][trow, t].astype(np.int64)
    ck.check(np.all((nb >= 0) & (nb <= lengths)), "%s type %d: %sNboundByType outside [0, length]" % (tag, t, prefix))
    ncount = nprefix = nsuffix = nrank = 0
    examples = []
    for s, l, k, tid in zip(starts.tolist(), lengths.tolist(), nb.tolist(), tracks.tolist()):
        own = bt[s:s + l] == tid
        nown = int(np.count_nonzero(own))
        if nown != k:
            ncount += 1
            if len(examples) < 5:
                examples.append("%d: %d vs %d" % (tid, k, nown))
        if k > 0 and not np.all(own[:k]):
            nprefix += 1
        if np.any(own[max(k, 0):]):
            nsuffix += 1
        if br is not None and l > 1 and np.any(np.diff(br[s:s + l]) < 0):
            nrank += 1
    ck.check(ncount == 0, "%s type %d: %d %s slices whose %sNboundByType differs from their rows bound to the track "
             "(TrackId: recorded vs counted %s)" % (tag, t, ncount, prefix, prefix, ", ".join(examples)))
    ck.check(nprefix == 0, "%s type %d: %d %s bound prefixes contain rows bound to another track" % (tag, t, nprefix, prefix))
    ck.check(nsuffix == 0, "%s type %d: %d %s slices have rows bound to the track after the bound prefix" % (tag, t, nsuffix, prefix))
    if br is not None:
        ck.check(nrank == 0, "%s type %d: %d %s slices not ordered by BindRank" % (tag, t, nrank, prefix))


def cmd_consistency(args):
    rc = 0
    for p in args.pigdirs:
        ck = Checker("consistency %s" % p)
        check_consistency(p, ck, phase1=not args.no_phase1, expect_field=args.expect_field,
                          allow_incomplete=args.allow_incomplete)
        rc |= ck.report()
    return rc


# ---------------------------------------------------------------------------------- tracking
def _all_types(cat, names, section=None):
    """Concatenate per-particle blocks over types; section 0 = in-group rows, 1 = field rows."""
    out = {n: [] for n in names}
    for t in range(NTYPES):
        ns = cat.nsaved(t)
        if ns == 0 or not cat.has_particle_tags(t):
            continue
        ng = cat.ningroup(t)
        sl = slice(0, ns) if section is None else (slice(0, ng) if section == 0 else slice(ng, ns))
        for n in names:
            if n == "GroupRow":
                out[n].append(np.repeat(np.arange(cat.ngroups), cat.groups["LengthByType"][:, t])[sl]
                              if section != 1 else np.full(ns - ng, -1))
            else:
                out[n].append(cat.read(t, n)[sl])
    return {n: (np.concatenate(v) if v else np.zeros(0, np.int64)) for n, v in out.items()}


def cmd_tracking(args):
    rule = "Phase 1 majority vote" if args.phase1_vote else "Phase 2a weighted ten-tracer vote"
    ck = Checker("tracking %s -> %s (%s)" % (args.prev, args.next, rule))
    try:
        A, B = SubhaloCatalogue(args.prev), SubhaloCatalogue(args.next)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2
    ck.check(A.complete() and B.complete(), "both PIGs must be complete (%s, %s)"
             % (attr(A.header, "SubhaloStateComplete"), attr(B.header, "SubhaloStateComplete")))
    if args.phase1_vote:
        _tracking_phase1(ck, A, B, args)
    else:
        _tracking_phase2a(ck, A, B, args)
    return ck.report()


def _tracking_phase1(ck, A, B, args):
    """The Phase 1 rules (halofinder_phase1_plan.md section 6, check 3)."""
    ha, hb = A.header, B.header
    snap_next = int(attr(hb, "SnapshotIndex", snapnum_of(args.next)))
    ntr_a = int(attr(ha, "NumTracksTotal"))
    ntr_b = int(attr(hb, "NumTracksTotal"))

    # host vote (audit H12): over the current population (every saved row of the next PIG, plus
    # --part), with pre-call tags joined from the previous PIG; previously tagged particles absent
    # from the next PIG are omitted (Phase 1 saves every surviving tagged particle, so they were
    # removed). Largest count, ties to the smaller GroupID; "no group" only with a strict majority.
    pop = precall_population(A, B, part=args.part, keep_absent=False)
    gid_b = B.groups["GroupID"]
    bound = (pop["TrackId"] >= 0) & (pop["BoundTrackId"] == pop["TrackId"])
    vtr, vgrp = pop["TrackId"][bound], pop["GroupID"][bound]
    print("  vote population: %d current particles (%d spawned children inheriting a tag), %d previously tagged "
          "particles absent and omitted" % (pop["n_current"], pop["n_inherited"], pop["n_absent"]))
    host = {}
    if vtr.size:
        pairs, counts = np.unique(np.stack([vtr, vgrp], axis=1), axis=0, return_counts=True)
        tids, tstart = np.unique(pairs[:, 0], return_index=True)
        tend = np.append(tstart[1:], pairs.shape[0])
        for T, s, e in zip(tids, tstart, tend):
            g, c = pairs[s:e, 1], counts[s:e]
            total = c.sum()
            nog = c[g == -1].sum()
            if 2 * nog > total:
                host[int(T)] = -1
                continue
            m = g >= 0
            if not np.any(m):
                host[int(T)] = -1
                continue
            g, c = g[m], c[m]
            best = c.max()
            host[int(T)] = int(g[c == best].min())
    SA, SB, OB = A.subhalos, B.subhalos, B.orphans
    nbound_prev = dict(zip(SA["TrackId"].tolist(), SA["Nbound"].tolist()))

    # expected winner per next group. Only tracks alive in the previous PIG are candidates: an
    # orphan (sunk or dead) keeps the tags of its leftover field particles but never wins a group
    # and never sinks again (Phase 1 never resurrects orphans); a group whose only voters are
    # orphans is newborn.
    alive_prev = set(int(x) for x in SA["TrackId"].tolist())
    cand = {}
    for T, g in host.items():
        if g >= 0 and T in alive_prev:
            cand.setdefault(g, []).append(T)
    hostg = SB["HostGroup"]
    central = SB["CentralTrackId"] == SB["TrackId"]
    actual = {}
    for r in np.nonzero((hostg >= 0) & central)[0]:
        actual.setdefault(int(hostg[r]), []).append(int(SB["TrackId"][r]))
    newborn = []
    nwrong = 0
    for g in gid_b.tolist():
        act = actual.get(g, [])
        if not ck.check(len(act) == 1, "group %d hosts %d central subhalos" % (g, len(act))):
            continue
        a = act[0]
        if g in cand:
            exp = min(cand[g], key=lambda T: (-nbound_prev.get(T, 0), T))
            if a != exp:
                nwrong += 1
                if nwrong <= 20:
                    ck.fail("group %d: winner TrackId %d, expected %d (candidates %s with previous Nbound %s)"
                            % (g, a, exp, cand[g][:8], [nbound_prev.get(T, 0) for T in cand[g][:8]]))
            # sinks
            for T in cand[g]:
                if T == exp:
                    continue
                r = B.orphan_row(T)
                if ck.check(r >= 0, "track %d (host %d, not the winner) is not an orphan in %s" % (T, g, args.next)):
                    ck.check(OB["SinkTrackId"][r] == a and OB["SnapSink"][r] == snap_next,
                             "track %d sank into %d at %d, expected %d at %d"
                             % (T, OB["SinkTrackId"][r], OB["SnapSink"][r], a, snap_next))
        else:
            newborn.append((g, a))
    if nwrong > 20:
        ck.fail("... %d groups with the wrong winner in all" % nwrong)
    # newborn numbering: GrNr order, continuing the counter
    newborn.sort()
    got = [a for _, a in newborn]
    expect = list(range(ntr_a, ntr_a + len(newborn)))
    ck.check(got == expect, "newborn TrackIds %s... are not %s... (GrNr order from NumTracksTotal %d)"
             % (got[:8], expect[:8], ntr_a))
    ck.check(ntr_b == ntr_a + len(newborn), "NumTracksTotal %d, expected %d + %d newborn" % (ntr_b, ntr_a, len(newborn)))
    # every previous track persists; previous orphans stay orphans (Phase 1)
    for T in A.subhalos["TrackId"].tolist():
        ck.check(B.subhalo_row(T) >= 0 or B.orphan_row(T) >= 0, "track %d vanished" % T)
    for T in A.orphans["TrackId"].tolist():
        ck.check(B.orphan_row(T) >= 0, "orphan %d was resurrected or lost" % T)
    # hostless tracks: keep their field particles, or die
    nf = _all_types(B, ["TrackId"], section=1)["TrackId"]
    ftid, fcnt = np.unique(nf, return_counts=True) if nf.size else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    fcount = dict(zip(ftid.tolist(), fcnt.tolist()))
    winners = set(a for _, a in newborn) | set(int(x) for x in SB["TrackId"][(hostg >= 0)])
    nhostless = 0
    for T in A.subhalos["TrackId"].tolist():
        if host.get(int(T), -1) != -1 or T in winners:
            continue
        nhostless += 1
        if fcount.get(T, 0) > 0:
            r = B.subhalo_row(T)
            if ck.check(r >= 0, "hostless track %d has %d field particles but no Subhalo row" % (T, fcount[T])):
                ck.check(SB["HostGroup"][r] == -1 and SB["Nbound"][r] == fcount[T],
                         "hostless track %d: HostGroup %d Nbound %d, expected -1 and %d"
                         % (T, SB["HostGroup"][r], SB["Nbound"][r], fcount[T]))
        else:
            r = B.orphan_row(T)
            if ck.check(r >= 0, "track %d has no particles left but is not an orphan" % T):
                ck.check(OB["SnapDeath"][r] == snap_next, "dead track %d: SnapDeath %d, expected %d"
                         % (T, OB["SnapDeath"][r], snap_next))
    print("  %d groups, %d continued, %d newborn, %d hostless/dead candidates, %d new orphans"
          % (len(gid_b), len(gid_b) - len(newborn), len(newborn), nhostless, B.norphans - A.norphans))


# ---------------------------------------------------------------------------------- Phase 2a helpers
def particles(cat, extra=()):
    """Every saved row of every type with the tag blocks, as one dict of arrays:
    Type, Row (row in <t>/), Section (0 in a group, 1 field), GroupID (the row's FOF GroupID, -1 in
    section 1), ID, TrackId, BindRank (int64), BoundTrackId, plus the blocks named in extra
    (e.g. "Mass") where present (NaN-free zeros otherwise)."""
    names = ["ID", "TrackId", "BindRank", "BoundTrackId"] + list(extra)
    out = {n: [] for n in names + ["Type", "Row", "Section", "GroupID"]}
    gid = cat.groups["GroupID"]
    for t in range(NTYPES):
        ns = cat.nsaved(t)
        if ns == 0 or not cat.has_particle_tags(t):
            continue
        ng = cat.ningroup(t)
        out["Type"].append(np.full(ns, t, dtype=np.int32))
        out["Row"].append(np.arange(ns, dtype=np.int64))
        sec = np.ones(ns, dtype=np.int8)
        sec[:ng] = 0
        out["Section"].append(sec)
        g = np.full(ns, -1, dtype=np.int64)
        g[:ng] = np.repeat(gid, cat.groups["LengthByType"][:, t])
        out["GroupID"].append(g)
        for n in names:
            if n in ("ID", "TrackId", "BindRank", "BoundTrackId") or "%d/%s" % (t, n) in cat.blocks:
                a = cat.read(t, n)
                if a.shape[0] != ns:
                    raise SubhaloFormatError("%s: %d/%s has %d rows, %d saved" % (cat.pigdir, t, n, a.shape[0], ns))
                out[n].append(a)
            else:
                out[n].append(np.zeros(ns, dtype=np.float64))
    res = {}
    for n, v in out.items():
        res[n] = np.concatenate(v) if v else np.zeros(0, dtype=np.int64)
    res["BindRank"] = res["BindRank"].astype(np.int64)
    res["TrackId"] = res["TrackId"].astype(np.int64)
    res["BoundTrackId"] = res["BoundTrackId"].astype(np.int64)
    return res


def lookup(sorted_keys, keys, values=None, default=-1):
    """Positions (or values[position]) of keys in a sorted array; default where absent."""
    keys = np.asarray(keys)
    out = np.full(keys.shape, default, dtype=np.int64 if values is None else np.asarray(values).dtype)
    if sorted_keys.size == 0 or keys.size == 0:
        return out
    pos = np.searchsorted(sorted_keys, keys)
    posc = np.minimum(pos, sorted_keys.size - 1)
    hit = sorted_keys[posc] == keys
    out[hit] = posc[hit] if values is None else np.asarray(values)[posc[hit]]
    return out


ID_GEN_SHIFT = np.uint64(56)
ID_BASE_MASK = np.uint64((1 << 56) - 1)


def precall_population(A, B, part=None, keep_absent=True):
    """The particles alive at the call that wrote PIG B, with their pre-call tags (audit H12).

    Current population: every saved row of B (both sections), or with part = PART_xxx of the same
    snapshot every particle in it (types and IDs; group membership still from B's section 0).
    Pre-call tags are joined from A as the restart join does (subhalo_io.c): an exact saved ID
    match takes the saved record (an untagged saved record gives untagged); otherwise a spawned
    child (generation ID >> 56 > 0) inherits the record of its generation-0 base ID
    (ID & (2^56 - 1)) when that is saved; otherwise untagged. GroupID is the current FOF GroupID
    (-1 outside groups).

    Previously tagged rows of A whose ID is not in the current population: without part they may
    be removed particles or live field particles that the call left untagged (not saved); with
    keep_absent they are appended with Absent = True and their A tags (GroupID -1), else dropped.
    Returns a dict of arrays (ID, Type, GroupID, TrackId, BoundTrackId, BindRank, Absent) plus the
    counts n_current, n_inherited, n_absent."""
    PB = particles(B)
    sec0 = PB["Section"] == 0
    if part is not None:
        fp = bigfile.File(part)
        ids, typ = [], []
        for t in range(NTYPES):
            if "%d/ID" % t in fp.blocks:
                a = fp["%d/ID" % t][:]
                ids.append(a.astype(np.uint64))
                typ.append(np.full(a.shape[0], t, np.int32))
        cid = np.concatenate(ids) if ids else np.zeros(0, np.uint64)
        ctyp = np.concatenate(typ) if typ else np.zeros(0, np.int32)
        so = np.argsort(PB["ID"][sec0].astype(np.uint64))
        cgid = lookup(PB["ID"][sec0].astype(np.uint64)[so], cid, PB["GroupID"][sec0][so], default=-1)
    else:
        cid, ctyp = PB["ID"].astype(np.uint64), PB["Type"]
        cgid = PB["GroupID"].astype(np.int64)
    PA = particles(A)
    aid = PA["ID"].astype(np.uint64)
    so = np.argsort(aid, kind="stable")
    aid = aid[so]
    arow = lookup(aid, cid)
    gen = cid >> ID_GEN_SHIFT
    base = cid & ID_BASE_MASK
    brow = lookup(aid, base)
    inherit = (arow < 0) & (gen > 0) & (brow >= 0)
    src = np.where(arow >= 0, arow, np.where(inherit, brow, -1))
    out = dict(ID=cid, Type=ctyp, GroupID=cgid)
    for n in ("TrackId", "BoundTrackId", "BindRank"):
        v = PA[n][so]
        col = np.full(cid.size, -1, dtype=np.int64)
        col[src >= 0] = v[src[src >= 0]]
        out[n] = col
    out["Absent"] = np.zeros(cid.size, bool)
    cs = np.sort(cid)
    absent = (PA["TrackId"][so] >= 0) & (lookup(cs, aid) < 0)
    out["n_current"] = int(cid.size)
    out["n_inherited"] = int(np.count_nonzero(inherit & (out["TrackId"] >= 0)))
    out["n_absent"] = int(np.count_nonzero(absent))
    if keep_absent and np.any(absent):
        out["ID"] = np.concatenate([cid, aid[absent]])
        out["Type"] = np.concatenate([ctyp, PA["Type"][so][absent]])
        out["GroupID"] = np.concatenate([cgid, np.full(np.count_nonzero(absent), -1, np.int64)])
        for n in ("TrackId", "BoundTrackId", "BindRank"):
            out[n] = np.concatenate([out[n], PA[n][so][absent]])
        out["Absent"] = np.concatenate([out["Absent"], np.ones(np.count_nonzero(absent), bool)])
    return out


def tracer_votes(pop, ntracers, tracer_types):
    """Phase 2a host vote (plan section 1 steps 0 and 2) over a precall_population: the tracers of
    track T are the ntracers lowest-BindRank (ties by ID) particles of a tracer type whose pre-call
    tags have TrackId == BoundTrackId == T; tracer i votes with weight 1/(1 + sqrt(i)) for its
    current GroupID (-1 outside groups); the largest weight wins, ties to the smaller GroupID (-1
    first). Returns dict T -> (GroupID, near_tie, inconclusive): near_tie flags a runner-up within
    1e-6; inconclusive flags a track one of whose ten tracers is an Absent row (removed, or a live
    field particle left untagged and not saved: only a PART of the snapshot can tell)."""
    sel = (pop["TrackId"] >= 0) & (pop["BoundTrackId"] == pop["TrackId"]) & np.isin(pop["Type"], tracer_types)
    T, R, I = pop["TrackId"][sel], pop["BindRank"][sel], pop["ID"][sel]
    G, X = pop["GroupID"][sel], pop["Absent"][sel]
    o = np.lexsort((I, R, T))
    T, G, X = T[o], G[o], X[o]
    votes = {}
    if T.size == 0:
        return votes
    ut, first, cnt = np.unique(T, return_index=True, return_counts=True)
    pos = np.arange(T.size) - np.repeat(first, cnt)
    keep = pos < ntracers
    T, G, X, pos = T[keep], G[keep], X[keep], pos[keep]
    inconcl = set(np.unique(T[X]).tolist())
    w = 1.0 / (1.0 + np.sqrt(pos.astype(np.float64)))
    pairs, inv = np.unique(np.stack([T, G], axis=1), axis=0, return_inverse=True)
    wsum = np.bincount(inv.ravel(), weights=w, minlength=pairs.shape[0])
    o = np.lexsort((pairs[:, 1], -wsum, pairs[:, 0]))
    pairs, wsum = pairs[o], wsum[o]
    tids, tstart, tcnt = np.unique(pairs[:, 0], return_index=True, return_counts=True)
    for T_, s_, c_ in zip(tids.tolist(), tstart.tolist(), tcnt.tolist()):
        near = c_ > 1 and abs(wsum[s_] - wsum[s_ + 1]) < 1e-6
        votes[T_] = (int(pairs[s_, 1]), bool(near), T_ in inconcl)
    return votes


def _group_members(cat):
    """(ID, GroupID) of every section-0 row of every type, i.e. FOF membership at this PIG."""
    P = particles(cat)
    m = P["Section"] == 0
    return P["ID"][m], P["GroupID"][m]


def _tracking_phase2a(ck, A, B, args):
    """Phase 2a rules (halofinder_phase2a_plan.md section 1 steps 1, 2, 7b, 7d, 10)."""
    snap_next = int(attr(B.header, "SnapshotIndex", snapnum_of(args.next)))
    ntr_a = int(attr(A.header, "NumTracksTotal"))
    ntr_b = int(attr(B.header, "NumTracksTotal"))
    ntracers = args.ntracers or A.int_parameter("SubhaloNumTracerHostFinding", 10)
    ratio = float(B.parameters().get("SubhaloMajorProgenitorMassRatio", 0.8))
    SA, OA, SB, OB = A.subhalos, A.orphans, B.subhalos, B.orphans
    gid_b = B.groups["GroupID"]
    mid, mgid = _group_members(B)
    pop = precall_population(A, B, part=args.part, keep_absent=args.part is None)
    votes = tracer_votes(pop, ntracers, A.tracer_types())
    print("  vote: %d lowest-BindRank own-bound tracers of types %s, weights 1/(1+sqrt(i)); "
          "major-progenitor ratio %g" % (ntracers, A.tracer_types(), ratio))
    print("  vote population: %d current particles (%d spawned children inheriting a tag), %d previously tagged "
          "particles absent from %s%s" % (pop["n_current"], pop["n_inherited"], pop["n_absent"],
                                          "the PART" if args.part else "the PIG",
                                          " (omitted)" if args.part else " (a vote they enter is inconclusive)"))
    inconcl = sorted(T for T, v in votes.items() if v[2])
    if inconcl:
        ck.warn("%d tracks have an absent particle among their tracers; their vote is inconclusive without --part "
                "(e.g. %s)" % (len(inconcl), inconcl[:8]))

    tid_sb, tid_ob = SB["TrackId"], OB["TrackId"]
    # 1. every previous track persists (resolved or orphan)
    prev_all = np.concatenate([SA["TrackId"], OA["TrackId"]])
    gone = prev_all[(lookup(tid_sb, prev_all) < 0) & (lookup(tid_ob, prev_all) < 0)]
    ck.check(gone.size == 0, "%d previous tracks are in neither table of %s (e.g. %s)"
             % (gone.size, args.next, gone[:8].tolist()))

    # host of each previous resolved track; tracks with no tracer at all are hostless
    host = {int(T): votes.get(int(T), (-1, False, False))[0] for T in SA["TrackId"].tolist()}
    # an inconclusive vote cannot be checked: the downstream checks take the track's own HostGroup
    for T in inconcl:
        rb = B.subhalo_row(T)
        if rb >= 0:
            host[T] = int(SB["HostGroup"][rb])
    notracer = [int(T) for T in SA["TrackId"].tolist() if int(T) not in votes]
    if notracer:
        ck.note("%d previous resolved tracks have no own-bound tracer (treated as hostless): %s"
                % (len(notracer), notracer[:8]))
    # 2. resolved previous tracks that are resolved now: HostGroup == vote
    nwrong = 0
    for r in np.nonzero(lookup(SA["TrackId"], tid_sb) >= 0)[0].tolist():
        T = int(tid_sb[r])
        exp, near, incon = votes.get(T, (-1, False, False))
        got = int(SB["HostGroup"][r])
        if got != exp:
            if incon:
                ck.warn("track %d: HostGroup %d, vote %d is inconclusive (absent tracer)" % (T, got, exp))
                continue
            if near:
                ck.warn("track %d: HostGroup %d, vote %d is a near tie" % (T, got, exp))
                continue
            nwrong += 1
            if nwrong <= 20:
                ck.fail("track %d: HostGroup %d, the weighted ten-tracer vote gives %d" % (T, got, exp))
    if nwrong > 20:
        ck.fail("... %d tracks with a HostGroup other than their vote" % nwrong)

    # 3. orphan hosts from their MostBoundID (step 1); resurrected orphans
    so = np.argsort(mid)
    orph_alive = OA["SnapTracerLost"] < 0 if "SnapTracerLost" in OA else np.ones(len(OA["TrackId"]), bool)
    orph_host = lookup(mid[so], OA["MostBoundID"].astype(mid.dtype), mgid[so], default=-1)
    orph_host[~orph_alive] = -1
    resurrected = OA["TrackId"][lookup(tid_sb, OA["TrackId"]) >= 0]
    for T in resurrected.tolist():
        ra, rb = A.orphan_row(T), B.subhalo_row(T)
        ck.check(orph_alive[ra], "orphan %d was resurrected although its tracer was lost at %d"
                 % (T, OA["SnapTracerLost"][ra]))
        ck.check(SB["HostGroup"][rb] == orph_host[ra] and SB["Rank"][rb] == 0,
                 "resurrected orphan %d: HostGroup %d Rank %d, expected the central of group %d "
                 "(the group holding its MostBoundID %d)" % (T, SB["HostGroup"][rb], SB["Rank"][rb],
                                                             orph_host[ra], OA["MostBoundID"][ra]))

    # 4. members per group and the central decision (step 7b)
    cand, ocand = {}, {}
    mb_prev = dict(zip(SA["TrackId"].tolist(), SA["Mbound"].astype(np.float64).tolist()))
    for T, g in host.items():
        if g >= 0:
            cand.setdefault(g, []).append(T)
    for T, g, pk in zip(OA["TrackId"].tolist(), orph_host.tolist(), OA["MboundPeak"].astype(np.float64).tolist()):
        if g >= 0:
            ocand.setdefault(g, []).append((T, pk))
    rank0 = {}
    hosted = SB["HostGroup"] >= 0
    for r in np.nonzero(hosted & (SB["Rank"] == 0))[0].tolist():
        rank0.setdefault(int(SB["HostGroup"][r]), []).append(int(tid_sb[r]))
    ncen_bad = 0
    for g in gid_b.tolist():
        cen = rank0.get(g, [])
        if not ck.check(len(cen) <= 1, "group %d has %d Rank 0 members %s" % (g, len(cen), cen[:5])):
            continue
        c = cen[0] if cen else None
        if g in cand:
            mmax = max(mb_prev[T] for T in cand[g])
            elig = sorted(T for T in cand[g] if mb_prev[T] >= ratio * mmax * (1 - 1e-6))
            if c is None:
                died = [T for T in elig if B.orphan_row(T) >= 0]
                if died:
                    ck.warn("group %d has no Rank 0 member; eligible candidates %s died or sank" % (g, died[:5]))
                else:
                    ck.fail("group %d has resolved candidates %s but no Rank 0 member" % (g, cand[g][:5]))
            elif c not in elig:
                ncen_bad += 1
                if ncen_bad <= 20:
                    ck.fail("group %d: central %d is not an eligible candidate %s (Mbound_prev >= %g x %g)"
                            % (g, c, elig[:8], ratio, mmax))
        elif g in ocand:
            exp = min(ocand[g], key=lambda x: (-x[1], x[0]))[0]
            if c is None:
                ck.warn("group %d: orphan %d should have been resurrected; no Rank 0 member (died again?)" % (g, exp))
            else:
                ck.check(c == exp, "group %d: central %d, expected the resurrected orphan %d (largest MboundPeak of %s)"
                         % (g, c, exp, [x[0] for x in ocand[g]][:8]))
        elif c is not None:
            ck.check(c >= ntr_a, "group %d has no previous member but its central %d is not newborn" % (g, c))
    if ncen_bad > 20:
        ck.fail("... %d groups with an ineligible central" % ncen_bad)

    # 5. newborns: only FOF centrals, in host GroupID order from NumTracksTotal(prev)
    nbr = np.nonzero(tid_sb >= ntr_a)[0]
    ck.check(np.all(SB["Rank"][nbr] == 0) and np.all(SB["HostGroup"][nbr] >= 0),
             "%d newborn tracks are not FOF centrals" % np.count_nonzero((SB["Rank"][nbr] != 0) | (SB["HostGroup"][nbr] < 0)))
    o = np.argsort(SB["HostGroup"][nbr], kind="stable")
    got = tid_sb[nbr][o].tolist()
    expect = list(range(ntr_a, ntr_a + len(nbr)))
    ck.check(got == expect, "newborn TrackIds by HostGroup %s... are not %s... (host order from NumTracksTotal %d)"
             % (got[:8], expect[:8], ntr_a))
    ck.check(ntr_b == ntr_a + len(nbr), "NumTracksTotal %d, expected %d + %d newborn" % (ntr_b, ntr_a, len(nbr)))
    ck.check(not np.any(tid_ob >= ntr_a), "%d orphans with newborn TrackIds (a newborn that dies gets no TrackId)"
             % np.count_nonzero(tid_ob >= ntr_a))
    bad_nb = [int(tid_sb[r]) for r in nbr.tolist() if int(SB["HostGroup"][r]) in cand or int(SB["HostGroup"][r]) in ocand]
    ck.check(not bad_nb, "newborn tracks %s in groups that had previous members" % bad_nb[:8])

    # 6. new orphans: died or sank at this call
    new_orph = OB["TrackId"][lookup(SA["TrackId"], tid_ob) >= 0]
    nsink = ndead = nchain = 0
    for T in new_orph.tolist():
        r = B.orphan_row(T)
        ck.check(OB["SnapDeath"][r] == snap_next, "track %d became an orphan with SnapDeath %d, expected %d"
                 % (T, OB["SnapDeath"][r], snap_next))
        s = int(OB["SinkTrackId"][r])
        if s >= 0:
            nsink += 1
            ck.check(OB["SnapSink"][r] == snap_next, "track %d sank with SnapSink %d, expected %d" % (T, OB["SnapSink"][r], snap_next))
            # SinkTrackId is the sink edge of the call, not the final owner: in a same-call chain
            # L -> P -> C the target P sank (or died) after L sank into it, so P is an orphan of
            # this call too (the tree check's rule, subhalo_track.c merge_subtree, the in-place
            # rounds). Follow the orphan targets that died in this call, without cycles, to the
            # first resolved one; its host is the vote's.
            seen = {T}
            t = s
            rs = B.subhalo_row(t)
            ok = True
            while rs < 0:
                ro = B.orphan_row(t)
                if not ck.check(ro >= 0 and OB["SnapDeath"][ro] == snap_next,
                                "track %d sank into %d, which is neither a resolved subhalo nor an orphan of this call" % (T, t)):
                    ok = False
                    break
                if not ck.check(t not in seen, "track %d: same-call sink chain with a cycle at %d" % (T, t)):
                    ok = False
                    break
                seen.add(t)
                nxt = int(OB["SinkTrackId"][ro])
                if nxt < 0:
                    break           # the chain ends in a track that died in this call
                t = nxt
                rs = B.subhalo_row(t)
            if t != s:
                nchain += 1
            if ok and rs >= 0:
                ck.check(SB["HostGroup"][rs] == host.get(T, -1),
                         "track %d (vote %d) sank into %d (via %d) hosted by %d" % (T, host.get(T, -1), t, s, SB["HostGroup"][rs]))
        else:
            ndead += 1
    if nchain:
        print("  %d sunk tracks whose sink target sank or died in the same call (same-call sink chains)" % nchain)
    # previous orphans that stay orphans keep their sink and death
    old_orph = OA["TrackId"][lookup(tid_ob, OA["TrackId"]) >= 0]
    for T in old_orph.tolist():
        ra, rb = A.orphan_row(T), B.orphan_row(T)
        ck.check(OA["SinkTrackId"][ra] == OB["SinkTrackId"][rb] and OA["SnapDeath"][ra] == OB["SnapDeath"][rb],
                 "orphan %d changed SinkTrackId/SnapDeath (%d, %d) -> (%d, %d)"
                 % (T, OA["SinkTrackId"][ra], OA["SnapDeath"][ra], OB["SinkTrackId"][rb], OB["SnapDeath"][rb]))
    nhostless = sum(1 for g in host.values() if g < 0)
    print("  %d groups, %d with previous resolved members, %d with only orphans, %d newborn, %d hostless votes, "
          "%d sank, %d died, %d resurrected" % (len(gid_b), len(cand), len(set(ocand) - set(cand)), len(nbr),
                                                nhostless, nsink, ndead, len(resurrected)))


# ---------------------------------------------------------------------------------- hierarchy
def check_hierarchy(pigdir, ck):
    try:
        cat = SubhaloCatalogue(pigdir)
    except SubhaloFormatError as e:
        ck.fail(str(e))
        return
    tag = os.path.basename(pigdir.rstrip("/"))
    S = cat.subhalos
    tid, host, par = S["TrackId"], S["HostGroup"], S["NestedParentTrackId"]
    cen, rank, depth, mb = S["CentralTrackId"], S["Rank"].astype(np.int64), S["Depth"].astype(np.int64), S["Mbound"]
    n = tid.size
    gid = cat.groups["GroupID"]
    hosted = host >= 0
    bad = hosted & ~np.isin(host, gid)
    ck.check(not np.any(bad), "%s: %d subhalos with a HostGroup that is not a FOFGroups GroupID (e.g. %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    ck.check(np.all(rank >= 0) and np.all(depth >= 0), "%s: %d negative Rank or Depth" % (tag, np.count_nonzero((rank < 0) | (depth < 0))))
    # parents
    has_par = par >= 0
    prow = lookup(tid, par)
    nores = has_par & (prow < 0)
    ck.check(not np.any(nores), "%s: %d subhalos whose NestedParentTrackId is not a resolved subhalo (e.g. %s -> %s)"
             % (tag, np.count_nonzero(nores), tid[nores][:5].tolist(), par[nores][:5].tolist()))
    ok = has_par & (prow >= 0)
    pr = prow[ok]
    bad = depth[ok] != depth[pr] + 1
    ck.check(not np.any(bad), "%s: %d subhalos with Depth != Depth(parent) + 1 (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[ok][bad][:5].tolist()))
    bad = host[ok] != host[pr]
    ck.check(not np.any(bad), "%s: %d subhalos whose parent is hosted by another group (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[ok][bad][:5].tolist()))
    bad = (depth == 0) != ~has_par
    ck.check(not np.any(bad), "%s: %d subhalos where Depth 0 and 'no parent' disagree (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    bad = (rank == 0) != (depth == 0)
    ck.check(not np.any(bad), "%s: %d subhalos where Rank 0 and Depth 0 disagree (the central is the root; e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    # hostless: own centrals (a pseudo-key holds one track)
    hl = ~hosted
    bad = hl & ((rank != 0) | (depth != 0) | has_par | (cen != tid))
    ck.check(not np.any(bad), "%s: %d hostless subhalos that are not their own central (Rank 0, Depth 0, no parent; e.g. %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    # per group: Rank dense, one central, CentralTrackId, Ranks 1.. by Mbound descending
    rows = np.nonzero(hosted)[0]
    o = rows[np.lexsort((rank[rows], host[rows]))]
    ndense = ncen = nmass = 0
    if o.size:
        ug, first, cnt = np.unique(host[o], return_index=True, return_counts=True)
        pos = np.arange(o.size) - np.repeat(first, cnt)
        dense = rank[o] == pos
        ndense = len(np.unique(host[o][~dense]))
        central_of = np.repeat(tid[o][first], cnt)
        okc = (rank[o][first] == 0)
        badc = (cen[o] != central_of) | ~np.repeat(okc, cnt)
        ncen = len(np.unique(host[o][badc]))
        # Mbound non-increasing over Ranks 1.. within a group
        same = np.zeros(o.size, bool)
        same[1:] = host[o][1:] == host[o][:-1]
        sat_pair = same & (pos >= 2)
        incr = sat_pair & (mb[o] > mb[np.roll(o, 1)])
        nmass = len(np.unique(host[o][incr]))
        print("  %s: %d subhalos in %d groups (max %d per group), %d hostless, max Depth %d, %d groups without subhalos"
              % (tag, n, len(ug), cnt.max(), np.count_nonzero(hl), depth.max() if n else 0, len(np.setdiff1d(gid, ug))))
    else:
        print("  %s: %d subhalos, none hosted" % (tag, n))
    ck.check(ndense == 0, "%s: Rank not dense 0..n-1 in %d groups" % (tag, ndense))
    ck.check(ncen == 0, "%s: in %d groups CentralTrackId is not the Rank 0 member" % (tag, ncen))
    ck.check(nmass == 0, "%s: in %d groups Ranks 1.. are not ordered by Mbound descending" % (tag, nmass))


def cmd_hierarchy(args):
    rc = 0
    for p in args.pigdirs:
        ck = Checker("hierarchy %s" % p)
        check_hierarchy(p, ck)
        rc |= ck.report()
    return rc


# ---------------------------------------------------------------------------------- tags
def check_tags(pigdir, ck, ntracers=None, tracer_rank=True, mass_rtol=1e-4, phase1=False):
    """With phase1, section 1 is ignored: Phase 1 never retags field particles, which keep the
    TrackId, BindRank and BoundTrackId of the last call that processed them, and its Nbound and
    NboundExclusive count the in-group slice only."""
    try:
        cat = SubhaloCatalogue(pigdir)
        P = particles(cat, extra=("Mass",))
    except SubhaloFormatError as e:
        ck.fail(str(e))
        return
    if phase1:
        # Phase 1 ranks by distance over every type, so the tracer rule does not apply either
        keep = P["Section"] == 0
        P = {k: v[keep] for k, v in P.items()}
        tracer_rank = False
    tag = os.path.basename(pigdir.rstrip("/"))
    S, O = cat.subhalos, cat.orphans
    tid = S["TrackId"]
    ntrk = tid.size
    T, B, R, typ = P["TrackId"], P["BoundTrackId"], P["BindRank"], P["Type"]
    ttypes = cat.tracer_types()
    istr = np.isin(typ, ttypes)
    if ntracers is None:
        ntracers = cat.int_parameter("SubhaloMinNumTracerPart", 10)

    # untagged rows are never bound; bound owners are resolved subhalos
    bad = (T < 0) & (B != -1)
    ck.check(not np.any(bad), "%s: %d untagged rows with BoundTrackId != -1" % (tag, np.count_nonzero(bad)))
    bown = B >= 0
    brow = lookup(tid, B[bown])
    ck.check(np.all(brow >= 0), "%s: %d rows bound to a track that is not a resolved subhalo (e.g. %s)"
             % (tag, np.count_nonzero(brow < 0), np.unique(B[bown][brow < 0])[:5].tolist()))
    # tags of orphans: only reported (Phase 1 leaves them on field particles)
    trow = lookup(tid, T)
    orph_tag = (T >= 0) & (trow < 0)
    if np.any(orph_tag):
        known = np.isin(T[orph_tag], O["TrackId"])
        ck.check(np.all(known), "%s: %d rows tagged by unknown tracks %s" % (tag, np.count_nonzero(~known), np.unique(T[orph_tag][~known])[:5].tolist()))
        ck.warn("%s: %d rows carry the tag of an orphan track (%d tracks)" % (tag, np.count_nonzero(orph_tag), len(np.unique(T[orph_tag]))))

    # BoundTrackId is the tag, an ancestor of the tag, or -1
    other = (T >= 0) & bown & (B != T)
    if np.any(other):
        all_tid = np.concatenate([tid, O["TrackId"]])
        all_par = np.concatenate([S["NestedParentTrackId"], O["NestedParentTrackId"]])
        so = np.argsort(all_tid)
        all_tid, all_par = all_tid[so], all_par[so]
        cur = T[other].copy()
        found = np.zeros(cur.size, bool)
        for _ in range(int(S["Depth"].max()) + 2 if ntrk else 1):
            cur = lookup(all_tid, cur, all_par, default=-1)
            found |= (cur == B[other]) & (cur >= 0)
            if not np.any((cur >= 0) & ~found):
                break
        ck.check(np.all(found), "%s: %d rows bound to a track that is neither their tag nor an ancestor of it "
                 "(e.g. tag %s bound %s)" % (tag, np.count_nonzero(~found), T[other][~found][:5].tolist(), B[other][~found][:5].tolist()))
    print("  %s: %d rows, %d tagged, %d bound, %d bound to an ancestor of their tag, tracer types %s"
          % (tag, T.size, np.count_nonzero(T >= 0), np.count_nonzero(bown), np.count_nonzero(other), ttypes))

    # own-tag lists of resolved tracks: BindRank dense, exclusive bound prefix first, tracer ranks
    own = (T >= 0) & (trow >= 0)
    o = np.nonzero(own)[0]
    o = o[np.lexsort((R[o], T[o]))]
    nown = np.zeros(ntrk, np.int64)
    if o.size:
        ut, first, cnt = np.unique(T[o], return_index=True, return_counts=True)
        rows = lookup(tid, ut)
        nown[rows] = cnt
        pos = np.arange(o.size) - np.repeat(first, cnt)
        dense = R[o] == pos
        ck.check(np.all(dense), "%s: own-tag BindRank not dense 0..n-1 in %d tracks (e.g. %s)"
                 % (tag, len(np.unique(T[o][~dense])), np.unique(T[o][~dense])[:5].tolist()))
        nbe = np.repeat(S["NboundExclusive"][rows].astype(np.int64), cnt)
        isb = B[o] == T[o]
        prefix_ok = isb == (pos < nbe)
        ck.check(np.all(prefix_ok), "%s: in %d tracks the own-bound rows (BoundTrackId == TrackId) are not exactly "
                 "BindRank 0..NboundExclusive-1 (e.g. %s)" % (tag, len(np.unique(T[o][~prefix_ok])), np.unique(T[o][~prefix_ok])[:5].tolist()))
        if tracer_rank:
            # J12: BindRank is the energy order of every own-tag particle; the reservation only
            # guarantees that the first ntracers tracer-type rows in rank order are bound to T
            to = o[istr[o]]
            if to.size:
                tut, tfirst, tcnt = np.unique(T[to], return_index=True, return_counts=True)
                tpos = np.arange(to.size) - np.repeat(tfirst, tcnt)
                badt = (tpos < ntracers) & (B[to] != T[to])
                ck.check(not np.any(badt), "%s: in %d tracks one of the first %d tracer-type (%s) rows in BindRank order "
                         "is not bound to the track (e.g. %s)"
                         % (tag, len(np.unique(T[to][badt])), ntracers, ttypes, np.unique(T[to][badt])[:5].tolist()))
                # MostBoundID is the first tracer-type own-tag particle in BindRank order
                if "MostBoundID" in S and "ID" in P:
                    trows = lookup(tid, tut)
                    mbid = P["ID"][to[tfirst]]
                    badm = S["MostBoundID"][trows].astype(np.uint64) != mbid.astype(np.uint64)
                    ck.check(not np.any(badm), "%s: %d tracks whose MostBoundID is not their first tracer-type own-tag "
                             "particle in BindRank order (e.g. %s)" % (tag, np.count_nonzero(badm), tut[badm][:5].tolist()))
        # per type exclusive and tail counts
        for t in range(NTYPES):
            m = typ[o] == t
            ex = np.bincount(lookup(tid, T[o][m & isb]), minlength=ntrk) if ntrk else np.zeros(0, np.int64)
            tl = np.bincount(lookup(tid, T[o][m & ~isb]), minlength=ntrk) if ntrk else np.zeros(0, np.int64)
            bad = ex != S["NboundExclusiveType"][:, t]
            ck.check(not np.any(bad), "%s type %d: NboundExclusiveType differs from the own-bound rows in %d tracks (e.g. %s)"
                     % (tag, t, np.count_nonzero(bad), tid[bad][:5].tolist()))
            bad = tl != S["NtailType"][:, t]
            ck.check(not np.any(bad), "%s type %d: NtailType differs from the own-tag rows not bound to the track in %d tracks (e.g. %s)"
                     % (tag, t, np.count_nonzero(bad), tid[bad][:5].tolist()))
            sl = S["SliceNboundByType"][:, t].astype(np.int64)
            if not phase1:
                sl = sl + S["FieldSliceNboundByType"][:, t]
            bad = sl != S["NboundExclusiveType"][:, t]
            ck.check(not np.any(bad), "%s type %d: SliceNbound + FieldSliceNbound != NboundExclusiveType in %d tracks"
                     % (tag, t, np.count_nonzero(bad)))
    if tracer_rank:
        few = nown < np.minimum(ntracers, S["Nbound"].astype(np.int64))
        ck.check(not np.any(few), "%s: %d tracks with fewer than %d own-tag rows" % (tag, np.count_nonzero(few), ntracers))
    nbe = S["NboundExclusive"].astype(np.int64)
    ck.check(np.array_equal(nbe, S["NboundExclusiveType"].sum(axis=1)), "%s: NboundExclusive != sum(NboundExclusiveType)" % tag)
    bad = nbe > S["Nbound"]
    ck.check(not np.any(bad), "%s: %d tracks with NboundExclusive > Nbound" % (tag, np.count_nonzero(bad)))

    # full bound lists from BoundTrackId: counts per type, tracers, mass, and inside the key
    b = np.nonzero(bown)[0]
    br = lookup(tid, B[b])
    okb = br >= 0
    b, br = b[okb], br[okb]
    for t in range(NTYPES):
        m = typ[b] == t
        cnt = np.bincount(br[m], minlength=ntrk) if ntrk else np.zeros(0, np.int64)
        bad = cnt != S["NboundType"][:, t]
        ck.check(not np.any(bad), "%s type %d: the full bound list (BoundTrackId) differs from NboundType in %d tracks (e.g. %s: %s vs %s)"
                 % (tag, t, np.count_nonzero(bad), tid[bad][:3].tolist(), cnt[bad][:3].tolist(), S["NboundType"][bad, t][:3].tolist()))
        if "%d/Mass" % t in cat.blocks and ntrk:
            ms = np.bincount(br[m], weights=P["Mass"][b][m].astype(np.float64), minlength=ntrk)
            bad = ~np.isclose(ms, S["MboundType"][:, t].astype(np.float64), rtol=mass_rtol, atol=0)
            ck.check(not np.any(bad), "%s type %d: MboundType differs from the full bound list's mass in %d tracks (e.g. %s)"
                     % (tag, t, np.count_nonzero(bad), tid[bad][:5].tolist()))
    ntr = np.bincount(br[istr[b]], minlength=ntrk) if ntrk else np.zeros(0, np.int64)
    bad = ntr != S["NboundTracer"]
    ck.check(not np.any(bad), "%s: NboundTracer differs from the tracer-type rows of the full bound list in %d tracks (e.g. %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    # inside the key: a section-0 row lies in the HostGroup of its bound owner; a section-1 row is
    # tagged by a track with the same HostGroup (hosted owner) or by the owner itself (hostless)
    owner_host = S["HostGroup"][br]
    sec0 = P["Section"][b] == 0
    bad0 = sec0 & (P["GroupID"][b] != owner_host)
    rtag = lookup(tid, T[b])
    tag_host = np.where(rtag >= 0, S["HostGroup"][np.maximum(rtag, 0)], -2)
    bad1 = ~sec0 & np.where(owner_host >= 0, tag_host != owner_host, T[b] != B[b])
    ck.check(not np.any(bad0), "%s: %d bound rows in a FOF group other than their owner's HostGroup (e.g. owners %s)"
             % (tag, np.count_nonzero(bad0), np.unique(B[b][bad0])[:5].tolist()))
    ck.check(not np.any(bad1), "%s: %d bound field rows outside their owner's key (e.g. owners %s)"
             % (tag, np.count_nonzero(bad1), np.unique(B[b][bad1])[:5].tolist()))


def cmd_tags(args):
    rc = 0
    for p in args.pigdirs:
        ck = Checker("tags %s" % p)
        check_tags(p, ck, ntracers=args.ntracers, tracer_rank=not args.no_tracer_rank, phase1=args.phase1)
        rc |= ck.report()
    return rc


# ---------------------------------------------------------------------------------- orphans
def check_orphans(pigdir, ck, tracer_pos=False):
    try:
        cat = SubhaloCatalogue(pigdir)
    except SubhaloFormatError as e:
        ck.fail(str(e))
        return
    tag = os.path.basename(pigdir.rstrip("/"))
    O, S = cat.orphans, cat.subhalos
    snap = int(attr(cat.header, "SnapshotIndex", snapnum_of(pigdir)))
    box = float(attr(cat.header, "BoxSize", np.nan))
    tid = O["TrackId"]
    n = tid.size
    alltid = np.sort(np.concatenate([S["TrackId"], tid]))
    pos = O["TracerPos"]
    if np.isfinite(box):
        bad = ~np.all((pos >= 0) & (pos < box), axis=1) if n else np.zeros(0, bool)
        ck.check(not np.any(bad), "%s: %d orphans with TracerPos outside [0, BoxSize = %g) (e.g. %s)"
                 % (tag, np.count_nonzero(bad), box, tid[bad][:5].tolist()))
    else:
        ck.fail("%s: Header has no BoxSize" % tag)
    ck.check(np.all(np.isfinite(pos)) and np.all(np.isfinite(O["TracerVel"])), "%s: non-finite TracerPos or TracerVel" % tag)
    lost = O["SnapTracerLost"].astype(np.int64)
    bad = (lost != -1) & ((lost < 0) | (lost > snap))
    ck.check(not np.any(bad), "%s: %d orphans with SnapTracerLost neither -1 nor in [0, %d] (e.g. %s)"
             % (tag, np.count_nonzero(bad), snap, tid[bad][:5].tolist()))
    death = O["SnapDeath"].astype(np.int64)
    bad = (death < 0) | (death > snap)
    ck.check(not np.any(bad), "%s: %d orphans with SnapDeath outside [0, %d]" % (tag, np.count_nonzero(bad), snap))
    bad = (O["SnapBirth"] < 0) | (O["SnapBirth"] > death)
    ck.check(not np.any(bad), "%s: %d orphans with SnapBirth < 0 or > SnapDeath" % (tag, np.count_nonzero(bad)))
    bad = (lost >= 0) & (lost < death)
    ck.check(not np.any(bad), "%s: %d orphans whose tracer was lost before they died" % (tag, np.count_nonzero(bad)))
    sink = O["SinkTrackId"]
    sunk = sink >= 0
    bad = sunk & ((O["SnapSink"] != O["SnapDeath"]) | (sink == tid))
    ck.check(not np.any(bad), "%s: %d sunk orphans with SnapSink != SnapDeath or sunk into themselves" % (tag, np.count_nonzero(bad)))
    bad = ~sunk & (O["SnapSink"] != -1)
    ck.check(not np.any(bad), "%s: %d orphans with SnapSink but no SinkTrackId" % (tag, np.count_nonzero(bad)))
    for col in ("SinkTrackId", "DescendantTrackId", "NestedParentTrackId", "CentralTrackId"):
        v = O[col]
        m = v >= 0
        bad = m & (lookup(alltid, v) < 0)
        ck.check(not np.any(bad), "%s: %d orphans whose %s is not an existing track (e.g. %s -> %s)"
                 % (tag, np.count_nonzero(bad), col, tid[bad][:5].tolist(), v[bad][:5].tolist()))
        ck.check(np.all(v >= -1), "%s: %s values below -1" % (tag, col))
    # TracerPos against the tracer particle when it is saved in this PIG (only after the step-1
    # lookup of Phase 2a; Phase 1 never updates TracerPos)
    P = particles(cat, extra=("Position",))
    so = np.argsort(P["ID"])
    r = lookup(P["ID"][so], O["MostBoundID"].astype(P["ID"].dtype)) if n else np.zeros(0, np.int64)
    have = (r >= 0) & (lost < 0)
    nmis = 0
    if np.any(have) and P["Position"].ndim == 2:
        ppos = P["Position"][so][r[have]]
        d = ppos - pos[have]
        d -= box * np.round(d / box)
        far = np.sqrt((d ** 2).sum(axis=1)) > 1e-6 * box
        nmis = int(np.count_nonzero(far))
    msg = "%s: %d of %d orphans with a live tracer saved in this PIG have TracerPos != its Position" % (tag, nmis, np.count_nonzero(have))
    if tracer_pos:
        ck.check(nmis == 0, msg)
    elif nmis:
        ck.warn(msg + " (fails with --tracer-pos; Phase 1 never updates TracerPos)")
    print("  %s: %d orphans (%d sunk, %d tracer lost), %d with the tracer saved in this PIG"
          % (tag, n, np.count_nonzero(sunk), np.count_nonzero(lost >= 0), np.count_nonzero(r >= 0)))


def cmd_orphans(args):
    rc = 0
    for p in args.pigdirs:
        ck = Checker("orphans %s" % p)
        check_orphans(p, ck, tracer_pos=args.tracer_pos)
        rc |= ck.report()
    return rc


# ---------------------------------------------------------------------------------- props (Phase 3)
GCGS = 6.674e-8
SO_NAMES = ("vir", "200b", "200c", "500c", "2500c")
# Phase 3 float columns whose defined values are positive (> 0) or, second set, non-negative
_PROPS_POSITIVE = ("MBoundSO", "RBoundSO", "VmaxVir", "RmaxVir", "Rhalf", "Renclose", "RsKlypin", "MSO", "RSO", "Rs")
_PROPS_NONNEG = ("SpinPeebles", "SpinBullock", "Xoff", "Voff", "Vrms")
_PROPS_FINITE = _PROPS_POSITIVE + _PROPS_NONNEG + (
    "RockstarCoreVel", "BulkVel", "J", "JVir", "BtoA", "CtoA", "BtoA500", "CtoA500", "AxisA", "AxisA500", "InertiaTensor",
    "MTypeSO200c", "MTypeSOVir", "StarFormationRate", "GasMetalMass", "StellarMetalMass", "BlackholeMass",
    "BlackholeAccretionRate", "MassHeIonized", "GasMetalElemMass", "StellarMetalElemMass")
FLAG_SOCUTOFF = 1 << 3
FLAG_SONOCROSS = 1 << 4


def so_thresholds(header):
    """(rho[5], source): Header SOThresholds (design 7.10, internal comoving densities) or, when absent,
    the formulas of design 7.10 from the Header cosmology without radiation (ordering and a 1e-2 check)."""
    thr = header.get("SOThresholds")
    if thr is not None and np.atleast_1d(thr).size == 5:
        return np.atleast_1d(thr).astype(np.float64), "Header SOThresholds"
    g = lambda k, d: float(attr(header, k, d))  # noqa: E731
    a, Om, OL = g("Time", 1.0), g("Omega0", 0.3), g("OmegaLambda", 0.7)
    ul, um, uv = g("UnitLength_in_cm", 3.085678e21), g("UnitMass_in_g", 1.989e43), g("UnitVelocity_in_cm_per_s", 1e5)
    ut = ul / uv
    G = GCGS * um * ut ** 2 / ul ** 3
    H0 = 100e5 / 3.085678e24 * ut
    E2 = Om / a ** 3 + OL + (1 - Om - OL) / a ** 2
    rhoc = 3 * H0 ** 2 * E2 * a ** 3 / (8 * np.pi * G)
    rhom = Om * 3 * H0 ** 2 / (8 * np.pi * G)
    x = Om / a ** 3 / E2 - 1
    dvir = 18 * np.pi ** 2 + 82 * x - 39 * x ** 2
    return np.array([dvir * rhoc, 200 * rhom, 200 * rhoc, 500 * rhoc, 2500 * rhoc]), "computed from the Header cosmology"


def _rows2d(v):
    """A column as [rows, items] (items = 1 for scalar columns), also for an empty table."""
    return v.reshape(v.shape[0], int(np.prod(v.shape[1:]))) if v.ndim > 1 else v[:, None]


def _defined(v):
    """Rows (or row-items) holding a value, i.e. not the -1 'not computed' sentinel."""
    return v != -1


def _check_so_set(ck, tag, tid, M, R, rho, rtol_r, name_m, name_r, rows):
    """Outermost-crossing rules for one SO set over the given rows: definedness and M, R
    non-increasing with increasing threshold density; R = (3 M / 4 pi rho)^(1/3)."""
    order = np.argsort(rho)
    Mo, Ro = M[rows][:, order], R[rows][:, order]
    t = tid[rows]
    dm, dr = _defined(Mo), _defined(Ro)
    bad = np.any(dm != dr, axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with %s and %s defined for different definitions (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), name_m, name_r, t[bad][:5].tolist()))
    # a looser threshold (lower density) has a crossing whenever a tighter one has
    bad = np.any(dm[:, 1:] & ~dm[:, :-1], axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos where %s is defined for a higher threshold but not a lower one "
             "(e.g. TrackId %s)" % (tag, np.count_nonzero(bad), name_m, t[bad][:5].tolist()))
    both = dm[:, 1:] & dm[:, :-1]
    bad = np.any(both & (Mo[:, 1:] > Mo[:, :-1] * (1 + 1e-5)), axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with %s increasing with the threshold density (order %s; e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), name_m, " < ".join(SO_NAMES[k] for k in order), t[bad][:5].tolist()))
    both = dr[:, 1:] & dr[:, :-1]
    bad = np.any(both & (Ro[:, 1:] > Ro[:, :-1] * (1 + 1e-5)), axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with %s increasing with the threshold density (order %s; e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), name_r, " < ".join(SO_NAMES[k] for k in order), t[bad][:5].tolist()))
    d = dm & dr
    with np.errstate(invalid="ignore", divide="ignore"):
        mr = 4.0 / 3 * np.pi * Ro ** 3 * rho[order][None, :] / Mo
    bad = np.any(d & (np.abs(mr - 1) > rtol_r), axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with %s != (3 %s / 4 pi rho)^(1/3) beyond %.0e in mass (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), name_r, name_m, rtol_r, t[bad][:5].tolist()))


def check_props(pigdir, ck, vmax_rtol=1e-3, field_warn=0.02):
    """Phase 3 (halofinder_phase3_plan.md section 6): internal consistency of the property columns.
    Every rule applies to defined values only, so a catalogue before Phase 3 (all -1) passes."""
    try:
        cat = SubhaloCatalogue(pigdir)
    except SubhaloFormatError as e:
        ck.fail(str(e))
        return
    tag = os.path.basename(pigdir.rstrip("/"))
    S = cat.subhalos
    tid = S["TrackId"]
    n = tid.size
    rho, src = so_thresholds(cat.header)
    rtol_r = 1e-3 if src.startswith("Header") else 1e-2
    central = (S["Rank"] == 0) & (S["HostGroup"] >= 0)
    flags = S["Flags"].astype(np.int64)
    # finite, -1 or in range
    for c in _PROPS_FINITE:
        v = S[c]
        bad = ~np.all(np.isfinite(_rows2d(v)), axis=1)
        ck.check(not np.any(bad), "%s: %d subhalos with non-finite %s (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), c, tid[bad][:5].tolist()))
    for c in _PROPS_POSITIVE + _PROPS_NONNEG:
        v = _rows2d(S[c])
        lo = 0.0 if c in _PROPS_NONNEG else np.nextafter(np.float32(0), np.float32(1))
        bad = np.any(_defined(v) & np.isfinite(v) & (v < lo), axis=1)
        ck.check(not np.any(bad), "%s: %d subhalos with %s neither -1 nor %s (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), c, ">= 0" if lo == 0 else "> 0", tid[bad][:5].tolist()))
    allrows = np.arange(n)
    _check_so_set(ck, tag, tid, S["MBoundSO"].astype(np.float64), S["RBoundSO"].astype(np.float64), rho, rtol_r,
                  "MBoundSO", "RBoundSO", allrows)
    mb = S["Mbound"].astype(np.float64)
    bad = np.any(_defined(S["MBoundSO"]) & (S["MBoundSO"] > mb[:, None] * (1 + 1e-5)), axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with MBoundSO > Mbound (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    vv, v = S["VmaxVir"], S["Vmax"]
    bad = _defined(vv) & _defined(v) & (vv > v * (1 + vmax_rtol))
    ck.check(not np.any(bad), "%s: %d subhalos with VmaxVir > Vmax (rtol %g; e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), vmax_rtol, tid[bad][:5].tolist()))
    rv = S["RBoundSO"][:, 0]
    bad = _defined(S["RmaxVir"]) & (~_defined(rv) | (S["RmaxVir"] > rv * (1 + 1e-5)))
    ck.check(not np.any(bad), "%s: %d subhalos with RmaxVir defined beyond RBoundSO[vir] or without it (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    bad = _defined(vv) != _defined(S["RmaxVir"])
    ck.check(not np.any(bad), "%s: %d subhalos with only one of VmaxVir, RmaxVir defined" % (tag, np.count_nonzero(bad)))
    bad = _defined(S["Rhalf"]) & _defined(S["Renclose"]) & (S["Rhalf"] > S["Renclose"] * (1 + 1e-6))
    ck.check(not np.any(bad), "%s: %d subhalos with Rhalf > Renclose (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    for suffix, rcol in (("", 0), ("500", 3)):
        b, c, ax = S["BtoA" + suffix], S["CtoA" + suffix], S["AxisA" + suffix].astype(np.float64)
        bad = _defined(b) != _defined(c)
        ck.check(not np.any(bad), "%s: %d subhalos with only one of BtoA%s, CtoA%s defined" % (tag, np.count_nonzero(bad), suffix, suffix))
        for name, r in (("BtoA" + suffix, b), ("CtoA" + suffix, c)):
            bad = _defined(r) & ~((r > 0) & (r <= 1 + 1e-6))
            ck.check(not np.any(bad), "%s: %d subhalos with %s outside (0, 1] (e.g. TrackId %s)"
                     % (tag, np.count_nonzero(bad), name, tid[bad][:5].tolist()))
        bad = _defined(b) & _defined(c) & (c > b * (1 + 1e-6))
        ck.check(not np.any(bad), "%s: %d subhalos with CtoA%s > BtoA%s (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), suffix, suffix, tid[bad][:5].tolist()))
        norm = np.sqrt((ax ** 2).sum(axis=1))
        bad = _defined(b) & (np.abs(norm - 1) > 1e-3)
        ck.check(not np.any(bad), "%s: %d subhalos with AxisA%s not a unit vector (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), suffix, tid[bad][:5].tolist()))
        bad = ~_defined(b) & ~np.all(ax == -1, axis=1)
        ck.check(not np.any(bad), "%s: %d subhalos with AxisA%s set but BtoA%s -1 (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), suffix, suffix, tid[bad][:5].tolist()))
        bad = _defined(b) & ~_defined(S["RBoundSO"][:, rcol])
        ck.check(not np.any(bad), "%s: %d subhalos with BtoA%s but no RBoundSO[%s] aperture"
                 % (tag, np.count_nonzero(bad), suffix, SO_NAMES[rcol]))
    it = S["InertiaTensor"]
    dit = ~np.all(it == -1, axis=1)
    bad = dit & np.any(it[:, :3] < 0, axis=1)
    ck.check(not np.any(bad), "%s: %d subhalos with a negative InertiaTensor diagonal (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    # field SO: centrals only
    fieldcols = ("MSO", "RSO", "MTypeSO200c", "MTypeSOVir")
    for c in fieldcols:
        bad = ~central & ~np.all(S[c] == -1, axis=1)
        ck.check(not np.any(bad), "%s: %d non-central subhalos (satellites or hostless) with %s != -1 (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), c, tid[bad][:5].tolist()))
    _check_so_set(ck, tag, tid, S["MSO"].astype(np.float64), S["RSO"].astype(np.float64), rho, rtol_r, "MSO", "RSO",
                  np.nonzero(central)[0])
    bad = ~central & ((flags & (FLAG_SOCUTOFF | FLAG_SONOCROSS)) != 0)
    ck.check(not np.any(bad), "%s: %d non-central subhalos with Flags bit 3 or 4 (field SO) set" % (tag, np.count_nonzero(bad)))
    anyso = np.any(_defined(S["MSO"]), axis=1)
    queried = central & anyso
    nocross = queried & np.any(~_defined(S["MSO"]), axis=1)
    bad = queried & (nocross != ((flags & FLAG_SONOCROSS) != 0))
    ck.check(not np.any(bad), "%s: %d centrals where Flags bit 4 disagrees with an MSO definition being -1 (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    bad = central & ~anyso & ((flags & FLAG_SOCUTOFF) != 0)
    ck.check(not np.any(bad), "%s: %d centrals with Flags bit 3 but no MSO" % (tag, np.count_nonzero(bad)))
    for c, k in (("MTypeSO200c", 2), ("MTypeSOVir", 0)):
        mt = S[c].astype(np.float64)
        dm = _defined(S["MSO"][:, k])
        bad = central & dm & (np.abs(mt.sum(axis=1) - S["MSO"][:, k]) > 1e-4 * np.abs(S["MSO"][:, k]) + 1e-12)
        ck.check(not np.any(bad), "%s: %d centrals with sum(%s) != MSO[%s] (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), c, SO_NAMES[k], tid[bad][:5].tolist()))
        bad = central & ~dm & ~np.all(mt == -1, axis=1)
        ck.check(not np.any(bad), "%s: %d centrals with %s set but MSO[%s] -1" % (tag, np.count_nonzero(bad), c, SO_NAMES[k]))
    # field >= bound up to the pass-1 bin resolution: a warning
    both = central[:, None] & _defined(S["MSO"]) & _defined(S["MBoundSO"])
    low = np.any(both & (S["MSO"] < S["MBoundSO"] * (1 - field_warn)), axis=1)
    if np.any(low):
        ck.warn("%s: %d centrals with MSO more than %.0f%% below MBoundSO for some definition (the sphere holds the "
                "bound set; e.g. TrackId %s)" % (tag, np.count_nonzero(low), 100 * field_warn, tid[low][:5].tolist()))
    # group C: sums of absent types are zero
    nbt = S["NboundType"]
    for c, t in (("StarFormationRate", 0), ("GasMetalMass", 0), ("MassHeIonized", 0), ("StellarMetalMass", 4),
                 ("BlackholeMass", 5), ("BlackholeAccretionRate", 5)):
        v = S[c]
        bad = _defined(v) & ((v < 0) | ((v > 0) & (nbt[:, t] == 0)))
        ck.check(not np.any(bad), "%s: %d subhalos with %s negative or non-zero without bound type-%d particles"
                 % (tag, np.count_nonzero(bad), c, t))
    # MostMassiveBHID: (MyIDType) -1, the only sentinel, without a bound black hole
    # (subhalo_props_baryons); 0 is a real particle ID
    bhid = S["MostMassiveBHID"]
    none = bhid == np.iinfo(bhid.dtype).max
    bad = ~none & (nbt[:, 5] == 0)
    ck.check(not np.any(bad), "%s: %d subhalos with a MostMassiveBHID but no bound black hole (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    bad = none & (nbt[:, 5] > 0) & _defined(S["BlackholeMass"])
    ck.check(not np.any(bad), "%s: %d subhalos with bound black holes and group C filled but no MostMassiveBHID (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    _check_props_deferred(ck, tag, S, tid)
    cnt = lambda c: int(np.count_nonzero(np.any(_defined(_rows2d(S[c])), axis=1)))  # noqa: E731
    print("  %s: %d subhalos (%d centrals); thresholds %s; defined: MBoundSO %d, VmaxVir %d, Rhalf %d, SpinPeebles %d, "
          "SpinBullock %d, RsKlypin %d, BtoA %d, BtoA500 %d, InertiaTensor %d, MSO %d (centrals), Flags bit 3 %d, bit 4 %d"
          % (tag, n, np.count_nonzero(central), src, cnt("MBoundSO"), cnt("VmaxVir"), cnt("Rhalf"), cnt("SpinPeebles"),
             cnt("SpinBullock"), cnt("RsKlypin"), cnt("BtoA"), cnt("BtoA500"), int(np.count_nonzero(dit)), cnt("MSO"),
             np.count_nonzero(flags & FLAG_SOCUTOFF), np.count_nonzero(flags & FLAG_SONOCROSS)))


# Rockstar's MIN_SCALE_PART (nfw.c): the NFW fit needs this many particles in the virial aperture
NFW_MINPART = 100


def _check_props_deferred(ck, tag, S, tid, corevel_warn=1.0):
    """Phase 5 (halofinder_phase5_plan.md section 1 item 1, subhalo_props_deferred): Rs, Vrms, BulkVel,
    RockstarCoreVel. Skipped (a note) when every value is -1, i.e. a catalogue written before Phase 5.
     - all four defined exactly when RBoundSO[vir] is (the aperture is the bound virial sphere), except
       Rs, which is -1 below NFW_MINPART bound particles (the aperture is a subset of the bound set) and
       otherwise > 0, or -1 (a warning: fewer than NFW_MINPART particles inside Rvir, or a failed fit);
     - Vrms >= 0 when RBoundSO[vir] > 0; Rs inside the aperture (below 1.5 RBoundSO[vir]: the outer bin
       edge may lie half a particle spacing beyond the crossing);
     - |RockstarCoreVel - CoreVel| above corevel_warn Vrms: a warning only (two different core definitions:
       r < 0.1 Rvir or the innermost 100 against the CoreSize most-bound particles)."""
    names = ("Rs", "Vrms", "BulkVel", "RockstarCoreVel")
    if any(c not in S for c in names):
        ck.note("%s: no Phase 5 columns (%s); the deferred-property rules are skipped" % (tag, ", ".join(c for c in names if c not in S)))
        return
    rs, vrms = S["Rs"].astype(np.float64), S["Vrms"].astype(np.float64)
    bv, rcv = S["BulkVel"].astype(np.float64), S["RockstarCoreVel"].astype(np.float64)
    if not (np.any(_defined(rs)) or np.any(_defined(vrms)) or np.any(_defined(bv)) or np.any(_defined(rcv))):
        ck.note("%s: Rs, Vrms, BulkVel, RockstarCoreVel all -1 (a catalogue before Phase 5); rules skipped" % tag)
        return
    rv = S["RBoundSO"][:, 0].astype(np.float64)
    hasr = _defined(rv) & (rv > 0)
    nb = S["Nbound"].astype(np.int64)
    for name, d in (("Vrms", _defined(vrms)), ("BulkVel", np.all(_defined(bv), axis=1)),
                    ("RockstarCoreVel", np.all(_defined(rcv), axis=1))):
        bad = d != hasr
        ck.check(not np.any(bad), "%s: %d subhalos with %s defined %s RBoundSO[vir] (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), name, "without" if np.any(d & ~hasr) else "-1 with", tid[bad][:5].tolist()))
    for name, v in (("BulkVel", bv), ("RockstarCoreVel", rcv)):
        bad = np.any(_defined(v), axis=1) & ~np.all(_defined(v), axis=1)
        ck.check(not np.any(bad), "%s: %d subhalos with %s partly -1 (e.g. TrackId %s)"
                 % (tag, np.count_nonzero(bad), name, tid[bad][:5].tolist()))
    bad = hasr & ~(vrms >= 0)
    ck.check(not np.any(bad), "%s: %d subhalos with RBoundSO[vir] > 0 but Vrms < 0 (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    drs = _defined(rs)
    bad = drs & (nb < NFW_MINPART)
    ck.check(not np.any(bad), "%s: %d subhalos with Rs defined below %d bound particles (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), NFW_MINPART, tid[bad][:5].tolist()))
    bad = drs & ~hasr
    ck.check(not np.any(bad), "%s: %d subhalos with Rs but no RBoundSO[vir] (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    bad = drs & hasr & ~(rs < 1.5 * rv)
    ck.check(not np.any(bad), "%s: %d subhalos with Rs beyond 1.5 RBoundSO[vir] (e.g. TrackId %s)"
             % (tag, np.count_nonzero(bad), tid[bad][:5].tolist()))
    big = nb >= NFW_MINPART
    miss = big & ~drs
    if np.any(miss):
        ck.warn("%s: %d of %d subhalos with Nbound >= %d have Rs = -1 (%d of them without RBoundSO[vir]; the others "
                "with fewer than %d particles inside Rvir or a failed fit; e.g. TrackId %s)"
                % (tag, np.count_nonzero(miss), np.count_nonzero(big), NFW_MINPART, np.count_nonzero(miss & ~hasr),
                   NFW_MINPART, tid[miss][:5].tolist()))
    if "CoreVel" in S:
        cv = S["CoreVel"].astype(np.float64)
        both = np.all(_defined(rcv), axis=1) & np.all(_defined(cv), axis=1) & _defined(vrms)
        dv = np.sqrt(((rcv - cv) ** 2).sum(axis=1))
        far = both & (dv > corevel_warn * vrms)
        if np.any(both):
            print("  %s: |RockstarCoreVel - CoreVel| / Vrms median %.3f, 90th percentile %.3f over %d subhalos"
                  % (tag, np.median(dv[both] / np.maximum(vrms[both], 1e-30)),
                     np.percentile(dv[both] / np.maximum(vrms[both], 1e-30), 90), np.count_nonzero(both)))
        if np.any(far):
            ck.warn("%s: %d subhalos with |RockstarCoreVel - CoreVel| > %g Vrms (different core definitions; e.g. TrackId %s)"
                    % (tag, np.count_nonzero(far), corevel_warn, tid[far][:5].tolist()))
    print("  %s: Phase 5 defined: Rs %d (of %d with Nbound >= %d), Vrms %d, BulkVel %d, RockstarCoreVel %d"
          % (tag, np.count_nonzero(drs), np.count_nonzero(big), NFW_MINPART, np.count_nonzero(_defined(vrms)),
             np.count_nonzero(np.all(_defined(bv), axis=1)), np.count_nonzero(np.all(_defined(rcv), axis=1))))


def cmd_props(args):
    rc = 0
    for p in args.pigdirs:
        ck = Checker("props %s" % p)
        check_props(p, ck, vmax_rtol=args.vmax_rtol)
        rc |= ck.report()
    return rc


# ---------------------------------------------------------------------------------- tree (Phase 4)
def check_tree(inputs, ck, show=5):
    """Phase 4 plan item 4: consistency of the track history over a PIG sequence (design 7.7 and
    subhalo_history_update). A node is (TrackId, output); `prev` is the same track's node at the
    previous output of the sequence."""
    from subhalo_reader import SubhaloSequence
    seq = SubhaloSequence(inputs[0] if len(inputs) == 1 and not os.path.basename(inputs[0].rstrip("/")).startswith("PIG_")
                          else inputs)
    if len(seq) == 0:
        ck.fail("no PIG with subhalo finder blocks in %s" % " ".join(inputs))
        return
    if seq.skipped:
        ck.note("skipped (no finder blocks): %s" % ", ".join(seq.skipped))
    N = seq.nodes()
    if "TrackId" not in N or N["TrackId"].size == 0:
        ck.note("no tracks in the sequence")
        return
    tid = N["TrackId"].astype(np.int64)
    out = N["Output"].astype(np.int64)
    snap = N["Snapshot"].astype(np.int64)
    res = N["Resolved"]
    nout = len(seq)
    first = np.ones(tid.size, bool)
    first[1:] = tid[1:] != tid[:-1]
    last = np.ones(tid.size, bool)
    last[:-1] = tid[1:] != tid[:-1]
    hasprev = np.zeros(tid.size, bool)
    hasprev[1:] = ~first[1:] & (out[1:] == out[:-1] + 1)
    pv = np.maximum(np.arange(tid.size) - 1, 0)
    prev_res = hasprev & res[pv]
    sat = res & (N["Rank"] > 0) & (N["HostGroup"] >= 0)
    cen = res & ~sat
    prev_snap_of = np.array([-1] + seq.snapshots[:-1], dtype=np.int64)
    print("  %d outputs (SnapshotIndex %s), %d tracks, %d nodes (%d resolved, %d orphan)" % (
        nout, seq.snapshots, np.unique(tid).size, tid.size, np.count_nonzero(res), np.count_nonzero(~res)))
    if np.any(np.diff(np.array(seq.snapshots)) != 1):
        ck.warn("the SnapshotIndex values are not consecutive: SnapLastIsolation = SnapNum - 1 at the first "
                "satellite output then differs from the previous output")

    def rule(mask, msg):
        bad = np.nonzero(mask)[0]
        ex = ", ".join("%d@%d" % (tid[k], snap[k]) for k in bad[:show])
        ck.check(bad.size == 0, "%s: %d nodes (TrackId@Snapshot %s)" % (msg, bad.size, ex))

    def resolved_at(o, t):
        """Is track t resolved at output o (arrays)."""
        key = out[res] * (int(tid.max()) + 2) + tid[res]
        ks = np.sort(key)
        q = np.asarray(o) * (int(tid.max()) + 2) + np.asarray(t)
        p = np.clip(np.searchsorted(ks, q), 0, max(ks.size - 1, 0))
        return (ks.size > 0) & (ks[p] == q) if ks.size else np.zeros(np.shape(q), bool)

    # node index of (output, track), -1 when the track has no node there (a negative or unknown
    # TrackId included)
    nkey_mul = int(tid.max()) + 2
    nkey = out * nkey_mul + tid
    nkey_order = np.argsort(nkey, kind="stable")
    nkey_sorted = nkey[nkey_order]

    def node_at(o, t):
        o, t = np.asarray(o, np.int64), np.asarray(t, np.int64)
        q = o * nkey_mul + t
        p = np.clip(np.searchsorted(nkey_sorted, q), 0, max(nkey_sorted.size - 1, 0))
        hit = (t >= 0) & (t < nkey_mul) & (nkey_sorted[p] == q)
        return np.where(hit, nkey_order[p], -1)

    # -- presence: a track appears at consecutive outputs up to the last one, born resolved
    gap = ~first & ~hasprev
    rule(gap, "track missing at an output between two of its nodes")
    rule(last & (out != nout - 1), "track absent from the last output (tracks are never deleted)")
    born_inside = first & (out > 0)
    rule(born_inside & ~res, "track first appears as an orphan after the first output")
    sb = N["SnapBirth"].astype(np.int64)
    rule(born_inside & res & (sb != snap), "SnapBirth differs from the output of the track's first appearance")
    rule(hasprev & (sb != sb[pv]), "SnapBirth changes along the branch")
    rule((sb < 0) | (sb > snap), "SnapBirth outside [0, Snapshot]")
    # -- resolved rows: identity
    rule(res & (N["DescendantTrackId"] != tid), "resolved node whose DescendantTrackId is not itself")
    rule(res & ((N["SnapDeath"] != -1) | (N["SnapSink"] != -1) | (N["SinkTrackId"] != -1)),
         "resolved node with SnapDeath, SnapSink or SinkTrackId set")
    # -- orphan rows: death episode
    death = N["SnapDeath"].astype(np.int64)
    ssink = N["SnapSink"].astype(np.int64)
    sinkt = N["SinkTrackId"].astype(np.int64)
    desct = N["DescendantTrackId"].astype(np.int64)
    dies = ~res & prev_res  # first orphan node after a resolved one
    cont = ~res & hasprev & ~res[pv]
    rule(dies & (death != snap), "SnapDeath is not the first output at which the track is an orphan")
    rule(cont & ((death != death[pv]) | (ssink != ssink[pv]) | (sinkt != sinkt[pv]) | (desct != desct[pv])),
         "SnapDeath, SnapSink, SinkTrackId or DescendantTrackId changes while the track stays an orphan")
    rule(~res & ((death < 0) | (death > snap)), "orphan with SnapDeath outside [0, Snapshot]")
    rule(~res & ~((ssink == -1) | (ssink == death)), "orphan with SnapSink neither -1 nor SnapDeath")
    rule(~res & ((sinkt >= 0) != (ssink >= 0)), "orphan with SinkTrackId and SnapSink inconsistent")
    rule(~res & (sinkt == tid), "orphan sunk into itself")
    # links at the death output (the value at output n describes the transition n-1 -> n)
    dout = np.full(tid.size, -1, np.int64)
    pos = np.clip(np.searchsorted(np.array(seq.snapshots), death), 0, nout - 1)
    inseq = ~res & (np.array(seq.snapshots)[pos] == death)
    dout[inseq] = pos[inseq]
    m = inseq & (desct >= 0)
    ok = resolved_at(dout[m], desct[m])
    bad = np.zeros(tid.size, bool)
    bad[np.nonzero(m)[0][~ok]] = True
    rule(bad, "DescendantTrackId not resolved at the SnapDeath output")
    # SinkTrackId is the historical sink edge (the ancestor the track sank into during the call),
    # not the final bound owner: the target need not survive the call. In a same-call chain
    # L -> P -> C, P sinks after L sank into it, so P is an orphan at the output too
    # (subhalo_track.c merge_subtree; HBT subhalo_merge.cpp). Required: the target has a node at
    # the SnapSink output; if it is an orphan there, it died in the same call (SnapDeath = this
    # SnapSink: only a track alive at the call's start takes sinkers); following orphan targets
    # never comes back to a track already on the chain.
    m = np.nonzero(inseq & (sinkt >= 0))[0]
    tgt = node_at(dout[m], sinkt[m])
    bad = np.zeros(tid.size, bool)
    bad[m[tgt < 0]] = True
    rule(bad, "SinkTrackId has no node (resolved or orphan) at the SnapSink output")
    hasn = tgt >= 0
    mo, to = m[hasn], tgt[hasn]
    otgt = ~res[to]
    bad = np.zeros(tid.size, bool)
    bad[mo[otgt & (death[to] != death[mo])]] = True
    rule(bad, "SinkTrackId target is an orphan at the SnapSink output that did not die in the same call")
    bad = np.zeros(tid.size, bool)
    for k, j in zip(mo[otgt].tolist(), to[otgt].tolist()):
        seen = {k}
        while j >= 0 and not res[j] and sinkt[j] >= 0 and death[j] == death[k]:
            if j in seen:
                bad[k] = True
                break
            seen.add(j)
            j = int(node_at(dout[k], sinkt[j]))
    rule(bad, "same-call sink chain with a cycle")
    nchain = int(np.count_nonzero(dies & (sinkt >= 0) & np.isin(np.arange(tid.size), mo[otgt])))
    if nchain:
        ck.note("%d sunk tracks whose sink target sank or died in the same call (same-call sink chains)" % nchain)
    rule(dies & (desct == tid), "dead track whose DescendantTrackId is itself")
    nsunk = np.count_nonzero(dies & (sinkt >= 0))
    ndiff = np.count_nonzero(dies & (sinkt >= 0) & (desct != sinkt))
    if ndiff:
        ck.note("%d of %d sunk tracks have DescendantTrackId != SinkTrackId (allowed, as in HBT)" % (ndiff, nsunk))
    # -- peaks: running maxima over the branch (subhalo_history_update)
    mb, vm = N["Mbound"], N["Vmax"]
    mp, vp = N["MboundPeak"], N["VmaxPeak"]
    smp, svp = N["SnapMboundPeak"].astype(np.int64), N["SnapVmaxPeak"].astype(np.int64)
    vamp = N["VmaxAtMboundPeak"]
    up = res & hasprev & (mb > mp[pv])
    keep = hasprev & ~up
    rule(up & ((mp != mb) | (smp != snap) | (vamp != vm)), "MboundPeak not raised to Mbound (or SnapMboundPeak/VmaxAtMboundPeak not set) when Mbound exceeds the previous peak")
    rule(keep & ((mp != mp[pv]) | (smp != smp[pv]) | (vamp != vamp[pv])), "MboundPeak, SnapMboundPeak or VmaxAtMboundPeak changed without a new maximum")
    up = res & hasprev & (vm > vp[pv])
    keep = hasprev & ~up
    rule(up & ((vp != vm) | (svp != snap)), "VmaxPeak not raised to Vmax when Vmax exceeds the previous peak")
    rule(keep & ((vp != vp[pv]) | (svp != svp[pv])), "VmaxPeak or SnapVmaxPeak changed without a new maximum")
    rule(res & ((mp < mb) | (vp < vm)), "peak below the current value")
    rule((smp > snap) | (svp > snap) | (smp < sb) | (svp < sb), "SnapMboundPeak or SnapVmaxPeak outside [SnapBirth, Snapshot]")
    b = born_inside & res
    rule(b & ((mp != mb) | (smp != snap) | (vp != vm) | (svp != snap)), "newborn whose peaks are not its current values")
    # -- accretion and isolation history against the Rank history
    sacc = N["SnapAcc"].astype(np.int64)
    sfa = N["SnapFirstAcc"].astype(np.int64)
    sli = N["SnapLastIsolation"].astype(np.int64)
    slmm = N["SnapLastMajorMerger"].astype(np.int64)
    macc, vacc = N["MboundAcc"], N["VmaxAcc"]
    mfa, vfa = N["MboundFirstAcc"], N["VmaxFirstAcc"]
    rule(cen & ((sacc != snap) | (macc != mb) | (vacc != vm)), "central or hostless node whose SnapAcc/MboundAcc/VmaxAcc are not its current state")
    rule(sat & hasprev & ((sacc != sacc[pv]) | (macc != macc[pv]) | (vacc != vacc[pv])), "satellite whose accretion state changed")
    rule(~res & hasprev & ((sacc != sacc[pv]) | (sfa != sfa[pv]) | (slmm != slmm[pv])),
         "orphan whose SnapAcc, SnapFirstAcc or SnapLastMajorMerger changed")
    # SnapLastIsolation is constant on orphan nodes after the death node; at the death node a
    # track that died or sank inside a group with another central is a satellite there (HBT,
    # subhalo_tracking.cpp: the new orphan is nested in its host and gets the previous output
    # if it had only ever been central), so -1 may become the previous output, nothing else.
    rule(cont & (sli != sli[pv]), "orphan whose SnapLastIsolation changed after its death output")
    hosted_by_other = dies & (N["HostGroup"] >= 0) & (N["CentralTrackId"] >= 0) & (N["CentralTrackId"] != tid)
    set_at_death = dies & (sli != sli[pv])
    rule(set_at_death & ((sli[pv] >= 0) | (sli != prev_snap_of[out])),
         "death node whose SnapLastIsolation changed other than from -1 to the previous output")
    newsat = sat & hasprev & (sfa[pv] < 0)
    rule(newsat & ((sfa != sacc[pv]) | (mfa != macc[pv]) | (vfa != vacc[pv])), "first satellite output whose SnapFirstAcc state is not the previous accretion state")
    rule(hasprev & (sfa[pv] >= 0) & ((sfa != sfa[pv]) | (mfa != mfa[pv]) | (vfa != vfa[pv])), "SnapFirstAcc state changed after it was set")
    rule(cen & hasprev & (sfa != sfa[pv]), "central whose SnapFirstAcc changed")
    rule(born_inside & res & (sfa != -1), "newborn with SnapFirstAcc set")
    rule((sfa >= 0) & ((sfa > snap) | (sfa < sb)), "SnapFirstAcc outside [SnapBirth, Snapshot]")
    # SnapAcc against the Rank history: the last output at which the track was a central
    last_cen = np.full(tid.size, -1, np.int64)
    cur = -1
    for k in range(tid.size):
        if first[k]:
            cur = -1
        if cen[k]:
            cur = snap[k]
        last_cen[k] = cur
    known = np.zeros(tid.size, bool)  # the track's whole past is in the sequence
    kn = False
    for k in range(tid.size):
        if first[k]:
            kn = born_inside[k] or (out[k] == 0 and sb[k] == snap[k])
        known[k] = kn
    rule(known & (last_cen >= 0) & (sacc != last_cen), "SnapAcc is not the last output at which the track was a central (Rank history)")
    # SnapLastIsolation (HBT): -1 while only ever central; the previous output at the first
    # satellite output; the current output whenever the track is central again
    rule(sat & hasprev & (sli[pv] < 0) & (sli != prev_snap_of[out]), "first satellite output whose SnapLastIsolation is not the previous output")
    rule(sat & hasprev & (sli[pv] >= 0) & (sli != sli[pv]), "satellite whose SnapLastIsolation changed")
    rule(cen & hasprev & (sli[pv] >= 0) & (sli != snap), "central (after a satellite phase) whose SnapLastIsolation is not the current output")
    rule(cen & hasprev & (sli[pv] < 0) & (sli != -1), "central (never a satellite) with SnapLastIsolation set")
    rule(sat & hasprev & (slmm != slmm[pv]), "satellite whose SnapLastMajorMerger changed")
    rule(cen & hasprev & (slmm != slmm[pv]) & (slmm != snap), "central whose SnapLastMajorMerger moved to an earlier output")
    rule((slmm > snap), "SnapLastMajorMerger after the output")
    # the death rule against the death node's host (HostGroup / CentralTrackId at the death
    # output): set exactly for the tracks that had only ever been central and died inside a group
    # with another central
    rule(set_at_death & ~hosted_by_other,
         "death node whose SnapLastIsolation was set although it did not die inside another central's group")
    rule(hosted_by_other & (sli[pv] < 0) & (sli == -1),
         "track that died inside another central's group keeps SnapLastIsolation = -1 (HBT sets the previous output)")
    nres = int(np.count_nonzero(res & hasprev & ~res[pv]))
    print("  %d deaths (%d sunk, %d with DescendantTrackId -1, %d inside another central's group, %d of them "
          "SnapLastIsolation set at death), %d resurrections, %d first-satellite events" % (
        np.count_nonzero(dies), nsunk, np.count_nonzero(dies & (desct < 0)), np.count_nonzero(hosted_by_other),
        np.count_nonzero(set_at_death), nres, np.count_nonzero(newsat)))


def cmd_tree(args):
    ck = Checker("tree %s" % " ".join(args.inputs))
    check_tree(args.inputs, ck, show=args.show)
    return ck.report()


def _add_tree_parser(sub):
    s = sub.add_parser("tree", help="Phase 4: track history consistency over a PIG sequence")
    s.add_argument("inputs", nargs="+", help="output directory holding PIG_* (or the PIG directories in order)")
    s.add_argument("--show", type=int, default=5, help="example nodes printed per failed rule (default 5)")


# ---------------------------------------------------------------------------------- compare
def _compare_tables(ck, name, ta, tb, rtol, atol, ignore, mask_inplace):
    if "TrackId" not in ta or "TrackId" not in tb:
        ck.fail("%s: TrackId column missing" % name)
        return
    oa, ob = np.argsort(ta["TrackId"], kind="stable"), np.argsort(tb["TrackId"], kind="stable")
    if not ck.check(np.array_equal(ta["TrackId"][oa], tb["TrackId"][ob]),
                    "%s: TrackId sets differ (%d vs %d rows)" % (name, len(oa), len(ob))):
        return
    cols = sorted(set(ta) | set(tb))
    for c in cols:
        if c in ignore:
            continue
        if not ck.check(c in ta and c in tb, "%s/%s present in only one catalogue" % (name, c)):
            continue
        a, b = ta[c][oa], tb[c][ob]
        if c == "Flags" and mask_inplace:
            a, b = a & ~FLAG_INPLACE, b & ~FLAG_INPLACE
        if np.issubdtype(a.dtype, np.floating) and rtol > 0:
            eq = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            ok = np.all(eq)
        else:
            eq = (a == b) | ((a != a) & (b != b))
            ok = np.all(eq)
        if not ok:
            bad = ~eq if eq.ndim == 1 else ~np.all(eq.reshape(eq.shape[0], -1), axis=1)
            rows = np.nonzero(bad)[0]
            extra = ""
            if np.issubdtype(a.dtype, np.floating):
                extra = ", max |diff| %.3e" % float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            ck.fail("%s/%s differs in %d rows (first TrackIds %s)%s" % (name, c, len(rows), ta["TrackId"][oa][rows[:5]].tolist(), extra))


def cmd_compare(args):
    ck = Checker("compare %s %s" % (args.pig_a, args.pig_b))
    try:
        A, B = SubhaloCatalogue(args.pig_a), SubhaloCatalogue(args.pig_b)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2
    rtol = 0.0 if args.exact else args.rtol
    ignore = set(args.ignore or [])
    for key in ("NumSubhalosTotal", "NumOrphansTotal", "NumTracksTotal", "SubhaloSchemaVersion", "SnapshotIndex"):
        va, vb = attr(A.header, key), attr(B.header, key)
        ck.check(np.array_equal(np.atleast_1d(va), np.atleast_1d(vb)), "Header %s: %s vs %s" % (key, va, vb))
    _compare_tables(ck, "Subhalo", A.subhalos, B.subhalos, rtol, args.atol, ignore, not args.compare_inplace_flag)
    _compare_tables(ck, "Orphan", A.orphans, B.orphans, rtol, args.atol, ignore, not args.compare_inplace_flag)
    # per-particle tags matched by ID, per type (same snapshot, so types agree)
    for t in range(NTYPES):
        na, nb_ = A.nsaved(t), B.nsaved(t)
        if na == 0 and nb_ == 0:
            continue
        if not ck.check(A.has_particle_tags(t) and B.has_particle_tags(t), "type %d: tag blocks missing" % t):
            continue
        ia, ib = A.read(t, "ID"), B.read(t, "ID")
        oa, ob = np.argsort(ia), np.argsort(ib)
        if not ck.check(np.array_equal(ia[oa], ib[ob]), "type %d: saved particle IDs differ (%d vs %d)" % (t, na, nb_)):
            continue
        for n in ("TrackId", "BindRank", "BoundTrackId", "GroupID"):
            if n == "GroupID" and ("%d/GroupID" % t not in A.blocks or "%d/GroupID" % t not in B.blocks):
                continue
            a, b = A.read(t, n)[oa], B.read(t, n)[ob]
            nd = int(np.count_nonzero(a != b))
            if nd and n == "BindRank":
                # say whether the differing particles belong to keys processed in place in either run
                inpl = set()
                for cat in (A, B):
                    fl = cat.subhalos["Flags"]
                    inpl |= set(cat.subhalos["TrackId"][(fl & FLAG_INPLACE) != 0].tolist())
                tr = A.read(t, "TrackId")[oa]
                d = np.nonzero(a != b)[0]
                nin = sum(1 for x in tr[d].tolist() if x in inpl)
                ck.fail("type %d: BindRank differs for %d particles (%d of them in tracks processed in place in a run)" % (t, nd, nin))
            else:
                ck.check(nd == 0, "type %d: %s differs for %d particles" % (t, n, nd))
    return ck.report()


# ---------------------------------------------------------------------------------- markincomplete
def _attr_file_set_i4(path, name, value):
    """Rewrite one i4 attribute in a BigFile attr-v2 file (format in the module docstring)."""
    with open(path) as f:
        lines = f.read().splitlines()
    hexval = np.array([value], dtype="<i4").tobytes().hex().upper()
    newline = "%s <i4 1 %s #HUMANE [ %d ]" % (name, hexval, value)
    out, done = [], False
    for ln in lines:
        if ln.split(" ", 1)[0] == name:
            out.append(newline)
            done = True
        else:
            out.append(ln)
    if not done:
        out.append(newline)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def cmd_markincomplete(args):
    hdr = os.path.join(args.pigdir, "Header")
    if not os.path.isdir(hdr):
        print("error: %s has no Header block" % args.pigdir)
        return 2
    f = bigfile.File(args.pigdir)
    old = attr(header_attrs(f), "SubhaloStateComplete")
    if old is None:
        print("error: %s Header has no SubhaloStateComplete attribute (not a finder PIG?)" % args.pigdir)
        return 2
    method = "python bigfile"
    try:
        h = f["Header"]
        h.attrs["SubhaloStateComplete"] = np.array([args.value], dtype="i4")
        h.close()
    except Exception as e:  # pragma: no cover - fallback path
        print("  python bigfile could not rewrite the attribute (%s); editing attr-v2 directly" % e)
        method = "attr-v2 text edit"
        _attr_file_set_i4(os.path.join(hdr, "attr-v2"), "SubhaloStateComplete", args.value)
    new = attr(header_attrs(bigfile.File(args.pigdir)), "SubhaloStateComplete")
    if int(new) != args.value:
        print("  rewrite through %s did not stick (read back %s); editing attr-v2 directly" % (method, new))
        _attr_file_set_i4(os.path.join(hdr, "attr-v2"), "SubhaloStateComplete", args.value)
        method = "attr-v2 text edit"
        new = attr(header_attrs(bigfile.File(args.pigdir)), "SubhaloStateComplete")
    print("%s: SubhaloStateComplete %s -> %s (%s)" % (args.pigdir, old, new, method))
    return 0 if int(new) == args.value else 1


# ---------------------------------------------------------------------------------- tagcounts
def cmd_tagcounts(args):
    try:
        cat = SubhaloCatalogue(args.pigdir, strict=False)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2
    rows = cat.tag_counts()
    print("%s (SnapshotIndex %s, complete %s)" % (args.pigdir, attr(cat.header, "SnapshotIndex"), attr(cat.header, "SubhaloStateComplete")))
    print("type    saved   in-group   tagged   (in groups, in field)   own-bound")
    tot = np.zeros(5, dtype=np.int64)
    for c in rows:
        tagged = c["tagged_group"] + c["tagged_field"]
        print("%4d %8d %10d %8d   (%9d, %8d)   %9d" % (c["ptype"], c["saved"], c["ingroup"], tagged,
                                                       c["tagged_group"], c["tagged_field"], c["own_bound"]))
        tot += [c["saved"], c["ingroup"], tagged, c["tagged_field"], c["own_bound"]]
    print(" all %8d %10d %8d   (%9d, %8d)   %9d" % (tot[0], tot[1], tot[2], tot[2] - tot[3], tot[3], tot[4]))
    print("tagged per type: %s" % " ".join(str(c["tagged_group"] + c["tagged_field"]) for c in rows))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("offon", help="check 1: finder off vs on")
    s.add_argument("offdir")
    s.add_argument("ondir")
    s.add_argument("--strict-bytes", action="store_true",
                   help="fail when PART_000 is not byte-identical (default: warn and compare by ID)")
    s.add_argument("--tol", type=float, default=None,
                   help="max displacement allowed in later PARTs (default: mean DM separation / 300, a tenth of a 1/30 softening)")
    s = sub.add_parser("consistency", help="check 2: reader rules on PIGs")
    s.add_argument("pigdirs", nargs="+")
    s.add_argument("--expect-field", choices=["auto", "empty", "nonempty", "any"], default="auto",
                   help="section 1 (tagged field particles): auto = must be empty at PIG_000, a warning if "
                        "empty later; nonempty = must be non-empty; empty = must be empty; any = no rule")
    s.add_argument("--allow-incomplete", action="store_true", help="do not fail on SubhaloStateComplete = 0")
    s.add_argument("--no-phase1", action="store_true", help="skip the Phase 1 rule (one subhalo per group, no tails)")
    s = sub.add_parser("tracking", help="check 3: tracking between consecutive PIGs")
    s.add_argument("prev")
    s.add_argument("next")
    s.add_argument("--phase1-vote", action="store_true",
                   help="Phase 1 rules: majority vote over every own-bound particle, the most massive candidate "
                        "wins and the others sink (default: Phase 2a weighted ten-tracer vote)")
    s.add_argument("--ntracers", type=int, default=None,
                   help="tracers per track in the vote (default SubhaloNumTracerHostFinding of the previous PIG, else 10)")
    s.add_argument("--part", default=None,
                   help="PART_xxx of the next snapshot: the live population for the vote (default: the next PIG's "
                        "saved rows; Phase 2a votes that need an unsaved particle are then inconclusive)")
    s = sub.add_parser("hierarchy", help="Phase 2a: Rank/Depth/parent/central consistency")
    s.add_argument("pigdirs", nargs="+")
    s = sub.add_parser("tags", help="Phase 2a: D10 tag rule; BindRank the energy order of all own-tag particles "
                                    "(bound prefix first), the first tracer-type ones bound with MostBoundID the "
                                    "first of them; BoundTrackId; full bound lists")
    s.add_argument("pigdirs", nargs="+")
    s.add_argument("--ntracers", type=int, default=None,
                   help="tracer-type particles, first in BindRank order, that must be bound to the track "
                        "(default SubhaloMinNumTracerPart, else 10)")
    s.add_argument("--no-tracer-rank", action="store_true",
                   help="skip the tracer rule (the first tracer-type particles bound, MostBoundID the first of them; "
                        "Phase 1 hydro PIGs rank by distance)")
    s.add_argument("--phase1", action="store_true",
                   help="Phase 1 PIGs: ignore section 1, whose rows keep the stale tags of the last call that processed "
                        "them, and skip the tracer-rank rule (Phase 1 ranks by distance over every type)")
    s = sub.add_parser("orphans", help="Phase 2a: Orphan/* consistency")
    s.add_argument("pigdirs", nargs="+")
    s.add_argument("--tracer-pos", action="store_true",
                   help="fail when TracerPos differs from the saved tracer's Position (Phase 2a; a warning otherwise)")
    s = sub.add_parser("props", help="Phase 3: consistency of the property and field SO columns")
    s.add_argument("pigdirs", nargs="+")
    s.add_argument("--vmax-rtol", type=float, default=1e-3, help="tolerance of VmaxVir <= Vmax (default 1e-3)")
    s = sub.add_parser("compare", help="check 5: two PIGs of the same snapshot are equal")
    s.add_argument("pig_a")
    s.add_argument("pig_b")
    s.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for float columns (default 1e-6)")
    s.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for float columns")
    s.add_argument("--exact", action="store_true", help="bitwise equality for float columns too")
    s.add_argument("--ignore", action="append", help="column to skip (repeatable)")
    s.add_argument("--compare-inplace-flag", action="store_true",
                   help="also compare the in-place Flags bit (depends on the rank count)")
    s = sub.add_parser("markincomplete", help="check 6 helper: set SubhaloStateComplete")
    s.add_argument("pigdir")
    s.add_argument("--value", type=int, default=0)
    s = sub.add_parser("tagcounts", help="tagged particles per type")
    s.add_argument("pigdir")
    _add_tree_parser(sub)
    args = p.parse_args(argv)
    cmds = dict(offon=cmd_offon, consistency=cmd_consistency, tracking=cmd_tracking, compare=cmd_compare,
                markincomplete=cmd_markincomplete, tagcounts=cmd_tagcounts, hierarchy=cmd_hierarchy,
                tags=cmd_tags, orphans=cmd_orphans, props=cmd_props)
    cmds["tree"] = cmd_tree
    if args.cmd not in cmds:
        p.print_help()
        return 2
    try:
        return cmds[args.cmd](args)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
