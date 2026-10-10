#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripted integration checks for Phase 1 of the runtime subhalo finder
(halofinder_phase1_plan.md section 6). Every subcommand exits 0 on success and 1 on a failed
check, printing each failure; 2 means the input could not be read.

  offon OFFDIR ONDIR          check 1: PART_000 byte-identical between the finder-off and -on
                              runs (a warning plus a by-ID roundoff check if not, since MP-Gadget
                              is not bitwise reproducible run to run; --strict-bytes to fail);
                              later PART_nnn matched by ID (max displacement report);
                              PIG_000 FOFGroups identical, finder blocks present and complete.
  consistency PIGDIR [...]    check 2: the reader rules on one or more PIGs (groups = subhalos,
                              slices tile the groups, BindRank dense, bound prefixes, counts).
  tracking PIG_prev PIG_next  check 3: winners, newborn numbering, sinks and hostless tracks,
                              recomputed from the two PIGs with the Phase 1 rule.
  compare PIG_A PIG_B         check 5: every Subhalo/* and Orphan/* column equal after sorting by
                              TrackId, and the per-particle tags equal when matched by ID.
  markincomplete PIGDIR       check 6 helper: set Header SubhaloStateComplete (default 0).
  tagcounts PIGDIR            tagged particles per type (to compare with the run log, check 4).

Examples (from the directory holding the run outputs):
  python3 tools/subhalo_check_phase1.py offon off/output on/output
  python3 tools/subhalo_check_phase1.py consistency on/output/PIG_00?
  python3 tools/subhalo_check_phase1.py tracking on/output/PIG_000 on/output/PIG_001
  python3 tools/subhalo_check_phase1.py compare on/output/PIG_001 offline/output/PIG_001
  python3 tools/subhalo_check_phase1.py markincomplete restart/output/PIG_001
  python3 tools/subhalo_check_phase1.py tagcounts on/output/PIG_001

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
    """SliceNbound/FieldSliceNbound: the first nb rows of the slice are bound to the track,
    and BindRank (if given) is non-decreasing within the slice."""
    nb = S[prefix + "NboundByType"][trow, t].astype(np.int64)
    ck.check(np.all((nb >= 0) & (nb <= lengths)), "%s type %d: %sNboundByType outside [0, length]" % (tag, t, prefix))
    nbad = 0
    nrank = 0
    for s, l, k, tid in zip(starts, lengths, nb, tracks):
        if k > 0 and not np.all(bt[s:s + k] == tid):
            nbad += 1
        if br is not None and l > 1 and np.any(np.diff(br[s:s + l]) < 0):
            nrank += 1
    ck.check(nbad == 0, "%s type %d: %d %s bound prefixes contain rows bound to another track" % (tag, t, nbad, prefix))
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
    ck = Checker("tracking %s -> %s" % (args.prev, args.next))
    try:
        A, B = SubhaloCatalogue(args.prev), SubhaloCatalogue(args.next)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2
    ha, hb = A.header, B.header
    snap_next = int(attr(hb, "SnapshotIndex", snapnum_of(args.next)))
    ck.check(A.complete() and B.complete(), "both PIGs must be complete (%s, %s)"
             % (attr(ha, "SubhaloStateComplete"), attr(hb, "SubhaloStateComplete")))
    ntr_a = int(attr(ha, "NumTracksTotal"))
    ntr_b = int(attr(hb, "NumTracksTotal"))

    # previous tags (all saved rows: every tagged particle is saved) and next group membership
    pa = _all_types(A, ["ID", "TrackId", "BoundTrackId"])
    nb = _all_types(B, ["ID", "GroupRow"], section=0)
    gid_b = B.groups["GroupID"]
    bound = (pa["TrackId"] >= 0) & (pa["BoundTrackId"] == pa["TrackId"])
    vid, vtr = pa["ID"][bound], pa["TrackId"][bound]
    order = np.argsort(nb["ID"])
    sid, sgr = nb["ID"][order], nb["GroupRow"][order]
    pos = np.searchsorted(sid, vid)
    found = (pos < sid.size) & (sid[np.minimum(pos, max(sid.size - 1, 0))] == vid) if sid.size else np.zeros(vid.size, bool)
    vgrp = np.full(vid.size, -1, dtype=np.int64)
    vgrp[found] = gid_b[sgr[pos[found]]]

    # host vote: largest count, ties to the smaller GroupID; "no group" only with a strict majority
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
    return ck.report()


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
    args = p.parse_args(argv)
    cmds = dict(offon=cmd_offon, consistency=cmd_consistency, tracking=cmd_tracking, compare=cmd_compare,
                markincomplete=cmd_markincomplete, tagcounts=cmd_tagcounts)
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
