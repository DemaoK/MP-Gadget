#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reader for the runtime subhalo finder's catalogues in MP-Gadget PIG_xxx directories.

Schema: halofinder_design.md section 9 and Appendix B (SubhaloSchemaVersion 1).

Layout of a PIG written with SubhaloFinderOn = 1:

  FOFGroups/*            unchanged FOF tables, one row per group in GroupID order
  Subhalo/<column>       one row per resolved track (Nbound > 0), sorted by TrackId
  Orphan/<column>        one row per orphan track (Nbound == 0), sorted by TrackId
  <t>/TrackId (i8), <t>/BindRank (i4), <t>/BoundTrackId (i8), next to <t>/ID, GroupID, ...
                         for every saved particle of type t, in two sections:
                           section 0: every FOF group in GroupID order, exactly LengthByType[g, t]
                                      rows each (cumulating LengthByType addresses the groups);
                                      within a group untagged rows first, then each track's
                                      in-group slice ordered by BindRank;
                           section 1: rows NumPartInGroupTotal[t] .. NumPartSavedTotal[t]: tagged
                                      particles outside any group, ordered by (TrackId, BindRank),
                                      one contiguous field slice per track.
  Header attrs           NumSubhalosTotal, NumOrphansTotal, NumTracksTotal, NumPartSavedTotal,
                         SnapshotIndex, SubhaloLastCallSnapshot, SubhaloStateSnapshot,
                         SubhaloSchemaVersion, SubhaloStateComplete, SubhaloStateTime,
                         TimeSincePreviousFinderCall, SubhaloParameters, ...

A track's in-group slice in <t>/ is rows [SliceOffsetByType[t], + SliceLengthByType[t]) and its
bound prefix the first SliceNboundByType[t] of them; FieldSlice* do the same in section 1.

Library use:

    from subhalo_reader import SubhaloCatalogue
    cat = SubhaloCatalogue("output/PIG_002")
    cat.summary()
    rows = cat.exclusive_bound(17, 1)          # row indices into PIG_002/1/
    ids = cat.read(1, "ID")[rows]

CLI:  python3 tools/subhalo_reader.py PIG_001 [--track N]
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import bigfile

NTYPES = 6

# Appendix B, PIG_xxx/Subhalo/ : (name, dtype, items per row)
SUBHALO_COLUMNS = [
    ("TrackId", "i8", 1), ("HostGroup", "i8", 1), ("NestedParentTrackId", "i8", 1),
    ("CentralTrackId", "i8", 1), ("SinkTrackId", "i8", 1), ("DescendantTrackId", "i8", 1),
    ("Rank", "i4", 1), ("Depth", "i4", 1),
    ("SnapBirth", "i4", 1), ("SnapDeath", "i4", 1), ("SnapSink", "i4", 1), ("SnapLastIsolation", "i4", 1),
    ("MostBoundID", "u8", 1),
    ("Nbound", "i4", 1), ("NboundExclusive", "i4", 1), ("NboundTracer", "i4", 1),
    ("Flags", "i4", 1),
    ("NboundType", "i4", 6), ("NboundExclusiveType", "i4", 6), ("NtailType", "i4", 6),
    ("SliceOffsetByType", "i8", 6), ("SliceLengthByType", "i4", 6), ("SliceNboundByType", "i4", 6),
    ("FieldSliceOffsetByType", "i8", 6), ("FieldSliceLengthByType", "i4", 6), ("FieldSliceNboundByType", "i4", 6),
    ("Mbound", "f4", 1), ("MboundType", "f4", 6),
    ("CorePos", "f8", 3), ("MostBoundPos", "f8", 3), ("CoMPos", "f8", 3),
    ("CoreVel", "f4", 3), ("MostBoundVel", "f4", 3), ("CoMVel", "f4", 3),
    ("CoreSigmaX", "f4", 1), ("CoreSigmaV", "f4", 1),
    ("Vmax", "f4", 1), ("Rmax", "f4", 1), ("VmaxVir", "f4", 1), ("RmaxVir", "f4", 1),
    ("Rhalf", "f4", 1), ("Renclose", "f4", 1),
    ("MBoundSO", "f4", 5), ("RBoundSO", "f4", 5),
    ("J", "f4", 3), ("JVir", "f4", 3),
    ("SpinPeebles", "f4", 1), ("SpinBullock", "f4", 1), ("Ekin", "f4", 1), ("Epot", "f4", 1),
    ("KinToPot", "f4", 1), ("Xoff", "f4", 1), ("Voff", "f4", 1), ("RsKlypin", "f4", 1),
    ("BtoA", "f4", 1), ("CtoA", "f4", 1), ("BtoA500", "f4", 1), ("CtoA500", "f4", 1),
    ("AxisA", "f4", 3), ("AxisA500", "f4", 3),
    ("InertiaTensor", "f4", 6),
    ("Rs", "f4", 1), ("RockstarCoreVel", "f4", 3), ("BulkVel", "f4", 3), ("Vrms", "f4", 1),
    ("StarFormationRate", "f4", 1), ("GasMetalMass", "f4", 1), ("StellarMetalMass", "f4", 1),
    ("BlackholeMass", "f4", 1), ("BlackholeAccretionRate", "f4", 1), ("MassHeIonized", "f4", 1),
    ("GasMetalElemMass", "f4", 9), ("StellarMetalElemMass", "f4", 9),
    ("MostMassiveBHID", "u8", 1),
    ("MSO", "f4", 5), ("RSO", "f4", 5),
    ("MTypeSO200c", "f4", 6), ("MTypeSOVir", "f4", 6),
    ("MboundPeak", "f4", 1), ("VmaxAtMboundPeak", "f4", 1), ("VmaxPeak", "f4", 1),
    ("MboundAcc", "f4", 1), ("VmaxAcc", "f4", 1), ("MboundFirstAcc", "f4", 1), ("VmaxFirstAcc", "f4", 1),
    ("SnapMboundPeak", "i4", 1), ("SnapVmaxPeak", "i4", 1), ("SnapAcc", "i4", 1),
    ("SnapFirstAcc", "i4", 1), ("SnapLastMajorMerger", "i4", 1),
    ("GeomParentTrackId", "i8", 1), ("GeomUpParentTrackId", "i8", 1),
]

_GROUP_E = [c for c in SUBHALO_COLUMNS if c[0] in (
    "MboundPeak", "VmaxAtMboundPeak", "VmaxPeak", "MboundAcc", "VmaxAcc", "MboundFirstAcc",
    "VmaxFirstAcc", "SnapMboundPeak", "SnapVmaxPeak", "SnapAcc", "SnapFirstAcc",
    "SnapLastMajorMerger", "GeomParentTrackId", "GeomUpParentTrackId")]

# Appendix B, PIG_xxx/Orphan/
ORPHAN_COLUMNS = [
    ("TrackId", "i8", 1), ("HostGroup", "i8", 1), ("NestedParentTrackId", "i8", 1),
    ("CentralTrackId", "i8", 1), ("SinkTrackId", "i8", 1), ("DescendantTrackId", "i8", 1),
    ("MostBoundID", "u8", 1),
    ("TracerPos", "f8", 3), ("TracerVel", "f4", 3),
    ("SnapBirth", "i4", 1), ("SnapDeath", "i4", 1), ("SnapSink", "i4", 1),
    ("SnapLastIsolation", "i4", 1), ("SnapTracerLost", "i4", 1),
] + _GROUP_E

# Appendix B marks these optional.
OPTIONAL_COLUMNS = ("GeomParentTrackId", "GeomUpParentTrackId")

PARTICLE_TAG_BLOCKS = [("TrackId", "i8"), ("BindRank", "i4"), ("BoundTrackId", "i8")]

HEADER_ATTRS = [
    "NumSubhalosTotal", "NumOrphansTotal", "NumTracksTotal", "NumPartSavedTotal",
    "SnapshotIndex", "SubhaloLastCallSnapshot", "SubhaloStateSnapshot", "SubhaloSchemaVersion",
    "SubhaloStateComplete", "SubhaloStateTime", "TimeSincePreviousFinderCall", "SubhaloParameters",
]

# Flags bits (design section 8, subhalo.h)
FLAG_ITERCAP = 1 << 0
FLAG_INPLACE = 1 << 1
FLAG_PLANCENTRAL = 1 << 2
FLAG_SOCUTOFF = 1 << 3
FLAG_SONOCROSS = 1 << 4

# GroupID of a particle outside any group: GrNr = -1 written as u4.
NO_GROUP_U4 = np.uint32(0xFFFFFFFF)


class SubhaloFormatError(RuntimeError):
    """The PIG does not hold (complete, well-formed) subhalo finder output."""


def _unwrap(value):
    """BigFile attributes come back as arrays; unwrap length-1 arrays to scalars."""
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0]
    return value


def header_attrs(bf, block="Header"):
    """Attributes of a block of an open bigfile.File as a plain dict, length-1 arrays unwrapped.
    (The bigfile AttrSet supports iteration and indexing but not .items() or .get().)"""
    attrs = bf[block].attrs
    return {k: _unwrap(v) for k, v in dict(attrs).items()}


class SubhaloCatalogue(object):
    """The subhalo finder output of one PIG_xxx directory.

    Attributes
    ----------
    pigdir : str
    header : dict of Header attributes (length-1 arrays unwrapped)
    subhalos : dict column -> numpy array, one row per resolved track, sorted by TrackId
    orphans : dict column -> numpy array, one row per orphan track, sorted by TrackId
    groups : dict with 'GroupID' and 'LengthByType' (int64 [Ngroups, 6]) of FOFGroups
    group_offsets : int64 [Ngroups + 1, 6], cumulative LengthByType (section-0 group starts)
    missing_columns : list of Appendix B columns absent from the file (optional ones only,
                      unless strict=False)
    """

    def __init__(self, pigdir, strict=True):
        self.pigdir = pigdir
        if not os.path.isdir(pigdir):
            raise SubhaloFormatError("%s: no such PIG directory" % pigdir)
        self.file = bigfile.File(pigdir)
        self.blocks = set(self.file.blocks)
        if "Subhalo/TrackId" not in self.blocks and "Orphan/TrackId" not in self.blocks:
            raise SubhaloFormatError(
                "%s has no subhalo finder blocks (Subhalo/*, Orphan/*): it was written with "
                "SubhaloFinderOn = 0, or by a build without the finder" % pigdir)
        self.header = header_attrs(self.file)
        self.missing_columns = []
        self.subhalos = self._read_table("Subhalo", SUBHALO_COLUMNS, strict)
        self.orphans = self._read_table("Orphan", ORPHAN_COLUMNS, strict)
        self._read_groups()
        self._cache = {}
        self._sub_index = None
        self._orphan_index = None

    # ------------------------------------------------------------------ loading
    def _read_table(self, prefix, columns, strict):
        out = {}
        nrows = None
        for name, dtype, items in columns:
            block = "%s/%s" % (prefix, name)
            if block not in self.blocks:
                self.missing_columns.append(block)
                if strict and name not in OPTIONAL_COLUMNS:
                    raise SubhaloFormatError("%s: missing column %s (Appendix B)" % (self.pigdir, block))
                continue
            arr = self.file[block][:]
            if nrows is None:
                nrows = arr.shape[0]
            elif arr.shape[0] != nrows:
                raise SubhaloFormatError("%s: column %s has %d rows, others %d"
                                         % (self.pigdir, block, arr.shape[0], nrows))
            if strict and arr.dtype != np.dtype(dtype):
                raise SubhaloFormatError("%s: column %s has dtype %s, Appendix B says %s"
                                         % (self.pigdir, block, arr.dtype, dtype))
            width = 1 if arr.ndim == 1 else arr.shape[1]
            if strict and width != items:
                raise SubhaloFormatError("%s: column %s has %d items per row, Appendix B says %d"
                                         % (self.pigdir, block, width, items))
            out[name] = arr
        # Extra columns not in the schema are read as well, so nothing is hidden.
        known = set(n for n, _, _ in columns)
        for block in sorted(self.blocks):
            if block.startswith(prefix + "/") and block.split("/", 1)[1] not in known:
                out[block.split("/", 1)[1]] = self.file[block][:]
        return out

    def _read_groups(self):
        if "FOFGroups/LengthByType" in self.blocks:
            lbt = self.file["FOFGroups/LengthByType"][:].astype(np.int64)
            gid = self.file["FOFGroups/GroupID"][:].astype(np.int64)
        else:
            lbt = np.zeros((0, NTYPES), dtype=np.int64)
            gid = np.zeros(0, dtype=np.int64)
        self.groups = {"GroupID": gid, "LengthByType": lbt}
        self.group_offsets = np.zeros((lbt.shape[0] + 1, NTYPES), dtype=np.int64)
        if lbt.shape[0] > 0:
            self.group_offsets[1:] = np.cumsum(lbt, axis=0)
        # GroupID -> row lookup (GroupID is dense in row order; do not assume its base)
        self._gid_sorter = np.argsort(gid, kind="stable")

    # ------------------------------------------------------------------ basic accessors
    @property
    def nsubhalos(self):
        return len(self.subhalos.get("TrackId", []))

    @property
    def norphans(self):
        return len(self.orphans.get("TrackId", []))

    @property
    def ngroups(self):
        return self.groups["GroupID"].shape[0]

    def complete(self):
        return int(self.header.get("SubhaloStateComplete", 0)) == 1

    def parameters(self):
        """The finder parameters of the Header attribute SubhaloParameters ("Name = value" lines,
        stored as a character array) as a dict of strings; empty if the attribute is absent."""
        raw = self.header.get("SubhaloParameters")
        if raw is None:
            return {}
        if isinstance(raw, (bytes, np.bytes_)):
            text = raw.decode("ascii", "replace")
        elif isinstance(raw, str):
            text = raw
        else:
            text = "".join(c.decode("ascii", "replace") if isinstance(c, (bytes, np.bytes_)) else str(c)
                           for c in np.atleast_1d(raw).tolist())
        out = {}
        for line in text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def tracer_types(self, default=(1, 4)):
        """Particle types of the SubhaloTracerTypes bitmask in SubhaloParameters (default types 1, 4)."""
        p = self.parameters()
        if "SubhaloTracerTypes" not in p:
            return list(default)
        mask = int(p["SubhaloTracerTypes"])
        return [t for t in range(NTYPES) if mask & (1 << t)]

    def int_parameter(self, name, default):
        """An integer finder parameter from SubhaloParameters, or default."""
        p = self.parameters()
        try:
            return int(p[name])
        except (KeyError, ValueError):
            return default

    def nsaved(self, ptype):
        """Rows in PIG/<ptype>/ (both sections)."""
        block = "%d/TrackId" % ptype
        if block in self.blocks:
            return self.file[block].size
        if "NumPartSavedTotal" in self.header:
            return int(np.atleast_1d(self.header["NumPartSavedTotal"])[ptype])
        return 0

    def ningroup(self, ptype):
        """Rows of section 0 in PIG/<ptype>/: the sum of LengthByType[:, ptype]."""
        return int(self.group_offsets[-1, ptype])

    def read(self, ptype, name):
        """Whole per-particle block <ptype>/<name> (cached); empty array if the block is absent."""
        key = (ptype, name)
        if key not in self._cache:
            block = "%d/%s" % (ptype, name)
            if block in self.blocks:
                self._cache[key] = self.file[block][:]
            else:
                dt = dict(PARTICLE_TAG_BLOCKS).get(name, "i8")
                self._cache[key] = np.zeros(0, dtype=dt)
        return self._cache[key]

    def has_particle_tags(self, ptype):
        return all("%d/%s" % (ptype, n) in self.blocks for n, _ in PARTICLE_TAG_BLOCKS)

    def group_index(self, groupid):
        """Row of FOFGroups for a GroupID (== a Subhalo HostGroup). -1 if there is none."""
        gid = self.groups["GroupID"]
        if gid.shape[0] == 0 or groupid < 0:
            return -1
        pos = np.searchsorted(gid, groupid, sorter=self._gid_sorter)
        if pos < gid.shape[0] and gid[self._gid_sorter[pos]] == groupid:
            return int(self._gid_sorter[pos])
        return -1

    def group_slice(self, gr, ptype):
        """Rows of FOF group row `gr` (0-based row of FOFGroups, not the GroupID) in <ptype>/,
        from the LengthByType cumulation (section 0)."""
        if gr < 0 or gr >= self.ngroups:
            raise IndexError("group row %d out of range [0, %d)" % (gr, self.ngroups))
        return slice(int(self.group_offsets[gr, ptype]), int(self.group_offsets[gr + 1, ptype]))

    def _index(self, which):
        if which == "subhalo":
            if self._sub_index is None:
                self._sub_index = self.subhalos.get("TrackId", np.zeros(0, dtype="i8"))
            return self._sub_index
        if self._orphan_index is None:
            self._orphan_index = self.orphans.get("TrackId", np.zeros(0, dtype="i8"))
        return self._orphan_index

    def subhalo_row(self, trackid):
        """Row of the resolved track in Subhalo/*, or -1 (orphan or unknown)."""
        tid = self._index("subhalo")
        pos = np.searchsorted(tid, trackid)
        if pos < tid.shape[0] and tid[pos] == trackid:
            return int(pos)
        return -1

    def orphan_row(self, trackid):
        tid = self._index("orphan")
        pos = np.searchsorted(tid, trackid)
        if pos < tid.shape[0] and tid[pos] == trackid:
            return int(pos)
        return -1

    def track(self, trackid):
        """The record of a track as a dict (Subhalo row, or Orphan row), with 'table' set."""
        row = self.subhalo_row(trackid)
        if row >= 0:
            rec = {k: v[row] for k, v in self.subhalos.items()}
            rec["table"] = "Subhalo"
            return rec
        row = self.orphan_row(trackid)
        if row >= 0:
            rec = {k: v[row] for k, v in self.orphans.items()}
            rec["table"] = "Orphan"
            return rec
        raise KeyError("TrackId %d is in neither Subhalo/* nor Orphan/* of %s" % (trackid, self.pigdir))

    def _require_subhalo(self, trackid):
        row = self.subhalo_row(trackid)
        if row < 0:
            raise KeyError("TrackId %d is not a resolved subhalo in %s (orphans have no slices)"
                           % (trackid, self.pigdir))
        return row

    # ------------------------------------------------------------------ slices (design 9)
    def track_slice(self, trackid, ptype):
        """In-group slice of the track in <ptype>/ (section 0)."""
        row = self._require_subhalo(trackid)
        off = int(self.subhalos["SliceOffsetByType"][row, ptype])
        n = int(self.subhalos["SliceLengthByType"][row, ptype])
        return slice(off, off + n) if n > 0 else slice(0, 0)

    def track_field_slice(self, trackid, ptype):
        """Field slice of the track in <ptype>/ (section 1)."""
        row = self._require_subhalo(trackid)
        off = int(self.subhalos["FieldSliceOffsetByType"][row, ptype])
        n = int(self.subhalos["FieldSliceLengthByType"][row, ptype])
        return slice(off, off + n) if n > 0 else slice(0, 0)

    def exclusive_bound(self, trackid, ptype):
        """Row indices (into <ptype>/) of the track's exclusive bound list: the bound prefix of
        its in-group slice followed by the bound prefix of its field slice."""
        row = self._require_subhalo(trackid)
        parts = []
        for prefix in ("Slice", "FieldSlice"):
            off = int(self.subhalos[prefix + "OffsetByType"][row, ptype])
            nb = int(self.subhalos[prefix + "NboundByType"][row, ptype])
            if nb > 0:
                parts.append(np.arange(off, off + nb, dtype=np.int64))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def key_ranges(self, trackid, ptype):
        """The rows of <ptype>/ that can hold particles bound to the track (design 9): the host
        FOF group's LengthByType range plus the field slices of every track with the same
        HostGroup (the key's field slices); for a hostless track, its own field slice."""
        row = self._require_subhalo(trackid)
        host = int(self.subhalos["HostGroup"][row])
        ranges = []
        if host >= 0:
            gr = self.group_index(host)
            if gr < 0:
                raise SubhaloFormatError("%s: TrackId %d has HostGroup %d, which is not a FOFGroups GroupID"
                                         % (self.pigdir, trackid, host))
            ranges.append(self.group_slice(gr, ptype))
            members = np.nonzero(self.subhalos["HostGroup"] == host)[0]
        else:
            members = np.array([row])
        for m in members:
            off = int(self.subhalos["FieldSliceOffsetByType"][m, ptype])
            n = int(self.subhalos["FieldSliceLengthByType"][m, ptype])
            if n > 0:
                ranges.append(slice(off, off + n))
        return ranges

    def full_bound(self, trackid, ptype):
        """Row indices (into <ptype>/) of the track's full bound list: every row of the key's
        range (key_ranges) whose BoundTrackId is the track."""
        bound = self.read(ptype, "BoundTrackId")
        parts = []
        for s in self.key_ranges(trackid, ptype):
            idx = np.arange(s.start, s.stop, dtype=np.int64)
            parts.append(idx[bound[s] == trackid])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ report
    def tag_counts(self):
        """Per type: saved rows, rows in groups, tagged rows (TrackId >= 0) in groups and in the
        field, and rows bound to their own track (BoundTrackId == TrackId >= 0)."""
        out = []
        for t in range(NTYPES):
            ns = self.nsaved(t)
            ng = self.ningroup(t)
            if not self.has_particle_tags(t) or ns == 0:
                out.append(dict(ptype=t, saved=ns, ingroup=ng, tagged_group=0, tagged_field=0, own_bound=0))
                continue
            tr = self.read(t, "TrackId")
            bt = self.read(t, "BoundTrackId")
            out.append(dict(ptype=t, saved=ns, ingroup=ng,
                            tagged_group=int(np.count_nonzero(tr[:ng] >= 0)),
                            tagged_field=int(np.count_nonzero(tr[ng:] >= 0)),
                            own_bound=int(np.count_nonzero((tr >= 0) & (bt == tr)))))
        return out

    def summary(self, out=sys.stdout):
        h = self.header
        print("%s" % self.pigdir, file=out)
        print("  SnapshotIndex %s  Time %s  complete %s  schema %s" % (
            h.get("SnapshotIndex"), h.get("Time"), h.get("SubhaloStateComplete"),
            h.get("SubhaloSchemaVersion")), file=out)
        print("  SubhaloStateSnapshot %s  SubhaloLastCallSnapshot %s  TimeSincePreviousFinderCall %s" % (
            h.get("SubhaloStateSnapshot"), h.get("SubhaloLastCallSnapshot"),
            h.get("TimeSincePreviousFinderCall")), file=out)
        print("  FOF groups %d  subhalos %d  orphans %d  NumTracksTotal %s" % (
            self.ngroups, self.nsubhalos, self.norphans, h.get("NumTracksTotal")), file=out)
        if self.nsubhalos:
            host = self.subhalos["HostGroup"]
            nb = self.subhalos["Nbound"]
            fl = self.subhalos["Flags"]
            print("  hosted %d  hostless %d  centrals %d  in place %d  Nbound min/median/max %d/%d/%d" % (
                np.count_nonzero(host >= 0), np.count_nonzero(host < 0),
                np.count_nonzero(self.subhalos["Rank"] == 0),
                np.count_nonzero(fl & FLAG_INPLACE), nb.min(), int(np.median(nb)), nb.max()), file=out)
        if self.norphans:
            print("  orphans sunk %d  dead (not sunk) %d" % (
                np.count_nonzero(self.orphans["SinkTrackId"] >= 0),
                np.count_nonzero(self.orphans["SinkTrackId"] < 0)), file=out)
        print("  type   saved  in-group  tagged(group)  tagged(field)  own-bound", file=out)
        for c in self.tag_counts():
            if c["saved"] == 0:
                continue
            print("  %4d %7d %9d %14d %14d %10d" % (
                c["ptype"], c["saved"], c["ingroup"], c["tagged_group"], c["tagged_field"], c["own_bound"]), file=out)
        if self.missing_columns:
            print("  missing columns: %s" % ", ".join(self.missing_columns), file=out)

    def describe_track(self, trackid, out=sys.stdout):
        rec = self.track(trackid)
        print("TrackId %d (%s)" % (trackid, rec["table"]), file=out)
        for k in sorted(rec):
            if k == "table":
                continue
            v = rec[k]
            if isinstance(v, np.ndarray):
                v = " ".join(str(x) for x in v)
            print("  %-26s %s" % (k, v), file=out)
        if rec["table"] != "Subhalo":
            return
        print("  type  slice           bound  field-slice     bound  exclusive  full", file=out)
        for t in range(NTYPES):
            s = self.track_slice(trackid, t)
            f = self.track_field_slice(trackid, t)
            if s.stop == s.start and f.stop == f.start:
                continue
            row = self.subhalo_row(trackid)
            full = len(self.full_bound(trackid, t)) if self.has_particle_tags(t) else -1
            print("  %4d  [%d,%d) %6d  [%d,%d) %6d %10d %5d" % (
                t, s.start, s.stop, self.subhalos["SliceNboundByType"][row, t],
                f.start, f.stop, self.subhalos["FieldSliceNboundByType"][row, t],
                len(self.exclusive_bound(trackid, t)), full), file=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read the subhalo finder catalogue of a PIG_xxx directory.")
    parser.add_argument("pigdir", help="PIG_xxx directory")
    parser.add_argument("--track", type=int, default=None, help="print the record and slices of this TrackId")
    parser.add_argument("--lenient", action="store_true", help="do not fail on missing or mistyped columns")
    args = parser.parse_args(argv)
    try:
        cat = SubhaloCatalogue(args.pigdir, strict=not args.lenient)
    except SubhaloFormatError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    cat.summary()
    if args.track is not None:
        try:
            cat.describe_track(args.track)
        except KeyError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
