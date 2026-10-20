#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reader for the runtime subhalo finder's catalogues in MP-Gadget PIG_xxx directories.

Schema: halofinder_design.md section 9 and Appendix B (SubhaloSchemaVersion 2; version 1 lacks the
Orphan followed-set columns, which are then filled with the legacy values, see ORPHAN_SET_COLUMNS).

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
                         TimeSincePreviousFinderCall, SubhaloParameters,
                         SubhaloWallTime (s, max over ranks), SubhaloPeakBytes (main-allocator
                         used high water, max over ranks), SubhaloPeakIncrementBytes (the finder's
                         own increment: high water without the field SO walks' elastic export
                         tables minus the usage at entry, max over ranks and stages; Phase 5d)
                         of the last finder call, ...

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
    ("RsSO", "f4", 1), ("RsKlypinSO", "f4", 1), ("VmaxSO", "f4", 1), ("RmaxSO", "f4", 1),
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

# Schema 2 (decision 70): the particles an orphan follows. OrphanIDs (MostBoundID first, then the
# first step-0 tracers, UINT64_MAX padded), their count, the markers kept at the last lookup, the
# rms spread of the kept set about TracerPos and the clip scale fixed at death (-1: none).
MAXTRACERS = 16
ORPHAN_SET_COLUMNS = [
    ("OrphanIDs", "u8", MAXTRACERS), ("NOrphanTracers", "i4", 1), ("NOrphanFound", "i4", 1),
    ("OrphanSigmaX", "f4", 1), ("OrphanScale", "f4", 1),
]
ORPHAN_COLUMNS = ORPHAN_COLUMNS + ORPHAN_SET_COLUMNS
NO_ID = np.uint64(0xFFFFFFFFFFFFFFFF)

# Appendix B marks these optional; the decision-44 SO profile columns are absent from catalogues
# written before them (read leniently: the props rules then skip them).
OPTIONAL_COLUMNS = ("GeomParentTrackId", "GeomUpParentTrackId", "RsSO", "RsKlypinSO", "VmaxSO", "RmaxSO") \
    + tuple(c[0] for c in ORPHAN_SET_COLUMNS)

PARTICLE_TAG_BLOCKS = [("TrackId", "i8"), ("BindRank", "i4"), ("BoundTrackId", "i8")]

HEADER_ATTRS = [
    "NumSubhalosTotal", "NumOrphansTotal", "NumTracksTotal", "NumPartSavedTotal",
    "SnapshotIndex", "SubhaloLastCallSnapshot", "SubhaloStateSnapshot", "SubhaloSchemaVersion",
    "SubhaloStateComplete", "SubhaloStateTime", "TimeSincePreviousFinderCall", "SubhaloParameters",
    "SubhaloWallTime", "SubhaloPeakBytes", "SubhaloPeakIncrementBytes",
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
        self._fill_legacy_orphan_set()
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

    def _fill_legacy_orphan_set(self):
        """A catalogue without the followed-set columns (schema 1): every orphan follows its
        MostBoundID alone, unclipped, as the finder's restart reads such a state (subhalo_io.c)."""
        O = self.orphans
        if "MostBoundID" not in O:
            return
        mb = O["MostBoundID"].astype(np.uint64)
        n = mb.shape[0]
        has = mb != NO_ID
        if "OrphanIDs" not in O:
            ids = np.full((n, MAXTRACERS), NO_ID, dtype=np.uint64)
            ids[:, 0] = mb
            O["OrphanIDs"] = ids
        if "NOrphanTracers" not in O:
            O["NOrphanTracers"] = has.astype(np.int32)
        if "NOrphanFound" not in O:
            lost = O["SnapTracerLost"] >= 0 if "SnapTracerLost" in O else np.zeros(n, bool)
            O["NOrphanFound"] = (has & ~lost).astype(np.int32)
        for c in ("OrphanSigmaX", "OrphanScale"):
            if c not in O:
                O[c] = np.full(n, -1, dtype=np.float32)

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

    def clear_cache(self):
        """Drop the cached per-particle blocks; the tables stay (for callers keeping many catalogues)."""
        self._cache.clear()

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

    # ------------------------------------------------------------------ bound ID lists (Phase 4)
    @property
    def snapshot(self):
        """SnapshotIndex of the Header (the PIG number)."""
        return int(np.atleast_1d(self.header.get("SnapshotIndex", -1))[0])

    @property
    def scale_factor(self):
        return float(np.atleast_1d(self.header.get("Time", np.nan))[0])

    def tagged_types(self):
        """Particle types with saved rows and the three tag blocks."""
        return [t for t in range(NTYPES) if self.nsaved(t) > 0 and self.has_particle_tags(t)]

    def full_bound_ids(self, trackid, return_types=False):
        """Particle IDs of the track's full bound list over every type, in binding order:

        1. the exclusive bound list (rows tagged with the track and bound to it: the bound prefixes
           of its in-group and field slices, every type) ordered by BindRank, which is the exact
           energy order of the final unbinding pass (BindRank is dense over the track's own-tag
           rows, bound first);
        2. then the overlap particles: rows of the key's range (key_ranges) bound to the track but
           tagged with a descendant in the nest tree (a child's tail). Their position in the
           track's own energy order is not recorded; they follow in (BindRank in their tag's list,
           ID) order.

        The first entries (the most-bound tracers, reserved for the track by the D10 tag rule) are
        therefore exact. Returns an int64/uint64 ID array, and with return_types the type of each.
        """
        rank, ids, types, overlap = [], [], [], []
        for t in self.tagged_types():
            rows = self.full_bound(trackid, t)
            if rows.size == 0:
                continue
            rank.append(self.read(t, "BindRank")[rows].astype(np.int64))
            ids.append(self.read(t, "ID")[rows])
            types.append(np.full(rows.size, t, np.int8))
            overlap.append(self.read(t, "TrackId")[rows] != trackid)
        if not ids:
            e = np.zeros(0, np.uint64)
            return (e, np.zeros(0, np.int8)) if return_types else e
        rank, ids, types, overlap = (np.concatenate(v) for v in (rank, ids, types, overlap))
        o = np.lexsort((ids, rank, overlap))
        return (ids[o], types[o]) if return_types else ids[o]

    def all_full_bound_ids(self):
        """full_bound_ids for every resolved track at once (vectorised; same order as
        full_bound_ids). Selects rows by BoundTrackId over the whole file rather than over each
        key's range; the two agree when the `tags` check passes (every row bound to T lies in T's
        key). Returns (trackids, offsets, ids, types): the list of trackids[k] is
        ids[offsets[k]:offsets[k + 1]]; trackids is Subhalo/TrackId (sorted)."""
        B, R, I, T, OV = [], [], [], [], []
        for t in self.tagged_types():
            bt = self.read(t, "BoundTrackId").astype(np.int64)
            m = bt >= 0
            B.append(bt[m])
            R.append(self.read(t, "BindRank")[m].astype(np.int64))
            I.append(self.read(t, "ID")[m])
            T.append(np.full(np.count_nonzero(m), t, np.int8))
            OV.append(self.read(t, "TrackId")[m].astype(np.int64) != bt[m])
        tids = self.subhalos.get("TrackId", np.zeros(0, np.int64)).astype(np.int64)
        if not B:
            return tids, np.zeros(tids.size + 1, np.int64), np.zeros(0, np.uint64), np.zeros(0, np.int8)
        B, R, I, T, OV = (np.concatenate(v) for v in (B, R, I, T, OV))
        o = np.lexsort((I, R, OV, B))
        B, I, T = B[o], I[o], T[o]
        keep = np.isin(B, tids)
        B, I, T = B[keep], I[keep], T[keep]
        counts = np.bincount(np.searchsorted(tids, B), minlength=tids.size) if tids.size else np.zeros(0, np.int64)
        offsets = np.zeros(tids.size + 1, np.int64)
        offsets[1:] = np.cumsum(counts)
        return tids, offsets, I, T

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
        if h.get("SubhaloWallTime") is not None:
            print("  SubhaloWallTime %s s  SubhaloPeakBytes %s  SubhaloPeakIncrementBytes %s" % (h.get("SubhaloWallTime"),
                  h.get("SubhaloPeakBytes"), h.get("SubhaloPeakIncrementBytes")), file=out)
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


def _fill_value(name, dtype):
    """Value of a Subhalo-only column on an orphan node: 0 for counts and bound masses (an orphan
    binds nothing), -1 otherwise (the finder's 'not available')."""
    if name.startswith("Nbound") or name in ("Mbound", "MboundType", "NtailType"):
        return 0
    if np.dtype(dtype).kind == "u":
        return np.iinfo(np.dtype(dtype)).max
    return -1


# Orphan columns standing in for Subhalo columns on orphan nodes: the tracer is the most-bound
# particle of the track's last bound set.
ORPHAN_AS_SUBHALO = {"MostBoundPos": "TracerPos", "MostBoundVel": "TracerVel"}


class SubhaloSequence(object):
    """The PIG_xxx outputs of one run in SnapshotIndex order: the tree of tracks is implicit in
    the persistent TrackIds (design 7.7), a node being (TrackId, output).

        seq = SubhaloSequence("output")              # every PIG_* holding finder blocks
        for cat in seq: ...                          # SubhaloCatalogue per output (cached)
        br = seq.branch(17)                          # the rows of track 17 over time
        snaps, m = seq.history(17, "Mbound")

    Catalogues are opened lazily and kept. `outputdir` is a directory of PIG_* or a list of
    PIG directories; PIGs without finder blocks are skipped with a note in `skipped`.
    """

    def __init__(self, outputdir, strict=True, pattern="PIG_*"):
        import glob
        if isinstance(outputdir, (list, tuple)):
            dirs = list(outputdir)
        else:
            dirs = sorted(glob.glob(os.path.join(outputdir, pattern)))
        self.strict = strict
        self.skipped = []
        entries = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            try:
                f = bigfile.File(d)
                blocks = set(f.blocks)
                if "Subhalo/TrackId" not in blocks and "Orphan/TrackId" not in blocks:
                    self.skipped.append(d)
                    continue
                h = header_attrs(f)
            except Exception:  # unreadable directory: skip, reported in `skipped`
                self.skipped.append(d)
                continue
            snap = int(np.atleast_1d(h.get("SnapshotIndex", -1))[0])
            entries.append((snap, float(np.atleast_1d(h.get("Time", np.nan))[0]), d))
        entries.sort()
        if len(set(e[0] for e in entries)) != len(entries):
            raise SubhaloFormatError("two PIGs with the same SnapshotIndex in %s" % (outputdir,))
        self.snapshots = [e[0] for e in entries]
        self.scale_factors = np.array([e[1] for e in entries], dtype=np.float64)
        self.paths = [e[2] for e in entries]
        self._cats = {}
        self._nodes = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """Catalogue of the i-th output of the sequence (not the SnapshotIndex)."""
        if i not in self._cats:
            self._cats[i] = SubhaloCatalogue(self.paths[i], strict=self.strict)
        return self._cats[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index_of(self, snapshot):
        """Position in the sequence of a SnapshotIndex, -1 if absent."""
        try:
            return self.snapshots.index(int(snapshot))
        except ValueError:
            return -1

    def catalogue(self, snapshot):
        i = self.index_of(snapshot)
        if i < 0:
            raise KeyError("SnapshotIndex %d is not in the sequence" % snapshot)
        return self[i]

    def scale_of(self, snapshot, default=-1.0):
        """Scale factor of SnapshotIndex values (array-like; default where absent or < 0)."""
        snap = np.atleast_1d(np.asarray(snapshot, dtype=np.int64))
        out = np.full(snap.shape, default, dtype=np.float64)
        if not self.snapshots:
            return out
        s = np.array(self.snapshots, dtype=np.int64)
        pos = np.clip(np.searchsorted(s, snap), 0, s.size - 1)
        hit = s[pos] == snap
        out[hit] = self.scale_factors[pos[hit]]
        return out

    # ------------------------------------------------------------------ the node table
    def node_columns(self):
        """Column names of the node table: the union of Subhalo/* and Orphan/* columns over the
        sequence, plus TrackId, Output, Snapshot, ScaleFactor, Resolved."""
        sub, orph = [], []
        for cat in self:
            for k in cat.subhalos:
                if k not in sub:
                    sub.append(k)
            for k in cat.orphans:
                if k not in orph:
                    orph.append(k)
        return sub + [k for k in orph if k not in sub]

    def nodes(self):
        """Every (track, output) node of the sequence as a dict of arrays sorted by
        (TrackId, Output): Output (position in the sequence), Snapshot, ScaleFactor, Resolved
        (True for a Subhalo/* row, False for an Orphan/* row), and every column of node_columns().
        On orphan nodes Subhalo-only columns are 0 for counts and bound masses and -1 (or the u8
        maximum) otherwise, except MostBoundPos/MostBoundVel which take TracerPos/TracerVel; on
        resolved nodes Orphan-only columns are -1 except TracerPos/TracerVel, which take
        MostBoundPos/MostBoundVel. Cached."""
        if self._nodes is not None:
            return self._nodes
        cols = self.node_columns()
        parts = {k: [] for k in cols}
        meta = {"Output": [], "Snapshot": [], "ScaleFactor": [], "Resolved": []}
        proto = {}
        for i, cat in enumerate(self):
            for tab in (cat.subhalos, cat.orphans):
                for k, v in tab.items():
                    proto.setdefault(k, (v.dtype, v.shape[1:]))
        inverse = {v: k for k, v in ORPHAN_AS_SUBHALO.items()}
        for i, cat in enumerate(self):
            for tab, resolved in ((cat.subhalos, True), (cat.orphans, False)):
                n = len(tab.get("TrackId", []))
                if n == 0:
                    continue
                for k in cols:
                    dt, shp = proto[k]
                    if k in tab:
                        v = tab[k]
                    elif not resolved and k in ORPHAN_AS_SUBHALO and ORPHAN_AS_SUBHALO[k] in tab:
                        v = tab[ORPHAN_AS_SUBHALO[k]].astype(dt)
                    elif resolved and k in inverse and inverse[k] in tab:
                        v = tab[inverse[k]].astype(dt)
                    else:
                        v = np.full((n,) + shp, _fill_value(k, dt), dtype=dt)
                    parts[k].append(v)
                meta["Output"].append(np.full(n, i, np.int32))
                meta["Snapshot"].append(np.full(n, self.snapshots[i], np.int32))
                meta["ScaleFactor"].append(np.full(n, self.scale_factors[i], np.float64))
                meta["Resolved"].append(np.full(n, resolved, bool))
        out = {}
        for k, v in list(meta.items()) + list(parts.items()):
            if v:
                out[k] = np.concatenate(v)
            else:
                dt, shp = proto.get(k, (np.dtype("i8"), ()))
                out[k] = np.zeros((0,) + tuple(shp), dtype=dt if k not in meta else np.int64)
        if "TrackId" in out and out["TrackId"].size:
            o = np.lexsort((out["Output"], out["TrackId"]))
            out = {k: v[o] for k, v in out.items()}
        self._nodes = out
        return out

    def track_ids(self):
        """Every TrackId that appears in the sequence (resolved or orphan), sorted."""
        n = self.nodes()
        return np.unique(n["TrackId"]) if "TrackId" in n else np.zeros(0, np.int64)

    def branch(self, trackid):
        """The rows of a track over time: a dict of arrays (one entry per output at which the
        track is in Subhalo/* or Orphan/*, in output order) with the node_columns() plus Output,
        Snapshot, ScaleFactor and Resolved. Empty arrays if the track never appears."""
        n = self.nodes()
        tid = n.get("TrackId", np.zeros(0, np.int64))
        lo, hi = np.searchsorted(tid, trackid, "left"), np.searchsorted(tid, trackid, "right")
        return {k: v[lo:hi] for k, v in n.items()}

    def history(self, trackid, column, resolved_only=False):
        """(SnapshotIndex values, column values) of a track over the sequence."""
        b = self.branch(trackid)
        m = b["Resolved"] if resolved_only else np.ones(b["Snapshot"].size, bool)
        return b["Snapshot"][m], b[column][m]

    def resolved_snapshots(self, trackid):
        return self.history(trackid, "TrackId", resolved_only=True)[0]

    def first_resolved(self, trackid):
        """SnapshotIndex of the first output at which the track is resolved, -1 if never."""
        s = self.resolved_snapshots(trackid)
        return int(s[0]) if s.size else -1

    def last_resolved(self, trackid):
        s = self.resolved_snapshots(trackid)
        return int(s[-1]) if s.size else -1

    def merger_descendant(self, trackid):
        """The merger relation of design 7.7 for a track that stops being resolved: (SnapshotIndex
        of the first output at which it is an orphan after its last resolved output,
        DescendantTrackId read from that output's row). (-1, -1) for a track resolved at the last
        output or never resolved."""
        b = self.branch(trackid)
        res = np.nonzero(b["Resolved"])[0]
        if res.size == 0 or res[-1] == b["Resolved"].size - 1:
            return -1, -1
        k = res[-1] + 1
        return int(b["Snapshot"][k]), int(b["DescendantTrackId"][k])


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
