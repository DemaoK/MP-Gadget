#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the runtime subhalo finder's catalogue in a PIG_xxx with HBT-BF's SubSnap_xxx
(halofinder_phase2a_plan.md section 6, halofinder_design.md section 13 and D3).

Usage:
  subhalo_compare_hbt.py PIG_xxx HBT [options]
      one snapshot. HBT is HBT's SubhaloPath (per-snapshot directories NNN/), one NNN/ directory,
      or one SubSnap_NNN.*.hdf5 file (its sibling files are read too); NNN is the PIG's
      SnapshotIndex plus --hbt-snap-offset.
  subhalo_compare_hbt.py --track-pairs PIG_a PIG_b [PIG_c ...] HBT [options]
      every snapshot of the sequence (in order), then across each consecutive pair of outputs:
      temporal identity (a pair (our TrackId, HBT TrackId) matched at one output should be
      matched again at the next when both tracks are still resolved; the fraction and every
      break are listed), deaths (tracks that died at the later output on both sides, their
      SinkTrackId and DescendantTrackId mapped through the pair map of the earlier output) and
      orphan hosts (our orphan HostGroup against HBT's HostHaloId of the Nbound == 0 row, the
      tracks paired by the TrackId pairs of the earlier outputs). HBT may also be given as the
      first positional argument.

Matching (--match):
  tracers (default)  each side's ten most-bound tracers vote, with HBT's weights 1/(1 + sqrt(i)),
                     for the other side's subhalo that binds them; a pair is matched when the two
                     votes elect each other. Ours: the first --ntracers own-tag rows of a tracer type
                     (SubhaloTracerTypes) in BindRank order, i.e. the start of the in-group slice
                     and field slice. HBT's: the first --ntracers particles of a tracer type in its
                     SubhaloParticles list (types from our PIG's ID blocks, or --part; IDs of unknown
                     type are taken as tracers, which is exact for DM-only runs).
  jaccard            each subhalo's partner is the one with the largest Jaccard index of the full
                     bound sets; a pair is matched when the choice is mutual.

Per matched pair: the Jaccard index of the full bound sets (ours: every row with BoundTrackId ==
TrackId, i.e. SubhaloCatalogue.full_bound; HBT's: SubhaloParticles), Nbound(ours)/Nbound(HBT), and
Vmax(ours)/VmaxPhysical(HBT) where our Vmax is set (> 0). Summary: matched fractions overall and
above --min-np HBT particles, Jaccard and ratio quantiles, unmatched counts both ways, and the
Phase 2a acceptance figure: the fraction of HBT subhalos above --min-np that are matched with
Jaccard >= --jaccard-min and |Nbound ratio - 1| <= --nbound-tol (target --accept-frac). Pairs and
HBT subhalos that miss it are classified by the D3 deviations: subsampled (either Nbound above
SubhaloPotentialSampleSize), in place (Flags), iteration cap (Flags), few tracers (fewer than
SubhaloCoreSize bound tracers on either side, the sinking core fill-up), spawned stars (type-4
particles in our set absent from HBT's), else unexplained.

Failure classes: every matched pair that misses the criterion, and every unmatched HBT subhalo, is
labelled by cause (counts after the acceptance line for >= --min-np and for the near-threshold
population, the class and its numbers under --show, a fail_class column in --pairs-out). Groups are
compared directly: HBT's HostHaloId of an MP-Gadget PIG is the FOF GroupID, as our HostGroup.
  swap              the pair is in the same FOF group on both sides, Rank 0 on one side and Rank >= 1
                    on the other; or both Rank 0 but different tracks (the HBT central's twin is
                    another track of ours; twins are the pairs of the --prev output, or with
                    --track-pairs each track's earliest pairing while that partner is resolved).
  in-swapped-group  same group and role on both sides, in a group with a swap at this output.
  nbound            same group and role, no swap seen there: Nbound outside --nbound-tol and/or
                    Jaccard below --jaccard-min (the ratio and Jaccard are printed).
  survival          the HBT subhalo is unmatched at this output (our twin may be dead, orphan,
                    absent, unmatched or paired elsewhere; it need not have died).
  other             anything else, with the reason (e.g. different groups on the two sides).
  A both-Rank-0 swap is visible only through twins from before it began: with --prev alone, or one
  that began at or before the first output of a sequence, it counts as nbound (the report says so).
  Tie check: once per swap group, at the output where the swap first appears, which must directly
  follow the previous supplied output (else not checked), on our previous saved Mbound of our
  central and of the historical twin of HBT's central against the group's largest. tie-0.8 when
  either (i) "exact 4:5": |5 M_i - 4 M_0| within the float32 rounding of two exact 4:5 masses, or
  (ii) "float32 cut differs": HBT's float32 M_i < (float)(M_0 * 0.8f) and ours, M_i < 0.8 M_0 in
  double, disagree; both count, --show names which. HBT re-sums its own masses in float32, so (ii)
  on our saved values is exact only for identical bound sets; (i) covers equal-mass DM ties.
  With --track-pairs the label persists with the swap; without a previous output the report says
  the check was not done. A second acceptance figure leaves out the swaps labelled tie-0.8.
  --prev must be an earlier output; with an HBT NNN/ directory or SubSnap file argument its sibling
  snapshot is used, and every SubSnap file's SnapshotId must be the requested snapshot.

Links and history, at every snapshot for the matched pairs: NestedParentTrackId (mapped through
the pair map of that snapshot), HostGroup against HostHaloId (both FOF GroupIDs), Rank and Depth;
MboundPeak / LastMaxMass and VmaxPeak / LastMaxVmaxPhysical (quantiles and the fractions within
1% and 10%), SnapLastIsolation, SnapBirth, SnapDeath and SnapSink against HBT's SnapshotOf*
(minus --hbt-snap-offset); the same history for the orphan pairs of a sequence. Every block is
reported for all pairs, for HBT Nbound >= --min-np and for the near-threshold population
--near-np <= HBT Nbound < --min-np (default 20-99) separately; --list caps the listed mismatches.

Exit status: 0 when the acceptance figure (and, with --track-pairs, the persistence fraction
against --persist-frac) is met or --report-only is given; 1 when it is not; 2 when an input could
not be read.

Phase 1 catalogues (every FOF group one subhalo, field particles keeping stale tags): pass
--phase1 so that only section 0 (the FOF groups) is read; matching by tracer IDs still works.

Examples:
  python3 tools/subhalo_compare_hbt.py output/PIG_004 hbt/ --phase1 --report-only
  python3 tools/subhalo_compare_hbt.py output/PIG_004 hbt/ --match jaccard --pairs-out pairs_004.txt
  python3 tools/subhalo_compare_hbt.py --track-pairs output/PIG_003 output/PIG_004 hbt/
  python3 tools/subhalo_compare_hbt.py hbt/ --track-pairs output/PIG_00?
  python3 tools/subhalo_compare_hbt.py output/PIG_018 hbt/ --prev output/PIG_017 --show 20
"""

from __future__ import print_function

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloCatalogue, SubhaloFormatError, NTYPES, FLAG_INPLACE, FLAG_ITERCAP  # noqa: E402

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class InputError(RuntimeError):
    pass


def _lookup(sorted_keys, values, query, default=-1):
    """values[position of query in sorted_keys], default where absent."""
    query = np.asarray(query)
    out = np.full(query.shape, default, dtype=np.asarray(values).dtype if len(values) else np.int64)
    if sorted_keys.size == 0 or query.size == 0:
        return out
    pos = np.searchsorted(sorted_keys, query)
    posc = np.minimum(pos, sorted_keys.size - 1)
    hit = sorted_keys[posc] == query
    out[hit] = values[posc[hit]]
    return out


def _weighted_vote(voter, cand, weight):
    """For each distinct voter the candidate with the largest summed weight, ties to the smaller
    candidate (-1, 'nobody', first). Returns (voters, winners)."""
    if voter.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs, inv = np.unique(np.stack([voter, cand], axis=1), axis=0, return_inverse=True)
    w = np.bincount(inv.ravel(), weights=weight, minlength=pairs.shape[0])
    o = np.lexsort((pairs[:, 1], -w, pairs[:, 0]))
    pairs = pairs[o]
    first = np.ones(pairs.shape[0], bool)
    first[1:] = pairs[1:, 0] != pairs[:-1, 0]
    return pairs[first, 0], pairs[first, 1]


def _first_n_per_group(group, n):
    """Mask of the first n entries of each run of equal values in an already grouped array."""
    if group.size == 0:
        return np.zeros(0, bool)
    start = np.ones(group.size, bool)
    start[1:] = group[1:] != group[:-1]
    idx = np.arange(group.size)
    first = np.maximum.accumulate(np.where(start, idx, 0))
    return (idx - first) < n


# ---------------------------------------------------------------------------------- our side
class Ours(object):
    """Our catalogue reduced to what the comparison needs."""

    def __init__(self, pigdir, ntracers, phase1=False):
        self.cat = cat = SubhaloCatalogue(pigdir)
        self.pigdir = pigdir
        self.snap = int(np.atleast_1d(cat.header.get("SnapshotIndex", -1))[0])
        self.params = cat.parameters()
        self.tracer_types = cat.tracer_types()
        S = cat.subhalos
        self.S = S
        self.tid = S["TrackId"].astype(np.int64)
        ID, T, R, B, typ = [], [], [], [], []
        for t in range(NTYPES):
            ns = cat.nsaved(t)
            if ns == 0 or not cat.has_particle_tags(t):
                continue
            end = cat.ningroup(t) if phase1 else ns
            ID.append(cat.read(t, "ID")[:end].astype(np.int64))
            T.append(cat.read(t, "TrackId")[:end].astype(np.int64))
            R.append(cat.read(t, "BindRank")[:end].astype(np.int64))
            B.append(cat.read(t, "BoundTrackId")[:end].astype(np.int64))
            typ.append(np.full(end, t, np.int32))
        cc = lambda v, dt: np.concatenate(v) if v else np.zeros(0, dt)  # noqa: E731
        ID, T, R, B, typ = cc(ID, np.int64), cc(T, np.int64), cc(R, np.int64), cc(B, np.int64), cc(typ, np.int32)
        # ID -> type over every saved row (used to type HBT's particles)
        o = np.argsort(ID, kind="stable")
        self.id_sorted, self.id_type = ID[o], typ[o]
        # full bound sets: every row with BoundTrackId == T (resolved tracks only)
        bown = (B >= 0) & np.isin(B, self.tid)
        o = np.argsort(ID[bown], kind="stable")
        self.bid_sorted, self.bid_owner, self.bid_type = ID[bown][o], B[bown][o], typ[bown][o]
        self.nfull = np.bincount(np.searchsorted(self.tid, self.bid_owner), minlength=self.tid.size) \
            if self.tid.size else np.zeros(0, np.int64)
        # tracers: own-tag rows of a tracer type in BindRank order
        sel = (T >= 0) & np.isin(typ, self.tracer_types) & np.isin(T, self.tid)
        o = np.lexsort((ID[sel], R[sel], T[sel]))
        tt, ti = T[sel][o], ID[sel][o]
        keep = _first_n_per_group(tt, ntracers)
        self.tr_track, self.tr_id = tt[keep], ti[keep]
        self.tr_pos = np.zeros(self.tr_track.size, np.int64)
        if self.tr_track.size:
            start = np.ones(self.tr_track.size, bool)
            start[1:] = self.tr_track[1:] != self.tr_track[:-1]
            idx = np.arange(self.tr_track.size)
            self.tr_pos = idx - np.maximum.accumulate(np.where(start, idx, 0))
        self.nbound = S["Nbound"].astype(np.int64)
        self.vmax = S["Vmax"].astype(np.float64)
        self.flags = S["Flags"].astype(np.int64)
        self.ntracer = S["NboundTracer"].astype(np.int64)
        self.rank = S["Rank"].astype(np.int64)
        mism = np.count_nonzero(self.nfull != self.nbound)
        self.nfull_mismatch = mism
        self.O = cat.orphans
        self.otid = self.O["TrackId"].astype(np.int64) if cat.norphans else np.zeros(0, np.int64)
        self._orows, self._rows = np.arange(self.otid.size), np.arange(self.tid.size)
        cat.clear_cache()   # the raw particle blocks are folded into the arrays above

    def orow(self, trackids):
        return _lookup(self.otid, self._orows, trackids)

    def row(self, trackids):
        return _lookup(self.tid, self._rows, trackids)

    def owner(self, ids):
        return _lookup(self.bid_sorted, self.bid_owner, ids)

    def types(self, ids):
        return _lookup(self.id_sorted, self.id_type.astype(np.int64), ids)

    def drop_particles(self):
        """Free the particle-level arrays; row(), orow() and the tables keep working."""
        self.id_sorted = self.id_type = self.bid_sorted = self.bid_owner = self.bid_type = None
        self.tr_track = self.tr_id = self.tr_pos = None


# ---------------------------------------------------------------------------------- HBT side
def hbt_files(path, snap):
    """SubSnap files of snapshot `snap` under an HBT SubhaloPath, a NNN/ directory, or a file; a
    directory or file of another snapshot leads to its sibling NNN/ (checked in HBT())."""
    if os.path.isfile(path):
        m = re.match(r"(.*SubSnap_(\d+))\.\d+\.hdf5$", path) or re.match(r"(.*SubSnap_(\d+))\.hdf5$", path)
        if not m:
            return [path]
        if int(m.group(2)) == snap:
            return sorted(glob.glob(m.group(1) + ".*.hdf5"), key=_fileno) or [path]
        path = os.path.dirname(os.path.abspath(path))
    up = os.path.dirname(os.path.abspath(path))
    cands = [path, os.path.join(path, "%03d" % snap), os.path.join(path, "%04d" % snap),
             os.path.join(up, "%03d" % snap), os.path.join(up, "%04d" % snap)]
    for d in cands:
        files = glob.glob(os.path.join(d, "SubSnap_%03d.*.hdf5" % snap)) + glob.glob(os.path.join(d, "SubSnap_%03d.hdf5" % snap))
        if files:
            return sorted(files, key=_fileno)
    raise InputError("no SubSnap_%03d files under %s" % (snap, path))


def _filesnap(f):
    m = re.search(r"SubSnap_(\d+)(\.\d+)?\.hdf5$", f)
    return int(m.group(1)) if m else None


def _fileno(f):
    m = re.search(r"\.(\d+)\.hdf5$", f)
    return int(m.group(1)) if m else -1


class HBT(object):
    """HBT's subhalos of one snapshot: the Subhalos table and the bound ID lists."""

    def __init__(self, path, snap):
        if h5py is None:
            raise InputError("h5py is not available")
        self.files = hbt_files(path, snap)
        subs, parts = [], []
        for fn in self.files:
            with h5py.File(fn, "r") as f:
                if "Subhalos" not in f:
                    raise InputError("%s has no Subhalos dataset" % fn)
                # the file's snapshot must be the one asked for (SnapshotId, else the name)
                sid = int(np.atleast_1d(f["SnapshotId"][()])[0]) if "SnapshotId" in f else _filesnap(fn)
                if sid is not None and sid != snap:
                    raise InputError("%s is snapshot %d, not the requested %d" % (fn, sid, snap))
                subs.append(f["Subhalos"][:])
                parts.extend(list(f["SubhaloParticles"][:]))
                if "Cosmology/ScaleFactor" in f:
                    self.scale_factor = float(f["Cosmology/ScaleFactor"][0])
        self.subs = np.concatenate(subs) if subs else np.zeros(0)
        for col in ("TrackId", "Nbound", "Rank"):
            if self.subs.dtype.names is None or col not in self.subs.dtype.names:
                raise InputError("%s: Subhalos has no %s" % (self.files[0], col))
        self.snap = snap
        self.ntot = self.subs.shape[0]
        self.trackid = self.subs["TrackId"].astype(np.int64)
        self.nbound = self.subs["Nbound"].astype(np.int64)
        lens = np.array([len(p) for p in parts], dtype=np.int64)
        bad = lens != self.nbound
        if np.any(bad):
            raise InputError("%d HBT subhalos with len(SubhaloParticles) != Nbound" % np.count_nonzero(bad))
        self.ids = np.concatenate(parts).astype(np.int64) if lens.sum() else np.zeros(0, np.int64)
        self.owner_of = np.repeat(np.arange(self.ntot, dtype=np.int64), lens)
        self.pos_in_list = np.arange(self.ids.size) - np.repeat(np.cumsum(lens) - lens, lens)
        o = np.argsort(self.ids, kind="stable")
        self.ids_sorted, self.owner_sorted = self.ids[o], self.owner_of[o]
        ndup = int(np.count_nonzero(self.ids_sorted[1:] == self.ids_sorted[:-1])) if self.ids.size else 0
        self.nduplicate = ndup
        self.vmax = self.subs["VmaxPhysical"].astype(np.float64) if "VmaxPhysical" in self.subs.dtype.names else np.full(self.ntot, np.nan)
        self.rank = self.subs["Rank"].astype(np.int64)
        self.alive = self.nbound > 0
        o = np.argsort(self.trackid, kind="stable")
        self.tid_sorted, self.tid_row = self.trackid[o], o.astype(np.int64)

    def row(self, trackids):
        return _lookup(self.tid_sorted, self.tid_row, trackids)

    def owner(self, ids):
        return _lookup(self.ids_sorted, self.owner_sorted, ids)

    def drop_particles(self):
        """Free the bound ID lists; row() and the Subhalos table keep working."""
        self.ids = self.owner_of = self.pos_in_list = self.ids_sorted = self.owner_sorted = None


# ---------------------------------------------------------------------------------- matching
class Match(object):
    pass


def match(ours, hbt, mode, ntracers):
    """Returns a Match with arrays over matched pairs: h (HBT row), o (our row), inter, jac."""
    # intersection table of the full bound sets: every HBT bound ID looked up in our bound owners
    o_of_h = ours.owner(hbt.ids)
    m = o_of_h >= 0
    pair_h = hbt.owner_of[m]
    pair_o = ours.row(o_of_h[m])
    if pair_h.size:
        pairs, cnt = np.unique(np.stack([pair_h, pair_o], axis=1), axis=0, return_counts=True)
    else:
        pairs, cnt = np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    nh = hbt.nbound[pairs[:, 0]]
    no = ours.nfull[pairs[:, 1]]
    jac = cnt / np.maximum(nh + no - cnt, 1).astype(np.float64)
    inter = {}
    for (h, o), c, j in zip(pairs.tolist(), cnt.tolist(), jac.tolist()):
        inter[(h, o)] = (c, j)

    if mode == "jaccard":
        # each side's best partner by Jaccard, ties to the smaller index
        oh = np.lexsort((pairs[:, 1], -jac, pairs[:, 0]))
        best_o_of_h = {}
        for k in oh.tolist():
            best_o_of_h.setdefault(int(pairs[k, 0]), int(pairs[k, 1]))
        oo = np.lexsort((pairs[:, 0], -jac, pairs[:, 1]))
        best_h_of_o = {}
        for k in oo.tolist():
            best_h_of_o.setdefault(int(pairs[k, 1]), int(pairs[k, 0]))
    else:
        # HBT tracers vote for our subhalo binding them
        typ = ours.types(hbt.ids)
        is_tr = np.isin(typ, ours.tracer_types) | (typ < 0)
        hsel = np.nonzero(is_tr)[0]
        hown = hbt.owner_of[hsel]
        keep = _first_n_per_group(hown, ntracers)
        hsel, hown = hsel[keep], hown[keep]
        i = np.zeros(hsel.size, np.int64)
        if hsel.size:
            start = np.ones(hsel.size, bool)
            start[1:] = hown[1:] != hown[:-1]
            idx = np.arange(hsel.size)
            i = idx - np.maximum.accumulate(np.where(start, idx, 0))
        cand = ours.row(ours.owner(hbt.ids[hsel]))
        v, w = _weighted_vote(hown, cand, 1.0 / (1.0 + np.sqrt(i)))
        best_o_of_h = dict(zip(v.tolist(), w.tolist()))
        # our tracers vote for HBT's subhalo binding them
        cand = hbt.owner(ours.tr_id)
        v, w = _weighted_vote(ours.row(ours.tr_track), cand, 1.0 / (1.0 + np.sqrt(ours.tr_pos)))
        best_h_of_o = dict(zip(v.tolist(), w.tolist()))
        ours.n_unknown_type = int(np.count_nonzero(typ < 0))

    M = Match()
    hs, os_ = [], []
    for h, o in best_o_of_h.items():
        if o >= 0 and best_h_of_o.get(o, -1) == h:
            hs.append(h)
            os_.append(o)
    M.h = np.array(hs, dtype=np.int64)
    M.o = np.array(os_, dtype=np.int64)
    srt = np.argsort(M.h)
    M.h, M.o = M.h[srt], M.o[srt]
    M.inter = np.array([inter.get((h, o), (0, 0.0))[0] for h, o in zip(M.h.tolist(), M.o.tolist())], dtype=np.int64)
    M.jac = np.array([inter.get((h, o), (0, 0.0))[1] for h, o in zip(M.h.tolist(), M.o.tolist())], dtype=np.float64)
    M.inter_table = inter
    return M


# ---------------------------------------------------------------------------------- report
D3_CLASSES = ["subsampled", "in place", "iteration cap", "few tracers", "spawned stars", "unexplained"]


def classify_pair(ours, hbt, h, o, sample, core, spawned):
    if sample > 0 and (hbt.nbound[h] > sample or ours.nbound[o] > sample):
        return "subsampled"
    if ours.flags[o] & FLAG_INPLACE:
        return "in place"
    if ours.flags[o] & FLAG_ITERCAP:
        return "iteration cap"
    if ours.ntracer[o] < core or hbt_tracers(hbt, ours, h) < core:
        return "few tracers"
    if spawned.get(o, 0) > 0:
        return "spawned stars"
    return "unexplained"


def hbt_tracers(hbt, ours, h):
    """HBT's bound tracer count; a DM-only HBT build writes no NboundType, every particle is DM."""
    if "NboundType" not in hbt.subs.dtype.names:
        return int(hbt.nbound[h])
    tt = [t for t in ours.tracer_types if t < hbt.subs["NboundType"].shape[1]]
    return int(hbt.subs["NboundType"][h, tt].sum())


# ---------------------------------------------------------------------------------- failure classes
FAIL_CLASSES = ["swap", "in-swapped-group", "nbound", "survival", "other"]
F32_U = 2.0 ** -24


class History(object):
    """Earlier outputs for the failure classes: the previous catalogue and its masses, each track's
    twin at its earliest pairing (both ways), the previous output's swaps and the first output."""

    def __init__(self):
        self.prev = self.prev_mass = self.first = None
        self.ident, self.rident = {}, {}
        self.swaps = {}

    def advance(self, ours, hbt, M, swaps):
        if self.first is None:
            self.first = ours.snap
        self.prev = ours
        self.prev_mass = ours.S["Mbound"].astype(np.float64)
        for t, th in zip(ours.tid[M.o].tolist(), hbt.trackid[M.h].tolist()):
            self.ident.setdefault(t, th)
            self.rident.setdefault(th, t)
        first = ("not checked", "already at SnapshotIndex %d, the first output" % ours.snap)
        self.swaps = {k: first if v == ("not checked", "no previous output") else v for k, v in swaps.items()}


def tie08(m_i, m_0):
    """(exact 4:5, float32 cut differs) for saved float32 masses m_i, m_0. Two exact 4:5 masses
    rounded to float32 (relative error <= 2^-24 each) give |5 m_i - 4 m_0| <= 2^-23 4 m_0 / (1 - 2^-24)."""
    m_i, m_0 = float(m_i), float(m_0)
    exact45 = abs(5.0 * m_i - 4.0 * m_0) <= 2 * F32_U * 4.0 * m_0 / (1 - F32_U)
    f = np.float32
    hbt_drop = bool(f(m_i) < f(f(m_0) * f(0.8)))
    return exact45, hbt_drop != (m_i < 0.8 * m_0)


def _central_rows(host, rank, alive):
    sel = np.nonzero(alive & (rank == 0) & (host >= 0))[0]
    return dict(zip(host[sel].tolist(), sel.tolist()))


def _members(host, ids, groups):
    """{group: [ids]} for the given groups, from one sort of host."""
    o = np.argsort(host, kind="stable")
    hs = host[o]
    out = {}
    for g in groups:
        lo, hi = np.searchsorted(hs, [g, g + 1])
        out[g] = ids[o[lo:hi]].tolist()
    return out


class Failures(object):
    """Classes of the failing pairs and unmatched HBT subhalos of one output (see the docstring).
    ratio, unmatched and pmap are report()'s Nbound ratios, unmatched HBT rows and pair map."""

    def __init__(self, ours, hbt, M, good, ratio, unmatched, pmap, hist):
        self.hist = hist = hist if hist is not None else History()
        self.ohost, self.hhost = ours.S["HostGroup"].astype(np.int64), hbt.subs["HostHaloId"].astype(np.int64)
        self.pm, rpm = pmap, {v: k for k, v in pmap.items()}
        self.oset = oset = set(ours.tid.tolist())
        hset = set(hbt.trackid[hbt.alive].tolist())

        # twins: the earliest pairing while that track is still resolved, else this output's pair
        def twin_h(t):
            x = hist.ident.get(t)
            return x if x is not None and x in hset else pmap.get(t)

        def twin_o(t):
            x = hist.rident.get(t)
            return x if x is not None and x in oset else rpm.get(t)

        oc = _central_rows(self.ohost, ours.rank, np.ones(ours.tid.size, bool))
        hc = _central_rows(self.hhost, hbt.rank, hbt.alive)
        cand = set(oc) & set(hc)
        # a pair with Rank 0 on one side only makes its group a swap group
        og, hg = self.ohost[M.o], self.hhost[M.h]
        role = (og == hg) & (og >= 0) & ((ours.rank[M.o] == 0) != (hbt.rank[M.h] == 0))
        rgroups = set(og[role].tolist()) & cand
        self.groups = {}
        for g in cand:
            A, C = int(ours.tid[oc[g]]), int(hbt.trackid[hc[g]])
            B, Ah = twin_o(C), twin_h(A)
            if g in rgroups or (B is not None and B != A) or (Ah is not None and Ah != C):
                # tie candidate: the historical twin of HBT's central even if dead now
                Bt = hist.rident.get(C, B)
                self.groups[g] = (A, C, B, Bt)
        self.ties = self._ties(ours)
        self.swaps = {(A, C): self.ties[g] for g, (A, C, B, Bt) in self.groups.items()}

        # pairs
        self.ratio = ratio
        self.cls, self.why = [None] * M.h.size, [None] * M.h.size
        for k in np.nonzero(~good)[0].tolist():
            o, h = M.o[k], M.h[k]
            g, gh, ro, rh = int(og[k]), int(hg[k]), int(ours.rank[o]), int(hbt.rank[h])
            nums = "Rank %d / %d, Nbound %d / %d (ratio %.3f), Jaccard %.3f" % (
                ro, rh, ours.nbound[o], hbt.nbound[h], ratio[k], M.jac[k])
            if g != gh:
                c, w = "other", "groups differ: our HostGroup %d, HBT HostHaloId %d; %s" % (g, gh, nums)
            elif g in self.groups and (ro == 0 or rh == 0):
                c, w = "swap", self._swap_text(g, nums)
            elif g >= 0 and (ro == 0) != (rh == 0):
                c, w = "other", "Rank 0 on one side, group %d has no central on the other; %s" % (g, nums)
            elif g in self.groups:
                c, w = "in-swapped-group", "group %d (centrals ours %d / HBT %d); %s" % (g, self.groups[g][0], self.groups[g][1], nums)
            else:
                c, w = "nbound", "group %d; %s" % (g, nums)
            self.cls[k], self.why[k] = c, w
        self.tie_pair = np.array([c == "swap" and self.ties[int(og[k])][0] == "tie-0.8"
                                  for k, c in enumerate(self.cls)], dtype=bool)
        self.unmatched = unmatched
        self.hbt_nbound_unmatched = hbt.nbound[unmatched]

    def _ties(self, ours):
        """{group: (label, text)}: tie-0.8, no tie or not checked, with the previous masses."""
        hist, P, out = self.hist, self.hist.prev, {}
        todo = []
        for g, (A, C, B, Bt) in self.groups.items():
            if P is None:
                out[g] = ("not checked", "no previous output")
            elif P.snap != ours.snap - 1:
                out[g] = ("not checked", "gap: the previous supplied output is SnapshotIndex %d" % P.snap)
            elif (A, C) in hist.swaps:
                out[g] = hist.swaps[(A, C)]
            elif Bt is None or Bt == A:
                out[g] = ("not checked", "no second candidate of ours for HBT's central")
            else:
                todo.append(g)
        if not todo:
            return out
        # candidates: the group's tracks now, resolved or sunk into it at this output, resolved before
        mem = _members(self.ohost, ours.tid, todo)
        if ours.otid.size:
            d = ours.O["SnapDeath"].astype(np.int64) == ours.snap
            dm = _members(ours.O["HostGroup"].astype(np.int64)[d], ours.otid[d], todo)
        m = hist.prev_mass
        for g in todo:
            A, C, B, Bt = self.groups[g]
            rA, rB = P.row([A, Bt]).tolist()
            if rA < 0 or rB < 0:
                out[g] = ("not checked", "candidate %d not resolved at SnapshotIndex %d" % (A if rA < 0 else Bt, P.snap))
                continue
            tr = mem[g] + (dm[g] if ours.otid.size else [])
            rows = [r for r in P.row(tr).tolist() if r >= 0] + [rA, rB]
            r0 = max(rows, key=lambda r: (m[r], -P.tid[r]))
            tests = set()
            for r in (rA, rB):
                if r != r0:
                    e, f = tie08(m[r], m[r0])
                    tests |= {"exact 4:5"} if e else set()
                    tests |= {"float32 cut differs"} if f else set()
            txt = "checked at SnapshotIndex %d on the Mbound (Nbound) of %d: %s; max %d %.7g (%d)" % (
                ours.snap, P.snap, ", ".join("%d %.7g (%d, ratio %.7f)" % (P.tid[r], m[r], P.nbound[r], m[r] / m[r0])
                                             for r in (rA, rB)), P.tid[r0], m[r0], P.nbound[r0])
            if tests:
                txt = "%s; %s" % (" and ".join(sorted(tests)), txt)
            out[g] = ("tie-0.8" if tests else "no tie", txt)
        return out

    def _swap_text(self, g, nums):
        A, C, B, Bt = self.groups[g]
        lab, txt = self.ties[g]
        tw = "-" if B is None else str(B)
        if Bt is not None and Bt != B:
            tw += ", historical %d" % Bt
        return "group %d, centrals ours %d / HBT %d (our twin %s); %s; %s: %s" % (g, A, C, tw, nums, lab, txt)

    def unmatched_text(self, ours, hbt, h):
        """One unmatched HBT subhalo and the state of its historical twin of ours (for --show)."""
        t = self.hist.rident.get(int(hbt.trackid[h]))
        if t is None:
            w = "no earlier pairing"
        elif t in self.oset:
            r = int(ours.row([t])[0])
            w = "our twin %d resolved (Nbound %d, Rank %d, group %d)%s" % (
                t, ours.nbound[r], ours.rank[r], self.ohost[r],
                ", paired with HBT %d" % self.pm[t] if t in self.pm else ", unmatched")
        elif ours.orow([t])[0] >= 0:
            w = "our twin %d is an orphan (sink %d)" % (t, ours.O["SinkTrackId"][ours.orow([t])[0]])
        else:
            w = "our twin %d is absent" % t
        return "HBT %d (Rank %d, Nbound %d, group %d%s): %s" % (
            hbt.trackid[h], hbt.rank[h], hbt.nbound[h], self.hhost[h],
            ", a swap group" if int(self.hhost[h]) in self.groups else "", w)

    def label(self, k):
        c = self.cls[k]
        if c == "swap" and self.tie_pair[k]:
            return "swap:tie-0.8"
        return c

    def counts(self, sel_pairs, sel_unmatched):
        cnt = {c: 0 for c in FAIL_CLASSES}
        for k in np.nonzero(sel_pairs)[0].tolist():
            if self.cls[k] is not None:
                cnt[self.cls[k]] += 1
        cnt["survival"] += int(np.count_nonzero(sel_unmatched))
        ntie = int(np.count_nonzero(sel_pairs & self.tie_pair))
        return cnt, ntie

    def count_line(self, lab, cnt, ntie):
        n = sum(cnt.values())
        parts = ["%s %d%s" % (c, cnt[c], " (tie-0.8 %d)" % ntie if c == "swap" else "")
                 for c in FAIL_CLASSES if cnt[c]]
        return "  failing %s: %d%s" % (lab, n, " = " + " + ".join(parts) if parts else "")


def _q(a, qs=(10, 50, 90)):
    if a.size == 0:
        return "n/a"
    return "/".join("%.3f" % np.percentile(a, q) for q in qs)


def class_lines(F, ours, M, args, fail_big, fail_near, nacc, nbig, out):
    """The failure-class lines after the acceptance line."""
    unb = F.hbt_nbound_unmatched
    cnt, ntie = F.counts(fail_big, unb >= args.min_np)
    print(F.count_line("HBT subhalos >= %d" % args.min_np, cnt, ntie), file=out)
    groups = sorted(set(int(F.ohost[M.o[k]]) for k in np.nonzero(fail_big)[0].tolist() if F.cls[k] == "swap"))
    hist = F.hist
    if hist.prev is None:
        print("  tie check not done: no previous output (give --prev PIG_n-1, or use --track-pairs)", file=out)
    elif not groups:
        print("  tie check: no swap >= %d" % args.min_np, file=out)
    else:
        tl = {}
        for g in groups:
            lab, txt = F.ties[g]
            key = lab if lab != "not checked" else "not checked (%s)" % txt
            tl[key] = tl.get(key, 0) + 1
        print("  tie check of the %d group%s with a swap >= %d: %s" % (
            len(groups), "" if len(groups) == 1 else "s", args.min_np,
            ", ".join("%s %d" % kv for kv in sorted(tl.items()))), file=out)
    # both-Rank-0 swaps are seen only through twins from earlier outputs
    if hist.first is None:
        print("  swap history: none; a swap with both sides Rank 0 on different tracks is invisible and counts "
              "as nbound (use --prev, or a consecutive --track-pairs sequence from before its origin)", file=out)
    else:
        print("  swap history: twins from SnapshotIndex %d on; a swap with both sides Rank 0 that began at or "
              "before it is invisible and counts as nbound (a consecutive sequence from before its origin "
              "is needed)" % hist.first, file=out)
    print("  acceptance excluding tie-level pairs (%d swap%s labelled tie-0.8): %d of %d, %.1f%%" % (
        ntie, "" if ntie == 1 else "s", nacc, nbig - ntie, 100.0 * nacc / max(nbig - ntie, 1)), file=out)
    cnt, ntie = F.counts(fail_near, (unb >= args.near_np) & (unb < args.min_np))
    print(F.count_line("near threshold (%d <= HBT Nbound < %d)" % (args.near_np, args.min_np), cnt, ntie), file=out)


def report(ours, hbt, M, args, out=sys.stdout, hist=None):
    sample = int(ours.params.get("SubhaloPotentialSampleSize", 1000))
    core = int(ours.params.get("SubhaloCoreSize", 20))
    alive = np.nonzero(hbt.alive)[0]
    big = alive[hbt.nbound[alive] >= args.min_np]
    matched_h = np.zeros(hbt.ntot, bool)
    matched_h[M.h] = True
    matched_o = np.zeros(ours.tid.size, bool)
    matched_o[M.o] = True
    ratio = ours.nbound[M.o] / np.maximum(hbt.nbound[M.h], 1).astype(np.float64)
    vr = np.where(ours.vmax[M.o] > 0, ours.vmax[M.o] / np.where(hbt.vmax[M.h] > 0, hbt.vmax[M.h], np.nan), np.nan)
    good = (M.jac >= args.jaccard_min) & (np.abs(ratio - 1) <= args.nbound_tol)
    isbig = hbt.nbound[M.h] >= args.min_np
    # spawned stars: type-4 particles in our full set absent from HBT's
    spawned = {}
    s4 = ours.bid_type == 4
    if np.any(s4):
        ids4, own4 = ours.bid_sorted[s4], ours.bid_owner[s4]
        nohbt = hbt.owner(ids4) < 0
        r, c = np.unique(ours.row(own4[nohbt]), return_counts=True)
        spawned = {int(k): int(v) for k, v in zip(r.tolist(), c.tolist())}

    print("Our catalogue %s (SnapshotIndex %d): %d subhalos, %d orphans%s" % (
        ours.pigdir, ours.snap, ours.tid.size, ours.cat.norphans, " [--phase1: section 0 only]" if args.phase1 else ""), file=out)
    if ours.nfull_mismatch:
        print("  WARNING: %d subhalos whose BoundTrackId count differs from Nbound (run subhalo_check.py tags)"
              % ours.nfull_mismatch, file=out)
    print("HBT %s: %d subhalos (%d with Nbound > 0, %d orphans), %d with Nbound >= %d" % (
        os.path.dirname(hbt.files[0]), hbt.ntot, alive.size, hbt.ntot - alive.size, big.size, args.min_np), file=out)
    if hbt.nduplicate:
        print("  WARNING: %d particle IDs appear in more than one HBT bound list" % hbt.nduplicate, file=out)
    if args.match == "tracers" and getattr(ours, "n_unknown_type", 0):
        print("  note: %d HBT bound IDs are not saved in the PIG; their type is unknown and they count as tracers"
              % ours.n_unknown_type, file=out)
    print("Matching: %s (%d tracers)" % (args.match, args.ntracers) if args.match == "tracers" else "Matching: jaccard", file=out)
    print("  matched pairs %d: %.1f%% of HBT subhalos (Nbound > 0), %.1f%% of HBT subhalos with Nbound >= %d" % (
        M.h.size, 100.0 * M.h.size / max(alive.size, 1), 100.0 * np.count_nonzero(matched_h[big]) / max(big.size, 1),
        args.min_np), file=out)
    isnear = (hbt.nbound[M.h] >= args.near_np) & (hbt.nbound[M.h] < args.min_np)
    near = alive[(hbt.nbound[alive] >= args.near_np) & (hbt.nbound[alive] < args.min_np)]
    print("  near threshold: %d HBT subhalos with %d <= Nbound < %d, matched %.1f%%" % (
        near.size, args.near_np, args.min_np, 100.0 * np.count_nonzero(matched_h[near]) / max(near.size, 1)), file=out)
    for lab, sel in (("all", np.ones(M.h.size, bool)), (">= %d" % args.min_np, isbig),
                     ("%d-%d" % (args.near_np, args.min_np - 1), isnear)):
        print("  pairs %-6s n=%-5d Jaccard p10/p50/p90 %s mean %s   Nbound ratio p10/p50/p90 %s   Vmax ratio p10/p50/p90 %s (%d with Vmax)" % (
            lab, np.count_nonzero(sel), _q(M.jac[sel]), "%.3f" % M.jac[sel].mean() if np.any(sel) else "n/a",
            _q(ratio[sel]), _q(vr[sel][np.isfinite(vr[sel])]), np.count_nonzero(np.isfinite(vr[sel]))), file=out)
    for lab, rk in (("centrals (HBT Rank 0)", 0), ("satellites (HBT Rank > 0)", 1)):
        hsel = big[(hbt.rank[big] == 0) if rk == 0 else (hbt.rank[big] > 0)]
        msel = isbig & ((hbt.rank[M.h] == 0) if rk == 0 else (hbt.rank[M.h] > 0))
        print("  HBT %-26s >= %d: %4d, matched %4d, median Jaccard %s, median Nbound ratio %s" % (
            lab, args.min_np, hsel.size, np.count_nonzero(matched_h[hsel]),
            "%.3f" % np.median(M.jac[msel]) if np.any(msel) else "n/a",
            "%.3f" % np.median(ratio[msel]) if np.any(msel) else "n/a"), file=out)
    um_h = alive[~matched_h[alive]]
    um_o = np.nonzero(~matched_o)[0]
    print("  unmatched HBT subhalos: %d (%d with Nbound >= %d: %d centrals, %d satellites)" % (
        um_h.size, np.count_nonzero(hbt.nbound[um_h] >= args.min_np),
        args.min_np, np.count_nonzero((hbt.nbound[um_h] >= args.min_np) & (hbt.rank[um_h] == 0)),
        np.count_nonzero((hbt.nbound[um_h] >= args.min_np) & (hbt.rank[um_h] > 0))), file=out)
    print("  unmatched subhalos of ours: %d (%d with Nbound >= %d: %d centrals, %d satellites)" % (
        um_o.size, np.count_nonzero(ours.nbound[um_o] >= args.min_np), args.min_np,
        np.count_nonzero((ours.nbound[um_o] >= args.min_np) & (ours.rank[um_o] == 0)),
        np.count_nonzero((ours.nbound[um_o] >= args.min_np) & (ours.rank[um_o] > 0))), file=out)

    # acceptance and D3 classes
    nacc = np.count_nonzero(good & isbig)
    frac = nacc / float(max(big.size, 1))
    print("Acceptance: %d of %d HBT subhalos with Nbound >= %d matched with Jaccard >= %.2f and Nbound within %.0f%%: "
          "%.1f%% (target %.0f%%)" % (nacc, big.size, args.min_np, args.jaccard_min, 100 * args.nbound_tol,
                                     100 * frac, 100 * args.accept_frac), file=out)
    M.fail = F = None
    if "HostHaloId" not in hbt.subs.dtype.names:
        print("  failure classes not available: %s has no Subhalos/HostHaloId" % hbt.files[0], file=out)
    else:
        M.fail = F = Failures(ours, hbt, M, good, ratio, um_h, pair_map(ours, hbt, M), hist)
        class_lines(F, ours, M, args, isbig & ~good, isnear & ~good, nacc, big.size, out)
    cls = {c: 0 for c in D3_CLASSES}
    for k in np.nonzero(isbig & ~good)[0].tolist():
        cls[classify_pair(ours, hbt, M.h[k], M.o[k], sample, core, spawned)] += 1
    ucls = {"subsampled": 0, "few tracers": 0, "unexplained": 0}
    for h in um_h[hbt.nbound[um_h] >= args.min_np].tolist():
        if sample > 0 and hbt.nbound[h] > sample:
            ucls["subsampled"] += 1
        elif hbt_tracers(hbt, ours, h) < core:
            ucls["few tracers"] += 1
        else:
            ucls["unexplained"] += 1
    print("  D3 classes of matched pairs >= %d below the criterion: %s" % (
        args.min_np, ", ".join("%s %d" % (c, cls[c]) for c in D3_CLASSES if cls[c]) or "none"), file=out)
    print("  D3 classes of unmatched HBT subhalos >= %d: %s" % (
        args.min_np, ", ".join("%s %d" % (c, v) for c, v in ucls.items() if v) or "none"), file=out)
    nacc_near = np.count_nonzero(good & isnear)
    print("  near threshold (%d <= HBT Nbound < %d): %d of %d matched with Jaccard >= %.2f and Nbound within %.0f%%: %.1f%%"
          % (args.near_np, args.min_np, nacc_near, near.size, args.jaccard_min, 100 * args.nbound_tol,
             100.0 * nacc_near / max(near.size, 1)), file=out)
    if spawned:
        print("  note: %d of our subhalos bind type-4 particles absent from HBT's lists (%d particles)"
              % (len(spawned), sum(spawned.values())), file=out)

    if args.show:
        o = np.argsort(M.jac)[:args.show]
        print("  worst %d pairs:  HBT_TrackId HBT_Rank HBT_Nbound  our_TrackId our_Rank our_Nbound  Jaccard  Nratio  Vratio" % len(o), file=out)
        for k in o.tolist():
            h, oo = M.h[k], M.o[k]
            print("                  %11d %8d %10d  %11d %8d %10d  %7.3f  %6.3f  %6s" % (
                hbt.trackid[h], hbt.rank[h], hbt.nbound[h], ours.tid[oo], ours.rank[oo], ours.nbound[oo],
                M.jac[k], ratio[k], "%.3f" % vr[k] if np.isfinite(vr[k]) else "-"), file=out)
            if F is not None and F.cls[k] is not None:
                print("                    %s: %s" % (F.label(k), F.why[k]), file=out)
        uw = um_h[hbt.nbound[um_h] >= args.min_np] if F is not None else []
        if len(uw):
            print("  unmatched HBT subhalos >= %d (survival)%s:" % (
                args.min_np, ", first %d" % args.show if len(uw) > args.show else ""), file=out)
            for h in uw[:args.show].tolist():
                print("                  %s" % F.unmatched_text(ours, hbt, h), file=out)
    if args.pairs_out:
        with open(args.pairs_out, "w") as f:
            f.write("# HBT_TrackId HBT_Rank HBT_Depth HBT_Nbound our_TrackId our_Rank our_Depth our_Nbound "
                    "intersection Jaccard Nbound_ratio Vmax_ratio class fail_class\n")
            for k in range(M.h.size):
                h, oo = M.h[k], M.o[k]
                c = "ok" if good[k] else classify_pair(ours, hbt, h, oo, sample, core, spawned).replace(" ", "_")
                f.write("%d %d %d %d %d %d %d %d %d %.5f %.5f %s %s %s\n" % (
                    hbt.trackid[h], hbt.rank[h], hbt.subs["Depth"][h], hbt.nbound[h], ours.tid[oo], ours.rank[oo],
                    ours.S["Depth"][oo], ours.nbound[oo], M.inter[k], M.jac[k], ratio[k],
                    "%.5f" % vr[k] if np.isfinite(vr[k]) else "nan", c, (F.label(k) or "-") if F is not None else "n/a"))
        print("  pairs written to %s" % args.pairs_out, file=out)
    return frac


def load(pigdir, hbtpath, args):
    ours = Ours(pigdir, args.ntracers, phase1=args.phase1)
    hbt = HBT(hbtpath, ours.snap + args.hbt_snap_offset)
    return ours, hbt


def prev_history(pig, hbtpath, args, snap):
    """History holding the output before SnapshotIndex snap; swaps already there are not tie-checked."""
    ours, hbt = load(pig, hbtpath, args)
    if ours.snap >= snap:
        raise InputError("--prev %s is SnapshotIndex %d, not before %d" % (pig, ours.snap, snap))
    M = match(ours, hbt, args.match, args.ntracers)
    swaps = {}
    if "HostHaloId" in hbt.subs.dtype.names:
        um = np.setdiff1d(np.nonzero(hbt.alive)[0], M.h)
        ratio = ours.nbound[M.o] / np.maximum(hbt.nbound[M.h], 1).astype(np.float64)
        F = Failures(ours, hbt, M, np.ones(M.h.size, bool), ratio, um, pair_map(ours, hbt, M), None)
        swaps = {k: ("not checked", "already at SnapshotIndex %d" % ours.snap) for k in F.swaps}
    hist = History()
    hist.advance(ours, hbt, M, swaps)
    return hist


# ---------------------------------------------------------------------------------- links, history
UNMAPPED = -(2 ** 62)


def pair_map(ours, hbt, M):
    """{our TrackId: HBT TrackId} of the matched pairs."""
    return dict(zip(ours.tid[M.o].tolist(), hbt.trackid[M.h].tolist()))


def _map_ids(ids, *maps):
    """Our TrackIds through the first map that holds them: -1 stays -1, UNMAPPED when none does."""
    out = np.full(np.shape(ids), UNMAPPED, np.int64)
    for k, t in enumerate(np.asarray(ids, np.int64).tolist()):
        if t < 0:
            out[k] = -1
            continue
        for mp in maps:
            if t in mp:
                out[k] = mp[t]
                break
    return out


def _hsnap(v, off):
    """HBT snapshot numbers in our SnapshotIndex (-1 kept)."""
    v = np.asarray(v, np.int64)
    return np.where(v >= 0, v - off, -1)


def _pct(e, n):
    return "%5.1f%%" % (100.0 * e / n) if n else "  n/a"


def _pops(nb, args):
    nb = np.asarray(nb)
    return [("all", np.ones(nb.size, bool)), (">=%d" % args.min_np, nb >= args.min_np),
            ("%d-%d" % (args.near_np, args.min_np - 1), (nb >= args.near_np) & (nb < args.min_np))]


def _eq_line(name, eq, valid, pops, out, note=""):
    parts = []
    for lab, m in pops:
        v = valid & m
        n, e = int(np.count_nonzero(v)), int(np.count_nonzero(eq & v))
        parts.append("%s %d/%d %s" % (lab, e, n, _pct(e, n)))
    print("    %-30s equal  %s%s" % (name, "   ".join(parts), note), file=out)


def _ratio_line(name, r, valid, pops, out):
    parts = []
    for lab, m in pops:
        v = valid & m & np.isfinite(r)
        x = r[v]
        n = x.size
        parts.append("%s n=%d p10/p50/p90 %s within 1%% %s 10%% %s" % (
            lab, n, _q(x), _pct(np.count_nonzero(np.abs(x - 1) <= 0.01), n).strip(),
            _pct(np.count_nonzero(np.abs(x - 1) <= 0.1), n).strip()))
    print("    %-30s ratio  %s" % (name, ";  ".join(parts)), file=out)


def _list(out, title, rows, args):
    if not rows:
        return
    print("      %s (%d%s):" % (title, len(rows), ", first %d" % args.list if len(rows) > args.list else ""), file=out)
    for r in rows[:args.list]:
        print("        %s" % r, file=out)


def link_block(ours, hbt, M, pmap, args, out=sys.stdout):
    """Links of the matched pairs at one snapshot, through that snapshot's pair map."""
    o, h = M.o, M.h
    S, H = ours.S, hbt.subs
    pops = _pops(hbt.nbound[h], args)
    print("  links of the %d matched pairs (%s):" % (h.size, " / ".join(lab for lab, _ in pops)), file=out)
    if h.size == 0:
        return
    ones = np.ones(h.size, bool)
    pid, hpid = S["NestedParentTrackId"][o].astype(np.int64), H["NestedParentTrackId"][h].astype(np.int64)
    mp = _map_ids(pid, pmap)
    valid = mp != UNMAPPED
    eq = np.where(pid < 0, hpid < 0, mp == hpid)
    _eq_line("NestedParentTrackId", eq, valid, pops, out,
             "   (%d with our parent unmatched)" % np.count_nonzero(~valid))
    hg, hh = S["HostGroup"][o].astype(np.int64), H["HostHaloId"][h].astype(np.int64)
    _eq_line("HostGroup / HostHaloId", hg == hh, ones, pops, out)
    rk, hrk = ours.rank[o], hbt.rank[h]
    _eq_line("Rank", rk == hrk, ones, pops, out)
    dp, hdp = S["Depth"][o].astype(np.int64), H["Depth"][h].astype(np.int64)
    _eq_line("Depth", dp == hdp, ones, pops, out)
    rows = []
    for k in np.nonzero(~(eq | ~valid) | (hg != hh) | (rk != hrk) | (dp != hdp))[0].tolist():
        rows.append("our %d / HBT %d (HBT Nbound %d): pid %d->%s vs %d, host %d vs %d, rank %d vs %d, depth %d vs %d" % (
            ours.tid[o[k]], hbt.trackid[h[k]], hbt.nbound[h[k]], pid[k], "?" if mp[k] == UNMAPPED else mp[k], hpid[k],
            hg[k], hh[k], rk[k], hrk[k], dp[k], hdp[k]))
    _list(out, "pairs with a link difference", rows, args)


def history_block(title, oT, hrows, ours_tab, hbt, args, out=sys.stdout, nb=None):
    """History columns of paired tracks: ours_tab rows oT (Subhalo or Orphan table), HBT rows hrows."""
    H = hbt.subs
    off = args.hbt_snap_offset
    pops = _pops(nb, args) if nb is not None else [("all", np.ones(len(hrows), bool))]
    print("  history of %s (%d, %s):" % (title, len(hrows), " / ".join(lab for lab, _ in pops)), file=out)
    if len(hrows) == 0:
        return
    ones = np.ones(len(hrows), bool)
    lm = H["LastMaxMass"][hrows].astype(np.float64)
    r = ours_tab["MboundPeak"][oT].astype(np.float64) / np.where(lm > 0, lm, np.nan)
    _ratio_line("MboundPeak / LastMaxMass", r, ones, pops, out)
    lv = H["LastMaxVmaxPhysical"][hrows].astype(np.float64)
    r = ours_tab["VmaxPeak"][oT].astype(np.float64) / np.where(lv > 0, lv, np.nan)
    _ratio_line("VmaxPeak / LastMaxVmaxPhys", r, ones, pops, out)
    rows = []
    for ours_col, hbt_col in (("SnapLastIsolation", "SnapshotOfLastIsolation"), ("SnapBirth", "SnapshotOfBirth"),
                              ("SnapDeath", "SnapshotOfDeath"), ("SnapSink", "SnapshotOfSink")):
        a = ours_tab[ours_col][oT].astype(np.int64)
        b = _hsnap(H[hbt_col][hrows], off)
        eq = a == b
        _eq_line("%s / %s" % (ours_col, hbt_col[len("SnapshotOf"):]), eq, ones, pops, out)
        for k in np.nonzero(~eq)[0].tolist():
            rows.append("our %d / HBT %d: %s %d vs %d" % (ours_tab["TrackId"][oT[k]], hbt.trackid[hrows[k]], ours_col, a[k], b[k]))
    _list(out, "snapshot differences", rows, args)


def death_block(B, HB, cum_prev, map_prev, map_cur, args, out=sys.stdout):
    """Tracks that died (or sank) at B's output on either side, paired by the earlier TrackId pairs."""
    snap, off = B.snap, args.hbt_snap_offset
    H = HB.subs
    od = np.nonzero(B.O["SnapDeath"].astype(np.int64) == snap)[0] if B.otid.size else np.zeros(0, np.int64)
    hd = np.nonzero((HB.nbound == 0) & (_hsnap(H["SnapshotOfDeath"], off) == snap))[0]
    rev = {v: k for k, v in cum_prev.items()}
    cnt = {"both": 0, "ours only (HBT partner resolved)": 0, "ours only (HBT partner dead earlier)": 0,
           "ours, no earlier partner": 0, "HBT only (our partner resolved)": 0, "HBT only (our partner dead earlier)": 0,
           "HBT, no earlier partner": 0}
    both_o, both_h = [], []
    rows = []
    for r in od.tolist():
        T = int(B.otid[r])
        if T not in cum_prev:
            cnt["ours, no earlier partner"] += 1
            continue
        hr = int(HB.row([cum_prev[T]])[0])
        if hr >= 0 and HB.nbound[hr] == 0 and _hsnap([H["SnapshotOfDeath"][hr]], off)[0] == snap:
            cnt["both"] += 1
            both_o.append(r)
            both_h.append(hr)
        elif hr >= 0 and HB.nbound[hr] > 0:
            cnt["ours only (HBT partner resolved)"] += 1
            rows.append("our %d died, HBT %d resolved (Nbound %d)" % (T, cum_prev[T], HB.nbound[hr]))
        else:
            cnt["ours only (HBT partner dead earlier)"] += 1
    for hr in hd.tolist():
        Th = int(HB.trackid[hr])
        if Th not in rev:
            cnt["HBT, no earlier partner"] += 1
            continue
        T = rev[Th]
        if B.orow([T])[0] >= 0 and B.O["SnapDeath"][B.orow([T])[0]] == snap:
            continue
        if B.row([T])[0] >= 0:
            cnt["HBT only (our partner resolved)"] += 1
            rows.append("HBT %d died, our %d resolved (Nbound %d)" % (Th, T, B.nbound[B.row([T])[0]]))
        else:
            cnt["HBT only (our partner dead earlier)"] += 1
    print("  deaths at SnapshotIndex %d: ours %d, HBT %d; %s" % (
        snap, od.size, hd.size, ", ".join("%s %d" % kv for kv in cnt.items())), file=out)
    _list(out, "deaths on one side only", rows, args)
    if not both_o:
        return
    bo, bh = np.array(both_o), np.array(both_h)
    ones = np.ones(bo.size, bool)
    pops = [("both died", ones)]
    sk, hsk = B.O["SinkTrackId"][bo].astype(np.int64), H["SinkTrackId"][bh].astype(np.int64)
    ds, hds = B.O["DescendantTrackId"][bo].astype(np.int64), H["DescendantTrackId"][bh].astype(np.int64)
    msk, mds = _map_ids(sk, map_prev, map_cur), _map_ids(ds, map_prev, map_cur)
    _eq_line("sunk (SinkTrackId >= 0)", (sk >= 0) == (hsk >= 0), ones, pops, out)
    eqs = np.where(sk < 0, hsk < 0, msk == hsk)
    _eq_line("SinkTrackId (mapped)", eqs, msk != UNMAPPED, pops, out,
             "   (%d unmapped)" % np.count_nonzero(msk == UNMAPPED))
    eqd = np.where(ds < 0, hds < 0, mds == hds)
    _eq_line("DescendantTrackId (mapped)", eqd, mds != UNMAPPED, pops, out,
             "   (%d unmapped)" % np.count_nonzero(mds == UNMAPPED))
    rows = []
    for k in np.nonzero(~eqs | ~eqd)[0].tolist():
        rows.append("our %d / HBT %d: sink %d->%s vs %d, descendant %d->%s vs %d" % (
            B.otid[bo[k]], HB.trackid[bh[k]], sk[k], "?" if msk[k] == UNMAPPED else msk[k], hsk[k],
            ds[k], "?" if mds[k] == UNMAPPED else mds[k], hds[k]))
    _list(out, "deaths with a link difference", rows, args)


def orphan_block(B, HB, cum_prev, args, out=sys.stdout):
    """Orphans at B's output paired by the earlier TrackId pairs: hosts and history."""
    H = HB.subs
    po, ph = [], []
    alive_h = 0
    nopair = 0
    for r in range(B.otid.size):
        T = int(B.otid[r])
        if T not in cum_prev:
            nopair += 1
            continue
        hr = int(HB.row([cum_prev[T]])[0])
        if hr < 0:
            nopair += 1
        elif HB.nbound[hr] > 0:
            alive_h += 1
        else:
            po.append(r)
            ph.append(hr)
    print("  orphans at SnapshotIndex %d: ours %d, HBT %d; paired %d, HBT partner resolved %d, no earlier partner %d" % (
        B.snap, B.otid.size, np.count_nonzero(HB.nbound == 0), len(po), alive_h, nopair), file=out)
    if not po:
        return
    po, ph = np.array(po), np.array(ph)
    ones = np.ones(po.size, bool)
    hg, hh = B.O["HostGroup"][po].astype(np.int64), H["HostHaloId"][ph].astype(np.int64)
    _eq_line("orphan HostGroup / HostHaloId", hg == hh, ones, [("all", ones), ("hosted (HBT)", hh >= 0)], out,
             "   (ours -1: %d, HBT -1: %d)" % (np.count_nonzero(hg < 0), np.count_nonzero(hh < 0)))
    rows = ["our %d / HBT %d: HostGroup %d vs HostHaloId %d" % (B.otid[po[k]], HB.trackid[ph[k]], hg[k], hh[k])
            for k in np.nonzero(hg != hh)[0].tolist()]
    _list(out, "orphans with a host difference", rows, args)
    history_block("the orphan pairs", po, ph, B.O, HB, args, out)


def sequence(args, pigs, hbtpath):
    cats, matches, maps = [], [], []
    frac = 0.0
    hist = None
    for pig in pigs:
        ours, hbt = load(pig, hbtpath, args)
        if hist is None:
            hist = prev_history(args.prev, hbtpath, args, ours.snap) if args.prev else History()
        elif ours.snap <= hist.prev.snap:
            raise InputError("--track-pairs: %s (SnapshotIndex %d) does not follow SnapshotIndex %d"
                             % (pig, ours.snap, hist.prev.snap))
        M = match(ours, hbt, args.match, args.ntracers)
        print("== %s" % pig)
        frac = report(ours, hbt, M, args, hist=hist)
        hist.advance(ours, hbt, M, M.fail.swaps if M.fail is not None else {})
        pm = pair_map(ours, hbt, M)
        link_block(ours, hbt, M, pm, args)
        history_block("the matched pairs", M.o, M.h, ours.S, hbt, args, nb=hbt.nbound[M.h])
        cats.append((ours, hbt))
        matches.append(M)
        maps.append(pm)
        if len(cats) > 1:   # the previous output's particles served its tie check; only tables are needed now
            for c in cats[-2]:
                c.drop_particles()
    cum = {}
    tot_persist, tot_both = 0, 0
    for i in range(1, len(pigs)):
        A, HA = cats[i - 1]
        B, HB = cats[i]
        MA = matches[i - 1]
        cum.update(maps[i - 1])
        nxt = maps[i]
        hb_alive = dict(zip(HB.trackid.tolist(), HB.alive.tolist()))
        print("== %s -> %s" % (pigs[i - 1], pigs[i]))
        nbA = HA.nbound[MA.h]
        persist_line = []
        breaks = []
        for lab, sel in _pops(nbA, args):
            cnt = {"persist": 0, "both dead or sunk": 0, "only ours dead or sunk": 0, "only HBT's dead or sunk": 0,
                   "ours unmatched": 0, "other partner": 0}
            for k in np.nonzero(sel)[0].tolist():
                to, th = int(A.tid[MA.o[k]]), int(HA.trackid[MA.h[k]])
                o_alive = B.row([to])[0] >= 0
                h_alive = bool(hb_alive.get(th, False))
                why = None
                if not o_alive and not h_alive:
                    cnt["both dead or sunk"] += 1
                elif not o_alive:
                    cnt["only ours dead or sunk"] += 1
                    why = "ours dead or sunk, HBT's resolved"
                elif not h_alive:
                    cnt["only HBT's dead or sunk"] += 1
                    why = "HBT's dead or sunk, ours resolved"
                elif to not in nxt:
                    cnt["ours unmatched"] += 1
                    why = "ours unmatched at the next output"
                elif nxt[to] != th:
                    cnt["other partner"] += 1
                    why = "now paired with HBT %d" % nxt[to]
                else:
                    cnt["persist"] += 1
                if why and lab == "all":
                    breaks.append("our %d / HBT %d (HBT Nbound %d): %s" % (to, th, HA.nbound[MA.h[k]], why))
            both = cnt["persist"] + cnt["ours unmatched"] + cnt["other partner"]
            if lab == ">=%d" % args.min_np:
                tot_persist += cnt["persist"]
                tot_both += both
            persist_line.append("%s: %d of %d %s (%s)" % (lab, cnt["persist"], both, _pct(cnt["persist"], both).strip(),
                                                         ", ".join("%s %d" % kv for kv in cnt.items() if kv[0] != "persist" and kv[1])
                                                         or "no break"))
        print("  temporal identity, pairs persisting where both tracks are still resolved (by HBT Nbound at the earlier output):",
              file=sys.stdout)
        for line in persist_line:
            print("    %s" % line)
        _list(sys.stdout, "identity breaks", breaks, args)
        death_block(B, HB, cum, maps[i - 1], nxt, args)
        orphan_block(B, HB, cum, args)
    pf = tot_persist / float(max(tot_both, 1))
    print("== TrackId persistence over the sequence (HBT Nbound >= %d): %d of %d, %.1f%% (target %.0f%%)"
          % (args.min_np, tot_persist, tot_both, 100 * pf, 100 * args.persist_frac))
    ok = frac >= args.accept_frac and pf >= args.persist_frac
    return 0 if (ok or args.report_only) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="PIG_xxx HBT (one snapshot), or HBT alone with --track-pairs")
    p.add_argument("--track-pairs", nargs="+", metavar="PIG",
                   help="a sequence of PIGs in order (then the HBT path, unless given positionally): every "
                        "snapshot, temporal identity, deaths and orphan hosts across consecutive outputs")
    p.add_argument("--match", choices=["tracers", "jaccard"], default="tracers")
    p.add_argument("--ntracers", type=int, default=10, help="tracers per subhalo for matching (default 10)")
    p.add_argument("--min-np", type=int, default=100, help="HBT Nbound threshold of the summary (default 100)")
    p.add_argument("--near-np", type=int, default=20,
                   help="lower end of the near-threshold population --near-np <= HBT Nbound < --min-np (default 20)")
    p.add_argument("--list", type=int, default=20, help="mismatches and identity breaks listed per block (default 20)")
    p.add_argument("--jaccard-min", type=float, default=0.8)
    p.add_argument("--nbound-tol", type=float, default=0.2)
    p.add_argument("--accept-frac", type=float, default=0.9)
    p.add_argument("--persist-frac", type=float, default=0.9)
    p.add_argument("--hbt-snap-offset", type=int, default=0,
                   help="HBT snapshot index minus our SnapshotIndex (0 when HBT ran on the same outputs)")
    p.add_argument("--phase1", action="store_true", help="Phase 1 catalogue: read section 0 only")
    p.add_argument("--report-only", action="store_true", help="exit 0 whatever the figures")
    p.add_argument("--show", type=int, default=0, help="print the N pairs with the lowest Jaccard")
    p.add_argument("--pairs-out", default=None, help="write every matched pair to this text file")
    p.add_argument("--prev", default=None, metavar="PIG",
                   help="the output before PIG_xxx (or before the first PIG of --track-pairs), for the "
                        "swap identities and the tie-0.8 check")
    args = p.parse_args(argv)
    try:
        if args.track_pairs:
            pigs = list(args.track_pairs)
            if len(args.paths) == 1:
                hbtpath = args.paths[0]
            elif len(args.paths) == 0 and len(pigs) >= 3:
                hbtpath = pigs.pop()
            else:
                p.error("with --track-pairs give two or more PIGs and the HBT path")
            if len(pigs) < 2:
                p.error("--track-pairs needs at least two PIGs")
            if args.pairs_out:
                p.error("--pairs-out is for one snapshot")
            return sequence(args, pigs, hbtpath)
        if len(args.paths) != 2:
            p.error("give PIG_xxx and the HBT path")
        ours, hbt = load(args.paths[0], args.paths[1], args)
        hist = prev_history(args.prev, args.paths[1], args, ours.snap) if args.prev else None
        M = match(ours, hbt, args.match, args.ntracers)
        frac = report(ours, hbt, M, args, hist=hist)
        link_block(ours, hbt, M, pair_map(ours, hbt, M), args)
        history_block("the matched pairs", M.o, M.h, ours.S, hbt, args, nb=hbt.nbound[M.h])
        return 0 if (frac >= args.accept_frac or args.report_only) else 1
    except (SubhaloFormatError, InputError, IOError, OSError) as e:
        print("error: %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
