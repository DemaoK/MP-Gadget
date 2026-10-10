#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the runtime subhalo finder's catalogue in a PIG_xxx with HBT-BF's SubSnap_xxx
(halofinder_phase2a_plan.md section 6, halofinder_design.md section 13 and D3).

Usage:
  subhalo_compare_hbt.py PIG_xxx HBT [options]
      one snapshot. HBT is HBT's SubhaloPath (per-snapshot directories NNN/), one NNN/ directory,
      or one SubSnap_NNN.*.hdf5 file (its sibling files are read too); NNN is the PIG's
      SnapshotIndex plus --hbt-snap-offset.
  subhalo_compare_hbt.py --track-pairs PIG_prev PIG_next HBT [options]
      both snapshots, then TrackId persistence: a pair (our TrackId, HBT TrackId) matched at the
      first output should be matched again at the second when both tracks are still resolved.

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

Exit status: 0 when the acceptance figure (and, with --track-pairs, the persistence fraction
against --persist-frac) is met or --report-only is given; 1 when it is not; 2 when an input could
not be read.

Phase 1 catalogues (every FOF group one subhalo, field particles keeping stale tags): pass
--phase1 so that only section 0 (the FOF groups) is read; matching by tracer IDs still works.

Examples:
  python3 tools/subhalo_compare_hbt.py output/PIG_004 hbt/ --phase1 --report-only
  python3 tools/subhalo_compare_hbt.py output/PIG_004 hbt/ --match jaccard --pairs-out pairs_004.txt
  python3 tools/subhalo_compare_hbt.py --track-pairs output/PIG_003 output/PIG_004 hbt/
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

    def row(self, trackids):
        return _lookup(self.tid, np.arange(self.tid.size), trackids)

    def owner(self, ids):
        return _lookup(self.bid_sorted, self.bid_owner, ids)

    def types(self, ids):
        return _lookup(self.id_sorted, self.id_type.astype(np.int64), ids)


# ---------------------------------------------------------------------------------- HBT side
def hbt_files(path, snap):
    """SubSnap files of snapshot `snap` under an HBT SubhaloPath, a NNN/ directory, or a file."""
    if os.path.isfile(path):
        m = re.match(r"(.*SubSnap_\d+)\.\d+\.hdf5$", path)
        return sorted(glob.glob(m.group(1) + ".*.hdf5"), key=_fileno) if m else [path]
    cands = [path, os.path.join(path, "%03d" % snap), os.path.join(path, "%04d" % snap)]
    for d in cands:
        files = glob.glob(os.path.join(d, "SubSnap_%03d.*.hdf5" % snap)) + glob.glob(os.path.join(d, "SubSnap_%03d.hdf5" % snap))
        if files:
            return sorted(files, key=_fileno)
    raise InputError("no SubSnap_%03d files under %s" % (snap, path))


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
                subs.append(f["Subhalos"][:])
                parts.extend(list(f["SubhaloParticles"][:]))
                if "Cosmology/ScaleFactor" in f:
                    self.scale_factor = float(f["Cosmology/ScaleFactor"][0])
        self.subs = np.concatenate(subs) if subs else np.zeros(0)
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

    def owner(self, ids):
        return _lookup(self.ids_sorted, self.owner_sorted, ids)


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


def _q(a, qs=(10, 50, 90)):
    if a.size == 0:
        return "n/a"
    return "/".join("%.3f" % np.percentile(a, q) for q in qs)


def report(ours, hbt, M, args, out=sys.stdout):
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
    for lab, sel in (("all", np.ones(M.h.size, bool)), (">= %d" % args.min_np, isbig)):
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
    if args.pairs_out:
        with open(args.pairs_out, "w") as f:
            f.write("# HBT_TrackId HBT_Rank HBT_Depth HBT_Nbound our_TrackId our_Rank our_Depth our_Nbound "
                    "intersection Jaccard Nbound_ratio Vmax_ratio class\n")
            for k in range(M.h.size):
                h, oo = M.h[k], M.o[k]
                c = "ok" if good[k] else classify_pair(ours, hbt, h, oo, sample, core, spawned).replace(" ", "_")
                f.write("%d %d %d %d %d %d %d %d %d %.5f %.5f %s %s\n" % (
                    hbt.trackid[h], hbt.rank[h], hbt.subs["Depth"][h], hbt.nbound[h], ours.tid[oo], ours.rank[oo],
                    ours.S["Depth"][oo], ours.nbound[oo], M.inter[k], M.jac[k], ratio[k],
                    "%.5f" % vr[k] if np.isfinite(vr[k]) else "nan", c))
        print("  pairs written to %s" % args.pairs_out, file=out)
    return frac


def load(pigdir, hbtpath, args):
    ours = Ours(pigdir, args.ntracers, phase1=args.phase1)
    hbt = HBT(hbtpath, ours.snap + args.hbt_snap_offset)
    return ours, hbt


def track_pairs(args):
    A, HA = load(args.track_pairs[0], args.paths[0], args)
    B, HB = load(args.track_pairs[1], args.paths[0], args)
    MA = match(A, HA, args.match, args.ntracers)
    MB = match(B, HB, args.match, args.ntracers)
    print("== %s" % args.track_pairs[0])
    report(A, HA, MA, args)
    print("== %s" % args.track_pairs[1])
    fracB = report(B, HB, MB, args)
    # pairs at the first output, by TrackIds
    nxt = dict(zip(B.tid[MB.o].tolist(), HB.trackid[MB.h].tolist()))
    hb_alive = dict(zip(HB.trackid.tolist(), HB.alive.tolist()))
    big = HA.nbound[MA.h] >= args.min_np
    cnt = {"persist": 0, "both dead or sunk": 0, "only ours dead or sunk": 0, "only HBT's dead or sunk": 0,
           "ours unmatched": 0, "other partner": 0}
    examples = []
    for k in range(MA.h.size):
        if args.min_np and not big[k]:
            continue
        to, th = int(A.tid[MA.o[k]]), int(HA.trackid[MA.h[k]])
        o_alive = B.row([to])[0] >= 0
        h_alive = bool(hb_alive.get(th, False))
        if not o_alive and not h_alive:
            cnt["both dead or sunk"] += 1
        elif not o_alive:
            cnt["only ours dead or sunk"] += 1
        elif not h_alive:
            cnt["only HBT's dead or sunk"] += 1
        elif to not in nxt:
            cnt["ours unmatched"] += 1
            examples.append((to, th, "unmatched"))
        elif nxt[to] != th:
            cnt["other partner"] += 1
            examples.append((to, th, "now HBT %d" % nxt[to]))
        else:
            cnt["persist"] += 1
    both = cnt["persist"] + cnt["ours unmatched"] + cnt["other partner"]
    pf = cnt["persist"] / float(max(both, 1))
    print("== TrackId persistence %s -> %s (pairs with HBT Nbound >= %d at the first output)" % (
        args.track_pairs[0], args.track_pairs[1], args.min_np))
    print("  %s" % ", ".join("%s %d" % kv for kv in cnt.items()))
    print("  persisting: %d of %d pairs whose tracks are both still resolved: %.1f%% (target %.0f%%)"
          % (cnt["persist"], both, 100 * pf, 100 * args.persist_frac))
    for to, th, why in examples[:args.show or 10]:
        print("    our %d / HBT %d: %s" % (to, th, why))
    ok = fracB >= args.accept_frac and pf >= args.persist_frac
    return 0 if (ok or args.report_only) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="PIG_xxx HBT (one snapshot), or HBT alone with --track-pairs")
    p.add_argument("--track-pairs", nargs=2, metavar=("PIG_PREV", "PIG_NEXT"),
                   help="compare both snapshots and check that matched TrackId pairs persist")
    p.add_argument("--match", choices=["tracers", "jaccard"], default="tracers")
    p.add_argument("--ntracers", type=int, default=10, help="tracers per subhalo for matching (default 10)")
    p.add_argument("--min-np", type=int, default=100, help="HBT Nbound threshold of the summary (default 100)")
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
    args = p.parse_args(argv)
    try:
        if args.track_pairs:
            if len(args.paths) != 1:
                p.error("with --track-pairs give only the HBT path")
            if args.pairs_out:
                p.error("--pairs-out is for one snapshot")
            return track_pairs(args)
        if len(args.paths) != 2:
            p.error("give PIG_xxx and the HBT path")
        ours, hbt = load(args.paths[0], args.paths[1], args)
        M = match(ours, hbt, args.match, args.ntracers)
        frac = report(ours, hbt, M, args)
        return 0 if (frac >= args.accept_frac or args.report_only) else 1
    except (SubhaloFormatError, InputError, IOError, OSError) as e:
        print("error: %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
