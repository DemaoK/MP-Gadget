#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How well the finder's marker particles survive between two outputs, and whether other rules
would decide hosts and descendants differently (CDM vs SIDM study).

Usage:
  subhalo_tracer_study.py PIG_A PIG_B [--part PART_A] [--rows OUT.npz] [options]
  subhalo_tracer_study.py --sequence OUTPUT_DIR [--first N --last M]   (one line per pair)

Markers of a track resolved in A (the finder's own definitions):
  tracers  the first --ntracers (10) tracer-type rows with TrackId == BoundTrackId == the track,
           in (BindRank, ID) order, as subhalo_host_vote() takes them; ranks j = 0..9.
  core     the first --ncore (20) own-bound rows, tracer types first, then the others.
  bound    the full bound set: every row with BoundTrackId == the track (own tag first, then the
           overlap rows of nested children), mass-weighted; "rank" is the position in that order.

Sections (each stratified by Nbound in A, 20-100 / 100-1000 / >1000, and central / satellite):
  1 fate     per marker set, % of the markers (bound: of the mass) that at B are bound to the
             same track / bound to another track / saved but unbound / absent (in no group and
             untagged, so not stored). For tracks with tracers bound elsewhere, how often the
             plurality holder is the A central, the A nested parent or the B DescendantTrackId.
  2 host     tracks resolved in A and B. "sanity" = % where HBT's vote (weight 1/(1+sqrt j)) of the
             tracers' B GroupID does not reproduce B's HostGroup (expect 0). Rules: a = the first
             --nwide (50) tracers, same weights; b = the 10 tracers minus kicked ones (B BindRank
             > --kick-rank, unbound or absent; all kicked -> the 10-tracer vote; "kicked" = % of
             tracers kicked); c = the whole bound set weighted by 1/(1+sqrt rank). "diff" = % of
             tracks where the rule's group differs from HostGroup, "alt/fin" = in those, % where
             the mass-plurality group (the B group holding most of the A bound mass, -1 = no group)
             is the rule's / the finder's.
             "agree" = % agreement of finder, a, b, c with the mass-plurality group.
  3 deaths   tracks resolved in A that die at B (Orphan row with SnapDeath == B). Agreement of
             DescendantTrackId with a = tracer plurality of B BoundTrackId (the finder's rule, ties
             to the smaller id; expect ~100), b = rank-weighted bound-set plurality, c = holder of
             MostBoundID. Margin = (top - second tracer count) / ntracers.
  4 centre   tracks resolved in A and B: |X_B - X_A| (raw) and |X_B - X_A - CoMVel_A * D| (drift)
             for X = MostBoundPos, CorePos, CoMPos, in units of A's CoreSigmaX; median and 90th
             percentile. D = int da / (a^2 H) in comoving length per peculiar km/s (catalogue
             velocities are physical peculiar km/s, positions comoving), LCDM from the Header, no
             radiation (a 0.2 % error at a = 0.13).
  5 SIDM     with --part (A's PART holding 1/SIDMSupport, the kernel support h): expected number of
             scatterings per particle over the interval, N = rho sigma0/m f((v/w)^2) v dt with
             rho = m Nngb / (4 pi h^3 / 3) (physical), v = the track's CoreSigmaV, dt proper time,
             f(y) = 6 int_0^1 t(1-t)/(1+y t)^2 dt (f(0) = 1). Tracer retention (bound to the same
             track) in N bins and median N of tracers / core / bound set. Skipped without --part.

--rows writes the per-track table (one row per track resolved in A) to an .npz file.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import bigfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloCatalogue, SubhaloSequence, NO_GROUP_U4  # noqa: E402

CM_PER_MPC = 3.085678e24
NBINS = [(20, 100, "20-100"), (100, 1000, "100-1e3"), (1000, np.inf, ">1e3")]
NOBALLOT = -2
FATES = ("same", "other", "unbnd", "absent")


def run_position(key):
    """Position of each entry inside its run of equal values (key already grouped)."""
    if key.size == 0:
        return np.zeros(0, np.int64)
    start = np.ones(key.size, bool)
    start[1:] = key[1:] != key[:-1]
    idx = np.arange(key.size)
    return idx - np.maximum.accumulate(np.where(start, idx, 0))


def vote(voter, cand, weight, nvoter, skip_negative=False):
    """Per voter 0..nvoter-1: the candidate with the largest summed weight (ties to the smaller, as
    the finder), its sum and the runner-up's sum. NOBALLOT where a voter has no ballot."""
    win = np.full(nvoter, NOBALLOT, np.int64)
    top, second = np.zeros(nvoter), np.zeros(nvoter)
    if skip_negative:
        m = cand >= 0
        voter, cand, weight = voter[m], cand[m], weight[m]
    if voter.size == 0:
        return win, top, second
    span = int(cand.max()) + 2
    key = voter.astype(np.int64) * span + (cand.astype(np.int64) + 1)
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv.ravel(), weights=weight, minlength=uk.size)  # sums in input (rank) order
    kv, kc = uk // span, uk % span - 1
    o = np.lexsort((kc, -w, kv))
    kv, kc, w = kv[o], kc[o], w[o]
    pos = run_position(kv)
    first = pos == 0
    win[kv[first]], top[kv[first]] = kc[first], w[first]
    sec = pos == 1
    second[kv[sec]] = w[sec]
    return win, top, second


def cosmo_integrals(h, a0, a1):
    """(D, dt): comoving drift per km/s in internal length units and proper time in s, a0 -> a1."""
    om = float(np.atleast_1d(h.get("Omega0", 0.3))[0])
    ol = float(np.atleast_1d(h.get("OmegaLambda", 1 - om))[0])
    hub = float(np.atleast_1d(h.get("HubbleParam", 0.7))[0])
    ulen = float(np.atleast_1d(h.get("UnitLength_in_cm", 3.085678e21))[0])
    x, w = np.polynomial.legendre.leggauss(32)
    a = 0.5 * (a1 - a0) * x + 0.5 * (a1 + a0)
    E = np.sqrt(om / a ** 3 + (1 - om - ol) / a ** 2 + ol)
    half = 0.5 * (a1 - a0)
    drift = np.sum(w / (a ** 2 * E)) * half / 100.0 * CM_PER_MPC / ulen   # (Mpc/h per km/s) -> internal
    dt = np.sum(w / (a * E)) * half * CM_PER_MPC / (1e7 * hub)
    return drift, dt


def fshape(y):
    """6 int_0^1 t(1-t)/(1+y t)^2 dt, the normalised viscosity cross-section shape (f(0) = 1)."""
    x, w = np.polynomial.legendre.leggauss(24)
    t = 0.5 * (x + 1)
    y = np.asarray(y, np.float64)[..., None]
    return 3.0 * np.sum(w * t * (1 - t) / (1 + y * t) ** 2, axis=-1)


def read_particles(cat, names):
    """Concatenated per-particle blocks over the tagged types, plus 'type'."""
    out = {n: [] for n in names + ["type"]}
    for t in cat.tagged_types():
        for n in names:
            out[n].append(cat.read(t, n))
        out["type"].append(np.full(cat.nsaved(t), t, np.int8))
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


class Earlier(object):
    """Per-track marker lists of catalogue A; no particle blocks are kept."""

    def __init__(self, cat, P, args):
        S = cat.subhalos
        self.snap, self.a, self.header = cat.snapshot, cat.scale_factor, cat.header
        self.tid = S["TrackId"].astype(np.int64)
        for k in ("Nbound", "Rank", "CentralTrackId", "NestedParentTrackId", "MostBoundID", "CoreSigmaX",
                  "CoreSigmaV", "CoMPos", "CoMVel", "CorePos", "MostBoundPos"):
            setattr(self, k, S[k])
        bt = P["BoundTrackId"].astype(np.int64)
        sel = np.nonzero(bt >= 0)[0]
        sel = sel[np.isin(bt[sel], self.tid)]
        row = np.searchsorted(self.tid, bt[sel])
        own = P["TrackId"][sel].astype(np.int64) == bt[sel]
        br, ids, typ = P["BindRank"][sel].astype(np.int64), P["ID"][sel].astype(np.int64), P["type"][sel]
        tracer = np.isin(typ, cat.tracer_types())
        # full bound set: own tag in BindRank order, then the children's overlap rows
        o = np.lexsort((ids, br, ~own, row))
        self.fb_row, self.fb_id, self.fb_type = row[o].astype(np.int32), ids[o], typ[o]
        self.fb_mass = P["Mass"][sel][o].astype(np.float64)
        self.fb_rank = run_position(self.fb_row)
        # tracers: own-bound tracer types in (BindRank, ID) order; keep --nwide for rule a
        m = own & tracer
        o = np.lexsort((ids[m], br[m], row[m]))
        r, i = row[m][o], ids[m][o]
        j = run_position(r)
        keep = j < max(args.nwide, args.ntracers)
        self.tw_row, self.tw_id, self.tw_j = r[keep].astype(np.int32), i[keep], j[keep]
        # core: own-bound, tracer types first
        o = np.lexsort((ids[own], br[own], ~tracer[own], row[own]))
        r, i = row[own][o], ids[own][o]
        keep = run_position(r) < args.ncore
        self.core_row, self.core_id = r[keep].astype(np.int32), i[keep]


class Later(object):
    """Catalogue B as a sorted-ID index of BoundTrackId, BindRank and GroupID, plus its tables."""

    def __init__(self, cat, P):
        self.snap, self.a = cat.snapshot, cat.scale_factor
        S, O = cat.subhalos, cat.orphans
        self.tid, self.host = S["TrackId"].astype(np.int64), S["HostGroup"].astype(np.int64)
        self.S = {k: S[k] for k in ("CoMPos", "CorePos", "MostBoundPos")}
        self.desc = S["DescendantTrackId"].astype(np.int64)
        self.otid = O["TrackId"].astype(np.int64) if cat.norphans else np.zeros(0, np.int64)
        self.odesc = O["DescendantTrackId"].astype(np.int64) if cat.norphans else self.otid
        self.odeath = O["SnapDeath"] if cat.norphans else np.zeros(0, np.int32)
        ids = P["ID"].astype(np.int64)
        o = np.argsort(ids, kind="stable")
        self.ids = ids[o]
        self.bt = P["BoundTrackId"].astype(np.int64)[o]
        self.br = P["BindRank"].astype(np.int64)[o]
        g = P["GroupID"][o]
        self.gid = np.where(g == NO_GROUP_U4, -1, g.astype(np.int64))

    def look(self, q):
        """(present, BoundTrackId, BindRank, GroupID) of particle IDs q; -1 / huge / -1 if absent."""
        pos = np.minimum(np.searchsorted(self.ids, q), max(self.ids.size - 1, 0))
        pres = self.ids[pos] == q if self.ids.size else np.zeros(q.size, bool)
        bt = np.where(pres, self.bt[pos], -1)
        br = np.where(pres, self.br[pos], np.iinfo(np.int64).max)
        gid = np.where(pres, self.gid[pos], -1)
        return pres, bt, br, gid

    def row(self, tid, which="sub"):
        keys = self.tid if which == "sub" else self.otid
        pos = np.minimum(np.searchsorted(keys, tid), max(keys.size - 1, 0))
        return np.where((keys.size > 0) & (keys[pos] == tid), pos, -1)


def fate(trackid, pres, bt):
    return np.select([bt == trackid, bt >= 0, pres], [0, 1, 2], 3)


def strata(nb, rank):
    out = [("all", np.ones(nb.size, bool))]
    for lo, hi, lab in NBINS:
        m = (nb >= lo) & (nb < hi)
        out += [(lab + " cen", m & (rank == 0)), (lab + " sat", m & (rank != 0))]
    return out


def pct(num, den):
    return 100.0 * num / den if den > 0 else np.nan


def periodic(d, box):
    return d - box * np.round(d / box)


def sidm_part(A, part, args):
    """Kernel support h of every bound particle of A from PART/1/SIDMSupport (None if absent)."""
    try:
        f = bigfile.File(part)
        if "1/SIDMSupport" not in f.blocks or "1/ID" not in f.blocks:
            return None
    except Exception:
        return None
    m1 = A.fb_type == 1
    q, first = np.unique(A.fb_id[m1], return_index=True)
    mq, hq = A.fb_mass[m1][first], np.full(q.size, np.nan)
    bid, bh = f["1/ID"], f["1/SIDMSupport"]
    step = 1 << 24
    for s in range(0, bid.size, step):  # chunked: PART is the whole box
        ids = bid[s:s + step].astype(np.int64)
        pos = np.minimum(np.searchsorted(q, ids), max(q.size - 1, 0))
        hit = q[pos] == ids
        hq[pos[hit]] = bh[s:s + step][hit]
    if abs(float(np.atleast_1d(f["Header"].attrs["Time"])[0]) - A.a) > 1e-6 * A.a:
        print("warning: %s is not at the scale factor of PIG_A" % part, file=sys.stderr)
    return q, hq, mq


def study(A, B, args, part=None, out=sys.stdout, quiet=False):
    n = A.tid.size
    nb, rank = A.Nbound.astype(np.int64), A.Rank
    R = {"TrackId": A.tid, "Nbound": nb, "Rank": rank}
    # 1. fates of tracers, core and bound set
    ten = A.tw_j < args.ntracers
    tr_row, tr_id, tr_j = A.tw_row[ten], A.tw_id[ten], A.tw_j[ten]
    tpres, tbt, tbr, tgid = B.look(tr_id)
    tfate = fate(A.tid[tr_row], tpres, tbt)
    R["tracer"] = np.bincount(tr_row * 4 + tfate, minlength=4 * n).reshape(n, 4)
    cpres, cbt = B.look(A.core_id)[:2]
    cfate = fate(A.tid[A.core_row], cpres, cbt)
    R["core"] = np.bincount(A.core_row * 4 + cfate, minlength=4 * n).reshape(n, 4)
    fpres, fbt, _, fgid = B.look(A.fb_id)
    ffate = fate(A.tid[A.fb_row], fpres, fbt)
    msum = np.bincount(A.fb_row * 4 + ffate, weights=A.fb_mass, minlength=4 * n).reshape(n, 4)
    R["bound_mass"] = msum
    oth = tfate == 1
    R["other_holder"] = vote(tr_row[oth], tbt[oth], np.ones(oth.sum()), n)[0]
    brow_any = B.row(A.tid)
    orow_any = B.row(A.tid, "orphan")
    R["desc_B"] = np.where(brow_any >= 0, B.desc[np.maximum(brow_any, 0)], -1)
    if B.otid.size:
        R["desc_B"] = np.where(orow_any >= 0, B.odesc[np.maximum(orow_any, 0)], R["desc_B"])
    # 2. host votes over B GroupID; the mass-plurality group of the A bound set
    w = 1.0 / (1.0 + np.sqrt(A.tw_j))
    R["host_B"] = np.where(brow_any >= 0, B.host[np.maximum(brow_any, 0)], NOBALLOT)
    R["vote10"] = vote(tr_row, tgid, w[ten], n)[0]
    wgid = B.look(A.tw_id)[3]
    R["vote_a"] = vote(A.tw_row, wgid, w, n)[0]
    kicked = (tbr > args.kick_rank) | (tbt < 0)
    vb = vote(tr_row[~kicked], tgid[~kicked], w[ten][~kicked], n)[0]
    R["nkicked"] = np.bincount(tr_row[kicked], minlength=n)
    R["vote_b"] = np.where(vb == NOBALLOT, R["vote10"], vb)
    R["vote_c"] = vote(A.fb_row, fgid, 1.0 / (1.0 + np.sqrt(A.fb_rank)), n)[0]
    R["mass_group"] = vote(A.fb_row, fgid, A.fb_mass, n)[0]
    for k in ("vote10", "vote_a", "vote_b", "vote_c"):   # no ballot: the finder keeps Host = -1
        R[k] = np.where(R[k] == NOBALLOT, -1, R[k])
    # 3. death descendants
    R["dies"] = (brow_any < 0) & (orow_any >= 0)
    if B.otid.size:
        R["dies"] &= B.odeath[np.maximum(orow_any, 0)] == B.snap
    R["desc_a"], top, sec = vote(tr_row, tbt, np.ones(tr_row.size), n, skip_negative=True)
    R["margin"] = (top - sec) / float(args.ntracers)
    R["desc_b"] = vote(A.fb_row, fbt, 1.0 / (1.0 + np.sqrt(A.fb_rank)), n, skip_negative=True)[0]
    R["desc_c"] = B.look(A.MostBoundID.astype(np.int64))[1]
    for k in ("desc_a", "desc_b"):
        R[k] = np.where(R[k] == NOBALLOT, -1, R[k])
    # 4. centre continuity
    drift, dt = cosmo_integrals(A.header, A.a, B.a)
    box = float(np.atleast_1d(A.header.get("BoxSize", np.inf))[0])
    both = brow_any >= 0
    br_ = np.maximum(brow_any, 0)
    sig = np.where(A.CoreSigmaX > 0, A.CoreSigmaX, np.nan)
    for k in ("MostBoundPos", "CorePos", "CoMPos"):
        d = periodic(B.S[k][br_] - getattr(A, k), box)
        R["raw_" + k] = np.where(both, np.linalg.norm(d, axis=1) / sig, np.nan)
        dp = periodic(d - A.CoMVel.astype(np.float64) * drift, box)
        R["res_" + k] = np.where(both, np.linalg.norm(dp, axis=1) / sig, np.nan)
    # 5. SIDM scattering proxy
    if part is not None:
        q, hq, mq = part
        hub, ul, um = (float(np.atleast_1d(A.header[k])[0]) for k in ("HubbleParam", "UnitLength_in_cm", "UnitMass_in_g"))

        def rho_of(ids):   # physical g/cm^3 from the kernel support; nan where PART has no entry
            if q.size == 0:
                return np.full(ids.size, np.nan)
            pos = np.minimum(np.searchsorted(q, ids), q.size - 1)
            hit = q[pos] == ids
            hh = np.where(hit, hq[pos] * A.a * ul / hub, np.nan)
            return np.where(hit, mq[pos], np.nan) * um / hub * args.sidm_neighbors / (4 * np.pi / 3 * hh ** 3)
        v = A.CoreSigmaV.astype(np.float64)
        perv = args.sigma0 * fshape((v / args.wturn) ** 2) * v * 1e5 * dt   # N per unit rho, per track
        tN, cN, fbN = (rho_of(i) * perv[r] for i, r in ((tr_id, tr_row), (A.core_id, A.core_row), (A.fb_id, A.fb_row)))
        R["N_tracer"] = np.full((n, args.ntracers), np.nan)
        R["N_tracer"][tr_row, tr_j] = tN
        for lab, x, r in (("N_core_mean", cN, A.core_row), ("N_bound_mean", fbN, A.fb_row)):
            ok = np.isfinite(x)
            R[lab] = np.bincount(r[ok], weights=x[ok], minlength=n) / np.bincount(r[ok], minlength=n)
    else:
        tN = cN = fbN = None
    if args.rows:
        np.savez(args.rows, **R)
    S = summary_numbers(R, both)
    if quiet:
        return S
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p("== %s -> %s  (snap %d -> %d, a %.6f -> %.6f, dt %.4g Myr)  tracks resolved in A: %d, in both: %d, dying: %d"
      % (args.pig_a, args.pig_b, A.snap, B.snap, A.a, B.a, dt / 3.15576e13, n, both.sum(), R["dies"].sum()))
    p("\n1 fate at B, % of markers (bound set: of mass): same / other track / unbound / absent")
    p("%-12s %7s | %-27s | %-27s | %-27s" % ("stratum", "Ntrk", "tracers", "core", "bound mass"))
    for lab, m in strata(nb, rank):
        cols = []
        for key in ("tracer", "core", "mass"):
            c = R[key if key != "mass" else "bound_mass"][m].sum(0)
            cols.append(" ".join("%6.2f" % pct(x, c.sum()) for x in c))
        p("%-12s %7d | %s | %s | %s" % (lab, m.sum(), cols[0], cols[1], cols[2]))
    p("\n1(ii) tracks with tracers bound to another track: % whose plurality holder is A central / A parent / B descendant")
    for lab, m in strata(nb, rank):
        mm = m & (R["tracer"][:, 1] > 0)
        hold = R["other_holder"][mm]
        p("%-12s %7d   %6.2f %6.2f %6.2f" % (lab, mm.sum(), pct(np.sum(hold == A.CentralTrackId[mm]), mm.sum()),
          pct(np.sum(hold == A.NestedParentTrackId[mm]), mm.sum()), pct(np.sum(hold == R["desc_B"][mm]), mm.sum())))
    p("\n2 host vote at B (tracks in both): sanity = % vote10 != HostGroup; per rule diff% (alt/fin); agree with mass plurality")
    p("%-12s %7s %7s %7s | %-21s| %-21s| %-21s| %s" % ("stratum", "N", "sanity", "kicked", "a (50 tracers)",
      "b (no kicked)", "c (bound set)", "agree fin   a     b     c"))
    for lab, m in strata(nb, rank):
        m = m & both
        hb, mg = R["host_B"][m], R["mass_group"][m]
        cols = []
        for r in ("vote_a", "vote_b", "vote_c"):
            v = R[r][m]
            d = v != hb
            cols.append("%7.3f (%5.1f/%5.1f)" % (pct(d.sum(), m.sum()), pct(np.sum(mg[d] == v[d]), d.sum()),
                                                pct(np.sum(mg[d] == hb[d]), d.sum())))
        ag = [pct(np.sum(x == mg), m.sum()) for x in (hb, R["vote_a"][m], R["vote_b"][m], R["vote_c"][m])]
        p("%-12s %7d %7.3f %7.3f | %s| %s| %s| %s" % (lab, m.sum(), pct(np.sum(R["vote10"][m] != hb), m.sum()),
          pct(R["nkicked"][m].sum(), R["tracer"][m].sum()), cols[0], cols[1], cols[2], " ".join("%5.1f" % x for x in ag)))
    p("\n3 deaths at B: % of DescendantTrackId equal to a (tracer plurality) / b (bound set) / c (most-bound holder)")
    p("%-12s %7s   %6s %6s %6s   %s" % ("stratum", "Ndeath", "a", "b", "c", "desc=-1  margin q10/q50/q90"))
    dd = R["desc_B"]
    for lab, m in strata(nb, rank):
        m = m & R["dies"]
        mq = np.percentile(R["margin"][m], [10, 50, 90]) if m.any() else [np.nan] * 3
        p("%-12s %7d   %6.2f %6.2f %6.2f   %6.2f   %.1f/%.1f/%.1f" % (
            lab, m.sum(), *[pct(np.sum(R[k][m] == dd[m]), m.sum()) for k in ("desc_a", "desc_b", "desc_c")],
            pct(np.sum(dd[m] < 0), m.sum()), *mq))
    hist = np.bincount(np.rint(R["margin"][R["dies"]] * args.ntracers).astype(int), minlength=args.ntracers + 1)
    p("  tracer-vote margin (count top - second) histogram 0..%d: %s" % (args.ntracers, " ".join(map(str, hist))))
    p("\n4 centre displacement / CoreSigmaX_A (tracks in both), median / 90th pct; drift = CoMVel_A * D, D = %.4g" % drift)
    p("%-12s %7s | %-17s %-17s %-17s| %-17s %-17s %-17s" % ("Nbound", "N", "raw MostBound", "raw Core", "raw CoM",
      "drift MostBound", "drift Core", "drift CoM"))
    for lo, hi, lab in NBINS:
        m = both & (nb >= lo) & (nb < hi) & np.isfinite(sig)
        q = []
        for k in ("raw_", "res_"):
            for c in ("MostBoundPos", "CorePos", "CoMPos"):
                q.append("%7.3f / %7.3f" % tuple(np.percentile(R[k + c][m], [50, 90])) if m.any() else "   -")
        p("%-12s %7d | %s %s %s| %s %s %s" % (lab, m.sum(), *q))
    if tN is not None:
        p("\n5 SIDM proxy: sigma0/m %g cm^2/g, w %g km/s, Nngb %d; tracer retention (same track) by expected scatterings N"
          % (args.sigma0, args.wturn, args.sidm_neighbors))
        edges = [0, 0.03, 0.1, 0.3, 1, np.inf]
        same = tfate == 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (tN >= lo) & (tN < hi)
            p("  N in [%5g, %5g)  tracers %8d  retained %6.2f %%" % (lo, hi, m.sum(), pct(np.sum(same[m]), m.sum())))
        p("  no SIDMSupport (not found / not type 1): %d tracers" % np.sum(~np.isfinite(tN)))
        p("  median N: %-12s %10s %10s %10s" % ("Nbound", "tracers", "core", "bound"))
        for lo, hi, lab in NBINS:
            mt, mc, mf = [(nb[r] >= lo) & (nb[r] < hi) for r in (tr_row, A.core_row, A.fb_row)]
            p("            %-12s %10.4g %10.4g %10.4g" % (lab, *[np.nanmedian(x[mm]) if np.any(mm) else np.nan
                                                             for x, mm in ((tN, mt), (cN, mc), (fbN, mf))]))
    return S


def summary_numbers(R, both):
    """The sequence-mode figures over all tracks."""
    t, c = R["tracer"].sum(0), R["core"].sum(0)
    msum = R["bound_mass"].sum(0)
    hb = R["host_B"][both]
    v10 = R["vote10"][both]
    d = R["dies"]
    out = dict(tr=pct(t[0], t.sum()), core=pct(c[0], c.sum()), mass=pct(msum[0], msum.sum()),
               tr100=pct(R["tracer"][R["Nbound"] >= 100].sum(0)[0], R["tracer"][R["Nbound"] >= 100].sum()),
               sanity=pct(np.sum(v10 != hb), both.sum()), ndeath=int(d.sum()))
    for r in "abc":
        v = R["vote_" + r][both]
        out["host_" + r] = pct(np.sum(v != hb), both.sum())
        out["desc_" + r] = pct(np.sum(R["desc_" + r][d] == R["desc_B"][d]), d.sum())
    return out


def load(cat, earlier, later, args):
    """Earlier/Later views of one catalogue; its particle blocks are read once and released."""
    names = ["ID", "BoundTrackId", "BindRank"] + (["TrackId", "Mass"] if earlier else []) + (["GroupID"] if later else [])
    P = read_particles(cat, names)
    cat.clear_cache()
    out = (Earlier(cat, P, args) if earlier else None, Later(cat, P) if later else None)
    del P
    return out


def sequence(args):
    seq = SubhaloSequence(args.sequence)
    idx = [i for i, s in enumerate(seq.snapshots) if args.first <= s <= (args.last if args.last >= 0 else s)]
    print("# %s: tr/core/mass = %% retained by the same track (all tracks), tr100 = tracers of Nbound >= 100;"
          " sanity = %% vote10 != HostGroup; host a/b/c = %% rule != HostGroup; desc a/b/c = %% agreeing with"
          " DescendantTrackId" % args.sequence)
    print("%-9s %9s %7s %7s %7s %7s %7s | %6s %6s %6s | %6s %6s %6s %6s" % (
        "pair", "a_B", "tr", "tr100", "core", "mass", "sanity", "host_a", "host_b", "host_c", "ndeath", "desc_a",
        "desc_b", "desc_c"))
    A = None
    for k, i in enumerate(idx):
        cat = SubhaloCatalogue(seq.paths[i])   # opened per step: past tables are not kept
        E, L = load(cat, k + 1 < len(idx), A is not None, args)
        if A is not None:
            args.pig_a, args.pig_b = seq.paths[idx[k - 1]], seq.paths[i]
            S = study(A, L, args, quiet=True)
            print("%03d->%03d %9.6f %7.2f %7.2f %7.2f %7.2f %7.3f | %6.2f %6.2f %6.2f | %6d %6.2f %6.2f %6.2f" % (
                A.snap, L.snap, L.a, S["tr"], S["tr100"], S["core"], S["mass"], S["sanity"], S["host_a"],
                S["host_b"], S["host_c"], S["ndeath"], S["desc_a"], S["desc_b"], S["desc_c"]))
            sys.stdout.flush()
        A, L = E, None
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pig_a", nargs="?")
    ap.add_argument("pig_b", nargs="?")
    ap.add_argument("--sequence", help="output directory: every consecutive pair of PIG_*")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=-1)
    ap.add_argument("--part", help="PART snapshot of A with 1/SIDMSupport (section 5)")
    ap.add_argument("--sigma0", type=float, default=147.1, help="sigma0/m in cm^2/g")
    ap.add_argument("--wturn", type=float, default=120.0, help="velocity scale w in km/s")
    ap.add_argument("--sidm-neighbors", type=int, default=38)
    ap.add_argument("--ntracers", type=int, default=10)
    ap.add_argument("--nwide", type=int, default=50, help="tracers of host rule a")
    ap.add_argument("--ncore", type=int, default=20)
    ap.add_argument("--kick-rank", type=int, default=100, help="host rule b: B BindRank above this is kicked")
    ap.add_argument("--rows", help="write the per-track table to this .npz")
    args = ap.parse_args(argv)
    if args.sequence:
        args.rows, args.part = None, None
        return sequence(args)
    if not (args.pig_a and args.pig_b):
        ap.error("give PIG_A PIG_B or --sequence")
    A = load(SubhaloCatalogue(args.pig_a), True, False, args)[0]
    part = sidm_part(A, args.part, args) if args.part else None
    B = load(SubhaloCatalogue(args.pig_b), False, True, args)[1]
    study(A, B, args, part=part)
    return 0


if __name__ == "__main__":
    sys.exit(main())
