#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How long the finder's marker particles, tagged when a satellite falls in, stay bound to it over
many outputs, where they go, and whether the track itself survives (CDM vs SIDM study).

Usage:
  subhalo_tracer_survival.py --sequence DIR [--first N --last M] [--tag infall|all] [--nmin 100]
                             [--maxj 40] [--density knn] [--rows OUT.npz] [--ntracers 10 --nwide 50 --ncore 20]
                             [--sigma0 147.1 --wturn 120 --sidm-neighbors 38]

Catalogues are read one at a time (subhalo_tracer_study.load: the Earlier and Later views of one
catalogue); only the marker IDs of the open cohort entries are carried from output to output.

Cohort entries, at output k of the walk: tracks resolved at k with Rank > 0 and Nbound >= --nmin,
  infall  (default) and resolved with Rank == 0 at the previous output of the sequence (for the
          first output of the range that output's (TrackId, Rank) is read from its tables alone;
          none if the range starts at the sequence's first output). A track falling in again opens
          a new entry. Tracks born as satellites never enter.
  all     every such satellite at every output (one entry per track and output).
Markers (the study tool's definitions at k): ten = the finder's --ntracers tracers (own-bound tracer
types in (BindRank, ID) order), wide = the first --nwide of that order, core = the --ncore most-bound
own-bound rows, tracer types first. j = outputs since tagging (outputs are equally spaced in ln a).

At k' = k + j, j = 1 .. --maxj, each marker is
  same    BoundTrackId == the track
  host    BoundTrackId == the track's CentralTrackId at k' (resolved), else its DescendantTrackId at
          k' (orphan) when >= 0
  other   bound to any other track;  unbnd  stored but unbound;  absent  not stored at k'
and the track is SAT (resolved, Rank > 0), CEN (resolved, Rank == 0), SUNK (orphan, SinkTrackId >= 0),
DEAD (orphan, not sunk) or ABSENT (in neither table; should not happen). overlap = how many of the
original ten are among the finder's current ten of the track (0 if not resolved). j = 0 is the tag
output: all markers "same" by construction. Beyond the sequence end everything is censored (-1 / NaN).

Report (stdout), per stratum of Nbound at tagging (bins of the study tool from --nmin up, and all):
  1 markers  for j = 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40: entries defined at j, the mean over
             entries of the % of each entry's ten in each fate, same / host % of the core and the
             wide set, and the mean overlap % of the ten (over entries resolved at j).
  2 tracks   % of the defined entries SAT / CEN / SUNK / DEAD at j; over the alive (SAT or CEN) ones,
             % keeping at least half of their ten "same" and % with none "same" (fully re-tagged).
  3 half-life Kaplan-Meier: event = the first j at which fewer than half of the entry's ten are
             "same" (< 5 of 10) while the track is alive (SAT or CEN). An entry is censored at the
             last j it was alive with at least half "same" when at the next j the track is SUNK,
             DEAD or ABSENT (a disrupted track loses its tracers by definition; its fate is section
             2's) or the sequence ends. Median j = the first j with S(j) <= 0.5, "> J" if S stays
             above 0.5 up to J, the last j with entries at risk; S at j = 4, 8, 16, 32.
  4 SIDM     with --density knn: the expected scatterings over one output, N_tag, of each tracer at
             tagging (section 5 of subhalo_tracer_study.py: knn support over the stored particles,
             v = CoreSigmaV of the track, dt to the next output of the sequence; NaN at its last
             output), and the % of tracers still "same" at j = 1, 2, 4, 8, 16 by N_tag bin.
Entries with fewer markers than requested use their own count as the denominator; those with no
tracer are left out of the tracer figures.

--rows writes per entry: TrackId, snap_tag, a_tag, Nbound_tag, Rank_tag, host_tag, n_ten, n_wide,
n_core, N_tag [n, ntracers]; per (entry, j): fate_ten / fate_wide / fate_core (int16 [n, maxj+1, 5],
counts in the order same, host, other, unbnd, absent), fate_tracer (int8 [n, maxj+1, ntracers], the
fate code of each of the ten, -1 if missing), status (int8, 0..4 = SAT CEN SUNK DEAD ABSENT), Nbound_j
(int32), overlap (int8), a_of_j (float32), snap_of_j (int32); censored = -1 / NaN; and args.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import bigfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloCatalogue, SubhaloSequence  # noqa: E402
from subhalo_tracer_study import load, cosmo_integrals, fshape, pct, take, NBINS  # noqa: E402

FATES = ("same", "host", "other", "unbnd", "absent")
STATUS = ("SAT", "CEN", "SUNK", "DEAD", "ABSENT")
SAT, CEN, SUNK, DEAD, ABSENT = range(5)
JLIST = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40)
KM_J = (4, 8, 16, 32)
NBIN_J = (1, 2, 4, 8, 16)
NEDGES = (0, 0.03, 0.1, 0.3, 1, np.inf)


def prev_ranks(path):
    """(TrackId, Rank) of a catalogue's Subhalo table, without its particle blocks."""
    f = bigfile.File(path)
    if "Subhalo/TrackId" not in f.blocks:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return f["Subhalo/TrackId"][:].astype(np.int64), f["Subhalo/Rank"][:].astype(np.int64)


def in_sorted(keys, q):
    """Row of q in the sorted array keys, -1 where absent."""
    if keys.size == 0:
        return np.full(np.shape(q), -1, np.int64)
    pos = np.minimum(np.searchsorted(keys, q), keys.size - 1)
    return np.where(keys[pos] == q, pos, -1)


def n_expected(E, ids, erow, a_next, args):
    """Expected scatterings over one output of particles ids of E-rows erow (study section 5, knn)."""
    q, hq, mq = E.knn
    if not np.isfinite(a_next) or ids.size == 0 or q.size == 0:
        return np.full(ids.size, np.nan)
    _, dt = cosmo_integrals(E.header, E.a, a_next)
    hub, ul, um = (float(np.atleast_1d(E.header[k])[0]) for k in ("HubbleParam", "UnitLength_in_cm", "UnitMass_in_g"))
    pos = np.minimum(np.searchsorted(q, ids), q.size - 1)
    hit = q[pos] == ids
    h = np.where(hit, hq[pos], np.nan) * E.a * ul / hub   # physical cm
    m = np.where(hit, mq[pos], np.nan)
    rho = m * um / hub * args.sidm_neighbors / (4 * np.pi / 3 * h ** 3)
    v = E.CoreSigmaV[erow].astype(np.float64)
    return rho * args.sigma0 * fshape((v / args.wturn) ** 2) * v * 1e5 * dt


class Cohorts(object):
    """Per-entry results (grown as entries open) and the flat marker lists of the open entries."""

    def __init__(self, args):
        self.args = args
        J, nt = args.maxj + 1, args.ntracers
        self.R = dict(TrackId=np.zeros(0, np.int64), snap_tag=np.zeros(0, np.int32), a_tag=np.zeros(0),
                      Nbound_tag=np.zeros(0, np.int64), Rank_tag=np.zeros(0, np.int64),
                      host_tag=np.zeros(0, np.int64), n_ten=np.zeros(0, np.int32), n_wide=np.zeros(0, np.int32),
                      n_core=np.zeros(0, np.int32), N_tag=np.zeros((0, nt)),
                      fate_ten=np.zeros((0, J, 5), np.int16), fate_wide=np.zeros((0, J, 5), np.int16),
                      fate_core=np.zeros((0, J, 5), np.int16), fate_tracer=np.zeros((0, J, nt), np.int8),
                      status=np.zeros((0, J), np.int8), Nbound_j=np.zeros((0, J), np.int32),
                      overlap=np.zeros((0, J), np.int8), a_of_j=np.zeros((0, J), np.float32),
                      snap_of_j=np.zeros((0, J), np.int32))
        self.tagpos = np.zeros(0, np.int64)    # walk position of the tag output
        # open markers: global entry row, ID, rank in the (BindRank, ID) order (wide) / core
        self.w_row, self.w_id, self.w_j = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.c_row, self.c_id = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.r0 = 0   # entries < r0 are closed (open entries are a suffix: rows are added in tag order)

    @property
    def n(self):
        return self.tagpos.size

    def open(self, E, sel, p, a_next):
        """New entries for the E-rows sel at walk position p; j = 0 filled without a lookup."""
        args, R = self.args, self.R
        n0, nn = self.n, sel.size
        J, nt = args.maxj + 1, args.ntracers
        local = np.full(E.tid.size, -1, np.int64)
        local[sel] = np.arange(nn)
        mw = local[E.tw_row] >= 0
        w_row, w_id, w_j = local[E.tw_row[mw]], E.tw_id[mw], E.tw_j[mw]
        mc = local[E.core_row] >= 0
        c_row, c_id = local[E.core_row[mc]], E.core_id[mc]
        ten = w_j < nt
        new = dict(TrackId=E.tid[sel], snap_tag=np.full(nn, E.snap, np.int32), a_tag=np.full(nn, E.a),
                   Nbound_tag=E.Nbound[sel].astype(np.int64), Rank_tag=E.Rank[sel].astype(np.int64),
                   host_tag=E.CentralTrackId[sel].astype(np.int64),
                   n_ten=np.bincount(w_row[ten], minlength=nn).astype(np.int32),
                   n_wide=np.bincount(w_row, minlength=nn).astype(np.int32),
                   n_core=np.bincount(c_row, minlength=nn).astype(np.int32),
                   N_tag=np.full((nn, nt), np.nan))
        if hasattr(E, "knn"):
            new["N_tag"][w_row[ten], w_j[ten]] = n_expected(E, w_id[ten], sel[w_row[ten]], a_next, args)
        for k, dt in (("fate_ten", np.int16), ("fate_wide", np.int16), ("fate_core", np.int16)):
            new[k] = np.full((nn, J, 5), -1, dt)
            new[k][:, 0] = 0
            new[k][:, 0, 0] = new["n" + k[4:]]
        new["fate_tracer"] = np.full((nn, J, nt), -1, np.int8)
        new["fate_tracer"][w_row[ten], 0, w_j[ten]] = 0
        for k, dt, v0 in (("status", np.int8, SAT), ("Nbound_j", np.int32, new["Nbound_tag"]),
                          ("overlap", np.int8, new["n_ten"])):
            new[k] = np.full((nn, J), -1, dt)
            new[k][:, 0] = v0
        new["a_of_j"] = np.full((nn, J), np.nan, np.float32)
        new["a_of_j"][:, 0] = E.a
        new["snap_of_j"] = np.full((nn, J), -1, np.int32)
        new["snap_of_j"][:, 0] = E.snap
        for k in R:
            R[k] = np.concatenate([R[k], new[k]])
        self.tagpos = np.concatenate([self.tagpos, np.full(nn, p, np.int64)])
        self.w_row, self.w_id = np.concatenate([self.w_row, w_row + n0]), np.concatenate([self.w_id, w_id])
        self.w_j = np.concatenate([self.w_j, w_j])
        self.c_row, self.c_id = np.concatenate([self.c_row, c_row + n0]), np.concatenate([self.c_id, c_id])
        return nn

    def follow(self, E, L, osink, p):
        """Fates, status, Nbound and overlap of the open entries at walk position p (j >= 1)."""
        args, R = self.args, self.R
        rows = np.arange(self.r0, self.n)
        rows = rows[self.tagpos[rows] < p]   # entries opened at p are filled by open()
        if rows.size == 0:
            return 0
        j = p - self.tagpos[rows]
        tid = R["TrackId"][rows]
        srow, orow = L.row(tid, "sub"), L.row(tid, "orphan")
        sub, orp = srow >= 0, orow >= 0
        status = np.full(rows.size, ABSENT, np.int8)
        status[sub] = np.where(E.Rank[srow[sub]] > 0, SAT, CEN)
        status[orp & ~sub] = np.where(osink[orow[orp & ~sub]] >= 0, SUNK, DEAD)
        # host at p: the current central, else the orphan's descendant; -1 never matches
        hostt = np.full(rows.size, -1, np.int64)
        hostt[sub] = E.CentralTrackId[srow[sub]]
        oo = orp & ~sub
        hostt[oo] = L.odesc[orow[oo]]
        loc = np.full(self.n - self.r0, -1, np.int64)   # entry row -> index into rows
        loc[rows - self.r0] = np.arange(rows.size)
        nr = rows.size

        def fates(mrow, mid):
            li = loc[mrow - self.r0]
            k = li >= 0
            li, mid = li[k], mid[k]
            pres, bt = L.look(mid)[:2]
            f = np.select([bt == tid[li], (bt == hostt[li]) & (hostt[li] >= 0), bt >= 0, pres], [0, 1, 2, 3], 4)
            return k, li, f, np.bincount(li * 5 + f, minlength=5 * nr).reshape(nr, 5)

        kw, lw, fw, cw = fates(self.w_row, self.w_id)
        ten = self.w_j[kw] < args.ntracers
        cten = np.bincount(lw[ten] * 5 + fw[ten], minlength=5 * nr).reshape(nr, 5)
        cc = fates(self.c_row, self.c_id)[3]
        R["fate_wide"][rows, j], R["fate_ten"][rows, j], R["fate_core"][rows, j] = cw, cten, cc
        R["fate_tracer"][rows[lw[ten]], j[lw[ten]], self.w_j[kw][ten]] = fw[ten]
        R["status"][rows, j] = status
        R["Nbound_j"][rows, j] = take(E.Nbound.astype(np.int64), srow, 0)
        R["a_of_j"][rows, j], R["snap_of_j"][rows, j] = E.a, E.snap
        # overlap: original ten among the finder's current ten, pairs (track row at p, ID)
        cur = E.tw_j < args.ntracers
        qrow, qid = srow[lw[ten]], self.w_id[kw][ten]
        uq, inv = np.unique(np.concatenate([E.tw_id[cur], qid]), return_inverse=True)
        span = max(uq.size, 1)
        kc = E.tw_row[cur].astype(np.int64) * span + inv[:cur.sum()]
        kq = qrow * span + inv[cur.sum():]
        hit = np.isin(kq, kc) & (qrow >= 0)
        R["overlap"][rows, j] = np.where(sub, np.bincount(lw[ten], weights=hit.astype(np.float64), minlength=nr), 0)
        return nr

    def prune(self, p):
        """Close the entries whose next j would exceed --maxj and drop their markers."""
        while self.r0 < self.n and p + 1 - self.tagpos[self.r0] > self.args.maxj:
            self.r0 += 1
        kw, kc = self.w_row >= self.r0, self.c_row >= self.r0
        self.w_row, self.w_id, self.w_j = self.w_row[kw], self.w_id[kw], self.w_j[kw]
        self.c_row, self.c_id = self.c_row[kc], self.c_id[kc]


def walk(args):
    seq = SubhaloSequence(args.sequence)
    idx = [i for i, s in enumerate(seq.snapshots) if args.first <= s <= (args.last if args.last >= 0 else s)]
    C = Cohorts(args)
    prev = prev_ranks(seq.paths[idx[0] - 1]) if idx and idx[0] > 0 and args.tag == "infall" else None
    for p, i in enumerate(idx):
        t0 = time.time()
        a_next = seq.scale_factors[i + 1] if i + 1 < len(seq) else np.nan
        cat = SubhaloCatalogue(seq.paths[i])
        largs = argparse.Namespace(**vars(args))
        if not np.isfinite(a_next):
            largs.density = None   # N needs the next output: no tree at the sequence's last one
        E, L = load(cat, True, True, largs)
        for k in ("fb_row", "fb_id", "fb_type", "fb_mass", "fb_rank"):   # full bound set: not needed
            delattr(E, k)
        osink = cat.orphans["SinkTrackId"].astype(np.int64) if cat.norphans else np.zeros(0, np.int64)
        nf = C.follow(E, L, osink, p)
        m = (E.Rank > 0) & (E.Nbound >= args.nmin)
        if args.tag == "infall":
            if prev is None:
                m[:] = False
            else:
                r = in_sorted(prev[0], E.tid)
                m &= (r >= 0) & (prev[1][np.maximum(r, 0)] == 0)
        no = C.open(E, np.nonzero(m)[0], p, a_next)
        C.prune(p)
        prev = (E.tid, E.Rank.astype(np.int64))
        del E, L, cat
        print("snap %03d a %.6f  opened %6d  followed %7d  %.0f s" % (seq.snapshots[i], seq.scale_factors[i], no, nf,
              time.time() - t0), file=sys.stderr)
        sys.stderr.flush()
    return [seq.snapshots[i] for i in idx], C.R


def report(args, snaps, R, out=sys.stdout):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    J = args.maxj + 1
    nb = R["Nbound_tag"]
    strata = [("all", np.ones(nb.size, bool))] + [(lab, (nb >= lo) & (nb < hi)) for lo, hi, lab in NBINS
                                                  if hi > args.nmin]
    rng = "%03d-%03d" % (snaps[0], snaps[-1]) if snaps else "none"
    p("# %s  tag %s  nmin %d  maxj %d  snapshots %s (%d outputs)  entries: %s" % (
        args.sequence, args.tag, args.nmin, args.maxj, rng, len(snaps),
        "  ".join("%s %d" % (lab, m.sum()) for lab, m in strata)))
    st, ten = R["status"], R["fate_ten"]
    defined = st >= 0
    alive = (st == SAT) | (st == CEN)
    nten = R["n_ten"].astype(np.float64)
    hasten = nten > 0
    jl = [j for j in JLIST if j < J and defined[:, j].any()]

    def mean_pct(cnt, nmark, m):   # mean over entries of the per-entry %; entries without markers left out
        m = m & (nmark > 0)
        return 100.0 * np.mean(cnt[m] / nmark[m]) if m.any() else np.nan

    p("\n1 markers vs j (outputs since tagging): mean over entries of the % of each entry's markers in each fate")
    p("%-8s %3s %7s | %-34s | %-13s | %-13s | %6s" % ("stratum", "j", "N", "ten: same   host  other  unbnd absent",
      "core sm  host", "wide sm  host", "ovlp"))
    for lab, m in strata:
        for j in jl:
            d = m & defined[:, j]
            ft = " ".join("%6.2f" % mean_pct(ten[:, j, f], nten, d) for f in range(5))
            fc = " ".join("%6.2f" % mean_pct(R["fate_core"][:, j, f], R["n_core"], d) for f in (0, 1))
            fw = " ".join("%6.2f" % mean_pct(R["fate_wide"][:, j, f], R["n_wide"], d) for f in (0, 1))
            ov = mean_pct(R["overlap"][:, j], nten, d & alive[:, j])
            p("%-8s %3d %7d | %s | %s | %s | %6.2f" % (lab, j, d.sum(), ft, fc, fw, ov))
    p("\n2 track status vs j: % of defined entries; over alive (SAT/CEN) entries, % with >= half of the ten"
      " same and % with none same")
    p("%-8s %3s %7s | %6s %6s %6s %6s | %7s %6s %6s" % ("stratum", "j", "N", "SAT", "CEN", "SUNK", "DEAD", "Nalive",
      ">=half", "none"))
    for lab, m in strata:
        for j in jl:
            d = m & defined[:, j]
            s = [pct(np.sum(st[d, j] == c), d.sum()) for c in (SAT, CEN, SUNK, DEAD)]
            a = d & alive[:, j] & hasten
            same = ten[:, j, 0]
            p("%-8s %3d %7d | %s | %7d %6.2f %6.2f" % (lab, j, d.sum(), " ".join("%6.2f" % x for x in s), a.sum(),
              pct(np.sum(2 * same[a] >= nten[a]), a.sum()), pct(np.sum(same[a] == 0), a.sum())))
    nabs = np.sum(st == ABSENT)
    if nabs:
        p("  warning: %d (entry, j) with the track in neither table (ABSENT)" % nabs)
    T, ev = km_times(R, J)
    p("\n3 half-life (Kaplan-Meier): event = fewer than half of the ten same while alive; censored at"
      " sink / death / sequence end")
    p("%-8s %7s %7s %7s | %8s | %s" % ("stratum", "N", "events", "cens", "median j", "  ".join("S(%2d)" % j for j in KM_J)))
    for lab, m in strata:
        m = m & hasten
        S, jmax = km(T[m], ev[m], J)
        med = next((j for j in range(1, jmax + 1) if S[j] <= 0.5), None)
        meds = "%d" % med if med is not None else "> %d" % jmax
        sj = "  ".join("%5.3f" % S[j] if j <= jmax else "    -" for j in KM_J)
        p("%-8s %7d %7d %7d | %8s | %s" % (lab, m.sum(), ev[m].sum(), np.sum(~ev[m] & (T[m] > 0)), meds, sj))
    if args.density != "knn":
        return
    p("\n4 tracer survival by expected scatterings over one output at tagging, N_tag (knn, sigma0/m %g cm^2/g,"
      " w %g km/s, Nngb %d): %% same (tracers)" % (args.sigma0, args.wturn, args.sidm_neighbors))
    js = [j for j in NBIN_J if j < J and defined[:, j].any()]
    p("%-8s %-15s | %s" % ("stratum", "N_tag", " | ".join("j=%-2d %%same   ntr" % j for j in js)))
    ft, N = R["fate_tracer"], R["N_tag"]
    for lab, m in strata:
        for lo, hi in zip(NEDGES[:-1], NEDGES[1:]):
            inb = (N >= lo) & (N < hi) & m[:, None]
            cols = []
            for j in js:
                t = inb & defined[:, j][:, None] & (ft[:, j] >= 0)
                cols.append("%10.2f %7d" % (pct(np.sum(ft[:, j][t] == 0), t.sum()), t.sum()))
            p("%-8s [%5g, %5g) | %s" % (lab, lo, hi, " | ".join(cols)))
    p("  tracers without N_tag: %d of %d" % (np.sum(~np.isfinite(N) & (ft[:, 0] >= 0)), np.sum(ft[:, 0] >= 0)))


def km_times(R, J):
    """(T, event) per entry: the event j, or the last j at risk (censored); see the docstring."""
    st, same, nten = R["status"], R["fate_ten"][:, :, 0], R["n_ten"]
    alive = (st == SAT) | (st == CEN)
    low = 2 * same.astype(np.int64) < nten[:, None]
    ok = alive & ~low
    ok[:, 0] = True
    run = np.where(ok.all(1), J, np.argmin(ok, axis=1))   # leading at-risk outputs, j = 0 .. run-1
    nxt = np.minimum(run, J - 1)
    ev = (run < J) & alive[np.arange(st.shape[0]), nxt] & low[np.arange(st.shape[0]), nxt]
    return np.where(ev, run, run - 1), ev


def km(T, ev, J):
    """KM survival S[j], j = 0 .. J-1, and the last j with entries at risk."""
    S = np.ones(J)
    jmax = 0
    for j in range(1, J):
        risk = np.sum(T >= j)
        if risk == 0:
            S[j:] = S[j - 1]
            break
        jmax = j
        S[j] = S[j - 1] * (1.0 - np.sum(ev & (T == j)) / float(risk))
    return S, jmax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sequence", required=True, help="output directory of PIG_*")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=-1)
    ap.add_argument("--tag", choices=("infall", "all"), default="infall")
    ap.add_argument("--nmin", type=int, default=100, help="Nbound at tagging")
    ap.add_argument("--maxj", type=int, default=40, help="outputs followed after tagging")
    ap.add_argument("--density", choices=("knn",), default=None, help="N_tag from the Nngb-th neighbour distance")
    ap.add_argument("--sigma0", type=float, default=147.1, help="sigma0/m in cm^2/g")
    ap.add_argument("--wturn", type=float, default=120.0, help="velocity scale w in km/s")
    ap.add_argument("--sidm-neighbors", type=int, default=38)
    ap.add_argument("--ntracers", type=int, default=10)
    ap.add_argument("--nwide", type=int, default=50)
    ap.add_argument("--ncore", type=int, default=20)
    ap.add_argument("--rows", help="write the per-entry arrays to this .npz")
    args = ap.parse_args(argv)
    args.nwide = max(args.nwide, args.ntracers)
    if args.ntracers > 127 or args.maxj < 0:
        ap.error("--ntracers must fit the int8 overlap (<= 127), --maxj >= 0")
    snaps, R = walk(args)
    report(args, snaps, R)
    if args.rows:
        np.savez(args.rows, snapshots=np.array(snaps, np.int32), fate_names=np.array(FATES),
                 status_names=np.array(STATUS), args=np.array(["%s=%s" % kv for kv in sorted(vars(args).items())]),
                 **R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
