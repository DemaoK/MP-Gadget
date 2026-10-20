#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How long the finder's marker particles, tagged when a satellite falls in, stay bound to it over
many outputs, where they go, and whether the track itself survives (CDM vs SIDM study); or, with
--tag death, how coherent the markers of a dying track stay after its death.

Usage:
  subhalo_tracer_survival.py --sequence DIR [--first N --last M] [--tag infall|all|death] [--nmin 100]
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
             above 0.5 up to J, the last j with entries at risk; S at j = 4, 8, 16, 32. "cens" =
             every non-event entry (also those never at risk: tagged at the last output, or dead at j = 1).
  4 SIDM     with --density knn: the expected scatterings over one output, N_tag, of each tracer at
             tagging (section 5 of subhalo_tracer_study.py: knn support over the stored particles,
             v = CoreSigmaV of the track, dt to the next output of the sequence; NaN at its last
             output), and the % of tracers still "same" at j = 1, 2, 4, 8, 16 by N_tag bin.
Entries with fewer markers than requested use their own count as the denominator; those with no
tracer are left out of the tracer figures.

--tag death: an entry per track that dies or sinks at output k (Orphan row with SnapDeath == k, not
in k's Subhalo table) and was resolved with Nbound >= --nmin at the previous output k-1 (for the
first output of the range k-1's particles are read too). Its markers are those of k-1, the set the
finder's descendant vote used (the ten's first is the MostBoundID the orphan follows); strata and
the *_tag scalars are k-1's, snap_tag / a_tag are k's. j = 0 is k itself, followed with a lookup;
"host" = the DescendantTrackId at k when >= 0, else the SinkTrackId at k. Spatial coherence of the
ten at k + j, from the Position / Velocity of the stored ones ("found"; field markers are absent):
  centre      component-wise median of the found positions (unwrapped about the first found marker,
              in rank order) and of their velocities; distances below are minimum-image residuals
              about it
  sigma_x/v   rms 3D distance of the found markers from the centre (comoving catalogue units;
              physical peculiar km/s: the Velocity block times UnitVelocity_in_cm_per_s / 1e5, and
              divided by a when the Header's UsePeculiarVelocity is 0, i.e. it holds a v_pec),
              *_rel = / CoreSigmaX or CoreSigmaV at k-1 (physical peculiar km/s); one found -> 0,
              none -> NaN (no centre)
  n_coherent  found markers within 3 CoreSigmaX(k-1) of the centre (the remnant)
  d_clip      distance of the mean of those markers from the centre, / CoreSigmaX (NaN if none)
  d_mb        distance of the ten's first marker from the centre, / CoreSigmaX; NaN if not stored
  d_orphan    the same for the Orphan row's TracerPos at k + j while SnapTracerLost < 0 (equals
              d_mb while that particle is stored and is the orphan's MostBoundID; a sunk track's
              orphan can follow another particle)
  plurality   the BoundTrackId holding most found markers (ties to the smaller id, -1 if none is
              bound), frac_plurality = its count / n_found.
Death report: fates of section 1 (no overlap) for j = 0, 1, 2, 4, 8, 16, then per stratum and j the
medians / 90th percentiles of sigma_x_rel, sigma_v_rel, medians of n_found, n_coherent, d_mb,
d_clip, % with d_mb > 3 (of those with d_mb), % with at least half of the ten coherent, % whose
plurality is the DescendantTrackId (of entries with DescendantTrackId >= 0). --density knn is not
used in this mode.

--rows writes per entry: TrackId, snap_tag, a_tag, Nbound_tag, Rank_tag, host_tag, n_ten, n_wide,
n_core, N_tag [n, ntracers]; per (entry, j): fate_ten / fate_wide / fate_core (int16 [n, maxj+1, 5],
counts in the order same, host, other, unbnd, absent), fate_tracer (int8 [n, maxj+1, ntracers], the
fate code of each of the ten, -1 if missing), status (int8, 0..4 = SAT CEN SUNK DEAD ABSENT), Nbound_j
(int32), overlap (int8), a_of_j (float32), snap_of_j (int32); censored = -1 / NaN; and args.
--tag death adds per entry desc_death, sink_death, host_target, a_prev, CoreSigmaX_tag,
CoreSigmaV_tag, mbid_prev (MostBoundID at k-1), tracer_id (Orphan MostBoundID at k), ten0_id; per
(entry, j) sigma_x, sigma_v, sigma_x_rel, sigma_v_rel, d_mb, d_orphan, d_clip, frac_plurality
(float64), n_found, n_coherent, mb_found (int8), plurality (int64), centre_x, centre_v [n, maxj+1, 3].
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
from subhalo_tracer_study import (load, cosmo_integrals, fshape, pct, take, vote, periodic,  # noqa: E402
                                  read_particles, Earlier, Later, NBINS, NOBALLOT)

FATES = ("same", "host", "other", "unbnd", "absent")
STATUS = ("SAT", "CEN", "SUNK", "DEAD", "ABSENT")
SAT, CEN, SUNK, DEAD, ABSENT = range(5)
JLIST = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40)
DEATH_J = (0, 1, 2, 4, 8, 16)
KM_J = (4, 8, 16, 32)
NBIN_J = (1, 2, 4, 8, 16)
NEDGES = (0, 0.03, 0.1, 0.3, 1, np.inf)
FULL_BOUND = ("fb_row", "fb_id", "fb_type", "fb_mass", "fb_rank")
# death mode: per-entry scalars and per-(entry, j) spatial arrays (censored = NaN / -1)
DEATH_SCALARS = (("desc_death", np.int64), ("sink_death", np.int64), ("host_target", np.int64),
                 ("a_prev", np.float64), ("CoreSigmaX_tag", np.float64), ("CoreSigmaV_tag", np.float64),
                 ("mbid_prev", np.int64), ("tracer_id", np.int64), ("ten0_id", np.int64))
SPATIAL = (("sigma_x", np.float64), ("sigma_v", np.float64), ("sigma_x_rel", np.float64),
           ("sigma_v_rel", np.float64), ("d_mb", np.float64), ("d_orphan", np.float64), ("d_clip", np.float64),
           ("frac_plurality", np.float64), ("n_found", np.int8), ("n_coherent", np.int8), ("mb_found", np.int8),
           ("plurality", np.int64))


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


def group_median(g, x, n):
    """Median of x per group g = 0..n-1 (NaN for empty groups)."""
    out = np.full(n, np.nan)
    if g.size == 0:
        return out
    xs = x[np.lexsort((x, g))]
    cnt = np.bincount(g, minlength=n)
    start = np.cumsum(cnt) - cnt
    m = cnt > 0
    out[m] = 0.5 * (xs[(start + (cnt - 1) // 2)[m]] + xs[(start + cnt // 2)[m]])
    return out


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


def pecvel_flag(h):
    """UsePeculiarVelocity of a Header: missing -> 0 (petaio's default); a string -> 1 if 'peculiar_physical'."""
    v = h.get("UsePeculiarVelocity", 0)
    if isinstance(v, (str, bytes)) or np.asarray(v).dtype.kind in "SUO":
        t = "".join(x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(v))
        return int("peculiar_physical" in t)
    return int(np.atleast_1d(v)[0])


class LaterPV(Later):
    """Later plus Position / Velocity in its sorted-ID order and the Orphan tracer columns (death mode).
    P must already be sorted by ID, so Later's own argsort is the identity."""

    def __init__(self, cat, P):
        Later.__init__(self, cat, P)
        self.pos, self.vel = P["Position"].astype(np.float64), P["Velocity"].astype(np.float64)
        self.box = float(np.atleast_1d(cat.header["BoxSize"])[0])
        # Velocity block: peculiar velocity if UsePeculiarVelocity (else a * v_pec, petaio's default 0)
        self.pecvel = pecvel_flag(cat.header)
        uvel = float(np.atleast_1d(cat.header.get("UnitVelocity_in_cm_per_s", 1e5))[0])
        self.vel *= (uvel / 1e5) / (1.0 if self.pecvel else cat.scale_factor)   # -> physical peculiar km/s
        O, no = cat.orphans, cat.norphans
        self.osink = O["SinkTrackId"].astype(np.int64) if no else np.zeros(0, np.int64)
        self.olost = O["SnapTracerLost"].astype(np.int64) if no else np.zeros(0, np.int64)
        self.otpos = O["TracerPos"].astype(np.float64) if no else np.zeros((0, 3))
        self.ombid = O["MostBoundID"].astype(np.int64) if no else np.zeros(0, np.int64)

    def pv(self, q):
        """(Position, Velocity) of particle IDs q; NaN where not stored."""
        row = in_sorted(self.ids, q)
        ok = row >= 0
        x, v = np.full((q.size, 3), np.nan), np.full((q.size, 3), np.nan)
        x[ok], v[ok] = self.pos[row[ok]], self.vel[row[ok]]
        return x, v


def load_death(cat, args):
    """Earlier and LaterPV views of one catalogue from one read of its blocks, sorted by ID once."""
    P = read_particles(cat, ["ID", "BoundTrackId", "BindRank", "TrackId", "Mass", "GroupID", "Position", "Velocity"])
    cat.clear_cache()
    o = np.argsort(P["ID"].astype(np.int64), kind="stable")
    for k in list(P):   # Earlier's orders are fully keyed, so sorting P first changes nothing there
        P[k] = P[k][o]
    del o
    E = Earlier(cat, P, args)
    for k in FULL_BOUND:
        delattr(E, k)
    L = LaterPV(cat, P)
    del P
    return E, L


class Cohorts(object):
    """Per-entry results (grown as entries open) and the flat marker lists of the open entries."""

    def __init__(self, args):
        self.args = args
        self.death = args.tag == "death"
        self.R = self.blank(0)
        self.tagpos = np.zeros(0, np.int64)    # walk position of the tag output
        # open markers: global entry row, ID, rank in the (BindRank, ID) order (wide) / core
        self.w_row, self.w_id, self.w_j = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.c_row, self.c_id = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.r0 = 0   # entries < r0 are closed (open entries are a suffix: rows are added in tag order)
        self.nskip = 0   # death mode: deaths not resolved at k-1

    @property
    def n(self):
        return self.tagpos.size

    def blank(self, nn):
        """nn entries with every per-j value censored."""
        J, nt = self.args.maxj + 1, self.args.ntracers
        R = dict(TrackId=np.zeros(nn, np.int64), snap_tag=np.zeros(nn, np.int32), a_tag=np.zeros(nn),
                 Nbound_tag=np.zeros(nn, np.int64), Rank_tag=np.zeros(nn, np.int64),
                 host_tag=np.zeros(nn, np.int64), n_ten=np.zeros(nn, np.int32), n_wide=np.zeros(nn, np.int32),
                 n_core=np.zeros(nn, np.int32), N_tag=np.full((nn, nt), np.nan),
                 fate_ten=np.full((nn, J, 5), -1, np.int16), fate_wide=np.full((nn, J, 5), -1, np.int16),
                 fate_core=np.full((nn, J, 5), -1, np.int16), fate_tracer=np.full((nn, J, nt), -1, np.int8),
                 status=np.full((nn, J), -1, np.int8), Nbound_j=np.full((nn, J), -1, np.int32),
                 overlap=np.full((nn, J), -1, np.int8), a_of_j=np.full((nn, J), np.nan, np.float32),
                 snap_of_j=np.full((nn, J), -1, np.int32))
        if self.death:
            for k, dt in DEATH_SCALARS:
                R[k] = np.zeros(nn, dt)
            for k, dt in SPATIAL:
                R[k] = np.full((nn, J), np.nan if dt == np.float64 else -1, dt)
            R["centre_x"], R["centre_v"] = np.full((nn, J, 3), np.nan), np.full((nn, J, 3), np.nan)
        return R

    def markers(self, E, sel):
        """Flat marker lists (local entry row, ID[, rank]) of the E-rows sel, in E-row order."""
        local = np.full(E.tid.size, -1, np.int64)
        local[sel] = np.arange(sel.size)
        mw = local[E.tw_row] >= 0
        mc = local[E.core_row] >= 0
        return local[E.tw_row[mw]], E.tw_id[mw], E.tw_j[mw], local[E.core_row[mc]], E.core_id[mc]

    def scalars(self, E, sel, mk):
        """A blank block for the E-rows sel with the per-entry values taken from E."""
        nn = sel.size
        w_row, _, w_j, c_row, _ = mk
        ten = w_j < self.args.ntracers
        new = self.blank(nn)
        new.update(TrackId=E.tid[sel], snap_tag=np.full(nn, E.snap, np.int32), a_tag=np.full(nn, E.a),
                   Nbound_tag=E.Nbound[sel].astype(np.int64), Rank_tag=E.Rank[sel].astype(np.int64),
                   host_tag=E.CentralTrackId[sel].astype(np.int64),
                   n_ten=np.bincount(w_row[ten], minlength=nn).astype(np.int32),
                   n_wide=np.bincount(w_row, minlength=nn).astype(np.int32),
                   n_core=np.bincount(c_row, minlength=nn).astype(np.int32))
        return new

    def append(self, new, p, mk):
        n0 = self.n
        w_row, w_id, w_j, c_row, c_id = mk
        for k in self.R:
            self.R[k] = np.concatenate([self.R[k], new[k]])
        self.tagpos = np.concatenate([self.tagpos, np.full(new["TrackId"].size, p, np.int64)])
        self.w_row, self.w_id = np.concatenate([self.w_row, w_row + n0]), np.concatenate([self.w_id, w_id])
        self.w_j = np.concatenate([self.w_j, w_j])
        self.c_row, self.c_id = np.concatenate([self.c_row, c_row + n0]), np.concatenate([self.c_id, c_id])
        return new["TrackId"].size

    def open(self, E, sel, p, a_next):
        """New entries for the E-rows sel at walk position p; j = 0 filled without a lookup."""
        args = self.args
        mk = self.markers(E, sel)
        w_row, w_id, w_j = mk[:3]
        ten = w_j < args.ntracers
        new = self.scalars(E, sel, mk)
        if hasattr(E, "knn"):
            new["N_tag"][w_row[ten], w_j[ten]] = n_expected(E, w_id[ten], sel[w_row[ten]], a_next, args)
        for k in ("fate_ten", "fate_wide", "fate_core"):
            new[k][:, 0] = 0
            new[k][:, 0, 0] = new["n" + k[4:]]
        new["fate_tracer"][w_row[ten], 0, w_j[ten]] = 0
        new["status"][:, 0], new["Nbound_j"][:, 0], new["overlap"][:, 0] = SAT, new["Nbound_tag"], new["n_ten"]
        new["a_of_j"][:, 0], new["snap_of_j"][:, 0] = E.a, E.snap
        return self.append(new, p, mk)

    def open_death(self, Ep, L, p):
        """Entries for the tracks dying or sinking at L (walk position p), with Ep = Earlier(k-1)."""
        orow = np.nonzero((L.odeath == L.snap) & (L.row(L.otid, "sub") < 0))[0]
        prow = in_sorted(Ep.tid, L.otid[orow])
        self.nskip += np.sum(prow < 0)
        keep = prow >= 0
        keep[keep] = Ep.Nbound[prow[keep]] >= self.args.nmin
        orow, prow = orow[keep], prow[keep]
        mk = self.markers(Ep, prow)
        new = self.scalars(Ep, prow, mk)
        new["snap_tag"][:], new["a_tag"][:], new["a_prev"] = L.snap, L.a, np.full(prow.size, Ep.a)
        new["desc_death"], new["sink_death"] = L.odesc[orow], L.osink[orow]
        new["host_target"] = np.where(new["desc_death"] >= 0, new["desc_death"], new["sink_death"])
        new["CoreSigmaX_tag"] = Ep.CoreSigmaX[prow].astype(np.float64)
        new["CoreSigmaV_tag"] = Ep.CoreSigmaV[prow].astype(np.float64)
        new["mbid_prev"], new["tracer_id"] = Ep.MostBoundID[prow].astype(np.int64), L.ombid[orow]
        new["ten0_id"] = np.full(prow.size, -1, np.int64)
        first = mk[2] == 0
        new["ten0_id"][mk[0][first]] = mk[1][first]
        return self.append(new, p, mk)

    def follow(self, E, L, osink, p):
        """Fates, status, Nbound and overlap of the open entries at walk position p (j >= 1; death: j >= 0)."""
        args, R = self.args, self.R
        rows = np.arange(self.r0, self.n)
        if not self.death:
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
        if self.death:   # host fixed at death: the descendant, else the sink
            hostt = R["host_target"][rows]
        else:   # host at p: the current central, else the orphan's descendant; -1 never matches
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
        if self.death:
            self.spatial(L, rows, j, orow, lw[ten], qid, self.w_j[kw][ten])
        return nr

    def spatial(self, L, rows, j, orow, li, ids, rank):
        """Coherence of the ten (index li into rows, rank order within an entry) at this output."""
        R, nr = self.R, rows.size
        pres, bt = L.look(ids)[:2]
        X, V = L.pv(ids)
        lf, X, V, rk, bf = li[pres], X[pres], V[pres], rank[pres], bt[pres]
        nf = np.bincount(lf, minlength=nr)
        has = nf > 0
        ref = np.full((nr, 3), np.nan)
        u, first = np.unique(lf, return_index=True)   # the first found marker in rank order
        ref[u] = X[first]
        D = periodic(X - ref[lf], L.box)
        med = np.stack([group_median(lf, D[:, c], nr) for c in range(3)], 1)
        vmed = np.stack([group_median(lf, V[:, c], nr) for c in range(3)], 1)
        cen = ref + med
        res = periodic(D - med[lf], L.box)   # minimum image about the median, not about the first marker
        r = np.linalg.norm(res, axis=1)
        rv = np.linalg.norm(V - vmed[lf], axis=1)
        nfd = np.where(has, nf, np.nan)
        sx = np.sqrt(np.bincount(lf, weights=r ** 2, minlength=nr) / nfd)
        sv = np.sqrt(np.bincount(lf, weights=rv ** 2, minlength=nr) / nfd)
        csx = R["CoreSigmaX_tag"][rows]
        csx = np.where(csx > 0, csx, np.nan)
        csv = R["CoreSigmaV_tag"][rows]
        csv = np.where(csv > 0, csv, np.nan)
        ins = r <= 3 * csx[lf]
        ncoh = np.bincount(lf[ins], minlength=nr)
        # clipped centre: mean offset of the markers inside 3 CoreSigmaX, relative to the median
        cm = np.zeros((nr, 3))   # float even when no marker passes (bincount of nothing is integer)
        for c in range(3):
            cm[:, c] = np.bincount(lf[ins], weights=res[ins, c], minlength=nr)
        cm /= np.where(ncoh > 0, ncoh, np.nan)[:, None]
        dclip = np.linalg.norm(cm, axis=1) / csx
        mb = rk == 0
        dmb, mbf = np.full(nr, np.nan), np.zeros(nr, np.int8)
        dmb[lf[mb]] = np.linalg.norm(periodic(X[mb] - cen[lf[mb]], L.box), axis=1) / csx[lf[mb]]
        mbf[lf[mb]] = 1
        dorph = np.full(nr, np.nan)
        ok = (orow >= 0) & has
        ok[ok] = L.olost[orow[ok]] < 0
        dorph[ok] = np.linalg.norm(periodic(L.otpos[orow[ok]] - cen[ok], L.box), axis=1) / csx[ok]
        win, top, _ = vote(lf, bf, np.ones(lf.size), nr, skip_negative=True)
        out = dict(sigma_x=sx, sigma_v=sv, sigma_x_rel=sx / csx, sigma_v_rel=sv / csv, d_mb=dmb, d_orphan=dorph,
                   d_clip=dclip, frac_plurality=np.where(has, top / np.maximum(nf, 1), np.nan), n_found=nf,
                   n_coherent=ncoh, mb_found=mbf, plurality=np.where(win == NOBALLOT, -1, win))
        for k, v in out.items():
            R[k][rows, j] = v
        cw = np.mod(cen, L.box)
        cw[cw >= L.box] = 0.0   # mod of a tiny negative coordinate can round to the box size
        R["centre_x"][rows, j], R["centre_v"][rows, j] = cw, vmed

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
    death = args.tag == "death"
    prev = prev_ranks(seq.paths[idx[0] - 1]) if idx and idx[0] > 0 and args.tag == "infall" else None
    prevE = None
    if death and idx and idx[0] > 0:   # the markers of deaths at the first output come from the one before
        prevE = load(SubhaloCatalogue(seq.paths[idx[0] - 1]), True, False, args)[0]
        for k in FULL_BOUND:
            delattr(prevE, k)
    for p, i in enumerate(idx):
        t0 = time.time()
        a_next = seq.scale_factors[i + 1] if i + 1 < len(seq) else np.nan
        cat = SubhaloCatalogue(seq.paths[i])
        if death:
            E, L = load_death(cat, args)
        else:
            largs = argparse.Namespace(**vars(args))
            if not np.isfinite(a_next):
                largs.density = None   # N needs the next output: no tree at the sequence's last one
            E, L = load(cat, True, True, largs)
            for k in FULL_BOUND:   # full bound set: not needed
                delattr(E, k)
        osink = cat.orphans["SinkTrackId"].astype(np.int64) if cat.norphans else np.zeros(0, np.int64)
        if death:
            no = C.open_death(prevE, L, p) if prevE is not None else 0
            nf = C.follow(E, L, osink, p)
            prevE = E
        else:
            nf = C.follow(E, L, osink, p)
            m = (E.Rank > 0) & (E.Nbound >= args.nmin)
            if args.tag == "infall":
                if prev is None:
                    m[:] = False
                else:
                    r = in_sorted(prev[0], E.tid)
                    m &= (r >= 0) & (take(prev[1], r, -1) == 0)   # take: the previous table may be empty
            no = C.open(E, np.nonzero(m)[0], p, a_next)
            prev = (E.tid, E.Rank.astype(np.int64))
        C.prune(p)
        del E, L, cat
        print("snap %03d a %.6f  opened %6d  followed %7d  %.0f s" % (seq.snapshots[i], seq.scale_factors[i], no, nf,
              time.time() - t0), file=sys.stderr)
        sys.stderr.flush()
    if death and C.nskip:
        print("note: %d deaths not resolved at the previous output (skipped)" % C.nskip, file=sys.stderr)
    return [seq.snapshots[i] for i in idx], C.R


def mean_pct(cnt, nmark, m):
    """Mean over entries of the per-entry %; entries without markers left out."""
    m = m & (nmark > 0)
    return 100.0 * np.mean(cnt[m] / nmark[m]) if m.any() else np.nan


def fate_table(p, strata, R, jl, ovlp=True):
    """Section 1: the mean fate % of the ten, same / host of core and wide [, overlap] per stratum and j."""
    defined = R["status"] >= 0
    alive = (R["status"] == SAT) | (R["status"] == CEN)
    nten = R["n_ten"].astype(np.float64)
    p("%-8s %3s %7s | %-34s | %-13s | %-13s%s" % ("stratum", "j", "N", "ten: same   host  other  unbnd absent",
      "core sm  host", "wide sm  host", " | %6s" % "ovlp" if ovlp else ""))
    for lab, m in strata:
        for j in jl:
            d = m & defined[:, j]
            ft = " ".join("%6.2f" % mean_pct(R["fate_ten"][:, j, f], nten, d) for f in range(5))
            fc = " ".join("%6.2f" % mean_pct(R["fate_core"][:, j, f], R["n_core"], d) for f in (0, 1))
            fw = " ".join("%6.2f" % mean_pct(R["fate_wide"][:, j, f], R["n_wide"], d) for f in (0, 1))
            ov = " | %6.2f" % mean_pct(R["overlap"][:, j], nten, d & alive[:, j]) if ovlp else ""
            p("%-8s %3d %7d | %s | %s | %s%s" % (lab, j, d.sum(), ft, fc, fw, ov))


def header(p, args, snaps, strata):
    rng = "%03d-%03d" % (snaps[0], snaps[-1]) if snaps else "none"
    p("# %s  tag %s  nmin %d  maxj %d  snapshots %s (%d outputs)  entries: %s" % (
        args.sequence, args.tag, args.nmin, args.maxj, rng, len(snaps),
        "  ".join("%s %d" % (lab, m.sum()) for lab, m in strata)))


def report(args, snaps, R, out=sys.stdout):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    J = args.maxj + 1
    nb = R["Nbound_tag"]
    strata = [("all", np.ones(nb.size, bool))] + [(lab, (nb >= lo) & (nb < hi)) for lo, hi, lab in NBINS
                                                  if hi > args.nmin]
    header(p, args, snaps, strata)
    if args.tag == "death":
        return report_death(p, args, R, strata)
    st, ten = R["status"], R["fate_ten"]
    defined = st >= 0
    alive = (st == SAT) | (st == CEN)
    nten = R["n_ten"].astype(np.float64)
    hasten = nten > 0
    jl = [j for j in JLIST if j < J and defined[:, j].any()]
    p("\n1 markers vs j (outputs since tagging): mean over entries of the % of each entry's markers in each fate")
    fate_table(p, strata, R, jl)
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
        med = next((j for j in range(1, jmax + 1) if S[j] <= 0.5 + 1e-9), None)   # roundoff of exact crossings
        meds = "%d" % med if med is not None else "> %d" % jmax
        sj = "  ".join("%5.3f" % S[j] if j <= jmax else "    -" for j in KM_J)
        p("%-8s %7d %7d %7d | %8s | %s" % (lab, m.sum(), ev[m].sum(), np.sum(~ev[m]), meds, sj))
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


def report_death(p, args, R, strata):
    J = args.maxj + 1
    st = R["status"]
    defined = st >= 0
    n = R["TrackId"].size
    jl = [j for j in DEATH_J if j < J and defined[:, j].any()]
    mis = R["ten0_id"] != R["tracer_id"]
    p("  deaths: %d sunk (SinkTrackId >= 0), %d with DescendantTrackId < 0, Rank at k-1 > 0: %d;"
      " ten[0] != MostBoundID(k-1): %d, != Orphan MostBoundID(k): %d (%d of them sunk)" % (
          np.sum(R["sink_death"] >= 0), np.sum(R["desc_death"] < 0), np.sum(R["Rank_tag"] > 0),
          np.sum(R["ten0_id"] != R["mbid_prev"]), mis.sum(), np.sum(mis & (R["sink_death"] >= 0))))
    rev = np.sum(((st == SAT) | (st == CEN)).any(1))
    if rev or np.sum(st == ABSENT):
        p("  warning: %d entries resolved again after death, %d (entry, j) ABSENT" % (rev, np.sum(st == ABSENT)))
    p("\n1 markers vs j (outputs since death, j = 0 the death output; host = DescendantTrackId at death, else"
      " SinkTrackId): mean % of each entry's markers in each fate")
    fate_table(p, strata, R, jl, ovlp=False)
    p("\n2 spatial coherence of the ten (found = stored at k + j), x / v in CoreSigmaX / CoreSigmaV at k-1:"
      " median and 90th pct; n of %d; %%d_mb>3 of entries with d_mb; %%coh = n_coherent >= half;"
      " %%plur=desc of entries with DescendantTrackId >= 0" % args.ntracers)
    p("%-8s %3s %6s | %-13s %-13s | %5s %5s | %6s %6s %6s %6s %6s %6s" % (
        "stratum", "j", "N", "sigx med/p90", "sigv med/p90", "nfnd", "ncoh", "d_mb", "d_clip", "nomb%", "dmb>3",
        "coh%", "plur="))

    def q(x, m, c):   # percentile c of the finite values of x[m]
        x = x[m]
        x = x[np.isfinite(x)]
        return np.percentile(x, c) if x.size else np.nan
    for lab, m in strata:
        for j in jl:
            d = m & defined[:, j]
            sx, sv = R["sigma_x_rel"][:, j], R["sigma_v_rel"][:, j]
            dmb = R["d_mb"][:, j]
            hasmb = d & np.isfinite(dmb)
            hasd = d & (R["desc_death"] >= 0)
            p("%-8s %3d %6d | %6.2f %6.2f %6.2f %6.2f | %5.1f %5.1f | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f" % (
                lab, j, d.sum(), q(sx, d, 50), q(sx, d, 90), q(sv, d, 50), q(sv, d, 90),
                q(R["n_found"][:, j].astype(float), d, 50), q(R["n_coherent"][:, j].astype(float), d, 50),
                q(dmb, d, 50), q(R["d_clip"][:, j], d, 50), pct(np.sum(d & ~hasmb), d.sum()),
                pct(np.sum(dmb[hasmb] > 3), hasmb.sum()),
                pct(np.sum(2 * R["n_coherent"][d, j].astype(np.int64) >= R["n_ten"][d]), d.sum()),
                pct(np.sum(R["plurality"][hasd, j] == R["desc_death"][hasd]), hasd.sum())))
    # d_orphan == d_mb only where the orphan follows ten[0] (sunk tracks may follow another particle)
    both = defined & np.isfinite(R["d_mb"]) & np.isfinite(R["d_orphan"])
    same = both & (R["ten0_id"] == R["tracer_id"])[:, None]
    rel = np.abs(R["d_orphan"] - R["d_mb"]) / np.maximum(R["d_mb"], 1e-30)
    p("  d_orphan vs d_mb where both defined and the orphan's tracer is ten[0]: %d (entry, j), max relative"
      " difference %.3g; other tracer: %d (entry, j), median relative difference %.3g; d_orphan without d_mb: %d"
      % (same.sum(), rel[same].max() if same.any() else np.nan, np.sum(both & ~same),
         np.median(rel[both & ~same]) if (both & ~same).any() else np.nan,
         np.sum(defined & np.isfinite(R["d_orphan"]) & ~np.isfinite(R["d_mb"]))))
    nf = R["n_found"]
    p("  entries x j with n_found 0 / 1: %d / %d (sigma NaN / 0); of %d entries x j defined, %d entries" % (
        np.sum(defined & (nf == 0)), np.sum(defined & (nf == 1)), defined.sum(), n))


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
    ap.add_argument("--tag", choices=("infall", "all", "death"), default="infall")
    ap.add_argument("--nmin", type=int, default=100, help="Nbound at tagging (death: at the previous output)")
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
    if max(args.nwide, args.ncore) > 32767:
        ap.error("--nwide and --ncore must fit the int16 fate counts (<= 32767)")
    if args.tag == "death" and args.density:
        ap.error("--density knn is not used with --tag death")
    snaps, R = walk(args)
    report(args, snaps, R)
    if args.rows:
        np.savez(args.rows, snapshots=np.array(snaps, np.int32), fate_names=np.array(FATES),
                 status_names=np.array(STATUS), args=np.array(["%s=%s" % kv for kv in sorted(vars(args).items())]),
                 **R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
