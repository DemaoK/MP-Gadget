#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-to-one satellite comparison of the runtime subhalo finder with Rockstar and consistent-trees.

Usage:
  subhalo_rockstar_satellites.py --ours PIGDIR --rockstar OUTBASE --hlists HLISTDIR --out DIR
      [--census 128,140,152] [--anchors 128,140] [--steps 1,2,5] [--rs-snap-offset 0]
      [--maxset 20000] [--no-unbind] [--min-np 20] [--nproc N]

Inputs. Ours: PIGDIR/PIG_k (the Ours reader of subhalo_compare_hbt.py) and PIGDIR/PART_k (type 1, for the
independent unbinding). Rockstar: OUTBASE/out_N.list and halos_*.bin of N = k + --rs-snap-offset (the
Rockstar reader of subhalo_compare_rockstar.py: pid by the parents.c rule, unique particle membership
from the .bin files; a particle ID listed more than once, a halo with Np > 0 and no listed particle (a
missing .bin chunk), a halo or parent ID that is not an integer below 2**53 in magnitude (the parser reads
IDs through float64) and a pid >= 0 absent from the catalogue are input errors). consistent-trees: HLISTDIR/hlist_<scale>.list, the hlist of output k being the one within 2e-5 in
scale of PIG_k's Time (hlist scales carry five decimals); a row continues at the next hlist as the row with
id == desc_id when mmp? == 1, and a desc_id >= 0 absent from the next hlist is an input error (a missing
hlist when desc_scale is not that hlist's scale, else an inconsistent tree). Real hlist rows carry the
Rockstar ID in Orig_halo_ID; phantom rows carry a number with no Rockstar halo behind it (counted in the
report). A particle ID in more than one of our exclusive sets is an input error.

Matching: mutual election (match_ids): our ten most-bound tracers vote with weights 1/(1 + sqrt(i)) for
the Rockstar halo listing them; every listed particle the PIG saves votes for our subhalo binding it.
Both one-directional votes are kept. Rockstar's lists leave out the kinematically cold core of every
large halo, so the most-bound particles of our large centrals vote for no Rockstar halo; two rules pair
what the election misses:
  central rule  an unmatched central c and the unmatched Rockstar host h whose listed particles bound in
                ours are mostly c's, when c's tracers vote for nobody (or c has none) or for a halo whose
                top host is h, and h's centre lies within c's RBoundSO[0].
  footprint     an unmatched satellite o and an unmatched Rockstar halo r holding >= 50 % of o's
                exclusive set, o being the plurality owner of r's listed particles bound to our
                satellites (from our side; from Rockstar's side r is a subhalo with that plurality).
Census pairs leave out our hostless tracks (as the compare tool); the identity steps use every pair.
Each pair carries its source (election, central-rule, footprint) in the tables and CSVs.

Destination tables, per output:
  Rockstar halo  its listed particles by our row binding them (BoundTrackId), -1 bound to nobody, -2 not
                 saved in the PIG: top destination (ties to the smaller value) and its share, shares of
                 -1 and -2.
  our subhalo    its exclusive bound set (the bound prefixes of its in-group and field slices) by the
                 Rockstar halo listing each particle, -1 none: top destination and share, share of -1.

Coverage (census): the share of our exclusive bound particles in any Rockstar list, for centrals and
satellites by size bin, and for centrals with Nbound >= 500 the listed share of their 1000 most-bound.

Satellite universe: ours Depth >= 1; Rockstar pid != -1 with Mvir / m_p >= --min-np (Mvir is Rockstar's
bound mass). Size bins 20-49, 50-99, 100-499, >= 500 by Nbound or Mvir / m_p.

Census categories (census outputs):
  A   matched, both satellites; same top host when the Rockstar partner of our CentralTrackId is the
      Rockstar top host (pid followed to -1). Ratios Mbound/Mvir, Vmax/vmax, centre offset (kpc/h and
      / Rvir), shared fraction |exclusive set & Rockstar list| / Nbound and / Np.
  B1  matched, ours satellite, Rockstar host: distance of our CorePos to the Rockstar partner of our
      central over its Rvir (> 1: FOF-linked beyond Rvir, a definition; <= 1: a disagreement).
  B2  matched, ours central, Rockstar subhalo: distance of our CorePos to Rockstar's top host over its
      Rvir, and that host's partner (a central of another FOF group of ours, a satellite, none).
  C   ours-only satellites by the top destination of the exclusive set: C1 the Rockstar partner of our
      central, C2s/C2h a Rockstar subhalo/host matched to another of ours, C3 an unmatched Rockstar
      halo, C4 no Rockstar halo. Host distance over Rvir (the Rockstar partner of the central, else
      our central's CorePos and RBoundSO[0]), Mbound/MboundPeak, Depth, RESCUED flag, age k - SnapAcc,
      phantom nearby (a phantom hlist row with pid != -1 within 0.5 of its Rvir, Mvir within a factor
      3 of Mbound).
  D   Rockstar-only subhaloes by the top destination of the list: D1 a central of ours (D1x hostless),
      D2s/D2h a satellite of ours matched to another Rockstar subhalo/host, D3 an unmatched satellite
      of ours, D4u bound to nobody, D4a not in the PIG. Orphan test (an orphan of ours with an
      OrphanIDs marker in the list and its TracerPos within the halo's Rvir: the most markers, then the
      latest death), independent unbinding of
      the list, host distance over the top host's Rvir, Rockstar bound fraction Mvir / (m_p Np).
  Phantoms of the hlist of k (pid != -1, Mvir / m_p >= --min-np) with one of our satellites within 0.5
      Rvir (mass factor 3).

Identity (anchors k0, steps j): every A pair (t, r) at k0, r followed through the hlists to k0 + j.
  E0  r has no row in the hlist of k0.
  E1  both alive and (t, chain row) a pair at k0 + j (election, central rule or footprint; the cases
      that are E1 only through an added pair are counted with the E2 label they would carry).
  E2  both alive, not paired with each other; self-overlaps |B_t(k0) & B_t(k0+j)| / |B_t(k0)| (exclusive
      sets) and |P_r(k0) & P_rj(k0+j)| / |P_r(k0)| (Rockstar lists); a side below 0.25 jumped; a phantom
      chain row has no list (its own attribution).
  E3  ours alive, Rockstar chain ended: t paired at k0 + j ("Rockstar identity break") or not
      ("Rockstar lost it, ours continues").
  E4  Rockstar alive, ours dead: phantom ("raw Rockstar lost it too"), paired with the track t sank
      into or its DescendantTrackId ("merger decision"), paired with another track ("other"), unpaired
      ("death Rockstar does not see": independent unbinding, orphan TracerPos inside Rvir).
  E5  both ended.

Independent unbinding: HBT-style iterative self-unbinding with Plummer softening box/512/30 (comoving),
G m_p / a potentials, physical peculiar velocities plus the Hubble flow H(a) a dx (H in km/s per kpc/h,
0.1 E(a)); the frame is the mean of CORE = 20 particles: on the first pass the listed particles nearest
Rockstar's centre (a Rockstar list is not in binding order), afterwards the 20 most bound. Converged means
a pass changes neither the bound set nor the frame particles; a list still changing after MAXIT = 40
passes is "unconverged" and counted apart. Self-bound means at least 20 bound at convergence. Lists
above --maxset particles are "skipped", lists with IDs missing from the PART "incomplete", empty lists
"empty", and lists spanning box/4 or more after one periodic wrap about their centre "noncompact"; all
four are counted apart and left out of the self-bound / not self-bound counts. The PART must be at the
PIG's Time (both finite), its kernel parameters finite and positive, and the gathered positions and
velocities and Rockstar's centres finite. PIGs must be complete (SubhaloStateComplete) and every header in MP-Gadget's default
units (kpc/h, km/s, 1e10 Msun/h).

Outputs in --out: census_<k>.txt, census_<k>_ours.csv, census_<k>_rockstar.csv, identity_<k0>.txt,
identity_<k0>.csv and summary.txt (every table; printed at the end). Progress goes to stderr.
"""

from __future__ import print_function

import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloFormatError, NO_ID  # noqa: E402
from subhalo_compare_hbt import Ours, _weighted_vote, _lookup, _q, _first_n_per_group  # noqa: E402
from subhalo_compare_rockstar import Rockstar, match_ids, Units, InputError  # noqa: E402

EDGES = np.array([20, 50, 100, 500])
BIN_NAMES = ("20-49", "50-99", "100-499", ">=500")
NTRACERS = 10
RESCUED = 1 << 5              # SUBHALO_FLAG_RESCUED
SELF_BOUND = 20               # the finder's resolution limit: a remnant with >= 20 bound is a subhalo
CORE = 20                     # frame particles of the unbinding
MAXIT = 40
JUMP = 0.25                   # self-overlap below which a side changed identity
FOOTPRINT = 0.5               # share of our exclusive set inside one Rockstar list for a footprint pair
SRC_ELECTION, SRC_CENTRAL, SRC_FOOTPRINT = 0, 1, 2
SRC_NAMES = {-1: "-", SRC_ELECTION: "election", SRC_CENTRAL: "central-rule", SRC_FOOTPRINT: "footprint"}
PAIR_CHUNK = 4000000          # pair distances per block of the unbinding potential (96 MB of differences)
HLIST_TOL = 2e-5              # hlist scales are written with five decimals; a PIG Time matches within this
T0 = time.time()


def log(*s):
    print("[%7.0f s]" % (time.time() - T0), *s, file=sys.stderr)
    sys.stderr.flush()


def size_bin(n):
    """bin index of each size, -1 below the first edge"""
    return np.searchsorted(EDGES, np.asarray(n, np.float64), side="right") - 1


def pdist(a, b, box):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d -= box * np.round(d / box)
    return np.sqrt((d ** 2).sum(axis=-1))


def top_per_group(group, cand, cnt):
    """For key arrays sorted by group: (groups, top candidate, its count, group total); ties to the
    smaller candidate."""
    if group.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    o = np.lexsort((cand, -cnt, group))
    g, c, n = group[o], cand[o], cnt[o]
    first = np.ones(g.size, bool)
    first[1:] = g[1:] != g[:-1]
    tot = np.add.reduceat(n, np.flatnonzero(first))
    return g[first], c[first], n[first], tot


def expand(starts, lengths):
    """Indices start..start+length-1 of every (start, length), concatenated."""
    lengths = np.asarray(lengths, np.int64)
    tot = int(lengths.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    off = np.repeat(np.asarray(starts, np.int64) - (np.cumsum(lengths) - lengths), lengths)
    return off + np.arange(tot, dtype=np.int64)


# ---------------------------------------------------------------------------------- hlists
class Hlists(object):
    """consistent-trees hlists in scale order, each read once; output k maps to the nearest scale."""
    INT = ("id", "desc_id", "pid", "upid", "mmp?", "phantom", "Orig_halo_ID")
    FLT = ("scale", "desc_scale", "Mvir", "Rvir", "x", "y", "z")

    def __init__(self, path):
        named = []
        for f in glob.glob(os.path.join(path, "hlist_*.list")):
            m = re.search(r"hlist_([0-9.]+)\.list$", f)
            if m:
                named.append((float(m.group(1)), f))
        if not named:
            raise InputError("no hlist_*.list under %s" % path)
        named.sort()
        self.scales = np.array([a for a, _ in named])
        self.files = [f for _, f in named]
        self.data, self.nxt = {}, {}

    def index_of(self, a, k=None):
        """the nearest hlist, accepted within HLIST_TOL in a"""
        i = int(np.argmin(np.abs(self.scales - a)))
        if not abs(self.scales[i] - a) <= HLIST_TOL:
            raise InputError("output %s (a = %.6f): nearest hlist scale %.5f (%s) is %.2e away, above %.2e"
                             % (k, a, self.scales[i], self.files[i], abs(self.scales[i] - a), HLIST_TOL))
        return i

    def read(self, i):
        if i in self.data:
            return self.data[i]
        fn = self.files[i]
        with open(fn) as f:
            head = f.readline()
            if not head.startswith("#"):
                raise InputError("%s has no header line" % fn)
            names = [re.sub(r"\(\d+\)$", "", t) for t in head[1:].split()]
            try:
                col = {c: names.index(c) for c in self.INT + self.FLT}
            except ValueError as e:
                raise InputError("%s: %s" % (fn, e))
            m = max(col.values()) + 1
            rows = [line.split(None, m)[:m] for line in f if line[0] != "#"]
        tab = np.array(rows) if rows else np.zeros((0, m), "U1")
        del rows
        d = {}
        for c in self.INT:
            d[c] = tab[:, col[c]].astype(np.int64)
        for c in self.FLT:
            d[c] = tab[:, col[c]].astype(np.float64)
        del tab
        if d["scale"].size and np.abs(d["scale"] - self.scales[i]).max() > 1e-5:
            raise InputError("%s holds rows of another scale" % fn)
        o = np.argsort(d["id"], kind="stable")
        d["id_sorted"], d["id_row"] = d["id"][o], o.astype(np.int64)
        real = np.flatnonzero(d["phantom"] == 0)
        o = real[np.argsort(d["Orig_halo_ID"][real], kind="stable")]
        d["orig_sorted"], d["orig_row"] = d["Orig_halo_ID"][o], o.astype(np.int64)
        d["n"] = d["id"].size
        self.data[i] = d
        log("hlist %s: %d rows, %d phantom" % (os.path.basename(fn), d["n"], int(np.count_nonzero(d["phantom"] != 0))))
        return d

    def row_of_rockstar(self, i, ids):
        """the real row of hlist i whose Orig_halo_ID is each Rockstar ID, -1 where none"""
        d = self.read(i)
        return _lookup(d["orig_sorted"], d["orig_row"], np.asarray(ids, np.int64))

    def next_row(self, i):
        """row of hlist i + 1 continuing each row of hlist i as its main progenitor, -1 where none; a
        desc_id >= 0 that hlist i + 1 does not hold is an input error, not a chain end"""
        if i not in self.nxt:
            d, e = self.read(i), self.read(i + 1)
            r = _lookup(e["id_sorted"], e["id_row"], d["desc_id"])
            lost = (r < 0) & (d["desc_id"] >= 0)
            if lost.any():
                other = int(np.count_nonzero(lost & ~(np.abs(d["desc_scale"] - self.scales[i + 1]) <= HLIST_TOL)))
                raise InputError("%s: %d rows have a desc_id absent from the next hlist %s: %d with desc_scale not "
                                 "%.5f (a missing hlist), %d at that scale (an inconsistent tree)"
                                 % (self.files[i], int(lost.sum()), os.path.basename(self.files[i + 1]), other,
                                    self.scales[i + 1], int(lost.sum()) - other))
            self.nxt[i] = np.where((r >= 0) & (d["mmp?"] == 1), r, -1)
        return self.nxt[i]

    def advance(self, rows, i, ib):
        for m in range(i, ib):
            rows = np.where(rows >= 0, self.next_row(m)[np.maximum(rows, 0)], -1)
        return rows


DEFAULT_UNITS = (("UnitLength_in_cm", 3.085678e21), ("UnitVelocity_in_cm_per_s", 1e5), ("UnitMass_in_g", 1.989e43))


def check_units(header, label):
    """the kernel and the tables assume MP-Gadget's default units (kpc/h, km/s, 1e10 Msun/h)"""
    keys = set(header.keys())
    for k, ref in DEFAULT_UNITS:
        if k not in keys:
            raise InputError("%s: no %s in the Header" % (label, k))
        v = float(np.atleast_1d(header[k])[0])
        if not abs(v - ref) <= 1e-6 * ref:          # a NaN fails too
            raise InputError("%s: %s = %.7g, the tool assumes %.7g" % (label, k, v, ref))


# ---------------------------------------------------------------------------------- unbinding
KP = {}   # kernel parameters of the PART being unbound (module level so that forked workers see them)


class NonCompactList(InputError):
    """a list spanning box/4 or more after one periodic wrap: its own status, not a run error"""


def kernel_params(part):
    from bigfile import File
    with File(part) as f:
        h = f["Header"].attrs
        g = lambda k: float(np.atleast_1d(h[k])[0])  # noqa: E731
        a, box, Om, OL, mpart = g("Time"), g("BoxSize"), g("Omega0"), g("OmegaLambda"), float(h["MassTable"][1])
        pec = int(np.atleast_1d(h["UsePeculiarVelocity"])[0]) if "UsePeculiarVelocity" in h.keys() else 1
        check_units(h, part)
    # the header values first: the derived parameters divide by a
    for k, val in (("Time", a), ("BoxSize", box), ("MassTable[1]", mpart)):
        if not (np.isfinite(val) and val > 0):
            raise InputError("%s: header %s = %r is not finite and positive" % (part, k, val))
    # UsePeculiarVelocity = 0 snapshots store a v_pec (libgadget/petaio.c)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        kp = dict(a=a, box=box, mpart=mpart, vfac=1.0 if pec else 1.0 / a, pec=pec,
                  eps=box / 512.0 / 30.0,                     # GravitySoftening: 1/30 of the mean separation
                  Ha=0.1 * np.sqrt(Om / np.float64(a) ** 3 + OL) * a,   # H(a) a in km/s per comoving kpc/h
                  GMa=43009.1 * mpart / a)                    # G m_p / a in (km/s)^2 comoving kpc/h
    for k in ("a", "box", "mpart", "eps", "Ha", "GMa", "vfac"):
        if not (np.isfinite(kp[k]) and kp[k] > 0):
            raise InputError("%s: kernel parameter %s = %r is not finite and positive" % (part, k, kp[k]))
    return kp


def gather(part, need):
    """positions (kpc/h) and physical peculiar velocities (km/s) of the sorted unique IDs `need`;
    the found mask. Reads the PART once and keeps only the rows asked for."""
    from bigfile import File
    with File(part) as f:
        ids = f["1/ID"][:].astype(np.int64)
        p = np.searchsorted(need, ids)
        p[p >= need.size] = 0
        hit = np.flatnonzero(need[p] == ids) if need.size else np.zeros(0, np.int64)
        which = p[hit]
        del ids, p
        x = np.zeros((need.size, 3))
        v = np.zeros((need.size, 3))
        x[which] = f["1/Position"][:][hit]
        v[which] = f["1/Velocity"][:][hit].astype(np.float64) * KP["vfac"]
    bad = int(np.count_nonzero(~(np.isfinite(x[which]).all(1) & np.isfinite(v[which]).all(1))))
    if bad:
        raise InputError("%s: %d of the gathered IDs have a non-finite position or velocity" % (part, bad))
    found = np.zeros(need.size, bool)
    found[which] = True
    return x, v, found


def potential(x, xb):
    """-G m_p / a sum_j 1/sqrt(|x - xb_j|^2 + eps^2) for every row of x, in blocks"""
    out = np.empty(len(x))
    eps2 = KP["eps"] ** 2
    step = max(1, PAIR_CHUNK // max(len(xb), 1))
    for s in range(0, len(x), step):
        d = x[s:s + step, None, :] - xb[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        out[s:s + step] = -KP["GMa"] * (1.0 / np.sqrt(r2 + eps2)).sum(1)
    return out


def unbind(x, v, centre=None, core=CORE, maxit=MAXIT, key=None):
    """HBT-style iterative self-unbinding of one set; returns (bound mask, converged). The first-pass
    frame is the first `core` particles of the list or, with `centre`, the `core` nearest to it; later
    passes use the `core` most bound. Converged: a pass leaves both the bound set and the frame
    particles unchanged (a set left with fewer than two bound particles has converged to none)."""
    N = len(x)
    if N < 2:
        return np.zeros(N, bool), True
    box = KP["box"]
    x = x - (x[0] if centre is None else centre)
    x -= box * np.rint(x / box)
    # one wrap about one centre: the list must be compact against the box
    ext = float((x.max(0) - x.min(0)).max())
    if ext >= box / 4:
        raise NonCompactList("unbinding list %s spans %.1f kpc/h after wrapping about its centre (box %.1f kpc/h)"
                             % (key, ext, box))
    order = np.arange(N) if centre is None else np.argsort((x * x).sum(1), kind="stable")
    bound = np.ones(N, bool)
    self_term = KP["GMa"] / KP["eps"]
    for _ in range(maxit):
        xb = x[bound]
        if len(xb) < 2:
            return np.zeros(N, bool), True
        phi = potential(x, xb)
        phi[bound] += self_term                      # a bound particle does not bind itself
        c = order[:min(core, len(order))]
        xf, vf = x[c].mean(0), v[c].mean(0)
        dv = v - vf + KP["Ha"] * (x - xf)
        E = 0.5 * (dv * dv).sum(1) + phi
        newb = E < 0
        order = np.argsort(E, kind="stable")         # most bound first, the next frame
        order = order[newb[order]] if newb.any() else order
        cn = order[:min(core, len(order))]
        if (newb == bound).all() and np.array_equal(np.sort(c), np.sort(cn)):
            return newb, True
        bound = newb
        if bound.sum() < 2:
            return np.zeros(N, bool), True           # the next pass would unbind a lone survivor
    return bound, False


def _unbind_task(task):
    """(key, nbound, status) of one set; serial and worker paths both run it"""
    key, x, v, centre = task
    try:
        m, conv = unbind(x, v, centre=centre, key=key)
    except NonCompactList:
        return key, -1, "noncompact"
    return key, int(m.sum()), "done" if conv else "unconverged"


def run_unbinding(part, sets, nproc, maxset, a_pig):
    """sets: {key: (ids, centre kpc/h)}; returns {key: (nlist, nfound, nbound, status)}, status "done",
    "unconverged" (MAXIT passes), "skipped" (more than maxset IDs), "incomplete" (IDs missing from the
    PART), "empty" (no ID) or "noncompact" (spans box/4 or more); nbound -1 for the last four."""
    res = {}
    if not sets:
        return res
    # a NaN centre would wrap every offset to NaN and leave no particle bound
    badc = [key for key, (ids, centre) in sets.items() if centre is not None and not np.isfinite(centre).all()]
    if badc:
        raise InputError("%d unbinding lists have a non-finite Rockstar centre (e.g. list %r at %s)"
                         % (len(badc), badc[0], sets[badc[0]][1]))
    KP.clear()
    KP.update(kernel_params(part))
    if not (np.isfinite(a_pig) and a_pig > 0 and abs(KP["a"] - a_pig) <= 1e-6 * a_pig):
        raise InputError("%s is at a = %.8f, the PIG at a = %.8f" % (part, KP["a"], a_pig))
    need = np.unique(np.concatenate([s[0] for s in sets.values()]))
    t0 = time.time()
    x, v, found = gather(part, need)
    log("  PART %s: a = %.4f, %d IDs gathered (%d not found) in %.0f s" % (
        os.path.basename(os.path.realpath(part)), KP["a"], need.size, int(np.count_nonzero(~found)), time.time() - t0))
    tasks = []
    for key, (ids, centre) in sets.items():
        p = np.searchsorted(need, ids)
        ok = found[p]
        nf = int(np.count_nonzero(ok))
        if ids.size == 0:
            res[key] = (0, 0, -1, "empty")
            continue
        if ids.size > maxset:
            res[key] = (ids.size, nf, -1, "skipped")
            continue
        if nf < ids.size:
            res[key] = (ids.size, nf, -1, "incomplete")
            continue
        if nf < 2:
            res[key] = (ids.size, nf, 0, "done")
            continue
        res[key] = (ids.size, nf, None, None)
        tasks.append((key, x[p[ok]], v[p[ok]], centre))
    del x, v, found, need
    tasks.sort(key=lambda t: -len(t[1]))
    t0 = time.time()
    if nproc > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(nproc) as pool:
            for key, nb, st in pool.imap_unordered(_unbind_task, tasks, chunksize=1):
                res[key] = res[key][:2] + (nb, st)
    else:
        for t in tasks:
            key, nb, st = _unbind_task(t)
            res[key] = res[key][:2] + (nb, st)
    nst = lambda st: sum(1 for r in res.values() if r[3] == st)  # noqa: E731
    log("  unbinding: %d sets (largest %d) in %.0f s; unconverged %d, skipped above %d: %d, incomplete %d, "
        "empty %d, noncompact %d" % (len(tasks), len(tasks[0][1]) if tasks else 0, time.time() - t0,
                                     nst("unconverged"), maxset, nst("skipped"), nst("incomplete"), nst("empty"),
                                     nst("noncompact")))
    return res


# ---------------------------------------------------------------------------------- one output
def check_rockstar_ids(rs):
    """the catalogue parser reads halo and parent IDs through float64: each value read must be an integer
    below 2**53 in magnitude, where float64 is exact, for the int64 IDs to be the catalogue's"""
    bad, ex = 0, None
    for col in ("id", "pid"):
        if col not in rs.columns:
            continue                             # pid by the parents.c rule is built from the checked IDs
        f = rs.columns[col]
        with np.errstate(invalid="ignore"):
            m = ~(np.isfinite(f) & (np.abs(f) < 2 ** 53) & (f == np.floor(f)))
        bad += int(np.count_nonzero(m))
        if ex is None and m.any():
            ex = "%s %r" % (col, float(f[np.flatnonzero(m)[0]]))
    if bad:
        raise InputError("%s: %d halo or parent IDs are not integers below 2**53 in magnitude, not exact through "
                         "float64 (e.g. %s as read)" % (rs.catfile, bad, ex))


def parent_rows(rs):
    """catalogue row of each halo's parent, -1 for a host; a pid >= 0 absent from the catalogue is an input
    error (the top-host walk would stop at the orphaned halo and take it for a host)"""
    prow = np.where(rs.pid >= 0, rs.row(rs.pid), -1)
    miss = np.flatnonzero((rs.pid >= 0) & (prow < 0))
    if miss.size:
        raise InputError("%s: %d halos have a pid absent from the catalogue (e.g. halo %d with pid %d)"
                         % (rs.catfile, miss.size, int(rs.id[miss[0]]), int(rs.pid[miss[0]])))
    return prow


class Output(object):
    """Our catalogue and Rockstar's at one output, the votes, pairs and destination tables."""

    def __init__(self, k, args):
        t0 = time.time()
        self.k = k
        self.pigdir = os.path.join(args.ours, "PIG_%03d" % k)
        self.part = os.path.join(args.ours, "PART_%03d" % k)
        ours = self.ours = Ours(self.pigdir, NTRACERS)
        if ours.snap != k:
            raise InputError("%s has SnapshotIndex %d" % (self.pigdir, ours.snap))
        if not ours.cat.complete():
            raise InputError("%s: the finder state is not complete (SubhaloStateComplete != 1)" % self.pigdir)
        check_units(ours.cat.header, self.pigdir)
        u = self.units = Units(ours.cat.header)
        self.a = u.a
        self.box = float(np.atleast_1d(ours.cat.header["BoxSize"])[0]) * u.rad      # kpc/h
        self.mp = u.mpart                                                         # Msun/h
        rs = self.rs = Rockstar(args.rockstar, k + args.rs_snap_offset, need_ids=True, box=u.box_mpc)
        if not rs.bin_files:
            raise InputError("no halos_*.bin of snapshot %d under %s" % (rs.snap, args.rockstar))
        if np.isfinite(rs.a) and abs(rs.a - u.a) > 1e-3 * u.a:
            raise InputError("%s is at a = %.5f, the PIG at a = %.5f" % (rs.catfile, rs.a, u.a))
        check_rockstar_ids(rs)
        if rs.nduplicate:
            raise InputError("%s: %d particle IDs listed more than once in the halos_*.bin lists (ambiguous "
                             "ownership)" % (rs.catfile, rs.nduplicate))
        log("output %d (a = %.4f): read %d subhalos, %d orphans, %d Rockstar halos, %d listed IDs in %.0f s" % (
            k, u.a, ours.tid.size, ours.otid.size, rs.n, rs.pids.size, time.time() - t0))
        S = ours.S
        self.nO, self.nR = ours.tid.size, rs.n
        self.depth = S["Depth"].astype(np.int64)
        self.host = S["HostGroup"].astype(np.int64)
        self.sat = self.depth >= 1
        self.cen = (self.depth == 0) & (self.host >= 0)
        self.hostless = self.host < 0
        self.nbound = ours.nbound
        self.pos = S["CorePos"].astype(np.float64)
        self.mbound = S["Mbound"].astype(np.float64) * u.mass
        self.rvir = S["RBoundSO"][:, 0].astype(np.float64) * u.rad
        self.cen_row = ours.row(S["CentralTrackId"].astype(np.int64))
        # Rockstar side in kpc/h
        self.rpos = rs.pos * 1e3
        self.rsize = rs.mvir / self.mp
        self.sub = rs.pid != -1
        self.np_rs = rs.columns["np"].astype(np.int64) if "np" in rs.columns else np.zeros(rs.n, np.int64)
        # with unique membership a listed halo holds all its particles: Np > 0 and none listed is a lost chunk
        nolist = int(np.count_nonzero((self.np_rs > 0) & (np.bincount(rs.p_owner_row, minlength=rs.n) == 0)))
        if nolist:
            raise InputError("%s: %d halos with Np > 0 have no particle in the %d halos_*.bin files read (a missing "
                             "chunk)" % (rs.catfile, nolist, len(rs.bin_files)))
        prow = parent_rows(rs)
        self.pid_missing = 0                     # parent_rows rejects a pid absent from the catalogue
        # top host by pointer doubling; a top that still has a parent lies on a pid cycle
        top = np.where(prow >= 0, prow, np.arange(rs.n))
        for _ in range(64):
            nxt = top[top]
            if np.array_equal(nxt, top):
                break
            top = nxt
        cyc = int(np.count_nonzero(prow[top] >= 0)) if rs.n else 0
        if cyc:
            raise InputError("%s: %d halos whose pid chain reaches no host (a pid cycle)" % (rs.catfile, cyc))
        self.top = top
        t1 = time.time()
        self._exclusive()
        self._votes()
        self._rs_lists()
        log("output %d: exclusive sets, votes, pairs and destinations in %.0f s" % (k, time.time() - t1))

    def _exclusive(self):
        """exclusive bound sets: the bound prefix of the in-group slice and of the field slice"""
        S = self.ours.S
        ids = self.ours.cat.file["1/ID"][:].astype(np.int64)
        parts_i, parts_r = [], []
        for pre in ("Slice", "FieldSlice"):
            if pre + "OffsetByType" not in S:
                continue
            off = S[pre + "OffsetByType"][:, 1].astype(np.int64)
            nb = S[pre + "NboundByType"][:, 1].astype(np.int64)
            ok = (off >= 0) & (nb > 0)
            nb = np.where(ok, nb, 0)
            parts_i.append(ids[expand(off[ok], nb[ok])])
            parts_r.append(np.repeat(np.flatnonzero(ok), nb[ok]))
        del ids
        ex_id = np.concatenate(parts_i)
        ex_row = np.concatenate(parts_r)
        # row-grouped copy (in-group then field slice of each row) for the identity sets
        o = np.argsort(ex_row, kind="stable")
        self.ex_id_by_row, self.ex_row_by_row = ex_id[o], ex_row[o]
        self.ex_start = np.concatenate([[0], np.cumsum(np.bincount(ex_row, minlength=self.nO))])
        o = np.argsort(ex_id, kind="stable")
        self.ex_sorted, self.ex_owner = ex_id[o], ex_row[o]
        self.ex_dup = int(np.count_nonzero(self.ex_sorted[1:] == self.ex_sorted[:-1]))
        if self.ex_dup:
            rep = self.ex_sorted[1:] == self.ex_sorted[:-1]
            raise InputError("%s: %d particle IDs repeated in the exclusive bound sets, %d of them in sets of "
                             "different rows (ambiguous ownership)" % (self.pigdir, self.ex_dup, int(np.count_nonzero(
                                 rep & (self.ex_owner[1:] != self.ex_owner[:-1])))))
        self.nexcl = np.bincount(ex_row, minlength=self.nO)

    def ex_set(self, row):
        return self.ex_id_by_row[self.ex_start[row]:self.ex_start[row + 1]]

    def ex_owner_of(self, ids):
        return _lookup(self.ex_sorted, self.ex_owner, ids)

    def _votes(self):
        ours, rs = self.ours, self.rs
        # Rockstar members: our row binding each listed particle, -1 nobody, -2 not saved in the PIG
        saved = ours.types(rs.pids) >= 0
        dest = np.full(rs.pids.size, -2, np.int64)
        dest[saved] = ours.row(ours.owner(rs.pids[saved]))
        del saved
        cand = rs.owner_row(ours.tr_id)
        v, w = _weighted_vote(ours.row(ours.tr_track), cand, 1.0 / (1.0 + np.sqrt(ours.tr_pos)))
        self.vote_o = np.full(self.nO, -3, np.int64)
        self.vote_o[v] = w
        self._rs_destinations(dest)
        del dest
        # mutual pairs from the compare tool, checked against the two votes
        o, r = match_ids(ours, rs, NTRACERS)
        mo = np.flatnonzero((self.vote_o >= 0) & (self.vote_r[np.maximum(self.vote_o, 0)] == np.arange(self.nO)))
        self.match_check = (o.size == mo.size) and np.array_equal(np.sort(o), mo) and \
            np.array_equal(r[np.argsort(o)], self.vote_o[mo])
        self.o2r_all = np.full(self.nO, -1, np.int64)
        self.r2o_all = np.full(self.nR, -1, np.int64)
        self.o2r_all[o], self.r2o_all[r] = r, o
        self.osrc = np.full(self.nO, -1, np.int64)           # source of each of our rows' pair
        self.osrc[o] = SRC_ELECTION
        self.o2r_el_all, self.r2o_el_all = self.o2r_all.copy(), self.r2o_all.copy()
        self.n_election = o.size
        self.npairs_hostless = int(np.count_nonzero(self.hostless[o]))
        self._ours_destinations()
        self._extra_pairs()
        # census pairs leave out our hostless tracks
        o = np.flatnonzero((self.o2r_all >= 0) & ~self.hostless)
        r = self.o2r_all[o]
        self.po, self.pr, self.psrc = o, r, self.osrc[o]
        self.o2r = np.full(self.nO, -1, np.int64)
        self.r2o = np.full(self.nR, -1, np.int64)
        self.o2r[o], self.r2o[r] = r, o

    def _rs_destinations(self, dest):
        """Rockstar destination table; the Rockstar vote (every saved listed particle one vote for our
        row binding it, -1 included, ties to the smaller) is its top entry without the -2 rows"""
        rs = self.rs
        base = self.nO + 2
        key = rs.p_owner_row * base + (dest + 2)
        uk, cnt = np.unique(key, return_counts=True)
        del key
        g, c = uk // base, uk % base - 2
        del uk
        self.r_nlist = np.bincount(g, weights=cnt, minlength=self.nR).astype(np.int64)
        self.r_unb = np.bincount(g[c == -1], weights=cnt[c == -1], minlength=self.nR).astype(np.int64)
        self.r_abs = np.bincount(g[c == -2], weights=cnt[c == -2], minlength=self.nR).astype(np.int64)
        gg, tc, tn, _ = top_per_group(g, c, cnt)
        self.r_top = np.full(self.nR, -3, np.int64)
        self.r_topn = np.zeros(self.nR, np.int64)
        self.r_top[gg], self.r_topn[gg] = tc, tn
        sv = c != -2
        gg, tc, _, _ = top_per_group(g[sv], c[sv], cnt[sv])
        self.vote_r = np.full(self.nR, -3, np.int64)       # -3: no saved particle, no vote
        self.vote_r[gg] = tc
        # plurality owner among our rows, and among our satellites, of each list
        rw = c >= 0
        gg, tc, tn, _ = top_per_group(g[rw], c[rw], cnt[rw])
        self.r_rowplur = np.full(self.nR, -1, np.int64)
        self.r_rowplur_n = np.zeros(self.nR, np.int64)
        self.r_rowplur[gg], self.r_rowplur_n[gg] = tc, tn
        sw = np.zeros(c.size, bool)
        sw[rw] = self.sat[c[rw]]
        gg, tc, _, _ = top_per_group(g[sw], c[sw], cnt[sw])
        self.r_satplur = np.full(self.nR, -1, np.int64)
        self.r_satplur[gg] = tc

    def _ours_destinations(self):
        """our destination table over the exclusive sets"""
        rs = self.rs
        rso = rs.owner_row(self.ex_id_by_row)
        base = self.nR + 1
        key = self.ex_row_by_row * base + (rso + 1)
        del rso
        self.o_uk, self.o_cnt = np.unique(key, return_counts=True)
        del key
        g, c = self.o_uk // base, self.o_uk % base - 1
        self.o_none = np.bincount(g[c == -1], weights=self.o_cnt[c == -1], minlength=self.nO).astype(np.int64)
        gg, tc, tn, _ = top_per_group(g, c, self.o_cnt)
        self.o_top = np.full(self.nO, -2, np.int64)
        self.o_topn = np.zeros(self.nO, np.int64)
        self.o_top[gg], self.o_topn[gg] = tc, tn

    def _extra_pairs(self):
        """Pairs the mutual election misses: Rockstar's lists leave out the kinematically cold cores of
        large halos, so their most-bound particles vote for no Rockstar halo.
        Central rule: an unmatched central c and the unmatched Rockstar host h whose listed particles
        bound in ours are mostly c's, when c's tracers vote for nobody (or c has none) or for a halo whose
        top host is h, and h's centre lies within c's RBoundSO[0]; per central the host with the most
        such particles.
        Footprint: an unmatched satellite o and an unmatched Rockstar halo r holding >= FOOTPRINT of o's
        exclusive set, o being the plurality owner of r's listed particles bound to our satellites;
        found from our side (r = o's top destination) and from Rockstar's (r a subhalo, o its
        satellite plurality)."""
        o2r, r2o = self.o2r_all, self.r2o_all
        h = np.flatnonzero(~self.sub & (r2o < 0) & (self.r_rowplur >= 0))
        c = self.r_rowplur[h]
        ok = self.cen[c] & (o2r[c] < 0)
        h, c = h[ok], c[ok]
        v = self.vote_o[c]
        ok = (v < 0) | (self.top[np.maximum(v, 0)] == h)
        ok &= (self.rvir[c] > 0) & (pdist(self.rpos[h], self.pos[c], self.box) <= self.rvir[c])
        h, c = h[ok], c[ok]
        o = np.lexsort((h, -self.r_rowplur_n[h], c))
        h, c = h[o], c[o]
        first = np.ones(c.size, bool)
        first[1:] = c[1:] != c[:-1]
        h, c = h[first], c[first]
        o2r[c], r2o[h] = h, c
        self.osrc[c] = SRC_CENTRAL
        self.n_central_rule = int(c.size)
        nex = np.maximum(self.nexcl, 1).astype(np.float64)
        so = np.flatnonzero(self.sat & (o2r < 0) & (self.o_top >= 0))
        r = self.o_top[so]
        ok = (self.o_topn[so] / nex[so] >= FOOTPRINT) & (r2o[r] < 0) & (self.r_satplur[r] == so)
        fo, fr = so[ok], r[ok]
        rr = np.flatnonzero(self.sub & (r2o < 0) & (self.r_satplur >= 0))
        oo = self.r_satplur[rr]
        ok = (o2r[oo] < 0) & (self.inter(oo, rr) / nex[oo] >= FOOTPRINT)
        go, gr = oo[ok], rr[ok]
        pair = dict(zip(fo.tolist(), fr.tolist()))
        self.n_fp_ours, self.n_fp_rs = int(fo.size), int(go.size)
        self.n_fp_both = sum(1 for a, b in zip(go.tolist(), gr.tolist()) if pair.get(a) == b)
        used = set(fr.tolist())
        for a, b in zip(go.tolist(), gr.tolist()):
            if a not in pair and b not in used:
                pair[a] = b
                used.add(b)
        fo = np.array(sorted(pair), np.int64)
        fr = np.array([pair[a] for a in fo.tolist()], np.int64)
        o2r[fo], r2o[fr] = fr, fo
        self.osrc[fo] = SRC_FOOTPRINT
        self.n_fp = int(fo.size)

    def inter(self, o, r):
        """|exclusive set of our row o & Rockstar list of r| per pair"""
        key = np.asarray(o, np.int64) * (self.nR + 1) + (np.asarray(r, np.int64) + 1)
        p = np.searchsorted(self.o_uk, key)
        p = np.minimum(p, max(self.o_uk.size - 1, 0))
        hit = self.o_uk[p] == key if self.o_uk.size else np.zeros(key.size, bool)
        return np.where(hit, self.o_cnt[p], 0)

    def _rs_lists(self):
        o = np.argsort(self.rs.p_owner_row, kind="stable")
        self.rl_ids = self.rs.pids[o]
        self.rl_start = np.concatenate([[0], np.cumsum(np.bincount(self.rs.p_owner_row, minlength=self.nR))])

    def rs_list(self, r):
        return self.rl_ids[self.rl_start[r]:self.rl_start[r + 1]]

    def host_dist(self, o):
        """our row o to its host over R: the Rockstar partner of our central (its centre and Rvir), else
        our central (CorePos, RBoundSO[0]); (d/R, source) with source 'rs', 'ours' or 'none'"""
        c = self.cen_row[o]
        if c < 0:
            return np.nan, "none"
        cp = self.o2r[c]
        if cp >= 0 and self.rs.rvir_kpc[cp] > 0:
            return pdist(self.pos[o], self.rpos[cp], self.box) / self.rs.rvir_kpc[cp], "rs"
        if self.rvir[c] > 0:
            return pdist(self.pos[o], self.pos[c], self.box) / self.rvir[c], "ours"
        return np.nan, "none"

    def tid_of(self, row):
        return int(self.ours.tid[row]) if row >= 0 else int(row)

    def rid_of(self, r):
        return int(self.rs.id[r]) if r >= 0 else int(r)

    def drop(self):
        """free the particle-level arrays; the tables, votes and destinations stay"""
        self.ours.drop_particles()
        rs = self.rs
        rs.pids = rs.p_owner = rs.p_owner_row = None
        self.ex_id_by_row = self.ex_row_by_row = self.ex_sorted = self.ex_owner = None
        self.rl_ids = None
        self.o_uk = self.o_cnt = None


# ---------------------------------------------------------------------------------- census
def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else "%.5g" % v
    return str(v)


def bin_line(label, b, sel, width=28):
    sel = np.asarray(sel, bool)
    parts = ["%7d" % int(np.count_nonzero(sel & (b == x))) for x in range(len(EDGES))]
    below = int(np.count_nonzero(sel & (b < 0)))
    return "  %-*s %7d |%s%s" % (width, label, int(np.count_nonzero(sel)), "".join(parts),
                                 "   (+%d below %d)" % (below, EDGES[0]) if below else "")


def bin_header(width=28, what="bins"):
    return "  %-*s %7s |%s" % (width, what, "all", "".join("%7s" % n for n in BIN_NAMES))


def qline(label, x, width=34):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return "    %-*s n=%-6d p10/p50/p90 %s" % (width, label, x.size, _q(x))


class Census(object):
    def __init__(self, s, hl, args):
        self.s, self.args = s, args
        self.k = s.k
        self.lines = []
        self.unbind_sets = {}
        rs, ours = s.rs, s.ours
        S = ours.S
        mn = args.min_np
        P = self.lines.append
        po, pr = s.po, s.pr
        bo = size_bin(s.nbound)
        br = size_bin(s.rsize)
        univ_r = s.sub & (s.rsize >= mn)
        univ_o = s.sat
        P("census at output %d (a = %.4f): %s, Rockstar %s" % (s.k, s.a, s.pigdir, rs.catfile))
        P("  ours: %d rows, %d satellites (Depth >= 1), %d centrals, %d hostless; Rockstar: %d halos, %d subhaloes "
          "(pid != -1, %s), %d with Mvir/m_p >= %d; m_p = %.6g Msun/h" % (
              s.nO, int(univ_o.sum()), int(s.cen.sum()), int(s.hostless.sum()), s.nR, int(s.sub.sum()), rs.pid_source,
              int(univ_r.sum()), mn, s.mp))
        P("  mutual pairs %d (%s match_ids and the two votes), %d with a hostless track of ours left out; "
          "exclusive sets: %d IDs in more than one set; Rockstar: %d IDs in more than one list, %d pids not in the catalogue"
          % (s.n_election, "identical in" if s.match_check else "DIFFERENT between",
             s.npairs_hostless, s.ex_dup, rs.nduplicate, s.pid_missing))
        P("  added pairs: central rule %d; footprint %d (from our side %d, from Rockstar's %d, found from both %d); "
          "census pairs %d = election %d + central rule %d + footprint %d" % (
              s.n_central_rule, s.n_fp, s.n_fp_ours, s.n_fp_rs, s.n_fp_both, po.size,
              int(np.count_nonzero(s.psrc == SRC_ELECTION)), int(np.count_nonzero(s.psrc == SRC_CENTRAL)),
              int(np.count_nonzero(s.psrc == SRC_FOOTPRINT))))
        cen_un = s.cen & (s.o2r < 0)
        P("  our centrals unmatched: %d of %d (Nbound >= 500: %d of %d)" % (
            int(cen_un.sum()), int(s.cen.sum()), int(np.count_nonzero(cen_un & (s.nbound >= 500))),
            int(np.count_nonzero(s.cen & (s.nbound >= 500)))))
        # coverage of our exclusive bound sets by Rockstar's lists
        lst = (s.nexcl - s.o_none).astype(np.float64)
        P("  coverage: share of our exclusive bound particles in any Rockstar list")
        for lab, m in (("centrals", s.cen), ("satellites", s.sat)):
            P("    %-10s all %.4f (%d of %d); by Nbound bin %s" % (
                lab, lst[m].sum() / max(s.nexcl[m].sum(), 1), int(lst[m].sum()), int(s.nexcl[m].sum()),
                ", ".join("%s %.4f" % (bn, lst[m & (bo == x)].sum() / max(s.nexcl[m & (bo == x)].sum(), 1))
                          for x, bn in enumerate(BIN_NAMES))))
        bigc = np.flatnonzero(s.cen & (s.nbound >= 500))
        if bigc.size:
            # the in-group and field bound prefixes merged by BindRank
            idx, own = [], []
            for pre in ("Slice", "FieldSlice"):
                if pre + "OffsetByType" not in S:
                    continue
                off = S[pre + "OffsetByType"][bigc, 1].astype(np.int64)
                nbp = S[pre + "NboundByType"][bigc, 1].astype(np.int64)
                ok = (off >= 0) & (nbp > 0)
                idx.append(expand(off[ok], nbp[ok]))
                own.append(np.repeat(np.flatnonzero(ok), nbp[ok]))
            idx, own = np.concatenate(idx), np.concatenate(own)
            rank = ours.cat.file["1/BindRank"][:][idx]
            o = np.lexsort((rank, own))
            idx, own = idx[o], own[o]
            keep = _first_n_per_group(own, 1000)
            idx, own = idx[keep], own[keep]
            listed = (rs.owner_row(ours.cat.file["1/ID"][:][idx].astype(np.int64)) >= 0).astype(float)
            ln = np.bincount(own, minlength=bigc.size)
            f1000 = np.bincount(own, weights=listed, minlength=bigc.size) / np.maximum(ln, 1)
            P("    centrals with Nbound >= 500: %d; listed fraction of their 1000 most-bound particles (bound prefixes "
              "merged by BindRank) median %.4f, p10/p50/p90 %s" % (bigc.size, float(np.median(f1000)), _q(f1000)))
        npdiff = int(np.count_nonzero(s.np_rs != s.r_nlist))
        P("  Rockstar Np equals the listed particle count for %d of %d halos" % (s.nR - npdiff, s.nR))
        oc, rh = s.cen[po], ~s.sub[pr]
        osat = s.sat[po]
        kinds = [("central-host", oc & rh), ("central-sub", oc & ~rh), ("satellite-host", osat & rh),
                 ("satellite-sub", osat & ~rh)]
        P("  pairs by kind: %s (other: %d)" % (", ".join("%s %d" % (n, int(m.sum())) for n, m in kinds),
                                                 int(np.count_nonzero(~oc & ~osat))))
        for src in (SRC_ELECTION, SRC_CENTRAL, SRC_FOOTPRINT):
            ms = s.psrc == src
            P("    %-12s %s" % (SRC_NAMES[src], ", ".join("%s %d" % (n, int(np.count_nonzero(m & ms))) for n, m in kinds)))

        # ---------------- A
        Am = osat & ~rh
        ao, ar = po[Am], pr[Am]
        asrc = s.psrc[Am]
        cp = np.where(s.cen_row[ao] >= 0, s.o2r[np.maximum(s.cen_row[ao], 0)], -1)
        same = (cp >= 0) & (cp == s.top[ar])
        inter = s.inter(ao, ar)
        mratio = s.mbound[ao] / rs.mvir[ar]
        vratio = np.where((S["Vmax"][ao] > 0) & (rs.vmax[ar] > 0), S["Vmax"][ao] / np.where(rs.vmax[ar] > 0, rs.vmax[ar], 1), np.nan)
        off = pdist(s.pos[ao], s.rpos[ar], s.box)
        offR = off / np.where(rs.rvir_kpc[ar] > 0, rs.rvir_kpc[ar], np.nan)
        shN = inter / s.nbound[ao].astype(np.float64)
        shP = inter / np.where(s.np_rs[ar] > 0, s.np_rs[ar], np.nan)
        inU = s.rsize[ar] >= mn
        P("")
        P("A  matched, both satellites: %d pairs = election %d + footprint %d (%d with Rockstar Mvir/m_p >= %d, in both "
          "universes)" % (ao.size, int(np.count_nonzero(asrc == SRC_ELECTION)), int(np.count_nonzero(asrc == SRC_FOOTPRINT)),
                          int(inU.sum()), mn))
        P(bin_header(what="A by our Nbound"))
        P(bin_line("all", bo[ao], np.ones(ao.size, bool)))
        P(bin_line("election", bo[ao], asrc == SRC_ELECTION))
        P(bin_line("footprint", bo[ao], asrc == SRC_FOOTPRINT))
        P(bin_line("same top host", bo[ao], same))
        P(bin_line("different top host", bo[ao], ~same & (cp >= 0)))
        P(bin_line("our central unmatched", bo[ao], cp < 0))
        P(bin_header(what="A by Rockstar Mvir/m_p"))
        P(bin_line("all", br[ar], np.ones(ao.size, bool)))
        P(bin_line("election", br[ar], asrc == SRC_ELECTION))
        P(bin_line("footprint", br[ar], asrc == SRC_FOOTPRINT))
        P(bin_line("same top host", br[ar], same))
        cvote = np.where(s.cen_row[ao] >= 0, s.vote_o[np.maximum(s.cen_row[ao], 0)], -3)
        same_vote = (cp < 0) & (cvote >= 0) & (cvote == s.top[ar])
        P(bin_line("  ... central's own vote = top host", br[ar], same_vote))
        for lab, m in (("all A (union)", np.ones(ao.size, bool)), ("election", asrc == SRC_ELECTION),
                       ("footprint", asrc == SRC_FOOTPRINT), ("same top host", same),
                       ("different top host (central matched)", ~same & (cp >= 0)), ("our central unmatched", cp < 0)):
            P("  ratios ours/Rockstar, %s:" % lab)
            P(qline("Mbound/Mvir", mratio[m]))
            P(qline("Vmax/vmax", vratio[m]))
            P(qline("centre offset kpc/h", off[m]))
            P(qline("centre offset / Rvir(Rockstar)", offR[m]))
            P(qline("shared / Nbound", shN[m]))
            P(qline("shared / Np", shP[m]))
        P("  by our Nbound bin (all A): Mbound/Mvir | shared/Nbound | shared/Np | offset/Rvir (p10/p50/p90)")
        for x, bn in enumerate(BIN_NAMES):
            m = bo[ao] == x
            P("    %-8s n=%-5d %s | %s | %s | %s" % (bn, int(m.sum()), _q(mratio[m]), _q(shN[m]), _q(shP[m][np.isfinite(shP[m])]),
                                                 _q(offR[m][np.isfinite(offR[m])])))

        # centrals of our satellites without a Rockstar partner: their own votes and the Rockstar host there
        sc = s.cen_row[np.flatnonzero(s.sat)]
        sc = sc[sc >= 0]
        cu, ncu = np.unique(sc[s.o2r[sc] < 0], return_counts=True)
        P("  our centrals hosting satellites without a Rockstar partner: %d, hosting %d of %d satellites; the largest:"
          % (cu.size, int(ncu.sum()), int(univ_o.sum())))
        for c, nsat in sorted(zip(cu.tolist(), ncu.tolist()), key=lambda x: -x[1])[:6]:
            v = int(s.vote_o[c])
            vtxt = "votes for no Rockstar halo" if v < 0 else "votes for Rockstar %d (pid %d, Mvir/m_p %.0f), whose vote is %s" % (
                rs.id[v], rs.pid[v], s.rsize[v], "our %d" % ours.tid[s.vote_r[v]] if s.vote_r[v] >= 0 else str(s.vote_r[v]))
            hosts = np.flatnonzero(~s.sub & (pdist(s.rpos, s.pos[c], s.box) <= max(s.rvir[c], 1.0)))
            if hosts.size:
                h = hosts[np.argmax(rs.mvir[hosts])]
                htxt = "largest Rockstar host within RBoundSO[0]: %d (Mvir/m_p %.0f, d %.1f kpc/h, Np %d, its vote %s, partner %s)" % (
                    rs.id[h], s.rsize[h], pdist(s.rpos[h], s.pos[c], s.box), s.np_rs[h],
                    "our %d" % ours.tid[s.vote_r[h]] if s.vote_r[h] >= 0 else str(s.vote_r[h]),
                    "our %d" % ours.tid[s.r2o_all[h]] if s.r2o_all[h] >= 0 else "none")
            else:
                htxt = "no Rockstar host within RBoundSO[0]"
            P("    TrackId %d (Nbound %d, Flags %d, %d satellites): %s; %s" % (
                ours.tid[c], s.nbound[c], int(S["Flags"][c]), nsat, vtxt, htxt))

        # ---------------- B1
        B1 = osat & rh
        b1o, b1r = po[B1], pr[B1]
        b1src = s.psrc[B1]
        dR1 = np.full(b1o.size, np.nan)
        cp1 = np.where(s.cen_row[b1o] >= 0, s.o2r[np.maximum(s.cen_row[b1o], 0)], -1)
        okp = cp1 >= 0
        dR1[okp] = pdist(s.pos[b1o[okp]], s.rpos[cp1[okp]], s.box) / rs.rvir_kpc[cp1[okp]]
        self_cp = okp & (cp1 == b1r)
        P("")
        P("B1 matched, ours satellite, Rockstar host: %d pairs = election %d + footprint %d" % (
            b1o.size, int(np.count_nonzero(b1src == SRC_ELECTION)), int(np.count_nonzero(b1src == SRC_FOOTPRINT))))
        P(bin_header(what="B1 by our Nbound"))
        P(bin_line("all", bo[b1o], np.ones(b1o.size, bool)))
        P(bin_line("central's partner beyond Rvir", bo[b1o], okp & (dR1 > 1)))
        P(bin_line("central's partner within Rvir", bo[b1o], okp & (dR1 <= 1)))
        P(bin_line("our central has no partner", bo[b1o], ~okp))
        P(bin_line("partner is a Rockstar subhalo", bo[b1o], okp & s.sub[np.maximum(cp1, 0)]))
        P(qline("d / Rvir(partner of central)", dR1))
        P(qline("  d/R where <= 1", dR1[okp & (dR1 <= 1)]))
        P("    our Mbound / Rockstar Mvir of the B1 pairs: %s; Rockstar host Mvir/m_p: %s; our Nbound: %s" % (
            _q(s.mbound[b1o] / rs.mvir[b1r]), _q(s.rsize[b1r]), _q(s.nbound[b1o].astype(float))))
        if self_cp.any():
            P("    note: %d B1 satellites share their Rockstar partner with our central" % int(self_cp.sum()))

        # ---------------- B2
        B2all = oc & ~rh
        B2 = B2all & (s.rsize[pr] >= mn)
        self.n_b2_left = int(np.count_nonzero(B2all & ~B2))
        b2o, b2r = po[B2], pr[B2]
        th = s.top[b2r]
        dR2 = pdist(s.pos[b2o], s.rpos[th], s.box) / np.where(rs.rvir_kpc[th] > 0, rs.rvir_kpc[th], np.nan)
        dR2r = pdist(s.rpos[b2r], s.rpos[th], s.box) / np.where(rs.rvir_kpc[th] > 0, rs.rvir_kpc[th], np.nan)
        hp = s.r2o[th]
        h_cen_other = (hp >= 0) & s.cen[np.maximum(hp, 0)] & (s.host[np.maximum(hp, 0)] != s.host[b2o])
        h_sat = (hp >= 0) & s.sat[np.maximum(hp, 0)]
        h_none = hp < 0
        h_rest = ~(h_cen_other | h_sat | h_none)
        P("")
        P("B2 matched, ours central, Rockstar subhalo with Mvir/m_p >= %d: %d pairs; below --min-np: %d (left out)"
          % (mn, b2o.size, self.n_b2_left))
        P(bin_header(what="B2 by Rockstar Mvir/m_p"))
        P(bin_line("all", br[b2r], np.ones(b2o.size, bool)))
        P(bin_line("top host matched to central of another group", br[b2r], h_cen_other))
        P(bin_line("top host matched to a satellite", br[b2r], h_sat))
        P(bin_line("top host unmatched", br[b2r], h_none))
        if h_rest.any():
            P(bin_line("top host matched otherwise", br[b2r], h_rest))
        P(bin_line("within top host Rvir (our CorePos)", br[b2r], dR2 <= 1))
        P(bin_line("within top host Rvir (Rockstar centre)", br[b2r], dR2r <= 1))
        P(bin_header(what="B2 by our Nbound"))
        P(bin_line("all", bo[b2o], np.ones(b2o.size, bool)))
        P(qline("d / Rvir(top host), our CorePos", dR2))
        P(qline("d / Rvir(top host), Rockstar centre", dR2r))
        P(qline("  host matched to another group's central", dR2[h_cen_other]))
        P(qline("  host unmatched", dR2[h_none]))
        P("    top host Mvir / Rockstar subhalo Mvir: %s; our FOF group of the B2 central: Nbound %s" % (
            _q(rs.mvir[th] / rs.mvir[b2r]), _q(s.nbound[b2o].astype(float))))

        # ---------------- phantoms of the hlist of k
        hi = hl.index_of(s.a, s.k)
        hd = hl.read(hi)
        self.hlist_file = hl.files[hi]
        ph = np.flatnonzero((hd["phantom"] != 0) & (hd["pid"] != -1) & (hd["Mvir"] / s.mp >= mn))
        allph = np.flatnonzero(hd["phantom"] != 0)
        orig_ph = hd["Orig_halo_ID"][allph]
        orig_real = hd["Orig_halo_ID"][hd["phantom"] == 0]
        self.rs_in_hlist = np.isin(rs.id, orig_real)      # Rockstar halo kept by consistent-trees as a real row
        n_ph_in_rs = int(np.count_nonzero(rs.row(orig_ph) >= 0))
        n_real_in_rs = int(np.count_nonzero(rs.row(orig_real) >= 0))
        self.phantom_note = ("hlist %s: %d rows, %d phantom; real rows' Orig_halo_ID found among Rockstar IDs: %d of %d; "
                             "phantom rows' Orig_halo_ID %s (Rockstar IDs of out_%d: %d..%d), %d of them equal to a Rockstar ID"
                             % (os.path.basename(self.hlist_file), hd["n"], allph.size, n_real_in_rs, orig_real.size,
                                "%d..%d" % (orig_ph.min(), orig_ph.max()) if orig_ph.size else "-", rs.snap,
                                rs.id.min(), rs.id.max(), n_ph_in_rs))
        satrows = np.flatnonzero(s.sat)
        near_sat = {}               # our satellite row -> nearby phantom rows
        ph_has = np.zeros(ph.size, bool)
        if ph.size and satrows.size:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.mod(s.pos[satrows], s.box), boxsize=s.box)
            ppos = np.stack([hd["x"][ph], hd["y"][ph], hd["z"][ph]], axis=1) * 1e3
            for j, h in enumerate(ph.tolist()):
                cand = tree.query_ball_point(np.mod(ppos[j], s.box), 0.5 * hd["Rvir"][h])
                for c in cand:
                    o = satrows[c]
                    q = s.mbound[o] / hd["Mvir"][h]
                    if 1.0 / 3 <= q <= 3:
                        ph_has[j] = True
                        near_sat.setdefault(int(o), []).append(h)
        P("")
        P("Phantoms of %s (pid != -1, Mvir/m_p >= %d): %d; with one of our satellites within 0.5 Rvir and Mvir within a "
          "factor 3 of its Mbound (we have it, Rockstar's raw finder lost it this snapshot): %d"
          % (os.path.basename(self.hlist_file), mn, ph.size, int(ph_has.sum())))
        bph = size_bin(hd["Mvir"][ph] / s.mp)
        P(bin_header(what="phantoms by Mvir/m_p"))
        P(bin_line("all", bph, np.ones(ph.size, bool)))
        P(bin_line("with a satellite of ours", bph, ph_has))
        if ph_has.any():
            matched_near = [o for o in near_sat if s.o2r[o] >= 0]
            P("    satellites of ours next to a phantom: %d (%d of them matched to a Rockstar halo)" % (
                len(near_sat), len(matched_near)))
        P("  " + self.phantom_note)

        # ---------------- C
        C = np.flatnonzero(univ_o & (s.o2r < 0))
        cpC = np.where(s.cen_row[C] >= 0, s.o2r[np.maximum(s.cen_row[C], 0)], -1)
        top = s.o_top[C]
        topm = np.where(top >= 0, s.r2o[np.maximum(top, 0)], -1)
        catC = np.full(C.size, "C3", object)
        catC[top == -1] = "C4"
        catC[(top >= 0) & (topm >= 0) & s.sub[np.maximum(top, 0)]] = "C2s"
        catC[(top >= 0) & (topm >= 0) & ~s.sub[np.maximum(top, 0)]] = "C2h"
        catC[(top >= 0) & (top == cpC)] = "C1"
        hdC = np.full(C.size, np.nan)
        hsC = np.full(C.size, "none", object)
        for j, o in enumerate(C.tolist()):
            hdC[j], hsC[j] = s.host_dist(o)
        mpk = S["MboundPeak"][C].astype(np.float64)
        mfrac = np.where(mpk > 0, S["Mbound"][C] / np.where(mpk > 0, mpk, 1), np.nan)
        resc = (S["Flags"][C].astype(np.int64) & RESCUED) != 0
        sacc = S["SnapAcc"][C].astype(np.int64)
        age = np.where(sacc >= 0, s.k - sacc, -1)
        phn = np.array([len(near_sat.get(int(o), [])) for o in C], np.int64)
        tshare = s.o_topn[C] / np.maximum(s.nexcl[C], 1).astype(np.float64)
        nshare = s.o_none[C] / np.maximum(s.nexcl[C], 1).astype(np.float64)
        dC = s.depth[C]
        P("")
        P("C  ours-only satellites (no mutual partner): %d of %d satellites" % (C.size, int(univ_o.sum())))
        P(bin_header(what="C by our Nbound"))
        P(bin_line("all", bo[C], np.ones(C.size, bool)))
        CN = (("C1", "C1 in the host (central's partner)"), ("C2s", "C2s in a matched Rockstar sub"),
              ("C2h", "C2h in another matched host"), ("C3", "C3 in an unmatched Rockstar"),
              ("C4", "C4 in no Rockstar halo"))
        for c, lab in CN:
            P(bin_line(lab, bo[C], catC == c))
        P(bin_line("phantom nearby", bo[C], phn > 0))
        P(bin_line("RESCUED flag", bo[C], resc))
        P("  per category: top share | d/R host (R: partner of the central, else our central) frac > 1 | "
          "Mbound/MboundPeak | age k-SnapAcc | Depth 1/2/>=3 | rescued | phantom nearby")
        for c, lab in CN + (("*", "all C"),):
            m = np.ones(C.size, bool) if c == "*" else (catC == c)
            if not m.any():
                continue
            hd_ok = m & np.isfinite(hdC)
            P("    %-4s n=%-5d share %s | d/R %s, > 1: %d of %d | M/Mpeak %s | age %s | Depth %d/%d/%d | %d | %d" % (
                c, int(m.sum()), _q(tshare[m]), _q(hdC[hd_ok]), int(np.count_nonzero(hd_ok & (hdC > 1))),
                int(hd_ok.sum()), _q(mfrac[m & np.isfinite(mfrac)]), _q(age[m & (age >= 0)].astype(float)),
                int(np.count_nonzero(m & (dC == 1))), int(np.count_nonzero(m & (dC == 2))),
                int(np.count_nonzero(m & (dC >= 3))), int(np.count_nonzero(m & resc)), int(np.count_nonzero(m & (phn > 0)))))
        P("    host distance from: Rockstar partner of the central %d, our central %d, none %d" % (
            int(np.count_nonzero(hsC == "rs")), int(np.count_nonzero(hsC == "ours")), int(np.count_nonzero(hsC == "none"))))
        P("    share of the exclusive set in no Rockstar halo, p10/p50/p90: C4 %s, others %s" % (
            _q(nshare[catC == "C4"]), _q(nshare[catC != "C4"])))
        for c in ("C1", "C3"):
            cc = C[catC == c]
            d = s.o_top[cc]
            rp = s.r_rowplur[d]
            P("    %s destination lists by the plurality owner of their particles bound in ours: our central %d, the "
              "satellite itself %d, another satellite %d, another central or hostless %d, none bound %d; the list's "
              "vote (saved particles, -1 included) is -1 for %d; satellite share >= %.1f of its set %d" % (
                  c, int(np.count_nonzero(rp == s.cen_row[cc])), int(np.count_nonzero(rp == cc)),
                  int(np.count_nonzero((rp >= 0) & (rp != cc) & s.sat[np.maximum(rp, 0)])),
                  int(np.count_nonzero((rp >= 0) & (rp != s.cen_row[cc]) & ~s.sat[np.maximum(rp, 0)])),
                  int(np.count_nonzero(rp < 0)), int(np.count_nonzero(s.vote_r[d] == -1)), FOOTPRINT,
                  int(np.count_nonzero(s.o_topn[cc] >= FOOTPRINT * np.maximum(s.nexcl[cc], 1)))))

        # ---------------- D
        D = np.flatnonzero(univ_r & (s.r2o < 0))
        od = s.r_top[D]
        odr = np.maximum(od, 0)
        catD = np.full(D.size, "D4u", object)
        catD[od == -2] = "D4a"
        catD[od == -3] = "D4a"      # no listed particle at all
        isrow = od >= 0
        catD[isrow & (s.depth[odr] == 0) & (s.host[odr] >= 0)] = "D1"
        catD[isrow & (s.host[odr] < 0)] = "D1x"
        m2 = isrow & s.sat[odr] & (s.o2r[odr] >= 0)
        catD[m2 & s.sub[np.maximum(s.o2r[odr], 0)]] = "D2s"
        catD[m2 & ~s.sub[np.maximum(s.o2r[odr], 0)]] = "D2h"
        catD[isrow & s.sat[odr] & (s.o2r[odr] < 0)] = "D3"
        nl = np.maximum(s.r_nlist[D], 1).astype(np.float64)
        dshare = s.r_topn[D] / nl
        dunb = s.r_unb[D] / nl
        dabs = s.r_abs[D] / nl
        nopig = s.r_abs[D] == s.r_nlist[D]
        thD = s.top[D]
        dRD = pdist(s.rpos[D], s.rpos[thD], s.box) / np.where(rs.rvir_kpc[thD] > 0, rs.rvir_kpc[thD], np.nan)
        bfrac = rs.mvir[D] / (s.mp * np.where(s.np_rs[D] > 0, s.np_rs[D], np.nan))
        # orphan test: markers of our orphans inside each D list
        O = ours.O
        orph_t = np.full(D.size, -1, np.int64)
        orph_n = np.zeros(D.size, np.int64)
        orph_age = np.full(D.size, -1, np.int64)
        orph_sank = np.zeros(D.size, bool)
        if ours.otid.size and D.size:
            mk = O["OrphanIDs"]
            ok = mk != NO_ID
            m_id = mk[ok].astype(np.int64)
            m_or = np.repeat(np.arange(mk.shape[0]), ok.sum(1))
            o = np.argsort(m_id, kind="stable")
            m_id, m_or = m_id[o], m_or[o]
            lens = s.rl_start[D + 1] - s.rl_start[D]
            lids = s.rl_ids[expand(s.rl_start[D], lens)]
            lh = np.repeat(np.arange(D.size), lens)
            lo = np.searchsorted(m_id, lids, side="left")
            hi_ = np.searchsorted(m_id, lids, side="right")
            nn = hi_ - lo
            ph_ = np.repeat(lh, nn)
            por = m_or[expand(lo, nn)]
            if ph_.size:
                base = int(mk.shape[0]) + 1
                uk, cnt = np.unique(ph_ * base + por, return_counts=True)
                hh, oo = uk // base, uk % base
                # a marker in the list is credited only with the orphan's TracerPos inside the halo's Rvir
                self.marker_halos = np.unique(hh).size
                inside = pdist(O["TracerPos"][oo].astype(np.float64), s.rpos[D[hh]], s.box) <= rs.rvir_kpc[D[hh]]
                hh, oo, cnt = hh[inside], oo[inside], cnt[inside]
                sd = O["SnapDeath"][oo].astype(np.int64)
                order = np.lexsort((ours.otid[oo], -sd, -cnt, hh))
                hh, oo, cnt = hh[order], oo[order], cnt[order]
                first = np.ones(hh.size, bool)
                first[1:] = hh[1:] != hh[:-1]
                hh, oo, cnt = hh[first], oo[first], cnt[first]
                orph_t[hh] = ours.otid[oo]
                orph_n[hh] = cnt
                orph_age[hh] = s.k - O["SnapDeath"][oo].astype(np.int64)
                orph_sank[hh] = O["SinkTrackId"][oo].astype(np.int64) >= 0
        self.D = D
        # closing identities: our satellites = A + B1 + C; Rockstar universe = A (in it) + B2 + D
        self.acct = (int(univ_o.sum()), ao.size, b1o.size, C.size, int(univ_r.sum()), int(inU.sum()), b2o.size, D.size)
        if not hasattr(self, "marker_halos"):
            self.marker_halos = 0
        if not args.no_unbind:
            for r in D.tolist():
                self.unbind_sets[int(r)] = (s.rs_list(r).copy(), s.rpos[r].copy())
        self.C, self.catC, self.catD = C, catC, catD
        # rows for the CSVs (unbinding columns filled after the unbinding)
        self.ours_rows = []
        A_of = dict(zip(ao.tolist(), range(ao.size)))
        B1_of = dict(zip(b1o.tolist(), range(b1o.size)))
        C_of = dict(zip(C.tolist(), range(C.size)))
        hdr_o = ("TrackId,Nbound,bin,Depth,HostGroup,CentralTrackId,category,rs_partner_id,rs_partner_pid,rs_partner_Mvir_mp,"
                 "top_dest_rs_id,top_share,share_none,central_rs_partner_id,host_dist_R,host_ref,same_top_host,"
                 "Mbound_over_Mvir,Vmax_ratio,offset_kpc,offset_over_Rvir,shared,shared_over_Nbound,shared_over_Np,"
                 "Mbound_over_MboundPeak,rescued,age,phantoms_nearby,pair_source")
        self.hdr_o = hdr_o
        nex = np.maximum(s.nexcl, 1).astype(np.float64)
        for o in np.flatnonzero(univ_o).tolist():
            r = int(s.o2r[o])
            c = int(s.cen_row[o])
            cpo = int(s.o2r[c]) if c >= 0 else -1
            hdv, hsrc = s.host_dist(o)
            mpk_o = float(S["MboundPeak"][o])
            sa = int(S["SnapAcc"][o])
            row = dict(TrackId=int(ours.tid[o]), Nbound=int(s.nbound[o]), bin=BIN_NAMES[bo[o]] if bo[o] >= 0 else "<20",
                       Depth=int(s.depth[o]), HostGroup=int(s.host[o]), CentralTrackId=int(S["CentralTrackId"][o]),
                       rs_partner_id=s.rid_of(r), rs_partner_pid=int(rs.pid[r]) if r >= 0 else -1,
                       rs_partner_Mvir_mp=float(s.rsize[r]) if r >= 0 else np.nan,
                       top_dest_rs_id=s.rid_of(int(s.o_top[o])), top_share=float(s.o_topn[o] / nex[o]),
                       share_none=float(s.o_none[o] / nex[o]), central_rs_partner_id=s.rid_of(cpo),
                       host_dist_R=hdv, host_ref=hsrc,
                       Mbound_over_MboundPeak=float(S["Mbound"][o]) / mpk_o if mpk_o > 0 else np.nan,
                       rescued=int((int(S["Flags"][o]) & RESCUED) != 0), age=s.k - sa if sa >= 0 else -1,
                       phantoms_nearby=len(near_sat.get(o, [])), pair_source=SRC_NAMES[int(s.osrc[o]) if r >= 0 else -1])
            if o in A_of:
                j = A_of[o]
                row.update(category="A", same_top_host=int(same[j]), Mbound_over_Mvir=mratio[j], Vmax_ratio=vratio[j],
                           offset_kpc=off[j], offset_over_Rvir=offR[j], shared=int(inter[j]), shared_over_Nbound=shN[j],
                           shared_over_Np=shP[j])
            elif o in B1_of:
                row.update(category="B1", shared=int(s.inter([o], [r])[0]))
            elif o in C_of:
                row.update(category=catC[C_of[o]])
            else:
                row.update(category="matched-other")
            self.ours_rows.append(row)
        self.rs_rows = []
        D_of = dict(zip(D.tolist(), range(D.size)))
        A_r = dict(zip(ar.tolist(), range(ar.size)))
        B2_r = dict(zip(b2r.tolist(), range(b2r.size)))
        self.hdr_r = ("rs_id,pid,top_host_id,Mvir_mp,bin,Np,nlist,category,our_partner_TrackId,top_dest_TrackId,top_share,"
                      "share_unbound,share_absent,host_dist_R,bound_frac,orphan_TrackId,orphan_markers,since_death,sank,"
                      "orphan_class,unbind_found,unbind_nbound,self_bound,unbind_status,pair_source,in_hlist")
        nlr = np.maximum(s.r_nlist, 1).astype(np.float64)
        for r in np.flatnonzero(univ_r).tolist():
            th_ = int(s.top[r])
            rv = rs.rvir_kpc[th_]
            row = dict(rs_id=int(rs.id[r]), pid=int(rs.pid[r]), top_host_id=int(rs.id[th_]), Mvir_mp=float(s.rsize[r]),
                       bin=BIN_NAMES[br[r]] if br[r] >= 0 else "<20", Np=int(s.np_rs[r]), nlist=int(s.r_nlist[r]),
                       our_partner_TrackId=s.tid_of(int(s.r2o[r])),
                       top_dest_TrackId=s.tid_of(int(s.r_top[r])), top_share=float(s.r_topn[r] / nlr[r]),
                       share_unbound=float(s.r_unb[r] / nlr[r]), share_absent=float(s.r_abs[r] / nlr[r]),
                       host_dist_R=float(pdist(s.rpos[r], s.rpos[th_], s.box) / rv) if rv > 0 else np.nan,
                       bound_frac=float(rs.mvir[r] / (s.mp * s.np_rs[r])) if s.np_rs[r] > 0 else np.nan,
                       unbind_status="", pair_source=SRC_NAMES[int(s.osrc[s.r2o[r]]) if s.r2o[r] >= 0 else -1],
                       in_hlist=int(self.rs_in_hlist[r]))
            if r in D_of:
                j = D_of[r]
                row.update(category=catD[j], orphan_TrackId=int(orph_t[j]), orphan_markers=int(orph_n[j]),
                           since_death=int(orph_age[j]), sank=int(orph_sank[j]),
                           orphan_class="dead-track" if orph_t[j] >= 0 else ("no-pig-particle" if nopig[j] else "no-orphan-credited"),
                           unbind_status="off" if args.no_unbind else "")
            elif r in A_r:
                row.update(category="A")
            elif r in B2_r:
                row.update(category="B2")
            else:
                row.update(category="matched-other")
            self.rs_rows.append(row)
        self.Dinfo = dict(od=od, dshare=dshare, dunb=dunb, dabs=dabs, nopig=nopig, dRD=dRD, bfrac=bfrac,
                          orph_t=orph_t, orph_age=orph_age, orph_sank=orph_sank, br=br[D])

    def finish(self, ures):
        """D table with the unbinding results, then write the files"""
        s, args, P = self.s, self.args, self.lines.append
        D, catD, I = self.D, self.catD, self.Dinfo
        nb = np.full(D.size, -1, np.int64)
        nfound = np.full(D.size, -1, np.int64)
        ust = np.full(D.size, "off", object)      # done, unconverged, skipped, incomplete, empty, noncompact, off
        for j, r in enumerate(D.tolist()):
            if r in ures:
                nfound[j], nb[j], ust[j] = ures[r][1], ures[r][2], ures[r][3]
        done = ust == "done"
        sb = done & (nb >= SELF_BOUND)
        unc = ust == "unconverged"
        skp = ust == "skipped"
        inc = ust == "incomplete"
        emp = ust == "empty"
        ncp = ust == "noncompact"
        P("")
        P("D  Rockstar-only subhaloes (Mvir/m_p >= %d, no mutual partner): %d" % (self.args.min_np, D.size))
        P(bin_header(what="D by Rockstar Mvir/m_p"))
        P(bin_line("all", I["br"], np.ones(D.size, bool)))
        DN = (("D1", "D1 in our central"), ("D1x", "D1x in a hostless track"), ("D2s", "D2s in sat matched to RS sub"),
              ("D2h", "D2h in sat matched to RS host"), ("D3", "D3 in an unmatched sat"),
              ("D4u", "D4u bound to nobody"), ("D4a", "D4a not in the PIG"))
        for c, lab in DN:
            if np.any(catD == c):
                P(bin_line(lab, I["br"], catD == c))
        P("  orphan test: lists holding an orphan marker %d; credited (marker in the list and TracerPos within the "
          "halo's Rvir) %d" % (self.marker_halos, int(np.count_nonzero(I["orph_t"] >= 0))))
        nx = catD != "D1x"          # D1x: the destination is a live hostless track of ours (own line above)
        P(bin_line("our dead track (orphan credited)", I["br"], nx & (I["orph_t"] >= 0)))
        P(bin_line("  ... it sank", I["br"], nx & (I["orph_t"] >= 0) & I["orph_sank"]))
        P(bin_line("no orphan credited", I["br"], nx & (I["orph_t"] < 0)))
        P(bin_line("  ... no PIG particle in the list", I["br"], nx & (I["orph_t"] < 0) & I["nopig"]))
        if args.no_unbind:
            P("  independent unbinding: off (--no-unbind)")
        else:
            P(bin_line("self-bound (>= %d bound)" % SELF_BOUND, I["br"], sb))
            P(bin_line("not self-bound", I["br"], done & ~sb))
            P(bin_line("unconverged (maxit)", I["br"], unc))
            P(bin_line("incomplete (IDs not in the PART)", I["br"], inc))
            P(bin_line("skipped (list > --maxset)", I["br"], skp))
            P(bin_line("empty (no listed particle)", I["br"], emp))
            P(bin_line("noncompact (spans >= box/4)", I["br"], ncp))
            inh = self.rs_in_hlist[D]
            P(bin_line("kept by consistent-trees (hlist row)", I["br"], inh))
            P(bin_line("  ... of the self-bound", I["br"], inh & sb))
            P(bin_line("  ... of the not self-bound", I["br"], inh & done & ~sb))
            dead = I["orph_t"] >= 0
            P(bin_header(what="D sub-categories, self-bound"))
            for lab, m in (("D1 our dead track", (catD == "D1") & dead), ("D1 no orphan credited", (catD == "D1") & ~dead),
                           ("D1x hostless", catD == "D1x"), ("D2 (D2s + D2h)", (catD == "D2s") | (catD == "D2h")),
                           ("D3", catD == "D3"), ("D4 (D4u + D4a)", (catD == "D4u") | (catD == "D4a"))):
                P(bin_line(lab, I["br"], m))
                P(bin_line("  self-bound", I["br"], m & sb))
                P(bin_line("  not self-bound", I["br"], m & done & ~sb))
                P(bin_line("  unconverged (maxit)", I["br"], m & unc))
                if np.any(m & inc):
                    P(bin_line("  incomplete", I["br"], m & inc))
                if np.any(m & skp):
                    P(bin_line("  skipped", I["br"], m & skp))
                if np.any(m & emp):
                    P(bin_line("  empty", I["br"], m & emp))
                if np.any(m & ncp):
                    P(bin_line("  noncompact", I["br"], m & ncp))
        P("  per category: top share | share -1 | share -2 | d/R top host, > 1 | bound frac Mvir/(m_p Np) | dead track "
          "(sank) | k-SnapDeath | self-bound of unbound | bound/found")
        for c, lab in DN + (("*", "all D"),):
            m = np.ones(D.size, bool) if c == "*" else (catD == c)
            if not m.any():
                continue
            dr = I["dRD"][m]
            dt = m & (I["orph_t"] >= 0)
            P("    %-4s n=%-5d %s | %s | %s | %s, %d | %s | %d (%d) | %s | %s of %d | %s" % (
                c, int(m.sum()), _q(I["dshare"][m]), _q(I["dunb"][m]), _q(I["dabs"][m]), _q(dr[np.isfinite(dr)]),
                int(np.count_nonzero(dr > 1)), _q(I["bfrac"][m & np.isfinite(I["bfrac"])]), int(dt.sum()),
                int(np.count_nonzero(dt & I["orph_sank"])), _q(I["orph_age"][dt].astype(float)),
                int(np.count_nonzero(m & sb)), int(np.count_nonzero(m & done)),
                _q((nb[m & done] / np.maximum(nfound[m & done], 1)).astype(float))))
        if not args.no_unbind:
            sk = [(int(s.rs.id[r]), int(ures[r][0])) for r in D.tolist() if r in ures and ures[r][3] == "skipped"]
            P("  skipped lists above %d particles: %d%s" % (args.maxset, len(sk), (": " + ", ".join(
                "Rockstar %d (%d)" % x for x in sk)) if sk else ""))
        ns, na, nb1, nc, nu, nau, nb2, nd = self.acct
        P("  accounting: our satellites %d = A %d + B1 %d + C %d (%s); Rockstar universe %d = A %d + B2 %d + D %d (%s)" % (
            ns, na, nb1, nc, "closes" if ns == na + nb1 + nc else "DOES NOT CLOSE", nu, nau, nb2, nd,
            "closes" if nu == nau + nb2 + nd else "DOES NOT CLOSE"))
        # fill the CSV unbinding columns
        D_of = dict(zip(D.tolist(), range(D.size)))
        rid_row = {int(s.rs.id[r]): r for r in D.tolist()}
        for row in self.rs_rows:
            r = rid_row.get(row["rs_id"])
            if r is None:
                continue
            j = D_of[r]
            if not args.no_unbind:
                row.update(unbind_found=int(nfound[j]), unbind_nbound=int(nb[j]) if nb[j] >= 0 else -1,
                           self_bound=int(sb[j]) if done[j] else -1, unbind_status=ust[j])
        k = self.k
        write_csv(os.path.join(args.out, "census_%d_ours.csv" % k), self.hdr_o, self.ours_rows)
        write_csv(os.path.join(args.out, "census_%d_rockstar.csv" % k), self.hdr_r, self.rs_rows)
        text = "\n".join(self.lines) + "\n"
        with open(os.path.join(args.out, "census_%d.txt" % k), "w") as f:
            f.write(text)
        return text


def write_csv(path, header, rows):
    cols = header.split(",")
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(fmt(r.get(c, "")) for c in cols) + "\n")


# ---------------------------------------------------------------------------------- identity
class Anchor(object):
    """A pairs at k0 with their exclusive sets, Rockstar lists and hlist rows."""

    def __init__(self, s, hl):
        self.k0 = s.k
        po, pr = s.po, s.pr
        m = s.sat[po] & s.sub[pr]
        self.o, self.r = po[m], pr[m]
        self.t = s.ours.tid[self.o].copy()
        self.src = s.osrc[self.o].copy()
        self.rid = s.rs.id[self.r].copy()
        self.nb0 = s.nbound[self.o].copy()
        self.bin = size_bin(self.nb0)
        self.n = self.o.size
        sets_o = [s.ex_set(o) for o in self.o.tolist()]
        sets_r = [s.rs_list(r) for r in self.r.tolist()]
        self.B_len = np.array([x.size for x in sets_o], np.int64)
        self.P_len = np.array([x.size for x in sets_r], np.int64)
        self.B_ids = np.concatenate(sets_o) if sets_o else np.zeros(0, np.int64)
        self.P_ids = np.concatenate(sets_r) if sets_r else np.zeros(0, np.int64)
        self.B_pair = np.repeat(np.arange(self.n), self.B_len)
        self.P_pair = np.repeat(np.arange(self.n), self.P_len)
        self.i0 = hl.index_of(s.a, s.k)
        self.hfile0 = hl.files[self.i0]
        self.chain = hl.row_of_rockstar(self.i0, self.rid)
        self.in_hlist = self.chain >= 0
        self.ci = self.i0
        self.a0 = s.a
        self.steps = {}

    def step(self, s, hl, args):
        j = s.k - self.k0
        ib = hl.index_of(s.a, s.k)
        if ib <= self.ci:
            raise InputError("output %d maps to hlist %d, not after %d" % (s.k, ib, self.ci))
        self.chain = hl.advance(self.chain, self.ci, ib)
        self.ci = ib
        hd = hl.read(ib)
        n = self.n
        cr = self.chain
        alive_r = cr >= 0
        phant = np.zeros(n, bool)
        phant[alive_r] = hd["phantom"][cr[alive_r]] != 0
        rj = np.full(n, -1, np.int64)
        real = alive_r & ~phant
        rj[real] = s.rs.row(hd["Orig_halo_ID"][cr[real]])
        real_missing = int(np.count_nonzero(real & (rj < 0)))
        ot = s.ours.row(self.t)
        alive_o = ot >= 0
        p_t = np.where(alive_o, s.o2r_all[np.maximum(ot, 0)], -1)
        p_t_el = np.where(alive_o, s.o2r_el_all[np.maximum(ot, 0)], -1)
        p_rj = np.where(rj >= 0, s.r2o_all[np.maximum(rj, 0)], -1)
        # self-overlaps
        own_j = s.ex_owner_of(self.B_ids)
        ours_self = np.bincount(self.B_pair, weights=((own_j == ot[self.B_pair]) & (ot[self.B_pair] >= 0)).astype(float),
                                minlength=n) / np.maximum(self.B_len, 1)
        rown = s.rs.owner_row(self.P_ids)
        rs_self = np.bincount(self.P_pair, weights=((rown == rj[self.P_pair]) & (rj[self.P_pair] >= 0)).astype(float),
                              minlength=n) / np.maximum(self.P_len, 1)
        rs_self = np.where(rj >= 0, rs_self, np.nan)
        cat = np.full(n, "", object)
        lab = np.full(n, "", object)
        e0 = ~self.in_hlist
        e1 = ~e0 & alive_o & alive_r & (rj >= 0) & (p_t == rj)
        moved = e1 & (p_t_el != rj)                   # E1 only through a central-rule or footprint pair
        psrc_j = np.where(alive_o, s.osrc[np.maximum(ot, 0)], -1)
        e2 = ~e0 & alive_o & alive_r & ~e1
        e3 = ~e0 & alive_o & ~alive_r
        e4 = ~e0 & ~alive_o & alive_r
        e5 = ~e0 & ~alive_o & ~alive_r
        cat[e0], cat[e1], cat[e2], cat[e3], cat[e4], cat[e5] = "E0", "E1", "E2", "E3", "E4", "E5"
        oj = ours_self < JUMP
        rjmp = rs_self < JUMP
        lab[e2 & phant] = "rockstar-phantom"
        lab[e2 & real & (rj < 0)] = "unavailable"   # real chain row whose Orig_halo_ID is not in the catalogue
        nph = e2 & ~phant & (rj >= 0)
        lab[nph & oj & rjmp] = "both-jumped"
        lab[nph & oj & ~rjmp] = "ours-jumped"
        lab[nph & ~oj & rjmp] = "rockstar-jumped"
        lab[nph & ~oj & ~rjmp] = "neither-jumped"
        lab[e3 & (p_t >= 0)] = "rockstar-identity-break"
        lab[e3 & (p_t < 0)] = "rockstar-lost-ours-continues"
        lab[e0 & alive_o] = "ours-alive"
        lab[e0 & ~alive_o] = "ours-dead"
        # dead tracks of ours
        orow = s.ours.orow(self.t)
        O = s.ours.O
        sd = np.full(n, -1, np.int64)
        sink = np.full(n, -1, np.int64)
        desc = np.full(n, -1, np.int64)
        tpos = np.full((n, 3), np.nan)
        dd = orow >= 0
        if dd.any():
            sd[dd] = O["SnapDeath"][orow[dd]]
            sink[dd] = O["SinkTrackId"][orow[dd]]
            desc[dd] = O["DescendantTrackId"][orow[dd]]
            tpos[dd] = O["TracerPos"][orow[dd]]
        ptid = np.where(p_rj >= 0, s.ours.tid[np.maximum(p_rj, 0)], -1)
        lab[e4 & phant] = "raw-rockstar-lost-it-too"
        e4r = e4 & ~phant
        merger = e4r & (ptid >= 0) & ((ptid == sink) | (ptid == desc))
        lab[merger] = "merger-decision"
        lab[e4r & (ptid >= 0) & ~merger] = "other"
        unseen = e4r & (ptid < 0) & (rj >= 0)
        lab[unseen] = "death-rockstar-does-not-see"
        lab[e4r & (rj < 0)] = "unavailable"
        lab[e5] = "both-ended"
        for q in np.flatnonzero((e4 | e5) & (orow < 0)).tolist():
            lab[q] += ";ours-absent"
        tr_in = np.full(n, -1, np.int64)
        for q in np.flatnonzero(unseen & (rj >= 0)).tolist():
            if np.all(np.isfinite(tpos[q])) and s.rs.rvir_kpc[rj[q]] > 0:
                tr_in[q] = int(pdist(tpos[q], s.rpos[rj[q]], s.box) <= s.rs.rvir_kpc[rj[q]])
        hdist = np.full(n, np.nan)
        for q in np.flatnonzero(e3 | e2).tolist():
            hdist[q] = s.host_dist(int(ot[q]))[0]
        nbj = np.where(alive_o, s.nbound[np.maximum(ot, 0)], 0)
        depj = np.where(alive_o, s.depth[np.maximum(ot, 0)], -1)
        unb = {}
        if not args.no_unbind:
            for q in np.flatnonzero(unseen & (rj >= 0)).tolist():
                unb[int(rj[q])] = (s.rs_list(int(rj[q])).copy(), s.rpos[rj[q]].copy())
        lab_el = np.full(n, "", object)              # E2 attribution these pairs would have under the election alone
        lab_el[moved & oj & rjmp] = "both-jumped"
        lab_el[moved & oj & ~rjmp] = "ours-jumped"
        lab_el[moved & ~oj & rjmp] = "rockstar-jumped"
        lab_el[moved & ~oj & ~rjmp] = "neither-jumped"
        self.steps[j] = dict(moved=moved, lab_el=lab_el, psrc_j=psrc_j, k=s.k, a=s.a, cat=cat, lab=lab, ours_self=ours_self, rs_self=rs_self, rj=rj,
                             rj_id=np.where(rj >= 0, s.rs.id[np.maximum(rj, 0)], -1), cr=cr.copy(),
                             cr_id=np.where(cr >= 0, hd["id"][np.maximum(cr, 0)], -1), phant=phant,
                             p_t_id=np.where(p_t >= 0, s.rs.id[np.maximum(p_t, 0)], -1), ptid=ptid, sd=sd, sink=sink,
                             desc=desc, tr_in=tr_in, hdist=hdist, nbj=nbj, depj=depj, unseen=unseen,
                             real_missing=real_missing, hfile=hl.files[ib], unb_found=np.full(n, -1, np.int64),
                             unb_nb=np.full(n, -1, np.int64), unb_status=np.full(n, "", object))
        return unb

    def fill_unbinding(self, j, ures):
        st = self.steps[j]
        for q in np.flatnonzero(st["unseen"] & (st["rj"] >= 0)).tolist():
            r = int(st["rj"][q])
            if r in ures:
                st["unb_found"][q], st["unb_nb"][q], st["unb_status"][q] = ures[r][1], ures[r][2], ures[r][3]

    def report(self, args):
        L = []
        P = L.append
        P("identity from anchor %d (a = %.4f, %s): %d A pairs (our satellite, Rockstar subhalo); %d of their Rockstar "
          "halos have no row in that hlist (E0)" % (self.k0, self.a0, os.path.basename(self.hfile0), self.n,
                                                     int(np.count_nonzero(~self.in_hlist))))
        P("  bins by our Nbound at %d: %s" % (self.k0, ", ".join("%s %d" % (bn, int(np.count_nonzero(self.bin == x)))
                                                               for x, bn in enumerate(BIN_NAMES))))
        P("  anchor pairs: election %d, footprint %d" % (int(np.count_nonzero(self.src == SRC_ELECTION)),
                                                       int(np.count_nonzero(self.src == SRC_FOOTPRINT))))
        order = [("E1", ""), ("E2", "neither-jumped"), ("E2", "ours-jumped"), ("E2", "rockstar-jumped"),
                 ("E2", "both-jumped"), ("E2", "rockstar-phantom"), ("E3", "rockstar-identity-break"),
                 ("E3", "rockstar-lost-ours-continues"), ("E4", "raw-rockstar-lost-it-too"), ("E4", "merger-decision"),
                 ("E4", "other"), ("E4", "death-rockstar-does-not-see"), ("E2", "unavailable"), ("E4", "unavailable"),
                 ("E5", "both-ended"), ("E0", "ours-alive"),
                 ("E0", "ours-dead")]
        for j in sorted(self.steps):
            st = self.steps[j]
            cat, lab = st["cat"], st["lab"]
            base = np.array([x.split(";")[0] for x in lab], object) if lab.size else lab
            P("")
            P("  step j = %d -> output %d (a = %.4f, %s)%s" % (
                j, st["k"], st["a"], os.path.basename(st["hfile"]),
                "; WARNING: %d real chain rows whose Orig_halo_ID is not in out_%d" % (st["real_missing"], st["k"])
                if st["real_missing"] else ""))
            P(bin_header(width=40, what="category (bins: our Nbound at k0)"))
            for c, l in order:
                m = (cat == c) & ((base == l) if l else np.ones(cat.size, bool))
                if c == "E1" or m.any():
                    P(bin_line("%s %s" % (c, l), self.bin, m, width=40))
            mv = st["moved"]
            P("    E1 through a central-rule or footprint pair at %d (E2 under the election alone): %d = %s; by our Nbound "
              "bin %s" % (st["k"], int(mv.sum()), ", ".join("%s %d" % (l, int(np.count_nonzero(st["lab_el"] == l))) for l in
                                                           ("neither-jumped", "ours-jumped", "rockstar-jumped", "both-jumped")),
                         "/".join(str(int(np.count_nonzero(mv & (self.bin == x)))) for x in range(len(EDGES)))))
            for src in (SRC_ELECTION, SRC_FOOTPRINT):
                ms = self.src == src
                P("    anchor %-9s pairs %5d: E1 %d, E2 %d, E3 %d, E4 %d, E5 %d, E0 %d" % (
                    SRC_NAMES[src], int(ms.sum()), *[int(np.count_nonzero(ms & (cat == c))) for c in
                                                     ("E1", "E2", "E3", "E4", "E5", "E0")]))
            nabs = int(np.count_nonzero(np.array([";ours-absent" in x for x in lab], bool)))
            if nabs:
                P("    note: %d dead tracks of ours absent from the Orphan table (included above)" % nabs)
            e2 = cat == "E2"
            e2r = e2 & np.isfinite(st["rs_self"])
            P(qline("E2 ours self-overlap", st["ours_self"][e2], width=40))
            P(qline("E2 Rockstar self-overlap (real rows)", st["rs_self"][e2r], width=40))
            for x, bn in enumerate(BIN_NAMES):
                mb = self.bin == x
                P("      %-8s E2 %4d: ours %s | Rockstar %s" % (bn, int(np.count_nonzero(e2 & mb)), _q(st["ours_self"][e2 & mb]),
                                                          _q(st["rs_self"][e2r & mb])))
            e1 = cat == "E1"
            P(qline("E1 ours self-overlap", st["ours_self"][e1], width=40))
            P(qline("E1 Rockstar self-overlap", st["rs_self"][e1 & np.isfinite(st["rs_self"])], width=40))
            e2pt = e2 & (st["p_t_id"] >= 0)
            P("    E2: our track paired with another Rockstar halo %d, Rockstar chain halo paired with another track %d" % (
                int(e2pt.sum()), int(np.count_nonzero(e2 & (st["ptid"] >= 0)))))
            lost = (cat == "E3") & (base == "rockstar-lost-ours-continues")
            if lost.any():
                P("    E3 lost: our Nbound at k %s; d/R host %s; Depth at k 0/1/>=2: %d/%d/%d" % (
                    _q(st["nbj"][lost].astype(float)), _q(st["hdist"][lost & np.isfinite(st["hdist"])]),
                    int(np.count_nonzero(lost & (st["depj"] == 0))), int(np.count_nonzero(lost & (st["depj"] == 1))),
                    int(np.count_nonzero(lost & (st["depj"] >= 2)))))
            un = (cat == "E4") & (base == "death-rockstar-does-not-see")
            if un.any():
                nbv = st["unb_nb"][un]
                us = st["unb_status"][un]
                dn = us == "done"
                P("    E4 unseen: %d; independent unbinding of the Rockstar list: self-bound (>= %d) %d, not %d, "
                  "unconverged (maxit) %d, skipped %d, incomplete %d, empty %d, noncompact %d, off %d; bound/found %s; "
                  "orphan TracerPos inside "
                  "Rvir %d of %d; k - SnapDeath %s" % (
                      int(un.sum()), SELF_BOUND, int(np.count_nonzero(dn & (nbv >= SELF_BOUND))),
                      int(np.count_nonzero(dn & (nbv < SELF_BOUND))), int(np.count_nonzero(us == "unconverged")),
                      int(np.count_nonzero(us == "skipped")), int(np.count_nonzero(us == "incomplete")),
                      int(np.count_nonzero(us == "empty")), int(np.count_nonzero(us == "noncompact")),
                      int(np.count_nonzero((us == "off") | (us == ""))),
                      _q((nbv[dn] / np.maximum(st["unb_found"][un][dn], 1)).astype(float)),
                      int(np.count_nonzero(st["tr_in"][un] == 1)), int(np.count_nonzero(st["tr_in"][un] >= 0)),
                      _q((st["k"] - st["sd"][un][st["sd"][un] >= 0]).astype(float))))
                for x, bn in enumerate(BIN_NAMES):
                    mb = un & (self.bin == x)
                    if mb.any():
                        v, vs = st["unb_nb"][mb], st["unb_status"][mb]
                        P("      %-8s unseen %4d: self-bound %d, not %d, unconverged %d, tracer inside Rvir %d" % (
                            bn, int(mb.sum()), int(np.count_nonzero((vs == "done") & (v >= SELF_BOUND))),
                            int(np.count_nonzero((vs == "done") & (v < SELF_BOUND))),
                            int(np.count_nonzero(vs == "unconverged")), int(np.count_nonzero(st["tr_in"][mb] == 1))))
            e4 = cat == "E4"
            if e4.any():
                P("    E4 deaths: sank %d (SinkTrackId >= 0) of %d with an Orphan row" % (
                    int(np.count_nonzero(e4 & (st["sink"] >= 0))), int(np.count_nonzero(e4 & (st["sd"] >= 0)))))
        return "\n".join(L) + "\n"

    def write_csv(self, path):
        cols = ("k0,j,k,TrackId,rs_id_k0,bin,Nbound_k0,category,label,Nbound_k,Depth_k,hlist_id,phantom,rs_id_k,"
                "partner_of_track_rs_id,partner_of_chain_TrackId,ours_self,rs_self,SnapDeath,SinkTrackId,DescendantTrackId,"
                "unbind_found,unbind_nbound,unbind_status,tracer_in_rvir,host_dist_R,anchor_pair_source,pair_source_k,"
                "e1_by_added_pair")
        with open(path, "w") as f:
            f.write(cols + "\n")
            for j in sorted(self.steps):
                st = self.steps[j]
                for q in range(self.n):
                    v = [self.k0, j, st["k"], int(self.t[q]), int(self.rid[q]),
                         BIN_NAMES[self.bin[q]] if self.bin[q] >= 0 else "<20", int(self.nb0[q]), st["cat"][q], st["lab"][q],
                         int(st["nbj"][q]), int(st["depj"][q]), int(st["cr_id"][q]), int(st["phant"][q]), int(st["rj_id"][q]),
                         int(st["p_t_id"][q]), int(st["ptid"][q]), float(st["ours_self"][q]), float(st["rs_self"][q]),
                         int(st["sd"][q]), int(st["sink"][q]), int(st["desc"][q]), int(st["unb_found"][q]),
                         int(st["unb_nb"][q]), st["unb_status"][q] or "-", int(st["tr_in"][q]), float(st["hdist"][q]),
                         SRC_NAMES[int(self.src[q])],
                         SRC_NAMES[int(st["psrc_j"][q])], int(st["moved"][q])]
                    f.write(",".join(fmt(x) for x in v) + "\n")


# ---------------------------------------------------------------------------------- driver
def parse_list(s):
    return sorted(set(int(x) for x in s.split(",") if x.strip())) if s else []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", required=True, help="directory of PIG_k and PART_k")
    p.add_argument("--rockstar", required=True, help="Rockstar OUTBASE (out_N.list, halos_*.bin)")
    p.add_argument("--hlists", required=True, help="directory of the consistent-trees hlist_<scale>.list")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--census", default="128,140,152", help="census outputs (default 128,140,152)")
    p.add_argument("--anchors", default="128,140", help="identity anchors (default 128,140; empty: none)")
    p.add_argument("--steps", default="1,2,5", help="identity steps (default 1,2,5)")
    p.add_argument("--rs-snap-offset", type=int, default=0, help="Rockstar snapshot minus our SnapshotIndex (default 0)")
    p.add_argument("--maxset", type=int, default=20000, help="largest list unbound (48 N^2 bytes of pairs unchunked)")
    p.add_argument("--no-unbind", action="store_true", help="skip the independent unbinding (no PART is read)")
    p.add_argument("--min-np", type=int, default=20, help="Rockstar satellite universe: Mvir / m_p >= this (default 20)")
    p.add_argument("--nproc", type=int, default=0, help="unbinding workers (default: the CPUs of this process)")
    args = p.parse_args(argv)
    census, anchors, steps = parse_list(args.census), parse_list(args.anchors), parse_list(args.steps)
    if args.nproc <= 0:
        args.nproc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    os.makedirs(args.out, exist_ok=True)
    try:
        hl = Hlists(args.hlists)
        need = sorted(set(census) | set(anchors) | set(k0 + j for k0 in anchors for j in steps))
        log("outputs %s; census %s, anchors %s, steps %s; unbinding %s (%d workers, maxset %d)" % (
            need, census, anchors, steps, "off" if args.no_unbind else "on", args.nproc, args.maxset))
        texts = {}
        open_anchors = {}
        for k in need:
            tk = time.time()
            s = Output(k, args)
            sets = {}
            cen = None
            if k in census:
                cen = Census(s, hl, args)
                sets.update(cen.unbind_sets)
            if k in anchors:
                open_anchors[k] = Anchor(s, hl)
                log("output %d: anchor with %d A pairs" % (k, open_anchors[k].n))
            stepped = []
            for k0, an in open_anchors.items():
                j = k - k0
                if j in steps:
                    sets.update(an.step(s, hl, args))
                    stepped.append((an, j))
            s.drop()
            ures = {}
            if sets and not args.no_unbind:
                ures = run_unbinding(s.part, sets, args.nproc, args.maxset, s.a)
            if cen is not None:
                texts["census_%d" % k] = cen.finish(ures)
            for an, j in stepped:
                an.fill_unbinding(j, ures)
            for k0, an in list(open_anchors.items()):
                if all(j in an.steps or k0 + j > max(need) for j in steps):
                    txt = an.report(args)
                    with open(os.path.join(args.out, "identity_%d.txt" % k0), "w") as f:
                        f.write(txt)
                    an.write_csv(os.path.join(args.out, "identity_%d.csv" % k0))
                    texts["identity_%d" % k0] = txt
                    del open_anchors[k0]
            del s
            log("output %d done in %.0f s" % (k, time.time() - tk))
        summary = ["subhalo_rockstar_satellites: ours %s, Rockstar %s (snapshot offset %d), hlists %s" % (
            args.ours, args.rockstar, args.rs_snap_offset, args.hlists),
            "census %s, anchors %s, steps %s, min-np %d, unbinding %s (maxset %d)"
            % (census, anchors, steps, args.min_np, "off" if args.no_unbind else "on", args.maxset), ""]
        for k in census:
            summary.append(texts.get("census_%d" % k, "census_%d: not produced\n" % k))
        for k0 in anchors:
            summary.append(texts.get("identity_%d" % k0, "identity_%d: not produced\n" % k0))
        text = "\n".join(summary)
        with open(os.path.join(args.out, "summary.txt"), "w") as f:
            f.write(text)
        print(text)
        log("total wall time %.0f s" % (time.time() - T0))
        return 0
    except (SubhaloFormatError, InputError, IOError, OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
