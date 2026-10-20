#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the runtime subhalo finder's catalogue in a PIG_xxx with a Rockstar-BigFile run
(halofinder_phase3_plan.md section 6, halofinder_design.md 4.3, 7.10 and section 8).

Usage:
  subhalo_compare_rockstar.py PIG_xxx ROCKSTAR [options]
      ROCKSTAR is Rockstar's OUTBASE (out_N.list is read, N = the PIG's SnapshotIndex plus
      --rs-snap-offset, and the halos_*.bin files of snapshot N for the particle IDs), or one
      catalogue file: out_N.list, a find_parents output (out_N.list plus PID), a halos_*.ascii
      chunk or a consistent-trees hlist (Orig_halo_ID is then the Rockstar ID).
  subhalo_compare_rockstar.py PIG_xxx ROCKSTAR --field STRICT
      also compare the field SO masses MSO of our centrals with the Rockstar run made with
      STRICT_SO_MASSES = 1 (its Mvir_all is the SO mass from every particle in the sphere).

Hosts and subhalos. Rockstar's catalogue PID is the parents.c rule: the halo whose Rvir sphere holds
the centre, with a larger Rvir, and the smallest Vmax among those. A PID or pid column is used when the
catalogue has one (find_parents output, hlists); otherwise it is computed here with the same rule.
Rockstar hosts are PID = -1. Ours: centrals are Rank 0 with HostGroup >= 0, satellites Rank > 0;
hostless tracks are left out (counted).

Matching (--match):
  ids (default when the .bin files exist)
      Rockstar's .bin files list every halo's own particle IDs (its unique membership, unbound
      particles included; a host's list excludes its subhalos' particles). Our ten most-bound
      tracers of each subhalo (BindRank order, SubhaloTracerTypes) vote with weights
      1/(1 + sqrt(i)) for the Rockstar halo holding them; every particle of a Rockstar halo saved
      in the PIG votes for our subhalo binding it (BoundTrackId; -1, nobody, is a candidate;
      particles the PIG does not save do not vote). A pair is matched
      when the two votes elect each other. The tolerance below is then reported per pair.
  position
      centrals against Rockstar hosts, then satellites against Rockstar subhalos: the nearest
      Rockstar centre within the tolerance, one-to-one, closest pairs first.
  Tolerance: centre distance (our CorePos, Rockstar X/Y/Z, periodic) <= --rtol x R, R = our
  RBoundSO[vir] where defined, else Rockstar's Rvir; and the mass ratio (--mass, default
  MBoundSO[0], against Rockstar's Mvir) within a factor --mass-factor. Where our --mass value is
  undefined (-1) Mbound stands in, counted in the report.

Per matched pair (ours / Rockstar, both converted to Rockstar's units): Mvir (--mass), M200b, M200c,
M500c, M2500c (MBoundSO[1..4]), Rvir (RBoundSO[0]), Vmax (VmaxVir), Rs_Klypin (RsKlypin against
rs_klypin), Rs(fit) (our Phase 5 NFW fit Rs against Rockstar's fitted rs), Klypin/fit (our RsKlypin
against Rockstar's fitted rs, for reference), and the decision-44 versions on the field-SO virial
sphere of our centrals (every particle inside RSO[vir], Rockstar's host set): Rs(SO) (RsSO against
rs), Rs_Klypin(SO) (RsKlypinSO against rs_klypin), Klypin(SO)/fit (RsKlypinSO against rs), Vmax(SO)
(VmaxSO against vmax); Vrms, Spin (SpinPeebles), Spin_Bullock, b_to_a,
c_to_a, b_to_a(500c), c_to_a(500c), Xoff, Voff (VmaxVir, Vrms and Voff are physical km/s in our
catalogue as in Rockstar's, so they are not converted); p10/p50/p90 of each ratio over pairs where both sides are
defined (> 0), with the counts of pairs where our value is undefined (-1 until Phase 3 fills the
column), Rockstar's is (0), or our column is absent from the file. Rockstar's values are
bound-only (BOUND_PROPS 1); its host's set includes its subhalos' particles while our central's
bound set does not (design 4.1, D10), and our central's source is its FOF group only (D3, HBT
FeedCentrals), so central masses are expected slightly below Rockstar's.

--field: our centrals matched the same way against the strict run's hosts; MSO[0] against the
strict Mvir_all, MSO[1..4] against M200b..M2500c unless every halo has M200b = M200c = M500c = M2500c
= Mvir_all, the io_bgc2.c:230-231 threshold bug of builds before the fork's cc1630f (design
Appendix A), which is detected and reported;
RSO[0] against (3 M / 4 pi rho_vir)^(1/3) of the strict mass when the Header has SOThresholds.

Phase 5 summary (centrals <-> hosts within the tolerance, our Nbound >= --min-np): the median ratio
and the fraction of pairs within 20 % (Rs(fit), Rs_Klypin, Klypin/fit and the SO versions Rs(SO),
Rs_Klypin(SO), Klypin(SO)/fit), 5 % (Vrms) and 5 % (Vmax, Vmax(SO)), bound-set and SO versions side by
side, and
|RockstarCoreVel - Rockstar VX/VY/VZ| / Rockstar vrms (Rockstar's VX is its core velocity). Rockstar
falls back to its Klypin rs below 100 bound particles while our Rs is -1 there ('ours und').

Exit status: 0 when the fraction of our centrals with Nbound >= --min-np matched to a Rockstar host
within the tolerance is at least --accept-frac and the median Mvir ratio (and with --field the median
MSO[0] ratio) is within --median-tol of 1, the median Rs(fit) ratio within --rs-median-tol and the
median Vrms ratio within --vrms-median-tol (each skipped when fewer than 10 pairs define it or its
tolerance is <= 0); for hosts the Rs tolerance applies to the SO version Rs(SO) (decision 44: the
quantity computed on Rockstar's host set), falling back to the bound-set Rs(fit) with a note for a
catalogue without RsSO; --rs-bound-median-tol (default 0, off) adds the bound-set Rs(fit) target; or
--report-only is given; 1 otherwise; 2 when an input could not be read.

Phase 1 catalogues (every FOF group one subhalo): pass --phase1 so that only section 0 of the
particle blocks is read, and --mass Mbound, since MBoundSO is -1 before Phase 3.

Examples:
  python3 tools/subhalo_compare_rockstar.py output/PIG_004 rockstar-dm-small/ --pairs-out pairs_004.txt
  python3 tools/subhalo_compare_rockstar.py output/PIG_004 rockstar-dm-small/ --field rockstar-dm-small-strict/
  python3 tools/subhalo_compare_rockstar.py output/PIG_004 rockstar-dm-small/ --phase1 --mass Mbound --report-only
"""

from __future__ import print_function

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloFormatError  # noqa: E402
from subhalo_check import deferred_velocity_valid  # noqa: E402  (the Phase 5 velocity mask, audit R8)
from subhalo_compare_hbt import Ours, _weighted_vote, _lookup, _q  # noqa: E402

MSUN_G = 1.989e33          # MP-Gadget's solar mass (UnitMass_in_g = 1.989e43 for 1e10 Msun/h)
MPC_CM = 3.085678e24
KPC_CM = 3.085678e21
NSO = 5
SO_NAMES = ["vir", "200b", "200c", "500c", "2500c"]


class InputError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------- Rockstar side
_BIN_HEADER = np.dtype([
    ("magic", "<u8"), ("snap", "<i8"), ("chunk", "<i8"), ("scale", "<f4"), ("Om", "<f4"), ("Ol", "<f4"),
    ("h0", "<f4"), ("bounds", "<f4", 6), ("num_halos", "<i8"), ("num_particles", "<i8"), ("box_size", "<f4"),
    ("particle_mass", "<f4"), ("particle_type", "<i8"), ("format_revision", "<i4"), ("version", "S12")])
_BIN_HEADER_SIZE = 256
ROCKSTAR_MAGIC = 0xfadedacec0c0d0d0
# struct halo of halo.h (HALO_FORMAT_REVISION 1), natural alignment: 288 bytes
_BIN_HALO = np.dtype([
    ("id", "<i8"), ("pos", "<f4", 6), ("corevel", "<f4", 3), ("bulkvel", "<f4", 3),
    ("m", "<f4"), ("r", "<f4"), ("child_r", "<f4"), ("vmax_r", "<f4"), ("mgrav", "<f4"), ("vmax", "<f4"),
    ("rvmax", "<f4"), ("rs", "<f4"), ("klypin_rs", "<f4"), ("vrms", "<f4"), ("J", "<f4", 3), ("energy", "<f4"),
    ("spin", "<f4"), ("alt_m", "<f4", 4), ("Xoff", "<f4"), ("Voff", "<f4"), ("b_to_a", "<f4"), ("c_to_a", "<f4"),
    ("A", "<f4", 3), ("b_to_a2", "<f4"), ("c_to_a2", "<f4"), ("A2", "<f4", 3), ("bullock_spin", "<f4"),
    ("kin_to_pot", "<f4"), ("m_pe_b", "<f4"), ("m_pe_d", "<f4"), ("num_p", "<i8"), ("num_child_particles", "<i8"),
    ("p_start", "<i8"), ("desc", "<i8"), ("flags", "<i8"), ("n_core", "<i8"), ("min_pos_err", "<f4"),
    ("min_vel_err", "<f4"), ("min_bulkvel_err", "<f4"), ("type", "<i4"), ("sm", "<f4"), ("gas", "<f4"),
    ("bh", "<f4"), ("peak_density", "<f4"), ("av_density", "<f4")], align=True)
assert _BIN_HALO.itemsize == 288 and _BIN_HEADER.itemsize <= _BIN_HEADER_SIZE

# canonical name -> catalogue column names, lower case, "(n)" suffixes of hlists stripped; first found wins
_ALIASES = {
    "id": ["orig_halo_id", "id"],
    "pid": ["pid"],
    "mvir": ["mbound_vir", "mvir"],        # halos_*.ascii: mvir is all particles, mbound_vir bound
    "mvir_all": ["mvir_all"],
    "m200b": ["m200b"], "m200c": ["m200c"], "m500c": ["m500c"], "m2500c": ["m2500c"],
    "rvir": ["rvir"], "vmax": ["vmax"], "rs": ["rs"], "rs_klypin": ["rs_klypin"],
    "spin": ["spin"], "spin_bullock": ["spin_bullock"],
    "b_to_a": ["b_to_a"], "c_to_a": ["c_to_a"], "b_to_a(500c)": ["b_to_a(500c)"], "c_to_a(500c)": ["c_to_a(500c)"],
    "xoff": ["xoff"], "voff": ["voff"], "vrms": ["vrms"], "vx": ["vx"], "vy": ["vy"], "vz": ["vz"],
    "x": ["x"], "y": ["y"], "z": ["z"], "np": ["np", "num_p"], "type": ["type"], "scale": ["scale"],
}


def _parents_rule(pos, rvir_mpc, vmax, ids, box):
    """Rockstar's parents.c: halos in order of decreasing Vmax; each assigns itself as the parent of
    every halo with a smaller Rvir whose centre lies within its Rvir (range capped at box/2.01), later
    (smaller Vmax) halos overwriting earlier ones."""
    n = ids.size
    pid = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return pid
    order = np.argsort(-vmax, kind="stable")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.mod(pos, box), boxsize=box)
        query = lambda i, r: np.asarray(tree.query_ball_point(np.mod(pos[i], box), r), dtype=np.int64)  # noqa: E731
    except ImportError:  # pragma: no cover
        def query(i, r):
            d = pos - pos[i]
            d -= box * np.round(d / box)
            return np.nonzero((d ** 2).sum(axis=1) <= r * r)[0]
    for i in order.tolist():
        r = min(rvir_mpc[i], box / 2.01)
        nb = query(i, r)
        nb = nb[rvir_mpc[nb] < rvir_mpc[i]]
        pid[nb] = ids[i]
    return pid


class Rockstar(object):
    """One snapshot of a Rockstar run: the catalogue table and, when present, the .bin particle IDs."""

    def __init__(self, path, snap, need_ids=True, box=None, label="Rockstar"):
        self.label = label
        if os.path.isdir(path):
            self.dir = path
            self.catfile = os.path.join(path, "out_%d.list" % snap)
            if not os.path.isfile(self.catfile):
                raise InputError("%s: no out_%d.list" % (path, snap))
        elif os.path.isfile(path):
            self.dir = os.path.dirname(os.path.abspath(path))
            self.catfile = path
        else:
            raise InputError("%s: no such Rockstar directory or catalogue" % path)
        self.snap = snap
        self._read_catalogue(box)
        self.bin_files = []
        self.pids = np.zeros(0, np.int64)
        self.p_owner = np.zeros(0, np.int64)
        self.p_owner_row = np.zeros(0, np.int64)
        self.nduplicate = 0
        if need_ids:
            self._read_bins()

    def _read_catalogue(self, box):
        header, meta = None, {}
        with open(self.catfile) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if header is None:
                    header = line[1:].split()
                m = re.match(r"#a = ([0-9.eE+-]+)", line)
                if m:
                    meta["a"] = float(m.group(1))
                m = re.match(r"#Box size: ([0-9.eE+-]+)", line)
                if m:
                    meta["box"] = float(m.group(1))
                m = re.match(r"#Particle mass: ([0-9.eE+-]+)", line)
                if m:
                    meta["mpart"] = float(m.group(1))
        if header is None:
            raise InputError("%s: no column header line" % self.catfile)
        names = [re.sub(r"\(\d+\)$", "", h).lower() for h in header]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # an empty catalogue (no halos yet) is legal
            data = np.loadtxt(self.catfile, comments="#", ndmin=2)
        if data.size == 0:
            data = np.zeros((0, len(names)))
        if data.shape[1] < len(names) - 1:
            raise InputError("%s: %d columns in the header, %d in the rows" % (self.catfile, len(names), data.shape[1]))
        self.columns = {}
        for key, aliases in _ALIASES.items():
            for a in aliases:
                if a in names and names.index(a) < data.shape[1]:
                    self.columns[key] = data[:, names.index(a)]
                    break
        if "mbound_vir" in names and "mvir" in names and "mvir_all" not in self.columns:
            self.columns["mvir_all"] = data[:, names.index("mvir")]
        for need in ("id", "mvir", "rvir", "vmax", "x", "y", "z"):
            if need not in self.columns:
                raise InputError("%s: no %s column (header: %s)" % (self.catfile, need, " ".join(header[:12])))
        self.names = names
        self.n = data.shape[0]
        self.id = self.columns["id"].astype(np.int64)
        if "scale" in self.columns and self.n:
            sc = np.unique(np.round(self.columns["scale"], 5))
            meta["a"] = float(sc[0]) if sc.size == 1 else np.nan
        self.a = meta.get("a", np.nan)
        self.box_mpc = box if box else meta.get("box", np.nan)
        self.mpart = meta.get("mpart", np.nan)
        self.pos = np.stack([self.columns["x"], self.columns["y"], self.columns["z"]], axis=1) \
            if self.n else np.zeros((0, 3))
        self.mvir = self.columns["mvir"]
        self.rvir_kpc = self.columns["rvir"]
        self.vmax = self.columns["vmax"]
        self.type = self.columns["type"].astype(np.int64) if "type" in self.columns else np.zeros(self.n, np.int64)
        if "pid" in self.columns:
            self.pid = self.columns["pid"].astype(np.int64)
            self.pid_source = "PID column"
        else:
            if not np.isfinite(self.box_mpc):
                raise InputError("%s: no box size to compute PIDs" % self.catfile)
            self.pid = _parents_rule(self.pos, self.rvir_kpc * 1e-3, self.vmax, self.id, self.box_mpc)
            self.pid_source = "computed (parents.c rule)"
        self.host = self.pid < 0
        o = np.argsort(self.id, kind="stable")
        self._id_sorted, self._id_row = self.id[o], o.astype(np.int64)

    def row(self, ids):
        return _lookup(self._id_sorted, self._id_row, ids)

    def _read_bins(self):
        files = sorted(glob.glob(os.path.join(self.dir, "halos_*.bin")))
        ids, owner = [], []
        for fn in files:
            if fn.endswith(".coalesced.bin"):
                continue
            with open(fn, "rb") as f:
                raw = f.read(_BIN_HEADER_SIZE)
                if len(raw) < _BIN_HEADER_SIZE:
                    continue
                h = np.frombuffer(raw[:_BIN_HEADER.itemsize], _BIN_HEADER)[0]
                if int(h["magic"]) != ROCKSTAR_MAGIC:
                    raise InputError("%s: not a Rockstar binary catalogue (magic %x)" % (fn, int(h["magic"])))
                if int(h["snap"]) != self.snap:
                    continue
                nh, npart = int(h["num_halos"]), int(h["num_particles"])
                halos = np.frombuffer(f.read(nh * _BIN_HALO.itemsize), _BIN_HALO)
                p = np.frombuffer(f.read(npart * 8), "<i8")
            if halos.shape[0] != nh or p.shape[0] != npart or int(halos["num_p"].sum()) != npart:
                raise InputError("%s: truncated or inconsistent (halos %d/%d, particles %d/%d)"
                                 % (fn, halos.shape[0], nh, p.shape[0], npart))
            self.bin_files.append(fn)
            ids.append(p.astype(np.int64))
            owner.append(np.repeat(halos["id"].astype(np.int64), halos["num_p"].astype(np.int64)))
        if not self.bin_files:
            return
        ids, owner = np.concatenate(ids), np.concatenate(owner)
        # keep the rows of halos in the catalogue (hlists drop some), first occurrence of an ID
        keep = self.row(owner) >= 0
        ids, owner = ids[keep], owner[keep]
        o = np.argsort(ids, kind="stable")
        ids, owner = ids[o], owner[o]
        first = np.ones(ids.size, bool)
        first[1:] = ids[1:] != ids[:-1]
        self.nduplicate = int(np.count_nonzero(~first))
        self.pids, self.p_owner = ids[first], owner[first]
        self.p_owner_row = self.row(self.p_owner)

    def owner_row(self, ids):
        """Rockstar row of the halo whose .bin list holds each particle ID, -1 for none."""
        return _lookup(self.pids, self.p_owner_row, ids)


# ---------------------------------------------------------------------------------- our side
def parse_column(spec):
    m = re.match(r"^(\w+)(?:\[(\d+)\])?$", spec)
    if not m:
        raise ValueError("bad column %r (Name or Name[i])" % spec)
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None)


def our_column(S, spec, nrows):
    """(values, present) of a Subhalo column or one item of it, as float64; NaN-filled if absent."""
    name, idx = parse_column(spec)
    if name not in S:
        return np.full(nrows, np.nan), False
    v = S[name]
    if idx is not None:
        if v.ndim != 2 or idx >= v.shape[1]:
            return np.full(nrows, np.nan), False
        v = v[:, idx]
    elif v.ndim != 1:
        raise ValueError("column %s has %d items per row; give an index" % (name, v.shape[1]))
    return v.astype(np.float64), True


class Units(object):
    """Our unit system relative to Rockstar's: masses Msun/h, positions Mpc/h, radii kpc/h, km/s."""

    def __init__(self, header):
        g = lambda k, d: float(np.atleast_1d(header.get(k, d))[0])  # noqa: E731
        self.mass = g("UnitMass_in_g", 1.989e43) / MSUN_G
        ulen = g("UnitLength_in_cm", KPC_CM)
        self.pos = ulen / MPC_CM
        self.rad = ulen / KPC_CM
        # internal (particle) velocities only: the catalogue's velocity columns are km/s already
        self.vel = g("UnitVelocity_in_cm_per_s", 1e5) / 1e5
        self.box_mpc = g("BoxSize", np.nan) * self.pos
        self.a = g("Time", np.nan)
        mt = np.atleast_1d(header.get("MassTable", np.zeros(6)))
        self.mpart = float(mt[1]) * self.mass if mt.size > 1 and mt[1] > 0 else np.nan


# (label, our column, Rockstar key, kind); kind selects the unit conversion of our value:
# "mass", "rad" from the run's units; "kms" a catalogue velocity column, written in physical km/s
# by the property pass (subhalo_props.c) and not converted again; "vel" an internal velocity
# (UnitVelocity_in_cm_per_s), which no catalogue column compared here is; "one" dimensionless
QUANTITIES = [
    ("Mvir", None, "mvir", "mass"),
    ("M200b", "MBoundSO[1]", "m200b", "mass"),
    ("M200c", "MBoundSO[2]", "m200c", "mass"),
    ("M500c", "MBoundSO[3]", "m500c", "mass"),
    ("M2500c", "MBoundSO[4]", "m2500c", "mass"),
    ("Rvir", "RBoundSO[0]", "rvir", "rad"),
    ("Vmax", "VmaxVir", "vmax", "kms"),
    ("Rs_Klypin", "RsKlypin", "rs_klypin", "rad"),
    ("Rs(fit)", "Rs", "rs", "rad"),
    ("Klypin/fit", "RsKlypin", "rs", "rad"),
    ("Vmax(SO)", "VmaxSO", "vmax", "kms"),
    ("Rs_Klypin(SO)", "RsKlypinSO", "rs_klypin", "rad"),
    ("Rs(SO)", "RsSO", "rs", "rad"),
    ("Klypin(SO)/fit", "RsKlypinSO", "rs", "rad"),
    ("Vrms", "Vrms", "vrms", "kms"),
    ("Spin", "SpinPeebles", "spin", "one"),
    ("Spin_Bullock", "SpinBullock", "spin_bullock", "one"),
    ("b_to_a", "BtoA", "b_to_a", "one"),
    ("c_to_a", "CtoA", "c_to_a", "one"),
    ("b_to_a(500c)", "BtoA500", "b_to_a(500c)", "one"),
    ("c_to_a(500c)", "CtoA500", "c_to_a(500c)", "one"),
    ("Xoff", "Xoff", "xoff", "rad"),
    ("Voff", "Voff", "voff", "kms"),
]


def _conv(units, kind):
    return {"mass": units.mass, "rad": units.rad, "vel": units.vel, "kms": 1.0, "one": 1.0}[kind]


# ---------------------------------------------------------------------------------- matching
class Match(object):
    pass


def match_ids(ours, rs, ntracers):
    """Mutual election: our tracers vote for the Rockstar halo holding them, Rockstar members vote
    for our subhalo binding them. Returns (our rows, Rockstar rows)."""
    cand = rs.owner_row(ours.tr_id)
    v, w = _weighted_vote(ours.row(ours.tr_track), cand, 1.0 / (1.0 + np.sqrt(ours.tr_pos)))
    best_r_of_o = dict(zip(v.tolist(), w.tolist()))
    # particles absent from the PIG (field particles never tagged) carry no information: no vote
    saved = ours.types(rs.pids) >= 0
    o_rows = ours.row(ours.owner(rs.pids[saved]))
    v, w = _weighted_vote(rs.p_owner_row[saved], o_rows, np.ones(o_rows.size))
    best_o_of_r = dict(zip(v.tolist(), w.tolist()))
    oo, rr = [], []
    for o, r in best_r_of_o.items():
        if o >= 0 and r >= 0 and best_o_of_r.get(r, -2) == o:
            oo.append(o)
            rr.append(r)
    return np.array(oo, np.int64), np.array(rr, np.int64)


def _periodic_dist(a, b, box):
    d = a - b
    d -= box * np.round(d / box)
    return np.sqrt((d ** 2).sum(axis=-1))


def match_positions(o_sel, r_sel, opos, rpos, otol, rtol_r, omass, rmass, factor, box):
    """Nearest within tolerance, one-to-one, closest pairs first. otol: per-our-row distance tolerance
    (NaN: use the Rockstar halo's rtol_r)."""
    cand = []
    if o_sel.size == 0 or r_sel.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.mod(rpos[r_sel], box), boxsize=box)
        rmax = np.nanmax(rtol_r[r_sel])
        for o in o_sel.tolist():
            reach = otol[o] if np.isfinite(otol[o]) else rmax
            for k in tree.query_ball_point(np.mod(opos[o], box), reach):
                cand.append((o, int(r_sel[k])))
    except ImportError:  # pragma: no cover
        for o in o_sel.tolist():
            d = _periodic_dist(rpos[r_sel], opos[o], box)
            reach = otol[o] if np.isfinite(otol[o]) else np.inf
            for k in np.nonzero(d <= np.minimum(reach, np.where(np.isfinite(otol[o]), np.inf, rtol_r[r_sel])))[0]:
                cand.append((o, int(r_sel[k])))
    if not cand:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    c = np.array(cand, np.int64)
    d = _periodic_dist(opos[c[:, 0]], rpos[c[:, 1]], box)
    tol = np.where(np.isfinite(otol[c[:, 0]]), otol[c[:, 0]], rtol_r[c[:, 1]])
    ratio = omass[c[:, 0]] / np.where(rmass[c[:, 1]] > 0, rmass[c[:, 1]], np.nan)
    ok = (d <= tol) & (ratio <= factor) & (ratio >= 1.0 / factor)
    c, d = c[ok], d[ok]
    used_o, used_r, oo, rr = set(), set(), [], []
    for k in np.argsort(d, kind="stable").tolist():
        o, r = int(c[k, 0]), int(c[k, 1])
        if o in used_o or r in used_r:
            continue
        used_o.add(o)
        used_r.add(r)
        oo.append(o)
        rr.append(r)
    return np.array(oo, np.int64), np.array(rr, np.int64)


# ---------------------------------------------------------------------------------- comparison
class Side(object):
    """Our catalogue with the columns the comparison needs, in Rockstar's units."""

    def __init__(self, ours, units, mass_spec, rtol):
        S = ours.S
        n = ours.tid.size
        self.n = n
        self.S = S
        self.units = units
        self.pos = S["CorePos"].astype(np.float64) * units.pos
        self.nbound = ours.nbound
        self.rank = ours.rank
        self.host = S["HostGroup"].astype(np.int64)
        self.central = (self.rank == 0) & (self.host >= 0)
        self.satellite = self.rank > 0
        self.hostless = self.host < 0
        m, self.mass_present = our_column(S, mass_spec, n)
        m = m * units.mass
        mb = S["Mbound"].astype(np.float64) * units.mass
        self.mass_fallback = ~(m > 0)
        self.mass = np.where(self.mass_fallback, mb, m)
        r, _ = our_column(S, "RBoundSO[0]", n)
        self.rvir_kpc = np.where(r > 0, r * units.rad, np.nan)
        self.tol_mpc = rtol * self.rvir_kpc * 1e-3


def tolerance(side, rs, o, r, rtol, factor, box):
    d = _periodic_dist(side.pos[o], rs.pos[r], box)
    R = np.where(np.isfinite(side.rvir_kpc[o]), side.rvir_kpc[o], rs.rvir_kpc[r]) * 1e-3
    ratio = side.mass[o] / np.where(rs.mvir[r] > 0, rs.mvir[r], np.nan)
    ok = (d <= rtol * R) & (ratio <= factor) & (ratio >= 1.0 / factor)
    return ok, d / np.where(R > 0, R, np.nan), ratio


def ratio_table(side, rs, o, r, quantities, out, title):
    """Print the ratio table for pairs (o, r); returns {label: ratios over defined pairs}."""
    print("  %s" % title, file=out)
    print("    %-13s %-18s %6s %8s %8s %8s %8s   ratio p10/p50/p90" % (
        "quantity", "ours", "pairs", "defined", "ours und", "rs und", "absent"), file=out)
    res = {}
    for label, col, key, kind in quantities:
        if key not in rs.columns:
            print("    %-13s %-18s %6d  Rockstar column %s absent" % (label, col, o.size, key), file=out)
            continue
        ov, present = our_column(side.S, col, side.n)
        ov = ov[o] * _conv(side.units, kind)
        rv = rs.columns[key][r]
        if not present:
            print("    %-13s %-18s %6d %8d %8s %8s %8d   column absent from the PIG" % (
                label, col, o.size, 0, "-", "-", o.size), file=out)
            res[label] = np.zeros(0)
            continue
        oundef = ~(ov > 0) | ~np.isfinite(ov)
        rundef = ~(rv > 0)
        both = ~oundef & ~rundef
        rat = ov[both] / rv[both]
        res[label] = rat
        print("    %-13s %-18s %6d %8d %8d %8d %8d   %s" % (
            label, col, o.size, np.count_nonzero(both), np.count_nonzero(oundef), np.count_nonzero(rundef & ~oundef),
            0, _q(rat)), file=out)
    return res


def strict_bug(rs):
    """Fraction of halos whose four alternative strict masses all equal Mvir_all (io_bgc2.c:230-231)."""
    keys = ("m200b", "m200c", "m500c", "m2500c")
    if "mvir_all" not in rs.columns or not all(k in rs.columns for k in keys) or rs.n == 0:
        return np.nan
    ref = rs.columns["mvir_all"]
    same = np.ones(rs.n, bool)
    for k in keys:
        same &= np.abs(rs.columns[k] - ref) <= 1e-3 * np.abs(ref)
    return np.count_nonzero(same) / float(rs.n)


def run_match(side, ours, rs, args, box):
    M = Match()
    use_ids = args.match == "ids" or (args.match == "auto" and rs.bin_files)
    if args.match == "ids" and not rs.bin_files:
        raise InputError("--match ids: no halos_*.bin files of snapshot %d next to %s" % (rs.snap, rs.catfile))
    M.mode = "ids" if use_ids else "position"
    rsel = rs.type == 0
    if use_ids:
        o, r = match_ids(ours, rs, args.ntracers)
        keep = rsel[r] & ~side.hostless[o]
        M.o, M.r = o[keep], r[keep]
    else:
        rtol_r = args.rtol * rs.rvir_kpc * 1e-3
        oc, rc = match_positions(np.nonzero(side.central)[0], np.nonzero(rs.host & rsel)[0], side.pos, rs.pos,
                                 side.tol_mpc, rtol_r, side.mass, rs.mvir, args.mass_factor, box)
        os_, rsub = match_positions(np.nonzero(side.satellite)[0], np.nonzero(~rs.host & rsel)[0], side.pos, rs.pos,
                                    side.tol_mpc, rtol_r, side.mass, rs.mvir, args.mass_factor, box)
        M.o, M.r = np.concatenate([oc, os_]), np.concatenate([rc, rsub])
    M.ok, M.dR, M.mratio = tolerance(side, rs, M.o, M.r, args.rtol, args.mass_factor, box)
    return M


def report(side, ours, rs, M, args, box, out=sys.stdout, field=None):
    big = side.nbound >= args.min_np
    cen_big = np.nonzero(side.central & big)[0]
    print("Our catalogue %s (SnapshotIndex %d, a = %.4f): %d subhalos: %d centrals (%d with Nbound >= %d), "
          "%d satellites, %d hostless (left out)%s" % (
              ours.pigdir, ours.snap, side.units.a, side.n, np.count_nonzero(side.central), cen_big.size,
              args.min_np, np.count_nonzero(side.satellite), np.count_nonzero(side.hostless),
              " [--phase1: section 0 only]" if args.phase1 else ""), file=out)
    print("%s %s (a = %.4f): %d halos: %d hosts (PID -1), %d subhalos, %d not DM (left out); PID %s" % (
        rs.label, rs.catfile, rs.a, rs.n, np.count_nonzero(rs.host), np.count_nonzero(~rs.host),
        np.count_nonzero(rs.type != 0), rs.pid_source), file=out)
    if rs.bin_files:
        print("  particle IDs: %d .bin files, %d IDs%s" % (
            len(rs.bin_files), rs.pids.size,
            ", WARNING: %d IDs in more than one halo list (first kept)" % rs.nduplicate if rs.nduplicate else ""), file=out)
    if not side.mass_present:
        print("  note: --mass column %s is absent from the PIG; Mbound stands in" % args.mass, file=out)
    nfb = np.count_nonzero(side.mass_fallback & (side.central | side.satellite))
    if nfb:
        print("  note: %d subhalos with %s undefined: Mbound stands in for the mass tolerance" % (nfb, args.mass), file=out)
    nrv = np.count_nonzero(~np.isfinite(side.rvir_kpc) & (side.central | side.satellite))
    if nrv:
        print("  note: %d subhalos with RBoundSO[0] undefined: Rockstar's Rvir sets the distance tolerance" % nrv, file=out)
    print("Matching: %s%s; tolerance: distance <= %.2f R and %s / Mvir within a factor %.2f" % (
        M.mode, " (%d most-bound tracers vote; Rockstar members vote by plurality)" % args.ntracers if M.mode == "ids" else
        " (nearest within the tolerance, one-to-one)", args.rtol, args.mass, args.mass_factor), file=out)

    oc = side.central[M.o]
    rh = rs.host[M.r]
    kinds = [("central-host", oc & rh), ("central-sub", oc & ~rh), ("satellite-host", ~oc & rh), ("satellite-sub", ~oc & ~rh)]
    print("  matched pairs %d: %s; within tolerance %d" % (
        M.o.size, ", ".join("%s %d" % (k, np.count_nonzero(s)) for k, s in kinds), np.count_nonzero(M.ok)), file=out)
    if M.o.size:
        print("  pair distance / R p10/p50/p90 %s   %s / Mvir p10/p50/p90 %s" % (
            _q(M.dR[np.isfinite(M.dR)]), args.mass, _q(M.mratio[np.isfinite(M.mratio)])), file=out)

    in_pair = np.full(side.n, -1, np.int64)
    in_pair[M.o] = np.arange(M.o.size)
    pair_host = np.zeros(side.n, bool)       # our row is paired with a Rockstar host
    pair_ok = np.zeros(side.n, bool)         # ... and the pair is within the tolerance
    pair_host[M.o] = rs.host[M.r]
    pair_ok[M.o] = M.ok
    k = in_pair[cen_big]
    matched_host = pair_host[cen_big]
    within = matched_host & pair_ok[cen_big]
    frac = np.count_nonzero(within) / float(max(cen_big.size, 1))
    print("  our centrals with Nbound >= %d: %d; matched to a Rockstar host %d; within tolerance %d (%.1f%%)" % (
        args.min_np, cen_big.size, np.count_nonzero(matched_host), np.count_nonzero(within), 100 * frac), file=out)
    sat_big = np.nonzero(side.satellite & big)[0]
    ks = in_pair[sat_big]
    sat_sub = (ks >= 0) & ~pair_host[sat_big]
    sat_ok = sat_sub & pair_ok[sat_big]
    print("  our satellites with Nbound >= %d: %d; matched to a Rockstar subhalo %d; within tolerance %d" % (
        args.min_np, sat_big.size, np.count_nonzero(sat_sub), np.count_nonzero(sat_ok)), file=out)
    # unmatched
    rbig = rs.mvir >= args.min_np * (side.units.mpart if np.isfinite(side.units.mpart) else rs.mpart)
    rmatched = np.zeros(rs.n, bool)
    rmatched[M.r] = True
    um_c = cen_big[k < 0]
    print("  unmatched: our centrals >= %d: %d; our satellites >= %d: %d; Rockstar hosts with Mvir >= %d particles: %d "
          "of %d; Rockstar subhalos >= %d particles: %d of %d" % (
              args.min_np, um_c.size, args.min_np, np.count_nonzero(ks < 0), args.min_np,
              np.count_nonzero(rbig & rs.host & ~rmatched & (rs.type == 0)), np.count_nonzero(rbig & rs.host & (rs.type == 0)),
              args.min_np, np.count_nonzero(rbig & ~rs.host & ~rmatched & (rs.type == 0)),
              np.count_nonzero(rbig & ~rs.host & (rs.type == 0))), file=out)

    # ratio tables
    qs = [(lab, args.mass if col is None else col, key, kind) for lab, col, key, kind in QUANTITIES]
    sel_c = np.nonzero(oc & rh & M.ok & big[M.o])[0]
    sel_s = np.nonzero(~oc & ~rh & M.ok & big[M.o])[0]
    print("Ratios ours / Rockstar (pairs within tolerance, our Nbound >= %d; defined: both sides > 0; 'ours und': our "
          "value -1 (not computed) or <= 0; 'rs und': Rockstar's <= 0; 'absent': column not in the PIG):" % args.min_np, file=out)
    res = ratio_table(side, rs, M.o[sel_c], M.r[sel_c], qs, out, "centrals <-> hosts: %d pairs" % sel_c.size)
    ratio_table(side, rs, M.o[sel_s], M.r[sel_s], qs, out, "satellites <-> subhalos (secondary): %d pairs" % sel_s.size)
    medians = [("Mvir", res.get("Mvir", np.zeros(0)), args.median_tol)]
    phase5_summary(side, rs, M.o[sel_c], M.r[sel_c], res, out)
    # decision 44: the Rs target applies to the SO version (Rockstar's host set) where present
    if "RsSO" in side.S:
        medians.append(("Rs(SO)", res.get("Rs(SO)", np.zeros(0)), args.rs_median_tol))
        medians.append(("Rs(fit)", res.get("Rs(fit)", np.zeros(0)), args.rs_bound_median_tol))
    else:
        print("  note: no RsSO column (a catalogue before decision 44): --rs-median-tol applies to the bound-set Rs(fit)",
              file=out)
        medians.append(("Rs(fit)", res.get("Rs(fit)", np.zeros(0)), args.rs_median_tol))
    medians.append(("Vrms", res.get("Vrms", np.zeros(0)), args.vrms_median_tol))
    if field is not None:
        res_f = field_report(side, field, args, box, out)
        if res_f is not None:
            medians.append(("MSO[0]", res_f.get("MSO[0]", np.zeros(0)), args.median_tol))

    if cen_big.size:
        ok = frac >= args.accept_frac
        print("Acceptance: %.1f%% of our centrals with Nbound >= %d matched to a Rockstar host within tolerance (target %.0f%%)"
              % (100 * frac, args.min_np, 100 * args.accept_frac), file=out)
    else:
        ok = True
        print("Acceptance: no centrals with Nbound >= %d, not tested" % args.min_np, file=out)
    for label, rat, tol in medians:
        if tol > 0 and rat.size >= 10:
            med = float(np.median(rat))
            mok = abs(med - 1) <= tol
            print("  median %s ratio %.3f (target within %.2f of 1): %s" % (label, med, tol, "ok" if mok else "FAILED"),
                  file=out)
            ok = ok and mok
        else:
            print("  median %s ratio not tested (%d pairs define it%s)" % (
                label, rat.size, ", tolerance <= 0" if tol <= 0 else ""), file=out)

    if args.pairs_out:
        with open(args.pairs_out, "w") as f:
            f.write("# our_TrackId our_Rank our_Nbound rs_ID rs_PID within_tol dist/R mass_ratio %s\n" % " ".join(
                "%s/rs" % q[0] for q in qs))
            for j in range(M.o.size):
                o, r = int(M.o[j]), int(M.r[j])
                vals = []
                for lab, col, key, kind in qs:
                    ov, present = our_column(side.S, col, side.n)
                    v = ov[o] * _conv(side.units, kind)
                    rv = rs.columns.get(key)
                    vals.append("%.5g" % (v / rv[r]) if present and rv is not None and v > 0 and rv[r] > 0 else "nan")
                f.write("%d %d %d %d %d %d %.4f %.4f %s\n" % (
                    ours.tid[o], side.rank[o], side.nbound[o], rs.id[r], rs.pid[r], int(M.ok[j]), M.dR[j], M.mratio[j],
                    " ".join(vals)))
        print("  pairs written to %s" % args.pairs_out, file=out)
    return ok


def phase5_summary(side, rs, o, r, res, out):
    """The Phase 5 columns over the central-host pairs (o, r): median ratio and the fraction within
    20 % (radii) or 5 % (Vrms) of the pairs where both sides define the value; the core velocity
    offset against Rockstar's VX/VY/VZ (its core velocity) in units of Rockstar's vrms."""
    print("Phase 5 (centrals <-> hosts, %d pairs; bound set, then the field-SO virial sphere of decision 44):" % o.size,
          file=out)
    for label, tol in (("Rs(fit)", 0.2), ("Rs(SO)", 0.2), ("Rs_Klypin", 0.2), ("Rs_Klypin(SO)", 0.2), ("Klypin/fit", 0.2),
                       ("Klypin(SO)/fit", 0.2), ("Vmax", 0.05), ("Vmax(SO)", 0.05), ("Vrms", 0.05)):
        rat = res.get(label)
        if rat is None or rat.size == 0:
            print("    %-14s no pairs define it" % label, file=out)
            continue
        print("    %-14s %4d pairs: median ratio %.3f, within %2.0f%%: %.1f%%" % (
            label, rat.size, float(np.median(rat)), 100 * tol, 100.0 * np.count_nonzero(np.abs(rat - 1) <= tol) / rat.size),
            file=out)
    # the velocities are signed: valid where our virial aperture exists (Vrms >= 0) and every component
    # is finite, never "no component equal to -1" (Phase 5 audit R8; subhalo_check.py's mask)
    ovrms = side.S["Vrms"][o].astype(np.float64) if "Vrms" in side.S else np.full(o.size, -1.0)
    if all(k in rs.columns for k in ("vx", "vy", "vz", "vrms")) and "RockstarCoreVel" in side.S and o.size:
        ov = side.S["RockstarCoreVel"][o].astype(np.float64)
        rv = np.stack([rs.columns[k][r] for k in ("vx", "vy", "vz")], axis=1)
        sig = rs.columns["vrms"][r]
        d = deferred_velocity_valid(ovrms, ov) & (sig > 0)
        if np.any(d):
            x = np.sqrt(((ov[d] - rv[d]) ** 2).sum(axis=1)) / sig[d]
            print("    |RockstarCoreVel - Rockstar VX| / vrms: %d pairs, p10/p50/p90 %s" % (np.count_nonzero(d), _q(x)), file=out)
    if "BulkVel" in side.S and "CoreVel" in side.S and o.size:
        bv, cv = side.S["BulkVel"][o].astype(np.float64), side.S["CoreVel"][o].astype(np.float64)
        sig = ovrms
        d = deferred_velocity_valid(sig, bv) & np.all(np.isfinite(cv), axis=1) & (sig > 0)
        if np.any(d):
            x = np.sqrt(((bv[d] - cv[d]) ** 2).sum(axis=1)) / sig[d]
            print("    ours |BulkVel - CoreVel| / Vrms: p10/p50/p90 %s" % _q(x), file=out)


def field_report(side, field, args, box, out):
    """MSO of our centrals against the STRICT_SO_MASSES = 1 run."""
    rs, M = field
    print("Field SO: %s (a = %.4f), %d halos, %d hosts; matching %s" % (
        rs.catfile, rs.a, rs.n, np.count_nonzero(rs.host), M.mode), file=out)
    if "mvir_all" not in rs.columns:
        print("  the strict catalogue has no Mvir_all column; no field comparison", file=out)
        return None
    bug = strict_bug(rs)
    bugged = np.isfinite(bug) and bug > 0.5
    print("  strict alternative masses equal to Mvir_all for %.1f%% of halos%s" % (
        100 * (bug if np.isfinite(bug) else 0),
        ": the io_bgc2.c:230-231 threshold bug, only the vir mass is compared" if bugged else ""), file=out)
    big = side.nbound >= args.min_np
    sel = np.nonzero(side.central[M.o] & rs.host[M.r] & M.ok & big[M.o])[0]
    o, r = M.o[sel], M.r[sel]
    qs = [("MSO[0]", "MSO[0]", "mvir_all", "mass")]
    if not bugged:
        qs += [("MSO[%d]" % i, "MSO[%d]" % i, k, "mass") for i, k in ((1, "m200b"), (2, "m200c"), (3, "m500c"), (4, "m2500c"))]
    res = ratio_table(side, rs, o, r, qs, out, "centrals <-> strict hosts: %d pairs (within tolerance, Nbound >= %d)"
                      % (sel.size, args.min_np))
    # MSO against the bound mass of the same object, and RSO against the strict radius
    mso, present = our_column(side.S, "MSO[0]", side.n)
    mb, _ = our_column(side.S, "MBoundSO[0]", side.n)
    if present:
        d = (mso[o] > 0) & (mb[o] > 0)
        print("    MSO[0] / MBoundSO[0] of the same centrals: %d defined, p10/p50/p90 %s" % (
            np.count_nonzero(d), _q(mso[o][d] / mb[o][d])), file=out)
    thr = side.so_thresholds
    rso, rpresent = our_column(side.S, "RSO[0]", side.n)
    if thr is not None and rpresent:
        rho = float(thr[0]) * side.units.mass / side.units.rad ** 3   # Msun/h per (kpc/h)^3, comoving
        rstrict = (3 * rs.columns["mvir_all"][r] / (4 * np.pi * rho)) ** (1.0 / 3)
        v = rso[o] * side.units.rad
        d = (v > 0) & (rstrict > 0)
        print("    RSO[0] / (3 Mvir_all / 4 pi rho_vir)^(1/3): %d defined, p10/p50/p90 %s" % (
            np.count_nonzero(d), _q(v[d] / rstrict[d])), file=out)
    return res


def load(args):
    ours = Ours(args.pigdir, args.ntracers, phase1=args.phase1)
    units = Units(ours.cat.header)
    side = Side(ours, units, args.mass, args.rtol)
    thr = ours.cat.header.get("SOThresholds")
    side.so_thresholds = np.atleast_1d(thr).astype(np.float64) if thr is not None else None
    snap = ours.snap + args.rs_snap_offset if args.rs_snap is None else args.rs_snap
    need_ids = args.match != "position"
    rs = Rockstar(args.rockstar, snap, need_ids=need_ids, box=units.box_mpc)
    box = units.box_mpc
    for r in [rs]:
        if np.isfinite(r.a) and np.isfinite(units.a) and abs(r.a - units.a) > 1e-3 * units.a:
            raise InputError("%s is at a = %.5f, the PIG at a = %.5f: give --rs-snap or --rs-snap-offset"
                             % (r.catfile, r.a, units.a))
    field = None
    if args.field:
        rf = Rockstar(args.field, snap, need_ids=need_ids, box=units.box_mpc, label="Rockstar strict")
        if np.isfinite(rf.a) and np.isfinite(units.a) and abs(rf.a - units.a) > 1e-3 * units.a:
            raise InputError("%s is at a = %.5f, the PIG at a = %.5f" % (rf.catfile, rf.a, units.a))
        field = rf
    return ours, side, rs, field, box


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pigdir", help="PIG_xxx")
    p.add_argument("rockstar", help="Rockstar OUTBASE (out_N.list, halos_*.bin) or one catalogue file")
    p.add_argument("--field", default=None, help="the STRICT_SO_MASSES = 1 run (OUTBASE or catalogue file): compare MSO")
    p.add_argument("--match", choices=["auto", "ids", "position"], default="auto",
                   help="ids when the .bin files exist (auto), else position")
    p.add_argument("--mass", default="MBoundSO[0]", help="our mass compared with Rockstar's Mvir (default MBoundSO[0]; "
                   "Mbound for Phase 1/2a catalogues)")
    p.add_argument("--rtol", type=float, default=0.5, help="centre distance tolerance in units of R (default 0.5)")
    p.add_argument("--mass-factor", type=float, default=2.0, help="mass ratio tolerance factor (default 2)")
    p.add_argument("--ntracers", type=int, default=10, help="tracers per subhalo for the ID match (default 10)")
    p.add_argument("--min-np", type=int, default=100, help="our Nbound threshold of the summary (default 100)")
    p.add_argument("--accept-frac", type=float, default=0.9, help="target matched fraction of centrals (default 0.9)")
    # decision 63: the FOF-only central source (decision 52) puts the median Mvir ratio to Rockstar at
    # 0.90-0.97 over the 512^3 run, so the target is 20 %, not 10 %
    p.add_argument("--median-tol", type=float, default=0.2,
                   help="target |median ratio - 1| for Mvir and, with --field, MSO[0]; <= 0 disables (default 0.2)")
    p.add_argument("--rs-median-tol", type=float, default=0.2,
                   help="target |median Rs ratio - 1| of hosts: Rs(SO) (RsSO, decision 44), or the bound-set Rs(fit) "
                   "for a catalogue without RsSO (default 0.2; <= 0 disables)")
    p.add_argument("--rs-bound-median-tol", type=float, default=0.0,
                   help="target |median bound-set Rs(fit) ratio - 1| when RsSO is present (default 0: off)")
    p.add_argument("--vrms-median-tol", type=float, default=0.05,
                   help="target |median Vrms ratio - 1| (Phase 5; default 0.05; <= 0 disables)")
    p.add_argument("--rs-snap", type=int, default=None, help="Rockstar snapshot number (default: SnapshotIndex + offset)")
    p.add_argument("--rs-snap-offset", type=int, default=0, help="Rockstar snapshot minus our SnapshotIndex (default 0)")
    p.add_argument("--phase1", action="store_true", help="Phase 1 catalogue: read section 0 of the particle blocks only")
    p.add_argument("--report-only", action="store_true", help="exit 0 whatever the figures")
    p.add_argument("--pairs-out", default=None, help="write every matched pair to this text file")
    args = p.parse_args(argv)
    try:
        parse_column(args.mass)
        ours, side, rs, rf, box = load(args)
        M = run_match(side, ours, rs, args, box)
        field = None
        if rf is not None:
            field = (rf, run_match(side, ours, rf, args, box))
        ok = report(side, ours, rs, M, args, box, field=field)
        return 0 if (ok or args.report_only) else 1
    except (SubhaloFormatError, InputError, IOError, OSError, ValueError) as e:
        print("error: %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
