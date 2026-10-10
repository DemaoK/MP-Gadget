#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a sequence of the runtime subhalo finder's PIG catalogues as consistent-trees input
(halofinder_phase4_plan.md section 1 item 2, halofinder_design.md D9, 5.1-5.4 and 7.7).

Usage:
  subhalo_to_ctrees.py OUTPUTDIR|PIG_xxx... -o CTDIR [--mass auto|bound|field] [--orphans]
                       [--orphan-mass M] [--geometric-parents] [--force-res MPC] [--validate]
                       [--run CTREES_SRC]
  subhalo_to_ctrees.py --compare-hlist HLIST OUTPUTDIR [--rockstar-hlist HLIST2]

Writes, as Rockstar plus consistent-trees' gen_merger_cfg.pl would:
  CTDIR/out_<SnapshotIndex>.list      one ascii catalogue per PIG with Rockstar's header
  CTDIR/outputs/scales.txt            "<SnapshotIndex> <scale>" for every output with halos
  CTDIR/outputs/merger_tree.cfg       Om/Ol/h0, BOX_WIDTH, the paths, MASS_RES_OK, the extra
                                      columns and their interpolation codes
so that `perl do_merger_tree.pl CTDIR/outputs/merger_tree.cfg` (then halo_trees_to_catalog.pl)
runs from the consistent-trees directory (--run does both). consistent-trees aborts on an output
with too few halos to calibrate its metric ("too few halos at scale factor ..."): leave such early
outputs out of scales.txt with --first-snapshot.

Nodes and links (design 7.7). A node is (TrackId, output). Every resolved track (Subhalo/*) is a
halo; with --orphans, a track whose merger descendant is -1 is also exported as orphan nodes from
its death to the last output (or to its resurrection), with Mvir = --orphan-mass (default 0, the
design's M = 0 node; consistent-trees silently drops rows with Mvir <= 0, halo_io.c:230, so pass a
small positive mass to keep them in its trees) and Vmax = 0. IDs are dense per output, resolved
tracks by TrackId first, then orphan nodes by TrackId. DescID of node (T, n), n' the next output:
  identity   T resolved (or exported as an orphan node) at n'      -> the ID of (T, n')
  merger     T an orphan at n' with DescendantTrackId D >= 0 in its row at n' (the value stored
             at n' describes the transition n -> n')                -> the ID of (D, n')
  terminal   D = -1, or the last output                             -> -1
A track that resurrects after a merger edge reappears with no progenitor (its TrackId column
links the two pieces); consistent-trees allows one descendant per halo.

Columns (ours -> consistent-trees input, units as Rockstar's out_*.list):
  ID        dense per output                  DescID   as above
  Mvir      Msun/h: the export mass (--mass): field = MSO[vir] for centrals (Rank 0) when the
            field SO is filled, bound = MBoundSO[vir] for centrals when filled; otherwise, and
            for every satellite, Mbound. auto (default) = field if any central has MSO[vir] > 0,
            else bound. The header and cfg record the choice and the fallback counts.
  Vmax      Vmax (km/s, physical; HBT convention, bound particles)
  Vrms      Vrms when filled (Phase 3), else sqrt(2 Ekin), the rms speed of the bound set in its
            CoM frame, km/s
  Rvir      kpc/h comoving: RSO[vir] / RBoundSO[vir] matching the mass when filled, else
            consistent-trees' own fallback cbrt(3 M / (4 pi Delta_vir(a) rho_m)) (halo_io.c:232)
  Rs        Rs when filled, else RsKlypin when filled, else Rvir / c(M, a) with consistent-trees'
            concentration (masses.c:45)
  Np        Nbound (0 for orphan nodes)
  X Y Z     CorePos (Mpc/h comoving; orphan nodes TracerPos)
  VX VY VZ  CoreVel (km/s physical peculiar; orphan nodes TracerVel)
  JX JY JZ  JVir x Mvir (Msun/h Mpc/h km/s physical) when JVir is filled, else 0
  Spin      SpinPeebles when filled, else 0
  extras (EXTRA_PARAMS, passed through by consistent-trees into trees and hlists):
    rs_klypin Mvir_all M200b M200c M500c M2500c Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y]
    A[z] T/|U|  (Rockstar's names: RsKlypin, MSO[vir] or the export mass, the 200b..2500c masses
    of the export definition, Xoff, Voff, SpinBullock, BtoA, CtoA, AxisA, KinToPot; -1 where the
    finder has not filled them; Rockstar's 500c shape columns are dropped because
    consistent-trees takes at most 30 extra columns)
    TrackId Mbound Nbound Rank Depth HostGroup ParentTrackId UpParentTrackId MboundPeak VmaxPeak
    SnapAcc Orphan  (ours, copied unchanged for phantoms: interpolation code 'd')
  ParentTrackId/UpParentTrackId are pid/upid as TrackIds: NestedParentTrackId and CentralTrackId
  (-1 for centrals, a central is not its own upid) by default; with --geometric-parents, the
  consistent-trees rule (find_parents_and_cleanup.c:742-772: halos in Vmax-descending order, a
  halo whose centre lies inside the Rvir of a halo with larger Vmax gets pid = the last such
  halo, upid = the first). consistent-trees ignores input pid/upid and recomputes its own
  geometric ones, so these columns are the only way the membership parents reach its outputs.

--validate parses the written catalogues back and checks the structural rules: IDs dense and
unique per output, every DescID an ID of the next output with halos (or -1), Mvir > 0 and Vmax >
0 on every non-orphan row, every resolved track present, identity edges for every track resolved
at consecutive outputs, merger edges equal to DescendantTrackId, parents existing at the same
output. Exit 1 on a violation.

--compare-hlist HLIST OUTPUTDIR: match the rows of a consistent-trees hlist (made from our
export, so it carries TrackId) to our tracks at the hlist's scale, and compare Mpeak with
MboundPeak, Vpeak with VmaxPeak and Acc_Scale with the scale of SnapAcc (in the export's mass
units). With --rockstar-hlist, also match the hlist's halos to those of a consistent-trees run on
Rockstar at the same scale (nearest centre within the Rockstar Rvir, mutual) and compare Mvir,
Vmax, Mpeak, Vpeak, Acc_Scale.
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloSequence, SubhaloFormatError  # noqa: E402

CM_PER_MPC = 3.085678e24
G_PER_MSUN = 1.989e33
RHO_CRIT0 = 2.77519737e11  # Msun/h / (Mpc/h)^3, consistent-trees halo_io.c

REGULAR = ["ID", "DescID", "Mvir", "Vmax", "Vrms", "Rvir", "Rs", "Np", "X", "Y", "Z", "VX", "VY", "VZ",
           "JX", "JY", "JZ", "Spin"]
# consistent-trees accepts at most MAX_EXTRA_PARAMS = 30 extra columns (src/tree_halo.h): Rockstar's
# 500c shape columns are left out to make room for ours (27 in all).
ROCKSTAR_EXTRAS = ["rs_klypin", "Mvir_all", "M200b", "M200c", "M500c", "M2500c", "Xoff", "Voff",
                   "spin_bullock", "b_to_a", "c_to_a", "A[x]", "A[y]", "A[z]", "T/|U|"]
MAX_EXTRA_PARAMS = 30
OUR_EXTRAS = ["TrackId", "Mbound", "Nbound", "Rank", "Depth", "HostGroup", "ParentTrackId",
              "UpParentTrackId", "MboundPeak", "VmaxPeak", "SnapAcc", "Orphan"]
EXTRAS = ROCKSTAR_EXTRAS + OUR_EXTRAS
INTERP = "c" + "l" * (len(ROCKSTAR_EXTRAS) - 1) + "d" * len(OUR_EXTRAS)
assert len(EXTRAS) <= MAX_EXTRA_PARAMS
INT_COLS = set(["ID", "DescID", "Np", "TrackId", "Nbound", "Rank", "Depth", "HostGroup", "ParentTrackId",
                "UpParentTrackId", "SnapAcc", "Orphan"])


def _attr(h, k, default):
    v = h.get(k, default)
    return float(np.atleast_1d(v)[0]) if v is not None else default


def delta_vir(a, om, ol):
    """consistent-trees masses.c:40 (Bryan & Norman relative to the mean matter density)."""
    x = 1.0 / (1.0 + a ** 3 * ol / om) - 1.0
    return (18 * np.pi ** 2 + 82.0 * x - 39 * x * x) / (1.0 + x)


def rvir_from_mass(m, a, om, ol):
    """kpc/h comoving, as consistent-trees halo_io.c:232 does for rows without Rvir."""
    rho = delta_vir(a, om, ol) * RHO_CRIT0 * om
    return np.cbrt(np.maximum(m, 0) / (4.0 * np.pi * rho / 3.0)) * 1000.0


def concentration(m, a):
    """consistent-trees masses.c:45."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, 10 ** (2.2358 - 0.10 * np.log10(np.where(m > 0, m, 1))) * a, 0)


def _filled(v):
    """Rows where a finder column holds a value (the finder writes -1 for 'not computed')."""
    v = np.asarray(v)
    return v != -1 if v.ndim == 1 else ~np.all(v == -1, axis=1)


class Export(object):
    """The consistent-trees node table built from a SubhaloSequence."""

    def __init__(self, seq, mass="auto", orphans=False, orphan_mass=0.0, geometric=False, force_res=None,
                 first_snapshot=-1):
        self.seq = seq
        self.first_snapshot = first_snapshot
        if len(seq) == 0:
            raise SubhaloFormatError("no PIG with subhalo finder blocks")
        h = seq[len(seq) - 1].header
        self.om = _attr(h, "Omega0", 0.3)
        self.ol = _attr(h, "OmegaLambda", 1 - self.om)
        self.h0 = _attr(h, "HubbleParam", 0.7)
        self.mconv = _attr(h, "UnitMass_in_g", 1.989e43) / G_PER_MSUN
        self.lmpc = _attr(h, "UnitLength_in_cm", 3.085678e21) / CM_PER_MPC
        self.box = _attr(h, "BoxSize", np.nan) * self.lmpc
        mt = np.atleast_1d(h.get("MassTable", np.zeros(6)))
        self.part_mass = float(mt[1]) * self.mconv if len(mt) > 1 and mt[1] > 0 else np.nan
        N = seq.nodes()
        if not np.isfinite(self.part_mass):
            r = N["Resolved"] & (N["Nbound"] > 0)
            self.part_mass = float(np.min(N["Mbound"][r] / N["Nbound"][r])) * self.mconv if np.any(r) else 1.0
        self.force_res = force_res if force_res else self._default_force_res()
        cen = N["Resolved"] & (N["Rank"] == 0)
        has_field = bool(np.any(N["MSO"][cen][:, 0] > 0)) if "MSO" in N and np.any(cen) else False
        self.mass = ("field" if has_field else "bound") if mass == "auto" else mass
        self.orphans = orphans
        self.orphan_mass = orphan_mass
        self.geometric = geometric
        self.counts = {}
        self._build(N)

    def _default_force_res(self):
        """1/30 of the mean DM spacing (MP-Gadget's default softening), from the sibling PART's
        TotNumPart when present, else from the DM particle mass (DM-only estimate). Mpc/h."""
        n = None
        try:
            import bigfile
            pig = self.seq.paths[-1]
            part = os.path.join(os.path.dirname(pig.rstrip("/")), os.path.basename(pig.rstrip("/")).replace("PIG_", "PART_"))
            if os.path.isdir(part):
                n = float(np.atleast_1d(bigfile.File(part)["Header"].attrs["TotNumPart"])[1])
        except Exception:
            n = None
        if not n:
            n = self.om * RHO_CRIT0 * self.box ** 3 / self.part_mass
        return self.box / np.cbrt(n) / 30.0

    # ------------------------------------------------------------------ nodes and links
    def _build(self, N):
        seq = self.seq
        nout = len(seq)
        tid, out, res = N["TrackId"].astype(np.int64), N["Output"].astype(np.int64), N["Resolved"]
        exported = res.copy()
        orphan_node = np.zeros(tid.size, bool)
        # merger relation: for each node (T, n) resolved with (T, n+1) an orphan row
        nxt_same = np.zeros(tid.size, bool)
        nxt_same[:-1] = (tid[1:] == tid[:-1]) & (out[1:] == out[:-1] + 1)
        nxt = np.arange(tid.size) + 1
        terminal_from = np.zeros(tid.size, bool)
        dying = res & nxt_same & ~np.append(res[1:], False)
        desc_track = np.full(tid.size, -1, np.int64)
        desc_track[dying] = N["DescendantTrackId"][nxt[dying]]
        if self.orphans:
            # orphan nodes after a terminal death (D = -1) until resurrection or the last output
            k = np.nonzero(dying & (desc_track < 0))[0]
            for i in k.tolist():
                j = i + 1
                while j < tid.size and tid[j] == tid[i] and not res[j]:
                    orphan_node[j] = True
                    j += 1
                terminal_from[i] = True
            exported |= orphan_node
        self.counts["merger edges"] = int(np.count_nonzero(dying & (desc_track >= 0)))
        self.counts["terminal deaths"] = int(np.count_nonzero(dying & (desc_track < 0)))
        self.counts["orphan nodes"] = int(np.count_nonzero(orphan_node))
        self.counts["resurrections"] = int(np.count_nonzero(res[1:] & ~res[:-1] & (tid[1:] == tid[:-1])))
        # ids: dense per output, resolved by TrackId then orphan nodes by TrackId
        ids = np.full(tid.size, -1, np.int64)
        for i in range(nout):
            m = np.nonzero(exported & (out == i))[0]
            o = np.lexsort((tid[m], ~res[m]))
            ids[m[o]] = np.arange(m.size)
        # lookup (output, TrackId) -> node index among exported
        key = out * (int(tid.max()) + 2 if tid.size else 1) + tid
        ex = np.nonzero(exported)[0]
        so = np.argsort(key[ex])
        skey, sidx = key[ex][so], ex[so]

        def find(o, t):
            q = o * (int(tid.max()) + 2 if tid.size else 1) + t
            p = np.clip(np.searchsorted(skey, q), 0, max(skey.size - 1, 0))
            hit = (skey.size > 0) & (skey[p] == q) if skey.size else np.zeros(np.shape(q), bool)
            return np.where(hit, sidx[p] if skey.size else -1, -1)

        desc = np.full(tid.size, -1, np.int64)
        # identity: next row of the same track at the next output, exported
        ident = exported & nxt_same & np.append(exported[1:], False)
        desc[ident] = ids[nxt[ident]]
        # merger edges
        mer = exported & dying & (desc_track >= 0)
        tgt = find(out[mer] + 1, desc_track[mer])
        bad = tgt < 0
        self.counts["merger edges to unresolved tracks"] = int(np.count_nonzero(bad))
        d = np.where(bad, -1, ids[np.where(bad, 0, tgt)])
        desc[np.nonzero(mer)[0]] = d
        self.N = N
        self.exported = exported
        self.orphan_node = orphan_node
        self.ids = ids
        self.desc = desc
        self.desc_track = desc_track
        self.dying = dying
        self.find = find

    # ------------------------------------------------------------------ columns
    def rows(self, i):
        """The out_N.list table of output i: dict column -> array, rows in ID order."""
        N = self.N
        m = np.nonzero(self.exported & (N["Output"] == i))[0]
        m = m[np.argsort(self.ids[m])]
        a = float(self.seq.scale_factors[i])
        res = N["Resolved"][m]
        orph = ~res
        cen = res & (N["Rank"][m] == 0)
        mb = N["Mbound"][m].astype(np.float64) * self.mconv
        mass = mb.copy()
        rad = np.zeros(m.size)
        src = "MSO" if self.mass == "field" else "MBoundSO"
        rsrc = "RSO" if self.mass == "field" else "RBoundSO"
        so_m = N[src][m][:, 0].astype(np.float64) * self.mconv
        so_r = N[rsrc][m][:, 0].astype(np.float64) * self.lmpc * 1000.0
        use = cen & (so_m > 0)
        if self.mass == "field":
            # field SO missing on a central: fall back to the bound SO, then Mbound
            bso = N["MBoundSO"][m][:, 0].astype(np.float64) * self.mconv
            bsr = N["RBoundSO"][m][:, 0].astype(np.float64) * self.lmpc * 1000.0
            fb = cen & ~use & (bso > 0)
            mass[fb], rad[fb] = bso[fb], np.where(bsr[fb] > 0, bsr[fb], 0)
            self.counts["centrals with the bound SO fallback"] = self.counts.get("centrals with the bound SO fallback", 0) + int(np.count_nonzero(fb))
        mass[use] = so_m[use]
        rad[use] = np.where(so_r[use] > 0, so_r[use], 0)
        from_so = use.copy()
        if self.mass == "field":
            from_so |= fb
        nfb = int(np.count_nonzero(cen & ~from_so))
        self.counts["centrals exported with Mbound"] = self.counts.get("centrals exported with Mbound", 0) + nfb
        bsr_sat = N["RBoundSO"][m][:, 0].astype(np.float64) * self.lmpc * 1000.0
        sat = res & ~cen
        rad[sat] = np.where(bsr_sat[sat] > 0, bsr_sat[sat], 0)
        mass[orph] = self.orphan_mass
        need = ~(rad > 0)
        self.counts["Rvir from Mvir"] = self.counts.get("Rvir from Mvir", 0) + int(np.count_nonzero(need & res))
        rad[need] = rvir_from_mass(mass[need], a, self.om, self.ol)
        rs = N["Rs"][m].astype(np.float64) * self.lmpc * 1000.0
        rsk = N["RsKlypin"][m].astype(np.float64) * self.lmpc * 1000.0
        c = concentration(mass, a)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs_fb = np.where(c > 0, rad / np.where(c > 0, c, 1), 0)
        rs = np.where(rs > 0, rs, np.where(rsk > 0, rsk, rs_fb))
        vmax = np.where(res, N["Vmax"][m], 0).astype(np.float64)
        ek = N["Ekin"][m].astype(np.float64)
        vrms = np.where(N["Vrms"][m] > 0, N["Vrms"][m], np.where(ek > 0, np.sqrt(2 * np.maximum(ek, 0)), 0))
        vrms = np.where(res, vrms, 0)
        pos = np.where(res[:, None], N["CorePos"][m], N["TracerPos"][m]) * self.lmpc
        pos = np.mod(pos, self.box)
        vel = np.where(res[:, None], N["CoreVel"][m], N["TracerVel"][m]).astype(np.float64)
        jv = N["JVir"][m].astype(np.float64)
        J = np.where((_filled(jv) & res)[:, None], jv * 1e-3 * self.lmpc * 1000.0 * mass[:, None], 0.0)
        spin = np.where(res & (N["SpinPeebles"][m] >= 0), N["SpinPeebles"][m], 0)
        T = {
            "ID": self.ids[m], "DescID": self.desc[m], "Mvir": mass, "Vmax": vmax, "Vrms": vrms, "Rvir": rad,
            "Rs": rs, "Np": np.where(res, N["Nbound"][m], 0), "X": pos[:, 0], "Y": pos[:, 1], "Z": pos[:, 2],
            "VX": vel[:, 0], "VY": vel[:, 1], "VZ": vel[:, 2], "JX": J[:, 0], "JY": J[:, 1], "JZ": J[:, 2],
            "Spin": spin,
        }
        neg = lambda col, conv=1.0: np.where(res & _filled(N[col][m]), N[col][m] * conv, -1.0)  # noqa: E731
        kpc = self.lmpc * 1000.0
        T["rs_klypin"] = neg("RsKlypin", kpc)
        mso_all = N["MSO"][m][:, 0].astype(np.float64) * self.mconv
        T["Mvir_all"] = np.where(cen & (mso_all > 0), mso_all, np.where(res, mass, -1))
        for j, name in enumerate(["M200b", "M200c", "M500c", "M2500c"]):
            f = N["MSO"][m][:, j + 1].astype(np.float64) * self.mconv
            b = N["MBoundSO"][m][:, j + 1].astype(np.float64) * self.mconv
            v = np.where(b > 0, b, -1)
            if self.mass == "field":
                v = np.where(cen & (f > 0), f, v)
            T[name] = np.where(res, v, -1)
        T["Xoff"] = neg("Xoff", kpc)
        T["Voff"] = neg("Voff")
        T["spin_bullock"] = neg("SpinBullock")
        T["b_to_a"], T["c_to_a"] = neg("BtoA"), neg("CtoA")
        for nm, col in (("A[%s]", "AxisA"),):
            v = N[col][m].astype(np.float64)
            ok = (_filled(v) & res)[:, None]
            v = np.where(ok, v, -1)
            for d, ax in enumerate("xyz"):
                T[nm % ax] = v[:, d]
        T["T/|U|"] = neg("KinToPot")
        tidm = N["TrackId"][m].astype(np.int64)
        T["TrackId"] = tidm
        T["Mbound"] = np.where(res, mb, 0)
        T["Nbound"] = np.where(res, N["Nbound"][m], 0)
        T["Rank"] = np.where(res, N["Rank"][m], -1)
        T["Depth"] = np.where(res, N["Depth"][m], -1)
        T["HostGroup"] = N["HostGroup"][m].astype(np.int64)
        T["MboundPeak"] = N["MboundPeak"][m].astype(np.float64) * self.mconv
        T["VmaxPeak"] = N["VmaxPeak"][m].astype(np.float64)
        T["SnapAcc"] = N["SnapAcc"][m].astype(np.int64)
        T["Orphan"] = orph.astype(np.int64)
        if self.geometric:
            pid, upid = geometric_parents(pos, rad, vmax, self.box)
            T["ParentTrackId"] = np.where(pid >= 0, tidm[np.maximum(pid, 0)], -1)
            T["UpParentTrackId"] = np.where(upid >= 0, tidm[np.maximum(upid, 0)], -1)
        else:
            T["ParentTrackId"] = N["NestedParentTrackId"][m].astype(np.int64)
            ct = N["CentralTrackId"][m].astype(np.int64)
            T["UpParentTrackId"] = np.where((ct >= 0) & (ct != tidm), ct, -1)
        return T

    # ------------------------------------------------------------------ files
    def header(self, i):
        a = float(self.seq.scale_factors[i])
        lines = [
            "#" + " ".join(REGULAR + EXTRAS),
            "#a = %f" % a,
            "#Om = %f; Ol = %f; h = %f" % (self.om, self.ol, self.h0),
            "#Particle mass: %g Msun/h" % self.part_mass,
            "#Box size: %f Mpc/h" % self.box,
            "#Force resolution assumed: %g Mpc/h" % self.force_res,
            "#Units: Masses in Msun / h",
            "#Units: Positions in Mpc / h (comoving)",
            "#Units: Velocities in km / s (physical, peculiar)",
            "#Units: Halo Distances, Lengths, and Radii in kpc / h (comoving)",
            "#Units: Angular Momenta in (Msun/h) * (Mpc/h) * km/s (physical)",
            "#Units: Spins are dimensionless",
            "#Np is Nbound, the number of bound particles of the track.",
            "#Source: MP-Gadget runtime subhalo finder (HBT-HERONS membership), %s" % os.path.abspath(self.seq.paths[i]),
            "#Exported by tools/subhalo_to_ctrees.py: one halo per resolved track (TrackId column)%s." % (
                "; orphan nodes (Orphan = 1, Mvir = %g) after terminal deaths" % self.orphan_mass if self.orphans else ""),
            "#Mass definition: %s" % self.mass_note(),
            "#Parents: ParentTrackId/UpParentTrackId are %s (TrackIds; -1 for distinct halos)." % (
                "geometric (centre inside the Rvir of a halo with larger Vmax, consistent-trees rule)"
                if self.geometric else "the membership parents NestedParentTrackId/CentralTrackId"),
            "#DescID: identity (the track at the next output) or merger (DescendantTrackId of the dead track), design 7.7.",
        ]
        return "\n".join(lines) + "\n"

    def mass_note(self):
        if self.mass == "field":
            return ("Mvir = MSO[vir] (field spherical overdensity, all particles) for centrals, falling back to "
                    "MBoundSO[vir] then Mbound where unfilled; Mbound for satellites; vir = Bryan & Norman.")
        return ("Mvir = MBoundSO[vir] (bound particles) for centrals, falling back to Mbound where unfilled; "
                "Mbound for satellites; vir = Bryan & Norman.")

    def write(self, outdir):
        for sub in ("", "outputs", "trees", "hlists"):
            d = os.path.join(outdir, sub)
            if not os.path.isdir(d):
                os.makedirs(d)
        scales = []
        self.files = []
        for i in range(len(self.seq)):
            T = self.rows(i)
            snap = self.seq.snapshots[i]
            fn = os.path.join(outdir, "out_%d.list" % snap)
            cols = REGULAR + EXTRAS
            with open(fn, "w") as f:
                f.write(self.header(i))
                n = len(T["ID"])
                if n:
                    fmt = []
                    for c in cols:
                        if c in INT_COLS:
                            fmt.append("%d")
                        elif c in ("X", "Y", "Z"):
                            fmt.append("%.5f")
                        elif c in ("Vmax", "Vrms", "VX", "VY", "VZ", "Voff", "VmaxPeak"):
                            fmt.append("%.2f")
                        elif c in ("Rvir", "Rs", "rs_klypin", "Xoff"):
                            fmt.append("%.3f")
                        else:
                            fmt.append("%.5g")
                    arr = np.empty((n, len(cols)), dtype=object)
                    for k, c in enumerate(cols):
                        v = T[c]
                        arr[:, k] = v.astype(np.int64) if c in INT_COLS else v.astype(np.float64)
                    line = " ".join(fmt) + "\n"
                    f.writelines(line % tuple(r) for r in arr.tolist())
            self.files.append(fn)
            if len(T["ID"]) and snap >= self.first_snapshot:
                scales.append("%d %f\n" % (snap, self.seq.scale_factors[i]))
        with open(os.path.join(outdir, "outputs", "scales.txt"), "w") as f:
            f.writelines(scales)
        self.write_cfg(outdir, len(scales))
        return os.path.join(outdir, "outputs", "merger_tree.cfg")

    def write_cfg(self, outdir, nscales):
        outdir = os.path.abspath(outdir)
        size = sum(os.path.getsize(f) for f in self.files)
        box_div = max(1, int((size / 2e9) ** (1.0 / 3)))
        ts = int(nscales / 34)
        sub_ts = max(1, 2 * ts)
        ts = max(1, ts)
        mps = max(1, int(ts / 2))
        cfg = """# consistent-trees configuration written by MP-Gadget tools/subhalo_to_ctrees.py
# (same keys and defaults as consistent-trees' gen_merger_cfg.pl).
# Mass definition exported: {massnote}
# Parents in ParentTrackId/UpParentTrackId: {parents}
Om={om} #Omega_Matter
Ol={ol} #Omega_Lambda
h0={h0} #h0

SCALEFILE = "{o}/outputs/scales.txt"
INBASE = "{o}"
OUTBASE = "{o}/outputs"
TREE_OUTBASE = "{o}/trees"
HLIST_OUTBASE = "{o}/hlists"

BOX_DIVISIONS={bd}
BOX_WIDTH={box}

MIN_TIMESTEPS_TRACKED = {ts}
MIN_TIMESTEPS_SUB_TRACKED = {sts}
MIN_TIMESTEPS_SUB_MMP_TRACKED = {sts}
MAX_PHANTOM_FRACTION = 0.25

MAJOR_MERGER=0.3 #The merger ratio which constitutes a major merger
MIN_MMP_MASS_RATIO=0.3 #The minimum mass for a progenitor to be considered an MMP
MIN_MMP_VMAX_RATIO=0.7 #The minimum vmax for a progenitor to be considered an MMP

PADDING_TIMESTEPS=0

LAST_DITCH_SEARCH_LIMIT=1.0
LAST_DITCH_VMAX_RATIO_1=1.4
LAST_DITCH_VMAX_RATIO_2=2.5
MAX_PHANTOM={ts}
MAX_PHANTOM_SMALL={mps}
SMALL_PARTICLE_LIMIT=49
TIDAL_FORCE_LIMIT=0.4
RECURSION_LIMIT=5
METRIC_LIMIT=7
METRIC_BREAK_LIMIT=3.2
MASS_RES_OK={mres}

EXTRA_PARAMS = {nex}
EXTRA_PARAM_LABELS = "{labels}"
EXTRA_PARAM_DESCRIPTIONS = "{descr}"
EXTRA_PARAM_INTERPOLATIONS = "{interp}"
ROCKSTAR_TYPE_PARAM = -1
""".format(massnote=self.mass_note(),
           parents="geometric" if self.geometric else "membership (NestedParentTrackId, CentralTrackId)",
           om=self.om, ol=self.ol, h0=self.h0, o=outdir, bd=box_div, box="%g" % self.box, ts=ts, sts=sub_ts,
           mps=mps, mres="%g" % (self.part_mass * 1000), nex=len(EXTRAS), labels=" ".join(EXTRAS),
           descr="".join("#%s: %s\\n" % (e, "Rockstar extra field" if e in ROCKSTAR_EXTRAS else
                                         "MP-Gadget subhalo finder field") for e in EXTRAS),
           interp=INTERP)
        with open(os.path.join(outdir, "outputs", "merger_tree.cfg"), "w") as f:
            f.write(cfg)


def geometric_parents(pos, rvir, vmax, box):
    """consistent-trees find_parents (find_parents_and_cleanup.c:742-772) on one output: rows in
    Vmax-descending order; every row h2 within the Rvir of h1 (periodic, at most box/2.01) with
    Vmax(h2) < Vmax(h1) gets pid = h1 (the last such h1) and, if unset, upid = h1. Returns row
    indices (-1 for none). pos in Mpc/h, rvir in kpc/h."""
    n = pos.shape[0]
    pid = np.full(n, -1, np.int64)
    upid = np.full(n, -1, np.int64)
    if n == 0:
        return pid, upid
    order = np.argsort(-vmax, kind="stable")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.mod(pos, box), boxsize=box)
        near = lambda i, r: tree.query_ball_point(np.mod(pos[i], box), r)  # noqa: E731
    except ImportError:  # pragma: no cover
        def near(i, r):
            d = pos - pos[i]
            d -= box * np.round(d / box)
            return np.nonzero((d ** 2).sum(axis=1) <= r * r)[0]
    maxd = box / 2.01
    for i in order.tolist():
        r = min(rvir[i] / 1e3, maxd)
        for j in near(i, r):
            if vmax[j] < vmax[i]:
                pid[j] = i
                if upid[j] < 0:
                    upid[j] = i
    return pid, upid


# ---------------------------------------------------------------------------------- validation
def read_list(fn):
    """An out_N.list back as (scale, dict column -> array)."""
    names, scale = None, None
    rows = []
    with open(fn) as f:
        for line in f:
            if line.startswith("#"):
                if names is None and line.startswith("#ID"):
                    names = line[1:].split()
                elif line.startswith("#a = "):
                    scale = float(line.split("=")[1])
                continue
            if line.strip():
                rows.append(line.split())
    arr = np.array(rows, dtype=np.float64).reshape(len(rows), len(names))
    return scale, {n: arr[:, k] for k, n in enumerate(names)}


def validate(exp, outdir):
    """Structural rules of the written files (see the module docstring). Returns failures."""
    fails = []
    seq, N = exp.seq, exp.N
    tabs = []
    for i, snap in enumerate(seq.snapshots):
        a, T = read_list(os.path.join(outdir, "out_%d.list" % snap))
        tabs.append(T)
        if abs(a - seq.scale_factors[i]) > 1e-6:
            fails.append("out_%d.list: a = %g, the PIG has %g" % (snap, a, seq.scale_factors[i]))
        n = len(T["ID"])
        if n and not np.array_equal(np.sort(T["ID"]), np.arange(n)):
            fails.append("out_%d.list: IDs are not dense and unique" % snap)
        real = T["Orphan"] == 0
        if np.any(T["Mvir"][real] <= 0) or np.any(T["Vmax"][real] <= 0):
            fails.append("out_%d.list: %d non-orphan rows with Mvir <= 0 or Vmax <= 0" % (
                snap, np.count_nonzero((T["Mvir"][real] <= 0) | (T["Vmax"][real] <= 0))))
        cat = seq[i]
        want = np.sort(cat.subhalos["TrackId"]) if cat.nsubhalos else np.zeros(0)
        if not np.array_equal(np.sort(T["TrackId"][real]), want):
            fails.append("out_%d.list: the resolved TrackIds differ from Subhalo/TrackId" % snap)
        tset = set(T["TrackId"].astype(np.int64).tolist())
        for col in ("ParentTrackId", "UpParentTrackId"):
            p = T[col].astype(np.int64)
            missing = [int(x) for x in p[p >= 0] if int(x) not in tset]
            if missing:
                fails.append("out_%d.list: %d %s not exported at this output (e.g. %s)" % (snap, len(missing), col, missing[:5]))
        if np.any((T["UpParentTrackId"] >= 0) & (T["UpParentTrackId"] == T["TrackId"])):
            fails.append("out_%d.list: a halo is its own UpParentTrackId" % snap)
    nonempty = [i for i in range(len(tabs)) if len(tabs[i]["ID"])]
    for k, i in enumerate(nonempty):
        T = tabs[i]
        last = k == len(nonempty) - 1
        if last:
            if np.any(T["DescID"] >= 0):
                fails.append("out_%d.list (last): DescID set" % seq.snapshots[i])
            continue
        j = nonempty[k + 1]
        U = tabs[j]
        if j != i + 1:
            fails.append("outputs %d and %d are not consecutive in the sequence" % (seq.snapshots[i], seq.snapshots[j]))
        d = T["DescID"].astype(np.int64)
        bad = (d >= len(U["ID"])) | (d < -1)
        if np.any(bad):
            fails.append("out_%d.list: %d DescIDs outside the next output's IDs" % (seq.snapshots[i], np.count_nonzero(bad)))
            continue
        uid = {int(x): kk for kk, x in enumerate(U["ID"].astype(np.int64))}
        ut = U["TrackId"].astype(np.int64)
        uset = set(ut.tolist())
        nxt_cat = seq[j]
        nres = set(nxt_cat.subhalos["TrackId"].tolist()) if nxt_cat.nsubhalos else set()
        orow = {int(t): r for r, t in enumerate(nxt_cat.orphans["TrackId"].tolist())} if nxt_cat.norphans else {}
        nid, nmer, nterm = 0, 0, 0
        for r in range(len(T["ID"])):
            t = int(T["TrackId"][r])
            dd = int(d[r])
            if t in nres or (exp.orphans and dd >= 0 and int(ut[uid[dd]]) == t):
                exp_t = t
                nid += 1
            elif t in orow:
                exp_t = int(nxt_cat.orphans["DescendantTrackId"][orow[t]])
                if T["Orphan"][r] == 1:
                    exp_t = t if exp.orphans and exp_t < 0 else exp_t
                nmer += exp_t >= 0
                nterm += exp_t < 0
            else:
                fails.append("track %d at out_%d.list is absent from PIG %d" % (t, seq.snapshots[i], seq.snapshots[j]))
                continue
            got = int(ut[uid[dd]]) if dd >= 0 else -1
            if exp_t >= 0 and exp_t not in uset:
                # the merger descendant is not exported at the next output (an orphan there, which the
                # tree check reports): the export leaves the node terminal
                if got != -1:
                    fails.append("track %d: merger descendant %d is not exported, DescID should be -1" % (t, exp_t))
                continue
            if got != exp_t and not (exp_t < 0 and got == -1):
                fails.append("track %d at out_%d.list: DescID points to track %d, expected %d" % (
                    t, seq.snapshots[i], got, exp_t))
        print("  out_%d.list -> out_%d.list: %d identity (incl. orphan nodes), %d merger, %d terminal" % (
            seq.snapshots[i], seq.snapshots[j], nid, nmer, nterm))
    return fails


# ---------------------------------------------------------------------------------- hlist comparison
def read_hlist(fn):
    names, scale, rows = None, None, []
    with open(fn) as f:
        for line in f:
            if line.startswith("#"):
                if names is None:
                    names = [n.split("(")[0] if not n.startswith("A[") and not n.startswith("b_to_a(") and not
                             n.startswith("c_to_a(") else n.rsplit("(", 1)[0] for n in line[1:].split()]
                if line.startswith("#a = "):
                    scale = float(line.split("=")[1])
                continue
            if line.strip():
                rows.append(line.split())
    if scale is None:
        base = os.path.basename(fn)
        scale = float(base[len("hlist_"):-len(".list")])
    arr = np.array(rows, dtype=np.float64).reshape(len(rows), len(names))
    out = {n: arr[:, k] for k, n in enumerate(names)}
    for n in list(out):  # consistent-trees writes Mvir, vmax, x, ...: lower-case aliases too
        out.setdefault(n.lower(), out[n])
    return scale, out


def _q(x):
    x = x[np.isfinite(x)]
    return "n/a" if x.size == 0 else "%.3f/%.3f/%.3f" % tuple(np.percentile(x, [10, 50, 90]))


def compare_hlist(args):
    a, H = read_hlist(args.compare_hlist)
    seq = SubhaloSequence(args.inputs[0] if len(args.inputs) == 1 else args.inputs)
    i = int(np.argmin(np.abs(seq.scale_factors - a)))
    cat = seq[i]
    h = seq[len(seq) - 1].header
    mconv = _attr(h, "UnitMass_in_g", 1.989e43) / G_PER_MSUN
    print("hlist %s (a = %g): %d halos; our PIG %s (a = %g): %d subhalos" % (
        args.compare_hlist, a, len(H["id"]), seq.paths[i], seq.scale_factors[i], cat.nsubhalos))
    if "TrackId" not in H:
        print("error: the hlist has no TrackId column (not made from subhalo_to_ctrees.py output)")
        return 2
    S = cat.subhalos
    t = H["TrackId"].astype(np.int64)
    row = np.searchsorted(S["TrackId"], t)
    row = np.minimum(row, max(S["TrackId"].size - 1, 0))
    ok = (S["TrackId"].size > 0) & (S["TrackId"][row] == t) & (H["Orphan"] == 0)
    nphantom = int(np.count_nonzero(H["phantom"] > 0)) if "phantom" in H else 0
    print("  hlist rows with a resolved TrackId: %d (%d phantom rows); tracks absent from the hlist: %d" % (
        np.count_nonzero(ok), nphantom, cat.nsubhalos - np.unique(t[ok]).size))
    r = row[ok]
    # identity kept by consistent-trees: the mmp progenitor in the previous hlist has the same TrackId
    hdir = os.path.dirname(os.path.abspath(args.compare_hlist))
    prev = sorted((float(f[6:-5]), f) for f in os.listdir(hdir) if f.startswith("hlist_") and f.endswith(".list"))
    prev = [f for s, f in prev if s < a - 1e-6]
    if prev:
        _, P = read_hlist(os.path.join(hdir, prev[-1]))
        mm = P["mmp?"] > 0
        pid_of = dict(zip(P["desc_id"][mm].astype(np.int64).tolist(), P["TrackId"][mm].astype(np.int64).tolist()))
        hid = H["id"][ok].astype(np.int64)
        progt = np.array([pid_of.get(int(x), -2) for x in hid])
        kept = progt == t[ok]
        noprog = progt == -2
        print("  main progenitor in %s: same TrackId %d, another track %d, none %d (of %d)" % (
            prev[-1], np.count_nonzero(kept), np.count_nonzero(~kept & ~noprog), np.count_nonzero(noprog), kept.size))
    sacc = seq.scale_of(S["SnapAcc"][r])
    for lab, ct, ours in (("Mpeak / MboundPeak", H["Mpeak"][ok], S["MboundPeak"][r] * mconv),
                          ("Vpeak / VmaxPeak", H["Vpeak"][ok], S["VmaxPeak"][r]),
                          ("Mvir / exported mass (Mbound here)", H["mvir"][ok], S["Mbound"][r] * mconv)):
        rr = ct / np.where(ours > 0, ours, np.nan)
        print("  %-36s p10/p50/p90 %s, exact (1e-3) %.1f%%" % (lab, _q(rr), 100 * np.mean(np.abs(rr - 1) < 1e-3)))
    same = np.isclose(H["Acc_Scale"][ok], sacc, atol=2e-3)
    print("  Acc_Scale == scale(SnapAcc): %.1f%% (%d of %d)" % (100 * np.mean(same), np.count_nonzero(same), same.size))
    upid_ct = H["upid"][ok] >= 0
    sat = S["Rank"][r] > 0
    print("  satellites: ours (Rank > 0) %d, consistent-trees upid >= 0 %d, both %d" % (
        np.count_nonzero(sat), np.count_nonzero(upid_ct), np.count_nonzero(sat & upid_ct)))
    for lab, sel in (("centrals both", ~sat & ~upid_ct), ("satellite either", sat | upid_ct)):
        print("    %-17s n=%4d  Acc_Scale agree %.1f%%  Mpeak ratio %s" % (
            lab, np.count_nonzero(sel), 100 * np.mean(same[sel]) if np.any(sel) else 0,
            _q(H["Mpeak"][ok][sel] / (S["MboundPeak"][r][sel] * mconv))))
    if args.rockstar_hlist:
        a2, R = read_hlist(args.rockstar_hlist)
        print("Rockstar hlist %s (a = %g): %d halos" % (args.rockstar_hlist, a2, len(R["id"])))
        box = _attr(h, "BoxSize", np.nan) * _attr(h, "UnitLength_in_cm", 3.085678e21) / CM_PER_MPC
        P = np.stack([H["x"], H["y"], H["z"]], 1)
        Q = np.stack([R["x"], R["y"], R["z"]], 1)
        from scipy.spatial import cKDTree
        tq = cKDTree(np.mod(Q, box), boxsize=box)
        tp = cKDTree(np.mod(P, box), boxsize=box)
        dpq, ipq = tq.query(np.mod(P, box))
        dqp, iqp = tp.query(np.mod(Q, box))
        mutual = (iqp[ipq] == np.arange(len(P))) & (dpq * 1e3 < R["rvir"][ipq])
        mp = np.nonzero(mutual & (H["Orphan"] == 0))[0]
        mq = ipq[mp]
        part_mass = float(np.atleast_1d(h.get("MassTable", np.ones(6)))[1]) * mconv
        print("  mutual nearest pairs within the Rockstar Rvir: %d of %d (ours) / %d (Rockstar)" % (mp.size, len(P), len(Q)))
        for lab, sel in (("all", np.ones(mp.size, bool)),
                         ("distinct both", (H["upid"][mp] < 0) & (R["upid"][mq] < 0)),
                         ("Mvir(Rockstar) >= 100 mp", R["mvir"][mq] >= 100 * part_mass)):
            print("    %-14s n=%4d  Mvir %s  Vmax %s  Mpeak %s  Vpeak %s  |dAcc_Scale| < 0.01: %.1f%%" % (
                lab, np.count_nonzero(sel), _q(H["mvir"][mp][sel] / R["mvir"][mq][sel]),
                _q(H["vmax"][mp][sel] / R["vmax"][mq][sel]), _q(H["Mpeak"][mp][sel] / R["Mpeak"][mq][sel]),
                _q(H["Vpeak"][mp][sel] / R["Vpeak"][mq][sel]),
                100 * np.mean(np.abs(H["Acc_Scale"][mp][sel] - R["Acc_Scale"][mq][sel]) < 0.01) if np.any(sel) else 0))
        print("    (ratios ours / Rockstar, p10/p50/p90)")
    return 0


# ---------------------------------------------------------------------------------- main
def run_ctrees(ctdir, cfg, log):
    rc = 0
    with open(log, "w") as f:
        for cmd in (["perl", "do_merger_tree.pl", cfg], ["perl", "halo_trees_to_catalog.pl", cfg]):
            f.write("$ %s\n" % " ".join(cmd))
            f.flush()
            rc = subprocess.call(cmd, cwd=ctdir, stdout=f, stderr=subprocess.STDOUT)
            f.write("exit %d\n" % rc)
            if rc:
                break
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="output directory holding PIG_* (or the PIG directories)")
    p.add_argument("-o", "--outdir", help="consistent-trees input directory to write")
    p.add_argument("--mass", choices=["auto", "bound", "field"], default="auto")
    p.add_argument("--orphans", action="store_true", help="export orphan nodes after terminal deaths")
    p.add_argument("--orphan-mass", type=float, default=0.0, help="Mvir of orphan nodes, Msun/h (default 0)")
    p.add_argument("--geometric-parents", action="store_true")
    p.add_argument("--first-snapshot", type=int, default=-1,
                   help="leave outputs before this SnapshotIndex out of scales.txt (their out_N.list is still "
                        "written): consistent-trees aborts on an output with too few halos for its metric")
    p.add_argument("--force-res", type=float, default=None, help="force resolution for the header, Mpc/h")
    p.add_argument("--validate", action="store_true", help="parse the files back and check the structural rules")
    p.add_argument("--run", metavar="CTREES_DIR", default=None,
                   help="run do_merger_tree.pl and halo_trees_to_catalog.pl from this consistent-trees directory")
    p.add_argument("--compare-hlist", metavar="HLIST", default=None)
    p.add_argument("--rockstar-hlist", metavar="HLIST", default=None)
    args = p.parse_args(argv)
    try:
        if args.compare_hlist:
            return compare_hlist(args)
        if not args.outdir:
            p.error("-o is required")
        seq = SubhaloSequence(args.inputs[0] if len(args.inputs) == 1 and not os.path.basename(
            args.inputs[0].rstrip("/")).startswith("PIG_") else args.inputs)
        exp = Export(seq, mass=args.mass, orphans=args.orphans, orphan_mass=args.orphan_mass,
                     geometric=args.geometric_parents, force_res=args.force_res,
                     first_snapshot=args.first_snapshot)
        cfg = exp.write(args.outdir)
    except SubhaloFormatError as e:
        print("error: %s" % e)
        return 2
    print("wrote %d catalogues (snapshots %s) and %s" % (len(exp.files), seq.snapshots, cfg))
    print("  mass: %s" % exp.mass_note())
    print("  %s" % ", ".join("%s %d" % kv for kv in sorted(exp.counts.items())))
    rc = 0
    if args.validate:
        fails = validate(exp, args.outdir)
        for f in fails[:50]:
            print("  - %s" % f)
        print("%s validate: %d failures" % ("FAILED" if fails else "OK", len(fails)))
        rc |= 1 if fails else 0
    if args.run:
        log = os.path.join(os.path.abspath(args.outdir), "outputs", "ctrees_run.log")
        r = run_ctrees(args.run, os.path.abspath(cfg), log)
        print("consistent-trees: exit %d (log %s)" % (r, log))
        rc |= 1 if r else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
