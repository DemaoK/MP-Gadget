#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the runtime subhalo finder's catalogue of a PIG_xxx into HBT's SubSnap layout
(halofinder_phase4_plan.md section 1 item 1, halofinder_design.md section 9).

Usage:
  subhalo_to_hbt.py PIG_xxx [PIG_yyy ...] -o OUTDIR [--bigfile] [--no-orphans]

writes OUTDIR/NNN/SubSnap_NNN.0.hdf5 (NNN = the PIG's SnapshotIndex, one file per snapshot,
NumberOfFiles = 1), the layout HBT-BF writes (src/io/subhalo_io.cpp, WriteBoundSubfile) and
tools/subhalo_compare_hbt.py reads:

  NumberOfFiles (i4), SnapshotId (i4), NumberOfNewSubhalos (i8), NumberOfFakeHalos (i8, 0),
  NumberOfSubhalosInAllFiles (i8), Cosmology/{OmegaM0, OmegaLambda0, HubbleParam, ScaleFactor}
  (f4; HubbleParam is HBT's Hz, H(a) in km/s per internal length unit), Units/{LengthInMpch,
  MassInMsunh, VelInKmS} (f4), Header attributes Git_branch/Git_commit plus Converter/Source,
  Subhalos (compound, one row per track: resolved tracks from Subhalo/*, then orphans from
  Orphan/* with Nbound = 0, sorted by TrackId as HBT keeps every track),
  SubhaloParticles (vlen u8: the full bound IDs of each row in binding order, empty for orphans),
  NestedSubhalos (vlen u8: TrackIds of the first-level children by NestedParentTrackId).

Subhalos fields (HBT-BF subhalo.h names and on-disk types of the non-DM_ONLY build, HBTInt = i8,
HBTReal = f4) and their source:
  TrackId <- TrackId               Nbound (i8) <- Nbound      Mbound <- Mbound
  NboundType (i8[6]) <- NboundType MboundType (f4[6]) <- MboundType
  TracerIndex <- position in SubhaloParticles of the first particle of a tracer type (0 orphans)
  HostHaloId <- HostGroup (HBT keeps MP-Gadget's one-based GroupID; -1 hostless)
  Rank (i8) <- Rank, Depth (i4) <- Depth (orphans: after the resolved members of their host by
               TrackId; Depth from NestedParentTrackId; hostless orphans 0/0)
  LastMaxMass <- MboundPeak        SnapshotOfLastMaxMass <- SnapMboundPeak
  SnapshotOfLastIsolation <- SnapLastIsolation
  SnapshotOfBirth/Death/Sink <- SnapBirth/SnapDeath/SnapSink
  RmaxComoving <- Rmax             RmaxComovingOfLastMaxVmax <- -1 (not stored)
  VmaxPhysical <- Vmax             LastMaxVmaxPhysical <- VmaxPeak
  SnapshotOfLastMaxVmax <- SnapVmaxPeak
  REncloseComoving <- Renclose     RHalfComoving <- Rhalf
  BoundR200CritComoving <- RBoundSO[200c]   BoundM200Crit <- MBoundSO[200c]
  SpecificSelfPotentialEnergy <- 2 Epot (HBT's <phi> counts each pair twice; our Epot once)
  SpecificSelfKineticEnergy <- Ekin
  SpecificAngularMomentum <- J (physical, full bound set)
  InertialTensor <- InertiaTensor  InertialTensorWeighted <- -1 (not computed)
  ComovingAveragePosition <- CoMPos         PhysicalAverageVelocity <- CoMVel
  ComovingMostBoundPosition <- MostBoundPos PhysicalMostBoundVelocity <- MostBoundVel
     (orphans: TracerPos / TracerVel for both pairs, as HBT's orphans carry their tracer)
  MostBoundParticleId <- MostBoundID
  SinkTrackId, NestedParentTrackId <- same names
  DescendantTrackId <- DescendantTrackId for orphans; -1 for resolved tracks, as HBT writes them
     (the PIG stores the identity, the track itself, design 7.7)
Orphan rows have no Subhalo/* properties: Mbound, Vmax, Rmax, energies and the like are 0 as in
HBT's orphans; history fields come from the Orphan/* group-E columns. Fields the finder has not
filled yet keep its -1.

SubhaloParticles order (SubhaloCatalogue.full_bound_ids): the exclusive bound list (own tag and
BoundTrackId == the track: the bound prefixes of the in-group and field slices, every type) in
BindRank order, which is the final unbinding pass's energy order; then the overlap particles
(bound to the track, tagged with a nested child: the children's tails) in (their BindRank, ID)
order, since their place in the parent's energy order is not recorded. The ten most-bound tracers
used by the HBT-style matching are always in the exclusive part.

Without h5py (or with --bigfile) the same datasets are written as a BigFile directory
OUTDIR/NNN/SubSnap_NNN/: Subhalos/<field> columns, SubhaloParticles/ID (flat) with
Subhalos/ParticleOffset and Subhalos/ParticleLength, NestedSubhalos/TrackId with
Subhalos/NestedOffset and Subhalos/NestedLength, the scalars and Cosmology/Units as Header
attributes.

Check: python3 tools/subhalo_compare_hbt.py PIG_xxx OUTDIR --match jaccard --min-np 0 must match
every subhalo with itself at Jaccard 1 (subhalo_to_hbt.py --verify runs that comparison).
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subhalo_reader import SubhaloCatalogue, SubhaloFormatError, NTYPES  # noqa: E402

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

# HBT-BF H5T_SubhaloInDisk, non-DM_ONLY build (subhalo_io.cpp:13-98), HBTInt = i8, HBTReal = f4
HBT_FIELDS = [
    ("TrackId", "<i8", ()), ("Nbound", "<i8", ()), ("Mbound", "<f4", ()),
    ("NboundType", "<i8", (NTYPES,)), ("MboundType", "<f4", (NTYPES,)),
    ("TracerIndex", "<i8", ()), ("HostHaloId", "<i8", ()), ("Rank", "<i8", ()), ("Depth", "<i4", ()),
    ("LastMaxMass", "<f4", ()), ("SnapshotOfLastMaxMass", "<i4", ()), ("SnapshotOfLastIsolation", "<i4", ()),
    ("SnapshotOfBirth", "<i4", ()), ("SnapshotOfDeath", "<i4", ()), ("SnapshotOfSink", "<i4", ()),
    ("RmaxComoving", "<f4", ()), ("RmaxComovingOfLastMaxVmax", "<f4", ()), ("VmaxPhysical", "<f4", ()),
    ("LastMaxVmaxPhysical", "<f4", ()), ("SnapshotOfLastMaxVmax", "<i4", ()),
    ("REncloseComoving", "<f4", ()), ("RHalfComoving", "<f4", ()),
    ("BoundR200CritComoving", "<f4", ()), ("BoundM200Crit", "<f4", ()),
    ("SpecificSelfPotentialEnergy", "<f4", ()), ("SpecificSelfKineticEnergy", "<f4", ()),
    ("SpecificAngularMomentum", "<f4", (3,)),
    ("InertialTensor", "<f4", (6,)), ("InertialTensorWeighted", "<f4", (6,)),
    ("ComovingAveragePosition", "<f4", (3,)), ("PhysicalAverageVelocity", "<f4", (3,)),
    ("ComovingMostBoundPosition", "<f4", (3,)), ("PhysicalMostBoundVelocity", "<f4", (3,)),
    ("MostBoundParticleId", "<i8", ()),
    ("SinkTrackId", "<i8", ()), ("DescendantTrackId", "<i8", ()), ("NestedParentTrackId", "<i8", ()),
]
HBT_DTYPE = np.dtype([(n, d, s) if s else (n, d) for n, d, s in HBT_FIELDS])

# HBT field <- our Subhalo/* column (index into a vector column where given)
FROM_SUBHALO = {
    "TrackId": "TrackId", "Mbound": "Mbound", "NboundType": "NboundType", "MboundType": "MboundType",
    "HostHaloId": "HostGroup", "Rank": "Rank", "Depth": "Depth",
    "LastMaxMass": "MboundPeak", "SnapshotOfLastMaxMass": "SnapMboundPeak",
    "SnapshotOfLastIsolation": "SnapLastIsolation", "SnapshotOfBirth": "SnapBirth",
    "SnapshotOfDeath": "SnapDeath", "SnapshotOfSink": "SnapSink",
    "RmaxComoving": "Rmax", "VmaxPhysical": "Vmax", "LastMaxVmaxPhysical": "VmaxPeak",
    "SnapshotOfLastMaxVmax": "SnapVmaxPeak", "REncloseComoving": "Renclose", "RHalfComoving": "Rhalf",
    "BoundR200CritComoving": ("RBoundSO", 2), "BoundM200Crit": ("MBoundSO", 2),
    "SpecificSelfPotentialEnergy": "Epot", "SpecificSelfKineticEnergy": "Ekin",
    "SpecificAngularMomentum": "J", "InertialTensor": "InertiaTensor",
    "ComovingAveragePosition": "CoMPos", "PhysicalAverageVelocity": "CoMVel",
    "ComovingMostBoundPosition": "MostBoundPos", "PhysicalMostBoundVelocity": "MostBoundVel",
    "MostBoundParticleId": "MostBoundID",
    "SinkTrackId": "SinkTrackId", "DescendantTrackId": "DescendantTrackId",
    "NestedParentTrackId": "NestedParentTrackId",
}
FROM_ORPHAN = {
    "TrackId": "TrackId", "HostHaloId": "HostGroup",
    "LastMaxMass": "MboundPeak", "SnapshotOfLastMaxMass": "SnapMboundPeak",
    "SnapshotOfLastIsolation": "SnapLastIsolation", "SnapshotOfBirth": "SnapBirth",
    "SnapshotOfDeath": "SnapDeath", "SnapshotOfSink": "SnapSink",
    "LastMaxVmaxPhysical": "VmaxPeak", "SnapshotOfLastMaxVmax": "SnapVmaxPeak",
    "ComovingAveragePosition": "TracerPos", "PhysicalAverageVelocity": "TracerVel",
    "ComovingMostBoundPosition": "TracerPos", "PhysicalMostBoundVelocity": "TracerVel",
    "MostBoundParticleId": "MostBoundID",
    "SinkTrackId": "SinkTrackId", "DescendantTrackId": "DescendantTrackId",
    "NestedParentTrackId": "NestedParentTrackId",
}
NOT_STORED = ("RmaxComovingOfLastMaxVmax", "InertialTensorWeighted")

CM_PER_MPC = 3.085678e24
G_PER_MSUN = 1.989e33


def _h(cat, name, default=None):
    v = cat.header.get(name, default)
    return np.atleast_1d(v)[0] if v is not None else default


def _fill(table, src, n):
    if isinstance(src, tuple):
        return table[src[0]][:, src[1]] if src[0] in table else None
    return table.get(src)


def orphan_rank_depth(cat):
    """HBT-style Rank and Depth of the orphans (HBT ranks orphans after the resolved members of
    their host since it sorts members by Mbound; ties by TrackId here): Rank = number of resolved
    members of the host + position among the host's orphans; Depth = Depth(parent) + 1 along
    NestedParentTrackId (1 when the parent is unknown); hostless orphans 0/0."""
    S, O = cat.subhalos, cat.orphans
    n = cat.norphans
    rank = np.zeros(n, np.int64)
    depth = np.zeros(n, np.int32)
    if n == 0:
        return rank, depth
    host = O["HostGroup"].astype(np.int64)
    nres = {}
    if cat.nsubhalos:
        h, c = np.unique(S["HostGroup"].astype(np.int64), return_counts=True)
        nres = dict(zip(h.tolist(), c.tolist()))
    o = np.lexsort((O["TrackId"], host))
    seen = {}
    for k in o.tolist():
        hk = int(host[k])
        if hk < 0:
            continue
        rank[k] = nres.get(hk, 0) + seen.get(hk, 0)
        seen[hk] = seen.get(hk, 0) + 1
    known = {}
    if cat.nsubhalos:
        known = dict(zip(S["TrackId"].tolist(), S["Depth"].astype(int).tolist()))
    parent = dict(zip(O["TrackId"].tolist(), O["NestedParentTrackId"].tolist()))

    def dep(t, guard=0):
        if t in known:
            return known[t]
        p = parent.get(t, -1)
        if p < 0 or guard > 64:
            return 0
        d = dep(p, guard + 1) + 1
        known[t] = d
        return d

    for k, t in enumerate(O["TrackId"].tolist()):
        if host[k] < 0:
            depth[k] = 0
            continue
        p = int(O["NestedParentTrackId"][k])
        depth[k] = dep(p) + 1 if p >= 0 else 1
    return rank, depth


def build(cat, orphans=True):
    """(Subhalos record array, list of ID arrays, list of NestedSubhalos arrays, header dict)."""
    S, O = cat.subhalos, cat.orphans
    ns = cat.nsubhalos
    no = cat.norphans if orphans else 0
    sub = np.zeros(ns + no, dtype=HBT_DTYPE)
    for name, dt, shp in HBT_FIELDS:
        sub[name] = -1 if name in NOT_STORED else 0
    # resolved
    for name, src in FROM_SUBHALO.items():
        v = _fill(S, src, ns)
        if v is not None and ns:
            sub[name][:ns] = v
    if ns:
        ep = S["Epot"]
        sub["SpecificSelfPotentialEnergy"][:ns] = np.where(ep != -1, 2 * ep, -1)
        sub["DescendantTrackId"][:ns] = -1
    tids, off, ids, types = cat.all_full_bound_ids()
    if ns and not np.array_equal(tids, S["TrackId"].astype(np.int64)):
        raise SubhaloFormatError("%s: Subhalo/TrackId is not sorted" % cat.pigdir)
    lens = np.diff(off)
    sub["Nbound"][:ns] = lens
    nbad = int(np.count_nonzero(lens != S["Nbound"])) if ns else 0
    tracer = cat.tracer_types()
    lists = [ids[off[k]:off[k + 1]].astype(np.uint64) for k in range(ns)]
    for k in range(ns):
        tt = types[off[k]:off[k + 1]]
        w = np.nonzero(np.isin(tt, tracer))[0]
        sub["TracerIndex"][k] = int(w[0]) if w.size else 0
    # orphans
    if no:
        for name, src in FROM_ORPHAN.items():
            v = _fill(O, src, no)
            if v is not None:
                sub[name][ns:] = v
        r, d = orphan_rank_depth(cat)
        sub["Rank"][ns:] = r
        sub["Depth"][ns:] = d
        sub["TracerIndex"][ns:] = 0
        lists += [np.zeros(0, np.uint64)] * no
    order = np.argsort(sub["TrackId"], kind="stable")
    sub = sub[order]
    lists = [lists[k] for k in order.tolist()]
    # first-level children
    par = sub["NestedParentTrackId"]
    nested = [np.zeros(0, np.uint64) for _ in range(sub.size)]
    if sub.size:
        m = par >= 0
        prow = np.searchsorted(sub["TrackId"], par[m])
        prow = np.minimum(prow, sub.size - 1)
        ok = sub["TrackId"][prow] == par[m]
        kids = {}
        for p, c in zip(prow[ok].tolist(), sub["TrackId"][m][ok].tolist()):
            kids.setdefault(p, []).append(c)
        for p, c in kids.items():
            nested[p] = np.array(sorted(c), dtype=np.uint64)
    a = cat.scale_factor
    om = float(_h(cat, "Omega0", 0.3))
    ol = float(_h(cat, "OmegaLambda", 1 - om))
    ulen = float(_h(cat, "UnitLength_in_cm", 3.085678e21))
    umass = float(_h(cat, "UnitMass_in_g", 1.989e43))
    uvel = float(_h(cat, "UnitVelocity_in_cm_per_s", 1e5))
    lmpch = ulen / CM_PER_MPC
    # HBT's Hz: H(a) in velocity units per length unit (0.1 at a = 1 for kpc/h and km/s)
    hz = 100.0 * lmpch / (uvel / 1e5) * np.sqrt(om / a ** 3 + (1 - om - ol) / a ** 2 + ol)
    snap = cat.snapshot
    newborn = int(np.count_nonzero(S["SnapBirth"] == snap)) if ns else 0
    head = dict(NumberOfFiles=1, SnapshotId=snap, NumberOfNewSubhalos=newborn, NumberOfFakeHalos=0,
                NumberOfSubhalosInAllFiles=sub.size, OmegaM0=om, OmegaLambda0=ol, HubbleParam=hz,
                ScaleFactor=a, LengthInMpch=lmpch, MassInMsunh=umass / G_PER_MSUN, VelInKmS=uvel / 1e5,
                nbound_mismatch=nbad)
    return sub, lists, nested, head


def _git():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        b = subprocess.check_output(["git", "-C", here, "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return b, c
    except Exception:
        return "not available", "not available"


def write_hdf5(fn, sub, lists, nested, head, source):
    vl = h5py.vlen_dtype(np.dtype("uint64"))
    with h5py.File(fn, "w") as f:
        f.create_dataset("NumberOfFiles", data=np.array([head["NumberOfFiles"]], np.int32))
        f.create_dataset("SnapshotId", data=np.array([head["SnapshotId"]], np.int32))
        f.create_dataset("NumberOfNewSubhalos", data=np.array([head["NumberOfNewSubhalos"]], np.int64))
        f.create_dataset("NumberOfFakeHalos", data=np.array([head["NumberOfFakeHalos"]], np.int64))
        f.create_dataset("NumberOfSubhalosInAllFiles", data=np.array([head["NumberOfSubhalosInAllFiles"]], np.int64))
        g = f.create_group("Cosmology")
        for k in ("OmegaM0", "OmegaLambda0", "HubbleParam", "ScaleFactor"):
            g.create_dataset(k, data=np.array([head[k]], np.float32))
        g = f.create_group("Units")
        for k in ("LengthInMpch", "MassInMsunh", "VelInKmS"):
            g.create_dataset(k, data=np.array([head[k]], np.float32))
        g = f.create_group("Header")
        br, cm = _git()
        g.attrs["Git_branch"] = np.bytes_(br)
        g.attrs["Git_commit"] = np.bytes_(cm)
        g.attrs["Converter"] = np.bytes_("MP-Gadget tools/subhalo_to_hbt.py")
        g.attrs["Source"] = np.bytes_(os.path.abspath(source))
        g.attrs["ParticleOrder"] = np.bytes_(
            "exclusive bound list in BindRank order, then overlap particles (children's tails bound to the "
            "track) in (BindRank, ID) order")
        d = f.create_dataset("NestedSubhalos", (len(nested),), dtype=vl)
        for i, v in enumerate(nested):
            d[i] = v
        d.attrs["Comment"] = np.bytes_("List of the TrackIds of first-level sub-subhaloes within each subhalo.")
        d = f.create_dataset("SubhaloParticles", (len(lists),), dtype=vl)
        for i, v in enumerate(lists):
            d[i] = v
        f.create_dataset("Subhalos", data=sub)


def write_bigfile(dirname, sub, lists, nested, head, source):
    import bigfile
    with bigfile.File(dirname, create=True) as f:
        h = f.create("Header")
        for k, v in head.items():
            if k != "nbound_mismatch":
                h.attrs[k] = v
        h.attrs["Source"] = os.path.abspath(source)
        for name in sub.dtype.names:
            col = np.ascontiguousarray(sub[name])
            dt = np.dtype((col.dtype.base, col.shape[1:])) if col.ndim > 1 else col.dtype
            b = f.create("Subhalos/%s" % name, dtype=dt, size=col.shape[0], Nfile=1)
            b.write(0, col)
        for label, arrs, idname in (("Particle", lists, "SubhaloParticles/ID"), ("Nested", nested, "NestedSubhalos/TrackId")):
            lens = np.array([len(x) for x in arrs], np.int64)
            offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64) if lens.size else lens
            flat = np.concatenate(arrs).astype(np.uint64) if lens.sum() else np.zeros(0, np.uint64)
            for nm, arr in (("Subhalos/%sOffset" % label, offs), ("Subhalos/%sLength" % label, lens), (idname, flat)):
                b = f.create(nm, dtype=arr.dtype, size=arr.shape[0], Nfile=1)
                b.write(0, arr)


def convert(pigdir, outdir, use_bigfile=False, orphans=True, quiet=False):
    cat = SubhaloCatalogue(pigdir)
    sub, lists, nested, head = build(cat, orphans=orphans)
    snap = head["SnapshotId"]
    d = os.path.join(outdir, "%03d" % snap)
    if not os.path.isdir(d):
        os.makedirs(d)
    if use_bigfile:
        fn = os.path.join(d, "SubSnap_%03d" % snap)
        write_bigfile(fn, sub, lists, nested, head, pigdir)
    else:
        fn = os.path.join(d, "SubSnap_%03d.0.hdf5" % snap)
        write_hdf5(fn, sub, lists, nested, head, pigdir)
    if not quiet:
        nres = int(np.count_nonzero(sub["Nbound"] > 0))
        print("%s -> %s: %d subhalos (%d resolved, %d orphans), %d bound IDs" % (
            pigdir, fn, sub.size, nres, sub.size - nres, sum(len(x) for x in lists)))
        if head["nbound_mismatch"]:
            print("  WARNING: %d subhalos whose BoundTrackId count differs from Nbound; Nbound written is the "
                  "length of SubhaloParticles (run subhalo_check.py tags)" % head["nbound_mismatch"])
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pigdirs", nargs="+", help="PIG_xxx directories")
    p.add_argument("-o", "--outdir", required=True, help="HBT-style SubhaloPath to write (NNN/SubSnap_NNN.0.hdf5)")
    p.add_argument("--bigfile", action="store_true", help="write BigFile instead of HDF5 (automatic without h5py)")
    p.add_argument("--no-orphans", action="store_true", help="omit the Orphan/* rows (HBT keeps them, Nbound = 0)")
    p.add_argument("--verify", action="store_true",
                   help="run subhalo_compare_hbt.py --match jaccard --min-np 0 on each PIG against its conversion "
                        "and require every subhalo matched with itself at Jaccard 1")
    args = p.parse_args(argv)
    use_bf = args.bigfile
    if h5py is None and not use_bf:
        print("note: h5py is not available; writing BigFile directories with the same dataset names")
        use_bf = True
    rc = 0
    for pig in args.pigdirs:
        try:
            convert(pig, args.outdir, use_bigfile=use_bf, orphans=not args.no_orphans)
        except SubhaloFormatError as e:
            print("error: %s" % e)
            return 2
        if args.verify:
            if use_bf:
                print("  --verify needs HDF5 output; skipped")
                continue
            rc |= verify(pig, args.outdir)
    return rc


def verify(pig, outdir):
    """Self-comparison through subhalo_compare_hbt: every resolved track matched with itself, the
    same TrackId, Jaccard 1 and Nbound ratio 1."""
    import subhalo_compare_hbt as C
    ns = argparse.Namespace(ntracers=10, phase1=False, hbt_snap_offset=0, match="jaccard")
    ours, hbt = C.load(pig, outdir, ns)
    ok = True
    for mode in ("jaccard", "tracers"):
        M = C.match(ours, hbt, mode, 10)
        same = np.count_nonzero(ours.tid[M.o] == hbt.trackid[M.h])
        perfect = np.count_nonzero((M.jac == 1.0) & (ours.tid[M.o] == hbt.trackid[M.h]))
        nalive = int(np.count_nonzero(hbt.alive))
        good = perfect == ours.tid.size == nalive
        ok &= good
        print("  verify (%s): %d of %d subhalos matched, %d with their own TrackId, %d at Jaccard 1 -> %s" % (
            mode, M.h.size, ours.tid.size, same, perfect, "OK" if good else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
