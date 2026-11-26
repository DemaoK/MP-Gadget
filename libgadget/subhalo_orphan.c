/* The tracks no key processed this call (dead, or hostless with own-tag field particles) and the
 * links of the orphan records to their tracer's group and its central after the store. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/openmpsort.h"
#include "utils/system.h"

#include "subhalo.h"
#include "subhalo_internal.h"

/* ------------------------------------------------------------------ hostless and dead tracks */

/* The common periodic reference of a hostless track: its global minimum-ID live field particle. */
struct FieldRef {
    int64_t TrackId;
    MyIDType ID;
    double Pos[3];
};

/* Partial sums of a track's own-tag field particles on one rank, positions as nearest-image
 * displacements from the track's common reference. */
struct FieldSum {
    int64_t TrackId;
    int64_t count;
    int64_t ntracer;
    int64_t LenType[6];
    double MassType[6];
    double mass;
    double ref[3];      /* the common reference (identical in every contribution) */
    double mdx[3];      /* sum m * NEAREST(x - ref) */
    double mv[3];       /* sum m * Vel */
    /* the most-bound particle: the lowest (not a tracer type, BindRank, ID) */
    int64_t mbkey[3];
    MyIDType mbID;
    double mbPos[3], mbVel[3];
};

static int
fieldsum_mb_less(const int64_t a[3], const int64_t b[3])
{
    if(a[0] != b[0]) return a[0] < b[0];
    if(a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

static int
cmp_fieldsum(const void * a, const void * b)
{
    const int64_t x = ((const struct FieldSum *) a)->TrackId, y = ((const struct FieldSum *) b)->TrackId;
    return (x > y) - (x < y);
}

static int
cmp_fieldref(const void * a, const void * b)
{
    const struct FieldRef * x = (const struct FieldRef *) a, * y = (const struct FieldRef *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

struct TagIndex {
    int64_t TrackId;
    int64_t i;
};

static int
cmp_tagindex(const void * a, const void * b)
{
    const struct TagIndex * x = (const struct TagIndex *) a, * y = (const struct TagIndex *) b;
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    /* ID order makes the local sums independent of the particle order */
    const MyIDType ix = P[x->i].ID, iy = P[y->i].ID;
    return (ix > iy) - (ix < iy);
}

/* Stable merge sort by TrackId: contributions of one track keep the source-rank order of the
 * Alltoallv receive buffer, so the home rank sums them in rank order. */
static void
fieldsum_stable_sort(struct FieldSum * in, int64_t nin)
{
    struct FieldSum * tmp = (struct FieldSum *) mymalloc("SubFieldTmp", sizeof(struct FieldSum) * (nin + 1));
    int64_t width;
    for(width = 1; width < nin; width *= 2) {
        int64_t lo;
        for(lo = 0; lo < nin; lo += 2 * width) {
            int64_t mid = lo + width < nin ? lo + width : nin;
            int64_t hi = lo + 2 * width < nin ? lo + 2 * width : nin;
            int64_t a = lo, b = mid, o = lo;
            while(a < mid && b < hi) {
                if(cmp_fieldsum(&in[b], &in[a]) < 0) tmp[o++] = in[b++];
                else tmp[o++] = in[a++];
            }
            while(a < mid) tmp[o++] = in[a++];
            while(b < hi) tmp[o++] = in[b++];
        }
        memcpy(in, tmp, sizeof(struct FieldSum) * nin);
    }
    myfree(tmp);
}

/* Tracks no key claimed: dead if resolved before; a hostless resolved track (unreachable from
 * subhalo_find, where oversized pseudo-keys go in place) keeps its field particles, centred on
 * per-rank ID-ordered sums from a common reference. Returns the global counts. Collective. */
void
subhalo_update_unhosted(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nhostless, int64_t * ndead, MPI_Comm Comm)
{
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i, j;
    int k;
    const double BoxSize = PartManager->BoxSize;
    const int TracerTypes = SubParams.SubhaloTracerTypes;
    const double vfac = subhalo_vel_to_kms(atime);

    /* own-tag live field particles, grouped by track */
#define UNH_PRED (subhalo_part_live(&P[i]) && P[i].GrNr < 0 && P[i].TrackId >= 0)
    const int nc = subhalo_par_nchunks(PartManager->NumPart);
    int64_t * off = ta_malloc("SubFieldOff", int64_t, nc + 1);
    SUBHALO_COUNT_CHUNKS(PartManager->NumPart, nc, off, UNH_PRED);
    int64_t nf = off[nc];
    struct TagIndex * idx = (struct TagIndex *) mymalloc("SubFieldIdx", sizeof(struct TagIndex) * (nf + 1));
    SUBHALO_FILL_CHUNKS(PartManager->NumPart, nc, off, UNH_PRED, {
            idx[m].TrackId = P[i].TrackId;
            idx[m].i = i;
        });
#undef UNH_PRED
    ta_free(off);
    qsort_openmp(idx, nf, sizeof(struct TagIndex), cmp_tagindex);
    /* the first entry of every track (threaded, order-preserving) */
#define TRACK_START(i) ((i) == 0 || idx[i].TrackId != idx[(i)-1].TrackId)
    const int ncb = subhalo_par_nchunks(nf);
    int64_t * offb = ta_malloc("SubFieldOffB", int64_t, ncb + 1);
    SUBHALO_COUNT_CHUNKS(nf, ncb, offb, TRACK_START(i));
    const int64_t ns = offb[ncb];
    int64_t * tstart = (int64_t *) mymalloc("SubFieldStart", sizeof(int64_t) * (ns + 1));
    SUBHALO_FILL_CHUNKS(nf, ncb, offb, TRACK_START(i), tstart[m] = i);
#undef TRACK_START
    tstart[ns] = nf;
    ta_free(offb);

    /* 1. the common reference: local minimum-ID particle of each track to the home rank, the
     *    global minimum back to every holder (request/reply in the same order) */
    struct FieldRef * myref = (struct FieldRef *) mymalloc("SubFieldRef", sizeof(struct FieldRef) * (ns + 1));
    int64_t s;
    #pragma omp parallel for private(k)
    for(s = 0; s < ns; s++) {
        const int64_t i0 = tstart[s];
        myref[s].TrackId = idx[i0].TrackId;
        myref[s].ID = P[idx[i0].i].ID;
        for(k = 0; k < 3; k++)
            myref[s].Pos[k] = P[idx[i0].i].Pos[k];
    }
    int64_t * sendcounts = ta_malloc("SubFieldCounts", int64_t, NTask);
    int64_t * recvcounts = ta_malloc("SubFieldRecvCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    /* sorted by TrackId, hence by home rank */
    for(j = 0; j < ns; j++)
        sendcounts[subhalo_table_home_rank(tab, myref[j].TrackId)]++;
    {
        struct FieldRef * inref = NULL;
        int64_t nin = subhalo_alltoallv(myref, sendcounts, sizeof(struct FieldRef), (void **) &inref, 0, recvcounts, Comm);
        struct FieldRef * minref = (struct FieldRef *) mymalloc("SubFieldMinRef", sizeof(struct FieldRef) * (nin + 1));
        memcpy(minref, inref, sizeof(struct FieldRef) * nin);
        qsort(minref, nin, sizeof(struct FieldRef), cmp_fieldref);
        /* keep the first (minimum-ID) entry of each track */
        int64_t nmin = 0;
        for(i = 0; i < nin; i++)
            if(nmin == 0 || minref[i].TrackId != minref[nmin-1].TrackId)
                minref[nmin++] = minref[i];
        for(i = 0; i < nin; i++) {
            struct FieldRef key = {inref[i].TrackId, 0, {0, 0, 0}};
            /* binary search on TrackId alone */
            int64_t lo = 0, hi = nmin;
            while(lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if(minref[mid].TrackId < key.TrackId) lo = mid + 1;
                else hi = mid;
            }
            if(lo >= nmin || minref[lo].TrackId != key.TrackId)
                endrun(5, "Hostless reference of track %ld lost on its home rank.\n", key.TrackId);
            inref[i] = minref[lo];
        }
        myfree(minref);
        struct FieldRef * back = NULL;
        int64_t nback = subhalo_alltoallv(inref, recvcounts, sizeof(struct FieldRef), (void **) &back, 0, NULL, Comm);
        if(nback != ns)
            endrun(5, "Hostless reference exchange returned %ld entries for %ld tracks.\n", nback, ns);
        for(j = 0; j < ns; j++) {
            if(back[j].TrackId != myref[j].TrackId)
                endrun(5, "Hostless reference exchange out of order (%ld for %ld).\n", back[j].TrackId, myref[j].TrackId);
            myref[j] = back[j];
        }
        myfree(back);
        myfree(inref);
    }

    /* 2. local sums relative to the common reference, in ID order */
    struct FieldSum * sums = (struct FieldSum *) mymalloc("SubFieldSums", sizeof(struct FieldSum) * (ns + 1));
    /* per track in ID order, independent of the thread count; the tracks in parallel */
    #pragma omp parallel for private(i, k)
    for(s = 0; s < ns; s++) {
      memset(&sums[s], 0, sizeof(struct FieldSum));
      sums[s].TrackId = idx[tstart[s]].TrackId;
      for(k = 0; k < 3; k++)
          sums[s].ref[k] = myref[s].Pos[k];
      for(i = tstart[s]; i < tstart[s + 1]; i++) {
        const struct particle_data * p = &P[idx[i].i];
        const double m = p->Mass;
        sums[s].count++;
        sums[s].LenType[p->Type]++;
        sums[s].MassType[p->Type] += m;
        sums[s].mass += m;
        if(TracerTypes & (1 << p->Type))
            sums[s].ntracer++;
        for(k = 0; k < 3; k++) {
            sums[s].mdx[k] += m * NEAREST(p->Pos[k] - sums[s].ref[k], BoxSize);
            sums[s].mv[k] += m * p->Vel[k];
        }
        const int64_t mk[3] = {(TracerTypes & (1 << p->Type)) ? 0 : 1, p->BindRank, (int64_t) p->ID};
        if(sums[s].count == 1 || fieldsum_mb_less(mk, sums[s].mbkey)) {
            for(k = 0; k < 3; k++) {
                sums[s].mbkey[k] = mk[k];
                sums[s].mbPos[k] = p->Pos[k];
                sums[s].mbVel[k] = p->Vel[k];
            }
            sums[s].mbID = p->ID;
        }
      }
    }
    struct FieldSum * in = NULL;
    int64_t nin = subhalo_alltoallv(sums, sendcounts, sizeof(struct FieldSum), (void **) &in, 0, NULL, Comm);
    fieldsum_stable_sort(in, nin);

    /* 3. the home ranks: hostless resolved tracks and dead tracks */
    int64_t nh = 0, nd = 0;
    int64_t a = 0;
    int toolarge = 0;
    for(j = 0; j < tab->NLocal; j++) {
        struct SubTrack * t = &tab->Track[j];
        while(a < nin && in[a].TrackId < t->TrackId)
            a++;
        int64_t b = a;
        while(b < nin && in[b].TrackId == t->TrackId)
            b++;
        if(t->Alive || t->SnapDeath >= 0) {
            a = b;
            continue;
        }
        if(b > a) {
            /* hostless resolved track: its field particles, all bound; sums in rank order */
            int64_t count = 0, ntracer = 0, LenType[6] = {0};
            double MassType[6] = {0}, mass = 0, mdx[3] = {0}, mv[3] = {0};
            int64_t c, cmb = a;
            for(c = a; c < b; c++) {
                if(fieldsum_mb_less(in[c].mbkey, in[cmb].mbkey))
                    cmb = c;
                count += in[c].count;
                ntracer += in[c].ntracer;
                for(k = 0; k < 6; k++) {
                    LenType[k] += in[c].LenType[k];
                    MassType[k] += in[c].MassType[k];
                }
                mass += in[c].mass;
                for(k = 0; k < 3; k++) {
                    mdx[k] += in[c].mdx[k];
                    mv[k] += in[c].mv[k];
                }
            }
            /* the i4 count columns: abort rather than wrap */
            if(count > INT32_MAX || ntracer > INT32_MAX)
                toolarge = 1;
            /* identity and history carried, every property of the previous call cleared */
            {
                const struct SubTrack old = *t;
                subhalo_track_carry(t, &old, old.TrackId, SnapNum);
            }
            t->Flags = SUBHALO_FLAG_INPLACE;
            t->HostGroup = -1;
            t->Rank = t->Depth = 0;
            t->NestedParentTrackId = -1;
            t->CentralTrackId = t->TrackId;
            t->SinkTrackId = -1;
            t->DescendantTrackId = t->TrackId;
            t->Nbound = t->NboundExclusive = (int32_t) count;
            t->NboundTracer = (int32_t) ntracer;
            t->Mbound = mass;
            for(k = 0; k < 6; k++) {
                t->NboundType[k] = t->NboundExclusiveType[k] = (int32_t) LenType[k];
                t->NtailType[k] = 0;
                t->MboundType[k] = MassType[k];
            }
            for(k = 0; k < 3; k++) {
                const double cen = in[a].ref[k] + (mass > 0 ? mdx[k] / mass : 0);
                t->CorePos[k] = t->CoMPos[k] = subhalo_wrap_box(cen - PartManager->CurrentParticleOffset[k], BoxSize);
                t->CoreVel[k] = t->CoMVel[k] = (mass > 0 ? mv[k] / mass : 0) * vfac;
            }
            /* MostBound*: the lowest (not a tracer type, BindRank, ID) over the ranks (no energies here) */
            t->MostBoundID = in[cmb].mbID;
            for(k = 0; k < 3; k++) {
                t->MostBoundPos[k] = subhalo_wrap_box(in[cmb].mbPos[k] - PartManager->CurrentParticleOffset[k], BoxSize);
                t->MostBoundVel[k] = in[cmb].mbVel[k] * vfac;
            }
            t->Alive = 1;
            subhalo_history_update(t, SnapNum, 0);
            nh++;
            if(toolarge)
                endrun(1, "Hostless track %ld has %ld particles, more than the i4 count columns hold.\n", t->TrackId, count);
        }
        else if(t->NboundPrev > 0) {
            /* resolved at the previous call, no live particles anywhere now and not sunk: dead */
            subhalo_track_make_orphan(t, SUBHALO_NOTRACK, SnapNum);
            nd++;
        }
        a = b;
    }
    myfree(in);
    myfree(sums);
    ta_free(recvcounts);
    ta_free(sendcounts);
    myfree(myref);
    myfree(tstart);
    myfree(idx);

    MPI_Allreduce(&nh, nhostless, 1, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&nd, ndead, 1, MPI_INT64, MPI_SUM, Comm);
}


/* ------------------------------------------------------------------ orphan links */

/* On the home records after the store: an orphan's HostGroup is its tracer's group this call and
 * CentralTrackId that group's central (in-place winner from ipl, else as the serial key set, -1
 * in the field); links fixed at death are kept. Not collective. */
int64_t
subhalo_sync_orphan_links(struct SubTrackTable * tab, const struct InplaceWinnerEntry * ipl, int64_t nipl)
{
    int64_t i, nchanged = 0;
    for(i = 0; i < tab->NLocal; i++) {
        struct SubTrack * t = &tab->Track[i];
        if(t->Nbound > 0 || t->SnapDeath < 0)
            continue;
        const int64_t host = t->Host >= 0 ? t->Host : -1;
        if(t->HostGroup != host)
            nchanged++;
        t->HostGroup = host;
        if(host < 0) {
            t->CentralTrackId = -1;
            continue;
        }
        struct InplaceWinnerEntry key = {host, -1};
        const struct InplaceWinnerEntry * w = (const struct InplaceWinnerEntry *)
            bsearch(&key, ipl, nipl, sizeof(key), cmp_iplwinner);
        if(w)
            t->CentralTrackId = w->winner;
    }
    return nchanged;
}

/* ipl[k] = (g, final TrackId of the winner) of in-place key k, on every rank. Collective. */
void
gather_inplace_winners(const struct SubhaloPlan * plan, const struct SubhaloResult * res, int64_t nres,
                       struct InplaceWinnerEntry * ipl, MPI_Comm Comm)
{
    int64_t i;
    const int64_t K = plan->NInplace;
    int64_t * w = (int64_t *) mymalloc("SubIplW", sizeof(int64_t) * (K + 1));
    int64_t * gw = (int64_t *) mymalloc("SubIplGW", sizeof(int64_t) * (K + 1));
    for(i = 0; i < K; i++)
        w[i] = -1;
    for(i = 0; i < nres; i++)
        if(res[i].valid && res[i].ipk >= 0)
            w[res[i].ipk] = res[i].rec.TrackId;
    if(K > 0)
        MPI_Allreduce(w, gw, K, MPI_INT64, MPI_MAX, Comm);
    for(i = 0; i < K; i++) {
        ipl[i].g = plan->Inplace[i].g;
        ipl[i].winner = gw[i];
    }
    myfree(gw);
    myfree(w);
}
