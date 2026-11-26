/* Step 10 of the driver (design 7.2): the newborn keys numbered in host order on their directory
 * ranks, then every provisional tag and record reference rewritten to the final TrackIds. */

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

/* ------------------------------------------------------------------ step 10: new TrackIds */

struct NewbornEntry {
    int64_t g;
    int64_t final;
};

static void
radix_newborn_g(const void * a, void * radix, void * arg)
{
    ((uint64_t *) radix)[0] = ((const struct NewbornEntry *) a)->g;
}

static int
cmp_int64(const void * a, const void * b)
{
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

static int
cmp_newborn_g(const void * a, const void * b)
{
    const int64_t x = ((const struct NewbornEntry *) a)->g, y = ((const struct NewbornEntry *) b)->g;
    return (x > y) - (x < y);
}

static int64_t
newborn_lookup(const struct NewbornEntry * map, int64_t n, int64_t ordinal)
{
    struct NewbornEntry key = {ordinal, -1};
    const struct NewbornEntry * hit = (const struct NewbornEntry *) bsearch(&key, map, n, sizeof(map[0]), cmp_newborn_g);
    return hit ? hit->final : SUBHALO_NOTRACK;
}

static int64_t
rewrite_id(int64_t T, const struct NewbornEntry * map, int64_t n)
{
    if(!SUBHALO_IS_PROVISIONAL(T))
        return T;
    int64_t f = newborn_lookup(map, n, SUBHALO_PROVISIONAL_ORDINAL(T));
    return f >= 0 ? f : T;
}

/* Numbers the newborn keys in host (GrNr) order and rewrites every provisional tag and record
 * reference. Returns the global number of newborns. Collective. */
int64_t
subhalo_number_newborns(struct SubhaloResult * res, int64_t nres, int64_t TotNgroups, int64_t NTotalBefore, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t i;

    int64_t nlocal = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid && res[i].newborn)
            nlocal++;
    int64_t ntot = 0;
    MPI_Allreduce(&nlocal, &ntot, 1, MPI_INT64, MPI_SUM, Comm);
    if(ntot == 0)
        return 0;

    /* 1. global host order: mpsort on g, then an exclusive scan of the sorted counts */
    struct NewbornEntry * ent = (struct NewbornEntry *) mymalloc("SubNewborn", sizeof(struct NewbornEntry) * (nlocal + 1));
    int64_t n = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid && res[i].newborn) {
            ent[n].g = SUBHALO_PROVISIONAL_ORDINAL(res[i].rec.TrackId);
            ent[n].final = -1;
            n++;
        }
    mpsort_mpi(ent, nlocal, sizeof(struct NewbornEntry), radix_newborn_g, 8, NULL, Comm);
    int64_t offset = 0;
    MPI_Exscan(&nlocal, &offset, 1, MPI_INT64, MPI_SUM, Comm);
    if(ThisTask == 0)
        offset = 0;
    for(i = 0; i < nlocal; i++) {
        ent[i].final = NTotalBefore + offset + i;
        if(i > 0 && ent[i].g <= ent[i-1].g)
            endrun(5, "Newborn key %ld appears twice.\n", ent[i].g);
    }

    /* 2. store the (ordinal -> final) pairs on the key directory rank of g */
    const int64_t KChunk = subhalo_key_chunk(TotNgroups, NTask);
    int64_t * sendcounts = ta_malloc("SubNbCounts", int64_t, NTask);
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    for(i = 0; i < nlocal; i++)
        sendcounts[subhalo_key_dir_rank(ent[i].g, KChunk)]++;
    /* ent is sorted by g, hence by directory rank */
    struct NewbornEntry * dir = NULL;
    int64_t ndir = subhalo_alltoallv(ent, sendcounts, sizeof(struct NewbornEntry), (void **) &dir, 0, NULL, Comm);
    qsort(dir, ndir, sizeof(struct NewbornEntry), cmp_newborn_g);

    /* 3. every rank asks for the ordinals it holds: particle tags and its records */
    /* the particles' provisional tags: threaded, order-preserving (chunk counts, offsets, fills) */
    const int64_t NumPart = PartManager->NumPart;
    const int nc = subhalo_par_nchunks(NumPart);
    int64_t * off = ta_malloc("SubNbOff", int64_t, nc + 1);
    int c;
    off[0] = 0;
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t cnt = 0, k;
        for(k = c * NumPart / nc; k < (c + 1) * NumPart / nc; k++) {
            if(P[k].IsGarbage) continue;
            if(SUBHALO_IS_PROVISIONAL(P[k].TrackId)) cnt++;
            if(SUBHALO_IS_PROVISIONAL(P[k].BoundTrackId)) cnt++;
        }
        off[c + 1] = cnt;
    }
    for(c = 0; c < nc; c++)
        off[c + 1] += off[c];
    int64_t nreq = off[nc];
    int j;
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        if(SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            nreq++;
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                nreq++;
    }
    int64_t * req = (int64_t *) mymalloc("SubNbReq", sizeof(int64_t) * (nreq + 1));
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t m = off[c], k;
        for(k = c * NumPart / nc; k < (c + 1) * NumPart / nc; k++) {
            if(P[k].IsGarbage) continue;
            if(SUBHALO_IS_PROVISIONAL(P[k].TrackId)) req[m++] = SUBHALO_PROVISIONAL_ORDINAL(P[k].TrackId);
            if(SUBHALO_IS_PROVISIONAL(P[k].BoundTrackId)) req[m++] = SUBHALO_PROVISIONAL_ORDINAL(P[k].BoundTrackId);
        }
    }
    nreq = off[nc];
    ta_free(off);
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        if(SUBHALO_IS_PROVISIONAL(res[i].rec.TrackId))
            req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(res[i].rec.TrackId);
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                req[nreq++] = SUBHALO_PROVISIONAL_ORDINAL(res[i].DescOwner[j]);
    }
    qsort_openmp(req, nreq, sizeof(int64_t), cmp_int64);
    int64_t nuniq = 0;
    for(i = 0; i < nreq; i++)
        if(nuniq == 0 || req[i] != req[nuniq-1])
            req[nuniq++] = req[i];
    memset(sendcounts, 0, sizeof(int64_t) * NTask);
    int badord = 0;
    for(i = 0; i < nuniq; i++) {
        if(req[i] < 1 || req[i] > TotNgroups) {
            badord = 1;
            continue;
        }
        sendcounts[subhalo_key_dir_rank(req[i], KChunk)]++;
    }
    if(badord)
        endrun(5, "A provisional TrackId names a key outside [1, %ld].\n", TotNgroups);
    int64_t * recvcounts = ta_malloc("SubNbRecvCounts", int64_t, NTask);
    int64_t * asked = NULL;
    int64_t nasked = subhalo_alltoallv(req, sendcounts, sizeof(int64_t), (void **) &asked, 0, recvcounts, Comm);
    struct NewbornEntry * answer = (struct NewbornEntry *) mymalloc("SubNbAnswer", sizeof(struct NewbornEntry) * (nasked + 1));
    for(i = 0; i < nasked; i++) {
        answer[i].g = asked[i];
        answer[i].final = newborn_lookup(dir, ndir, asked[i]);
    }
    struct NewbornEntry * map = NULL;
    int64_t nmap = subhalo_alltoallv(answer, recvcounts, sizeof(struct NewbornEntry), (void **) &map, 0, NULL, Comm);
    if(nmap != nuniq)
        endrun(5, "Newborn lookup returned %ld answers for %ld requests.\n", nmap, nuniq);
    /* answers come back in request order, which is sorted by g */

    /* 4. rewrite */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        P[i].TrackId = rewrite_id(P[i].TrackId, map, nmap);
        P[i].BoundTrackId = rewrite_id(P[i].BoundTrackId, map, nmap);
    }
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        struct SubTrack * t = &res[i].rec;
        t->TrackId = rewrite_id(t->TrackId, map, nmap);
        t->NestedParentTrackId = rewrite_id(t->NestedParentTrackId, map, nmap);
        t->CentralTrackId = rewrite_id(t->CentralTrackId, map, nmap);
        t->SinkTrackId = rewrite_id(t->SinkTrackId, map, nmap);
        t->DescendantTrackId = rewrite_id(t->DescendantTrackId, map, nmap);
        for(j = 0; j < res[i].ndesc; j++)
            res[i].DescOwner[j] = rewrite_id(res[i].DescOwner[j], map, nmap);
        res[i].newborn = 0;
    }

    /* 5. nothing provisional may survive */
    int left = 0;
    #pragma omp parallel for reduction(|: left)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) continue;
        if(SUBHALO_IS_PROVISIONAL(P[i].TrackId) || SUBHALO_IS_PROVISIONAL(P[i].BoundTrackId))
            left = 1;
    }
    for(i = 0; i < nres; i++) {
        if(!res[i].valid) continue;
        const struct SubTrack * t = &res[i].rec;
        if(SUBHALO_IS_PROVISIONAL(t->TrackId) || SUBHALO_IS_PROVISIONAL(t->NestedParentTrackId) ||
           SUBHALO_IS_PROVISIONAL(t->CentralTrackId) || SUBHALO_IS_PROVISIONAL(t->SinkTrackId) ||
           SUBHALO_IS_PROVISIONAL(t->DescendantTrackId))
            left = 1;
        for(j = 0; j < res[i].ndesc; j++)
            if(SUBHALO_IS_PROVISIONAL(res[i].DescOwner[j]))
                left = 1;
    }
    if(MPIU_Any(left, Comm))
        endrun(5, "Provisional TrackIds left after the newborn numbering.\n");

    myfree(map);
    myfree(answer);
    myfree(asked);
    ta_free(recvcounts);
    myfree(req);
    myfree(dir);
    ta_free(sendcounts);
    myfree(ent);
    return ntot;
}
