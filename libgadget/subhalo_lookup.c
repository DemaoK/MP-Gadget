/* Batched lookups of particle IDs on every rank (design 7.2 steps 1 and 11): orphans' followed sets
 * and the tracers of dead or sunk tracks outside their keys, with their descendants' plurality. */

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

/* ------------------------------------------------------------------ batched ID lookup (steps 1, 11) */

struct IdIndex {
    MyIDType ID;
    int64_t i;
};

static int
cmp_idindex(const void * a, const void * b)
{
    const struct IdIndex * x = (const struct IdIndex *) a, * y = (const struct IdIndex *) b;
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->i > y->i) - (x->i < y->i);
}

/* A hit returned to the asking rank: the position of the id in that rank's batch. */
struct IdReply {
    int64_t q;
    struct SubhaloIdHit h;
};

/* Test knob: at most this many ids per rank and batch (0: the byte cap alone). */
static int64_t SubLookupMaxBatch = 0;
/* High-water mark of the batch buffers (gathered ids, outgoing and returning replies) of the
 * last subhalo_lookup_ids call on this rank. */
static double SubLookupLastPeak = 0;

void
subhalo_lookup_set_max_batch(int64_t maxbatch)
{
    SubLookupMaxBatch = maxbatch > 0 ? maxbatch : 0;
}

double
subhalo_lookup_last_peak_bytes(void)
{
    return SubLookupLastPeak;
}

/* Ids per rank and batch: the largest cap (at least 1) whose worst-case batch buffers,
 * NTask * cap * (id + reply) + cap * reply bytes on any rank, fit SubhaloLookupBatchBytes. */
static int64_t
lookup_cap(int NTask)
{
    const double per = (double) NTask * (double) (sizeof(MyIDType) + sizeof(struct IdReply)) + (double) sizeof(struct IdReply);
    const int64_t bb = SubParams.SubhaloLookupBatchBytes > 0 ? SubParams.SubhaloLookupBatchBytes : 64L * 1024L * 1024L;
    int64_t cap = (int64_t) ((double) bb / per);
    if(cap > INT_MAX / NTask)
        cap = INT_MAX / NTask;
    if(SubLookupMaxBatch > 0 && cap > SubLookupMaxBatch)
        cap = SubLookupMaxBatch;
    return cap > 0 ? cap : 1;
}

double
subhalo_lookup_batch_peak_bytes(int NTask)
{
    const int64_t cap = lookup_cap(NTask);
    return (double) NTask * cap * (double) (sizeof(MyIDType) + sizeof(struct IdReply)) + (double) cap * sizeof(struct IdReply)
        + 3 * 4096.;
}

double
subhalo_lookup_index_bytes(int64_t NumPart)
{
    /* the (ID, index) index and the direct merge-sort copy of qsort_openmp */
    return 2. * sizeof(struct IdIndex) * (double) (NumPart + 1) + 4096.;
}

/* How a rank answers a batch: 0 scan for few queries, else the sorted ID index; 1 always the
 * index; 2 always the scan (test knob). Same answers: the lowest P index holding the ID. */
static int SubLookupMode = 0;

void
subhalo_lookup_set_mode(int mode)
{
    SubLookupMode = mode;
}

struct LookupQuery {
    MyIDType ID;
    int64_t pos;
};

static int
cmp_lookupquery(const void * a, const void * b)
{
    const struct LookupQuery * x = (const struct LookupQuery *) a, * y = (const struct LookupQuery *) b;
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->pos > y->pos) - (x->pos < y->pos);
}

/* The sorted (ID, index) index of the local live particles. */
static struct IdIndex *
lookup_build_index(int64_t * nlout)
{
    const int64_t N = PartManager->NumPart;
    const int nc = subhalo_par_nchunks(N);
    int64_t * off = ta_malloc("SubLookupOff", int64_t, nc + 1);
    int c;
    off[0] = 0;
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t cnt = 0, i;
        for(i = c * N / nc; i < (c + 1) * N / nc; i++)
            if(subhalo_part_live(&P[i]))
                cnt++;
        off[c + 1] = cnt;
    }
    for(c = 0; c < nc; c++)
        off[c + 1] += off[c];
    const int64_t nl = off[nc];
    struct IdIndex * idx = (struct IdIndex *) mymalloc("SubIdIndex", sizeof(struct IdIndex) * (nl + 1));
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t m = off[c], i;
        for(i = c * N / nc; i < (c + 1) * N / nc; i++)
            if(subhalo_part_live(&P[i])) {
                idx[m].ID = P[i].ID;
                idx[m].i = i;
                m++;
            }
    }
    ta_free(off);
    qsort_openmp(idx, nl, sizeof(struct IdIndex), cmp_idindex);
    *nlout = nl;
    return idx;
}

/* all[0 .. tot) on entry: the gathered ids; on exit: the index in P of the live particle with
 * that id (the lowest if duplicated), or (MyIDType) -1. */
static void
lookup_answer_scan(MyIDType * all, int64_t tot)
{
    struct LookupQuery * qs = (struct LookupQuery *) mymalloc("SubLookupQs", sizeof(struct LookupQuery) * (tot + 1));
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < tot; i++) {
        qs[i].ID = all[i];
        qs[i].pos = i;
        all[i] = (MyIDType) -1;
    }
    qsort_openmp(qs, tot, sizeof(struct LookupQuery), cmp_lookupquery);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!subhalo_part_live(&P[i]))
            continue;
        const MyIDType id = P[i].ID;
        int64_t lo = 0, hi = tot;
        while(lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if(qs[mid].ID < id) lo = mid + 1;
            else hi = mid;
        }
        for(; lo < tot && qs[lo].ID == id; lo++) {
            MyIDType * slot = &all[qs[lo].pos];
            MyIDType old = __atomic_load_n(slot, __ATOMIC_RELAXED);
            while((MyIDType) i < old && !__atomic_compare_exchange_n(slot, &old, (MyIDType) i, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                ;
        }
    }
    myfree(qs);
}

void
subhalo_lookup_ids(const MyIDType * ids, int64_t n, struct SubhaloIdHit * out, MPI_Comm Comm)
{
    int NTask, r;
    MPI_Comm_size(Comm, &NTask);
    int64_t i;
    memset(out, 0, sizeof(struct SubhaloIdHit) * n);

    /* the live particles; the sorted (ID, index) index is built at the first batch that uses it */
    int64_t nlive = 0;
    #pragma omp parallel for reduction(+: nlive)
    for(i = 0; i < PartManager->NumPart; i++)
        if(subhalo_part_live(&P[i]))
            nlive++;
    /* scan when all queries together are few against the live particles; else build the index once */
    int64_t ntot = 0;
    MPI_Allreduce(&n, &ntot, 1, MPI_INT64, MPI_SUM, Comm);
    const int scan = SubLookupMode == 2 || (SubLookupMode == 0 && ntot <= nlive / 16);
    struct IdIndex * idx = NULL;
    int64_t nl = 0;
    if(!scan)
        idx = lookup_build_index(&nl);

    /* batches of at most cap ids per rank (lookup_cap) */
    const int64_t cap = lookup_cap(NTask);
    SubLookupLastPeak = 0;
    int * counts = ta_malloc("SubLookupCounts", int, NTask);
    int * displs = ta_malloc("SubLookupDispls", int, NTask);
    int64_t * sendcounts = ta_malloc("SubLookupSend", int64_t, NTask);
    MPI_Datatype idtype;
    MPI_Type_contiguous(sizeof(MyIDType), MPI_BYTE, &idtype);
    MPI_Type_commit(&idtype);
    int64_t done = 0;
    while(1) {
        const int64_t nb = n - done < cap ? n - done : cap;
        if(!MPIU_Any(nb > 0, Comm))
            break;
        int nbi = (int) nb;
        MPI_Allgather(&nbi, 1, MPI_INT, counts, 1, MPI_INT, Comm);
        int64_t tot = 0;
        for(r = 0; r < NTask; r++) {
            displs[r] = (int) tot;
            tot += counts[r];
        }
        MyIDType * all = (MyIDType *) mymalloc("SubLookupIds", sizeof(MyIDType) * (tot + 1));
        MPI_Allgatherv((void *) (ids + done), nbi, idtype, all, counts, displs, idtype, Comm);
        /* the answering particle of every gathered id */
        if(scan)
            lookup_answer_scan(all, tot);
        else {
            #pragma omp parallel for
            for(i = 0; i < tot; i++) {
                const MyIDType id = all[i];
                int64_t lo = 0, hi = nl;
                while(lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if(idx[mid].ID < id) lo = mid + 1;
                    else hi = mid;
                }
                all[i] = (lo < nl && idx[lo].ID == id) ? (MyIDType) idx[lo].i : (MyIDType) -1;
            }
        }
        /* hits for every asking rank, in rank order */
        int64_t nhit = 0;
        memset(sendcounts, 0, sizeof(int64_t) * NTask);
        for(r = 0; r < NTask; r++)
            for(i = displs[r]; i < displs[r] + counts[r]; i++)
                if(all[i] != (MyIDType) -1) {
                    sendcounts[r]++;
                    nhit++;
                }
        struct IdReply * rep = (struct IdReply *) mymalloc("SubLookupRep", sizeof(struct IdReply) * (nhit + 1));
        nhit = 0;
        for(r = 0; r < NTask; r++)
            for(i = displs[r]; i < displs[r] + counts[r]; i++) {
                if(all[i] == (MyIDType) -1)
                    continue;
                const struct particle_data * p = &P[all[i]];
                struct IdReply * e = &rep[nhit++];
                memset(e, 0, sizeof(*e));
                e->q = i - displs[r];
                e->h.GrNr = p->GrNr >= 0 ? p->GrNr : -1;
                e->h.BoundTrackId = p->BoundTrackId;
                e->h.Penalty = p->TracerPenalty;
                int d;
                for(d = 0; d < 3; d++) {
                    e->h.Pos[d] = p->Pos[d];
                    e->h.Vel[d] = p->Vel[d];
                }
                e->h.found = 1;
            }
        struct IdReply * back = NULL;
        int64_t nback = subhalo_alltoallv(rep, sendcounts, sizeof(struct IdReply), (void **) &back, 0, NULL, Comm);
        const double peak = (double) (tot + 1) * sizeof(MyIDType) + (double) (nhit + 1) * sizeof(struct IdReply)
            + (double) (nback + 1) * sizeof(struct IdReply);
        if(peak > SubLookupLastPeak)
            SubLookupLastPeak = peak;
        /* an ID lives on one rank; if it were duplicated the last answer wins */
        for(i = 0; i < nback; i++)
            out[done + back[i].q] = back[i].h;
        myfree(back);
        myfree(rep);
        myfree(all);
        done += nb;
    }
    MPI_Type_free(&idtype);
    ta_free(sendcounts);
    ta_free(displs);
    ta_free(counts);
    if(idx)
        myfree(idx);
}

/* An orphan record homed here whose markers are looked up this call. */
static int
orphan_looks_up(const struct SubTrack * t)
{
    const MyIDType * ids;
    return t->NboundPrev <= 0 && t->SnapTracerLost < 0 && orphan_markers(t, &ids) > 0;
}

/* Lower median of v[0..n) (n >= 1); v is reordered. */
static double
lower_median(double * v, int n)
{
    int i, j;
    for(i = 1; i < n; i++) {
        const double x = v[i];
        for(j = i; j > 0 && v[j-1] > x; j--)
            v[j] = v[j-1];
        v[j] = x;
    }
    return v[(n - 1) / 2];
}

int
subhalo_orphan_update(struct SubTrack * t, const struct SubhaloIdHit * hit, int n, const double Offset[3], double BoxSize, double vfac, int SnapNum)
{
    int f[SUBHALO_MAXTRACERS], keep[SUBHALO_MAXTRACERS];
    double x[SUBHALO_MAXTRACERS][3], c[3];
    int nf = 0, nk = 0, j, k, d;
    if(n > SUBHALO_MAXTRACERS)
        endrun(5, "Orphan %ld follows %d markers, at most %d supported.\n", t->TrackId, n, SUBHALO_MAXTRACERS);
    for(j = 0; j < n; j++)
        if(hit[j].found)
            f[nf++] = j;
    if(nf == 0) {
        /* no marker exists any more: the record keeps its last position */
        t->Host = -1;
        t->SnapTracerLost = SnapNum;
        t->NOrphanFound = 0;
        t->OrphanSigmaX = -1;
        t->MarkerWeight = 0;
        return 0;
    }
    /* positions unwrapped about the first found marker; the centre: one or two found, the first,
     * most-bound one (a median of two would mix particles), else the component-wise lower median */
    for(k = 0; k < nf; k++)
        for(d = 0; d < 3; d++)
            x[k][d] = hit[f[0]].Pos[d] + NEAREST(hit[f[k]].Pos[d] - hit[f[0]].Pos[d], BoxSize);
    for(d = 0; d < 3; d++) {
        double v[SUBHALO_MAXTRACERS];
        for(k = 0; k < nf; k++)
            v[k] = x[k][d];
        c[d] = nf <= 2 ? x[0][d] : lower_median(v, nf);
    }
    /* the minimum image about the centre (a set wider than half the box); unchanged when no shift is needed */
    for(k = 0; k < nf; k++)
        for(d = 0; d < 3; d++) {
            if(x[k][d] - c[d] > 0.5 * BoxSize)
                x[k][d] -= BoxSize;
            else if(x[k][d] - c[d] < -0.5 * BoxSize)
                x[k][d] += BoxSize;
        }
    /* clip at 3 OrphanScale about the centre; if nothing is left, the marker closest to it */
    const double r2max = 9. * (double) t->OrphanScale * (double) t->OrphanScale;
    int closest = 0;
    double dmin = -1;
    for(k = 0; k < nf; k++) {
        double r2 = 0;
        for(d = 0; d < 3; d++)
            r2 += (x[k][d] - c[d]) * (x[k][d] - c[d]);
        if(dmin < 0 || r2 < dmin) {
            dmin = r2;
            closest = k;
        }
        if(t->OrphanScale <= 0 || r2 <= r2max)
            keep[nk++] = k;
    }
    if(nk == 0)
        keep[nk++] = closest;
    if(nk == 1) {
        /* one marker: its own values, bit for bit as the single-particle orphan */
        const struct SubhaloIdHit * h = &hit[f[keep[0]]];
        t->Host = h->GrNr;
        for(d = 0; d < 3; d++) {
            t->TracerPos[d] = subhalo_wrap_box(h->Pos[d] - Offset[d], BoxSize);
            t->TracerVel[d] = h->Vel[d] * vfac;
        }
        t->NOrphanFound = 1;
        t->OrphanSigmaX = -1;
        t->MarkerWeight = 1.0 / (1.0 + sqrt((double) f[keep[0]]));
        return 1;
    }
    /* marker weights exp(-penalty) / (1 + sqrt(rank in the followed set)): a scattered marker traces the remnant less well */
    int64_t cand[SUBHALO_MAXTRACERS];
    double w[SUBHALO_MAXTRACERS], W = 0, mx[3] = {0, 0, 0}, mv[3] = {0, 0, 0};
    for(k = 0; k < nk; k++) {
        w[k] = subhalo_tracer_weight(hit[f[keep[k]]].Penalty) / (1.0 + sqrt((double) f[keep[k]]));
        W += w[k];
    }
    /* every final weight underflowed (huge penalties): the rank weights alone, whose sum is > 0 */
    if(!(W > 0)) {
        W = 0;
        for(k = 0; k < nk; k++) {
            w[k] = 1.0 / (1.0 + sqrt((double) f[keep[k]]));
            W += w[k];
        }
    }
    for(k = 0; k < nk; k++) {
        const int q = keep[k];
        const struct SubhaloIdHit * h = &hit[f[q]];
        cand[k] = h->GrNr + 1;   /* -1 (no group) is a legitimate ballot */
        for(d = 0; d < 3; d++) {
            mx[d] += w[k] * x[q][d];
            mv[d] += w[k] * h->Vel[d];
        }
    }
    double s2 = 0;
    for(d = 0; d < 3; d++) {
        mx[d] /= W;
        mv[d] /= W;
        for(k = 0; k < nk; k++)
            s2 += (x[keep[k]][d] - mx[d]) * (x[keep[k]][d] - mx[d]);
    }
    t->Host = subhalo_weighted_plurality(cand, w, nk, NULL, NULL) - 1;
    /* the membership weight of the resurrection order: rank weights only, a scattered marker inside
     * the group is still a member */
    t->MarkerWeight = 0;
    for(k = 0; k < nk; k++)
        if(cand[k] - 1 == t->Host)
            t->MarkerWeight += 1.0 / (1.0 + sqrt((double) f[keep[k]]));
    for(d = 0; d < 3; d++) {
        t->TracerPos[d] = subhalo_wrap_box(mx[d] - Offset[d], BoxSize);
        t->TracerVel[d] = mv[d] * vfac;
    }
    t->NOrphanFound = nk;
    t->OrphanSigmaX = sqrt(s2 / nk);
    return 1;
}

void
subhalo_orphan_lookup(struct SubTrackTable * tab, int SnapNum, double atime, int64_t * nlook, int64_t * nfound, MPI_Comm Comm)
{
    int64_t i, n = 0, norph = 0;
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * t = &tab->Track[i];
        const MyIDType * mk;
        if(orphan_looks_up(t)) {
            n += orphan_markers(t, &mk);
            norph++;
        }
    }
    MyIDType * ids = (MyIDType *) mymalloc("SubOrphanIds", sizeof(MyIDType) * (n + 1));
    int64_t * which = (int64_t *) mymalloc("SubOrphanWhich", sizeof(int64_t) * (norph + 1));
    struct SubhaloIdHit * hit = (struct SubhaloIdHit *) mymalloc("SubOrphanHits", sizeof(struct SubhaloIdHit) * (n + 1));
    n = norph = 0;
    for(i = 0; i < tab->NLocal; i++) {
        const struct SubTrack * t = &tab->Track[i];
        if(!orphan_looks_up(t))
            continue;
        const MyIDType * mk;
        const int nm = orphan_markers(t, &mk);
        int j;
        for(j = 0; j < nm; j++)
            ids[n++] = mk[j];
        which[norph++] = i;
    }
    subhalo_lookup_ids(ids, n, hit, Comm);
    const double vfac = subhalo_vel_to_kms(atime);
    const double BoxSize = PartManager->BoxSize;
    int64_t nf = 0, o = 0;
    /* HBT's one-tracer vote (subhalo_tracking.cpp:231-236) and the tracer state
     * (subhalo_update_mostbound.cpp:20-70), generalised to the followed set; physical frame, km/s */
    for(i = 0; i < norph; i++) {
        struct SubTrack * t = &tab->Track[which[i]];
        const MyIDType * mk;
        const int nm = orphan_markers(t, &mk);
        int j;
        for(j = 0; j < nm; j++)
            nf += hit[o + j].found;
        subhalo_orphan_update(t, hit + o, nm, PartManager->CurrentParticleOffset, BoxSize, vfac, SnapNum);
        o += nm;
    }
    myfree(hit);
    myfree(which);
    myfree(ids);
    int64_t loc[2] = {n, nf}, tot[2];
    MPI_Allreduce(loc, tot, 2, MPI_INT64, MPI_SUM, Comm);
    *nlook = tot[0];
    *nfound = tot[1];
}


/* ------------------------------------------------------------------ step 11: remote descendants */

int64_t
subhalo_weighted_plurality(const int64_t * cand, const double * w, int n, double * top, double * second)
{
    int64_t uc[SUBHALO_MAXTRACERS];
    double us[SUBHALO_MAXTRACERS];
    int k, c, nu = 0, best = -1;
    if(n > SUBHALO_MAXTRACERS)
        endrun(5, "Weighted plurality of %d ballots, at most %d supported.\n", n, SUBHALO_MAXTRACERS);
    /* per-candidate sums in ballot order (an indexed update, so never reassociated) */
    for(k = 0; k < n; k++) {
        if(cand[k] < 0)
            continue;
        for(c = 0; c < nu && uc[c] != cand[k]; c++) {}
        if(c == nu) {
            uc[nu] = cand[k];
            us[nu++] = 0;
        }
        us[c] += w[k];
    }
    for(c = 0; c < nu; c++)
        if(best < 0 || us[c] > us[best] || (us[c] == us[best] && uc[c] < uc[best]))
            best = c;
    double run = 0;
    for(c = 0; c < nu; c++)
        if(c != best && us[c] > run)
            run = us[c];
    if(top)
        *top = best >= 0 ? us[best] : 0;
    if(second)
        *second = run;
    return best >= 0 ? uc[best] : SUBHALO_NOTRACK;
}

/* Dead or sunk tracks: tracers outside their key looked up on the restored layout, then the
 * BoundTrackId with the largest summed tracer weight exp(-TracerPenalty) (the majority as HBT's
 * merger_tree.cpp without penalties; ties to the smaller TrackId, -1 if none). Collective. */
void
subhalo_finish_descendants(struct SubhaloResult * res, int64_t nres, MPI_Comm Comm)
{
    int64_t i, n = 0;
    int j;
    for(i = 0; i < nres; i++)
        if(res[i].valid)
            for(j = 0; j < res[i].ndesc; j++)
                if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING)
                    n++;
    MyIDType * ids = (MyIDType *) mymalloc("SubDescIds", sizeof(MyIDType) * (n + 1));
    struct SubhaloIdHit * hit = (struct SubhaloIdHit *) mymalloc("SubDescHits", sizeof(struct SubhaloIdHit) * (n + 1));
    n = 0;
    for(i = 0; i < nres; i++)
        if(res[i].valid)
            for(j = 0; j < res[i].ndesc; j++)
                if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING)
                    ids[n++] = res[i].rec.TracerIDs[j];
    subhalo_lookup_ids(ids, n, hit, Comm);
    n = 0;
    int64_t nremote = 0, nfound = 0, ndecided = 0;
    for(i = 0; i < nres; i++) {
        if(!res[i].valid || res[i].ndesc <= 0)
            continue;
        for(j = 0; j < res[i].ndesc; j++)
            if(res[i].DescOwner[j] == SUBHALO_DESC_PENDING) {
                res[i].DescOwner[j] = hit[n].found ? hit[n].BoundTrackId : -1;
                res[i].DescWeight[j] = hit[n].found ? subhalo_tracer_weight(hit[n].Penalty) : 1;
                nremote++;
                nfound += hit[n].found;
                n++;
            }
        double w[SUBHALO_MAXTRACERS];
        for(j = 0; j < res[i].ndesc; j++)
            w[j] = res[i].DescWeight[j];
        res[i].rec.DescendantTrackId = subhalo_weighted_plurality(res[i].DescOwner, w, res[i].ndesc, NULL, NULL);
        ndecided++;
    }
    myfree(hit);
    myfree(ids);
    int64_t loc[3] = {ndecided, nremote, nfound}, tot[3];
    MPI_Allreduce(loc, tot, 3, MPI_INT64, MPI_SUM, Comm);
    message(0, "Subhalo descendants: %ld dead or sunk tracks; %ld tracers looked up outside their keys, %ld found.\n",
            tot[0], tot[1], tot[2]);
}
