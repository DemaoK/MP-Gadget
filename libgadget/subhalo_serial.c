/* Runtime subhalo finder, the serial keys' steps 4, 5, 7a and 9 (design 7.1, 7.3): the whole-key
 * exchange, local sort and segments, member fetch, domain restore and the serial-key scheduler with
 * its per-thread arenas. The exchange follows fof_distribute_particles: target rank in P[i].TopLeaf,
 * domain_exchange, local sort; domain_maintain restores the domain. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/openmpsort.h"
#include "utils/system.h"

#include "partmanager.h"
#include "slotsmanager.h"
#include "exchange.h"
#include "domain.h"
#include "walltime.h"
#include "subhalo.h"
#include "subhalo_internal.h"
#include "subhalo_plan.h"

/* ------------------------------------------------------------------ step 4: exchange */

static int
subhalo_layout(int i, const void * userdata)
{
    const struct part_manager_type * pman = (const struct part_manager_type *) userdata;
    /* Reuse the topleaf to hold the target task, as fof_sorted_layout does. The domain is
     * restored with domain_maintain afterwards. */
    return pman->Base[i].TopLeaf;
}

int
subhalo_plan_exchange(struct SubhaloPlan * plan, MPI_Comm Comm)
{
    /* the exchange volume: live particles leaving, and arriving from the change of the live count */
    int64_t i, before = 0, out = 0, after = 0;
    #pragma omp parallel for reduction(+: before, out)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage)
            continue;
        before++;
        if(P[i].TopLeaf != plan->ThisTask)
            out++;
    }
    int failure = domain_exchange(subhalo_layout, PartManager, NULL, PartManager, SlotsManager, 10000, Comm);
    #pragma omp parallel for reduction(+: after)
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage)
            after++;
    plan->ExchOut = out;
    plan->ExchIn = after - (before - out);
    if(failure)
        message(1, "Subhalo exchange did not complete on this rank.\n");
    if(subhalo_get_params()->SubhaloDebugForceExchangeFailure) {
        message(0, "Subhalo debug: forcing the exchange failure path.\n");
        failure = 1;
    }
    return failure;
}

/* ------------------------------------------------------------------ local sort and segments */

static int
order_by_key_track_rank(const void * a, const void * b)
{
    const struct particle_data * pa = (const struct particle_data *) a;
    const struct particle_data * pb = (const struct particle_data *) b;
    if(pa->IsGarbage != pb->IsGarbage)
        return (pa->IsGarbage > pb->IsGarbage) - (pa->IsGarbage < pb->IsGarbage);
    /* the processing key: the group, or the encoded key of a travelling field particle */
    const int64_t ka = subhalo_particle_key(pa), kb = subhalo_particle_key(pb);
    if(ka != kb)
        return (ka > kb) - (ka < kb);
    if(pa->TrackId != pb->TrackId)
        return (pa->TrackId > pb->TrackId) - (pa->TrackId < pb->TrackId);
    if(pa->BindRank != pb->BindRank)
        return (pa->BindRank > pb->BindRank) - (pa->BindRank < pb->BindRank);
    /* the ID makes the order, and every sum over it, independent of the arrival order */
    return (pa->ID > pb->ID) - (pa->ID < pb->ID);
}

/* A particle's sort record: the keys of order_by_key_track_rank and its index in 32 bytes, sorted
 * directly, then P is permuted once. The index as last key gives the stable sort's order. */
struct LocalSortRec {
    int64_t key;
    int64_t TrackId;
    MyIDType ID;
    int32_t BindRank;
    uint32_t gi;    /* bit 31: IsGarbage; bits 0-30: the index in P */
};

static int
cmp_localsortrec(const void * a, const void * b)
{
    const struct LocalSortRec * x = (const struct LocalSortRec *) a, * y = (const struct LocalSortRec *) b;
    const uint32_t gx = x->gi >> 31, gy = y->gi >> 31;
    if(gx != gy)
        return (gx > gy) - (gx < gy);
    if(x->key != y->key)
        return (x->key > y->key) - (x->key < y->key);
    if(x->TrackId != y->TrackId)
        return (x->TrackId > y->TrackId) - (x->TrackId < y->TrackId);
    if(x->BindRank != y->BindRank)
        return (x->BindRank > y->BindRank) - (x->BindRank < y->BindRank);
    if(x->ID != y->ID)
        return (x->ID > y->ID) - (x->ID < y->ID);
    const uint32_t ix = x->gi & 0x7fffffffu, iy = y->gi & 0x7fffffffu;
    return (ix > iy) - (ix < iy);
}

static int LocalSortMode = 0;

void
subhalo_local_sort_set_mode(int mode)
{
    LocalSortMode = mode;
}

int
subhalo_par_nchunks(int64_t n)
{
    int64_t nc = 4 * (int64_t) omp_get_max_threads();
    if(nc > n / 4096 + 1)
        nc = n / 4096 + 1;
    return nc > 0 ? (int) nc : 1;
}

/* Bytes of the record sort beyond the particle table: the records and qsort_openmp's copy, then
 * the records, the source and cycle lists and the bitmaps of permute_openmp. */
static double
local_sort_bytes(int64_t n)
{
    return 64. * (double) (n + 1) + 2. * (n / 8 + 1) + 4. * 4096. + 2. * (double) sizeof(struct particle_data) * (n / 1024 + 2)
        + (double) sizeof(struct particle_data) * omp_get_max_threads() + 65536.;
}

void
subhalo_local_sort(void)
{
    const int64_t n = PartManager->NumPart;
    /* the record sort when the index fits and the memory allows it (it needs 64 bytes per
     * particle, the planner reserves 32 per MaxPart); otherwise the indirect sort of P (two
     * pointers per particle). Both give the same order. */
    if(LocalSortMode == 1 || n >= (int64_t) INT32_MAX || local_sort_bytes(n) > (double) mymalloc_freebytes()) {
        qsort_openmp(PartManager->Base, n, sizeof(struct particle_data), order_by_key_track_rank);
        return;
    }
    struct LocalSortRec * rec = (struct LocalSortRec *) mymalloc("SubSortRec", sizeof(struct LocalSortRec) * (n + 1));
    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < n; i++) {
        const struct particle_data * p = &P[i];
        rec[i].key = subhalo_particle_key(p);
        rec[i].TrackId = p->TrackId;
        rec[i].ID = p->ID;
        rec[i].BindRank = p->BindRank;
        rec[i].gi = (uint32_t) i | (p->IsGarbage ? 0x80000000u : 0u);
    }
    qsort_openmp(rec, n, sizeof(struct LocalSortRec), cmp_localsortrec);
    int * src = (int *) mymalloc("SubSortSrc", sizeof(int) * (n + 1));
    int * order = (int *) mymalloc("SubSortOrder", sizeof(int) * (n + 1));
    #pragma omp parallel for
    for(i = 0; i < n; i++)
        src[i] = (int) (rec[i].gi & 0x7fffffffu);
    permute_openmp(PartManager->Base, n, sizeof(struct particle_data), src, order);
    myfree(order);
    myfree(src);
    myfree(rec);
}

int64_t
subhalo_build_segments(const struct SubhaloPlan * plan, struct SubhaloKeySeg ** segs)
{
    const int64_t NumPart = PartManager->NumPart;
    int64_t i, nlive = 0;
    /* the live particles are a prefix after the sort */
    #pragma omp parallel for reduction(+: nlive)
    for(i = 0; i < NumPart; i++)
        if(!P[i].IsGarbage)
            nlive++;
    /* the segment starts in order (chunked, order-preserving) */
#define SEG_START(i) (subhalo_particle_key(&P[i]) >= 0 && ((i) == 0 || subhalo_particle_key(&P[i]) != subhalo_particle_key(&P[(i)-1])))
    const int nc = subhalo_par_nchunks(nlive);
    int64_t * off = ta_malloc("SubSegOff", int64_t, nc + 1);
    int c;
    off[0] = 0;
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t cnt = 0, j;
        for(j = c * nlive / nc; j < (c + 1) * nlive / nc; j++)
            if(SEG_START(j))
                cnt++;
        off[c + 1] = cnt;
    }
    for(c = 0; c < nc; c++)
        off[c + 1] += off[c];
    const int64_t n = off[nc];
    struct SubhaloKeySeg * s = (struct SubhaloKeySeg *) mymalloc2("SubKeySegs", sizeof(struct SubhaloKeySeg) * (n + 1));
    #pragma omp parallel for
    for(c = 0; c < nc; c++) {
        int64_t m = off[c], j;
        for(j = c * nlive / nc; j < (c + 1) * nlive / nc; j++)
            if(SEG_START(j))
                s[m++].start = j;
    }
#undef SEG_START
    ta_free(off);
    int64_t k, nserial = 0;
    int bad = 0;
    #pragma omp parallel for reduction(+: nserial) reduction(|: bad)
    for(k = 0; k < n; k++) {
        struct SubhaloKeySeg * seg = &s[k];
        const int64_t start = seg->start;
        const int64_t key = subhalo_particle_key(&P[start]);
        int64_t j = start;
        while(j < nlive && subhalo_particle_key(&P[j]) == key)
            j++;
        memset(seg, 0, sizeof(*seg));
        seg->g = key;
        seg->start = start;
        seg->n = j - start;
        seg->ipindex = -1;
        /* the particles of a key are sorted by TrackId: count the distinct tags */
        {
            int64_t q;
            for(q = start; q < j; q++)
                if(P[q].TrackId >= 0 && (q == start || P[q].TrackId != P[q-1].TrackId))
                    seg->Tracks++;
        }
        struct SubhaloAssignedKey akey = {seg->g, 0};
        const struct SubhaloAssignedKey * a = (const struct SubhaloAssignedKey *)
            bsearch(&akey, plan->Assigned, plan->NAssigned, sizeof(akey), cmp_assigned_g);
        if(a) {
            seg->Length = a->Length;
            seg->inplace = 0;
            if(seg->n != seg->Length)
                bad = 1;
            nserial++;
        }
        else {
            struct SubhaloInplaceKey ikey;
            ikey.g = seg->g;
            const struct SubhaloInplaceKey * ip = (const struct SubhaloInplaceKey *)
                bsearch(&ikey, plan->Inplace, plan->NInplace, sizeof(ikey), cmp_inplace_g);
            if(!ip)
                bad = 1;
            else {
                seg->Length = ip->Length;
                seg->inplace = 1;
                seg->ipindex = ip - plan->Inplace;
            }
        }
    }
    if(nserial != plan->NAssigned)
        bad = 1;
    if(bad)
        endrun(5, "Subhalo exchange left rank %d with incomplete keys (%ld serial runs for %ld assigned keys).\n",
               plan->ThisTask, nserial, plan->NAssigned);
    *segs = s;
    return n;
}

/* ------------------------------------------------------------------ step 7a: member records */

int64_t
subhalo_fetch_members(const struct SubhaloKeySeg * segs, int64_t nsegs, const int64_t * extra, int64_t nextra,
                      struct SubTrackTable * tab, struct SubTrack ** members, MPI_Comm Comm)
{
    int64_t k, i, n = nextra;
    for(k = 0; k < nsegs; k++)
        for(i = segs[k].start; i < segs[k].start + segs[k].n; i++)
            if(P[i].TrackId >= 0 && (i == segs[k].start || P[i].TrackId != P[i-1].TrackId))
                n++;
    int64_t * ids = (int64_t *) mymalloc("SubMemberIds", sizeof(int64_t) * (n + 1));
    for(n = 0; n < nextra; n++)
        ids[n] = extra[n];
    for(k = 0; k < nsegs; k++)
        for(i = segs[k].start; i < segs[k].start + segs[k].n; i++)
            if(P[i].TrackId >= 0 && (i == segs[k].start || P[i].TrackId != P[i-1].TrackId))
                ids[n++] = P[i].TrackId;
    qsort(ids, n, sizeof(int64_t), cmp_int64);
    int64_t nu = 0;
    for(i = 0; i < n; i++)
        if(nu == 0 || ids[i] != ids[nu-1])
            ids[nu++] = ids[i];
    struct SubTrack * mem = (struct SubTrack *) mymalloc2("SubMembers", sizeof(struct SubTrack) * (nu + 1));
    subhalo_table_fetch(tab, ids, nu, mem, Comm);
    myfree(ids);
    *members = mem;
    return nu;
}

/* ------------------------------------------------------------------ step 9: domain restore */

void
subhalo_restore_domain(DomainDecomp * ddecomp)
{
    /* Peano-sort first so that one exchange iteration does not overfill a rank. */
    slots_gc_sorted(PartManager, SlotsManager);
    if(domain_maintain(ddecomp, NULL))
        endrun(0, "Domain exchange after the subhalo finder did not complete!\n");
    slots_gc_sorted(PartManager, SlotsManager);
}

/* ------------------------------------------------------------------ serial-key scheduler, per-thread arenas (design 7.1) */

struct KeyOrder {
    double cost;
    int64_t n;
    int64_t g;
    int64_t k;
};

static int
cmp_keyorder(const void * a, const void * b)
{
    const struct KeyOrder * x = (const struct KeyOrder *) a, * y = (const struct KeyOrder *) b;
    /* most expensive first, ties by GrNr */
    if(x->cost != y->cost)
        return (x->cost < y->cost) - (x->cost > y->cost);
    return (x->g > y->g) - (x->g < y->g);
}

static struct SubhaloSerialStats SerialStats;

const struct SubhaloSerialStats *
subhalo_serial_last_stats(void)
{
    return &SerialStats;
}

void
subhalo_serial_reset_stats(void)
{
    memset(&SerialStats, 0, sizeof(SerialStats));
}

static void
serial_stats_key(double dt, int64_t n, int alone)
{
    struct SubhaloSerialStats * ss = &SerialStats;
    ss->NKeys++;
    if(n > ss->LargestN)
        ss->LargestN = n;
    if(dt > ss->KeyTimeMax) {
        ss->KeyTimeMax = dt;
        ss->KeyTimeMaxN = n;
    }
    if(alone) {
        ss->NAlone++;
        return;
    }
    const double x = (double) n * log2(n > 2 ? (double) n : 2.);
    ss->FitN += 1;
    ss->FitX += x;
    ss->FitXX += x * x;
    ss->FitY += dt;
    ss->FitXY += x * dt;
}

double
subhalo_run_serial_keys(const struct SubhaloKeySeg * segs, int64_t nsegs, size_t bytes_per_part, const int64_t * extra,
                        subhalo_key_fn fn, void * userdata)
{
    int64_t k, ns = 0;
    double totalcost = 0;
    int t;
    const int nthreads = omp_get_max_threads();
    for(k = 0; k < nsegs; k++)
        if(!segs[k].inplace)
            ns++;
    struct KeyOrder * ord = (struct KeyOrder *) mymalloc("SubKeyOrder", sizeof(struct KeyOrder) * (ns + 1));
    ns = 0;
    for(k = 0; k < nsegs; k++)
        if(!segs[k].inplace) {
            ord[ns].cost = subhalo_key_cost(segs[k].n, segs[k].Tracks);
            ord[ns].n = segs[k].n;
            ord[ns].g = segs[k].g;
            ord[ns].k = k;
            totalcost += ord[ns].cost;
            ns++;
        }
    /* most expensive first, ties by g */
    qsort(ord, ns, sizeof(struct KeyOrder), cmp_keyorder);
#define KEY_ARENA_BYTES(kk) ((size_t) bytes_per_part * (size_t) segs[kk].n + (size_t) (extra ? extra[kk] : 0) + 64)
    double peakfrac = 0;
    SerialStats.NQueues = nthreads;

    /* keys above one thread's share of the cost run alone, with every thread inside the key */
    const double share = totalcost / nthreads;
    int64_t nalone = 0;
    while(nthreads > 1 && nalone < ns && ord[nalone].cost > share)
        nalone++;
    if(nalone > 0) {
        const double ta = MPI_Wtime();
        struct SubhaloArena arena;
        size_t amax = 0;
        for(k = 0; k < nalone; k++)
            if(KEY_ARENA_BYTES(ord[k].k) > amax)
                amax = KEY_ARENA_BYTES(ord[k].k);
        arena.base = (char *) mymalloc2("SubArenaAlone", amax);
        for(k = 0; k < nalone; k++) {
            /* the key sees exactly its own allowance, never the larger buffer, so its tree node
             * budget (leaf size, potentials) does not depend on the keys sharing the buffer */
            arena.size = KEY_ARENA_BYTES(ord[k].k);
            arena.used = 0;
            arena.peak = 0;
            const double t0 = MPI_Wtime();
            fn(&segs[ord[k].k], ord[k].k, &arena, nthreads, userdata);
            serial_stats_key(MPI_Wtime() - t0, ord[k].n, 1);
            const double f = (double) arena.peak / (double) KEY_ARENA_BYTES(ord[k].k);
            if(f > peakfrac)
                peakfrac = f;
        }
        myfree(arena.base);
        SerialStats.AloneWall += MPI_Wtime() - ta;
    }

    /* the rest: static longest-processing-time-first assignment of the costs to the threads; each
     * thread processes its keys in the order of ord (non-increasing cost) */
    double * load = (double *) mymalloc("SubThreadLoad", sizeof(double) * nthreads);
    int64_t * maxb = (int64_t *) mymalloc("SubThreadMaxB", sizeof(int64_t) * nthreads);
    double * tpeak = (double *) mymalloc("SubThreadPeak", sizeof(double) * nthreads);
    double * tbusy = (double *) mymalloc("SubThreadBusy", sizeof(double) * nthreads);   /* per actual worker */
    double * kt = (double *) mymalloc("SubKeyTime", sizeof(double) * (ns + 1));
    int * owner = (int *) mymalloc("SubKeyOwner", sizeof(int) * (ns + 1));
    for(t = 0; t < nthreads; t++) {
        load[t] = 0;
        maxb[t] = 0;
        tpeak[t] = tbusy[t] = 0;
    }
    for(k = nalone; k < ns; k++) {
        int best = 0;
        for(t = 1; t < nthreads; t++)
            if(load[t] < load[best])
                best = t;
        owner[k] = best;
        load[best] += ord[k].cost;
        if((int64_t) KEY_ARENA_BYTES(ord[k].k) > maxb[best])
            maxb[best] = KEY_ARENA_BYTES(ord[k].k);
    }
    struct SubhaloArena * arenas = (struct SubhaloArena *) mymalloc("SubArenas", sizeof(struct SubhaloArena) * nthreads);
    for(t = 0; t < nthreads; t++) {
        arenas[t].size = maxb[t] > 0 ? maxb[t] : 64;
        arenas[t].used = 0;
        arenas[t].peak = 0;
        arenas[t].base = (char *) mymalloc2("SubArena", arenas[t].size);
    }
    const double tthr = MPI_Wtime();
    int team = 1;
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        int tt;
        if(tid == 0)
            team = nt;
        /* if the runtime gives fewer threads, each covers several queues; the busy time is the
         * worker's */
        for(tt = tid; tt < nthreads; tt += nt) {
            int64_t kk;
            for(kk = nalone; kk < ns; kk++) {
                if(owner[kk] != tt) continue;
                /* the key's own allowance; the queue's buffer is the maximum over its keys */
                arenas[tt].size = KEY_ARENA_BYTES(ord[kk].k);
                arenas[tt].used = 0;
                arenas[tt].peak = 0;
                const double t0 = omp_get_wtime();
                fn(&segs[ord[kk].k], ord[kk].k, &arenas[tt], 1, userdata);
                kt[kk] = omp_get_wtime() - t0;
                tbusy[tid] += kt[kk];
                /* relative to what the key itself was allowed (the arena's logical size) */
                const double f = (double) arenas[tt].peak / (double) KEY_ARENA_BYTES(ord[kk].k);
                if(f > tpeak[tt])
                    tpeak[tt] = f;
            }
        }
    }
#undef KEY_ARENA_BYTES
    SerialStats.NThreads = team;
    if(ns > nalone) {
        SerialStats.ThreadedWall += MPI_Wtime() - tthr;
        double bmax = 0, bsum = 0;
        for(t = 0; t < team; t++) {
            bsum += tbusy[t];
            if(tbusy[t] > bmax)
                bmax = tbusy[t];
        }
        SerialStats.ThreadMax += bmax;
        SerialStats.ThreadSum += bsum;
        SerialStats.ThreadMeanSum += bsum / team;
    }
    for(k = nalone; k < ns; k++)
        serial_stats_key(kt[k], ord[k].n, 0);
    for(t = 0; t < nthreads; t++)
        if(tpeak[t] > peakfrac)
            peakfrac = tpeak[t];
    for(t = nthreads - 1; t >= 0; t--)
        myfree(arenas[t].base);
    myfree(arenas);
    myfree(owner);
    myfree(kt);
    myfree(tbusy);
    myfree(tpeak);
    myfree(maxb);
    myfree(load);
    myfree(ord);
    return peakfrac;
}
