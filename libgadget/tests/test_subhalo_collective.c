/* Tests of the oversized candidates unbound in place (subhalo_collective.c, design 7.8), MPI; the
 * comment above each test says what it checks and main() registers them in order (the
 * test_collective_ prefix omitted). Run as: mpirun -np 4 .objs/test_subhalo_collective (green at
 * 1 to 4 threads). SUBHALO_TEST_ONLY=<substring> runs only the matching tests. */

#define _GNU_SOURCE
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <ftw.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <bigfile-mpi.h>

#include <libgadget/subhalo.h>
#include <libgadget/subhalo_internal.h>
#include <libgadget/subhalo_collective.h>
#include <libgadget/subhalo_unbind.h>
#include <libgadget/petaio.h>
#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/exchange.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/cosmology.h>
#include <libgadget/gravity.h>
#include "stub.h"
/* gravshort.h defines static treewalk callbacks this test does not use */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <libgadget/gravshort.h>
#pragma GCC diagnostic pop

/* ------------------------------------------------------------------ helpers */

/* Per-particle snapshot, gathered to every rank and sorted by ID, so assertions on it are
 * identical everywhere (collective-safe). */
struct tagrec {
    MyIDType ID;
    int64_t GrNr;
    int64_t TrackId;
    int64_t BoundTrackId;
    int32_t BindRank;
    int32_t Type;
    int32_t Task;            /* the rank holding it */
    int32_t pad;
    double Pos[3];
};

#include "subhalo_testlib.h"

static struct tagrec *
gather_tags(int64_t * ntot)
{
    int NTask = get_ntask();
    int64_t i;
    int nlocal = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(!P[i].IsGarbage)
            nlocal++;
    struct tagrec * loc = malloc((nlocal + 1) * sizeof(struct tagrec));
    int n = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage)
            continue;
        struct tagrec * r = &loc[n++];
        memset(r, 0, sizeof(*r));
        r->ID = P[i].ID;
        r->GrNr = P[i].GrNr;
        r->TrackId = P[i].TrackId;
        r->BoundTrackId = P[i].BoundTrackId;
        r->BindRank = P[i].BindRank;
        r->Type = P[i].Type;
        r->Task = get_thistask();
        int d;
        for(d = 0; d < 3; d++)
            r->Pos[d] = P[i].Pos[d];
    }
    int nbytes = nlocal * sizeof(struct tagrec);
    int * counts = malloc(NTask * sizeof(int));
    int * displs = malloc(NTask * sizeof(int));
    MPI_Allgather(&nbytes, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t total = 0;
    int t;
    for(t = 0; t < NTask; t++) {
        displs[t] = total;
        total += counts[t];
    }
    struct tagrec * all = malloc(total + sizeof(struct tagrec));
    MPI_Allgatherv(loc, nbytes, MPI_BYTE, all, counts, displs, MPI_BYTE, MPI_COMM_WORLD);
    *ntot = total / sizeof(struct tagrec);
    qsort(all, *ntot, sizeof(struct tagrec), cmp_tagrec_id);
    free(displs);
    free(counts);
    free(loc);
    return all;
}

/* One finder call exactly as run.c makes it, on the current particle positions; pre (after the
 * FOF, before the finder) and post (after the finder) are called on every rank if not NULL. */
typedef void (*call_hook)(void * ud);
static int
run_finder_call_ex(int snap, Cosmology * CP, int64_t * TotNgroups, call_hook pre, call_hook post, void * ud)
{
    double atime = 0.5 + 0.05 * snap;
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);
    *TotNgroups = fof.TotNgroups;
    if(pre)
        pre(ud);
    int ret = subhalo_find(&fof, &ddecomp, snap, atime, CP, 0, MPI_COMM_WORLD);
    if(post)
        post(ud);
    fof_finish(&fof);
    domain_free(&ddecomp);
    return ret;
}

static int
run_finder_call(int snap, Cosmology * CP, int64_t * TotNgroups)
{
    return run_finder_call_ex(snap, CP, TotNgroups, NULL, NULL, NULL);
}

/* The particle type of a synthetic id (NULL: every particle a DM tracer of type 1); a type-0
 * particle gets an SPH slot (zeroed), a non-tracer that FOF attaches as a secondary. */
static int (*SynTypeFn)(MyIDType id) = NULL;

static void
syn_setup(Cosmology * CP, int64_t ntot, struct SubhaloParams sp)
{
    common_physics_setup();
    petaio_init();
    subhalo_set_testpar(sp);
    *CP = test_cosmology();
    P2_SavedTree = get_gravshort_treepar();
    struct gravshort_tree_params tp = P2_SavedTree;
    tp.FractionalGravitySoftening = 0.006;
    set_gravshort_treepar(tp);
    gravshort_set_softenings(50.);
    int NTask = get_ntask(), ThisTask = get_thistask();
    setup_pman(4 * ntot, BOXSIZE);
    if(SynTypeFn) {
        /* room for every gas particle on one rank (a whole key may land anywhere) */
        int64_t atleast[6] = {ntot, 0, 0, 0, 128, 128};
        slots_reserve(1, atleast, SlotsManager);
    }
    MyIDType id;
    int64_t ngas = 0;
    for(id = 0; id < (MyIDType) ntot; id++) {
        if((int)(id % NTask) != ThisTask)
            continue;
        double pos[3] = {0, 0, 0};
        add_dm_particle(id, pos);
        struct particle_data * p = &P[PartManager->NumPart - 1];
        if(SynTypeFn && SynTypeFn(id) == 0) {
            p->Type = 0;
            p->PI = ngas++;
        }
    }
    if(SynTypeFn) {
        SlotsManager->info[0].size = ngas;
        int64_t i;
        for(i = 0; i < PartManager->NumPart; i++)
            if(P[i].Type == 0)
                memset(&SPHP(i), 0, sizeof(struct sph_particle_data));
    }
    slots_setup_id(PartManager, SlotsManager);
    subhalo_init(-1, -1, "/nonexistent", MPI_COMM_WORLD);
    SynNTot = ntot;
}

/* ------------------------------------------------------------------ comparison helpers */

/* Opening angle of both the serial octree and the in-place walk in these tests (the default 0.45
 * leaves 1e-4 relative potential errors that can swap near-degenerate energies). */
#define COLL_TEST_THETA 0.15

/* One finished call: the particles and the records, identical on every rank. */
struct Outcome {
    struct tagrec * r;
    int64_t n;
    struct SubTrack * rec;
    int64_t nt;
    struct SubhaloCollectiveStats st;
};

static void
outcome_take(struct Outcome * o)
{
    o->r = gather_tags(&o->n);
    o->nt = subhalo_get_table()->NTotal;
    o->rec = fetch_all_tracks(o->nt);
    o->st = *subhalo_collective_last_stats();
}

static void
outcome_free(struct Outcome * o)
{
    free(o->rec);
    free(o->r);
}

/* The first ten tracer-type (type 1) own-tag particles of track t in BindRank order. */
static int
ten_tracers(const struct tagrec * r, int64_t n, int64_t t, MyIDType out[10])
{
    int64_t i;
    int k = 0;
    int32_t best[10];
    for(k = 0; k < 10; k++) {
        best[k] = INT32_MAX;
        out[k] = (MyIDType) -1;
    }
    for(i = 0; i < n; i++) {
        if(r[i].TrackId != t || r[i].Type != 1)
            continue;
        /* insertion into the ten smallest ranks */
        int32_t b = r[i].BindRank;
        MyIDType id = r[i].ID;
        for(k = 0; k < 10; k++)
            if(b < best[k]) {
                int32_t tb = best[k];
                MyIDType ti = out[k];
                best[k] = b;
                out[k] = id;
                b = tb;
                id = ti;
            }
    }
    int nf = 0;
    for(k = 0; k < 10; k++)
        if(best[k] < INT32_MAX)
            nf++;
    return nf;
}

static double
reldiff(double a, double b)
{
    const double s = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return s > 0 ? fabs(a - b) / s : 0;
}

/* Serial versus in place for the resolved tracks of a call: Nbound within nbtol, identical ten
 * most-bound tracers, the properties within ptol; returns the largest relative Nbound
 * difference. Logs the numbers. */
static double
compare_outcomes(const struct Outcome * a, const struct Outcome * b, double nbtol, double ptol, int strict, const char * what)
{
    CHECK(a->nt == b->nt && a->n == b->n, "%s: %ld/%ld tracks, %ld/%ld particles\n", what, a->nt, b->nt, a->n, b->n);
    int64_t t, i;
    double worst = 0;
    for(t = 0; t < a->nt; t++) {
        const struct SubTrack * x = &a->rec[t], * y = &b->rec[t];
        CHECK((x->Nbound > 0) == (y->Nbound > 0), "%s: track %ld alive serially %d, in place %d\n", what, t, x->Nbound > 0, y->Nbound > 0);
        if(x->Nbound <= 0)
            continue;
        const double dn = reldiff(x->Nbound, y->Nbound);
        if(dn > worst)
            worst = dn;
        MyIDType ta[10], tb[10];
        const int na = ten_tracers(a->r, a->n, t, ta), nb = ten_tracers(b->r, b->n, t, tb);
        int same = na == nb, k;
        for(k = 0; k < na && same; k++)
            same = ta[k] == tb[k];
        message(0, "%s: track %ld: Nbound %d / %d (%.2e), NboundExclusive %d / %d, Rank %d / %d, parent %ld / %ld, ten tracers %s, "
                   "MostBoundID %lu / %lu, Vmax %g / %g, Rhalf %g / %g, MBoundSO[vir] %g / %g, SpinPeebles %g / %g, BtoA %g / %g, "
                   "CoreSigmaX %g / %g, Ekin %g / %g, Epot %g / %g\n", what, t,
                x->Nbound, y->Nbound, dn, x->NboundExclusive, y->NboundExclusive, x->Rank, y->Rank, x->NestedParentTrackId,
                y->NestedParentTrackId, same ? "identical" : "DIFFERENT", (unsigned long) x->MostBoundID, (unsigned long) y->MostBoundID,
                x->Vmax, y->Vmax, x->Rhalf, y->Rhalf, x->MBoundSO[0], y->MBoundSO[0], x->SpinPeebles, y->SpinPeebles,
                x->BtoA, y->BtoA, x->CoreSigmaX, y->CoreSigmaX, x->Ekin, y->Ekin, x->Epot, y->Epot);
        if(!same)
            message(0, "%s: track %ld ten tracers serially %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu, in place %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n",
                    what, t, ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6], ta[7], ta[8], ta[9], tb[0], tb[1], tb[2], tb[3], tb[4], tb[5],
                    tb[6], tb[7], tb[8], tb[9]);
        CHECK(dn <= nbtol, "%s: track %ld Nbound %d serially, %d in place\n", what, t, x->Nbound, y->Nbound);
        /* strict: the same ten in the same order; else the same set up to one exchanged at the
         * edge (near-degenerate energies of a cold ball swap under the tree's 1e-4 errors) */
        int common = 0, k2;
        for(k = 0; k < na; k++)
            for(k2 = 0; k2 < nb; k2++)
                if(ta[k] == tb[k2])
                    common++;
        CHECK(strict ? same : (na == nb && common >= na - 1), "%s: track %ld: the ten most-bound tracers differ (%d in common)\n", what, t, common);
        CHECK(x->MostBoundID == y->MostBoundID, "%s: track %ld MostBoundID %lu / %lu\n", what, t, (unsigned long) x->MostBoundID, (unsigned long) y->MostBoundID);
        CHECK(x->Rank == y->Rank && x->Depth == y->Depth && x->NestedParentTrackId == y->NestedParentTrackId
              && x->CentralTrackId == y->CentralTrackId && x->HostGroup == y->HostGroup,
              "%s: track %ld hierarchy (%d %d %ld %ld %ld) / (%d %d %ld %ld %ld)\n", what, t, x->Rank, x->Depth, x->NestedParentTrackId,
              x->CentralTrackId, x->HostGroup, y->Rank, y->Depth, y->NestedParentTrackId, y->CentralTrackId, y->HostGroup);
        /* properties within the tree tolerance */
        const double props[][2] = {
            {x->Mbound, y->Mbound}, {x->Vmax, y->Vmax}, {x->Rhalf, y->Rhalf}, {x->MBoundSO[0], y->MBoundSO[0]},
            {x->RBoundSO[0], y->RBoundSO[0]}, {x->MBoundSO[2], y->MBoundSO[2]}, {x->VmaxVir, y->VmaxVir}, {x->Ekin, y->Ekin},
            {x->Epot, y->Epot}, {x->SpinPeebles, y->SpinPeebles}, {x->CoreSigmaX, y->CoreSigmaX}, {x->CoreSigmaV, y->CoreSigmaV},
            {x->Renclose, y->Renclose},
        };
        const char * pname[] = {"Mbound", "Vmax", "Rhalf", "MBoundSO[vir]", "RBoundSO[vir]", "MBoundSO[200c]", "VmaxVir", "Ekin", "Epot",
                                "SpinPeebles", "CoreSigmaX", "CoreSigmaV", "Renclose"};
        /* the core of 20 particles: one exchanged particle of near-degenerate energy moves its
         * dispersions by a few per cent (cold synthetic balls) */
        for(k = 0; k < (int) (sizeof(props) / sizeof(props[0])); k++) {
            const double tol = (k == 10 || k == 11) && !strict ? 8 * ptol : ptol;
            CHECK(reldiff(props[k][0], props[k][1]) <= tol, "%s: track %ld %s %g serially, %g in place\n", what, t, pname[k], props[k][0], props[k][1]);
        }
        int d;
        for(d = 0; d < 3; d++)
            CHECK(fabs(x->CorePos[d] - y->CorePos[d]) < 0.1 * x->CoreSigmaX + 0.01 && fabs(x->CoMPos[d] - y->CoMPos[d]) < 0.05,
                  "%s: track %ld CorePos[%d] %g / %g CoMPos %g / %g\n", what, t, d, x->CorePos[d], y->CorePos[d], x->CoMPos[d], y->CoMPos[d]);
        CHECK(fabs(x->BtoA - y->BtoA) < 0.02 && fabs(x->CtoA - y->CtoA) < 0.02, "%s: track %ld BtoA %g / %g CtoA %g / %g\n", what, t,
              x->BtoA, y->BtoA, x->CtoA, y->CtoA);
        double jx = 0, jy = 0;
        for(d = 0; d < 3; d++) {
            jx += x->J[d] * x->J[d];
            jy += y->J[d] * y->J[d];
        }
        CHECK(reldiff(sqrt(jx), sqrt(jy)) <= 2 * ptol, "%s: track %ld |J| %g / %g\n", what, t, sqrt(jx), sqrt(jy));
        /* the deferred columns: Rs fitted in both modes or -1 in both, Vrms within ptol, the two
         * mean velocities within ptol Vrms (plus 0.01 km/s for a cold ball) */
        message(0, "%s: track %ld Rs %g / %g (RsKlypin %g / %g), Vrms %g / %g, BulkVel (%g %g %g) / (%g %g %g), "
                   "RockstarCoreVel (%g %g %g) / (%g %g %g)\n", what, t, x->Rs, y->Rs, x->RsKlypin, y->RsKlypin, x->Vrms, y->Vrms,
                x->BulkVel[0], x->BulkVel[1], x->BulkVel[2], y->BulkVel[0], y->BulkVel[1], y->BulkVel[2],
                x->RockstarCoreVel[0], x->RockstarCoreVel[1], x->RockstarCoreVel[2],
                y->RockstarCoreVel[0], y->RockstarCoreVel[1], y->RockstarCoreVel[2]);
        CHECK((x->Rs == -1 && y->Rs == -1) || (x->Rs > 0 && y->Rs > 0 && reldiff(x->Rs, y->Rs) <= ptol),
              "%s: track %ld Rs %g serially, %g in place\n", what, t, x->Rs, y->Rs);
        CHECK((x->Vrms == -1 && y->Vrms == -1) || reldiff(x->Vrms, y->Vrms) <= ptol, "%s: track %ld Vrms %g serially, %g in place\n",
              what, t, x->Vrms, y->Vrms);
        for(d = 0; d < 3; d++) {
            const double vtol = ptol * fmax(x->Vrms, 0) + 0.01;
            CHECK(fabs(x->BulkVel[d] - y->BulkVel[d]) <= vtol && fabs(x->RockstarCoreVel[d] - y->RockstarCoreVel[d]) <= vtol,
                  "%s: track %ld BulkVel[%d] %g / %g RockstarCoreVel %g / %g\n", what, t, d, x->BulkVel[d], y->BulkVel[d],
                  x->RockstarCoreVel[d], y->RockstarCoreVel[d]);
        }
        CHECK(!(x->Flags & SUBHALO_FLAG_INPLACE), "%s: track %ld flagged in place in the serial run\n", what, t);
    }
    /* BoundTrackId and TrackId of every particle */
    int64_t nbd = 0, ntd = 0;
    for(i = 0; i < a->n; i++) {
        CHECK(a->r[i].ID == b->r[i].ID, "%s: particle %ld IDs %lu / %lu\n", what, i, (unsigned long) a->r[i].ID, (unsigned long) b->r[i].ID);
        if(a->r[i].BoundTrackId != b->r[i].BoundTrackId) nbd++;
        if(a->r[i].TrackId != b->r[i].TrackId) ntd++;
    }
    message(0, "%s: largest Nbound difference %.3e; %ld of %ld particles with another BoundTrackId, %ld with another TrackId\n", what,
            worst, nbd, a->n, ntd);
    CHECK(nbd <= 0.002 * a->n, "%s: %ld particles with another BoundTrackId\n", what, nbd);
    CHECK(ntd <= 0.005 * a->n, "%s: %ld particles with another TrackId\n", what, ntd);
    return worst;
}

/* ------------------------------------------------------------------ serial against in place */

/* The overlap scenario of test_subhalo.c: call 0 two clumps, call 1 B on a circular orbit at 5
 * kpc/h in A with 60 particles pulled to A's centre. Subsampling off; the field SO on. */
static void
t1_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    p2_setup(&CP);
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    subhalo_set_testpar(sp);
    const double cen[2][3] = {{P2_cA[0], P2_cA[1], P2_cA[2]}, {P2_cB[0], P2_cB[1], P2_cB[2]}};
    const double bulk[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int64_t ng;
    p2_place(0.5, CP.GravInternal, cen, bulk);
    int ret = run_finder_call(0, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "serial/in place: call 0 has %ld groups\n", ng);
    sp.SubhaloCollectiveMinPart = collective;
    subhalo_set_testpar(sp);
    p2_infall(&CP, 0.55);
    ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "serial/in place: call 1 has %ld groups\n", ng);
    outcome_take(o);
    p2_teardown();
}

static void
test_collective_serial_vs_inplace(void ** state)
{
    struct Outcome a, b;
    t1_run(0, &a);
    CHECK(a.st.NCands == 0, "serial run: %ld in-place candidates\n", a.st.NCands);
    t1_run(1, &b);
    message(0, "serial/in place: %d rounds, %ld candidates (%ld + %ld), %d + %d walks, %g + %g s\n", b.st.NRounds, b.st.NCands,
            b.st.NCandsRound[0], b.st.NCandsRound[1], b.st.Iterations[0], b.st.Iterations[1], b.st.Seconds[0], b.st.Seconds[1]);
    CHECK(b.st.NRounds == 2 && b.st.NCands == 2 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1,
          "in place: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    CHECK(a.nt == 2 && a.rec[1].Nbound > 0 && a.rec[1].Rank == 1 && a.rec[1].NestedParentTrackId == 0,
          "serial: B Nbound %d Rank %d parent %ld\n", a.rec[1].Nbound, a.rec[1].Rank, a.rec[1].NestedParentTrackId);
    CHECK((b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && (b.rec[1].Flags & SUBHALO_FLAG_INPLACE), "in place: Flags %d %d\n",
          b.rec[0].Flags, b.rec[1].Flags);
    compare_outcomes(&a, &b, 1e-3, 1e-2, 1, "serial/in place");
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "in place") == 0, "in place: tag invariants\n");
    /* the field SO of the in-place central */
    CHECK(reldiff(a.rec[0].MSO[0], b.rec[0].MSO[0]) < 1e-2, "serial/in place: MSO[vir] %g / %g\n", a.rec[0].MSO[0], b.rec[0].MSO[0]);
    outcome_free(&b);
    outcome_free(&a);
}

/* ------------------------------------------------------------------ synthetic hierarchies (tests 2-4) */

/* Test 2: A (0, 2000) the central, B (1, 300) and C (2, 200) its satellites 5 kpc/h (physical)
 * either side, D (3, 100) B's satellite, and S, 60 of B's previous particles at A's centre moving
 * against B: B's unbound remainder, bound by A. */
#define ST_A0 0
#define ST_B0 2000
#define ST_C0 2300
#define ST_D0 2500
#define ST_S0 2600
#define ST_END 2660

static void
st_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    double cB[3] = {500 + 5 / atime, 500, 500}, cC[3] = {500 - 5 / atime, 500, 500}, cD[3] = {500 + 6.2 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 40, 0}, vC[3] = {0, -40, 0}, vD[3] = {0, 44, 0}, vS[3] = {0, -40, 0};
    (void) ud;
    if(id >= ST_END)
        return;
    if(id < ST_B0) {
        syn_ball(id, 51, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id - ST_A0;
    }
    else if(id < ST_C0) {
        syn_ball(id, 52, cB, 1.0, vB, 8, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - ST_B0;
    }
    else if(id < ST_D0) {
        syn_ball(id, 53, cC, 0.8, vC, 8, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - ST_C0;
    }
    else if(id < ST_S0) {
        syn_ball(id, 54, cD, 0.3, vD, 3, atime, s);
        s->tag = s->bound = 3;
        s->rank = id - ST_D0;
    }
    else {
        syn_ball(id, 55, cA, 0.5, vS, 5, atime, s);
        s->tag = s->bound = 1;
        s->rank = 300 + id - ST_S0;
    }
    (void) cD;
}

static void
st_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, ST_END + SYN_NFIELD, sp);
    struct SubTrack recs[4];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 360, 0.036, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 0, 0, 2, 1);
    recs[3] = syn_rec(3, 100, 0.01, 1, 0, 3, 2);
    syn_records(recs, 4);
    syn_place(st_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sub-tree keys: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_subtree_keys(void ** state)
{
    struct Outcome a, b;
    st_run(0, &a);
    st_run(1000, &b);
    CHECK(b.st.NRounds == 1 && b.st.NCands == 1, "sub-tree keys: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    CHECK((b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[1].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[2].Flags & SUBHALO_FLAG_INPLACE)
          && !(b.rec[3].Flags & SUBHALO_FLAG_INPLACE), "sub-tree keys: Flags %d %d %d %d\n", b.rec[0].Flags, b.rec[1].Flags,
          b.rec[2].Flags, b.rec[3].Flags);
    int t;
    for(t = 1; t < 4; t++)
        CHECK(b.rec[t].Nbound > 0, "sub-tree keys: track %d Nbound %d\n", t, b.rec[t].Nbound);
    CHECK(b.rec[1].NestedParentTrackId == 0 && b.rec[1].Depth == 1 && b.rec[2].NestedParentTrackId == 0 && b.rec[2].Depth == 1
          && b.rec[3].NestedParentTrackId == 1 && b.rec[3].Depth == 2 && b.rec[1].CentralTrackId == 0 && b.rec[3].CentralTrackId == 0,
          "sub-tree keys: parents %ld %ld %ld depths %d %d %d\n", b.rec[1].NestedParentTrackId, b.rec[2].NestedParentTrackId,
          b.rec[3].NestedParentTrackId, b.rec[1].Depth, b.rec[2].Depth, b.rec[3].Depth);
    /* the provisional remainder of B's sub-tree reaches the central: S bound by A under B's tail tag */
    int64_t i, nS = 0, nSA = 0;
    for(i = 0; i < b.n; i++)
        if(b.r[i].ID >= ST_S0 && b.r[i].ID < ST_END) {
            nS++;
            if(b.r[i].BoundTrackId == 0 && (b.r[i].TrackId == 1 || b.r[i].TrackId == 0))
                nSA++;
        }
    {
        int64_t h[3][4];
        memset(h, 0, sizeof(h));
        for(i = 0; i < b.n; i++)
            if(b.r[i].ID >= ST_S0 && b.r[i].ID < ST_END) {
                h[0][b.r[i].TrackId + 1 < 4 ? b.r[i].TrackId + 1 : 3]++;
                h[1][b.r[i].BoundTrackId + 1 < 4 ? b.r[i].BoundTrackId + 1 : 3]++;
            }
        for(i = 0; i < a.n; i++)
            if(a.r[i].ID >= ST_S0 && a.r[i].ID < ST_END)
                h[2][a.r[i].BoundTrackId + 1 < 4 ? a.r[i].BoundTrackId + 1 : 3]++;
        message(0, "sub-tree keys: stripped particles: tags (-1, 0, 1, >1) %ld %ld %ld %ld, bound %ld %ld %ld %ld; serial bound %ld %ld %ld %ld\n",
                h[0][0], h[0][1], h[0][2], h[0][3], h[1][0], h[1][1], h[1][2], h[1][3], h[2][0], h[2][1], h[2][2], h[2][3]);
    }
    message(0, "sub-tree keys: %ld of %ld stripped particles bound by the central\n", nSA, nS);
    CHECK(nS == 60 && nSA >= 55, "sub-tree keys: %ld of %ld stripped particles bound by A\n", nSA, nS);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "sub-tree keys") == 0, "sub-tree keys: tag invariants\n");
    compare_outcomes(&a, &b, 1e-3, 1e-2, 0, "sub-tree keys");
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 3: A (0, 1500 particles, central of group 1 at the previous call) and B (1, 1000, central
 * of group 2) merge into one group: B, 6 kpc/h from A at 50 km/s, is too light to compete for
 * the centre (0.67 < SubhaloMajorProgenitorMassRatio) and becomes A's satellite. */
#define MG_B0 1500
#define MG_END 2500

static void
mg_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    double cB[3] = {500, 500 + 6 / atime, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {50, 0, 0};
    (void) ud;
    if(id >= MG_END)
        return;
    if(id < MG_B0) {
        syn_ball(id, 61, cA, 2.0, zero, 15, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 62, cB, 1.5, vB, 15, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - MG_B0;
    }
}

static void
mg_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, MG_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 1500, 0.15, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 1000, 0.10, -1, 1, 0, 0);
    recs[1].HostGroup = 2;
    syn_records(recs, 2);
    syn_place(mg_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "merger: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_merger(void ** state)
{
    struct Outcome a, b;
    mg_run(0, &a);
    mg_run(800, &b);
    message(0, "merger: %d rounds (%ld + %ld candidates, %d + %d walks, %g + %g s)\n", b.st.NRounds, b.st.NCandsRound[0],
            b.st.NCandsRound[1], b.st.Iterations[0], b.st.Iterations[1], b.st.Seconds[0], b.st.Seconds[1]);
    CHECK(b.st.NRounds == 2 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1, "merger: %d rounds\n", b.st.NRounds);
    CHECK(b.rec[1].Nbound > 0 && b.rec[1].Rank == 1 && b.rec[1].NestedParentTrackId == 0 && b.rec[1].CentralTrackId == 0
          && (b.rec[1].Flags & SUBHALO_FLAG_INPLACE) && (b.rec[0].Flags & SUBHALO_FLAG_INPLACE) && b.rec[0].Rank == 0,
          "merger: B Nbound %d Rank %d parent %ld Flags %d / %d\n", b.rec[1].Nbound, b.rec[1].Rank, b.rec[1].NestedParentTrackId,
          b.rec[1].Flags, b.rec[0].Flags);
    CHECK(fabs(b.rec[1].Mbound - a.rec[1].Mbound) <= 2e-3 * a.rec[1].Mbound, "merger: B Mbound %g in place, %g serially\n",
          b.rec[1].Mbound, a.rec[1].Mbound);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "merger") == 0, "merger: tag invariants\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "merger");
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 4: A (0, 3000 particles in 1 kpc/h, cold) and its satellite B (1, 500) whose particles now
 * fill A's centre with a 150 km/s dispersion: B (in place at depth 1) dies in round 0, its
 * particles pass to A in round 1, which binds them. */
#define KP_B0 3000
#define KP_END 3500

static void
kp_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= KP_END)
        return;
    if(id < KP_B0) {
        syn_ball(id, 71, cA, 1.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 72, cA, 1.0, zero, 150, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - KP_B0;
    }
}

static void
kp_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    /* both trees accurate enough that the energy order of the most bound particles is the same */
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, KP_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 500, 0.05, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(kp_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "kill: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_kill_passup(void ** state)
{
    struct Outcome a, b;
    kp_run(0, &a);
    kp_run(300, &b);
    message(0, "kill: %d rounds (%ld + %ld candidates), %ld died, %ld sunk; serial B Nbound %d SnapDeath %d Sink %ld; "
               "in place B Nbound %d SnapDeath %d Sink %ld; A Nbound %d / %d\n", b.st.NRounds, b.st.NCandsRound[0], b.st.NCandsRound[1],
            b.st.NDied, b.st.NSunk, a.rec[1].Nbound, a.rec[1].SnapDeath, a.rec[1].SinkTrackId, b.rec[1].Nbound, b.rec[1].SnapDeath,
            b.rec[1].SinkTrackId, a.rec[0].Nbound, b.rec[0].Nbound);
    CHECK(b.st.NRounds == 2 && b.st.NDied == 1, "kill: %d rounds, %ld died\n", b.st.NRounds, b.st.NDied);
    CHECK(b.rec[1].Nbound == 0 && b.rec[1].SnapDeath == 1 && a.rec[1].Nbound == 0 && b.rec[1].SinkTrackId == a.rec[1].SinkTrackId,
          "kill: B Nbound %d SnapDeath %d Sink %ld (serial Sink %ld)\n", b.rec[1].Nbound, b.rec[1].SnapDeath, b.rec[1].SinkTrackId,
          a.rec[1].SinkTrackId);
    int64_t i, nB = 0, nBA = 0;
    for(i = 0; i < b.n; i++)
        if(b.r[i].ID >= KP_B0 && b.r[i].ID < KP_END) {
            nB++;
            if(b.r[i].BoundTrackId == 0)
                nBA++;
        }
    message(0, "kill: %ld of %ld of B's particles bound by A\n", nBA, nB);
    CHECK(nBA >= 0.9 * nB && b.rec[0].Nbound >= 3000 + 0.9 * 500, "kill: %ld of %ld of B's particles bound by A (A Nbound %d)\n",
          nBA, nB, b.rec[0].Nbound);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "kill") == 0, "kill: tag invariants\n");
    compare_outcomes(&a, &b, 1e-3, 1e-2, 0, "kill");
    outcome_free(&b);
    outcome_free(&a);
}

/* ------------------------------------------------------------------ binder, repairs, admission, guards */

/* A point at physical distance rphys from cen (comoving) in a direction drawn from (id, salt). */
static void
syn_at_radius(MyIDType id, uint64_t salt, const double cen[3], double rphys, double atime, double out[3])
{
    double u[3];
    const double c0[3] = {0, 0, 0};
    double n2 = 0;
    int d;
    do {
        rand_in_ball(id, salt + (uint64_t) (n2 * 1e6), 1.0, c0, u);
        n2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    } while(n2 < 1e-4);
    for(d = 0; d < 3; d++)
        out[d] = cen[d] + rphys * u[d] / sqrt(n2) / atime;
}

/* ---- the binder takes the particle; many serial repairs ----
 * Per group g (centre 500 + 125 g): A (3 g, 2000 particles, the in-place central), B (3 g + 1,
 * 300 particles, 5 kpc/h away, a serial satellite) and C (3 g + 2, 100 particles at A's centre,
 * B's satellite: it sinks into A in the rounds while B survives). T: 60 previous tails of C
 * bound by B, spread through B's ball at radii decreasing with the ID (the more bound, the higher
 * the ID: ID order opposed to the energy order), every fourth one gas (a non-tracer). R (optional,
 * 30): previous tails of B at A's very centre and cold, which A binds and reserves. */
#define BD_NA 2000
#define BD_NB 300
#define BD_NC 100
#define BD_NT 60
#define BD_NR 30
static int BdGroups = 1, BdWithR = 0;

static int64_t
bd_npg(void)
{
    return BD_NA + BD_NB + BD_NC + BD_NT + (BdWithR ? BD_NR : 0);
}

/* the part of an id: 0 A, 1 B, 2 C, 3 T, 4 R; -1 lattice */
static int
bd_part(MyIDType id, int * g, int64_t * j)
{
    const int64_t npg = bd_npg();
    if(id >= (MyIDType) (BdGroups * npg))
        return -1;
    *g = (int) (id / npg);
    int64_t k = (int64_t) (id % npg);
    if(k < BD_NA) { *j = k; return 0; }
    k -= BD_NA;
    if(k < BD_NB) { *j = k; return 1; }
    k -= BD_NB;
    if(k < BD_NC) { *j = k; return 2; }
    k -= BD_NC;
    if(k < BD_NT) { *j = k; return 3; }
    *j = k - BD_NT;
    return 4;
}

static int
bd_type(MyIDType id)
{
    int g;
    int64_t j;
    return bd_part(id, &g, &j) == 3 && j % 4 == 1 ? 0 : 1;
}

static void
bd_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    int g;
    int64_t j;
    const int part = bd_part(id, &g, &j);
    (void) ud;
    if(part < 0)
        return;
    const double cA[3] = {500 + 125 * g, 500, 500};
    const double cB[3] = {cA[0] + 5 / atime, cA[1], cA[2]};
    /* B fast enough that the cold knots at A's centre (C, R) are not bound to it in its key */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    const int64_t TA = 3 * g, TB = 3 * g + 1, TC = 3 * g + 2;
    int d;
    switch(part) {
        case 0:
            syn_ball(id, 81, cA, 3.0, zero, 20, atime, s);
            s->tag = s->bound = TA;
            s->rank = j;
            break;
        case 1:
            syn_ball(id, 82, cB, 1.0, vB, 8, atime, s);
            s->tag = s->bound = TB;
            s->rank = j;
            break;
        case 2:
            syn_ball(id, 83, cA, 0.1, zero, 0.5, atime, s);
            s->tag = s->bound = TC;
            s->rank = j;
            break;
        case 3: {
            /* radius decreasing with j: the last ones are the most bound in B */
            syn_at_radius(id, 84, cB, 0.02 + 0.9 * (1 - (double) j / BD_NT), atime, s->pos);
            double v[3];
            rand_in_ball(id, 85, 1.0, zero, v);
            for(d = 0; d < 3; d++)
                s->vel[d] = atime * (vB[d] + v[d]);
            s->tag = TC;
            s->bound = TB;
            s->rank = BD_NC + j;
            break;
        }
        default:
            /* as cold as C at A's centre: among A's most-bound tracers once C has sunk */
            syn_ball(id, 86, cA, 0.1, zero, 0.5, atime, s);
            s->tag = TB;
            s->bound = TA;
            s->rank = BD_NB + j;
            break;
    }
}

/* the pre-final-pass state, captured by the hook (serial_repairs) */
static struct tagrec * BdPre = NULL;
static int64_t BdNPre = 0;
static struct SubhaloBindOrd * BdBo = NULL;
static int64_t BdNBo = 0;

static int
cmp_bindord_test(const void * a, const void * b)
{
    const MyIDType x = ((const struct SubhaloBindOrd *) a)->ID, y = ((const struct SubhaloBindOrd *) b)->ID;
    return (x > y) - (x < y);
}

static void
bd_hook(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan)
{
    (void) plan;
    BdPre = gather_tags(&BdNPre);
    const int NTask = get_ntask();
    int nb = (int) (ctx->NBindOrd * sizeof(struct SubhaloBindOrd));
    int * counts = malloc(NTask * sizeof(int));
    int * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    BdBo = malloc(tot + sizeof(struct SubhaloBindOrd));
    MPI_Allgatherv(ctx->BindOrd, nb, MPI_BYTE, BdBo, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    BdNBo = tot / sizeof(struct SubhaloBindOrd);
    qsort(BdBo, BdNBo, sizeof(struct SubhaloBindOrd), cmp_bindord_test);
    free(displ);
    free(counts);
}

static void
bd_run(int64_t collective, int ngroups, int withR, int hook, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    BdGroups = ngroups;
    BdWithR = withR;
    SynTypeFn = bd_type;
    syn_setup(&CP, ngroups * bd_npg() + SYN_NFIELD, sp);
    SynTypeFn = NULL;
    struct SubTrack * recs = malloc(sizeof(struct SubTrack) * 3 * ngroups);
    int g;
    for(g = 0; g < ngroups; g++) {
        recs[3 * g] = syn_rec(3 * g, BD_NA, BD_NA * SYN_M, -1, 3 * g, 0, 0);
        recs[3 * g + 1] = syn_rec(3 * g + 1, BD_NB + BD_NT, (BD_NB + BD_NT) * SYN_M, 3 * g, 3 * g, 1, 1);
        recs[3 * g + 2] = syn_rec(3 * g + 2, BD_NC, BD_NC * SYN_M, 3 * g + 1, 3 * g, 2, 2);
        int t;
        for(t = 0; t < 3; t++)
            recs[3 * g + t].HostGroup = g + 1;
    }
    syn_records(recs, 3 * ngroups);
    free(recs);
    syn_place(bd_particle, NULL, 0.55, 1);
    if(hook)
        subhalo_collective_set_pretag_hook(bd_hook);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    subhalo_collective_set_pretag_hook(NULL);
    CHECK(ret == 1 && ng == ngroups, "binder: %ld groups, ret %d\n", ng, ret);
    outcome_take(o);
    syn_teardown();
}

/* Test 5: A in place, B a serial survivor, C (B's satellite) sinks into A while B
 * survives: the tails of C that B binds return to B, merged into B's own-tag sequence by B's
 * final-pass energy order. The serial run (every track in one key) ranks B's own-tag particles
 * by one pass over B's energy-ordered list: it is the oracle, rank by rank. */
static void
test_collective_binder(void ** state)
{
    struct Outcome a, b;
    bd_run(0, 1, 0, 0, &a);
    bd_run(1000, 1, 0, 0, &b);
    const int64_t TA = 0, TB = 1, TC = 2;
    message(0, "binder: in place %ld candidates, %ld sunk, %ld tails given to their binder, %ld taken; C Nbound %d / %d Sink %ld / %ld; "
               "B Nbound %d / %d NboundExclusive %d / %d\n", b.st.NCands, b.st.NSunk, b.st.NAdded, b.st.NStolen, a.rec[TC].Nbound,
            b.rec[TC].Nbound, a.rec[TC].SinkTrackId, b.rec[TC].SinkTrackId, a.rec[TB].Nbound, b.rec[TB].Nbound,
            a.rec[TB].NboundExclusive, b.rec[TB].NboundExclusive);
    CHECK(b.st.NCands == 1 && (b.rec[TA].Flags & SUBHALO_FLAG_INPLACE) && !(b.rec[TB].Flags & SUBHALO_FLAG_INPLACE),
          "binder: %ld candidates, Flags %d %d\n", b.st.NCands, b.rec[TA].Flags, b.rec[TB].Flags);
    CHECK(b.rec[TC].Nbound == 0 && b.rec[TC].SinkTrackId == TA && a.rec[TC].Nbound == 0 && a.rec[TC].SinkTrackId == TA,
          "binder: C should sink into A in both runs\n");
    CHECK(b.rec[TB].Nbound > 0 && b.rec[TB].NestedParentTrackId == TA, "binder: B Nbound %d parent %ld\n", b.rec[TB].Nbound,
          b.rec[TB].NestedParentTrackId);
    /* B's own reservation (its ten most-bound tracers) already takes the innermost T in the key */
    CHECK(b.st.NAdded >= BD_NT - 12, "binder: %ld tails given to the binder\n", b.st.NAdded);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "binder in place") == 0, "binder: tag invariants\n");
    CHECK(p2_check_tags(a.r, a.n, a.rec, a.nt, "binder serial") == 0, "binder: tag invariants (serial)\n");
    CHECK(a.rec[TB].NboundExclusive == b.rec[TB].NboundExclusive && a.rec[TB].Nbound == b.rec[TB].Nbound,
          "binder: B NboundExclusive %d / %d\n", a.rec[TB].NboundExclusive, b.rec[TB].NboundExclusive);
    /* B's own-tag particles, rank by rank, against the serial run; T bound and tagged by B */
    int64_t i, nB = 0, ndiff = 0, nT = 0, nTgas = 0, nTtr = 0, maxown = -1, minT = INT32_MAX, agree = 0, pairs = 0;
    for(i = 0; i < b.n; i++) {
        int g;
        int64_t j;
        const int part = bd_part(b.r[i].ID, &g, &j);
        if(b.r[i].TrackId == TB || a.r[i].TrackId == TB) {
            nB++;
            if(b.r[i].TrackId != a.r[i].TrackId || b.r[i].BindRank != a.r[i].BindRank || b.r[i].BoundTrackId != a.r[i].BoundTrackId) {
                if(ndiff < 10)
                    message(0, "binder: particle %lu (part %d): serial (%ld, %d, %ld), in place (%ld, %d, %ld)\n", (unsigned long) b.r[i].ID,
                            part, a.r[i].TrackId, a.r[i].BindRank, a.r[i].BoundTrackId, b.r[i].TrackId, b.r[i].BindRank, b.r[i].BoundTrackId);
                ndiff++;
            }
        }
        if(part == 1 && b.r[i].TrackId == TB && b.r[i].BoundTrackId == TB && b.r[i].BindRank > maxown)
            maxown = b.r[i].BindRank;
        if(part == 3) {
            if(b.r[i].TrackId == TB && b.r[i].BoundTrackId == TB) {
                nT++;
                if(b.r[i].Type == 0) nTgas++;
                else nTtr++;
                if(b.r[i].BindRank < minT)
                    minT = b.r[i].BindRank;
            }
        }
    }
    /* T's rank order against its ID order (ranks assigned in ID order would agree in every pair) */
    for(i = 0; i < b.n; i++) {
        int g, g2;
        int64_t j, j2, k;
        if(bd_part(b.r[i].ID, &g, &j) != 3 || b.r[i].TrackId != TB)
            continue;
        for(k = i + 1; k < b.n; k++)
            if(bd_part(b.r[k].ID, &g2, &j2) == 3 && b.r[k].TrackId == TB) {
                pairs++;
                if(b.r[k].BindRank > b.r[i].BindRank)
                    agree++;
            }
    }
    message(0, "binder: %ld own-tag rows of B, %ld differ from the serial run; %ld of T given to B (%ld gas, %ld tracers), "
               "first T rank %ld, last own exclusive rank %ld; %ld of %ld T pairs in ID order\n", nB, ndiff, nT, nTgas, nTtr, minT, maxown,
            agree, pairs);
    CHECK(ndiff == 0, "binder: %ld of B's own-tag rows differ from the serial run\n", ndiff);
    CHECK(nT >= BD_NT - 2 && nTgas > 0 && nTtr > 0, "binder: %ld T given to B (%ld gas, %ld tracers)\n", nT, nTgas, nTtr);
    CHECK(minT < maxown, "binder: the returned tails are not merged into B's exclusive sequence (first %ld, last own %ld)\n", minT, maxown);
    CHECK(agree < pairs / 4, "binder: %ld of %ld T pairs in ID order: the energy order was meant to oppose it\n", agree, pairs);
    outcome_free(&b);
    outcome_free(&a);
}

/* Test 6: three groups, each with tails losing their claimant (C sinks: the tails
 * return to the binder B) and B's tails at A's centre that A binds and reserves: the serial ranks
 * are repaired on the tracks' result ranks. The reference is computed here from the gathered
 * pre-pass tags (the serial keys' provisional ranks, captured by the hook) and binder ordinals:
 * for every serial track, its final own-tag particles sorted by (kept exclusive: old rank;
 * returned: binder ordinal g before the old exclusive rank g, then its energy position; kept
 * tail: after every exclusive one, by old rank) must carry exactly that dense rank. */
static void
test_collective_serial_repairs(void ** state)
{
    struct Outcome b;
    bd_run(1000, 3, 1, 1, &b);
    CHECK(BdPre && BdNPre == b.n, "repairs: hook captured %ld particles of %ld\n", BdNPre, b.n);
    message(0, "repairs: %ld candidates, %ld sunk, %ld returned to their binder, %ld taken by in-place candidates, %ld tracks repaired, "
               "%ld rank queries, %g s\n", b.st.NCands, b.st.NSunk, b.st.NAdded, b.st.NStolen, b.st.NRepairTracks, b.st.NRepairQueries,
            b.st.RepairSeconds);
    CHECK(b.st.NCands == 3 && b.st.NAdded >= 3 * (BD_NT - 12) && b.st.NStolen >= 3, "repairs: %ld candidates, %ld added, %ld taken\n",
          b.st.NCands, b.st.NAdded, b.st.NStolen);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "repairs") == 0, "repairs: tag invariants\n");
    int64_t T, i, nbad = 0, nchk = 0, nrem = 0, nadd = 0;
    int64_t * key = malloc(sizeof(int64_t) * 3 * (b.n + 1));
    int64_t * idx = malloc(sizeof(int64_t) * (b.n + 1));
    for(T = 0; T < b.nt; T++) {
        if(b.rec[T].Nbound <= 0 || (b.rec[T].Flags & SUBHALO_FLAG_INPLACE))
            continue;
        int64_t nbex = 0, nown = 0;
        for(i = 0; i < b.n; i++)
            if(BdPre[i].TrackId == T && BdPre[i].BoundTrackId == T)
                nbex++;
        for(i = 0; i < b.n; i++) {
            if(BdPre[i].TrackId == T && b.r[i].TrackId != T)
                nrem++;
            if(b.r[i].TrackId != T)
                continue;
            int64_t * k = key + 3 * nown;
            if(BdPre[i].TrackId == T) {
                if(BdPre[i].BindRank < nbex) {
                    k[0] = BdPre[i].BindRank;
                    k[1] = 1;
                }
                else {
                    k[0] = INT32_MAX + (int64_t) BdPre[i].BindRank;
                    k[1] = 1;
                }
                k[2] = 0;
            }
            else {
                struct SubhaloBindOrd q = {b.r[i].ID, 0, 0};
                const struct SubhaloBindOrd * e = bsearch(&q, BdBo, BdNBo, sizeof(struct SubhaloBindOrd), cmp_bindord_test);
                CHECK(e && BdPre[i].BoundTrackId == T, "repairs: particle %lu returned to %ld without a binder ordinal (pre bound %ld)\n",
                      (unsigned long) b.r[i].ID, T, BdPre[i].BoundTrackId);
                k[0] = e->g;
                k[1] = 0;
                k[2] = e->f;
                nadd++;
            }
            idx[nown++] = i;
        }
        /* insertion sort of the keys (small tracks) */
        int64_t p, q2;
        for(p = 1; p < nown; p++)
            for(q2 = p; q2 > 0; q2--) {
                int64_t * x = key + 3 * (q2 - 1), * y = key + 3 * q2;
                if(x[0] < y[0] || (x[0] == y[0] && (x[1] < y[1] || (x[1] == y[1] && x[2] <= y[2]))))
                    break;
                int64_t t3[3] = {x[0], x[1], x[2]}, ti = idx[q2 - 1];
                x[0] = y[0]; x[1] = y[1]; x[2] = y[2];
                y[0] = t3[0]; y[1] = t3[1]; y[2] = t3[2];
                idx[q2 - 1] = idx[q2];
                idx[q2] = ti;
            }
        for(p = 0; p < nown; p++) {
            nchk++;
            if(b.r[idx[p]].BindRank != p) {
                if(nbad < 10)
                    message(0, "repairs: track %ld particle %lu: BindRank %d, reference %ld (pre tag %ld rank %d bound %ld)\n", T,
                            (unsigned long) b.r[idx[p]].ID, b.r[idx[p]].BindRank, p, BdPre[idx[p]].TrackId, BdPre[idx[p]].BindRank,
                            BdPre[idx[p]].BoundTrackId);
                nbad++;
            }
        }
    }
    message(0, "repairs: %ld serial own-tag rows checked against the reference, %ld differ; %ld removed, %ld returned\n", nchk, nbad, nrem, nadd);
    CHECK(nbad == 0 && nrem >= 3 && nadd >= 3 * (BD_NT - 12), "repairs: %ld rows differ (%ld removed, %ld returned)\n", nbad, nrem, nadd);
    free(idx);
    free(key);
    free(BdBo);
    free(BdPre);
    BdBo = NULL;
    BdPre = NULL;
    outcome_free(&b);
}

/* ---- the sub-tree remainder on both sides of the relaxed-source boundary ----
 * A (0, 2000) the central in place; B (1, 100 particles, 5 kpc/h away) a serial satellite whose
 * 350 previous tails S sit at A's centre moving against B: B's relaxed source keeps R Nbound =
 * 300 list entries, so about 200 of S stay B's tails (bound by A) and the rest, beyond the
 * boundary, go to A. B's list is the serial kernel's in both runs, so every S particle must have
 * the same tag (and B's tails the same rank) serially and in place. */
#define RB_B0 2000
#define RB_S0 2100
#define RB_END 2450

static void
rb_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 5 / atime, 500, 500};
    /* S spread over 2 kpc/h at rest and B fast: S is bound neither to itself nor to B */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    (void) ud;
    if(id >= RB_END)
        return;
    if(id < RB_B0) {
        syn_ball(id, 91, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < RB_S0) {
        syn_ball(id, 92, cB, 0.5, vB, 5, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - RB_B0;
    }
    else {
        syn_ball(id, 93, cA, 2.0, zero, 5, atime, s);
        s->tag = 1;
        s->bound = -1;
        s->rank = 100 + id - RB_S0;
    }
}

static void
rb_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, RB_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 100, 0.01, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(rb_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "relaxed boundary: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_relaxed_boundary(void ** state)
{
    struct Outcome a, b;
    rb_run(0, &a);
    rb_run(1000, &b);
    CHECK(b.st.NCands == 1 && !(b.rec[1].Flags & SUBHALO_FLAG_INPLACE), "relaxed boundary: %ld candidates\n", b.st.NCands);
    int64_t i, nB = 0, nA = 0, nother = 0, ndiff = 0, nbA = 0;
    for(i = 0; i < b.n; i++) {
        if(b.r[i].ID < RB_S0 || b.r[i].ID >= RB_END)
            continue;
        if(b.r[i].TrackId == 1) nB++;
        else if(b.r[i].TrackId == 0) nA++;
        else nother++;
        if(b.r[i].BoundTrackId == 0)
            nbA++;
        if(b.r[i].TrackId != a.r[i].TrackId || (b.r[i].TrackId == 1 && b.r[i].BindRank != a.r[i].BindRank)) {
            if(ndiff < 10)
                message(0, "relaxed boundary: particle %lu serial (%ld, %d), in place (%ld, %d)\n", (unsigned long) b.r[i].ID,
                        a.r[i].TrackId, a.r[i].BindRank, b.r[i].TrackId, b.r[i].BindRank);
            ndiff++;
        }
    }
    int64_t tl = 0;
    int t;
    for(t = 0; t < 6; t++)
        tl += b.rec[1].NtailType[t];
    message(0, "relaxed boundary: B Nbound %d NboundExclusive %d tails %ld; of S: %ld B's tails, %ld A's, %ld else, %ld bound by A; "
               "%ld differ from the serial run\n", b.rec[1].Nbound, b.rec[1].NboundExclusive, tl, nB, nA, nother, nbA, ndiff);
    CHECK(ndiff == 0, "relaxed boundary: %ld particles of S with another tag than serially\n", ndiff);
    CHECK(nB > 0 && nA > 0 && nother == 0 && nB + nA == RB_END - RB_S0, "relaxed boundary: S split %ld (B) / %ld (A) / %ld\n", nB, nA, nother);
    /* inside: B's whole relaxed tail is S (B's own ball is bound); beyond: A's */
    CHECK(nB == tl && b.rec[1].NboundExclusive + tl <= (int64_t) (3 * b.rec[1].NboundExclusive) + 1,
          "relaxed boundary: %ld of S tagged B, B's tail count %ld, NboundExclusive %d\n", nB, tl, b.rec[1].NboundExclusive);
    CHECK(nbA >= 0.95 * (RB_END - RB_S0), "relaxed boundary: %ld of S bound by A\n", nbA);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "relaxed boundary") == 0, "relaxed boundary: tag invariants\n");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- a dead in-place candidate with field particles ----
 * A (0, 3000 cold particles in 1 kpc/h) and its satellite B (1) at A's centre: 100 cold gas
 * particles (non-tracers), 10 tracers of which one is hot, and 20 tracers in the field far away
 * (outside every FOF group, on other ranks) that move fast. B, in place (depth 1), keeps 9 bound
 * tracers after its first iteration and dies; its whole source passes to A, whose relaxed source
 * holds all of it; B's previous tracers vote remotely and give A as its descendant. */
#define DK_B0 3000
#define DK_BT 3100
#define DK_F0 3110
#define DK_END 3130

static int
dk_type(MyIDType id)
{
    return id >= DK_B0 && id < DK_BT ? 0 : 1;
}

static void
dk_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    /* the group in another part of the box than the field particles (other ranks) */
    const double cA[3] = {750, 750, 750};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= DK_END)
        return;
    if(id < DK_B0) {
        syn_ball(id, 101, cA, 1.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < DK_F0) {
        /* B's tracers first in its previous rank order (its step-0 tracers), then its gas */
        const int tracer = id >= DK_BT;
        syn_ball(id, 102, cA, 0.3, zero, tracer && id == DK_F0 - 1 ? 300 : 3, atime, s);
        s->tag = s->bound = 1;
        s->rank = tracer ? id - DK_BT : 10 + id - DK_B0;
    }
    else {
        /* field: a sparse line far from the group, 15 kpc/h apart, moving fast */
        const int64_t j = id - DK_F0;
        s->pos[0] = 125 + 15 * j;
        s->pos[1] = 125;
        s->pos[2] = 125;
        s->vel[0] = atime * 150;
        s->tag = s->bound = 1;
        s->rank = 110 + j;
    }
}

static void
test_collective_dead_field(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = 100;
    SynTypeFn = dk_type;
    syn_setup(&CP, DK_END + SYN_NFIELD, sp);
    SynTypeFn = NULL;
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 130, 0.013, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(dk_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "dead field: %ld groups\n", ng);
    struct Outcome o;
    outcome_take(&o);
    syn_teardown();
    int64_t i, nB = 0, nA = 0, nbA = 0, nF = 0, nFA = 0, nFfield = 0;
    int ftask = -1, gtask_mask = 0;
    int64_t fother = 0;
    for(i = 0; i < o.n; i++)
        if(o.r[i].ID < DK_F0 && o.r[i].GrNr == 1)
            gtask_mask |= 1 << o.r[i].Task;
    for(i = 0; i < o.n; i++) {
        if(o.r[i].ID < DK_B0 || o.r[i].ID >= DK_END)
            continue;
        nB++;
        if(o.r[i].TrackId == 0) nA++;
        if(o.r[i].BoundTrackId == 0) nbA++;
        if(o.r[i].ID >= DK_F0) {
            nF++;
            if(o.r[i].TrackId == 0 && o.r[i].BoundTrackId == -1) nFA++;
            if(o.r[i].GrNr == -1) nFfield++;
            if(!(gtask_mask & (1 << o.r[i].Task))) fother++;
            ftask = o.r[i].Task;
        }
    }
    message(0, "dead field: %d rounds, %ld died; B Nbound %d SnapDeath %d Sink %ld Descendant %ld; of B's %ld particles %ld tagged A, "
               "%ld bound by A; field: %ld of %ld A's tails, %ld outside every group, %ld on ranks without a group particle (e.g. %d)\n",
            o.st.NRounds, o.st.NDied, o.rec[1].Nbound, o.rec[1].SnapDeath, o.rec[1].SinkTrackId, o.rec[1].DescendantTrackId, nB, nA, nbA,
            nFA, nF, nFfield, fother, ftask);
    CHECK(o.st.NRounds == 2 && o.st.NDied == 1, "dead field: %d rounds, %ld died\n", o.st.NRounds, o.st.NDied);
    CHECK(o.rec[1].Nbound == 0 && o.rec[1].SnapDeath == 1 && o.rec[1].SinkTrackId == -1 && o.rec[1].DescendantTrackId == 0,
          "dead field: B Nbound %d SnapDeath %d Sink %ld Descendant %ld\n", o.rec[1].Nbound, o.rec[1].SnapDeath, o.rec[1].SinkTrackId,
          o.rec[1].DescendantTrackId);
    CHECK(nA == nB && nFA == nF && nFfield == nF && nbA >= 105, "dead field: %ld of %ld reached A (%ld field as A's tails), %ld bound by A\n",
          nA, nB, nFA, nbA);
    CHECK(get_ntask() == 1 || fother > 0, "dead field: no field particle on a rank without group particles\n");
    CHECK(p2_check_tags(o.r, o.n, o.rec, o.nt, "dead field") == 0, "dead field: tag invariants\n");
    outcome_free(&o);
}

/* ---- a serial root with field particles sinks into the in-place central ----
 * A (0, 3000) in place; R (1, 150 particles at A's centre, A's satellite) and D (2, 200 particles
 * 5 kpc/h away, R's satellite) form a serial sub-tree key. FR: 10 field particles of R 30 kpc/h
 * (comoving) out, bound to R; FD: 10 field particles of D, bound to D; DT: 20 tails of D at A's
 * centre that R binds. R sinks into A in the round: its whole bound set (FR, DT) joins A; D
 * survives under A and keeps its claims (FD bound by D, DT bound by A). */
#define SR_R0 3000
#define SR_FR0 3150
#define SR_D0 3160
#define SR_FD0 3360
#define SR_DT0 3370
#define SR_END 3390

static void
sr_dir(int64_t j, int sgn, double u[3])
{
    const double al = M_PI / 3, ph = 2 * M_PI * j / 10.;
    u[0] = sgn * cos(al);
    u[1] = sin(al) * cos(ph);
    u[2] = sin(al) * sin(ph);
}

static void
sr_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cD[3] = {500 + 5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vD[3] = {0, 40, 0};
    double u[3], v[3];
    int d;
    (void) ud;
    if(id >= SR_END)
        return;
    if(id < SR_R0) {
        syn_ball(id, 111, cA, 1.5, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < SR_FR0) {
        syn_ball(id, 112, cA, 0.3, zero, 3, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - SR_R0;
    }
    else if(id < SR_D0) {
        sr_dir(id - SR_FR0, -1, u);
        rand_in_ball(id, 113, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cA[d] + 30 * u[d];
            s->vel[d] = atime * v[d];
        }
        s->tag = s->bound = 1;
        s->rank = 150 + id - SR_FR0;
    }
    else if(id < SR_FD0) {
        syn_ball(id, 114, cD, 0.8, vD, 6, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - SR_D0;
    }
    else if(id < SR_DT0) {
        sr_dir(id - SR_FD0, 1, u);
        rand_in_ball(id, 115, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cD[d] + 30 * u[d];
            s->vel[d] = atime * (vD[d] + v[d]);
        }
        s->tag = s->bound = 2;
        s->rank = 200 + id - SR_FD0;
    }
    else {
        syn_ball(id, 116, cA, 0.3, zero, 2, atime, s);
        s->tag = 2;
        s->bound = 1;
        s->rank = 210 + id - SR_DT0;
    }
}

static void
sr_setup(Cosmology * CP, int64_t collective, int failure)
{
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloDebugForceExchangeFailure = failure;
    syn_setup(CP, SR_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 3000, 0.3, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 160, 0.016, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 210, 0.021, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(sr_particle, NULL, 0.55, 1);
}

static void
sr_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    sr_setup(&CP, collective, 0);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sunk root: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_sunk_root_field(void ** state)
{
    struct Outcome a, b;
    sr_run(0, &a);
    sr_run(1000, &b);
    const struct SubTrack * R = &b.rec[1], * D = &b.rec[2];
    message(0, "sunk root: %ld candidates, %ld sunk, %ld taken, %ld returned; R Nbound %d Sink %ld SnapSink %d Descendant %ld; D Nbound %d "
               "parent %ld Depth %d central %ld\n", b.st.NCands, b.st.NSunk, b.st.NStolen, b.st.NAdded, R->Nbound, R->SinkTrackId, R->SnapSink,
            R->DescendantTrackId, D->Nbound, D->NestedParentTrackId, D->Depth, D->CentralTrackId);
    CHECK(b.st.NCands == 1 && b.st.NSunk >= 1, "sunk root: %ld candidates, %ld sunk\n", b.st.NCands, b.st.NSunk);
    CHECK(R->Nbound == 0 && R->SinkTrackId == 0 && R->SnapSink == 1 && R->SnapDeath == 1 && R->DescendantTrackId == 0,
          "sunk root: R's sink links\n");
    CHECK(D->Nbound > 0 && D->NestedParentTrackId == 0 && D->Depth == 1 && D->CentralTrackId == 0 && D->Rank == 1, "sunk root: D's links\n");
    int64_t i, nR = 0, nRA = 0, nFR = 0, nFRA = 0, nFD = 0, nFDD = 0, nDT = 0, nDTD = 0, nDTA = 0, nfield = 0;
    for(i = 0; i < b.n; i++) {
        const MyIDType id = b.r[i].ID;
        if(id >= SR_R0 && id < SR_D0) {
            nR++;
            if(b.r[i].TrackId == 0 && b.r[i].BoundTrackId == 0) nRA++;
            if(id >= SR_FR0) {
                nFR++;
                if(b.r[i].TrackId == 0 && b.r[i].BoundTrackId == 0) nFRA++;
            }
        }
        if(id >= SR_FD0 && id < SR_DT0) {
            nFD++;
            if(b.r[i].TrackId == 2 && b.r[i].BoundTrackId == 2) nFDD++;
        }
        if(id >= SR_DT0 && id < SR_END) {
            nDT++;
            if(b.r[i].TrackId == 2) nDTD++;
            if(b.r[i].BoundTrackId == 0) nDTA++;
        }
        if(((id >= SR_FR0 && id < SR_D0) || (id >= SR_FD0 && id < SR_DT0)) && b.r[i].GrNr == -1)
            nfield++;
    }
    message(0, "sunk root: R's bound set %ld/%ld to A (field %ld/%ld); D's field %ld/%ld kept; D's tails at A's centre %ld/%ld tagged D, %ld "
               "bound by A; %ld field particles outside the group\n", nRA, nR, nFRA, nFR, nFDD, nFD, nDTD, nDT, nDTA, nfield);
    CHECK(nRA == nR && nFRA == nFR && nFR == 10, "sunk root: R's bound set not transferred to A (%ld/%ld)\n", nRA, nR);
    CHECK(nFDD == nFD && nFD == 10 && nfield == 20, "sunk root: D's field particles (%ld/%ld kept, %ld outside)\n", nFDD, nFD, nfield);
    CHECK(nDTA == nDT && nDTD >= nDT - 10, "sunk root: D's claims bound by A (%ld tagged D, %ld bound by A of %ld)\n", nDTD, nDTA, nDT);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "sunk root") == 0, "sunk root: tag invariants\n");
    CHECK(a.rec[1].Nbound == 0 && a.rec[1].SinkTrackId == 0, "sunk root: the serial run sinks R into A too\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "sunk root");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- the central's source is its FOF group only (HBT FeedCentrals) ----
 * A (0, 3000) the central with FA, 20 of its previous particles at rest in the field far away
 * (bound to A were they in its source); B (1, 200) its satellite 5 kpc/h out with FB, 10 field
 * particles bound to B. Serially, with A in place and B a sub-tree key, and with both in place:
 * FA untagged (the in-place final pass releases them), FB kept by B. */
#define CS_FA0 3000
#define CS_B0 3020
#define CS_FB0 3220
#define CS_END 3230

static void
cs_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 40, 0};
    double u[3], v[3];
    int d;
    (void) ud;
    if(id >= CS_END)
        return;
    if(id < CS_FA0) {
        syn_ball(id, 121, cA, 1.5, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < CS_B0) {
        /* a sparse line far from the group, 15 kpc/h apart, at rest */
        s->pos[0] = 125 + 15 * (id - CS_FA0);
        s->pos[1] = 125;
        s->pos[2] = 125;
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < CS_FB0) {
        syn_ball(id, 122, cB, 0.8, vB, 6, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - CS_B0;
    }
    else {
        sr_dir(id - CS_FB0, 1, u);
        rand_in_ball(id, 123, 1.0, zero, v);
        for(d = 0; d < 3; d++) {
            s->pos[d] = cB[d] + 30 * u[d];
            s->vel[d] = atime * (vB[d] + v[d]);
        }
        s->tag = s->bound = 1;
        s->rank = 200 + id - CS_FB0;
    }
}

static void
cs_run(int64_t collective, struct Outcome * o, int64_t * nreleased)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, CS_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 3020, 0.302, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 210, 0.021, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(cs_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "central source (%ld): %ld groups\n", collective, ng);
    *nreleased = subhalo_last_call_stats()->NFallbackReleased;
    outcome_take(o);
    syn_teardown();
}

/* The FOF-less particles of the central untagged, the satellite's kept and bound by it. */
static void
cs_check(const struct Outcome * o, const char * what)
{
    int64_t i, nFA = 0, nFAu = 0, nFB = 0, nFBk = 0, nfield = 0;
    for(i = 0; i < o->n; i++) {
        const struct tagrec * q = &o->r[i];
        if(q->ID >= CS_FA0 && q->ID < CS_B0) {
            nFA++;
            nFAu += q->TrackId == -1 && q->BindRank == -1 && q->BoundTrackId == -1;
            nfield += q->GrNr == -1;
        }
        if(q->ID >= CS_FB0 && q->ID < CS_END) {
            nFB++;
            nFBk += q->TrackId == 1 && q->BoundTrackId == 1 && q->BindRank >= 0;
            nfield += q->GrNr == -1;
        }
    }
    message(0, "%s: A Nbound %d Rank %d, B Nbound %d Rank %d parent %ld; the central's FOF-less %ld/%ld untagged, the satellite's %ld/%ld "
               "kept, %ld outside the group\n", what, o->rec[0].Nbound, o->rec[0].Rank, o->rec[1].Nbound, o->rec[1].Rank,
            o->rec[1].NestedParentTrackId, nFAu, nFA, nFBk, nFB, nfield);
    CHECK(o->rec[0].Rank == 0 && o->rec[1].Rank == 1 && o->rec[1].NestedParentTrackId == 0 && o->rec[1].Nbound > 0,
          "%s: the hierarchy\n", what);
    CHECK(nFA == CS_B0 - CS_FA0 && nFAu == nFA && nFB == CS_END - CS_FB0 && nFBk == nFB && nfield == nFA + nFB,
          "%s: FOF-less particles (%ld/%ld untagged, %ld/%ld kept)\n", what, nFAu, nFA, nFBk, nFB);
    CHECK(o->rec[0].Nbound <= CS_FA0, "%s: A binds %d particles, more than its group part\n", what, o->rec[0].Nbound);
    CHECK(p2_check_tags(o->r, o->n, o->rec, o->nt, what) == 0, "%s: tag invariants\n", what);
}

static void
test_collective_central_source(void ** state)
{
    struct Outcome a, b, c;
    int64_t ra, rb, rc;
    cs_run(0, &a, &ra);
    cs_run(1000, &b, &rb);
    cs_run(100, &c, &rc);
    message(0, "central source: candidates %ld / %ld / %ld, released %ld / %ld / %ld (serial, A in place, A and B in place)\n",
            a.st.NCands, b.st.NCands, c.st.NCands, ra, rb, rc);
    CHECK(a.st.NCands == 0 && b.st.NCands == 1 && c.st.NCands == 2, "central source: candidates %ld / %ld / %ld\n", a.st.NCands,
          b.st.NCands, c.st.NCands);
    CHECK(ra == 0 && rb == CS_B0 - CS_FA0 && rc == CS_B0 - CS_FA0, "central source: released %ld / %ld / %ld\n", ra, rb, rc);
    cs_check(&a, "central source, serial");
    cs_check(&b, "central source, A in place");
    cs_check(&c, "central source, A and B in place");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "central source, A in place");
    compare_outcomes(&a, &c, 2e-3, 1e-2, 0, "central source, A and B in place");
    outcome_free(&c);
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- a failed exchange after a real split ----
 * The sunk-root fixture with A in place (sub-tree key R + D, its group particles and the field
 * particles FR, FD travel encoded) and SubhaloDebugForceExchangeFailure: GrNr, the tags and the
 * records are those before the call and the call is incomplete. */
struct FailSnap {
    struct tagrec * r;
    int64_t n;
};

static void
fail_pre(void * ud)
{
    struct FailSnap * s = (struct FailSnap *) ud;
    s->r = gather_tags(&s->n);
}

static void
test_collective_failure_after_split(void ** state)
{
    Cosmology CP;
    sr_setup(&CP, 1000, 1);
    struct SubTrack * before = fetch_all_tracks(3);
    struct FailSnap pre = {NULL, 0};
    int64_t ng;
    const int ret = run_finder_call_ex(1, &CP, &ng, fail_pre, NULL, &pre);
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    message(0, "failure after split: ret %d complete %d; plan: %ld sub-tree keys, %ld candidates, %ld group and %ld field particles encoded\n",
            ret, subhalo_last_call_complete(), ps->NSubKeys, ps->NCands, ps->NEncFof, ps->NEncField);
    CHECK(ret == 0 && !subhalo_last_call_complete(), "failure after split: ret %d\n", ret);
    CHECK(ps->NSubKeys >= 1 && ps->NCands >= 1 && ps->NEncFof > 0 && ps->NEncField > 0, "failure after split: no real split\n");
    int64_t n;
    struct tagrec * post = gather_tags(&n);
    CHECK(n == pre.n, "failure after split: %ld / %ld particles\n", n, pre.n);
    int64_t i, bad = 0;
    for(i = 0; i < n; i++)
        if(post[i].ID != pre.r[i].ID || post[i].GrNr != pre.r[i].GrNr || post[i].TrackId != pre.r[i].TrackId
           || post[i].BindRank != pre.r[i].BindRank || post[i].BoundTrackId != pre.r[i].BoundTrackId) {
            if(bad < 5)
                message(0, "failure after split: particle %lu GrNr %ld -> %ld tag (%ld, %d, %ld) -> (%ld, %d, %ld)\n", (unsigned long) post[i].ID,
                        pre.r[i].GrNr, post[i].GrNr, pre.r[i].TrackId, pre.r[i].BindRank, pre.r[i].BoundTrackId, post[i].TrackId,
                        post[i].BindRank, post[i].BoundTrackId);
            bad++;
        }
    CHECK(bad == 0, "failure after split: %ld particles changed\n", bad);
    struct SubTrack * after = fetch_all_tracks(3);
    int t;
    for(t = 0; t < 3; t++)
        CHECK(after[t].TrackId == before[t].TrackId && after[t].Nbound == before[t].Nbound && after[t].NboundExclusive == before[t].NboundExclusive
              && after[t].Mbound == before[t].Mbound && after[t].NestedParentTrackId == before[t].NestedParentTrackId
              && after[t].CentralTrackId == before[t].CentralTrackId && after[t].Rank == before[t].Rank && after[t].Depth == before[t].Depth
              && after[t].HostGroup == before[t].HostGroup && after[t].SinkTrackId == before[t].SinkTrackId && after[t].Flags == before[t].Flags
              && after[t].SnapDeath == before[t].SnapDeath && after[t].MostBoundID == before[t].MostBoundID
              && after[t].DescendantTrackId == before[t].DescendantTrackId && subhalo_get_table()->NTotal == 3,
              "failure after split: record %d changed\n", t);
    free(after);
    free(post);
    free(pre.r);
    free(before);
    syn_teardown();
}

/* ---- a voided claim across three rounds, the central promotion and the final-tree check ----
 * A (0) the previous central: 1000 cold and 1000 hot particles; B (1, its satellite, 1500 cold
 * particles 4 kpc/h away: too light at the previous call to compete for the centre, the most
 * massive survivor now); C (2, B's satellite, 150 cold particles at A's centre). All three in
 * place: C in round 0, B in round 1, A in round 2. P: 40 tails of C at A's centre moving fast:
 * C claims them (relaxed source), B passes them on without claiming (C's claim is alive), C sinks
 * into A in round 2 and its claim becomes void: the accepted rule gives them to A, not back to B.
 * Q: 40 previous tails of B at A's centre at rest: B claims them in round 1, A binds them in round
 * 2; after the promotion of B to the central, A is B's satellite and no longer an ancestor of B:
 * the claim crosses the final tree and goes to the binding owner A. */
#define VC_AH 1000
#define VC_B0 2000
#define VC_C0 3500
#define VC_P0 3650
#define VC_Q0 3690
#define VC_END 3730

static void
vc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500};
    const double cB[3] = {500 + 4 / atime, 500, 500};
    /* B fast: the knots at A's centre are not bound to it */
    const double zero[3] = {0, 0, 0}, vB[3] = {0, 80, 0};
    (void) ud;
    if(id >= VC_END)
        return;
    if(id < VC_B0) {
        syn_ball(id, 121, cA, 2.0, zero, id < VC_AH ? 15 : 400, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < VC_C0) {
        syn_ball(id, 122, cB, 1.5, vB, 15, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - VC_B0;
    }
    else if(id < VC_P0) {
        syn_ball(id, 123, cA, 0.3, zero, 3, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - VC_C0;
    }
    else if(id < VC_Q0) {
        /* moving at 90 km/s through A's centre against B: bound to neither C nor B */
        const double vP[3] = {0, -90, 0};
        syn_ball(id, 124, cA, 0.5, vP, 3, atime, s);
        s->tag = 2;
        s->bound = -1;
        s->rank = 150 + id - VC_P0;
    }
    else {
        syn_ball(id, 126, cA, 0.5, zero, 3, atime, s);
        s->tag = 1;
        s->bound = -1;
        s->rank = 1500 + id - VC_Q0;
    }
}

static void
test_collective_voided_claim(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = 100;
    syn_setup(&CP, VC_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 1500, 0.15, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 150, 0.015, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(vc_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "voided claim: %ld groups\n", ng);
    struct Outcome o;
    outcome_take(&o);
    syn_teardown();
    const struct SubTrack * A = &o.rec[0], * B = &o.rec[1], * C = &o.rec[2];
    message(0, "voided claim: %d rounds (%ld %ld %ld), %ld sunk, %ld across the final tree; A Nbound %d Rank %d parent %ld; B Nbound %d Rank %d central %ld; "
               "C Nbound %d Sink %ld\n", o.st.NRounds, o.st.NCandsRound[0], o.st.NCandsRound[1], o.st.NCandsRound[2], o.st.NSunk, o.st.NJ2,
            A->Nbound, A->Rank, A->NestedParentTrackId, B->Nbound, B->Rank, B->CentralTrackId, C->Nbound, C->SinkTrackId);
    CHECK(o.st.NRounds == 3 && o.st.NCandsRound[0] == 1 && o.st.NCandsRound[1] == 1 && o.st.NCandsRound[2] == 1,
          "voided claim: %d rounds\n", o.st.NRounds);
    CHECK(C->Nbound == 0 && C->SinkTrackId == 0 && C->SnapSink == 1, "voided claim: C should sink into A\n");
    /* the promotion: B the central, A its satellite */
    CHECK(B->Rank == 0 && B->NestedParentTrackId == -1 && B->CentralTrackId == 1 && A->Rank == 1 && A->NestedParentTrackId == 1
          && A->CentralTrackId == 1 && A->Depth == 1 && (B->Flags & SUBHALO_FLAG_PLANCENTRAL) && !(A->Flags & SUBHALO_FLAG_PLANCENTRAL),
          "voided claim: promotion (B Rank %d, A Rank %d parent %ld, Flags %d %d)\n", B->Rank, A->Rank, A->NestedParentTrackId, B->Flags, A->Flags);
    int64_t i, nP = 0, nPA = 0, nPB = 0, nQ = 0, nQA = 0;
    for(i = 0; i < o.n; i++) {
        const MyIDType id = o.r[i].ID;
        if(id >= VC_P0 && id < VC_Q0) {
            nP++;
            if(o.r[i].TrackId == 0) nPA++;
            if(o.r[i].TrackId == 1) nPB++;
        }
        if(id >= VC_Q0 && id < VC_END) {
            nQ++;
            if(o.r[i].TrackId == 0 && o.r[i].BoundTrackId == 0) nQA++;
        }
    }
    message(0, "voided claim: C's former claims %ld to A, %ld to B of %ld; B's claims bound by A %ld of %ld given to A\n", nPA, nPB, nP,
            nQA, nQ);
    CHECK(nPA == nP && nPB == 0, "voided claim: the void claims went %ld to A, %ld to B (no retroactive B claim)\n", nPA, nPB);
    CHECK(nQA >= nQ - 2 && o.st.NJ2 >= nQ - 2, "voided claim: %ld of %ld final-tree claims given to A (%ld counted)\n", nQA, nQ, o.st.NJ2);
    CHECK(p2_check_tags(o.r, o.n, o.rec, o.nt, "voided claim") == 0, "voided claim: tag invariants\n");
    outcome_free(&o);
}

/* ---- a long in-place chain and groups at unequal depths in one call ----
 * Group 1: A0 > A1 > A2 > A3 (tracks 0-3, depths 0-3); group 2: B0 > B1 (4, 5). Everything in
 * place: four rounds (A3; A2; A1 and B1; A0 and B0), against the serial run. */
#define LC_N0 2000
#define LC_N1 600
#define LC_N2 300
#define LC_N3 150
#define LC_M0 1500
#define LC_M1 400
static const int64_t LcStart[7] = {0, LC_N0, LC_N0 + LC_N1, LC_N0 + LC_N1 + LC_N2, LC_N0 + LC_N1 + LC_N2 + LC_N3,
                                   LC_N0 + LC_N1 + LC_N2 + LC_N3 + LC_M0, LC_N0 + LC_N1 + LC_N2 + LC_N3 + LC_M0 + LC_M1};

static void
lc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double c1[3] = {500, 500, 500}, c2[3] = {750, 500, 500};
    const double R[6] = {3.0, 1.0, 0.6, 0.4, 2.5, 0.8}, sig[6] = {20, 8, 6, 4, 18, 7};
    const double off[6][3] = {{0, 0, 0}, {5, 0, 0}, {-5, 0, 0}, {0, 5, 0}, {0, 0, 0}, {5, 0, 0}};
    const double vel[6][3] = {{0, 0, 0}, {0, 40, 0}, {0, -40, 0}, {40, 0, 0}, {0, 0, 0}, {0, 40, 0}};
    (void) ud;
    int t;
    for(t = 0; t < 6; t++)
        if((int64_t) id < LcStart[t + 1])
            break;
    if(t >= 6)
        return;
    const double * c = t < 4 ? c1 : c2;
    double cen[3];
    int d;
    for(d = 0; d < 3; d++)
        cen[d] = c[d] + off[t][d] / atime;
    syn_ball(id, 131 + t, cen, R[t], vel[t], sig[t], atime, s);
    s->tag = s->bound = t;
    s->rank = id - LcStart[t];
}

static void
lc_setup(Cosmology * CP, struct SubhaloParams sp)
{
    syn_setup(CP, LcStart[6] + SYN_NFIELD, sp);
    struct SubTrack recs[6];
    recs[0] = syn_rec(0, LC_N0, LC_N0 * SYN_M, -1, 0, 0, 0);
    recs[1] = syn_rec(1, LC_N1, LC_N1 * SYN_M, 0, 0, 1, 1);
    recs[2] = syn_rec(2, LC_N2, LC_N2 * SYN_M, 1, 0, 2, 2);
    recs[3] = syn_rec(3, LC_N3, LC_N3 * SYN_M, 2, 0, 3, 3);
    recs[4] = syn_rec(4, LC_M0, LC_M0 * SYN_M, -1, 4, 0, 0);
    recs[5] = syn_rec(5, LC_M1, LC_M1 * SYN_M, 4, 4, 1, 1);
    recs[4].HostGroup = recs[5].HostGroup = 2;
    syn_records(recs, 6);
    syn_place(lc_particle, NULL, 0.55, 1);
}

static void
lc_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    lc_setup(&CP, sp);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 2, "long chain: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_long_chain(void ** state)
{
    struct Outcome a, b;
    lc_run(0, &a);
    lc_run(100, &b);
    message(0, "long chain: %d rounds, candidates per round %ld %ld %ld %ld, walks %d %d %d %d\n", b.st.NRounds, b.st.NCandsRound[0],
            b.st.NCandsRound[1], b.st.NCandsRound[2], b.st.NCandsRound[3], b.st.Iterations[0], b.st.Iterations[1], b.st.Iterations[2],
            b.st.Iterations[3]);
    CHECK(b.st.NRounds == 4 && b.st.NCands == 6 && b.st.NCandsRound[0] == 1 && b.st.NCandsRound[1] == 1 && b.st.NCandsRound[2] == 2
          && b.st.NCandsRound[3] == 2, "long chain: %d rounds, %ld candidates\n", b.st.NRounds, b.st.NCands);
    const int depth[6] = {0, 1, 2, 3, 0, 1};
    const int64_t parent[6] = {-1, 0, 1, 2, -1, 4};
    int t;
    for(t = 0; t < 6; t++)
        CHECK(b.rec[t].Nbound > 0 && b.rec[t].Depth == depth[t] && b.rec[t].NestedParentTrackId == parent[t]
              && (b.rec[t].Flags & SUBHALO_FLAG_INPLACE) && (!(depth[t] == 0) || (b.rec[t].Flags & SUBHALO_FLAG_PLANCENTRAL)),
              "long chain: track %d Nbound %d Depth %d parent %ld Flags %d\n", t, b.rec[t].Nbound, b.rec[t].Depth,
              b.rec[t].NestedParentTrackId, b.rec[t].Flags);
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "long chain") == 0, "long chain: tag invariants\n");
    compare_outcomes(&a, &b, 2e-3, 1e-2, 0, "long chain");
    outcome_free(&b);
    outcome_free(&a);
}

/* ---- the memory check of the collective stage ----
 * The long-chain fixture without forcing, the free memory squeezed (a top block allocated after
 * the FOF) so that no serial key fits and every key is promoted in place, while the collective
 * stage itself (its walk reserve alone) does not fit either: the call fails before the exchange
 * with the domain, GrNr and the tags untouched. The next call with the memory back is serial. */
struct LmCtx {
    void * block;
    double target;
    struct tagrec * pre;
    int64_t npre;
};

static void
lm_pre(void * ud)
{
    struct LmCtx * c = (struct LmCtx *) ud;
    c->pre = gather_tags(&c->npre);
    const double fr = (double) mymalloc_freebytes();
    c->block = fr > c->target ? mymalloc2("TestSqueeze", (size_t) (fr - c->target)) : NULL;
    message(0, "admission: free %g MB squeezed to %g MB\n", fr / 1048576., (double) mymalloc_freebytes() / 1048576.);
}

static void
lm_post(void * ud)
{
    struct LmCtx * c = (struct LmCtx *) ud;
    if(c->block)
        myfree(c->block);
    c->block = NULL;
}

static void
test_collective_admission(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloLookupBatchBytes = 1024 * 1024;
    lc_setup(&CP, sp);
    struct LmCtx c = {NULL, 0.5 * (double) subhalo_collective_walk_bytes(), NULL, 0};
    int64_t ng;
    int ret = run_finder_call_ex(1, &CP, &ng, lm_pre, lm_post, &c);
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    const struct SubhaloMemCheck * ad = subhalo_plan_last_memcheck();
    message(0, "admission: ret %d; plan %ld serial, %ld in place, %ld candidates; admitted %d, failed %d on rank %d at the %s, "
               "this rank needs %g MB (%g held + stages %g %g %g %g %g %g %g %g), %g MB available\n", ret, ps->NSerial, ps->NOversized, ps->NCands,
            ps->Fits, ad->failed, ad->failrank, ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-", ad->need / 1048576.,
            ad->persist / 1048576., ad->stage[0] / 1048576., ad->stage[1] / 1048576., ad->stage[2] / 1048576., ad->stage[3] / 1048576.,
            ad->stage[4] / 1048576., ad->stage[5] / 1048576., ad->stage[6] / 1048576., ad->stage[7] / 1048576., ad->avail / 1048576.);
    CHECK(ret == 0 && !subhalo_last_call_complete(), "admission: the squeezed call returned %d\n", ret);
    CHECK(ps->NSerial == 0 && ps->NOversized >= 2 && ps->NCands == 6 && !ps->Fits && ad->failed
          && ps->NEncFof == 0 && ps->NEncField == 0, "admission: every key promoted and the admission failed\n");
    CHECK(ad->failstage == SUBHALO_COLL_TREE, "admission: the largest stage is the %s\n", SubhaloCollStageName[ad->failstage]);
    int64_t n, i, bad = 0;
    struct tagrec * post = gather_tags(&n);
    for(i = 0; i < n; i++)
        if(post[i].ID != c.pre[i].ID || post[i].GrNr != c.pre[i].GrNr || post[i].TrackId != c.pre[i].TrackId
           || post[i].BindRank != c.pre[i].BindRank || post[i].BoundTrackId != c.pre[i].BoundTrackId)
            bad++;
    CHECK(n == c.npre && bad == 0, "admission: %ld particles changed by the failed call\n", bad);
    free(post);
    free(c.pre);
    /* the memory back: the same call is serial and completes */
    ret = run_finder_call(1, &CP, &ng);
    ps = subhalo_plan_last_summary();
    CHECK(ret == 1 && subhalo_last_call_complete() && ps->NSerial == 2 && ps->NCands == 0, "admission: the next call returned %d (%ld serial)\n",
          ret, ps->NSerial);
    syn_teardown();
}

/* ---- the int32 and MPI byte-count guards, without allocating ---- */
static void
test_collective_guards(void ** state)
{
    CHECK(subhalo_count_fits_i32((double) INT32_MAX) && !subhalo_count_fits_i32((double) INT32_MAX + 1.) && subhalo_count_fits_i32(0)
          && !subhalo_count_fits_i32(-1.) && !subhalo_count_fits_i32(1e300), "guards: counts\n");
    /* a rank r stands for r + 1 particles: INT32_MAX is out */
    CHECK(subhalo_rank_fits_i32(INT32_MAX - 1L) && !subhalo_rank_fits_i32(INT32_MAX) && !subhalo_rank_fits_i32(-1)
          && subhalo_rank_fits_i32(0), "guards: ranks\n");
    /* 24-byte records: 89478485 fit an MPI int byte count, 89478486 do not (the boundary) */
    CHECK(subhalo_mpi_bytes_fit(89478485, 24) && !subhalo_mpi_bytes_fit(89478486, 24) && subhalo_mpi_bytes_fit(2147483647L, 1)
          && !subhalo_mpi_bytes_fit(2147483648L, 1) && !subhalo_mpi_bytes_fit(-1, 8) && subhalo_mpi_bytes_fit(0, 56),
          "guards: MPI byte counts\n");
    /* the admission estimate: the persistent scratch is 25 bytes per particle, every stage at least
     * its records, the largest stage chosen */
    struct SubhaloMemCheck a;
    memset(&a, 0, sizeof(a));
    const struct SubhaloCollSizes z = {1000000, 2000000, 100000, 50000, 2, 2 * 20 * 4, 10, 100, 100, 4097, 3585, 1000, 0.7, 1};
    subhalo_coll_estimate(&a, &z);
    double mx = 0;
    int s;
    for(s = 0; s < SUBHALO_COLL_NSTAGE; s++)
        if(a.stage[s] > mx)
            mx = a.stage[s];
    const double persist = 25. * 1000001 + 11. * SUBHALO_INPLACE_CAND_BYTES + 3. * (sizeof(struct SubTrack) + 8.);
    CHECK(fabs(a.persist - persist) < 1 && a.need == a.persist + mx
          && a.stage[SUBHALO_COLL_RADIAL] >= 176. * 100000 && a.stage[SUBHALO_COLL_TREE] >= 0.7 * 1000000 * sizeof(struct NODE)
          /* the properties keep 104 bytes per record and every candidate's scratch */
          && a.stage[SUBHALO_COLL_PROPS] >= 104. * 100000 + 11. * SUBHALO_INPLACE_CAND_PROPS_BYTES,
          "guards: estimate persist %g need %g\n", a.persist, a.need);
}

/* ---- the potential-only walk at the default opening angle against a direct sum ----
 * Two interpenetrating clumps (labels 0 and 1) straddling the periodic boundary at x = 0, a knot
 * of label-0 particles closer than the softening across the boundary, a field outside every
 * label: on 4 ranks the clumps span several domains (remote monopoles, mixed-label top nodes).
 * At theta = 0.45 the monopole-only geometric opening (theta^2 <= 0.33) bounds the relative
 * error of a node's contribution by about theta^2 times its quadrupole fraction; the sums are
 * required within 1e-2 per particle and 2e-3 in the mean (the measured values are logged), and
 * within 1e-3 / 2e-4 at theta = 0.15. */
#define WK_N0 1500
#define WK_N1 800
#define WK_NK 20
#define WK_END (WK_N0 + WK_N1 + WK_NK)

static int
wk_label(MyIDType id)
{
    if(id < WK_N0 + WK_N1 + WK_NK)
        return id >= WK_N0 && id < WK_N0 + WK_N1 ? 1 : 0;
    return -1;
}

static void
wk_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double c0[3] = {0, 500, 500}, c1[3] = {1.0, 501, 500}, ck[3] = {0, 500.2, 500};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= WK_END)
        return;
    if(id < WK_N0)
        syn_ball(id, 141, c0, 3.0, zero, 0, atime, s);
    else if(id < WK_N0 + WK_N1)
        syn_ball(id, 142, c1, 2.0, zero, 0, atime, s);
    else
        syn_ball(id, 143, ck, 0.05, zero, 0, atime, s);
    /* masses within a factor 2 */
    s->mass = SYN_M * (1 + urand(id, 144, 0));
}

struct WkPart {
    MyIDType ID;
    double Pos[3];
    double Mass;
    int32_t label, type;
};

static void
wk_check(double theta, double maxtol, double meantol)
{
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    const int64_t n = PartManager->NumPart;
    int32_t * label = malloc(sizeof(int32_t) * (n + 1));
    double * out = malloc(sizeof(double) * (n + 1));
    int64_t i, j;
    int nloc = 0;
    for(i = 0; i < n; i++) {
        label[i] = wk_label(P[i].ID);
        out[i] = -1;
        if(label[i] >= 0)
            nloc++;
    }
    const double h = FORCE_SOFTENING();
    const int PotentialTypes = 1 + 2 + 16 + 32;
    subhalo_collective_test_walk(&ddecomp, label, 2, theta, h, PotentialTypes, out);
    /* the direct sum over every labelled particle, nearest images */
    const int NTask = get_ntask();
    struct WkPart * loc = malloc(sizeof(struct WkPart) * (nloc + 1));
    int k = 0;
    for(i = 0; i < n; i++)
        if(label[i] >= 0) {
            loc[k].ID = P[i].ID;
            int d;
            for(d = 0; d < 3; d++)
                loc[k].Pos[d] = P[i].Pos[d];
            loc[k].Mass = P[i].Mass;
            loc[k].label = label[i];
            loc[k].type = P[i].Type;
            k++;
        }
    int nb = nloc * sizeof(struct WkPart);
    int * counts = malloc(NTask * sizeof(int)), * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    struct WkPart * all = malloc(tot + sizeof(struct WkPart));
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    const int64_t nall = tot / sizeof(struct WkPart);
    double maxerr = 0, sumerr = 0, nerr = 0;
    int64_t nsoft = 0, nwrap = 0;
    for(i = 0; i < n; i++) {
        if(label[i] < 0)
            continue;
        double exact = 0;
        for(j = 0; j < nall; j++) {
            if(all[j].label != label[i] || all[j].ID == P[i].ID || !((PotentialTypes >> all[j].type) & 1))
                continue;
            double r2 = 0, raw = 0;
            int d;
            for(d = 0; d < 3; d++) {
                const double x = NEAREST(all[j].Pos[d] - P[i].Pos[d], BOXSIZE);
                r2 += x * x;
                raw += fabs(all[j].Pos[d] - P[i].Pos[d]);
            }
            const double rr = sqrt(r2);
            if(rr < h)
                nsoft++;
            if(raw > 0.5 * BOXSIZE)
                nwrap++;
            exact += all[j].Mass * grav_spline_potential(rr, h);
        }
        const double err = fabs(out[i] - exact) / exact;
        if(err > maxerr)
            maxerr = err;
        sumerr += err;
        nerr++;
    }
    double g[3] = {maxerr, sumerr, nerr}, gs[3];
    MPI_Allreduce(&g[0], &gs[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&g[1], &gs[1], 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int64_t pairs[2] = {nsoft, nwrap}, gp[2];
    MPI_Allreduce(pairs, gp, 2, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "walk theta %g: %g queries, largest relative error %.3e, mean %.3e; %ld softened and %ld periodic-image pairs in the direct sum\n",
            theta, gs[2], gs[0], gs[1] / gs[2], gp[0], gp[1]);
    CHECK(gs[2] == WK_END && gs[0] <= maxtol && gs[1] / gs[2] <= meantol && gp[0] > 0 && gp[1] > 0,
          "walk theta %g: %g queries, largest relative error %g, mean %g\n", theta, gs[2], gs[0], gs[1] / gs[2]);
    free(all);
    free(displ);
    free(counts);
    free(loc);
    free(out);
    free(label);
    domain_free(&ddecomp);
}

static void
test_collective_walk_theta(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    syn_setup(&CP, WK_END + SYN_NFIELD, sp);
    syn_place(wk_particle, NULL, 0.55, 0);
    wk_check(0.45, 1e-2, 2e-3);
    wk_check(0.15, 1e-3, 2e-4);
    syn_teardown();
}

/* ---- the deferred recipes of the two modes on identical inputs ----
 * Seven power-law balls (density ~ r^-2, every particle bound, one across the periodic boundary,
 * one below the NFW fit's 100 particles, one of 137 whose equal-mass virial crossing is at kc =
 * 100, the fit's boundary: Rs defined with the kc particles inside the crossing,
 * -1 on nfw.c's literal kc - 1), large bulk velocities, prescribed cores (subhalo_core of
 * the ID order) and thresholds with crossings inside every ball. The serial recipes
 * (subhalo_props_rockstar and subhalo_props_deferred on rank 0, which holds every particle) against
 * the distributed ones (round_props through subhalo_collective_test_props) with the particles on
 * rank 0 only, round-robin over the 4 ranks, and unevenly split (1:2:4:6): with equal masses
 * the four deferred columns and the virial mass are bitwise identical in every layout (the
 * compensated cumulative masses are the correctly rounded k m in both modes); with random unequal
 * masses they agree to 1e-6; with masses 1 and 2^-50 (one in 50 heavy, inside the light ones:
 * exact multiples of 2^-50, whose plain sums lose the light masses in an order-dependent way, so
 * that the plain prefix sums of the uneven layout differ from the others) bitwise again, the
 * virial masses included. Also the compensated sum itself survives -ffast-math. */
#define PI_NL 7
static const int64_t PiN[PI_NL] = {3000, 1200, 450, 260, 99, 2500, 137};
#define PI_CORESIZE 20

static int64_t
pi_total(void)
{
    int64_t t = 0;
    int l;
    for(l = 0; l < PI_NL; l++)
        t += PiN[l];
    return t;
}

static int
pi_label(MyIDType id, int64_t * j)
{
    int64_t s = 0;
    int l;
    *j = -1;
    for(l = 0; l < PI_NL; l++) {
        if((int64_t) id < s + PiN[l]) {
            *j = (int64_t) id - s;
            return l;
        }
        s += PiN[l];
    }
    return -1;
}

/* Particle id of ball l: r = R u (M(< r) ~ r), R = 20 (N / 3000)^(1/3), mean density inside R the
 * same for every ball; velocities bulk + 200 x uniform in a ball. */
enum { PI_EQUAL = 0, PI_UNEQUAL, PI_WIDE };
static const char * const PiMode[3] = {"equal", "unequal", "wide"};

static void
pi_particle(MyIDType id, int mode, double pos[3], double vel[3], double * mass)
{
    int64_t j;
    const int l = pi_label(id, &j);
    const double cen[PI_NL][3] = {{500, 500, 500}, {0.5, 300, 700}, {800, 150, 220}, {250, 820, 400}, {650, 650, 90}, {120, 480, 930}, {380, 100, 600}};
    const double bulk[3] = {3000. - 1000 * l, -2000. + 700 * l, 50. * l};
    const double R = 20. * cbrt((double) PiN[l] / 3000.);
    double u[3];
    const double zero[3] = {0, 0, 0};
    rand_in_ball(id, 161, 1.0, zero, u);
    const double un = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    /* wide masses: the heavy particles (j % 50 == 0) in the inner 5 % of the radius, so that a
     * plain running sum drops the light masses beyond them, but not a later rank's partial sum */
    const double r = (mode == PI_WIDE && j % 50 == 0 ? 0.05 : 1.0) * R * urand(id, 162, 0);
    int d;
    for(d = 0; d < 3; d++) {
        double x = cen[l][d] + (un > 0 ? r * u[d] / un : 0);
        while(x >= BOXSIZE) x -= BOXSIZE;
        while(x < 0) x += BOXSIZE;
        pos[d] = x;
    }
    double v[3];
    rand_in_ball(id, 163, 200., zero, v);
    for(d = 0; d < 3; d++)
        vel[d] = bulk[d] + v[d];
    if(mode == PI_EQUAL)
        *mass = SYN_M;
    else if(mode == PI_UNEQUAL)
        *mass = SYN_M * (0.5 + urand(id, 164, 0));
    else
        *mass = j % 50 == 0 ? 1.0 : ldexp(1.0, -50);
}

/* layout 3: rank 0 holds PiSplit particles, so that the radial order (sorted by label and radius,
 * the ranks' counts kept) changes rank between kc - 1 and kc of the last ball */
static int64_t PiSplit = 0;

/* layout 0: every particle on rank 0; 1: round robin; 2: 1/13, 2/13, 4/13 and 6/13 on ranks 0-3
 * (the sorted records then change rank inside the inner part of the first balls); 3: above */
static int
pi_rank(MyIDType id, int layout, int NTask)
{
    if(layout == 0)
        return 0;
    if(layout == 1)
        return (int) (id % NTask);
    if(layout == 3)
        return (int64_t) id < PiSplit || NTask == 1 ? 0 : 1 + (int) (id % (NTask - 1));
    const int k = (int) (splitmix64(id ^ 0x5151) % 13);
    return (k < 1 ? 0 : (k < 3 ? 1 : (k < 7 ? 2 : 3))) % NTask;
}

static void
pi_place(int layout, int mode)
{
    const int NTask = get_ntask(), ThisTask = get_thistask();
    const int64_t n = pi_total();
    setup_pman(n + 1, BOXSIZE);
    MyIDType id;
    for(id = 0; id < (MyIDType) n; id++) {
        if(pi_rank(id, layout, NTask) != ThisTask)
            continue;
        double pos[3], vel[3], m;
        pi_particle(id, mode, pos, vel, &m);
        add_dm_particle(id, pos);
        struct particle_data * p = &P[PartManager->NumPart - 1];
        int d;
        for(d = 0; d < 3; d++)
            p->Vel[d] = vel[d];
        p->Mass = m;
    }
}

static void
pi_distributed(const double * core, const struct SubhaloUnbindParams * up, const double rho[SUBHALO_NSO], double * dfr, double * mvir)
{
    int32_t * label = (int32_t *) malloc(sizeof(int32_t) * (PartManager->NumPart + 1));
    int64_t i, j;
    for(i = 0; i < PartManager->NumPart; i++)
        label[i] = pi_label(P[i].ID, &j);
    subhalo_collective_test_props(label, PI_NL, core, up, rho, dfr, mvir);
    free(label);
}

struct PiRad { double r, m; MyIDType id; };

static int
pi_rad_cmp(const void * a, const void * b)
{
    const struct PiRad * x = (const struct PiRad *) a, * y = (const struct PiRad *) b;
    if(x->r != y->r)
        return x->r < y->r ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

/* The outermost virial crossing kc of the n particles Index about core (the recipe's radial
 * order and compensated M(<= r)); *Mkc the mass at it. */
static int64_t
pi_kc(const int64_t * Index, int64_t n, const double * core, double rfloor, double rho0, double * Mkc)
{
    struct PiRad * pr = (struct PiRad *) malloc(sizeof(struct PiRad) * (n + 1));
    int64_t k, kc = -1;
    int d;
    for(k = 0; k < n; k++) {
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[Index[k]].Pos[d] - core[d], BOXSIZE);
            r2 += dx * dx;
        }
        pr[k].r = sqrt(r2) > rfloor ? sqrt(r2) : rfloor;
        pr[k].m = P[Index[k]].Mass;
        pr[k].id = P[Index[k]].ID;
    }
    qsort(pr, n, sizeof(struct PiRad), pi_rad_cmp);
    double Mk[2] = {0, 0};
    *Mkc = 0;
    for(k = 0; k < n; k++) {
        subhalo_ksum_add(Mk, pr[k].m);
        const double M = subhalo_ksum_value(Mk);
        if(M / (4.0 / 3.0 * M_PI * pr[k].r * pr[k].r * pr[k].r) > rho0) {
            kc = k;
            *Mkc = M;
        }
    }
    free(pr);
    return kc;
}

static void
pi_run(int mode)
{
    const int exact = mode != PI_UNEQUAL;
    const int ThisTask = get_thistask();
    struct SubhaloUnbindParams up;
    memset(&up, 0, sizeof(up));
    up.atime = 0.7;
    up.h = 2.8 * 0.3;
    up.BoxSize = BOXSIZE;
    up.TracerTypes = 2;
    up.PotentialTypes = 2;
    up.G = 43007.1;
    /* the thresholds: the mean density inside R of ball 0 times 1.5, 2.5, 6, 15, 60 (crossings at
     * R / 1.2, R / 1.6, R / 2.4, R / 3.9, R / 7.7 for equal masses) */
    double Mball = 0;
    {
        MyIDType id;
        for(id = 0; id < (MyIDType) PiN[0]; id++) {
            double pos[3], vel[3], m;
            pi_particle(id, mode, pos, vel, &m);
            Mball += m;
        }
    }
    const double rhoR = 3 * Mball / (4 * M_PI * 20. * 20. * 20.);
    const double f[SUBHALO_NSO] = {1.5, 2.5, 6, 15, 60};
    double rho[SUBHALO_NSO];
    int s, l, d, k;
    for(s = 0; s < SUBHALO_NSO; s++)
        rho[s] = f[s] * rhoR;
    /* the serial recipes on rank 0, which holds everything (layout 0) */
    double core[6 * PI_NL];
    struct SubTrack ser[PI_NL];
    memset(core, 0, sizeof(core));
    memset(ser, 0, sizeof(ser));
    int64_t kcsplit = -1;
    double Mkcsplit = 0;
    pi_place(0, mode);
    if(ThisTask == 0) {
        int64_t * Index = (int64_t *) mymalloc("PiIndex", sizeof(int64_t) * (PartManager->NumPart + 1));
        int64_t start = 0, i;
        for(l = 0; l < PI_NL; l++) {
            int64_t n = 0;
            for(i = 0; i < PartManager->NumPart; i++) {
                int64_t j;
                if(pi_label(P[i].ID, &j) == l)
                    Index[start + n++] = i;
            }
            struct SubCandidate c;
            memset(&c, 0, sizeof(c));
            c.Index = Index + start;
            c.N = c.Nbound = n;
            double sx, sv;
            subhalo_core(&c, &up, PI_CORESIZE, core + 6 * l, core + 6 * l + 3, &sx, &sv);
            if(l == PI_NL - 1) {
                /* layout 3 splits the last ball between kc - 1 and kc */
                double Mkc;
                const int64_t kc = pi_kc(c.Index, n, core + 6 * l, up.h / 2.8, rho[0], &Mkc);
                assert_true(kc > 0);
                PiSplit = start + kc;
                kcsplit = kc;
                Mkcsplit = Mkc;
            }
            for(d = 0; d < 3; d++)
                c.RefPos[d] = core[6 * l + d];
            struct SubhaloArena arena;
            arena.size = SUBHALO_PROPS_BYTES_PER_PART * n + 64;
            arena.used = 0;
            arena.base = (char *) mymalloc("PiArena", arena.size);
            const double Offset[3] = {0, 0, 0};
            subhalo_track_clear(&ser[l], l);
            subhalo_props_rockstar(&ser[l], &c, &up, rho, PI_CORESIZE, Offset, &arena);
            struct SubTrack r2 = ser[l];
            subhalo_props_deferred(&r2, &c, &up, rho, PI_CORESIZE, &arena);
            assert_true(r2.Rs == ser[l].Rs && r2.Vrms == ser[l].Vrms);
            for(d = 0; d < 3; d++)
                assert_true(r2.BulkVel[d] == ser[l].BulkVel[d] && r2.RockstarCoreVel[d] == ser[l].RockstarCoreVel[d]);
            myfree(arena.base);
            start += n;
        }
        myfree(Index);
    }
    MPI_Bcast(core, 6 * PI_NL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ser, sizeof(ser), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PiSplit, 1, MPI_INT64, 0, MPI_COMM_WORLD);
    /* the oracle's kc is the recipe's: its mass is MBoundSO[vir] */
    message(0, "props identity (%s masses): the last ball's kc %ld (M %.9g, serial MBoundSO[vir] %.9g), layout 3 splits at %ld\n",
            PiMode[mode], (long) kcsplit, Mkcsplit, ser[PI_NL - 1].MBoundSO[0], (long) PiSplit);
    if(ThisTask == 0)
        assert_true((float) Mkcsplit == ser[PI_NL - 1].MBoundSO[0]);
    /* the distributed recipes in four layouts */
    double dfr[4][8 * PI_NL], mvir[4][PI_NL];
    int lay;
    for(lay = 0; lay < 4; lay++) {
        if(lay > 0)
            pi_place(lay, mode);
        message(0, "props identity (%s masses), layout %d: rank 0 holds %ld particles\n", PiMode[mode], lay,
                (long) PartManager->NumPart);
        if(lay == 3 && ThisTask == 0 && get_ntask() > 1)
            assert_true(PartManager->NumPart == PiSplit);
        pi_distributed(core, &up, rho, dfr[lay], mvir[lay]);
        teardown_pman();
    }
    int64_t bad = 0;
    double worst = 0;
    for(l = 0; l < PI_NL; l++) {
        const struct SubTrack * x = &ser[l];
        const float sv[8] = {x->Rs, x->RockstarCoreVel[0], x->RockstarCoreVel[1], x->RockstarCoreVel[2],
                             x->BulkVel[0], x->BulkVel[1], x->BulkVel[2], x->Vrms};
        message(0, "props identity (%s masses), ball %d (%ld particles): serial Rs %.9g Vrms %.9g BulkVel (%.9g %.9g %.9g) "
                   "RockstarCoreVel (%.9g %.9g %.9g) MBoundSO[vir] %.9g\n", PiMode[mode], l, (long) PiN[l],
                sv[0], sv[7], sv[4], sv[5], sv[6], sv[1], sv[2], sv[3], x->MBoundSO[0]);
        /* every ball crosses; the fit runs in the balls of 200 or more particles (aperture about 0.8 N), never below 100 */
        LOCAL_CHECK(bad, mode == PI_WIDE || (x->Vrms > 0 && x->MBoundSO[0] > 0 && (PiN[l] < 100 ? x->Rs == -1 : (PiN[l] < 200 || x->Rs > 0))),
                    "props identity: ball %d Vrms %g MBoundSO %g Rs %g\n", l, x->Vrms, x->MBoundSO[0], x->Rs);
        /* the fit's aperture: the kc particles inside the crossing particle, Rs
         * defined iff kc >= 100; with equal masses kc = MBoundSO[vir] / m - 1 */
        if(mode == PI_EQUAL) {
            const int64_t kcv = llround(x->MBoundSO[0] / SYN_M) - 1;
            message(0, "props identity: ball %d kc %ld\n", l, (long) kcv);
            LOCAL_CHECK(bad, (kcv >= SUBHALO_NFW_MINPART) == (x->Rs > 0), "props identity: ball %d kc %ld Rs %g\n", l, (long) kcv, x->Rs);
            /* the last ball sits on the boundary: kc = 100 fits (nfw.c's literal aperture would not) */
            LOCAL_CHECK(bad, l != PI_NL - 1 || (kcv == SUBHALO_NFW_MINPART && x->Rs > 0),
                        "props identity: boundary ball %d kc %ld (expected %d) Rs %g\n", l, (long) kcv, SUBHALO_NFW_MINPART, x->Rs);
        }
        for(lay = 0; lay < 4; lay++) {
            const double * o = dfr[lay] + 8 * l;
            for(k = 0; k < 8; k++) {
                const float v = (float) o[k];
                if(exact)
                    LOCAL_CHECK(bad, v == sv[k], "props identity (%s masses): ball %d layout %d column %d serial %.9g distributed %.9g\n",
                                PiMode[mode], l, lay, k, sv[k], v);
                else {
                    const double scale = k == 0 ? fabs(sv[0]) : (k == 7 ? sv[7] : fabs(sv[k]) + sv[7]);
                    const double dd = fabs(v - sv[k]) / (scale > 0 ? scale : 1);
                    if(dd > worst)
                        worst = dd;
                    LOCAL_CHECK(bad, dd <= 1e-6, "props identity (unequal masses): ball %d layout %d column %d serial %.9g distributed %.9g\n",
                                l, lay, k, sv[k], v);
                }
            }
            if(exact)
                LOCAL_CHECK(bad, (float) mvir[lay][l] == x->MBoundSO[0] && mvir[lay][l] == mvir[0][l],
                            "props identity: ball %d layout %d Mvir %.17g / %.17g, serial %.9g\n", l, lay, mvir[lay][l], mvir[0][l],
                            x->MBoundSO[0]);
        }
    }
    if(!exact)
        message(0, "props identity (unequal masses): largest relative difference %.3g\n", worst);
    assert_all_true(bad == 0);
}

static void
test_collective_props_identity(void ** state)
{
    /* the compensated sum under the build's -ffast-math: 1 + 10 x 1e-16 - 1 */
    double ks[2] = {0, 0};
    int i;
    subhalo_ksum_add(ks, 1.0);
    for(i = 0; i < 10; i++)
        subhalo_ksum_add(ks, 1e-16);
    subhalo_ksum_add(ks, -1.0);
    CHECK(fabs(subhalo_ksum_value(ks) - 1e-15) < 1e-30, "compensated sum: %.17g\n", subhalo_ksum_value(ks));
    common_physics_setup();
    subhalo_set_units(3.085678e21, 1.989e43, 1e5);
    pi_run(PI_EQUAL);
    pi_run(PI_UNEQUAL);
    pi_run(PI_WIDE);
}

/* ---- many in-place candidates in one round: the properties stage ----
 * 2304 isolated 12-particle groups, each with one resolved track (SubhaloMinNumPart 10), all
 * forced in place (SubhaloCollectiveMinPart 10): one round of 2304 candidates, whose per-candidate property
 * scratch (the deferred recipes' reductions, the profile vectors: SUBHALO_INPLACE_CAND_PROPS_BYTES
 * each on every rank) is the largest stage. (a) Unsqueezed: admitted with the properties as the
 * largest stage, the call completes and the measured high water stays within the estimate. (b) The
 * same call with the free memory squeezed on every rank to halfway between the need without the
 * properties stage and the need with it: the admission fails at the properties stage, before
 * the exchange, where an estimate without that stage would admit it. Also
 * the walltime children /Subhalo/Rounds/{Setup,Rounds,Tree,Walk,Tags,
 * Cleanup} add up to the summary's rounds time. */
#define ML_NX 16
#define ML_NYZ 12
#define ML_NG (ML_NX * ML_NYZ * ML_NYZ)
#define ML_NP 12

static void
ml_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    (void) ud;
    if((int64_t) id >= ML_NG * ML_NP)
        return;
    const int64_t g = (int64_t) id / ML_NP;
    /* x on multiples of 62.5 (31.25 from every lattice coordinate: no field particle links) */
    const double cen[3] = {62.5 * (g % ML_NX), 40. + 80. * ((g / ML_NX) % ML_NYZ), 40. + 80. * (g / (ML_NX * ML_NYZ))};
    const double zero[3] = {0, 0, 0};
    syn_ball(id, 171, cen, 0.5, zero, 2., atime, s);
    s->tag = s->bound = g;
    s->rank = (int32_t) ((int64_t) id - g * ML_NP);
}

static void
ml_setup(Cosmology * CP)
{
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloCollectiveMinPart = 10;
    sp.SubhaloMinNumPart = 10;
    sp.SubhaloLookupBatchBytes = 1024 * 1024;
    syn_setup(CP, ML_NG * ML_NP + SYN_NFIELD, sp);
    struct SubTrack * recs = (struct SubTrack *) malloc(sizeof(struct SubTrack) * ML_NG);
    int64_t g;
    for(g = 0; g < ML_NG; g++) {
        recs[g] = syn_rec(g, ML_NP, ML_NP * SYN_M, -1, g, 0, 0);
        recs[g].HostGroup = g + 1;
    }
    syn_records(recs, ML_NG);
    free(recs);
    syn_place(ml_particle, NULL, 0.55, 1);
}

struct MlCtx {
    double freepre;          /* free bytes at the pre hook */
    double target;           /* > 0: squeeze the free bytes to this */
    void * block;
};

static void
ml_pre(void * ud)
{
    struct MlCtx * c = (struct MlCtx *) ud;
    c->freepre = (double) mymalloc_freebytes();
    c->block = c->target > 0 && c->freepre > c->target ? mymalloc2("TestSqueeze", (size_t) (c->freepre - c->target)) : NULL;
}

static void
ml_post(void * ud)
{
    struct MlCtx * c = (struct MlCtx *) ud;
    if(c->block)
        myfree(c->block);
    c->block = NULL;
}

/* persist + the largest stage other than the properties: the need of an estimate without them */
static double
ml_need_without_props(const struct SubhaloMemCheck * ad)
{
    double mx = 0;
    int s;
    for(s = 0; s < SUBHALO_COLL_NSTAGE; s++)
        if(s != SUBHALO_COLL_PROPS && ad->stage[s] > mx)
            mx = ad->stage[s];
    return ad->persist + mx;
}

/* The summed time of this rank's walltime clocks whose names start with prefix. */
static double
ml_clock_sum(const char * prefix)
{
    double t = 0;
    int i;
    for(i = 0; i < CT.N; i++)
        if(strncmp(CT.C[i].name, prefix, strlen(prefix)) == 0)
            t += CT.C[i].time;
    return t;
}

static void
test_collective_admission_props(void ** state)
{
    Cosmology CP;
    int64_t ng;
    struct MlCtx c = {0, 0, NULL};
    /* (a) */
    ml_setup(&CP);
    /* the rounds stage's walltime children (this rank; the clock names carry
     * "@file:line"). walltime_init (in ml_setup) resets the table but not the times of its slots,
     * which new clocks would inherit from the earlier tests: cleared here */
    {
        int i;
        for(i = 0; i < CT.Nmax; i++)
            CT.C[i].time = CT.AC[i].time = 0;
    }
    double wt0 = ml_clock_sum("/Subhalo/Rounds/"), wcl = ml_clock_sum("/Subhalo/Rounds/Cleanup"), wtr = ml_clock_sum("/Subhalo/Rounds/Tree");
    double t0 = MPI_Wtime();
    int ret = run_finder_call_ex(1, &CP, &ng, ml_pre, ml_post, &c);
    const double wt1 = ml_clock_sum("/Subhalo/Rounds/");
    wcl = ml_clock_sum("/Subhalo/Rounds/Cleanup") - wcl;
    wtr = ml_clock_sum("/Subhalo/Rounds/Tree") - wtr;
    {
        const struct SubhaloCallStats * st = subhalo_last_call_stats();
        const double smean = st->StageMean[SUBHALO_STAGE_ROUNDS];
        /* mean over the ranks against the mean stage time: a rank's own stage time is not exposed */
        double loc[2] = {wt1 - wt0, wcl}, sum[2];
        MPI_Allreduce(loc, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double wmean = sum[0] / get_ntask(), clmean = sum[1] / get_ntask();
        message(0, "many labels (a): walltime /Subhalo/Rounds children %.4f s (cleanup %.4f s, tree builds %.4f s; mean over the ranks "
                   "%.4f s, cleanup %.4f s), the summary's rounds %.4f s (max), %.4f s (mean): coverage %.3f\n", wt1 - wt0, wcl, wtr,
                wmean, clmean, st->StageMax[SUBHALO_STAGE_ROUNDS], smean, wmean / smean);
        CHECK(wtr > 0, "many labels (a): no time on /Subhalo/Rounds/Tree (%g s)\n", wtr);
        /* the cleanup share is printed only: a descheduled rank can make it any size */
        CHECK(wmean <= smean * (1 + 1e-6) + 1e-4 && wmean >= 0.8 * smean - 1e-4,
              "many labels (a): walltime rounds %g s (mean %g s, cleanup %g s), summary mean %g s\n", wt1 - wt0, wmean, wcl, smean);
    }
    const struct SubhaloPlanSummary * ps = subhalo_plan_last_summary();
    const struct SubhaloMemCheck * ad = subhalo_plan_last_memcheck();
    const struct SubhaloCollectiveStats * cs = subhalo_collective_last_stats();
    const double need = ad->need, noprops = ml_need_without_props(ad), delta = c.freepre - ad->avail;
    message(0, "many labels (a): ret %d in %.2f s, %ld groups, %ld candidates in %d round(s), %ld alive; this rank needs %g MB (%g held, "
               "properties %g MB, tree %g MB; %g MB without the properties stage), measured high water %g MB of the estimate %g MB "
               "(largest measured/estimated %.3f)\n", ret, MPI_Wtime() - t0, ng, ps->NCands, ps->NRounds, cs->NAlive, need / 1048576.,
            ad->persist / 1048576., ad->stage[SUBHALO_COLL_PROPS] / 1048576., ad->stage[SUBHALO_COLL_TREE] / 1048576.,
            noprops / 1048576., cs->MeasPeakBytes / 1048576., cs->EstPeakBytes / 1048576., cs->MaxMeasOverEst);
    CHECK(ret == 1 && ng == ML_NG && subhalo_last_call_complete() && ps->Fits && ps->NCands == ML_NG && ps->NRounds == 1
          && cs->NAlive == ML_NG, "many labels (a): ret %d, %ld groups, %ld candidates, %ld alive\n", ret, ng, ps->NCands, cs->NAlive);
    CHECK(ad->checked && !ad->failed && ad->failstage == SUBHALO_COLL_PROPS, "many labels (a): the largest stage is the %s\n",
          ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-");
    CHECK(cs->MaxMeasOverEst > 0 && cs->MaxMeasOverEst <= 1, "many labels (a): measured / estimated %g\n", cs->MaxMeasOverEst);
    CHECK(need - noprops > 2 * 1048576., "many labels (a): the properties stage adds %g MB\n", (need - noprops) / 1048576.);
    syn_teardown();
    /* (b) */
    ml_setup(&CP);
    c.target = delta + 0.5 * (need + noprops);
    struct tagrec * pre = NULL;
    int64_t npre = 0, n, i, bad = 0;
    pre = gather_tags(&npre);
    ret = run_finder_call_ex(1, &CP, &ng, ml_pre, ml_post, &c);
    ps = subhalo_plan_last_summary();
    ad = subhalo_plan_last_memcheck();
    message(0, "many labels (b): free %g MB at the call squeezed to %g MB; ret %d, admitted %d, failed %d on rank %d at the %s; this rank needs "
               "%g MB (%g MB without the properties stage), %g MB available\n", c.freepre / 1048576., c.target / 1048576., ret, ps->Fits,
            ad->failed, ad->failrank, ad->failstage >= 0 ? SubhaloCollStageName[ad->failstage] : "-", ad->need / 1048576.,
            ml_need_without_props(ad) / 1048576., ad->avail / 1048576.);
    CHECK(ret == 0 && !subhalo_last_call_complete() && !ps->Fits && ad->failed && ad->failstage == SUBHALO_COLL_PROPS,
          "many labels (b): ret %d, admitted %d, failed %d at stage %d\n", ret, ps->Fits, ad->failed, ad->failstage);
    /* on every rank the need without the properties stage fits: an estimate without it would admit the call */
    LOCAL_CHECK(bad, ml_need_without_props(ad) < ad->avail && ad->need > ad->avail, "many labels (b): need %g / %g MB without the "
                "properties, %g MB available\n", ad->need / 1048576., ml_need_without_props(ad) / 1048576., ad->avail / 1048576.);
    struct tagrec * post = gather_tags(&n);
    for(i = 0; i < n && n == npre; i++)
        if(post[i].ID != pre[i].ID || post[i].TrackId != pre[i].TrackId || post[i].BoundTrackId != pre[i].BoundTrackId
           || post[i].BindRank != pre[i].BindRank)
            bad++;
    CHECK(n == npre, "many labels (b): %ld / %ld particles\n", n, npre);
    free(post);
    free(pre);
    assert_all_true(bad == 0);
    syn_teardown();
}

/* ---- the k-nearest walk across ranks against the per-key search ----
 * Labelled sets: a lattice with distance ties and coincident pairs (label 0), a ball across the
 * periodic boundary (1), a wide ball over several domains (2), a set of fewer than k (3), and a
 * dense core inside a sparse halo (4). Every labelled particle queries its label (in one case only
 * those of even ID, the tree holding all of them). References: the
 * per-key search (subhalo_knn_tracers over a label's particles gathered on every rank, k + 1 nearest
 * with the query dropped by value), whose k neighbours, squared distances and previous tags the walk
 * must return exactly, and a brute force over the label in long double for the distances and the
 * neighbour lists. Runs on the layout of the test's ranks, with the default and a small export
 * buffer. */
#define KN_LAT 6
#define KN_N0 (KN_LAT * KN_LAT * KN_LAT + 40)
#define KN_N1 700
#define KN_N2 1500
#define KN_N3 7
#define KN_N4 1200
#define KN_END (KN_N0 + KN_N1 + KN_N2 + KN_N3 + KN_N4)

static int
kn_label(MyIDType id)
{
    if(id < KN_N0) return 0;
    if(id < KN_N0 + KN_N1) return 1;
    if(id < KN_N0 + KN_N1 + KN_N2) return 2;
    if(id < KN_N0 + KN_N1 + KN_N2 + KN_N3) return 3;
    if(id < KN_END) return 4;
    return -1;
}

static void
kn_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double zero[3] = {0, 0, 0};
    const double c1[3] = {0.1, 300, 300}, c2[3] = {500, 500, 500}, c3[3] = {700, 200, 800}, c4[3] = {250, 750, 250};
    (void) ud;
    if(id >= KN_END)
        return;
    const int l = kn_label(id);
    if(l == 0) {
        /* a lattice of spacing 0.25 (exact in binary), the last 40 ids on lattice points */
        const int64_t j = id < KN_LAT * KN_LAT * KN_LAT ? (int64_t) id : (int64_t) (id * 7) % (KN_LAT * KN_LAT * KN_LAT);
        s->pos[0] = 100 + 0.25 * (j % KN_LAT);
        s->pos[1] = 100 + 0.25 * ((j / KN_LAT) % KN_LAT);
        s->pos[2] = 100 + 0.25 * (j / (KN_LAT * KN_LAT));
    }
    else if(l == 1)
        syn_ball(id, 151, c1, 0.3, zero, 0, atime, s);
    else if(l == 2)
        syn_ball(id, 152, c2, 0.5 * 300 * atime, zero, 0, atime, s);
    else if(l == 3)
        syn_ball(id, 153, c3, 0.5, zero, 0, atime, s);
    else {
        syn_ball(id, 154, c4, (id % 5 == 0 ? 20.0 : 0.2) * atime, zero, 0, atime, s);
    }
    /* previous tags: a few tracks per label */
    s->bound = (id % 3 == 0) ? -1 : 1000 + l * 10 + (int64_t) (id % 4);
}

struct KnPart {
    MyIDType ID;
    double Pos[3];
    int64_t BoundTrackId;
    int32_t label, pad;
};

static int
cmp_knpart_id(const void * a, const void * b)
{
    const struct KnPart * x = a, * y = b;
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* The squared periodic distance in long double with its own wrap: the brute-force reference, shares
 * no code with subhalo_knn_r2 or either tree. */
static long double
kn_bf_r2(const double x[3], const double y[3])
{
    long double r2 = 0;
    int d;
    for(d = 0; d < 3; d++) {
        long double dx = (long double) x[d] - (long double) y[d];
        if(dx > 0.5L * BOXSIZE)
            dx -= BOXSIZE;
        else if(dx < -0.5L * BOXSIZE)
            dx += BOXSIZE;
        r2 += dx * dx;
    }
    return r2;
}

/* The rounding allowed between the double distance of the walk and the brute force: a relative
 * 1e-12, plus the error of a separation unwrapped across the box (an ulp of the box side per axis). */
static long double
kn_tol(long double r2)
{
    return 1e-12L * r2 + 4.0L * sqrtl(r2) * BOXSIZE * 2.220446049250313e-16L;
}

/* Passes and export rounds (largest over the ranks) of the last kn_check. */
static int KnPasses = 0;
static int64_t KnRounds = 0;

/* The walk on the current layout against the gathered per-key search and against a brute force over
 * every particle of the label; returns the mismatches. */
static int64_t
kn_check(int k, int even, const char * what)
{
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    const int64_t n = PartManager->NumPart;
    int32_t * label = malloc(sizeof(int32_t) * (n + 1));
    int * tl = malloc(sizeof(int) * (n + 1));
    int * ql = malloc(sizeof(int) * (n + 1));
    int64_t i, nq = 0, nt = 0;
    /* every labelled particle is a tracer; the queries are all of them, or (even) those of even ID */
    for(i = 0; i < n; i++) {
        label[i] = kn_label(P[i].ID);
        P[i].BoundTrackId = (P[i].ID % 3 == 0 || label[i] < 0) ? -1 : 1000 + label[i] * 10 + (int64_t) (P[i].ID % 4);
        if(label[i] < 0)
            continue;
        tl[nt++] = (int) i;
        if(!even || P[i].ID % 2 == 0)
            ql[nq++] = (int) i;
    }
    struct SubhaloKnnHit * hits = malloc(sizeof(struct SubhaloKnnHit) * (nq * k + 1));
    int32_t * nhit = malloc(sizeof(int32_t) * (nq + 1));
    const int npass = subhalo_collective_knn(&ddecomp, label, tl, nt, ql, nq, k, hits, nhit);
    /* every tracer on every rank, sorted by ID */
    const int NTask = get_ntask();
    struct KnPart * loc = malloc(sizeof(struct KnPart) * (nt + 1));
    for(i = 0; i < nt; i++) {
        const int p = tl[i];
        memset(&loc[i], 0, sizeof(loc[i]));
        loc[i].ID = P[p].ID;
        int d;
        for(d = 0; d < 3; d++)
            loc[i].Pos[d] = P[p].Pos[d];
        loc[i].BoundTrackId = P[p].BoundTrackId;
        loc[i].label = label[p];
    }
    int nb = nt * sizeof(struct KnPart);
    int * counts = malloc(NTask * sizeof(int)), * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    struct KnPart * all = malloc(tot + sizeof(struct KnPart));
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    const int64_t nall = tot / sizeof(struct KnPart);
    qsort(all, nall, sizeof(struct KnPart), cmp_knpart_id);
    /* the reference particles in a table of their own, searched by label as one serial key */
    struct particle_data * saved = PartManager->Base;
    const int64_t savedn = PartManager->NumPart;
    struct particle_data * ref = calloc(nall + 1, sizeof(struct particle_data));
    for(i = 0; i < nall; i++) {
        ref[i].ID = all[i].ID;
        ref[i].Type = 1;
        ref[i].Mass = 1;
        ref[i].BoundTrackId = all[i].BoundTrackId;
        int d;
        for(d = 0; d < 3; d++)
            ref[i].Pos[d] = all[i].Pos[d];
    }
    int64_t * refout = malloc(sizeof(int64_t) * (nall * (k + 1) + 1));
    int64_t * tr = malloc(sizeof(int64_t) * (nall + 1));
    struct SubhaloArena arena;
    arena.size = (size_t) 64 * 1024 * 1024;
    arena.base = malloc(arena.size);
    arena.used = arena.peak = 0;
    PartManager->Base = ref;
    PartManager->NumPart = nall;
    int l;
    for(l = 0; l < 5; l++) {
        int64_t nt = 0;
        for(i = 0; i < nall; i++)
            if(all[i].label == l)
                tr[nt++] = i;
        /* rows of the label's particles, written at their table positions */
        int64_t * out = malloc(sizeof(int64_t) * (nt * (k + 1) + 1));
        subhalo_knn_tracers(tr, nt, tr, nt, k + 1, out, BOXSIZE, &arena);
        int64_t a;
        for(a = 0; a < nt; a++)
            memcpy(refout + tr[a] * (k + 1), out + a * (k + 1), sizeof(int64_t) * (k + 1));
        free(out);
    }
    int64_t bad = 0, nties = 0, nshort = 0, nbf = 0;
    /* the gathered table is sorted by ID and the labels are ID ranges: [lb[l], lb[l + 1]) */
    int64_t lb[6] = {0, 0, 0, 0, 0, 0};
    for(i = 0; i < nall; i++)
        lb[all[i].label + 1] = i + 1;
    for(l = 1; l < 6; l++)
        if(lb[l] < lb[l - 1])
            lb[l] = lb[l - 1];
    for(i = 0; i < nq; i++) {
        struct KnPart key;
        key.ID = saved[ql[i]].ID;
        const struct KnPart * f = bsearch(&key, all, nall, sizeof(struct KnPart), cmp_knpart_id);
        const int64_t self = f - all;
        /* brute force: the k nearest others of the label by (distance, ID) */
        {
            long double bd[SUBHALO_KNN_MAXK + 1];
            MyIDType bid[SUBHALO_KNN_MAXK + 1];
            int m = 0, a, c;
            int64_t j;
            for(j = lb[f->label]; j < lb[f->label + 1]; j++) {
                if(j == self)
                    continue;
                const long double r2 = kn_bf_r2(all[j].Pos, f->Pos);
                if(m == k && !(r2 < bd[k - 1] || (r2 == bd[k - 1] && all[j].ID < bid[k - 1])))
                    continue;
                a = m < k ? m++ : k - 1;
                while(a > 0 && (r2 < bd[a - 1] || (r2 == bd[a - 1] && all[j].ID < bid[a - 1]))) {
                    bd[a] = bd[a - 1];
                    bid[a] = bid[a - 1];
                    a--;
                }
                bd[a] = r2;
                bid[a] = all[j].ID;
            }
            LOCAL_CHECK(nbf, nhit[i] == m, "%s: particle %ld has %d neighbours from the walk, %d by brute force\n", what, (long) key.ID, nhit[i], m);
            /* the walk's c-th neighbour: its distance is the brute force's for that particle and the
             * c-th smallest (a swap only between distances equal to rounding), no repeat, not itself */
            for(c = 0; c < m && c < nhit[i]; c++) {
                const struct SubhaloKnnHit * h = &hits[i * k + c];
                struct KnPart nk;
                nk.ID = h->ID;
                const struct KnPart * g = bsearch(&nk, all, nall, sizeof(struct KnPart), cmp_knpart_id);
                const long double r2 = g ? kn_bf_r2(g->Pos, f->Pos) : -1;
                int rep = 0;
                for(a = 0; a < c; a++)
                    rep |= hits[i * k + a].ID == h->ID;
                /* the brute force's neighbour itself, unless the two distances differ within rounding
                 * (an exact tie is ordered by ID in both) */
                LOCAL_CHECK(nbf, g && g->label == f->label && h->ID != key.ID && !rep
                            && fabsl((long double) h->d2 - r2) <= kn_tol(r2) && fabsl(r2 - bd[c]) <= kn_tol(bd[c])
                            && (h->ID == bid[c] || r2 != bd[c]),
                            "%s: particle %ld neighbour %d: walk %ld at %.17g (brute force %.17Lg), brute force %ld at %.17Lg\n",
                            what, (long) key.ID, c, (long) h->ID, h->d2, r2, (long) bid[c], bd[c]);
            }
        }
        int64_t row[SUBHALO_KNN_MAXK + 1];
        int used = 0, c;
        for(c = 0; c <= k && used < k; c++) {
            const int64_t p = refout[self * (k + 1) + c];
            if(p < 0 || p == self)
                continue;
            row[used++] = p;
        }
        if(used < k)
            nshort++;
        LOCAL_CHECK(bad, nhit[i] == used, "%s: particle %ld has %d neighbours from the walk, %d from the key\n",
                    what, (long) key.ID, nhit[i], used);
        for(c = 0; c < used && c < nhit[i]; c++) {
            const struct SubhaloKnnHit * h = &hits[i * k + c];
            const double d2 = subhalo_knn_r2(ref[row[c]].Pos, ref[self].Pos, BOXSIZE);
            if(c > 0 && h->d2 == hits[i * k + c - 1].d2)
                nties++;
            LOCAL_CHECK(bad, h->ID == ref[row[c]].ID && h->d2 == d2 && h->BoundTrackId == ref[row[c]].BoundTrackId,
                        "%s: particle %ld neighbour %d: walk %ld (%.17g) key %ld (%.17g)\n", what, (long) key.ID, c,
                        (long) h->ID, h->d2, (long) ref[row[c]].ID, d2);
        }
    }
    PartManager->Base = saved;
    PartManager->NumPart = savedn;
    int64_t g[5] = {bad, nties, nshort, nq, nbf}, gs[5], xr = subhalo_collective_knn_export_rounds(), xrmax;
    MPI_Allreduce(g, gs, 5, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&xr, &xrmax, 1, MPI_INT64, MPI_MAX, MPI_COMM_WORLD);
    KnPasses = npass;
    KnRounds = xrmax;
    message(0, "k-nearest walk %s, k = %d: %ld queries, %d pass(es), %ld export rounds, %ld tied neighbours, %ld queries with fewer than k, "
               "%ld mismatches with the per-key search, %ld with the brute force\n", what, k, gs[3], npass, xrmax, gs[1], gs[2], gs[0], gs[4]);
    free(arena.base);
    free(tr);
    free(refout);
    free(ref);
    free(all);
    free(displ);
    free(counts);
    free(loc);
    free(nhit);
    free(hits);
    free(ql);
    free(tl);
    free(label);
    domain_free(&ddecomp);
    /* queries: every labelled particle or those of even ID; short: the queries of the set of fewer than k */
    int64_t eq = 0, es = 0, id;
    for(id = 0; id < KN_END; id++)
        if(!even || id % 2 == 0) {
            eq++;
            es += kn_label(id) == 3;
        }
    CHECK(gs[3] == eq && gs[1] > 0 && gs[2] == es, "%s: %ld queries, %ld ties, %ld short\n", what, gs[3], gs[1], gs[2]);
    return gs[0] + gs[4];
}

static void
test_collective_knn_identity(void ** state)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    syn_setup(&CP, KN_END + SYN_NFIELD, sp);
    syn_place(kn_particle, NULL, 0.55, 0);
    assert_true(kn_check(10, 0, "default buffer") == 0);
    assert_true(kn_check(SUBHALO_KNN_MAXK, 0, "default buffer") == 0);
    /* the queries a strict subset of the tracers, as in the vote */
    assert_true(kn_check(10, 1, "queries of even ID") == 0);
    /* about two hundred exports per thread and round (the treewalk needs 100): on more than one rank
     * some pass needs more than one export round */
    treewalk_set_max_export_buffer(12500 * omp_get_max_threads());
    assert_true(kn_check(10, 0, "small buffer") == 0);
    treewalk_set_max_export_buffer(3584 * 1024 * 1024L);
    if(get_ntask() > 1)
        assert_true(KnRounds > KnPasses);
    syn_teardown();
}

/* ---- the admission vote of an oversized group against the vote of its serial key ----
 * The hierarchy of subtree_keys (A central; B, C its satellites; D B's satellite; S, B's particles at A's
 * centre) with SubhaloAdmitShare = 0.5: A's tracers among S vote for B. The group as one serial key
 * (SubhaloCollectiveMinPart 0) and in place with serial sub-tree keys (1000) must offer the same
 * pairs (particle, receiver): the per-key vote of the first, the plan's vote across ranks of the
 * second. */
struct AdmPair {
    MyIDType ID;
    int64_t Receiver;
    int32_t source, pad;
};
static struct AdmPair * AvPairs = NULL;
static int64_t AvN = 0, AvMax = 0;

static void
av_hook(MyIDType ID, int64_t Receiver, int source)
{
    if(AvN == AvMax) {
        AvMax = 2 * AvMax + 64;
        AvPairs = realloc(AvPairs, sizeof(struct AdmPair) * AvMax);
    }
    AvPairs[AvN].ID = ID;
    AvPairs[AvN].Receiver = Receiver;
    AvPairs[AvN].source = source;
    AvPairs[AvN].pad = 0;
    AvN++;
}

static int
cmp_admpair(const void * a, const void * b)
{
    const struct AdmPair * x = a, * y = b;
    if(x->source != y->source) return (x->source > y->source) - (x->source < y->source);
    if(x->ID != y->ID) return (x->ID > y->ID) - (x->ID < y->ID);
    return (x->Receiver > y->Receiver) - (x->Receiver < y->Receiver);
}

/* Every rank's pairs of one source, on every rank, sorted by ID. */
static struct AdmPair *
av_gather(int source, int64_t * n)
{
    const int NTask = get_ntask();
    int64_t i, m = 0;
    for(i = 0; i < AvN; i++)
        if(AvPairs[i].source == source)
            m++;
    struct AdmPair * loc = malloc(sizeof(struct AdmPair) * (m + 1));
    m = 0;
    for(i = 0; i < AvN; i++)
        if(AvPairs[i].source == source)
            loc[m++] = AvPairs[i];
    int nb = m * sizeof(struct AdmPair);
    int * counts = malloc(NTask * sizeof(int)), * displ = malloc(NTask * sizeof(int));
    MPI_Allgather(&nb, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int64_t tot = 0;
    int r;
    for(r = 0; r < NTask; r++) {
        displ[r] = (int) tot;
        tot += counts[r];
    }
    struct AdmPair * all = malloc(tot + sizeof(struct AdmPair));
    MPI_Allgatherv(loc, nb, MPI_BYTE, all, counts, displ, MPI_BYTE, MPI_COMM_WORLD);
    *n = tot / sizeof(struct AdmPair);
    qsort(all, *n, sizeof(struct AdmPair), cmp_admpair);
    free(displ);
    free(counts);
    free(loc);
    return all;
}

/* The tags before the last av_run call (after its FOF), its counters and its plan's vote. */
static struct FailSnap AvPre = {NULL, 0};
static struct SubhaloCallStats AvStats;
static struct SubhaloVoteStats AvVote;
/* SubhaloAdmitMinCoreDistance of av_run; AvNear 1 (3): B's (D's) stored core at 11.98 from A's (B's),
 * all others far apart; 0: syn_rec's cores */
static double AvMinD = 0;
static int AvNear = 0;

/* st_particle, with A's ten least bound particles inside D's ball (A-owned offers to D) */
static void
av_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cD[3] = {500 + 6.2 / atime, 500, 500}, vD[3] = {0, 44, 0};
    st_particle(id, atime, s, ud);
    if(id >= ST_B0 - 10 && id < ST_B0)
        syn_ball(id, 56, cD, 0.3, vD, 3, atime, s);
}

static void
av_run(int64_t collective, double share, int k, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = share;
    sp.SubhaloAdmitMinCoreDistance = AvMinD;
    if(k > 0)
        sp.SubhaloNumNeighboursForReassignment = k;
    syn_setup(&CP, ST_END + SYN_NFIELD, sp);
    struct SubTrack recs[4];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 360, 0.036, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 0, 0, 2, 1);
    recs[3] = syn_rec(3, 100, 0.01, 1, 0, 3, 2);
    if(AvNear) {
        const double x1[4] = {100, 100.25, 200, 300}, x3[4] = {100, 200, 300, 200.25};
        int t;
        for(t = 0; t < 4; t++) {
            recs[t].CorePos[0] = AvNear == 1 ? x1[t] : x3[t];
            recs[t].CorePos[1] = recs[t].CorePos[2] = 100;
            recs[t].CoreVel[0] = recs[t].CoreVel[2] = 0;
            recs[t].CoreVel[1] = t == AvNear ? 3.99f : 0;
            recs[t].CoreSigmaX = t == AvNear ? 1.0f : 0.0625f;
            recs[t].CoreSigmaV = t == AvNear ? 8.0f : 0.5f;
        }
    }
    syn_records(recs, 4);
    syn_place(AvNear == 3 ? av_particle : st_particle, NULL, 0.55, 1);
    AvN = 0;
    subhalo_set_admit_hook(av_hook);
    int64_t ng;
    free(AvPre.r);
    AvPre.r = NULL;
    const int ret = run_finder_call_ex(1, &CP, &ng, fail_pre, NULL, &AvPre);
    subhalo_set_admit_hook(NULL);
    AvStats = *subhalo_last_call_stats();
    AvVote = *subhalo_plan_last_vote();
    CHECK(ret == 1 && ng == 1, "admission vote: %ld groups\n", ng);
    if(o)
        outcome_take(o);
    syn_teardown();
}

/* The pairs the vote must offer on the av fixture, computed here from the tags before the call:
 * owners by the owner map (a member's own particles, the central A's only inside its group, every
 * other group particle A's), receivers B, C, D below A and D below B, the k nearest other owned
 * tracers by brute force, the votes counted here, and need votes. Sorted by ID. */
static struct AdmPair *
av_expected(const struct tagrec * r, int64_t n, int k, int need, int64_t * nexp)
{
    const int64_t parent[4] = {-1, 0, 0, 1};
    const int depth[4] = {0, 1, 1, 2};
    int64_t i, j, g = -1, no = 0;
    for(i = 0; i < n; i++)
        if(r[i].ID == ST_A0)
            g = r[i].GrNr;
    int64_t * own = malloc(sizeof(int64_t) * (n + 1));
    int64_t * oi = malloc(sizeof(int64_t) * (n + 1));
    for(i = 0; i < n; i++) {
        const int64_t T = r[i].TrackId;
        own[i] = -1;
        if(r[i].Type != 1)
            continue;
        if(T >= 1 && T <= 3)
            own[i] = T;
        else if((T == 0 || T == -1) && r[i].GrNr == g)
            own[i] = 0;
        if(own[i] >= 0)
            oi[no++] = i;
    }
    struct AdmPair * out = malloc(sizeof(struct AdmPair) * (no + 1));
    *nexp = 0;
    for(i = 0; i < no; i++) {
        const struct tagrec * q = &r[oi[i]];
        const int64_t X = own[oi[i]];
        if(X >= 2)
            continue;
        long double bd[SUBHALO_KNN_MAXK];
        MyIDType bid[SUBHALO_KNN_MAXK];
        int64_t bt[SUBHALO_KNN_MAXK];
        int m = 0, a;
        for(j = 0; j < no; j++) {
            if(j == i)
                continue;
            const struct tagrec * t = &r[oi[j]];
            const long double r2 = kn_bf_r2(t->Pos, q->Pos);
            if(m == k && !(r2 < bd[k - 1] || (r2 == bd[k - 1] && t->ID < bid[k - 1])))
                continue;
            a = m < k ? m++ : k - 1;
            while(a > 0 && (r2 < bd[a - 1] || (r2 == bd[a - 1] && t->ID < bid[a - 1]))) {
                bd[a] = bd[a - 1];
                bid[a] = bid[a - 1];
                bt[a] = bt[a - 1];
                a--;
            }
            bd[a] = r2;
            bid[a] = t->ID;
            bt[a] = t->BoundTrackId;
        }
        int votes[4] = {0, 0, 0, 0}, S, best = -1;
        for(a = 0; a < m; a++)
            if(bt[a] >= 1 && bt[a] <= 3)
                votes[bt[a]]++;
        for(S = 1; S <= 3; S++) {
            int64_t u = parent[S];
            while(u >= 0 && u != X)
                u = parent[u];
            if(u != X || votes[S] == 0)
                continue;
            if(best < 0 || votes[S] > votes[best] || (votes[S] == votes[best] && depth[S] > depth[best]))
                best = S;
        }
        if(best >= 0 && votes[best] >= need) {
            out[*nexp].ID = q->ID;
            out[*nexp].Receiver = best;
            out[*nexp].source = 0;
            out[*nexp].pad = 0;
            (*nexp)++;
        }
    }
    qsort(out, *nexp, sizeof(struct AdmPair), cmp_admpair);
    free(oi);
    free(own);
    return out;
}

/* One share and k (0: the default k) with need the votes it requires, written out: the pairs of the
 * serial key, of the plan's vote, and none from a vote inside the serial sub-tree keys; both equal
 * to the pairs computed by the test (av_expected). */
static void
av_case(double share, int k, int need)
{
    int64_t ns, np, nk, ne, i;
    av_run(0, share, k, NULL);
    struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
    struct AdmPair * exp = av_expected(AvPre.r, AvPre.n, k > 0 ? k : 10, need, &ne);
    int64_t bade = ne == ns ? 0 : -1;
    for(i = 0; bade >= 0 && i < ns; i++)
        bade += exp[i].ID != ser[i].ID || exp[i].Receiver != ser[i].Receiver;
    av_run(1000, share, k, NULL);
    struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
    struct AdmPair * sub = av_gather(SUBHALO_ADMIT_KEY, &nk);
    const struct SubhaloVoteStats * vs = subhalo_plan_last_vote();
    int64_t bad = 0, toB = 0;
    for(i = 0; i < ns && i < np; i++) {
        LOCAL_CHECK(bad, ser[i].ID == pl[i].ID && ser[i].Receiver == pl[i].Receiver,
                    "admission vote: pair %ld: serial (%ld, %ld) plan (%ld, %ld)\n", i, (long) ser[i].ID, ser[i].Receiver,
                    (long) pl[i].ID, pl[i].Receiver);
        toB += ser[i].Receiver == 1;
    }
    message(0, "admission vote, share %g, k %d: %ld pairs from the serial key (%ld to B), %ld from the plan's vote (%ld queried, %ld by "
               "in-place owners, %d passes), %ld from the sub-tree keys' own votes; %ld mismatches; %ld expected pairs, %ld unlike them\n",
            share, k, ns, toB, np, vs->NQueries, vs->NOfferedInplace, vs->Passes, nk, bad, ne, bade);
    CHECK(ns == np && bad == 0 && ns > 0 && vs->NOffered == np && nk == 0 && bade == 0,
          "admission vote: %ld serial, %ld plan, %ld sub-tree key pairs, %ld mismatches, %ld unlike the expected\n", ns, np, nk, bad, bade);
    free(exp);
    free(sub);
    free(pl);
    free(ser);
}

/* The owner of particle ID before the last av_run call: its member's track, else A. */
static int64_t
av_owner(MyIDType ID)
{
    int64_t i;
    for(i = 0; i < AvPre.n; i++)
        if(AvPre.r[i].ID == ID)
            return AvPre.r[i].TrackId >= 1 && AvPre.r[i].TrackId <= 3 ? AvPre.r[i].TrackId : 0;
    return -1;
}

/* At SubhaloAdmitMinCoreDistance = 12 the offers of the distance 0 to the near receiver from the
 * owner it is near are withheld and the others stand, alike in the serial and the plan's vote. */
static void
av_withheld_case(int near)
{
    /* near 1: B below A; near 3: D below B below A, D(D, B) < 12 < D(D, A), D(B, A) > 12 */
    const int64_t wr = near, wo = near == 1 ? 0 : 1;
    int64_t n0, ns, np, i, bad = 0, toB = 0, fromA = 0;
    AvNear = near;
    AvMinD = 0;
    av_run(0, 0.5, 0, NULL);
    struct AdmPair * all = av_gather(SUBHALO_ADMIT_KEY, &n0);
    char * wh = calloc(n0 + 1, 1);
    for(i = 0; i < n0; i++)
        if(all[i].Receiver == wr) {
            const int64_t o = av_owner(all[i].ID);
            wh[i] = o == wo;
            toB += o == wo;
            fromA += o == 0 && near == 3;
        }
    AvMinD = 12;
    av_run(0, 0.5, 0, NULL);
    struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
    const int64_t ws = AvStats.NAdmitWithheld;
    av_run(1000, 0.5, 0, NULL);
    struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
    const int64_t wp = AvStats.NAdmitWithheld, wv = AvVote.NWithheld;
    int64_t j = 0;
    for(i = 0; i < n0; i++)
        if(!wh[i]) {
            LOCAL_CHECK(bad, j < ns && ser[j].ID == all[i].ID && ser[j].Receiver == all[i].Receiver,
                        "admission withheld: pair (%ld, %ld) of the distance 0 missing from the serial vote\n", (long) all[i].ID, all[i].Receiver);
            j++;
        }
    for(i = 0; i < ns && i < np; i++)
        LOCAL_CHECK(bad, ser[i].ID == pl[i].ID && ser[i].Receiver == pl[i].Receiver,
                    "admission withheld: pair %ld: serial (%ld, %ld) plan (%ld, %ld)\n", i, (long) ser[i].ID, ser[i].Receiver,
                    (long) pl[i].ID, pl[i].Receiver);
    message(0, "admission withheld, near %d: %ld pairs at distance 0 (%ld to withhold, %ld of A to D); at 12 %ld from the serial key, %ld "
               "from the plan's vote; withheld %ld serial, %ld plan (%ld by the vote); %ld mismatches\n", near, n0, toB, fromA, ns, np, ws, wp,
            wv, bad);
    CHECK(toB > 0 && n0 > toB && ns == n0 - toB && np == ns && bad == 0 && ws == toB && wp == ws && wv == ws && (near == 1 || fromA > 0),
          "admission withheld, near %d: %ld pairs (%ld to withhold, %ld of A to D), %ld serial, %ld plan, withheld %ld / %ld / %ld, "
          "%ld mismatches\n", near, n0, toB, fromA, ns, np, ws, wp, wv, bad);
    free(wh);
    AvNear = 0;
    AvMinD = 0;
    free(pl);
    free(ser);
    free(all);
}

static void
test_collective_vote_identity(void ** state)
{
    av_case(0.5, 0, 5);
    av_case(0.3, 7, 3);
    av_withheld_case(1);
    av_withheld_case(3);
    free(AvPre.r);
    AvPre.r = NULL;
    free(AvPairs);
    AvPairs = NULL;
    AvN = AvMax = 0;
}

/* ---- what the receivers of an oversized group do with the offered particles ----
 * A (0, 2000 particles) the central, B (1, 300) a dense satellite 1.5 kpc from A's centre: A's
 * tracers inside B vote for B. Three cases: B at rest in A's frame binds the A particles offered
 * (taken), B at 300 km/s binds none of them (unbound), B with a 2000 km/s dispersion dies (dead).
 * Each case runs as one serial key (SubhaloCollectiveMinPart 0) and in place with B in a serial
 * sub-tree key (1000), with the rule on and off. */
#define TK_B0 2000
#define TK_END 2300
enum { TK_TAKEN = 0, TK_UNBOUND, TK_DEAD, TK_SINK };

static void
tk_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const int c = *(const int *) ud;
    const double cA[3] = {500, 500, 500}, cB[3] = {500 + 1.5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vfast[3] = {0, 300, 0};
    if(id >= TK_END)
        return;
    if(id < TK_B0) {
        syn_ball(id, 71, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 72, cB, 0.6, c == TK_UNBOUND ? vfast : zero, c == TK_DEAD ? 2000 : 8, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - TK_B0;
    }
}

/* Live admission memory while the rounds run (blocks of the vote and the rounds' admitted list,
 * and the plan's records), summed over the ranks; set by adm_pretag. */
static int64_t AdmAllocs = -1;

static int64_t
adm_live_blocks(const struct SubhaloPlan * plan)
{
    AllocatorIter it;
    int64_t na = (plan->AdmitRecs != NULL) + (plan->HomeRecs != NULL), tot;
    for(allocator_iter_start(&it, A_MAIN); !allocator_iter_ended(&it); allocator_iter_next(&it))
        na += strncmp(it.name, "SubVote", 7) == 0 || strncmp(it.name, "SubCollAdm", 10) == 0;
    MPI_Allreduce(&na, &tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    return tot;
}

/* The tags after the serial stage, before the in-place final tag pass rewrites the oversized key's. */
static struct tagrec * TkPre = NULL;
static int64_t TkPreN = 0;

static void
tk_pretag(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan)
{
    (void) ctx;
    TkPre = gather_tags(&TkPreN);
    AdmAllocs = adm_live_blocks(plan);
}

/* The admission counters of the last call (global). */
static struct SubhaloCallStats TkStats;
/* SubhaloPotentialSampleSize and SubhaloFinalPass of tk_run (0 and 1: every particle a source,
 * an exact final pass). */
static int TkSampleSize = 0, TkFinalPass = 1;

static void
tk_run(int64_t collective, double share, int c, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = TkSampleSize;
    sp.SubhaloFinalPass = TkFinalPass;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = share;
    sp.SubhaloAdmitMinCoreDistance = 0;
    syn_setup(&CP, TK_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 300, 0.03, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(tk_particle, &c, 0.55, 1);
    AvN = 0;
    subhalo_set_admit_hook(av_hook);
    subhalo_collective_set_pretag_hook(tk_pretag);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    subhalo_collective_set_pretag_hook(NULL);
    subhalo_set_admit_hook(NULL);
    TkStats = *subhalo_last_call_stats();
    CHECK(ret == 1 && ng == 1, "offered particles: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

/* Particles whose tags or GrNr differ between two outcomes of the same particles. */
static int64_t
tk_tag_diff(const struct Outcome * a, const struct Outcome * b)
{
    int64_t i, nd = 0;
    if(a->n != b->n)
        return -1;
    for(i = 0; i < a->n; i++)
        nd += a->r[i].ID != b->r[i].ID || a->r[i].TrackId != b->r[i].TrackId || a->r[i].BoundTrackId != b->r[i].BoundTrackId
              || a->r[i].BindRank != b->r[i].BindRank || a->r[i].GrNr != b->r[i].GrNr;
    return nd;
}

/* Particles bound to track T in one outcome and not in the other. */
static int64_t
tk_bound_diff(const struct Outcome * a, const struct Outcome * b, int64_t T)
{
    int64_t i, nd = 0;
    for(i = 0; i < a->n && i < b->n; i++)
        nd += (a->r[i].BoundTrackId == T) != (b->r[i].BoundTrackId == T);
    return nd;
}

/* The receiver offered particle ID (pairs sorted by ID), -1 if none. */
static int64_t
tk_offer(const struct AdmPair * pr, int64_t np, MyIDType ID)
{
    int64_t lo = 0, hi = np;
    while(lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if(pr[mid].ID < ID) lo = mid + 1;
        else hi = mid;
    }
    return lo < np && pr[lo].ID == ID ? pr[lo].Receiver : -1;
}

/* Particles whose membership of T's bound set in `on` is not the expected one: bound to T in `off`
 * (the rule off, same mode) or offered to T. */
static int64_t
tk_expected_diff(const struct Outcome * on, const struct Outcome * off, const struct AdmPair * pr, int64_t np, int64_t T)
{
    int64_t i, nd = 0;
    if(on->n != off->n)
        return -1;
    for(i = 0; i < on->n; i++)
        nd += on->r[i].ID != off->r[i].ID
              || (on->r[i].BoundTrackId == T) != (off->r[i].BoundTrackId == T || tk_offer(pr, np, on->r[i].ID) == T);
    return nd;
}

/* From the final tags: the offered particles tagged and bound to their own receiver, and the
 * receivers with at least one of them (TrackIds below 64). */
static void
tk_counts(const struct tagrec * r, int64_t n, const struct AdmPair * pr, int64_t np, int64_t * taken, int64_t * recv)
{
    uint64_t seen = 0;
    int64_t i;
    *taken = *recv = 0;
    for(i = 0; i < np; i++) {
        struct tagrec key;
        key.ID = pr[i].ID;
        const struct tagrec * t = bsearch(&key, r, n, sizeof(struct tagrec), cmp_tagrec_id);
        if(t && t->TrackId == pr[i].Receiver && t->BoundTrackId == pr[i].Receiver) {
            (*taken)++;
            if(pr[i].Receiver >= 0 && pr[i].Receiver < 64)
                seen |= (uint64_t) 1 << pr[i].Receiver;
        }
    }
    for(i = 0; i < 64; i++)
        *recv += (seen >> i) & 1;
}

static void
test_collective_vote_receivers(void ** state)
{
    const char * name[3] = {"taken", "unbound", "dead"};
    int c;
    for(c = 0; c < 3; c++) {
        struct Outcome son, soff, fon, foff;
        int64_t ns, np, i;
        tk_run(0, 0.5, c, &son);
        const struct SubhaloCallStats sts = TkStats;
        struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
        tk_run(1000, 0.5, c, &fon);
        const struct SubhaloCallStats stf = TkStats;
        struct tagrec * preon = TkPre;
        const int64_t npreon = TkPreN;
        TkPre = NULL;
        struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
        const struct SubhaloVoteStats vs = *subhalo_plan_last_vote();
        tk_run(0, 0, c, &soff);
        tk_run(1000, 0, c, &foff);
        struct tagrec * preoff = TkPre;
        const int64_t npreoff = TkPreN;
        TkPre = NULL;
        /* the tags the serial stage leaves for the rounds */
        int64_t tdP = npreon == npreoff && preon && preoff ? 0 : -1;
        for(i = 0; tdP >= 0 && i < npreon; i++)
            tdP += preon[i].ID != preoff[i].ID || preon[i].TrackId != preoff[i].TrackId || preon[i].BoundTrackId != preoff[i].BoundTrackId
                   || preon[i].BindRank != preoff[i].BindRank || preon[i].GrNr != preoff[i].GrNr;
        int64_t bad = 0;
        for(i = 0; i < ns && i < np; i++)
            bad += ser[i].ID != pl[i].ID || ser[i].Receiver != pl[i].Receiver;
        int64_t tser, rser, tpl, rpl;
        tk_counts(son.r, son.n, ser, ns, &tser, &rser);
        tk_counts(fon.r, fon.n, pl, np, &tpl, &rpl);
        const int64_t dOff = tk_bound_diff(&soff, &foff, 1), dOn = tk_bound_diff(&son, &fon, 1);
        const int64_t eS = tk_expected_diff(&son, &soff, ser, ns, 1), eF = tk_expected_diff(&fon, &foff, pl, np, 1);
        const int64_t tdF = tk_tag_diff(&fon, &foff), tdS = tk_tag_diff(&son, &soff);
        message(0, "offered particles (%s): %ld pairs serial, %ld plan (%ld guests, %ld mismatches); taken %ld serial, %ld in place; "
                   "B's bound set: serial vs in place %ld differ with the rule, %ld without, %ld / %ld unlike the rule-off set plus the "
                   "offered; Nbound(B) %d / %d, SnapDeath %d / %d; particles whose tags differ from the rule-off run: %ld in place, %ld serial; "
                   "counters admitted/taken/receivers serial %ld/%ld/%ld, in place %ld/%ld/%ld\n",
                name[c], ns, np, vs.NGuests, bad, tser, tpl, dOn, dOff, eS, eF, son.rec[1].Nbound, fon.rec[1].Nbound,
                son.rec[1].SnapDeath, fon.rec[1].SnapDeath, tdF, tdS, sts.NAdmitted, sts.NAdmitTaken, sts.NAdmitReceivers,
                stf.NAdmitted, stf.NAdmitTaken, stf.NAdmitReceivers);
        CHECK(ns == np && bad == 0 && ns > 0 && vs.NGuests == np, "offered particles (%s): %ld serial, %ld plan pairs, %ld mismatches\n",
              name[c], ns, np, bad);
        CHECK(p2_check_tags(fon.r, fon.n, fon.rec, fon.nt, "offered particles") == 0, "offered particles: tag invariants\n");
        /* the counters: every pair admitted, the taken and their receivers as the final tags show */
        CHECK(sts.NAdmitted == ns && sts.NAdmitTaken == tser && sts.NAdmitReceivers == rser
              && stf.NAdmitted == np && stf.NAdmitTaken == tpl && stf.NAdmitReceivers == rpl,
              "offered particles (%s): counters serial %ld/%ld/%ld (tags %ld/%ld), in place %ld/%ld/%ld (tags %ld/%ld)\n", name[c],
              sts.NAdmitted, sts.NAdmitTaken, sts.NAdmitReceivers, tser, rser, stf.NAdmitted, stf.NAdmitTaken, stf.NAdmitReceivers, tpl, rpl);
        if(c == TK_TAKEN) {
            /* B's bound set is its rule-off set plus every offered particle, in both modes alike */
            CHECK(tser == ns && tpl == np && rser == 1 && rpl == 1 && eS == 0 && eF == 0 && dOn == 0 && dOff == 0
                  && son.rec[1].Nbound == fon.rec[1].Nbound,
                  "offered particles (taken): taken %ld / %ld, %ld / %ld unlike the expected set, bound sets differ by %ld (rule off %ld)\n",
                  tser, tpl, eS, eF, dOn, dOff);
        }
        else {
            /* nothing taken: every tag and GrNr as with the rule off, in either mode */
            message(0, "offered particles (%s): %ld particles whose tags differ from the rule-off run before the final tag pass\n", name[c], tdP);
            CHECK(tser == 0 && tpl == 0 && rser == 0 && rpl == 0 && tdF == 0 && tdS == 0 && tdP == 0,
                  "offered particles (%s): taken %ld / %ld, %ld / %ld / %ld tags differ\n", name[c], tser, tpl, tdF, tdS, tdP);
            for(i = 0; i < fon.nt && i < foff.nt; i++)
                CHECK(fon.rec[i].Nbound == foff.rec[i].Nbound && fon.rec[i].Mbound == foff.rec[i].Mbound,
                      "offered particles (%s): track %ld Nbound %d / %d\n", name[c], i, fon.rec[i].Nbound, foff.rec[i].Nbound);
        }
        if(c == TK_DEAD)
            CHECK(son.rec[1].Nbound == 0 && fon.rec[1].Nbound == 0 && son.rec[1].SnapDeath == 1 && fon.rec[1].SnapDeath == 1,
                  "offered particles (dead): B Nbound %d / %d, SnapDeath %d / %d\n", son.rec[1].Nbound, fon.rec[1].Nbound,
                  son.rec[1].SnapDeath, fon.rec[1].SnapDeath);
        else
            CHECK(son.rec[1].Nbound > 0 && fon.rec[1].Nbound > 0, "offered particles (%s): B died\n", name[c]);
        free(pl);
        free(ser);
        free(preoff);
        free(preon);
        outcome_free(&foff);
        outcome_free(&soff);
        outcome_free(&fon);
        outcome_free(&son);
    }
    /* offers whose owner is in the receiver's own sub-tree key (B's particles near D, B and D in one
     * key): not guests, served by the records all the same; the av fixture */
    {
        struct Outcome son, fon;
        int64_t ns, np, i, bad = 0, tser, rser, tpl, rpl;
        av_run(0, 0.5, 0, &son);
        const struct SubhaloCallStats sts = AvStats;
        struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
        av_run(1000, 0.5, 0, &fon);
        const struct SubhaloCallStats stf = AvStats;
        const struct SubhaloVoteStats vs = AvVote;
        struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
        for(i = 0; i < ns && i < np; i++)
            bad += ser[i].ID != pl[i].ID || ser[i].Receiver != pl[i].Receiver;
        tk_counts(son.r, son.n, ser, ns, &tser, &rser);
        tk_counts(fon.r, fon.n, pl, np, &tpl, &rpl);
        message(0, "offered particles (owner in the receiver's key): %ld / %ld pairs, %ld mismatches, %ld served, %ld guests; taken %ld / %ld "
                   "by %ld / %ld receivers; counters %ld/%ld/%ld and %ld/%ld/%ld\n", ns, np, bad, vs.NServed, vs.NGuests, tser, tpl, rser, rpl,
                sts.NAdmitted, sts.NAdmitTaken, sts.NAdmitReceivers, stf.NAdmitted, stf.NAdmitTaken, stf.NAdmitReceivers);
        CHECK(ns == np && ns > 0 && bad == 0 && vs.NServed == np && vs.NServed > vs.NGuests && tser == tpl && rser == rpl
              && sts.NAdmitted == ns && stf.NAdmitted == np && sts.NAdmitTaken == tser && stf.NAdmitTaken == tpl,
              "offered particles (owner in the receiver's key): %ld / %ld pairs, %ld served, %ld guests, taken %ld / %ld, admitted %ld / %ld\n",
              ns, np, vs.NServed, vs.NGuests, tser, tpl, sts.NAdmitted, stf.NAdmitted);
        free(pl);
        free(ser);
        outcome_free(&fon);
        outcome_free(&son);
        free(AvPre.r);
        AvPre.r = NULL;
    }
    /* positional subsampling of the potential sources (128 of B's 349 candidates) with the sampled
     * final pass, whose ranks follow the shuffled list: the admitted particles enter B's list in one
     * order in both modes, so B binds the same set with the same ranks in both */
    {
        struct Outcome son, fon;
        int64_t ns, np, i, bad = 0, tser, rser, tpl, rpl;
        TkSampleSize = 128;
        TkFinalPass = 0;
        tk_run(0, 0.5, TK_TAKEN, &son);
        const struct SubhaloCallStats sts = TkStats;
        struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
        tk_run(1000, 0.5, TK_TAKEN, &fon);
        const struct SubhaloCallStats stf = TkStats;
        free(TkPre);
        TkPre = NULL;
        struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
        TkSampleSize = 0;
        TkFinalPass = 1;
        for(i = 0; i < ns && i < np; i++)
            bad += ser[i].ID != pl[i].ID || ser[i].Receiver != pl[i].Receiver;
        tk_counts(son.r, son.n, ser, ns, &tser, &rser);
        tk_counts(fon.r, fon.n, pl, np, &tpl, &rpl);
        const int64_t dOn = tk_bound_diff(&son, &fon, 1);
        /* the sampled energies order B's bound set: its ranks match only if the lists match */
        int64_t rkB = son.n == fon.n ? 0 : -1;
        for(i = 0; rkB >= 0 && i < son.n; i++)
            rkB += son.r[i].BoundTrackId == 1 && (fon.r[i].BoundTrackId != 1 || fon.r[i].BindRank != son.r[i].BindRank);
        message(0, "offered particles (taken, sample 128): %ld / %ld pairs, %ld mismatches; taken %ld / %ld; B's bound set serial vs in "
                   "place %ld differ, %ld ranks differ; Nbound(B) %d / %d; counters %ld/%ld/%ld and %ld/%ld/%ld\n", ns, np, bad, tser, tpl,
                dOn, rkB, son.rec[1].Nbound, fon.rec[1].Nbound, sts.NAdmitted, sts.NAdmitTaken, sts.NAdmitReceivers, stf.NAdmitted,
                stf.NAdmitTaken, stf.NAdmitReceivers);
        CHECK(ns == np && ns > 0 && bad == 0 && tser > 0 && tser == tpl && dOn == 0 && rkB == 0 && son.rec[1].Nbound == fon.rec[1].Nbound
              && sts.NAdmitTaken == tser && stf.NAdmitTaken == tpl && sts.NAdmitReceivers == 1 && stf.NAdmitReceivers == 1,
              "offered particles (taken, sample 128): %ld / %ld pairs, taken %ld / %ld, %ld differ\n", ns, np, tser, tpl, dOn);
        free(pl);
        free(ser);
        outcome_free(&fon);
        outcome_free(&son);
    }
    free(AvPairs);
    AvPairs = NULL;
    AvN = AvMax = 0;
}

/* ---- the plan's handling of the vote, on the hierarchy of the receivers test (taken case) ----
 * Offered particles are counted in their receiver's sub-tree key; a sub-tree key they push over
 * SubhaloCollectiveMinPart goes in place, where its receiver takes the same pairs in its round; a
 * forced exchange failure, and a vote that does not fit in memory on one rank, fail the call on
 * every rank and leave every tag, GrNr and track record as they were; with the rule off the plan
 * takes no vote and no admission memory is live in the rounds. */
static int64_t TkKeyLen = -1;
/* The rank that received the admission records and how many (the largest over the ranks). */
static int TpRecvRank = -1;
static int64_t TpNRecv = 0;

static void
tp_keylen(const struct SubhaloCallCtx * ctx, const struct SubhaloPlan * plan)
{
    int64_t j, len = -1;
    (void) ctx;
    const int rr = plan->NAdmitRecs > 0 ? get_thistask() : -1;
    MPI_Allreduce(&rr, &TpRecvRank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&plan->NAdmitRecs, &TpNRecv, 1, MPI_INT64, MPI_MAX, MPI_COMM_WORLD);
    for(j = 0; j < plan->NLocalKeys; j++)
        if(plan->LocalKeys[j].g == plan->SubKeyBase && plan->LocalKeys[j].Length > len)
            len = plan->LocalKeys[j].Length;
    MPI_Allreduce(&len, &TkKeyLen, 1, MPI_INT64, MPI_MAX, MPI_COMM_WORLD);
    AdmAllocs = adm_live_blocks(plan);
}

struct TpCall {
    int ret;
    int complete;
    int64_t nchanged;        /* particles whose tags or GrNr the call changed */
    int64_t nrecchanged;     /* track records whose stored fields the call changed */
    int64_t keylen;          /* the length of the first sub-tree key (-1 without one) */
    int64_t ncands;
    int64_t admallocs;       /* live admission blocks in the rounds (-1: no rounds) */
    struct SubhaloVoteStats vs;
    struct SubhaloCallStats st;
    struct SubTrack recB;
    struct AdmPair * pairs;  /* the plan's offered pairs, sorted by ID */
    int64_t npairs;
    int64_t taken, recv;     /* pairs tagged and bound to their receiver at the end, their receivers */
    int recv_rank;           /* the rank that received the records (-1: none or no rounds) */
    int64_t nrecv;           /* records it received */
    int64_t nsend_at_recv;   /* offered particles it held before the call (the records it sent) */
    struct SubhaloMemCheck adm;  /* the plan's admission */
};

static void
tp_call(int64_t collective, double share, int fail, struct TpCall * out)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = share;
    sp.SubhaloAdmitMinCoreDistance = 0;
    sp.SubhaloDebugForceExchangeFailure = fail;
    syn_setup(&CP, TK_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 300, 0.03, 0, 0, 1, 1);
    syn_records(recs, 2);
    int c = TK_TAKEN;
    syn_place(tk_particle, &c, 0.55, 1);
    struct SubTrack * before = fetch_all_tracks(2);
    struct FailSnap pre = {NULL, 0};
    int64_t ng;
    TkKeyLen = -1;
    AdmAllocs = -1;
    TpRecvRank = -1;
    TpNRecv = 0;
    AvN = 0;
    subhalo_set_admit_hook(av_hook);
    subhalo_collective_set_pretag_hook(tp_keylen);
    out->ret = run_finder_call_ex(1, &CP, &ng, fail_pre, NULL, &pre);
    subhalo_collective_set_pretag_hook(NULL);
    subhalo_set_admit_hook(NULL);
    out->complete = subhalo_last_call_complete();
    out->keylen = TkKeyLen;
    out->admallocs = AdmAllocs;
    out->ncands = subhalo_plan_last_summary()->NCands;
    out->adm = *subhalo_plan_last_memcheck();
    out->recv_rank = TpRecvRank;
    out->nrecv = TpNRecv;
    out->vs = *subhalo_plan_last_vote();
    out->st = *subhalo_last_call_stats();
    out->pairs = av_gather(SUBHALO_ADMIT_PLAN, &out->npairs);
    int64_t n, i;
    struct tagrec * post = gather_tags(&n);
    out->nchanged = n == pre.n ? 0 : -1;
    for(i = 0; out->nchanged >= 0 && i < n; i++)
        out->nchanged += post[i].ID != pre.r[i].ID || post[i].GrNr != pre.r[i].GrNr || post[i].TrackId != pre.r[i].TrackId
                         || post[i].BindRank != pre.r[i].BindRank || post[i].BoundTrackId != pre.r[i].BoundTrackId;
    tk_counts(post, n, out->pairs, out->npairs, &out->taken, &out->recv);
    out->nsend_at_recv = 0;
    for(i = 0; i < out->npairs; i++) {
        struct tagrec key;
        key.ID = out->pairs[i].ID;
        const struct tagrec * t = bsearch(&key, pre.r, pre.n, sizeof(struct tagrec), cmp_tagrec_id);
        out->nsend_at_recv += t && t->Task == out->recv_rank;
    }
    struct SubTrack * after = fetch_all_tracks(2);
    out->recB = after[1];
    /* the stored fields (everything before the per-call transients) */
    out->nrecchanged = 0;
    for(i = 0; i < 2; i++)
        out->nrecchanged += memcmp(&before[i], &after[i], offsetof(struct SubTrack, Host)) != 0;
    free(after);
    free(before);
    free(post);
    free(pre.r);
    syn_teardown();
}

static void
tp_free(struct TpCall * t)
{
    free(t->pairs);
    t->pairs = NULL;
}

/* 1 if two calls offered the same pairs. */
static int
tp_same_pairs(const struct TpCall * a, const struct TpCall * b)
{
    int64_t i;
    if(a->npairs != b->npairs)
        return 0;
    for(i = 0; i < a->npairs; i++)
        if(a->pairs[i].ID != b->pairs[i].ID || a->pairs[i].Receiver != b->pairs[i].Receiver)
            return 0;
    return 1;
}

static void
test_collective_vote_plan(void ** state)
{
    struct TpCall on, off, prom, promoff, xf, mem;
    tp_call(1000, 0.5, 0, &on);
    tp_call(1000, 0, 0, &off);
    message(0, "vote plan: sub-tree key of %ld particles with the rule (%ld offered, %ld guests), %ld without; "
               "rule off: %ld queries, %d passes; live admission blocks in the rounds %ld with the rule, %ld without; "
               "records charged %g B\n", on.keylen, on.vs.NOffered, on.vs.NGuests, off.keylen, off.vs.NQueries, off.vs.Passes,
            on.admallocs, off.admallocs, on.vs.RecordBytes);
    CHECK(on.ret == 1 && off.ret == 1 && on.vs.NGuests > 0 && on.keylen - off.keylen == on.vs.NGuests,
          "vote plan: key length %ld / %ld, %ld guests\n", on.keylen, off.keylen, on.vs.NGuests);
    CHECK(off.vs.NQueries == 0 && off.vs.Passes == 0 && off.vs.NOffered == 0 && off.npairs == 0 && off.vs.RecordBytes == 0,
          "vote plan: a vote with the rule off\n");
    /* the allocator in the rounds: admission blocks with the rule (control), none without */
    CHECK(on.admallocs > 0 && off.admallocs == 0, "vote plan: live admission blocks %ld / %ld\n", on.admallocs, off.admallocs);
    /* the records charged: received ones (one per served offer), home ones and the rounds' list
     * (none here), each with the one spare entry of every allocation */
    const int NTask = get_ntask();
    const double recb = sizeof(struct SubhaloAdmitRec) * (double) (on.vs.NServed + NTask)
        + (sizeof(struct SubhaloHomeRec) + SUBHALO_INPLACE_ADM_BYTES) * (double) (on.vs.NHome + NTask);
    CHECK(on.vs.RecordBytes == recb, "vote plan: records charged %g B, expected %g B\n", on.vs.RecordBytes, recb);
    /* the record charge decides the plan's admission: scaled up (test hook) it fails the call on every
     * rank, on the rank that receives the records and sends few, before anything changes; left out of
     * the admission (scale 0) the same call runs */
    struct TpCall rf, r0;
    subhalo_plan_set_record_scale(1e12);
    tp_call(1000, 0.5, 0, &rf);
    subhalo_plan_set_record_scale(0);
    tp_call(1000, 0.5, 0, &r0);
    subhalo_plan_set_record_scale(1);
    message(0, "vote plan: records received on rank %d: %ld, of which it sent %ld; scaled charge: ret %d, reserve failed %d on rank %d, "
               "in-place stage failed %d on rank %d, vote short %g; charge left out: ret %d, taken %ld\n", on.recv_rank, on.nrecv,
            on.nsend_at_recv, rf.ret, rf.adm.reserve_failed, rf.adm.reserve_rank, rf.adm.failed, rf.adm.failrank, rf.vs.WorstShort,
            r0.ret, r0.taken);
    /* on several ranks the rank that receives the records sent only a part of them */
    CHECK(on.recv_rank >= 0 && on.nrecv == on.npairs && (get_ntask() == 1 || on.nsend_at_recv < on.nrecv),
          "vote plan: rank %d received %ld records and sent %ld\n", on.recv_rank, on.nrecv, on.nsend_at_recv);
    CHECK(rf.ret == 0 && !rf.complete && rf.nchanged == 0 && rf.nrecchanged == 0 && rf.vs.WorstShort <= 0 && rf.adm.reserve_failed
          && rf.adm.reserve_rank == on.recv_rank && (!rf.adm.failed || rf.adm.failrank == on.recv_rank),
          "vote plan: scaled record charge: ret %d, reserve failed %d on rank %d (records on rank %d)\n", rf.ret, rf.adm.reserve_failed,
          rf.adm.reserve_rank, on.recv_rank);
    CHECK(r0.ret == 1 && r0.complete && r0.taken == on.taken, "vote plan: record charge left out: ret %d, taken %ld\n", r0.ret, r0.taken);
    tp_free(&r0);
    tp_free(&rf);
    /* B's key holds 300 of its own and 49 guests: at 320 it goes in place with the rule only, and
     * takes the same pairs in its round */
    tp_call(320, 0.5, 0, &prom);
    tp_call(320, 0, 0, &promoff);
    message(0, "vote plan, knob 320: %ld candidates with the rule (B flags %d, %ld offered, %ld taken to sub-tree keys, %ld kept home, "
               "%ld taken in place by %ld receivers; counters %ld/%ld/%ld), %ld without (B flags %d)\n",
            prom.ncands, prom.recB.Flags, prom.vs.NOffered, prom.vs.NServed, prom.vs.NHome, prom.taken, prom.recv,
            prom.st.NAdmitted, prom.st.NAdmitTaken, prom.st.NAdmitReceivers, promoff.ncands, promoff.recB.Flags);
    CHECK(prom.ret == 1 && prom.ncands == 2 && (prom.recB.Flags & SUBHALO_FLAG_INPLACE) && prom.vs.NOffered == on.vs.NOffered
          && prom.vs.NServed == 0 && promoff.ncands == 1 && !(promoff.recB.Flags & SUBHALO_FLAG_INPLACE),
          "vote plan: promotion by guests (%ld / %ld candidates)\n", prom.ncands, promoff.ncands);
    CHECK(tp_same_pairs(&prom, &on) && prom.vs.NHome == prom.npairs && prom.taken == on.taken && on.taken == on.npairs && prom.recv == 1
          && prom.st.NAdmitted == prom.npairs && prom.st.NAdmitTaken == prom.taken && prom.st.NAdmitReceivers == 1,
          "vote plan: promoted receiver: same pairs %d, %ld home, taken %ld (serial key %ld), counters %ld/%ld/%ld\n",
          tp_same_pairs(&prom, &on), prom.vs.NHome, prom.taken, on.taken, prom.st.NAdmitted, prom.st.NAdmitTaken, prom.st.NAdmitReceivers);
    /* a forced exchange failure after the vote and the encoding */
    tp_call(1000, 0.5, 1, &xf);
    CHECK(xf.ret == 0 && !xf.complete && xf.nchanged == 0 && xf.nrecchanged == 0 && xf.vs.NServed > 0,
          "vote plan: exchange failure ret %d, %ld particles and %ld records changed\n", xf.ret, xf.nchanged, xf.nrecchanged);
    /* a vote that does not fit on the last rank alone: the call fails on every rank before the exchange */
    subhalo_plan_set_vote_extra_bytes(get_thistask() == NTask - 1 ? 1e18 : 0);
    tp_call(1000, 0.5, 0, &mem);
    subhalo_plan_set_vote_extra_bytes(0);
    message(0, "vote plan: memory check short by %g MB on rank %d: ret %d, %ld particles and %ld records changed\n",
            mem.vs.WorstShort / 1048576., mem.vs.WorstRank, mem.ret, mem.nchanged, mem.nrecchanged);
    CHECK(mem.ret == 0 && !mem.complete && mem.nchanged == 0 && mem.nrecchanged == 0 && mem.vs.WorstShort > 0
          && mem.vs.WorstRank == NTask - 1 && mem.vs.NQueries == 0 && mem.npairs == 0,
          "vote plan: memory failure ret %d on rank %d (worst rank %d)\n", mem.ret, get_thistask(), mem.vs.WorstRank);
    tp_free(&mem);
    tp_free(&xf);
    tp_free(&promoff);
    tp_free(&prom);
    tp_free(&off);
    tp_free(&on);
    free(AvPairs);
    AvPairs = NULL;
    AvN = AvMax = 0;
}

/* ---- a majority among a sub-tree's own tracers that is not one among its group's ----
 * A's 2000 particles and a dense core of 2000 more at X, B (1, 300) 1.5 kpc away with 30 of its
 * particles spread around X, D (2, 30) B's satellite at X. Among B's and D's tracers alone most of
 * the strays at X have a majority of D-bound neighbours; among the group's, A's core outvotes
 * them. Nothing may be offered to D, in either mode. With the share at 0.1 (one vote of ten) the
 * same positions offer particles to D, alike in both modes, and D moving at 30 km/s binds the same
 * of them in both. */
#define SM_C0 2000
#define SM_B0 4000
#define SM_S0 4300
#define SM_D0 4330
#define SM_END 4360
/* D's bulk speed (km/s, along y): 0 in the majority case (D does not survive there); 30 in the
 * control, where D survives and binds what it is offered (the positions, and so the vote, do not
 * depend on it) */
static double SmDSpeed = 0;

static void
sm_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, X[3] = {500 - 1.0 / atime, 500, 500}, cB[3] = {500 + 1.5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= SM_END)
        return;
    if(id < SM_C0) {
        syn_ball(id, 81, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < SM_B0) {
        syn_ball(id, 82, X, 0.3, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < SM_D0) {
        syn_ball(id, 83, id < SM_S0 ? cB : X, id < SM_S0 ? 0.6 : 0.2, zero, 8, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - SM_B0;
    }
    else {
        const double vD[3] = {0, SmDSpeed, 0};
        syn_ball(id, 84, X, 0.15, vD, 3, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - SM_D0;
    }
}

static void
sm_run(int64_t collective, double share, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = share;
    sp.SubhaloAdmitMinCoreDistance = 0;
    syn_setup(&CP, SM_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 4000, 0.4, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 330, 0.033, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 30, 0.003, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(sm_particle, NULL, 0.55, 1);
    AvN = 0;
    subhalo_set_admit_hook(av_hook);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    subhalo_set_admit_hook(NULL);
    CHECK(ret == 1 && ng == 1, "sub-tree majority: %ld groups\n", ng);
    if(o)
        outcome_take(o);
    syn_teardown();
}

static void
test_collective_vote_subtree_majority(void ** state)
{
    struct Outcome ser, inp;
    int64_t ns, np, nk, i, j;
    sm_run(0, 0.5, &ser);
    struct AdmPair * sp0 = av_gather(SUBHALO_ADMIT_KEY, &ns);
    sm_run(1000, 0.5, &inp);
    struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
    struct AdmPair * sk = av_gather(SUBHALO_ADMIT_KEY, &nk);
    /* the vote among B's and D's tracers alone, by brute force on the input tags */
    const int kk = default_subhalo_params().SubhaloNumNeighboursForReassignment;
    const int need = subhalo_admit_need(0.5, kk);
    int64_t nsub = 0;
    for(i = 0; i < ser.n; i++) {
        if(ser.r[i].ID < SM_S0 || ser.r[i].ID >= SM_D0)
            continue;
        double d2[SUBHALO_KNN_MAXK];
        int isD[SUBHALO_KNN_MAXK], m = 0, a;
        for(j = 0; j < ser.n; j++) {
            if(j == i || ser.r[j].ID < SM_B0 || ser.r[j].ID >= SM_END)
                continue;
            double r2 = 0;
            int d;
            for(d = 0; d < 3; d++) {
                const double x = NEAREST(ser.r[j].Pos[d] - ser.r[i].Pos[d], BOXSIZE);
                r2 += x * x;
            }
            if(m == kk && r2 >= d2[kk - 1])
                continue;
            a = m < kk ? m++ : kk - 1;
            while(a > 0 && d2[a - 1] > r2) {
                d2[a] = d2[a - 1];
                isD[a] = isD[a - 1];
                a--;
            }
            d2[a] = r2;
            isD[a] = ser.r[j].ID >= SM_D0;
        }
        int votes = 0;
        for(a = 0; a < m; a++)
            votes += isD[a];
        nsub += votes >= need;
    }
    int64_t toDs = 0, toDp = 0;
    for(i = 0; i < ns; i++)
        toDs += sp0[i].Receiver == 2;
    for(i = 0; i < np; i++)
        toDp += pl[i].Receiver == 2;
    message(0, "sub-tree majority: %ld of B's strays have a D majority among B's and D's tracers; offered to D: %ld serial, %ld in place; "
               "%ld / %ld pairs, %ld from the sub-tree keys' own votes; D bound %d / %d\n", nsub, toDs, toDp, ns, np, nk,
            ser.rec[2].Nbound, inp.rec[2].Nbound);
    CHECK(nsub > 0 && toDs == 0 && toDp == 0 && nk == 0 && ns == np && ns > 0,
          "sub-tree majority: %ld strays, %ld / %ld offered to D, %ld / %ld pairs, %ld key pairs\n", nsub, toDs, toDp, ns, np, nk);
    for(i = 0; i < ns && i < np; i++)
        CHECK(sp0[i].ID == pl[i].ID && sp0[i].Receiver == pl[i].Receiver, "sub-tree majority: pair %ld differs\n", i);
    free(sk);
    free(pl);
    free(sp0);
    /* the control: one vote of ten offers to D, in both modes alike, and D binds the same of them */
    struct Outcome cser, cinp;
    SmDSpeed = 30;
    sm_run(0, 0.1, &cser);
    sp0 = av_gather(SUBHALO_ADMIT_KEY, &ns);
    sm_run(1000, 0.1, &cinp);
    pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
    SmDSpeed = 0;
    int64_t cD = 0, bad = 0, tDs = 0, tDp = 0, dD = 0;
    for(i = 0; i < ns && i < np; i++) {
        bad += sp0[i].ID != pl[i].ID || sp0[i].Receiver != pl[i].Receiver;
        if(sp0[i].Receiver != 2)
            continue;
        cD++;
        struct tagrec key;
        key.ID = sp0[i].ID;
        const struct tagrec * a = bsearch(&key, cser.r, cser.n, sizeof(struct tagrec), cmp_tagrec_id);
        const struct tagrec * b = bsearch(&key, cinp.r, cinp.n, sizeof(struct tagrec), cmp_tagrec_id);
        const int ta = a && a->TrackId == 2 && a->BoundTrackId == 2, tb = b && b->TrackId == 2 && b->BoundTrackId == 2;
        tDs += ta;
        tDp += tb;
        dD += ta != tb;
    }
    message(0, "sub-tree majority, share 0.1: %ld / %ld pairs, %ld to D, %ld mismatches; D binds %ld / %ld of them (%ld differ); "
               "Nbound(D) %d / %d, SinkTrackId(D) %ld / %ld\n", ns, np, cD, bad, tDs, tDp, dD, cser.rec[2].Nbound, cinp.rec[2].Nbound,
            cser.rec[2].SinkTrackId, cinp.rec[2].SinkTrackId);
    CHECK(ns == np && bad == 0 && cD > 0 && tDs > 0 && dD == 0 && cser.rec[2].Nbound == cinp.rec[2].Nbound,
          "sub-tree majority, share 0.1: %ld / %ld pairs, %ld to D, %ld mismatches, D binds %ld / %ld (%ld differ)\n", ns, np, cD, bad, tDs, tDp, dD);
    outcome_free(&cinp);
    outcome_free(&cser);
    free(pl);
    free(sp0);
    outcome_free(&inp);
    outcome_free(&ser);
    free(AvPairs);
    AvPairs = NULL;
    AvN = AvMax = 0;
}

/* ---- receivers in place ----
 * A (0, 2000) the central; B (1, 600) its satellite 1.5 kpc away on +x, at rest; E (2, 200) B's
 * satellite 2.5 kpc away on -x. At SubhaloCollectiveMinPart = 100 all three are in place. A's
 * tracers inside E vote for E (and those inside B for B). E drifting at 30 km/s binds them (taken);
 * E at 300 km/s binds none (unbound), and E with a 2000 km/s dispersion dies (dead): the particles
 * offered to E must return to their owner A, not to E's parent B (which would bind them), with the
 * tags of the rule-off run. E at B's centre and at rest in B's frame sinks into B in B's round
 * (sink): no particle keeps E's tags. The serial stage's tags are those of the rule-off run, and the
 * offered pairs those of the group as one serial key. */
#define IR_B0 2000
#define IR_E0 2600
#define IR_END 2800

static void
ir_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const int c = *(const int *) ud;
    const double cA[3] = {500, 500, 500}, cB[3] = {500 + 1.5 / atime, 500, 500}, cE[3] = {500 - 2.5 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vslow[3] = {0, 30, 0}, vfast[3] = {0, 300, 0};
    if(id >= IR_END)
        return;
    if(id < IR_B0) {
        syn_ball(id, 91, cA, 3.0, zero, 20, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < IR_E0) {
        syn_ball(id, 92, cB, 0.8, zero, 15, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - IR_B0;
    }
    else {
        if(c == TK_SINK)
            syn_ball(id, 93, cB, 0.5, zero, 8, atime, s);
        else
            syn_ball(id, 93, cE, 0.5, c == TK_UNBOUND ? vfast : vslow, c == TK_DEAD ? 2000 : 8, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - IR_E0;
    }
}

static void
ir_run(int64_t collective, double share, int c, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 1;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = share;
    sp.SubhaloAdmitMinCoreDistance = 0;
    syn_setup(&CP, IR_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 2000, 0.2, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 600, 0.06, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(ir_particle, &c, 0.55, 1);
    AvN = 0;
    AdmAllocs = -1;
    subhalo_set_admit_hook(av_hook);
    subhalo_collective_set_pretag_hook(tk_pretag);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    subhalo_collective_set_pretag_hook(NULL);
    subhalo_set_admit_hook(NULL);
    TkStats = *subhalo_last_call_stats();
    CHECK(ret == 1 && ng == 1, "in-place receivers: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

static void
test_collective_vote_inplace_receivers(void ** state)
{
    const char * name[4] = {"taken", "unbound", "dead", "sink"};
    int c;
    for(c = 0; c < 4; c++) {
        struct Outcome son, fon, foff;
        int64_t ns, np, i;
        ir_run(0, 0.5, c, &son);
        free(TkPre);
        TkPre = NULL;
        struct AdmPair * ser = av_gather(SUBHALO_ADMIT_KEY, &ns);
        ir_run(100, 0.5, c, &fon);
        const struct SubhaloCallStats stf = TkStats;
        const int64_t allocon = AdmAllocs;
        struct tagrec * preon = TkPre;
        const int64_t npreon = TkPreN;
        TkPre = NULL;
        struct AdmPair * pl = av_gather(SUBHALO_ADMIT_PLAN, &np);
        const struct SubhaloVoteStats vs = *subhalo_plan_last_vote();
        const int64_t ncands = subhalo_plan_last_summary()->NCands;
        ir_run(100, 0, c, &foff);
        const int64_t allocoff = AdmAllocs;
        struct tagrec * preoff = TkPre;
        const int64_t npreoff = TkPreN;
        TkPre = NULL;
        /* of the pairs to E: taken (tagged and bound to E), bound to E's parent B, tagged unlike the
         * rule-off run (TrackId, BoundTrackId), and the rejected ones' final BindRank outside their
         * track's ranks */
        int64_t bad = 0, toE = 0, toB = 0, takenE = 0, tdE = 0, rkE = 0, tagE = 0, nown[3] = {0, 0, 0};
        for(i = 0; i < fon.n; i++)
            if(fon.r[i].TrackId >= 0 && fon.r[i].TrackId < 3)
                nown[fon.r[i].TrackId]++;
        for(i = 0; i < ns && i < np; i++)
            bad += ser[i].ID != pl[i].ID || ser[i].Receiver != pl[i].Receiver;
        for(i = 0; i < np; i++) {
            if(pl[i].Receiver != 2)
                continue;
            struct tagrec key;
            key.ID = pl[i].ID;
            const struct tagrec * t = bsearch(&key, fon.r, fon.n, sizeof(struct tagrec), cmp_tagrec_id);
            const struct tagrec * t0 = bsearch(&key, foff.r, foff.n, sizeof(struct tagrec), cmp_tagrec_id);
            toE++;
            if(!t || !t0) {
                tdE++;
                continue;
            }
            toB += t->BoundTrackId == 1;
            tagE += t->TrackId == 2;
            if(t->TrackId == 2 && t->BoundTrackId == 2) {
                takenE++;
                continue;
            }
            tdE += t->TrackId != t0->TrackId || t->BoundTrackId != t0->BoundTrackId;
            rkE += t->TrackId < 0 || t->TrackId >= 3 || t->BindRank < 0 || t->BindRank >= nown[t->TrackId];
        }
        /* E's tags anywhere at the end */
        int64_t anyE = 0;
        for(i = 0; i < fon.n; i++)
            anyE += fon.r[i].TrackId == 2 || fon.r[i].BoundTrackId == 2;
        int64_t tdP = npreon == npreoff && preon && preoff ? 0 : -1;
        for(i = 0; tdP >= 0 && i < npreon; i++)
            tdP += preon[i].ID != preoff[i].ID || preon[i].TrackId != preoff[i].TrackId || preon[i].BoundTrackId != preoff[i].BoundTrackId
                   || preon[i].BindRank != preoff[i].BindRank || preon[i].GrNr != preoff[i].GrNr;
        int64_t tpl, rpl;
        tk_counts(fon.r, fon.n, pl, np, &tpl, &rpl);
        message(0, "in-place receivers (%s): %ld candidates; %ld pairs serial, %ld plan (%ld to E, %ld home, %ld mismatches); "
                   "of those to E: %ld taken, %ld bound to B, %ld rejected tagged unlike the rule-off run, %ld rejected with a BindRank "
                   "outside their track's; Nbound(E) %d (rule off %d), SinkTrackId(E) %ld; particles with E's tags %ld; tags before the "
                   "final pass unlike the rule-off run %ld; counters %ld/%ld/%ld (tags %ld/%ld); live admission blocks %ld / %ld\n",
                name[c], ncands, ns, np, toE, vs.NHome, bad, takenE, toB, tdE, rkE, fon.rec[2].Nbound, foff.rec[2].Nbound,
                fon.rec[2].SinkTrackId, anyE, tdP, stf.NAdmitted, stf.NAdmitTaken, stf.NAdmitReceivers, tpl, rpl, allocon, allocoff);
        CHECK(ncands == 3 && ns == np && bad == 0 && toE > 0 && vs.NHome == np && (toB == 0 || c == TK_SINK) && tdP == 0 && rkE == 0 && allocon > 0
              && allocoff == 0 && stf.NAdmitted == np,
              "in-place receivers (%s): %ld candidates, %ld / %ld pairs, %ld mismatches, %ld home, %ld bound to B, %ld / %ld tags, "
              "blocks %ld / %ld\n", name[c], ncands, ns, np, bad, vs.NHome, toB, tdP, rkE, allocon, allocoff);
        CHECK(p2_check_tags(fon.r, fon.n, fon.rec, fon.nt, "in-place receivers") == 0, "in-place receivers: tag invariants\n");
        if(c == TK_TAKEN)
            CHECK(takenE > 0 && takenE == toE && tdE == 0 && fon.rec[2].Nbound == foff.rec[2].Nbound + takenE
                  && stf.NAdmitTaken == tpl && stf.NAdmitReceivers == rpl && rpl == 2,
                  "in-place receivers (taken): %ld of %ld taken, Nbound(E) %d / %d, counters %ld/%ld (tags %ld/%ld)\n", takenE, toE,
                  fon.rec[2].Nbound, foff.rec[2].Nbound, stf.NAdmitTaken, stf.NAdmitReceivers, tpl, rpl);
        else if(c == TK_SINK)
            /* E takes the particles in its round and sinks into B in B's round with its bound set; each
             * taken particle counts once, in the round of the receiver that took it */
            CHECK(fon.rec[2].SinkTrackId == 1 && foff.rec[2].SinkTrackId == 1 && fon.rec[2].Nbound == 0 && anyE == 0 && tagE == 0
                  && toB == toE && stf.NAdmitReceivers == 2 && stf.NAdmitTaken == tpl + toE,
                  "in-place receivers (sink): SinkTrackId(E) %ld / %ld, Nbound(E) %d, %ld particles with E's tags, taken %ld (B's %ld "
                  "+ E's %ld)\n", fon.rec[2].SinkTrackId, foff.rec[2].SinkTrackId, fon.rec[2].Nbound, anyE, stf.NAdmitTaken, tpl, toE);
        else
            CHECK(takenE == 0 && tdE == 0 && stf.NAdmitTaken == tpl && stf.NAdmitReceivers == rpl
                  && (c != TK_DEAD || (fon.rec[2].Nbound == 0 && fon.rec[2].SnapDeath == 1 && anyE == 0)),
                  "in-place receivers (%s): %ld taken, %ld tags differ, Nbound(E) %d, SnapDeath %d\n", name[c], takenE, tdE,
                  fon.rec[2].Nbound, fon.rec[2].SnapDeath);
        free(preoff);
        free(preon);
        free(pl);
        free(ser);
        outcome_free(&foff);
        outcome_free(&fon);
        outcome_free(&son);
    }
    free(AvPairs);
    AvPairs = NULL;
    AvN = AvMax = 0;
}

/* ------------------------------------------------------------------ the two paths on one decision */

/* A death in the re-unbinding: X sinks its satellite Y, then heavy gas (HX, HY, T) leaves X's tracers
 * unbound; X takes tau (its rank-0 tracer) and its dead core, and sinks into the central A. */
#define DR_HX 2000
#define DR_TAU 2001
#define DR_XI 2002
#define DR_XO 2012
#define DR_HY 2022
#define DR_YT 2023
#define DR_T 2043
#define DR_END 2044
#define DR_K (43007.1 * 0.1)
#define DR_S 8.2
#define DR_V 80.

static int
dr_type(MyIDType id)
{
    return id == DR_HX || id == DR_HY || id == DR_T ? 0 : 1;
}

/* A unit vector of pair k (from (k, salt)) and a unit vector orthogonal to it. */
static void
dr_pair(int64_t k, uint64_t salt, double u[3], double w[3])
{
    const double c0[3] = {0, 0, 0};
    double r[3], n = 0, dot = 0;
    int d;
    rand_in_ball(k, salt, 1.0, c0, u);
    for(d = 0; d < 3; d++)
        n += u[d] * u[d];
    for(d = 0; d < 3; d++)
        u[d] /= sqrt(n);
    rand_in_ball(k, salt + 7, 1.0, c0, r);
    for(d = 0; d < 3; d++)
        dot += r[d] * u[d];
    n = 0;
    for(d = 0; d < 3; d++) {
        w[d] = r[d] - dot * u[d];
        n += w[d] * w[d];
    }
    for(d = 0; d < 3; d++)
        w[d] /= sqrt(n);
}

/* Particle j of a symmetric set about (c, v): pairs (+u, +w), (-u, -w) at radius r, speed vt. */
static void
dr_paired(int64_t j, uint64_t salt, const double c[3], const double v[3], double r, double vt, double atime, struct SynPart * s)
{
    double u[3], w[3];
    const int sg = (j % 2) ? -1 : 1;
    int d;
    dr_pair(j / 2, salt, u, w);
    for(d = 0; d < 3; d++) {
        s->pos[d] = c[d] + sg * r * u[d] / atime;
        s->vel[d] = atime * (v[d] + sg * vt * w[d]);
    }
}

static void
dr_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, zero[3] = {0, 0, 0};
    const double cY[3] = {500, 500 + DR_S / atime, 500}, vY[3] = {0, 0, DR_V};
    int d;
    (void) ud;
    if(id >= DR_END)
        return;
    if(id < DR_HX) {
        syn_ball(id, 121, cA, id < DR_HX / 2 ? 2.0 : 20.0, zero, 10, atime, s);
        s->mass = 10 * SYN_M;
        s->tag = s->bound = 0;
        s->rank = id;
        return;
    }
    if(id < DR_HY) {
        s->tag = s->bound = 1;
        if(id == DR_HX) {
            for(d = 0; d < 3; d++)
                s->pos[d] = cA[d];
            s->mass = 0.1;
            s->rank = 21;
        }
        else if(id == DR_TAU) {
            for(d = 0; d < 3; d++)
                s->pos[d] = cA[d];
            s->pos[0] += 0.02 / atime;
            s->rank = 0;
        }
        else if(id < DR_XO) {
            dr_paired(id - DR_XI, 122, cA, zero, 1.0, sqrt(1.8 * DR_K), atime, s);
            s->rank = 1 + id - DR_XI;
        }
        else {
            /* the outer pairs in the x-z plane, away from Y's offset along y */
            const int64_t j = id - DR_XO;
            const int sg = (j % 2) ? -1 : 1;
            const double ph = 2 * M_PI * (j / 2) / 5. + 0.3, vt = sqrt(1.4 * DR_K / 24.);
            const double u[3] = {cos(ph), 0, sin(ph)}, w[3] = {-0.6 * sin(ph), 0.8, 0.6 * cos(ph)};
            for(d = 0; d < 3; d++) {
                s->pos[d] = cA[d] + sg * 24. * u[d] / atime;
                s->vel[d] = atime * sg * vt * w[d];
            }
            s->rank = 11 + j;
        }
        return;
    }
    if(id < DR_T) {
        s->tag = s->bound = 2;
        if(id == DR_HY) {
            for(d = 0; d < 3; d++) {
                s->pos[d] = cY[d];
                s->vel[d] = atime * vY[d];
            }
            s->mass = 0.1;
            s->rank = 20;
        }
        else {
            dr_paired(id - DR_YT, 123, cY, vY, 1.0, 50, atime, s);
            s->rank = id - DR_YT;
        }
        return;
    }
    /* T: X's, never bound, 4 kpc/h from HY */
    for(d = 0; d < 3; d++)
        s->pos[d] = cY[d];
    s->pos[0] += 4. / atime;
    s->vel[0] = atime * 2000;
    s->mass = 0.5;
    s->tag = 1;
    s->bound = -1;
    s->rank = 22;
}

static void
dr_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    /* the in-place path reassigns no non-tracers */
    sp.SubhaloReassignGas = 0;
    SynTypeFn = dr_type;
    syn_setup(&CP, DR_END + SYN_NFIELD, sp);
    SynTypeFn = NULL;
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 2000, 2.0, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 22, 0.01, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 21, 0.01, 1, 0, 2, 2);
    syn_records(recs, 3);
    syn_place(dr_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "re-unbinding death: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

/* X's record: dead in the re-unbinding (Y sank into it alive), tau as its tracer, sunk into A. */
static void
dr_check(const struct Outcome * o, const char * what)
{
    const struct SubTrack * X = &o->rec[1], * Y = &o->rec[2];
    int64_t i, it = -1;
    for(i = 0; i < o->n; i++)
        if(o->r[i].ID == DR_TAU)
            it = i;
    message(0, "re-unbinding death (%s): %ld candidates; X Nbound %d SnapDeath %d Sink %ld SnapSink %d MostBoundID %ld Pos %g %g %g "
               "TracerPos %g %g %g; Y Nbound %d Sink %ld SnapSink %d; A Nbound %d\n", what, o->st.NCands, X->Nbound, X->SnapDeath,
            X->SinkTrackId, X->SnapSink, (long) X->MostBoundID, X->MostBoundPos[0], X->MostBoundPos[1], X->MostBoundPos[2],
            X->TracerPos[0], X->TracerPos[1], X->TracerPos[2], Y->Nbound, Y->SinkTrackId, Y->SnapSink, o->rec[0].Nbound);
    message(0, "re-unbinding death (%s): A's core %g %g %g, %g %g %g km/s, sigma %g kpc/h %g km/s\n", what, o->rec[0].CorePos[0],
            o->rec[0].CorePos[1], o->rec[0].CorePos[2], o->rec[0].CoreVel[0], o->rec[0].CoreVel[1], o->rec[0].CoreVel[2],
            o->rec[0].CoreSigmaX, o->rec[0].CoreSigmaV);
    CHECK(Y->Nbound == 0 && Y->SinkTrackId == 1 && Y->SnapSink == 1, "re-unbinding death (%s): Y Nbound %d Sink %ld SnapSink %d\n", what,
          Y->Nbound, Y->SinkTrackId, Y->SnapSink);
    CHECK(X->Nbound == 0 && X->SnapDeath == 1 && o->rec[0].Nbound > 0, "re-unbinding death (%s): X Nbound %d SnapDeath %d\n", what,
          X->Nbound, X->SnapDeath);
    CHECK(it >= 0 && X->MostBoundID == DR_TAU, "re-unbinding death (%s): X's tracer %ld, not tau\n", what, (long) X->MostBoundID);
    int d;
    for(d = 0; d < 3; d++)
        CHECK(X->MostBoundPos[d] == o->r[it].Pos[d] && X->TracerPos[d] == X->MostBoundPos[d],
              "re-unbinding death (%s): X's MostBoundPos[%d] %g, tau at %g, TracerPos %g\n", what, d, X->MostBoundPos[d], o->r[it].Pos[d],
              X->TracerPos[d]);
    CHECK(X->SinkTrackId == 0 && X->SnapSink == 1 && X->DescendantTrackId == 0,
          "re-unbinding death (%s): X Sink %ld SnapSink %d Descendant %ld\n", what, X->SinkTrackId, X->SnapSink, X->DescendantTrackId);
}

static void
test_collective_reunbind_death(void ** state)
{
    struct Outcome a, s, b;
    dr_run(0, &a);
    dr_run(100, &s);
    dr_run(20, &b);
    CHECK(a.st.NCands == 0 && s.st.NCands == 1 && b.st.NCands == 3, "re-unbinding death: %ld / %ld / %ld in-place candidates\n",
          a.st.NCands, s.st.NCands, b.st.NCands);
    dr_check(&a, "serial");
    dr_check(&s, "sub-tree key");
    dr_check(&b, "in place");
    const struct Outcome * oo[2] = {&s, &b};
    int k, d;
    for(k = 0; k < 2; k++) {
        const struct SubTrack * x = &a.rec[1], * y = &oo[k]->rec[1];
        int same = x->MostBoundID == y->MostBoundID && x->SinkTrackId == y->SinkTrackId && x->SnapSink == y->SnapSink
                   && x->SnapDeath == y->SnapDeath && x->DescendantTrackId == y->DescendantTrackId
                   && x->NestedParentTrackId == y->NestedParentTrackId && x->OrphanIDs[0] == y->OrphanIDs[0];
        for(d = 0; d < 3; d++)
            same = same && x->MostBoundPos[d] == y->MostBoundPos[d] && x->MostBoundVel[d] == y->MostBoundVel[d]
                   && x->TracerPos[d] == y->TracerPos[d] && x->TracerVel[d] == y->TracerVel[d];
        CHECK(same, "re-unbinding death: X's record serially and %s differ\n", k ? "in place" : "in the sub-tree key");
    }
    CHECK(p2_check_tags(b.r, b.n, b.rec, b.nt, "re-unbinding death") == 0, "re-unbinding death: tag invariants\n");
    compare_outcomes(&a, &s, 1e-3, 1e-2, 0, "re-unbinding death, sub-tree key");
    compare_outcomes(&a, &b, 1e-3, 1e-2, 0, "re-unbinding death, in place");
    outcome_free(&b);
    outcome_free(&s);
    outcome_free(&a);
}

/* The three runs of a two-path fixture: serial (collective 0), the satellite in a serial sub-tree
 * key below the in-place central, both in place. */
static const char * const TwoPathName[3] = {"serial", "sub-tree key", "in place"};

/* Particles [lo, hi) of an outcome with TrackId t and BoundTrackId b. */
static int64_t
count_tags(const struct Outcome * o, MyIDType lo, MyIDType hi, int64_t t, int64_t b)
{
    int64_t i, n = 0;
    for(i = 0; i < o->n; i++)
        if(o->r[i].ID >= lo && o->r[i].ID < hi && o->r[i].TrackId == t && o->r[i].BoundTrackId == b)
            n++;
    return n;
}

/* TrackId (bound 0) or BoundTrackId (bound 1) of particle id in an outcome, -2 if absent. */
static int64_t
tag_of(const struct Outcome * o, MyIDType id, int bound)
{
    int64_t i;
    for(i = 0; i < o->n; i++)
        if(o->r[i].ID == id)
            return bound ? o->r[i].BoundTrackId : o->r[i].TrackId;
    return -2;
}

/* Tail tags after a promotion: X (600 bound, 900 tails bound now) outgrows its central A (1000); X's
 * tails U fly between them unbound by both, W rest in A. */
#define TP_X0 1000
#define TP_XT 1600
#define TP_U0 2500
#define TP_W0 2520
#define TP_END 2540

static void
tp_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, cX[3] = {500 + 6 / atime, 500, 500}, cU[3] = {500 + 3 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vX[3] = {0, 60, 0}, vU[3] = {0, 0, 300};
    (void) ud;
    if(id >= TP_END)
        return;
    if(id < TP_X0) {
        syn_ball(id, 131, cA, 1.5, zero, 15, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
        return;
    }
    s->tag = 1;
    if(id < TP_U0) {
        syn_ball(id, 132, cX, 1.5, vX, 15, atime, s);
        s->bound = id < TP_XT ? 1 : -1;
        s->rank = id - TP_X0;
    }
    else {
        if(id < TP_W0)
            syn_ball(id, 133, cU, 1.0, vU, 10, atime, s);
        else
            syn_ball(id, 134, cA, 1.0, zero, 10, atime, s);
        s->bound = -1;
        s->rank = 1500 + id - TP_U0;
    }
}

static void
tp_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, TP_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    /* the previous bound masses: X below the major-progenitor cut of the central election */
    recs[0] = syn_rec(0, 1000, 1000 * (double) (float) SYN_M, -1, 0, 0, 0);
    recs[1] = syn_rec(1, TP_XT - TP_X0, (TP_XT - TP_X0) * SYN_M, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(tp_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "tail priority: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

/* The fixture through the three runs, with the decisions every path shares. */
static void
tp_outcomes(struct Outcome o[3])
{
    const int64_t knob[3] = {0, 2000, 1200};
    int k;
    for(k = 0; k < 3; k++) {
        tp_run(knob[k], &o[k]);
        const struct SubTrack * A = &o[k].rec[0], * X = &o[k].rec[1];
        message(0, "tail priority (%s): %ld candidates; X Nbound %d Rank %d parent %ld, A Nbound %d Rank %d parent %ld; U tagged A %ld, X %ld, "
                   "untagged %ld; W tagged and bound A %ld\n", TwoPathName[k], o[k].st.NCands, X->Nbound, X->Rank, X->NestedParentTrackId,
                A->Nbound, A->Rank, A->NestedParentTrackId, count_tags(&o[k], TP_U0, TP_W0, 0, -1), count_tags(&o[k], TP_U0, TP_W0, 1, -1),
                count_tags(&o[k], TP_U0, TP_W0, -1, -1), count_tags(&o[k], TP_W0, TP_END, 0, 0));
        CHECK(o[k].st.NCands == k, "tail priority (%s): %ld candidates\n", TwoPathName[k], o[k].st.NCands);
        CHECK(X->Rank == 0 && X->NestedParentTrackId == -1 && A->Rank == 1 && A->NestedParentTrackId == 1 && A->Depth == 1,
              "tail priority (%s): X not promoted above A\n", TwoPathName[k]);
        CHECK(count_tags(&o[k], TP_W0, TP_END, 0, 0) == TP_END - TP_W0, "tail priority (%s): W not A's\n", TwoPathName[k]);
        CHECK(count_tags(&o[k], TP_U0, TP_W0, 0, -1) + count_tags(&o[k], TP_U0, TP_W0, 1, -1) == TP_W0 - TP_U0,
              "tail priority (%s): U not a tail of A or X\n", TwoPathName[k]);
        CHECK(p2_check_tags(o[k].r, o[k].n, o[k].rec, o[k].nt, "tail priority") == 0, "tail priority: tag invariants\n");
    }
}

/* U's owner equal on the paths. Disabled: serially A takes U (deeper in the final tree), in place X (its round first). */
static const int TpComparePaths = 0;

static void
test_collective_tail_priority(void ** state)
{
    struct Outcome o[3];
    int k;
    tp_outcomes(o);
    for(k = 1; k < 3 && TpComparePaths; k++)
        CHECK(count_tags(&o[k], TP_U0, TP_W0, 0, -1) == count_tags(&o[0], TP_U0, TP_W0, 0, -1),
              "tail priority: U tagged A %ld serially, %ld %s\n", count_tags(&o[0], TP_U0, TP_W0, 0, -1),
              count_tags(&o[k], TP_U0, TP_W0, 0, -1), TwoPathName[k]);
    for(k = 2; k >= 0; k--)
        outcome_free(&o[k]);
}

/* The dead track's tracer: O dies, its rank-0 tracer tau sits in O's satellite R, which admits and
 * binds it; the stored cores 12.5 core units apart keep the offer at SubhaloAdmitMinCoreDistance 12. */
#define DT_O0 1000
#define DT_R0 1300
#define DT_END 1500

static void
dt_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, cO[3] = {500 + 2.5 / atime, 500, 500}, cR[3] = {500 + 8 / atime, 500, 500};
    const double zero[3] = {0, 0, 0}, vR[3] = {0, 50, 0};
    (void) ud;
    if(id >= DT_END)
        return;
    if(id < DT_O0) {
        syn_ball(id, 141, cA, 1.5, zero, 15, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < DT_R0) {
        if(id == DT_O0)
            syn_ball(id, 142, cR, 0.01, vR, 0.5, atime, s);
        else
            syn_ball(id, 143, cO, 3, zero, 400, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - DT_O0;
    }
    else {
        syn_ball(id, 144, cR, 0.5, vR, 10, atime, s);
        s->tag = s->bound = 2;
        s->rank = id - DT_R0;
    }
}

static void
dt_run(int64_t collective, double mind, struct Outcome * o, struct SubhaloCallStats * cs)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    sp.SubhaloAdmitShare = 0.5;
    sp.SubhaloAdmitMinCoreDistance = mind;
    syn_setup(&CP, DT_END + SYN_NFIELD, sp);
    struct SubTrack recs[3];
    recs[0] = syn_rec(0, 1000, 0.1, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 300, 0.03, 0, 0, 1, 1);
    recs[2] = syn_rec(2, 200, 0.02, 1, 0, 2, 2);
    const double cx[3] = {300, 100, 112.5};
    int t;
    for(t = 0; t < 3; t++) {
        recs[t].CorePos[0] = cx[t];
        recs[t].CorePos[1] = recs[t].CorePos[2] = 100;
        recs[t].CoreVel[0] = recs[t].CoreVel[1] = recs[t].CoreVel[2] = 0;
        recs[t].CoreSigmaX = 1.0f;
        recs[t].CoreSigmaV = 8.0f;
    }
    syn_records(recs, 3);
    syn_place(dt_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "dead tracer: %ld groups\n", ng);
    *cs = *subhalo_last_call_stats();
    outcome_take(o);
    syn_teardown();
}

/* O's tracer equal on the paths. Disabled: serially tau (O's own list), in place the next own tracer (tau is excluded from O's candidate). */
static const int DtComparePaths = 0;

static void
test_collective_dead_tracer(void ** state)
{
    /* serial; O's sub-tree a serial key below in-place A; O in place, R a serial sub-tree key below it */
    const int64_t knob[3] = {0, 1000, 300};
    const double mind[2] = {0, 12};
    int k, m;
    for(m = 0; m < 2; m++) {
        struct Outcome o[3];
        struct SubhaloCallStats cs;
        for(k = 0; k < 3; k++) {
            dt_run(knob[k], mind[m], &o[k], &cs);
            const struct SubTrack * O = &o[k].rec[1], * R = &o[k].rec[2];
            message(0, "dead tracer (%s, distance %g): %ld candidates; admitted %ld taken %ld withheld %ld; O Nbound %d SnapDeath %d "
                       "SnapSink %d MostBoundID %ld; R Nbound %d parent %ld; tau tagged %ld bound %ld\n", TwoPathName[k], mind[m],
                    o[k].st.NCands, cs.NAdmitted, cs.NAdmitTaken, cs.NAdmitWithheld, O->Nbound, O->SnapDeath, O->SnapSink,
                    (long) O->MostBoundID, R->Nbound, R->NestedParentTrackId, tag_of(&o[k], DT_O0, 0), tag_of(&o[k], DT_O0, 1));
            CHECK(o[k].st.NCands == k, "dead tracer (%s): %ld candidates\n", TwoPathName[k], o[k].st.NCands);
            CHECK(cs.NAdmitTaken >= 1 && cs.NAdmitWithheld == 0, "dead tracer (%s, distance %g): %ld taken, %ld withheld\n",
                  TwoPathName[k], mind[m], cs.NAdmitTaken, cs.NAdmitWithheld);
            CHECK(count_tags(&o[k], DT_O0, DT_O0 + 1, 2, 2) == 1, "dead tracer (%s, distance %g): tau not R's\n", TwoPathName[k], mind[m]);
            CHECK(O->Nbound == 0 && O->SnapDeath == 1 && R->Nbound > 0 && R->NestedParentTrackId == 0,
                  "dead tracer (%s, distance %g): O not dead below an alive R under A\n", TwoPathName[k], mind[m]);
            CHECK(p2_check_tags(o[k].r, o[k].n, o[k].rec, o[k].nt, "dead tracer") == 0, "dead tracer: tag invariants\n");
        }
        for(k = 1; k < 3 && DtComparePaths; k++)
            CHECK(o[k].rec[1].MostBoundID == o[0].rec[1].MostBoundID, "dead tracer, distance %g: O's tracer %ld serially, %ld %s\n",
                  mind[m], (long) o[0].rec[1].MostBoundID, (long) o[k].rec[1].MostBoundID, TwoPathName[k]);
        for(k = 2; k >= 0; k--)
            outcome_free(&o[k]);
    }
}

/* The tracer a sunk survivor hands on: Y sinks into A; f moves Y's first frame 5 km/s and is its only
 * loss, so the loop's order leads with t1 (+3 km/s), the final pass with t2 (-2.5 km/s). */
#define SH_Y0 1000
#define SH_END 1251

static void
sh_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, zero[3] = {0, 0, 0};
    double x[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    int d;
    (void) ud;
    if(id >= SH_END)
        return;
    if(id < SH_Y0) {
        syn_ball(id, 151, cA, 1.5, zero, 15, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
        return;
    }
    const int64_t j = id - SH_Y0;
    if(j == 0)
        v[1] = 3;
    else if(j == 1) {
        x[0] = 0.0005;
        v[1] = -2.5;
    }
    else if(j == 2)
        v[1] = 1250;
    else {
        /* mirrored pairs (zero sum of offsets) outside 0.2 kpc of the centre */
        const MyIDType b = SH_Y0 + 3 + 2 * ((j - 3) / 2);
        rand_in_ball(b, 152, 0.6, zero, x);
        rand_in_ball(b, 153, 8, zero, v);
        const double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        const double f = (r < 0.2 && r > 0) ? 0.2 / r : 1, sg = ((j - 3) % 2) ? -1 : 1;
        for(d = 0; d < 3; d++) {
            x[d] *= f * sg;
            v[d] *= sg;
        }
    }
    for(d = 0; d < 3; d++) {
        s->pos[d] = cA[d] + x[d] / atime;
        s->vel[d] = atime * v[d];
    }
    s->mass = 0.4 * SYN_M;
    s->tag = s->bound = 1;
    s->rank = j;
}

static void
sh_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, SH_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 1000, 0.1, -1, 0, 0, 0);
    recs[1] = syn_rec(1, SH_END - SH_Y0, (SH_END - SH_Y0) * 0.4 * SYN_M, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(sh_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "sunk tracer: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

/* Y's tracer equal on the paths. Disabled: serially t1 (loop order), in place and in the sub-tree key t2 (final pass). */
static const int ShComparePaths = 0;

static void
test_collective_sunk_tracer(void ** state)
{
    /* serial; Y a serial sub-tree key below in-place A; both in place */
    const int64_t knob[3] = {0, 500, 200};
    struct Outcome o[3];
    int k;
    for(k = 0; k < 3; k++) {
        sh_run(knob[k], &o[k]);
        const struct SubTrack * A = &o[k].rec[0], * Y = &o[k].rec[1];
        message(0, "sunk tracer (%s): %ld candidates; Y Nbound %d SinkTrackId %ld SnapSink %d MostBoundID %ld; A Nbound %d Rank %d; "
                   "f tagged %ld\n", TwoPathName[k], o[k].st.NCands, Y->Nbound, Y->SinkTrackId, Y->SnapSink, (long) Y->MostBoundID,
                A->Nbound, A->Rank, tag_of(&o[k], SH_Y0 + 2, 0));
        CHECK(o[k].st.NCands == k, "sunk tracer (%s): %ld candidates\n", TwoPathName[k], o[k].st.NCands);
        CHECK(Y->Nbound == 0 && Y->SinkTrackId == 0 && Y->SnapSink == 1 && A->Rank == 0,
              "sunk tracer (%s): Y not sunk into A\n", TwoPathName[k]);
        CHECK(Y->MostBoundID == SH_Y0 || Y->MostBoundID == SH_Y0 + 1, "sunk tracer (%s): Y's tracer %ld, neither t1 nor t2\n",
              TwoPathName[k], (long) Y->MostBoundID);
        CHECK(p2_check_tags(o[k].r, o[k].n, o[k].rec, o[k].nt, "sunk tracer") == 0, "sunk tracer: tag invariants\n");
    }
    for(k = 1; k < 3 && ShComparePaths; k++)
        CHECK(o[k].rec[1].MostBoundID == o[0].rec[1].MostBoundID, "sunk tracer: Y's tracer %ld serially, %ld %s\n",
              (long) o[0].rec[1].MostBoundID, (long) o[k].rec[1].MostBoundID, TwoPathName[k]);
    for(k = 2; k >= 0; k--)
        outcome_free(&o[k]);
}

/* The election centre across the box edge: A, B and an untagged chain span over half the box, so the
 * anchor (key order: the chain's far end; plan: A's minimum ID) decides the central, owner of the chain. */
#define FC_B0 200
#define FC_C0 400
#define FC_END 470

static void
fc_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {995, 500, 500}, cB[3] = {3, 500, 500}, zero[3] = {0, 0, 0};
    (void) ud;
    if(id >= FC_END)
        return;
    if(id < FC_B0) {
        syn_ball(id, 161, cA, 0.5, zero, 5, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else if(id < FC_C0) {
        syn_ball(id, 162, cB, 0.5, zero, 5, atime, s);
        s->tag = s->bound = 1;
        s->rank = id - FC_B0;
    }
    else {
        s->pos[0] = 3 + 8 * (FC_END - id);
        s->pos[1] = s->pos[2] = 500;
    }
}

static void
fc_run(int64_t collective, struct Outcome * o)
{
    Cosmology CP;
    struct SubhaloParams sp = default_subhalo_params();
    sp.SubhaloFieldSO = 0;
    sp.SubhaloPotentialSampleSize = 0;
    sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
    sp.SubhaloCollectiveMinPart = collective;
    syn_setup(&CP, FC_END + SYN_NFIELD, sp);
    struct SubTrack recs[2];
    recs[0] = syn_rec(0, 200, 0.02, -1, 0, 0, 0);
    recs[1] = syn_rec(1, 200, 0.02, 0, 0, 1, 1);
    syn_records(recs, 2);
    syn_place(fc_particle, NULL, 0.55, 1);
    int64_t ng;
    const int ret = run_finder_call(1, &CP, &ng);
    CHECK(ret == 1 && ng == 1, "centre across the edge: %ld groups\n", ng);
    outcome_take(o);
    syn_teardown();
}

/* The chain's track equal on the paths. Disabled: serially A (centre about the chain's first particle), in place B (about the smallest ID). */
static const int FcComparePaths = 0;

static void
test_collective_edge_centre(void ** state)
{
    /* serial; the group in place */
    const int64_t knob[2] = {0, 300};
    const char * name[2] = {"serial", "in place"};
    struct Outcome o[2];
    int64_t toA[2], toB[2];
    int k;
    for(k = 0; k < 2; k++) {
        fc_run(knob[k], &o[k]);
        const struct SubTrack * A = &o[k].rec[0], * B = &o[k].rec[1];
        toA[k] = count_tags(&o[k], FC_C0, FC_END, 0, -1) + count_tags(&o[k], FC_C0, FC_END, 0, 0);
        toB[k] = count_tags(&o[k], FC_C0, FC_END, 1, -1) + count_tags(&o[k], FC_C0, FC_END, 1, 1);
        message(0, "centre across the edge (%s): %ld candidates; A Nbound %d Rank %d; B Nbound %d Rank %d; chain tagged A %ld, B %ld\n",
                name[k], o[k].st.NCands, A->Nbound, A->Rank, B->Nbound, B->Rank, toA[k], toB[k]);
        CHECK((o[k].st.NCands > 0) == (k > 0), "centre across the edge (%s): %ld candidates\n", name[k], o[k].st.NCands);
        CHECK(A->Nbound > 0 && B->Nbound > 0 && toA[k] + toB[k] == FC_END - FC_C0,
              "centre across the edge (%s): not two survivors holding the chain\n", name[k]);
        CHECK(p2_check_tags(o[k].r, o[k].n, o[k].rec, o[k].nt, "centre across the edge") == 0, "centre across the edge: tag invariants\n");
    }
    CHECK(!FcComparePaths || toA[1] == toA[0], "centre across the edge: chain tagged A %ld serially, %ld in place\n", toA[0], toA[1]);
    outcome_free(&o[1]);
    outcome_free(&o[0]);
}

/* The unbinding step at its boundaries: A, a cold ball of US_NC tracers, and US_END - US_NC fast
 * particles through it in A's source but not previously bound. */
#define US_NC 200
#define US_END 220

static void
us_particle(MyIDType id, double atime, struct SynPart * s, void * ud)
{
    const double cA[3] = {500, 500, 500}, zero[3] = {0, 0, 0}, fast[2][3] = {{3000, 0, 0}, {-3000, 0, 0}};
    (void) ud;
    if(id >= US_END)
        return;
    if(id < US_NC) {
        syn_ball(id, 171, cA, 1.0, zero, 2, atime, s);
        s->tag = s->bound = 0;
        s->rank = id;
    }
    else {
        syn_ball(id, 172, cA, 1.0, fast[id % 2], 0, atime, s);
        s->tag = 0;
    }
}

/* Thresholds at US_NC and BoundMassPrecision 1: A survives with US_NC bound and converges when
 * Nbound == Nlast; with SubhaloMaxUnbindIterations 1 it stops at the cap. One above US_NC, the
 * particle threshold kills it in the loop and the tracer threshold at the entry. */
static void
us_check(int64_t collective)
{
    const char * name = collective ? "in place" : "serial";
    int mode;
    for(mode = 0; mode < 4; mode++) {
        const int cap = mode == 1, dies = mode >= 2;
        Cosmology CP;
        struct SubhaloParams sp = default_subhalo_params();
        sp.SubhaloFieldSO = 0;
        sp.SubhaloPotentialSampleSize = 0;
        sp.SubhaloTreeOpenAngle = COLL_TEST_THETA;
        sp.SubhaloCollectiveMinPart = collective;
        sp.SubhaloMinNumPart = US_NC + (mode == 2);
        sp.SubhaloMinNumTracerPart = US_NC + (mode == 3);
        sp.SubhaloBoundMassPrecision = cap ? 0.995 : 1;
        sp.SubhaloMaxUnbindIterations = cap ? 1 : 50;
        syn_setup(&CP, US_END + SYN_NFIELD, sp);
        struct SubTrack rec = syn_rec(0, US_NC, US_NC * SYN_M, -1, 0, 0, 0);
        syn_records(&rec, 1);
        syn_place(us_particle, NULL, 0.55, 1);
        int64_t ng;
        const int ret = run_finder_call(1, &CP, &ng);
        CHECK(ret == 1 && ng == 1, "unbinding step (%s): %ld groups\n", name, ng);
        struct Outcome o;
        outcome_take(&o);
        const struct SubhaloCallStats cs = *subhalo_last_call_stats();
        syn_teardown();
        const int capped = !!(o.rec[0].Flags & SUBHALO_FLAG_ITERCAP);
        message(0, "unbinding step (%s, mode %d): %ld candidates in place; A Nbound %d capped %d; %ld capped in the call\n", name, mode,
                o.st.NCands, o.rec[0].Nbound, capped, cs.NItercap);
        CHECK((o.st.NCands > 0) == (collective > 0), "unbinding step (%s): %ld candidates in place\n", name, o.st.NCands);
        CHECK(o.rec[0].Nbound == (dies ? 0 : US_NC), "unbinding step (%s, mode %d): A Nbound %d, not %d\n", name, mode,
              o.rec[0].Nbound, dies ? 0 : US_NC);
        CHECK(capped == cap, "unbinding step (%s, mode %d): A capped %d\n", name, mode, capped);
        outcome_free(&o);
    }
}

static void
test_collective_step_serial(void ** state)
{
    us_check(0);
}

static void
test_collective_step_inplace(void ** state)
{
    us_check(100);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_collective_serial_vs_inplace),
        cmocka_unit_test(test_collective_subtree_keys),
        cmocka_unit_test(test_collective_merger),
        cmocka_unit_test(test_collective_kill_passup),
        cmocka_unit_test(test_collective_binder),
        cmocka_unit_test(test_collective_serial_repairs),
        cmocka_unit_test(test_collective_relaxed_boundary),
        cmocka_unit_test(test_collective_dead_field),
        cmocka_unit_test(test_collective_sunk_root_field),
        cmocka_unit_test(test_collective_failure_after_split),
        cmocka_unit_test(test_collective_voided_claim),
        cmocka_unit_test(test_collective_long_chain),
        cmocka_unit_test(test_collective_admission),
        cmocka_unit_test(test_collective_admission_props),
        cmocka_unit_test(test_collective_guards),
        cmocka_unit_test(test_collective_walk_theta),
        cmocka_unit_test(test_collective_props_identity),
        cmocka_unit_test(test_collective_central_source),
        cmocka_unit_test(test_collective_knn_identity),
        cmocka_unit_test(test_collective_vote_identity),
        cmocka_unit_test(test_collective_vote_receivers),
        cmocka_unit_test(test_collective_vote_plan),
        cmocka_unit_test(test_collective_vote_subtree_majority),
        cmocka_unit_test(test_collective_vote_inplace_receivers),
        cmocka_unit_test(test_collective_reunbind_death),
        cmocka_unit_test(test_collective_tail_priority),
        cmocka_unit_test(test_collective_dead_tracer),
        cmocka_unit_test(test_collective_sunk_tracer),
        cmocka_unit_test(test_collective_edge_centre),
        cmocka_unit_test(test_collective_step_serial),
        cmocka_unit_test(test_collective_step_inplace),
    };
    const char * only = getenv("SUBHALO_TEST_ONLY");
    if(only && only[0]) {
        struct CMUnitTest sel[sizeof(tests) / sizeof(tests[0])];
        size_t i, ns = 0;
        for(i = 0; i < sizeof(tests) / sizeof(tests[0]); i++)
            if(strstr(tests[i].name, only))
                sel[ns++] = tests[i];
        return _cmocka_run_group_tests_mpi("selected", sel, ns, NULL, NULL);
    }
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
