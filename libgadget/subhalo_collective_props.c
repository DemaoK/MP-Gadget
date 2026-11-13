/* Properties of the in-place candidates as a distributed ordered set: holder sums, the radial
 * profile by rank-ordered compensated sums, shapes and the deferred Rockstar recipes. */

#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <gsl/gsl_eigen.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include "utils/mpsort.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "subhalo_collective_internal.h"

/* ------------------------------------------------------------------ properties: the distributed ordered set */

static void
radrec_radix(const void * a, void * radix, void * arg)
{
    memcpy(radix, ((const struct RadRec *) a)->key, 24);
}

static void
coll_eigen3(double mat[9], double eval[3], double evec[9], gsl_eigen_symmv_workspace * ws)
{
    gsl_matrix_view m = gsl_matrix_view_array(mat, 3, 3);
    gsl_vector_view ev = gsl_vector_view_array(eval, 3);
    gsl_matrix_view em = gsl_matrix_view_array(evec, 3, 3);
    gsl_eigen_symmv(&m.matrix, &ev.vector, &em.matrix, ws);
    gsl_eigen_symmv_sort(&ev.vector, &em.matrix, GSL_EIGEN_SORT_VAL_DESC);
}

/* The Allgood fit of subhalo_props.c (allgood_shape) for every candidate with an aperture,
 * the tensor sums reduced over the ranks each iteration. np[c]: particles of the aperture (the
 * global positions 0 .. np - 1), R[c] its radius; out: BtoA, CtoA, AxisA per candidate. */
static void
shapes_distributed(const struct Coll * C, const struct RadRec * rr, int64_t nr, const int64_t * pos, int64_t nc,
                   const int64_t * np, const double * R, double rfloor, float * out, int stride)
{
    int64_t c, k;
    int d, e, it;
    double * orth = (double *) mymalloc("SubCollShOrth", sizeof(double) * 9 * (nc + 1));
    double * ax2 = (double *) mymalloc("SubCollShAx", sizeof(double) * 3 * (nc + 1));
    double * prev = (double *) mymalloc("SubCollShPrev", sizeof(double) * 2 * (nc + 1));
    int * active = (int *) mymalloc("SubCollShAct", sizeof(int) * (nc + 1));
    double * T = (double *) mymalloc("SubCollShT", sizeof(double) * 8 * (nc + 1));
    double * TG = (double *) mymalloc("SubCollShTG", sizeof(double) * 8 * (nc + 1));
    gsl_eigen_symmv_workspace * ws = gsl_eigen_symmv_alloc(3);
    if(!ws)
        endrun(1, "Subhalo properties: gsl_eigen_symmv_alloc failed\n");
    int nact = 0;
    for(c = 0; c < nc; c++) {
        active[c] = np[c] >= SUBHALO_SHAPE_MINPART && R[c] > 0;
        nact += active[c];
        for(d = 0; d < 9; d++)
            orth[9 * c + d] = (d % 4 == 0) ? 1 : 0;
        for(d = 0; d < 3; d++)
            ax2[3 * c + d] = R[c] * R[c];
        prev[2 * c] = prev[2 * c + 1] = 0;
    }
    for(it = 0; it < SUBHALO_SHAPE_ITERATIONS && nact > 0; it++) {
        memset(T, 0, sizeof(double) * 8 * (nc + 1));
        for(k = 0; k < nr; k++) {
            c = (int64_t) rr[k].key[2];
            if(!active[c] || pos[k] >= np[c])
                continue;
            const double minq2 = (rfloor / R[c]) * (rfloor / R[c]);
            double q2 = 0;
            for(e = 0; e < 3; e++) {
                const double * o = orth + 9 * c + 3 * e;
                const double pe = o[0] * rr[k].dx[0] + o[1] * rr[k].dx[1] + o[2] * rr[k].dx[2];
                q2 += pe * pe / ax2[3 * c + e];
            }
            if(q2 < minq2)
                q2 = minq2;
            if(!(q2 <= 1))
                continue;
            const double w = rr[k].m / q2;
            double * t = T + 8 * c;
            t[6] += 1;
            t[7] += w;
            int q = 0;
            for(d = 0; d < 3; d++)
                for(e = 0; e <= d; e++)
                    t[q++] += w * rr[k].dx[d] * rr[k].dx[e];
        }
        MPI_Allreduce(T, TG, 8 * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        nact = 0;
        for(c = 0; c < nc; c++) {
            if(!active[c])
                continue;
            const double * t = TG + 8 * c;
            if(t[6] < SUBHALO_SHAPE_MINPART || !(t[7] > 0)) {
                active[c] = 0;
                continue;
            }
            double M[9], eval[3], evec[9];
            int q = 0;
            for(d = 0; d < 3; d++)
                for(e = 0; e <= d; e++) {
                    M[3 * d + e] = t[q++] / t[7];
                    M[3 * e + d] = M[3 * d + e];
                }
            coll_eigen3(M, eval, evec, ws);
            if(!(eval[0] > 0 && eval[1] > 0 && eval[2] > 0)) {
                active[c] = 0;
                continue;
            }
            const double ba = sqrt(eval[1] / eval[0]), ca = sqrt(eval[2] / eval[0]);
            double A[3] = {evec[0], evec[3], evec[6]};
            int big = 0;
            for(d = 1; d < 3; d++)
                if(fabs(A[d]) > fabs(A[big]))
                    big = d;
            const double sgn = A[big] < 0 ? -1 : 1;
            float * o = out + stride * c;
            o[0] = (float) ba;
            o[1] = (float) ca;
            for(d = 0; d < 3; d++)
                o[2 + d] = (float) (sgn * A[d]);
            if(fabs(ba - prev[2 * c]) < 0.01 * prev[2 * c] && fabs(ca - prev[2 * c + 1]) < 0.01 * prev[2 * c + 1]) {
                active[c] = 0;
                continue;
            }
            prev[2 * c] = ba;
            prev[2 * c + 1] = ca;
            for(e = 0; e < 3; e++) {
                for(d = 0; d < 3; d++)
                    orth[9 * c + 3 * e + d] = evec[3 * d + e];
                ax2[3 * c + e] = R[c] * R[c] * eval[e] / eval[0];
            }
            nact++;
        }
    }
    gsl_eigen_symmv_free(ws);
    myfree(TG);
    myfree(T);
    myfree(active);
    myfree(prev);
    myfree(ax2);
    myfree(orth);
}

/* The compensated sums (subhalo_ksum_*) as a non-commutative MPI operation on pairs of doubles:
 * MPI applies it in rank order, the order of the radial ordered set. */
static MPI_Datatype KsumType = MPI_DATATYPE_NULL;
static MPI_Op KsumOp = MPI_OP_NULL;

static void
ksum_mpi_op(void * in, void * inout, int * len, MPI_Datatype * dt)
{
    const double * a = (const double *) in;
    double * b = (double *) inout;
    int i;
    (void) dt;
    for(i = 0; i < *len; i++) {
        double r[2] = {a[2 * i], a[2 * i + 1]};
        subhalo_ksum_merge(r, b + 2 * i);
        b[2 * i] = r[0];
        b[2 * i + 1] = r[1];
    }
}

static void
ksum_mpi_init(void)
{
    if(KsumOp != MPI_OP_NULL)
        return;
    MPI_Type_contiguous(2, MPI_DOUBLE, &KsumType);
    MPI_Type_commit(&KsumType);
    MPI_Op_create(ksum_mpi_op, 0, &KsumOp);
}

/* Per-candidate transient scratch of round_props on every rank, charged by subhalo_coll_estimate:
 * the vectors kept to the end plus the largest later stage (asserted below). */
#define COLL_NL (8 + 2 * SUBHALO_NSO)
#define COLL_DFR_NA 8                          /* (1) ncore, M, Sb 3, S2 3: pairs */
#define COLL_DFR_NW 4                          /* (4) the core: Mcore, Sc 3: pairs */
#define COLL_DFR_NB (5 * SUBHALO_NFW_MAXBINS)  /* (4) per bin: r at pb - 1, pb, pb + 1, m, Mc */
#define COLL_DFR_DOUBLES (2 * (2 * COLL_DFR_NA) + 2 * (SUBHALO_NFW_MAXBINS + 1) + 2 * COLL_DFR_NB + 2 * (2 * COLL_DFR_NW) + 2)
#define COLL_PROPS_KEPT_BYTES (2 * 8 + 4 * 8 + 2 * COLL_NL * 8)
#define COLL_PROPS_DFR_BYTES (COLL_DFR_DOUBLES * 8 + 3 * 8)
#define COLL_PROPS_SHAPE_BYTES ((9 + 3 + 2 + 8 + 8) * 8 + 4 + 8 + 8)
#define COLL_PROPS_CAND_BYTES (COLL_PROPS_KEPT_BYTES + COLL_PROPS_DFR_BYTES)
_Static_assert((NHS + 2) * 8 <= COLL_PROPS_CAND_BYTES && COLL_PROPS_KEPT_BYTES + 2 * COLL_NL * 8 <= COLL_PROPS_CAND_BYTES
               && COLL_PROPS_KEPT_BYTES + 9 * 8 <= COLL_PROPS_CAND_BYTES && COLL_PROPS_SHAPE_BYTES <= COLL_PROPS_DFR_BYTES,
               "COLL_PROPS_CAND_BYTES is not the largest per-candidate stage of round_props");
_Static_assert(COLL_PROPS_CAND_BYTES <= SUBHALO_INPLACE_CAND_PROPS_BYTES,
               "SUBHALO_INPLACE_CAND_PROPS_BYTES does not cover the per-candidate scratch of round_props");


/* subhalo_props_deferred's recipes on the round's radial set (rr by (label, r, ID), pos, Mc = M(<= r)):
 * Rockstar's aperture pos < kc[vir] for the velocities and the NFW fit, in four
 * reductions (1)-(4). Results in rp->dfr (km/s, run length unit), -1 undefined. Collective. */
static void
deferred_distributed(const struct Coll * C, const struct CoreOut * core, const struct RadRec * rr, int64_t nr,
                     const int64_t * pos, const double * Mc, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const double * rho = C->ctx->rho;
    const double rfloor = C->up->h / 2.8;
    const double vkms = subhalo_vel_to_kms(C->up->atime);
    const int NB = SUBHALO_NFW_MAXBINS;
    const int NA = 2 * COLL_DFR_NA, NW = 2 * COLL_DFR_NW, NV = COLL_DFR_NB;
    int64_t c, k;
    int d, b;
    if(nc <= 0)
        return;
    ksum_mpi_init();
    /* the allocations COLL_PROPS_DFR_BYTES counts */
    double * A = (double *) mymalloc("SubCollDfA", sizeof(double) * NA * nc);
    double * AG = (double *) mymalloc("SubCollDfAG", sizeof(double) * NA * nc);
    double * X = (double *) mymalloc("SubCollDfX", sizeof(double) * (NB + 1) * nc);
    double * XG = (double *) mymalloc("SubCollDfXG", sizeof(double) * (NB + 1) * nc);
    double * V = (double *) mymalloc("SubCollDfV", sizeof(double) * NV * nc);
    double * VG = (double *) mymalloc("SubCollDfVG", sizeof(double) * NV * nc);
    double * W = (double *) mymalloc("SubCollDfW", sizeof(double) * NW * nc);
    double * WG = (double *) mymalloc("SubCollDfWG", sizeof(double) * NW * nc);
    int64_t * kc = (int64_t *) mymalloc("SubCollDfKc", sizeof(int64_t) * 3 * nc);   /* kc, ncore, nb */
    double * Rv = (double *) mymalloc("SubCollDfRv", sizeof(double) * 2 * nc);       /* Rvir, mpbin */
    for(c = 0; c < nc; c++) {
        kc[3 * c] = rp->kc[SUBHALO_NSO * c];
        Rv[2 * c] = kc[3 * c] >= 0 ? cbrt(3 * rp->Mso[SUBHALO_NSO * c] / (4 * M_PI * rho[0])) : -1;
    }

    /* (1) inside the aperture: velocity sums about the core velocity and the count inside 0.1 Rvir
     * (compensated pairs); (2) the largest particle mass and M(<= r) at kc - 1 (one record: a MAX) */
    memset(A, 0, sizeof(double) * NA * nc);
    for(c = 0; c < nc; c++)
        X[(NB + 1) * c] = X[(NB + 1) * c + 1] = 0;
    for(k = 0; k < nr; k++) {
        c = (int64_t) rr[k].key[2];
        if(pos[k] >= kc[3 * c])
            continue;
        const struct SubhaloCore * cp = &core->d[c].core;
        const double m = rr[k].m;
        double * a = A + NA * c;
        if(rr[k].r < 0.1 * Rv[2 * c])
            a[0] += 1;
        subhalo_ksum_add(a + 2, m);
        for(d = 0; d < 3; d++) {
            const double dv = rr[k].vel[d] - cp->Vel[d];
            subhalo_ksum_add(a + 4 + 2 * d, m * dv);
            subhalo_ksum_add(a + 10 + 2 * d, m * dv * dv);
        }
        if(m > X[(NB + 1) * c])
            X[(NB + 1) * c] = m;
        if(pos[k] == kc[3 * c] - 1)
            X[(NB + 1) * c + 1] = Mc[k];
    }
    MPI_Allreduce(A, AG, COLL_DFR_NA * nc, KsumType, KsumOp, C->Comm);
    for(c = 0; c < nc; c++) {
        XG[2 * c] = X[(NB + 1) * c];
        XG[2 * c + 1] = X[(NB + 1) * c + 1];
    }
    MPI_Allreduce(XG, V, 2 * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    for(c = 0; c < nc; c++) {
        kc[3 * c + 1] = 0;
        kc[3 * c + 2] = 0;
        Rv[2 * c + 1] = 0;
        if(kc[3 * c] < 1)
            continue;
        const int64_t nap = kc[3 * c], nfit = kc[3 * c];
        const double * a = AG + NA * c;
        int64_t ncore = (int64_t) subhalo_ksum_value(a);
        if(ncore < SUBHALO_ROCKSTAR_CORE_MINPART)
            ncore = nap < SUBHALO_ROCKSTAR_CORE_MINPART ? nap : SUBHALO_ROCKSTAR_CORE_MINPART;
        kc[3 * c + 1] = ncore;
        /* the bin plan from the fit aperture's mass Mfit and largest mass, as the serial recipe */
        kc[3 * c + 2] = subhalo_nfw_bin_plan(nfit, V[2 * c + 1], V[2 * c], &Rv[2 * c + 1]);
    }

    /* (3) the closing position of each bin: the first record reaching its threshold, the global
     * minimum by a MAX of the negated positions */
    for(k = 0; k < (NB + 1) * nc; k++)
        X[k] = -1e300;
    {
        int64_t lab = -1;
        int nextb = 1;
        for(k = 0; k < nr; k++) {
            c = (int64_t) rr[k].key[2];
            if(c != lab) {
                lab = c;
                nextb = 1;
            }
            if(pos[k] >= kc[3 * c])
                continue;
            while(nextb <= kc[3 * c + 2] && Mc[k] >= subhalo_nfw_threshold(nextb, Rv[2 * c + 1])) {
                X[(NB + 1) * c + nextb] = -(double) pos[k];
                nextb++;
            }
        }
    }
    MPI_Allreduce(X, XG, (NB + 1) * nc, MPI_DOUBLE, MPI_MAX, C->Comm);

    /* (4) the radii at and around each closing position, the mass and cumulative mass there (one
     * record each: exact sums), and the core velocity sums (compensated pairs) */
    memset(V, 0, sizeof(double) * NV * nc);
    memset(W, 0, sizeof(double) * NW * nc);
    for(k = 0; k < nr; k++) {
        c = (int64_t) rr[k].key[2];
        double * v = V + NV * c;
        const int64_t pk = pos[k];
        if(pk < kc[3 * c + 1]) {
            const struct SubhaloCore * cp = &core->d[c].core;
            double * w = W + NW * c;
            subhalo_ksum_add(w, rr[k].m);
            for(d = 0; d < 3; d++)
                subhalo_ksum_add(w + 2 + 2 * d, rr[k].m * (rr[k].vel[d] - cp->Vel[d]));
        }
        for(b = 1; b <= kc[3 * c + 2]; b++) {
            const double xb = XG[(NB + 1) * c + b];
            if(xb < -1e299)
                break;
            const int64_t pb = (int64_t) -xb;
            double * e = v + 5 * (b - 1);
            if(pk == pb - 1)
                e[0] += rr[k].r;
            if(pk == pb) {
                e[1] += rr[k].r;
                e[3] += rr[k].m;
                e[4] += Mc[k];
            }
            if(pk == pb + 1)
                e[2] += rr[k].r;
        }
    }
    MPI_Allreduce(V, VG, NV * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    MPI_Allreduce(W, WG, COLL_DFR_NW * nc, KsumType, KsumOp, C->Comm);
    for(c = 0; c < nc; c++) {
        if(kc[3 * c] < 1)
            continue;
        const double * v = VG + NV * c;
        const double * a = AG + NA * c, * w = WG + NW * c;
        const struct SubhaloCore * cp = &core->d[c].core;
        double * o = rp->dfr + 8 * c;
        /* the velocity trio: the serial recipe's function on the same sums */
        double Sb[3][2], S2[3][2], Sc[3][2];
        float bv[3], cv[3], vrms;
        for(d = 0; d < 3; d++) {
            Sb[d][0] = a[4 + 2 * d];
            Sb[d][1] = a[5 + 2 * d];
            S2[d][0] = a[10 + 2 * d];
            S2[d][1] = a[11 + 2 * d];
            Sc[d][0] = w[2 + 2 * d];
            Sc[d][1] = w[3 + 2 * d];
        }
        subhalo_deferred_velocities(bv, cv, &vrms, cp->Vel, subhalo_ksum_value(a + 2), subhalo_ksum_value(w), Sb, S2, Sc, vkms);
        for(d = 0; d < 3; d++) {
            o[1 + d] = cv[d];
            o[4 + d] = bv[d];
        }
        o[7] = vrms;
        const int nb = (int) kc[3 * c + 2];
        if(nb <= 0)
            continue;
        const double mpbin = Rv[2 * c + 1];
        const int64_t nbound = (int64_t) rp->hs[NHS * c + HS_N];
        double edge[SUBHALO_NFW_MAXBINS + 1];
        edge[0] = 0;
        int nf = 0;
        for(b = 1; b <= nb; b++) {
            const double xb = XG[(NB + 1) * c + b];
            if(xb < -1e299)
                break;
            const int64_t pb = (int64_t) -xb;
            const double * e = v + 5 * (b - 1);
            const double rnext = pb + 1 < nbound ? e[2] : e[1];
            edge[b] = subhalo_nfw_edge(b, mpbin, e[0], pb > 0, e[1], rnext, e[4], e[3]);
            nf = b;
        }
        const double * prf = rp->prof + 8 * c;
        const double rs0 = subhalo_klypin_rs(Rv[2 * c], rp->Mso[SUBHALO_NSO * c], prf[2], prf[3]);
        const double rs = subhalo_nfw_fit(rs0, edge, nf, 3 * rfloor);
        o[0] = rs > 0 ? rs : -1;
    }
    myfree(Rv);
    myfree(kc);
    myfree(WG);
    myfree(W);
    myfree(VG);
    myfree(V);
    myfree(XG);
    myfree(X);
    myfree(AG);
    myfree(A);
}

/* The holder-side sums of the round's alive candidates (counts, masses, energies, J, inertia,
 * baryons) and the most massive BH, reduced into rp->hs/bhm/bhid (allocated by the caller);
 * returns the local bound count. Collective. */
static int64_t
round_holder_sums(const struct Coll * C, const struct CoreOut * core, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const struct SubhaloUnbindParams * up = C->up;
    const double a = up->atime, BoxSize = up->BoxSize;
    int64_t i, c;
    int d;
    double * hs = (double *) mymalloc("SubCollHsLoc", sizeof(double) * NHS * (nc + 1));
    double * bm = (double *) mymalloc("SubCollBhmLoc", sizeof(double) * (nc + 1));
    MyIDType * bid = (MyIDType *) mymalloc("SubCollBhidLoc", sizeof(MyIDType) * (nc + 1));
    memset(hs, 0, sizeof(double) * NHS * (nc + 1));
    for(c = 0; c < nc; c++) {
        bm[c] = -1;
        bid[c] = (MyIDType) -1;
    }
    int64_t nb = 0;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || !(C->Flag[i] & CF_BOUND) || C->cs[cc].state != SUBHALO_ST_ALIVE || !subhalo_part_live(&P[i]))
            continue;
        nb++;
        const struct CandState * cs = &C->cs[cc];
        const struct SubhaloCore * cp = &core->d[cc - C->c0].core;
        double * h = hs + NHS * (cc - C->c0);
        const double m = P[i].Mass;
        const int type = P[i].Type;
        h[HS_N] += 1;
        if(is_tracer(i, up->TracerTypes))
            h[HS_NTR] += 1;
        h[HS_NT + type] += 1;
        h[HS_MT + type] += m;
        h[HS_M] += m;
        double k2 = 0, dxc[3], dvc[3];
        for(d = 0; d < 3; d++) {
            const double dx = NEAREST(P[i].Pos[d] - cs->RefPos[d], BoxSize);
            const double dv = (P[i].Vel[d] - cs->RefVel[d]) / a + up->Hubble * a * dx;
            k2 += dv * dv;
            dxc[d] = NEAREST(P[i].Pos[d] - cp->Pos[d], BoxSize);
            dvc[d] = P[i].Vel[d] - cs->RefVel[d];
        }
        double phi = C->En[i] - 0.5 * k2;
        if(up->WithThermal)
            phi -= subhalo_thermal_energy(i, a);
        h[HS_K] += m * 0.5 * k2;
        h[HS_U] += m * phi;
        h[HS_J + 0] += m * (dxc[1] * dvc[2] - dxc[2] * dvc[1]);
        h[HS_J + 1] += m * (dxc[2] * dvc[0] - dxc[0] * dvc[2]);
        h[HS_J + 2] += m * (dxc[0] * dvc[1] - dxc[1] * dvc[0]);
        h[HS_I + 0] += m * dxc[0] * dxc[0];
        h[HS_I + 1] += m * dxc[1] * dxc[1];
        h[HS_I + 2] += m * dxc[2] * dxc[2];
        h[HS_I + 3] += m * dxc[0] * dxc[1];
        h[HS_I + 4] += m * dxc[0] * dxc[2];
        h[HS_I + 5] += m * dxc[1] * dxc[2];
        if(type == 0) {
            h[HS_HE] += m * P[i].HeIIIionized;
            h[HS_SFR] += SPHP(i).Sfr;
            h[HS_GZ] += SPHP(i).Metallicity * m;
            for(d = 0; d < NMETALS; d++)
                h[HS_GZE + d] += SPHP(i).Metals[d] * m;
        }
        else if(type == 4) {
            h[HS_SZ] += STARP(i).Metallicity * m;
            for(d = 0; d < NMETALS; d++)
                h[HS_SZE + d] += STARP(i).Metals[d] * m;
        }
        else if(type == 5) {
            const double bmass = BHP(i).Mass;
            h[HS_BHM] += bmass;
            h[HS_MDOT] += BHP(i).Mdot;
            const int64_t q = cc - C->c0;
            if(bmass > bm[q] || (bmass == bm[q] && P[i].ID < bid[q])) {
                bm[q] = bmass;
                bid[q] = P[i].ID;
            }
        }
    }
    if(nc > 0) {
        MPI_Allreduce(hs, rp->hs, NHS * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        MPI_Allreduce(bm, rp->bhm, nc, MPI_DOUBLE, MPI_MAX, C->Comm);
        for(c = 0; c < nc; c++)
            if(bm[c] != rp->bhm[c])
                bid[c] = (MyIDType) -1;
        MPI_Allreduce(bid, rp->bhid, nc, MPI_UINT64_T, MPI_MIN, C->Comm);
    }
    myfree(bid);
    myfree(bm);
    myfree(hs);
    return nb;
}

/* The properties of the round's alive candidates with the formulas of subhalo_props_hbt, _rockstar
 * and _baryons, from the final-pass energies, frames and cores. Collective; rp is allocated here
 * (bottom) and freed by the caller (free_round_props). */
void
round_props(const struct Coll * C, const struct CoreOut * core, struct RoundProps * rp)
{
    const int64_t nc = C->c1 - C->c0;
    const struct SubhaloUnbindParams * up = C->up;
    const double BoxSize = up->BoxSize;
    const double rfloor = up->h / 2.8;
    const double * rho = C->ctx->rho;
    int64_t i, c, k;
    int d, s;

    rp->hs = (double *) mymalloc("SubCollHs", sizeof(double) * NHS * (nc + 1));
    rp->bhm = (double *) mymalloc("SubCollBhm", sizeof(double) * (nc + 1));
    rp->bhid = (MyIDType *) mymalloc("SubCollBhid", sizeof(MyIDType) * (nc + 1));
    rp->Mso = (double *) mymalloc("SubCollMso", sizeof(double) * SUBHALO_NSO * (nc + 1));
    rp->kc = (int64_t *) mymalloc("SubCollKc", sizeof(int64_t) * SUBHALO_NSO * (nc + 1));
    rp->prof = (double *) mymalloc("SubCollProf", sizeof(double) * 8 * (nc + 1));
    rp->jv = (double *) mymalloc("SubCollJv", sizeof(double) * 9 * (nc + 1));
    rp->shape = (float *) mymalloc("SubCollShape", sizeof(float) * 10 * (nc + 1));
    for(k = 0; k < 10 * nc; k++)
        rp->shape[k] = -1;
    rp->dfr = (double *) mymalloc("SubCollDfr", sizeof(double) * 8 * (nc + 1));
    for(k = 0; k < 8 * nc; k++)
        rp->dfr[k] = -1;

    /* ---- the holder-side sums ---- */
    int64_t nb = round_holder_sums(C, core, rp);

    /* ---- the radial ordered set about the cores: (label, r, ID), M(<= r) by a vector Exscan ---- */
    int64_t ntot = 0;
    MPI_Allreduce(&nb, &ntot, 1, MPI_INT64, MPI_SUM, C->Comm);
    for(c = 0; c < nc; c++) {
        for(s = 0; s < SUBHALO_NSO; s++) {
            rp->kc[SUBHALO_NSO * c + s] = -1;
            rp->Mso[SUBHALO_NSO * c + s] = 0;
        }
        for(d = 0; d < 8; d++)
            rp->prof[8 * c + d] = -1;
        for(d = 0; d < 9; d++)
            rp->jv[9 * c + d] = 0;
    }
    if(ntot == 0)
        return;
    struct RadRec * rr = (struct RadRec *) mymalloc("SubCollRad", sizeof(struct RadRec) * (nb + 1));
    nb = 0;
    for(i = 0; i < C->NumPart; i++) {
        const int32_t cc = C->Cand[i];
        if(!in_round(C, cc) || !(C->Flag[i] & CF_BOUND) || C->cs[cc].state != SUBHALO_ST_ALIVE || !subhalo_part_live(&P[i]))
            continue;
        const struct SubhaloCore * cp = &core->d[cc - C->c0].core;
        struct RadRec * e = &rr[nb++];
        double r2 = 0;
        for(d = 0; d < 3; d++) {
            e->dx[d] = NEAREST(P[i].Pos[d] - cp->Pos[d], BoxSize);
            e->vel[d] = P[i].Vel[d];
            r2 += e->dx[d] * e->dx[d];
        }
        const double r = sqrt(r2);
        e->r = r > rfloor ? r : rfloor;
        e->m = P[i].Mass;
        e->key[0] = P[i].ID;
        e->key[1] = double_key(e->r);
        e->key[2] = (uint64_t) (cc - C->c0);
    }
    mpsort_mpi(rr, nb, sizeof(struct RadRec), radrec_radix, 24, NULL, C->Comm);
    /* positions and cumulative masses: a local prefix per label plus a vector Exscan, the masses as
     * compensated pairs in rank order, so M(<= r) is the serial recipe's sum */
    ksum_mpi_init();
    int64_t * cnt = (int64_t *) mymalloc("SubCollRadCnt", sizeof(int64_t) * 2 * (nc + 1));
    double * msum = (double *) mymalloc("SubCollRadM", sizeof(double) * 4 * (nc + 1));
    memset(cnt, 0, sizeof(int64_t) * 2 * (nc + 1));
    memset(msum, 0, sizeof(double) * 4 * (nc + 1));
    for(k = 0; k < nb; k++) {
        cnt[rr[k].key[2]]++;
        subhalo_ksum_add(msum + 2 * rr[k].key[2], rr[k].m);
    }
    MPI_Exscan(cnt, cnt + nc, nc, MPI_INT64, MPI_SUM, C->Comm);
    MPI_Exscan(msum, msum + 2 * nc, nc, KsumType, KsumOp, C->Comm);
    if(C->ThisTask == 0)
        for(c = 0; c < nc; c++) {
            cnt[nc + c] = 0;
            msum[2 * nc + 2 * c] = msum[2 * nc + 2 * c + 1] = 0;
        }
    int64_t * pos = (int64_t *) mymalloc("SubCollRadPos", sizeof(int64_t) * (nb + 1));
    /* local candidates of each per-label quantity (the global one is reduced below) */
    const int NL = COLL_NL;
    double * L = (double *) mymalloc("SubCollRadL", sizeof(double) * NL * (nc + 1));
    double * G = (double *) mymalloc("SubCollRadG", sizeof(double) * NL * (nc + 1));
    /* L: 0 hbt best, 1 vir best, 2 rhalf position (min), 3 Renclose (max r), 4.. kc[s] (max), (4 + NSO).. unused */
    for(c = 0; c < nc; c++) {
        double * l = L + NL * c;
        l[0] = -1;
        l[1] = 0;
        l[2] = 1e300;
        l[3] = -1;
        for(s = 0; s < SUBHALO_NSO; s++)
            l[4 + s] = -1;
    }
    double * Mc = (double *) mymalloc("SubCollRadMc", sizeof(double) * (nb + 1));
    {
        int64_t lab = -1, j = 0;
        double Mk[2] = {0, 0};
        for(k = 0; k < nb; k++) {
            c = (int64_t) rr[k].key[2];
            if(c != lab) {
                lab = c;
                j = 0;
                Mk[0] = msum[2 * nc + 2 * c];
                Mk[1] = msum[2 * nc + 2 * c + 1];
            }
            pos[k] = cnt[nc + c] + j++;
            subhalo_ksum_add(Mk, rr[k].m);
            const double M = subhalo_ksum_value(Mk);
            Mc[k] = M;
            const double r = rr[k].r;
            const double Mtot = rp->hs[NHS * c + HS_M];
            const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
            double * l = L + NL * c;
            for(s = 0; s < SUBHALO_NSO; s++)
                if(dens > rho[s])
                    l[4 + s] = (double) pos[k];
            if(M / r > l[0])
                l[0] = M / r;
            if(dens > rho[0] && M / r > l[1])
                l[1] = M / r;
            if(M >= 0.5 * Mtot && (double) pos[k] < l[2])
                l[2] = (double) pos[k];
            if(r > l[3])
                l[3] = r;
        }
    }
    /* maxima of the values and the crossing positions, the minimum of the half-mass position */
    double * MX = (double *) mymalloc("SubCollRadMx", sizeof(double) * NL * (nc + 1));
    for(c = 0; c < nc; c++) {
        double * l = L + NL * c, * g = G + NL * c;
        g[0] = l[0];
        g[1] = l[1];
        g[2] = -l[2];
        g[3] = l[3];
        for(s = 0; s < SUBHALO_NSO; s++)
            g[4 + s] = l[4 + s];
    }
    MPI_Allreduce(G, MX, NL * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    /* second scan: the position attaining each maximum first, the masses at the crossings, the radii */
    for(c = 0; c < nc; c++) {
        double * g = G + NL * c;
        for(d = 0; d < NL; d++)
            g[d] = 0;
        g[0] = g[1] = 1e300;        /* positions of the first maximum: min */
    }
    for(k = 0; k < nb; k++) {
        c = (int64_t) rr[k].key[2];
        const double * mx = MX + NL * c;
        double * g = G + NL * c;
        const double r = rr[k].r, M = Mc[k];
        const double dens = M / (4.0 / 3.0 * M_PI * r * r * r);
        if(M / r == mx[0] && (double) pos[k] < g[0])
            g[0] = (double) pos[k];
        if(mx[1] > 0 && dens > rho[0] && M / r == mx[1] && (double) pos[k] < g[1])
            g[1] = (double) pos[k];
    }
    double * G2 = (double *) mymalloc("SubCollRadG2", sizeof(double) * NL * (nc + 1));
    for(c = 0; c < nc; c++) {
        G[NL * c] = -G[NL * c];
        G[NL * c + 1] = -G[NL * c + 1];
    }
    MPI_Allreduce(G, G2, NL * nc, MPI_DOUBLE, MPI_MAX, C->Comm);
    for(c = 0; c < nc; c++) {
        double * g = G + NL * c;
        for(d = 0; d < NL; d++)
            g[d] = 0;
    }
    /* sums of the values at the chosen positions (one particle each) */
    for(k = 0; k < nb; k++) {
        c = (int64_t) rr[k].key[2];
        const double * mx = MX + NL * c, * p2 = G2 + NL * c;
        double * g = G + NL * c;
        const double pk = (double) pos[k];
        if(pk == -p2[0]) g[0] += rr[k].r;                 /* Rmax */
        if(pk == -p2[1]) g[1] += rr[k].r;                 /* RmaxVir */
        if(pk == -mx[2]) g[2] += rr[k].r;                 /* Rhalf */
        for(s = 0; s < SUBHALO_NSO; s++)
            if(mx[4 + s] >= 0 && pk == mx[4 + s])
                g[4 + s] += Mc[k];                         /* Mso */
    }
    MPI_Allreduce(G, G2, NL * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
    for(c = 0; c < nc; c++) {
        const double * mx = MX + NL * c, * g = G2 + NL * c;
        double * pr = rp->prof + 8 * c;
        pr[0] = mx[0];                  /* best M/r (HBT, all bound) */
        pr[1] = g[0];                   /* Rmax */
        pr[2] = mx[1];                  /* best M/r inside the virial threshold, 0 if none */
        pr[3] = mx[1] > 0 ? g[1] : -1;  /* RmaxVir */
        pr[4] = g[2];                   /* Rhalf */
        pr[5] = mx[3];                  /* Renclose */
        for(s = 0; s < SUBHALO_NSO; s++) {
            rp->kc[SUBHALO_NSO * c + s] = (int64_t) mx[4 + s];
            rp->Mso[SUBHALO_NSO * c + s] = mx[4 + s] >= 0 ? g[4 + s] : 0;
        }
    }
    myfree(G2);
    myfree(MX);
    /* inside the virial aperture (crossing particle included): JVir in the CoreVel frame and the
     * mean offsets (Rockstar properties.c:417-458) */
    {
        double * jl = (double *) mymalloc("SubCollJvLoc", sizeof(double) * 9 * (nc + 1));
        memset(jl, 0, sizeof(double) * 9 * (nc + 1));
        for(k = 0; k < nb; k++) {
            c = (int64_t) rr[k].key[2];
            if(pos[k] > rp->kc[SUBHALO_NSO * c])
                continue;
            const struct SubhaloCore * cp = &core->d[c].core;
            double dvk[3];
            for(d = 0; d < 3; d++)
                dvk[d] = rr[k].vel[d] - cp->Vel[d];
            const double m = rr[k].m;
            double * j = jl + 9 * c;
            j[0] += m * (rr[k].dx[1] * dvk[2] - rr[k].dx[2] * dvk[1]);
            j[1] += m * (rr[k].dx[2] * dvk[0] - rr[k].dx[0] * dvk[2]);
            j[2] += m * (rr[k].dx[0] * dvk[1] - rr[k].dx[1] * dvk[0]);
            for(d = 0; d < 3; d++) {
                j[3 + d] += m * rr[k].dx[d];
                j[6 + d] += m * dvk[d];
            }
        }
        MPI_Allreduce(jl, rp->jv, 9 * nc, MPI_DOUBLE, MPI_SUM, C->Comm);
        myfree(jl);
    }
    /* the deferred recipes: Rs fit, RockstarCoreVel, BulkVel, Vrms */
    deferred_distributed(C, core, rr, nb, pos, Mc, rp);
    /* shapes at Rvir and R500c over the particles up to each crossing */
    {
        int64_t * np = (int64_t *) mymalloc("SubCollShNp", sizeof(int64_t) * (nc + 1));
        double * R = (double *) mymalloc("SubCollShR", sizeof(double) * (nc + 1));
        int ap;
        for(ap = 0; ap < 2; ap++) {
            const int sdef = ap == 0 ? 0 : 3;
            for(c = 0; c < nc; c++) {
                const int64_t kcs = rp->kc[SUBHALO_NSO * c + sdef];
                np[c] = kcs >= 0 ? kcs + 1 : 0;
                R[c] = kcs >= 0 ? cbrt(3 * rp->Mso[SUBHALO_NSO * c + sdef] / (4 * M_PI * rho[sdef])) : -1;
            }
            shapes_distributed(C, rr, nb, pos, nc, np, R, rfloor, rp->shape + 5 * ap, 10);
        }
        myfree(R);
        myfree(np);
    }
    myfree(Mc);
    myfree(G);
    myfree(L);
    myfree(pos);
    myfree(msum);
    myfree(cnt);
    myfree(rr);
}

void
free_round_props(struct RoundProps * rp)
{
    myfree(rp->dfr);
    myfree(rp->shape);
    myfree(rp->jv);
    myfree(rp->prof);
    myfree(rp->kc);
    myfree(rp->Mso);
    myfree(rp->bhid);
    myfree(rp->bhm);
    myfree(rp->hs);
}

/* ------------------------------------------------------------------ test hook: the round's properties */

void
subhalo_collective_test_props(const int32_t * label, int64_t nlabels, const double * core, const struct SubhaloUnbindParams * up,
                              const double rho[SUBHALO_NSO], double * dfr, double * mvir)
{
    struct Coll C;
    struct SubhaloCallCtx ctx;
    struct CoreOut co;
    struct RoundProps rp;
    int64_t i, c;
    int d;
    memset(&C, 0, sizeof(C));
    memset(&ctx, 0, sizeof(ctx));
    memset(&co, 0, sizeof(co));
    for(d = 0; d < SUBHALO_NSO; d++)
        ctx.rho[d] = rho[d];
    C.ctx = &ctx;
    C.up = up;
    C.Comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &C.NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &C.ThisTask);
    C.NumPart = PartManager->NumPart;
    C.c0 = 0;
    C.c1 = nlabels;
    C.cs = (struct CandState *) mymalloc("SubTestPropsCs", sizeof(struct CandState) * (nlabels + 1));
    memset(C.cs, 0, sizeof(struct CandState) * (nlabels + 1));
    co.d = (struct CoreRow *) mymalloc("SubTestPropsCore", sizeof(struct CoreRow) * (nlabels + 1));
    memset(co.d, 0, sizeof(struct CoreRow) * (nlabels + 1));
    for(c = 0; c < nlabels; c++) {
        C.cs[c].state = SUBHALO_ST_ALIVE;
        for(d = 0; d < 3; d++) {
            C.cs[c].RefPos[d] = co.d[c].core.Pos[d] = core[6 * c + d];
            C.cs[c].RefVel[d] = co.d[c].core.Vel[d] = core[6 * c + 3 + d];
        }
    }
    C.Cand = (int32_t *) mymalloc("SubTestPropsCand", sizeof(int32_t) * (C.NumPart + 1));
    C.Flag = (uint8_t *) mymalloc("SubTestPropsFlag", C.NumPart + 1);
    C.En = (double *) mymalloc("SubTestPropsEn", sizeof(double) * (C.NumPart + 1));
    for(i = 0; i < C.NumPart; i++) {
        C.Cand[i] = label[i] >= 0 && label[i] < nlabels ? label[i] : -1;
        C.Flag[i] = C.Cand[i] >= 0 ? CF_BOUND : 0;
        C.En[i] = 0;
    }
    round_props(&C, &co, &rp);
    for(c = 0; c < 8 * nlabels; c++)
        dfr[c] = rp.dfr[c];
    if(mvir)
        for(c = 0; c < nlabels; c++)
            mvir[c] = rp.kc[SUBHALO_NSO * c] >= 0 ? rp.Mso[SUBHALO_NSO * c] : -1;
    free_round_props(&rp);
    myfree(C.En);
    myfree(C.Flag);
    myfree(C.Cand);
    myfree(co.d);
    myfree(C.cs);
}
