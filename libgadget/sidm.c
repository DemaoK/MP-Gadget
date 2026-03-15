#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gravity.h"
#include "physconst.h"
#include "sidm.h"
#include "timebinmgr.h"
#include "timefac.h"
#include "treewalk.h"
#include "utils.h"
#include "utils/spinlocks.h"
#include "utils/system.h"
#include "walltime.h"

/*! \file sidm.c
 *  \brief Compute self-interacting dark matter scattering impulses via
 * pairwise interactions. The comoving SIDM solve proceeds in two phases:
 * 1. Find candidate neighbors and compute tentative scatter weights.
 * 2. Resolve accepted pair scatters, handle conflicts, and apply impulses.
 */

/* SIDMFULLCHECK enables verbose per-scatter logging and implicitly keeps the
 * cheaper SIDMCHECK counters/summaries enabled as well. */
#if defined(SIDMFULLCHECK) && !defined(SIDMCHECK)
#define SIDMCHECK
#endif

#define MAX_SIDM_NEIGHBORS                                                     \
  38                   /* Maximum number of neighbors for SIDM interactions*/
#define ALPHA_FACTOR 3 /* Weighting factor in the SIDM partner-selection draw  \
                        */
#define SIDM_NO_PARTNER ((MyIDType)(-1)) /* Sentinel for no chosen SIDM partner. */
/* Piecewise-linear lookup for the vdSIDM shape factor as a function of vw^2.
 * The table is dense near vw^2 ~ 0, where the function changes rapidly, and
 * coarse at large vw^2, where it is smooth. Values beyond the last edge fall
 * back to the exact expression. */
#define SIDM_VD_SIGMA_LOOKUP_SEGMENTS 6
#define SIDM_VD_SIGMA_LOOKUP_BINS 2048

/* This structure stores fixed parameters for the SIDM module. The idea is that
 * these are specified in the parameter file and the values never change over
 * the course of a simulation.*/
static struct sidm_params {
  /* Enables SIDM */
  int SIDMOn;
  int vdSIDMOn;
  double sigma0;
  double wTurn;
  double MultiscatteringThreshold; /* If >= 0, keep a conflicting step-2 scatter
                                      with this probability. If < 0, use strict
                                      single-scatter claim mode. */
} SIDMParams;

static const double
    SIDMVdSigmaShapeEdges[SIDM_VD_SIGMA_LOOKUP_SEGMENTS + 1] = {
        0.0, 1.0, 16.0, 256.0, 4096.0, 65536.0, 1048576.0};
static float SIDMVdSigmaShapeLookup[SIDM_VD_SIGMA_LOOKUP_SEGMENTS]
                                   [SIDM_VD_SIGMA_LOOKUP_BINS + 1];
static double SIDMVdSigmaShapeLookupInvDx[SIDM_VD_SIGMA_LOOKUP_SEGMENTS];
static int SIDMVdSigmaShapeLookupReady = 0;

static inline double sidm_vd_sigma_shape_exact(double vw2) {
  if (vw2 <= 0)
    return 1.0;
  /* Series expansion of the exact expression around vw2 = 0:
   * 1 - vw2 + 9/10 vw2^2 + O(vw2^3). */
  if (vw2 < 1e-4)
    return 1.0 - vw2 + 0.9 * vw2 * vw2;

  const double num = (2.0 + vw2) * log1p(vw2) - 2.0 * vw2;
  return 6.0 * num / (vw2 * vw2 * vw2);
}

static void sidm_init_vd_sigma_lookup(void) {
  if (SIDMVdSigmaShapeLookupReady)
    return;

  for (int seg = 0; seg < SIDM_VD_SIGMA_LOOKUP_SEGMENTS; seg++) {
    const double xmin = SIDMVdSigmaShapeEdges[seg];
    const double xmax = SIDMVdSigmaShapeEdges[seg + 1];
    const double dx = (xmax - xmin) / SIDM_VD_SIGMA_LOOKUP_BINS;
    SIDMVdSigmaShapeLookupInvDx[seg] = 1.0 / dx;
    for (int i = 0; i <= SIDM_VD_SIGMA_LOOKUP_BINS; i++) {
      SIDMVdSigmaShapeLookup[seg][i] =
          (float)sidm_vd_sigma_shape_exact(xmin + i * dx);
    }
  }
  SIDMVdSigmaShapeLookupReady = 1;
}

static inline int sidm_vd_sigma_lookup_segment(double vw2) {
  if (vw2 < SIDMVdSigmaShapeEdges[1])
    return 0;
  if (vw2 < SIDMVdSigmaShapeEdges[2])
    return 1;
  if (vw2 < SIDMVdSigmaShapeEdges[3])
    return 2;
  if (vw2 < SIDMVdSigmaShapeEdges[4])
    return 3;
  if (vw2 < SIDMVdSigmaShapeEdges[5])
    return 4;
  if (vw2 < SIDMVdSigmaShapeEdges[6])
    return 5;
  return -1;
}

static inline double sidm_vd_sigma_shape_lookup(double vw2, int seg) {
  const double xmin = SIDMVdSigmaShapeEdges[seg];
  const double pos = (vw2 - xmin) * SIDMVdSigmaShapeLookupInvDx[seg];
  int idx = (int)pos;
  if (idx < 0)
    idx = 0;
  if (idx >= SIDM_VD_SIGMA_LOOKUP_BINS)
    idx = SIDM_VD_SIGMA_LOOKUP_BINS - 1;
  double frac = pos - idx;
  if (frac < 0)
    frac = 0;
  if (frac > 1)
    frac = 1;
  const double y0 = SIDMVdSigmaShapeLookup[seg][idx];
  const double y1 = SIDMVdSigmaShapeLookup[seg][idx + 1];
  return y0 + frac * (y1 - y0);
}

/* Read and broadcast the run-global SIDM parameters. */
void set_sidm_params(ParameterSet *ps) {
  int ThisTask;
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  if (ThisTask == 0) {
    SIDMParams.SIDMOn = param_get_int(ps, "SIDMOn");
    SIDMParams.vdSIDMOn = param_get_int(ps, "vdSIDMOn");
    SIDMParams.sigma0 = param_get_double(ps, "sigma0");
    SIDMParams.wTurn = param_get_double(ps, "wTurn");
    SIDMParams.MultiscatteringThreshold =
        param_get_double(ps, "MultiscatteringThreshold");
    if (SIDMParams.vdSIDMOn && SIDMParams.wTurn == 0.0) {
      endrun(1, "vdSIDMOn=1 requires non-zero wTurn. Please set wTurn in the "
                "parameter file.\n");
    }
  }
  MPI_Bcast(&SIDMParams, sizeof(struct sidm_params), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  if (SIDMParams.vdSIDMOn)
    sidm_init_vd_sigma_lookup();
}

/* Cubic-spline smoothing kernel used for the SIDM pair weighting.
 * Adopted from arXiv:1201.5892. */
double CubicSplineKernel(double r, double h) {
  double q = r / h;
  if (q < 0.5)
    return 1.0 - 6.0 * q * q + 6.0 * q * q * q;
  else if (q < 1.0)
    return 2.0 * (1.0 - q) * (1.0 - q) * (1.0 - q);
  else
    return 0;
}
/*******************************************************************/

/* Shared module state for a single sidm_force() call. */
struct SIDMPriv {
  /* This stores the 'global variables' associated with the SIDM module.
   * Cosmological time could be here, as an example.*/
  double hubble;
  double atime;
  struct UnitSystem units;
  RandTable *rnd;
  Cosmology *CP;
  /* Kick/drift helper state shared with other force modules. */
  DriftKickTimes const *times;
  struct kick_factor_data kf;
  /* Map local particle index -> SIDM-active slot (for step-1 partner draw). */
  int *ActiveSlotByPart;
  int *CandidateCount;
  float *CandidateDist;
  double *CandidateWeight;
  MyIDType *CandidateID;
  /* Cached conversion for sigma/m in code units. */
  double SigmaUnitFac;
  /* Cached constant sigma/m in code units when vdSIDMOn == 0. */
  double Sigma0Const;
  /* Cached vdSIDM turnover velocity squared in internal code units. */
  double WTurnCode2;
  /* Cached exact drift factor for each gravity timebin in this step. */
  double DriftFactorByTimeBin[TIMEBINS + 1];
  /* Per-particle locks for scatter claim. */
  struct SpinLocks *PartLocks;
#ifdef SIDMCHECK
  /* Per-step debug counters for accepted and suppressed scatters. */
  int64_t Step1ProbRejected;
  int64_t Step2Accepted;
  int64_t Step2Suppressed;
  int64_t Step2SuppressedClaimedPartner;
  int64_t Step2SuppressedThreshold;
  int64_t Step2MutualHighIdSkip;
  /* Per-thread vdSIDM lookup usage counters for step 1. */
  int VdDebugThreads;
  int64_t *Step1VdLookupSegmentHits;
  int64_t *Step1VdExactTailHits;
#endif
};

/* Access the treewalk-private SIDM state. */
#define SIDM_GET_PRIV(tw) ((struct SIDMPriv *)((tw)->priv))

/* Shared neighbor-iterator state for both SIDM treewalk phases. */
typedef struct {
  TreeWalkNgbIterBase base;
} TreeWalkNgbIterSIDM;

/* Returns 1 if a particle needs SIDM work in the current step, 0 otherwise. */
static int sidm_haswork(int n, TreeWalk *tw);
static int sidm_haswork_scatter(int n, TreeWalk *tw);

/* This struct stores bits of a particle which are used as *inputs* to the SIDM
 * functions. sidm_copy_ngb is supposed to copy the fields from the particle
 * structure P into this struct, and is run for each particle. */
/* Step 1 func*/
typedef struct {
  TreeWalkQueryBase base;
  MyFloat Vel[3];
  double Hsml;
  double Mass;
  double DriftFactor;
  MyIDType ID;
} TreeWalkQuerySIDM_NGB;

/* This struct stores bits of a particle which are used as *outputs* to the SIDM
 * functions. sidm_reduce is supposed to copy the fields from this struct back
 * into the particle structure P, and is run for each particle. */
/* Holds up to N candidate neighbors for step-1 scattering evaluation. */
/* Step 1 func*/
typedef struct {
  TreeWalkResultBase base;
  int nCandidates;
  float candidate_dist[MAX_SIDM_NEIGHBORS];
  /* Velocity-dependent pair prefactor before the kernel normalization. */
  float candidate_pair_prefac[MAX_SIDM_NEIGHBORS];
  MyIDType candidate_id[MAX_SIDM_NEIGHBORS];
} TreeWalkResultSIDM_NGB;

static inline double
sidm_result_max_candidate_dist(const TreeWalkResultSIDM_NGB *O) {
  if (O->nCandidates <= 0)
    return 0;

  double max_dist = O->candidate_dist[0];
  for (int i = 1; i < O->nCandidates; i++) {
    if (O->candidate_dist[i] > max_dist)
      max_dist = O->candidate_dist[i];
  }
  return max_dist;
}

/*Step 1 function*/

static void sidm_ngbiter_ngb(TreeWalkQuerySIDM_NGB *I,
                             TreeWalkResultSIDM_NGB *O,
                             TreeWalkNgbIterSIDM *iter, LocalTreeWalk *lv);

/* Step 1 function*/
static void sidm_copy_ngb(int place, TreeWalkQuerySIDM_NGB *input,
                          TreeWalk *tw);

/* Step 1 function*/
static void sidm_reduce_ngb(int place, TreeWalkResultSIDM_NGB *result,
                            enum TreeWalkReduceMode mode, TreeWalk *tw);
static void sidm_postprocess_ngb(int place, TreeWalk *tw);

/*******************************************************************/

/* This struct stores bits of a particle which are used as *inputs* to the SIDM
 * functions. sidm_copy_scatter is supposed to copy the fields from the particle
 * structure P into this struct, and is run for each particle. */

/* Step 2 func*/
typedef struct {
  TreeWalkQueryBase base;
  MyFloat Vel[3];
  double Hsml;
  double Mass;
  MyIDType ID;
/* SIDM-related quantities from step 1 */
#ifdef SIDM
  double SIDMProb;
  MyIDType Partner;
  int Scattered;
#endif
} TreeWalkQuerySIDM_SCATTER;

/* This struct stores bits of a particle which are used as *outputs* to the SIDM
 * functions. sidm_reduce is supposed to copy the fields from this struct back
 * into the particle structure P, and is run for each particle. */
/* We use this to store actual scattering results. */
/* Step 2 func*/
typedef struct {
  TreeWalkResultBase base;
  double SIDMAccel[3];
  double SIDMProb;
  MyIDType Partner;
  int Update;         /* Flag to update the particle*/
  int SuppressReason; /* 0 none, 1 partner-claimed, 2 threshold */
} TreeWalkResultSIDM_SCATTER;

/* Step 2 function.
 * Resolve accepted pair interactions, including conflict handling for shared
 * partners, and produce velocity impulses. */
static void sidm_ngbiter_scatter(TreeWalkQuerySIDM_SCATTER *I,
                                 TreeWalkResultSIDM_SCATTER *O,
                                 TreeWalkNgbIterSIDM *iter, LocalTreeWalk *lv);

/* Step 2 function*/
static void sidm_copy_scatter(int place, TreeWalkQuerySIDM_SCATTER *input,
                              TreeWalk *tw);

/* Step 2 function*/

static void sidm_reduce_scatter(int place, TreeWalkResultSIDM_SCATTER *result,
                                enum TreeWalkReduceMode mode, TreeWalk *tw);
#ifdef SIDMCHECK
static void sidm_debug_check_accel_conservation(void);
static void sidm_debug_report_unit_conversions(struct SIDMPriv *priv,
                                               double atime);
static void sidm_debug_report_step_stats(struct SIDMPriv *priv, double atime);
#endif

/* SIDM scattering is restricted to gravity-active DM particles in this step. */
static inline int sidm_is_active_dm(const int i, const DriftKickTimes *times) {
  if (P[i].Type != 1 || P[i].IsGarbage || P[i].Swallowed)
    return 0;
  const int bin = P[i].TimeBinGravity;
  if (bin < 0 || bin > TIMEBINS)
    return 0;
  return is_timebin_active(bin, times->Ti_Current);
}

/* Deterministic hash seed for an unordered particle pair. */
static inline uint64_t sidm_pair_seed(MyIDType id_a, MyIDType id_b,
                                      uint64_t salt) {
  uint64_t lo = (uint64_t)id_a;
  uint64_t hi = (uint64_t)id_b;
  if (lo > hi) {
    uint64_t tmp = lo;
    lo = hi;
    hi = tmp;
  }
  return (lo * 11400714819323198485ull) ^ (hi * 14029467366897019727ull) ^ salt;
}

/* Keep a global nearest-candidate pool (capped at MAX_SIDM_NEIGHBORS) per
 * active particle across all reduce() calls (primary + ghost reductions). */
static inline void sidm_store_candidate(struct SIDMPriv *priv, int place,
                                        double dist, MyIDType partner_id,
                                        double weight) {
  if (weight <= 0 || !priv->ActiveSlotByPart)
    return;

  const int slot = priv->ActiveSlotByPart[place];
  if (slot < 0)
    return;

  const size_t base = (size_t)slot * MAX_SIDM_NEIGHBORS;
  int n = priv->CandidateCount[slot];
  float *const dists = priv->CandidateDist + base;
  double *const weights = priv->CandidateWeight + base;
  MyIDType *const ids = priv->CandidateID + base;

  /* Defensive merge for duplicate partner IDs. */
  for (int j = 0; j < n; j++) {
    if (ids[j] == partner_id) {
      if ((float)dist < dists[j])
        dists[j] = (float)dist;
      weights[j] += weight;
      return;
    }
  }

  if (n < MAX_SIDM_NEIGHBORS) {
    dists[n] = (float)dist;
    weights[n] = weight;
    ids[n] = partner_id;
    priv->CandidateCount[slot] = n + 1;
    return;
  }

  int far_idx = 0;
  float far_dist = dists[0];
  MyIDType far_id = ids[0];
  for (int j = 1; j < MAX_SIDM_NEIGHBORS; j++) {
    if (dists[j] > far_dist || (dists[j] == far_dist && ids[j] > far_id)) {
      far_idx = j;
      far_dist = dists[j];
      far_id = ids[j];
    }
  }

  if ((float)dist < far_dist ||
      ((float)dist == far_dist && partner_id < ids[far_idx])) {
    dists[far_idx] = (float)dist;
    weights[far_idx] = weight;
    ids[far_idx] = partner_id;
  }
}

/*******************************************************************/
/*! Driver routine for the SIDM scatter solve on gravity-active DM particles.
 */
void sidm_force(const ActiveParticles *act, const double atime,
                const double hubble, Cosmology *CP, RandTable *rnd,
                const struct UnitSystem units, DriftKickTimes *times,
                const ForceTree *const tree) {
#ifdef SIDMFULLCHECK
  message(0, "--------IN SIDM_FORCE --------\n");
#endif

  if (SIDMParams.vdSIDMOn && !SIDMVdSigmaShapeLookupReady)
    sidm_init_vd_sigma_lookup();

  int i;
  struct SIDMPriv priv[1] = {0};
  priv->atime = atime;
  priv->CP = CP;
  priv->hubble = hubble;
  priv->units = units;
  priv->rnd = rnd;
  priv->times = times;
  init_kick_factor_data(&priv->kf, times, CP);
  if (SIDMParams.MultiscatteringThreshold < 0 && PartManager->MaxPart > 0)
    priv->PartLocks = init_spinlocks(PartManager->MaxPart);
#ifdef SIDMCHECK
  if (SIDMParams.vdSIDMOn) {
    const size_t nthreads = (size_t)omp_get_max_threads();
    priv->VdDebugThreads = (int)nthreads;
    priv->Step1VdLookupSegmentHits =
        mymalloc("SIDMVDLookupSegmentHits",
                 nthreads * SIDM_VD_SIGMA_LOOKUP_SEGMENTS * sizeof(int64_t));
    memset(priv->Step1VdLookupSegmentHits, 0,
           nthreads * SIDM_VD_SIGMA_LOOKUP_SEGMENTS * sizeof(int64_t));
    priv->Step1VdExactTailHits = mymalloc(
        "SIDMVDExactTailHits", nthreads * sizeof(int64_t));
    memset(priv->Step1VdExactTailHits, 0, nthreads * sizeof(int64_t));
  }
#endif
  priv->SigmaUnitFac = CP->HubbleParam * units.UnitMass_in_g /
                       (units.UnitLength_in_cm * units.UnitLength_in_cm) /
                       (atime * atime);
  priv->Sigma0Const = SIDMParams.sigma0 * priv->SigmaUnitFac;
  /* Convert physical wTurn (km/s) to internal velocity units.
   * For comoving integration, internal velocities carry an extra factor of a.
   */
  {
    const double wturn_physical_cgs = SIDMParams.wTurn * 1e5;
    const double wturn_internal =
        (wturn_physical_cgs / units.UnitVelocity_in_cm_per_s) * atime;
    priv->WTurnCode2 = wturn_internal * wturn_internal;
  }
  for (i = 0; i <= TIMEBINS; i++) {
    inttime_t dti = dti_from_timebin(i);
    inttime_t t_start = times->Ti_kick[i] - dti / 2;
    inttime_t t_end = times->Ti_kick[i] + dti / 2;
    priv->DriftFactorByTimeBin[i] = get_exact_drift_factor(CP, t_start, t_end);
  }
  if (PartManager->NumPart > 0) {
    priv->ActiveSlotByPart = (int *)mymalloc(
        "SIDMActiveSlotByPart", (size_t)PartManager->NumPart * sizeof(int));
    memset(priv->ActiveSlotByPart, 0xFF,
           (size_t)PartManager->NumPart * sizeof(int));
  }

  /* Build local SIDM-active mapping for step-1 partner accumulation. */
  int sidm_active_count = 0;
  if (priv->ActiveSlotByPart) {
    for (i = 0; i < act->NumActiveParticle; i++) {
      int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
      if (!sidm_is_active_dm(p, times))
        continue;
      priv->ActiveSlotByPart[p] = sidm_active_count++;
    }
  }
  if (sidm_active_count > 0) {
    const size_t nslots = (size_t)sidm_active_count;
    const size_t nentries = nslots * MAX_SIDM_NEIGHBORS;
    priv->CandidateCount =
        (int *)mymalloc("SIDMCandidateCount", nslots * sizeof(int));
    priv->CandidateDist =
        (float *)mymalloc("SIDMCandidateDist", nentries * sizeof(float));
    priv->CandidateWeight =
        (double *)mymalloc("SIDMCandidateWeight", nentries * sizeof(double));
    priv->CandidateID =
        (MyIDType *)mymalloc("SIDMCandidateID", nentries * sizeof(MyIDType));

    memset(priv->CandidateCount, 0, nslots * sizeof(int));
  }
  /* --- First phase: Collect candidate lists and evaluate tentative scatters.
   * This does not yet resolve partner conflicts/multiple-scatter suppression.
   */
  {
    TreeWalk tw_ngb[1] = {{0}};

    tw_ngb->ev_label = "SIDM_NGB";
    /* Allow the walk radius to shrink to the farthest retained candidate. */
    tw_ngb->visit = (TreeWalkVisitFunction)treewalk_visit_nolist_ngbiter;
    tw_ngb->ngbiter = (TreeWalkNgbIterFunction)sidm_ngbiter_ngb;
    tw_ngb->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSIDM);
    tw_ngb->NoNgblist = 1;
    tw_ngb->haswork = sidm_haswork;
    tw_ngb->fill = (TreeWalkFillQueryFunction)sidm_copy_ngb;
    tw_ngb->reduce = (TreeWalkReduceResultFunction)sidm_reduce_ngb;
    tw_ngb->postprocess = (TreeWalkProcessFunction)sidm_postprocess_ngb;
    tw_ngb->query_type_elsize = sizeof(TreeWalkQuerySIDM_NGB);
    tw_ngb->result_type_elsize = sizeof(TreeWalkResultSIDM_NGB);
    tw_ngb->tree = tree;
    tw_ngb->priv = priv;

    /* Initialize the step-local search radius and SIDM bookkeeping. The
     * initial search radius matches the short-range force softening scale. */

    double global_closeness = FORCE_SOFTENING();
#pragma omp parallel for
    for (i = 0; i < act->NumActiveParticle; i++) {
      int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
      if (sidm_haswork(p, tw_ngb)) {
        P[p].Hsml = global_closeness;
        /* Initialize particle SIDM state for this step. */
#ifdef SIDM
        P[p].Partner = SIDM_NO_PARTNER;
        P[p].SIDMPartnerDist = 0;
        P[p].SIDMAccel[0] = 0;
        P[p].SIDMAccel[1] = 0;
        P[p].SIDMAccel[2] = 0;
        P[p].SIDMProb = -1;
        P[p].Scattered = 0;
#endif
      }
    }

    treewalk_run(tw_ngb, act->ActiveParticle, act->NumActiveParticle);

    walltime_measure("/SPH/SIDM_NGB");
  }

  /* --- Second phase: resolve accepted pairs and apply conflict policy. --- */
  TreeWalk tw_scatter[1] = {{0}};

  tw_scatter->ev_label = "SIDM_SCATTER";
  tw_scatter->visit = (TreeWalkVisitFunction)treewalk_visit_ngbiter;
  tw_scatter->ngbiter = (TreeWalkNgbIterFunction)sidm_ngbiter_scatter;
  tw_scatter->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSIDM);
  tw_scatter->haswork = sidm_haswork_scatter;
  tw_scatter->fill = (TreeWalkFillQueryFunction)sidm_copy_scatter;
  tw_scatter->reduce = (TreeWalkReduceResultFunction)sidm_reduce_scatter;
  tw_scatter->postprocess = NULL;
  tw_scatter->query_type_elsize = sizeof(TreeWalkQuerySIDM_SCATTER);
  tw_scatter->result_type_elsize = sizeof(TreeWalkResultSIDM_SCATTER);
  tw_scatter->tree = tree;
  tw_scatter->priv = priv;

  /* Phase 2 reuses the chosen partner distance from phase 1 as a tight search
   * radius around the accepted pair. */
  SIDM_GET_PRIV(tw_scatter)->atime = atime;
  treewalk_run(tw_scatter, act->ActiveParticle, act->NumActiveParticle);

  walltime_measure("/SPH/SIDM_SCATTER");
#ifdef SIDMCHECK
  sidm_debug_check_accel_conservation();
  sidm_debug_report_unit_conversions(priv, atime);
  sidm_debug_report_step_stats(priv, atime);
#endif
  if (priv->CandidateID)
    myfree(priv->CandidateID);
  if (priv->CandidateWeight)
    myfree(priv->CandidateWeight);
  if (priv->CandidateDist)
    myfree(priv->CandidateDist);
  if (priv->CandidateCount)
    myfree(priv->CandidateCount);
  if (priv->ActiveSlotByPart)
    myfree(priv->ActiveSlotByPart);
#ifdef SIDMCHECK
  if (priv->Step1VdExactTailHits)
    myfree(priv->Step1VdExactTailHits);
  if (priv->Step1VdLookupSegmentHits)
    myfree(priv->Step1VdLookupSegmentHits);
#endif
  if (priv->PartLocks)
    free_spinlocks(priv->PartLocks);
  priv->CandidateID = NULL;
  priv->CandidateWeight = NULL;
  priv->CandidateDist = NULL;
  priv->CandidateCount = NULL;
  priv->ActiveSlotByPart = NULL;
  priv->PartLocks = NULL;
#ifdef SIDMCHECK
  priv->Step1VdLookupSegmentHits = NULL;
  priv->Step1VdExactTailHits = NULL;
  priv->VdDebugThreads = 0;
#endif
  /* --- End of SIDM force calculation --- */
}

/*******************************************************************/

/* Step 1 operates only on gravity-active DM particles. */
static int sidm_haswork(int i, TreeWalk *tw) {
  const DriftKickTimes *times = SIDM_GET_PRIV(tw)->times;
  return sidm_is_active_dm(i, times);
}

/* Phase 2 only needs particles that survived the step-1 acceptance draw. */
static int sidm_haswork_scatter(int i, TreeWalk *tw) {
  if (!sidm_haswork(i, tw))
    return 0;
#ifdef SIDM
  return P[i].SIDMProb >= 0 && P[i].Partner != SIDM_NO_PARTNER;
#else
  return 0;
#endif
}

/*******************************************************************/

/* Copy step-1 query inputs from the particle table into the treewalk buffer. */
static void sidm_copy_ngb(int place, TreeWalkQuerySIDM_NGB *input,
                          TreeWalk *tw) {
  input->Vel[0] = P[place].Vel[0];
  input->Vel[1] = P[place].Vel[1];
  input->Vel[2] = P[place].Vel[2];
  input->Hsml = P[place].Hsml;
  input->Mass = P[place].Mass;
  input->DriftFactor =
      SIDM_GET_PRIV(tw)->DriftFactorByTimeBin[P[place].TimeBinGravity];
  input->ID = P[place].ID;
}

/* Accumulate tentative step-1 pair weights and candidate neighbors.
 * Step-1 partner draw is finalized in sidm_postprocess_ngb() using strict
 * nearest-to-farthest cumulative weighting. */
static void sidm_reduce_ngb(int place, TreeWalkResultSIDM_NGB *result,
                            enum TreeWalkReduceMode mode, TreeWalk *tw) {
  (void)mode;
  int nNGB = result->nCandidates;

  if (nNGB > 0) {
    /* W_ij is normalized with the farthest local neighbor distance. */
    double h_j = result->candidate_dist[0];
    for (int i = 1; i < nNGB; i++) {
      if (result->candidate_dist[i] > h_j)
        h_j = result->candidate_dist[i];
    }
    if (h_j > 0) {
      const double kernel_norm = 4.0 / (M_PI * h_j * h_j * h_j);
      for (int i = 0; i < nNGB; i++) {
        double spb = result->candidate_pair_prefac[i];
        double Wij = CubicSplineKernel(result->candidate_dist[i], h_j);
        /* See arXiv:1201.5892 Eq. 3 and arXiv:2205.03392 Eq. 2.10. */
        spb *= kernel_norm * Wij;

        if (spb <= 0)
          continue;

        const double w = 2.0 * ALPHA_FACTOR * spb;
        sidm_store_candidate(SIDM_GET_PRIV(tw), place,
                             result->candidate_dist[i], result->candidate_id[i],
                             w);
      }
    }
  }
}

#ifdef SIDMCHECK
static inline void sidm_debug_count_vd_lookup_segment(struct SIDMPriv *priv,
                                                      int seg) {
  if (!priv->Step1VdLookupSegmentHits)
    return;
  const int tid = omp_get_thread_num();
  priv->Step1VdLookupSegmentHits
      [(size_t)tid * SIDM_VD_SIGMA_LOOKUP_SEGMENTS + seg] += 1;
}

static inline void sidm_debug_count_vd_exact_tail(struct SIDMPriv *priv) {
  if (!priv->Step1VdExactTailHits)
    return;
  const int tid = omp_get_thread_num();
  priv->Step1VdExactTailHits[tid] += 1;
}

/* Debug-only check: SIDM acceleration impulse should conserve total momentum.
 * Checks global sum_d m * SIDMAccel_d ~= 0 for DM particles after each SIDM
 * step.
 */
static void sidm_debug_check_accel_conservation(void) {
  double local_px = 0, local_py = 0, local_pz = 0;
  double local_ax = 0, local_ay = 0, local_az = 0;
  int64_t local_nimp = 0;

#pragma omp parallel for reduction(+ : local_px, local_py, local_pz, local_ax, \
                                       local_ay, local_az, local_nimp)
  for (int i = 0; i < PartManager->NumPart; i++) {
    if (P[i].Type != 1 || P[i].IsGarbage || P[i].Swallowed)
      continue;
    const double mx = (double)P[i].Mass * (double)P[i].SIDMAccel[0];
    const double my = (double)P[i].Mass * (double)P[i].SIDMAccel[1];
    const double mz = (double)P[i].Mass * (double)P[i].SIDMAccel[2];

    local_px += mx;
    local_py += my;
    local_pz += mz;
    local_ax += fabs(mx);
    local_ay += fabs(my);
    local_az += fabs(mz);

    if (P[i].SIDMAccel[0] != 0 || P[i].SIDMAccel[1] != 0 ||
        P[i].SIDMAccel[2] != 0) {
      local_nimp++;
    }
  }

  double global_p[3] = {local_px, local_py, local_pz};
  double global_a[3] = {local_ax, local_ay, local_az};
  int64_t global_nimp = local_nimp;
  MPI_Allreduce(MPI_IN_PLACE, global_p, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, global_a, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_nimp, 1, MPI_INT64, MPI_SUM,
                MPI_COMM_WORLD);

  const double atol = 1e-14;
  const double rtol = 1e-10;
  const double tol_x = DMAX(atol, rtol * global_a[0]);
  const double tol_y = DMAX(atol, rtol * global_a[1]);
  const double tol_z = DMAX(atol, rtol * global_a[2]);

  if (fabs(global_p[0]) > tol_x || fabs(global_p[1]) > tol_y ||
      fabs(global_p[2]) > tol_z) {
    message(
        0,
        "WARNING: SIDM momentum check residual: sum(m*SIDMAccel)=(%g,%g,%g), "
        "abs-sum=(%g,%g,%g), tol=(%g,%g,%g), impacted=%ld\n",
        global_p[0], global_p[1], global_p[2], global_a[0], global_a[1],
        global_a[2], tol_x, tol_y, tol_z, global_nimp);
  }
}

/* Debug-only per-step dump of the SIDM unit conversions used in this step. */
static void sidm_debug_report_unit_conversions(struct SIDMPriv *priv,
                                               double atime) {
  int ThisTask;
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  if (ThisTask != 0)
    return;

  const double sigma0_roundtrip =
      priv->SigmaUnitFac > 0 ? priv->Sigma0Const / priv->SigmaUnitFac : 0.0;
  const double wturn_code = sqrt(DMAX(priv->WTurnCode2, 0.0));
  const double wturn_roundtrip =
      (atime > 0 && priv->units.UnitVelocity_in_cm_per_s > 0)
          ? (wturn_code / atime) * priv->units.UnitVelocity_in_cm_per_s / 1e5
          : 0.0;

  message(0,
          "SIDM unit conversion (a=%g): sigma0_phys=%g cm^2/g, "
          "sigma_unit_fac=%g, sigma0_code=%g, sigma0_roundtrip=%g cm^2/g, "
          "wTurn_phys=%g km/s, wTurn_code=%g, wTurn_roundtrip=%g km/s, "
          "units(L=%g cm, M=%g g, V=%g cm/s), h=%g\n",
          atime, SIDMParams.sigma0, priv->SigmaUnitFac, priv->Sigma0Const,
          sigma0_roundtrip, SIDMParams.wTurn, wturn_code, wturn_roundtrip,
          priv->units.UnitLength_in_cm, priv->units.UnitMass_in_g,
          priv->units.UnitVelocity_in_cm_per_s, priv->CP->HubbleParam);
}

/* Debug-only summary of accepted/suppressed SIDM outcomes for this step. */
static void sidm_debug_report_step_stats(struct SIDMPriv *priv, double atime) {
  int64_t local[6] = {
      priv->Step1ProbRejected,        priv->Step2Accepted,
      priv->Step2Suppressed,          priv->Step2SuppressedClaimedPartner,
      priv->Step2SuppressedThreshold, priv->Step2MutualHighIdSkip};
  int64_t global[6] = {0};
  int64_t local_vd_segments[SIDM_VD_SIGMA_LOOKUP_SEGMENTS] = {0};
  int64_t global_vd_segments[SIDM_VD_SIGMA_LOOKUP_SEGMENTS] = {0};
  int64_t local_vd_exact_tail = 0;
  int64_t global_vd_exact_tail = 0;
  MPI_Allreduce(local, global, 6, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

  if (priv->Step1VdLookupSegmentHits) {
    for (int tid = 0; tid < priv->VdDebugThreads; tid++) {
      local_vd_exact_tail += priv->Step1VdExactTailHits[tid];
      for (int seg = 0; seg < SIDM_VD_SIGMA_LOOKUP_SEGMENTS; seg++) {
        local_vd_segments[seg] +=
            priv->Step1VdLookupSegmentHits
                [(size_t)tid * SIDM_VD_SIGMA_LOOKUP_SEGMENTS + seg];
      }
    }
    MPI_Allreduce(local_vd_segments, global_vd_segments,
                  SIDM_VD_SIGMA_LOOKUP_SEGMENTS, MPI_INT64, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_vd_exact_tail, &global_vd_exact_tail, 1, MPI_INT64,
                  MPI_SUM, MPI_COMM_WORLD);
  }

  const int64_t attempted = global[1] + global[2];
  int64_t total_vd = global_vd_exact_tail;
  for (int seg = 0; seg < SIDM_VD_SIGMA_LOOKUP_SEGMENTS; seg++) {
    total_vd += global_vd_segments[seg];
  }

  if (attempted == 0 && global[0] == 0 && global[5] == 0 && total_vd == 0)
    return;

  message(0,
          "SIDM step summary (a=%g): accepted=%ld suppressed=%ld "
          "(prob-reject=%ld claimed-partner=%ld threshold=%ld) "
          "mutual-skip=%ld attempted=%ld\n",
          atime, global[1], global[2], global[0], global[3], global[4],
          global[5], attempted);

  if (total_vd > 0) {
    message(0,
            "SIDM vd-shape usage (a=%g, vw2): [0,1)=%ld [1,16)=%ld "
            "[16,256)=%ld [256,4096)=%ld [4096,65536)=%ld "
            "[65536,1.04858e+06)=%ld tail-exact=%ld (%.2f%%) total=%ld\n",
            atime, global_vd_segments[0], global_vd_segments[1],
            global_vd_segments[2], global_vd_segments[3],
            global_vd_segments[4], global_vd_segments[5], global_vd_exact_tail,
            100.0 * global_vd_exact_tail / total_vd, total_vd);
  }
}
#endif

/* Finalize step-1 scatter decision after all reductions are complete. */
static void sidm_postprocess_ngb(int place, TreeWalk *tw) {
#ifdef SIDM
  struct SIDMPriv *priv = SIDM_GET_PRIV(tw);
  if (!priv->ActiveSlotByPart || !priv->CandidateCount ||
      !priv->CandidateDist || !priv->CandidateWeight || !priv->CandidateID) {
    P[place].SIDMProb = -1;
    P[place].Partner = SIDM_NO_PARTNER;
    P[place].SIDMPartnerDist = 0;
    return;
  }

  const int slot = priv->ActiveSlotByPart[place];
  if (slot < 0) {
    P[place].SIDMProb = -1;
    P[place].Partner = SIDM_NO_PARTNER;
    P[place].SIDMPartnerDist = 0;
    return;
  }

  const int n = priv->CandidateCount[slot];
  if (n <= 0) {
    P[place].SIDMProb = -1;
    P[place].Partner = SIDM_NO_PARTNER;
    P[place].SIDMPartnerDist = 0;
    return;
  }

  const size_t base = (size_t)slot * MAX_SIDM_NEIGHBORS;
  const float *candidate_dist = priv->CandidateDist + base;
  const double *candidate_weight = priv->CandidateWeight + base;
  const MyIDType *candidate_id = priv->CandidateID + base;

  int order[MAX_SIDM_NEIGHBORS];
  for (int i = 0; i < n; i++)
    order[i] = i;

  /* Enforce strict nearest-first traversal for partner draw. */
  for (int i = 0; i < n - 1; i++) {
    int best = i;
    for (int j = i + 1; j < n; j++) {
      const int oi = order[best];
      const int oj = order[j];
      if (candidate_dist[oj] < candidate_dist[oi] ||
          (candidate_dist[oj] == candidate_dist[oi] &&
           candidate_id[oj] < candidate_id[oi])) {
        best = j;
      }
    }
    if (best != i) {
      const int tmp = order[i];
      order[i] = order[best];
      order[best] = tmp;
    }
  }

  double total_weight = 0;
  for (int i = 0; i < n; i++) {
    total_weight += candidate_weight[order[i]];
  }

  if (total_weight <= 0) {
    P[place].SIDMProb = -1;
    P[place].Partner = SIDM_NO_PARTNER;
    P[place].SIDMPartnerDist = 0;
    return;
  }

  P[place].SIDMProb = total_weight / (2.0 * ALPHA_FACTOR);
  const double rNB = get_random_number(P[place].ID, priv->rnd);
  if (rNB > P[place].SIDMProb) {
#ifdef SIDMCHECK
#pragma omp atomic update
    priv->Step1ProbRejected += 1;
#endif
    P[place].SIDMProb = -1;
    P[place].Partner = SIDM_NO_PARTNER;
    P[place].SIDMPartnerDist = 0;
    return;
  }

  /* Reuse the step-1 acceptance draw for the nearest-first cumulative
   * threshold, matching the original single-draw design. */
  const double threshold = rNB;
  double cumulative = 0;
  MyIDType chosen_partner = SIDM_NO_PARTNER;
  double chosen_partner_dist = 0;
  for (int i = 0; i < n; i++) {
    /* Note: here each particle has weighted probablity not normalized. This is
     * by design not a mistake. */
    cumulative += candidate_weight[order[i]];
    if (cumulative >= threshold) {
      chosen_partner = candidate_id[order[i]];
      chosen_partner_dist = candidate_dist[order[i]];
      break;
    }
  }

  if (chosen_partner == SIDM_NO_PARTNER) {
    chosen_partner = candidate_id[order[n - 1]];
    chosen_partner_dist = candidate_dist[order[n - 1]];
  }
  P[place].Partner = chosen_partner;
  P[place].SIDMPartnerDist = chosen_partner_dist;
#else
  (void)place;
  (void)tw;
#endif
}

/*! This function is the 'core' of the force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer. The communication is managed transparently.
 */
static void sidm_ngbiter_ngb(TreeWalkQuerySIDM_NGB *I,
                             TreeWalkResultSIDM_NGB *O,
                             TreeWalkNgbIterSIDM *iter, LocalTreeWalk *lv) {
  /* This clause initialises all member variables of the TreeWalkNgbIterSIDM
   * struct. It is important!*/
  if (iter->base.other == -1) {
    iter->base.Hsml = I->Hsml;
    iter->base.mask = DMMASK;
    /* Note: the specification of ASYMMETRIC is allowed because all SIDM
     * particles here have the same interaction radius. If that changes then you
     * should specify NGB_TREEFIND_SYMMETRIC and be more careful when building
     * the force tree*/
    iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
    /* Start with an empty local candidate list. */
    O->nCandidates = 0;
    return;
  }

  /* Second particle in the pair interaction*/
  int other = iter->base.other;
  /* Distance vector is not needed in the step-1 kernel. */
  (void)iter->base.dist;
  /* Scalar euclidean distance*/
  double r = iter->base.r;
  /* reject particles larger than Hsml*/
  if (r > I->Hsml)
    return;

  /* Reject self-particle just for safe*/
  if (P[other].ID == I->ID)
    return;

  /* Active-active mode: only scatter with gravity-active DM partners. */
  if (!sidm_is_active_dm(other, SIDM_GET_PRIV(lv->tw)->times))
    return;

  /* Find out predicted velocity for the other */
  /* copied from veldisp.c */
  MyFloat VelPred[3];
  DM_VelPred(other, VelPred, &SIDM_GET_PRIV(lv->tw)->kf);

  /* Precompute the velocity-dependent pair factor before the kernel term.
   * h_j depends on the full partial candidate list, so the kernel is still
   * applied later in sidm_reduce_ngb(). */
  const double sigma0_const = SIDM_GET_PRIV(lv->tw)->Sigma0Const;
  const double wTurn_2 = SIDM_GET_PRIV(lv->tw)->WTurnCode2;
  const int vd_sidm_on = SIDMParams.vdSIDMOn;
  double v_rel[3];
  v_rel[0] = VelPred[0] - I->Vel[0];
  v_rel[1] = VelPred[1] - I->Vel[1];
  v_rel[2] = VelPred[2] - I->Vel[2];
  double v_rel_mag_2 =
      v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2];

  double sigma_over_m;
  if (vd_sidm_on) { /* Rutherford viscosity cross section */
    /* The small floor avoids division by zero when v_rel -> 0. */
    double vw2 = (v_rel_mag_2 + 0.001) / wTurn_2;
    const int vd_lookup_seg = sidm_vd_sigma_lookup_segment(vw2);
    if (vd_lookup_seg >= 0) {
      sigma_over_m =
          sigma0_const * sidm_vd_sigma_shape_lookup(vw2, vd_lookup_seg);
#ifdef SIDMCHECK
      sidm_debug_count_vd_lookup_segment(SIDM_GET_PRIV(lv->tw), vd_lookup_seg);
#endif
    } else {
      sigma_over_m = sigma0_const * sidm_vd_sigma_shape_exact(vw2);
#ifdef SIDMCHECK
      sidm_debug_count_vd_exact_tail(SIDM_GET_PRIV(lv->tw));
#endif
    }
  } else {
    sigma_over_m = sigma0_const;
  }
  double pair_prefac =
      sigma_over_m * I->Mass * sqrt(v_rel_mag_2) * I->DriftFactor;

  /* Fill result structure with neighbors. */
  if (O->nCandidates < MAX_SIDM_NEIGHBORS) {
    int pos = O->nCandidates;
    O->candidate_dist[pos] = (float)r;
    O->candidate_pair_prefac[pos] = (float)pair_prefac;
    O->candidate_id[pos] = P[other].ID;
    O->nCandidates++;
    if (O->nCandidates == MAX_SIDM_NEIGHBORS) {
      iter->base.Hsml = sidm_result_max_candidate_dist(O);
    }
  } else {
    /* If the candidate list is full, we replace the candidate with the largest
     * distance if r is smaller */
    int max_idx = 0;
    double max_dist = O->candidate_dist[0];
    for (int j = 1; j < MAX_SIDM_NEIGHBORS; j++) {
      if (O->candidate_dist[j] > max_dist) {
        max_dist = O->candidate_dist[j];
        max_idx = j;
      }
    }
    if (r < max_dist) {
      O->candidate_dist[max_idx] = (float)r;
      O->candidate_pair_prefac[max_idx] = (float)pair_prefac;
      O->candidate_id[max_idx] = P[other].ID;
      iter->base.Hsml = sidm_result_max_candidate_dist(O);
    }
  }
}

/*******************************************************************/

/* Step 2 func*/

/* Copy fields from the particle table to the communication struct*/
static void sidm_copy_scatter(int place, TreeWalkQuerySIDM_SCATTER *input,
                              TreeWalk *tw) {
  /* NOTE ON TREEWALK COMMUNICATION:
   * A single target particle may be exported to multiple MPI ranks in one
   * treewalk pass (different top-tree branches). Also, fill() can be called
   * multiple times for the same target across export-buffer iterations.
   * Keep this function as a pure snapshot of query inputs. */
  input->Vel[0] = P[place].Vel[0];
  input->Vel[1] = P[place].Vel[1];
  input->Vel[2] = P[place].Vel[2];
  input->Hsml = 0;
  input->Mass = P[place].Mass;
  input->ID = P[place].ID;
#ifdef SIDM
  input->SIDMProb = P[place].SIDMProb;
  input->Partner = P[place].Partner;
  input->Scattered = P[place].Scattered;
  if (P[place].SIDMProb >= 0 && P[place].Partner != SIDM_NO_PARTNER) {
    const double partner_dist = P[place].SIDMPartnerDist;
    if (partner_dist > 0) {
      const double eps = 1e-6 * DMAX((double)P[place].Hsml, partner_dist);
      input->Hsml = DMIN((double)P[place].Hsml, partner_dist + eps);
    } else {
      input->Hsml = P[place].Hsml;
    }
  }
#endif
}

/* Apply the resolved step-2 result back to the local particle. */
static void sidm_reduce_scatter(int place, TreeWalkResultSIDM_SCATTER *result,
                                enum TreeWalkReduceMode mode, TreeWalk *tw) {
  /* NOTE:
   * reduce() may be called repeatedly for the same target (primary + multiple
   * ghost reductions from different export tasks). Writes here therefore affect
   * globally visible particle state during the current treewalk iteration. */
  if (result->Update == 0) {
    return;
  }

#ifdef SIDM
  MyIDType Partner = result->Partner;
  const int accepted_scatter =
      (result->Partner != SIDM_NO_PARTNER && result->SIDMProb >= 0);

  /* Only accepted scatters should consume the particle for this step. */
  if (accepted_scatter) {
#pragma omp atomic write
    P[place].Scattered = 1;
    int k;
    for (k = 0; k < 3; k++) {
#pragma omp atomic update
      P[place].SIDMAccel[k] += result->SIDMAccel[k];
    }
  }
#ifdef SIDMCHECK
  if (mode == TREEWALK_PRIMARY) {
    if (accepted_scatter) {
#pragma omp atomic update
      SIDM_GET_PRIV(tw)->Step2Accepted += 1;
    } else {
#pragma omp atomic update
      SIDM_GET_PRIV(tw)->Step2Suppressed += 1;
      if (result->SuppressReason == 1) {
#pragma omp atomic update
        SIDM_GET_PRIV(tw)->Step2SuppressedClaimedPartner += 1;
      } else if (result->SuppressReason == 2) {
#pragma omp atomic update
        SIDM_GET_PRIV(tw)->Step2SuppressedThreshold += 1;
      }
    }
  }
#endif
  /* Keep partner/probability assignment from the primary result only. */
  if (mode == TREEWALK_PRIMARY) {
    P[place].Partner = Partner;
    P[place].SIDMProb = result->SIDMProb;
  }
#endif
}

/* Step 2 functions */
/* Find the step-1 partner and then check runtime conflict flags. */
static void sidm_ngbiter_scatter(TreeWalkQuerySIDM_SCATTER *I,
                                 TreeWalkResultSIDM_SCATTER *O,
                                 TreeWalkNgbIterSIDM *iter, LocalTreeWalk *lv) {
  /* This clause initialises all member variables of the TreeWalkNgbIterSIDM
   * struct. It is important!*/
  if (iter->base.other == -1) {
    /* Initialize the step-2 result state. */
    O->Update = 0;
    O->Partner = SIDM_NO_PARTNER;
    O->SuppressReason = 0;
#ifdef SIDM
    if (I->Scattered == 1 || I->SIDMProb < 0 || I->Partner == SIDM_NO_PARTNER ||
        I->Hsml <= 0) {
      iter->base.stop = 1;
      return;
    }
#endif
    iter->base.Hsml = I->Hsml;
    iter->base.mask = DMMASK;
    /* Note: the specification of ASYMMETRIC is allowed because all SIDM
     * particles here have the same interaction radius. If that changes then you
     * should specify NGB_TREEFIND_SYMMETRIC and be more careful when building
     * the force tree*/
    iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
    return;
  }

/* Return if the current particle is already claimed or has no valid
 * scattering probability. */
#ifdef SIDM
  int current_scattered;
  double current_prob;

  /* These are query-time values copied by sidm_copy_scatter(). They are not
   * automatically refreshed while this query is in flight, even if other
   * threads/ranks update P[]. */
  current_scattered = I->Scattered;
  current_prob = I->SIDMProb;
  if (current_scattered == 1)
    return;
  if (current_prob < 0)
    return;
#endif

  /* Second particle in the pair interaction*/
  int other = iter->base.other;
  /* 3D vector with distances between particles (unused currently) */
  (void)iter->base.dist;
  /* Scalar euclidean distance (unused currently) */
  (void)iter->base.r;

  /* Reject self-particle just for safe*/
  if (P[other].ID == I->ID)
    return;

  /* Active-active mode: reject inactive partners before any claim/update. */
  if (!sidm_is_active_dm(other, SIDM_GET_PRIV(lv->tw)->times))
    return;
/* Look for the single partner chosen in step 1. */
#ifdef SIDM
  if (P[other].ID == I->Partner) {
#ifdef SIDMFULLCHECK
    message(3, "SIDM: Particle %ld found partner %ld.\n", I->ID, P[other].ID);
#endif
    /* Snapshot the partner currently advertised by the other particle. */
    MyIDType partner_partner;
#pragma omp atomic read
    partner_partner = P[other].Partner;

    /* Handle cases where A picked B while B picked C (which may or may not be
     * A). In dense regions these partner conflicts are common. */

    /* CRITICAL: Check for mutual partners FIRST, regardless of multiscattering
       mode. We fundamentally want to avoid "Double Counting" where A and B
       *both* calculate the kick. This happens if A selects B, and B selects A.
       (Mutual Partners). To fix this, we enforce that only the particle with
       the LOWER ID performs the calculation for the pair. (Arbitrary, could
       be Higher ID, but must be consistent).
    */
    if (partner_partner == I->ID) {
      /* Mutual Partner Case */
      if (I->ID > P[other].ID) {
#ifdef SIDMCHECK
#pragma omp atomic update
        SIDM_GET_PRIV(lv->tw)->Step2MutualHighIdSkip += 1;
#endif
        /* We are the higher ID. The other particle (Lower ID) will handle
           this interaction. We just return. We do NOT need to set Update=1
           because the other particle will update P[OurID].Accel via atomic
           add. We rely on the other particle to set P[OurID].Scattered = 1.
         */
        iter->base.stop = 1;
        return;
      }
      /* Else: We are the lower ID. We proceed to claim and calculate. */
    }

    /* Now handle the MultiscatteringThreshold-specific logic */
    if (SIDMParams.MultiscatteringThreshold < 0) {
      /* Single-scattering mode: claim partner with a protected check+set. */
      /* We try to "Claim" the partner to prevent race conditions with third
         parties (C->A or C->B). We perform a lock/critical check-and-set on
         the Scattered flag.
      */
      int already_scattered;
      /* Claim partner atomically (check+set must be one critical section). */
      if (SIDM_GET_PRIV(lv->tw)->PartLocks) {
        lock_spinlock(other, SIDM_GET_PRIV(lv->tw)->PartLocks);
        already_scattered = P[other].Scattered;
        if (already_scattered == 0) {
          P[other].Scattered = 1;
        }
        unlock_spinlock(other, SIDM_GET_PRIV(lv->tw)->PartLocks);
      } else {
#pragma omp critical(sidm_claim_partner)
        {
          already_scattered = P[other].Scattered;
          if (already_scattered == 0) {
            P[other].Scattered = 1;
          }
        }
      }

      /* Partner was already claimed by a third party or by a previous pass. */
      if (already_scattered == 1) {
        /* We lost the race to claim this particle. Suppress current scattering.
         */
        /* Note: If we are in the Mutual case (Lower ID), and we find it
           scattered, it means a THIRD party C claimed B. In that case, A failed
           to scatter with B. A should potentially try again? But we only have
           one partner candidate passed from Step 1. So A simply fails to
           scatter this step.
        */
        O->Update = 1;
        O->Partner = SIDM_NO_PARTNER;
        O->SIDMProb = -1;
        O->SIDMAccel[0] = 0;
        O->SIDMAccel[1] = 0;
        O->SIDMAccel[2] = 0;
        O->SuppressReason = 1;
#ifdef SIDMFULLCHECK
        message(3,
                "SIDM: Particle %ld suppresses self scatter with %ld, SIDMProb "
                "= %g\n",
                I->ID, P[other].ID, I->SIDMProb);
#endif
        iter->base.stop = 1;
        return;
      } else {
        /* Partner was free, and we just claimed it. We can proceed. */
        /* Note: If partner_partner != I->ID (Non-mutual), we proceed as normal.
           If partner_partner == I->ID (Mutual), we checked ID order above, so
           we are the allowed one.
        */
      }
    } else {
      /* Multiscattering mode:
       * only apply threshold suppression when this partner is in conflict.
       * This avoids globally thinning all accepted step-1 pairs. */
      int partner_scattered;
#pragma omp atomic read
      partner_scattered = P[other].Scattered;
      const int partner_conflict =
          (partner_partner != SIDM_NO_PARTNER && partner_partner != I->ID) ||
          (partner_scattered == 1);

      if (partner_conflict) {
        /* Draw a random number to determine whether to suppress this
         * conflicting scatter event. */
        double rNB = get_random_number(I->ID + 1, SIDM_GET_PRIV(lv->tw)->rnd);
        /* Keep conflicting scatters with probability =
         * MultiscatteringThreshold. Example: threshold=0.1 keeps ~10% of
         * conflicts (a small increase over strict single-scatter mode). */
        if (rNB > SIDMParams.MultiscatteringThreshold) {
          O->Update = 1;
          O->Partner = SIDM_NO_PARTNER;
          O->SIDMProb = -1;
          O->SIDMAccel[0] = 0;
          O->SIDMAccel[1] = 0;
          O->SIDMAccel[2] = 0;
          O->SuppressReason = 2;
#ifdef SIDMFULLCHECK
          message(3,
                  "SIDM: Particle %ld self suppresses scatter with %ld, rNB = "
                  "%g, MultiscatteringThreshold = %g\n",
                  I->ID, P[other].ID, rNB, SIDMParams.MultiscatteringThreshold);
#endif
          iter->base.stop = 1;
          return;
        }
      }
    }

    /* Publish the reciprocal partner and scatter claim immediately so later
     * queries observe the reservation. */
#pragma omp atomic write
    P[other].Partner = I->ID;
    /* Scattered flag already set by the protected claim above if Threshold < 0
     */
    if (SIDMParams.MultiscatteringThreshold >= 0) {
#pragma omp atomic write
      P[other].Scattered = 1;
    }

    O->Update = 1;
    O->Partner = P[other].ID;
    O->SIDMProb = I->SIDMProb;
    O->SuppressReason = 0;

  } else
    return;
#endif

  /* Find out predicted velocity for the other */
  /* copied from veldisp.c */
  MyFloat VelPred[3];
  DM_VelPred(other, VelPred, &SIDM_GET_PRIV(lv->tw)->kf);
  /* Find relative velocities. */
  double v_rel[3];
  v_rel[0] = VelPred[0] - I->Vel[0];
  v_rel[1] = VelPred[1] - I->Vel[1];
  v_rel[2] = VelPred[2] - I->Vel[2];
  /* Calculate the magnitude of the relative velocity. */
  double v_rel_mag =
      sqrt(v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]);
  /* Calculate total mass and mass ratios. */
  double m_tot = I->Mass + P[other].Mass;
  double m_ratio_self = I->Mass / m_tot;
  double m_ratio_other = P[other].Mass / m_tot;
  /* Calculate center-of-mass velocity. */
  double v_cm[3];
  int c;
  for (c = 0; c < 3; c++) {
    v_cm[c] = ((I->Vel[c] * I->Mass) + (VelPred[c] * P[other].Mass)) / m_tot;
  }
  /* Draw Unit Vector */
  /* Copied from blackhole.c */
  double dir[3];
  const uint64_t dir_seed_theta =
      sidm_pair_seed(I->ID, P[other].ID, 0xA24BAED4963EE407ull);
  const uint64_t dir_seed_phi =
      sidm_pair_seed(I->ID, P[other].ID, 0x9FB21C651E98DF25ull);
  double theta = acos(
      2 * get_random_number(dir_seed_theta, SIDM_GET_PRIV(lv->tw)->rnd) - 1);
  double phi =
      2 * M_PI * get_random_number(dir_seed_phi, SIDM_GET_PRIV(lv->tw)->rnd);
  dir[0] = sin(theta) * cos(phi);
  dir[1] = sin(theta) * sin(phi);
  dir[2] = cos(theta);
/* Calculate scattering-induced velocity changes. */
#ifdef SIDM
  int d;
  for (d = 0; d < 3; d++) {
    /* Self velocity change */
    double dv_self = v_rel_mag * m_ratio_other;
    dv_self *= dir[d];
    O->SIDMAccel[d] = v_cm[d] + dv_self - I->Vel[d];
    /* Partner velocity change */
    double dv_other = v_rel_mag * m_ratio_self;
    dv_other *= -dir[d];
    dv_other += v_cm[d] - VelPred[d];
    /* Atomic update partner's acceleration */

#ifdef SIDMFULLCHECK
    double current_accel = P[other].SIDMAccel[d];
    /* Log the partner's accumulated kick before this update. */
    message(3, "SIDM: Particle %ld current Accel[%d] = (%g)\n", P[other].ID, d,
            current_accel);
#endif

#pragma omp atomic update
    P[other].SIDMAccel[d] += dv_other;
  }
  iter->base.stop = 1;

#ifdef SIDMFULLCHECK
  message(3,
          "--------- SIDM: Particle %ld scatters with %ld, v_rel = %g, v_cm = "
          "(%g, %g, %g)---------\n",
          I->ID, P[other].ID, v_rel_mag, v_cm[0], v_cm[1], v_cm[2]);
  /* Log the resolved scatter state for this pair. */
  message(3, "SIDM: Particle %ld mass ratio = %g, %ld mass ratio = %g\n", I->ID,
          m_ratio_self, P[other].ID, m_ratio_other);
  message(3, "SIDM: Particle %ld velocity before scatter = (%g, %g, %g)\n",
          I->ID, I->Vel[0], I->Vel[1], I->Vel[2]);
  message(3, "SIDM: Particle %ld velocity before scatter = (%g, %g, %g)\n",
          P[other].ID, VelPred[0], VelPred[1], VelPred[2]);
  message(3, "SIDM: Particle %ld scatter direction = (%g, %g, %g)\n", I->ID,
          dir[0], dir[1], dir[2]);
  message(3, "SIDM: Particle %ld new Accel = (%g, %g, %g)\n", I->ID,
          O->SIDMAccel[0], O->SIDMAccel[1], O->SIDMAccel[2]);
  message(3, "SIDM: Particle %ld partner %ld new Accel = (%g, %g, %g)\n",
          P[other].ID, I->ID, P[other].SIDMAccel[0], P[other].SIDMAccel[1],
          P[other].SIDMAccel[2]);
  message(
      3,
      "--------- SIDM: End of scatterting for Particle %ld with %ld --------\n",
      I->ID, P[other].ID);
#endif
#endif
}
