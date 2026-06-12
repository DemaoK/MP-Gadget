#include <float.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "pmzoom.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/system.h"
#include "walltime.h"

static PetaPMRegion * pmzoom_prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions);
static void pmzoom_particle_position(int i, double pos[3], void * userdata);
static void pmzoom_potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void pmzoom_force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void pmzoom_force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void pmzoom_force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void pmzoom_readout_potential(PetaPM * pm, int i, double * mesh, double weight);
static void pmzoom_readout_force_x(PetaPM * pm, int i, double * mesh, double weight);
static void pmzoom_readout_force_y(PetaPM * pm, int i, double * mesh, double weight);
static void pmzoom_readout_force_z(PetaPM * pm, int i, double * mesh, double weight);

static PMZoomRegion * CurrentPMZoom;

static PetaPMFunctions pmzoom_functions [] =
{
    {"PMZoomPotential", NULL, pmzoom_readout_potential},
    {"PMZoomForceX", pmzoom_force_x_transfer, pmzoom_readout_force_x},
    {"PMZoomForceY", pmzoom_force_y_transfer, pmzoom_readout_force_y},
    {"PMZoomForceZ", pmzoom_force_z_transfer, pmzoom_readout_force_z},
    {NULL, NULL, NULL},
};

static double
pmzoom_nearest(double dx, double boxsize)
{
    if(dx > 0.5 * boxsize)
        return dx - boxsize;
    if(dx < -0.5 * boxsize)
        return dx + boxsize;
    return dx;
}

static int
pmzoom_particle_is_highres(int64_t i, int highres_types)
{
    if(P[i].IsGarbage || P[i].Swallowed)
        return 0;
    return (highres_types & (1 << P[i].Type)) != 0;
}

static double
pmzoom_floor_cell(double x, double cellsize)
{
    return floor(x / cellsize) * cellsize;
}

static double
pmzoom_upper_cell(double x, double cellsize)
{
    return (floor(x / cellsize) + 1) * cellsize;
}

void
pmzoom_init(PMZoomRegion * zoom, const struct header_data * header,
            int enabled, int highres_types, int nmesh,
            double base_asmth, int base_nmesh, double tree_rcut)
{
    memset(zoom, 0, sizeof(*zoom));
    zoom->Enabled = enabled;
    if(!zoom->Enabled)
        return;

    zoom->HighResTypes = highres_types;
    if(zoom->HighResTypes < 0) {
        if(!header->ZoomTypeMasksPresent)
            endrun(0, "PMZoomCorrectionOn requires PMZoomHighResTypes or IC ZoomHighResTypes metadata.\n");
        zoom->HighResTypes = header->ZoomHighResTypes;
    }

    const int valid_types = (1 << 6) - 1;
    if(zoom->HighResTypes <= 0 || (zoom->HighResTypes & ~valid_types))
        endrun(0, "Invalid PMZoomHighResTypes=%d. Expected a non-zero mask over particle types 0..5.\n", zoom->HighResTypes);
    if(header->ZoomTypeMasksPresent && (zoom->HighResTypes & header->ZoomBoundaryTypes))
        endrun(0, "PMZoomHighResTypes=%d overlaps ZoomBoundaryTypes=%d. Boundary particles cannot define the high-res PM region.\n",
               zoom->HighResTypes, header->ZoomBoundaryTypes);

    zoom->Nmesh = nmesh > 0 ? nmesh : base_nmesh;
    if(zoom->Nmesh <= 10)
        endrun(0, "PMZoomNmesh=%d is too small. Gadget-4-style zero padding needs PMZoomNmesh > 10.\n", zoom->Nmesh);

    if(base_nmesh <= 0)
        endrun(0, "Invalid base Nmesh=%d for zoom PM correction.\n", base_nmesh);

    if(base_asmth <= 0 || tree_rcut <= 0)
        endrun(0, "Invalid PM split parameters for zoom PM correction: Asmth=%g TreeRcut=%g.\n", base_asmth, tree_rcut);

    zoom->BoxSize = header->BoxSize;
    zoom->SplitAsmth = base_asmth;
    zoom->TreeRcut = tree_rcut;
    const double base_cell = header->BoxSize / base_nmesh;
    zoom->BaseRcut = tree_rcut * base_asmth * base_cell;

    message(0, "PMZoomCorrection: enabled high-res-types=%d PMZoomNmesh=%d base-cell=%g base-Rcut=%g\n",
            zoom->HighResTypes, zoom->Nmesh, base_cell, zoom->BaseRcut);
}

void
pmzoom_update_region(PMZoomRegion * zoom)
{
    if(!zoom || !zoom->Enabled)
        return;

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    int64_t local_count = 0;
    double local_reference[3] = {0};
    int found_reference = 0;

    int64_t i;
    int k;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!pmzoom_particle_is_highres(i, zoom->HighResTypes))
            continue;
        local_count++;
        if(!found_reference) {
            for(k = 0; k < 3; k++)
                local_reference[k] = P[i].Pos[k];
            found_reference = 1;
        }
    }

    MPI_Allreduce(&local_count, &zoom->NumHighRes, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(zoom->NumHighRes <= 0)
        endrun(0, "PMZoomCorrection: no particles match high-res type mask %d.\n", zoom->HighResTypes);

    int owner[2] = {found_reference ? ThisTask : NTask, ThisTask};
    MPI_Allreduce(MPI_IN_PLACE, owner, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
    if(owner[0] >= NTask)
        endrun(0, "PMZoomCorrection: failed to select a high-res reference particle.\n");

    for(k = 0; k < 3; k++)
        zoom->Reference[k] = local_reference[k];
    MPI_Bcast(zoom->Reference, 3, MPI_DOUBLE, owner[1], MPI_COMM_WORLD);

    double local_min[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
    double local_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

    for(i = 0; i < PartManager->NumPart; i++) {
        if(!pmzoom_particle_is_highres(i, zoom->HighResTypes))
            continue;
        for(k = 0; k < 3; k++) {
            const double unwrapped = zoom->Reference[k] + pmzoom_nearest(P[i].Pos[k] - zoom->Reference[k], zoom->BoxSize);
            if(unwrapped < local_min[k])
                local_min[k] = unwrapped;
            if(unwrapped > local_max[k])
                local_max[k] = unwrapped;
        }
    }

    MPI_Allreduce(local_min, zoom->Min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max, zoom->Max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const double placement_cell = zoom->BoxSize / zoom->Nmesh;
    zoom->EnclosingSize = 0;
    for(k = 0; k < 3; k++) {
        const double lower = pmzoom_floor_cell(zoom->Min[k], placement_cell);
        const double upper = pmzoom_upper_cell(zoom->Max[k], placement_cell);
        zoom->Min[k] = lower;
        zoom->Max[k] = upper;
        zoom->Span[k] = upper - lower;
        if(zoom->Span[k] >= 0.5 * zoom->BoxSize)
            endrun(0, "PMZoomCorrection: high-res span %g along axis %d is too large for unambiguous periodic placement in box %g.\n",
                   zoom->Span[k], k, zoom->BoxSize);
        if(zoom->Span[k] > zoom->EnclosingSize)
            zoom->EnclosingSize = zoom->Span[k];
    }

    if(zoom->EnclosingSize <= 0)
        endrun(0, "PMZoomCorrection: degenerate high-res region size %g.\n", zoom->EnclosingSize);

    zoom->TotalMeshSize = 2.0 * zoom->EnclosingSize * (zoom->Nmesh / ((double) (zoom->Nmesh - 10)));
    zoom->CellSize = zoom->TotalMeshSize / zoom->Nmesh;
    zoom->Asmth = zoom->SplitAsmth * zoom->CellSize;
    zoom->Rcut = zoom->TreeRcut * zoom->Asmth;
    if(zoom->Rcut > zoom->BaseRcut)
        endrun(0, "PMZoomCorrection: high-res Rcut=%g exceeds base Rcut=%g. Increase PMZoomNmesh or shrink the high-res region.\n",
               zoom->Rcut, zoom->BaseRcut);

    for(k = 0; k < 3; k++)
        zoom->Corner[k] = zoom->Min[k] - 2.0 * zoom->CellSize;

    message(0, "PMZoomCorrection: high-res-count=%lld reference=(%g %g %g)\n",
            (long long) zoom->NumHighRes, zoom->Reference[0], zoom->Reference[1], zoom->Reference[2]);
    message(0, "PMZoomCorrection: high-res-region min=(%g %g %g) max=(%g %g %g) span=(%g %g %g)\n",
            zoom->Min[0], zoom->Min[1], zoom->Min[2],
            zoom->Max[0], zoom->Max[1], zoom->Max[2],
            zoom->Span[0], zoom->Span[1], zoom->Span[2]);
    message(0, "PMZoomCorrection: isolated-mesh total-size=%g cell-size=%g Asmth=%g Rcut=%g corner=(%g %g %g)\n",
            zoom->TotalMeshSize, zoom->CellSize, zoom->Asmth, zoom->Rcut,
            zoom->Corner[0], zoom->Corner[1], zoom->Corner[2]);
}

static int
pmzoom_kernel_is_current(const PMZoomRegion * zoom, double ratio)
{
    if(!zoom->PMInitialized || !zoom->KernelK)
        return 0;
    if(zoom->KernelTotalMeshSize != zoom->TotalMeshSize)
        return 0;
    if(fabs(zoom->KernelAsmthRatio - ratio) > 1e-12)
        return 0;
    return 1;
}

static void
pmzoom_fill_kernel(PMZoomRegion * zoom, double ratio)
{
    PetaPM * pm = &zoom->PM;
    if(zoom->KernelK)
        myfree(zoom->KernelK);

    double * real = (double *) mymalloc2("PMZoomKernelReal", pm->priv->fftsize * sizeof(double));
    memset(real, 0, pm->priv->fftsize * sizeof(double));

    const PetaPMRegion * region = petapm_get_real_region(pm);
    const double asmth_by_grid = zoom->SplitAsmth / zoom->Nmesh;

    ptrdiff_t ix;
#pragma omp parallel for
    for(ix = 0; ix < region->size[0]; ix++) {
        ptrdiff_t iy;
        for(iy = 0; iy < region->size[1]; iy++) {
            ptrdiff_t iz;
            for(iz = 0; iz < region->size[2]; iz++) {
                const ptrdiff_t meshpos[3] = {
                    ix + region->offset[0],
                    iy + region->offset[1],
                    iz + region->offset[2],
                };
                double dx[3];
                int k;
                for(k = 0; k < 3; k++) {
                    dx[k] = meshpos[k] / ((double) zoom->Nmesh);
                    if(dx[k] >= 0.5)
                        dx[k] -= 1.0;
                }
                const double r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                double kernel;
                if(r > 0) {
                    const double u = 0.5 * r / asmth_by_grid;
                    const double fac = erfc(u * ratio) - erfc(u);
                    kernel = -fac / r;
                }
                else {
                    const double fac = 1.0 - ratio;
                    kernel = -fac / (sqrt(M_PI) * asmth_by_grid);
                }
                real[ix * region->strides[0] + iy * region->strides[1] + iz * region->strides[2]] = kernel;
            }
        }
    }

    zoom->KernelK = (pfft_complex *) mymalloc2("PMZoomKernelK", pm->priv->fftsize * sizeof(double));
    pfft_execute_dft_r2c(pm->priv->plan_forw, real, zoom->KernelK);
    myfree(real);

    zoom->KernelTotalMeshSize = zoom->TotalMeshSize;
    zoom->KernelAsmthRatio = ratio;
}

static void
pmzoom_ensure_pm(PMZoomRegion * zoom, double G)
{
    const double base_asmth = zoom->BaseRcut / zoom->TreeRcut;
    const double ratio = zoom->Asmth / base_asmth;
    if(ratio <= 0 || ratio > 1)
        endrun(0, "PMZoomCorrection: invalid high/base Asmth ratio %g. high=%g base=%g\n",
               ratio, zoom->Asmth, base_asmth);

    if(!zoom->PMInitialized || zoom->PM.Nmesh != zoom->Nmesh || zoom->PM.BoxSize != zoom->TotalMeshSize) {
        if(zoom->PMInitialized) {
            if(zoom->KernelK) {
                myfree(zoom->KernelK);
                zoom->KernelK = NULL;
            }
            petapm_destroy(&zoom->PM);
        }
        petapm_init(&zoom->PM, zoom->TotalMeshSize, zoom->SplitAsmth, zoom->Nmesh, G, MPI_COMM_WORLD);
        zoom->PMInitialized = 1;
        zoom->KernelTotalMeshSize = 0;
        zoom->KernelAsmthRatio = 0;
    }

    if(!pmzoom_kernel_is_current(zoom, ratio))
        pmzoom_fill_kernel(zoom, ratio);
}

void
pmzoom_force(PMZoomRegion * zoom, double G, const PetaPMParticleStruct * pstruct)
{
    if(!zoom || !zoom->Enabled)
        return;

    pmzoom_ensure_pm(zoom, G);

    PetaPMParticleStruct zoom_pstruct = *pstruct;
    zoom_pstruct.RegionInd = NULL;
    zoom_pstruct.position = pmzoom_particle_position;
    zoom_pstruct.position_userdata = zoom;

    PetaPMGlobalFunctions global_functions = {NULL, NULL, pmzoom_potential_transfer};
    CurrentPMZoom = zoom;
    petapm_force(&zoom->PM, pmzoom_prepare, &global_functions, pmzoom_functions, &zoom_pstruct, zoom);
    CurrentPMZoom = NULL;
    walltime_measure("/PMZoom");
}

double
pmzoom_unwrap_position(const PMZoomRegion * zoom, double pos, int axis)
{
    if(!zoom || !zoom->Enabled)
        return pos;
    return zoom->Reference[axis] + pmzoom_nearest(pos - zoom->Reference[axis], zoom->BoxSize);
}

int
pmzoom_point_location(const PMZoomRegion * zoom, const double pos[3])
{
    if(!zoom || !zoom->Enabled)
        return PMZOOM_OUTSIDE;

    int k;
    for(k = 0; k < 3; k++) {
        const double unwrapped = pmzoom_unwrap_position(zoom, pos[k], k);
        if(unwrapped < zoom->Min[k] || unwrapped >= zoom->Max[k])
            return PMZOOM_OUTSIDE;
    }
    return PMZOOM_INSIDE;
}

int
pmzoom_node_location(const PMZoomRegion * zoom, const double center[3], double len)
{
    if(!zoom || !zoom->Enabled)
        return PMZOOM_OUTSIDE;

    if(len >= 0.5 * zoom->BoxSize)
        return PMZOOM_BOUNDARY;

    int any_boundary = 0;
    int k;
    for(k = 0; k < 3; k++) {
        const double c = pmzoom_unwrap_position(zoom, center[k], k);
        const double left = c - 0.5 * len;
        const double right = c + 0.5 * len;

        if(right <= zoom->Min[k] || left >= zoom->Max[k])
            return PMZOOM_OUTSIDE;
        if(left < zoom->Min[k] || right > zoom->Max[k])
            any_boundary = 1;
    }
    return any_boundary ? PMZOOM_BOUNDARY : PMZOOM_INSIDE;
}

static void
pmzoom_particle_position(int i, double pos[3], void * userdata)
{
    PMZoomRegion * zoom = (PMZoomRegion *) userdata;
    int k;
    for(k = 0; k < 3; k++)
        pos[k] = pmzoom_unwrap_position(zoom, P[i].Pos[k], k) - zoom->Corner[k];
}

static int
pmzoom_particle_cell(PMZoomRegion * zoom, int i, int cell[3])
{
    double pos[3];
    int k;
    for(k = 0; k < 3; k++) {
        const double unwrapped = pmzoom_unwrap_position(zoom, P[i].Pos[k], k);
        if(unwrapped < zoom->Min[k] || unwrapped >= zoom->Max[k])
            return 0;
        pos[k] = unwrapped - zoom->Corner[k];
        const double meshpos = pos[k] / zoom->CellSize;
        if(meshpos < 0 || meshpos >= zoom->Nmesh - 1)
            return 0;
        cell[k] = floor(meshpos);
        if(cell[k] < 0 || cell[k] >= zoom->Nmesh - 1)
            return 0;
    }
    return 1;
}

static PetaPMRegion *
pmzoom_prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions)
{
    PMZoomRegion * zoom = (PMZoomRegion *) userdata;
    pstruct->RegionInd = (int *) mymalloc2("PMZoomRegionInd", pstruct->NumPart * sizeof(int));

    int64_t i;
#pragma omp parallel for
    for(i = 0; i < pstruct->NumPart; i++)
        pstruct->RegionInd[i] = -1;

    int local_min[3] = {zoom->Nmesh, zoom->Nmesh, zoom->Nmesh};
    int local_max[3] = {-1, -1, -1};
    int64_t local_count = 0;

    for(i = 0; i < pstruct->NumPart; i++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        if(pstruct->active && !pstruct->active(i))
            continue;

        int cell[3];
        if(!pmzoom_particle_cell(zoom, i, cell))
            continue;

        pstruct->RegionInd[i] = 0;
        local_count++;
        int k;
        for(k = 0; k < 3; k++) {
            if(cell[k] < local_min[k])
                local_min[k] = cell[k];
            if(cell[k] + 1 > local_max[k])
                local_max[k] = cell[k] + 1;
        }
    }

    int64_t global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "PMZoomCorrection: applying isolated PM correction to %lld particles inside fine mesh\n",
            (long long) global_count);

    PetaPMRegion * regions = (PetaPMRegion *) mymalloc2("PMZoomRegions", sizeof(PetaPMRegion));
    memset(regions, 0, sizeof(PetaPMRegion));
    *Nregions = 1;

    int k;
    if(local_count == 0) {
        for(k = 0; k < 3; k++) {
            regions[0].offset[k] = 0;
            regions[0].size[k] = 1;
            regions[0].center[k] = zoom->Corner[k] + 0.5 * zoom->CellSize;
        }
    }
    else {
        for(k = 0; k < 3; k++) {
            regions[0].offset[k] = local_min[k];
            regions[0].size[k] = local_max[k] - local_min[k] + 1;
            regions[0].center[k] = zoom->Corner[k] + (0.5 * (local_min[k] + local_max[k])) * zoom->CellSize;
        }
    }
    regions[0].len = 0;
    regions[0].numpart = local_count;
    regions[0].no = -1;
    petapm_region_init_strides(&regions[0]);
    (void) pm;
    return regions;
}

static double
pmzoom_sinc_unnormed(double x)
{
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
    }
    return sin(x) / x;
}

static double
pmzoom_cic_deconvolution(PetaPM * pm, const int kpos[3])
{
    double f = 1.0;
    int k;
    for(k = 0; k < 3; k++) {
        double tmp = (kpos[k] * M_PI) / pm->Nmesh;
        tmp = pmzoom_sinc_unnormed(tmp);
        f *= 1.0 / (tmp * tmp);
    }
    return f * f;
}

static void
pmzoom_potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value)
{
    (void) k2;
    PMZoomRegion * zoom = CurrentPMZoom;
    if(!zoom || !zoom->KernelK)
        endrun(0, "PMZoomCorrection: missing isolated PM kernel.\n");

    const ptrdiff_t ip = petapm_fourier_index_from_kpos(pm, kpos);
    if(ip < 0)
        endrun(0, "PMZoomCorrection: failed to map Fourier cell (%d %d %d).\n", kpos[0], kpos[1], kpos[2]);

    const double re = value[0][0] * zoom->KernelK[ip][0] - value[0][1] * zoom->KernelK[ip][1];
    const double im = value[0][0] * zoom->KernelK[ip][1] + value[0][1] * zoom->KernelK[ip][0];
    const double scale = pm->G * pmzoom_cic_deconvolution(pm, kpos) /
        (pm->BoxSize * pow((double) pm->Nmesh, 3));

    value[0][0] = re * scale;
    value[0][1] = im * scale;
}

static double
pmzoom_diff_kernel(double w)
{
    return (8.0 * sin(w) - sin(2.0 * w)) / 6.0;
}

static void
pmzoom_force_transfer(PetaPM * pm, int k, pfft_complex * value)
{
    const double fac = -pmzoom_diff_kernel(k * (2.0 * M_PI / pm->Nmesh)) * (pm->Nmesh / pm->BoxSize);
    const double tmp0 = -value[0][1] * fac;
    const double tmp1 = value[0][0] * fac;
    value[0][0] = tmp0;
    value[0][1] = tmp1;
}

static void
pmzoom_force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value)
{
    (void) k2;
    pmzoom_force_transfer(pm, kpos[0], value);
}

static void
pmzoom_force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value)
{
    (void) k2;
    pmzoom_force_transfer(pm, kpos[1], value);
}

static void
pmzoom_force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value)
{
    (void) k2;
    pmzoom_force_transfer(pm, kpos[2], value);
}

static void
pmzoom_readout_potential(PetaPM * pm, int i, double * mesh, double weight)
{
    (void) pm;
    P[i].Potential += weight * mesh[0];
}

static void
pmzoom_readout_force_x(PetaPM * pm, int i, double * mesh, double weight)
{
    (void) pm;
    P[i].GravPM[0] += weight * mesh[0];
}

static void
pmzoom_readout_force_y(PetaPM * pm, int i, double * mesh, double weight)
{
    (void) pm;
    P[i].GravPM[1] += weight * mesh[0];
}

static void
pmzoom_readout_force_z(PetaPM * pm, int i, double * mesh, double weight)
{
    (void) pm;
    P[i].GravPM[2] += weight * mesh[0];
}
