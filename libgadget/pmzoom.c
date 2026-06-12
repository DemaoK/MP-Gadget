#include <float.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "pmzoom.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/system.h"

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

void
pmzoom_require_force_implemented(const PMZoomRegion * zoom)
{
    if(!zoom || !zoom->Enabled)
        return;

    endrun(0, "PMZoomCorrectionOn selected a Gadget-4-style high-res PM region, "
              "but the isolated PM force/readout path is not implemented yet.\n");
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
