#ifndef PMZOOM_H
#define PMZOOM_H

#include <stdint.h>

#include "petaio.h"
#include "petapm.h"

enum PMZoomLocation {
    PMZOOM_OUTSIDE = 0,
    PMZOOM_INSIDE = 1,
    PMZOOM_BOUNDARY = 2,
};

typedef struct PMZoomRegion {
    int Enabled;
    int HighResTypes;
    int Nmesh;
    int64_t NumHighRes;
    double BoxSize;
    double Reference[3];
    double Min[3];
    double Max[3];
    double Span[3];
    double EnclosingSize;
    double TotalMeshSize;
    double CellSize;
    double SplitAsmth;
    double TreeRcut;
    double BaseRcut;
    double Asmth;
    double Rcut;
    double Corner[3];
    int PMInitialized;
    double KernelTotalMeshSize;
    double KernelAsmthRatio;
    PetaPM PM;
    pfft_complex * KernelK;
} PMZoomRegion;

void pmzoom_init(PMZoomRegion * zoom, const struct header_data * header,
                 int enabled, int highres_types, int nmesh,
                 double base_asmth, int base_nmesh, double tree_rcut);

void pmzoom_update_region(PMZoomRegion * zoom);
void pmzoom_force(PMZoomRegion * zoom, double G, const PetaPMParticleStruct * pstruct);
double pmzoom_unwrap_position(const PMZoomRegion * zoom, double pos, int axis);
int pmzoom_point_location(const PMZoomRegion * zoom, const double pos[3]);
int pmzoom_node_location(const PMZoomRegion * zoom, const double center[3], double len);

#endif
