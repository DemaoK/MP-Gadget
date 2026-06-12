#ifndef PMZOOM_H
#define PMZOOM_H

#include <stdint.h>

#include "petaio.h"

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
} PMZoomRegion;

void pmzoom_init(PMZoomRegion * zoom, const struct header_data * header,
                 int enabled, int highres_types, int nmesh,
                 double base_asmth, int base_nmesh, double tree_rcut);

void pmzoom_update_region(PMZoomRegion * zoom);
void pmzoom_require_force_implemented(const PMZoomRegion * zoom);

#endif
