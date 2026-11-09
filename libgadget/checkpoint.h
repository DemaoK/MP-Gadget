#ifndef CHECKPOINT_H
#define CHECKPOINT_H
#include "cosmology.h"

/* Write the particle snapshot snapnum and list it in Snapshots.txt. Checkpoint = 1 marks a write outside the
 * output sequence (header attribute Checkpoint); a restart from it reuses snapnum for the next output. */
void write_checkpoint(int snapnum, int WriteGroupID, int MetalReturnOn, double Time, const Cosmology * CP, const char * OutputDir, const int OutputDebugFields, const int Checkpoint);
void dump_snapshot(const char * dump, const double Time, const Cosmology * CP, const char * OutputDir);
int find_last_snapnum(const char * OutputDir);

#endif
