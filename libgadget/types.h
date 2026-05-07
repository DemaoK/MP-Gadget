#ifndef __TYPES_H
#define __TYPES_H

#include <stdint.h>

/*Define some useful types*/

typedef int64_t inttime_t;

typedef uint64_t MyIDType;
#define IDTYPE_MAX UINT64_MAX

/* Particle type masks. Bit ptype marks a particle type. */
#define ALLMASK ((1 << 6) - 1)
#define GASMASK (1 << 0)
#define DMMASK (1 << 1)
#define NUMASK (1 << 2)
#define STARMASK (1 << 4)
#define BHMASK (1 << 5)

#ifndef LOW_PRECISION
#define LOW_PRECISION double
#endif

typedef LOW_PRECISION MyFloat;

#define HAS(val, flag) ((flag & (val)) == (flag))

#endif
