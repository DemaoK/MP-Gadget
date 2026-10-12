#ifndef SIDM_RANDOM_H
#define SIDM_RANDOM_H
#include <stdint.h>

static inline uint64_t sidm_random_mix(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

/* Preserve the established stream exactly; combine the base seed and epoch
 * here instead of distributing the event-seed definition across callers. */
static inline uint64_t sidm_event_seed(uint64_t seed, uint64_t epoch)
{
    return sidm_random_mix(seed ^ epoch ^ sidm_random_mix(epoch));
}
#endif
