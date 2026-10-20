#ifndef OPENMPSORT_H
#define OPENMPSORT_H

#include <stddef.h>

void qsort_openmp(void *base, size_t nmemb, size_t size,
                         int(*compar)(const void *, const void *));

/* In-place permutation: element j of the result is element src[j] of the input; order is
 * caller scratch of nmemb ints. Threaded record moves; the result is thread-count independent. */
void permute_openmp(void * base, size_t nmemb, size_t size, const int * src, int * order);

#endif
