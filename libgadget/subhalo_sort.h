#ifndef SUBHALO_SORT_H
#define SUBHALO_SORT_H

#include <stdint.h>

/* In-place introsort NAME(T * a, int64_t n), allocation-free and thread-safe. Every comparator the
 * finder gives it is a strict total order (ties by ID and P index), so the result is unique. */
#define SUBHALO_DEFINE_SORT(NAME, T, LESS)                                              \
static void NAME##_isort(T * a, int64_t n) {                                            \
    int64_t i, j;                                                                       \
    for(i = 1; i < n; i++) {                                                            \
        T x = a[i];                                                                     \
        for(j = i; j > 0 && LESS((&x), (&a[j - 1])); j--)                               \
            a[j] = a[j - 1];                                                            \
        a[j] = x;                                                                       \
    }                                                                                   \
}                                                                                       \
static void NAME##_sift(T * a, int64_t root, int64_t n) {                               \
    for(;;) {                                                                           \
        int64_t child = 2 * root + 1;                                                   \
        if(child >= n) return;                                                          \
        if(child + 1 < n && LESS((&a[child]), (&a[child + 1]))) child++;                \
        if(!LESS((&a[root]), (&a[child]))) return;                                      \
        T t = a[root]; a[root] = a[child]; a[child] = t;                                \
        root = child;                                                                   \
    }                                                                                   \
}                                                                                       \
static void NAME##_hsort(T * a, int64_t n) {                                            \
    int64_t i;                                                                          \
    for(i = n / 2 - 1; i >= 0; i--) NAME##_sift(a, i, n);                               \
    for(i = n - 1; i > 0; i--) {                                                        \
        T t = a[0]; a[0] = a[i]; a[i] = t;                                              \
        NAME##_sift(a, 0, i);                                                           \
    }                                                                                   \
}                                                                                       \
static void NAME##_intro(T * a, int64_t n, int depth) {                                 \
    while(n > 16) {                                                                     \
        if(depth-- <= 0) { NAME##_hsort(a, n); return; }                                \
        const int64_t mid = n / 2;                                                      \
        T t;                                                                            \
        if(LESS((&a[mid]), (&a[0]))) { t = a[mid]; a[mid] = a[0]; a[0] = t; }           \
        if(LESS((&a[n - 1]), (&a[0]))) { t = a[n - 1]; a[n - 1] = a[0]; a[0] = t; }     \
        if(LESS((&a[n - 1]), (&a[mid]))) { t = a[n - 1]; a[n - 1] = a[mid]; a[mid] = t; } \
        const T pivot = a[mid];                                                         \
        int64_t i = -1, j = n;                                                          \
        for(;;) {                                                                       \
            do i++; while(LESS((&a[i]), (&pivot)));                                     \
            do j--; while(LESS((&pivot), (&a[j])));                                     \
            if(i >= j) break;                                                           \
            t = a[i]; a[i] = a[j]; a[j] = t;                                            \
        }                                                                               \
        /* [0, j] and [j + 1, n), both non-empty; recurse on the smaller */             \
        if(j + 1 < n - j - 1) { NAME##_intro(a, j + 1, depth); a += j + 1; n -= j + 1; } \
        else { NAME##_intro(a + j + 1, n - j - 1, depth); n = j + 1; }                  \
    }                                                                                   \
    NAME##_isort(a, n);                                                                 \
}                                                                                       \
static void NAME(T * a, int64_t n) {                                                    \
    int depth = 2;                                                                      \
    int64_t m = n;                                                                      \
    while(m > 1) { m >>= 1; depth += 2; }                                               \
    NAME##_intro(a, n, depth);                                                          \
}

#endif
