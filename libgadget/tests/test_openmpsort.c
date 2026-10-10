#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include <string.h>
#include <stdint.h>
#include "stub.h"
#include "../utils/openmpsort.h"

#if 0
static int checksorted(int * start, int nmemb) {
    int i;
    for(i = 1; i < nmemb; i ++) {
        if( start[i-1] > start[i]) {
            return i;
        }
    }
    return 0;
}
#endif
static int compare(const void *a, const void *b) {
        return ( *(int*)a - *(int*)b );
}

struct __data
{
    int dd[200];
};
/*This exists because large structs are sorted indirectly*/
static void test_openmpsort_struct(void ** state) {
    int i;
    // set up array to be sorted
    //a bad case is 87763 threads = 12
    int size = 187763;
    struct __data *a = (struct __data *) malloc(size * sizeof(struct __data));

    srand48(8675309);
    for(i = 0; i < size; i++)
        a[i].dd[0] = (int) (size * drand48());

    double start, end;

    start = omp_get_wtime();
    qsort_openmp(a, size, sizeof(struct __data), compare);
    end = omp_get_wtime();

    message(1,"parallel sort time = %g s %d threads\n",end-start, omp_get_max_threads());
    for(i=1; i<size; i++) {
        if( a[i-1].dd[0] >a[i].dd[0] )
            message(1,"BAD: %d %d %d\n",i,a[i-1].dd[0],a[i].dd[0]);
        assert_true(a[i-1].dd[0] <= a[i].dd[0]);
    }

    srand48(8675309);
    for(i = 0; i < size; i++)
        a[i].dd[0] = (int) (size * drand48());

    start = omp_get_wtime();
    qsort(a, size, sizeof(struct __data), compare);
    end = omp_get_wtime();

    message(1, "serial (glibc) sort time = %g s\n",end-start);

}
static void test_openmpsort(void ** state) {
    int i;
    // set up array to be sorted
    //a bad case is 87763 threads = 12
    int size = 87763;
    int *a = (int *) malloc(size * sizeof(int));

    srand48(8675309);
    for(i = 0; i < size; i++) 
        a[i] = (int) (size * drand48());

    double start, end;

    start = omp_get_wtime();
    qsort_openmp(a, size, sizeof(int), compare);
    end = omp_get_wtime();

    message(1,"parallel sort time = %g s %d threads\n",end-start, omp_get_max_threads());
    for(i=1; i<size; i++) {
        if( a[i-1]>a[i] )
            message(1,"BAD: %d %d %d\n",i,a[i-1],a[i]);
        assert_true(a[i-1] <= a[i]);
    }

    srand48(8675309);
    for(i = 0; i < size; i++) 
        a[i] = (int) (size * drand48());

    start = omp_get_wtime();
    qsort(a, size, sizeof(int), compare);
    end = omp_get_wtime();

    message(1, "serial sort time = %g s\n",end-start);

}


/* Phase 5d: the parallel merge is stable and its result does not depend on the thread count. */
struct rec16 { int key; int orig; int64_t pad; };
struct rec48 { int key; int orig; char pad[40]; };
static int cmp_key16(const void * a, const void * b) {
    const int x = ((const struct rec16 *) a)->key, y = ((const struct rec16 *) b)->key;
    return (x > y) - (x < y);
}
static int cmp_full16(const void * a, const void * b) {
    const struct rec16 * x = (const struct rec16 *) a, * y = (const struct rec16 *) b;
    if(x->key != y->key) return (x->key > y->key) - (x->key < y->key);
    return (x->orig > y->orig) - (x->orig < y->orig);
}
static int cmp_key48(const void * a, const void * b) {
    const int x = ((const struct rec48 *) a)->key, y = ((const struct rec48 *) b)->key;
    return (x > y) - (x < y);
}
static void test_openmpsort_stable(void ** state) {
    const int sizes[4] = {5, 3000, 87763, 300001};
    const int threads[5] = {1, 2, 3, 4, 7};
    int s, t, i;
    const int maxt = omp_get_max_threads();
    for(s = 0; s < 4; s++) {
        const int n = sizes[s];
        struct rec16 * a = (struct rec16 *) malloc(n * sizeof(struct rec16));
        struct rec16 * ref = (struct rec16 *) malloc(n * sizeof(struct rec16));
        struct rec48 * b = (struct rec48 *) malloc(n * sizeof(struct rec48));
        srand48(1234 + s);
        for(i = 0; i < n; i++) {
            ref[i].key = (int) (50 * drand48());
            ref[i].orig = i;
            ref[i].pad = i;
        }
        qsort(ref, n, sizeof(struct rec16), cmp_full16);
        for(t = 0; t < 5; t++) {
            omp_set_num_threads(threads[t]);
            srand48(1234 + s);
            for(i = 0; i < n; i++) {
                a[i].key = (int) (50 * drand48());
                a[i].orig = i;
                a[i].pad = i;
                b[i].key = a[i].key;
                b[i].orig = i;
                memset(b[i].pad, i & 0xff, sizeof(b[i].pad));
            }
            qsort_openmp(a, n, sizeof(struct rec16), cmp_key16);
            qsort_openmp(b, n, sizeof(struct rec48), cmp_key48);
            for(i = 0; i < n; i++) {
                assert_int_equal(a[i].key, ref[i].key);
                assert_int_equal(a[i].orig, ref[i].orig);
                assert_int_equal(b[i].key, ref[i].key);
                assert_int_equal(b[i].orig, ref[i].orig);
                assert_int_equal((unsigned char) b[i].pad[39], ref[i].orig & 0xff);
            }
        }
        free(b);
        free(ref);
        free(a);
    }
    omp_set_num_threads(maxt);
}

/* permute_openmp equals the out-of-place gather for random and block-local permutations. */
static void test_permute_openmp(void ** state) {
    const int n = 200003;
    const int maxt = omp_get_max_threads();
    int * src = (int *) malloc(n * sizeof(int));
    int * order = (int *) malloc(n * sizeof(int));
    struct rec48 * a = (struct rec48 *) malloc(n * sizeof(struct rec48));
    int mode, t, i;
    for(mode = 0; mode < 3; mode++) {
        for(t = 1; t <= 4; t++) {
            omp_set_num_threads(t);
            for(i = 0; i < n; i++)
                src[i] = i;
            srand48(77 + mode);
            for(i = n - 1; i > 0; i--) {
                /* 0: random; 1: shuffles within blocks of 1000; 2: one long rotation */
                int j;
                if(mode == 0) j = (int) ((i + 1) * drand48());
                else if(mode == 1) { int lo = (i / 1000) * 1000; j = lo + (int) ((i - lo + 1) * drand48()); }
                else j = i - 1;
                int x = src[i]; src[i] = src[j]; src[j] = x;
            }
            for(i = 0; i < n; i++) {
                a[i].key = i;
                memset(a[i].pad, (i * 7) & 0xff, sizeof(a[i].pad));
            }
            permute_openmp(a, n, sizeof(struct rec48), src, order);
            for(i = 0; i < n; i++) {
                assert_int_equal(a[i].key, src[i]);
                assert_int_equal((unsigned char) a[i].pad[17], (src[i] * 7) & 0xff);
            }
        }
    }
    omp_set_num_threads(maxt);
    free(a);
    free(order);
    free(src);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_openmpsort),
        cmocka_unit_test(test_openmpsort_struct),
        cmocka_unit_test(test_openmpsort_stable),
        cmocka_unit_test(test_permute_openmp),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
