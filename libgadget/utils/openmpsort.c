#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "mymalloc.h"
/* Below is a merge-sort routine copied directly from glibc version 2.26
 * (although the code is the same since Dec. 2010, glibc 2.13).
 * The copy is so that we can control our memory allocation
 * to use mymalloc instead of the system malloc.
 * It also means that on machines where libc is not glibc we will
 * use glibc's routine, which is almost certainly better.*/

/*============================================*/
/* An alternative to qsort, with an identical interface.
   This file is part of the GNU C Library.
   Copyright (C) 1992-2017 Free Software Foundation, Inc.
   Written by Mike Haertel, September 1988.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*Define for convenience*/
typedef int(*__compar_fn_t)(const void *, const void *);

struct msort_param
{
  size_t s;
  size_t var;
  __compar_fn_t cmp;
  void *arg;
  char *t;
};
static void msort_with_tmp (const struct msort_param *p, void *b, size_t n);

static void
msort_with_tmp (const struct msort_param *p, void *b, size_t n)
{
  char *b1, *b2;
  size_t n1, n2;

  if (n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = (char *) b;
  b2 = (char *) b + (n1 * p->s);

  msort_with_tmp (p, b1, n1);
  msort_with_tmp (p, b2, n2);

  char *tmp = p->t;
  const size_t s = p->s;
  __compar_fn_t cmp = p->cmp;
  switch (p->var)
    {
    case 0:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              *(uint32_t *) tmp = *(uint32_t *) b1;
              b1 += sizeof (uint32_t);
              --n1;
            }
          else
            {
              *(uint32_t *) tmp = *(uint32_t *) b2;
              b2 += sizeof (uint32_t);
              --n2;
            }
          tmp += sizeof (uint32_t);
        }
      break;
    case 1:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              *(uint64_t *) tmp = *(uint64_t *) b1;
              b1 += sizeof (uint64_t);
              --n1;
            }
          else
            {
              *(uint64_t *) tmp = *(uint64_t *) b2;
              b2 += sizeof (uint64_t);
              --n2;
            }
          tmp += sizeof (uint64_t);
        }
      break;
    case 2:
      while (n1 > 0 && n2 > 0)
        {
          unsigned long *tmpl = (unsigned long *) tmp;
          unsigned long *bl;

          tmp += s;
          if ((*cmp) (b1, b2) <= 0)
            {
              bl = (unsigned long *) b1;
              b1 += s;
              --n1;
            }
          else
            {
              bl = (unsigned long *) b2;
              b2 += s;
              --n2;
            }
          while (tmpl < (unsigned long *) tmp)
            *tmpl++ = *bl++;
        }
      break;
    case 3:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (*(const void **) b1, *(const void **) b2) <= 0)
            {
              *(void **) tmp = *(void **) b1;
              b1 += sizeof (void *);
              --n1;
            }
          else
            {
              *(void **) tmp = *(void **) b2;
              b2 += sizeof (void *);
              --n2;
            }
          tmp += sizeof (void *);
        }
      break;
    default:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              tmp = (char *) memcpy (tmp, b1, s) + s;
              b1 += s;
              --n1;
            }
          else
            {
              tmp = (char *) memcpy (tmp, b2, s) + s;
              b2 += s;
              --n2;
            }
        }
      break;
    }

  if (n1 > 0)
    memcpy (tmp, b1, n1 * s);
  memcpy (b, p->t, (n - n2) * s);
}

/*End code copied from glibc*/
/*=====================================================*/

/* Stable merge of x[0..nx) and y[0..ny) (ties to x) into output. */
static void merge(const void * base1, size_t nmemb1, const void * base2, size_t nmemb2, void * output, size_t size,
         int(*compar)(const void *, const void *), int indirect) {
    const char * p1 = (const char *) base1;
    const char * p2 = (const char *) base2;
    char * po = (char *) output;
    const char * s1 = p1 + nmemb1 * size, *s2 = p2 + nmemb2 * size;
    while(p1 < s1 && p2 < s2) {
        int cmp;
        if(indirect) {
            cmp = compar(*(void * const *)p1, *(void * const *)p2);
        }
        else {
            cmp = compar(p1, p2);
        }
        if(cmp <= 0) {
            memcpy(po, p1, size);
            p1 += size;
        } else {
            memcpy(po, p2, size);
            p2 += size;
        }
        po += size;
    }
    if(p1 < s1) {
        memcpy(po, p1, s1 - p1);
    }
    if(p2 < s2) {
        memcpy(po, p2, s2 - p2);
    }
}

/* Merge path: the number of elements of x among the first d outputs of the stable merge of x
 * and y (ties to x), by bisection on the cross diagonal d. */
static size_t
merge_split(const char * x, size_t nx, const char * y, size_t ny, size_t d, size_t size,
            int(*compar)(const void *, const void *), int indirect)
{
    size_t lo = d > ny ? d - ny : 0, hi = d < nx ? d : nx;
    while(lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const char * a = x + mid * size;
        const char * b = y + (d - 1 - mid) * size;
        const int c = indirect ? compar(*(void * const *) a, *(void * const *) b) : compar(a, b);
        if(c <= 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* Bit helpers of permute_openmp. */
static inline int
bit_get(const unsigned char * v, size_t i)
{
    return (v[i >> 3] >> (i & 7)) & 1;
}
static inline void
bit_set(unsigned char * v, size_t i)
{
    v[i >> 3] |= (unsigned char) (1u << (i & 7));
}

/* Applies a permutation in place: element j of the result is element src[j] of the input. The
 * cycles are listed serially (order[], int-only work), cut into units of consecutive cycle
 * positions, and the record moves of the units run in parallel; the element that a unit's last
 * move reads from the next unit, and the start of a cycle that began in an earlier unit, are
 * copied before any move. The result does not depend on the number of threads. order must hold
 * nmemb ints; src is not modified. */
void
permute_openmp(void * base, size_t nmemb, size_t size, const int * src, int * order)
{
    char * b0 = (char *) base;
    if(nmemb <= 1)
        return;
    const size_t nbytes = nmemb / 8 + 1;
    unsigned char * vis = (unsigned char *) mymalloc("PermVis", nbytes);
    unsigned char * cs = (unsigned char *) mymalloc("PermCs", nbytes);
    memset(vis, 0, nbytes);
    memset(cs, 0, nbytes);
    const int Nt = omp_get_max_threads();
    size_t S = nmemb / (16 * (size_t) Nt) + 1;
    if(S < 1024)
        S = 1024;
    const size_t umax = nmemb / S + 2;
    size_t * ucs = (size_t *) mymalloc("PermUcs", sizeof(size_t) * umax);
    size_t no = 0, cstart = 0, i;
    for(i = 0; i < nmemb; i++) {
        if((size_t) src[i] == i || bit_get(vis, i))
            continue;
        size_t j = i;
        cstart = no;
        bit_set(cs, no);
        do {
            bit_set(vis, j);
            if(no % S == 0)
                ucs[no / S] = cstart;
            order[no++] = (int) j;
            j = (size_t) src[j];
        } while(j != i);
    }
    const size_t nunits = (no + S - 1) / S;
    char * saveB = (char *) mymalloc("PermSaveB", size * (nunits + 1));
    char * saveS = (char *) mymalloc("PermSaveS", size * (nunits + 1));
    char * tmps = (char *) mymalloc("PermTmp", size * Nt);
    int64_t u;
    /* the values a unit reads before they are overwritten by another unit */
    #pragma omp parallel for
    for(u = 0; u < (int64_t) nunits; u++) {
        const size_t a = u * S, b = a + S < no ? a + S : no;
        if(b < no && !bit_get(cs, b))
            memcpy(saveB + u * size, b0 + (size_t) order[b] * size, size);
        if(ucs[u] < a)
            memcpy(saveS + u * size, b0 + (size_t) order[ucs[u]] * size, size);
    }
    #pragma omp parallel
    {
        char * tmp = tmps + size * omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for(u = 0; u < (int64_t) nunits; u++) {
            const size_t a = u * S, b = a + S < no ? a + S : no;
            int started_here = 0;
            size_t k;
            for(k = a; k < b; k++) {
                char * dk = b0 + (size_t) order[k] * size;
                if(bit_get(cs, k)) {
                    memcpy(tmp, dk, size);
                    started_here = 1;
                }
                const char * sp;
                if(k + 1 < no && !bit_get(cs, k + 1))
                    sp = (k + 1 < b) ? b0 + (size_t) order[k + 1] * size : saveB + u * size;
                else
                    sp = started_here ? tmp : saveS + u * size;
                memcpy(dk, sp, size);
            }
        }
    }
    myfree(tmps);
    myfree(saveS);
    myfree(saveB);
    myfree(ucs);
    myfree(cs);
    myfree(vis);
}

/* A parallel stable merge sort (the result equals that of any stable sort, independent of the
 * number of threads): every thread sorts a contiguous chunk with glibc's msort, then the runs
 * are merged pairwise, log2(threads) levels, every thread producing an equal share of each
 * level's output through merge-path splits. Records larger than 32 bytes are sorted through
 * pointers and moved once at the end (permute_openmp). */
void qsort_openmp(void *base, size_t nmemb, size_t size,
                         int(*compar)(const void *, const void *)) {
    if(nmemb <= 1)
        return;
    /*Should I use indirect sorting?*/
    const int indirect = size > 32;
    const size_t es = indirect ? sizeof(void *) : size;
    /*NOTE: if this allocation becomes a problem,
     * switch to glibc's quicksort (serial!) */
    char * tmp;
    if(indirect)
        tmp = (char *) mymalloc("qsort",2*nmemb*sizeof(void *) + size);
    else
        tmp = (char *) mymalloc("qsort", size * nmemb);
    char * A = indirect ? tmp + nmemb * sizeof(void *) : (char *) base;
    char * B = tmp;

    int var = 4;
    if(indirect)
        var = 3;
    else {
        /*Copied from glibc*/
        if ((size & (sizeof (uint32_t) - 1)) == 0
            && ((char *) base - (char *) 0) % __alignof__ (uint32_t) == 0)
          {
            if (size == sizeof (uint32_t))
              var = 0;
            else if (size == sizeof (uint64_t)
                     && ((char *) base - (char *) 0) % __alignof__ (uint64_t) == 0)
              var = 1;
            else if ((size & (sizeof (unsigned long) - 1)) == 0
                     && ((char *) base - (char *) 0)
                        % __alignof__ (unsigned long) == 0)
              var = 2;
          }
        /*End copied from glibc*/
    }

    /* at least 1024 elements per thread */
    int Nt = omp_get_max_threads();
    if((size_t) Nt > nmemb / 1024)
        Nt = (int) (nmemb / 1024);
    if(Nt < 1)
        Nt = 1;
    size_t * bnd = ta_malloc("qsortbnd", size_t, Nt + 1);
    char * res = A;

#pragma omp parallel num_threads(Nt)
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
#pragma omp single
        {
            int r;
            for(r = 0; r <= nt; r++)
                bnd[r] = (size_t) r * nmemb / nt;
        }
        const size_t start = bnd[tid], end = bnd[tid + 1];
        if(indirect) {
            void ** tp = (void **) A;
            size_t i;
            for(i = start; i < end; i++)
                tp[i] = (char *) base + i * size;
        }
        struct msort_param p;
        p.s = es;
        p.var = var;
        p.cmp = compar;
        p.arg = NULL;
        p.t = B + start * es;
        msort_with_tmp (&p, A + start * es, end - start);
#pragma omp barrier
        char * src = A, * dst = B;
        int w;
        const size_t o0 = start, o1 = end;
        for(w = 1; w < nt; w *= 2) {
            int q;
            for(q = 0; 2 * q * w < nt; q++) {
                const int c0 = 2 * q * w;
                const int c1 = c0 + w < nt ? c0 + w : nt;
                const int c2 = c0 + 2 * w < nt ? c0 + 2 * w : nt;
                const size_t s0 = bnd[c0], s1 = bnd[c1], s2 = bnd[c2];
                const size_t lo = o0 > s0 ? o0 : s0, hi = o1 < s2 ? o1 : s2;
                if(lo >= hi)
                    continue;
                const char * x = src + s0 * es, * y = src + s1 * es;
                const size_t nx = s1 - s0, ny = s2 - s1;
                const size_t i0 = merge_split(x, nx, y, ny, lo - s0, es, compar, indirect);
                const size_t i1 = merge_split(x, nx, y, ny, hi - s0, es, compar, indirect);
                const size_t j0 = lo - s0 - i0, j1 = hi - s0 - i1;
                merge(x + i0 * es, i1 - i0, y + j0 * es, j1 - j0, dst + lo * es, es, compar, indirect);
            }
            char * t = src;
            src = dst;
            dst = t;
#pragma omp barrier
        }
        /* src holds the sorted elements; a direct sort copies them back */
        if(!indirect && src != (char *) base)
            memcpy((char *) base + o0 * es, src + o0 * es, (o1 - o0) * es);
#pragma omp single
        res = src;
    }
    ta_free(bnd);

    if(indirect) {
        void ** tp = (void **) res;
        char * other = (res == A) ? B : A;
        if(nmemb <= (size_t) INT_MAX) {
            /* the source index of every result slot and the cycle list in the other half */
            int * srcidx = (int *) other;
            int * order = srcidx + nmemb;
            size_t i;
            #pragma omp parallel for
            for(i = 0; i < nmemb; i++)
                srcidx[i] = (int) (((char *) tp[i] - (char *) base) / size);
            permute_openmp(base, nmemb, size, srcidx, order);
        }
        else {
            /*Copied from glibc */
            /* tp[0] .. tp[n - 1] is now sorted, copy around entries of
               the original array (done serially).  Knuth vol. 3 (2nd ed.) exercise 5.2-10.  */
            char *kp, *ip;
            size_t i;
            void *tmp_storage = (void *) (tmp + 2 * nmemb * sizeof(void *));
            for (i = 0, ip = (char *) base; i < nmemb; i++, ip += size)
              if ((kp = ((char **)tp)[i]) != ip)
                {
                  size_t j = i;
                  char *jp = ip;
                  memcpy (tmp_storage, ip, size);

                  do {
                    size_t k = (kp - (char *) base) / size;
                    tp[j] = jp;
                    memcpy (jp, kp, size);
                    j = k;
                    jp = kp;
                    kp = ((char **)tp)[k];
                  } while (kp != ip);

                  tp[j] = jp;
                  memcpy (jp, tmp_storage, size);
                }
        }
    }
    myfree(tmp);
}
