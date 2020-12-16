#include "vec.h"

#ifndef INLINE
__attribute__((always_inline)) inline
void vadd (double *dst, double *src1, double *src2, int sz)
{
    for (int i = 0; i < sz; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

__attribute__((always_inline)) inline
void vmul (double *dst, double *src1, double *src2, int sz)
{
    for (int i = 0; i < sz; i++) {
        dst[i] = src1[i] * src2[i];
    }
}
#endif


