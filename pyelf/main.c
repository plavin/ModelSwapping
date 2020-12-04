#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void vadd (double *dst, double *src1, double *src2, int sz)
{
    for (int i = 0; i < sz; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

void vmul (double *dst, double *src1, double *src2, int sz)
{
    for (int i = 0; i < sz; i++) {
        dst[i] = src1[i] * src2[i];
    }
}

int main (int argv, char ** argc)
{
    double a = 0.9;
    double b = acos(a);
    printf("The arc cosine of %.2lf is %.2lf radians.\n", a, b);

    int sz = 25;
    double *c = (double*)malloc(sizeof(double) * sz);
    double *d = (double*)malloc(sizeof(double) * sz);
    double *e = (double*)malloc(sizeof(double) * sz);
    double *f = (double*)malloc(sizeof(double) * sz);

   for (int i = 0; i < sz; i++) {
       c[i] = i * i;
       d[i] = i + i;
   }

   vadd(e, c, d, sz);
   vmul(f, c, d, sz);

   printf("Result: %.2f\n", e[2]*f[3]);
}
