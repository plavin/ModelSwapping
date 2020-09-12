#include <stdlib.h>
#include "phase_boundary.h"
void __attribute__ ((noinline)) phase_boundary8  (long *arr, long arr_len, long val, long iter) {
    int count = 10;
    long *d = (long*)malloc(sizeof(long)*count);

    for(int i = 0; i < count; i++) {
        d[i] = 11;
    }

    __asm__ __volatile__(
            "movq $0, %%r8;"
            "iter:"
            "movq $0, %%rbx;"
            "loop:"
            "movq %1, %%rax;"
            "movq %%rax, (%0, %%rbx, 8);"
            "incq %%rbx;"
            "cmpq %%rcx, %%rbx;"
            "jl loop;"
            "incq %%r8;"
            "cmpq %%rdx, %%r8;"
            "jl iter;"
            :
            :"r"(arr), "r"(val), "c"(arr_len), "d"(iter)
            :"rax","rbx","r8"
            );
}
