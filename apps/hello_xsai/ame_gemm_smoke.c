#include <stdint.h>
#include <stdio.h>
#include "mem.h"

#define TILE 64
#define ELEM_COUNT (TILE * TILE)

int ame_gemm_smoke(void) {
    int8_t *smoke_a = (int8_t *)my_malloc(ELEM_COUNT * sizeof(int8_t));
    int8_t *smoke_b = (int8_t *)my_malloc(ELEM_COUNT * sizeof(int8_t));
    int32_t *smoke_c = (int32_t *)my_malloc(ELEM_COUNT * sizeof(int32_t));

    printf("AME GEMM smoke started: C[64x64] = A[64x64] * B[64x64]^T\n");

    if (!smoke_a || !smoke_b || !smoke_c) {
        printf("AME GEMM smoke failed: allocation failed\n");
        return 1;
    }

    for (int i = 0; i < ELEM_COUNT; i++) {
        smoke_a[i] = 1;
        smoke_b[i] = 1;
        smoke_c[i] = 0;
    }

    asm volatile(
        "li t0, 64\n\t"
        "msettilem t0\n\t"
        "msettilen t0\n\t"
        "msettilek t0\n\t"
        "msettilemi 64\n\t"
        "msettileni 64\n\t"
        "msettileki 64\n\t"

        "msyncregreset sync1\n\t"

        "li t6, 0x02\n\t"
        "msetcfg mcfg0, t6\n\t"
        "msetcfg mcfg1, t6\n\t"
        "msetcfg mcfg2, t6\n\t"
        "msetcfg mcfg3, t6\n\t"
        "li t6, 0x04\n\t"
        "msetcfg mcfg4, t6\n\t"
        "msetcfg mcfg5, t6\n\t"
        "msetcfg mcfg6, t6\n\t"
        "msetcfg mcfg7, t6\n\t"

        "li t1, 64\n\t"
        "li t3, 64\n\t"
        "li t5, 256\n\t"

        "mlc acc0, (%[c]), t5\n\t"
        "mla tr0, (%[a]), t1\n\t"
        "mlb tr1, (%[b]), t3\n\t"
        "mmacc acc0, tr1, tr0\n\t"
        "msc acc0, (%[c]), t5\n\t"

        "mrelease sync1\n\t"
        "li t0, 1\n\t"
        "macquire sync1, t0\n\t"
        "fence rw, rw\n\t"
        :
        : [a] "r"(smoke_a), [b] "r"(smoke_b), [c] "r"(smoke_c)
        : "t0", "t1", "t3", "t5", "t6", "memory");

    for (int i = 0; i < ELEM_COUNT; i++) {
        if (smoke_c[i] != TILE) {
            printf("AME GEMM smoke failed: C[%d] = %d, expected %d\n",
                   i, smoke_c[i], TILE);
            printf("samples: C[0]=%d C[63]=%d C[64]=%d C[4095]=%d\n",
                   smoke_c[0], smoke_c[63], smoke_c[64], smoke_c[4095]);
            return 1;
        }
    }

    printf("AME GEMM smoke passed: all %d outputs are %d\n", ELEM_COUNT, TILE);
    return 0;
}
