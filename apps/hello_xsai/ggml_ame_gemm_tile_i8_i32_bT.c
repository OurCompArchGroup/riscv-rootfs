#include "ame.h"
#include <stddef.h>
#include <stdio.h>
// #include "bm_run.h"

// Zicbop prefetch hint — reads the cache line containing addr into dcache.
// No exception is raised if addr is invalid; purely a performance hint.
#define PREFETCH_R(addr) \
    asm volatile("prefetch.r 0(%0)" :: "r"(addr) : )

#define AME_TRACE(label) \
    do { \
        printf("[ame-trace] %s\n", label); \
        fflush(stdout); \
    } while (0)

static int tile_i8_all_zero(const int8_t *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (x[i] != 0) {
            return 0;
        }
    }
    return 1;
}

// INT8 GEMM using RISC-V AME instructions
// Tile size: M=AME_TILE_M, K=AME_TILE_K, N=AME_TILE_N (atomic AME variant)
// C(MxN) = A(MxK) × B^T(NxK), where B is transposed in memory
// This function computes a single MxN tile output
void ggml_ame_init(void) {
    const int TILE_M = AME_TILE_M;
    const int TILE_K = AME_TILE_K;
    const int TILE_N = AME_TILE_N;
    const unsigned long cfg_i8 = AME_MCFG_INT8;
    const unsigned long cfg_i32 = AME_MCFG_INT32;
    int tmp;

    AME_TRACE("init before msettilem");
    MSETTILEM(tmp, TILE_M);
    AME_TRACE("init after msettilem");
    AME_TRACE("init before msettilek");
    MSETTILEK(tmp, TILE_K);
    AME_TRACE("init after msettilek");
    AME_TRACE("init before msettilen");
    MSETTILEN(tmp, TILE_N);
    AME_TRACE("init after msettilen");
    AME_TRACE("init before msetcfg mcfg0");
    MSETCFG(mcfg0, cfg_i8);
    AME_TRACE("init after msetcfg mcfg0");
    AME_TRACE("init before msetcfg mcfg1");
    MSETCFG(mcfg1, cfg_i8);
    AME_TRACE("init after msetcfg mcfg1");
    AME_TRACE("init before msetcfg mcfg2");
    MSETCFG(mcfg2, cfg_i8);
    AME_TRACE("init after msetcfg mcfg2");
    AME_TRACE("init before msetcfg mcfg3");
    MSETCFG(mcfg3, cfg_i8);
    AME_TRACE("init after msetcfg mcfg3");
    AME_TRACE("init before msetcfg mcfg4");
    MSETCFG(mcfg4, cfg_i32);
    AME_TRACE("init after msetcfg mcfg4");
    AME_TRACE("init before msetcfg mcfg5");
    MSETCFG(mcfg5, cfg_i32);
    AME_TRACE("init after msetcfg mcfg5");
    AME_TRACE("init before msetcfg mcfg6");
    MSETCFG(mcfg6, cfg_i32);
    AME_TRACE("init after msetcfg mcfg6");
    AME_TRACE("init before msetcfg mcfg7");
    MSETCFG(mcfg7, cfg_i32);
    AME_TRACE("init after msetcfg mcfg7");
}

void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,      // Input matrix A: MxK
    const int8_t * B,      // Input matrix B (transposed): NxK
    int32_t * C            // Output matrix C: MxN
) {
#ifndef AME_DISABLE_ZERO_FAST_PATH
    if (tile_i8_all_zero(A, AME_TILE_M * AME_TILE_K) ||
        tile_i8_all_zero(B, AME_TILE_N * AME_TILE_K)) {
        AME_TRACE("kernel all-zero fast path return");
        return;
    }
#endif

    AME_TRACE("kernel before msyncregreset sync1");
    MSYNC_RESET(sync1);
    AME_TRACE("kernel after msyncregreset sync1");

    // ----------------------------------------------------------------
    // Prefetch A, B, C tiles into dcache before issuing AME loads.
    //
    // A: AME_TILE_M x AME_TILE_K  int8  = 64 x 64  =  4 KB
    //    row stride = AME_TILE_K = 64 B = 1 cache line (64 B)
    //    -> 64 rows x 1 line/row = 64 prefetches
    //
    // B^T: AME_TILE_N x AME_TILE_K int8  = 64 x 64  =  4 KB
    //    same layout, same count
    //
    // C: AME_TILE_M x AME_TILE_N  int32 = 64 x 64  = 16 KB
    //    row stride = AME_TILE_N * 4 = 256 B = 4 cache lines (64 B each)
    //    -> 64 rows x 4 lines/row = 256 prefetches
    // ----------------------------------------------------------------
   
#ifdef AME_PREFETCH
    const int CL = 64; /* cache line size in bytes */

    /* Prefetch A: 64 rows, each 64 B = 1 cache line */
    for (int i = 0; i < AME_TILE_M; i++) {
        PREFETCH_R(A + i * AME_TILE_K);
    }

    /* Prefetch B^T: 64 rows, each 64 B = 1 cache line */
    for (int i = 0; i < AME_TILE_N; i++) {
        PREFETCH_R(B + i * AME_TILE_K);
    }

    /* Prefetch C: 64 rows x 256 B/row = 4 cache lines per row */
    {
        const int stride_c_bytes = AME_TILE_N * (int)sizeof(int32_t); /* 256 B */
        const int lines_per_row  = stride_c_bytes / CL;               /* 4 */
        for (int i = 0; i < AME_TILE_M; i++) {
            for (int j = 0; j < lines_per_row; j++) {
                PREFETCH_R((const char*)C + i * stride_c_bytes + j * CL);
            }
        }
    }
#endif /* AME_PREFETCH */
    /* Preload C matrix to initialize accumulator */
    int32_t *addr_c = C;
    int stride_c = AME_TILE_N; /* Row stride (in elements) */

#ifdef AME_ENABLE_MZERO_ACC
    AME_TRACE("kernel before mzero acc0");
    MZERO_ACC(acc0);
    AME_TRACE("kernel after mzero acc0");
#endif

#ifdef AME_TILE_TRACE
    printf("addr_c: %p, stride_c: %d\n", (void*)addr_c, stride_c);
#endif
    printf("[ame-trace] ptr A=%p B=%p C=%p stride_c_bytes=%d\n",
           (const void *)A, (const void *)B, (void *)C, stride_c * 4);
    fflush(stdout);

    AME_TRACE("kernel before mlc acc0");
    MLCE32(acc0, addr_c, stride_c * 4);
    AME_TRACE("kernel after mlc acc0");

    /* Load left matrix A tile: MxK */
    const int8_t *addr_a = A;
    AME_TRACE("kernel before mla tr0");
    MLAE8(tr0, addr_a, AME_TILE_K);
    AME_TRACE("kernel after mla tr0");

    /* Load right matrix B tile (transposed): NxK */
    const int8_t *addr_b = B;
    AME_TRACE("kernel before mlb tr1");
    MLBE8(tr1, addr_b, AME_TILE_K);
    AME_TRACE("kernel after mlb tr1");

    /* INT8 matrix multiply-accumulate: C(MxN) = A(MxK) x B^T(NxK) */
    AME_TRACE("kernel before mmacc acc0 tr1 tr0");
    MQMA(acc0, tr0, tr1);
    AME_TRACE("kernel after mmacc acc0 tr1 tr0");

    /* Store INT32 result to C (MxN) */
    AME_TRACE("kernel before msc acc0");
    MSCE32(acc0, addr_c, stride_c * 4);
    AME_TRACE("kernel after msc acc0");

    AME_TRACE("kernel before mrelease sync1");
    MRELEASE(sync1);
    AME_TRACE("kernel after mrelease sync1");
    int acquire_target = 1;
    AME_TRACE("kernel before macquire sync1");
    MACQUIRE(sync1, acquire_target);
    AME_TRACE("kernel after macquire sync1");
    AME_TRACE("kernel before mfence");
    MFENCE();
    AME_TRACE("kernel after mfence");
}
