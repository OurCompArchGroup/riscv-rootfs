#ifndef GGML_RISCV_AME_H
#define GGML_RISCV_AME_H

#include <stdint.h>
#include <stddef.h>  // for size_t
#include <stdarg.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// AME debug logging can be forced at build time with AME_DEBUG=1 or enabled at
// runtime with GGML_AME_LOG=1 so logs are visible in QEMU serial output.
#ifndef AME_DEBUG
#define AME_DEBUG 0
#endif

static inline int ame_log_enabled(void) {
#if AME_DEBUG
    return 1;
#else
    static int cached = -1;
    if (cached == -1) {
        const char * value = getenv("GGML_AME_LOG");
        cached = 0;
        if (value != NULL && value[0] != '\0' && strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "off") != 0 && strcmp(value, "no") != 0) {
            cached = 1;
        }
    }
    return cached;
#endif
}

static inline uint64_t ame_log_seq_next(void) {
    static uint64_t seq = 0;
#if defined(__GNUC__) || defined(__clang__)
    return __atomic_fetch_add(&seq, 1, __ATOMIC_RELAXED);
#else
    return seq++;
#endif
}

static inline void ame_logf(const char * fmt, ...) {
    if (!ame_log_enabled()) {
        return;
    }

    const unsigned long long ame_seq = (unsigned long long) ame_log_seq_next();
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[AME:%016llx] ", ame_seq);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

#define AME_LOG(...) \
    do {             \
        ame_logf(__VA_ARGS__); \
    } while (0)

#define AME_TILE_M 64
#define AME_TILE_K 64
#define AME_TILE_N 64

#define AME_Q8_PACK_K 64

// Helper function to check if AME can be used for given dimensions.
//
// The current AME backend only accelerates Q8_0 GEMM. It only reaches the
// AME core on full MxN tiles; smaller shapes fall back to RVV/scalar helpers
// and usually lose to the generic CPU path once the extra quantize/pack cost is
// accounted for.
static inline int ggml_ame_can_use(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    if (K % 32 != 0) return 0;
    if (M < AME_TILE_M) return 0;
    if (N < AME_TILE_N) return 0;
    return 1;
}

// Repacked Q4_0 format for AME (pre-unpacked to int8)
// This avoids unpacking overhead during every matmul
typedef struct {
    uint16_t d;         // scale factor FP16 (same format as block_q4_0)
    int8_t qs[32];      // pre-unpacked 4-bit values to int8 [-8, 7]
} block_q4_0_ame;

typedef struct {
    uint16_t d;
    int8_t qs[AME_Q8_PACK_K];
} block_q8_ame64;

#define AME_MCFG_INT8  0x02
#define AME_MCFG_UINT8 0x03
#define AME_MCFG_INT32 0x04

// Matrix configuration instructions
#ifdef STC
#define MSETSEW(RD, SEW) \
    asm volatile ( \
        "msetsew %0, %1" \
        : "=r"(RD) \
        : "i"(SEW) \
        : \
    )

#define MSETINT8(RD, VAL) \
    asm volatile ( \
        "msetint8 %0, %1" \
        : "=r"(RD) \
        : "i"(VAL) \
        : \
    )

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

// Matrix accumulator zero instruction
#define MZERO(ACC) \
    asm volatile ( \
        "mzero " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLA(REG, SRC, N) \
    asm volatile ( \
        "mla " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLB(REG, SRC, N) \
    asm volatile ( \
        "mlb " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLC(REG, SRC, N) \
    asm volatile ( \
        "mlc " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSC(REG, DST, N) \
    asm volatile ( \
        "msc " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMACC(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : \
    )
#else
#define MSETSEW(RD, SEW) ((void)0) //非STC没有这条指令, 空指令

#define MSETINT8(RD, VAL) ((void)0) //非STC没有这条指令, 空指令

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

// Matrix accumulator zero instruction
#define MZERO(ACC) \
    asm volatile ( \
        "mzero " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLA(REG, SRC, N) \
    asm volatile ( \
        "mla " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLB(REG, SRC, N) \
    asm volatile ( \
        "mlb " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLC(REG, SRC, N) \
    asm volatile ( \
        "mlc " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSC(REG, DST, N) \
    asm volatile ( \
        "msc " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMACC(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : \
    )
#endif

#define MSETCFG(CFG, VAL) \
    asm volatile ( \
        "msetcfg " #CFG ", %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    )

static inline void ggml_ame_config_i8_i32(void) {
#if defined(__riscv)
    const unsigned long cfg_i8 = AME_MCFG_INT8;
    const unsigned long cfg_i32 = AME_MCFG_INT32;
    MSETCFG(mcfg0, cfg_i8);
    MSETCFG(mcfg1, cfg_i8);
    MSETCFG(mcfg2, cfg_i8);
    MSETCFG(mcfg3, cfg_i8);
    MSETCFG(mcfg4, cfg_i32);
    MSETCFG(mcfg5, cfg_i32);
    MSETCFG(mcfg6, cfg_i32);
    MSETCFG(mcfg7, cfg_i32);
#endif
}

#ifdef __cplusplus
extern "C" {
#endif

// Atomic tile GEMM (16x32x16)
void ggml_ame_gemm_tile_i8_i32_bT(
    const int8_t * A,
    const int8_t * B,
    int32_t * C
);

// Core AME GEMM function for INT8 matrix multiplication
// C(M×N) += A(M×K) × B(K×N), where B is transposed in memory
void ggml_ame_gemm_q8_0(
    const int8_t * A,
    const int8_t * B,
    int32_t * C,
    int M,
    int K,
    int N
);

// Quantize a row of F32 values to Q8_0 format
void ggml_ame_quantize_row_f32_to_q8_0(const float * x, void * y, int k);

// Q4_0 weight repacking (called once during set_tensor)
void ggml_ame_repack_q4_0(
    void * dst,              // Output: block_q4_0_ame array
    const void * src,        // Input: block_q4_0 array
    int64_t nblocks          // Number of Q4_0 blocks
);

// Experimental Q8_0 -> AME-native packed K64 repack.
void ggml_ame_repack_q8_0_to_ame64(
    void * dst,
    const void * src,
    int64_t nrows,
    int64_t k
);

// GGML integration wrapper for Q8_0 quantized matrix multiplication
void ggml_ame_mul_mat_q8_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,  // stride in bytes for src1 columns (nb[1])
    void * work_data,
    size_t work_size
);

void ggml_ame_mul_mat_q8_0_ame64(
    const void * src0,
    const void * src1_key,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride,
    int graph_id,
    void * work_data,
    size_t work_size
);

// GGML integration wrapper for Q4_0 quantized matrix multiplication
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride
);

#ifdef __cplusplus
}
#endif

#endif // GGML_RISCV_AME_H
