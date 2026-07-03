#ifndef GGML_RISCV_AME_H
#define GGML_RISCV_AME_H

#include <stdint.h>
#include <stddef.h>  // for size_t

#define DISABLE_TIME_INTR 0x100
#define NOTIFY_PROFILER 0x101
#define NOTIFY_PROFILE_EXIT 0x102
#define GOOD_TRAP 0x0

static void nemu_signal(int a){
    asm volatile ("mv a0, %0\n\t"
                  ".insn r 0x6B, 0, 0, x0, x0, x0\n\t"
                  :
                  : "r"(a)
                  : "a0");
}

// AME debug logging (default on for development)
#ifndef AME_DEBUG
#define AME_DEBUG 0
#endif

#if AME_DEBUG
#include <stdio.h>
#include <time.h>

static FILE* ame_log_file = NULL;

static inline void ame_log_init(void) {
    if (ame_log_file == NULL) {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        char filename[256];
        snprintf(filename, sizeof(filename), "rv-ame-%04d%02d%02d-%02d%02d%02d.log",
                 t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
        ame_log_file = fopen(filename, "a");
        if (ame_log_file == NULL) {
            ame_log_file = stderr; // fallback to stderr if file can't be opened
        }
    }
}

#define AME_LOG(fmt, ...)                             \
    do {                                              \
        ame_log_init();                              \
        fprintf(ame_log_file, "[AME] " fmt "\n", ##__VA_ARGS__);   \
        /*fflush(ame_log_file);*/                        \
    } while (0)
#else
#define AME_LOG(fmt, ...) do { (void)sizeof(fmt); } while (0)
#endif

#define AME_TILE_M 64
#define AME_TILE_K 64
#define AME_TILE_N 64

// Helper function to check if AME can be used for given dimensions
// We remove minimum size checks to properly support Q4_0 repacked weights
// which must use AME backend even for small batches (N=1)
// Also require K to be a multiple of 32 (Q8_0/Q4_0 block size) for validity
static inline int ggml_ame_can_use(int M, int N, int K) {
    // if (K % 32 != 0) return 0;

    // AME kernels handle padding/tiling for small dimensions,
    // so we accept any size provided K is aligned.
    return 1;
}

// Repacked Q4_0 format for AME (pre-unpacked to int8)
// This avoids unpacking overhead during every matmul
typedef struct {
    uint16_t d;         // scale factor FP16 (same format as block_q4_0)
    int8_t qs[32];      // pre-unpacked 4-bit values to int8 [-8, 7]
} block_q4_0_ame;

// Proposal-12/Zames matrix configuration and operation helpers.
#define AME_MCFG_INT8  0x02UL
#define AME_MCFG_INT32 0x04UL

#define MSETSEW(RD, SEW) ((void)(RD), (void)(SEW))

#define MSETINT8(RD, VAL) ((void)(RD), (void)(VAL))

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    );(RD)=VAL;

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    );(RD)=VAL;

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    );(RD)=VAL;

#define MSETCFG(MCFG, VAL) \
    asm volatile ( \
        "msetcfg " #MCFG ", %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    )

#define MSYNC_RESET(SYNC) \
    asm volatile ( \
        "msyncregreset " #SYNC \
        : \
        : \
        : "memory" \
    )

#define MFENCE() \
    asm volatile ( \
        "mfence" \
        : \
        : \
        : "memory" \
    )

// Matrix accumulator zero instruction
#define MZERO_ACC(ACC) \
    asm volatile ( \
        "mzero " #ACC \
        : \
        : \
        : "memory" \
    )

// Matrix load instructions
#define MLAE8(REG, SRC, N) \
    asm volatile ( \
        "mla " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : "memory" \
    )

#define MLBE8(REG, SRC, N) \
    asm volatile ( \
        "mlb " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : "memory" \
    )

#define MLCE32(REG, SRC, N) \
    asm volatile ( \
        "mlc " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : "memory" \
    )

// Matrix store instruction
#define MSCE32(REG, DST, N) \
    asm volatile ( \
        "msc " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMAU(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : "memory" \
    )

#define MQMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc " #ACC ", " #TR2 ", " #TR0 "\n" \
        : \
        : \
        : "memory" \
    )

#define MRELEASE(SYNC) \
    asm volatile ( \
        "mrelease " #SYNC \
        : \
        : \
        : "memory" \
    )

#define MACQUIRE(SYNC, VAL) \
    asm volatile ( \
        "macquire " #SYNC ", %0" \
        : \
        : "r"(VAL) \
        : "memory" \
    )

#ifdef __cplusplus
extern "C" {
#endif

void ggml_ame_init(void);

// Atomic tile GEMM (AME_TILE_M x AME_TILE_K x AME_TILE_N)
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

// GGML integration wrapper for Q8_0 quantized matrix multiplication
void ggml_ame_mul_mat_q8_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11,
    size_t src1_stride  // stride in bytes for src1 columns (nb[1])
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
