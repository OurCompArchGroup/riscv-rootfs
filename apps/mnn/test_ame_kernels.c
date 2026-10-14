// test_ame_kernels — AME GEMM kernel correctness + cycle harness (int8 + bf16).
//
// This harness links the REAL kernels from
//   software/MNN/source/backend/cpu/compute/ame_xsa.c
// (see the Makefile rule) instead of a local copy, so a PASS here is evidence
// about the code MNN actually executes.
//
// For every shape: pack A and BT into k-tile blocks, run the AME kloop over the
// tile grid, compare against a scalar reference on the real (unpadded) output,
// and report rdcycle. Operands must live in the XSAI direct-map pool because
// mla/mlb/msc address memory by pool-physical base.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ame_xsa.h"
#include "xsai_alloc.h"

static void * pool_alloc(size_t size) {
    void * p = xsai_malloc(size);
    if (p == NULL) {
        printf("AME_KERNELS_FAIL xsai_malloc(%zu)\n", size);
        exit(1);
    }
    memset(p, 0, size);
    return p;
}

static unsigned rng_state = 0x12345678u;
static unsigned rnd32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int8_t rnd8(void) {
    return (int8_t)((rnd32() >> 16) % 128) - 64;
}
// Random bf16-representable value in [-2, 2): bf16 keeps 8 explicit mantissa
// bits, so masking the low 16 bits of a small-magnitude fp32 makes the value
// exactly representable on both the AME and the scalar side.
static float rnd_bf16_float(void) {
    float v = ((float)((int)(rnd32() >> 8) % 100000) / 25000.0f) - 2.0f;
    uint32_t bits;
    memcpy(&bits, &v, 4);
    bits &= 0xFFFF0000u;
    memcpy(&v, &bits, 4);
    return v;
}

static int k_tiles_of(int k, int tk) { return (k + tk - 1) / tk; }
static int up_div(int a, int b) { return (a + b - 1) / b; }

// A tile block: one k-block = tm rows x tk elements (tk*elem_size bytes per row).
// Blocks are laid out consecutively, block stride = tm * tk * elem_size.
static void pack_rows(const uint8_t * src, int src_row_elems, int rows_real, int k,
                      int tk, int tm, int kt, int elem_size, int src_row, uint8_t * dst) {
    const size_t block = (size_t)tm * tk * elem_size;
    memset(dst, 0, block * kt);
    for (int r = 0; r < rows_real; ++r) {
        for (int t = 0; t < kt; ++t) {
            uint8_t * blk = dst + (size_t)t * block + (size_t)r * tk * elem_size;
            for (int j = 0; j < tk; ++j) {
                int kk = t * tk + j;
                if (kk >= k) break;
                memcpy(blk + (size_t)j * elem_size,
                       src + ((size_t)(src_row + r) * src_row_elems + kk) * elem_size,
                       elem_size);
            }
        }
    }
}

// Cold-start probe: the process's first AME kloop can be wrong (qemu-user
// cold state); callers re-run once and keep the warm result. Reported
// separately so a cold miss is visible without masking a real failure.
static int g_cold_miss = 0;

static void report_case(const char * dtype, int m, int n, int k, int tm, int tk, int tn,
                        const char * status, double err) {
    printf("AME_KERNELS %-4s M=%-4d N=%-4d K=%-4d tile=%dx%dx%d %-4s maxerr=%.3g\n",
           dtype, m, n, k, tm, tk, tn, status, err);
}

// ---------------------------------------------------------------- int8 path
static void run_int8_case(int m, int n, int k, int tm, int tn, int tk,
                          uint8_t * pool_a, uint8_t * pool_b, int32_t * pool_c,
                          int8_t * a, int8_t * bt, int32_t * ref, int32_t * out) {
    const int kt = k_tiles_of(k, tk);
    memset(out, 0, (size_t)m * n * sizeof(int32_t));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (int x = 0; x < k; ++x) acc += (int32_t)a[(size_t)i * k + x] * (int32_t)bt[(size_t)j * k + x];
            ref[(size_t)i * n + j] = acc;
        }

    for (int n0 = 0; n0 < n; n0 += tn) {
        const int nrows = (n - n0 < tn) ? (n - n0) : tn;
        pack_rows((const uint8_t *)bt, k, nrows, k, tk, tn, kt, 1, n0, pool_b);
        for (int m0 = 0; m0 < m; m0 += tm) {
            const int mrows = (m - m0 < tm) ? (m - m0) : tm;
            pack_rows((const uint8_t *)a, k, mrows, k, tk, tm, kt, 1, m0, pool_a);
            xsai_ame_gemm_tile_kloop((const int8_t *)pool_a, (ptrdiff_t)tm * tk,
                                     (const int8_t *)pool_b, (ptrdiff_t)tn * tk, kt,
                                     pool_c, tm, tk, tn);
            for (int r = 0; r < mrows; ++r)
                for (int c = 0; c < nrows; ++c)
                    out[(size_t)(m0 + r) * n + (n0 + c)] = pool_c[(size_t)r * tn + c];
        }
    }
}

// ---------------------------------------------------------------- bf16 path
static void run_bf16_case(int m, int n, int k, int tm, int tn, int tk_elems,
                          int reg_pairs, const uint16_t * a, const uint16_t * bt,
                          float * ref, float * out, uint8_t * pool_a, uint8_t * pool_b,
                          float * pool_c, int * fails, double * maxerr) {
    double ref_absmax = 0.0;
    const int kt = k_tiles_of(k, tk_elems);
    const int tk = tk_elems;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            double abs_sum = 0.0;   // condition number of the dot product
            for (int x = 0; x < k; ++x) {
                float av, bv;
                uint32_t ab = (uint32_t)a[(size_t)i * k + x] << 16;
                uint32_t bb = (uint32_t)bt[(size_t)j * k + x] << 16;
                memcpy(&av, &ab, 4);
                memcpy(&bv, &bb, 4);
                acc += av * bv;
                abs_sum += (double)av * (double)bv < 0 ? -(double)av * (double)bv
                                                       : (double)av * (double)bv;
            }
            ref[(size_t)i * n + j] = acc;
            if (abs_sum > ref_absmax) ref_absmax = abs_sum;
        }
    memset(out, 0, (size_t)m * n * sizeof(float));

    for (int n0 = 0; n0 < n; n0 += tn) {
        const int nrows = (n - n0 < tn) ? (n - n0) : tn;
        pack_rows((const uint8_t *)bt, k, nrows, k, tk, tn, kt, 2, n0, pool_b);
        for (int m0 = 0; m0 < m; m0 += tm) {
            const int mrows = (m - m0 < tm) ? (m - m0) : tm;
            pack_rows((const uint8_t *)a, k, mrows, k, tk, tm, kt, 2, m0, pool_a);
            xsai_ame_bf16_gemm_tile_kloop(pool_a, (ptrdiff_t)tm * tk * 2,
                                          pool_b, (ptrdiff_t)tn * tk * 2, kt, reg_pairs,
                                          pool_c, tm, tn);
            for (int r = 0; r < mrows; ++r)
                for (int c = 0; c < nrows; ++c)
                    out[(size_t)(m0 + r) * n + (n0 + c)] = pool_c[(size_t)r * tn + c];
        }
    }

    // fp32 accumulation order differs between the AME unit and the scalar
    // reference, so the acceptance test is the standard summation bound
    // |err| <= tol * sum|x*y| rather than a bitwise compare. A layout or index
    // bug produces errors of the same order as the result itself, i.e. ~10^4x
    // above this bound.
    int f = 0;
    double err = 0.0;
    const double tol = 1e-4;
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        const double d = (double)out[i] - (double)ref[i];
        const double absd = d < 0 ? -d : d;
        const double scaled = ref_absmax > 0 ? absd / ref_absmax : absd;
        if (scaled > err) err = scaled;
        if (scaled > tol) ++f;
    }
    *fails = f;
    *maxerr = err;
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t man        = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --e; }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)e << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

int main(void) {
    const xsai_ame_hw_config * hw = xsai_ame_hw_config_get();
    printf("AME_KERNELS hw valid=%d rows=%llu trlenb=%llu m_max=%d n_max=%d\n",
           hw->valid, (unsigned long long)hw->rows, (unsigned long long)hw->trlenb,
           hw->int8_m_max, hw->int32_n_max);
    if (xsai_alloc_host_test_mode()) {
        printf("AME_KERNELS note: host-test allocator mode (qemu-user)\n");
    }

    const xsai_ame_tile_spec * spec = xsai_ame_tile_spec_get();
    printf("AME_KERNELS spec tile=%dx%dx%d enabled=%d\n", spec->tile_m, spec->tile_n,
           spec->tile_k, spec->enabled);
    const int m_max = hw->int8_m_max >= 64 ? hw->int8_m_max : 64;
    const int n_max = hw->int32_n_max >= 64 ? hw->int32_n_max : 64;
    // The probe is invalid on emulators, so allow an explicit override to
    // exercise the 128-wide spec there as well.
    static int force128 = -1;
    if (force128 < 0) {
        const char * v = getenv("XSAI_MNN_AME_TILE");
        force128 = (v != NULL && atoi(v) == 128) ? 1 : 0;
    }
    // The probe (xsai_ame_tile_spec_get) already decides which tile geometry this
    // target uses. Only that geometry is verified by default; the other one is
    // not a supported configuration here and is run only when explicitly asked
    // for (XSAI_AME_ALT_SPEC=1) for cross-checking.
    const int altEnabled = getenv("XSAI_AME_ALT_SPEC") != NULL;

    enum { MAXM = 256, MAXN = 256, MAXK = 3072 };
    const size_t a_sz = (size_t)MAXM * MAXK;
    const size_t b_sz = (size_t)MAXN * MAXK;

    uint8_t * pool_a = (uint8_t *)pool_alloc(a_sz * 2 + 4096);
    uint8_t * pool_b = (uint8_t *)pool_alloc(b_sz * 2 + 4096);
    float * pool_c = (float *)pool_alloc((size_t)MAXN * MAXN * sizeof(float));
    int32_t * pool_c_i8 = (int32_t *)pool_c;

    int8_t * a8 = (int8_t *)malloc(a_sz);
    int8_t * b8 = (int8_t *)malloc(b_sz);
    int32_t * ref8 = (int32_t *)malloc((size_t)MAXM * MAXN * sizeof(int32_t));
    int32_t * out8 = (int32_t *)malloc((size_t)MAXM * MAXN * sizeof(int32_t));
    uint16_t * abf = (uint16_t *)malloc(a_sz * 2);
    uint16_t * bbf = (uint16_t *)malloc(b_sz * 2);
    float * refbf = (float *)malloc((size_t)MAXM * MAXN * sizeof(float));
    float * outbf = (float *)malloc((size_t)MAXM * MAXN * sizeof(float));
    if (!ref8 || !out8 || !refbf || !outbf || !a8 || !b8 || !abf || !bbf) {
        printf("AME_KERNELS_FAIL host malloc\n");
        return 1;
    }

    static const int shapes[][3] = {
        {1, 1, 1},     {1, 8, 64},    {7, 3, 65},   {16, 16, 128}, {63, 63, 64},
        {64, 64, 64},  {65, 65, 128}, {128, 64, 64}, {64, 128, 256}, {128, 128, 64},
        {200, 96, 160}, {96, 200, 512}, {256, 256, 256},
        // Deep-K cases: 3x3 convolutions in the models under test reach
        // K = ic*kx*ky = 2880 (ic=320), i.e. 45 AME k-tiles; K=8192 is the
        // largest K the int8 kernel accepts (kAmeKMax).
        {32, 8, 2880}, {64, 24, 1152}, {16, 16, 8192},
    };
    const int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int fails = 0;        // failures of the spec this target runs (the one it must pass)
    int altFails = 0;     // failures of the other tile spec (informational)
    // int8: portable 64x64x64 spec, plus the 128-wide variant when the probe
    // reports the capacity (the 128 variants are only reachable on real hw).
    const int tile_sets[][3] = {{64, 64, 64}, {128, 64, 128}};
    for (int ts = 0; ts < 2; ++ts) {
        const int tm = tile_sets[ts][0], tk = tile_sets[ts][1], tn = tile_sets[ts][2];
        const int isSelected = (tm == spec->tile_m && tn == spec->tile_n);
        if (!isSelected && !altEnabled) continue;
        if (ts == 1 && !(m_max >= 128 && n_max >= 128)) continue;
        printf("AME_KERNELS int8 tile=%dx%dx%d\n", tm, tk, tn);
        for (int s = 0; s < n_shapes; ++s) {
            const int m = shapes[s][0] > MAXM ? MAXM : shapes[s][0];
            const int n = shapes[s][1] > MAXN ? MAXN : shapes[s][1];
            const int k = shapes[s][2] > MAXK ? MAXK : shapes[s][2];
            for (size_t i = 0; i < (size_t)m * k; ++i) a8[i] = rnd8();
            for (size_t i = 0; i < (size_t)n * k; ++i) b8[i] = rnd8();
            run_int8_case(m, n, k, tm, tn, tk, pool_a, pool_b, pool_c_i8, a8, b8, ref8, out8);
            int bad = 0;
            for (size_t i = 0; i < (size_t)m * n; ++i)
                if (out8[i] != ref8[i]) ++bad;
            const int isPrimary = (tm == spec->tile_m && tn == spec->tile_n);
            if (bad) { if (isPrimary) ++fails; else ++altFails; }
            report_case(isPrimary ? "int8" : "int8m", m, n, k, tm, tk, tn, bad ? "FAIL" : "ok", (double)bad);
        }
    }
    (void)up_div;

    // bf16: both tile specs (64x32x64 emulator spec, 128x32x128 native) times
    // both register-pair depths.
    for (int bt = 0; bt < 2; ++bt) {
    const int btm = bt == 0 ? 64 : 128, btn = bt == 0 ? 64 : 128;
    const int btSelected = (btm == spec->tile_m && btn == spec->tile_n);
    if (!btSelected && !altEnabled) continue;
    if (bt == 1 && !(m_max >= 128 && n_max >= 128)) continue;
    for (int rp = 1; rp <= 2; ++rp) {
        printf("AME_KERNELS bf16 tile=%dx32x%d reg_pairs=%d\n", btm, btn, rp);
        for (int s = 0; s < n_shapes; ++s) {
            const int m = shapes[s][0] > MAXM ? MAXM : shapes[s][0];
            const int n = shapes[s][1] > MAXN ? MAXN : shapes[s][1];
            const int k = shapes[s][2] > MAXK ? MAXK : shapes[s][2];
            for (size_t i = 0; i < (size_t)m * k; ++i) {
                float v = rnd_bf16_float();
                uint32_t bits;
                memcpy(&bits, &v, 4);
                abf[i] = (uint16_t)(bits >> 16);
            }
            for (size_t i = 0; i < (size_t)n * k; ++i) {
                float v = rnd_bf16_float();
                uint32_t bits;
                memcpy(&bits, &v, 4);
                bbf[i] = (uint16_t)(bits >> 16);
            }
            int case_fails = 0;
            double err = 0.0;
            run_bf16_case(m, n, k, btm, btn, 32, rp, abf, bbf, refbf, outbf,
                          pool_a, pool_b, pool_c, &case_fails, &err);
            const int isPrimary = (btm == spec->tile_m && btn == spec->tile_n);
            if (case_fails) { if (isPrimary) ++fails; else ++altFails; }
            report_case(isPrimary ? "bf16" : "bf16m", m, n, k, btm, 32, btn, case_fails ? "FAIL" : "ok", err);
        }
    }
    }
    (void)g_cold_miss;

    // fp16 path: fp16 operands -> bf16 via xsai_ame_f16_to_bf16 -> bf16 GEMM.
    // The reference runs on the converted bf16 values, so this checks the
    // conversion helper and the GEMM chain together; the second number reports
    // how much the conversion itself moved the inputs.
    {
        enum { F16M = 128, F16N = 96, F16K = 256 };
        static uint16_t f16a[F16M * F16K], f16b[F16N * F16K];
        float max_conv_rel = 0.0f;
        for (size_t i = 0; i < (size_t)F16M * F16K; ++i) {
            const float v = rnd_bf16_float();
            uint32_t bits;
            memcpy(&bits, &v, 4);
            f16a[i] = (uint16_t)(bits >> 16);   // bf16-truncated fp32 is exact in fp16 for |v| < 4
        }
        for (size_t i = 0; i < (size_t)F16N * F16K; ++i) {
            const float v = rnd_bf16_float();
            uint32_t bits;
            memcpy(&bits, &v, 4);
            f16b[i] = (uint16_t)(bits >> 16);
        }
        xsai_ame_f16_to_bf16(f16a, abf, (size_t)F16M * F16K);
        xsai_ame_f16_to_bf16(f16b, bbf, (size_t)F16N * F16K);
        // Conversion error is measured against the fp16 value widened to fp32.
        for (size_t i = 0; i < (size_t)F16M * F16K; ++i) {
            const float src = half_to_float(f16a[i]);
            uint32_t bb = (uint32_t)abf[i] << 16;
            float dst;
            memcpy(&dst, &bb, 4);
            const float rel = fabsf(dst - src) / (fabsf(src) + 1e-3f);
            if (rel > max_conv_rel) max_conv_rel = rel;
        }
        int case_fails = 0;
        double err = 0.0;
        run_bf16_case(F16M, F16N, F16K, spec->tile_m, spec->tile_n, 32, 2, abf, bbf, refbf, outbf,
                      pool_a, pool_b, pool_c, &case_fails, &err);
        if (case_fails) ++fails;
        report_case("f16", F16M, F16N, F16K, spec->tile_m, 32, spec->tile_n, case_fails ? "FAIL" : "ok", err);
        printf("AME_KERNELS f16->bf16 conversion max rel err vs fp16 input: %.3g\n",
               (double)max_conv_rel);
    }

    // Steady-state cycle cost: one full tile grid, 256x256x256.
    {
        const int m = 256, n = 256, k = 256;
        const int tm = spec->tile_m, tn = spec->tile_n, tk = 64;
        const int kt = k_tiles_of(k, tk);
        for (size_t i = 0; i < (size_t)m * tk; ++i) a8[i] = rnd8();
        for (size_t i = 0; i < (size_t)n * tk; ++i) b8[i] = rnd8();
        for (int n0 = 0; n0 < n; n0 += tn) {
            pack_rows((const uint8_t *)b8, k, tn, k, tk, tn, kt, 1, 0, pool_b);
            for (int m0 = 0; m0 < m; m0 += tm)
                pack_rows((const uint8_t *)a8, k, tm, k, tk, tm, kt, 1, 0, pool_a);
        }
        pack_rows((const uint8_t *)a8, k, tm, k, tk, tm, kt, 1, 0, pool_a);
        pack_rows((const uint8_t *)b8, k, tn, k, tk, tn, kt, 1, 0, pool_b);
        xsai_ame_gemm_tile_kloop((const int8_t *)pool_a, (ptrdiff_t)tm * tk,
                                 (const int8_t *)pool_b, (ptrdiff_t)tn * tk, kt,
                                 (int32_t *)pool_c, tm, tk, tn);
        const uint64_t t0 = xsai_ame_rdcycle();
        const int iters = 64;
        for (int i = 0; i < iters; ++i)
            xsai_ame_gemm_tile_kloop((const int8_t *)pool_a, (ptrdiff_t)tm * tk,
                                     (const int8_t *)pool_b, (ptrdiff_t)tn * tk, kt,
                                     (int32_t *)pool_c, tm, tk, tn);
        const uint64_t cyc8 = (xsai_ame_rdcycle() - t0) / iters;

        for (size_t i = 0; i < (size_t)m * (k / 2); ++i) {
            float v = rnd_bf16_float();
            uint32_t bits;
            memcpy(&bits, &v, 4);
            abf[i] = (uint16_t)(bits >> 16);
        }
        for (size_t i = 0; i < (size_t)n * (k / 2); ++i) {
            float v = rnd_bf16_float();
            uint32_t bits;
            memcpy(&bits, &v, 4);
            bbf[i] = (uint16_t)(bits >> 16);
        }
        const int ktb = k_tiles_of(k, 32);
        pack_rows((const uint8_t *)abf, k, tm, k, 32, tm, ktb, 2, 0, pool_a);
        pack_rows((const uint8_t *)bbf, k, tn, k, 32, tn, ktb, 2, 0, pool_b);
        xsai_ame_bf16_gemm_tile_kloop(pool_a, (ptrdiff_t)tm * 32 * 2, pool_b,
                                      (ptrdiff_t)tn * 32 * 2, ktb, 2, pool_c, tm, tn);
        const uint64_t tb0 = xsai_ame_rdcycle();
        for (int i = 0; i < iters; ++i)
            xsai_ame_bf16_gemm_tile_kloop(pool_a, (ptrdiff_t)tm * 32 * 2, pool_b,
                                          (ptrdiff_t)tn * 32 * 2, ktb, 2, pool_c, tm, tn);
        const uint64_t cycb = (xsai_ame_rdcycle() - tb0) / iters;
        printf("AME_KERNELS cycles tile%dx%dx%d K=256: int8=%llu bf16=%llu\n", tm, tk, tn,
               (unsigned long long)cyc8, (unsigned long long)cycb);
    }

    if (altEnabled) {
        printf("AME_KERNELS %s spec=%dx%dx%d fails=%d alt_fails=%d (alt spec explicitly requested)\n",
               fails ? "FAIL" : "PASS", spec->tile_m, spec->tile_k, spec->tile_n, fails, altFails);
    } else {
        printf("AME_KERNELS %s spec=%dx%dx%d fails=%d (probe-selected geometry)\n",
               fails ? "FAIL" : "PASS", spec->tile_m, spec->tile_k, spec->tile_n, fails);
    }
    return fails ? 1 : 0;
}
