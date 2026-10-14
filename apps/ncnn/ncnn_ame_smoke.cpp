// ncnn_ame_smoke.cpp — differential smoke for the XSAI AME int8 conv path.
//
// Runs the NCNN "Convolution" layer (registry entry → Convolution_riscv on a
// riscv build) on a set of deterministic int8 problems in two modes:
//
//   ./ncnn_ame_smoke ref   → XSAI_NCNN_AME forced OFF  (stock scalar kernel)
//   ./ncnn_ame_smoke ame   → XSAI_NCNN_AME forced ON   (AME int8 GEMM path)
//
// Both modes dump every top_blob to <mode>.out as text. The caller diffs the
// two files (same code path except the AME branch, same arithmetic per the
// base-kernel epilogue → bit-identical output is expected on both qemu-user
// and hardware). Shapes cover K = c*kh*kw around 64-byte tile boundaries,
// multi-tile M/N, stride/dilation and the requant (int8 out) epilogue.
#include "net.h"
#include "mat.h"
#include "layer.h"
#include "layer/convolution.h"
#include "paramdict.h"
#include "modelbin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// deterministic LCG so ref/ame runs see identical inputs
static unsigned int g_seed = 20260930u;
static float frand(float lo, float hi)
{
    g_seed = g_seed * 1103515245u + 12345u;
    const float r = (float)((g_seed >> 8) & 0xffffff) / (float)0xffffff;
    return lo + r * (hi - lo);
}

static ncnn::Mat random_mat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    float* p = m;
    for (int i = 0; i < m.total(); i++)
        p[i] = frand(-1.f, 1.f);
    return m;
}

// per-row absmax scale (mirrors tests/testutil.cpp scales_mat semantics)
static ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx)
{
    ncnn::Mat s(m);
    for (int i = 0; i < m; i++)
    {
        const float* ptr = (const float*)mat.data + (size_t)i * ldx;
        float amax = 0.f;
        for (int j = 0; j < k; j++)
        {
            const float v = fabsf(ptr[j]);
            if (v > amax) amax = v;
        }
        if (amax < 1e-8f) amax = 1e-8f;
        s[i] = 127.f / amax;
    }
    return s;
}

struct Case
{
    int w, h, c, outch, k, d, s, p, bias;
    int int8_scale_term; // 1 = dequant fp32 out, 101 = requant int8 out
};

static int run_case(const Case& t, FILE* out)
{
    g_seed = 20260930u + (unsigned)t.w * 7 + (unsigned)t.c * 13 + (unsigned)t.k;

    ncnn::Mat a = random_mat(t.w, t.h, t.c);
    const int maxk = t.k * t.k;
    const size_t wtotal = (size_t)t.outch * t.c * maxk;
    ncnn::Mat wmat((int)wtotal);
    {
        float* p = wmat;
        for (size_t i = 0; i < wtotal; i++)
            p[i] = frand(-1.f, 1.f);
    }

    ncnn::Mat weight_scales = scales_mat(wmat, t.outch, t.c * maxk, t.c * maxk);
    ncnn::Mat input_scales = scales_mat(a, 1, a.total(), a.cstep);
    ncnn::Mat top_scales = (t.int8_scale_term > 100) ? scales_mat(a, 1, a.total(), a.cstep) : ncnn::Mat();

    ncnn::ParamDict pd;
    pd.set(0, t.outch);
    pd.set(1, t.k);
    pd.set(2, t.d);
    pd.set(3, t.s);
    pd.set(4, t.p);
    pd.set(5, t.bias);
    pd.set(6, (int)wtotal);
    pd.set(8, t.int8_scale_term);
    pd.set(9, 0); // no activation for determinism
    pd.set(10, ncnn::Mat());

    std::vector<ncnn::Mat> weights(t.bias ? 5 : 4);
    weights[0] = wmat;
    if (t.bias)
    {
        weights[1] = ncnn::Mat(t.outch);
        for (int i = 0; i < t.outch; i++)
            weights[1][i] = frand(-0.5f, 0.5f);
        weights[2] = weight_scales;
        weights[3] = input_scales;
        weights[4] = top_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
        weights[3] = top_scales;
    }

    ncnn::Layer* op = ncnn::create_layer("Convolution");
    if (!op)
    {
        fprintf(stderr, "create_layer(Convolution) failed\n");
        return -1;
    }
    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;
    opt.use_int8_inference = true;
    opt.use_int8_storage = true;
    opt.blob_allocator = 0;
    opt.workspace_allocator = 0;

    op->create_pipeline(opt);

    ncnn::Mat top;
    const int ret = op->forward(a, top, opt);
    op->destroy_pipeline(opt);
    delete op;
    if (ret != 0)
    {
        fprintf(stderr, "forward failed ret=%d\n", ret);
        return -1;
    }

    fprintf(out, "case w=%d h=%d c=%d oc=%d k=%d d=%d s=%d p=%d bias=%d scale_term=%d out=%dx%dx%d es=%zu\n",
            t.w, t.h, t.c, t.outch, t.k, t.d, t.s, t.p, t.bias, t.int8_scale_term,
            top.w, top.h, top.c, top.elemsize);
    for (int q = 0; q < top.c; q++)
    {
        const ncnn::Mat ch = top.channel(q);
        const int n_out = ch.w * ch.h; // NOT total(): cstep padding is uninitialized
        if (top.elemsize == 4)
        {
            const float* p = (const float*)ch;
            for (int i = 0; i < n_out; i++)
                fprintf(out, "%.6e\n", (double)p[i]);
        }
        else
        {
            const signed char* p = (const signed char*)ch.data;
            for (int i = 0; i < n_out; i++)
                fprintf(out, "%d\n", (int)p[i]);
        }
    }
    fflush(out);
    return 0;
}

// ---------------------------------------------------------------------------
// bf16 cases: fp32 layer with the AME bf16 matmul inside (XSAI_NCNN_AME_BF16).
// Primary gate is in-process: AME output vs a scalar bf16 reference (same
// truncating fp32->bf16 conversion, fp32 accumulation) — only accumulation
// order differs, so the tolerance is a few ulps. The cross-process dump is
// kept for a loose fp32 sanity compare by the Makefile target.
// ---------------------------------------------------------------------------
struct Bf16Case
{
    int w, h, c, outch, k, d, s, p, bias;
};

static inline uint16_t f32_to_bf16_trunc(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(uint16_t v)
{
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// scalar direct conv on the bf16 grid, zero padding, no activation
static void conv_ref_bf16(const Bf16Case& t, const ncnn::Mat& a, const ncnn::Mat& wmat,
                          const ncnn::Mat& bias, std::vector<float>& out)
{
    const int maxk = t.k * t.k;
    const int ext = t.d * (t.k - 1) + 1;
    const int outw = (t.w + 2 * t.p - ext) / t.s + 1;
    const int outh = (t.h + 2 * t.p - ext) / t.s + 1;
    out.assign((size_t)t.outch * outw * outh, 0.f);

    // bf16-grid copies of input and weights (truncation matches the kernel pack)
    std::vector<uint16_t> ain((size_t)t.w * t.h * t.c);
    for (int q = 0; q < t.c; q++)
        for (int y = 0; y < t.h; y++)
            for (int x = 0; x < t.w; x++)
                ain[((size_t)q * t.h + y) * t.w + x] = f32_to_bf16_trunc(a.channel(q).row(y)[x]);
    std::vector<uint16_t> win((size_t)t.outch * t.c * maxk);
    for (size_t i = 0; i < win.size(); i++)
        win[i] = f32_to_bf16_trunc(((const float*)wmat.data)[i]);

    for (int oc = 0; oc < t.outch; oc++)
    {
        for (int oy = 0; oy < outh; oy++)
        {
            for (int ox = 0; ox < outw; ox++)
            {
                float acc = 0.f;
                const int iy0 = oy * t.s - t.p;
                const int ix0 = ox * t.s - t.p;
                for (int q = 0; q < t.c; q++)
                {
                    for (int ky = 0; ky < t.k; ky++)
                    {
                        for (int kx = 0; kx < t.k; kx++)
                        {
                            const int iy = iy0 + ky * t.d;
                            const int ix = ix0 + kx * t.d;
                            float v = 0.f;
                            if (iy >= 0 && iy < t.h && ix >= 0 && ix < t.w)
                                v = bf16_to_f32(ain[((size_t)q * t.h + iy) * t.w + ix]);
                            const float wv = bf16_to_f32(win[((size_t)oc * t.c + q) * maxk + ky * t.k + kx]);
                            acc += v * wv;
                        }
                    }
                }
                if (t.bias)
                    acc += bias[oc];
                out[((size_t)oc * outh + oy) * outw + ox] = acc;
            }
        }
    }
}

static int run_bf16_case(const Bf16Case& t, FILE* out)
{
    g_seed = 20261001u + (unsigned)t.w * 11 + (unsigned)t.c * 17 + (unsigned)t.k * 3;

    ncnn::Mat a = random_mat(t.w, t.h, t.c);
    const int maxk = t.k * t.k;
    const size_t wtotal = (size_t)t.outch * t.c * maxk;
    ncnn::Mat wmat((int)wtotal);
    {
        float* p = wmat;
        for (size_t i = 0; i < wtotal; i++)
            p[i] = frand(-1.f, 1.f);
    }
    ncnn::Mat bias;
    if (t.bias)
    {
        bias = ncnn::Mat(t.outch);
        for (int i = 0; i < t.outch; i++)
            bias[i] = frand(-0.5f, 0.5f);
    }

    ncnn::ParamDict pd;
    pd.set(0, t.outch);
    pd.set(1, t.k);
    pd.set(2, t.d);
    pd.set(3, t.s);
    pd.set(4, t.p);
    pd.set(5, t.bias);
    pd.set(6, (int)wtotal);
    pd.set(8, 0); // no int8
    pd.set(9, 0);
    pd.set(10, ncnn::Mat());

    std::vector<ncnn::Mat> weights(2);
    weights[0] = wmat;
    weights[1] = bias;

    ncnn::Layer* op = ncnn::create_layer("Convolution");
    if (!op)
        return -1;
    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;
    opt.use_int8_inference = false;
    opt.use_int8_storage = false;

    op->create_pipeline(opt);

    ncnn::Mat top;
    const int ret = op->forward(a, top, opt);
    op->destroy_pipeline(opt);
    delete op;
    if (ret != 0)
    {
        fprintf(stderr, "[bf16-smoke] forward failed ret=%d\n", ret);
        return -1;
    }

    std::vector<float> ref;
    conv_ref_bf16(t, a, wmat, bias, ref);

    // in-process gate: AME vs scalar bf16 reference (accumulation order only)
    const bool ame_mode = (getenv("XSAI_NCNN_AME_BF16") != 0 && getenv("XSAI_NCNN_AME_BF16")[0] == '1');
    double max_abs = 0.0, max_rel = 0.0;
    size_t bad = 0;
    size_t idx = 0;
    for (int q = 0; q < t.outch; q++)
    {
        const float* tp = (const float*)top.channel(q);
        for (int i = 0; i < top.w * top.h; i++, idx++)
        {
            const double got = tp[i];
            const double want = ref[idx];
            const double ad = fabs(got - want);
            const double rd = ad / (fabs(want) > 1e-3 ? fabs(want) : 1e-3);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (ad > 2e-3 + 1e-4 * fabs(want))
                bad++;
        }
    }
    fprintf(stderr, "[bf16-smoke] w=%d h=%d c=%d oc=%d k=%d d=%d s=%d p=%d bias=%d ame=%d "
                    "max_abs=%.3e max_rel=%.3e bad=%zu/%zu\n",
            t.w, t.h, t.c, t.outch, t.k, t.d, t.s, t.p, t.bias, (int)ame_mode,
            max_abs, max_rel, bad, idx);
    if (ame_mode && bad > 0)
        return -1;

    fprintf(out, "bf16case w=%d h=%d c=%d oc=%d k=%d d=%d s=%d p=%d bias=%d out=%dx%dx%d es=%zu\n",
            t.w, t.h, t.c, t.outch, t.k, t.d, t.s, t.p, t.bias,
            top.w, top.h, top.c, top.elemsize);
    for (int q = 0; q < top.c; q++)
    {
        const ncnn::Mat ch = top.channel(q);
        const int n_out = ch.w * ch.h;
        const float* p = (const float*)ch;
        for (int i = 0; i < n_out; i++)
            fprintf(out, "%.6e\n", (double)p[i]);
    }
    fflush(out);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s ref|ame <int8-out-file> [bf16-out-file]\n", argv[0]);
        return 2;
    }
    const bool ame = (strcmp(argv[1], "ame") == 0);
    FILE* out = fopen(argv[2], "w");
    if (!out)
    {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 2;
    }
    FILE* bf16_out = (argc >= 4) ? fopen(argv[3], "w") : out;
    if (!bf16_out)
    {
        fprintf(stderr, "cannot open %s\n", argv[3]);
        fclose(out);
        return 2;
    }
    // must run before the first layer call: the gate caches on first use
    setenv("XSAI_NCNN_AME", ame ? "1" : "0", 1);
    // XSAI_ALLOC_MODE is deliberately NOT set here: anon is a qemu-user-only
    // host-test mode (anonymous memory is NOT physically contiguous, the AMU
    // would read garbage PAs on real hardware). qemu runs must export
    // XSAI_ALLOC_MODE=anon themselves; the board uses the default /dev/mem
    // physical pool, exactly like the MNN flows.
    // Exercise even tiny shapes in BOTH modes so the two runs walk the same
    // case set through different kernels (differential comparison).
    setenv("XSAI_MNN_AME_MIN_MACS", "1", 1);
    // bf16 gate is separate: it changes float precision (see header notes)
    setenv("XSAI_NCNN_AME_BF16", ame ? "1" : "0", 1);

    static const Case cases[] = {
        // K around the 64-byte tile boundary, bias on/off, dequant + requant
        {7, 7, 16, 32, 3, 1, 1, 1, 1, 1},     // K=144
        {9, 9, 8, 8, 3, 1, 2, 1, 1, 1},       // K=72, stride 2
        {14, 14, 64, 64, 3, 1, 1, 1, 1, 1},   // K=576 M=144 (multi M-tile @64)
        {6, 6, 7, 5, 3, 2, 1, 2, 0, 1},       // K=63 (partial k-block)
        {5, 5, 3, 129, 3, 1, 1, 1, 1, 1},     // N=129 (multi N-tile)
        {8, 8, 32, 32, 1, 1, 1, 0, 1, 1},     // 1x1, K=32
        {7, 7, 16, 32, 3, 1, 1, 1, 1, 101},   // requant int8 out
        {12, 12, 48, 48, 3, 1, 1, 1, 0, 101}, // requant, K=432
        {10, 10, 16, 16, 5, 1, 2, 2, 1, 1},   // K=400 5x5 stride2
    };

    int ret = 0;
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < n && ret == 0; i++)
    {
        fprintf(stderr, "[smoke] case %d/%d\n", i + 1, n);
        ret = run_case(cases[i], out);
    }

    static const Bf16Case bf16_cases[] = {
        {7, 7, 16, 32, 3, 1, 1, 1, 1},     // K=144
        {9, 9, 8, 8, 3, 1, 2, 1, 1},       // K=72 stride 2
        {14, 14, 64, 64, 3, 1, 1, 1, 1},   // K=576 M=196 (multi M-tile @128)
        {6, 6, 7, 5, 3, 2, 1, 2, 0},       // K=63 partial k-block, dilation 2
        {5, 5, 3, 129, 3, 1, 1, 1, 1},     // N=129 multi N-tile
        {8, 8, 32, 32, 1, 1, 1, 0, 1},     // 1x1 K=32
        {12, 12, 48, 48, 3, 1, 1, 1, 0},   // K=432
        {10, 10, 16, 16, 5, 1, 2, 2, 1},   // 5x5 stride 2 K=400
    };
    const int bn = (int)(sizeof(bf16_cases) / sizeof(bf16_cases[0]));
    for (int i = 0; i < bn && ret == 0; i++)
    {
        fprintf(stderr, "[smoke] bf16 case %d/%d\n", i + 1, bn);
        ret = run_bf16_case(bf16_cases[i], bf16_out);
    }
    if (bf16_out != out)
        fclose(bf16_out);
    fclose(out);
    fprintf(stderr, "[smoke] mode=%s ret=%d\n", ame ? "ame" : "ref", ret);
    return ret;
}
