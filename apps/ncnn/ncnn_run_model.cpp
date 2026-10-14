// ncnn_run_model.cpp — minimal end-to-end model runner for the XSAI AME port.
//
// Loads an ncnn .param/.bin pair, feeds a deterministic input tensor from a
// text file (one float per line, CHW — the same format MNNV2Basic consumes,
// produced by apps/mnn/scripts/gen-input.py), runs the forward pass and
// prints argmax + top-5. Used for the cross-framework check (same ONNX
// weights, same input floats → same argmax as the MNN reference flow) and
// the AME differential (env on/off → same argmax).
//
// usage: ncnn_run_model <model.param> <model.bin> <input.txt> [w h c]
#include "net.h"
#include "mat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

static uint64_t rdcycle()
{
#if defined(__riscv)
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0; // host debug build: no cycle counter, prints 0
#endif
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <model.param> <model.bin> <input.txt> [w h c]\n", argv[0]);
        return 2;
    }
    const int w = argc > 5 ? atoi(argv[4]) : 224;
    const int h = argc > 6 ? atoi(argv[5]) : 224;
    const int c = argc > 7 ? atoi(argv[6]) : 3;

    ncnn::Net net;
    // keep the AME-eligible paths plain: pack1 fp32/int8, no winograd/sgemm
    // dispatch surprises; the AME branches gate themselves by env.
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_sgemm_convolution = false;
    net.opt.use_winograd_convolution = false;
    net.opt.use_int8_inference = true; // honoured only for int8 weights

    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
    {
        fprintf(stderr, "load model failed\n");
        return 2;
    }

    ncnn::Mat in(w, h, c);
    {
        FILE* f = fopen(argv[3], "r");
        if (!f)
        {
            fprintf(stderr, "open input %s failed\n", argv[3]);
            return 2;
        }
        // the MNN flow's input_0.txt is HWC-interleaved (RGB per pixel,
        // [0,1] domain; the ONNX-zoo graph carries its own mean/std layers),
        // so reshuffle into ncnn's CHW planar layout here.
        std::vector<float> buf((size_t)w * h * c);
        for (size_t i = 0; i < buf.size(); i++)
        {
            if (fscanf(f, "%f", &buf[i]) != 1)
            {
                fprintf(stderr, "input file too short at %zu\n", i);
                fclose(f);
                return 2;
            }
        }
        fclose(f);
        for (int q = 0; q < c; q++)
        {
            float* plane = in.channel(q);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    plane[y * w + x] = buf[((size_t)y * w + x) * c + q];
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(0, in) != 0)
    {
        fprintf(stderr, "set input failed\n");
        return 2;
    }
    // optional intermediate-blob probe: NCNN_E2E_DUMP_BLOB=<name>
    if (const char* probe = getenv("NCNN_E2E_DUMP_BLOB"))
    {
        ncnn::Mat mid;
        if (ex.extract(probe, mid) == 0)
        {
            double sum = 0, absmax = 0;
            const float* mp = mid;
            const size_t mn = mid.total();
            for (size_t i = 0; i < mn; i++)
            {
                sum += mp[i];
                const double a = fabs((double)mp[i]);
                if (a > absmax) absmax = a;
            }
            fprintf(stderr, "[ncnn-e2e] blob %s: %dx%dx%d sum=%.4f mean=%.6f absmax=%.4f [0]=%.6f [1]=%.6f\n",
                    probe, mid.w, mid.h, mid.c, sum, sum / (double)mn, absmax, (double)mp[0], (double)mp[1]);
        }
        else
        {
            fprintf(stderr, "[ncnn-e2e] blob %s not found\n", probe);
        }
    }
    ncnn::Mat out;
    const uint64_t c0 = rdcycle();
    // the output blob is the graph sink; prefer the canonical name "output",
    // fall back to the Net's last-registered blob
    int ret_ex = ex.extract("output", out);
    if (ret_ex != 0 && net.blobs().size() > 0)
        ret_ex = ex.extract(net.blobs()[net.blobs().size() - 1].name.c_str(), out);
    if (ret_ex != 0)
    {
        fprintf(stderr, "extract failed\n");
        return 2;
    }
    const uint64_t c1 = rdcycle();
    fprintf(stderr, "[ncnn-e2e] forward cycles=%llu out=%dx%dx%d elemsize=%zu\n",
            (unsigned long long)(c1 - c0), out.w, out.h, out.c, out.elemsize);

    // argmax over the output vector (softmax head may be absent; raw logits fine)
    const int n = out.w * out.h * out.c;
    std::vector<std::pair<float, int>> scores(n);
    const float* op = out;
    for (int i = 0; i < n; i++)
        scores[i] = std::make_pair(op[i], i);
    const int topn = n < 5 ? n : 5;
    std::partial_sort(scores.begin(), scores.begin() + topn, scores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });
    printf("argmax=%d score=%.6f\n", scores[0].second, scores[0].first);
    fprintf(stderr, "[ncnn-e2e] top5:");
    for (int i = 0; i < topn; i++)
        fprintf(stderr, " %d:%.4f", scores[i].second, scores[i].first);
    fprintf(stderr, "\n");
    return 0;
}
