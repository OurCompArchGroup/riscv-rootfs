#!/usr/bin/env bash
# Lightweight wrapper regression; no framework compilation, fetch, or cleanup.
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
app_dir=$(cd -- "$test_dir/.." && pwd)
rootfs_dir=$(cd -- "$app_dir/../.." && pwd)
workspace=${XSAI_ENV_ROOT:-$(cd -- "$app_dir/../../../.." && pwd)}
scratch=$(mktemp -d "${TMPDIR:-/tmp}/xsai-wrapper-test.XXXXXX")
printf 'Test artifacts (retained): %s\n' "$scratch"

# Isolate make from caller overrides and source trees. Seed local repo copies
# while leaving canonical directories absent: implicit fallback must NOT occur.
unset MNN_SRC MNN_HOME NCNN_SRC NCNN_HOME LLAMA_CPP_SRC LLAMA_CPP_HOME
unset XSAI_ALLOC_SRC XSAI_ALLOC_HEADER MAKEFLAGS MFLAGS MAKEOVERRIDES
unset MOBILENET_V2_ONNX_URL MOBILENET_V2_ONNX_SHA256 ONNX_FILE
unset XSAI_ENV_ROOT
mkdir -p "$scratch/workspace/mk"
cp "$workspace/mk/memory.mk" "$scratch/workspace/mk/"
for app in mnn ncnn llama.cpp; do
    wrapper="$scratch/workspace/firmware/riscv-rootfs/apps/$app"
    mkdir -p "$wrapper/repo"
    cp "$rootfs_dir/apps/$app/Makefile" "$wrapper/"
    cp "$test_dir/allocator-fixture/CMakeLists.txt" "$wrapper/repo/"
done

expect_failure() {
    local pattern=$1
    shift
    if "$@" >"$scratch/failure.log" 2>&1; then
        echo "FAIL: expected rejection: $*" >&2
        exit 1
    fi
    if ! grep -Fq "$pattern" "$scratch/failure.log"; then
        echo "FAIL: expected diagnostic '$pattern'" >&2
        sed -n '1,100p' "$scratch/failure.log" >&2
        exit 1
    fi
}

fake_apps="$scratch/workspace/firmware/riscv-rootfs/apps"
for goal in source-ready repo; do
    expect_failure 'set MNN_SRC=' make -s -C "$fake_apps/mnn" "$goal"
    expect_failure 'set NCNN_SRC=' make -s -C "$fake_apps/ncnn" "$goal" MNN_SRC="$workspace/software/MNN"
    expect_failure 'set LLAMA_CPP_SRC=' make -s -C "$fake_apps/llama.cpp" "$goal" MNN_SRC="$workspace/software/MNN"
done

# Existing HOME aliases still work, SRC is authoritative when both are supplied.
make -s -C "$fake_apps/mnn" source-ready MNN_HOME="$workspace/software/MNN"
make -s -C "$fake_apps/mnn" source-ready MNN_HOME="$scratch/missing" MNN_SRC="$workspace/software/MNN"
make -s -C "$fake_apps/ncnn" source-ready NCNN_HOME=repo MNN_HOME="$workspace/software/MNN"
make -s -C "$fake_apps/ncnn" source-ready NCNN_HOME="$scratch/missing" NCNN_SRC=repo MNN_SRC="$workspace/software/MNN"
make -s -C "$fake_apps/llama.cpp" source-ready LLAMA_CPP_HOME=repo MNN_HOME="$workspace/software/MNN"
make -s -C "$fake_apps/llama.cpp" source-ready LLAMA_CPP_HOME="$scratch/missing" LLAMA_CPP_SRC=repo MNN_SRC="$workspace/software/MNN"
expect_failure 'requires the MNN' make -s -C "$fake_apps/ncnn" source-ready NCNN_SRC=repo MNN_SRC="$scratch/missing"
expect_failure 'requires the MNN' make -s -C "$fake_apps/llama.cpp" source-ready LLAMA_CPP_SRC=repo MNN_SRC="$scratch/missing"
expect_failure 'requires the MNN' make -s -C "$fake_apps/ncnn" ame-lib NCNN_SRC=repo MNN_SRC="$scratch/missing"
expect_failure 'AME LLVM tool missing' make -s -C "$fake_apps/ncnn" ame-lib NCNN_SRC=repo MNN_SRC="$workspace/software/MNN" LLVM_HOME="$scratch/missing"
cp "$rootfs_dir/apps/mnn/nemu-signal.c" "$fake_apps/mnn/"
expect_failure 'RISC-V GCC not found' make -s -C "$fake_apps/mnn" "$fake_apps/mnn/build-riscv/nemu-signal" CROSS_GCC="$scratch/missing-gcc"

# Model-source checks do not invoke conversion tools or contact the network.
printf 'ONNX checksum fixture\n' > "$scratch/model.onnx"
model_hash=$(sha256sum "$scratch/model.onnx" | cut -d' ' -f1)
make -s -C "$fake_apps/mnn" model-source-ready ONNX_FILE="$scratch/model.onnx" MOBILENET_V2_ONNX_SHA256="$model_hash"
expect_failure 'MOBILENET_V2_ONNX_SHA256 is required' make -s -C "$fake_apps/mnn" model-source-ready ONNX_FILE="$scratch/model.onnx"
expect_failure 'FAILED' make -s -C "$fake_apps/mnn" model-source-ready ONNX_FILE="$scratch/model.onnx" MOBILENET_V2_ONNX_SHA256=0000000000000000000000000000000000000000000000000000000000000000
expect_failure 'supply ONNX_FILE' make -s -C "$fake_apps/mnn" model-source-ready ONNX_FILE="$scratch/absent.onnx"

# Absolute canonical default is stable even though the local repo exists.
make -s -C "$fake_apps/llama.cpp" print-config >"$scratch/config.log"
grep -Fxq "LLAMA_CPP_SRC=$scratch/workspace/software/llama.cpp" "$scratch/config.log"

allocator="$workspace/software/MNN/source/backend/cpu/ame"
fixture="$test_dir/allocator-fixture"
before=$(sha256sum "$fixture/ggml-cpu/xsai_alloc.c")
cmake -S "$fixture" -B "$scratch/cmake-ok" \
    -DCMAKE_PROJECT_INCLUDE="$app_dir/cmake/use-mnn-allocator.cmake" \
    -DXSAI_ALLOC_SRC="$allocator/xsai_alloc.c" \
    -DXSAI_ALLOC_HEADER="$allocator/xsai_alloc.h" >"$scratch/cmake-ok.log" 2>&1
test "$before" = "$(sha256sum "$fixture/ggml-cpu/xsai_alloc.c")"
expect_failure 'No ggml-cpu xsai_alloc.c source found' cmake -S "$fixture" -B "$scratch/cmake-bad" \
    -DCMAKE_PROJECT_INCLUDE="$app_dir/cmake/use-mnn-allocator.cmake" \
    -DXSAI_ALLOC_SRC="$allocator/xsai_alloc.c" \
    -DXSAI_ALLOC_HEADER="$allocator/xsai_alloc.h" -DFIXTURE_MISSING_CONTRACT=ON
expect_failure 'must name an existing absolute MNN allocator file' cmake -S "$fixture" -B "$scratch/cmake-missing" \
    -DCMAKE_PROJECT_INCLUDE="$app_dir/cmake/use-mnn-allocator.cmake" \
    -DXSAI_ALLOC_SRC="$scratch/missing.c" -DXSAI_ALLOC_HEADER="$allocator/xsai_alloc.h"
printf 'PASS: source selection, aliases, dependency errors, allocator graph and no source writes\n'
