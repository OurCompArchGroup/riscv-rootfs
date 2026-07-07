#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_REPO="$SCRIPT_DIR/repo"
XSAI_ENV_ROOT="${XSAI_ENV_ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"

BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build-riscv}"
QEMU_USER="${QEMU_USER:-$XSAI_ENV_ROOT/qemu/build/qemu-riscv64}"
QEMU_CPU="${QEMU_CPU:-rv64,v=true,vlen=128,h=true,zvfh=true,zvfhmin=true,x-matrix=true,rlen=512,mlen=65536,melen=32}"
AME_SHAPES="${AME_SHAPES:-512,128,512 288,128,288}"
BUFT="${BUFT:-RISCV_AME}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ -f "$XSAI_ENV_ROOT/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$XSAI_ENV_ROOT/env.sh"
fi

if [[ -z "${RISCV_SYSROOT:-}" ]]; then
    echo "error: RISCV_SYSROOT is not set; source xsai-env env.sh or set RISCV_SYSROOT" >&2
    exit 1
fi

if [[ ! -x "$QEMU_USER" ]]; then
    echo "error: qemu-riscv64 not found at $QEMU_USER" >&2
    exit 1
fi

TEST_BACKEND_OPS="$BUILD_DIR/bin/test-backend-ops"
if [[ "$SKIP_BUILD" != "1" ]]; then
    cmake --build "$BUILD_DIR" --target test-backend-ops -j"$(nproc)"
fi

if [[ ! -x "$TEST_BACKEND_OPS" ]]; then
    echo "error: test-backend-ops not found at $TEST_BACKEND_OPS" >&2
    exit 1
fi

for shape in $AME_SHAPES; do
    IFS=',' read -r ame_m ame_n ame_k extra <<< "$shape"
    if [[ -z "$ame_m" || -z "$ame_n" || -z "$ame_k" || -n "${extra:-}" ]]; then
        echo "error: invalid AME_SHAPES entry '$shape'; expected M,N,K" >&2
        exit 1
    fi

    shape_filter="type_a=q8_0,type_b=f32,m=${ame_m},n=${ame_n},k=${ame_k},bs=\\[1,1\\],nr=\\[1,1\\]"
    echo "[xsai-ame-verify] correctness: MUL_MAT Q8_0/F32 m=$ame_m n=$ame_n k=$ame_k buft=$BUFT"
    XSAI_ALLOC_MODE=malloc "$QEMU_USER" \
        -cpu "$QEMU_CPU" \
        -L "$RISCV_SYSROOT" \
        "$TEST_BACKEND_OPS" \
        test -o MUL_MAT --buft "$BUFT" -p "$shape_filter"
done
