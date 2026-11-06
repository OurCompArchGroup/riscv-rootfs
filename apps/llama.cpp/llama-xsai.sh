#!/bin/sh

set -eu

runner=${LLAMA_RUNNER:-/bin/llama-simple}
model=${LLAMA_MODEL:-/root/llama-model.gguf}
prompt=${LLAMA_PROMPT:-Once}
model_set=0

xsai_mode=${XSAI_GGML_BACKEND:-1}
q8_mode=${XSAI_GGML_Q8:-1}
diagnose=0

usage() {
  cat <<'USAGE'
usage: llama-xsai [--model PATH] [--prompt TEXT] [--xsai on|off] [--q8 on|off] [--diagnose] [MODEL.gguf] [PROMPT...]

  --xsai on     XSAI_GGML_BACKEND=1 (default). Registered ops use xsai-blas plans.
  --xsai off    XSAI_GGML_BACKEND=0. Every op stays on the ggml CPU path.
  --q8 on       XSAI_GGML_Q8=1 (default). Q8_0 MUL_MAT is placed on the xsai
                buffer and dispatched to xsai-blas GEMM.
  --q8 off      XSAI_GGML_Q8=0. Q8_0 MUL_MAT stays with AME/CPU even when the
                backend is on; useful for A/B runs.
  --diagnose    Print the dispatch mode before exec. The backend writes route
                counts, plan-create, key-miss, execute, and prepared bytes to stderr.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      model=$2
      model_set=1
      shift 2
      ;;
    --prompt)
      prompt=$2
      shift 2
      ;;
    --xsai)
      case "$2" in
        on|1) xsai_mode=1 ;;
        off|0) xsai_mode=0 ;;
        *)
          echo "llama-xsai: --xsai expects on or off" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --q8)
      case "$2" in
        on|1) q8_mode=1 ;;
        off|0) q8_mode=0 ;;
        *)
          echo "llama-xsai: --q8 expects on or off" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --diagnose)
      diagnose=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -gt 0 ] && [ "$model_set" -eq 0 ] && [ -f "$1" ]; then
  model=$1
  shift
fi

if [ "$#" -gt 0 ]; then
  prompt="$*"
fi

if [ ! -f "$model" ]; then
  echo "llama-xsai: model not found: $model" >&2
  exit 1
fi

export XSAI_GGML_BACKEND="$xsai_mode"
export XSAI_GGML_Q8="$q8_mode"
if [ "$xsai_mode" = "0" ]; then
  echo "llama-xsai: dispatch=cpu-fallback XSAI_GGML_BACKEND=0" >&2
elif [ "$q8_mode" = "0" ]; then
  echo "llama-xsai: dispatch=xsai XSAI_GGML_BACKEND=1 XSAI_GGML_Q8=0 (Q8 matmul stays on AME/CPU)" >&2
else
  echo "llama-xsai: dispatch=xsai XSAI_GGML_BACKEND=1 XSAI_GGML_Q8=1" >&2
fi
if [ "$diagnose" -eq 1 ]; then
  echo "llama-xsai: diagnostic=stderr route counts, plan_create, key_miss, execute, prepared_bytes" >&2
fi

exec "$runner" -m "$model" "$prompt"
