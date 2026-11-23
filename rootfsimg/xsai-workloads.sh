# XSAI guest workload dispatcher (slim).
#
# Stable workloads:
#   test-backend-ops       AME correctness gate: Q8_0 MUL_MAT, CPU softmax,
#                          BF16 whole-K MUL_MAT (fixed path, always on).
#   llama-bf16-test-ops    small BF16 smoke: CPY, 64x64x32 MUL_MAT, softmax.
#   llama-fake-bench-suite default ci workload: gate + fake-like llama-bench.
#   llama-bench / llama-simple(-xsai) / llama-xsai
#   llama-bench-full      historical three-model fake-like suite
#   llama-bench-real      opt-in real GGUF, all layers, warmup + repetitions
#   llama-mxfp4-runtime   CPU runtime/reference quantization-byte gate
#   llama-f16-attention   independent F16 attention numerical gate
#   mnn-smoke / ncnn-smoke (opt-in app + additive manifest)
#   hello_xsai / gemm_precomp / pmu_test
#
# Removed 2026-09-01: one-off experiment harnesses (BF16 whole-K/TR sweeps,
# Q8 output-pipeline preflights, F16-via-BF16 batch probe, 15M infer, AME
# large matmul, panel probe, FlashAttention sweep/ckpt variants) and their
# local test switches.  Whole-K BF16 is the fixed AME path now.

run_and_capture()
{
  if run_workload_cmd "$@"; then
    WORKLOAD_STATUS=0
  else
    WORKLOAD_STATUS=$?
  fi
}

run_llama_bench()
{
  run_workload_cmd /bin/llama-bench \
    -m "$LLAMA_MODEL" \
    -t 1 -p 512 -n 0 \
    --no-warmup --estimate-layers 1
}

run_llama_bench_full()
(
  # Preserve the original canonical full suite. "full" historically names the
  # model suite, not real weights or complete-layer execution.
  if [ "${WORKLOAD:-llama-bench-full}" = llama-bench-full ]; then
    # Optional program argument lets an explicit same-boot comparison reuse
    # this handler's complete fake-like argv and status contract.
    full_program="${1:-/bin/llama-bench}"
    if [ ! -x "$full_program" ]; then
      echo "[xsai-workload] LLAMA_BENCH_FULL_FAIL missing=$full_program"
      return 127
    fi
    if [ "${XSAI_LLAMA_FULL_PREFLIGHT:-0}" = 1 ]; then
      run_llama_test_backend_ops || return "$?"
    fi
    models="/root/stories15M-q8_0.gguf.om
/root/qwen3_0.6b_q8.gguf.om
/root/llama3.2_1b_q8.gguf.om"
    if [ -n "${XSAI_LLAMA_FAKE_BENCH_MODELS:-}" ]; then
      models="$XSAI_LLAMA_FAKE_BENCH_MODELS"
    fi
    full_threads="${XSAI_LLAMA_FAKE_BENCH_T:-1}"
    full_prompt="${XSAI_LLAMA_FAKE_BENCH_P:-256}"
    full_batch="${XSAI_LLAMA_FAKE_BENCH_B:-}"
    full_ubatch="${XSAI_LLAMA_FAKE_BENCH_UB:-}"
    full_decode="${XSAI_LLAMA_FAKE_BENCH_N:-8}"
    full_repetitions="${XSAI_LLAMA_FAKE_BENCH_R:-1}"
    full_warmup="${XSAI_LLAMA_FAKE_BENCH_WARMUP:-0}"
    full_estimate="${XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS:-1}"
    full_fa="${XSAI_LLAMA_FAKE_BENCH_FA:-0}"
    full_type_k="${XSAI_LLAMA_FAKE_BENCH_TYPE_K:-f16}"
    full_type_v="${XSAI_LLAMA_FAKE_BENCH_TYPE_V:-f16}"
    full_output_err="${XSAI_LLAMA_FAKE_BENCH_OUTPUT_ERR:-none}"
    if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
      set -x
    fi
    full_status=0
    for model in $models; do
      echo "[xsai-workload] llama-bench full: $model"
      if (
        export GGML_AME_PACKED_Q8=1
        export GGML_AME_USE_PACKED_B_PANEL=1
        export GGML_AME_WHOLE_K_Q8=1
        export GGML_AME_WHOLE_K_FUSED_EPILOGUE=1
        export GGML_AME_F16_BF16=1
        export GGML_AME_WHOLE_K_BF16=1
        export GGML_AME_BF16_REG_PAIRS=2
        set -- -t "$full_threads" -p "$full_prompt" -n "$full_decode" \
          -r "$full_repetitions" -fa "$full_fa" \
          -ctk "$full_type_k" -ctv "$full_type_v"
        if [ -n "$full_batch" ]; then
          set -- "$@" -b "$full_batch"
        fi
        if [ -n "$full_ubatch" ]; then
          set -- "$@" -ub "$full_ubatch"
        fi
        if [ "$full_warmup" = 0 ]; then
          set -- "$@" --no-warmup
        fi
        if [ "$full_output_err" != none ]; then
          set -- "$@" -oe "$full_output_err"
        fi
        run_workload_cmd "$full_program" --fake-like "$model" \
          "$@" --estimate-layers "$full_estimate"
      ); then
        :
      else
        rc=$?
        if [ "$full_status" = 0 ]; then
          full_status=$rc
        fi
      fi
    done
    if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
      set +x
    fi
    if [ "$full_status" = 0 ]; then
      echo "[xsai-workload] LLAMA_BENCH_FULL_PASS"
      return 0
    fi
    echo "[xsai-workload] LLAMA_BENCH_FULL_FAIL rc=$full_status"
    return "$full_status"
  fi

  # Explicit real-model branch of the same handler, with isolated controls.
  full_model="${XSAI_LLAMA_FULL_MODEL:-$LLAMA_MODEL}"
  full_threads="${XSAI_LLAMA_FULL_T:-1}"
  full_prompt="${XSAI_LLAMA_FULL_P:-512}"
  # Match llama-bench's existing defaults when no batch override is requested.
  full_batch="${XSAI_LLAMA_FULL_B:-2048}"
  full_ubatch="${XSAI_LLAMA_FULL_UB:-512}"
  full_decode="${XSAI_LLAMA_FULL_N:-0}"
  full_repetitions="${XSAI_LLAMA_FULL_R:-3}"
  full_warmup="${XSAI_LLAMA_FULL_WARMUP:-1}"
  full_progress="${XSAI_LLAMA_FULL_PROGRESS:-0}"
  full_fa="${XSAI_LLAMA_FULL_FA:-0}"
  full_type_k="${XSAI_LLAMA_FULL_TYPE_K:-f16}"
  full_type_v="${XSAI_LLAMA_FULL_TYPE_V:-f16}"

  # llama-bench itself splits -m on commas; this selector promises one model.
  case "$full_model" in
    *,*)
      echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL one model path required"
      return 2
      ;;
  esac
  case "$full_model" in
    *.gguf) ;;
    *)
      echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL real .gguf required: $full_model"
      return 2
      ;;
  esac
  if [ ! -f "$full_model" ] || [ ! -r "$full_model" ] || [ ! -s "$full_model" ]; then
    echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL missing/empty model: $full_model"
    return 1
  fi
  # Scalar counts make one requested prefill/decode pair unambiguous and stop
  # zero repetitions or zero matched tests from masquerading as completion.
  for full_count in "$full_threads" "$full_prompt" "$full_decode" "$full_repetitions" \
      "$full_batch" "$full_ubatch"; do
    case "$full_count" in
      ''|*[!0-9]*)
        echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL invalid count: $full_count"
        return 2
        ;;
    esac
    if ! [ "$full_count" -le 2147483647 ] 2>/dev/null; then
      echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL count out of range: $full_count"
      return 2
    fi
  done
  if [ "$full_threads" -eq 0 ] || [ "$full_repetitions" -eq 0 ] ||
     [ "$full_batch" -eq 0 ] || [ "$full_ubatch" -eq 0 ] ||
     { [ "$full_prompt" -eq 0 ] && [ "$full_decode" -eq 0 ]; }; then
    echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL empty workload"
    return 2
  fi
  case "$full_warmup:$full_fa:${XSAI_LLAMA_FULL_PREFLIGHT:-0}:$full_progress" in
    [01]:[01]:[01]:[01]) ;;
    *)
      echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL invalid warmup/fa/preflight/progress"
      return 2
      ;;
  esac
  if [ "${XSAI_LLAMA_FULL_PREFLIGHT:-0}" = 1 ]; then
    run_llama_test_backend_ops || return "$?"
  fi

  # Preserve explicit caller overrides and the canonical allocator. Real-model
  # measurements default to the original K32 Q8_0 path, not requantized weights
  # or F16->BF16 approximation; optimized variants must be selected explicitly.
  export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-0}"
  export GGML_AME_USE_PACKED_B_PANEL="${GGML_AME_USE_PACKED_B_PANEL:-0}"
  export GGML_AME_WHOLE_K_Q8="${GGML_AME_WHOLE_K_Q8:-0}"
  export GGML_AME_WHOLE_K_FUSED_EPILOGUE="${GGML_AME_WHOLE_K_FUSED_EPILOGUE:-0}"
  export GGML_AME_F16_BF16="${GGML_AME_F16_BF16:-0}"
  export GGML_AME_WHOLE_K_BF16="${GGML_AME_WHOLE_K_BF16:-0}"
  export GGML_AME_BF16_REG_PAIRS="${GGML_AME_BF16_REG_PAIRS:-2}"
  export GGML_AME_PROFILE="${GGML_AME_PROFILE:-0}"
  export GGML_XSAI_OP_PROFILE="${GGML_XSAI_OP_PROFILE:-0}"
  export GGML_AME_LOG="${GGML_AME_LOG:-0}"
  export GGML_AME_PANEL_LOG="${GGML_AME_PANEL_LOG:-0}"
  export GGML_AME_PANEL_PROGRESS_LOG="${GGML_AME_PANEL_PROGRESS_LOG:-0}"
  export GGML_AME_TILE_PROGRESS_LOG="${GGML_AME_TILE_PROGRESS_LOG:-0}"
  export GGML_XSAI_OP_PROGRESS_LOG="${GGML_XSAI_OP_PROGRESS_LOG:-0}"
  set -- -m "$full_model" -t "$full_threads" -p "$full_prompt" -n "$full_decode" \
    -r "$full_repetitions" -fa "$full_fa" -ctk "$full_type_k" -ctv "$full_type_v" \
    -b "$full_batch" -ub "$full_ubatch" -oe jsonl
  if [ "$full_warmup" = 0 ]; then
    set -- "$@" --no-warmup
  fi
  if [ "$full_progress" = 1 ]; then
    # llama-bench prints per-sample progress inside its cycle bracket.
    set -- "$@" --progress
  fi
  echo "[xsai-workload] LLAMA_BENCH_REAL_BEGIN model=$full_model t=$full_threads p=$full_prompt b=$full_batch ub=$full_ubatch n=$full_decode rep=$full_repetitions warmup=$full_warmup diagnostic_progress=$full_progress packed_q8=$GGML_AME_PACKED_Q8 whole_k_q8=$GGML_AME_WHOLE_K_Q8 f16_bf16=$GGML_AME_F16_BF16 ame_profile=$GGML_AME_PROFILE op_profile=$GGML_XSAI_OP_PROFILE real_weights=1 all_layers=1"
  if run_workload_cmd /bin/llama-bench "$@"; then
    echo "[xsai-workload] LLAMA_BENCH_REAL_PASS"
    return 0
  else
    full_status=$?
    echo "[xsai-workload] LLAMA_BENCH_REAL_FAIL rc=$full_status"
    return "$full_status"
  fi
)

run_llama_simple()
{
  export GGML_AME_LOG=0
  export GGML_AME_PACKED_Q8=1
  run_workload_cmd /bin/llama-simple-xsai -m "$LLAMA_MODEL" "$LLAMA_PROMPT"
}

run_llama_xsai()
{
  run_workload_cmd /bin/llama-xsai --model "$LLAMA_MODEL" --prompt "$LLAMA_PROMPT"
}

run_llama_test_backend_ops()
{
  if [ "${XSAI_LLAMA_TEST_BACKEND_OPS:-1}" = 0 ]; then
    echo "[xsai-workload] test-backend-ops disabled"
    return 0
  fi

  echo "[xsai-workload] test-backend-ops AME MUL_MAT Q8_0/F32"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_AME_PACKED_Q8=1
    export GGML_AME_PROFILE=0
    export GGML_XSAI_OP_PROFILE=0
    export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL=0
    run_workload_cmd /bin/test-backend-ops \
      test -o MUL_MAT --buft RISCV_AME \
      -p 'type_a=q8_0,type_b=f32,m=(288|768),n=128,k=(288|768),bs=\[1,1\],nr=\[1,1\]'
  ) || return $?

  echo "[xsai-workload] test-backend-ops CPU SOFT_MAX F16 mask"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_AME_PACKED_Q8=1
    export GGML_AME_PROFILE=0
    export GGML_XSAI_OP_PROFILE=0
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU -o SOFT_MAX \
      -p 'type=f32,ne=\[32,2,32,1\],mask=1,sinks=0,m_prec=f16,nr23=\[1,1\],scale=0.100000,max_bias=0.000000,inplace=0'
  ) || return $?

  # BF16 whole-K correctness: fixed path, no longer behind a gate.
  echo "[xsai-workload] test-backend-ops AME BF16 whole-K"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_AME_F16_BF16=1
    export GGML_AME_F16_BF16_REQUIRE_AME_BUFFER=1
    export GGML_AME_WHOLE_K_BF16=1
    export GGML_AME_BF16_REG_PAIRS=2
    export GGML_AME_PROFILE=0
    export GGML_AME_BF16_PROFILE=0
    export GGML_XSAI_OP_PROFILE=0
    export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL=0
    export XSAI_TEST_BACKEND_OPS_PRINT_ERROR=1
    for shape_filter in \
      'type_a=bf16,type_b=f32,m=128,n=128,k=256,bs=\[1,1\],nr=\[1,1\]' \
      'type_a=bf16,type_b=f32,m=256,n=128,k=128,bs=\[1,1\],nr=\[1,1\]' \
      'type_a=f16,type_b=f32,m=256,n=128,k=128,bs=\[2,1\],nr=\[8,1\],per=\[0,2,1,3\]'
    do
      echo "[xsai-workload] BF16_WHOLE_K_CORRECTNESS shape=$shape_filter"
      run_workload_cmd /bin/test-backend-ops \
        test -b CPU --buft RISCV_AME -o MUL_MAT -p "$shape_filter" || exit $?
    done
  ) || return $?
}

run_llama_bf16_test_ops()
{
  echo "[xsai-workload] BF16_TESTS_BEGIN"

  if [ ! -x /bin/test-backend-ops ]; then
    echo "[xsai-workload] BF16_TESTS_FAIL missing /bin/test-backend-ops"
    return 127
  fi

  echo "[xsai-workload] BF16 CPY bf16 -> f32/bf16"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_XSAI_OP_PROFILE=0
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU -o CPY \
      -p 'type_src=bf16,type_dst=(f32|bf16),ne=\[256,4,4,4\],permute_src=\[0,0,0,0\],permute_dst=\[0,0,0,0\],_src_transpose=0'
  ) || {
    rc=$?
    echo "[xsai-workload] BF16_TESTS_FAIL stage=CPY rc=$rc"
    return "$rc"
  }

  echo "[xsai-workload] BF16 MUL_MAT 64x64x32"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_XSAI_OP_PROFILE=0
    export GGML_AME_WHOLE_K_BF16=1
    export GGML_AME_BF16_REG_PAIRS=2
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU --buft RISCV_AME -o MUL_MAT \
      -p 'type_a=bf16,type_b=(f32|bf16),m=64,n=64,k=32,bs=\[1,1\],nr=\[1,1\]'
  ) || {
    rc=$?
    echo "[xsai-workload] BF16_TESTS_FAIL stage=MUL_MAT rc=$rc"
    return "$rc"
  }

  echo "[xsai-workload] SOFT_MAX regression 32x2x32"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_XSAI_OP_PROFILE=0
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU -o SOFT_MAX \
      -p 'type=f32,ne=\[32,2,32,1\],mask=1,sinks=0,m_prec=f16,nr23=\[1,1\],scale=0.100000,max_bias=0.000000,inplace=0'
  ) || {
    rc=$?
    echo "[xsai-workload] BF16_TESTS_FAIL stage=SOFT_MAX rc=$rc"
    return "$rc"
  }

  echo "[xsai-workload] BF16_TESTS_PASS"
}

run_llama_fake_bench_suite()
{
  status=0
  # Keep the benchmark focused on the requested prefill/decode shapes.
  # Set XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS=0 to execute every model layer.
  bench_prompt="${XSAI_LLAMA_FAKE_BENCH_P:-128}"
  bench_decode="${XSAI_LLAMA_FAKE_BENCH_N:-0}"
  bench_repetitions="${XSAI_LLAMA_FAKE_BENCH_R:-1}"
  bench_warmup="${XSAI_LLAMA_FAKE_BENCH_WARMUP:-0}"
  bench_estimate_layers="${XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS:-1}"
  bench_fa="${XSAI_LLAMA_FAKE_BENCH_FA:-0}"
  bench_type_k="${XSAI_LLAMA_FAKE_BENCH_TYPE_K:-f16}"
  bench_type_v="${XSAI_LLAMA_FAKE_BENCH_TYPE_V:-f16}"
  bench_args="-t 1 -p $bench_prompt -n $bench_decode -r $bench_repetitions -fa $bench_fa -ctk $bench_type_k -ctv $bench_type_v"
  if [ "$bench_warmup" = 0 ]; then
    bench_args="$bench_args --no-warmup"
  fi
  if [ "$bench_estimate_layers" != 0 ]; then
    bench_args="$bench_args --estimate-layers $bench_estimate_layers"
  fi

  echo "[xsai-workload] llama-bench attention fa=$bench_fa type_k=$bench_type_k type_v=$bench_type_v"

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set -x
  fi

  if run_llama_test_backend_ops; then
    :
  else
    rc=$?
    if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
      set +x
    fi
    return "$rc"
  fi

  models="/root/stories15M-q8_0.gguf.om
/root/llama3.2_1b_q8.gguf.om"
  if [ -n "${XSAI_LLAMA_FAKE_BENCH_MODELS:-}" ]; then
    models="$XSAI_LLAMA_FAKE_BENCH_MODELS"
  fi

  for model in $models
  do
    echo "[xsai-workload] llama-bench AME profile: $model"
    if (
      export GGML_AME_LOG="${GGML_AME_LOG:-0}"
      export GGML_AME_PACKED_Q8=1
      export GGML_AME_PROFILE=1
      export GGML_XSAI_OP_PROFILE=1
      export LLAMA_BENCH_PRINT_LAYERS_01=0
      run_workload_cmd /bin/llama-bench --fake-like "$model" $bench_args
    ); then
      :
    else
      rc=$?
      if [ "$status" = 0 ]; then
        status=$rc
      fi
    fi
  done

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set +x
  fi

  return "$status"
}

run_mnn_smoke()
(
  for input in /bin/MNNV2Basic.out /root/mobilenet-v2-int8.mnn /root/mnn-input-0.txt; do
    if [ ! -f "$input" ]; then
      echo "[xsai-workload] MNN_SMOKE_FAIL missing $input"
      return 1
    fi
  done
  # MNN writes beside argv[0]. Keep generated outputs away from /bin.
  scratch=$(mktemp -d /tmp/mnn-smoke.XXXXXX) || return 1
  cp /bin/MNNV2Basic.out "$scratch/MNNV2Basic.out" || return 1
  cp /root/mnn-input-0.txt "$scratch/input_0.txt" || return 1
  cd "$scratch" || return 1
  export XSAI_MNN_AME="${XSAI_MNN_AME:-0}"
  if run_workload_cmd "$scratch/MNNV2Basic.out" /root/mobilenet-v2-int8.mnn \
      "${XSAI_MNN_LOOPS:-3}" "${XSAI_MNN_MASK:-0}" 0 1 0; then
    :
  else
    rc=$?
    echo "[xsai-workload] MNN_SMOKE_FAIL rc=$rc"
    return "$rc"
  fi
  if [ ! -s output.txt ]; then
    echo "[xsai-workload] MNN_SMOKE_FAIL missing output tensor"
    return 1
  fi
  argmax=$(awk '{for(i=1;i<=NF;i++){if(n==0 || $i>m){m=$i;idx=n} n++}} END{print idx+0}' output.txt)
  if [ -n "${XSAI_MNN_EXPECTED_ARGMAX:-}" ] && [ "$argmax" != "$XSAI_MNN_EXPECTED_ARGMAX" ]; then
    echo "[xsai-workload] MNN_SMOKE_FAIL argmax=$argmax expected=$XSAI_MNN_EXPECTED_ARGMAX"
    return 1
  fi
  echo "[xsai-workload] MNN_SMOKE_PASS argmax=$argmax output=$scratch/output.txt"
)

run_ncnn_smoke()
(
  if [ ! -x /bin/ncnn_ame_smoke ]; then
    echo "[xsai-workload] NCNN_AME_SMOKE_FAIL missing /bin/ncnn_ame_smoke"
    return 127
  fi
  scratch=$(mktemp -d /tmp/ncnn-smoke.XXXXXX) || return 1
  if run_workload_cmd /bin/ncnn_ame_smoke ref "$scratch/ref.txt" "$scratch/ref-bf16.txt"; then
    :
  else
    rc=$?
    echo "[xsai-workload] NCNN_AME_SMOKE_FAIL reference rc=$rc"
    return "$rc"
  fi
  if (
    export XSAI_NCNN_AME=1 XSAI_NCNN_AME_BF16=1
    run_workload_cmd /bin/ncnn_ame_smoke ame "$scratch/ame.txt" "$scratch/ame-bf16.txt"
  ); then
    :
  else
    rc=$?
    echo "[xsai-workload] NCNN_AME_SMOKE_FAIL AME rc=$rc"
    return "$rc"
  fi
  if [ ! -s "$scratch/ref.txt" ] || [ ! -s "$scratch/ame.txt" ] || ! cmp -s "$scratch/ref.txt" "$scratch/ame.txt"; then
    echo "[xsai-workload] NCNN_AME_SMOKE_FAIL int8 outputs differ or are empty"
    return 1
  fi
  if [ "${XSAI_NCNN_TEST:-0}" = 1 ]; then
    if (
      export XSAI_NCNN_AME=1
      run_workload_cmd /bin/ncnn_test_convolution_3
    ); then
      echo "[xsai-workload] NCNN_TEST_PASS"
    else
      rc=$?
      echo "[xsai-workload] NCNN_TEST_FAIL rc=$rc"
      return "$rc"
    fi
  fi
  echo "[xsai-workload] NCNN_AME_SMOKE_PASS output=$scratch"
)

run_selected_workload()
{
  case "$WORKLOAD" in
    none)
      log "workload disabled"
      WORKLOAD_STATUS=0
      ;;
    shell)
      log "workload set to shell; skipping automatic execution"
      WORKLOAD_STATUS=0
      ;;
    hello_xsai)
      log "launching hello_xsai"
      run_and_capture /bin/hello_xsai
      ;;
    gemm_precomp)
      log "launching gemm_precomp"
      run_and_capture /bin/gemm_precomp
      ;;
    pmu_test)
      log "launching pmu_test"
      run_and_capture /bin/pmu_test
      ;;
    test-backend-ops)
      log "launching test-backend-ops"
      if run_llama_test_backend_ops; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-mxfp4-runtime)
      log "launching MXFP4 runtime quantization-byte gate"
      run_and_capture /bin/test-quantize-fns --mxfp4-runtime-only
      ;;
    llama-mxfp4-packed-check)
      # The second binary is an explicit additive CONFIG_INITRAMFS_SOURCE
      # directory input, never an unconditional default-manifest dependency.
      log "launching control/packed byte, flag and panel-boundary gates"
      if (
        # The canonical manifest installs LP64D libraries in /lib; this
        # toolchain's loader may otherwise search /lib64/lp64d only.
        export LD_LIBRARY_PATH="/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        for program in /bin/test-quantize-fns /bin/test-quantize-fns-packed; do
          if [ ! -x "$program" ]; then
            echo "[xsai-workload] MXFP4_PACKED_CHECK_FAIL missing=$program"
            exit 127
          fi
          sha256sum "$program" || exit "$?"
          run_workload_cmd "$program" --mxfp4-runtime-only || exit "$?"
        done
        # Finish both independent quantizer gates before any native consumer.
        for program in /bin/test-quantize-fns /bin/test-quantize-fns-packed; do
          for fixture in 0 1 2 3 4 5 6 7 8 9; do
            echo "[xsai-workload] MXFP4_PANEL_CHECK binary=$program fixture=$fixture"
            run_workload_cmd "$program" --ame-rmsnorm-fusion-native-only \
              --ame-rmsnorm-fixture "$fixture" || exit "$?"
          done
        done
      ); then
        echo "[xsai-workload] MXFP4_PACKED_CHECK_PASS"
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
        echo "[xsai-workload] MXFP4_PACKED_CHECK_FAIL rc=$WORKLOAD_STATUS"
      fi
      ;;
    llama-quantize-perf)
      log "launching quantization-only correctness and cycle measurements"
      if run_workload_cmd /bin/test-quantize-fns --mxfp4-runtime-only &&
         run_workload_cmd /bin/test-quantize-perf \
           --type mxfp4 --type q8_0 \
           --op quantize_row_q_reference --op quantize_row_q \
           --size 256 --size 4096 --size 65536 -i 10; then
        echo "[xsai-workload] QUANTIZE_PERF_PASS warmup=5 iterations=10 native_matrix=not-issued"
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-mxfp4-packed-perf)
      log "launching same-boot control/packed graph perf and nonreal 15M .om comparison"
      if (
        export LD_LIBRARY_PATH="/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        # Never use a simulator host-test allocator to admit physical AMU DMA.
        case "${XSAI_ALLOC_MODE:-}" in
          malloc|libc|host|anon|anonymous|qemu)
            echo "[xsai-workload] MXFP4_PACKED_PERF_FAIL nonphysical_allocator"
            exit 2
            ;;
        esac
        export GGML_AME_PROFILE=1 GGML_XSAI_OP_PROFILE=1
        export GGML_AME_LOG=0 GGML_AME_PANEL_LOG=0 GGML_AME_PANEL_PROGRESS_LOG=0
        export GGML_AME_TILE_PROGRESS_LOG=0 GGML_XSAI_OP_PROGRESS_LOG=0
        for program in /bin/test-quantize-fns /bin/test-quantize-fns-packed; do
          sha256sum "$program" || exit "$?"
          run_workload_cmd "$program" --mxfp4-runtime-only || exit "$?"
          # The original fixture rotates all four modes, with one warmup and
          # three measured samples each; oracle work is outside its timers.
          run_workload_cmd "$program" --ame-rmsnorm-fusion-perf-only \
            --ame-rmsnorm-fixture 0 || exit "$?"
        done
        export GGML_AME_MX_CONFIG=CUTE_2Tops_128SCP GGML_AME_MXFP4=1
        export GGML_AME_MX32_VERIFIED=1 GGML_AME_MXFP4_K64=1
        XSAI_LLAMA_FAKE_BENCH_MODELS=/root/llama-model.gguf
        XSAI_LLAMA_FAKE_BENCH_T=1
        XSAI_LLAMA_FAKE_BENCH_P=256
        XSAI_LLAMA_FAKE_BENCH_B=256
        XSAI_LLAMA_FAKE_BENCH_UB=256
        XSAI_LLAMA_FAKE_BENCH_N=0
        XSAI_LLAMA_FAKE_BENCH_R=3
        XSAI_LLAMA_FAKE_BENCH_WARMUP=1
        XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS=1
        XSAI_LLAMA_FAKE_BENCH_OUTPUT_ERR=jsonl
        XSAI_LLAMA_FAKE_BENCH_FA=0
        XSAI_LLAMA_FAKE_BENCH_TYPE_K=f16
        XSAI_LLAMA_FAKE_BENCH_TYPE_V=f16
        XSAI_LLAMA_FULL_PREFLIGHT=0
        TRACE_WORKLOAD=0
        for program in /bin/llama-bench /bin/llama-bench-packed; do
          sha256sum "$program" || exit "$?"
          for schedule in serial pipeline-chunk; do
            export GGML_AME_RMSNORM_MXFP4="$schedule"
            echo "[xsai-workload] MXFP4_15M_BEGIN binary=$program schedule=$schedule real_weights=0 measured_layers=1 projected_total=yes warmup=1 repetitions=3"
            WORKLOAD=llama-bench-full run_llama_bench_full "$program" || exit "$?"
            echo "[xsai-workload] MXFP4_15M_PASS binary=$program schedule=$schedule"
          done
        done
      ); then
        echo "[xsai-workload] MXFP4_PACKED_PERF_PASS"
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
        echo "[xsai-workload] MXFP4_PACKED_PERF_FAIL rc=$WORKLOAD_STATUS"
      fi
      ;;
    llama-rmsnorm-mxfp4-512|llama-rmsnorm-mxfp4-512-checkpoint)
      log "launching one-row512 RMSNorm/gamma/MXFP4/AME-pack microbenchmark (no GEMM)"
      if [ "$WORKLOAD" = llama-rmsnorm-mxfp4-512-checkpoint ]; then
        set -- --rmsnorm-mxfp4-512-checkpoint
      else
        set -- --rmsnorm-mxfp4-512
      fi
      run_and_capture /bin/test-quantize-perf "$@" -i 10
      ;;
    llama-f16-attention)
      log "launching independent F16 attention gate"
      if (
        export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL=0
        export XSAI_TEST_BACKEND_OPS_SEED=20261109
        export GGML_AME_F16_BF16=1
        export GGML_AME_F16_BF16_REQUIRE_AME_BUFFER=1
        run_workload_cmd /bin/test-backend-ops \
          test -b CPU --threads 1 -o MUL_MAT -p 'f16_oracle=1'
      ); then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-mx128-native)
      log "launching native128 exact prepared-byte gate"
      run_and_capture /bin/test-quantize-fns --ame-mx128-native-only
      ;;
    llama-mx128-qwen-native)
      log "launching Qwen-shaped native128 exact prepared-byte gate"
      run_and_capture /bin/test-quantize-fns --ame-mx128-qwen-native-only
      ;;
    llama-rmsnorm-mx-fusion|llama-rmsnorm-mx-fusion-perf)
      log "launching single-core RMSNorm/gamma/MXFP4/GEMM two-panel gate"
      if (
        export GGML_AME_PROFILE=1
        if [ "$WORKLOAD" = llama-rmsnorm-mx-fusion-perf ]; then
          set -- --ame-rmsnorm-fusion-perf-only
        else
          set -- --ame-rmsnorm-fusion-native-only
        fi
        run_workload_cmd /bin/test-quantize-fns "$@"
      ); then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-bf16-test-ops)
      log "launching llama BF16 test-backend-ops"
      if run_llama_bf16_test_ops; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-bench)
      log "launching llama-bench"
      if run_llama_bench; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-bench-full|llama-bench-real)
      log "launching $WORKLOAD"
      if run_llama_bench_full; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-fake-bench-suite)
      log "launching llama fake-like bench suite"
      if run_llama_fake_bench_suite; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-simple|llama-simple-xsai)
      log "launching llama-simple-xsai"
      if run_llama_simple; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-xsai)
      log "launching llama-xsai"
      if run_llama_xsai; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    mnn-smoke)
      log "launching mnn smoke"
      if run_mnn_smoke; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    ncnn-smoke)
      log "launching ncnn smoke"
      if run_ncnn_smoke; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    *)
      warn "unknown WORKLOAD: $WORKLOAD"
      WORKLOAD_STATUS=2
      ;;
  esac
}
