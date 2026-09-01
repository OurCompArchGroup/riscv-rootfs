# XSAI guest workload dispatcher (slim).
#
# Stable workloads:
#   test-backend-ops       AME correctness gate: Q8_0 MUL_MAT, CPU softmax,
#                          BF16 whole-K MUL_MAT (fixed path, always on).
#   llama-bf16-test-ops    small BF16 smoke: CPY, 64x64x32 MUL_MAT, softmax.
#   llama-fake-bench-suite default ci workload: gate + fake-like llama-bench.
#   llama-bench / llama-simple(-xsai) / llama-xsai
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
    *)
      warn "unknown WORKLOAD: $WORKLOAD"
      WORKLOAD_STATUS=2
      ;;
  esac
}
