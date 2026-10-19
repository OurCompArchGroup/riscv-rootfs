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
  run_workload_cmd /bin/llama-simple-xsai -m "$LLAMA_MODEL" "$LLAMA_PROMPT"
}

run_llama_xsai()
{
  run_workload_cmd /bin/llama-xsai --model "$LLAMA_MODEL" --prompt "$LLAMA_PROMPT"
}

export_llama_ame_config()
{
  export GGML_AME_LOG="${GGML_AME_LOG:-0}"
  export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-1}"
  export GGML_AME_FLASH_ATTN="${GGML_AME_FLASH_ATTN:-1}"
  export GGML_AME_FLASH_ATTN_PROFILE="${GGML_AME_FLASH_ATTN_PROFILE:-0}"
  export GGML_AME_FLASH_ATTN_VALIDATE="${GGML_AME_FLASH_ATTN_VALIDATE:-0}"
  export GGML_AME_WHOLE_K_Q8="${GGML_AME_WHOLE_K_Q8:-0}"
  export GGML_AME_WHOLE_K_FUSED_EPILOGUE="${GGML_AME_WHOLE_K_FUSED_EPILOGUE:-0}"
  export GGML_AME_F16_BF16="${GGML_AME_F16_BF16:-0}"
  export GGML_AME_WHOLE_K_BF16="${GGML_AME_WHOLE_K_BF16:-0}"
  export GGML_AME_PROFILE="${GGML_AME_PROFILE:-1}"
  export GGML_XSAI_OP_PROFILE="${GGML_XSAI_OP_PROFILE:-1}"
  export GGML_XSAI_OP_TIMING_LOG="${GGML_XSAI_OP_TIMING_LOG:-0}"
  export GGML_AME_USE_PACKED_B_PANEL="${XSAI_AME_USE_PACKED_B_PANEL:-0}"
}

run_llama_test_backend_ops()
{
  if [ "${XSAI_LLAMA_TEST_BACKEND_OPS:-1}" = 0 ]; then
    echo "[xsai-workload] test-backend-ops disabled"
    return 0
  fi

  echo "[xsai-workload] test-backend-ops AME MUL_MAT Q8_0/F32"
  (
    export_llama_ame_config
    run_workload_cmd /bin/test-backend-ops \
      test -o MUL_MAT --buft RISCV_AME \
      -p 'type_a=q8_0,type_b=f32,m=(288|768),n=128,k=(288|768),bs=\[1,1\],nr=\[1,1\]'
  ) || return $?

  echo "[xsai-workload] test-backend-ops CPU SOFT_MAX F16 mask"
  (
    export_llama_ame_config
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU -o SOFT_MAX \
      -p 'type=f32,ne=\[32,2,32,1\],mask=1,sinks=0,m_prec=f16,nr23=\[1,1\],scale=0.100000,max_bias=0.000000,inplace=0'
  )
}

run_llama_fake_bench_suite()
{
  status=0
  bench_prompt="${XSAI_LLAMA_FAKE_BENCH_P:-128}"
  bench_generation="${XSAI_LLAMA_FAKE_BENCH_N:-0}"
  bench_repetitions="${XSAI_LLAMA_FAKE_BENCH_R:-1}"
  bench_estimate_layers="${XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS:-1}"
  bench_flash_attn="${XSAI_LLAMA_FAKE_BENCH_FA:-0}"
  bench_args="-t 1 -p $bench_prompt -n $bench_generation -r $bench_repetitions --estimate-layers $bench_estimate_layers -fa $bench_flash_attn"

  export GGML_AME_LOG="${GGML_AME_LOG:-0}"
  export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-1}"
  export GGML_XSAI_MEM_TRACE_NAME="${GGML_MEM_TRACE_NAME:-*}"
  export GGML_XSAI_MEM_TRACE_OP="${GGML_MEM_TRACE_OP:-*}"
  export GGML_XSAI_MEM_TRACE_LAYER="${GGML_MEM_TRACE_LAYER:-1}"

  # Select the simulation window by exact compute graph node names.
  # CHECKPOINT_HERE and GOOD_TRAP are triggered before their selected nodes execute.
  # Effective only when 'GGML_EASY_SIM_WINDOW_AFTER_CHECKPOINT' is defined at compile time.
  export CHECKPOINT_HERE_BEFORE="${CHECKPOINT_HERE_BEFORE:-}"
  export GOOD_TRAP_BEFORE="${GOOD_TRAP_BEFORE:-}"
  case "${XSAI_LLAMA_FAKE_BENCH_WARMUP:-0}" in
    0|false|FALSE|off|OFF|no|NO) bench_args="$bench_args --no-warmup" ;;
  esac
  if [ -n "${XSAI_LLAMA_FAKE_BENCH_TYPE_K:-}" ]; then
    bench_args="$bench_args -ctk $XSAI_LLAMA_FAKE_BENCH_TYPE_K"
  fi
  if [ -n "${XSAI_LLAMA_FAKE_BENCH_TYPE_V:-}" ]; then
    bench_args="$bench_args -ctv $XSAI_LLAMA_FAKE_BENCH_TYPE_V"
  fi

  models="
/root/stories15M-q8_0.gguf.om
/root/qwen3_0.6b_q8.gguf.om
/root/llama3.2_1b_q8.gguf.om
/root/DeepSeek-R1-Distill-Qwen-1.5B-q8_0.gguf.om
"
  if [ -n "${XSAI_LLAMA_FAKE_BENCH_MODELS:-}" ]; then
    models="$XSAI_LLAMA_FAKE_BENCH_MODELS"
  fi

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set -x
  fi

  if [ -x /bin/before_workload ]; then
    echo "[xsai-workload] checkpoint boundary before llama fake-like bench suite"
    /bin/before_workload
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

  for model in $models
  do
    echo "[xsai-workload] llama-bench fake-like: $model"
    if (
      export_llama_ame_config
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
