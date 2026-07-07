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
  # LLAMA_PROMPT="Your name is John"
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
    export GGML_AME_LOG=0
    export GGML_AME_PANEL_PROGRESS_LOG="${GGML_AME_PANEL_PROGRESS_LOG:-0}"
    export GGML_AME_TILE_PROGRESS_LOG="${GGML_AME_TILE_PROGRESS_LOG:-0}"
    export GGML_AME_PACKED_Q8=1
    export GGML_AME_PROFILE=0
    export GGML_XSAI_OP_PROFILE=0
    export XSAI_TEST_BACKEND_OPS_PROGRESS_LOG="${XSAI_TEST_BACKEND_OPS_PROGRESS_LOG:-0}"
    export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL=0
    if [ -n "${XSAI_LLAMA_TEST_ALLOC_MODE:-}" ]; then
      export XSAI_ALLOC_MODE="$XSAI_LLAMA_TEST_ALLOC_MODE"
    fi
    if [ -n "${XSAI_AME_USE_PACKED_B_PANEL:-}" ]; then
      export GGML_AME_USE_PACKED_B_PANEL="$XSAI_AME_USE_PACKED_B_PANEL"
    fi
    if [ -n "${XSAI_AME_PACKED_B_PANEL_MAX_M:-}" ]; then
      export GGML_AME_PACKED_B_PANEL_MAX_M="$XSAI_AME_PACKED_B_PANEL_MAX_M"
    fi
    if [ -n "${XSAI_AME_DISABLE_PACKED_A_TILES:-}" ]; then
      export GGML_AME_DISABLE_PACKED_A_TILES="$XSAI_AME_DISABLE_PACKED_A_TILES"
    fi
    if [ -n "${XSAI_AME_SKIP_TILE_B_ZERO:-}" ]; then
      export GGML_AME_SKIP_TILE_B_ZERO="$XSAI_AME_SKIP_TILE_B_ZERO"
    fi
    if [ -n "${XSAI_AME_SKIP_TILE_C_ZERO:-}" ]; then
      export GGML_AME_SKIP_TILE_C_ZERO="$XSAI_AME_SKIP_TILE_C_ZERO"
    fi
    run_workload_cmd /bin/test-backend-ops \
      test -o MUL_MAT --buft RISCV_AME \
      -p 'type_a=q8_0,type_b=f32,m=(288|768),n=128,k=(288|768),bs=\[1,1\],nr=\[1,1\]'
  ) || return $?

  echo "[xsai-workload] test-backend-ops CPU SOFT_MAX F16 mask"
  (
    export GGML_AME_LOG=0
    export GGML_AME_PACKED_Q8=1
    export GGML_AME_PROFILE=0
    export GGML_XSAI_OP_PROFILE=0
    if [ -n "${XSAI_LLAMA_TEST_ALLOC_MODE:-}" ]; then
      export XSAI_ALLOC_MODE="$XSAI_LLAMA_TEST_ALLOC_MODE"
    fi
    run_workload_cmd /bin/test-backend-ops \
      test -b CPU -o SOFT_MAX \
      -p 'type=f32,ne=\[32,2,32,1\],mask=1,sinks=0,m_prec=f16,nr23=\[1,1\],scale=0.100000,max_bias=0.000000,inplace=0'
  )
}

run_llama_ame_large_matmul()
{
  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set -x
  fi

  echo "[xsai-workload] test-backend-ops perf AME large MUL_MAT Q8_0/F32 m=8192 n=128 k=2048"
  (
    export GGML_AME_LOG="${GGML_AME_LOG:-0}"
    export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-1}"
    export GGML_AME_PROFILE="${GGML_AME_PROFILE:-1}"
    export GGML_XSAI_OP_PROFILE=0
    if [ -n "${XSAI_AME_SKIP_TILE_B_ZERO:-}" ]; then
      export GGML_AME_SKIP_TILE_B_ZERO="$XSAI_AME_SKIP_TILE_B_ZERO"
    fi
    if [ -n "${XSAI_AME_SKIP_TILE_C_ZERO:-}" ]; then
      export GGML_AME_SKIP_TILE_C_ZERO="$XSAI_AME_SKIP_TILE_C_ZERO"
    fi
    if [ -n "${XSAI_AME_USE_PACKED_B_PANEL:-}" ]; then
      export GGML_AME_USE_PACKED_B_PANEL="$XSAI_AME_USE_PACKED_B_PANEL"
    fi
    run_workload_cmd /bin/test-backend-ops \
      perf -o MUL_MAT --buft RISCV_AME \
      -p 'type_a=q8_0,type_b=f32,m=8192,n=128,k=2048,bs=\[1,1\],nr=\[1,1\]'
  )
  rc=$?

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set +x
  fi
  return "$rc"
}

run_llama_ame_k8192_matmul()
{
  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set -x
  fi

  repeat="${XSAI_AME_REPEAT:-1}"
  ame_m="${XSAI_AME_M:-2048}"
  ame_n="${XSAI_AME_N:-128}"
  ame_k="${XSAI_AME_K:-8192}"
  shape_filter="type_a=q8_0,type_b=f32,m=${ame_m},n=${ame_n},k=${ame_k},bs=\\[1,1\\],nr=\\[1,1\\]"
  echo "[xsai-workload] test-backend-ops perf AME MUL_MAT Q8_0/F32 m=$ame_m n=$ame_n k=$ame_k repeat=$repeat"
  i=1
  while [ "$i" -le "$repeat" ]
  do
    echo "[xsai-workload] k8192 iteration $i/$repeat"
    if [ "$i" = 1 ] && [ "${XSAI_AME_CKPT_BEFORE_TEST_OP:-0}" = 1 ]; then
      if [ -x /bin/before_workload ]; then
        echo "[xsai-workload] checkpoint marker before test-backend-ops"
        /bin/before_workload
      else
        echo "[xsai-workload] warning: /bin/before_workload not executable"
      fi
    fi
    (
      export GGML_AME_LOG="${GGML_AME_LOG:-0}"
      export GGML_AME_PANEL_LOG="${GGML_AME_PANEL_LOG:-0}"
      export GGML_AME_PANEL_PROGRESS_LOG="${GGML_AME_PANEL_PROGRESS_LOG:-0}"
      export GGML_AME_TILE_PROGRESS_LOG="${GGML_AME_TILE_PROGRESS_LOG:-0}"
      export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-1}"
      export GGML_AME_PROFILE="${GGML_AME_PROFILE:-1}"
      export GGML_XSAI_OP_PROFILE=0
      if [ "${XSAI_AME_CKPT_BEFORE_TEST_OP:-0}" = 1 ]; then
        export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL=0
      elif [ -n "${XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL:-}" ]; then
        export XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL="$XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL"
      fi
      if [ -n "${XSAI_AME_SKIP_TILE_B_ZERO:-}" ]; then
        export GGML_AME_SKIP_TILE_B_ZERO="$XSAI_AME_SKIP_TILE_B_ZERO"
      fi
      if [ -n "${XSAI_AME_SKIP_TILE_C_ZERO:-}" ]; then
        export GGML_AME_SKIP_TILE_C_ZERO="$XSAI_AME_SKIP_TILE_C_ZERO"
      fi
      if [ -n "${XSAI_AME_USE_PACKED_B_PANEL:-}" ]; then
        export GGML_AME_USE_PACKED_B_PANEL="$XSAI_AME_USE_PACKED_B_PANEL"
      fi
      if [ -n "${XSAI_AME_PACKED_B_PANEL_MAX_M:-}" ]; then
        export GGML_AME_PACKED_B_PANEL_MAX_M="$XSAI_AME_PACKED_B_PANEL_MAX_M"
      fi
      if [ -n "${XSAI_AME_DISABLE_PACKED_A_TILES:-}" ]; then
        export GGML_AME_DISABLE_PACKED_A_TILES="$XSAI_AME_DISABLE_PACKED_A_TILES"
      fi
      run_workload_cmd /bin/test-backend-ops \
        perf -o MUL_MAT --buft RISCV_AME \
        -p "$shape_filter"
    ) || {
      rc=$?
      if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
        set +x
      fi
      return "$rc"
    }
    i=$((i + 1))
  done

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set +x
  fi
  return 0
}

run_ame_panel_probe()
{
  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set -x
  fi

  (
    export AME_PANEL_PROBE_KB="${XSAI_AME_PANEL_PROBE_KB:-128}"
    export AME_PANEL_PROBE_PASSES="${XSAI_AME_PANEL_PROBE_PASSES:-1}"
    export AME_PANEL_PROBE_M_TILES="${XSAI_AME_PANEL_PROBE_M_TILES:-16}"
    export AME_PANEL_PROBE_START_MTILE="${XSAI_AME_PANEL_PROBE_START_MTILE:-0}"
    export AME_PANEL_PROBE_START_KB="${XSAI_AME_PANEL_PROBE_START_KB:-0}"
    export AME_PANEL_PROBE_STEP_KB="${XSAI_AME_PANEL_PROBE_STEP_KB:-1}"
    export AME_PANEL_PROBE_PACKED_LOOP="${XSAI_AME_PANEL_PROBE_PACKED_LOOP:-0}"
    if [ -n "${XSAI_AME_PANEL_PROBE_REUSE_MTILE:-}" ]; then
      export AME_PANEL_PROBE_REUSE_MTILE="$XSAI_AME_PANEL_PROBE_REUSE_MTILE"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_LIMIT_KB:-}" ]; then
      export AME_PANEL_PROBE_LIMIT_KB="$XSAI_AME_PANEL_PROBE_LIMIT_KB"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_PACK_A:-}" ]; then
      export AME_PANEL_PROBE_PACK_A="$XSAI_AME_PANEL_PROBE_PACK_A"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_SCALE:-}" ]; then
      export AME_PANEL_PROBE_SCALE="$XSAI_AME_PANEL_PROBE_SCALE"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_STORE:-}" ]; then
      export AME_PANEL_PROBE_STORE="$XSAI_AME_PANEL_PROBE_STORE"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_LOG_EACH:-}" ]; then
      export AME_PANEL_PROBE_LOG_EACH="$XSAI_AME_PANEL_PROBE_LOG_EACH"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_PROGRESS_INTERVAL:-}" ]; then
      export AME_PANEL_PROBE_PROGRESS_INTERVAL="$XSAI_AME_PANEL_PROBE_PROGRESS_INTERVAL"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_ITER_PROGRESS:-}" ]; then
      export AME_PANEL_PROBE_ITER_PROGRESS="$XSAI_AME_PANEL_PROBE_ITER_PROGRESS"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_C_BUFFERS:-}" ]; then
      export AME_PANEL_PROBE_C_BUFFERS="$XSAI_AME_PANEL_PROBE_C_BUFFERS"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_ZERO_C:-}" ]; then
      export AME_PANEL_PROBE_ZERO_C="$XSAI_AME_PANEL_PROBE_ZERO_C"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_FENCE_BEFORE:-}" ]; then
      export AME_PANEL_PROBE_FENCE_BEFORE="$XSAI_AME_PANEL_PROBE_FENCE_BEFORE"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_FENCE_AFTER:-}" ]; then
      export AME_PANEL_PROBE_FENCE_AFTER="$XSAI_AME_PANEL_PROBE_FENCE_AFTER"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_LEGACY_HOT_LOOP:-}" ]; then
      export AME_PANEL_PROBE_LEGACY_HOT_LOOP="$XSAI_AME_PANEL_PROBE_LEGACY_HOT_LOOP"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_DETAIL_START:-}" ]; then
      export AME_PANEL_PROBE_DETAIL_START="$XSAI_AME_PANEL_PROBE_DETAIL_START"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_DETAIL_END:-}" ]; then
      export AME_PANEL_PROBE_DETAIL_END="$XSAI_AME_PANEL_PROBE_DETAIL_END"
    fi
    if [ -n "${XSAI_AME_PANEL_PROBE_STOP_AFTER_TOTAL:-}" ]; then
      export AME_PANEL_PROBE_STOP_AFTER_TOTAL="$XSAI_AME_PANEL_PROBE_STOP_AFTER_TOTAL"
    fi
    run_workload_cmd /bin/ame_panel_probe
  )
  rc=$?

  if [ "${TRACE_WORKLOAD:-1}" = 1 ]; then
    set +x
  fi
  return "$rc"
}

run_llama_fake_bench_suite()
{
  status=0
  # Prompt-only runs keep the focus on prefill/matmul and avoid decode noise.
  # --estimate-layers 1 estimates prompt cost from l_out[0,1).
  bench_prompt="${XSAI_LLAMA_FAKE_BENCH_P:-128}"
  clean_bench_args="-t 1 -p $bench_prompt -n 0 -r 1 --estimate-layers 1 -fa 0 --no-warmup"
  profile_bench_args="-t 1 -p $bench_prompt -n 0 -r 1 --estimate-layers 1 -fa 0 --no-warmup"

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

  models="
/root/stories15M-q8_0.gguf.om
/root/qwen3_0.6b_q8.gguf.om
/root/llama3.2_1b_q8.gguf.om
/root/DeepSeek-R1-Distill-Qwen-1.5B-q8_0.gguf.om
"
models="/root/stories15M-q8_0.gguf.om
/root/llama3.2_1b_q8.gguf.om"
  if [ -n "${XSAI_LLAMA_FAKE_BENCH_MODELS:-}" ]; then
    models="$XSAI_LLAMA_FAKE_BENCH_MODELS"
  fi

  for model in $models
  do
    # echo "[xsai-workload] llama-bench clean: $model"
    # if (
    #   export GGML_AME_LOG=0
    #   export GGML_AME_PACKED_Q8=1
    #   export GGML_AME_PROFILE=0
    #   export GGML_XSAI_OP_PROFILE=0
    #   export LLAMA_BENCH_PRINT_LAYERS_01=0
    #   run_workload_cmd /bin/llama-bench --fake-like "$model" $clean_bench_args
    # ); then
    #   :
    # else
    #   rc=$?
    #   if [ "$status" = 0 ]; then
    #     status=$rc
    #   fi
    # fi

    echo "[xsai-workload] llama-bench AME profile: $model"
    if (
      export GGML_AME_LOG="${GGML_AME_LOG:-0}"
      export GGML_AME_PANEL_LOG="${GGML_AME_PANEL_LOG:-0}"
      export GGML_AME_PANEL_PROGRESS_LOG="${GGML_AME_PANEL_PROGRESS_LOG:-0}"
      export GGML_AME_CKPT_ON_SHAPE="${GGML_AME_CKPT_ON_SHAPE:-0}"
      export GGML_AME_CKPT_M="${GGML_AME_CKPT_M:-2048}"
      export GGML_AME_CKPT_N="${GGML_AME_CKPT_N:-128}"
      export GGML_AME_CKPT_K="${GGML_AME_CKPT_K:-8192}"
      export GGML_AME_PACKED_Q8="${GGML_AME_PACKED_Q8:-1}"
      export GGML_AME_PROFILE="${GGML_AME_PROFILE:-1}"
      export GGML_XSAI_OP_PROFILE=1
      export GGML_XSAI_OP_PROGRESS_LOG="${GGML_XSAI_OP_PROGRESS_LOG:-0}"
      export LLAMA_BENCH_PRINT_LAYERS_01=0
      if [ -n "${XSAI_AME_USE_PACKED_B_PANEL:-}" ]; then
        export GGML_AME_USE_PACKED_B_PANEL="$XSAI_AME_USE_PACKED_B_PANEL"
      fi
      if [ -n "${XSAI_AME_PACKED_B_PANEL_MAX_M:-}" ]; then
        export GGML_AME_PACKED_B_PANEL_MAX_M="$XSAI_AME_PACKED_B_PANEL_MAX_M"
      fi
      if [ -n "${XSAI_AME_SKIP_TILE_B_ZERO:-}" ]; then
        export GGML_AME_SKIP_TILE_B_ZERO="$XSAI_AME_SKIP_TILE_B_ZERO"
      fi
      if [ -n "${XSAI_AME_SKIP_TILE_C_ZERO:-}" ]; then
        export GGML_AME_SKIP_TILE_C_ZERO="$XSAI_AME_SKIP_TILE_C_ZERO"
      fi
      run_workload_cmd /bin/llama-bench --fake-like "$model" $profile_bench_args
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
    llama-ame-large-matmul)
      log "launching llama AME large matmul perf"
      if run_llama_ame_large_matmul; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    llama-ame-k8192-matmul)
      log "launching llama AME k8192 matmul perf"
      if run_llama_ame_k8192_matmul; then
        WORKLOAD_STATUS=0
      else
        WORKLOAD_STATUS=$?
      fi
      ;;
    ame-panel-probe)
      log "launching AME panel probe"
      if run_ame_panel_probe; then
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
