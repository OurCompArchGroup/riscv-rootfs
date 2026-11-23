DEFAULT_LLAMA_MODEL=/root/llama-model.gguf
DEFAULT_LLAMA_PROMPT='once upon a time, in a forest where the trees never lost their leaves, a young woodcutter discovered a door hidden beneath an ancient oak. the door was no taller than his knee and sealed with a lock shaped like a crescent moon. he had passed this way a hundred times but never noticed it before. that morning, something made him kneel in the soft earth and press his ear against the weathered wood. a voice from inside whispered his name. he reached into his coat pocket and found, to his astonishment, a silver key he had never seen before. with trembling '

MODEL_MNT="${MODEL_MNT:-/models}"
LLAMA_MODEL="${LLAMA_MODEL:-$DEFAULT_LLAMA_MODEL}"
LLAMA_PROMPT="${LLAMA_PROMPT:-$DEFAULT_LLAMA_PROMPT}"

log()
{
  echo "[xsai-init] $*"
}

warn()
{
  echo "[xsai-init] warning: $*"
}

setup_base_mounts()
{
  /bin/busybox --install -s

  mount -t proc proc /proc
  mount -t sysfs sysfs /sys
  mount -t devtmpfs devtmpfs /dev
}

apply_cmdline_overrides()
{
  [ -r /proc/cmdline ] || return

  for arg in $(cat /proc/cmdline); do
    case "$arg" in
      xsai.mode=*) XSAI_MODE="${arg#xsai.mode=}" ;;
      xsai.workload=*) WORKLOAD="${arg#xsai.workload=}" ;;
      xsai.auto_exit=*) AUTO_EXIT="${arg#xsai.auto_exit=}" ;;
      xsai.drop_shell=*) DROP_SHELL="${arg#xsai.drop_shell=}" ;;
      xsai.run_after_workload=*) RUN_AFTER_WORKLOAD="${arg#xsai.run_after_workload=}" ;;
      xsai.dump_thp=*) DUMP_THP="${arg#xsai.dump_thp=}" ;;
      xsai.llama_model=*) LLAMA_MODEL="${arg#xsai.llama_model=}" ;;
      xsai.llama_full_model=*) XSAI_LLAMA_FULL_MODEL="${arg#xsai.llama_full_model=}" ;;
      xsai.llama_full_t=*) XSAI_LLAMA_FULL_T="${arg#xsai.llama_full_t=}" ;;
      xsai.llama_full_p=*) XSAI_LLAMA_FULL_P="${arg#xsai.llama_full_p=}" ;;
      xsai.llama_full_b=*) XSAI_LLAMA_FULL_B="${arg#xsai.llama_full_b=}" ;;
      xsai.llama_full_ub=*) XSAI_LLAMA_FULL_UB="${arg#xsai.llama_full_ub=}" ;;
      xsai.llama_full_n=*) XSAI_LLAMA_FULL_N="${arg#xsai.llama_full_n=}" ;;
      xsai.llama_full_r=*) XSAI_LLAMA_FULL_R="${arg#xsai.llama_full_r=}" ;;
      xsai.llama_full_warmup=*) XSAI_LLAMA_FULL_WARMUP="${arg#xsai.llama_full_warmup=}" ;;
      xsai.llama_full_progress=*) XSAI_LLAMA_FULL_PROGRESS="${arg#xsai.llama_full_progress=}" ;;
      xsai.llama_full_fa=*) XSAI_LLAMA_FULL_FA="${arg#xsai.llama_full_fa=}" ;;
      xsai.llama_full_type_k=*) XSAI_LLAMA_FULL_TYPE_K="${arg#xsai.llama_full_type_k=}" ;;
      xsai.llama_full_type_v=*) XSAI_LLAMA_FULL_TYPE_V="${arg#xsai.llama_full_type_v=}" ;;
      xsai.llama_full_preflight=*) XSAI_LLAMA_FULL_PREFLIGHT="${arg#xsai.llama_full_preflight=}" ;;
      xsai.pldm_exit_marker=*) XSAI_PLDM_EXIT_MARKER="${arg#xsai.pldm_exit_marker=}" ;;
      xsai.ggml_ame_packed_q8=*) export GGML_AME_PACKED_Q8="${arg#xsai.ggml_ame_packed_q8=}" ;;
      xsai.ggml_ame_whole_k_q8=*) export GGML_AME_WHOLE_K_Q8="${arg#xsai.ggml_ame_whole_k_q8=}" ;;
      xsai.ggml_ame_f16_bf16=*) export GGML_AME_F16_BF16="${arg#xsai.ggml_ame_f16_bf16=}" ;;
      xsai.ggml_ame_profile=*) export GGML_AME_PROFILE="${arg#xsai.ggml_ame_profile=}" ;;
      xsai.ggml_op_profile=*) export GGML_XSAI_OP_PROFILE="${arg#xsai.ggml_op_profile=}" ;;
      xsai.ggml_ame_log=*) export GGML_AME_LOG="${arg#xsai.ggml_ame_log=}" ;;
      xsai.ggml_ame_mx_config=*) export GGML_AME_MX_CONFIG="${arg#xsai.ggml_ame_mx_config=}" ;;
      xsai.ggml_ame_mxfp4=*) export GGML_AME_MXFP4="${arg#xsai.ggml_ame_mxfp4=}" ;;
      xsai.ggml_ame_mxfp8=*) export GGML_AME_MXFP8="${arg#xsai.ggml_ame_mxfp8=}" ;;
      xsai.ggml_ame_mxfp8_format=*) export GGML_AME_MXFP8_FORMAT="${arg#xsai.ggml_ame_mxfp8_format=}" ;;
      xsai.ggml_ame_mx32_verified=*) export GGML_AME_MX32_VERIFIED="${arg#xsai.ggml_ame_mx32_verified=}" ;;
      xsai.ggml_ame_mxfp4_k64=*) export GGML_AME_MXFP4_K64="${arg#xsai.ggml_ame_mxfp4_k64=}" ;;
    esac
  done
}

select_model_from_disk()
{
  candidate=

  if [ -f "$MODEL_MNT/model.gguf" ]; then
    LLAMA_MODEL="$MODEL_MNT/model.gguf"
    return
  fi

  for candidate in "$MODEL_MNT"/*.gguf; do
    if [ -f "$candidate" ]; then
      LLAMA_MODEL="$candidate"
      return
    fi
  done
}

mount_model_disk()
{
  mkdir -p "$MODEL_MNT"

  if [ ! -b /dev/vda ]; then
    log "no /dev/vda; using initramfs model"
    return
  fi

  log "trying to mount /dev/vda"
  if mount -o ro,noload /dev/vda "$MODEL_MNT" 2>/dev/null || \
     mount -o ro /dev/vda "$MODEL_MNT" 2>/dev/null || \
     mount /dev/vda "$MODEL_MNT" 2>/dev/null; then
    select_model_from_disk
  else
    warn "failed to mount /dev/vda; using initramfs model"
  fi
}

maybe_write_value()
{
  if [ -f "$1" ]; then
    echo "$2" > "$1"
  fi
}

configure_perf_and_thp()
{
  maybe_write_value /proc/sys/kernel/perf_user_access 2
  maybe_write_value /sys/kernel/mm/transparent_hugepage/enabled madvise
  maybe_write_value /sys/kernel/mm/transparent_hugepage/shmem_enabled advise
}

setup_dsid_cgroup()
{
  log "trying DSID cgroup setup"

  if [ -d /sys/fs/cgroup ]; then
    CGROUP_ROOT=/sys/fs/cgroup
  else
    CGROUP_ROOT=/cgroup
    mkdir -p "$CGROUP_ROOT"
    mount -t tmpfs cgroup_root "$CGROUP_ROOT" || true
  fi

  mkdir -p "$CGROUP_ROOT"/dsid/test-1 "$CGROUP_ROOT"/dsid/test-2
  mount -t cgroup -o dsid dsid "$CGROUP_ROOT"/dsid || true
  echo 1 > "$CGROUP_ROOT"/dsid/test-1/dsid.dsid-set || true
  echo 2 > "$CGROUP_ROOT"/dsid/test-2/dsid.dsid-set || true
}

probe_ame_hw()
{
  if [ -x /bin/ame_hw_probe ]; then
    /bin/ame_hw_probe || warn "AME hardware probe failed"
  fi
}

dump_thp_counters()
{
  log "THP counters $1 llama:"
  grep -E 'AnonHugePages|FileHugePages|ShmemHugePages' /proc/meminfo || true
  grep 'thp_fault_alloc\|thp_collapse_alloc\|thp_file_mapped\|thp_file_alloc' /proc/vmstat || true
}

run_workload_cmd()
{
  if [ ! -x "$1" ]; then
    warn "workload binary not executable: $1"
    return 127
  fi

  "$@"
}

run_after_workload()
{
  if [ "$RUN_AFTER_WORKLOAD" != 1 ]; then
    return
  fi

  if [ -x /bin/after_workload ]; then
    log "launching after_workload with status $WORKLOAD_STATUS"
    /bin/after_workload "$WORKLOAD_STATUS" || true
  fi
}

run_workload()
{
  if [ "$DUMP_THP" = 1 ]; then
    dump_thp_counters before
  fi

  run_selected_workload
  # Emit the receipt before after_workload's existing status/trap instruction:
  # a TrapFinish-enabled testbench need not return from that instruction.
  if [ "${XSAI_PLDM_EXIT_MARKER:-0}" = 1 ] &&
     [ "$WORKLOAD" = llama-bench-real ] && [ "$WORKLOAD_STATUS" = 0 ]; then
    log "PLDM_WORKLOAD_DONE workload=$WORKLOAD model=${XSAI_LLAMA_FULL_MODEL:-$LLAMA_MODEL} real_weights=1 all_layers=1 status=0"
  fi
  run_after_workload

  if [ "$DUMP_THP" = 1 ]; then
    dump_thp_counters after
  fi
}

env_init()
{
  setup_base_mounts
  apply_cmdline_overrides
  mount_model_disk
  configure_perf_and_thp
  setup_dsid_cgroup
  probe_ame_hw

  log "model path: $LLAMA_MODEL"
}
