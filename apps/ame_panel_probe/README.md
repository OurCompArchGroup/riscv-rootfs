# AME Panel Probe

`ame_panel_probe` is a small FPGA-side reproducer for the packed-Q8 AME tile loop used by llama.cpp.

## What It Isolates

The failing 1B path uses `GGML_AME_PACKED_Q8=1` and dispatches Q8_0 x F32 `MUL_MAT` to `ggml_ame_mul_mat_q8_0_ame64()`.

The switch mapping is:

- `XSAI_AME_USE_PACKED_B_PANEL=1` enables the packed-B-panel branch in `ame-matmul.c`.
- `XSAI_AME_SKIP_TILE_C_ZERO=1` skips the per-tile C memset before `ggml_ame_gemm_tile_i8_i32_bT()`.
- `XSAI_AME_SKIP_TILE_B_ZERO=1` only affects the non-packed-B branch. It is not used inside the packed-B-panel branch.

Current evidence points to the packed-B-panel branch as the trigger. Logs from May 15 show the 1B fake bench completes without packed B, while runs with `GGML_AME_USE_PACKED_B_PANEL=1` reach the 1B `llama-bench` command and stop before the result row/profile output. A 180 second observation window is too short for the 1B bench: successful no-packed-B runs took about 8-10 minutes wall time, so use at least 12 minutes for FPGA observation.

## Probe Shape

The default probe settings are safe for a short smoke run and mirror the switch semantics:

- `XSAI_AME_PANEL_PROBE_PACKED_LOOP` follows `XSAI_AME_USE_PACKED_B_PANEL`.
- `XSAI_AME_PANEL_PROBE_ZERO_C` defaults to `0` when `XSAI_AME_SKIP_TILE_C_ZERO=1`.
- `XSAI_AME_PANEL_PROBE_PACK_A=1` mirrors the packed-Q8 AME path copying A rows into the tile buffer.

For a stronger B-panel stress case:

```sh
make firmware \
  XSAI_WORKLOAD=ame-panel-probe \
  XSAI_AUTO_EXIT=1 \
  XSAI_DROP_SHELL=0 \
  XSAI_AME_PANEL_PROBE_KB=128 \
  XSAI_AME_PANEL_PROBE_M_TILES=2048 \
  XSAI_AME_PANEL_PROBE_PACKED_LOOP=1 \
  XSAI_AME_PANEL_PROBE_PACK_A=1 \
  XSAI_AME_PANEL_PROBE_SCALE=0 \
  XSAI_AME_PANEL_PROBE_STORE=0 \
  XSAI_AME_PANEL_PROBE_PROGRESS_INTERVAL=128 \
  XSAI_AME_PANEL_PROBE_ZERO_C=0
```

Then run FPGA with a timeout based on a known-good run plus 20 percent. For the current 1B fake bench evidence, `FPGA_TIMEOUT=720` is the minimum practical default.
