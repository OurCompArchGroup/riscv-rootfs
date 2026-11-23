#!/usr/bin/env python3
"""Exercise workload exit propagation in a host-only fixture filesystem."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parents[1] / "xsai-workloads.sh"
COMMON_INIT = WORKLOADS.with_name("xsai-init-common.sh")
ROOTFS_MAKEFILE = WORKLOADS.parents[1] / "Makefile"


class AiWorkloadTests(unittest.TestCase):
    quantize_modes = (
        ("llama-mx128-native", "--ame-mx128-native-only"),
        ("llama-mx128-qwen-native", "--ame-mx128-qwen-native-only"),
        ("llama-mxfp4-runtime", "--mxfp4-runtime-only"),
    )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scratch = Path(self.directory.name)
        self.bin_dir = self.scratch / "bin"
        self.models = self.scratch / "models"
        self.bin_dir.mkdir()
        self.models.mkdir()
        # Only fixture paths change. The control flow and shell status handling
        # come from the production workload script, under the init's set -eu.
        self.source = WORKLOADS.read_text().replace("/bin/", f"{self.bin_dir}/")
        self.source = self.source.replace("/root/", f"{self.models}/")
        self.cmdline = self.scratch / "cmdline"
        self.cmdline.write_text("")
        self.common = COMMON_INIT.read_text().replace("/bin/", f"{self.bin_dir}/")
        self.common = self.common.replace("/root/", f"{self.models}/")
        self.common = self.common.replace("/proc/cmdline", str(self.cmdline))

    def binary(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def run_workload(
        self, workload, *, run_init=False, cmdline=None, select_disk=False,
        config=None, **overrides,
    ):
        if cmdline is not None:
            self.cmdline.write_text(cmdline)
        script = (
            "set -eu\n"
            + (f". {shlex.quote(str(config))}\n" if config else "")
            + self.common
            + self.source
            + '\nWORKLOAD="' + workload + '"\n'
            + ("apply_cmdline_overrides\n" if cmdline is not None else "")
            + ("select_model_from_disk\n" if select_disk else "")
            + ("run_workload\n" if run_init else "run_selected_workload\n")
            + 'exit "$WORKLOAD_STATUS"\n'
        )
        return subprocess.run(
            ["/bin/sh"], input=script, text=True, capture_output=True,
            env={
                **os.environ, "RUN_AFTER_WORKLOAD": "1", "DUMP_THP": "0",
                "TRACE_WORKLOAD": "0", **overrides,
            },
        )

    def prepare_mnn(self, body):
        (self.models / "mobilenet-v2-int8.mnn").write_text("model")
        (self.models / "mnn-input-0.txt").write_text("input")
        self.binary("MNNV2Basic.out", body)

    def test_mnn_failure_is_not_success(self):
        self.prepare_mnn("exit 7")
        result = self.run_workload("mnn-smoke")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("MNN_SMOKE_FAIL rc=7", result.stdout)
        self.assertNotIn("MNN_SMOKE_PASS", result.stdout)

    def test_mnn_empty_and_wrong_output_are_failures(self):
        self.prepare_mnn("exit 0")
        self.assertEqual(self.run_workload("mnn-smoke").returncode, 1)
        self.prepare_mnn("printf '0.1 0.9\\n' > output.txt")
        result = self.run_workload("mnn-smoke", XSAI_MNN_EXPECTED_ARGMAX="0")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("MNN_SMOKE_PASS", result.stdout)
        result = self.run_workload("mnn-smoke", XSAI_MNN_EXPECTED_ARGMAX="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MNN_SMOKE_PASS", result.stdout)

    def test_ncnn_reference_failure_and_diff_are_failures(self):
        self.binary("ncnn_ame_smoke", 'if [ "$1" = ref ]; then exit 9; fi')
        result = self.run_workload("ncnn-smoke")
        self.assertEqual(result.returncode, 9)
        self.assertNotIn("NCNN_AME_SMOKE_PASS", result.stdout)
        self.binary("ncnn_ame_smoke", 'printf "%s\\n" "$1" > "$2"')
        self.assertEqual(self.run_workload("ncnn-smoke").returncode, 1)

    def test_ncnn_optional_suite_propagates_failure(self):
        self.binary("ncnn_ame_smoke", 'printf "same\\n" > "$2"')
        self.binary("ncnn_test_convolution_3", "exit 6")
        result = self.run_workload("ncnn-smoke", XSAI_NCNN_TEST="1")
        self.assertEqual(result.returncode, 6)
        self.assertNotIn("NCNN_AME_SMOKE_PASS", result.stdout)
        result = self.run_workload("ncnn-smoke", XSAI_NCNN_TEST="0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("NCNN_AME_SMOKE_PASS", result.stdout)

    def prepare_bench(self, body=None):
        # These files test shell dispatch/argv, not GGUF loading or inference.
        model = self.models / "llama-model.gguf"
        model.write_text("host shell fixture only")
        self.argv_log = self.scratch / "bench-argv.log"
        self.binary(
            "llama-bench",
            body or 'printf "%s\\n" "$@" > "$XSAI_TEST_ARGV_LOG"',
        )
        return {
            "XSAI_TEST_ARGV_LOG": str(self.argv_log),
            "LLAMA_MODEL": str(model),
        }

    def test_mx128_native_uses_existing_test_without_allocator_override(self):
        argv_log = self.scratch / "native-argv.log"
        self.binary(
            "test-quantize-fns",
            'printf "%s\\n" "$@" > "$XSAI_TEST_ARGV_LOG"\n'
            'echo "ALLOC=${XSAI_ALLOC_MODE:-unset}"',
        )
        for workload, argument in self.quantize_modes:
            with self.subTest(workload=workload):
                result = self.run_workload(
                    workload, XSAI_TEST_ARGV_LOG=str(argv_log),
                    XSAI_ALLOC_MODE="devmem",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(argv_log.read_text().splitlines(), [argument])
                self.assertIn("ALLOC=devmem", result.stdout)
                self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)

    def test_mx128_native_failure_reaches_existing_after_workload(self):
        self.binary("test-quantize-fns", "exit 7")
        self.binary("after_workload", 'echo "AFTER_STATUS=$1"; exit 0')
        for workload, _ in self.quantize_modes:
            with self.subTest(workload=workload):
                result = self.run_workload(workload, run_init=True)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIn("AFTER_STATUS=7", result.stdout)
                self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_mx128_native_missing_or_unsupported_is_not_success(self):
        for workload, _ in self.quantize_modes:
            with self.subTest(workload=workload, failure="missing"):
                result = self.run_workload(workload)
                self.assertEqual(result.returncode, 127, result.stderr)
        self.binary("test-quantize-fns", "exit 77")
        for workload, _ in self.quantize_modes:
            with self.subTest(workload=workload, failure="unsupported"):
                result = self.run_workload(workload)
                self.assertEqual(result.returncode, 77, result.stderr)

    def test_mx128_qwen_native_cmdline_selects_existing_dispatch(self):
        argv_log = self.scratch / "qwen-native-argv.log"
        self.binary(
            "test-quantize-fns",
            'printf "%s\\n" "$@" > "$XSAI_TEST_ARGV_LOG"',
        )
        result = self.run_workload(
            "llama-mx128-native",
            cmdline="xsai.workload=llama-mx128-qwen-native",
            XSAI_TEST_ARGV_LOG=str(argv_log),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            argv_log.read_text().splitlines(), ["--ame-mx128-qwen-native-only"],
        )

    def test_f16_attention_selects_existing_fixture_and_preserves_allocator(self):
        argv_log = self.scratch / "attention-argv.log"
        self.binary(
            "test-backend-ops",
            'printf "%s\\n" "$@" > "$XSAI_TEST_ARGV_LOG"\n'
            'echo "ALLOC=$XSAI_ALLOC_MODE F16=$GGML_AME_F16_BF16 '
            'REQUIRE=$GGML_AME_F16_BF16_REQUIRE_AME_BUFFER '
            'SEED=$XSAI_TEST_BACKEND_OPS_SEED SIGNAL=$XSAI_TEST_BACKEND_OPS_NEMU_SIGNAL"',
        )
        result = self.run_workload(
            "none", cmdline="xsai.workload=llama-f16-attention",
            XSAI_TEST_ARGV_LOG=str(argv_log), XSAI_ALLOC_MODE="devmem",
            GGML_AME_F16_BF16="0", GGML_AME_F16_BF16_REQUIRE_AME_BUFFER="0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argv_log.read_text().splitlines(), [
            "test", "-b", "CPU", "--threads", "1", "-o", "MUL_MAT",
            "-p", "f16_oracle=1",
        ])
        self.assertIn("ALLOC=devmem F16=1 REQUIRE=1 SEED=20261109 SIGNAL=0", result.stdout)
        self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)
        self.assertNotIn("LLAMA_BENCH_FULL_PASS", result.stdout)

    def test_f16_attention_failure_uses_existing_after_workload(self):
        self.binary("after_workload", 'echo "AFTER_STATUS=$1"; exit 0')
        result = self.run_workload("llama-f16-attention", run_init=True)
        self.assertEqual(result.returncode, 127, result.stderr)
        self.assertIn("AFTER_STATUS=127", result.stdout)
        self.binary("test-backend-ops", "exit 7")
        result = self.run_workload(
            "llama-f16-attention", run_init=True, XSAI_PLDM_EXIT_MARKER="1",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("AFTER_STATUS=7", result.stdout)
        self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def prepare_bench_environment(self):
        self.env_log = self.scratch / "bench-env.log"
        return {
            **self.prepare_bench(
                'env > "$XSAI_TEST_ENV_LOG"\n'
                'printf "%s\\n" "$@" > "$XSAI_TEST_ARGV_LOG"'
            ),
            "XSAI_TEST_ENV_LOG": str(self.env_log),
        }

    def init_config(self, *, environment=None, **settings):
        # Run only the canonical config target inside the temporary fixture,
        # never in the real rootfs build tree and never any app/build target.
        variables = {"XSAI_DUMP_THP": "0", "XSAI_TRACE_WORKLOAD": "0", **settings}
        result = subprocess.run(
            ["make", "--no-print-directory", "-f", str(ROOTFS_MAKEFILE),
             "xsai-init-config", *(f"{k}={v}" for k, v in variables.items())],
            cwd=self.scratch, text=True, capture_output=True,
            env={**os.environ, **(environment or {})},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.scratch / "rootfsimg/build/xsai-init.conf"

    def bench_environment(self):
        return dict(line.split("=", 1) for line in self.env_log.read_text().splitlines())

    def test_llama_bench_historical_single_model_is_unchanged(self):
        result = self.run_workload("llama-bench", **self.prepare_bench())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.argv_log.read_text().splitlines(),
            ["-m", str(self.models / "llama-model.gguf"), "-t", "1", "-p",
             "512", "-n", "0", "--no-warmup", "--estimate-layers", "1"],
        )

    def test_llama_fullbench_historical_fake_defaults(self):
        self.binary("llama-bench", 'printf "%s\\n" "$*" >> "$XSAI_TEST_ARGV_LOG"')
        argv_log = self.scratch / "fake-argv.log"
        result = self.run_workload(
            "llama-bench-full", XSAI_TEST_ARGV_LOG=str(argv_log),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = argv_log.read_text().splitlines()
        self.assertEqual(len(commands), 3)
        for line in commands:
            self.assertIn("--fake-like ", line)
            self.assertIn(
                "-t 1 -p 256 -n 8 -r 1 -fa 0 -ctk f16 -ctv f16 "
                "--no-warmup --estimate-layers 1", line,
            )
        self.assertIn("LLAMA_BENCH_FULL_PASS", result.stdout)
        self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)

    def test_llama_fullbench_preserves_first_failure(self):
        self.prepare_bench('case "$2" in *stories*) exit 7;; esac; exit 9')
        result = self.run_workload("llama-bench-full")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertNotIn("LLAMA_BENCH_FULL_PASS", result.stdout)

    def test_fakebench_mx_single_layer_pp256_generated_controls(self):
        settings = self.prepare_bench_environment()
        model = self.models / "stories15M-mxfp4.gguf.om"
        model.write_text("metadata-only shell fixture")
        config = self.init_config(
            XSAI_WORKLOAD="llama-bench-full",
            XSAI_LLAMA_FAKE_BENCH_MODELS=str(model),
            XSAI_LLAMA_FAKE_BENCH_P="256",
            XSAI_LLAMA_FAKE_BENCH_B="256",
            XSAI_LLAMA_FAKE_BENCH_UB="256",
            XSAI_LLAMA_FAKE_BENCH_N="0",
            XSAI_LLAMA_FAKE_BENCH_R="1",
            XSAI_LLAMA_FAKE_BENCH_WARMUP="0",
            XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS="1",
            XSAI_LLAMA_FAKE_BENCH_OUTPUT_ERR="jsonl",
            XSAI_GGML_AME_MX_CONFIG="CUTE_2Tops_128SCP",
            XSAI_GGML_AME_MXFP4="1",
            XSAI_GGML_AME_MX32_VERIFIED="1",
            XSAI_GGML_AME_MXFP4_K64="1",
            XSAI_GGML_AME_PROFILE="1",
            XSAI_GGML_OP_PROFILE="1",
        )
        result = self.run_workload("llama-bench-full", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.argv_log.read_text().splitlines(),
            ["--fake-like", str(model), "-t", "1", "-p", "256", "-n", "0",
             "-r", "1", "-fa", "0", "-ctk", "f16", "-ctv", "f16",
             "-b", "256", "-ub", "256", "--no-warmup", "-oe", "jsonl",
             "--estimate-layers", "1"],
        )
        env = self.bench_environment()
        self.assertEqual(env["GGML_AME_MX_CONFIG"], "CUTE_2Tops_128SCP")
        for name in ("GGML_AME_MXFP4", "GGML_AME_MX32_VERIFIED",
                     "GGML_AME_MXFP4_K64", "GGML_AME_PROFILE", "GGML_XSAI_OP_PROFILE"):
            self.assertEqual(env[name], "1", name)
        self.assertIn("LLAMA_BENCH_FULL_PASS", result.stdout)
        self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)

    def test_fakebench_require_buffer_is_explicit_and_preserves_argv(self):
        settings = self.prepare_bench_environment()
        controls = {
            "XSAI_WORKLOAD": "llama-bench-full",
            "XSAI_LLAMA_FAKE_BENCH_MODELS": settings["LLAMA_MODEL"],
            "XSAI_LLAMA_FAKE_BENCH_P": "256",
            "XSAI_LLAMA_FAKE_BENCH_B": "256",
            "XSAI_LLAMA_FAKE_BENCH_UB": "256",
            "XSAI_LLAMA_FAKE_BENCH_N": "0",
            "XSAI_LLAMA_FAKE_BENCH_R": "1",
            "XSAI_LLAMA_FAKE_BENCH_WARMUP": "0",
            "XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS": "1",
            "XSAI_LLAMA_FAKE_BENCH_OUTPUT_ERR": "jsonl",
        }
        config = self.init_config(**controls)
        baseline_config = config.read_text()
        variable = "GGML_AME_F16_BF16_REQUIRE_AME_BUFFER"
        self.assertNotIn(f"export {variable}=", baseline_config)
        result = self.run_workload(
            "llama-bench-full", config=config, **settings,
            GGML_AME_F16_BF16_REQUIRE_AME_BUFFER="0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        baseline_argv = self.argv_log.read_text()
        self.assertEqual(self.bench_environment()[variable], "0")
        for value in ("1", "0"):
            with self.subTest(explicit=value):
                config = self.init_config(
                    **controls, XSAI_GGML_AME_F16_BF16_REQUIRE_AME_BUFFER=value,
                )
                line = f"export {variable}='{value}'\n"
                self.assertEqual(config.read_text().count(line), 1)
                self.assertEqual(config.read_text().replace(line, ""), baseline_config)
                result = self.run_workload(
                    "llama-bench-full", config=config, **settings,
                    GGML_AME_F16_BF16_REQUIRE_AME_BUFFER=str(1 - int(value)),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.argv_log.read_text(), baseline_argv)
                self.assertEqual(self.bench_environment()[variable], value)
                self.assertIn("LLAMA_BENCH_FULL_PASS", result.stdout)

    def test_llama_fullbench_historical_lists_and_warmup(self):
        argv_log = self.scratch / "fake-argv.log"
        self.binary("llama-bench", 'printf "%s\\n" "$*" >> "$XSAI_TEST_ARGV_LOG"')
        result = self.run_workload(
            "llama-bench-full", XSAI_TEST_ARGV_LOG=str(argv_log),
            XSAI_LLAMA_FAKE_BENCH_T="1,2", XSAI_LLAMA_FAKE_BENCH_P="256,512",
            XSAI_LLAMA_FAKE_BENCH_N="0", XSAI_LLAMA_FAKE_BENCH_R="3",
            XSAI_LLAMA_FAKE_BENCH_WARMUP="1",
            XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS="0",
            XSAI_LLAMA_FAKE_BENCH_OUTPUT_ERR="jsonl",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = argv_log.read_text().splitlines()
        self.assertEqual(len(commands), 3)
        for line in commands:
            self.assertIn("-t 1,2 -p 256,512 -n 0 -r 3", line)
            self.assertIn("--estimate-layers 0", line)
            self.assertIn("-oe jsonl", line)
            self.assertNotIn("--no-warmup", line)

    def test_llama_realbench_all_layers_with_warmup_defaults(self):
        result = self.run_workload("llama-bench-real", **self.prepare_bench())
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.argv_log.read_text().splitlines()
        self.assertEqual(
            args,
            ["-m", str(self.models / "llama-model.gguf"), "-t", "1", "-p",
             "512", "-n", "0", "-r", "3", "-fa", "0", "-ctk", "f16",
             "-ctv", "f16", "-b", "2048", "-ub", "512", "-oe", "jsonl"],
        )
        self.assertNotIn("--fake-like", args)
        self.assertNotIn("--estimate-layers", args)
        self.assertNotIn("--no-warmup", args)
        self.assertNotIn("--progress", args)
        self.assertIn("diagnostic_progress=0 packed_q8=0 whole_k_q8=0 f16_bf16=0", result.stdout)
        self.assertIn("LLAMA_BENCH_REAL_PASS", result.stdout)
        self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_explicit_model_and_controls(self):
        settings = self.prepare_bench()
        model = self.models / "real model.gguf"
        model.write_text("host shell fixture only")
        result = self.run_workload(
            "llama-bench-real", **settings,
            XSAI_LLAMA_FULL_MODEL=str(model), XSAI_LLAMA_FULL_T="2",
            XSAI_LLAMA_FULL_P="256", XSAI_LLAMA_FULL_N="8",
            XSAI_LLAMA_FULL_R="4", XSAI_LLAMA_FULL_WARMUP="0",
            XSAI_LLAMA_FULL_FA="1", XSAI_LLAMA_FULL_TYPE_K="bf16",
            XSAI_LLAMA_FULL_TYPE_V="bf16",
            # Historical fake/estimate knobs must never affect this path.
            XSAI_LLAMA_FAKE_BENCH_MODELS="/missing.gguf.om",
            XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.argv_log.read_text().splitlines()
        self.assertEqual(args[:2], ["-m", str(model)])
        self.assertEqual(
            args[2:16],
            ["-t", "2", "-p", "256", "-n", "8", "-r", "4", "-fa", "1",
             "-ctk", "bf16", "-ctv", "bf16"],
        )
        self.assertIn("--no-warmup", args)
        self.assertNotIn("--fake-like", args)
        self.assertNotIn("--estimate-layers", args)

    def test_llama_realbench_qwen_audit_shape_uses_exact_batch_and_ubatch(self):
        settings = self.prepare_bench()
        model = self.models / "Qwen3-0.6B-q8_0.gguf"
        model.write_text("host shell fixture only")
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings,
            XSAI_LLAMA_FULL_MODEL=str(model), XSAI_LLAMA_FULL_T="1",
            XSAI_LLAMA_FULL_P="256", XSAI_LLAMA_FULL_B="256",
            XSAI_LLAMA_FULL_UB="256", XSAI_LLAMA_FULL_N="0",
            XSAI_PLDM_EXIT_MARKER="1",
            XSAI_LLAMA_FAKE_BENCH_MODELS="/missing.gguf.om",
            XSAI_LLAMA_FAKE_BENCH_ESTIMATE_LAYERS="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.argv_log.read_text().splitlines(),
            ["-m", str(model), "-t", "1", "-p", "256", "-n", "0", "-r", "3",
             "-fa", "0", "-ctk", "f16", "-ctv", "f16", "-b", "256", "-ub",
             "256", "-oe", "jsonl"],
        )
        self.assertIn("t=1 p=256 b=256 ub=256 n=0", result.stdout)
        self.assertIn(
            f"model={model} real_weights=1 all_layers=1 status=0", result.stdout,
        )

    def test_llama_realbench_failure_reaches_after_workload_without_done(self):
        settings = self.prepare_bench("exit 7")
        self.binary("after_workload", 'echo "AFTER_STATUS=$1"; exit 9')
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings,
            XSAI_PLDM_EXIT_MARKER="1",
            XSAI_LLAMA_FULL_P="256", XSAI_LLAMA_FULL_B="256",
            XSAI_LLAMA_FULL_UB="256", XSAI_LLAMA_FULL_N="0",
            XSAI_LLAMA_FULL_T="1",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("LLAMA_BENCH_REAL_FAIL rc=7", result.stdout)
        self.assertIn("AFTER_STATUS=7", result.stdout)
        self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)
        self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_done_precedes_existing_after_workload_trap(self):
        settings = self.prepare_bench()
        self.binary("after_workload", 'echo "AFTER_STATUS=$1"; exit 0')
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings,
            XSAI_PLDM_EXIT_MARKER="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PLDM_WORKLOAD_DONE"), 1)
        self.assertIn(
            f"model={settings['LLAMA_MODEL']} real_weights=1 all_layers=1 status=0",
            result.stdout,
        )
        self.assertLess(
            result.stdout.index("LLAMA_BENCH_REAL_PASS"),
            result.stdout.index("PLDM_WORKLOAD_DONE"),
        )
        self.assertLess(
            result.stdout.index("PLDM_WORKLOAD_DONE"),
            result.stdout.index("AFTER_STATUS=0"),
        )

    def test_llama_realbench_keeps_after_workload_opt_out(self):
        settings = self.prepare_bench()
        self.binary("after_workload", 'echo "UNEXPECTED_AFTER"; exit 0')
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings,
            RUN_AFTER_WORKLOAD="0", XSAI_PLDM_EXIT_MARKER="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PLDM_WORKLOAD_DONE", result.stdout)
        self.assertNotIn("UNEXPECTED_AFTER", result.stdout)

    def test_llama_realbench_marker_never_reports_skipped_or_fake_workload(self):
        for workload in ("none", "shell", "unknown", "llama-bench-full"):
            with self.subTest(workload=workload):
                settings = self.prepare_bench()
                result = self.run_workload(
                    workload, run_init=True, **settings, XSAI_PLDM_EXIT_MARKER="1",
                )
                self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_rejects_metadata_missing_and_empty_models(self):
        settings = self.prepare_bench()
        empty = self.models / "empty.gguf"
        empty.touch()
        comma = self.models / "first,second.gguf"
        comma.write_text("host shell fixture only")
        for model in ("/missing.gguf", str(empty), "/metadata.gguf.om", str(comma)):
            with self.subTest(model=model):
                result = self.run_workload(
                    "llama-bench-real", run_init=True, **settings,
                    XSAI_LLAMA_FULL_MODEL=model, XSAI_PLDM_EXIT_MARKER="1",
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("LLAMA_BENCH_REAL_PASS", result.stdout)
                self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)
                self.assertFalse(self.argv_log.exists())

    def test_llama_realbench_rejects_empty_or_invalid_measurements(self):
        settings = self.prepare_bench()
        for invalid in (
            {"XSAI_LLAMA_FULL_R": "0"},
            {"XSAI_LLAMA_FULL_T": "0"},
            {"XSAI_LLAMA_FULL_B": "0"},
            {"XSAI_LLAMA_FULL_UB": "0"},
            {"XSAI_LLAMA_FULL_B": "1,2"},
            {"XSAI_LLAMA_FULL_UB": "-1"},
            {"XSAI_LLAMA_FULL_B": "2147483648"},
            {"XSAI_LLAMA_FULL_P": "0", "XSAI_LLAMA_FULL_N": "0"},
            {"XSAI_LLAMA_FULL_P": "-1"},
            {"XSAI_LLAMA_FULL_P": "1,2"},
            {"XSAI_LLAMA_FULL_N": "2147483648"},
            {"XSAI_LLAMA_FULL_R": "9999999999999999999999"},
            {"XSAI_LLAMA_FULL_WARMUP": "maybe"},
            {"XSAI_LLAMA_FULL_FA": "2"},
            {"XSAI_LLAMA_FULL_PREFLIGHT": "maybe"},
            {"XSAI_LLAMA_FULL_PROGRESS": "maybe"},
        ):
            with self.subTest(invalid=invalid):
                result = self.run_workload(
                    "llama-bench-real", run_init=True, **settings, **invalid,
                    XSAI_PLDM_EXIT_MARKER="1",
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.argv_log.exists())
                self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_missing_binary_and_preflight_fail_closed(self):
        settings = self.prepare_bench()
        (self.bin_dir / "llama-bench").unlink()
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings, XSAI_PLDM_EXIT_MARKER="1",
        )
        self.assertEqual(result.returncode, 127, result.stderr)
        self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)
        settings = self.prepare_bench()
        self.binary("test-backend-ops", "exit 7")
        result = self.run_workload(
            "llama-bench-real", run_init=True, **settings,
            XSAI_LLAMA_FULL_PREFLIGHT="1", XSAI_PLDM_EXIT_MARKER="1",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertNotIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_cmdline_controls_override_config(self):
        settings = self.prepare_bench()
        result = self.run_workload(
            "none", run_init=True, **settings, XSAI_LLAMA_FULL_P="1024",
            XSAI_LLAMA_FULL_B="1024", XSAI_LLAMA_FULL_UB="512",
            cmdline=(
                "xsai.workload=llama-bench-real xsai.llama_full_t=2 "
                "xsai.llama_full_p=256 xsai.llama_full_n=8 xsai.llama_full_r=4 "
                "xsai.llama_full_b=256 xsai.llama_full_ub=256 "
                "xsai.llama_full_warmup=0 xsai.llama_full_fa=1 "
                "xsai.llama_full_type_k=bf16 xsai.llama_full_type_v=bf16 "
                "xsai.llama_full_preflight=0 xsai.pldm_exit_marker=1 "
                f"xsai.llama_full_model={settings['LLAMA_MODEL']}"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.argv_log.read_text().splitlines()
        self.assertEqual(args[args.index("-p") + 1], "256")
        self.assertEqual(args[args.index("-b") + 1], "256")
        self.assertEqual(args[args.index("-ub") + 1], "256")
        self.assertEqual(args[args.index("-t") + 1], "2")
        self.assertEqual(args[args.index("-n") + 1], "8")
        self.assertEqual(args[args.index("-r") + 1], "4")
        self.assertEqual(args[args.index("-fa") + 1], "1")
        self.assertEqual(args[args.index("-ctk") + 1], "bf16")
        self.assertEqual(args[args.index("-ctv") + 1], "bf16")
        self.assertIn("--no-warmup", args)
        self.assertIn("PLDM_WORKLOAD_DONE", result.stdout)

    def test_llama_realbench_preserves_allocator_and_backend_overrides(self):
        settings = self.prepare_bench(
            'echo "ALLOC=$XSAI_ALLOC_MODE PANEL=$GGML_AME_USE_PACKED_B_PANEL '
            'Q8=$GGML_AME_WHOLE_K_Q8"; exit 0'
        )
        result = self.run_workload(
            "llama-bench-real", **settings, XSAI_ALLOC_MODE="malloc",
            GGML_AME_USE_PACKED_B_PANEL="0", GGML_AME_WHOLE_K_Q8="0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALLOC=malloc PANEL=0 Q8=0", result.stdout)

    def test_realbench_generated_baseline_overrides_inherited_packed_default(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(
            environment={"XSAI_GGML_AME_PACKED_Q8": "1"},
            XSAI_WORKLOAD="llama-bench-real", XSAI_GGML_AME_PROFILE="1",
            XSAI_GGML_OP_PROFILE="1",
        )
        result = self.run_workload("llama-bench-real", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        for name in (
            "GGML_AME_PACKED_Q8", "GGML_AME_WHOLE_K_Q8", "GGML_AME_F16_BF16",
            "GGML_AME_USE_PACKED_B_PANEL", "GGML_AME_WHOLE_K_FUSED_EPILOGUE",
            "GGML_AME_WHOLE_K_BF16", "GGML_AME_LOG", "GGML_AME_PANEL_LOG",
            "GGML_AME_PANEL_PROGRESS_LOG", "GGML_AME_TILE_PROGRESS_LOG",
            "GGML_XSAI_OP_PROGRESS_LOG",
        ):
            self.assertEqual(env[name], "0", name)
        self.assertEqual(env["GGML_AME_PROFILE"], "1")
        self.assertEqual(env["GGML_XSAI_OP_PROFILE"], "1")
        self.assertNotIn("--progress", self.argv_log.read_text().splitlines())

    def test_realbench_generated_config_allows_explicit_optimized_variant(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(
            XSAI_WORKLOAD="llama-bench-real", XSAI_GGML_AME_PACKED_Q8="1",
            XSAI_GGML_AME_WHOLE_K_Q8="1", XSAI_GGML_AME_F16_BF16="1",
            XSAI_GGML_AME_PROFILE="1", XSAI_GGML_OP_PROFILE="0",
        )
        result = self.run_workload("llama-bench-real", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        for name in ("GGML_AME_PACKED_Q8", "GGML_AME_WHOLE_K_Q8", "GGML_AME_F16_BF16"):
            self.assertEqual(env[name], "1", name)
        self.assertEqual(env["GGML_AME_PROFILE"], "1")
        self.assertEqual(env["GGML_XSAI_OP_PROFILE"], "0")
        self.assertNotIn("--progress", self.argv_log.read_text().splitlines())

    def test_realbench_mx_optins_default_off_and_can_be_explicit(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(XSAI_WORKLOAD="llama-bench-real")
        result = self.run_workload("llama-bench-real", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        self.assertEqual(env["GGML_AME_MX_CONFIG"], "")
        for name in ("GGML_AME_MXFP4", "GGML_AME_MXFP8",
                     "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP4_K64"):
            self.assertEqual(env[name], "0", name)
        config = self.init_config(
            XSAI_WORKLOAD="llama-bench-real",
            XSAI_GGML_AME_MX_CONFIG="CUTE_2Tops_128SCP",
            XSAI_GGML_AME_MXFP4="1", XSAI_GGML_AME_MX32_VERIFIED="1",
            XSAI_GGML_AME_MXFP4_K64="1",
        )
        result = self.run_workload("llama-bench-real", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        self.assertEqual(env["GGML_AME_MX_CONFIG"], "CUTE_2Tops_128SCP")
        for name in ("GGML_AME_MXFP4", "GGML_AME_MX32_VERIFIED", "GGML_AME_MXFP4_K64"):
            self.assertEqual(env[name], "1", name)
        self.assertEqual(env["GGML_AME_MXFP8"], "0")

    def test_realbench_mx_cmdline_overrides_are_exported_without_allocator_change(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(XSAI_WORKLOAD="llama-bench-real")
        result = self.run_workload(
            "llama-bench-real", config=config, **settings,
            XSAI_ALLOC_MODE="devmem",
            cmdline=(
                "xsai.ggml_ame_mx_config=CUTE_2Tops_128SCP "
                "xsai.ggml_ame_mxfp4=0 xsai.ggml_ame_mxfp8=1 "
                "xsai.ggml_ame_mxfp8_format=e5m2 "
                "xsai.ggml_ame_mx32_verified=1 xsai.ggml_ame_mxfp4_k64=1"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        self.assertEqual(env["GGML_AME_MX_CONFIG"], "CUTE_2Tops_128SCP")
        self.assertEqual(env["GGML_AME_MXFP4"], "0")
        self.assertEqual(env["GGML_AME_MXFP8"], "1")
        self.assertEqual(env["GGML_AME_MXFP8_FORMAT"], "e5m2")
        self.assertEqual(env["GGML_AME_MX32_VERIFIED"], "1")
        self.assertEqual(env["GGML_AME_MXFP4_K64"], "1")
        self.assertEqual(env["XSAI_ALLOC_MODE"], "devmem")

    def test_realbench_cmdline_exports_override_config_and_flag_diagnostic_progress(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(
            XSAI_WORKLOAD="llama-bench-real", XSAI_GGML_AME_PACKED_Q8="1",
            XSAI_GGML_AME_WHOLE_K_Q8="1", XSAI_GGML_AME_F16_BF16="1",
        )
        result = self.run_workload(
            "llama-bench-real", config=config, **settings,
            cmdline=(
                "xsai.ggml_ame_packed_q8=0 xsai.ggml_ame_whole_k_q8=0 "
                "xsai.ggml_ame_f16_bf16=0 xsai.ggml_ame_profile=1 "
                "xsai.ggml_op_profile=1 xsai.ggml_ame_log=0 "
                "xsai.llama_full_progress=1"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        for name in ("GGML_AME_PACKED_Q8", "GGML_AME_WHOLE_K_Q8", "GGML_AME_F16_BF16", "GGML_AME_LOG"):
            self.assertEqual(env[name], "0", name)
        self.assertEqual(env["GGML_AME_PROFILE"], "1")
        self.assertEqual(env["GGML_XSAI_OP_PROFILE"], "1")
        self.assertEqual(self.argv_log.read_text().splitlines().count("--progress"), 1)
        self.assertIn("diagnostic_progress=1", result.stdout)

    def test_realbench_generated_stories_boot_smoke_shape_is_not_estimated(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(
            XSAI_WORKLOAD="llama-bench-real", XSAI_LLAMA_FULL_P="128",
            XSAI_LLAMA_FULL_B="128", XSAI_LLAMA_FULL_UB="128",
            XSAI_LLAMA_FULL_N="0", XSAI_LLAMA_FULL_T="1",
            XSAI_LLAMA_FULL_R="1", XSAI_LLAMA_FULL_WARMUP="0",
        )
        result = self.run_workload("llama-bench-real", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.argv_log.read_text().splitlines(),
            ["-m", settings["LLAMA_MODEL"], "-t", "1", "-p", "128", "-n", "0",
             "-r", "1", "-fa", "0", "-ctk", "f16", "-ctv", "f16",
             "-b", "128", "-ub", "128", "-oe", "jsonl", "--no-warmup"],
        )

    def test_historical_fullbench_environment_stays_optimized(self):
        settings = self.prepare_bench_environment()
        config = self.init_config(XSAI_WORKLOAD="llama-bench-full")
        result = self.run_workload("llama-bench-full", config=config, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.bench_environment()
        for name in ("GGML_AME_PACKED_Q8", "GGML_AME_WHOLE_K_Q8", "GGML_AME_F16_BF16"):
            self.assertEqual(env[name], "1", name)
        args = self.argv_log.read_text().splitlines()
        self.assertIn("--fake-like", args)
        self.assertIn("--estimate-layers", args)
        self.assertNotIn("--progress", args)

    def test_llama_realbench_keeps_model_disk_selection_and_explicit_precedence(self):
        settings = self.prepare_bench()
        disk_model = self.models / "model.gguf"
        disk_model.write_text("host shell fixture only")
        for explicit in ("", settings["LLAMA_MODEL"]):
            with self.subTest(explicit=explicit):
                result = self.run_workload(
                    "llama-bench-real", select_disk=True, **settings,
                    MODEL_MNT=str(self.models), XSAI_LLAMA_FULL_MODEL=explicit,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.argv_log.read_text().splitlines()
                self.assertEqual(args[:2], ["-m", explicit or str(disk_model)])

    def test_rootfs_make_serializes_realbench_controls_without_build(self):
        result = subprocess.run(
            ["make", "-n", "-f", str(ROOTFS_MAKEFILE), "xsai-init-config",
             "XSAI_WORKLOAD=llama-bench-real", "XSAI_LLAMA_FULL_T=2",
             "XSAI_LLAMA_FULL_P=256", "XSAI_LLAMA_FULL_N=8",
             "XSAI_LLAMA_FULL_B=256", "XSAI_LLAMA_FULL_UB=256",
             "XSAI_LLAMA_FULL_R=4", "XSAI_LLAMA_FULL_WARMUP=0",
             "XSAI_LLAMA_FULL_PROGRESS=1",
             "XSAI_LLAMA_FULL_FA=1", "XSAI_LLAMA_FULL_TYPE_K=bf16",
             "XSAI_LLAMA_FULL_TYPE_V=bf16", "XSAI_LLAMA_FULL_PREFLIGHT=1",
             "XSAI_LLAMA_FULL_MODEL=/models/real.gguf", "XSAI_PLDM_EXIT_MARKER=1",
             "XSAI_GGML_AME_PACKED_Q8=0", "XSAI_GGML_AME_WHOLE_K_Q8=0",
             "XSAI_GGML_AME_F16_BF16=0", "XSAI_GGML_AME_PROFILE=1",
             "XSAI_GGML_OP_PROFILE=1"],
            cwd=self.scratch, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in (
            ("WORKLOAD", "llama-bench-real"), ("XSAI_LLAMA_FULL_T", "2"),
            ("XSAI_LLAMA_FULL_P", "256"), ("XSAI_LLAMA_FULL_N", "8"),
            ("XSAI_LLAMA_FULL_B", "256"), ("XSAI_LLAMA_FULL_UB", "256"),
            ("XSAI_LLAMA_FULL_R", "4"), ("XSAI_LLAMA_FULL_WARMUP", "0"),
            ("XSAI_LLAMA_FULL_PROGRESS", "1"),
            ("XSAI_LLAMA_FULL_FA", "1"), ("XSAI_LLAMA_FULL_TYPE_K", "bf16"),
            ("XSAI_LLAMA_FULL_TYPE_V", "bf16"), ("XSAI_LLAMA_FULL_PREFLIGHT", "1"),
            ("XSAI_LLAMA_FULL_MODEL", "/models/real.gguf"), ("XSAI_PLDM_EXIT_MARKER", "1"),
        ):
            self.assertIn(f'''printf "{key}='%s'\\n" "{value}"''', result.stdout)
        for key, value in (
            ("GGML_AME_PACKED_Q8", "0"), ("GGML_AME_WHOLE_K_Q8", "0"),
            ("GGML_AME_F16_BF16", "0"), ("GGML_AME_PROFILE", "1"),
            ("GGML_XSAI_OP_PROFILE", "1"),
        ):
            self.assertIn(f'''printf "export {key}='%s'\\n" "{value}"''', result.stdout)
        self.assertFalse((self.scratch / "rootfsimg/build/xsai-init.conf").exists())


if __name__ == "__main__":
    unittest.main()
