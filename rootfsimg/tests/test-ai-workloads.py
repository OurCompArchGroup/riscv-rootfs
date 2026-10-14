#!/usr/bin/env python3
"""Exercise workload exit propagation in a host-only fixture filesystem."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parents[1] / "xsai-workloads.sh"


class AiWorkloadTests(unittest.TestCase):
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

    def binary(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def run_workload(self, workload, **overrides):
        script = (
            "set -eu\n"
            'log(){ echo "$*"; }\n'
            'warn(){ echo "$*"; }\n'
            'run_workload_cmd(){ "$@"; }\n'
            + self.source
            + '\nWORKLOAD="' + workload + '"\n'
            + 'run_selected_workload\nexit "$WORKLOAD_STATUS"\n'
        )
        return subprocess.run(
            ["/bin/sh"], input=script, text=True, capture_output=True,
            env={**os.environ, **overrides},
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


if __name__ == "__main__":
    unittest.main()
