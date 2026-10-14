#!/usr/bin/env python3
"""Regression gates for the default and optional AI manifest boundary."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOTFSIMG = Path(__file__).resolve().parents[1]
CHECKER = ROOTFSIMG / "check-manifest.py"


def entries(name):
    return {
        fields[1]: fields[2]
        for line in (ROOTFSIMG / name).read_text().splitlines()
        if (fields := line.split()) and fields[0] == "file"
    }


class AiManifestTests(unittest.TestCase):
    def test_default_gate_and_no_cache_models(self):
        default = entries("initramfs-disk-xsai.txt")
        self.assertIn("/bin/test-backend-ops", default)
        self.assertNotIn("/bin/llama-cli", default)
        self.assertNotIn("/bin/run_test.out", default)
        self.assertNotIn("/bin/MNNV2Basic.out", default)
        self.assertNotIn("/bin/ncnn_ame_smoke", default)
        for source in default.values():
            self.assertNotIn("bf16-root", source)
            self.assertNotIn("/apps/mnn/models/", source)
            self.assertNotIn("/apps/ncnn/models/", source)

    def test_additive_manifests_have_disjoint_install_outputs(self):
        default = entries("initramfs-disk-xsai.txt")
        mnn = entries("initramfs-xsai-mnn.txt")
        ncnn = entries("initramfs-xsai-ncnn.txt")
        self.assertEqual(len(mnn), 6)
        self.assertEqual(len(ncnn), 3)
        self.assertFalse(default.keys() & mnn.keys())
        self.assertFalse(default.keys() & ncnn.keys())
        self.assertFalse(mnn.keys() & ncnn.keys())
        for source in list(mnn.values()) + list(ncnn.values()):
            self.assertTrue(source.startswith("${RISCV_ROOTFS_HOME}/rootfsimg/build/"))

    def test_checker_rejects_missing_and_duplicate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            source = scratch / "source"
            source.write_text("fixture")
            manifest = scratch / "manifest"
            manifest.write_text("file /bin/demo ${CHECK_FIXTURE}/source 755 0 0\n")
            env = dict(os.environ, CHECK_FIXTURE=directory)

            def run():
                return subprocess.run(
                    [sys.executable, str(CHECKER), str(manifest)],
                    env=env, capture_output=True, text=True,
                )

            self.assertEqual(run().returncode, 0)
            manifest.write_text(manifest.read_text() * 2)
            self.assertIn("duplicate destination", run().stderr)
            source.unlink()
            self.assertIn("missing or unreadable source", run().stderr)
            self.assertNotEqual(run().returncode, 0)


if __name__ == "__main__":
    unittest.main()
