#!/usr/bin/env python3
"""The differential comparator must reject empty and non-finite outputs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

COMPARATOR = Path(__file__).resolve().parents[1] / "compare_smoke_bf16.py"


class ComparisonTests(unittest.TestCase):
    def compare(self, reference, candidate):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "ref"
            candidate_path = Path(directory) / "ame"
            reference_path.write_text(reference)
            candidate_path.write_text(candidate)
            return subprocess.run(
                [sys.executable, str(COMPARATOR), str(reference_path), str(candidate_path)],
                text=True, capture_output=True,
            )

    def test_finite_match(self):
        self.assertEqual(self.compare("bf16case 1\n0.5\n", "bf16case 1\n0.5\n").returncode, 0)

    def test_empty_or_missing_cases_fail(self):
        for reference, candidate in [("", ""), ("bf16case 1\n", "bf16case 1\n"),
                                      ("bf16case 1\n1\n", "")]:
            with self.subTest(reference=reference, candidate=candidate):
                self.assertNotEqual(self.compare(reference, candidate).returncode, 0)

    def test_nonfinite_and_mismatched_cases_fail(self):
        for value in ("nan", "inf", "-inf", "99"):
            with self.subTest(value=value):
                self.assertNotEqual(
                    self.compare("bf16case 1\n1\n", f"bf16case 1\n{value}\n").returncode, 0
                )
        self.assertNotEqual(
            self.compare("bf16case 1\n1\n", "bf16case 2\n1\n").returncode, 0
        )


if __name__ == "__main__":
    unittest.main()
