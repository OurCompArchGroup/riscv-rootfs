#!/usr/bin/env python3
"""Compare two MNNV2Basic output.txt dumps (whitespace-separated floats)."""

import sys


def load(path):
    with open(path) as f:
        return [float(x) for x in f.read().split()]


def main():
    if len(sys.argv) != 3:
        print("usage: compare-outputs.py int8-output.txt fp32-output.txt")
        return 2
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    if len(a) != len(b):
        print("MNN_COMPARE_FAIL length mismatch: %d vs %d" % (len(a), len(b)))
        return 1
    diffs = [abs(a[i] - b[i]) for i in range(len(a))]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    ia = max(range(len(a)), key=lambda i: a[i])
    ib = max(range(len(b)), key=lambda i: b[i])
    print("MNN_COMPARE max_abs_diff=%.6f mean_abs_diff=%.6f" % (max_diff, mean_diff))
    print("MNN_COMPARE argmax int8=%d (%.4f) fp32=%d (%.4f)" % (ia, a[ia], ib, b[ib]))
    if ia != ib:
        print("MNN_COMPARE_FAIL argmax mismatch")
        return 1
    print("MNN_COMPARE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
