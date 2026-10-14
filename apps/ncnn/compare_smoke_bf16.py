#!/usr/bin/env python3
"""Loose cross-process comparator for the ncnn AME bf16 smoke dumps.

The bf16 AME path truncates fp32 operands to the bf16 grid, so its output
cannot match the stock fp32 conv bit-exactly; the tight (few-ulp) gate runs
in-process inside ncnn_ame_smoke against a scalar bf16 reference. This script
is the secondary sanity check: AME bf16 output vs the stock fp32 conv within
bf16 rounding tolerance.

Usage: compare_smoke_bf16.py <ref-bf16.out> <ame-bf16.out> [rtol] [atol]
Exit 0 on pass, 1 on failure (with a short report).
"""
import sys
import math

def load(path):
    cases = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("bf16case"):
                cur = {"hdr": line, "vals": []}
                cases.append(cur)
            elif line and cur is not None:
                cur["vals"].append(float(line))
    return cases

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-2
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2
    ref = load(sys.argv[1])
    ame = load(sys.argv[2])
    if not ref or any(not case["vals"] for case in ref + ame):
        print("FAIL: missing bf16 cases or empty output")
        return 1
    if len(ref) != len(ame):
        print(f"FAIL: case count mismatch ref={len(ref)} ame={len(ame)}")
        return 1
    worst_rel = 0.0
    worst_abs = 0.0
    worst_hdr = ""
    for r, a in zip(ref, ame):
        if r["hdr"] != a["hdr"] or len(r["vals"]) != len(a["vals"]):
            print(f"FAIL: shape mismatch\n  ref: {r['hdr']}\n  ame: {a['hdr']}")
            return 1
        scale = max(abs(v) for v in r["vals"]) or 1.0
        for rv, av in zip(r["vals"], a["vals"]):
            if not math.isfinite(rv) or not math.isfinite(av):
                print(f"FAIL: non-finite output in {r['hdr']}")
                return 1
            ad = abs(rv - av)
            rel = ad / scale
            if rel > worst_rel:
                worst_rel, worst_abs, worst_hdr = rel, ad, r["hdr"]
            if ad > atol + rtol * scale:
                print(f"FAIL: {r['hdr']}\n  ref={rv!r} ame={av!r} abs={ad:.3e} scale={scale:.3e}")
                return 1
    print(f"PASS: {len(ref)} bf16 cases within rtol={rtol} atol={atol} "
          f"(worst rel={worst_rel:.3e} abs={worst_abs:.3e})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
