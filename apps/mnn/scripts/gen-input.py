#!/usr/bin/env python3
"""Generate the deterministic input_0.txt for MNNV2Basic (text floats).

The guest smoke test classifies the MNN repo's sample cat photo: the packed
tensor is the photo resized to 224x224, scaled to [0,1] (the domain the ONNX
zoo MobileNetV2 expects and the calibration pretreat produces), laid out in
the model's NHWC input order. Falls back to a synthetic gradient when the
photo is unavailable.
"""

import argparse

NORMAL = 1.0 / 255.0


def photo_rows(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return list(img.getdata())


def synthetic_rows(size, seed):
    def lcg(seed):
        while True:
            seed = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            yield (seed >> 33) & 0xFF
    noise = lcg(seed)
    rows = []
    for y in range(size):
        for x in range(size):
            n = next(noise)
            rows.append(((x * 7 + n) % 256, (y * 5 + n) % 256, ((x + y) * 3 + n) % 256))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--photo", default="",
                        help="real photo to encode (preferred when present)")
    parser.add_argument("--seed", type=int, default=23457)
    args = parser.parse_args()

    import os
    if args.photo and os.path.isfile(args.photo):
        pixels = photo_rows(args.photo, args.size)
    else:
        pixels = synthetic_rows(args.size, args.seed)

    with open(args.out, "w") as f:
        for r, g, b in pixels:
            f.write("%.6g %.6g %.6g\n" % (r * NORMAL, g * NORMAL, b * NORMAL))
    print("%s: %d NHWC pixels" % (args.out, len(pixels)))


if __name__ == "__main__":
    main()
