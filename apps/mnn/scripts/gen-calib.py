#!/usr/bin/env python3
"""Generate deterministic BMP calibration images for MNN PTQ.

Prefers real-image statistics: derives 8 crops/flips from the two sample
photos bundled in the MNN repo (resource/images). Falls back to synthetic
gradients+noise when those photos are unavailable. stb_image (the calibration
reader) handles 24-bit BMP directly, so BMP is the only format we emit.
"""

import argparse
import os
import struct


def lcg_bytes(seed):
    while True:
        seed = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield (seed >> 33) & 0xFF


def write_bmp(path, size, rgb_rows):
    width = height = size
    row_raw = width * 3
    padding = (4 - (row_raw % 4)) % 4
    row_size = row_raw + padding
    pixel_bytes = row_size * height
    header = b"BM" + struct.pack("<IHHI", 54 + pixel_bytes, 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0,
                      pixel_bytes, 2835, 2835, 0, 0)
    rows = []
    for y in range(height - 1, -1, -1):  # BMP stores rows bottom-up
        row = bytearray()
        for x in range(width):
            r, g, b = rgb_rows[y][x]
            row += bytes((b, g, r))  # BMP stores BGR
        row += b"\x00" * padding
        rows.append(bytes(row))
    with open(path, "wb") as f:
        f.write(header + dib + b"".join(rows))


def synthetic_rows(size, seed):
    noise = lcg_bytes(seed)
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            r = int(255 * x / max(size - 1, 1))
            g = int(255 * y / max(size - 1, 1))
            b = (x + y) % 256
            n = next(noise)
            row.append(((r + n) % 256, (g + n) % 256, (b + n) % 256))
        rows.append(row)
    return rows


def real_rows(photo_path, size, variant):
    from PIL import Image
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    # 4 deterministic center-crop zoom levels; odd variants are mirrored.
    zooms = [1.0, 1.3, 1.7, 2.2]
    zoom = zooms[variant % len(zooms)]
    cw, ch = int(w / zoom), int(h / zoom)
    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    if variant >= len(zooms):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((size, size), Image.BILINEAR)
    data = list(img.getdata())
    rows = [data[y * size:(y + 1) * size] for y in range(size)]
    return rows, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23457)
    parser.add_argument("--photos", default="",
                        help="comma-separated real photos to derive crops from")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    photos = [p for p in args.photos.split(",") if p and os.path.isfile(p)]
    for i in range(args.count):
        path = os.path.join(args.out_dir, "xsai-calib-%03d.bmp" % i)
        if photos:
            rows, _ = real_rows(photos[i % len(photos)], args.size, i)
        else:
            rows = synthetic_rows(args.size, args.seed + i * 7919)
        write_bmp(path, args.size, rows)
        print(path)


if __name__ == "__main__":
    main()
