#!/usr/bin/env python3
"""Check that each initramfs `file` entry has a readable source."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    errors = []
    destinations = set()
    files = 0
    for manifest in args.manifests:
        for number, raw in enumerate(manifest.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] != "file":
                continue
            if len(fields) != 6:
                errors.append(f"{manifest}:{number}: malformed file entry")
                continue
            _, destination, source, mode, uid, gid = fields
            if destination in destinations:
                errors.append(f"{manifest}:{number}: duplicate destination {destination}")
            destinations.add(destination)
            resolved = Path(os.path.expandvars(source))
            if "$" in str(resolved) or not resolved.is_file() or not os.access(resolved, os.R_OK):
                errors.append(f"{manifest}:{number}: missing or unreadable source {resolved}")
            files += 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {files} initramfs file sources exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
