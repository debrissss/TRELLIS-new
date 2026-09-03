#!/usr/bin/env python3
"""Extract only model/up.color files to lossless RGB PNG images.

The source files contain one raw 1280x720 8-bit BGR image per file.
The output is written beside each source as model/up.png.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
CHANNELS = 3
EXPECTED_BYTES = WIDTH * HEIGHT * CHANNELS


def extract_up_color(dataset_root: Path) -> int:
    """Convert every */model/up.color below dataset_root and return the count."""
    sources = sorted(dataset_root.glob("*/model/up.color"))
    if not sources:
        raise FileNotFoundError(f"No */model/up.color files found under {dataset_root}")

    converted = 0
    for source in sources:
        raw = np.fromfile(source, dtype=np.uint8)
        if raw.size != EXPECTED_BYTES:
            raise ValueError(
                f"Unexpected size for {source}: {raw.size} bytes; "
                f"expected {EXPECTED_BYTES}"
            )

        # up.color is BGR. OpenCV writes this array as a standard RGB PNG.
        image_bgr = raw.reshape(HEIGHT, WIDTH, CHANNELS)
        destination = source.with_name("up.png")
        temporary = source.with_name("up.part.png")
        if temporary.exists():
            temporary.unlink()

        ok = cv2.imwrite(
            str(temporary),
            image_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not ok:
            raise OSError(f"Could not write {destination}")

        os.replace(temporary, destination)
        converted += 1
        print(f"{converted:02d}/{len(sources)} {destination}")

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract only model/up.color files as 1280x720 RGB PNGs."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("/root/autodl-tmp/TRELLIS-new/面扫测试数据"),
        help="Dataset directory containing the per-ID folders.",
    )
    args = parser.parse_args()
    count = extract_up_color(args.dataset_root)
    print(f"Converted {count} up.color files; other directions were not read.")


if __name__ == "__main__":
    main()
