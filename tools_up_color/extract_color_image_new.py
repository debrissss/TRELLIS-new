#!/usr/bin/env python3
"""Extract valid frames from each raw colorImageNew file.

Each source file has a two-byte ``a0 5a`` header followed by nine
1280x720 8-bit BGR frame slots. In the current dataset the first four
slots contain images and the remaining slots are zero-filled padding.
Each non-empty slot is written beside the source as
colorImageNew_01.png, colorImageNew_02.png, and so on.
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
SLOT_COUNT = 9
HEADER = bytes.fromhex("a05a")
EXPECTED_BYTES = len(HEADER) + SLOT_COUNT * WIDTH * HEIGHT * CHANNELS


def extract_color_image_new(dataset_root: Path) -> int:
    """Extract non-empty frames from every colorImageNew below dataset_root."""
    sources = sorted(dataset_root.glob("*/colorImageNew"))
    if not sources:
        raise FileNotFoundError(f"No */colorImageNew files found under {dataset_root}")

    written = 0
    for source in sources:
        raw = np.fromfile(source, dtype=np.uint8)
        if raw.size != EXPECTED_BYTES:
            raise ValueError(
                f"Unexpected size for {source}: {raw.size} bytes; "
                f"expected {EXPECTED_BYTES}"
            )
        if raw[:2].tobytes() != HEADER:
            raise ValueError(
                f"Unexpected header for {source}: {raw[:2].tobytes().hex()}"
            )

        # The payload is a sequence of independent BGR frames, not one tiled image.
        slots_bgr = raw[2:].reshape(SLOT_COUNT, HEIGHT, WIDTH, CHANNELS)
        valid_slots = [i for i, frame in enumerate(slots_bgr) if np.any(frame)]
        if not valid_slots:
            raise ValueError(f"No non-empty frames found in {source}")

        # Remove stale outputs for zero-filled slots, without touching the raw source.
        for i in range(SLOT_COUNT):
            if i not in valid_slots:
                stale = source.with_name(f"colorImageNew_{i + 1:02d}.png")
                if stale.exists():
                    stale.unlink()

        for i in valid_slots:
            destination = source.with_name(f"colorImageNew_{i + 1:02d}.png")
            temporary = source.with_name(f"colorImageNew_{i + 1:02d}.part.png")
            if temporary.exists():
                temporary.unlink()

            # The raw payload is BGR; OpenCV writes it as a standard RGB PNG.
            ok = cv2.imwrite(
                str(temporary),
                slots_bgr[i],
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not ok:
                raise OSError(f"Could not write {destination}")

            os.replace(temporary, destination)
            written += 1
            print(f"{written:03d} {destination}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract non-empty 1280x720 frames from only colorImageNew files."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("/root/autodl-tmp/TRELLIS-new/面扫测试数据"),
        help="Dataset directory containing the per-ID folders.",
    )
    args = parser.parse_args()
    count = extract_color_image_new(args.dataset_root)
    print(f"Extracted {count} non-empty frames; other source files were not read.")


if __name__ == "__main__":
    main()
