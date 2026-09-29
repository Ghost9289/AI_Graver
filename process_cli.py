from __future__ import annotations

import argparse
from pathlib import Path

from processor import EngravingOptions, process_portrait


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a portrait for Graver")
    parser.add_argument("image", type=Path, help="Source JPG, PNG, BMP, or TIFF")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--contrast", type=int, default=32)
    parser.add_argument("--detail", type=int, default=32)
    parser.add_argument("--shadows", type=int, default=24)
    parser.add_argument("--portrait-mode", choices=("chest", "full"), default="chest")
    parser.add_argument("--keep-background", action="store_true", help="Do not replace light edge background")
    parser.add_argument("--png-only", action="store_true", help="Do not export BMP")
    parser.add_argument(
        "--restoration-mode",
        choices=("natural", "strong", "old_photo"),
        default="natural",
        help="natural keeps the Lanczos upscaler; strong/old_photo use the FSRCNN super-resolution pass",
    )
    args = parser.parse_args()
    result = process_portrait(
        args.image,
        args.output,
        EngravingOptions(
            contrast=args.contrast,
            detail=args.detail,
            shadows=args.shadows,
            black_background=not args.keep_background,
            export_bmp=not args.png_only,
            portrait_mode=args.portrait_mode,
            restoration_mode=args.restoration_mode,
        ),
    )
    print(result.resolve())


if __name__ == "__main__":
    main()
