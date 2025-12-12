from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from ct23d.core.models import PreprocessConfig, ProjectConfig
from ct23d.core import preprocessing


def build_arg_parser() -> argparse.ArgumentParser:
    """
    CLI for preprocessing CT slices:
      - remove colored overlays (numbers, text, markers),
      - optionally remove bed/headrest,
      - save processed slices to a directory.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess CT slice images (overlay & bed removal) for CT23D."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw CT slice images (JPEG/PNG).",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store processed slices. "
            "If omitted, defaults to <input_dir>/processed_slices."
        ),
    )

    parser.add_argument(
        "--grayscale-tolerance",
        type=int,
        default=1,
        help=(
            "Max RGB channel difference to still consider a pixel grayscale "
            "(default: 1)."
        ),
    )

    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.08,
        help=(
            "HSV saturation threshold above which pixels are considered "
            "colored overlays (default: 0.08)."
        ),
    )

    parser.add_argument(
        "--no-bed-removal",
        action="store_true",
        help="Disable automatic bed/headrest removal.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Do not reuse an existing processed_slices directory; "
            "always re-run preprocessing."
        ),
    )

    parser.add_argument(
        "--project-config",
        type=Path,
        default=None,
        help=(
            "Optional project YAML configuration. "
            "NOTE: YAML loading is not implemented yet in this CLI; "
            "this argument is reserved for future use."
        ),
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.project_config is not None:
        parser.error(
            "--project-config is reserved for future use; "
            "YAML-based configs are not implemented yet in this CLI."
        )

    input_dir: Path = args.input_dir
    processed_dir: Optional[Path] = args.processed_dir

    pre_cfg = PreprocessConfig(
        input_dir=input_dir,
        processed_dir=processed_dir,
        use_cache=not args.no_cache,
        grayscale_tolerance=args.grayscale_tolerance,
        saturation_threshold=args.saturation_threshold,
        remove_bed=not args.no_bed_removal,
    )

    project = ProjectConfig(
        name="CT23D Preprocess CLI Project",
        preprocess=pre_cfg,
    )

    out_dir = preprocessing.preprocess_slices(project.preprocess)

    # Print path as given (no forced absolute resolution)
    print("Processed slices directory:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
