from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from ct23d.core.models import MeshingConfig, ProjectConfig
from ct23d.core import volume as volmod
from ct23d.core import bins as binsmod
from ct23d.core import meshing as meshmod

from ct23d.core.ct_compat import parse_spacing  # reuse spacing parser


def build_arg_parser() -> argparse.ArgumentParser:
    """
    CLI for generating 3D meshes from preprocessed CT slices.
    """
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from preprocessed CT slices (CT23D core CLI)."
    )

    parser.add_argument(
        "processed_dir",
        type=Path,
        help="Directory containing preprocessed slice images (e.g. processed_slices).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where PLY meshes will be written.",
    )

    parser.add_argument(
        "--spacing",
        type=parse_spacing,
        default=(1.6, 1.0, 1.0),
        metavar="Z,Y,X",
        help="Voxel spacing in mm as 'Z,Y,X' (default: 1.6,1.0,1.0).",
    )

    parser.add_argument(
        "--non-black-threshold",
        type=int,
        default=15,
        help=(
            "Intensity threshold below which voxels are treated as background "
            "for the global mask (default: 15)."
        ),
    )

    parser.add_argument(
        "--min-component-size",
        type=int,
        default=500,
        help=(
            "Minimum connected component size (in voxels) for mask cleaning "
            "(default: 500)."
        ),
    )

    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.0,
        help=(
            "Sigma for Gaussian smoothing (in voxels) when building masks "
            "(default: 1.0)."
        ),
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=6,
        help="Number of uniform intensity bins to generate (default: 6).",
    )

    parser.add_argument(
        "--min-intensity",
        type=int,
        default=1,
        help="Minimum grayscale value included in binning (default: 1).",
    )

    parser.add_argument(
        "--max-intensity",
        type=int,
        default=255,
        help="Maximum grayscale value considered in binning (default: 255).",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ct_layer",
        help="Prefix for output PLY filenames (default: ct_layer).",
    )

    parser.add_argument(
        "--no-global-mask",
        action="store_true",
        help="Disable the global foreground mask (use per-bin masks only).",
    )

    parser.add_argument(
        "--no-colors",
        action="store_true",
        help=(
            "Do not sample per-vertex colors from the volume; "
            "meshes will be uncolored."
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

    processed_dir: Path = args.processed_dir
    output_dir: Path = args.output_dir

    # ---------------- MeshingConfig ----------------
    mesh_cfg = MeshingConfig(
        spacing=args.spacing,
        non_black_threshold=args.non_black_threshold,
        min_component_size=args.min_component_size,
        smoothing_sigma=args.smoothing_sigma,
        n_bins=args.n_bins,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        adaptive_component_filtering=True,
        skip_empty_bins=True,
    )

    project = ProjectConfig(
        name="CT23D Mesh CLI Project",
        meshing=mesh_cfg,
    )

    # 1) Load volumes
    #    - Always load grayscale for bin masks
    #    - Optionally load RGB for colors
    vol_gray = volmod.load_volume_from_dir(processed_dir, as_rgb=False)

    vol_color = None
    if not args.no_colors:
        vol_color = volmod.load_volume_from_dir(processed_dir, as_rgb=True)

    # 2) Global mask (unless disabled)
    global_mask = None
    if not args.no_global_mask:
        # If we have color volume, use it for nicer global mask
        if vol_color is not None:
            global_mask = meshmod.build_global_mask(vol_color, project.meshing)
        else:
            global_mask = meshmod.build_global_mask(vol_gray, project.meshing)

    # 3) Bins
    #    For now: always uniform bins (no custom presets in this CLI yet)
    bins_list = binsmod.generate_uniform_bins(project.meshing)

    # 4) Meshing
    outputs = meshmod.generate_meshes_for_bins(
        volume_gray=vol_gray,
        bins=bins_list,
        cfg=project.meshing,
        volume_color=vol_color,
        global_mask=global_mask,
    )

    if outputs:
        print("Generated meshes:")
        for p in outputs:
            print(f"  {p}")
    else:
        print("No meshes generated (all bins empty or filtered).")


if __name__ == "__main__":
    main()
