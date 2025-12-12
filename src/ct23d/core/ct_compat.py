from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .models import PreprocessConfig, MeshingConfig, ProjectConfig
from . import preprocessing
from . import volume as volmod
from . import bins as binsmod
from . import meshing as meshmod


Spacing3D = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_spacing(s: str) -> Spacing3D:
    """
    Parse a spacing string 'Z,Y,X' into a 3-tuple of floats.

    This mirrors the behavior of the legacy CT_to_3D.py script's parser.
    """
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Spacing must be 'Z,Y,X', got '{s}'."
        )
    try:
        z, y, x = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse spacing from '{s}'."
        ) from exc
    return (z, y, x)


def _compute_output_dir_and_prefix(output: Path) -> tuple[Path, str]:
    """
    Decide output directory and filename prefix from the legacy '--output' arg.

    Legacy behavior (approximate):
      - If 'output' has a file suffix (e.g. .ply):
          * output_dir = output.parent
          * prefix     = output.stem
      - Else:
          * output_dir = output
          * prefix     = 'ct_layer'
    """
    output = Path(output)

    if output.suffix:
        output_dir = output.parent
        prefix = output.stem
    else:
        output_dir = output
        prefix = "ct_layer"

    return output_dir, prefix


# ---------------------------------------------------------------------------
# Programmatic legacy-style pipeline
# ---------------------------------------------------------------------------

def run_legacy_pipeline(
    input_dir: Path,
    output: Path,
    *,
    spacing: Spacing3D = (1.6, 1.0, 1.0),
    grayscale_tolerance: int = 1,
    saturation_threshold: float = 0.08,
    remove_bed: bool = True,
    non_black_threshold: int = 15,
    min_component_size: int = 500,
    smoothing_sigma: float = 1.0,
    n_bins: int = 6,
    min_intensity: int = 1,
    max_intensity: int = 255,
    output_prefix: Optional[str] = None,
    use_cache: bool = True,
) -> List[Path]:
    """
    Legacy-style one-shot pipeline: from input directory of CT slices
    to a set of PLY meshes, using the new ct23d.core modules.

    This is meant to mirror the behavior of the original CT_to_3D.py script,
    but implemented on top of the refactored core.

    Parameters
    ----------
    input_dir:
        Directory containing raw CT slice images.
    output:
        Either:
          - Path to a .ply file (only its parent / stem are used), or
          - Path to an output directory (no suffix).
    spacing:
        Voxel spacing (Z, Y, X) in mm.
    grayscale_tolerance, saturation_threshold, remove_bed:
        Preprocessing options for overlay/bed removal.
    non_black_threshold, min_component_size, smoothing_sigma:
        Meshing/mask parameters.
    n_bins, min_intensity, max_intensity:
        Intensity binning parameters.
    output_prefix:
        Optional override for the mesh file prefix.
        If None, it is derived from `output` as in the legacy script.
    use_cache:
        If True, reuse processed slices if they exist.

    Returns
    -------
    list of Path
        List of generated mesh file paths.
    """
    input_dir = Path(input_dir)

    # Decide output directory and prefix similarly to the legacy script
    output_dir, default_prefix = _compute_output_dir_and_prefix(output)
    prefix = output_prefix or default_prefix

    # ---------------- PreprocessConfig ----------------
    pre_cfg = PreprocessConfig(
        input_dir=input_dir,
        processed_dir=input_dir / "processed_slices",
        use_cache=use_cache,
        grayscale_tolerance=grayscale_tolerance,
        saturation_threshold=saturation_threshold,
        remove_bed=remove_bed,
    )

    # ---------------- MeshingConfig ----------------
    mesh_cfg = MeshingConfig(
        spacing=spacing,
        non_black_threshold=non_black_threshold,
        min_component_size=min_component_size,
        smoothing_sigma=smoothing_sigma,
        n_bins=n_bins,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        output_dir=output_dir,
        output_prefix=prefix,
        adaptive_component_filtering=True,
        skip_empty_bins=True,
    )

    # ---------------- ProjectConfig wrapper ----------------
    project = ProjectConfig(
        name="CT23D Legacy Project",
        preprocess=pre_cfg,
        meshing=mesh_cfg,
        bins=[],  # use uniform bins by default
        config_path=None,
    )

    # 1) Preprocess slices (overlay removal + optional bed removal)
    processed_dir = preprocessing.preprocess_slices(project.preprocess)

    # 2) Load volume (RGB + grayscale)
    volume_color = volmod.load_volume_from_dir(processed_dir, as_rgb=True)
    volume_gray = volmod.to_grayscale(volume_color)

    # 3) Build global mask (optional but generally helpful)
    global_mask = meshmod.build_global_mask(volume_color, project.meshing)

    # 4) Prepare bins (uniform or custom)
    bins_list = binsmod.select_bins(project)

    # 5) Generate meshes
    outputs = meshmod.generate_meshes_for_bins(
        volume_gray=volume_gray,
        bins=bins_list,
        cfg=project.meshing,
        volume_color=volume_color,
        global_mask=global_mask,
    )

    return outputs


# ---------------------------------------------------------------------------
# CLI main() – approximate legacy CT_to_3D.py interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build an argparse.ArgumentParser approximating the original CT_to_3D CLI.
    """
    parser = argparse.ArgumentParser(
        description="Convert CT slice images to colored PLY meshes (legacy-style)."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CT slice images (JPEG/PNG).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ct_body_mesh.ply"),
        help=(
            "Output path. If a .ply file, its parent & stem define the "
            "output directory and prefix. If a directory, meshes are saved "
            "there with a default prefix (ct_layer)."
        ),
    )
    parser.add_argument(
        "--spacing",
        type=parse_spacing,
        default=(1.6, 1.0, 1.0),
        metavar="Z,Y,X",
        help="Voxel spacing in mm as 'Z,Y,X' (default: 1.6,1.0,1.0).",
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
            "HSV saturation threshold above which pixels are treated as "
            "colored overlays (default: 0.08)."
        ),
    )
    parser.add_argument(
        "--no-bed-removal",
        action="store_true",
        help="Disable automatic bed/headrest removal.",
    )
    parser.add_argument(
        "--non-black-threshold",
        type=int,
        default=15,
        help=(
            "Intensity threshold below which voxels are treated as background "
            "(default: 15)."
        ),
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=500,
        help=(
            "Minimum connected component size (in voxels) to keep during mask "
            "cleaning (default: 500)."
        ),
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian smoothing before thresholding (default: 1.0).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=6,
        help=(
            "Number of grayscale bins to convert to separate meshes "
            "(default: 6)."
        ),
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
        default=None,
        help=(
            "Prefix for output PLY files. If omitted, derived from --output "
            "as in the legacy script."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not reuse existing 'processed_slices' folder; always reprocess.",
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """
    CLI entrypoint approximating the original CT_to_3D.py main().

    It delegates all real work to `run_legacy_pipeline`.
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_dir: Path = args.input_dir
    output: Path = args.output

    remove_bed = not args.no_bed_removal
    use_cache = not args.no_cache

    outputs = run_legacy_pipeline(
        input_dir=input_dir,
        output=output,
        spacing=args.spacing,
        grayscale_tolerance=args.grayscale_tolerance,
        saturation_threshold=args.saturation_threshold,
        remove_bed=remove_bed,
        non_black_threshold=args.non_black_threshold,
        min_component_size=args.min_component_size,
        smoothing_sigma=args.smoothing_sigma,
        n_bins=args.n_bins,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
        output_prefix=args.output_prefix,
        use_cache=use_cache,
    )

    if outputs:
        print("Generated the following meshes:")
        for p in outputs:
            # Print as relative path to current working dir for readability
            print("  -", p)
    else:
        print("No meshes were generated (all bins empty or filtered out).")
