from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening
from skimage import measure, morphology

from .models import MeshingConfig, IntensityBin
from . import volume as volmod
from . import bins as binsmod


# ---------------------------------------------------------------------------
# Global / per-bin mask building
# ---------------------------------------------------------------------------


def build_global_mask(
    volume_color: np.ndarray,
    cfg: MeshingConfig,
) -> np.ndarray:
    """
    Build a cleaned binary mask from a color or grayscale volume.

    Steps (inspired by the legacy CT_to_3D.py 'build_mask' logic):
      1. Convert to grayscale if needed.
      2. Gaussian smoothing to reduce noise.
      3. Intensity threshold to separate foreground from background.
      4. Morphological closing and opening to remove specks and holes.
      5. Remove small connected components (< min_component_size).

    Parameters
    ----------
    volume_color:
        4D volume [Z, Y, X, 3] or 3D [Z, Y, X], typically uint8.
    cfg:
        MeshingConfig providing thresholds and parameters.

    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] where True means foreground.
    """
    if volume_color.ndim == 4 and volume_color.shape[-1] == 3:
        gray = volmod.to_grayscale(volume_color)
    elif volume_color.ndim == 3:
        gray = volume_color.astype(np.uint8)
    else:
        raise ValueError(
            f"build_global_mask expects [Z, Y, X] or [Z, Y, X, 3], got {volume_color.shape}"
        )

    # Smooth to reduce noise
    smooth = gaussian_filter(gray.astype(np.float32), sigma=cfg.smoothing_sigma)

    # Threshold: foreground = intensities above non_black_threshold
    mask = smooth > float(cfg.non_black_threshold)

    # Morphological cleanup
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)

    # Remove small connected components
    mask = morphology.remove_small_objects(mask, min_size=cfg.min_component_size)

    if np.count_nonzero(mask) == 0:
        raise RuntimeError(
            "Global mask is empty after cleaning. "
            "Try lowering 'non_black_threshold' or 'min_component_size'."
        )

    return mask


def clean_bin_mask(
    bin_mask: np.ndarray,
    cfg: MeshingConfig,
) -> np.ndarray:
    """
    Clean a per-bin mask via connected-component analysis.

    If cfg.adaptive_component_filtering is True:
      - Find the largest connected component size L.
      - Compute an effective min size:
           eff_min_size = max(cfg.min_component_size, int(0.01 * L))
      - Remove components smaller than eff_min_size.

    Otherwise:
      - Remove components smaller than cfg.min_component_size.
    """
    if bin_mask.ndim != 3:
        raise ValueError(
            f"clean_bin_mask expects a 3D mask [Z, Y, X], got {bin_mask.shape}"
        )

    labels = measure.label(bin_mask, connectivity=1)
    if labels.max() == 0:
        # no components; return all False
        return np.zeros_like(bin_mask, dtype=bool)

    # Compute component sizes
    props = measure.regionprops(labels)
    sizes = np.array([p.area for p in props], dtype=np.int64)
    largest = int(sizes.max())

    if cfg.adaptive_component_filtering:
        eff_min_size = max(cfg.min_component_size, int(0.01 * largest))
    else:
        eff_min_size = cfg.min_component_size

    cleaned = morphology.remove_small_objects(labels, min_size=eff_min_size) > 0
    return cleaned


def smooth_mask(
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Smooth a binary mask using a Gaussian filter and re-threshold at 0.5.

    This is mainly to make the isosurface from marching cubes smoother.

    Parameters
    ----------
    mask:
        Boolean mask [Z, Y, X].
    sigma:
        Standard deviation for Gaussian smoothing (in voxels).

    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] after smoothing and thresholding.
    """
    if sigma <= 0.0:
        return mask

    smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return smoothed > 0.5


# ---------------------------------------------------------------------------
# Mesh extraction & saving
# ---------------------------------------------------------------------------


def extract_mesh(
    mask: np.ndarray,
    volume_color: Optional[np.ndarray],
    cfg: MeshingConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract a surface mesh from a binary mask using marching cubes.

    Strategy:
      - Use marching cubes on the binary mask (float, level=0.5).
      - Use voxel coordinates (spacing = 1,1,1) to sample colors if a color
        volume is provided.
      - Scale vertices by cfg.spacing to get real-world coordinates.

    Parameters
    ----------
    mask:
        Boolean mask [Z, Y, X].
    volume_color:
        Optional 4D [Z, Y, X, 3] uint8 volume to sample vertex colors from.
        If None, no colors are returned.
    cfg:
        MeshingConfig providing voxel spacing.

    Returns
    -------
    (vertices, faces, colors)
        vertices:
            Array [N, 3] of vertex coordinates in physical units.
        faces:
            Array [M, 3] of int indices.
        colors:
            Optional array [N, 3] of uint8 RGB colors, or None.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"extract_mesh expects a 3D mask [Z, Y, X], got {mask.shape}"
        )

    # marching_cubes expects values, not booleans
    vol_f = mask.astype(np.float32)

    # Run marching cubes in voxel coordinates (spacing=1,1,1)
    verts_vox, faces, _normals, _values = measure.marching_cubes(
        vol_f,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
    )

    # Sample colors if volume_color is provided
    colors = None
    if volume_color is not None:
        if volume_color.ndim != 4 or volume_color.shape[-1] != 3:
            raise ValueError(
                f"volume_color must be [Z, Y, X, 3], got {volume_color.shape}"
            )

        zmax, ymax, xmax, _ = volume_color.shape
        # verts_vox are in (z, y, x)
        z_idx = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, zmax - 1)
        y_idx = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, ymax - 1)
        x_idx = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, xmax - 1)

        colors = volume_color[z_idx, y_idx, x_idx].astype(np.uint8)

    # Scale to physical units using cfg.spacing = (z, y, x)
    sz, sy, sx = cfg.spacing
    scale = np.array([sz, sy, sx], dtype=np.float32)
    verts_phys = verts_vox * scale[None, :]

    return verts_phys.astype(np.float32), faces.astype(np.int32), colors


def save_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Save a mesh as an ASCII PLY file with optional per-vertex RGB colors.

    Parameters
    ----------
    path:
        Destination file path.
    vertices:
        Array [N, 3] of float vertices.
    faces:
        Array [M, 3] of int vertex indices.
    colors:
        Optional array [N, 3] of uint8 RGB colors.
        If None, colors are omitted from the PLY.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    has_colors = colors is not None
    if has_colors:
        if colors.shape != (n_verts, 3):
            raise ValueError(
                f"colors must have shape (N, 3) matching vertices, "
                f"got {colors.shape} for {n_verts} vertices."
            )

    with path.open("w", encoding="utf-8") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        if has_colors:
            for (x, y, z), (r, g, b) in zip(vertices, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in vertices:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # Faces
        for a, b, c in faces:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


# ---------------------------------------------------------------------------
# High-level meshing over bins
# ---------------------------------------------------------------------------


def generate_meshes_for_bins(
    volume_gray: np.ndarray,
    bins: Sequence[IntensityBin],
    cfg: MeshingConfig,
    *,
    volume_color: Optional[np.ndarray] = None,
    global_mask: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    Generate meshes for a list of intensity bins.

    Workflow:
      1. For each enabled bin:
         - Build a bin mask from `volume_gray`.
         - Apply global_mask (if provided).
         - Clean small components (adaptive if enabled).
         - Smooth the mask.
         - Run marching cubes to get a mesh.
         - Sample per-vertex colors if volume_color is given.
         - Save as PLY in cfg.output_dir.

    Parameters
    ----------
    volume_gray:
        3D grayscale volume [Z, Y, X], e.g. from volume.to_grayscale().
    bins:
        Sequence of IntensityBin objects (typically from bins.select_bins()).
    cfg:
        MeshingConfig containing spacing, thresholds, output directory, etc.
    volume_color:
        Optional [Z, Y, X, 3] RGB volume for sampling per-vertex colors.
        If None, meshes are saved without colors.
    global_mask:
        Optional boolean [Z, Y, X] mask to restrict meshing to a region
        (e.g. from build_global_mask()).

    Returns
    -------
    list of Path
        Paths to the generated PLY files.
    """
    if volume_gray.ndim != 3:
        raise ValueError(
            f"generate_meshes_for_bins expects a 3D grayscale volume [Z, Y, X], "
            f"got shape {volume_gray.shape}"
        )

    if cfg.output_dir is None:
        raise ValueError(
            "MeshingConfig.output_dir must be set before generating meshes."
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []

    for bin_ in bins:
        if not bin_.enabled:
            continue

        # 1) Build mask for this bin
        bin_mask = binsmod.build_bin_mask(volume_gray, bin_)

        # Skip empty if configured
        if cfg.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
            continue

        # 2) Apply global mask if provided
        if global_mask is not None:
            if global_mask.shape != bin_mask.shape:
                raise ValueError(
                    "global_mask shape does not match volume shape: "
                    f"{global_mask.shape} vs {bin_mask.shape}"
                )
            bin_mask &= global_mask

        # After applying global mask, maybe it's empty
        if cfg.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
            continue

        # 3) Clean small components
        bin_mask_clean = clean_bin_mask(bin_mask, cfg)

        if cfg.skip_empty_bins and np.count_nonzero(bin_mask_clean) == 0:
            continue

        # 4) Smooth mask
        bin_mask_smooth = smooth_mask(bin_mask_clean, cfg.smoothing_sigma)

        if cfg.skip_empty_bins and np.count_nonzero(bin_mask_smooth) == 0:
            continue

        # 5) Extract mesh
        verts, faces, colors = extract_mesh(bin_mask_smooth, volume_color, cfg)

        if verts.size == 0 or faces.size == 0:
            # No geometry produced (e.g. extremely small bin) – skip
            continue

        # 6) Save mesh
        fname = (
            f"{cfg.output_prefix}_bin_{bin_.index:02d}"
            f"_{bin_.low}_{bin_.high}.ply"
        )
        out_path = output_dir / fname
        save_mesh_ply(out_path, verts, faces, colors)
        outputs.append(out_path)

    return outputs
