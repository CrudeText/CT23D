"""
ct23d.core.images

Centralized image utilities for the CT23D project.

Includes:
- slice file listing
- image loading / saving
- volume loading
- preprocessing helpers that operate on RGB numpy arrays
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from PIL import Image

# Allowed extensions for slice images
_IMAGE_EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]

PathLikeOrList = Union[Path, Iterable[Union[Path, str]]]


# -------------------------------------------------------------------------
# Slice discovery
# -------------------------------------------------------------------------
def list_slice_files(source: PathLikeOrList) -> List[Path]:
    """
    Public helper used by preprocessing and meshing.

    It accepts either:
      - a Path to a directory containing slice images
      - or an iterable (list/tuple) of Path/str objects

    Returns a sorted list of slice image paths.
    Raises a clear error if no images are found.
    """
    # Case 1: directory path
    if isinstance(source, Path):
        folder = source

        if not folder.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {folder}")

        paths: List[Path] = []
        for ext in _IMAGE_EXTS:
            paths.extend(folder.glob(f"*{ext}"))

        paths = sorted(paths)
        if not paths:
            raise RuntimeError(
                f"No slice image files found in folder:\n{folder}\n"
                f"Expected extensions: {_IMAGE_EXTS}"
            )
        return paths

    # Case 2: iterable of paths
    paths = [Path(p) for p in source]
    paths = sorted(paths)
    if not paths:
        raise RuntimeError("No slice paths provided to list_slice_files().")
    return paths


# -------------------------------------------------------------------------
# Image IO
# -------------------------------------------------------------------------
def load_image_rgb(path: Path) -> np.ndarray:
    """
    Load an image as an RGB numpy array with dtype uint8.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return arr


def save_image_rgb(path: Path, arr: np.ndarray) -> None:
    """
    Save a 3-channel uint8 numpy array to an image file.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    im = Image.fromarray(arr, mode="RGB")
    im.save(path)


# -------------------------------------------------------------------------
# Volume IO
# -------------------------------------------------------------------------
def load_slices_to_volume(source: PathLikeOrList) -> np.ndarray:
    """
    Load slice images into a 4D RGB volume.

    Parameters
    ----------
    source : Path or iterable of Path/str
        Either:
          - a directory containing slice images, or
          - a list/tuple of image paths.

    Returns
    -------
    np.ndarray
        Array of shape (Z, Y, X, 3), dtype uint8.
    """
    slice_paths = list_slice_files(source)
    slices = [load_image_rgb(p) for p in slice_paths]
    volume = np.stack(slices, axis=0)  # (Z, Y, X, 3)
    return volume.astype(np.uint8, copy=False)


# -------------------------------------------------------------------------
# Preprocessing helpers
# -------------------------------------------------------------------------
def _remove_colored_overlays(
    rgb: np.ndarray,
    grayscale_tolerance: int,
    saturation_threshold: float,
) -> np.ndarray:
    """
    Core logic that removes numbers / colored overlays.

    The approach:
      - Convert to float and compute saturation via (max-min)/max
      - If saturation > threshold → treat pixel as an overlay → desaturate
      - Also remove extreme differences between channels using grayscale tolerance
    """
    arr = rgb.astype(np.float32)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    rng = mx - mn

    # Handle zero max values safely
    sat = np.zeros_like(mx)
    mask = mx > 0
    sat[mask] = rng[mask] / mx[mask]

    # Highly saturated pixels (numbers, UI labels)
    overlay_mask = sat > saturation_threshold

    # Also mask pixels that deviate strongly from grayscale
    dev_mask = rng > grayscale_tolerance

    full_mask = overlay_mask | dev_mask

    # Replace overlay pixels with the mean intensity (grayscale)
    gray = ((r + g + b) / 3.0).astype(np.float32)
    new_arr = arr.copy()
    new_arr[full_mask, 0] = gray[full_mask]
    new_arr[full_mask, 1] = gray[full_mask]
    new_arr[full_mask, 2] = gray[full_mask]

    return new_arr.astype(np.uint8)


def _remove_bed_placeholder(rgb: np.ndarray) -> np.ndarray:
    """
    Placeholder for bed/headrest removal.

    Current version is a no-op (returns input unchanged).
    Later versions will apply cropping / segmentation.
    """
    return rgb


def preprocess_volume_rgb(
    volume_rgb: np.ndarray,
    grayscale_tolerance: int,
    saturation_threshold: float,
    remove_bed: bool,
) -> np.ndarray:
    """
    Apply preprocessing steps slice-by-slice:

      - Overlay removal
      - (Optional) bed removal

    Parameters
    ----------
    volume_rgb : np.ndarray
        Shape (Z, Y, X, 3), dtype uint8
    grayscale_tolerance : int
    saturation_threshold : float
    remove_bed : bool

    Returns
    -------
    np.ndarray
        Processed copy of volume_rgb, same shape/dtype.
    """
    processed = []
    for rgb in volume_rgb:
        cleaned = _remove_colored_overlays(
            rgb,
            grayscale_tolerance=grayscale_tolerance,
            saturation_threshold=saturation_threshold,
        )

        if remove_bed:
            cleaned = _remove_bed_placeholder(cleaned)

        processed.append(cleaned)

    return np.stack(processed, axis=0)
