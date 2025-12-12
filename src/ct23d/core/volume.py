from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from . import images as imgio


def load_volume_from_dir(
    slices_dir: Path,
    *,
    as_rgb: bool = True,
) -> np.ndarray:
    """
    Load a directory of slice images into a volume.

    Parameters
    ----------
    slices_dir:
        Directory containing processed slice images. The caller is responsible
        for resolving this path appropriately (project-relative, config-relative, etc.).
    as_rgb:
        If True, the output is a 4D volume [Z, Y, X, 3] (RGB).
        If False, the output is a 3D volume [Z, Y, X] (grayscale converted).

    Returns
    -------
    np.ndarray
        Volume with shape [Z, Y, X, 3] (if as_rgb=True) or [Z, Y, X] (if False),
        dtype=uint8.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no valid slice images.
    ValueError
        If slice shapes are inconsistent.
    """
    slices_dir = Path(slices_dir)
    files = imgio.list_slice_files(slices_dir)
    rgb_volume = imgio.load_slices_to_volume(files)  # [Z, Y, X, 3]

    if as_rgb:
        return rgb_volume

    gray_volume = to_grayscale(rgb_volume)
    return gray_volume


def to_grayscale(volume: np.ndarray) -> np.ndarray:
    """
    Convert a volume to grayscale [Z, Y, X].

    Parameters
    ----------
    volume:
        Either:
          - 4D array: [Z, Y, X, 3] (RGB, uint8), or
          - 3D array: [Z, Y, X] (already grayscale).

    Returns
    -------
    np.ndarray
        3D array [Z, Y, X], dtype=uint8.

    Notes
    -----
    For RGB input, this uses a standard luminance conversion:
        Y = 0.299 R + 0.587 G + 0.114 B
    and clamps to 0–255.
    """
    if volume.ndim == 3:
        # Already grayscale [Z, Y, X]
        return volume.astype(np.uint8)

    if volume.ndim != 4 or volume.shape[-1] != 3:
        raise ValueError(
            f"to_grayscale expects volume of shape [Z, Y, X, 3] or [Z, Y, X], "
            f"got shape {volume.shape}"
        )

    rgb = volume.astype(np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def normalize_intensity(
    volume: np.ndarray,
    *,
    in_range: Tuple[int, int] | None = None,
    out_dtype: type = np.float32,
) -> np.ndarray:
    """
    Normalize a grayscale volume to [0, 1] (or another dtype range).

    Parameters
    ----------
    volume:
        3D [Z, Y, X] grayscale volume, or 4D [Z, Y, X, C] treated channel-wise.
    in_range:
        (low, high) input range. If None, the min and max of the volume are used.
    out_dtype:
        Output dtype, typically np.float32.

    Returns
    -------
    np.ndarray
        Volume with values in [0, 1] (for float dtypes) or scaled accordingly.

    Notes
    -----
    This helper is optional for the meshing pipeline; some marching cubes
    implementations like values roughly between 0 and 1.
    """
    vol = volume.astype(np.float32)

    if in_range is None:
        vmin = float(vol.min())
        vmax = float(vol.max())
    else:
        vmin, vmax = map(float, in_range)

    if vmax <= vmin:
        # Avoid division by zero; return zeros
        return np.zeros_like(vol, dtype=out_dtype)

    norm = (vol - vmin) / (vmax - vmin)
    return norm.astype(out_dtype)


def apply_mask(
    volume: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a boolean mask to a volume: set masked-out voxels to zero.

    Parameters
    ----------
    volume:
        3D [Z, Y, X] or 4D [Z, Y, X, C] array.
    mask:
        Boolean array [Z, Y, X]. True means "keep", False means "zero-out".

    Returns
    -------
    np.ndarray
        New volume with the same shape as input, where masked-out voxels
        have been set to zero.
    """
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D [Z, Y, X], got shape {mask.shape}")

    if volume.shape[:3] != mask.shape:
        raise ValueError(
            f"Volume spatial shape {volume.shape[:3]} does not match "
            f"mask shape {mask.shape}"
        )

    vol = volume.copy()

    if vol.ndim == 3:
        # [Z, Y, X]
        vol[~mask] = 0
    elif vol.ndim == 4:
        # [Z, Y, X, C] -> broadcast mask along last axis
        vol[~mask, :] = 0
    else:
        raise ValueError(
            f"Expected volume ndim 3 or 4, got {vol.ndim}"
        )

    return vol


def crop_to_nonzero(
    volume: np.ndarray,
    margin: int = 0,
) -> tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    Crop a volume to the minimal bounding box containing all non-zero voxels.

    Parameters
    ----------
    volume:
        3D [Z, Y, X] or 4D [Z, Y, X, C] array.
    margin:
        Extra voxels to keep around the bounding box in each direction.
        Can be zero or positive.

    Returns
    -------
    (cropped_volume, slices)
        cropped_volume:
            The cropped volume (same ndim as input).
        slices:
            A tuple of slices (sz, sy, sx) that can be used to map coordinates
            back to the original volume.

    Notes
    -----
    If the volume is entirely zero, the original volume and full slices are
    returned.
    """
    if volume.ndim not in (3, 4):
        raise ValueError(
            f"crop_to_nonzero expects volume with ndim 3 or 4, got {volume.ndim}"
        )

    # Work on magnitude across channels if needed
    if volume.ndim == 4:
        # [Z, Y, X, C] -> aggregate via max across channels
        nonzero_mask = np.any(volume != 0, axis=-1)
    else:
        nonzero_mask = volume != 0

    coords = np.argwhere(nonzero_mask)

    if coords.size == 0:
        # All zeros: return original volume and identity slices
        sz = slice(0, volume.shape[0])
        sy = slice(0, volume.shape[1])
        sx = slice(0, volume.shape[2])
        return volume, (sz, sy, sx)

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)

    z_max = min(z_max + margin, volume.shape[0] - 1)
    y_max = min(y_max + margin, volume.shape[1] - 1)
    x_max = min(x_max + margin, volume.shape[2] - 1)

    sz = slice(z_min, z_max + 1)
    sy = slice(y_min, y_max + 1)
    sx = slice(x_min, x_max + 1)

    if volume.ndim == 4:
        cropped = volume[sz, sy, sx, :]
    else:
        cropped = volume[sz, sy, sx]

    return cropped, (sz, sy, sx)
