from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

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
        Volume with shape [Z, Y, X, 3] (if as_rgb=True) or [Z, Y, X] (if False).
        dtype=uint8 for standard image formats, dtype=uint16 for DICOM files.

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
          - 4D array: [Z, Y, X, 3] (RGB, uint8 or uint16), or
          - 3D array: [Z, Y, X] (already grayscale).

    Returns
    -------
    np.ndarray
        3D array [Z, Y, X], dtype matches input (uint8 or uint16).
        For uint16 input, preserves full intensity range (0-65535).
        For uint8 input, preserves standard range (0-255).

    Notes
    -----
    For RGB input, this uses a standard luminance conversion:
        Y = 0.299 R + 0.587 G + 0.114 B
    For uint8 input, clamps to 0–255.
    For uint16 input, preserves full range (0-65535).
    """
    if volume.ndim == 3:
        # Already grayscale [Z, Y, X] - preserve dtype
        return volume.copy()

    if volume.ndim != 4 or volume.shape[-1] != 3:
        raise ValueError(
            f"to_grayscale expects volume of shape [Z, Y, X, 3] or [Z, Y, X], "
            f"got shape {volume.shape}"
        )

    # Preserve dtype
    is_uint16 = volume.dtype == np.uint16
    
    rgb = volume.astype(np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    if is_uint16:
        # Preserve full uint16 range (0-65535)
        gray = np.clip(gray, 0, 65535).astype(np.uint16)
    else:
        # Standard uint8 range (0-255)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def to_intensity_max(volume: np.ndarray) -> np.ndarray:
    """
    Extract intensity from RGB volume using maximum channel value.
    
    This is better than luminance conversion for medical imaging with colored
    overlays, as it preserves the highest intensity value regardless of which
    color channel it's in. This prevents information loss when colors represent
    different intensity ranges.

    Parameters
    ----------
    volume:
        Either:
          - 4D array: [Z, Y, X, 3] (RGB, uint8 or uint16), or
          - 3D array: [Z, Y, X] (already grayscale).

    Returns
    -------
    np.ndarray
        3D array [Z, Y, X], dtype matches input (uint8 or uint16).
        For uint16 input, preserves full intensity range (0-65535).
        For uint8 input, preserves standard range (0-255).

    Notes
    -----
    For RGB input, this takes the maximum value across R, G, B channels:
        I = max(R, G, B)
    This preserves the highest intensity value, which is important when
    colors represent different intensity ranges in medical imaging.
    """
    if volume.ndim == 3:
        # Already grayscale [Z, Y, X] - preserve dtype
        return volume.copy()

    if volume.ndim != 4 or volume.shape[-1] != 3:
        raise ValueError(
            f"to_intensity_max expects volume of shape [Z, Y, X, 3] or [Z, Y, X], "
            f"got shape {volume.shape}"
        )

    # Take maximum across RGB channels
    # This preserves the highest intensity value regardless of color channel
    intensity = np.max(volume, axis=-1)
    
    # Preserve dtype (no conversion needed, max preserves dtype)
    return intensity


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


# ---------------------------------------------------------------------------
# Canonical Volume Persistence
# ---------------------------------------------------------------------------

def prepare_volume_data_for_canonical(
    data: np.ndarray,
    prefer_int16: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Prepare volume data for CanonicalVolume, converting to int16 if possible.
    
    Attempts to fit data into int16 range (-32768 to +32767). If data is out of range
    and prefer_int16 is True, returns float32 version and raises a ValueError asking
    for confirmation. If prefer_int16 is False, converts to float32.
    
    Parameters
    ----------
    data : np.ndarray
        3D volume array [Z, Y, X] of any numeric dtype.
    prefer_int16 : bool
        If True, prefer int16 and raise ValueError if data doesn't fit.
        If False, convert to float32 if data doesn't fit in int16.
        
    Returns
    -------
    (converted_data, intensity_kind)
        converted_data : np.ndarray
            Data converted to int16 or float32 as appropriate.
        intensity_kind : str
            "HU" if converted to int16, "HU_float" if float32.
            
    Raises
    ------
    ValueError
        If prefer_int16 is True and data doesn't fit in int16 range.
        The error message will indicate that float32 should be used instead.
    """
    data_min = float(data.min())
    data_max = float(data.max())
    
    # Check if data fits in int16 range
    int16_min = -32768
    int16_max = 32767
    
    if data_min >= int16_min and data_max <= int16_max:
        # Fits in int16 - convert to int16
        return data.astype(np.int16), "HU"
    else:
        # Doesn't fit in int16
        if prefer_int16:
            raise ValueError(
                f"Volume data range [{data_min:.2f}, {data_max:.2f}] exceeds int16 range "
                f"[{int16_min}, {int16_max}]. Cannot use int16 dtype. "
                f"Please confirm use of float32 dtype instead."
            )
        else:
            # Convert to float32
            return data.astype(np.float32), "HU_float"


@dataclass
class CanonicalVolume:
    """
    Canonical representation of a 3D medical imaging volume.
    
    This is the ground truth data structure for voxel volumes in CT23D.
    Volumes are stored in array order [Z, Y, X] but spacing/origin/direction
    are in physical space order (x, y, z).
    
    Attributes
    ----------
    data : np.ndarray
        3D numpy array with shape (Z, Y, X) and dtype float32 or int16.
        Array order matches the order slices are loaded (Z = slice index).
    spacing : Tuple[float, float, float]
        Voxel spacing in millimeters as (sx, sy, sz) in physical space order.
        Note: This is (x, y, z) spacing, but data is stored in (Z, Y, X) order.
    origin : Tuple[float, float, float], optional
        Origin position in millimeters as (ox, oy, oz) in physical space.
        Defaults to (0, 0, 0).
    direction : np.ndarray, optional
        3x3 orientation matrix (direction cosines) as numpy array.
        Defaults to identity matrix (no rotation).
    intensity_kind : str
        Type of intensity values, e.g., "HU" (Hounsfield Units) or "raw".
        Use "HU" if values have been converted to Hounsfield Units.
    provenance : Dict
        Dictionary storing preprocessing steps, source info, timestamps, version.
        Used for tracking data transformations and reproducibility.
    """
    data: np.ndarray
    spacing: Tuple[float, float, float]  # (sx, sy, sz) in physical space order
    intensity_kind: str
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Optional[np.ndarray] = None
    provenance: Dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if self.data.ndim != 3:
            raise ValueError(
                f"CanonicalVolume.data must be 3D [Z, Y, X], got shape {self.data.shape}"
            )
        
        if self.data.dtype not in (np.float32, np.int16):
            raise ValueError(
                f"CanonicalVolume.data dtype must be float32 or int16, got {self.data.dtype}"
            )
        
        if len(self.spacing) != 3:
            raise ValueError(f"spacing must have 3 elements, got {len(self.spacing)}")
        
        if len(self.origin) != 3:
            raise ValueError(f"origin must have 3 elements, got {len(self.origin)}")
        
        if self.direction is None:
            # Default to identity matrix
            self.direction = np.eye(3, dtype=np.float64)
        else:
            direction = np.asarray(self.direction)
            if direction.shape != (3, 3):
                raise ValueError(
                    f"direction must be a 3x3 matrix, got shape {direction.shape}"
                )
            self.direction = direction.astype(np.float64)


def save_volume_nrrd(volume: CanonicalVolume, path: str | Path) -> None:
    """
    Save a CanonicalVolume to NRRD format.
    
    Parameters
    ----------
    volume : CanonicalVolume
        The volume to save.
    path : str | Path
        Output file path. Should have .nrrd or .nhdr extension.
        For compressed output, use .nrrd (pynrrd will handle compression).
        
    Notes
    -----
    - Spacing and origin are saved in NRRD header.
    - Direction matrix is saved if not identity; otherwise stored in sidecar JSON.
    - Provenance is saved to a sidecar JSON file with same basename.
    - Uses gzip compression by default if path ends with .nrrd.
    """
    try:
        import nrrd
    except ImportError:
        raise ImportError(
            "pynrrd is required for NRRD I/O. Install with: pip install pynrrd"
        )
    
    path = Path(path)
    
    # Convert data to appropriate format for NRRD
    # NRRD works well with float32 and int16
    data = volume.data
    
    # NRRD expects data in (X, Y, Z) order for some operations, but we store (Z, Y, X)
    # pynrrd should handle this correctly when we specify spacing in header
    # However, we need to transpose for NRRD which expects (X, Y, Z) order
    # Actually, let me check pynrrd behavior - it might preserve order correctly
    # For now, we'll transpose to (X, Y, Z) which is what NRRD typically expects
    data_nrrd = np.transpose(data, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    
    # Prepare header
    header = {}
    
    # Spacing: NRRD uses space directions, but also supports spacing
    # NRRD spacing is in (x, y, z) order, which matches our spacing tuple
    sx, sy, sz = volume.spacing
    header['space directions'] = [
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ]
    
    # Origin: NRRD uses space origin in (x, y, z) order
    ox, oy, oz = volume.origin
    header['space origin'] = [ox, oy, oz]
    
    # Check if direction is identity
    is_identity = np.allclose(volume.direction, np.eye(3))
    
    # Type information
    if data.dtype == np.float32:
        header['type'] = 'float'
    elif data.dtype == np.int16:
        header['type'] = 'short'
    else:
        # Convert to float32 if unknown type
        data_nrrd = data_nrrd.astype(np.float32)
        header['type'] = 'float'
    
    # Encoding: use gzip if .nrrd extension (not .nhdr)
    if path.suffix == '.nrrd':
        header['encoding'] = 'gzip'
    else:
        header['encoding'] = 'raw'
    
    # Custom fields for intensity_kind
    header['intensity_kind'] = volume.intensity_kind
    
    # Save direction in sidecar if not identity (NRRD doesn't directly support custom direction matrices)
    sidecar_data = {}
    if not is_identity:
        # Store direction matrix in sidecar JSON
        sidecar_data['direction'] = volume.direction.tolist()
    
    # Write NRRD file
    nrrd.write(str(path), data_nrrd, header=header)
    
    # Save provenance and direction (if not identity) to sidecar JSON
    if volume.provenance or sidecar_data:
        sidecar_path = path.with_suffix('.json')
        sidecar_dict = {
            'provenance': volume.provenance,
            **sidecar_data
        }
        save_provenance(sidecar_path, sidecar_dict)


def load_volume_nrrd(path: str | Path) -> CanonicalVolume:
    """
    Load a CanonicalVolume from NRRD format.
    
    Parameters
    ----------
    path : str | Path
        Path to .nrrd or .nhdr file.
        
    Returns
    -------
    CanonicalVolume
        The loaded volume with all metadata.
        
    Notes
    -----
    - Loads spacing and origin from NRRD header.
    - Loads direction matrix from sidecar JSON if present, otherwise assumes identity.
    - Loads provenance from sidecar JSON if present.
    """
    try:
        import nrrd
    except ImportError:
        raise ImportError(
            "pynrrd is required for NRRD I/O. Install with: pip install pynrrd"
        )
    
    path = Path(path)
    
    # Load NRRD file
    data_nrrd, header = nrrd.read(str(path))
    
    # Transpose from (X, Y, Z) back to (Z, Y, X) for our convention
    data = np.transpose(data_nrrd, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
    
    # Extract spacing from space directions
    if 'space directions' in header:
        space_dirs = header['space directions']
        # Extract diagonal elements (assuming no rotation in spacing)
        sx = abs(space_dirs[0][0])
        sy = abs(space_dirs[1][1])
        sz = abs(space_dirs[2][2])
        spacing = (sx, sy, sz)
    else:
        # Fallback: assume unit spacing
        spacing = (1.0, 1.0, 1.0)
    
    # Extract origin
    if 'space origin' in header:
        origin_list = header['space origin']
        origin = tuple(float(x) for x in origin_list)
    else:
        origin = (0.0, 0.0, 0.0)
    
    # Extract intensity_kind
    intensity_kind = header.get('intensity_kind', 'raw')
    
    # Load direction and provenance from sidecar JSON
    sidecar_path = path.with_suffix('.json')
    direction = None
    provenance = {}
    
    if sidecar_path.exists():
        sidecar_dict = load_provenance(sidecar_path)
        provenance = sidecar_dict.get('provenance', {})
        if 'direction' in sidecar_dict:
            direction = np.array(sidecar_dict['direction'], dtype=np.float64)
    
    # If no direction in sidecar, use identity
    if direction is None:
        direction = np.eye(3, dtype=np.float64)
    
    return CanonicalVolume(
        data=data,
        spacing=spacing,
        origin=origin,
        direction=direction,
        intensity_kind=intensity_kind,
        provenance=provenance
    )


def save_provenance(path_base: str | Path, provenance_dict: Dict) -> None:
    """
    Save provenance/metadata dictionary to a JSON sidecar file.
    
    Parameters
    ----------
    path_base : str | Path
        Base path. The function will save to {path_base}.json
        If path_base already has .json extension, it's used as-is.
    provenance_dict : Dict
        Dictionary to save (will be JSON serialized).
    """
    path = Path(path_base)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(provenance_dict, f, indent=2, ensure_ascii=False)


def load_provenance(path_base: str | Path) -> Dict:
    """
    Load provenance/metadata dictionary from a JSON sidecar file.
    
    Parameters
    ----------
    path_base : str | Path
        Base path. The function will load from {path_base}.json
        If path_base already has .json extension, it's used as-is.
        
    Returns
    -------
    Dict
        Loaded dictionary. Returns empty dict if file doesn't exist.
    """
    path = Path(path_base)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    
    if not path.exists():
        return {}
    
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    """
    Test script demonstrating roundtrip save/load of CanonicalVolume.
    
    Usage:
        python -m ct23d.core.volume
    """
    import tempfile
    
    print("=" * 70)
    print("Testing prepare_volume_data_for_canonical()")
    print("=" * 70)
    
    # Test 1: Data that fits in int16 (typical CT HU range)
    print("\n1. Testing data that fits in int16 range (typical CT HU: -1000 to +3000)...")
    test_data_int16 = np.random.randint(-1000, 3001, size=(5, 10, 10), dtype=np.int32)
    try:
        converted, intensity_kind = prepare_volume_data_for_canonical(test_data_int16, prefer_int16=True)
        assert converted.dtype == np.int16, f"Expected int16, got {converted.dtype}"
        assert intensity_kind == "HU", f"Expected 'HU', got '{intensity_kind}'"
        print(f"   ✓ Successfully converted to int16, intensity_kind='{intensity_kind}'")
        print(f"   Data range: [{converted.min()}, {converted.max()}]")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        exit(1)
    
    # Test 2: Data that exceeds int16 (should raise ValueError with prefer_int16=True)
    print("\n2. Testing data that exceeds int16 range (should ask for confirmation)...")
    test_data_overflow = np.random.randint(-50000, 50001, size=(5, 10, 10), dtype=np.int32)
    try:
        converted, intensity_kind = prepare_volume_data_for_canonical(test_data_overflow, prefer_int16=True)
        print(f"   ✗ Should have raised ValueError, but succeeded instead")
        exit(1)
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError (asking for confirmation)")
        print(f"   Error message: {str(e)[:80]}...")
    except Exception as e:
        print(f"   ✗ Raised unexpected exception: {e}")
        exit(1)
    
    # Test 3: Data that exceeds int16, but prefer_int16=False (should convert to float32)
    print("\n3. Testing data that exceeds int16 with prefer_int16=False (auto float32)...")
    try:
        converted, intensity_kind = prepare_volume_data_for_canonical(test_data_overflow, prefer_int16=False)
        assert converted.dtype == np.float32, f"Expected float32, got {converted.dtype}"
        assert intensity_kind == "HU_float", f"Expected 'HU_float', got '{intensity_kind}'"
        print(f"   ✓ Successfully converted to float32, intensity_kind='{intensity_kind}'")
        print(f"   Data range: [{converted.min():.2f}, {converted.max():.2f}]")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        exit(1)
    
    # Test 4: Float data that fits in int16 range
    print("\n4. Testing float data that fits in int16 range...")
    test_data_float = np.random.uniform(-1000, 3000, size=(5, 10, 10)).astype(np.float32)
    try:
        converted, intensity_kind = prepare_volume_data_for_canonical(test_data_float, prefer_int16=True)
        assert converted.dtype == np.int16, f"Expected int16, got {converted.dtype}"
        assert intensity_kind == "HU", f"Expected 'HU', got '{intensity_kind}'"
        print(f"   ✓ Successfully converted float to int16, intensity_kind='{intensity_kind}'")
        print(f"   Data range: [{converted.min()}, {converted.max()}]")
        print(f"   Note: Float values are truncated to integers (expected behavior)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        exit(1)
    
    print("\n" + "=" * 70)
    print("All prepare_volume_data_for_canonical() tests passed!")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Testing CanonicalVolume save/load roundtrip...")
    print("=" * 70)
    
    # Create synthetic test data
    print("\n1. Creating synthetic volume...")
    z_size, y_size, x_size = 10, 20, 30
    test_data = np.random.randint(0, 1000, size=(z_size, y_size, x_size), dtype=np.int16)
    spacing = (1.0, 0.5, 0.5)  # (sx, sy, sz) in mm
    origin = (10.0, 20.0, 30.0)  # (ox, oy, oz) in mm
    direction = np.eye(3)  # Identity matrix
    intensity_kind = "HU"
    provenance = {
        "source": "test_synthetic",
        "preprocessing_steps": ["test_step1", "test_step2"],
        "version": "0.5.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    original_volume = CanonicalVolume(
        data=test_data,
        spacing=spacing,
        origin=origin,
        direction=direction,
        intensity_kind=intensity_kind,
        provenance=provenance
    )
    
    print(f"   Created volume with shape: {original_volume.data.shape}")
    print(f"   Spacing: {original_volume.spacing}")
    print(f"   Origin: {original_volume.origin}")
    print(f"   Intensity kind: {original_volume.intensity_kind}")
    
    # Save to temporary file
    print("\n2. Saving volume to NRRD format...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_volume.nrrd"
        save_volume_nrrd(original_volume, test_path)
        print(f"   Saved to: {test_path}")
        
        # Check if sidecar JSON was created
        sidecar_path = test_path.with_suffix('.json')
        if sidecar_path.exists():
            print(f"   Sidecar JSON created: {sidecar_path}")
        
        # Load back
        print("\n3. Loading volume from NRRD format...")
        loaded_volume = load_volume_nrrd(test_path)
        print(f"   Loaded volume with shape: {loaded_volume.data.shape}")
        print(f"   Spacing: {loaded_volume.spacing}")
        print(f"   Origin: {loaded_volume.origin}")
        print(f"   Intensity kind: {loaded_volume.intensity_kind}")
        
        # Verify roundtrip
        print("\n4. Verifying roundtrip...")
        
        # Check data
        data_match = np.array_equal(original_volume.data, loaded_volume.data)
        print(f"   Data matches: {data_match}")
        if not data_match:
            print(f"   Original data dtype: {original_volume.data.dtype}, shape: {original_volume.data.shape}")
            print(f"   Loaded data dtype: {loaded_volume.data.dtype}, shape: {loaded_volume.data.shape}")
            print(f"   Max difference: {np.abs(original_volume.data.astype(np.float32) - loaded_volume.data.astype(np.float32)).max()}")
        
        # Check spacing
        spacing_match = np.allclose(original_volume.spacing, loaded_volume.spacing)
        print(f"   Spacing matches: {spacing_match}")
        if not spacing_match:
            print(f"   Original spacing: {original_volume.spacing}")
            print(f"   Loaded spacing: {loaded_volume.spacing}")
        
        # Check origin
        origin_match = np.allclose(original_volume.origin, loaded_volume.origin)
        print(f"   Origin matches: {origin_match}")
        if not origin_match:
            print(f"   Original origin: {original_volume.origin}")
            print(f"   Loaded origin: {loaded_volume.origin}")
        
        # Check direction
        direction_match = np.allclose(original_volume.direction, loaded_volume.direction)
        print(f"   Direction matches: {direction_match}")
        
        # Check intensity_kind
        intensity_match = original_volume.intensity_kind == loaded_volume.intensity_kind
        print(f"   Intensity kind matches: {intensity_match}")
        
        # Check provenance
        provenance_match = original_volume.provenance == loaded_volume.provenance
        print(f"   Provenance matches: {provenance_match}")
        if not provenance_match:
            print(f"   Original provenance: {original_volume.provenance}")
            print(f"   Loaded provenance: {loaded_volume.provenance}")
        
        # Overall result
        all_match = (data_match and spacing_match and origin_match and 
                    direction_match and intensity_match and provenance_match)
        print(f"\n{'✓' if all_match else '✗'} Roundtrip test: {'PASSED' if all_match else 'FAILED'}")
        
        if all_match:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed. Please review the output above.")
            exit(1)
