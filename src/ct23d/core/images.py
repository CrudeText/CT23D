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
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# Allowed extensions for slice images
_IMAGE_EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".dcm", ".dicom"]

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
def _is_dicom_file(path: Path) -> bool:
    """Check if a file is a DICOM file based on extension."""
    return path.suffix.lower() in [".dcm", ".dicom"]


def get_dicom_z_position(path: Path) -> Optional[float]:
    """
    Extract Z position from a DICOM file.
    
    Parameters
    ----------
    path : Path
        Path to DICOM file
        
    Returns
    -------
    Optional[float]
        Z position (ImagePositionPatient[2] or SliceLocation), or None if not available
    """
    if not HAS_PYDICOM:
        return None
    
    if not _is_dicom_file(path):
        return None
    
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)  # Read metadata only
        
        # Try ImagePositionPatient first (most accurate - 3D position)
        if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient is not None:
            if len(ds.ImagePositionPatient) >= 3:
                return float(ds.ImagePositionPatient[2])  # Z coordinate
        
        # Fallback to SliceLocation
        if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
            return float(ds.SliceLocation)
        
        # Fallback to InstanceNumber (less reliable but sometimes works)
        if hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
            return float(ds.InstanceNumber)
        
        return None
    except Exception:
        return None


def get_dicom_patient_info(path: Path) -> Optional[dict[str, str]]:
    """
    Extract patient information from a DICOM file.
    
    Parameters
    ----------
    path : Path
        Path to DICOM file
        
    Returns
    -------
    Optional[dict[str, str]]
        Dictionary with patient information fields, or None if not a DICOM file or error occurs.
        Keys: 'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex', 
              'StudyDate', 'StudyTime', 'StudyDescription', 'Modality'
    """
    if not HAS_PYDICOM:
        return None
    
    if not _is_dicom_file(path):
        return None
    
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)  # Read metadata only
        
        info: dict[str, str] = {}
        
        # Patient information
        if hasattr(ds, 'PatientName') and ds.PatientName:
            info['PatientName'] = str(ds.PatientName)
        if hasattr(ds, 'PatientID') and ds.PatientID:
            info['PatientID'] = str(ds.PatientID)
        if hasattr(ds, 'PatientBirthDate') and ds.PatientBirthDate:
            info['PatientBirthDate'] = str(ds.PatientBirthDate)
        if hasattr(ds, 'PatientSex') and ds.PatientSex:
            info['PatientSex'] = str(ds.PatientSex)
        
        # Study information
        if hasattr(ds, 'StudyDate') and ds.StudyDate:
            info['StudyDate'] = str(ds.StudyDate)
        if hasattr(ds, 'StudyTime') and ds.StudyTime:
            info['StudyTime'] = str(ds.StudyTime)
        if hasattr(ds, 'StudyDescription') and ds.StudyDescription:
            info['StudyDescription'] = str(ds.StudyDescription)
        if hasattr(ds, 'Modality') and ds.Modality:
            info['Modality'] = str(ds.Modality)
        
        return info if info else None
    except Exception:
        return None


def _load_dicom_image(path: Path, rotation: int = 0) -> np.ndarray:
    """
    Load a DICOM file as a grayscale array preserving full intensity range.
    
    Parameters
    ----------
    path : Path
        Path to DICOM file
    rotation : int
        Rotation angle in degrees (0, 90, 180, 270). Positive = clockwise.
        
    Returns
    -------
    np.ndarray
        Grayscale image array, shape (Y, X), dtype uint16 or int16 depending on data
    """
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom is required to load DICOM files. "
            "Install it with: pip install pydicom"
        )
    
    ds = pydicom.dcmread(str(path))
    
    # Get pixel array
    pixel_array = ds.pixel_array
    
    # Handle different data types
    if pixel_array.dtype == np.uint16:
        arr = pixel_array.astype(np.uint16)
    elif pixel_array.dtype == np.int16:
        arr = pixel_array.astype(np.int16)
    elif pixel_array.dtype == np.uint8:
        arr = pixel_array.astype(np.uint16)  # Promote to uint16 for consistency
    else:
        # Convert to uint16, handling signed/unsigned appropriately
        if pixel_array.dtype.kind == 'i':  # signed integer
            # Shift to make it non-negative if needed
            arr_min = pixel_array.min()
            if arr_min < 0:
                arr = (pixel_array - arr_min).astype(np.uint16)
            else:
                arr = pixel_array.astype(np.uint16)
        else:
            arr = pixel_array.astype(np.uint16)
    
    # Apply rotation if needed
    if rotation != 0:
        # Use scipy for rotation to preserve dtype
        from scipy.ndimage import rotate
        # scipy rotates counter-clockwise, so negate
        arr = rotate(arr, -rotation, reshape=False, order=1, mode='constant', cval=0)
        # Ensure dtype is preserved
        if pixel_array.dtype == np.int16:
            arr = arr.astype(np.int16)
        else:
            arr = arr.astype(np.uint16)
    
    return arr


def load_image_rgb(path: Path, rotation: int = 0) -> np.ndarray:
    """
    Load an image as an RGB numpy array.
    
    For standard image formats (PNG, JPG, etc.), returns uint8 RGB array.
    For DICOM files, returns uint16 RGB array (grayscale replicated to 3 channels)
    preserving the full intensity range.
    
    Parameters
    ----------
    path : Path
        Path to image file
    rotation : int
        Rotation angle in degrees (0, 90, 180, 270). Positive = clockwise.
        
    Returns
    -------
    np.ndarray
        RGB image array, shape (Y, X, 3)
        - dtype uint8 for standard image formats
        - dtype uint16 for DICOM files (preserves full intensity range)
    """
    # Check if it's a DICOM file
    if _is_dicom_file(path):
        # Load DICOM as grayscale with full intensity range
        gray = _load_dicom_image(path, rotation=rotation)
        # Replicate to RGB channels, preserving dtype
        arr = np.stack([gray, gray, gray], axis=-1)
        return arr
    
    # Standard image format - use PIL
    with Image.open(path) as im:
        im = im.convert("RGB")
        if rotation != 0:
            # PIL rotates counter-clockwise, so negate
            im = im.rotate(-rotation, expand=False)
        arr = np.array(im, dtype=np.uint8)
    return arr


def rotate_image_rgb(arr: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate an RGB image array.
    
    Parameters
    ----------
    arr : np.ndarray
        RGB image array, shape (Y, X, 3), dtype uint8, uint16, or int16
    rotation : int
        Rotation angle in degrees (0, 90, 180, 270). Positive = clockwise.
        
    Returns
    -------
    np.ndarray
        Rotated RGB image array, same shape/dtype as input
        For uint16/int16, preserves dtype; for uint8, returns uint8
    """
    if rotation == 0:
        return arr.copy()
    
    # For uint8, use PIL (fast and good quality)
    if arr.dtype == np.uint8:
        from PIL import Image
        im = Image.fromarray(arr, mode="RGB")
        # PIL rotates counter-clockwise, so negate
        im = im.rotate(-rotation, expand=False)
        return np.array(im, dtype=np.uint8)
    
    # For uint16/int16 (DICOM), use scipy to preserve dtype
    from scipy.ndimage import rotate
    
    # scipy rotates counter-clockwise, so negate
    # Rotate each channel separately to preserve RGB structure
    rotated_channels = []
    for channel_idx in range(3):
        channel = arr[:, :, channel_idx]
        rotated_channel = rotate(
            channel, 
            -rotation, 
            reshape=False, 
            order=1,  # Bilinear interpolation
            mode='constant', 
            cval=0
        )
        rotated_channels.append(rotated_channel)
    
    # Stack channels back together
    rotated = np.stack(rotated_channels, axis=-1)
    
    # Preserve original dtype
    return rotated.astype(arr.dtype, copy=False)


def save_image_dicom(path: Path, arr: np.ndarray, reference_dicom: Optional[Path] = None) -> None:
    """
    Save an array as a DICOM file.
    
    Parameters
    ----------
    path : Path
        Output DICOM file path
    arr : np.ndarray
        Image array, shape (Y, X) or (Y, X, 3). If RGB, converts to grayscale.
        Should be uint16 or int16 to preserve full intensity range.
    reference_dicom : Optional[Path]
        Optional reference DICOM file to copy metadata from. If None, creates minimal DICOM.
    """
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom is required to save DICOM files. "
            "Install it with: pip install pydicom"
        )
    
    # Convert to grayscale if RGB
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # Use mean of RGB channels
        arr = arr.mean(axis=-1).astype(arr.dtype)
    
    # Ensure appropriate dtype for DICOM
    if arr.dtype == np.uint8:
        arr = arr.astype(np.uint16)  # Promote to uint16
    elif arr.dtype not in (np.uint16, np.int16):
        arr = arr.astype(np.uint16)
    
    # Create or load DICOM dataset
    if reference_dicom is not None and reference_dicom.exists():
        ds = pydicom.dcmread(str(reference_dicom))
        # Ensure file meta information exists
        if not hasattr(ds, 'file_meta') or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    else:
        # Create minimal DICOM dataset
        ds = pydicom.Dataset()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CT Image Storage
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.Modality = "CT"
        ds.PatientName = ""
        ds.PatientID = ""
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = "1"
        
        # Set file meta information with Transfer Syntax UID
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.is_implicit_VR = False
        ds.is_little_endian = True
    
    # Set pixel data
    ds.PixelData = arr.tobytes()
    ds.Rows = arr.shape[0]
    ds.Columns = arr.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 if arr.dtype == np.uint16 else 1
    
    # Save with explicit encoding parameters
    ds.save_as(str(path), write_like_original=False)


def save_image_rgb(path: Path, arr: np.ndarray, reference_dicom: Optional[Path] = None) -> None:
    """
    Save an image array to a file.
    
    For standard image formats (PNG, JPG, etc.), saves as uint8 RGB.
    For DICOM files (.dcm, .dicom), saves preserving full intensity range.
    
    Parameters
    ----------
    path : Path
        Output file path
    arr : np.ndarray
        Image array, shape (Y, X, 3) for RGB or (Y, X) for grayscale
        Can be uint8 (standard images) or uint16/int16 (DICOM)
    reference_dicom : Optional[Path]
        Optional reference DICOM file for metadata (only used when saving DICOM)
    """
    # Check if output should be DICOM
    if _is_dicom_file(path):
        save_image_dicom(path, arr, reference_dicom=reference_dicom)
        return
    
    # Standard image format - convert to uint8 if needed
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # RGB array
        if arr.dtype != np.uint8:
            # Normalize to uint8 range
            if arr.dtype in (np.uint16, np.int16):
                # Preserve relative intensities by scaling to 0-255
                arr_min = arr.min()
                arr_max = arr.max()
                if arr_max > arr_min:
                    arr = ((arr.astype(np.float32) - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)
        im = Image.fromarray(arr, mode="RGB")
    else:
        # Grayscale array
        if arr.dtype != np.uint8:
            # Normalize to uint8 range
            if arr.dtype in (np.uint16, np.int16):
                arr_min = arr.min()
                arr_max = arr.max()
                if arr_max > arr_min:
                    arr = ((arr.astype(np.float32) - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)
        im = Image.fromarray(arr, mode="L")
        im = im.convert("RGB")
    
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
        Array of shape (Z, Y, X, 3)
        - dtype uint8 for standard image formats
        - dtype uint16 for DICOM files (preserves full intensity range)
    """
    slice_paths = list_slice_files(source)
    slices = [load_image_rgb(p) for p in slice_paths]
    
    # Check if any slices are DICOM (uint16)
    has_dicom = any(s.dtype == np.uint16 for s in slices)
    
    if has_dicom:
        # Ensure all slices are uint16 (promote uint8 if mixed)
        slices = [s.astype(np.uint16) if s.dtype == np.uint8 else s for s in slices]
        volume = np.stack(slices, axis=0)  # (Z, Y, X, 3)
        return volume.astype(np.uint16, copy=False)
    else:
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

    # Preserve original dtype
    if arr.dtype == np.uint16:
        return new_arr.astype(np.uint16)
    else:
        return new_arr.astype(np.uint8)


def _remove_bed_placeholder(rgb: np.ndarray) -> np.ndarray:
    """
    Placeholder for bed/headrest removal.

    Current version is a no-op (returns input unchanged).
    Later versions will apply cropping / segmentation.
    """
    return rgb


def auto_detect_bed_headrest(
    rgb: np.ndarray,
    bottom_region_ratio: float = 0.4,
    min_intensity: int = 150,
    min_size_ratio: float = 0.01,
    max_size_ratio: float = 0.4,
    scan_upward: bool = True,
    grey_tolerance: int = 30,
    aggressivity: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Automatically detect bed/headrest in a CT slice.
    
    IMPORTANT: This function scans from the BOTTOM UP.
    The bed/headrest must be located underneath the body/head for detection to work.
    If the bed is on the side or top, rotate the images first.
    
    The bed/headrest is typically:
    - Located in the bottom portion of the image
    - High intensity (bright white)
    - Can consist of multiple connected components
    - May have grey pixels surrounding or embedded within
    
    Parameters
    ----------
    rgb : np.ndarray
        RGB image array, shape (Y, X, 3), dtype uint8 or uint16
    bottom_region_ratio : float
        Fraction of image height to start scanning from bottom (0-1)
        Default 0.4 means start from bottom 40% and scan upward
    min_intensity : int
        Minimum grayscale intensity to consider (0-255 for uint8, scaled for uint16)
        Default 150 (bright, but not requiring pure white)
        For uint16 (DICOM), this is automatically scaled proportionally
    min_size_ratio : float
        Minimum object size as fraction of image area (0-1)
        Default 0.01 (1% of image) - catches smaller bed parts
    max_size_ratio : float
        Maximum object size as fraction of image area (0-1)
        Default 0.4 (40% of image) - filters out entire patient
    scan_upward : bool
        If True, scan from bottom upward to find all bed components
    grey_tolerance : int
        Intensity difference tolerance for including grey pixels around bed (0-255 for uint8, scaled for uint16)
        Default 30 - includes pixels within 30 grayscale units of bed intensity
        For uint16 (DICOM), this is automatically scaled proportionally
        
    Returns
    -------
    Optional[np.ndarray]
        Boolean mask of detected bed/headrest (including grey pixels), or None if not found
    """
    from skimage import measure, morphology
    from scipy import ndimage
    
    # Convert to grayscale
    gray = (rgb[..., 0].astype(np.float32) + 
           rgb[..., 1].astype(np.float32) + 
           rgb[..., 2].astype(np.float32)) / 3.0
    
    height, width = gray.shape
    total_pixels = height * width
    
    # Detect intensity range and scale thresholds accordingly
    # For uint16 (DICOM), scale thresholds proportionally
    max_possible = 255.0 if rgb.dtype == np.uint8 else 65535.0
    scale_factor = max_possible / 255.0
    
    # Scale thresholds for uint16
    scaled_min_intensity = min_intensity * scale_factor
    scaled_grey_tolerance = grey_tolerance * scale_factor
    scaled_black_threshold = 10.0 * scale_factor
    
    # Remove black background
    black_mask = gray < scaled_black_threshold
    
    # Start from bottom and scan upward
    bottom_start = int(height * (1 - bottom_region_ratio))
    
    # Create mask for bright objects (bed/headrest is typically bright white)
    bright_mask = gray >= scaled_min_intensity
    bright_mask = bright_mask & ~black_mask
    
    if np.sum(bright_mask) == 0:
        return None
    
    # Find all connected components in the image (not just bottom)
    labels = measure.label(bright_mask, connectivity=2)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background
    
    result_mask = np.zeros_like(gray, dtype=bool)
    candidate_objects = []
    
    # First pass: find objects that touch the bottom or are in bottom region
    for label_id in unique_labels:
        obj_mask = (labels == label_id)
        obj_size = np.sum(obj_mask)
        size_ratio = obj_size / total_pixels
        
        # Filter by size
        if size_ratio < min_size_ratio or size_ratio > max_size_ratio:
            continue
        
        obj_coords = np.where(obj_mask)
        if len(obj_coords[0]) == 0:
            continue
        
        obj_y_coords = obj_coords[0]
        obj_y_min = obj_y_coords.min()
        obj_y_max = obj_y_coords.max()
        obj_y_mean = np.mean(obj_y_coords)
        
        # Check if object touches bottom border or is in bottom region
        touches_bottom = np.any(obj_y_coords == height - 1)
        in_bottom_region = obj_y_mean >= bottom_start
        
        if touches_bottom or in_bottom_region:
            candidate_objects.append((obj_mask, obj_size, obj_y_mean, touches_bottom))
    
    if not candidate_objects:
        return None
    
    # Sort by bottom touch and Y position (prefer objects touching bottom, then lower objects)
    candidate_objects.sort(key=lambda x: (x[3], -x[2]), reverse=True)
    
    # Start with bottom-most objects
    base_mask = np.zeros_like(gray, dtype=bool)
    for obj_mask, obj_size, obj_y_mean, touches_bottom in candidate_objects:
        if touches_bottom:
            base_mask = base_mask | obj_mask
            break  # Start with the object touching bottom
    
    if np.sum(base_mask) == 0 and candidate_objects:
        # If no object touches bottom, use the lowest one
        base_mask = candidate_objects[0][0]
    
    # Second pass: scan upward and find connected objects
    if scan_upward:
        # Dilate base mask to find nearby objects
        dilated_base = ndimage.binary_dilation(base_mask, iterations=5)
        
        # Find objects that are close to or overlap with the base
        for obj_mask, obj_size, obj_y_mean, touches_bottom in candidate_objects:
            # Skip if already included
            if np.any(obj_mask & base_mask):
                continue
            
            # Check if object overlaps with dilated base or is directly above it
            overlap = np.sum(obj_mask & dilated_base)
            if overlap > 0:
                # Object is connected to base - include it
                base_mask = base_mask | obj_mask
            else:
                # Check if object is directly above base (vertical alignment)
                base_coords = np.where(base_mask)
                if len(base_coords[0]) > 0:
                    base_x_min = base_coords[1].min()
                    base_x_max = base_coords[1].max()
                    base_y_max = base_coords[0].max()
                    
                    obj_coords = np.where(obj_mask)
                    obj_x_min = obj_coords[1].min()
                    obj_x_max = obj_coords[1].max()
                    obj_y_min = obj_coords[0].min()
                    
                    # Check horizontal overlap and vertical proximity
                    horizontal_overlap = not (obj_x_max < base_x_min or obj_x_min > base_x_max)
                    vertical_proximity = (obj_y_min - base_y_max) < (height * 0.1)  # Within 10% of image height
                    
                    if horizontal_overlap and vertical_proximity:
                        base_mask = base_mask | obj_mask
    
    result_mask = base_mask.copy()
    
    # Third pass: include grey pixels surrounding or embedded in the bed
    if np.sum(result_mask) > 0 and grey_tolerance > 0:
        # Get intensity range of detected bed
        bed_intensities = gray[result_mask]
        if len(bed_intensities) > 0:
            bed_min_intensity = bed_intensities.min()
            bed_max_intensity = bed_intensities.max()
            bed_mean_intensity = bed_intensities.mean()
            
            # Find grey pixels near bed intensity (within tolerance)
            # Use a wider tolerance to catch more grey noise, including dark greys
            # Include all non-black, non-bright pixels that are near the bed intensity range
            # Apply aggressivity multiplier
            grey_tolerance_extended = (scaled_grey_tolerance + 80 * scale_factor) * aggressivity  # Even more aggressive tolerance for dark greys
            grey_mask = (
                (gray >= max(20 * scale_factor, bed_min_intensity - grey_tolerance_extended)) &  # Include darker greys (but not black)
                (gray <= bed_max_intensity + grey_tolerance_extended) &
                ~black_mask &
                (gray < scaled_min_intensity)  # Only grey, not bright white
            )
            
            # Dilate bed mask extremely aggressively to include nearby grey pixels (scaled by aggressivity)
            dilation_iterations = int(20 * aggressivity)
            dilated_bed = ndimage.binary_dilation(result_mask, iterations=dilation_iterations)
            
            # Include grey pixels that are near the bed
            nearby_grey = grey_mask & dilated_bed
            
            # Only include grey pixels that are connected to the bed
            grey_labels = measure.label(nearby_grey, connectivity=2)
            bed_labels = grey_labels[result_mask]
            bed_label_ids = np.unique(bed_labels[bed_labels > 0])
            
            for label_id in bed_label_ids:
                grey_obj = (grey_labels == label_id)
                result_mask = result_mask | grey_obj
            
            # Also include any grey pixels that are completely surrounded by the bed
            # (embedded grey noise) - be extremely aggressive (scaled by aggressivity)
            dilated_bed_inner = ndimage.binary_dilation(result_mask, iterations=int(8 * aggressivity))
            embedded_grey = grey_mask & dilated_bed_inner & ~result_mask
            result_mask = result_mask | embedded_grey
            
            # Find and include small grey elements near the bed (even if not directly connected)
            # This catches small dark grey fragments - be very aggressive (scaled by aggressivity)
            small_grey_elements = grey_mask & dilated_bed & ~result_mask
            if np.sum(small_grey_elements) > 0:
                small_labels = measure.label(small_grey_elements, connectivity=2)
                # Include small grey elements that are very close to the bed
                for label_id in np.unique(small_labels[small_labels > 0]):
                    small_obj = (small_labels == label_id)
                    # Check if this small element is very close to the bed - more aggressive
                    dilated_small = ndimage.binary_dilation(small_obj, iterations=int(6 * aggressivity))
                    if np.sum(dilated_small & result_mask) > 0:
                        result_mask = result_mask | small_obj
    
    # Clean up - extremely aggressive to remove grey noise and small elements (scaled by aggressivity)
    if np.sum(result_mask) > 0:
        # Fill small holes extremely aggressively
        closing_disk_1 = int(10 * aggressivity)
        result_mask = morphology.binary_closing(result_mask, morphology.disk(closing_disk_1))
        # Remove very small disconnected parts (be extremely aggressive with small grey elements)
        min_size_multiplier = 0.15 / aggressivity  # More aggressive = lower threshold
        result_mask = morphology.remove_small_objects(
            result_mask, 
            min_size=max(30, int(total_pixels * min_size_ratio * min_size_multiplier))  # Even lower threshold to remove more small elements
        )
        # Multiple closing operations to smooth edges and remove remaining noise
        result_mask = morphology.binary_closing(result_mask, morphology.disk(int(7 * aggressivity)))
        result_mask = morphology.binary_closing(result_mask, morphology.disk(int(5 * aggressivity)))
        result_mask = morphology.binary_closing(result_mask, morphology.disk(int(3 * aggressivity)))
        
        # Final check: make sure we have something substantial
        if np.sum(result_mask) >= int(total_pixels * min_size_ratio):
            return result_mask
    
    return None


def _remove_non_grayscale(rgb: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """
    Remove all non-grayscale pixels by turning them black based on saturation threshold.
    
    Parameters
    ----------
    rgb : np.ndarray
        RGB image array, shape (Y, X, 3), dtype uint8
    threshold : float
        Saturation threshold (0-1). Pixels with saturation above this are turned black.
        Lower values = more aggressive (removes more color).
        
    Returns
    -------
    np.ndarray
        Processed image with non-grayscale pixels set to black (0,0,0)
    """
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    
    # Compute saturation (same as overlay removal)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    rng = mx - mn
    
    # Handle zero max values safely
    sat = np.zeros_like(mx)
    mask = mx > 0
    sat[mask] = rng[mask] / mx[mask]
    
    # Pixels with saturation above threshold are turned black
    non_grayscale_mask = sat > threshold
    
    # Set non-grayscale pixels to black
    result = arr.copy()
    result[non_grayscale_mask, 0] = 0
    result[non_grayscale_mask, 1] = 0
    result[non_grayscale_mask, 2] = 0
    
    # Preserve original dtype
    if arr.dtype == np.uint16:
        return result.astype(np.uint16)
    else:
        return result.astype(np.uint8)


def _apply_object_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a boolean mask to remove selected objects (set to black).
    
    Parameters
    ----------
    rgb : np.ndarray
        RGB image array, shape (Y, X, 3), dtype uint8
    mask : np.ndarray
        Boolean mask, shape (Y, X), True = pixels to remove
        
    Returns
    -------
    np.ndarray
        Image with masked pixels set to black
    """
    result = rgb.copy()
    result[mask, :] = 0
    return result


def _find_overlapping_objects_in_slice(
    reference_mask: np.ndarray,
    target_slice: np.ndarray,
    min_overlap_ratio: float = 0.1,
    include_bright_objects: bool = True,
    bright_threshold: Optional[int] = None,
    grey_tolerance: int = 30,
) -> np.ndarray:
    """
    Find objects in target slice that overlap with the reference mask.
    
    The overlapping object(s) become the new mask for propagation.
    This allows tracking objects even as they change shape or when other
    objects (like the body) overlap them.
    
    Also detects bright white objects that overlap with deleted regions,
    and includes grey pixels associated with them.
    
    Parameters
    ----------
    reference_mask : np.ndarray
        Boolean mask from previous slice, shape (Y, X)
    target_slice : np.ndarray
        Target slice RGB image, shape (Y, X, 3)
    min_overlap_ratio : float
        Minimum overlap ratio (0-1). Objects must overlap at least this
        much of the reference mask to be considered.
    include_bright_objects : bool
        If True, also detect bright white objects that overlap with reference mask
    bright_threshold : int
        Intensity threshold for bright objects (0-255)
    grey_tolerance : int
        Intensity tolerance for including grey pixels around objects
        
    Returns
    -------
    np.ndarray
        Boolean mask for target slice containing all overlapping objects
    """
    from scipy import ndimage
    from skimage import measure, morphology
    
    # Convert to grayscale
    def to_gray(rgb):
        return (rgb[..., 0].astype(np.float32) + 
                rgb[..., 1].astype(np.float32) + 
                rgb[..., 2].astype(np.float32)) / 3.0
    
    target_gray = to_gray(target_slice)
    
    # Detect intensity range and scale thresholds accordingly
    max_possible = 255.0 if target_slice.dtype == np.uint8 else 65535.0
    scale_factor = max_possible / 255.0
    
    # Scale thresholds for uint16
    if bright_threshold is None:
        bright_threshold = 150  # Default
    scaled_bright_threshold = bright_threshold * scale_factor
    scaled_grey_tolerance = grey_tolerance * scale_factor
    scaled_black_threshold = 10.0 * scale_factor
    
    # Remove black background from consideration
    black_mask = target_gray < scaled_black_threshold
    
    # Dilate reference mask to account for shifts and find nearby objects
    # Reduced iterations for speed (was 10, now 8)
    dilated_ref = ndimage.binary_dilation(reference_mask, iterations=8)
    
    result_mask = np.zeros_like(reference_mask, dtype=bool)
    ref_area = np.sum(reference_mask)
    
    # First: Find all non-black objects that overlap with reference
    non_black_mask = ~black_mask
    
    if np.sum(non_black_mask) > 0:
        # Find connected components in the non-black region
        # Use connectivity=1 (4-connected) instead of 2 (8-connected) for speed
        labels = measure.label(non_black_mask, connectivity=1)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        # Pre-compute dilated_ref area for faster overlap checks
        dilated_ref_area = np.sum(dilated_ref)
        
        for label_id in unique_labels:
            obj_mask = (labels == label_id)
            
            # Quick overlap check - if object doesn't intersect dilated_ref at all, skip
            if np.sum(obj_mask & dilated_ref) == 0:
                continue
            
            # Check if this object overlaps with the reference mask
            overlap = np.sum(obj_mask & dilated_ref)
            overlap_ratio = overlap / max(ref_area, 1)
            
            # Also check overlap with object area (more lenient)
            obj_area = np.sum(obj_mask)
            overlap_ratio_obj = overlap / max(obj_area, 1)
            
            # If there's significant overlap (either way), include this entire object
            if overlap_ratio >= min_overlap_ratio or overlap_ratio_obj >= 0.05:  # 5% of object overlaps
                result_mask = result_mask | obj_mask
    
    # Second: Find bright white objects that overlap with reference (new bed objects)
    if include_bright_objects:
        bright_mask = target_gray >= scaled_bright_threshold
        bright_mask = bright_mask & ~black_mask
        
        if np.sum(bright_mask) > 0:
            # Use connectivity=1 for speed
            bright_labels = measure.label(bright_mask, connectivity=1)
            bright_unique_labels = np.unique(bright_labels)
            bright_unique_labels = bright_unique_labels[bright_unique_labels > 0]
            
            for label_id in bright_unique_labels:
                bright_obj = (bright_labels == label_id)
                
                # Quick overlap check - if object doesn't intersect dilated_ref or result_mask, skip
                if np.sum(bright_obj & dilated_ref) == 0:
                    if np.sum(result_mask) == 0 or np.sum(bright_obj & result_mask) == 0:
                        continue
                
                # Check if this bright object overlaps with reference mask
                overlap = np.sum(bright_obj & dilated_ref)
                overlap_ratio = overlap / max(ref_area, 1)
                
                # Also check overlap with object area
                bright_obj_area = np.sum(bright_obj)
                overlap_ratio_obj = overlap / max(bright_obj_area, 1)
                
                # Also check if it overlaps with already detected objects
                if np.sum(result_mask) > 0:
                    overlap_with_result = np.sum(bright_obj & result_mask)
                    if overlap_with_result > 0:
                        overlap_ratio = 1.0  # Definitely include if overlaps with result
                
                # If there's any overlap, include this bright object (more lenient)
                if overlap_ratio >= min_overlap_ratio or overlap_ratio_obj >= 0.05:
                    result_mask = result_mask | bright_obj
    
    # Third: Include grey pixels associated with detected objects
    if np.sum(result_mask) > 0 and grey_tolerance > 0:
        # Get intensity range of detected objects
        obj_intensities = target_gray[result_mask]
        if len(obj_intensities) > 0:
            obj_min_intensity = obj_intensities.min()
            obj_max_intensity = obj_intensities.max()
            obj_mean_intensity = obj_intensities.mean()
            
            # Find grey pixels near object intensity (within tolerance)
            # Be more aggressive - include darker greys and extend tolerance
            grey_tolerance_extended = scaled_grey_tolerance + 60 * scale_factor  # More aggressive for dark greys
            grey_mask = (
                (target_gray >= max(15 * scale_factor, obj_min_intensity - grey_tolerance_extended)) &  # Include darker greys
                (target_gray <= obj_max_intensity + grey_tolerance_extended) &
                ~black_mask &
                (target_gray < scaled_bright_threshold)  # Only grey, not bright white
            )
            
            # Dilate result mask to include nearby grey pixels
            # Reduced iterations for speed (was 15, now 8)
            dilated_result = ndimage.binary_dilation(result_mask, iterations=8)
            
            # Include grey pixels that are near the objects
            nearby_grey = grey_mask & dilated_result
            
            # Include all nearby grey pixels (simplified - no need for separate labeling)
            # This is faster than doing multiple label operations
            result_mask = result_mask | nearby_grey
    
    # Clean up the result - optimized for speed
    if np.sum(result_mask) > 0:
        # Single closing operation (reduced from 3 separate operations)
        result_mask = morphology.binary_closing(result_mask, morphology.disk(5))
        # Remove very small objects
        min_size = max(50, int(ref_area * 0.02))
        result_mask = morphology.remove_small_objects(result_mask, min_size=min_size)
    
    return result_mask


def _track_mask_across_slices(
    initial_mask: np.ndarray,
    current_slice: np.ndarray,
    previous_slice: np.ndarray,
    overlap_threshold: float = 0.1,
) -> np.ndarray:
    """
    Track a mask from previous slice to current slice using overlapping objects.
    
    Finds objects in current slice that overlap with the mask from previous slice.
    The overlapping object(s) become the new mask for further propagation.
    
    Parameters
    ----------
    initial_mask : np.ndarray
        Boolean mask from previous slice, shape (Y, X)
    current_slice : np.ndarray
        Current slice RGB image, shape (Y, X, 3)
    previous_slice : np.ndarray
        Previous slice RGB image, shape (Y, X, 3) (unused, kept for compatibility)
    overlap_threshold : float
        Minimum overlap ratio (0-1) to propagate mask
        
    Returns
    -------
    np.ndarray
        Boolean mask for current slice containing overlapping objects
    """
    return _find_overlapping_objects_in_slice(
        initial_mask,
        current_slice,
        min_overlap_ratio=overlap_threshold,
    )


def preprocess_volume_rgb(
    volume_rgb: np.ndarray,
    grayscale_tolerance: int,
    saturation_threshold: float,
    remove_bed: bool,
    remove_non_grayscale: bool = False,
    remove_overlays: bool = True,  # New: control automatic overlay removal (grayscale conversion)
    object_mask: Optional[np.ndarray] = None,
    object_mask_slice_index: Optional[int] = None,
    non_grayscale_slice_ranges: Optional[List[Tuple[int, int]]] = None,  # List of (min, max) slice ranges for non-grayscale removal
    object_removal_objects: Optional[List[dict]] = None,  # List of objects with masks and slice ranges
    progress_cb: Optional[Callable[[str, int, int, int], None]] = None,
) -> np.ndarray:
    """
    Apply preprocessing steps slice-by-slice:

      - (Optional) Overlay removal (converts colored overlays to grayscale)
      - (Optional) bed removal
      - (Optional) remove non-grayscale pixels
      - (Optional) remove selected objects using mask tracking

    Parameters
    ----------
    volume_rgb : np.ndarray
        Shape (Z, Y, X, 3), dtype uint8 or uint16
    grayscale_tolerance : int
    saturation_threshold : float
    remove_bed : bool
    remove_non_grayscale : bool
        If True, turn all non-grayscale pixels black
    remove_overlays : bool
        If True, automatically remove colored overlays (text, markers) by converting them to grayscale.
        If False, preserves original colors. Default is True (original behavior).
    object_mask : Optional[np.ndarray]
        Optional 2D boolean mask (Y, X) for selected slice to remove objects.
        Will be tracked across Z-stack using object matching.
    object_mask_slice_index : Optional[int]
        Index of the slice where object_mask was selected (0-based).
        If None, defaults to 0 (first slice).

    Returns
    -------
    np.ndarray
        Processed copy of volume_rgb, same shape/dtype.
    """
    processed = []
    
    # Initialize masks for all slices (None means no mask)
    slice_masks = [None] * len(volume_rgb)
    # Track which slices need non-grayscale removal
    non_grayscale_slices = set()
    
    # Track linear progress counter (1 to total_slices)
    total_slices = len(volume_rgb)
    progress_counter = 0
    
    # Process object removal objects with their slice ranges
    # Make it more aggressive by dilating masks to catch nearby pixels
    if object_removal_objects:
        from scipy import ndimage
        for obj in object_removal_objects:
            obj_mask = obj.get('mask')
            slice_min = obj.get('slice_min', 0)
            slice_max = obj.get('slice_max', total_slices - 1)
            
            if obj_mask is not None:
                # Dilate mask to be more aggressive (catch nearby pixels)
                dilated_mask = ndimage.binary_dilation(obj_mask, iterations=3)
                
                # Clamp slice range
                slice_min = max(0, min(slice_min, total_slices - 1))
                slice_max = max(slice_min, min(slice_max, total_slices - 1))
                
                # Apply dilated mask to slices in range
                for z_idx in range(slice_min, slice_max + 1):
                    if slice_masks[z_idx] is None:
                        slice_masks[z_idx] = dilated_mask.copy()
                    else:
                        # Combine masks (union)
                        slice_masks[z_idx] = slice_masks[z_idx] | dilated_mask
    
    # Legacy support: if object_mask is provided, apply to all slices
    if object_mask is not None:
        for z_idx in range(len(volume_rgb)):
            if slice_masks[z_idx] is None:
                slice_masks[z_idx] = object_mask.copy()
            else:
                slice_masks[z_idx] = slice_masks[z_idx] | object_mask
    
    # Process non-grayscale removal slice ranges
    if non_grayscale_slice_ranges:
        for slice_min, slice_max in non_grayscale_slice_ranges:
            slice_min = max(0, min(slice_min, total_slices - 1))
            slice_max = max(slice_min, min(slice_max, total_slices - 1))
            for z_idx in range(slice_min, slice_max + 1):
                non_grayscale_slices.add(z_idx)
    
    # Legacy support: if remove_non_grayscale is True, apply to all slices
    if remove_non_grayscale:
        for z_idx in range(total_slices):
            non_grayscale_slices.add(z_idx)
    
    # Now process all slices with their masks
    # This applies: overlay removal, non-grayscale removal, and object mask removal
    for z_idx, rgb in enumerate(volume_rgb):
        # Report progress during slice processing (continue linear progress)
        if progress_cb is not None:
            progress_counter += 1
            try:
                progress_cb("processing", progress_counter, total_slices, total_slices)
            except InterruptedError:
                raise  # Re-raise to stop processing
        # Apply non-grayscale removal BEFORE overlay removal if enabled for this slice
        # This ensures we catch all colored pixels before they get converted to grayscale
        if z_idx in non_grayscale_slices:
            rgb = _remove_non_grayscale(rgb, threshold=saturation_threshold)
        
        # Only apply overlay removal (grayscale conversion) if enabled
        if remove_overlays:
            cleaned = _remove_colored_overlays(
                rgb,
                grayscale_tolerance=grayscale_tolerance,
                saturation_threshold=saturation_threshold,
            )
        else:
            cleaned = rgb.copy()

        if remove_bed:
            cleaned = _remove_bed_placeholder(cleaned)
        
        # Apply object mask if available for this slice
        if slice_masks[z_idx] is not None:
            cleaned = _apply_object_mask(cleaned, slice_masks[z_idx])

        processed.append(cleaned)

    return np.stack(processed, axis=0)


# -------------------------------------------------------------------------
# Non-body removal (body mask computation)
# -------------------------------------------------------------------------

def get_dicom_hu_conversion(path: Path) -> Optional[tuple[float, float]]:
    """
    Get RescaleSlope and RescaleIntercept from a DICOM file for HU conversion.
    
    HU = pixel_value * RescaleSlope + RescaleIntercept
    
    Parameters
    ----------
    path : Path
        Path to DICOM file
        
    Returns
    -------
    Optional[tuple[float, float]]
        (RescaleSlope, RescaleIntercept) if available, None otherwise
    """
    if not HAS_PYDICOM or not _is_dicom_file(path):
        return None
    
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        slope = getattr(ds, 'RescaleSlope', None)
        intercept = getattr(ds, 'RescaleIntercept', None)
        if slope is not None and intercept is not None:
            return (float(slope), float(intercept))
    except Exception:
        pass
    return None


def compute_body_mask_2d(
    slice_gray: np.ndarray,
    body_threshold: float,
    closing_radius_px: float = 8.0,
    min_component_size_px: int = 1000,
    outside_only: bool = True,
    center_of_mass_threshold: float = 0.6,
) -> np.ndarray:
    """
    Compute 2D body mask for a single slice.
    
    Parameters
    ----------
    slice_gray : np.ndarray
        Grayscale slice image, shape (Y, X), any numeric dtype
    body_threshold : float
        Intensity threshold for body tissue (in HU if available, otherwise raw intensity)
    closing_radius_px : float
        Closing radius in pixels for morphological operations
    min_component_size_px : int
        Minimum size of connected components to keep (in pixels)
    outside_only : bool
        If True, only remove pixels outside the body (flood fill from borders)
        If False, remove all non-body pixels
    
    Returns
    -------
    np.ndarray
        Boolean mask, shape (Y, X), True = body tissue
    """
    from scipy import ndimage
    from skimage import measure, morphology
    
    # Convert to float32 for processing
    gray_float = slice_gray.astype(np.float32)
    
    # Threshold to get body mask (body = intensities >= threshold)
    body_mask = gray_float >= body_threshold
    
    # Morphological closing to fill small gaps
    if closing_radius_px > 0:
        disk = morphology.disk(max(1, int(round(closing_radius_px))))
        body_mask = morphology.binary_closing(body_mask, disk)
    
    # Remove small objects first
    if min_component_size_px > 0:
        body_mask = morphology.remove_small_objects(body_mask, min_size=min_component_size_px)
    
    # Identify and exclude bedrest (large component in bottom region)
    # This helps when bedrest is connected to the body
    # Use more aggressive criteria for better consistency across slices
    if np.sum(body_mask) > 0:
        labeled = measure.label(body_mask, connectivity=2)
        h, w = body_mask.shape
        bottom_region_start = h - int(h * 0.35)  # Bottom 35% of image
        
        # Find components that are primarily in the bottom region
        for label_id in np.unique(labeled[labeled > 0]):
            component_mask = (labeled == label_id)
            component_pixels = np.sum(component_mask)
            bottom_pixels = np.sum(component_mask[bottom_region_start:h, :])
            
            # Lower threshold for bedrest detection (45% instead of 50%) for consistency
            if component_pixels > min_component_size_px * 1.5:  # Moderate-large component
                bottom_ratio = bottom_pixels / component_pixels if component_pixels > 0 else 0
                if bottom_ratio > 0.45:  # More than 45% in bottom region
                    # Check if it's also close to bottom edge (strong indicator of bedrest)
                    # Use last 30 pixels (or 5% of height, whichever is larger) for edge detection
                    edge_height = max(30, int(h * 0.05))
                    bottom_edge_pixels = np.sum(component_mask[(h - edge_height):h, :])
                    if bottom_edge_pixels > component_pixels * 0.15:  # At least 15% on bottom edge (lowered from 20%)
                        # Remove this component (likely bedrest)
                        body_mask[component_mask] = False
    
    # Now identify the main body component and preserve connected body parts
    # Instead of just keeping the largest, we keep components that are likely body parts
    if np.sum(body_mask) > 0:
        labeled = measure.label(body_mask, connectivity=2)
        if labeled.max() > 0:
            # Find the largest component (main body)
            largest_label = 0
            largest_size = 0
            component_info = {}
            for label_id in np.unique(labeled[labeled > 0]):
                component_mask = (labeled == label_id)
                size = np.sum(component_mask)
                y_coords, x_coords = np.where(component_mask)
                if len(y_coords) > 0:
                    y_min, y_max = y_coords.min(), y_coords.max()
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_center = np.mean(y_coords)
                    x_center = np.mean(x_coords)
                    component_info[label_id] = {
                        'size': size,
                        'y_min': y_min,
                        'y_max': y_max,
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_center': y_center,
                        'x_center': x_center,
                    }
                if size > largest_size:
                    largest_size = size
                    largest_label = label_id
            
            if largest_label > 0:
                # Keep the largest component (main body)
                main_body = component_info[largest_label]
                body_mask = (labeled == largest_label)
                
                # Also keep other components that are likely body parts (limbs, hands, feet)
                # Components are considered body parts if they:
                # 1. Are within vertical range of main body (legs, arms extending down/up)
                # 2. Are horizontally near main body (arms/hands extending sideways)
                # 3. Are not too small relative to main body (at least 1% of main body size)
                # 4. Are not in suspicious locations (very bottom edge, far from body)
                
                main_y_range = (main_body['y_min'], main_body['y_max'])
                main_x_range = (main_body['x_min'], main_body['x_max'])
                
                for label_id, info in component_info.items():
                    if label_id == largest_label:
                        continue
                    
                    # Check if component is vertically aligned with main body
                    # (within extended range to include limbs)
                    y_overlap = (info['y_max'] >= main_y_range[0] - int(h * 0.1) and 
                                info['y_min'] <= main_y_range[1] + int(h * 0.1))
                    
                    # Check if component is horizontally near main body
                    # (within extended range to include extended arms/hands)
                    x_overlap = (info['x_max'] >= main_x_range[0] - int(w * 0.15) and 
                                info['x_min'] <= main_x_range[1] + int(w * 0.15))
                    
                    # If vertically aligned OR horizontally near, likely a body part
                    if y_overlap or x_overlap:
                        # Additional check: avoid components at very bottom edge that are small
                        # (likely bedrest remnants)
                        bottom_edge_y = h - max(20, int(h * 0.05))
                        is_near_bottom_edge = info['y_max'] >= bottom_edge_y
                        is_small = info['size'] < largest_size * 0.02  # Less than 2% of main body
                        
                        # Only exclude if it's at bottom edge AND small AND not horizontally aligned with body
                        # (this preserves hand tips at side edges that might be small)
                        if is_near_bottom_edge and is_small and not x_overlap:
                            # Likely bedrest remnant at bottom, skip it
                            continue
                        
                        # Keep this component (likely a body part, including small hand tips)
                        body_mask = body_mask | (labeled == label_id)
                    
                    # Also keep very small components at side edges if they're horizontally aligned
                    # (hand tips that extend to edges but are very small)
                    elif x_overlap:
                        # Component is horizontally aligned but maybe not vertically
                        # If it's small and at side edge, might be a hand tip - preserve it
                        is_at_side_edge = (info['x_min'] <= int(w * 0.05) or 
                                          info['x_max'] >= w - int(w * 0.05))
                        is_very_small = info['size'] < largest_size * 0.01  # Less than 1% of main body
                        
                        if is_at_side_edge and is_very_small:
                            # Likely a hand tip at edge - preserve it
                            body_mask = body_mask | (labeled == label_id)
            
            # Additional check: if the body has a very large bottom portion that might be bedrest,
            # try to remove it (helps when bedrest is connected to body)
            # Use conservative criteria to avoid removing legs
            component_mask = body_mask.copy()
            h, w = component_mask.shape
            bottom_region_start = h - int(h * 0.35)  # Bottom 35% of image (more conservative)
            bottom_pixels = np.sum(component_mask[bottom_region_start:h, :])
            total_pixels = np.sum(component_mask)
            
            # Only remove if a very large portion (55%+) is in bottom region (more conservative than 45%)
            # This makes it less likely to remove legs
            if total_pixels > 0 and bottom_pixels > total_pixels * 0.55:
                y_coords, x_coords = np.where(component_mask)
                if len(y_coords) > 0:
                    # Calculate vertical center of mass
                    y_center = np.mean(y_coords)
                    # Only remove if center of mass is very low (lower portion of image)
                    # This helps preserve legs which have a higher center of mass
                    # center_of_mass_threshold of 0.6 means center must be in lower 40% of image
                    # Use a slightly higher threshold (0.65) to be more conservative
                    conservative_threshold = max(center_of_mass_threshold, 0.65)
                    if y_center > h * conservative_threshold:
                        # Try to remove bottom portion that's likely bedrest
                        # But only if we still have a reasonable body mask left
                        temp_mask = body_mask.copy()
                        # Remove bottom 30% (more conservative than 35% to preserve more body/legs)
                        temp_bottom_start = h - int(h * 0.3)
                        temp_mask[temp_bottom_start:h, :] = False
                        
                        # Check if we still have a substantial body mask
                        if np.sum(temp_mask) > min_component_size_px:
                            # Re-select largest component
                            labeled = measure.label(temp_mask, connectivity=2)
                            if labeled.max() > 0:
                                largest_label = 0
                                largest_size_new = 0
                                for label_id in np.unique(labeled[labeled > 0]):
                                    size = np.sum(labeled == label_id)
                                    if size > largest_size_new:
                                        largest_size_new = size
                                        largest_label = label_id
                                # Only use the new mask if we still have a reasonable body
                                # (at least 40% of original size - more conservative than 30%)
                                if largest_size_new > min_component_size_px and largest_size_new > total_pixels * 0.4:
                                    body_mask = (labeled == largest_label)
    
    # If outside_only, perform flood fill from borders to remove external air
    # Be careful to avoid removing body parts at edges (legs, hands, etc.)
    if outside_only and np.sum(body_mask) > 0:
        # Create inverse mask (non-body)
        non_body_mask = ~body_mask
        
        # Flood fill from border pixels, but be selective to preserve body parts at edges
        h, w = body_mask.shape
        filled = np.zeros_like(body_mask, dtype=bool)
        
        # Fill from bottom edge (most likely to be bedrest/external)
        # Be very conservative - only fill from bottom 5% to avoid removing legs
        bottom_fill_start = max(0, h - max(5, int(h * 0.05)))
        for y in range(bottom_fill_start, h):
            for x in range(w):
                if non_body_mask[y, x] and not filled[y, x]:
                    # Flood fill this region
                    seed_mask = np.zeros_like(body_mask, dtype=bool)
                    seed_mask[y, x] = True
                    region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                    filled = filled | region
        
        # Fill from top edge (typically no body parts)
        for x in range(w):
            if non_body_mask[0, x] and not filled[0, x]:
                seed_mask = np.zeros_like(body_mask, dtype=bool)
                seed_mask[0, x] = True
                region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                filled = filled | region
        
        # Fill from side edges, but be very conservative - only from very top corners
        # Don't fill from side edges at all to preserve hands/arms that extend to edges
        # Only fill from top corners (first few pixels of top-left and top-right corners)
        corner_limit = max(1, min(3, int(h * 0.02)))  # Top 2% or 3 pixels, whichever is smaller
        for x in [0, w - 1]:
            for y in range(corner_limit):
                if non_body_mask[y, x] and not filled[y, x]:
                    seed_mask = np.zeros_like(body_mask, dtype=bool)
                    seed_mask[y, x] = True
                    region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                    filled = filled | region
        
        # Body is the inverse of filled external regions
        body_mask = ~filled
    
    return body_mask


def compute_body_mask_3d(
    volume_gray: np.ndarray,
    spacing: tuple[float, float, float],
    body_threshold: float,
    closing_radius_mm: float = 8.0,
    min_component_size_vox: int = 1000,
    outside_only: bool = True,
) -> np.ndarray:
    """
    Compute 3D body mask for a volume.
    
    Parameters
    ----------
    volume_gray : np.ndarray
        Grayscale volume, shape (Z, Y, X), any numeric dtype
    spacing : tuple[float, float, float]
        Voxel spacing in mm as (sx, sy, sz) in physical space order
        Note: volume is in (Z, Y, X) order, spacing is (x, y, z)
    body_threshold : float
        Intensity threshold for body tissue (in HU if available, otherwise raw intensity)
    closing_radius_mm : float
        Closing radius in millimeters for morphological operations
    min_component_size_vox : int
        Minimum size of connected components to keep (in voxels)
    outside_only : bool
        If True, only remove pixels outside the body (flood fill from borders)
        If False, remove all non-body pixels
    
    Returns
    -------
    np.ndarray
        Boolean mask, shape (Z, Y, X), True = body tissue
    """
    from scipy import ndimage
    from skimage import measure, morphology
    
    # Convert to float32 for processing
    gray_float = volume_gray.astype(np.float32)
    
    # Threshold to get body mask (body = intensities >= threshold)
    body_mask = gray_float >= body_threshold
    
    # Convert closing radius from mm to voxels using spacing
    # Use average of Y and X spacing (slice plane spacing)
    slice_spacing_avg = (spacing[1] + spacing[2]) / 2.0  # (sy + sx) / 2
    closing_radius_px = closing_radius_mm / slice_spacing_avg if slice_spacing_avg > 0 else closing_radius_mm
    
    # Morphological closing to fill small gaps (2D per slice, faster than 3D)
    if closing_radius_px > 0:
        disk = morphology.disk(max(1, int(round(closing_radius_px))))
        for z_idx in range(body_mask.shape[0]):
            body_mask[z_idx] = morphology.binary_closing(body_mask[z_idx], disk)
    
    # Keep only the largest connected component (the body) - use 3D connectivity
    if min_component_size_vox > 0:
        body_mask = morphology.remove_small_objects(body_mask, min_size=min_component_size_vox, connectivity=3)
    
    # If outside_only, perform 3D flood fill from borders to remove external air
    if outside_only and np.sum(body_mask) > 0:
        # Create inverse mask (non-body)
        non_body_mask = ~body_mask
        
        # Flood fill from all border voxels (3D)
        z_max, y_max, x_max = body_mask.shape
        filled = np.zeros_like(body_mask, dtype=bool)
        
        # Fill from Z borders (top and bottom slices)
        for z in [0, z_max - 1]:
            for y in range(y_max):
                for x in range(x_max):
                    if non_body_mask[z, y, x] and not filled[z, y, x]:
                        seed_mask = np.zeros_like(body_mask, dtype=bool)
                        seed_mask[z, y, x] = True
                        region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                        filled = filled | region
        
        # Fill from Y borders (front and back)
        for z in range(z_max):
            for y in [0, y_max - 1]:
                for x in range(x_max):
                    if non_body_mask[z, y, x] and not filled[z, y, x]:
                        seed_mask = np.zeros_like(body_mask, dtype=bool)
                        seed_mask[z, y, x] = True
                        region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                        filled = filled | region
        
        # Fill from X borders (left and right)
        for z in range(z_max):
            for y in range(y_max):
                for x in [0, x_max - 1]:
                    if non_body_mask[z, y, x] and not filled[z, y, x]:
                        seed_mask = np.zeros_like(body_mask, dtype=bool)
                        seed_mask[z, y, x] = True
                        region = ndimage.binary_propagation(seed_mask, mask=non_body_mask)
                        filled = filled | region
        
        # Body is the inverse of filled external regions
        body_mask = ~filled
    
    return body_mask