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
def load_image_rgb(path: Path, rotation: int = 0) -> np.ndarray:
    """
    Load an image as an RGB numpy array with dtype uint8.
    
    Parameters
    ----------
    path : Path
        Path to image file
    rotation : int
        Rotation angle in degrees (0, 90, 180, 270). Positive = clockwise.
        
    Returns
    -------
    np.ndarray
        RGB image array, shape (Y, X, 3), dtype uint8
    """
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
        RGB image array, shape (Y, X, 3), dtype uint8
    rotation : int
        Rotation angle in degrees (0, 90, 180, 270). Positive = clockwise.
        
    Returns
    -------
    np.ndarray
        Rotated RGB image array, same shape/dtype
    """
    from PIL import Image
    im = Image.fromarray(arr, mode="RGB")
    # PIL rotates counter-clockwise, so negate
    im = im.rotate(-rotation, expand=False)
    return np.array(im, dtype=np.uint8)


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
        RGB image array, shape (Y, X, 3), dtype uint8
    bottom_region_ratio : float
        Fraction of image height to start scanning from bottom (0-1)
        Default 0.4 means start from bottom 40% and scan upward
    min_intensity : int
        Minimum grayscale intensity to consider (0-255)
        Default 150 (bright, but not requiring pure white)
    min_size_ratio : float
        Minimum object size as fraction of image area (0-1)
        Default 0.01 (1% of image) - catches smaller bed parts
    max_size_ratio : float
        Maximum object size as fraction of image area (0-1)
        Default 0.4 (40% of image) - filters out entire patient
    scan_upward : bool
        If True, scan from bottom upward to find all bed components
    grey_tolerance : int
        Intensity difference tolerance for including grey pixels around bed (0-255)
        Default 30 - includes pixels within 30 grayscale units of bed intensity
        
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
    
    # Remove black background
    black_mask = gray < 10
    
    # Start from bottom and scan upward
    bottom_start = int(height * (1 - bottom_region_ratio))
    
    # Create mask for bright objects (bed/headrest is typically bright white)
    bright_mask = gray >= min_intensity
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
            grey_tolerance_extended = int((grey_tolerance + 80) * aggressivity)  # Even more aggressive tolerance for dark greys
            grey_mask = (
                (gray >= max(20, bed_min_intensity - grey_tolerance_extended)) &  # Include darker greys (but not black)
                (gray <= bed_max_intensity + grey_tolerance_extended) &
                ~black_mask &
                (gray < min_intensity)  # Only grey, not bright white
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
    bright_threshold: int = 150,
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
    
    # Remove black background from consideration
    black_mask = target_gray < 10
    
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
        bright_mask = target_gray >= bright_threshold
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
            grey_tolerance_extended = grey_tolerance + 60  # More aggressive for dark greys
            grey_mask = (
                (target_gray >= max(15, obj_min_intensity - grey_tolerance_extended)) &  # Include darker greys
                (target_gray <= obj_max_intensity + grey_tolerance_extended) &
                ~black_mask &
                (target_gray < bright_threshold)  # Only grey, not bright white
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
    object_mask: Optional[np.ndarray] = None,
    object_mask_slice_index: Optional[int] = None,
    non_grayscale_slice_ranges: Optional[List[Tuple[int, int]]] = None,  # List of (min, max) slice ranges for non-grayscale removal
    object_removal_objects: Optional[List[dict]] = None,  # List of objects with masks and slice ranges
    progress_cb: Optional[Callable[[str, int, int, int], None]] = None,
) -> np.ndarray:
    """
    Apply preprocessing steps slice-by-slice:

      - Overlay removal
      - (Optional) bed removal
      - (Optional) remove non-grayscale pixels
      - (Optional) remove selected objects using mask tracking

    Parameters
    ----------
    volume_rgb : np.ndarray
        Shape (Z, Y, X, 3), dtype uint8
    grayscale_tolerance : int
    saturation_threshold : float
    remove_bed : bool
    remove_non_grayscale : bool
        If True, turn all non-grayscale pixels black
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
        
        cleaned = _remove_colored_overlays(
            rgb,
            grayscale_tolerance=grayscale_tolerance,
            saturation_threshold=saturation_threshold,
        )

        if remove_bed:
            cleaned = _remove_bed_placeholder(cleaned)
        
        # Apply object mask if available for this slice
        if slice_masks[z_idx] is not None:
            cleaned = _apply_object_mask(cleaned, slice_masks[z_idx])

        processed.append(cleaned)

    return np.stack(processed, axis=0)
