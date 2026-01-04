# CT23D v0.2.0 Release

## Major Features

### Enhanced Preprocessing
- **Interactive Object Selection Tools**
  - Box selection tool for rectangular area selection
  - Lasso selection tool for freeform selection
  - Click-to-select with hover highlighting and configurable tolerance
  - Selected objects management table with optional labels
  - Delete individual selections

- **Automatic Bed/Headrest Detection**
  - Bottom-up scanning algorithm
  - Configurable aggressivity and detection parameters
  - Settings dialog for fine-tuning
  - Handles multiple bed components and surrounding grey pixels

- **Image Rotation**
  - 90° clockwise, 90° counter-clockwise, and 180° rotation
  - Preview rotation before processing
  - Applied to entire volume during preprocessing

- **Remove Non-Grayscale Pixels**
  - Configurable saturation threshold
  - Turns colored pixels black for cleaner processing

- **Improved Progress Tracking**
  - Phase-aware progress (Loading, Processing, Saving)
  - Per-phase slice counters
  - Elapsed time and time remaining
  - Cancellable operations with proper cleanup

### Enhanced Meshing & Export

- **3D Histogram Visualization**
  - X-axis: Slice number
  - Y-axis: Intensity
  - Z-axis: Pixel count (shown as color/intensity heatmap)
  - Includes all pixels (including zeros) for accurate representation
  - Auto-zoom to fit data

- **Interactive Bin Management**
  - Add/delete bins dynamically
  - Modify bin min/max values (integer intensity values 0-255)
  - Draggable bin boundaries on histogram with real-time table updates
  - Continuous bins mode (automatically adjusts adjacent bins)
  - Bin colors with live preview
  - Enable/disable individual bins

- **Slice Preview**
  - Large preview with slice navigation
  - Pixels colored according to assigned bins
  - Real-time updates when bin parameters change

- **Export Functionality**
  - PLY format support with per-vertex RGB colors and alpha channel
  - Single file (all bins combined) or multiple files (one per bin)
  - Export with/without colors
  - Export with/without opacity
  - Blender-compatible sRGB colors
  - Format capabilities system (ready for future format additions)
  - Integer intensity values in filenames

- **Default Directory Linking**
  - Preprocessing output automatically set as default meshing input
  - Seamless workflow between preprocessing and meshing tabs

## Technical Improvements

- Simplified mask propagation (same coordinates across all slices)
- Phase-aware progress callbacks throughout the pipeline
- Independent timer for progress dialogs
- Improved UI responsiveness with proper threading
- Better error handling and cancellation support
- Code organization with new modules (`export.py`, `histogram_3d_view.py`, `slice_preview.py`)

## UI/UX Enhancements

- Format selection with dynamic option enabling/disabling
- Tooltips explaining format capabilities
- Visual feedback for format support (colors, opacity)
- Improved progress dialog with phase status indicators
- Better layout organization and spacing

## Breaking Changes

- Bin intensity values are now integers (0-255) throughout the interface
- Export replaces "Generate meshes" button with inline export options
- Histogram visualization changed from 2D to 3D heatmap representation

## Bug Fixes

- Fixed histogram gap issues by including zero-intensity pixels
- Fixed bin label positioning and display
- Fixed color modification in bin table
- Fixed preview loading and slice navigation
- Fixed progress bar reset and phase tracking
- Fixed cancellation handling in preprocessing

## Documentation

- Updated README with comprehensive feature descriptions
- Added detailed workflows for preprocessing and meshing
- Documented export formats and capabilities
- Added development notes

---

**Full Changelog**: See commit history for detailed changes.













