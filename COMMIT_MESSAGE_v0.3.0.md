# CT23D v0.3.0 Release

## Major Features

### Enhanced Preprocessing

- **Black Threshold (Background Removal)**
  - Min/max intensity range selection (replaces single grayscale tolerance)
  - Visualize button to highlight pixels within threshold range
  - Select button to add threshold selection to removal list
  - Applied to all slices or specified slice ranges

- **Image Transformation & Cropping**
  - Combined rotation and cropping into single section
  - **Cropping:**
    - Box or lasso selection tools
    - Multiple crops with different slice ranges
    - Cancel crop button
    - All pixels within crop area preserved (including black pixels)
    - Editable slice ranges in modifications table
  - **Rotation:** 90° clockwise, 90° counter-clockwise, 180°
  - Preview transformations before processing

- **Unified Modifications Table**
  - Single table listing all modifications:
    - Object removals (with slice ranges)
    - Crops (with slice ranges)
    - Non-grayscale removal (with slice ranges)
    - Black threshold removal (with slice ranges)
  - Editable slice ranges (min/max) for each modification
  - Clear indication of modification type
  - All modifications applied during preprocessing

- **Export Range Selection**
  - Export all slices (default)
  - Export specific slice range
  - Min/max slice selection controls
  - Only slices within range are saved

- **Slice Navigation Improvements**
  - Slider for quick slice navigation
  - Synced with slice selector spinbox
  - Improved preview responsiveness

### Enhanced Meshing & Visualization

- **Tabbed Histogram View**
  - **Aggregated Histogram Tab:**
    - Overall intensity distribution (Intensity vs. Pixel Count)
    - Intensity 0 pixels filtered out for cleaner visualization
    - Linear scale (logarithmic removed for better proportion)
    - Vertical bin boundary lines with labels
    - Draggable bin boundaries
  - **Slice-by-Slice Heatmap Tab:**
    - 2D heatmap (Slice Number vs. Intensity)
    - Intensity on horizontal axis, Slice Number on vertical axis
    - Color represents pixel count (log scale for visualization)
    - Vertical bin boundary lines aligned with intensity axis
    - Draggable bin boundaries
  - Removed 3D Surface Plot (performance optimization)

- **Bin Management Improvements**
  - Bin limits always start at intensity 1 minimum (0 is background/air)
  - Continuous bins enabled by default
  - Improved bin line visibility (thicker lines, higher Z-order)
  - Better bin boundary dragging with real-time updates
  - Bin lines properly recreated after histogram updates

- **Slice Preview Enhancements**
  - Slice navigation slider for quick browsing
  - Improved responsiveness to layout changes
  - Better image scaling and display

### Export Enhancements

- **STL Format Support**
  - Binary or ASCII format selection
  - Geometry-only export (no colors/opacity)
  - Widely compatible with 3D software

- **Mesh Processing Options**
  - **Component Filtering:** Optional removal of small disconnected components
    - Configurable minimum component size
    - Helps reduce noise in meshes
  - **Gaussian Smoothing:** Optional mesh surface smoothing
    - Configurable smoothing strength (sigma)
    - Creates smoother surfaces but may lose small details
  - Both options are optional with checkboxes and tooltips
  - Spacing controls moved to Mesh Processing Options

- **File Size Estimation**
  - Calculate approximate file size button
  - Shows total size in MB for all files
  - Non-blocking calculation with progress indicator
  - File size-based progress for saving phase

- **Improved Progress Tracking**
  - File size-based progress for saving phase
  - Better cancellation handling
  - Files properly flushed and synced (Windows compatibility)

## Technical Improvements

- Removed 3D surface plot for better performance
- Single unified progress dialog for volume loading
- Improved UI responsiveness during histogram computation
- Better error handling and cancellation support
- Intensity 0 filtering in aggregated histogram
- Bin limits enforced to start at intensity 1 minimum
- Improved bin line visibility and management
- Better file handling (flush and sync for Windows)

## UI/UX Enhancements

- Unified modifications table for all preprocessing changes
- Slice range editing directly in table
- Slice navigation slider for quick browsing
- Tabbed histogram view for better organization
- Improved tab visibility (larger, bolder)
- Better progress dialog content (relevant phase information)
- Tooltips and explanations for mesh processing options
- File size calculation button with clear display

## Breaking Changes

- Black Threshold now uses min/max range instead of single value
- Grayscale tolerance renamed to "Black Threshold"
- Image Rotation section renamed to "Image Transformation & Cropping"
- Bin limits now have minimum value of 1 (cannot start at 0)
- Intensity 0 pixels filtered from aggregated histogram display
- 3D Surface Plot removed (performance optimization)
- Logarithmic scale removed from aggregated histogram Y-axis

## Bug Fixes

- Fixed bin lines not appearing in histogram views
- Fixed bin line visibility and positioning
- Fixed histogram axes orientation in heatmap
- Fixed missing pixels/slices in histogram displays
- Fixed file locking issues on Windows
- Fixed export cancellation not working properly
- Fixed UI freezing during mesh extraction
- Fixed progress dialog showing incorrect phase status
- Fixed crop preserving black pixels across slices
- Fixed loading screen not covering all computations

## Documentation

- Updated README with all v0.3.0 features
- Documented new preprocessing workflow
- Documented tabbed histogram view
- Documented STL export format
- Documented mesh processing options
- Updated version to 0.3.0

---

**Full Changelog**: See commit history for detailed changes.

