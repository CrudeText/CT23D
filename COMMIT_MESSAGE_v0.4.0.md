# CT23D v0.4.0 Release

## Major Features

### GUI Enhancements

- **Patient Info Display**
  - Patient Info box in main window showing DICOM metadata
  - Displays patient name, ID, birth date, sex, study date/time, study description, modality
  - Automatically updates when DICOM files are loaded
  - Positioned at top right, leaving space for logo
  - Two-column layout for compact display
  - Only visible when DICOM files are loaded with metadata

- **Auto-bin System**
  - Automatic intensity bin detection based on intensity distribution
  - Configurable intensity range (min/max) that adapts to DICOM (uint16) vs standard (uint8) images
  - Configurable number of bins
  - Uniformity parameter (0-1): 1 for uniform bins, 0 for full control by distribution (default: 0)
  - Aggressive uniformity factor for better control
  - Apply button to generate bins automatically

- **Enhanced Histogram Visualization**
  - Automatic graph computation when processed slices directory is selected
  - Combined loading dialog for volume loading and histogram computation
  - Improved progress tracking during histogram computation
  - Draggable intensity range lines on histogram and heatmap (min/max)
  - Visualize checkbox to show/hide range lines
  - Range lines have labels on graphs
  - Intensity range highlighting on slice preview when visualized

- **Improved UI Layout**
  - Compact layout for directory selectors (buttons on left, paths on right)
  - Export buttons with larger, bold fonts for better visibility
  - Patient Info box positioned to avoid overlapping content
  - Improved spacing and alignment throughout
  - Logo scaled to 140px height for better visibility
  - Window icon supports up to 1024px resolution

- **Slice Preview Enhancements**
  - Automatic loading of slice previews when directory is selected
  - Grayscale preview when no bins are defined (no longer black)
  - Intensity range highlighting on slices when visualized
  - Helpful placeholder text: "No image loaded\nSelect an input folder"

### Preprocessing Improvements

- **Streamlined UI**
  - Crop buttons aligned with "Crop:" label on same row
  - Removed note about slice ranges (now self-evident from table)
  - Checkboxes for grayscale conversion on same row
  - Simplified text: "Remove non-grayscale pixels" and "Convert to grayscale"
  - Improved spacing between elements

- **Z Height Calculation**
  - Real-time Z height calculation (number of slices × Z voxel spacing)
  - Updates automatically when directory is loaded or Z spacing changes
  - Displayed under Z voxel spacing control

### Meshing Improvements

- **Automatic Graph Computation**
  - Histogram and heatmap computed automatically when directory is selected
  - No separate "Compute graphs" button needed
  - Unified loading progress dialog
  - Graphs appear immediately after volume loads

- **Intensity Range Visualization**
  - Visualize checkbox to show/hide min/max intensity range lines
  - Draggable range lines on both histogram and heatmap views
  - Labels on range lines for clarity
  - Intensity range highlighting on slice preview
  - Range lines update intensity bin controls in real-time

## Technical Improvements

- Combined volume loading and histogram computation into single progress dialog
- Improved UI responsiveness during histogram computation (more frequent processEvents calls)
- Better coordinate handling for widget positioning
- Fixed widget positioning to account for actual visual boundaries
- Enhanced DICOM metadata extraction
- Improved layout spacing and alignment
- Better z-ordering for overlay widgets (Patient Info, logo)

## UI/UX Enhancements

- Patient Info box with DICOM metadata display
- Compact, efficient layouts throughout
- Automatic graph computation for smoother workflow
- Real-time intensity range visualization
- Better visual feedback with intensity highlighting
- Improved button and label positioning
- More prominent export buttons
- Helpful placeholder text in preview areas

## Bug Fixes

- Fixed "Is not responding" during graph loading
- Fixed max range line getting stuck to min line when dragging
- Fixed inverted axes display in heatmap (intensity now on Y-axis, slice number on X-axis)
- Fixed range lines being vertical in heatmap (now horizontal to match intensity axis)
- Fixed pause between loading windows
- Fixed black slice previews when no bins defined
- Fixed Patient Info box overlapping with content
- Fixed widget positioning calculations

## Documentation

- Updated README with new tool screenshots
- Added references to meshing_tools.png and preprocessing_tools.png
- Updated workflow descriptions

---

**Full Changelog**: See commit history for detailed changes.

