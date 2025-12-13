# CT23D – CT Slice to 3D Mesh Pipeline

CT23D is a modular Python application for converting stacks of CT slice images into 3D meshes.
It provides both a graphical interface and a programmatic core to preprocess CT slices, build volumetric data, analyze intensity distributions, and generate intensity-based 3D meshes.

The project is designed to be:
- Cleanly structured
- Extensible
- Usable both interactively (GUI) and headless (CLI / scripting)

---

## Features

### Preprocessing
- Load raw CT slice images (PNG, JPG, TIFF, etc.)
- **Interactive object selection tools:**
  - Box selection
  - Lasso selection
  - Click-to-select with hover highlighting
  - Configurable selection tolerance
- **Automatic bed/headrest detection:**
  - Scans from bottom upward
  - Configurable aggressivity and parameters
  - Settings dialog for fine-tuning
- **Image rotation:**
  - 90° clockwise, 90° counter-clockwise, 180°
  - Preview rotation before processing
  - Applied to entire volume during preprocessing
- **Remove non-grayscale pixels:**
  - Configurable saturation threshold
  - Turns colored pixels black
- Remove overlays such as:
  - Numbers
  - Text
  - Scanner bed / headrest
- **Selected objects management:**
  - Table listing all selected objects
  - Optional labels (bed, headrest, other, etc.)
  - Delete individual selections
- Adjustable preprocessing parameters:
  - Grayscale tolerance
  - Saturation threshold
- **Batch processing with detailed progress:**
  - Phase-aware progress (Loading, Processing, Saving)
  - Per-phase slice counters
  - Elapsed time and time remaining
  - Cancellable operations
- Output clean, processed slices ready for meshing

### Meshing
- Load processed slices into a 3D volume
- **3D histogram visualization:**
  - X-axis: Slice number
  - Y-axis: Intensity
  - Z-axis: Pixel count (shown as color/intensity heatmap)
  - Includes all pixels (including zeros) for accurate representation
- **Interactive bin management:**
  - Add/delete bins
  - Modify bin min/max values (integer intensity values)
  - Draggable bin boundaries on histogram
  - Continuous bins mode (automatically adjusts adjacent bins)
  - Bin colors with preview
  - Enable/disable individual bins
- **Slice preview:**
  - Large preview with slice navigation
  - Pixels colored according to assigned bins
  - Real-time updates when bin parameters change
- Automatic or manual intensity bin definition
- Per-bin 3D mesh generation
- Supports multiple output meshes per dataset

### Export
- **Format support:**
  - **PLY (Polygon File Format)** ✓ - Fully implemented
    - Per-vertex RGB colors
    - Per-vertex alpha channel (opacity)
    - Blender-compatible sRGB colors
  - Other formats (OBJ, STL, GLTF) - Planned for future
- **Export options:**
  - Single file (all bins combined) or multiple files (one per bin)
  - Export with/without colors
  - Export with/without opacity
  - Custom filename prefix
- **Progress tracking:**
  - Phase-aware progress (Building masks, Extracting meshes, Saving files)
  - Detailed per-phase counters
  - Elapsed time and time remaining

### GUI
- Built with PySide6
- Fully threaded execution (no UI freezing)
- **Enhanced progress dialogs:**
  - Phase-aware progress with per-phase counters
  - "Complete" indicators for finished phases
  - Independent timer (elapsed/remaining time)
  - Progress bar resets for each phase
- Clear separation between preprocessing and meshing steps
- Default directory linking (preprocessing output → meshing input)

---

## Project Structure

```
CT23D/
├── pyproject.toml
├── README.md
│
├── scripts/
│   └── run_ct23d_gui.py
│
└── src/
    └── ct23d/
        ├── core/
        │   ├── images.py
        │   ├── preprocessing.py
        │   ├── volume.py
        │   ├── bins.py
        │   ├── meshing.py
        │   ├── export.py
        │   └── models.py
        │
        ├── gui/
        │   ├── app.py
        │   ├── main_window.py
        │   ├── status.py
        │   ├── workers.py
        │   │
        │   ├── preproc/
        │   │   └── wizard.py
        │   │
        │   └── mesher/
        │       ├── wizard.py
        │       ├── histogram_3d_view.py
        │       └── slice_preview.py
        │
        └── cli/
            ├── preprocess_cli.py
            └── mesh_cli.py
```


---

## Installation

### Requirements
- Python 3.10 or newer
- Conda recommended (Anaconda or Miniconda)

### Create environment
conda create -n ct23d-env python=3.10
conda activate ct23d-env

### Install CT23D
pip install -e .

This installs CT23D in editable mode so source changes are reflected immediately.

---

## Running the GUI

From the project root directory:

python scripts/run_ct23d_gui.py

---

## Preprocessing Workflow

1. Open the Preprocessing tab
2. Select an input directory containing raw CT slices
3. Select an output directory for processed slices
4. **Optional: Remove non-grayscale pixels** - Check the option and adjust threshold if needed
5. **Optional: Select objects to remove:**
   - Choose a tool mode (Box, Lasso, or Click to Select)
   - Select objects in the preview image
   - Assign labels to selected objects (optional)
   - Objects will be removed from all slices during preprocessing
6. **Optional: Auto-detect bed/headrest:**
   - Click "Auto-Detect Bed/Headrest" (requires bed/headrest to be at the bottom)
   - Adjust settings via the settings icon if needed
7. **Optional: Rotate images:**
   - Use rotation buttons (90° CW, 90° CCW, 180°)
   - Preview shows rotation in "After preprocessing" view
   - Rotation is applied during full preprocessing
8. Adjust preprocessing parameters if needed
9. Click "Run preprocessing"

Processed slices are written to the selected output directory. The output directory will be automatically set as the default input for the meshing tab.

---

## Meshing Workflow

1. Open the Meshing tab
2. **Select a directory containing processed slices:**
   - The preprocessing output directory is automatically set as default
   - You can still select a different directory if needed
3. **Set meshing parameters:**
   - Spacing (Z, Y, X in mm)
   - Number of bins
   - Intensity range (min/max, integers 0-255)
4. Click "Compute histogram"
5. **Inspect the 3D histogram:**
   - X-axis: Slice number
   - Y-axis: Intensity
   - Color/intensity: Pixel count
6. **Manage intensity bins:**
   - Default bins are created automatically
   - Add/delete bins as needed
   - Adjust bin min/max values (integers)
   - Drag bin boundaries on the histogram for quick adjustment
   - Enable "Continuous bins" to automatically adjust adjacent bins
   - Assign colors to bins (double-click color cell)
   - Enable/disable bins using checkboxes
7. **Preview slices:**
   - Navigate through slices using the slice selector
   - Preview shows pixels colored according to their assigned bins
8. **Configure export options:**
   - Format: PLY (currently supported)
   - File organization: Single file (all bins combined) or Multiple files (one per bin)
   - Export with colors: Enable/disable per-vertex RGB colors
   - Export with opacity: Enable/disable per-vertex alpha channel
9. Select an output directory
10. Set filename prefix (optional)
11. Click "Export meshes"

Each enabled bin produces a separate mesh file (or all bins combined into one file, depending on your selection). Meshes include bin colors and optional opacity as configured.

---

## Output

### Preprocessing Output
- Clean, processed slice images
- Non-grayscale pixels removed (if enabled)
- Selected objects removed from all slices
- Rotated images (if rotation was applied)

### Meshing Output
- **Format:** PLY (Polygon File Format)
- **File organization:**
  - **Multiple files:** One PLY file per enabled bin
    - Filename format: `{prefix}_bin_{id:02d}_{low}_{high}.ply`
    - Example: `ct_layer_bin_01_12_85.ply`
  - **Single file:** All bins combined into one PLY file
    - Filename format: `{prefix}_combined.ply`
    - Example: `ct_layer_combined.ply`
- **Features:**
  - Per-vertex RGB colors (if enabled) - sRGB-compatible for Blender
  - Per-vertex alpha channel (if opacity enabled)
  - Integer intensity values in filenames

---

## Design Principles

- Core logic is GUI-independent
- GUI uses background threads for all heavy operations
- Clear separation between:
  - Image I/O
  - Preprocessing
  - Volume construction
  - Histogram analysis
  - Mesh generation
- Legacy compatibility preserved through wrappers when required

---

## Development Notes

- GUI workers are QThread-based
- Progress updates are phase-aware with per-phase counters
- Histogram visualization is handled via PyQtGraph (3D heatmap)
- All pixels (including zeros) are included in histogram for accurate slice representation
- Bin boundaries are draggable on the histogram with real-time table updates
- Object selection uses connected component analysis with configurable tolerance
- Mask propagation applies selected objects to all slices at the same coordinates
- Export supports per-vertex colors and opacity in PLY format

---

## Roadmap

- Additional mesh export formats (OBJ, STL, GLTF/GLB, FBX)
- Preset-based workflows (bone, soft tissue, etc.)
- Volume preview (orthogonal slice views)
- Batch and headless processing
- Extended documentation and examples
- Advanced mask propagation algorithms

---

## License

License to be defined (MIT recommended).
