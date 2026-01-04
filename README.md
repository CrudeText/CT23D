# CT23D

Convert medical CT data into 3D meshes with DICOM-native workflows and advanced preprocessing.

CT23D is a Python toolkit for converting medical CT scan data into 3D surface meshes. It is designed with DICOM as the primary input format, preserving 16-bit intensity ranges and metadata throughout the processing pipeline. PNG and JPEG formats are supported as secondary fallback options. The tool is built for researchers, engineers, and technical users working with medical imaging data who need reliable, reproducible workflows from raw scans to exportable 3D models.

## Key Features

- **DICOM-first processing:** Native support for DICOM files with automatic 16-bit intensity handling and metadata preservation
- **Robust preprocessing:** Automatic overlay removal, bed/headrest detection, and manual object selection tools
- **Intensity-based meshing:** Flexible binning system for generating multiple meshes per dataset
- **Separation of concerns:** Core processing logic is independent of GUI and CLI, enabling both interactive and headless workflows
- **YAML-based configuration:** Project configuration system using relative paths for reproducibility
- **Multiple export formats:** PLY, STL, and NRRD support with per-vertex colors and opacity

## Development Status

CT23D is actively developed. The GUI provides a functional end-to-end workflow but continues to evolve with new features and refinements. Core processing functionality is stable and well-tested.

---

## Graphical User Interface Overview

CT23D provides a three-tab graphical interface built with PySide6, covering the complete workflow from preprocessing to visualization.

### Image Processing Tab

The preprocessing tab handles raw CT slice preparation, including overlay removal, object selection, bed detection, and image transformation.

![Preprocessing Tools](./images/preprocessing_tools.png)
*Preprocessing interface with comprehensive tools for image transformation, object selection, and parameter configuration*

![Preprocessing Example](./images/preprocessing_example.png)
*Preprocessing example showing automatic bed detection (red bounding box) and before/after comparison*

### Meshing Tab

The meshing tab loads processed slices into a 3D volume, analyzes intensity distributions, manages intensity bins, and exports meshes in multiple formats.

![Meshing Tools](./images/meshing_tools.png)
*Meshing interface with histogram visualization, intensity bin management, and slice preview tools*

![Meshing Example](./images/meshing_example.png)
*Meshing example showing intensity histogram analysis and slice preview with bin-colored pixels*

### 3D Preview Tab

The 3D preview tab provides interactive mesh visualization for canonical volumes exported from the meshing workflow.

![3D Preview](./images/3D_Preview_example.png)
*3D Preview tab with interactive mesh visualization, opacity controls, and mesh generation tools*

---

## Installation

### Requirements

- Python 3.10 or newer
- Recommended: Conda (Anaconda or Miniconda) for environment management

### Setup

1. Create a conda environment (optional but recommended):

```bash
conda create -n ct23d-env python=3.10
conda activate ct23d-env
```

2. Install CT23D in editable mode:

```bash
pip install -e .
```

This installs CT23D and its dependencies, including the numerical stack (NumPy, SciPy), imaging libraries (scikit-image, imageio, pydicom), meshing tools (trimesh), and GUI frameworks (PySide6, pyqtgraph, pyvistaqt).

---

## Quick Start

### GUI Workflow (Primary Entry Point)

Launch the GUI from the project root:

```bash
python scripts/run_ct23d_gui.py
```

**End-to-end workflow:**

1. **Image Processing tab:** Load your DICOM directory, configure preprocessing operations (overlay removal, bed detection, cropping), and export processed slices
2. **Meshing tab:** Load processed slices, analyze intensity distributions using the histogram tools, configure intensity bins, and export meshes (PLY, STL, or NRRD)
3. **3D Preview tab:** Load exported NRRD volumes and generate interactive 3D visualizations with adjustable opacity and mesh properties

The preprocessing output directory automatically links to the meshing input directory for seamless workflow transitions.

**GUI status:**
- Core functionality (preprocessing, meshing, export) is stable
- Interactive features (slice previews, histogram tools) are functional
- Live mesh previews and advanced visualization features continue to evolve

### CLI Workflow

The command-line interface is preferable for batch processing, reproducibility, and integration into automated pipelines.

**Preprocessing CLI:**

```bash
python -m ct23d.cli.preprocess_cli --input-dir /path/to/dicom --processed-dir /path/to/output
```

**Meshing CLI:**

```bash
python -m ct23d.cli.mesh_cli --processed-dir /path/to/processed --output-dir /path/to/meshes --spacing 1.6 1.0 1.0
```

**YAML-based project configuration:**

CT23D supports YAML configuration files for reproducible workflows. Configuration files use relative paths, enabling project portability across systems. Default configurations and presets (bone, soft tissue) are available in `src/ct23d/config/`.

### Legacy Script

The `CT_to_3D_legacy.py` script is maintained for backward compatibility. It provides a simplified interface that bypasses parts of the current project system, making it useful for quick one-off conversions or migration scenarios. This script is a compatibility wrapper around the core processing logic.

**Note:** The legacy script is currently being refactored to use the new modular architecture. Once complete, it will call `ct23d.core.ct_compat.main()`.

---

## Project Structure

```
CT23D/
├── pyproject.toml          # Project metadata and dependencies
├── README.md
│
├── scripts/
│   ├── run_ct23d_gui.py    # GUI entry point
│   ├── CT_to_3D_legacy.py  # Legacy compatibility wrapper
│   └── view_nrrd.py        # NRRD viewer utility
│
├── src/ct23d/
│   ├── core/               # Pure processing logic (no UI dependencies)
│   │   ├── images.py       # Image I/O and DICOM handling
│   │   ├── preprocessing.py # Preprocessing operations
│   │   ├── volume.py       # Volume construction
│   │   ├── bins.py         # Intensity binning logic
│   │   ├── meshing.py      # Mesh generation (marching cubes)
│   │   ├── export.py       # Mesh export (PLY, STL, NRRD)
│   │   ├── models.py       # Configuration dataclasses
│   │   └── config_io.py    # YAML config loading/saving
│   │
│   ├── gui/                # PySide6 graphical interface
│   │   ├── app.py          # Application entry point
│   │   ├── main_window.py  # Main window and tab structure
│   │   ├── workers.py      # Background thread workers
│   │   ├── status.py       # Status and progress dialogs
│   │   ├── preproc/        # Preprocessing tab components
│   │   ├── mesher/         # Meshing tab components
│   │   └── processing3d/   # 3D preview tab components
│   │
│   ├── cli/                # Command-line tools
│   │   ├── preprocess_cli.py
│   │   └── mesh_cli.py
│   │
│   └── config/             # Configuration system
│       ├── defaults.yaml   # Default project configuration
│       └── presets/        # Preset configurations (bone, soft tissue)
│
└── images/                 # README screenshots and examples
```

**Design philosophy:**

- **Core logic separation:** All processing algorithms live in `core/`, independent of GUI or CLI
- **Testable architecture:** Pure functions and dataclasses enable unit testing without UI dependencies
- **Modular components:** Each module has a clear responsibility (I/O, preprocessing, meshing, export)

---

## Processing Pipeline

The CT23D pipeline transforms raw CT data into exportable 3D meshes through the following stages:

1. **DICOM loading:** Primary format with 16-bit intensity preservation (0-65535). PNG and JPEG are supported as fallback formats with 8-bit intensity ranges (0-255). Automatic intensity range adaptation throughout the UI.
2. **Overlay and text removal:** Automatic detection and removal of scan annotations, overlays, and text artifacts that interfere with meshing.
3. **Bed and headrest removal:** Automatic detection of scanning table artifacts with configurable aggressivity parameters.
4. **Volume construction:** Processed slices are assembled into a 3D volumetric array with proper Z-ordering (automatic for DICOM via metadata, alphabetical for other formats).
5. **Intensity binning:** Intensity distribution analysis enables definition of bins for tissue separation. Auto-bin system detects optimal bin boundaries, or bins can be manually configured.
6. **Mask cleanup:** Optional component filtering removes small disconnected regions, and optional Gaussian smoothing reduces surface noise.
7. **Marching cubes extraction:** Per-bin surface meshes are generated using the marching cubes algorithm with configurable voxel spacing.
8. **Mesh export:** Meshes are exported in PLY (with colors and opacity), STL (geometry only), or NRRD (canonical volume format) with provenance metadata.

---

## Configuration System

CT23D uses YAML-based project configuration files for reproducible workflows.

**Key features:**

- **Relative paths:** All paths in configuration files are relative to the YAML file location, enabling project portability
- **Defaults and presets:** Default configuration templates are provided, along with presets for common use cases (bone segmentation, soft tissue analysis)
- **Validation:** Configuration files are validated against dataclass schemas to ensure correctness
- **CLI integration:** The command-line tools can load project configurations, overriding individual parameters via command-line flags

**Configuration structure:**

- `preprocess`: Input/output directories, caching, preprocessing parameters
- `meshing`: Spacing, thresholds, binning parameters, export settings
- `bins`: Custom intensity bin definitions (optional; auto-binning can be used instead)

Example presets are available in `src/ct23d/config/presets/` for reference.

---

## Outputs

### Preprocessing Output

Processed slice images are written to the specified output directory with all modifications applied (overlay removal, rotations, crops, object removals). DICOM metadata is preserved when applicable.

### Meshing Output

**Supported formats:**

- **PLY (Polygon File Format):** Binary format with per-vertex RGB colors and optional alpha channel (opacity). Colors are Blender-compatible sRGB.
- **STL (Stereolithography):** Binary or ASCII format, geometry only (no colors). Widely compatible with CAD software.
- **NRRD (Nearly Raw Raster Data):** Canonical volume format with spacing metadata, gzip compression, and sidecar JSON for provenance. Compatible with the 3D Preview tab.

**File organization:**

- **Multiple files mode:** One mesh file per enabled intensity bin
  - Format: `{prefix}_bin_{id:02d}_{min_intensity}_{max_intensity}.{ext}`
  - Example: `ct_layer_bin_01_12_85.ply`
- **Single file mode:** All bins combined into one mesh file
  - Format: `{prefix}_combined.{ext}`
  - Example: `ct_layer_combined.ply`

**NRRD volume export:**

- Includes voxel spacing metadata (Z, Y, X in mm) from UI settings
- Includes provenance metadata (source directory, timestamps, application version, bin information)
- Automatic data type optimization (int16 or float32 based on intensity range)

---

## Roadmap

CT23D development priorities include:

- **GUI refinements:** Enhanced live slice previews, real-time mesh preview during export, improved histogram interaction
- **Export formats:** Additional formats (OBJ, GLTF/GLB, FBX) for broader software compatibility
- **DICOM tooling:** Deeper DICOM metadata utilization, window/level presets, modality-specific optimizations
- **Advanced features:** Label volume support in 3D Preview, ML-based organ mapping and segmentation, orthogonal slice views
- **Distribution:** Application packaging for easier deployment (PyInstaller, briefcase)

This roadmap reflects ongoing development priorities and may evolve based on user feedback and requirements.

---

## Contributing

CT23D is designed with clear separation between core processing logic and user interfaces. This architecture enables:

- **Testable code:** Core functions operate on NumPy arrays and dataclasses, facilitating unit tests without GUI dependencies
- **Reusable components:** Processing modules can be integrated into other workflows or scripts
- **Maintainable structure:** UI changes do not affect core algorithms, and core improvements benefit all interfaces

Contributions are welcome. Please open issues for bug reports, feature requests, or questions. Pull requests should maintain the separation of concerns and include appropriate tests for core functionality.

---

## License

MIT License.
