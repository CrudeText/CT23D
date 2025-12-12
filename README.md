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
- Remove overlays such as:
  - Numbers
  - Text
  - Scanner bed / headrest
- Adjustable preprocessing parameters:
  - Grayscale tolerance
  - Saturation threshold
- Batch processing with progress tracking
- Output clean, processed slices ready for meshing

### Meshing
- Load processed slices into a 3D volume
- Compute intensity histogram with background automatically ignored
- Interactive histogram visualization using PyQtGraph
- Automatic or manual intensity bin definition
- Per-bin 3D mesh generation
- Supports multiple output meshes per dataset

### GUI
- Built with PySide6
- Fully threaded execution (no UI freezing)
- Progress dialogs with per-slice and per-bin counters
- Clear separation between preprocessing and meshing steps

---

## Project Structure

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
        │       └── histogram_view.py
        │
        └── cli/
            ├── preprocess_cli.py
            └── mesh_cli.py

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
4. Adjust preprocessing parameters if needed
5. Click Run preprocessing

Processed slices are written to the selected output directory.

---

## Meshing Workflow

1. Open the Meshing tab
2. Select a directory containing processed slices
3. Click Load volume / compute histogram
4. Inspect the histogram (zero-intensity background is excluded)
5. Adjust intensity bins if needed
6. Select an output directory
7. Click Generate meshes

Each enabled bin produces a separate mesh file.

---

## Output

- One mesh per enabled intensity bin
- Filenames follow the chosen prefix and bin name
- Output formats depend on meshing backend (PLY by default)

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
- Progress updates are emitted slice-by-slice or bin-by-bin
- Histogram visualization is handled via PyQtGraph
- Zero-intensity voxels are excluded from histogram scaling

---

## Roadmap

- Manual slice annotation and masking
- Preset-based workflows (bone, soft tissue, etc.)
- Additional mesh export formats
- Volume preview (orthogonal slice views)
- Batch and headless processing
- Extended documentation and examples

---

## License

License to be defined (MIT recommended).
