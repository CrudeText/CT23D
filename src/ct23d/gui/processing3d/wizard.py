from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QLabel,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QColorDialog,
    QSlider,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QGridLayout,
)

from ct23d.core.volume import CanonicalVolume, load_volume_nrrd
from ct23d.core.models import IntensityBin
from ct23d.core import bins as binsmod
from ct23d.core import meshing as meshingmod
from ct23d.gui.status import StatusController
from ct23d.gui.workers import FunctionWorker, WorkerBase
from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QApplication

try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
    PVISTA_AVAILABLE = True
except ImportError:
    PVISTA_AVAILABLE = False


class Processing3DWizard(QWidget):
    """
    3D Processing tab widget.
    
    - Load canonical volume from NRRD files
    - Display volume metadata
    - Placeholder UI for mesh generation and ML operations
    """

    def __init__(self, status: StatusController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.status = status
        
        self._canonical_volume: Optional[CanonicalVolume] = None
        self._volume_path: Optional[Path] = None
        self._stats_worker: Optional[FunctionWorker] = None  # Keep reference to prevent garbage collection
        self._viewer: Optional[QtInteractor] = None  # 3D viewer instance
        self._current_meshes: list[tuple[IntensityBin, pv.PolyData]] = []  # List of (bin, mesh) tuples
        self._mesh_actors: list = []  # List of mesh actors for removal/updates
        self._mesh_worker: Optional[WorkerBase] = None  # Keep reference to mesh generation worker
        self._opacity_worker: Optional[WorkerBase] = None  # Keep reference to opacity changes worker
        self._mesh_opacity_spinboxes: list[QSpinBox] = []  # List of opacity spinboxes for each mesh
        self._mesh_opacity_sliders: list[QSlider] = []  # List of opacity sliders for each mesh
        self._mesh_visibility_checkboxes: list = []  # List of visibility checkboxes for each mesh
        self._mesh_original_opacities: list[int] = []  # Store original opacity values to track changes
        self._mesh_display_timer: Optional[QTimer] = None  # Timer for incremental mesh display
        self._pending_meshes: list[tuple[int, IntensityBin, pv.PolyData]] = []  # Meshes waiting to be displayed
        
        self._build_ui()
    
    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top row: Load Canonical Volume button and file path (left-aligned, with space for patient info on right)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.btn_load_volume = QPushButton("Load Canonical Volume...")
        self.btn_load_volume.setMaximumWidth(200)
        self.btn_load_volume.clicked.connect(self._on_load_volume_clicked)
        top_row.addWidget(self.btn_load_volume)
        
        # File path label next to button
        self.info_path_label = QLabel("(No volume loaded)")
        self.info_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.info_path_label.setStyleSheet("color: white; padding-left: 10px;")
        top_row.addWidget(self.info_path_label)
        
        top_row.addStretch()  # Push to left, leaving space on right for patient info
        # Add spacing to ensure load button ends before patient info box (patient info is ~380px wide)
        top_row.addSpacing(400)  # Reserve space for patient info box
        main_layout.addLayout(top_row)
        
        # Add small spacing between button and controls
        main_layout.addSpacing(10)
        
        # Volume Info and Mesh Generation boxes (horizontal layout, right under load button)
        top_controls_layout = QHBoxLayout()
        top_controls_layout.setSpacing(15)
        
        # Volume Info panel (read-only) - 2 columns
        info_group = QGroupBox("Volume Info")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(4)  # Reduced spacing between lines
        info_layout.setVerticalSpacing(4)  # Reduced vertical spacing
        info_layout.setColumnStretch(1, 1)  # Make value columns stretchable
        info_layout.setColumnStretch(3, 1)
        
        self.info_shape_label = QLabel("(No volume loaded)")
        self.info_dtype_label = QLabel("(No volume loaded)")
        self.info_spacing_label = QLabel("(No volume loaded)")
        self.info_intensity_kind_label = QLabel("(No volume loaded)")
        self.info_min_label = QLabel("(No volume loaded)")
        self.info_max_label = QLabel("(No volume loaded)")
        self.info_mean_label = QLabel("(No volume loaded)")
        
        # Make labels selectable for copy-paste
        for label in [
            self.info_shape_label, self.info_dtype_label, self.info_spacing_label,
            self.info_intensity_kind_label, self.info_min_label, self.info_max_label,
            self.info_mean_label
        ]:
            label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            label.setStyleSheet("color: white;")
        
        # Column 1 (left column)
        info_layout.addWidget(QLabel("Shape (Z,Y,X):"), 0, 0)
        info_layout.addWidget(self.info_shape_label, 0, 1)
        info_layout.addWidget(QLabel("Data type:"), 1, 0)
        info_layout.addWidget(self.info_dtype_label, 1, 1)
        info_layout.addWidget(QLabel("Spacing (sx,sy,sz):"), 2, 0)
        info_layout.addWidget(self.info_spacing_label, 2, 1)
        info_layout.addWidget(QLabel("Intensity kind:"), 3, 0)
        info_layout.addWidget(self.info_intensity_kind_label, 3, 1)
        
        # Column 2 (right column)
        info_layout.addWidget(QLabel("Min:"), 0, 2)
        info_layout.addWidget(self.info_min_label, 0, 3)
        info_layout.addWidget(QLabel("Max:"), 1, 2)
        info_layout.addWidget(self.info_max_label, 1, 3)
        info_layout.addWidget(QLabel("Mean:"), 2, 2)
        info_layout.addWidget(self.info_mean_label, 2, 3)
        
        top_controls_layout.addWidget(info_group)
        top_controls_layout.addStretch()
        
        main_layout.addLayout(top_controls_layout)
        
        # Add spacing between controls and main content
        main_layout.addSpacing(15)
        
        # Main content area: 3D viewer on left, mesh table on right
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Left side: 3D viewer
        viewer_group = QGroupBox("3D Viewer")
        viewer_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_layout.setContentsMargins(8, 12, 8, 8)  # Set margins: left, top (extra for title), right, bottom
        viewer_layout.setSpacing(5)  # Spacing between widgets
        
        # Viewer controls (top row)
        controls_layout = QHBoxLayout()
        self.btn_reset_camera = QPushButton("Reset Camera")
        self.btn_reset_camera.setMaximumWidth(120)
        self.btn_reset_camera.clicked.connect(self._on_reset_camera)
        controls_layout.addWidget(self.btn_reset_camera)
        
        self.btn_toggle_axes = QPushButton("Toggle Axes")
        self.btn_toggle_axes.setMaximumWidth(120)
        self.btn_toggle_axes.setCheckable(True)
        self.btn_toggle_axes.setChecked(False)
        self.btn_toggle_axes.clicked.connect(self._on_toggle_axes)
        controls_layout.addWidget(self.btn_toggle_axes)
        
        self.btn_background_color = QPushButton("Background Color")
        self.btn_background_color.setMaximumWidth(140)
        self.btn_background_color.clicked.connect(self._on_background_color)
        controls_layout.addWidget(self.btn_background_color)
        
        controls_layout.addStretch()
        viewer_layout.addLayout(controls_layout)
        
        # Create PyVistaQt viewer
        if PVISTA_AVAILABLE:
            # Create QtInteractor widget
            # QtInteractor is a QWidget that can be added directly to layout
            self._viewer = QtInteractor(viewer_group)
            self._viewer.set_background("black")  # Dark background
            
            # Enable depth peeling for proper transparency rendering
            try:
                render_window = self._viewer.renderer.GetRenderWindow()
                if render_window is not None:
                    # Enable depth peeling
                    render_window.SetAlphaBitPlanes(1)
                    render_window.SetMultiSamples(0)  # Disable multisampling for depth peeling
                    # Enable depth peeling on renderer
                    self._viewer.renderer.SetUseDepthPeeling(True)
                    self._viewer.renderer.SetMaximumNumberOfPeels(4)
                    self._viewer.renderer.SetOcclusionRatio(0.0)
            except Exception:
                # Depth peeling not available, continue without it
                pass
            
            viewer_layout.addWidget(self._viewer)
        else:
            # Fallback placeholder if PyVistaQt is not available
            viewer_placeholder = QLabel(
                "3D Viewer\n(PyVistaQt not available.\n"
                "Install with: pip install pyvista pyvistaqt)"
            )
            viewer_placeholder.setAlignment(Qt.AlignCenter)
            viewer_placeholder.setStyleSheet("color: gray; font-style: italic; background-color: #2b2b2b; min-height: 400px;")
            viewer_layout.addWidget(viewer_placeholder)
        
        content_layout.addWidget(viewer_group, stretch=3)  # Give viewer more space (stretch=3)
        
        # Right side: Mesh Visibility and Properties (big box containing table, mesh generation, and apply button)
        mesh_table_group = QGroupBox("Mesh Visibility and Properties")
        mesh_table_layout = QVBoxLayout(mesh_table_group)
        mesh_table_layout.setSpacing(10)
        
        # Create table with columns: Visible, Color, Intensity Range, Opacity
        self.meshes_table = QTableWidget(0, 4, self)
        self.meshes_table.setHorizontalHeaderLabels(["Visible", "Color", "Intensity Range", "Opacity"])
        self.meshes_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Read-only except opacity
        self.meshes_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.meshes_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.meshes_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.meshes_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.meshes_table.verticalHeader().setVisible(False)  # Hide row numbers
        self.meshes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.meshes_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Make table larger
        self.meshes_table.setMinimumHeight(400)
        self.meshes_table.setMinimumWidth(350)
        # Connect itemChanged to track visibility checkbox changes (but don't render immediately)
        self.meshes_table.itemChanged.connect(self._on_mesh_visibility_changed)
        
        mesh_table_layout.addWidget(self.meshes_table)
        
        # Mesh Generation section - moved here, under table
        mesh_group = QGroupBox("Mesh Generation")
        mesh_layout = QGridLayout(mesh_group)
        mesh_layout.setSpacing(4)  # Reduced spacing
        mesh_layout.setVerticalSpacing(4)  # Reduced vertical spacing
        mesh_layout.setColumnStretch(1, 1)  # Make second column stretchable
        mesh_layout.setColumnMinimumWidth(1, 200)  # Set minimum width for controls column
        
        # Row 0: Source on its own line
        mesh_layout.addWidget(QLabel("Source:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Intensity threshold", "Label volume (future)"])
        self.source_combo.currentIndexChanged.connect(self._on_mesh_param_changed)
        mesh_layout.addWidget(self.source_combo, 0, 1)
        
        # Row 1: Intensity threshold on its own line
        mesh_layout.addWidget(QLabel("Intensity threshold:"), 1, 0)
        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(4)
        threshold_row.addWidget(QLabel("Min:"))
        self.threshold_min_spin = QSpinBox()
        self.threshold_min_spin.setRange(-10000, 10000)
        self.threshold_min_spin.setSingleStep(1)
        self.threshold_min_spin.setValue(0)
        self.threshold_min_spin.setEnabled(False)  # Disabled until volume is loaded
        self.threshold_min_spin.setMinimumWidth(100)  # Set minimum width for spinbox
        self.threshold_min_spin.valueChanged.connect(self._on_mesh_param_changed)
        threshold_row.addWidget(self.threshold_min_spin)
        threshold_row.addWidget(QLabel("Max:"))
        self.threshold_max_spin = QSpinBox()
        self.threshold_max_spin.setRange(-10000, 10000)
        self.threshold_max_spin.setSingleStep(1)
        self.threshold_max_spin.setValue(1000)
        self.threshold_max_spin.setEnabled(False)  # Disabled until volume is loaded
        self.threshold_max_spin.setMinimumWidth(100)  # Set minimum width for spinbox
        self.threshold_max_spin.valueChanged.connect(self._on_mesh_param_changed)
        threshold_row.addWidget(self.threshold_max_spin)
        threshold_row.addStretch()  # Add stretch to push controls to the left
        threshold_widget = QWidget()
        threshold_widget.setLayout(threshold_row)
        mesh_layout.addWidget(threshold_widget, 1, 1)
        
        # Row 2: LOD, Smooth shading, and Show edges on same line
        mesh_layout.addWidget(QLabel("LOD:"), 2, 0)
        self.lod_combo = QComboBox()
        self.lod_combo.addItems(["Preview", "Standard", "High"])
        self.lod_combo.setCurrentIndex(0)  # Default to Preview for better performance
        self.lod_combo.currentIndexChanged.connect(self._on_mesh_param_changed)
        mesh_layout.addWidget(self.lod_combo, 2, 1)
        
        self.smooth_shading_cb = QCheckBox("Smooth shading")
        self.smooth_shading_cb.setChecked(False)  # Default to flat for better performance
        self.smooth_shading_cb.setEnabled(False)  # Enabled when mesh is displayed
        self.smooth_shading_cb.toggled.connect(self._on_mesh_param_changed)
        mesh_layout.addWidget(self.smooth_shading_cb, 2, 2)
        
        self.show_edges_cb = QCheckBox("Show edges")
        self.show_edges_cb.setChecked(False)  # Default to hidden
        self.show_edges_cb.setEnabled(False)  # Enabled when mesh is displayed
        self.show_edges_cb.toggled.connect(self._on_mesh_param_changed)
        mesh_layout.addWidget(self.show_edges_cb, 2, 3)
        
        mesh_table_layout.addWidget(mesh_group)
        
        # Apply Changes button
        apply_button_layout = QHBoxLayout()
        apply_button_layout.addStretch()
        self.btn_apply_changes = QPushButton("Apply Changes")
        self.btn_apply_changes.setEnabled(False)  # Disabled until changes are made
        self.btn_apply_changes.clicked.connect(self._on_apply_changes_clicked)
        apply_button_layout.addWidget(self.btn_apply_changes)
        apply_button_layout.addStretch()
        mesh_table_layout.addLayout(apply_button_layout)
        
        # Add mesh table group to content layout (now it's a QGroupBox, not a widget)
        mesh_table_group.setMinimumWidth(450)  # Minimum width for the big box
        mesh_table_group.setMaximumWidth(550)  # Maximum width for the big box
        content_layout.addWidget(mesh_table_group, stretch=1)  # Give table less space (stretch=1)
        
        # ML section (placeholders) - outside the big box
        ml_group = QGroupBox("ML Operations")
        ml_layout = QVBoxLayout(ml_group)
        
        ml_button_layout = QHBoxLayout()
        self.btn_run_organ_mapping = QPushButton("Run Organ Mapping (placeholder)")
        self.btn_run_organ_mapping.setMaximumWidth(220)
        self.btn_run_organ_mapping.clicked.connect(self._on_run_organ_mapping_clicked)
        ml_button_layout.addWidget(self.btn_run_organ_mapping)
        ml_button_layout.addStretch()
        ml_layout.addLayout(ml_button_layout)
        
        # Future ML settings area (disabled for now)
        ml_settings_area = QLabel("(Future: Model selection and settings)")
        ml_settings_area.setStyleSheet("color: gray; font-style: italic;")
        ml_settings_area.setEnabled(False)
        ml_layout.addWidget(ml_settings_area)
        
        # Add ML section below the content layout
        main_layout.addWidget(ml_group)
        
        main_layout.addLayout(content_layout, stretch=1)  # Main content area takes most space
        
        # Bottom section: Reserved for exporting etc. (empty for now, will be added later)
        bottom_section = QHBoxLayout()
        # Placeholder for future export controls
        bottom_section.addStretch()
        main_layout.addLayout(bottom_section)
    
    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_lod_step_size(self) -> int:
        """
        Get step_size for marching_cubes based on LOD setting.
        
        Returns:
            step_size: 1 for High, 2 for Standard, 4 for Preview
        """
        lod_text = self.lod_combo.currentText()
        if lod_text == "Preview":
            return 4  # 4x downsampling = 16x fewer vertices
        elif lod_text == "Standard":
            return 2  # 2x downsampling = 4x fewer vertices
        else:  # High
            return 1  # Full resolution
    
    # ------------------------------------------------------------------ #
    # Load volume
    # ------------------------------------------------------------------ #
    def _on_load_volume_clicked(self) -> None:
        """Handle Load Canonical Volume button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Canonical Volume",
            "",
            "NRRD files (*.nrrd);;All files (*.*)"
        )
        
        if not path:
            return
        
        self._load_volume(Path(path))
    
    def _load_volume(self, path: Path) -> None:
        """Load canonical volume from file and automatically generate mesh."""
        from PySide6.QtCore import Signal
        
        # Create a combined worker that loads volume and generates meshes for each bin
        class LoadAndMeshWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def __init__(self, nrrd_path: Path, step_size: int):
                super().__init__()
                self.nrrd_path = nrrd_path
                self.step_size = step_size
                self.volume: Optional[CanonicalVolume] = None
                self.meshes: list[tuple[IntensityBin, pv.PolyData]] = []  # List of (bin, mesh) tuples
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    # Phase 1: Load NRRD file (overall phase 0)
                    self.phase_progress.emit("Reading NRRD file", 0, 1, 0)
                    self.volume = load_volume_nrrd(self.nrrd_path)
                    self.phase_progress.emit("Reading NRRD file", 1, 1, 0)
                    
                    # Phase 2: Parse bins from provenance
                    bins: list[IntensityBin] = []
                    if self.volume.provenance:
                        intensity_bins_data = self.volume.provenance.get("intensity_bins")
                        if intensity_bins_data and isinstance(intensity_bins_data, list):
                            for idx, bin_data in enumerate(intensity_bins_data):
                                if isinstance(bin_data, dict):
                                    color_val = bin_data.get("color")
                                    color_tuple = None
                                    if color_val and isinstance(color_val, (list, tuple)) and len(color_val) >= 3:
                                        # Normalize color to 0-1 range if needed
                                        r, g, b = color_val[0], color_val[1], color_val[2]
                                        if r > 1.0 or g > 1.0 or b > 1.0:
                                            r, g, b = r / 255.0, g / 255.0, b / 255.0
                                        color_tuple = (float(r), float(g), float(b))
                                    
                                    bins.append(IntensityBin(
                                        index=bin_data.get("index", idx),
                                        low=bin_data.get("low", 0),
                                        high=bin_data.get("high", 255),
                                        name=bin_data.get("name", f"bin_{idx:02d}"),
                                        color=color_tuple,
                                        enabled=bin_data.get("enabled", True),
                                    ))
                    
                    if not bins:
                        # No bins in provenance - create a fallback mesh using a simple threshold
                        self.phase_progress.emit("Extracting mesh", 0, 1, 1)
                        vol = self.volume.data
                        threshold = float(vol.min())
                        mask = (vol >= threshold).astype(np.float32)
                        from skimage import measure
                        sx, sy, sz = self.volume.spacing
                        spacing_for_marching = (sz, sy, sx)
                        # Apply step_size for LOD: larger step_size = fewer vertices = faster rendering
                        verts_physical, faces, _normals, _values = measure.marching_cubes(
                            mask, level=0.5, spacing=spacing_for_marching, step_size=self.step_size
                        )
                        if len(faces) > 0:
                            verts_pyvista = np.column_stack([
                                verts_physical[:, 2], verts_physical[:, 1], verts_physical[:, 0]
                            ])
                            faces_vtk = np.empty((len(faces), 4), dtype=np.int32)
                            faces_vtk[:, 0] = 3
                            faces_vtk[:, 1:] = faces
                            mesh = pv.PolyData(verts_pyvista, faces_vtk.flatten())
                            mesh = mesh.compute_normals()
                            # Use white color for fallback
                            fallback_bin = IntensityBin(
                                index=0, low=int(threshold), high=int(vol.max()),
                                name="fallback", color=(1.0, 1.0, 1.0), enabled=True
                            )
                            self.meshes = [(fallback_bin, mesh)]
                        self.phase_progress.emit("Extracting mesh", 1, 1, 1)
                        return
                    
                    # Filter to enabled bins only
                    enabled_bins = [b for b in bins if b.enabled]
                    total = len(enabled_bins)
                    
                    if total == 0:
                        return
                    
                    # Phase 3: Build masks for each bin (overall phase 1)
                    self.phase_progress.emit("Building masks", 0, total, total)
                    bin_masks = []
                    vol = self.volume.data
                    for i, bin_ in enumerate(enabled_bins):
                        self.phase_progress.emit("Building masks", i + 1, total, total * 2)
                        bin_mask = binsmod.build_bin_mask(vol, bin_)
                        bin_masks.append(bin_mask)
                    
                    # Phase 4: Extract meshes for each bin (overall phase 2)
                    self.phase_progress.emit("Extracting meshes", 0, total, total * 2)
                    from skimage import measure
                    sx, sy, sz = self.volume.spacing
                    spacing_for_marching = (sz, sy, sx)  # Convert to (z, y, x) array order
                    
                    for i, (bin_, bin_mask) in enumerate(zip(enabled_bins, bin_masks)):
                        self.phase_progress.emit("Extracting meshes", i + 1, total, total * 2)
                        
                        if np.count_nonzero(bin_mask) == 0:
                            continue  # Skip empty bins
                        
                        # Extract mesh using marching_cubes directly with LOD step_size
                        # This is faster than using extract_mesh which doesn't support step_size
                        mask_float = bin_mask.astype(np.float32)
                        # Apply step_size for LOD: larger step_size = fewer vertices = faster rendering
                        verts_physical, faces, _normals, _values = measure.marching_cubes(
                            mask_float,
                            level=0.5,
                            spacing=spacing_for_marching,
                            step_size=self.step_size,
                        )
                        
                        if len(faces) == 0:
                            continue  # Skip empty meshes
                        
                        # Convert vertices from (z, y, x) array order to (x, y, z) physical space
                        verts = np.column_stack([
                            verts_physical[:, 2],  # x
                            verts_physical[:, 1],  # y
                            verts_physical[:, 0],  # z
                        ])
                        
                        # Convert to PyVista PolyData
                        n_faces = len(faces)
                        faces_vtk = np.empty((n_faces, 4), dtype=np.int32)
                        faces_vtk[:, 0] = 3
                        faces_vtk[:, 1:] = faces
                        mesh = pv.PolyData(verts, faces_vtk.flatten())
                        mesh = mesh.compute_normals()
                        
                        # Apply bin color to all vertices
                        if bin_.color is not None:
                            r, g, b = bin_.color
                            # Convert to 0-255 range
                            r_int = int(r * 255)
                            g_int = int(g * 255)
                            b_int = int(b * 255)
                            n_verts = mesh.n_points
                            vertex_colors = np.full((n_verts, 3), [r_int, g_int, b_int], dtype=np.uint8)
                            mesh['colors'] = vertex_colors
                        
                        self.meshes.append((bin_, mesh))
                    
                    self.finished.emit((self.volume, self.meshes))
                    
                except Exception as e:
                    self.error.emit(str(e))
        
        # Get step_size from LOD setting
        step_size = self._get_lod_step_size()
        worker = LoadAndMeshWorker(path, step_size)
        
        def on_success(result: tuple) -> None:
            volume, meshes = result
            self._canonical_volume = volume
            self._volume_path = path
            self._update_volume_info_basic()
            
            # Update stats
            data = volume.data
            min_val = float(data.min())
            max_val = float(data.max())
            mean_val = float(data.mean())
            self.info_min_label.setText(f"{min_val:.2f}")
            self.info_max_label.setText(f"{max_val:.2f}")
            self.info_mean_label.setText(f"{mean_val:.2f}")
            
            # Enable threshold controls and set ranges
            min_int = int(np.floor(min_val))
            max_int = int(np.ceil(max_val))
            self.threshold_min_spin.setRange(min_int, max_int)
            self.threshold_max_spin.setRange(min_int, max_int)
            
            # Set default threshold values based on volume data
            default_min = max(min_int, int(round(max(min_val, mean_val * 0.3))))
            default_max = max(default_min + 1, min(int(round(max_val)), max_int))
            self.threshold_min_spin.setValue(default_min)
            self.threshold_max_spin.setValue(default_max)
            self.threshold_min_spin.setEnabled(True)
            self.threshold_max_spin.setEnabled(True)
            
            # Store meshes with bin information
            self._current_meshes = meshes
            
            # Display meshes
            if meshes:
                total_points = sum(mesh.n_points for _, mesh in meshes)
                if total_points > 0:
                    self._display_meshes(meshes)
                    bin_count = len(meshes)
                    self.status.show_message(f"Volume loaded with {bin_count} bin mesh{'es' if bin_count > 1 else ''}: {path.name}", timeout_ms=3000)
                else:
                    self.status.show_error("Mesh generation produced empty meshes.")
            else:
                self.status.show_error("No meshes generated. Check that bins are enabled in the NRRD file.")
        
        def on_error(error_msg: str) -> None:
            self.status.show_error(f"Failed to load volume: {error_msg}")
        
        self.status.run_threaded_with_progress(
            worker,
            "Loading canonical volume and generating mesh...",
            on_success=on_success,
        )
        worker.error.connect(on_error)
    
    def _update_volume_info_basic(self) -> None:
        """Update Volume Info panel with basic volume data (shape, dtype, spacing, etc.)."""
        if self._canonical_volume is None:
            self.info_shape_label.setText("(No volume loaded)")
            self.info_dtype_label.setText("(No volume loaded)")
            self.info_spacing_label.setText("(No volume loaded)")
            self.info_intensity_kind_label.setText("(No volume loaded)")
            self.info_min_label.setText("(No volume loaded)")
            self.info_max_label.setText("(No volume loaded)")
            self.info_mean_label.setText("(No volume loaded)")
            self.info_path_label.setText("(No volume loaded)")
            # Disable controls when no volume is loaded
            self.threshold_min_spin.setEnabled(False)
            self.threshold_max_spin.setEnabled(False)
            return
        
        volume = self._canonical_volume
        
        # Update shape
        shape = volume.data.shape
        self.info_shape_label.setText(f"({shape[0]}, {shape[1]}, {shape[2]})")
        
        # Update dtype
        self.info_dtype_label.setText(str(volume.data.dtype))
        
        # Update spacing
        sx, sy, sz = volume.spacing
        self.info_spacing_label.setText(f"({sx:.4f}, {sy:.4f}, {sz:.4f})")
        
        # Update intensity kind
        self.info_intensity_kind_label.setText(volume.intensity_kind)
        
        # Update path
        if self._volume_path:
            self.info_path_label.setText(str(self._volume_path))
        else:
            self.info_path_label.setText("(Path unknown)")
    
    def _compute_volume_stats(self) -> None:
        """Compute volume statistics in background to avoid UI freeze."""
        if self._canonical_volume is None:
            return
        
        # Clean up previous stats worker if it exists
        if self._stats_worker is not None:
            if self._stats_worker.isRunning():
                self._stats_worker.terminate()
                self._stats_worker.wait(1000)
            self._stats_worker = None
        
        # Show "Computing..." while stats are calculated
        self.info_min_label.setText("(Computing...)")
        self.info_max_label.setText("(Computing...)")
        self.info_mean_label.setText("(Computing...)")
        
        def compute_task() -> tuple[float, float, float]:
            data = self._canonical_volume.data
            return (float(data.min()), float(data.max()), float(data.mean()))
        
        worker = FunctionWorker(compute_task)
        self._stats_worker = worker  # Keep reference to prevent garbage collection
        
        def on_success(stats: tuple[float, float, float]) -> None:
            min_val, max_val, mean_val = stats
            self.info_min_label.setText(f"{min_val:.2f}")
            self.info_max_label.setText(f"{max_val:.2f}")
            self.info_mean_label.setText(f"{mean_val:.2f}")
            
            # Update threshold ranges and set smart defaults (not full range)
            if self._canonical_volume is not None:
                # QSpinBox requires integer ranges
                min_int = int(np.floor(min_val))
                max_int = int(np.ceil(max_val))
                self.threshold_min_spin.setRange(min_int, max_int)
                self.threshold_max_spin.setRange(min_int, max_int)
                
                # Try to get default min from lowest intensity bin in provenance
                default_min = None
                if self._canonical_volume.provenance:
                    intensity_bins = self._canonical_volume.provenance.get("intensity_bins")
                    if intensity_bins and isinstance(intensity_bins, list) and len(intensity_bins) > 0:
                        # Find the bin with the minimum "low" value
                        lowest_bin = min(intensity_bins, key=lambda b: b.get("low", float('inf')))
                        if "low" in lowest_bin:
                            default_min = int(lowest_bin["low"])
                
                # Fallback: use mean-based thresholding if no bins in provenance
                if default_min is None:
                    mean_val = self._canonical_volume.data.mean()
                    default_min = int(round(max(min_val, mean_val * 0.3)))
                
                # Default max: use actual max (this is usually fine as upper bound)
                default_max = int(round(max_val))
                
                # Ensure min < max and within range
                default_min = max(min_int, min(default_min, max_int - 1))
                default_max = max(default_min + 1, min(default_max, max_int))
                
                self.threshold_min_spin.setValue(default_min)
                self.threshold_max_spin.setValue(default_max)
                
                # Automatically generate mesh with default threshold values
                if self._viewer is not None and self._canonical_volume is not None:
                    self._generate_mesh_async(default_min, default_max)
            
            # Clear reference when done
            if self._stats_worker == worker:
                self._stats_worker = None
        
        def on_error(error_msg: str) -> None:
            self.info_min_label.setText("(Error)")
            self.info_max_label.setText("(Error)")
            self.info_mean_label.setText("(Error)")
            # Clear reference when done
            if self._stats_worker == worker:
                self._stats_worker = None
        
        # Use background worker but without progress dialog (silent computation)
        worker.finished.connect(on_success)
        worker.error.connect(on_error)
        worker.start()
    
    # ------------------------------------------------------------------ #
    # Viewer controls
    # ------------------------------------------------------------------ #
    def _on_reset_camera(self) -> None:
        """Reset camera to default view."""
        if self._viewer is not None:
            try:
                if self._current_meshes:
                    # Reset camera to fit all meshes
                    # Compute combined bounds from all meshes
                    all_bounds = [mesh.bounds for _, mesh in self._current_meshes if mesh.n_points > 0]
                    if all_bounds:
                        # Get min/max across all bounds
                        min_bounds = [min(b[i] for b in all_bounds) for i in range(0, 6, 2)]
                        max_bounds = [max(b[i] for b in all_bounds) for i in range(1, 6, 2)]
                        combined_bounds = [min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2]]
                        self._viewer.reset_camera(bounds=combined_bounds)
                    else:
                        self._viewer.reset_camera()
                else:
                    # No meshes loaded, just reset camera
                    self._viewer.reset_camera()
                # Force render and update
                self._viewer.render()
                self._viewer.update()
            except Exception as e:
                # Camera reset failed - non-critical, continue
                pass
    
    def _on_toggle_axes(self, checked: bool) -> None:
        """Toggle axes display."""
        if self._viewer is None:
            return
        
        # Use PyVista's axes widget functionality
        if checked:
            self._viewer.show_axes()
        else:
            self._viewer.hide_axes()
        
        self._viewer.update()
    
    def _on_background_color(self) -> None:
        """Change background color."""
        if self._viewer is None:
            return
        
        # Get current background color (PyVista stores it as a tuple of RGB values 0-1)
        try:
            current_bg = self._viewer.background_color
            if isinstance(current_bg, (tuple, list)) and len(current_bg) >= 3:
                # Convert to 0-255 range for QColor
                color = QColor(
                    int(current_bg[0] * 255) if current_bg[0] <= 1.0 else int(current_bg[0]),
                    int(current_bg[1] * 255) if current_bg[1] <= 1.0 else int(current_bg[1]),
                    int(current_bg[2] * 255) if current_bg[2] <= 1.0 else int(current_bg[2])
                )
            else:
                color = QColor(0, 0, 0)  # Default to black
        except (AttributeError, TypeError):
            color = QColor(0, 0, 0)  # Default to black
        
        # Show color dialog
        color = QColorDialog.getColor(color, self, "Select Background Color")
        if color.isValid():
            # Convert to 0-1 range for PyVista
            r, g, b, _ = color.getRgbF()
            self._viewer.set_background((r, g, b))
            self._viewer.update()
    
    # ------------------------------------------------------------------ #
    # Mesh generation
    # ------------------------------------------------------------------ #
    def _on_generate_mesh_clicked(self) -> None:
        """Handle Generate Mesh button click."""
        if self._canonical_volume is None:
            self.status.show_error("Please load a volume first.")
            return
        
        if self._viewer is None:
            self.status.show_error("3D viewer is not available.")
            return
        
        # Get threshold values
        tmin = self.threshold_min_spin.value()
        tmax = self.threshold_max_spin.value()
        
        if tmin >= tmax:
            self.status.show_error("Threshold min must be less than threshold max.")
            return
        
        # Start mesh generation in background thread
        self._generate_mesh_async(tmin, tmax)
    
    def _generate_mesh_async(self, tmin: float, tmax: float) -> None:
        """Generate mesh in background thread."""
        if self._canonical_volume is None:
            return
        
        # Clean up previous mesh worker if it exists
        if self._mesh_worker is not None:
            if self._mesh_worker.isRunning():
                self._mesh_worker.terminate()
                self._mesh_worker.wait(1000)
            self._mesh_worker = None
        
        # Create mesh generation worker
        class MeshGenerationWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def __init__(self, volume: CanonicalVolume, tmin: float, tmax: float, step_size: int):
                super().__init__()
                self.volume = volume
                self.tmin = tmin
                self.tmax = tmax
                self.step_size = step_size
                self.result: Optional[pv.PolyData] = None
                self.mask_size: int = 0
                self.warning_message: Optional[str] = None
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    # Phase 1: Create binary mask
                    self.phase_progress.emit("Creating mask", 0, 1, 0)
                    vol = self.volume.data
                    
                    # Use tmin as the threshold (simple >= threshold, not a range)
                    threshold = float(self.tmin)
                    
                    self.phase_progress.emit("Creating mask", 1, 1, 1)
                    
                    # Phase 2: Extract mesh using marching cubes (EXACT same as view_nrrd.py)
                    self.phase_progress.emit("Extracting mesh", 0, 1, 1)
                    from skimage import measure
                    
                    # Create binary mask using threshold (EXACT same as view_nrrd.py)
                    mask = (vol >= threshold).astype(np.float32)
                    
                    # Run marching cubes with actual spacing and LOD step_size
                    # volume.spacing is (sx, sy, sz) in physical space order (x, y, z)
                    # marching_cubes expects spacing in array order (z, y, x), so convert (sz, sy, sx)
                    sx, sy, sz = self.volume.spacing
                    spacing_for_marching = (sz, sy, sx)  # Convert to (z, y, x) array order
                    # Apply step_size for LOD: larger step_size = fewer vertices = faster rendering
                    # step_size affects all dimensions equally
                    verts_physical, faces, _normals, _values = measure.marching_cubes(
                        mask,
                        level=0.5,
                        spacing=spacing_for_marching,
                        step_size=self.step_size,
                    )
                    
                    # Check if we got any faces
                    if len(faces) == 0:
                        # Empty mesh
                        mesh = pv.PolyData()
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        self.phase_progress.emit("Scaling mesh", 1, 1, 3)
                    else:
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        
                        # Phase 3: Convert to PyVista coordinate system (vertices already in physical space)
                        self.phase_progress.emit("Scaling mesh", 0, 1, 2)
                        # marching_cubes returns vertices in (z, y, x) array order in physical space
                        # PyVista expects (x, y, z) physical space order
                        verts_pyvista = np.column_stack([
                            verts_physical[:, 2],  # x (was sx)
                            verts_physical[:, 1],  # y (was sy)
                            verts_physical[:, 0],  # z (was sz)
                        ])
                        
                        # Convert faces to VTK format (EXACT same as view_nrrd.py)
                        n_faces = len(faces)
                        faces_vtk = np.empty((n_faces, 4), dtype=np.int32)
                        faces_vtk[:, 0] = 3
                        faces_vtk[:, 1:] = faces
                        faces_vtk = faces_vtk.flatten()
                        
                        # Create mesh
                        mesh = pv.PolyData(verts_pyvista, faces_vtk)
                        mesh = mesh.compute_normals()
                        self.phase_progress.emit("Scaling mesh", 1, 1, 3)
                    
                    self.result = mesh
                    self.finished.emit(mesh)
                    
                except Exception as e:
                    self.error.emit(str(e))
        
        # Get step_size from LOD setting
        step_size = self._get_lod_step_size()
        worker = MeshGenerationWorker(self._canonical_volume, tmin, tmax, step_size)
        self._mesh_worker = worker  # Keep reference
        
        def on_success(result: object) -> None:
            # result is the mesh (pv.PolyData)
            if isinstance(result, pv.PolyData):
                # Check if mesh is empty
                if result.n_points == 0:
                    self.status.show_error(
                        "Mesh generation produced an empty mesh. "
                        "Try adjusting the threshold values - no voxels match the selected range."
                    )
                else:
                    # Manual mesh generation - convert to bin format for display
                    # Create a fallback bin for the threshold-based mesh
                    from ct23d.core.models import IntensityBin
                    fallback_bin = IntensityBin(
                        index=0,
                        low=int(tmin),
                        high=int(tmax),
                        name="threshold_mesh",
                        color=(1.0, 1.0, 1.0),  # White color
                        enabled=True
                    )
                    self._current_meshes = [(fallback_bin, result)]
                    self._display_meshes(self._current_meshes)
                    # Show warning if mask was large
                    if worker.warning_message:
                        self.status.show_info(worker.warning_message)
                    else:
                        self.status.show_message(
                            f"Mesh generated: {len(result.points):,} vertices, "
                            f"{result.n_cells:,} faces",
                            timeout_ms=3000
                        )
            # Clear reference when done
            if self._mesh_worker == worker:
                self._mesh_worker = None
        
        def on_error(error_msg: str) -> None:
            self.status.show_error(f"Mesh generation failed: {error_msg}")
            # Clear reference when done
            if self._mesh_worker == worker:
                self._mesh_worker = None
        
        # If we're already in a loading process (from _load_volume), reuse that progress dialog
        # Otherwise, start a new one
        # Check if there's an active progress dialog by checking if we're in a loading context
        # For now, always use a separate progress dialog for mesh generation
        # (The title will indicate it's part of the overall process)
        self.status.run_threaded_with_progress(
            worker,
            "Generating mesh...",
            on_success=on_success,
        )
        worker.error.connect(on_error)
    
    def _display_meshes(self, meshes: list[tuple[IntensityBin, pv.PolyData]]) -> None:
        """Display meshes in the 3D viewer and populate the table."""
        if self._viewer is None:
            return
        
        # Clear existing meshes
        self._clear_meshes()
        
        # Store meshes
        self._current_meshes = meshes
        
        # Enable mesh display controls
        self.smooth_shading_cb.setEnabled(True)
        self.show_edges_cb.setEnabled(True)
        
        # Get current settings
        smooth_shading = self.smooth_shading_cb.isChecked()
        show_edges = self.show_edges_cb.isChecked()
        
        # Clear and populate the table (fast operation)
        self.meshes_table.setRowCount(len(meshes))
        self._mesh_opacity_spinboxes.clear()
        self._mesh_visibility_checkboxes.clear()
        self._mesh_original_opacities.clear()
        
        # Disable rendering during batch additions
        if self._viewer is not None:
            self._viewer.renderer.GetRenderWindow().SetDesiredUpdateRate(0)
        
        # Populate table first (fast, no mesh operations)
        for row, (bin_, mesh) in enumerate(meshes):
            if mesh.n_points == 0:
                continue
            
            # Table column 0: Visible checkbox
            visible_item = QTableWidgetItem()
            visible_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            visible_item.setCheckState(Qt.Checked)  # Default to visible
            visible_item.setToolTip("Show/hide this mesh in the 3D viewer")
            self.meshes_table.setItem(row, 0, visible_item)
            visible_item.setData(Qt.UserRole, row)  # Store row index
            
            # Table column 1: Color
            color_item = QTableWidgetItem("")
            if bin_.color is not None:
                r, g, b = bin_.color
                # Convert to 0-255 range if needed
                if r <= 1.0 and g <= 1.0 and b <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                else:
                    r, g, b = int(r), int(g), int(b)
                color = QColor(r, g, b)
            else:
                color = QColor(255, 255, 255)  # White fallback
            color_item.setBackground(color)
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self.meshes_table.setItem(row, 1, color_item)
            
            # Table column 2: Intensity Range (read-only)
            range_item = QTableWidgetItem(f"{bin_.low} - {bin_.high}")
            range_item.setFlags(range_item.flags() & ~Qt.ItemIsEditable)
            if bin_.name:
                range_item.setToolTip(f"{bin_.name}: {bin_.low} - {bin_.high}")
            self.meshes_table.setItem(row, 2, range_item)
            
            # Table column 3: Opacity (spinbox + slider)
            opacity_widget = QWidget()
            opacity_layout = QHBoxLayout(opacity_widget)
            opacity_layout.setContentsMargins(0, 0, 0, 0)
            opacity_layout.setSpacing(5)
            
            opacity_spin = QSpinBox()
            opacity_spin.setRange(0, 100)
            opacity_spin.setValue(100)  # Default to fully opaque
            opacity_spin.setSuffix("%")
            opacity_spin.setMinimumWidth(80)
            opacity_spin.setMaximumWidth(80)
            # Connect to handler to track changes (but don't apply immediately)
            opacity_spin.valueChanged.connect(self._on_mesh_opacity_value_changed)
            self._mesh_opacity_spinboxes.append(opacity_spin)
            opacity_layout.addWidget(opacity_spin)
            
            opacity_slider = QSlider(Qt.Horizontal)
            opacity_slider.setRange(0, 100)
            opacity_slider.setValue(100)  # Default to fully opaque
            opacity_slider.setMinimumWidth(100)
            # Connect slider to spinbox (bidirectional)
            opacity_slider.valueChanged.connect(opacity_spin.setValue)
            opacity_spin.valueChanged.connect(opacity_slider.setValue)
            self._mesh_opacity_sliders.append(opacity_slider)
            opacity_layout.addWidget(opacity_slider)
            opacity_layout.addStretch()
            
            self.meshes_table.setCellWidget(row, 3, opacity_widget)
            # Store original opacity
            self._mesh_original_opacities.append(100)
            
            # Store checkbox reference
            self._mesh_visibility_checkboxes.append(visible_item)
        
        # Process events after table population
        QApplication.processEvents()
        
        # Now add meshes to viewer incrementally using a timer
        # This prevents UI freezing
        self._pending_meshes = [(row, bin_, mesh) for row, (bin_, mesh) in enumerate(meshes) if mesh.n_points > 0]
        
        # Initialize actors list
        self._mesh_actors = [None] * len(meshes)
        
        # Disable rendering during batch addition to prevent flashing
        if self._viewer is not None:
            try:
                # Disable automatic rendering completely
                render_window = self._viewer.renderer.GetRenderWindow()
                render_window.SetDesiredUpdateRate(0)
                # Also disable rendering in the renderer
                render_window.SetAbortRender(True)
            except Exception:
                pass
        
        # Use timer to add meshes one at a time
        if self._mesh_display_timer is not None:
            self._mesh_display_timer.stop()
            self._mesh_display_timer.deleteLater()
        
        self._mesh_display_timer = QTimer()
        self._mesh_display_timer.timeout.connect(self._add_next_mesh_to_viewer)
        self._mesh_display_timer.setSingleShot(False)
        self._mesh_display_timer.start(10)  # Add one mesh every 10ms
        
        # Camera reset and final render will happen after all meshes are added
    
    def _add_next_mesh_to_viewer(self) -> None:
        """Add the next pending mesh to the viewer (called by timer)."""
        if not self._pending_meshes or self._viewer is None:
            # All meshes added, stop timer and finalize
            if self._mesh_display_timer is not None:
                self._mesh_display_timer.stop()
            
            # Re-enable rendering
            if self._viewer is not None:
                try:
                    render_window = self._viewer.renderer.GetRenderWindow()
                    render_window.SetAbortRender(False)
                    render_window.SetDesiredUpdateRate(10)
                except Exception:
                    pass
            
            # Reset camera to fit all meshes
            if self._mesh_actors and self._viewer is not None:
                QApplication.processEvents()
                all_bounds = [mesh.bounds for _, mesh in self._current_meshes if mesh.n_points > 0]
                if all_bounds:
                    min_bounds = [min(b[i] for b in all_bounds) for i in range(0, 6, 2)]
                    max_bounds = [max(b[i] for b in all_bounds) for i in range(1, 6, 2)]
                    combined_bounds = [min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2]]
                    self._viewer.reset_camera(bounds=combined_bounds)
            
            # Final render (only once at the end)
            if self._viewer is not None:
                QApplication.processEvents()
                self._viewer.render()
                self._viewer.update()
            return
        
        # Get next mesh to add
        row, bin_, mesh = self._pending_meshes.pop(0)
        
        # Get current settings
        smooth_shading = self.smooth_shading_cb.isChecked()
        show_edges = self.show_edges_cb.isChecked()
        
        # Get opacity from spinbox (defaults to 100 if not yet set)
        opacity = 1.0
        if row < len(self._mesh_opacity_spinboxes):
            opacity = self._mesh_opacity_spinboxes[row].value() / 100.0
        
        # Add mesh to viewer - disable rendering during addition
        try:
            # Temporarily disable rendering
            render_window = self._viewer.renderer.GetRenderWindow()
            old_update_rate = render_window.GetDesiredUpdateRate()
            render_window.SetDesiredUpdateRate(0)
        except Exception:
            old_update_rate = None
        
        try:
            # Try to add mesh without rendering
            if 'colors' in mesh.point_data:
                actor = self._viewer.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                    render=False,  # Don't render immediately
                )
            elif bin_.color is not None:
                r, g, b = bin_.color
                actor = self._viewer.add_mesh(
                    mesh,
                    color=(r, g, b),
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                    render=False,  # Don't render immediately
                )
            else:
                actor = self._viewer.add_mesh(
                    mesh,
                    color='white',
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                    render=False,  # Don't render immediately
                )
        except TypeError:
            # render parameter not supported, fall back to old method
            if 'colors' in mesh.point_data:
                actor = self._viewer.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                )
            elif bin_.color is not None:
                r, g, b = bin_.color
                actor = self._viewer.add_mesh(
                    mesh,
                    color=(r, g, b),
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                )
            else:
                actor = self._viewer.add_mesh(
                    mesh,
                    color='white',
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=show_edges,
                )
        finally:
            # Restore update rate
            if old_update_rate is not None:
                try:
                    render_window.SetDesiredUpdateRate(old_update_rate)
                except Exception:
                    pass
        
        # Set render order based on opacity (higher opacity = render first)
        # This helps ensure opaque meshes are rendered before transparent ones
        try:
            prop = actor.GetProperty()
            if prop is not None:
                # Set render order: higher opacity values get lower render order (rendered first)
                # Render order is inverted: lower number = rendered first
                render_order = int((1.0 - opacity) * 1000)  # 0 for opaque, 1000 for transparent
                prop.SetRenderOrder(render_order)
        except Exception:
            pass
        
        # Ensure we have enough actors in the list
        while len(self._mesh_actors) <= row:
            self._mesh_actors.append(None)
        self._mesh_actors[row] = actor
        
        # Don't render here - rendering is disabled during batch addition
        # Just process events to keep UI responsive
        QApplication.processEvents()
    
    def _clear_meshes(self) -> None:
        """Clear all meshes from the viewer and table."""
        # Stop any pending mesh additions
        if self._mesh_display_timer is not None:
            self._mesh_display_timer.stop()
        self._pending_meshes.clear()
        
        if self._viewer is not None and self._mesh_actors:
            for actor in self._mesh_actors:
                if actor is not None:
                    self._viewer.remove_actor(actor)
            self._mesh_actors = []
            self._viewer.render()
            self._viewer.update()
        
        # Clear table
        self.meshes_table.setRowCount(0)
        self._mesh_opacity_spinboxes.clear()
        self._mesh_opacity_sliders.clear()
        self._mesh_visibility_checkboxes.clear()
        self._mesh_original_opacities.clear()
        self.btn_apply_changes.setEnabled(False)
    
    def _refresh_all_meshes(self) -> None:
        """Clear all actors and re-add only visible meshes. More reliable than removing individual actors."""
        if self._viewer is None or not self._current_meshes:
            return
        
        # Clear all existing actors - use more aggressive removal
        for i, actor in enumerate(self._mesh_actors):
            if actor is not None:
                try:
                    # Try multiple removal methods
                    self._viewer.remove_actor(actor)
                    self._viewer.renderer.RemoveActor(actor)
                except Exception:
                    try:
                        self._viewer.renderer.RemoveActor(actor)
                    except Exception:
                        pass
        
        # Also try clearing all actors from renderer directly
        try:
            actors = self._viewer.renderer.GetActors()
            actors.InitTraversal()
            actor_list = []
            while True:
                actor = actors.GetNextActor()
                if actor is None:
                    break
                actor_list.append(actor)
            
            for actor in actor_list:
                try:
                    self._viewer.renderer.RemoveActor(actor)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Clear actor references
        self._mesh_actors = [None] * len(self._current_meshes)
        
        # Force render to clear the display
        if self._viewer is not None:
            try:
                self._viewer.renderer.Render()
                self._viewer.render()
                self._viewer.update()
            except Exception:
                pass
        
        # Get current settings
        smooth_shading = self.smooth_shading_cb.isChecked()
        show_edges = self.show_edges_cb.isChecked()
        
        # Enable depth peeling for proper transparency rendering
        try:
            render_window = self._viewer.renderer.GetRenderWindow()
            if render_window is not None:
                # Enable depth peeling
                render_window.SetAlphaBitPlanes(1)
                render_window.SetMultiSamples(0)  # Disable multisampling for depth peeling
                # Enable depth peeling on renderer
                self._viewer.renderer.SetUseDepthPeeling(True)
                self._viewer.renderer.SetMaximumNumberOfPeels(4)
                self._viewer.renderer.SetOcclusionRatio(0.0)
        except Exception:
            # Depth peeling not available, continue without it
            pass
        
        # Collect visible meshes with their indices and opacities, then sort by opacity (most opaque first)
        visible_meshes = []
        for i, (bin_, mesh) in enumerate(self._current_meshes):
            if mesh.n_points == 0:
                continue
            
            # Check if mesh is visible
            is_visible = True
            if i < len(self._mesh_visibility_checkboxes):
                checkbox_item = self._mesh_visibility_checkboxes[i]
                is_visible = checkbox_item.checkState() == Qt.Checked
            
            if not is_visible:
                continue  # Skip hidden meshes
            
            # Get opacity for this mesh
            opacity = 1.0
            if i < len(self._mesh_opacity_spinboxes):
                opacity = self._mesh_opacity_spinboxes[i].value() / 100.0
            
            visible_meshes.append((i, bin_, mesh, opacity))
        
        # Sort by opacity (most opaque first) - this ensures opaque meshes render first
        visible_meshes.sort(key=lambda x: x[3], reverse=True)
        
        # Add meshes in sorted order (opaque first, then transparent)
        for i, bin_, mesh, opacity in visible_meshes:
            try:
                if 'colors' in mesh.point_data:
                    actor = self._viewer.add_mesh(
                        mesh,
                        scalars='colors',
                        rgb=True,
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                elif bin_.color is not None:
                    r, g, b = bin_.color
                    actor = self._viewer.add_mesh(
                        mesh,
                        color=(r, g, b),
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                else:
                    actor = self._viewer.add_mesh(
                        mesh,
                        color='white',
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                
                # Set render order based on opacity (higher opacity = render first)
                # This helps ensure opaque meshes are rendered before transparent ones
                try:
                    prop = actor.GetProperty()
                    if prop is not None:
                        # Set render order: higher opacity values get lower render order (rendered first)
                        # Render order is inverted: lower number = rendered first
                        render_order = int((1.0 - opacity) * 1000)  # 0 for opaque, 1000 for transparent
                        prop.SetRenderOrder(render_order)
                except Exception:
                    pass
                
                # Store actor reference
                if i >= len(self._mesh_actors):
                    self._mesh_actors.extend([None] * (i - len(self._mesh_actors) + 1))
                self._mesh_actors[i] = actor
            except Exception:
                pass
        
        # Force render update
        if self._viewer is not None:
            try:
                self._viewer.renderer.Render()
                self._viewer.render()
                self._viewer.update()
                self._viewer.repaint()
            except Exception:
                pass
    
    def _update_mesh_display(self) -> None:
        """Update mesh display properties based on current control settings."""
        if self._viewer is None or not self._current_meshes:
            return
        
        # Get current settings
        smooth_shading = self.smooth_shading_cb.isChecked()
        show_edges = self.show_edges_cb.isChecked()
        
        # Update each visible mesh actor with current opacity and settings
        for i, (bin_, mesh) in enumerate(self._current_meshes):
            if mesh.n_points == 0:
                continue
            
            # Check if mesh is visible
            is_visible = True
            if i < len(self._mesh_visibility_checkboxes):
                is_visible = self._mesh_visibility_checkboxes[i].checkState() == Qt.Checked
            
            if not is_visible:
                # Skip hidden meshes - but remove actor if it exists
                if i < len(self._mesh_actors) and self._mesh_actors[i] is not None:
                    self._viewer.remove_actor(self._mesh_actors[i])
                    self._mesh_actors[i] = None
                continue
            
            # Get opacity for this mesh from the table
            opacity = 1.0
            if i < len(self._mesh_opacity_spinboxes):
                opacity = self._mesh_opacity_spinboxes[i].value() / 100.0
            
            # Check if actor exists
            if i >= len(self._mesh_actors) or self._mesh_actors[i] is None:
                # Need to create the actor
                if 'colors' in mesh.point_data:
                    actor = self._viewer.add_mesh(
                        mesh,
                        scalars='colors',
                        rgb=True,
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                elif bin_.color is not None:
                    r, g, b = bin_.color
                    actor = self._viewer.add_mesh(
                        mesh,
                        color=(r, g, b),
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                else:
                    actor = self._viewer.add_mesh(
                        mesh,
                        color='white',
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                
                # Ensure we have enough actors in the list
                while len(self._mesh_actors) <= i:
                    self._mesh_actors.append(None)
                self._mesh_actors[i] = actor
            else:
                # Update existing actor - remove and re-add with new properties
                actor = self._mesh_actors[i]
                self._viewer.remove_actor(actor)
                
                # Re-add with updated properties
                if 'colors' in mesh.point_data:
                    actor = self._viewer.add_mesh(
                        mesh,
                        scalars='colors',
                        rgb=True,
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                elif bin_.color is not None:
                    r, g, b = bin_.color
                    actor = self._viewer.add_mesh(
                        mesh,
                        color=(r, g, b),
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                else:
                    actor = self._viewer.add_mesh(
                        mesh,
                        color='white',
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                
                self._mesh_actors[i] = actor
        
        # Force render and update
        self._viewer.render()
        self._viewer.update()
    
    def _check_and_enable_apply_button(self) -> None:
        """Check if there are any pending changes and enable/disable Apply Changes button."""
        if not self._current_meshes:
            self.btn_apply_changes.setEnabled(False)
            return
        
        has_changes = False
        
        # Check opacity changes
        for i in range(len(self._mesh_opacity_spinboxes)):
            if i < len(self._mesh_original_opacities):
                if self._mesh_opacity_spinboxes[i].value() != self._mesh_original_opacities[i]:
                    has_changes = True
                    break
        
        # Check visibility changes (compare current checkbox states with what's displayed)
        if not has_changes and self._viewer is not None:
            for i in range(len(self._mesh_visibility_checkboxes)):
                if i < len(self._current_meshes):
                    checkbox_item = self._mesh_visibility_checkboxes[i]
                    is_visible_in_table = checkbox_item.checkState() == Qt.Checked
                    is_visible_in_viewer = (i < len(self._mesh_actors) and self._mesh_actors[i] is not None)
                    if is_visible_in_table != is_visible_in_viewer:
                        has_changes = True
                        break
        
        # Mesh generation parameter changes always enable the button
        # (we assume any change to these requires regeneration)
        
        self.btn_apply_changes.setEnabled(has_changes)
    
    def _on_mesh_opacity_value_changed(self, value: int) -> None:
        """Track opacity value changes (but don't apply immediately)."""
        self._check_and_enable_apply_button()
    
    def _on_apply_changes_clicked(self) -> None:
        """Apply all pending changes: opacity, visibility, and mesh generation parameters."""
        if self._viewer is None or not self._current_meshes:
            return
        
        # Check if mesh generation parameters changed (need to regenerate meshes)
        # For now, we'll just refresh the display with current parameters
        # TODO: In future, if threshold/LOD changed, regenerate meshes
        
        # Apply all changes by refreshing all meshes with current settings
        # This handles opacity, visibility, smooth shading, and show edges
        self._refresh_all_meshes()
        
        # Update original opacities to current values
        for i in range(len(self._mesh_opacity_spinboxes)):
            if i < len(self._mesh_original_opacities):
                self._mesh_original_opacities[i] = self._mesh_opacity_spinboxes[i].value()
        
        # Disable Apply Changes button
        self.btn_apply_changes.setEnabled(False)
        
        # OLD CODE BELOW - keeping for reference but using simpler approach above
        if False:
            # Find which meshes have been modified
            modified_rows = []
            for i in range(len(self._mesh_opacity_spinboxes)):
                if i < len(self._mesh_original_opacities):
                    current_value = self._mesh_opacity_spinboxes[i].value()
                    original_value = self._mesh_original_opacities[i]
                    if current_value != original_value:
                        modified_rows.append(i)
            
            if not modified_rows:
                self.btn_apply_changes.setEnabled(False)
                return
        
        # Create a worker that emits progress signals, but actual mesh updates happen on main thread
        class ApplyOpacityChangesWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            apply_mesh = Signal(int)  # Signal to apply changes for a specific mesh row
            
            def __init__(self, modified_rows: list[int]):
                super().__init__()
                self.modified_rows = modified_rows
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    total = len(self.modified_rows)
                    for idx, row in enumerate(self.modified_rows):
                        # Emit progress with mesh number information
                        self.phase_progress.emit(f"Applying opacity changes (mesh {idx + 1}/{total})", idx + 1, total, total)
                        # Emit signal to apply this mesh (will be handled on main thread)
                        self.apply_mesh.emit(row)
                        # Small delay to allow UI to update
                        self.msleep(10)
                    
                    self.finished.emit(None)
                except Exception as e:
                    self.error.emit(str(e))
        
        worker = ApplyOpacityChangesWorker(modified_rows)
        
        # Connect signal to apply mesh changes on main thread
        def apply_single_mesh(row: int) -> None:
            """Apply opacity change for a single mesh (called on main thread)."""
            if row >= len(self._current_meshes) or row >= len(self._mesh_opacity_spinboxes):
                return
            
            # Get new opacity value
            opacity_value = self._mesh_opacity_spinboxes[row].value()
            opacity = opacity_value / 100.0
            
            # Check if mesh is visible
            is_visible = True
            if row < len(self._mesh_visibility_checkboxes):
                is_visible = self._mesh_visibility_checkboxes[row].checkState() == Qt.Checked
            
            if not is_visible:
                # Just update the stored value, don't update actor
                if row < len(self._mesh_original_opacities):
                    self._mesh_original_opacities[row] = opacity_value
                return
            
            bin_, mesh = self._current_meshes[row]
            
            # Check if actor already exists
            if row < len(self._mesh_actors) and self._mesh_actors[row] is not None:
                # Update existing actor's opacity directly (faster and more reliable)
                actor = self._mesh_actors[row]
                actor_property = actor.GetProperty()
                if actor_property is not None:
                    actor_property.SetOpacity(opacity)
                    # Force render to update the display
                    self._viewer.render()
                    self._viewer.update()
            else:
                # Actor doesn't exist, need to create it
                smooth_shading = self.smooth_shading_cb.isChecked()
                show_edges = self.show_edges_cb.isChecked()
                
                if 'colors' in mesh.point_data:
                    actor = self._viewer.add_mesh(
                        mesh,
                        scalars='colors',
                        rgb=True,
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                elif bin_.color is not None:
                    r, g, b = bin_.color
                    actor = self._viewer.add_mesh(
                        mesh,
                        color=(r, g, b),
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                else:
                    actor = self._viewer.add_mesh(
                        mesh,
                        color='white',
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        show_edges=show_edges,
                    )
                
                # Ensure we have enough actors in the list
                while len(self._mesh_actors) <= row:
                    self._mesh_actors.append(None)
                self._mesh_actors[row] = actor
                
                # Force render to update the display
                self._viewer.render()
                self._viewer.update()
            
            # Update original opacity
            if row < len(self._mesh_original_opacities):
                self._mesh_original_opacities[row] = opacity_value
            
            # Process events to keep UI responsive
            QApplication.processEvents()
        
        # Connect all signals BEFORE starting the worker to avoid "Signal source has been deleted" errors
        worker.apply_mesh.connect(apply_single_mesh)
        
        def on_success(_result: object) -> None:
            # Update all original opacities to current values (in case any were missed)
            for i in range(len(self._mesh_opacity_spinboxes)):
                if i < len(self._mesh_original_opacities):
                    self._mesh_original_opacities[i] = self._mesh_opacity_spinboxes[i].value()
            
            # Disable Apply Changes button
            self.btn_apply_changes.setEnabled(False)
            
            # Final render
            if self._viewer is not None:
                self._viewer.render()
                self._viewer.update()
            
            # Clear worker reference
            if self._opacity_worker == worker:
                self._opacity_worker = None
            
            self.status.show_message(f"Applied opacity changes to {len(modified_rows)} mesh(es)", timeout_ms=2000)
        
        def on_error(error_msg: str) -> None:
            self.status.show_error(f"Failed to apply changes: {error_msg}")
            # Clear worker reference on error
            if self._opacity_worker == worker:
                self._opacity_worker = None
        
        # Connect error signal BEFORE starting the worker
        worker.error.connect(on_error)
        
        # Store reference to prevent garbage collection
        self._opacity_worker = worker
        
        # Now start the worker
        self.status.run_threaded_with_progress(
            worker,
            "Applying opacity changes...",
            on_success=on_success,
        )
    
    def _on_mesh_param_changed(self) -> None:
        """Track changes to mesh generation parameters (don't render immediately)."""
        if self._current_meshes:
            self._check_and_enable_apply_button()
    
    def _on_smooth_shading_toggled(self, checked: bool) -> None:
        """Handle smooth shading toggle (track changes, don't render immediately)."""
        self._on_mesh_param_changed()
    
    def _on_show_edges_toggled(self, checked: bool) -> None:
        """Handle show edges toggle (track changes, don't render immediately)."""
        self._on_mesh_param_changed()
    
    def _on_mesh_visibility_changed(self, item: QTableWidgetItem) -> None:
        """Track visibility checkbox changes (don't render immediately)."""
        # Only handle changes in the Visible column (column 0)
        if item.column() != 0:
            return
        
        # Just track the change, don't render
        if self._current_meshes:
            self._check_and_enable_apply_button()
            
            # OLD CODE BELOW - individual removal doesn't work reliably with PyVista QtInteractor
            if False and row < len(self._mesh_actors) and self._mesh_actors[row] is not None:
                actor = self._mesh_actors[row]
                
                # Try multiple removal methods - be very aggressive
                removed = False
                
                # Method 1: PyVista remove_actor
                try:
                    self._viewer.remove_actor(actor)
                    removed = True
                    pass
                except Exception:
                    pass
                
                # Method 2: Direct renderer removal (always try this)
                try:
                    self._viewer.renderer.RemoveActor(actor)
                    removed = True
                except Exception:
                    pass
                
                # Method 3: Find and remove by iterating through all actors
                try:
                    actors = self._viewer.renderer.GetActors()
                    actors.InitTraversal()
                    while True:
                        current = actors.GetNextActor()
                        if current is None:
                            break
                        if current == actor:
                            try:
                                self._viewer.renderer.RemoveActor(current)
                                removed = True
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # Method 4: Try to get all actors and remove matching ones
                try:
                    actors_collection = self._viewer.renderer.GetActors()
                    actors_list = []
                    actors_collection.InitTraversal()
                    while True:
                        a = actors_collection.GetNextActor()
                        if a is None:
                            break
                        actors_list.append(a)
                    
                    # Remove all matching actors
                    for a in actors_list:
                        if a == actor:
                            try:
                                self._viewer.renderer.RemoveActor(a)
                                removed = True
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # Clear actor reference
                self._mesh_actors[row] = None
                
                # Force render window to update and clear
                try:
                    render_window = self._viewer.renderer.GetRenderWindow()
                    if render_window is not None:
                        render_window.MakeCurrent()
                        render_window.EraseOn()
                        render_window.Render()
                        if hasattr(render_window, 'SwapBuffers'):
                            render_window.SwapBuffers()
                except Exception:
                    pass
        
        # Force multiple render updates using different methods
        # Use QTimer to ensure render happens after event loop processes removal
        from PySide6.QtCore import QTimer
        
        def force_render_updates():
            if self._viewer is not None:
                try:
                    # Method 1: PyVista render
                    self._viewer.render()
                except Exception:
                    pass
                
                try:
                    # Method 2: Renderer render
                    self._viewer.renderer.Render()
                except Exception:
                    pass
                
                try:
                    # Method 3: Render window render
                    render_window = self._viewer.renderer.GetRenderWindow()
                    if render_window is not None:
                        render_window.Render()
                        render_window.SwapBuffers()
                except Exception:
                    pass
                
                try:
                    # Method 4: Qt update
                    self._viewer.update()
                except Exception:
                    pass
                
                try:
                    # Method 5: Qt repaint
                    self._viewer.repaint()
                except Exception:
                    pass
        
        # Immediate render
        force_render_updates()
        
        # Also schedule delayed render to ensure it happens after event loop
        QTimer.singleShot(10, force_render_updates)
        QTimer.singleShot(50, force_render_updates)
    
    def _on_run_organ_mapping_clicked(self) -> None:
        """Handle Run Organ Mapping button click (placeholder)."""
        if self._canonical_volume is None:
            self.status.show_error("Please load a volume first.")
            return
        
        self.status.show_info("Not implemented yet")

