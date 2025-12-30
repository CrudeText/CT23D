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
)

from ct23d.core.volume import CanonicalVolume, load_volume_nrrd
from ct23d.gui.status import StatusController
from ct23d.gui.workers import FunctionWorker, WorkerBase
from PySide6.QtCore import Signal

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
        self._current_mesh: Optional[pv.PolyData] = None  # Current displayed mesh
        self._mesh_actor = None  # Current mesh actor for removal/updates
        self._mesh_worker: Optional[WorkerBase] = None  # Keep reference to mesh generation worker
        
        self._build_ui()
    
    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top row: Load Canonical Volume button (left-aligned, with space for patient info on right)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.btn_load_volume = QPushButton("Load Canonical Volume...")
        self.btn_load_volume.setMaximumWidth(200)
        self.btn_load_volume.clicked.connect(self._on_load_volume_clicked)
        top_row.addWidget(self.btn_load_volume)
        top_row.addStretch()  # Push to left, leaving space on right for patient info
        # Add spacing to ensure load button ends before patient info box (patient info is ~380px wide)
        top_row.addSpacing(400)  # Reserve space for patient info box
        main_layout.addLayout(top_row)
        
        # Add small spacing to start content just below patient info box (135px height + small buffer)
        main_layout.addSpacing(150)
        
        # Main content area: 3D viewer on left, controls on right
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
        
        # Right side: Controls (Volume Info, Mesh Generation, ML Operations)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        # Volume Info panel (read-only)
        info_group = QGroupBox("Volume Info")
        info_layout = QFormLayout(info_group)
        info_layout.setSpacing(8)
        
        self.info_shape_label = QLabel("(No volume loaded)")
        self.info_dtype_label = QLabel("(No volume loaded)")
        self.info_spacing_label = QLabel("(No volume loaded)")
        self.info_intensity_kind_label = QLabel("(No volume loaded)")
        self.info_min_label = QLabel("(No volume loaded)")
        self.info_max_label = QLabel("(No volume loaded)")
        self.info_mean_label = QLabel("(No volume loaded)")
        self.info_path_label = QLabel("(No volume loaded)")
        
        # Make labels selectable for copy-paste
        for label in [
            self.info_shape_label, self.info_dtype_label, self.info_spacing_label,
            self.info_intensity_kind_label, self.info_min_label, self.info_max_label,
            self.info_mean_label, self.info_path_label
        ]:
            label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            label.setStyleSheet("color: white;")
        
        info_layout.addRow("Shape (Z,Y,X):", self.info_shape_label)
        info_layout.addRow("Data type:", self.info_dtype_label)
        info_layout.addRow("Spacing (sx,sy,sz):", self.info_spacing_label)
        info_layout.addRow("Intensity kind:", self.info_intensity_kind_label)
        info_layout.addRow("Min:", self.info_min_label)
        info_layout.addRow("Max:", self.info_max_label)
        info_layout.addRow("Mean:", self.info_mean_label)
        info_layout.addRow("File path:", self.info_path_label)
        
        right_panel.addWidget(info_group)
        
        # Mesh Generation section (placeholders)
        mesh_group = QGroupBox("Mesh Generation")
        mesh_layout = QVBoxLayout(mesh_group)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Intensity threshold", "Label volume (future)"])
        source_layout.addWidget(self.source_combo)
        source_layout.addStretch()
        mesh_layout.addLayout(source_layout)
        
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel("Intensity threshold:")
        threshold_layout.addWidget(threshold_label)
        
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Min:"))
        self.threshold_min_spin = QSpinBox()
        self.threshold_min_spin.setRange(-10000, 10000)
        self.threshold_min_spin.setSingleStep(1)
        self.threshold_min_spin.setValue(0)
        self.threshold_min_spin.setEnabled(False)  # Disabled until volume is loaded
        threshold_row.addWidget(self.threshold_min_spin)
        
        threshold_row.addWidget(QLabel("Max:"))
        self.threshold_max_spin = QSpinBox()
        self.threshold_max_spin.setRange(-10000, 10000)
        self.threshold_max_spin.setSingleStep(1)
        self.threshold_max_spin.setValue(1000)
        self.threshold_max_spin.setEnabled(False)  # Disabled until volume is loaded
        threshold_row.addWidget(self.threshold_max_spin)
        threshold_row.addStretch()
        threshold_layout.addLayout(threshold_row)
        mesh_layout.addLayout(threshold_layout)
        
        lod_layout = QHBoxLayout()
        lod_layout.addWidget(QLabel("LOD:"))
        self.lod_combo = QComboBox()
        self.lod_combo.addItems(["Preview", "Standard", "High"])
        self.lod_combo.setCurrentIndex(1)  # Default to Standard
        lod_layout.addWidget(self.lod_combo)
        lod_layout.addStretch()
        mesh_layout.addLayout(lod_layout)
        
        # Opacity control (slider + spinbox, like slice preview)
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)  # Default to 100 (fully opaque)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.setEnabled(False)  # Enabled when mesh is displayed
        self.opacity_spin.valueChanged.connect(self._on_opacity_spin_changed)
        opacity_layout.addWidget(self.opacity_spin)
        
        # Slider for quick adjustment
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addStretch()
        mesh_layout.addLayout(opacity_layout)
        
        # Smooth shading toggle
        shading_layout = QHBoxLayout()
        self.smooth_shading_cb = QCheckBox("Smooth shading")
        self.smooth_shading_cb.setChecked(True)  # Default to smooth
        self.smooth_shading_cb.setEnabled(False)  # Enabled when mesh is displayed
        self.smooth_shading_cb.toggled.connect(self._on_smooth_shading_toggled)
        shading_layout.addWidget(self.smooth_shading_cb)
        shading_layout.addStretch()
        mesh_layout.addLayout(shading_layout)
        
        # Show edges toggle (optional)
        edges_layout = QHBoxLayout()
        self.show_edges_cb = QCheckBox("Show edges")
        self.show_edges_cb.setChecked(False)  # Default to hidden
        self.show_edges_cb.setEnabled(False)  # Enabled when mesh is displayed
        self.show_edges_cb.toggled.connect(self._on_show_edges_toggled)
        edges_layout.addWidget(self.show_edges_cb)
        edges_layout.addStretch()
        mesh_layout.addLayout(edges_layout)
        
        # Note: Mesh generation happens automatically when volume is loaded
        # No button needed - it generates automatically after stats are computed
        
        right_panel.addWidget(mesh_group)
        
        # ML section (placeholders)
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
        
        right_panel.addWidget(ml_group)
        right_panel.addStretch()  # Push controls to top of right panel
        
        # Add right panel to content layout
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(400)  # Limit width of right panel
        content_layout.addWidget(right_widget, stretch=1)  # Give controls less space (stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)  # Main content area takes most space
        
        # Bottom section: Reserved for exporting etc. (empty for now, will be added later)
        bottom_section = QHBoxLayout()
        # Placeholder for future export controls
        bottom_section.addStretch()
        main_layout.addLayout(bottom_section)
    
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
        
        # Create a combined worker that loads volume and generates mesh
        class LoadAndMeshWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def __init__(self, nrrd_path: Path, threshold_min: int, threshold_max: int):
                super().__init__()
                self.nrrd_path = nrrd_path
                self.threshold_min = threshold_min
                self.threshold_max = threshold_max
                self.volume: Optional[CanonicalVolume] = None
                self.mesh: Optional[pv.PolyData] = None
            
            def run(self) -> None:  # type: ignore[override]
                import sys
                try:
                    # Phase 1: Load NRRD file (overall phase 0 of 4)
                    self.phase_progress.emit("Reading NRRD file", 0, 1, 0)
                    self.volume = load_volume_nrrd(self.nrrd_path)
                    self.phase_progress.emit("Reading NRRD file", 1, 1, 0)
                    
                    # Compute threshold from bins if available, otherwise use default
                    vol = self.volume.data
                    threshold = float(self.threshold_min)  # Default
                    
                    if self.volume.provenance:
                        intensity_bins = self.volume.provenance.get("intensity_bins")
                        if intensity_bins and isinstance(intensity_bins, list) and len(intensity_bins) > 0:
                            lowest_bin = min(intensity_bins, key=lambda b: b.get("low", float('inf')))
                            if "low" in lowest_bin:
                                threshold = float(lowest_bin["low"])
                    
                    # Phase 2: Create mask (overall phase 1 of 4)
                    self.phase_progress.emit("Creating mask", 0, 1, 1)
                    mask = (vol >= threshold).astype(np.float32)
                    self.phase_progress.emit("Creating mask", 1, 1, 1)
                    
                    # Phase 3: Extract mesh (overall phase 2 of 4)
                    self.phase_progress.emit("Extracting mesh", 0, 1, 2)
                    from skimage import measure
                    verts_vox, faces, _normals, _values = measure.marching_cubes(
                        mask,
                        level=0.5,
                        spacing=(1.0, 1.0, 1.0),
                    )
                    
                    if len(faces) == 0:
                        self.mesh = pv.PolyData()
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        self.phase_progress.emit("Scaling mesh", 1, 1, 3)
                    else:
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        
                        # Phase 4: Scale mesh (overall phase 3 of 4)
                        self.phase_progress.emit("Scaling mesh", 0, 1, 3)
                        sx, sy, sz = self.volume.spacing
                        verts_physical = verts_vox.copy()
                        verts_physical[:, 0] *= sz  # Z
                        verts_physical[:, 1] *= sy  # Y
                        verts_physical[:, 2] *= sx  # X
                        
                        verts_pyvista = np.column_stack([
                            verts_physical[:, 2],  # x
                            verts_physical[:, 1],  # y
                            verts_physical[:, 0],  # z
                        ])
                        
                        n_faces = len(faces)
                        faces_vtk = np.empty((n_faces, 4), dtype=np.int32)
                        faces_vtk[:, 0] = 3
                        faces_vtk[:, 1:] = faces
                        faces_vtk = faces_vtk.flatten()
                        
                        # Create mesh
                        self.mesh = pv.PolyData(verts_pyvista, faces_vtk)
                        
                        # Assign vertex colors based on bin intensities
                        if self.volume.provenance:
                            intensity_bins = self.volume.provenance.get("intensity_bins")
                            if intensity_bins and isinstance(intensity_bins, list) and len(intensity_bins) > 0:
                                # Sample intensity values at vertex positions (in voxel coordinates)
                                # verts_vox are in (z, y, x) order, need to convert to integer indices
                                n_verts = verts_vox.shape[0]
                                vertex_intensities = np.zeros(n_verts, dtype=vol.dtype)
                                
                                # Convert vertex positions to integer indices for sampling
                                z_indices = np.clip(verts_vox[:, 0].astype(int), 0, vol.shape[0] - 1)
                                y_indices = np.clip(verts_vox[:, 1].astype(int), 0, vol.shape[1] - 1)
                                x_indices = np.clip(verts_vox[:, 2].astype(int), 0, vol.shape[2] - 1)
                                
                                # Sample intensity values at vertex positions
                                for i in range(n_verts):
                                    vertex_intensities[i] = vol[z_indices[i], y_indices[i], x_indices[i]]
                                
                                # Create color array: one RGB color per vertex
                                vertex_colors = np.zeros((n_verts, 3), dtype=np.uint8)
                                
                                # Map intensities to bin colors
                                for i, intensity in enumerate(vertex_intensities):
                                    # Find which bin this intensity belongs to
                                    color_found = False
                                    for bin_data in intensity_bins:
                                        if isinstance(bin_data, dict):
                                            low = bin_data.get("low")
                                            high = bin_data.get("high")
                                            if low is not None and high is not None and low <= intensity < high:
                                                color = bin_data.get("color")
                                                if color and isinstance(color, (list, tuple)) and len(color) >= 3:
                                                    r, g, b = color[0], color[1], color[2]
                                                    # Ensure values are in 0-1 range
                                                    if r > 1.0 or g > 1.0 or b > 1.0:
                                                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                                                    # Convert to 0-255 range for PyVista
                                                    vertex_colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
                                                    color_found = True
                                                    break
                                    
                                    # Default to white if no bin matches
                                    if not color_found:
                                        vertex_colors[i] = [255, 255, 255]
                                
                                # Assign colors to mesh as point data (vertices)
                                # PyVista expects RGB values in 0-255 range for rgb=True
                                self.mesh['colors'] = vertex_colors
                        
                        self.mesh = self.mesh.compute_normals()
                        self.phase_progress.emit("Scaling mesh", 1, 1, 3)
                    
                    self.finished.emit((self.volume, self.mesh))
                    
                except Exception as e:
                    self.error.emit(str(e))
        
        # Use default threshold of 200 (same as view_nrrd.py)
        # The actual threshold will be computed from bins if available in the worker
        worker = LoadAndMeshWorker(path, 200, 4095)
        
        def on_success(result: tuple) -> None:
            volume, mesh = result
            self._canonical_volume = volume
            self._volume_path = path
            self._update_volume_info_basic()
            
            # Update stats
            data = volume.data
            self.info_min_label.setText(f"{float(data.min()):.2f}")
            self.info_max_label.setText(f"{float(data.max()):.2f}")
            self.info_mean_label.setText(f"{float(data.mean()):.2f}")
            
            # Set threshold ranges and values
            min_val = float(data.min())
            max_val = float(data.max())
            mean_val = float(data.mean())
            min_int = int(np.floor(min_val))
            max_int = int(np.ceil(max_val))
            self.threshold_min_spin.setRange(min_int, max_int)
            self.threshold_max_spin.setRange(min_int, max_int)
            
            # Compute default threshold (same logic as before)
            default_min = None
            if volume.provenance:
                intensity_bins = volume.provenance.get("intensity_bins")
                if intensity_bins and isinstance(intensity_bins, list) and len(intensity_bins) > 0:
                    lowest_bin = min(intensity_bins, key=lambda b: b.get("low", float('inf')))
                    if "low" in lowest_bin:
                        default_min = int(lowest_bin["low"])
            
            if default_min is None:
                default_min = int(round(max(min_val, mean_val * 0.3)))
            default_max = int(round(max_val))
            default_min = max(min_int, min(default_min, max_int - 1))
            default_max = max(default_min + 1, min(default_max, max_int))
            
            self.threshold_min_spin.setValue(default_min)
            self.threshold_max_spin.setValue(default_max)
            self.threshold_min_spin.setEnabled(True)
            self.threshold_max_spin.setEnabled(True)
            
            # Display mesh
            if mesh.n_points > 0:
                self._display_mesh(mesh)
                self.status.show_message(f"Volume and mesh loaded: {path.name}", timeout_ms=3000)
            else:
                self.status.show_error("Mesh generation produced an empty mesh. Try adjusting threshold values.")
        
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
                if self._current_mesh is not None and self._current_mesh.n_points > 0:
                    # Reset camera to fit current mesh
                    self._viewer.reset_camera(bounds=self._current_mesh.bounds)
                else:
                    # No mesh loaded, just reset camera
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
            
            def __init__(self, volume: CanonicalVolume, tmin: float, tmax: float):
                super().__init__()
                self.volume = volume
                self.tmin = tmin
                self.tmax = tmax
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
                    
                    # Run marching cubes (EXACT same as view_nrrd.py)
                    verts_vox, faces, _normals, _values = measure.marching_cubes(
                        mask,
                        level=0.5,
                        spacing=(1.0, 1.0, 1.0),
                    )
                    
                    # Check if we got any faces
                    if len(faces) == 0:
                        # Empty mesh
                        mesh = pv.PolyData()
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        self.phase_progress.emit("Scaling mesh", 1, 1, 3)
                    else:
                        self.phase_progress.emit("Extracting mesh", 1, 1, 2)
                        
                        # Phase 3: Scale vertices by spacing (EXACT same as view_nrrd.py)
                        self.phase_progress.emit("Scaling mesh", 0, 1, 2)
                        sx, sy, sz = self.volume.spacing
                        verts_physical = verts_vox.copy()
                        verts_physical[:, 0] *= sz  # Z
                        verts_physical[:, 1] *= sy  # Y
                        verts_physical[:, 2] *= sx  # X
                        
                        # Convert to PyVista format (x, y, z) (EXACT same as view_nrrd.py)
                        verts_pyvista = np.column_stack([
                            verts_physical[:, 2],  # x
                            verts_physical[:, 1],  # y
                            verts_physical[:, 0],  # z
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
        
        worker = MeshGenerationWorker(self._canonical_volume, tmin, tmax)
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
                    # Display mesh in viewer
                    self._display_mesh(result)
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
    
    def _display_mesh(self, mesh: pv.PolyData) -> None:
        """Display mesh in the 3D viewer."""
        if self._viewer is None:
            return
        
        # Clear previous mesh if any
        if self._mesh_actor is not None:
            self._viewer.remove_actor(self._mesh_actor)
            self._mesh_actor = None
        
        # Store mesh
        self._current_mesh = mesh
        
        # Update controls
        self.opacity_slider.setEnabled(True)
        self.opacity_spin.setEnabled(True)
        self.smooth_shading_cb.setEnabled(True)
        self.show_edges_cb.setEnabled(True)
        
        try:
            # Clear any existing actors first
            self._viewer.clear()
            
            # Add mesh to viewer with vertex colors if available, otherwise use uniform color
            if 'colors' in mesh.point_data:
                # Use vertex colors (RGB per vertex)
                self._mesh_actor = self._viewer.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,  # Interpret scalars as RGB colors
                    smooth_shading=True,
                )
            else:
                # Fallback to uniform color
                self._mesh_actor = self._viewer.add_mesh(
                    mesh,
                    color='white',
                    smooth_shading=True,
                )
            
            # Reset camera and render
            self._viewer.reset_camera()
            self._viewer.render()
            self._viewer.update()
            
        except Exception as e:
            self.status.show_error(f"Error displaying mesh: {e}")
    
    def _update_mesh_display(self) -> None:
        """Update mesh display properties based on current control settings."""
        if self._viewer is None or self._current_mesh is None or self._mesh_actor is None:
            return
        
        # Get current opacity (0-100 -> 0.0-1.0)
        opacity = self.opacity_slider.value() / 100.0
        
        # Get current settings
        smooth_shading = self.smooth_shading_cb.isChecked()
        
        # Remove old mesh actor
        self._viewer.remove_actor(self._mesh_actor)
        
        # Add mesh with updated properties
        # Use vertex colors if available, otherwise use uniform white
        if 'colors' in self._current_mesh.point_data:
            # Use vertex colors (RGB per vertex)
            self._mesh_actor = self._viewer.add_mesh(
                self._current_mesh,
                scalars='colors',
                rgb=True,  # Interpret scalars as RGB colors
                opacity=opacity,
                smooth_shading=smooth_shading,
            )
        else:
            # Fallback to uniform color
            self._mesh_actor = self._viewer.add_mesh(
                self._current_mesh,
                color='white',
                opacity=opacity,
                smooth_shading=smooth_shading,
            )
        
        # Force render and update
        self._viewer.render()
        self._viewer.update()
    
    def _on_opacity_slider_changed(self, value: int) -> None:
        """Handle opacity slider change - sync with spinbox."""
        if self.opacity_spin.value() != value:
            self.opacity_spin.blockSignals(True)
            self.opacity_spin.setValue(value)
            self.opacity_spin.blockSignals(False)
        self._update_mesh_display()
    
    def _on_opacity_spin_changed(self, value: int) -> None:
        """Handle opacity spinbox change - sync with slider."""
        if self.opacity_slider.value() != value:
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(value)
            self.opacity_slider.blockSignals(False)
        self._update_mesh_display()
    
    def _on_smooth_shading_toggled(self, checked: bool) -> None:
        """Handle smooth shading toggle."""
        self._update_mesh_display()
    
    def _on_show_edges_toggled(self, checked: bool) -> None:
        """Handle show edges toggle."""
        self._update_mesh_display()
    
    def _on_run_organ_mapping_clicked(self) -> None:
        """Handle Run Organ Mapping button click (placeholder)."""
        if self._canonical_volume is None:
            self.status.show_error("Please load a volume first.")
            return
        
        self.status.show_info("Not implemented yet")

