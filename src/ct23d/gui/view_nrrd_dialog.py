"""
Dialog for viewing NRRD files with metadata and 3D visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QFormLayout,
)

from ct23d.core.volume import load_volume_nrrd

try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    PVISTA_AVAILABLE = True
except ImportError:
    PVISTA_AVAILABLE = False


class ViewNRRDDialog(QDialog):
    """Dialog for viewing NRRD files with metadata and 3D visualization."""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("View NRRD File")
        self.resize(600, 500)
        
        self._volume: Optional[object] = None
        self._viewer_window: Optional[object] = None
        
        self._build_ui()
    
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Select an NRRD file to view its metadata and 3D visualization."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Select file button
        file_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Select NRRD File...")
        self.select_file_btn.clicked.connect(self._on_select_file)
        file_layout.addWidget(self.select_file_btn)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Metadata group
        metadata_group = QGroupBox("Volume Metadata")
        metadata_layout = QFormLayout(metadata_group)
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(200)
        self.metadata_text.setStyleSheet("font-family: monospace;")
        metadata_layout.addRow(self.metadata_text)
        
        layout.addWidget(metadata_group)
        
        # 3D View button
        self.view_3d_btn = QPushButton("Open 3D Viewer")
        self.view_3d_btn.setEnabled(False)
        self.view_3d_btn.clicked.connect(self._on_view_3d)
        layout.addWidget(self.view_3d_btn)
        
        layout.addStretch()
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
    
    def _on_select_file(self) -> None:
        """Handle file selection."""
        from PySide6.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NRRD File",
            "",
            "NRRD files (*.nrrd);;All files (*.*)"
        )
        
        if not path:
            return
        
        self._load_nrrd(Path(path))
    
    def _load_nrrd(self, path: Path) -> None:
        """Load NRRD file and display metadata."""
        try:
            volume = load_volume_nrrd(path)
            self._volume = volume
            
            # Format metadata
            metadata_lines = [
                f"File: {path}",
                "",
                f"Shape (Z, Y, X): {volume.data.shape}",
                f"Data type: {volume.data.dtype}",
                f"Spacing (sx, sy, sz): {volume.spacing}",
                f"Origin (ox, oy, oz): {volume.origin}",
                f"Intensity kind: {volume.intensity_kind}",
                f"Min value: {volume.data.min()}",
                f"Max value: {volume.data.max()}",
                f"Mean value: {volume.data.mean():.2f}",
                f"Non-zero voxels: {np.count_nonzero(volume.data):,} / {volume.data.size:,}",
            ]
            
            if volume.provenance:
                metadata_lines.append("")
                metadata_lines.append("Provenance:")
                for key, value in volume.provenance.items():
                    if isinstance(value, dict):
                        metadata_lines.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            metadata_lines.append(f"    {subkey}: {subvalue}")
                    else:
                        metadata_lines.append(f"  {key}: {value}")
            
            self.metadata_text.setPlainText("\n".join(metadata_lines))
            self.view_3d_btn.setEnabled(True)
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load NRRD file:\n{str(e)}")
    
    def _on_view_3d(self) -> None:
        """Open 3D viewer for the loaded volume."""
        if self._volume is None:
            return
        
        if not PVISTA_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "PyVista not available",
                "PyVista is required for 3D visualization. Please install it with:\npip install pyvista pyvistaqt"
            )
            return
        
        try:
            # Create mesh using threshold of 200
            data_min = float(self._volume.data.min())
            data_max = float(self._volume.data.max())
            
            threshold = 200.0
            
            from skimage import measure
            
            # Create binary mask using threshold
            mask = (self._volume.data >= threshold).astype(np.float32)
            
            # Run marching cubes
            verts_vox, faces, _normals, _values = measure.marching_cubes(
                mask,
                level=0.5,
                spacing=(1.0, 1.0, 1.0),
            )
            
            if len(faces) == 0:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "No mesh generated",
                    "No mesh could be generated from this volume with the current threshold.\n"
                    "Try adjusting the threshold or check the volume data."
                )
                return
            
            # Scale vertices by spacing
            sx, sy, sz = self._volume.spacing
            verts_physical = verts_vox.copy()
            verts_physical[:, 0] *= sz  # Z
            verts_physical[:, 1] *= sy  # Y
            verts_physical[:, 2] *= sx  # X
            
            # Convert to PyVista format (x, y, z)
            verts_pyvista = np.column_stack([
                verts_physical[:, 2],  # x
                verts_physical[:, 1],  # y
                verts_physical[:, 0],  # z
            ])
            
            # Convert faces to VTK format
            n_faces = len(faces)
            faces_vtk = np.empty((n_faces, 4), dtype=np.int32)
            faces_vtk[:, 0] = 3
            faces_vtk[:, 1:] = faces
            faces_vtk = faces_vtk.flatten()
            
            # Create mesh
            mesh = pv.PolyData(verts_pyvista, faces_vtk)
            mesh = mesh.compute_normals()
            
            # Create plotter window
            plotter = BackgroundPlotter(show=False)
            plotter.add_mesh(mesh, color='white', smooth_shading=True)
            plotter.add_axes()
            plotter.background_color = 'black'
            plotter.reset_camera()
            plotter.show()
            
            self._viewer_window = plotter
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            import traceback
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create 3D visualization:\n{str(e)}\n\n{traceback.format_exc()}"
            )

