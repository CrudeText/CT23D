from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QEvent
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QPolygonF
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QHeaderView,
    QAbstractItemView,
    QProgressDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)
from PySide6.QtCore import QPoint, QRect

from ct23d.core.models import PreprocessConfig, ProjectConfig
from ct23d.core import preprocessing
from ct23d.core import images
from ct23d.gui.status import StatusController  # type: ignore[import]


class AutoDetectSettingsDialog(QDialog):
    """Dialog for configuring auto-detection parameters."""
    
    def __init__(self, parent: Optional[QWidget], params: dict) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Detection Settings")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Aggressivity multiplier
        self.aggressivity_spin = QDoubleSpinBox()
        self.aggressivity_spin.setRange(0.5, 3.0)
        self.aggressivity_spin.setSingleStep(0.1)
        self.aggressivity_spin.setValue(params.get('aggressivity', 1.0))
        self.aggressivity_spin.setToolTip(
            "Controls how aggressively grey pixels are included.\n"
            "Higher values = more aggressive (includes more grey pixels)."
        )
        form_layout.addRow("Aggressivity:", self.aggressivity_spin)
        
        # Grey tolerance
        self.grey_tolerance_spin = QSpinBox()
        self.grey_tolerance_spin.setRange(10, 100)
        self.grey_tolerance_spin.setValue(params.get('grey_tolerance', 30))
        self.grey_tolerance_spin.setToolTip(
            "Tolerance for including grey pixels near detected objects.\n"
            "Higher values = include more grey pixels."
        )
        form_layout.addRow("Grey Tolerance:", self.grey_tolerance_spin)
        
        # Minimum intensity
        self.min_intensity_spin = QSpinBox()
        self.min_intensity_spin.setRange(100, 255)
        self.min_intensity_spin.setValue(params.get('min_intensity', 150))
        self.min_intensity_spin.setToolTip(
            "Minimum intensity for detecting bright objects (bed/headrest).\n"
            "Lower values = detect dimmer objects."
        )
        form_layout.addRow("Min Intensity:", self.min_intensity_spin)
        
        # Bottom region ratio
        self.bottom_region_spin = QDoubleSpinBox()
        self.bottom_region_spin.setRange(0.1, 0.8)
        self.bottom_region_spin.setSingleStep(0.05)
        self.bottom_region_spin.setValue(params.get('bottom_region_ratio', 0.4))
        self.bottom_region_spin.setToolTip(
            "Fraction of image height to scan from bottom.\n"
            "0.4 = scan from bottom 40% of image upward."
        )
        form_layout.addRow("Bottom Region Ratio:", self.bottom_region_spin)
        
        # Min size ratio
        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(0.001, 0.1)
        self.min_size_spin.setSingleStep(0.001)
        self.min_size_spin.setDecimals(3)
        self.min_size_spin.setValue(params.get('min_size_ratio', 0.01))
        self.min_size_spin.setToolTip(
            "Minimum size of objects to detect (as fraction of image).\n"
            "0.01 = 1% of image pixels."
        )
        form_layout.addRow("Min Size Ratio:", self.min_size_spin)
        
        # Max size ratio
        self.max_size_spin = QDoubleSpinBox()
        self.max_size_spin.setRange(0.1, 0.9)
        self.max_size_spin.setSingleStep(0.05)
        self.max_size_spin.setValue(params.get('max_size_ratio', 0.4))
        self.max_size_spin.setToolTip(
            "Maximum size of objects to detect (as fraction of image).\n"
            "0.4 = 40% of image pixels."
        )
        form_layout.addRow("Max Size Ratio:", self.max_size_spin)
        
        # Scan upward checkbox
        self.scan_upward_cb = QCheckBox()
        self.scan_upward_cb.setChecked(params.get('scan_upward', True))
        self.scan_upward_cb.setToolTip(
            "If enabled, scans upward from bottom to find connected bed components."
        )
        form_layout.addRow("Scan Upward:", self.scan_upward_cb)
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.params = params
    
    def get_params(self) -> dict:
        """Return the current parameter values."""
        return {
            'aggressivity': self.aggressivity_spin.value(),
            'grey_tolerance': self.grey_tolerance_spin.value(),
            'min_intensity': self.min_intensity_spin.value(),
            'bottom_region_ratio': self.bottom_region_spin.value(),
            'min_size_ratio': self.min_size_spin.value(),
            'max_size_ratio': self.max_size_spin.value(),
            'scan_upward': self.scan_upward_cb.isChecked(),
        }


class SelectableImageLabel(QLabel):
    """
    Enhanced QLabel that supports multiple selection tools:
    - Box selection (rectangle)
    - Lasso selection (freeform)
    - Click-to-select (object detection)
    - Hover highlighting
    """
    object_selected = Signal(object, object)  # Emits (mask, object_id)
    hover_highlight = Signal(object)  # Emits hover mask for highlighting
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking for hover
        self.drawing = False
        self.tool_mode = "box"  # "box", "lasso", or "click"
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.lasso_points: list[QPoint] = []
        self.original_pixmap: Optional[QPixmap] = None
        self.original_image: Optional[np.ndarray] = None  # Original RGB image
        self.hover_mask: Optional[np.ndarray] = None
        self.hover_point: Optional[QPoint] = None
        
    def set_tool_mode(self, mode: str) -> None:
        """Set the selection tool mode: 'box', 'lasso', or 'click'."""
        self.tool_mode = mode.lower()
        if self.tool_mode == "click":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.tool_mode == "lasso":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:  # box
            self.setCursor(Qt.CursorShape.CrossCursor)
            
    def set_original_image(self, image_rgb: np.ndarray) -> None:
        """Store the original RGB image for object detection."""
        self.original_image = image_rgb
        
    def setPixmap(self, pixmap: QPixmap) -> None:
        """Override to store original pixmap."""
        self.original_pixmap = pixmap
        self._update_display()
        
    def _label_to_image_coords(self, label_point: QPoint) -> Optional[tuple[int, int]]:
        """Convert label coordinates to image coordinates."""
        if self.original_pixmap is None:
            return None
            
        label_size = self.size()
        pixmap_rect = self.original_pixmap.rect()
        displayed_rect = QRect(
            (label_size.width() - pixmap_rect.width()) // 2,
            (label_size.height() - pixmap_rect.height()) // 2,
            pixmap_rect.width(),
            pixmap_rect.height()
        )
        
        if not displayed_rect.contains(label_point):
            return None
            
        # Adjust for centering
        x = label_point.x() - displayed_rect.x()
        y = label_point.y() - displayed_rect.y()
        
        # Scale to original image size if needed
        if self.original_image is not None:
            scale_x = self.original_image.shape[1] / pixmap_rect.width()
            scale_y = self.original_image.shape[0] / pixmap_rect.height()
            x = int(x * scale_x)
            y = int(y * scale_y)
            x = max(0, min(x, self.original_image.shape[1] - 1))
            y = max(0, min(y, self.original_image.shape[0] - 1))
            
        return (x, y)
        
    def _detect_object_at_point(self, x: int, y: int, tolerance: float = 50.0) -> Optional[np.ndarray]:
        """Detect object at clicked point using connected components."""
        if self.original_image is None:
            return None
            
        try:
            from skimage import measure, morphology
            
            # Convert to grayscale
            gray = (self.original_image[..., 0].astype(np.float32) + 
                   self.original_image[..., 1].astype(np.float32) + 
                   self.original_image[..., 2].astype(np.float32)) / 3.0
            
            # Filter out black background (intensity < 10)
            black_mask = gray < 10
            if black_mask[y, x]:
                # Clicked on black background, don't select
                return None
            
            # Get intensity at clicked point and surrounding region
            seed_intensity = gray[y, x]
            
            # Use a larger region around the click point to get better intensity estimate
            y_min, y_max = max(0, y-10), min(gray.shape[0], y+11)
            x_min, x_max = max(0, x-10), min(gray.shape[1], x+11)
            region = gray[y_min:y_max, x_min:x_max]
            # Exclude black pixels from region mean
            region_non_black = region[region >= 10]
            if len(region_non_black) > 0:
                seed_intensity = np.mean(region_non_black)
            else:
                seed_intensity = gray[y, x]
            
            # Create mask of similar intensity pixels (within tolerance)
            # Use larger tolerance for better object detection
            intensity_mask = np.abs(gray - seed_intensity) < tolerance
            # Exclude black background
            intensity_mask = intensity_mask & ~black_mask
            
            # Find connected component containing the seed point
            labels = measure.label(intensity_mask, connectivity=2)
            seed_label = labels[y, x]
            
            if seed_label == 0:
                return None
                
            # Extract the component
            object_mask = (labels == seed_label)
            
            # Expand the mask more aggressively to capture more of the object
            # Use larger dilation for better coverage
            object_mask = morphology.binary_dilation(object_mask, morphology.disk(5))
            # Clean up small holes with larger closing
            object_mask = morphology.binary_closing(object_mask, morphology.disk(3))
            # One more dilation to capture edge pixels
            object_mask = morphology.binary_dilation(object_mask, morphology.disk(2))
            
            return object_mask.astype(bool)
        except Exception:
            return None
            
    def _get_hover_mask(self, x: int, y: int) -> Optional[np.ndarray]:
        """Get mask for object under cursor (for hover highlighting)."""
        # Use same tolerance as click detection
        tolerance = 50.0
        if hasattr(self, 'parent_wizard') and hasattr(self.parent_wizard, 'click_tolerance_spin'):
            tolerance = self.parent_wizard.click_tolerance_spin.value()
        mask = self._detect_object_at_point(x, y, tolerance)
        # For hover, we can use a simpler version without aggressive dilation
        if mask is not None and self.original_image is not None:
            from skimage import morphology
            # Just a light dilation for hover (don't expand too much)
            mask = morphology.binary_dilation(mask, morphology.disk(2))
        return mask
        
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start drawing selection."""
        if self.original_pixmap is None:
            super().mousePressEvent(event)
            return
            
        # Check if selection mode is enabled
        if hasattr(self, 'parent_wizard') and hasattr(self.parent_wizard, 'selection_mode'):
            if not self.parent_wizard.selection_mode:
                super().mousePressEvent(event)
                return
                
            if event.button() == Qt.MouseButton.LeftButton:
                if self.tool_mode == "click":
                    # Click-to-select: detect object immediately
                    coords = self._label_to_image_coords(event.position().toPoint())
                    if coords is not None:
                        # Get tolerance from parent wizard if available
                        tolerance = 50.0
                        if hasattr(self, 'parent_wizard') and hasattr(self.parent_wizard, 'click_tolerance_spin'):
                            tolerance = self.parent_wizard.click_tolerance_spin.value()
                        mask = self._detect_object_at_point(coords[0], coords[1], tolerance)
                        if mask is not None:
                            object_id = str(uuid.uuid4())
                            self.object_selected.emit(mask, object_id)
                else:
                    # Box or lasso: start drawing
                    self.drawing = True
                    self.start_point = event.position().toPoint()
                    self.end_point = self.start_point
                    if self.tool_mode == "lasso":
                        self.lasso_points = [self.start_point]
                    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update selection while dragging or hovering."""
        if self.original_pixmap is None:
            super().mouseMoveEvent(event)
            return
            
        # Check if selection mode is enabled
        if hasattr(self, 'parent_wizard') and hasattr(self.parent_wizard, 'selection_mode'):
            if not self.parent_wizard.selection_mode:
                # Still allow hover highlighting
                if self.tool_mode == "click":
                    coords = self._label_to_image_coords(event.position().toPoint())
                    if coords is not None:
                        self.hover_mask = self._get_hover_mask(coords[0], coords[1])
                        self.hover_point = event.position().toPoint()
                        self._update_display()
                    else:
                        # Mouse is outside image area, clear hover
                        self.hover_mask = None
                        self.hover_point = None
                        self._update_display()
                super().mouseMoveEvent(event)
                return
                
        if self.drawing:
            if self.tool_mode == "lasso":
                self.lasso_points.append(event.position().toPoint())
            self.end_point = event.position().toPoint()
            self._update_display()
        elif self.tool_mode == "click":
            # Hover highlighting for click mode
            coords = self._label_to_image_coords(event.position().toPoint())
            if coords is not None:
                self.hover_mask = self._get_hover_mask(coords[0], coords[1])
                self.hover_point = event.position().toPoint()
                self._update_display()
            else:
                # Mouse is outside image area, clear hover
                self.hover_mask = None
                self.hover_point = None
                self._update_display()
                
    def leaveEvent(self, event: QEvent) -> None:
        """Clear hover highlight when mouse leaves the widget."""
        self.hover_mask = None
        self.hover_point = None
        self._update_display()
        super().leaveEvent(event)
                
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish drawing selection."""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            
            if self.tool_mode == "box":
                rect = QRect(self.start_point, self.end_point).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    mask = self._create_box_mask(rect)
                    if mask is not None:
                        object_id = str(uuid.uuid4())
                        self.object_selected.emit(mask, object_id)
            elif self.tool_mode == "lasso":
                if len(self.lasso_points) > 3:
                    mask = self._create_lasso_mask(self.lasso_points)
                    if mask is not None:
                        object_id = str(uuid.uuid4())
                        self.object_selected.emit(mask, object_id)
                self.lasso_points = []
                
            self._update_display()
            
    def _create_box_mask(self, rect: QRect) -> Optional[np.ndarray]:
        """Create mask from rectangle selection."""
        if self.original_image is None:
            return None
            
        mask = np.zeros((self.original_image.shape[0], self.original_image.shape[1]), dtype=bool)
        
        # Convert rect to image coordinates
        label_size = self.size()
        pixmap_rect = self.original_pixmap.rect()
        displayed_rect = QRect(
            (label_size.width() - pixmap_rect.width()) // 2,
            (label_size.height() - pixmap_rect.height()) // 2,
            pixmap_rect.width(),
            pixmap_rect.height()
        )
        
        if not displayed_rect.contains(rect.center()):
            return None
            
        adjusted_rect = QRect(
            rect.x() - displayed_rect.x(),
            rect.y() - displayed_rect.y(),
            rect.width(),
            rect.height()
        )
        
        scale_x = self.original_image.shape[1] / pixmap_rect.width()
        scale_y = self.original_image.shape[0] / pixmap_rect.height()
        
        x1 = max(0, int(adjusted_rect.left() * scale_x))
        y1 = max(0, int(adjusted_rect.top() * scale_y))
        x2 = min(mask.shape[1], int(adjusted_rect.right() * scale_x))
        y2 = min(mask.shape[0], int(adjusted_rect.bottom() * scale_y))
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
            
            # Filter out black background (intensity < 10)
            gray = (self.original_image[..., 0].astype(np.float32) + 
                   self.original_image[..., 1].astype(np.float32) + 
                   self.original_image[..., 2].astype(np.float32)) / 3.0
            black_mask = gray < 10
            mask = mask & ~black_mask
            
            return mask
        return None
        
    def _create_lasso_mask(self, points: list[QPoint]) -> Optional[np.ndarray]:
        """Create mask from lasso (polygon) selection."""
        if self.original_image is None or len(points) < 3:
            return None
            
        from skimage.draw import polygon
        
        mask = np.zeros((self.original_image.shape[0], self.original_image.shape[1]), dtype=bool)
        
        # Convert points to image coordinates
        label_size = self.size()
        pixmap_rect = self.original_pixmap.rect()
        displayed_rect = QRect(
            (label_size.width() - pixmap_rect.width()) // 2,
            (label_size.height() - pixmap_rect.height()) // 2,
            pixmap_rect.width(),
            pixmap_rect.height()
        )
        
        scale_x = self.original_image.shape[1] / pixmap_rect.width()
        scale_y = self.original_image.shape[0] / pixmap_rect.height()
        
        # Convert polygon points
        rows = []
        cols = []
        for point in points:
            if displayed_rect.contains(point):
                x = int((point.x() - displayed_rect.x()) * scale_x)
                y = int((point.y() - displayed_rect.y()) * scale_y)
                x = max(0, min(x, mask.shape[1] - 1))
                y = max(0, min(y, mask.shape[0] - 1))
                rows.append(y)
                cols.append(x)
                
        if len(rows) < 3:
            return None
            
        # Fill polygon
        try:
            rr, cc = polygon(rows, cols, shape=mask.shape)
            mask[rr, cc] = True
            
            # Filter out black background (intensity < 10)
            gray = (self.original_image[..., 0].astype(np.float32) + 
                   self.original_image[..., 1].astype(np.float32) + 
                   self.original_image[..., 2].astype(np.float32)) / 3.0
            black_mask = gray < 10
            mask = mask & ~black_mask
            
            return mask
        except Exception:
            return None
            
    def _update_display(self) -> None:
        """Update the displayed pixmap with selection overlay and hover highlight."""
        if self.original_pixmap is None:
            return
            
        # Create a copy for drawing
        display_pixmap = self.original_pixmap.copy()
        painter = QPainter(display_pixmap)
        
        # Draw hover highlight (semi-transparent yellow)
        if self.hover_mask is not None and self.original_image is not None:
            # Convert mask to pixmap coordinates for drawing
            pixmap_rect = self.original_pixmap.rect()
            scale_x = pixmap_rect.width() / self.original_image.shape[1]
            scale_y = pixmap_rect.height() / self.original_image.shape[0]
            
            # Create overlay image
            overlay = np.zeros((pixmap_rect.height(), pixmap_rect.width(), 4), dtype=np.uint8)
            from scipy.ndimage import zoom
            hover_scaled = zoom(self.hover_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
            overlay[hover_scaled, 0] = 255  # Yellow
            overlay[hover_scaled, 1] = 255
            overlay[hover_scaled, 2] = 0
            overlay[hover_scaled, 3] = 100  # Semi-transparent
            
            overlay_image = QImage(overlay.data, overlay.shape[1], overlay.shape[0], 
                                  overlay.shape[1] * 4, QImage.Format.Format_RGBA8888)
            overlay_pixmap = QPixmap.fromImage(overlay_image)
            painter.drawPixmap(0, 0, overlay_pixmap)
        
        # Draw existing selections (red)
        painter.setPen(QPen(QColor(255, 0, 0, 255), 2))
        if hasattr(self, 'parent_wizard') and hasattr(self.parent_wizard, 'selected_objects'):
            for obj in self.parent_wizard.selected_objects:
                if 'mask' in obj and obj['mask'] is not None:
                    # Draw mask outline (simplified - just draw bounding box)
                    mask = obj['mask']
                    if mask.shape == self.original_image.shape[:2]:
                        coords = np.where(mask)
                        if len(coords[0]) > 0:
                            pixmap_rect = self.original_pixmap.rect()
                            scale_x = pixmap_rect.width() / mask.shape[1]
                            scale_y = pixmap_rect.height() / mask.shape[0]
                            
                            y_min, y_max = int(coords[0].min() * scale_y), int(coords[0].max() * scale_y)
                            x_min, x_max = int(coords[1].min() * scale_x), int(coords[1].max() * scale_x)
                            painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Draw current selection being drawn
        if self.drawing:
            painter.setPen(QPen(QColor(255, 0, 0, 180), 2))
            if self.tool_mode == "box":
                current_rect = QRect(self.start_point, self.end_point).normalized()
                label_size = self.size()
                pixmap_rect = self.original_pixmap.rect()
                displayed_rect = QRect(
                    (label_size.width() - pixmap_rect.width()) // 2,
                    (label_size.height() - pixmap_rect.height()) // 2,
                    pixmap_rect.width(),
                    pixmap_rect.height()
                )
                if displayed_rect.contains(current_rect.center()):
                    adjusted_rect = QRect(
                        current_rect.x() - displayed_rect.x(),
                        current_rect.y() - displayed_rect.y(),
                        current_rect.width(),
                        current_rect.height()
                    )
                    painter.drawRect(adjusted_rect)
            elif self.tool_mode == "lasso" and len(self.lasso_points) > 1:
                label_size = self.size()
                pixmap_rect = self.original_pixmap.rect()
                displayed_rect = QRect(
                    (label_size.width() - pixmap_rect.width()) // 2,
                    (label_size.height() - pixmap_rect.height()) // 2,
                    pixmap_rect.width(),
                    pixmap_rect.height()
                )
                polygon = QPolygonF()
                for point in self.lasso_points:
                    if displayed_rect.contains(point):
                        adjusted_point = QPointF(
                            point.x() - displayed_rect.x(),
                            point.y() - displayed_rect.y()
                        )
                        polygon.append(adjusted_point)
                if polygon.size() > 1:
                    painter.drawPolyline(polygon)
                
        painter.end()
        super().setPixmap(display_pixmap)
        
    def clear_selection(self) -> None:
        """Clear all selections."""
        self.lasso_points = []
        self.hover_mask = None
        self.hover_point = None
        if self.original_pixmap is not None:
            super().setPixmap(self.original_pixmap)


class PreprocessWizard(QWidget):
    """
    Simple functional preprocessing tool.

    Allows you to:
      - select an input directory with raw CT slices
      - choose the processed output directory
      - configure overlay / bed removal parameters
      - run the preprocessing pipeline
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        status: Optional[StatusController] = None,
    ) -> None:
        super().__init__(parent)
        self._last_output_dir: Optional[Path] = None

        self.status = status
        self._preproc_worker = None  # keep reference to avoid GC

        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.image_paths: list[Path] = []
        self.current_image_index: int = 0
        self.object_mask: Optional[np.ndarray] = None  # Combined mask for all selected objects
        self.selection_mode: bool = False  # Whether user is in selection mode
        self.tool_mode: str = "box"  # "box", "lasso", or "click"
        self.selected_objects: list[dict] = []  # List of {mask, label, id, slice_index} for each selected object
        self.image_rotation: int = 0  # Rotation angle in degrees (0, 90, 180, 270)
        self.mask_selected_slice_index: Optional[int] = None  # Slice where mask was originally selected
        
        # Auto-detection parameters
        self.auto_detect_params = {
            'bottom_region_ratio': 0.4,
            'min_intensity': 150,
            'min_size_ratio': 0.01,
            'max_size_ratio': 0.4,
            'scan_upward': True,
            'grey_tolerance': 30,
            'aggressivity': 1.0,  # Multiplier for dilation/closing operations
        }

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ------------------------------------------------------------------
        # Input directory selector
        # ------------------------------------------------------------------
        row1 = QHBoxLayout()
        self.input_label = QLabel("Input directory: (none)")
        choose_input_btn = QPushButton("Select Input Folder")
        choose_input_btn.clicked.connect(self.select_input_dir)
        row1.addWidget(self.input_label)
        row1.addWidget(choose_input_btn)
        layout.addLayout(row1)

        # ------------------------------------------------------------------
        # Processed output directory selector
        # ------------------------------------------------------------------
        row2 = QHBoxLayout()
        self.output_label = QLabel("Processed directory: (auto)")
        choose_output_btn = QPushButton("Select Output Folder")
        choose_output_btn.clicked.connect(self.select_output_dir)
        row2.addWidget(self.output_label)
        row2.addWidget(choose_output_btn)
        layout.addLayout(row2)

        # Parameters will be moved to the center column between images

        # ------------------------------------------------------------------
        # Image Preview Section
        # ------------------------------------------------------------------
        try:
            preview_group = QGroupBox("Image Preview")
            preview_group.setVisible(True)  # Ensure it's visible
            preview_group.setMinimumHeight(500)  # Ensure it has minimum height
            preview_group_layout = QVBoxLayout(preview_group)
            
            # Image selector
            selector_layout = QHBoxLayout()
            selector_layout.addWidget(QLabel("Preview slice:"))
            self.image_selector = QSpinBox()
            self.image_selector.setRange(1, 1)
            self.image_selector.setValue(1)
            self.image_selector.setEnabled(False)
            self.image_selector.valueChanged.connect(self.on_image_selected)
            selector_layout.addWidget(self.image_selector)
            self.image_count_label = QLabel("of 0")
            selector_layout.addWidget(self.image_count_label)
            selector_layout.addStretch()
            preview_group_layout.addLayout(selector_layout)

            # Preview images with controls in the middle
            preview_images_layout = QHBoxLayout()
            
            # Before image (left) - make it interactive for drawing
            before_layout = QVBoxLayout()
            before_layout.addWidget(QLabel("Before preprocessing:"))
            self.before_label = SelectableImageLabel(self)
            self.before_label.setMinimumSize(400, 400)
            self.before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.before_label.setStyleSheet("border: 1px solid gray; background-color: black;")
            self.before_label.setText("No image loaded")
            self.before_label.setScaledContents(False)  # We'll scale manually
            self.before_label.object_selected.connect(self.on_object_selected)
            self.before_label.parent_wizard = self
            before_layout.addWidget(self.before_label, stretch=1)
            preview_images_layout.addLayout(before_layout, stretch=1)  # Equal stretch for images
            
            # Controls column (middle)
            controls_column = QVBoxLayout()
            controls_column.setSpacing(10)
            
            # Preprocessing parameters
            params_group = QGroupBox("Preprocessing Parameters")
            params_layout = QVBoxLayout(params_group)
            
            # grayscale tolerance
            hl1 = QHBoxLayout()
            hl1.addWidget(QLabel("Grayscale tolerance:"))
            self.gray_spin = QSpinBox()
            self.gray_spin.setRange(0, 50)
            self.gray_spin.setValue(1)
            self.gray_spin.valueChanged.connect(self.update_preview)
            hl1.addWidget(self.gray_spin)
            params_layout.addLayout(hl1)

            # saturation threshold
            hl2 = QHBoxLayout()
            hl2.addWidget(QLabel("Saturation threshold:"))
            self.sat_spin = QDoubleSpinBox()
            self.sat_spin.setRange(0.0, 1.0)
            self.sat_spin.setSingleStep(0.01)
            self.sat_spin.setValue(0.08)
            self.sat_spin.valueChanged.connect(self.update_preview)
            hl2.addWidget(self.sat_spin)
            params_layout.addLayout(hl2)

            # remove non-grayscale checkbox
            self.remove_non_grayscale_cb = QCheckBox("Remove non-grayscale pixels (turn black)")
            self.remove_non_grayscale_cb.setChecked(False)
            self.remove_non_grayscale_cb.toggled.connect(self.update_preview)
            params_layout.addWidget(self.remove_non_grayscale_cb)
            
            controls_column.addWidget(params_group)
            
            # Object selection controls
            selection_group = QGroupBox("Object Selection")
            selection_layout = QVBoxLayout(selection_group)
            
            # Tool mode selection
            tool_mode_layout = QHBoxLayout()
            tool_mode_layout.addWidget(QLabel("Tool:"))
            self.tool_mode_combo = QComboBox()
            self.tool_mode_combo.addItems(["Box", "Lasso", "Click to Select"])
            self.tool_mode_combo.setCurrentText("Box")
            self.tool_mode_combo.currentTextChanged.connect(self.on_tool_mode_changed)
            tool_mode_layout.addWidget(self.tool_mode_combo)
            selection_layout.addLayout(tool_mode_layout)
            
            # Click tolerance (only shown for click mode)
            click_tolerance_layout = QHBoxLayout()
            click_tolerance_layout.addWidget(QLabel("Click tolerance:"))
            self.click_tolerance_spin = QDoubleSpinBox()
            self.click_tolerance_spin.setRange(10.0, 200.0)
            self.click_tolerance_spin.setValue(50.0)
            self.click_tolerance_spin.setSingleStep(10.0)
            self.click_tolerance_spin.setSuffix(" (higher = larger selection)")
            self.click_tolerance_spin.setEnabled(False)  # Only enabled in click mode
            self.click_tolerance_spin.valueChanged.connect(self.update_preview)
            click_tolerance_layout.addWidget(self.click_tolerance_spin)
            selection_layout.addLayout(click_tolerance_layout)
            
            self.select_objects_btn = QPushButton("Start Selection Mode")
            self.select_objects_btn.setEnabled(False)
            self.select_objects_btn.clicked.connect(self.toggle_selection_mode)
            selection_layout.addWidget(self.select_objects_btn)
            
            self.clear_selection_btn = QPushButton("Clear All Selections")
            self.clear_selection_btn.setEnabled(False)
            self.clear_selection_btn.clicked.connect(self.clear_object_selection)
            selection_layout.addWidget(self.clear_selection_btn)
            
            # Auto-detect bed/headrest button with settings
            auto_detect_layout = QHBoxLayout()
            self.auto_detect_btn = QPushButton("Auto-Detect Bed/Headrest")
            self.auto_detect_btn.setEnabled(False)
            self.auto_detect_btn.clicked.connect(self.auto_detect_bed_headrest)
            self.auto_detect_btn.setToolTip(
                "⚠️ IMPORTANT: Auto-detection scans from BOTTOM UPWARD.\n"
                "The bed/headrest MUST be located UNDERNEATH the body/head.\n"
                "It will find multiple bed components and include grey pixels.\n"
                "If the bed is on the side or top, use the rotation buttons first."
            )
            auto_detect_layout.addWidget(self.auto_detect_btn)
            
            # Settings icon button
            self.auto_detect_settings_btn = QPushButton("⚙")
            self.auto_detect_settings_btn.setEnabled(False)
            self.auto_detect_settings_btn.setFixedWidth(30)
            self.auto_detect_settings_btn.setToolTip("Configure auto-detection parameters")
            self.auto_detect_settings_btn.clicked.connect(self.show_auto_detect_settings)
            auto_detect_layout.addWidget(self.auto_detect_settings_btn)
            
            selection_layout.addLayout(auto_detect_layout)
            
            # Rotation controls
            rotation_group = QGroupBox("Image Rotation")
            rotation_layout = QVBoxLayout(rotation_group)
            rotation_info = QLabel(
                "⚠️ Auto-detect scans from BOTTOM UPWARD to find bed/headrest.\n"
                "The bed/headrest must be UNDERNEATH the body/head.\n"
                "If needed, rotate images so the bed is at the bottom."
            )
            rotation_info.setWordWrap(True)
            rotation_info.setStyleSheet("color: orange; font-weight: bold;")
            rotation_layout.addWidget(rotation_info)
            
            rotation_buttons_layout = QHBoxLayout()
            self.rotate_90_cw_btn = QPushButton("Rotate 90° CW")
            self.rotate_90_cw_btn.setEnabled(False)
            self.rotate_90_cw_btn.clicked.connect(lambda: self.rotate_images(90))
            rotation_buttons_layout.addWidget(self.rotate_90_cw_btn)
            
            self.rotate_90_ccw_btn = QPushButton("Rotate 90° CCW")
            self.rotate_90_ccw_btn.setEnabled(False)
            self.rotate_90_ccw_btn.clicked.connect(lambda: self.rotate_images(-90))
            rotation_buttons_layout.addWidget(self.rotate_90_ccw_btn)
            
            self.rotate_180_btn = QPushButton("Rotate 180°")
            self.rotate_180_btn.setEnabled(False)
            self.rotate_180_btn.clicked.connect(lambda: self.rotate_images(180))
            rotation_buttons_layout.addWidget(self.rotate_180_btn)
            
            rotation_layout.addLayout(rotation_buttons_layout)
            
            rotation_status_layout = QHBoxLayout()
            rotation_status_layout.addWidget(QLabel("Current rotation:"))
            self.rotation_label = QLabel("0°")
            self.rotation_label.setStyleSheet("font-weight: bold;")
            rotation_status_layout.addWidget(self.rotation_label)
            rotation_status_layout.addStretch()
            rotation_layout.addLayout(rotation_status_layout)
            
            selection_layout.addWidget(rotation_group)
            
            # Selected objects table
            selection_layout.addWidget(QLabel("Selected Objects:"))
            self.objects_table = QTableWidget(0, 2, self)
            self.objects_table.setHorizontalHeaderLabels(["Label (optional)", "Actions"])
            self.objects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.objects_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.objects_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.objects_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            selection_layout.addWidget(self.objects_table)
            
            controls_column.addWidget(selection_group)
            
            controls_column.addStretch()  # Push controls to top
            
            preview_images_layout.addLayout(controls_column, stretch=0)  # Fixed width for controls
            
            # After image (right)
            after_layout = QVBoxLayout()
            after_layout.addWidget(QLabel("After preprocessing:"))
            self.after_label = QLabel()
            self.after_label.setMinimumSize(400, 400)
            self.after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.after_label.setStyleSheet("border: 1px solid gray; background-color: black;")
            self.after_label.setText("No image loaded")
            self.after_label.setScaledContents(False)  # We'll scale manually
            after_layout.addWidget(self.after_label, stretch=1)
            preview_images_layout.addLayout(after_layout, stretch=1)  # Equal stretch for images
            
            preview_group_layout.addLayout(preview_images_layout)
            layout.addWidget(preview_group)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Add a simple label to show something went wrong
            error_label = QLabel(f"Preview section error: {e}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)

        # ------------------------------------------------------------------
        # Run button
        # ------------------------------------------------------------------
        run_btn = QPushButton("Run preprocessing")
        run_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(run_btn)

        # Don't add stretch - let preview section be visible

    # ----------------------------------------------------------------------
    # Directory selection slots
    # ----------------------------------------------------------------------
    def select_input_dir(self) -> None:
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setWindowTitle("Select input CT slice directory")

        if dlg.exec():
            dir_path = Path(dlg.selectedFiles()[0])
            self.input_dir = dir_path
            self.input_label.setText(f"Input directory: {dir_path}")

            # Auto-suggest processed_slices under the input dir
            auto_out = dir_path / "processed_slices"
            self.output_dir = auto_out
            self.output_label.setText(f"Processed directory: {auto_out}")

            # Load image paths and initialize preview
            self.load_image_paths()

    def select_output_dir(self) -> None:
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setWindowTitle("Select processed output directory")

        if dlg.exec():
            dir_path = Path(dlg.selectedFiles()[0])
            self.output_dir = dir_path
            self.output_label.setText(f"Processed directory: {dir_path}")

    # ----------------------------------------------------------------------
    # Run preprocessing
    # ----------------------------------------------------------------------
    def run_preprocessing(self) -> None:
        """
        Trigger the preprocessing pipeline.

        If a StatusController is available, this runs in a background QThread
        (PreprocessWorker) with a determinate progress bar (slices processed / total).

        If no StatusController is provided (e.g. in tests), it falls back
        to a simple synchronous call.
        """
        if self.input_dir is None:
            QMessageBox.warning(
                self,
                "Missing input directory",
                "Please select an input directory containing CT slices.",
            )
            return

        # Combine all selected object masks
        combined_mask = None
        if self.selected_objects:
            combined_mask = np.zeros(
                (self.selected_objects[0]['mask'].shape[0], 
                 self.selected_objects[0]['mask'].shape[1]),
                dtype=bool
            )
            for obj in self.selected_objects:
                if obj['mask'] is not None:
                    combined_mask = combined_mask | obj['mask']

        # Get the slice index where mask was selected (use the tracked slice index)
        mask_slice_index = self.mask_selected_slice_index if combined_mask is not None else None
        
        # Disable cache if we have object masks or rotation, since those change the output
        use_cache = (combined_mask is None and self.image_rotation == 0)
        
        cfg = PreprocessConfig(
            input_dir=self.input_dir,
            processed_dir=self.output_dir,
            use_cache=use_cache,
            grayscale_tolerance=self.gray_spin.value(),
            saturation_threshold=self.sat_spin.value(),
            remove_bed=False,  # Removed - use object selection instead
            remove_non_grayscale=self.remove_non_grayscale_cb.isChecked(),
            object_mask=combined_mask,
            object_mask_slice_index=mask_slice_index,
            rotation=self.image_rotation,  # Apply rotation during preprocessing
        )

        project = ProjectConfig(
            name="GUI Preprocessing",
            preprocess=cfg,
        )

        # ------------------------------------------------------------------
        # Fallback path: no StatusController -> synchronous
        # ------------------------------------------------------------------
        if self.status is None:
            try:
                out_dir = preprocessing.preprocess_slices(project.preprocess)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(
                    self,
                    "Error during preprocessing",
                    f"An error occurred while preprocessing slices:\n\n{e}",
                )
                return

            QMessageBox.information(
                self,
                "Preprocessing complete",
                f"Processed slices saved to:\n{out_dir}",
            )
            return

        # ------------------------------------------------------------------
        # Threaded path: use PreprocessWorker + run_threaded_with_progress
        # ------------------------------------------------------------------
        from ct23d.gui.workers import PreprocessWorker  # local import to avoid cycles

        worker = PreprocessWorker(project.preprocess)
        self._preproc_worker = worker  # keep reference

        def on_finished(out_dir: Path) -> None:
            QMessageBox.information(
                self,
                "Preprocessing complete",
                f"Processed slices saved to:\n{out_dir}",
            )
            self.status.show_message("Preprocessing complete")
            # Store output directory for meshing wizard to access
            self._last_output_dir = out_dir
            
            # Try to notify meshing wizard if parent is MainWindow
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'meshing_tab'):
                    # Found MainWindow - update meshing wizard
                    parent.meshing_tab.wizard.set_default_processed_dir(out_dir)
                    break
                parent = parent.parent()

        def on_error(message: str) -> None:
            # message is a string, not an Exception
            if "cancelled" in message.lower() or "cancel" in message.lower():
                self.status.show_message("Preprocessing was cancelled")
            else:
                QMessageBox.critical(
                    self,
                    "Error during preprocessing",
                    f"An error occurred while preprocessing slices:\n\n{message}",
                )
                self.status.show_message("Preprocessing failed")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        self.status.run_threaded_with_progress(
            worker,
            title="Preprocessing",
            parent=self,
            cancellable=True,
        )

    # ----------------------------------------------------------------------
    # Preview functionality
    # ----------------------------------------------------------------------
    def load_image_paths(self) -> None:
        """Load list of image files from input directory."""
        if self.input_dir is None:
            return

        try:
            self.image_paths = preprocessing._list_image_files(self.input_dir)
            num_images = len(self.image_paths)
            
            if num_images > 0:
                # Set up image selector
                self.image_selector.setRange(1, num_images)
                self.image_selector.setEnabled(True)
                self.image_count_label.setText(f"of {num_images}")
                
                # Enable object selection buttons
                self.select_objects_btn.setEnabled(True)
                self.auto_detect_btn.setEnabled(True)
                self.auto_detect_settings_btn.setEnabled(True)
                self.auto_detect_settings_btn.setEnabled(True)
                self.rotate_90_cw_btn.setEnabled(True)
                self.rotate_90_ccw_btn.setEnabled(True)
                self.rotate_180_btn.setEnabled(True)
                if len(self.selected_objects) > 0:
                    self.clear_selection_btn.setEnabled(True)
                
                # Start with middle image (1-indexed)
                middle_idx = (num_images // 2) + 1
                self.current_image_index = middle_idx - 1  # Convert to 0-indexed
                # Block signals temporarily to avoid double update
                self.image_selector.blockSignals(True)
                self.image_selector.setValue(middle_idx)
                self.image_selector.blockSignals(False)
                
                # Update preview immediately
                self.update_preview()
                # Schedule another update after a short delay to ensure UI is fully laid out
                QTimer.singleShot(50, self.update_preview)
                QTimer.singleShot(200, self.update_preview)  # Second update after layout settles
            else:
                self.image_selector.setEnabled(False)
                self.image_count_label.setText("of 0")
                self.before_label.clear()
                self.before_label.setText("No images found")
                self.after_label.clear()
                self.after_label.setText("No images found")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error loading images",
                f"Could not load images from directory:\n\n{e}",
            )
            self.image_paths = []
            self.image_selector.setEnabled(False)
            self.image_count_label.setText("of 0")
            self.before_label.setText(f"Error: {str(e)}")
            self.after_label.setText(f"Error: {str(e)}")

    def on_image_selected(self, value: int) -> None:
        """Handle when user changes the image selector."""
        self.current_image_index = value - 1  # Convert to 0-indexed
        self.update_preview()

    def numpy_to_qpixmap(self, arr: np.ndarray) -> QPixmap:
        """Convert numpy RGB array to QPixmap."""
        height, width, channels = arr.shape
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # Ensure array is contiguous and uint8
        arr_copy = np.ascontiguousarray(arr, dtype=np.uint8)
        
        # Convert to QImage format (RGB888)
        # QImage constructor takes data pointer, so we need to ensure data stays alive
        # until QPixmap is created (which copies the data)
        qimage = QImage(
            arr_copy.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        # QPixmap.fromImage copies the data, so arr_copy can be garbage collected
        return QPixmap.fromImage(qimage)

    def update_preview(self) -> None:
        """Update the before/after preview images."""
        if not self.image_paths:
            self.before_label.clear()
            self.before_label.setText("No images loaded")
            self.after_label.clear()
            self.after_label.setText("No images loaded")
            return

        if self.current_image_index < 0 or self.current_image_index >= len(self.image_paths):
            return

        try:
            # Load original image WITHOUT rotation (rotation only applied to "after" preview)
            image_path = self.image_paths[self.current_image_index]
            original_rgb = images.load_image_rgb(image_path, rotation=0)
            
            # Store original image for object detection (without rotation)
            self.before_label.set_original_image(original_rgb)
            
            # Display original
            original_pixmap = self.numpy_to_qpixmap(original_rgb)
            # Scale to fit label while maintaining aspect ratio
            # Use actual size if available, otherwise use minimum size
            label_size = self.before_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                label_size = self.before_label.minimumSize()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_original = original_pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                # Fallback: use a reasonable default size
                scaled_original = original_pixmap.scaled(
                    400, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            # Use setPixmap which will handle the SelectableImageLabel properly
            if isinstance(self.before_label, SelectableImageLabel):
                self.before_label.setPixmap(scaled_original)
            else:
                self.before_label.clear()  # Clear any text
                self.before_label.setPixmap(scaled_original)
            self.before_label.update()  # Force immediate repaint

            # Process image with current settings
            # Combine all selected object masks for this preview
            combined_mask = None
            if self.selected_objects:
                # Use mask from first object as base (they should all be same size)
                if self.selected_objects[0]['mask'] is not None:
                    combined_mask = np.zeros_like(self.selected_objects[0]['mask'], dtype=bool)
                    for obj in self.selected_objects:
                        if obj['mask'] is not None:
                            combined_mask = combined_mask | obj['mask']
            
            # For preview: only apply mask if this is the slice where it was selected
            # Propagation across slices will be computed during full preprocessing
            apply_mask_in_preview = False
            if combined_mask is not None and self.mask_selected_slice_index is not None:
                # Only apply mask if viewing the slice where it was selected
                if self.current_image_index == self.mask_selected_slice_index:
                    apply_mask_in_preview = True
            
            # Apply rotation to original for processing preview (rotation only affects "after" preview)
            rotated_rgb = images.rotate_image_rgb(original_rgb, self.image_rotation)
            
            # Process single slice (no propagation in preview for performance)
            processed_rgb = images.preprocess_volume_rgb(
                np.expand_dims(rotated_rgb, axis=0),  # Add Z dimension
                grayscale_tolerance=self.gray_spin.value(),
                saturation_threshold=self.sat_spin.value(),
                remove_bed=False,  # Removed - use object selection instead
                remove_non_grayscale=self.remove_non_grayscale_cb.isChecked(),
                object_mask=combined_mask if apply_mask_in_preview else None,
                object_mask_slice_index=0 if apply_mask_in_preview else None,
            )
            processed_rgb = processed_rgb[0]  # Remove Z dimension

            # Display processed
            processed_pixmap = self.numpy_to_qpixmap(processed_rgb)
            
            # Add note about propagation if mask exists but not on selected slice
            if combined_mask is not None and self.mask_selected_slice_index is not None:
                if self.current_image_index != self.mask_selected_slice_index:
                    # Create overlay text on the pixmap
                    from PySide6.QtGui import QPainter, QFont
                    painter = QPainter(processed_pixmap)
                    painter.setPen(QColor(255, 255, 0))  # Yellow text
                    font = QFont()
                    font.setPointSize(12)
                    font.setBold(True)
                    painter.setFont(font)
                    painter.drawText(
                        10, 30,
                        "Note: Mask propagation will be applied during\nfull preprocessing"
                    )
                    painter.end()
            # Scale to fit label while maintaining aspect ratio
            # Use actual size if available, otherwise use minimum size
            label_size = self.after_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                label_size = self.after_label.minimumSize()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_processed = processed_pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                # Fallback: use a reasonable default size
                scaled_processed = processed_pixmap.scaled(
                    400, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.after_label.clear()  # Clear any text
            self.after_label.setPixmap(scaled_processed)
            self.after_label.update()  # Force immediate repaint

        except Exception as e:
            import traceback
            error_msg = f"Could not update preview:\n\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.warning(
                self,
                "Preview error",
                error_msg,
            )
            self.before_label.setText(f"Error: {str(e)}")
            self.after_label.setText(f"Error: {str(e)}")

    def on_tool_mode_changed(self, mode: str) -> None:
        """Handle tool mode change."""
        mode_lower = mode.lower()
        if "box" in mode_lower:
            self.tool_mode = "box"
            self.click_tolerance_spin.setEnabled(False)
        elif "lasso" in mode_lower:
            self.tool_mode = "lasso"
            self.click_tolerance_spin.setEnabled(False)
        elif "click" in mode_lower:
            self.tool_mode = "click"
            self.click_tolerance_spin.setEnabled(True)
        else:
            self.tool_mode = "box"
            self.click_tolerance_spin.setEnabled(False)
            
        if hasattr(self, 'before_label') and isinstance(self.before_label, SelectableImageLabel):
            self.before_label.set_tool_mode(self.tool_mode)

    def toggle_selection_mode(self) -> None:
        """Toggle object selection mode for drawing masks."""
        if self.selection_mode:
            # Disable selection mode
            self.selection_mode = False
            self.select_objects_btn.setText("Start Selection Mode")
            self.before_label.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Enable selection mode
            if not self.image_paths:
                QMessageBox.warning(
                    self,
                    "No Images",
                    "Please select an input folder with images first.",
                )
                return
            self.selection_mode = True
            self.select_objects_btn.setText("Stop Selection Mode")
            if hasattr(self, 'before_label') and isinstance(self.before_label, SelectableImageLabel):
                self.before_label.set_tool_mode(self.tool_mode)

    def on_object_selected(self, mask: np.ndarray, object_id: str) -> None:
        """Handle when an object is selected."""
        # Add to selected objects list with default label
        obj = {
            'id': object_id,
            'mask': mask,
            'label': 'Other',  # Default label
            'slice_index': self.current_image_index,  # Track which slice this was selected on
        }
        self.selected_objects.append(obj)
        
        # Track the slice where mask was selected (use first selected object's slice)
        if self.mask_selected_slice_index is None:
            self.mask_selected_slice_index = self.current_image_index
        
        # Update table
        self._update_objects_table()
        
        # Update combined mask and preview
        self._update_combined_mask()
        self.update_preview()
        
        # Enable clear button
        self.clear_selection_btn.setEnabled(True)
        
    def _update_objects_table(self) -> None:
        """Update the selected objects table."""
        self.objects_table.setRowCount(len(self.selected_objects))
        
        label_options = ["Bed", "Headrest", "Writing", "Logo", "Other"]
        
        for row, obj in enumerate(self.selected_objects):
            # Label combo box
            label_combo = QComboBox()
            label_combo.addItems(label_options)
            label_combo.setCurrentText(obj['label'])
            label_combo.currentTextChanged.connect(
                lambda text, obj_id=obj['id']: self._update_object_label(obj_id, text)
            )
            self.objects_table.setCellWidget(row, 0, label_combo)
            
            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(
                lambda checked, obj_id=obj['id']: self._delete_object(obj_id)
            )
            self.objects_table.setCellWidget(row, 1, delete_btn)
            
    def _update_object_label(self, object_id: str, label: str) -> None:
        """Update the label for a selected object."""
        for obj in self.selected_objects:
            if obj['id'] == object_id:
                obj['label'] = label
                break
                
    def _delete_object(self, object_id: str) -> None:
        """Delete a selected object."""
        self.selected_objects = [obj for obj in self.selected_objects if obj['id'] != object_id]
        self._update_objects_table()
        self._update_combined_mask()
        self.update_preview()
        
        if len(self.selected_objects) == 0:
            self.clear_selection_btn.setEnabled(False)
            
    def _update_combined_mask(self) -> None:
        """Update the combined mask from all selected objects."""
        if not self.selected_objects:
            self.object_mask = None
            return
            
        # Combine all masks
        combined = np.zeros_like(self.selected_objects[0]['mask'], dtype=bool)
        for obj in self.selected_objects:
            if obj['mask'] is not None:
                combined = combined | obj['mask']
        self.object_mask = combined

    def clear_object_selection(self) -> None:
        """Clear all selected objects."""
        self.selected_objects.clear()
        self.object_mask = None
        self.mask_selected_slice_index = None
        self.before_label.clear_selection()
        self._update_objects_table()
        self.clear_selection_btn.setEnabled(False)
        if self.selection_mode:
            self.toggle_selection_mode()  # Turn off selection mode
        self.update_preview()
    
    def auto_detect_bed_headrest(self) -> None:
        """Automatically detect and select bed/headrest in current slice."""
        if not self.image_paths or self.current_image_index < 0:
            return
        
        # Show loading dialog
        progress = QProgressDialog(
            "Auto-detecting bed/headrest...\nThis may take a few seconds.",
            None,  # No cancel button
            0, 0,  # Indeterminate progress
            self
        )
        progress.setWindowTitle("Auto-Detection")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.show()
        
        # Process events to show the dialog
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        try:
            # Load current image without rotation (rotation only for preview)
            image_path = self.image_paths[self.current_image_index]
            original_rgb = images.load_image_rgb(image_path, rotation=0)
            
            # Apply rotation for auto-detection (so it works on rotated preview)
            if self.image_rotation != 0:
                original_rgb = images.rotate_image_rgb(original_rgb, self.image_rotation)
            
            # Process events periodically during detection
            QApplication.processEvents()
            
            # Run auto-detection with current parameters
            bed_mask = images.auto_detect_bed_headrest(
                original_rgb,
                bottom_region_ratio=self.auto_detect_params['bottom_region_ratio'],
                min_intensity=self.auto_detect_params['min_intensity'],
                min_size_ratio=self.auto_detect_params['min_size_ratio'],
                max_size_ratio=self.auto_detect_params['max_size_ratio'],
                scan_upward=self.auto_detect_params['scan_upward'],
                grey_tolerance=self.auto_detect_params['grey_tolerance'],
                aggressivity=self.auto_detect_params['aggressivity'],
            )
            
            # Close progress dialog
            progress.close()
            
            if bed_mask is not None and np.sum(bed_mask) > 0:
                # Add as a new selected object
                object_id = str(uuid.uuid4())
                self.selected_objects.append({
                    'id': object_id,
                    'mask': bed_mask,
                    'label': 'bed',  # Default label
                    'slice_index': self.current_image_index,  # Track which slice this was selected on
                })
                
                # Track the slice where mask was selected
                if self.mask_selected_slice_index is None:
                    self.mask_selected_slice_index = self.current_image_index
                
                # Update UI
                self._update_objects_table()
                self._update_combined_mask()
                self.clear_selection_btn.setEnabled(True)
                self.update_preview()
                
                QMessageBox.information(
                    self,
                    "Auto-Detection Successful",
                    f"Detected bed/headrest with {np.sum(bed_mask)} pixels.\n"
                    "The mask will be propagated across all slices."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Auto-Detection Failed",
                    "⚠️ Could not automatically detect bed/headrest in this slice.\n\n"
                    "Remember: Auto-detection scans from BOTTOM UPWARD.\n"
                    "The bed/headrest MUST be UNDERNEATH the body/head.\n\n"
                    "If the bed is on the side or top, use the rotation buttons first."
                )
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Auto-Detection Error",
                f"Error during auto-detection: {str(e)}"
            )
    
    def show_auto_detect_settings(self) -> None:
        """Show settings dialog for auto-detection parameters."""
        dialog = AutoDetectSettingsDialog(self, self.auto_detect_params)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.auto_detect_params.update(dialog.get_params())
    
    def rotate_images(self, angle: int) -> None:
        """
        Update rotation state for preview. Rotation is only applied to the preview
        and will be applied to all images when preprocessing is run.
        
        Parameters
        ----------
        angle : int
            Rotation angle in degrees. Positive = clockwise, negative = counter-clockwise.
        """
        if not self.image_paths or self.input_dir is None:
            return
        
        # Update rotation state
        self.image_rotation = (self.image_rotation + angle) % 360
        if self.image_rotation < 0:
            self.image_rotation += 360
        
        # Update rotation label
        self.rotation_label.setText(f"{self.image_rotation}°")
        
        # Refresh preview to show rotated "after" image
        self.update_preview()
