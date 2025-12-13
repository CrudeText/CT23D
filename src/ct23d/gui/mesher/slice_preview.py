from __future__ import annotations

from typing import Optional, List

import numpy as np
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSpinBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor

from ct23d.core.models import IntensityBin


class SlicePreviewWidget(QFrame):
    """
    Preview widget that displays a slice with bin colors applied.
    
    Pixels are colored according to which intensity bin they belong to.
    Includes slice navigation controls.
    """
    
    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title and slice selector
        header_layout = QHBoxLayout()
        self._title_label = QLabel("Slice Preview")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self._title_label)
        layout.addLayout(header_layout)
        
        # Slice navigation
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Slice:"))
        self._slice_selector = QSpinBox()
        self._slice_selector.setRange(1, 1)
        self._slice_selector.setValue(1)
        self._slice_selector.setEnabled(False)
        self._slice_selector.valueChanged.connect(self._on_slice_changed)
        nav_layout.addWidget(self._slice_selector)
        self._slice_count_label = QLabel("of 0")
        nav_layout.addWidget(self._slice_count_label)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)
        
        # Image display - make it bigger
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(500, 500)  # Bigger minimum size
        self._image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        layout.addWidget(self._image_label, stretch=1)
        
        self._volume: Optional[np.ndarray] = None
        self._bins: List[IntensityBin] = []
        self._slice_index: int = 0
    
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the volume to preview.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
        """
        self._volume = volume
        if volume is not None and volume.shape[0] > 0:
            num_slices = volume.shape[0]
            # Set up slice selector
            self._slice_selector.setRange(1, num_slices)
            self._slice_selector.setEnabled(True)
            self._slice_count_label.setText(f"of {num_slices}")
            # Set to middle slice
            middle_slice = (num_slices // 2) + 1
            self._slice_selector.setValue(middle_slice)
            self._slice_index = middle_slice - 1  # Convert to 0-indexed
        else:
            self._slice_selector.setEnabled(False)
            self._slice_count_label.setText("of 0")
        self._update_preview()
    
    def _on_slice_changed(self, value: int) -> None:
        """Handle slice selector change."""
        self._slice_index = value - 1  # Convert to 0-indexed
        self._update_preview()
    
    def set_bins(self, bins: List[IntensityBin]) -> None:
        """
        Set the intensity bins to use for coloring.
        
        Parameters
        ----------
        bins : List[IntensityBin]
            List of intensity bins with colors
        """
        self._bins = bins
        self._update_preview()
    
    def _update_preview(self) -> None:
        """Update the preview image with bin colors."""
        if self._volume is None:
            self._image_label.clear()
            return
        
        if self._volume.shape[0] == 0:
            self._image_label.clear()
            return
        
        # Get the current slice
        slice_idx = max(0, min(self._slice_index, self._volume.shape[0] - 1))
        slice_data = self._volume[slice_idx]
        
        # Convert to grayscale if RGB
        if slice_data.ndim == 3 and slice_data.shape[2] == 3:
            grayscale = slice_data.mean(axis=2).astype(np.float32)
        else:
            grayscale = slice_data.astype(np.float32)
        
        # Create colored preview based on bins
        height, width = grayscale.shape
        colored_preview = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a mask for each bin and apply colors
        for bin_obj in self._bins:
            if not bin_obj.enabled:
                continue
            
            # Create mask for pixels in this bin's intensity range
            mask = (grayscale >= bin_obj.low) & (grayscale < bin_obj.high)
            
            if np.any(mask):
                # Get color for this bin
                if bin_obj.color is not None:
                    r, g, b = bin_obj.color
                    color = (int(r * 255), int(g * 255), int(b * 255))
                else:
                    # Default gray if no color set
                    color = (128, 128, 128)
                
                # Apply color to masked pixels
                colored_preview[mask, 0] = color[0]
                colored_preview[mask, 1] = color[1]
                colored_preview[mask, 2] = color[2]
        
        # Pixels not in any bin remain black (0, 0, 0)
        
        # Convert to QImage and display
        qimage = QImage(
            colored_preview.data,
            width,
            height,
            colored_preview.strides[0],
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self._image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            # Use minimum size if actual size not available yet
            label_size = self._image_label.minimumSize()
        
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled_pixmap)
        else:
            # Fallback: use a reasonable default size
            scaled_pixmap = pixmap.scaled(
                500, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled_pixmap)
        
        # Update title with slice info
        if self._volume is not None:
            self._title_label.setText(f"Slice Preview (Slice {slice_idx + 1} / {self._volume.shape[0]})")
        else:
            self._title_label.setText("Slice Preview")

