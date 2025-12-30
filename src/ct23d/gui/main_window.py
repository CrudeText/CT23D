from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .preproc import PreprocessWizard
from .mesher import MesherWizard
from .processing3d import Processing3DWizard
from .status import StatusController


def _find_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml or setup.py.
    """
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    # Fallback: assume we're in src/ct23d/gui, so go up 3 levels to project root
    return Path(__file__).resolve().parent.parent.parent


def _get_icon_path(icon_name: str) -> Path:
    """Get the path to an icon file in the assets/icons directory."""
    project_root = _find_project_root()
    return project_root / "assets" / "icons" / icon_name


class PreprocessingTab(QWidget):
    """
    Wrapper around the actual preprocessing wizard UI.
    """

    def __init__(self, status: Optional[StatusController], parent: Optional[Widget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.wizard = PreprocessWizard(parent=self, status=status)
        layout.addWidget(self.wizard)


class MeshingTab(QWidget):
    """
    Wrapper around the meshing wizard UI.
    """

    def __init__(self, status: Optional[StatusController], parent: Optional[Widget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.wizard = MesherWizard(parent=self, status=status)
        layout.addWidget(self.wizard)


class Processing3DTab(QWidget):
    """
    Wrapper around the 3D Processing wizard UI.
    """

    def __init__(self, status: Optional[StatusController], parent: Optional[Widget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.wizard = Processing3DWizard(parent=self, status=status)
        layout.addWidget(self.wizard)


class MainWindow(QMainWindow):
    """
    Main window for the CT23D application.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("CT23D – CT to 3D")
        self.resize(1100, 700)

        # Set window icon (for taskbar and window title bar)
        self._set_window_icon()

        # These will be set in helpers
        self.status_bar: QStatusBar
        self.status_controller: StatusController
        self.logo_label: Optional[QLabel] = None
        self.patient_info_group: Optional[QGroupBox] = None
        self.patient_info_label: Optional[QLabel] = None

        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_tabs()
        self._add_logo_widget()
        self._add_patient_info_widget()
        
        # Show maximized after all widgets are created
        self.showMaximized()

    # ------------------------------------------------------------------ #
    # UI creation helpers
    # ------------------------------------------------------------------ #

    def _set_window_icon(self) -> None:
        """Set the window icon from the .ico file."""
        icon_path = _get_icon_path("CT23D_icon.ico")
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            # Request largest available icon size (try up to 1024x1024 for maximum visibility)
            # Try multiple sizes in descending order and use the largest one that actually exists
            # Check for exact size match to ensure we got the real size from the file, not a scaled version
            best_pixmap = None
            best_size = 0
            
            for size in [1024, 512, 256, 128, 64, 48, 32]:
                pixmap = icon.pixmap(size, size)
                # Check if we got exactly the size we requested (not a scaled down version)
                if not pixmap.isNull():
                    actual_width = pixmap.width()
                    actual_height = pixmap.height()
                    # Accept if it's exactly the requested size or very close (within 2px)
                    # This means the size exists in the .ico file
                    if actual_width >= size - 2 and actual_height >= size - 2:
                        if size > best_size:
                            best_size = size
                            best_pixmap = pixmap
            
            if best_pixmap is not None and best_size > 0:
                # Create a new icon with the largest pixmap found
                large_icon = QIcon(best_pixmap)
                self.setWindowIcon(large_icon)
            else:
                # Fallback to original icon if no large size found
                self.setWindowIcon(icon)
        else:
            # Fallback to PNG if .ico not found
            png_path = _get_icon_path("CT23D_icon.png")
            if png_path.exists():
                pixmap = QPixmap(str(png_path))
                if not pixmap.isNull():
                    # Scale up the PNG for the icon (use 1024x1024 for maximum size)
                    # Note: The OS may still display it smaller, but we provide the largest size available
                    scaled = pixmap.scaled(1024, 1024, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    icon = QIcon(scaled)
                    self.setWindowIcon(icon)

    def _create_actions(self) -> None:
        self.action_open_project = QAction("Open Project…", self)
        self.action_open_project.setStatusTip("Open an existing CT23D project file")
        self.action_open_project.triggered.connect(self.on_open_project)

        self.action_save_project = QAction("Save Project As…", self)
        self.action_save_project.setStatusTip(
            "Save current settings as a CT23D project file"
        )
        self.action_save_project.triggered.connect(self.on_save_project)

        self.action_view_nrrd = QAction("View NRRD File…", self)
        self.action_view_nrrd.setStatusTip("View NRRD file metadata and 3D visualization")
        self.action_view_nrrd.triggered.connect(self._on_view_nrrd)

        self.action_exit = QAction("Exit", self)
        self.action_exit.setStatusTip("Quit CT23D")
        self.action_exit.triggered.connect(self.close)

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.action_open_project)
        file_menu.addAction(self.action_save_project)
        file_menu.addSeparator()
        file_menu.addAction(self.action_view_nrrd)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About CT23D", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

        # Add logo to the top right of the menu bar
        self._add_logo_to_menu_bar(menu_bar)

    def _add_logo_to_menu_bar(self, menu_bar) -> None:
        """Add the logo image to the top right of the menu bar using corner widget."""
        logo_path = _get_icon_path("CT23D_icon.png")
        if not logo_path.exists():
            return

        # Create a widget to hold the logo
        logo_widget = QWidget()
        logo_layout = QHBoxLayout(logo_widget)
        logo_layout.setContentsMargins(5, 2, 10, 2)  # Some padding
        logo_layout.setSpacing(0)

        # Create label for the logo
        logo_label = QLabel()
        pixmap = QPixmap(str(logo_path))
        
        # Scale the logo to a reasonable size (e.g., 60px height, maintaining aspect ratio)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaledToHeight(60, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setMinimumSize(scaled_pixmap.size())
            logo_label.setMaximumSize(scaled_pixmap.size())
        else:
            return
        
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        logo_layout.addStretch()  # Push logo to the right
        logo_layout.addWidget(logo_label)
        
        # Set fixed size for the widget
        logo_widget.setFixedHeight(60)
        logo_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Set the logo widget as the menu bar's corner widget (top right)
        menu_bar.setCornerWidget(logo_widget, Qt.TopRightCorner)

    def _add_logo_widget(self) -> None:
        """Add logo as a positioned widget in the top-right corner of the window."""
        logo_path = _get_icon_path("CT23D_icon.png")
        if not logo_path.exists():
            return

        # Create a label for the logo
        self.logo_label = QLabel(self)
        pixmap = QPixmap(str(logo_path))
        
        if pixmap.isNull():
            return
        
        # Scale the logo to a larger size for better visibility
        scaled_pixmap = pixmap.scaledToHeight(140, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled_pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        # Position it in the top right corner
        self.logo_label.setFixedSize(scaled_pixmap.size())
        
        # Ensure logo stays on top
        self.logo_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # Don't block mouse events
        self.logo_label.setStyleSheet("background: transparent;")
        
        # Set initial position - use a timer to ensure it's positioned after window is shown
        QTimer.singleShot(0, self._update_logo_position)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Override resize event to update logo and patient info positions."""
        super().resizeEvent(event)
        # Use a small delay to ensure geometry is updated
        QTimer.singleShot(10, self._update_logo_position)

    def showEvent(self, event: QShowEvent) -> None:
        """Override show event to position logo and patient info after window is shown."""
        super().showEvent(event)
        QTimer.singleShot(50, self._update_logo_position)

    def _update_logo_position(self) -> None:
        """Update the logo position to keep it in the top-right corner."""
        if self.logo_label is None or not self.logo_label.isVisible():
            return
        
        # Get menu bar height to position logo below it
        menu_bar_height = self.menuBar().height() if self.menuBar().isVisible() else 0
        
        # Calculate position in client coordinates (excluding window frame)
        x = self.width() - self.logo_label.width() - 15  # 15px from right edge
        y = menu_bar_height + 8  # 8px below menu bar
        
        # Ensure logo is within window bounds
        if x < 0:
            x = 0
        if y < menu_bar_height:
            y = menu_bar_height + 2
        
        self.logo_label.move(x, y)
        self.logo_label.raise_()  # Always bring to front
        self.logo_label.show()  # Ensure it's visible
        
        # Update patient info position as well
        self._update_patient_info_position()
    
    def _add_patient_info_widget(self) -> None:
        """Add Patient Info box as a positioned widget in the top-right area (left of logo)."""
        # Create Patient Info group box
        self.patient_info_group = QGroupBox("Patient Info", self)
        self.patient_info_group.setStyleSheet("""
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
        
        # Create two-column grid layout for more compact display
        patient_info_layout = QGridLayout(self.patient_info_group)
        patient_info_layout.setContentsMargins(8, 6, 8, 4)  # Reduced bottom margin from 8 to 4
        patient_info_layout.setSpacing(4)
        patient_info_layout.setColumnStretch(0, 1)  # Left column (labels)
        patient_info_layout.setColumnStretch(1, 2)  # Right column (values)
        
        # Create labels for patient info fields (will be populated dynamically)
        # We'll create placeholders that will be updated with actual data
        self.patient_info_label = QLabel("No DICOM files loaded")
        font = QFont()
        font.setPointSize(8)  # Smaller font size
        self.patient_info_label.setFont(font)
        self.patient_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.patient_info_label.setWordWrap(True)
        self.patient_info_label.setStyleSheet("color: gray;")
        # Span both columns for the "No DICOM" message
        patient_info_layout.addWidget(self.patient_info_label, 0, 0, 1, 2)
        
        # Set initial size - wider but shorter for two-column layout
        self.patient_info_group.setFixedWidth(380)  # Wider for two columns
        self.patient_info_group.setFixedHeight(135)  # Reduced height from 140 to 135 to avoid overlapping Slice Preview
        
        # Ensure it stays on top but doesn't block mouse events from tabs
        self.patient_info_group.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Allow interactions
        self.patient_info_group.setStyleSheet(self.patient_info_group.styleSheet() + "background: rgba(40, 40, 40, 240);")
        
        # Set initial position - use a timer to ensure it's positioned after window is shown
        QTimer.singleShot(0, self._update_patient_info_position)
    
    def _update_patient_info_position(self) -> None:
        """Update the Patient Info box position to keep it at top right (left of logo), same level as tabs."""
        if self.patient_info_group is None or not self.patient_info_group.isVisible():
            return
        
        # Get menu bar height
        menu_bar_height = self.menuBar().height() if self.menuBar().isVisible() else 0
        
        # Try to get the actual visual top position of the preview_group QGroupBox
        # by finding it in the preprocessing tab and getting its real position
        preview_group_visual_top = None
        if hasattr(self, 'tabs') and self.tabs is not None and self.tabs.currentIndex() == 0:
            # We're in preprocessing tab - try to find preview_group
            try:
                # The preview_group should be in the preprocessing wizard's layout
                preproc_wizard = self.preproc_tab.wizard
                # Search for the QGroupBox with title "Image Preview"
                def find_preview_group(widget):
                    """Recursively find the QGroupBox with title 'Image Preview'."""
                    if isinstance(widget, QGroupBox) and widget.title() == "Image Preview":
                        return widget
                    for child in widget.children():
                        if isinstance(child, QWidget):
                            result = find_preview_group(child)
                            if result:
                                return result
                    return None
                
                preview_group = find_preview_group(preproc_wizard)
                if preview_group and preview_group.isVisible() and preview_group.geometry().isValid():
                    # Get the widget's top-left corner position relative to the main window
                    # mapTo converts coordinates from the source widget to the target widget
                    # QPoint(0, 0) represents the top-left corner of preview_group in its own coordinate system
                    from PySide6.QtCore import QPoint
                    preview_top_left_in_main = preview_group.mapTo(self, QPoint(0, 0))
                    # This gives us the visual top position of the QGroupBox border
                    preview_group_visual_top = preview_top_left_in_main.y()
            except Exception:
                pass  # Fallback to estimated calculation
        
        # If we couldn't get actual position, use estimated calculation
        if preview_group_visual_top is None:
            # Get tab bar height
            tab_bar_height = 0
            if hasattr(self, 'tabs') and self.tabs is not None:
                tab_bar = self.tabs.tabBar()
                if tab_bar is not None and tab_bar.isVisible():
                    tab_bar_height = tab_bar.height()
            
            # Estimated calculation (fallback)
            layout_top_margin = 10
            directory_row_height = 55
            layout_spacing = 10
            directory_rows_total = directory_row_height + layout_spacing + directory_row_height
            preview_group_visual_top = menu_bar_height + tab_bar_height + layout_top_margin + directory_rows_total
        
        # Position Patient Info box so its bottom (y + height) is CLEARLY above the preview box visual border
        # Account for QGroupBox border (2px from stylesheet) - the visual border starts at preview_group_visual_top
        # Patient Info box height is 140px
        # We want: y + 140 + clearance < preview_group_visual_top (top of preview box border)
        # So: y < preview_group_visual_top - 140 - clearance
        clearance = 15  # Generous clearance to ensure no visual overlap between the grey borders
        max_y_for_bottom = preview_group_visual_top - self.patient_info_group.height() - clearance
        
        # Position as high as possible: position at menu_bar level
        preferred_y = menu_bar_height + 2  # Just below menu bar
        
        # Use whichever is higher (smaller y value) to position the box as high as possible
        # while ensuring bottom doesn't overlap preview
        y = min(preferred_y, max_y_for_bottom)
        
        # Final safety check: ensure bottom is definitely above preview box
        if y + self.patient_info_group.height() + clearance >= preview_group_visual_top:
            # Force it higher
            y = preview_group_visual_top - self.patient_info_group.height() - clearance
        
        # Now calculate x position (on the right side, left of logo)
        if self.logo_label is not None and self.logo_label.isVisible():
            # Logo is positioned at: self.width() - logo_width - 15
            logo_width = self.logo_label.width()
            logo_x = self.width() - logo_width - 15
            # Patient Info should be to the left of logo with 15px spacing
            x = logo_x - self.patient_info_group.width() - 15
        else:
            # If no logo, position at right edge with margin
            x = self.width() - self.patient_info_group.width() - 15
        
        # Ensure it's within window bounds
        if x < 0:
            # If it would be off-screen to the left, position it at a safe distance from left edge
            x = max(10, self.width() - self.patient_info_group.width() - 15)
        if y < menu_bar_height:
            y = menu_bar_height + 2
        
        self.patient_info_group.move(x, y)
        self.patient_info_group.raise_()  # Bring to front
        self.patient_info_group.show()  # Ensure it's visible
    
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change to update patient info."""
        # Update patient info based on active tab
        self._update_patient_info_from_active_tab()
        # Update position after tab change (layout might have changed)
        QTimer.singleShot(50, self._update_patient_info_position)
    
    def _update_patient_info_from_active_tab(self) -> None:
        """Update patient info box based on the currently active tab."""
        if self.patient_info_label is None or self.patient_info_group is None:
            return
        
        try:
            patient_info_dict = None
            if self.tabs.currentIndex() == 0:  # Preprocessing tab
                # Get patient info from preprocessing wizard
                if hasattr(self.preproc_tab.wizard, 'get_patient_info'):
                    text, style = self.preproc_tab.wizard.get_patient_info()
                    # Parse the HTML text to extract key-value pairs for two-column layout
                    if hasattr(self.preproc_tab.wizard, 'image_paths') and self.preproc_tab.wizard.image_paths:
                        from ct23d.core import images
                        patient_info_dict = images.get_dicom_patient_info(self.preproc_tab.wizard.image_paths[0])
            elif self.tabs.currentIndex() == 1:  # Meshing tab
                # Get patient info from meshing wizard
                if hasattr(self.meshing_tab.wizard, 'get_patient_info'):
                    text, style = self.meshing_tab.wizard.get_patient_info()
                    # Parse to get dict
                    if hasattr(self.meshing_tab.wizard, '_processed_dir') and self.meshing_tab.wizard._processed_dir:
                        from ct23d.core import images
                        try:
                            paths = images.list_slice_files(self.meshing_tab.wizard._processed_dir)
                            if paths:
                                patient_info_dict = images.get_dicom_patient_info(paths[0])
                        except Exception:
                            pass
            
            # Update display: use two-column layout if we have structured data, otherwise single message
            layout = self.patient_info_group.layout()
            if layout is None:
                return
            
            # Clear existing widgets (except keep the group box)
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            if patient_info_dict and any(patient_info_dict.values()):
                # Two-column layout for structured data
                row = 0
                fields = [
                    ('PatientName', 'Patient Name:'),
                    ('PatientID', 'Patient ID:'),
                    ('PatientBirthDate', 'Birth Date:'),
                    ('PatientSex', 'Sex:'),
                    ('StudyDate', 'Study Date:'),
                    ('StudyTime', 'Study Time:'),
                    ('StudyDescription', 'Study Description:'),
                    ('Modality', 'Modality:'),
                ]
                
                # Use smaller font for patient info labels
                font = QFont()
                font.setPointSize(8)  # Smaller font size
                
                for key, label_text in fields:
                    if key in patient_info_dict and patient_info_dict[key]:
                        label = QLabel(f"<b>{label_text}</b>")
                        label.setFont(font)
                        label.setStyleSheet("color: white;")
                        value = QLabel(str(patient_info_dict[key]))
                        value.setFont(font)
                        value.setStyleSheet("color: white;")
                        value.setWordWrap(True)
                        layout.addWidget(label, row, 0)
                        layout.addWidget(value, row, 1)
                        row += 1
                
                if row == 0:
                    # No data found
                    no_data_label = QLabel("DICOM file loaded (no patient info available)")
                    font = QFont()
                    font.setPointSize(8)  # Smaller font size
                    no_data_label.setFont(font)
                    no_data_label.setStyleSheet("color: gray;")
                    layout.addWidget(no_data_label, 0, 0, 1, 2)
            else:
                # Single message for no data
                no_data_label = QLabel("No DICOM files loaded")
                font = QFont()
                font.setPointSize(8)  # Smaller font size
                no_data_label.setFont(font)
                no_data_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                no_data_label.setStyleSheet("color: gray;")
                layout.addWidget(no_data_label, 0, 0, 1, 2)
                
        except Exception:
            # Fallback: clear and show default message
            layout = self.patient_info_group.layout()
            if layout:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                no_data_label = QLabel("No DICOM files loaded")
                font = QFont()
                font.setPointSize(8)  # Smaller font size
                no_data_label.setFont(font)
                no_data_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                no_data_label.setStyleSheet("color: gray;")
                layout.addWidget(no_data_label, 0, 0, 1, 2)

    def _create_status_bar(self) -> None:
        status = QStatusBar(self)
        status.showMessage("Ready")
        self.setStatusBar(status)

        self.status_bar = status
        self.status_controller = StatusController(status, parent=self)

    def _create_central_tabs(self) -> None:
        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)  # Keep document mode for original look
        tabs.setTabPosition(QTabWidget.North)
        
        # Make tabs more visible with subtle styling that preserves original colors
        tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 10px 20px;
                margin-right: 2px;
                font-size: 12pt;
                font-weight: bold;
                min-width: 150px;
            }
        """)

        self.preproc_tab = PreprocessingTab(status=self.status_controller, parent=self)
        self.meshing_tab = MeshingTab(status=self.status_controller, parent=self)
        self.processing3d_tab = Processing3DTab(status=self.status_controller, parent=self)

        tabs.addTab(self.preproc_tab, "Preprocessing")
        tabs.addTab(self.meshing_tab, "Meshing")
        tabs.addTab(self.processing3d_tab, "3D Processing")

        self.setCentralWidget(tabs)
        
        # Store reference to tabs widget
        self.tabs = tabs
        
        # Connect tab change signal to update patient info
        tabs.currentChanged.connect(self._on_tab_changed)
        
        # Set up patient info update callbacks for wizards
        def update_patient_info_callback() -> None:
            self._update_patient_info_from_active_tab()
        
        # Set callback on wizards so they can notify when directories change
        if hasattr(self.preproc_tab.wizard, '_patient_info_callback'):
            self.preproc_tab.wizard._patient_info_callback = update_patient_info_callback
        else:
            self.preproc_tab.wizard._patient_info_callback = update_patient_info_callback
        
        if hasattr(self.meshing_tab.wizard, '_patient_info_callback'):
            self.meshing_tab.wizard._patient_info_callback = update_patient_info_callback
        else:
            self.meshing_tab.wizard._patient_info_callback = update_patient_info_callback
        
        # Initial update
        QTimer.singleShot(100, self._update_patient_info_from_active_tab)
        
        # Connect preprocessing output to meshing input
        # When preprocessing completes, set it as default for meshing
        def on_preprocessing_complete() -> None:
            # Get the last output directory from preprocessing wizard
            if hasattr(self.preproc_tab.wizard, '_last_output_dir'):
                output_dir = self.preproc_tab.wizard._last_output_dir
                if output_dir is not None:
                    self.meshing_tab.wizard.set_default_processed_dir(output_dir)
        
        # Connect to preprocessing completion
        # We'll need to add a signal or check periodically
        # For now, we'll use a property accessor

    # ------------------------------------------------------------------ #
    # Slots / actions
    # ------------------------------------------------------------------ #

    def on_open_project(self) -> None:
        dlg = QFileDialog(self)
        dlg.setWindowTitle("Open CT23D project")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("CT23D Project (*.yaml *.yml);;All Files (*.*)")

        if dlg.exec():
            paths = dlg.selectedFiles()
            if not paths:
                return
            proj_path = Path(paths[0])

            QMessageBox.information(
                self,
                "Open Project (not implemented yet)",
                f"You selected project file:\n\n{proj_path}\n\n"
                "Loading of project configs will be wired later.",
            )

    def on_save_project(self) -> None:
        dlg = QFileDialog(self)
        dlg.setWindowTitle("Save CT23D project as…")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilter("CT23D Project (*.yaml *.yml);;All Files (*.*)")

        if dlg.exec():
            paths = dlg.selectedFiles()
            if not paths:
                return
            proj_path = Path(paths[0])

            QMessageBox.information(
                self,
                "Save Project (not implemented yet)",
                f"You chose to save project file:\n\n{proj_path}\n\n"
                "Saving of project configs will be wired later.",
            )

    def on_about(self) -> None:
        QMessageBox.information(
            self,
            "About CT23D",
            "CT23D – CT slices to 3D meshes\n\n"
            "A modular Python application for converting stacks of CT slice images "
            "into 3D meshes. Provides both a graphical interface and programmatic core "
            "to preprocess CT slices, build volumetric data, analyze intensity distributions, "
            "and generate intensity-based 3D meshes.\n\n"
            "For detailed documentation and usage instructions, please refer to:\n"
            "https://github.com/CrudeText/CT23D",
        )
    
    def _on_view_nrrd(self) -> None:
        """Handle View NRRD File menu action."""
        from .view_nrrd_dialog import ViewNRRDDialog
        
        dialog = ViewNRRDDialog(self)
        dialog.exec()
