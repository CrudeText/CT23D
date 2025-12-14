from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .preproc import PreprocessWizard
from .mesher import MesherWizard
from .status import StatusController


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


class MainWindow(QMainWindow):
    """
    Main window for the CT23D application.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("CT23D – CT to 3D")
        self.resize(1100, 700)

        # These will be set in helpers
        self.status_bar: QStatusBar
        self.status_controller: StatusController

        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_tabs()
        
        # Show maximized after all widgets are created
        self.showMaximized()

    # ------------------------------------------------------------------ #
    # UI creation helpers
    # ------------------------------------------------------------------ #

    def _create_actions(self) -> None:
        self.action_open_project = QAction("Open Project…", self)
        self.action_open_project.setStatusTip("Open an existing CT23D project file")
        self.action_open_project.triggered.connect(self.on_open_project)

        self.action_save_project = QAction("Save Project As…", self)
        self.action_save_project.setStatusTip(
            "Save current settings as a CT23D project file"
        )
        self.action_save_project.triggered.connect(self.on_save_project)

        self.action_exit = QAction("Exit", self)
        self.action_exit.setStatusTip("Quit CT23D")
        self.action_exit.triggered.connect(self.close)

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.action_open_project)
        file_menu.addAction(self.action_save_project)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About CT23D", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

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

        tabs.addTab(self.preproc_tab, "Preprocessing")
        tabs.addTab(self.meshing_tab, "Meshing")

        self.setCentralWidget(tabs)
        
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
            "Core refactored into a clean library with CLI and GUI.\n"
            "GUI is under active development (Phase 2).",
        )
