from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QStatusBar,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

from ct23d.gui.workers import WorkerBase


class StatusController(QObject):
    """
    Central helper to manage status bar messages, message boxes,
    and running background workers with a modal progress dialog.
    """

    def __init__(self, status_bar: QStatusBar, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._status_bar = status_bar
        self._parent = parent

        # Keep strong references to running workers so they are not
        # garbage-collected while still executing.
        self._active_workers: List[WorkerBase] = []

    # ------------------------------------------------------------------ #
    # Simple status / message helpers
    # ------------------------------------------------------------------ #
    def show_message(self, text: str, timeout_ms: int = 5000) -> None:
        """Show a transient message in the status bar."""
        if self._status_bar is not None:
            self._status_bar.showMessage(text, timeout_ms)

    def show_error(self, text: str) -> None:
        """Show an error in the status bar and a modal message box."""
        if self._status_bar is not None:
            self._status_bar.showMessage(text, 8000)
        QMessageBox.critical(self._parent, "Error", text)

    def show_info(self, text: str) -> None:
        """Show an information message."""
        if self._status_bar is not None:
            self._status_bar.showMessage(text, 5000)
        QMessageBox.information(self._parent, "Information", text)

    # ------------------------------------------------------------------ #
    # Threaded workers with progress dialog
    # ------------------------------------------------------------------ #
    def run_threaded_with_progress(
        self,
        worker: WorkerBase,
        title: Optional[str] = None,
        on_success: Optional[Callable[[object], None]] = None,
        **_: object,
    ) -> None:
        """
        Run a WorkerBase (QThread) with a modal QProgressDialog.

        Parameters
        ----------
        worker:
            Instance of WorkerBase (or subclass) which emits:
            - progress(int current, int total)
            - finished(object result)
            - error(str message)

        title:
            Optional dialog title. If None, a generic "Working..." is used.

        on_success:
            Optional callback called with the worker's result once it finishes
            successfully.

        **_:
            Ignored extra keyword arguments, for backward compatibility
            (e.g. callers passing label_text, progress_max, etc.).
        """
        dlg_title = title or "Working..."

        dialog = QProgressDialog(dlg_title, "Cancel", 0, 0, self._parent)
        dialog.setWindowTitle(dlg_title)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        # Track the current maximum (total steps) for the progress bar.
        state = {"max": 0}

        # Keep a strong reference to the worker while it is running.
        self._active_workers.append(worker)

        def cleanup_worker() -> None:
            """Remove worker from active list and wait for it to finish."""
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            # Ensure the thread has fully stopped before Qt destroys it.
            worker.wait()

        def on_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            if state["max"] != total:
                state["max"] = total
                dialog.setMaximum(total)
            dialog.setValue(done)
            dialog.setLabelText(f"{done} / {total}")

        def on_error(message: str) -> None:
            dialog.hide()
            cleanup_worker()
            self.show_error(message)

        def on_finished(result: object) -> None:
            dialog.hide()
            cleanup_worker()
            if on_success is not None:
                on_success(result)

        # Wire up signals
        dialog.canceled.connect(worker.requestInterruption)
        worker.progress.connect(on_progress)   # type: ignore[arg-type]
        worker.error.connect(on_error)        # type: ignore[arg-type]
        worker.finished.connect(on_finished)  # type: ignore[arg-type]
        worker.finished.connect(worker.deleteLater)

        # Start background work and show dialog
        worker.start()
        dialog.show()
