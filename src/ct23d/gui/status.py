from __future__ import annotations

import time
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (
    QApplication,
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
        state = {"max": 0, "current_phase": "starting", "phase_current": 0, "phase_total": 0, "phase_start_time": None}
        
        # Determine phase names based on title (different for preprocessing vs meshing vs file size vs loading volume)
        title_lower = (title or "").lower()
        is_meshing = "meshes" in title_lower
        is_file_size = "file size" in title_lower or "calculating" in title_lower
        is_loading_volume = "loading volume" in title_lower or "loading image" in title_lower
        
        if is_file_size:
            # File size calculation: simple single phase
            all_phases = ["calculating"]
            phase_names = {
                "calculating": "Calculating file size"
            }
        elif is_meshing:
            all_phases = ["Building masks", "Extracting meshes", "Saving files"]
            phase_names = {
                "Building masks": "Building masks",
                "Extracting meshes": "Extracting meshes",
                "Saving files": "Saving files"
            }
        elif is_loading_volume:
            # Loading volume for meshing
            all_phases = ["Loading image slices", "Building volume", "Computing histograms"]
            phase_names = {
                "Loading image slices": "Loading image slices",
                "Building volume": "Building volume",
                "Computing histograms": "Computing histograms"
            }
        else:
            all_phases = ["loading", "processing", "saving"]
            phase_names = {
                "loading": "Loading images",
                "processing": "Processing volume",
                "saving": "Saving slices"
            }
        
        # Track completed phases dynamically
        completed_phases: dict[str, bool] = {}
        for p in all_phases:
            completed_phases[p] = False
        
        # Track start time for elapsed time calculation
        start_time = time.time()
        
        # Timer for independent time updates
        timer = QTimer()
        timer.setInterval(100)  # Update every 100ms

        # Keep a strong reference to the worker while it is running.
        self._active_workers.append(worker)

        def cleanup_worker() -> None:
            """Remove worker from active list and wait for it to finish."""
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            # Ensure the thread has fully stopped before Qt destroys it.
            # Check if worker is still alive before calling wait() to avoid RuntimeError
            try:
                if worker.isRunning():
                    worker.wait(1000)  # Wait up to 1 second
            except RuntimeError:
                # Worker object already deleted, ignore
                pass

        def format_time(seconds: float) -> str:
            """Format seconds as MM:SS or HH:MM:SS."""
            seconds = int(seconds)
            if seconds < 3600:
                minutes = seconds // 60
                secs = seconds % 60
                return f"{minutes:02d}:{secs:02d}"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        def update_timer_display() -> None:
            """Update timer display independently of progress updates."""
            elapsed = time.time() - start_time
            elapsed_str = format_time(elapsed)
            
            # Get current phase info
            phase = state["current_phase"]
            current = state["phase_current"]
            phase_total = state["phase_total"]
            
            # Build phase status lines
            if is_file_size:
                # For file size calculation, show simple progress
                if phase_total > 0:
                    status_text = f"{phase_names.get(phase, 'Calculating')}: {current} / {phase_total}"
                else:
                    status_text = phase_names.get(phase, "Calculating file size...")
            else:
                status_lines = []
                for p in all_phases:
                    p_name = phase_names.get(p, p.capitalize())
                    if completed_phases.get(p, False):
                        status_lines.append(f"✓ {p_name}: Complete")
                    elif p == phase and phase_total > 0:
                        status_lines.append(f"→ {p_name}: {current} / {phase_total}")
                    else:
                        status_lines.append(f"  {p_name}: Pending")
                
                status_text = "\n".join(status_lines)
            
            # Calculate estimated time remaining for current phase
            if current > 0 and phase_total > 0 and state["phase_start_time"] is not None:
                phase_elapsed = time.time() - state["phase_start_time"]
                estimated_phase_total = phase_elapsed * phase_total / current if current > 0 else 0
                phase_remaining = max(0, estimated_phase_total - phase_elapsed)
                remaining_str = format_time(phase_remaining)
                status_text += f"\n\nElapsed: {elapsed_str} | Remaining: ~{remaining_str}"
            else:
                status_text += f"\n\nElapsed: {elapsed_str}"
            
            dialog.setLabelText(status_text)
        
        def on_phase_progress(phase: str, current: int, phase_total: int, overall_total: int) -> None:
            """Handle phase-aware progress updates."""
            # Update overall maximum if it changed (should increase as we discover total)
            if overall_total > 0 and dialog.maximum() != overall_total:
                dialog.setMaximum(overall_total)
            
            # Check if we're switching phases
            if state["current_phase"] != phase:
                # New phase starting
                state["current_phase"] = phase
                state["phase_current"] = 0
                state["phase_total"] = phase_total
                state["phase_start_time"] = time.time()
                # Don't reset progress bar value - keep overall progress
            
            # Update phase progress
            state["phase_current"] = current
            state["phase_total"] = phase_total
            
            # Check if phase is complete
            if current >= phase_total and phase_total > 0:
                completed_phases[phase] = True
            
            # Update progress bar with overall progress
            # For phase-aware progress, use the overall_total directly (worker calculates it)
            # But we need to calculate it ourselves: completed phases progress + current phase progress
            # Since we're tracking phases, calculate based on overall_total structure
            # The worker emits current as progress within the phase, and overall_total as total across all
            # For simplicity, track completed progress and add current phase progress
            # But for now, use the overall_total calculation from worker signals
            # If overall_total matches phase_total, we're in the first phase, so use current directly
            # Otherwise, calculate: previous phases progress + current phase progress
            if overall_total > phase_total:
                # Multi-phase: calculate overall progress
                # Estimate: (overall_total - phase_total) represents completed phases
                # Add current phase progress
                # Actually, the worker should be emitting overall progress correctly in the progress signal
                # For phase_progress, we'll calculate based on phase completion
                # Simple approach: use a calculated value based on completed phases
                completed_progress = overall_total - phase_total  # Previous phases
                dialog.setValue(completed_progress + current)
            else:
                # Single phase or first phase
                dialog.setValue(current)
            
            # Update display (timer will also update it, but this ensures immediate update)
            update_timer_display()
            
            # Force UI update
            QApplication.processEvents()
        
        def on_progress(done: int, total: int) -> None:
            """Fallback progress handler (for compatibility)."""
            if total <= 0:
                return
            if state["max"] != total:
                state["max"] = total
                dialog.setMaximum(total)
            dialog.setValue(done)
            
            # For file size calculation, update state to show progress
            if is_file_size:
                state["current_phase"] = "calculating"
                state["phase_current"] = done
                state["phase_total"] = total
                if state["phase_start_time"] is None:
                    state["phase_start_time"] = time.time()
                update_timer_display()
            
            QApplication.processEvents()

        def on_error(message: str) -> None:
            timer.stop()
            # Stop the worker immediately
            worker.requestInterruption()
            worker.terminate()  # Force stop if it's still running
            cleanup_worker()
            
            # Don't hide dialog immediately - let user see the cancellation
            if "cancelled" in message.lower() or "cancel" in message.lower():
                # Update dialog to show cancellation message clearly
                dialog.setLabelText(f"⚠️ {message}\n\nThe operation has been stopped.")
                dialog.setCancelButtonText("Close")
                dialog.setValue(dialog.maximum())  # Set to 100% to show it's done
                # Force UI update immediately
                QApplication.processEvents()
                # Keep dialog open and let user close it manually
                # Don't auto-close - user can click "Close" button when ready
            else:
                dialog.hide()
                self.show_error(message)

        def on_finished(result: object) -> None:
            # Only process result if not cancelled
            if not worker.isInterruptionRequested():
                # Don't hide dialog yet - let on_success callback complete first
                # Update dialog to show we're finalizing
                dialog.setLabelText("Finalizing...\n\nSetting up preview and graphs...")
                dialog.setValue(dialog.maximum())
                QApplication.processEvents()
                
                # Call on_success callback while dialog is still visible
                # Pass dialog and progress updater to callback if it accepts them
                if on_success is not None:
                    try:
                        import inspect
                        sig = inspect.signature(on_success)
                        params = list(sig.parameters.keys())
                        # If callback accepts dialog and progress updater, pass them
                        if len(params) >= 3:
                            # Has result, progress_dialog, and progress_updater parameters
                            on_success(result, dialog, on_phase_progress)
                        elif len(params) >= 1:
                            # Has at least result parameter
                            on_success(result)
                        else:
                            # No parameters
                            on_success()
                    except Exception:
                        # Fallback: just call with result if signature detection fails
                        try:
                            on_success(result)
                        except Exception:
                            pass
                    finally:
                        # Hide dialog after on_success completes
                        dialog.hide()
                else:
                    dialog.hide()
                cleanup_worker()
            else:
                # Was cancelled, just clean up silently
                dialog.hide()
                cleanup_worker()

        # Wire up signals
        def on_cancel() -> None:
            """Handle cancel button click."""
            worker.requestInterruption()
            # Force immediate termination of the thread
            if worker.isRunning():
                worker.terminate()
            dialog.setLabelText(f"{dialog.labelText()}\n\n⚠️ Cancelling... Please wait.")
            dialog.setCancelButtonText("Cancelling...")
            # Force immediate UI update
            QApplication.processEvents()
        
        dialog.canceled.connect(on_cancel)
        worker.progress.connect(on_progress)   # type: ignore[arg-type]
        
        # Connect phase-aware progress if available (for PreprocessWorker)
        if hasattr(worker, 'phase_progress'):
            worker.phase_progress.connect(on_phase_progress)  # type: ignore[attr-defined]
        
        worker.error.connect(on_error)        # type: ignore[arg-type]
        worker.finished.connect(on_finished)  # type: ignore[arg-type]
        worker.finished.connect(worker.deleteLater)
        
        # Connect timer to update display independently
        timer.timeout.connect(update_timer_display)
        timer.start()

        # Start background work and show dialog
        # Set initial label based on title (different phases for different tasks)
        if "meshes" in (title or "").lower():
            initial_text = "  Building masks: Pending\n  Extracting meshes: Pending\n  Saving files: Pending\n\nElapsed: 00:00"
        else:
            initial_text = "  Loading images: Pending\n  Processing volume: Pending\n  Saving slices: Pending\n\nElapsed: 00:00"
        dialog.setLabelText(initial_text)
        worker.start()
        dialog.show()
        
        # Stop timer when finished
        worker.finished.connect(timer.stop)
        worker.error.connect(timer.stop)
        
        # Force initial UI update
        QApplication.processEvents()
