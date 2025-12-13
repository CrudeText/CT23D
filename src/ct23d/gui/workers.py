from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ct23d.core.models import PreprocessConfig
from ct23d.core.preprocessing import preprocess_slices


# ---------------------------------------------------------------------------
# Base QThread worker
# ---------------------------------------------------------------------------

class WorkerBase(QThread):
    """
    Base class for threaded workers.

    Signals:
        progress(current, total)
        finished(result)
        error(message)
    """

    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)

    def _handle_exception(self, exc: BaseException) -> None:  # noqa: BLE001
        self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Specialized worker for preprocessing
# ---------------------------------------------------------------------------

class PreprocessWorker(WorkerBase):
    """
    Run the preprocessing pipeline in a background thread.
    """
    
    # Custom signal for phase-aware progress
    phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total

    def __init__(self, cfg: PreprocessConfig) -> None:
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:  # type: ignore[override]
        try:
            # Create a cancellation flag that can be checked
            self._cancelled = False
            
            def progress_cb(phase: str, current: int, phase_total: int, overall_total: int) -> None:
                # Check for interruption request FIRST, before any other operations
                if self.isInterruptionRequested():
                    self._cancelled = True
                    raise InterruptedError("Preprocessing was cancelled by user")
                # Emit both the standard progress (for compatibility) and phase-aware progress
                self.progress.emit(current, overall_total)
                self.phase_progress.emit(phase, current, phase_total, overall_total)

            result = preprocess_slices(
                self._cfg,
                progress_cb=progress_cb,
            )
            
            # Check one more time before finishing
            if self.isInterruptionRequested() or self._cancelled:
                raise InterruptedError("Preprocessing was cancelled by user")
            
            self.finished.emit(result)
        except InterruptedError as e:
            # User cancelled - emit error signal
            self.error.emit("Preprocessing was cancelled by user")
        except BaseException as exc:  # noqa: BLE001
            self._handle_exception(exc)


# ---------------------------------------------------------------------------
# Generic worker for arbitrary long-running functions
# ---------------------------------------------------------------------------

class FunctionWorker(WorkerBase):
    """
    Generic worker that runs an arbitrary callable in a thread.

    If with_progress=True, the target function must accept a first argument
    `progress_cb(current, total)` and is expected to call it periodically.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *args: Any,
        with_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._with_progress = with_progress

    def run(self) -> None:  # type: ignore[override]
        try:
            if self._with_progress:
                def progress_cb(done: int, total: int) -> None:
                    self.progress.emit(done, total)

                result = self._func(progress_cb, *self._args, **self._kwargs)
            else:
                result = self._func(*self._args, **self._kwargs)

            self.finished.emit(result)
        except BaseException as exc:  # noqa: BLE001
            self._handle_exception(exc)
