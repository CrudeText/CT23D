from __future__ import annotations

from dataclasses import dataclass
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

@dataclass
class PreprocessWorkerConfig:
    input_dir: Path
    output_dir: Path
    config: PreprocessConfig


class PreprocessWorker(WorkerBase):
    """
    Run the preprocessing pipeline in a background thread.
    """

    def __init__(self, cfg: PreprocessWorkerConfig) -> None:
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:  # type: ignore[override]
        try:
            def progress_cb(done: int, total: int) -> None:
                self.progress.emit(done, total)

            result = preprocess_slices(
                input_dir=self._cfg.input_dir,
                output_dir=self._cfg.output_dir,
                config=self._cfg.config,
                progress_callback=progress_cb,
            )
            self.finished.emit(result)
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
