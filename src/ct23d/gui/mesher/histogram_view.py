from __future__ import annotations

from typing import Optional, Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QFrame, QVBoxLayout


class HistogramView(QFrame):
    """
    Wrapper around a PyQtGraph PlotWidget to display intensity histograms.

    Supported public API:

    1) From intensities (used by the mesher wizard):

        set_histogram(intensities, n_bins=256, value_range=(vmin, vmax))

       where `intensities` is a 1D or ND array of scalar values.

    2) From precomputed histogram (counts and bin edges):

        set_histogram(counts=counts, bin_edges=bin_edges)

    3) From a volume:

        update_from_volume(volume, num_bins=256, vmin=0, vmax=255)
        update_histogram(...)   # alias

    The implementation ensures PyQtGraph's requirement for stepMode=True:
        len(x) = len(y) + 1
    """

    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(self)
        layout.addWidget(self._plot)

        self._plot.setLabel("bottom", "Intensity")
        self._plot.setLabel("left", "Count")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Basic operations
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Clear the histogram plot."""
        self._plot.clear()

    # ------------------------------------------------------------------ #
    # Core plotting helper
    # ------------------------------------------------------------------ #
    def _plot_histogram(self, counts: np.ndarray, bin_edges: np.ndarray) -> None:
        """
        Internal helper that actually draws the histogram.

        Enforces len(x) = len(y) + 1 for stepMode=True.
        """
        y = np.asarray(counts, dtype=float).ravel()
        x = np.asarray(bin_edges, dtype=float).ravel()

        # Adjust x so that len(x) = len(y) + 1 for stepMode=True
        if x.shape[0] == y.shape[0]:
            # Assume these are bin centers; create edges from them.
            if len(x) > 1:
                step = x[1] - x[0]
            else:
                step = 1.0
            x = np.concatenate([x - 0.5 * step, [x[-1] + 0.5 * step]])
        elif x.shape[0] != y.shape[0] + 1:
            # Fallback: generic 0..N+1 edges
            x = np.arange(len(y) + 1, dtype=float)

        self._plot.clear()
        self._plot.plot(
            x,
            y,
            stepMode=True,
            fillLevel=0,
            brush=(150, 150, 255, 80),
            pen="w",
        )

    # ------------------------------------------------------------------ #
    # Public API: unified set_histogram
    # ------------------------------------------------------------------ #
    def set_histogram(
        self,
        data: Optional[Iterable[float]] = None,
        *,
        n_bins: Optional[int] = None,
        value_range: Optional[tuple[float, float]] = None,
        counts: Optional[Iterable[float]] = None,
        bin_edges: Optional[Iterable[float]] = None,
    ) -> None:
        """
        Set or compute and show a histogram.

        Two main usage patterns:

        1) From raw intensities (wizard):

            set_histogram(intensities, n_bins=256, value_range=(vmin, vmax))

        2) From precomputed histogram:

            set_histogram(counts=counts, bin_edges=bin_edges)
        """
        # Case 1: direct counts / edges supplied
        if counts is not None and bin_edges is not None:
            c = np.asarray(counts)
            e = np.asarray(bin_edges)
            self._plot_histogram(c, e)
            return

        # Case 2: compute histogram from data
        if data is None:
            self.clear()
            return

        vals = np.asarray(data, dtype=float).ravel()
        if vals.size == 0:
            self.clear()
            return

        if n_bins is None:
            n_bins = 256

        if value_range is None:
            vmin = float(vals.min())
            vmax = float(vals.max())
        else:
            vmin, vmax = value_range

        hist, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
        self._plot_histogram(hist, edges)

    # ------------------------------------------------------------------ #
    # Convenience functions for volume data
    # ------------------------------------------------------------------ #
    def update_from_volume(
        self,
        volume: np.ndarray,
        num_bins: int = 256,
        vmin: float = 0.0,
        vmax: float = 255.0,
    ) -> None:
        """
        Compute and display a histogram from a raw or RGB volume.

        volume can be:
          - (Z, Y, X) grayscale
          - (Z, Y, X, 3) RGB
        """
        if volume is None:
            self.clear()
            return

        arr = np.asarray(volume)

        if arr.ndim == 4 and arr.shape[-1] == 3:
            # Convert RGB to grayscale for histogram
            vals = arr.mean(axis=-1).ravel()
        else:
            vals = arr.ravel()

        self.set_histogram(vals, n_bins=num_bins, value_range=(vmin, vmax))

    # Alias for older call sites
    def update_histogram(
        self,
        volume: np.ndarray,
        num_bins: int = 256,
        vmin: float = 0.0,
        vmax: float = 255.0,
    ) -> None:
        self.update_from_volume(volume, num_bins=num_bins, vmin=vmin, vmax=vmax)
