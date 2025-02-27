from typing import Dict, Optional, Set

import numpy as np
import zarr
from scipy import sparse

from .base import FluorescentObject


class Neuron(FluorescentObject):
    """Class representing a detected neuron in calcium imaging."""

    def __init__(
        self,
        footprint: np.ndarray | sparse.csr_matrix,
        time_trace: np.ndarray | zarr.Array,
        confidence_level: Optional[float] = 0.0,
        overlapping_objects: Optional[Set["FluorescentObject"]] = None,
        deconvolved_signal: Optional[np.ndarray] = None,
        spike_times: Optional[np.ndarray] = None,
        cell_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
        rise_time_constant: Optional[float] = None,
        decay_time_constant: Optional[float] = None,
    ):
        """Initialize a Neuron.

        Args:
            footprint: 2D array representing spatial distribution. Will be converted to sparse matrix.
            time_trace: 1D array of fluorescence intensity. Will be converted to zarr array.
            confidence_level: Confidence in object detection/separation.
            overlapping_objects: Set of objects with overlapping footprints.
            deconvolved_signal: Deconvolved neural activity.
            spike_times: Estimated spike times.
            cell_type: Identified cell type if available.
            metadata: Additional metadata.
            rise_time_constant: Tau rise in seconds.
            decay_time_constant: Tau decay in seconds.
        """
        super().__init__(footprint, time_trace, confidence_level, overlapping_objects)
        self._deconvolved_signal = deconvolved_signal
        self._spike_times = spike_times
        self._cell_type = cell_type
        self._metadata = metadata or {}
        self._rise_time_constant = rise_time_constant
        self._decay_time_constant = decay_time_constant

    @property
    def deconvolved_signal(self) -> Optional[np.ndarray]:
        """Get the deconvolved signal."""
        return self._deconvolved_signal

    @deconvolved_signal.setter
    def deconvolved_signal(self, value: Optional[np.ndarray]) -> None:
        """Set the deconvolved signal."""
        self._deconvolved_signal = value

    @property
    def spike_times(self) -> Optional[np.ndarray]:
        """Get the spike times."""
        return self._spike_times

    @spike_times.setter
    def spike_times(self, value: Optional[np.ndarray]) -> None:
        """Set the spike times."""
        self._spike_times = value

    @property
    def metadata(self) -> Dict:
        """Get the metadata dictionary."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict) -> None:
        """Set the metadata dictionary."""
        self._metadata = value

    def deconvolve_signal(self, method: str = "default") -> np.ndarray:
        """Deconvolve calcium signal to estimate neural activity."""
        raise NotImplementedError

    def detect_spikes(self, threshold: float = 2.0) -> np.ndarray:
        """Detect spike times from deconvolved signal."""
        raise NotImplementedError

    def classify_cell_type(self) -> str:
        """Classify neuron type based on its properties."""
        raise NotImplementedError
