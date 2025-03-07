from dataclasses import dataclass, field
from typing import Iterator, Tuple

import numpy as np
import xarray as xr

from cala.streaming.core.components.stores import BaseStore
from cala.streaming.types.types import Traces


@dataclass
class TraceStore(BaseStore):
    """Manages temporal traces for components.
    assign (.array) just refreshes the entire thing. (whole) ✅
    replace swaps in existing traces. must match length. (one ✅ & batch ✅)
    update lengthens by n frames (one ✅ & batch ✅)
    insert concatenates new components. (one ✅ & batch ✅)
    """

    frame_axis: str = "frames"
    """The axis of the frames."""

    _traces: Traces = field(default_factory=lambda: Traces(), repr=False)
    """The traces of the components. Shape: (n_components, n_frames)."""

    @property
    def array(self) -> Traces:
        """Returns the traces as an xarray DataArray."""
        return self._traces

    @array.setter
    def array(self, value: Traces) -> None:
        self._traces = Traces(value)

    @property
    def dimensions(self) -> Tuple[str, str]:
        """Returns the dimensions of the traces."""
        return (self.component_axis, self.frame_axis)

    def insert(self, trace: Traces) -> None:
        """Add a new trace for a component."""
        if not hasattr(self, "_traces"):
            self._traces = Traces[trace]
        else:
            self._traces = Traces(
                xr.concat([self._traces, trace], dim=self.component_axis)
            )

    def remove(self, component_id: int) -> None:
        """Remove a trace."""
        self._traces = self._traces.drop_sel({self.component_axis: component_id})

    def replace(self, traces: Traces) -> None:
        """Update existing trace(s)."""
        if traces.sizes[self.frame_axis] != self._traces.sizes[self.frame_axis]:
            raise ValueError("New trace length doesn't match existing trace")

        # Create a condition mask for the component we want to update
        condition = (
            self._traces[self.component_axis] == traces.coords[self.component_axis]
        )
        # Use where to update only the values for this component
        self._traces = Traces(xr.where(condition, traces, self._traces))

    def update(self, new_traces: Traces) -> None:
        """Append new frames of traces for all components."""
        if set(new_traces.dims) != set(self.dimensions):
            raise ValueError(f"Traces dimensions must be {self.dimensions}")

        # Create zero-filled array for all components
        all_traces = Traces(
            np.zeros(
                (
                    self._traces.sizes[self.component_axis],
                    new_traces.sizes[self.frame_axis],
                )
            ),
            coords={
                self.component_axis: self._traces.coords[self.component_axis],
                self.frame_axis: new_traces.coords[self.frame_axis],
            },
            dims=self.dimensions,
        )

        # Update values for components that have explicit traces
        input_ids = new_traces.coords[self.component_axis].values
        if len(input_ids) > 0:
            all_traces.loc[{self.component_axis: list(input_ids)}] = new_traces

        # Concatenate with existing traces
        self._traces = Traces(
            xr.concat([self._traces, all_traces], dim=self.frame_axis)
        )

    def get_batch(self, start_time: int, end_time: int) -> Traces:
        """Get a batch of time traces for all components."""
        return self._traces.sel({self.frame_axis: slice(start_time, end_time)})

    def iterate_batches(
        self, batch_size: int = 1000
    ) -> Iterator[Tuple[int, int, Traces]]:
        """Iterate over time traces in batches.

        Args:
            batch_size: Number of frames per batch.

        Returns:
            Iterator yielding tuples of (start_idx, end_idx, batch_data).
            Both start_idx and end_idx are inclusive.

        Note:
            The last batch may be smaller than batch_size.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        total_time = self._traces.sizes[self.frame_axis]
        if total_time == 0:
            return

        # Adjust batch_size to account for inclusive end
        adjusted_batch_size = batch_size - 1

        for start_idx in range(0, total_time, batch_size):
            # For each batch, end_idx is start_idx + (batch_size - 1) to make it inclusive
            # But don't exceed total_time - 1
            end_idx = min(start_idx + adjusted_batch_size, total_time - 1)
            yield start_idx, end_idx, self.get_batch(start_idx, end_idx)
