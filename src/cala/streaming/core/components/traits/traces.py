from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass
class TraceManager:
    """Manages temporal traces for components."""

    component_axis: str = "component"
    """The axis of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    _traces: xr.DataArray = field(init=False, repr=False)
    """The traces of the components. Shape: (n_components, n_frames)."""

    @property
    def traces(self) -> xr.DataArray:
        """Returns the traces as an xarray DataArray."""
        return self._traces

    @property
    def traces_dimensions(self) -> Tuple[str, str]:
        """Returns the dimensions of the traces."""
        return (self.component_axis, self.frame_axis)

    def initialize(self, component_ids: list[int]) -> None:
        """Initialize empty traces for components."""
        self._traces = xr.DataArray(
            np.zeros((len(component_ids), 0)),
            coords={
                self.component_axis: component_ids,
                self.frame_axis: np.arange(0),
            },
            dims=self.traces_dimensions,
        )

    def add_trace(
        self, component_id: int, trace: Optional[xr.DataArray] = None
    ) -> None:
        """Add a new trace for a component."""
        if trace is None:
            # Create empty trace matching existing dimensions
            trace = xr.DataArray(
                np.zeros((1, self._traces.sizes[self.frame_axis])),
                coords={
                    self.component_axis: [component_id],
                    self.frame_axis: self._traces.coords[self.frame_axis],
                },
                dims=self.traces_dimensions,
            )
        else:
            # Expand and assign coordinates
            trace = trace.expand_dims(self.component_axis).assign_coords(
                {self.component_axis: [component_id]}
            )

        if not hasattr(self, "_traces"):
            self._traces = trace
        else:
            self._traces = xr.concat([self._traces, trace], dim=self.component_axis)

    def remove_trace(self, component_id: int) -> None:
        """Remove a trace."""
        self._traces = self._traces.drop_sel({self.component_axis: component_id})

    def update_trace(self, component_id: int, trace: xr.DataArray) -> None:
        """Update an existing trace."""
        if trace.shape != self._traces.loc[{self.component_axis: component_id}].shape:
            raise ValueError("New trace shape doesn't match existing trace")
        self._traces.loc[{self.component_axis: component_id}] = trace

    def append_frames(self, new_traces: xr.DataArray) -> None:
        """Append new frames of traces for all components."""
        if set(new_traces.dims) != set(self.traces_dimensions):
            raise ValueError(f"Traces dimensions must be {self.traces_dimensions}")

        # Create zero-filled array for all components
        all_traces = xr.DataArray(
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
            dims=self.traces_dimensions,
        )

        # Update values for components that have explicit traces
        input_ids = new_traces.coords[self.component_axis].values
        if len(input_ids) > 0:
            all_traces.loc[{self.component_axis: list(input_ids)}] = new_traces

        # Concatenate with existing traces
        self._traces = xr.concat([self._traces, all_traces], dim=self.frame_axis)

    def get_batch(self, start_time: int, end_time: int) -> xr.DataArray:
        """Get a batch of time traces for all components."""
        return self._traces.sel({self.frame_axis: slice(start_time, end_time)})

    def iterate_batches(
        self, batch_size: int = 1000
    ) -> Iterator[Tuple[int, int, xr.DataArray]]:
        """Iterate over time traces in batches."""
        total_time = self._traces.sizes[self.frame_axis]

        for start_idx in range(0, total_time, batch_size):
            end_idx = min(start_idx + batch_size, total_time)
            yield start_idx, end_idx, self.get_batch(start_idx, end_idx)
