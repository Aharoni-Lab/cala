from dataclasses import dataclass

import numpy as np
import xarray as xr

from cala.streaming.core import Parameters, Axis
from cala.streaming.nodes import Node
from cala.streaming.stores.common import Footprints, Traces


@dataclass
class DupeSnifferParams(Parameters, Axis):
    merge_threshold: float  # this should get later replaced by confidence level

    def validate(self) -> None:
        assert 0 <= self.merge_threshold <= 1, "merge_threshold must be between 0 and 1."


@dataclass
class DupeSniffer(Node):
    params: DupeSnifferParams

    def process(
        self, new_fp, new_tr, existing_fp, existing_tr, residuals
    ) -> dict[str, float] | None:
        """
        determines whether the new component overlaps with an existing component.
        if novel, return None.
        if similar to existing components (above threshold), return the component IDs.

        :param new_fp:
        :param new_tr:
        :param existing_fp:
        :param existing_tr:
        :param residuals:
        :return:
        """

        overlapping_components = self._find_overlap_ids(new_fp, existing_fp)

        if not overlapping_components:
            return True

        overlapped_traces = self._get_overlapped_traces(overlapping_components, existing_tr)

        synced_traces = self._get_synced_traces(new_tr, overlapped_traces)

        if synced_traces:
            return synced_traces

        return True

    def _find_overlap_ids(
        self, new_footprints: Footprints, existing_footprints: Footprints
    ) -> np.ndarray:
        if existing_footprints.size == 0:
            return np.array([])

        overlaps = (new_footprints @ existing_footprints) > 0
        overlaps: xr.DataArray[bool]

        return overlaps.where(overlaps, drop=True).coords[self.params.id_coordinates].values

    def _get_overlapped_traces(
        self, overlapping_components: np.ndarray, existing_tr: xr.DataArray
    ) -> xr.DataArray:
        return (
            existing_tr.set_xindex(self.params.id_coordinates)
            .sel({Axis.id_coordinates: overlapping_components})
            .reset_index(Axis.id_coordinates)
        )

    def _get_synced_traces(
        self,
        new_trace: Traces,
        existing_traces: Traces,
    ) -> dict[str, float]:
        """Validate new component against spatial and temporal criteria."""

        relevant_traces = existing_traces.isel(
            {self.params.frames_axis: slice(-new_trace.sizes[self.params.frames_axis], None)}
        )
        # For components with spatial overlap, check temporal correlation
        temporal_corr = xr.corr(
            new_trace,
            relevant_traces,
            dim=self.params.frames_axis,
        )

        dupes = temporal_corr.where(temporal_corr >= self.params.merge_threshold, drop=True)
        dupe_ids = dupes.coords[self.params.id_coordinates].values.tolist()
        dupe_scores = dupes.values.tolist()

        return dict(zip(dupe_ids, dupe_scores))
