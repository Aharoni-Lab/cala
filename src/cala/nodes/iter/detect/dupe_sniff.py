from operator import itemgetter

import numpy as np
import xarray as xr
from noob.node import Node

from cala.stores.common import Footprints, Traces


class DupeSniffer(Node):
    merge_threshold: float  # this should get later replaced by confidence level

    def process(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        existing_fp: xr.DataArray,
        existing_tr: xr.DataArray,
        residuals: xr.DataArray,
    ) -> list[tuple[str, float]] | None:
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
            return None

        overlapped_traces = self._get_overlapped_traces(overlapping_components, existing_tr)

        synced_traces = self._get_synced_traces(new_tr, overlapped_traces)

        if synced_traces:
            return synced_traces

        return None

    def _find_overlap_ids(
        self, new_footprints: Footprints, existing_footprints: Footprints
    ) -> np.ndarray:
        if existing_footprints.size == 0:
            return np.array([])

        overlaps = (new_footprints @ existing_footprints) > 0
        overlaps: xr.DataArray[bool]

        return overlaps.where(overlaps, drop=True).coords[self.params.id_coord].values

    def _get_overlapped_traces(
        self, overlapping_components: np.ndarray, existing_tr: xr.DataArray
    ) -> xr.DataArray:
        return (
            existing_tr.set_xindex(self.params.id_coord)
            .sel({self.params.id_coord: overlapping_components})
            .reset_index(self.params.id_coord)
        )

    def _get_synced_traces(
        self,
        new_trace: Traces,
        existing_traces: Traces,
    ) -> list[tuple[str, float]]:
        """Validate new component against spatial and temporal criteria."""

        relevant_traces = existing_traces.isel(
            {self.params.frames_dim: slice(-new_trace.sizes[self.params.frames_dim], None)}
        )
        # For components with spatial overlap, check temporal correlation
        temporal_corr = xr.corr(
            new_trace,
            relevant_traces,
            dim=self.params.frames_dim,
        )

        dupes = temporal_corr.where(temporal_corr >= self.params.merge_threshold, drop=True)
        dupe_ids = dupes.coords[self.params.id_coord].values.tolist()
        dupe_scores = dupes.values.tolist()

        return sorted(list(zip(dupe_ids, dupe_scores)), key=itemgetter(1))
