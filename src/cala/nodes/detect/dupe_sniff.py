from operator import itemgetter
from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node

from cala.assets import Footprint, Footprints, Residual, Trace, Traces
from cala.models import AXIS


class DupeSniffer(Node):
    merge_threshold: float  # this should get later replaced by confidence level

    def process(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints,
        existing_tr: Traces,
        residuals: Residual,
    ) -> A[list[tuple[str, float]] | None, Name("dupes")]:
        """
        determines whether the new component overlaps with an existing component.
        if novel, return None.
        if similar to existing components (above threshold), return the component IDs.
        """

        overlapping_components = self._find_overlap_ids(new_fp.array, existing_fp.array)

        if not overlapping_components:
            return None

        overlapped_traces = self._get_overlapped_traces(overlapping_components, existing_tr.array)

        synced_traces = self._get_synced_traces(new_tr.array, overlapped_traces)

        if synced_traces:
            return synced_traces

        return None

    def _find_overlap_ids(
        self, new_footprints: xr.DataArray, existing_footprints: xr.DataArray
    ) -> np.ndarray:
        if existing_footprints.size == 0:
            return np.array([])

        overlaps = (new_footprints @ existing_footprints) > 0
        overlaps: xr.DataArray[bool]

        return overlaps.where(overlaps, drop=True).coords[AXIS.id_coord].values

    def _get_overlapped_traces(
        self, overlapping_components: np.ndarray, existing_tr: xr.DataArray
    ) -> xr.DataArray:
        return (
            existing_tr.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: overlapping_components})
            .reset_index(AXIS.id_coord)
        )

    def _get_synced_traces(
        self, new_trace: xr.DataArray, existing_traces: xr.DataArray
    ) -> list[tuple[str, float]]:
        """Validate new component against spatial and temporal criteria."""

        relevant_traces = existing_traces.isel(
            {AXIS.frames_dim: slice(-new_trace.sizes[AXIS.frames_dim], None)}
        )
        # For components with spatial overlap, check temporal correlation
        temporal_corr = xr.corr(new_trace, relevant_traces, dim=AXIS.frames_dim)

        dupes = temporal_corr.where(temporal_corr >= self.merge_threshold, drop=True)
        dupe_ids = dupes.coords[AXIS.id_coord].values.tolist()
        dupe_scores = dupes.values.tolist()

        return sorted(list(zip(dupe_ids, dupe_scores)), key=itemgetter(1))
