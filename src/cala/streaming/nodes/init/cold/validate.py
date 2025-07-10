from dataclasses import dataclass

import numpy as np
import xarray as xr

from cala.streaming.core import Parameters, Axis
from cala.streaming.nodes import Node
from cala.streaming.stores.common import Footprints, Traces


@dataclass
class ValidateParams(Parameters, Axis):
    spatial_threshold: float
    temporal_threshold: float


@dataclass
class Validate(Node):
    params: ValidateParams

    def _validate_new_component(
        self,
        new_footprint: Footprints,
        new_trace: Traces,
        footprints: Footprints,
        traces: Traces,
        residuals: xr.DataArray,
    ) -> bool:
        """Validate new component against spatial and temporal criteria."""

        # for the next round of discovery, we do need to register and subtract
        # so this is what we do
        # we grab everything. r_spatial < thresh, include it
        # we however mark it "temporary"
        # after we exhaust the entire frame, we do a review
        # is it a cell, background, or just a plain wrong estimator - how do we distinguish these?
        # estimator_confidence, and neuron_confidence
        # estimator_confidence - how much of it can be explained by others
        # this can be remedied in the merge / split step? first it gets split, then it gets merged
        # neuron_confidence - how likely is it a neuron

        # we need a few different mechanisms - register, merge, toss
        # register - it's a brand-new thing.
        #       * isolated footprint, any trace (could be correlated cells)
        #       * overlapping footprint, unique trace
        # merge - overlapping footprint, similar trace (more similar than confidence level)
        #       * what if overlapping & similar trace with more than one cell
        #           * start by overlapping with the highest - then measure the confidence of the larger two,
        #           * worry about merging the large two then

        # not sure if this step is necessary or even makes sense
        # how would a rank-1 nmf be not similar to the mean, unless the nmf error was massive?
        # and if the error is massive, maybe it just means it's overlapping with another luminescent object?
        # instead of tossing, we do candidates - cell, background, UNKNOWN
        # we gather everything. we merge everything as much as possible. and then we decide what to do.
        valid_spatial = self._check_spatial_validity(
            new_footprint=new_footprint, residuals=residuals
        )

        if footprints.size == 0:
            return True

        # Check for duplicates by computing spatial overlap with existing footprints
        overlapping_components = self._find_overlap_ids(new_footprint, footprints)

        if not overlapping_components.any():
            return True

        # instead of tossing when duplicate, we should merge
        synced_components = self._get_synced_traces(
            suspect_ids=overlapping_components,
            traces=traces,
            new_traces=new_trace,
            residual_length=residuals.sizes[self.params.frames_axis],
        )

        if not synced_components.any():
            return True

        # begin merging
        a_new, c_new = self._deconstruct_new_components(
            new_footprint, new_trace, synced_components, traces, footprints
        )

        return False

    def _deconstruct_new_components(
        self, new_footprint, new_trace, synced_components, traces, footprints
    ) -> xr.DataArray:
        """Deconstruct new components."""
        synced_traces = traces.set_xindex(self.params.id_coordinates).sel(
            {
                self.params.id_coordinates: synced_components.coords[
                    self.params.id_coordinates
                ].values
            }
        )
        synced_footprints = footprints.set_xindex(self.params.id_coordinates).sel(
            {
                self.params.id_coordinates: synced_components.coords[
                    self.params.id_coordinates
                ].values
            }
        )

    def _find_overlap_ids(self, new_footprints: Footprints, original_footprints: Footprints):
        overlaps = (new_footprints @ original_footprints) > 0

        return overlaps.where(overlaps, drop=True).coords[self.params.id_coordinates].values

    def _check_spatial_validity(self, new_footprint: xr.DataArray, residuals: xr.DataArray) -> bool:
        nonzero_ax_to_idx = {
            ax: sorted([int(x) for x in set(idx)])
            for ax, idx in zip(new_footprint.dims, new_footprint.values.nonzero())
        }  # nonzero coordinates, like [[0, 1, 0, 1], [0, 0, 1, 1]] for [0, 0], [0, 1], [1, 0], [1, 1]

        if len(list(nonzero_ax_to_idx.values())[0]) == 0:
            return False

        # it should look like something from a residual. paper does not specify this,
        # but i think we should only get correlation from the new footprint perimeter,
        # since otherwise the correlation will get drowned out by the mismatch
        # from where the detected cell is NOT present.
        mean_residual = residuals.mean(dim=self.params.frames_axis)

        # is this step necessary? what exact cases would this filter out?
        # if the trace is similar enough, we should accept it regardless. - right? what's the downside here
        # it doesn't look like the mean residual - but has a trace that looks like one of the og.
        # hmm?
        # if the trace is not similar, only THEN we check if it looks like the residual.
        a_norm = new_footprint.isel(nonzero_ax_to_idx) / new_footprint.sum()
        res_norm = mean_residual.isel(nonzero_ax_to_idx) / mean_residual.sum()

        # instead of rejecting, we should attach this value to the component
        r_spatial = xr.corr(a_norm, res_norm) if np.abs(a_norm - res_norm).max() > 1e-7 else 1.0

        if r_spatial <= self.params.spatial_threshold:
            return False
        return True

    def _get_synced_traces(
        self, suspect_ids: xr.DataArray, traces: Traces, residual_length: int, new_traces: Traces
    ) -> xr.DataArray:
        relevant_traces = (
            traces.set_xindex(self.params.id_coordinates)
            .sel(
                {
                    self.params.id_coordinates: suspect_ids,
                }
            )
            .isel({self.params.frames_axis: slice(-residual_length, None)})
        )
        # For components with spatial overlap, check temporal correlation
        temporal_corr = xr.corr(
            new_traces,
            relevant_traces,
            dim=self.params.frames_axis,
        )

        # --> subtract trace from new, merge footprint into original
        return temporal_corr.where(temporal_corr > self.params.temporal_threshold, drop=True)
