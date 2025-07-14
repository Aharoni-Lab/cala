from dataclasses import dataclass

import numpy as np
import xarray as xr

from cala.streaming.core import Parameters, Axis
from cala.streaming.nodes import Node
from cala.streaming.stores.common import Footprints, Traces


@dataclass
class CatalogerParams(Parameters, Axis):
    merge_threshold: float

    def validate(self) -> None: ...


@dataclass
class Cataloger(Node):
    params: CatalogerParams

    def process(self, new_fp, new_tr, fp_inventory, tr_inventory, residuals) -> str | bool:
        """
        determines whether the new component overlaps with an existing component.
        if not valid at all, return False
        if novel, return True
        if similar to existing components (above threshold), return the component IDs.

        :param new_fp:
        :param new_tr:
        :param fp_inventory:
        :param tr_inventory:
        :param residuals:
        :return:
        """
        s_validity = self._validate_spatial(new_fp, residuals)
        if not s_validity >= self.params.spatial_threshold:
            return False

        overlapping_cells = self._find_overlap_ids(new_fp, fp_inventory)

        t_validity = self._validate_temporal(new_tr, tr_inventory, overlapping_cells)
        if not t_validity >= self.params.merge_threshold:
            return True

        return True

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
        return temporal_corr.where(temporal_corr > self.params.merge_threshold, drop=True)
