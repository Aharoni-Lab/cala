from typing import List

import dask.array as da
import pandas as pd
import sparse
import xarray as xr


class Footprinter:
    """
    Constructs spatial footprints from the correlations.
    """

    def __init__(self, core_axes: List[str], correlation_threshold: float):
        self.core_axes = core_axes
        self.correlation_threshold = correlation_threshold

    def construct_spatial_footprints(
        self, movie: xr.DataArray, nodes: pd.DataFrame, correlations: pd.DataFrame
    ) -> xr.DataArray:
        """
        Construct spatial footprints for each seed based on correlations.
        """
        print("Constructing spatial footprints")

        # Filter correlations
        correlations = correlations[correlations["corr"] > self.correlation_threshold]

        # Reconstruct node DataFrame after relabeling
        seeds = nodes[nodes["index"].notnull()].astype({"index": int})

        # Create coordinate to index mappings
        coord_maps = {
            dim: {v: i for i, v in enumerate(movie.coords[dim].values)}
            for dim in self.core_axes
        }

        footprint_shape = tuple(movie.sizes[dim] for dim in self.core_axes)
        footprints = []

        # Build mapping from node IDs to coordinates
        node_id_to_coords = nodes[self.core_axes].to_dict("index")

        # Construct spatial footprints for each seed
        for seed_id, seed_data in seeds.iterrows():
            # Get connected nodes
            connected_nodes = self._get_connected_nodes(seed_id, correlations, nodes)

            # Map coordinates to array indices
            for dim in self.core_axes:
                connected_nodes[f"i{dim}"] = connected_nodes[dim].map(coord_maps[dim])

            # Get correlation values
            corr_values = connected_nodes["corr"].values

            # Create sparse array for the spatial footprint
            coords = tuple(connected_nodes[f"i{dim}"].values for dim in self.core_axes)
            current_footprint = sparse.COO(coords, corr_values, shape=footprint_shape)
            footprints.append(current_footprint)

        # Stack footprints into a DataArray
        footprint_data = da.stack(
            [
                da.from_array(footprint.todense(), chunks=footprint_shape)
                for footprint in footprints
            ]
        )
        footprint_da = xr.DataArray(
            footprint_data,
            dims=["unit_id"] + self.core_axes,
            coords={
                "unit_id": seeds["index"].values,
                **{dim: movie.coords[dim].values for dim in self.core_axes},
            },
        )
        return footprint_da

    def _get_connected_nodes(
        self, seed_id: int, correlations: pd.DataFrame, nodes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retrieve all nodes connected to a seed along with their correlations.
        """
        # Extract correlations where the seed is involved
        seed_correlations = correlations[
            (correlations["source"] == seed_id) | (correlations["target"] == seed_id)
        ].copy()

        # Map node indices to coordinates
        node_indices = pd.concat(
            [seed_correlations["source"], seed_correlations["target"]]
        ).unique()
        connected_nodes = nodes.loc[node_indices].copy()

        # Add correlation values
        connected_nodes["corr"] = seed_correlations["corr"].values

        # Ensure the seed itself is included with correlation 1
        if seed_id not in connected_nodes.index:
            seed_coords = nodes.loc[seed_id, self.core_axes]
            seed_series = pd.Series(
                {**seed_coords.to_dict(), "corr": 1.0}, name=seed_id
            )
            connected_nodes = connected_nodes.append(seed_series)

        return connected_nodes
