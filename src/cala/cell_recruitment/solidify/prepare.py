from typing import List

import numpy as np
import pandas as pd
import xarray as xr


class Preparer:
    """
    Handles data preparation tasks such as merging seeds with pixel coordinates.
    """

    def __init__(self, core_axes: List[str]):
        self.core_axes = core_axes

    def prepare_node_dataframe(
        self, movie: xr.DataArray, seeds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare the node DataFrame by merging all pixel coordinates with seeds.
        """
        # Extract coordinate values
        coord_values = [movie.coords[dim].values for dim in self.core_axes]
        mesh = np.array(np.meshgrid(*coord_values, indexing="ij"))
        all_coords = mesh.reshape(len(self.core_axes), -1).T
        nodes = pd.DataFrame(all_coords, columns=self.core_axes)

        # Merge with seeds to identify seed pixels
        seeds_reset = seeds.reset_index()
        nodes = nodes.merge(seeds_reset, how="left", on=self.core_axes)
        return nodes
