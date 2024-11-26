from dataclasses import dataclass
from typing import Optional

import pandas as pd
import xarray as xr

from .base import BaseDemixer
from .prepare import Preparer
from .graph import Grapher
from .correlate import Correlator
from .footprint import Footprinter


@dataclass
class SpatialUpdater(BaseDemixer):
    correlation_threshold: float = 0.8
    neighborhood_radius: int = 10
    smoothing_frequency: Optional[float] = None
    movie_: xr.DataArray = None
    seeds_: pd.DataFrame = None
    _is_fitted: bool = False

    def __post_init__(self):
        self.preparer = Preparer(core_axes=self.core_axes)
        self.grapher = Grapher(
            core_axes=self.core_axes, neighborhood_radius=self.neighborhood_radius
        )
        self.correlator = Correlator(
            smoothing_frequency=self.smoothing_frequency,
            core_axes=self.core_axes,
            iter_axis=self.iter_axis,
        )
        self.footprinter = Footprinter(
            core_axes=self.core_axes,
            correlation_threshold=self.correlation_threshold,
        )

    def fit_kernel(self, X: xr.DataArray, y):
        pass

    def fit(self, X: xr.DataArray, y: pd.DataFrame):
        """
        Fit the estimator to the data.
        """
        if not isinstance(X, xr.DataArray):
            raise TypeError("X must be an xarray.DataArray")
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame")
        self.movie_ = X
        self.seeds_ = y
        self._is_fitted = True
        return self

    def transform_kernel(self, X: xr.DataArray, y):
        pass

    def transform(self, X: xr.DataArray = None, y: pd.DataFrame = None) -> xr.DataArray:
        """
        Transform the input data to initialize spatial footprints.
        """
        if not self._is_fitted:
            raise RuntimeError("You must fit the estimator before transforming data.")

        movie = X if X is not None else self.movie_
        seeds = y if y is not None else self.seeds_

        if not isinstance(movie, xr.DataArray):
            raise TypeError("Movie must be an xarray.DataArray")
        if not isinstance(seeds, pd.DataFrame):
            raise TypeError("Seeds must be a pandas DataFrame")

        # Step 1: Prepare data
        nodes = self.preparer.prepare_node_dataframe(movie, seeds)

        # Step 2: Build graph of pixels and neighbors
        seeds_graph = self.grapher.build_pixel_graph(nodes)

        # Step 3: Compute correlations
        correlations = self.correlator.compute_correlations(movie, seeds_graph)

        # Step 4: Construct spatial footprints
        footprints = self.footprinter.construct_spatial_footprints(
            movie, nodes, correlations
        )

        return footprints
