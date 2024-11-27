from dataclasses import dataclass, field
from typing import Optional, List

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import radius_neighbors_graph

from demixing.correlate import Correlator


@dataclass
class Merger(BaseEstimator, TransformerMixin):
    max_projection_frame: xr.DataArray  # motion-corrected, float type, .max(iter_axis)
    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    iter_axis: str = "frames"
    distance_threshold: int = 5
    correlation_threshold: float = 0.6
    smoothing_frequency: Optional[float] = None

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return self.seeds_merge(varr=X, seeds=y)

    def seeds_merge(
        self,
        varr: xr.DataArray,
        seeds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge seeds based on spatial distance and temporal correlation of their
        activities.

        This function build an adjacency matrix by thresholding spatial distance
        between seeds and temporal correlation between activities of seeds. It then
        merge seeds using the adjacency matrix by only keeping the seed with maximum
        intensity in the max projection within each connected group of seeds. The
        merge is therefore transitive.

        Parameters
        ----------
        varr : xr.DataArray
            Input movie data. Should have dimension "height", "width" and "frame".
        seeds : pd.DataFrame
            Dataframe of seeds to be merged.

        Returns
        -------
        seeds : pd.DataFrame
            The resulting seeds dataframe with an additional column "mask_mrg",
            indicating whether the seed should be kept after the merge. If the
            column already exists in input `seeds` it will be overwritten.
        """
        neighbor_graph = radius_neighbors_graph(
            seeds[self.core_axes], self.distance_threshold
        )
        adjacency_matrix = self.adj_corr(varr, neighbor_graph, seeds[self.core_axes])
        adjacency_matrix = adjacency_matrix > self.correlation_threshold
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        labels = self.label_connected(adjacency_matrix, only_connected=True)
        iso = np.where(labels < 0)[0]
        seeds_final = set(iso.tolist())
        for cur_cmp in np.unique(labels):
            if cur_cmp < 0:
                continue
            cur_smp = np.where(labels == cur_cmp)[0]
            cur_max = np.array(
                [
                    self.max_projection_frame.sel(
                        {
                            axis_name: seeds.iloc[s][axis_name]
                            for axis_name in self.core_axes
                        }
                    )
                    for s in cur_smp
                ]
            )
            max_seed = cur_smp[np.argmax(cur_max)]
            seeds_final.add(max_seed)
        seeds["merge_mask"] = False
        seeds.loc[list(seeds_final), "merge_mask"] = True
        return seeds

    def adj_corr(
        self,
        movie: xr.DataArray,
        adjacency_matrix: np.ndarray,
        nodes: pd.DataFrame,
    ) -> sp.sparse.csr_matrix:
        """
        Compute correlation in an optimized fashion given an adjacency matrix and
        node attributes.

        Wraps around :func:`graph_optimize_corr` and construct computation graph
        from `adj` and `nod_df`. Also convert the result into a sparse matrix with
        same shape as `adj`.

        Parameters
        ----------
        movie : xr.DataArray
            Input time series. Should have "frame" dimension in addition to column
            names of `nod_df`.
        adjacency_matrix : np.ndarray
            Adjacency matrix.
        nodes : pd.DataFrame
            Dataframe containing node attributes. Should have length `adj.shape[0]`
            and only contain columns relevant to index the time series.
        freq : float
            Cut-off frequency for the optional smoothing. If `None` then no
            smoothing will be done.

        Returns
        -------
        adj_corr : scipy.sparse.csr_matrix
            Sparse matrix of the same shape as `adj` but with values corresponding
            the computed correlation.
        """
        graph = nx.Graph()
        graph.add_nodes_from(
            [(idx, node) for idx, node in enumerate(nodes.to_dict("records"))]
        )
        graph.add_edges_from(
            [(source, target) for source, target in zip(*adjacency_matrix.nonzero())]
        )

        correlator = Correlator(
            smoothing_frequency=self.smoothing_frequency,
            core_axes=self.core_axes,
            iter_axis=self.iter_axis,
        )
        correlations = correlator.compute_correlations(varr=movie, graph=graph)
        return sp.sparse.csr_matrix(
            (correlations["corr"], (correlations["source"], correlations["target"])),
            shape=adjacency_matrix.shape,
        )

    @staticmethod
    def label_connected(
        adjacency_matrix: np.ndarray, only_connected=False
    ) -> np.ndarray:
        """
        Label connected components given adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Adjacency matrix. Should be 2d symmetric matrix.
        only_connected : bool, optional
            Whether to keep only the labels of connected components. If `True`, then
            all components with only one node (isolated) will have their labels set
            to -1. Otherwise, all components will have unique label. By default,
            `False`.

        Returns
        -------
        labels : np.ndarray
            The labels for each components. Should have length `adj.shape[0]`.
        """
        try:
            np.fill_diagonal(adjacency_matrix, 0)
            adjacency_matrix = np.triu(adjacency_matrix)
            graph = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        except Exception:
            graph = nx.convert_matrix.from_scipy_sparse_array(adjacency_matrix)
        labels = np.zeros(adjacency_matrix.shape[0], dtype=np.int32)
        for idx, component in enumerate(nx.connected_components(graph)):
            component = list(component)
            if only_connected and len(component) == 1:
                labels[component] = -1
            else:
                labels[component] = idx
        return labels
