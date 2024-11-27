from typing import Optional, List, Dict, Any, Tuple

import dask.array as da
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
import pymetis
import xarray as xr

from ...utilities import frequency_filter


class Correlator:
    """
    Computes correlations based on the graph.
    """

    def __init__(
        self,
        smoothing_frequency: Optional[float],
        core_axes: List[str],
        iter_axis: str,
        chunk_size: int = 600,
        step_size: int = 50,
    ):
        self.smoothing_frequency = smoothing_frequency
        self.core_axes = core_axes
        self.iter_axis = iter_axis
        self.chunk_size = chunk_size
        self.step_size = step_size

    def compute_correlations(self, varr: xr.DataArray, graph: nx.Graph) -> pd.DataFrame:
        """
        Compute correlations in an optimized fashion given a computation graph.
        """
        print("Computing correlations")
        num_partitions = max(int(np.ceil(graph.number_of_nodes() / self.chunk_size)), 1)
        _, membership = pymetis.part_graph(
            num_partitions, adjacency=self.adjacent_seeds(graph)
        )
        partition_map = {
            node: part for node, part in zip(sorted(graph.nodes), membership)
        }
        nx.set_node_attributes(graph, partition_map, "part")

        # Convert graph to edge DataFrame
        edge_data = nx.to_pandas_edgelist(graph)
        edge_data["source_group"] = edge_data["source"].map(partition_map)
        edge_data["target_group"] = edge_data["target"].map(partition_map)
        edge_data["same_group"] = edge_data["source_group"] == edge_data["target_group"]

        idx_dict = {dim: nx.get_node_attributes(graph, dim) for dim in self.core_axes}
        corr_list = []
        index_list = []

        # Process edges within the same partition
        same_part_edges = edge_data[edge_data["same_group"]]
        for part, edges in same_part_edges.groupby("source_group"):
            pixels = np.unique(edges[["source", "target"]].values)
            corr_chunk, idx_chunk = self._compute_correlations_chunk(
                varr, idx_dict, edges, pixels
            )
            corr_list.append(corr_chunk)
            index_list.append(idx_chunk)

        # Process edges across different partitions
        diff_part_edges = edge_data[~edge_data["same_group"]]
        for idx in range(0, len(diff_part_edges), self.step_size):
            edges = diff_part_edges.iloc[idx : idx + self.step_size]
            pixels = np.unique(edges[["source", "target"]].values)
            corr_chunk, idx_chunk = self._compute_correlations_chunk(
                varr, idx_dict, edges, pixels
            )
            corr_list.append(corr_chunk)
            index_list.append(idx_chunk)

        # Compute correlations
        corr_values = da.concatenate(corr_list).compute()
        corr_indices = np.concatenate(index_list)
        corr_series = pd.Series(corr_values, index=corr_indices, name="corr")

        # Add correlations to DataFrame
        edge_data.loc[corr_series.index, "corr"] = corr_series.values
        return edge_data[["source", "target", "corr"]]

    def _compute_correlations_chunk(
        self,
        varr: xr.DataArray,
        idx_dict: Dict[str, Dict[Any, Any]],
        edges: pd.DataFrame,
        pixels: np.ndarray,
    ) -> Tuple[da.Array, np.ndarray]:
        """
        Compute correlations for a chunk of edges.
        """
        # Map pixel indices to positions
        pixel_map = {pixel: idx for idx, pixel in enumerate(pixels)}
        row_idx = edges["source"].map(pixel_map).values
        col_idx = edges["target"].map(pixel_map).values

        # Build indexers for varr
        idx_selectors = {
            dim: [idx_dict[dim][pixel] for pixel in pixels] for dim in self.core_axes
        }
        vsub = varr.sel({dim: idx_selectors[dim] for dim in self.core_axes}).data
        vsub = vsub.transpose(..., self.iter_axis)
        vsub = vsub.chunk({self.iter_axis: -1})

        # Compute correlations
        corr_chunk = smooth_and_partial_correlation(
            vsub, row_idx, col_idx, cutoff_frequency=self.smoothing_frequency
        )
        return corr_chunk, edges.index.values

    @staticmethod
    def adjacent_seeds(graph: nx.Graph) -> List[List[int]]:
        """
        Generate adjacency list representation from graph.
        """
        return [list(graph.neighbors(node)) for node in sorted(graph.nodes())]


@da.as_gufunc(signature="(n,p),(k),(k)->(k)", output_dtypes=float)
def smooth_and_partial_correlation(
    X: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    cutoff_frequency: Optional[float],
) -> np.ndarray:
    """
    Smooth the time series and compute partial pairwise correlations.
    """
    if cutoff_frequency:
        X = frequency_filter(X, cutoff_frequency=cutoff_frequency, filter_pass="low")
    return idx_corr(X, row_idx, col_idx)


@nb.njit(cache=True, parallel=True)
def idx_corr(X: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    """
    Compute partial pairwise correlations based on indices.
    """
    n_pixels, n_frames = X.shape
    n_pairs = row_idx.size
    corr = np.zeros(n_pairs)

    # Center and normalize X
    for i in nb.prange(n_pixels):
        Xi = X[i] - X[i].mean()
        norm = np.sqrt((Xi**2).sum())
        if norm > 0:
            X[i] = Xi / norm
        else:
            X[i] = 0

    # Compute correlations
    for idx in nb.prange(n_pairs):
        i = row_idx[idx]
        j = col_idx[idx]
        corr[idx] = np.dot(X[i], X[j])

    return corr
