from typing import List

import networkx as nx
import pandas as pd
from sklearn.neighbors import KDTree


class Grapher:
    """
    Constructs the pixel graph where nodes are pixels and edges connect neighboring pixels.
    """

    def __init__(self, core_axes: List[str], neighborhood_radius: int):
        self.core_axes = core_axes
        self.neighborhood_radius = neighborhood_radius

    def build_pixel_graph(self, nodes: pd.DataFrame) -> nx.Graph:
        """
        Build a graph where nodes are pixels and edges connect neighboring pixels.
        """
        # Identify seed pixels
        seeds = nodes[nodes["index"].notnull()].copy()
        seed_indices = seeds.index.values

        # Build KDTree for neighbor search
        nn_tree = KDTree(nodes[self.core_axes].values)
        nearest_neighbors = nn_tree.query_radius(
            seeds[self.core_axes].values, r=self.neighborhood_radius
        )

        # Initialize graph
        seed_graph = nx.Graph()
        node_attrs = nodes[self.core_axes + ["index"]].to_dict("index")
        seed_graph.add_nodes_from(node_attrs.items())

        # Add edges between seeds and their neighbors
        for seed_idx, neighbors in zip(seed_indices, nearest_neighbors):
            edges = [
                (seed_idx, neighbor) for neighbor in neighbors if neighbor != seed_idx
            ]
            seed_graph.add_edges_from(edges)

        # Remove isolated nodes and relabel nodes to integers
        seed_graph.remove_nodes_from(list(nx.isolates(seed_graph)))
        seed_graph = nx.convert_node_labels_to_integers(seed_graph)

        return seed_graph

    def adjacent_seeds(self, graph: nx.Graph) -> List[List[int]]:
        """
        Generate adjacency list representation from graph.
        """
        return [list(graph.neighbors(node)) for node in sorted(graph.nodes())]
