from src.models.network import WirelessNetwork
from typing import List, Tuple, Optional, Dict, Set


class KruskalMST:
    """Kruskal's MST implementation."""

    def __init__(self, mst):
        """Initialize with network and MountainMST instance."""
        self.network = mst.network
        self.mst = mst

    def find_mst(self) -> List[Tuple[int, int]]:
        """Calculate the Minimum Spanning Tree (MST) using Kruskal's algorithm."""
        edges = []

        # Extract edges from the network
        for node1_id in self.network.nodes:
            for node2_id in self.network.nodes:
                if node1_id < node2_id:  # Avoid duplicates
                    node1 = self.network.nodes[node1_id]
                    node2 = self.network.nodes[node2_id]
                    cost = self.mst._calculate_edge_cost(node1, node2)
                    edges.append((node1_id, node2_id, cost))

        # Sort edges by cost
        edges.sort(key=lambda x: x[2])

        # Initialize disjoint sets
        for v in self.network.nodes:
            self.mst._make_set(v)

        # Build MST
        mst_edges = []
        for v1, v2, cost in edges:
            if self.mst._find(v1) != self.mst._find(v2):
                self.mst._union(v1, v2)
                mst_edges.append((v1, v2))

        return mst_edges