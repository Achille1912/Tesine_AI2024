from src.models.network import WirelessNetwork
from typing import List, Tuple, Optional, Dict, Set

class KruskalMST:
    """Kruskal's MST implementation."""

    def __init__(self, network: WirelessNetwork):
        """Initialize with network instance."""
        self.network = network
        self.parent = {}
        self.rank = {}

    def _make_set(self, node_id: int):
        """Initialize disjoint set for a node."""
        self.parent[node_id] = node_id
        self.rank[node_id] = 0

    def _find(self, node_id: int) -> int:
        """Find the representative of the set containing node_id."""
        if self.parent[node_id] != node_id:
            self.parent[node_id] = self._find(self.parent[node_id])
        return self.parent[node_id]

    def _union(self, node1_id: int, node2_id: int):
        """Union the sets containing node1_id and node2_id."""
        root1 = self._find(node1_id)
        root2 = self._find(node2_id)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

    def find_mst(self) -> List[Tuple[int, int]]:
        """Calculate the Minimum Spanning Tree (MST) using Kruskal's algorithm."""
        edges = []

        # Extract edges from the network
        for node1_id, node2_id, data in self.network.graph.edges(data=True):
            cost = data['weight']
            edges.append((node1_id, node2_id, cost))

        # Sort edges by cost
        edges.sort(key=lambda x: x[2])

        # Initialize disjoint sets
        for v in self.network.nodes:
            self._make_set(v)

        # Build MST
        mst_edges = []
        for v1, v2, cost in edges:
            if self._find(v1) != self._find(v2):
                self._union(v1, v2)
                mst_edges.append((v1, v2))

        return mst_edges