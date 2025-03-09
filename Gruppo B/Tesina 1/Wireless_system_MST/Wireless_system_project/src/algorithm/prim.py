from src.models.network import WirelessNetwork
from typing import List, Tuple, Dict, Optional
import heapq

class PrimMST:
    """Prim's MST implementation."""

    def __init__(self, network: WirelessNetwork, calculate_edge_cost: callable):
        """Initialize with network instance and edge cost calculation function."""
        self.network = network
        self.calculate_edge_cost = calculate_edge_cost

    def find_mst(self) -> List[Tuple[int, int]]:
        """Calculate the Minimum Spanning Tree (MST) using Prim's algorithm."""
        start_node = next(iter(self.network.nodes))
        visited = set()
        mst_edges = []
        edge_heap = []

        def add_edges(node_id):
            visited.add(node_id)
            for neighbor_id in self.network.graph.neighbors(node_id):
                if neighbor_id not in visited:
                    cost = self.calculate_edge_cost(self.network.nodes[node_id], self.network.nodes[neighbor_id])
                    heapq.heappush(edge_heap, (cost, node_id, neighbor_id))

        add_edges(start_node)

        while edge_heap and len(mst_edges) < len(self.network.nodes) - 1:
            cost, node1_id, node2_id = heapq.heappop(edge_heap)
            if node2_id not in visited:
                mst_edges.append((node1_id, node2_id))
                add_edges(node2_id)

        return mst_edges