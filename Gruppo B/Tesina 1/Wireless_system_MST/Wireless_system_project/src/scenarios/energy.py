"""
Energy optimization scenario implementation.
Goal: Minimize power consumption while maintaining connectivity.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork
import networkx as nx
from src.models.node import Node
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST


class EnergyMST:
    """MST implementation optimized for energy zones."""

    def __init__(self, network: WirelessNetwork):
        """Initialize with network."""
        self.network = network
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def _make_set(self, v: int) -> None:
        """Initialize disjoint set for vertex."""
        self.parent[v] = v
        self.rank[v] = 0

    def _find(self, v: int) -> int:
        """Find set representative with path compression."""
        if self.parent[v] != v:
            self.parent[v] = self._find(self.parent[v])
        return self.parent[v]

    def _union(self, v1: int, v2: int) -> None:
        """Union sets by rank."""
        root1 = self._find(v1)
        root2 = self._find(v2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def _calculate_edge_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate the cost of an edge between two nodes considering various factors:
                # - Minimize total power consumption
                # - Balance load across nodes
                # - Ensure sufficient power capacity
                # - Optimize transmission distances
        """
        base_cost = node1.get_link_cost(node2)
        
        # Power requirement factor
        power_requirement = node1.get_power_requirement(node2)
        power_factor = 1.0 + (power_requirement / node1.power_capacity)

        cost = base_cost * power_factor 
        return cost

def calculate_mst_metrics(network: WirelessNetwork, 
                         mst_edges: List[Tuple[int, int]]) -> Dict:
    """Calculate metrics for MST solution."""
    total_distance = 0.0
    max_elevation_diff = 0.0
    total_elevation_change = 0.0
    max_edge_cost = 0.0
    power_capacity = 0.0
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Distance
        distance = node1.distance_to(node2)
        total_distance += distance
        
        # Elevation
        elev_diff = abs(node1.elevation - node2.elevation)
        max_elevation_diff = max(max_elevation_diff, elev_diff)
        total_elevation_change += elev_diff
        
        # Cost
        edge_data = network.graph.get_edge_data(edge[0], edge[1])
        if edge_data:
            max_edge_cost = max(max_edge_cost, edge_data['weight'])

        # Power Capacity
        power_capacity += node1.power_capacity

        # Calculate betweenness centrality
        subgraph = network.graph.edge_subgraph(mst_edges).copy()
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        
        # Extract the node with maximum betweenness centrality
        max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
            
    return {
        'total_distance': total_distance,
        'max_elevation_diff': max_elevation_diff,
        'avg_elevation_change': total_elevation_change / len(mst_edges),
        'max_edge_cost': max_edge_cost,
        'power_capacity': power_capacity,
        'betweenness_centrality': max_betweenness_node
    }

def solve_energy_scenario(network: WirelessNetwork,
                         algorithm: str = 'kruskal',
                         constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal MST considering power consumption.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - max_power_per_node: Maximum power per node
                    - total_power_budget: Total network power budget
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """
    # TODO: Student Implementation
    
    # 1. Apply power-based cost adjustments
    # - Consider power requirements for connections
    # - Account for node power capacities
    # - Factor in distance-based power needs
    
    # 2. Initialize MST algorithm
    # if algorithm == 'kruskal':
    #     mst_solver = KruskalMST(network)
    # else:
    #     mst_solver = PrimMST(network)
    
    # 3. Consider:
    # - Minimize total power consumption
    # - Balance load across nodes
    # - Ensure sufficient power capacity
    # - Optimize transmission distances
    
    # 4. Find and validate MST solution
    # mst_edges = mst_solver.find_mst()
    # if validate_energy_solution(network, mst_edges, constraints):
    #     return mst_edges
     # Adjust edge weights based on seismic risk
    for edge in network.graph.edges():
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]

        energy_cost = EnergyMST(network)._calculate_edge_cost(node1, node2)

        network.graph.edges[edge]['weight'] = energy_cost

    # Find MST
    sismic_mst = EnergyMST(network)
    if algorithm == 'kruskal':
        mst_solver = KruskalMST(sismic_mst)
    elif algorithm == 'prim':
        mst_solver = PrimMST(sismic_mst)
    else:
        raise NotImplementedError("Only Kruskal's and Prim's algorithms are implemented")

    mst_edges = mst_solver.find_mst()

    # Validate solution
    if mst_edges and validate_energy_solution(network, mst_edges, constraints):
        # Calculate and log metrics
        metrics = calculate_mst_metrics(network, mst_edges)
        print(f"\Energy MST Metrics:")
        print(f"Total Distance: {metrics['total_distance']:.2f}")
        print(f"Max Elevation Difference: {metrics['max_elevation_diff']:.2f}m")
        print(f"Average Elevation Change: {metrics['avg_elevation_change']:.2f}m")
        print(f"Maximum Edge Cost: {metrics['max_edge_cost']:.2f}")
        print(f"Betweeness Centrality: {metrics['betweenness_centrality']}")
        print(f"Power Capacity: {metrics['power_capacity']:.2f}")

        return mst_edges

    return None
    

def validate_energy_solution(network: WirelessNetwork,
                           mst_edges: List[Tuple[int, int]],
                           constraints: Dict) -> bool:
    """Validate MST solution for energy optimization scenario."""
    if not mst_edges:
        return False
        
    max_power_per_node = constraints.get('max_power_per_node', 100.0)
    total_power_budget = constraints.get('total_power_budget', float('inf'))
    
    # Calculate power requirements
    total_power = 0.0
    node_power = {node_id: 0.0 for node_id in network.nodes}
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        power_req = node1.get_power_requirement(node2)
        total_power += power_req
        
        # Add power requirements to both nodes
        node_power[edge[0]] += power_req / 2
        node_power[edge[1]] += power_req / 2
        
    # Check constraints
    if total_power > total_power_budget:
        return False
        
    for power in node_power.values():
        if power > max_power_per_node:
            return False
    
    return True