"""
Mountain scenario implementation.
Goal: Optimize network topology in mountainous terrain.
"""

from typing import List, Tuple, Optional, Dict, Set
import networkx as nx
import numpy as np
from src.models.network import WirelessNetwork
from src.models.node import Node
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST

def calculate_edge_cost(node1: Node, node2: Node) -> float:
    """
    Calculate edge cost considering mountainous terrain.
            
    Factors:
    1. Base distance
    2. Elevation difference
    3. Terrain difficulty
    4. Weather impact (higher elevations = higher cost)
    """
    base_cost = node1.get_link_cost(node2)
            
    # Weather impact factor (higher elevation = worse weather)
    avg_elevation = (node1.elevation + node2.elevation) / 2
    weather_factor = 1.0 + (avg_elevation / 1000.0) ** 2
                    
    return base_cost * weather_factor


def calculate_mst_metrics(network: WirelessNetwork, 
                         mst_edges: List[Tuple[int, int]]) -> Dict:
    """Calculate metrics for MST solution."""
    total_distance = 0.0
    max_elevation_diff = 0.0
    total_elevation_change = 0.0
    max_edge_cost = 0.0
    
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
        'betweenness_centrality': max_betweenness_node
    }

def solve_mountain_scenario(network: WirelessNetwork,
                          algorithm: str = 'prim',
                          constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal MST considering mountainous terrain.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - max_link_distance: Maximum allowed link distance
                    - elevation_factor: Impact of elevation differences
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """

    # Apply terrain-based cost adjustments
    for edge in network.graph.edges():
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Update edge weight with terrain-aware cost
        # elevation_diff = abs(node1.elevation - node2.elevation)
        # terrain_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        # weather_factor = 1.0 + (max(node1.elevation, node2.elevation) / 1000.0)
        
        #base_weight = network.graph.edges[edge]['weight']
        # adjusted_weight = base_weight * (1 + elevation_diff/100) * terrain_factor * weather_factor
        adjusted_weight = calculate_edge_cost(node1, node2)
        network.graph.edges[edge]['weight'] = adjusted_weight
    
    # Find MST
    #mountain_mst = MountainMST(network)
    
    if algorithm == 'kruskal':
        mst_solver = KruskalMST(network)
    elif algorithm == 'prim':
        mst_solver = PrimMST(network, calculate_edge_cost)
    else:
        raise NotImplementedError("Only Kruskal's and Prim's algorithms are implemented")

    mst_edges = mst_solver.find_mst()
    
    # Validate solution
    if mst_edges and validate_mountain_solution(network, mst_edges, constraints):
        # Calculate and log metrics
        metrics = calculate_mst_metrics(network, mst_edges)
        print(f"\nMountain MST Metrics:")
        print(f"Total Distance: {metrics['total_distance']:.2f}m")
        print(f"Max Elevation Difference: {metrics['max_elevation_diff']:.2f}m")
        print(f"Average Elevation Change: {metrics['avg_elevation_change']:.2f}m")
        print(f"Maximum Edge Cost: {metrics['max_edge_cost']:.2f}")
        print(f"Betweeness Centrality: Node {metrics['betweenness_centrality']}")

        
        return mst_edges
        
    return None

def validate_mountain_solution(network: WirelessNetwork,
                             mst_edges: List[Tuple[int, int]],
                             constraints: Dict) -> bool:
    """
    Validate MST solution for mountain scenario.
    
    Args:
        network: The wireless network
        mst_edges: Proposed MST solution
        constraints: Scenario constraints
        
    Returns:
        bool: True if solution is valid
    """
    if not mst_edges:
        return False
        
    max_link_distance = constraints.get('max_link_distance', float('inf'))
    elevation_factor = constraints.get('elevation_factor', 1.5)
    max_elevation_diff = constraints.get('max_elevation_diff', float('inf'))
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Check distance constraint
        if node1.distance_to(node2) > max_link_distance:
            return False
            
        # Check elevation constraints
        elevation_diff = abs(node1.elevation - node2.elevation)
        if elevation_diff > max_elevation_diff:
            return False
            
        if elevation_diff * elevation_factor > max_link_distance:
            return False
            
        # Check terrain difficulty
        terrain_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        if terrain_factor * node1.distance_to(node2) > max_link_distance * 1.5:
            return False
    
    return True