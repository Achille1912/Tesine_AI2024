"""
Seismic zone scenario implementation.
Goal: Ensure network resilience in seismic areas.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork
from src.models.node import Node
import networkx as nx
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST


def calculate_edge_cost(node1: Node, node2: Node) -> float:
    """
    Calculate the cost of an edge between two nodes considering various factors:
    1. Base distance
    2. Seismic vulnerability
    3. Terrain stability
    4. Network redundancy requirements
    """

    base_cost = node1.get_link_cost(node2)
    # Seismic vulnerability factor (higher = riskier)
    vulnerability_score = node1.get_vulnerability_score(node2)
    vulnerability_factor = 1.0 + (vulnerability_score * 2)  # Penalizza connessioni tra nodi vulnerabili

    # Redundancy factor (if a node is critical, ensure redundancy)
    redundancy_factor = 2 if vulnerability_score > 0.7 else 1.5

    return base_cost * vulnerability_factor * redundancy_factor
    

def calculate_mst_metrics(network: WirelessNetwork, 
                         mst_edges: List[Tuple[int, int]]) -> Dict:
    """Calculate metrics for MST solution."""
    total_distance = 0.0
    max_elevation_diff = 0.0
    total_elevation_change = 0.0
    max_edge_cost = 0.0
    max_vulnerability_score = 0.0
    
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
        
        # Vulnerability Score
        vulnerability_score = node1.get_vulnerability_score(node2)
        max_vulnerability_score = max(vulnerability_score, max_vulnerability_score)

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
        'vulnerability_score': max_vulnerability_score,
        'betweenness_centrality': max_betweenness_node
    }

def solve_seismic_scenario(network: WirelessNetwork, algorithm: str = 'kruskal', constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal MST considering seismic risks.

    Args:
        network: The wireless network
        constraints: Dictionary of constraints including:
                    - max_vulnerability: Maximum allowed vulnerability score
                    - min_redundancy: Minimum redundancy level

    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """

    # TODO: Student Implementation

    # 1. Apply vulnerability-based cost adjustments
    # - Consider node vulnerability scores
    # - Account for terrain stability
    # - Factor in redundancy requirements

    # 2. Initialize MST algorithm
    # if algorithm == 'kruskal':
    #     mst_solver = KruskalMST(network)
    # else:
    #     mst_solver = PrimMST(network)

    # 3. Consider:
    # - Minimize vulnerability scores
    # - Ensure redundant paths where needed
    # - Account for seismic zone characteristics
    # - Balance redundancy with cost

    # 4. Find and validate MST solution
    # mst_edges = mst_solver.find_mst()
    # if validate_seismic_solution(network, mst_edges, constraints):
    #     return mst_edges


    # Adjust edge weights based on seismic risk
    for edge in network.graph.edges():
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]

        seismic_cost = calculate_edge_cost(node1, node2)
        network.graph.edges[edge]['weight'] = seismic_cost

    # Find MST
    if algorithm == 'kruskal':
        mst_solver = KruskalMST(network)
    elif algorithm == 'prim':
        mst_solver = PrimMST(network, calculate_edge_cost)
    else:
        raise NotImplementedError("Only Kruskal's and Prim's algorithms are implemented")

    mst_edges = mst_solver.find_mst()

    # Validate solution
    if mst_edges and validate_seismic_solution(network, mst_edges, constraints):
        # Calculate and log metrics
        metrics = calculate_mst_metrics(network, mst_edges)
        print(f"\Seismic MST Metrics:")
        print(f"Total Distance: {metrics['total_distance']:.2f}")
        print(f"Max Elevation Difference: {metrics['max_elevation_diff']:.2f}m")
        print(f"Average Elevation Change: {metrics['avg_elevation_change']:.2f}m")
        print(f"Maximum Edge Cost: {metrics['max_edge_cost']:.2f}")
        print(f"Betweeness Centrality: {metrics['betweenness_centrality']}")
        print(f"Vulnerability Score: {metrics['vulnerability_score']:.2f}")

        return mst_edges

    return None


def validate_seismic_solution(network: WirelessNetwork,
                            mst_edges: List[Tuple[int, int]],
                            constraints: Dict) -> bool:
    """Validate MST solution for seismic scenario."""
    if not mst_edges:
        return False

    max_vulnerability = constraints.get('max_vulnerability', 0.7)
    min_redundancy = constraints.get('min_redundancy', 1.5)  ##
    redundancy_factor = constraints.get('redundancy_factor', 2.0)
    
    # Check vulnerability constraints
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        if node1.get_vulnerability_score(node2) > 1.5: # max_vulnerability:
            print(node1.get_vulnerability_score(node2))
            print(node1)
            print(node2)
            return False
    
    # Check redundancy requirements
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        vulnerability_score = node1.get_vulnerability_score(node2)
        red_factor = 2 if vulnerability_score > 0.7 else 1.5
        if red_factor < min_redundancy or red_factor > redundancy_factor:
            return False
    

    
    return True