import numpy as np
import networkx as nx
from math import exp

def random_initial_parameters(e_range, b_range, f_range, n_nodes, n_ed):
    """
    Generate random initial parameters for a network with given ranges and dimensions.
    Returns tuple of (ej_list, bij_list, fij_list) containing random values within specified ranges.
    """
    ej_list = (2 * np.random.random(n_nodes) - 1) * e_range
    bij_list = (2 * np.random.random(n_edges) - 1) * b_range
    fij_list = (2 * np.random.random(n_edges) - 1) * f_range
    
    return (ej_list, bij_list, fij_list)

def update_weight_matrix(graph_edges, ej_list, bij_list, fij_list):
    """
    Update the weight matrix of a network based on given parameters and edge structure.
    
    Parameters:
    - graph_edges: List of edge pairs [(source, target), ...]
    - ej_list: Array of node parameters
    - bij_list: Array of edge bias parameters
    - fij_list: Array of edge flow parameters
    
    Returns:
    - numpy array containing the weight matrix
    """
    n_nodes = len(ej_list)
    wij_mat = np.zeros((n_nodes, n_nodes))
    
    # Populate the weight matrix using given formulas
    for e, (n1, n2) in enumerate(graph_edges):
        bij = bij_list[e]
        fij = fij_list[e]
        ei = ej_list[n1]
        ej = ej_list[n2]
        
        # Calculate weights
        wij = exp(-bij + ej + fij/2)
        wji = exp(-bij + ei - fij/2)
        
        # Update matrix elements
        wij_mat[n1, n2] = wij
        wij_mat[n2, n1] = wji
    
    # Adjust the diagonal elements of the weight matrix
    for i in range(n_nodes):
        wij_mat[i, i] = -np.sum(wij_mat[:, i])
    
    return wij_mat

# Main execution
#if __name__ == "__main__":
print("new.")

# Initialize parameters
n_edges = 5
n_nodes = 10
e_range = 1
b_range = 1
f_range = 1

# Create random graph
# Note: NetworkX's density-based random graph is different from Julia's SimpleGraph
# We'll use Erdős-Rényi random graph with approximately the desired number of edges
probability = (2 * n_edges) / (n_nodes * (n_nodes - 1))  # Factor of 2 because edges are undirected
g = nx.erdos_renyi_graph(n_nodes, probability, seed=None)

# Get edges (ensuring we have exactly n_edges)
graph_edges = list(g.edges())[:n_edges]

# Get adjacency matrix
adj_mat = nx.adjacency_matrix(g).toarray()

# Generate random parameters
ej_list, bij_list, fij_list = random_initial_parameters(e_range, b_range, f_range, n_nodes, n_edges)

# Calculate weight matrix
wij_mat = update_weight_matrix(graph_edges, ej_list, bij_list, fij_list)

# Display results
print("Weight matrix:")
print(wij_mat)

print("done.")