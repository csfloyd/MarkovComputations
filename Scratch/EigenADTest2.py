import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr
import jax
import jax.numpy as jnp
import jax.experimental as jexp
from jax.experimental import sparse as jexps
import networkx as nx
from math import exp
from functools import partial

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

def get_steady_state(A):
    """
    Solve for steady state using LSMR algorithm.
    Equivalent to Julia's LinearSolve with KrylovJL_LSMR solver.
    """
    n = A.shape[0]
    # Add constraint row for probability conservation
    A_augmented = np.vstack([A, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1  # Last element is 1 for probability conservation
    
    # Solve using LSMR
    x, status = lsmr(A_augmented, b)
    return x

# # Make a JAX-compatible version for automatic differentiation
# @partial(jax.jit, static_argnums=(1,))
# def get_steady_state_jax(A, n):
#     """JAX version of get_steady_state for automatic differentiation"""
#     #A_augmented = jnp.vstack([A, jnp.ones((1, n))])
#     #b = jnp.zeros(n + 1).at[-1].set(1)
    
#     # Use JAX's linear solve
#     # x = jax.scipy.linalg.solve(A_augmented.T @ A_augmented, 
#     #                           A_augmented.T @ b,
#     #                           assume_a='pos')

#     # Use JAX's CG solver
#     x, *_ = jax.scipy.sparse.linalg.cg(A_augmented.T @ A_augmented, A_augmented.T @ b, tol = 1e-2, maxiter = 10)

#     # Use JAX's CG solver
#     # x, *_ = jax.scipy.sparse.linalg.bicgstab(A_augmented.T @ A_augmented, A_augmented.T @ b, tol = 1e-4, maxiter = 100)
    
#     #return x[:n]
#     return x[2:10]


@partial(jax.jit, static_argnums=(1,))
def get_steady_state_jax(A, n):
    #A_augmented = jnp.vstack([A, jnp.ones((1, n))])
    x, *_ = jax.scipy.sparse.linalg.bicgstab(A.T @ A, A.T @ b, tol=1e-4, maxiter=1000)
    return x



# Main execution
if __name__ == "__main__":
    print("New start")

    # Initialize parameters
    n_nodes = 100
    n_edges = round(n_nodes / 2)
    n = n_nodes
    e_range = 1
    b_range = 1
    f_range = 1

      # # Create sparse transition rate matrix
    # W = np.zeros((n, n))
    # for j in range(n):
    #     for i in range(n):
    #         if i != j:
    #             W[i, j] = np.random.random()
    #     W[j, j] = -np.sum(W[:, j])  # Ensure column sum is zero

    # Create random graph
    # Note: NetworkX's density-based random graph is different from Julia's SimpleGraph
    # We'll use Erdős-Rényi random graph with approximately the desired number of edges
    probability = (2 * n_edges) / (n_nodes * (n_nodes - 1))  # Factor of 2 because edges are undirected
    g = nx.erdos_renyi_graph(n_nodes, probability, seed=None)

    # Get edges (ensuring we have exactly n_edges)
    graph_edges = list(g.edges())[:n_edges]

    # Generate random parameters
    ej_list, bij_list, fij_list = random_initial_parameters(e_range, b_range, f_range, n_nodes, n_edges)

    # Calculate weight matrix
    wij_mat = update_weight_matrix(graph_edges, ej_list, bij_list, fij_list)
    A_augmented = jnp.vstack([wij_mat, jnp.ones((1, n))])
    
    # Convert to dense matrix for JAX compatibility
    A = A_augmented
    b2 = np.ones(n) / n
    b = jnp.zeros(n + 1).at[-1].set(1)
    
    # Calculate steady state
    ss = get_steady_state_jax(A, n)
    print(f"Steady state: {ss}")
    print(f"Sum of steady state: {np.sum(ss[:n])}")  # Sum only the state probabilities
    
    # Calculate Jacobian using JAX
    jacobian_fn = jax.jacfwd(lambda A: get_steady_state_jax(A, n))
    dA = jacobian_fn(A)
    
    #print(f"Jacobian shape: {dA.shape}")
    #print("\nJacobian matrix:")
    #print(dA)
    print("done")