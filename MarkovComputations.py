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
import timeit
import random
import copy
from sklearn import datasets
from sklearn.datasets import fetch_openml
import torch


class WeightMatrix:
    """
    Represents a weight matrix for a network, computing transition probabilities
    based on node and edge parameters.
    """

    def __init__(self, g, Ej_list, Bij_list, Fij_list):
        """
        Initializes the weight matrix.

        Parameters:
        - g: NetworkX graph object
        - Ej_list: List of node parameters
        - Bij_list: List of edge bias parameters
        - Fij_list: List of edge flow parameters
        ...
        """
        self.g = g
        self.graph_edges = list(g.edges())
        self.n_nodes = len(g.nodes())
        self.n_edges = len(g.edges())
        self.edge_array = self._get_edge_array()
        self.Ej_list = np.array(Ej_list, dtype=np.float64)
        self.Bij_list = np.array(Bij_list, dtype=np.float64)
        self.Fij_list = np.array(Fij_list, dtype=np.float64)
        self.W_mat = self._create_W_mat()
        self.zero_array = jnp.zeros(self.n_nodes + 1).at[-1].set(1)

        # Derivative masks for efficient computation
        self.dWijdE_list = [np.array(self.edge_array[:, 1] == k, dtype=int) for k in range(self.n_nodes)]
        self.dWjidE_list = [np.array(self.edge_array[:, 0] == k, dtype=int) for k in range(self.n_nodes)]

    def _get_edge_array(self):
        """Returns an array representation of graph edges."""
        return np.array(self.graph_edges)

    def _create_W_mat(self):
        """Computes and returns the weight matrix based on current parameters."""
        i_inds, j_inds = self.edge_array[:, 0], self.edge_array[:, 1]
        Wij_list = np.exp(-self.Bij_list + self.Ej_list[j_inds] + self.Fij_list / 2)
        Wji_list = np.exp(-self.Bij_list + self.Ej_list[i_inds] - self.Fij_list / 2)

        W_mat = np.zeros((self.n_nodes, self.n_nodes))
        W_mat[i_inds, j_inds] = Wij_list
        W_mat[j_inds, i_inds] = Wji_list

        # Ensure row sums are zero
        np.fill_diagonal(W_mat, -np.sum(W_mat, axis=0))

        return W_mat

    @staticmethod
    @partial(jax.jit)
    def get_steady_state(A, zero_array):
        """Computes the steady-state distribution using conjugate gradient solver."""
        x, *_ = jax.scipy.sparse.linalg.cg(A.T @ A, A.T @ zero_array, tol=1e-8, maxiter=100000)
        return x

    def current_steady_state(self):
        """Returns the steady-state solution of the current weight matrix."""
        return self.get_steady_state(self.augmented_W_mat(), self.zero_array)

    def set_W_mat(self, Ej_list, Bij_list, Fij_list):
        """
        Updates the weight matrix based on new parameters.

        Parameters:
        - Ej_list: Updated node parameters
        - Bij_list: Updated edge bias parameters
        - Fij_list: Updated edge flow parameters
        """
        i_inds, j_inds = self.edge_array[:, 0], self.edge_array[:, 1]
        Wij_list = np.exp(-Bij_list + Ej_list[j_inds] + Fij_list / 2)
        Wji_list = np.exp(-Bij_list + Ej_list[i_inds] - Fij_list / 2)

        self.W_mat.fill(0.0)
        self.W_mat[i_inds, j_inds] = Wij_list
        self.W_mat[j_inds, i_inds] = Wji_list
        np.fill_diagonal(self.W_mat, -np.sum(self.W_mat, axis=0))

        self.Ej_list, self.Bij_list, self.Fij_list = Ej_list, Bij_list, Fij_list

    def apply_inputs(self, input_inds, inputs):
        """
        Applies external inputs to modify the weight matrix.

        Parameters:
        - input_inds: Indices of edges receiving inputs
        - inputs: Input values applied to selected edges
        """
        W_mat_mod = self.W_mat.copy()
        for m, input_ind in enumerate(input_inds):
            for sub_input_ind in input_ind:
                (i, j) = self.graph_edges[sub_input_ind]
                W_mat_mod[i, j] *= np.exp(inputs[m] / 2)
                W_mat_mod[j, i] *= np.exp(-inputs[m] / 2)
        
        for i in range(self.n_nodes):
            W_mat_mod[i, i] = 0.0
            W_mat_mod[i, i] = -np.sum(W_mat_mod[:, i])
        
        return W_mat_mod

    def get_Wijs(self):
        """Returns Wij and Wji parameters of own weight matrix."""
        Wij_list = np.array(self.W_mat[self.edge_array[:, 0], self.edge_array[:, 1]], dtype = np.float64)  # Vectorized access
        Wji_list = np.array(self.W_mat[self.edge_array[:, 1], self.edge_array[:, 0]], dtype = np.float64)  # Vectorized access
        return Wij_list, Wji_list  # Already NumPy arrays
    
    def get_external_Wijs(self, W_mat):
        """Returns Wij and Wji parameters of own external matrix."""
        dWij_list = np.array(W_mat[self.edge_array[:, 0], self.edge_array[:, 1]], dtype = np.float64)  # Vectorized access
        dWji_list = np.array(W_mat[self.edge_array[:, 1], self.edge_array[:, 0]], dtype = np.float64)  # Vectorized access
        return dWij_list, dWji_list  # Already NumPy arrays

    def get_dWijs(weight_matrix, dW_mat):
        """Returns dpi_m / dW_ij elements from a dW_mat object, accounting for W_ij elements appearing also on the diagonal."""
        src, tgt = weight_matrix.edge_array[:, 0], weight_matrix.edge_array[:, 1]
        dWij_list = dW_mat[:, src, tgt] - dW_mat[:, tgt, tgt]  # Vectorized access
        dWji_list = dW_mat[:, tgt, src] - dW_mat[:, src, src]  # Vectorized access
        return dWij_list, dWji_list  # Already NumPy arrays

    def augmented_W_mat(self):
        """Returns an augmented version of the weight matrix for steady-state calculations."""
        return jnp.vstack([self.W_mat, jnp.ones((1, self.n_nodes))])

    def compute_ss_on_inputs(self, input_inds, inputs):
        """Computes the steady-state distribution under given inputs."""
        input_W_mat = self.apply_inputs(input_inds, inputs)
        return self.get_steady_state(self.augment_external(input_W_mat), self.zero_array)

    def augmented_input_W_mat(self, input_inds, inputs):
        """Returns an augmented version of an own weight matrix."""
        return jnp.vstack([self.apply_inputs(input_inds, inputs), jnp.ones((1, self.n_nodes))])

    def augment_external(self, other_W_mat):
        """Returns an augmented version of an externally provided weight matrix."""
        return jnp.vstack([other_W_mat, jnp.ones((1, self.n_nodes))])

    def lower_output_energies(self, output_inds, fac):
        """Reduce the output energies by a given factor."""
        for output_ind in output_inds:
            self.Ej_list[output_ind] -= fac
        
        # Update weight matrix with modified energy values
        self.set_W_mat(self.Ej_list, self.Bij_list, self.Fij_list)

    def nudged_ss(self, nudge, input_inds, inputs, output_inds):
        """Compute the steady state with nudged outputs.

        Args:
            nudge: List of nudging factors applied to output indices.
            input_inds: Indices of input nodes.
            inputs: Input values for the corresponding input nodes.
            output_inds: Indices of output nodes to be nudged.

        Returns:
            Steady-state distribution after applying inputs and nudges.
        """
        W_mat_nudged = self.apply_inputs(input_inds, inputs)

        # Apply exponential scaling to selected output columns
        for i, output_ind in enumerate(output_inds):
            W_mat_nudged[:, output_ind] *= np.exp(-nudge[i])

        # Enforce row sum constraints
        for i in range(self.n_nodes):
            W_mat_nudged[i, i] = 0.0  # Ensure no self-connections
            W_mat_nudged[i, i] = -np.sum(W_mat_nudged[:, i])  # Maintain row balance

        return self.get_steady_state(self.augment_external(W_mat_nudged), self.zero_array)

    def compute_nudged_ss(self, input_data, class_number, input_inds, inputs, output_inds, delta_E):
        """Apply a nudge corresponding to the class label and return the new steady state."""
        nudge = delta_E * (2 * input_data.labels[class_number] - 1) 
        return self.nudged_ss(nudge, input_inds, inputs, output_inds)

    def derivatives_of_ss(self, input_inds, inputs):
        """Compute derivatives of the steady state with respect to energy, bias, and force parameters.

        Args:
            input_inds: Indices of input nodes.
            inputs: Input values for the corresponding input nodes.

        Returns:
            dEj_lists: Derivatives with respect to energy parameters (shape: n_nodes).
            dBij_lists: Derivatives with respect to bias parameters (shape: n_nodes, n_edges).
            dFij_lists: Derivatives with respect to force parameters (shape: n_nodes, n_edges).
        """
        # Compute the augmented weight matrix given inputs
        A = self.augmented_input_W_mat(input_inds, inputs)

        # Compute the Jacobian of the steady-state function with respect to A
        jacobian_fn = jax.jacrev(lambda A: WeightMatrix.get_steady_state(A, self.zero_array))
        dW_mat = jacobian_fn(A)

        # Extract forward and reverse transition weights
        Wijs, Wjis = self.get_external_Wijs(A)

        # Compute derivatives of transition weights
        dWijs_full, dWjis_full = self.get_dWijs(dW_mat)

        # Compute weighted derivatives
        dWijs_times_Wijs = dWijs_full * Wijs  # Shape: (n_nodes, n_edges)
        dWjis_times_Wjis = dWjis_full * Wjis  # Shape: (n_nodes, n_edges)

        # Compute derivatives for each parameter type
        dEj_lists = np.array([
            np.dot(dWijs_times_Wijs, self.dWijdE_list[k]) + np.dot(dWjis_times_Wjis, self.dWjidE_list[k])
            for k in range(self.n_nodes)
        ]).T  # Shape: (n_nodes, n_edges)

        dBij_lists = -dWijs_times_Wijs - dWjis_times_Wjis  # Bias derivatives
        dFij_lists = 0.5 * (dWijs_times_Wijs - dWjis_times_Wjis)  # Force derivatives

        return dEj_lists, dBij_lists, dFij_lists

    def update(self, input_inds, inputs, ss_frac, eta):
        """Update the weight matrix parameters based on the error, by computing the derivatives using autodiff."""

        dEj_lists, dBij_lists, dFij_lists = self.derivatives_of_ss(input_inds, inputs) # get the derivatives of the steady state

        incrEj_list = np.einsum('n,nj->j', ss_frac, dEj_lists) # multiply derivatives by ss_frac to get the increments
        incrBij_list = np.einsum('n,nk->k', ss_frac, dBij_lists)
        incrFij_list = np.einsum('n,nk->k', ss_frac, dFij_lists)

        self.set_W_mat( # update the parameters
            self.Ej_list + eta * incrEj_list, 
            self.Bij_list + eta * incrBij_list, 
            self.Fij_list + eta * incrFij_list)

    def compute_output(self, inputs, input_inds, output_inds):
        """
        Computes output probabilities for the given inputs and output indices.
        Assumes the steady-state values at output_inds are already normalized probabilities.
        """
        ss = self.compute_ss_on_inputs(input_inds, inputs)
        probs = np.array([ss[out] for out in output_inds])
        return probs
    
    


class StackedWeightMatrices:
    """
    Represents a weight matrix for a network, computing transition probabilities
    based on node and edge parameters.
    """

    def __init__(self, weight_matrix_list,
                external_dims, internal_dims, # external dims is [external_input_dim, external_output_dim], and similarly for input dims
                M_vals, A_fac, b_fac, rand_bool = True):
        """
        Initializes the weight matrix.

        Parameters:

        """
        self.weight_matrix_list = weight_matrix_list
        self.L = len(weight_matrix_list)
        self._set_input_output_inds(external_dims, internal_dims, M_vals, rand_bool)
        self._set_A_matrices(A_fac)
        self._set_b_vectors(b_fac)
        

    def _set_input_output_inds(self, external_dims, internal_dims, M_vals, rand_bool):

        ## check that lengths are correct
        assert(self.L == len(M_vals))
        assert(self.L-1 == len(internal_dims[0]))
        assert(self.L-1 == len(internal_dims[1]))

        self.external_input_inds = get_input_inds(self.weight_matrix_list[0].n_edges, external_dims[0], M_vals[0], rand_bool)
        self.external_output_inds = get_output_inds(self.weight_matrix_list[0].n_nodes, external_dims[1], rand_bool) 
        
        self.internal_input_inds = [get_input_inds(self.weight_matrix_list[l+1].n_edges, internal_dims[0][l], M_vals[l+1], rand_bool) for l in range(0, self.L-1)]
        self.internal_output_inds = [get_output_inds(self.weight_matrix_list[l].n_nodes, internal_dims[1][l], rand_bool) for l in range(0, self.L-1)]


    def _set_A_matrices(self, A_fac):

        A_matrices_list = []

        for l in range(self.L-1):
            if len(self.internal_output_inds[l]) == len(self.internal_input_inds[l]):
                A_matrices_list.append(A_fac * np.identity(len(self.internal_output_inds[l])))
            else:
                A_matrices_list.append(A_fac * np.random.rand(len(self.internal_input_inds[l]), len(self.internal_output_inds[l])))
        
        self.A_matrices_list = A_matrices_list


    def _set_b_vectors(self, b_fac):

        b_vectors_list = []

        for l in range(self.L-1):
            b_vectors_list.append(-0.5 * b_fac * np.ones(len(self.internal_input_inds[l])))
        
        self.b_vectors_list = b_vectors_list


    def compute_stacked_ss_on_inputs(self, inputs):

        ss_list = [self.weight_matrix_list[0].compute_ss_on_inputs(self.external_input_inds, inputs)]
        inputs_list = [inputs]

        for l in range(self.L-1):
            A = self.A_matrices_list[l]
            x = [ss_list[l][i] for i in self.internal_output_inds[l]]
            new_inputs = np.dot(A, x) + self.b_vectors_list[l]
            ss_list.append(self.weight_matrix_list[l+1].compute_ss_on_inputs(self.internal_input_inds[l], new_inputs))
            inputs_list.append(new_inputs)

        return ss_list, inputs_list
        

    def stacked_derivatives_of_ss(self, ss_list, inputs_list):

        full_input_inds = [(self.external_input_inds if l == 0 else self.internal_input_inds[l-1]) for l in range(self.L)]

        ## get dpi_l / dEi, dpi_l / dBij, dpi_l / dFij for each l
        dpil_dthetal_lists = [self.weight_matrix_list[l].derivatives_of_ss(full_input_inds[l], inputs_list[l]) for l in range(self.L)]

        ## get x at the internal steady states
        x_lists = [ss_list[l][np.array(self.internal_output_inds[l])]  for l in range(self.L-1)]
        
        dpil_dFIl_lists = []
        for l in range(self.L):
            n_nodes = self.weight_matrix_list[l].n_nodes
            full_inds = full_input_inds[l]
            dpil_dFIl = np.zeros((n_nodes, len(full_inds)))
            for m, inds in enumerate(full_inds):  # Iterate over columns
                dpil_dFIl[:, m] = np.sum(
                    [dpil_dthetal_lists[l][2][:, ind] for ind in inds], axis=0)
            dpil_dFIl_lists.append(dpil_dFIl)
        

        dFIl_dAl_lists = [] # L-1 of these
        dFIl_dbl_lists = [] # L-1 of these
        for l in range(1, self.L):
            dFIl_dAl = np.zeros((len(full_input_inds[l]), len(full_input_inds[l]), len(self.internal_output_inds[l-1])))
            for k in range(len(self.internal_output_inds[l-1])):
                dFIl_dAl[:,:,k] = np.identity(len(full_input_inds[l])) * x_lists[l-1][k]
            dFIl_dAl_lists.append(dFIl_dAl)
            dFIl_dbl_lists.append(np.identity(len(full_input_inds[l])))

        dpiL_dpiol_lists = [] # L-1 of these
        if self.L > 1: 
            dpiL_dpiol_lists.append(np.dot(dpil_dFIl_lists[self.L-1], self.A_matrices_list[self.L-2])) 
        for l in range(self.L-3, -1, -1):
            dpiol_dpiolm1 = np.dot(dpil_dFIl_lists[l+1][np.array(self.internal_output_inds[l+1])], self.A_matrices_list[l])
            dpiL_dpiol_lists.insert(0, np.dot(dpiL_dpiol_lists[0], dpiol_dpiolm1))

        
        dpiL_dthetal_lists = [] # L of these
        for l in range(self.L-1): # add last one outside the loop
            dpiL_dEil = np.dot(dpiL_dpiol_lists[l], dpil_dthetal_lists[l][0][np.array(self.internal_output_inds[l])])
            dpiL_dBijl = np.dot(dpiL_dpiol_lists[l], dpil_dthetal_lists[l][1][np.array(self.internal_output_inds[l])])
            dpiL_dFijl = np.dot(dpiL_dpiol_lists[l], dpil_dthetal_lists[l][2][np.array(self.internal_output_inds[l])])
            dpiL_dthetal_lists.append([dpiL_dEil, dpiL_dBijl, dpiL_dFijl])
        dpiL_dthetal_lists.append(dpil_dthetal_lists[self.L-1])


        dpiL_dAl_lists = [] # L-1 of these
        dpiL_dbl_lists = [] # L-1 of these
        for l in range(self.L-1):  
            if l < self.L-2:
                dpiL_dFIlp1 = np.dot(dpiL_dpiol_lists[l+1], dpil_dFIl_lists[l+1][np.array(self.internal_output_inds[l+1])])
            else: 
                dpiL_dFIlp1 = dpil_dFIl_lists[l+1]
            dFIlp1_dAl = np.dot(dpiL_dFIlp1, dFIl_dAl_lists[l])
            dFIlp1_dbl = np.dot(dpiL_dFIlp1, dFIl_dbl_lists[l])
            dpiL_dAl_lists.append(dFIlp1_dAl)
            dpiL_dbl_lists.append(dFIlp1_dbl)


        return dpiL_dthetal_lists, dpiL_dAl_lists, dpiL_dbl_lists

    
    def update(self, ss_list, inputs_list, class_number, eta):
        """Update the weight matrix parameters based on the error, by computing the derivatives using autodiff."""

        dpiL_dthetal_lists, dpiL_dAl_lists, dpiL_dbl_lists = self.stacked_derivatives_of_ss(ss_list, inputs_list) # get the derivatives of the steady state

        out_ind = self.external_output_inds[class_number]
        fac = 1 / (ss_list[self.L-1][out_ind])
        for l in range(self.L):
            dpiL_dthetal = dpiL_dthetal_lists[l]
            incrEj_list = fac * dpiL_dthetal[0][out_ind]
            incrBij_list = fac * dpiL_dthetal[1][out_ind]
            incrFij_list = fac * dpiL_dthetal[2][out_ind]

            self.weight_matrix_list[l].set_W_mat( # update the parameters
                self.weight_matrix_list[l].Ej_list + eta * incrEj_list, 
                self.weight_matrix_list[l].Bij_list + eta * incrBij_list, 
                self.weight_matrix_list[l].Fij_list + eta * incrFij_list)
            
        
        for l in range(self.L-1):
            incrAl = fac * dpiL_dAl_lists[l][out_ind]
            incrbl = fac * dpiL_dbl_lists[l][out_ind]

            self.A_matrices_list[l] += eta * incrAl
            self.b_vectors_list[l] += eta * incrbl

        
    def compute_output(self, inputs):
        """
        Computes output probabilities for the given inputs using the stacked Markov layers.
        Assumes the steady-state values at external_output_inds are already normalized probabilities.
        """
        ss_list, _ = self.compute_stacked_ss_on_inputs(inputs)
        probs = np.array([ss_list[-1][out] for out in self.external_output_inds])
        return probs

    def zero_gradients(self):
        """Initialize/zero accumulators for Markov gradients."""
        self.markov_grad_accum = {
            'theta': [[np.zeros_like(wm.Ej_list), np.zeros_like(wm.Bij_list), np.zeros_like(wm.Fij_list)] for wm in self.weight_matrix_list],
            'A': [np.zeros_like(A) for A in self.A_matrices_list],
            'b': [np.zeros_like(b) for b in self.b_vectors_list]
        }

    def compute_gradients_single(self, inputs, target, markov_ss_list=None, inputs_list=None):
        """Compute gradients for a single sample (target is class index)."""
        # Forward pass
        if markov_ss_list is None or inputs_list is None:
            markov_ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
        dpiL_dthetal_lists, dpiL_dAl_lists, dpiL_dbl_lists = self.stacked_derivatives_of_ss(markov_ss_list, inputs_list)
        out_ind = self.external_output_inds[target]
        fac = 1 / (markov_ss_list[self.L-1][out_ind])
        markov_gradients = {
            'theta': [],
            'A': [],
            'b': []
        }
        for l in range(self.L):
            dpiL_dthetal = dpiL_dthetal_lists[l]
            Ej_grad = fac * dpiL_dthetal[0][out_ind]
            Bij_grad = fac * dpiL_dthetal[1][out_ind]
            Fij_grad = fac * dpiL_dthetal[2][out_ind]
            markov_gradients['theta'].append([Ej_grad, Bij_grad, Fij_grad])
        for l in range(self.L-1):
            A_grad = fac * dpiL_dAl_lists[l][out_ind]
            b_grad = fac * dpiL_dbl_lists[l][out_ind]
            markov_gradients['A'].append(A_grad)
            markov_gradients['b'].append(b_grad)
        return markov_gradients

    def accumulate_gradients(self, markov_grads):
        """Add single-sample gradients to the running total."""
        for l in range(self.L):
            for i in range(3):
                self.markov_grad_accum['theta'][l][i] += markov_grads['theta'][l][i]
        for l in range(self.L - 1):
            self.markov_grad_accum['A'][l] += markov_grads['A'][l]
            self.markov_grad_accum['b'][l] += markov_grads['b'][l]

    def apply_gradients(self, batch_size, eta_markov):
        """Update parameters using the averaged gradients."""
        for l in range(self.L):
            Ej_grad, Bij_grad, Fij_grad = [g / batch_size for g in self.markov_grad_accum['theta'][l]]
            wm = self.weight_matrix_list[l]
            new_Ej = wm.Ej_list + eta_markov * Ej_grad
            new_Bij = wm.Bij_list + eta_markov * Bij_grad
            new_Fij = wm.Fij_list + eta_markov * Fij_grad
            wm.set_W_mat(new_Ej, new_Bij, new_Fij)
        for l in range(self.L - 1):
            self.A_matrices_list[l] += eta_markov * (self.markov_grad_accum['A'][l] / batch_size)
            self.b_vectors_list[l] += eta_markov * (self.markov_grad_accum['b'][l] / batch_size)

    def apply_adam_gradients(self, batch_size, eta_markov, beta1, beta2, epsilon):
        """
        Update parameters using the Adam optimizer with batch-averaged gradients.
        Maintains Adam state for each parameter.
        """
        # Initialize Adam state if not present
        if not hasattr(self, 'adam_state'):
            self.adam_state = {'t': 0, 'markov': {'theta': [], 'A': [], 'b': []}, 'perceptron': {}}
            # Markov parameters
            for l in range(self.L):
                wm = self.weight_matrix_list[l]
                self.adam_state['markov']['theta'].append([
                    {'m': np.zeros_like(wm.Ej_list), 'v': np.zeros_like(wm.Ej_list)},
                    {'m': np.zeros_like(wm.Bij_list), 'v': np.zeros_like(wm.Bij_list)},
                    {'m': np.zeros_like(wm.Fij_list), 'v': np.zeros_like(wm.Fij_list)}
                ])
            for l in range(self.L - 1):
                self.adam_state['markov']['A'].append({'m': np.zeros_like(self.A_matrices_list[l]), 'v': np.zeros_like(self.A_matrices_list[l])})
                self.adam_state['markov']['b'].append({'m': np.zeros_like(self.b_vectors_list[l]), 'v': np.zeros_like(self.b_vectors_list[l])})
        # Increment timestep
        self.adam_state['t'] += 1
        t = self.adam_state['t']
        # --- Markov parameters ---
        for l in range(self.L):
            Ej_grad, Bij_grad, Fij_grad = [g / batch_size for g in self.markov_grad_accum['theta'][l]]
            wm = self.weight_matrix_list[l]
            # Explicit Adam update for Ej_list
            state_Ej = self.adam_state['markov']['theta'][l][0]
            state_Ej['m'] = beta1 * state_Ej['m'] + (1 - beta1) * Ej_grad
            state_Ej['v'] = beta2 * state_Ej['v'] + (1 - beta2) * (Ej_grad ** 2)
            m_hat_Ej = state_Ej['m'] / (1 - beta1 ** t)
            v_hat_Ej = state_Ej['v'] / (1 - beta2 ** t)
            wm.Ej_list = wm.Ej_list + eta_markov * m_hat_Ej / (np.sqrt(v_hat_Ej) + epsilon)
            # Explicit Adam update for Bij_list
            state_Bij = self.adam_state['markov']['theta'][l][1]
            state_Bij['m'] = beta1 * state_Bij['m'] + (1 - beta1) * Bij_grad
            state_Bij['v'] = beta2 * state_Bij['v'] + (1 - beta2) * (Bij_grad ** 2)
            m_hat_Bij = state_Bij['m'] / (1 - beta1 ** t)
            v_hat_Bij = state_Bij['v'] / (1 - beta2 ** t)
            wm.Bij_list = wm.Bij_list + eta_markov * m_hat_Bij / (np.sqrt(v_hat_Bij) + epsilon)
            # Explicit Adam update for Fij_list
            state_Fij = self.adam_state['markov']['theta'][l][2]
            state_Fij['m'] = beta1 * state_Fij['m'] + (1 - beta1) * Fij_grad
            state_Fij['v'] = beta2 * state_Fij['v'] + (1 - beta2) * (Fij_grad ** 2)
            m_hat_Fij = state_Fij['m'] / (1 - beta1 ** t)
            v_hat_Fij = state_Fij['v'] / (1 - beta2 ** t)
            wm.Fij_list = wm.Fij_list + eta_markov * m_hat_Fij / (np.sqrt(v_hat_Fij) + epsilon)
            # Set updated parameters
            #print(wm.Ej_list)
            wm.set_W_mat(wm.Ej_list, wm.Bij_list, wm.Fij_list)
            #print(wm.Ej_list)
        for l in range(self.L - 1):
            A_grad = self.markov_grad_accum['A'][l] / batch_size
            b_grad = self.markov_grad_accum['b'][l] / batch_size
            A_state = self.adam_state['markov']['A'][l]
            b_state = self.adam_state['markov']['b'][l]
            # Update moments for A
            A_state['m'] = beta1 * A_state['m'] + (1 - beta1) * A_grad
            A_state['v'] = beta2 * A_state['v'] + (1 - beta2) * (A_grad ** 2)
            m_hat_A = A_state['m'] / (1 - beta1 ** t)
            v_hat_A = A_state['v'] / (1 - beta2 ** t)
            self.A_matrices_list[l] += eta_markov * m_hat_A / (np.sqrt(v_hat_A) + epsilon)
            # Update moments for b
            b_state['m'] = beta1 * b_state['m'] + (1 - beta1) * b_grad
            b_state['v'] = beta2 * b_state['v'] + (1 - beta2) * (b_grad ** 2)
            m_hat_b = b_state['m'] / (1 - beta1 ** t)
            v_hat_b = b_state['v'] / (1 - beta2 ** t)
            self.b_vectors_list[l] += eta_markov * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    def compute_input_gradient(self, inputs):
        """
        Computes the gradient of the perceptron output (for class_idx) with respect to the input vector.
        Args:
            inputs: input vector (numpy array or torch tensor)
            class_idx: integer, class index to maximize
        Returns:
            grad: numpy array, gradient of class prediction wrt input vector
        """
        ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
        dpiL_dthetal_lists, _, _ = self.stacked_derivatives_of_ss(ss_list, inputs_list)
        full_inds = self.external_input_inds
        n_nodes = self.weight_matrix_list[-1].n_nodes
        dpiL_dInputs = np.zeros((n_nodes, len(inputs)))
        for m, inds in enumerate(full_inds):  # Iterate over columns
            dpiL_dInputs[:, m] = np.sum(
                [dpiL_dthetal_lists[0][2][:, ind] for ind in inds], axis=0)
        return dpiL_dInputs


    def compute_marginal_distribution(self, input_data, n_nodes, n_classes, n_samples=1000):
        """Compute the marginal steady state distribution by sampling from input data.
        
        Args:
            stacked_weight_matrices: StackedWeightMatrices object
            input_data: InputData object containing training data
            n_nodes: Number of nodes in the network
            n_classes: Number of classes in the input data
            n_samples: Number of samples to use for computing marginal (default 1000)
            
        Returns:
            marginal_ss: Marginal steady state distribution averaged over samples
        """
        marginal_ss = np.zeros(n_nodes)
        for _ in range(n_samples):
            class_number = random.randrange(n_classes)  # draw a random class label to present
            inputs = input_data.get_next_training_sample(class_number)

            ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)

            marginal_ss += ss_list[-1]

        marginal_ss /= n_samples
        return marginal_ss

    def compute_MI(self, input_data, n_nodes, n_classes, n_samples=1000):
        """Compute the mutual information between the input and the output.
        
        Args:
            stacked_weight_matrices: StackedWeightMatrices object
            input_data: InputData object containing training data
            n_nodes: Number of nodes in the network
            n_classes: Number of classes in the input data
            n_samples: Number of samples to use for computing MI (default 1000)
            
        Returns:
            mi: Mutual information in bits between input and output
        """
        marginal_ss = self.compute_marginal_distribution(input_data, n_nodes, n_classes, n_samples)
        mi = 0
        for _ in range(n_samples):
            class_number = random.randrange(n_classes)  # draw a random class label to present
            inputs = input_data.get_next_training_sample(class_number)

            ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
            mi += np.sum(ss_list[-1] * np.log(ss_list[-1] / marginal_ss))

        mi /= n_samples
        return mi / np.log(2) # return in bits
    
    def compute_MI_gradient(self, input_data, n_classes, n_samples=1000):
        """
        Compute the gradient of the mutual information between the input and the output.
        Only works for L = 1 (single layer).
        """
        # First pass: accumulate marginal steady state and marginal gradients
        marginal_ss = None
        marginal_grads = None
        for sample in range(n_samples):
            class_number = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(class_number)

            ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
            dpiL_dthetal_lists, _, _ = self.stacked_derivatives_of_ss(ss_list, inputs_list)
            grad = dpiL_dthetal_lists[-1]  # [dE, dB, dF], each shape (n_nodes, param_dim)
            ss = ss_list[-1]               # steady state vector, shape (n_nodes,)

            if sample == 0:
                marginal_ss = np.array(ss, dtype=np.float64)
                marginal_grads = [np.array(g, dtype=np.float64) for g in grad]
            else:
                marginal_ss += ss
                for i in range(len(marginal_grads)):
                    marginal_grads[i] += grad[i]

        marginal_ss /= n_samples
        marginal_grads = [g / n_samples for g in marginal_grads]

        # Second pass: accumulate MI gradient
        mi_grad = [0.0 for _ in marginal_grads]  # one for each parameter type
        for sample in range(n_samples):
            class_number = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(class_number)
            ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
            dpiL_dthetal_lists, _, _ = self.stacked_derivatives_of_ss(ss_list, inputs_list)
            grad = dpiL_dthetal_lists[-1]  # [dE, dB, dF], each shape (n_nodes, param_dim)
            ss = ss_list[-1]

            log_ratio = np.log(ss / marginal_ss)
            ratio = ss / marginal_ss

            for i in range(len(mi_grad)):
                # grad[i] and marginal_grads[i] are (n_nodes, param_dim)
                # log_ratio and ratio are (n_nodes,)
                # We want to sum over n_nodes: (n_nodes,) * (n_nodes, param_dim) -> (param_dim,)
                term_1 = np.sum(log_ratio[:, None] * grad[i], axis=0)
                term_2 = np.sum(ratio[:, None] * marginal_grads[i], axis=0)
                mi_grad[i] += term_1 - term_2

        mi_grad = [g / n_samples for g in mi_grad]
        return tuple(mi_grad)
    
    



class StackedWeightMatricesWithPerceptron(StackedWeightMatrices):
    """
    Extends StackedWeightMatrices by adding a perceptron layer implemented in PyTorch.
    The perceptron takes the final steady-state output and performs additional processing.
    """

    def __init__(self, weight_matrix_list,
                external_dims, internal_dims,
                M_vals, A_fac, b_fac,
                perceptron_hidden_dims,  # List of hidden layer dimensions
                perceptron_output_dim,   # Final output dimension
                rand_bool=True):
        """
        Initializes the combined Markov-Perceptron architecture.

        Parameters:
        - weight_matrix_list: List of WeightMatrix objects for the Markov computation
        - external_dims: [external_input_dim, external_output_dim] for Markov layers
        - internal_dims: Internal dimensions for Markov layers
        - M_vals: Number of edges per input for each layer
        - A_fac: Scaling factor for A matrices
        - b_fac: Scaling factor for b vectors
        - perceptron_hidden_dims: List of hidden layer dimensions for the perceptron
        - perceptron_output_dim: Output dimension of the perceptron
        - rand_bool: Whether to use random initialization
        """
        # Initialize the base StackedWeightMatrices
        super().__init__(weight_matrix_list, external_dims, internal_dims, 
                        M_vals, A_fac, b_fac, rand_bool)
        
        # Store perceptron dimensions
        self.perceptron_input_dim = len(self.external_output_inds)
        self.perceptron_hidden_dims = perceptron_hidden_dims
        self.perceptron_output_dim = perceptron_output_dim
        
        # Initialize the perceptron layers
        self._init_perceptron(perceptron_hidden_dims, perceptron_output_dim)

    def _init_perceptron(self, hidden_dims, output_dim):
        """
        Initializes the perceptron layers using PyTorch.
        Creates a multi-layer perceptron with ReLU activations and log-softmax output.
        """
        layers = []
        in_dim = self.perceptron_input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU()
            ])
            in_dim = hidden_dim

        # Add output layer with log-softmax activation for numerical stability
        layers.extend([
            torch.nn.Linear(in_dim, output_dim),
            torch.nn.LogSoftmax(dim=1)
        ])
        
        # Create sequential model
        self.perceptron = torch.nn.Sequential(*layers)
        
        # Loss function
        self.criterion = torch.nn.NLLLoss() # negative log-likelihood

    def compute_combined_output(self, inputs):
        """
        Computes the full forward pass through both Markov and perceptron layers.

        Parameters:
        - inputs: Input data for the Markov computation

        Returns:
        - markov_ss_list: List of steady states from Markov layers
        - inputs_list: List of inputs used at each Markov layer
        - perceptron_output: Final output after perceptron processing (log probabilities)
        """
        # Get Markov steady states
        markov_ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
        
        # Extract final layer steady states at output indices
        final_ss = np.array(markov_ss_list[-1])  # Convert JAX array to numpy
        markov_outputs = torch.tensor([final_ss[i] for i in self.external_output_inds], 
                                    dtype=torch.float32).unsqueeze(0)
        
        # Forward pass through perceptron
        self.perceptron.eval()  # Set to evaluation mode
        with torch.no_grad():
            log_probs = self.perceptron(markov_outputs)
            perceptron_output = torch.exp(log_probs)  # Convert back to probabilities
        
        return markov_ss_list, inputs_list, perceptron_output
    
    def compute_output(self, inputs):
        """
        Computes output probabilities for the given inputs using the Markov-Perceptron model.
        Returns the perceptron output probabilities as a numpy array.
        """
        _, _, perceptron_output = self.compute_combined_output(inputs)
        return perceptron_output.squeeze().detach().cpu().numpy()

    def compute_combined_gradients(self, inputs, target, markov_ss_list=None, inputs_list=None):
        """
        Computes gradients for both Markov and perceptron parameters.
        Uses chain rule to backpropagate the loss through both the perceptron and Markov components.
        If markov_ss_list and inputs_list are provided, uses them instead of recomputing.
        """
        # Get Markov steady states
        if markov_ss_list is None or inputs_list is None:
            markov_ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs)
        
        # Convert JAX arrays to numpy arrays
        markov_ss_list = [np.array(ss) for ss in markov_ss_list]
        inputs_list = [np.array(inp) for inp in inputs_list]
        
        # Extract final layer steady states at output indices
        final_ss = markov_ss_list[-1]
        markov_outputs = torch.tensor([final_ss[i] for i in self.external_output_inds], 
                                    dtype=torch.float32).unsqueeze(0)
        markov_outputs.requires_grad_(True)
        
        # Forward pass through perceptron
        self.perceptron.train() # sets to training mode
        for param in self.perceptron.parameters():
            if param.grad is not None:
                param.grad.zero_() # zero out the gradient if it exists
                
        log_probs = self.perceptron(markov_outputs) 
        
        # Compute loss and get initial gradient
        target_tensor = torch.tensor([target], dtype=torch.long)
        loss = self.criterion(log_probs, target_tensor) # computes the target element of the log_probs
        loss.backward() # gradents are stored in .grad attribute of each parameter
        
        # Get gradient with respect to Markov outputs
        dLoss_dMarkov = markov_outputs.grad.numpy() # get the gradient of the loss with respect to the Markov outputs
        
        # Get perceptron gradients
        perceptron_gradients = {name: param.grad.clone() for name, param in self.perceptron.named_parameters()}
        
        # Compute Markov gradients using the chain rule
        dpiL_dthetal_lists, dpiL_dAl_lists, dpiL_dbl_lists = self.stacked_derivatives_of_ss(markov_ss_list, inputs_list)
        
        # Convert JAX arrays to numpy if needed
        dpiL_dthetal_lists = [[np.array(grad) for grad in layer_grads] for layer_grads in dpiL_dthetal_lists]
        dpiL_dAl_lists = [np.array(grad) for grad in dpiL_dAl_lists]
        dpiL_dbl_lists = [np.array(grad) for grad in dpiL_dbl_lists]
        
        markov_gradients = {
            'theta': [],
            'A': [],
            'b': []
        }
        
        # Process theta gradients for each layer
        for l in range(self.L):
            wm = self.weight_matrix_list[l]
            
            # Initialize gradients with correct shapes
            Ej_grad = np.zeros_like(wm.Ej_list)
            Bij_grad = np.zeros_like(wm.Bij_list)
            Fij_grad = np.zeros_like(wm.Fij_list)
            
            # Accumulate gradients for each output
            for out_idx, out_ind in enumerate(self.external_output_inds):
                # Ensure proper broadcasting by matching shapes
                dtheta = dpiL_dthetal_lists[l]
                Ej_grad += dLoss_dMarkov[0, out_idx] * dtheta[0][out_ind]
                Bij_grad += dLoss_dMarkov[0, out_idx] * dtheta[1][out_ind]
                Fij_grad += dLoss_dMarkov[0, out_idx] * dtheta[2][out_ind]
    
            markov_gradients['theta'].append([Ej_grad, Bij_grad, Fij_grad])
        
        # Process A and b gradients
        for l in range(self.L - 1):
            A_grad = np.zeros_like(self.A_matrices_list[l])
            b_grad = np.zeros_like(self.b_vectors_list[l])
            
            for out_idx, out_ind in enumerate(self.external_output_inds):
                A_grad += dLoss_dMarkov[0, out_idx] * dpiL_dAl_lists[l][out_ind]
                b_grad += dLoss_dMarkov[0, out_idx] * dpiL_dbl_lists[l][out_ind]
            
            markov_gradients['A'].append(A_grad)
            markov_gradients['b'].append(b_grad)
        
        return markov_gradients, perceptron_gradients
    
    def _perceptron_step(self, eta_perceptron):
        """
        Performs a simple gradient descent step on the perceptron parameters.
        
        Parameters:
        - eta_perceptron: Learning rate for perceptron parameters
        """
        with torch.no_grad():
            for layer in self.perceptron:
                if isinstance(layer, torch.nn.Linear):
                    # Update weights
                    layer.weight -= eta_perceptron * layer.weight.grad
                    # Update bias
                    layer.bias -= eta_perceptron * layer.bias.grad

    def update_combined(self, inputs, target, eta_markov, eta_perceptron, markov_ss_list=None, inputs_list=None):
        """
        Updates both Markov and perceptron parameters using computed gradients.
        If markov_ss_list and inputs_list are provided, uses them instead of recomputing.
        """
        # Compute all gradients with proper backpropagation
        markov_gradients, _ = self.compute_combined_gradients(
            inputs, target, markov_ss_list=markov_ss_list, inputs_list=inputs_list
        )
        
        # Update Markov parameters using the backpropagated gradients
        for l in range(self.L):
            Ej_grad, Bij_grad, Fij_grad = markov_gradients['theta'][l]
            wm = self.weight_matrix_list[l]
            
            # Ensure gradients have correct shapes
            Ej_grad = np.asarray(Ej_grad).reshape(wm.Ej_list.shape)
            Bij_grad = np.asarray(Bij_grad).reshape(wm.Bij_list.shape)
            Fij_grad = np.asarray(Fij_grad).reshape(wm.Fij_list.shape)
            
            # Update parameters with shape verification 
            new_Ej = wm.Ej_list - eta_markov * Ej_grad
            new_Bij = wm.Bij_list - eta_markov * Bij_grad
            new_Fij = wm.Fij_list - eta_markov * Fij_grad
               
            # Update parameters
            wm.set_W_mat(new_Ej, new_Bij, new_Fij)
        
        # Update A matrices and b vectors
        for l in range(self.L - 1):
            A_grad = np.asarray(markov_gradients['A'][l]).reshape(self.A_matrices_list[l].shape)
            b_grad = np.asarray(markov_gradients['b'][l]).reshape(self.b_vectors_list[l].shape)

            
            self.A_matrices_list[l] -= eta_markov * A_grad
            self.b_vectors_list[l] -= eta_markov * b_grad
        
        # Update perceptron parameters using simple gradient descent
        self._perceptron_step(eta_perceptron)

    def zero_gradients(self):
        """Initialize/zero accumulators for Markov and perceptron gradients."""
        self.markov_grad_accum = {
            'theta': [[np.zeros_like(wm.Ej_list), np.zeros_like(wm.Bij_list), np.zeros_like(wm.Fij_list)] for wm in self.weight_matrix_list],
            'A': [np.zeros_like(A) for A in self.A_matrices_list],
            'b': [np.zeros_like(b) for b in self.b_vectors_list]
        }
        self.perceptron_grad_accum = {name: torch.zeros_like(param) for name, param in self.perceptron.named_parameters()}

    def compute_gradients_single(self, inputs, target, markov_ss_list=None, inputs_list=None):
        """Compute gradients for a single sample."""
        return self.compute_combined_gradients(inputs, target, markov_ss_list, inputs_list)

    def accumulate_gradients(self, markov_grads, perceptron_grads):
        """Add single-sample gradients to the running total."""
        for l in range(self.L):
            for i in range(3):
                self.markov_grad_accum['theta'][l][i] += markov_grads['theta'][l][i]
        for l in range(self.L - 1):
            self.markov_grad_accum['A'][l] += markov_grads['A'][l]
            self.markov_grad_accum['b'][l] += markov_grads['b'][l]
        for name in self.perceptron_grad_accum:
            self.perceptron_grad_accum[name] += perceptron_grads[name]

    def apply_gradients(self, batch_size, eta_markov, eta_perceptron):
        """Update parameters using the averaged gradients."""
        # Average gradients and update Markov parameters
        for l in range(self.L):
            Ej_grad, Bij_grad, Fij_grad = [g / batch_size for g in self.markov_grad_accum['theta'][l]]
            wm = self.weight_matrix_list[l]
            new_Ej = wm.Ej_list - eta_markov * Ej_grad
            new_Bij = wm.Bij_list - eta_markov * Bij_grad
            new_Fij = wm.Fij_list - eta_markov * Fij_grad
            wm.set_W_mat(new_Ej, new_Bij, new_Fij)
        for l in range(self.L - 1):
            self.A_matrices_list[l] -= eta_markov * (self.markov_grad_accum['A'][l] / batch_size)
            self.b_vectors_list[l] -= eta_markov * (self.markov_grad_accum['b'][l] / batch_size)
        # Average gradients and update perceptron parameters
        with torch.no_grad():
            for name, param in self.perceptron.named_parameters():
                param -= eta_perceptron * (self.perceptron_grad_accum[name] / batch_size)

    def apply_adam_gradients(self, batch_size, eta_markov, eta_perceptron, beta1, beta2, epsilon):
        """
        Update parameters using the Adam optimizer with batch-averaged gradients.
        Maintains Adam state for each parameter.
        """
        # Initialize Adam state if not present
        if not hasattr(self, 'adam_state'):
            self.adam_state = {'t': 0, 'markov': {'theta': [], 'A': [], 'b': []}, 'perceptron': {}}
            # Markov parameters
            for l in range(self.L):
                wm = self.weight_matrix_list[l]
                self.adam_state['markov']['theta'].append([
                    {'m': np.zeros_like(wm.Ej_list), 'v': np.zeros_like(wm.Ej_list)},
                    {'m': np.zeros_like(wm.Bij_list), 'v': np.zeros_like(wm.Bij_list)},
                    {'m': np.zeros_like(wm.Fij_list), 'v': np.zeros_like(wm.Fij_list)}
                ])
            for l in range(self.L - 1):
                self.adam_state['markov']['A'].append({'m': np.zeros_like(self.A_matrices_list[l]), 'v': np.zeros_like(self.A_matrices_list[l])})
                self.adam_state['markov']['b'].append({'m': np.zeros_like(self.b_vectors_list[l]), 'v': np.zeros_like(self.b_vectors_list[l])})
            # Perceptron parameters
            for name, param in self.perceptron.named_parameters():
                self.adam_state['perceptron'][name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param)
                }
        # Increment timestep
        self.adam_state['t'] += 1
        t = self.adam_state['t']
        # --- Markov parameters ---
        for l in range(self.L):
            Ej_grad, Bij_grad, Fij_grad = [g / batch_size for g in self.markov_grad_accum['theta'][l]]
            wm = self.weight_matrix_list[l]
            for i, (grad, param, state) in enumerate(zip([Ej_grad, Bij_grad, Fij_grad],
                                                         [wm.Ej_list, wm.Bij_list, wm.Fij_list],
                                                         self.adam_state['markov']['theta'][l])):
                # Update moments
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                # Adam update
                param -= eta_markov * m_hat / (np.sqrt(v_hat) + epsilon)
            # Set updated parameters
            wm.set_W_mat(wm.Ej_list, wm.Bij_list, wm.Fij_list)
        for l in range(self.L - 1):
            A_grad = self.markov_grad_accum['A'][l] / batch_size
            b_grad = self.markov_grad_accum['b'][l] / batch_size
            A_state = self.adam_state['markov']['A'][l]
            b_state = self.adam_state['markov']['b'][l]
            # Update moments for A
            A_state['m'] = beta1 * A_state['m'] + (1 - beta1) * A_grad
            A_state['v'] = beta2 * A_state['v'] + (1 - beta2) * (A_grad ** 2)
            m_hat_A = A_state['m'] / (1 - beta1 ** t)
            v_hat_A = A_state['v'] / (1 - beta2 ** t)
            self.A_matrices_list[l] -= eta_markov * m_hat_A / (np.sqrt(v_hat_A) + epsilon)
            # Update moments for b
            b_state['m'] = beta1 * b_state['m'] + (1 - beta1) * b_grad
            b_state['v'] = beta2 * b_state['v'] + (1 - beta2) * (b_grad ** 2)
            m_hat_b = b_state['m'] / (1 - beta1 ** t)
            v_hat_b = b_state['v'] / (1 - beta2 ** t)
            self.b_vectors_list[l] -= eta_markov * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
        # --- Perceptron parameters ---
        with torch.no_grad():
            for name, param in self.perceptron.named_parameters():
                grad = self.perceptron_grad_accum[name] / batch_size
                state = self.adam_state['perceptron'][name]
                # Update moments
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                # Adam update
                param -= eta_perceptron * m_hat / (torch.sqrt(v_hat) + epsilon)

    def compute_input_gradient(self, inputs, class_idx):
        """
        Computes the gradient of the perceptron output (for class_idx) with respect to the input vector.
        Args:
            inputs: input vector (numpy array or torch tensor)
            class_idx: integer, class index to maximize
        Returns:
            grad: numpy array, gradient of class prediction wrt input vector
        """
        # Ensure input is a torch tensor with gradients enabled
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        else:
            inputs = inputs.clone().detach().requires_grad_(True)

        # Forward pass through Markov layers (using numpy, so detach and re-wrap as torch)
        ss_list, inputs_list = self.compute_stacked_ss_on_inputs(inputs.detach().cpu().numpy())
        final_ss = np.array(ss_list[-1])
        markov_outputs = torch.tensor([final_ss[i] for i in self.external_output_inds], dtype=torch.float32).unsqueeze(0)
        markov_outputs.requires_grad_(True)

        self.perceptron.eval()
        output = self.perceptron(markov_outputs)
        prob = torch.exp(output)[0, class_idx]
     
        # Backpropagate to markov_outputs
        self.perceptron.zero_grad()
        prob.backward(retain_graph=True)
        dprob_dmarkov = markov_outputs.grad.clone().detach().cpu().numpy()[0]


        dpiL_dthetal_lists, _, _ = self.stacked_derivatives_of_ss(ss_list, inputs_list)
        full_inds = self.external_input_inds
        n_nodes = self.weight_matrix_list[-1].n_nodes
        dpiL_dInputs = np.zeros((n_nodes, len(inputs)))
        for m, inds in enumerate(full_inds):  # Iterate over columns
            dpiL_dInputs[:, m] = np.sum(
                [dpiL_dthetal_lists[0][2][:, ind] for ind in inds], axis=0)
        dpioL_dInputs = dpiL_dInputs[self.external_output_inds, :]

        grad = np.dot(dprob_dmarkov, dpioL_dInputs)
        return grad



class InputData:
    """
    Manages labeled input data for training and testing.
    """

    def __init__(self, n_classes, data_list, split_fac=0.75):
        """
        Initializes training and testing datasets.

        Parameters:
        - n_classes: Number of output classes
        - data_list: List of data samples per class
        - split_fac: Fraction of data used for training
        """
        self.n_classes = n_classes
        self.labels = self._create_labels()
        self.data_list = data_list 
        self.split_fac = split_fac
        self.training_data, self.testing_data = self._split_shuffle_data(data_list, split_fac)

    def _create_labels(self):
        """Creates one-hot encoded labels for classification."""
        return [np.eye(self.n_classes)[n] for n in range(self.n_classes)]

    def _split_shuffle_data(self, data_list, split_fac):
        """Splits data into training and testing sets."""
        tr_data, te_data = [], []
        for nc in range(self.n_classes):
            sub_data = data_list[nc]
            random.shuffle(sub_data)
            n_train = round(split_fac * len(sub_data))
            tr_data.append(iter(sub_data[:n_train]))
            te_data.append(iter(sub_data[n_train:]))
        return tr_data, te_data

    def refill_iterators(self):
        """Refills the training and testing iterators from data_list."""
        self.training_data, self.testing_data = self._split_shuffle_data(self.data_list, self.split_fac)

    def get_next_training_sample(self, class_number):
        """Returns the next training sample for the given class, refilling iterators as needed."""
        try:
            return next(self.training_data[class_number])
        except StopIteration:
            self.refill_iterators()
            return next(self.training_data[class_number])


def compute_soft_maxed_output(ss, output_inds):
    """
    Computes a softmax transformation over selected steady-state values.

    Parameters:
    - ss: Steady-state vector
    - output_inds: Indices of output nodes

    Returns:
    - Softmax-normalized probabilities
    """
    vec = np.array([ss[x] for x in output_inds])
    exp_shifted = np.exp(vec - np.max(vec))
    return exp_shifted / np.sum(exp_shifted)


def get_input_inds(n_edges, input_dim, M, rand_bool=True):
    """
    Selects random edges to receive inputs.

    Parameters:
    - n_edges: Total number of edges
    - input_dim: Number of input channels
    - M: Number of edges per input

    Returns:
    - List of selected edge indices per input channel
    """
    if rand_bool:
        shuffled_inds = list(range(n_edges))
        random.shuffle(shuffled_inds)
        return [shuffled_inds[m * M:(m + 1) * M] for m in range(input_dim)]
    else:
        inds = list(range(n_edges))
        return [inds[m * M:(m + 1) * M] for m in range(input_dim)]


def get_output_inds(n_nodes, n_classes, rand_bool=True):
    """
    Selects output nodes.

    Parameters:
    - n_nodes: Total number of nodes
    - n_classes: Number of output classes
    - rand_bool: Whether to shuffle node selection

    Returns:
    - List of output node indices
    """
    output_inds = list(range(n_nodes))
    if rand_bool:
        random.shuffle(output_inds)
    return output_inds[:n_classes]

def random_initial_parameters(e_range, b_range, f_range, n_nodes, n_edges):
    """
    Generate random initial parameters for a network with given ranges and dimensions.
    Returns tuple of (ej_list, bij_list, fij_list) containing random values within specified ranges.
    """
    ej_list = (2 * np.random.random(n_nodes) - 1) * e_range
    bij_list = (2 * np.random.random(n_edges) - 1) * b_range
    fij_list = (2 * np.random.random(n_edges) - 1) * f_range
    
    return (ej_list, bij_list, fij_list)


def compute_error(ss, input_data, class_number, output_inds):
    ss_at_outputs = [ss[output_ind] for output_ind in output_inds] 
    true_output = input_data.labels[class_number]
    return np.array(ss_at_outputs - true_output)


def downsample_avg(image, m=2):
    h, w = image.shape
    assert h % m == 0 and w % m == 0, "Image dimensions must be divisible by m"
    return image.reshape(h//m, m, w//m, m).mean(axis=(1, 3))


def load_and_format_mnist(n_classes, scale, m):
    # Load the MNIST dataset from sklearn
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache = True)
    x_all, y_all = mnist.data, mnist.target.astype(int)  # Convert labels to integers

    # Normalize pixel values to range [0,1]
    x_all = scale * (x_all.astype(np.float32) / 255.0 - 0.0)

    # Reshape images from (784,) to (28,28) for processing
    x_all = x_all.reshape(-1, 28, 28)

    # Initialize an empty dictionary with keys for each digit class (0-9)
    mnist_dict = {i: [] for i in range(10)}

    # Populate the dictionary
    for img, label in zip(x_all, y_all):
        img_reshape = downsample_avg(img, m)
        mnist_dict[label].append(np.array(img_reshape).flatten())

    ###  create InputData object
    if isinstance(n_classes, int):
        return InputData(n_classes, [mnist_dict[key] for key in range(n_classes)])
    else:
        return InputData(len(n_classes), [mnist_dict[key] for key in n_classes])


def load_and_format_iris(n_classes, scale):
    # Load the Iris dataset
    iris = datasets.load_iris()
    x_all, y_all = iris.data, iris.target  # Features and labels

    # Normalize features to range [0,1]
    x_all = scale * (x_all - x_all.min(axis=0)) / (x_all.max(axis=0) - x_all.min(axis=0))
    

    # Initialize a dictionary with keys for each class
    iris_dict = {i: [] for i in range(3)}

    # Populate the dictionary with feature vectors
    for features, label in zip(x_all, y_all):
        iris_dict[label].append(np.array(features).flatten())

    # Create and return InputData object
    return InputData(n_classes, [iris_dict[key] for key in range(n_classes)])



def evaluate_accuracy(model, input_data, n_classes, n_evals, input_inds=None, output_inds=None):
    """
    Generalized accuracy evaluation for any model with a compute_output method.

    Parameters:
    - model: The model instance (must have compute_output method)
    - input_data: InputData object
    - n_classes: Number of classes
    - n_evals: Number of evaluation samples
    - input_inds: (Optional) Input indices for WeightMatrix
    - output_inds: (Optional) Output indices for WeightMatrix

    Returns:
    - accuracy: Classification accuracy (float)
    """
    accuracy = 0.0
    for n in range(n_evals):
        class_number = random.randrange(n_classes)
        try:
            inputs = next(input_data.testing_data[class_number])
        except StopIteration:
            input_data.refill_iterators()
            inputs = next(input_data.testing_data[class_number])

        if isinstance(model, WeightMatrix):
            probs = model.compute_output(inputs, input_inds, output_inds)
        else:
            probs = model.compute_output(inputs)

        pred = np.argmax(probs)
        if pred == class_number:
            accuracy += 1.0

    return accuracy / n_evals

def evaluate_accuracy_per_class(model, input_data, n_evals, n_classes, input_inds=None, output_inds=None):
    """
    Generalized accuracy-per-class evaluation for any model with a compute_output method.

    Parameters:
    - model: The model instance (must have compute_output method)
    - input_data: InputData object
    - n_evals: Number of evaluation samples per class
    - n_classes: Number of classes
    - input_inds: (Optional) Input indices for WeightMatrix
    - output_inds: (Optional) Output indices for WeightMatrix

    Returns:
    - predictions_per_class: n_classes x n_classes confusion matrix
    """
    predictions_per_class = np.zeros((n_classes, n_classes))

    for class_number in range(n_classes):
        for n in range(n_evals):
            try:
                inputs = next(input_data.testing_data[class_number])
            except StopIteration:
                input_data.refill_iterators()
                inputs = next(input_data.testing_data[class_number])

            if isinstance(model, WeightMatrix):
                probs = model.compute_output(inputs, input_inds, output_inds)
            else:
                probs = model.compute_output(inputs)

            pred = np.argmax(probs)
            predictions_per_class[pred][class_number] += 1.0

    return predictions_per_class


def get_spanning_trees(n_nodes,edge_w,N_trees,maximum=True):
    #provide edge weights of a complete graoh with n_nodes to obtain N_trees spanning trees 
    #from decreasing (maximum=True) or increasing value (maximum=False)
    for wij in edge_w:
        dir_g = nx.DiGraph()
        dir_g.add_nodes_from(range(n_nodes))
        for kk in range(n_nodes):
            for ll in range(n_nodes):
                if kk != ll:
                    dir_g.add_edge(kk,ll,weight= edge_w[kk][ll]) #these are the reveresed edges to account for the inverse of the root direction used in graph theory  
    
    span_trees = []
    span_trees_roots = []
    
    if N_trees < 2 :
        if maximum:
            span_trees.append(nx.maximum_spanning_arborescence(dir_g).reverse())
            span_trees_roots.append(np.argmin(list(dict(span_trees[-1].out_degree()).values())))
    
        else: 
            span_trees.append(nx.minimum_spanning_arborescence(dir_g).reverse())
            span_trees_roots.append(np.argmin(list(dict(span_trees[-1].out_degree()).values())))
        
    else:
    
        tree_iter = nx.algorithms.tree.branchings.ArborescenceIterator(dir_g, minimum= not maximum)
    
        for tree_i, s_tree in enumerate(tree_iter):
            span_trees.append(s_tree.reverse())
            span_trees_roots.append(np.argmin(list(dict(span_trees[-1].out_degree()).values())))
            print('finished with tree',tree_i)
            if tree_i == N_trees - 1 :
                break
    
    return (span_trees_roots, span_trees)


def append_r_before_slash(path, id=1):
    """
    Insert '_r' before the slash at the given index (from the right, like rfind).
    id = -1 means the last slash, -2 the second-to-last, etc.
    """
    # Find all slash positions
    slash_indices = [i for i, c in enumerate(path) if c == '/']
    if not slash_indices or abs(id) > len(slash_indices):
        # If id is out of range, just append at the end
        return path + '_r'
    idx = slash_indices[id]
    return path[:idx] + '_r' + path[idx:]


def compute_flux_and_frenesy_stacked(stacked_wm, inputs):
    """
    For a StackedWeightMatrices object and input vector, compute for each layer and each edge:
      - flux: J_{i->j} = pi_i * W_ij - pi_j * W_ji
      - frenesy: F_{i->j} = pi_i * W_ij + pi_j * W_ji
    Returns a list (one per layer) of dicts with keys: 'edge', 'flux', 'frenesy'.
    """
    results = []
    # Get steady states for all layers
    ss_list, _ = stacked_wm.compute_stacked_ss_on_inputs(inputs)
    for l, wm in enumerate(stacked_wm.weight_matrix_list):
        pi = np.array(ss_list[l])
        edge_array = wm.edge_array
        W_mat = wm.W_mat
        layer_result = []
        for idx, (i, j) in enumerate(edge_array):
            flux = pi[i] * W_mat[i, j] - pi[j] * W_mat[j, i]
            frenesy = pi[i] * W_mat[i, j] + pi[j] * W_mat[j, i]
            layer_result.append({'edge': (i, j), 'flux': flux, 'frenesy': frenesy})
        results.append(layer_result)
    return results



def train_mi_conjugate_gradient(
        stacked_weight_matrices,
        input_data,
        n_nodes,
        n_classes,
        n_samples=1000,
        n_epochs=100,
        step_size=1e-2,
        tol=1e-6,
        verbose=True
    ):
        """
        Train stacked_weight_matrices to maximize MI using conjugate gradient on Markov parameters.
        Only updates Ej_list, Bij_list, Fij_list for each layer.
        """
        # Initialize previous gradients and directions
        prev_grads = None
        prev_dirs = None

        mi_list = []
        for epoch in range(n_epochs):
            # Compute MI gradient (tuple of arrays for each param type)
            mi_grads = stacked_weight_matrices.compute_MI_gradient(input_data, n_classes, n_samples)
            mi = stacked_weight_matrices.compute_MI(input_data, n_nodes, n_classes, n_samples)
            mi_list.append(mi)
            # mi_grads: [dE, dB, dF] for each layer (for L=1, just one set)

            # Flatten all gradients for CG direction calculation
            flat_grads = []
            for l, (dE, dB, dF) in enumerate([mi_grads]):
                flat_grads.append(np.concatenate([dE.flatten(), dB.flatten(), dF.flatten()]))
            flat_grads = np.concatenate(flat_grads)

            # Compute conjugate direction
            if prev_grads is None:
                direction = flat_grads
            else:
                # Polak–Ribiere formula
                y = flat_grads - prev_grads
                beta = np.dot(flat_grads, y) / (np.dot(prev_grads, prev_grads) + 1e-12)
                direction = flat_grads + beta * prev_dirs

            # Unflatten direction into parameter shapes
            offset = 0

            wm = stacked_weight_matrices.weight_matrix_list[-1]
            # Shapes
            E_shape = wm.Ej_list.shape
            B_shape = wm.Bij_list.shape
            F_shape = wm.Fij_list.shape
            E_size = np.prod(E_shape)
            B_size = np.prod(B_shape)
            F_size = np.prod(F_shape)

            # Extract direction slices
            dE_dir = direction[offset:offset+E_size].reshape(E_shape)
            offset += E_size
            dB_dir = direction[offset:offset+B_size].reshape(B_shape)
            offset += B_size
            dF_dir = direction[offset:offset+F_size].reshape(F_shape)
            offset += F_size

            # Update parameters (gradient ascent)
            wm.set_W_mat(
                wm.Ej_list + step_size * dE_dir,
                wm.Bij_list + step_size * dB_dir,
                wm.Fij_list + step_size * dF_dir
            )

            
            # Check for convergence
            grad_norm = np.linalg.norm(flat_grads)
            if verbose:
                print(f"Epoch {epoch+1}: grad_norm={grad_norm:.4e}")
            if grad_norm < tol:
                print("Converged.")
                break

            # Store for next iteration
            prev_grads = flat_grads
            prev_dirs = direction

        print("Training complete.")
        return mi_list