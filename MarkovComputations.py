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
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn import datasets


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
        x, *_ = jax.scipy.sparse.linalg.cg(A.T @ A, A.T @ zero_array, tol=1e-10, maxiter=100000)
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


def get_input_inds(n_edges, input_dim, M):
    """
    Selects random edges to receive inputs.

    Parameters:
    - n_edges: Total number of edges
    - input_dim: Number of input channels
    - M: Number of edges per input

    Returns:
    - List of selected edge indices per input channel
    """
    shuffled_inds = list(range(n_edges))
    random.shuffle(shuffled_inds)
    return [shuffled_inds[m * M:(m + 1) * M] for m in range(input_dim)]


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
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to range [0,1]
    x_train = scale * (x_train.astype(np.float32) / 255.0 - 0.0)
    x_test = scale * (x_test.astype(np.float32) / 255.0 - 0.0)

    # Combine train and test sets
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

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


def evaluate_accuracy(weight_matrix, input_inds, input_data, output_inds, n_classes, n_evals):
    accuracy = 0.0
    for n in range(n_evals):
        class_number = random.randrange(n_classes) # draw a random class label to present
        # inputs = next(input_data.training_data[class_number]) # get the next data point from the iterator for this class

        try:
            inputs = next(input_data.testing_data[class_number])
        except StopIteration:
            input_data.refill_iterators()  # Refill iterators if exhausted
            inputs = next(input_data.testing_data[class_number])  # Try again
        
        ss = weight_matrix.compute_ss_on_inputs(input_inds, inputs) # apply the data as input and get the steady state
        ss_at_outputs = [ss[out] for out in output_inds]
        if (np.argmax(ss_at_outputs) == class_number):
            accuracy += 1.0
        

    return accuracy / n_evals

def evaluate_accuracy_per_class(weight_matrix, input_inds, input_data, output_inds, n_evals, n_classes):
    predictions_per_class = np.zeros((n_classes, n_classes))

    for class_number in range(n_classes):

        for n in range(n_evals):
            try:
                inputs = next(input_data.testing_data[class_number])
            except StopIteration:
                input_data.refill_iterators()  # Refill iterators if exhausted
                inputs = next(input_data.testing_data[class_number])  # Try again
            
            ss = weight_matrix.compute_ss_on_inputs(input_inds, inputs) # apply the data as input and get the steady state
            ss_at_outputs = [ss[out] for out in output_inds]

            pred = np.argmax(ss_at_outputs) 
            predictions_per_class[pred][class_number] += 1.0    

    return predictions_per_class






