#################################################
################  Import things #################
#################################################

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
import time
import random
import copy
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn import datasets
import pickle
import argparse


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output


## here are the user-defined functions and classes
from MarkovComputations import StackedWeightMatrices, WeightMatrix, InputData, get_input_inds, get_output_inds, random_initial_parameters, compute_error, downsample_avg, load_and_format_mnist, load_and_format_iris, evaluate_accuracy, evaluate_accuracy_stacked, evaluate_accuracy_per_class


#########################################################
################  Parameter definitions #################
#########################################################

#random.seed(args.param1)
random.seed(10)

### Define parameters of classification
M = args.param1 # how many edges affected per input dimension
#M = 3

# n_classes = 5 # D, how many classes

classes = [0,1,6,7,8]
#classes = [0,7]
#classes = [0,1,2,3,4,5,6,7,8,9]
n_classes = len(classes)

#input_dim = 14**2 # D, how many components of each input data

#n_classes = 2
input_dim = 14**2

### Define parameters of graph object and initial weights
n_nodes = 50 # assuming a complete graph
E_range = 0 # range of uniform distribution for Ej, etc.
B_range = 0
F_range = 0

#dim = args.param1
dim = 20
L = 2
external_input_dim = input_dim
external_output_dim = n_classes

if L == 2:
    internal_input_dims = [dim+2]
    internal_output_dims = [dim]
    M_vals = [M for l in range(L)]
    n_nodes_list = [n_nodes for l in range(L)]

if L == 1:
    internal_input_dims = []
    internal_output_dims = []
    M_vals = [M for l in range(L)]
    n_nodes_list = [n_nodes for l in range(L)]

A_fac = 20
b_fac = 0

### Define parameters of trainig
n_training_iters = 1000 # how many training steps to take
eta = 2 # learning rate (increment of Ej, Bij, Fij)


rand_output_bool = False

####################################################################
################  Initialize stacked weight_matrix #################
####################################################################

weight_matrix_list = []
for l in range(L):
    n_nodes = n_nodes_list[l]
    g = nx.complete_graph(n_nodes)
    n_edges = len(list(g.edges())) 
    Ej_list, Bij_list, Fij_list = random_initial_parameters(E_range, B_range, F_range, n_nodes, n_edges)
    weight_matrix_list.append(WeightMatrix(g, Ej_list, Bij_list, Fij_list))

    
external_input_inds = get_input_inds(n_edges, input_dim, M)
stacked_weight_matrices = StackedWeightMatrices(weight_matrix_list, 
                                                [external_input_dim, external_output_dim],
                                                [internal_input_dims, internal_output_dims],
                                                M_vals, A_fac, b_fac, rand_output_bool)

# stacked_weight_matrices.weight_matrix_list[-1].lower_output_energies(stacked_weight_matrices.external_output_inds, 4) # lower energies at the output nodes to ease training


############################################################
################  Load classification data #################
############################################################

input_data = load_and_format_mnist(classes, 10, 2)
#input_data = load_and_format_mnist(n_classes, 10, 2)

######  Gaussian example
n_samples = 20000

## high-dimensional example
mu_1 = -10 * np.ones(input_dim)
cov_1 = 1.0 * np.diag(np.ones(input_dim))
dist_1 = np.random.multivariate_normal(mu_1, cov_1, n_samples)

mu_2 = 10 * np.ones(input_dim)
cov_2 = 1.0 * np.diag(np.ones(input_dim))
dist_2 = np.random.multivariate_normal(mu_2, cov_2, n_samples)

#data_list = [[dat for dat in dist_1], [dat for dat in dist_2]]

###  create InputData object
#input_data = InputData(n_classes, data_list)



################################################
################  Run training #################
################################################


error_list = [] # track errors during training
accuracy_list = [] # track errors during training
accuracy_stride = 20

print("Starting training.")

start_time = time.time()
for training_iter in range(n_training_iters):

    class_number = random.randrange(n_classes) # draw a random class label to present

    try:
        inputs = next(input_data.training_data[class_number])
    except StopIteration:
        input_data.refill_iterators()  # Refill iterators if exhausted
        inputs = next(input_data.training_data[class_number])  # Try again

    ss_list, inputs_list = stacked_weight_matrices.compute_stacked_ss_on_inputs(inputs)
    error_list.append(np.linalg.norm(compute_error(ss_list[-1], input_data, class_number, stacked_weight_matrices.external_output_inds))) # save error of this iteration

    if (training_iter % accuracy_stride == 0): # compute accuracy for list
        accuracy_list.append(evaluate_accuracy_stacked(stacked_weight_matrices, input_data, n_classes, 100))

    
    stacked_weight_matrices.update(ss_list, inputs_list, class_number, eta) # update the weight_matrices

    

end_time = time.time()
print(f"Execution Time: {end_time - start_time:.6f} seconds")

# Save to a file
with open(output_dir + "/SavedData.pkl", "wb") as file:
    pickle.dump((stacked_weight_matrices, input_data, accuracy_list, error_list), file)

print("Data saved successfully.")
    
