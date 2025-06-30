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
from MarkovComputations import *


#########################################################
################  Parameter definitions #################
#########################################################

restart_bool = False

#random.seed(args.param1)
random.seed(10)

### Define parameters of classification
#M = 10 # how many edges affected per input dimension
#M = 3
M = args.param1
# n_classes = 5 # D, how many classes

classes = [0,1,6,7,8]
#classes = [0,7,8]
classes = [0,1,2,3,4,5,6,7,8,9]
n_classes = len(classes)

#input_dim = 14**2 # D, how many components of each input data

#n_classes = 2
input_dim = 14**2

### Define parameters of graph object and initial weights
#n_nodes = 75 # assuming a complete graph
n_nodes = 30
E_range = 0 # range of uniform distribution for Ej, etc.
B_range = 0
F_range = 0

#dim = args.param1
#dim = 50
dim = 30
L = 1
external_input_dim = input_dim
external_output_dim = n_classes
rand_output_bool = False

if L == 2:
    internal_input_dims = [dim+10]
    internal_output_dims = [dim]
    M_vals = [M for l in range(L)]
    n_nodes_list = [n_nodes for l in range(L)]

if L == 1:
    internal_input_dims = []
    internal_output_dims = []
    M_vals = [M for l in range(L)]
    n_nodes_list = [n_nodes for l in range(L)]

A_fac = 10
b_fac = 0


############################################################
################  Load classification data #################
############################################################

input_data = load_and_format_mnist(classes, 10, 2)


####################################################################
################  Initialize stacked weight_matrix #################
####################################################################

if not restart_bool:
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
else:
    rep_dir = append_r_before_slash(output_dir, id = -1)
    with open(rep_dir + "/SavedData.pkl", "rb") as file:
        stacked_weight_matrices, input_data, accuracy_list, error_list = pickle.load(file)


################################################
################  Run training #################
################################################

### Define parameters of trainig
n_training_iters = 200  # how many training steps to take
n_samples = 100
accuracy_stride = 20


print("Starting training.")

start_time = time.time()

mi_list = train_mi_conjugate_gradient(
    stacked_weight_matrices,
    input_data,
    n_nodes,
    n_classes,
    n_samples=n_samples,
    n_epochs=n_training_iters,
    step_size=1e1,
    tol=1e-6,
    verbose=False
)
    

end_time = time.time()
print(f"Execution Time: {end_time - start_time:.6f} seconds")

# Save to a file
with open(output_dir + "/SavedData.pkl", "wb") as file:
    pickle.dump((stacked_weight_matrices, input_data, mi_list), file)

print("Data saved successfully.")
    
