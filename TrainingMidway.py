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
from MarkovComputations import WeightMatrix, InputData, get_input_inds, get_output_inds, random_initial_parameters, compute_error, downsample_avg, load_and_format_mnist, load_and_format_iris, evaluate_accuracy, evaluate_accuracy_per_class


#########################################################
################  Parameter definitions #################
#########################################################

random.seed(args.param2)

### Define parameters of classification
M = args.param1 # how many edges affected per input dimension

# n_classes = 5 # D, how many classes

classes = [0,1,6,7,8]
#classes = [0,1,2,3,4,5,6,7,8,9]
n_classes = len(classes)

input_dim = 14**2 # D, how many components of each input data
#input_dim = 4

### Define parameters of graph object and initial weights
n_nodes = 80 # assuming a complete graph
E_range = 1 # range of uniform distribution for Ej, etc.
B_range = 1
F_range = 1

### Define parameters of trainig
n_training_iters = 1500 # how many training steps to take
eta = 1.5 # learning rate (increment of Ej, Bij, Fij)
delta_E = 2 # nuding factor (change in Ej at output nodes during nudging)

############################################################
################  Initialize weight_matrix #################
############################################################

## create graph
g = nx.complete_graph(n_nodes) # assuming a complete graph for now - easy to change
n_edges = len(list(g.edges())) 
print("Fraction of edges with inputs applied:", M * input_dim / n_edges)


## get input and output inds
output_inds = get_output_inds(n_nodes, n_classes, False)
input_inds = get_input_inds(n_edges, input_dim, M)

## initialize first edge rates
Ej_list, Bij_list, Fij_list = random_initial_parameters(E_range, B_range, F_range, n_nodes, n_edges)

# Create WeightMatrix object
weight_matrix = WeightMatrix(g, Ej_list, Bij_list, Fij_list)
weight_matrix.lower_output_energies(output_inds, 4) # lower energies at the output nodes to ease training


############################################################
################  Load classification data #################
############################################################

input_data = load_and_format_mnist(classes, 10, 2)
# input_data = load_and_format_mnist(n_classes, 10, 2)


################################################
################  Run training #################
################################################

weight_matrix_orig = copy.deepcopy(weight_matrix) # save original rate matrix 

error_list = [] # track errors during training
accuracy_list = [] # track errors during training
accuracy_stride = 10
delE_stride = 100
delE = 0 * 0.02

print("Starting training.")

start_time = time.time()
for training_iter in range(n_training_iters):

    class_number = random.randrange(n_classes) # draw a random class label to present
    # inputs = next(input_data.training_data[class_number]) # get the next data point from the iterator for this class

    try:
        inputs = next(input_data.training_data[class_number])
    except StopIteration:
        input_data.refill_iterators()  # Refill iterators if exhausted
        inputs = next(input_data.training_data[class_number])  # Try again


    ss = weight_matrix.compute_ss_on_inputs(input_inds, inputs) # apply the data as input and get the steady state
    error_list.append(np.linalg.norm(compute_error(ss, input_data, class_number, output_inds))) # save error of this iteration
    
    if (training_iter % accuracy_stride == 0): # compute accuracy for list
        accuracy_list.append(evaluate_accuracy(weight_matrix, input_inds, input_data, output_inds, n_classes, 100))
    
    nudged_ss = weight_matrix.compute_nudged_ss(input_data, class_number, input_inds, inputs, output_inds, delta_E) # get the steady state after applying a nudge on this input
    ss_frac = nudged_ss / ss # compute pi_nudge / pi
    
    weight_matrix.update(input_inds, inputs, ss_frac, eta) # update the weight_matrix

    if (training_iter % delE_stride == 0):
        weight_matrix.set_W_mat(weight_matrix.Ej_list - delE, weight_matrix.Bij_list, weight_matrix.Fij_list)
    

end_time = time.time()
print(f"Execution Time: {end_time - start_time:.6f} seconds")

   # Save to a file
with open(output_dir + "/SavedData.pkl", "wb") as file:
    pickle.dump((weight_matrix, input_data, accuracy_list, input_inds, output_inds), file)

print("Data saved successfully.")
    
