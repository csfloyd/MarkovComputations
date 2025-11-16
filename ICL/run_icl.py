#################################################
################  Import things #################
#################################################


import torch
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader
import os
import time

# Import from refactored modular structure
from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from models import MatrixTreeMarkovICL
from training import train_model
from evaluation import test_icl


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output


#### discrete test
# Pre-define all parameters
L = 64
K = args.param2
#K = 1024
#D = 2
D = args.param1
N = 8
#N = args.param1
B = 1
#n_nodes = args.param2
n_nodes = 8
epochs = 1000
lr = 0.0025
batch_size = 64
#train_samples = 200000
#val_samples = 4000
train_samples = 100000
val_samples = 2000
epsilon = 1e-3
seed = 22
exact_copy = True
method = 'direct_solve'
temperature = 1.0
shuffle_context = True
learn_base_rates = False
offset = 0.0
min_max_choice = None
unique_labels = False

# Set parameters
params = {
    'K': K,                      # Number of GMM classes
    'L': L,                      # Number of output classes (can be different from K)
    'D': D,                      # Dimension
    'N': N,                      # Context examples
    'B': B,                      # Burstiness
    'n_nodes': n_nodes,          # Markov nodes
    'epochs': epochs,            # Training epochs
    'lr': lr,                    # Learning rate
    'batch_size': batch_size,
    'train_samples': train_samples,
    'val_samples': val_samples,
    'epsilon': epsilon,          # Within-class noise
    'seed': seed,
    'exact_copy': exact_copy,    # Query is exact copy of context item
    'method': method,
    'temperature': temperature,   # Softmax temperature
    'shuffle_context': shuffle_context,
    'learn_base_rates': learn_base_rates,
    'offset': offset,
    'min_max_choice': min_max_choice,
    'unique_labels' : unique_labels
}

print("="*70)
print("MARKOV ICL - CLASSIFICATION (Softmax Output)")
print("="*70)
print(f"K={params['K']}, D={params['D']}, N={params['N']}, B={params['B']}, nodes={params['n_nodes']}")
print(f"Method: {params['method']}, Temperature: {params['temperature']}")
print("="*70)

# Set random seeds
torch.manual_seed(params['seed'])
np.random.seed(params['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create GMM with discrete labels (1 to L)
print("Creating GMM with discrete labels...")
gmm = GaussianMixtureModel(K=params['K'], D=params['D'], L=params['L'], epsilon=params['epsilon'], seed=params['seed'], offset=params['offset'])
print(f"  GMM: {params['K']} classes with labels randomly assigned from {{1, ..., {params['L']}}}")
print(f"  First 10 class labels: {gmm.class_to_label[:min(10, params['K'])].numpy()}")

# Generate data
print("\nGenerating data...")
train_data = generate_icl_gmm_data(gmm, params['train_samples'], params['N'], 
                                   novel_classes=False, exact_copy=params['exact_copy'], 
                                   B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'], 
                                 novel_classes=False, exact_copy=params['exact_copy'], 
                                 B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

# Create model
print("\nCreating model...")
model = MatrixTreeMarkovICL(n_nodes=params['n_nodes'], z_dim=params['D'], 
                           L=params['L'], N=params['N'], learn_base_rates=params['learn_base_rates'])

# Train with ICL/IWL tracking
start_time = time.time()
print("\nTraining...")
print("="*70)
history = train_model(model, train_loader, val_loader, device, 
                     n_epochs=params['epochs'], lr=params['lr'], 
                     method=params['method'], temperature=params['temperature'],
                     gmm=gmm, N=params['N'], B=params['B'], 
                     L=params['L'], exact_copy=params['exact_copy'],
                     eval_frequency=1, n_eval_samples=500, min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])
                     
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Test
results = test_icl(model, gmm, params['N'], device, n_samples=1000, 
                  exact_copy=params['exact_copy'], B=params['B'], 
                  method=params['method'], L=params['L'],
                  temperature=params['temperature'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])


# Save results
os.makedirs(output_dir, exist_ok=True)

# Save model weights (small, portable)
model_path = f'{output_dir}/model.pt'
torch.save(model.state_dict(), model_path)

# Save results and metadata (for analysis)
results_data = {
    'results': results,
    'history': history,
    'params': params,
    'execution_time': end_time - start_time
}
results_path = f'{output_dir}/results.pkl'
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

print(f"\n✓ Saved model to {model_path}")
print(f"✓ Saved results to {results_path}")
print(f"✓ Execution Time: {end_time - start_time:.2f} seconds")
    
