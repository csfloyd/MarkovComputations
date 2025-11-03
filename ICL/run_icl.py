#################################################
################  Import things #################
#################################################


import torch
import numpy as np
import pickle
import argparse
# NEW (classification version)
from markov_icl_gmm_discrete_linear import (
    GaussianMixtureModel, 
    generate_icl_gmm_data, 
    ICLGMMDataset,
    collate_fn,
    MatrixTreeMarkovICL,
    train_model,
    test_icl
)
from torch.utils.data import DataLoader
import os
import time


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
K_classes = 75
K = K_classes
D = 5
N = 10
B = 2
#n_nodes = 5
n_nodes = args.param1
epochs = 1000
lr = 0.001
batch_size = 64
train_samples = 10000
val_samples = 2000
epsilon = 0.1
seed = 42
exact_copy = True
method = 'direct_solve'
temperature = 1.0

# Set parameters
params = {
    'K': K,                      # Number of GMM classes
    'K_classes': K_classes,      # Number of output classes (can be different from K)
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
    'temperature': temperature   # Softmax temperature
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

# Create GMM with discrete labels (1 to K)
print("Creating GMM with discrete labels...")
gmm = GaussianMixtureModel(K=params['K'], D=params['D'], epsilon=params['epsilon'], seed=params['seed'])
print(f"  Class labels: {gmm.class_to_label[:min(10, params['K'])].numpy()}... (1 to {params['K']})")

# Generate data
print("\nGenerating data...")
train_data = generate_icl_gmm_data(gmm, params['train_samples'], params['N'], 
                                   novel_classes=False, exact_copy=params['exact_copy'], 
                                   B=params['B'], K_classes=params['K_classes'])
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'], 
                                 novel_classes=False, exact_copy=params['exact_copy'], 
                                 B=params['B'], K_classes=params['K_classes'])

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

# Create model
print("\nCreating model...")
model = MatrixTreeMarkovICL(n_nodes=params['n_nodes'], z_dim=params['D'], 
                           K_classes=params['K_classes'], N=params['N'])

# Train
start_time = time.time()
print("\nTraining...")
print("="*70)
history = train_model(model, train_loader, val_loader, device, 
                     n_epochs=params['epochs'], lr=params['lr'], 
                     method=params['method'], temperature=params['temperature'])
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Test
results = test_icl(model, gmm, params['N'], device, n_samples=1000, 
                  exact_copy=params['exact_copy'], B=params['B'], 
                  method=params['method'], K_classes=params['K_classes'],
                  temperature=params['temperature'])

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
    
