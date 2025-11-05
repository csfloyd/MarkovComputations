"""
Main experiment runner for ICL models.

This script orchestrates data generation, model creation, training, and evaluation.
"""

import torch
import numpy as np
import argparse
import pickle
import os
import time
from torch.utils.data import DataLoader

from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from models import MatrixTreeMarkovICL
from training import train_model
from evaluation import test_icl
from config import ExperimentConfig, create_config_from_args


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ICL experiments with different model architectures."
    )
    
    # Data parameters
    parser.add_argument('--K', type=int, default=75, help='Number of GMM classes')
    parser.add_argument('--K_classes', type=int, default=None, help='Number of output classes')
    parser.add_argument('--D', type=int, default=5, help='Feature dimension')
    parser.add_argument('--N', type=int, default=10, help='Context examples')
    parser.add_argument('--B', type=int, default=2, help='Burstiness')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Within-class noise')
    
    # Model parameters
    parser.add_argument('--n_nodes', type=int, default=15, help='Number of nodes')
    parser.add_argument('--model_type', type=str, default='markov', 
                       choices=['markov', 'polynomial'], help='Model architecture')
    parser.add_argument('--use_label_mod', action='store_true', help='Use label modulation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_samples', type=int, default=10000, help='Train samples')
    parser.add_argument('--val_samples', type=int, default=2000, help='Val samples')
    
    # Evaluation parameters
    parser.add_argument('--exact_copy', action='store_true', default=True)
    parser.add_argument('--no_exact_copy', dest='exact_copy', action='store_false')
    parser.add_argument('--method', type=str, default='direct_solve',
                       choices=['matrix_tree', 'linear_solver', 'direct_solve'],
                       help='Steady state computation method')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature')
    parser.add_argument('--eval_frequency', type=int, default=10, help='ICL/IWL eval frequency')
    parser.add_argument('--n_eval_samples', type=int, default=500, help='Samples per eval')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def create_model(config):
    """
    Create model based on configuration.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        ICL model instance
    """
    if config.model_type == 'markov':
        model = MatrixTreeMarkovICL(
            n_nodes=config.n_nodes,
            z_dim=config.D,
            K_classes=config.K_classes,
            N=config.N,
            use_label_mod=config.use_label_mod
        )
    elif config.model_type == 'polynomial':
        # Placeholder for future polynomial model
        raise NotImplementedError("Polynomial model not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model


def main():
    """Main experiment execution."""
    args = parse_args()
    config = create_config_from_args(args)
    
    print("="*70)
    print("ICL EXPERIMENT")
    print("="*70)
    print(config)
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # === Create GMM ===
    print("\nCreating Gaussian Mixture Model...")
    gmm = GaussianMixtureModel(
        K=config.K,
        D=config.D,
        epsilon=config.epsilon,
        seed=config.seed
    )
    print(f"  GMM: {config.K} classes, {config.D} dimensions")
    print(f"  Labels: {gmm.class_to_label[:min(10, config.K)].numpy()}... (1 to {config.K})")
    
    # === Generate Data ===
    print("\nGenerating training and validation data...")
    train_data = generate_icl_gmm_data(
        gmm, config.train_samples, config.N,
        novel_classes=False, exact_copy=config.exact_copy,
        B=config.B, K_classes=config.K_classes
    )
    val_data = generate_icl_gmm_data(
        gmm, config.val_samples, config.N,
        novel_classes=False, exact_copy=config.exact_copy,
        B=config.B, K_classes=config.K_classes
    )
    
    train_loader = DataLoader(
        ICLGMMDataset(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        ICLGMMDataset(val_data),
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    
    # === Create Model ===
    print("\nCreating model...")
    model = create_model(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"\n✓ Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print("✓ Checkpoint loaded successfully!")
    
    # === Train ===
    print("\nTraining...")
    print("="*70)
    start_time = time.time()
    
    history = train_model(
        model, train_loader, val_loader, device,
        n_epochs=config.epochs,
        lr=config.lr,
        method=config.method,
        temperature=config.temperature,
        gmm=gmm,
        N=config.N,
        B=config.B,
        K_classes=config.K_classes,
        exact_copy=config.exact_copy,
        eval_frequency=config.eval_frequency,
        n_eval_samples=config.n_eval_samples
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")
    
    # === Test ===
    print("\nTesting...")
    results = test_icl(
        model, gmm, config.N, device,
        n_samples=1000,
        exact_copy=config.exact_copy,
        B=config.B,
        method=config.method,
        K_classes=config.K_classes,
        temperature=config.temperature
    )
    
    # === Save Results ===
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save results and metadata
    results_data = {
        'results': results,
        'history': history,
        'config': config.to_dict(),
        'execution_time': training_time
    }
    results_path = os.path.join(output_dir, 'results.pkl')
    with open(results_path, "wb") as f:
        pickle.dump(results_data, f)
    
    # Save config separately as JSON for easy reading
    import json
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"\n✓ Saved model to {model_path}")
    print(f"✓ Saved results to {results_path}")
    print(f"✓ Saved config to {config_path}")
    print(f"✓ Execution Time: {training_time:.2f} seconds")
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

