"""
Markov ICL with GMM using Matrix Tree Theorem (Direct K matrix formulation)

This implementation uses the rate matrix K where:
- K_ij ≥ 0 for i ≠ j (transition rates from state j to state i)
- Columns sum to zero: Σᵢ K_ij = 0
- Steady state satisfies: K p = 0

The Matrix Tree Theorem gives:
    p_i = det(K^(i)) / Σⱼ det(K^(j))
    
where K^(i) is obtained by deleting row i and column i from K.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse


# ==================== GMM DATA GENERATION ====================

class GaussianMixtureModel:
    """Gaussian Mixture Model with K classes for ICL task."""
    
    def __init__(self, K, D, L=None, epsilon=0.1, seed=None, label_min=0.0, label_max=1.0):
        self.K = K
        self.D = D
        self.L = L if L is not None else K
        self.epsilon = epsilon
        self.label_min = label_min
        self.label_max = label_max
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.class_means = torch.randn(K, D) / np.sqrt(D)
        self.class_to_label = torch.rand(K) * (label_max - label_min) + label_min
        
    def sample_from_class(self, class_idx, n_samples=1):
        mu_k = self.class_means[class_idx]
        noise = torch.randn(n_samples, self.D) / np.sqrt(self.D)
        return mu_k + self.epsilon * noise
    
    def get_label(self, class_idx):
        return self.class_to_label[class_idx]


def generate_icl_gmm_data(gmm, n_samples, N, novel_classes=False, exact_copy=True, B=1, 
                          label_min=None, label_max=None):
    """Generate ICL data from GMM."""
    assert 1 <= B <= N and N % B == 0
    n_classes_in_context = N // B
    lmin = label_min if label_min is not None else gmm.label_min
    lmax = label_max if label_max is not None else gmm.label_max
    data = []
    
    for _ in range(n_samples):
        if novel_classes:
            if B == 1:
                novel_means = torch.randn(N, gmm.D) / np.sqrt(gmm.D)
                novel_labels = torch.rand(N) * (lmax - lmin) + lmin
                z_context = []
                labels = []
                for i in range(N):
                    noise = torch.randn(gmm.D) / np.sqrt(gmm.D)
                    z_context.append(novel_means[i] + gmm.epsilon * noise)
                    labels.append(novel_labels[i])
                copy_idx = torch.randint(0, N, (1,)).item()
                if exact_copy:
                    z_query = z_context[copy_idx].clone()
                else:
                    z_query = novel_means[copy_idx] + gmm.epsilon * torch.randn(gmm.D) / np.sqrt(gmm.D)
                target_label = novel_labels[copy_idx]
            else:
                novel_means = torch.randn(n_classes_in_context, gmm.D) / np.sqrt(gmm.D)
                novel_labels = torch.rand(n_classes_in_context) * (lmax - lmin) + lmin
                z_context = []
                labels = []
                for class_idx in range(n_classes_in_context):
                    for _ in range(B):
                        noise = torch.randn(gmm.D) / np.sqrt(gmm.D)
                        z_context.append(novel_means[class_idx] + gmm.epsilon * noise)
                        labels.append(novel_labels[class_idx])
                query_class_idx = torch.randint(0, n_classes_in_context, (1,)).item()
                if exact_copy:
                    copy_offset = torch.randint(0, B, (1,)).item()
                    z_query = z_context[query_class_idx * B + copy_offset].clone()
                else:
                    z_query = novel_means[query_class_idx] + gmm.epsilon * torch.randn(gmm.D) / np.sqrt(gmm.D)
                target_label = novel_labels[query_class_idx]
        else:
            if B == 1:
                class_indices = torch.randint(0, gmm.K, (N,))
                z_context = []
                labels = []
                for i in range(N):
                    z_context.append(gmm.sample_from_class(class_indices[i].item()).squeeze(0))
                    labels.append(gmm.get_label(class_indices[i].item()))
                copy_idx = torch.randint(0, N, (1,)).item()
                query_class = class_indices[copy_idx].item()
                if exact_copy:
                    z_query = z_context[copy_idx].clone()
                else:
                    z_query = gmm.sample_from_class(query_class).squeeze(0)
                target_label = gmm.get_label(query_class)
            else:
                context_classes = torch.randint(0, gmm.K, (n_classes_in_context,))
                z_context = []
                labels = []
                for class_idx in context_classes:
                    class_label = gmm.get_label(class_idx.item())
                    for _ in range(B):
                        z_context.append(gmm.sample_from_class(class_idx.item()).squeeze(0))
                        labels.append(class_label)
                query_class_position = torch.randint(0, n_classes_in_context, (1,)).item()
                query_class = context_classes[query_class_position].item()
                if exact_copy:
                    copy_offset = torch.randint(0, B, (1,)).item()
                    z_query = z_context[query_class_position * B + copy_offset].clone()
                else:
                    z_query = gmm.sample_from_class(query_class).squeeze(0)
                target_label = gmm.get_label(query_class)
        
        z_seq = torch.stack(z_context + [z_query])
        data.append((z_seq, torch.tensor(labels), target_label))
    
    return data


class ICLGMMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    z_seqs = torch.stack([item[0] for item in batch])
    labels_seqs = torch.stack([item[1] for item in batch])
    targets = torch.tensor([item[2] for item in batch])
    return z_seqs, labels_seqs, targets


def test_icl(model, gmm, N, device, n_samples=1000, exact_copy=True, B=1, test_label_shifts=False, method='matrix_tree'):
    """Test in-context learning on novel classes."""
    model.eval()
    
    print("\n" + "="*70)
    print("TESTING IN-CONTEXT LEARNING")
    print("="*70)
    
    print("\n1. In-Distribution Test:")
    test_data_id = generate_icl_gmm_data(gmm, n_samples, N, False, exact_copy, B)
    errors_id = []
    with torch.no_grad():
        for z_seq, labels, target in test_data_id:
            pred = model(z_seq.unsqueeze(0).to(device), labels.unsqueeze(0).to(device), method=method).item()
            errors_id.append(abs(pred - target.item()))
    mae_id = np.mean(errors_id)
    print(f"   MAE: {mae_id:.4f}")
    
    print("\n2. Out-of-Distribution Test (TRUE ICL):")
    test_data_ood = generate_icl_gmm_data(gmm, n_samples, N, True, exact_copy, B)
    errors_ood = []
    with torch.no_grad():
        for z_seq, labels, target in test_data_ood:
            pred = model(z_seq.unsqueeze(0).to(device), labels.unsqueeze(0).to(device), method=method).item()
            errors_ood.append(abs(pred - target.item()))
    mae_ood = np.mean(errors_ood)
    print(f"   MAE: {mae_ood:.4f}")
    
    print("\n" + "="*70)
    print(f"  In-Distribution:  {mae_id:.4f}")
    print(f"  Novel Classes:    {mae_ood:.4f}")
    
    if mae_ood < 0.10:
        print("  ✓ SUCCESS: ICL working!")
    elif mae_ood < 0.15:
        print("  ○ PARTIAL ICL")
    else:
        print("  ✗ Needs more training")
    print("="*70)
    
    return {'in_dist': mae_id, 'novel_classes': mae_ood}

class MatrixTreeMarkovICL(nn.Module):
    """
    Matrix Tree Theorem implementation using rate matrix K.
    
    K is the master equation rate matrix where:
    - K[i,j] = rate from state j to state i (for i≠j)
    - K[j,j] = -Σ_{k≠j} K[k,j] (negative sum of column j)
    - Columns sum to zero: Σᵢ K[i,j] = 0
    
    Steady state p satisfies: K p = 0
    
    Matrix Tree Theorem: p_i = det(K^(i)) / Σⱼ det(K^(j))
    where K^(i) is K with row i and column i deleted.
    """
    
    def __init__(self, n_nodes=10, z_dim=2, n_labels=1, N=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.z_dim = z_dim
        self.n_labels = n_labels
        self.N = N
        
        z_full_dim = (N + 1) * z_dim
        l_full_dim = N
        
        # Initialize parameters with proper scaling
        init_scale_K = 0.05 / np.sqrt(n_nodes)
        init_scale_B = 0.1 / np.sqrt(n_nodes)
        init_base = -2.0 - 0.5 * np.log(n_nodes)
        
        # Learnable parameters for rate matrix
        self.K_params = nn.Parameter(torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K)
        self.B = nn.Parameter(torch.randn(n_nodes, n_labels, l_full_dim) * init_scale_B)
        self.base_log_rates = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1 + init_base)
        
        print(f"  Initialized Matrix Tree method (rate matrix K formulation)")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def compute_rate_matrix_K(self, z_batch):
        """
        Compute the rate matrix K where columns sum to zero.
        
        K[i,j] = exp(base[i,j] + K_params[i,j] · z) for i≠j
        K[j,j] = -Σ_{k≠j} K[k,j]
        
        Args:
            z_batch: (batch_size, z_dim)
        Returns:
            K_batch: (batch_size, n_nodes, n_nodes) with columns summing to zero
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        
        # Compute modulation: K_params · z
        K_expanded = self.K_params.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rate_mod = torch.einsum('bijd,bd->bij', K_expanded, z_batch)
        
        # Add base rates
        base_expanded = self.base_log_rates.unsqueeze(0).expand(batch_size, -1, -1)
        log_rates = base_expanded + rate_mod
        
        # Clamp for numerical stability
        log_rates = torch.clamp(log_rates, min=-15.0, max=3.0)
        
        # Exponentiate to get rates
        rates = torch.exp(log_rates)
        
        # Zero out diagonal (we'll set it later)
        eye = torch.eye(n, device=rates.device).unsqueeze(0)
        rates = rates * (1 - eye)
        
        # Construct K with proper diagonal so columns sum to zero
        # K[j,j] = -Σ_{k≠j} K[k,j] = -Σ_k K[k,j] (since diagonal is currently zero)
        col_sums = rates.sum(dim=1)  # Sum over first index (rows)
        
        # Set diagonal: K[j,j] = -col_sums[j]
        K_batch = rates - torch.diag_embed(col_sums)
        
        # Verify columns sum to zero (for debugging)
        # col_check = K_batch.sum(dim=1)  # Should be all zeros
        
        return K_batch
    
    def matrix_tree_steady_state(self, K_batch):
        """
        Compute steady state using Matrix Tree Theorem.
        
        For rate matrix K with columns summing to zero,
        the steady state p satisfying K p = 0 is given by:
        
            p_i = det(K^(i)) / Σⱼ det(K^(j))
        
        where K^(i) is K with row i and column i deleted.
        
        Args:
            K_batch: (batch_size, n_nodes, n_nodes)
        Returns:
            p_batch: (batch_size, n_nodes)
        """
        batch_size, n = K_batch.shape[0], self.n_nodes
        device = K_batch.device
        
        # Compute determinants of all minors
        p_batch = torch.zeros(batch_size, n, device=device)
        
        for i in range(n):
            # Delete row i and column i
            indices = [j for j in range(n) if j != i]
            K_minor = K_batch[:, indices, :][:, :, indices]  # (batch, n-1, n-1)
            
            # Compute determinant
            # Use abs() since determinants should be positive (numerically may have sign issues)
            det = torch.det(K_minor)
            det = torch.abs(det)
            
            # Clamp for numerical stability
            det = torch.clamp(det, min=1e-10, max=1e10)
            
            p_batch[:, i] = det
        
        # Normalize: p_i = det(K^(i)) / Σⱼ det(K^(j))
        Z = p_batch.sum(dim=1, keepdim=True)
        Z = torch.clamp(Z, min=1e-8)
        p_batch = p_batch / Z
        
        # Handle NaN/Inf (fallback to uniform)
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / n
        
        return p_batch

    def linear_solver_steady_state(self, K_batch):
        """
        Compute steady state using linear solver (more efficient than Matrix Tree Theorem).
        
        Solves the augmented system:
            [K      ]     [0]
            [1,1,...] p = [1]
        
        Where K p = 0 (steady state condition) and sum(p) = 1 (normalization).
        
        Args:
            K_batch: (batch_size, n_nodes, n_nodes) - rate matrices with columns summing to zero
        Returns:
            p_batch: (batch_size, n_nodes) - steady state distributions
        """
        batch_size, n = K_batch.shape[0], self.n_nodes
        device = K_batch.device
        
        # Augment system: add constraint that probabilities sum to 1
        # A = [K; ones(1, n)]  shape: (n+1, n)
        ones_row = torch.ones(batch_size, 1, n, device=device)
        A_augmented = torch.cat([K_batch, ones_row], dim=1)  # (batch, n+1, n)
        
        # Target: [0, 0, ..., 0, 1]^T
        b = torch.zeros(batch_size, n + 1, device=device)
        b[:, -1] = 1.0  # Last element = 1 for normalization constraint
        
        # Solve using least squares: A^T A p = A^T b
        # This is more stable than solving A p = b directly for overdetermined systems
        try:
            # Method 1: Use torch.linalg.lstsq (recommended)
            p_batch = torch.linalg.lstsq(A_augmented, b.unsqueeze(-1)).solution.squeeze(-1)
            
        except RuntimeError:
            # Fallback: Manual normal equations with regularization
            AtA = torch.bmm(A_augmented.transpose(1, 2), A_augmented)
            # Add small regularization for numerical stability
            AtA = AtA + 1e-6 * torch.eye(n, device=device).unsqueeze(0)
            Atb = torch.bmm(A_augmented.transpose(1, 2), b.unsqueeze(-1))
            p_batch = torch.linalg.solve(AtA, Atb).squeeze(-1)
        
        # Ensure non-negativity and normalization
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / (p_batch.sum(dim=1, keepdim=True) + 1e-8)
        
        # Handle NaN/Inf (fallback to uniform)
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / n
        
        return p_batch

    
    def direct_solve_steady_state(self, K_batch):
        """Replace last row of K with normalization constraint."""
        batch_size, n = K_batch.shape[0], self.n_nodes
        device = K_batch.device
        
        # Modify K: replace last row with [1, 1, 1, ..., 1]
        K_modified = K_batch.clone()
        K_modified[:, -1, :] = 1.0
        
        # RHS: [0, 0, ..., 0, 1]
        b = torch.zeros(batch_size, n, device=device)
        b[:, -1] = 1.0
        
        # Solve K_modified @ p = b
        p_batch = torch.linalg.solve(K_modified, b)
        
        # Ensure non-negativity
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / p_batch.sum(dim=1, keepdim=True)
        
        return p_batch
    
    def forward(self, z_seq_batch, labels_seq_batch, method='matrix_tree'):
        """
        Forward pass.
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N)
        Returns:
            output: (batch_size,)
        """
        batch_size = z_seq_batch.shape[0]
        
        # Flatten z sequences
        z_flat = z_seq_batch.reshape(batch_size, -1)
        
        # Compute rate matrix K (columns sum to zero)
        K_batch = self.compute_rate_matrix_K(z_flat)
        
        # Compute steady state via Matrix Tree Theorem
        if method == 'matrix_tree':
            p_batch = self.matrix_tree_steady_state(K_batch)
        elif method == 'linear_solver':
            p_batch = self.linear_solver_steady_state(K_batch)
        elif method == 'direct_solve':
            p_batch = self.direct_solve_steady_state(K_batch)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        # Compute output: Σᵢ p_i * (B_i · labels)
        B_expanded = self.B.unsqueeze(0).expand(batch_size, -1, -1, -1)
        gamma_batch = torch.einsum('bijd,bd->bij', B_expanded, labels_seq_batch)
        output = torch.einsum('bi,bij->bj', p_batch, gamma_batch)
        
        return output.squeeze(1)


# Training function (same as before)
def train_model(model, train_loader, val_loader, device, n_epochs=200, lr=0.001, method='matrix_tree'):
    """Train the model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_errors = []
        
        for z_seq, labels, targets in train_loader:
            z_seq = z_seq.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            pred = model(z_seq, labels, method)
            loss = torch.abs(pred - targets).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_errors.append(torch.abs(pred - targets).detach().cpu().numpy())
        
        # Validation
        model.eval()
        val_losses = []
        val_errors = []
        
        with torch.no_grad():
            for z_seq, labels, targets in val_loader:
                z_seq = z_seq.to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                
                pred = model(z_seq, labels, method)
                loss = torch.abs(pred - targets).mean()
                
                val_losses.append(loss.item())
                val_errors.append(torch.abs(pred - targets).cpu().numpy())
        
        train_mae = np.mean(np.concatenate(train_errors))
        val_mae = np.mean(np.concatenate(val_errors))
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_mae:.4f} | Val: {val_mae:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=75, help='Number of GMM classes')
    parser.add_argument('--D', type=int, default=8, help='Dimension')
    parser.add_argument('--N', type=int, default=6, help='Context examples')
    parser.add_argument('--B', type=int, default=2, help='Burstiness')
    parser.add_argument('--n_nodes', type=int, default=15, help='Markov nodes')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_samples', type=int, default=10000, help='Train samples')
    parser.add_argument('--val_samples', type=int, default=2000, help='Val samples')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Within-class noise')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exact_copy', action='store_true', default=True)
    parser.add_argument('--no_exact_copy', dest='exact_copy', action='store_false')
    parser.add_argument('--test_label_shifts', action='store_true', help='Test label shifts')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--method', type=str, default='matrix_tree', help='Method to use for steady state computation')
    args = parser.parse_args()
    
    print("="*70)
    print("MARKOV ICL - MATRIX TREE THEOREM (K matrix formulation)")
    print("="*70)
    print(f"K={args.K}, D={args.D}, N={args.N}, B={args.B}, nodes={args.n_nodes}")
    print("="*70)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create GMM
    print("Creating GMM...")
    gmm = GaussianMixtureModel(K=args.K, D=args.D, epsilon=args.epsilon, seed=args.seed)
    
    # Generate data
    print("Generating data...")
    train_data = generate_icl_gmm_data(gmm, args.train_samples, args.N, 
                                       novel_classes=False, exact_copy=args.exact_copy, B=args.B)
    val_data = generate_icl_gmm_data(gmm, args.val_samples, args.N, 
                                     novel_classes=False, exact_copy=args.exact_copy, B=args.B)
    
    train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=args.batch_size,
                           collate_fn=collate_fn)
    
    # Create model
    print("\nCreating model...")
    model = MatrixTreeMarkovICL(n_nodes=args.n_nodes, z_dim=args.D, n_labels=1, N=args.N)
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"\n✓ Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print("✓ Checkpoint loaded successfully!")
    
    # Train
    print("\nTraining...")
    print("="*70)
    history = train_model(model, train_loader, val_loader, device, 
                         n_epochs=args.epochs, lr=args.lr, method=args.method)
    
    # Test
    results = test_icl(model, gmm, args.N, device, n_samples=1000, 
                      exact_copy=args.exact_copy, B=args.B, test_label_shifts=args.test_label_shifts, method=args.method)
    
    # Verification: Compare MTT vs Direct Eigenvalue Method
    print("\n" + "="*70)
    print("VERIFICATION: Matrix Tree Theorem vs Direct Eigenvalue")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Create a random test example from the GMM
        test_data = generate_icl_gmm_data(gmm, 1, args.N, novel_classes=True, 
                                         exact_copy=args.exact_copy, B=args.B)
        z_seq, labels_seq, target = test_data[0]
        z_seq = z_seq.unsqueeze(0).to(device)  # [1, N+1, D]
        labels_seq = labels_seq.unsqueeze(0).to(device)  # [1, N]
        
        # Extract context encoding (what model uses internally)
        z_flat = z_seq.flatten(start_dim=1)  # [1, (N+1)*D]
        l_flat = labels_seq.flatten(start_dim=1)  # [1, N]
        
        # Compute W matrix (rate matrix with columns summing to zero)
        # This replicates the model's forward pass logic
        base = model.base_log_rates.unsqueeze(0)  # [1, n, n]
        K_component = torch.einsum('ijk,bk->bij', model.K_params, z_flat)  # [1, n, n]
        
        # Compute rates: exp(base + K·z)
        rates = torch.exp(base + K_component)
        
        # Zero diagonal
        eye = torch.eye(model.n_nodes, device=device).unsqueeze(0)
        rates = rates * (1 - eye)
        
        # Construct W: columns sum to zero
        col_sums = rates.sum(dim=1, keepdim=True)  # [1, 1, n]
        W = rates - torch.diag_embed(col_sums.squeeze(1))  # [1, n, n]
        W = W.squeeze(0).cpu()  # [n, n] on CPU for eigenvalue computation
        
        # Method 1: Matrix Tree Theorem (as implemented in model)
        pi_mtt = torch.zeros(model.n_nodes)
        for i in range(model.n_nodes):
            indices = [j for j in range(model.n_nodes) if j != i]
            W_minor = W[indices, :][:, indices]
            det = torch.det(W_minor)
            pi_mtt[i] = abs(det.item())
        pi_mtt = pi_mtt / pi_mtt.sum()
        
        # Method 2: Direct Eigenvalue Decomposition
        # Find eigenvector with eigenvalue closest to 0
        eigenvalues, eigenvectors = torch.linalg.eig(W)
        eigenvalues = eigenvalues.real
        idx = torch.argmin(torch.abs(eigenvalues))
        pi_eig = eigenvectors[:, idx].real
        pi_eig = torch.abs(pi_eig)  # Take absolute values
        pi_eig = pi_eig / pi_eig.sum()  # Normalize
        
        print(f"\nSteady-state distribution (n={model.n_nodes} nodes):\n")
        print("Node |   MTT Method   | Eigenvalue Method |  Difference")
        print("-" * 60)
        for i in range(model.n_nodes):
            diff = abs(pi_mtt[i].item() - pi_eig[i].item())
            print(f" {i:2d}  |     {pi_mtt[i]:.6f}   |      {pi_eig[i]:.6f}    |   {diff:.8f}")
        
        max_diff = torch.max(torch.abs(pi_mtt - pi_eig)).item()
        print("-" * 60)
        print(f"Maximum difference: {max_diff:.10f}")
        
        if max_diff < 1e-5:
            print("✓ VERIFIED: MTT and Eigenvalue methods agree!")
        else:
            print("⚠ WARNING: Significant difference between methods")
    
    print("="*70 + "\n")
    
    # Save
    os.makedirs('results', exist_ok=True)
    model_path = f'results/markov_icl_gmm_K{args.K}_N{args.N}_{args.method}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved: {model_path}")


if __name__ == '__main__':
    main()

