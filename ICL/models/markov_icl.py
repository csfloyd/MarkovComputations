"""
Markov ICL Model using Matrix Tree Theorem.

This implementation uses a rate matrix K where:
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
from .base_icl_model import BaseICLModel


class MatrixTreeMarkovICL(BaseICLModel):
    """
    Matrix Tree Theorem implementation using rate matrix K with CLASSIFICATION output.
    
    Architecture:
    1. Compute context-dependent rate matrix K
    2. Solve for steady state distribution π
    3. Map π to attention over context positions
    4. Aggregate attention by label to get class logits
    """
    
    def __init__(self, n_nodes=10, z_dim=2, L=75, N=4, use_label_mod=False):
        """
        Initialize Markov ICL model.
        
        Args:
            n_nodes: Number of Markov chain nodes
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
            use_label_mod: Whether to modulate rates by context labels
        """
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod
        
        z_full_dim = (N + 1) * z_dim  # Flatten all context + query
        l_full_dim = N
        
        # Initialize parameters with proper scaling
        init_scale_K = 0.05 / np.sqrt(n_nodes)
        init_scale_B = 0.1 / np.sqrt(N)
        init_base = -2.0 - 0.5 * np.log(n_nodes)
        
        # Learnable parameters for rate matrix (modulated by z)
        self.K_params = nn.Parameter(torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K)
        
        # Optional: modulate rates by context labels
        if self.use_label_mod:
            self.label_modulation = nn.Parameter(
                torch.randn(n_nodes, n_nodes, l_full_dim) * init_scale_K * 0.5
            )
        else:
            self.label_modulation = None
        
        # B maps steady state to context position scores (attention mechanism)
        self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
        
        # Base log rates
        self.base_log_rates = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1 + init_base)
        
        print(f"  Initialized ICL Attention model (L={L} classes, "
              f"attention over {N} context items)")
        print(f"  Label modulation: {self.use_label_mod}")
        print(f"  Parameters: {self.get_num_parameters():,}")
    
    def compute_rate_matrix_K(self, z_batch, labels_batch=None):
        """
        Compute the rate matrix K where columns sum to zero.
        
        K[i,j] = exp(base[i,j] + K_params[i,j] · z + label_mod[i,j] · labels) for i≠j
        K[j,j] = -Σ_{k≠j} K[k,j]
        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            K_batch: (batch_size, n_nodes, n_nodes) with columns summing to zero
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        
        # Compute modulation: K_params · z
        K_expanded = self.K_params.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rate_mod = torch.einsum('bijd,bd->bij', K_expanded, z_batch)
        
        # Optional: Add label modulation
        if self.use_label_mod and labels_batch is not None:
            label_expanded = self.label_modulation.unsqueeze(0).expand(batch_size, -1, -1, -1)
            label_mod = torch.einsum('bijd,bd->bij', label_expanded, labels_batch)
            rate_mod = rate_mod + label_mod
        
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
        # K[j,j] = -Σ_{k≠j} K[k,j]
        col_sums = rates.sum(dim=1)  # Sum over rows
        K_batch = rates - torch.diag_embed(col_sums)
        
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
            p_batch: (batch_size, n_nodes) - steady state distributions
        """
        batch_size, n = K_batch.shape[0], self.n_nodes
        device = K_batch.device
        
        # Compute determinants of all minors
        p_batch = torch.zeros(batch_size, n, device=device)
        
        for i in range(n):
            # Delete row i and column i
            indices = [j for j in range(n) if j != i]
            K_minor = K_batch[:, indices, :][:, :, indices]
            
            # Compute determinant
            det = torch.det(K_minor)
            det = torch.abs(det)  # Handle numerical sign issues
            det = torch.clamp(det, min=1e-10, max=1e10)
            
            p_batch[:, i] = det
        
        # Normalize
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
        Compute steady state using linear solver (more efficient than Matrix Tree).
        
        Solves the augmented system:
            [K      ]     [0]
            [1,1,...] p = [1]
        
        Where K p = 0 (steady state) and sum(p) = 1 (normalization).
        
        Args:
            K_batch: (batch_size, n_nodes, n_nodes)
            
        Returns:
            p_batch: (batch_size, n_nodes)
        """
        batch_size, n = K_batch.shape[0], self.n_nodes
        device = K_batch.device
        
        # Augment system
        ones_row = torch.ones(batch_size, 1, n, device=device)
        A_augmented = torch.cat([K_batch, ones_row], dim=1)
        
        # Target
        b = torch.zeros(batch_size, n + 1, device=device)
        b[:, -1] = 1.0
        
        # Solve using least squares
        try:
            p_batch = torch.linalg.lstsq(A_augmented, b.unsqueeze(-1)).solution.squeeze(-1)
        except RuntimeError:
            # Fallback: Manual normal equations with regularization
            AtA = torch.bmm(A_augmented.transpose(1, 2), A_augmented)
            AtA = AtA + 1e-6 * torch.eye(n, device=device).unsqueeze(0)
            Atb = torch.bmm(A_augmented.transpose(1, 2), b.unsqueeze(-1))
            p_batch = torch.linalg.solve(AtA, Atb).squeeze(-1)
        
        # Ensure non-negativity and normalization
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / (p_batch.sum(dim=1, keepdim=True) + 1e-8)
        
        # Handle NaN/Inf
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / self.n_nodes
        
        return p_batch
    
    def direct_solve_steady_state(self, K_batch):
        """
        Replace last row of K with normalization constraint and solve directly.
        
        Args:
            K_batch: (batch_size, n_nodes, n_nodes)
            
        Returns:
            p_batch: (batch_size, n_nodes)
        """
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
        
        # Ensure non-negativity and normalization
        p_batch = torch.clamp(p_batch, min=0.0)
        p_batch = p_batch / p_batch.sum(dim=1, keepdim=True)
        
        return p_batch
    
    def forward(self, z_seq_batch, labels_seq_batch, method='direct_solve', temperature=1.0):
        """
        Forward pass with attention over context items.
        
        Architecture:
        1. Steady state π from Markov chain
        2. Context position scores: q_m = Σ_k B_{k,m} * π_k
        3. Attention: softmax(q / temperature)
        4. Class logits: sum attention weights by context label
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: str - method for computing steady state
                'matrix_tree', 'linear_solver', or 'direct_solve'
            temperature: float - softmax temperature (default 1.0)
            
        Returns:
            logits: (batch_size, L) - class logits (log-probabilities)
        """
        batch_size = z_seq_batch.shape[0]
        device = z_seq_batch.device
        
        # Flatten z sequences
        z_flat = z_seq_batch.reshape(batch_size, -1)
        
        # Compute rate matrix K
        K_batch = self.compute_rate_matrix_K(z_flat)
        
        # Compute steady state
        if method == 'matrix_tree':
            p_batch = self.matrix_tree_steady_state(K_batch)
        elif method == 'linear_solver':
            p_batch = self.linear_solver_steady_state(K_batch)
        elif method == 'direct_solve':
            p_batch = self.direct_solve_steady_state(K_batch)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        # Compute context position scores: q_m = Σ_k B_{k,m} * π_k
        q = torch.matmul(p_batch, self.B)  # (batch_size, N)
        
        # Apply temperature and softmax to get attention over context positions
        attention = torch.softmax(q / temperature, dim=1)  # (batch_size, N)
        
        # Convert context labels to class logits (VECTORIZED)
        # One-hot encode labels: (batch, N) → (batch, N, L)
        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,  # Convert 1-indexed to 0-indexed
            num_classes=self.L
        ).float()
        
        # Aggregate attention weights by label class
        # For each class k, sum attention weights where label = k
        logits = torch.einsum('bn,bnk->bk', attention, labels_one_hot)
        
        # Convert to log-probabilities for NLLLoss
        logits = logits.clamp(min=1e-6, max=1.0)
        logits = torch.log(logits)
        
        return logits

