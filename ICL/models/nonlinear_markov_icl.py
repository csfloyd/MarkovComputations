"""
Nonlinear Markov ICL Model.

This implementation uses a nonlinear steady-state equation:
    W p + p Y p = 0
    sum(p) = 1

where:
- W is an (n_nodes, n_nodes) rate matrix computed from parameters K
- Y is an (n_nodes, n_nodes, n_nodes) rate tensor computed from parameters L
- p is the steady-state distribution
- (p Y p)_i = sum_{j,k} Y_{i,j,k} * p_j * p_k

Naming convention:
- K: learnable parameters (weights) that map context to rate matrix W
- L: learnable parameters (weights) that map context to rate tensor Y
- W, Y: computed rate matrix/tensor from K, L and context
"""

import torch
import torch.nn as nn
import numpy as np
from .base_icl_model import BaseICLModel


class NonlinearMarkovICL(BaseICLModel):
    """
    Nonlinear Markov implementation with W p + p Y p = 0 dynamics.
    
    Architecture:
    1. Compute context-dependent rate matrix W (from parameters K) and rate tensor Y (from parameters L)
    2. Solve nonlinear steady state: W p + p Y p = 0 with sum(p) = 1
    3. Map π to attention over context positions
    4. Aggregate attention by label to get class logits
    """
    
    def __init__(self, n_nodes=10, z_dim=2, L=75, N=4, use_label_mod=False, 
                 learn_base_rates=True, transform_func='exp', 
                 sparsity_rho_edge_K=1.0, sparsity_rho_all_K=1.0,
                 sparsity_rho_edge_L=1.0, sparsity_rho_all_L=1.0,
                 sparsity_rho_edge_base_W=1.0, sparsity_rho_edge_base_Y=1.0,
                 base_mask_value=0.0):
        """
        Initialize Nonlinear Markov ICL model.
        
        Args:
            n_nodes: Number of Markov chain nodes
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
            use_label_mod: Whether to modulate rates by context labels
            learn_base_rates: Whether to allow gradient updates to unmasked base rates
            transform_func: Transformation function for rates ('exp', 'relu', 'elu')
            sparsity_rho_edge_K: Fraction of non-zero elements in per-edge mask for K_params
            sparsity_rho_all_K: Fraction of non-zero elements in per-element mask for K_params
            sparsity_rho_edge_L: Fraction of non-zero elements in per-triplet mask for L_params
            sparsity_rho_all_L: Fraction of non-zero elements in per-element mask for L_params
            sparsity_rho_edge_base_W: Fraction of (i,j) edges with base rates in W
            sparsity_rho_edge_base_Y: Fraction of (i,j,k) triplets with base rates in Y
            base_mask_value: Value for masked base rates (0.0 or float('-inf'))
        """
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod
        self.transform_func = transform_func
        self.sparsity_rho_edge_K = sparsity_rho_edge_K
        self.sparsity_rho_all_K = sparsity_rho_all_K
        self.sparsity_rho_edge_L = sparsity_rho_edge_L
        self.sparsity_rho_all_L = sparsity_rho_all_L
        self.sparsity_rho_edge_base_W = sparsity_rho_edge_base_W
        self.sparsity_rho_edge_base_Y = sparsity_rho_edge_base_Y
        self.base_mask_value = base_mask_value
        self.learn_base_rates = learn_base_rates
        
        z_full_dim = (N + 1) * z_dim  # Flatten all context + query
        l_full_dim = N
        
        # Initialize parameters with proper scaling
        init_scale_K = 0.05 / np.sqrt(n_nodes)
        init_scale_L = 0.05 / (n_nodes)  # Smaller for rank-3 tensor
        init_scale_B = 0.1 / np.sqrt(N)
        init_base_K = -2.0 - 0.5 * np.log(n_nodes)
        init_base_L = -3.0 - 0.5 * np.log(n_nodes)  # Smaller base for nonlinear term
        
        # Learnable parameters for rate matrix W (K maps z → W)
        self.K_params = nn.Parameter(torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K)
        
        # Learnable parameters for rate tensor Y (L maps z → Y)
        self.L_params = nn.Parameter(torch.randn(n_nodes, n_nodes, n_nodes, z_full_dim) * init_scale_L)
        
        # Optional: modulate rates by context labels
        if self.use_label_mod:
            self.label_modulation_K = nn.Parameter(
                torch.randn(n_nodes, n_nodes, l_full_dim) * init_scale_K * 0.5
            )
            self.label_modulation_L = nn.Parameter(
                torch.randn(n_nodes, n_nodes, n_nodes, l_full_dim) * init_scale_L * 0.5
            )
        else:
            self.label_modulation_K = None
            self.label_modulation_L = None
        
        # B maps steady state to context position scores (attention mechanism)
        self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
        
        # Base log rates for W and Y
        # Note: To get zero base rates, set sparsity_rho_edge_base_W/Y = 0.0 with base_mask_value = 0.0
        self.base_log_rates_W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1 + init_base_K)
        self.base_log_rates_Y = nn.Parameter(torch.randn(n_nodes, n_nodes, n_nodes) * 0.1 + init_base_L)
        
        # Create sparsity masks for K_params, L_params, and base rates
        self._create_sparsity_masks(z_full_dim)
        
        # Set up gradient masking for base rates if learn_base_rates is False
        # or if we want to only learn unmasked base rates
        if not learn_base_rates:
            self.base_log_rates_W.requires_grad = False
            self.base_log_rates_Y.requires_grad = False
        else:
            # Register hooks to zero out gradients for masked base rates
            self.base_log_rates_W.register_hook(
                lambda grad: grad * self.base_log_rates_W_mask
            )
            self.base_log_rates_Y.register_hook(
                lambda grad: grad * self.base_log_rates_Y_mask
            )
        
        print(f"  Initialized Nonlinear Markov ICL model (L={L} classes, "
              f"attention over {N} context items)")
        print(f"  Nonlinear dynamics: W p + p Y p = 0")
        print(f"  Label modulation: {self.use_label_mod}")
        print(f"  Base rates learnable: {learn_base_rates}")
        print(f"  Base mask value: {base_mask_value}")
        print(f"  Sparsity K: rho_edge={sparsity_rho_edge_K:.3f}, rho_all={sparsity_rho_all_K:.3f}")
        print(f"  Sparsity L: rho_edge={sparsity_rho_edge_L:.3f}, rho_all={sparsity_rho_all_L:.3f}")
        print(f"  Sparsity base_W: rho_edge={sparsity_rho_edge_base_W:.3f}")
        print(f"  Sparsity base_Y: rho_edge={sparsity_rho_edge_base_Y:.3f}")
        sparsity_stats = self.get_sparsity_stats()
        if sparsity_stats:
            print(f"  K_params sparsity: {sparsity_stats['K_actual_sparsity']:.3f} "
                  f"({sparsity_stats['K_num_active']}/{sparsity_stats['K_num_total']} active)")
            print(f"  L_params sparsity: {sparsity_stats['L_actual_sparsity']:.3f} "
                  f"({sparsity_stats['L_num_active']}/{sparsity_stats['L_num_total']} active)")
            print(f"  base_W sparsity: {sparsity_stats['base_W_actual_sparsity']:.3f} "
                  f"({sparsity_stats['base_W_num_active']}/{sparsity_stats['base_W_num_total']} active)")
            print(f"  base_Y sparsity: {sparsity_stats['base_Y_actual_sparsity']:.3f} "
                  f"({sparsity_stats['base_Y_num_active']}/{sparsity_stats['base_Y_num_total']} active)")
        print(f"  Parameters: {self.get_num_parameters():,}")
    
    def _create_sparsity_masks(self, z_full_dim):
        """
        Create sparsity masks for K_params, L_params, and base rates.
        
        For K_params (n_nodes, n_nodes, z_full_dim):
            - Per-edge mask: (n_nodes, n_nodes, 1) - controls which (i,j) edges exist
            - Per-element mask: (n_nodes, n_nodes, z_full_dim) - controls sparsity within each edge
        
        For L_params (n_nodes, n_nodes, n_nodes, z_full_dim):
            - Per-triplet mask: (n_nodes, n_nodes, n_nodes, 1) - controls which (i,j,k) triplets exist
            - Per-element mask: (n_nodes, n_nodes, n_nodes, z_full_dim) - controls sparsity within each triplet
        
        For base_log_rates_W (n_nodes, n_nodes):
            - Per-edge mask: (n_nodes, n_nodes) - controls which (i,j) edges have base rates
            - IMPORTANT: Automatically set to 1 (enabled) for any edge where K_params is active
              This ensures edges with learnable K parameters aren't permanently disabled by -inf base rates
        
        For base_log_rates_Y (n_nodes, n_nodes, n_nodes):
            - Per-triplet mask: (n_nodes, n_nodes, n_nodes) - controls which (i,j,k) triplets have base rates
            - IMPORTANT: Automatically set to 1 (enabled) for any triplet where L_params is active
              This ensures triplets with learnable L parameters aren't permanently disabled by -inf base rates
        
        Final mask is element-wise product: only survives if both masks are 1.
        
        Args:
            z_full_dim: Full dimension of z features
        """
        n = self.n_nodes
        
        # ==================== K_params mask ====================
        # Per-edge mask: same across all input dimensions
        if self.sparsity_rho_edge_K < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge_K
            edge_mask_samples = torch.rand(n, n, 1)
            edge_mask_K = (edge_mask_samples < self.sparsity_rho_edge_K).float()
            # Broadcast to full dimension
            edge_mask_K = edge_mask_K.expand(-1, -1, z_full_dim).contiguous()
        else:
            edge_mask_K = torch.ones(n, n, z_full_dim)
        
        # Per-element mask: independent for each element
        if self.sparsity_rho_all_K < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_all_K
            element_mask_samples_K = torch.rand(n, n, z_full_dim)
            element_mask_K = (element_mask_samples_K < self.sparsity_rho_all_K).float()
        else:
            element_mask_K = torch.ones(n, n, z_full_dim)
        
        # Combine masks: element survives only if both masks are 1
        combined_mask_K = edge_mask_K * element_mask_K
        
        # Register as buffer (moves with model to device, not trained)
        self.register_buffer('K_params_mask', combined_mask_K)
        
        # ==================== L_params mask ====================
        # Per-triplet mask: same across all input dimensions
        if self.sparsity_rho_edge_L < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge_L
            triplet_mask_samples = torch.rand(n, n, n, 1)
            triplet_mask_L = (triplet_mask_samples < self.sparsity_rho_edge_L).float()
            # Broadcast to full dimension
            triplet_mask_L = triplet_mask_L.expand(-1, -1, -1, z_full_dim).contiguous()
        else:
            triplet_mask_L = torch.ones(n, n, n, z_full_dim)
        
        # Per-element mask: independent for each element
        if self.sparsity_rho_all_L < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_all_L
            element_mask_samples_L = torch.rand(n, n, n, z_full_dim)
            element_mask_L = (element_mask_samples_L < self.sparsity_rho_all_L).float()
        else:
            element_mask_L = torch.ones(n, n, n, z_full_dim)
        
        # Combine masks: element survives only if both masks are 1
        combined_mask_L = triplet_mask_L * element_mask_L
        
        # Register as buffer (moves with model to device, not trained)
        self.register_buffer('L_params_mask', combined_mask_L)
        
        # ==================== base_log_rates_W mask ====================
        if self.sparsity_rho_edge_base_W < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge_base_W
            edge_mask_samples_W = torch.rand(n, n)
            base_mask_W = (edge_mask_samples_W < self.sparsity_rho_edge_base_W).float()
        else:
            base_mask_W = torch.ones(n, n)
        
        # IMPORTANT: Override base mask for edges where K_params is active
        # If K_params has active parameters for edge (i,j), force base_W mask to 1
        # This ensures -inf base rates don't permanently disable edges that can be learned via K
        k_edge_active = (combined_mask_K.sum(dim=2) > 0).float()  # (n, n) - 1 if any K param is active
        base_mask_W = torch.maximum(base_mask_W, k_edge_active)
        
        # Register as buffer
        self.register_buffer('base_log_rates_W_mask', base_mask_W)
        
        # ==================== base_log_rates_Y mask ====================
        if self.sparsity_rho_edge_base_Y < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge_base_Y
            triplet_mask_samples_Y = torch.rand(n, n, n)
            base_mask_Y = (triplet_mask_samples_Y < self.sparsity_rho_edge_base_Y).float()
        else:
            base_mask_Y = torch.ones(n, n, n)
        
        # IMPORTANT: Override base mask for triplets where L_params is active
        # If L_params has active parameters for triplet (i,j,k), force base_Y mask to 1
        # This ensures -inf base rates don't permanently disable triplets that can be learned via L
        l_triplet_active = (combined_mask_L.sum(dim=3) > 0).float()  # (n, n, n) - 1 if any L param is active
        base_mask_Y = torch.maximum(base_mask_Y, l_triplet_active)
        
        # Register as buffer
        self.register_buffer('base_log_rates_Y_mask', base_mask_Y)
    
    def compute_rate_matrix_W(self, z_batch, labels_batch=None):
        """
        Compute the rate matrix W from parameters K where columns sum to zero.
        
        W[i,j] = transform(base_K[i,j] + K_params[i,j] · z + label_mod_K[i,j] · labels) for i≠j
        W[j,j] = -Σ_{k≠j} W[k,j]
        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            W_batch: (batch_size, n_nodes, n_nodes) - computed rate matrix with columns summing to zero
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        
        # Apply sparsity mask to K_params
        K_params_masked = self.K_params * self.K_params_mask
        
        # Compute modulation: K_params · z
        K_expanded = K_params_masked.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rate_mod = torch.einsum('bijd,bd->bij', K_expanded, z_batch)
        
        # Optional: Add label modulation
        if self.use_label_mod and labels_batch is not None:
            label_expanded = self.label_modulation_K.unsqueeze(0).expand(batch_size, -1, -1, -1)
            label_mod = torch.einsum('bijd,bd->bij', label_expanded, labels_batch)
            rate_mod = rate_mod + label_mod
        
        # Add base rates with masking
        # Apply mask: set masked elements to base_mask_value (0.0 or -inf)
        base_masked = torch.where(
            self.base_log_rates_W_mask.bool(),
            self.base_log_rates_W,
            torch.full_like(self.base_log_rates_W, self.base_mask_value)
        )
        base_expanded = base_masked.unsqueeze(0).expand(batch_size, -1, -1)
        log_rates = base_expanded + rate_mod
        
        # Clamp for numerical stability
        log_rates = torch.clamp(log_rates, min=-15.0, max=15.0)
        
        # Apply transformation to get rates
        if self.transform_func == 'exp':
            rates = torch.exp(log_rates)
        elif self.transform_func == 'relu':
            rates = torch.relu(log_rates) + 1e-10
        elif self.transform_func == 'elu':
            rates = torch.nn.functional.elu(log_rates) + 1e-10
        else:
            raise ValueError(f"Invalid transform function: {self.transform_func}")
        
        # Zero out diagonal (we'll set it later)
        eye = torch.eye(n, device=rates.device).unsqueeze(0)
        rates = rates * (1 - eye)
        
        # Construct W with proper diagonal so columns sum to zero
        # W[j,j] = -Σ_{k≠j} W[k,j]
        col_sums = rates.sum(dim=1)  # Sum over rows
        W_batch = rates - torch.diag_embed(col_sums)
        
        return W_batch
    
    def compute_rate_tensor_Y(self, z_batch, labels_batch=None):
        """
        Compute the rate tensor Y from parameters L for nonlinear interactions.
        
        Y[i,j,k] = transform(base_L[i,j,k] + L_params[i,j,k] · z + label_mod_L[i,j,k] · labels) for i≠j
        Y[j,j,k] = -Σ_{i≠j} Y[i,j,k] for all j, k
        
        The nonlinear term is: (p Y p)_i = sum_{j,k} Y[i,j,k] * p_j * p_k
        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            Y_batch: (batch_size, n_nodes, n_nodes, n_nodes) - computed rate tensor with constraint Y[j,j,k] = -Σ_{i≠j} Y[i,j,k]
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        
        # Apply sparsity mask to L_params
        L_params_masked = self.L_params * self.L_params_mask
        
        # Compute modulation: L_params · z
        L_expanded = L_params_masked.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        rate_mod = torch.einsum('bijkd,bd->bijk', L_expanded, z_batch)
        
        # Optional: Add label modulation
        if self.use_label_mod and labels_batch is not None:
            label_expanded = self.label_modulation_L.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            label_mod = torch.einsum('bijkd,bd->bijk', label_expanded, labels_batch)
            rate_mod = rate_mod + label_mod
        
        # Add base rates with masking
        # Apply mask: set masked elements to base_mask_value (0.0 or -inf)
        base_masked = torch.where(
            self.base_log_rates_Y_mask.bool(),
            self.base_log_rates_Y,
            torch.full_like(self.base_log_rates_Y, self.base_mask_value)
        )
        base_expanded = base_masked.unsqueeze(0).expand(batch_size, -1, -1, -1)
        log_rates = base_expanded + rate_mod
        
        # Clamp for numerical stability
        log_rates = torch.clamp(log_rates, min=-15.0, max=15.0)
        
        # Apply transformation to get rates
        if self.transform_func == 'exp':
            rates = torch.exp(log_rates)
        elif self.transform_func == 'relu':
            rates = torch.relu(log_rates) + 1e-10
        elif self.transform_func == 'elu':
            rates = torch.nn.functional.elu(log_rates) + 1e-10
        else:
            raise ValueError(f"Invalid transform function: {self.transform_func}")
        
        # Zero out diagonal elements Y[j,j,k] (we'll set them later)
        # Create a mask for the diagonal in the first two dimensions: i==j
        diag_mask = torch.eye(n, device=rates.device).unsqueeze(0).unsqueeze(-1)  # (1, n, n, 1)
        rates = rates * (1 - diag_mask)  # Zero out Y[j,j,k] for all j, k
        
        # Apply constraint: Y[j,j,k] = -Σ_{i≠j} Y[i,j,k]
        # For each j and k, sum over i (which now excludes i=j since we zeroed it)
        # rates has shape (batch_size, n_nodes, n_nodes, n_nodes) = (batch, i, j, k)
        sum_over_i = rates.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, n_nodes, n_nodes) = (batch, 1, j, k)
        
        # Set diagonal elements Y[j,j,k] = -sum using the diagonal mask
        Y_batch = rates - diag_mask * sum_over_i
        
        return Y_batch
    
    
    def direct_solve_steady_state(self, W_batch, Y_batch, n_iter=50, step_size=0.1, eps=1e-12):
        """
        Solve nonlinear steady state using fixed-point iteration (gradient descent).
        
        Solves: W p + p Y p = 0 with sum(p) = 1 and p >= 0
        where (p Y p)_i = sum_{j,k} Y[i,j,k] p_j p_k
        
        Uses gradient descent to minimize ||W p + p Y p||, projecting onto the probability simplex.
        
        Args:
            W_batch: (batch_size, n_nodes, n_nodes) - linear rate matrix
            Y_batch: (batch_size, n_nodes, n_nodes, n_nodes) - nonlinear rate tensor
            n_iter: Number of fixed-point iterations (default: 50)
            step_size: Step size for gradient descent (default: 0.1)
            eps: Minimum value for clamping (default: 1e-12)
            
        Returns:
            p_batch: (batch_size, n_nodes) - steady state distributions
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Initialize from uniform distribution
        p_batch = torch.full((batch_size, n), 1.0/n, device=device)
        
        for _ in range(n_iter):
            # Compute quadratic term: Q_i = sum_{j,k} Y[i,j,k] p_j p_k
            # Outer product: p_j * p_k
            outer = p_batch.unsqueeze(2) * p_batch.unsqueeze(1)  # (batch_size, n, n)
            Q = torch.einsum("bijk,bjk->bi", Y_batch, outer)  # (batch_size, n)
            
            # Compute linear term: L_i = sum_j W[i,j] p_j
            L = torch.einsum("bij,bj->bi", W_batch, p_batch)  # (batch_size, n)
            
            # Total drift function F = W p + p Y p
            F = L + Q  # (batch_size, n)
            
            # Gradient descent step toward F(p) = 0
            p_batch = p_batch + step_size * F
            
            # Project onto probability simplex: clamp and normalize
            p_batch = torch.clamp(p_batch, min=eps)
            p_batch = p_batch / p_batch.sum(dim=1, keepdim=True)
        
        # Handle NaN/Inf (fallback to uniform)
        mask = torch.isnan(p_batch).any(dim=1) | torch.isinf(p_batch).any(dim=1)
        if mask.any():
            p_batch[mask] = 1.0 / n
        
        return p_batch

    def newton_steady_state(self, W_batch, Y_batch, n_iter=5, eps=1e-12, tol=1e-8):
        """
        Newton's method for solving W p + p Y p = 0.
        
        Much faster convergence (typically 5-10 iterations vs 50+)
        """
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        
        # Initialize from uniform distribution
        p_batch = torch.full((batch_size, n), 1.0/n, device=device)
        
        for _ in range(n_iter):
            # Compute F(p) = W p + p Y p
            outer = p_batch.unsqueeze(2) * p_batch.unsqueeze(1)
            Q = torch.einsum("bijk,bjk->bi", Y_batch, outer)
            L = torch.bmm(W_batch, p_batch.unsqueeze(-1)).squeeze(-1)
            F = L + Q
            
            # Check convergence
            if torch.abs(F).max() < tol:
                break
            
            # Compute Jacobian: J[i,j] = dF_i/dp_j
            # J = W + (Y[:,:,j,k] + Y[:,i,k,j]) * p_k (summed over k)
            # For quadratic term: d/dp_j (sum_k,l Y[i,k,l] p_k p_l) = sum_l Y[i,j,l]*p_l + sum_k Y[i,k,j]*p_k
            
            # Simplified: J = W + 2 * (Y contracted with p on last index)
            Y_p_contract = torch.einsum("bijk,bk->bij", Y_batch, p_batch)  # (b, n, n)
            Y_p_contract_T = torch.einsum("bijk,bj->bik", Y_batch, p_batch)  # (b, n, n)
            J = W_batch + Y_p_contract + Y_p_contract_T  # (batch, n, n)
            
            # Add constraint: last row enforces sum(p) = 1
            J_constrained = J.clone()
            J_constrained[:, -1, :] = 1.0
            
            # RHS
            b = -F
            b[:, -1] = 1.0 - p_batch.sum(dim=1)  # Enforce sum constraint
            
            # Solve J * delta_p = -F
            delta_p = torch.linalg.solve(J_constrained, b)
            
            # Update with line search damping
            alpha = 1.0
            p_new = p_batch + alpha * delta_p
            p_new = torch.clamp(p_new, min=eps)
            p_new = p_new / p_new.sum(dim=1, keepdim=True)
            
            p_batch = p_new
        
        return p_batch
    
    def forward(self, z_seq_batch, labels_seq_batch, method='newton', temperature=1.0, 
                n_iter=50, step_size=0.1):
        """
        Forward pass with attention over context items.
        
        Architecture:
        1. Compute rate matrix W and rate tensor Y from context
        2. Solve nonlinear steady state: W p + p Y p = 0
        3. Context position scores: q_m = Σ_k B_{k,m} * π_k
        4. Attention: softmax(q / temperature)
        5. Class logits: sum attention weights by context label
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: str - method for computing steady state (currently only 'direct_solve' supported)
            temperature: float - softmax temperature (default 1.0)
            n_iter: int - number of iterations for steady state solver (default 50)
            step_size: float - step size for gradient descent in solver (default 0.1)
            
        Returns:
            logits: (batch_size, L) - class logits (log-probabilities)
        """
        batch_size = z_seq_batch.shape[0]
        device = z_seq_batch.device
        
        # Flatten z sequences
        z_flat = z_seq_batch.reshape(batch_size, -1)
        
        # Compute rate matrix W and rate tensor Y
        W_batch = self.compute_rate_matrix_W(z_flat)
        Y_batch = self.compute_rate_tensor_Y(z_flat)
        
        # Compute steady state (currently only direct_solve is implemented for nonlinear case)
        if method == 'newton':
            p_batch = self.newton_steady_state(W_batch, Y_batch, n_iter=10)
        elif method == 'direct_solve':
            p_batch = self.direct_solve_steady_state(W_batch, Y_batch, n_iter=n_iter, step_size=step_size)
        else:
            raise ValueError(f"Method '{method}' not supported for nonlinear model. Use 'direct_solve'.")
        
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
    
    def get_sparsity_stats(self):
        """
        Get statistics about K_params, L_params, and base rate sparsity.
        
        Returns:
            dict with sparsity information for K, L, base_W, and base_Y, or None if no masks exist
        """
        if not hasattr(self, 'K_params_mask') or not hasattr(self, 'L_params_mask'):
            return None
        
        # K_params stats
        mask_K = self.K_params_mask
        num_total_K = mask_K.numel()
        num_active_K = mask_K.sum().item()
        actual_sparsity_K = 1.0 - (num_active_K / num_total_K)
        
        # L_params stats
        mask_L = self.L_params_mask
        num_total_L = mask_L.numel()
        num_active_L = mask_L.sum().item()
        actual_sparsity_L = 1.0 - (num_active_L / num_total_L)
        
        # base_log_rates_W stats
        mask_base_W = self.base_log_rates_W_mask
        num_total_base_W = mask_base_W.numel()
        num_active_base_W = mask_base_W.sum().item()
        actual_sparsity_base_W = 1.0 - (num_active_base_W / num_total_base_W)
        
        # base_log_rates_Y stats
        mask_base_Y = self.base_log_rates_Y_mask
        num_total_base_Y = mask_base_Y.numel()
        num_active_base_Y = mask_base_Y.sum().item()
        actual_sparsity_base_Y = 1.0 - (num_active_base_Y / num_total_base_Y)
        
        return {
            'rho_edge_K': self.sparsity_rho_edge_K,
            'rho_all_K': self.sparsity_rho_all_K,
            'rho_edge_L': self.sparsity_rho_edge_L,
            'rho_all_L': self.sparsity_rho_all_L,
            'rho_edge_base_W': self.sparsity_rho_edge_base_W,
            'rho_edge_base_Y': self.sparsity_rho_edge_base_Y,
            'K_actual_sparsity': actual_sparsity_K,
            'K_num_active': int(num_active_K),
            'K_num_total': num_total_K,
            'K_fraction_active': num_active_K / num_total_K,
            'L_actual_sparsity': actual_sparsity_L,
            'L_num_active': int(num_active_L),
            'L_num_total': num_total_L,
            'L_fraction_active': num_active_L / num_total_L,
            'base_W_actual_sparsity': actual_sparsity_base_W,
            'base_W_num_active': int(num_active_base_W),
            'base_W_num_total': num_total_base_W,
            'base_W_fraction_active': num_active_base_W / num_total_base_W,
            'base_Y_actual_sparsity': actual_sparsity_base_Y,
            'base_Y_num_active': int(num_active_base_Y),
            'base_Y_num_total': num_total_base_Y,
            'base_Y_fraction_active': num_active_base_Y / num_total_base_Y
        }
    
    def resample_sparsity_mask(self):
        """
        Re-randomize the sparsity masks with same rho values.
        Useful for experiments testing different random masks.
        Resamples masks for K_params, L_params, base_log_rates_W, and base_log_rates_Y.
        """
        z_full_dim = self.K_params.shape[2]
        self._create_sparsity_masks(z_full_dim)
        
        # Re-register gradient hooks for base rates if learn_base_rates is True
        if self.learn_base_rates:
            # Remove old hooks (they're automatically replaced when re-registering)
            self.base_log_rates_W.register_hook(
                lambda grad: grad * self.base_log_rates_W_mask
            )
            self.base_log_rates_Y.register_hook(
                lambda grad: grad * self.base_log_rates_Y_mask
            )
    
    def get_active_edges(self):
        """
        Get list of (i, j) node pairs in W matrix that have at least one active parameter in K.
        
        Returns:
            List of tuples [(i, j), ...] representing active edges in W
        """
        if not hasattr(self, 'K_params_mask'):
            # No mask, all edges are active
            return [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        
        # Sum across z_dim to see which (i,j) pairs have any active params
        edge_active = self.K_params_mask.sum(dim=2) > 0  # (n_nodes, n_nodes)
        active_indices = torch.nonzero(edge_active, as_tuple=False)
        
        return [(i.item(), j.item()) for i, j in active_indices]
    
    def get_active_triplets(self):
        """
        Get list of (i, j, k) node triplets in Y tensor that have at least one active parameter in L.
        
        Returns:
            List of tuples [(i, j, k), ...] representing active triplets in Y
        """
        if not hasattr(self, 'L_params_mask'):
            # No mask, all triplets are active
            return [(i, j, k) for i in range(self.n_nodes) 
                    for j in range(self.n_nodes) 
                    for k in range(self.n_nodes)]
        
        # Sum across z_dim to see which (i,j,k) triplets have any active params
        triplet_active = self.L_params_mask.sum(dim=3) > 0  # (n_nodes, n_nodes, n_nodes)
        active_indices = torch.nonzero(triplet_active, as_tuple=False)
        
        return [(i.item(), j.item(), k.item()) for i, j, k in active_indices]

