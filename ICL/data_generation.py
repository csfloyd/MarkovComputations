"""
Data Generation for In-Context Learning with Gaussian Mixture Models

This module provides functions for generating ICL training and testing data
from Gaussian Mixture Models with discrete labels.
"""

import torch
import numpy as np


class GaussianMixtureModel:
    """Gaussian Mixture Model with K classes for ICL task with DISCRETE labels."""
    
    def __init__(self, K, D, L=None, epsilon=0.1, seed=None, label_min=0.0, label_max=1.0):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            K: Number of classes
            D: Dimension of feature space
            L: Number of labels (defaults to K)
            epsilon: Within-class noise scale
            seed: Random seed for reproducibility
            label_min: Minimum label value (unused for discrete labels)
            label_max: Maximum label value (unused for discrete labels)
        """
        self.K = K
        self.D = D
        self.L = L if L is not None else K
        self.epsilon = epsilon
        self.label_min = label_min
        self.label_max = label_max
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Sample class means from standard Gaussian scaled by 1/sqrt(D)
        self.class_means = torch.randn(K, D) / np.sqrt(D)
        
        # DISCRETE labels from 1 to K
        self.class_to_label = torch.arange(1, K + 1, dtype=torch.float32)
        
    def sample_from_class(self, class_idx, n_samples=1):
        """
        Sample points from a specific class.
        
        Args:
            class_idx: Index of the class to sample from (0 to K-1)
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, D)
        """
        mu_k = self.class_means[class_idx]
        noise = torch.randn(n_samples, self.D) / np.sqrt(self.D)
        return mu_k + self.epsilon * noise
    
    def get_label(self, class_idx):
        """
        Get the label for a specific class.
        
        Args:
            class_idx: Index of the class (0 to K-1)
            
        Returns:
            Label (1 to K)
        """
        return self.class_to_label[class_idx]


def generate_icl_gmm_data(gmm, n_samples, N, novel_classes=False, exact_copy=True, B=1, 
                          label_min=None, label_max=None, K_classes=None):
    """
    Generate ICL data from GMM with DISCRETE labels.
    
    Creates sequences of N context examples plus 1 query, where the query's class
    appears in the context (optionally repeated B times for "burstiness").
    
    Args:
        gmm: GaussianMixtureModel instance
        n_samples: Number of sequences to generate
        N: Number of context examples
        novel_classes: If True, create new class means not in GMM
        exact_copy: If True, query is exact copy of a context item
        B: Burstiness - number of repetitions per class in context
        label_min: Unused (kept for backwards compatibility)
        label_max: Unused (kept for backwards compatibility)
        K_classes: Number of possible label classes (defaults to gmm.K)
        
    Returns:
        List of tuples (z_seq, labels, target_label) where:
            - z_seq: (N+1, D) tensor of features
            - labels: (N,) tensor of context labels
            - target_label: scalar target label for query
    """
    assert 1 <= B <= N and N % B == 0, f"Invalid B={B} for N={N}"
    n_classes_in_context = N // B
    K_labels = K_classes if K_classes is not None else gmm.K
    data = []
    
    for _ in range(n_samples):
        if novel_classes:
            # Create completely new classes (not in training GMM)
            if B == 1:
                # Each context item is from a different novel class
                novel_means = torch.randn(N, gmm.D) / np.sqrt(gmm.D)
                novel_labels = torch.randint(1, K_labels + 1, (N,), dtype=torch.float32)
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
                # N/B novel classes, each repeated B times
                novel_means = torch.randn(n_classes_in_context, gmm.D) / np.sqrt(gmm.D)
                novel_labels = torch.randint(1, K_labels + 1, (n_classes_in_context,), dtype=torch.float32)
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
            # Use existing GMM classes
            if B == 1:
                # Each context item can be from any GMM class
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
                # N/B GMM classes, each repeated B times
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


def generate_icl_gmm_data_with_label_swap(gmm, n_samples, N, exact_copy=True, B=1, K_classes=None):
    """
    Generate ICL data with SWAPPED labels for testing ICL (secondary metric).
    
    Uses existing GMM classes but with randomly permuted labels. This tests whether
    the model can learn new label mappings from context rather than relying on
    learned weights.
    
    Args:
        gmm: GaussianMixtureModel instance
        n_samples: Number of sequences to generate
        N: Number of context examples
        exact_copy: Whether query is exact copy of context item
        B: Burstiness (repetitions per class)
        K_classes: Number of label classes
        
    Returns:
        List of (z_seq, labels, target_label) tuples with swapped labels
    """
    assert 1 <= B <= N and N % B == 0
    n_classes_in_context = N // B
    K_labels = K_classes if K_classes is not None else gmm.K
    data = []
    
    for _ in range(n_samples):
        # Create a random label permutation (swap)
        label_permutation = torch.randperm(K_labels) + 1  # Permuted labels from 1 to K
        
        if B == 1:
            # Sample N classes from GMM
            class_indices = torch.randint(0, gmm.K, (N,))
            z_context = []
            labels = []
            for i in range(N):
                z_context.append(gmm.sample_from_class(class_indices[i].item()).squeeze(0))
                # Use swapped label instead of original
                original_label = int(gmm.get_label(class_indices[i].item()))
                swapped_label = label_permutation[original_label - 1].item()
                labels.append(float(swapped_label))
            
            copy_idx = torch.randint(0, N, (1,)).item()
            query_class = class_indices[copy_idx].item()
            if exact_copy:
                z_query = z_context[copy_idx].clone()
            else:
                z_query = gmm.sample_from_class(query_class).squeeze(0)
            
            # Target label is the swapped label
            original_target = int(gmm.get_label(query_class))
            target_label = float(label_permutation[original_target - 1].item())
        else:
            # Sample N/B classes from GMM
            context_classes = torch.randint(0, gmm.K, (n_classes_in_context,))
            z_context = []
            labels = []
            for class_idx in context_classes:
                # Get swapped label
                original_label = int(gmm.get_label(class_idx.item()))
                swapped_label = float(label_permutation[original_label - 1].item())
                
                for _ in range(B):
                    z_context.append(gmm.sample_from_class(class_idx.item()).squeeze(0))
                    labels.append(swapped_label)
            
            query_class_position = torch.randint(0, n_classes_in_context, (1,)).item()
            query_class = context_classes[query_class_position].item()
            if exact_copy:
                copy_offset = torch.randint(0, B, (1,)).item()
                z_query = z_context[query_class_position * B + copy_offset].clone()
            else:
                z_query = gmm.sample_from_class(query_class).squeeze(0)
            
            # Target label is the swapped label
            original_target = int(gmm.get_label(query_class))
            target_label = float(label_permutation[original_target - 1].item())
        
        z_seq = torch.stack(z_context + [z_query])
        data.append((z_seq, torch.tensor(labels), target_label))
    
    return data

