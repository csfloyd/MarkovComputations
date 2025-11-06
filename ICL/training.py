"""
Training functions for ICL models.

Provides training loops with ICL/IWL tracking.
"""

import torch
import torch.nn as nn
import numpy as np
from evaluation import evaluate_iwl, evaluate_icl_novel#, evaluate_icl_swap


def train_model(model, train_loader, val_loader, device, n_epochs=200, lr=0.001, 
                method='direct_solve', temperature=1.0, gmm=None, N=None, B=1, 
                L=None, exact_copy=True, eval_frequency=10, n_eval_samples=500):
    """
    Train the classification model with ICL/IWL tracking.
    
    Args:
        model: ICL model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        n_epochs: Number of training epochs
        lr: Learning rate
        method: Method for steady state computation (model-specific)
        temperature: Softmax temperature
        gmm: GaussianMixtureModel instance (needed for ICL/IWL evaluation)
        N: Number of context examples
        B: Burstiness parameter
        L: Number of output classes
        exact_copy: Whether query is exact copy of context item
        eval_frequency: How often to evaluate ICL/IWL (in epochs)
        n_eval_samples: Number of samples for each ICL/IWL evaluation
        
    Returns:
        dict: Training history with losses, accuracies, and ICL/IWL metrics
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # Use NLLLoss since model outputs log-probabilities
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        # ICL and IWL tracking
        'iwl_acc': [],           # In-Weight Learning
        'icl_acc': []     # ICL Primary: Novel classes
        #'icl_swap_acc': []       # ICL Secondary: Label swapping
    }
    
    for epoch in range(n_epochs):
        # === Training Phase ===
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for z_seq, labels, targets in train_loader:
            z_seq = z_seq.to(device)
            labels = labels.to(device)
            targets = targets.to(device).long() - 1  # Convert 1-indexed to 0-indexed
            
            optimizer.zero_grad()
            logits = model(z_seq, labels, method, temperature)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
        
        # === Validation Phase ===
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for z_seq, labels, targets in val_loader:
                z_seq = z_seq.to(device)
                labels = labels.to(device)
                targets = targets.to(device).long() - 1
                
                logits = model(z_seq, labels, method, temperature)
                loss = criterion(logits, targets)
                
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # === Evaluate ICL and IWL metrics periodically ===
        if gmm is not None and N is not None and (epoch + 1) % eval_frequency == 0:
            model.eval()
            
            # 1. IWL: Target class unlikely to appear in context
            iwl_acc = evaluate_iwl(
                model, gmm, N, device, n_eval_samples, L, method, temperature
            )
            history['iwl_acc'].append(iwl_acc)
            
            # 2. ICL Primary: Novel classes with B copies in context
            icl_novel_acc = evaluate_icl_novel(
                model, gmm, N, device, n_eval_samples, exact_copy, B, L, method, temperature
            )
            history['icl_acc'].append(icl_novel_acc)
            
            # # 3. ICL Secondary: Label swapping
            # icl_swap_acc = evaluate_icl_swap(
            #     model, gmm, N, device, n_eval_samples, exact_copy, B, L, method, temperature
            # )
            # history['icl_swap_acc'].append(icl_swap_acc)
        else:
            # Append None when not evaluating to keep list lengths consistent
            history['iwl_acc'].append(None)
            history['icl_novel_acc'].append(None)
            #history['icl_swap_acc'].append(None)
        
        # === Print Progress ===
        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1:3d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%"
            if (epoch + 1) % eval_frequency == 0 and gmm is not None:
                msg += f" | IWL: {iwl_acc:.2f}% | ICL: {icl_novel_acc:.2f}%"# | ICL-Swap: {icl_swap_acc:.2f}%"
            print(msg)
    
    return history

