"""
Visualize Markov network with discrete labels
Generates network graphs with entropy production and full context display
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# Import from the discrete linear file
import sys
sys.path.insert(0, '.')
from markov_icl_gmm_discrete_linear import (
    GaussianMixtureModel, 
    generate_icl_gmm_data,
    MatrixTreeMarkovICL
)


def compute_entropy_production(W, pi):
    """Compute entropy production rate."""
    n = W.shape[0]
    sigma = 0.0
    
    for i in range(n):
        for j in range(n):
            if i != j and W[i,j] > 1e-10 and W[j,i] > 1e-10:
                J_ij = pi[j] * W[i,j]
                J_ji = pi[i] * W[j,i]
                if J_ij > 1e-10 and J_ji > 1e-10:
                    sigma += J_ij * np.log(J_ij / J_ji)
    
    return sigma


def visualize_network(W, pi, context_info, output_file, example_idx=0):
    """Visualize one example with network graph and context."""
    n = W.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    
    # Left: Network graph
    ax_graph = plt.subplot(1, 2, 1)
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # Add edges with weights
    threshold = np.max(W) * 0.05
    edges_to_draw = []
    for i in range(n):
        for j in range(n):
            if i != j and W[i,j] > threshold:
                G.add_edge(j, i, weight=W[i,j])
                edges_to_draw.append((j, i, W[i,j]))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Draw nodes (size proportional to steady-state)
    node_sizes = [max(100, pi[i] * 8000) for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=pi, cmap='YlOrRd',
                          vmin=0, vmax=max(pi),
                          ax=ax_graph, edgecolors='black', linewidths=1.5)
    
    # Draw edges (width proportional to rate)
    max_weight = max([w for _, _, w in edges_to_draw]) if edges_to_draw else 1.0
    for u, v, w in edges_to_draw:
        width = max(0.5, (w / max_weight) * 4)
        nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                              alpha=0.6, edge_color='gray',
                              arrows=True, arrowsize=15,
                              ax=ax_graph, connectionstyle="arc3,rad=0.1")
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax_graph)
    
    # Compute entropy production
    sigma = compute_entropy_production(W, pi)
    
    ax_graph.set_title(f'Network (σ={sigma:.3f})', fontsize=12, fontweight='bold')
    ax_graph.axis('off')
    
    # Right: Context information
    ax_text = plt.subplot(1, 2, 2)
    ax_text.axis('off')
    
    # Extract context info
    z_context = context_info['z_context']
    labels = context_info['labels']
    z_query = context_info['z_query']
    pred = context_info['pred']
    target = context_info['target']
    B = context_info.get('B', 2)
    
    N = len(z_context)
    n_classes = N // B
    
    # Build text display
    text_lines = []
    text_lines.append(f"Example {example_idx + 1} (Discrete Labels)")
    text_lines.append("="*40)
    text_lines.append("")
    text_lines.append(f"CONTEXT (N={N}, B={B}):")
    text_lines.append("-"*40)
    
    # Display context with burstiness structure
    for class_idx in range(n_classes):
        text_lines.append(f"\nClass {class_idx+1} (label={int(labels[class_idx*B])}):")
        for b in range(B):
            idx = class_idx * B + b
            vec_str = ", ".join([f"{v:.3f}" for v in z_context[idx]])
            text_lines.append(f"  [{vec_str}]")
    
    text_lines.append("")
    text_lines.append("-"*40)
    text_lines.append("QUERY:")
    vec_str = ", ".join([f"{v:.3f}" for v in z_query])
    text_lines.append(f"[{vec_str}]")
    
    text_lines.append("")
    text_lines.append("="*40)
    text_lines.append(f"TRUE LABEL:      {int(target)}")
    text_lines.append(f"PREDICTED:       {pred:.2f}")
    text_lines.append(f"ERROR:           {abs(pred - target):.3f}")
    text_lines.append("="*40)
    
    # Display text
    text_content = "\n".join(text_lines)
    ax_text.text(0.05, 0.98, text_content, 
                transform=ax_text.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--K_classes', type=int, default=None)
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--N', type=int, default=6)
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--n_nodes', type=int, default=25)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--n_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default='direct_solve')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.K_classes is None:
        args.K_classes = args.K
    
    print("="*70)
    print("VISUALIZE DISCRETE LABEL MARKOV NETWORK")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Parameters: K={args.K}, D={args.D}, N={args.N}, B={args.B}")
    print(f"Nodes: {args.n_nodes}, Method: {args.method}")
    print(f"Labels: {args.K_classes} discrete classes")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Load model
    model = MatrixTreeMarkovICL(n_nodes=args.n_nodes, z_dim=args.D, n_labels=1, N=args.N)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    model.eval()
    print(f"\n✓ Loaded model from {args.checkpoint}")
    
    # Create GMM
    gmm = GaussianMixtureModel(K=args.K, D=args.D, epsilon=args.epsilon, seed=args.seed)
    print(f"✓ Created GMM with {args.K} classes")
    print(f"  Discrete labels: 1 to {args.K_classes}")
    
    # Generate examples and visualize
    examples_data = []
    
    for idx in range(args.n_examples):
        seed = args.seed + idx + 1000
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate novel class data
        test_data = generate_icl_gmm_data(gmm, 1, args.N, novel_classes=True, 
                                         exact_copy=False, B=args.B, 
                                         K_classes=args.K_classes)
        z_seq, labels_seq, target = test_data[0]
        
        # Get prediction
        with torch.no_grad():
            pred = model(z_seq.unsqueeze(0), labels_seq.unsqueeze(0), 
                        method=args.method).item()
        
        # Extract W matrix and steady state
        z_flat = z_seq.flatten().unsqueeze(0)
        K_batch = model.compute_rate_matrix_K(z_flat)
        
        if args.method == 'matrix_tree':
            pi = model.matrix_tree_steady_state(K_batch)
        elif args.method == 'linear_solver':
            pi = model.linear_solver_steady_state(K_batch)
        else:
            pi = model.direct_solve_steady_state(K_batch)
        
        pi = pi.squeeze(0).detach().cpu().numpy()
        W = K_batch.squeeze(0).detach().cpu().numpy()
        
        # Prepare context info
        z_context = [z_seq[i].numpy() for i in range(args.N)]
        labels = labels_seq.numpy()
        z_query = z_seq[-1].numpy()
        
        context_info = {
            'z_context': z_context,
            'labels': labels,
            'z_query': z_query,
            'pred': pred,
            'target': target.item(),
            'B': args.B
        }
        
        examples_data.append({
            'W': W,
            'pi': pi,
            'context_info': context_info
        })
        
        print(f"\n{'='*70}")
        print(f"Example {idx+1}:")
        print(f"  True label: {int(target.item())}")
        print(f"  Predicted:  {pred:.2f}")
        print(f"  Error:      {abs(pred - target.item()):.3f}")
        print(f"  Entropy σ:  {compute_entropy_production(W, pi):.3f}")
    
    # Save each example as separate PDF
    for idx, ex_data in enumerate(examples_data):
        output_name = f'discrete_example_{idx+1}.pdf'
        visualize_network(ex_data['W'], ex_data['pi'], 
                        ex_data['context_info'], output_name, idx)
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {args.n_examples} visualization(s)")
    print(f"  Files: discrete_example_1.pdf, discrete_example_2.pdf, ...")
    print("="*70)


if __name__ == '__main__':
    main()

