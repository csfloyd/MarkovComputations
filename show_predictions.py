"""
Show detailed predictions on novel class examples
Displays context, query, target, and model prediction for inspection
"""

import torch
import argparse
from markov_icl_gmm import ExactMarkovICL, LinearSolverMarkovICL, GaussianMixtureModel, generate_icl_gmm_data


def show_detailed_predictions(model, gmm, N, device, n_examples=5, B=1, exact_copy=True, 
                             label_min=None, label_max=None):
    """
    Show detailed predictions for manual inspection
    
    Args:
        model: Trained model
        gmm: GaussianMixtureModel (for generating novel data)
        N: Context length
        device: torch device
        n_examples: Number of examples to show
        B: Burstiness parameter
        exact_copy: Whether query is exact copy or new sample
        label_min: Min label for test data (None = use gmm default)
        label_max: Max label for test data (None = use gmm default)
    """
    model.eval()
    
    print("="*90)
    print(f"DETAILED PREDICTIONS ON NOVEL CLASSES (N={N}, B={B})")
    print("="*90)
    print(f"Query mode: {'EXACT COPY' if exact_copy else 'NEW SAMPLE (different noise)'}")
    print(f"Number of unique classes in context: {N // B}")
    lmin = label_min if label_min is not None else gmm.label_min
    lmax = label_max if label_max is not None else gmm.label_max
    print(f"Test label range: [{lmin:.2f}, {lmax:.2f}]")
    print("="*90)
    
    # Generate test data with novel classes
    # We'll manually generate and track class means for verification
    all_class_means = []  # Track all class means across examples
    
    with torch.no_grad():
        for example_idx in range(n_examples):
            # Generate one example at a time to track its class means
            test_data_single = generate_icl_gmm_data(gmm, n_samples=1, N=N, 
                                                     novel_classes=True, exact_copy=exact_copy, B=B,
                                                     label_min=lmin, label_max=lmax)
            z_seq, labels, target = test_data_single[0]
            
            # Infer the class means from the data (approximate by averaging items from same class)
            n_unique_classes = N // B
            inferred_means = []
            for class_idx in range(n_unique_classes):
                # Get all items from this class
                class_items = []
                for b_idx in range(B):
                    item_idx = class_idx * B + b_idx
                    class_items.append(z_seq[item_idx])
                # Average to approximate the class mean
                class_mean = torch.stack(class_items).mean(dim=0)
                inferred_means.append(class_mean)
            
            all_class_means.append(inferred_means)
            
            print(f"\n{'─'*90}")
            print(f"EXAMPLE {example_idx + 1}/{n_examples}")
            print(f"{'─'*90}")
            
            # Show the class means for this example
            print(f"\n🎲 NOVEL CLASS MEANS (sampled fresh for this example):")
            for class_idx, class_mean in enumerate(inferred_means):
                mean_str = ', '.join([f'{class_mean[d]:7.4f}' for d in range(len(class_mean))])
                print(f"  Class {class_idx + 1} mean ≈ [{mean_str}]")
            
            # Show context
            print("\n📝 CONTEXT (what the model learns from):")
            n_unique_classes = N // B
            
            if B == 1:
                # No burstiness: each item from different class
                for i in range(N):
                    z_i = z_seq[i]
                    label_i = labels[i]
                    z_str = ', '.join([f'{z_i[d]:7.4f}' for d in range(len(z_i))])
                    print(f"  {i+1}. z = [{z_str}] → label = {label_i:.4f}")
            else:
                # Burstiness: show which items are from same class
                for class_idx in range(n_unique_classes):
                    print(f"\n  Class {class_idx + 1} (appears {B} times):")
                    for b_idx in range(B):
                        item_idx = class_idx * B + b_idx
                        z_i = z_seq[item_idx]
                        label_i = labels[item_idx]
                        z_str = ', '.join([f'{z_i[d]:7.4f}' for d in range(len(z_i))])
                        print(f"    {item_idx+1}. z = [{z_str}] → label = {label_i:.4f}")
            
            # Show query
            print(f"\n🔍 QUERY (what we're predicting):")
            z_query = z_seq[-1]
            z_query_str = ', '.join([f'{z_query[d]:7.4f}' for d in range(len(z_query))])
            print(f"  z_query = [{z_query_str}]")
            
            # Make prediction
            z_seq_batch = z_seq.unsqueeze(0).to(device)
            labels_batch = labels.unsqueeze(0).to(device)
            prediction = model(z_seq_batch, labels_batch).item()
            
            # Show results
            print(f"\n🎯 RESULTS:")
            print(f"  Target label:     {target:.4f}")
            print(f"  Model prediction: {prediction:.4f}")
            error = abs(prediction - target)
            print(f"  Absolute error:   {error:.4f}")
            
            # Status
            if error < 0.05:
                status = "✅ EXCELLENT"
            elif error < 0.10:
                status = "✓ GOOD"
            elif error < 0.15:
                status = "○ OK"
            else:
                status = "✗ POOR"
            print(f"  Status: {status}")
            
            # Identify which context item(s) the query matches (for B>1)
            if B > 1:
                print(f"\n💡 Analysis:")
                # Find which class the query belongs to by checking which context items are closest
                distances = []
                for i in range(N):
                    dist = torch.norm(z_query - z_seq[i]).item()
                    distances.append((i, dist))
                
                # Sort by distance
                distances.sort(key=lambda x: x[1])
                closest_items = distances[:min(3, N)]
                
                print(f"  Closest context items to query:")
                for rank, (idx, dist) in enumerate(closest_items):
                    class_num = (idx // B) + 1
                    print(f"    {rank+1}. Item {idx+1} (Class {class_num}): distance = {dist:.4f}, label = {labels[idx]:.4f}")
    
    # Verification: Check that class means are different across examples
    print("\n" + "="*90)
    print("VERIFICATION: Are novel classes different across examples?")
    print("="*90)
    
    if n_examples > 1:
        # Compare class means between examples
        print("\nComparing Class 1 means across all examples:")
        for i in range(n_examples):
            mean = all_class_means[i][0]  # First class mean of example i
            mean_str = ', '.join([f'{mean[d]:7.4f}' for d in range(len(mean))])
            print(f"  Example {i+1}: [{mean_str}]")
        
        # Calculate pairwise distances
        print("\nPairwise distances between Class 1 means:")
        different_count = 0
        total_pairs = 0
        for i in range(n_examples):
            for j in range(i+1, n_examples):
                dist = torch.norm(all_class_means[i][0] - all_class_means[j][0]).item()
                print(f"  Example {i+1} ↔ Example {j+1}: distance = {dist:.4f}")
                total_pairs += 1
                if dist > 0.01:  # Threshold for "different"
                    different_count += 1
        
        print("\n" + "─"*90)
        if different_count == total_pairs:
            print(f"✅ CONFIRMED: All {total_pairs} pairs have different class means!")
            print("   → Novel classes ARE being sampled independently for each example")
        else:
            print(f"⚠️  WARNING: Only {different_count}/{total_pairs} pairs are different")
            print("   → There might be an issue with novel class generation")
    
    print("\n" + "="*90)
    print("DONE!")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description='Show detailed model predictions on novel classes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model')
    parser.add_argument('--K', type=int, default=100, help='Number of classes')
    parser.add_argument('--D', type=int, default=2, help='Dimension')
    parser.add_argument('--N', type=int, default=6, help='Number of context examples')
    parser.add_argument('--B', type=int, default=1, help='Burstiness (1 to N)')
    parser.add_argument('--n_nodes', type=int, default=10, help='Number of Markov nodes')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Within-class variability')
    parser.add_argument('--n_examples', type=int, default=5, help='Number of examples to show')
    parser.add_argument('--exact_copy', action='store_true', default=True)
    parser.add_argument('--no_exact_copy', dest='exact_copy', action='store_false')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random)')
    parser.add_argument('--method', type=str, default='exact', choices=['exact', 'linear'],
                       help='Method used for training: exact or linear')
    parser.add_argument('--label_min', type=float, default=None, help='Min label value for test (None = use training range)')
    parser.add_argument('--label_max', type=float, default=None, help='Max label value for test (None = use training range)')
    
    args = parser.parse_args()
    
    print("\n" + "="*90)
    print("SHOW PREDICTIONS ON NOVEL CLASSES")
    print("="*90)
    print(f"Model: {args.checkpoint}")
    print(f"Setup: K={args.K}, D={args.D}, N={args.N}, B={args.B}, ε={args.epsilon}")
    print("="*90 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create GMM (with different seed for novel classes, or None for random)
    # Default label range [0, 1] unless specified
    label_min = args.label_min if args.label_min is not None else 0.0
    label_max = args.label_max if args.label_max is not None else 1.0
    gmm = GaussianMixtureModel(K=args.K, D=args.D, epsilon=args.epsilon, seed=args.seed,
                               label_min=label_min, label_max=label_max)
    
    if args.label_min is not None or args.label_max is not None:
        print(f"⚠️  TESTING WITH SHIFTED LABELS: [{label_min:.2f}, {label_max:.2f}]")
        print(f"   (Model was likely trained on [0.0, 1.0])")
        print(f"   This tests label-shift generalization!\n")
    
    # Load model
    if args.method == 'exact':
        model = ExactMarkovICL(n_nodes=args.n_nodes, z_dim=args.D, n_labels=1, N=args.N)
    else:
        model = LinearSolverMarkovICL(n_nodes=args.n_nodes, z_dim=args.D, n_labels=1, N=args.N)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Show predictions
    show_detailed_predictions(model, gmm, args.N, device, 
                             n_examples=args.n_examples, 
                             B=args.B, 
                             exact_copy=args.exact_copy,
                             label_min=args.label_min,
                             label_max=args.label_max)


if __name__ == '__main__':
    main()

