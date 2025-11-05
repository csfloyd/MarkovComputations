# ICL Codebase - Refactored Structure

## Overview

This codebase has been refactored into modular components for easy experimentation with different ICL model architectures.

## File Structure

```
ICL/
├── data_generation.py       # GMM and ICL data generation
├── datasets.py               # PyTorch dataset wrappers
├── models/
│   ├── __init__.py          # Model exports
│   ├── base_icl_model.py    # Abstract base class for all ICL models
│   └── markov_icl.py        # Matrix Tree Theorem implementation
├── training.py               # Training loop with ICL/IWL tracking
├── evaluation.py             # Testing and metrics
├── config.py                 # Configuration management
├── run_experiment.py         # Main experiment runner
└── analyze_data.ipynb        # Results analysis notebook
```

## Module Descriptions

### `data_generation.py`
- **`GaussianMixtureModel`**: GMM with K classes and discrete labels
- **`generate_icl_gmm_data()`**: Generate ICL training/testing data
- **`generate_icl_gmm_data_with_label_swap()`**: Generate label-swapped data for ICL testing

### `datasets.py`
- **`ICLGMMDataset`**: PyTorch Dataset wrapper
- **`collate_fn()`**: Batch collation function

### `models/base_icl_model.py`
- **`BaseICLModel`**: Abstract base class defining the ICL model interface
- All models must implement `forward()` method

### `models/markov_icl.py`
- **`MatrixTreeMarkovICL`**: Markov-based ICL model using Matrix Tree Theorem
- Supports multiple steady-state solvers: `matrix_tree`, `linear_solver`, `direct_solve`

### `training.py`
- **`train_model()`**: Main training loop with automatic ICL/IWL metric tracking

### `evaluation.py`
- **`test_icl()`**: Comprehensive ICL testing (in-dist + novel classes)
- **`evaluate_iwl()`**: In-Weight Learning metric
- **`evaluate_icl_novel()`**: ICL primary metric (novel classes)
- **`evaluate_icl_swap()`**: ICL secondary metric (label swapping)

### `config.py`
- **`ExperimentConfig`**: Dataclass for experiment configuration
- Centralizes all hyperparameters with validation

### `run_experiment.py`
- Main entry point that orchestrates all components
- Handles CLI arguments, training, testing, and saving results

## Usage

### Basic Training

```bash
python run_experiment.py --output results/exp1
```

### Custom Configuration

```bash
python run_experiment.py \
    --K 75 \
    --D 5 \
    --N 10 \
    --B 2 \
    --n_nodes 20 \
    --epochs 1000 \
    --lr 0.001 \
    --method direct_solve \
    --temperature 1.0 \
    --output results/exp1
```

### Resume from Checkpoint

```bash
python run_experiment.py \
    --resume results/exp1/model.pt \
    --output results/exp1_continued
```

## Adding New Models

To add a new model architecture (e.g., polynomial):

1. **Create model file**: `models/polynomial_icl.py`
2. **Inherit from BaseICLModel**:
   ```python
   from models.base_icl_model import BaseICLModel
   
   class PolynomialICL(BaseICLModel):
       def __init__(self, n_nodes=None, z_dim=2, K_classes=75, N=4):
           super().__init__(n_nodes, z_dim, K_classes, N)
           # Your initialization
       
       def forward(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0):
           # Your forward pass
           return logits  # (batch_size, K_classes)
   ```

3. **Export in `models/__init__.py`**:
   ```python
   from .polynomial_icl import PolynomialICL
   __all__ = ['BaseICLModel', 'MatrixTreeMarkovICL', 'PolynomialICL']
   ```

4. **Add to `run_experiment.py`**:
   ```python
   elif config.model_type == 'polynomial':
       model = PolynomialICL(...)
   ```

5. **Run experiments**:
   ```bash
   python run_experiment.py --model_type polynomial --output results/poly_exp1
   ```

## Output Structure

Each experiment creates:
```
results/exp1/
├── model.pt          # Model state dict
├── results.pkl       # Full results (history, metrics, config, timing)
└── config.json       # Human-readable configuration
```

## Metrics Tracked

- **Training**: Loss, Accuracy (per epoch)
- **Validation**: Loss, Accuracy (per epoch)
- **IWL**: In-Weight Learning accuracy (periodic)
- **ICL Novel**: Novel class ICL accuracy (periodic)
- **ICL Swap**: Label swap ICL accuracy (periodic)

## Key Features

1. **Modular**: Easy to swap model architectures
2. **Reproducible**: Seed control + config saving
3. **Comprehensive**: ICL/IWL metrics during training
4. **Extensible**: Clear interfaces for new models
5. **Clean**: Separation of concerns (data, model, training, eval)

## Migration from Old Code

The old monolithic files (`markov_icl_gmm_discrete_linear.py`, `run_icl.py`) are preserved but can be deprecated. The new structure provides:

- ✅ Same functionality
- ✅ Better organization
- ✅ Easier experimentation
- ✅ Cleaner imports
- ✅ Consistent interface

## Example: Compare Multiple Models

```python
# In your analysis notebook or script

from models import MatrixTreeMarkovICL, PolynomialICL
from training import train_model
from evaluation import test_icl

# Train Markov model
markov_model = MatrixTreeMarkovICL(...)
markov_history = train_model(markov_model, ...)
markov_results = test_icl(markov_model, ...)

# Train Polynomial model
poly_model = PolynomialICL(...)
poly_history = train_model(poly_model, ...)
poly_results = test_icl(poly_model, ...)

# Compare
print(f"Markov ICL: {markov_results['novel_classes']:.2f}%")
print(f"Polynomial ICL: {poly_results['novel_classes']:.2f}%")
```

## Contact

For questions or issues with the refactored codebase, refer to the original implementations in `polynomial_icl.py` or `markov_icl_gmm_discrete_linear.py`.

