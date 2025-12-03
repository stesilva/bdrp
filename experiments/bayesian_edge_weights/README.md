# Bayesian Edge Weights

This directory contains our uncertainty-aware Bayesian approach to edge weighting in R-GCN.

## Overview

We extend R-GCN with Bayesian neural network techniques to model uncertainty in edge weights, providing a principled approach to handling noisy or uncertain edges in knowledge graphs.

## Key Innovation

Instead of using fixed edge weights, we model edge weights as probability distributions, allowing the model to:
- Quantify uncertainty in edge confidence
- Adaptively weight edges based on learned uncertainty
- Improve robustness to noisy or uncertain edges

## Architecture

The Bayesian edge weight model:
- Models edge weights as distributions (e.g., Gaussian)
- Learns both mean and variance parameters
- Incorporates uncertainty into message passing

## Usage

### Training with Bayesian Edge Weights

```bash
python main_bayesian.py --edge-weight-mode bayesian --gpu 0 --dataset cn15k
```

### Comparison with Other Modes

```bash
# Bayesian (default)
python main_bayesian.py --edge-weight-mode bayesian

# Other modes for comparison
python main_bayesian.py --edge-weight-mode normalize
python main_bayesian.py --edge-weight-mode learnable
```

## Model Files

- `models_edge_weight_bayesian.py`: Bayesian R-GCN implementation
- `main_bayesian.py`: Training script with Bayesian edge weights
- `utils.py`: Data loading utilities

## Key Contributions

1. **Uncertainty Quantification**: First application of Bayesian methods to edge weighting in R-GCN
2. **Robustness**: Improved performance on noisy knowledge graphs
3. **Theoretical Foundation**: Principled approach to handling uncertain edges

## Results

Bayesian edge weighting shows improved performance, especially on datasets with noisy or uncertain edges, by allowing the model to learn which edges to trust.

