# Edge Weight Integration Experiments

This directory contains experiments incorporating edge weights and confidence scores into R-GCN message passing.

## Overview

We explore different strategies for incorporating edge confidence scores (from datasets like PPI5k) into the R-GCN architecture to improve link prediction performance.

## Edge Weight Strategies

### 1. Normalized Edge Weights (`normalize`)
Normalizes edge weights by node degree, similar to standard edge normalization but using confidence scores.

### 2. Concatenated Features (`concat`)
Concatenates edge weights as additional node features during message passing.

### 3. Raw Multipliers (`none`)
Uses raw confidence scores as direct multipliers without normalization.

### 4. Learnable Transformation (`learnable`)
Uses a learnable MLP to transform edge weights before applying them.

## Datasets

- **PPI5k**: Protein-protein interaction graph with confidence scores in `softlogic.tsv`

## Usage

### Training with Normalized Edge Weights

```bash
python main.py --edge-weight-mode normalize --gpu 0 --dataset ppi5k
```

### Training with Learnable Edge Weights

```bash
python main.py --edge-weight-mode learnable --gpu 0 --dataset ppi5k
```

## Model Files

- `models_edge_weight.py`: R-GCN model with edge weight support
- `utils.py`: Data loading utilities with confidence score handling
- `main.py`: Training script with edge weight mode selection

## Key Contributions

1. **Edge Weight Integration**: First systematic exploration of confidence scores in R-GCN
2. **Multiple Strategies**: Comparison of normalization, concatenation, and learnable approaches
3. **PPI5k Dataset**: Adaptation of R-GCN for biological networks with confidence scores

## Results

Results demonstrate the impact of edge confidence on link prediction accuracy, particularly for noisy biological networks.

