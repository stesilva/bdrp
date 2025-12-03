# Standard R-GCN Baseline

This directory contains the baseline implementation of Relational Graph Convolutional Networks (R-GCN) for link prediction, based on PyTorch Geometric.

## Overview

This is a standard R-GCN implementation that reproduces the baseline results from the original paper:
- **Paper**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
- **Architecture**: 2-layer R-GCN encoder + DistMult decoder
- **Framework**: PyTorch Geometric

## Features

- Basis decomposition for parameter regularization
- Edge normalization
- Negative sampling for training
- Filtered evaluation metrics (MRR, Hits@1, Hits@3, Hits@10)

## Datasets

Tested on:
- **CN15k**: Chinese knowledge graph
- **FB15k-237**: Subset of Freebase with 237 relation types

## Usage

### Training on CN15k

```bash
python main.py --gpu 0 --n-epochs 10000 --evaluate-every 500
```

### Training on FB15k-237

```bash
cd FB15K
python main.py --gpu 0 --n-epochs 10000 --evaluate-every 500
```

## Model Architecture

- **Entity Embedding**: 100-dimensional embeddings
- **Relation Embedding**: 100-dimensional relation parameters
- **R-GCN Layers**: 2 layers with ReLU activation
- **Basis Decomposition**: Configurable number of bases (default: 4)
- **Dropout**: 0.2 (between layers)

## Results

Model checkpoints are saved as `best_mrr_model.pth` when validation MRR improves.

## Requirements

See `requirements.txt` in the root directory.

