# Core R-GCN Implementation

This directory contains the core implementation of our R-GCN models for link prediction.

## Components

### Models (`models.py`)

1. **LinkPredictor**: Standard R-GCN with DistMult decoder
   - Configurable number of layers (1 or 2)
   - Basis or block decomposition
   - Customizable weight initialization

2. **CompressionRelationPredictor**: Bottleneck architecture for model compression
   - Encoding layer to compress embeddings
   - Decoding layer to reconstruct embeddings
   - Maintains compatibility with standard training

### Layers (`layers.py`)

- **RelationalGraphConvolutionLP**: R-GCN layer for link prediction
- **DistMult**: DistMult scoring function for triple scoring

### Convolution (`rgcn_conv.py`)

- **RGCNConv**: Efficient R-GCN convolution module using PyTorch Geometric

### Training (`predict_links.py`)

Main training script with support for:
- Multiple datasets (CN15k, NL27k)
- Configurable training parameters
- Evaluation metrics (MRR, Hits@K)
- Filtered evaluation

## Usage

### Basic Training

```python
from predict_links import train

config = {
    "dataset": {"name": "cn15k"},
    "training": {
        "epochs": 1000,
        "use_cuda": True,
        "graph_batch_size": 20000,
        ...
    },
    "encoder": {
        "model": "rgcn",  # or "c-rgcn" for compression
        "node_embedding": 200,
        "num_layers": 2,
        ...
    },
    ...
}

train(**config)
```

### Compression R-GCN

Set `encoder["model"]` to `"c-rgcn"` to use the compression architecture.

## Key Features

- **Flexible Architecture**: Support for standard and compression R-GCN
- **Multiple Decompositions**: Basis and block decomposition
- **Custom Initialization**: Glorot, Schlichtkrull, and standard normal initialization
- **Comprehensive Evaluation**: Filtered and raw metrics

