import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter
import json
import os
from pathlib import Path

from utils import uniform

class EdgeWeightLogger:
    """Utility class to log edge weights during training/inference"""
    
    def __init__(self, save_dir="edge_weight_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
        self.enabled = False
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def log(self, layer_name, step, original_weights, transformed_weights, 
            edge_types=None, additional_info=None):
        """
        Log edge weights for a specific layer and step
        
        Args:
            layer_name: Name of the layer (e.g., 'conv1', 'conv2')
            step: Training step or batch number
            original_weights: Original edge weights [num_edges, k] or [num_edges, 1]
            transformed_weights: Transformed edge weights [num_edges, 1]
            edge_types: Edge type indices [num_edges] (optional)
            additional_info: Dict with any additional metadata (optional)
        """
        if not self.enabled:
            return
        
        # Handle multi-dimensional weights
        if original_weights.dim() > 1 and original_weights.shape[1] > 1:
            # Multi-dimensional: log stats per dimension
            orig_stats = {
                'mean': [float(original_weights[:, i].mean()) for i in range(original_weights.shape[1])],
                'std': [float(original_weights[:, i].std()) for i in range(original_weights.shape[1])],
                'min': [float(original_weights[:, i].min()) for i in range(original_weights.shape[1])],
                'max': [float(original_weights[:, i].max()) for i in range(original_weights.shape[1])],
                'num_dimensions': original_weights.shape[1]
            }
        else:
            # Single dimension
            orig_stats = {
                'mean': float(original_weights.mean()),
                'std': float(original_weights.std()),
                'min': float(original_weights.min()),
                'max': float(original_weights.max()),
            }
            
        log_entry = {
            'layer': layer_name,
            'step': step,
            'original_weights': orig_stats,
            'transformed_weights': {
                'mean': float(transformed_weights.mean()),
                'std': float(transformed_weights.std()),
                'min': float(transformed_weights.min()),
                'max': float(transformed_weights.max()),
            }
        }
        
        if edge_types is not None:
            log_entry['edge_types'] = edge_types.detach().cpu().numpy().tolist()
            
        if additional_info is not None:
            log_entry['additional_info'] = additional_info
            
        self.logs.append(log_entry)
    
    def save(self, filename="edge_weights.json"):
        """Save all logs to a JSON file"""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Saved edge weight logs to {filepath}")
        
    def save_summary(self, filename="edge_weights_summary.json"):
        """Save a summary (without full value arrays) for quick inspection"""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Saved edge weight summary to {filepath}")
        
    def clear(self):
        """Clear all logs"""
        self.logs = []


class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, 
                 edge_weight_mode="normalize", edge_attr_dim=None, edge_weight_logger=None):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))
        
        # Store edge_weight_mode and edge_attr_dim
        self.edge_weight_mode = edge_weight_mode
        self.edge_attr_dim = edge_attr_dim

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # Pass edge_attr_dim to convolutional layers
        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode,
            edge_attr_dim=edge_attr_dim,
            layer_name="conv1",
            edge_weight_logger=edge_weight_logger)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode,
            edge_attr_dim=edge_attr_dim,
            layer_name="conv2",
            edge_weight_logger=edge_weight_logger)

        self.dropout_ratio = dropout
        self.current_step = 0
        self.edge_weight_logger = edge_weight_logger

    def forward(self, entity, edge_index, edge_type, edge_norm=None, edge_weight=None, edge_attr=None):
        """
        Args:
            entity: Entity indices
            edge_index: Graph connectivity
            edge_type: Relation type for each edge
            edge_norm: Legacy parameter (deprecated, use edge_weight instead)
            edge_weight: Scalar edge attributes for weighted aggregation (backward compat)
            edge_attr: Multi-dimensional edge attributes [num_edges, k] for Bayesian mode
        """
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_weight=edge_weight, edge_attr=edge_attr,
                              step=self.current_step))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight, edge_attr=edge_attr,
                      step=self.current_step)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, edge_weight_mode="none", 
                 edge_attr_dim=None, layer_name=None, edge_weight_logger=None, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_weight_mode = edge_weight_mode
        self.edge_attr_dim = edge_attr_dim  # Number of weight dimensions (1, 2, 3, ...)
        self.layer_name = layer_name or "unnamed_layer"
        self.edge_weight_logger = edge_weight_logger

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.edge_weight_mode == "concat":
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(in_channels + 1, out_channels),
                nn.ReLU()
            )
        elif self.edge_weight_mode == "learnable":
            self.edge_weight_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )
                for _ in range(self.num_relations)
            ])
            self.edge_weight_mlp = None
        elif self.edge_weight_mode == "bayesian":
            # ================================================================
            # GENERALIZED BAYESIAN MODE - Handles k-dimensional weights
            # ================================================================
            
            # Determine input dimension for Bayesian networks
            if edge_attr_dim is not None and edge_attr_dim >= 1:
                input_dim = edge_attr_dim
                print(f"✓ Initializing Bayesian RGCN layer '{layer_name}' with {input_dim}-dimensional edge attributes per relation")
            else:
                input_dim = 1  # Fallback to single weight
                print(f"⚠️  edge_attr_dim not specified for layer '{layer_name}', defaulting to 1-dimensional")
            
            # One Bayesian transform (mean/var) per relation
            # Input: k-dimensional edge attributes -> Output: scalar mean and variance
            self.edge_weight_mean_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),  # k -> 16
                    nn.ReLU(),
                    nn.Linear(16, 8),          # 16 -> 8
                    nn.ReLU(),
                    nn.Linear(8, 1),           # 8 -> 1 (scalar mean)
                    nn.Sigmoid()               # Mean in [0, 1]
                )
                for _ in range(self.num_relations)
            ])
            self.edge_weight_var_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),  # k -> 16
                    nn.ReLU(),
                    nn.Linear(16, 8),          # 16 -> 8
                    nn.ReLU(),
                    nn.Linear(8, 1),           # 8 -> 1 (scalar variance)
                    nn.Sigmoid()               # Variance in [0, 1], will be scaled
                )
                for _ in range(self.num_relations)
            ])
            self.var_scale = nn.Parameter(torch.tensor(0.1))
            self.edge_weight_mlp = None
        else:
            self.edge_weight_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

        if self.edge_weight_mode == "concat" and self.edge_weight_mlp is not None:
            nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_mlp[0].bias)

        if self.edge_weight_mode == "learnable":
            for mlp in self.edge_weight_mlps:
                nn.init.xavier_uniform_(mlp[0].weight)
                nn.init.zeros_(mlp[0].bias)
                nn.init.xavier_uniform_(mlp[2].weight)
                nn.init.zeros_(mlp[2].bias)
        
        if self.edge_weight_mode == "bayesian":
            for mean_mlp in self.edge_weight_mean_mlps:
                for layer in mean_mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    mean_mlp[4].weight.fill_(0.1)
                    mean_mlp[4].bias.fill_(0.5)
            
            for var_mlp in self.edge_weight_var_mlps:
                for layer in var_mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    var_mlp[4].weight.fill_(0.1)
                    var_mlp[4].bias.fill_(0.5)

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, edge_attr=None,
                size=None, step=None):
        if self.edge_weight_mode == "baseline":
            edge_weight = None
            edge_norm = None
            edge_attr = None
        
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
        
        self.current_step = step
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, 
                            edge_weight=edge_weight, edge_attr=edge_attr)

    def message(self, x_j, edge_index_j, edge_type, edge_weight, edge_attr=None):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            # BASELINE MODE
            if self.edge_weight_mode == "baseline":
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                return out
            
            if self.edge_weight_mode == "concat":
                assert edge_weight is not None, "edge_weight must be provided for 'concat' mode"
                edge_weight = edge_weight.view(-1, 1)
                x_j_w_concat = torch.cat([x_j, edge_weight], dim=-1)
                out = self.edge_weight_mlp(x_j_w_concat)

            elif self.edge_weight_mode == "learnable":
                assert edge_weight is not None, "edge_weight must be provided for 'learnable' mode"
                edge_weight = edge_weight.view(-1, 1)
                original_weight = edge_weight.clone()

                transformed_weight = torch.zeros_like(edge_weight)
                for r in range(self.num_relations):
                    mask = (edge_type == r)
                    if not mask.any():
                        continue
                    ew_r = edge_weight[mask]
                    tw_r = self.edge_weight_mlps[r](ew_r)
                    transformed_weight[mask] = tw_r
                
                if (self.edge_weight_logger is not None and 
                    self.edge_weight_logger.enabled and 
                    self.current_step is not None):
                    self.edge_weight_logger.log(
                        layer_name=self.layer_name,
                        step=self.current_step,
                        original_weights=original_weight,
                        transformed_weights=transformed_weight,
                        edge_types=edge_type,
                        additional_info={'num_edges': len(edge_weight), 'training': self.training}
                    )

                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                out = out * transformed_weight.view(-1, 1)

            elif self.edge_weight_mode == "bayesian":
                # ================================================================
                # GENERALIZED BAYESIAN MODE
                # Handles k-dimensional edge attributes (k = 1, 2, 3, ...)
                # ================================================================
                
                if edge_attr is not None:
                    if edge_attr.dim() == 1:
                        edge_attr_input = edge_attr.view(-1, 1)
                    else:
                        edge_attr_input = edge_attr
                    
                    assert edge_attr_input.shape[1] == self.edge_attr_dim, \
                        f"Expected {self.edge_attr_dim} dimensions, got {edge_attr_input.shape[1]}"
                else:
                    assert edge_weight is not None, "Either edge_attr or edge_weight must be provided for Bayesian mode"
                    edge_attr_input = edge_weight.view(-1, 1)
                
                weight_mean = torch.zeros(edge_attr_input.shape[0], 1, device=edge_attr_input.device)
                weight_var = torch.zeros(edge_attr_input.shape[0], 1, device=edge_attr_input.device)

                for r in range(self.num_relations):
                    mask = (edge_type == r)
                    if not mask.any():
                        continue
                    
                    attr_r = edge_attr_input[mask]
                    wm_r = self.edge_weight_mean_mlps[r](attr_r)
                    wv_r_raw = self.edge_weight_var_mlps[r](attr_r)
                    wv_r = wv_r_raw * torch.abs(self.var_scale) + 1e-6

                    weight_mean[mask] = wm_r
                    weight_var[mask] = wv_r

                effective_weight = weight_mean / (1.0 + weight_var)
                
                if (self.edge_weight_logger is not None and 
                    self.edge_weight_logger.enabled and 
                    self.current_step is not None):
                    self.edge_weight_logger.log(
                        layer_name=self.layer_name,
                        step=self.current_step,
                        original_weights=edge_attr_input,
                        transformed_weights=effective_weight,
                        edge_types=edge_type,
                        additional_info={
                            'num_edges': len(edge_attr_input),
                            'num_dimensions': edge_attr_input.shape[1],
                            'training': self.training,
                            'weight_mean_stats': {'mean': float(weight_mean.mean()), 'std': float(weight_mean.std())},
                            'weight_var_stats': {'mean': float(weight_var.mean()), 'std': float(weight_var.std())},
                            'var_scale': float(self.var_scale)
                        }
                    )
                
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                out = out * effective_weight.view(-1, 1)

            else:
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

                if edge_weight is not None:
                    if self.edge_weight_mode == "normalize":
                        dst_nodes = edge_index_j
                        key = dst_nodes * self.num_relations + edge_type
                        num_nodes = x_j.size(0)
                        norm_denom = scatter(edge_weight, key, dim=0,
                                            dim_size=num_nodes * self.num_relations,
                                            reduce='sum')
                        normed_weights = edge_weight / (norm_denom[key] + 1e-8)
                        out = out * normed_weights.view(-1, 1)
                    elif self.edge_weight_mode == "none":
                        out = out * edge_weight.view(-1, 1)
                    else:
                        raise ValueError(f"Invalid edge_weight_mode: {self.edge_weight_mode}")

        return out

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, num_relations={self.num_relations}, edge_weight_mode={self.edge_weight_mode}, edge_attr_dim={self.edge_attr_dim})'