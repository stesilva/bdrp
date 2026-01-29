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
            original_weights: Original edge weights [num_edges, 1]
            transformed_weights: Transformed edge weights [num_edges, 1]
            edge_types: Edge type indices [num_edges] (optional)
            additional_info: Dict with any additional metadata (optional)
        """
        if not self.enabled:
            return
            
        log_entry = {
            'layer': layer_name,
            'step': step,
            'original_weights': {
                'mean': float(original_weights.mean()),
                'std': float(original_weights.std()),
                'min': float(original_weights.min()),
                'max': float(original_weights.max()),
                'values': original_weights.detach().cpu().numpy().flatten().tolist()
            },
            'transformed_weights': {
                'mean': float(transformed_weights.mean()),
                'std': float(transformed_weights.std()),
                'min': float(transformed_weights.min()),
                'max': float(transformed_weights.max()),
                'values': transformed_weights.detach().cpu().numpy().flatten().tolist()
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
        summary_logs = []
        for log in self.logs:
            summary = {
                'layer': log['layer'],
                'step': log['step'],
                'original_weights': {k: v for k, v in log['original_weights'].items() if k != 'values'},
                'transformed_weights': {k: v for k, v in log['transformed_weights'].items() if k != 'values'}
            }
            if 'additional_info' in log:
                summary['additional_info'] = log['additional_info']
            summary_logs.append(summary)
            
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary_logs, f, indent=2)
        print(f"Saved edge weight summary to {filepath}")
        
    def clear(self):
        """Clear all logs"""
        self.logs = []


class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, 
                 edge_weight_mode="normalize", edge_weight_logger=None):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # Pass edge_weight_logger to convolutional layers
        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode, 
            layer_name="conv1",
            edge_weight_logger=edge_weight_logger)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode,
            layer_name="conv2",
            edge_weight_logger=edge_weight_logger)

        self.dropout_ratio = dropout
        self.current_step = 0
        self.edge_weight_logger = edge_weight_logger

    def forward(self, entity, edge_index, edge_type, edge_norm=None, edge_weight=None):
        """
        Args:
            entity: Entity indices
            edge_index: Graph connectivity
            edge_type: Relation type for each edge
            edge_norm: Legacy parameter (deprecated, use edge_weight instead)
            edge_weight: Scalar edge attributes for weighted aggregation
        """
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_weight=edge_weight, 
                              step=self.current_step))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight,
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
                 layer_name=None, edge_weight_logger=None, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_weight_mode = edge_weight_mode
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
            # Single shared MLP when concatenating edge weights to features
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(in_channels + 1, out_channels),
                nn.ReLU()
            )
        elif self.edge_weight_mode == "learnable":
            # One learnable scalar transform per relation
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
            # One Bayesian transform (mean/var) per relation
            self.edge_weight_mean_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )
                for _ in range(self.num_relations)
            ])
            self.edge_weight_var_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
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

        # Edge-weight specific parameter initialisation
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
                nn.init.xavier_uniform_(mean_mlp[0].weight)
                nn.init.zeros_(mean_mlp[0].bias)
                nn.init.xavier_uniform_(mean_mlp[2].weight)
                nn.init.zeros_(mean_mlp[2].bias)
                with torch.no_grad():
                    mean_mlp[4].weight.fill_(0.1)
                    mean_mlp[4].bias.fill_(0.0)
            
            for var_mlp in self.edge_weight_var_mlps:
                nn.init.xavier_uniform_(var_mlp[0].weight)
                nn.init.zeros_(var_mlp[0].bias)
                nn.init.xavier_uniform_(var_mlp[2].weight)
                nn.init.zeros_(var_mlp[2].bias)
                with torch.no_grad():
                    var_mlp[4].weight.fill_(-0.5)
                    var_mlp[4].bias.fill_(0.5)

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, 
                size=None, step=None):
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
        
        # Store step for logging in message()
        self.current_step = step
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, 
                            edge_weight=edge_weight)

    def message(self, x_j, edge_index_j, edge_type, edge_weight):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            print("x_j is none")
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            if self.edge_weight_mode == "concat":
                assert edge_weight is not None, "edge_weight must be provided for 'concat' mode"
                edge_weight = edge_weight.view(-1, 1)
                x_j_w_concat = torch.cat([x_j, edge_weight], dim=-1)
                out = self.edge_weight_mlp(x_j_w_concat)

            elif self.edge_weight_mode == "learnable":
                assert edge_weight is not None, "edge_weight must be provided for 'learnable' mode"
                edge_weight = edge_weight.view(-1, 1)
                original_weight = edge_weight.clone()  # Store original

                # Allocate per-edge transformed weights
                transformed_weight = torch.zeros_like(edge_weight)

                # Apply the correct MLP per relation
                for r in range(self.num_relations):
                    mask = (edge_type == r)
                    if not mask.any():
                        continue
                    ew_r = edge_weight[mask]
                    tw_r = self.edge_weight_mlps[r](ew_r)
                    transformed_weight[mask] = tw_r
                
                # Log if logger is enabled
                if (self.edge_weight_logger is not None and 
                    self.edge_weight_logger.enabled and 
                    self.current_step is not None):
                    self.edge_weight_logger.log(
                        layer_name=self.layer_name,
                        step=self.current_step,
                        original_weights=original_weight,
                        transformed_weights=transformed_weight,
                        edge_types=edge_type,
                        additional_info={
                            'num_edges': len(edge_weight),
                            'training': self.training
                        }
                    )

                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                out = out * transformed_weight.view(-1, 1)

            elif self.edge_weight_mode == "bayesian":
                assert edge_weight is not None, "edge_weight must be provided for 'bayesian' mode"
                edge_weight = edge_weight.view(-1, 1)

                # Allocate tensors for per-edge mean and variance
                weight_mean = torch.zeros_like(edge_weight)
                weight_var = torch.zeros_like(edge_weight)

                # Compute relation-specific Bayesian transforms
                for r in range(self.num_relations):
                    mask = (edge_type == r)
                    if not mask.any():
                        continue
                    ew_r = edge_weight[mask]
                    wm_r = self.edge_weight_mean_mlps[r](ew_r)
                    wv_r_raw = self.edge_weight_var_mlps[r](ew_r)
                    wv_r = wv_r_raw * torch.abs(self.var_scale) + 1e-6

                    weight_mean[mask] = wm_r
                    weight_var[mask] = wv_r

                effective_weight = weight_mean / (1.0 + weight_var)
                
                # Log if logger is enabled
                if (self.edge_weight_logger is not None and 
                    self.edge_weight_logger.enabled and 
                    self.current_step is not None):
                    self.edge_weight_logger.log(
                        layer_name=self.layer_name,
                        step=self.current_step,
                        original_weights=edge_weight,
                        transformed_weights=effective_weight,
                        edge_types=edge_type,
                        additional_info={
                            'num_edges': len(edge_weight),
                            'training': self.training,
                            'weight_mean_stats': {
                                'mean': float(weight_mean.mean()),
                                'std': float(weight_mean.std())
                            },
                            'weight_var_stats': {
                                'mean': float(weight_var.mean()),
                                'std': float(weight_var.std())
                            },
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
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, num_relations={self.num_relations}, edge_weight_mode={self.edge_weight_mode})'