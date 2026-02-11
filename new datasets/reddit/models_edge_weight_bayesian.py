import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, edge_weight_mode="normalize", edge_attr_dim=None):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))
        self.edge_attr_dim = edge_attr_dim  # Store for multi-attribute mode

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # Pass edge_weight_mode and edge_attr_dim to convolutional layers
        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode, edge_attr_dim=edge_attr_dim)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, 
            edge_weight_mode=edge_weight_mode, edge_attr_dim=edge_attr_dim)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm=None, edge_weight=None, edge_attr=None):
        """
        Args:
            entity: Entity indices
            edge_index: Graph connectivity
            edge_type: Relation type for each edge
            edge_norm: Legacy parameter (deprecated, use edge_weight instead)
            edge_weight: Scalar edge attributes for weighted aggregation (backward compatibility)
            edge_attr: Multi-dimensional edge attributes [num_edges, k] for multi-attribute Bayesian mode
        """
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_weight=edge_weight, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight, edge_attr=edge_attr)
        
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
                 root_weight=True, bias=True, edge_weight_mode="none", edge_attr_dim=None, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_weight_mode = edge_weight_mode
        self.edge_attr_dim = edge_attr_dim  # Store for multi-attribute Bayesian mode

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
            # MLP that takes concatenated features (x_j + edge_weight) and outputs transformed features
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(in_channels + 1, out_channels),
                nn.ReLU()
            )
        elif self.edge_weight_mode == "learnable":
            # Simple 2-layer MLP transforming scalar edge weight to scalar
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()  # Keep weights in (0,1)
            )
        elif self.edge_weight_mode == "bayesian":
            # Bayesian Multi-Attribute edge weights
            # Architecture: edge_attr -> Shared Encoder -> z -> (μ head, σ² head)
            # Supports both scalar confidence (backward compatibility) and multi-attribute edge_attr
            
            # Check if edge_attr_dim is provided (for multi-attribute mode)
            edge_attr_dim = getattr(self, 'edge_attr_dim', None)
            
            if edge_attr_dim is not None and edge_attr_dim > 1:
                # Multi-Attribute Bayesian Mode
                # Shared Encoder: learns joint representation from multiple attributes
                self.edge_encoder = nn.Sequential(
                    nn.Linear(edge_attr_dim, 16),  # k -> 16D
                    nn.ReLU(),
                    nn.Linear(16, 16),  # 16D -> 16D hidden representation
                    nn.ReLU()
                )
                # Mean head: estimates expected strength μ ∈ (0, 1)
                self.edge_weight_mean_head = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()  # μ in [0, 1]
                )
                # Variance head: estimates uncertainty σ² ≥ 0
                self.edge_weight_var_head = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Softplus()  # σ² ≥ 0 (ensures positive)
                )
                self.edge_weight_mean_mlp = None  # Not used in multi-attr mode
                self.edge_weight_var_mlp = None   # Not used in multi-attr mode
            else:
                # Single-attribute Bayesian Mode (backward compatibility)
                # Mean network: refines input confidence
                self.edge_weight_mean_mlp = nn.Sequential(
                    nn.Linear(1, 16),  # Input: confidence score
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()  # Mean in [0, 1]
                )
                # Variance network: learns uncertainty from confidence
                self.edge_weight_var_mlp = nn.Sequential(
                    nn.Linear(1, 16),  # Input: confidence score
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()  # Variance in [0, 1], will be scaled
                )
                self.edge_encoder = None
                self.edge_weight_mean_head = None
                self.edge_weight_var_head = None
            
            # Scale factor for variance (learnable, for single-attr mode)
            self.var_scale = nn.Parameter(torch.tensor(0.1))  # Initialize small variance
            # Set edge_weight_mlp to None since we use separate mean/var networks
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

        # Initialize edge_weight_mlp if it exists (for concat/learnable modes)
        if self.edge_weight_mlp is not None:
            if self.edge_weight_mode == "concat":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
            elif self.edge_weight_mode == "learnable":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
                nn.init.xavier_uniform_(self.edge_weight_mlp[2].weight)
                nn.init.zeros_(self.edge_weight_mlp[2].bias)
        
        # Initialize Bayesian networks (mean and variance)
        if self.edge_weight_mode == "bayesian":
            if self.edge_encoder is not None:
                # Initialize multi-attribute encoder
                for layer in self.edge_encoder:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                # Initialize mean head
                for layer in self.edge_weight_mean_head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                # Initialize variance head
                for layer in self.edge_weight_var_head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif self.edge_weight_mean_mlp is not None:
                # Single-Attribute Bayesian Mode: Initialize mean and variance MLPs
                # Initialize mean network to pass through confidence (identity-like)
                nn.init.xavier_uniform_(self.edge_weight_mean_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mean_mlp[0].bias)
                nn.init.xavier_uniform_(self.edge_weight_mean_mlp[2].weight)
                nn.init.zeros_(self.edge_weight_mean_mlp[2].bias)
                # Initialize last layer to be close to identity
                with torch.no_grad():
                    self.edge_weight_mean_mlp[4].weight.fill_(0.1)
                    self.edge_weight_mean_mlp[4].bias.fill_(0.0)
                
                # Initialize variance network: higher variance for lower confidence
                nn.init.xavier_uniform_(self.edge_weight_var_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_var_mlp[0].bias)
                nn.init.xavier_uniform_(self.edge_weight_var_mlp[2].weight)
                nn.init.zeros_(self.edge_weight_var_mlp[2].bias)
                # Initialize to produce higher variance for lower confidence
                with torch.no_grad():
                    self.edge_weight_var_mlp[4].weight.fill_(-0.5)  # Negative to invert confidence
                    self.edge_weight_var_mlp[4].bias.fill_(0.5)  # Bias to ensure some variance

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, edge_attr=None, size=None):
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_weight=edge_weight, edge_attr=edge_attr)

    def message(self, x_j, edge_index_j, edge_type, edge_weight, edge_attr=None):
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
                # Pass raw edge weight through MLP to get transformed scalar weight
                edge_weight = edge_weight.view(-1, 1)
                transformed_weight = self.edge_weight_mlp(edge_weight)  # Output shape [num_edges, 1]

                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

                # Scale messages by transformed weights
                out = out * transformed_weight.view(-1, 1)

            elif self.edge_weight_mode == "bayesian":
                # Multi-Attribute Bayesian Mode or Single-Attribute Mode
                if edge_attr is not None and self.edge_encoder is not None:
                    # Multi-Attribute Bayesian Mode
                    # edge_attr shape: [num_edges, k] where k is number of attributes
                    assert edge_attr.dim() == 2, f"edge_attr must be 2D [num_edges, k], got shape {edge_attr.shape}"
                    
                    # Shared Encoder: Learn joint representation z from edge attributes
                    z = self.edge_encoder(edge_attr)  # [num_edges, 16]
                    
                    # Mean head: Estimate expected strength μ ∈ (0, 1)
                    weight_mean = self.edge_weight_mean_head(z)  # [num_edges, 1]
                    
                    # Variance head: Estimate uncertainty σ² ≥ 0
                    weight_var = self.edge_weight_var_head(z)  # [num_edges, 1]
                    weight_var = weight_var + 1e-6  # Ensure positive
                    
                else:
                    # Single-Attribute Bayesian Mode (backward compatibility)
                    assert edge_weight is not None, "edge_weight must be provided for 'bayesian' mode"
                    edge_weight = edge_weight.view(-1, 1)  # Shape: [num_edges, 1]
                    
                    # Compute mean and variance of edge weight distribution
                    # μ: refined mean from input confidence
                    weight_mean = self.edge_weight_mean_mlp(edge_weight)  # [num_edges, 1]
                    
                    # σ²: variance (uncertainty) - higher for lower confidence
                    weight_var_raw = self.edge_weight_var_mlp(edge_weight)  # [num_edges, 1]
                    weight_var = weight_var_raw * torch.abs(self.var_scale) + 1e-6  # Ensure positive, scaled
                
                # Bayesian uncertainty-weighted aggregation
                # Formula: w_eff = μ / (1 + σ²)
                # High σ² reduces influence of unreliable edges
                effective_weight = weight_mean / (1.0 + weight_var)
                
                # Apply relation-specific transformation
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                
                # Scale messages by Bayesian uncertainty-weighted edge weights
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

