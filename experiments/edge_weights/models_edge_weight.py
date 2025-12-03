import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, edge_weight_mode="normalize"):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # Pass edge_weight_mode to convolutional layers
        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, edge_weight_mode=edge_weight_mode)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases, edge_weight_mode=edge_weight_mode)

        self.dropout_ratio = dropout

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
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight)
        
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
                 root_weight=True, bias=True, edge_weight_mode="none", **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_weight_mode = edge_weight_mode

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
        else:
            self.edge_weight_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

        if self.edge_weight_mlp is not None:
            if self.edge_weight_mode == "concat":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
            elif self.edge_weight_mode == "learnable":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
                nn.init.xavier_uniform_(self.edge_weight_mlp[2].weight)
                nn.init.zeros_(self.edge_weight_mlp[2].bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, size=None):
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_weight=edge_weight)

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
                # Pass raw edge weight through MLP to get transformed scalar weight
                edge_weight = edge_weight.view(-1, 1)
                transformed_weight = self.edge_weight_mlp(edge_weight)  # Output shape [num_edges, 1]

                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

                # Scale messages by transformed weights
                out = out * transformed_weight.view(-1, 1)

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
