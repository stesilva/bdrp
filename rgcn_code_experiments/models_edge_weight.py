import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

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
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator with edge weight support.
    
    Modified from the original RGCN paper to support edge-attribute weighted aggregation:
    
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \alpha_{ij} \mathbf{\Theta}_r \cdot \mathbf{x}_j,
    
    where :math:`\alpha_{ij} = \frac{e_{ij}}{\sum_{k \in \mathcal{N}_r(i)} e_{ik}}`
    is the normalized edge weight.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

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

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, size=None):
        """
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_type: Relation type for each edge
            edge_norm: Legacy parameter (deprecated)
            edge_weight: Scalar edge attributes for weighted aggregation
            size: Size of source and target nodes
        """
        # For backward compatibility: if edge_norm is provided but not edge_weight, use edge_norm
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_weight=edge_weight)

    def message(self, x_j, edge_index_j, edge_type, edge_weight):
        """
        Compute messages from neighbors with edge-weighted aggregation.
        
        Args:
            x_j: Source node features [num_edges, in_channels]
            edge_index_i: Target node indices [num_edges]
            edge_type: Relation types [num_edges]
            edge_weight: Edge weights [num_edges]
        """
        # Compute relation-specific weights from basis decomposition
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, implement embedding lookup
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        # Apply edge-attribute weighted aggregation
        if edge_weight is not None:
            # Normalize edge weights per target node and relation type
            # Key: unique identifier for each (target_node, relation_type) pair
            dst_nodes = edge_index_j
            key = dst_nodes * self.num_relations + edge_type
            
            # Compute sum of edge weights for each (target, relation) pair
            num_nodes = x_j.size(0) if isinstance(x_j, torch.Tensor) else len(x_j)
            norm_denom = scatter(edge_weight, key, dim=0, 
                                dim_size=num_nodes * self.num_relations, 
                                reduce='sum')
            
            # Normalize: α_ij = e_ij / Σ_k e_ik
            normed_weights = edge_weight / (norm_denom[key] + 1e-8)
            
            # Apply normalized weights to messages
            out = out * normed_weights.view(-1, 1)

        return out

    def update(self, aggr_out, x):
        """
        Update node embeddings after aggregation.
        
        Args:
            aggr_out: Aggregated messages
            x: Original node features
        """
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)