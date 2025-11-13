from rgcn_utils import *
from rgcn_conv import RGCNConv
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import torch


class DistMult(Module):
    """ DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """
    def __init__(self,
                 indim,
                 outdim,
                 num_nodes,
                 num_rel,
                 w_init='standard-normal',
                 w_gain=False,
                 b_init=None):
        super(DistMult, self).__init__()
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        # Create weights & biases
        # FIX: Sizing the relation matrix for augmented relations (2 * nrel + 1)
        self.relations = nn.Parameter(torch.FloatTensor(num_rel * 2 + 1, outdim))
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
            self.pbias = Parameter(torch.FloatTensor(num_rel * 2 + 1))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)
            self.register_parameter('pbias', None)

        self.initialise_parameters()

    def initialise_parameters(self):
        """
        Initialise weights and biases

        Options for initialising weights include:
            glorot-uniform - glorot (aka xavier) initialisation using a uniform distribution
            glorot-normal - glorot (aka xavier) initialisation using a normal distribution
            schlichtkrull-uniform - schlichtkrull initialisation using a uniform distribution
            schlichtkrull-normal - schlichtkrull initialisation using a normal distribution
            normal - using a standard normal distribution
            uniform - using a uniform distribution

        Options for initialising biases include:
            ones - setting all values to one
            zeros - setting all values to zero
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """
        # Weights
        init = select_w_init(self.w_init)
        if self.w_gain:
            gain = nn.init.calculate_gain('relu')
            init(self.relations, gain=gain)
        else:
            init(self.relations)

        # Checkpoint 6
        # print('min', torch.min(self.relations))
        # print('max', torch.max(self.relations))
        # print('mean', torch.mean(self.relations))
        # print('std', torch.std(self.relations))
        # print('size', self.relations.size())

        # Biases
        if self.b_init:
            init = select_b_init(self.b_init)
            init(self.sbias)
            init(self.pbias)
            init(self.obias)

    def s_penalty(self, triples, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = split_spo(triples)
        
        # Ensure indices are on the same device as nodes
        device = nodes.device
        s_index = s_index.to(device)
        p_index = p_index.to(device)
        o_index = o_index.to(device)

        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()
    
    
    def forward(self, triples, nodes):
        """ Score candidate triples """

        s_index, p_index, o_index = split_spo(triples)
        
        # Ensure indices are on the same device as nodes
        device = nodes.device
        s_index = s_index.to(device)
        p_index = p_index.to(device)
        o_index = o_index.to(device)

        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]

        scores = (s * p * o).sum(dim=-1)

        if self.b_init:
            scores = scores + (self.sbias[s_index] + self.pbias[p_index] + self.obias[o_index])

        return scores


class RelationalGraphConvolutionLP(Module):
    """
    Relational Graph Convolution (RGC) Layer for Link Prediction
    (as described in https://arxiv.org/abs/1703.06103)
    """

    def __init__(self,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 out_features=None,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 decomposition=None,
                 vertical_stacking=False,
                 w_init='glorot-normal',
                 w_gain=False,
                 b_init=None):
        super(RelationalGraphConvolutionLP, self).__init__()

        assert (num_nodes is not None or num_relations is not None or out_features is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        # Unpack arguments
        weight_decomp = decomposition['type'] if decomposition is not None and 'type' in decomposition else None
        num_bases = decomposition['num_bases'] if decomposition is not None and 'num_bases' in decomposition else None
        num_blocks = decomposition['num_blocks'] if decomposition is not None and 'num_blocks' in decomposition else None

        self.num_nodes = num_nodes
        self.num_relations_augmented = num_relations
        self.num_relations_original = (num_relations - 1) // 2 
        self.edge_dropout = edge_dropout

        # FIX: Initialize the core convolutional module using the imported RGCNConv
        self.conv = RGCNConv(
            in_channels=in_features,
            out_channels=out_features,
            num_relations=num_relations, # Pass the augmented count
            num_bases=num_bases,
            num_blocks=num_blocks,
            bias=(b_init is not None),
            aggr='mean' # Standard R-GCN aggregation
        )
        
    def forward(self, graph, features=None):
        """ Embed relational graph and perform message passing using the external RGCNConv. """
        
        device = features.device if features is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Graph Augmentation (Required for R-GCN logic) ---
        
        # 1. Original edges
        edge_index = graph[:, [0, 2]].t().contiguous().to(device)
        edge_type = graph[:, 1].contiguous().to(device)
        
        # 2. Inverse edges: swap s/o, set type to p + n_rel_original
        inv_edge_index = graph[:, [2, 0]].t().contiguous().to(device)
        inv_edge_type = (graph[:, 1] + self.num_relations_original).to(device)
        
        # 3. Self-loops: s=o, set type to 2 * n_rel_original
        num_nodes = self.num_nodes
        self_loop_edge_index = torch.arange(num_nodes, device=device).repeat(2, 1)
        self_loop_edge_type = torch.full((num_nodes,), 2 * self.num_relations_original, 
                                        dtype=torch.long, device=device)

        # Combine all edges (full_edge_index is used as the adjacency matrix indices)
        full_edge_index = torch.cat([edge_index, inv_edge_index, self_loop_edge_index], dim=1)
        full_edge_type = torch.cat([edge_type, inv_edge_type, self_loop_edge_type], dim=0)

        # --- Apply the R-GCN message passing (using the external layer) ---
        output = self.conv(features, full_edge_index, full_edge_type)
        
        return output
        