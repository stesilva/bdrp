"""
R-GCN Link Prediction Test
Tests the standard PyTorch Geometric RGCNConv on a synthetic knowledge graph
"""

import torch
import torch.nn.functional as F
from rgcn_conv import RGCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class RGCNLinkPredictor(torch.nn.Module):
    """
    R-GCN model for link prediction in knowledge graphs.
    Uses DistMult decoder for scoring entity-relation-entity triples.
    """
    
    def __init__(self, num_nodes, num_relations, hidden_channels=32, num_layers=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        
        # Node embeddings (learnable initial features)
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels)
        
        # R-GCN layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations)
            )
        
        # Relation embeddings for scoring
        self.rel_emb = torch.nn.Embedding(num_relations, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
    
    def encode(self, edge_index, edge_type):
        """
        Encode nodes using R-GCN layers.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
        
        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        x = self.node_emb.weight
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        return x
    
    def decode(self, z, edge_index, edge_type):
        """
        Decode using DistMult scoring function.
        Score(h, r, t) = <h, r, t> = Σ h_i * r_i * t_i
        
        Args:
            z: Node embeddings [num_nodes, hidden_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
        
        Returns:
            Scores for each edge [num_edges]
        """
        head = z[edge_index[0]]  # [num_edges, hidden]
        tail = z[edge_index[1]]  # [num_edges, hidden]
        rel = self.rel_emb(edge_type)  # [num_edges, hidden]
        
        # DistMult: element-wise multiplication and sum
        scores = torch.sum(head * rel * tail, dim=1)
        
        return scores
    
    def forward(self, edge_index, edge_type):
        """Full forward pass for link prediction."""
        z = self.encode(edge_index, edge_type)
        return z


def create_synthetic_knowledge_graph(num_nodes=50, num_relations=4, num_edges=200):
    """
    Create a synthetic knowledge graph for testing.
    
    Returns:
        data: PyG Data object with train/val/test splits
    """
    
    # Generate random edges with different relation types
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # Remove self-loops and duplicate edges
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_type = edge_type[mask]
    
    # Remove duplicates
    edge_set = set()
    filtered_edges = []
    filtered_types = []
    
    for i in range(edge_index.size(1)):
        edge = (edge_index[0, i].item(), edge_index[1, i].item(), edge_type[i].item())
        if edge not in edge_set:
            edge_set.add(edge)
            filtered_edges.append([edge_index[0, i].item(), edge_index[1, i].item()])
            filtered_types.append(edge_type[i].item())
    
    edge_index = torch.tensor(filtered_edges).t()
    edge_type = torch.tensor(filtered_types)
    
    # Split edges into train/val/test (70%/15%/15%)
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)
    
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    
    # Create data object
    data = Data(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_type=edge_type,
    )
    
    # Store splits
    data.train_idx = train_idx
    data.val_idx = val_idx
    data.test_idx = test_idx
    
    return data


def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    # Use only training edges for message passing
    train_edge_index = data.edge_index[:, data.train_idx].to(device)
    train_edge_type = data.edge_type[data.train_idx].to(device)
    
    # Encode
    z = model.encode(train_edge_index, train_edge_type)
    
    # Positive samples
    pos_scores = model.decode(z, train_edge_index, train_edge_type)
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edge_index.size(1)
    ).to(device)
    
    # Random edge types for negative samples
    neg_edge_type = torch.randint(
        0, model.num_relations, (neg_edge_index.size(1),)
    ).to(device)
    
    neg_scores = model.decode(z, neg_edge_index, neg_edge_type)
    
    # Binary cross-entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    
    loss = pos_loss + neg_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, device):
    """Evaluate on validation or test set."""
    model.eval()
    
    # Use all training edges for message passing
    train_edge_index = data.edge_index[:, data.train_idx].to(device)
    train_edge_type = data.edge_type[data.train_idx].to(device)
    
    # Encode
    z = model.encode(train_edge_index, train_edge_type)
    
    # Evaluate on the specified split
    eval_edge_index = data.edge_index[:, split_idx].to(device)
    eval_edge_type = data.edge_type[split_idx].to(device)
    
    # Positive samples
    pos_scores = model.decode(z, eval_edge_index, eval_edge_type)
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=eval_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=eval_edge_index.size(1)
    ).to(device)
    
    neg_edge_type = torch.randint(
        0, model.num_relations, (neg_edge_index.size(1),)
    ).to(device)
    
    neg_scores = model.decode(z, neg_edge_index, neg_edge_type)
    
    # Compute metrics
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).cpu().numpy()
    
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    return auc, ap


def main():
    """Main training and evaluation loop."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create synthetic dataset
    print("Creating synthetic knowledge graph...")
    data = create_synthetic_knowledge_graph(
        num_nodes=50,
        num_relations=4,
        num_edges=200
    )
    
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of relations: {data.edge_type.max().item() + 1}")
    print(f"Total edges: {data.edge_index.size(1)}")
    print(f"Train edges: {len(data.train_idx)}")
    print(f"Val edges: {len(data.val_idx)}")
    print(f"Test edges: {len(data.test_idx)}\n")
    
    # Initialize model
    model = RGCNLinkPredictor(
        num_nodes=data.num_nodes,
        num_relations=data.edge_type.max().item() + 1,
        hidden_channels=32,
        num_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    print("Training...")
    print("-" * 70)
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val AUC':>10} {'Val AP':>10}")
    print("-" * 70)
    
    best_val_auc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, 201):
        # Train
        loss = train_epoch(model, data, optimizer, device)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            val_auc, val_ap = evaluate(model, data, data.val_idx, device)
            
            print(f"{epoch:5d} {loss:12.4f} {val_auc:10.4f} {val_ap:10.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_rgcn_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print("-" * 70)
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load('best_rgcn_model.pt'))
    test_auc, test_ap = evaluate(model, data, data.test_idx, device)
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP:  {test_ap:.4f}")
    
    # Example prediction
    print("\n" + "="*70)
    print("Example predictions:")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        train_edge_index = data.edge_index[:, data.train_idx].to(device)
        train_edge_type = data.edge_type[data.train_idx].to(device)
        z = model.encode(train_edge_index, train_edge_type)
        
        # Sample a few test edges
        for i in range(min(5, len(data.test_idx))):
            idx = data.test_idx[i]
            head = data.edge_index[0, idx].item()
            tail = data.edge_index[1, idx].item()
            rel = data.edge_type[idx].item()
            
            score = model.decode(
                z,
                data.edge_index[:, idx:idx+1].to(device),
                data.edge_type[idx:idx+1].to(device)
            ).item()
            
            prob = torch.sigmoid(torch.tensor(score)).item()
            
            print(f"Edge ({head}, rel_{rel}, {tail}): score={score:.4f}, prob={prob:.4f}")


if __name__ == "__main__":
    main()