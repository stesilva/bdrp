"""
R-GCN Link Prediction - Corrected Version
Proper train/val/test split without data leakage
"""

import torch
import torch.nn.functional as F
from rgcn_conv import RGCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from collections import defaultdict


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
            x = F.relu(x)
            if i < len(self.convs) - 1:  # No dropout on last layer
                x = F.dropout(x, p=0.1, training=self.training)  # Lower dropout
            x = F.normalize(x, p=2, dim=1)  # L2 normalization for stability
        
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


def load_dataset(local_path='dataset'):
    """
    Load dataset with proper train/val/test separation.
    Returns separate data structures to prevent data leakage.
    """
    print("Loading dataset...")

    def load_split(filename):
        return np.loadtxt(f"{local_path}/{filename}", dtype=str, delimiter='\t')
    
    # Load splits
    train_triples = load_split('train.tsv')
    val_triples = load_split('val.tsv')
    test_triples = load_split('test.tsv')

    # Build entity and relation ID mappings from ALL data
    entities = set()
    relations = set()
    for triples in [train_triples, val_triples, test_triples]:
        entities.update(triples[:, 0])
        entities.update(triples[:, 2])
        relations.update(triples[:, 1])

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    num_nodes = len(entity2id)
    num_relations = len(relation2id)

    # Convert training triples to PyG format
    def triples_to_tensors(triples):
        heads = torch.tensor([entity2id[h] for h in triples[:, 0]], dtype=torch.long)
        tails = torch.tensor([entity2id[t] for t in triples[:, 2]], dtype=torch.long)
        rels = torch.tensor([relation2id[r] for r in triples[:, 1]], dtype=torch.long)
        edge_index = torch.stack([heads, tails], dim=0)
        return edge_index, rels

    train_edge_index, train_edge_type = triples_to_tensors(train_triples)

    # Create training data object (only contains training edges!)
    train_data = Data(
        num_nodes=num_nodes,
        edge_index=train_edge_index,
        edge_type=train_edge_type,
    )

    print(f"Dataset loaded successfully!")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of relations: {num_relations}")
    print(f"Train triples: {len(train_triples)}")
    print(f"Validation triples: {len(val_triples)}")
    print(f"Test triples: {len(test_triples)}\n")

    return train_data, val_triples, test_triples, entity2id, relation2id, num_nodes, num_relations


def train_epoch(model, train_data, optimizer, device):
    """Train for one epoch using only training edges."""
    model.train()
    
    # Use ONLY training edges for message passing
    edge_index = train_data.edge_index.to(device)
    edge_type = train_data.edge_type.to(device)
    
    # Encode
    z = model.encode(edge_index, edge_type)
    
    # Positive samples (training edges)
    pos_scores = model.decode(z, edge_index, edge_type)
    
    # Negative sampling - corrupt BOTH head AND tail (more realistic for PPI)
    num_edges = edge_index.size(1)
    
    # Corrupt tails
    neg_edge_index_tail = negative_sampling(
        edge_index=edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=num_edges
    ).to(device)
    
    # Corrupt heads (swap source and target in negative sampling)
    neg_edge_index_head = negative_sampling(
        edge_index=torch.stack([edge_index[1], edge_index[0]]),  # Flipped
        num_nodes=train_data.num_nodes,
        num_neg_samples=num_edges
    ).to(device)
    neg_edge_index_head = torch.stack([neg_edge_index_head[1], neg_edge_index_head[0]])  # Flip back
    
    # Combine both types of corruptions
    neg_edge_index = torch.cat([neg_edge_index_tail, neg_edge_index_head], dim=1)
    neg_edge_type = torch.cat([edge_type, edge_type])
    
    neg_scores = model.decode(z, neg_edge_index, neg_edge_type)
    
    # Margin ranking loss (better for link prediction than BCE)
    pos_scores_expanded = pos_scores.repeat(2)  # Match negative sample size
    margin = 1.0
    loss = torch.mean(torch.clamp(margin - pos_scores_expanded + neg_scores, min=0))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, train_data, eval_triples, entity2id, relation2id, device, use_filtered=True):
    """
    Evaluate using proper knowledge graph metrics: MRR and Hits@K.
    
    Args:
        model: Trained R-GCN model
        train_data: Training graph data
        eval_triples: Evaluation triples
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        device: Device to run on
        use_filtered: If True, use filtered ranking (exclude known triples)
    
    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    
    # Encode using ONLY training graph
    edge_index = train_data.edge_index.to(device)
    edge_type = train_data.edge_type.to(device)
    z = model.encode(edge_index, edge_type)
    
    # Build set of all known triples for filtered evaluation
    if use_filtered:
        all_triples = set()
        # Add training triples
        for i in range(train_data.edge_index.size(1)):
            h = train_data.edge_index[0, i].item()
            t = train_data.edge_index[1, i].item()
            r = train_data.edge_type[i].item()
            all_triples.add((h, r, t))
        
        # Add eval triples
        for triple in eval_triples:
            h, r, t = triple[0], triple[1], triple[2]
            h_id = entity2id[h]
            t_id = entity2id[t]
            r_id = relation2id[r]
            all_triples.add((h_id, r_id, t_id))
    
    ranks = []
    
    # Evaluate each triple
    for idx, triple in enumerate(eval_triples):
        if (idx + 1) % 100 == 0:
            print(f"  Evaluated {idx + 1}/{len(eval_triples)} triples...", end='\r')
        
        h, r, t = triple[0], triple[1], triple[2]
        h_id = entity2id[h]
        t_id = entity2id[t]
        r_id = relation2id[r]
        
        # Score all possible tails for this (head, relation) pair
        # Create batch of all possible triples
        num_entities = train_data.num_nodes
        all_tails = torch.arange(num_entities, dtype=torch.long).to(device)
        batch_heads = torch.full((num_entities,), h_id, dtype=torch.long).to(device)
        batch_rels = torch.full((num_entities,), r_id, dtype=torch.long).to(device)
        
        # Stack into edge_index format
        batch_edge_index = torch.stack([batch_heads, all_tails], dim=0)
        
        # Score all candidates
        scores = model.decode(z, batch_edge_index, batch_rels)
        scores = scores.cpu().numpy()
        
        # Filter out known triples (except the target)
        if use_filtered:
            for candidate_t in range(num_entities):
                if candidate_t != t_id and (h_id, r_id, candidate_t) in all_triples:
                    scores[candidate_t] = -np.inf
        
        # Get rank of true tail
        true_score = scores[t_id]
        # Rank is number of entities with score > true_score, plus 1
        rank = np.sum(scores > true_score) + 1
        
        ranks.append(rank)
    
    print()  # New line after progress
    
    ranks = np.array(ranks)
    
    # Compute metrics
    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks <= 1)
    hits_at_3 = np.mean(ranks <= 3)
    hits_at_10 = np.mean(ranks <= 10)
    
    return {
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@10': hits_at_10
    }


def main():
    """Main training and evaluation loop."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    train_data, val_triples, test_triples, entity2id, relation2id, num_nodes, num_relations = load_dataset('dataset')
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of relations: {num_relations}")
    print(f"Train edges: {train_data.edge_index.size(1)}")
    print(f"Val edges: {len(val_triples)}")
    print(f"Test edges: {len(test_triples)}\n")
    
    # Initialize model with better config for PPI5K
    model = RGCNLinkPredictor(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_channels=128,  # Larger for biological networks
        num_layers=3  # More layers for complex interactions
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Optimizer with lower learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Training loop
    print("Training...")
    print("-" * 80)
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val MRR':>10} {'Hits@1':>10} {'Hits@10':>10}")
    print("-" * 80)
    
    best_val_mrr = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, 201):
        # Train
        loss = train_epoch(model, train_data, optimizer, device)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            print(f"\nValidating epoch {epoch}...")
            val_metrics = evaluate(model, train_data, val_triples, entity2id, relation2id, device)
            
            print(f"{epoch:5d} {loss:12.4f} {val_metrics['MRR']:10.4f} {val_metrics['Hits@1']:10.4f} {val_metrics['Hits@10']:10.4f}")
            
            # Early stopping based on MRR
            if val_metrics['MRR'] > best_val_mrr:
                best_val_mrr = val_metrics['MRR']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_rgcn_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print("-" * 80)
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load('best_rgcn_model.pt'))
    test_metrics = evaluate(model, train_data, test_triples, entity2id, relation2id, device)
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"MRR:      {test_metrics['MRR']:.4f}")
    print(f"Hits@1:   {test_metrics['Hits@1']:.4f}")
    print(f"Hits@3:   {test_metrics['Hits@3']:.4f}")
    print(f"Hits@10:  {test_metrics['Hits@10']:.4f}")
    print("="*70)
    
    # Example predictions
    print("\n" + "="*70)
    print("Example predictions on test set:")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        z = model.encode(train_data.edge_index.to(device), train_data.edge_type.to(device))
        
        # Sample a few test triples
        for i in range(min(5, len(test_triples))):
            triple = test_triples[i]
            h, r, t = triple[0], triple[1], triple[2]
            h_id = entity2id[h]
            t_id = entity2id[t]
            r_id = relation2id[r]
            
            score = model.decode(
                z,
                torch.tensor([[h_id], [t_id]], dtype=torch.long).to(device),
                torch.tensor([r_id], dtype=torch.long).to(device)
            ).item()
            
            prob = torch.sigmoid(torch.tensor(score)).item()
            
            print(f"Triple ({h}, {r}, {t}): score={score:.4f}, prob={prob:.4f}")


if __name__ == "__main__":
    main()