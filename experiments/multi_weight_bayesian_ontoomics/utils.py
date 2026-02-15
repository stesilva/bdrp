import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import scatter

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def read_triplets_numeric(file_path):
    """Read triplets where entities/relations are already numeric IDs.
    
    Returns:
        triplets: numpy array of shape (n, 3) with columns [head, relation, tail]
        edge_attr: numpy array of shape (n, k) with k-dimensional weights, or None if not present
                   k can be 0 (no weights), 1 (single weight), or more (multi-weight)
    """
    triplets = []
    edge_attrs = []
    num_weights = None  # Will be determined from first line

    with open(file_path) as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            
            if len(parts) < 3:
                continue  # Skip malformed lines
                
            head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])
            triplets.append((head, relation, tail))
            
            # Determine number of weights from first line
            if line_idx == 0:
                num_weights = len(parts) - 3  # Everything after head, rel, tail
            
            # Extract weights if present
            if num_weights > 0:
                weights = [float(parts[3 + i]) for i in range(num_weights)]
                edge_attrs.append(weights)

    triplets = np.array(triplets)
    
    if num_weights > 0:
        edge_attrs = np.array(edge_attrs, dtype=np.float32)
        print(f"  Loaded {num_weights} weight(s) per edge")
        return triplets, edge_attrs
    else:
        print(f"  No edge weights found")
        return triplets, None

def load_data(file_path):
    '''
        argument:
            file_path: ./data/biokg 
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
            (and optionally k-dimensional edge attributes for each split)
    '''

    print("load data from {}".format(file_path))

    # Check for different file formats
    # Format 1: entities.txt / relations.txt (BioKG format with URIs)
    entity_txt = os.path.join(file_path, 'entities.txt')
    relation_txt = os.path.join(file_path, 'relations.txt')
    
    # Format 2: entity_id.csv / relation_id.csv (CN15k format)
    entity_csv = os.path.join(file_path, 'entity_id.csv')
    relation_csv = os.path.join(file_path, 'relation_id.csv')
    
    # Format 3: entities.dict / relations.dict (FB15k-237 format)
    entity_dict = os.path.join(file_path, 'entities.dict')
    relation_dict = os.path.join(file_path, 'relations.dict')

    # Load entity and relation mappings based on available files
    if os.path.exists(entity_txt) and os.path.exists(relation_txt):
        # BioKG format: entities.txt and relations.txt with URIs
        print("Detected BioKG format (entities.txt / relations.txt)")
        
        entity2id = {}
        with open(entity_txt, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid, entity_uri = parts[0], parts[1]
                    entity2id[entity_uri] = int(eid)
        
        relation2id = {}
        with open(relation_txt, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rid, relation_uri = parts[0], parts[1]
                    relation2id[relation_uri] = int(rid)
        
        # Load triplets with weights
        print("\nLoading train split...")
        train_triplets, train_attr = read_triplets_numeric(os.path.join(file_path, 'train.tsv'))
        print("Loading validation split...")
        valid_triplets, valid_attr = read_triplets_numeric(os.path.join(file_path, 'val.tsv'))
        print("Loading test split...")
        test_triplets, test_attr = read_triplets_numeric(os.path.join(file_path, 'test.tsv'))
        
    elif os.path.exists(entity_csv) and os.path.exists(relation_csv):
        # CN15k format: CSV files with header
        print("Detected CN15k format (entity_id.csv / relation_id.csv)")
        import csv
        
        entity2id = {}
        with open(entity_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                entity, eid = row[0], int(row[1])
                entity2id[entity] = eid
        
        relation2id = {}
        with open(relation_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                relation, rid = row[0], int(row[1])
                relation2id[relation] = rid
        
        print("\nLoading train split...")
        train_triplets, train_attr = read_triplets_numeric(os.path.join(file_path, 'train.tsv'))
        print("Loading validation split...")
        valid_triplets, valid_attr = read_triplets_numeric(os.path.join(file_path, 'val.tsv'))
        print("Loading test split...")
        test_triplets, test_attr = read_triplets_numeric(os.path.join(file_path, 'test.tsv'))
        
    elif os.path.exists(entity_dict) and os.path.exists(relation_dict):
        # FB15k-237 format: dict files
        print("Detected FB15k-237 format (entities.dict / relations.dict)")
        
        with open(entity_dict) as f:
            entity2id = dict()
            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(relation_dict) as f:
            relation2id = dict()
            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        print("\nLoading train split...")
        train_triplets, train_attr = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
        print("Loading validation split...")
        valid_triplets, valid_attr = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
        print("Loading test split...")
        test_triplets, test_attr = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)
    else:
        raise FileNotFoundError(
            "Could not find entity/relation mapping files. Expected one of:\n"
            "  - entities.txt / relations.txt (BioKG format)\n"
            "  - entity_id.csv / relation_id.csv (CN15k format)\n"
            "  - entities.dict / relations.dict (FB15k-237 format)"
        )

    print('\n' + '='*60)
    print('DATASET STATISTICS')
    print('='*60)
    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))
    
    # Print sample of relation names for BioKG format
    if os.path.exists(relation_txt):
        print("\nRelation types:")
        # Create reverse mapping for display
        id2relation = {v: k for k, v in relation2id.items()}
        for rid in sorted(id2relation.keys())[:10]:  # Show first 10
            rel_name = id2relation[rid].split('#')[-1]  # Extract name from URI
            print(f"  {rid}: {rel_name}")
        if len(id2relation) > 10:
            print(f"  ... and {len(id2relation) - 10} more")
    
    # Check and report on edge attributes
    print('\n' + '='*60)
    print('EDGE ATTRIBUTES')
    print('='*60)
    if train_attr is not None:
        num_weights = train_attr.shape[1]
        print(f'✓ Loaded {num_weights}-dimensional edge attributes')
        print(f'  Shape: {train_attr.shape} [num_edges, {num_weights}]')
        
        # Print statistics for each weight dimension
        weight_names = ['hastscore', 'hasescore', 'hasdscore']  # Default names
        for i in range(num_weights):
            weight_name = weight_names[i] if i < len(weight_names) else f'weight_{i}'
            non_zero = (train_attr[:, i] != 0).sum()
            non_zero_pct = (non_zero / len(train_attr)) * 100
            
            print(f'\n  {weight_name} (dim {i}):')
            print(f'    mean:     {train_attr[:, i].mean():.4f}')
            print(f'    std:      {train_attr[:, i].std():.4f}')
            print(f'    min:      {train_attr[:, i].min():.4f}')
            print(f'    max:      {train_attr[:, i].max():.4f}')
            print(f'    non-zero: {non_zero:,} / {len(train_attr):,} ({non_zero_pct:.1f}%)')
    else:
        print('⚠️  No edge attributes found - will use uniform weights')
    
    print('='*60 + '\n')

    return entity2id, relation2id, (train_triplets, train_attr), (valid_triplets, valid_attr), (test_triplets, test_attr)

def read_triplets(file_path, entity2id, relation2id):
    """Read triplets where entities/relations are strings that need mapping.
    
    Returns:
        triplets: numpy array of shape (n, 3) with columns [head, relation, tail]
        edge_attr: numpy array of shape (n, k) with k-dimensional weights, or None if not present
    """
    triplets = []
    edge_attrs = []
    num_weights = None

    with open(file_path) as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            
            if len(parts) < 3:
                continue
                
            head, relation, tail = parts[0], parts[1], parts[2]
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
            
            # Determine number of weights from first line
            if line_idx == 0:
                num_weights = len(parts) - 3
            
            # Extract weights if present
            if num_weights > 0:
                weights = [float(parts[3 + i]) for i in range(num_weights)]
                edge_attrs.append(weights)

    triplets = np.array(triplets)
    
    if num_weights > 0:
        edge_attrs = np.array(edge_attrs, dtype=np.float32)
        return triplets, edge_attrs
    else:
        return triplets, None

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick (degree-based)
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate, edge_attr=None):
    """
        Get training graph and signals with k-dimensional edge attributes.
        
        Args:
            triplets: Training triplets (numpy array of shape [N, 3])
            sample_size: Number of edges to sample
            split_size: Fraction of sampled edges to use as graph structure
            num_entity: Total number of entities
            num_rels: Number of relation types
            negative_rate: Number of negative samples per positive
            edge_attr: Optional edge attributes (numpy array of shape [N, k])
                      k=0: no weights (edge_attr=None)
                      k=1: single weight per edge
                      k>1: multi-dimensional weights per edge
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges_sampled = triplets[edges]  # Shape: [sample_size, 3]
    src, rel, dst = edges_sampled.transpose()
    
    # Sample corresponding edge attributes if available
    sampled_attr = None
    num_weights = 0
    if edge_attr is not None:
        sampled_attr = edge_attr[edges]  # Shape: [sample_size, k]
        num_weights = sampled_attr.shape[1] if len(sampled_attr.shape) > 1 else 1
        if num_weights == 1 and len(sampled_attr.shape) == 1:
            sampled_attr = sampled_attr.reshape(-1, 1)  # Ensure 2D
    
    uniq_entity, edges_relabeled = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges_relabeled, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()
    
    # Get edge attributes for graph edges
    if sampled_attr is not None:
        graph_attr = torch.tensor(sampled_attr[graph_split_ids], dtype=torch.float).contiguous()
    else:
        graph_attr = None

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))
    
    # Handle edge attributes for bidirectional edges
    if graph_attr is not None:
        # Duplicate edge attributes for bidirectional edges
        graph_attr = torch.cat((graph_attr, graph_attr))  # [2*split_size, k]
        
        # Set edge_attr (multi-dimensional)
        edge_attr_final = graph_attr
        
        # For backward compatibility: edge_weight as scalar
        if num_weights == 1:
            edge_weight_final = graph_attr.squeeze(-1)  # [num_edges]
        else:
            # Use mean of all weights as scalar representation
            edge_weight_final = graph_attr.mean(dim=1)  # [num_edges]
    else:
        # No weights provided - use uniform
        num_edges = src.size(0)
        edge_attr_final = None
        edge_weight_final = torch.ones(num_edges, dtype=torch.float)

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    
    # Set both edge_attr (multi-dim) and edge_weight (scalar)
    data.edge_attr = edge_attr_final
    data.edge_weight = edge_weight_final
        
    # Keep edge_norm for backward compatibility
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets, edge_attr=None, max_edges=None):
    """
    Build test graph from triplets with k-dimensional edge attributes.
    
    Args:
        num_nodes: Number of nodes
        num_rels: Number of relations
        triplets: Training triplets (numpy array)
        edge_attr: Optional edge attributes (numpy array of shape [N, k])
        max_edges: Maximum number of edges to use (for memory efficiency). 
                   If None, uses all triplets. If specified, randomly samples max_edges.
    """
    if max_edges is not None and len(triplets) > max_edges:
        # Sample a subset of triplets to avoid OOM
        indices = np.random.choice(len(triplets), size=max_edges, replace=False)
        triplets = triplets[indices]
        if edge_attr is not None:
            edge_attr = edge_attr[indices]
    
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src).long()
    rel = torch.from_numpy(rel).long()
    dst = torch.from_numpy(dst).long()
    
    # Handle edge attributes
    graph_attr = None
    num_weights = 0
    if edge_attr is not None:
        graph_attr = torch.from_numpy(edge_attr).float()
        num_weights = graph_attr.shape[1] if len(graph_attr.shape) > 1 else 1
        if num_weights == 1 and len(graph_attr.shape) == 1:
            graph_attr = graph_attr.reshape(-1, 1)
        # Duplicate for bidirectional edges
        graph_attr = torch.cat((graph_attr, graph_attr))  # [2N, k]

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))
    
    # Handle edge attributes
    if graph_attr is not None:
        edge_attr_final = graph_attr
        
        # Scalar weight for backward compatibility
        if num_weights == 1:
            edge_weight_final = graph_attr.squeeze(-1)
        else:
            edge_weight_final = graph_attr.mean(dim=1)
    else:
        # No weights - use uniform
        num_edges = src.size(0)
        edge_attr_final = None
        edge_weight_final = torch.ones(num_edges, dtype=torch.float)

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    
    # Set both formats
    data.edge_attr = edge_attr_final
    data.edge_weight = edge_weight_final
    
    # Keep edge_norm for backward compatibility
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[], num_rels=None, relation2id=None):
    """
    Calculate MRR and Hits with overall and per-relation breakdowns.
    
    Args:
        embedding: Entity embeddings
        w: Relation weights
        test_triplets: Test triplets to evaluate
        all_triplets: All known triplets (for filtering)
        hits: List of hit values to compute (e.g., [1, 3, 10])
        num_rels: Number of relations (optional, for better reporting)
        relation2id: Dict mapping relation URIs to IDs (optional, for better reporting)
    """
    with torch.no_grad():
        
        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []
        relations = []  # Track which relation each rank corresponds to

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            if embedding.is_cuda and not subject_relation.is_cuda:
                subject_relation = subject_relation.cuda()
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1)
            if delete_entity_index.is_cuda:
                delete_entity_index_np = delete_entity_index.cpu().numpy()
            else:
                delete_entity_index_np = delete_entity_index.numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index_np)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            if embedding.is_cuda:
                perturb_entity_index = perturb_entity_index.cuda()
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            
            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)
            
            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)
            
            target = torch.tensor(len(perturb_entity_index) - 1)
            if embedding.is_cuda:
                target = target.cuda()
            rank_s = sort_and_rank(score, target)
            ranks_s.append(rank_s)
            relations.append(relation.view(-1))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            if embedding.is_cuda:
                object_relation = object_relation.cuda()
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1)
            if delete_entity_index.is_cuda:
                delete_entity_index_np = delete_entity_index.cpu().numpy()
            else:
                delete_entity_index_np = delete_entity_index.numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index_np)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            if embedding.is_cuda:
                perturb_entity_index = perturb_entity_index.cuda()
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            if embedding.is_cuda:
                target = target.cuda()
            rank_o = sort_and_rank(score, target)
            ranks_o.append(rank_o)
            relations.append(relation.view(-1))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)
        relations = torch.cat(relations)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        # Overall metrics
        mrr = torch.mean(1.0 / ranks.float())
        mr = torch.mean(ranks.float())

        print("\n" + "="*60)
        print("OVERALL METRICS")
        print("="*60)
        print("MRR (filtered): {:.6f}".format(mrr.item()))
        print("MR (filtered): {:.6f}".format(mr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

        # Per-relation metrics
        print("\n" + "="*60)
        print("PER-RELATION METRICS")
        print("="*60)
        
        unique_relations = torch.unique(relations)
        
        # Create reverse mapping if relation2id provided
        id2relation = None
        if relation2id is not None:
            id2relation = {v: k for k, v in relation2id.items()}
        
        for rel in unique_relations:
            rel_mask = relations == rel
            rel_ranks = ranks[rel_mask]
            
            if len(rel_ranks) == 0:
                continue
                
            rel_mrr = torch.mean(1.0 / rel_ranks.float())
            rel_mr = torch.mean(rel_ranks.float())
            
            # Display relation name - extract from URI if available
            rel_name = f"Relation {rel.item()}"
            if id2relation is not None and rel.item() in id2relation:
                full_uri = id2relation[rel.item()]
                # Extract the readable name from URI (last part after #)
                readable_name = full_uri.split('#')[-1] if '#' in full_uri else full_uri
                rel_name = f"{readable_name} (ID: {rel.item()})"
            
            print(f"\n{rel_name}:")
            print(f"  Count: {len(rel_ranks)}")
            print(f"  MRR: {rel_mrr.item():.6f}")
            print(f"  MR: {rel_mr.item():.6f}")
            
            for hit in hits:
                rel_hit = torch.mean((rel_ranks <= hit).float())
                print(f"  Hits @ {hit}: {rel_hit.item():.6f}")
        
        print("="*60 + "\n")
            
    return mrr.item()