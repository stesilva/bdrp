import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def read_triplets_numeric(file_path):
    """Read triplets where entities/relations are already numeric IDs."""
    triplets = []

    with open(file_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            # Handle both 3-column and 4-column (with confidence) formats
            head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])
            # Ignore 4th column (confidence score) if present
            triplets.append((head, relation, tail))

    return np.array(triplets)

def load_data(file_path):
    '''
        argument:
            file_path: ./data/cn15k 
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    # Check if CN15k format (CSV files) or FB15k-237 format (dict files)
    entity_csv = os.path.join(file_path, 'entity_id.csv')
    relation_csv = os.path.join(file_path, 'relation_id.csv')
    entity_dict = os.path.join(file_path, 'entities.dict')
    relation_dict = os.path.join(file_path, 'relations.dict')

    if os.path.exists(entity_csv) and os.path.exists(relation_csv):
        # CN15k format: CSV files with header
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
        
        # CN15k uses .tsv files and numeric IDs directly in triplets
        train_triplets = read_triplets_numeric(os.path.join(file_path, 'train.tsv'))
        valid_triplets = read_triplets_numeric(os.path.join(file_path, 'val.tsv'))
        test_triplets = read_triplets_numeric(os.path.join(file_path, 'test.tsv'))
        
    elif os.path.exists(entity_dict) and os.path.exists(relation_dict):
        # FB15k-237 format: dict files
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

        train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
        valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
        test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)
    else:
        raise FileNotFoundError("Could not find entity/relation mapping files. Expected either entity_id.csv/relation_id.csv (CN15k) or entities.dict/relations.dict (FB15k-237)")

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    """Read triplets where entities/relations are strings that need mapping."""
    triplets = []

    with open(file_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            # Handle both 3-column (FB15k-237) and 4-column (CN15k with confidence) formats
            head, relation, tail = parts[0], parts[1], parts[2]
            # Ignore 4th column (confidence score) if present
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

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
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
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

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets, max_edges=None):
    """
    Build test graph from triplets.
    
    Args:
        num_nodes: Number of nodes
        num_rels: Number of relations
        triplets: Training triplets (numpy array)
        max_edges: Maximum number of edges to use (for memory efficiency). 
                   If None, uses all triplets. If specified, randomly samples max_edges.
    """
    if max_edges is not None and len(triplets) > max_edges:
        # Sample a subset of triplets to avoid OOM
        indices = np.random.choice(len(triplets), size=max_edges, replace=False)
        triplets = triplets[indices]
    
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src).long()
    rel = torch.from_numpy(rel).long()
    dst = torch.from_numpy(dst).long()

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():
        
        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

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
            ranks_s.append(sort_and_rank(score, target))

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
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
            
    return mrr.item()
