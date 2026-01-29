import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import random
import os
import json

from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models_edge_weight_bayesian import RGCN, EdgeWeightLogger


def filter_triplets_by_relation(triplets, confidence, relation_id):
    """Filter triplets to only include specific relation type."""
    mask = triplets[:, 1] == relation_id
    filtered_triplets = triplets[mask]
    filtered_conf = confidence[mask] if confidence is not None else None
    return filtered_triplets, filtered_conf


def train_per_relation(train_triplets, train_confidence, model, use_cuda, batch_size, 
                       split_size, negative_sample, reg_ratio, num_entities, 
                       num_relations, relation_id, step=None):
    """Train on a specific relation type only."""
    # Filter to only this relation
    rel_triplets, rel_conf = filter_triplets_by_relation(
        train_triplets, train_confidence, relation_id
    )
    
    if len(rel_triplets) == 0:
        return None
    
    # Use smaller batch size if relation has few samples
    actual_batch_size = min(batch_size, len(rel_triplets))
    
    train_data = generate_sampled_graph_and_labels(
        rel_triplets, actual_batch_size, split_size, 
        num_entities, num_relations, negative_sample, 
        rel_conf
    )

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)
    
    edge_weight = train_data.edge_weight if hasattr(train_data, 'edge_weight') else getattr(train_data, 'edge_norm', None)
    
    if step is not None:
        model.current_step = step
    
    entity_embedding = model(train_data.entity, train_data.edge_index, 
                            train_data.edge_type, edge_weight=edge_weight)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + \
           reg_ratio * model.reg_loss(entity_embedding)

    return loss


def valid_per_relation(valid_triplets, model, test_graph, all_triplets, use_cuda, 
                       relation_id, step=None, relation2id=None):
    """Validate on a specific relation type only."""
    # Filter to only this relation
    rel_valid, _ = filter_triplets_by_relation(valid_triplets.cpu().numpy(), None, relation_id)
    
    if len(rel_valid) == 0:
        return 0.0
    
    rel_valid = torch.LongTensor(rel_valid)
    
    if use_cuda:
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        test_graph = test_graph.to(device)
        rel_valid = rel_valid.to(device)
        all_triplets = all_triplets.to(device)

    edge_weight = test_graph.edge_weight if hasattr(test_graph, 'edge_weight') else getattr(test_graph, 'edge_norm', None)
    
    if step is not None:
        model.current_step = step
    
    entity_embedding = model(test_graph.entity, test_graph.edge_index, 
                            test_graph.edge_type, edge_weight=edge_weight)
    
    # Create a filtered relation2id with just this relation
    rel_dict = None
    if relation2id is not None:
        for rel_name, rel_id in relation2id.items():
            if rel_id == relation_id:
                rel_dict = {rel_name: rel_id}
                break
    
    mrr = calc_mrr(entity_embedding, model.relation_embedding, rel_valid, 
                   all_triplets, hits=[1, 3, 10], relation2id=rel_dict)
    
    if use_cuda:
        torch.cuda.empty_cache()

    return mrr


def main_per_relation(args):
    """Main training loop with separate model per relation."""
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # Load data
    entity2id, relation2id, (train_triplets, train_conf), (valid_triplets, valid_conf), (test_triplets, test_conf) = load_data('./data/biokg')
    
    print(f"train_conf is None: {train_conf is None}")
    if train_conf is not None:
        print(f"Confidence stats: min={train_conf.min()}, max={train_conf.max()}, mean={train_conf.mean()}")
    
    num_relations = len(relation2id)
    num_entities = len(entity2id)
    
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    test_graph = build_test_graph(num_entities, num_relations, train_triplets, train_conf, 
                                  max_edges=args.test_graph_size)
    valid_triplets_tensor = torch.LongTensor(valid_triplets)
    test_triplets_tensor = torch.LongTensor(test_triplets)
    
    # Create reverse mapping for relation names
    id2relation = {v: k for k, v in relation2id.items()}
    
    # Storage for results
    results_per_relation = {}
    
    # Train separate model for each relation
    for rel_id in range(num_relations):
        rel_name = id2relation.get(rel_id, f"Relation_{rel_id}")
        
        # Count samples for this relation
        train_count = np.sum(train_triplets[:, 1] == rel_id)
        valid_count = np.sum(valid_triplets[:, 1] == rel_id)
        test_count = np.sum(test_triplets[:, 1] == rel_id)
        
        print(f"\n{'='*80}")
        print(f"Training Model for: {rel_name} (ID: {rel_id})")
        print(f"Train samples: {train_count}, Valid: {valid_count}, Test: {test_count}")
        print(f"{'='*80}")
        
        if train_count == 0:
            print(f"Skipping {rel_name} - no training samples")
            continue
        
        # Create model for this relation
        # Option 1: Full model (current approach - works fine)
        model = RGCN(num_entities, num_relations, num_bases=args.n_bases, 
                    dropout=args.dropout, edge_weight_mode=args.edge_weight_mode)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if use_cuda:
            model.cuda()
        
        best_mrr = 0
        best_epoch = 0
        
        # Training loop for this relation
        for epoch in trange(1, args.n_epochs + 1, desc=f'{rel_name}', position=0):
            model.train()
            optimizer.zero_grad()
            
            loss = train_per_relation(
                train_triplets, train_conf, model, use_cuda, 
                batch_size=args.graph_batch_size, 
                split_size=args.graph_split_size, 
                negative_sample=args.negative_sample, 
                reg_ratio=args.regularization, 
                num_entities=num_entities, 
                num_relations=num_relations,
                relation_id=rel_id,
                step=epoch
            )
            
            if loss is None:
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            
            # Validation
            if epoch % args.evaluate_every == 0:
                model.eval()
                valid_mrr = valid_per_relation(
                    valid_triplets_tensor, model, test_graph, all_triplets, 
                    use_cuda, rel_id, step=f"valid_epoch_{epoch}", 
                    relation2id=relation2id
                )
                
                tqdm.write(f"[{rel_name}] Epoch {epoch}: Loss={loss.item():.4f}, Valid MRR={valid_mrr:.4f}")
                
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    best_epoch = epoch
                    # Save best model for this relation
                    model_path = f'best_model_{args.edge_weight_mode}_rel_{rel_id}.pth'
                    torch.save({
                        'state_dict': model.state_dict(), 
                        'epoch': epoch,
                        'relation_id': rel_id,
                        'relation_name': rel_name
                    }, model_path)
        
        print(f"\n[{rel_name}] Best Valid MRR: {best_mrr:.4f} at epoch {best_epoch}")
        
        # Test with best model
        model.eval()
        checkpoint = torch.load(f'best_model_{args.edge_weight_mode}_rel_{rel_id}.pth')
        model.load_state_dict(checkpoint['state_dict'])
        
        if use_cuda:
            model.cuda()
        
        # Filter test triplets for this relation
        rel_test, _ = filter_triplets_by_relation(test_triplets, None, rel_id)
        if len(rel_test) > 0:
            rel_test_tensor = torch.LongTensor(rel_test)
            if use_cuda:
                rel_test_tensor = rel_test_tensor.cuda()
            
            test_mrr = valid_per_relation(
                torch.LongTensor(test_triplets), model, test_graph, 
                all_triplets, use_cuda, rel_id, step="final_test",
                relation2id=relation2id
            )
            
            results_per_relation[rel_name] = {
                'relation_id': rel_id,
                'train_samples': int(train_count),
                'test_samples': int(test_count),
                'best_valid_mrr': float(best_mrr),
                'test_mrr': float(test_mrr)
            }
            
            print(f"[{rel_name}] Test MRR: {test_mrr:.4f}")
        
        if use_cuda:
            torch.cuda.empty_cache()
    
    # Save all results
    results_file = f'per_relation_results_{args.edge_weight_mode}_seed_{args.seed}.json'
    with open(results_file, 'w') as f:
        json.dump(results_per_relation, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - Per-Relation Models ({args.edge_weight_mode.upper()})")
    print(f"{'='*80}")
    for rel_name, results in results_per_relation.items():
        print(f"{rel_name:40s} | Test MRR: {results['test_mrr']:.4f} | Samples: {results['test_samples']}")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-Relation RGCN Training')

    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=5000)
    parser.add_argument("--evaluate-every", type=int, default=500)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--test-graph-size", type=int, default=-1)

    parser.add_argument("--edge-weight-mode", type=str, default="learnable",
                        choices=["normalize", "concat", "none", "learnable", "bayesian"],
                        help="Edge weight usage mode")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.test_graph_size == -1:
        args.test_graph_size = None
    
    print(args)
    main_per_relation(args)