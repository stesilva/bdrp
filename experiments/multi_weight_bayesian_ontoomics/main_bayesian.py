import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import random
import os

from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models_edge_weight_per_relation import RGCN, EdgeWeightLogger


def train(train_triplets, train_edge_attr, model, use_cuda, batch_size, split_size, 
          negative_sample, reg_ratio, num_entities, num_relations, step=None):
    train_data = generate_sampled_graph_and_labels(triplets=train_triplets, 
                                                   sample_size=batch_size, 
                                                   split_size=split_size,
                                                   num_entity=num_entities, 
                                                   num_rels=num_relations, 
                                                   negative_rate=negative_sample,
                                                   edge_attr=train_edge_attr)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)
    
    # Get edge attributes (multi-dimensional) or edge weights (scalar)
    edge_attr = train_data.edge_attr if hasattr(train_data, 'edge_attr') else None
    edge_weight = train_data.edge_weight if hasattr(train_data, 'edge_weight') else None
    
    # Set current step for logging
    if step is not None:
        model.current_step = step
    
    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, 
                            edge_weight=edge_weight, edge_attr=edge_attr)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + \
           reg_ratio * model.reg_loss(entity_embedding)

    return loss


def valid(valid_triplets, model, test_graph, all_triplets, use_cuda, step=None, relation2id=None):
    if use_cuda:
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        test_graph = test_graph.to(device)
        valid_triplets = valid_triplets.to(device)
        all_triplets = all_triplets.to(device)

    edge_attr = test_graph.edge_attr if hasattr(test_graph, 'edge_attr') else None
    edge_weight = test_graph.edge_weight if hasattr(test_graph, 'edge_weight') else None
    
    # Set current step for logging
    if step is not None:
        model.current_step = step
    
    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, 
                            edge_weight=edge_weight, edge_attr=edge_attr)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, 
                   hits=[1, 3, 10], relation2id=relation2id)
    
    if use_cuda:
        torch.cuda.empty_cache()

    return mrr


def test(test_triplets, model, test_graph, all_triplets, use_cuda, step=None, relation2id=None):
    if use_cuda:
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        test_graph = test_graph.to(device)
        test_triplets = test_triplets.to(device)
        all_triplets = all_triplets.to(device)

    edge_attr = test_graph.edge_attr if hasattr(test_graph, 'edge_attr') else None
    edge_weight = test_graph.edge_weight if hasattr(test_graph, 'edge_weight') else None
    
    # Set current step for logging
    if step is not None:
        model.current_step = step
    
    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, 
                            edge_weight=edge_weight, edge_attr=edge_attr)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, 
                   hits=[1, 3, 10], relation2id=relation2id)
    
    if use_cuda:
        torch.cuda.empty_cache()

    return mrr


def main(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0

    # Load data with edge attributes
    entity2id, relation2id, (train_triplets, train_attr), (valid_triplets, valid_attr), (test_triplets, test_attr) = load_data('./data/biokg')
    
    # Determine edge_attr_dim for Bayesian mode
    edge_attr_dim = None
    if train_attr is not None:
        edge_attr_dim = train_attr.shape[1]
        print(f"\n✓ Detected {edge_attr_dim}-dimensional edge attributes")
        print(f"  This will be used for Bayesian multi-attribute learning")
    else:
        print("\n⚠️  No edge attributes found - using uniform weights")
    
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets, 
                                  edge_attr=train_attr, max_edges=args.test_graph_size)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    
    # ============================================================================
    # EDGE WEIGHT LOGGING SETUP
    # ============================================================================
    edge_weight_logger = None
    if args.log_edge_weights:
        log_dir = os.path.join(args.log_dir, f"run_{args.edge_weight_mode}_seed_{args.seed}")
        edge_weight_logger = EdgeWeightLogger(save_dir=log_dir)
        print(f"\n{'='*80}")
        print(f"Edge weight logging ENABLED")
        print(f"Log directory: {log_dir}")
        print(f"Logging frequency: every {args.log_frequency} epochs")
        print(f"{'='*80}\n")
    
    # ============================================================================
    # BASELINE MODE: Disable logging if mode is baseline (no weights to log)
    # ============================================================================
    if args.edge_weight_mode == "baseline":
        print(f"\n{'='*80}")
        print("BASELINE MODE: Running standard RGCN without edge weights")
        print("All edge weights/attributes will be ignored during training and evaluation")
        print(f"{'='*80}\n")
        # Disable logging for baseline since there are no weights to transform
        if edge_weight_logger is not None:
            print("Note: Edge weight logging disabled for baseline mode")
            edge_weight_logger = None
        # Set edge_attr_dim to None for baseline
        edge_attr_dim = None
    elif args.edge_weight_mode == "bayesian" and edge_attr_dim is None:
        print(f"\n{'='*80}")
        print("WARNING: Bayesian mode selected but no edge attributes found!")
        print("Falling back to uniform weights (equivalent to baseline)")
        print(f"{'='*80}\n")
    
    # Create model with edge_attr_dim
    print(f"\n{'='*80}")
    print("MODEL INITIALIZATION")
    print(f"{'='*80}")
    print(f"Mode: {args.edge_weight_mode}")
    if edge_attr_dim is not None:
        print(f"Edge attribute dimension: {edge_attr_dim}")
    print(f"{'='*80}\n")
    
    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, 
                dropout=args.dropout, edge_weight_mode=args.edge_weight_mode,
                edge_attr_dim=edge_attr_dim,
                edge_weight_logger=edge_weight_logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    if use_cuda:
        model.cuda()

    # Training loop
    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):
        model.train()
        optimizer.zero_grad()
        
        # Enable logging at specified frequency (skip if baseline mode)
        should_log = (args.log_edge_weights and 
                     edge_weight_logger is not None and 
                     args.edge_weight_mode != "baseline" and
                     (epoch % args.log_frequency == 0 or epoch == 1 or epoch == args.n_epochs))
        
        if should_log:
            edge_weight_logger.enable()
            tqdm.write(f"[Epoch {epoch}] Logging edge weights...")
        else:
            if edge_weight_logger is not None:
                edge_weight_logger.disable()

        loss = train(train_triplets, train_attr, model, use_cuda, 
                    batch_size=args.graph_batch_size, split_size=args.graph_split_size, 
                    negative_sample=args.negative_sample, reg_ratio=args.regularization, 
                    num_entities=len(entity2id), num_relations=len(relation2id),
                    step=epoch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        # Save logs if this was a logging epoch
        if should_log:
            edge_weight_logger.save(f"weights_epoch_{epoch:05d}.json")
            edge_weight_logger.save_summary(f"weights_summary_epoch_{epoch:05d}.json")
            edge_weight_logger.clear()  # Clear to save memory
            tqdm.write(f"[Epoch {epoch}] Saved edge weight logs")

        if epoch % args.evaluate_every == 0:
            tqdm.write(f"Train Loss {loss.item():.4f} at epoch {epoch}")

            if use_cuda:
                torch.cuda.empty_cache()

            model.eval()
            
            # Optional: log during validation (skip if baseline)
            if args.log_edge_weights and args.log_validation and args.edge_weight_mode != "baseline":
                edge_weight_logger.enable()
            
            valid_mrr = valid(valid_triplets, model, test_graph, all_triplets, use_cuda, 
                            step=f"valid_epoch_{epoch}", relation2id=relation2id)
            
            if args.log_edge_weights and args.log_validation and edge_weight_logger is not None:
                edge_weight_logger.save(f"weights_valid_epoch_{epoch:05d}.json")
                edge_weight_logger.save_summary(f"weights_summary_valid_epoch_{epoch:05d}.json")
                edge_weight_logger.clear()
                edge_weight_logger.disable()
            
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, 
                          'best_mrr_model.pth')

    # ============================================================================
    # FINAL TEST EVALUATION WITH LOGGING
    # ============================================================================
    model.eval()
    checkpoint = torch.load('best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    
    if use_cuda:
        model.cuda()

    # Enable logging for final test (skip if baseline)
    if args.log_edge_weights and args.edge_weight_mode != "baseline":
        edge_weight_logger.enable()
        tqdm.write("\n[FINAL TEST] Logging edge weights...")

    test_mrr = test(test_triplets, model, test_graph, all_triplets, use_cuda, 
                    step="final_test", relation2id=relation2id)
    
    # Save final test logs (skip if baseline)
    if args.log_edge_weights and edge_weight_logger is not None:
        edge_weight_logger.save("weights_final_test.json")
        edge_weight_logger.save_summary("weights_summary_final_test.json")
        tqdm.write(f"[FINAL TEST] Saved edge weight logs to {edge_weight_logger.save_dir}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("EDGE WEIGHT LOGGING SUMMARY")
        print(f"{'='*80}")
        print(f"Total log files saved: {len(list(edge_weight_logger.save_dir.glob('*.json')))}")
        print(f"Log directory: {edge_weight_logger.save_dir}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN with Bayesian Multi-Attribute Edge Weights')

    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=500)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--test-graph-size", type=int, default=-1,
                        help="Maximum number of training triplets to use for test graph (to avoid OOM). Set to -1 to use all triplets.")

    parser.add_argument("--edge-weight-mode", type=str, default="bayesian",
                        choices=["baseline", "normalize", "concat", "none", "learnable", "bayesian"],
                        help="""Edge weight mode:
                        - baseline: Standard RGCN (ignores all edge weights)
                        - normalize: Normalized edge weights
                        - concat: Concatenate weights to features
                        - none: Raw edge weight multiplication
                        - learnable: MLP-transformed weights
                        - bayesian: Uncertainty-aware Bayesian multi-attribute weights (RECOMMENDED)""")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # ============================================================================
    # EDGE WEIGHT LOGGING ARGUMENTS
    # ============================================================================
    parser.add_argument("--log-edge-weights", action="store_true",
                        help="Enable edge weight logging (original vs transformed)")
    parser.add_argument("--log-dir", type=str, default="edge_weight_logs",
                        help="Directory to save edge weight logs")
    parser.add_argument("--log-frequency", type=int, default=100,
                        help="Log edge weights every N epochs (default: 100)")
    parser.add_argument("--log-validation", action="store_true",
                        help="Also log edge weights during validation")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
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

    main(args)