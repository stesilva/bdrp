import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models import RGCN

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss

def valid(valid_triplets, model, test_graph, all_triplets, use_cuda):

    if use_cuda:
        # Clear GPU cache before validation to free up memory
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        test_graph = test_graph.to(device)
        valid_triplets = valid_triplets.to(device)
        all_triplets = all_triplets.to(device)

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])
    
    if use_cuda:
        # Clear cache after validation
        torch.cuda.empty_cache()

    return mrr

def test(test_triplets, model, test_graph, all_triplets, use_cuda):

    if use_cuda:
        # Clear GPU cache before testing
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        test_graph = test_graph.to(device)
        test_triplets = test_triplets.to(device)
        all_triplets = all_triplets.to(device)

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])
    
    if use_cuda:
        torch.cuda.empty_cache()

    return mrr

def main(args):

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./data/FB15k-237')
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    # Limit test graph size to avoid OOM (use max_edges=None to use all triplets if you have enough memory)
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets, max_edges=args.test_graph_size)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    
    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)

    if use_cuda:
        model.cuda()

    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):

        model.train()
        optimizer.zero_grad()

        loss = train(train_triplets, model, use_cuda, batch_size=args.graph_batch_size, split_size=args.graph_split_size, 
            negative_sample=args.negative_sample, reg_ratio = args.regularization, num_entities=len(entity2id), num_relations=len(relation2id))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        if epoch % args.evaluate_every == 0:

            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

            if use_cuda:
                # Clear GPU cache before validation
                torch.cuda.empty_cache()

            model.eval()
            valid_mrr = valid(valid_triplets, model, test_graph, all_triplets, use_cuda)
            
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                            'best_mrr_model.pth')
    
    model.eval()

    checkpoint = torch.load('best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    
    if use_cuda:
        model.cuda()

    test(test_triplets, model, test_graph, all_triplets, use_cuda)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RGCN')
    
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
    parser.add_argument("--test-graph-size", type=int, default=200000, 
                        help="Maximum number of training triplets to use for test graph (to avoid OOM). Set to -1 to use all triplets.")

    args = parser.parse_args()
    
    # Convert -1 to None to use all triplets
    if args.test_graph_size == -1:
        args.test_graph_size = None
    print(args)

    main(args)