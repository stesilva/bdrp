#!/usr/bin/env python3
"""
Convert Reddit Hyperlink Network data to format expected by Bayesian RGCN model.

The Reddit dataset has:
- SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP, POST_LABEL, POST_PROPERTIES
- POST_LABEL: -1 (negative) or +1 (positive/neutral)
- POST_PROPERTIES: comma-separated vector of text properties

We convert this to:
- Triple format: (source_subreddit_id, relation_id, target_subreddit_id)
- Confidence scores: derived from POST_PROPERTIES or POST_LABEL
- Relation types: We can use POST_LABEL as relation type, or treat all as same relation
"""

import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def normalize_confidence(properties_vector):
    """
    Derive confidence score from POST_PROPERTIES vector.
    Uses compound sentiment and other features to create a confidence score.
    Returns value in [0, 1] range.
    """
    if len(properties_vector) < 21:  # Need at least compound sentiment (index 21)
        return 0.5  # Default confidence
    
    # Use compound sentiment (index 21) and normalize to [0, 1]
    compound_sentiment = properties_vector[21]
    # VADER compound sentiment is in [-1, 1], normalize to [0, 1]
    confidence = (compound_sentiment + 1) / 2
    
    # Also consider text quality metrics (e.g., readability, length)
    # Longer, more readable posts might be more reliable
    if len(properties_vector) > 17:
        readability = properties_vector[17]  # Automated readability index
        # Normalize readability (typically 0-100, but can vary)
        readability_norm = min(max(readability / 100.0, 0), 1)
        # Combine sentiment and readability
        confidence = 0.7 * confidence + 0.3 * readability_norm
    
    return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

def load_reddit_data(tsv_file, use_sentiment_as_relation=True, use_properties_for_confidence=True):
    """
    Load Reddit hyperlink network data.
    
    Args:
        tsv_file: Path to Reddit TSV file
        use_sentiment_as_relation: If True, use POST_LABEL (-1/+1) as relation type
                                   If False, use single relation type for all links
        use_properties_for_confidence: If True, derive confidence from POST_PROPERTIES
                                       If False, use uniform confidence
    
    Returns:
        triplets: numpy array of shape (n, 3) with [head, relation, tail]
        confidences: numpy array of shape (n,) with confidence scores
        entity2id: dict mapping subreddit name to entity ID
        relation2id: dict mapping relation name to relation ID
    """
    print(f"Loading Reddit data from {tsv_file}...")
    
    # Read TSV file
    data = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Skip header row
        print(f"Header: {header}")
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            
            source_subreddit = parts[0]
            target_subreddit = parts[1]
            post_id = parts[2]
            timestamp = parts[3]
            try:
                post_label = int(parts[4])  # LINK_SENTIMENT: -1 or +1
            except ValueError:
                continue  # Skip header or invalid lines
            post_properties_str = parts[5]
            
            # Parse POST_PROPERTIES vector
            try:
                properties_vector = [float(x) for x in post_properties_str.split(',')]
            except:
                properties_vector = []
            
            data.append({
                'source': source_subreddit,
                'target': target_subreddit,
                'post_label': post_label,
                'properties': properties_vector
            })
    
    print(f"Loaded {len(data)} hyperlinks")
    
    # Create entity mapping (subreddits)
    all_subreddits = set()
    for item in data:
        all_subreddits.add(item['source'])
        all_subreddits.add(item['target'])
    
    entity2id = {subreddit: idx for idx, subreddit in enumerate(sorted(all_subreddits))}
    print(f"Found {len(entity2id)} unique subreddits")
    
    # Create relation mapping
    if use_sentiment_as_relation:
        # Use POST_LABEL as relation type: negative (-1) or positive (+1)
        relation2id = {
            'negative': 0,  # POST_LABEL = -1
            'positive': 1   # POST_LABEL = +1
        }
    else:
        # Single relation type for all links
        relation2id = {'hyperlink': 0}
    
    print(f"Using {len(relation2id)} relation types")
    
    # Convert to triplets format
    triplets = []
    confidences = []
    
    for item in data:
        source_id = entity2id[item['source']]
        target_id = entity2id[item['target']]
        
        if use_sentiment_as_relation:
            if item['post_label'] == -1:
                relation_id = relation2id['negative']
            else:  # +1
                relation_id = relation2id['positive']
        else:
            relation_id = relation2id['hyperlink']
        
        triplets.append([source_id, relation_id, target_id])
        
        # Derive confidence score
        if use_properties_for_confidence and len(item['properties']) > 0:
            confidence = normalize_confidence(item['properties'])
        else:
            # Use uniform confidence or derive from POST_LABEL
            # Positive links might be more reliable
            confidence = 0.7 if item['post_label'] == 1 else 0.5
        
        confidences.append(confidence)
    
    triplets = np.array(triplets, dtype=np.int32)
    confidences = np.array(confidences, dtype=np.float32)
    
    print(f"Created {len(triplets)} triplets")
    print(f"Confidence stats: min={confidences.min():.3f}, max={confidences.max():.3f}, mean={confidences.mean():.3f}")
    
    return triplets, confidences, entity2id, relation2id

def save_data_format(triplets, confidences, entity2id, relation2id, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Save data in CN15k format (TSV files with numeric IDs).
    
    Format:
    - entity_id.csv: entity_name,entity_id
    - relation_id.csv: relation_name,relation_id
    - train.tsv: head_id\trelation_id\ttail_id\tconfidence
    - val.tsv: same format
    - test.tsv: same format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save entity mapping
    with open(os.path.join(output_dir, 'entity_id.csv'), 'w') as f:
        f.write('entity,id\n')
        for entity, eid in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f'{entity},{eid}\n')
    
    # Save relation mapping
    with open(os.path.join(output_dir, 'relation_id.csv'), 'w') as f:
        f.write('relation,id\n')
        for relation, rid in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f'{relation},{rid}\n')
    
    # Split data
    n_total = len(triplets)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Shuffle before splitting
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_triplets = triplets[train_indices]
    train_conf = confidences[train_indices]
    val_triplets = triplets[val_indices]
    val_conf = confidences[val_indices]
    test_triplets = triplets[test_indices]
    test_conf = confidences[test_indices]
    
    # Save splits
    for split_name, split_triplets, split_conf in [
        ('train', train_triplets, train_conf),
        ('val', val_triplets, val_conf),
        ('test', test_triplets, test_conf)
    ]:
        output_file = os.path.join(output_dir, f'{split_name}.tsv')
        with open(output_file, 'w') as f:
            for triplet, conf in zip(split_triplets, split_conf):
                f.write(f'{triplet[0]}\t{triplet[1]}\t{triplet[2]}\t{conf:.6f}\n')
        print(f"Saved {split_name}: {len(split_triplets)} triplets")
    
    print(f"\nData saved to {output_dir}")
    print(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}, Test: {len(test_triplets)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Reddit hyperlink network to RGCN format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to Reddit TSV file (e.g., soc-redditHyperlinks-body.tsv)')
    parser.add_argument('--output', type=str, default='../../data/reddit',
                        help='Output directory for converted data')
    parser.add_argument('--use-sentiment-relation', action='store_true', default=True,
                        help='Use POST_LABEL as relation type (negative/positive)')
    parser.add_argument('--use-properties-confidence', action='store_true', default=True,
                        help='Derive confidence from POST_PROPERTIES vector')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Load and convert data
    triplets, confidences, entity2id, relation2id = load_reddit_data(
        args.input,
        use_sentiment_as_relation=args.use_sentiment_relation,
        use_properties_for_confidence=args.use_properties_confidence
    )
    
    # Save in expected format
    save_data_format(
        triplets, confidences, entity2id, relation2id,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

if __name__ == '__main__':
    main()
