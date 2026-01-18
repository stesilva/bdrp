import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 80)
print("KNOWLEDGE GRAPH TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)
print(f"\nSplit ratios: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")

# Load data
print("\nLoading data...")
nodes_df = pd.read_csv('edges.filtered.tsv.entities.tsv', sep='\t', header=None, 
                       names=['id', 'node_uri'])
edges_df = pd.read_csv('edges.filtered.tsv', sep='\t', header=None)
edges_df.columns = ['node1', 'relation_id', 'node2', 'weight'] + \
                   [f'extra_{i}' for i in range(len(edges_df.columns) - 4)]

# Keep only the essential columns
edges_df = edges_df[['node1', 'relation_id', 'node2', 'weight']]

print(f"Loaded {len(nodes_df):,} nodes and {len(edges_df):,} edges")

# Check for duplicates
print("\nChecking for duplicate edges...")
edges_df['edge_id'] = edges_df.apply(lambda x: f"{x['node1']}-{x['relation_id']}-{x['node2']}", axis=1)
duplicates = edges_df[edges_df.duplicated(subset=['node1', 'relation_id', 'node2'], keep=False)]
if len(duplicates) > 0:
    print(f"  ⚠ WARNING: Found {len(duplicates)} duplicate triplets!")
    print(f"  Removing duplicates...")
    edges_df = edges_df.drop_duplicates(subset=['node1', 'relation_id', 'node2'], keep='first')
    print(f"  Kept {len(edges_df):,} unique edges")
else:
    print(f"  ✓ No duplicate triplets found")

# Check for multi-relational edges
node_pairs = edges_df.groupby(['node1', 'node2']).size()
multi_rel_pairs = node_pairs[node_pairs > 1]
print(f"\nMulti-relational edge analysis:")
print(f"  Unique node pairs: {len(node_pairs):,}")
print(f"  Node pairs with multiple relations: {len(multi_rel_pairs):,}")
if len(multi_rel_pairs) > 0:
    print(f"  Max relations per node pair: {multi_rel_pairs.max()}")
    print(f"  Average relations per multi-rel pair: {multi_rel_pairs.mean():.2f}")

# Classify nodes for analysis
def classify_node(uri):
    if pd.isna(uri):
        return 'Unknown'
    uri = str(uri)
    if 'Protein_' in uri:
        return 'Protein'
    elif 'Sample_' in uri:
        return 'Sample'
    elif 'GO_' in uri or '/GO_' in uri:
        return 'GO_Term'
    elif 'Pathway_' in uri:
        return 'Pathway'
    elif 'Reaction_' in uri:
        return 'Reaction'
    else:
        return 'Other'

nodes_df['node_type'] = nodes_df['node_uri'].apply(classify_node)

# ============================================================================
# CORRECTED STRATEGY: Stratified + Entity-Aware + PROPER Edge Splitting
# ============================================================================
print("\n" + "=" * 80)
print("CREATING TRAIN/VAL/TEST SPLIT (NO EDGE LEAKAGE)")
print("=" * 80)

def proper_split(edges_df, train_ratio, val_ratio, test_ratio, max_attempts=10):
    """
    Proper split that guarantees:
    1. No edge overlap between splits
    2. Stratification by relation type
    3. No new entities in val/test
    """
    best_split = None
    best_val_test_size = 0
    
    for attempt in range(max_attempts):
        train_edges = []
        val_edges = []
        test_edges = []
        
        # Split each relation type independently
        for rel_id in edges_df['relation_id'].unique():
            rel_edges = edges_df[edges_df['relation_id'] == rel_id].copy()
            n = len(rel_edges)
            
            # Shuffle with different seed for each attempt
            rel_edges = rel_edges.sample(frac=1, random_state=RANDOM_SEED + attempt).reset_index(drop=True)
            
            # Calculate split indices
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)
            
            # CRITICAL: Use iloc to ensure non-overlapping slices
            train = rel_edges.iloc[:train_idx].copy()
            val_candidate = rel_edges.iloc[train_idx:val_idx].copy()
            test_candidate = rel_edges.iloc[val_idx:].copy()
            
            train_edges.append(train)
            val_edges.append(val_candidate)
            test_edges.append(test_candidate)
        
        # Concatenate all relation types
        train = pd.concat(train_edges, ignore_index=True)
        val_candidate = pd.concat(val_edges, ignore_index=True)
        test_candidate = pd.concat(test_edges, ignore_index=True)
        
        # Get training nodes
        train_nodes = set(train['node1']).union(set(train['node2']))
        
        # Filter val/test to only include edges with both nodes in training
        val_filtered = val_candidate[
            val_candidate['node1'].isin(train_nodes) & 
            val_candidate['node2'].isin(train_nodes)
        ].copy()
        
        test_filtered = test_candidate[
            test_candidate['node1'].isin(train_nodes) & 
            test_candidate['node2'].isin(train_nodes)
        ].copy()
        
        # Add edges with unseen nodes back to training
        val_leftover = val_candidate[~val_candidate.index.isin(val_filtered.index)]
        test_leftover = test_candidate[~test_candidate.index.isin(test_filtered.index)]
        
        train = pd.concat([train, val_leftover, test_leftover], ignore_index=True)
        
        # Track best attempt
        val_test_size = len(val_filtered) + len(test_filtered)
        if val_test_size > best_val_test_size:
            best_val_test_size = val_test_size
            best_split = (train, val_filtered, test_filtered)
            
    return best_split

print("Performing split (trying multiple random seeds for best coverage)...")
train, val, test = proper_split(edges_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

print(f"\nSplit sizes:")
print(f"  Train: {len(train):,} ({len(train)/len(edges_df):.1%})")
print(f"  Val:   {len(val):,} ({len(val)/len(edges_df):.1%})")
print(f"  Test:  {len(test):,} ({len(test)/len(edges_df):.1%})")

# ============================================================================
# CRITICAL: Verify NO edge overlap
# ============================================================================
print("\n" + "=" * 80)
print("EDGE OVERLAP VERIFICATION (CRITICAL CHECK)")
print("=" * 80)

# Create edge sets using (node1, relation, node2) tuples
train_edge_set = set(zip(train['node1'], train['relation_id'], train['node2']))
val_edge_set = set(zip(val['node1'], val['relation_id'], val['node2']))
test_edge_set = set(zip(test['node1'], test['relation_id'], test['node2']))

val_train_overlap = len(val_edge_set & train_edge_set)
test_train_overlap = len(test_edge_set & train_edge_set)
test_val_overlap = len(test_edge_set & val_edge_set)

print(f"\nEdge overlap check:")
print(f"  Val ∩ Train:  {val_train_overlap} edges ({'✓ PASS' if val_train_overlap == 0 else '✗ CRITICAL FAIL'})")
print(f"  Test ∩ Train: {test_train_overlap} edges ({'✓ PASS' if test_train_overlap == 0 else '✗ CRITICAL FAIL'})")
print(f"  Test ∩ Val:   {test_val_overlap} edges ({'✓ PASS' if test_val_overlap == 0 else '✗ CRITICAL FAIL'})")

if val_train_overlap > 0 or test_train_overlap > 0 or test_val_overlap > 0:
    print("\n  ⚠⚠⚠ CRITICAL ERROR: Edge leakage detected!")
    print("  This should not happen. Investigating...")
    
    # Debug: Show examples of overlapping edges
    if val_train_overlap > 0:
        overlap_edges = val_edge_set & train_edge_set
        print(f"\n  Example Val-Train overlaps: {list(overlap_edges)[:3]}")
    if test_train_overlap > 0:
        overlap_edges = test_edge_set & train_edge_set
        print(f"  Example Test-Train overlaps: {list(overlap_edges)[:3]}")
else:
    print("\n  ✓✓✓ All edge overlap checks PASSED!")

# ============================================================================
# Entity coverage verification
# ============================================================================
print("\n" + "=" * 80)
print("ENTITY COVERAGE VERIFICATION")
print("=" * 80)

train_nodes = set(train['node1']).union(set(train['node2']))
val_nodes = set(val['node1']).union(set(val['node2']))
test_nodes = set(test['node1']).union(set(test['node2']))

print(f"\nEntity coverage:")
print(f"  Train nodes: {len(train_nodes):,} ({len(train_nodes)/len(nodes_df):.1%})")
print(f"  Val nodes:   {len(val_nodes):,}")
print(f"  Test nodes:  {len(test_nodes):,}")

val_unseen = len(val_nodes - train_nodes)
test_unseen = len(test_nodes - train_nodes)

print(f"\nNode leakage check:")
print(f"  Val nodes not in train:  {val_unseen} ({'✓ PASS' if val_unseen == 0 else '✗ FAIL'})")
print(f"  Test nodes not in train: {test_unseen} ({'✓ PASS' if test_unseen == 0 else '✗ FAIL'})")

if val_unseen > 0 or test_unseen > 0:
    print("\n  ⚠ WARNING: Some entities in val/test are not in training!")
    print("  This is a transductive setting violation.")

# ============================================================================
# Relation type distribution
# ============================================================================
print("\n" + "=" * 80)
print("RELATION TYPE DISTRIBUTION")
print("=" * 80)

print(f"\nDistribution across splits:")
for rel_id in sorted(edges_df['relation_id'].unique()):
    train_count = (train['relation_id'] == rel_id).sum()
    val_count = (val['relation_id'] == rel_id).sum()
    test_count = (test['relation_id'] == rel_id).sum()
    total = train_count + val_count + test_count
    
    if total > 0:
        print(f"  Rel {rel_id:2d}: Train={train_count:6,} ({train_count/total*100:5.1f}%), "
              f"Val={val_count:5,} ({val_count/total*100:5.1f}%), "
              f"Test={test_count:5,} ({test_count/total*100:5.1f}%)")

# ============================================================================
# Node type distribution
# ============================================================================
print("\n" + "=" * 80)
print("NODE TYPE DISTRIBUTION")
print("=" * 80)

node_id_to_type = dict(zip(nodes_df['id'], nodes_df['node_type']))

def get_type_dist(node_set):
    types = [node_id_to_type.get(n, 'Unknown') for n in node_set]
    return Counter(types)

train_type_dist = get_type_dist(train_nodes)
val_type_dist = get_type_dist(val_nodes)
test_type_dist = get_type_dist(test_nodes)

print("\nNode types across splits:")
all_types = set(train_type_dist.keys()) | set(val_type_dist.keys()) | set(test_type_dist.keys())
for node_type in sorted(all_types):
    train_cnt = train_type_dist.get(node_type, 0)
    val_cnt = val_type_dist.get(node_type, 0)
    test_cnt = test_type_dist.get(node_type, 0)
    print(f"  {node_type:15s}: Train={train_cnt:5,}, Val={val_cnt:5,}, Test={test_cnt:5,}")

# ============================================================================
# Save splits
# ============================================================================
print("\n" + "=" * 80)
print("SAVING SPLITS")
print("=" * 80)

# Remove the edge_id column before saving
train_clean = train[['node1', 'relation_id', 'node2', 'weight']].copy()
val_clean = val[['node1', 'relation_id', 'node2', 'weight']].copy()
test_clean = test[['node1', 'relation_id', 'node2', 'weight']].copy()

train_clean.to_csv('edges_train.tsv', sep='\t', header=False, index=False)
val_clean.to_csv('edges_val.tsv', sep='\t', header=False, index=False)
test_clean.to_csv('edges_test.tsv', sep='\t', header=False, index=False)

print(f"\nSaved files:")
print(f"  edges_train.tsv: {len(train_clean):,} edges")
print(f"  edges_val.tsv:   {len(val_clean):,} edges")
print(f"  edges_test.tsv:  {len(test_clean):,} edges")

# ============================================================================
# Final verification by re-loading files
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERIFICATION (Re-loading saved files)")
print("=" * 80)

train_loaded = pd.read_csv('edges_train.tsv', sep='\t', header=None, 
                           names=['node1', 'relation_id', 'node2', 'weight'])
val_loaded = pd.read_csv('edges_val.tsv', sep='\t', header=None,
                         names=['node1', 'relation_id', 'node2', 'weight'])
test_loaded = pd.read_csv('edges_test.tsv', sep='\t', header=None,
                          names=['node1', 'relation_id', 'node2', 'weight'])

train_set_final = set(zip(train_loaded['node1'], train_loaded['relation_id'], train_loaded['node2']))
val_set_final = set(zip(val_loaded['node1'], val_loaded['relation_id'], val_loaded['node2']))
test_set_final = set(zip(test_loaded['node1'], test_loaded['relation_id'], test_loaded['node2']))

final_val_train = len(val_set_final & train_set_final)
final_test_train = len(test_set_final & train_set_final)
final_test_val = len(test_set_final & val_set_final)

print(f"\nFinal edge overlap check (from saved files):")
print(f"  Val ∩ Train:  {final_val_train} ({'✓✓✓ PASS' if final_val_train == 0 else '✗✗✗ FAIL'})")
print(f"  Test ∩ Train: {final_test_train} ({'✓✓✓ PASS' if final_test_train == 0 else '✗✗✗ FAIL'})")
print(f"  Test ∩ Val:   {final_test_val} ({'✓✓✓ PASS' if final_test_val == 0 else '✗✗✗ FAIL'})")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SPLIT COMPLETE!")
print("=" * 80)

if final_val_train == 0 and final_test_train == 0 and final_test_val == 0:
    print("\n✓✓✓ SUCCESS! All quality checks passed:")
    print("  ✓ No edge leakage between splits")
    print("  ✓ No unseen entities in validation/test")
    print("  ✓ Stratified by relation type")
    print("  ✓ Files saved and verified")
    print("\nYour data is ready for link prediction!")
else:
    print("\n✗✗✗ FAILURE: Edge leakage detected in saved files!")
    print("  Please review the script and check for issues.")