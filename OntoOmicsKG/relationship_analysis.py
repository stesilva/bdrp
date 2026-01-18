import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re

print("=" * 80)
print("GRAPH ANALYSIS REPORT")
print("=" * 80)

# Load the relations mapping
relations_df = pd.read_csv('edges.filtered.tsv.relations.tsv', sep='\t', 
                           header=None, names=['id', 'relation_uri'])

# Load the nodes data
nodes_df = pd.read_csv('edges.filtered.tsv.entities.tsv', sep='\t', header=None, 
                       names=['id', 'node_uri'])

# Load the edges data
edges_df = pd.read_csv('edges.filtered.tsv', sep='\t', header=None)
edges_df.columns = ['node1', 'relation_id', 'node2', 'weight'] + \
                   [f'extra_{i}' for i in range(len(edges_df.columns) - 4)]

# ============================================================================
# 1. BASIC COUNTS
# ============================================================================
print("\n1. BASIC GRAPH STATISTICS")
print("-" * 80)

num_nodes = len(nodes_df)
num_edges = len(edges_df)
num_unique_edges = len(edges_df.drop_duplicates(subset=['node1', 'relation_id', 'node2']))
num_relations = len(relations_df)

print(f"Total Nodes: {num_nodes:,}")
print(f"Total Edges: {num_edges:,}")
print(f"Unique Edges: {num_unique_edges:,}")
print(f"Duplicate Edges: {num_edges - num_unique_edges:,}")
print(f"Relation Types: {num_relations}")

# ============================================================================
# 2. NODE TYPE CLASSIFICATION
# ============================================================================
print("\n2. NODE TYPE DISTRIBUTION")
print("-" * 80)

def classify_node(uri):
    """Classify node based on URI pattern"""
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
    elif 'Disease_' in uri:
        return 'Disease'
    elif 'Drug_' in uri or 'Compound_' in uri:
        return 'Drug/Compound'
    elif 'Gene_' in uri:
        return 'Gene'
    elif 'Reaction_' in uri:
        return 'Reaction'
    else:
        return 'Other'

nodes_df['node_type'] = nodes_df['node_uri'].apply(classify_node)
node_type_counts = nodes_df['node_type'].value_counts()

print("\nNode Types:")
for node_type, count in node_type_counts.items():
    pct = (count / num_nodes) * 100
    print(f"  {node_type:<20} {count:>8,} ({pct:>5.2f}%)")

# ============================================================================
# 3. EDGE WEIGHT STATISTICS
# ============================================================================
print("\n3. EDGE WEIGHT STATISTICS")
print("-" * 80)

weights = edges_df['weight']
non_zero_weights = weights[weights > 0]

print("\nOverall Weight Statistics:")
print(f"  Mean:     {weights.mean():.6f}")
print(f"  Median:   {weights.median():.6f}")
print(f"  Min:      {weights.min():.6f}")
print(f"  Max:      {weights.max():.6f}")
print(f"  Std Dev:  {weights.std():.6f}")

print(f"\nZero Weights: {(weights == 0).sum():,} ({(weights == 0).sum() / len(weights) * 100:.2f}%)")
print(f"Non-Zero Weights: {len(non_zero_weights):,} ({len(non_zero_weights) / len(weights) * 100:.2f}%)")

if len(non_zero_weights) > 0:
    print(f"\nNon-Zero Weight Statistics:")
    print(f"  Mean:     {non_zero_weights.mean():.6f}")
    print(f"  Median:   {non_zero_weights.median():.6f}")
    print(f"  Min:      {non_zero_weights.min():.6f}")
    print(f"  Max:      {non_zero_weights.max():.6f}")

# ============================================================================
# 4. RELATION TYPE ANALYSIS
# ============================================================================
print("\n4. RELATION TYPE STATISTICS")
print("-" * 80)

relation_counts = edges_df['relation_id'].value_counts()
edges_with_relations = edges_df.merge(relations_df, left_on='relation_id', 
                                       right_on='id', how='left')

print("\nEdges per Relation Type:")
for rel_id in relation_counts.index:
    count = relation_counts[rel_id]
    rel_name = relations_df[relations_df['id'] == rel_id]['relation_uri'].values
    rel_name = rel_name[0].split('#')[-1] if len(rel_name) > 0 else 'Unknown'
    pct = (count / num_edges) * 100
    print(f"  {rel_id:>2} | {rel_name:<40} {count:>8,} ({pct:>5.2f}%)")

# ============================================================================
# 5. WEIGHT STATISTICS BY RELATION TYPE
# ============================================================================
print("\n5. WEIGHT STATISTICS BY RELATION TYPE")
print("-" * 80)

for rel_id in sorted(edges_df['relation_id'].unique()):
    rel_edges = edges_df[edges_df['relation_id'] == rel_id]
    rel_weights = rel_edges['weight']
    rel_name = relations_df[relations_df['id'] == rel_id]['relation_uri'].values
    rel_name = rel_name[0].split('#')[-1] if len(rel_name) > 0 else 'Unknown'
    
    print(f"\n{rel_id} - {rel_name}:")
    print(f"  Count:  {len(rel_edges):,}")
    print(f"  Mean:   {rel_weights.mean():.6f}")
    print(f"  Median: {rel_weights.median():.6f}")
    print(f"  Min:    {rel_weights.min():.6f}")
    print(f"  Max:    {rel_weights.max():.6f}")

# ============================================================================
# 6. NODE DEGREE ANALYSIS
# ============================================================================
print("\n6. NODE DEGREE ANALYSIS")
print("-" * 80)

# Out-degree (edges from node)
out_degree = edges_df['node1'].value_counts()
# In-degree (edges to node)
in_degree = edges_df['node2'].value_counts()
# Total degree
all_nodes_in_edges = pd.concat([edges_df['node1'], edges_df['node2']])
total_degree = all_nodes_in_edges.value_counts()

print("\nOut-Degree Statistics:")
print(f"  Mean:   {out_degree.mean():.2f}")
print(f"  Median: {out_degree.median():.2f}")
print(f"  Min:    {out_degree.min()}")
print(f"  Max:    {out_degree.max()}")

print("\nIn-Degree Statistics:")
print(f"  Mean:   {in_degree.mean():.2f}")
print(f"  Median: {in_degree.median():.2f}")
print(f"  Min:    {in_degree.min()}")
print(f"  Max:    {in_degree.max()}")

print("\nTotal Degree Statistics:")
print(f"  Mean:   {total_degree.mean():.2f}")
print(f"  Median: {total_degree.median():.2f}")
print(f"  Min:    {total_degree.min()}")
print(f"  Max:    {total_degree.max()}")

# Top connected nodes
print("\nTop 10 Most Connected Nodes (by total degree):")
for i, (node_id, degree) in enumerate(total_degree.head(10).items(), 1):
    node_uri = nodes_df[nodes_df['id'] == node_id]['node_uri'].values
    node_uri = node_uri[0].split('#')[-1] if len(node_uri) > 0 else 'Unknown'
    node_type = nodes_df[nodes_df['id'] == node_id]['node_type'].values
    node_type = node_type[0] if len(node_type) > 0 else 'Unknown'
    print(f"  {i:>2}. Node {node_id} ({node_type}): {degree} connections - {node_uri[:60]}")

# ============================================================================
# 7. NETWORK CONNECTIVITY
# ============================================================================
print("\n7. NETWORK CONNECTIVITY")
print("-" * 80)

nodes_in_edges = set(edges_df['node1']).union(set(edges_df['node2']))
isolated_nodes = num_nodes - len(nodes_in_edges)

print(f"Nodes with connections: {len(nodes_in_edges):,}")
print(f"Isolated nodes: {isolated_nodes:,} ({isolated_nodes / num_nodes * 100:.2f}%)")
print(f"Average degree: {(num_edges * 2) / len(nodes_in_edges):.2f}")

# ============================================================================
# 8. EDGE TYPE PATTERNS
# ============================================================================
print("\n8. EDGE TYPE PATTERNS (Node Type to Node Type)")
print("-" * 80)

edges_with_types = edges_df.copy()
edges_with_types['source_type'] = edges_with_types['node1'].map(
    nodes_df.set_index('id')['node_type']
)
edges_with_types['target_type'] = edges_with_types['node2'].map(
    nodes_df.set_index('id')['node_type']
)

edge_patterns = edges_with_types.groupby(['source_type', 'target_type']).size()
edge_patterns = edge_patterns.sort_values(ascending=False)

print("\nMost Common Node Type Connections:")
for (src, tgt), count in edge_patterns.head(15).items():
    pct = (count / num_edges) * 100
    print(f"  {src:<20} -> {tgt:<20} {count:>8,} ({pct:>5.2f}%)")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)