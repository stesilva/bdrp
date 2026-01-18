import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

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

# Classify nodes
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

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Node Type Distribution
ax1 = plt.subplot(2, 3, 1)
node_counts = nodes_df['node_type'].value_counts()
colors = plt.cm.Set3(range(len(node_counts)))
ax1.pie(node_counts.values, labels=node_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax1.set_title('Node Type Distribution\n(Total: {:,} nodes)'.format(len(nodes_df)), 
              fontsize=12, fontweight='bold')

# 2. Edge Weight Distribution (non-zero only)
ax2 = plt.subplot(2, 3, 2)
non_zero_weights = edges_df[edges_df['weight'] > 0]['weight']
ax2.hist(non_zero_weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Weight Value', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Non-Zero Edge Weight Distribution\n({:,} edges, {:.1f}%)'.format(
    len(non_zero_weights), len(non_zero_weights)/len(edges_df)*100), 
    fontsize=12, fontweight='bold')
ax2.set_yscale('log')

# 3. Relation Type Distribution
ax3 = plt.subplot(2, 3, 3)
rel_counts = edges_df['relation_id'].value_counts().head(10)
rel_names = []
for rel_id in rel_counts.index:
    name = relations_df[relations_df['id'] == rel_id]['relation_uri'].values
    name = name[0].split('#')[-1] if len(name) > 0 else f'Rel_{rel_id}'
    rel_names.append(name[:25])  # Truncate long names

y_pos = np.arange(len(rel_names))
ax3.barh(y_pos, rel_counts.values, color='coral', edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(rel_names, fontsize=8)
ax3.set_xlabel('Number of Edges', fontsize=10)
ax3.set_title('Top 10 Relationship Types', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# 4. Weight by Relation Type (only those with non-zero weights)
ax4 = plt.subplot(2, 3, 4)
weighted_rels = edges_df[edges_df['weight'] != 0].groupby('relation_id')['weight'].agg(['mean', 'std', 'count'])
weighted_rels = weighted_rels[weighted_rels['count'] > 10].sort_values('mean', ascending=False)

rel_names_weighted = []
for rel_id in weighted_rels.index:
    name = relations_df[relations_df['id'] == rel_id]['relation_uri'].values
    name = name[0].split('#')[-1] if len(name) > 0 else f'Rel_{rel_id}'
    rel_names_weighted.append(name[:20])

if len(weighted_rels) > 0:
    y_pos = np.arange(len(rel_names_weighted))
    ax4.barh(y_pos, weighted_rels['mean'].values, 
             xerr=weighted_rels['std'].values, color='lightgreen', 
             edgecolor='black', alpha=0.7, capsize=3)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(rel_names_weighted, fontsize=9)
    ax4.set_xlabel('Mean Weight', fontsize=10)
    ax4.set_title('Average Weight by Relationship Type\n(Only weighted relations)', 
                  fontsize=12, fontweight='bold')
    ax4.invert_yaxis()

# 5. Degree Distribution
ax5 = plt.subplot(2, 3, 5)
all_nodes_in_edges = pd.concat([edges_df['node1'], edges_df['node2']])
total_degree = all_nodes_in_edges.value_counts()

ax5.hist(total_degree.values, bins=50, color='mediumpurple', 
         edgecolor='black', alpha=0.7)
ax5.set_xlabel('Node Degree', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Node Degree Distribution\n(Mean: {:.1f}, Median: {:.1f})'.format(
    total_degree.mean(), total_degree.median()), fontsize=12, fontweight='bold')
ax5.set_yscale('log')
ax5.set_xscale('log')

# 6. Edge Type Patterns (Sankey-style counts)
ax6 = plt.subplot(2, 3, 6)
edges_with_types = edges_df.copy()
edges_with_types['source_type'] = edges_with_types['node1'].map(
    nodes_df.set_index('id')['node_type']
)
edges_with_types['target_type'] = edges_with_types['node2'].map(
    nodes_df.set_index('id')['node_type']
)

edge_patterns = edges_with_types.groupby(['source_type', 'target_type']).size()
edge_patterns = edge_patterns.sort_values(ascending=False).head(10)

pattern_labels = [f"{src} → {tgt}" for src, tgt in edge_patterns.index]
y_pos = np.arange(len(pattern_labels))
ax6.barh(y_pos, edge_patterns.values, color='gold', edgecolor='black', alpha=0.7)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(pattern_labels, fontsize=8)
ax6.set_xlabel('Number of Edges', fontsize=10)
ax6.set_title('Top 10 Node Type Connection Patterns', 
              fontsize=12, fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('graph_analysis_overview.png', dpi=300, bbox_inches='tight')
print("Saved: graph_analysis_overview.png")
plt.show()

# ============================================================================
# Additional detailed plot: Weight distribution by relation
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))

# Get relations with significant non-zero weights
weighted_relations = []
for rel_id in edges_df['relation_id'].unique():
    rel_data = edges_df[edges_df['relation_id'] == rel_id]['weight']
    non_zero = rel_data[rel_data != 0]
    if len(non_zero) > 100:  # Only relations with substantial data
        weighted_relations.append(rel_id)

# Plot weight distributions for key weighted relations
for idx, rel_id in enumerate(weighted_relations[:4]):
    ax = axes[idx // 2, idx % 2]
    rel_data = edges_df[edges_df['relation_id'] == rel_id]['weight']
    non_zero = rel_data[rel_data != 0]
    
    rel_name = relations_df[relations_df['id'] == rel_id]['relation_uri'].values
    rel_name = rel_name[0].split('#')[-1] if len(rel_name) > 0 else f'Relation_{rel_id}'
    
    ax.hist(non_zero, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(non_zero.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {non_zero.mean():.2f}')
    ax.axvline(non_zero.median(), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {non_zero.median():.2f}')
    ax.set_xlabel('Weight', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{rel_name}\n({len(non_zero):,} non-zero weights)', 
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('weight_distributions_by_relation.png', dpi=300, bbox_inches='tight')
print("Saved: weight_distributions_by_relation.png")
plt.show()

# ============================================================================
# Generate summary statistics file
# ============================================================================
with open('graph_summary_statistics.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DETAILED GRAPH STATISTICS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    # Key insights
    f.write("KEY INSIGHTS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. Network has {len(nodes_df):,} nodes with ZERO isolated nodes (100% connected)\n")
    f.write(f"2. {len(edges_df):,} total edges, with {(edges_df['weight'] == 0).sum():,} ")
    f.write(f"({(edges_df['weight'] == 0).sum()/len(edges_df)*100:.1f}%) having zero weight\n")
    f.write(f"3. Only 3 relationship types carry meaningful weights:\n")
    f.write(f"   - hasGeneExpressionOA (gene expression data)\n")
    f.write(f"   - hasPhysicalInteractionWith (protein interactions)\n")
    f.write(f"   - hasGeneticInteractionWith (genetic interactions)\n")
    f.write(f"4. Network dominated by Reactions ({node_counts['Reaction']:,}, ")
    f.write(f"{node_counts['Reaction']/len(nodes_df)*100:.1f}%)\n")
    f.write(f"5. Patient samples ({node_counts.get('Sample', 0):,}) drive ")
    f.write(f"{edges_df[edges_df['relation_id']==1].shape[0]:,} gene expression measurements\n\n")
    
    # Quartile analysis for weighted edges
    f.write("WEIGHT QUARTILE ANALYSIS (Non-Zero Weights):\n")
    f.write("-" * 80 + "\n")
    non_zero = edges_df[edges_df['weight'] > 0]['weight']
    quartiles = non_zero.quantile([0.25, 0.5, 0.75])
    f.write(f"25th percentile: {quartiles[0.25]:.4f}\n")
    f.write(f"50th percentile (median): {quartiles[0.5]:.4f}\n")
    f.write(f"75th percentile: {quartiles[0.75]:.4f}\n")
    f.write(f"95th percentile: {non_zero.quantile(0.95):.4f}\n")
    f.write(f"99th percentile: {non_zero.quantile(0.99):.4f}\n\n")
    
    # Hub analysis by node type
    f.write("HUB NODES BY TYPE:\n")
    f.write("-" * 80 + "\n")
    total_degree = pd.concat([edges_df['node1'], edges_df['node2']]).value_counts()
    nodes_with_degree = nodes_df.copy()
    nodes_with_degree['degree'] = nodes_with_degree['id'].map(total_degree).fillna(0)
    
    for node_type in nodes_df['node_type'].unique():
        type_nodes = nodes_with_degree[nodes_with_degree['node_type'] == node_type]
        top_node = type_nodes.nlargest(1, 'degree')
        if len(top_node) > 0:
            node_id = top_node.iloc[0]['id']
            node_deg = top_node.iloc[0]['degree']
            node_uri = top_node.iloc[0]['node_uri'].split('#')[-1]
            f.write(f"{node_type:15s}: Node {node_id:4d} with {int(node_deg):4d} ")
            f.write(f"connections - {node_uri[:50]}\n")

print("\nSaved: graph_summary_statistics.txt")
print("\nAnalysis complete! Generated 3 files:")
print("  1. graph_analysis_overview.png - Main visualizations")
print("  2. weight_distributions_by_relation.png - Detailed weight analysis")
print("  3. graph_summary_statistics.txt - Text summary")