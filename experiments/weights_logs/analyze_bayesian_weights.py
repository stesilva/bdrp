"""
Comprehensive analysis script for Bayesian edge weights
Analyzes how the learnable/Bayesian transformation affects edge weights across layers and training
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats

sns.set_style("whitegrid")

class BayesianWeightAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.logs = self._load_all_logs()
        
    def _load_all_logs(self):
        """Load all JSON log files"""
        log_files = sorted(self.log_dir.glob("weights_epoch_*.json"))
        logs = []
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                data = json.load(f)
                logs.append({
                    'file': log_file.name,
                    'epoch': self._extract_epoch(log_file.name),
                    'data': data
                })
        
        return logs
    
    def _extract_epoch(self, filename):
        """Extract epoch number from filename"""
        import re
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    def analyze_transformation_pattern(self, layer='conv1'):
        """Analyze how transformation changes across training"""
        
        epochs = []
        mean_diffs = []
        std_diffs = []
        compression_ratios = []  # How much weights are compressed
        
        for log in self.logs:
            layer_data = [l for l in log['data'] if l['layer'] == layer]
            if not layer_data:
                continue
                
            # Use last step in epoch
            data = layer_data[-1]
            
            orig_mean = data['original_weights']['mean']
            trans_mean = data['transformed_weights']['mean']
            orig_std = data['original_weights']['std']
            trans_std = data['transformed_weights']['std']
            
            epochs.append(log['epoch'])
            mean_diffs.append(trans_mean - orig_mean)
            std_diffs.append(trans_std - orig_std)
            
            # Compression ratio: how much the network squeezes/expands weights
            if orig_std > 0:
                compression_ratios.append(trans_std / orig_std)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{layer}: Transformation Analysis Across Training', fontsize=16)
        
        # Mean shift
        axes[0, 0].plot(epochs, mean_diffs, 'o-', linewidth=2, markersize=4)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean Shift (Trans - Orig)')
        axes[0, 0].set_title('How transformation shifts mean weight')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Std shift
        axes[0, 1].plot(epochs, std_diffs, 'o-', linewidth=2, markersize=4, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Std Shift (Trans - Orig)')
        axes[0, 1].set_title('How transformation changes variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Compression ratio
        axes[1, 0].plot(epochs, compression_ratios, 'o-', linewidth=2, markersize=4, color='green')
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No compression')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Compression Ratio (Trans_std / Orig_std)')
        axes[1, 0].set_title('Weight Variance Compression\n(<1: compression, >1: expansion)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of compression ratios
        axes[1, 1].hist(compression_ratios, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=1, color='r', linestyle='--', linewidth=2, label='No change')
        axes[1, 1].axvline(x=np.mean(compression_ratios), color='green', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(compression_ratios):.3f}')
        axes[1, 1].set_xlabel('Compression Ratio')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Compression Ratios')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'{layer}_transformation_pattern.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {layer}_transformation_pattern.png")
        
        return {
            'epochs': epochs,
            'mean_diffs': mean_diffs,
            'std_diffs': std_diffs,
            'compression_ratios': compression_ratios
        }
    
    def analyze_weight_distributions(self, epochs_to_plot=None, layer='conv1'):
        """Compare weight distributions at different epochs"""
        
        if epochs_to_plot is None:
            # Auto-select: first, 25%, 50%, 75%, last
            all_epochs = [log['epoch'] for log in self.logs]
            if len(all_epochs) >= 5:
                indices = [0, len(all_epochs)//4, len(all_epochs)//2, 
                          3*len(all_epochs)//4, len(all_epochs)-1]
                epochs_to_plot = [all_epochs[i] for i in indices]
            else:
                epochs_to_plot = all_epochs
        
        n_epochs = len(epochs_to_plot)
        fig, axes = plt.subplots(n_epochs, 3, figsize=(15, 4*n_epochs))
        if n_epochs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{layer}: Weight Distribution Evolution', fontsize=16)
        
        for idx, epoch in enumerate(epochs_to_plot):
            # Find log for this epoch
            log = next((l for l in self.logs if l['epoch'] == epoch), None)
            if log is None:
                continue
                
            layer_data = [l for l in log['data'] if l['layer'] == layer]
            if not layer_data:
                continue
                
            data = layer_data[-1]
            orig = np.array(data['original_weights']['values'])
            trans = np.array(data['transformed_weights']['values'])
            
            # 1. Histograms
            axes[idx, 0].hist(orig, bins=50, alpha=0.6, label='Original', density=True, color='blue')
            axes[idx, 0].hist(trans, bins=50, alpha=0.6, label='Transformed', density=True, color='red')
            axes[idx, 0].set_ylabel(f'Epoch {epoch}')
            axes[idx, 0].legend()
            if idx == 0:
                axes[idx, 0].set_title('Weight Distributions')
            
            # 2. Scatter: Original vs Transformed
            axes[idx, 1].scatter(orig, trans, alpha=0.3, s=1)
            axes[idx, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Identity')
            axes[idx, 1].set_xlabel('Original Weight')
            axes[idx, 1].set_ylabel('Transformed Weight')
            axes[idx, 1].legend()
            if idx == 0:
                axes[idx, 1].set_title('Transformation Function')
            
            # 3. Difference distribution
            diff = trans - orig
            axes[idx, 2].hist(diff, bins=50, edgecolor='black', alpha=0.7, color='purple')
            axes[idx, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[idx, 2].axvline(x=np.mean(diff), color='green', linestyle='--', 
                               linewidth=2, label=f'Mean: {np.mean(diff):.4f}')
            axes[idx, 2].set_xlabel('Difference (Trans - Orig)')
            axes[idx, 2].legend()
            if idx == 0:
                axes[idx, 2].set_title('Transformation Effect')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'{layer}_distribution_evolution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {layer}_distribution_evolution.png")
    
    def compare_layers(self, epoch=None):
        """Compare how conv1 vs conv2 transform weights"""
        
        # Use last epoch if not specified
        if epoch is None:
            epoch = self.logs[-1]['epoch']
        
        log = next((l for l in self.logs if l['epoch'] == epoch), None)
        if log is None:
            print(f"No data for epoch {epoch}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Layer Comparison at Epoch {epoch}', fontsize=16)
        
        for layer_idx, layer_name in enumerate(['conv1', 'conv2']):
            layer_data = [l for l in log['data'] if l['layer'] == layer_name]
            if not layer_data:
                continue
                
            data = layer_data[-1]
            orig = np.array(data['original_weights']['values'])
            trans = np.array(data['transformed_weights']['values'])
            
            # Distribution
            axes[layer_idx, 0].hist(orig, bins=50, alpha=0.6, label='Original', density=True)
            axes[layer_idx, 0].hist(trans, bins=50, alpha=0.6, label='Transformed', density=True)
            axes[layer_idx, 0].set_ylabel(layer_name, fontsize=12, fontweight='bold')
            axes[layer_idx, 0].legend()
            if layer_idx == 0:
                axes[layer_idx, 0].set_title('Distributions')
            
            # Scatter
            axes[layer_idx, 1].scatter(orig, trans, alpha=0.3, s=1)
            axes[layer_idx, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[layer_idx, 1].set_xlabel('Original')
            axes[layer_idx, 1].set_ylabel('Transformed')
            if layer_idx == 0:
                axes[layer_idx, 1].set_title('Transformation')
            
            # Statistics
            stats_text = f"Original:\n  Mean: {np.mean(orig):.4f}\n  Std: {np.std(orig):.4f}\n\n"
            stats_text += f"Transformed:\n  Mean: {np.mean(trans):.4f}\n  Std: {np.std(trans):.4f}\n\n"
            stats_text += f"Correlation: {np.corrcoef(orig, trans)[0,1]:.4f}"
            
            axes[layer_idx, 2].text(0.1, 0.5, stats_text, fontsize=11, 
                                   verticalalignment='center', family='monospace',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[layer_idx, 2].axis('off')
            if layer_idx == 0:
                axes[layer_idx, 2].set_title('Statistics')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'layer_comparison_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: layer_comparison_epoch_{epoch}.png")
    
    def analyze_bayesian_uncertainty(self, layer='conv1'):
        """Specific analysis for Bayesian mode (if additional_info contains uncertainty data)"""
        
        epochs = []
        mean_means = []
        mean_stds = []
        var_means = []
        var_stds = []
        var_scales = []
        
        for log in self.logs:
            layer_data = [l for l in log['data'] if l['layer'] == layer]
            if not layer_data:
                continue
                
            data = layer_data[-1]
            
            # Check if Bayesian-specific info is available
            if 'additional_info' in data and 'weight_mean_stats' in data['additional_info']:
                info = data['additional_info']
                epochs.append(log['epoch'])
                mean_means.append(info['weight_mean_stats']['mean'])
                mean_stds.append(info['weight_mean_stats']['std'])
                var_means.append(info['weight_var_stats']['mean'])
                var_stds.append(info['weight_var_stats']['std'])
                var_scales.append(info['var_scale'])
        
        if not epochs:
            print("No Bayesian uncertainty data found. This analysis requires edge_weight_mode='bayesian'")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{layer}: Bayesian Uncertainty Analysis', fontsize=16)
        
        # Mean evolution
        axes[0, 0].plot(epochs, mean_means, 'o-', label='Mean of μ', linewidth=2)
        axes[0, 0].fill_between(epochs, 
                                np.array(mean_means) - np.array(mean_stds),
                                np.array(mean_means) + np.array(mean_stds),
                                alpha=0.3, label='±1 std')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Weight Mean (μ)')
        axes[0, 0].set_title('Evolution of Weight Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variance evolution
        axes[0, 1].plot(epochs, var_means, 'o-', label='Mean of σ²', linewidth=2, color='orange')
        axes[0, 1].fill_between(epochs,
                                np.array(var_means) - np.array(var_stds),
                                np.array(var_means) + np.array(var_stds),
                                alpha=0.3, label='±1 std', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weight Variance (σ²)')
        axes[0, 1].set_title('Evolution of Weight Uncertainty')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Variance scale parameter
        axes[1, 0].plot(epochs, var_scales, 'o-', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Variance Scale Parameter')
        axes[1, 0].set_title('Learned Variance Scale')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty-to-mean ratio
        uncertainty_ratio = np.array(var_means) / (np.array(mean_means) + 1e-8)
        axes[1, 1].plot(epochs, uncertainty_ratio, 'o-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Uncertainty/Mean Ratio')
        axes[1, 1].set_title('Relative Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'{layer}_bayesian_uncertainty.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {layer}_bayesian_uncertainty.png")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        print(f"\n{'='*80}")
        print("EDGE WEIGHT ANALYSIS REPORT")
        print(f"{'='*80}\n")
        
        print(f"Log directory: {self.log_dir}")
        print(f"Number of epochs logged: {len(self.logs)}")
        
        if self.logs:
            print(f"Epoch range: {self.logs[0]['epoch']} to {self.logs[-1]['epoch']}")
            
            # Get layer info from first log
            first_log_data = self.logs[0]['data']
            layers = list(set([l['layer'] for l in first_log_data]))
            print(f"Layers found: {', '.join(layers)}")
            
            # Generate all analyses
            print(f"\n{'='*80}")
            print("Generating visualizations...")
            print(f"{'='*80}\n")
            
            for layer in layers:
                print(f"\nAnalyzing {layer}...")
                self.analyze_transformation_pattern(layer=layer)
                self.analyze_weight_distributions(layer=layer)
                self.analyze_bayesian_uncertainty(layer=layer)
            
            print(f"\nComparing layers...")
            self.compare_layers()
            
            print(f"\n{'='*80}")
            print(f"Analysis complete! All plots saved to: {self.log_dir}")
            print(f"{'='*80}\n")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Bayesian edge weights')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Directory containing edge weight logs')
    parser.add_argument('--layer', type=str, default='conv1',
                       choices=['conv1', 'conv2'],
                       help='Layer to analyze')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to analyze (default: last)')
    
    args = parser.parse_args()
    
    analyzer = BayesianWeightAnalyzer(args.log_dir)
    
    # Generate full report
    analyzer.generate_report()
    
    # Or run specific analyses
    # analyzer.analyze_transformation_pattern(layer=args.layer)
    # analyzer.analyze_weight_distributions(layer=args.layer)
    # analyzer.compare_layers(epoch=args.epoch)
    # analyzer.analyze_bayesian_uncertainty(layer=args.layer)