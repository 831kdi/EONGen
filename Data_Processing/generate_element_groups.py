"""
Generate element grouping from correlation matrix using hierarchical clustering

This script analyzes a correlation matrix between elements and automatically
identifies groups of elements with similar structural energy patterns.

Usage:
    python generate_element_groups.py --input correlations.csv --output element_groups.csv
    python generate_element_groups.py --input correlations.csv --n_groups 5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')


class ElementGroupGenerator:
    """
    Generate element groups from correlation matrix using hierarchical clustering
    """
    
    def __init__(self, correlation_csv: str, method: str = 'ward'):
        """
        Args:
            correlation_csv: Path to correlation matrix CSV file
            method: Clustering method ('ward', 'average', 'complete', 'single')
        """
        self.correlation_csv = Path(correlation_csv)
        self.method = method
        
        print("="*80)
        print("ELEMENT GROUPING FROM CORRELATION MATRIX")
        print("="*80)
        print(f"Input: {correlation_csv}")
        print(f"Method: {method}")
        
        # Load correlation matrix
        self.load_correlation_matrix()
        
    def load_correlation_matrix(self):
        """Load and validate correlation matrix"""
        print("\nLoading correlation matrix...")
        
        self.corr_df = pd.read_csv(self.correlation_csv, index_col=0)
        self.elements = self.corr_df.columns.tolist()
        self.n_elements = len(self.elements)
        
        print(f"  Elements: {self.n_elements}")
        print(f"  Matrix shape: {self.corr_df.shape}")
        
        # Validate it's a square matrix
        if self.corr_df.shape[0] != self.corr_df.shape[1]:
            raise ValueError(f"Correlation matrix must be square! Got shape {self.corr_df.shape}")
        
        # Check correlation range
        corr_min = self.corr_df.values.min()
        corr_max = self.corr_df.values.max()
        
        print(f"\nCorrelation statistics:")
        print(f"  Min: {corr_min:.4f}")
        print(f"  Max: {corr_max:.4f}")
        print(f"  Mean: {self.corr_df.values.mean():.4f}")
        
        if corr_min < -1 or corr_max > 1:
            print("  WARNING: Correlation values outside [-1, 1] range!")
    
    def compute_distance_matrix(self):
        """Convert correlation to distance matrix"""
        print("\nComputing distance matrix...")
        
        # Distance = 1 - correlation
        # High correlation (close to 1) → low distance (close to 0)
        # Low/negative correlation → high distance
        self.distance_matrix = 1 - self.corr_df.values
        
        # Make symmetric and set diagonal to 0
        np.fill_diagonal(self.distance_matrix, 0)
        
        print(f"  Distance range: [{self.distance_matrix.min():.4f}, {self.distance_matrix.max():.4f}]")
        
        # Convert to condensed distance matrix for linkage
        self.condensed_dist = squareform(self.distance_matrix)
    
    def perform_clustering(self):
        """Perform hierarchical clustering"""
        print(f"\nPerforming hierarchical clustering ({self.method} method)...")
        
        # Compute linkage
        self.linkage_matrix = linkage(self.condensed_dist, method=self.method)
        
        print("  Linkage matrix computed")
    
    def determine_optimal_clusters(self, min_clusters: int = 3, max_clusters: int = 8):
        """
        Determine optimal number of clusters by testing different values
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
        
        Returns:
            Recommended number of clusters
        """
        print("\n" + "="*80)
        print("TESTING DIFFERENT NUMBERS OF CLUSTERS")
        print("="*80)
        
        cluster_results = {}
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            
            # Count elements per cluster
            unique, counts = np.unique(clusters, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            # Calculate balance metric (std of cluster sizes)
            balance = np.std(counts)
            
            cluster_results[n_clusters] = {
                'sizes': cluster_sizes,
                'balance': balance,
                'min_size': counts.min(),
                'max_size': counts.max(),
            }
            
            print(f"\n{n_clusters} clusters:")
            print(f"  Sizes: {dict(sorted(cluster_sizes.items()))}")
            print(f"  Balance (std): {balance:.2f}")
            print(f"  Range: {counts.min()}-{counts.max()} elements")
        
        # Recommend based on balance and avoiding too small/large clusters
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        
        # Score each option (lower is better)
        scores = {}
        for n, result in cluster_results.items():
            # Penalize:
            # - High imbalance (std)
            # - Very small clusters (<3)
            # - Very large clusters (>40% of total)
            score = result['balance']
            if result['min_size'] < 3:
                score += 10  # Heavy penalty for tiny clusters
            if result['max_size'] > 0.4 * self.n_elements:
                score += 5   # Penalty for huge clusters
            scores[n] = score
        
        recommended = min(scores, key=scores.get)
        
        print(f"\nRecommended: {recommended} clusters")
        print(f"  Most balanced grouping with reasonable cluster sizes")
        print(f"  Score: {scores[recommended]:.2f} (lower is better)")
        
        return recommended
    
    def create_groups(self, n_clusters: int = None):
        """
        Create element groups
        
        Args:
            n_clusters: Number of clusters. If None, automatically determine.
        """
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        
        print("\n" + "="*80)
        print(f"CREATING {n_clusters} GROUPS")
        print("="*80)
        
        # Get cluster assignments
        clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        # Create mapping
        self.element_groups = {}
        for elem, cluster_id in zip(self.elements, clusters):
            if cluster_id not in self.element_groups:
                self.element_groups[cluster_id] = []
            self.element_groups[cluster_id].append(elem)
        
        # Print groups
        print("\nElement Groups:")
        for group_id in sorted(self.element_groups.keys()):
            elements = self.element_groups[group_id]
            print(f"\nGroup {group_id}: {len(elements)} elements")
            print(f"  {', '.join(sorted(elements))}")
        
        # Create DataFrame
        group_data = []
        for elem, cluster_id in zip(self.elements, clusters):
            group_data.append({'element': elem, 'group': int(cluster_id)})
        
        self.groups_df = pd.DataFrame(group_data)
        
        return self.groups_df
    
    def save_groups(self, output_csv: str, output_txt: str = None):
        """
        Save element groups to files
        
        Args:
            output_csv: Path to output CSV file
            output_txt: Path to output text file (optional)
        """
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save CSV
        self.groups_df.to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")
        
        # Save text file
        if output_txt is None:
            output_txt = str(Path(output_csv).with_suffix('.txt'))
        
        with open(output_txt, 'w') as f:
            f.write("# Element Grouping Based on Correlation Clustering\n")
            f.write(f"# Method: Hierarchical Clustering ({self.method} method)\n")
            f.write(f"# Number of Groups: {len(self.element_groups)}\n")
            f.write(f"# Input: {self.correlation_csv}\n")
            f.write("\n")
            
            for group_id in sorted(self.element_groups.keys()):
                elements = self.element_groups[group_id]
                f.write(f"Group {group_id}: {len(elements)} elements\n")
                f.write(",".join(sorted(elements)) + "\n")
                f.write("\n")
        
        print(f"Saved TXT: {output_txt}")
    
    def plot_dendrogram(self, output_path: str = None):
        """
        Plot dendrogram of hierarchical clustering
        
        Args:
            output_path: Path to save figure (optional)
        """
        print("\nGenerating dendrogram...")
        
        plt.figure(figsize=(15, 8))
        
        dendrogram(
            self.linkage_matrix,
            labels=self.elements,
            leaf_font_size=10,
            leaf_rotation=90,
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.method} method)', fontsize=14)
        plt.xlabel('Element', fontsize=12)
        plt.ylabel('Distance (1 - Correlation)', fontsize=12)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved dendrogram: {output_path}")
        else:
            print("Dendrogram generated (not saved)")
        
        plt.close()
    
    def plot_correlation_matrix(self, output_path: str = None):
        """
        Plot correlation matrix heatmap with clustering order
        
        Args:
            output_path: Path to save figure (optional)
        """
        print("\nGenerating correlation heatmap...")
        
        # Reorder based on clustering
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(self.linkage_matrix)
        
        corr_ordered = self.corr_df.iloc[order, order]
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            corr_ordered,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        
        plt.title('Correlation Matrix (Clustered)', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap: {output_path}")
        else:
            print("Heatmap generated (not saved)")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate element groups from correlation matrix"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to correlation matrix CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='element_groups.csv',
        help='Output CSV file path (default: element_groups.csv)'
    )
    parser.add_argument(
        '--n_groups',
        type=int,
        default=None,
        help='Number of groups (default: automatically determined)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='ward',
        choices=['ward', 'average', 'complete', 'single'],
        help='Clustering method (default: ward)'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save dendrogram and heatmap plots'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ElementGroupGenerator(args.input, method=args.method)
    
    # Compute distance matrix
    generator.compute_distance_matrix()
    
    # Perform clustering
    generator.perform_clustering()
    
    # Create groups
    groups_df = generator.create_groups(n_clusters=args.n_groups)
    
    # Save results
    output_csv = args.output
    output_txt = str(Path(output_csv).with_suffix('.txt'))
    generator.save_groups(output_csv, output_txt)
    
    # Generate plots if requested
    if args.save_plots:
        output_dir = Path(output_csv).parent
        dendrogram_path = output_dir / 'dendrogram.png'
        heatmap_path = output_dir / 'correlation_heatmap.png'
        
        generator.plot_dendrogram(str(dendrogram_path))
        generator.plot_correlation_matrix(str(heatmap_path))
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_csv}")
    print(f"  - {output_txt}")
    if args.save_plots:
        print(f"  - {dendrogram_path}")
        print(f"  - {heatmap_path}")
    
    print(f"\nNext step:")
    print(f"  Use '{output_csv}' in split_xyz_data.py for correlation-based splitting")


if __name__ == "__main__":
    main()
