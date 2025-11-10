"""
Split atomic cluster XYZ data into train/val/test sets using three strategies:
1. Random split (8:1:1)
2. Size-based split (small→train, medium→val, large→test)
3. Correlation-based split (by element groups with balanced data counts)

Usage:
    python split_xyz_data.py --input clusters_full.xyz --output_dir data/splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from itertools import combinations
from ase.io import read, write
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class XYZDataSplitter:
    """
    Split XYZ cluster data using three different strategies
    """
    
    def __init__(self, xyz_path: str, output_dir: str, random_seed: int = 42):
        """
        Args:
            xyz_path: Path to multi-structure XYZ file
            output_dir: Directory to save split files
            random_seed: Random seed for reproducibility
        """
        self.xyz_path = Path(xyz_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        print(f"Reading from: {xyz_path}")
        
        # Read all structures
        self.structures = read(str(self.xyz_path), index=':')
        print(f"Loaded {len(self.structures)} structures")
        
        # Extract metadata
        self._extract_metadata()
        
    def _extract_metadata(self):
        """Extract element, size, and other metadata from structures"""
        print("\nExtracting metadata...")
        
        self.metadata = []
        
        for idx, atoms in enumerate(tqdm(self.structures, desc="Processing")):
            # Get basic info
            symbols = atoms.get_chemical_symbols()
            element = symbols[0]  # Primary element
            n_atoms = len(atoms)
            
            # Parse comment line for additional info
            comment = atoms.info.get('comment', '')
            
            # Extract cluster ID
            cluster_id = None
            if 'ID:' in comment:
                try:
                    cluster_id = comment.split('ID:')[1].split('|')[0].strip()
                except:
                    pass
            if cluster_id is None:
                cluster_id = f"{element}/{n_atoms}/{idx}"
            
            # Extract energy
            energy = 0.0
            if 'Energy' in comment:
                try:
                    energy_str = comment.split('Energy')[1].split('eV')[0]
                    energy = float(energy_str.replace('=', '').strip())
                except:
                    pass
            
            # Extract relative energy
            energy_rel = 0.0
            if 'E_rel:' in comment:
                try:
                    energy_rel = float(comment.split('E_rel:')[1].split()[0])
                except:
                    pass
            
            self.metadata.append({
                'index': idx,
                'cluster_id': cluster_id,
                'element': element,
                'n_atoms': n_atoms,
                'energy': energy,
                'energy_rel': energy_rel,
            })
        
        # Convert to DataFrame for easier manipulation
        self.df = pd.DataFrame(self.metadata)
        
        # Print summary
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"Total structures: {len(self.df)}")
        print(f"Elements: {self.df['element'].nunique()}")
        print(f"Size range: {self.df['n_atoms'].min()}-{self.df['n_atoms'].max()} atoms")
        
        print("\nElement distribution:")
        elem_counts = self.df['element'].value_counts()
        for elem, count in elem_counts.head(10).items():
            print(f"  {elem}: {count}")
        if len(elem_counts) > 10:
            print(f"  ... and {len(elem_counts)-10} more elements")
        
        print("\nSize distribution:")
        size_counts = self.df['n_atoms'].value_counts().sort_index()
        print(f"  Min size: {size_counts.index.min()} atoms ({size_counts.iloc[0]} structures)")
        print(f"  Max size: {size_counts.index.max()} atoms ({size_counts.iloc[-1]} structures)")
        print(f"  Mean size: {self.df['n_atoms'].mean():.1f} atoms")
    
    def random_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Strategy 1: Random split (8:1:1)
        """
        print("\n" + "="*80)
        print("STRATEGY 1: RANDOM SPLIT")
        print("="*80)
        
        n_total = len(self.df)
        indices = np.random.permutation(n_total)
        
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create split dataframes
        train_df = self.df.iloc[train_indices].copy()
        val_df = self.df.iloc[val_indices].copy()
        test_df = self.df.iloc[test_indices].copy()
        
        # Add split labels
        self.df['split_random'] = 'train'
        self.df.loc[val_indices, 'split_random'] = 'val'
        self.df.loc[test_indices, 'split_random'] = 'test'
        
        self._print_split_stats('random', train_df, val_df, test_df)
        
        # Save splits
        self._save_split('random', train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def size_based_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Strategy 2: Size-based split
        Small clusters → train, medium → val, large → test
        """
        print("\n" + "="*80)
        print("STRATEGY 2: SIZE-BASED SPLIT")
        print("="*80)
        
        # Sort by size
        df_sorted = self.df.sort_values('n_atoms').reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        # Split by cumulative count
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train+n_val].copy()
        test_df = df_sorted.iloc[n_train+n_val:].copy()
        
        # Add split labels
        self.df['split_size'] = 'train'
        self.df.loc[val_df['index'], 'split_size'] = 'val'
        self.df.loc[test_df['index'], 'split_size'] = 'test'
        
        self._print_split_stats('size', train_df, val_df, test_df)
        
        # Save splits
        self._save_split('size', train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _optimize_group_assignment(self, group_counts, target_train, target_val, target_test):
        """
        Find optimal assignment of groups to splits using exhaustive search
        
        For small number of groups (≤10), tries all possible assignments.
        For larger numbers, uses heuristic search.
        
        Args:
            group_counts: Series with group_id -> count
            target_train, target_val, target_test: Target counts
        
        Returns:
            train_groups, val_groups, test_groups: Lists of group IDs
        """
        from itertools import combinations
        
        groups = list(group_counts.index)
        n_groups = len(groups)
        
        print(f"\nOptimizing group assignment ({n_groups} groups)...")
        
        if n_groups <= 7:
            # Exhaustive search for small number of groups
            return self._exhaustive_search(groups, group_counts, target_train, target_val, target_test)
        else:
            # Heuristic search for larger number of groups
            return self._heuristic_search(groups, group_counts, target_train, target_val, target_test)
    
    def _exhaustive_search(self, groups, group_counts, target_train, target_val, target_test):
        """Try all possible group assignments"""
        best_score = float('inf')
        best_assignment = None
        
        n_groups = len(groups)
        
        # Try all possible ways to split groups into 3 sets
        # Each group can go to train, val, or test
        for n_test in range(1, n_groups):  # At least 1 group in test
            for test_groups in combinations(groups, n_test):
                remaining = [g for g in groups if g not in test_groups]
                
                for n_val in range(1, len(remaining)):  # At least 1 group in val
                    for val_groups in combinations(remaining, n_val):
                        train_groups = [g for g in remaining if g not in val_groups]
                        
                        if len(train_groups) == 0:  # Need at least 1 group in train
                            continue
                        
                        # Calculate counts
                        train_count = sum(group_counts[g] for g in train_groups)
                        val_count = sum(group_counts[g] for g in val_groups)
                        test_count = sum(group_counts[g] for g in test_groups)
                        
                        # Calculate score (sum of squared errors from targets)
                        score = (
                            (train_count - target_train) ** 2 +
                            (val_count - target_val) ** 2 +
                            (test_count - target_test) ** 2
                        )
                        
                        if score < best_score:
                            best_score = score
                            best_assignment = (list(train_groups), list(val_groups), list(test_groups))
        
        print(f"  Searched all combinations, found optimal assignment (score: {best_score:.0f})")
        return best_assignment
    
    def _heuristic_search(self, groups, group_counts, target_train, target_val, target_test):
        """
        Heuristic search for large number of groups
        Uses best-fit decreasing algorithm
        """
        # Sort groups by count (largest first)
        sorted_groups = sorted(groups, key=lambda g: group_counts[g], reverse=True)
        
        train_groups = []
        val_groups = []
        test_groups = []
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        for group_id in sorted_groups:
            count = group_counts[group_id]
            
            # Calculate error if we add this group to each split
            train_error = abs((train_count + count) - target_train)
            val_error = abs((val_count + count) - target_val)
            test_error = abs((test_count + count) - target_test)
            
            # Also consider current balance
            train_deficit = target_train - train_count
            val_deficit = target_val - val_count
            test_deficit = target_test - test_count
            
            # Assign to split with largest deficit that this group helps
            if train_deficit > 0 and count <= train_deficit * 1.5:
                train_groups.append(group_id)
                train_count += count
            elif val_deficit > 0 and count <= val_deficit * 1.5:
                val_groups.append(group_id)
                val_count += count
            elif test_deficit > 0 and count <= test_deficit * 1.5:
                test_groups.append(group_id)
                test_count += count
            else:
                # Choose split that minimizes error
                if train_error <= val_error and train_error <= test_error:
                    train_groups.append(group_id)
                    train_count += count
                elif val_error <= test_error:
                    val_groups.append(group_id)
                    val_count += count
                else:
                    test_groups.append(group_id)
                    test_count += count
        
        print(f"  Used heuristic search")
        return train_groups, val_groups, test_groups
    
    def correlation_based_split(
        self, 
        element_groups_csv: str = None,
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1
    ):
        """
        Strategy 3: Correlation-based split
        Split by element groups, balancing number of structures
        
        Args:
            element_groups_csv: Path to element_groups.csv file
            If None, will look in current directory
        """
        print("\n" + "="*80)
        print("STRATEGY 3: CORRELATION-BASED SPLIT")
        print("="*80)
        
        # Load element groups
        if element_groups_csv is None:
            # Try to find it
            candidates = [
                'element_groups.csv',
                'data/element_groups.csv',
                '../element_groups.csv'
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    element_groups_csv = candidate
                    break
        
        if element_groups_csv is None or not Path(element_groups_csv).exists():
            print("WARNING: element_groups.csv not found!")
            print("\nTo use correlation-based splitting:")
            print("  1. Provide correlations.csv (Pearson correlation matrix)")
            print("  2. Run: python scripts/generate_element_groups.py --input correlations.csv")
            print("  3. This will create element_groups.csv")
            print("  4. Then run this script again with --element_groups element_groups.csv")
            print("\nSkipping correlation-based split...")
            return None, None, None
        
        # Load element groups
        groups_df = pd.read_csv(element_groups_csv)
        element_to_group = dict(zip(groups_df['element'], groups_df['group']))
        
        print(f"Loaded element groups from: {element_groups_csv}")
        
        # Add group column to dataframe
        self.df['group'] = self.df['element'].map(element_to_group)
        
        # Count structures per group
        group_counts = self.df['group'].value_counts().sort_index()
        print("\nStructures per group:")
        for group_id, count in group_counts.items():
            elements_in_group = groups_df[groups_df['group'] == group_id]['element'].tolist()
            print(f"  Group {group_id}: {count} structures ({len(elements_in_group)} elements)")
        
        # Strategy: Assign groups to splits to balance data counts
        # Target: train ~80%, val ~10%, test ~10%
        n_total = len(self.df)
        target_train = int(train_ratio * n_total)
        target_val = int(val_ratio * n_total)
        target_test = int(test_ratio * n_total)
        
        print(f"\nTarget counts: train={target_train}, val={target_val}, test={target_test}")
        
        # Find best group assignment using optimization
        train_groups, val_groups, test_groups = self._optimize_group_assignment(
            group_counts, target_train, target_val, target_test
        )
        
        train_count = sum(group_counts[g] for g in train_groups)
        val_count = sum(group_counts[g] for g in val_groups)
        test_count = sum(group_counts[g] for g in test_groups)
        
        print(f"\nGroup assignments:")
        print(f"  Train: Groups {sorted(train_groups)} ({train_count} structures)")
        print(f"  Val:   Groups {sorted(val_groups)} ({val_count} structures)")
        print(f"  Test:  Groups {sorted(test_groups)} ({test_count} structures)")
        
        # Create split dataframes
        train_df = self.df[self.df['group'].isin(train_groups)].copy()
        val_df = self.df[self.df['group'].isin(val_groups)].copy()
        test_df = self.df[self.df['group'].isin(test_groups)].copy()
        
        # Add split labels
        self.df['split_correlation'] = 'train'
        self.df.loc[val_df['index'], 'split_correlation'] = 'val'
        self.df.loc[test_df['index'], 'split_correlation'] = 'test'
        
        self._print_split_stats('correlation', train_df, val_df, test_df)
        
        # Save splits
        self._save_split('correlation', train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _print_split_stats(self, split_name: str, train_df, val_df, test_df):
        """Print statistics for a split"""
        n_total = len(self.df)
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} ({len(train_df)/n_total*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/n_total*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
        
        print(f"\nSize ranges:")
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"  {name}: {df['n_atoms'].min()}-{df['n_atoms'].max()} atoms "
                  f"(mean: {df['n_atoms'].mean():.1f})")
        
        print(f"\nElement distribution:")
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            n_elem = df['element'].nunique()
            print(f"  {name}: {n_elem} unique elements")
    
    def _save_split(self, split_name: str, train_df, val_df, test_df):
        """Save split to XYZ files"""
        print(f"\nSaving {split_name} split...")
        
        for split_label, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Get structures for this split
            indices = df['index'].tolist()
            structures = [self.structures[i] for i in indices]
            
            # Save to XYZ file
            output_path = self.output_dir / f"clusters_{split_name}_{split_label}.xyz"
            write(str(output_path), structures)
            print(f"  Saved {len(structures)} structures to: {output_path}")
    
    def save_metadata(self):
        """Save metadata with split assignments"""
        output_path = self.output_dir / "splits_metadata.csv"
        self.df.to_csv(output_path, index=False)
        print(f"\nSaved metadata to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split atomic cluster XYZ data into train/val/test sets"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input XYZ file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--element_groups', 
        type=str, 
        default=None,
        help='Path to element_groups.csv (for correlation-based split)'
    )
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio', 
        type=float, 
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test_ratio', 
        type=float, 
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"ERROR: Ratios must sum to 1.0 (got {total_ratio})")
        return
    
    print("="*80)
    print("ATOMIC CLUSTER DATA SPLITTING")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.random_seed}")
    
    # Create splitter
    splitter = XYZDataSplitter(
        args.input, 
        args.output_dir, 
        random_seed=args.random_seed
    )
    
    # Perform all three splitting strategies
    print("\n" + "="*80)
    print("PERFORMING SPLITS")
    print("="*80)
    
    # Strategy 1: Random
    splitter.random_split(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Strategy 2: Size-based
    splitter.size_based_split(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Strategy 3: Correlation-based
    splitter.correlation_based_split(
        args.element_groups,
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    # Save metadata
    splitter.save_metadata()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Output files saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - clusters_random_{train,val,test}.xyz")
    print("  - clusters_size_{train,val,test}.xyz")
    print("  - clusters_correlation_{train,val,test}.xyz")
    print("  - splits_metadata.csv")


if __name__ == "__main__":
    main()
