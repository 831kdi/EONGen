"""
Modified XYZ data splitter that:
1. Identifies structures with E_rel = 0 (ground states)
2. Splits those ground states using 3 strategies (8:1:1)
3. Adds all E_rel != 0 structures to training set

Usage:
    python split_xyz_ground_states.py --input clusters_full.xyz --output_dir data/splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
import re


def parse_xyz_file(xyz_file):
    """Parse XYZ file and extract structures with metadata"""
    structures = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        
        if i + 1 >= len(lines):
            break
        
        comment = lines[i + 1].strip()
        
        # Parse metadata
        energy = None
        e_rel = None
        element = None
        struct_id = None
        
        energy_match = re.search(r'Energy=([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', comment)
        if energy_match:
            energy = float(energy_match.group(1))
        
        e_rel_match = re.search(r'E_rel:=T\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', comment)
        if e_rel_match:
            e_rel = float(e_rel_match.group(1))
        
        element_match = re.search(r'Element:=T\s+(\w+)=T', comment)
        if element_match:
            element = element_match.group(1)
        
        id_match = re.search(r'ID:=T\s+([^=]+)=T', comment)
        if id_match:
            struct_id = id_match.group(1).strip()
        
        # Read atom lines
        if i + 2 + n_atoms > len(lines):
            break
        
        atom_lines = []
        for j in range(n_atoms):
            atom_lines.append(lines[i + 2 + j])
        
        # Store full structure
        structure = {
            'n_atoms': n_atoms,
            'comment': comment,
            'atom_lines': atom_lines,
            'energy': energy,
            'e_rel': e_rel if e_rel is not None else 0.0,
            'element': element,
            'id': struct_id,
        }
        
        structures.append(structure)
        i += n_atoms + 2
    
    return structures


def write_xyz_file(structures, output_file):
    """Write structures to XYZ file"""
    with open(output_file, 'w') as f:
        for struct in structures:
            f.write(f"{struct['n_atoms']}\n")
            f.write(f"{struct['comment']}\n")
            for line in struct['atom_lines']:
                f.write(line)


class GroundStateXYZSplitter:
    """
    Split XYZ data with special handling for ground states (E_rel=0)
    """
    
    def __init__(self, xyz_path: str, output_dir: str, random_seed: int = 42):
        self.xyz_path = Path(xyz_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        print("="*80)
        print("LOADING AND FILTERING DATA")
        print("="*80)
        print(f"Reading from: {xyz_path}")
        
        # Parse structures
        all_structures = parse_xyz_file(xyz_path)
        print(f"Total structures loaded: {len(all_structures)}")
        
        # Separate ground states (E_rel=0) from excited states
        self.ground_states = []
        self.excited_states = []
        
        for struct in all_structures:
            if abs(struct['e_rel']) < 1e-6:  # E_rel = 0 (with numerical tolerance)
                self.ground_states.append(struct)
            else:
                self.excited_states.append(struct)
        
        print(f"\nData separation:")
        print(f"  Ground states (E_rel=0): {len(self.ground_states)}")
        print(f"  Excited states (E_rel≠0): {len(self.excited_states)}")
        
        # Extract metadata for ground states
        self.extract_metadata()
    
    def extract_metadata(self):
        """Extract metadata from ground state structures"""
        self.sizes = []
        self.elements = []
        self.energies = []
        
        for struct in self.ground_states:
            self.sizes.append(struct['n_atoms'])
            self.elements.append(struct['element'])
            self.energies.append(struct['energy'])
        
        print(f"\nGround state statistics:")
        print(f"  Size range: {min(self.sizes)} - {max(self.sizes)} atoms")
        print(f"  Elements: {set(self.elements)}")
        print(f"  Energy range: {min(self.energies):.2f} - {max(self.energies):.2f} eV")
    
    def random_split(self):
        """Strategy 1: Random split of ground states (8:1:1)"""
        print("\n" + "="*80)
        print("STRATEGY 1: RANDOM SPLIT (Ground States Only)")
        print("="*80)
        
        n_total = len(self.ground_states)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val
        
        # Shuffle indices
        indices = np.random.permutation(n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Get structures
        train_ground = [self.ground_states[i] for i in train_idx]
        val_structures = [self.ground_states[i] for i in val_idx]
        test_structures = [self.ground_states[i] for i in test_idx]
        
        # Add ALL excited states to training
        train_structures = train_ground + self.excited_states
        
        print(f"Split sizes:")
        print(f"  Training:   {len(train_structures)} ({len(train_ground)} ground + {len(self.excited_states)} excited)")
        print(f"  Validation: {len(val_structures)} (ground states only)")
        print(f"  Test:       {len(test_structures)} (ground states only)")
        
        # Save
        output_subdir = self.output_dir / 'strategy1_random'
        output_subdir.mkdir(exist_ok=True)
        
        write_xyz_file(train_structures, output_subdir / 'train.xyz')
        write_xyz_file(val_structures, output_subdir / 'val.xyz')
        write_xyz_file(test_structures, output_subdir / 'test.xyz')
        
        print(f"\nSaved to: {output_subdir}")
        
        return train_structures, val_structures, test_structures
    
    def size_based_split(self):
        """Strategy 2: Size-based split of ground states (8:1:1)"""
        print("\n" + "="*80)
        print("STRATEGY 2: SIZE-BASED SPLIT (Ground States Only)")
        print("="*80)
        
        # Sort by size
        sorted_indices = np.argsort(self.sizes)
        
        n_total = len(self.ground_states)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_idx = sorted_indices[:n_train]
        val_idx = sorted_indices[n_train:n_train+n_val]
        test_idx = sorted_indices[n_train+n_val:]
        
        train_ground = [self.ground_states[i] for i in train_idx]
        val_structures = [self.ground_states[i] for i in val_idx]
        test_structures = [self.ground_states[i] for i in test_idx]
        
        # Add ALL excited states to training
        train_structures = train_ground + self.excited_states
        
        train_sizes = [s['n_atoms'] for s in train_ground]
        val_sizes = [s['n_atoms'] for s in val_structures]
        test_sizes = [s['n_atoms'] for s in test_structures]
        
        print(f"Size distributions:")
        print(f"  Training:   {min(train_sizes)}-{max(train_sizes)} atoms ({len(train_structures)} total)")
        print(f"  Validation: {min(val_sizes)}-{max(val_sizes)} atoms ({len(val_structures)} total)")
        print(f"  Test:       {min(test_sizes)}-{max(test_sizes)} atoms ({len(test_structures)} total)")
        
        # Save
        output_subdir = self.output_dir / 'strategy2_size'
        output_subdir.mkdir(exist_ok=True)
        
        write_xyz_file(train_structures, output_subdir / 'train.xyz')
        write_xyz_file(val_structures, output_subdir / 'val.xyz')
        write_xyz_file(test_structures, output_subdir / 'test.xyz')
        
        print(f"\nSaved to: {output_subdir}")
        
        return train_structures, val_structures, test_structures
    
    def correlation_based_split(self):
        """Strategy 3: Element group-based split of ground states (8:1:1)"""
        print("\n" + "="*80)
        print("STRATEGY 3: ELEMENT GROUP-BASED SPLIT (Ground States Only)")
        print("="*80)
        
        # Group by element
        element_groups = defaultdict(list)
        for i, struct in enumerate(self.ground_states):
            element_groups[struct['element']].append(i)
        
        print(f"Element groups:")
        for elem, indices in element_groups.items():
            print(f"  {elem}: {len(indices)} structures")
        
        # Split each group proportionally
        train_idx = []
        val_idx = []
        test_idx = []
        
        for elem, indices in element_groups.items():
            n_total = len(indices)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            
            # Shuffle within group
            shuffled = np.random.permutation(indices)
            
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train:n_train+n_val])
            test_idx.extend(shuffled[n_train+n_val:])
        
        train_ground = [self.ground_states[i] for i in train_idx]
        val_structures = [self.ground_states[i] for i in val_idx]
        test_structures = [self.ground_states[i] for i in test_idx]
        
        # Add ALL excited states to training
        train_structures = train_ground + self.excited_states
        
        print(f"\nSplit sizes:")
        print(f"  Training:   {len(train_structures)} ({len(train_ground)} ground + {len(self.excited_states)} excited)")
        print(f"  Validation: {len(val_structures)} (ground states only)")
        print(f"  Test:       {len(test_structures)} (ground states only)")
        
        # Save
        output_subdir = self.output_dir / 'strategy3_element_groups'
        output_subdir.mkdir(exist_ok=True)
        
        write_xyz_file(train_structures, output_subdir / 'train.xyz')
        write_xyz_file(val_structures, output_subdir / 'val.xyz')
        write_xyz_file(test_structures, output_subdir / 'test.xyz')
        
        print(f"\nSaved to: {output_subdir}")
        
        return train_structures, val_structures, test_structures
    
    def run_all_strategies(self):
        """Run all three splitting strategies"""
        print("\n" + "="*80)
        print("RUNNING ALL SPLITTING STRATEGIES")
        print("="*80)
        
        self.random_split()
        self.size_based_split()
        self.correlation_based_split()
        
        print("\n" + "="*80)
        print("SPLITTING COMPLETE!")
        print("="*80)
        print(f"\nAll splits saved to: {self.output_dir}")
        print("\nStrategy directories:")
        print("  - strategy1_random/")
        print("  - strategy2_size/")
        print("  - strategy3_element_groups/")
        print("\nEach contains: train.xyz, val.xyz, test.xyz")
        print("\nKey point: Validation and test sets contain ONLY ground states (E_rel=0)")
        print("          Training set contains ground states + ALL excited states")


def main():
    parser = argparse.ArgumentParser(description='Split XYZ data with ground state focus')
    parser.add_argument('--input', type=str, required=True, help='Input XYZ file')
    parser.add_argument('--output_dir', type=str, default='splits', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create splitter and run
    splitter = GroundStateXYZSplitter(
        xyz_path=args.input,
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    splitter.run_all_strategies()


if __name__ == '__main__':
    main()
