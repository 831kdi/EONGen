"""
Data preprocessing script to convert split XYZ files to PyTorch Geometric format
for nanoparticle generation with CFM + GoTenNet

Usage:
    python preprocess_xyz_to_pyg.py --train_xyz data/train.xyz --val_xyz data/val.xyz --test_xyz data/test.xyz --output_dir processed_data/
"""

import argparse
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import re


def parse_xyz_file(xyz_file):
    """
    Parse XYZ file with extended XYZ format
    Format:
        <n_atoms>
        Properties=... Energy=<value> ... Element:=T <element>=T ...
        <atom> <x> <y> <z>
        ...
    """
    structures = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read number of atoms
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        
        # Read comment line
        if i + 1 >= len(lines):
            break
        comment = lines[i + 1].strip()
        
        # Parse energy from comment
        # Handle both "Energy=" and "Energy = " formats
        energy = None
        energy_match = re.search(r'Energy\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', comment)
        if energy_match:
            energy = float(energy_match.group(1))
        
        # Parse E_rel from comment (IMPORTANT: relative energy for filtering!)
        # Format: "E_rel: 0.0" or "E_rel:0.0"
        e_rel = None
        e_rel_match = re.search(r'E_rel:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', comment)
        if e_rel_match:
            e_rel = float(e_rel_match.group(1))
        
        # Parse element
        # Handle both "Element:=T Nb=T" and "Element: Nb" formats
        element = None
        element_match = re.search(r'Element:\s*(?:=T\s+)?(\w+)', comment)
        if element_match:
            element = element_match.group(1)
        
        # Parse ID
        # Handle both "ID:=T xxx=T" and "ID: xxx" formats
        structure_id = None
        id_match = re.search(r'ID:\s*(?:=T\s+)?([^|=]+?)(?:=T|\s*\||$)', comment)
        if id_match:
            structure_id = id_match.group(1).strip()
        
        # Read atom positions
        if i + 2 + n_atoms > len(lines):
            break
            
        atoms = []
        positions = []
        for j in range(n_atoms):
            line = lines[i + 2 + j].strip().split()
            if len(line) >= 4:
                symbol = line[0]
                pos = [float(line[1]), float(line[2]), float(line[3])]
                atoms.append(symbol)
                positions.append(pos)
        
        # Infer element if not parsed
        if element is None and atoms:
            element = atoms[0]
        
        structure = {
            'n_atoms': n_atoms,
            'symbols': atoms,
            'positions': np.array(positions),
            'energy': energy if energy is not None else 0.0,
            'e_rel': e_rel if e_rel is not None else 0.0,  # NEW: Store E_rel
            'element': element,
            'id': structure_id,
            'comment': comment
        }
        
        structures.append(structure)
        
        # Move to next structure
        i += n_atoms + 2
    
    return structures


class NanoparticleDataset(Dataset):
    """
    PyTorch Geometric Dataset for nanoparticle structures
    
    Note: edge_index is NOT stored - GoTenNet will build it internally
          during training based on its own cutoff parameter.
    """
    def __init__(self, xyz_file, element_mapping=None):
        super().__init__()
        
        # Create element mapping if not provided
        if element_mapping is None:
            self.element_mapping = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
                'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
                'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
                'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            }
        else:
            self.element_mapping = element_mapping
        
        # Load all structures
        print(f"Loading structures from {xyz_file}...")
        self.structures = parse_xyz_file(xyz_file)
        print(f"Loaded {len(self.structures)} structures")
        
        # Process metadata
        self.process_metadata()
        
    def process_metadata(self):
        """Process parsed structures into metadata"""
        self.metadata = []
        
        for struct in tqdm(self.structures, desc="Processing metadata"):
            meta = {}
            
            # Get element ID
            element = struct['element']
            if element:
                meta['element'] = element
                meta['element_id'] = self.element_mapping.get(element, 0)
            else:
                meta['element'] = 'Unknown'
                meta['element_id'] = 0
            
            # Get atomic numbers from symbols
            atomic_numbers = []
            for symbol in struct['symbols']:
                atomic_numbers.append(self.element_mapping.get(symbol, 0))
            
            meta['n_atoms'] = struct['n_atoms']
            meta['atomic_numbers'] = np.array(atomic_numbers, dtype=np.int64)
            meta['positions'] = struct['positions']
            meta['energy'] = struct['energy']
            meta['e_rel'] = struct['e_rel']  # NEW: Store E_rel
            meta['e_rel_per_atom'] = struct['e_rel'] / struct['n_atoms']  # NEW: E_rel per atom
            meta['id'] = struct['id'] if struct['id'] else f"{element}/{struct['n_atoms']}/?"
            
            self.metadata.append(meta)
    
    def len(self):
        return len(self.metadata)
    
    def get(self, idx):
        """Get a single data sample"""
        meta = self.metadata[idx]
        
        # Atomic numbers (element types)
        z = torch.from_numpy(meta['atomic_numbers']).long()
        
        # 3D positions (center at origin)
        pos = torch.from_numpy(meta['positions']).float()
        pos = pos - pos.mean(dim=0, keepdim=True)  # Center at origin
        
        # E_rel per atom (IMPORTANT: Use this for filtering, not absolute energy!)
        # Store in data.y for easy access during training
        e_rel_per_atom = torch.tensor([meta['e_rel_per_atom']], dtype=torch.float)
        
        # Number of atoms
        n_atoms = torch.tensor([meta['n_atoms']], dtype=torch.long)
        
        # Element type as categorical
        element_id = torch.tensor([meta['element_id']], dtype=torch.long)
        
        # Create Data object (NO edge_index - GoTenNet builds it internally!)
        data = Data(
            z=z,
            pos=pos,
            y=e_rel_per_atom,  # E_rel/atom for filtering
            n_atoms=n_atoms,
            element_id=element_id,
            element=meta['element'],
            structure_id=meta['id'],
        )
        
        return data


def normalize_energies(train_dataset, val_dataset, test_dataset):
    """
    Print E_rel/atom statistics (no normalization - already relative!)
    """
    # Collect all training e_rel_per_atom values
    train_e_rel = []
    for i in range(len(train_dataset)):
        meta = train_dataset.metadata[i]
        train_e_rel.append(meta['e_rel_per_atom'])
    
    train_e_rel = np.array(train_e_rel)
    
    # Compute statistics
    mean_e_rel = train_e_rel.mean()
    std_e_rel = train_e_rel.std()
    
    print(f"\nE_rel/atom statistics (no normalization applied):")
    print(f"  Mean: {mean_e_rel:.4f} eV/atom")
    print(f"  Std:  {std_e_rel:.4f} eV/atom")
    print(f"  Min:  {train_e_rel.min():.4f} eV/atom")
    print(f"  Max:  {train_e_rel.max():.4f} eV/atom")
    
    # Count ground states (E_rel ≈ 0)
    ground_states = np.sum(np.abs(train_e_rel) < 0.001)
    print(f"  Ground states (E_rel ≈ 0): {ground_states} / {len(train_e_rel)}")
    
    # Save statistics
    stats = {
        'e_rel_per_atom_mean': mean_e_rel,
        'e_rel_per_atom_std': std_e_rel,
        'e_rel_per_atom_min': train_e_rel.min(),
        'e_rel_per_atom_max': train_e_rel.max(),
    }
    
    return stats


def save_datasets(train_dataset, val_dataset, test_dataset, output_dir, stats):
    """Save processed datasets"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as pickle files (preserves all metadata)
    print("\nSaving datasets...")
    
    with open(output_dir / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    print(f"  Saved training dataset: {len(train_dataset)} samples")
    
    with open(output_dir / 'val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    print(f"  Saved validation dataset: {len(val_dataset)} samples")
    
    with open(output_dir / 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    print(f"  Saved test dataset: {len(test_dataset)} samples")
    
    # Save statistics
    with open(output_dir / 'statistics.pkl', 'wb') as f:
        pickle.dump(stats, f)
    print(f"  Saved statistics")
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"  Training:   {len(train_dataset)} structures")
    print(f"  Validation: {len(val_dataset)} structures")
    print(f"  Test:       {len(test_dataset)} structures")
    
    # Print sample data
    print("\nSample data point:")
    sample = train_dataset.get(0)
    print(f"  Element: {sample.element}")
    print(f"  Structure ID: {sample.structure_id}")
    print(f"  N atoms: {sample.n_atoms.item()}")
    print(f"  E_rel/atom: {sample.y.item():.4f} eV/atom")
    print(f"  Atomic numbers (z): {sample.z.tolist()[:5]}... (showing first 5)")
    print(f"\n  Positions (Å, centered at origin):")
    print(f"    Shape: {sample.pos.shape}")
    for i in range(min(10, len(sample.pos))):
        x, y, z = sample.pos[i]
        print(f"    Atom {i:2d}: [{x:8.4f}, {y:8.4f}, {z:8.4f}]")
    if len(sample.pos) > 10:
        print(f"    ... ({len(sample.pos) - 10} more atoms)")
    print(f"\n  Note: edge_index will be built by GoTenNet during training")


def main():
    parser = argparse.ArgumentParser(description='Preprocess XYZ files to PyTorch Geometric format')
    parser.add_argument('--train_xyz', type=str, required=True, help='Path to training XYZ file')
    parser.add_argument('--val_xyz', type=str, required=True, help='Path to validation XYZ file')
    parser.add_argument('--test_xyz', type=str, required=True, help='Path to test XYZ file')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--n_max', type=int, default=None, help='Filter by max number of atoms (e.g., 20)')
    parser.add_argument('--debug', action='store_true', help='Print debug info about XYZ format')
    
    args = parser.parse_args()
    
    # Debug mode: inspect XYZ format
    if args.debug:
        print("\n" + "="*80)
        print("DEBUG MODE: Inspecting XYZ file format")
        print("="*80)
        
        from ase.io import read
        
        # Read first structure
        print(f"\nReading first structure from: {args.train_xyz}")
        struct = read(args.train_xyz, index=0)
        
        print(f"\nStructure info:")
        print(f"  Number of atoms: {len(struct)}")
        print(f"  Chemical formula: {struct.get_chemical_formula()}")
        print(f"  Symbols: {struct.get_chemical_symbols()[:5]}...")  # First 5
        
        print(f"\nASE info dict contents:")
        for key, value in struct.info.items():
            print(f"  {key}: {value}")
        
        print(f"\nDirect attributes:")
        if hasattr(struct, 'comment'):
            print(f"  struct.comment: {struct.comment}")
        
        print("\n" + "="*80)
        print("END DEBUG MODE")
        print("="*80)
        print("\nIf energy is missing, you may need to:")
        print("1. Check your XYZ file format")
        print("2. Add energy to ASE info during file reading")
        print("3. Use a different energy source (e.g., from filename)")
        return
    
    # Create datasets
    print("="*80)
    print("CREATING PYTORCH GEOMETRIC DATASETS")
    print("="*80)
    
    train_dataset = NanoparticleDataset(args.train_xyz)
    val_dataset = NanoparticleDataset(args.val_xyz)
    test_dataset = NanoparticleDataset(args.test_xyz)
    
    # ========== SIZE FILTERING (OPTIONAL) ==========
    if args.n_max is not None:
        print(f"\n🔢 Filtering structures with n_atoms ≤ {args.n_max}...")
        
        def filter_by_size(dataset, n_max):
            """Filter dataset by max number of atoms"""
            original_metadata = dataset.metadata
            filtered_metadata = []
            
            for meta in original_metadata:
                if meta['n_atoms'] <= n_max:
                    filtered_metadata.append(meta)
            
            dataset.metadata = filtered_metadata
            return dataset, len(original_metadata) - len(filtered_metadata)
        
        train_before = len(train_dataset.metadata)
        val_before = len(val_dataset.metadata)
        test_before = len(test_dataset.metadata)
        
        train_dataset, train_skipped = filter_by_size(train_dataset, args.n_max)
        val_dataset, val_skipped = filter_by_size(val_dataset, args.n_max)
        test_dataset, test_skipped = filter_by_size(test_dataset, args.n_max)
        
        print(f"  Train: {train_before} → {len(train_dataset.metadata)} "
              f"({100*len(train_dataset.metadata)/train_before:.1f}% kept, {train_skipped} filtered)")
        print(f"  Val:   {val_before} → {len(val_dataset.metadata)} "
              f"({100*len(val_dataset.metadata)/val_before:.1f}% kept, {val_skipped} filtered)")
        print(f"  Test:  {test_before} → {len(test_dataset.metadata)} "
              f"({100*len(test_dataset.metadata)/test_before:.1f}% kept, {test_skipped} filtered)")
    # ===============================================
    
    # Normalize energies
    stats = normalize_energies(train_dataset, val_dataset, test_dataset)
    
    # Save datasets
    save_datasets(train_dataset, val_dataset, test_dataset, args.output_dir, stats)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nProcessed data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review the sample data output above")
    print("  2. Train the CFM model using train_cfm.py")
    print("  3. Generate new nanoparticles using generate.py")


if __name__ == '__main__':
    main()
