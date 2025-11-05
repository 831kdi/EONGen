# EONGen
Energy-Optimized Nanocluster (EON) Generative Model by Conditional Flow Matching
# Project Setup Summary

### What We've Built

This model is for building a Conditional Flow Matching (CFM) model to generate stable atomic cluster configurations.

#### 1. **Environment Setup**
- **GPU Installation** (`install_cuda124.sh`): Automatic setup for CUDA 12.4 environment
- **CPU Installation** (`install_cpu.sh`): CPU-only environment for deployment
- **Dependencies** (`requirements.txt`): All required libraries listed

#### 2. **Data Processing Pipeline**
- **Data Utilities** (`src/utils/data_utils.py`):
  - XYZ string parsing
  - Position centering and normalization
  - Element-to-atomic-number conversion
  
- **PyTorch Dataset** (`src/data/dataset.py`):
  - `ClusterDataset` class for loading cluster data
  - Custom `collate_fn` for variable-sized clusters
  - Automatic filtering by element and size
  
- **Data Splitting** (`src/data/split_data.py`):
  - **Strategy 1**: Random split (8:1:1)
  - **Strategy 2**: Size-based split (small→train, large→test)
  - **Strategy 3**: Correlation-based split (similar structures grouped)

#### 3. **Documentation**
- **Main README**: Complete project guide with examples
- **Exploration Guide** (`notebooks/data_exploration.md`): Step-by-step data analysis

---

## Immediate Action Items

### Step 1: Download and Setup
```bash
# Download the project
# Extract to your working directory
cd cluster_cfm
```

### Step 2: Place Your Data
```bash
# Put your full dataset (61,690 structures) here:
# data/cluster_data_full.csv

# Required CSV columns:
# - cluster_id
# - element_symbol
# - n_atoms
# - structure_xyz
# - energy_relative
# - similar_structures (optional)
```

### Step 3: Install Environment
```bash
# For GPU (CUDA 12.4)
chmod +x install_cuda124.sh
./install_cuda124.sh
conda activate cluster_cfm

# Or for CPU only
chmod +x install_cpu.sh
./install_cpu.sh
conda activate cluster_cfm_cpu
```

### Step 4: Verify Installation
```bash
# Test data parsing
python src/utils/data_utils.py

# Test imports
python -c "from src.data.dataset import ClusterDataset; print('Success!')"
```

### Step 5: Split Your Data
```bash
# Edit scripts/split_data.sh to set your data path
# Then run:
chmod +x scripts/split_data.sh
./scripts/split_data.sh
```

This will create 9 CSV files in `data/splits/`:
- `cluster_data_random_{train,val,test}.csv`
- `cluster_data_size_{train,val,test}.csv`
- `cluster_data_correlation_{train,val,test}.csv`

---

