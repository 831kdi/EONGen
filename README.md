# EONGen
**Energy-Optimised Nanocluster Generative Model**

EONGen is a Conditional Flow Matching (CFM) framework for generating stable atomic nanocluster structures. It couples a [GoTenNet](https://github.com/semodi/gotennet) E(3)-equivariant encoder to a CFM objective, with Kabsch+Hungarian symmetry-aware training and element/size conditioning.

---

## Repository Structure

```
EONGen/
├── Data_Processing/
│   ├── split_xyz_ground_states.py   # Split XYZ dataset into train/val/test
│   ├── preprocess_xyz_to_pyg.py     # Convert split XYZ → PyTorch Geometric (.pkl)
│   └── generate_element_groups.py   # Hierarchical clustering of element correlations
│
├── Training Scripts/
│   └── train_cfm.py                 # CFM training with GoTenNet backbone
│
├── Generation Scripts/
│   ├── generate.py                  # Standard Euler/RK4 ODE generation
│   └── fk_gen.py                    # Feynman-Kac steered generation (CN steering)
│
├── Evaluation Scripts/
│   ├── evaluate.py                  # Quick Re_Phys evaluation (RMSD, Chamfer, RoG)
│   ├── evaluate_re_phys.py          # Re_Phys: RMSD + RoG, size/element breakdown
│   ├── evaluate_geo_val.py          # Gen_Geo: steric clashes + radius of gyration spread
│   ├── evaluate_re_lat.py           # Re_Lat: SOAP Wasserstein + MACE cosine + PCA
│   └── evaluate_mlff.py             # Gen_Energy: RMSD_relax, Hit Rate H₀.₅, C_rank
│
├── Setup Scripts/
│   ├── install_gotennet.sh          # gotennet conda environment (training/generation/eval)
│   └── install_mace_eval.sh         # mace_eval conda environment (Re_Lat + Gen_Energy)
│
└── README.md
```

---

## Environments

EONGen uses **two separate conda environments** to avoid dependency conflicts between PyTorch Geometric (GoTenNet) and MACE.

| Environment | Used for |
|---|---|
| `gotennet` | Data processing, training, generation, `evaluate.py`, `evaluate_re_phys.py`, `evaluate_geo_val.py` |
| `mace_eval` | `evaluate_re_lat.py`, `evaluate_mlff.py` |

### Install `gotennet`
```bash
chmod +x Setup\ Scripts/install_gotennet.sh
./Setup\ Scripts/install_gotennet.sh
conda activate gotennet
```

### Install `mace_eval`
```bash
chmod +x Setup\ Scripts/install_mace_eval.sh
./Setup\ Scripts/install_mace_eval.sh
conda activate mace_eval
```

---

## Pipeline

### Step 1 — Data Splitting

Separates ground states (E_rel = 0) from excited states, then splits ground states 8:1:1 across three strategies. Excited states are always added to the training set.

```bash
conda activate gotennet
python Data_Processing/split_xyz_ground_states.py \
    --input clusters_full.xyz \
    --output_dir data/splits \
    --seed 42
```

**Output:** Three strategy directories under `data/splits/`:
```
strategy1_random/          # random 8:1:1
strategy2_size/            # sorted by n_atoms
strategy3_element_groups/  # stratified by element
```
Each contains `train.xyz`, `val.xyz`, `test.xyz`.

---

### Step 2 — Preprocess to PyTorch Geometric

Converts XYZ files to `NanoparticleDataset` objects (`.pkl`) with centred positions, atomic numbers, and E_rel/atom labels. `edge_index` is **not** stored — GoTenNet builds it at training time.

```bash
python Data_Processing/preprocess_xyz_to_pyg.py \
    --train_xyz data/splits/strategy1_random/train.xyz \
    --val_xyz   data/splits/strategy1_random/val.xyz \
    --test_xyz  data/splits/strategy1_random/test.xyz \
    --output_dir processed_data/
```

Optional: filter by cluster size with `--n_max 20`.

**Output:** `processed_data/{train,val,test}_dataset.pkl`, `statistics.pkl`

---

### Step 3 — Training

Trains the CFM model with cosine annealing LR schedule and Kabsch+Hungarian symmetry correction. Default hyperparameters (confirmed optimal): `hidden_dim=256`, `n_interactions=4`, `cutoff=5.0 Å`, `energy_cutoff=0.03 eV/atom`.

```bash
python Training\ Scripts/train_cfm.py \
    --train_data processed_data/train_dataset.pkl \
    --val_data   processed_data/val_dataset.pkl \
    --output_dir checkpoints/ \
    --hidden_dim 256 \
    --n_interactions 4 \
    --cutoff 5.0 \
    --energy_cutoff 0.03 \
    --epochs 200
```

**Output:** `checkpoints/best_model.pt`, `hyperparameters.json`

---

### Step 4 — Generation

#### Standard generation (Euler/RK4 ODE)

```bash
python Generation\ Scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data  processed_data/test_dataset.pkl \
    --output_dir results/ \
    --n_samples  100 \
    --n_steps    100 \
    --integrator euler \
    --device     cuda:0
```

**Output:** `results/generated_all.xyz`, `results/target_all.xyz`, `results/best/`, `results/best_summary.json`

#### Feynman-Kac steered generation

Steers generation toward high or low coordination number (CN) regimes using SMC reweighting at inference time — no retraining required. GoTenNet does not support multi-graph batching, so particles are evaluated sequentially; use `--max_targets` to limit runtime.

```bash
# High CN (compact/bulk-like structures)
python Generation\ Scripts/fk_gen.py \
    --checkpoint  checkpoints/best_model.pt \
    --test_data   processed_data/test_dataset.pkl \
    --output_dir  results_fk/high_cn \
    --mode        high_cn \
    --gamma       1.0 \
    --n_particles 8 \
    --n_samples   10 \
    --max_targets 20 \
    --device      cuda:0

# Low CN (open/under-coordinated structures)
python Generation\ Scripts/fk_gen.py \
    --checkpoint  checkpoints/best_model.pt \
    --test_data   processed_data/test_dataset.pkl \
    --output_dir  results_fk/low_cn \
    --mode        low_cn \
    --gamma       1.0 \
    --lambda_conn 5.0 \
    --n_particles 8 \
    --n_samples   10 \
    --max_targets 20 \
    --device      cuda:0
```

**Output (per target):** `{elem}_{n}/sample_XXXX.xyz`, `fk_summary.json`, `cn_distribution.png`

---

### Step 5 — Evaluation

Five benchmark categories. Run scripts in the appropriate environment.

#### `gotennet` environment

**Quick Re_Phys check** (faster, no size/element breakdown):
```bash
python Evaluation\ Scripts/evaluate.py \
    --generated results/generated_all.xyz \
    --target    results/target_all.xyz \
    --output    results/evaluation_detailed.json
```

**Re_Phys** (RMSD, RoG — size + element breakdown):
```bash
python Evaluation\ Scripts/evaluate_re_phys.py \
    --generated results/generated_all.xyz \
    --target    results/target_all.xyz \
    --output    results/re_phys/
```

**Gen_Geo** (steric clashes, ΔR_g spread):
```bash
python Evaluation\ Scripts/evaluate_geo_val.py \
    --generated results/generated_all.xyz \
    --target    results/target_all.xyz \
    --output    results/geo_val/
```

#### `mace_eval` environment

**Re_Lat** (SOAP Wasserstein distance, MACE cosine distance, PCA):
```bash
conda activate mace_eval
python Evaluation\ Scripts/evaluate_re_lat.py \
    --generated  results/generated_all.xyz \
    --target     results/target_all.xyz \
    --corr_csv   Data_Processing/correlations.csv \
    --output     results/re_lat/ \
    --use_mace \
    --mace_device cuda
```

**Gen_Energy** (RMSD_relax, Hit Rate H₀.₅, Basin Population-Energy Consistency C_rank):
```bash
python Evaluation\ Scripts/evaluate_mlff.py \
    --generated results/generated_all.xyz \
    --output    results/mlff/ \
    --device    cuda
```
  note    = {Under review}
}
```
