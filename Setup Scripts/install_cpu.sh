#!/bin/bash
# Installation script for CPU-only environment (for deployment)

echo "Creating conda environment for cluster CFM (CPU)..."
conda create -n cluster_cfm_cpu python=3.11 -y
source activate cluster_cfm_cpu

echo "Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing PyTorch Geometric (CPU)..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

echo "Installing other dependencies..."
pip install pytorch-lightning hydra-core omegaconf
pip install torchcfm pandas numpy ase scipy tqdm matplotlib seaborn scikit-learn openpyxl

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo "Installation complete! Activate with: conda activate cluster_cfm_cpu"
