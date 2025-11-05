#!/bin/bash
# Installation script for CUDA 12.4 environment

echo "Creating conda environment for cluster CFM..."
conda create -n cluster_cfm python=3.11 -y
source activate cluster_cfm

echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyTorch Geometric with CUDA 12.4..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

echo "Installing other dependencies..."
pip install pytorch-lightning hydra-core omegaconf
pip install torchcfm pandas numpy ase scipy tqdm matplotlib seaborn scikit-learn openpyxl

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"

echo "Installation complete! Activate with: conda activate cluster_cfm"
