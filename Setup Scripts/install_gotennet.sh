#!/bin/bash
# install_gotennet.sh
# Sets up the `gotennet` conda environment for EONGen training, generation,
# and most evaluation scripts (evaluate.py, evaluate_re_phys.py, evaluate_geo_val.py).
#
# Tested on: CUDA 12.4, Ubuntu 22.04
# For other CUDA versions, change TORCH_CUDA below (e.g. cu118, cu121).
#
# Usage:
#   chmod +x install_gotennet.sh
#   ./install_gotennet.sh

set -e

ENV_NAME="gotennet"
PYTHON_VERSION="3.10"

# ── CUDA version for PyTorch ──────────────────────────────────────────────────
# PyG does not yet publish cu124 wheels; cu121 wheels are forward-compatible.
TORCH_CUDA="cu124"   # for torch install
PYG_CUDA="cu121"     # for PyG extension wheels

echo "======================================================"
echo "  Creating conda environment: ${ENV_NAME}"
echo "======================================================"

conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ""
echo "======================================================"
echo "  Installing PyTorch (CUDA ${TORCH_CUDA})"
echo "======================================================"
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo ""
echo "======================================================"
echo "  Installing PyTorch Geometric"
echo "======================================================"
pip install torch_geometric

# Strip '+cuXXX' suffix from torch version string for PyG wheel URL
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${PYG_CUDA}.html"
echo "PyG wheel URL: ${PYG_URL}"

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "${PYG_URL}"

echo ""
echo "======================================================"
echo "  Installing GoTenNet"
echo "======================================================"
pip install git+https://github.com/semodi/gotennet.git

echo ""
echo "======================================================"
echo "  Installing core dependencies"
echo "======================================================"
pip install \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    ase \
    dscribe \
    mendeleev \
    tensorboard \
    pyyaml

echo ""
echo "======================================================"
echo "  Verifying install"
echo "======================================================"
python -c "import torch; import torch_geometric; print(f'torch {torch.__version__}  |  pyg {torch_geometric.__version__}')"

echo ""
echo "======================================================"
echo "  Done! Activate with:"
echo "    conda activate ${ENV_NAME}"
echo "======================================================"
