#!/bin/bash
# install_mace_eval.sh
# Sets up the `mace_eval` conda environment for MACE-based evaluation:
#   - evaluate_re_lat.py  (SOAP Wasserstein + MACE cosine distance + PCA)
#   - evaluate_mlff.py    (RMSD_relax, Hit Rate H0.5, Basin C_rank)
#
# Kept separate from `gotennet` to avoid PyTorch version conflicts.
#
# Usage:
#   chmod +x install_mace_eval.sh
#   ./install_mace_eval.sh

set -e

ENV_NAME="mace_eval"
PYTHON_VERSION="3.10"

echo "======================================================"
echo "  Creating conda environment: ${ENV_NAME}"
echo "======================================================"

conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ""
echo "======================================================"
echo "  Installing PyTorch (CUDA 12.1)"
echo "======================================================"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "======================================================"
echo "  Installing MACE"
echo "======================================================"
pip install mace-torch

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
    mendeleev

echo ""
echo "======================================================"
echo "  Verifying MACE install"
echo "======================================================"
python -c "from mace.calculators import mace_mp; print('MACE import OK')"

echo ""
echo "======================================================"
echo "  Done! Activate with:"
echo "    conda activate ${ENV_NAME}"
echo "======================================================"
