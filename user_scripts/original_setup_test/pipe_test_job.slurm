#!/bin/bash -l
#SBATCH -M wice
#SBATCH -A intro_vsc38504
#SBATCH --job-name=test_bioml_rag
#SBATCH --output=/data/leuven/385/vsc38504/thesis/llms4om/LLMs4OM/logs/test_bioml_%j.out
#SBATCH --error=/data/leuven/385/vsc38504/thesis/llms4om/LLMs4OM/logs/test_bioml_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1

set -euo pipefail

# conda activation in batch
source /data/leuven/385/vsc38504/miniconda3/etc/profile.d/conda.sh
conda activate llms4om

# put HF caches on scratch (faster, avoids home quota)
export HF_HOME="$VSC_SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONPYCACHEPREFIX="$VSC_SCRATCH/pycache"
mkdir -p "$TRANSFORMERS_CACHE" "$PYTHONPYCACHEPREFIX"

# Optional: Log job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "GPU info:"
nvidia-smi

cd $VSC_DATA/thesis/llms4om/LLMs4OM

python -u user_scripts/original_setup_test/test_bioml_rag.py

echo "Job finished at $(date)"