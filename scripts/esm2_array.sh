#!/bin/bash
#SBATCH --job-name=pilot_esm2
#SBATCH --output=/scratch/amoldwin/logs/esm2_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/esm2_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --time=01-00:30:00
# Do NOT set --array here; pass via sbatch --array=0-9

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch --array=0-9 scripts/esm2_array.sh MUT_LIST FEATURE_DIR}
FEATURE_DIR=${2:?Usage: sbatch --array=0-9 scripts/esm2_array.sh MUT_LIST FEATURE_DIR}

# Robust conda activation (batch-safe)
source ~/PROJECTS/miniconda/etc/profile.d/conda.sh
conda activate pilot

# Cache torch hub downloads
export TORCH_HOME=/scratch/amoldwin/torch_cache

TASK_ID=${SLURM_ARRAY_TASK_ID}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-10}

# Filter blank/comment lines so indexing is stable
FILTERED=$(mktemp)
trap 'rm -f "${FILTERED}"' EXIT
grep -vE '^\s*(#|$)' "${MUT_LIST}" > "${FILTERED}"

N=$(wc -l < "${FILTERED}")
echo "ESM2 chunk runner: task=${TASK_ID}/${NTASKS} total_lines=${N} host=$(hostname)"

# Process every NTASKS-th line starting at TASK_ID (0-based)
for ((i=TASK_ID; i<N; i+=NTASKS)); do
  LINE=$(awk -v idx="$i" 'NR==idx+1{print;exit}' "${FILTERED}")
  if [[ -z "${LINE}" ]]; then
    continue
  fi
  echo "i=${i} LINE=${LINE}"

  python gen_features.py \
    -i <(echo "${LINE}") \
    -d "${FEATURE_DIR}" \
    -s esm2
done