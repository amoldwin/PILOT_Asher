#!/bin/bash
#SBATCH --job-name=pilot_precompute_rosetta
#SBATCH --output=/scratch/amoldwin/logs/precompute_rosetta_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/precompute_rosetta_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01-12:00:00

# IMPORTANT:
# Do NOT hardcode --array here; set it at submit-time:
#   sbatch --array=0-999 scripts/rosetta_precompute_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch --array=0-N scripts/rosetta_precompute_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
FEATURE_DIR=${2:?Usage: sbatch --array=0-N scripts/rosetta_precompute_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
SASA_BACKEND=${3:-freesasa}

# --- environment ---
source /projects/ashehu/amoldwin/miniconda/etc/profile.d/conda.sh
conda activate pilot

# Rosetta module (Hopper prerequisites + rosetta)
module purge
module load hosts/hopper gnu9/9.3.0 openmpi4/4.0.4
module load rosetta/3.13

# Optional (if you use MSMS for depth; gen_features.py already warns/falls back if missing)
export PATH="/projects/ashehu/amoldwin/software/msms/bin:$PATH"

# HHblits DB (adjust to your actual path)
export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30

# --- select a single mutation line (robust against blanks/comments) ---
TASK_ID=${SLURM_ARRAY_TASK_ID}
FILTERED=$(mktemp)
trap 'rm -f "${FILTERED}"' EXIT

grep -vE '^\s*(#|$)' "${MUT_LIST}" > "${FILTERED}"

N=$(wc -l < "${FILTERED}")
if [[ "${TASK_ID}" -ge "${N}" ]]; then
  echo "Task ${TASK_ID} out of range (N=${N}); exiting."
  exit 0
fi

LINE=$(awk -v idx="${TASK_ID}" 'NR==idx+1{print;exit}' "${FILTERED}")
echo "Selected idx=${TASK_ID}/${N} line=${LINE}"

# --- run PILOT feature generation ---
python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s precompute \
  --sasa-backend "${SASA_BACKEND}" \
  --mutator-backend rosetta \
  --rosetta-scripts-path rosetta_scripts.static.linuxgccrelease