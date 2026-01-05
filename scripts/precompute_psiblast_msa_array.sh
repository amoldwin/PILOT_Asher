#!/bin/bash
#SBATCH --job-name=pilot_psiblast_msa
#SBATCH --output=/scratch/amoldwin/logs/psiblast_msa_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/psiblast_msa_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02-00:00:00

# Submit with sbatch --array=0-$((N-1))%THROTTLE ...

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch ... precompute_psiblast_msa_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
FEATURE_DIR=${2:?Usage: sbatch ... precompute_psiblast_msa_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
SASA_BACKEND=${3:-freesasa}

# Adjust this path to your conda install (yours seems to be ~/PROJECTS/miniconda)
source /projects/ashehu/amoldwin/miniconda/etc/profile.d/conda.sh
conda activate pilot


FILTERED=$(mktemp)
trap 'rm -f "${FILTERED}"' EXIT
grep -vE '^\s*(#|$)' "${MUT_LIST}" > "${FILTERED}"
LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${FILTERED}")

echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "LINE=${LINE}"

if [[ -z "${LINE}" ]]; then
  echo "No usable line for index ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi

python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s precompute_psiblast_msa \
  --sasa-backend "${SASA_BACKEND}" \
  --freesasa-path freesasa \
  --mutator-backend proxy