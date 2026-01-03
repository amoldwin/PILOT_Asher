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
#SBATCH --mem=8G
#SBATCH --time=02-00:00:00

# IMPORTANT:
# Set this to number_of_variants-1 and optionally throttle concurrency with %.
# Example for 5000 variants: 0-4999%200
#SBATCH --array=0-4999%200

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch precompute_psiblast_msa_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
FEATURE_DIR=${2:?Usage: sbatch precompute_psiblast_msa_array.sh MUT_LIST FEATURE_DIR [SASA_BACKEND]}
SASA_BACKEND=${3:-freesasa}

source ../miniconda/bin/activate
conda activate pilot

# hhblits DB not needed in this stage, but harmless to export:
export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30/UniRef30_2021_03

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