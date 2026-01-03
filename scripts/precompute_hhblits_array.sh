#!/bin/bash
#SBATCH --job-name=pilot_hhblits
#SBATCH --output=/scratch/amoldwin/logs/hhblits_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/hhblits_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=02-00:00:00

# Example for 5000 variants: 0-4999%400
#SBATCH --array=0-4999%400

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch precompute_hhblits_array.sh MUT_LIST FEATURE_DIR}
FEATURE_DIR=${2:?Usage: sbatch precompute_hhblits_array.sh MUT_LIST FEATURE_DIR}

source ../miniconda/bin/activate
conda activate pilot

# REQUIRED for hhblits:
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
  -s precompute_hhblits \
  --mutator-backend proxy