#!/bin/bash
#SBATCH --job-name=pilot_assemble
#SBATCH --output=/scratch/amoldwin/logs/assemble_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/assemble_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Adjust these to your CPU partition/qos
#SBATCH --partition=contrib
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01-04:00:00

# NOTE:
#  - Do NOT hardcode --array here.
#  - Pass --array at submission time using the N argument.
#
# Usage:
#   sbatch --array=0-$((N-1))%MAX_PARALLEL scripts/assemble_array.sh MUT_LIST FEATURE_DIR N [MAX_PARALLEL]
#sbatch --dependency=afterany:5289062 --array=0-$((4917-1))%50  scripts/assemble_array.sh dataset/mutation_list_dTm.txt /scratch/amoldwin/datasets/PILOT_dTm_esmfold 4917 50
#
# Example:
#   N=$(grep -vE '^\s*(#|$)' dataset/mutation_list_filtered.txt | wc -l)
#   sbatch --array=0-$((N-1))%50 scripts/assemble_array.sh dataset/mutation_list_filtered.txt /scratch/.../PILOT_dTm_esmfold "$N" 50

set -euo pipefail

MUT_LIST=${1:?Usage: sbatch --array=0-\$((N-1))%50 scripts/assemble_array.sh MUT_LIST FEATURE_DIR N [MAX_PARALLEL]}
FEATURE_DIR=${2:?Usage: sbatch --array=0-\$((N-1))%50 scripts/assemble_array.sh MUT_LIST FEATURE_DIR N [MAX_PARALLEL]}
N=${3:?Usage: sbatch --array=0-\$((N-1))%50 scripts/assemble_array.sh MUT_LIST FEATURE_DIR N [MAX_PARALLEL]}
MAX_PARALLEL=${4:-}

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -le 0 ]]; then
  echo "ERROR: N must be a positive integer, got: '$N'" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: This script must be run as a SLURM array job." >&2
  echo "Submit like:" >&2
  if [[ -n "$MAX_PARALLEL" ]]; then
    echo "  sbatch --array=0-$((N-1))%${MAX_PARALLEL} $0 $MUT_LIST $FEATURE_DIR $N $MAX_PARALLEL" >&2
  else
    echo "  sbatch --array=0-$((N-1))%50 $0 $MUT_LIST $FEATURE_DIR $N 50" >&2
  fi
  exit 3
fi

# Guard: don't exceed N
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]]; then
  echo "Array task id ${SLURM_ARRAY_TASK_ID} >= N=${N}; exiting." >&2
  exit 0
fi

# Robust conda activation (batch-safe)
source ~/PROJECTS/miniconda/etc/profile.d/conda.sh
conda activate pilot

export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30
export PATH="/projects/ashehu/amoldwin/software/msms/bin:$PATH"

# Filter blank/comment lines so indexing is stable
FILTERED=$(mktemp)
trap 'rm -f "${FILTERED}"' EXIT
grep -vE '^\s*(#|$)' "${MUT_LIST}" > "${FILTERED}"

N_FILTERED=$(wc -l < "${FILTERED}")
if [[ "$N_FILTERED" -lt "$N" ]]; then
  echo "WARNING: filtered mutation list has fewer lines than N (filtered=$N_FILTERED N=$N)." >&2
  echo "         You probably want N=$N_FILTERED." >&2
fi

LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${FILTERED}")
echo "Assemble idx=${SLURM_ARRAY_TASK_ID}/${N} host=$(hostname)"
echo "Assemble: ${LINE}"

if [[ -z "${LINE}" ]]; then
  echo "No usable line for idx=${SLURM_ARRAY_TASK_ID}; exiting." >&2
  exit 0
fi

# Assemble expects ESM2 .pt present; will compute DSSP/depth and parse/collate to .npy
python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s assemble \
  --mutator-backend rosetta \
  --rosetta-scripts-path rosetta_scripts.static.linuxgccrelease \
  --skip-pdb-download \
  --row-pdb-name-mode pdb_chain \
  --continue-on-error