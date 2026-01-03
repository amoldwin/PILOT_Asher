#!/bin/bash
# Usage:
#   ./submit_precompute_split.sh ./mutation_list.txt /scratch/amoldwin/datasets/PILOT freesasa
set -euo pipefail

MUT_LIST=${1:?mutation list required}
FEATURE_DIR=${2:?feature dir required}
SASA_BACKEND=${3:-freesasa}

PSIJOB=$(sbatch scripts/precompute_psiblast_msa_array.sh "${MUT_LIST}" "${FEATURE_DIR}" "${SASA_BACKEND}" | awk '{print $4}')
HHJOB=$(sbatch --dependency=afterok:${PSIJOB} scripts/precompute_hhblits_array.sh "${MUT_LIST}" "${FEATURE_DIR}" | awk '{print $4}')

echo "Submitted: psiblast+msa=${PSIJOB}, hhblits=${HHJOB}"