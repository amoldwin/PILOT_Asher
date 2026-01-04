#!/bin/bash
set -euo pipefail
## bash scripts/submit_precompute_split.sh dataset/mutation_list.txt /scratch/amoldwin/datasets/PILOT freesasa

MUT_LIST=${1:?Usage: ./scripts/submit_precompute_split.sh MUT_LIST FEATURE_DIR [SASA_BACKEND] [PSI_THROTTLE] [HH_THROTTLE]}
FEATURE_DIR=${2:?Usage: ./scripts/submit_precompute_split.sh MUT_LIST FEATURE_DIR [SASA_BACKEND] [PSI_THROTTLE] [HH_THROTTLE]}
SASA_BACKEND=${3:-freesasa}
PSI_THROTTLE=${4:-150}
HH_THROTTLE=${5:-300}

N=$(grep -vE '^\s*(#|$)' "$MUT_LIST" | wc -l)
if [[ "$N" -le 0 ]]; then
  echo "ERROR: no usable lines found in $MUT_LIST" >&2
  exit 2
fi
LAST=$((N-1))

echo "Usable lines: N=$N => array 0-$LAST"
echo "Throttles: psiblast_msa=%${PSI_THROTTLE}, hhblits=%${HH_THROTTLE}"

PSIJOB=$(sbatch --array=0-${LAST}%${PSI_THROTTLE} \
  scripts/precompute_psiblast_msa_array.sh "$MUT_LIST" "$FEATURE_DIR" "$SASA_BACKEND" | awk '{print $4}')

HHJOB=$(sbatch --dependency=afterok:${PSIJOB} --array=0-${LAST}%${HH_THROTTLE} \
  scripts/precompute_hhblits_array.sh "$MUT_LIST" "$FEATURE_DIR" | awk '{print $4}')

echo "Submitted: psiblast+msa=${PSIJOB}"
echo "Submitted: hhblits-only=${HHJOB} (afterok on ${PSIJOB})"