#!/bin/bash
# Submit the full pipeline with dependencies
# Usage: ./submit_pipeline.sh ./mutation_list.txt /features [sasa_backend]
MUT_LIST=$1
FEATURE_DIR=$2
SASA_BACKEND=${3:-naccess}

PREJOB=$(sbatch slurm_precompute_array.sh "${MUT_LIST}" "${FEATURE_DIR}" "${SASA_BACKEND}" | awk '{print $4}')
ESMJOB=$(sbatch --dependency=afterok:${PREJOB} slurm_esm2_array.sh "${MUT_LIST}" "${FEATURE_DIR}" | awk '{print $4}')
sbatch --dependency=afterok:${PREJOB}:${ESMJOB} slurm_assemble_array.sh "${MUT_LIST}" "${FEATURE_DIR}"
echo "Submitted jobs: precompute=${PREJOB}, esm2=${ESMJOB}"