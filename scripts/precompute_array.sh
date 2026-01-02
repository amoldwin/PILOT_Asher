#!/bin/bash
#SBATCH --job-name=pilot_precompute
#SBATCH --output=/scratch/amoldwin/logs/precompute_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/precompute_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Adjust these to your CPU partition/qos
#SBATCH --partition=contrib
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=01-12:00:00
#SBATCH --array=0-30

# Usage: sbatch slurm_precompute_array.sh ./mutation_list.txt /features naccess
#        sbatch slurm_precompute_array.sh ./mutation_list.txt /features freesasa

##tail -n +2 dataset/training_set.txt | awk -F'\t' '{print $1, $2, $3, $4}' > dataset/mutation_list_train.txt
##tail -n +2 dataset/test_set.txt | awk -F'\t' '{print $1, $2, $3, $4}' > dataset/mutation_list_test.txt

## rm /features/sasa/{pdb}{chain}{WT}{pos}{Mut}.rsa /features/sasa/{pdb}{chain}{WT}{pos}{Mut}.asa


MUT_LIST=dataset/mutation_list_test.txt
FEATURE_DIR=/scratch/amoldwin/datasets/PILOT
SASA_BACKEND=freesasa

source ../miniconda/bin/activate
conda activate pilot
export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30
UNIREF30_DIR=/scratch/amoldwin/datasets/Uniref30/UniRef30_2021_03

# Select line after filtering out blank/comment lines (0-based index)
FILTERED=$(mktemp)
grep -vE '^\s*(#|$)' "${MUT_LIST}" > "${FILTERED}"

LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${FILTERED}")

echo "Selected index=${SLURM_ARRAY_TASK_ID}"
echo "LINE(raw)=${LINE}"
printf 'LINE(hex)='; printf '%s' "$LINE" | hexdump -C

if [[ -z "${LINE}" ]]; then
  echo "No usable line for index ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi

python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s precompute \
  --sasa-backend "${SASA_BACKEND}" \
  --freesasa-path freesasa \
  --mutator-backend proxy