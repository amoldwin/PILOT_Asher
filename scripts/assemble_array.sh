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
#SBATCH --array=0-15

# Usage: sbatch slurm_assemble_array.sh dataset./mutation_list_filtered.txt /features
MUT_LIST=$1
FEATURE_DIR=$2

source /projects/ashehu/amoldwin/miniconda/etc/profile.d/conda.sh
conda activate pilot

export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30
export PATH="/projects/ashehu/amoldwin/software/msms/bin:$PATH"



LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${MUT_LIST}")
echo "Assemble: ${LINE}"

# Assemble expects ESM2 .pt present; will compute DSSP/depth and parse/collate to .npy
python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s assemble