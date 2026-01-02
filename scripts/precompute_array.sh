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
#SBATCH --array=0-15

# Usage: sbatch slurm_precompute_array.sh ./mutation_list.txt /features naccess
#        sbatch slurm_precompute_array.sh ./mutation_list.txt /features freesasa
MUT_LIST=$1        # e.g., ./mutation_list.txt
FEATURE_DIR=$2     # e.g., /projects/ashehu/amoldwin/features
SASA_BACKEND=${3:-naccess}   # naccess or freesasa

source ../miniconda/bin/activate
conda activate pilot

# HHblits database location for your cluster
export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30
UNIREF30_DIR=/scratch/amoldwin/datasets/Uniref30/UniRef30_2021_03

# Optional: BLAST DB (if needed by your setup)
# export BLASTDB=/scratch/amoldwin/datasets/Uniref90

# Pick the line by 0-based array index
LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${MUT_LIST}")
echo "Precompute: ${LINE}"

# Run CPU-heavy precompute (SASA + PSI-BLAST + MSA + HHblits)
# Ensure gen_features.py has uniRef30_path set to ${UNIREF30_DIR}
python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s precompute \
  --sasa-backend "${SASA_BACKEND}" \
  --freesasa-path freesasa