#!/bin/bash
#SBATCH --job-name=pilot_esm2
#SBATCH --output=/scratch/amoldwin/logs/esm2_%A_%a.out
#SBATCH --error=/scratch/amoldwin/logs/esm2_%A_%a.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=128G
#SBATCH --time=01-00:30:00
#SBATCH --array=0-10

# Usage: sbatch slurm_esm2_array.sh ./mutation_list.txt /features
MUT_LIST=$1
FEATURE_DIR=$2

source ../miniconda/bin/activate
conda activate pilot
export HHBLITS_DB=/scratch/amoldwin/datasets/Uniref30

# Optional: set a local torch hub cache if you modify use_esm2.py to respect TORCH_HOME
# export TORCH_HOME=/scratch/amoldwin/torch_cache

LINE=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+1{print;exit}' "${MUT_LIST}")
echo "ESM2: ${LINE}"

python gen_features.py \
  -i <(echo "${LINE}") \
  -d "${FEATURE_DIR}" \
  -s esm2