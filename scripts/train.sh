#!/bin/bash
#SBATCH --job-name=pilot_esm2
#SBATCH --output=/scratch/amoldwin/logs/train_%A.out
#SBATCH --error=/scratch/amoldwin/logs/train_%A.err
#SBATCH --mail-user=amoldwin@gmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:1
##gpu:A100.80gb:1
#SBATCH --mem=16G
#SBATCH --time=01-00:30:00
# Do NOT set --array here; pass via sbatch --array=0-9

source ~/PROJECTS/miniconda/etc/profile.d/conda.sh
conda activate pilot
export TORCH_HOME=/scratch/amoldwin/torch_cache


## python train.py --train dataset/training_set_filtered.txt \
##   --test dataset/test_set.txt \
##   --feature-dir /scratch/amoldwin/datasets/PILOT \
##   --job-id ddg_$SLURM_JOB_ID \
##   --seed 123 \
##   --epochs 30 \
##   --lr 1e-4 \
##   --out-dir runs

python train.py --train dataset/training_set_filtered.txt \
  --test dataset/test_set.txt \
  --feature-dir /scratch/amoldwin/datasets/PILOT \
  --job-id ddg_$SLURM_JOB_ID \
  --seed 123 \
  --epochs 30 \
  --lr 1e-4 \
  --out-dir runs \
  --mutator-backend rosetta \
  --label-col rosetta