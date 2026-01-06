#!/bin/bash
#SBATCH --job-name=PILOT
#SBATCH --output=/scratch/amoldwin/logs/%j.out
#SBATCH --error=/scratch/amoldwin/logs/%j.err
#SBATCH --mail-user=<amoldwin@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=normal        # contrib-gpuq, gpuq
#SBATCH --nodes=1 
#SBATCH --mem=16G 
#SBATCH --time=03-00:30:00

##cd /scratch/amoldwin/datasets
##wget https://wwwuser.gwdguser.de/~compbiol/uniclust/2021_03/UniRef30_2021_03.tar.gz
##tar -xzf UniRef30_2021_03.tar.gz
##export HHBLITS_DB=/path/to/uniclust30_2021_03



source ../miniconda/bin/activate
conda activate pilot

# example
##makeblastdb -in /scratch/amoldwin/datasets/uniref90.fasta \
##  -dbtype prot \
##  -parse_seqids \
##  -out /scratch/amoldwin/datasets/uniref90

## python gen_features.py \
##   -i dataset/mutation_list_missing.txt \
##   -d /scratch/amoldwin/datasets/PILOT \
##   -s precompute_psiblast_msa \
##   --sasa-backend freesasa \
##   --mutator-backend proxy

python import_geostab_files.py \
  --source /projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/ \
  --feature-dir /scratch/amoldwin/datasets/PILOT_dTm \
  --overwrite
  
