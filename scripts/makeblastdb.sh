#!/bin/bash
#SBATCH --job-name=makeblastdb_uniref90
#SBATCH --output=/scratch/amoldwin/logs/makeblastdb_uniref90_%j.out
#SBATCH --error=/scratch/amoldwin/logs/makeblastdb_uniref90_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
## If your cluster requires it, uncomment and set appropriately:
#SBATCH --partition=contrib
#SBATCH --qos=normal

set -euo pipefail

IN_FASTA="/scratch/amoldwin/datasets/uniref90.fasta"
OUT_PREFIX="/scratch/amoldwin/datasets/uniref90"

mkdir -p /scratch/amoldwin/logs
mkdir -p "$(dirname "$OUT_PREFIX")"

# Activate your environment (adjust if your conda init differs)
source ../miniconda/bin/activate
conda activate pilot


# Sanity checks
echo "Host: $(hostname)"
echo "CWD:  $(pwd)"
echo "IN_FASTA:   $IN_FASTA"
echo "OUT_PREFIX: $OUT_PREFIX"
echo "CPUS: ${SLURM_CPUS_PER_TASK:-1}"
echo

if [[ ! -s "$IN_FASTA" ]]; then
  echo "ERROR: input FASTA not found or empty: $IN_FASTA" >&2
  exit 2
fi

if ! command -v makeblastdb >/dev/null 2>&1; then
  echo "ERROR: makeblastdb not found on PATH. Install BLAST+ (e.g., conda install -c bioconda blast)." >&2
  exit 3
fi

# Optional: avoid partial DB on failure
TMP_PREFIX="${OUT_PREFIX}.tmp_${SLURM_JOB_ID}"
trap 'echo "Cleaning up tmp db..."; rm -f ${TMP_PREFIX}.*' EXIT

echo "Running makeblastdb..."
makeblastdb \
  -in "$IN_FASTA" \
  -dbtype prot \
  -parse_seqids \
  -out "$TMP_PREFIX" \
  -title "UniRef90"

echo "Renaming tmp db -> final prefix..."
# Move all created files to final prefix basename
for f in "${TMP_PREFIX}".*; do
  [[ -e "$f" ]] || continue
  mv -f "$f" "${OUT_PREFIX}.${f##*.}"
done

# Disable cleanup trap now that we moved files
trap - EXIT

echo "Done. Created files:"
ls -lh "${OUT_PREFIX}".*