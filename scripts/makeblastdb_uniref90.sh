#!/bin/bash
#SBATCH --job-name=makeblastdb_uniref90
#SBATCH --output=/scratch/amoldwin/logs/makeblastdb_uniref90_%j.out
#SBATCH --error=/scratch/amoldwin/logs/makeblastdb_uniref90_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=contrib
#SBATCH --qos=normal

set -euo pipefail

IN_FASTA="/scratch/amoldwin/datasets/uniref90.fasta"
OUT_PREFIX="/scratch/amoldwin/datasets/uniref90/uniref90"
LOCKDIR="/scratch/amoldwin/datasets/.makeblastdb_uniref90.lock"

mkdir -p /scratch/amoldwin/logs

# Activate env (adjust to your setup)
source ~/PROJECTS/miniconda/etc/profile.d/conda.sh
conda activate pilot

if [[ ! -s "$IN_FASTA" ]]; then
  echo "ERROR: missing/empty FASTA: $IN_FASTA" >&2
  exit 2
fi
if ! command -v makeblastdb >/dev/null 2>&1; then
  echo "ERROR: makeblastdb not on PATH (conda install -c bioconda blast)" >&2
  exit 3
fi

# Simple lock to prevent concurrent rebuilds
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another makeblastdb may be running." >&2
  exit 4
fi
trap 'rmdir "$LOCKDIR" || true' EXIT

echo "Building BLAST DB:"
echo "  IN : $IN_FASTA"
echo "  OUT: $OUT_PREFIX"
echo "  Host: $(hostname)"
echo "  Start: $(date)"

# (Optional) Clean stale lock files that can confuse BLAST
rm -f "${OUT_PREFIX}".*-lock || true

# Build directly to final prefix so internal alias references match
makeblastdb \
  -in "$IN_FASTA" \
  -dbtype prot \
  -parse_seqids \
  -out "$OUT_PREFIX" \
  -title "UniRef90"

echo "Done: $(date)"
ls -lh "${OUT_PREFIX}".* | sed -n '1,40p'