#!/bin/bash
#SBATCH -p p.cajal
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# Generic one-node Slurm wrapper around rerun/run.sh:
#   sbatch -J <name> --export=ALL,BIO_ANALYSIS_PREP=<dir>,... rerun/sbatch_run.sh general/connectivity_fraction_per_ct.py
#   sbatch -J <name> --export=ALL,... rerun/sbatch_run.sh -m bio_analysis.rerun.build_synapse_table
# Submit from the repository root (SLURM_SUBMIT_DIR is used to find it; sbatch copies the script to the spool dir).
mkdir -p logs
REPO=${BIO_ANALYSIS_REPO:-$SLURM_SUBMIT_DIR}
echo "host $(hostname), prep ${BIO_ANALYSIS_PREP:-default}, results ${BIO_ANALYSIS_RESULTS:-default}, args: $*"
exec "$REPO/rerun/run.sh" "$@"
