#!/bin/bash
#SBATCH -J percell
#SBATCH -p p.cajal
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=500G
#SBATCH -t 06:00:00
#SBATCH -o logs/percell_%j.out
# One-node build of the synapse cache and the per-cell morphology table / prep cache (rerun/build_percell_table.py).
#   sbatch --export=ALL,BIO_ANALYSIS_PREP=<dir>,... rerun/sbatch_percell.sh [extra args for build_percell_table.py]
mkdir -p logs
REPO=${BIO_ANALYSIS_REPO:-$SLURM_SUBMIT_DIR}
"$REPO/rerun/run.sh" -m bio_analysis.rerun.build_synapse_table && \
"$REPO/rerun/run.sh" -m bio_analysis.rerun.build_percell_table --nb_cpus 64 "$@"
