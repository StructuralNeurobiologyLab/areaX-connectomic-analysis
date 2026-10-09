#!/bin/bash
# Creates the run root used by rerun/run.sh: a directory in which the relative output/input paths of the analysis
# scripts resolve to the results root and the prep cache. Idempotent.
set -e
ROOT=${BIO_ANALYSIS_RUN_ROOT:?set BIO_ANALYSIS_RUN_ROOT}
RESULTS=${BIO_ANALYSIS_RESULTS:?set BIO_ANALYSIS_RESULTS}
PREP=${BIO_ANALYSIS_PREP:?set BIO_ANALYSIS_PREP}
mkdir -p "$ROOT/cajal/scratch/users/arother" "$ROOT/cajal/nvmescratch/users/arother" "$RESULTS" "$PREP"
for sub in general dir_indir_pathway_analysis LMAN_MSN_analysis single_vesicle_analysis TAN_DA_axo_analysis gain_estimation for_eval other; do
    mkdir -p "$RESULTS/$sub"
done
ln -sfn "$RESULTS" "$ROOT/cajal/scratch/users/arother/bio_analysis_results"
ln -sfn "$PREP" "$ROOT/cajal/nvmescratch/users/arother/j0251v6_prep"
echo "run root: $ROOT -> results $RESULTS, prep $PREP"
