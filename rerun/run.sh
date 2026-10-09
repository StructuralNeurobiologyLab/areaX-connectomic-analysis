#!/bin/bash
# Run one analysis script (or a rerun module) without editing the paths inside the scripts.
#
#   rerun/run.sh general/connectivity_fraction_per_ct.py
#   rerun/run.sh -m bio_analysis.rerun.build_synapse_table --verify
#
# The analysis scripts write to the RELATIVE path cajal/scratch/users/arother/bio_analysis_results/<folder>/<run>,
# and read other runs the same way. They are therefore run with cwd = RUN_ROOT, where that relative path is a
# symlink to the results root (setup_run_root.sh). Absolute paths (working dir, prep cache, cell-type key) are
# overridden through the BIO_ANALYSIS_* variables read by general/analysis_params.py and
# general/analysis_conn_helper.py. Required: BIO_ANALYSIS_RUN_ROOT, BIO_ANALYSIS_RESULTS, BIO_ANALYSIS_PREP,
# BIO_ANALYSIS_ENV (conda environment with SyConn); for the v8 cell types also BIO_ANALYSIS_WD and
# BIO_ANALYSIS_CT_KEY=celltype_pts_v8_analysis.
set -e
: "${BIO_ANALYSIS_RUN_ROOT:?set BIO_ANALYSIS_RUN_ROOT}" "${BIO_ANALYSIS_RESULTS:?set BIO_ANALYSIS_RESULTS}" "${BIO_ANALYSIS_PREP:?set BIO_ANALYSIS_PREP}" "${BIO_ANALYSIS_ENV:?set BIO_ANALYSIS_ENV}"
export BIO_ANALYSIS_RUN_ROOT BIO_ANALYSIS_RESULTS BIO_ANALYSIS_PREP
ENV=$BIO_ANALYSIS_ENV
REPO=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$BIO_ANALYSIS_PREP" "$BIO_ANALYSIS_RESULTS"
"$REPO/rerun/setup_run_root.sh" >/dev/null
cd "$BIO_ANALYSIS_RUN_ROOT"
if [ "$1" = "-m" ]; then
    exec "$ENV/bin/python" "$@"
else
    SCRIPT=$1; shift
    exec "$ENV/bin/python" "$REPO/$SCRIPT" "$@"
fi
