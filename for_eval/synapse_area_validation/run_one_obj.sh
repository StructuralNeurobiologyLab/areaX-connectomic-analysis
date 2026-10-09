#!/bin/bash
# Dump the pipeline's per-object voxels for one painted cube.
#   argument: "<mask .npz path>|<manual|pred>"
# Writes to objs/<sid>_<mode>/ only on success, so a killed run leaves no
# half-written directory that a later run would skip.
cd "$(dirname "$0")"
PY=${SYN_AREA_PY:-python}   # python of a SyConn environment
f="${1%%|*}"; mode="${1##*|}"
b=$(basename "$f" _sj.npz); sid=${b%%_*}
d="objs/${sid}_${mode}"
if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then echo "SKIP $sid $mode"; exit 0; fi
extra=""; [ "$mode" = pred ] && extra="--predicted"
tmp="${d}.part"; rm -rf "$tmp"
mkdir -p logs
"$PY" gt_sj_pipeline.py --gt "$f" $extra \
    --save-voxels "$tmp" --out "results/final/${sid}_${mode}_objs.csv" \
    > "logs/${sid}_${mode}.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -n "$(ls -A "$tmp" 2>/dev/null)" ]; then
    mv "$tmp" "$d"; echo "OK $sid $mode"
else
    rm -rf "$tmp"; echo "FAIL $sid $mode (rc=$rc)"
fi
