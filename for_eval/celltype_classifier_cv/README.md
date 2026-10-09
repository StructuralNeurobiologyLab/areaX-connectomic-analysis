# PTv3 cell type classifier: cross-validation, calibration and the whole-dataset run

The classifier itself (vendored Point Transformer V3, cell type wrapper, training script, inference loader and the
whole-dataset inference/merge scripts) lives in SyConn (branch `ptv3_celltype`: `syconn/cnn/ptv3/`,
`syconn/cnn/cnn_celltype_ptv3_j0251.py`, `syconn/exec/exec_celltype_ptv3_inference.py`,
`syconn/exec/exec_celltype_ptv3_merge.py`). This folder holds what belongs to the paper: the per-cell cross-validation
evaluation at several redundancies (Tables S6, S7) and the temperature calibration of the certainty.

Environment: SyConn with torch 2.x / spconv on an H100 node (see the SyConn branch), the v8 ground truth
(`curation/celltype_classifier_cv_v8/j0251_celltype_gt_v8_*.csv`, 458 cells, 19 classes) and the hybrid-cloud cache of the
ground truth cells. Variables used below:

    export SYCONN_CT_NFOLDS=8 SYCONN_CT_GT_CSV=<gt csv> SYCONN_CT_HC_DIR=<hybrid cloud cache of the GT cells>
    export syconn_wd=<j0251 working directory>
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

1. Cross-validation, one job per fold k = 0..7 (25k steps, ~2.6 h on one H100):

       python -u syconn/cnn/cnn_celltype_ptv3_j0251.py --sr <run dir>/fold$k --cval $k --seed 0 \
         --steps 25000 --eval_every 1000 --eval_batches 80 --workers 12 \
         --width 32 --drop_path 0.3 --lr 1e-4 --wd 0.1 --label_smoothing 0.1 --frag_p 0.25 --num_classes 19 --eval_frag 160

2. Held-out per-cell predictions at redundancies 1, 10, 20, 50 (`percell_eval.py`, writes `fold$k.csv`; with
   `--save_logits fold$k.npz` also the per-context logits for the calibration):

       python -u percell_eval.py --cval $k --infer_mode --ckpt <run dir>/fold$k/state_dict_best.pth \
         --out <cv dir>/fold$k.csv --save_logits <cv dir>/fold$k.npz --R 50 --num_classes 19 --workers 8

   The eight csvs are in `curation/celltype_classifier_cv_v8/`; `for_eval/celltype_classifier_cv_tables.py` turns them
   into Tables S5-S7.

3. Temperature: `python calibrate_temperature.py <cv dir> temperature.npy` (v8: T = 0.826).

4. Final model on the whole ground truth (40k steps): step 1 with `--cval -1 --steps 40000 --eval_every 2000`.

5. Whole-dataset inference (20 contexts per cell, 8 jobs of one H100 each over disjoint SSV id lists) and merge into the
   working directory arrays (`celltype_pts_v8`, `_certainty`, `_nunique`, `_probas`):

       python -u syconn/exec/exec_celltype_ptv3_inference.py --ids <ids_k.npy> --out <part_k.npz> \
         --ckpt <final>/state_dict.pth --num_classes 19 --temperature <T>
       python -u syconn/exec/exec_celltype_ptv3_merge.py --wd $syconn_wd --parts "<part*.npz>" --out <array dir> --temperature temperature.npy
