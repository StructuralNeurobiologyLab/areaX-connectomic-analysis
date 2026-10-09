# Manual validation of the synaptic-area measure (Figure S1E, F)

The automated synapse size of the paper is the mesh area of the `syn_ssv` object divided by two. For the validation,
40 synapses were drawn from the automatically detected synapses (predicted area >= 0.05 um2, five logarithmically spaced
size bins, eight per bin), their synaptic junction was painted blind by one annotator on the raw EM (only the pre/post segmentation
shown), and the painted voxels were fed through SyConn's own synapse pipeline (contact-site detection, intersection with
the sj mask, clustering, meshing, mesh area / 2) exactly like the predicted sj voxels. The control arm runs the network's
own sj voxels through the same pipeline and reproduces the published sizes.

Data (curation/synapse_area_validation/): `sampled_synapses_40.csv` (the sample, API coordinates in mag1 voxels),
`painted_sj/<id>_offset_x_y_z_sj.npz` (painted sj masks + cube offsets, with the json metadata), `synapse_areas_40.csv`
(per synapse: API size, pipeline size of the predicted sj (control) and of the painted sj = the values of Figure S1E).

Scripts (run in a SyConn environment, the EM/segmentation are read from the public SyConn server, the sj prediction from
the working directory, `SYN_AREA_WD`):

    python sample_synapses.py --n-bins 5 --n-per-bin 8 --size-min 0.05 --shuffle --seed <seed> --out sample.csv
    python fetch_cubes.py --csv sample.csv --seg --pred          # 4 x 4 x 4 um cubes: raw, pre/post segmentation, predicted sj
    python verify_cubes.py sample.csv                            # cubes against the server before painting
    (painting: the annotator paints the junction on the raw EM; the painting tool is not part of this repository)
    for f in painted_sj/*_sj.npz; do echo "$f|manual"; echo "$f|pred"; done > jobs.txt
    SYN_AREA_PY=<python> xargs -a jobs.txt -P 8 -I{} ./run_one_obj.sh "{}"   # gt_sj_pipeline.py per synapse and arm
    SAMPLE_CSV=sample.csv python make_ssv_masks.py
    python render_all20.py --csv sample.csv --out final40.png   # pooled areas csv (+ 3D overview) from the pipeline objects
    python diagnose20.py                                         # per synapse: single junction, within 100 nm of the paint?
    python plot_fit.py final40_areas.csv final40                 # log-log and linear fits, bootstrap CI

`../synapse_area_manual_validation.py` reproduces the statistics of Figure S1E from `synapse_areas_40.csv` alone.
