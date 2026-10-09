# Manual curation data (dataset j0251)

Files produced by manual inspection that the analysis scripts read. Checksums in `SHA256SUMS`. The pickled
id lists are read by `general/analysis_params.py` (`load_known_mergers`, `load_handpicked_ids`) from the
prep-cache directory (`BIO_ANALYSIS_PREP`), i.e. they are copied there before a run.

| file | content | used by |
|---|---|---|
| `merger_arr.pkl` (30 ids), `excluded_full_cells_v8.csv` (same ids, readable) | full cells of GPe, GPi, LTS, STN and TAN excluded after manual review: mergers with a fragment of >= 50 um pathlength or with >= 1 synapse, or a missing compartment (Table S3) | every analysis run (`exclude_known_mergers = True`) |
| `manual_review_v8.xlsx` | the joined manual review of all inspected cells: merger review of the GPe/GPi/LTS/STN/TAN full cells, the random INT1-3 sample and 500 random MSNs; autapse review; soma shape; nuclear indentation; Table S4 myelin column. One row per cell, see its README sheet | `rerun/manual_review_tables_v8.py` (Tables S3, S12) |
| `240723_j0251v6_all_cellids_for_exclusion/` | the review tables behind the workbook: id lists of the inspected cells (`all_cellids.csv`, `cellids_no_msn.csv`, `all_msn_ids.csv`, `random_msn_ids_500*.csv`, `rndm_int_ids.csv`, `241122_rndm_500_msn_ids.csv`, `250912_rndm_full_msn_stn_ids.csv`), the merger reviews (`241016_*`, `241024_*_final.csv`, `241212_*_num_mergers.csv`, `240826_Random MSN IDS_RM_check.csv`) and the autapse reviews (`2503*_RM_autapses.csv`, `2504*_RM_autapses*.csv`, `25*_autapse_*.csv`) | `for_eval/merger_eval.py`, `for_eval/merger_eval_pathlength.py`, `dir_indir_pathway_analysis/GPi_autoapse_comp.py` |
| `LMAN_handpicked_arr.pkl` (46), `LMAN_handpicked.csv` | hand-picked LMAN axons | `LMAN_MSN_analysis/LMAN_number_per_MSN.py`, `LMAN_overlap_analysis.py` |
| `handpicked_{ASTRO,OLIGO,MICRO,OPC}_ids.npy`, `handpicked_MIGR_ids.{npy,pkl}` | manually identified glia and migratory neurons | `general/map_org2cells.py`, `ct_org_volume_density_cell_volume.py`, `ct_ov_morph_params.py` |
| `soma_nuclear_indentation_labels.csv` | blind classification of the non-MSN neuronal somata on raw EM z-stacks (regular / indentation / excluded) | `for_eval/nuclear_indentation_census.py` (Fig 5B, S11E) |
| `umap_outlier_group_msn_only_reviewed.csv`, `umap_outlier_group_all_types_reviewed.csv` | the two groups of MSNs apart from the UMAP clouds that were manually reviewed for mergers | `general/ct_morph_umap_outlier_groups.py` (Fig S6) |
| `synapse_area_validation/` | the 40 sampled synapses (`sampled_synapses_40.csv`), the painted synaptic-junction masks (`painted_sj/`, one `.npz` + `.json` per synapse) and the per-synapse areas (`synapse_areas_40.csv`) | `for_eval/synapse_area_manual_validation.py`, `for_eval/synapse_area_validation/` (Fig S1E, F) |
| `celltype_classifier_cv_v8/` | cell type ground truth (458 cells, 19 classes; `j0251_celltype_gt_v8_*.csv`) and the held-out per-cell predictions of the 8-fold cross-validation at redundancies 1/10/20/50 (`fold<k>.csv`) | `for_eval/celltype_classifier_cv_tables.py` (Tables S5-S7) |
