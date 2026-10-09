# rerun — running the manuscript analyses

The analysis scripts in the other folders are run unchanged; everything that concerns *how* they run (caches,
parameters per manuscript panel, cluster jobs) lives here.

## Inputs

- The SyConn working directory of the j0251 data set with the cell-type arrays `celltype_pts_v8_analysis`
  (`build_v8_celltype_array.py` remaps the classifier output `celltype_pts_v8` to the numbering of
  `general/analysis_params.py`).
- The prep cache (`BIO_ANALYSIS_PREP`): per-cell dictionaries `full_<CT>_dict.pkl` / `full_<CT>_arr.pkl`,
  axon dictionaries `ax_<CT>_dict.pkl`, the filtered synapse cache and the curation files of `../curation/`
  (`merger_arr.pkl`, hand-picked ids, cell type ground truth).
- Environment: SyConn (the branch with `compartmentalize_mesh_fromskel(ssv, k=1)`), see `../environment.yml`.

## Environment variables

`general/analysis_params.py` and `general/analysis_conn_helper.py` read

| variable | meaning |
|---|---|
| `BIO_ANALYSIS_WD` | SyConn working directory (replaces the hard-coded path of the version) |
| `BIO_ANALYSIS_PREP` | prep-cache directory |
| `BIO_ANALYSIS_CT_KEY` | ssd cell-type key, `celltype_pts_v8_analysis` for the v8 cell types |
| `BIO_ANALYSIS_RESULTS` | results root; the scripts write `<folder>/<run name>` below it |
| `BIO_ANALYSIS_RUN_ROOT` | directory used as cwd so that the relative result paths inside the scripts resolve to `BIO_ANALYSIS_RESULTS` (`setup_run_root.sh`) |
| `BIO_ANALYSIS_ENV` | conda environment with SyConn |

## Steps

1. `build_synapse_table.py` — the filtered synapse cache (`syn_cache_sp<p>_ms<s>_<ct key>.npz`): synapse
   probability >= 0.6, synaptic area >= 0.1 um2; exactly the return value of
   `analysis_conn_helper.filter_synapse_caches_general`, which the scripts pick up through that function.
2. `build_percell_table.py` — one pass over all candidate full cells that runs the per-cell worker functions
   (`analysis_prep_func.get_per_cell_morphology_params`, spine density, compartment radii, soma radius, myelin
   fraction, dendrite info, 7 um border rule) and writes the prep cache plus `percell_morph.csv`
   (`sbatch_percell.sh` runs steps 1 and 2 on one node).
3. `build_axon_dicts.py` — `ax_<CT>_dict.pkl` for the projection axon types (DA, LMAN, HVC).
4. `run_variants.py` — runs the manuscript variants of `variants.yml` (`--group fast | organelle | stage2 | stage3 |
   stage4`, in this order; later groups read csvs of earlier ones) as Slurm jobs (`--submit`) or in the current
   process (`--local`). Each variant sets the hard-coded parameters of one script through `run_variant.py`
   without editing the file; the run directory is named as in `variants.yml`.
5. The scripts that are not parameter variants run directly, e.g. `run.sh dir_indir_pathway_analysis/GPi_subpopulations.py`;
   `manifest.yml` lists every panel and table with its script and variant names.
6. `manual_review_tables_v8.py` — Tables S3 and S12 from `../curation/manual_review_v8.xlsx`.
7. `paper_numbers.py` — every number quoted in the text, looked up in the result csvs of one results root.

Usage: `rerun/run.sh <script> [args]` or `rerun/run.sh -m bio_analysis.rerun.<module> [args]` on a node, or
`sbatch -J <name> --export=ALL,BIO_ANALYSIS_...=... rerun/sbatch_run.sh <script>` from the repository root.
`mp.py` is the multiprocessing map used by the cache builders.
