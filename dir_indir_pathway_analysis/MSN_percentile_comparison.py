#script for looking at MSN percentile connectivity with GPe/i, FS, STN, TAN


from u.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
from u.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity
from u.arother.bio_analysis.dir_indir_pathway_analysis.spiness_sorting import saving_spiness_percentiles
import time
from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset
import os as os

global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
start = time.time()
f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211122_j0251v3_MSN_percentile_comparison"
if not os.path.exists(f_name):
    os.mkdir(f_name)
log = initialize_logging('MSN percentile comparison connectivity', log_dir=f_name + '/logs/')
log.info("MSN percentile comparison starts")
time_stamps = [time.time()]
step_idents = ['t-0']
#ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               #10: "NGF"}
percentile = [10, 25, 50]
comp_lengths = [100, 200, 500, 1000]

log.info("Step 1/8: MSN percentile compartment comparison")
#create MSN spiness percentiles with different comp_lengths
filename_spiness_saving = "/wholebrain/scratch/arother/j0251v3_prep/"
for cl in comp_lengths:
    filename_spiness_results = "%s/spiness_percentiles_mcl%i" % (f_name, cl)
    if not os.path.exists(filename_spiness_results):
        os.mkdir(filename_spiness_results)
    saving_spiness_percentiles(ssd, celltype = 2, filename_saving = filename_spiness_saving, filename_plotting = filename_spiness_results, percentiles = percentile, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["spiness percentiles calculated"]

log.info("Step 2/8: MSN percentile compartment comparison")
# calculate parameters such as axon/dendrite length, volume, tortuosity and compare within celltypes
for cl in comp_lengths:
    for p in percentile:
        result_MSN_filename_p1 = axon_den_arborization_ct(ssd, celltype=2, percentile = p, filename=f_name, full_cells=True, handpicked=False, min_comp_len = cl)
        result_MSN_filename_p2 = axon_den_arborization_ct(ssd, celltype=2, percentile = 100 - p, filename=f_name, full_cells=True, handpicked=False, min_comp_len = cl)
        compare_compartment_volume_ct(celltype1=2, percentile = p, filename=f_name, filename1=result_MSN_filename_p1, filename2=result_MSN_filename_p2, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["compartment comparison finished"]

log.info("Step 3/8: MSN connectivity between percentiles")
# see how MSN percentiles are connected
for cl in comp_lengths:
    for p in percentile:
        MSN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile = p, filename=f_name, foldername_ct1=MSN_connectivity_resultsfolder, foldername_ct2=MSN_connectivity_resultsfolder, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["connctivity between MSN percentiles finished"]

log.info("Step 4/8: MSN - GPe connectivity different percentiles")
# see how MSN percentiles are connected to GPe
for cl in comp_lengths:
    for p in percentile:
        MSN_GPe_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
        MSN_GPe_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile = p, connected_ct=6, filename=f_name, foldername_ct1=MSN_GPe_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPe_p2_connectivity_resultsfolder, min_comp_len = cl)

log.info("Step 5/8: MSN - GPi connectivity different percentiles")
# see how MSN percentiles are connected to GPi
for cl in comp_lengths:
    for p in percentile:
        MSN_GPi_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
        MSN_GPi_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile = p, connected_ct=7, filename=f_name, foldername_ct1=MSN_GPi_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPi_p2_connectivity_resultsfolder, min_comp_len = cl)

log.info("Step 6/8: MSN - STN connectivity different percentiles")
# see how MSN percentiles are connected to STN
for cl in comp_lengths:
    for p in percentile:
        MSN_STN_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
        MSN_STN_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile = p, connected_ct=0, filename=f_name, foldername_ct1=MSN_STN_p1_connectivity_resultsfolder, foldername_ct2=MSN_STN_p2_connectivity_resultsfolder, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["connctivity MSN - STN finished"]

log.info("Step 7/8: MSN - FS connectivity")
# see how MSN percentiles are connected to FS
for cl in comp_lengths:
    for p in percentile:
        MSN_FS_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
        MSN_FS_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile = p, connected_ct=8, filename=f_name, foldername_ct1=MSN_FS_p1_connectivity_resultsfolder, foldername_ct2=MSN_FS_p2_connectivity_resultsfolder, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["connctivity MSN - FS finished"]

log.info("Step 8/8: MSN - TAN connectivity")
# see how MSN percentiles are connected to TAN
for cl in comp_lengths:
    for p in percentile:
        MSN_TAN_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=True, handpicked2=True, min_comp_len = cl)
        MSN_TAN_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=True, handpicked2=True, min_comp_len = cl)
        compare_connectivity(comp_ct1=2, percentile= p, connected_ct=5, filename=f_name, foldername_ct1=MSN_TAN_p1_connectivity_resultsfolder, foldername_ct2=MSN_FS_p2_connectivity_resultsfolder, min_comp_len = cl)

time_stamps = [time.time()]
step_idents = ["connctivity MSN - TAN finished"]

log.info("MSN percentile compartment and connectivity analysis finished")
step_idents = ["MSN percentile compartment and connectivity analysis finished"]