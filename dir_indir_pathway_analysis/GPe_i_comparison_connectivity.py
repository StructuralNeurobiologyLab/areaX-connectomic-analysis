#script for looking at GPe/i connectivity with FS, STN, TAN


from u.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
from u.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity
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
f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211120_j0251v3_GPe_i_comparison"
if not os.path.exists(f_name):
    os.mkdir(f_name)
log = initialize_logging('GPe, GPi comparison connectivity', log_dir=f_name + '/logs/')
log.info("GPe/i comparison starts")
time_stamps = [time.time()]
step_idents = ['t-0']
#ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               #10: "NGF"}

log.info("Step 1/5: GPe/i compartment comparison")
# calculate parameters such as axon/dendrite length, volume, tortuosity and compare within celltypes
result_GPe_filename = axon_den_arborization_ct(ssd, celltype=6, filename=f_name, full_cells=True, handpicked=True)
result_GPi_filename = axon_den_arborization_ct(ssd, celltype=7, filename=f_name, full_cells=True, handpicked=True)
compare_compartment_volume_ct(celltype1=6, celltype2=7, filename=f_name, filename1=result_GPe_filename, filename2=result_GPe_filename, percentile=None)

time_stamps = [time.time()]
step_idents = ["compartment comparison finished"]

log.info("Step 2/5: GPe and GPi connectivity")
# see how GPe and GPi are connected
GPe_GPi_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=6, celltype2=7, filename=f_name, full_cells=True, handpicked1=True, handpicked2=True)
compare_connectivity(comp_ct1=6, comp_ct2=7, filename=f_name, foldername_ct1=GPe_GPi_connectivity_resultsfolder, foldername_ct2=GPe_GPi_connectivity_resultsfolder)

time_stamps = [time.time()]
step_idents = ["connctivity among GPe/i finished"]

log.info("Step 3/5: GPe/i - STN connectivity")
# see how GPe and GPi are connected to STN
GPe_STN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=6, celltype2=0, filename=f_name, full_cells=True, handpicked1=True, handpicked2=False)
GPi_STN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=7, celltype2=0, filename=f_name, full_cells=True, handpicked1=True, handpicked2=False)
compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=0, filename=f_name, foldername_ct1=GPe_STN_connectivity_resultsfolder, foldername_ct2=GPi_STN_connectivity_resultsfolder)

time_stamps = [time.time()]
step_idents = ["connctivity GPe/i - STN finished"]

log.info("Step 4/5: GPe/i - FS connectivity")
# see how GPe and GPi are connected to FS
GPe_FS_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=6, celltype2=8, filename=f_name, full_cells=True, handpicked1=True, handpicked2=False)
GPi_FS_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=7, celltype2=8, filename=f_name, full_cells=True, handpicked1=True, handpicked2=False)
compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=8, filename=f_name, foldername_ct1=GPe_FS_connectivity_resultsfolder, foldername_ct2=GPi_FS_connectivity_resultsfolder)

time_stamps = [time.time()]
step_idents = ["connctivity GPe/i - FS finished"]

log.info("Step 5/5: GPe/i - TAN connectivity")
# see how GPe and GPi are connected to TAN
GPe_TAN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=6, celltype2=5, filename=f_name, full_cells=True, handpicked1=True, handpicked2=True)
GPi_TAN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=7, celltype2=5, filename=f_name, full_cells=True, handpicked1=True, handpicked2=True)
compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=5, filename=f_name, foldername_ct1=GPe_TAN_connectivity_resultsfolder, foldername_ct2=GPi_TAN_connectivity_resultsfolder)

time_stamps = [time.time()]
step_idents = ["connctivity GPe/i - TAN finished"]

log.info("GPe/i compartment and connectivity analysis finished")
step_idents = ["GPe/i compartment and connectivity analysis finished"]


