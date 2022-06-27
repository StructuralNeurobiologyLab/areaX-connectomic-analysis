#script to debug contact sites

import numpy as np
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.extraction import cs_processing_steps as cps
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from time import time


global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

#example cells and synapse id
cellid1 = 1820916030
cellid2 = 64210280
synid1 = 18562800

#get synapse parameters
sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
syn_ids = sd_synssv.ids
syn_prob = sd_synssv.load_numpy_data("syn_prob")
syn_partners = sd_synssv.load_numpy_data("neuron_partners")
syn_cts = sd_synssv.load_numpy_data("partner_celltypes")
syn_axs = sd_synssv.load_numpy_data("partner_axoness")
syn_sizes_mesh = sd_synssv.load_numpy_data("mesh_area")/2
syn_sizes = sd_synssv.load_numpy_data("size")
syn_coords = sd_synssv.load_numpy_data("rep_coord")
syn_cs_ids = sd_synssv.load_numpy_data("cs_ids")

#get cs parameters
sd_cs = SegmentationDataset("cs", working_dir=global_params.config.working_dir)
cs_ids = sd_cs.ids
cs_coords = sd_cs.load_numpy_data("rep_coord")
cs_sizes = sd_cs.load_numpy_data("size")

#get cs_ssv parameters
sd_cs_ssv = SegmentationDataset("cs_ssv", working_dir=global_params.config.working_dir)
cs_ssv_ids = sd_cs_ssv.ids
cs_ssv_coords = sd_cs_ssv.load_numpy_data("rep_coord")
cs_ssv_cs_ids = sd_cs_ssv.load_numpy_data("cs_ids")
cs_ssv_partners = sd_cs_ssv.load_numpy_data("neuron_partners")
cs_ssv_sizes = sd_cs_ssv.load_numpy_data("size")

start = time()

#get parameters for exmaple syn id
synid1_index = np.where(syn_ids == synid1)[0]
synid1_partners = syn_partners[synid1_index]
synid1_mesh_size = syn_sizes_mesh[synid1_index]
synid1_size = syn_sizes[synid1_index]
synid1_cts = syn_cts[synid1_index]
synid1_coord = syn_coords[synid1_index]
synid1_cs_id = syn_cs_ids[synid1_index]

#get parameters for cs_ssv between example cells
c1_mask = np.any(cs_ssv_partners == cellid1, axis = 1)
c1_cs_ssv_ids = cs_ssv_ids[c1_mask]
c1_cs_ssv_partners = cs_ssv_partners[c1_mask]
c1_cs_ssv_cs_ids = cs_ssv_cs_ids[c1_mask]
c1_cs_ssv_coords = cs_ssv_coords[c1_mask]
c1_cs_ssv_sizes = cs_ssv_sizes[c1_mask]
c12_mask = np.any(c1_cs_ssv_partners == cellid2, axis = 1)
c12_cs_ssv_ids = c1_cs_ssv_ids[c12_mask]
c12_cs_ssv_partners = c1_cs_ssv_partners[c12_mask]
c12_cs_ssv_cs_ids = c1_cs_ssv_cs_ids[c12_mask]
c12_cs_ssv_coords = c1_cs_ssv_coords[c12_mask]
c12_cs_ssv_sizes = c1_cs_ssv_sizes[c12_mask]
print(f"cs_ssv coords: {c12_cs_ssv_coords}")

#get synapses between cellids
sync1_mask = np.any(syn_partners == cellid1, axis = 1)
c1_syn_ids = syn_ids[sync1_mask]
c1_syn_partners = syn_partners[sync1_mask]
c1_syn_cs_ids = syn_cs_ids[sync1_mask]
c1_syn_coords = syn_coords[sync1_mask]
c1_syn_sizes = syn_sizes[sync1_mask]
sync12_mask = np.any(c1_syn_partners == cellid2, axis = 1)
c12_syn_ids = c1_syn_ids[sync12_mask]
c12_syn_partners = c1_syn_partners[sync12_mask]
c12_syn_cs_ids = c1_syn_cs_ids[sync12_mask]
c12_syn_coords = c1_syn_coords[sync12_mask]
c12_syn_sizes = c1_syn_sizes[sync12_mask]
print(f"syn_ssv coords: {c12_syn_coords}")

print(f"{time() - start}")

#raise ValueError

#get corresponding cs_parameters
#cs_syn_index = np.where(cs_ids == synid1_cs_id)[0][0]
#cs_syn_coord = cs_coords[cs_syn_index]
#cs_syn_size = cs_sizes[cs_syn_index]

#cs_cs1_index = np.where(cs_ids == c12_cs_ssv_cs_ids[0])[0]
#cs_cs1_coord = cs_coords[cs_cs1_index]
#cs_cs1_size = cs_sizes[cs_cs1_index]

#go through steps in cps
#filtered_cs = cps.filter_relevant_syn(sd_cs, ssd, log = None)



raise ValueError