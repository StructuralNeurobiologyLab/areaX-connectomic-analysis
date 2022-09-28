import numpy as np
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset, SegmentationObject
from syconn.handler.config import initialize_logging
from syconn.extraction.cs_processing_steps import combine_and_split_cs
import matplotlib.pyplot as plt
import seaborn as sns
from syconn.reps import connectivity_helper as ch
from functools import partial
from multiprocessing import pool
from tqdm import tqdm

global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
highest_cs_number = 10000
f_name = "cajal/nvmescratch/users/arother/cs_debugging/20220928_threshold_tests_%i/" % highest_cs_number
f_name_saving = "cajal/nvmescratch/users/arother/j0251v4_prep/"
pathlength_threshold = 11000 #µm
sd_cs = SegmentationDataset("cs", working_dir=global_params.config.working_dir)
log = initialize_logging('220927 cs debugging', log_dir=f_name + '/logs/')
log.info(f"Threshold started with a pathlength threshold of {pathlength_threshold} µm, largest cs number of {highest_cs_number} cs")
cs_sizes = sd_cs.load_numpy_data('size')
cs_ids = sd_cs.ids

log.info(f"Step 1: get ssv ids for cells larger {pathlength_threshold} in MSN")
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
ssv_ids = ssd.ssv_ids
ssv_total_pathlengths = ssd.load_numpy_data('total_edge_length')
ssv_bounding_boxs = ssd.load_numpy_data('bounding_box')
ssv_celltypes = ssd.load_numpy_data('celltype_cnn_e3')
msn_ids = ssv_ids[ssv_celltypes == 2]
msn_pathlengths = ssv_total_pathlengths[ssv_celltypes == 2] /1000 #µm
excluded_msn_ids = msn_ids[msn_pathlengths >= pathlength_threshold]
write_obj2pkl("%s/msn_over_11mm_pathlength.pkl" % f_name, excluded_msn_ids)

log.info(f'{len(excluded_msn_ids)} MSNs found with a pathlength >={pathlength_threshold} µm')

log.info("Step 2: Check cs sizes")

max_size = np.max(cs_sizes)
ind_max = np.argmax(cs_sizes)
max_id = cs_ids[ind_max]
max_ssvs = ch.cs_id_to_partner_ids_vec([max_id])[0]
log.info(f'Max cs size is {max_size} voxel, cs id = {max_id}, ssv_ids are {max_ssvs}')

high_cs_sorting_ind = np.argsort(cs_sizes)
sorted_sizes = cs_sizes[high_cs_sorting_ind]
sorted_ids = cs_ids[high_cs_sorting_ind]

log.info(f'cs size of {highest_cs_number}th cs is {sorted_sizes[-highest_cs_number]} voxel')
'''
sns.histplot(sorted_sizes[:-highest_cs_number])
plt.xlabel("cs sizes ordered [voxel]")
plt.ylabel("count of cs")
plt.title("Contact site sizes of largest %i cs" % highest_cs_number)
plt.savefig("%s/high_%i_cs_hist.png" % (f_name, highest_cs_number))
plt.close()

log.info(f'The largest {highest_cs_number} cs are plotted')
'''

log.info("Step 3: check if MSN mergers are in large ssv_ids")
largest_ssv_ids = np.zeros((highest_cs_number, 2))
for i in range(highest_cs_number):
    largest_ssv_ids[i] = ch.cs_id_to_partner_ids_vec([sorted_ids[-(i + 1)]])[0]
largest_cs_ssv_ids = np.unique(np.concatenate(largest_ssv_ids))
log.info(f'{len(largest_cs_ssv_ids)} cells are part of the largest {highest_cs_number} cs.')
ratio_excl_msns = len(excluded_msn_ids) / len(largest_cs_ssv_ids) * 100
log.info('%.2f percent of the cells included in largest cs are misclassified/merged MSNs (pathlength >= %i µm)' % (ratio_excl_msns, pathlength_threshold))
cell_merger_inds = np.any(np.in1d(largest_ssv_ids, excluded_msn_ids).reshape(len(largest_ssv_ids), 2), axis = 1)
large_cs_ids_excl_cells = largest_ssv_ids[cell_merger_inds]
ratio_excl_cells_cs = len(large_cs_ids_excl_cells) / len(largest_ssv_ids) * 100
log.info('At least one misclassified/merged MSNs was involved in %.2f percent of largest cs.' % ratio_excl_cells_cs)

log.info("Step 4: Calculate ssvs bounding box vs pathlength")
ssvs_bb_scaled = ssv_bounding_boxs * ssd.scaling
ssvs_bb_dist = np.linalg.norm(ssvs_bb_scaled[:, 0] - ssvs_bb_scaled[:, 1], axis=1)
bb2pathlength_ratio = ssv_total_pathlengths / ssvs_bb_dist
#exclude msns with ratio over 50
over_50_ids = ssv_ids[bb2pathlength_ratio >= 50]
over_50_msns = over_50_ids[np.in1d(over_50_ids, msn_ids)]
write_obj2pkl("%s/msn_over_50_bb2pathlength.pkl" % f_name, over_50_msns)
astros_2_exclude = np.unique(np.hstack([over_50_msns, excluded_msn_ids]))
write_obj2pkl("%s/pot_astro_ids.pkl" % f_name_saving, astros_2_exclude)

ratio_excl_astros = len(astros_2_exclude) / len(largest_cs_ssv_ids) * 100
log.info('%.2f percent of the cells included in largest cs are misclassified/merged MSNs (bbratio > 50, pathlength >= %i)' % (ratio_excl_astros, pathlength_threshold))
astro_merger_inds = np.any(np.in1d(largest_ssv_ids, astros_2_exclude).reshape(len(largest_ssv_ids), 2), axis = 1)
large_cs_ids_astros = largest_ssv_ids[astro_merger_inds]
ratio_excl_astros_cs = len(large_cs_ids_astros) / len(largest_ssv_ids) * 100
log.info('At least one misclassified/merged MSNs was involved in %.2f percent of largest cs.' % ratio_excl_astros_cs)

raise ValueError

def find_jumps(i, array, difference):
    if i == len(array) - 1:
        return 0
    if array[i + 1] - array[i] > difference:
        return i

p = pool.Pool()
sorted_sizes_part = sorted_sizes[sorted_sizes > global_params.config['meshes']['mesh_min_obj_vx'] * 100]
sorted_sizes_part = sorted_sizes_part[sorted_sizes_part < 10**6]
potential_jumps = p.map(partial(find_jumps, array = sorted_sizes_part, difference = 10000), tqdm(range(len(sorted_sizes_part))))

raise ValueError



sns.histplot(cs_sizes, bins=100)
plt.xlabel("cs sizes [voxel]")
plt.ylabel("count of cs")
plt.title("Contact site sizes of all cs")
plt.savefig("%s/all_cs_hist.png" % f_name)
plt.close()

log.info('All cs sizes plotted in one histogram')



