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


#debugging to find attr_dict for cell
#load all of them



raise ValueError


#debugging part two
cellid1 = 1820916030
cellid2 = 64210280
synid1 = 18562800
cs_id = 275781054487918910
cs_ssv_id = 662539735

import numpy as np
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset, SegmentationObject
global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
sd_cs_ssv = SegmentationDataset("cs_ssv", working_dir=global_params.config.working_dir)
cs_ssv_ids = sd_cs_ssv.ids
cs_ssv_coords = sd_cs_ssv.load_numpy_data("rep_coord")
cs_ssv_cs_ids = sd_cs_ssv.load_numpy_data("cs_ids")
cs_ssv_sizes = sd_cs_ssv.load_numpy_data("size")
h=input()
cs_id_index = np.where(cs_ssv_cs_ids == cs_id)[0]
example_cs_ssv_id = cs_ssv_ids[cs_id_index]

attr_dc = {}

for ssvpartners_enc, cs_ids in tqdm(rel_cs_items9255):
    n_items_for_path += 1
    ssv_ids = ch.cs_id_to_partner_ids_vec([ssvpartners_enc])[0]

    # verify ssv_partner_ids
    cs_lst = sd_cs.get_segmentation_object(cs_ids)
    vxl_iter_lst = []
    vx_cnt = 0
    for cs in cs_lst:
        vx_store = VoxelStorageDyn(cs.voxel_path, read_only=True,
                                   disable_locking=True)
        vxl_iter_lst.append(vx_store.iter_voxelmask_offset(cs.id, overlap=1))
        vx_cnt += vx_store.object_size(cs.id)
    if mesh_min_obj_vx > vx_cnt:
        ccs = []
    else:
        # generate connected component meshes; vertices are in nm
        ccs = gen_mesh_voxelmask(chain(*vxl_iter_lst), scale=scaling, **meshing_kws)

    for mesh_cc in ccs:
        cs_ssv = sd_cs_ssv.get_segmentation_object(cs_ssv_id, create=True)
        csssv_attr_dc = dict(neuron_partners=ssv_ids)
        # don't store normals
        cs_ssv._mesh = [mesh_cc[0], mesh_cc[1], np.zeros((0,), dtype=np.float32)]
        csssv_attr_dc["mesh_bb"] = cs_ssv.mesh_bb
        csssv_attr_dc["mesh_area"] = cs_ssv.mesh_area
        csssv_attr_dc["bounding_box"] = (cs_ssv.mesh_bb // scaling).astype(np.int32)
        csssv_attr_dc["rep_coord"] = (seghelp.calc_center_of_mass(mesh_cc[1].reshape((-1, 3))) // scaling).astype(
            np.int64)
        csssv_attr_dc["cs_ids"] = list(cs_ids)
        csssv_attr_dc["size"] = 0

        # add cs_ssv dict to AttributeStorage
        attr_dc[cs_ssv_id] = csssv_attr_dc
        if cs_ids[0] == test_cs_id:
            print(cs_ssv_id)
            print(attr_dc[cs_ssv_id])
            print(base_dir)
        if use_new_subfold:
            cs_ssv_id += np.uint(1)
            if cs_ssv_id - base_id >= div_base:
                # next ID chunk mapped to this storage
                id_chunk_cnt += 1
                old_base_id = base_id
                base_id += np.uint(sd_cs_ssv.n_folders_fs * div_base) * id_chunk_cnt
                assert subfold_from_ix(base_id, sd_cs_ssv.n_folders_fs, old_version=False) == \
                       subfold_from_ix(old_base_id, sd_cs_ssv.n_folders_fs, old_version=False)
                cs_ssv_id = base_id
        else:
            cs_ssv_id += np.uint(sd_cs.n_folders_fs)

    if n_items_for_path > n_per_voxel_path:
        cur_path_id += 1
        if len(voxel_rel_paths) == cur_path_id:
            raise ValueError(f'Worker ran out of possible storage paths for storing {sd_cs_ssv.type}.')
        n_items_for_path = 0
        id_chunk_cnt = 0
        base_id = ix_from_subfold(voxel_rel_paths[cur_path_id], sd_cs.n_folders_fs)
        cs_ssv_id = base_id
        base_dir = path2dir + voxel_rel_paths[cur_path_id]
        # base_dir = sd_cs_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
        print("new base_dir = %s" % base_dir)


