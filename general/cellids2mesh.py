#small script to export cell meshes as kzip
import pandas as pd

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_organell_ids_comps
    from analysis_conn_helper import filter_synapse_caches_for_ct
    import numpy as np
    from tqdm import tqdm
    from syconn.proc.meshes import write_mesh2kzip, write_meshes2kzip
    from scipy.spatial import cKDTree
    from collections import Counter

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    f_name = 'cajal/scratch/users/arother/exported_meshes/230804_neuron_example_meshes'
    #f_name = 'cajal/scratch/users/arother/240115_LMAN_example_meshes'

    bio_params = Analysis_Params(version = 'v6')
    ct_dict = bio_params.ct_dict(with_glia = True)
    global_params.wd = bio_params.working_dir()
    axon_cts = bio_params.axon_cts()
    whole_cell = False
    organelle_class = ['golgi', 'er']
    get_orgs = True
    get_orgs_comp_sep = False
    get_only_myelin = False
    get_single_ves_coords = False
    get_membrane_close_vesicles_separate = False
    get_syns = False
    get_syns_comp = 1
    compartment_dict = {0:'dendrite', 1:'axon', 2:'soma'}


    # cellids = [ 126798179, 1155532413, 15724767, 24397945, 32356701, 26790127, 379072583]
    # cellids = [15521116, 10157981]
    # cellids = [1080627023]
    # cellids = [3171878, 18222490, 50542644, 96194764, 436157555]
    # cellids = [32356701, 26790127]
    # cellids = [844683784, 373956306, 820388630, 975932938, 1355540633]
    # gt HVC, LMAN, DA, example LTS, MSNs aroudn TAN, oligo + GPI (x2), other example glia cells
    # cellids = [195998712, 139212645,  88265889, 832232717, 841444450, 436157555, 1126849047, 379072583, 1469886143,1503488997, 2834161,561503453, 155343800, 1644151292]
    # example cellids INT1, MSN, INT3, INT2, STN, TAN, LTS, GPe, HVC, DA
    # example cells TAN, GPi, GPe, INT2, INT3, MSN
    #cellids = [10157981, 26790127, 32356701, 126798179, 24397945, 832232717]
    #get wrongly segmented bv and associated astrocytes
    #cellids = [2332213096, 2491837340, 2287912642, 2129941466, 2211357026, 2412109485]
    cellids = [1644151292]

    if get_orgs:
        org_color_rgba = np.array([189, 195, 199, 1])
        sd_org_dict = {}
        org_id_dict = {}
        org_coord_dict = {}
        for oc in organelle_class:
            sd_org = SegmentationDataset(oc, working_dir=global_params.wd)
            org_id_dict[oc] = sd_org.ids
            org_coord_dict[oc] = sd_org.load_numpy_data('rep_coord')
            sd_org_dict[oc] = sd_org
    if get_single_ves_coords:
        np_presaved_loc = bio_params.file_locations
        ves_columns = ['coord x', 'coord y', 'coord z', 'coord x blend', 'coord y blend', 'coord z blend']
        blender_scaling = 10 ** (-5)
        if get_membrane_close_vesicles_separate:
            dist_threshold = 10
    if get_syns:
        org_color_rgba = np.array([189, 195, 199, 1])
        min_syn_size = 0.1
        syn_prob_thresh = 0.6
        sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=global_params.wd)

    for cellid in tqdm(cellids):
        cell = SuperSegmentationObject(cellid)
        cell.load_attr_dict()
        ct_num = cell.attr_dict['celltype_pts_e3']
        celltype = ct_dict[ct_num]
        #if full cell mesh
        if whole_cell:
            indices, vertices, normals = cell.mesh
            kzip_out = f'{f_name}/{cellid}_{celltype}_mesh'
            write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), normals, None,
                            f'{cellid}.ply')
        if get_orgs:
            for oc in organelle_class:
                if oc == 'er':
                    orgs = [sd_org_dict[oc].get_segmentation_object(cellid)]
                else:
                    orgs = np.array(cell.get_seg_objects(oc))
                if get_orgs_comp_sep:
                    cell_org_ids = cell.lookup_in_attribute_dict(oc)
                    comp_org_dict = get_organell_ids_comps([cellid, org_id_dict[oc], org_coord_dict[oc], cell_org_ids])
                    for key in comp_org_dict:
                        comp_org_ids = comp_org_dict[key]
                        comp_orgs = orgs[np.in1d(cell_org_ids, comp_org_ids)]
                        org_inds = []
                        org_verts = []
                        org_norms = []
                        org_cols = []
                        for org in comp_orgs:
                            indices, vertices, normals = org.mesh
                            org_inds.append(indices.astype(np.float32))
                            org_verts.append(vertices.astype(np.float32))
                            org_norms.append(normals.astype(np.float32))
                            org_cols.append(org_color_rgba)
                        kzip_out = f'{f_name}/{cellid}_{celltype}_{key}_{oc}_mesh'
                        ply_f_names = [f'{cellid}_{key}_{oc}_{i}.ply' for i in range(len(org_inds))]
                        write_meshes2kzip(kzip_out, org_inds, org_verts, org_norms, org_cols,
                                          ply_f_names)
                else:
                    org_inds = []
                    org_verts = []
                    org_norms = []
                    org_cols = []
                    for org in orgs:
                        indices, vertices, normals = org.mesh
                        org_inds.append(indices.astype(np.float32))
                        org_verts.append(vertices.astype(np.float32))
                        org_norms.append(normals.astype(np.float32))
                        org_cols.append(org_color_rgba)
                    kzip_out = f'{f_name}/{cellid}_{celltype}_{oc}_mesh'
                    ply_f_names = [f'{cellid}_{oc}_{i}.ply' for i in range(len(org_inds))]
                    write_meshes2kzip(kzip_out, org_inds, org_verts, org_norms, org_cols,
                                    ply_f_names)
        if get_only_myelin:
            #get mesh of axon where myelin is
                #get skeleton myelin coords
                #map to vertex
            cell.load_skeleton()
            # load skeleton axoness, spiness attributes
            nodes = cell.skeleton['nodes'] * cell.scaling
            myelin_labels = cell.skeleton['myelin']
            # load mesh and put skeleton annotations on mesh
            indices, vertices, normals = cell.mesh
            vertices = vertices.reshape((-1, 3))
            kdt = cKDTree(nodes)
            dists, node_inds = kdt.query(vertices)
            vert_myelin_labels = myelin_labels[node_inds]
            #code based on meshes.compartmentalize_mesh
            ind_myelin = vert_myelin_labels[indices]
            indices= indices.reshape(-1, 3)
            ind_myelin = ind_myelin.reshape(-1, 3)
            ind_comp_maj = np.zeros((len(indices)), dtype=np.uint8)
            #get only indices where at least 2 have myelin
            for ii in range(len(indices)):
                triangle = ind_myelin[ii]
                cnt = Counter(triangle)
                my, n = cnt.most_common(1)[0]
                ind_comp_maj[ii] = my
            indices_myelin = indices[ind_comp_maj == 1].flatten()
            unique_my_inds = np.unique(indices_myelin)
            myelin_verts = vertices[unique_my_inds]
            remap_dict = {}
            for i in range(len(unique_my_inds)):
                remap_dict[unique_my_inds[i]] = i
            myelin_ind = np.array([remap_dict[i] for i in indices_myelin], dtype=np.uint64)
            kzip_out = f'{f_name}/{cellid}_{celltype}_myelin_mesh'
            write_mesh2kzip(kzip_out, myelin_ind.astype(np.float32), myelin_verts.astype(np.float32), None, None,
                            f'{cellid}_myelin.ply')
        if get_syns:
            m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(
                pre_cts=[ct_num],
                post_cts=None,
                syn_prob_thresh=syn_prob_thresh,
                min_syn_size=min_syn_size,
                axo_den_so=True,
                synapses_caches=None,
                sd_synssv=sd_syn_ssv)
            #get only synapses where one partner is cellid
            cellid_inds = np.any(np.in1d(m_ssv_partners, cellid).reshape(len(m_ssv_partners),2), axis = 1)
            cellid_ssv_partners = m_ssv_partners[cellid_inds]
            cellid_axs = m_axs[cellid_inds]
            cellid_ids = m_ids[cellid_inds]
            if get_syns_comp is not None:
                #get only synapses where cell is compartment
                cellid_ind = np.in1d(cellid_ssv_partners, cellid).reshape(len(cellid_ssv_partners), 2)
                comp_inds = np.in1d(cellid_axs, get_syns_comp).reshape(len(cellid_ssv_partners), 2)
                comp_cellid_inds = np.any(cellid_ind == comp_inds, axis=1)
                cellid_ids = cellid_ids[comp_cellid_inds]
            cell_syn_ids = cell.lookup_in_attribute_dict('syn_ssv')
            cell_syns = cell.syn_ssv
            suitable_inds = np.in1d(cell_syn_ids, cellid_ids)
            cell_syns = np.array(cell_syns)[suitable_inds]
            cell_syns_inds = []
            cell_syns_verts = []
            cell_syns_norms = []
            cell_syns_cols = []
            for syn in cell_syns:
                indices, vertices, normals = syn.mesh
                cell_syns_inds.append(indices.astype(np.float32))
                cell_syns_verts.append(vertices.astype(np.float32))
                cell_syns_norms.append(normals.astype(np.float32))
                cell_syns_cols.append(org_color_rgba)
            if get_syns_comp is not None:
                kzip_out = f'{f_name}/{cellid}_{celltype}_syn_{compartment_dict[get_syns_comp]}_mesh'
                ply_f_names = [f'{cellid}_syn_{compartment_dict[get_syns_comp]}_{i}.ply' for i in range(len(cell_syns_inds))]
            else:
                kzip_out = f'{f_name}/{cellid}_{celltype}_syn_mesh'
                ply_f_names = [f'{cellid}_syn_{i}.ply' for i in
                               range(len(cell_syns_inds))]
            write_meshes2kzip(kzip_out, cell_syns_inds, cell_syns_verts, cell_syns_norms, cell_syns_cols,
                              ply_f_names)
        if get_single_ves_coords:
            if ct_num in axon_cts:
                ct_ves_ids = np.load(f'{np_presaved_loc}/{celltype}_ids.npy')
                ct_ves_map2ssvids = np.load(f'{np_presaved_loc}/{celltype}_mapping_ssv_ids.npy')
                ct_ves_coords = np.load(f'{np_presaved_loc}/{celltype}_rep_coords.npy')
                if get_membrane_close_vesicles_separate:
                    ct_dist2membrane = np.load(f'{np_presaved_loc}/{celltype}_dist2matrix.npy')
            else:
                ct_ves_ids = np.load(f'{np_presaved_loc}/{celltype}_ids_fullcells.npy')
                ct_ves_map2ssvids = np.load(f'{np_presaved_loc}/{celltype}_mapping_ssv_ids_fullcells.npy')
                ct_ves_axoness = np.load(f'{np_presaved_loc}/{celltype}_axoness_coarse_fullcells.npy')
                ct_ves_coords = np.load(f'{np_presaved_loc}/{celltype}_rep_coords_fullcells.npy')
                ax_inds = ct_ves_axoness == 1
                ct_ves_map2ssvids = ct_ves_map2ssvids[ax_inds]
                ct_ves_coords = ct_ves_coords[ax_inds]
                if get_membrane_close_vesicles_separate:
                    ct_dist2membrane = np.load(f'{np_presaved_loc}/{celltype}_dist2matrix_fullcells.npy')
                    ct_dist2membrane = ct_dist2membrane[ax_inds]
            cell_ves_coords = ct_ves_coords[ct_ves_map2ssvids == cellid]
            if get_membrane_close_vesicles_separate:
                cell_dist2membrane = ct_dist2membrane[ct_ves_map2ssvids == cellid]
                close_coords = cell_ves_coords[cell_dist2membrane < dist_threshold]
                far_coords = cell_ves_coords[cell_dist2membrane >= dist_threshold]
                assert(len(cell_ves_coords) == len(close_coords) + len(far_coords))
                cell_close_coords_df = pd.DataFrame(columns = ves_columns, index=range(len(close_coords)))
                cell_close_coords_df['coord x'] = close_coords[:, 0]
                cell_close_coords_df['coord y'] = close_coords[:, 1]
                cell_close_coords_df['coord z'] = close_coords[:, 2]
                cell_close_coords_df['coord x blend'] = close_coords[:, 0].astype(np.float32) * cell.scaling[
                    0] * blender_scaling
                cell_close_coords_df['coord y blend'] = close_coords[:, 1] * cell.scaling[1].astype(
                    np.float32) * blender_scaling
                cell_close_coords_df['coord z blend'] = close_coords[:, 2] * cell.scaling[2].astype(
                    np.float32) * blender_scaling
                cell_close_coords_df.to_csv(f'{f_name}/{cellid}_{celltype}_membrane_close_vesicle_coords.csv')
                cell_far_coords_df = pd.DataFrame(columns=ves_columns, index=range(len(far_coords)))
                cell_far_coords_df['coord x'] = far_coords[:, 0]
                cell_far_coords_df['coord y'] = far_coords[:, 1]
                cell_far_coords_df['coord z'] = far_coords[:, 2]
                cell_far_coords_df['coord x blend'] = far_coords[:, 0].astype(np.float32) * cell.scaling[
                    0] * blender_scaling
                cell_far_coords_df['coord y blend'] = far_coords[:, 1] * cell.scaling[1].astype(
                    np.float32) * blender_scaling
                cell_far_coords_df['coord z blend'] = far_coords[:, 2] * cell.scaling[2].astype(
                    np.float32) * blender_scaling
                cell_far_coords_df.to_csv(f'{f_name}/{cellid}_{celltype}_membrane_far_vesicle_coords.csv')

            else:
                cell_ves_coords_df = pd.DataFrame(columns = ves_columns, index = range(len(cell_ves_coords)))
                cell_ves_coords_df['coord x'] = cell_ves_coords[:, 0]
                cell_ves_coords_df['coord y'] = cell_ves_coords[:, 1]
                cell_ves_coords_df['coord z'] = cell_ves_coords[:, 2]
                cell_ves_coords_df['coord x blend'] = cell_ves_coords[:, 0].astype(np.float32) * cell.scaling[0] * blender_scaling
                cell_ves_coords_df['coord y blend'] = cell_ves_coords[:, 1] * cell.scaling[1].astype(np.float32) * blender_scaling
                cell_ves_coords_df['coord z blend'] = cell_ves_coords[:, 2] * cell.scaling[2].astype(np.float32) * blender_scaling
                cell_ves_coords_df.to_csv(f'{f_name}/{cellid}_{celltype}_vesicle_coords.csv')