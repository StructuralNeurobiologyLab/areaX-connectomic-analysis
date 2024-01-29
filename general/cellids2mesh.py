#small script to export cell meshes as kzip

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_organell_ids_comps
    import numpy as np
    from tqdm import tqdm
    from syconn.proc.meshes import write_mesh2kzip, write_meshes2kzip
    from scipy.spatial import cKDTree
    from collections import Counter

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    f_name = 'cajal/scratch/users/arother/230804_neuron_example_meshes'
    #f_name = 'cajal/scratch/users/arother/240115_LMAN_example_meshes'

    bio_params = Analysis_Params(working_dir = global_params.wd, version = 'v6')
    ct_dict = bio_params.ct_dict()
    whole_cell = True
    get_mitos = False
    get_mitos_comp_sep = False
    get_only_myelin = False

    #cellids = [ 126798179, 1155532413, 15724767, 24397945, 32356701, 26790127, 379072583]
    #cellids = [15521116, 10157981]
    #cellids = [1080627023]
    #cellids = [3171878, 18222490, 50542644, 96194764, 436157555]
    #cellids = [32356701, 26790127]
    cellids = [844683784, 373956306, 820388630, 975932938, 1355540633]

    if whole_cell:
        for cellid in tqdm(cellids):
            cell = SuperSegmentationObject(cellid)
            cell.load_attr_dict()
            indices, vertices, normals = cell.mesh
            celltype = ct_dict[cell.attr_dict['celltype_pts_e3']]
            kzip_out = f'{f_name}/{cellid}_{celltype}_mesh'
            write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), normals, None,
                            f'{cellid}.ply')

    if get_mitos:
        mito_color_rgba = np.array([189, 195, 199, 1])
        if get_mitos_comp_sep == True:
            sd_mito = SegmentationDataset('mi', working_dir=global_params.wd)
            mito_ids = sd_mito.ids
            mito_rep_coords = sd_mito.load_numpy_data('rep_coord')
            for cellid in tqdm(cellids):
                comp_mito_dict = get_organell_ids_comps([cellid, mito_ids, mito_rep_coords])
                cell = SuperSegmentationObject(cellid)
                cell.load_attr_dict()
                mitos = np.array(cell.mis)
                cell_mito_ids = cell.mi_ids
                celltype = ct_dict[cell.attr_dict['celltype_pts_e3']]
                for key in comp_mito_dict:
                    comp_mito_ids = comp_mito_dict[key]
                    comp_mitos = mitos[np.in1d(cell_mito_ids, comp_mito_ids)]
                    mito_inds = []
                    mito_verts = []
                    mito_norms = []
                    mito_cols = []
                    for mito in comp_mitos:
                        indices, vertices, normals = mito.mesh
                        mito_inds.append(indices.astype(np.float32))
                        mito_verts.append(vertices.astype(np.float32))
                        mito_norms.append(normals.astype(np.float32))
                        mito_cols.append(mito_color_rgba)
                    kzip_out = f'{f_name}/{cellid}_{celltype}_{key}_mito_mesh'
                    ply_f_names = [f'{cellid}_{key}_mito_{i}.ply' for i in range(len(mito_inds))]
                    write_meshes2kzip(kzip_out, mito_inds, mito_verts, mito_norms, mito_cols,
                                      ply_f_names)
        else:
            for cellid in tqdm(cellids):
                cell = SuperSegmentationObject(cellid)
                cell.load_attr_dict()
                mitos = cell.mis
                mito_inds = []
                mito_verts = []
                mito_norms = []
                mito_cols = []
                for mito in mitos:
                    indices, vertices, normals = mito.mesh
                    mito_inds.append(indices.astype(np.float32))
                    mito_verts.append(vertices.astype(np.float32))
                    mito_norms.append(normals.astype(np.float32))
                    mito_cols.append(mito_color_rgba)
                celltype = ct_dict[cell.attr_dict['celltype_pts_e3']]
                kzip_out = f'{f_name}/{cellid}_{celltype}_mito_mesh'
                ply_f_names = [f'{cellid}_mito_{i}.ply' for i in range(len(mito_inds))]
                write_meshes2kzip(kzip_out, mito_inds, mito_verts, mito_norms, mito_cols,
                                ply_f_names)

    if get_only_myelin:
        #get mesh of axon where myelin is
        for cellid in tqdm(cellids):
            cell = SuperSegmentationObject(cellid)
            cell.load_attr_dict()
            celltype = ct_dict[cell.attr_dict['celltype_pts_e3']]
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