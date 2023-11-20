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

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    f_name = 'cajal/scratch/users/arother/230804_neuron_example_meshes'

    bio_params = Analysis_Params(working_dir = global_params.wd, version = 'v5')
    ct_dict = bio_params.ct_dict()
    whole_cell = False
    get_mitos = True
    get_mitos_comp_sep = True

    #cellids = [ 126798179, 1155532413, 15724767, 24397945, 32356701, 26790127, 379072583]
    #cellids = [15521116, 10157981]
    cellids = [1080627023]

    if whole_cell:
        for cellid in tqdm(cellids):
            cell = SuperSegmentationObject(cellid)
            indices, vertices, normals = cell.mesh
            celltype = ct_dict[cell.celltype()]
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
                mitos = np.array(cell.mis)
                cell_mito_ids = cell.mi_ids
                celltype = ct_dict[cell.celltype()]
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
                celltype = ct_dict[cell.celltype()]
                kzip_out = f'{f_name}/{cellid}_{celltype}_mito_mesh'
                ply_f_names = [f'{cellid}_mito_{i}.ply' for i in range(len(mito_inds))]
                write_meshes2kzip(kzip_out, mito_inds, mito_verts, mito_norms, mito_cols,
                                ply_f_names)