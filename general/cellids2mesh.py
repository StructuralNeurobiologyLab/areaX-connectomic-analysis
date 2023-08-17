#small script to export cell meshes as kzip

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationObject
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import numpy as np
    from tqdm import tqdm
    from syconn.proc.meshes import write_mesh2kzip

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    f_name = 'cajal/scratch/users/arother/230804_neuron_example_meshes'

    bio_params = Analysis_Params(working_dir = global_params.wd, version = 'v5')
    ct_dict = bio_params.ct_dict()

    cellids = [ 662789385, 542544908, 1340425305, 728819856, 1200237873,27161078]

    for cellid in tqdm(cellids):
        cell = SuperSegmentationObject(cellid)
        indices, vertices, normals = cell.mesh
        celltype = ct_dict[cell.celltype()]
        kzip_out = f'{f_name}/{cellid}_{celltype}_mesh'
        write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), normals, None,
                        f'{cellid}.ply')