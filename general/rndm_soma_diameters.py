#get random soma samples, calculate diameter
#3 from each celltype
#save in table

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct, get_cell_soma_radius
    from analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    celltypes = analysis_params.load_celltypes_full_cells(with_glia=False)
    min_comp_len = 200
    n_samples = 3
    use_gt = True
    use_skel = False  # if true would use skeleton labels for getting soma; vertex labels more exact
    np.random.seed(42)
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230808_j0251v5_rndm_cts_soma_radius_mcl_%i_n_%i_gt%s" % (
        min_comp_len, n_samples, use_gt)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('generate random samples for soam diameter testing', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, use ground_truth = %s, n samples per celltype = %i" % (
    min_comp_len, use_gt, n_samples))
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
    known_mergers = load_pkl2obj("/cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl")
    if use_gt:
        v6_gt = pd.read_csv("cajal/nvmescratch/projects/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v6_j0251_72_seg_20210127_agglo2_IDs.csv", names = ["cellids", "celltype"])

    log.info("Step 1/2: Iterate over celltypes to randomly select %i cells per celltype" % n_samples)
    num_cts = len(celltypes)
    columns = ['cellid', 'celltype', 'diameter', 'radius', 'soma centre voxel x', 'soma centre voxel y', 'soma centre voxel z']
    selected_cellids_pd = pd.DataFrame(columns = columns, index = range(num_cts * n_samples))
    for i, ct in enumerate(tqdm(celltypes)):
        #only get cells with min_comp_len
        cell_dict = analysis_params.load_cell_dict(ct)
        ct_str = ct_dict[ct]
        if use_gt:
            cellids = np.array(v6_gt["cellids"][v6_gt["celltype"] == ct_str])
            cell_dict_ids = list(cell_dict.keys())
            cellids = cellids[np.in1d(cellids, cell_dict_ids)]
        else:
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = analysis_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_str))
        #randomly select fraction of cells, number = cells_per_celltye
        rndm_cellids = np.random.choice(cellids, size=n_samples, replace=False)
        selected_cellids_pd.loc[i * n_samples: (i + 1) * n_samples -1, 'cellid'] = rndm_cellids
        selected_cellids_pd.loc[i * n_samples: (i + 1) * n_samples -1, 'celltype'] = ct_str
    log.info(f'All random samples (total: {len(selected_cellids_pd)}) from {num_cts} celltypes selected')

    log.info('Step 2/2: Get soma diameter from all randomly selected cells')
    #get soma centre estimate and radius in nm
    output = start_multiprocess_imap(get_cell_soma_radius, selected_cellids_pd['cellid'])
    output = np.array(output, dtype='object')
    soma_centres = np.concatenate(output[:, 0]).reshape(len(output), 3)
    soma_centres_vox = soma_centres/[10, 10, 25]
    soma_radii = output[:, 1].astype(float)
    selected_cellids_pd['soma centre voxel x'] = soma_centres_vox[:, 0].astype(int)
    selected_cellids_pd['soma centre voxel y'] = soma_centres_vox[:, 1].astype(int)
    selected_cellids_pd['soma centre voxel z'] = soma_centres_vox[:, 2].astype(int)
    selected_cellids_pd['radius'] = soma_radii
    selected_cellids_pd['diameter'] = soma_radii * 2
    selected_cellids_pd = selected_cellids_pd.round(2)
    selected_cellids_pd.to_csv(f'{f_name}/rndm_soma_diameters.csv')
    log.info('Soma diameter of random samples selected')