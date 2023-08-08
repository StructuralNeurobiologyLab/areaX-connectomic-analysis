#get radius of soma for all celltypes

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct, get_cell_soma_radius
    from analysis_conn_helper import filter_synapse_caches_for_ct, get_number_sum_size_synapses
    from result_helper import ConnMatrix
    from analysis_colors import CelltypeColors
    from analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.mp.mp_utils import start_multiprocess_imap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    celltype_key = analysis_params.celltype_key()
    min_comp_len_cells = 200
    exclude_known_mergers = True
    cls = CelltypeColors()
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGP'
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230807_j0251v5_cts_soma_radius_mcl_%i_%s" % (
    min_comp_len_cells,color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    save_svg = True
    log = initialize_logging('soma raidus calculation', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, known mergers excluded = %s, colors = %s" % (
        min_comp_len_cells,exclude_known_mergers, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #filter cells for min_comp_len
    log.info("Step 1/X: Load cell dicts and get suitable cellids")
    if exclude_known_mergers:
        known_mergers = analysis_params.load_known_mergers()
    # To Do: also exlclude MSNs from list
    celltypes = analysis_params.load_celltypes_full_cells(with_glia=False)
    num_cts = len(celltypes)
    full_cell_dicts = {}
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_suitable_ids_cts = []
    for ct in tqdm(celltypes):
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        if ct == 2:
            misclassified_asto_ids = analysis_params.load_potential_astros()
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict,
                                                min_comp_len=min_comp_len_cells,
                                                axon_only=False,
                                                max_path_len=None)
        full_cell_dicts[ct] = cell_dict
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)
        all_suitable_ids_cts.append(np.zeros(len(cellids_checked)) + ct)

    log.info('Step 2/X: Get soma radius from all cellids')
    columns = ['cellid', 'celltype', 'soma radius', 'soma diameter', 'soma centre voxel coords x', 'soma centre voxel coords y', 'soma centre voxel coords z']
    soma_results_pd = pd.DataFrame(columns=columns, index = range(len(all_suitable_ids)))
    soma_results_pd['cellid'] = all_suitable_ids
    soma_results_pd['celltype'] = all_suitable_ids_cts
    output = start_multiprocess_imap(get_cell_soma_radius, all_suitable_ids)
    output = np.array(output, dtype='object')
    soma_centres = np.concatenate(output[:, 0]).reshape(len(output), 3)
    soma_centres_vox = soma_centres / [10, 10, 25]
    soma_radii = output[:, 1]
    soma_results_pd['soma centre voxel x'] = soma_centres_vox[:, 0].astype(int)
    soma_results_pd['soma centre voxel y'] = soma_centres_vox[:, 1].astype(int)
    soma_results_pd['soma centre voxel z'] = soma_centres_vox[:, 2].astype(int)
    soma_results_pd['soma radius'] = soma_radii
    soma_results_pd['soma diameter'] = soma_radii * 2
    soma_results_pd.to_csv(f'{f_name}/soma_radius_results.csv')

    log.info('Step 3/X: Plot results')

    #multiprocess get_cell_soma_radius
    #write additional function to verify radius manually


