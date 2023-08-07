#get radius of soma for all celltypes

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
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

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    start = time.time()
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
    columns = ['cellid', 'celltype', 'soma centre coords', 'soma centre voxel coords', 'soma radius']
    soma_results_pd = pd.DataFrame(columns=columns, index = range(len(all_suitable_ids)))
    soma_results_pd['cellid'] = all_suitable_ids
    soma_results_pd['celltype'] = all_suitable_ids_cts

    # calculate soma radius with the following steps
    # get meshes related to soma
    # calculate average coordinate to get soma centre
    # compare to images/ compare to soma centre from skeleton points
    # for each mesh coordinate get distance to soma centre
    # take median distance as radius
    # compare again to images

    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_comp_meshes = compartmentalize_mesh_fromskel(cell, 'axoness_avg10000')
    soma_mesh = cell_comp_meshes['soma']
    ind, vert, norm = soma_mesh
    soma_vert_coords = vert.reshape((-1, 3))
    soma_vert_avg = np.mean(soma_vert_coords, axis = 0)

