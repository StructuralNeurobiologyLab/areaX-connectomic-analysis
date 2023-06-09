#mito volume density axon per celltype
if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_organell_volume_density
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats
    from itertools import combinations

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    start = time.time()
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    with_glia = False
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    mito_k = 3
    cls = CelltypeColors()
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230609_j0251v5_ct_mito_vol_density_mcl_%i_ax%i_k%i_%s" % (
        min_comp_len_cell, min_comp_len_ax, mito_k, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get volume density mito per celltype', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, mito k = %i, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, mito_k, color_key))
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    known_mergers = analysis_params.load_known_mergers()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    if with_glia:
        glia_cts = analysis_params._glia_cts
    sd_mi = SegmentationDataset('mi', working_dir=global_params.wd)
    mi_ids = sd_mi.ids
    mito_coords = sd_mi.load_numpy_data("rep_coord")
    mito_volumes = sd_mi.load_numpy_data("size")
    columns = ['axon mito density [1/µm]', 'axon mito volume density [µm³/µm]', 'dendrite mito volume density [1/µm]',
               'dendrite mito volume density [µm³/µm]', 'celltype', 'mean firing rate singing']
    result_pd = pd.DataFrame(colums = columns, index = range(20000))
    firing_rate_dict = {1: 15, 2: 1.58, 3: 34.9, 4: 1, 5: 65.1, 6: 135, 7: 258, 8: 19.1, 9: 35.8}

    log.info('Iterate over each celltype')
    for ct in tqdm(range(num_cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if ct in axon_cts:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = analysis_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info('Get mito volume density per cell')
        #ellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict, k, min_comp_len = input
        if ct in axon_cts:
            input = [[cellid, mi_ids, mito_coords, mito_volumes, cell_dict, mito_k, min_comp_len_ax, True] for cellid in cellids]
        else:
            input = [[cellid, mi_ids, mito_coords, mito_volumes, cell_dict, mito_k, min_comp_len_cell, False] for cellid in cellids]
        output = start_multiprocess_imap(get_organell_volume_density, input)
        output = np.array(output, dtype='object')
        axon_so_density = np.concatenate(output[:, 0])
        axon_volume_density = np.concatenate(output[:, 1])
        if ct not in axon_cts:
            dendrite_so_density = np.concatenate(output[:, 2])
            dendrite_volume_density = np.concatenate(output[:, 3])

