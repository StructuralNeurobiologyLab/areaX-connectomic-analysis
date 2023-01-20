#load single vesicle data
#filter cells for completeness
#check distribution of dist2matrix

def get_ves_distance_per_cell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return the corresponding distances to matrix
    in nm. Filters vesicles with certain distance to matrix if filtering parameter given.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix
    :return: number of vesicles, vesicle number per pathlength
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    axon_pathlength = cell_input[5]
    #load cell skeleton, filter all vesicles not close to axon
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_ves_coords, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    cell_axo_ves_coords = cell_ves_coords[axo == 1]
    cell_axo_dist2matrix = cell_dist2matrix[axo == 1]
    #now filter according to distance to matrix
    cell_axo_ves_coords_thresh = cell_axo_ves_coords[cell_axo_dist2matrix < distance_threshold]
    number_vesicles = len(cell_axo_ves_coords)
    number_vesicles_close = len(cell_axo_ves_coords_thresh)
    #calculate density
    vesicle_density = number_vesicles / axon_pathlength
    vesicle_density_close = number_vesicles_close / axon_pathlength
    return [number_vesicles, number_vesicles_close, vesicle_density, vesicle_density_close]

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationObject
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import scipy.spatial
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    dist_threshold = 15 #nm
    cls = CelltypeColors()
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/general/230120_j0251v4_ct_dist2matrix_mcl_%i_dt_%i_%s" % (
        min_comp_len, dist_threshold, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get distribution of sit2matrxi for single vesicles', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, colors = %s" % (
            min_comp_len, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = load_pkl2obj("cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl")
    log.info("Step 1/3: Load single vesicle info")
    ves_wd = 'cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/single_vesicles'
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')
    ves_dist2matrix = np.load(f'{ves_wd}/dist2matrix.npy')

    log.info("Step 2/3: Iterate over celltypes to get suitable cellids, filter vesicles")
    cts = list(ct_dict.keys())
    ax_ct = [1, 3, 4]
    cts_str = [ct_dict[ct] for ct in ct_dict]
    ves_density_all = pd.DataFrame(columns=cts_str)
    ves_density_close = pd.DataFrame(columns=cts_str)
    for i, ct in enumerate(tqdm(cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        if ct in ax_ct:
            cell_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/ax_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=True, max_path_len=None)
        else:
            cell_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_arr.pkl" % ct_dict[ct])
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = load_pkl2obj('cajal/nvmescratch/users/arother/j0251v4_prep/pot_astro_ids.pkl')
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info('Prefilter vesicles for celltype')
        ct_ind = np.in1d(ves_map2ssvids, cellids)
        ct_ves_ids = single_ves_ids[ct_ind]
        ct_ves_map2ssvids = ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ves_dist2matrix[ct_ind]
        ct_ves_coords = single_ves_coords[ct_ind]
        assert len(np.unique(ct_ves_map2ssvids)) == len(cellids)
        log.info('Iterate over cells to get vesicles associated to axon')
        #get axon_pathlength for corrensponding cellids
        axon_pathlengths = np.zeros(len(cellids))
        for c, cellid in enumerate(tqdm(cellids)):
            axon_pathlengths[c] = cell_dict[cellid]['axon length']

        cell_inputs = [[cellids[i], ct_ves_coords, ct_ves_map2ssvids, ct_ves_dist2matrix, dist_threshold, axon_pathlengths[i]] for i in range(len(cellids))]
        outputs = start_multiprocess_imap(get_ves_distance_per_cell, cell_inputs)
        outputs = np.array(outputs)
        ct_ves_number = outputs[:, 0]
        ct_ves_number_close = outputs[:, 1]
        ct_ves_density = outputs[:, 2]
        ct_ves_density_close = outputs[:, 3]
        ves_density_all[ct_dict[ct]] = ct_ves_density
        ves_density_close[ct_dict[ct]] = ct_ves_density_close

    log.info('Step 3/3: Plot results')
    ves_density_all.to_csv(f'{f_name}/ves_density_all.csv')
    ves_density_close.to_csv(f'{f_name}/ves_density_close_{dist_threshold}nm.csv')
    ct_palette = cls.ct_palette(color_key, num=False)
    sns.boxplot(ves_density_all, palette=ct_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title('Number of vesicles per axon pathlength')
    plt.savefig(f'{f_name}/all_ves_box.svg')
    plt.close()
    sns.boxplot(ves_density_close, palette=ct_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title(f'Number of vesicles closer than {dist_threshold} to membrane per axon pathlength')
    plt.savefig(f'{f_name}/close_ves_{dist_threshold}nm_box.svg')
    plt.close()

    log.info(f'Analysis for vesicles closer to {dist_threshold}nm in all celltypes done')




