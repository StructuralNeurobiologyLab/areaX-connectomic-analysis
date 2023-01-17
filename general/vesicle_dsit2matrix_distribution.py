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
    if distance_threshold is not None:
        cell_axo_ves_coords = cell_axo_ves_coords[cell_axo_dist2matrix < distance_threshold]
    number_vesicles = len(cell_axo_ves_coords)
    #calculate density
    axon_pathlength =

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ComparingMultipleForPLotting
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

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    cls = CelltypeColors()
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/general/230117_j0251v4_ct_dist2matrix_mcl_%i_%s" % (
        min_comp_len, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get distribution of sit2matrxi for single vesicles', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, colors = %s" % (
            min_comp_len, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = load_pkl2obj("cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl")
    log.info("Step 1/X: Load single vesicle info")
    ves_wd = 'cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/single_vesicles'
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')
    ves_dist2matrix = np.load(f'{ves_wd}/dist2matrix.npy')

    log.info("Step 2/X: Iterate over celltypes to get suitable cellids")
    cts = list(ct_dict.keys())
    ax_ct = [1, 3, 4]
    suitable_ids_dict = {}
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
        suitable_ids_dict[ct] = cellids
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info('Prefilter vesicles for celltype')
        ct_ind = np.in1d(ves_map2ssvids, cellids)
        ct_ves_ids = single_ves_ids[ct_ind]
        ct_ves_map2ssvids = ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ves_dist2matrix[ct_ind]
        ct_ves_coords = single_ves_coords[ct_ind]
        assert len(np.unique(ct_ves_map2ssvids)) == len(cellids)
        log.info('Iterate over cells to get vesicles associated to axon')
        #for example cell
        #multiprocess this part
        cellid = cellids[20]
        cell = SuperSegmentationObject(cellid)
        cell.load_skeleton()
        cell_ves_ind = np.in1d(ct_ves_map2ssvids, cellid)
        cell_ves_ids = ct_ves_ids[cell_ves_ind]
        cell_ves_coords = ct_ves_coords[cell_ves_ind]
        cell_dist2matrix = ct_ves_dist2matrix[cell_ves_ind]
        kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
        close_node_ids = kdtree.query(cell_ves_coords * cell.scaling, k=1)[1].astype(int)
        axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
        axo[axo == 3] = 1
        axo[axo == 4] = 1
        cell_axo_ves_ids = cell_ves_ids[axo == 1]
        cell_axo_ves_coords = cell_ves_coords[axo == 1]
        cell_axo_dist2matrix = cell_dist2matrix[axo == 1]
        cell_axo_ves_coords_dist = cell_axo_ves_coords[cell_axo_dist2matrix < 10]

        raise ValueError


