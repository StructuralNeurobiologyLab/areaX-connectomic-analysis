
#file to create a list with random vesicle coordinates for evaluation and checking
#information to save: celltype, cellid, distance to membrane, distance to synapse



if __name__ == '__main__':
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import \
        get_vesicle_distance_information_per_cell
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    from sklearn.utils import shuffle

    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    start = time.time()
    version = 'v5'
    bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    dist_thresholds = [5, 10, 15]  # nm
    min_syn_size = 0.1
    syn_prob_thresh = 0.8
    syn_dist_threshold_close = 500 #nm
    syn_dist_threshold_far = 5000 #nm
    size_dist_cat = 5
    size_syn_cat = 3
    gt_version = 'v7'
    color_key = 'TePkBr'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/231218_j0251{version}_ct_random_ves_eval_mcl_%i_dt_%i_st_%i_%i_{gt_version}gt" % (
        min_comp_len, dist_thresholds[-1], syn_dist_threshold_close, syn_dist_threshold_far)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('select random subset of single vesicles for evaluation',
                             log_dir=f_name + '/logs/')
    log.info(
        f"min_comp_len = %i, min_syn_size = %.1f, syn_prob_thresh = %.1f, distance threshold to membrane = %s nm, "
        f"distance threshold to synapse = {dist_thresholds} nm, distance threshold for not at synapse = %i nm, colors = %s" % (
            min_comp_len, min_syn_size, syn_prob_thresh, syn_dist_threshold_close, syn_dist_threshold_far,
            color_key))
    log.info(f'Select random vesicles from gt cells, from each celltype get {size_dist_cat} per distance to membrane category \n'
             f'and {size_syn_cat} that are within distance to membrane but in two different distance categories per synapse. \n'
             'Save in table, select randomly and shuffle.')
    log.info(f'GT version is {gt_version}')

    log.info('Step 1/5: Load cellids and celltypes from groundtruth')
    cts = list(ct_dict.keys())
    num_cts = len(cts)
    cts_str = bio_params.ct_str(with_glia=False)
    celltype_gt = pd.read_csv(
        f"cajal/nvmescratch/projects/data/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_{gt_version}_j0251_72_seg_20210127_agglo2_IDs.csv",
        names=["cellids", "celltype"])
    celltype_gt_gt = celltype_gt[np.in1d(celltype_gt['celltype'], cts_str)]
    cellids = np.array(celltype_gt['cellids'])
    ct_gt = np.array(celltype_gt['celltype'])

    log.info('Step 2/5: Filter synapse caches')
    #filtering taken from filter_synapse_caches in analysis_conn_helper but not filtering per celltype
    syn_prob = sd_synssv.load_numpy_data("syn_prob")
    m = syn_prob > syn_prob_thresh
    m_ids = sd_synssv.ids[m]
    m_axs = sd_synssv.load_numpy_data("partner_axoness")[m]
    m_axs[m_axs == 3] = 1
    m_axs[m_axs == 4] = 1
    m_cts = sd_synssv.load_numpy_data("partner_celltypes")[m]
    m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")[m]
    m_sizes = sd_synssv.load_numpy_data("mesh_area")[m] / 2
    m_spiness = sd_synssv.load_numpy_data("partner_spiness")[m]
    m_rep_coord = sd_synssv.load_numpy_data("rep_coord")[m]
    #filter min_syn_size
    size_inds = m_sizes > min_syn_size
    m_cts = m_cts[size_inds]
    m_ids = m_ids[size_inds]
    m_axs = m_axs[size_inds]
    m_ssv_partners = m_ssv_partners[size_inds]
    m_sizes = m_sizes[size_inds]
    m_spiness = m_spiness[size_inds]
    m_rep_coord = m_rep_coord[size_inds]
    #filter axo-dendritic
    axs_inds = np.any(m_axs == 1, axis=1)
    m_cts = m_cts[axs_inds]
    m_ids = m_ids[axs_inds]
    m_axs = m_axs[axs_inds]
    m_ssv_partners = m_ssv_partners[axs_inds]
    m_sizes = m_sizes[axs_inds]
    m_spiness = m_spiness[axs_inds]
    m_rep_coord = m_rep_coord[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    m_spiness = m_spiness[den_so_inds]
    m_rep_coord = m_rep_coord[den_so_inds]
    ct_inds = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
    comp_inds = np.in1d(m_axs, 1).reshape(len(m_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    syn_coords = m_rep_coord[filtered_inds]
    syn_axs = m_axs[filtered_inds]
    syn_ssv_partners = m_ssv_partners[filtered_inds]

    log.info('Step 3/5: Load single vesicle information and filter for gt cellids')
    ves_wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_organell_seg/231216_single_vesicles_slurm/'
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')
    ves_dist2matrix = np.load(f'{ves_wd}/dist2matrix.npy')
    ct_ind = np.in1d(ves_map2ssvids, cellids)
    ves_ids = single_ves_ids[ct_ind]
    ves_map2ssvids = ves_map2ssvids[ct_ind]
    ves_dist2matrix = ves_dist2matrix[ct_ind]
    ves_coords = single_ves_coords[ct_ind]

    #prepare input to give for saving vesicle information from cells
    log.info('Step 4/5: Filter single vesicles per cell and save per vesicle information')
    cell_inputs = [
        [cellids[i], ves_coords, ves_map2ssvids, ves_dist2matrix, syn_coords, syn_axs,
         syn_ssv_partners, ct_gt[i]] for i in range(len(cellids))]
    #write information about each vesicle into DataFrame
    outputs = start_multiprocess_imap(get_vesicle_distance_information_per_cell, cell_inputs)
    #merge Dataframes
    ct_result_df = pd.concat(outputs)

    log.info('Step 5/5: For each celltype randomly select single vesicles for evaluation')
    #for each celltype
    #get 5 vesicles per distance category
    #get 3 vesicles close to synapse and close to vesicle, 3 far away from synapse and close to vesicle
    random_selected_vesicles = []
    np.random.seed(42)
    for ct in tqdm(cts):
        ct_str = ct_dict[ct]
        ct_df = ct_result_df[ct_result_df['celltype'] == ct_str]
        #sort dataframe into the different categories and get random vesicles
        #columns = ['cellid', 'celltype', 'ves coord x','ves coord y', 'ves coord z', 'dist 2 membrane', 'dist 2 synapse']
        # all vesicles far from membrane and randomly select vesicles
        ct_df_far_membrane = ct_df[ct_df['dist 2 membrane'] > dist_thresholds[-1]]
        random_inds_far = np.random.choice(range(len(ct_df_far_membrane)), size = size_dist_cat, replace=False)
        rnd_df_far_membrane = ct_df_far_membrane.iloc[random_inds_far]
        random_selected_vesicles.append(rnd_df_far_membrane)
        #get all vesicles close to membrane
        ct_df_close_membrane = ct_df[ct_df['dist 2 membrane'] <= dist_thresholds[-1]]
        #close to membrane and close to synapse
        ct_df_close_membrane_syn = ct_df_close_membrane[ct_df_close_membrane['dist 2 synapse'] <= syn_dist_threshold_close]
        random_inds_close_syn = np.random.choice(range(len(ct_df_close_membrane_syn)), size=size_syn_cat, replace=False)
        rnd_df_close_membrane_syn = ct_df_close_membrane_syn.iloc[random_inds_close_syn]
        random_selected_vesicles.append(rnd_df_close_membrane_syn)
        #close to membrane and far from synapse
        ct_df_close_membrane_nonsyn = ct_df_close_membrane[ct_df_close_membrane['dist 2 synapse'] >= syn_dist_threshold_close]
        random_inds_close_nonsyn = np.random.choice(range(len(ct_df_close_membrane_nonsyn)), size=size_syn_cat, replace=False)
        rnd_df_close_membrane_nonsyn = ct_df_close_membrane_nonsyn.iloc[random_inds_close_nonsyn]
        random_selected_vesicles.append(rnd_df_close_membrane_nonsyn)
        #iterate over distance thresholds and randomly choose within them
        for di, dist in enumerate(dist_thresholds):
            if di == 0:
                df_dist = ct_df_close_membrane[ct_df_close_membrane['dist 2 membrane'] <= dist]
                random_inds_dist = np.random.choice(range(len(df_dist)), size = size_dist_cat, replace=False)
                rnd_df_dist = df_dist.iloc[random_inds_dist]
                random_selected_vesicles.append(rnd_df_dist)
            else:
                df_dist = ct_df_close_membrane[ct_df_close_membrane['dist 2 membrane'] <= dist]
                df_dist = df_dist[df_dist['dist 2 membrane'] > dist_thresholds[di  - 1]]
                random_inds_dist = np.random.choice(range(len(df_dist)), size=size_dist_cat, replace=False)
                rnd_df_dist = df_dist.iloc[random_inds_dist]
                random_selected_vesicles.append(rnd_df_dist)

    random_selected_ves_df = pd.concat(random_selected_vesicles)
    random_selected_ves_df = shuffle(random_selected_ves_df)

    random_selected_ves_df.to_csv(f'{f_name}/random_selected_vesicles.csv')
    log.info('Randomly selecting vesicles for evaluation done')
