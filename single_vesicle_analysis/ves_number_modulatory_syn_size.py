#analyses to compare small and large synapses and if there are more or less vesicles in vicinity
#small: lowest quantile of mesh area
#large: highest quantile of mesh area
#add option to use only non-synaptic vesicles
#independent of close or far from membrane

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, filter_synapse_caches_general
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_non_synaptic_vesicle_coords
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums
    from scipy.spatial import KDTree
    from tqdm import tqdm

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    min_comp_len = 200
    full_cell = True
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    nonsyn_dist_threshold = 3  # nm
    release_thresh = 5 #µm
    celltype = 0
    ct_str = ct_dict[celltype]
    fontsize = 20
    suitable_ids_only = True
    #spiness is list of spiness values that should be selected
    #spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    spiness = None
    #pre and post_cts is list of cell type numbers to be filtered for in synapses
    #if selected glia synapses will be filtered out automatically
    pre_cts = [2]
    post_cts = [3]
    #number of samples for each bootstrapping iteration to determine statistics
    bootstrap_n = 1000
    #number of iterations for bootstrapping
    n_it = 1000

    if pre_cts is None and post_cts is not None:
        raise ValueError('to select a postsynaptic cell type you need to select a presynaptic cell type also')
    if nonsyn_dist_threshold is None:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250103_j0251{version}_{ct_str}_ves_num_syn_size_modulatory_%i_r%i_HVC_MSN_it{n_it}_bn{bootstrap_n}" % (
            min_comp_len, release_thresh)
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250103_j0251{version}_{ct_str}_ves_num_syn_size_modulatory_%i_syn%i_r%i_HVC_MSN_it{n_it}_bn{bootstrap_n}" % (
            min_comp_len, nonsyn_dist_threshold, release_thresh)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'ves_num_syn_size_mod_{ct_str}', log_dir=f_name)
    log.info("min_comp_len = %i" % (min_comp_len))
    log.info(f'min syn size = {min_syn_size} µm², syn prob thresh = {syn_prob_thresh}, '
             f'threshold to putative release site = {release_thresh} µm, non syn dist threshold = {nonsyn_dist_threshold} nm')
    log.info(f'{bootstrap_n} values used for bootstrapping statistics, with {n_it} iterations')
    if full_cell:
        log.info('Only full cells will be processed')
    if with_glia:
        log.info('Putative glia synapses are not filtered out')
    else:
        log.info('Putative glia synapses are filtered out')
    if suitable_ids_only:
        id_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                     '240524_j0251v6_ct_morph_analyses_mcl_200_ax200_TeBKv6MSNyw_fs20npca2_umap5/ct_morph_df.csv'
        log.info(f'Only suitable ids will be used, loaded from {id_path}')
        loaded_morph_df = pd.read_csv(id_path, index_col=0)
        suitable_ids = np.array(loaded_morph_df['cellid'])
        suitable_cts = np.array(loaded_morph_df['celltype'])
        rev_ct_dict = {ct_str: ct for ct, ct_str in ct_dict.items()}
        all_cts = [rev_ct_dict[ct_str] for ct_str in suitable_cts]
        all_cts = np.unique(all_cts)
        log.info(f'{len(suitable_ids)} cells were selected to be suitable for analysis')
    else:
        ssd = SuperSegmentationDataset(working_dir=global_params.wd)
        suitable_ids = ssd.ssv_ids
        suitable_cts = ssd.load_numpy_data('celltype_pts_e3')
        all_cts = np.unique(suitable_cts)
        suitable_cts = [ct_dict[ct] for ct in suitable_cts]
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    #remove suitable ids from ct that is looked at to avoid over-representation of own synapses
    suitable_ids = suitable_ids[suitable_cts != ct_str]
    suitable_cts = suitable_cts[suitable_cts != ct_str]

    known_mergers = analysis_params.load_known_mergers()
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    cache_name = analysis_params.file_locations

    log.info('Step 1/5: Filter suitable cellids')
    cell_dict = analysis_params.load_cell_dict(celltype)
    # get ids with min compartment length
    cellids = np.array(list(cell_dict.keys()))
    merger_inds = np.in1d(cellids, known_mergers) == False
    cellids = cellids[merger_inds]
    #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
    #cellids = cellids[astro_inds]
    axon_cts = analysis_params.axon_cts()
    if full_cell and celltype not in axon_cts:
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
    else:
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                        axon_only=True,
                                        max_path_len=None)
    cellids = np.sort(cellids)
    log.info(f'{len(cellids)} {ct_str} cells fulfill criteria')

    log.info('Step 2/5: Get all vesicle coordinates from these cells')
    # load caches prefiltered for celltype
    if celltype in axon_cts:
        ct_ves_ids = np.load(f'{cache_name}/{ct_str}_ids.npy')
        ct_ves_coords = np.load(f'{cache_name}/{ct_str}_rep_coords.npy')
        ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_str}_mapping_ssv_ids.npy')
    else:
        ct_ves_ids = np.load(f'{cache_name}/{ct_str}_ids_fullcells.npy')
        ct_ves_coords = np.load(f'{cache_name}/{ct_str}_rep_coords_fullcells.npy')
        ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_str}_mapping_ssv_ids_fullcells.npy')
        ct_ves_axoness = np.load(f'{cache_name}/{ct_str}_axoness_coarse_fullcells.npy')
    # filter for selected cellids
    ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
    ct_ves_ids = ct_ves_ids[ct_ind]
    ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
    ct_ves_coords = ct_ves_coords[ct_ind]
    if celltype not in axon_cts:
        ct_ves_axoness = ct_ves_axoness[ct_ind]
        # make sure for full cells vesicles are only in axon
        ax_ind = np.in1d(ct_ves_axoness, 1)
        ct_ves_ids = ct_ves_ids[ax_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ax_ind]
        ct_ves_coords = ct_ves_coords[ax_ind]
    assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
    all_ves_number = len(ct_ves_ids)
    log.info(f'There are {all_ves_number} vesicles in {ct_str} cells.')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    if nonsyn_dist_threshold is not None:
        log.info(f'Use only non-synaptic synapses with threshold {nonsyn_dist_threshold} nm')
        # get synapses outgoing from this celltype
        ct_syn_cts, ct_syn_ids, ct_syn_axs, ct_syn_ssv_partners, ct_syn_sizes, ct_syn_spiness, ct_syn_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[celltype],
            post_cts=None,
            syn_prob_thresh=syn_prob_thresh,
            min_syn_size=min_syn_size,
            axo_den_so=True,
            synapses_caches=None, sd_synssv=sd_synssv)
        # filter so that only filtered cellids are included and are all presynaptic
        ct_inds = np.in1d(ct_syn_ssv_partners, cellids).reshape(len(ct_syn_ssv_partners), 2)
        comp_inds = np.in1d(ct_syn_axs, 1).reshape(len(ct_syn_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        ct_syn_coords = ct_syn_rep_coord[filtered_inds]
        ct_syn_axs = ct_syn_axs[filtered_inds]
        ct_syn_ssv_partners = ct_syn_ssv_partners[filtered_inds]
        # get non-synaptic vesicles
        cell_input = [[cellid, ct_ves_ids, ct_ves_coords, ct_ves_map2ssvids, ct_syn_coords, nonsyn_dist_threshold] for
                      cellid in cellids]
        cell_output = start_multiprocess_imap(get_non_synaptic_vesicle_coords, cell_input)
        cell_output = np.array(cell_output, dtype='object')
        # output still list of arrays
        non_syn_ves_ids = cell_output[:, 0]
        non_syn_ves_coords = cell_output[:, 1]
        # get all ves ids
        non_syn_ves_ids_con = np.concatenate(non_syn_ves_ids)
        non_syn_ves_coords_con = np.concatenate(non_syn_ves_coords)
        log.info(
            f'{len(non_syn_ves_ids_con)} vesicles are non-synaptic ({100 * len(non_syn_ves_ids_con) / len(ct_ves_ids):.2f} %)')
        ct_ves_coords = non_syn_ves_coords_con


    log.info('Step 3/5: Get suitable synapses')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    # filter for min syn size and syn prob to get all suitable synapses in dataset
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size)
    synapse_caches = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    if pre_cts is not None:
        pre_ct_strs = [ct_dict[ct] for ct in pre_cts]
        if post_cts is not None:
            post_ct_strs = [ct_dict[ct] for ct in post_cts]
        else:
            post_ct_strs = post_cts
        log.info(f' Synapses are only selected between the following pre-synaptic celltypes {pre_ct_strs} and postsynptic cell types {post_ct_strs}')
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=pre_cts,
            post_cts=post_cts,
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches, sd_synssv=None)
    else:
        # make sure only axo-dendritic or axo-somatic synapses
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        m_spiness = m_spiness[axs_inds]
        m_rep_coord = m_rep_coord[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
        m_cts = m_cts[den_so_inds]
        m_axs = m_axs[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        m_spiness = m_spiness[den_so_inds]
        m_rep_coord = m_rep_coord[den_so_inds]
        # if with_glia = False, filter out all synapses not between neurons
        if not with_glia:
            neuron_inds = np.all(np.in1d(m_cts, all_cts).reshape(len(m_cts), 2), axis=1)
            m_sizes = m_sizes[neuron_inds]
            m_ssv_partners = m_ssv_partners[neuron_inds]
            m_spiness = m_spiness[neuron_inds]
            m_rep_coord = m_rep_coord[neuron_inds]
    if suitable_ids_only:
        suit_inds = np.all(np.in1d(m_ssv_partners, suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
        m_sizes = m_sizes[suit_inds]
        m_spiness = m_spiness[suit_inds]
        m_rep_coord = m_rep_coord[suit_inds]
    #select only synapses on spine head and spine shaft
    if spiness is not None:
        spiness_dict = {0: 'spine neck', 1: ' spine head', 2: 'dendritic shaft', 3: 'other'}
        log.info(f'Synapses will only be selected with the following spiness values {[spiness_dict[sp] for sp in spiness]}')
        spine_inds = np.any(np.in1d(m_spiness, spiness).reshape(len(m_spiness), 2), axis = 1)
        m_sizes = m_sizes[spine_inds]
        m_rep_coord = m_rep_coord[spine_inds]
    log.info(f'{len(m_sizes)} synapses are selected for analysis.')
    #split synapses into quantiles, use lowest quantile for small synapses
    #highest quantile for large synapses
    lowest_quantile = np.quantile(m_sizes, 0.25)
    highest_quantile = np.quantile(m_sizes, 0.75)
    small_syn_inds = m_sizes <= lowest_quantile
    small_syn_coords = m_rep_coord[small_syn_inds]
    large_syn_inds = m_sizes >= highest_quantile
    large_syn_coords = m_rep_coord[large_syn_inds]
    log.info(f' The lowest quantile (0.25) of syn area is at {lowest_quantile:.2f} µm². {len(small_syn_coords)} synapses are <= this number '
             f'and will be selected as small synapses.')
    log.info(
        f' The highest quantile (0.75) of syn area is at {highest_quantile:.2f} µm². {len(large_syn_coords)} synapses are >= this number '
        f'and will be selected as large synapses.')

    log.info(f'Step 4/5: Get number of vesicles in {release_thresh} µm distance around synapses.')
    scaling = global_params.config['scaling']
    small_syns_tree = KDTree(small_syn_coords * scaling)
    large_syns_tree = KDTree(large_syn_coords * scaling)
    #get all vesicles that are within a max of release_thresh dist and calculate distance (nm)
    #then use this to count number in each distance
    #vesicles are only counted once, so if two synapses are close to same vesicles, it is only counted once with the closest distance
    ves_dists_small, small_syn_inds = small_syns_tree.query(ct_ves_coords * scaling, distance_upper_bound=release_thresh * 1000)
    ves_dists_large, large_syn_inds = large_syns_tree.query(ct_ves_coords * scaling,
                                                            distance_upper_bound=release_thresh * 1000)
    #remove all vesicles that are not in close range to any small or large synapse
    ves_dists_small = ves_dists_small[ves_dists_small < np.inf]
    ves_dists_large = ves_dists_large[ves_dists_large < np.inf]
    num_small_close_ves = len(ves_dists_small)
    log.info(f'{num_small_close_ves} vesicles are within {release_thresh} µm to small synapses, '
             f'{len(ves_dists_large)} vesicles to large synapses.')
    #make dataframe with all vesicles
    num_all_ves = num_small_close_ves + len(ves_dists_large)
    ves_dist_df = pd.DataFrame(columns = ['dist 2 syn', 'distance bin', 'synapse type'], index = range(num_all_ves))
    ves_dist_df.loc[0: num_small_close_ves - 1, 'dist 2 syn'] = ves_dists_small/ 1000 #in µm
    ves_dist_df.loc[0: num_small_close_ves - 1, 'synapse type'] = 'small'
    ves_dist_df.loc[num_small_close_ves: num_all_ves -1, 'dist 2 syn'] = ves_dists_large / 1000 #in µm
    ves_dist_df.loc[num_small_close_ves: num_all_ves -1, 'synapse type'] = 'large'
    #sort into distance bins with 200 µm
    dist_cat_bins = np.arange(0, release_thresh + 0.2, 0.2)
    dist_cat_labels = dist_cat_bins[1:]
    ves_dist_cats = np.array(pd.cut(ves_dist_df['dist 2 syn'], dist_cat_bins, right=False, labels=dist_cat_labels))
    ves_dist_df['distance bin'] = ves_dist_cats
    #save only if not too many vesicles due to disc storage
    if len(ves_dist_df) < 100000:
        ves_dist_df.to_csv(f'{f_name}/vesicle_distance_df.csv')

    log.info('Step 5/5: Get overview statistics and plot results')
    # do ranksums for statistics
    syn_type_groups = ves_dist_df.groupby('synapse type')
    stats, p_value = ranksums(np.array(syn_type_groups.get_group('small')['dist 2 syn']),
                              np.array(syn_type_groups.get_group('large')['dist 2 syn']))
    log.info(f'Ranksums result on distances: stats = {stats:.2f}, p-value = {p_value}.')
    #get number of vesicles in each distance bin
    num_cats = len(dist_cat_labels)
    overview_df = pd.DataFrame(columns = ['number vesicles', 'distance bin', 'synapse type'], index = range(num_cats * 2))
    overview_df.loc[0 : num_cats - 1, 'distance bin'] = dist_cat_labels
    overview_df.loc[0 : num_cats -1, 'synapse type'] = 'small'
    small_ves_hist = np.histogram(syn_type_groups.get_group('small')['dist 2 syn'], bins=dist_cat_bins)
    overview_df.loc[0 : num_cats - 1, 'number vesicles'] = small_ves_hist[0]
    overview_df.loc[num_cats: 2*num_cats - 1, 'distance bin'] = dist_cat_labels
    overview_df.loc[num_cats: 2*num_cats - 1, 'synapse type'] = 'small'
    large_ves_hist = np.histogram(syn_type_groups.get_group('large')['dist 2 syn'], bins=dist_cat_bins)
    overview_df.loc[num_cats: 2*num_cats - 1, 'number vesicles'] = large_ves_hist[0]
    overview_df.to_csv(f'{f_name}/hist_numbers_ov.csv')

    #plot results showing number of vesicles and dist 2 synapse in bins on x-axis
    size_palette = {'small': '#232121', 'large': '#15AEAB'}
    sns.histplot(x='dist 2 syn', data=ves_dist_df, hue='synapse type', palette=size_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, bins=dist_cat_bins)
    plt.ylabel('number of vesicles', fontsize=fontsize)
    plt.xlabel('distance to synapse [µm]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('vesicle distance to closest synapse')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_cats.png')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_cats.svg')
    plt.close()
    sns.histplot(x='dist 2 syn', data=ves_dist_df, hue='synapse type', palette=size_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, bins=dist_cat_bins, stat = 'percent')
    plt.ylabel('% of vesicles', fontsize=fontsize)
    plt.xlabel('distance to synapse [nm]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('vesicle distance to closest synapse')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_cats_perc.png')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_cats_perc.svg')
    plt.close()
    sns.histplot(x='dist 2 syn', data=ves_dist_df, hue='synapse type', palette=size_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel('number of vesicles', fontsize=fontsize)
    plt.xlabel('distance to synapse [µm]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('vesicle distance to closest synapse')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist.png')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist.svg')
    plt.close()
    sns.histplot(x='dist 2 syn', data=ves_dist_df, hue='synapse type', palette=size_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of vesicles', fontsize=fontsize)
    plt.xlabel('distance to synapse [µm]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('vesicle distance to closest synapse')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_perc.png')
    plt.savefig(f'{f_name}/num_ves_dist2syn_hist_perc.svg')
    plt.close()

    #high sample size tends to skew p-values towards smaller numbers
    #bootstrap statistics with several iterations of calculating the p-value of a random sample
    #then use mean of p-value
    log.info(f'Get p-value with bootstrapping {bootstrap_n} samples over {n_it} iterations')
    p_values_boot = np.empty(n_it)
    stats_boot = np.empty(n_it)
    small_syn_dists = np.array(ves_dist_df['dist 2 syn'][ves_dist_df['synapse type'] == 'small'])
    large_syn_dists = np.array(ves_dist_df['dist 2 syn'][ves_dist_df['synapse type'] == 'large'])
    for i in tqdm(range(n_it)):
        #draw random sample of small and large syn distances
        rndm_small = np.random.choice(small_syn_dists, bootstrap_n, replace=False)
        rndm_large = np.random.choice(large_syn_dists, bootstrap_n, replace=False)
        #get p-value from subset
        stats, p_value = ranksums(rndm_small, rndm_large)
        stats_boot[i] = stats
        p_values_boot[i] = p_value
        #plot random samples for first three iterations as example
        if i < 3:
            rndm_df = pd.DataFrame(columns = ['synapse type', 'dist 2 syn'], index = range(bootstrap_n * 2))
            rndm_df.loc[0: bootstrap_n - 1, 'synapse type'] = 'small'
            rndm_df.loc[0: bootstrap_n - 1, 'dist 2 syn'] = rndm_small
            rndm_df.loc[bootstrap_n: 2*bootstrap_n - 1, 'synapse type'] = 'large'
            rndm_df.loc[bootstrap_n: 2*bootstrap_n - 1, 'dist 2 syn'] = rndm_large
            rndm_df.to_csv(f'{f_name}/rndm_samples_{bootstrap_n}_{i}.csv')
            sns.histplot(x='dist 2 syn', data=rndm_df, hue='synapse type', palette=size_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of vesicles', fontsize=fontsize)
            plt.xlabel('distance to synapse [µm]', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('vesicle distance to closest synapse')
            plt.savefig(f'{f_name}/num_ves_dist2syn_hist_{bootstrap_n}_{i}.png')
            plt.savefig(f'{f_name}/num_ves_dist2syn_hist_{bootstrap_n}_{i}.svg')
            plt.close()
            sns.histplot(x='dist 2 syn', data=rndm_df, hue='synapse type', palette=size_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of vesicles', fontsize=fontsize)
            plt.xlabel('distance to synapse [µm]', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('vesicle distance to closest synapse')
            plt.savefig(f'{f_name}/num_ves_dist2syn_hist_perc_{bootstrap_n}_{i}.png')
            plt.savefig(f'{f_name}/num_ves_dist2syn_hist_perc_{bootstrap_n}_{i}.svg')
            plt.close()

    bootstrapped_stats = pd.DataFrame(columns = ['stats', 'p value'], index = range(n_it))
    bootstrapped_stats['stats'] = stats_boot
    bootstrapped_stats['p value'] = p_values_boot
    bootstrapped_stats.to_csv(f'{f_name}/bootstrapped_{bootstrap_n}_ranksum_values.csv')
    #plot p values
    sns.histplot(x='p value', data=bootstrapped_stats, color='black', common_norm=False,
                 fill=False, element="step", linewidth=3)
    plt.ylabel('number of iterations', fontsize=fontsize)
    plt.xlabel('p value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'bootstrapped p-values with {bootstrap_n} rndm samples')
    plt.savefig(f'{f_name}/p_values_bootstrapped_n{bootstrap_n}_it{n_it}.png')
    plt.savefig(f'{f_name}/p_values_bootstrapped_n{bootstrap_n}_it{n_it}.svg')
    plt.close()
    sns.histplot(x='p value', data=bootstrapped_stats, color='black', common_norm=False,
                 fill=False, element="step", linewidth=3, stat='percent')
    plt.ylabel('% of iterations', fontsize=fontsize)
    plt.xlabel('p-value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'bootstrapped p-values with {bootstrap_n} rndm samples')
    plt.savefig(f'{f_name}/p_values_bootstrapped_n{bootstrap_n}_it{n_it}_perc.png')
    plt.savefig(f'{f_name}/p_values_bootstrapped_n{bootstrap_n}_it{n_it}_perc.svg')
    plt.close()

    log.info(f' The mean p-value over {n_it} iterations with {bootstrap_n} samples each is: {np.mean(p_values_boot)}')

    log.info('Analysis finished.')