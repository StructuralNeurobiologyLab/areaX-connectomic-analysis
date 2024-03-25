#test if there is a dependency of number of vesicles per synapse size
#plot for each celltype, for complete cells, also once averaged per cell

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, filter_synapse_caches_general
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_ves_synsize_percell
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import ChainMap
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict()
    min_comp_len = 200
    dist_threshold = 15 #nm
    min_syn_size = 0.1
    syn_prob_thresh = 0.8
    syn_dist_threshold = 500 #nm
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/240320_j0251{version}_number_ves_synsize_mcl_%i_dt_%i_st_%i_%s" % (
        min_comp_len, dist_threshold, syn_dist_threshold, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Vesicle - synsize - dependency', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, min_syn_size = %.1f, syn_prob_thresh = %.1f, distance threshold to membrane = %s nm, "
        "distance threshold to synapse = %i nm, colors = %s" % (
            min_comp_len, min_syn_size, syn_prob_thresh, dist_threshold, syn_dist_threshold, color_key))

    log.info("Step 1/4: Load synapse segmentation dataset")
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    cache_name = analysis_params.file_locations
    known_mergers = load_pkl2obj(f"{cache_name}/merger_arr.pkl")

    cts = list(ct_dict.keys())
    ax_ct = analysis_params.axon_cts()
    num_cts = len(cts)
    cts_str = [ct_dict[i] for i in range(num_cts)]
    ct_palette = cls.ct_palette(color_key, num=False)
    result_df_list = []

    log.info('Step 2/3 Get information of vesicle number at the synapses')
    # prefilter synapses for synapse prob thresh and min syn size
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size)
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    for ct in tqdm(range(num_cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in ax_ct:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=True, max_path_len=None)
        else:
            if ct == 2:
                misclassified_asto_ids = load_pkl2obj(f'{cache_name}/pot_astro_ids.pkl')
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info('Prefilter synapses for celltype')
        #filter synapses to only have specific celltype
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct],
                                                                                                            post_cts=None,
                                                                                                            syn_prob_thresh=None,
                                                                                                            min_syn_size=None,
                                                                                                            axo_den_so=True,
                                                                                                            synapses_caches = synapse_cache)
        #filter so that only filtered cellids are included and are all presynaptic
        ct_inds = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
        comp_inds = np.in1d(m_axs, 1).reshape(len(m_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_coords = m_rep_coord[filtered_inds]
        syn_axs = m_axs[filtered_inds]
        syn_ssv_partners = m_ssv_partners[filtered_inds]
        syn_sizes = m_sizes[filtered_inds]
        log.info('Prefilter vesicles for celltype')
        #load caches prefiltered for celltype
        if ct in ax_ct:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix.npy')
        else:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids_fullcells.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords_fullcells.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids_fullcells.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix_fullcells.npy')
            ct_ves_axoness = np.load(f'{cache_name}/{ct_dict[ct]}_axoness_coarse_fullcells.npy')
        #filter for selected cellids
        ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
        ct_ves_ids = ct_ves_ids[ct_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ct_ves_dist2matrix[ct_ind]
        ct_ves_coords = ct_ves_coords[ct_ind]
        if ct not in ax_ct:
            ct_ves_axoness = ct_ves_axoness[ct_ind]
            # make sure for full cells vesicles are only in axon
            ax_ind = np.in1d(ct_ves_axoness, 1)
            ct_ves_ids = ct_ves_ids[ax_ind]
            ct_ves_map2ssvids = ct_ves_map2ssvids[ax_ind]
            ct_ves_dist2matrix = ct_ves_dist2matrix[ax_ind]
            ct_ves_coords = ct_ves_coords[ax_ind]
        assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
        log.info('Iterate over cells to get vesicles associated to axon, vesicle info for synapses')
        #prepare inputs for multiprocessing
        cell_inputs = [
            [cellids[i], ct_ves_coords, ct_ves_map2ssvids, ct_ves_dist2matrix, dist_threshold, syn_coords, syn_axs,
             syn_ssv_partners, syn_sizes, syn_dist_threshold] for i in range(len(cellids))]
        outputs = start_multiprocess_imap(get_ves_synsize_percell, cell_inputs)
        #output is a list of dataframes with the columns ['cellid', 'synapse size [µm²]', 'number of vesicles', 'number of close-membrane vesicles']
        ct_result_df = pd.concat(outputs)
        ct_result_df['celltype'] = ct_str
        result_df_list.append(ct_result_df)
        medians = ct_result_df.median(numeric_only=True)
        syn_size_median = medians['synapse size [µm²]']
        num_ves_median = medians['number of vesicles']
        close_num_ves_median = medians['number of membrane-close vesicles']
        log.info(f'In total {len(ct_result_df)} synapses were meeting the requirements in {ct_str} cells.')
        log.info(f'They have a median size of {syn_size_median:.2f} µm² with {num_ves_median} vesicles \n'
                 f'and {close_num_ves_median} membrane-close vesicles per synapse ({syn_dist_threshold} nm radius)')
        #plot per ct result

        sns.scatterplot(x = 'synapse size [µm²]', y = 'number of vesicles', data=ct_result_df, alpha = 0.5, color = ct_palette[ct_str])
        plt.title(f'Vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_allves_syns_scatter.png')
        plt.close()
        sns.scatterplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=ct_result_df, alpha=0.5,
                        color=ct_palette[ct_str])
        plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_closemem_ves_{dist_threshold}nm_syns_scatter.png')
        plt.close()
        sns.kdeplot(x='synapse size [µm²]', y='number of vesicles', data=ct_result_df,
                        color=ct_palette[ct_str])
        plt.title(f'Vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_allves_syns_kde.png')
        plt.close()
        sns.kdeplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=ct_result_df,
                        color=ct_palette[ct_str])
        plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_closemem_ves_{dist_threshold}nm_syns_kde.png')
        plt.close()
        sns.regplot(x='synapse size [µm²]', y='number of vesicles', data=ct_result_df, scatter_kws={'alpha':0.1},
                        color=ct_palette[ct_str])
        plt.title(f'Vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_allves_syns_reg.png')
        plt.close()
        sns.regplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=ct_result_df, scatter_kws={'alpha':0.1},
                        color=ct_palette[ct_str])
        plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_closemem_ves_{dist_threshold}nm_syns_reg.png')
        plt.close()


    log.info('Step 3/3: Plot results')
    combined_results = pd.concat(result_df_list)
    combined_results.to_csv(f'{f_name}/all_syns_ves.csv')
    #plot results for whole dataset
    sns.regplot(x='synapse size [µm²]', y='number of vesicles', data=combined_results,
                scatter_kws={'alpha': 0.1},
                color='black')
    plt.title(f'Vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_num_ves_{dist_threshold}nm_syns_regscatter.png')
    plt.close()
    sns.regplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=combined_results,
                scatter_kws={'alpha': 0.1},
                color='black')
    plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_closemem_ves_{dist_threshold}nm_syns_regscatter.png')
    plt.close()
    sns.regplot(x='synapse size [µm²]', y='number of vesicles', data=combined_results,
                scatter=False,
                color='black')
    plt.title(f'Vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_num_ves_{dist_threshold}nm_syns_reg.svg')
    plt.close()
    sns.regplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=combined_results,
                scatter=False,
                color='black')
    plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_closemem_ves_{dist_threshold}nm_syns_reg.svg')
    plt.close()
    #plots with different celltypes
    sns.lmplot(x='synapse size [µm²]', y='number of vesicles', data=combined_results,
                hue = 'celltype', scatter=False, palette=ct_palette)
    plt.title(f'Vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_cts_num_ves_{dist_threshold}nm_syns_lm.svg')
    plt.close()
    sns.lmplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=combined_results,
                hue = 'celltype', scatter=False, palette=ct_palette)
    plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_cts_closemem_ves_{dist_threshold}nm_syns_lm.svg')
    plt.close()
    sns.kdeplot(x='synapse size [µm²]', y='number of vesicles', data=combined_results,
               hue='celltype', scatter=False, palette=ct_palette)
    plt.title(f'Vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_cts_num_ves_{dist_threshold}nm_syns_kde.png')
    plt.close()
    sns.kdeplot(x='synapse size [µm²]', y='number of membrane-close vesicles', data=combined_results,
               hue='celltype', scatter=False, palette=ct_palette)
    plt.title(f'Membrane-close vesicle number and synapse size in {ct_str}')
    plt.savefig(f'{f_name}/all_cts_closemem_ves_{dist_threshold}nm_syns_kde.png')
    plt.close()
    #plot a scatter plot with synapse size and number of vesicles (color in celltypes)
    #scatter plot with synapse size and number of close vesicles (color for celltype)

    log.info(f' Analysis looking at relationships between number of vesicles per synapse and synapse size done')