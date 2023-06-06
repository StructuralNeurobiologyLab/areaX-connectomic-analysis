#analysis to see which fraction of vesicles close to membrane is close to the synapse

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_synapse_proximity_vesicle_percell
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
    analysis_params = Analysis_Params(working_dir = global_params.wd, version = 'v5')
    ct_dict = analysis_params.ct_dict()
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    dist_threshold = 10 #nm
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    syn_dist_threshold = 500 #nm
    nonsyn_dist_threshold = 5000 #nm
    cls = CelltypeColors()
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    f_name = "cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/230606_j0251v5_ct_syn_fraction_closemembrane_mcl_%i_ax%i_dt_%i_st_%i_%i_%s" % (
        min_comp_len_cell, min_comp_len_ax, dist_threshold, syn_dist_threshold, nonsyn_dist_threshold, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get fraction of vesicles close to membrane which are not close to synapse', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, min_syn_size = %.1f, syn_prob_thresh = %.1f, distance threshold to membrane = %s nm, "
        "distance threshold to synapse = %i nm, distance threshold for not at synapse = %i nm, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, min_syn_size, syn_prob_thresh, dist_threshold, syn_dist_threshold, nonsyn_dist_threshold, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = analysis_params.load_known_mergers()
    log.info("Step 1/4: synapse segmentation dataset")

    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    cache_name = analysis_params.file_locations

    log.info('Step 2/4: Prepare dataframes for results')

    cts = list(ct_dict.keys())
    ax_ct = [1, 3, 4]
    num_cts = len(cts)
    cts_str = [ct_dict[i] for i in range(num_cts)]
    ct_palette = cls.ct_palette(color_key, num=False)

    fraction_nonsyn_df = pd.DataFrame(columns=cts_str, index=range(10500))
    density_non_syn_df = pd.DataFrame(columns=cts_str, index=range(10500))
    density_syn_df = pd.DataFrame(columns=cts_str, index=range(10500))
    columns = ['celltype', 'fraction of non-synaptic membrane-close vesicles',
               'density of non_synaptic membrane-close vesicles',
               'density of synaptic membrane-close vesicles']
    median_values_df = pd.DataFrame(columns = columns, index=range(num_cts))
    median_plotting_df = pd.DataFrame(columns = ['celltype', 'vesicle density', 'location'], index=range(num_cts*2))

    log.info('Step 3/4 Get information for vesicles close to membrane and synapse for all celltypes')
    for ct in tqdm(range(num_cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if ct in ax_ct:
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
        log.info('Prefilter synapses for celltype')
        #filter synapses to only have specific celltype
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                            pre_cts=[
                                                                                                                ct],
                                                                                                            post_cts=None,
                                                                                                            syn_prob_thresh=syn_prob_thresh,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True)
        #filter so that only filtered cellids are included and are all presynaptic
        ct_inds = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
        comp_inds = np.in1d(m_axs, 1).reshape(len(m_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_coords = m_rep_coord[filtered_inds]
        syn_axs = m_axs[filtered_inds]
        syn_ssv_partners = m_ssv_partners[filtered_inds]
        log.info('Prefilter vesicles for celltype')
        #load caches prefiltered for celltype
        ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids.npy')
        ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords.npy')
        ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids.npy')
        ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix.npy')
        #filter for selected cellids
        ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
        ct_ves_ids = ct_ves_ids[ct_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ct_ves_dist2matrix[ct_ind]
        ct_ves_coords = ct_ves_coords[ct_ind]
        assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
        # get axon_pathlength for corrensponding cellids
        axon_pathlengths = np.zeros(len(cellids))
        for c, cellid in enumerate(tqdm(cellids)):
            axon_pathlengths[c] = cell_dict[cellid]['axon length']
        log.info('Iterate over cells to get vesicles associated to axon, vesicle info for synapses')
        #prepare inputs for multiprocessing
        cell_inputs = [
            [cellids[i], ct_ves_coords, ct_ves_map2ssvids, ct_ves_dist2matrix, dist_threshold, syn_coords, syn_axs,
             syn_ssv_partners, syn_dist_threshold, nonsyn_dist_threshold, axon_pathlengths[i]] for i in range(len(cellids))]
        outputs = start_multiprocess_imap(get_synapse_proximity_vesicle_percell, cell_inputs)
        outputs = np.array(outputs)
        fraction_non_syn_mem_vesicles = outputs[:, 0]
        density_non_syn_mem_vesicles = outputs[:, 1]
        density_syn_mem_vesicles = outputs[:, 2]
        fraction_nonsyn_df.loc[0:len(cellids) - 1, ct_str] = fraction_non_syn_mem_vesicles
        density_non_syn_df.loc[0:len(cellids) - 1, ct_str] = density_non_syn_mem_vesicles
        density_syn_df.loc[0:len(cellids) - 1, ct_str] = density_syn_mem_vesicles
        median_values_df.loc[ct, 'celltype'] = ct_str
        median_fraction = np.nanmedian(fraction_non_syn_mem_vesicles)
        median_nonsyn_density = np.nanmedian(density_non_syn_mem_vesicles)
        median_syn_density = np.nanmedian(density_syn_mem_vesicles)
        median_values_df.loc[ct, columns[1]] = median_fraction
        median_values_df.loc[ct, columns[2]] = median_nonsyn_density
        median_values_df.loc[ct, columns[3]] = median_syn_density
        median_plotting_df.loc[ct * 2: ct * 2 + 1, 'celltype'] = ct_str
        median_plotting_df.loc[ct * 2, 'vesicle density'] = median_nonsyn_density
        median_plotting_df.loc[ct * 2, 'location'] = 'non-synaptic'
        median_plotting_df.loc[ct * 2 + 1, 'vesicle density'] = median_syn_density
        median_plotting_df.loc[ct * 2 + 1, 'location'] = 'synaptic'
        log.info(f'{ct_str} cells have a median median fraction of membrane-close vesicles '
                 f'not at synapses of {median_fraction:.2f} ')

    log.info('Step 4/5 calculate statistics')
    stats_combinations = combinations(np.arange(num_cts), 2)
    columns = ['Fraction of non- synaptic vesicles', 'Density non-synaptic vesicles', 'Density synaptic vesicles']

    stats_results = pd.DataFrame(columns = columns, index = ['Kruskal'])


    log.info('Step 5/5: Plot results')
    fraction_nonsyn_df.to_csv(f'{f_name}/fraction_nonsyn_mem_vesicles.csv')
    density_non_syn_df.to_csv(f'{f_name}/density_nonsyn_mem_vesicles.csv')
    density_syn_df.to_csv(f'{f_name}/density_syn_mem_vesicles.csv')
    median_values_df.to_csv(f'{f_name}/median_values_mem_vesicles.csv')
    median_plotting_df.to_csv(f'{f_name}/median_values_den_plot_vesicles.csv')
    sns.boxplot(fraction_nonsyn_df, palette=ct_palette)
    plt.ylabel('fraction of non-synaptic vesicles')
    plt.title(f'Fraction of vesicles close to membrane and not close to synapse ({syn_dist_threshold} nm)')
    plt.savefig(f'{f_name}/fraction_nonsyn_mem_{dist_threshold}nm_{syn_dist_threshold}nm.svg')
    plt.close()
    sns.boxplot(density_non_syn_df, palette=ct_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title('Number of vesicles per axon pathlength close to membrane and not to synapse')
    plt.savefig(f'{f_name}/density_nonsyn_mem_{dist_threshold}nm_{syn_dist_threshold}nm.svg')
    plt.close()
    sns.boxplot(density_syn_df, palette=ct_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title('Number of vesicles per axon pathlength close to membrane and to synapse')
    plt.savefig(f'{f_name}/density_syn_mem_{dist_threshold}nm_{syn_dist_threshold}nm.svg')
    plt.close()
    palette = {'non-synaptic': 'black', 'synaptic': '#00BFB2' }
    sns.pointplot(x = 'celltype', y = 'vesicle density', data = median_plotting_df, hue='location', palette=palette, join=False)
    plt.ylabel('median vesicle density [1/µm]')
    plt.title('Median density of vesicles close to membrane')
    plt.savefig(f'{f_name}/mem_close_comb_median_point.svg')
    plt.close()

    log.info(f'Analysis for vesicles closer to {dist_threshold}nm, split into synaptic '
             f'and non-synaptic in all celltypes done')