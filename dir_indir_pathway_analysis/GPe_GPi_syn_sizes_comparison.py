#plot synapses from all GPe and GPi connections
#once whole cell to whole cells
#whole cells to and from everything
#all synapses independent of where from


if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper \
        import filter_synapse_caches_for_ct, get_percell_number_sumsize, filter_synapse_caches_general, get_ct_syn_number_sumsize
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_histogram_selection
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums
    from tqdm import tqdm

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    syn_prob_thresh = 0.6
    min_syn_size = 0.1
    # celltype that gives input or output
    # celltypes that are compared
    gpe_ct = 6
    gpi_ct = 7
    color_key = 'STNGP'
    fontsize_jointplot = 12
    comp_ax = 50
    comp_full = 200
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/231117_j0251v5_GPe_GPi_synsize_overview_synprob_%.2f_ax%i_full%i" % (
        syn_prob_thresh, comp_ax, comp_full)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe vs GPi syn size comparison', log_dir=f_name + '/logs/')
    ct_colors = CelltypeColors()
    ct_palette = ct_colors.ct_palette(key=color_key)
    log.info("Analysis of GPe and GPi syn sizes starts")
    log.info(f'syn_prob = {syn_prob_thresh}, min_syn_size = {min_syn_size} µm², \n'
             f'min comp length ax = {comp_ax} µm, min comp len full cells = {comp_full}')
    log.info('Step 1/4: filter synapses for syn_prob and min_syn_size and also axo-dendritic and axo-somatic only')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord, syn_prob = filter_synapse_caches_general(sd_synssv,
                                                                                                 syn_prob_thresh=syn_prob_thresh,
                                                                                                 min_syn_size=min_syn_size)
    #make sure only axo-dendritic or axo-somatic
    axs_inds = np.any(m_axs == 1, axis=1)
    m_cts = m_cts[axs_inds]
    m_axs = m_axs[axs_inds]
    m_ssv_partners = m_ssv_partners[axs_inds]
    m_sizes = m_sizes[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    #make sure synapses are between neurons only
    neuron_cts = list(ct_dict.keys())
    ct_inds = np.all(np.in1d(m_cts, neuron_cts).reshape(len(m_cts), 2), axis=1)
    m_cts = m_cts[ct_inds]
    m_axs = m_axs[ct_inds]
    m_ssv_partners = m_ssv_partners[ct_inds]
    m_sizes = m_sizes[ct_inds]
    #get all synapses which have either GPe or GPi as pre or post
    ct_inds = np.any(np.in1d(m_cts, [gpe_ct, gpi_ct]).reshape(len(m_cts), 2), axis = 1)
    m_cts = m_cts[ct_inds]
    m_axs = m_axs[ct_inds]
    m_ssv_partners = m_ssv_partners[ct_inds]
    m_sizes = m_sizes[ct_inds]

    log.info('Step 2/4: Get more info (pre, postcelltypes, full cell) about synapses and save in DataFrame')
    syn_columns = ['synapse size', 'celltype pre', 'celltype post', 'cellid pre', 'cellid post', 'full cell pre', 'full cell post']
    all_syns_df = pd.DataFrame(columns=syn_columns, index = range(len(m_sizes)))
    all_syns_df['synapse size'] = m_sizes
    ax_inds = np.where(m_axs == 1)
    denso_inds = np.where(m_axs != 1)
    ax_cts = m_cts[ax_inds]
    ax_ssv_partners = m_ssv_partners[ax_inds]
    denso_cts = m_cts[denso_inds]
    denso_ssv_partners = m_ssv_partners[denso_inds]
    all_syns_df['celltype pre'] = [ct_dict[ax_cts[i]] for i in range(len(ax_cts))]
    all_syns_df['celltype post'] = [ct_dict[denso_cts[i]] for i in range(len(denso_cts))]
    all_syns_df['cellid pre'] = ax_ssv_partners
    all_syns_df['cellid post'] = denso_ssv_partners
    gpe_str = ct_dict[gpe_ct]
    gpi_str = ct_dict[gpi_ct]

    log.info('Check cellids for full cells')
    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    num_cts = len(celltypes)
    axon_cts = analysis_params.axon_cts()
    suitable_ids_dict = {}
    all_suitable_ids = []
    known_mergers = analysis_params.load_known_mergers()
    misclassified_astros = analysis_params.load_potential_astros()
    for ct in tqdm(range(num_cts)):
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in axon_cts:
            # get ids with min compartment length
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict,
                                                    min_comp_len=comp_ax, axon_only=True,
                                                    max_path_len=None)
        else:
            astro_inds = np.in1d(cellids, misclassified_astros) == False
            cellids = cellids[astro_inds]
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict,
                                                    min_comp_len=comp_full,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    all_suitable_ids = np.concatenate(all_suitable_ids)
    ax_suit_inds = np.in1d(ax_ssv_partners, all_suitable_ids)
    denso_suit_inds = np.in1d(denso_ssv_partners, all_suitable_ids)
    all_syns_df.loc[ax_suit_inds, 'full cell pre'] = 1
    all_syns_df.loc[ax_suit_inds == False, 'full cell pre'] = 0
    all_syns_df.loc[denso_suit_inds, 'full cell post'] = 1
    all_syns_df.loc[denso_suit_inds == False, 'full cell post'] = 0
    all_syns_df.to_csv(f'{f_name}/all_syns_df_gpe_gpi.csv')

    log.info(f'{len(all_syns_df)} synapses are either with GPe or GPi')
    log.info('Step 3/4: Get information about all synapses, make statistics and plots')
    summary_cats = ['all syns', 'all syns pre', 'all syns post','full cells', 'full cells pre', 'full cells post']
    summary_df = pd.DataFrame(columns=['total number', 'number GPe', 'number GPi'], index = summary_cats)
    summary_df.loc['all syns', 'total number'] = len(all_syns_df)
    summary_df.loc['all syns', 'median'] = np.median(all_syns_df['synapse size'])
    gpe_syn_df = all_syns_df[np.any([all_syns_df['celltype pre'] == gpe_str,all_syns_df['celltype post'] == gpe_str ], axis = 0)]
    gpi_syn_df = all_syns_df[
        np.any([all_syns_df['celltype pre'] == gpi_str, all_syns_df['celltype post'] == gpi_str], axis=0)]
    summary_df.loc['all syns', 'number GPe'] = len(gpe_syn_df)
    summary_df.loc['all syns', 'number GPi'] = len(gpi_syn_df)
    summary_df.loc['all syns', 'median GPe'] = np.median(gpe_syn_df['synapse size'])
    summary_df.loc['all syns', 'median GPi'] = np.median(gpi_syn_df['synapse size'])
    gp_pre_syn_df = all_syns_df[np.in1d(all_syns_df['celltype pre'], [gpe_str, gpi_str])]
    gp_post_syn_df = all_syns_df[np.in1d(all_syns_df['celltype post'], [gpe_str, gpi_str])]
    gpe_pre_syn_df = all_syns_df[all_syns_df['celltype pre'] == gpe_str]
    gpi_pre_syn_df = all_syns_df[all_syns_df['celltype pre'] == gpi_str]
    gpe_post_syn_df = all_syns_df[all_syns_df['celltype post'] == gpe_str]
    gpi_post_syn_df = all_syns_df[all_syns_df['celltype post'] == gpi_str]
    summary_df.loc['all syns pre', 'total number'] = len(gpe_pre_syn_df) + len(gpi_pre_syn_df)
    summary_df.loc['all syns pre', 'median'] = np.median(gp_pre_syn_df['synapse size'])
    summary_df.loc['all syns pre', 'number GPe'] = len(gpe_post_syn_df)
    summary_df.loc['all syns pre', 'median GPe'] = np.median(gpi_post_syn_df['synapse size'])
    summary_df.loc['all syns pre', 'number GPi'] = len(gpi_pre_syn_df)
    summary_df.loc['all syns pre', 'median GPi'] = np.median(gpi_pre_syn_df['synapse size'])
    summary_df.loc['all syns post', 'total number'] = len(gpe_post_syn_df) + len(gpi_post_syn_df)
    summary_df.loc['all syns post', 'number GPe'] = len(gpe_post_syn_df)
    summary_df.loc['all syns post', 'number GPi'] = len(gpi_post_syn_df)
    summary_df.loc['all syns post', 'median'] = np.median(gp_post_syn_df['synapse size'])
    summary_df.loc['all syns post', 'median GPe'] = np.median(gpe_post_syn_df['synapse size'])
    summary_df.loc['all syns post', 'median GPi'] = np.median(gpi_post_syn_df['synapse size'])
    #ranksum test on size of synapses
    ranksum_results = pd.DataFrame(index = ['GPe vs GPi pre all syns', 'GPe vs GPi post all syns', 'GPe vs GPi pre full cells', 'GPe vs GPi post full cells'], columns = ['stats', 'p_value'])
    stats, p_value = ranksums(gpe_pre_syn_df['synapse size'], gpi_pre_syn_df['synapse size'])
    ranksum_results.loc['GPe vs GPi pre all syns', 'stats'] = stats
    ranksum_results.loc['GPe vs GPi pre all syns', 'stats'] = p_value
    stats, p_value = ranksums(gpe_post_syn_df['synapse size'], gpi_post_syn_df['synapse size'])
    ranksum_results.loc['GPe vs GPi post all syns', 'stats'] = stats
    ranksum_results.loc['GPe vs GPi post all syns', 'stats'] = p_value
    #plot histograms for all synapses where GP cells are pre
    plot_histogram_selection(dataframe=gp_pre_syn_df, x_data='synapse size', color_palette=ct_palette,
                             label = 'all_syns_pre', count = 'synapses',
                             foldername = f_name, hue_data='celltype pre',
                             title='all synapse sizes for presynaptic GPe and GPi')
    #plot histograms for all synapses where GP cells are post
    plot_histogram_selection(dataframe=gp_post_syn_df, x_data='synapse size', color_palette=ct_palette,
                             label='all_syns_post', count='synapses',
                             foldername=f_name, hue_data='celltype post',
                             title='all synapse sizes for postsynaptic GPe and GPi')

    log.info('Step 4/4: Get information about synapses only with filtered cells')
    #get synapses only between full cells
    full_pre_df = all_syns_df[all_syns_df['full cell pre'] == 1]
    full_cells_df = full_pre_df[full_pre_df['full cell post'] == 1]
    full_cells_df.to_csv(f'{f_name}/full_cells_syns_{gpe_str}_{gpi_str}_{comp_full}_{comp_ax}.csv')
    log.info(f'{len(full_cells_df)} synapses are made with filtered cells')
    gpe_syn_df = full_cells_df[
        np.any([full_cells_df['celltype pre'] == gpe_str, full_cells_df['celltype post'] == gpe_str], axis=0)]
    gpi_syn_df = full_cells_df[
        np.any([full_cells_df['celltype pre'] == gpi_str, full_cells_df['celltype post'] == gpi_str], axis=0)]
    summary_df.loc['full cells', 'number GPe'] = len(gpe_syn_df)
    summary_df.loc['full cells', 'number GPi'] = len(gpi_syn_df)
    summary_df.loc['full cells', 'median GPe'] = np.median(gpe_syn_df['synapse size'])
    summary_df.loc['full cells', 'median GPi'] = np.median(gpi_syn_df['synapse size'])
    gp_pre_syn_df = full_cells_df[np.in1d(full_cells_df['celltype pre'], [gpe_str, gpi_str])]
    gp_post_syn_df = full_cells_df[np.in1d(full_cells_df['celltype post'], [gpe_str, gpi_str])]
    gpe_pre_syn_df = full_cells_df[full_cells_df['celltype pre'] == gpe_str]
    gpi_pre_syn_df = full_cells_df[full_cells_df['celltype pre'] == gpi_str]
    gpe_post_syn_df = full_cells_df[full_cells_df['celltype post'] == gpe_str]
    gpi_post_syn_df = full_cells_df[full_cells_df['celltype post'] == gpi_str]
    summary_df.loc['full cells pre', 'total number'] = len(gpe_pre_syn_df) + len(gpi_pre_syn_df)
    summary_df.loc['full cells pre', 'number GPe'] = len(gpe_post_syn_df)
    summary_df.loc['full cells pre', 'number GPi'] = len(gpi_pre_syn_df)
    summary_df.loc['full cells post', 'total number'] = len(gpe_post_syn_df) + len(gpi_post_syn_df)
    summary_df.loc['full cells post', 'number GPe'] = len(gpe_post_syn_df)
    summary_df.loc['full cells post', 'number GPi'] = len(gpi_post_syn_df)
    summary_df.loc['full cells pre', 'median'] = np.median(gp_pre_syn_df['synapse size'])
    summary_df.loc['full cells pre', 'median GPe'] = np.median(gpe_post_syn_df['synapse size'])
    summary_df.loc['full cells pre', 'median GPi'] = np.median(gpi_pre_syn_df['synapse size'])
    summary_df.loc['full cells post', 'median'] = np.median(gp_post_syn_df['synapse size'])
    summary_df.loc['full cells post', 'median GPe'] = np.median(gpe_post_syn_df['synapse size'])
    summary_df.loc['full cells post', 'median GPi'] = np.median(gpi_post_syn_df['synapse size'])
    summary_df.to_csv(f'{f_name}/summary_stats.csv')
    # ranksum test on size of synapses
    stats, p_value = ranksums(gpe_pre_syn_df['synapse size'], gpi_pre_syn_df['synapse size'])
    ranksum_results.loc['GPe vs GPi pre full cells', 'stats'] = stats
    ranksum_results.loc['GPe vs GPi pre full cells', 'stats'] = p_value
    stats, p_value = ranksums(gpe_post_syn_df['synapse size'], gpi_post_syn_df['synapse size'])
    ranksum_results.loc['GPe vs GPi post full cells', 'stats'] = stats
    ranksum_results.loc['GPe vs GPi post full cells', 'stats'] = p_value
    # plot histograms for all synapses where GP cells are pre
    plot_histogram_selection(dataframe=gp_pre_syn_df, x_data='synapse size', color_palette=ct_palette,
                             label='full_cells_syns_pre', count='synapses',
                             foldername=f_name, hue_data='celltype pre',
                             title='full cells synapse sizes for presynaptic GPe and GPi')
    # plot histograms for all synapses where GP cells are post
    plot_histogram_selection(dataframe=gp_post_syn_df, x_data='synapse size', color_palette=ct_palette,
                             label='full_cells_syns_post', count='synapses',
                             foldername=f_name, hue_data='celltype post',
                             title='full cells synapse sizes for postsynaptic GPe and GPi')

    log.info('Analysis done')


