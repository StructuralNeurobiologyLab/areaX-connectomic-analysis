#script to plot spine density for all msns
#plot spine density with msn subgroups overlayed
#plot spine density as jointplot vs GPe/ GPi ratio in synapse number and synapse sum size
#GPe/GPi ratios = (GPe + GPi)/GPi

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple, compare_soma_diameters
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_spine_density
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_nx_graph
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from syconn.mp.mp_utils import start_multiprocess_imap
    import seaborn as sns
    import matplotlib.pyplot as plt

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 2
    gpe_ct = 6
    gpi_ct = 7
    fontsize_jointplot = 12
    kde = True
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230814_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_replot" % (
    min_comp_len, syn_prob, kde)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN spine density vs GP ratio', log_dir=f_name + '/logs/')
    log.info("Analysis of spine density vs GP ratio starts")
    if kde:
        log.info('Centre of jointplot will be kdeplot')
    else:
        log.info('Centre of jointplot will be scatter')

    log.info('Step 1/5: Load and check all MSN cells')
    known_mergers = analysis_params.load_known_mergers()
    MSN_dict = analysis_params.load_cell_dict(celltype=msn_ct)
    MSN_ids = np.array(list(MSN_dict.keys()))
    merger_inds = np.in1d(MSN_ids, known_mergers) == False
    MSN_ids = MSN_ids[merger_inds]
    misclassified_asto_ids = analysis_params.load_potential_astros()
    astro_inds = np.in1d(MSN_ids, misclassified_asto_ids) == False
    MSN_ids = MSN_ids[astro_inds]
    MSN_ids = check_comp_lengths_ct(cellids=MSN_ids, fullcelldict=MSN_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    columns = ['cellid', 'spine density', 'celltype', 'GP ratio syn number', 'GP ratio sum syn size', 'syn number to GPe', 'syn size to GPe', 'syn number to GPi', 'syn size to GPi']
    msn_result_df = pd.DataFrame(columns=columns, index=range(len(MSN_ids)))
    MSN_ids = np.sort(MSN_ids)
    msn_result_df['cellid'] = MSN_ids

    '''

    log.info('Step 2/5: Get spine density of all MSN cells')
    ngf_input = [[msn_id, min_comp_len, MSN_dict] for msn_id in MSN_ids]
    spine_density = start_multiprocess_imap(get_spine_density, ngf_input)
    spine_density = np.array(spine_density)
    msn_result_df['spine density'] = spine_density
    msn_result_df.to_csv(f'{f_name}/msn_spine_density_results.csv')
    xhist = 'spine density [1/µm]'
    sns.histplot(x='spine density', data=msn_result_df, color='black', common_norm=True,
                 fill=False, element="step", linewidth=3)
    plt.ylabel('fraction of cells')
    plt.xlabel(xhist)
    plt.title('spine density in MSN')
    plt.savefig(f'{f_name}/spine_density_hist_norm.png')
    plt.close()
    sns.histplot(x='spine density', data=msn_result_df, color='black', common_norm=False,
                 fill=False, element="step", linewidth=3)
    plt.ylabel('number of cells')
    plt.xlabel(xhist)
    plt.title('spine density in MSN')
    plt.savefig(f'{f_name}/spine_density_hist.png')
    plt.close()


    log.info('Step 3/5: Get GP ratio for MSNs')
    msn_result_df['syn number to GPe'] = 0
    msn_result_df['syn size to GPe'] = 0
    msn_result_df['syn number to GPi'] = 0
    msn_result_df['syn size to GPi'] = 0
    log.info('Get suitable cellids from GPe and GPi')
    GPe_dict = analysis_params.load_cell_dict(gpe_ct)
    GPe_ids = np.array(list(GPe_dict.keys()))
    merger_inds = np.in1d(GPe_ids, known_mergers) == False
    GPe_ids = GPe_ids[merger_inds]
    GPe_ids = check_comp_lengths_ct(cellids=GPe_ids, fullcelldict=GPe_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    GPi_dict = analysis_params.load_cell_dict(gpi_ct)
    GPi_ids = np.array(list(GPi_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(GPe_ids)} suitable for analysis and {len(GPi_ids)} suitable for analysis')

    log.info('Get information from synapses to GPe and GPi')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    #prefilter synapses for syn_prob, min_syn_size and celltype
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv = sd_synssv,
                                                                                                        pre_cts=[msn_ct],
                                                                                                        post_cts=[gpe_ct, gpi_ct],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=None)
    #filter synapses to only include full cells
    all_suitable_ids = np.hstack([MSN_ids, GPe_ids, GPi_ids])
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    #get per cell information about synapses to GPe
    gpe_inds = np.where(m_cts == gpe_ct)[0]
    gpe_ssv_partners = m_ssv_partners[gpe_inds]
    gpe_sizes = m_sizes[gpe_inds]
    gpe_cts = m_cts[gpe_inds]
    gpe_syn_numbers, gpe_sum_sizes, unique_msn_gpe = get_ct_syn_number_sumsize(syn_sizes = gpe_sizes,
                                                                               syn_ssv_partners = gpe_ssv_partners,
                                                                               syn_cts= gpe_cts, ct = msn_ct)
    sort_inds_gpe = np.argsort(unique_msn_gpe)
    unique_msn_gpe_sorted = unique_msn_gpe[sort_inds_gpe]
    gpe_syn_numbers_sorted = gpe_syn_numbers[sort_inds_gpe]
    gpe_sum_sizes_sorted = gpe_sum_sizes[sort_inds_gpe]
    gpe_df_inds = np.in1d(msn_result_df['cellid'], unique_msn_gpe_sorted)
    msn_result_df.loc[gpe_df_inds, 'syn number to GPe'] = gpe_syn_numbers_sorted
    msn_result_df.loc[gpe_df_inds, 'syn size to GPe'] = gpe_sum_sizes_sorted

    #get per cell information about synapses to GPi
    gpi_inds = np.where(m_cts == gpi_ct)[0]
    gpi_ssv_partners = m_ssv_partners[gpi_inds]
    gpi_sizes = m_sizes[gpi_inds]
    gpi_cts = m_cts[gpi_inds]
    gpi_syn_numbers, gpi_sum_sizes, unique_msn_gpi = get_ct_syn_number_sumsize(syn_sizes=gpi_sizes,
                                                                               syn_ssv_partners=gpi_ssv_partners,
                                                                               syn_cts=gpi_cts, ct=msn_ct)
    sort_inds_gpi = np.argsort(unique_msn_gpi)
    unique_msn_gpi_sorted = unique_msn_gpi[sort_inds_gpi]
    gpi_syn_numbers_sorted = gpi_syn_numbers[sort_inds_gpi]
    gpi_sum_sizes_sorted = gpi_sum_sizes[sort_inds_gpi]
    gpi_df_inds = np.in1d(msn_result_df['cellid'], unique_msn_gpi_sorted)
    msn_result_df.loc[gpi_df_inds, 'syn number to GPi'] = gpi_syn_numbers_sorted
    msn_result_df.loc[gpi_df_inds, 'syn size to GPi'] = gpi_sum_sizes_sorted
    #get GP ratio per msn cell and put in dataframe: GPi/(GPi + GPe)
    nonzero_inds = np.any([msn_result_df['syn number to GPi'] > 0, msn_result_df['syn number to GPe'] > 0], axis = 0)
    msn_result_df.loc[nonzero_inds, 'GP ratio syn number'] = msn_result_df.loc[nonzero_inds, 'syn number to GPi'] / \
                                                             (msn_result_df.loc[nonzero_inds, 'syn number to GPi']  + msn_result_df.loc[nonzero_inds, 'syn number to GPe'])
    msn_result_df.loc[nonzero_inds, 'GP ratio sum syn size'] =  msn_result_df.loc[nonzero_inds, 'syn size to GPi']/ \
                                                                 (msn_result_df.loc[nonzero_inds, 'syn size to GPe'] + msn_result_df.loc[nonzero_inds, 'syn size to GPi'])
    msn_result_df.to_csv(f'{f_name}/msn_spine_density_GPratio.csv')
    #plot histograms for all cells, GP ratio only for those connected to GP
    for key in msn_result_df.keys():
        if 'cellid' in key or 'density' in key:
            continue
        if 'GP ratio' in key:
            xhist = '(GPe + GPi)/GPi'
        elif 'syn size' in key:
            xhist = f'{key} [µm2]'
        else:
            xhist = key
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist.png')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=True,
                     fill=False, element="step", linewidth=3
                     )
        plt.ylabel('fraction of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_norm.png')
        plt.close()
    '''
    f_loading = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230814_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i" % (
    min_comp_len, syn_prob, kde)
    msn_result_df = pd.read_csv(f'{f_loading}/msn_spine_density_GPratio.csv')

    log.info('Step 4/5: Plot spine density vs GP ratio as joint plot')
    ratio_key_list = ['GP ratio syn number', 'GP ratio sum syn size']
    for rkey in ratio_key_list:
        g = sns.JointGrid(data=msn_result_df, x='spine density', y=rkey)
        g.plot_joint(sns.kdeplot, color = "#EAAE34")
        g.plot_joint(sns.scatterplot, color = 'black', alpha = 0.3)
        g.plot_marginals(sns.histplot, fill=False,
                         kde=False, bins='auto', color='black')
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_xlabel('spine density [1/µm]')
        if 'number' in rkey:
            g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn numbers')
        else:
            g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn area')
        plt.savefig(f'{f_name}/spine_density_{rkey}_all.png')
        plt.savefig(f'{f_name}/spine_density_{rkey}_all.svg')
        plt.close()


    log.info('Step 5/5: Divide MSN into groups depending on connectivity, plot again and compare groups')
    gpe_zero = msn_result_df['syn number to GPe'] == 0
    gpi_zero = msn_result_df['syn number to GPi'] == 0
    no_gp = np.all([gpe_zero, gpi_zero], axis = 0)
    msn_result_df.loc[no_gp, 'celltype'] = 'MSN no GP'
    both_GP = np.any([gpe_zero, gpi_zero], axis = 0) == False
    msn_result_df.loc[both_GP, 'celltype'] = 'MSN both GPs'
    only_GPe = np.all([gpe_zero == False, gpi_zero], axis = 0)
    msn_result_df.loc[only_GPe, 'celltype'] = 'MSN only GPe'
    only_GPi = np.all([gpe_zero, gpi_zero == False], axis=0)
    msn_result_df.loc[only_GPi, 'celltype'] = 'MSN only GPi'
    msn_result_df.to_csv(f'{f_name}/msn_spine_density_GPratio.csv')
    celltype_groups = msn_result_df.groupby('celltype')
    msn_subgroup_size = celltype_groups.size()
    log.info(f'There are the following MSN subgroups with corresponding sizes: {msn_subgroup_size}')
    #create summary for number of subgroup and create plot
    msn_subgroup_percent = msn_subgroup_size * 100 / len(MSN_ids)
    msn_groups_str = np.unique(msn_result_df['celltype'])
    msn_subgroup_df = pd.DataFrame(columns = ['celltype', 'number of MSN cells', '% of MSN cells'], index = range(len(msn_groups_str)))
    msn_subgroup_df['celltype'] = msn_groups_str
    msn_subgroup_df['number of MSN cells'] = np.array(msn_subgroup_size)
    msn_subgroup_df['% of MSN cells'] = np.array(msn_subgroup_percent)
    msn_subgroup_df.to_csv(f'{f_name}/msn_subgroup_numbers.csv')
    msn_colors = ["#EAAE34", "black", "#707070", '#2F86A8']
    msn_palette = {ct: msn_colors[i] for i, ct in enumerate(msn_groups_str)}
    '''
    for key in msn_subgroup_df.keys():
        if 'celltype' in key:
            continue
        sns.barplot(data = msn_subgroup_df, x = 'celltype', y = key, palette=msn_palette)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_overview_bar.png')
        plt.savefig(f'{f_name}/{key}_overview_bar.svg')
        plt.close()
    '''
    #overlay jointplot with density plot and plot again
    ratio_key_list = ['GP ratio syn number', 'GP ratio sum syn size']
    raise ValueError
    for rkey in ratio_key_list:
        g = sns.JointGrid(data=msn_result_df, x='spine density', y=rkey, hue="celltype", palette=msn_palette)
        g.plot_joint(sns.kdeplot, hue = 'celltype', palette = msn_palette)
        g.plot_joint(sns.scatterplot, color = 'black', alpha = 0.3)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins='auto', palette=msn_palette)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_xlabel('spine density [1/µm]')
        if 'number' in rkey:
            g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn numbers')
        else:
            g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn area')
        plt.savefig(f'{f_name}/spine_density_{rkey}.png')
        plt.savefig(f'{f_name}/spine_density_{rkey}.svg')
        plt.close()


    log.info('MSN subgroup analysis of spine density and GP ratio done')
