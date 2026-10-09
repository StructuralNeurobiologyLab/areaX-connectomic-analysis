#Two potential GPi subpopulations (Fig S8E-H): per GPi cell the fraction of incoming synaptic area from INT1 and the fraction
#of outgoing synaptic area onto MSN (two measures that do not involve GPi-GPi synapses) separate GPi1 (INT1 input, little
#MSN output) from GPi2 (no INT1 input, MSN output); synaptic area between all pairs of GPi cells and the pooled output
#targets of both groups

if __name__ == '__main__':
    from bio_analysis.general.analysis_params import Analysis_Params
    from bio_analysis.general.analysis_colors import CelltypeColors
    from bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, filter_synapse_caches_for_ct
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.handler.basics import load_pkl2obj
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    global_params.wd = bio_params.working_dir()
    axon_cts = bio_params.axon_cts()
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    gpi_ct = 7
    int1_ct = 9
    msn_ct = 3
    #a GPi cell belongs to GPi1 if at least this fraction of its incoming synaptic area comes from INT1,
    #and to GPi2 if more than this fraction of its outgoing synaptic area goes to MSN; both criteria must agree
    int1_input_thresh = 0.03
    msn_output_thresh = 0.15
    color_key = 'STNGPINTv6'
    fontsize = 20
    #suitable cell ids of the connectivity analysis (Fig 6), all cell types
    conn_f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/251023_j0251{version}_cts_percentages_mcl_%i_ax%i_synprob_%.2f_TePkBrNGF_newmergers_bw_fs_%i_STNnoGP" % (
        min_comp_len_cell, min_comp_len_ax, syn_prob, fontsize)
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/261003_j0251{version}_GPi_subpopulations_mcl_%i_synprob_%.2f_int1_%.2f_msn_%.2f_fs%i" % (
        min_comp_len_cell, syn_prob, int1_input_thresh, msn_output_thresh, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPi_subpopulations_log', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, %i for projecting axons, syn_prob = %.1f, min_syn_size = %.1f" % (
        min_comp_len_cell, min_comp_len_ax, syn_prob, min_syn_size))
    log.info(f'GPi1: INT1 input fraction >= {int1_input_thresh}, GPi2: MSN output fraction > {msn_output_thresh}')
    log.info(f'Suitable cell ids from {conn_f_name}')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    group_palette = {'GPi1': '#49bb38', 'GPi2': '#8c8c8c'}

    log.info('Step 1/5: Load suitable cellids')
    suitable_ids_dict = load_pkl2obj(f'{conn_f_name}/suitable_ids_allct.pkl')
    all_suitable_ids = np.concatenate([suitable_ids_dict[ct] for ct in suitable_ids_dict])
    gpi_ids = np.sort(suitable_ids_dict[gpi_ct])
    log.info(f'{len(gpi_ids)} GPi cells')

    log.info('Step 2/5: Filter synapses')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob_arr = filter_synapse_caches_general(
        sd_synssv, syn_prob_thresh=syn_prob, min_syn_size=min_syn_size)
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob_arr]
    #axo-dendritic and axo-somatic synapses between the suitable cells, as in the connectivity matrix
    pre_cts = [ct for ct in ct_dict]
    post_cts = [ct for ct in ct_dict if ct not in axon_cts]
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=pre_cts, post_cts=post_cts, syn_prob_thresh=None, min_syn_size=None, axo_den_so=True, synapses_caches=synapse_cache)
    suit_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_inds]
    m_axs = m_axs[suit_inds]
    m_ssv_partners = m_ssv_partners[suit_inds]
    m_sizes = m_sizes[suit_inds]
    log.info(f'{len(m_sizes)} synapses between suitable cells')
    pre_inds = np.where(m_axs == 1)
    post_inds = np.where(m_axs != 1)
    pre_ids = m_ssv_partners[pre_inds]
    post_ids = m_ssv_partners[post_inds]
    pre_cts = m_cts[pre_inds]
    post_cts = m_cts[post_inds]

    log.info('Step 3/5: Fractions of incoming and outgoing synaptic area per GPi cell')
    in_columns = [f'in {ct_dict[ct]}' for ct in ct_dict]
    out_columns = [f'out {ct_dict[ct]}' for ct in ct_dict if ct not in axon_cts]
    gpi_df = pd.DataFrame(columns=in_columns + ['in sum size', 'in number'] + out_columns + ['out sum size', 'out number'], index=gpi_ids)
    gpi_df.index.name = 'cellid'
    for cellid in gpi_ids:
        in_inds = post_ids == cellid
        in_sum = np.sum(m_sizes[in_inds])
        for ct in ct_dict:
            gpi_df.loc[cellid, f'in {ct_dict[ct]}'] = np.sum(m_sizes[in_inds & (pre_cts == ct)]) / in_sum
        gpi_df.loc[cellid, 'in sum size'] = in_sum
        gpi_df.loc[cellid, 'in number'] = np.sum(in_inds)
        out_inds = pre_ids == cellid
        out_sum = np.sum(m_sizes[out_inds])
        for ct in ct_dict:
            if ct in axon_cts:
                continue
            gpi_df.loc[cellid, f'out {ct_dict[ct]}'] = np.sum(m_sizes[out_inds & (post_cts == ct)]) / out_sum
        gpi_df.loc[cellid, 'out sum size'] = out_sum
        gpi_df.loc[cellid, 'out number'] = np.sum(out_inds)
    gpi_df = gpi_df.astype(float)
    int1_input = gpi_df[f'in {ct_dict[int1_ct]}']
    msn_output = gpi_df[f'out {ct_dict[msn_ct]}']
    gpi_df['subpopulation'] = np.where(int1_input >= int1_input_thresh, 'GPi1', 'GPi2')
    msn_based = np.where(msn_output <= msn_output_thresh, 'GPi1', 'GPi2')
    if np.any(gpi_df['subpopulation'] != msn_based):
        log.info(f'WARNING: INT1 input and MSN output criteria disagree for cells {gpi_df.index[gpi_df["subpopulation"] != msn_based].tolist()}')
    gpi_df.to_csv(f'{f_name}/GPi_conn_fractions_percell.csv')
    for group in ['GPi1', 'GPi2']:
        group_inds = gpi_df['subpopulation'] == group
        log.info(f'{group}: n = {np.sum(group_inds)}, INT1 input fraction {int1_input[group_inds].min():.3f} - {int1_input[group_inds].max():.3f}, '
                 f'MSN output fraction {msn_output[group_inds].min():.3f} - {msn_output[group_inds].max():.3f}')
    sns.scatterplot(data=gpi_df, x=f'in {ct_dict[int1_ct]}', y=f'out {ct_dict[msn_ct]}', hue='subpopulation', palette=group_palette)
    plt.xlabel('fraction of incoming synaptic area from INT1', fontsize=fontsize)
    plt.ylabel('fraction of outgoing synaptic area onto MSN', fontsize=fontsize)
    plt.savefig(f'{f_name}/GPi_subpopulations_definition.png')
    plt.savefig(f'{f_name}/GPi_subpopulations_definition.svg')
    plt.close()

    log.info('Step 4/5: Synaptic area between all pairs of GPi cells')
    gpi_syn_inds = np.in1d(pre_ids, gpi_ids) & np.in1d(post_ids, gpi_ids)
    gpi_pairs = pd.DataFrame({'pre': pre_ids[gpi_syn_inds], 'post': post_ids[gpi_syn_inds], 'size': m_sizes[gpi_syn_inds]})
    order = np.concatenate([gpi_df.index[gpi_df['subpopulation'] == group] for group in ['GPi1', 'GPi2']])
    adjacency_area = gpi_pairs.pivot_table(index='pre', columns='post', values='size', aggfunc='sum').reindex(index=order, columns=order).fillna(0)
    adjacency_number = gpi_pairs.pivot_table(index='pre', columns='post', values='size', aggfunc='count').reindex(index=order, columns=order).fillna(0)
    adjacency_area.to_csv(f'{f_name}/GPi_GPi_adjacency_sum_size.csv')
    adjacency_number.to_csv(f'{f_name}/GPi_GPi_adjacency_number.csv')
    total_gpi_area = adjacency_area.values.sum()
    log.info(f'{len(gpi_pairs)} GPi-GPi synapses with a summed area of {total_gpi_area:.2f} µm²')
    for pre_group in ['GPi1', 'GPi2']:
        for post_group in ['GPi1', 'GPi2']:
            pre_cells = gpi_df.index[gpi_df['subpopulation'] == pre_group]
            post_cells = gpi_df.index[gpi_df['subpopulation'] == post_group]
            group_area = adjacency_area.loc[pre_cells, post_cells].values.sum()
            group_number = adjacency_number.loc[pre_cells, post_cells].values.sum()
            log.info(f'{pre_group} -> {post_group}: {group_number:.0f} synapses, {group_area:.2f} µm² = {100 * group_area / total_gpi_area:.1f} % of the GPi-GPi synaptic area')
    plt.figure(figsize=(8, 7))
    sns.heatmap(adjacency_area + 1e-3, norm=LogNorm(0.1, 100), cmap='Greens', cbar_kws={'label': 'summed synaptic area [µm²]'})
    plt.xlabel('postsynaptic GPi', fontsize=fontsize)
    plt.ylabel('presynaptic GPi', fontsize=fontsize)
    plt.savefig(f'{f_name}/GPi_GPi_adjacency_sum_size.png')
    plt.savefig(f'{f_name}/GPi_GPi_adjacency_sum_size.svg')
    plt.close()

    log.info('Step 5/5: Pooled output targets per subpopulation')
    #fraction of the summed outgoing synaptic area of all cells of a subpopulation per postsynaptic cell type,
    #with GPi targets split by subpopulation
    target_columns = ['GPi1', 'GPi2'] + [ct_dict[ct] for ct in ct_dict if ct not in axon_cts and ct != gpi_ct]
    output_df = pd.DataFrame(columns=target_columns, index=['GPi1', 'GPi2'])
    output_df.index.name = 'subpopulation'
    for group in ['GPi1', 'GPi2']:
        group_cells = gpi_df.index[gpi_df['subpopulation'] == group]
        total_output = gpi_df.loc[group_cells, 'out sum size'].sum()
        for target in target_columns:
            if target in ['GPi1', 'GPi2']:
                target_cells = gpi_df.index[gpi_df['subpopulation'] == target]
                target_area = adjacency_area.loc[group_cells, target_cells].values.sum()
            else:
                target_area = (gpi_df.loc[group_cells, f'out {target}'] * gpi_df.loc[group_cells, 'out sum size']).sum()
            output_df.loc[group, target] = 100 * target_area / total_output
    output_df = output_df.astype(float)
    output_df.to_csv(f'{f_name}/GPi_subpopulations_output_percentages.csv')
    log.info('Percentage of outgoing synaptic area per target:\n' + output_df.round(1).to_string())
    output_df.T.plot(kind='bar', color=[group_palette['GPi1'], group_palette['GPi2']])
    plt.ylabel('% of outgoing synaptic area', fontsize=fontsize)
    plt.xlabel('postsynaptic cell type', fontsize=fontsize)
    plt.savefig(f'{f_name}/GPi_subpopulations_output_percentages.png')
    plt.savefig(f'{f_name}/GPi_subpopulations_output_percentages.svg')
    plt.close()
    log.info('Analysis finished')
