#Connectivity of the INT1-4 full cells per cell: fractions of incoming and outgoing synaptic area per partner cell type,
#reconstruction completeness (number of synapses with the other cells of the connectivity analysis), the most complete
#80 % / 50 % of each type (Fig 5B, Fig S11C, E-H) and UMAP embeddings on connectivity alone (Fig S11D)

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
    import umap
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import matplotlib.pyplot as plt

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    global_params.wd = bio_params.working_dir()
    axon_cts = bio_params.axon_cts()
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    int_cts = [9, 10, 11, 12]
    int_ct_str = [ct_dict[ct] for ct in int_cts]
    #fractions of the most complete cells per type shown in the figures
    complete_fractions = [0.8, 0.5]
    #UMAP parameters
    n_neighbors = 15
    min_dist = 0.3
    random_state = 0
    color_key = 'STNGPINTv6'
    fontsize = 20
    #suitable cell ids of the connectivity analysis (Fig 6), all cell types
    conn_f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/251023_j0251{version}_cts_percentages_mcl_%i_ax%i_synprob_%.2f_TePkBrNGF_newmergers_bw_fs_%i_STNnoGP" % (
        min_comp_len_cell, min_comp_len_ax, syn_prob, fontsize)
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/261003_j0251{version}_INT_conn_completeness_umap_mcl_%i_synprob_%.2f_nn%i_md%.1f_fs%i" % (
        min_comp_len_cell, syn_prob, n_neighbors, min_dist, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('INT_conn_completeness_umap_log', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, %i for projecting axons, syn_prob = %.1f, min_syn_size = %.1f" % (
        min_comp_len_cell, min_comp_len_ax, syn_prob, min_syn_size))
    log.info(f'Suitable cell ids from {conn_f_name}')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)

    log.info('Step 1/5: Load suitable cellids')
    suitable_ids_dict = load_pkl2obj(f'{conn_f_name}/suitable_ids_allct.pkl')
    all_suitable_ids = np.concatenate([suitable_ids_dict[ct] for ct in suitable_ids_dict])
    int_ids = np.concatenate([suitable_ids_dict[ct] for ct in int_cts])
    int_ids_ct = np.concatenate([[ct] * len(suitable_ids_dict[ct]) for ct in int_cts])
    log.info(f'{len(int_ids)} INT cells: ' + ', '.join([f'{ct_dict[ct]} = {len(suitable_ids_dict[ct])}' for ct in int_cts]))

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
    #presynaptic partner = the axon, postsynaptic partner = dendrite or soma
    pre_inds = np.where(m_axs == 1)
    post_inds = np.where(m_axs != 1)
    pre_ids = m_ssv_partners[pre_inds]
    post_ids = m_ssv_partners[post_inds]
    pre_cts = m_cts[pre_inds]
    post_cts = m_cts[post_inds]

    log.info('Step 3/5: Fractions of incoming and outgoing synaptic area per INT cell')
    in_columns = [f'in {ct_dict[ct]}' for ct in ct_dict]
    out_columns = [f'out {ct_dict[ct]}' for ct in ct_dict if ct not in axon_cts]
    columns = ['celltype'] + in_columns + ['in sum size', 'in number'] + out_columns + ['out sum size', 'out number']
    int_df = pd.DataFrame(columns=columns, index=int_ids)
    int_df.index.name = 'cellid'
    int_df['celltype'] = [ct_dict[ct] for ct in int_ids_ct]
    for cellid in int_ids:
        in_inds = post_ids == cellid
        in_sum = np.sum(m_sizes[in_inds])
        for ct in ct_dict:
            int_df.loc[cellid, f'in {ct_dict[ct]}'] = 100 * np.sum(m_sizes[in_inds & (pre_cts == ct)]) / in_sum if in_sum > 0 else np.nan
        int_df.loc[cellid, 'in sum size'] = in_sum
        int_df.loc[cellid, 'in number'] = np.sum(in_inds)
        out_inds = pre_ids == cellid
        out_sum = np.sum(m_sizes[out_inds])
        for ct in ct_dict:
            if ct in axon_cts:
                continue
            int_df.loc[cellid, f'out {ct_dict[ct]}'] = 100 * np.sum(m_sizes[out_inds & (post_cts == ct)]) / out_sum if out_sum > 0 else np.nan
        int_df.loc[cellid, 'out sum size'] = out_sum
        int_df.loc[cellid, 'out number'] = np.sum(out_inds)
    int_df['number of synapses'] = int_df['in number'] + int_df['out number']
    for col in columns[1:] + ['number of synapses']:
        int_df[col] = int_df[col].astype(float)
    int_df.to_csv(f'{f_name}/INT_conn_fractions_percell.csv')
    log.info('Median percentages of incoming synaptic area per INT type:\n' + int_df.groupby('celltype')[in_columns].median().round(1).to_string())
    log.info('Median percentages of outgoing synaptic area per INT type:\n' + int_df.groupby('celltype')[out_columns].median().round(1).to_string())
    log.info('Cells without outgoing synapses: ' + str(int_df['out number'].eq(0).groupby(int_df['celltype']).sum().to_dict()))

    log.info('Step 4/5: Most complete cells per type')
    #completeness = number of synapses with the other cells of the connectivity analysis; a cell is kept if its number is
    #at least the (1 - fraction) quantile of its type
    for fraction in complete_fractions:
        col = f'most complete {int(fraction * 100)} percent'
        int_df[col] = False
        thresholds = {}
        for ct_str in int_ct_str:
            ct_inds = int_df['celltype'] == ct_str
            thresholds[ct_str] = int_df.loc[ct_inds, 'number of synapses'].quantile(1 - fraction)
            int_df.loc[ct_inds & (int_df['number of synapses'] >= thresholds[ct_str]), col] = True
        log.info(f'{col}: thresholds ' + ', '.join([f'{ct_str} = {thresholds[ct_str]:.0f} synapses' for ct_str in int_ct_str]) +
                 '; cells ' + str(int_df.loc[int_df[col]].groupby('celltype').size().to_dict()))
    int_df.to_csv(f'{f_name}/INT_conn_fractions_percell.csv')
    int_df[['celltype', 'number of synapses'] + [f'most complete {int(fraction * 100)} percent' for fraction in complete_fractions]].to_csv(
        f'{f_name}/INT_completeness_percell.csv')

    log.info('Step 5/5: UMAP on connectivity alone')
    #features: z-scored percentages of incoming synaptic area per presynaptic type and outgoing per postsynaptic type;
    #only cells with incoming and outgoing synapses
    feature_columns = in_columns + out_columns
    umap_subsets = {'most complete 50 percent': int_df['most complete 50 percent'], 'all cells': int_df['number of synapses'] > 0}
    umap_results = []
    fig, axs = plt.subplots(1, len(umap_subsets), figsize=(6 * len(umap_subsets), 6))
    for ii, subset in enumerate(umap_subsets):
        subset_inds = umap_subsets[subset] & (int_df['in number'] > 0) & (int_df['out number'] > 0)
        subset_df = int_df.loc[subset_inds, ['celltype'] + feature_columns].copy()
        features = StandardScaler().fit_transform(subset_df[feature_columns].values)
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit_transform(features)
        subset_df['UMAP 1'] = embedding[:, 0]
        subset_df['UMAP 2'] = embedding[:, 1]
        subset_df['subset'] = subset
        log.info(f'UMAP {subset}: ' + str(subset_df.groupby('celltype').size().to_dict()))
        umap_results.append(subset_df)
        sns.scatterplot(data=subset_df, x='UMAP 1', y='UMAP 2', hue='celltype', palette=ct_palette, ax=axs[ii], s=30)
        axs[ii].set_title(subset, fontsize=fontsize)
    umap_df = pd.concat(umap_results)
    umap_df[['celltype', 'subset', 'UMAP 1', 'UMAP 2']].to_csv(f'{f_name}/INT_conn_umap.csv')
    plt.savefig(f'{f_name}/INT_conn_umap.png')
    plt.savefig(f'{f_name}/INT_conn_umap.svg')
    plt.close()
    log.info('Analysis finished')
