#get an estimate about the average synapse density per celltype per axon

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_percell_number_sumsize
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from itertools import combinations
    from scipy.stats import kruskal, ranksums
    import seaborn as sns
    import matplotlib.pyplot as plt

    #global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len = 200
    min_syn_size = 0.1
    syn_prob = 0.6
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'STNGPINTv6'
    fontsize = 20
    zero_soma_fill = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/241025_j0251{version}_avg_syn_den_sb_%.2f_mcl_%i_%s_newmerger" % (
        syn_prob, min_comp_len, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get average synapse density of axon per celltype', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, syn_prob = %.2f, min_syn_size = %.i, colors = %s" % (
            min_comp_len, syn_prob, min_syn_size, color_key))
    if zero_soma_fill:
        log.info('Cells without soma synapses will not be excluded for soma synapse densities but values set to 0.')
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = analysis_params.load_known_mergers()
    log.info("Iterate over celltypes to get eachs estimate of synapse density")
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    cts = list(ct_dict.keys())
    ax_ct = analysis_params.axon_cts()
    ct_str_list = analysis_params.ct_str(with_glia=False)
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_suitable_cts = []
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    log.info('Step 1/4: Filter cells')
    for i, ct in enumerate(tqdm(cts)):
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in ax_ct:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=True, max_path_len=None)
        else:
            #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            #cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_suitable_cts.append([ct_dict[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_suitable_cts = np.concatenate(all_suitable_cts)

    log.info('Step: 2/3: Get synapse density axon per cell')

    syn_columns = ['cellid', 'celltype', 'axon synapse density', 'axon synaptic area density',
                   'axon synaptic area density per surface area',
                   'dendrite synapse density', 'dendrite synaptic area density',
                   'dendrite synaptic area density per surface area', 'soma synaptic area density per surface area',
                   'axon dist between syns', 'dendrite dist between syns']
    synapse_res_df = pd.DataFrame(columns=syn_columns, index = range(len(all_suitable_ids)))
    synapse_res_df['cellid'] = all_suitable_ids
    synapse_res_df['celltype'] = all_suitable_cts

    morph_columns = ['axon pathlength', 'dendrite pathlength', 'axon surface area', 'dendrite surface area', 'soma surface area']
    comp_dict = {0: 'dendrite', 1: 'axon', 2:'soma'}
    comp_nums = list(comp_dict.keys())
    for ct in cts:
        ct_str = ct_dict[ct]
        log.info("Get axon, dendrite pathlength, surface area soma, axon dendrite per cell %s" % ct_str)
        cellids = suitable_ids_dict[ct]
        morph_data_df = pd.DataFrame(columns=morph_columns, index = range(len(cellids)))
        cell_dict = all_cell_dict[ct]
        for i, cellid in enumerate(cellids):
            for comp in comp_nums:
                if ct in ax_ct and comp != 1:
                    continue
                if comp < 2:
                    morph_data_df.loc[i, f'{comp_dict[comp]} pathlength'] = cell_dict[cellid][f'{comp_dict[comp]} length']
                morph_data_df.loc[i, f'{comp_dict[comp]} surface area'] = cell_dict[cellid][f'{comp_dict[comp]} mesh surface area']
        #exclude cells from analyses where surface area is 0

        morph_data_df.to_csv(f'{f_name}/{ct_str}_morph_pathlength_surface_areas.csv')
        log.info("Get number of synapses per cell %s" % ct_str)
        #filter synapse caches for synapses with only synapses of celltypes
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv = sd_synssv,
                                                                                                            pre_cts=[
                                                                                                                ct],
                                                                                                            syn_prob_thresh=syn_prob,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True)
        # only those from cellids
        ct_inds = np.any(np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2), axis=1)
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_axs = m_axs[ct_inds]
        #iterate over compartments to get synapse number and synaptic area per cell for each compartment
        testct = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
        for comp in comp_nums:
            if ct in ax_ct and comp != 1:
                continue
            # filter synapses to get ones where cellids is compartment
            testax = np.in1d(m_axs, comp).reshape(len(m_ssv_partners), 2)
            comp_ct_inds = np.all(testct == testax, axis=1)
            comp_axs = m_axs[comp_ct_inds]
            comp_ssv_partners = m_ssv_partners[comp_ct_inds]
            comp_sizes = m_sizes[comp_ct_inds]
            #get number, sum size per cell
            ct_inds = np.where(comp_axs == comp)
            ct_ssv_ids = comp_ssv_partners[ct_inds]
            comp_syn_numbers, comp_syn_ssv_sizes, comp_unique_ssv_ids = get_percell_number_sumsize(ct_ssv_ids, comp_sizes)
            #reorder to match order of cellids (which were just sorted above)
            sorted_cellids_inds = np.argsort(comp_unique_ssv_ids)
            sorted_unique_ids = comp_unique_ssv_ids[sorted_cellids_inds]
            sorted_comp_syn_numbers = comp_syn_numbers[sorted_cellids_inds]
            sorted_comp_syn_sizes = comp_syn_ssv_sizes[sorted_cellids_inds]
            comp_inds_ct = np.in1d(cellids, sorted_unique_ids)
            comp_inds_all = np.in1d(all_suitable_ids, sorted_unique_ids)
            #get synapse density
            #for axon, dendrite calculate density via pathlength and surface area
            if comp < 2:
                comp_ct_pathlength = morph_data_df.loc[comp_inds_ct, f'{comp_dict[comp]} pathlength']
                comp_syn_density = sorted_comp_syn_numbers / comp_ct_pathlength
                comp_syn_size_density = sorted_comp_syn_sizes / comp_ct_pathlength
                synapse_res_df.loc[comp_inds_all, f'{comp_dict[comp]} synapse density'] = np.array(comp_syn_density)
                synapse_res_df.loc[comp_inds_all, f'{comp_dict[comp]} synaptic area density'] = np.array(comp_syn_size_density)
                comp_dist_syns = comp_ct_pathlength / sorted_comp_syn_numbers
                synapse_res_df.loc[comp_inds_all, f'{comp_dict[comp]} dist between syns'] = np.array(comp_dist_syns)
            comp_syn_area_density = sorted_comp_syn_sizes / morph_data_df.loc[comp_inds_ct, f'{comp_dict[comp]} surface area']
            synapse_res_df.loc[comp_inds_all, f'{comp_dict[comp]} synaptic area density per surface area'] = np.array(comp_syn_area_density)

    params = syn_columns[2:]
    synapse_res_df = synapse_res_df.astype({param: np.float for param in params})
    synapse_res_df.to_csv(f'{f_name}/syn_density_results.csv')
    #make dictionary with dataframe with cells that are specific to each compartment
    ct_groups_comp_dict = {}
    syn_result_df_dict = {}
    num_cells_before = len(synapse_res_df)
    num_cts_before = synapse_res_df.groupby('celltype').size()
    log.info(f'In total {num_cells_before} cells were processed, in the following celltypes: {num_cts_before}')
    for comp in comp_nums:
        comp_str = comp_dict[comp]
        comp_res_df = synapse_res_df[synapse_res_df[f'{comp_str} synaptic area density per surface area'] > 0]
        syn_result_df_dict[comp_str] = comp_res_df
        comp_ct_groups = comp_res_df.groupby('celltype')
        num_cells_comp = len(comp_res_df)
        ct_num_comp = comp_ct_groups.size()
        ct_groups_comp_dict[comp_str] = comp_ct_groups
        log.info(
            f'Number of cells with {comp_str} synapses is {num_cells_comp}, in the following celltypes: {ct_num_comp}')
        log.info(f'{num_cells_before - num_cells_comp} do not have synapses in this compartment')
    if zero_soma_fill:
        log.info(
            'As not having any soma synaspes might be a feature, the full cells with dendritic synapses that have no soma synapses will get a value of 0')
        syn_result_df_dict['soma'] = syn_result_df_dict['dendrite'].fillna(0)
        ct_groups_comp_dict['soma'] = syn_result_df_dict['soma'].groupby('celltype')

    log.info('Step 3/3: Get overview parameters and calculate and plot results')
    overview_columns = [[f'{param} mean', f'{param} std', f'{param} median'] for param in params]
    overview_columns = np.concatenate(overview_columns)
    unique_cts = np.unique(synapse_res_df['celltype'])
    overview_df = pd.DataFrame(columns=overview_columns, index=unique_cts)
    group_comps = list(combinations(unique_cts, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    axon_ct_str = [ct_dict[ct] for ct in ax_ct]
    full_cell_str_list = np.array(ct_str_list)[np.in1d(ct_str_list, axon_ct_str) == False]
    kruskal_df = pd.DataFrame(columns = ['stats', 'p-value'], index = params)
    log.info(f'Only parameters regarding axon will be done on all celltypes, for the rest projecting axon celltypes {axon_ct_str} will be excluded')
    ct_palette = cls.ct_palette(color_key, num=False)
    for comp in comp_nums:
        comp_str = comp_dict[comp]
        results_df = syn_result_df_dict[comp_str]
        ct_res_groups = ct_groups_comp_dict[comp_str]
        for param in params:
            if comp_str not in param:
                continue
            #get overview params
            overview_df.loc[unique_cts, f'{param} median'] = ct_res_groups[param].median()
            overview_df.loc[unique_cts, f'{param} mean'] = ct_res_groups[param].mean()
            overview_df.loc[unique_cts, f'{param} std'] = ct_res_groups[param].std()
            # calculate kruskal wallis for differences between the two groups
            param_groups = [group[param].values for name, group in
                            results_df.groupby('celltype')]
            kruskal_res = kruskal(*param_groups, nan_policy='omit')
            kruskal_df.loc[param, 'stats'] = kruskal_res[0]
            kruskal_df.loc[param, 'p-value'] = kruskal_res[1]
            # get ranksum results if significant
            if kruskal_res[1] < 0.05:
                for group in group_comps:
                    if (group[0] in axon_ct_str or group[1] in axon_ct_str) and not 'axon' in param:
                        continue
                    ranksum_res = ranksums(ct_res_groups.get_group(group[0])[param],
                                           ct_res_groups.get_group(group[1])[param])
                    ranksum_group_df.loc[f'{param} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
                    ranksum_group_df.loc[f'{param} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]
            #plot results
            if 'surface area' in param:
                ylabel = f'{param} [µm²/µm²]'
            elif 'area' in param:
                ylabel = f'{param} [µm²/µm]'
            elif 'density' in param:
                ylabel = f'{param} [1/µm]'
            elif'dist' in param:
                ylabel = f'{param} [µm]'
            else:
                raise ValueError('no ylabel defined for this parameter')
            if comp == 1:
                plot_order = ct_str_list
            else:
                plot_order = full_cell_str_list
            sns.boxplot(data=results_df, x='celltype', y=param, palette=ct_palette, order=plot_order)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xlabel('celltype', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.title(f'{param}')
            plt.savefig(f'{f_name}/{param}_box.svg')
            plt.savefig(f'{f_name}/{param}_box.png')
            plt.close()
            sns.violinplot(data=synapse_res_df, x='celltype', y=param, palette=ct_palette, order=plot_order)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xlabel('celltype', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.title(f'{param}')
            plt.savefig(f'{f_name}/{param}_violin.svg')
            plt.savefig(f'{f_name}/{param}_violin.png')
            plt.close()

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')
    kruskal_df.to_csv(f'{f_name}/kruskal_res.csv')

    log.info("Analysis finished")