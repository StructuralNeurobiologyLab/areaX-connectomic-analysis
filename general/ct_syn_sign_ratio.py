#get majority of outgoing synapse sign for all celltypes
#also calculate ratio of asymmetric to symmetric synapses

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_conn_helper import filter_synapse_caches_for_ct, get_percell_number_sumsize
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal
    from itertools import combinations
    from tqdm import tqdm

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    min_syn_size = 0.1
    syn_prob = 0.6
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    color_key = 'TePkBrNGF'
    fontsize = 14
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/240523_j0251{version}_ct_syn_sign_mcl_%i_ax%i_%s_fs%i" % (
                 min_comp_len_cell, min_comp_len_ax, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('ct_syn_sign_analyses', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, color_key))
    log.info(f' min syn size = {min_syn_size} µm², syn prob threshold = {syn_prob}')
    log.info('Asymmetric  = 1, symmetric = -1')
    log.info('Get ratio of asymmetric to symmetric synapses for outgoing synapses of each celltype')

    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    np_presaved_loc = analysis_params.file_locations
    ct_types = np.arange(0, num_cts)
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)

    log.info('Step 1/4: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_celltypes = []
    all_celltypes_num = []
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in axon_cts:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_celltypes.append([ct_dict[ct] for i in cellids])
        all_celltypes_num.append([[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)
    all_celltypes_num = np.concatenate(all_celltypes_num)
    sorted_suitable_ids = np.sort(all_suitable_ids)

    #symmetric = -1, asymmetric = 1
    columns = ['cellid', 'celltype', 'syn number asymmetric', 'syn number symmetric', 'sum syn area asymmetric', 'sum syn area symmetric',
               'asymmetric syn ratio', 'asymmetric syn area ratio', 'majority syn sign', 'majority syn area sign']
    syn_sign_df = pd.DataFrame(columns = columns, index = range(len(all_suitable_ids)))
    syn_sign_df['cellid'] = all_suitable_ids
    syn_sign_df['celltype'] = all_celltypes
    param_list = columns[2:]

    log.info('Step 2/4: Get asymmetric sign ratio and majority syn sign for all celltypes')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    sign_dict = {-1:'symmetric', 1:'asymmetric'}
    signs = list(sign_dict.keys())
    unique_cts = np.unique(syn_sign_df['celltype'])
    maj_columns = ['celltype', 'cell number (syn number maj)', 'cell fraction (syn number maj)',
                   'cell number (syn area maj)', 'cell fraction (syn area maj)',
                   'syn number', 'syn fraction',
                   'sum syn area', 'syn area fraction', 'syn sign']
    majority_df = pd.DataFrame(columns=maj_columns, index=range(num_cts * 3))
    for i in range(3):
        majority_df.loc[i * num_cts: (i + 1) * num_cts - 1, 'celltype'] = unique_cts
    for ct in tqdm(ct_types):
        ct_str = ct_dict[ct]
        # filter synapse caches for synapses with only synapses of celltypes
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, m_syn_sign = filter_synapse_caches_for_ct(
            sd_synssv=sd_synssv,
            pre_cts=[
                ct],
            syn_prob_thresh=syn_prob,
            min_syn_size=min_syn_size,
            axo_den_so=True, with_sign = True)
        # filter so that only filtered cellids are included and are all presynaptic
        ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[ct]).reshape(len(m_ssv_partners), 2), axis = 1)
        m_rep_coord = m_rep_coord[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_syn_sign = m_syn_sign[ct_inds]
        ct_inds = np.in1d(m_ssv_partners, suitable_ids_dict[ct]).reshape(len(m_ssv_partners), 2)
        comp_inds = np.in1d(m_axs, 1).reshape(len(m_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_coords = m_rep_coord[filtered_inds]
        syn_axs = m_axs[filtered_inds]
        syn_ssv_partners = m_ssv_partners[filtered_inds]
        syn_sizes = m_sizes[filtered_inds]
        syn_sign = m_syn_sign[filtered_inds]
        #get total number of synapses per ct and total area
        total_num_syns = len(syn_sizes)
        total_area_syns = np.sum(syn_sizes)
        #get number of asymmetric and symmetric synapses per cell
        for i, sign in enumerate(signs):
            # filter synapses for correct sign
            sign_inds = np.in1d(syn_sign, sign)
            sign_axs = syn_axs[sign_inds]
            sign_ssv_partners = syn_ssv_partners[sign_inds]
            sign_sizes = syn_sizes[sign_inds]
            # get number, sum size per cell
            ct_inds = np.where(sign_axs == 1)
            ct_ssv_ids = sign_ssv_partners[ct_inds]
            sign_syn_numbers, sign_syn_ssv_sizes, sign_unique_ssv_ids = get_percell_number_sumsize(ct_ssv_ids,
                                                                                                   sign_sizes)
            # reorder to match order of cellids (which were just sorted above)
            sorted_cellids_inds = np.argsort(sign_unique_ssv_ids)
            sorted_unique_ids = sign_unique_ssv_ids[sorted_cellids_inds]
            sorted_sign_syn_numbers = sign_syn_numbers[sorted_cellids_inds]
            sorted_sign_syn_sizes = sign_syn_ssv_sizes[sorted_cellids_inds]
            sign_inds_ct = np.in1d(cellids, sorted_unique_ids)
            sign_inds_all = np.in1d(all_suitable_ids, sorted_unique_ids)
            syn_sign_df.loc[sign_inds_all, f'syn number {sign_dict[sign]}'] = sorted_sign_syn_numbers
            syn_sign_df.loc[sign_inds_all, f'sum syn area {sign_dict[sign]}'] = sorted_sign_syn_sizes
            ct_ind = np.where(unique_cts == ct_str)[0][0]
            majority_df.loc[ct_ind + i*num_cts, 'syn number'] = len(sign_sizes)
            majority_df.loc[ct_ind + i*num_cts, 'syn fraction'] = len(sign_sizes) / total_num_syns
            majority_df.loc[ct_ind + i * num_cts, 'sum syn area'] = np.sum(sign_sizes)
            majority_df.loc[ct_ind + i * num_cts, 'syn area fraction'] = np.sum(sign_sizes) / total_area_syns


    #get total number of synapses to calculate ratio and remove cells without synapses
    syn_sign_df['total syn number'] = syn_sign_df['syn number asymmetric'] + syn_sign_df['syn number symmetric']
    syn_sign_df['total syn area'] = syn_sign_df['sum syn area asymmetric'] + syn_sign_df['sum syn area symmetric']
    #remove cells without synapses
    num_cells_before = len(syn_sign_df)
    num_cts_before = syn_sign_df.groupby('celltype').size()
    log.info(f'In total {num_cells_before} cells were processed, in the following celltypes: {num_cts_before}')
    syn_sign_df = syn_sign_df[syn_sign_df['total syn number'] > 0]
    num_cells_after = len(syn_sign_df)
    ct_groups = syn_sign_df.groupby('celltype')
    num_cts_after = ct_groups.size()
    log.info(
        f'Number of cells with presynaptic synapses is {num_cells_after}, in the following celltypes: {num_cts_after}')
    log.info(f'{num_cells_before - num_cells_after} cells were removed')
    # get asymmetric ratio
    syn_sign_df['asymmetric syn ratio'] = syn_sign_df['syn number asymmetric'] / syn_sign_df['total syn number']
    syn_sign_df['asymmetric syn area ratio'] = syn_sign_df['sum syn area asymmetric'] / syn_sign_df['total syn area']
    #get majority syn sign
    asym_maj_inds = syn_sign_df['syn number asymmetric'] > syn_sign_df['syn number symmetric']
    sym_maj_inds = syn_sign_df['syn number asymmetric'] < syn_sign_df['syn number symmetric']
    equ_inds = syn_sign_df['syn number asymmetric'] == syn_sign_df['syn number symmetric']
    syn_sign_df.loc[asym_maj_inds, 'majority syn sign'] = 1
    syn_sign_df.loc[sym_maj_inds, 'majority syn sign'] = -1
    syn_sign_df.loc[equ_inds, 'majority syn sign'] = 0
    asym_maj_inds = syn_sign_df['sum syn area asymmetric'] > syn_sign_df['sum syn area symmetric']
    sym_maj_inds = syn_sign_df['sum syn area asymmetric'] < syn_sign_df['sum syn area symmetric']
    equ_inds = syn_sign_df['sum syn area asymmetric'] == syn_sign_df['sum syn area symmetric']
    syn_sign_df.loc[asym_maj_inds, 'majority syn area sign'] = 1
    syn_sign_df.loc[sym_maj_inds, 'majority syn area sign'] = -1
    syn_sign_df.loc[equ_inds, 'majority syn area sign'] = 0
    syn_sign_df = syn_sign_df.astype({'syn number asymmetric': int, 'syn number symmetric': int,
                                      'sum syn area asymmetric': float, 'sum syn area symmetric': float,
                                      'asymmetric syn ratio': float, 'asymmetric syn area ratio':float})
    syn_sign_df.to_csv(f'{f_name}/syn_sign_df_percell.csv')

    log.info('Step 3/4: Get overview params and calculate statistics')
    overview_columns = [[f'{param} mean', f'{param} std', f'{param} median'] for param in param_list]
    overview_columns = np.concatenate(overview_columns)
    overview_df = pd.DataFrame(columns=overview_columns, index=unique_cts)
    group_comps = list(combinations(unique_cts, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    kruskal_df = pd.DataFrame(columns=['stats', 'p-value'], index=param_list)
    for param in param_list:
        # calculate kruskal wallis for differences between the two groups
        if 'majority' in param:
            continue
        param_groups = [group[param].values for name, group in
                        syn_sign_df.groupby('celltype')]
        overview_df.loc[unique_cts, f'{param} median'] = ct_groups[param].median()
        overview_df.loc[unique_cts, f'{param} mean'] = ct_groups[param].mean()
        overview_df.loc[unique_cts, f'{param} std'] = ct_groups[param].std()

        kruskal_res = kruskal(*param_groups, nan_policy='omit')
        kruskal_df.loc[param, 'stats'] = kruskal_res[0]
        kruskal_df.loc[param, 'p-value'] = kruskal_res[1]
        # get ranksum results if significant
        if kruskal_res[1] < 0.05:
            for group in group_comps:
                ranksum_res = ranksums(ct_groups.get_group(group[0])[param],
                                       ct_groups.get_group(group[1])[param])
                ranksum_group_df.loc[f'{param} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
                ranksum_group_df.loc[f'{param} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]

    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')
    kruskal_df.to_csv(f'{f_name}/kruskal_res.csv')

    #drop majority columns that were not filled
    overview_df = overview_df.dropna(axis = 1)
    #get number of cells for each celltype that are asymmetric, symmetric
    overview_df['cell number'] = ct_groups.size()
    maj_sign = [-1, 1, 0]
    sign_dict = {-1: 'symmetric', 1: 'asymmetric', 0:'equal'}

    for i, sign in enumerate(maj_sign):
        sign_syn_df = syn_sign_df[syn_sign_df['majority syn sign'] == sign]
        sign_groups = sign_syn_df.groupby('celltype')
        majority_df.loc[i * num_cts: (i + 1) * num_cts - 1, 'syn sign'] = sign_dict[sign]
        for ct in unique_cts:
            if ct in sign_groups.groups.keys():
                ct_ind = np.where(unique_cts == ct_str)[0][0]
                num_ct_cells = len(sign_groups.get_group(ct))
                fraction_cells = num_ct_cells / len(ct_groups.get_group(ct))
                majority_df.loc[ct_ind + i * num_cts, 'cell number (syn number maj)'] = num_ct_cells
                majority_df.loc[ct_ind + i * num_cts, 'cell fraction (syn number maj)'] = fraction_cells
        sign_syn_df = syn_sign_df[syn_sign_df['majority syn area sign'] == sign]
        sign_groups = sign_syn_df.groupby('celltype')
        for ct in sign_groups.groups.keys():
            if ct in sign_groups:
                ct_ind = np.where(unique_cts == ct_str)[0][0]
                num_ct_cells = len(sign_groups.get_group(ct))
                fraction_cells = num_ct_cells / len(ct_groups.get_group(ct))
                majority_df.loc[ct_ind + i * num_cts, 'cell number (syn area maj)'] = num_ct_cells
                majority_df.loc[ct_ind + i * num_cts, 'cell fraction (syn area maj)'] = fraction_cells

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    majority_df.to_csv(f'{f_name}/ct_maj_sign.csv')

    log.info('Step 4/4: Plot params')
    ct_palette = cls.ct_palette(color_key, num=False)
    for param in param_list:
        if 'total' in param or 'str' in param:
            continue
        if 'sum syn area' in param:
            ylabel = f'{param} [µm]'
        else:
            ylabel = param
        sns.boxplot(data=syn_sign_df, x='celltype', y=param, palette=ct_palette, order=ct_str_list)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'{param}')
        plt.savefig(f'{f_name}/{param}_box.svg')
        plt.savefig(f'{f_name}/{param}_box.png')
        plt.close()
        if 'majority' in param:
            sns.catplot(data = syn_sign_df, x = 'celltype', y = param, palette=ct_palette, order=ct_str_list)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xlabel('celltype', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.title(f'{param}')
            plt.savefig(f'{f_name}/{param}_swarm.svg')
            plt.savefig(f'{f_name}/{param}_swarm.png')
            plt.close()
        else:
            sns.violinplot(data=syn_sign_df, x='celltype', y=param, palette=ct_palette, order=ct_str_list)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xlabel('celltype', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.title(f'{param}')
            plt.savefig(f'{f_name}/{param}_violin.svg')
            plt.savefig(f'{f_name}/{param}_violin.png')
            plt.close()

    sign_palette = {'symmetric': '#3287A8', 'asymmetric': '#E8AA47','equal': '#BD3748'}
    for param in maj_columns:
        if 'celltype' in param or 'sign' in param:
            continue
        sns.barplot(data = majority_df, x = 'celltype', y = param, hue = 'syn sign',
                    order = ct_str_list, palette=sign_palette)
        plt.ylabel(param, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'{param}')
        plt.savefig(f'{f_name}/{param}_maj_bar.svg')
        plt.savefig(f'{f_name}/{param}_maj_bar.png')
        plt.close()

    log.info('Analysis done')





