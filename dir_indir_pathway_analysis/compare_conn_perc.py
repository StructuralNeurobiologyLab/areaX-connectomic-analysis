#write script here to compare connectvity between two celltypes
#use synapse_dict_per_ct from connectivity_subgroups_per_ct
#make boxplot with hue

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.handler.basics import load_pkl2obj
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()
    #celltypes that are compared
    ct1 = 6
    ct2 = 7
    color_key = 'STNGPINTv6'
    fontsize = 20
    #select which incoming an outgoing celltypes should be plottet extra as well
    zoom_cts = ['GPe', 'GPi']
    if len(zoom_cts) > 0:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240529_j0251{version}_%s_%s_connectivity_comparison_f{fontsize}_zoom{zoom_cts}" % (
            ct_dict[ct1], ct_dict[ct2])
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240529_j0251{version}_%s_%s_connectivity_comparison_f{fontsize}" % (
                ct_dict[ct1], ct_dict[ct2])
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Connectivity comparison', log_dir=f_name + '/logs/')
    ct_colors = CelltypeColors(ct_dict = ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)

    conn_filename = 'cajal/scratch/users/arother/bio_analysis_results/general/240411_j0251v6_cts_percentages_mcl_200_ax50_synprob_0.60_TePkBrNGF_annot_bw_fs_20'
    log.info(f'Step 1/3: Load connectivity data from {conn_filename}')
    conn_dict = load_pkl2obj(f'{conn_filename}/synapse_dict_per_ct.pkl')

    log.info(f'Step 2/2 generate dataframe for celltypes {ct_dict[ct1]}, {ct_dict[ct2]}')
    #use similar code as in connectivity_fraction_per_ct
    key_list = ['outgoing synapse sum size percentage', 'outgoing synapse sum size',
                'incoming synapse sum size percentage', 'incoming synapse sum size' ]
    #ct_dict[10] = 'TP1'
    #ct_dict[11] = 'TP2'
    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    num_cts = len(celltypes)
    axon_cts = analysis_params.axon_cts()
    non_ax_celltypes = celltypes[np.in1d(np.arange(0, num_cts), axon_cts) == False]
    result_df_lst_ct1 = []
    result_df_lst_ct2 = []
    for key_name in key_list:
        conn_ct1 = conn_dict[ct1]
        conn_ct2 = conn_dict[ct2]
        if 'incoming' in key_name:
            plt_celltypes = celltypes
            if 'percentage' in key_name:
                conn_str = ' of '
            else:
                conn_str = ' from '
        else:
            plt_celltypes = non_ax_celltypes
            if 'percentage' in key_name:
                conn_str = ' of '
            else:
                conn_str = ' to '
        lengths_ct1 = [len(conn_ct1[key_name + conn_str + c]) for c in plt_celltypes]
        lengths_ct2 = [len(conn_ct2[key_name + conn_str + c]) for c in plt_celltypes]
        max_length_ct1 = np.max(lengths_ct1)
        max_length_ct2 = np.max(lengths_ct2)
        result_df_ct1 = pd.DataFrame(columns=plt_celltypes, index=range(max_length_ct1))
        result_df_ct2 = pd.DataFrame(columns=plt_celltypes, index=range(max_length_ct2))
        for i, c in enumerate(plt_celltypes):
            result_df_ct1.loc[0:lengths_ct1[i] - 1, c] = conn_ct1[key_name + conn_str + c]
            result_df_ct2.loc[0:lengths_ct2[i] - 1, c] = conn_ct2[key_name + conn_str + c]
        # fill up with zeros so that each cell that makes at least one synapse with another suitable cell is included in analysis
        result_df_ct1 = result_df_ct1.fillna(0)
        result_df_ct2 = result_df_ct2.fillna(0)
        result_df_lst_ct1.append(result_df_ct1)
        result_df_lst_ct2.append(result_df_ct2)

    #put in one big dataframe
    max_entries = np.max([len(result_df_lst_ct1[0]), len(result_df_lst_ct1[3]), len(result_df_lst_ct2[0]), len(result_df_lst_ct2[3])])
    result_df = pd.DataFrame(columns=np.hstack([key_list, 'plt celltype', 'conn celltype']), index = range(2 * max_entries * num_cts))
    result_df.loc[0: max_entries * num_cts - 1, 'plt celltype'] = ct_dict[ct1]
    result_df.loc[max_entries * num_cts: 2 * max_entries * num_cts - 1, 'plt celltype'] = ct_dict[ct2]
    for ci, ct in enumerate(celltypes):
        start_ct1 = ci * max_entries
        end_ct1 = (ci + 1) * max_entries -  1
        start_ct2 = max_entries * num_cts + ci * max_entries
        end_ct2 = max_entries * num_cts + (ci + 1) * max_entries - 1
        result_df.loc[start_ct1:  end_ct1, 'conn celltype'] = ct
        result_df.loc[start_ct2: end_ct2, 'conn celltype'] = ct
        for ki, k in enumerate(key_list):
            if not ('outgoing' in k and ct not in non_ax_celltypes):
                end_key_ct1 = start_ct1 + len(result_df_lst_ct1[ki][ct]) - 1
                end_key_ct2 = start_ct2 + len(result_df_lst_ct2[ki][ct]) - 1
                result_df.loc[start_ct1: end_key_ct1, k] = np.array(result_df_lst_ct1[ki][ct])
                result_df.loc[start_ct2: end_key_ct2, k] = np.array(result_df_lst_ct2[ki][ct])

    #remove rows where incoming synapse sum size is NaN, leaves 0 values in there
    result_df = result_df[result_df['incoming synapse sum size'].isnull() == False]
    result_df = result_df.reset_index(drop= True)
    result_df.to_csv(f'{f_name}/{ct_dict[ct1]}_{ct_dict[ct2]}_conn_results.csv')

    log.info('Step 3/3: Get statistics between two celltypes and plot results')
    ranksum_results = pd.DataFrame()
    ct1_results = result_df[result_df['plt celltype'] == ct_dict[ct1]]
    ct1_results_ct_lst = [ct1_results[ct1_results['conn celltype'] == ct] for ct in celltypes]
    ct2_results = result_df[result_df['plt celltype'] == ct_dict[ct2]]
    ct2_results_ct_lst = [ct2_results[ct2_results['conn celltype'] == ct] for ct in celltypes]
    outgoing_result_df = result_df.dropna()
    for key in key_list:
        for ci, ct in enumerate(celltypes):
            if not ('outgoing' in key and ct not in non_ax_celltypes):
                ct1_results_ct = ct1_results_ct_lst[ci]
                ct2_results_ct = ct2_results_ct_lst[ci]
                stats, p_value = ranksums(ct1_results_ct[key], ct2_results_ct[key])
                ranksum_results.loc[f'{ct} stats', key] = stats
                ranksum_results.loc[f'{ct} p-value', key] = p_value
        if 'outgoing' in key:
            plot_result_df = outgoing_result_df
        else:
            plot_result_df = result_df
        sns.boxplot(data = plot_result_df, x = 'conn celltype', y = key, hue = 'plt celltype', palette=ct_palette)
        plt.title(key)
        if 'percentage' in key:
            ylabel = '%'
        else:
            ylabel = 'sum of synaptic area [µm²]'
        plt.ylabel(ylabel)
        if 'incoming' in key:
            plt.xlabel('presynaptic celltypes')
        else:
            plt.xlabel('postsynaptic celltypes')
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.savefig(f'{f_name}/comp_syn_box_{key}.png')
        plt.savefig(f'{f_name}/comp_syn_box_{key}.svg')
        plt.close()
        if len(zoom_cts) > 0:
            zoom_result_df = plot_result_df[np.in1d(plot_result_df['conn celltype'], zoom_cts)]
            sns.boxplot(data=zoom_result_df, x='conn celltype', y=key, hue='plt celltype', palette=ct_palette)
            if 'incoming' in key:
                plt.xlabel('presynaptic celltypes')
            else:
                plt.xlabel('postsynaptic celltypes')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/comp_syn_box_{key}_zoom.png')
            plt.savefig(f'{f_name}/comp_syn_box_{key}_zoom.svg')
            plt.close()
    ranksum_results.to_csv(f'{f_name}/ranksum_results_{ct_dict[ct1]}_{ct_dict[ct2]}.csv')
