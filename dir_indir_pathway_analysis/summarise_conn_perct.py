#summarise connectivity for a given celltype
#use similar code as in compare_conn_ct
#make boxplots per celltype but just with one or two celltypes and then other

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

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    #celltypes that are compared
    ct = 2
    color_key = 'STNGP'
    fontsize_jointplot = 12
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/231130_j0251v5_%s_connectivity_summary" % (
            ct_dict[ct])
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Connectivity summary', log_dir=f_name + '/logs/')
    ct_colors = CelltypeColors()
    ct_palette = ct_colors.ct_palette(key=color_key)
    ct_palette['other'] = '#707070'
    ct_str = ct_dict[ct]
    incoming_celltypes = [3, 4]
    incoming_ct_str = [ct_dict[i] for i in incoming_celltypes]
    outgoing_celltypes = [6, 7]
    outgoing_ct_str = [ct_dict[i] for i in outgoing_celltypes]
    log.info(f'Summary will focus on celltype {ct_str}, incoming synapses from {incoming_ct_str}'
             f'and outgoing synapses to {outgoing_ct_str}')

    conn_filename = 'cajal/scratch/users/arother/bio_analysis_results/general/230920_j0251v5_cts_percentages_ngf_subgroups_mcl_200_ax50_synprob_0.60_TePkBrNGF_annot_bw_fs_12'
    log.info(f'Step 1/3: Load connectivity data from {conn_filename}')
    conn_dict = load_pkl2obj(f'{conn_filename}/synapse_dict_per_ct.pkl')

    log.info(f'Step 2/2 generate dataframe for celltypes {ct_dict[ct]}')
    #use similar code as in connectivity_fraction_per_ct
    key_list = ['outgoing synapse sum size percentage', 'outgoing synapse sum size',
                'incoming synapse sum size percentage', 'incoming synapse sum size' ]
    ct_dict[10] = 'TP1'
    ct_dict[11] = 'TP2'
    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    axon_cts = analysis_params.axon_cts()
    num_cts = len(celltypes)
    non_ax_celltypes = celltypes[np.in1d(np.arange(0, num_cts), axon_cts) == False]
    result_df_lst_in = []
    key_list_in = []
    result_df_lst_out = []
    key_list_out = []
    for key_name in key_list:
        conn_ct = conn_dict[ct]
        if 'incoming' in key_name:
            plt_celltypes = celltypes
            sum_celltypes = incoming_ct_str
            key_list_in.append(key_name)
            if 'percentage' in key_name:
                conn_str = ' of '
            else:
                conn_str = ' from '
        else:
            plt_celltypes = non_ax_celltypes
            sum_celltypes = outgoing_ct_str
            key_list_out.append(key_name)
            if 'percentage' in key_name:
                conn_str = ' of '
            else:
                conn_str = ' to '
        lengths = [len(conn_ct[key_name + conn_str + c]) for c in plt_celltypes]
        max_length = np.max(lengths)
        all_result_df = pd.DataFrame(columns=plt_celltypes, index=range(max_length))
        for i, c in enumerate(plt_celltypes):
            all_result_df.loc[0:lengths[i] - 1, c] = conn_ct[key_name + conn_str + c]
        other_celltypes_sum = np.sum(all_result_df, axis = 1)
        for c in sum_celltypes:
            other_celltypes_sum = other_celltypes_sum - all_result_df[c]
        all_result_df['other'] = other_celltypes_sum
        result_df_key = pd.DataFrame(all_result_df, columns = np.hstack([sum_celltypes, 'other']))
        # fill up with zeros so that each cell that makes at least one synapse with another suitable cell is included in analysis
        result_df_key = result_df_key.fillna(0)
        if 'incoming' in key_name:
            result_df_lst_in.append(result_df_key)
        else:
            result_df_lst_out.append(result_df_key)

    #put in two big dataframes
    num_cts_in = len(incoming_ct_str) + 1
    max_entries_in = np.max(len(result_df_lst_in[0]))
    incoming_result_df = pd.DataFrame(columns=np.hstack([key_list_in, 'celltype']),
                                      index=range(max_entries_in * num_cts_in))
    columns_in = np.hstack([incoming_ct_str, 'other'])
    for ci, col in enumerate(columns_in):
        start = ci * max_entries_in
        end = (ci + 1) * max_entries_in - 1
        incoming_result_df.loc[start:  end, 'celltype'] = col
        for ki, k in enumerate(key_list_in):
            end_key = start + len(result_df_lst_in[ki][col]) - 1
            incoming_result_df.loc[start: end_key, k] = np.array(result_df_lst_in[ki][col])
    # remove rows where incoming synapse sum size is NaN, leaves 0 values in there
    incoming_result_df = incoming_result_df[incoming_result_df['incoming synapse sum size'].isnull() == False]
    incoming_result_df = incoming_result_df.reset_index(drop=True)
    incoming_result_df.to_csv(f'{f_name}/{ct_dict[ct]}_conn_results_incoming.csv')

    num_cts_out = len(outgoing_ct_str) + 1
    max_entries_out = np.max(len(result_df_lst_out[0]))
    outgoing_result_df = pd.DataFrame(columns=np.hstack([key_list_out, 'celltype']),
                                      index=range(max_entries_out * num_cts_out))
    columns_out = np.hstack([outgoing_ct_str, 'other'])
    for ci, col in enumerate(columns_out):
        start = ci * max_entries_out
        end = (ci + 1) * max_entries_out - 1
        outgoing_result_df.loc[start:  end, 'celltype'] = col
        for ki, k in enumerate(key_list_out):
            end_key = start + len(result_df_lst_out[ki][col]) - 1
            outgoing_result_df.loc[start: end_key, k] = np.array(result_df_lst_out[ki][col])
    # remove rows where incoming synapse sum size is NaN, leaves 0 values in there
    outgoing_result_df = outgoing_result_df[outgoing_result_df['outgoing synapse sum size'].isnull() == False]
    outgoing_result_df = outgoing_result_df.reset_index(drop=True)
    outgoing_result_df.to_csv(f'{f_name}/{ct_dict[ct]}_conn_results_outgoing.csv')

    log.info('Step 3/3: Plot results')
    for key in key_list_in:
        sns.boxplot(data = incoming_result_df, x = 'celltype', y = key, palette=ct_palette)
        plt.title(key)
        if 'percentage' in key:
            ylabel = '%'
        else:
            ylabel = 'sum of synaptic area [µm²]'
        plt.ylabel(ylabel)
        plt.xlabel('presynaptic celltypes')
        plt.savefig(f'{f_name}/comp_syn_box_{key}.png')
        plt.savefig(f'{f_name}/comp_syn_box_{key}.svg')
        plt.close()

    for key in key_list_out:
        sns.boxplot(data = outgoing_result_df, x = 'celltype', y = key, palette=ct_palette)
        plt.title(key)
        if 'percentage' in key:
            ylabel = '%'
        else:
            ylabel = 'sum of synaptic area [µm²]'
        plt.ylabel(ylabel)
        plt.xlabel('postsynaptic celltypes')
        plt.savefig(f'{f_name}/comp_syn_box_{key}.png')
        plt.savefig(f'{f_name}/comp_syn_box_{key}.svg')
        plt.close()

    log.info('Analysis done')