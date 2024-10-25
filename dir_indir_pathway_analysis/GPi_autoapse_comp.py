#get function to compare synapse density, mito density axon and vesicle density in GPi cells with and without autapses
#also compare soma size

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums
    import matplotlib.colors as co

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia= False)
    min_comp_len = 200
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    ct = 7
    ct_str = ct_dict[ct]
    use_selected_ids = True
    fontsize = 16
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241016_j0251{version}_{ct_str}_autapse_comp_mcl_%i_ms_%.1f_%i" % (
        min_comp_len, min_syn_size, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('synprob_mesh_visualisation',
                             log_dir=f_name + '/logs/')
    log.info(
        f"min_comp_len = %i, minimum synapse size = %.1f, syn prob thresh = %.1f" % (
            min_comp_len, min_syn_size, syn_prob_thresh))
    log.info(f'celltype {ct_str} will be analysed')

    #load GPi ids from merger list, remove mergers and separate autapse GPi from non-autapse GPi
    ct_autapse_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                        '240723_j0251v6_all_cellids_for_exclusion/' \
                       '241024_all_full_cell_ids_no_msn_manuall_checks_final.csv'

    log.info(f'Load information about manually checked autapses from {ct_autapse_path}')
    ct_autapse_df = pd.read_csv(ct_autapse_path)
    #get only GPi and remove mergers
    ct_autapse_df = ct_autapse_df[ct_autapse_df['celltype'] == ct_str]
    ct_autapse_df = ct_autapse_df[ct_autapse_df['include?'] == 'y']
    log.info(f'{len(ct_autapse_df)} cells of {ct_str} are suitable and without mergers.')
    cellids_autapse = np.array(ct_autapse_df['cellid'][ct_autapse_df['autapse?'] == 'y']).astype(int)
    cellids_no_autapse = np.array(ct_autapse_df['cellid'][ct_autapse_df['autapse?'] == 'n']).astype(int)
    log.info(f'{len(cellids_autapse)} cells have at least one autapse ({100 * len(cellids_autapse)/ len(ct_autapse_df):.2f} %)')
    morph_info_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                      '240613_j0251v6_ct_morph_analyses_mcl_200_ax200_TeBKv6MSNyw_fs20npca2_umap5_fc_noMSN_synaxmives/' \
                       'ct_morph_df.csv'
    log.info(f'morphological information loaded from {morph_info_path}')
    morph_df = pd.read_csv(morph_info_path, index_col=0)
    #get only suitable ids
    suitable_cellids = np.array(ct_autapse_df['cellid']).astype(int)
    #only do this if morph info is also sorted
    suitable_cellids = np.sort(suitable_cellids)
    morph_df = morph_df[np.in1d(morph_df['cellid'], suitable_cellids)]
    log.info(f'for {len(morph_df)} cells morphological parameters were found in morph table')
    if len(morph_df) < len(suitable_cellids):
        non_morph_cellids = suitable_cellids[np.in1d(suitable_cellids, morph_df['cellid']) == False]
        log.info(f' These cellids were not in the morph parameter table: {non_morph_cellids}')
    assert (np.unique(np.array(morph_df['celltype'])) == ct_str)
    autapse_inds = np.in1d(morph_df['cellid'], np.sort(cellids_autapse))
    non_autapse_inds = np.in1d(morph_df['cellid'], np.sort(cellids_no_autapse))
    morph_df.loc[autapse_inds, 'autapse group'] = 'autapse'
    morph_df.loc[non_autapse_inds, 'autapse group'] = 'no autapse'
    morph_df.to_csv(f'{f_name}/{ct_str}_autapse_df')

    #do statistical test on the parameters
    log.info('Do ranksum test to compare autapse and non-autapse in different parameters')
    # save mean, median and std for all parameters per ct
    aut_gr_str = np.unique(morph_df['autapse group'])
    aut_groups = morph_df.groupby('autapse group')
    summary_gr_df = pd.DataFrame(index=aut_gr_str)
    summary_gr_df['numbers'] = aut_groups.size()
    param_list = list(morph_df.columns)[2:-1]
    for key in param_list:
        summary_gr_df[f'{key} mean'] = aut_groups[key].mean()
        summary_gr_df[f'{key} std'] = aut_groups[key].std()
        summary_gr_df[f'{key} median'] = aut_groups[key].median()
    summary_gr_df.to_csv(f'{f_name}/summary_params_autapse_groups_{ct_str}.csv')

    # ranksum results to compare group
    ranksum_columns = [f'autapse vs no autapse {ct_str} stats', f'autapse vs no autapse {ct_str} p-value']
    ranksum_df = pd.DataFrame(columns=ranksum_columns, index=param_list)
    for key in param_list:
            ranksum_res = ranksums(aut_groups.get_group('autapse')[key], aut_groups.get_group('no autapse')[key])
            ranksum_df.loc[key, f'autapse vs no autapse {ct_str} stats'] = ranksum_res[0]
            ranksum_df.loc[key, f'autapse vs no autapse {ct_str} p-value'] = ranksum_res[1]

    ranksum_df.to_csv(f'{f_name}/ranksum_results_{ct_str}.csv')

    #plot violin plot
    aut_palette = {'no autapse': '#232121', 'autapse': '#15AEAB'}
    strip_palette = {'no autapse': 'white', 'autapse': 'black'}
    #strip_colors = ['white', 'black']
    #aut_colors = ['#232121', '#15AEAB']
    #colors_rgba = co.to_rgba_array(aut_colors)
    #colors_rgba_int = colors_rgba * 255
    for param in param_list:
        sns.stripplot(data=morph_df, x='autapse group',hue='autapse group', y=param, palette=strip_palette, alpha=0.2,
                      dodge=False, size=2, legend = False)
        sns.violinplot(data=morph_df, x='autapse group',hue='autapse group', y=param, palette=aut_palette, inner="box", legend = False, dodge=False)
        plt.title(param)
        plt.ylabel(param, fontsize=fontsize)
        plt.xlabel('autapse group', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{param}_violin.png')
        plt.savefig(f'{f_name}/{param}_violin.svg')
        plt.close()

    log.info('Analysis done')