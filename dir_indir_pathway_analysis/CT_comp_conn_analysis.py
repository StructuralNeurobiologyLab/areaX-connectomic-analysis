#get information about compartment specific connectivity between different cells
#similar to CT_input_syn_distance_analysis
if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import get_compartment_specific_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import ranksums
    import scipy
    import seaborn as sns
    import matplotlib.pyplot as plt

    global_params.wd = "ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    start = time.time()
    bio_params = Analysis_Params(global_params.wd)
    ct_dict = bio_params.ct_dict()
    #min_comp_len = bio_params.min_comp_length()
    min_comp_len = 200
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGP'
    post_ct = 10
    post_ct_str = ct_dict[post_ct]
    #comp color keys: 'MudGrays', 'GreenGrays', 'TeYw', 'NeRe', 'BeRd, TeBk'}
    comp_color_key = 'TeBk'
    save_svg = True
    fontsize = 20
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221129_j0251v4_%s_input_comps_mcl_%i_synprob_%.2f_%s_%s" % (
    post_ct_str, min_comp_len, syn_prob, color_key, comp_color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Analysis of synaptic inputs to compartments of %s' % post_ct_str, log_dir=f_name + '/logs/')
    cts_for_loading = [2, 3, 4, 6, 7, 10]
    cts_str_analysis = [ct_dict[ct] for ct in cts_for_loading]
    num_cts = len(cts_for_loading)
    log.info(
        "min_comp_len = %i, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s" % (
        min_comp_len, syn_prob, min_syn_size, exclude_known_mergers, color_key))
    log.info(f'Distance of synapses for celltypes {cts_str_analysis} will be compared to {post_ct_str}')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/3: Load celltypes and check suitability")

    axon_cts = bio_params.axon_cts()
    cls = CelltypeColors()
    ct_palette = cls.ct_palette(color_key, num=False)
    comp_cls = CompColors()
    comp_palette = comp_cls.comp_palette(comp_color_key, num = False, denso=True)
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = bio_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
        if ct in axon_cts:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=True,
                                                    max_path_len=None)
        else:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked

    number_ids = [len(suitable_ids_dict[ct]) for ct in cts_for_loading]
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")
    time_stamps = [time.time()]
    step_idents = ['loading cells']

    log.info("Step 2/3: Get compartments for synapses to %s" % post_ct_str)
    #set uo to dataframes, one with results summarised per celltype (float) and another
    #with results summarised per postsynaptic cell (np.array)
    compartments = ['soma', 'spine neck', 'spine head', 'dendritic shaft']
    num_comps = len(compartments)
    complete_param_titles = [f'Number of synapses from ct to {post_ct_str} per {post_ct_str} cell',
                        f'Summed synapse size from ct to {post_ct_str} per {post_ct_str} cell',
                        f'Percentage of synapses from ct to {post_ct_str} per {post_ct_str} cell',
                        f'Percentage of synapse sizes from ct to {post_ct_str} per {post_ct_str} cell',
                        f'Number of synapses from ct to {post_ct_str}',
                        f'Summed synapse size from ct to {post_ct_str}',
                        f'Percentage of synapses from ct to {post_ct_str}',
                        f'Percentage of synapse sizes from ct to {post_ct_str}']
    param_titles = [pt.split('from')[0] for pt in complete_param_titles]
    columns = np.hstack([np.array(param_titles[:int(len(param_titles)/2)]), np.array(['celltype', 'compartment'])])
    all_comps_results_dict_percell = pd.DataFrame(columns=columns, index=range(num_comps * num_cts * len(suitable_ids_dict[post_ct])))
    all_comps_results_dict = pd.DataFrame(columns = columns, index=range(num_comps * num_cts))
    num_postcellids = len(suitable_ids_dict[post_ct])
    for ic, ct in enumerate(tqdm(cts_for_loading)):
        ct_str = ct_dict[ct]
        #get median, min, max synapse distance to soma per cell
        #function uses multiprocessing
        if ct == post_ct:
            percell_params, syn_params, all_syns_df = get_compartment_specific_connectivity(ct_post=post_ct,
                                                                               cellids_post=suitable_ids_dict[post_ct],
                                                                               sd_synssv=sd_synssv,
                                                                               syn_prob=syn_prob,
                                                                               min_syn_size=min_syn_size,
                                                                               ct_pre=None, cellids_pre=None)
        else:
            percell_params, syn_params, all_syns_df = get_compartment_specific_connectivity(ct_post=post_ct,
                                                                               cellids_post=suitable_ids_dict[post_ct],
                                                                               sd_synssv=sd_synssv,
                                                                               syn_prob=syn_prob,
                                                                               min_syn_size=min_syn_size,
                                                                               ct_pre=ct, cellids_pre=suitable_ids_dict[ct])
        #parameters per postsynaptic cell
        #syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
        #parameters for all synapses independent of cell
        #all_syn_numbers, all_sum_sizes, all_syn_nums_perc, all_syn_sizes_perc = syn_params
        for i_comp, compartment in enumerate(compartments):
            #fill in numbers that summarise all synapses per compartment per celltype
            ind_all_syns = ic*num_comps + i_comp
            all_comps_results_dict.loc[ind_all_syns, 'compartment'] = compartment
            all_comps_results_dict.loc[ind_all_syns, 'celltype'] = ct_str
            # fill in data per postsynaptic cell per compartment per celltype
            start_ind = ic * num_comps * num_postcellids + i_comp * num_postcellids
            len_comp_percell = len(percell_params[0][compartment])
            end_ind = start_ind + len_comp_percell - 1
            all_comps_results_dict_percell.loc[start_ind: end_ind, 'compartment'] = compartment
            all_comps_results_dict_percell.loc[start_ind: end_ind, 'celltype'] = ct_str
            for iy in range(len(syn_params)):
                all_comps_results_dict.loc[ind_all_syns, param_titles[iy]] = syn_params[iy][compartment]
                all_comps_results_dict_percell.loc[start_ind: end_ind, param_titles[iy]] = percell_params[iy][compartment]

        f_name_ct = f'{f_name}/{ct_str}'
        if not os.path.exists(f_name_ct):
            os.mkdir(f_name_ct)
        df_percell = all_comps_results_dict_percell[all_comps_results_dict_percell['celltype'] == ct_str]
        df_percell = df_percell.convert_dtypes()
        df = all_comps_results_dict[all_comps_results_dict['celltype'] == ct_str]
        for ik, key in enumerate(complete_param_titles):
            param_title = param_titles[ik]
            if 'cell' in key:
                df_percell[param_title] = df_percell[param_title].astype(float)
                sns.boxplot(x = 'compartment', y = param_title, data = df_percell, palette=comp_palette)
                plt.ylabel(param_title)
                plt.title(key)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if 'Percentage' in key:
                    plt.ylim(0, 100)
                plt.savefig(f'{f_name_ct}/{param_title}_per_cell_box.png')
                if save_svg:
                    plt.savefig(f'{f_name_ct}/{param_title}_per_cell_box.svg')
                plt.close()
                sns.stripplot(x = 'compartment', y = param_title, data = df_percell, color="black", alpha=0.2,
                              dodge=True, size=2)
                sns.violinplot(x = 'compartment', y = param_title, data = df_percell, palette=comp_palette)
                plt.ylabel(param_title)
                plt.title(key)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if 'Percentage' in key:
                    plt.ylim(0, 100)
                plt.savefig(f'{f_name_ct}/{param_title}_per_cell_violin.png')
                if save_svg:
                    plt.savefig(f'{f_name_ct}/{param_title}_per_cell_violin.svg')
                plt.close()
            else:
                sns.barplot(x = 'compartment', y = param_title, data = df, palette=comp_palette)
                plt.ylabel(param_title)
                plt.title(key)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if 'Percentage' in key:
                    plt.ylim(0, 100)
                plt.savefig(f'{f_name_ct}/{param_title}_allsyns_bar.png')
                if save_svg:
                    plt.savefig(f'{f_name_ct}/{param_title}_allsyns_bar.svg')
                plt.close()

    #remove empty rows
    all_comps_results_dict_percell = all_comps_results_dict_percell.dropna()
    all_comps_results_dict_percell.to_csv(f'{f_name}/all_comps_results_per_{post_ct_str}_cell.csv')
    all_comps_results_dict.to_csv(f'{f_name}/all_comps_results_cts.csv')
    time_stamps = [time.time()]
    step_idents = ['get compartment specific synapse information']

    log.info("Step 3/3 Plot results for comparison between celltypes and calculate statistics")
    for ik, key in enumerate(tqdm(complete_param_titles)):
        #use ranksum test (non-parametric) to calculate results
        param_title = param_titles[ik]
        if 'cell' in key:
            #make plots for data which is summarised per postsynaptic cell
            ranksum_results = pd.DataFrame()
            for comp in compartments:
                comp_res_pc = all_comps_results_dict_percell[all_comps_results_dict_percell['compartment'] == comp]
                comp_res_pc[param_title] = comp_res_pc[param_title].astype(float)
                for c1 in cts_for_loading:
                    c1_str = ct_dict[c1]
                    c1_res = comp_res_pc[comp_res_pc['celltype'] == c1_str][param_title]
                    p_c1 = np.array(c1_res).astype(float)
                    for c2 in cts_for_loading:
                        if c1 >= c2:
                            continue
                        c2_str = ct_dict[c2]
                        c2_res = comp_res_pc[comp_res_pc['celltype'] == c2_str][param_title]
                        p_c2 = np.array(c2_res).astype(float)
                        stats, p_value = ranksums(p_c1, p_c2, nan_policy = 'omit')
                        ranksum_results.loc["stats " + comp + ' of ' + post_ct_str, c1_str + " vs " + c2_str] = stats
                        ranksum_results.loc["p value " + comp + ' of ' + post_ct_str, c1_str + " vs " + c2_str] = p_value
                # make violinplot, boxplot per compartment with all celltypes
                ylabel = param_title
                sns.stripplot(x = 'celltype', y = param_title, data=comp_res_pc, color="black", alpha=0.2,
                              dodge=True, size=2)
                sns.violinplot(x = 'celltype', y = param_title, data=comp_res_pc, inner="box",
                               palette=ct_palette)
                plt.title(param_title + ' to ' + comp + ' of ' + post_ct_str)
                plt.ylabel(ylabel)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if 'Percentage' in key:
                    plt.ylim(0, 100)
                plt.savefig('%s/%s_syn_comps_%s_percell_violin.png' % (f_name, param_title, comp))
                if save_svg:
                    plt.savefig('%s/%s_syn_comps_%s_percell_violin.svg' % (f_name, param_title, comp))
                plt.close()
                sns.boxplot(x = 'celltype', y = param_title, data=comp_res_pc,
                               palette=ct_palette)
                plt.title(param_title + ' to ' + comp + ' of ' + post_ct_str)
                plt.ylabel(ylabel)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if 'Percentage' in key:
                    plt.ylim(0, 100)
                plt.savefig('%s/%s_syn_comps_%s_percell_box.png' % (f_name, param_title, comp))
                if save_svg:
                    plt.savefig('%s/%s_syn_comps_%s_percell_box.png' % (f_name, param_title, comp))
                plt.close()
            ranksum_results.to_csv(f'{f_name}/{param_title}_ranksum_results.csv')
            #make plot with all compartments
            all_comps_results_dict_percell[param_title] = all_comps_results_dict_percell[param_title].astype(float)
            ylabel = param_title
            sns.stripplot(x = 'celltype', y = param_title, hue= 'compartment', data=all_comps_results_dict_percell, color="black", alpha=0.2,
                          dodge=True, size=2, legend=False)
            sns.violinplot(x = 'celltype', y = param_title, hue= 'compartment', data=all_comps_results_dict_percell, inner="box",
                           palette=comp_palette)
            plt.title(param_title + ' to '+ post_ct_str)
            plt.ylabel(ylabel)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if 'Percentage' in key:
                plt.ylim(0, 100)
            plt.savefig('%s/%s_syn_percell_violin.png' % (f_name, param_title))
            if save_svg:
                plt.savefig('%s/%s_syn_percell_violin.svg' % (f_name, param_title))
            plt.close()
            sns.boxplot(x = 'celltype', y = param_title, hue= 'compartment', data=all_comps_results_dict_percell,
                        palette=comp_palette)
            plt.title(param_title + ' to ' + post_ct_str)
            plt.ylabel(ylabel)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if 'Percentage' in key:
                plt.ylim(0, 100)
            plt.savefig('%s/%s_syn_percell_box.png' % (f_name, param_title))
            if save_svg:
                plt.savefig('%s/%s_syn_percell_box.svg' % (f_name, param_title))
            plt.close()
        else:
            #make plots for summary with all synapses together
            sns.barplot(x='celltype', y=param_title, hue='compartment', data=all_comps_results_dict,
                        palette=comp_palette)
            plt.title(param_title + ' to ' + post_ct_str + ' for all synapses per ct')
            plt.ylabel(ylabel)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if 'Percentage' in key:
                plt.ylim(0, 100)
            plt.savefig('%s/%s_syn_cts_box.png' % (f_name, param_title))
            if save_svg:
                plt.savefig('%s/%s_syn_cts_box.svg' % (f_name, param_title))
            plt.close()


    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    log.info('Compartment synapse analysis done')
    time_stamps = time.time()
    step_idents = ['Plotting finished']