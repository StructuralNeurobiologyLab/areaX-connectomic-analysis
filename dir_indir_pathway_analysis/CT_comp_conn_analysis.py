#get information about compartment specific connectivity between different cells
#similar to CT_input_syn_distance_analysis
if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    form cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connecivity_between2cts import get_compartment_specific_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.synapse_input_distance import get_syn_distances
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import analysis_params
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
    bio_params = analysis_params
    ct_dict = bio_params.ct_dict()
    min_comp_len = bio_params.min_comp_length()
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    post_ct = 6
    post_ct_str = ct_dict[post_ct]
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221109_j0251v4_%s_input_comps_mcl_%i_synprob_%.2f_%s" % (
    post_ct_str, min_comp_len, syn_prob, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Analysis of synaptic inputs to compartments of %s' % post_ct_str, log_dir=f_name + '/logs/')
    cts_for_loading = [0, 2, 3, 6, 7, 8]
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
        suitable_ids_dict[ct] = cellids

    number_ids = [len(suitable_ids_dict[ct]) for ct in cts_for_loading]
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")
    time_stamps = [time.time()]
    step_idents = ['loading cells']

    log.info("Step 2/3: Get compartments for synapses to %s" % post_ct_str)
    compartments = ['soma', 'spine neck', 'spine head', 'dendritic shaft']
    syn_numbers_percell_cts = {}
    syn_sum_sizes_percell_cts = {}
    syn_number_perc_percell_cts = {}
    syn_sum_sizes_perc_percell_cts = {}
    for compartment in compartments:
        syn_numbers_percell_cts[compartment] = pd.DataFrame(columns=cts_str_analysis)
        syn_sum_sizes_percell_cts[compartment] = pd.DataFrame(columns=cts_str_analysis)
        syn_number_perc_percell_cts[compartment] = pd.DataFrame(columns=cts_str_analysis)
        syn_sum_sizes_perc_percell_cts[compartment] = pd.DataFrame(columns=cts_str_analysis)
    syn_numbers_cts = pd.DataFrame(columns=cts_str_analysis, index=compartments)
    syn_sum_sizes_cts = pd.DataFrame(columns=cts_str_analysis, index=compartments)
    syn_numbers_perc_cts = pd.DataFrame(columns=cts_str_analysis, index=compartments)
    syn_sum_sizes_perc_cts = pd.DataFrame(columns=cts_str_analysis, index=compartments)
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        #get median, min, max synapse distance to soma per cell
        #function uses multiprocessing
        if ct == post_ct_str:
            percell_params, syn_params = get_compartment_specific_connectivity(ct_post=post_ct,
                                                                               cellids_post=suitable_ids_dict[post_ct],
                                                                               sd_synssv=sd_synssv,
                                                                               syn_prob=syn_prob,
                                                                               min_syn_size=min_syn_size,
                                                                               ct_pre=None, cellids_pre=None)
        else:
            percell_params, syn_params = get_compartment_specific_connectivity(ct_post=post_ct,
                                                                               cellids_post=suitable_ids_dict[post_ct],
                                                                               sd_synssv=sd_synssv,
                                                                               syn_prob=syn_prob,
                                                                               min_syn_size=min_syn_size,
                                                                               ct_pre=ct, cellids_pre=suitable_ids_dict[ct])
        #parameters per postsynaptic cell
        syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
        #parameters for all synapses independent of cell
        all_syn_numbers, all_sum_sizes, all_syn_nums_perc, all_syn_sizes_perc = syn_params
        results_dict = {f'Number of synapses from {ct} to {post_ct_str} per {post_ct_str} cell': syn_numbers_ct,
                        f'Summed synapse size from {ct} to {post_ct_str} per {post_ct_str} cell': sum_sizes_ct,
                        f'Percentage of synapses from {ct} to {post_ct_str} per {post_ct_str} cell': syn_number_perc_ct,
                        f'Percentage of synapse sizes from {ct} to {post_ct_str} per {post_ct_str} cell': sum_sizes_perc_ct,
                        f'Number of synapses from {ct} to {post_ct_str}': all_syn_numbers,
                        f'Summed synapse size from {ct} to {post_ct_str}': all_sum_sizes,
                        f'Percentage of synapses from {ct} to {post_ct_str}': all_syn_nums_perc,
                        f'Percentage of synapse sizes from {ct} to {post_ct_str}': all_syn_sizes_perc}
        for compartment in compartments:
            syn_numbers_percell_cts[compartment][ct] = syn_numbers_ct[compartment]
            syn_sum_sizes_percell_cts[compartment][ct] = sum_sizes_ct[compartment]
            syn_number_perc_percell_cts[compartment][ct] = syn_number_perc_ct[compartment]
            syn_sum_sizes_perc_percell_cts[compartment][ct] = sum_sizes_perc_ct[compartment]
            syn_numbers_cts.loc[compartment, ct] = all_syn_numbers[compartment]
            syn_sum_sizes_cts.loc[compartment, ct] = all_sum_sizes[compartment]
            syn_numbers_perc_cts.loc[compartment, ct] = all_syn_nums_perc[compartment]
            syn_sum_sizes_perc_cts.loc[compartment, ct] = all_syn_sizes_perc[compartment]

        f_name_ct = f'{f_name}/{ct_str}'
        if not os.path.exists(f_name_ct):
            os.mkdir(f_name_ct)
        ct_color = ct_palette[ct_str]
        for key in results_dict:
            results_dict[key] = pd.DataFrame(results_dict[key])
            if 'cell' in key:
                sns.boxplot(data = results_dict[key], color=ct_color)
                param_title = key.split(' from ')[0]
                plt.ylabel(param_title)
                plt.title(key)
                plt.savefig(f'{f_name_ct}/param_title_per_cell_box.png')
                plt.close()
                sns.stripplot(data=results_dict[key], color="black", alpha=0.2,
                              dodge=True, size=2)
                sns.violinplot(data=results_dict[key], color=ct_color)
                plt.ylabel(param_title)
                plt.title(key)
                plt.savefig(f'{f_name_ct}/param_title_per_cell_box.png')
                plt.close()
            else:
                sns.barplot(data=results_dict[key], color=ct_color)
                param_title = key.split(' from ')[0]
                plt.ylabel(param_title)
                plt.title(key)
                plt.savefig(f'{f_name_ct}/param_title_per_cell_box.png')
                plt.close()

    cts_results_dict = {f'Number of synapses per {post_ct_str} cell': syn_numbers_percell_cts,
                    f'Summed synapse size per {post_ct_str} cell': syn_sum_sizes_percell_cts,
                    f'Percentage of synapses per {post_ct_str} cell': syn_number_perc_percell_cts,
                    f'Percentage of synapse sizes per {post_ct_str} cell': syn_sum_sizes_perc_percell_cts,
                    f'Number of synapses to {post_ct_str}': syn_numbers_cts,
                    f'Summed synapse size to {post_ct_str}': syn_sum_sizes_cts,
                    f'Percentage of synapses to {post_ct_str}': syn_numbers_perc_cts,
                    f'Percentage of synapse sizes to {post_ct_str}': syn_sum_sizes_perc_cts}
    write_obj2pkl('%s/comp_result_dict.pkl' % f_name, cts_results_dict)
    for key in cts_results_dict:
        if 'cell' in key:
            continue
        cts_results_dict[key].to_csv(f'{f_name}/{key}.csv')
    time_stamps = [time.time()]
    step_idents = ['get compartment specific synapse information']

    log.info("Step 3/3 Plot results for comparison between celltypes and calculate statistics")
    ranksum_results = pd.DataFrame()
    for key in cts_results_dict.keys():
        #use ranksum test (non-parametric) to calculate results
        for c1 in cts_for_loading:
            for c2 in cts_for_loading:
                if c1 >= c2:
                    continue
                p_c1 = np.array(param[ct_dict[c1]]).astype(float)
                p_c2 = np.array(param[ct_dict[c2]]).astype(float)
                stats, p_value = ranksums(p_c1, p_c2, nan_policy = 'omit')
                ranksum_results.loc["stats " + str_params[i] + ' of ' + dist2ct_str, ct_dict[c1] + " vs " + ct_dict[c2]] = stats
                ranksum_results.loc["p value " + str_params[i] + ' of ' + dist2ct_str, ct_dict[c1] + " vs " + ct_dict[c2]] = p_value
        #make violinplot, boxplot, histplot
        ylabel = 'distance in µm'
        sns.stripplot(data=param, color="black", alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(data=param, inner="box",
                       palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.savefig('%s/%s_syn_dst2soma_violin.png' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.boxplot(data=param, palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.savefig('%s/%s_syn_dst2soma_box.png' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.histplot(data=param, palette=ct_palette, legend= True, fill=True, element="step")
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.xlabel(ylabel)
        plt.ylabel('count of cells')
        plt.savefig('%s/%s_syn_dst2soma_dist.png' % (f_name, str_params[i].split()[0]))
        plt.close()

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    log.info('Distance to synapse analysis done')
    time_stamps = time.time()
    step_idents = ['Plotting finished']