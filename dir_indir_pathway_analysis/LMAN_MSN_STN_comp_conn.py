#Alexandra Rother
#script to see if LMAN axons going to MSN also synapse to STN
#is there a compartment preference of LMAN -> MSN shaft and STN spine head?

#check which of them also make STN synapses
#how many synapses, percentage of compartment
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import get_compartment_specific_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ResultsForPlotting
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import scipy
    import seaborn as sns

    #global_params.wd = "cajal/nvmescrastch/projects/from_ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    global_params.wd = "ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    analysis_params = Analysis_Params(global_params.wd)
    ct_dict = analysis_params.ct_dict()
    min_comp_len = 200
    handpicked_LMAN = True
    # samples per ct
    syn_prob = 0.8
    min_syn_size = 0.1
    msn_ct = 2
    lman_ct = 3
    stn_ct = 0
    color_key = 'TeBk'
    fontsize = 20
    if handpicked_LMAN:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230319_j0251v4_ct_LMAN_MSN_STN_mcl_%i_k%s_sp_%.1f_ms_%.1f_LMANhp" % (
            min_comp_len, color_key, syn_prob, min_syn_size)
    else:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230319_j0251v4_ct_LMAN_MSN_STN_mcl_%i_k%s_sp_%.1f_ms_%.1f" % (
            min_comp_len, color_key, syn_prob, min_syn_size)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Connectivity from LMAN to MSN and STN',
                             log_dir=f_name + '/logs/')
    log.info(
        f"min_comp_len = %i, key to color to = %s, handpicked LMAN = %s, syn_prob = %.1f, min_syn_size = %.1f µm²" % (
            min_comp_len, color_key, handpicked_LMAN, syn_prob, min_syn_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    cts_for_analysis = [lman_ct, msn_ct, stn_ct]
    cts_str = [ct_dict[i] for i in cts_for_analysis]

    log.info("Step 1/8: load suitable LMAN, MSN and STN, filter for min_comp_lenn")
    known_mergers = analysis_params.load_known_mergers()
    pot_astros = analysis_params.load_potential_astros()
    cache_name = "/cajal/nvmescratch/users/arother/j0251v4_prep"
    cellids_dict = dict()
    for ct in tqdm(cts_for_analysis):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if ct == lman_ct:
            if handpicked_LMAN == True:
                cellids = load_pkl2obj(f"{cache_name}/LMAN_handpicked_arr.pkl")
            else:
                cell_dict = analysis_params.load_cell_dict(ct)
                cellids = np.array(list(cell_dict.keys()))
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=True, max_path_len=None)
            cellids_dict[ct] = cellids
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = load_pkl2obj(
                f"{cache_name}/full_%.3s_arr.pkl" % ct_dict[ct])
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = load_pkl2obj(f'{cache_name}/pot_astro_ids.pkl')
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False, max_path_len=None)
            cellids_dict[ct] = cellids
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    time_stamps = [time.time()]
    step_idents = ["load and filter cells"]

    # get all LMAN ids that synapse to MSN
    # save information on how many synapses, percentage of going to shaft
    #similar to function from LMAN_number_per_MSN

    compartments = ['soma', 'spine neck', 'spine head', 'dendritic shaft']
    num_comps = len(compartments)

    log.info("Step 2/8: Get comp information from each LMAN to MSN")
    #make per LMAN dictionary
    #parameters: number of synapses per MSN, sum of syn size, number of syns per comp, sum size per comp,
    #percentage of sum size per comp


    perlman_params, syn_params, all_syns_msn = get_compartment_specific_connectivity(ct_post=msn_ct,
                                                                       cellids_post=cellids_dict[msn_ct],
                                                                       sd_synssv=sd_synssv,
                                                                       syn_prob=syn_prob,
                                                                       min_syn_size=min_syn_size,
                                                                       ct_pre=lman_ct, cellids_pre=cellids_dict[lman_ct],
                                                                       sort_per_postsyn_ct = False)

    # syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
    lman_ids2msn = perlman_params[-1]
    num_lman2msn_ids = len(lman_ids2msn)
    num_lman_cellids = len(cellids_dict[lman_ct])
    log.info(f'{num_lman2msn_ids} of {num_lman_cellids} LMAN cells '
             f'({100 * num_lman2msn_ids / num_lman_cellids}%) make synapses to MSN cells.')
    log.info(f'{len(all_syns_msn)} synapses are made in total to MSN cells with a median size of {np.median(all_syns_msn["synapse size"]):.2f} µm².')
    all_syns_msn['postsynaptic ct'] = ct_dict[msn_ct]

    percell_params_str = ['number of synapses', 'summed synapse size [µm²]', 'percentage of synapses',
               'percentage of synapse sizes', 'cellid']
    columns = np.hstack([percell_params_str, 'compartment of postynaptic cells', 'postsynaptic ct'])
    lman_msn_df = pd.DataFrame(columns=columns, index=range(num_lman2msn_ids*num_comps))

    for i_comp, compartment in enumerate(compartments):
        # fill in numbers that summarise all synapses per compartment per celltype
        len_comp_params = len(perlman_params[0][compartment])
        lman_msn_df.loc[i_comp * num_lman2msn_ids: i_comp * num_lman2msn_ids + len_comp_params - 1, 'compartment of postynaptic cells'] = compartment
        lman_msn_df.loc[i_comp * num_lman2msn_ids: i_comp * num_lman2msn_ids + len_comp_params - 1,
        'postsynaptic ct'] = ct_dict[msn_ct]
        for iy in range(len(percell_params_str)):
            if iy == 4:
                lman_msn_df.loc[i_comp * num_lman2msn_ids: i_comp * num_lman2msn_ids + len_comp_params - 1,
                percell_params_str[iy]] = perlman_params[iy]
            else:
                lman_msn_df.loc[i_comp * num_lman2msn_ids: i_comp * num_lman2msn_ids + len_comp_params - 1, percell_params_str[iy]] = perlman_params[iy][compartment]

    lman_msn_df['median synapse size [µm²]'] = 0.0
    nonzero_syn_inds = lman_msn_df['number of synapses'] > 0
    mean_syn_size = lman_msn_df['summed synapse size [µm²]'][nonzero_syn_inds]/lman_msn_df['number of synapses'][nonzero_syn_inds]
    lman_msn_df['mean synapse size [µm²]'] = mean_syn_size

    lman_msn_df.to_csv(f'{f_name}/lman_msn_comp_dict.csv')
    median_perc_shaft_syns = np.median(lman_msn_df['percentage of synapse sizes'][lman_msn_df['compartment of postynaptic cells'] == 'dendritic shaft'])
    median_perc_sh_syns = np.median(
        lman_msn_df['percentage of synapse sizes'][lman_msn_df['compartment of postynaptic cells'] == 'spine head'])
    log.info(f'Median percentage of shaft synapse sizes to MSN: {median_perc_shaft_syns}')
    log.info(f'Median percentage of spine head synapse sizes to MSM: {median_perc_sh_syns}')

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to MSN"]

    log.info("Step 3/8: Get comp information from each LMAN to STN")
    # make per LMAN dictionary
    # parameters: number of synapses per MSN, sum of syn size, number of syns per comp, sum size per comp,
    # percentage of sum size per comp

    perlman_params, syn_params, all_syns_stn = get_compartment_specific_connectivity(ct_post=stn_ct,
                                                                       cellids_post=cellids_dict[stn_ct],
                                                                       sd_synssv=sd_synssv,
                                                                       syn_prob=syn_prob,
                                                                       min_syn_size=min_syn_size,
                                                                       ct_pre=lman_ct,
                                                                       cellids_pre=cellids_dict[lman_ct],
                                                                       sort_per_postsyn_ct=False)

    # syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
    lman_ids2stn = perlman_params[-1]
    num_lman2stn_ids = len(lman_ids2stn)
    log.info(f'{num_lman2stn_ids} of {num_lman_cellids} LMAN cells '
             f'({100 * num_lman2stn_ids / num_lman_cellids:.2f}%) make synapses to STN cells.')
    all_syns_stn['postsynaptic ct'] = ct_dict[stn_ct]

    lman_stn_df = pd.DataFrame(columns=columns, index=range(num_lman2stn_ids * num_comps))
    for i_comp, compartment in enumerate(compartments):
        # fill in numbers that summarise all synapses per compartment per celltype
        len_comp_params = len(perlman_params[0][compartment])
        lman_stn_df.loc[i_comp * num_lman2stn_ids: i_comp * num_lman2stn_ids + len_comp_params - 1,
        'compartment of postynaptic cells'] = compartment
        lman_stn_df.loc[i_comp * num_lman2stn_ids: i_comp * num_lman2stn_ids + len_comp_params - 1,
        'postsynaptic ct'] = ct_dict[stn_ct]
        for iy in range(len(percell_params_str)):
            if iy == 4:
                lman_stn_df.loc[i_comp * num_lman2stn_ids: i_comp * num_lman2stn_ids + len_comp_params - 1,
                percell_params_str[iy]] = perlman_params[iy]
            else:
                lman_stn_df.loc[i_comp * num_lman2stn_ids: i_comp * num_lman2stn_ids + len_comp_params - 1,
                percell_params_str[iy]] = perlman_params[iy][compartment]

    lman_stn_df['median synapse size [µm²]'] = 0.0
    nonzero_syn_inds = lman_stn_df['number of synapses'] > 0
    mean_syn_size = lman_stn_df['summed synapse size [µm²]'][nonzero_syn_inds] / lman_stn_df['number of synapses'][
        nonzero_syn_inds]
    lman_stn_df['mean synapse size [µm²]'] = mean_syn_size

    lman_stn_df.to_csv(f'{f_name}/lman_stn_comp_dict.csv')
    median_perc_shaft_syns = np.median(
        lman_stn_df['percentage of synapse sizes'][lman_stn_df['compartment of postynaptic cells'] == 'dendritic shaft'])
    median_perc_sh_syns = np.median(
        lman_stn_df['percentage of synapse sizes'][lman_stn_df['compartment of postynaptic cells'] == 'spine head'])
    log.info(f'Median percentage of shaft synapse sizes to STN: {median_perc_shaft_syns:.2f}')
    log.info(f'Median percentage of spine head synapse sizes to STN: {median_perc_sh_syns:.2f}')

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to STN"]

    log.info("Step 4/8: Get information on correlation between LMAN connections to MSN and STN")
    #plot information on percentage of sum size of synapses of different compartments
    comp_cls = CompColors()
    comp_palette = comp_cls.comp_palette(color_key, num=False, denso=True)
    lman_df = pd.concat([lman_msn_df, lman_stn_df])
    for param in lman_df.columns:
        if 'cellid' in param or 'postsynaptic' in param:
            continue
        sns.barplot(data = lman_df, x = 'postsynaptic ct', y = param, hue='compartment of postynaptic cells', palette=comp_palette)
        plt.title(param)
        plt.savefig(f'{f_name}/comp_{param}_barplot.png')
        plt.close()
        sns.stripplot(data=lman_df, x='postsynaptic ct', y=param, hue='compartment of postynaptic cells', color="black", alpha=0.2,
                              dodge=True, size=2)
        sns.violinplot(data=pd.DataFrame(lman_df.to_dict()), x='postsynaptic ct', y=param, hue='compartment of postynaptic cells',
                    palette=comp_palette)
        plt.title(param)
        plt.savefig(f'{f_name}/comp_{param}_violinplot.png')
        plt.close()
        sns.boxplot(data=lman_df, x='postsynaptic ct', y=param, hue='compartment of postynaptic cells',
                    palette=comp_palette)
        plt.title(param)
        plt.savefig(f'{f_name}/comp_{param}_boxplot.png')
        plt.close()

    #make plot for all synapses about synapse size
    all_syns_df = pd.concat([all_syns_msn, all_syns_stn])
    sns.barplot(data=all_syns_df, x='postsynaptic ct', y='synapse size', hue='compartment',
                palette=comp_palette)
    plt.title('Synapse size from LMAN to different compartments')
    plt.ylabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/all_syns_synsize_barplot.png')
    plt.close()
    sns.stripplot(data=all_syns_df, x='postsynaptic ct', y='synapse size', hue='compartment of postynaptic cells', color="black",
                  alpha=0.2,
                  dodge=True, size=2)
    sns.violinplot(data=all_syns_df, x='postsynaptic ct', y='synapse size',
                   hue='compartment of postynaptic cells',
                   palette=comp_palette)
    plt.title('Synapse size from LMAN to different compartments')
    plt.ylabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/all_syns_synsize_violinplot.png')
    plt.close()
    sns.boxplot(data=all_syns_df, x='postsynaptic ct', y='synapse size', hue='compartment of postynaptic cells',
                palette=comp_palette)
    plt.title('Synapse size from LMAN to different compartments')
    plt.ylabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/all_syns_synsize_boxplot.png')
    plt.close()

    # create pd DataFrame for information on how many LMAN axons make which connections
    ids2msn_stn = lman_ids2msn[np.in1d(lman_ids2msn, lman_ids2stn)]
    log.info(f'{len(ids2msn_stn)} of {num_lman_cellids} LMAN cells '
             f'({100 * len(ids2msn_stn) / num_lman_cellids:.2f}%) make synapses to MSN and STN cells. \n'
             f'{100 * len(ids2msn_stn) / num_lman2msn_ids:.2f} % of cells from LMAN to MSN and '
             f'{100 * len(ids2msn_stn) / num_lman2stn_ids:.2f} % of cells from LMAN to STN.')
    ids_perc_columns = ['number', '% of total LMAN', '% of LMAN to MSN', '% of LMAN to STN', '% of LMAN to MSN and STN']
    lman_ids_perc_df = pd.DataFrame(columns=ids_perc_columns)
    lman_ids_msn_or_stn = np.unique(np.hstack([lman_ids2msn, lman_ids2stn]))
    lman_ids_perc_df.loc['LMAN to either MSN or STN', 'number'] = len(lman_ids_msn_or_stn)
    lman_ids_perc_df.loc['LMAN to either MSN or STN', '% of total LMAN'] = 100 * len(
        lman_ids_msn_or_stn) / num_lman_cellids
    lman_ids_perc_df.loc['LMAN ids to MSN and STN', 'number'] = len(ids2msn_stn)
    lman_ids_perc_df.loc['LMAN ids to MSN and STN', '% of total LMAN'] = 100 * len(ids2msn_stn) / num_lman_cellids
    lman_ids_perc_df.loc['LMAN ids to MSN and STN', '% of LMAN to MSN'] = 100 * len(ids2msn_stn) / num_lman2msn_ids
    lman_ids_perc_df.loc['LMAN ids to MSN and STN', '% of LMAN to STN'] = 100 * len(ids2msn_stn) / num_lman2stn_ids
    lman_ids_perc_df.loc['LMAN ids to MSN and STN', '% of LMAN to MSN and STN'] = 100 * len(ids2msn_stn) / len(
        ids2msn_stn)
    #get number of how many cells go to MSN and STN shaft and spine head (same and different)
    lman_ids_dict = {}
    lman_msn_shaft = lman_msn_df[lman_msn_df['compartment of postynaptic cells'] == 'dendritic shaft']
    lman_msn_shaft_ids = lman_msn_shaft['cellid'][lman_msn_shaft['number of synapses'] > 0]
    lman_ids_dict['LMAN ids to MSN shaft'] = lman_msn_shaft_ids
    lman_msn_shaft_ids_50 = lman_msn_shaft['cellid'][lman_msn_shaft['percentage of synapse sizes'] > 50]
    lman_ids_dict['LMAN ids to MSN shaft (> 50% sum size)'] = lman_msn_shaft_ids_50
    lman_msn_head = lman_msn_df[lman_msn_df['compartment of postynaptic cells'] == 'spine head']
    lman_msn_head_ids = lman_msn_head['cellid'][lman_msn_head['number of synapses'] > 0]
    lman_ids_dict['LMAN ids to MSN head'] = lman_msn_head_ids
    lman_msn_head_ids_50 = lman_msn_head['cellid'][lman_msn_head['percentage of synapse sizes'] > 50]
    lman_ids_dict['LMAN ids to MSN head (> 50% sum size)'] = lman_msn_head_ids_50
    lman_stn_shaft = lman_stn_df[lman_stn_df['compartment of postynaptic cells'] == 'dendritic shaft']
    lman_stn_shaft_ids = lman_stn_shaft['cellid'][lman_stn_shaft['number of synapses'] > 0]
    lman_ids_dict['LMAN ids to STN shaft'] = lman_stn_shaft_ids
    lman_stn_shaft_ids_50 = lman_stn_shaft['cellid'][lman_stn_shaft['percentage of synapse sizes'] > 50]
    lman_ids_dict['LMAN ids to STN shaft (> 50% sum size)'] = lman_stn_shaft_ids_50
    lman_stn_head = lman_stn_df[lman_stn_df['compartment of postynaptic cells'] == 'spine head']
    lman_stn_head_ids = lman_stn_head['cellid'][lman_stn_head['number of synapses'] > 0]
    lman_ids_dict['LMAN ids to STN head'] = lman_stn_head_ids
    lman_stn_head_ids_50 = lman_stn_head['cellid'][lman_stn_head['percentage of synapse sizes'] > 50]
    lman_ids_dict['LMAN ids to STN head (> 50% sum size)'] = lman_stn_head_ids_50
    msn_stn_shaft_ids = lman_msn_shaft_ids[np.in1d(lman_msn_shaft_ids, lman_stn_shaft_ids)]
    msn_stn_head_ids = lman_msn_shaft_ids[np.in1d(lman_msn_head_ids, lman_stn_head_ids)]
    msnshaft_stnhead_ids = lman_msn_shaft_ids[np.in1d(lman_msn_shaft_ids, lman_stn_head_ids)]
    msnhead_stnshaft_ids = lman_msn_head_ids[np.in1d(lman_msn_head_ids, lman_stn_shaft_ids)]
    lman_ids_dict['LMAN ids to MSN and STN shaft'] = msn_stn_shaft_ids
    lman_ids_dict['LMAN ids to MSN and STN head'] = msn_stn_head_ids
    lman_ids_dict['LMAN ids to MSN shaft and STN head'] = msnshaft_stnhead_ids
    lman_ids_dict['LMAN ids to MSN head and STN shaft'] = msnhead_stnshaft_ids
    msn_stn_shaft_ids50 = lman_msn_shaft_ids_50[np.in1d(lman_msn_shaft_ids_50, lman_stn_shaft_ids_50)]
    msn_stn_head_ids50 = lman_msn_head_ids_50[np.in1d(lman_msn_head_ids_50, lman_stn_head_ids_50)]
    msnshaft_stnhead_ids50 = lman_msn_shaft_ids_50[np.in1d(lman_msn_shaft_ids_50, lman_stn_head_ids_50)]
    msnhead_stnshaft_ids50 = lman_msn_head_ids_50[np.in1d(lman_msn_head_ids_50, lman_stn_shaft_ids_50)]
    lman_ids_dict['LMAN ids to MSN and STN shaft (> 50% sum size)'] = msn_stn_shaft_ids50
    lman_ids_dict['LMAN ids to MSN and STN head (> 50% sum size)'] = msn_stn_head_ids50
    lman_ids_dict['LMAN ids to MSN shaft and STN head (> 50% sum size)'] = msnshaft_stnhead_ids50
    lman_ids_dict['LMAN ids to MSN head and STN shaft (> 50% sum size)'] = msnhead_stnshaft_ids50
    for key in list(lman_ids_dict.keys()):
        num_cells = len(lman_ids_dict[key])
        lman_ids_perc_df.loc[key, 'number'] = num_cells
        lman_ids_perc_df.loc[key, '% of total LMAN'] = 100 * num_cells / num_lman_cellids
        lman_ids_perc_df.loc[key, '% of LMAN to MSN'] = 100 * num_cells / num_lman2msn_ids
        lman_ids_perc_df.loc[key, '% of LMAN to STN'] = 100 * num_cells / num_lman2stn_ids
        lman_ids_perc_df.loc[key, '% of LMAN to MSN and STN'] = 100 * num_cells / len(
            ids2msn_stn)

    #make scatter plot to visualize dependency between MSN and STN synapse size per cell
    #plot average synapse size to MSN and to STN for all cells that synapse to both independent of compartment
    scatter_cols = ['cellid', 'number of synapses to MSN', 'number of synapses to STN', 'summed synapse size [µm2] to MSN', 'summed synapse size [µm2] to STN']
    forscatter_df = pd.DataFrame(columns = scatter_cols, index = range(len(ids2msn_stn)))
    forscatter_df['cellid'] = ids2msn_stn
    for i, cellid in enumerate(ids2msn_stn):
        cell_df = lman_df[lman_df['cellid'] == cellid]
        forscatter_df.loc[i, 'number of synapses to MSN'] = cell_df['number of synapses'][cell_df['postsynaptic ct'] == 'MSN'].sum()
        forscatter_df.loc[i, 'number of synapses to STN'] = cell_df['number of synapses'][
            cell_df['postsynaptic ct'] == 'STN'].sum()
        forscatter_df.loc[i, 'summed synapse size [µm2] to MSN'] = cell_df['summed synapse size [µm2]'][
            cell_df['postsynaptic ct'] == 'MSN'].sum()
        forscatter_df.loc[i, 'summed synapse size [µm2] to STN'] = cell_df['summed synapse size [µm2]'][
            cell_df['postsynaptic ct'] == 'STN'].sum()
    forscatter_df['mean synapse size [µm²] to MSN'] = forscatter_df['summed synapse size [µm2] to MSN'] / forscatter_df['number of synapses to MSN']
    forscatter_df['mean synapse size [µm²] to STN'] = forscatter_df['summed synapse size [µm2] to STN'] / forscatter_df[
        'number of synapses to STN']
    lman2msn_stn_df = lman_df[np.in1d(lman_df['cellid'], ids2msn_stn)]
    msn_data = lman2msn_stn_df[lman2msn_stn_df['postsynaptic ct'] == 'MSN']
    stn_data = lman2msn_stn_df[lman2msn_stn_df['postsynaptic ct'] == 'STN']
    for col in lman_df.columns:
        if 'cellid' in col or 'postsynaptic' in col or 'percentage' in col:
            continue
        sns.scatter(data = forscatter_df, x=f'{col} to MSN', y = f'{col} to STN')
        plt.xlabel(f'{col} to MSN')
        plt.ylabel(f'{col} to STN')
        plt.title(col)
        plt.savefig(f'{f_name}/percell_i2msn_stn_{col}_scatter.png')
        plt.close()
        #plot scatter plot also dependent on compartment, MSN-STN shaft-shaft, head-head, shaft-head, head-shaft
        sns.scatterplot(x = msn_data[col][msn_data['compartment of postsynaptic cells'] == 'dendritic shaft'],
                        y = stn_data[col][msn_data['compartment of postsynaptic cells'] == 'dendritic shaft'])
        plt.xlabel(f'{col} to MSN dendritic shaft')
        plt.ylabel(f'{col} to STN dendritic shaft')
        plt.title(col)
        plt.savefig(f'{f_name}/percell_i2msn_stn_{col}_msnshaft_stnshaft_scatter.png')
        plt.close()
        sns.scatterplot(x=msn_data[col][msn_data['compartment of postsynaptic cells'] == 'spine head'],
                        y=stn_data[col][msn_data['compartment of postsynaptic cells'] == 'spine head'])
        plt.xlabel(f'{col} to MSN spine head')
        plt.ylabel(f'{col} to STN spine head')
        plt.title(col)
        plt.savefig(f'{f_name}/percell_i2msn_stn_{col}_msnhead_stnhead_scatter.png')
        plt.close()
        sns.scatterplot(x=msn_data[col][msn_data['compartment of postsynaptic cells'] == 'dendritic shaft'],
                        y=stn_data[col][msn_data['compartment of postsynaptic cells'] == 'spine head'])
        plt.xlabel(f'{col} to MSN dendritic shaft')
        plt.ylabel(f'{col} to STN spine head')
        plt.title(col)
        plt.savefig(f'{f_name}/percell_i2msn_stn_{col}_msnshaft_stnhead_scatter.png')
        plt.close()
        sns.scatterplot(x=msn_data[col][msn_data['compartment of postsynaptic cells'] == 'spine head'],
                        y=stn_data[col][msn_data['compartment of postsynaptic cells'] == 'dendritic shaft'])
        plt.xlabel(f'{col} to MSN spine head')
        plt.ylabel(f'{col} to STN dendritic shaft')
        plt.title(col)
        plt.savefig(f'{f_name}/percell_i2msn_stn_{col}_msnshaft_stnshaft_scatter.png')
        plt.close()





    write_obj2pkl(f'{f_name}/lman_ids_dict.pkl', lman_ids_dict)
    lman_ids_perc_df.to_csv(f'{f_name}/lman_msn_stn_perc.csv')
    forscatter_df.to_csv(f'{f_name}/lman2msn_stn_scatter_total.csv')

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to MSN"]

    log.info('Analysis for LMAN connectivity to MSN and STN done')


