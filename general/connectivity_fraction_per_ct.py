#write script that plots for each celltype the ratio in amount and sum of synapse size from
#full cells vs fragments
# fractions of amounts, sum size from different cell classes

#output for each cell class are six different cake plots/ bar plots (4 for axons):
# 1) input from fragments vs full cells/ long axons
# 2) output to fragments vs full cells/long axons
# 3) input: number of synapses from each celltype class
# 4) input: sum size of synapses from each celltype class
# 5) output: number of synapses from each celltype class
#6) output: sum size of synapses from each celltype class
#save this in dictionary, table as well (can plot table as matrix then; one table for input (fragments, full cells), output)

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_number_sum_size_synapses
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ResultsForPlotting
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
    import scipy


    global_params.wd = "ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    max_MSN_path_len = 11000
    syn_prob = 0.8
    min_syn_size = 0.1
    msn_ct = 2
    lman_ct = 3
    gpi_ct = 7
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/general/220927_j0251v4_cts_percentages_mcl_%i_synprob_%.2f" % (
    min_comp_len, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Celltypes input output percentages', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, max_MSN_path_len = %i, syn_prob = %.1f, min_syn_size = %.1f" % (min_comp_len, max_MSN_path_len, syn_prob, min_syn_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #this script should iterate over all celltypes
    axon_cts = [1, 3, 4]
    log.info("Step 1/X: Load cell dicts and get suitable cellids")
    celltypes = [ct_dict[ct] for ct in ct_dict]
    num_cts = len(celltypes)
    full_cell_dicts = {}
    suitable_ids_dict = {}
    all_suitable_ids = []
    cts_numbers_perct = pd.DataFrame(columns=['total number of cells', 'number of suitable cells', 'percentage of suitable cells'], index=celltypes)
    for ct in tqdm(range(num_cts)):
        ct_str = ct_dict[ct]
        if ct in axon_cts:
            cell_dict = load_pkl2obj(
            "/cajal/nvmescratch/users/arother/j0251v4_prep/ax_%.3s_dict.pkl" % ct_str)
            #get ids with min compartment length
            cellids = list(cell_dict.keys())
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len, axon_only=True,
                              max_path_len=None)
            cts_numbers_perct.loc[ct_str, 'total number of cells'] = len(cellids)
        else:
            cell_dict = load_pkl2obj(
                "/cajal/nvmescratch/users/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_str)
            cellids = list(cell_dict.keys())
            if ct == 2:
                cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=max_MSN_path_len)
            else:
                cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
            all_cellids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == ct]
            cts_numbers_perct.loc[ct_str, 'total number of cells'] = len(all_cellids)
            del all_cellids
        full_cell_dicts[ct] = cell_dict
        suitable_ids_dict[ct] = cellids_checked
        cts_numbers_perct.loc[ct_str, 'number of suitable cells'] = len(cellids_checked)
        cts_numbers_perct.loc[ct_str,'percentage of suitable cells'] = cts_numbers_perct.loc[ct_str, 'total number of cells']/\
                                                                   cts_numbers_perct.loc[ct_str, 'number of suitable cells']
        all_suitable_ids.append(cellids_checked)

    all_suitable_ids = np.hstack(np.array(all_suitable_ids))
    write_obj2pkl("%s/suitable_ids_allct.pkl" % f_name, suitable_ids_dict)
    cts_numbers_perct.to_csv("%s/numbers_perct.csv" % f_name)
    time_stamps = [time.time()]
    step_idents = ['Suitable cellids loaded']

    log.info("Step 2/X: Get synapse number and sum of synapse size fractions for each celltype")
    synapse_dict_perct = {i: {} for i in range(num_cts)}
    synapse_pd_perct = cts_numbers_perct = pd.DataFrame(index=celltypes)
    for ct in tqdm(range(num_cts)):
        ct_str = ct_dict[ct]
        #get synapses where ct is involved with syn_prob, min_syn_size
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                            pre_cts=[ct],
                                                                                                            post_cts=None,
                                                                                                            syn_prob_thresh=syn_prob,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True)
        #get rid of all synapses that no suitable cell of ct is involved in
        suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[ct]).reshape(len(m_ssv_partners), 2), axis=1)
        m_cts = m_cts[suit_ct_inds]
        m_ids = m_ids[suit_ct_inds]
        m_ssv_partners = m_ssv_partners[suit_ct_inds]
        m_sizes = m_sizes[suit_ct_inds]
        m_axs = m_axs[suit_ct_inds]
        #seperate in incoming and outgoing synapses
        #incoming synapses
        if ct not in axon_cts:
            #incoming synapses of suitable cells of ct only
            # calculate total number of incoming synapses and sum size (see also similar in analysis_prep_func)
            in_ids, in_sizes, in_ssv_partners, in_axs, in_cts, unique_in_ssvs, in_syn_sizes, in_syn_numbers = get_number_sum_size_synapses(
                syn_ids=m_ids,
                syn_sizes=m_sizes,
                syn_ssv_partners=m_ssv_partners,
                syn_axs=m_axs, syn_cts=m_cts,
                ct=ct, cellids=suitable_ids_dict[ct],
                filter_ax=[0, 2],
                filter_ids=None, return_syn_arrays=True)
            synapse_dict_perct[ct]['in cellids'] = unique_in_ssvs
            synapse_dict_perct[ct]['incoming total synapse number'] = in_syn_numbers
            synapse_dict_perct[ct]['incoming total synapse sum size'] = in_syn_sizes
            synapse_pd_perct.loc[ct_str, 'mean incoming total synapse number'] = np.mean(in_syn_numbers)
            synapse_pd_perct.loc[ct_str, 'mean incoming total synapse sum size'] = np.mean(in_syn_sizes)
            #get only synapses that are with other suitable ids from every cts
            in_ids, in_sizes, in_ssv_partners, in_axs, in_cts, unique_in_ssvs, in_syn_sizes, in_syn_numbers = get_number_sum_size_synapses(
                syn_ids=in_ids,
                syn_sizes=in_sizes,
                syn_ssv_partners=in_ssv_partners,
                syn_axs=in_axs, syn_cts=in_cts,
                ct=ct, cellids=suitable_ids_dict[ct],
                filter_ax=[0,2],
                filter_ids=all_suitable_ids, return_syn_arrays=True)
            synapse_dict_perct[ct]['in full cellids'] = unique_in_ssvs
            synapse_dict_perct[ct]['incoming full cell synapse number'] = in_syn_numbers
            synapse_dict_perct[ct]['incoming full cell synapse sum size'] = in_syn_sizes
            synapse_pd_perct.loc[ct_str, 'mean incoming full cell synapse number'] = np.mean(in_syn_numbers)
            synapse_pd_perct.loc[ct_str, 'mean incoming full cell synapse sum size'] = np.mean(in_syn_sizes)
            #calculate percentage of full cells
            synapse_dict_perct[ct]['incoming percentage full cells synapse number'] = in_syn_numbers / \
                                                                                      synapse_dict_perct[ct]['incoming total synapse number']
            synapse_dict_perct[ct]['incoming percentage full cells synapse sum size'] = in_syn_sizes / \
                                                                                        synapse_dict_perct[ct]['incoming total synapse sum size']
            synapse_pd_perct.loc[ct_str, 'mean incoming percentage full cell synapse number'] = np.mean(
                synapse_pd_perct[ct]['incoming percentage full cells synapse number'])
            synapse_pd_perct.loc[ct_str, 'mean incoming percentage full cell synapse sum size'] = np.mean(
                synapse_pd_perct[ct]['incoming percentage full cells synapse sum size'])
        #outgoing synapses
        # calculate total number of outgoing synapses and sum size (see also similar in analysis_prep_func)
        out_ids, out_sizes, out_ssv_partners, out_axs, out_cts, unique_out_ssvs, out_syn_sizes, out_syn_numbers = get_number_sum_size_synapses(
            syn_ids=m_ids,
            syn_sizes=m_sizes,
            syn_ssv_partners=m_ssv_partners,
            syn_axs=m_axs, syn_cts=m_cts,
            ct=ct, cellids=suitable_ids_dict[ct],
            filter_ax=[1],
            filter_ids=None, return_syn_arrays=True)
        synapse_dict_perct[ct]['out cellids'] = unique_out_ssvs
        synapse_dict_perct[ct]['outgoing total synapse number'] = out_syn_numbers
        synapse_dict_perct[ct]['outgoing total synapse sum size'] = out_syn_sizes
        synapse_pd_perct.loc[ct_str, 'mean outgoing total synapse number'] = np.mean(out_syn_numbers)
        synapse_pd_perct.loc[ct_str, 'mean outgoing total synapse sum size'] = np.mean(out_syn_sizes)
        # get only synapses that are with other suitable ids from every cts
        out_ids, out_sizes, out_ssv_partners, out_axs, out_cts, unique_out_ssvs, out_syn_sizes, out_syn_numbers = get_number_sum_size_synapses(
            syn_ids=out_ids,
            syn_sizes=out_sizes,
            syn_ssv_partners=out_ssv_partners,
            syn_axs=out_axs, syn_cts=out_cts,
            ct=ct, cellids=suitable_ids_dict[ct],
            filter_ax=[1],
            filter_ids=all_suitable_ids, return_syn_arrays=True)
        synapse_dict_perct[ct]['out full cellids'] = unique_out_ssvs
        synapse_dict_perct[ct]['outgoing full cell synapse number'] = out_syn_numbers
        synapse_dict_perct[ct]['outgoing full cell synapse sum size'] = out_syn_sizes
        synapse_pd_perct.loc[ct_str, 'mean outgoing full cell synapse number'] = np.mean(out_syn_numbers)
        synapse_pd_perct.loc[ct_str, 'mean outgoing full cell synapse sum size'] = np.mean(out_syn_sizes)
        # calculate percentage of full cells
        synapse_dict_perct[ct]['outgoing percentage full cells synapse number'] = out_syn_numbers / \
                                                                                  synapse_dict_perct[ct][
                                                                                      'incoming total synapse number']
        synapse_dict_perct[ct]['outgoing percentage full cells synapse sum size'] = out_syn_sizes / \
                                                                                    synapse_dict_perct[ct][
                                                                                        'incoming total synapse sum size']
        synapse_pd_perct.loc[ct_str, 'mean outgoing percentage full cell synapse number'] = np.mean(
            synapse_pd_perct[ct]['outgoing percentage full cells synapse number'])
        synapse_pd_perct.loc[ct_str, 'mean outgoing percentage full cell synapse sum size'] = np.mean(
            synapse_pd_perct[ct]['outgoing percentage full cells synapse sum size'])
        raise ValueError

        #make first plots with fragments vs full cells incoming/ outgoing (pie chart)
        #get sum size and sum of synapses for each celltype seperate (only full cells)

    #Step 3/X plotting
    #Plots:
    #individual celltypes: varying pie charts
    #all celltypes: bar plot with fragments vs full cells (incoming, outgoing), normalised
    #all celltypes: bar plot full cells (incoming, outgoing), normalised
    #matrix, incoming, outgoing, normalised (heatmap?), syn number, sum synapse size











