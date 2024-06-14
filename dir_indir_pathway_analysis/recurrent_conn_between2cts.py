#get cell-specific connectivity of recurrently connecte cellypes
#get cellids each cell get input from and projects to
#also other way around
#see if input and output of each cell matches
#calculate percentage of overlap based on cellid number
#calculate percentage of overlap based on synapse size

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, filter_synapse_caches_for_ct,get_ct_syn_number_sumsize, get_percell_number_sumsize
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from tqdm import tqdm
    from scipy.stats import ranksums, kruskal
    import seaborn as sns
    import matplotlib.pyplot as plt

    version = 'v6'
    bio_params = Analysis_Params(version = version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    global_params.wd = bio_params.working_dir()
    #min_comp_len = bio_params.min_comp_length()
    min_comp_len_cell = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGPINTv6'
    ct1 = 4
    ct2 = 6
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240614_j0251{version}_{ct1_str}_{ct2_str}_recurr_conn_mcl_%i_synprob_%.2f_%s_fs%i" % (
    min_comp_len_cell, syn_prob, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('recurr_conn_log', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s" % (
        min_comp_len_cell, syn_prob, min_syn_size, exclude_known_mergers))
    log.info(f'Cell-specific connectivity between {ct1_str} and {ct2_str} will be analysed')
    log.info('Goal is to see if input comes from same cells, output goes to')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)

    log.info('Step 1/3: Get suitable cellids')
    known_mergers = bio_params.load_known_mergers()
    misclassified_asto_ids = bio_params.load_potential_astros()
    cts = [ct1, ct2]
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in cts:
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    all_suitable_ids = np.concatenate(all_suitable_ids)

    log.info('Step 2/3: Filter synapses for celltypes')
    # prefilter synapses for synapse prob thresh and min syn size
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #make sure synapses only between suitable ids
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    syn_prob = syn_prob[suit_ct_inds]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    #get synapses from ct1 -> ct2
    log.info(f'Step 3/X: Get number and sumsize per cell for {ct1_str} to {ct2_str} synapses')
    ct1_2_ct2_cts, ct1_2_ct2_ids, ct1_2_ct2_axs, ct1_2_ct2_ssv_partners, ct1_2_ct2_sizes, ct1_2_ct2_spiness, ct1_2_ct2_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct1],
                                                                                                        post_cts=[ct2],
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)
    log.info(f'Total synaptic strength from {ct1_str} to {ct2_str} are {np.sum(ct1_2_ct2_sizes):.2f} µm² from {len(ct1_2_ct2_sizes)} synapses')
    # get ct1 ids that project to ct2
    ct1_out_syn_numbers, ct1_out_syn_ssv_sizes, ct1_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=ct1_2_ct2_sizes,
                                                                                  syn_ssv_partners=ct1_2_ct2_ssv_partners,
                                                                                  syn_cts=ct1_2_ct2_cts, ct=ct1)
    log.info(
        f'{len(ct1_proj_ssvs)} {ct1_str} project to {ct2_str}. These are {100 * len(ct1_proj_ssvs) / len(suitable_ids_dict[ct1]):.2f}'
        f' percent of {ct1_str} cells')
    log.info(
        f'The median number of synapses are {np.median(ct1_out_syn_numbers)}, sum size {np.median(ct1_out_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct1_str}_proj_{ct2_str}_ids.pkl', ct1_proj_ssvs)
    #iterate over each ct1 cell and get number of ct2 cells it connects to with the corresponding summed synapse size
    ct1_out_ct2_ids_dict = {}
    ct1_out_sumsizes_dict = {}
    ct1_out_syn_numbers_dict = {}
    ct1_2_ct2_cell_number = np.zeros(len(ct1_proj_ssvs))
    for ii, ct1_cellid in enumerate(ct1_proj_ssvs):
        #get only synapses for this cellid
        ind = np.where(ct1_2_ct2_ssv_partners == ct1_cellid)[0]
        id_partners = ct1_2_ct2_ssv_partners[ind]
        id_sizes = ct1_2_ct2_sizes[ind]
        ind = np.where(id_partners != ct1_cellid)
        ct2_partners = id_partners[ind]
        idct2_numbers, idct2_sizes, id_ct2_partners = get_percell_number_sumsize(ct2_partners, id_sizes)
        ct1_2_ct2_cell_number[ii] = len(id_ct2_partners)
        ct1_out_ct2_ids_dict[ct1_cellid] = id_ct2_partners
        ct1_out_syn_numbers_dict[ct1_cellid] = idct2_numbers
        ct1_out_sumsizes_dict[ct1_cellid] = idct2_sizes
    log.info(
        f'A median {ct1_str} projects to {np.median(ct1_2_ct2_cell_number)} {ct2_str} cells, with {np.median(ct1_out_syn_numbers / ct1_2_ct2_cell_number):.2f}'
        f' synapses and {np.median(ct1_out_syn_ssv_sizes / ct1_2_ct2_cell_number):.2f} synaptic area in µm²')
    write_obj2pkl(f'{f_name}/{ct1_str}_out_{ct2_str}_ids.pkl', ct1_out_ct2_ids_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_out_{ct2_str}_syn_numbers.pkl', ct1_out_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_out_{ct2_str}_syn_sumsizes.pkl', ct1_out_sumsizes_dict)
    #plot numbers of ct2 cells ct1 connect to
    sns.histplot(data=ct1_2_ct2_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct2_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} to {ct2_str} connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_out_{ct2_str}_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{{ct1_str}_out_{ct2_str}_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=ct1_2_ct2_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct2_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} to {ct2_str} connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_out_{ct2_str}_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct1_str}_out_{ct2_str}_num_partners_hist.svg')
    plt.close()
    #get ct2_ids that receive input from ct1
    ct2_in_syn_numbers, ct2_in_syn_ssv_sizes, ct2_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=ct1_2_ct2_sizes,
                                                                                          syn_ssv_partners=ct1_2_ct2_ssv_partners,
                                                                                          syn_cts=ct1_2_ct2_cts, ct=ct2)
    log.info(
        f'{len(ct2_rec_ssvs)} {ct2_str} get input from {ct1_str}. These are {100 * len(ct2_rec_ssvs) / len(suitable_ids_dict[ct2]):.2f}'
        f' percent of {ct2_str} cells')
    log.info(
        f'The median number of synapses are {np.median(ct2_in_syn_numbers)}, sum size {np.median(ct2_in_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct2_str}_rec_{ct1_str}_ids.pkl', ct2_rec_ssvs)
    #iterate over each ct2 id and make dictionary which cells it receives synapses from
    ct2_in_ct1_ids_dict = {}
    ct2_in_sumsizes_dict = {}
    ct2_in_syn_numbers_dict = {}
    ct2_from_ct1_cell_number = np.zeros(len(ct2_rec_ssvs))
    for ii, ct2_cellid in enumerate(ct2_rec_ssvs):
        # get only synapses for this cellid
        ind = np.where(ct1_2_ct2_ssv_partners == ct2_cellid)[0]
        id_partners = ct1_2_ct2_ssv_partners[ind]
        id_sizes = ct1_2_ct2_sizes[ind]
        ind = np.where(id_partners != ct2_cellid)
        ct1_partners = id_partners[ind]
        idct1_numbers, idct1_sizes, id_ct1_partners = get_percell_number_sumsize(ct1_partners, id_sizes)
        ct2_from_ct1_cell_number[ii] = len(id_ct1_partners)
        ct2_in_ct1_ids_dict[ct2_cellid] = id_ct1_partners
        ct2_in_syn_numbers_dict[ct2_cellid] = idct1_numbers
        ct2_in_sumsizes_dict[ct2_cellid] = idct1_sizes
    log.info(
        f'A median {ct2_str} receives synapses from {np.median(ct2_from_ct1_cell_number)} {ct1_str} cells, with {np.median(ct2_in_syn_numbers / ct2_from_ct1_cell_number):.2f}'
        f' synapses and {np.median(ct2_in_syn_ssv_sizes / ct2_from_ct1_cell_number):.2f} synaptic area in µm²')
    write_obj2pkl(f'{f_name}/{ct2_str}_in_{ct1_str}_ids.pkl', ct2_in_ct1_ids_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_in_{ct1_str}_syn_numbers.pkl', ct2_in_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_in_{ct1_str}_syn_sumsizes.pkl', ct2_in_sumsizes_dict)
    # plot numbers of ct1 cells ct2 gets input from
    sns.histplot(data=ct2_from_ct1_cell_number, color=ct_palette[ct2_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct2_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct1_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} to {ct2_str} connectivity')
    plt.savefig(f'{f_name}/{ct2_str}_in_{ct1_str}_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{ct2_str}_in_{ct1_str}_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=ct2_from_ct1_cell_number, color=ct_palette[ct2_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct2_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct1_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} to {ct2_str} connectivity')
    plt.savefig(f'{f_name}/{ct2_str}_in_{ct1_str}_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct2_str}_in_{ct1_str}_num_partners_hist.svg')
    plt.close()
    #get synapses from ct2 -> ct1

    log.info(f'Step 4/X: Get number and sumsize per cell for {ct2_str} to {ct1_str} synapses')
    ct2_2_ct1_cts, ct2_2_ct1_ids, ct2_2_ct1_axs, ct2_2_ct1_ssv_partners, ct2_2_ct1_sizes, ct2_2_ct1_spiness, ct2_2_ct1_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[ct2],
        post_cts=[ct1],
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)
    log.info(
        f'Total synaptic strength from {ct2_str} to {ct1_str} are {np.sum(ct2_2_ct1_sizes):.2f} µm² from {len(ct2_2_ct1_sizes)} synapses')
    # get ct2 ids that project to ct1
    ct2_out_syn_numbers, ct2_out_syn_ssv_sizes, ct2_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=ct2_2_ct1_sizes,
                                                                                          syn_ssv_partners=ct2_2_ct1_ssv_partners,
                                                                                          syn_cts=ct2_2_ct1_cts, ct=ct2)
    log.info(
        f'{len(ct2_proj_ssvs)} {ct2_str} project to {ct1_str}. These are {100 * len(ct2_proj_ssvs) / len(suitable_ids_dict[ct2]):.2f}'
        f' percent of {ct2_str} cells')
    log.info(
        f'The median number of synapses are {np.median(ct2_out_syn_numbers)}, sum size {np.median(ct2_out_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct2_str}_proj_{ct1_str}_ids.pkl', ct2_proj_ssvs)
    # iterate over each ct2 cell and get number of ct1 cells it connects to with the corresponding summed synapse size
    ct2_out_ct1_ids_dict = {}
    ct2_out_sumsizes_dict = {}
    ct2_out_syn_numbers_dict = {}
    ct2_2_ct1_cell_number = np.zeros(len(ct2_proj_ssvs))
    for ii, ct2_cellid in enumerate(ct2_proj_ssvs):
        # get only synapses for this cellid
        ind = np.where(ct2_2_ct1_ssv_partners == ct2_cellid)[0]
        id_partners = ct2_2_ct1_ssv_partners[ind]
        id_sizes = ct2_2_ct1_sizes[ind]
        ind = np.where(id_partners != ct2_cellid)
        ct1_partners = id_partners[ind]
        idct1_numbers, idct1_sizes, id_ct1_partners = get_percell_number_sumsize(ct1_partners, id_sizes)
        ct2_2_ct1_cell_number[ii] = len(id_ct1_partners)
        ct2_out_ct1_ids_dict[ct2_cellid] = id_ct1_partners
        ct2_out_syn_numbers_dict[ct2_cellid] = idct1_numbers
        ct2_out_sumsizes_dict[ct2_cellid] = idct1_sizes
    log.info(
        f'A median {ct2_str} projects to {np.median(ct2_2_ct1_cell_number)} {ct1_str} cells, with {np.median(ct2_out_syn_numbers / ct2_2_ct1_cell_number):.2f}'
        f' synapses and {np.median(ct2_out_syn_ssv_sizes / ct2_2_ct1_cell_number):.2f} synaptic area in µm²')
    write_obj2pkl(f'{f_name}/{ct2_str}_out_{ct1_str}_ids.pkl', ct2_out_ct1_ids_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_out_{ct1_str}_syn_numbers.pkl', ct2_out_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_out_{ct1_str}_syn_sumsizes.pkl', ct2_out_sumsizes_dict)
    # plot numbers of ct1 cells ct2 connect to
    sns.histplot(data=ct2_2_ct1_cell_number, color=ct_palette[ct2_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct2_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct1_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct2_str} to {ct1_str} connectivity')
    plt.savefig(f'{f_name}/{ct2_str}_out_{ct1_str}_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{ct2_str}_out_{ct1_str}_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=ct2_2_ct1_cell_number, color=ct_palette[ct2_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct2_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct1_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct2_str} to {ct1_str} connectivity')
    plt.savefig(f'{f_name}/{ct2_str}_out_{ct1_str}_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct2_str}_out_{ct1_str}_num_partners_hist.svg')
    plt.close()
    # get ct1_ids that receive input from ct2
    ct1_in_syn_numbers, ct1_in_syn_ssv_sizes, ct1_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=ct2_2_ct1_sizes,
                                                                                       syn_ssv_partners=ct2_2_ct1_ssv_partners,
                                                                                       syn_cts=ct2_2_ct1_cts, ct=ct1)
    log.info(
        f'{len(ct1_rec_ssvs)} {ct1_str} get input from {ct2_str}. These are {100 * len(ct1_rec_ssvs) / len(suitable_ids_dict[ct1]):.2f}'
        f' percent of {ct1_str} cells')
    log.info(
        f'The median number of synapses are {np.median(ct1_in_syn_numbers)}, sum size {np.median(ct1_in_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct1_str}_rec_{ct2_str}_ids.pkl', ct1_rec_ssvs)
    # iterate over each ct1 id and make dictionary which cells it receives synapses from
    ct1_in_ct2_ids_dict = {}
    ct1_in_sumsizes_dict = {}
    ct1_in_syn_numbers_dict = {}
    ct1_from_ct2_cell_number = np.zeros(len(ct1_rec_ssvs))
    for ii, ct1_cellid in enumerate(ct1_rec_ssvs):
        # get only synapses for this cellid
        ind = np.where(ct2_2_ct1_ssv_partners == ct1_cellid)[0]
        id_partners = ct2_2_ct1_ssv_partners[ind]
        id_sizes = ct2_2_ct1_sizes[ind]
        ind = np.where(id_partners != ct1_cellid)
        ct2_partners = id_partners[ind]
        idct2_numbers, idct2_sizes, id_ct2_partners = get_percell_number_sumsize(ct2_partners, id_sizes)
        ct1_from_ct2_cell_number[ii] = len(id_ct2_partners)
        ct1_in_ct2_ids_dict[ct1_cellid] = id_ct2_partners
        ct1_in_syn_numbers_dict[ct1_cellid] = idct2_numbers
        ct1_in_sumsizes_dict[ct1_cellid] = idct2_sizes
    log.info(
        f'A median {ct1_str} receives synapses from {np.median(ct1_from_ct2_cell_number)} {ct2_str} cells, with {np.median(ct1_in_syn_numbers / ct1_from_ct2_cell_number):.2f}'
        f' synapses and {np.median(ct1_in_syn_ssv_sizes / ct1_from_ct2_cell_number):.2f} synaptic area in µm²')
    write_obj2pkl(f'{f_name}/{ct1_str}_in_{ct2_str}_ids.pkl', ct1_in_ct2_ids_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_in_{ct2_str}_syn_numbers.pkl', ct1_in_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_in_{ct2_str}_syn_sumsizes.pkl', ct1_in_sumsizes_dict)
    # plot numbers of ct2 cells ct1 receives input from
    sns.histplot(data=ct1_from_ct2_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct2_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct2_str} to {ct1_str} connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_in_{ct2_str}_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{ct1_str}_in_{ct2_str}_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=ct1_from_ct2_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of {ct2_str} partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct2_str} to {ct1_str} connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_in_{ct2_str}_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct1_str}_in_{ct2_str}_num_partners_hist.svg')
    plt.close()

    log.info('Step 5/X: Calculate overlap between incoming and outgoing cells from same celltype')
    #compare ct1 outgoing dict with ct1 incoming dict
    #only do for cells that get and receive input
    ct1_in_out_ids = ct1_proj_ssvs[np.in1d(ct1_proj_ssvs, ct1_rec_ssvs)]
    log.info(f'{len(ct1_in_out_ids)} {ct1_str} cells project to and get input from {ct2_str}')
    log.info(f'This is {100*len(ct1_in_out_ids)/len(ct1_proj_ssvs):.2f} percent of projecting cells and '
             f'{100*len(ct1_in_out_ids)/len(ct1_rec_ssvs):.2f} percent of receiving cells.')
    #for each cell get percent overlap in number of cells, syn number, summed size
    #save in dataframe
    columns = ['cellid', 'celltype', 'fraction cell overlap in', 'fraction syn number overlap in', 'fraction syn sum size overlap in',
               'fraction cell overlap out', 'fraction syn number overlap out', 'fraction syn sum size overlap out']
    ct1_overlap_df = pd.DataFrame(columns=columns, index = range(len(ct1_in_out_ids)))
    ct1_overlap_df['cellid'] = ct1_in_out_ids
    ct1_overlap_df['celltype'] = ct1_str

    raise ValueError

    #calculate overlap between the two in terms of cellids
    #calculate overlap between the two in terms of syn number
    #plot for both celltypes

    #get statistics, overview params