#analysis to see how many cells are part of a pathway/ connectivity made up of 3 celltypes
#ct1 -> ct2 -> ct2
#use synapses to determine number of cells that get input from other celltype and project to
#also get synaptic area per celltype and mean/std per cell

#use code from GPe_NGf_MSN_loop analysis

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize, filter_synapse_caches_general
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import SubCT_Colors
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
    import seaborn as sns
    import matplotlib.pyplot as plt

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'


    version = 'v6'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict()
    global_params.wd = bio_params.working_dir()
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    min_comp_len = 200
    min_comp_len_ax = 50
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'MSN','TeYw','MudGrays'}
    color_key = 'STNGPINTv6'
    ct1 = 7
    ct2 = 6
    ct3 = 4
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    ct3_str = ct_dict[ct3]
    axon_cts = bio_params.axon_cts()
    comp_cts = [ct1, ct2, ct3]
    if np.any(np.in1d(comp_cts, axon_cts)):
        axon_ct_present = True
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'241212j0251{version}_cellnumber_pathway_analysis_{ct1_str}_{ct2_str}_{ct3_str}_{min_comp_len}_{min_comp_len_ax}_{color_key}'
    else:
        axon_ct_present = False
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'241212_j0251{version}_cellnumber_pathway_analysis_{ct1_str}_{ct2_str}_{ct3_str}_{min_comp_len}_{color_key}'
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'Pathway cell number analysis {ct1_str}, {ct2_str}, {ct3_str}', log_dir=f_name + '/logs/')
    if axon_ct_present:
        log.info(f'min comp len cell = {min_comp_len}, min comp len ax = {min_comp_len_ax}, exclude known mergers = {exclude_known_mergers}, '
                 f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}')
    else:
        log.info(
            f'min comp len cell = {min_comp_len}, exclude known mergers = {exclude_known_mergers}, '
            f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}')
    log.info("Step 1/5: Load celltypes and check suitability")

    cts_str_analysis = [ct_dict[i] for i in comp_cts]
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in tqdm(comp_cts):
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        if ct in axon_cts:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                                    axon_only=True,
                                                    max_path_len=None)
        else:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    number_ids = [len(suitable_ids_dict[ct]) for ct in comp_cts]
    all_suitable_ids = np.hstack(all_suitable_ids)
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")

    log.info('Step 2/5: Prefilter synapses with syn prob, min syn size and suitable cellids')
    # prefilter synapses for synapse prob thresh and min syn size
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #prefilter so that all synapses are between suitable ids
    suit_ids_ind = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ids_ind]
    m_ids = m_ids[suit_ids_ind]
    m_sizes = m_sizes[suit_ids_ind]
    m_axs = m_axs[suit_ids_ind]
    m_rep_coord = m_rep_coord[suit_ids_ind]
    m_spiness = m_spiness[suit_ids_ind]
    m_cts = m_cts[suit_ids_ind]
    syn_prob = syn_prob[suit_ids_ind]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    #make dataframe with all summary statistics
    sum_columns = [f'{ct1_str} to {ct2_str}', f'{ct2_str} from {ct1_str}', f'{ct2_str} to {ct3_str}', f'{ct3_str} from {ct2_str}']
    sum_index = ['number of cells', 'percent of cells', 'total synaptic area', 'total synaptic area dataset normed','median synapse number', 'median synapse size',
                 'mean synapse number', 'mean synapse size', 'std synapse number', 'std synapse size',
                 'number of other cells median', 'number of other cells mean', 'number of other cells std',
                 'median syn number per other cell', 'median syn area per other cell',
                 'mean syn number per other cell', 'mean syn area per other cell',
                 'std syn number per other cell', 'std syn area per other cell']
    summary_df = pd.DataFrame(columns=sum_columns, index = sum_index)

    #get summed synapse size of suitable synapse in full dataset
    all_syn_path = 'cajal/scratch/users/arother/bio_analysis_results/general/241024_j0251v6_cts_percentages_mcl_50_ax50_synprob_0.60_TePkBrNGF_newmergers_bw_fs_20/' \
                   'outgoing_syn_sizes_matrix_abs_sum.csv'
    summed_syns_all_df = pd.read_csv(all_syn_path, index_col = 0)
    total_sumsize = summed_syns_all_df.sum().sum()
    log.info(f'total synapse number was loaded from {all_syn_path} and is {total_sumsize:.2f} µm²')


    log.info(f'Step 3/5: Identify {ct2_str} cells that get {ct1_str} input')
    #prefilter synapses between ct1 and ct2, only use suitable cellids
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct1],
                                                                                                        post_cts=[ct2],
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)
    #get all cellids of ct1 that make synapses to ct2
    axo_inds = np.where(m_axs == 1)
    axo_ssv_partners = m_ssv_partners[axo_inds]
    ct1_ids_2ct2 = np.unique(axo_ssv_partners)
    #check that all of them are really ct1 suitable ids
    assert(np.all(np.in1d(ct1_ids_2ct2, suitable_ids_dict[ct1])))
    # get all cellids of ct2 that receive synapses from ct2
    denso_inds = np.where(m_axs != 1)
    denso_ssv_partners = m_ssv_partners[denso_inds]
    ct2_ids_fct1 = np.unique(denso_ssv_partners)
    # check that all of them are really ct1 suitable ids
    assert (np.all(np.in1d(ct2_ids_fct1, suitable_ids_dict[ct2])))
    log.info(f'Total synaptic strength from {ct1_str} to {ct2_str} are {np.sum(m_sizes):.2f} µm² from {len(m_sizes)} synapses')
    summary_df.loc['total synaptic area',f'{ct1_str} to {ct2_str}'] = np.sum(m_sizes)
    summary_df.loc['total synaptic area', f'{ct2_str} from {ct1_str}'] = np.sum(m_sizes)
    summary_df.loc['total synaptic area dataset normed', f'{ct1_str} to {ct2_str}'] = np.sum(m_sizes)/ total_sumsize
    summary_df.loc['total synaptic area dataset normed', f'{ct2_str} from {ct1_str}'] = np.sum(m_sizes)/ total_sumsize
    #get syn numbers and syn sizes per cell connectivity for ct1
    ct1_syn_numbers, ct1_syn_ssv_sizes, ct1_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                             syn_ssv_partners=m_ssv_partners,
                                                                             syn_cts=m_cts, ct=ct1)
    np.save(f'{f_name}/{ct1_str}_proj_ids_2{ct2_str}.npy', ct1_proj_ssvs)
    num_ct1_proj_ssvs = len(ct1_proj_ssvs)
    perc_ct1_proj_ssvs = 100 * len(ct1_proj_ssvs)/ len(suitable_ids_dict[ct1])
    med_syn_number = np.median(ct1_syn_numbers)
    med_syn_size = np.median(ct1_syn_ssv_sizes)
    log.info(f'{num_ct1_proj_ssvs} {ct1_str} project to {ct2_str}. These are {perc_ct1_proj_ssvs:.2f}'
             f' percent of {ct1_str} cells')
    log.info(f'The median number of synapses are {med_syn_number}, sum size {med_syn_size:.2f} per cell')
    summary_df.loc['number of cells', f'{ct1_str} to {ct2_str}'] = num_ct1_proj_ssvs
    summary_df.loc['percent of cells', f'{ct1_str} to {ct2_str}'] = perc_ct1_proj_ssvs
    summary_df.loc['median synapse number', f'{ct1_str} to {ct2_str}'] = med_syn_number
    summary_df.loc['median synapse size', f'{ct1_str} to {ct2_str}'] = med_syn_size
    summary_df.loc['mean synapse number', f'{ct1_str} to {ct2_str}'] = np.mean(ct1_syn_numbers)
    summary_df.loc['mean synapse size', f'{ct1_str} to {ct2_str}'] = np.mean(ct1_syn_ssv_sizes)
    summary_df.loc['std synapse number', f'{ct1_str} to {ct2_str}'] = np.std(ct1_syn_numbers)
    summary_df.loc['std synapse size', f'{ct1_str} to {ct2_str}'] = np.std(ct1_syn_ssv_sizes)
    #create lookup dictionary which ct1 projects to which ct2 for later use
    ct1_proj_dict = {id: [] for id in ct1_proj_ssvs}
    ct1_2ct2_number = np.zeros(len(ct1_proj_ssvs))
    for gi, ct1_id in enumerate(ct1_proj_ssvs):
        ind = np.where(m_ssv_partners == ct1_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ct1_id)
        ct2_ids = np.unique(id_partners[ind])
        ct1_proj_dict[ct1_id].append(ct2_ids)
        ct1_2ct2_number[gi] = len(ct2_ids)
    write_obj2pkl(f'{f_name}/{ct1_str}_proj_{ct2_str}_dict.pkl', ct1_proj_dict)
    log.info(f'A median {ct1_str} projects to {np.median(ct1_2ct2_number)} {ct2_str} cells, with {np.median(ct1_syn_numbers/ct1_2ct2_number):.2f}'
             f' synapses and {np.median(ct1_syn_ssv_sizes/ct1_2ct2_number):.2f} synaptic area in µm²')
    summary_df.loc['number of other cells median', f'{ct1_str} to {ct2_str}'] = np.median(ct1_2ct2_number)
    summary_df.loc['number of other cells mean', f'{ct1_str} to {ct2_str}'] = np.mean(ct1_2ct2_number)
    summary_df.loc['number of other cells std', f'{ct1_str} to {ct2_str}'] = np.std(ct1_2ct2_number)
    summary_df.loc['median syn number per other cell', f'{ct1_str} to {ct2_str}'] = np.median(ct1_syn_numbers/ct1_2ct2_number)
    summary_df.loc['median syn area per other cell', f'{ct1_str} to {ct2_str}'] = np.median(ct1_syn_ssv_sizes/ct1_2ct2_number)
    summary_df.loc['mean syn number per other cell', f'{ct1_str} to {ct2_str}'] = np.mean(ct1_syn_numbers/ct1_2ct2_number)
    summary_df.loc['mean syn area per other cell', f'{ct1_str} to {ct2_str}'] = np.mean(ct1_syn_ssv_sizes/ct1_2ct2_number)
    summary_df.loc['std syn number per other cell', f'{ct1_str} to {ct2_str}'] = np.std(ct1_syn_numbers/ct1_2ct2_number)
    summary_df.loc['std syn area per other cell', f'{ct1_str} to {ct2_str}'] = np.std(ct1_syn_ssv_sizes/ct1_2ct2_number)
    #get ct2 ids that ct1 projects to
    ct2_syn_numbers, ct2_syn_ssv_sizes, ct2_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ct2)
    np.save(f'{f_name}/{ct2_str}_rec_ids_{ct1_str}.npy', ct2_rec_ssvs)
    num_ct2_rec_ssvs = len(ct2_rec_ssvs)
    perc_ct2_rec_ssvs = 100 * len(ct2_rec_ssvs) / len(suitable_ids_dict[ct2])
    med_syn_number = np.median(ct2_syn_numbers)
    med_syn_size = np.median(ct2_syn_ssv_sizes)
    log.info(f'{num_ct2_rec_ssvs} {ct2_str} receive {ct1_str} projections. These are {perc_ct2_rec_ssvs:.2f} '
             f'percent of {ct2_str} cells')
    log.info(
        f'The median number of synapses are {med_syn_number}, sum size {med_syn_size:.2f} per cell')
    summary_df.loc['number of cells', f'{ct2_str} from {ct1_str}'] = num_ct2_rec_ssvs
    summary_df.loc['percent of cells', f'{ct2_str} from {ct1_str}'] = perc_ct2_rec_ssvs
    summary_df.loc['median synapse number', f'{ct2_str} from {ct1_str}'] = med_syn_number
    summary_df.loc['median synapse size', f'{ct2_str} from {ct1_str}'] = med_syn_size
    summary_df.loc['mean synapse number', f'{ct2_str} from {ct1_str}'] = np.mean(ct2_syn_numbers)
    summary_df.loc['mean synapse size', f'{ct2_str} from {ct1_str}'] = np.mean(ct2_syn_ssv_sizes)
    summary_df.loc['std synapse number', f'{ct2_str} from {ct1_str}'] = np.std(ct2_syn_numbers)
    summary_df.loc['std synapse size', f'{ct2_str} from {ct1_str}'] = np.std(ct2_syn_ssv_sizes)
    ct2_fct1_number = np.zeros(len(ct2_rec_ssvs))
    ct2_rec_dict = {id: [] for id in ct2_rec_ssvs}
    for ni, ct2_id in enumerate(ct2_rec_ssvs):
        ind = np.where(m_ssv_partners == ct2_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ct2_id)
        ct1_ids = np.unique(id_partners[ind])
        ct2_rec_dict[ct2_id].append(ct1_ids)
        ct2_fct1_number[ni] = len(ct1_ids)
    write_obj2pkl(f'{f_name}/{ct2_str}_rec_{ct1_str}_dict.pkl', ct2_rec_dict)
    log.info(
        f'A median {ct2_str} receives syns from {np.median(ct2_fct1_number)} {ct1_str} cells, with {np.median(ct2_syn_numbers / ct2_fct1_number):.2f}'
        f' synapses and {np.median(ct2_syn_ssv_sizes / ct2_fct1_number):.2f} synaptic area in µm²')
    summary_df.loc['number of other cells median', f'{ct2_str} from {ct1_str}'] = np.median(ct2_fct1_number)
    summary_df.loc['number of other cells mean', f'{ct2_str} from {ct1_str}'] = np.mean(ct2_fct1_number)
    summary_df.loc['number of other cells std', f'{ct2_str} from {ct1_str}'] = np.std(ct2_fct1_number)
    summary_df.loc['median syn number per other cell', f'{ct2_str} from {ct1_str}'] = np.median(
        ct2_syn_numbers / ct2_fct1_number)
    summary_df.loc['median syn area per other cell', f'{ct2_str} from {ct1_str}'] = np.median(
        ct2_syn_ssv_sizes / ct2_fct1_number)
    summary_df.loc['mean syn number per other cell', f'{ct2_str} from {ct1_str}'] = np.mean(
        ct2_syn_numbers / ct2_fct1_number)
    summary_df.loc['mean syn area per other cell', f'{ct2_str} from {ct1_str}'] = np.mean(
        ct2_syn_ssv_sizes / ct2_fct1_number)
    summary_df.loc['std syn number per other cell', f'{ct2_str} from {ct1_str}'] = np.std(
        ct2_syn_numbers / ct2_fct1_number)
    summary_df.loc['std syn area per other cell', f'{ct2_str} from {ct1_str}'] = np.std(
        ct2_syn_ssv_sizes / ct2_fct1_number)

    log.info(f'Step 4/5: Get {ct2_str} - {ct3_str} info ')
    # prefilter synapses between ct2 and ct3, only use suitable cellids
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct2],
                                                                                                        post_cts=[ct3],
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)
    # get all cellids of ct2 that make synapses to ct3
    axo_inds = np.where(m_axs == 1)
    axo_ssv_partners = m_ssv_partners[axo_inds]
    ct2_ids_2ct3 = np.unique(axo_ssv_partners)
    # check that all of them are really ct2 suitable ids
    assert (np.all(np.in1d(ct2_ids_2ct3, suitable_ids_dict[ct2])))
    # get all cellids of ct3 that receive synapses from ct2
    denso_inds = np.where(m_axs != 1)
    denso_ssv_partners = m_ssv_partners[denso_inds]
    ct3_ids_fct2 = np.unique(denso_ssv_partners)
    # check that all of them are really ct2 suitable ids
    assert (np.all(np.in1d(ct3_ids_fct2, suitable_ids_dict[ct3])))
    #make sure only ct2 cells that receive ct1 input are used for further analysis
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, ct2_rec_ssvs).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    log.info(f'Total synaptic strength from {ct2_str} to {ct3_str} are {np.sum(m_sizes):.2f} µm² from {len(m_sizes)} synapses')
    summary_df.loc['total synaptic area', f'{ct2_str} to {ct3_str}'] = np.sum(m_sizes)
    summary_df.loc['total synaptic area', f'{ct3_str} from {ct2_str}'] = np.sum(m_sizes)
    summary_df.loc['total synaptic area dataset normed', f'{ct2_str} to {ct3_str}'] = np.sum(m_sizes) / total_sumsize
    summary_df.loc['total synaptic area dataset normed', f'{ct3_str} from {ct2_str}'] = np.sum(m_sizes) / total_sumsize
    # get ct2 cells that project to ct3
    ct2ct3_syn_numbers, ct2ct3_syn_ssv_sizes, ct2_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ct2)
    num_ct2_proj_ssvs = len(ct2_proj_ssvs)
    perc_ct2_proj_ssvs = 100 * len(ct2_proj_ssvs) / len(ct2_rec_ssvs)
    perc_ct2_proj_ssvs_all = 100 * len(ct2_proj_ssvs) / len(suitable_ids_dict[ct2])
    med_syn_number = np.median(ct2ct3_syn_numbers)
    med_syn_size = np.median(ct2ct3_syn_ssv_sizes)
    log.info(f'{num_ct2_proj_ssvs} {ct2_str} project to {ct3_str}. These are {perc_ct2_proj_ssvs_all:.2f}'
             f' percent of {ct2_str} cells, and {perc_ct2_proj_ssvs:.2f} of the {ct2_str} cells that receive input from {ct1_str}')
    log.info(
        f'The median number of synapses are {med_syn_number}, sum size {med_syn_size:.2f} per cell')
    np.save(f'{f_name}/{ct2_str}_proj_ids_2{ct3_str}.npy', ct2_proj_ssvs)
    summary_df.loc['number of cells', f'{ct2_str} to {ct3_str}'] = num_ct2_proj_ssvs
    summary_df.loc['percent of cells', f'{ct2_str} to {ct3_str}'] = perc_ct2_proj_ssvs
    summary_df.loc['median synapse number', f'{ct2_str} to {ct3_str}'] = med_syn_number
    summary_df.loc['median synapse size', f'{ct2_str} to {ct3_str}'] = med_syn_size
    summary_df.loc['mean synapse number', f'{ct2_str} to {ct3_str}'] = np.mean(ct2ct3_syn_numbers)
    summary_df.loc['mean synapse size',f'{ct2_str} to {ct3_str}'] = np.mean(ct2ct3_syn_ssv_sizes)
    summary_df.loc['std synapse number', f'{ct2_str} to {ct3_str}'] = np.std(ct2ct3_syn_numbers)
    summary_df.loc['std synapse size', f'{ct2_str} to {ct3_str}'] = np.std(ct2ct3_syn_ssv_sizes)
    ct2_proj_dict = {id: [] for id in ct2_proj_ssvs}
    ct2_2ct3_number = np.zeros(len(ct2_proj_ssvs))
    for ni, ct2_id in enumerate(ct2_proj_ssvs):
        ind = np.where(m_ssv_partners == ct2_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ct2_id)
        ct3_ids = np.unique(id_partners[ind])
        ct2_proj_dict[ct2_id].append(ct3_ids)
        ct2_2ct3_number[ni]= len(ct3_ids)
    write_obj2pkl(f'{f_name}/{ct2_str}_proj_{ct3_str}_dict.pkl', ct2_proj_dict)
    log.info(
        f'A median {ct2_str} projects to {np.median(ct2_2ct3_number)} {ct3_str} cells, with {np.median(ct2ct3_syn_numbers / ct2_2ct3_number):.2f}'
        f' synapses and {np.median(ct2ct3_syn_ssv_sizes / ct2_2ct3_number):.2f} synaptic area in µm²')
    summary_df.loc['number of other cells median', f'{ct2_str} to {ct3_str}'] = np.median(ct2_2ct3_number)
    summary_df.loc['number of other cells mean', f'{ct2_str} to {ct3_str}'] = np.mean(ct2_2ct3_number)
    summary_df.loc['number of other cells std', f'{ct2_str} to {ct3_str}'] = np.std(ct2_2ct3_number)
    summary_df.loc['median syn number per other cell', f'{ct2_str} to {ct3_str}'] = np.median(
        ct2ct3_syn_numbers / ct2_2ct3_number)
    summary_df.loc['median syn area per other cell', f'{ct2_str} to {ct3_str}'] = np.median(
        ct2ct3_syn_ssv_sizes / ct2_2ct3_number)
    summary_df.loc['mean syn number per other cell', f'{ct2_str} to {ct3_str}'] = np.mean(
        ct2ct3_syn_numbers / ct2_2ct3_number)
    summary_df.loc['mean syn area per other cell', f'{ct2_str} to {ct3_str}'] = np.mean(
        ct2ct3_syn_ssv_sizes / ct2_2ct3_number)
    summary_df.loc['std syn number per other cell', f'{ct2_str} to {ct3_str}'] = np.std(
        ct2ct3_syn_numbers / ct2_2ct3_number)
    summary_df.loc['std syn area per other cell', f'{ct2_str} to {ct3_str}'] = np.std(
        ct2ct3_syn_ssv_sizes / ct2_2ct3_number)
    # get ct3 ids that ct2 project to
    ct3_syn_numbers, ct3_syn_ssv_sizes, ct3_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                 syn_ssv_partners=m_ssv_partners,
                                                                                 syn_cts=m_cts, ct=ct3)
    num_ct3_rec_ssvs = len(ct3_rec_ssvs)
    perc_ct3_rec_ssvs = 100 * len(ct3_rec_ssvs) / len(suitable_ids_dict[ct3])
    med_syn_number = np.median(ct3_syn_numbers)
    med_syn_size = np.median(ct3_syn_ssv_sizes)
    log.info(f'{num_ct3_rec_ssvs} {ct3_str} receive synapses from {ct2_str}. These are {perc_ct3_rec_ssvs:.2f} '
             f'percent of {ct3_str} cells')
    log.info(
        f'The median number of synapses are {med_syn_number}, sum size {med_syn_size:.2f} per cell')
    np.save(f'{f_name}/{ct3_str}_rec_ids_{ct2_str}.npy', ct3_rec_ssvs)
    summary_df.loc['number of cells', f'{ct3_str} from {ct2_str}'] = num_ct3_rec_ssvs
    summary_df.loc['percent of cells', f'{ct3_str} from {ct2_str}'] = perc_ct3_rec_ssvs
    summary_df.loc['median synapse number', f'{ct3_str} from {ct2_str}'] = med_syn_number
    summary_df.loc['median synapse size', f'{ct3_str} from {ct2_str}'] = med_syn_size
    summary_df.loc['mean synapse number', f'{ct3_str} from {ct2_str}'] = np.mean(ct3_syn_numbers)
    summary_df.loc['mean synapse size', f'{ct3_str} from {ct2_str}'] = np.mean(ct3_syn_ssv_sizes)
    summary_df.loc['std synapse number', f'{ct3_str} from {ct2_str}'] = np.std(ct3_syn_numbers)
    summary_df.loc['std synapse size', f'{ct3_str} from {ct2_str}'] = np.std(ct3_syn_ssv_sizes)
    ct3_fct2_number = np.zeros(len(ct3_rec_ssvs))
    ct3_rec_dict = {id: [] for id in ct3_rec_ssvs}
    for mi, ct3_id in enumerate(ct3_rec_ssvs):
        ind = np.where(m_ssv_partners == ct3_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ct3_id)
        ct2_ids = np.unique(id_partners[ind])
        ct3_rec_dict[ct3_id].append(ct2_ids)
        ct3_fct2_number[mi] = len(ct2_ids)
    write_obj2pkl(f'{f_name}/{ct3_str}_rec_{ct2_str}_dict.pkl', ct3_rec_dict)
    log.info(
        f'A median {ct3_str} receives syns from {np.median(ct3_fct2_number)} {ct2_str} cells, with {np.median(ct3_syn_numbers / ct3_fct2_number):.2f}'
        f' synapses and {np.median(ct3_syn_ssv_sizes / ct3_fct2_number):.2f} synaptic area in µm²')
    summary_df.loc['number of other cells median', f'{ct3_str} from {ct2_str}'] = np.median(ct3_fct2_number)
    summary_df.loc['number of other cells mean', f'{ct3_str} from {ct2_str}'] = np.mean(ct3_fct2_number)
    summary_df.loc['number of other cells std', f'{ct3_str} from {ct2_str}'] = np.std(ct3_fct2_number)
    summary_df.loc['median syn number per other cell', f'{ct3_str} from {ct2_str}'] = np.median(
        ct3_syn_numbers / ct3_fct2_number)
    summary_df.loc['median syn area per other cell', f'{ct3_str} from {ct2_str}'] = np.median(
        ct3_syn_ssv_sizes / ct3_fct2_number)
    summary_df.loc['mean syn number per other cell', f'{ct3_str} from {ct2_str}'] = np.mean(
        ct3_syn_numbers / ct3_fct2_number)
    summary_df.loc['mean syn area per other cell', f'{ct3_str} from {ct2_str}'] = np.mean(
        ct3_syn_ssv_sizes / ct3_fct2_number)
    summary_df.loc['std syn number per other cell', f'{ct3_str} from {ct2_str}'] = np.std(
        ct3_syn_numbers / ct3_fct2_number)
    summary_df.loc['std syn area per other cell', f'{ct3_str} from {ct2_str}'] = np.std(
        ct3_syn_ssv_sizes / ct3_fct2_number)
    summary_df.to_csv(f'{f_name}/summary_cell_numbers_{ct1_str}_{ct2_str}_{ct3_str}.csv')

    log.info(f'{ct1_str} - {ct2_str} - {ct3_str} analysis done')