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
    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    version = 'v6'
    bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    min_comp_len_ax = 50
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'MSN','TeYw','MudGrays'}
    color_key = 'STNGPNGF'
    ct1 = 4
    ct2 = 6
    ct3 = 7
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    ct3_str = ct_dict[ct3]
    axon_cts = bio_params.axon_cts()
    comp_cts = [ct1, ct2, ct3]
    if np.any(np.in1d(comp_cts, axon_cts)):
        axon_ct_present = True
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240117_j0251{version}_cellnumber_pathway_analysis_{ct1_str}_{ct2_str}_{ct3_str}_{min_comp_len}_{min_comp_len_ax}_{color_key}'
    else:
        axon_ct_present = False
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240117_j0251{version}_cellnumber_pathway_analysis_{ct1_str}_{ct2_str}_{ct3_str}_{min_comp_len}_{color_key}'
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
    log.info("Step 1/4: Load celltypes and check suitability")

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
            if ct == 2:
                misclassified_asto_ids = bio_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    number_ids = [len(suitable_ids_dict[ct]) for ct in comp_cts]
    all_suitable_ids = np.hstack(all_suitable_ids)
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")

    log.info('Step 2/X: Prefilter synapses with syn prob, min syn size and suitable cellids')
    # prefilter synapses for synapse prob thresh and min syn size
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #prefilter so that all synapses are between suitable ids
    suit_ids_ind = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ids_ind]
    m_sizes = m_sizes[suit_ids_ind]
    m_axs = m_axs[suit_ids_ind]
    m_rep_coord = m_rep_coord[suit_ids_ind]
    m_spiness = m_spiness[suit_ids_ind]
    m_cts = m_cts[suit_ids_ind]
    syn_prob = syn_prob[suit_ids_ind]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    #make dataframe with all summary statistics
    sum_columns = [f'{ct1_str} to {ct2_str}', f'{ct2_str} from {ct1_str}', f'{ct2_str} to {ct3_str}', f'{ct3_str} from {ct2_str}']
    sum_index = ['number of cells', 'percent of cells', 'total synaptic area', 'median synapse number', 'median synapse size',
                 'mean synapse number', 'mean synapse size', 'std synapse number', 'std synapse size',
                 'number of other cells median', 'number of other cells mean', 'number of other cells std',
                 'median syn number per other cell', 'median syn area per other cell',
                 'mean syn number per other cell', 'mean syn area per other cell',
                 'std syn number per other cell', 'std syn area per other cell']
    summary_df = pd.DataFrame(columns=sum_columns, index = sum_index)


    log.info(f'Step 3/X: Identify {ct2_str} cells that get {ct1_str} input')
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
    log.info(f'{num_ct2_rec_ssvs} {ct2_str} receive {ct1_str} projections. These are {perc_ct2_rec_ssvs:.2f}'
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
    for ni, ct2_id in enumerate(ct2_rec_ssvs):
        ind = np.where(m_ssv_partners == ct2_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ct2_id)
        gct1_ids = np.unique(id_partners[ind])
        ct2_fct1_number[ni] = len(gct1_ids)
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

    log.info(f'Step 4/X: Get {ct2_str} - {ct3_str} info ')
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
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[msn_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    log.info(f'Total synaptic strength from NGF to MSN are {np.sum(m_sizes):.2f} µm² from {len(m_sizes)} synapses')
    # get NGF ids that project to MSN
    ngfmsn_syn_numbers, ngfmsn_syn_ssv_sizes, ngf_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ngf_ct)
    log.info(f'{len(ngf_proj_ssvs)} NGF project to MSN. These are {100 * len(ngf_proj_ssvs) / len(suitable_ids_dict[ngf_ct]):.2f}'
             f' percent of NGF cells')
    log.info(
        f'The median number of synapses are {np.median(gpe_syn_numbers)}, sum size {np.median(gpe_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/NGF_proj_msn_ids.pkl', ngf_proj_ssvs)
    ngf_proj_dict = {id: [] for id in ngf_proj_ssvs}
    ngf_msn_number = np.zeros(len(ngf_proj_ssvs))
    for ni, ngf_id in enumerate(ngf_proj_ssvs):
        ind = np.where(m_ssv_partners == ngf_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != ngf_id)
        msn_ids = np.unique(id_partners[ind])
        ngf_proj_dict[ngf_id].append(msn_ids)
        ngf_msn_number[ni]= len(msn_ids)
    log.info(
        f'A median NGF projects to {np.median(ngf_msn_number)} MSN cells, with {np.median(ngfmsn_syn_numbers / ngf_msn_number):.2f}'
        f' synapses and {np.median(ngfmsn_syn_ssv_sizes / ngf_msn_number):.2f} synaptic area in µm²')
    # get MSN ids that NGF project to
    msn_syn_numbers, msn_syn_ssv_sizes, msn_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                 syn_ssv_partners=m_ssv_partners,
                                                                                 syn_cts=m_cts, ct=msn_ct)
    log.info(f'{len(msn_rec_ssvs)} MSN receive synapses from NGF. These are {100 *len(msn_rec_ssvs) / len(suitable_ids_dict[msn_ct]):.2f}'
             f'percent of MSN cells')
    log.info(
        f'The median number of synapses are {np.median(ngf_syn_numbers)}, sum size {np.median(ngf_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/MSN_rec_NGF_ids.pkl', msn_rec_ssvs)
    msn_ngf_number = np.zeros(len(msn_rec_ssvs))
    for mi, msn_id in enumerate(msn_rec_ssvs):
        ind = np.where(m_ssv_partners == msn_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != msn_id)
        ngf_ids = np.unique(id_partners[ind])
        msn_ngf_number[mi] = len(ngf_ids)
    log.info(
        f'A median MSN receives syns from {np.median(msn_ngf_number)} NGF cells, with {np.median(msn_syn_numbers / msn_ngf_number):.2f}'
        f' synapses and {np.median(msn_syn_ssv_sizes / msn_ngf_number):.2f} synaptic area in µm²')
    # create dataframe for results per cell
    ct_nums = [len(suitable_ids_dict[ci]) for ci in cts_for_loading]
    max_ct_id_length = np.max(ct_nums[:-1])
    result_df = pd.DataFrame(index=range(max_ct_id_length))
    # only put ngf, gpe that are part of loop
    result_df.loc[0: len(msn_rec_ssvs) -1, 'MSN ids'] = msn_rec_ssvs.astype(int)
    result_df.loc[0: len(msn_rec_ssvs) -1, 'MSN syn number from NGF'] = msn_syn_numbers
    result_df.loc[0: len(msn_rec_ssvs) -1, 'MSN sum syn size from NGF'] = msn_syn_ssv_sizes
    result_df.loc[0: len(msn_rec_ssvs) -1, 'MSN number of NGF cells'] = msn_ngf_number
    sort_inds_proj = np.argsort(ngf_proj_ssvs)
    sort_inds_rec = np.argsort(ngf_rec_ssvs)
    ngfmsn_syn_numbers = ngfmsn_syn_numbers[sort_inds_proj]
    ngfmsn_syn_ssv_sizes = ngfmsn_syn_ssv_sizes[sort_inds_proj]
    ngf_proj_ssvs = ngf_proj_ssvs[sort_inds_proj]
    ngf_msn_number = ngf_msn_number[sort_inds_proj]
    ngf_syn_numbers = ngf_syn_numbers[sort_inds_rec]
    ngf_syn_ssv_sizes = ngf_syn_ssv_sizes[sort_inds_rec]
    ngf_rec_ssvs = ngf_rec_ssvs[sort_inds_rec]
    ngf_gpe_number = ngf_gpe_number[sort_inds_rec]
    if len(ngf_proj_ssvs) != len(ngf_rec_ssvs):
        loop_inds_proj = np.in1d(ngf_proj_ssvs, ngf_rec_ssvs)
        ngfmsn_syn_numbers = ngfmsn_syn_numbers[loop_inds_proj]
        ngfmsn_syn_ssv_sizes = ngfmsn_syn_ssv_sizes[loop_inds_proj]
        ngf_msn_number = ngf_msn_number[loop_inds_proj]
        ngf_proj_ssvs = ngf_proj_ssvs[loop_inds_proj]
        loop_inds_rec = np.in1d(ngf_rec_ssvs, ngf_proj_ssvs)
        ngf_syn_numbers = ngf_syn_numbers[loop_inds_rec]
        ngf_syn_ssv_sizes = ngf_syn_ssv_sizes[loop_inds_rec]
        ngf_gpe_number = ngf_gpe_number[loop_inds_rec]
        log.info(f'{100 * len(ngf_proj_ssvs)/ len(ngf_rec_ssvs):.2f} percent of NGF that get GPe input project to MSN')
        gpengf_proj_inds = []
        for gi, gpe_id in enumerate(gpe_proj_ssvs):
            ngf_ids = gpe_proj_dict[gpe_id]
            if not np.all(np.in1d(ngf_proj_ssvs, ngf_ids)) == False:
                gpengf_proj_inds.append(gi)
        if len(gpengf_proj_inds) > 0:
            log.info(f'{len(gpengf_proj_inds)} GPe project only to NGF that do not project to MSN, that is {100 * len(gpengf_proj_inds)/ len(gpe_proj_ssvs):.2f} percent.')
            gpe_proj_ssvs  = gpe_proj_ssvs[gpengf_proj_inds]
            gpe_syn_numbers = gpe_syn_numbers[gpengf_proj_inds]
            gpe_syn_ssv_sizes = gpe_syn_ssv_sizes[gpengf_proj_inds]
            gpe_ngf_number = gpe_ngf_number[gpengf_proj_inds]
        else:
            log.info('All GPe project to NGF that then projct to MSN')
    else:
        log.info('All NGF that receive GPe input project to MSN')
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF ids'] = ngf_proj_ssvs.astype(int)
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF syn number from GPe'] = ngf_syn_numbers
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF sum syn size from GPe'] = ngf_syn_ssv_sizes
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF number of GPe cells'] = ngf_gpe_number
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF syn number to MSN'] = ngfmsn_syn_numbers
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF sum syn size to MSN'] = ngfmsn_syn_ssv_sizes
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF number of MSN cells'] = ngf_msn_number
    result_df.loc[0: len(gpe_proj_ssvs) -1,'GPe ids'] = gpe_proj_ssvs.astype(int)
    result_df.loc[0: len(gpe_proj_ssvs) -1,'GPe syn number to NGF'] = gpe_syn_numbers
    result_df.loc[0: len(gpe_proj_ssvs) -1,'GPe sum syn size to NGF'] = gpe_syn_ssv_sizes
    result_df.loc[0: len(gpe_proj_ssvs) -1,'GPe number of NGF cells'] = gpe_ngf_number
    #make histograms for each parameter
    for key in result_df:
        if 'ids' in key:
            continue
        if 'syn number' in key:
            xlabel = 'number of synapses'
        elif 'number of' in key:
            xlabel = ' '.join(key.split(' ')[1:])
        else:
            xlabel = 'sum of synapse mesh area [µm²]'
        sns.histplot(data=result_df[key], color='black', fill=False, element="step", bins = 15, common_norm=True)
        plt.ylabel('fraction of cells')
        plt.xlabel(xlabel)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}.png')
        plt.close()
    time_stamps = [time.time()]
    step_idents = ['get NGF-MSN information']

    log.info('Step 4/4: Get MSN ids for their connectivity to GPe and GPi')
    saving_dir = 'cajal/nvmescratch/users/arother/j0251v5_prep'
    msn_2_gpi_ids = load_pkl2obj(f'{saving_dir}/full_MSN_2_GPi_arr_{min_comp_len}.pkl')
    msn_2_gpe_ids = load_pkl2obj(f'{saving_dir}/full_MSN_2_GPe_arr_{min_comp_len}.pkl')
    msn_2_bothgp_ids = load_pkl2obj(f'{saving_dir}/full_MSN_2_GPeGPi_arr_{min_comp_len}.pkl')
    msn_2_bothgp_ids = list(msn_2_bothgp_ids.keys())
    msn_2_nogp_ids = load_pkl2obj(f'{saving_dir}/full_MSN_no_conn_GPeGPi_arr_{min_comp_len}.pkl')
    ngf_msn2gpi_ids = msn_rec_ssvs[np.in1d(msn_rec_ssvs, msn_2_gpi_ids)]
    ngf_msn2gpe_ids = msn_rec_ssvs[np.in1d(msn_rec_ssvs, msn_2_gpe_ids)]
    ngf_msn2both_ids = msn_rec_ssvs[np.in1d(msn_rec_ssvs, msn_2_bothgp_ids)]
    ngf_msn2no_ids = msn_rec_ssvs[np.in1d(msn_rec_ssvs, msn_2_nogp_ids)]
    perc_2gpi = 100 * len(ngf_msn2gpi_ids)/ len(msn_rec_ssvs)
    perc_2gpe = 100 * len(ngf_msn2gpe_ids) / len(msn_rec_ssvs)
    perc_2both = 100 * len(ngf_msn2both_ids) / len(msn_rec_ssvs)
    perc_2none = 100 * len(ngf_msn2no_ids) / len(msn_rec_ssvs)
    log.info(f' Out of {len(msn_rec_ssvs)} MSN cells that receive NGF input, {len(ngf_msn2gpi_ids)} cells ({perc_2gpi:.2f} %)'
             f' project only to GPi, {len(ngf_msn2gpe_ids)} cells ({perc_2gpe:.2f} %) project only to GPe, '
             f'{len(ngf_msn2both_ids)} cells ({perc_2both:.2f} %) to both and {len(ngf_msn2no_ids)} cells ({perc_2none:.2f} %) to none')
    labels = ['only GPi', 'only GPe', 'both GP', 'no GP']
    subtype_ids = [ngf_msn2gpi_ids, ngf_msn2gpe_ids, ngf_msn2both_ids, ngf_msn2no_ids]
    length = max(map(len, subtype_ids))
    subtype_ids = np.array([np.hstack([xi,[None]*(length-len(xi))]) for xi in subtype_ids])
    for mi, msn_id in enumerate(msn_rec_ssvs):
        type_ind = int(np.where(subtype_ids == msn_id)[0])
        result_df.loc[mi, 'MSN GP connectivity type'] = labels[type_ind]
    result_df.to_csv(f'{f_name}/gpe_ngf_msn_loop_results.csv')
    #plot results comparing different MSN Groups
    cts = SubCT_Colors(labels)
    try:
        msn_palette = cts.get_subct_palette(color_key)
    except KeyError:
        msn_palette = cts.get_subct_palette_fromct(key = color_key, ct = msn_ct, light= False)
    for key in result_df:
        if 'ids' in key or 'GPe' in key or not 'MSN' in key or 'type' in key:
            continue
        if 'to MSN' in key or 'from MSN' in key:
            continue
        if 'syn number' in key:
            ax_label = 'number of synapses'
        elif 'number of' in key:
            ax_label = ' '.join(key.split(' ')[1:])
        else:
            ax_label = 'sum of synapse mesh area [µm²]'
        sns.histplot(x = key, hue = 'MSN GP connectivity type', data=result_df, palette=msn_palette, fill=False,
                     element="step", bins = 15, common_norm=True, legend = True)
        plt.ylabel('Fraction of cells')
        plt.xlabel(ax_label)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_subtypes_hist_norm.png')
        if save_svg:
            plt.savefig(f'{f_name}/{key}_subtypes_hist_norm.svg')
        plt.close()
        sns.histplot(x=key, hue='MSN GP connectivity type', data=result_df, palette=msn_palette, fill=False,
                     element="step", bins=15, common_norm=False, legend = True)
        plt.ylabel('Count of cells')
        plt.xlabel(ax_label)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_subtypes_hist.png')
        if save_svg:
            plt.savefig(f'{f_name}/{key}_subtypes_hist.svg')
        plt.close()
        sns.boxplot(y=key, x = 'MSN GP connectivity type', data=result_df, palette=msn_palette)
        plt.ylabel(ax_label)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_subtypes_box.png')
        if save_svg:
            plt.savefig(f'{f_name}/{key}_subtypes_box.svg')
        plt.close()
        sns.stripplot(x='MSN GP connectivity type', y=key, data=result_df, color="black", alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(x='MSN GP connectivity type', y=key, data=result_df, palette=msn_palette)
        plt.ylabel(ax_label)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_subtypes_violin.png')
        if save_svg:
            plt.savefig(f'{f_name}/{key}_subtypes_violin.svg')
        plt.close()
    time_stamps = [time.time()]
    step_idents = ['get MSN subtype information']

    log.info('GPe-NGF-MSN Loop analysis done')