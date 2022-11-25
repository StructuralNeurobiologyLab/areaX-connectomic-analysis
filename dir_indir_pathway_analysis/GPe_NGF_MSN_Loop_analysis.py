#GP - NGF - MSN Loop Analysis
#Goal: see if MSN targeted by NGF have a selectivity for GPi or GPe

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
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
    min_comp_len = bio_params.min_comp_length()
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGP'
    save_svg = True
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221124_j0251v4_GPe_MSN_NGF_loop_mcl_%i_synprob_%.2f_%s" % (
    min_comp_len, syn_prob, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe MSN NGF Loop analysis', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s" % (
        min_comp_len, syn_prob, min_syn_size, exclude_known_mergers, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/X: Load celltypes and check suitability")

    ngf_ct = 10
    gpe_ct = 6
    msn_ct = 2
    cts_for_loading = [msn_ct, gpe_ct, ngf_ct]
    cts_str_analysis = [ct_dict[i] for i in cts_for_loading]
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
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked

    number_ids = [len(suitable_ids_dict[ct]) for ct in cts_for_loading]
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")
    time_stamps = [time.time()]
    step_idents = ['loading cells']

    log.info('Step 2/X: Identify NGF cells that get GPe input')
    #prefilter synapses between NGF and GPe, only use suitable cellids
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                        pre_cts=[
                                                                                                            gpe_ct],
                                                                                                        post_cts=[
                                                                                                            ngf_ct],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True)
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[ngf_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[gpe_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    log.info(f'Total synaptic strength from GPe to NGF are {np.sum(m_sizes)} µm² from {len(m_sizes)} synapses')
    #get GPe ids that project to NGF
    gpe_syn_numbers, gpe_syn_ssv_sizes, gpe_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                             syn_ssv_partners=m_ssv_partners,
                                                                             syn_cts=m_cts, ct=gpe_ct)
    log.info(f'{len(gpe_proj_ssvs)} GPe project to NGF. These are {len(gpe_proj_ssvs)/ len(suitable_ids_dict[gpe_ct])}'
             f'percent of GPe cells')
    log.info(f'The median number of synapses are {np.median(gpe_syn_numbers)}, sum size {np.median(gpe_syn_ssv_sizes)} per cell')
    write_obj2pkl(f'{f_name}/GPe_proj_NGF_ids.pkl', gpe_proj_ssvs)
    #create lookup dictionary which GPe projects to which NGF to exclude ones projecting to NGF that do not project to MSN later
    gpe_proj_dict = {id: [] for id in gpe_proj_ssvs}
    gpe_ngf_number = np.zeros(len(gpe_proj_ssvs))
    for gi, gpe_id in enumerate(gpe_proj_ssvs):
        ind = np.where(m_ssv_partners == gpe_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != gpe_id)
        ngf_ids = np.unique(id_partners[ind])
        gpe_proj_dict[gpe_id].append(ngf_ids)
        gpe_ngf_number[gi] = len(ngf_ids)
    #get NGF ids that GPe project to
    ngf_syn_numbers, ngf_syn_ssv_sizes, ngf_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ngf_ct)
    log.info(f'{len(ngf_rec_ssvs)} NGF receive GPe projections. These are {len(ngf_rec_ssvs) / len(suitable_ids_dict[ngf_ct])}'
             f'percent of NGF cells')
    log.info(
        f'The median number of synapses are {np.median(ngf_syn_numbers)}, sum size {np.median(ngf_syn_ssv_sizes)} per cell')
    write_obj2pkl(f'{f_name}/NGF_rec_GPe_ids.pkl', ngf_rec_ssvs)
    ngf_gpe_number = np.zeros(len(ngf_rec_ssvs))
    for ni, ngf_id in enumerate(ngf_rec_ssvs):
        ind = np.where(m_ssv_partners == ngf_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != gpe_id)
        gpe_ids = np.unique(id_partners[ind])
        gpe_ngf_number[ni] = len(gpe_ids)
    time_stamps = [time.time()]
    step_idents = ['get GPe-NGF information']

    log.info('Step 3/X: Get NGF - MSN info ')
    # prefilter synapses between NGF and MSN, only use suitable cellids
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                        pre_cts=[
                                                                                                            ngf_ct],
                                                                                                        post_cts=[
                                                                                                            msn_ct],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True)
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[ngf_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, suitable_ids_dict[msn_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_cts = m_cts[suit_ct_inds]
    log.info(f'Total synaptic strength from NGF to MSN are {np.sum(m_sizes)} µm² from {len(m_sizes)} synapses')
    # get NGF ids that project to MSN
    ngfmsn_syn_numbers, ngfmsn_syn_ssv_sizes, ngf_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ngf_ct)
    log.info(f'{len(ngf_proj_ssvs)} NGF project to MSN. These are {len(ngf_proj_ssvs) / len(suitable_ids_dict[ngf_ct])}'
             f'percent of NGF cells')
    log.info(
        f'The median number of synapses are {np.median(gpe_syn_numbers)}, sum size {np.median(gpe_syn_ssv_sizes)} per cell')
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
    # get MSN ids that NGF project to
    msn_syn_numbers, msn_syn_ssv_sizes, msn_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                 syn_ssv_partners=m_ssv_partners,
                                                                                 syn_cts=m_cts, ct=msn_ct)
    log.info(f'{len(msn_rec_ssvs)} MSN receive synapses from NGF. These are {len(msn_rec_ssvs) / len(suitable_ids_dict[msn_ct])}'
             f'percent of MSN cells')
    log.info(
        f'The median number of synapses are {np.median(ngf_syn_numbers)}, sum size {np.median(ngf_syn_ssv_sizes)} per cell')
    write_obj2pkl(f'{f_name}/NGF_rec_GPe_ids.pkl', ngf_rec_ssvs)
    ngf_gpe_number = np.zeros(len(ngf_rec_ssvs))
    for ni, ngf_id in enumerate(ngf_rec_ssvs):
        ind = np.where(m_ssv_partners == ngf_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != gpe_id)
        gpe_ids = np.unique(id_partners[ind])
        gpe_ngf_number[ni] = len(gpe_ids)

    # create dataframe for results per cell
    ct_nums = [len(suitable_ids_dict[ci]) for ci in range(len(cts_for_loading))]
    max_ct_id_length = np.max(ct_nums)
    result_df = pd.DataFrame(index=max_ct_id_length)
    # only put ngf, gpe that are part of loop
    if len(ngf_proj_ssvs) != len(ngf_rec_ssvs):
    time_stamps = [time.time()]
    step_idents = ['get GPe-NGF information']
#Step 2: Identify MSN cells that get targeted by Loop NGF
#plot characteristics of projection
#what percentage targets MSN, what percentage of MSN get NGF input

#Step 3: get MSN specificity for GPe/GPi (if saved: just load ids that are part of the GPe, GPi
# bot or none groups), else look at the specificity here
#plot if there is a bias in terms of synaptic input, number of synapses etc to either of these groups
