#GP - NGF - MSN Loop Analysis
#Goal: see if MSN targeted by NGF have a selectivity for GPi or GPe

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import SubCT_Colors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import \
        sort_by_connectivity
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
    #color keys: 'MSN','TeYw','MudGrays'}
    color_key = 'MSN'
    save_svg = True
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221125_j0251v4_GPe_MSN_NGF_loop_mcl_%i_synprob_%.2f_%s" % (
    min_comp_len, syn_prob, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe MSN NGF Loop analysis', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s" % (
        min_comp_len, syn_prob, min_syn_size, exclude_known_mergers, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/4: Load celltypes and check suitability")

    ngf_ct = 10
    gpe_ct = 6
    msn_ct = 2
    gpi_ct = 7
    cts_for_loading = [msn_ct, gpe_ct, ngf_ct, gpi_ct]
    cts_str_analysis = [ct_dict[i] for i in cts_for_loading]
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

    log.info('Step 2/4: Identify NGF cells that get GPe input')
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
    log.info(f'Total synaptic strength from GPe to NGF are {np.sum(m_sizes):.2f} µm² from {len(m_sizes)} synapses')
    #get GPe ids that project to NGF
    gpe_syn_numbers, gpe_syn_ssv_sizes, gpe_proj_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                             syn_ssv_partners=m_ssv_partners,
                                                                             syn_cts=m_cts, ct=gpe_ct)
    log.info(f'{len(gpe_proj_ssvs)} GPe project to NGF. These are {100 * len(gpe_proj_ssvs)/ len(suitable_ids_dict[gpe_ct]):.2f}'
             f' percent of GPe cells')
    log.info(f'The median number of synapses are {np.median(gpe_syn_numbers)}, sum size {np.median(gpe_syn_ssv_sizes):.2f} per cell')
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
    log.info(f'A median GPe projects to {np.median(gpe_ngf_number)} NGF cells, with {np.median(gpe_syn_numbers/gpe_ngf_number):.2f}'
             f' synapses and {np.median(gpe_syn_ssv_sizes/gpe_ngf_number):.2f} synaptic area in µm²')
    #get NGF ids that GPe project to
    ngf_syn_numbers, ngf_syn_ssv_sizes, ngf_rec_ssvs = get_ct_syn_number_sumsize(syn_sizes=m_sizes,
                                                                                  syn_ssv_partners=m_ssv_partners,
                                                                                  syn_cts=m_cts, ct=ngf_ct)
    log.info(f'{len(ngf_rec_ssvs)} NGF receive GPe projections. These are {100 * len(ngf_rec_ssvs) / len(suitable_ids_dict[ngf_ct]):.2f}'
             f'percent of NGF cells')
    log.info(
        f'The median number of synapses are {np.median(ngf_syn_numbers)}, sum size {np.median(ngf_syn_ssv_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/NGF_rec_GPe_ids.pkl', ngf_rec_ssvs)
    ngf_gpe_number = np.zeros(len(ngf_rec_ssvs))
    for ni, ngf_id in enumerate(ngf_rec_ssvs):
        ind = np.where(m_ssv_partners == ngf_id)[0]
        id_partners = m_ssv_partners[ind]
        ind = np.where(id_partners != gpe_id)
        gpe_ids = np.unique(id_partners[ind])
        ngf_gpe_number[ni] = len(gpe_ids)
    log.info(
        f'A median NGF receives syns from {np.median(ngf_gpe_number)} GPe cells, with {np.median(ngf_syn_numbers / ngf_gpe_number):.2f}'
        f' synapses and {np.median(ngf_syn_ssv_sizes / ngf_gpe_number):.2f} synaptic area in µm²')
    time_stamps = [time.time()]
    step_idents = ['get GPe-NGF information']

    log.info('Step 3/4: Get NGF - MSN info ')
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
        msn_ngf_number[ni] = len(ngf_ids)
    log.info(
        f'A median MSN receives syns from {np.median(msn_ngf_number)} NGF cells, with {np.median(msn_syn_numbers / msn_ngf_number):.2f}'
        f' synapses and {np.median(msn_syn_ssv_sizes / msn_ngf_number):.2f} synaptic area in µm²')
    # create dataframe for results per cell
    ct_nums = [len(suitable_ids_dict[ci]) for ci in cts_for_loading]
    max_ct_id_length = np.max(ct_nums[:-1])
    result_df = pd.DataFrame(index=range(max_ct_id_length))
    # only put ngf, gpe that are part of loop
    result_df.loc[0: len(msn_rec_ssvs) -1, 'MSN ids'] = msn_rec_ssvs
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
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF ids'] = ngf_proj_ssvs
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF syn number from GPe'] = ngf_syn_numbers
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF sum syn size from GPe'] = ngf_syn_ssv_sizes
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF number of GPe cells'] = ngf_gpe_number
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF syn number to MSN'] = ngfmsn_syn_numbers
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF sum syn size to MSN'] = ngfmsn_syn_ssv_sizes
    result_df.loc[0: len(ngf_proj_ssvs) -1,'NGF number of MSN cells'] = ngf_msn_number
    result_df.loc[0: len(gpe_proj_ssvs) -1,'GPe ids'] = gpe_proj_ssvs
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
    saving_dir = 'cajal/nvmescratch/users/arother/j0251v4_prep'
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
