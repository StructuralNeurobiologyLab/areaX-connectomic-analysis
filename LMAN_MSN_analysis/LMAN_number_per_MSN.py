
if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ResultsForPlotting
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
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors


    #global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    version = 'v6'
    bio_params = Analysis_Params(version=version)
    global_params.wd = bio_params.working_dir()
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 3
    lman_ct = 1
    gpi_ct = 7
    color_key = 'STNGPINTv6'
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(color_key, num=False)
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/LMAN_MSN_analysis/" \
                       f"241016_j0251{version}_lman_number_msn_mcl{min_comp_len}_syn{syn_prob}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('LMAN MSN connectivity estimate', log_dir=f_name + '/logs/')
    log.info(f"min_comp_len ={min_comp_len} µm, use handpicked LMAN axons, "
             f"syn_prob = {syn_prob}, min_syn_size = {min_syn_size} µm")

    # 1st part of the analysis: get estimate on how many "complete" LMAN branches
    # project to one MSN and how many MSN one LMAN projects to

    log.info("Step 1/8: load suitable LMAN and MSN, filter for min_comp_len and max_path_len")
    # load full MSN and filter for min_comp_len, also filter out if total_comp_len > 7500 mm (likely glia merger)
    LMAN_dict = bio_params.load_cell_dict(lman_ct)
    LMAN_ids = load_pkl2obj(f"{bio_params.file_locations}/LMAN_handpicked_arr.pkl")
    log.info(f'LMAN ids loaded from {bio_params.file_locations}/LMAN_handpicked_arr.pkl')
    MSN_dict = bio_params.load_cell_dict(msn_ct)
    MSN_ids = np.array(list(MSN_dict.keys()))
    known_mergers =bio_params.load_known_mergers()
    merger_inds = np.in1d(MSN_ids, known_mergers) == False
    MSN_ids = MSN_ids[merger_inds]
    misclassified_asto_ids = bio_params.load_potential_astros()
    astro_inds = np.in1d(MSN_ids, misclassified_asto_ids) == False
    MSN_ids = MSN_ids[astro_inds]
    MSN_ids = check_comp_lengths_ct(cellids=MSN_ids, fullcelldict=MSN_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    log.info("Step 2/8: filter synapses from suitable LMAN and MSN")
    #prefilter synapse caches from LMAN onto MSN synapses
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv=sd_synssv, pre_cts = [lman_ct],
                                                                                                        post_cts = [msn_ct], syn_prob_thresh = syn_prob,
                                                                                                        min_syn_size = min_syn_size, axo_den_so = True)
    #filter out synapses that are not from LMAN or MSN ids
    msnids_inds = np.any(np.in1d(m_ssv_partners, MSN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[msnids_inds]
    m_ids = m_ids[msnids_inds]
    m_ssv_partners = m_ssv_partners[msnids_inds]
    m_sizes  = m_sizes[msnids_inds]
    lmanids_inds = np.any(np.in1d(m_ssv_partners, LMAN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[lmanids_inds]
    m_ids = m_ids[lmanids_inds]
    m_ssv_partners = m_ssv_partners[lmanids_inds]
    m_sizes = m_sizes[lmanids_inds]

    log.info("Step 3/8: Create per cell dictionary for LMAN and MSN")
    #make per MSN dictionary of LMAN ids they are getting input from, number of LMAN
    #number of synapses, in total and per axon, sum of mesh area
    #create similar dictionary but with LMAN ids as key
    msn_inds = np.where(m_cts == msn_ct)
    msn_ssvsids = m_ssv_partners[msn_inds]
    msn_ssv_inds, unique_msn_ssvs = pd.factorize(msn_ssvsids)
    msn_syn_sumsizes = np.bincount(msn_ssv_inds, m_sizes)
    msn_syn_number = np.bincount(msn_ssv_inds)

    lman_inds = np.where(m_cts == lman_ct)
    lman_ssvsids = m_ssv_partners[lman_inds]
    lman_ssv_inds, unique_lman_ssvs = pd.factorize(lman_ssvsids)
    lman_syn_sumsizes = np.bincount(lman_ssv_inds, m_sizes)
    lman_syn_number = np.bincount(lman_ssv_inds)

    lman_ssv_pd = pd.DataFrame(lman_ssvsids)
    permsn_lman_ids_grouped = lman_ssv_pd.groupby(by = msn_ssv_inds)
    permsn_lman_groups = permsn_lman_ids_grouped.groups

    msn_ssv_pd = pd.DataFrame(msn_ssvsids)
    perlman_msn_ids_grouped = msn_ssv_pd.groupby(by=lman_ssv_inds)
    perlman_msn_groups = perlman_msn_ids_grouped.groups

    LMAN_proj_dict_percell = {id: {"MSN ids": np.unique(msn_ssvsids[perlman_msn_groups[i]]),
                           "number MSN cells": len(np.unique(msn_ssvsids[perlman_msn_groups[i]]))} for i, id in enumerate(unique_lman_ssvs)}
    number_msn_perlman = np.array([LMAN_proj_dict_percell[lman]["number MSN cells"] for lman in LMAN_proj_dict_percell])
    #calculate average soma distance between MSN targeted by same LMAN
    avg_soma_dist_msn = np.zeros(len(unique_lman_ssvs))
    max_soma_dist_msn = np.zeros(len(unique_lman_ssvs))
    for i,lman_id in enumerate(tqdm(unique_lman_ssvs)):
        lman_msn_ids = LMAN_proj_dict_percell[lman_id]["MSN ids"]
        soma_centres = np.zeros((len(lman_msn_ids), 3))
        for mi, msn_id in enumerate(lman_msn_ids):
            soma_centres[mi] = MSN_dict[msn_id]["soma centre"]
        pairwise_soma_distances = scipy.spatial.distance.pdist(soma_centres, metric="euclidean") / 1000
        avg_soma_dist_msn[i] = np.mean(pairwise_soma_distances)
        max_soma_dist_msn[i] = np.max(pairwise_soma_distances)

    LMAN_proj_dict = {"LMAN ids": unique_lman_ssvs, "number of synapses to MSN": lman_syn_number, "sum size synapses to MSN": lman_syn_sumsizes, "number MSN cells": number_msn_perlman,
                      "number of synapses per MSN": lman_syn_number/number_msn_perlman, "sum size synapses per MSM": lman_syn_sumsizes/number_msn_perlman,
                      "average soma distance MSN per LMAN": avg_soma_dist_msn, "maximum soma distance MSN per LMAN": max_soma_dist_msn}

    MSN_dict_percell = {id: {"LMAN ids": np.unique(lman_ssvsids[permsn_lman_groups[i]]),
                                   "number LMAN cells": len(np.unique(lman_ssvsids[permsn_lman_groups[i]]))} for
                              i, id in enumerate(unique_msn_ssvs)}
    number_lman_permsn = np.array([MSN_dict_percell[msn]["number LMAN cells"] for msn in MSN_dict_percell])
    MSN_rec_dict = {"MSN ids": unique_msn_ssvs, "number of synapses from LMAN": msn_syn_number, "sum size synapses from LMAN": msn_syn_sumsizes,
                      "number LMAN cells": number_lman_permsn,
                      "number of synapses per LMAN": msn_syn_number / number_lman_permsn,
                      "sum size synapses per LMAN": msn_syn_sumsizes / number_lman_permsn}

    write_obj2pkl("%s/lman_dict_percell.pkl" % f_name, LMAN_proj_dict_percell)
    write_obj2pkl("%s/lman_dict.pkl" % f_name, LMAN_proj_dict)
    write_obj2pkl("%s/msn_dict_percell" % f_name, MSN_dict_percell)
    write_obj2pkl("%s/msn_rec_dict.pkl" % f_name, MSN_rec_dict)

    lman_proj_pd = pd.DataFrame(LMAN_proj_dict)
    lman_proj_pd.to_csv("%s/lman_dict.csv" % f_name)

    msn_pd = pd.DataFrame(MSN_rec_dict)
    msn_pd.to_csv("%s/msn_rec_dict.csv" % f_name)

    log.info("Average number of MSNs per LMAN = %.2f" % np.mean(number_msn_perlman))
    log.info("Average number of LMAN per MSN = %.2f" % np.mean(number_lman_permsn))
    log.info("Average soma distance of MSN per LMAN = %.2f µm" % np.mean(avg_soma_dist_msn))
    log.info("Median number of MSNs per LMAN = %.2f" % np.median(number_msn_perlman))
    log.info("Median number of LMAN per MSN = %.2f" % np.median(number_lman_permsn))


    log.info("Step 4/8: Plot LMAN to MSN results")
    msn_results = ResultsForPlotting(celltype = ct_dict[msn_ct], filename = f_name, dictionary = MSN_rec_dict, color = ct_palette[ct_dict[msn_ct]])
    lman_results = ResultsForPlotting(celltype=ct_dict[lman_ct], filename=f_name, dictionary=LMAN_proj_dict, color = ct_palette[ct_dict[lman_ct]])

    for key in MSN_rec_dict:
        if "ids" in key:
            continue
        if "synapse" in key:
            msn_results.plot_hist(key, subcell="synapse", cells=False)
        else:
            msn_results.plot_hist(key, subcell="cell", cells=True)

    for key in LMAN_proj_dict:
        if "ids" in key:
            continue
        if "synapse" in key:
            lman_results.plot_hist(key, subcell="synapse", cells=False)
        else:
            lman_results.plot_hist(key, subcell="cell", cells=True)


    #2nd part of analysis: see how MSN from different LMANs project to GPi

    log.info("Step 5/8: load and filter GPi cells for min_comp_len")
    GPi_dict = bio_params.load_cell_dict(gpi_ct)
    GPi_ids = np.array(list(GPi_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    log.info("Step 6/8: Filter MSN to GPi synapses")
    #prefilter synapses from MSN -> GP
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv= sd_synssv,
                                                                                                        pre_cts=[msn_ct],
                                                                                                        post_cts=[gpi_ct],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True)
    #filter synapses again to only include selected MSN, GPi ids (only use MSN ids that are keys in the dictionary above)
    MSN_ids = list(MSN_dict_percell.keys())
    msnids_inds = np.any(np.in1d(m_ssv_partners, MSN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[msnids_inds]
    m_ids = m_ids[msnids_inds]
    m_ssv_partners = m_ssv_partners[msnids_inds]
    m_sizes = m_sizes[msnids_inds]
    gpiids_inds = np.any(np.in1d(m_ssv_partners, GPi_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[gpiids_inds]
    m_ids = m_ids[gpiids_inds]
    m_ssv_partners = m_ssv_partners[gpiids_inds]
    m_sizes = m_sizes[gpiids_inds]

    log.info("Step 7/8: Create per cell dictionary for GPi, add GPi info to existing MSN, LMAN dicts")
    #create dictionary with GPi as keys to see how many MSNs they get input from and how many LMANs
    #also add different GPis that MSN project to MSN dictionary
    #add GPis that are projected to via MSN to LMAN dictionary
    msn_inds = np.where(m_cts == msn_ct)
    msn_ssvsids = m_ssv_partners[msn_inds]
    msn_ssv_inds, unique_msn_ssvs = pd.factorize(msn_ssvsids)
    msn_syn_sumsizes = np.bincount(msn_ssv_inds, m_sizes)
    msn_syn_number = np.bincount(msn_ssv_inds)

    gpi_inds = np.where(m_cts == gpi_ct)
    gpi_ssvsids = m_ssv_partners[gpi_inds]
    gpi_ssv_inds, unique_gpi_ssvs = pd.factorize(gpi_ssvsids)
    gpi_syn_sumsizes = np.bincount(gpi_ssv_inds, m_sizes)
    gpi_syn_number = np.bincount(gpi_ssv_inds)

    gpi_ssv_pd = pd.DataFrame(gpi_ssvsids)
    permsn_gpi_ids_grouped = gpi_ssv_pd.groupby(by=msn_ssv_inds)
    permsn_gpi_groups = permsn_gpi_ids_grouped.groups

    msn_ssv_pd = pd.DataFrame(msn_ssvsids)
    pergpi_msn_ids_grouped = msn_ssv_pd.groupby(by=gpi_ssv_inds)
    pergpi_msn_groups = pergpi_msn_ids_grouped.groups

    GPi_rec_dict_percell = {id: {"MSN ids": np.unique(msn_ssvsids[pergpi_msn_groups[i]]),
                                   "number MSN cells": len(np.unique(msn_ssvsids[pergpi_msn_groups[i]]))} for
                              i, id in enumerate(unique_gpi_ssvs)}
    number_msn_pergpi = np.array([GPi_rec_dict_percell[gpi]["number MSN cells"] for gpi in GPi_rec_dict_percell])
    GPi_rec_dict = {"GPi ids": unique_gpi_ssvs, "number of synapses from MSN": gpi_syn_number, "sum size synapses from MSN": gpi_syn_sumsizes,
                      "number MSN cells": number_msn_pergpi,
                      "number of synapses per MSN": gpi_syn_number / number_msn_pergpi,
                      "sum size synapses per MSN": gpi_syn_sumsizes / number_msn_pergpi}

    MSN_proj_dict_percell = {id: {"GPi ids": np.unique(gpi_ssvsids[permsn_gpi_groups[i]]),
                                   "number MSN cells": len(np.unique(gpi_ssvsids[permsn_gpi_groups[i]]))} for
                              i, id in enumerate(unique_msn_ssvs)}
    number_gpi_permsn = np.array([len(np.unique(gpi_ssvsids[permsn_gpi_groups[i]])) for i in range(len(unique_msn_ssvs))])
    MSN_proj_dict = {"MSN ids": unique_msn_ssvs, "number of synapses to GPi": msn_syn_number, "sum size synapses to GPi": msn_syn_sumsizes,
                     "number of GPi cells": number_gpi_permsn, "number of synapses per GPi": msn_syn_number/ number_gpi_permsn,
                     "sum size synapses per GPi": msn_syn_sumsizes/ number_gpi_permsn}

    log.info("Average number of MSNs per GPi = %.2f" % np.mean(number_msn_pergpi))
    log.info("Average number of GPi per MSN = %.2f" % np.mean(number_gpi_permsn))
    log.info("Median number of MSNs per GPi = %.2f" % np.median(number_msn_pergpi))
    log.info("Median number of GPi per MSN = %.2f" % np.median(number_gpi_permsn))

    #compute highest percentage of MSN that go to one GP
    # compute number of GPi per LMAN
    # compute average soma distance between gpi of same lman
    number_gpi_perlman = np.zeros(len(LMAN_proj_dict_percell.keys()))
    hperc_samegpi_msn_lman = np.zeros(len(LMAN_proj_dict_percell.keys()))
    avg_soma_dist_gpi = np.zeros(len(LMAN_proj_dict_percell.keys()))
    max_soma_dist_gpi = np.zeros(len(LMAN_proj_dict_percell.keys()))
    for i, lman_id in enumerate(tqdm(LMAN_proj_dict_percell)):
        lman = LMAN_proj_dict_percell[lman_id]
        gpi_percell = []
        for msn_id in lman["MSN ids"]:
            try:
                gpi_percell.append(MSN_proj_dict_percell[msn_id]["GPi ids"])
            except KeyError:
                continue
        gpi_percell = np.concatenate(np.array(gpi_percell))
        gpi_inds, unique_gpi_percell = pd.factorize(gpi_percell)
        number_gpi_perlman[i] = len(unique_gpi_percell)
        lman["indirect GPi ids"] = unique_gpi_percell
        count_gpis = np.bincount(gpi_inds)
        max_count_gpis = np.max(count_gpis)
        hperc_samegpi_msn_lman[i] = max_count_gpis/len(gpi_percell)
        if len(unique_gpi_percell) > 0:
            soma_centres = np.zeros((len(unique_gpi_percell), 3))
            for gi, gpi_id in enumerate(unique_gpi_percell):
                soma_centres[gi] = GPi_dict[gpi_id]["soma centre"]
            pairwise_soma_distances = scipy.spatial.distance.pdist(soma_centres, metric = "euclidean") / 1000
            avg_soma_dist_gpi[i] = np.mean(pairwise_soma_distances)
            max_soma_dist_gpi[i] = np.max(pairwise_soma_distances)
        else:
            avg_soma_dist_gpi[i] = 0
            max_soma_dist_gpi[i] = 0

    # compute number of LMAN per GPi
    # compute highest percentage of MSN that receive input from one LMAN
    number_lman_pergpi = np.zeros(len(GPi_rec_dict_percell.keys()))
    hperc_samelman_msn_gpi = np.zeros(len(GPi_rec_dict_percell.keys()))
    for i, gpi_id in enumerate(tqdm(GPi_rec_dict_percell)):
        gpi = GPi_rec_dict_percell[gpi_id]
        lman_percell = []
        for msn_id in gpi["MSN ids"]:
            lman_percell.append(MSN_dict_percell[msn_id]["LMAN ids"])
        lman_percell = np.concatenate(np.array(lman_percell))
        lman_inds, unique_lman_percell = pd.factorize(lman_percell)
        number_lman_pergpi[i] = len(unique_lman_percell)
        gpi["indirect LMAN ids"] = unique_lman_percell
        count_lmans = np.bincount(lman_inds)
        max_count_lmans = np.max(count_lmans)
        hperc_samelman_msn_gpi[i] = max_count_lmans / len(lman_percell)

    # TO DO: save lman and gpi ids to check later
    #compute distance between GPi somata targeted by same LMAN

    LMAN_proj_dict["number of GPi"] = number_gpi_perlman
    GPi_rec_dict["number of LMAN"] = number_lman_pergpi
    LMAN_proj_dict["percentage of largest MSN group to same GPi"] = hperc_samegpi_msn_lman
    GPi_rec_dict["percentage of largest MSN group from same LMAN"] = hperc_samelman_msn_gpi
    LMAN_proj_dict["average soma distance GPi"] = avg_soma_dist_gpi
    LMAN_proj_dict["maximum soma distance GPi"] = max_soma_dist_gpi

    log.info("Average number of GPi from same LMAN via MSN = %.2f" % np.mean(number_gpi_perlman))
    log.info("Average number of LMAN from same GPi via MSN = %.2f" % np.mean(number_lman_pergpi))
    log.info("Average percentage of largest MSN group from LMAN to same GPi = %.2f" % np.mean(hperc_samegpi_msn_lman))
    log.info("Average percentage of largest MSN group to GPi from same LMAN = %.2f" % np.mean(hperc_samelman_msn_gpi))
    log.info("Average soma distance of GPi per LMAN = %.2f" % np.mean(avg_soma_dist_gpi))
    log.info("Median number of GPi from same LMAN via MSN = %.2f" % np.median(number_gpi_perlman))
    log.info("Median number of LMAN from same GPi via MSN = %.2f" % np.median(number_lman_pergpi))
    log.info("Median percentage of largest MSN group from LMAN to same GPi = %.2f" % np.median(hperc_samegpi_msn_lman))
    log.info("Median percentage of largest MSN group to GPi from same LMAN = %.2f" % np.median(hperc_samelman_msn_gpi))
    log.info("Number of GPi cells = %i" % len(unique_gpi_ssvs))
    log.info("Number of MSN cells = %i" % len(unique_msn_ssvs))
    log.info("Number of LMAN cells = %i" % len(unique_lman_ssvs))

    write_obj2pkl("%s/gpi_dict_percell.pkl" % f_name, GPi_rec_dict_percell)
    write_obj2pkl("%s/gpi_dict.pkl" % f_name, GPi_rec_dict)
    write_obj2pkl("%s/msn_proj_dict.pkl" % f_name, MSN_proj_dict)
    write_obj2pkl("%s/msn_proj_dict_percell.pkl" % f_name, MSN_proj_dict_percell)
    write_obj2pkl("%s/lman_dict.pkl" % f_name, LMAN_proj_dict)
    write_obj2pkl("%s/lman_dict_percell.pkl" % f_name, LMAN_proj_dict_percell)

    gpi_rec_pd = pd.DataFrame(GPi_rec_dict)
    gpi_rec_pd.to_csv("%s/gpi_dict.csv" % f_name)

    msn_pd = pd.DataFrame(MSN_proj_dict)
    msn_pd.to_csv("%s/msn_dict.csv" % f_name)

    lman_proj_pd = pd.DataFrame(LMAN_proj_dict)
    lman_proj_pd.to_csv("%s/lman_dict.csv" % f_name)

    log.info("Step 8/8: Plot results of LMAN -> MSN -> GPi connection")
    msn_results = ResultsForPlotting(celltype=ct_dict[msn_ct], filename=f_name, dictionary=MSN_proj_dict, color = ct_palette[ct_dict[msn_ct]])
    gpi_results = ResultsForPlotting(celltype=ct_dict[gpi_ct], filename=f_name, dictionary=GPi_rec_dict, color = ct_palette[ct_dict[gpi_ct]])
    lman_results = ResultsForPlotting(celltype=ct_dict[lman_ct], filename=f_name, dictionary=LMAN_proj_dict, color = ct_palette[ct_dict[lman_ct]])

    for key in MSN_proj_dict:
        if "MSN ids" in key:
            continue
        if "synapse" in key:
            msn_results.plot_hist(key, subcell="synapse", cells=False)
        else:
            msn_results.plot_hist(key, subcell="cells", cells=True)
    for key in GPi_rec_dict:
        if " GPi ids" in key:
            continue
        if "synapse" in key:
            gpi_results.plot_hist(key, subcell="synapse", cells=False)
        else:
            gpi_results.plot_hist(key, subcell="cells", cells=True)

    for key in LMAN_proj_dict:
        if "LMAN ids" in key:
            continue
        if not "GPi" in key:
            continue
        lman_results.plot_hist(key, subcell="cells", cells=True)

    log.info("LMAN, MSN, GPi number estimate analysis done")