if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import time
    from syconn.handler.config import initialize_logging
    from collections import defaultdict
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl

    #analyse the distance of neurons that receive the same inputs from LMAN, HVC using the middle of the soma skeleton nodes

    start = time.time()

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2"
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # find cells from modulatory category that are not only axons -> higher probability that its not a dopaminergic neuron; modulatory class is DA & TAN
    # celltypes j0126:STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    #celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10
    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime // 60, loadtime % 60))

    f_name = "u/arother/test_folder/200925_j0256_input_distance_analysis_80120"
    log = initialize_logging('LMAN_HVC distance analysis', log_dir=f_name + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #finding lman, HVC axons,msn
    log.info('Step 1/4 finding LMAN/ HVC axons, full MSNs, calculate middle of soma')
    lman_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 3]
    hvc_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 4]
    msn_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") ==2]

    #find whole MSNs and compute the medium of their MSN in a dictionary
    """
    full_msn_dict = defaultdict(lambda : np.zeros(3))

    for i, cell in enumerate(ssd.get_super_segmentation_object(msn_ids)):
        cell.load_skeleton()
        axoness = cell.skeleton["axoness_avg10000"]
        axoness[axoness == 3] = 1
        axoness[axoness == 4] = 1
        unique_preds = np.unique(axoness)
        if not (0 in unique_preds and 1 in unique_preds and 2 in unique_preds):
            continue
        soma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 2)[0]
        if len(soma_inds) == 0:
            raise ValueError
        positions = cell.skeleton["nodes"][soma_inds]
        soma_centre = np.sum(positions, axis = 0)/ len(positions)
        full_msn_dict[cell.id] = soma_centre
    """

    full_msn_dict = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_MSN_dict.pkl")

    #create dictionary for every axon to collect MSNs its synapsing to; also amount and size of synapses -> dictionary within dictionary
    lman_dict = {id: defaultdict(lambda : np.zeros(2)) for id in lman_ids}
    hvc_dict = {id: defaultdict(lambda : np.zeros(2)) for id in hvc_ids}

    celltime = time.time() - loadtime
    print("%.2f min, %.2f sec for finding cells" % (celltime // 60, celltime % 60))
    time_stamps = [time.time()]
    step_idents.append('finding cells')

    log.info('Step 2/4 sort synapses for LMAN/HVC & MSN synapses')

    #iterate over synapses to get axo-dendritic synapses to MSNs, which cells one axon synapses to, the amount and size of those synapses
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    syn_prob = sd_synssv.load_cached_data("syn_prob")
    syn_proba = 0.6
    m = syn_prob > syn_proba
    m_ids = sd_synssv.ids[m]
    m_axs = sd_synssv.load_cached_data("partner_axoness")[m]
    m_axs[m_axs == 3] = 1
    m_axs[m_axs == 4] = 1
    m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
    m_sizes = sd_synssv.load_cached_data("mesh_area")[m] / 2
    m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
    m_syn_sign = sd_synssv.load_cached_data("syn_sign")[m]
    ct_inds = np.any(m_cts == 2, axis=1)
    hvc_inds = np.any(m_cts == 4, axis=1)
    lman_inds = np.any(m_cts == 3, axis=1)
    m_cts = m_cts[ct_inds]
    m_sizes = m_sizes[ct_inds]
    m_ids = m_ids[ct_inds]
    m_axs = m_axs[ct_inds]
    m_ssv_hvcs = m_ssv_partners[hvc_inds]
    m_ssv_lmans = m_ssv_partners[lman_inds]
    m_ssvs = m_ssv_partners[ct_inds]
    m_syn_sign = m_syn_sign[ct_inds]
    target_ct = np.array([3, 4])
    for i, syn_id in enumerate(m_ids):
        syn_ax = m_axs[i]
        if not 1 in syn_ax:
            continue
        if syn_ax[0] == syn_ax[1]:  # no axo-axonic synapses
            continue
        if syn_ax[0] + syn_ax[1] != 1: #only axo-dendritic
            continue
        if syn_ax[0] == 1:
            ct1, ct2 = m_cts[i]
            ssv1, ssv2 = m_ssvs[i]
        else:
            ct2, ct1 = m_cts[i]
            ssv2, ssv1 = m_ssvs[i]
        if ct1 not in target_ct:
            continue
        #now there should be only axo-dendritic LMAN/HVC MSN synapses left
        if ssv2 not in full_msn_dict:
            continue
        if ct1 == 3:
            syn_amount = m_ssv_lmans[np.any(m_ssv_lmans == ssv1, axis = 1)]
        else:
            syn_amount = m_ssv_hvcs[np.any(m_ssv_hvcs == ssv1, axis = 1)]
        if len(syn_amount) < 80:
            continue
        if len(syn_amount) > 120:
            continue
        m_size = m_sizes[i]
        percentage = 100 * (i / len(m_ids))
        if percentage % 10 == 0:
            print("%.2f percent" % percentage)
        if ct1 == 3:
            lman_dict[ssv1][ssv2][0] += 1
            lman_dict[ssv1][ssv2][1] += m_size
        else:
            hvc_dict[ssv1][ssv2][0] += 1
            hvc_dict[ssv1][ssv2][1] += m_size

    syntime = time.time() - celltime
    print("%.2f min, %.2f sec for finding synapses" % (syntime // 60, syntime % 60))
    time_stamps = [time.time()]
    step_idents.append('sorting synapses')

    log.info('Step 3/4 calculate distance between MSN cells')



    #compute average distance between somas of one lman or hvc axon
    lman_dis = np.zeros(len(lman_dict))
    hvc_dis = np.zeros(len(hvc_dict))
    lman_cen = np.zeros(len(lman_dict))
    hvc_cen = np.zeros(len(hvc_dict))

    for il, lmanid in enumerate(lman_dict.keys()):
        lman = lman_dict[lmanid]
        # only compute ones with multiple msns
        if len(lman) <= 1:
            continue
        #compute distances between msn in lman
        distances = np.zeros((len(lman.keys())**2))
        coords = np.zeros((len(lman.keys()), 3))
        for i, msn_i in enumerate(lman.keys()):
            for j, msn_j in enumerate(lman.keys()):
                if i == j:
                    continue
                if i > j:
                    continue
                distances[i*len(lman.keys()) + j] = np.linalg.norm(full_msn_dict[msn_i] - full_msn_dict[msn_j])
            coords[i] = full_msn_dict[msn_i]
        centre_coord = np.sum(coords, axis = 0)/ len(coords)
        distances = distances[distances != 0]
        #compute average distance: sum/ len(lman)
        #add as new key to dict
        average_dis = np.mean(distances)
        lman["average distance"] = average_dis/1000 #in µm
        lman["center of MSN"] = centre_coord
        lman_dis[il] = average_dis/1000 #in µm
        #lman_cen[il] = centre_coord
    lman_inds = [lman_dis != 0]
    lman_dis = lman_dis[lman_inds]
    #lman_cen = lman_cen[lman_inds]



    for ih, hvcid in enumerate(hvc_dict.keys()):
        hvc = hvc_dict[hvcid]
        coords = np.zeros((len(hvc.keys()), 3))
        # only compute ones with multiple MSNs
        if len(hvc) <= 1:
            continue
        # compute distances between msn in lman
        distances = np.zeros((len(hvc.keys()) ** 2))
        for i, msn_i in enumerate(hvc.keys()):
            for j, msn_j in enumerate(hvc.keys()):
                if i == j:
                    continue
                if i > j:
                    continue
                distances[i * len(hvc.keys()) + j] = np.linalg.norm(full_msn_dict[msn_i] - full_msn_dict[msn_j])
        distances = distances[distances != 0]
        # compute average distance: sum/ len(lman)
        # add as new key to dict
        average_dis = np.mean(distances)
        hvc["average distance"] = average_dis/ 1000 #in µm
        hvc_dis[ih] = average_dis/ 1000 #in µm
    hvc_dis = hvc_dis[hvc_dis != 0]
    distime = time.time() - syntime
    print("%.2f min, %.2f sec for calculating MSN distances" % (distime // 60, distime % 60))
    time_stamps = [time.time()]
    step_idents.append('calculate MSN distances')


    log.info('Step 4/4 plot average MSM distances')

    sns.distplot(lman_dis, kde=False, bins=100)
    lman_dis_fname = ("%s/lman_dis.png" % f_name)
    plt.title('averaged distance of MSNs from same LMAN axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('count of cells')
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(hvc_dis, kde=False, bins=100)
    hvc_dis_fname = ("%s/hvc_dis.png" % f_name)
    plt.title('averaged distance of MSNs from same HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('count of cells')
    plt.savefig(hvc_dis_fname)
    plt.close()

    sns.distplot(np.log10(lman_dis*1000), kde=False)
    lman_dis_fname = ("%s/loglman_dis.png" % f_name)
    plt.title('log of averaged distance of MSNs from same LMAN axon ')
    plt.xlabel('log of distance in nm')
    plt.ylabel('count of cells')
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(np.log10(hvc_dis*1000), kde=False)
    hvc_dis_fname = ("%s/loghvc_dis.png" % f_name)
    plt.title('log of averaged distance of MSNs from same HVC axon ')
    plt.xlabel('log of distance in nm')
    plt.ylabel('count of cells')
    plt.savefig(hvc_dis_fname)
    plt.close()

    sns.distplot(lman_dis, kde=False, color="yellow", label="lman", bins = 100)
    sns.distplot(hvc_dis, kde=False, color="blue", label="hvc", bins= 100)
    lman_dis_fname = ("%s/lmanhvc_dis.png" % f_name)
    plt.title('averaged distance of MSNs from same LMAN/HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(np.log10(lman_dis*1000), kde=False, color="yellow", label="lman")
    sns.distplot(np.log10(hvc_dis*1000), kde=False, color="blue", label="hvc")
    lman_dis_fname = ("%s/loglmanhvc_dis.png" % f_name)
    plt.title('log of averaged distance of MSNs from same LMAN/HVC axon ')
    plt.xlabel('log of distance in nm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(lman_dis, kde=False, color="yellow", label="lman", bins=100, norm_hist=True)
    sns.distplot(hvc_dis, kde=False, color="blue", label="hvc", bins=100, norm_hist=True)
    lman_dis_fname = ("%s/lmanhvc_dis_norm.png" % f_name)
    plt.title('averaged distance of MSNs from same LMAN/HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('relative amount of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    plottime = time.time() - distime
    print("%.2f min, %.2f sec for loading" % (plottime // 60, plottime % 60))
    time_stamps = [time.time()]
    step_idents.append('plotting averaged distances')


    log.info('finished')

