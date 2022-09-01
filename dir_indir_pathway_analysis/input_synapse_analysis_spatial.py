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
    import matplotlib.collections as collections
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl

    #analyse the distance of neurons that receive the same inputs from LMAN, HVC using the middle of the soma skeleton nodes

    start = time.time()

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # find cells from modulatory category that are not only axons -> higher probability that its not a dopaminergic neuron; modulatory class is DA & TAN
    # celltypes j0126:STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    #celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10
    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime // 60, loadtime % 60))

    f_name = "u/arother/test_folder/201105_j0251v3_input_distance_analysis_spatial_4080"
    log = initialize_logging('LMAN_HVC distance analysis', log_dir=f_name + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #finding lman, HVC axons,msn
    log.info('Step 1/4 finding LMAN/ HVC axons, full MSNs, calculate middle of soma')
    lman_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 3]
    hvc_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 4]
    msn_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") ==2]

    #find whole MSNs and compute the medium of their MSN in a dictionary


    full_msn_dict = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_MSN_dict.pkl")


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
    same_syn_range = True
    if same_syn_range:
        lman_syn_amount = load_pkl2obj("wholebrain/scratch/arother/j0251v3_prep/ax_LMA_synam.pkl")
        hvc_syn_amount = load_pkl2obj("wholebrain/scratch/arother/j0251v3_prep/ax_HVC_synam.pkl")
    log.info('Step 2a/4 iterate over synapses for LMAN/HVC & MSN synapses')
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
        if same_syn_range:
            if ct1 == 3:
                syn_amount = lman_syn_amount[ssv1]
            else:
                syn_amount = hvc_syn_amount[ssv1]
            if syn_amount < 40:
                continue
            if syn_amount > 80:
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
    lman_cen = np.zeros((len(lman_dict), 3))
    lman_rad = np.zeros((len(lman_dict)))
    lman_maxrad = np.zeros((len(lman_dict)))
    hvc_cen = np.zeros((len(hvc_dict), 3))
    hvc_rad = np.zeros((len(hvc_dict)))
    hvc_maxrad = np.zeros((len(hvc_dict)))
    lman_amount = np.zeros(len(lman_dict))
    hvc_amount = np.zeros(len(hvc_dict))

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
        centre_coord = np.mean(coords, axis = 0)
        radii = np.linalg.norm([full_msn_dict[ix] - centre_coord for ix in lman.keys()])
        mean_radius = np.mean(radii)
        distances = distances[distances != 0]
        #compute average distance: sum/ len(lman)
        #add as new key to dict
        average_dis = np.mean(distances)
        lman["average distance"] = average_dis/1000 #in µm
        lman["center of MSN"] = centre_coord
        lman_dis[il] = average_dis/1000 #in µm
        lman_cen[il] = centre_coord/ 1000
        lman_rad[il] = mean_radius/ 1000
        lman_maxrad[il] = np.max(radii)/1000
        lman_amount[il] = len(lman.keys())
    lman_inds = [lman_dis != 0]
    lman_dis = lman_dis[lman_inds]
    lman_cen = lman_cen[lman_inds]
    lman_rad = lman_rad[lman_inds]
    lman_amount = lman_amount[lman_inds]
    lman_maxrad = lman_maxrad[lman_inds]


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
            coords[i] = full_msn_dict[msn_i]
        centre_coord = np.mean(coords, axis = 0)
        radii = np.linalg.norm([full_msn_dict[ix] - centre_coord for ix in hvc.keys()])
        mean_radius = np.mean(radii)
        distances = distances[distances != 0]
        # compute average distance: sum/ len(lman)
        # add as new key to dict
        average_dis = np.mean(distances)
        hvc["average distance"] = average_dis/ 1000 #in µm
        hvc_dis[ih] = average_dis/ 1000 #in µm
        hvc_cen[ih] = centre_coord/1000
        hvc_rad[ih] = mean_radius/1000
        hvc_maxrad[ih] = np.max(radii)/1000
        hvc_amount[ih] = len(hvc.keys())
    hvc_inds = [hvc_dis != 0]
    hvc_dis = hvc_dis[hvc_inds]
    hvc_cen = hvc_cen[hvc_inds]
    hvc_rad = hvc_rad[hvc_inds]
    hvc_maxrad = hvc_maxrad[hvc_inds]
    hvc_amount = hvc_amount[hvc_inds]
    distime = time.time() - syntime
    print("%.2f min, %.2f sec for calculating MSN distances" % (distime // 60, distime % 60))
    time_stamps = [time.time()]
    step_idents.append('calculate MSN distances')

    log.info('Step 4/4 plot average MSN distances')

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

    sns.distplot(lman_dis, kde=False, color="yellow", label="lman", bins=100, norm_hist=True)
    sns.distplot(hvc_dis, kde=False, color="blue", label="hvc", bins=100, norm_hist=True)
    lman_dis_fname = ("%s/lmanhvc_dis_norm.png" % f_name)
    plt.title('averaged distance of MSNs from same LMAN/HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('relative amount of cells')
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

    sns.distplot(np.log10(lman_dis * 1000), kde=False, color="yellow", label="lman", norm_hist=True)
    sns.distplot(np.log10(hvc_dis * 1000), kde=False, color="blue", label="hvc", norm_hist=True)
    lman_dis_fname = ("%s/loglmanhvc_dis_norm.png" % f_name)
    plt.title('log of averaged distance of MSNs from same LMAN/HVC axon ')
    plt.xlabel('log of distance in nm')
    plt.ylabel('normed count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(lman_amount, kde=False, color="yellow", label="lman")
    sns.distplot(hvc_amount, kde=False, color="blue", label="hvc")
    lman_dis_fname = ("%s/lmanhvc_amount.png" % f_name)
    plt.title('amount of MSNs from same LMAN/HVC axon ')
    plt.xlabel('number of MSNs')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(lman_amount, kde=False, color="yellow", label="lman", norm_hist=True)
    sns.distplot(hvc_amount, kde=False, color="blue", label="hvc", norm_hist=True)
    lman_dis_fname = ("%s/lmanhvc_amount_norm.png" % f_name)
    plt.title('amount of MSNs from same LMAN/HVC axon ')
    plt.xlabel('number of MSNs')
    plt.ylabel('normed count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(lman_maxrad, kde=False, color="yellow", label="lman", bins=100)
    sns.distplot(hvc_maxrad, kde=False, color="blue", label="hvc", bins=100)
    lman_dis_fname = ("%s/lmanhvc_maxrad.png" % f_name)
    plt.title('maximal radius of MSNs from same LMAN/HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    sns.distplot(lman_maxrad, kde=False, color="yellow", label="lman", bins=100, norm_hist=True)
    sns.distplot(hvc_maxrad, kde=False, color="blue", label="hvc", bins=100, norm_hist=True)
    lman_dis_fname = ("%s/lmanhvc_maxrad_norm.png" % f_name)
    plt.title('maximal radius of MSNs from same LMAN/HVC axon ')
    plt.xlabel('distance in µm')
    plt.ylabel('relative amount of cells')
    plt.legend()
    plt.savefig(lman_dis_fname)
    plt.close()


    if same_syn_range:
        lman_syn_arr = np.fromiter(lman_syn_amount.values(), dtype=np.int)
        hvc_syn_arr = np.fromiter(hvc_syn_amount.values(), dtype=np.int)

        sns.distplot(lman_syn_arr, kde=False, color="yellow", label="lman", bins=800)
        sns.distplot(hvc_syn_arr, kde=False, color="blue", label="hvc", bins=800)
        lman_dis_fname = ("%s/lmanhvc_synamount.png" % f_name)
        plt.title('amount of synapses from same LMAN/HVC axon with prob. = 0.6 ')
        plt.xlabel('number of synapses')
        plt.ylabel('count of cells')
        plt.xlim(0,200)
        plt.legend()
        plt.savefig(lman_dis_fname)
        plt.close()

        sns.distplot(lman_syn_arr, kde=False, color="yellow", label="lman",bins = 800, norm_hist=True)
        sns.distplot(hvc_syn_arr, kde=False, color="blue", label="hvc", bins= 800, norm_hist=True)
        lman_dis_fname = ("%s/lmanhvc_synamount_norm.png" % f_name)
        plt.title('amount of synapses from same LMAN/HVC axon with prob. = 0.6 ')
        plt.xlabel('number of synapses')
        plt.ylabel('normed count of cells')
        plt.xlim(0,200)
        plt.legend()
        plt.savefig(lman_dis_fname)
        plt.close()

        sns.distplot(np.log10(lman_syn_arr), kde=False, color="yellow", label="lman")
        sns.distplot(np.log10(hvc_syn_arr), kde=False, color="blue", label="hvc")
        lman_dis_fname = ("%s/loglmanhvc_synamount.png" % f_name)
        plt.title('log amount of synapses from same LMAN/HVC axon with prob. = 0.6 ')
        plt.xlabel('number of synapses in 10^')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(lman_dis_fname)
        plt.close()

        sns.distplot(np.log10(lman_syn_arr), kde=False, color="yellow", label="lman", norm_hist=True)
        sns.distplot(np.log10(hvc_syn_arr), kde=False, color="blue", label="hvc", norm_hist=True)
        lman_dis_fname = ("%s/loglmanhvc_synamount_norm.png" % f_name)
        plt.title('log amount of synapses from same LMAN/HVC axon with prob. = 0.6 ')
        plt.xlabel('number of synapses in 10^')
        plt.ylabel('normed count of cells')
        plt.legend()
        plt.savefig(lman_dis_fname)
        plt.close()

    ax_lim = 300
    ax_length = 7
    radius_red_factor = 0.1
    points_whole_ax = ax_length * 72
    lman_point_rad = 2 * (lman_rad*radius_red_factor) / ax_lim * points_whole_ax
    hvc_point_rad = 2 * (hvc_rad*radius_red_factor)/ ax_lim * points_whole_ax

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 0], lman_cen[:, 2], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/lman_xz_sc.png" % f_name)
    ax.set_title('x,z coordinates of lman spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('x coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 0], lman_cen[:, 1], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/lman_xy_sc.png" % f_name)
    ax.set_title('x,y coordinates of lman spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('x coordinate in µm')
    ax.set_ylabel('y coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 1], lman_cen[:, 2], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/lman_yz_sc.png" % f_name)
    ax.set_title('y,z coordinates of lman spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('y coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(hvc_cen[:, 0], hvc_cen[:, 2], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/hvc_xz_sc.png" % f_name)
    ax.set_title('x,z coordinates of hvc spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('x coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(hvc_cen[:, 0], hvc_cen[:, 1], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/hvc_xy_sc.png" % f_name)
    ax.set_title('x,y coordinates of hvc spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('x coordinate in µm')
    ax.set_ylabel('y coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(hvc_cen[:, 1], hvc_cen[:, 2], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black")
    lman_dis_fname = ("%s/hvc_yz_sc.png" % f_name)
    ax.set_title('y,z coordinates of hvc spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('y coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 0], lman_cen[:, 2], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black", label= "LMAN")
    lman_dis_fname = ("%s/hvclman_xz_sc.png" % f_name)
    ax.set_title('x,z coordinates of lman spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('x coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    ax.scatter(hvc_cen[:, 0], hvc_cen[:, 2], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black", label = "HVC")
    ax.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 0], lman_cen[:, 1], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black",
               label="LMAN")
    lman_dis_fname = ("%s/hvclman_xy_sc.png" % f_name)
    ax.set_title('x,y coordinates of hvc/lman spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('y coordinate in µm')
    ax.set_ylabel('y coordinate in µm')
    ax.scatter(hvc_cen[:, 0], hvc_cen[:, 1], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black",
               label="HVC")
    ax.legend()
    plt.savefig(lman_dis_fname)
    plt.close()

    plt.figure(figsize=[ax_length, ax_length])
    ax = plt.axes(xlim=(0, ax_lim), ylim=(0, ax_lim))
    ax.scatter(lman_cen[:, 1], lman_cen[:, 2], c="skyblue", s=lman_point_rad ** 2, alpha=0.3, edgecolors="black",
               label="LMAN")
    lman_dis_fname = ("%s/hvclman_yz_sc.png" % f_name)
    ax.set_title('y,z coordinates of lman/hvc spatial field with %.1f * average radius' % radius_red_factor)
    ax.set_xlabel('y coordinate in µm')
    ax.set_ylabel('z coordinate in µm')
    ax.scatter(hvc_cen[:, 1], hvc_cen[:, 2], c="mediumpurple", s=hvc_point_rad ** 2, alpha=0.3, edgecolors="black",
               label="HVC")
    ax.legend()
    plt.savefig(lman_dis_fname)
    plt.close()



    plottime = time.time() - distime
    print("%.2f min, %.2f sec for loading" % (plottime // 60, plottime % 60))
    time_stamps = [time.time()]
    step_idents.append('plotting averaged distances')


    log.info('finished')