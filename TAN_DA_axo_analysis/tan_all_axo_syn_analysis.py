if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import os as os
    import scipy
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl

    start = time.time()
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)

    # analysis to see spiness and their connection to GP via iteration trough synapses

    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime // 60, loadtime % 60))

    f_name = "u/arother/test_folder/210401_tan_da_syn_all_allaxons"
    log = initialize_logging('TAN DA axonic synapses', log_dir=f_name + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info('Step 1/4 load full cells')

    tan_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_TAN_handpicked.pkl")
    TAN_soma_centres = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_TAN_dict.pkl")
    da_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 1]
    stn_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 0]
    lman_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 3]
    hvc_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 4]
    axo_ids = np.hstack([da_ids, stn_ids, lman_ids, hvc_ids])

    log.info('Step 2/4 iterate trough synapses of TAN and axonal fragments')
    celltype = 5
    axo_cts = np.array([0, 1, 3, 4])
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
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
    m_rep_coord = sd_synssv.load_cached_data("rep_coord")[m]
    # make dictionary with da_ids as keys and tan_ids_amount and size
    #da_ax = {daid: {tanid: {"syn_amount": 0, "syn_size": 0} for tanid in tan_ids} for daid in da_ids}
    # make dictionary with da_ids as keys and tan_ids_amount and size
    #tan_dict = {tanid: {daid: {"syn_amount": 0, "syn_size": 0} for daid in da_ids} for tanid in tan_ids}
    #make dictionary with axon ids, synapse ids and locations
    axo_syn_dict = {axoid: {"syn_id": np.zeros(100), "syn_loc": np.zeros((100, 3)), "AIS": np.zeros(100), "axo_ct": np.zeros(100)} for axoid in axo_ids}
    syn_locations = np.zeros((len(m_ids), 3))
    syn_td_ids = np.zeros(len(m_ids))
    axo_synamount = np.zeros(len(axo_ids))
    axo_synsize = np.zeros(len(axo_ids))
    axo_nodeamount = np.zeros(len(axo_ids))
    tan_synamount = np.zeros(len(tan_ids))
    tan_synsize = np.zeros(len(tan_ids))
    int_percentage = 0
    axo_axo_synapses = 0
    axo_noskel_ids = np.zeros(len(axo_ids))
    #TO DO: if too close to soma: classify as AIS
    # either use shortestpathtosoma (needs to load skeleton) or load soma center and go from there
    for ix, id in enumerate(m_ids):
        syn_ax = m_axs[ix]
        if not 1 in syn_ax:
            continue
        if syn_ax[0] != syn_ax[1]:  #only axo-axonic synapses
            continue
        if not np.any(np.in1d(m_ssv_partners[ix], tan_ids)):
            continue
        if not np.any(np.in1d(m_cts[ix], axo_cts)):
            continue
        m_size = m_sizes[ix]
        if m_size < 0.1:
            continue
        percentage = 100 * (ix / len(m_ids))
        if int(percentage) != int_percentage:
            print("%.2f percent" % percentage)
            int_percentage = int(percentage)
        axo_ind = np.where(m_cts[ix] != celltype)
        axo_id = int(m_ssv_partners[ix][axo_ind])
        axo_ids_ind = np.where(axo_ids == axo_id)
        # filter da cells axons that are only fragments, so >= 10 nodes
        #da_ax = ssd.get_super_segmentation_object(da_id)
        #da_axo_synapses += 1
        #try:
        #    da_ax.load_skeleton()
        #except AttributeError:
        #    da_noskel_ids[da_ids_ind] = da_id
        #    continue
        #if len(da_ax.skeleton["nodes"]) < 10:
        #    continue
        syn_td_ids[ix] = id
        syncoord = m_rep_coord[ix]
        syn_locations[ix] = syncoord
        axo_synamount[axo_ids_ind] += 1
        axo_synsize[axo_ids_ind] += m_size
        #da_nodeamount[da_ids_ind] = len(da_ax.skeleton["nodes"])
        tan_ind = np.where(m_cts[ix] == celltype)
        tan_id = int(m_ssv_partners[ix][tan_ind])
        tan_ids_ind = np.where(tan_ids == tan_id)
        tan_synamount[tan_ids_ind] += 1
        tan_synsize[tan_ids_ind] += m_size
        axo_amount_ind = int(axo_synamount[axo_ids_ind]) - 1
        axo_syn_dict[axo_id]["syn_id"][axo_amount_ind] = id
        axo_syn_dict[axo_id]["syn_loc"][axo_amount_ind] = syncoord
        axo_syn_dict[axo_id]["axo_ct"][axo_amount_ind] = m_cts[ix][axo_ind]
        #check if potential AIS: if closer than 100 µm to soma centre
        TAN_soma_loc = TAN_soma_centres[tan_id]
        dist2soma = np.linalg.norm(TAN_soma_loc - syncoord*ssd.scaling) / 1000 #in µm
        if dist2soma < 100:
            axo_syn_dict[axo_id]["AIS"][axo_amount_ind] = 1


    t_inds = tan_synamount > 0
    d_inds = axo_synamount > 0
    s_inds = syn_td_ids > 0

    syn_locations = syn_locations[s_inds]
    tan_synamount = tan_synamount[t_inds]
    tan_synsize = tan_synsize[t_inds]
    axo_synamount = axo_synamount[d_inds]
    axo_synsize = axo_synsize[d_inds]
    axo_idssyn = axo_ids[d_inds]
    max_synamount = int(axo_synamount.max())
    # save ids, coordinates from DA axons with 3 or more synapses
    #da_multiple_synapses = np.zeros((len(np.where(da_synamount >= max_synamount - 2)[0])*5, 7)) #for only multiple synapses
    axo_multiple_synapses = np.zeros((len(axo_synamount) * 5, 8))
    prev_indexlength = 0
    for i in range(max_synamount):
        axo_i_multiple = axo_idssyn[np.where(axo_synamount == max_synamount - i)]
        curr_indexlength = len(axo_i_multiple)*(max_synamount-i) + prev_indexlength
        axo_multiple_synapses[prev_indexlength:curr_indexlength,0] = max_synamount - i
        for in_ax_ind, in_axon_id in enumerate(axo_i_multiple):
            axo_start_index = int(prev_indexlength + in_ax_ind*(max_synamount - i))
            axo_end_index = int(axo_start_index + max_synamount - i)
            syn_axo_ids = axo_syn_dict[in_axon_id]["syn_id"]  > 0
            in_ax_entry = axo_syn_dict[in_axon_id]
            in_ax_entry["syn_id"] = in_ax_entry["syn_id"][syn_axo_ids]
            in_ax_entry["syn_loc"] = in_ax_entry["syn_loc"][syn_axo_ids]
            in_ax_entry["AIS"] = in_ax_entry["AIS"][syn_axo_ids]
            in_ax_entry["axo_ct"] = in_ax_entry["axo_ct"][syn_axo_ids]
            axo_multiple_synapses[axo_start_index: axo_end_index,1] = in_axon_id
            axo_multiple_synapses[axo_start_index: axo_end_index,2] = in_ax_entry["syn_id"]
            axo_multiple_synapses[axo_start_index: axo_end_index,3] = in_ax_entry["syn_loc"][:, 0]
            axo_multiple_synapses[axo_start_index: axo_end_index,4] = in_ax_entry["syn_loc"][:, 1]
            axo_multiple_synapses[axo_start_index: axo_end_index,5] = in_ax_entry["syn_loc"][:,2]
            axo_multiple_synapses[axo_start_index: axo_end_index,6] = in_ax_entry["AIS"]
            axo_multiple_synapses[axo_start_index: axo_end_index, 7] = in_ax_entry["axo_ct"]
        prev_indexlength = axo_end_index


    time_stamps = [time.time()]
    step_idents = ['t-1']
    syntime = time.time() - loadtime
    print("%.2f min, %.2f sec for iterating over synapses" % (syntime // 60, syntime % 60))

    log.info('Step 3/4 plot write information about multiple synapses to same DA axon to table')

    axo_multiple_synapses = axo_multiple_synapses.astype(int)
    axo_multiple_inds = axo_multiple_synapses[:,1] > 0
    axo_multiple_synapses = axo_multiple_synapses[axo_multiple_inds]
    multiple_axo_pd = pd.DataFrame(data = axo_multiple_synapses, columns=["synapse amount", "DA axon ID", "synapse ID", "loc x", "loc y", "loc z", "AIS", "incoming axon ct"])
    multiple_axo_pd.to_csv("%s/multiple_axo_axo_synapses.csv" % f_name)

    time_stamps = [time.time()]
    step_idents = ['t-2']
    tabletime = time.time() - syntime
    print("%.2f min, %.2f sec for creating table" % (tabletime // 60, tabletime % 60))

    log.info('Step 4/4 plot synapse info about axonic synapses')

    #print("Of %.i suitable synapses, only %.i and %.2f % were on DA axons >= 50 nodes" % (da_axo_synapses, len(da_synamount), (len(da_synamount)/da_axo_synapses)))
    #print("Of %.i suitable synapses, %.i and %.2f % were excluded because of missing skeletons" % (da_axo_synapses, len(da_noskel_ids), (len(da_noskel_ids) / da_axo_synapses)))

    sns.distplot(axo_synamount, kde=False, color="skyblue")
    plt.title('amount of TAN axonic synapses with incoming axons')
    plt.xlabel('amount of synapses')
    plt.ylabel('count of cells')
    plt.savefig('%s/synam_axo.png' % (f_name))
    plt.close()

    sns.distplot(axo_synsize, kde=False, color="skyblue")
    plt.title('amount of TAN DA axonic synapses with incoming axons')
    plt.xlabel('sum of synapse size')
    plt.ylabel('count of cells')
    plt.savefig('%s/synsize_axo.png' % (f_name))
    plt.close()

    #sns.distplot(axo_nodeamount, kde=False, color="skyblue")
    #plt.title('skeleton length in axons >= 10 nodes in with incoming axons')
    #plt.xlabel('amount of nodes')
    #plt.ylabel('count of cells')
    #plt.savefig('%s/nodeam_axo.png' % (f_name))
    #plt.close()

    sns.distplot(tan_synamount, kde=False, color="skyblue")
    plt.title('amount of TAN DA axonic synapses %s' % ct_dict[celltype])
    plt.xlabel('amount of synapses')
    plt.ylabel('count of cells')
    plt.savefig('%s/synam_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(tan_synsize, kde=False, color="skyblue")
    plt.title('amount of TAN DA axonic synapses %s' % ct_dict[celltype])
    plt.xlabel('sum of synapse size')
    plt.ylabel('count of cells')
    plt.savefig('%s/synsize_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    """
    plt.scatter(x=da_nodeamount, y=da_synamount, c="skyblue")
    filename = ('%s/scdanode_daam_%s.png' % (f_name, ct_dict[celltype1]))
    plt.title('amount of TAN DA axonic synapses and skeleton length in %.4s' % ct_dict[celltype1])
    plt.xlabel('amount of skeleton nodes')
    plt.ylabel('amount of synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=da_nodeamount, y=da_synsize, c="skyblue")
    filename = ('%s/scdanode_daam_%s.png' % (f_name, ct_dict[celltype1]))
    plt.title('sum of synapses and skeleton length in %.4s' % ct_dict[celltype1])
    plt.xlabel('amount of skeleton nodes')
    plt.ylabel('sum of synapses')
    plt.savefig(filename)
    plt.close()
    """

    time_stamps = [time.time()]
    step_idents = ['t-3']
    plottime = time.time() - tabletime
    print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

    log.info('TAN DA axonic synapses')

    raise ValueError