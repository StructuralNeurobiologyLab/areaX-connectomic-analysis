if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import time
    from syconn.handler.config import initialize_logging
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl

    start = time.time()

    #global_params.wd = "ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # find cells from modulatory category that are not only axons -> higher probability that its not a dopaminergic neuron; modulatory class is DA & TAN
    #celltypes: j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    # celltypes: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime//60, loadtime%60))

    f_name = "u/arother/test_folder/201105_j0251_v3_tan_analysis_0.1"
    log = initialize_logging('tan analysis', log_dir=f_name + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info('Step 1/2 Finding TANs')
    mod_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 5]
    tan_ids = np.zeros(len(mod_ids)).astype(int)
    """
    for i, cell in enumerate(ssd.get_super_segmentation_object(mod_ids)):
        cell.load_skeleton()
        axoness = cell.skeleton["axoness_avg10000"]
        axoness[axoness == 3] = 1
        axoness[axoness == 4] = 1
        unique_preds = np.unique(axoness)
        if not (0 in unique_preds and 1 in unique_preds and 2 in unique_preds):
            continue
        tan_ids[i] = cell.id
        percentage = 100 * (i / len(mod_ids))
        print("%.2f percent" % percentage)

    tan_ids = tan_ids[tan_ids != 0]
    """
    tan_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_TAN_arr.pkl")
    celltime = time.time() - loadtime
    print("%.2f min, %.2f sec for finding TANS" % (celltime//60, celltime%60))
    time_stamps.append(time.time())
    step_idents.append('finding TANs')

    da_syn_amount = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/ax_DA_synam.pkl")
    hvc_syn_amount = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/ax_HVC_synam.pkl")
    lman_syn_amount = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/ax_LMA_synam.pkl")

    log.info('Step 2/2 analyse synapse properties of found TANS')
    #look at synapses to see input and output cells, compartments
    compartments = ["dendrite", "soma"]
    compartments_ct = ["soma", "dendritic shaft", "spine neck", "spine head"]
    celltypes = ["STN", "DA", "MSN", "LMAN", "HVC", "TAN", "GPe", "GPi", "FS", "LTS", "NGF"]
    signs = ["exc", "inh"]
    # partner_axoness: 0 = dendrite, 1 = axon, 2 = soma, 3 = en-passant bouton, 4 = terminal bouton
    #partner spiness: 0: dendritic shaft, 1: spine head, 2: spine neck
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
    m_spiness = sd_synssv.load_cached_data("partner_spiness")[m]
    ct_inds = np.any(m_cts == 5, axis = 1)
    m_cts = m_cts[ct_inds]
    m_sizes = m_sizes[ct_inds]
    m_ids = m_ids[ct_inds]
    m_axs = m_axs[ct_inds]
    m_ssv_partners = m_ssv_partners[ct_inds]
    m_syn_sign = m_syn_sign[ct_inds]
    m_spiness = m_spiness[ct_inds]
    output_celltypes_count = np.zeros((len(celltypes)))
    output_celltypes_size = np.zeros((len(celltypes)))
    input_celltypes_count = np.zeros((len(celltypes)))
    input_celltypes_size = np.zeros((len(celltypes)))
    #amount and size of axo-dendritic [0] or axo-somatic[1] synapses
    #from axoness: 2 = soma, from spiness: 0 = dendritic shaft, 1 = spine head, 2 = spine neck -> here: 0 = soma, 1 = dendritic shaft, 2 = spine head, 3 = spine_neck
    output_compartment_count = np.zeros((len(compartments)))
    output_compartment_size = np.zeros((len(compartments)))
    input_compartment_count = np.zeros((len(compartments)))
    input_compartment_size = np.zeros((len(compartments)))
    output_compct_count = np.zeros((len(celltypes), len(compartments_ct)))
    output_compct_size = np.zeros((len(celltypes), len(compartments_ct)))
    input_compct_count = np.zeros((len(celltypes), len(compartments_ct)))
    input_compct_size = np.zeros((len(celltypes), len(compartments_ct)))
    output_synsign_count = np.zeros((len(celltypes), len(signs)))
    output_synsign_size = np.zeros((len(celltypes), len(signs)))
    #iterate over synapses
    for ix, id in enumerate(m_ids):
        syn_ax = m_axs[ix]
        if not 1 in syn_ax:
            continue
        if syn_ax[0] == syn_ax[1]: #no axo-axonic synapses
            continue
        if not np.any(np.in1d(m_ssv_partners[ix], tan_ids)):
            continue
        m_size = m_sizes[ix]
        if m_size < 0.1:
            continue
        if syn_ax[0] == 1:
            ct1, ct2 = m_cts[ix]
            ssv1, ssv2 = m_ssv_partners[ix]
            spin1, spin2 = m_spiness[ix]
        elif syn_ax[1] == 1:
            ct2, ct1 = m_cts[ix]
            ssv2, ssv1 = m_ssv_partners[ix]
            spin2, spin1 = m_spiness[ix]
        else:
            raise ValueError("unknown axoness values")
        if (0 in syn_ax) and (spin2 == 3):#if spiness of dendrite is 3, it counts as dendritic shaft (hierarchy = axoness, then spiness)
            spin2 = 0
        synsign = m_syn_sign[ix]
        percentage = 100 * (ix / len(m_ids))
        print("%.2f percent" % percentage)
        if ssv1 in tan_ids:
            output_celltypes_count[ct2] += 1
            output_celltypes_size[ct2] += m_size
            if 0 in syn_ax:
                output_compartment_count[0] += 1
                output_compartment_size[0] += m_size
                output_compct_count[ct2][0] += 1
                output_compct_size[ct2][0] += m_size
                output_compct_count[ct2][spin2 + 1] += 1
                output_compct_size[ct2][spin2 + 1] += m_size
            else:
                output_compartment_count[1] += 1
                output_compartment_size[1] += m_size
                output_compct_count[ct2][0] += 1
                output_compct_size[ct2][0] += m_size
            if synsign == 1:
                output_synsign_count[ct2][0] += 1
                output_synsign_size[ct2][0] += m_size
            elif synsign == -1:
                output_synsign_count[ct2][1] += 1
                output_synsign_size[ct2][1] += m_size
        else:
            input_celltypes_count[ct1] += 1
            input_celltypes_size[ct1] += m_size
            if 0 in syn_ax:
                input_compartment_count[0] += 1
                input_compartment_size[0] += m_size
                input_compct_count[ct1][spin2 + 1] += 1
                input_compct_size[ct1][spin2 + 1] += m_size
            else:
                input_compartment_count[1] += 1
                input_compartment_size[1] += 1
                input_compct_count[ct1][0] += 1
                input_compct_size[ct1][0] += m_size

    syntime = time.time() - celltime
    print("%.2f min, %.2f sec for synapse iteration" % (syntime//60, syntime%60))
    time_stamps.append(time.time())
    step_idents.append('synapse analysis')
    comp_dict = {1: 'axons', 0: 'dendrites'}
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
    comp_ticks = np.array([0, 1])
    cell_ticks = np.arange(0, 11)

    compct_columns = ["count", "size", "celltype", "compartment"]
    signct_columns = ["count", "size", "celltype", "sign"]
    indexct = np.arange(0, len(celltypes)*len(compartments_ct))
    indexsign = np.arange(0, len(celltypes)* len(signs))
    pd_outputcomp = pd.DataFrame(pd.np.empty((len(compartments_ct)*len(celltypes), len(compct_columns))), columns=compct_columns, index= indexct)
    for i, ct in enumerate(output_compct_count):
        for ic, val in enumerate(ct):
            ix = i*len(compartments_ct) + ic
            pd_outputcomp.iloc[ix, 2] = ct_dict[i]
            pd_outputcomp.iloc[ix, 3] = compartments_ct[ic]
            pd_outputcomp.iloc[ix, 1] = val
            pd_outputcomp.iloc[ix, 0] = output_compct_size[i][ic]
    pd_outputsign = pd.DataFrame(pd.np.empty((len(signs) * len(celltypes), len(signct_columns))),columns=signct_columns, index = indexsign)
    for i, ct in enumerate(output_synsign_count):
        for ic, val in enumerate(ct):
            ix = i * len(signs) + ic
            pd_outputsign.iloc[ix, 2] = ct_dict[i]
            pd_outputsign.iloc[ix, 3] = signs[ic]
            pd_outputsign.iloc[ix, 1] = val
            pd_outputsign.iloc[ix, 0] = output_synsign_size[i][ic]
    pd_input = pd.DataFrame(pd.np.empty((len(compartments_ct)*len(celltypes), len(compct_columns))), columns=compct_columns, index = indexct)
    for i, ct in enumerate(input_compct_count):
        for ic, val in enumerate(ct):
            ix = i*len(compartments_ct) + ic
            pd_input.iloc[ix, 2] = ct_dict[i]
            pd_input.iloc[ix, 3] = compartments_ct[ic]
            pd_input.iloc[ix, 1] = val
            pd_input.iloc[ix, 0] = input_compct_size[i][ic]

    pd_outputcomp = pd_outputcomp[pd_outputcomp != 0].dropna()
    pd_outputsign = pd_outputsign[pd_outputsign != 0].dropna()
    pd_input = pd_input[pd_input != 0].dropna()


    sns.barplot(x = celltypes, y=output_celltypes_count, color="skyblue")
    plt.title('TAN output celltypes')
    plt.xlabel('number of output synapses per celltype')
    plt.ylabel('count of cells')
    plt.savefig('%s/out_cts_count.png' % (f_name))
    plt.xticks(cell_ticks, celltypes)
    plt.close()
    sns.barplot(x= celltypes, y=output_celltypes_size, color="skyblue")
    plt.title('TAN output celltypes')
    plt.xlabel('sum of synapse size in output synapses per celltype')
    plt.ylabel('sum of synapse size')
    plt.savefig('%s/out_cts_size.png' % (f_name))
    plt.xticks(cell_ticks, celltypes)
    plt.close()
    sns.barplot(x = compartments, y=output_compartment_count)
    plt.title('TAN output compartments')
    plt.xlabel('number of output synapses per compartment')
    plt.ylabel('count of cells')
    plt.savefig('%s/out_comp_count.png' % (f_name))
    plt.xticks(comp_ticks, compartments)
    plt.close()
    sns.barplot(x = compartments, y=output_compartment_size)
    plt.title('TAN output compartments')
    plt.xlabel('sum of synapse size in output synapses per compartment')
    plt.ylabel('sum of synapse size')
    plt.savefig('%s/out_comp_size.png' % (f_name))
    plt.xticks(comp_ticks, compartments)
    plt.close()
    sns.barplot(x=celltypes, y=input_celltypes_count, color="skyblue")
    plt.title('TAN input celltypes')
    plt.xlabel('number of input synapses per celltype')
    plt.ylabel('count of cells')
    plt.savefig('%s/in_cts_count.png' % (f_name))
    plt.xticks(cell_ticks, celltypes)
    plt.close()
    sns.barplot(x=celltypes, y=input_celltypes_size, color="skyblue")
    plt.title('TAN input celltypes')
    plt.xlabel('sum of synapse size in input synapses per celltype')
    plt.ylabel('sum of synapse size')
    plt.savefig('%s/in_cts_size.png' % (f_name))
    plt.xticks(cell_ticks, celltypes)
    plt.close()
    sns.barplot(x=compartments, y=input_compartment_count)
    plt.title('TAN input compartment')
    plt.xlabel('number of input synapses per compartment')
    plt.ylabel('count of cells')
    plt.savefig('%s/in_comp_count.png' % (f_name))
    plt.xticks(comp_ticks, compartments)
    plt.close()
    sns.barplot(x=compartments, y=input_compartment_size)
    plt.title('TAN input compartment')
    plt.xlabel('sum of synapse size in input synapses per compartment')
    plt.ylabel('sum of synapse size')
    plt.savefig('%s/in_comp_size.png' % (f_name))
    plt.xticks(comp_ticks, compartments)
    plt.close()

    sns.barplot(data=pd_outputcomp, x="celltype", y="count", hue="compartment")
    plt.title('TAN output compartment per celltype')
    plt.ylabel('count of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/out_comp_countct.png' % (f_name))
    plt.close()

    sns.barplot(data=pd_outputcomp, x="celltype", y="size", hue="compartment")
    plt.title('TAN output compartment per celltype')
    plt.ylabel('sum of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/out_comp_sizect.png' % (f_name))
    plt.xticks(comp_ticks, celltypes)
    plt.close()

    sns.barplot(data=pd_outputsign, x="celltype", y="count", hue="sign")
    plt.title('TAN output compartment per celltype')
    plt.ylabel('count of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/out_sign_countct.png' % (f_name))
    plt.close()

    sns.barplot(data=pd_outputsign, x="celltype", y="size", hue="sign")
    plt.title('TAN output sign per celltype')
    plt.ylabel('sum of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/out_sign_sizect.png' % (f_name))
    plt.xticks(comp_ticks, celltypes)
    plt.close()

    sns.barplot(data=pd_input, x="celltype", y="count", hue="compartment")
    plt.title('TAN input compartment per celltype')
    plt.ylabel('count of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/in_comp_countct.png' % (f_name))
    plt.close()

    sns.barplot(data=pd_input, x="celltype", y="size", hue="compartment")
    plt.title('TAN input compartment per celltype')
    plt.ylabel('sum of synapses in compartment per celltype')
    plt.xlabel('celltypes')
    plt.savefig('%s/in_comp_sizect.png' % (f_name))
    plt.xticks(comp_ticks, celltypes)
    plt.close()



    raise ValueError