if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    import os as os
    import scipy
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from collections import defaultdict

    start = time.time()
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)

    #analysis to see spiness and their connection to GP via iteration trough synapses

    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime // 60, loadtime % 60))

    f_name = "u/arother/test_folder/210502_j0251_v3_MSNGPsyn_fullGPhandpicked"
    log = initialize_logging('MSN GP synapse analysis', log_dir=f_name + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info('Step 1/5 load full cells')

    msn_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_MSN_arr.pkl")
    gpe_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPe_arr.pkl")
    gpi_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPi_arr.pkl")
    #gpe_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_GPe_arr.pkl")
    #gpi_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_GPi_arr.pkl")
    #gpe_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 6]
    #gpi_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == 7]
    gp_ids = np.hstack([gpe_ids, gpi_ids])

    log.info('Step 2/5 iterate trough synapses of MSN')
    celltype = 2
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
    m_spiness = sd_synssv.load_cached_data("partner_spiness")[m]
    ct_inds = np.any(m_cts == celltype, axis=1)
    m_cts = m_cts[ct_inds]
    m_sizes = m_sizes[ct_inds]
    m_ids = m_ids[ct_inds]
    m_axs = m_axs[ct_inds]
    m_ssv_partners = m_ssv_partners[ct_inds]
    m_syn_sign = m_syn_sign[ct_inds]
    m_spiness = m_spiness[ct_inds]
    # make dictionary with ratio of GPi to GPe
    msn_am_dict = {cellid: {"GPe_amount": 0, "GPe_size": 0, "GPi_amount": 0, "GPi_size": 0, "GP_amount":0, "GP_size": 0, "GP_ids":[], "GPe_ids":[], "GPi_ids":[]} for cellid in msn_ids}
    msn_gp_dict = {cellid: {gpid: {"syn_amount":0, "syn_size":0} for gpid in gp_ids} for cellid in msn_ids}
    #msn_gp_dict = {cellid: defaultdict(lambda: {"syn_amount":0, "syn_size":0}) for cellid in msn_ids}
    gp_dict =  {gpid: [] for gpid in gp_ids}
    int_percentage = 0
    # partner_axoness: 0 = dendrite, 1 = axon, 2 = soma, 3 = en-passant bouton, 4 = terminal bouton
    # partner spiness: 0: dendritic shaft, 1: spine head, 2: spine neck
    for ix, id in enumerate(m_ids):
        syn_ax = m_axs[ix]
        if not 1 in syn_ax:
            continue
        if syn_ax[0] == syn_ax[1]:  # no axo-axonic synapses
            continue
        if not np.any(np.in1d(m_ssv_partners[ix], msn_ids)):
            continue
        m_size = m_sizes[ix]
        if m_size < 0.1:
            continue
        percentage = 100 * (ix / len(m_ids))
        if int(percentage) != int_percentage:
            print("%.2f percent" % percentage)
            int_percentage = int(percentage)
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
        if ssv1 not in msn_ids:
            continue
        if ssv2 not in gp_ids:
            continue
        msn_am_dict[ssv1]["GP_amount"] += 1
        msn_am_dict[ssv1]["GP_size"] += m_size
        msn_gp_dict[ssv1][ssv2]["syn_amount"] += 1
        msn_gp_dict[ssv1][ssv2]["syn_size"] += m_size
        msn_am_dict[ssv1]["GP_ids"].append(ssv2)
        gp_dict[ssv2].append(ssv1)
        if ssv2 in gpe_ids:
            msn_am_dict[ssv1]["GPe_amount"] += 1
            msn_am_dict[ssv1]["GPe_size"] += m_size
            msn_am_dict[ssv1]["GPe_ids"].append(ssv2)
        elif ssv2 in gpi_ids:
            msn_am_dict[ssv1]["GPi_amount"] += 1
            msn_am_dict[ssv1]["GPi_size"] += m_size
            msn_am_dict[ssv1]["GPi_ids"].append(ssv2)

    time_stamps = [time.time()]
    step_idents = ['t-1']
    syntime = time.time() - loadtime
    print("%.2f min, %.2f sec for iterating over synapses" % (syntime // 60, syntime % 60))

    log.info('Step 3/5 iterate over MSN cells')
    msn_nogp = np.zeros(len(msn_ids))
    msn_nogpi = np.zeros(len(msn_ids))
    msn_nogpe = np.zeros(len(msn_ids))
    msn_gp_synamount = np.zeros(len(msn_ids))
    msn_gpe_synamount = np.zeros(len(msn_ids))
    msn_gpi_synamount = np.zeros(len(msn_ids))
    msn_gp_synsize = np.zeros(len(msn_ids))
    msn_gpe_synsize = np.zeros(len(msn_ids))
    msn_gpi_synsize = np.zeros(len(msn_ids))
    msn_multigp = np.zeros(len(msn_ids))
    msn_multigpe = np.zeros(len(msn_ids))
    msn_multigpi = np.zeros(len(msn_ids))
    msn_samegp_syn_amount = np.zeros(len(msn_ids))
    msn_samegp_syn_size = np.zeros(len(msn_ids))
    msn_samegpe_syn_amount = np.zeros(len(msn_ids))
    msn_samegpe_syn_size = np.zeros(len(msn_ids))
    msn_samegpi_syn_amount = np.zeros(len(msn_ids))
    msn_samegpi_syn_size = np.zeros(len(msn_ids))
    for ix, msnid in enumerate(msn_ids):
        if msn_am_dict[msnid]["GP_amount"] == 0:
            msn_nogp[ix] = msnid
            msn_nogpi[ix] = msnid
            msn_nogpe[ix] = msnid
        else:
            msn_gp_synamount[ix] = msn_am_dict[msnid]["GP_amount"]
            msn_gp_synsize[ix] = msn_am_dict[msnid]["GP_size"]
            msngp_ids = msn_am_dict[msnid]["GP_ids"]
            msn_multigp[ix] = len(msngp_ids)
            msn_samegp_synam = np.zeros(len(msngp_ids))
            msn_samegp_synsi = np.zeros(len(msngp_ids))
            for gi, gpid in enumerate(msngp_ids):
                msn_samegp_synam[gi] = msn_gp_dict[msnid][gpid]["syn_amount"]
                msn_samegp_synsi[gi] = msn_gp_dict[msnid][gpid]["syn_size"]
            msn_samegp_syn_amount[ix] = np.mean(msn_samegp_synam)
            msn_samegp_syn_size[ix] = np.mean(msn_samegp_synsi)
            if msn_am_dict[msnid]["GPe_amount"] != 0:
                msn_gpe_synamount[ix] = msn_am_dict[msnid]["GPe_amount"]
                msn_gpe_synsize[ix] = msn_am_dict[msnid]["GPe_size"]
                msngpe_ids = msn_am_dict[msnid]["GPe_ids"]
                msn_multigpe[ix] = len(msngpe_ids)
                msn_samegpe_synam = np.zeros(len(msngpe_ids))
                msn_samegpe_synsi = np.zeros(len(msngpe_ids))
                for gi, gpeid in enumerate(msngpe_ids):
                    msn_samegpe_synam[gi] = msn_gp_dict[msnid][gpeid]["syn_amount"]
                    msn_samegpe_synsi[gi] = msn_gp_dict[msnid][gpeid]["syn_size"]
                msn_samegpe_syn_amount[ix] = np.mean(msn_samegpe_synam)
                msn_samegpe_syn_size[ix] = np.mean(msn_samegpe_synsi)
            else:
                msn_nogpe[ix] = msnid
            if msn_am_dict[msnid]["GPi_amount"] != 0:
                msn_gpi_synamount[ix] = msn_am_dict[msnid]["GPi_amount"]
                msn_gpi_synsize[ix] = msn_am_dict[msnid]["GPi_size"]
                msn_multigp[ix] = len(msn_am_dict[msnid]["GPi_ids"])
                msngpi_ids = msn_am_dict[msnid]["GPi_ids"]
                msn_multigpi[ix] = len(msngpi_ids)
                msn_samegpi_synam = np.zeros(len(msngpi_ids))
                msn_samegpi_synsi = np.zeros(len(msngpi_ids))
                for gi, gpiid in enumerate(msngpi_ids):
                    msn_samegpi_synam[gi] = msn_gp_dict[msnid][gpiid]["syn_amount"]
                    msn_samegpi_synsi[gi] = msn_gp_dict[msnid][gpiid]["syn_size"]
                msn_samegpi_syn_amount[ix] = np.mean(msn_samegpi_synam)
                msn_samegpi_syn_size[ix] = np.mean(msn_samegpi_synsi)
            else:
                msn_nogpi[ix] = msnid
        percentage = 100 * (ix / len(msn_ids))
        print("%.2f percent" % percentage)

    msn_nogp = msn_nogp[msn_nogp > 0]
    msn_nogpe = msn_nogpe[msn_nogpe > 0]
    msn_nogpi = msn_nogpi[msn_nogpi > 0]
    msn_samegp_syn_amount = msn_samegp_syn_amount[msn_samegp_syn_amount > 0]
    msn_samegp_syn_size = msn_samegp_syn_size[msn_samegp_syn_size > 0]
    msn_multigp = msn_multigp[msn_multigp > 0]
    msn_samegpe_syn_amount = msn_samegpe_syn_amount[msn_samegpe_syn_amount > 0]
    msn_samegpe_syn_size = msn_samegpe_syn_size[msn_samegpe_syn_size > 0]
    msn_multigpe = msn_multigpe[msn_multigpe > 0]
    msn_samegpi_syn_amount = msn_samegpi_syn_amount[msn_samegpi_syn_amount > 0]
    msn_samegpi_syn_size = msn_samegpi_syn_size[msn_samegpi_syn_size > 0]
    msn_multigpi = msn_multigpi[msn_multigpi > 0]

    time_stamps = [time.time()]
    step_idents = ['t-3']
    celltime = time.time() - syntime
    print("%.2f min, %.2f sec for iterating over MSNs" % (celltime // 60, celltime % 60))

    log.info('Step 4/5 iterate over GP cells')
    gp_multiplemsn = np.zeros(len(gp_ids))
    gp_nomsn = np.zeros(len(gp_ids))
    for ip, gpid in enumerate(gp_ids):
        if len(gp_dict[gpid]) == 0:
            gp_nomsn[ip] = gpid
        else:
            gp_multiple = np.array(gp_dict[gpid])
            gp_multiplemsn[ip] = len(np.unique(gp_multiple))

    gpe_inds = np.in1d(gp_ids, gpe_ids)
    gpi_inds = np.in1d(gp_ids, gpi_ids)
    gpe_multiple = gp_multiplemsn[gpe_inds]
    gpe_nomsn = gp_nomsn[gpe_inds]
    gpi_multiple = gp_multiplemsn[gpi_inds]
    gpi_nomsn = gp_nomsn[gpi_inds]
    gp_nomsn = gp_nomsn[gp_nomsn > 0]
    gp_multiplemsn = gp_multiplemsn[gp_multiplemsn > 0]
    gpe_nomsn = gpe_nomsn[gpe_nomsn > 0]
    gpe_multiple = gpe_multiple[gpe_multiple > 0]
    gpi_nomsn = gpi_nomsn[gpi_nomsn > 0]
    gpi_multiple = gpi_multiple[gpi_multiple > 0]

    time_stamps = [time.time()]
    step_idents = ['t-4']
    cellgptime = time.time() - celltime
    print("%.2f min, %.2f sec for iterating over GPs" % (cellgptime // 60, cellgptime % 60))

    log.info('Step 5/5 plotting')

    #print percentage of msn without GP synapse, GP without MSN input, amount of GP(e/i) per MSN, amount of MSN input per GP(e/i), amount of synapses from MSN to same cell
    #write into dataframe/ table

    msn_gp_data = pd.DataFrame(columns=["% of MSN without GP", "% of GP without MSN", "average GP amount per MSN", "average MSN input per GP", "average amount of MSN synapses to GP","average amount of MSN synapses to same GP", "average GP summed synapse size per MSN", "average summed size of MSN synapses to same GP"],
                               index = ["GP", "GPe", "GPi"])
    msn_gp_data["% of MSN without GP"]["GP"] = 100 * len(msn_nogp)/len(msn_ids)
    msn_gp_data["% of MSN without GP"]["GPe"] = 100 * len(msn_nogpe) / len(msn_ids)
    msn_gp_data["% of MSN without GP"]["GPi"] = 100 * len(msn_nogpi) / len(msn_ids)
    msn_gp_data["% of GP without MSN"]["GP"] = 100 * len(gp_nomsn) / len(gp_ids)
    msn_gp_data["% of GP without MSN"]["GPe"] = 100 * len(gpe_nomsn) / len(gpe_ids)
    msn_gp_data["% of GP without MSN"]["GPi"] = 100 * len(gpi_nomsn) / len(gpi_ids)
    msn_gp_data["average GP amount per MSN"]["GP"] = np.mean(msn_multigp)
    msn_gp_data["average GP amount per MSN"]["GPe"] = np.mean(msn_multigpe)
    msn_gp_data["average GP amount per MSN"]["GPi"] = np.mean(msn_multigpi)
    msn_gp_data["average MSN input per GP"]["GP"] = np.mean(gp_multiplemsn)
    msn_gp_data["average MSN input per GP"]["GPe"] = np.mean(gpe_multiple)
    msn_gp_data["average MSN input per GP"]["GPi"] = np.mean(gpi_multiple)
    msn_gp_data["average amount of MSN synapses to same GP"]["GP"] = np.mean(msn_samegp_syn_amount)
    msn_gp_data["average amount of MSN synapses to same GP"]["GPe"] = np.mean(msn_samegpe_syn_amount)
    msn_gp_data["average amount of MSN synapses to same GP"]["GPi"] = np.mean(msn_samegpi_syn_amount)
    msn_gp_data["average amount of MSN synapses to GP"]["GP"] = np.mean(msn_gp_synamount)
    msn_gp_data["average amount of MSN synapses to GP"]["GPe"] = np.mean(msn_gpe_synamount)
    msn_gp_data["average amount of MSN synapses to GP"]["GPi"] = np.mean(msn_gpi_synamount)
    msn_gp_data["average summed size of MSN synapses to same GP"]["GP"] = np.mean(msn_samegp_syn_size)
    msn_gp_data["average summed size of MSN synapses to same GP"]["GPe"] = np.mean(msn_samegpe_syn_size)
    msn_gp_data["average summed size of MSN synapses to same GP"]["GPi"] = np.mean(msn_samegpi_syn_size)
    msn_gp_data["average GP summed synapse size per MSN"]["GP"] = np.mean(msn_gp_synsize)
    msn_gp_data["average GP summed synapse size per MSN"]["GPe"] = np.mean(msn_gpe_synsize)
    msn_gp_data["average GP summed synapse size per MSN"]["GPi"] = np.mean(msn_gpi_synsize)
    msn_gp_data.to_csv("%s/msn_gp_data.csv" % f_name)

    #plot distributions of parameters
    sns.distplot(gp_multiplemsn, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(gpe_multiple, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(gpi_multiple, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/gp_multiplemsn.png' % (f_name))
    plt.title('amount of MSN onto same GP')
    plt.xlabel("amount of MSNs per GP")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(msn_multigp, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(msn_multigpe, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(msn_multigpi, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msn_multiplegp.png' % (f_name))
    plt.title('amount of GP per MSN')
    plt.xlabel("amount of GPs per MSN")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(msn_gp_synamount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(msn_gpe_synamount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(msn_gpi_synamount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msn_gpsynamountoverall.png' % (f_name))
    plt.title('amount of MSN synapses onto GP')
    plt.xlabel("amount of synapses")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(msn_gp_synsize, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(msn_gpe_synsize, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(msn_gpi_synsize, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msn_gpsynsizeoverall.png' % (f_name))
    plt.title('sum of synapse size of MSN synapses onto GP')
    plt.xlabel("sum of synapse size")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(msn_samegp_syn_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(msn_samegpe_syn_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(msn_samegpi_syn_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msn_gsamegp_synamount_avg.png' % (f_name))
    plt.title('average amount of synapses from MSN to same GP cell')
    plt.xlabel("average synapse amount")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(msn_samegp_syn_size, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(msn_samegpe_syn_size, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(msn_samegpi_syn_size, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msn_gsamegp_synsize_avg.png' % (f_name))
    plt.title('sum of synapse size of MSN synapses onto same GP')
    plt.xlabel("sum of synapse size")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(np.log10(msn_gp_synsize), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(np.log10(msn_gpe_synsize),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(np.log10(msn_gpi_synsize),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msnlog_gpsynsizeoverall.png' % (f_name))
    plt.title('log sum of synapse size of MSN synapses onto GP')
    plt.xlabel("log sum of synapse size")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

    sns.distplot(np.log10(msn_gp_synamount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="GP")
    sns.distplot(np.log10(msn_gpe_synamount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                 kde=False, label="GPe")
    sns.distplot(np.log10(msn_gpi_synamount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                 kde=False, label="GPi")
    plot_filename = ('%s/msnlog_gpsynamountoverall.png' % (f_name))
    plt.title('log amount of MSN synapses onto GP')
    plt.xlabel("log amount of synapses")
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()


    time_stamps = [time.time()]
    step_idents = ['t-4']
    plottime = time.time() - cellgptime
    print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

    log.info('MSN GP synapse data')

    raise ValueError