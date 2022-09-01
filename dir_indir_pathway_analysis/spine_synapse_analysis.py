
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os as os
import time
from tqdm import tqdm
from syconn.handler.config import initialize_logging
import pandas as pd
from scipy.stats import ranksums
from syconn.handler.basics import write_obj2pkl


def spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, folder_name, min_synsize = 0.1, syn_prob_thresh = 0.6, ct_certainty = 0.6, min_den_length = 100, percentile_param = 10, min_ax_length = 100):
    """
    Analysis synapses between MSN axons and GPe/i dendrites in relation to spine density on MSN cells. Iterates over synapses to find MSN axon to GP dendrite synapses to analyse. Then iterates over MSN
    to calculate spine density as amount of spines per µm dendrite length. Returns different scatter, distplots and violin plots for the analysis.
    :param ssd: super segmenentation dataset
    :param sd_synssv: segmentation dataset synapses
    :param msn_ids: ids of MSN cells
    :param gpe_ids: ids of GPe cells
    :param gpi_ids: ids of GPi cells
    :param folder_name: name of folder where plots should be stored
    :param min_synsize: minimal synapse size
    :param syn_prob_thresh: threshold for synapse probability
    :param ct_certainty: threshold for celltype certainty
    :param min_den_length: minimal dendrite length in µm
    :param min_ax_length: minimal axon length in µm
    :param percentile_param: percentile that should be considered (e.g. 10: 10 and 90 will be used)
    :return: none
    """

    start = time.time()
    loadtime = time.time() - start
    print("%.2f min, %.2f sec for loading" % (loadtime // 60, loadtime % 60))

    param_name = "ms_%.2f_spt_%.2f_ct_%.2f_mdl_%i_perc_%i_mal_%i" % (min_synsize, syn_prob_thresh, ct_certainty, min_den_length, percentile_param, min_ax_length)
    f_name = "%s/%s" % (folder_name, param_name)
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    log = initialize_logging('spiness synapse analysis', log_dir=f_name + '/logs/')
    log.info("parameters: min_synsize = %.2f, syn_prob_thresh = %.2f, ct_certainty = %.2f, min_den_length = %i, percentile = %i, min_ax_lenth = %i" % (min_synsize, syn_prob_thresh, ct_certainty, min_den_length, percentile_param, min_ax_length))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info('Step 1/4 load full cells')
    gp_ids = np.hstack([gpe_ids, gpi_ids])

    log.info('Step 2/4 iterate trough synapses of MSN')
    celltype = 2
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    syn_prob = sd_synssv.load_cached_data("syn_prob")
    m = syn_prob > syn_prob_thresh
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
    m_spiness = m_spiness[ct_inds]
    #make dictionary with ratio of GPi to GPe
    msn_ax_dict = {cellid: {"GPe_amount": 0, "GPe_size":0, "GPi_amount": 0, "GPi_size": 0} for cellid in msn_ids}
    #dictionary with spine density for MSM
    msn_den_dict = {cellid: {"amount":0, "size": 0} for cellid in msn_ids}
    msn_den_gpspinedict = {cellid: {"GPe_amount": 0, "GPe_size":0, "GPi_amount": 0, "GPi_size": 0} for cellid in msn_ids}
    for ix, id in enumerate(tqdm(m_ids)):
        syn_ax = m_axs[ix]
        if not 1 in syn_ax:
            continue
        if syn_ax[0] == syn_ax[1]: #no axo-axonic synapses
            continue
        if not np.any(np.in1d(m_ssv_partners[ix], msn_ids)):
            continue
        m_size = m_sizes[ix]
        if m_size < min_synsize:
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
        if ssv1 in msn_ids:
            #if ssv1 not in cellproba_thresh_ids:
                #continue
            if (ssv2 not in gp_ids) and (ssv2 not in msn_ids):
                continue
            if ssv2 in gpe_ids:
                msn_ax_dict[ssv1]["GPe_amount"] += 1
                msn_ax_dict[ssv1]["GPe_size"] += m_size
            elif ssv2 in gpi_ids:
                msn_ax_dict[ssv1]["GPi_amount"] += 1
                msn_ax_dict[ssv1]["GPi_size"] += m_size
            elif ssv2 in msn_ids:
                if spin2 == 1:
                    msn_den_dict[ssv2]["amount"] += 1
                    msn_den_dict[ssv2]["size"] += m_size
        else:
            if (spin2 != 1) and (spin2 != 2):
                continue
            if spin2 == 1:
                msn_den_dict[ssv2]["amount"] += 1
                msn_den_dict[ssv2]["size"] += m_size
            if ssv1 in gp_ids:
                if ssv1 in gpe_ids:
                    msn_den_gpspinedict[ssv2]["GPe_amount"] += 1
                    msn_den_gpspinedict[ssv2]["GPe_size"] += m_size
                else:
                    msn_den_gpspinedict[ssv2]["GPe_amount"] += 1
                    msn_den_gpspinedict[ssv2]["GPe_size"] += m_size


    time_stamps = [time.time()]
    step_idents = ['t-1']
    syntime = time.time() - loadtime
    print("%.2f min, %.2f sec for iterating over synapses" % (syntime // 60, syntime % 60))

    log.info('Step 3/4 iterate over MSN cells')
    gpi_ratio_amount_axons = np.zeros(len(msn_ids))
    gpi_ratio_size_axons = np.zeros(len(msn_ids))
    gpi_ratio_amount_dens = np.zeros(len(msn_ids))
    gpi_ratio_size_dens = np.zeros(len(msn_ids))
    spine_density_synamount = np.zeros(len(msn_ids))
    spine_density_synsize = np.zeros(len(msn_ids))
    spine_density_skels = np.zeros(len(msn_ids))


    for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(msn_ids))):
        if msn_den_dict[cell.id]["amount"] == 0:
            continue
        if cell.certainty_celltype() < ct_certainty:
           continue
        cell.load_skeleton()
        nondendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 0)[0]
        nonaxon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 1)[0]
        spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
        spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
        spine_headinds = np.nonzero(cell.skeleton["spiness"] == 1)[0]
        spine_neckinds = np.nonzero(cell.skeleton["spiness"] == 2)[0]
        nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
        spine_inds = np.hstack([spine_headinds, spine_neckinds])
        g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
        axon_graph = g.copy()
        axon_graph.remove_nodes_from(nonaxon_inds)
        axo_length = axon_graph.size(weight = "weight")/ 1000 #in µm
        if axo_length < min_ax_length:
            continue
        dendrite_graph = g.copy()
        dendrite_graph.remove_nodes_from(nondendrite_inds)
        # calculate dendrite length of whole cell from edges in dendrite
        den_shaft_graph = dendrite_graph.copy()
        den_shaft_graph.remove_nodes_from(spine_inds)
        den_length = den_shaft_graph.size(weight="weight") / 1000  # in µm
        if den_length < min_den_length:
            continue
        spine_graph = dendrite_graph.copy()
        spine_graph.remove_nodes_from(nonspine_inds)
        spine_amount_skeleton = len(list(nx.connected_component_subgraphs(spine_graph)))
        spine_density_skels[i] = spine_amount_skeleton/den_length
        spine_density_synamount[i] = msn_den_dict[cell.id]["amount"]/den_length
        spine_density_synsize[i] = msn_den_dict[cell.id]["size"]/den_length
        #calculate GPi ratio = GPi/(GPe + GPi)
        ax_GPi_am = msn_ax_dict[cell.id]["GPi_amount"]
        ax_GPe_am = msn_ax_dict[cell.id]["GPe_amount"]
        if (ax_GPe_am == 0) and (ax_GPi_am == 0):
            continue
        if msn_ax_dict[cell.id]["GPi_size"] == 0:
            ax_GPi_si = 0
        else:
            ax_GPi_si = msn_ax_dict[cell.id]["GPi_size"]/ msn_ax_dict[cell.id]["GPi_amount"]
        if msn_ax_dict[cell.id]["GPe_size"] == 0:
            ax_GPe_si = 0
        else:
            ax_GPe_si = msn_ax_dict[cell.id]["GPe_size"] / msn_ax_dict[cell.id]["GPe_amount"]
        gpi_ratio_amount_axons[i] = ax_GPi_am/(ax_GPe_am + ax_GPi_am)
        gpi_ratio_size_axons[i] = ax_GPi_si/(ax_GPe_si + ax_GPi_si)
        den_GPe_am = msn_den_gpspinedict[cell.id]["GPe_amount"]
        den_GPi_am = msn_den_gpspinedict[cell.id]["GPi_amount"]
        if (den_GPe_am == 0) and (den_GPi_am == 0):
            continue
        if msn_den_gpspinedict[cell.id]["GPe_size"] == 0:
            den_GPe_si = 0
        else:
            den_GPe_si = msn_den_gpspinedict[cell.id]["GPe_size"]/msn_den_gpspinedict[cell.id]["GPe_amount"]
        if msn_den_gpspinedict[cell.id]["GPi_size"] == 0:
            den_GPi_si = 0
        else:
            den_GPi_si = msn_den_gpspinedict[cell.id]["GPi_size"]/msn_den_gpspinedict[cell.id]["GPi_amount"]
        gpi_ratio_amount_dens[i] = den_GPi_am/(den_GPi_am + den_GPe_am)
        gpi_ratio_size_axons[i] = den_GPi_si/(den_GPe_si + den_GPi_si)

    syn_inds = spine_density_synamount > 0
    spine_density_synsize = spine_density_synsize[syn_inds]
    spine_density_synamount = spine_density_synamount[syn_inds]
    spine_density_skels = spine_density_skels[syn_inds]
    gpi_ratio_amount_axons = gpi_ratio_amount_axons[syn_inds]
    gpi_ratio_size_axons = gpi_ratio_size_axons[syn_inds]
    gpi_ratio_amount_dens = gpi_ratio_amount_dens[syn_inds]
    gpi_ratio_size_dens = gpi_ratio_size_dens[syn_inds]
    # get 90, 80, 20, 10 percentile and make sure they don't overlap
    spine_den_synsize_per90ind = np.where(spine_density_synsize > np.percentile(spine_density_synsize, 100-percentile_param))
    spine_den_synamount_per90ind = np.where(spine_density_synamount > np.percentile(spine_density_synamount, 100-percentile_param))
    spine_den_synsize_per10ind = np.where(spine_density_synsize < np.percentile(spine_density_synsize, percentile_param))
    spine_den_synamount_per10ind = np.where(spine_density_synamount < np.percentile(spine_density_synamount, percentile_param))
    spine_den_synsize_per80ind = np.where(spine_density_synsize > np.percentile(spine_density_synsize, 100 - percentile_param*2))
    spine_den_synsize_per80ind = spine_den_synsize_per80ind[0][np.in1d(spine_den_synsize_per80ind, spine_den_synsize_per90ind) == False]
    spine_den_synamount_per80ind = np.where(spine_density_synamount > np.percentile(spine_density_synamount, 100 - percentile_param*2))
    spine_den_synamount_per80ind = spine_den_synamount_per80ind[0][np.in1d(spine_den_synamount_per80ind, spine_den_synamount_per90ind) == False]
    spine_den_synsize_per20ind = np.where(spine_density_synsize < np.percentile(spine_density_synsize, percentile_param*2))
    spine_den_synsize_per20ind = spine_den_synsize_per20ind[0][np.in1d(spine_den_synsize_per20ind, spine_den_synsize_per10ind) == False]
    spine_den_synamount_per20ind = np.where(spine_density_synamount < np.percentile(spine_density_synamount, percentile_param*2))
    spine_den_synamount_per20ind = spine_den_synamount_per20ind[0][np.in1d(spine_den_synamount_per20ind, spine_den_synamount_per10ind) == False]
    spine_den_synsi90 = spine_density_synsize[spine_den_synsize_per90ind]
    gpi_ratio_size_90 = gpi_ratio_size_axons[spine_den_synsize_per90ind]
    spine_den_synsi10 = spine_density_synsize[spine_den_synsize_per10ind]
    gpi_ratio_size_10 = gpi_ratio_size_axons[spine_den_synsize_per10ind]
    spine_den_synam90 = spine_density_synamount[spine_den_synamount_per90ind]
    gpi_ratio_am90 = gpi_ratio_amount_axons[spine_den_synamount_per90ind]
    spine_den_synam10 = spine_density_synamount[spine_den_synamount_per10ind]
    gpi_ratio_am10 = gpi_ratio_amount_axons[spine_den_synamount_per10ind]
    spine_den_synsi80 = spine_density_synsize[spine_den_synsize_per80ind]
    gpi_ratio_size_80 = gpi_ratio_size_axons[spine_den_synsize_per80ind]
    spine_den_synsi20 = spine_density_synsize[spine_den_synsize_per20ind]
    gpi_ratio_size_20 = gpi_ratio_size_axons[spine_den_synsize_per20ind]
    spine_den_synam80 = spine_density_synamount[spine_den_synamount_per80ind]
    gpi_ratio_am80 = gpi_ratio_amount_axons[spine_den_synamount_per80ind]
    spine_den_synam20 = spine_density_synamount[spine_den_synamount_per20ind]
    gpi_ratio_am20 = gpi_ratio_amount_axons[spine_den_synamount_per20ind]
    skel_inds = spine_density_skels > 0

    am_stathl, am_phl = ranksums(gpi_ratio_am90, gpi_ratio_am10)
    am_stathh, am_phh = ranksums(gpi_ratio_am90, gpi_ratio_am80)
    am_statll, am_pll = ranksums(gpi_ratio_am20, gpi_ratio_am10)
    am_statmhl, am_pmhl = ranksums(gpi_ratio_am80, gpi_ratio_am20)
    si_stathl, si_phl = ranksums(gpi_ratio_size_90, gpi_ratio_size_10)
    si_stathh, si_phh = ranksums(gpi_ratio_size_90, gpi_ratio_size_80)
    si_statll, si_pll = ranksums(gpi_ratio_size_20, gpi_ratio_size_10)
    si_statmhl, si_pmhl = ranksums(gpi_ratio_size_80, gpi_ratio_size_20)

    ranksum_results = pd.DataFrame(columns=["high (%i) vs low (%i) spiness" % (100 - percentile_param, percentile_param), "%i vs %i" % (100 - 2*percentile_param, percentile_param*2), "%i vs %i high spiness"
                                            % (100 - percentile_param, 100 - 2*percentile_param), "%i vs %i low spiness" % (2*percentile_param, percentile_param)],
                                   index=["amount stats", "amount p-value", "size stats", "size p_value"])
    ranksum_results["high (%i) vs low (%i) spiness" % (100 - percentile_param, percentile_param)]["amount stats"] = am_stathl
    ranksum_results["high (%i) vs low (%i) spiness" % (100 - percentile_param, percentile_param)]["amount p-value"] = am_phl
    ranksum_results["high (%i) vs low (%i) spiness" % (100 - percentile_param, percentile_param)]["size stats"] = si_stathl
    ranksum_results["high (%i) vs low (%i) spiness" % (100 - percentile_param, percentile_param)]["size p_value"] = si_phl
    ranksum_results["%i vs %i" % (100 - 2*percentile_param, percentile_param*2)]["amount stats"] = am_statmhl
    ranksum_results["%i vs %i" % (100 - 2*percentile_param, percentile_param*2)]["amount p-value"] = am_pmhl
    ranksum_results["%i vs %i" % (100 - 2*percentile_param, percentile_param*2)]["size stats"] = si_statmhl
    ranksum_results["%i vs %i" % (100 - 2*percentile_param, percentile_param*2)]["size p_value"] = si_pmhl
    ranksum_results["%i vs %i high spiness" % (100 - percentile_param, 100 - 2*percentile_param)]["amount stats"] = am_stathh
    ranksum_results["%i vs %i high spiness" % (100 - percentile_param, 100 - 2*percentile_param)]["amount p-value"] = am_phh
    ranksum_results["%i vs %i high spiness" % (100 - percentile_param, 100 - 2*percentile_param)]["size stats"] = si_stathh
    ranksum_results["%i vs %i high spiness" % (100 - percentile_param, 100 - 2*percentile_param)]["size p_value"] = si_phh
    ranksum_results["%i vs %i low spiness" % (2*percentile_param, percentile_param)]["amount stats"] = am_statll
    ranksum_results["%i vs %i low spiness" % (2*percentile_param, percentile_param)]["amount p-value"] = am_pll
    ranksum_results["%i vs %i low spiness" % (2*percentile_param, percentile_param)]["size stats"] = si_statll
    ranksum_results["%i vs %i low spiness" % (2*percentile_param, percentile_param)]["size p_value"] = si_pll
    ranksum_results.to_csv("%s/wilcoxon_ranksum_results.csv" % f_name)

    time_stamps = [time.time()]
    step_idents = ['t-2']
    celltime = time.time() - syntime
    print("%.2f min, %.2f sec for iterating over MSNs" % (celltime // 60, celltime % 60))

    log.info('Step 4/4 plot spine parameters of MSN cells')
    # violin plot with highest and lowest percentile
    high_low_spiness_GPi_am = pd.DataFrame()
    high_low_spiness_GPi_am["GPi ratio"]= np.hstack([gpi_ratio_am90, gpi_ratio_am10])
    high_low_spiness_GPi_am["spiness"] = str()
    high_low_spiness_GPi_am.loc[0:len(gpi_ratio_am90) - 1, "spiness"]= ("high spiness (>%i percentile)" % (100 - percentile_param))
    high_low_spiness_GPi_am.loc[len(gpi_ratio_am90):len(gpi_ratio_am90) + len(gpi_ratio_am10) - 1, "spiness"] = ("low spiness (<%i percentile)" % percentile_param)
    write_obj2pkl("%s/%s_high_low_spiness_GPi_amount_df.pkl" % (f_name, param_name), high_low_spiness_GPi_am)

    high_low_spiness_GPi_size = pd.DataFrame()
    high_low_spiness_GPi_size["GPi ratio"] = np.hstack([gpi_ratio_size_90, gpi_ratio_size_10])
    high_low_spiness_GPi_size["spiness"] = str()
    high_low_spiness_GPi_size.loc[0:len(gpi_ratio_am90) - 1, "spiness"] = (
                "high spiness (>%i percentile)" % (100 - percentile_param))
    high_low_spiness_GPi_size.loc[len(gpi_ratio_size_90):len(gpi_ratio_size_90) + len(gpi_ratio_size_10) - 1, "spiness"] = (
                "low spiness (<%i percentile)" % percentile_param)
    write_obj2pkl("%s/%s_high_low_spiness_GPi_size_df.pkl" % (f_name, param_name), high_low_spiness_GPi_size)

    spiness_pal = {"high spiness (>%i percentile)" % (100 - percentile_param): "darkturquoise", "low spiness (<%i percentile)" % percentile_param: "gray"}
    write_obj2pkl("%s/%s_palette_dict.pkl" % (f_name, param_name), spiness_pal)

    sns.violinplot(x = "spiness", y = "GPi ratio", data = high_low_spiness_GPi_am, palette=spiness_pal, inner="box")
    sns.stripplot(x = "spiness", y = "GPi ratio", data = high_low_spiness_GPi_am, color="black", alpha = 0.5)
    plt.title('spine density vs GPi ratio in %s' % ct_dict[celltype])
    filename = ("%s/vio_gpiam_spiness_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="spiness", y="GPi ratio", data=high_low_spiness_GPi_size, palette=spiness_pal, inner="box")
    sns.stripplot(x="spiness", y="GPi ratio", data=high_low_spiness_GPi_size, color="black", alpha=0.5)
    plt.title('spine density vs GPi ratio in %s' % ct_dict[celltype])
    filename = ("%s/vio_gpisize_spiness_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()


    # plot spine_density from spine heads vs spine density from skeletons
    plt.scatter(x=spine_density_synamount, y=spine_density_skels, c="skyblue")
    filename = ('%s/scspineamountsyn_vs_skel_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs skeleton in %.4s' % ct_dict[celltype])
    plt.xlabel('spine head synapse amount per µm')
    plt.ylabel('spine amount per µm from skeleton')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_synamount, y=spine_density_synsize, c="skyblue")
    filename = ('%s/scspinesynam_vs_size_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses in %.4s' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('avg size of spine head synapses per µm')
    plt.savefig(filename)
    plt.close()

    #plot gpi ratio vs synamount in size and amount
    plt.scatter(x=spine_density_skels, y=gpi_ratio_amount_axons, c="skyblue")
    filename = ('%s/scspinesskel_vs_gpiamax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from skeletons vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('amount of spines in skeleton per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_synamount, y=gpi_ratio_amount_axons, c="skyblue")
    filename = ('%s/scspinesssynam_vs_gpiamax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_synsize, y=gpi_ratio_size_axons, c="skyblue")
    filename = ('%s/scspinesssynsize_vs_gpisizeax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_skels, y=gpi_ratio_amount_dens, c="skyblue")
    filename = ('%s/scspinesskel_vs_gpiamden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from skeletons vs GPi amount in %.4s dendrites' % ct_dict[celltype])
    plt.xlabel('amount of spines in skeleton per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_synamount, y=gpi_ratio_amount_dens, c="skyblue")
    filename = ('%s/scspinesssynam_vs_gpiamden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s dendrite' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=spine_density_synsize, y=gpi_ratio_size_dens, c="skyblue")
    filename = ('%s/scspinesssynsize_vs_gpisizeden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s dendrites' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.close()

    #plot log spine density
    plt.scatter(x=np.log10(spine_density_skels[skel_inds]), y=gpi_ratio_amount_axons[skel_inds], c="skyblue")
    filename = ('%s/sclogspinesskel_vs_gpiamax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from skeletons vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('log amount of spines in skeleton per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=np.log10(spine_density_synamount), y=gpi_ratio_amount_axons, c="skyblue")
    filename = ('%s/sclogspinesssynam_vs_gpiamax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('log amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=np.log10(spine_density_synsize), y=gpi_ratio_size_axons, c="skyblue")
    filename = ('%s/sclogspinesssynsize_vs_gpisizeax_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('log avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=np.log10(spine_density_skels), y=gpi_ratio_amount_dens, c="skyblue")
    filename = ('%s/sclogspinesskel_vs_gpiamden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from skeletons vs GPi amount in %.4s dendrites' % ct_dict[celltype])
    plt.xlabel('log amount of spines in skeleton per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=np.log10(spine_density_synamount[skel_inds]), y=gpi_ratio_amount_dens[skel_inds], c="skyblue")
    filename = ('%s/sclogspinesssynam_vs_gpiamden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from synapses vs GPi amount in %.4s dendrites' % ct_dict[celltype])
    plt.xlabel('log amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x=np.log10(spine_density_synsize), y=gpi_ratio_size_dens, c="skyblue")
    filename = ('%s/sclogspinesssynsize_vs_gpisizeden_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('log spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('log avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.close()

    plt.scatter(x = spine_den_synam90, y= gpi_ratio_am90, c="darkturquoise", label="high spine density (>%i percentile)" % (100-percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synam10, y=gpi_ratio_am10, c="black", label="low spine density (<%i percentile)" % percentile_param ,alpha=0.7)
    filename = ('%s/scspinesssynamperc_vs_gpiamperc_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.legend()
    plt.close()

    plt.scatter(x=spine_den_synsi90, y=gpi_ratio_size_90, c="darkturquoise",
                label="high spine density (>%i percentile)" % (100 - percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synsi10, y=gpi_ratio_size_10, c="black", label="low spine density (< %i percentile)" % percentile_param,
                alpha=0.7)
    filename = ('%s/scspinesssynsizeperc_vs_gpisizeperc_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.legend()
    plt.close()

    plt.scatter(x=spine_den_synam90, y=gpi_ratio_am90, c="darkturquoise",
                label="high spine density (>%i percentile)" % (100-percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synam10, y=gpi_ratio_am10, c="black", label="low spine density (<%i percentile)" % percentile_param,
                alpha=0.7)
    filename = ('%s/scspinesssynamperc_vs_gpiamperc_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('GPi ratio of amount of GP synapses')
    plt.savefig(filename)
    plt.legend()
    plt.close()

    plt.scatter(x=spine_den_synsi90, y=gpi_ratio_size_90, c="darkturquoise",
                label="high spine density (>%i percentile)" % (100 - percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synsi80, y=gpi_ratio_size_80, c="paleturquoise",
                label="high spine density (%i < percentile < %i)" % (100 - 2*percentile_param, 100 - percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synsi20, y=gpi_ratio_size_20, c="gray",
                label="low spine density (%i < percentile < %i)" % (percentile_param, 2*percentile_param), alpha=0.7)
    plt.scatter(x=spine_den_synsi10, y=gpi_ratio_size_10, c="black",
                label="low spine density (<%i percentile)" % percentile_param,
                alpha=0.7)
    filename = ('%s/scspinesssynsizeperc4_vs_gpisizeperc4_%s.png' % (f_name, ct_dict[celltype]))
    plt.title('spine density from synapses vs GPi amount in %.4s axons' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('GPi ratio of size of GP synapses')
    plt.savefig(filename)
    plt.legend()
    plt.close()

    #make displots for parameters with log for densities

    sns.distplot(spine_density_skels[skel_inds], kde=False, color="skyblue")
    plt.title('spine density skeletons %s' % ct_dict[celltype])
    plt.xlabel('amount of spines per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/spinedenskel_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    #plots with highest and lowest percentile
    sns.distplot(gpi_ratio_size_90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(gpi_ratio_size_10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (< %i percentile)" % percentile_param)
    plt.title('GPi ratio of size of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/gpratiosize_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_am90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param), bins=10)
    sns.distplot(gpi_ratio_am10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param, bins=10)
    plt.title('GPi ratio of amount of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/gpratioam_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_am90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param), norm_hist=True, bins=10)
    sns.distplot(gpi_ratio_am10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param, norm_hist=True, bins=10)
    plt.title('GPi ratio of amount of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('fraction of cells')
    plt.legend()
    plt.savefig('%s/gpratioam_percentilesnorm_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_den_synsi90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(spine_den_synsi10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/synsize_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_den_synam90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(spine_den_synam10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/synamount_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    #plots with 4 percentiles
    sns.distplot(gpi_ratio_size_90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(gpi_ratio_size_10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % (percentile_param))
    sns.distplot(gpi_ratio_size_80, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile < %i)" % (100 - 2*percentile_param, 100 - percentile_param))
    sns.distplot(gpi_ratio_size_20, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile < %i)" % (percentile_param, 2*percentile_param))
    plt.title('GPi ratio of size of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/gpratiosize_percentiles4_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_am90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param), bins=10)
    sns.distplot(gpi_ratio_am10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param, bins=10)
    sns.distplot(gpi_ratio_am80, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile < %i)" % (100 - 2*percentile_param, 100 - percentile_param), bins=10)
    sns.distplot(gpi_ratio_am20, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile < %i)" % (percentile_param, 2*percentile_param), bins=10)
    plt.title('GPi ratio of amount of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/gpratioam_percentiles4_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_am90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param), norm_hist=True, bins=10)
    sns.distplot(gpi_ratio_am10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param, norm_hist=True, bins=10)
    sns.distplot(gpi_ratio_am80, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile < %i)" % (100 - 2*percentile_param, 100-percentile_param), norm_hist=True, bins=10)
    sns.distplot(gpi_ratio_am20, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile < %i)" % (percentile_param, 2*percentile_param), norm_hist=True, bins=10)
    plt.title('GPi ratio of amount of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('fraction of cells')
    plt.legend()
    plt.savefig('%s/gpratioam_percentiles4norm_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_den_synsi90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(spine_den_synsi10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % (percentile_param))
    sns.distplot(spine_den_synsi80, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile <%i)" % (100 - 2*percentile_param, 100 - percentile_param))
    sns.distplot(spine_den_synsi20, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile <%i)" % (percentile_param, 2*percentile_param))
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('avg size of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/synsize_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_den_synam90, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(spine_den_synam10, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    sns.distplot(spine_den_synam80, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile < %i)" % (100 - 2*percentile_param, 100 - percentile_param))
    sns.distplot(spine_den_synam20, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile <%i)" % (percentile_param, 2*percentile_param))
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/synamount_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_density_synamount, kde=False, color="skyblue")
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('amount of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/spinedensynam_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(spine_density_synsize, kde=False, color="skyblue")
    plt.title('spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('summed size of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/spinedensynsize_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_amount_axons, kde=False, color="skyblue")
    plt.title('GPi ratio of amount of %s axons to GP synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.savefig('%s/gpratioam_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_size_axons, kde=False, color="skyblue")
    plt.title('GPi ratio of size of %s axons to GP synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.savefig('%s/gpratiosize_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_amount_dens, kde=False, color="skyblue")
    plt.title('GPi ratio of amount of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.savefig('%s/gpratioamden_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(gpi_ratio_size_dens, kde=False, color="skyblue")
    plt.title('GPi ratio of size of %s spine synapses' % ct_dict[celltype])
    plt.xlabel('GPi ratio')
    plt.ylabel('count of cells')
    plt.savefig('%s/gpratiosizeden_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_density_skels[skel_inds]), kde=False, color="skyblue")
    plt.title('log spine density skeletons %s' % ct_dict[celltype])
    plt.xlabel('log amount of spines per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/logspinedenskel_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_density_synamount), kde=False, color="skyblue")
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log amount of spines per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/logspinedensynam_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_density_synsize), kde=False, color="skyblue")
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log size of spines per µm')
    plt.ylabel('count of cells')
    plt.savefig('%s/logspinedensynsize_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_den_synsi90), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(np.log10(spine_den_synsi10), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log avg size of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/logsynsize_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_den_synam90), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(np.log10(spine_den_synam10), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log amount of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/logsynamount_percentiles_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_den_synsi90),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(np.log10(spine_den_synsi10),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    sns.distplot(np.log10(spine_den_synsi80),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile <%i)" % (100 - 2*percentile_param, 100 - percentile_param))
    sns.distplot(np.log10(spine_den_synsi20),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile <%i)" % (percentile_param, 2 * percentile_param))
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log avg size of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/logsynsize_percentiles_4%s.png' % (f_name, ct_dict[celltype]))
    plt.close()

    sns.distplot(np.log10(spine_den_synam90),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                 kde=False, label="high spine density (>%i percentile)" % (100 - percentile_param))
    sns.distplot(np.log10(spine_den_synam10),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                 kde=False, label="low spine density (<%i percentile)" % percentile_param)
    sns.distplot(np.log10(spine_den_synam80),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "paleturquoise"},
                 kde=False, label="high spine density (%i < percentile <%i)" % (100 - 2*percentile_param, 100 - percentile_param))
    sns.distplot(np.log10(spine_den_synam20),
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                 kde=False, label="low spine density (%i < percentile <%i)" % (percentile_param, 2*percentile_param))
    plt.title('log spine density spine head synapses %s' % ct_dict[celltype])
    plt.xlabel('log amount of spine head synapses per µm')
    plt.ylabel('count of cells')
    plt.legend()
    plt.savefig('%s/logsynamount_percentiles4_%s.png' % (f_name, ct_dict[celltype]))
    plt.close()



    time_stamps = [time.time()]
    step_idents = ['t-3']
    plottime = time.time() - celltime
    print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

    log.info('spiness synapse analysis finished')





