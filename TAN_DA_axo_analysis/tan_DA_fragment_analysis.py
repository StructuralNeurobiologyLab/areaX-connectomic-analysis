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
    from tqdm import tqdm
    from syconn.handler.basics import write_obj2pkl
    from scipy.stats import ranksums
    from u.arother.bio_analysis.general.analysis_helper import get_comp_radii
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_vcssv = SegmentationDataset("vc", working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)


    def synapse_vc_per_axon(sso, cached_syn_ids, cached_syn_ssv_partners, cached_syn_partner_axoness, cached_syn_sizes, cached_syn_rep_coord,
                         cached_vc_ids, cached_vc_sizes,cached_vc_repcoord, min_syn_size=0.1):
        """
        iterate over synapses per cell and returns ids with minimal size and syn_probabilty threshold. Compares to cached array with synapse properties in axon.
        axoness: 0 = dendrite, 1 = axon, 2 = soma
        :param sso: super segmentation object
        :param min_syn_size: threshold for minimal synapse size
        :param cached_ids: synapses ids from same celltype, thresholded by synapse probability
        :param cached_partner_axoness: axoness synapse on both neurons, of synapses from same celltype, thresholded by synapse probability
        :param cached_ssv_partners: ssv.id of both neurons, of synapses from same celltpye, thresholded by synapse probability
        :return: synapse_ids, synapse_sizes, vc_to_skeleton_density, vc_to_synapse_density
        """
        syn_ssvs = sso.syn_ssv
        if len(syn_ssvs) == 0:
            return 0, 0
        # get ids from cell
        sso_syn_ids0 = np.where(cached_syn_ssv_partners == sso.id)[0]
        all_syn_ids = cached_syn_ids[sso_syn_ids0]
        all_syn_sizes = cached_syn_sizes[sso_syn_ids0]
        all_rep_coord = cached_syn_rep_coord[sso_syn_ids0]
        partner_axoness = cached_syn_partner_axoness[sso_syn_ids0]
        ssv_partners = cached_syn_ssv_partners[sso_syn_ids0]
        # filter those that don have minimum size
        min_syn_inds = all_syn_sizes > min_syn_size
        all_syn_ids = all_syn_ids[min_syn_inds]
        all_syn_sizes = all_syn_sizes[min_syn_inds]
        partner_axoness = partner_axoness[min_syn_inds]
        all_rep_coord = all_rep_coord[min_syn_inds]
        ssv_partners = ssv_partners[min_syn_inds]
        # determine axoness of synapse
        sso_syn_ids1 = np.where(ssv_partners == sso.id)
        axoness = partner_axoness[sso_syn_ids1]
        axo_inds = axoness == 1
        axo_syn_ids = all_syn_ids[axo_inds]
        if len(axo_syn_ids) == 0:
            return 0, 0
        syn_sizes = all_syn_sizes[axo_inds]
        syn_rep_coords = all_rep_coord[axo_inds]
        syn_dict = {"axo_ids": axo_syn_ids, "syn_sizes": syn_sizes}
        # get vc_id and size for sso id
        sso_vc_ids = sso.vc_ids
        if len(sso_vc_ids) == 0:
            return 0, 0
        vc_inds = np.in1d(cached_vc_ids, sso_vc_ids)
        sso_vc_sizes = cached_vc_sizes[vc_inds] * 10 ** (-9) * np.prod(sso.scaling)  # µm
        sso_vc_repcoord = cached_vc_repcoord[vc_inds]
        syn_kdtree = scipy.spatial.cKDTree(syn_rep_coords)
        distance_to_synapse, closest_syn_inds = syn_kdtree.query(sso_vc_repcoord, k=1)
        closest_syn_inds = closest_syn_inds.astype(int)
        closest_syn_ids = all_syn_ids[closest_syn_inds]
        axo_vc_inds = np.in1d(closest_syn_ids, axo_syn_ids)
        close_axo_syn_ids = closest_syn_ids[axo_vc_inds]
        axo_vc_ids = sso_vc_ids[axo_vc_inds]
        axo_vc_sizes = sso_vc_sizes[axo_vc_inds]
        distance_to_synapse = distance_to_synapse[axo_vc_inds] / 1000  # µm
        axo_vc_repcoord = sso_vc_repcoord[axo_vc_inds]
        if len(axo_vc_repcoord) == 0:
            return 0, 0
        axo_vc_dict = {"axo_ids": axo_vc_ids,
                       "vc_sizes": axo_vc_sizes, "distance2synapse": distance_to_synapse}
        return syn_dict, axo_vc_dict

    def ct_axo_analysis(ssd, celltype, foldername,  cached_vc_ids, cached_vc_rep_coords, cached_vc_sizes, syn_ids, syn_cts, syn_rep_coords, syn_sizes,
                        syn_ssv_partners, syn_partner_axoness,
                        min_axon_len = 200, syn_size_thresh = 0.1, handpicked = False, full_cells = False, ignore_full_cells = False):
        '''
        analysis of axons of different celltypes. Axons have to have a minimum skeleton length. Thickness/radius via size of nodes and vesicle cloud density using synapse_vc_per_axon
        will be analysed. For whole cells uses prefiltered array. Uses cached data for synapse and vesicle analysis. Saves parameters in a dictionary.
        Returns medium, min, max radius, synapse and vesicle cloud density in response to skeleton, vesicle cloud density related to synapses (µm³/µm²).
        :param ssd: super segmentation dataset
        :param celltype: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param min_axon_len: minimum axon length, compartment: 0 = dendrite, 1 = axon, 2 = soma
        :param cached_vc_ids, cached_vc_rep_coords, cached_vc_sizes: cached properties for vesicle clouds in whole datatset, needed to analyse synapses and vcs faster per cell
        :param syn_ids, syn_cts, syn_rep_coords, syn_sizes, syn_ssv_partners, syn_partner_axoness: cached properites for synapses needed to analyse synapses per cell faster, will be
        prefiltered for celltype here
        :param syn_size_thresh: threshold for synapse size
        :param handpicked: if true, use cells from array with handpicked cells, can only be True if full_cells also True
        :param full_cells: if true use prefiltered full cells
        :param ignore_full_cells: if true: array with full cells needs to exist and will be excluded from analysis, only makes sense if full_cells is false
        :return: median_radius, min_radius, Max_radius, synapse_amount_density, synapse_size_density, vc_amount_density, vc_vol_density, syn_vc_amount_density, syn_vc_size_density, cell_amount
        '''
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "%s/%s" % (foldername, ct_dict[celltype])
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        if full_cells:
            if ignore_full_cells:
                raise ValueError("full_cells and ignore_full_cells can be both set to true!")
            if handpicked:
                cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_handpicked.pkl" % ct_dict[celltype])
            else:
                cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        else:
            cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        if ignore_full_cells:
            full_cell_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
            full_cell_inds = np.in1d(cellids, full_cell_ids)
            cellids = np.delete(cellids, full_cell_inds)
        #prepare synapse caches for right celltype
        ct_inds = np.any(syn_cts == celltype, axis=1)
        m_ids = syn_ids[ct_inds]
        m_axs = syn_partner_axoness[ct_inds]
        m_ssv_partners = syn_ssv_partners[ct_inds]
        m_sizes = syn_sizes[ct_inds]
        m_rep_coords = syn_rep_coords[ct_inds]
        #initialise empty arrays for parameters per celltype to be filled
        ct_median_radius = np.zeros(len(cellids))
        ct_min_radius = np.zeros(len(cellids))
        ct_max_radius = np.zeros(len(cellids))
        synapse_amount_density = np.zeros(len(cellids))
        synapse_size_density = np.zeros(len(cellids))
        vc_amount_density = np.zeros(len(cellids))
        vc_vol_density = np.zeros(len(cellids))
        syn_vc_amount_density = np.zeros(len(cellids))
        syn_vc_size_density = np.zeros(len(cellids))
        # iterate over cells in celltype
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            cell.load_skeleton()
            if full_cells:
                non_axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 1)[0]
                axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
                axon_radii = get_comp_radii(cell, axon_inds)
                g = cell.weighted_graph(add_node_attr=('axoness_avg10000', ))
                axon_graph = g.copy()
                axon_graph.remove_nodes_from(non_axon_inds)
            else:
                axon_graph = cell.weighted_graph(add_node_attr=('axoness_avg10000', ))
                axon_radii = get_comp_radii(cell)
            axon_length = axon_graph.size(weight="weight") / 1000  # in µm
            if axon_length < min_axon_len:
                continue
            #calculate median, min, max radius from radius of nodes
            ct_min_radius[i] = np.min(axon_radii)
            ct_max_radius[i] = np.max(axon_radii)
            ct_median_radius[i] = np.median(axon_radii)
            cell_syn_dict, cell_vc_dict = synapse_vc_per_axon(cell, cached_syn_ids=m_ids, cached_syn_sizes=m_sizes, cached_syn_partner_axoness=m_axs,
                                                              cached_syn_ssv_partners=m_ssv_partners, cached_syn_rep_coord=m_rep_coords, cached_vc_ids=cached_vc_ids,
                                                              cached_vc_sizes=cached_vc_sizes, cached_vc_repcoord=cached_vc_rep_coords, min_syn_size=syn_size_thresh)
            if type(cell_syn_dict) == int:
                continue
            cell_synapse_amount = len(cell_syn_dict["axo_ids"])
            if cell_synapse_amount == 0:
                continue
            synapse_amount_density[i] = cell_synapse_amount / axon_length
            cell_sum_syn_sizes = np.sum(cell_syn_dict["syn_sizes"])
            synapse_size_density[i] = cell_sum_syn_sizes / axon_length
            cell_vc_amount = len(cell_vc_dict["axo_ids"])
            vc_amount_density[i] = cell_vc_amount / axon_length
            cell_sum_vc_vol = np.sum(cell_vc_dict["vc_sizes"])
            vc_vol_density[i] = cell_sum_vc_vol / axon_length
            syn_vc_amount_density[i] = cell_vc_amount/ cell_synapse_amount
            syn_vc_size_density[i] = cell_sum_vc_vol/ cell_sum_syn_sizes

        nonzero_inds = synapse_amount_density > 0
        ct_median_radius = ct_median_radius[nonzero_inds]
        ct_max_radius = ct_max_radius[nonzero_inds]
        ct_min_radius = ct_min_radius[nonzero_inds]
        synapse_amount_density = synapse_amount_density[nonzero_inds]
        synapse_size_density = synapse_size_density[nonzero_inds]
        vc_amount_density = vc_amount_density[nonzero_inds]
        vc_vol_density = vc_vol_density[nonzero_inds]
        syn_vc_amount_density = syn_vc_amount_density[nonzero_inds]
        syn_vc_size_density = syn_vc_size_density[nonzero_inds]
        cellids_nonzero = cellids[nonzero_inds].astype(int)

        ct_skel_dict = {"cell ids": cellids_nonzero, "axon_length": axon_length, "median radius": ct_median_radius, "max radius": ct_max_radius, "min radius": ct_min_radius}
        ct_axo_syn_vc_dict = {"syn amount density": synapse_amount_density, "syn size density": synapse_size_density, "vc amount density": vc_amount_density,
                              "vc volume density": vc_vol_density, "syn vc amount density": syn_vc_amount_density, "syn vc size density": syn_vc_size_density}
        if full_cells:
            if handpicked:
                write_obj2pkl("%s/%s_axo_skel_dict_hp.pkl" % (f_name, ct_dict[celltype]), ct_skel_dict)
                write_obj2pkl("%s/%s_axo_syn_vc_dict_hp.pkl" % (f_name, ct_dict[celltype]), ct_axo_syn_vc_dict)
            else:
                write_obj2pkl("%s/%s_axo_skel_dict_full.pkl" % (f_name, ct_dict[celltype]), ct_skel_dict)
                write_obj2pkl("%s/%s_axo_syn_vc_dict_full.pkl" % (f_name, ct_dict[celltype]), ct_axo_syn_vc_dict)
        else:
            write_obj2pkl("%s/%s_axo_skel_dict_frag.pkl" % (f_name, ct_dict[celltype]), ct_skel_dict)
            write_obj2pkl("%s/%s_axo_syn_vc_dict_frag.pkl" % (f_name, ct_dict[celltype]), ct_axo_syn_vc_dict)

        return ct_skel_dict, ct_axo_syn_vc_dict

    def tan_da_axon_comparison(min_axon_length = 200, syn_prob_thresh = 0.6, syn_size_thresh = 0.1, preprocessed_data = False):
        '''
        compare TAN fragments vs DA fragments vs full TAN cells (handpicked) in relation to their tickness (via skeleton node radius), their synapses and their vesicle clouds.
        Filter for fragments with enough length.
        Use ct axo analysis to iterate over cells. Plot results in violin plots, calculate statistics with a ranskum test.
        :param min_axon_length: minimal skeleton length of axon
        :param syn_prob_thresh: threshold for synapse probability
        :param syn_size_thresh: threshold for synapse size
        :param preprocessed_data: if True loads pickled dictionaries from ct_axo_analysis
        :return: ransksum_results
        '''
        start = time.time()
        foldername = ("u/arother/test_folder/210801_j0251v3_TAN_DA_comparison_minax_%.i_spt_%.2f_sst_%.2f" % (min_axon_length, syn_prob_thresh, syn_size_thresh))
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        log = initialize_logging('TAN_DA_comparison', log_dir=foldername + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        log.info("parameters: min_axon_length = %.i, syn_prob_threshold = %.2f, syn_size_threshold = %.2f" % (min_axon_length, syn_prob_thresh, syn_size_thresh))
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        # alternatively: dictionaries with values can be loaded here
        if preprocessed_data:
            log.info("Step 1/4: Load TAN dictionaries")
            tan_fragment_axo_skel_dict = load_pkl2obj("%s/%s/%s_axo_skel_dict_frag.pkl" % (foldername, ct_dict[5], ct_dict[5]))
            tan_fragment_axo_syn_vc_dict = load_pkl2obj("%s/%s/%s_axo_syn_vc_dict_frag.pkl" % (foldername, ct_dict[5], ct_dict[5]))
            tanftime = time.time() - start
            print("%.2f sec for loading TAN fragments" % tanftime)
            time_stamps.append(time.time())
            step_idents.append('load TAN fragments')
            log.info("Step 2/4: Load DA dictionaries")
            da_fragment_axo_skel_dict = load_pkl2obj("%s/%s/%s_axo_skel_dict_frag.pkl" % (foldername, ct_dict[1], ct_dict[1]))
            da_fragment_axo_syn_vc_dict = load_pkl2obj("%s/%s/%s_axo_syn_vc_dict_frag.pkl" % (foldername, ct_dict[1], ct_dict[1]))
            daftime = time.time() - tanftime
            print("%.2f sec for loading DA fragments" % daftime)
            time_stamps.append(time.time())
            step_idents.append('ploading DA fragments')
            log.info("Step 3/4: Load handpicked TAN cells")
            tan_hp_axo_skel_dict = load_pkl2obj("%s/%s/%s_axo_skel_dict_hp.pkl" % (foldername, ct_dict[5], ct_dict[5]))
            tan_hp_axo_syn_vc_dict = load_pkl2obj("%s/%s/%s_axo_syn_vc_dict_hp.pkl" % (foldername, ct_dict[5], ct_dict[5]))
            tanhptime = time.time() - daftime
            print("%.2f sec for loading handpicked TAN cells" % tanhptime)
            time_stamps.append(time.time())
            step_idents.append('loading handpicked TAN cells')
        else:
            # prepare vesicle cloud caches
            cached_vc_ids = sd_vcssv.ids
            cached_vc_rep_coords = sd_vcssv.load_cached_data("rep_coord")
            cached_vc_sizes = sd_vcssv.load_cached_data("size")
            # prepare synapse caches with synapse threshold for right celltype
            syn_prob = sd_synssv.load_cached_data("syn_prob")
            m = syn_prob > syn_prob_thresh
            m_ids = sd_synssv.ids[m]
            m_axs = sd_synssv.load_cached_data("partner_axoness")[m]
            m_axs[m_axs == 3] = 1
            m_axs[m_axs == 4] = 1
            m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
            m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
            m_sizes = sd_synssv.load_cached_data("mesh_area")[m] / 2
            m_rep_coords = sd_synssv.load_cached_data("rep_coord")[m]
            log.info("Step 1/4: Iterate over TAN fragments")
            tan_fragment_axo_skel_dict, tan_fragment_axo_syn_vc_dict = ct_axo_analysis(ssd, celltype=5, foldername= foldername, min_axon_len=min_axon_length,
                                                                                       syn_size_thresh=syn_size_thresh,
                                                                                       ignore_full_cells=True, cached_vc_ids=cached_vc_ids,
                                                                                       cached_vc_rep_coords=cached_vc_rep_coords, cached_vc_sizes= cached_vc_sizes, syn_ids=m_ids,
                                                                                       syn_sizes=m_sizes, syn_ssv_partners=m_ssv_partners, syn_rep_coords=m_rep_coords, syn_cts=m_cts,
                                                                                       syn_partner_axoness=m_axs)

            tanftime = time.time() - start
            print("%.2f sec for processing TAN fragments" % tanftime)
            time_stamps.append(time.time())
            step_idents.append('processing TAN fragments')
            log.info("Step 2/4: Iterate over DA fragments")
            da_fragment_axo_skel_dict, da_fragment_axo_syn_vc_dict = ct_axo_analysis(ssd, celltype=1, foldername= foldername,
                                                                                       min_axon_len=min_axon_length,
                                                                                       syn_size_thresh=syn_size_thresh, cached_vc_ids=cached_vc_ids,
                                                                                       cached_vc_rep_coords=cached_vc_rep_coords, cached_vc_sizes= cached_vc_sizes, syn_ids=m_ids,
                                                                                       syn_sizes=m_sizes, syn_ssv_partners=m_ssv_partners, syn_rep_coords=m_rep_coords, syn_cts=m_cts,
                                                                                     syn_partner_axoness=m_axs)
            daftime = time.time() - tanftime
            print("%.2f sec for processing DA fragments" % daftime)
            time_stamps.append(time.time())
            step_idents.append('processing DA fragments')
            log.info("Step 3/4: Iterate over handpicked TAN cells")
            tan_hp_axo_skel_dict, tan_hp_axo_syn_vc_dict = ct_axo_analysis(ssd, celltype=5, foldername= foldername,
                                                                                       min_axon_len=min_axon_length,
                                                                                       syn_size_thresh=syn_size_thresh, full_cells=True, handpicked=True, cached_vc_ids=cached_vc_ids,
                                                                                       cached_vc_rep_coords=cached_vc_rep_coords, cached_vc_sizes= cached_vc_sizes, syn_ids=m_ids,
                                                                                       syn_sizes=m_sizes, syn_ssv_partners=m_ssv_partners, syn_rep_coords=m_rep_coords, syn_cts=m_cts,
                                                                           syn_partner_axoness=m_axs)
            tanhptime = time.time() - daftime
            print("%.2f sec for processing handpicked TAN cells" % tanhptime)
            time_stamps.append(time.time())
            step_idents.append('processing handpicked TAN cells')

        log.info("Step 4/4: calculate statistics and plot results")
        tanf_amount = len(tan_fragment_axo_skel_dict["cell ids"])
        daf_amount = len(da_fragment_axo_skel_dict["cell ids"])
        tanhp_amount = len(tan_hp_axo_skel_dict["cell ids"])
        max_len = np.max([tanf_amount, daf_amount, tanhp_amount])
        cell_amounts = pd.DataFrame(index=range(max_len), columns=["DA fragment","TAN fragment", "TAN handpicked"])
        cell_amounts.loc["cell amount", "TAN fragment"] = tanf_amount
        cell_amounts.loc["cell amount", "TAN handpicked"] = tanhp_amount
        cell_amounts.loc["cell amount", "DA fragment"] = daf_amount
        cell_amounts.loc[0:tanf_amount, "TAN fragment"] = tan_fragment_axo_skel_dict["cell ids"]
        cell_amounts.loc[0:daf_amount, "DA fragment"] = da_fragment_axo_skel_dict["cell ids"]
        cell_amounts.loc[0:tanhp_amount, "TAN handpicked"] = tan_hp_axo_skel_dict["cell ids"]
        cell_amounts.to_csv("%s/cell_amounts.csv" % foldername)
        #ranksum tests
        ranksum_comparison = pd.DataFrame(columns=["TAN fragment vs DA fragment", "TAN fragment vs TAN handpicked", "DA fragment vs TAN handpicked"])
        #plot results as violinplot and boxplot
        for key in tan_fragment_axo_skel_dict.keys():
            if not "radius" in key:
                continue
            stats_tanf_daf, p_value_tanf_daf = ranksums(tan_fragment_axo_skel_dict[key], da_fragment_axo_skel_dict[key])
            ranksum_comparison.loc[key + " stats", "TAN fragment vs DA fragment"] = stats_tanf_daf
            ranksum_comparison.loc[key + " p value", "TAN fragment vs DA fragment"] = p_value_tanf_daf
            stats_tanf_tanhp, p_value_tanf_tanhp = ranksums(tan_fragment_axo_skel_dict[key], tan_hp_axo_skel_dict[key])
            ranksum_comparison.loc[key + " stats", "TAN fragment vs TAN handpicked"] = stats_tanf_tanhp
            ranksum_comparison.loc[key + " p value", "TAN fragment vs TAN handpicked"] = p_value_tanf_tanhp
            stats_daf_tanhp, p_value_daf_tanhp = ranksums(da_fragment_axo_skel_dict[key], tan_hp_axo_skel_dict[key])
            ranksum_comparison.loc[key + " stats", "DA fragment vs TAN handpicked"] = stats_daf_tanhp
            ranksum_comparison.loc[key + " p value", "DA fragment vs TAN handpicked"] = p_value_daf_tanhp
            results_df = pd.DataFrame(index=range(max_len), columns=["DA fragment", "TAN fragment", "TAN handpicked"])
            results_df.loc[0:tanf_amount-1, "TAN fragment"] = tan_fragment_axo_skel_dict[key]
            results_df.loc[0:daf_amount-1, "DA fragment"] = da_fragment_axo_skel_dict[key]
            results_df.loc[0:tanhp_amount-1, "TAN handpicked"] = tan_hp_axo_skel_dict[key]
            results_df.to_csv("%s/%s_results.csv" % (foldername, key))
            sns.violinplot(data = results_df ,inner="box")
            sns.stripplot(data =results_df,color="black", alpha=0.2)
            plt.title('%s' % key)
            plt.ylabel("radius in µm")
            filename = ("%s/%s_violin.png" % (foldername, key))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(data = results_df)
            plt.title('%s' % key)
            plt.ylabel("radius in µm")
            filename = ("%s/%s_box.png" % (foldername, key))
            plt.savefig(filename)
            plt.close()

        for key in tan_fragment_axo_syn_vc_dict.keys():
            stats_tanf_daf, p_value_tanf_daf = ranksums(tan_fragment_axo_syn_vc_dict[key], da_fragment_axo_syn_vc_dict[key])
            ranksum_comparison.loc[key + " stats", "TAN fragment vs DA fragment"] = stats_tanf_daf
            ranksum_comparison.loc[key + " p value", "TAN fragment vs DA fragment"] = p_value_tanf_daf
            stats_tanf_tanhp, p_value_tanf_tanhp = ranksums(tan_fragment_axo_syn_vc_dict[key],
                                                            tan_hp_axo_syn_vc_dict[key])
            ranksum_comparison.loc[key + " stats", "TAN fragment vs TAN handpicked"] = stats_tanf_tanhp
            ranksum_comparison.loc[key + " p value", "TAN fragment vs TAN handpicked"] = p_value_tanf_tanhp
            stats_daf_tanhp, p_value_daf_tanhp = ranksums(da_fragment_axo_syn_vc_dict[key], tan_hp_axo_syn_vc_dict[key])
            ranksum_comparison.loc[key + " stats", "DA fragment vs TAN handpicked"] = stats_daf_tanhp
            ranksum_comparison.loc[key + " p value", "DA fragment vs TAN handpicked"] = p_value_daf_tanhp
            results_df = pd.DataFrame(index=range(max_len), columns=["DA fragment", "TAN fragment", "TAN handpicked"])
            results_df.loc[0:tanf_amount - 1, "TAN fragment"] = tan_fragment_axo_syn_vc_dict[key]
            results_df.loc[0:daf_amount - 1, "DA fragment"] = da_fragment_axo_syn_vc_dict[key]
            results_df.loc[0:tanhp_amount - 1, "TAN handpicked"] = tan_hp_axo_syn_vc_dict[key]
            results_df.to_csv("%s/%s_results.csv" % (foldername, key))
            sns.violinplot(data=results_df, inner="box")
            sns.stripplot(data=results_df, color="black", alpha=0.2)
            plt.title('%s' % key)
            if "syn" in key:
                if "vc" in key:
                    if "size" in key:
                        plt.ylabel("vc to synapse size density [µm³/µm²]")
                    else:
                        plt.ylabel("vc to synapse amount density")
                else:
                    if "size" in key:
                        plt.ylabel("synapse size density [µm²/µm]")
                    else:
                        plt.ylabel("synapse amount density per µm")
            else:
                if "vol" in key:
                    plt.ylabel("vc volume density [µm³/µm]")
                else:
                    plt.ylabel("vc amount desnity per µm")
            filename = ("%s/%s_violin.png" % (foldername, key))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(data=results_df)
            plt.title('%s' % key)
            if "syn" in key:
                if "vc" in key:
                    if "size" in key:
                        plt.ylabel("vc to synapse size density [µm³/µm²]")
                    else:
                        plt.ylabel("vc to synapse amount density")
                else:
                    if "size" in key:
                        plt.ylabel("synapse size density [µm²/µm]")
                    else:
                        plt.ylabel("synapse amount density per µm")
            else:
                if "vol" in key:
                    plt.ylabel("vc volume density [µm³/µm]")
                else:
                    plt.ylabel("vc amount desnity per µm")
            filename = ("%s/%s_box.png" % (foldername, key))
            plt.savefig(filename)
            plt.close()

        ranksum_comparison.to_csv("%s/ranksum_results.csv" % foldername)

        stattime = time.time() - tanhptime
        print("%.2f sec for plotting results and calculating stats" % stattime)
        time_stamps.append(time.time())
        step_idents.append('plotting results and calculating stats with ranksum')
        log.info("TAN DA comparison finished")

    tan_da_axon_comparison(preprocessed_data = False, min_axon_length=100)










