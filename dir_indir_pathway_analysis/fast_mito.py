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
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

    def mito_prox_dis(sso, compartment = 0, radius = 50, min_comp_len = 100):
        """
        sorts mitochonria into proximal (<50 µm) or distal (>100 µm) and saves parameter like mitochondrium volume density, mitochondrium length density, amount of mitochondria and average distance to soma
        compartments: 0 = dendrite, 1 = axon, soma = 2
        """
        if compartment == 0:
            unwanted_compartment = 1
        elif compartment == 1:
            unwanted_compartment = 0
        else:
            print('unknown compartment value %.i. Compartment has to be either 0 or 1' % compartment)
            raise ValueError
        sso.load_skeleton()
        non_compartment_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
        compartment_graph = g.copy()
        compartment_graph.remove_nodes_from(non_compartment_inds)
        if compartment == 1:
            amount_compartment_subgraphs = 1
            #warning: if false positives make more than one axon, this default will overestimate the amount of proximal or distal mitos!
        else:
            amount_compartment_subgraphs = len(list(nx.connected_component_subgraphs(compartment_graph)))
        compartment_length = compartment_graph.size(weight = "weight") / 1000 #in µm
        if compartment_length < min_comp_len:
            return np.zeros(7), np.zeros(7), np.zeros(7)
        kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"]*sso.scaling)
        mis = sso.mis
        prox_ids = np.zeros(len(mis))
        dist_ids = np.zeros(len(mis))
        all_ids = np.zeros(len(mis))
        #mitochrondrial index and density (every 10 µm)
        prox_pathlength = np.zeros(len(mis))
        dist_pathlength = np.zeros(len(mis))
        all_pathlength = np.zeros(len(mis))
        prox_volume = np.zeros(len(mis))
        dist_volume = np.zeros(len(mis))
        all_volume = np.zeros(len(mis))
        prox_milength = np.zeros(len(mis))
        dist_milength = np.zeros(len(mis))
        all_milength = np.zeros(len(mis))
        for i, mi in enumerate(tqdm(mis)):
            #from syconn.reps.super_segmentation_object attr_for_coors
            close_node_ids = kdtree.query_ball_point(mi.rep_coord*sso.scaling, radius)
            radfactor = 1
            while len(close_node_ids) <= 2:
                close_node_ids = kdtree.query_ball_point(mi.rep_coord*sso.scaling, radius*(2**radfactor))
                radfactor += 1
                if radfactor == 5:
                    break
            axo = np.array(sso.skeleton["axoness_avg10000"][close_node_ids])
            axo[axo == 3] = 1
            axo[axo == 4] = 1
            if np.all(axo == unwanted_compartment) or np.all(axo == 2):
                continue
            if len(np.where(axo == compartment)[0]) < len(axo)/2:
                continue
            mi_rep_coord_pathlength = sso.shortestpath2soma([mi.rep_coord * sso.scaling])[0] / 1000 #in µm
            if mi_rep_coord_pathlength == 0:
                continue
            mi_diff = mi.mesh_bb[1] - mi.mesh_bb[0]
            mi_length = np.linalg.norm(mi_diff)/1000. #in µm
            all_ids[i] = mi.id
            all_pathlength[i] = mi_rep_coord_pathlength
            all_volume[i] = mi.size
            all_milength[i] = mi_length
            mi_start_pathlength = mi_rep_coord_pathlength - mi_length / 2
            mi_end_pathlength = mi_rep_coord_pathlength + mi_length / 2
            if mi_start_pathlength < 50:
                prox_ids[i] = mi.id
                prox_pathlength[i] = mi_rep_coord_pathlength
                prox_milength[i] = mi_length
                prox_volume[i] = mi.size
            elif mi_end_pathlength > 100:
                dist_ids[i] = mi.id
                dist_pathlength[i] = mi_rep_coord_pathlength
                dist_milength[i] = mi_length
                dist_volume[i] = mi.size
        prox_inds = prox_ids > 0
        dist_inds = dist_ids > 0
        all_inds = all_ids > 0
        if np.all(all_inds == False):
            all_mito_parameters_cell = np.zeros(7)
            prox_mito_parameters_cell = np.zeros(7)
            dist_mito_parameters_cell = np.zeros(7)
        else:
            all_ids = all_ids[all_inds]
            all_milength = all_milength[all_inds]
            all_pathlength = all_pathlength[all_inds]
            all_volume = all_volume[all_inds]
            prox_ids = prox_ids[prox_inds]
            prox_milength = prox_milength[prox_inds]
            prox_pathlength = prox_pathlength[prox_inds]
            prox_volume = prox_volume[prox_inds]
            dist_ids = dist_ids[dist_inds]
            dist_milength = dist_milength[dist_inds]
            dist_pathlength = dist_pathlength[dist_inds]
            dist_volume = dist_volume[dist_inds]
            mito_amount = len(all_ids)
            average_pathlength = np.mean(all_pathlength)
            average_volume = np.mean(all_volume)
            average_length = np.mean(all_milength)
            volume_density = np.sum(all_volume)/compartment_length
            length_density = np.sum(all_milength)/compartment_length
            amount_density = mito_amount/compartment_length
            all_mito_parameters_cell = [mito_amount, average_pathlength, average_volume, average_length, volume_density, length_density, amount_density]
            prox_mito_amount = len(prox_ids)
            prox_average_pathlength = np.mean(prox_pathlength)
            prox_average_volume = np.mean(prox_volume)
            prox_average_length = np.mean(prox_milength)
            prox_volume_density = np.sum(prox_volume) / (50 * amount_compartment_subgraphs)
            prox_length_density = np.sum(prox_milength) / (50 * amount_compartment_subgraphs)
            prox_amount_density = prox_mito_amount / (50 * amount_compartment_subgraphs)
            prox_mito_parameters_cell = [prox_mito_amount, prox_average_pathlength, prox_average_volume, prox_average_length, prox_volume_density, prox_length_density, prox_amount_density]
            dist_mito_amount = len(dist_ids)
            dist_average_pathlength = np.mean(dist_pathlength)
            dist_average_volume = np.mean(dist_volume)
            dist_average_length = np.mean(dist_milength)
            dist_volume_density = np.sum(dist_volume) / (compartment_length - (100 * amount_compartment_subgraphs))
            dist_length_density = np.sum(dist_milength) / (compartment_length - (100 * amount_compartment_subgraphs))
            dist_amount_density = dist_mito_amount / (compartment_length - (100 * amount_compartment_subgraphs))
            dist_mito_parameters_cell = [dist_mito_amount, dist_average_pathlength, dist_average_volume, dist_average_length, dist_volume_density,dist_length_density, dist_amount_density]
        return all_mito_parameters_cell, prox_mito_parameters_cell, dist_mito_parameters_cell


    def ct_mito_analysis(ssd, celltype, compartment=0, min_comp_len = 100):
        """
        analysis of mitochondria length in proximal [until 50 µm] and distal[from 100 µm onwards] dendrites.
        Parameters that will be plotted are mitochondrial lenth (norm of bounding box diameter, might need a more exact measurement later),
        mitochondrial volume(mi.size), mitochondrial density (amount of mitochondria per µm axon/dendrite), mitochondrial index (mitochondrial
        length per µm axon/dendritic length) (see chandra et al., 2019 for comparison)
        :param ssd: SuperSegmentationDataset
        :param celltype: 0:"STN", 1:"DA", 2:"MSN", 3:"LMAN", 4:"HVC", 5:"GP", 6:"FS", 7:"TAN", 8:"INT"
        celltypes: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
                compartment: 0 = dendrite, 1 = axon, 2 = soma
        :return: mitochondrial parameters in proximal and sital dendrites
        """
        start = time.time()
        f_name = ("u/arother/test_folder/210607_j0251v3_fastmito_MSN_mcl%i" % min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('mitchondrial_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        # cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        mito_amount = np.zeros(len(cellids))
        average_pathlength = np.zeros(len(cellids))
        average_volume = np.zeros(len(cellids))
        average_length = np.zeros(len(cellids))
        volume_density = np.zeros(len(cellids))
        length_density = np.zeros(len(cellids))
        amount_density = np.zeros(len(cellids))
        prox_mito_amount = np.zeros(len(cellids))
        prox_average_pathlength = np.zeros(len(cellids))
        prox_average_volume = np.zeros(len(cellids))
        prox_average_length = np.zeros(len(cellids))
        prox_volume_density = np.zeros(len(cellids))
        prox_length_density = np.zeros(len(cellids))
        prox_amount_density = np.zeros(len(cellids))
        dist_mito_amount = np.zeros(len(cellids))
        dist_average_pathlength = np.zeros(len(cellids))
        dist_average_volume = np.zeros(len(cellids))
        dist_average_length = np.zeros(len(cellids))
        dist_volume_density = np.zeros(len(cellids))
        dist_length_density = np.zeros(len(cellids))
        dist_amount_density = np.zeros(len(cellids))
        comp_dict = {1: 'axons', 0: 'dendrites'}
        log.info('Step 1/2 generating mitochondrial parameters per cell')
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            all_mito_params_cell, prox_mito_params_cell, dist_mito_params_cell = mito_prox_dis(cell, compartment, min_comp_len=min_comp_len)
            if np.all(all_mito_params_cell == 0):
                continue
            mito_amount[i] = all_mito_params_cell[0]
            average_pathlength[i] = all_mito_params_cell[1]
            average_volume[i] = all_mito_params_cell[2]
            average_length[i] = all_mito_params_cell[3]
            volume_density[i] = all_mito_params_cell[4]
            length_density[i] = all_mito_params_cell[5]
            amount_density[i] = all_mito_params_cell[6]
            prox_mito_amount[i] = prox_mito_params_cell[0]
            prox_average_pathlength [i] = prox_mito_params_cell[1]
            prox_average_volume[i] = prox_mito_params_cell[2]
            prox_average_length[i] = prox_mito_params_cell[3]
            prox_volume_density[i] = prox_mito_params_cell[4]
            prox_length_density[i] = prox_mito_params_cell[5]
            prox_amount_density[i] = prox_mito_params_cell[6]
            dist_mito_amount[i] = dist_mito_params_cell[0]
            dist_average_pathlength[i] = dist_mito_params_cell[1]
            dist_average_volume[i] = dist_mito_params_cell[2]
            dist_average_length[i] = dist_mito_params_cell[3]
            dist_volume_density[i] = dist_mito_params_cell[4]
            dist_length_density[i] = dist_mito_params_cell[5]
            dist_amount_density[i] = dist_mito_params_cell[6]

        all_mito_inds = mito_amount > 0
        prox_mito_inds = prox_mito_amount > 0
        dist_mito_inds = dist_mito_amount > 0
        mito_amount = mito_amount[all_mito_inds]
        average_pathlength = average_pathlength[all_mito_inds]
        average_volume = average_volume[all_mito_inds]
        average_length = average_length[all_mito_inds]
        volume_density = volume_density[all_mito_inds]
        length_density = length_density[all_mito_inds]
        amount_density = amount_density[all_mito_inds]
        prox_mito_amount = prox_mito_amount[prox_mito_inds]
        prox_average_pathlength = prox_average_pathlength[prox_mito_inds]
        prox_average_volume = prox_average_volume[prox_mito_inds]
        prox_average_length = prox_average_length[prox_mito_inds]
        prox_volume_density = prox_volume_density[prox_mito_inds]
        prox_length_density = prox_length_density[prox_mito_inds]
        prox_amount_density = prox_amount_density[prox_mito_inds]
        dist_mito_amount = dist_mito_amount[dist_mito_inds]
        dist_average_pathlength = dist_average_pathlength[dist_mito_inds]
        dist_average_volume = dist_average_volume[dist_mito_inds]
        dist_average_length = dist_average_length[dist_mito_inds]
        dist_volume_density = dist_volume_density[dist_mito_inds]
        dist_length_density = dist_length_density[dist_mito_inds]
        dist_amount_density = dist_amount_density[dist_mito_inds]

        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")

        sns.distplot(mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_mitoamount_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_mitoamount_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]) )
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitolength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s'% (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitolength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitovol_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s'% (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitovol_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitopathlength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average distance from soma of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitopathlength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average distance from soma of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_voldenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_voldenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_lendenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_lendenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_amdenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s'% (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_amdenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        #for proximal
        sns.distplot(prox_mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxmitoamount_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxmitoamount_%s_norm.png' %(f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitolength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitolength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitovol_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitovol_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitopathlength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average distance from soma of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitopathlength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict))
        plt.title('Average distance from soma of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxvoldenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxvoldenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxlendenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxlendenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxamdenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(prox_amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxamdenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s < 50 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        #for distal
        sns.distplot(dist_mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distmitoamount_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distmitoamount_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitolength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distavgmitolength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average length of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitovol_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distavgmitovol_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average volume of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitopathlength_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average distance from soma of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_average_pathlength, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,norm_hist=True)
        avg_filename = ('%s/%s_distavgmitopathlength_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Average distance from soma of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distvoldenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distvoldenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Volume density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distlendenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distlendenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Length density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distamdenmito_%s.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(dist_amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distamdenmito_%s_norm.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Amount of mitochondria per µm in %s %s > 100 µm from soma' % (ct_dict[celltype],  comp_dict[compartment]))
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        #plot log
        mito_amount = mito_amount[mito_amount > 0]
        sns.distplot(np.log10(mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                    kde=False)
        avg_filename = ('%s/%s_mitoamount_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_mitoamount_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        average_length = average_length[average_length > 0]
        sns.distplot(np.log10(average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitolength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average length of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitolength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average length of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        average_volume = average_volume[average_volume > 0]
        sns.distplot(np.log10(average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitovol_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average volume of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitovol_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average volume of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        average_pathlength = average_pathlength[average_pathlength > 0]
        sns.distplot(np.log10(average_pathlength), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_avgmitopathlength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average distance from soma of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(average_pathlength), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_avgmitopathlength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average distance from soma of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('Log pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        volume_density = volume_density[volume_density > 0]
        sns.distplot(np.log10(volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_voldenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log volume density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_voldenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log volume density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        length_density = length_density[length_density > 0]
        sns.distplot(np.log10(length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_lendenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log length density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_lendenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log length density of mitochondria in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        amount_density = amount_density[amount_density > 0]
        sns.distplot(np.log10(amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_amdenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria per µm in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_amdenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria per µm in %s %s' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        # for proximal
        prox_mito_amount = prox_mito_amount[prox_mito_amount > 0]
        sns.distplot(np.log10(prox_mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxmitoamount_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxmitoamount_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_average_length = prox_average_length[prox_average_length > 0]
        sns.distplot(np.log10(prox_average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitolength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average length of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitolength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average length of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_average_volume = prox_average_volume[prox_average_volume > 0]
        sns.distplot(np.log10(prox_average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitovol_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average volume of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitovol_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average volume of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_average_pathlength = prox_average_pathlength[prox_average_pathlength > 0]
        sns.distplot(np.log10(prox_average_pathlength),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxavgmitopathlength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average distance from soma of mitochondria in %s %s < 50 µm from soma' % (
        ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_average_pathlength),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxavgmitopathlength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict))
        plt.title('Log average distance from soma of mitochondria in %s %s < 50 µm from soma' % (
        ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_volume_density = prox_volume_density[prox_volume_density > 0]
        sns.distplot(np.log10(prox_volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxvoldenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log volume density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxvoldenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log volume density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('Log volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_length_density = prox_length_density[prox_length_density > 0]
        sns.distplot(np.log10(prox_length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxlendenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log length density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxlendenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log length density of mitochondria in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        prox_amount_density = prox_amount_density[prox_amount_density > 0]
        sns.distplot(np.log10(prox_amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_proxamdenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log amount of mitochondria per µm in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(prox_amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_proxamdenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log amount of mitochondria per µm in %s %s < 50 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        # for distal
        dist_mito_amount = dist_mito_amount[dist_mito_amount> 0]
        sns.distplot(np.log10(dist_mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distmitoamount_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_mito_amount), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distmitoamount_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log amount of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_average_length = dist_average_length[dist_average_length > 0]
        sns.distplot(np.log10(dist_average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitolength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log verage length of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_average_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distavgmitolength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average length of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average length in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_average_volume = dist_average_volume[dist_average_volume > 0]
        sns.distplot(np.log10(dist_average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitovol_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average volume of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_average_volume), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distavgmitovol_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log average volume of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_average_pathlength = dist_average_pathlength[dist_average_pathlength > 0]
        sns.distplot(np.log10(dist_average_pathlength),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distavgmitopathlength_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average distance from soma of mitochondria in %s %s > 100 µm from soma' % (
        ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_average_pathlength),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distavgmitopathlength_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title('Log average distance from soma of mitochondria in %s %s > 100 µm from soma' % (
        ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_volume_density = dist_volume_density[dist_volume_density > 0]
        sns.distplot(np.log10(dist_volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distvoldenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log volume density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_volume_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distvoldenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log volume density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_length_density = dist_length_density[dist_length_density > 0]
        sns.distplot(np.log10(dist_length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distlendenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log length density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_length_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distlendenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log length density of mitochondria in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        dist_amount_density = dist_amount_density[dist_amount_density > 0]
        sns.distplot(np.log10(dist_amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/%s_distamdenmito_%s_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log amount of mitochondria per µm in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(np.log10(dist_amount_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/%s_distamdenmito_%s_norm_log.png' % (f_name, comp_dict[compartment], ct_dict[celltype]))
        plt.title(
            'Log amount of mitochondria per µm in %s %s > 100 µm from soma' % (ct_dict[celltype], comp_dict[compartment]))
        plt.xlabel('log amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('fast mito for %s in %s finished' % (comp_dict[compartment], ct_dict[celltype]))


    ct_mito_analysis(ssd, celltype=2, compartment=1, min_comp_len=200)
    ct_mito_analysis(ssd, celltype=2, compartment=0, min_comp_len=200)
