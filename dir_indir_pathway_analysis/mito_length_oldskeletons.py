if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    import pandas as pd
    import os as os
    import scipy
    import time
    from tqdm import tqdm
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

    def mito_prox_dis(sso, compartment = 0, radius = 50):
        """
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
        kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"] * sso.scaling)
        mis = sso.mis
        prox_list = np.zeros(len(mis))
        dist_list = np.zeros(len(mis))
        prox_d = np.zeros(len(mis))
        dist_d = np.zeros(len(mis))
        #mitochrondrial index and density (every 10 µm)
        prox_num = np.zeros(5) #proximal until 50µm
        prox_den = np.zeros(5)
        dist_num = np.zeros(50) #from 100 to 500 µm
        dist_den = np.zeros(50)
        prox_pathlength = np.zeros(len(mis))
        dist_pathlength = np.zeros(len(mis))
        prox_vol = np.zeros(len(mis))
        dist_vol = np.zeros(len(mis))
        prox_bin_abs1 = np.zeros(8)
        prox_bin_abs1_5 = np.zeros(6)
        prox_bin_abs2 = np.zeros(5)
        dist_bin_abs1 = np.zeros(8)
        dist_bin_abs1_5 = np.zeros(6)
        dist_bin_abs2 = np.zeros(5)
        for i, mi in enumerate(tqdm(mis)):
            #from syconn.reps.super_segmentation_object attr_for_coors
            close_node_ids = kdtree.query_ball_point(mi.rep_coord * sso.scaling, radius)
            radfactor = 1
            while len(close_node_ids) <= 2:
                close_node_ids = kdtree.query_ball_point(mi.rep_coord * sso.scaling, radius*(2**radfactor))
                radfactor += 1
                if radfactor == 5:
                    break
            axo = np.array(sso.skeleton["axoness_avg10000"][close_node_ids])
            axo[axo == 3] = 1
            axo[axo == 4] = 1
            if np.all(axo) == unwanted_compartment or np.all(axo) == 2:
                continue
            if len(np.where(axo == compartment)[0]) < len(axo)/2:
                continue
            dists = sso.shortestpath2soma(mi.mesh_bb)
            distproxdist = sso.shortestpath2soma([mi.rep_coord * sso.scaling])[0]
            if distproxdist == 0:
                continue
            dist1 = dists[0]/1000. #in µm, kimimaro skeletons in physical parameters
            dist2 = dists[1]/1000.
            mi_diff = mi.mesh_bb[1] - mi.mesh_bb[0]
            mi_length = np.linalg.norm(mi_diff)/1000. #in µm
            if dist1 <= dist2:
                dist = dist1
                start_bin = int(dist1 / 10)
                end_bin = int(dist2 / 10)
                start_dist = dist1
                end_dist = dist2
            else:
                dist = dist2
                start_bin = int(dist2 / 10)
                end_bin = int(dist1 / 10)
                start_dist = dist2
                end_dist = dist1
            if distproxdist <= 50:
                prox_list[i] = mi.id
                prox_d[i] = mi_length
                prox_pathlength[i] = dist
                prox_vol[i] = mi.size
                prox_num[start_bin] += 1
                if start_bin == end_bin:
                    prox_den[start_bin] += mi_length
                elif start_bin != end_bin: #if spanning multiple bins then counted in every of them
                    span = end_bin - start_bin
                    if span == 1:
                        per_length = (end_dist - end_bin * 10) / mi_length
                        prox_den[start_bin] += mi_length * (1 - per_length)
                        if end_bin <= 4:
                            prox_num[end_bin] += 1
                            prox_den[end_bin] += mi_length * per_length
                    elif span > 1:
                        if span > 4:
                            span = 4
                            end_bin = 4
                        per_length_start = ((start_bin+1)*10 - start_dist/mi_length)
                        prox_den[start_bin] += mi_length*per_length_start
                        for i in range(1, span):
                            if start_bin + i < 4:
                                prox_num[start_bin + i] += 1
                                prox_den[start_bin + i] += mi_length*(10/mi_length)
                                if i == span:
                                    prox_den[end_bin] += mi_length*((end_dist - end_bin*10)/mi_length)
                            else:
                                if end_bin == 4:
                                    prox_den[end_bin] += mi_length * ((end_dist- end_bin * 10) / mi_length)
                                    prox_num[end_bin] += 1
                                else:
                                    prox_den[4] += 1
                                    prox_den[4] += mi_length*(10/mi_length)
                prox_1_i = int(mi_length)
                prox_1_5_i = int(mi_length/1.5)
                prox_2_i = int(mi_length/2)
                if prox_1_i > 7:
                    prox_1_i = 7
                if prox_1_5_i > 5:
                    prox_1_5_i = 5
                if prox_2_i > 4:
                    prox_2_i = 4
                prox_bin_abs1[prox_1_i] += 1
                prox_bin_abs1_5[prox_1_5_i] += 1
                prox_bin_abs2[prox_2_i] += 1
            elif distproxdist >= 100:
                dist_list[i] = mi.id
                dist_d[i] = mi_length
                dist_pathlength[i] = dist
                dist_vol[i] = mi.size
                start_bin = start_bin - 10  # starting at 100 µm
                end_bin = end_bin - 10
                dist_num[start_bin] += 1
                if start_bin == end_bin:
                    dist_den[start_bin] += mi_length
                elif start_bin != end_bin:  # if spanning multiple bins then counted in every of them
                    span = end_bin - start_bin
                    if span == 1:
                        dist_num[end_bin] += 1
                        per_length = (dist2 - end_bin * 10) / mi_length
                        dist_den[start_bin] += mi_length * (1 - per_length)
                        dist_den[end_bin] += mi_length * per_length
                    elif span > 1:
                        per_length_start = ((start_bin + 1) * 10 - dist1) / mi_length
                        dist_den[start_bin] += mi_length * per_length_start
                        for i in range(1, span):
                            dist_num[start_bin + i] += 1
                            dist_den[start_bin + i] += mi_length * (10 / mi_length)
                            if i == span:
                                dist_den[end_bin] += mi_length * ((dist2 - end_bin * 10) / mi_length)
                    dist_1_i = int(mi_length)
                    dist_1_5_i = int(mi_length / 1.5)
                    dist_2_i = int(mi_length / 2)
                    if dist_1_i > 7:
                        dist_1_i = 7
                    if dist_1_5_i > 5:
                        dist_1_5_i = 5
                    if dist_2_i > 4:
                        dist_2_i = 4
                    dist_bin_abs1[dist_1_i] += 1
                    dist_bin_abs1_5[dist_1_5_i] += 1
                    dist_bin_abs2[dist_2_i] += 1
            percentage = 100 * (i / len(mis))
            print("%.2f percent, %4i mitos" % (percentage, len(mis)))
        dist_id_array = dist_list[dist_list > 0]
        prox_id_array = prox_list[prox_list > 0]
        prox_d_array = prox_d[prox_d > 0]
        dist_d_array = dist_d[dist_d > 0]
        prox_pathlength = prox_pathlength[prox_pathlength > 0]
        dist_pathlength = dist_pathlength[dist_pathlength > 0]
        prox_num = prox_num[prox_num > 0]
        prox_den = prox_den[prox_den > 0]
        prox_vol = prox_vol[prox_vol > 0]
        dist_num = dist_num[dist_num > 0]
        dist_den = dist_den[dist_den > 0]
        dist_vol = dist_vol[dist_vol > 0]
        sum_prox_mis = np.sum(np.hstack(prox_bin_abs1))
        sum_dist_mis = np.sum(np.hstack(dist_bin_abs1))
        if sum_prox_mis > 0:
            prox_bin_rel1 = prox_bin_abs1/sum_prox_mis
            prox_bin_rel1_5 = prox_bin_abs1_5/sum_prox_mis
            prox_bin_rel2 = prox_bin_abs2/sum_prox_mis
        else:
            prox_bin_rel1 = 0
            prox_bin_rel1_5 = 0
            prox_bin_rel2 = 0
        if sum_dist_mis > 0:
            dist_bin_rel1 = dist_bin_abs1/sum_dist_mis
            dist_bin_rel1_5 = dist_bin_abs1_5/sum_dist_mis
            dist_bin_rel2 = dist_bin_abs2/sum_dist_mis
        else:
            dist_bin_rel1 = 0
            dist_bin_rel1_5 = 0
            dist_bin_rel2 = 0
        prox_comb = [prox_d_array, prox_id_array, prox_pathlength, prox_vol, prox_num,prox_den, prox_list,  prox_bin_abs1, prox_bin_abs1_5, prox_bin_abs2, prox_bin_rel1, prox_bin_rel1_5, prox_bin_rel2]
        dist_comb = [dist_d_array, dist_id_array, dist_pathlength, dist_vol, dist_num, dist_den, dist_list, dist_bin_abs1, dist_bin_abs1_5, dist_bin_abs2, dist_bin_rel1, dist_bin_rel1_5, dist_bin_rel2]
        return prox_comb, dist_comb

    def ct_mito_analysis(ssd, celltype, compartment=0):
        """
        analysis of mitochondria length in proximal [until 50 µm] and distal[from 100 µm onwards] dendrites.
        Parameters that will be plotted are mitochondrial lenth (norm of bounding box diameter, might need a more exact measurement later),
        mitochondrial volume(mi.size), mitochondrial density (amount of mitochondria per 10 µm axon/dendrite), mitochondrial index (mitochondrial
        length per 10 µm axon/dendritic length), also bins with mitochondrial length will be compared (see chandra et al., 2019 for comparison)
        :param ssd: SuperSegmentationDataset
        :param celltype: 0:"STN", 1:"DA", 2:"MSN", 3:"LMAN", 4:"HVC", 5:"GP", 6:"FS", 7:"TAN", 8:"INT"
        celltypes: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
                compartment: 0 = dendrite, 1 = axon, 2 = soma
        :return: mitochondrial parameters in proximal and sital dendrites
        """
        # per celltype function
        # for cell == celltype:
        # sholl_analysis(cell)
        #TO DO: mitochondrial index(mitochondrial length per 10 µm length), mitochondrial density(amount per 10 µm length)
        #chandra et al. 2019: comparison of mitochondria in each size range normalized to total number of mitochondria in each dendrite
        start = time.time()
        f_name = ("u/arother/test_folder/2010504_j0251v3_mitoMSN_analysis")
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('mitchondrial_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",10: "NGF"}
        #cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        #gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        #gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        #cellids = np.hstack(np.array([gpeids, gpiids]))
        #ct_dict = {6: "GP", 7: "GP"}
        #celltype = celltype[0]
        prox_mito_dict = dict()
        dist_mito_dict = dict()
        # length/diameter of mitochondria
        mean_prox_arr = np.zeros(len(cellids))
        med_prox_arr = np.zeros(len(cellids))
        mean_dist_arr = np.zeros(len(cellids))
        med_dist_arr = np.zeros(len(cellids))
        sum_prox = np.zeros((len(cellids), 1000))
        sum_dist = np.zeros((len(cellids), 1000))
        # volumen of mitochondria
        mean_prox_vol = np.zeros(len(cellids))
        med_prox_vol = np.zeros(len(cellids))
        mean_dist_vol = np.zeros(len(cellids))
        med_dist_vol = np.zeros(len(cellids))
        sum_prox_vol = np.zeros((len(cellids), 1000))
        sum_dist_vol = np.zeros((len(cellids), 1000))
        sum_prox_miids = np.zeros((len(cellids), 1000))
        sum_dist_miids = np.zeros((len(cellids), 3000))
        # density/ index of mitochondria
        mean_prox_den_arr = np.zeros(len(cellids))
        mean_prox_in_arr = np.zeros(len(cellids))
        mean_dist_den_arr = np.zeros(len(cellids))
        mean_dist_in_arr = np.zeros(len(cellids))
        med_prox_den_arr = np.zeros(len(cellids))
        med_prox_in_arr = np.zeros(len(cellids))
        med_dist_den_arr = np.zeros(len(cellids))
        med_dist_in_arr = np.zeros(len(cellids))
        prox_am = np.zeros(len(cellids))
        dist_am = np.zeros(len(cellids))
        prox_cellid = np.zeros(len(cellids))
        dist_cellid = np.zeros(len(cellids))
        bin_keys = ["abs1", "abs1_5", "abs2", "rel1", "rel1_5", "rel2"]
        bin_keys2 = np.array([8, 6, 5, 8, 6, 5])
        prox_bin_dict = {i: dict() for i in bin_keys}
        dist_bin_dict = {i: dict() for i in bin_keys}
        for ki, k in enumerate(prox_bin_dict.keys()):
            prox_bin_dict[k] = {i: np.zeros(len(cellids)) for i in range(bin_keys2[ki])}
        for ki, k in enumerate(dist_bin_dict.keys()):
            dist_bin_dict[k] = {i: np.zeros(len(cellids)) for i in range(bin_keys2[ki])}
        comp_dict = {1: 'axons', 0: 'dendrites'}
        log.info('Step 1/2 generating mitochondrial parameters per cell')
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            """
            cell.load_skeleton()
            unique_preds = np.unique(cell.skeleton['axoness_avg10000'])
            if not (0 in unique_preds and (1 in unique_preds or 3 in unique_preds or 4 in unique_preds) and 2 in unique_preds):
                continue
            """
            prox_mito_dict[cell.id] = dict()
            dist_mito_dict[cell.id] = dict()
            prox_list, dist_list  = mito_prox_dis(cell, compartment)
            #prox_list = [prox_d_array, prox_id_array, prox_pathlength, prox_vol, prox_num, prox_den, prox_bin_abs1, prox_bin_abs1_5, prox_bin_abs2, prox_bin_rel1, prox_bin_rel1_5, prox_bin_rel2]
            prox_mito_dict[cell.id] = {"diameter": prox_list[0], "id": prox_list[1], "pathlength":prox_list[2], "vol": prox_list[3],
                                       "density":prox_list[4], "index": prox_list[5], "mi_ids": prox_list[6], "bin_abs1": prox_list[7], "bin_abs1_5":prox_list[8],
                                       "bin_abs2":prox_list[9], "bin_rel1": prox_list[10], "bin_rel1_5": prox_list[11], "bin_rel2": prox_list[12]}
            dist_mito_dict[cell.id] = {"diameter": dist_list[0], "id": dist_list[1], "pathlength": dist_list[2],
                                       "vol": dist_list[3],"density": dist_list[4], "index": dist_list[5], "mi_ids":dist_list[6], "bin_abs1": dist_list[7],
                                       "bin_abs1_5": dist_list[8],"bin_abs2": dist_list[9], "bin_rel1": dist_list[10], "bin_rel1_5": dist_list[11],
                                       "bin_rel2": dist_list[12]}
            prox_cd = prox_mito_dict[cell.id]
            dist_cd = dist_mito_dict[cell.id]
            prox_ds = prox_cd["diameter"]
            dist_ds = dist_cd["diameter"]
            prox_vol = prox_cd["vol"]
            dist_vol = dist_cd["vol"]
            prox_den = prox_cd["density"]
            dist_den = dist_cd["density"]
            prox_ind = prox_cd["index"]
            dist_ind = dist_cd["index"]
            pmi_ids = prox_cd["mi_ids"]
            dmi_ids = dist_cd["mi_ids"]
            # plot mitos_per cell
            if len(np.nonzero(prox_ds)) == 0 and len(np.nonzero(dist_ds)) == 0:
                print("cell %i has no suitable mitochondria" % cell.id)
                percentage = 100 * (i / len(cellids))
                print("%.2f percent, cell %.4i done" % (percentage, cell.id))
                continue
            columns = ['pathlength in µm', ('count of %s' % comp_dict[compartment])]
            foldername = '%s/mito_percell_%s' % (f_name, ct_dict[celltype])
            if not os.path.exists(foldername):
                os.mkdir(foldername)
            if len(np.nonzero(prox_ds)[0]) > 0:
                sns.distplot(prox_ds, kde=False)
                plt.title('mitochondrium diameter in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('diameter in µm')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_l_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_vol, kde=False)
                plt.title('mitochondrium vol in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('vol in voxel')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_v_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_den, kde=False)
                plt.title('mitochondrium density in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('amount of mitchondria per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_den_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_ind, kde=False)
                plt.title('mitochondrium index in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrial lenght per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_ind_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                #abs and relative bins
                sns.barplot(prox_cd["bin_abs1"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(prox_cd["bin_abs1_5"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1.5 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(prox_cd["bin_abs2"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 2 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                if type(prox_cd["bin_rel1"]) == np.ndarray:
                    sns.barplot(prox_cd["bin_rel1"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(prox_cd["bin_rel1_5"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1.5 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(prox_cd["bin_rel2"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 2 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    # length/diameter
                prox_cellid[i] = cell.id
                sum_prox_miids[i] = pmi_ids
                med_prox = np.median(prox_ds)
                mean_prox = np.mean(prox_ds)
                med_prox_arr[i] = med_prox
                mean_prox_arr[i] = mean_prox
                prox_mito_dict[cell.id]["meanl"] = mean_prox
                prox_mito_dict[cell.id]["medianl"] = med_prox
                sum_prox[i, 0:len(prox_ds)] = prox_ds
                sum_prox_vol[i, 0:len(prox_ds)] = prox_vol
                prox_am[i] = len(prox_ds)
                # volume
                med_proxv = np.median(prox_vol)
                mean_proxv = np.mean(prox_vol)
                med_prox_vol[i] = med_proxv
                mean_prox_vol[i] = mean_proxv
                prox_mito_dict[cell.id]["meanv"] = mean_proxv
                prox_mito_dict[cell.id]["medianv"] = med_proxv
                # density
                med_proxd = np.median(prox_den)
                mean_proxd = np.mean(prox_den)
                med_prox_den_arr[i] = med_proxd
                mean_prox_den_arr[i] = mean_proxd
                prox_mito_dict[cell.id]["meand"] = mean_proxd
                prox_mito_dict[cell.id]["mediand"] = med_proxd
                #index
                med_proxi = np.median(prox_ind)
                mean_proxi = np.mean(prox_ind)
                med_prox_in_arr[i] = med_proxi
                mean_prox_in_arr[i] = mean_proxi
                prox_mito_dict[cell.id]["meani"] = mean_proxi
                prox_mito_dict[cell.id]["mediani"] = med_proxi
                #abs, rel frequency of mitochondria of different length
                for li in range(len(prox_cd["bin_abs1"])):
                    prox_bin_dict["abs1"][li][i] = prox_cd["bin_abs1"][li]
                    if type(prox_cd["bin_rel1"]) == np.ndarray:
                        prox_bin_dict["rel1"][li][i] = prox_cd["bin_rel1"][li]
                    else:
                        prox_bin_dict["rel1"][li][i] = 0
                for li in range(len(prox_cd["bin_abs1_5"])):
                    prox_bin_dict["abs1_5"][li][i] = prox_cd["bin_abs1_5"][li]
                    if type(prox_cd["bin_rel1_5"]) == np.ndarray:
                        prox_bin_dict["rel1_5"][li][i] = prox_cd["bin_rel1_5"][li]
                    else:
                        prox_bin_dict["rel1_5"][li][i] = 0
                for li in range(len(prox_cd["bin_abs2"])):
                    prox_bin_dict["abs2"][li][i] = prox_cd["bin_abs2"][li]
                    if type(prox_cd["bin_rel2"]) == np.ndarray:
                        prox_bin_dict["rel2"][li][i] = prox_cd["bin_rel2"][li]
                    else:
                        prox_bin_dict["rel2"][li][i] = 0
            if len(np.nonzero(dist_ds)[0]) > 0:
                sns.distplot(dist_ds, kde=False)
                plt.title('mitochondrium diameter in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('diameter in µm')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_l_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_vol, kde=False)
                plt.title('mitochondrium vol in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('vol in voxel')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_v_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_den, kde=False)
                plt.title('mitochondrium density in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('amount of mitchondria per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_den_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_ind, kde=False)
                plt.title('mitochondrium index in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrial lenght per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_ind_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                #rel, abs mitochondrial frequency
                sns.barplot(dist_cd["bin_abs1"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(dist_cd["bin_abs1_5"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1.5 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(dist_cd["bin_abs2"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 2 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                if type(dist_cd["bin_rel1"]) == np.ndarray:
                    sns.barplot(dist_cd["bin_rel1"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(dist_cd["bin_rel1_5"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1.5 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(dist_cd["bin_rel2"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 2 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                # length/diameter
                dist_cellid[i] = cell.id
                sum_dist_miids[i][:len(dmi_ids)] = dmi_ids
                med_dist = np.median(dist_ds)
                mean_dist = np.mean(dist_ds)
                med_dist_arr[i] = med_dist
                mean_dist_arr[i] = mean_dist
                dist_mito_dict[cell.id]["meanl"] = mean_dist
                dist_mito_dict[cell.id]["medianl"] = med_dist
                sum_dist[i, 0:len(dist_ds)] = dist_ds
                sum_dist_vol[i, 0:len(dist_vol)] = dist_vol
                dist_am[i] = len(dist_ds)
                # volume
                med_distv = np.median(dist_vol)
                mean_distv = np.mean(dist_vol)
                med_dist_vol[i] = med_distv
                mean_dist_vol[i] = mean_distv
                dist_mito_dict[cell.id]["meanv"] = mean_distv
                dist_mito_dict[cell.id]["medianv"] = med_distv
                # density
                med_distd = np.median(dist_den)
                mean_distd = np.mean(dist_den)
                med_dist_den_arr[i] = med_distd
                mean_dist_den_arr[i] = mean_distd
                dist_mito_dict[cell.id]["meand"] = mean_distd
                dist_mito_dict[cell.id]["mediand"] = med_distd
                # index
                med_disti = np.median(dist_ind)
                mean_disti = np.mean(dist_ind)
                med_dist_in_arr[i] = med_disti
                mean_dist_in_arr[i] = mean_disti
                dist_mito_dict[cell.id]["meani"] = mean_disti
                dist_mito_dict[cell.id]["mediani"] = med_disti
                # abs, rel frequency of mitochondria of different length
                for li in range(len(dist_cd["bin_abs1"])):
                    dist_bin_dict["abs1"][li][i] = dist_cd["bin_abs1"][li]
                    if type(dist_cd["bin_rel1"]) == np.ndarray:
                        dist_bin_dict["rel1"][li][i] = dist_cd["bin_rel1"][li]
                    else:
                        dist_bin_dict["rel1"][li][i] = 0
                for li in range(len(dist_cd["bin_abs1_5"])):
                    dist_bin_dict["abs1_5"][li][i] = dist_cd["bin_abs1_5"][li]
                    if type(dist_cd["bin_rel1_5"]) == np.ndarray:
                        dist_bin_dict["rel1_5"][li][i] = dist_cd["bin_rel1_5"][li]
                    else:
                        dist_bin_dict["rel1_5"][li][i] = 0
                for li in range(len(dist_cd["bin_abs2"])):
                    dist_bin_dict["abs2"][li][i] = dist_cd["bin_abs2"][li]
                    if type(dist_cd["bin_rel2"]) == np.ndarray:
                        dist_bin_dict["rel2"][li][i] = dist_cd["bin_rel2"][li]
                    else:
                        dist_bin_dict["rel2"][li][i] = 0
            percentage = 100 * (i / len(cellids))
            print("%.2f percent, cell %.4i done" % (percentage, cell.id))
        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")
        # graphs per celltype
        prox_am = prox_am[prox_am > 0]
        dist_am = dist_am[dist_am > 0]
        med_prox_arr = med_prox_arr[med_prox_arr > 0]
        mean_prox_arr = mean_prox_arr[mean_prox_arr > 0]
        med_dist_arr = med_dist_arr[med_dist_arr > 0]
        mean_dist_arr = mean_dist_arr[mean_dist_arr > 0]
        sum_prox = np.hstack(sum_prox)
        sum_prox = sum_prox[sum_prox > 0]
        sum_dist = np.hstack(sum_dist)
        sum_dist = sum_dist[sum_dist > 0]
        med_prox_vol = med_prox_vol[med_prox_vol > 0]
        mean_prox_vol = mean_prox_vol[mean_prox_vol > 0]
        med_dist_vol = med_dist_vol[med_dist_vol > 0]
        mean_dist_vol = mean_dist_vol[mean_dist_vol > 0]
        sum_prox_vol = np.hstack(sum_prox_vol)
        sum_prox_vol = sum_prox_vol[sum_prox_vol > 0]
        sum_dist_vol = np.hstack(sum_dist_vol)
        sum_dist_vol = sum_dist_vol[sum_dist_vol > 0]
        mean_prox_den_arr = mean_prox_den_arr[mean_prox_den_arr > 0]
        med_prox_den_arr = med_prox_den_arr[med_prox_den_arr > 0]
        mean_dist_den_arr = mean_dist_den_arr[mean_dist_den_arr > 0]
        med_dist_den_arr = med_dist_den_arr[med_dist_den_arr > 0]
        mean_prox_in_arr = mean_prox_in_arr[mean_prox_in_arr > 0]
        med_prox_in_arr = med_prox_in_arr[med_prox_in_arr > 0]
        mean_dist_in_arr = mean_dist_in_arr[mean_dist_in_arr > 0]
        med_dist_in_arr = med_dist_in_arr[med_dist_in_arr > 0]
        prox_cellid = prox_cellid[prox_cellid > 0]
        dist_cellid = dist_cellid[dist_cellid > 0]
        sum_prox_miids = np.hstack(sum_dist_miids)
        sum_prox_miids = sum_prox_miids[sum_prox_miids > 0]
        sum_dist_miids = np.hstack(sum_dist_miids)
        sum_dist_miids = sum_dist_miids[sum_dist_miids > 0]
        sp_sm_miids = sum_prox_miids[sum_prox_vol < 10**4.5]
        sp_lg_miids = sum_prox_miids[sum_prox_vol > 10**4.5]
        sd_sm_miids = sum_dist_miids[sum_dist_vol < 10**4.5]
        sd_lg_miids = sum_dist_miids[sum_dist_vol > 10**4.5]
        prox_sm_cellid = prox_cellid[np.where(med_prox_vol < 100000)]
        prox_lg_cellid = prox_cellid[np.where(med_prox_vol > 100000)]
        dist_sm_cellid = dist_cellid[np.where(med_dist_vol < 100000)]
        dist_lg_cellid = dist_cellid[np.where(med_dist_vol > 100000)]
        write_obj2pkl("%s/prox_sm_arr_%s.pkl" % (f_name,comp_dict[compartment]), prox_sm_cellid)
        write_obj2pkl("%s/prox_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), prox_lg_cellid)
        write_obj2pkl("%s/dist_sm_arr_%s.pkl" % (f_name, comp_dict[compartment]), dist_sm_cellid)
        write_obj2pkl("%s/dist_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), dist_lg_cellid)
        write_obj2pkl("%s/prox_smmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sp_sm_miids)
        write_obj2pkl("%s/prox_lgmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sp_lg_miids)
        write_obj2pkl("%s/dist_smmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sd_sm_miids)
        write_obj2pkl("%s/dist_lgmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sd_lg_miids)
        #plots for length/diameter
        sns.distplot(med_dist_arr, color = "blue", kde=False)
        med_dist_filename = ('%s/median_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment],ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(med_dist_filename)
        plt.close()
        sns.distplot(mean_dist_arr, color = "blue", kde=False)
        mean_dist_filename = ('%s/mean_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        sns.distplot(med_prox_arr, color = "blue", kde=False)
        med_prox_filename = ('%s/median_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(med_prox_filename)
        plt.close()
        sns.distplot(mean_prox_arr, color="blue", kde=False)
        mean_prox_filename = ('%s/mean_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_prox_filename)
        plt.close()
        sns.distplot(sum_prox, color="blue", kde=False)
        sum_prox_filename = ('%s/sum_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_prox_filename)
        plt.close()
        sns.distplot(sum_dist, color="blue", kde=False)
        sum_dist_filename = ('%s/sum_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_dist_filename)
        plt.close()
        #plots for volumen
        sns.distplot(med_dist_vol, color="blue", kde=False)
        med_distv_filename = ('%s/median_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(med_distv_filename)
        plt.close()
        sns.distplot(mean_dist_vol, color="blue", kde=False)
        mean_distv_filename = ('%s/mean_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(mean_distv_filename)
        plt.close()
        sns.distplot(med_prox_vol, color="blue", kde=False)
        med_proxv_filename = ('%s/median_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(med_proxv_filename)
        plt.close()
        sns.distplot(mean_prox_vol, color="blue", kde=False)
        mean_proxv_filename = ('%s/mean_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxv_filename)
        plt.close()
        sns.distplot(sum_prox_vol, color="blue", kde=False)
        sum_proxv_filename = ('%s/sum_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_proxv_filename)
        plt.close()
        sns.distplot(sum_dist_vol, color="blue", kde=False)
        sum_distv_filename = ('%s/sum_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium vol in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_distv_filename)
        plt.close()
        #plots for density
        sns.distplot(mean_prox_den_arr, color="blue", kde=False)
        meanproxden_filename = ('%s/mean_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(meanproxden_filename)
        plt.close()
        sns.distplot(med_prox_den_arr, color="blue", kde=False)
        med_proxd_filename = ('%s/med_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(med_proxd_filename)
        plt.close()
        sns.distplot(mean_dist_den_arr, color="blue", kde=False)
        mean_distd_filename = ('%s/mean_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title( 'mean mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_distd_filename)
        plt.close()
        sns.distplot(med_dist_den_arr, color="blue", kde=False)
        med_distd_filename = ('%s/med_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(med_distd_filename)
        plt.close()
        sns.distplot(mean_prox_in_arr, color="blue", kde=False)
        mean_proxi_filename = ('%s/mean_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochodrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxi_filename)
        plt.close()
        sns.distplot(med_prox_in_arr, color="blue", kde=False)
        med_proxi_filename = ('%s/med_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_proxi_filename)
        plt.close()
        sns.distplot(mean_dist_in_arr, color="blue", kde=False)
        mean_disti_filename = ('%s/mean_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_disti_filename)
        plt.close()
        sns.distplot(med_dist_in_arr, color="blue", kde=False)
        med_disti_filename = ('%s/med_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        plt.close()
        sns.distplot(dist_am, color="blue", kde=False)
        med_disti_filename = ('%s/distam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('amount of mitochondria in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        sns.distplot(prox_am, color="blue", kde=False)
        med_disti_filename = ('%s/proxam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('amount of mitochondrial in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        #log plots
        sns.distplot(np.log10(med_dist_arr), color="blue", kde=False)
        med_dist_filename = ('%s/logmedian_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(med_dist_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_arr), color="blue", kde=False)
        mean_dist_filename = ('%s/logmean_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_arr), color="blue", kde=False)
        med_prox_filename = ('%s/logmedian_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(med_prox_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_arr), color="blue", kde=False)
        mean_prox_filename = ('%s/logmean_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_prox_filename)
        plt.close()
        sns.distplot(np.log10(sum_prox), color="blue", kde=False)
        sum_prox_filename = ('%s/logsum_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_prox_filename)
        plt.close()
        sns.distplot(np.log10(sum_dist), color="blue", kde=False)
        sum_dist_filename = ('%s/logsum_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_dist_filename)
        plt.close()
        # plots for volumen
        sns.distplot(np.log10(med_dist_vol), color="blue", kde=False)
        med_distv_filename = ('%s/logmedian_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_distv_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_vol), color="blue", kde=False)
        mean_distv_filename = ('%s/logmean_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(mean_distv_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_vol), color="blue", kde=False)
        med_proxv_filename = ('%s/logmedian_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^')
        plt.ylabel('count of cells')
        plt.savefig(med_proxv_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_vol), color="blue", kde=False)
        mean_proxv_filename = ('%s/logmean_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^  ')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxv_filename)
        plt.close()
        sns.distplot(np.log10(sum_prox_vol), color="blue", kde=False)
        sum_proxv_filename = ('%s/logsum_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_proxv_filename)
        plt.close()
        sns.distplot(np.log10(sum_dist_vol), color="blue", kde=False)
        sum_distv_filename = ('%s/logsum_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium vol in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_distv_filename)
        plt.close()
        # plots for density
        sns.distplot(np.log10(mean_prox_den_arr), color="blue", kde=False)
        meanproxden_filename = ('%s/logmean_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(meanproxden_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_den_arr), color="blue", kde=False)
        med_proxd_filename = ('%s/logmed_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_proxd_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_den_arr), color="blue", kde=False)
        mean_distd_filename = ('%s/logmean_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title(' log of mean mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(mean_distd_filename)
        plt.close()
        sns.distplot(np.log10(med_dist_den_arr), color="blue", kde=False)
        med_distd_filename = ('%s/logmed_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_distd_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_in_arr), color="blue", kde=False)
        mean_proxi_filename = ('%s/logmean_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochodrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxi_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_in_arr), color="blue", kde=False)
        med_proxi_filename = ('%s/logmed_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_proxi_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_in_arr), color="blue", kde=False)
        mean_disti_filename = ('%s/logmean_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_disti_filename)
        plt.close()
        sns.distplot(np.log10(med_dist_in_arr), color="blue", kde=False)
        med_disti_filename = ('%s/logmed_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        plt.close()
        sns.distplot(np.log10(dist_am), color="blue", kde=False)
        med_disti_filename = ('%s/logdistam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log amount of mitochondria in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log amount of mitochondria in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        sns.distplot(np.log10(prox_am), color="blue", kde=False)
        med_disti_filename = ('%s/logproxam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log amount of mitochondrial in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log amount of mitochondria in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        #plots for bin of frequency of different mitochondrial length
        bin_name = ("%s/bin_plots_%s" % (f_name, ct_dict[celltype]))
        if not os.path.exists(bin_name):
            os.mkdir(bin_name)
        for li in range(len(prox_bin_dict["abs1"])):
            prox_bin_dict["abs1"][li] = prox_bin_dict["abs1"][li][np.nonzero(prox_bin_dict["abs1"][li])[0]]
            prox_bin_dict["rel1"][li] = prox_bin_dict["rel1"][li][np.nonzero(prox_bin_dict["rel1"][li])[0]]
            dist_bin_dict["abs1"][li] = dist_bin_dict["abs1"][li][np.nonzero(dist_bin_dict["abs1"][li])[0]]
            dist_bin_dict["rel1"][li] = dist_bin_dict["rel1"][li][np.nonzero(dist_bin_dict["rel1"][li])[0]]
            if len(prox_bin_dict["abs1"][li]) > 0:
                sns.distplot(prox_bin_dict["abs1"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs1_%i_%s_%s_n.png' % (bin_name,int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in proximal %s in %.4s' % (int(li+1),comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel1"][li]) > 0:
                sns.distplot(prox_bin_dict["rel1"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in proximal %s in %.4s' % (int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs1"][li]) > 0:
                sns.distplot(dist_bin_dict["abs1"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in distal %s in %.4s' % (int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel1"][li]) > 0:
                sns.distplot(dist_bin_dict["rel1"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in distal %s in %.4s' % (
                int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        for li in range(len(prox_bin_dict["abs1_5"])):
            prox_bin_dict["abs1_5"][li] = prox_bin_dict["abs1_5"][li][np.nonzero(prox_bin_dict["abs1_5"][li])[0]]
            prox_bin_dict["rel1_5"][li] = prox_bin_dict["rel1_5"][li][np.nonzero(prox_bin_dict["rel1_5"][li])[0]]
            dist_bin_dict["abs1_5"][li] = dist_bin_dict["abs1_5"][li][np.nonzero(dist_bin_dict["abs1_5"][li])[0]]
            dist_bin_dict["rel1_5"][li] = dist_bin_dict["rel1_5"][li][np.nonzero(dist_bin_dict["rel1_5"][li])[0]]
            if len(prox_bin_dict["abs1_5"][li]) > 0:
                sns.distplot(prox_bin_dict["abs1_5"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %.1f µm in proximal %s in %.4s' % ((li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel1_5"][li]) > 0:
                sns.distplot(prox_bin_dict["rel1_5"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %.1f µm in proximal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs1_5"][li]) > 0:
                sns.distplot(dist_bin_dict["abs1_5"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %.1f µm in distal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel1_5"][li]) > 0:
                sns.distplot(dist_bin_dict["rel1_5"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %.1f µm in distal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        for li in range(len(prox_bin_dict["abs2"])):
            prox_bin_dict["abs2"][li] = prox_bin_dict["abs2"][li][np.nonzero(prox_bin_dict["abs2"][li])[0]]
            prox_bin_dict["rel2"][li] = prox_bin_dict["rel2"][li][np.nonzero(prox_bin_dict["rel2"][li])[0]]
            dist_bin_dict["abs2"][li] = dist_bin_dict["abs2"][li][np.nonzero(dist_bin_dict["abs2"][li])[0]]
            dist_bin_dict["rel2"][li] = dist_bin_dict["rel2"][li][np.nonzero(dist_bin_dict["rel2"][li])[0]]
            if len(prox_bin_dict["abs2"][li]) > 0:
                sns.distplot(prox_bin_dict["abs2"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in proximal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel2"][li]) > 0:
                sns.distplot(prox_bin_dict["rel2"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in proximal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs2"][li]) > 0:
                sns.distplot(dist_bin_dict["abs2"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in distal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel2"][li]) > 0:
                sns.distplot(dist_bin_dict["rel2"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in distal %s in %.4s' % (
                    int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('plots per celltype and compartment')
        log.info("finished plots for %s in compartment: %s" % (ct_dict[celltype], comp_dict[compartment]))

        
        return prox_mito_dict, dist_mito_dict


    msn_den_prox_dict, msn_den_dist_dict = ct_mito_analysis(ssd, celltype=2)
    msn_ax_prox_dict, msn_ax_dist_dict = ct_mito_analysis(ssd, celltype=2, compartment=1)
    #gp_den_prox_dict, gp_den_dist_dict = ct_mito_analysis(ssd, celltype=[6,7])
    #gp_ax_prox_dict, gp_ax_dist_dict = ct_mito_analysis(ssd, celltype=[6,7], compartment = 1)

    raise ValueError
if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    import pandas as pd
    import os as os
    import scipy
    import time
    from tqdm import tqdm
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

    def mito_prox_dis(sso, compartment = 0, radius = 50):
        """
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
        kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"] * sso.scaling)
        mis = sso.mis
        prox_list = np.zeros(len(mis))
        dist_list = np.zeros(len(mis))
        prox_d = np.zeros(len(mis))
        dist_d = np.zeros(len(mis))
        #mitochrondrial index and density (every 10 µm)
        prox_num = np.zeros(5) #proximal until 50µm
        prox_den = np.zeros(5)
        dist_num = np.zeros(50) #from 100 to 500 µm
        dist_den = np.zeros(50)
        prox_pathlength = np.zeros(len(mis))
        dist_pathlength = np.zeros(len(mis))
        prox_vol = np.zeros(len(mis))
        dist_vol = np.zeros(len(mis))
        prox_bin_abs1 = np.zeros(8)
        prox_bin_abs1_5 = np.zeros(6)
        prox_bin_abs2 = np.zeros(5)
        dist_bin_abs1 = np.zeros(8)
        dist_bin_abs1_5 = np.zeros(6)
        dist_bin_abs2 = np.zeros(5)
        for i, mi in enumerate(tqdm(mis)):
            #from syconn.reps.super_segmentation_object attr_for_coors
            close_node_ids = kdtree.query_ball_point(mi.rep_coord * sso.scaling, radius)
            radfactor = 1
            while len(close_node_ids) <= 2:
                close_node_ids = kdtree.query_ball_point(mi.rep_coord * sso.scaling, radius*(2**radfactor))
                radfactor += 1
                if radfactor == 5:
                    break
            axo = np.array(sso.skeleton["axoness_avg10000"][close_node_ids])
            axo[axo == 3] = 1
            axo[axo == 4] = 1
            if np.all(axo) == unwanted_compartment or np.all(axo) == 2:
                continue
            if len(np.where(axo == compartment)[0]) < len(axo)/2:
                continue
            dists = sso.shortestpath2soma(mi.mesh_bb)
            distproxdist = sso.shortestpath2soma([mi.rep_coord * sso.scaling])[0]
            if distproxdist == 0:
                continue
            dist1 = dists[0]/1000. #in µm, kimimaro skeletons in physical parameters
            dist2 = dists[1]/1000.
            mi_diff = mi.mesh_bb[1] - mi.mesh_bb[0]
            mi_length = np.linalg.norm(mi_diff)/1000. #in µm
            if dist1 <= dist2:
                dist = dist1
                start_bin = int(dist1 / 10)
                end_bin = int(dist2 / 10)
                start_dist = dist1
                end_dist = dist2
            else:
                dist = dist2
                start_bin = int(dist2 / 10)
                end_bin = int(dist1 / 10)
                start_dist = dist2
                end_dist = dist1
            if distproxdist <= 50:
                prox_list[i] = mi.id
                prox_d[i] = mi_length
                prox_pathlength[i] = dist
                prox_vol[i] = mi.size
                prox_num[start_bin] += 1
                if start_bin == end_bin:
                    prox_den[start_bin] += mi_length
                elif start_bin != end_bin: #if spanning multiple bins then counted in every of them
                    span = end_bin - start_bin
                    if span == 1:
                        per_length = (end_dist - end_bin * 10) / mi_length
                        prox_den[start_bin] += mi_length * (1 - per_length)
                        if end_bin <= 4:
                            prox_num[end_bin] += 1
                            prox_den[end_bin] += mi_length * per_length
                    elif span > 1:
                        if span > 4:
                            span = 4
                            end_bin = 4
                        per_length_start = ((start_bin+1)*10 - start_dist/mi_length)
                        prox_den[start_bin] += mi_length*per_length_start
                        for i in range(1, span):
                            if start_bin + i < 4:
                                prox_num[start_bin + i] += 1
                                prox_den[start_bin + i] += mi_length*(10/mi_length)
                                if i == span:
                                    prox_den[end_bin] += mi_length*((end_dist - end_bin*10)/mi_length)
                            else:
                                if end_bin == 4:
                                    prox_den[end_bin] += mi_length * ((end_dist- end_bin * 10) / mi_length)
                                    prox_num[end_bin] += 1
                                else:
                                    prox_den[4] += 1
                                    prox_den[4] += mi_length*(10/mi_length)
                prox_1_i = int(mi_length)
                prox_1_5_i = int(mi_length/1.5)
                prox_2_i = int(mi_length/2)
                if prox_1_i > 7:
                    prox_1_i = 7
                if prox_1_5_i > 5:
                    prox_1_5_i = 5
                if prox_2_i > 4:
                    prox_2_i = 4
                prox_bin_abs1[prox_1_i] += 1
                prox_bin_abs1_5[prox_1_5_i] += 1
                prox_bin_abs2[prox_2_i] += 1
            elif distproxdist >= 100:
                dist_list[i] = mi.id
                dist_d[i] = mi_length
                dist_pathlength[i] = dist
                dist_vol[i] = mi.size
                start_bin = start_bin - 10  # starting at 100 µm
                end_bin = end_bin - 10
                dist_num[start_bin] += 1
                if start_bin == end_bin:
                    dist_den[start_bin] += mi_length
                elif start_bin != end_bin:  # if spanning multiple bins then counted in every of them
                    span = end_bin - start_bin
                    if span == 1:
                        dist_num[end_bin] += 1
                        per_length = (dist2 - end_bin * 10) / mi_length
                        dist_den[start_bin] += mi_length * (1 - per_length)
                        dist_den[end_bin] += mi_length * per_length
                    elif span > 1:
                        per_length_start = ((start_bin + 1) * 10 - dist1) / mi_length
                        dist_den[start_bin] += mi_length * per_length_start
                        for i in range(1, span):
                            dist_num[start_bin + i] += 1
                            dist_den[start_bin + i] += mi_length * (10 / mi_length)
                            if i == span:
                                dist_den[end_bin] += mi_length * ((dist2 - end_bin * 10) / mi_length)
                    dist_1_i = int(mi_length)
                    dist_1_5_i = int(mi_length / 1.5)
                    dist_2_i = int(mi_length / 2)
                    if dist_1_i > 7:
                        dist_1_i = 7
                    if dist_1_5_i > 5:
                        dist_1_5_i = 5
                    if dist_2_i > 4:
                        dist_2_i = 4
                    dist_bin_abs1[dist_1_i] += 1
                    dist_bin_abs1_5[dist_1_5_i] += 1
                    dist_bin_abs2[dist_2_i] += 1
            percentage = 100 * (i / len(mis))
            print("%.2f percent, %4i mitos" % (percentage, len(mis)))
        dist_id_array = dist_list[dist_list > 0]
        prox_id_array = prox_list[prox_list > 0]
        prox_d_array = prox_d[prox_d > 0]
        dist_d_array = dist_d[dist_d > 0]
        prox_pathlength = prox_pathlength[prox_pathlength > 0]
        dist_pathlength = dist_pathlength[dist_pathlength > 0]
        prox_num = prox_num[prox_num > 0]
        prox_den = prox_den[prox_den > 0]
        prox_vol = prox_vol[prox_vol > 0]
        dist_num = dist_num[dist_num > 0]
        dist_den = dist_den[dist_den > 0]
        dist_vol = dist_vol[dist_vol > 0]
        sum_prox_mis = np.sum(np.hstack(prox_bin_abs1))
        sum_dist_mis = np.sum(np.hstack(dist_bin_abs1))
        if sum_prox_mis > 0:
            prox_bin_rel1 = prox_bin_abs1/sum_prox_mis
            prox_bin_rel1_5 = prox_bin_abs1_5/sum_prox_mis
            prox_bin_rel2 = prox_bin_abs2/sum_prox_mis
        else:
            prox_bin_rel1 = 0
            prox_bin_rel1_5 = 0
            prox_bin_rel2 = 0
        if sum_dist_mis > 0:
            dist_bin_rel1 = dist_bin_abs1/sum_dist_mis
            dist_bin_rel1_5 = dist_bin_abs1_5/sum_dist_mis
            dist_bin_rel2 = dist_bin_abs2/sum_dist_mis
        else:
            dist_bin_rel1 = 0
            dist_bin_rel1_5 = 0
            dist_bin_rel2 = 0
        prox_comb = [prox_d_array, prox_id_array, prox_pathlength, prox_vol, prox_num,prox_den, prox_list,  prox_bin_abs1, prox_bin_abs1_5, prox_bin_abs2, prox_bin_rel1, prox_bin_rel1_5, prox_bin_rel2]
        dist_comb = [dist_d_array, dist_id_array, dist_pathlength, dist_vol, dist_num, dist_den, dist_list, dist_bin_abs1, dist_bin_abs1_5, dist_bin_abs2, dist_bin_rel1, dist_bin_rel1_5, dist_bin_rel2]
        return prox_comb, dist_comb

    def ct_mito_analysis(ssd, celltype, compartment=0):
        """
        analysis of mitochondria length in proximal [until 50 µm] and distal[from 100 µm onwards] dendrites.
        Parameters that will be plotted are mitochondrial lenth (norm of bounding box diameter, might need a more exact measurement later),
        mitochondrial volume(mi.size), mitochondrial density (amount of mitochondria per 10 µm axon/dendrite), mitochondrial index (mitochondrial
        length per 10 µm axon/dendritic length), also bins with mitochondrial length will be compared (see chandra et al., 2019 for comparison)
        :param ssd: SuperSegmentationDataset
        :param celltype: 0:"STN", 1:"DA", 2:"MSN", 3:"LMAN", 4:"HVC", 5:"GP", 6:"FS", 7:"TAN", 8:"INT"
        celltypes: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
                compartment: 0 = dendrite, 1 = axon, 2 = soma
        :return: mitochondrial parameters in proximal and sital dendrites
        """
        # per celltype function
        # for cell == celltype:
        # sholl_analysis(cell)
        #TO DO: mitochondrial index(mitochondrial length per 10 µm length), mitochondrial density(amount per 10 µm length)
        #chandra et al. 2019: comparison of mitochondria in each size range normalized to total number of mitochondria in each dendrite
        start = time.time()
        f_name = ("u/arother/test_folder/2010504_j0251v3_mitoMSN_analysis")
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('mitchondrial_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",10: "NGF"}
        #cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        #gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        #gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        #cellids = np.hstack(np.array([gpeids, gpiids]))
        #ct_dict = {6: "GP", 7: "GP"}
        #celltype = celltype[0]
        prox_mito_dict = dict()
        dist_mito_dict = dict()
        # length/diameter of mitochondria
        mean_prox_arr = np.zeros(len(cellids))
        med_prox_arr = np.zeros(len(cellids))
        mean_dist_arr = np.zeros(len(cellids))
        med_dist_arr = np.zeros(len(cellids))
        sum_prox = np.zeros((len(cellids), 1000))
        sum_dist = np.zeros((len(cellids), 1000))
        # volumen of mitochondria
        mean_prox_vol = np.zeros(len(cellids))
        med_prox_vol = np.zeros(len(cellids))
        mean_dist_vol = np.zeros(len(cellids))
        med_dist_vol = np.zeros(len(cellids))
        sum_prox_vol = np.zeros((len(cellids), 1000))
        sum_dist_vol = np.zeros((len(cellids), 1000))
        sum_prox_miids = np.zeros((len(cellids), 1000))
        sum_dist_miids = np.zeros((len(cellids), 3000))
        # density/ index of mitochondria
        mean_prox_den_arr = np.zeros(len(cellids))
        mean_prox_in_arr = np.zeros(len(cellids))
        mean_dist_den_arr = np.zeros(len(cellids))
        mean_dist_in_arr = np.zeros(len(cellids))
        med_prox_den_arr = np.zeros(len(cellids))
        med_prox_in_arr = np.zeros(len(cellids))
        med_dist_den_arr = np.zeros(len(cellids))
        med_dist_in_arr = np.zeros(len(cellids))
        prox_am = np.zeros(len(cellids))
        dist_am = np.zeros(len(cellids))
        prox_cellid = np.zeros(len(cellids))
        dist_cellid = np.zeros(len(cellids))
        bin_keys = ["abs1", "abs1_5", "abs2", "rel1", "rel1_5", "rel2"]
        bin_keys2 = np.array([8, 6, 5, 8, 6, 5])
        prox_bin_dict = {i: dict() for i in bin_keys}
        dist_bin_dict = {i: dict() for i in bin_keys}
        for ki, k in enumerate(prox_bin_dict.keys()):
            prox_bin_dict[k] = {i: np.zeros(len(cellids)) for i in range(bin_keys2[ki])}
        for ki, k in enumerate(dist_bin_dict.keys()):
            dist_bin_dict[k] = {i: np.zeros(len(cellids)) for i in range(bin_keys2[ki])}
        comp_dict = {1: 'axons', 0: 'dendrites'}
        log.info('Step 1/2 generating mitochondrial parameters per cell')
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            """
            cell.load_skeleton()
            unique_preds = np.unique(cell.skeleton['axoness_avg10000'])
            if not (0 in unique_preds and (1 in unique_preds or 3 in unique_preds or 4 in unique_preds) and 2 in unique_preds):
                continue
            """
            prox_mito_dict[cell.id] = dict()
            dist_mito_dict[cell.id] = dict()
            prox_list, dist_list  = mito_prox_dis(cell, compartment)
            #prox_list = [prox_d_array, prox_id_array, prox_pathlength, prox_vol, prox_num, prox_den, prox_bin_abs1, prox_bin_abs1_5, prox_bin_abs2, prox_bin_rel1, prox_bin_rel1_5, prox_bin_rel2]
            prox_mito_dict[cell.id] = {"diameter": prox_list[0], "id": prox_list[1], "pathlength":prox_list[2], "vol": prox_list[3],
                                       "density":prox_list[4], "index": prox_list[5], "mi_ids": prox_list[6], "bin_abs1": prox_list[7], "bin_abs1_5":prox_list[8],
                                       "bin_abs2":prox_list[9], "bin_rel1": prox_list[10], "bin_rel1_5": prox_list[11], "bin_rel2": prox_list[12]}
            dist_mito_dict[cell.id] = {"diameter": dist_list[0], "id": dist_list[1], "pathlength": dist_list[2],
                                       "vol": dist_list[3],"density": dist_list[4], "index": dist_list[5], "mi_ids":dist_list[6], "bin_abs1": dist_list[7],
                                       "bin_abs1_5": dist_list[8],"bin_abs2": dist_list[9], "bin_rel1": dist_list[10], "bin_rel1_5": dist_list[11],
                                       "bin_rel2": dist_list[12]}
            prox_cd = prox_mito_dict[cell.id]
            dist_cd = dist_mito_dict[cell.id]
            prox_ds = prox_cd["diameter"]
            dist_ds = dist_cd["diameter"]
            prox_vol = prox_cd["vol"]
            dist_vol = dist_cd["vol"]
            prox_den = prox_cd["density"]
            dist_den = dist_cd["density"]
            prox_ind = prox_cd["index"]
            dist_ind = dist_cd["index"]
            pmi_ids = prox_cd["mi_ids"]
            dmi_ids = dist_cd["mi_ids"]
            # plot mitos_per cell
            if len(np.nonzero(prox_ds)) == 0 and len(np.nonzero(dist_ds)) == 0:
                print("cell %i has no suitable mitochondria" % cell.id)
                percentage = 100 * (i / len(cellids))
                print("%.2f percent, cell %.4i done" % (percentage, cell.id))
                continue
            columns = ['pathlength in µm', ('count of %s' % comp_dict[compartment])]
            foldername = '%s/mito_percell_%s' % (f_name, ct_dict[celltype])
            if not os.path.exists(foldername):
                os.mkdir(foldername)
            if len(np.nonzero(prox_ds)[0]) > 0:
                sns.distplot(prox_ds, kde=False)
                plt.title('mitochondrium diameter in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('diameter in µm')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_l_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_vol, kde=False)
                plt.title('mitochondrium vol in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('vol in voxel')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_v_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_den, kde=False)
                plt.title('mitochondrium density in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('amount of mitchondria per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_den_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(prox_ind, kde=False)
                plt.title('mitochondrium index in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrial lenght per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_ind_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                #abs and relative bins
                sns.barplot(prox_cd["bin_abs1"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(prox_cd["bin_abs1_5"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1.5 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(prox_cd["bin_abs2"])
                plt.title('abs mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 2 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/prox_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                if type(prox_cd["bin_rel1"]) == np.ndarray:
                    sns.barplot(prox_cd["bin_rel1"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(prox_cd["bin_rel1_5"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1.5 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(prox_cd["bin_rel2"])
                    plt.title('relative mito frequency in proximal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 2 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/prox_mito_rel2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    # length/diameter
                prox_cellid[i] = cell.id
                sum_prox_miids[i] = pmi_ids
                med_prox = np.median(prox_ds)
                mean_prox = np.mean(prox_ds)
                med_prox_arr[i] = med_prox
                mean_prox_arr[i] = mean_prox
                prox_mito_dict[cell.id]["meanl"] = mean_prox
                prox_mito_dict[cell.id]["medianl"] = med_prox
                sum_prox[i, 0:len(prox_ds)] = prox_ds
                sum_prox_vol[i, 0:len(prox_ds)] = prox_vol
                prox_am[i] = len(prox_ds)
                # volume
                med_proxv = np.median(prox_vol)
                mean_proxv = np.mean(prox_vol)
                med_prox_vol[i] = med_proxv
                mean_prox_vol[i] = mean_proxv
                prox_mito_dict[cell.id]["meanv"] = mean_proxv
                prox_mito_dict[cell.id]["medianv"] = med_proxv
                # density
                med_proxd = np.median(prox_den)
                mean_proxd = np.mean(prox_den)
                med_prox_den_arr[i] = med_proxd
                mean_prox_den_arr[i] = mean_proxd
                prox_mito_dict[cell.id]["meand"] = mean_proxd
                prox_mito_dict[cell.id]["mediand"] = med_proxd
                #index
                med_proxi = np.median(prox_ind)
                mean_proxi = np.mean(prox_ind)
                med_prox_in_arr[i] = med_proxi
                mean_prox_in_arr[i] = mean_proxi
                prox_mito_dict[cell.id]["meani"] = mean_proxi
                prox_mito_dict[cell.id]["mediani"] = med_proxi
                #abs, rel frequency of mitochondria of different length
                for li in range(len(prox_cd["bin_abs1"])):
                    prox_bin_dict["abs1"][li][i] = prox_cd["bin_abs1"][li]
                    if type(prox_cd["bin_rel1"]) == np.ndarray:
                        prox_bin_dict["rel1"][li][i] = prox_cd["bin_rel1"][li]
                    else:
                        prox_bin_dict["rel1"][li][i] = 0
                for li in range(len(prox_cd["bin_abs1_5"])):
                    prox_bin_dict["abs1_5"][li][i] = prox_cd["bin_abs1_5"][li]
                    if type(prox_cd["bin_rel1_5"]) == np.ndarray:
                        prox_bin_dict["rel1_5"][li][i] = prox_cd["bin_rel1_5"][li]
                    else:
                        prox_bin_dict["rel1_5"][li][i] = 0
                for li in range(len(prox_cd["bin_abs2"])):
                    prox_bin_dict["abs2"][li][i] = prox_cd["bin_abs2"][li]
                    if type(prox_cd["bin_rel2"]) == np.ndarray:
                        prox_bin_dict["rel2"][li][i] = prox_cd["bin_rel2"][li]
                    else:
                        prox_bin_dict["rel2"][li][i] = 0
            if len(np.nonzero(dist_ds)[0]) > 0:
                sns.distplot(dist_ds, kde=False)
                plt.title('mitochondrium diameter in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('diameter in µm')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_l_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_vol, kde=False)
                plt.title('mitochondrium vol in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('vol in voxel')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_v_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_den, kde=False)
                plt.title('mitochondrium density in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('amount of mitchondria per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_den_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.distplot(dist_ind, kde=False)
                plt.title('mitochondrium index in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrial lenght per 10 µm length')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_ind_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                #rel, abs mitochondrial frequency
                sns.barplot(dist_cd["bin_abs1"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(dist_cd["bin_abs1_5"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 1.5 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                sns.barplot(dist_cd["bin_abs2"])
                plt.title('abs mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('mitochondrium length in 2 µm bins')
                plt.ylabel('count of cells')
                plt.savefig('%s/dist_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                plt.close()
                if type(dist_cd["bin_rel1"]) == np.ndarray:
                    sns.barplot(dist_cd["bin_rel1"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs1_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(dist_cd["bin_rel1_5"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 1.5 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs1_5_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                    sns.barplot(dist_cd["bin_rel2"])
                    plt.title('relative mito frequency in distal %s in %s' % (comp_dict[compartment], ct_dict[celltype]))
                    plt.xlabel('mitochondrium length in 2 µm bins')
                    plt.ylabel('count of cells')
                    plt.savefig('%s/dist_mito_abs2_%i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
                    plt.close()
                # length/diameter
                dist_cellid[i] = cell.id
                sum_dist_miids[i][:len(dmi_ids)] = dmi_ids
                med_dist = np.median(dist_ds)
                mean_dist = np.mean(dist_ds)
                med_dist_arr[i] = med_dist
                mean_dist_arr[i] = mean_dist
                dist_mito_dict[cell.id]["meanl"] = mean_dist
                dist_mito_dict[cell.id]["medianl"] = med_dist
                sum_dist[i, 0:len(dist_ds)] = dist_ds
                sum_dist_vol[i, 0:len(dist_vol)] = dist_vol
                dist_am[i] = len(dist_ds)
                # volume
                med_distv = np.median(dist_vol)
                mean_distv = np.mean(dist_vol)
                med_dist_vol[i] = med_distv
                mean_dist_vol[i] = mean_distv
                dist_mito_dict[cell.id]["meanv"] = mean_distv
                dist_mito_dict[cell.id]["medianv"] = med_distv
                # density
                med_distd = np.median(dist_den)
                mean_distd = np.mean(dist_den)
                med_dist_den_arr[i] = med_distd
                mean_dist_den_arr[i] = mean_distd
                dist_mito_dict[cell.id]["meand"] = mean_distd
                dist_mito_dict[cell.id]["mediand"] = med_distd
                # index
                med_disti = np.median(dist_ind)
                mean_disti = np.mean(dist_ind)
                med_dist_in_arr[i] = med_disti
                mean_dist_in_arr[i] = mean_disti
                dist_mito_dict[cell.id]["meani"] = mean_disti
                dist_mito_dict[cell.id]["mediani"] = med_disti
                # abs, rel frequency of mitochondria of different length
                for li in range(len(dist_cd["bin_abs1"])):
                    dist_bin_dict["abs1"][li][i] = dist_cd["bin_abs1"][li]
                    if type(dist_cd["bin_rel1"]) == np.ndarray:
                        dist_bin_dict["rel1"][li][i] = dist_cd["bin_rel1"][li]
                    else:
                        dist_bin_dict["rel1"][li][i] = 0
                for li in range(len(dist_cd["bin_abs1_5"])):
                    dist_bin_dict["abs1_5"][li][i] = dist_cd["bin_abs1_5"][li]
                    if type(dist_cd["bin_rel1_5"]) == np.ndarray:
                        dist_bin_dict["rel1_5"][li][i] = dist_cd["bin_rel1_5"][li]
                    else:
                        dist_bin_dict["rel1_5"][li][i] = 0
                for li in range(len(dist_cd["bin_abs2"])):
                    dist_bin_dict["abs2"][li][i] = dist_cd["bin_abs2"][li]
                    if type(dist_cd["bin_rel2"]) == np.ndarray:
                        dist_bin_dict["rel2"][li][i] = dist_cd["bin_rel2"][li]
                    else:
                        dist_bin_dict["rel2"][li][i] = 0
            percentage = 100 * (i / len(cellids))
            print("%.2f percent, cell %.4i done" % (percentage, cell.id))
        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")
        # graphs per celltype
        prox_am = prox_am[prox_am > 0]
        dist_am = dist_am[dist_am > 0]
        med_prox_arr = med_prox_arr[med_prox_arr > 0]
        mean_prox_arr = mean_prox_arr[mean_prox_arr > 0]
        med_dist_arr = med_dist_arr[med_dist_arr > 0]
        mean_dist_arr = mean_dist_arr[mean_dist_arr > 0]
        sum_prox = np.hstack(sum_prox)
        sum_prox = sum_prox[sum_prox > 0]
        sum_dist = np.hstack(sum_dist)
        sum_dist = sum_dist[sum_dist > 0]
        med_prox_vol = med_prox_vol[med_prox_vol > 0]
        mean_prox_vol = mean_prox_vol[mean_prox_vol > 0]
        med_dist_vol = med_dist_vol[med_dist_vol > 0]
        mean_dist_vol = mean_dist_vol[mean_dist_vol > 0]
        sum_prox_vol = np.hstack(sum_prox_vol)
        sum_prox_vol = sum_prox_vol[sum_prox_vol > 0]
        sum_dist_vol = np.hstack(sum_dist_vol)
        sum_dist_vol = sum_dist_vol[sum_dist_vol > 0]
        mean_prox_den_arr = mean_prox_den_arr[mean_prox_den_arr > 0]
        med_prox_den_arr = med_prox_den_arr[med_prox_den_arr > 0]
        mean_dist_den_arr = mean_dist_den_arr[mean_dist_den_arr > 0]
        med_dist_den_arr = med_dist_den_arr[med_dist_den_arr > 0]
        mean_prox_in_arr = mean_prox_in_arr[mean_prox_in_arr > 0]
        med_prox_in_arr = med_prox_in_arr[med_prox_in_arr > 0]
        mean_dist_in_arr = mean_dist_in_arr[mean_dist_in_arr > 0]
        med_dist_in_arr = med_dist_in_arr[med_dist_in_arr > 0]
        prox_cellid = prox_cellid[prox_cellid > 0]
        dist_cellid = dist_cellid[dist_cellid > 0]
        sum_prox_miids = np.hstack(sum_dist_miids)
        sum_prox_miids = sum_prox_miids[sum_prox_miids > 0]
        sum_dist_miids = np.hstack(sum_dist_miids)
        sum_dist_miids = sum_dist_miids[sum_dist_miids > 0]
        sp_sm_miids = sum_prox_miids[sum_prox_vol < 10**4.5]
        sp_lg_miids = sum_prox_miids[sum_prox_vol > 10**4.5]
        sd_sm_miids = sum_dist_miids[sum_dist_vol < 10**4.5]
        sd_lg_miids = sum_dist_miids[sum_dist_vol > 10**4.5]
        prox_sm_cellid = prox_cellid[np.where(med_prox_vol < 100000)]
        prox_lg_cellid = prox_cellid[np.where(med_prox_vol > 100000)]
        dist_sm_cellid = dist_cellid[np.where(med_dist_vol < 100000)]
        dist_lg_cellid = dist_cellid[np.where(med_dist_vol > 100000)]
        write_obj2pkl("%s/prox_sm_arr_%s.pkl" % (f_name,comp_dict[compartment]), prox_sm_cellid)
        write_obj2pkl("%s/prox_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), prox_lg_cellid)
        write_obj2pkl("%s/dist_sm_arr_%s.pkl" % (f_name, comp_dict[compartment]), dist_sm_cellid)
        write_obj2pkl("%s/dist_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), dist_lg_cellid)
        write_obj2pkl("%s/prox_smmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sp_sm_miids)
        write_obj2pkl("%s/prox_lgmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sp_lg_miids)
        write_obj2pkl("%s/dist_smmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sd_sm_miids)
        write_obj2pkl("%s/dist_lgmi_arr_%s.pkl" % (f_name, comp_dict[compartment]), sd_lg_miids)
        #plots for length/diameter
        sns.distplot(med_dist_arr, color = "blue", kde=False)
        med_dist_filename = ('%s/median_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment],ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(med_dist_filename)
        plt.close()
        sns.distplot(mean_dist_arr, color = "blue", kde=False)
        mean_dist_filename = ('%s/mean_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        sns.distplot(med_prox_arr, color = "blue", kde=False)
        med_prox_filename = ('%s/median_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(med_prox_filename)
        plt.close()
        sns.distplot(mean_prox_arr, color="blue", kde=False)
        mean_prox_filename = ('%s/mean_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_prox_filename)
        plt.close()
        sns.distplot(sum_prox, color="blue", kde=False)
        sum_prox_filename = ('%s/sum_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_prox_filename)
        plt.close()
        sns.distplot(sum_dist, color="blue", kde=False)
        sum_dist_filename = ('%s/sum_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_dist_filename)
        plt.close()
        #plots for volumen
        sns.distplot(med_dist_vol, color="blue", kde=False)
        med_distv_filename = ('%s/median_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(med_distv_filename)
        plt.close()
        sns.distplot(mean_dist_vol, color="blue", kde=False)
        mean_distv_filename = ('%s/mean_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(mean_distv_filename)
        plt.close()
        sns.distplot(med_prox_vol, color="blue", kde=False)
        med_proxv_filename = ('%s/median_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(med_proxv_filename)
        plt.close()
        sns.distplot(mean_prox_vol, color="blue", kde=False)
        mean_proxv_filename = ('%s/mean_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxv_filename)
        plt.close()
        sns.distplot(sum_prox_vol, color="blue", kde=False)
        sum_proxv_filename = ('%s/sum_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_proxv_filename)
        plt.close()
        sns.distplot(sum_dist_vol, color="blue", kde=False)
        sum_distv_filename = ('%s/sum_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mitochondrium vol in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_distv_filename)
        plt.close()
        #plots for density
        sns.distplot(mean_prox_den_arr, color="blue", kde=False)
        meanproxden_filename = ('%s/mean_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(meanproxden_filename)
        plt.close()
        sns.distplot(med_prox_den_arr, color="blue", kde=False)
        med_proxd_filename = ('%s/med_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(med_proxd_filename)
        plt.close()
        sns.distplot(mean_dist_den_arr, color="blue", kde=False)
        mean_distd_filename = ('%s/mean_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title( 'mean mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_distd_filename)
        plt.close()
        sns.distplot(med_dist_den_arr, color="blue", kde=False)
        med_distd_filename = ('%s/med_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria per 10 µm')
        plt.ylabel('count of cells')
        plt.savefig(med_distd_filename)
        plt.close()
        sns.distplot(mean_prox_in_arr, color="blue", kde=False)
        mean_proxi_filename = ('%s/mean_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochodrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxi_filename)
        plt.close()
        sns.distplot(med_prox_in_arr, color="blue", kde=False)
        med_proxi_filename = ('%s/med_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_proxi_filename)
        plt.close()
        sns.distplot(mean_dist_in_arr, color="blue", kde=False)
        mean_disti_filename = ('%s/mean_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_disti_filename)
        plt.close()
        sns.distplot(med_dist_in_arr, color="blue", kde=False)
        med_disti_filename = ('%s/med_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('median mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('mitochondrial length per 10 µm [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        plt.close()
        sns.distplot(dist_am, color="blue", kde=False)
        med_disti_filename = ('%s/distam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('amount of mitochondria in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        sns.distplot(prox_am, color="blue", kde=False)
        med_disti_filename = ('%s/proxam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('amount of mitochondrial in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        #log plots
        sns.distplot(np.log10(med_dist_arr), color="blue", kde=False)
        med_dist_filename = ('%s/logmedian_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(med_dist_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_arr), color="blue", kde=False)
        mean_dist_filename = ('%s/logmean_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_arr), color="blue", kde=False)
        med_prox_filename = ('%s/logmedian_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(med_prox_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_arr), color="blue", kde=False)
        mean_prox_filename = ('%s/logmean_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(mean_prox_filename)
        plt.close()
        sns.distplot(np.log10(sum_prox), color="blue", kde=False)
        sum_prox_filename = ('%s/logsum_prox_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_prox_filename)
        plt.close()
        sns.distplot(np.log10(sum_dist), color="blue", kde=False)
        sum_dist_filename = ('%s/logsum_dist_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium diameter in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of length in 10^ µm')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_dist_filename)
        plt.close()
        # plots for volumen
        sns.distplot(np.log10(med_dist_vol), color="blue", kde=False)
        med_distv_filename = ('%s/logmedian_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_distv_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_vol), color="blue", kde=False)
        mean_distv_filename = ('%s/logmean_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium vol in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(mean_distv_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_vol), color="blue", kde=False)
        med_proxv_filename = ('%s/logmedian_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^')
        plt.ylabel('count of cells')
        plt.savefig(med_proxv_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_vol), color="blue", kde=False)
        mean_proxv_filename = ('%s/logmean_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^  ')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxv_filename)
        plt.close()
        sns.distplot(np.log10(sum_prox_vol), color="blue", kde=False)
        sum_proxv_filename = ('%s/logsum_proxv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium vol in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_proxv_filename)
        plt.close()
        sns.distplot(np.log10(sum_dist_vol), color="blue", kde=False)
        sum_distv_filename = ('%s/logsum_distv_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mitochondrium vol in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of vol in voxel in 10^ ')
        plt.ylabel('count of mitochondria')
        plt.savefig(sum_distv_filename)
        plt.close()
        # plots for density
        sns.distplot(np.log10(mean_prox_den_arr), color="blue", kde=False)
        meanproxden_filename = ('%s/logmean_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(meanproxden_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_den_arr), color="blue", kde=False)
        med_proxd_filename = ('%s/logmed_proxd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial density in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_proxd_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_den_arr), color="blue", kde=False)
        mean_distd_filename = ('%s/logmean_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title(' log of mean mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(mean_distd_filename)
        plt.close()
        sns.distplot(np.log10(med_dist_den_arr), color="blue", kde=False)
        med_distd_filename = ('%s/logmed_distd_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial density in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of amount of mitochondria per 10 µm in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_distd_filename)
        plt.close()
        sns.distplot(np.log10(mean_prox_in_arr), color="blue", kde=False)
        mean_proxi_filename = ('%s/logmean_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochodrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_proxi_filename)
        plt.close()
        sns.distplot(np.log10(med_prox_in_arr), color="blue", kde=False)
        med_proxi_filename = ('%s/logmed_proxi_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial index in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_proxi_filename)
        plt.close()
        sns.distplot(np.log10(mean_dist_in_arr), color="blue", kde=False)
        mean_disti_filename = ('%s/logmean_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of mean mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(mean_disti_filename)
        plt.close()
        sns.distplot(np.log10(med_dist_in_arr), color="blue", kde=False)
        med_disti_filename = ('%s/logmed_disti_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of median mitochondrial index in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log of mitochondrial length per 10 µm in 10^ [µm]')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        plt.close()
        sns.distplot(np.log10(dist_am), color="blue", kde=False)
        med_disti_filename = ('%s/logdistam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log amount of mitochondria in distal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log amount of mitochondria in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        sns.distplot(np.log10(prox_am), color="blue", kde=False)
        med_disti_filename = ('%s/logproxam_%s_%s.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log amount of mitochondrial in proximal %s in %.s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('log amount of mitochondria in 10^ ')
        plt.ylabel('count of cells')
        plt.savefig(med_disti_filename)
        #plots for bin of frequency of different mitochondrial length
        bin_name = ("%s/bin_plots_%s" % (f_name, ct_dict[celltype]))
        if not os.path.exists(bin_name):
            os.mkdir(bin_name)
        for li in range(len(prox_bin_dict["abs1"])):
            prox_bin_dict["abs1"][li] = prox_bin_dict["abs1"][li][np.nonzero(prox_bin_dict["abs1"][li])[0]]
            prox_bin_dict["rel1"][li] = prox_bin_dict["rel1"][li][np.nonzero(prox_bin_dict["rel1"][li])[0]]
            dist_bin_dict["abs1"][li] = dist_bin_dict["abs1"][li][np.nonzero(dist_bin_dict["abs1"][li])[0]]
            dist_bin_dict["rel1"][li] = dist_bin_dict["rel1"][li][np.nonzero(dist_bin_dict["rel1"][li])[0]]
            if len(prox_bin_dict["abs1"][li]) > 0:
                sns.distplot(prox_bin_dict["abs1"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs1_%i_%s_%s_n.png' % (bin_name,int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in proximal %s in %.4s' % (int(li+1),comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel1"][li]) > 0:
                sns.distplot(prox_bin_dict["rel1"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in proximal %s in %.4s' % (int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs1"][li]) > 0:
                sns.distplot(dist_bin_dict["abs1"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in distal %s in %.4s' % (int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel1"][li]) > 0:
                sns.distplot(dist_bin_dict["rel1"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel1_%i_%s_%s.png' % (bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in distal %s in %.4s' % (
                int(li + 1), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        for li in range(len(prox_bin_dict["abs1_5"])):
            prox_bin_dict["abs1_5"][li] = prox_bin_dict["abs1_5"][li][np.nonzero(prox_bin_dict["abs1_5"][li])[0]]
            prox_bin_dict["rel1_5"][li] = prox_bin_dict["rel1_5"][li][np.nonzero(prox_bin_dict["rel1_5"][li])[0]]
            dist_bin_dict["abs1_5"][li] = dist_bin_dict["abs1_5"][li][np.nonzero(dist_bin_dict["abs1_5"][li])[0]]
            dist_bin_dict["rel1_5"][li] = dist_bin_dict["rel1_5"][li][np.nonzero(dist_bin_dict["rel1_5"][li])[0]]
            if len(prox_bin_dict["abs1_5"][li]) > 0:
                sns.distplot(prox_bin_dict["abs1_5"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %.1f µm in proximal %s in %.4s' % ((li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel1_5"][li]) > 0:
                sns.distplot(prox_bin_dict["rel1_5"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %.1f µm in proximal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs1_5"][li]) > 0:
                sns.distplot(dist_bin_dict["abs1_5"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %.1f µm in distal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel1_5"][li]) > 0:
                sns.distplot(dist_bin_dict["rel1_5"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel1_5_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %.1f µm in distal %s in %.4s' % (
                (li + 1.5), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        for li in range(len(prox_bin_dict["abs2"])):
            prox_bin_dict["abs2"][li] = prox_bin_dict["abs2"][li][np.nonzero(prox_bin_dict["abs2"][li])[0]]
            prox_bin_dict["rel2"][li] = prox_bin_dict["rel2"][li][np.nonzero(prox_bin_dict["rel2"][li])[0]]
            dist_bin_dict["abs2"][li] = dist_bin_dict["abs2"][li][np.nonzero(dist_bin_dict["abs2"][li])[0]]
            dist_bin_dict["rel2"][li] = dist_bin_dict["rel2"][li][np.nonzero(dist_bin_dict["rel2"][li])[0]]
            if len(prox_bin_dict["abs2"][li]) > 0:
                sns.distplot(prox_bin_dict["abs2"][li], color="blue", kde=False)
                bin_proxa_filename = ('%s/prox_binabs2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in proximal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxa_filename)
                plt.close()
            if len(prox_bin_dict["rel2"][li]) > 0:
                sns.distplot(prox_bin_dict["rel2"][li], color="blue", kde=False)
                bin_proxr_filename = ('%s/prox_binrel2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in proximal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_proxr_filename)
                plt.close()
            if len(dist_bin_dict["abs2"][li]) > 0:
                sns.distplot(dist_bin_dict["abs2"][li], color="blue", kde=False)
                bin_dista_filename = ('%s/dist_binabs2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('abs frequency of mitos up to %i µm in distal %s in %.4s' % (
                int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('absolute frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_dista_filename)
                plt.close()
            if len(dist_bin_dict["rel2"][li]) > 0:
                sns.distplot(dist_bin_dict["rel2"][li], color="blue", kde=False)
                bin_distr_filename = ('%s/dist_binrel2_%i_%s_%s.png' % (
                bin_name, int(li), ct_dict[celltype], comp_dict[compartment]))
                plt.title('rel frequency of mitos up to %i µm in distal %s in %.4s' % (
                    int(li + 2), comp_dict[compartment], ct_dict[celltype]))
                plt.xlabel('relative frequency')
                plt.ylabel('count of cells')
                plt.savefig(bin_distr_filename)
                plt.close()
        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('plots per celltype and compartment')
        log.info("finished plots for %s in compartment: %s" % (ct_dict[celltype], comp_dict[compartment]))


        return prox_mito_dict, dist_mito_dict


    msn_den_prox_dict, msn_den_dist_dict = ct_mito_analysis(ssd, celltype=2)
    msn_ax_prox_dict, msn_ax_dist_dict = ct_mito_analysis(ssd, celltype=2, compartment=1)
    #gp_den_prox_dict, gp_den_dist_dict = ct_mito_analysis(ssd, celltype=[6,7])
    #gp_ax_prox_dict, gp_ax_dist_dict = ct_mito_analysis(ssd, celltype=[6,7], compartment = 1)

    raise ValueError
