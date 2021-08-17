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

    def mito_myelin_percell(sso,min_ax_length = 0, compartment = 0, radius = 50):
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
        non_axon_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != 1)[0]
        non_myelin_inds = np.nonzero(sso.skeleton["myelin"] == 0)[0]
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "myelin"))
        axon_graph = g.copy()
        axon_graph.remove_nodes_from(non_axon_inds)
        axon_length = axon_graph.size(weight="weight") / 1000  # in µm
        if axon_length < min_ax_length:
            return np.zeros(6), -1, -1
        myelin_graph = axon_graph.copy()
        myelin_graph.remove_nodes_from(non_myelin_inds)
        absolute_myelin_length = myelin_graph.size(weight="weight") / 1000  # in µm
        relative_myelin_length = absolute_myelin_length / axon_length
        if compartment == 1:
                compartment_length = axon_length
        else:
                non_compartment_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
                g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
                compartment_graph = g.copy()
                compartment_graph.remove_nodes_from(non_compartment_inds)
                compartment_length = compartment_graph.size(weight = "weight") / 1000 #in µm
        kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"]*sso.scaling)
        mis = sso.mis
        all_ids = np.zeros(len(mis))
        #mitochrondrial index and density (every 10 µm)
        all_volume = np.zeros(len(mis))
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
            mi_diff = mi.mesh_bb[1] - mi.mesh_bb[0]
            mi_length = np.linalg.norm(mi_diff)/1000. #in µm
            all_ids[i] = mi.id
            all_volume[i] = mi.size
            all_milength[i] = mi_length
        all_inds = all_ids > 0
        if np.all(all_inds == False):
            all_mito_parameters_cell = np.zeros(6)
        else:
            all_ids = all_ids[all_inds]
            all_milength = all_milength[all_inds]
            all_volume = all_volume[all_inds]
            mito_amount = len(all_ids)
            average_volume = np.mean(all_volume)
            average_length = np.mean(all_milength)
            volume_density = np.sum(all_volume)/compartment_length
            length_density = np.sum(all_milength)/compartment_length
            amount_density = mito_amount/compartment_length
            all_mito_parameters_cell = [mito_amount, average_volume, average_length, volume_density, length_density, amount_density]
        return all_mito_parameters_cell, absolute_myelin_length, relative_myelin_length


    def ct_mito_analysis(ssd, celltype, compartment=0, min_ax_length = 0):
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
        f_name = ("u/arother/test_folder/210531_j0251v3_mitomyelin_GP_hp_reqax3mm")
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('mitchondrial_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        # cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        # cellids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        # gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        # gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPe_arr.pkl")
        gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPi_arr.pkl")
        cellids = np.hstack(np.array([gpeids, gpiids]))
        ct_dict = {6: "GP", 7: "GP"}
        celltype = celltype[0]

        mito_amount = np.zeros(len(cellids)) - 1
        average_volume = np.zeros(len(cellids)) -1
        average_length = np.zeros(len(cellids)) -1
        volume_density = np.zeros(len(cellids)) -1
        length_density = np.zeros(len(cellids)) -1
        amount_density = np.zeros(len(cellids)) -1
        absolute_myelin = np.zeros(len(cellids)) -1
        relative_myelin = np.zeros(len(cellids)) -1
        comp_dict = {1: 'axons', 0: 'dendrites'}
        log.info('Step 1/2 generating mitochondrial parameters per cell')
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            all_mito_params_cell, absolute_myelin_cell, relative_myelin_cell = mito_myelin_percell(cell, min_ax_length=min_ax_length, compartment= compartment)
            if absolute_myelin_cell == -1:
                continue
            if np.all(all_mito_params_cell == 0):
                continue
            mito_amount[i] = all_mito_params_cell[0]
            average_volume[i] = all_mito_params_cell[1]
            average_length[i] = all_mito_params_cell[2]
            volume_density[i] = all_mito_params_cell[3]
            length_density[i] = all_mito_params_cell[4]
            amount_density[i] = all_mito_params_cell[5]
            absolute_myelin[i] = absolute_myelin_cell
            relative_myelin[i] = relative_myelin_cell

        gpi_inds = np.in1d(cellids, gpiids)
        gpe_inds = np.in1d(cellids, gpeids)
        gpi_mito_amount = mito_amount[gpi_inds]
        gpi_average_volume = average_volume[gpi_inds]
        gpi_average_length = average_length[gpi_inds]
        gpi_volume_density = volume_density[gpi_inds]
        gpi_length_density = length_density[gpi_inds]
        gpi_amount_density = amount_density[gpi_inds]
        gpi_absolute_myelin = absolute_myelin[gpi_inds]
        gpi_relative_myelin = relative_myelin[gpi_inds]
        gpi_npos = gpi_average_volume >= 0
        gpi_mito_amount = gpi_mito_amount[gpi_npos]
        gpi_average_volume = gpi_average_volume[gpi_npos]
        gpi_average_length = gpi_average_length[gpi_npos]
        gpi_volume_density = gpi_volume_density[gpi_npos]
        gpi_length_density = gpi_length_density[gpi_npos]
        gpi_amount_density = gpi_amount_density[gpi_npos]
        gpi_absolute_myelin = gpi_absolute_myelin[gpi_npos]
        gpi_relative_myelin = gpi_relative_myelin[gpi_npos]

        gpe_mito_amount = mito_amount[gpe_inds]
        gpe_average_volume = average_volume[gpe_inds]
        gpe_average_length = average_length[gpe_inds]
        gpe_volume_density = volume_density[gpe_inds]
        gpe_length_density = length_density[gpe_inds]
        gpe_amount_density = amount_density[gpe_inds]
        gpe_absolute_myelin = absolute_myelin[gpe_inds]
        gpe_relative_myelin = relative_myelin[gpe_inds]
        gpe_npos = gpe_average_volume >= 0
        gpe_mito_amount = gpe_mito_amount[gpe_npos]
        gpe_average_volume = gpe_average_volume[gpe_npos]
        gpe_average_length = gpe_average_length[gpe_npos]
        gpe_volume_density = gpe_volume_density[gpe_npos]
        gpe_length_density = gpe_length_density[gpe_npos]
        gpe_amount_density = gpe_amount_density[gpe_npos]
        gpe_absolute_myelin = gpe_absolute_myelin[gpe_npos]
        gpe_relative_myelin = gpe_relative_myelin[gpe_npos]

        gp_npos = average_volume >= 0
        pos_cellids = cellids[gp_npos]
        mito_amount = mito_amount[gp_npos]
        average_volume = average_volume[gp_npos]
        average_length = average_length[gp_npos]
        volume_density = volume_density[gp_npos]
        length_density = length_density[gp_npos]
        amount_density = amount_density[gp_npos]
        absolute_myelin = absolute_myelin[gp_npos]
        relative_myelin = relative_myelin[gp_npos]

        #make table with GP_ids, seprating according to median in relative myelin and volume density
        myelin_mito_data = np.zeros((len(pos_cellids), 5))
        median_rel_myelin = np.median(relative_myelin)
        median_mito_vol_den = np.median(volume_density)
        for gi, gp_id in enumerate(pos_cellids):
            myelin_mito_data[gi, 0] = gp_id
            myelin_mito_data[gi, 3] = volume_density[gi]
            myelin_mito_data[gi, 4] = relative_myelin[gi]
            if gp_id in gpeids:
                myelin_mito_data[gi, 2] = 0
            else:
                myelin_mito_data[gi, 2] = 1
            if volume_density[gi] <= median_mito_vol_den:
                if relative_myelin[gi] <= median_rel_myelin:
                    myelin_mito_data[gi, 1] = 3
                else:
                    myelin_mito_data[gi, 1] = 4
            else:
                if relative_myelin[gi] <= median_rel_myelin:
                    myelin_mito_data[gi, 1] = 2
                else:
                    myelin_mito_data[gi, 1] = 1
        rel_mylein_mito_vol_den = pd.DataFrame(data = myelin_mito_data, columns = ["GP_ids", "Quadrant", "GPi", "mito volume density", "relative myelin length"])
        rel_mylein_mito_vol_den.to_csv("%s/rel_myelin_vol_den_GPqdata_%s.csv" % (f_name, comp_dict[compartment]))


        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")

        #scatterplots for myelin and mitochondria
        # only GP
        plt.scatter(x = mito_amount, y=absolute_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoamount_absmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount of mitochondria')
        plt.ylabel('myelin pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=mito_amount, y=relative_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoamount_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of myelin')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=volume_density, y=absolute_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitovolden_absmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('volume density [voxel/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=volume_density, y=relative_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitovolden_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('volume density [voxel/µm]')
        plt.ylabel('fraction of myelin')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=length_density, y=absolute_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitolenden_absmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('length density [µm/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=length_density, y=relative_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_lenden_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('length density [µm/µm]')
        plt.ylabel('fraction of myelin')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=amount_density, y=absolute_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoamden_absmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount density [1/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=amount_density, y=relative_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoamden_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount density [1/µm]')
        plt.ylabel('fraction of myelin')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=average_volume, y=absolute_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoavgvol_absmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('average mitochondria volume in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('mitochondria volume [voxel]')
        plt.ylabel('myelin pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=average_volume, y=relative_myelin, c="black", alpha = 0.7)
        avg_filename = ('%s/%s_mitoavgvol_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('average mitochondria volume in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('mitochondria volume [voxel]')
        plt.ylabel('fraction of myelin')
        plt.savefig(avg_filename)
        plt.close()

        #plots with GPi and GPe
        plt.scatter(x=gpi_mito_amount, y=gpi_absolute_myelin, c="springgreen", label = "GPi", alpha = 0.7)
        plt.scatter(x=gpe_mito_amount, y=gpe_absolute_myelin, c="mediumorchid", label="GPe", alpha = 0.7)
        avg_filename = ('%s/%s_mitoamount_absmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount of mitochondria')
        plt.ylabel('myelin pathlength [µm]')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_mito_amount, y=gpi_relative_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_mito_amount, y=gpe_relative_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitoamount_relmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of myelin')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_volume_density, y=gpi_absolute_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_volume_density, y=gpe_absolute_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitovolden_absmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('volume density [voxel/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_volume_density, y=gpi_relative_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_volume_density, y=gpe_relative_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitovolden_relmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('volume density [voxel/µm]')
        plt.ylabel('fraction of myelin')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_length_density, y=gpi_absolute_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_length_density, y=gpe_absolute_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitolenden_absmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('length density [µm/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_length_density, y=gpi_relative_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_length_density, y=gpe_relative_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_lenden_relmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('length density [µm/µm]')
        plt.ylabel('fraction of myelin')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_amount_density, y=gpi_absolute_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_amount_density, y=gpe_absolute_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitoamden_absmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount density [1/µm]')
        plt.ylabel('myelin pathlength [µm]')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_amount_density, y=gpi_relative_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_amount_density, y=gpe_relative_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitoamden_relmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('mitochondria density in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('amount density [1/µm]')
        plt.ylabel('fraction of myelin')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_average_volume, y=gpi_absolute_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_average_volume, y=gpe_absolute_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitoavgvol_absmyelin_GPei.png' % (f_name, comp_dict[compartment]))
        plt.title('average mitochondria volume in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('mitochondria volume [voxel]')
        plt.ylabel('myelin pathlength [µm]')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=gpi_average_volume, y=gpi_relative_myelin, c="springgreen", label="GPi", alpha=0.7)
        plt.scatter(x=gpe_average_volume, y=gpe_relative_myelin, c="mediumorchid", label="GPe", alpha=0.7)
        avg_filename = ('%s/%s_mitoavgvol_relmyelin_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('average mitochondria volume in %s vs myelin length in GP' % comp_dict[compartment])
        plt.xlabel('mitochondria volume [voxel]')
        plt.ylabel('fraction of myelin')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #distplots for mitos and mylein
        sns.distplot(mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_mitoamount_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in GP')
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(mito_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_mitoamount_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in GP')
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_avgmitolength_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Average length of mitochondria in GP')
        plt.xlabel('average length in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_avgmitolength_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Average length of mitochondria in GP')
        plt.xlabel('average length in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_avgmitovol_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Average volume of mitochondria in GP')
        plt.xlabel('average volume in voxel')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_volume, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_avgmitovol_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Average volume of mitochondria in GP')
        plt.xlabel('average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_voldenmito_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Volume density of mitochondria in GP')
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(volume_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_voldenmito_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Volume density of mitochondria in GP')
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_lendenmito_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Length density of mitochondria in GP')
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(length_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_lendenmito_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Length density of mitochondria in GP')
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_amdenmito_GP.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria per µm in GP')
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(amount_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpe_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_amdenmito_GP_norm.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria per µm in GP')
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #all plots without all GPs
        sns.distplot(gpe_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_mitoamount_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in GP')
        plt.xlabel('amount of mitochondria')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_mito_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_mitoamount_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria in GP')
        plt.xlabel('amount of mitochondria')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_avgmitolength_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Average length of mitochondria in GP')
        plt.xlabel('average length in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_avgmitolength_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Average length of mitochondria in GP')
        plt.xlabel('average length in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_avgmitovol_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Average volume of mitochondria in GP')
        plt.xlabel('average volume in voxel')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_volume,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_avgmitovol_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Average volume of mitochondria in GP')
        plt.xlabel('average volume in voxel')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_voldenmito_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Volume density of mitochondria in GP')
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_volume_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_voldenmito_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Volume density of mitochondria in GP')
        plt.xlabel('volume density in voxel/µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_lendenmito_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Length density of mitochondria in GP')
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_length_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_lendenmito_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Length density of mitochondria in GP')
        plt.xlabel('mito length [µm] per pathlength[µm]')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/%s_amdenmito_GP_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria per µm in GP')
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_amount_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/%s_amdenmito_GP_norm_sp.png' % (f_name, comp_dict[compartment]))
        plt.title('Amount of mitochondria per µm in GP')
        plt.xlabel('amount of mitochondria per µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #myelin plots
        #plot only when compartment = 1 to avoid printing it twice

        if compartment == 1:

            bin_amount = 10

            sns.distplot(absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False,
                         label="GP", norm_hist=True, bins=bin_amount)
            sns.distplot(gpe_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", norm_hist=True, bins=bin_amount)
            sns.distplot(gpi_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", norm_hist=True, bins=bin_amount)
            avg_filename = ('%s/absmylen_GP_norm.png' % f_name)
            plt.title('Absolute length of axon with myelin in GP')
            plt.xlabel('pathlength in µm')
            plt.ylabel('fraction of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False, label="GP", bins=bin_amount)
            sns.distplot(gpe_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", bins=bin_amount)
            sns.distplot(gpi_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", bins=bin_amount)
            avg_filename = ('%s/absmylen_GP.png' % f_name)
            plt.title('Absolute length of axon with myelin in GP')
            plt.xlabel('pathlength in µm')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False,
                         label="GP", norm_hist=True, bins=bin_amount)
            sns.distplot(gpe_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", norm_hist=True, bins=bin_amount)
            sns.distplot(gpi_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", norm_hist=True, bins=bin_amount)
            avg_filename = ('%s/relmylen_GP_norm.png' % f_name)
            plt.title('Relative length of axon with myelin in GP')
            plt.xlabel('fraction of mylein')
            plt.ylabel('fraction of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False, label="GP", bins=bin_amount)
            sns.distplot(gpe_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", bins=bin_amount)
            sns.distplot(gpi_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", bins=bin_amount)
            avg_filename = ('%s/relmylen_GP.png' % f_name)
            plt.title('Relative length of axon with myelin in GP')
            plt.xlabel('fraction of myelin')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            # plot without GP
            sns.distplot(gpe_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", norm_hist=True, bins=bin_amount)
            sns.distplot(gpi_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", norm_hist=True, bins=bin_amount)
            avg_filename = ('%s/absmylen_GP_norm_sp.png' % f_name)
            plt.title('Absolute length of axon with myelin in GP')
            plt.xlabel('pathlength in µm')
            plt.ylabel('fraction of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(gpe_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", bins=bin_amount)
            sns.distplot(gpi_absolute_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", bins=bin_amount)
            avg_filename = ('%s/absmylen_GP_sp.png' % f_name)
            plt.title('Absolute length of axon with myelin in GP')
            plt.xlabel('pathlength in µm')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(gpe_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", norm_hist=True, bins=bin_amount)
            sns.distplot(gpi_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", norm_hist=True, bins=bin_amount)
            avg_filename = ('%s/relmylen_GP_norm_sp.png' % f_name)
            plt.title('Relative length of axon with myelin in GP')
            plt.xlabel('fraction of mylein')
            plt.ylabel('fraction of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

            sns.distplot(gpe_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label="GPe", bins=bin_amount)
            sns.distplot(gpi_relative_myelin,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label="GPi", bins=bin_amount)
            avg_filename = ('%s/relmylen_GP_sp.png' % f_name)
            plt.title('Relative length of axon with myelin in GP')
            plt.xlabel('fraction of myelin')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avg_filename)
            plt.close()

        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('mito myelin for GP finished')


    ct_mito_analysis(ssd, celltype=[6, 7], compartment=1, min_ax_length=3000)
    ct_mito_analysis(ssd, celltype=[6, 7], compartment=0, min_ax_length=3000)
