if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os as os
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from tqdm import tqdm
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

    def myelin_sso(sso):
        '''
        calculates the absolute and relative length per cell
        :param sso: cell
        :return: abs_myelin_length, rel_myelin_length
        '''
        sso.load_skeleton()
        non_axon_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != 1)[0]
        non_myelin_inds = np.nonzero(sso.skeleton["myelin"] == 0)[0]
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "myelin"))
        axon_graph = g.copy()
        axon_graph.remove_nodes_from(non_axon_inds)
        axon_length = axon_graph.size(weight = "weight")/ 1000 #in µm
        myelin_graph = axon_graph.copy()
        myelin_graph.remove_nodes_from(non_myelin_inds)
        absolute_myelin_length = myelin_graph.size(weight = "weight") / 1000 #in µm
        relative_myelin_length = absolute_myelin_length/axon_length

        return absolute_myelin_length, relative_myelin_length


    def myelin_analysis(ssd, celltype):
        '''
        uses myelin_sso to calculate amount of mylein (absolute and relative) per cell. If GP: will compare GPe, GPi
        :param ssd: supersegmentation dataset
        :param celltype: 0251:ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        :return:
        '''
        start = time.time()
        f_name = "u/arother/test_folder/210527_myelin_GP_hp"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        ct_dict_GP = {6: "GP"}
        # cellids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        # gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        # gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[0]])
        gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[1]])
        cellids = np.hstack(np.array([gpeids, gpiids]))
        celltype = celltype[0]
        log.info('Step 1/2 myelin analysis per cell')
        absolute_myelin_length= np.zeros(len(cellids))
        relative_myelin_length = np.zeros(len(cellids))
        for pi, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            absolute_myelin_cell, relative_myelin_cell = myelin_sso(cell)
            absolute_myelin_length[pi] = absolute_myelin_cell
            relative_myelin_length[pi] = relative_myelin_cell

        gpi_inds = np.in1d(cellids, gpiids)
        gpe_inds = np.in1d(cellids, gpeids)
        gpi_absolute_myelin_length = absolute_myelin_length[gpi_inds]
        gpi_relative_myelin_length = relative_myelin_length[gpi_inds]
        gpe_absolute_myelin_length = absolute_myelin_length[gpe_inds]
        gpe_relative_myelin_length = relative_myelin_length[gpe_inds]

        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")

        bin_amount = 10

        sns.distplot(absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True, bins=bin_amount)
        sns.distplot(gpe_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True, bins=bin_amount)
        sns.distplot(gpi_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True, bins=bin_amount)
        avg_filename = ('%s/absmylen_GP_norm.png' % f_name)
        plt.title('Absolute length of axon with myelin in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", bins=bin_amount)
        sns.distplot(gpe_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", bins=bin_amount)
        sns.distplot(gpi_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", bins=bin_amount)
        avg_filename = ('%s/absmylen_GP.png' % f_name)
        plt.title('Absolute length of axon with myelin in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP", norm_hist=True, bins=bin_amount)
        sns.distplot(gpe_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True, bins=bin_amount)
        sns.distplot(gpi_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True, bins=bin_amount)
        avg_filename = ('%s/relmylen_GP_norm.png' % f_name)
        plt.title('Relative length of axon with myelin in GP')
        plt.xlabel('fraction of mylein')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", bins=bin_amount)
        sns.distplot(gpe_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", bins=bin_amount)
        sns.distplot(gpi_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", bins=bin_amount)
        avg_filename = ('%s/relmylen_GP.png' % f_name)
        plt.title('Relative length of axon with myelin in GP')
        plt.xlabel('fraction of myelin')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #plot without GP
        sns.distplot(gpe_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True, bins=bin_amount)
        sns.distplot(gpi_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True, bins=bin_amount)
        avg_filename = ('%s/absmylen_GP_norm_sp.png' % f_name)
        plt.title('Absolute length of axon with myelin in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", bins=bin_amount)
        sns.distplot(gpi_absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", bins=bin_amount)
        avg_filename = ('%s/absmylen_GP_sp.png' % f_name)
        plt.title('Absolute length of axon with myelin in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True, bins=bin_amount)
        sns.distplot(gpi_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True, bins=bin_amount)
        avg_filename = ('%s/relmylen_GP_norm_sp.png' % f_name)
        plt.title('Relative length of axon with myelin in GP')
        plt.xlabel('fraction of mylein')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", bins=bin_amount)
        sns.distplot(gpi_relative_myelin_length,
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

        log.info('myelin axo analysis for GP finished')

    myelin_analysis(ssd, celltype=[6,7])