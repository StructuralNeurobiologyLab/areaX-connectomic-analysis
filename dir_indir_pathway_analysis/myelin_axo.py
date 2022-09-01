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
        if axon_length == 0:
            return 0,0,0
        myelin_graph = axon_graph.copy()
        myelin_graph.remove_nodes_from(non_myelin_inds)
        absolute_myelin_length = myelin_graph.size(weight = "weight") / 1000 #in µm
        relative_myelin_length = absolute_myelin_length/axon_length

        return axon_length, absolute_myelin_length, relative_myelin_length


    def myelin_analysis(ssd, celltype):
        '''
        uses myelin_sso to calculate amount of mylein (absolute and relative) per cell
        :param ssd: supersegmentation dataset
        :param celltype: 0251:ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        :return:
        '''
        start = time.time()
        f_name = "u/arother/test_folder/210504_myelin_MSN_full"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        # cellids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        log.info('Step 1/2 myelin analysis per cell')
        absolute_myelin_length= np.zeros(len(cellids))
        relative_myelin_length = np.zeros(len(cellids))
        axon_length = np.zeros(len(cellids))
        for pi, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            axon_length_cell, absolute_myelin_cell, relative_myelin_cell = myelin_sso(cell)
            if axon_length_cell == 0:
                continue
            axon_length[pi] = axon_length_cell
            absolute_myelin_length[pi] = absolute_myelin_cell
            relative_myelin_length[pi] = relative_myelin_cell

        axon_inds = axon_length > 0
        absolute_myelin_length = absolute_myelin_length[axon_inds]
        relative_myelin_length = relative_myelin_length[axon_inds]

        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")


        sns.distplot(absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/absmylen_%s_norm.png' % (f_name, ct_dict[celltype]))
        plt.title('Absolute length of axon with myelin of mitochondria in %s' % ct_dict[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(absolute_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/absmylen_%s.png' %  (f_name, ct_dict[celltype]))
        plt.title('Absolute length of axon with myelin of mitochondria in %s' % ct_dict[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/relmylen_%s_norm.png' % (f_name, ct_dict[celltype]))
        plt.title('Relative length of axon with myelin of mitochondria in %s' % ct_dict[celltype])
        plt.xlabel('fraction of mylein')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(relative_myelin_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/relmylen_%s.png' % (f_name, ct_dict[celltype]))
        plt.title('Relative length of axon with myelin of mitochondria in %s' % ct_dict[celltype])
        plt.xlabel('fraction of myelin')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('myelin axo analysis for %s finished' % ct_dict[celltype])

    myelin_analysis(ssd, celltype=2)