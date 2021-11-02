if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import networkx as nx
    import os as os
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from tqdm import tqdm
    from syconn.handler.basics import write_obj2pkl
    from u.arother.bio_analysis.general.analysis_helper import get_spine_density

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    def saving_spiness_percentiles(ssd, celltype, full_cells = True, percentiles = [], min_comp_len = 100):
        """
        saves MSN IDS depending on their spiness. Spiness is determined via spiness skeleton nodes using counting_spiness as spine density in relation to the skeelton length
        of the dendrite. Cells without a minimal compartment length are discarded.
        :param ssd: super segmentation dataset
        :param celltype: ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
        :param full_cells: if True, cellids of preprocessed cells with axon, dendrite, soma are loaded
        :param percentiles: list of percentiles, that should be saved
        :param min_comp_len: minimal compartment length in µm
        :return:
        """
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "/wholebrain/scratch/arother/j0251v3_prep"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sorting cellids according to spiness', log_dir=f_name + '/logs/')
        log.info(
            "parameters: celltype = %s, min_comp_length = %.i" %
            (ct_dict[celltype], min_comp_len))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if full_cells:
            cellids = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        else:
            cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]

        log.info("Step 1/2: iterate over cells and get amount of spines")
        spine_densities = np.zeros(len(cellids))
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            spine_density  = get_spine_density(cell, min_comp_len=min_comp_len)
            if spine_density == 0:
                continue
            spine_densities[i] = spine_density

        spine_inds = spine_densities > 0
        spine_densities = spine_densities[spine_inds]
        cellids = cellids[spine_inds]

        spinetime = time.time() - start
        print("%.2f sec for iterating through %s cells" % (spinetime, ct_dict[celltype]))
        time_stamps.append(time.time())
        step_idents.append('iterating over %s cells' % ct_dict[celltype])

        log.info("Step 2/2 sort into percentiles")

        for percentile in percentiles:
            perc_low = np.percentile(spine_densities, percentile, interpolation="higher")
            perc_low_inds = np.where(spine_densities < perc_low)[0]
            perc_high = np.percentile(spine_densities, 100 - percentile, interpolation="lower")
            perc_high_inds = np.where(spine_densities > perc_high)[0]
            cellids_low = cellids[perc_low_inds]
            cellids_high = cellids[perc_high_inds]
            write_obj2pkl("%s/full_%3s_arr_%i_l_%i.pkl" % (f_name, ct_dict[celltype], percentile, min_comp_len), cellids_low)
            write_obj2pkl("%s/full_%3s_arr_%i_h_%i.pkl" % (f_name, ct_dict[celltype], 100 - percentile, min_comp_len), cellids_high)
            spine_amount_dict_low = {cellid: spine_amount for cellid, spine_amount in zip(cellids_low, spine_densities[perc_low_inds])}
            spine_amount_dict_high = {cellid: spine_amount for cellid, spine_amount in
                                     zip(cellids_high, spine_densities[perc_high_inds])}
            write_obj2pkl("%s/full_%3s_spine_dict_%i_%i.pkl" % (f_name, ct_dict[celltype], percentile, min_comp_len), spine_amount_dict_low)
            write_obj2pkl("%s/full_%3s_spine_dict_%i_%i.pkl" % (f_name, ct_dict[celltype], 100 - percentile, min_comp_len),
                          spine_amount_dict_high)

        perctime = time.time() - spinetime
        print("%.2f sec for creating percentile arrays" % perctime)
        time_stamps.append(time.time())
        step_idents.append('creating and saving percentile arrays')
        log.info("Analysis finished")

        raise ValueError

    comp_lengths = [100, 200, 500, 1000]
    for comp_length in comp_lengths:
        saving_spiness_percentiles(ssd, celltype=2, min_comp_len=comp_length, percentiles=[10, 25, 50])




