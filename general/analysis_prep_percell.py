if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import time
    import os
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    from syconn.handler.config import initialize_logging
    from syconn.mp import batchjob_utils as qu
    from syconn.handler.basics import chunkify_weighted
    from syconn.handler.config import initialize_logging
    from syconn.mp import batchjob_utils as qu
    from syconn import global_params
    import shutil

    global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10

    start = time.time()
    f_name = "/wholebrain/scratch/arother/j126_prep"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('analysis prep', log_dir=f_name + '/logs/')
    ct_list = [2]
    #ct_list = [2,5, 6, 7, 8, 9, 10]
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9:"LTS", 10:"NGF"}
    curr_time = time.time() - start
    max_n_jobs = min(global_params.config.ncore_total * 4, 1000)
    for ix, ct in enumerate(ct_list):
        log.info('Step %.1i/%.1i find full cells of celltype %.3s' % (ix+1,len(ct_list), ct_dict[ct]))
        cell_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == ct]
        multi_params = chunkify_weighted(cell_ids, max_n_jobs, ssd.load_cached_data('size')[ssd.load_cached_data("celltype_cnn_e3") == ct])
        out_dir = qu.batchjob_script(multi_params, "findfullcells", log=log, remove_jobfolder=False, n_cores=1,
                           max_iterations=10)
        cell_array = np.zeros(len(cell_ids))
        somas = np.zeros((len(cell_ids), 3))
        for i, file in enumerate(out_dir):
            part_list = load_pkl2obj(file)
            if len(part_list) != 2:
                continue
            part_cell_array = part_list[0]
            part_soma_centers = part_list[1]
            start = i*len(part_cell_array)
            cell_array[start:start+len(part_cell_array)-1] = part_cell_array.astype(int)
            somas[start:start+len(part_cell_array)-1] = part_soma_centers

        inds = np.array(cell_array != 0)
        cell_array = cell_array[inds]
        somas = somas[inds]
        cell_dict = {int(cell_array[i]): somas[i] for i in range(0, len(cell_array))}
        dict_path = ("%s/full_%.3s_dict.pkl" % (f_name,ct_dict[ct]))
        arr_path = ("%s/full_%.3s_arr.pkl" % (f_name,ct_dict[ct]))
        write_obj2pkl(dict_path, cell_dict)
        write_obj2pkl(arr_path, cell_array)
        shutil.rmtree(os.path.abspath(out_dir + '/../'))
        curr_time -= time.time()
        print("%.2f min, %.2f sec for finding  %s cells" % (curr_time // 60, curr_time % 60, ct_dict[ct]))

    raise ValueError
