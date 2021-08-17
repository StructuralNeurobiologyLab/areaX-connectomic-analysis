if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import time
    import os
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    from analysis_prep_func import find_full_cells, synapse_amount_percell
    from syconn.handler.config import initialize_logging

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10

    start = time.time()
    f_name = "/wholebrain/scratch/arother/j0251v3_prep"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('analysis prep', log_dir=f_name + '/logs/')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    ct_list = [0]
    #ax_list = [3, 4]
    #ct_list = [2,5, 6, 7, 8, 9, 10]
    #ax_list = [3, 4, 0, 1]
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9:"LTS", 10:"NGF"}
    curr_time = time.time() - start

    for ix, ct in enumerate(ct_list):
        log.info('Step %.1i/%.1i find full cells of celltype %.3s' % (ix+1,len(ct_list), ct_dict[ct]))
        cell_array, cell_dict= find_full_cells(ssd, celltype=ct, shortestpaths=False)
        dict_path = ("%s/full_%.3s_dict.pkl" % (f_name,ct_dict[ct]))
        arr_path = ("%s/full_%.3s_arr.pkl" % (f_name,ct_dict[ct]))
        #syn_dict = synapse_amount_percell(ct, sd_synssv, syn_proba=0.6, cellids=cell_array)
        #syn_path = ("%s/full_%.3s_synam.pkl" % (f_name,ct_dict[ct]))
        write_obj2pkl(dict_path, cell_dict)
        write_obj2pkl(arr_path, cell_array)
        #write_obj2pkl(syn_path, syn_dict)
        curr_time -= time.time()
        print("%.2f min, %.2f sec for finding  %s cells" % (curr_time // 60, curr_time % 60, ct_dict[ct]))

    raise ValueError

    for ia, axct in enumerate(ax_list):
        log.info('Step %.1i/%.1i find synapse amount of celltype %.3s' % (ia + 1, len(ax_list), ct_dict[axct]))
        cell_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == axct]
        syn_dict = synapse_amount_percell(axct, sd_synssv, syn_proba=0.6, cellids=cell_ids)
        syn_path = ("%s/ax_%.3s_synam.pkl" % (f_name, ct_dict[axct]))
        write_obj2pkl(syn_path, syn_dict)
        curr_time -= time.time()
        print("%.2f min, %.2f sec for finding  %s cells" % (curr_time // 60, curr_time % 60, ct_dict[axct]))

    raise ValueError
