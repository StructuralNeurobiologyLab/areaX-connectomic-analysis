#script for looking at MSN percentile connectivity with GPe/i, FS, STN, TAN

if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.spiness_sorting import saving_spiness_percentiles
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    cl = 200
    syn_prob = 0.8
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/220425_j0251v4_MSN_percentile_comparison_mcl_%i_synprob_%.2f" % (cl, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN percentile comparison connectivity', log_dir=f_name + '/logs/')
    log.info("MSN percentile comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    #percentile = [10, 25, 50]
    #comp_lengths = [100, 200, 500, 1000]
    percentiles = [10, 25, 50]



    log.info("Step 1/11: sort MSN into different percentiles depending on spiness")
    #create MSN spiness percentiles with different comp_lengths

    filename_spiness_saving = "/wholebrain/scratch/arother/j0251v4_prep/"
    filename_spiness_results = "%s/spiness_percentiles_mcl%i" % (f_name, cl)
    """
    if not os.path.exists(filename_spiness_results):
        os.mkdir(filename_spiness_results)
    saving_spiness_percentiles(ssd, celltype = 2, filename_saving = filename_spiness_saving, filename_plotting = filename_spiness_results, percentiles = percentiles, min_comp_len = cl)
    
    
    time_stamps = [time.time()]
    step_idents = ["spiness percentiles calculated"]
    
    """

    MSN_id_dict = {}

    percentiles = [10, 25, 49]

    for p in percentiles:
        MSN_id_dict[p] = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN p%.2i_arr_c%i.pkl" % (p, cl))
        MSN_id_dict[100 - p] = load_pkl2obj(
            "/wholebrain/scratch/arother/j0251v4_prep/full_MSN p%.2i_arr_c%i.pkl" % (100 - p, cl))

    """

    log.info("Step 2/11: MSN percentile compartment comparison")
    # calculate parameters such as axon/dendrite length, volume, tortuosity and compare within celltypes
    for p in percentiles:
        result_MSN_filename_p1 = axon_den_arborization_ct(ssd, celltype=2, percentile = p, filename=f_name, full_cells=True, cellids = MSN_id_dict[p], min_comp_len = cl)
        result_MSN_filename_p2 = axon_den_arborization_ct(ssd, celltype=2, percentile = 100 - p, filename=f_name, full_cells=True, cellids = MSN_id_dict[100 - p], min_comp_len = cl)
        compare_compartment_volume_ct(celltype1=2, percentile = p, filename=f_name, filename1=result_MSN_filename_p1, filename2=result_MSN_filename_p2, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["compartment comparison finished"]
    


    log.info("Step 3/11: MSN connectivity between percentiles")
    # see how MSN percentiles are connected
    for p in percentiles:
        MSN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = MSN_id_dict[100 - p], min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, filename=f_name, foldername_ct1=MSN_connectivity_resultsfolder, foldername_ct2=MSN_connectivity_resultsfolder, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["connctivity between MSN percentiles finished"]


    log.info("Step 4/11: MSN - GPe connectivity different percentiles")
    # see how MSN percentiles are connected to GPe
    GPe_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPe_arr.pkl")

    #GPe_ids = load_pkl2obj(
        #"/wholebrain/scratch/arother/j0251v4_prep/full_GPe_arr_hp_v3.pkl")

    for p in percentiles:
        MSN_GPe_p1_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = GPe_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        MSN_GPe_p2_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[100 - p], cellids2 = GPe_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_gpe_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=6, filename=f_name, foldername_ct1=MSN_GPe_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPe_p2_connectivity_resultsfolder, min_comp_len = cl)

    log.info("Step 5/11: MSN - GPi connectivity different percentiles")
    # see how MSN percentiles are connected to GPi
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr.pkl")

    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr_hp_v3.pkl")
    for p in percentiles:
        MSN_GPi_p1_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = GPi_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        MSN_GPi_p2_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[100- p], cellids2 = GPi_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_gpi_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=7, filename=f_name, foldername_ct1=MSN_GPi_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPi_p2_connectivity_resultsfolder, min_comp_len = cl)

    log.info("Step 6/11: MSN - STN connectivity different percentiles")
    # see how MSN percentiles are connected to STN
    STN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_STN_arr.pkl")
    for p in percentiles:
        MSN_STN_p1_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = STN_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        MSN_STN_p2_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[100 - p], cellids2 = STN_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_stn_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=0, filename=f_name, foldername_ct1=MSN_STN_p1_connectivity_resultsfolder, foldername_ct2=MSN_STN_p2_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - STN finished"]

    log.info("Step 7/11: MSN - FS connectivity")
    # see how MSN percentiles are connected to FS
    FS_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_FS_arr.pkl")
    for p in percentiles:
        MSN_FS_p1_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = FS_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        MSN_FS_p2_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[100 - p], cellids2 = FS_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_fs_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=8, filename=f_name, foldername_ct1=MSN_FS_p1_connectivity_resultsfolder, foldername_ct2=MSN_FS_p2_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - FS finished"]

    log.info("Step 8/11: MSN - TAN connectivity")
    # see how MSN percentiles are connected to TAN
    TAN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_TAN_arr.pkl")
    for p in percentiles:
        MSN_TAN_p1_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[p], cellids2 = TAN_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        MSN_TAN_p2_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, cellids1 = MSN_id_dict[100 - p], cellids2 = TAN_ids, min_comp_len = cl, syn_prob_thresh = syn_prob)
        msn_tan_summed_synapses = compare_connectivity(comp_ct1=2, percentile= p, connected_ct=5, filename=f_name, foldername_ct1=MSN_TAN_p1_connectivity_resultsfolder, foldername_ct2=MSN_TAN_p2_connectivity_resultsfolder, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - TAN finished"]
    

    log.info("Step 9/11: MSN - HVC connectivity")
    # see how MSN percentiles are connected to TAN
    HVC_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == 4]
    for p in percentiles:
        MSN_HVC_p1_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1 = 4, filename = f_name, cellids1 = HVC_ids, celltype2= 2, cellids2 = MSN_id_dict[p], full_cells_ct2= True,
                         min_comp_len = cl, syn_prob_thresh = syn_prob, label_ct1 = None, label_ct2 = "MSN p%.2i" % p)
        MSN_HVC_p2_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=4, celltype2=2,
                                                                     filename=f_name,
                                                                     full_cells_ct2=True, cellids1=HVC_ids,
                                                                     cellids2=MSN_id_dict[100 - p], min_comp_len=cl,
                                                                     syn_prob_thresh=syn_prob, label_ct2 = "MSN p%.2i" % (100 - p))
        msn_hvc_summed_synapses = compare_connectivity(comp_ct1=2, percentile=p, connected_ct=4, filename=f_name,
                                                       foldername_ct1=MSN_HVC_p1_connectivity_resultsfolder,
                                                       foldername_ct2=MSN_HVC_p2_connectivity_resultsfolder,
                                                       min_comp_len=cl, label_ct1 = "MSN p%.2i" % p, label_ct2 = "MSN p%.2i" % (100 - p))

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - HVC finished"]
    """

    log.info("Step 10/11: MSN - LMAN connectivity")
    # see how MSN percentiles are connected to TAN
    LMAN_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == 3]
    for p in percentiles:
        MSN_LMAN_p1_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, filename=f_name,
                                                               cellids1=LMAN_ids, celltype2=2, cellids2=MSN_id_dict[p],
                                                               full_cells_ct2=True,
                                                               min_comp_len=cl, syn_prob_thresh=syn_prob,
                                                               label_ct1=None, label_ct2="MSN p%.2i" % p)
        MSN_LMAN_p2_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, celltype2=2,
                                                               filename=f_name,
                                                               full_cells_ct2=True, cellids1=LMAN_ids,
                                                               cellids2=MSN_id_dict[100 - p], min_comp_len=cl,
                                                               syn_prob_thresh=syn_prob,
                                                               label_ct2="MSN p%.2i" % (100 - p))
        msn_lman_summed_synapses = compare_connectivity(comp_ct1=2, percentile=p, connected_ct=3,filename=f_name,
                                                       foldername_ct1=MSN_LMAN_p1_connectivity_resultsfolder,
                                                       foldername_ct2=MSN_LMAN_p2_connectivity_resultsfolder,
                                                       min_comp_len=cl, label_ct1="MSN p%.2i" % p,
                                                       label_ct2="MSN p%.2i" % (100 - p))

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - LMAN finished"]

    raise ValueError


    log.info("Step 11/11 Overview Graph")
    # make connectivity overview graph with networkx
    #first put all dictionaries together
    sum_synapse_dict = {**msn_summed_synapses, **msn_gpe_summed_synapses, **msn_gpi_summed_synapses, **msn_stn_summed_synapses, **msn_tan_summed_synapses, **msn_fs_summed_synapses}
    write_obj2pkl("%s/ct_sum_synapses.pkl" % f_name, sum_synapse_dict)
    #plot
    sum_synapse_dict = load_pkl2obj("%s/ct_sum_synapses.pkl" % f_name)
    plot_nx_graph(sum_synapse_dict, filename = ("%s/summed_synapses_nx_overview_mcl%i.png" % (f_name, cl)), title = "sum of synapses between celltypes")

    msn_summed_synapse_pd = pd.DataFrame(sum_synapse_dict, index=[0])
    msn_summed_synapse_pd.to_csv("%s/ct_summed_synapses.csv" % f_name)


    log.info("MSN percentile compartment and connectivity analysis finished")
    step_idents = ["MSN percentile compartment and connectivity analysis finished"]