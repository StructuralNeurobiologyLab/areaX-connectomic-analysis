#script for looking at MSN percentile connectivity with GPe/i, FS, STN, TAN

if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity
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
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/220227_j0251v4_MSN_percentile_comparison"
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
    cl = 200

    log.info("Step 1/8: MSN percentile compartment comparison")
    #create MSN spiness percentiles with different comp_lengths

    filename_spiness_saving = "/wholebrain/scratch/arother/j0251v4_prep/"
    filename_spiness_results = "%s/spiness_percentiles_mcl%i" % (f_name, cl)
    if not os.path.exists(filename_spiness_results):
        os.mkdir(filename_spiness_results)
    saving_spiness_percentiles(ssd, celltype = 2, filename_saving = filename_spiness_saving, filename_plotting = filename_spiness_results, percentiles = percentiles, min_comp_len = cl)
    
    
    time_stamps = [time.time()]
    step_idents = ["spiness percentiles calculated"]

    
    p = 49

    log.info("Step 2/8: MSN percentile compartment comparison")
    # calculate parameters such as axon/dendrite length, volume, tortuosity and compare within celltypes
    result_MSN_filename_p1 = axon_den_arborization_ct(ssd, celltype=2, percentile = p, filename=f_name, full_cells=True, handpicked=False, min_comp_len = cl)
    result_MSN_filename_p2 = axon_den_arborization_ct(ssd, celltype=2, percentile = 100 - p, filename=f_name, full_cells=True, handpicked=False, min_comp_len = cl)
    compare_compartment_volume_ct(celltype1=2, percentile = p, filename=f_name, filename1=result_MSN_filename_p1, filename2=result_MSN_filename_p2, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["compartment comparison finished"]

    raise ValueError

    log.info("Step 3/8: MSN connectivity between percentiles")
    # see how MSN percentiles are connected
    MSN_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
    msn_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, filename=f_name, foldername_ct1=MSN_connectivity_resultsfolder, foldername_ct2=MSN_connectivity_resultsfolder, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["connctivity between MSN percentiles finished"]

    log.info("Step 4/8: MSN - GPe connectivity different percentiles")
    # see how MSN percentiles are connected to GPe
    MSN_GPe_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    MSN_GPe_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=6, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    msn_gpe_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=6, filename=f_name, foldername_ct1=MSN_GPe_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPe_p2_connectivity_resultsfolder, min_comp_len = cl)

    log.info("Step 5/8: MSN - GPi connectivity different percentiles")
    # see how MSN percentiles are connected to GPi
    MSN_GPi_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    MSN_GPi_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=7, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    msn_gpi_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=7, filename=f_name, foldername_ct1=MSN_GPi_p1_connectivity_resultsfolder, foldername_ct2=MSN_GPi_p2_connectivity_resultsfolder, min_comp_len = cl)

    log.info("Step 6/8: MSN - STN connectivity different percentiles")
    # see how MSN percentiles are connected to STN
    MSN_STN_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
    MSN_STN_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=0, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
    msn_stn_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=0, filename=f_name, foldername_ct1=MSN_STN_p1_connectivity_resultsfolder, foldername_ct2=MSN_STN_p2_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - STN finished"]

    log.info("Step 7/8: MSN - FS connectivity")
    # see how MSN percentiles are connected to FS
    MSN_FS_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
    MSN_FS_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=8, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=False, min_comp_len = cl)
    msn_fs_summed_synapses = compare_connectivity(comp_ct1=2, percentile = p, connected_ct=8, filename=f_name, foldername_ct1=MSN_FS_p1_connectivity_resultsfolder, foldername_ct2=MSN_FS_p2_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - FS finished"]

    log.info("Step 8/8: MSN - TAN connectivity")
    # see how MSN percentiles are connected to TAN
    MSN_TAN_p1_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    MSN_TAN_p2_connectivity_resultsfolder = synapses_between2cts(ssd, sd_synssv, celltype1=2, celltype2=5, percentile_ct1 = 100 - p, filename=f_name, full_cells=True, handpicked1=False, handpicked2=True, min_comp_len = cl)
    msn_tan_summed_synapses = compare_connectivity(comp_ct1=2, percentile= p, connected_ct=5, filename=f_name, foldername_ct1=MSN_TAN_p1_connectivity_resultsfolder, foldername_ct2=MSN_TAN_p2_connectivity_resultsfolder, min_comp_len = cl)
    
    time_stamps = [time.time()]
    step_idents = ["connctivity MSN - TAN finished"]


    log.info("Step 9/9 Overview Graph")
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