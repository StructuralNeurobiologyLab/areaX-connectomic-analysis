if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple, compare_soma_diameters
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_nx_graph
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    sd_csssv = SegmentationDataset("cs_ssv", working_dir=global_params.wd)
    cl_cell = 200
    cl_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    gpe_ct = 6
    gpi_ct = 7
    msn_ct = 3
    ct1_str = ct_dict[msn_ct]
    ct2_str = ct_dict[gpe_ct]
    ct3_str = ct_dict[gpi_ct]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/" \
             f"240220_j0251{version}_{ct1_str}_{ct2_str}_{ct3_str}_comparison_mcl_{cl_cell}_ax{cl_ax}_synprob_{syn_prob}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{ct1_str} comparison connectivity, mergers excluded', log_dir=f_name + '/logs/')
    log.info(f"{ct1_str} percentile comparison starts")
    f_name_saving = analysis_params.file_locations

    known_mergers = analysis_params.load_known_mergers()

    GPe_dict = analysis_params.load_cell_dict(gpe_ct)
    GPe_ids = np.array(list(GPe_dict.keys()))
    merger_inds = np.in1d(GPe_ids, known_mergers) == False
    GPe_ids = GPe_ids[merger_inds]
    GPe_ids = check_comp_lengths_ct(cellids=GPe_ids, fullcelldict=GPe_dict, min_comp_len=cl_cell,
                                    axon_only=False,
                                    max_path_len=None)

    GPi_dict = analysis_params.load_cell_dict(gpi_ct)
    GPi_ids = np.array(list(GPi_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=cl_cell,
                                    axon_only=False,
                                    max_path_len=None)

    MSN_dict = analysis_params.load_cell_dict(celltype=msn_ct)
    MSN_ids = np.array(list(MSN_dict.keys()))
    merger_inds = np.in1d(MSN_ids, known_mergers) == False
    MSN_ids = MSN_ids[merger_inds]
    misclassified_asto_ids = analysis_params.load_potential_astros()
    astro_inds = np.in1d(MSN_ids, misclassified_asto_ids) == False
    MSN_ids = MSN_ids[astro_inds]
    MSN_ids = check_comp_lengths_ct(cellids=MSN_ids, fullcelldict=MSN_dict, min_comp_len=cl_cell,
                                    axon_only=False,
                                    max_path_len=None)

    cell_dicts = [MSN_dict, GPe_dict, GPi_dict]

    log.info("Step 1/12: sort MSN based on connectivity to GPe and GPi")
    msn2gpe_ids, msn2gpi_ids, msn2gpei_ids, msn_nogp_ids = sort_by_connectivity(sd_synssv, sd_csssv=None, ct1=msn_ct, ct2=gpe_ct,
                                                                                ct3=gpi_ct, cellids1=MSN_ids,
                                                                                cellids2=GPe_ids, cellids3=GPi_ids,
                                                                                f_name=f_name, celldicts=cell_dicts,
                                                                                f_name_saving=f_name_saving,
                                                                                min_comp_len=cl_cell,
                                                                                syn_prob_thresh=syn_prob,
                                                                                min_syn_size=min_syn_size,
                                                                                ct_dict = ct_dict)


    time_stamps = [time.time()]
    step_idents = ['sort MSN via connectivity to GPe/i finished']

    labels_cts = ["MSN only GPe", "MSN only GPi", "MSN both GPs", "MSN no GPs"]
    msn_cts = [2, 2, 2, 2]
    msn_colors = ["#EAAE34", '#2F86A8', "#707070", "black"]

    log.info('Step 2/12: Compare new MSN groups based on soma size')
    msn_cellids = [msn2gpe_ids, msn2gpi_ids, msn2gpei_ids, msn_nogp_ids]
    compare_soma_diameters(cellids = msn_cellids, celltypes = labels_cts, colours = msn_colors, filename = f_name)


    log.info("Step 3/12: Compare new MSN groups based on morphology")
    MSN_only_GPe_results = axon_den_arborization_ct(ssd, celltype = 2, filename = f_name, cellids = msn2gpe_ids,
                                                    min_comp_len = cl_cell, full_cell_dict = MSN_dict, percentile = None, label_cts = "MSN only GPe", spiness = True)
    MSN_only_GPi_results = axon_den_arborization_ct(ssd, celltype=2, filename=f_name, cellids=msn2gpi_ids,
                                                    min_comp_len=cl_cell, full_cell_dict = MSN_dict, percentile=None,
                                                    label_cts="MSN only GPi", spiness =True)
    MSN_both_GP_results = axon_den_arborization_ct(ssd, celltype=2, filename=f_name, cellids=msn2gpei_ids,
                                                    min_comp_len=cl_cell, full_cell_dict = MSN_dict, percentile=None,
                                                    label_cts="MSN both GPs", spiness = True)
    MSN_no_GP_results = axon_den_arborization_ct(ssd, celltype=2, filename=f_name, cellids=msn_nogp_ids,
                                                    min_comp_len=cl_cell, full_cell_dict = MSN_dict, percentile=None,
                                                    label_cts="MSN no GPs", spiness = True)
    result_files = [MSN_only_GPe_results, MSN_only_GPi_results, MSN_both_GP_results, MSN_no_GP_results]
    compare_compartment_volume_ct_multiple(celltypes= msn_cts, filename=f_name, filename_cts=result_files, min_comp_len=cl_cell, label_cts=labels_cts,
                                           colours=msn_colors)
    #compartment comparision for multiple groups
    #also compare spiness
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups finished']


    log.info("Step 4/12: Compare connectivity between MSN groups")
    MSN_onlyGPe_MSN_both_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                              filename=f_name, full_cells=True,
                                                              cellids1=msn2gpe_ids, cellids2=msn2gpei_ids,
                                                              min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                              label_ct1="MSN only GPe", label_ct2="MSN both GPs",
                                                              version = version)
    MSN_onlyGPe_MSN_onlyGPi_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                                 filename=f_name, full_cells=True,
                                                                 cellids1=msn2gpe_ids, cellids2=msn2gpi_ids,
                                                                 min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                 label_ct1="MSN only GPe", label_ct2="MSN only GPi",
                                                                 version = version)
    MSN_onlyGPe_MSN_none_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                              filename=f_name, full_cells=True,
                                                              cellids1=msn2gpe_ids, cellids2=msn_nogp_ids,
                                                              min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                              label_ct1="MSN only GPe", label_ct2="MSN no GPs",
                                                              version = version)

    MSN_onlyGPi_MSN_both_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                              filename=f_name, full_cells=True,
                                                              cellids1=msn2gpi_ids, cellids2=msn2gpei_ids,
                                                              min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                              label_ct1="MSN only GPi", label_ct2="MSN both GPs",
                                                              version = version)
    MSN_onlyGPi_MSN_none_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                              filename=f_name, full_cells=True,
                                                              cellids1=msn2gpi_ids, cellids2=msn_nogp_ids,
                                                              min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                              label_ct1="MSN only GPi", label_ct2="MSN no GPs",
                                                              version = version)
    MSN_none_MSN_both_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=2,
                                                           filename=f_name, full_cells=True,
                                                           cellids1=msn_nogp_ids, cellids2=msn2gpei_ids,
                                                           min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                           label_ct1="MSN no GPs", label_ct2="MSN both GPs",
                                                           version = version)

    msn_onlyGPe_msn_summed_synapse = compare_connectivity_multiple(comp_cts = msn_cts[1:], filename = f_name,
                                                                   foldernames = [MSN_onlyGPe_MSN_onlyGPi_resultsfolder, MSN_onlyGPe_MSN_both_resultsfolder, MSN_onlyGPe_MSN_none_resultsfolder],
                                                                   connected_ct = 2, min_comp_len = cl_cell, label_cts =["MSN only GPi", "MSN both GPs", "MSN no GPs"],
                                                                   label_conn_ct = "MSN only GPe", colours = msn_colors[2:])

    msn_onlyGPi_msn_summed_synapse = compare_connectivity_multiple(comp_cts=msn_cts[2:], filename=f_name,
                                                                   foldernames=[MSN_onlyGPi_MSN_both_resultsfolder,
                                                                                MSN_onlyGPi_MSN_none_resultsfolder],
                                                                   connected_ct=2, min_comp_len=cl_cell,
                                                                   label_cts=["MSN both GPs",
                                                                              "MSN no GPs"],
                                                                   label_conn_ct="MSN only GPi", colours=msn_colors[1:])
    msn_both_msn_summed_synapse = compare_connectivity(comp_ct1=2, comp_ct2 = 2, filename=f_name,
                                                   foldername_ct1=MSN_none_MSN_both_resultsfolder,
                                                   label_ct1 = "MSN no GPs", label_ct2 = "MSN both GPs",
                                                   min_comp_len=cl_cell)
    # connectivity comparison for multiple groups
    time_stamps = [time.time()]
    step_idents = ['compare MSN within groups connectivity finished']


    log.info("Step 5/12: Compare connectivity of two MSN groups (both, only GPe) to GPe")
    MSN_onlyGPe_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=6,
                                                                 filename=f_name, full_cells=True,
                                                                 cellids1=msn2gpe_ids, cellids2=GPe_ids,
                                                                 min_comp_len=cl_cell, syn_prob_thresh=syn_prob, label_ct1 = "MSN only GPe", limit_multisynapse = 10,
                                                                version = version)
    MSN_bothGPs_GPe_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=6,
                                                                 filename=f_name,
                                                                 full_cells=True, cellids1=msn2gpei_ids,
                                                                 cellids2=GPe_ids, min_comp_len=cl_cell,
                                                                 syn_prob_thresh=syn_prob, label_ct1 = "MSN both GPs", limit_multisynapse = 10,
                                                                      version = version)
    msn_gpe_summed_synapses = compare_connectivity(comp_ct1=2, comp_ct2 = 2, connected_ct=6, filename=f_name,
                                                   foldername_ct1=MSN_onlyGPe_connectivity_resultsfolder,
                                                   foldername_ct2=MSN_bothGPs_GPe_connectivity_resultsfolder,
                                                   label_ct1 = "MSN only GPe", label_ct2 = "MSN both GPs",
                                                   min_comp_len=cl_cell)
    time_stamps = [time.time()]
    step_idents = ['compare MSN connectivity to GPe finsihed']

    log.info("Step 6/12: Compare connectivity of two MSN groups (both, only GPi) to GPi")
    MSN_onlyGPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=7,
                                                                  filename=f_name, full_cells=True,
                                                                  cellids1=msn2gpi_ids, cellids2=GPi_ids,
                                                                  min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                  label_ct1="MSN only GPi", limit_multisynapse = 10,
                                                                  version = version)
    MSN_bothGPs_GPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=2, celltype2=7,
                                                                  filename=f_name,
                                                                  full_cells=True, cellids1=msn2gpei_ids,
                                                                  cellids2=GPi_ids, min_comp_len=cl_cell,
                                                                  syn_prob_thresh=syn_prob, label_ct1="MSN both GPs", limit_multisynapse = 10,
                                                                    version = version)
    msn_gpi_summed_synapses = compare_connectivity(comp_ct1=2, comp_ct2=2, connected_ct=7, filename=f_name,
                                                   foldername_ct1=MSN_onlyGPi_connectivity_resultsfolder,
                                                   foldername_ct2=MSN_bothGPs_GPi_connectivity_resultsfolder,
                                                   label_ct1="MSN only GPi", label_ct2="MSN both GPs",
                                                   min_comp_len=cl_cell)
    time_stamps = [time.time()]
    step_idents = ['compare MSN connectivity to GPi finsihed']


    labels_cts = ["MSN only GPe", "MSN only GPi", "MSN both GPs", "MSN no GPs"]
    msn_cts = [2, 2, 2, 2]
    msn_colors = ["#EAAE34", '#2F86A8', "#707070", "black"]


    log.info("Step 7/12: Compare connectivity of MSN groups to FS")
    FS_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_FS_arr.pkl")
    MSN_FS_onlyGPe_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=8, filename=f_name,
                                                                cellids1=FS_ids, celltype2=2, cellids2=msn2gpe_ids,
                                                                full_cells=True,
                                                                min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN only GPe",
                                                                     version = version)
    MSN_FS_onlyGPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=8, filename=f_name,
                                                                cellids1=FS_ids, celltype2=2, cellids2=msn2gpi_ids,
                                                                full_cells=True,
                                                                min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN only GPi",
                                                                     version = version)
    MSN_FS_bothGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=8, filename=f_name,
                                                                cellids1=FS_ids, celltype2=2, cellids2=msn2gpei_ids,
                                                                full_cells=True,
                                                                min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN both GPs",
                                                                     version = version)
    MSN_FS_noGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=8, filename=f_name,
                                                              cellids1=FS_ids, celltype2=2, cellids2=msn_nogp_ids,
                                                              full_cells=True,
                                                              min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                              label_ct1=None, label_ct2="MSN no GPs", wd = global_params.wd, version = version)
    # connectivity comparison for multiple groups
    result_folders = [MSN_FS_onlyGPe_connectivity_resultsfolder, MSN_FS_onlyGPi_connectivity_resultsfolder, MSN_FS_bothGPs_connectivity_resultsfolder, MSN_FS_noGPs_connectivity_resultsfolder]
    msn_fs_summed_synapses = compare_connectivity_multiple(comp_cts = msn_cts, filename = f_name, foldernames = result_folders,
                                                            connected_ct = 8, min_comp_len = cl_cell, label_cts =labels_cts, colours = msn_colors)
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups connectivity to FS finished']

    log.info("Step 8/12: Compare connectivity of MSN groups to TAN")
    TAN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_TAN_arr.pkl")
    MSN_TAN_onlyGPe_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=5, filename=f_name,
                                                                     cellids1=TAN_ids, celltype2=2, cellids2=msn2gpe_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN only GPe",
                                                                      version = version)
    MSN_TAN_onlyGPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=5, filename=f_name,
                                                                     cellids1=TAN_ids, celltype2=2, cellids2=msn2gpi_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN only GPi",
                                                                      version = version)
    MSN_TAN_bothGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=5, filename=f_name,
                                                                     cellids1=TAN_ids, celltype2=2,
                                                                     cellids2=msn2gpei_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN both GPs",
                                                                      version = version)
    MSN_TAN_noGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=5, filename=f_name,
                                                                   cellids1=TAN_ids, celltype2=2, cellids2=msn_nogp_ids,
                                                                   full_cells=True,
                                                                   min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                   label_ct1=None, label_ct2="MSN no GPs",
                                                                    version = version)
    # connectivity comparison for multiple groups
    result_folders = [MSN_TAN_onlyGPe_connectivity_resultsfolder, MSN_TAN_onlyGPi_connectivity_resultsfolder,
                      MSN_TAN_bothGPs_connectivity_resultsfolder, MSN_TAN_noGPs_connectivity_resultsfolder]
    msn_tan_summed_synapses = compare_connectivity_multiple(comp_cts=msn_cts, filename=f_name,
                                                           foldernames=result_folders,
                                                           connected_ct=5, min_comp_len=cl_cell, label_cts=labels_cts,
                                                           colours=msn_colors)
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups connectivity to TAN finished']

    log.info("Step 9/12: Compare connectivity of MSN groups to STN")
    STN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_STN_arr.pkl")
    MSN_STN_onlyGPe_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=0, filename=f_name,
                                                                     cellids1=STN_ids, celltype2=2, cellids2=msn2gpe_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN only GPe",
                                                                      version = version)
    MSN_STN_onlyGPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=0, filename=f_name,
                                                                     cellids1=STN_ids, celltype2=2, cellids2=msn2gpi_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN only GPi",
                                                                      version = version)
    MSN_STN_bothGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=0, filename=f_name,
                                                                     cellids1=STN_ids, celltype2=2,
                                                                     cellids2=msn2gpei_ids,
                                                                     full_cells=True,
                                                                     min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                     label_ct1=None, label_ct2="MSN both GPs",
                                                                      version = version)
    MSN_STN_noGPs_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=0, filename=f_name,
                                                                   cellids1=STN_ids, celltype2=2, cellids2=msn_nogp_ids,
                                                                   full_cells=True,
                                                                   min_comp_len=cl_cell, syn_prob_thresh=syn_prob,
                                                                   label_ct1=None, label_ct2="MSN no GPs",
                                                                    version = version)
    # connectivity comparison for multiple groups
    result_folders = [MSN_STN_onlyGPe_connectivity_resultsfolder, MSN_STN_onlyGPi_connectivity_resultsfolder,
                      MSN_STN_bothGPs_connectivity_resultsfolder, MSN_STN_noGPs_connectivity_resultsfolder]
    msn_stn_summed_synapses = compare_connectivity_multiple(comp_cts=msn_cts, filename=f_name,
                                                           foldernames=result_folders,
                                                           connected_ct=0, min_comp_len=cl_cell, label_cts=labels_cts,
                                                           colours=msn_colors)
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups connectivity to STN finished']

    log.info("Step 10/12: Compare connectivity of MSN groups to HVC axons")
    HVC_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == 4]
    MSN_HVC_onlyGPe_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=4, filename=f_name,
                                                            cellids1=HVC_ids, celltype2=2, cellids2=msn2gpe_ids,
                                                            full_cells_ct2=True,syn_prob_thresh=syn_prob,
                                                            label_ct1=None, label_ct2="MSN only GPe",
                                                                version = version,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax)
    MSN_HVC_onlyGPi_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=4, filename=f_name,
                                                                cellids1=HVC_ids, celltype2=2, cellids2=msn2gpi_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN only GPi",
                                                                version = version)
    MSN_HVC_bothGPs_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=4, filename=f_name,
                                                                cellids1=HVC_ids, celltype2=2, cellids2=msn2gpei_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN both GPs",
                                                                version = version)
    MSN_HVC_noGPs_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=4, filename=f_name,
                                                                cellids1=HVC_ids, celltype2=2, cellids2=msn_nogp_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN no GPs",
                                                                version = version)

    # connectivity comparison for multiple groups, comparison to axons
    result_folders = [MSN_HVC_onlyGPe_connectivity_resultsfolder, MSN_HVC_onlyGPi_connectivity_resultsfolder,
                      MSN_HVC_bothGPs_connectivity_resultsfolder, MSN_HVC_noGPs_connectivity_resultsfolder]
    msn_hvc_summed_synapses = compare_connectivity_multiple(comp_cts=msn_cts, filename=f_name,
                                                           foldernames=result_folders,
                                                           connected_ct=4, min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, label_cts=labels_cts,
                                                           colours=msn_colors)
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups connectivity to HVC finished']

    log.info("Step 11/12: Compare connectivity of MSN groups to LMAN")
    LMAN_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == 3]
    MSN_LMAN_onlyGPe_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, filename=f_name,
                                                                cellids1=LMAN_ids, celltype2=2, cellids2=msn2gpe_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN only GPe",
                                                                 version = version)
    MSN_LMAN_onlyGPi_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, filename=f_name,
                                                                cellids1=LMAN_ids, celltype2=2, cellids2=msn2gpi_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN only GPi",
                                                                 version = version)
    MSN_LMAN_bothGPs_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, filename=f_name,
                                                                cellids1=LMAN_ids, celltype2=2, cellids2=msn2gpei_ids,
                                                                full_cells_ct2=True,
                                                                min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                                label_ct1=None, label_ct2="MSN both GPs",
                                                                 version = version)
    MSN_LMAN_noGPs_connectivity_resultsfolder = synapses_ax2ct(sd_synssv, celltype1=3, filename=f_name,
                                                              cellids1=LMAN_ids, celltype2=2, cellids2=msn_nogp_ids,
                                                              full_cells_ct2=True,
                                                              min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, syn_prob_thresh=syn_prob,
                                                              label_ct1=None, label_ct2="MSN no GPs",
                                                               version = version)
    # connectivity comparison for multiple groups, comparison to axons
    result_folders = [MSN_LMAN_onlyGPe_connectivity_resultsfolder, MSN_LMAN_onlyGPi_connectivity_resultsfolder,
                      MSN_LMAN_bothGPs_connectivity_resultsfolder, MSN_LMAN_noGPs_connectivity_resultsfolder]
    msn_lman_summed_synapses = compare_connectivity_multiple(comp_cts=msn_cts, filename=f_name,
                                                           foldernames=result_folders,
                                                           connected_ct=3, min_comp_len_cells = cl_cell, min_comp_len_ax = cl_ax, label_cts=labels_cts,
                                                           colours=msn_colors)
    time_stamps = [time.time()]
    step_idents = ['compare MSN groups connectivity to LMAN finished']

    log.info("Step 12/12: Make nc overview graph for connectivity")
    # nx graph
    sum_synapse_dict = {**msn_onlyGPe_msn_summed_synapse, **msn_onlyGPi_msn_summed_synapse, **msn_both_msn_summed_synapse, **msn_gpe_summed_synapses, **msn_gpi_summed_synapses, **msn_stn_summed_synapses,
                        **msn_hvc_summed_synapses, **msn_tan_summed_synapses, **msn_fs_summed_synapses, **msn_lman_summed_synapses}
    write_obj2pkl("%s/ct_sum_synapses.pkl" % f_name, sum_synapse_dict)
    # plot
    sum_synapse_dict = load_pkl2obj("%s/ct_sum_synapses.pkl" % f_name)
    plot_nx_graph(sum_synapse_dict, filename=("%s/summed_synapses_nx_overview_mcl%i_ax%i.png" % (f_name, cl_cell, cl_ax)),
                  title="sum of synapses between celltypes")

    msn_summed_synapse_pd = pd.DataFrame(sum_synapse_dict, index=[0])
    msn_summed_synapse_pd.to_csv("%s/ct_summed_synapses.csv" % f_name)
    time_stamps = [time.time()]
    step_idents = ['NX Graph to visualise connectivity done']
    step_idents = ['MSN analysis based on groups based on GPe/i connectivity done']