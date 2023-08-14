#script to plot spine density for all msns
#plot spine density with msn subgroups overlayed
#plot spine density as jointplot vs GPe/ GPi ratio in synapse number and synapse sum size
#GPe/GPi ratios = (GPe + GPi)/GPi

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple, compare_soma_diameters
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_spine_density
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
    from syconn.mp.mp_utils import start_multiprocess_imap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 2
    gpe_ct = 6
    gpi_ct = 7
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230814_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_ranksums_med" % (
    min_comp_len, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN spine density vs GP ratio', log_dir=f_name + '/logs/')
    log.info("Analysis of spine density vs GP ratio starts")

    log.info('Step 1/X: Load and check all MSN cells')
    known_mergers = analysis_params.load_known_mergers()
    MSN_dict = analysis_params.load_cell_dict(celltype=msn_ct)
    MSN_ids = np.array(list(MSN_dict.keys()))
    merger_inds = np.in1d(MSN_ids, known_mergers) == False
    MSN_ids = MSN_ids[merger_inds]
    misclassified_asto_ids = analysis_params.load_potential_astros()
    astro_inds = np.in1d(MSN_ids, misclassified_asto_ids) == False
    MSN_ids = MSN_ids[astro_inds]
    MSN_ids = check_comp_lengths_ct(cellids=MSN_ids, fullcelldict=MSN_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    columns = ['cellid', 'spine density', 'celltype', 'GP ratio syn number', 'GP ratio sum syn size']
    msn_results_df = pd.DataFrame(columns=columns, index=range(len(MSN_ids)))
    msn_results_df['cellid'] = MSN_ids

    log.info('Step 2/X: Get spine density of all MSN cells')
    ngf_input = [[ngf_id, min_comp_len, MSN_dict] for ngf_id in MSN_ids]
    spine_density = start_multiprocess_imap(get_spine_density, ngf_input)
    spine_density = np.array(spine_density)
    msn_results_df['spine density'] = spine_density
    msn_results_df.to_csv(f'{f_name}/msn_spine_density_results.csv')
    #plot spine density as histogram

    log.info('Step 3/X: Get GP ratio for MSNs')
    log.info('Get suitable cellids from GPe and GPi')
    GPe_dict = analysis_params.load_cell_dict(gpe_ct)
    GPe_ids = np.array(list(GPe_dict.keys()))
    merger_inds = np.in1d(GPe_ids, known_mergers) == False
    GPe_ids = GPe_ids[merger_inds]
    GPe_ids = check_comp_lengths_ct(cellids=GPe_ids, fullcelldict=GPe_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    GPi_dict = analysis_params.load_cell_dict(gpi_ct)
    GPi_ids = np.array(list(GPi_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(GPe_ids)} suitable for analysis and {len(GPi_ids)} suitable for analysis')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    #generally prefilter synapses for min_comp_len, syn_prob and full cell_ids
    #prefilter connections from msn to gpe or gpi
    #get GP ratio per msn cell and put in dataframe
    #plot GP ratio as histogram
    #plot spine density vs GP ratio; once gray, once MSN colour

    #overlay with msn subgroups and plot again

