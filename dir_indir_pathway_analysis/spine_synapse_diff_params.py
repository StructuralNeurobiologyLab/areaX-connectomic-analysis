if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from spine_synapse_analysis import spine_synapse_analysis_GP_MSN
    from syconn.handler.basics import load_pkl2obj
    import os as os
    import numpy as np

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)

    msn_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_MSN_arr.pkl")
    # gpe_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_GPe_arr.pkl")
    # gpi_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_GPi_arr.pkl")
    gpe_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPe_arr.pkl")
    gpi_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPi_arr.pkl")

    foldername = "u/arother/test_folder/210604_j0251v3_spine_syn_analysis_GPhpminax_avgsinsize"
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    min_syn_size_array = np.array([0.05, 0.1, 0.5, 1])
    syn_prob_array = np.array([0.7, 0.8, 0.9])
    ct_certainty_array = np.array([0.7, 0.8, 0.9])
    den_len_array = np.array([0, 200])
    percentile_array = np.array([25, 50])
    larger_percentile_array = np.array([15, 20, 25])
    min_ax_array = np.array([0, 200, 500, 1000, 2000])

    #for min_sin_size in min_syn_size_array:
        #spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, min_synsize = min_sin_size)

    #for min_ax in min_ax_array:
        #spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, min_ax_length=min_ax)

    for syn_prob in syn_prob_array:
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, syn_prob_thresh=syn_prob)

    for ct_certainty in ct_certainty_array:
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, ct_certainty=ct_certainty)

    for den_len in den_len_array:
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, min_den_length=den_len)

    for perc in percentile_array:
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, percentile_param=perc)

    for perc in larger_percentile_array:
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, percentile_param=perc, syn_prob_thresh = 0.9)
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, percentile_param=perc,
                                      ct_certainty =0.9)
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, percentile_param=perc,
                                      min_synsize=0.5)
        spine_synapse_analysis_GP_MSN(ssd, sd_synssv, msn_ids, gpe_ids, gpi_ids, foldername, percentile_param=perc,
                                      min_synsize=1)