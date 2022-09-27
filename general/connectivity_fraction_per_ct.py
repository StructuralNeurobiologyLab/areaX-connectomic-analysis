#write script that plots for each celltype the ratio in amount and sum of synapse size from
#full cells vs fragments
# fractions of amounts, sum size from different cell classes

#output for each cell class are five different cake plots (4 for axons):
# 1) input from fragments vs full cells/ long axons
# 2) input: number of synapses from each celltype class
# 3) input: sum size of synapses from each celltype class
# 4) output: number of synapses from each celltype class
#5) output: sum size of synapses from each celltype class
#save this in dictionary, table as well (can plot table as matrix then; one table for input (fragments, full cells), output)

if __name__ == '__main__':
    from cajal.nvmescatch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescatch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from cajal.nvmescatch.users.arother.bio_analysis.general.result_helper import ResultsForPlotting
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import scipy


    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    max_MSN_path_len = 7500
    syn_prob = 0.8
    min_syn_size = 0.1
    msn_ct = 2
    lman_ct = 3
    gpi_ct = 7
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/general/220927_j0251v4_cts_percentages_mcl_%i_synprob_%.2f" % (
    min_comp_len, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Celltypes input output percentages', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, max_MSN_path_len = %i, syn_prob = %.1f, min_syn_size = %.1f" % (min_comp_len, max_MSN_path_len, syn_prob, min_syn_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #this script should iterate over all celltypes
    axon_cts = [1, 3, 4]
    log.info("Step 1/X: Load cell dicts and get suitable cellids")
    celltypes = list(ct_dict.keys())
    full_cell_dicts = {}
    suitable_ids_dict = {}
    cts_numbers_perct = pd.DataFrame(columns=['total number of cells', 'number of suitable cells', 'percentage of suitable cells'], index=celltypes)
    for ct in tqdm(celltypes):
        if ct in axon_cts:
            cell_dict = load_pkl2obj(
            "/cajal/nvmescratch/users/arother/j0251v4_prep/ax_%s_dict.pkl" % ct_dict[ct])
            #get ids with min compartment length
            cellids = list(cell_dict.keys())
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len, axon_only=True,
                              max_path_len=None)
            cts_numbers_perct.loc[ct, 'total number of cells'] = len(cellids)
        else:
            cell_dict = load_pkl2obj(
                "/cajal/nvmescratch/users/arother/j0251v4_prep/full_%s_dict.pkl" % ct_dict[ct])
            cellids = list(cell_dict.keys())
            if ct == 2:
                cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=max_MSN_path_len)
            else:
                cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
        full_cell_dicts[ct] = cell_dict
        suitable_ids_dict[ct] = cellids_checked
        cts_numbers_perct.loc[ct, 'number of suitable cells'] = len(cellids_checked)
        cts_numbers_perct.loc[ct,'percentage of suitable cells'] = cts_numbers_perct.loc[ct, 'total number of cells']/\
                                                                   cts_numbers_perct.loc[ct, 'number of suitable cells']

    write_obj2pkl("%s/suitable_ids_allct.pkl" % f_name, suitable_ids_dict)
    cts_numbers_perct.to_csv("%s/numbers_perct.csv" % f_name)
    time_stamps = [time.time()]
    step_idents = ['Suitable cellids loaded']











