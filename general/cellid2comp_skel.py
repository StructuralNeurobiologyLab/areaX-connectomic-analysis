#get compartment coloured skeleton for cellids

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import generate_colored_mesh_from_skel_data
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    #from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = True
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    key = 'spiness'
    only_coarse = True
    k = 1
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/251013_j0251{version}_cellid2comp_skel_spiness/"

    log = initialize_logging(f'cellid2comp_skel_mp', log_dir=f_name)

    log.info(f'key = {key}, only coarse = {only_coarse}, k = {k}')

    cellid_filename = "cajal/scratch/users/arother/bio_analysis_results/general/240723_j0251v6_all_cellids_for_exclusion/250912_rndm_full_msn_stn_ids.csv"

    cellid_df = pd.read_csv(cellid_filename)

    cellids = np.array(cellid_df['cellid'])
    #cellids = [1156947761]

    log.info(f'Get skeletons and meshes coloured by compartment from {len(cellids)} cells')

    skel_input = [[cellid, f_name, key, only_coarse, k, ct_dict] for cellid in cellids]

    _ = start_multiprocess_imap(generate_colored_mesh_from_skel_data, skel_input)

    log.info('Skeletons exported.')