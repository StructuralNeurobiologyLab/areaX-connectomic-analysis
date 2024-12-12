#get random axon ids of different length to manually check for mergers afterwards
if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from sklearn.utils import shuffle
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct

    version = 'v6'
    analysis_params = Analysis_Params('v6')
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia= True)
    min_comp_len = 200
    #samples per ct
    rnd_samples = 3
    handpicked_glia_only = True
    with_OPC = True
    full_cells_only = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240903_j0251{version}_rndm_cellids_golgi_eval_mcl_%i_samples_%i" % (
        min_comp_len, rnd_samples)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('rndm_cellids',
                             log_dir=f_name)
    log.info(
        f"min_comp_len = %i, number of samples per ct = %i" % (
            min_comp_len, rnd_samples))