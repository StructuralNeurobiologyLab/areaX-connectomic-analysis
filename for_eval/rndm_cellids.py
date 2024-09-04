#get random cellids from all celltypes

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
    ct_types = np.arange(0, len(ct_dict.keys()))
    #remove fragment class
    ct_types = ct_types[ct_types != 16]
    ax_cts = analysis_params.axon_cts()
    if full_cells_only:
        log.info('Only cell types with soma in the dataset will be processed.')
        ct_types = ct_types[np.in1d(ct_types, ax_cts) == False]
    if handpicked_glia_only:
        log.info('Only manually selected glia cells will be used')
    glia_types = analysis_params._glia_cts
    if with_OPC:
        log.info('Manually selected OPCs will also be used as their own celltype')
        ct_dict[17] = 'OPC'
        glia_types = np.hstack([glia_types, 17])
        ct_types = np.hstack([ct_types, 17])

    log.info(f'Iterate over celltypes to write out {rnd_samples} cells with compartments')
    num_cts = len(ct_types)
    cts_str = [ct_dict[ct] for ct in ct_types]
    known_mergers = analysis_params.load_known_mergers()
    pot_astros = analysis_params.load_potential_astros()
    np.random.seed(42)

    all_celltypes = []
    all_rndm_ids = []

    for ct in tqdm(ct_types):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]

        if ct in ax_cts:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=True, max_path_len=None)
        else:
            if ct in glia_types:
                if handpicked_glia_only:
                        cellids = analysis_params.load_handpicked_ids(ct, ct_dict=ct_dict)
            else:
                cell_dict = analysis_params.load_cell_dict(ct)
                cellids = analysis_params.load_full_cell_array(ct)
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
                if ct == 3:
                    misclassified_asto_ids = analysis_params.load_potential_astros()
                    astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                    cellids = cellids[astro_inds]
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        #select cells randomly
        rnd_cellids = np.random.choice(a=cellids, size=rnd_samples, replace=False)
        all_rndm_ids.append(rnd_cellids)
        all_celltypes.append([ct_dict[ct] for i in rnd_cellids])

    all_rndm_ids = np.concatenate(all_rndm_ids)
    all_celltypes = np.concatenate(all_celltypes)
    selected_ids = pd.DataFrame(columns=['cellid', 'celltype'], index=range(len(all_rndm_ids)))
    selected_ids['cellid'] = all_rndm_ids
    selected_ids['celltype'] = all_celltypes
    selected_ids = shuffle(selected_ids)
    selected_ids.to_csv(f'{f_name}/rnd_cellids.csv')

    log.info('Analysis finished.')