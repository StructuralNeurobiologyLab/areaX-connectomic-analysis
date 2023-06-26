#cache numpy arrays of celltypes with specific length
#make additional array with celltypes

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_params import Analysis_Params
    import numpy as np
    import pandas as pd


    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    analysis_params = Analysis_Params(version='v5', working_dir=global_params.wd)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    cache_name = analysis_params.file_locations
    full_cts = analysis_params.load_celltypes_full_cells(with_glia=False)
    full_cts_str = [ct_dict[ct] for ct in full_cts]
    min_comp_len = [200, 100]
    pot_astros = analysis_params.load_potential_astros()
    known_mergers = analysis_params.load_known_mergers()
    log = initialize_logging('cache cellids', log_dir=cache_name + '/logs/')
    log.info(f'Start caching cellids with celltypes for min comp lengths {min_comp_len} µm for axon and dendrite each')
    cell_numbers_caching = pd.DataFrame(columns = [f'{cl} µm' for cl in min_comp_len], index = full_cts_str)

    for cl in min_comp_len:
        log.info(f'Start caching for min comp len = {cl} µm')
        celltypes = []
        cellids_filtered = []
        for ct in full_cts:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = analysis_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=cl,
                                            axon_only=False, max_path_len=None)
            cellids_filtered.append(cellids)
            cts = np.zeros(len(cellids)) + ct
            celltypes.append(cts)
            cell_numbers_caching.loc[ct_dict[ct], f'{cl} µm'] = len(cellids)
        cellids_con = np.concatenate(cellids_filtered)
        celltypes_con = np.concatenate(celltypes)
        np.save(f'{cache_name}/full_cell_ids_{cl}_µm.npy', cellids_con)
        np.save(f'{cache_name}/full_cell_celltypes_{cl}_µm.npy', celltypes_con)
        log.info(f'Caches with min comp len = {cl} saved. Found {len(cellids_con)} cells.')

    cell_numbers_caching.to_csv(f'{cache_name}/cell_numbers_caching.csv')
    log.info('Cellids with suiatble min_comp_len without potential mergers and astrocytes cached. ')