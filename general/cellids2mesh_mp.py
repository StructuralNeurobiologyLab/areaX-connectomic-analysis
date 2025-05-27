
if __name__ == '__main__':
    from syconn import global_params
    from analysis_params import Analysis_Params
    from cellid2mesh_helper import whole_cellid2mesh, comp2mesh
    from analysis_morph_helper import check_comp_lengths_ct
    import numpy as np
    import os as os
    from syconn.handler.config import initialize_logging
    from syconn.mp.mp_utils import start_multiprocess_imap
    import pandas as pd

    version = 'v6'
    bio_params = Analysis_Params(version = version)
    ct_dict = bio_params.ct_dict(with_glia = True)
    global_params.wd = bio_params.working_dir()
    axon_cts = bio_params.axon_cts()
    ct = 0
    ct_str = ct_dict[ct]
    min_comp_len = 200
    load_handpicked = None
    ply_only = True
    cellids = [1080627023, 126798179, 24397945]

    f_name = 'cajal/scratch/users/arother/exported_meshes/'
    #f_name = f'cajal/scratch/users/arother/bio_analysis_results/for_eval/2450122_{ct_str}_mcl{min_comp_len}/'

    #if None is selected the full cells will be exported; otherwise expects list of compartments
    #possible compartments: 'axon', 'dendrite', 'soma'
    #if all three are given, the full cell mesh will be exported but seperated into compartments
    compartments = ['axon', 'dendrite', 'soma']
    with_skel = False
    cellid_path = None
    if cellids is None:
        if load_handpicked is None and ct == 12:
            cellid_path = f'{bio_params.file_locations}/random_astro_ids_50.csv'
        if compartments is None:
            foldername = f'{f_name}/250122_j0251{version}_{ct_str}_full_cells_mcl_{min_comp_len}/'
        elif len(compartments) == 3:
            foldername = f'{f_name}/250122_j0251{version}_{ct_str}_all_comps_sep_mcl_{min_comp_len}/'
        else:
            foldername = f'{f_name}/250122_j0251{version}_{ct_str}_{compartments[0]}_mcl_{min_comp_len}/'
            if not os.path.exists(foldername):
                os.mkdir(foldername)
    else:
        foldername = f_name
    log = initialize_logging(f'log_cellid2mesh_{ct_str}', log_dir=foldername)
    log.info(f' desired compartment = {compartments}, min comp len = {min_comp_len} µm')
    if with_skel:
        log.info('Skeletons will also be exported.')
    if cellids is not None:
        log.info(f'Exported cellids are {cellids}')

    log.info('Step 1/2: Get suitable cell ids')
    if cellids is None:
        if load_handpicked is not None:
            cellids = bio_params.load_handpicked_ids(load_handpicked)
        elif cellid_path is not None:
            log.info(f'cellids will be loaded from {cellid_path}')
            cellid_df = pd.read_csv(cellid_path)
            cellids = np.array(cellid_df['cellid'])
        else:
            cell_dict = bio_params.load_cell_dict(ct)
            # get ids with min compartment length
            cellids = np.array(list(cell_dict.keys()))
            known_mergers = bio_params.load_known_mergers()
            misclassified_asto_ids = bio_params.load_potential_astros()
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
            axon_cts = bio_params.axon_cts()
            if ct not in axon_cts:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
            else:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=True,
                                                max_path_len=None)
        log.info(f'{len(cellids)} {ct_str} cellids will be exported as meshes.')




    log.info('Step 2/2: export meshes')
    if compartments is None:
        mesh_input = [[cellid, foldername, version, with_skel, ply_only] for cellid in cellids]
        _ = start_multiprocess_imap(whole_cellid2mesh, mesh_input)
    else:
        mesh_input = [[cellid, foldername, version, compartments, ply_only] for cellid in cellids]
        _ = start_multiprocess_imap(comp2mesh, mesh_input)

    log.info('meshes are exported.')



