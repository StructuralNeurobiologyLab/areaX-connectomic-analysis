#for each cell type and also handpicked glial cells
#get celltype, certainty, volume and total pathlength in µm

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import check_comp_lengths_ct, get_cell_length, get_cell_volume_certainty_mp
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_params import Analysis_Params
    import os as os
    from syconn.reps.super_segmentation import SuperSegmentationObject
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal
    from itertools import combinations
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = True
    handpicked_glia = True
    with_OPC = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    min_comp_len_cell = 200
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/241213_j0251{version}_ct_morph_params_mcl{min_comp_len_cell}_{fontsize}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('ct_morph_params', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells" % (
            min_comp_len_cell))
    log.info('Use only cell types residing in area X')
    known_mergers = analysis_params.load_known_mergers()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    if with_glia:
        log.info('Also process glial cell types and migratory neurons')
        glia_cts = analysis_params.glia_cts()
        #remove fragment celltype
        num_cts = num_cts - 1
    np_presaved_loc = analysis_params.file_locations
    ct_types = np.arange(0, num_cts)
    ct_types = ct_types[np.in1d(ct_types, axon_cts) == False]
    ct_str_list = [ct_dict[ct] for ct in ct_types]
    if handpicked_glia:
        log.info('Manually selected glia cells will be used for glia cells in analysis')
        if with_OPC:
            log.info('manually selected OPC are included in analysis')
            ct_dict[17] = 'OPC'
            glia_cts = np.hstack([glia_cts, 17])


    log.info('Step 1/4: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_suitable_cts = []
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if handpicked_glia and ct in glia_cts:
            cellids = analysis_params.load_handpicked_ids(ct, ct_dict=ct_dict)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            all_cell_dict[ct] = cell_dict
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct in axon_cts:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=True, max_path_len=None)
            else:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_suitable_cts.append([[ct_str] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_suitable_cts = np.concatenate(all_suitable_cts)

    log.info('Get celltype certainty per cell')
    percell_param_df = pd.DataFrame(columns = ['cellid', 'celltype', 'celltype certainty', 'cell volume [µm³]', 'pathlength [µm]'], index = range(len(all_suitable_ids)))
    percell_param_df['cellid'] = all_suitable_ids.astype(int)
    percell_param_df['celltype'] = all_suitable_cts
    ct_certainty_key = analysis_params.celltype_certainty_key()

    log.info('Get cell volume')
    cellid_chunks = np.array_split(all_suitable_ids, np.ceil(len(all_suitable_ids) / 100))
    volume_cert_input = [[cellid_chunk, ct_certainty_key] for cellid_chunk in cellid_chunks]
    volume_cert_output = start_multiprocess_imap(get_cell_volume_certainty_mp, volume_cert_input)
    volume_cert_output = np.array(volume_cert_output, dtype='object')
    cell_volumes = np.concatenate(volume_cert_output[:, 0])
    cell_certainties = np.concatenate(volume_cert_output[:, 1])
    percell_param_df['cell volume [µm³]'] = cell_volumes
    percell_param_df['celltype certainty'] = cell_certainties

    log.info('Get cell pathlength')
    for ct in ct_types:
        ct_str = ct_dict[ct]
        cellids_series = percell_param_df['cellid'][percell_param_df['celltype'] == ct_str]
        cellids_ct = np.array(cellids_series)
        inds_ct = cellids_series.index
        if ct in glia_cts:
            complete_pathlength = start_multiprocess_imap(get_cell_length, cellids_ct)
            percell_param_df.loc[inds_ct, 'pathlength [µm]'] = np.array(complete_pathlength)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            for ind, cellid in tqdm(zip(inds_ct, cellids_ct)):
                total_pathlength_cell = cell_dict[cellid]['complete pathlength']
                percell_param_df.loc[ind, 'pathlength [µm]'] = total_pathlength_cell

    percell_param_df.to_csv(f'{f_name}/percell_param_df.csv')

    log.info('Analysis finished')

