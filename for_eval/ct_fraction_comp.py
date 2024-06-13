#get fraction of compartment per celltype full cells vs all cells

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_compartment_length_mp
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    min_comp_len_cell = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    color_key = 'TePkBrNGF'
    fontsize = 20
    comp_dict = {0: 'dendrite', 1: 'axon', 2: 'soma'}
    comp = 1
    comp_str = comp_dict[comp]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240613_j0251{version}_fraction_{comp_str}_fullvsfragment_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('fraction_comp_pathlength', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, colors = %s" % (
            min_comp_len_cell, color_key))
    log.info(f'Get fraction of pathlength of {comp_str} full cells vs fragments')
    ct_types = analysis_params.load_celltypes_full_cells()
    ct_str_list = [ct_dict[ct] for ct in ct_types]
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    log.info(f'Get from these celltypes: {ct_str_list}')
    log.info('Known mergers are excluded')

    log.info('Step 1/3: Get cellids from all celltypes')
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    dataset_cellids = ssd.ssv_ids
    dataset_celltypes = ssd.load_numpy_data('celltype_pts_e3')
    full_ids_dict = {}
    all_full_ids = []
    key_cellids_dict = {}
    all_key_cellids = []
    all_celltypes = []
    all_cell_ids = []
    all_cell_dict = {}
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    for ct in ct_types:
        ct_str = ct_dict[ct]
        #get all cellids
        ct_cellids = dataset_cellids[dataset_celltypes == ct]
        merger_inds = np.in1d(ct_cellids, known_mergers) == False
        ct_cellids = ct_cellids[merger_inds]
        astro_inds = np.in1d(ct_cellids, misclassified_asto_ids) == False
        ct_cellids = ct_cellids[astro_inds]
        ct_cellids = np.sort(ct_cellids)
        all_cell_ids.append(ct_cellids)
        all_celltypes.append([ct_dict[ct] for i in ct_cellids])
        log.info(f'In total there are {len(ct_cellids)} cellids from celltype {ct_str}')
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cell_dict_ids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cell_dict_ids, known_mergers) == False
        cell_dict_ids = cell_dict_ids[merger_inds]
        astro_inds = np.in1d(cell_dict_ids, misclassified_asto_ids) == False
        cell_dict_ids = cell_dict_ids[astro_inds]
        cell_dict_ids = np.sort(cell_dict_ids)
        key_cellids_dict[ct] = cell_dict_ids
        all_key_cellids.append(cell_dict_ids)
        full_cellids = check_comp_lengths_ct(cellids=cell_dict_ids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        full_ids_dict[ct] = full_cellids
        all_full_ids.append(full_cellids)
        log.info(f'{len(full_cellids)} fulfill critera for cell type {ct_str}')

    all_full_ids = np.concatenate(all_full_ids)
    all_celltypes = np.concatenate(all_celltypes)
    all_cell_ids = np.concatenate(all_cell_ids)
    all_key_cellids = np.concatenate(all_key_cellids)

    columns = ['cellid', 'celltype', 'full cell', f'{comp_str} pathlength']
    per_cell_res_df = pd.DataFrame(columns=columns, index = range(len(all_cell_ids)))
    per_cell_res_df['cellid'] = all_cell_ids
    per_cell_res_df['celltype'] = all_celltypes
    full_inds = np.in1d(all_cell_ids, all_full_ids)
    frag_inds = full_inds == False
    per_cell_res_df.loc[full_inds, 'full cell'] = 1
    per_cell_res_df.loc[frag_inds, 'full cell'] = 0

    log.info(f'Step 2/3: Get {comp_str} pathlength')
    #get first for all cells in celldicts
    log.info('Get parameters from all cells present in cell_dicts')
    for ct in tqdm(ct_types):
        cell_dict = all_cell_dict[ct]
        cell_dict_ids = key_cellids_dict[ct]
        comp_pathlength = np.zeros(len(cell_dict_ids))
        comp_surface_area = np.zeros(len(cell_dict_ids))
        for i, cell_id in enumerate(cell_dict_ids):
            comp_pathlength[i] = cell_dict[cell_id][f'{comp_str} length']
            #comp_surface_area[i] = cell_dict[cell_id][f'{comp_str} mesh surface area']
        ct_inds = np.in1d(per_cell_res_df['cellid'], cell_dict_ids)
        per_cell_res_df.loc[ct_inds, f'{comp_str} pathlength'] = comp_pathlength
        #per_cell_res_df.loc[ct_inds, f'{comp_str} surface area'] = comp_surface_area
    log.info('Now get pathlength from all other cells')
    non_key_cellids = all_cell_ids[np.in1d(all_cell_ids, all_key_cellids) == False]
    pathlength_input = [[cellid, comp, None, None] for cellid in non_key_cellids]
    pathlength_output = start_multiprocess_imap(get_compartment_length_mp, pathlength_input)
    non_key_inds = np.in1d(per_cell_res_df['cellid'], non_key_cellids)
    per_cell_res_df.loc[non_key_inds, f'{comp_str} pathlength'] = np.array(pathlength_output)
    #remove ids with 0 pathlength
    num_before = len(per_cell_res_df)
    per_cell_res_df = per_cell_res_df[per_cell_res_df[f'{comp_str} pathlength'] > 0]
    per_cell_res_df = per_cell_res_df.reset_index()
    num_after = len(per_cell_res_df)
    log.info(f'{num_before - num_after} fragments were removed because they did not have {comp_str} pathlength')
    per_cell_res_df.to_csv(f'{f_name}/per_cell_{comp_str}_pathlengths.csv')

    log.info('Step 3/3: Get fraction of full cells in pathlength')
    #get fraction of numbers and of pathlength and write in table
    unique_cts = np.unique(per_cell_res_df['celltype'])
    fraction_df = pd.DataFrame(columns = ['celltype', 'number fragments','number full cells','full cells fraction', 'full cells pathlength fraction'], index = range(len(unique_cts)))
    fraction_df['celltype'] = unique_cts
    ct_groups = per_cell_res_df.groupby('celltype')
    full_res_df = per_cell_res_df[per_cell_res_df['full cell'] == 1]
    full_ct_groups = full_res_df.groupby('celltype')
    all_cell_numbers = np.array(ct_groups.size())
    fraction_df['number fragments'] = all_cell_numbers
    full_cell_numbers = np.array(full_ct_groups.size())
    fraction_df['number full cells'] = full_cell_numbers
    fraction_df['full cells fraction'] = full_cell_numbers / all_cell_numbers
    all_cell_pathlengths = np.array(ct_groups[f'{comp_str} pathlength'].sum())
    full_cell_pathlengths = np.array(full_ct_groups[f'{comp_str} pathlength'].sum())
    fraction_df['full cells pathlength fraction'] = full_cell_pathlengths / all_cell_pathlengths
    fraction_df.to_csv(f'{f_name}/{comp_str}_fraction_pathlength.csv')
    #plot as barplot
    for param in fraction_df:
        if not 'fraction' in param:
            continue
        sns.barplot(data = fraction_df, x = 'celltype', y = param, palette=ct_palette, order = ct_str_list)
        plt.ylabel(param, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'{param}')
        plt.savefig(f'{f_name}/{param}_bar.svg')
        plt.savefig(f'{f_name}/{param}_bar.png')
        plt.close()

    log.info('Analyses done')