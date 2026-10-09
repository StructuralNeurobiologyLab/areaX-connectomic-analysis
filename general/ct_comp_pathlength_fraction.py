#get fraction of axon and dendrite pathlength per celltype that is contained in full cells (Table S8)
#full cells = cells with soma, axon and dendrite >= min_comp_len (same filtering as connectivity_fraction_per_ct),
#all cells = all ssvs predicted as this celltype, including fragments

if __name__ == '__main__':
    from bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from bio_analysis.general.analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.handler.basics import write_obj2pkl
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap

    version = 'v8'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    celltype_key = analysis_params.celltype_key()
    min_comp_len_cells = 200
    exclude_known_mergers = True
    chunk_size = 500
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/261001_j0251%s_ct_comp_pathlength_fraction_mcl_%i" % (
        version, min_comp_len_cells)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compartment pathlength fraction in full cells', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, known mergers excluded = %s, chunk size = %i" % (
        min_comp_len_cells, exclude_known_mergers, chunk_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    def get_axon_dendrite_length_chunk(cellids):
        """
        axon and dendrite skeleton length in µm for each cellid, from the skeleton node predictions as in
        analysis_morph_helper.get_compartment_length (axoness 3, 4 -> axon); nan if the skeleton can not be loaded
        :param cellids: list of cellids
        :return: array (len(cellids), 2) with dendrite and axon length
        """
        lengths = np.full((len(cellids), 2), np.nan)
        for i, cellid in enumerate(cellids):
            try:
                sso = SuperSegmentationObject(cellid)
                sso.load_skeleton()
                axoness = sso.skeleton["axoness_avg10000"]
                axoness[axoness == 3] = 1
                axoness[axoness == 4] = 1
                cell_graph = sso.weighted_graph()
                for compartment in [0, 1]:
                    non_comp_inds = np.nonzero(axoness != compartment)[0]
                    comp_graph = cell_graph.copy()
                    comp_graph.remove_nodes_from(non_comp_inds)
                    lengths[i, compartment] = comp_graph.size(weight="weight") / 1000  # in µm
            except Exception:
                continue
        return lengths

    #filter cells for min_comp_len
    log.info("Step 1/3: Load cell dicts and get suitable cellids")
    if exclude_known_mergers:
        known_mergers = analysis_params.load_known_mergers()
    full_cts = analysis_params.load_celltypes_full_cells(with_glia=False)
    ct_str_list = [ct_dict[ct] for ct in full_cts]
    ssv_cts = ssd.load_numpy_data(celltype_key)
    suitable_ids_dict = {}
    all_ids_dict = {}
    for ct in full_cts:
        cell_dict = analysis_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cells,
                                                axon_only=False, max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_ids_dict[ct] = ssd.ssv_ids[ssv_cts == ct]
        log.info(f'{ct_dict[ct]}: {len(cellids_checked)} suitable full cells, {len(all_ids_dict[ct])} cells and fragments in total')
    write_obj2pkl("%s/suitable_ids_allct.pkl" % f_name, suitable_ids_dict)
    time_stamps.append(time.time())
    step_idents.append('suitable cellids loaded')

    log.info("Step 2/3: Get axon and dendrite pathlength of all cells and fragments per celltype")
    columns = ['number of cells and fragments', 'number of cells without skeleton', 'number of full cells',
               'axon pathlength all [mm]', 'axon pathlength full cells [mm]', 'axon pathlength fraction',
               'dendrite pathlength all [mm]', 'dendrite pathlength full cells [mm]', 'dendrite pathlength fraction']
    result_df = pd.DataFrame(columns=columns, index=ct_str_list)
    for ct in tqdm(full_cts):
        ct_str = ct_dict[ct]
        all_ids = all_ids_dict[ct]
        chunks = [all_ids[i: i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
        lengths = np.concatenate(start_multiprocess_imap(get_axon_dendrite_length_chunk, chunks))
        no_skel = np.isnan(lengths[:, 0])
        full_inds = np.in1d(all_ids, suitable_ids_dict[ct])
        if full_inds.sum() != len(suitable_ids_dict[ct]):
            log.info(f'{ct_str}: {len(suitable_ids_dict[ct]) - full_inds.sum()} suitable cells are not predicted as {ct_str} in {celltype_key}')
        per_cell_df = pd.DataFrame({'cellid': all_ids, 'dendrite length': lengths[:, 0], 'axon length': lengths[:, 1],
                                    'full cell': full_inds})
        per_cell_df.to_csv(f'{f_name}/{ct_str}_comp_lengths_all_cells.csv')
        result_df.loc[ct_str, 'number of cells and fragments'] = len(all_ids)
        result_df.loc[ct_str, 'number of cells without skeleton'] = int(no_skel.sum())
        result_df.loc[ct_str, 'number of full cells'] = int(full_inds.sum())
        for comp, comp_str in enumerate(['dendrite', 'axon']):
            len_all = np.nansum(lengths[:, comp]) / 1000  # in mm
            len_full = np.nansum(lengths[full_inds, comp]) / 1000
            result_df.loc[ct_str, f'{comp_str} pathlength all [mm]'] = len_all
            result_df.loc[ct_str, f'{comp_str} pathlength full cells [mm]'] = len_full
            result_df.loc[ct_str, f'{comp_str} pathlength fraction'] = len_full / len_all
        log.info(f'{ct_str}: axon fraction {result_df.loc[ct_str, "axon pathlength fraction"]:.2f}, dendrite fraction '
                 f'{result_df.loc[ct_str, "dendrite pathlength fraction"]:.2f}, {int(no_skel.sum())} without skeleton')
        time_stamps.append(time.time())
        step_idents.append(f'{ct_str} lengths calculated')

    log.info("Step 3/3: Save results")
    result_df.to_csv(f'{f_name}/comp_pathlength_fraction_per_ct.csv')
    table_df = result_df[['axon pathlength fraction', 'dendrite pathlength fraction']].astype(float).round(2)
    table_df.to_csv(f'{f_name}/table_S8_comp_pathlength_fraction.csv')
    time_stamps.append(time.time())
    step_idents.append('results saved')
    time_stamps = np.array(time_stamps)
    time_diff = time_stamps[1:] - time_stamps[:-1]
    log.info('Steps and times: %s' % list(zip(step_idents[1:], np.round(time_diff, 1))))
    log.info('Analysis finished')
