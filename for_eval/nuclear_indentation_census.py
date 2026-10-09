#Manual census of nuclear indentations (Fig 5B, Fig S11E): every soma of the non-MSN neuron types was classified blind on
#raw EM z-stacks as regular, indented or excluded (bad data, wrong object, cut by the border). The labels are in the
#curation folder; here they are joined with the full cells of the analysis and counted per cell type

if __name__ == '__main__':
    from bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    import numpy as np

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    axon_cts = bio_params.axon_cts()
    min_comp_len_cell = 200
    exclude_known_mergers = True
    msn_ct = 3
    int_cts = [9, 10, 11, 12]
    labels_file = 'soma_nuclear_indentation_labels.csv'
    #most complete cells per INT type (INT_connectivity_completeness_umap.py), for the Fig 5B subset
    completeness_f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/261003_j0251{version}_INT_conn_completeness_umap_mcl_%i_synprob_0.60_nn15_md0.3_fs20" % min_comp_len_cell
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/261003_j0251{version}_nuclear_indentation_census_mcl_%i" % min_comp_len_cell
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('nuclear_indentation_census_log', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, known mergers excluded = %s" % (min_comp_len_cell, exclude_known_mergers))
    known_mergers = bio_params.load_known_mergers()

    log.info('Step 1/3: Load labels and full cells')
    labels = pd.read_csv(f'{bio_params.file_locations}/{labels_file}').set_index('cellid')['label']
    log.info(f'{len(labels)} labelled somata: ' + str(labels.value_counts().to_dict()))
    cell_rows = []
    for ct in ct_dict:
        if ct in axon_cts or ct == msn_ct:
            continue
        cell_dict = bio_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                        axon_only=False, max_path_len=None)
        for cellid in cellids:
            cell_rows.append({'cellid': cellid, 'celltype': ct_dict[ct]})
    cell_df = pd.DataFrame(cell_rows).set_index('cellid')
    cell_df['nuclear indentation'] = labels.reindex(cell_df.index).fillna('not labelled')
    cell_df.to_csv(f'{f_name}/nuclear_indentation_percell.csv')

    log.info('Step 2/3: Counts per cell type')
    counts = pd.crosstab(cell_df['celltype'], cell_df['nuclear indentation'])
    counts['fraction indented of labelled'] = counts.get('indentation', 0) / (counts.get('indentation', 0) + counts.get('regular', 0))
    counts.to_csv(f'{f_name}/nuclear_indentation_per_celltype.csv')
    log.info('Nuclear indentation per cell type (full cells):\n' + counts.round(2).to_string())

    log.info('Step 3/3: INT cells shown in Fig 5B (most complete 80 percent)')
    if os.path.exists(f'{completeness_f_name}/INT_completeness_percell.csv'):
        completeness = pd.read_csv(f'{completeness_f_name}/INT_completeness_percell.csv', index_col=0)
        int_df = cell_df.loc[cell_df['celltype'].isin([ct_dict[ct] for ct in int_cts])].join(completeness[['most complete 80 percent']])
        for subset, inds in [('all INT cells', int_df['most complete 80 percent'].notna()), ('most complete 80 percent', int_df['most complete 80 percent'] == True)]:
            sub_counts = pd.crosstab(int_df.loc[inds, 'celltype'], int_df.loc[inds, 'nuclear indentation'])
            log.info(f'{subset}:\n' + sub_counts.to_string())
            sub_counts.to_csv(f'{f_name}/nuclear_indentation_INT_{subset.replace(" ", "_")}.csv')
    else:
        log.info(f'{completeness_f_name} not found, INT subsets skipped')
    log.info('Analysis finished')
