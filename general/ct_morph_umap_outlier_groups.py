#Small groups of MSNs that lie apart from the main cloud in the UMAPs of ct_morph_analyses.py (Fig S6A, C): their
#GPe/GPi connectivity composition, and the overlap with the groups that were manually reviewed for mergers

if __name__ == '__main__':
    from bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    import numpy as np

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    syn_prob = 0.6
    fontsize = 20
    #UMAP runs of ct_morph_analyses.py (18-dimensional feature vector incl. synapse, mitochondria and vesicle cloud densities),
    #first embedding = first plot of the figure
    umap_f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/250711_j0251{version}_ct_morph_analyses_newmergers_mcl_%i_ax%i_TeBKv6MSNyw_fs%inpca1_umap5_a0.5_synmivc" % (
        min_comp_len_cell, min_comp_len_ax, fontsize)
    umap_run = 0
    #GPe / GPi connectivity groups of the MSNs from msn_spine_density_gp_ratio.py
    gp_f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241025_j0251{version}_MSN_3_GPe_GPiratio_spine_density_mcl_%i_synprob_%.2f_kde1_f%i" % (
        min_comp_len_cell, syn_prob, fontsize)
    #groups apart from the main cloud: [UMAP, name, UMAP 1 range, UMAP 2 range]
    #(first embedding of the seeded runs, random_state = 42)
    outlier_groups = [['MSN only', 'MSN-only UMAP 1 > 14', (14, np.inf), (-np.inf, np.inf)],
                      ['all cell types', 'all-types UMAP 1 < 3', (-np.inf, 3), (-np.inf, np.inf)],
                      ['all cell types', 'all-types UMAP 1 7.8-8.5, UMAP 2 6.5-7.5', (7.8, 8.5), (6.5, 7.5)]]
    #manually reviewed groups (curation folder)
    reviewed_files = {'MSN-only group reviewed for mergers': 'umap_outlier_group_msn_only_reviewed.csv',
                      'all-types group reviewed for mergers': 'umap_outlier_group_all_types_reviewed.csv'}
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/261003_j0251{version}_ct_morph_umap_outlier_groups_mcl_%i_ax%i_umap%i" % (
        min_comp_len_cell, min_comp_len_ax, umap_run)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('ct_morph_umap_outlier_groups_log', log_dir=f_name + '/logs/')
    log.info(f'UMAP embeddings {umap_run} of {umap_f_name} (all cell types) and {umap_f_name}_MSNonly (MSN only)')
    log.info(f'GPe / GPi connectivity groups from {gp_f_name}')

    log.info('Step 1/3: Load UMAP embeddings, connectivity groups and reviewed groups')
    umap_dfs = {'MSN only': pd.read_csv(f'{umap_f_name}_MSNonly/fc_umap_embeddings_{umap_run}.csv', index_col=0).set_index('cellid'),
                'all cell types': pd.read_csv(f'{umap_f_name}/fc_umap_embeddings_{umap_run}.csv', index_col=0).set_index('cellid')}
    gp_df = pd.read_csv(f'{gp_f_name}/MSN_spine_density_GPe_GPiratio.csv', index_col=0).set_index('cellid')
    gp_groups = gp_df['celltype']
    gp_group_names = ['MSN both', 'MSN only GPi', 'MSN only GPe', 'MSN none']
    reviewed = {name: pd.read_csv(f'{bio_params.file_locations}/{file}', index_col=0)['cellid'].values for name, file in reviewed_files.items()}
    for name in reviewed:
        log.info(f'{name}: {len(reviewed[name])} cells')
    all_msn_composition = gp_groups.value_counts(normalize=True).reindex(gp_group_names).fillna(0) * 100
    log.info('GPe / GPi connectivity of all MSNs: ' + ', '.join([f'{name} {all_msn_composition[name]:.1f} %' for name in gp_group_names]))

    log.info('Step 2/3: Composition of the groups apart from the main cloud')
    columns = ['UMAP', 'number of cells'] + [f'{name} [%]' for name in gp_group_names] + [f'cells in {name}' for name in reviewed]
    overview = pd.DataFrame(columns=columns, index=[group[1] for group in outlier_groups])
    group_ids = {}
    for umap_name, group_name, umap1_range, umap2_range in outlier_groups:
        umap_df = umap_dfs[umap_name]
        inds = (umap_df['celltype'] == 'MSN') & (umap_df['UMAP 1'] > umap1_range[0]) & (umap_df['UMAP 1'] < umap1_range[1]) & \
               (umap_df['UMAP 2'] > umap2_range[0]) & (umap_df['UMAP 2'] < umap2_range[1])
        cellids = umap_df.index[inds].values
        group_ids[group_name] = cellids
        overview.loc[group_name, 'UMAP'] = umap_name
        overview.loc[group_name, 'number of cells'] = len(cellids)
        composition = gp_groups.reindex(cellids).value_counts(normalize=True).reindex(gp_group_names).fillna(0) * 100
        for name in gp_group_names:
            overview.loc[group_name, f'{name} [%]'] = composition[name]
        for name in reviewed:
            overview.loc[group_name, f'cells in {name}'] = len(np.intersect1d(cellids, reviewed[name]))
        log.info(f'{group_name}: {len(cellids)} MSNs, ' + ', '.join([f'{name} {composition[name]:.1f} %' for name in gp_group_names]) +
                 '; ' + ', '.join([f'{len(np.intersect1d(cellids, reviewed[name]))} in {name}' for name in reviewed]))
        pd.DataFrame({'cellid': cellids, 'GP connectivity': gp_groups.reindex(cellids).values,
                      'UMAP 1': umap_df.loc[cellids, 'UMAP 1'].values, 'UMAP 2': umap_df.loc[cellids, 'UMAP 2'].values}).to_csv(
            f'{f_name}/{group_name.replace(" ", "_").replace(",", "").replace(">", "gt").replace("<", "lt")}_cellids.csv')
    overview.to_csv(f'{f_name}/outlier_groups_overview.csv')

    log.info('Step 3/3: Overlap between the groups')
    msn_only_group = group_ids[outlier_groups[0][1]]
    for umap_name, group_name, umap1_range, umap2_range in outlier_groups[1:]:
        overlap = len(np.intersect1d(group_ids[group_name], msn_only_group))
        log.info(f'{overlap} of the {len(group_ids[group_name])} cells of "{group_name}" are also in "{outlier_groups[0][1]}"')
    for name in reviewed:
        in_all_types = len(np.intersect1d(reviewed[name], group_ids[outlier_groups[1][1]]))
        log.info(f'{in_all_types} of the {len(reviewed[name])} cells of "{name}" lie in "{outlier_groups[1][1]}"')
    log.info('Analysis finished')
