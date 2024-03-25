#get matrix of full cells with calculations for gain estimation

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, get_percell_number_sumsize
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ConnMatrix
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from tqdm import tqdm

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    full_cells_only = False
    min_comp_len_cell = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    fontsize = 20
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_percell_gain_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('per_cell_conn_matrix_log', log_dir=f_name + '/logs/')
    log.info('Get per cell matrix that can be used for estimation')
    #load results from synaptic connectivity matrix, in µm²
    syn_percell_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_percell_conn_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    syn_conn_matrix_percell = pd.read_csv(f'{syn_percell_path}/percell_conn_matrix.csv')
    log.info(f'Step 1/5: matrix for per cell connectivity of full cells loaded from {syn_percell_path}')
    # value to estimate syn current from surface area; from Holler et al., 2021
    syn_curr_val = 1.19
    log.info(f'Factor to caclulate synaptic current from synaptic area from Holler et al., 2021 = {syn_curr_val}')
    log.info('Per cell connectivity matrix is now multiplied by this factor')
    full_cell_matrix = syn_conn_matrix_percell * syn_curr_val

    #get dendritc and somatic surface area per cell, in µm²
    den_so_surface_area_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_ct_den_so_surface_area_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    denso_surface_area_df = pd.read_csv(f'{den_so_surface_area_path}/den_so_surface_area.csv')
    log.info(f'Step 2/5: matrix for dendritic and somatic surface area of full cells loaded from {den_so_surface_area_path}')
    #order surface area according to cellids 8similar ordering as matrix)
    denso_cellids = denso_surface_area_df['cellid']
    denso_cellids_order_inds = np.argsort(denso_cellids)
    denso_surface_area_df_ordered = denso_surface_area_df[denso_cellids_order_inds]

    #values for threshold differences to action potential
    spike_threshold_ct = {'MSN': 0, 'STN': 0, 'TAN': 0, 'GPe': 0, 'GPi': 0, 'LTS': 0, 'INT1': 0, 'INT2': 0, 'INT3': 0}
    log.info(f'These spike thresholds are used in the matrix {spike_threshold_ct}')
    log.info('Step 3/5: Spike thresholds are now multiplied with dendritic- somatic surface area into res variable')
    for ct in list(spike_threshold_ct.keys()):
        ct_inds = np.where(denso_surface_area_df_ordered['celltype'] == ct)[0]
        denso_surface_area_df_ordered.loc[ct_inds, 'res variable'] = denso_surface_area_df_ordered.loc[ct_inds, 'denso surface area'] * spike_threshold_ct[ct]

    denso_surface_area_df_ordered.to_csv(f'{f_name}/_denso_surface_area_spike_threshold_est.csv')

    log.info('Step 4/5: Matrix is now divided by the res variable')
    #divide only nonzero values
    nonzero_inds = np.where(full_cell_matrix > 0)
    full_cell_matrix = full_cell_matrix[nonzero_inds] / denso_surface_area_df_ordered['res variable']

    full_cell_matrix.to_csv(f'{f_name}/full_cell_matrix')

    log.info(f'Max matrix value = {np.max(full_cell_matrix):.2f}')

    log.info('Step 5/5: Plot matrix as heatmap')
    cmap_heatmap = sns.light_palette('black', as_cmap=True)
    inc_numbers_abs = ConnMatrix(data=full_cell_matrix.transpose().astype(float),
                                 title='Gain matrix full cells', filename=f_name, cmap=cmap_heatmap)
    inc_numbers_abs.get_heatmap(save_svg=True, annot=False, fontsize=fontsize)

    log.info('Analyses done')
