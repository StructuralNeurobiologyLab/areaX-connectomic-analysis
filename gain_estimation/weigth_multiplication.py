#multiply input matrix x weight matrix with and without firing rate
#in the end get all GPi ids and see what the result is

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ConnMatrix
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns

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
    fontsize = 14
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240326_j0251{version}_weight_multiplication_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('matrix_mult_log', log_dir=f_name + '/logs/')
    log.info('Get per cell matrix that can be used for estimation')
    #load results from weight matrix full cells, in nA/µm²mV
    full_cell_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_percell_gain_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    full_cell_matrix = pd.read_csv(f'{full_cell_path}/full_cell_matrix.csv', index_col = 0)
    log.info(f'Step 1/5: weight matrix of full cells loaded from {full_cell_path}')
    # load results from weight matrix of input, in nA/µm²mV
    input_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240326_j0251{version}_percell_input_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    input_matrix = pd.read_csv(f'{full_cell_path}/input_matrix.csv', index_col=0)
    log.info(f'Step 2/5: weight matrix of full cells loaded from {full_cell_path}')
    log.info(f'This matrix includes input from {len(input_matrix.columns)} cells, {input_matrix.columns}')

    #get dendritc and somatic surface area per cell, in µm²
    den_so_surface_area_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_ct_den_so_surface_area_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    denso_surface_area_df = pd.read_csv(f'{den_so_surface_area_path}/den_so_surface_area.csv', index_col=0)
    log.info(f'Step 3/5: matrix for dendritic and somatic surface area of full cells loaded from {den_so_surface_area_path}')
    #order surface area according to cellids 8similar ordering as matrix)
    denso_cellids = denso_surface_area_df['cellid']
    denso_cellids_order_inds = np.argsort(denso_cellids)
    denso_surface_area_df_ordered = denso_surface_area_df.loc[denso_cellids_order_inds]
    denso_surface_area_df_ordered = denso_surface_area_df_ordered.reset_index(drop = True)
    cellids = np.array(denso_surface_area_df_ordered['cellid'])

    log.info('Step 4/5: Multiply matrices with each other')
    full_cell_array = full_cell_matrix.values
    input_array = input_matrix.values
    result_matrix = full_cell_array * input_array.reshape(-1, 1)
    result_matrix_df = pd.DataFrame(data = result_matrix, columns=cellids, index=cellids)
    result_matrix_df.to_csv(f'{f_name}/result_matrix.csv')

    log.info('Step 5/5: add information about firing rate to both matrices')
    firing_pred_path = f"cajal/scratch/users/arother/bio_analysis_results/general/240325_j0251{version}_ct_mi_vol_density_mcl_%i_ax200_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    firing_preds = pd.read_csv(f'{firing_pred_path}/overview_df_with_preds.csv', index_col=0)
    log.info(f'Prediction with linear regression from median axon mito volume density, loaded from {firing_pred_path}')
    full_ct_types = np.unique(denso_surface_area_df_ordered['celltype'])
    for ct in full_ct_types:
        firing_rate_ct = firing_preds.loc[ct, 'mean firing rate singing']
        ct_inds = np.where(denso_surface_area_df_ordered['celltype'] == ct)[0]
        denso_surface_area_df_ordered.loc[ct_inds, 'mean firing rate'] = denso_surface_area_df_ordered.loc[ct_inds, 'denso surface area'] * firing_rate_ct
    log.info('Multiply input matrix with firing rate from input celltype')
    input_celltype = 'LMAN'
    firing_rate_input = firing_preds.loc[input_celltype, 'mean firing rate singing']
    input_matrix_firing = input_matrix * firing_rate_input
    log.info('Multiply full cell matrix with firing rate of presynaptic cells')
    firing_rates_percell = np.array(denso_surface_area_df_ordered['mean firing rate'])
    full_cell_arr_firing = full_cell_array * firing_rates_percell
    full_cell_matrix_firing = pd.DataFrame(data = full_cell_arr_firing, columns=cellids, index=cellids)
    log.info('Multiply matrices with each other')
    input_arr_firing = input_matrix_firing.values
    result_matrix_firing = full_cell_arr_firing * input_arr_firing.reshape(-1, 1)
    result_matrix_firing_df = pd.DataFrame(data = result_matrix_firing, columns= cellids, index= cellids)

    log.info('Step 6/6: Identify GPi cells and their results')
    #get cellids
    GPi_cellds = cellids[denso_surface_area_df_ordered['celltype'] == 'GPi']
    #get entries for these cellids without firing rates
    gpi_res = result_matrix_df.loc[GPi_cellds]
    #get entries with firing rate
    gpi_res_firing = result_matrix_firing_df.loc[GPi_cellds]


    log.info('Analyses done')