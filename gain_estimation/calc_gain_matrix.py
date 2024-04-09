#get matrix of full cells with calculations for gain estimation

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
    fontsize = 20
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240327_j0251{version}_percell_gain_matrix_mcl_%i_fs%i" % (
        min_comp_len_cell, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('per_cell_conn_matrix_log', log_dir=f_name + '/logs/')
    log.info('Get per cell matrix that can be used for estimation')
    #load results from synaptic connectivity matrix, in µm²
    syn_percell_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_percell_conn_matrix_mcl_%i_TePkBrNGF_fs%i" % (
        min_comp_len_cell, fontsize)
    syn_conn_matrix_percell = pd.read_csv(f'{syn_percell_path}/percell_conn_matrix.csv', index_col = 0)
    log.info(f'Step 1/5: matrix for per cell connectivity of full cells loaded from {syn_percell_path}')
    # value to estimate syn current from surface area; from Holler et al., 2021
    syn_curr_val = 1.09
    log.info(f'Factor to caclulate synaptic current from synaptic area from Holler et al., 2021 = {syn_curr_val}')
    log.info('Per cell connectivity matrix is now multiplied by this factor')
    full_cell_matrix = syn_conn_matrix_percell * syn_curr_val

    #get dendritc and somatic surface area per cell, in µm²
    den_so_surface_area_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_ct_den_so_surface_area_mcl_%i_TePkBrNGF_fs20" % (
        min_comp_len_cell)
    denso_surface_area_df = pd.read_csv(f'{den_so_surface_area_path}/den_so_surface_area.csv', index_col=0)
    log.info(f'Step 2/5: matrix for dendritic and somatic surface area of full cells loaded from {den_so_surface_area_path}')
    #order surface area according to cellids 8similar ordering as matrix)
    denso_cellids = denso_surface_area_df['cellid']
    denso_cellids_order_inds = np.argsort(denso_cellids)
    denso_surface_area_df_ordered = denso_surface_area_df.loc[denso_cellids_order_inds]
    denso_surface_area_df_ordered = denso_surface_area_df_ordered.reset_index(drop = True)
    cellids = np.array(denso_surface_area_df_ordered['cellid'])

    #values for threshold differences to action potential
    #use values from Farries and Perkel, 2000
    #msn = -34.5, FS = -44.9 -> use for all INT types, LTS: -43.3
    #Farries and Perkel., 2002:
    #resting potential: msn = -72, LTS = -42, FS: -62, TAN = -51, GP = na
    #AP threshold = msn -37.7, LTS: -43.5, FS: -40.9, TAN = -43.7, GP = -47.7
    #use this to calculate values, INT1-3 use FS, GP use 60 as resting potential, STN use also FS value
    dt_spike_threshold_ct = {'MSN': 34.3, 'STN': 21.1, 'TAN': 7.3, 'GPe': 12.3, 'GPi': 12.3, 'LTS': 1.5, 'INT1': 21.1, 'INT2': 21.1, 'INT3': 21.1}
    log.info(f'These spike thresholds are used in the matrix {dt_spike_threshold_ct}')
    log.info('Step 3/5: Spike thresholds are now multiplied with dendritic- somatic surface area into res variable')
    #also give inhibitory celltypes negative sign
    inhibitory_list = ['MSN', 'LTS', 'INT1', 'INT2', 'INT3', 'GPe', 'GPi']
    for ct in list(dt_spike_threshold_ct.keys()):
        if ct in inhibitory_list:
            neuro_sign = -1
        else:
            neuro_sign = 1
        ct_inds = np.where(denso_surface_area_df_ordered['celltype'] == ct)[0]
        denso_surface_area_df_ordered.loc[ct_inds, 'res variable'] = denso_surface_area_df_ordered.loc[ct_inds, 'denso surface area'] * dt_spike_threshold_ct[ct] * neuro_sign

    denso_surface_area_df_ordered.to_csv(f'{f_name}/_denso_surface_area_spike_threshold_est.csv')

    log.info('Step 4/5: Matrix is now divided by the res variable')
    #for full_cell_matrix postsynaptic cells are in rows
    #multiply surface area * dt_spike_threshold with each postsynaptic cell -> each row
    matrix_values = np.array(full_cell_matrix)
    res_vector = np.array(denso_surface_area_df_ordered['res variable'])
    res_matrix = matrix_values / res_vector.reshape(-1, 1)
    full_cell_matrix_div = pd.DataFrame(data=res_matrix, columns=cellids, index=cellids)

    full_cell_matrix_div.to_csv(f'{f_name}/full_cell_matrix.csv')
    np.save(f'{f_name}/full_cell_matrix_array.npy', res_matrix)

    log.info(f'Max matrix value = {full_cell_matrix_div.max().max()} [nA/µm²*mV]')
    log.info(f'Min matrix value = {full_cell_matrix_div.min().min()} [nA/µm²*mV]')

    log.info('Step 5/5: Plot matrix as heatmap')
    cmap_heatmap = sns.diverging_palette(179,341, s= 70, l = 35, as_cmap=True)
    inc_numbers_abs = ConnMatrix(data=full_cell_matrix_div.transpose().astype(float),
                                 title='Gain matrix full cells', filename=f_name, cmap=cmap_heatmap)
    inc_numbers_abs.get_heatmap(save_svg=True, annot=False, fontsize=fontsize, center_zero=True)

    log.info('Analyses done')
