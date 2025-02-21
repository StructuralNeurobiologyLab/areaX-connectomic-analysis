#STN connectivity and mophology seperation
#test if differences in LMAN/HVC connectivity, relate to in/output differences in GPe/i connectivity and mitochondria/er density differences

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.handler.basics import load_pkl2obj
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_histogram_selection
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from itertools import combinations

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    #celltypes that are compared
    ct = 4
    ct_str = ct_dict[ct]
    #color_key = 'STNGP'
    color_key = 'STNGPINTv6'
    fontsize = 12
    togp_only = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251{version}_{ct_str}_morph_conn_subpop_{fontsize}_conn_only_2gp_only"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('stn_morph_conn', log_dir=f_name)
    n_comps = 1
    log.info(f'Number of components is {n_comps}')
    ct_colors = CelltypeColors(ct_dict = ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)
    stn_color = ct_palette[ct_str]

    log.info('Step 1/6: Load input connectivity information from LMAN/HVC')
    lman_hvc_in_conn_path = 'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251v6_STN_LMAN_HVC_GP_input_ratio_fs20/STN_LMAN_HVC_result_per_cell.csv'
    log.info(f'Load LMAN/HVC info from {lman_hvc_in_conn_path}')
    lman_hvc_input_data = pd.read_csv(lman_hvc_in_conn_path, index_col=0)

    log.info('Step 2/6: Load input connectivity information from GPe/GPi')
    gp_in_conn_path = 'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251v6_STN_GPi_GPe_GP_input_ratio_fs20/STN_GPi_GPe_result_per_cell.csv'
    log.info(f'Load GPe/i input info from {gp_in_conn_path}')
    gp_input_data = pd.read_csv(gp_in_conn_path, index_col=0)

    log.info('Step 3/6: Load output connectivity information from GPe/GPi')
    gp_out_conn_path = 'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251v6_STN_GPi_GPe_output_ratio_fs20/STN_GPi_GPe_result_per_cell.csv'
    log.info(f'Load GPe/i input info from {gp_out_conn_path}')
    gp_output_data = pd.read_csv(gp_out_conn_path, index_col=0)

    log.info('Step 4/6: Load morphological information')
    org_path = 'cajal/scratch/users/arother/bio_analysis_results/general/241108_j0251v6_ct_morph_analyses_newmergers_mcl_200_ax200_TeBKv6MSNyw_fs20npca1_umap5_fc_synfullmivesgolgier/ct_morph_df.csv'
    log.info(f'Organelle info from {org_path}')
    org_data = pd.read_csv(org_path, index_col=0)
    org_data = org_data[org_data['celltype'] == 'STN']
    org_data = org_data.reset_index()


    log.info('Step 5/6: Combine information and plot results')
    stn_columns = ['cellid', 'LMAN/(LMAN + HVC) syn area', 'GPi/(GPe + GPi) input syn area', 'GPi/(GPe + GPi) output syn area',
                   'axon mi volume density', 'dendrite mi volume density', 'axon er area density', 'dendrite er area density']
    stn_result_df = pd.DataFrame(columns=stn_columns, index = range(len(lman_hvc_input_data)))
    #check if always same cellids in same order
    assert(np.all(lman_hvc_input_data['cellid'] == gp_input_data['cellid']))
    assert (np.all(lman_hvc_input_data['cellid'] == gp_output_data['cellid']))
    assert (np.all(lman_hvc_input_data['cellid'] == org_data['cellid']))
    stn_result_df['cellid'] = lman_hvc_input_data['cellid']
    stn_result_df['LMAN/(LMAN + HVC) syn area'] = lman_hvc_input_data['ratio syn area']
    stn_result_df['GPi/(GPe + GPi) input syn area'] = gp_input_data['ratio syn area']
    stn_result_df['GPi/(GPe + GPi) output syn area'] = gp_output_data['ratio syn area']
    for param in stn_columns:
        if not 'density' in param:
            continue
        stn_result_df[param] = org_data[param]

    if togp_only:
        log.info('Use only STN cells, that project to either GPe or GPi')
        stn_result_df = stn_result_df.dropna(subset='GPi/(GPe + GPi) output syn area')
        log.info(f'{len(stn_result_df)} cells of STN project to either GPe/i or both.')
        log.info('STNs without input from GPe or GPi get the input ratio set to 0')
        stn_result_df = stn_result_df.fillna(0)
    else:
        log.info('STNs without input or output from GPe or GPi get input and output ratio set to 0')
        stn_result_df = stn_result_df.fillna(0)


    stn_result_df.to_csv(f'{f_name}/stn_params.csv')

    key_list = stn_columns[1:]
    combs = list(combinations(range(len(key_list)), 2))
    for i, comb in enumerate(combs):
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= stn_result_df, x = x, y = y)
        g.plot_joint(sns.scatterplot, color = stn_color)
        g.plot_marginals(sns.histplot,  fill = False,
                         kde=False, bins='auto', color = stn_color)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize = fontsize)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize= fontsize)
        if "volume density" in x:
            g.ax_joint.set_xlabel("%s [µm³/µm]" % x)
        if "volume density" in y:
            g.ax_joint.set_ylabel("%s [µm³/µm]" % y)
        plt.savefig(f'{f_name}/stn_params_{i}_comb.png')
        plt.savefig(f'{f_name}/stn_params_{i}_comb.svg')
        plt.close()

    log.info('Step 6/6: PCA')

    #pca_params = ['LMAN/(LMAN + HVC) syn area', 'GPi/(GPe + GPi) input syn area','GPi/(GPe + GPi) output syn area','axon mi volume density', 'dendrite mi volume density']
    pca_params = ['LMAN/(LMAN + HVC) syn area', 'GPi/(GPe + GPi) input syn area', 'GPi/(GPe + GPi) output syn area']
    log.info(f'apply PCA to the following features {pca_params}')
    #code based on chatgpt
    features = stn_result_df[pca_params]
    #standardize features for PCA
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    pca = PCA(n_components=n_comps)
    principal_components = pca.fit_transform(features_standardized)
    #new dataframe with principal components and labels
    if n_comps == 1:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
        pca_df.to_csv(f'{f_name}/pca_principal_component.csv')
        log.info('Step 3/3: Plot results')
        #errors with histogram and hue in seaborn 0.13.2 with this code, switch to displot
        #plot_histogram_selection(dataframe=pca_df, x_data='PC1',
        #                         color_palette=ct_palette, label='pca_one_comp', count='cells', foldername=f_name,
        #                         hue_data='celltype', title=f'Separation by first principal component', fontsize = fontsize)
        sns.displot(data=pca_df, x='PC1', kind='hist', element="step",
                    fill=False, common_norm=False, multiple="dodge",
                    color = stn_color, linewidth=3)
        plt.ylabel(f'number of cells', fontsize=fontsize)
        plt.xlabel('PC1', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Separation by first principal component')
        plt.savefig(f'{f_name}/pca_one_comp_hist.png')
        plt.savefig(f'{f_name}/pca_obe_comp_hist.svg')
        plt.close()
        sns.displot(data=pca_df, x='PC1', kind='hist', element="step",
                    fill=False, common_norm=False, multiple="dodge",
                    color=stn_color, linewidth=3, stat='percent')
        plt.ylabel(f'% of cells', fontsize=fontsize)
        plt.xlabel('PC1', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Separation by first principal component')
        plt.savefig(f'{f_name}/pca_one_comp_hist_perc.png')
        plt.savefig(f'{f_name}/pca_one_comp_hist_perc.svg')
        plt.close()

    log.info('Analysis done')