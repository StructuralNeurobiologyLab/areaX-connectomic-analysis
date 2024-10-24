#write function to plot morphological parameters for full cells (for two celltypes or more)

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

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    #celltypes that are compared
    ct1 = 6
    ct2 = 7
    #color_key = 'STNGP'
    color_key = 'STNGPINTv6'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241024_j0251{version}_%s_%s_morph_PCA_{fontsize}" % (
            ct_dict[ct1], ct_dict[ct2])
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('PCA morph', log_dir=f_name + '/logs/')
    n_comps = 1
    log.info(f'Number of components is {n_comps}')
    ct_colors = CelltypeColors(ct_dict = ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)
    #ct_palette = {'NGF type 1': '#232121', 'NGF type 2': '#38C2BA', 'NGF no type': '#707070', 'FS': '#912043'}

    #morph_data_path = 'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240220_j0251v5_INT_mito_radius_spiness_examplecells_mcl200_fs10_med1'
    morph_data_path = 'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241024_j0251v6_GPe_i_myelin_mito_radius_newmerger_mcl200_fs20_med1'
    log.info(f'Step 1/3: load dataframe with morphological parameters from {morph_data_path}')
    morph_data = pd.read_csv(f'{morph_data_path}/GPe_GPi_params.csv')
    #morph_data = pd.read_csv(f'{morph_data_path}/FS_ngf_params.csv')

    pca_params = ['axon median radius', 'axon mitochondria volume density', 'axon myelin fraction', 'soma diameter']
    #pca_params = ['axon median radius', 'axon mitochondria volume density', 'spine density', 'soma diameter']
    unique_groups = np.unique(morph_data['celltype'])
    log.info(f'Step 2/3: apply PCA to the following features {pca_params} and these groups {unique_groups}')
    #code based on chatgpt
    features = morph_data[pca_params]
    labels = morph_data['celltype']
    #standardize features for PCA
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    pca = PCA(n_components=n_comps)
    principal_components = pca.fit_transform(features_standardized)
    #new dataframe with principal components and labels
    if n_comps == 1:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
        pca_df['celltype'] = labels
        pca_df.to_csv(f'{f_name}/pca_principal_component.csv')
        log.info('Step 3/3: Plot results')
        plot_histogram_selection(dataframe=pca_df, x_data='PC1',
                                 color_palette=ct_palette, label='pca_one_comp', count='cells', foldername=f_name,
                                 hue_data='celltype', title=f'Separation by first principal component', fontsize = fontsize)
        sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel(f'number of cells', fontsize=fontsize)
        plt.xlabel('PC1', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Separation by first principal component')
        plt.savefig(f'{f_name}/pca_one_comp_hist.png')
        plt.savefig(f'{f_name}/pca_obe_comp_hist.svg')
        plt.close()
        sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel(f'% of cells', fontsize=fontsize)
        plt.xlabel('PC1', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Separation by first principal component')
        plt.savefig(f'{f_name}/pca_one_comp_hist_perc.png')
        plt.savefig(f'{f_name}/pca_one_comp_hist_perc.svg')
        plt.close()
    elif n_comps == 2:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['celltype'] = labels
        pca_df.to_csv(f'{f_name}/pca_two_principal_components.csv')

        log.info('Step 3/3: Plot results')
        g = sns.JointGrid(data=pca_df, x='PC1', y='PC2', hue="celltype", palette=ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10, palette=ct_palette)
        plt.savefig(f"{f_name}/pca_joinplot_celltypes.svg" )
        plt.savefig(f"{f_name}/pca_joinplot_celltypes.png" )
        plt.close()

    #plot them in grid plot with histplot in middle and scatterplots (see JK)
    #do UMAP on them

    log.info('Analysis done')