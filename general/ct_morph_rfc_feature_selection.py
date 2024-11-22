#use rfc to select features which are most important to separate cell types in this dataset.
#plot results afterwards; inspired by ChatGPT suggestions

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
from syconn.handler.config import initialize_logging
from analysis_params import Analysis_Params
from analysis_colors import CelltypeColors

if __name__ == '__main__':

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    full_cells_only = True
    axon_only = False
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    color_key = 'TeBKv6MSNyw'
    fontsize = 20
    n_umap_runs = 5
    alpha = 0.5
    remove_ct = None
    if remove_ct is not None:
        remove_ct_str = ct_dict[remove_ct]
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/241108_j0251{version}_ct_rfc_morph_nm_%s_fs%i" \
                 f"_umap{n_umap_runs}_a{alpha}_no{remove_ct_str}" % (
                     color_key, fontsize)
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/241108_j0251{version}_ct_rfc_morph_nm_%s_fs%i" \
                 f"_umap{n_umap_runs}_a{alpha}" % (
                     color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('ct_morph_rfc', log_dir=f_name + '/logs/')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)

    log.info('Step 1/3: Load and standardize the data')
    morph_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                 '241108_j0251v6_ct_morph_analyses_newmergers_mcl_200_ax200_TeBKv6MSNyw_fs20npca1_umap5_fc_synfullmivesgolgier/' \
                 'ct_morph_df.csv'
    #morph_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
    #             '241108_j0251v6_ct_morph_analyses_newmergers_mcl_200_ax200_TeBKv6MSNyw_fs20npca1_umap5_fc_synfullmivcgolgier/' \
    #             'ct_morph_df.csv'

    log.info(f'Use morphological parameters from {morph_path}')
    morph_df = pd.read_csv(morph_path, index_col=0)
    if remove_ct is not None:
        morph_df = morph_df[morph_df['celltype'] != remove_ct_str]
    unique_ct_str = np.unique(morph_df['celltype'])
    log.info(f'There are in total {len(morph_df)} cells in the dataset from the following celltypes: {unique_ct_str}')
    param_list = morph_df.columns[2:]
    log.info(f'The following parameters are loaded for rfc training: {param_list}')

    morph_features = morph_df[param_list]
    morph_labels = morph_df['celltype']

    #Normalize the data
    scaler = StandardScaler()
    morph_features_scaled = scaler.fit_transform(morph_features)  # Normalize features

    log.info('Step 2/3: Initialize RFC and do recursive feature elimination with cross-validation')
    # Initialize the Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    # Recursive Feature Elimination with Cross-Validation (RFECV)
    # StratifiedKFold ensures that each fold has the same proportion of each class
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # RFECV will train the RFC with different subsets of features to find the optimal set
    rfecv = RFECV(estimator=rfc, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(morph_features_scaled, morph_labels)  # Use normalized data for feature selection

    # View the optimal number of features
    optimal_num_features = rfecv.n_features_
    log.info(f"Optimal number of features: {optimal_num_features}")

    #plot number of features vs cross-validation score
    #see scikit-learn.org examples

    cv_results = pd.DataFrame(rfecv.cv_results_)
    cv_results['n_features'] = range(1, len(cv_results) + 1)
    cv_results.to_csv(f'{f_name}/rfecv_cv_results.csv')
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.savefig(f'{f_name}/rfecv_cv_scores.png')
    plt.savefig(f'{f_name}/rfecv_cv_scores.svg')
    plt.close()

    #print accuracy with and without each feature
    # Full model accuracy (using all features)
    full_model_accuracy = cross_val_score(rfc, morph_features_scaled, morph_labels, cv=cv, scoring='accuracy').mean()
    log.info(f"Accuracy with all features: {full_model_accuracy:.4f}")

    # Now, evaluate the accuracy after removing each feature one by one
    num_parameters = len(morph_features.columns)
    accuracy_rfc_wo_feature = pd.DataFrame(columns = ['parameter', 'accuracy without parameter'], index = range(num_parameters))
    for i, feature in enumerate(morph_features.columns):
        X_without_feature = np.delete(morph_features_scaled, i, axis=1)
        accuracy_without_feature = cross_val_score(rfc, X_without_feature, morph_labels, cv=cv, scoring='accuracy').mean()
        accuracy_rfc_wo_feature.loc[i, 'parameter'] = feature
        accuracy_rfc_wo_feature.loc[i, 'accuracy without parameter'] = accuracy_without_feature

    accuracy_rfc_wo_feature.to_csv(f'{f_name}/accuracy_without_parameters.csv')

    # Get the names of the selected features
    selected_features = morph_features.columns[rfecv.support_]
    log.info(f"Selected features: {selected_features.tolist()}")

    log.info('Step 3/3: use selected features and plot UMAP')
    # Create a reduced feature matrix with only the selected features
    morph_features_selected = morph_features_scaled[:, rfecv.support_]  # Use only the selected features
    # Use Cross-Validation on the selected features for confirmation
    cross_val_scores = cross_val_score(rfc, morph_features_selected, morph_labels, cv=cv, scoring='accuracy')
    log.info(f"Cross-validated accuracy with selected features: {cross_val_scores.mean():.2f} ± {cross_val_scores.std():.2f}")

    # Train-Test Split Evaluation
    test_size = 0.2
    morph_features_train, morph_features_test, labels_train, labels_test = train_test_split(morph_features_selected, morph_labels, test_size=0.2, random_state=42, stratify=morph_labels)

    # Train RFC on training data
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(morph_features_train, labels_train)

    # Predict and evaluate on test data
    y_pred = rfc.predict(morph_features_test)
    test_accuracy = accuracy_score(labels_test, y_pred)
    log.info(f"Test Accuracy with selected features on rfc trained on split samples into training and test (test size = {test_size}): {test_accuracy:.4f}")

    # use umap, see also code from ct_morph_analyses
    np.random.seed(42)
    log.info(f'UMAP will be run {n_umap_runs} times')
    if remove_ct is None:
        #if run with MSN, also run UMAP based on selected features to show separation without MSN
        no_msn_inds = morph_labels != 'MSN'
        nomsn_features_selected = morph_features_selected[no_msn_inds]
        nomsn_morph_df = morph_df[no_msn_inds]
    for i in range(n_umap_runs + 1):
        fc_reducer = UMAP()
        fc_embedding = fc_reducer.fit_transform(morph_features_selected)
        umap_df = pd.DataFrame(columns=['cellid', 'celltype', 'UMAP 1', 'UMAP 2'])
        umap_df['cellid'] = morph_df['cellid']
        umap_df['celltype'] = morph_df['celltype']
        umap_df['UMAP 1'] = fc_embedding[:, 0]
        umap_df['UMAP 2'] = fc_embedding[:, 1]
        umap_df.to_csv(f'{f_name}/fc_umap_embeddings_{i}.csv')
        sns.scatterplot(x=fc_embedding[:, 0], y=fc_embedding[:, 1], hue=morph_labels, palette=ct_palette, alpha=alpha)
        plt.title('UMAP Visualization of full cells')
        plt.xlabel('UMAP 1', fontsize=fontsize)
        plt.ylabel('UMAP 2', fontsize=fontsize)
        plt.legend()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/umap_selected_{i}.png')
        plt.savefig(f'{f_name}/umap_selected_{i}.svg')
        plt.close()
        if remove_ct is None:
            #plot also without MSN
            fcnoMSN_reducer = UMAP()
            fcnoMSN_embedding = fc_reducer.fit_transform(nomsn_features_selected)
            nomsn_umap_df = pd.DataFrame(columns=['cellid', 'celltype', 'UMAP 1', 'UMAP 2'])
            nomsn_umap_df['cellid'] = nomsn_morph_df['cellid']
            nomsn_umap_df['celltype'] = nomsn_morph_df['celltype']
            nomsn_umap_df['UMAP 1'] = fcnoMSN_embedding[:, 0]
            nomsn_umap_df['UMAP 2'] = fcnoMSN_embedding[:, 1]
            nomsn_umap_df.to_csv(f'{f_name}/fcnomsn_umap_embeddings_{i}.csv')
            sns.scatterplot(data = nomsn_umap_df, x='UMAP 1', y='UMAP 2', hue=morph_labels, palette=ct_palette,
                            alpha=alpha)
            plt.title('UMAP Visualization of full cells')
            plt.xlabel('UMAP 1', fontsize=fontsize)
            plt.ylabel('UMAP 2', fontsize=fontsize)
            plt.legend()
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/umap_selected_noMSN_{i}.png')
            plt.savefig(f'{f_name}/umap_selected_noMSN_{i}.svg')
            plt.close()

    #plot feature correlation
    selected_features = morph_features.columns[rfecv.support_]
    X_selected_df = pd.DataFrame(morph_features_selected, columns=selected_features)
    correlation_matrix = X_selected_df.corr()

    plt.figure(figsize=(10, 8))
    cmap_heatmap = sns.light_palette('black', as_cmap=True)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap_heatmap)
    plt.title("Correlation Matrix of Selected Features")
    plt.savefig(f'{f_name}/feature_corr_matrix.png')
    plt.savefig(f'{f_name}/feature_corr_matrix.svg')
    plt.close()

    log.info('Analysis done')

