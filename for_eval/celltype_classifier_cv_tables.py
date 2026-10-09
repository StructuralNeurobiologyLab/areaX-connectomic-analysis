#Tables S5-S7: ground truth cells per cell type, F1 scores per cell type / macro F1 / per-cell accuracy for the redundancies
#1, 10, 20 and 50 and the confusion matrix at redundancy 20, from the held-out predictions of the 8-fold cross-validation of the
#PTv3 cell type classifier (per-fold csvs written by the evaluation script of the classifier repository, see curation README)

if __name__ == '__main__':
    from bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    import os as os
    import pandas as pd
    import numpy as np

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    cv_folder = 'celltype_classifier_cv_v8'
    n_folds = 8
    gt_file = 'j0251_celltype_gt_v8_j0251_72_seg_20210127_agglo2_IDs.csv'
    redundancies = [1, 10, 20, 50]
    table_redundancy = 20
    #class ids of the v8 ground truth and their order in the tables
    gt_classes = {0: 'DA', 1: 'LMAN', 2: 'HVC', 3: 'MSN', 4: 'STN', 5: 'TAN', 6: 'GPe', 7: 'GPi', 8: 'LTS', 9: 'INT1', 10: 'INT2',
                  11: 'INT3', 12: 'ASTRO', 13: 'OLIGO', 14: 'MICRO', 15: 'MIGR', 16: 'FRAG', 17: 'INT4', 18: 'OPC'}
    table_order = ['DA', 'LMAN', 'HVC', 'MSN', 'STN', 'TAN', 'GPe', 'GPi', 'LTS', 'INT1', 'INT2', 'INT3', 'INT4', 'ASTRO', 'OLIGO',
                   'OPC', 'MICRO', 'MIGR', 'FRAG']
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/261003_j0251{version}_celltype_classifier_cv_tables_f%i_r%i" % (
        n_folds, table_redundancy)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('celltype_classifier_cv_tables_log', log_dir=f_name + '/logs/')
    log.info(f'{n_folds} folds from {bio_params.file_locations}/{cv_folder}, ground truth {gt_file}')

    log.info('Step 1/3: Table S5, ground truth cells per cell type')
    #no header, columns cellid and cell type name
    gt_df = pd.read_csv(f'{bio_params.file_locations}/{cv_folder}/{gt_file}', header=None, names=['cellid', 'celltype'])
    gt_counts = gt_df['celltype'].value_counts().reindex(table_order)
    gt_counts.index.name = 'cell type'
    gt_counts.rename('number of ground truth cells').to_csv(f'{f_name}/table_S5_ground_truth_cells.csv')
    log.info(f'{len(gt_df)} ground truth cells:\n' + gt_counts.to_string())

    log.info('Step 2/3: Table S6, F1 scores per redundancy')
    cv_df = pd.concat([pd.read_csv(f'{bio_params.file_locations}/{cv_folder}/fold{k}.csv') for k in range(n_folds)])
    assert len(cv_df) == len(gt_df) and set(cv_df['cell']) == set(gt_df['cellid'])
    labels = cv_df['label'].map(gt_classes).values
    class_ids = list(gt_classes.keys())
    f1_df = pd.DataFrame(index=table_order + ['f1 score macro', 'accuracy'], columns=[f'redundancy {r}' for r in redundancies], dtype=float)
    f1_df.index.name = 'cell type'
    for r in redundancies:
        preds = cv_df[f'pred_r{r}'].map(gt_classes).values
        f1_per_class = f1_score(labels, preds, labels=table_order, average=None)
        f1_df.loc[table_order, f'redundancy {r}'] = f1_per_class
        f1_df.loc['f1 score macro', f'redundancy {r}'] = f1_score(labels, preds, labels=table_order, average='macro')
        f1_df.loc['accuracy', f'redundancy {r}'] = accuracy_score(labels, preds)
    f1_df.round(3).to_csv(f'{f_name}/table_S6_f1_scores.csv')
    log.info('F1 scores:\n' + f1_df.round(3).to_string())

    log.info(f'Step 3/3: Table S7, confusion matrix at redundancy {table_redundancy}')
    preds = cv_df[f'pred_r{table_redundancy}'].map(gt_classes).values
    conf = pd.DataFrame(confusion_matrix(labels, preds, labels=table_order), index=table_order, columns=table_order)
    conf.index.name = 'ground truth \\ predicted'
    conf.to_csv(f'{f_name}/table_S7_confusion_matrix.csv')
    log.info(f'{np.trace(conf.values)} of {conf.values.sum()} cells correct\n' + conf.to_string())
    log.info('Analysis finished')
