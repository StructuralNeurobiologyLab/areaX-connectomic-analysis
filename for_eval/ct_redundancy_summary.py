#summarise rudundancy performance scores of celltype classifier with mean and sd

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = True
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    ct_str = analysis_params.ct_str(with_glia= with_glia)
    redundancies = [1, 10, 20, 50]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/241002_j0251{version}_redudun_performaces_summary"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'single_ves_eval_log', log_dir=f_name)

    eval_path = f'{global_params.wd}/celltype_training/11-12-23_celltype_cross_val/'

    log.info(f'Load 10-fold cross_validation scores from {eval_path} for redundancies {redundancies}')
    columns = ['quantity', 'f1 score mean', 'f1 score sd']
    unique_quantities = np.hstack([ct_str, 'f1 score macro', 'accuracy'])

    for redun in redundancies:
        log.info(f'Get performance summary for {redun} redundancies')
        redun_df = pd.read_csv(f'{eval_path}/redun{redun}_performances.csv', index_col = 0)
        result_df = pd.DataFrame(columns=columns, index = range(len(unique_quantities)))
        result_df['quantity'] = unique_quantities
        f1_scores = np.array(redun_df['f1score']).reshape(len(unique_quantities), 3)
        result_df['f1 score mean'] = np.mean(f1_scores, axis = 1)
        result_df['f1 score sd'] = np.std(f1_scores, axis=1)
        result_df.to_csv(f'{f_name}/redun{redun}_performance_summary.csv')
        log.info('Calculate average of confusion matrices')
        ev0_conf_matrix = pd.read_csv(f'{eval_path}/conf_matrix_{redun}_eval0.csv', index_col=0)
        ev1_conf_matrix = pd.read_csv(f'{eval_path}/conf_matrix_{redun}_eval1.csv', index_col=0)
        ev2_conf_matrix = pd.read_csv(f'{eval_path}/conf_matrix_{redun}_eval2.csv', index_col=0)
        mean_conf_matrix = pd.DataFrame(columns = ev0_conf_matrix.columns, index=ev0_conf_matrix.columns)
        for ci in ev0_conf_matrix.columns:
            for ii in ev0_conf_matrix.columns:
                mean_conf_matrix.loc[ci, ii] = np.mean([ev0_conf_matrix.loc[ci, ii], ev1_conf_matrix.loc[ci, ii], ev2_conf_matrix.loc[ci, ii]])
        mean_conf_matrix.to_csv(f'{f_name}/conf_matrix_mean_redun{redun}.csv')

    log.info('Analysis done')

