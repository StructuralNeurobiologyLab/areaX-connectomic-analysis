#Agreement between the automatically predicted synapse sizes and the sizes obtained from manually painted synaptic junctions
#(Fig S1E): 40 synapses stratified over five logarithmic size bins, painted blind, measured with the same SyConn pipeline
#(see for_eval/synapse_area_validation/README.md). Reads the per-synapse areas from the curation folder.

if __name__ == '__main__':
    from bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    areas_file = 'synapse_area_validation/synapse_areas_40.csv'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/261003_j0251{version}_synapse_area_manual_validation"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('synapse_area_manual_validation_log', log_dir=f_name + '/logs/')
    log.info(f'Areas from {bio_params.file_locations}/{areas_file}')

    log.info('Step 1/2: Load areas')
    #columns: api = published size, pred = predicted sj through the pipeline (control), manual = painted sj through the pipeline
    areas = pd.read_csv(f'{bio_params.file_locations}/{areas_file}')
    predicted = areas['api'].values
    manual = areas['manual'].values
    control = areas['pred'].values
    n = len(areas)
    log.info(f'{n} synapses, predicted size {predicted.min():.3f} - {predicted.max():.3f} µm², manual size {manual.min():.3f} - {manual.max():.3f} µm²')
    control_ratio = control / predicted
    log.info(f'Control (predicted sj through the pipeline / published size): median {np.median(control_ratio):.3f}, '
             f'max deviation {np.max(np.abs(control_ratio - 1)) * 100:.1f} %')

    log.info('Step 2/2: Agreement statistics')
    slope, intercept = np.polyfit(manual, predicted, 1)
    r2 = np.corrcoef(manual, predicted)[0, 1] ** 2
    log_slope, log_intercept = np.polyfit(np.log10(manual), np.log10(predicted), 1)
    log_r2 = np.corrcoef(np.log10(manual), np.log10(predicted))[0, 1] ** 2
    ratio = np.log(predicted / manual)
    results = pd.DataFrame({'n': [n], 'linear slope': [slope], 'linear intercept': [intercept], 'linear r2': [r2],
                            'log-log slope': [log_slope], 'log-log intercept': [log_intercept], 'log-log r2': [log_r2],
                            'geometric mean predicted/manual': [np.exp(ratio.mean())], 'sd log10 ratio': [np.log10(np.e) * ratio.std(ddof=1)],
                            'spearman': [spearmanr(predicted, manual)[0]]})
    results.to_csv(f'{f_name}/synapse_area_validation_results.csv', index=False)
    log.info('Results:\n' + results.round(3).T.to_string())
    lim = 1.05 * max(predicted.max(), manual.max())
    plt.figure(figsize=(6, 6))
    plt.plot([0, lim], [0, lim], color='gray', linestyle='--', label='identity')
    plt.scatter(manual, predicted, color='black', s=25)
    plt.plot([0, lim], [intercept, slope * lim + intercept], color='#3a7ca5', label=f'linear fit, n = {n}, slope {slope:.2f}, r² {r2:.2f}')
    plt.xlabel('manually painted synapse size [µm²]', fontsize=fontsize)
    plt.ylabel('predicted synapse size [µm²]', fontsize=fontsize)
    plt.legend()
    plt.savefig(f'{f_name}/synapse_area_validation.png')
    plt.savefig(f'{f_name}/synapse_area_validation.svg')
    plt.close()
    log.info('Analysis finished')
