#get number and plot mergers

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    version = 'v6'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240830_j0251{version}_manual_msn_merger_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'merger_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                '240723_j0251v6_all_cellids_for_exclusion/' \
                        '240826_Random MSN IDS_RM_check.csv'
    log.info(f'evaluation results loaded from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    eval_df.loc[eval_df['merger?'] == 'y', 'merger?'] = 'True'
    eval_df.loc[eval_df['merger?'] == 'n', 'merger?'] = 'False'
    assert(len(np.unique(eval_df['merger?'])) == 2)
    unique_cts = np.unique(eval_df['celltype'])
    num_cells = len(eval_df)
    log.info(f'In total {num_cells} cells of these celltypes')
    # get number of mergers overall
    true_cell_df = eval_df[eval_df['merger?'] == 'False']
    true_cell_num = len(true_cell_df)
    log.info(f'{true_cell_num} are cells without mergers.')
    merger_cell_df = eval_df[eval_df['merger?'] == 'True']
    num_merger_df = len(merger_cell_df)
    frac_merger_df = num_merger_df / num_cells
    overview_df = pd.DataFrame(columns = ['number', 'fraction', 'merger?'])
    overview_df.loc[0, 'number'] = true_cell_num
    overview_df.loc[0, 'fraction'] = true_cell_num / num_cells
    overview_df.loc[0, 'merger?'] = 'False'
    overview_df.loc[1, 'number'] = num_merger_df
    overview_df.loc[1, 'fraction'] = frac_merger_df
    overview_df.loc[1, 'merger?'] = 'True'
    overview_df.to_csv(f'{f_name}/overview_df.csv')
    log.info(f'{num_merger_df} cells are mergers ({100 * frac_merger_df:.2f} %)')

    #get merger categories
    merger_cats = np.unique(merger_cell_df['merger with'])
    merger_cat_df = pd.DataFrame(columns=['number', 'fraction mergers', 'fraction all', 'merger with'], index= range(len(merger_cats)))
    merger_cat_df['merger with'] = merger_cats
    merger_groups = merger_cell_df.groupby('merger with')
    merger_cat_df['number'] = np.array(merger_groups.size())
    merger_cat_df['fraction mergers'] = np.array(merger_groups.size()) / num_merger_df
    merger_cat_df['fraction all'] = np.array(merger_groups.size()) / num_cells
    merger_cat_df.to_csv(f'{f_name}/merger_categories.csv')

    log.info('Plot results as barplot')
    #plot result as barplot
    sns.barplot(data = overview_df, x = 'merger?', y = 'number')
    plt.ylabel('number of cells', fontsize = fontsize)
    plt.xlabel('merger?', fontsize = fontsize)
    plt.title('All eval cells')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/number_mergers.png')
    plt.savefig(f'{f_name}/number_mergers.svg')
    plt.close()
    sns.barplot(data=overview_df, x='merger?', y='fraction')
    plt.ylabel('fraction of cells', fontsize=fontsize)
    plt.xlabel('merger?', fontsize=fontsize)
    plt.title('All eval cells')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_mergers.png')
    plt.savefig(f'{f_name}/frac_mergers.svg')
    plt.close()
    sns.barplot(data=merger_cat_df, x='merger with', y='number')
    plt.ylabel('number of cells', fontsize=fontsize)
    plt.xlabel('merger category', fontsize=fontsize)
    plt.title('merger category')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/number_mergers_cats.png')
    plt.savefig(f'{f_name}/number_mergers_cats.svg')
    plt.close()
    sns.barplot(data=merger_cat_df, x='merger with', y='fraction mergers')
    plt.ylabel('fraction of cells', fontsize=fontsize)
    plt.xlabel('merger category', fontsize=fontsize)
    plt.title('merger category fractions')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_mergers_cats.png')
    plt.savefig(f'{f_name}/frac_mergers_cats.svg')
    plt.close()
    log.info('Analysis finished')