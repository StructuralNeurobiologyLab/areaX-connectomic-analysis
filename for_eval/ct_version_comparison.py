#check where new HVC_ids come from

if __name__ == '__main__':
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os as os
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors

    v6_wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    v5_wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'

    color_key = 'TePkBrGlia'
    ct_str = 'LMAN'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240111_j0251v6_vs_v5_{ct_str}_cellids_{color_key}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    v6_ct_dict = {0:'DA', 1:'LMAN', 2: 'HVC', 3:'MSN', 4:'STN', 5:'TAN', 6:'GPe', 7:'GPi', 8: 'LTS',
                          9:'INT1', 10:'INT2', 11:'INT3', 12:'ASTRO', 13:'OLIGO', 14:'MICRO', 15:'MIGR', 16:'FRAG'}
    v5_ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF", 11:"ASTRO", 12:"OLIGO", 13:'MICRO', 14:'FRAG'}
    ct_num_v6 = 1
    ct_num_v5 = 3
    ct_colors = CelltypeColors(ct_dict = v5_ct_dict)
    ct_palette = ct_colors.ct_palette(key = color_key)

    #load all cellids and celltypes
    ssd_v6 = SuperSegmentationDataset(working_dir=v6_wd)
    ssd_v5 = SuperSegmentationDataset(working_dir=v5_wd)
    v6_cellids = ssd_v6.ssv_ids
    v5_cellids = ssd_v5.ssv_ids
    v6_celltypes = ssd_v6.load_numpy_data('celltype_pts_e3')
    v5_celltypes = ssd_v5.load_numpy_data('celltype_pts_e3')
    summary_pd = pd.DataFrame(columns = ['v5', 'v6'])

    #get celltype ids from both versions
    v6_ct_ids = v6_cellids[v6_celltypes == ct_num_v6]
    v5_ct_ids = v5_cellids[v5_celltypes == ct_num_v5]
    summary_pd.loc[f'total number {ct_str} ids', 'v6'] = len(v6_ct_ids)
    summary_pd.loc[f'total number {ct_str} ids', 'v5'] = len(v5_ct_ids)
    #check which ids from v5 are also in v6
    same_inds = np.in1d(v6_ct_ids, v5_ct_ids)
    same_ct_ids = v6_ct_ids[same_inds]
    num_same_ct = len(same_ct_ids)
    summary_pd.loc[f'number same {ct_str} ids', 'v5'] = num_same_ct
    summary_pd.loc[f'number same {ct_str} ids', 'v6'] = num_same_ct
    summary_pd.loc[f'percent same {ct_str} ids', 'v5'] = 100 * num_same_ct/ len(v5_ct_ids)
    summary_pd.loc[f'percent same {ct_str} ids', 'v6'] = 100 * num_same_ct / len(v6_ct_ids)
    summary_pd.to_csv(f'{f_name}/summary_df.csv')
    #get celltypes of all cellids that are now hvc
    ct_celltypes_v5_all = v5_celltypes[np.in1d(v5_cellids, v6_ct_ids)]
    ct_celltypes_v5_all_str = [v5_ct_dict[i] for i in ct_celltypes_v5_all]
    v6_hvc_ids_df = pd.DataFrame(columns = ['cellid', 'celltype v5', 'celltype v6'], index = range(len(v6_ct_ids)))
    v6_hvc_ids_df['cellid'] = v6_ct_ids
    v6_hvc_ids_df['celltype v5'] = ct_celltypes_v5_all_str
    v6_hvc_ids_df['celltype v6'] = ct_str
    v6_hvc_ids_df.to_csv(f'{f_name}/{ct_str}_ids_v5_v6_cts.csv')
    #plot according to celltypes
    sns.countplot(data = v6_hvc_ids_df, x='celltype v5', palette=ct_palette)
    plt.ylabel('number of cells')
    plt.savefig(f'{f_name}/number_{ct_str}_v6_v5_cts.png')
    plt.close()
    sns.countplot(data=v6_hvc_ids_df, x='celltype v5', palette=ct_palette, stat='percent')
    plt.ylabel('% of cells')
    plt.savefig(f'{f_name}/perc_{ct_str}_v6_v5_cts.png')
    plt.close()