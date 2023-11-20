#check if ngf can be split into two different populations
#parameters should be soma diameter, axon mito density, spine density
#similar to GPe_i_mylein mito radius

if __name__ == '__main__':
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_per_cell_mito_myelin_info, \
        get_spine_density, get_cell_soma_radius
    from syconn.handler.basics import write_obj2pkl
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import ranksums, kruskal
    from itertools import combinations

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    bio_params = Analysis_Params(working_dir = global_params.wd, version = 'v5')
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    fontsize_jointplot = 10
    use_skel = False  # if true would use skeleton labels for getting soma; vertex labels more exact, also probably faster
    use_median = True  # if true use median of vertex coordinates to find centre
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/231120_j0251v5_ngf_fs_mito_radius_spiness_examplecells_mcl%i_fs%i_med%i" % \
             (min_comp_len, fontsize_jointplot, use_median)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('NGF seperation morphology', log_dir=f_name + '/logs/')
    log.info('Parameters to check: soma diameter, axon median radius, axon mitochondria density, axon myelin fraction, spine density')
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
    if use_median:
        log.info('Median of coords used to get soma centre')
    else:
        log.info('Mean of coords used to get soma centre')
    log.info("Finding potential ngf subpopulations starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    known_mergers = bio_params.load_known_mergers()
    ngf_cell_dict = bio_params.load_cell_dict(10)
    ngf_ids = np.array(list(ngf_cell_dict.keys()))
    merger_inds = np.in1d(ngf_ids, known_mergers) == False
    ngf_ids = ngf_ids[merger_inds]
    ngf_ids = check_comp_lengths_ct(cellids=ngf_ids, fullcelldict=ngf_cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False,
                                            max_path_len=None)

    log.info(f'{len(ngf_ids)} NGFs are suitable for analysis')

    fs_cell_dict = bio_params.load_cell_dict(8)
    fs_ids = np.array(list(fs_cell_dict.keys()))
    merger_inds = np.in1d(fs_ids, known_mergers) == False
    fs_ids = fs_ids[merger_inds]
    fs_ids = check_comp_lengths_ct(cellids=fs_ids, fullcelldict=fs_cell_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    log.info(f'{len(fs_ids)} Fs are suitable for analysis')

    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    cached_mito_ids = sd_mitossv.ids
    cached_mito_mesh_bb = sd_mitossv.load_numpy_data("mesh_bb")
    cached_mito_rep_coords = sd_mitossv.load_numpy_data("rep_coord")
    cached_mito_volumes = sd_mitossv.load_numpy_data("size")


    log.info("Step 1/4: Get information from NGF")
    log.info('Get information about mitos, myelin and axon radius')
    ngf_input = [[ngf_id, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, ngf_cell_dict] for ngf_id in ngf_ids]
    ngf_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, ngf_input)
    ngf_output = np.array(ngf_output)
    #[ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
    axon_median_radius_ngf = ngf_output[:, 0]
    axon_mito_volume_density_ngf = ngf_output[:, 1]
    axon_myelin_ngf = ngf_output[:, 2]
    total_mito_volume_density_ngf = ngf_output[:, 4]

    ngf_nonzero = axon_median_radius_ngf > 0
    # get soma diameter from all GPe
    log.info('Get information about soma diameter')
    ngf_soma_results = start_multiprocess_imap(get_cell_soma_radius, ngf_ids)
    ngf_soma_results = np.array(ngf_soma_results, dtype='object')
    ngf_diameters = ngf_soma_results[:, 1].astype(float) * 2
    #get spine density from all ngf cells
    log.info('Get cell spine density')
    ngf_input = [[ngf_id, min_comp_len, ngf_cell_dict] for ngf_id in ngf_ids]
    spine_density = start_multiprocess_imap(get_spine_density, ngf_input)
    spine_density = np.array(spine_density)
    ngf_params = {"axon median radius": axon_median_radius_ngf[ngf_nonzero], "axon mitochondria volume density": axon_mito_volume_density_ngf[ngf_nonzero],
                  'soma diameter': ngf_diameters[ngf_nonzero],'spine density': spine_density[ngf_nonzero],
                  'total mitochondria volume density': total_mito_volume_density_ngf[ngf_nonzero],
                  "cellids": ngf_ids[ngf_nonzero]}
    ngf_param_df = pd.DataFrame(ngf_params)
    ngf_param_df.to_csv("%s/ngf_params.csv" % f_name)

    log.info("Step 2/4 Plot results")
    key_list = list(ngf_params.keys())[:-1]

    for key in key_list:
        if 'results' in key or 'diameter' in key:
            xhist = f'{key} [µm]'
        elif 'volume density' in key:
            xhist = f'{key} [µm³/µm]'
        elif 'spine density' in key:
            xhist = f'{key} [1/µm]'
        else:
            xhist = key
        sns.histplot(x=key, data=ngf_param_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist.png')
        plt.close()
        sns.histplot(x=key, data=ngf_param_df, color='black', common_norm=True,
                     fill=False, element="step", linewidth=3
                     )
        plt.ylabel('fraction of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_norm.png')
        plt.close()

    combs = list(combinations(range(len(key_list)), 2))
    #sns.set(font_scale=1.5)
    for comb in combs:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= ngf_params, x = x, y = y)
        g.plot_joint(sns.scatterplot, color = 'black')
        g.plot_marginals(sns.histplot,  fill = False,
                         kde=False, bins='auto', color = 'black')
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize = fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize= fontsize_jointplot)
        if "radius" in x or 'diameter' in x:
            g.ax_joint.set_xlabel("%s [µm]" % x)
        elif "volume density" in x:
            g.ax_joint.set_xlabel("%s [µm³/µm]" % x)
        elif 'spine density' in x:
            g.ax_joint.set_xlabel('%s [1/µm]')

        if "radius" in y or 'diameter' in y:
            g.ax_joint.set_ylabel("%s [µm]" % y)
        elif "volume density" in y:
            g.ax_joint.set_ylabel("%s [µm³/µm]" % y)
        elif 'spine density' in y:
            g.ax_joint.set_ylabel('%s [1/µm]' % y)

        plt.savefig("%s/%s_%s_joinplot_all.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot_all.png" % (f_name, x, y))
        plt.close()

    soma_den_thresh = 11.5
    axon_med_radius_thresh = 0.11
    mito_density_thresh = 0.025
    spine_density_thresh = 0.025
    log.info(f'Step 3/3:Seperate populations by manual thresholds: \n'
             f'for soma density = {soma_den_thresh} µm, axon median radius = {axon_med_radius_thresh} µm, \n'
             f'mito volume density = {mito_density_thresh} µm³/µm, spine density = {spine_density_thresh} 1/µm')
    threshold = [axon_med_radius_thresh, mito_density_thresh, soma_den_thresh, spine_density_thresh]
    ct_palette = {'type 1': '#232121', 'type 2': '#38C2BA', 'no type':'#707070'}
    #seperate based on these threshold and if cells don't fit into either category, plot in grey
    type_1_inds_collected = np.zeros((len(ngf_param_df), 4))
    type_2_inds_collected = np.zeros((len(ngf_param_df), 4))
    for i, key in enumerate(key_list):
        if 'total mito' in key:
            continue
        if 'spine' in key:
            type_1_inds = ngf_params[key] >= threshold[i]
            type_2_inds = ngf_params[key] < threshold[i]
        else:
            type_1_inds = ngf_params[key] < threshold[i]
            type_2_inds = ngf_params[key] >= threshold[i]
        type_1_inds_collected[:, i] = type_1_inds
        type_2_inds_collected[:, i] = type_2_inds
    type_1_inds_all = np.all(type_1_inds_collected, axis = 1)
    type_2_inds_all = np.all(type_2_inds_collected, axis = 1)
    type1_ids = ngf_params['cellids'][type_1_inds_all]
    ngf_param_df.loc[type_1_inds_all, 'celltype'] = 'type 1'
    type2_ids = ngf_params['cellids'][type_2_inds_all]
    ngf_param_df.loc[type_2_inds_all, 'celltype'] = 'type 2'
    no_type_inds = np.all(np.array([type_1_inds_all, type_2_inds_all]) == False, axis = 0)
    no_type_ids = ngf_params['cellids'][no_type_inds]
    ngf_param_df.loc[no_type_inds, 'celltype'] = 'no type'
    assert(len(ngf_param_df) == len(type1_ids) + len(type2_ids) + len(no_type_ids))
    ngf_param_df.to_csv("%s/ngf_params.csv" % f_name)
    write_obj2pkl(f'{bio_params.file_locations}/ngf_type1_ids.pkl', type1_ids)
    write_obj2pkl(f'{bio_params.file_locations}/ngf_type2_ids.pkl', type2_ids)
    write_obj2pkl(f'{f_name}/ngf_type1_ids.pkl', type1_ids)
    write_obj2pkl(f'{f_name}/ngf_type2_ids.pkl', type2_ids)
    log.info(f'Type 1 NGF are {len(type1_ids)} cells, Type 2 NGF are {len(type2_ids)} cells, {len(no_type_ids)} belong to no type')

    example_cellids = [126798179, 1155532413, 15724767, 24397945]
    example_inds = np.in1d(ngf_params['cellids'], example_cellids)
    for comb in combs:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= ngf_param_df, x = x, y = y, hue = "celltype", palette = ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot,  fill = True, alpha = 0.3,
                         kde=False, bins=20, palette = ct_palette)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize = fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize= fontsize_jointplot)
        if "radius" in x or 'diameter' in x:
            g.ax_joint.set_xlabel("%s [µm]" % x)
            scatter_x = "%s [µm]" % x
        elif "volume density" in x:
            g.ax_joint.set_xlabel("%s [µm³/µm]" % x)
            scatter_x = "%s [µm³/µm]" % x
        elif 'spine density' in x:
            g.ax_joint.set_xlabel('%s [1/µm]' % x)
            scatter_x = "%s [1/µm]" % x
        else:
            scatter_x = x

        if "radius" in y or 'diameter' in y:
            g.ax_joint.set_ylabel("%s [µm]" % y)
            scatter_y = "%s [µm]" % y
        elif "volume density" in y:
            g.ax_joint.set_ylabel("%s [µm³/µm]" % y)
            scatter_y = "%s [µm³/µm]" % y
        elif 'spine density' in y:
            g.ax_joint.set_ylabel('%s [1/µm]' % y)
            scatter_y = "%s [1/µm]" % y
        else:
            scatter_y = y

        plt.savefig("%s/%s_%s_joinplot.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot.png" % (f_name, x, y))
        plt.close()

        example_x = ngf_params[x][example_inds]
        example_y = ngf_params[y][example_inds]
        plt.scatter(ngf_param_df[x], ngf_param_df[y], color = 'gray')
        plt.scatter(example_x, example_y, color = 'red')
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        plt.savefig(f'{f_name}/{x}_{y}_scatter_examplecells.png')
        plt.close()

    log.info('Step 3/4: Get information about FS and plot again')
    log.info('Get information about mitos, myelin and axon radius')
    fs_input = [[fs_id, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, fs_cell_dict] for
                 fs_id in fs_ids]
    fs_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, fs_input)
    fs_output = np.array(fs_output)
    # [ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
    axon_median_radius_fs = fs_output[:, 0]
    axon_mito_volume_density_fs = fs_output[:, 1]
    axon_myelin_fs = fs_output[:, 2]
    total_mito_volume_density_fs = fs_output[:, 4]
    fs_nonzero = axon_median_radius_fs > 0
    # get soma diameter from all FS
    log.info('Get information about soma diameter')
    fs_soma_results = start_multiprocess_imap(get_cell_soma_radius, fs_ids)
    fs_soma_results = np.array(fs_soma_results, dtype='object')
    fs_diameters = fs_soma_results[:, 1].astype(float) * 2
    # get spine density from all ngf cells
    log.info('Get cell spine density')
    fs_input = [[fs_id, min_comp_len, fs_cell_dict] for fs_id in fs_ids]
    spine_density = start_multiprocess_imap(get_spine_density, fs_input)
    spine_density = np.array(spine_density)
    log.info('Save information as df')
    fs_params = {"axon median radius": axon_median_radius_fs[fs_nonzero],
                  "axon mitochondria volume density": axon_mito_volume_density_fs[fs_nonzero],
                  'soma diameter': fs_diameters[fs_nonzero], 'spine density': spine_density[fs_nonzero],
                 'total mitochondria volume density': total_mito_volume_density_fs[fs_nonzero],
                  "cellids": fs_ids[fs_nonzero]}
    fs_param_df = pd.DataFrame(fs_params)
    fs_param_df.to_csv("%s/fs_params.csv" % f_name)
    fs_param_df['celltype'] = 'FS'
    comb_df = pd.concat([ngf_param_df, fs_param_df], axis = 0, ignore_index=True)
    comb_df.loc[comb_df['celltype'] == 'type 1', 'celltype'] = 'NGF type 1'
    comb_df.loc[comb_df['celltype'] == 'type 2', 'celltype'] = 'NGF type 2'
    comb_df.loc[comb_df['celltype'] == 'no type', 'celltype'] = 'NGF no type'
    ct_palette = {'NGF type 1': '#232121', 'NGF type 2': '#38C2BA', 'NGF no type': '#707070', 'FS': '#912043'}
    comb_df.to_csv(f'{f_name}/FS_ngf_params.csv')
    example_cellids = [126798179, 1155532413, 15724767, 24397945, 1080627023]
    example_inds = np.in1d(comb_df['cellids'], example_cellids)
    for comb in combs:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data=comb_df, x=x, y=y, hue="celltype", palette=ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=20, palette=ct_palette)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        if "radius" in x or 'diameter' in x:
            scatter_x = "%s [µm]" % x
        elif "volume density" in x:
            scatter_x = "%s [µm³/µm]" % x
        elif 'spine density' in x:
            scatter_x = '%s [1/µm]' % x
        else:
            scatter_x = key
        if "radius" in y or 'diameter' in y:
            scatter_y = "%s [µm]" % y
        elif "volume density" in y:
            scatter_y = "%s [µm³/µm]" % y
        elif 'spine density' in y:
            scatter_y = '%s [1/µm]' % y
        g.ax_joint.set_xlabel(scatter_x)
        g.ax_joint.set_ylabel(scatter_y)
        plt.savefig("%s/%s_%s_joinplot_comb.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot_comb.png" % (f_name, x, y))
        plt.close()
        example_x = comb_df[x][example_inds]
        example_y = comb_df[y][example_inds]
        plt.scatter(comb_df[x], comb_df[y], color='gray')
        plt.scatter(example_x, example_y, color='red')
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        plt.savefig(f'{f_name}/{x}_{y}_scatter_examplecells_ngffs.png')
        plt.close()

    log.info('Step 4/4 Calculate statistics')
    #exclude no type in statistics
    stat_df = comb_df[comb_df['celltype'] != 'NGF no type']
    #kruskal wallis test to get statistics over different celltypes
    kruskal_results_df = pd.DataFrame(columns=['stats', 'p-value'], index = key_list)
    # get kruskal for all syn sizes between groups
    #also get ranksum results between celltyes
    ct_str = np.unique(stat_df['celltype'])
    group_comps = list(combinations(range(len(ct_str)), 2))
    ranksum_columns = [f'{ct_str[gc[0]]} vs {ct_str[gc[1]]}' for gc in group_comps]
    ranksum_df = pd.DataFrame(columns=ranksum_columns)
    for key in key_list:
        key_groups = [group[key].values for name, group in
                            stat_df.groupby('celltype')]
        kruskal_res = kruskal(*key_groups, nan_policy='omit')
        kruskal_results_df.loc[key, 'stats'] = kruskal_res[0]
        kruskal_results_df.loc[key, 'p-value'] = kruskal_res[1]
        for gc in group_comps:
            ranksum_res = ranksums(key_groups[gc[0]], key_groups[gc[1]])
            ranksum_df.loc[f' {key} stats', f'{ct_str[gc[0]]} vs {ct_str[gc[1]]}'] = ranksum_res[0]
            ranksum_df.loc[f' {key} p-value', f'{ct_str[gc[0]]} vs {ct_str[gc[1]]}'] = ranksum_res[1]

    kruskal_results_df.to_csv(f'{f_name}/kruskal_results.csv')
    ranksum_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('NGF morphology analysis is done')