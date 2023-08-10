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
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230810_j0251v5_ngf_mito_radius_spiness_mcl%i_fs%i" % (min_comp_len, fontsize_jointplot)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('NGF seperation morphology', log_dir=f_name + '/logs/')
    log.info('Parameters to check: soma diameter, axon median radius, axon mitochondria density, axon myelin fraction, spine density')
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
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

    log.info(f'{len(ngf_ids)} are suitable for analysis')

    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    cached_mito_ids = sd_mitossv.ids
    cached_mito_mesh_bb = sd_mitossv.load_numpy_data("mesh_bb")
    cached_mito_rep_coords = sd_mitossv.load_numpy_data("rep_coord")
    cached_mito_volumes = sd_mitossv.load_numpy_data("size")


    log.info("Step 1/3: Get information from NGF")
    log.info('Get information about mitos, myelin and axon radius')
    ngf_input = [[ngf_id, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, ngf_cell_dict] for ngf_id in ngf_ids]
    ngf_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, ngf_input)
    ngf_output = np.array(ngf_output)
    #[ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
    axon_median_radius_ngf = ngf_output[:, 0]
    axon_mito_volume_density_ngf = ngf_output[:, 1]
    axon_myelin_ngf = ngf_output[:, 2]

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
                  "axon myelin fraction": axon_myelin_ngf[ngf_nonzero], 'soma diameter': ngf_diameters[ngf_nonzero],
                  'spine density': spine_density[ngf_nonzero], "cellids": ngf_ids[ngf_nonzero]}
    ngf_param_df = pd.DataFrame(ngf_params)
    ngf_param_df.to_csv("%s/ngf_params.csv" % f_name)

    time_stamps = [time.time()]
    step_idents = ["NGF parameters collected"]


    log.info("Step 2/3 Plot results")
    key_list = list(ngf_params.keys())[:-1]

    combinations = list(itertools.combinations(range(len(key_list)), 2))
    #sns.set(font_scale=1.5)
    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= ngf_params, x = x, y = y, palette='dark:black')
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot,  fill = False, alpha = 0.3,
                         kde=False, bins=10, color = 'black')
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
            g.ax_joint.set_ylabel('%s [1/µm]')

        plt.savefig("%s/%s_%s_joinplot_all.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot_all.png" % (f_name, x, y))
        plt.close()

    soma_den_thresh = 11.5
    log.info(f'Step 3/3:Seperate populations by manual threshold for soma density = {soma_den_thresh}')
    ngf_params['celltype'] = str
    ct_palette = {'type 1': '#232121', 'type 2': '#15AEAB'}
    raise ValueError
    type1_ids = ngf_params['cellids'][ngf_params['soma diameter'] < soma_den_thresh]
    type1_inds = ngf_params['soma diameter'] < soma_den_thresh
    ngf_params.loc[type1_inds, 'celltype'] = 'type 1'
    type2_ids = ngf_params['cellids'][ngf_params['soma diameter'] >= soma_den_thresh]
    type2_inds = ngf_params['soma diameter'] >= soma_den_thresh
    ngf_params.loc[type2_inds, 'celltype'] = 'type 2'
    ngf_param_df.to_csv("%s/ngf_params.csv" % f_name)
    write_obj2pkl(f'{f_name}/type1_ids.pkl', np.array(type1_ids))
    write_obj2pkl(f'{f_name}/type2_ids.pkl', np.array(type2_ids))

    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= ngf_param_df, x = x, y = y, hue = "celltype", palette = ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot,  fill = True, alpha = 0.3,
                         kde=False, bins=10, palette = ct_palette)
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
            g.ax_joint.set_ylabel('%s [1/µm]')

        plt.savefig("%s/%s_%s_joinplot.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot.png" % (f_name, x, y))
        plt.close()


    log.info('NGF morphology analysis is done')