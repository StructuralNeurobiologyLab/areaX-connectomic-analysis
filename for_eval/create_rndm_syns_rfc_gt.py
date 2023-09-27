#randomly select synapses that are used for synapse_gt of the rfc
#randomly select per celltype
#save results in table with pre-post synapse and synapse coordinate

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from sklearn.utils import shuffle

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    n_ct = 15
    f_name = "cajal/scratch/users/arother/bio_analysis_results/for_eval/230927_j0251v5_ct_rndm_syn_rfc_gt_nct%i" % (n_ct)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('select random subset of single vesicles for evaluation',
                             log_dir=f_name + '/logs/')
    log.info(f'Select random synapses for rfc gt, for cts {n_ct} for incoming and outgoing each, for axon_cts {n_ct * 2}')
    log.info('Only axo-dendritic or axo-somatic synaspes, only between neurons')

    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    num_cts = len(celltypes)
    axon_cts = analysis_params.axon_cts()
    columns = ['coord x', 'coord y', 'coord z', 'cellid 1', 'cellid 2', 'real synapse', 'artifical syn ids', 'ct']
    gt_df = pd.DataFrame(columns = columns, index = range(num_cts * n_ct * 2))
    np.random.seed(42)

    log.info('Load synapse caches and filter for axo-dendritic, axo-somatic and between neurons only')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    neuron_partners = sd_synssv.load_numpy_data('neuron_partners')
    partner_axs = sd_synssv.load_numpy_data('partner_axoness')
    partner_axs[partner_axs == 3] = 1
    partner_axs[partner_axs == 4] = 1
    partner_cts = sd_synssv.load_numpy_data('partner_celltypes')
    syn_coords = sd_synssv.load_numpy_data('rep_coord')
    #filter only axo-dendritic or axo-somatic
    ax_inds =  np.any(partner_axs == 1, axis=1)
    partner_axs = partner_axs[ax_inds]
    partner_cts = partner_cts[ax_inds]
    syn_coords = syn_coords[ax_inds]
    neuron_partners = neuron_partners[ax_inds]
    den_so_inds = np.any(np.in1d(partner_axs, [0, 2]).reshape(len(partner_axs), 2), axis=1)
    partner_axs = partner_axs[den_so_inds]
    partner_cts = partner_cts[den_so_inds]
    syn_coords = syn_coords[den_so_inds]
    neuron_partners = neuron_partners[den_so_inds]
    #filter only neuron celltypes
    neuron_inds = np.all(np.in1d(partner_cts, range(num_cts)).reshape(len(partner_cts), 2), axis = 1)
    partner_axs = partner_axs[neuron_inds]
    partner_cts = partner_cts[neuron_inds]
    syn_coords = syn_coords[neuron_inds]
    neuron_partners = neuron_partners[neuron_inds]
    art_syn_ids = np.arange(0, len(syn_coords))

    log.info('Iterate over celltypes and randomly select synapses')
    log.info('Make sure that no synapse is selected two times')
    for ct in tqdm(range(num_cts)):
        #remove synapses that were selected before
        notselected_id_inds = np.in1d(art_syn_ids, gt_df['artifical syn ids']) == False
        nt_neuron_partners = neuron_partners[notselected_id_inds]
        nt_partner_axs = partner_axs[notselected_id_inds]
        nt_partner_cts = partner_cts[notselected_id_inds]
        nt_syn_coords = syn_coords[notselected_id_inds]
        nt_art_syn_ids = art_syn_ids[notselected_id_inds]
        #filter per celltype
        ct_inds = np.any(np.in1d(nt_partner_cts, ct).reshape(len(nt_partner_cts), 2), axis = 1)
        ct_neuron_partners = nt_neuron_partners[ct_inds]
        ct_partner_axs = nt_partner_axs[ct_inds]
        ct_syn_coords = nt_syn_coords[ct_inds]
        ct_partner_cts = nt_partner_cts[ct_inds]
        ct_syn_ids = nt_art_syn_ids[ct_inds]
        if ct in axon_cts:
            rndm_inds = np.random.choice(range(len(ct_syn_ids)), n_ct * 2, replace=False)
            rndm_ids = ct_syn_ids[rndm_inds]
        else:
            #filter based on where ct is axon
            testct = np.in1d(ct_partner_cts, ct).reshape(len(ct_partner_cts), 2)
            testax = np.in1d(ct_partner_axs, 1).reshape(len(ct_partner_cts), 2)
            pre_ct_inds = np.any(testct == testax, axis=1)
            pre_cellids = ct_neuron_partners[pre_ct_inds]
            pre_coords = ct_neuron_partners[pre_ct_inds]
            pre_syn_ids = ct_syn_ids[pre_ct_inds]
            pre_rndm_ids = np.random.choice(pre_syn_ids, n_ct, replace=False)
            testdeso = np.in1d(ct_partner_axs, [0, 2]).reshape(len(ct_partner_cts), 2)
            post_ct_inds = np.any(testct == testdeso, axis=1)
            post_cellids = ct_neuron_partners[post_ct_inds]
            post_coords = ct_neuron_partners[post_ct_inds]
            post_syn_ids = ct_syn_ids[post_ct_inds]
            post_rndm_ids = np.random.choice(post_syn_ids, n_ct, replace=False)
            rndm_ids = np.sort(np.hstack([pre_rndm_ids, post_rndm_ids]))
            rndm_inds = np.in1d(ct_syn_ids, rndm_ids)
        rndm_coords = ct_syn_coords[rndm_inds]
        rndm_partners = ct_neuron_partners[rndm_inds]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'coord x'] = rndm_coords[:, 0]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'coord y'] = rndm_coords[:, 1]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'coord z'] = rndm_coords[:, 2]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'cellid 1'] = rndm_partners[:, 0]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'cellid 2'] = rndm_partners[:, 1]
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'artifical syn ids'] = rndm_ids
        gt_df.loc[ct * n_ct * 2: (ct + 1) * n_ct * 2 - 1, 'ct'] = ct_dict[ct]

    assert(len(gt_df) == num_cts * n_ct * 2)
    gt_df.to_csv(f'{f_name}/synapse_rfc_gt.csv')
    gt_df = shuffle(gt_df)
    gt_df.to_csv(f'{f_name}/synapse_rfc_gt_shuffled.csv')
    log.info('Random synapses for gt selected')





