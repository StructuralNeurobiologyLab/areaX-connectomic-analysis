#get surrounding synapses from astrocyte
#first construct sphere around astrocyte centre
#second get all synapses within sphere

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_cell_sphere_from_skel
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()
    syn_prob = 0.6
    min_syn_size = 0.1
    cellids = [1156947761]

    f_name = f"cajal/scratch/users/arother/bio_analysis_results/other/241014_j0251{version}_astrocyte_surr_syns"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'astrocyte surrounding synapses',
                             log_dir=f_name)
    log.info(f"min syn size = {min_syn_size}, synapse probability = {syn_prob}, only axo-dendritic and axo-somatic synapses between neurons selected")
    log.info(f'Synapses will be extracted in the surrounding of the following cellids: {cellids}')

    log.info('Step 1/3: Get centre and maximum radius of cell to calculate sphere in which synapses should be selected')
    astro_morph_output = start_multiprocess_imap(get_cell_sphere_from_skel, cellids) #nodes and max radius
    raise ValueError
    #do this with multiprocessing for each astrocyte in final code
    #load astrocyte skeleton and get median for centre
    #get maximum distance to centre for radius of sphere

    log.info('Step 2/3: Filter synapses')
    #filter synapses for syn_prob and min_syn_size
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #filter for axo-dendritic and axo-somatic only
    axs_inds = np.any(m_axs == 1, axis=1)
    m_cts = m_cts[axs_inds]
    m_ids = m_ids[axs_inds]
    m_axs = m_axs[axs_inds]
    m_ssv_partners = m_ssv_partners[axs_inds]
    m_sizes = m_sizes[axs_inds]
    m_spiness = m_spiness[axs_inds]
    m_rep_coord = m_rep_coord[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    m_spiness = m_spiness[den_so_inds]
    m_rep_coord = m_rep_coord[den_so_inds]
    #filter for neuron_only
    neuron_cts = np.array([*ct_dict])
    neuron_inds = np.all(np.in1d(m_cts, neuron_cts).reshape(len(m_cts), 2), axis = 1)
    m_cts = m_cts[neuron_inds]
    m_ids = m_ids[neuron_inds]
    m_axs = m_axs[neuron_inds]
    m_ssv_partners = m_ssv_partners[neuron_inds]
    m_sizes = m_sizes[neuron_inds]
    m_spiness = m_spiness[neuron_inds]
    m_rep_coord = m_rep_coord[neuron_inds]
    #make sure proj axon cts are only on presynaptic site
    axon_cts = analysis_params._axon_cts
    testct = np.in1d(m_cts, axon_cts).reshape(len(m_cts), 2)
    testax = np.in1d(m_axs, 1).reshape(len(m_cts), 2)
    ax_ct_inds = np.any(testct == testax, axis=1)
    m_cts = m_cts[ax_ct_inds]
    m_ids = m_ids[ax_ct_inds]
    m_axs = m_axs[ax_ct_inds]
    m_ssv_partners = m_ssv_partners[ax_ct_inds]
    m_sizes = m_sizes[ax_ct_inds]
    m_spiness = m_spiness[ax_ct_inds]
    m_rep_coord = m_rep_coord[ax_ct_inds]
    total_syn_num = len(m_cts)
    log.info(f'In total {total_syn_num} synapses fulfill criteria.')

    log.info('Step 3/3: Get synapses within certain distances to cell centre')
    #use synapse vertices instead of synapse rep coord to be more accurate;
    #load synapse vertices only in function
    #also export meshes for all synapses and save in folder specific for astrocyte
    #alternatively do 2-step process: use radius + 5 µm for rep_coordinate
    #then load vertex coordinates of all synapses and filter again for actual radius
    #use kdtree from astrocentre with radius as max distance of sphere determined in step 1
    #save synapse id, synapse size, rep coord, astrocyte id for each synapse that is selected and save in dataframe


    log.info('Analysis finished')
