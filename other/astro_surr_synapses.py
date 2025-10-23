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
    offset_radius = 3000 #additional radius for first filtering

    f_name = f"cajal/scratch/users/arother/bio_analysis_results/other/241015_j0251{version}_astrocyte_surr_syns"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'astrocyte surrounding synapses',
                             log_dir=f_name)
    log.info(f"min syn size = {min_syn_size}, synapse probability = {syn_prob}, only axo-dendritic and axo-somatic synapses between neurons selected")
    log.info(f'Synapses will be extracted in the surrounding of the following cellids: {cellids}')
    log.info(f'For first round of synapse filtering synapses will be filtered by their rep_coords only with addtional {offset_radius} nm;'
             f'for more exact filtering no offest and the synapse vertices will be used.')

    log.info('Step 1/3: Get centre and maximum radius of cell to calculate sphere in which synapses should be selected')
    astro_morph_output = start_multiprocess_imap(get_cell_sphere_from_skel, cellids) #nodes and max radius in nm
    astro_morph_output = np.array(astro_morph_output, dtype='object')
    astro_sphere_df = pd.DataFrame(columns = ['cellid', 'cell center nm', 'max radius nm', 'cell center vx x', 'cell center vx y', 'cell center vx z'])
    astro_sphere_df['cellid'] = cellids
    astro_sphere_df['cell center nm'] = astro_morph_output[:, 0]
    astro_sphere_df['max radius nm'] = astro_morph_output[:, 1]
    cell_center_vx = astro_morph_output[:, 0][0] / analysis_params._voxel_size
    if len(astro_morph_output) == 1:
        astro_sphere_df['cell center vx x'] = cell_center_vx[0]
        astro_sphere_df['cell center vx y'] = cell_center_vx[1]
        astro_sphere_df['cell center vx z'] = cell_center_vx[2]
    else:
        astro_sphere_df['cell center vx x'] = cell_center_vx[:, 0]
        astro_sphere_df['cell center vx y'] = cell_center_vx[:, 1]
        astro_sphere_df['cell center vx z'] = cell_center_vx[:, 2]

    astro_sphere_df.to_csv(f'{f_name}/astro_center_radius.csv')

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
    m_sizes = m_sizes[axs_inds]
    m_rep_coord = m_rep_coord[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    m_rep_coord = m_rep_coord[den_so_inds]
    #filter for neuron_only
    neuron_cts = np.array([*ct_dict])
    neuron_inds = np.all(np.in1d(m_cts, neuron_cts).reshape(len(m_cts), 2), axis = 1)
    m_cts = m_cts[neuron_inds]
    m_ids = m_ids[neuron_inds]
    m_axs = m_axs[neuron_inds]
    m_sizes = m_sizes[neuron_inds]
    m_rep_coord = m_rep_coord[neuron_inds]
    #make sure proj axon cts are only on presynaptic site
    axon_cts = analysis_params._axon_cts
    testct = np.in1d(m_cts, axon_cts).reshape(len(m_cts), 2)
    testax = np.in1d(m_axs, 1).reshape(len(m_cts), 2)
    ax_ct_inds = np.any(testct == testax, axis=1)
    m_ids = m_ids[ax_ct_inds]
    m_sizes = m_sizes[ax_ct_inds]
    m_rep_coord = m_rep_coord[ax_ct_inds]
    total_syn_num = len(m_cts)
    log.info(f'In total {total_syn_num} synapses fulfill criteria.')

    log.info('Step 3/3: Get synapses within certain distances to cell centre')
    cell_input = [[cellids[i], astro_sphere_df.loc[i, 'cell center nm'], astro_sphere_df.loc[i, 'max radius nm'],
                   offset_radius, m_ids, m_sizes, m_rep_coord, f_name] for i in range(len(cellids))]

    #use synapse vertices instead of synapse rep coord to be more accurate;
    #load synapse vertices only in function
    #also export meshes for all synapses and save in folder specific for astrocyte
    #alternatively do 2-step process: use radius + 5 µm for rep_coordinate
    #then load vertex coordinates of all synapses and filter again for actual radius
    #use kdtree from astrocentre with radius as max distance of sphere determined in step 1
    #save synapse id, synapse size, rep coord, astrocyte id for each synapse that is selected and save in dataframe


    log.info('Analysis finished')
