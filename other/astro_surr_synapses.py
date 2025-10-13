#get surrounding synapses from astrocyte
#first construct sphere around astrocyte centre
#second get all synapses within sphere

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_spine_density, get_cell_soma_radius, \
        get_dendrite_info_cell, check_cutoff_dendrites
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize, filter_contact_sites_axoness
    from syconn.handler.config import initialize_logging
    from syconn import global_params
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
    cellids = []

    f_name = f"cajal/scratch/users/arother/bio_analysis_results/other/241025_j0251{version}_astrocyte_surr_syns"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'astrocyte surrounding synapses',
                             log_dir=f_name)
    log.info(f"min syn size = {min_syn_size}, synapse probability = {syn_prob}, only axo-dendritic and axo-somatic synapses between neurons selected")
    log.info(f'Synapses will be extracted in the surrounding of the following cellids: {cellids}')

    log.info('Step 1/3: Get centre and maximum radius of cell to calculate sphere in which synapses should be selected')
    #do this with multiprocessing for each astrocyte in final code
    #load astrocyte skeleton and get median for centre
    #get maximum distance to centre for radius of sphere

    log.info('Step 2/3: Filter synapses')


    log.info('Step 3/3: Get synapses within certain distances to cell centre')
    #use synapse vertices instead of synapse rep coord to be more accurate;
    #load synapse vertices only in function
    #also export meshes for all synapses and save in folder specific for astrocyte
    #alternatively do 2-step process: use radius + 5 µm for rep_coordinate
    #then load vertex coordinates of all synapses and filter again for actual radius
    #use kdtree from astrocentre with radius as max distance of sphere determined in step 1
    #save synapse id, synapse size, rep coord, astrocyte id for each synapse that is selected and save in dataframe


    log.info('Analysis finished')
