#get all HVC, LMAN synapses onto MSN cells
#make table with cellid pre and post, post synaptic compartment, synapse size
#cell type pre and post, location of synapse relative to soma

from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_syn_location_per_cell
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_histogram_selection
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationDataset
import os as os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import ranksums


if __name__ == '__main__':

    version = 'v6'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict()
    global_params.wd = bio_params.working_dir()
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    min_comp_len = 200
    min_comp_len_ax = 50
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'MSN','TeYw','MudGrays'}
    color_key = 'STNGPINTv6'
    ct1 = 1
    ct2 = 2
    conn_ct = 3
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    conn_ct_str = ct_dict[conn_ct]
    axon_cts = bio_params.axon_cts()
    comp_cts = [ct1, ct2, conn_ct]
    fontsize = 20
    if np.any(np.in1d(comp_cts, axon_cts)):
        axon_ct_present = True
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'250429_j0251{version}_{ct1_str}_{ct2_str}_2_{conn_ct_str}_syn_locations_{min_comp_len}_{min_comp_len_ax}_{color_key}'
    else:
        axon_ct_present = False
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'250429_j0251{version}_{ct1_str}_{ct2_str}_2_{conn_ct_str}_syn_locations_{min_comp_len}_{color_key}'

    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'syn_locations_{ct1_str}_{ct2_str}_2_{conn_ct_str}', log_dir=f_name + '/logs/')
    if axon_ct_present:
        log.info(
            f'min comp len cell = {min_comp_len}, min comp len ax = {min_comp_len_ax}, exclude known mergers = {exclude_known_mergers}, '
            f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}')
    else:
        log.info(
            f'min comp len cell = {min_comp_len}, exclude known mergers = {exclude_known_mergers}, '
            f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}')




    log.info("Step 1/4: Load celltypes and check suitability")
    cts_str_analysis = [ct_dict[i] for i in comp_cts]
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in tqdm(comp_cts):
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        if ct in axon_cts:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict,
                                                    min_comp_len=min_comp_len_ax,
                                                    axon_only=True,
                                                    max_path_len=None)
        else:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    number_ids = [len(suitable_ids_dict[ct]) for ct in comp_cts]
    all_suitable_ids = np.hstack(all_suitable_ids)
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")

    log.info('Step 2/4: Filter synapses with syn prob, min syn size and suitable cellids')
    #only synapses from ct1 or ct2 to conn_ct
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct1, ct2],
                                                                                                        post_cts=[conn_ct],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=None,
                                                                                                        sd_synssv = sd_synssv)
    # prefilter so that all synapses are between suitable ids
    suit_ids_ind = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ids_ind]
    m_ids = m_ids[suit_ids_ind]
    m_sizes = m_sizes[suit_ids_ind]
    m_axs = m_axs[suit_ids_ind]
    m_rep_coord = m_rep_coord[suit_ids_ind]
    m_spiness = m_spiness[suit_ids_ind]
    m_cts = m_cts[suit_ids_ind]
    log.info(f'In total there are {len(m_sizes)} synapses from either {ct1_str} or {ct2_str} to {conn_ct_str}.')

    log.info('Step 3/4: Get synapse location in relation to soma')
    #for each conn_ct cellid, a dataframe will be created with information about synapse size, pre- and post cellids,
    #pre- and post celltypes, postsynaptic compartment, synapse location relative to soma
    spiness_dict = bio_params._spiness_dict
    syn_input = [[cellid, m_ssv_partners, m_sizes, m_rep_coord, m_cts, m_axs, m_spiness, spiness_dict, ct_dict] for cellid in suitable_ids_dict[conn_ct]]
    syn_output = start_multiprocess_imap(get_syn_location_per_cell, syn_input)
    #put outputs together in one big dataframe
    result_df = pd.concat(syn_output, ignore_index = True)
    result_df.to_csv(f'{f_name}/{ct1_str}_{ct2_str}_2_{conn_ct_str}_syn_loc_results.csv')

    log.info('Step 4/4: Plot histograms of syn locations')
    #histogram plots of synapse location
    ct_palette = {ct1_str: 'black', ct2_str: 'gray'}
    plot_histogram_selection(dataframe=result_df, x_data='syn dist 2 soma', color_palette=ct_palette,
                             label=f'syn_dist2soma', count='synapses', foldername=f_name,
                             hue_data='pre cell type',
                             title=f'Synapse distances to soma on {conn_ct_str} cells in µm',
                             fontsize=fontsize)

    pre_ct_groups = result_df.groupby('pre cell type')
    #get statistics on synapse size
    stats, p_value = ranksums(pre_ct_groups.get_group(ct1_str)['syn area'], pre_ct_groups.get_group(ct2_str)['syn area'])
    log.info(f'Results of the wilcoxon ranksums test on synapse size of {ct1_str} vs {ct2_str} as pre synapse: stats = {stats}, p-value = {p_value}')

    #get statistics on distance to soma of synapses
    stats, p_value = ranksums(pre_ct_groups.get_group(ct1_str)['syn dist 2 soma'],
                              pre_ct_groups.get_group(ct2_str)['syn dist 2 soma'])
    log.info(f'Results of the wilcoxon ranksums test on synapse distance to soma of {ct1_str} vs {ct2_str} as pre synapse: stats = {stats}, p-value = {p_value}')

    log.info('Analysis done.')


