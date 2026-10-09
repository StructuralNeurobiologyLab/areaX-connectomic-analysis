#Skeleton-based per-cell parameters of the INT1-4 full cells that are not part of the morphology table of
#ct_morph_analyses.py: axonal bouton fraction (Fig 5B, Fig S11E), output synapses per µm of axon and input synapses
#per µm of dendrite (Fig S11F, H), axonal cable per occupied 10 µm cube (Fig S11G) and the median distance of the
#dendritic skeleton nodes from the soma centre (Fig S11H)

def get_int_skeleton_params(cell_input):
    '''
    Per cell: bouton fraction of the axonal skeleton nodes (labels 3 = en-passant and 4 = terminal bouton of
    'axoness_avg10000', relative to all axon nodes 1, 3, 4), the number of occupied cubes of the axon
    (grid cubes of cube_size µm containing at least one axonal skeleton node) and the median euclidean distance of
    the dendritic skeleton nodes from the soma centre.
    :param cell_input: [cellid, soma centre in nm, cube_size in µm]
    :return: cellid, bouton fraction, number of occupied axon cubes, median dendritic distance in µm
    '''
    from syconn.reps.super_segmentation import SuperSegmentationObject
    import numpy as np
    cellid, soma_centre, cube_size = cell_input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    axoness = cell.skeleton['axoness_avg10000']
    nodes = cell.skeleton['nodes'] * cell.scaling / 1000  # µm
    axon_inds = np.in1d(axoness, [1, 3, 4])
    bouton_inds = np.in1d(axoness, [3, 4])
    bouton_fraction = np.sum(bouton_inds) / np.sum(axon_inds)
    axon_cubes = np.unique(np.floor(nodes[axon_inds] / cube_size).astype(int), axis=0)
    dendrite_nodes = nodes[axoness == 0]
    dendrite_distances = np.linalg.norm(dendrite_nodes - np.array(soma_centre) / 1000, axis=1)
    return [cellid, bouton_fraction, len(axon_cubes), np.median(dendrite_distances)]

if __name__ == '__main__':
    from bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from bio_analysis.general.analysis_params import Analysis_Params
    from bio_analysis.general.analysis_colors import CelltypeColors
    from syconn.handler.config import initialize_logging
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    version = 'v8'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    global_params.wd = bio_params.working_dir()
    min_comp_len_cell = 200
    exclude_known_mergers = True
    #edge length of the grid cubes for the axonal cable per occupied volume
    cube_size = 10  # µm
    int_cts = [9, 10, 11, 12]
    color_key = 'STNGPINTv6'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/261003_j0251{version}_INT_skeleton_features_mcl_%i_cube%i_fs%i" % (
        min_comp_len_cell, cube_size, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('INT_skeleton_features_log', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, known mergers excluded = %s, cube size = %i µm" % (
        min_comp_len_cell, exclude_known_mergers, cube_size))
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    known_mergers = bio_params.load_known_mergers()

    log.info('Step 1/3: Get suitable cellids')
    cell_inputs = []
    cell_rows = []
    for ct in int_cts:
        cell_dict = bio_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                        axon_only=False, max_path_len=None)
        log.info(f'{len(cellids)} {ct_dict[ct]} cells')
        for cellid in cellids:
            cell_inputs.append([cellid, cell_dict[cellid]['soma centre'], cube_size])
            cell_rows.append({'cellid': cellid, 'celltype': ct_dict[ct],
                              'axon length': cell_dict[cellid]['axon length'],
                              'dendrite length': cell_dict[cellid]['dendrite length'],
                              'axon synapse amount': cell_dict[cellid]['axon synapse amount'],
                              'dendrite synapse amount': cell_dict[cellid]['dendrite synapse amount']})
    param_df = pd.DataFrame(cell_rows).set_index('cellid')

    log.info('Step 2/3: Get skeleton parameters per cell')
    outputs = start_multiprocess_imap(get_int_skeleton_params, cell_inputs)
    outputs = pd.DataFrame(outputs, columns=['cellid', 'bouton fraction', 'number of occupied axon cubes', 'median dendrite distance']).set_index('cellid')
    param_df = param_df.join(outputs)
    param_df['output synapses per µm axon'] = param_df['axon synapse amount'] / param_df['axon length']
    param_df['input synapses per µm dendrite'] = param_df['dendrite synapse amount'] / param_df['dendrite length']
    param_df['axon cable per occupied cube'] = param_df['axon length'] / param_df['number of occupied axon cubes']
    param_df.to_csv(f'{f_name}/INT_skeleton_params.csv')

    log.info('Step 3/3: Plot and summarise')
    plot_params = ['bouton fraction', 'output synapses per µm axon', 'input synapses per µm dendrite',
                   'axon cable per occupied cube', 'median dendrite distance']
    log.info('Medians per cell type:\n' + param_df.groupby('celltype')[plot_params].median().round(3).to_string())
    for param in plot_params:
        sns.boxplot(data=param_df, x='celltype', y=param, palette=ct_palette)
        plt.ylabel(param, fontsize=fontsize)
        plt.xlabel('cell type', fontsize=fontsize)
        plt.savefig(f'{f_name}/{param.replace(" ", "_")}_box.png')
        plt.savefig(f'{f_name}/{param.replace(" ", "_")}_box.svg')
        plt.close()
    log.info('Analysis finished')
