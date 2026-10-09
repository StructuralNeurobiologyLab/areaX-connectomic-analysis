#Axon median radius of the INT1-4 full cells measured on the cell mesh (Fig S11C, F) instead of the skeleton node
#diameters: at up to 400 randomly selected axon shaft nodes a cylinder is fitted to the mesh vertices around the node

def get_axon_radius_mesh(cell_input):
    '''
    Per cell: up to n_nodes random axon skeleton nodes (axoness_avg10000 == 1, bouton nodes excluded). Around each node
    the mesh vertices within radius are fitted with a local cylinder: axis = first principal component of the vertex
    cloud, radius = median distance of the vertices to the axis. Nodes whose vertex cloud is not tube-like
    (sd along the first / second principal component < min_sd_ratio, e.g. branch points or varicosities) are skipped.
    :param cell_input: [cellid, n_nodes, radius in µm, min_sd_ratio, random seed]
    :return: cellid, number of axon nodes, number of fitted nodes, median mesh radius, 25th and 75th percentile of the
    mesh radii, median skeleton radius at the fitted nodes (µm)
    '''
    from syconn.reps.super_segmentation import SuperSegmentationObject
    from scipy.spatial import cKDTree
    import numpy as np
    cellid, n_nodes, radius, min_sd_ratio, seed = cell_input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    vertices = cell.mesh[1].reshape(-1, 3) / 1000  # µm
    nodes = cell.skeleton['nodes'] * cell.scaling / 1000  # µm
    #skeleton radius as in get_compartment_radii: diameter in voxels * x voxel size / 2
    skeleton_radii = cell.skeleton['diameters'] * cell.scaling[0] / 2000
    axon_inds = np.where(cell.skeleton['axoness_avg10000'] == 1)[0]
    rng = np.random.default_rng(seed)
    picked_inds = rng.choice(axon_inds, min(n_nodes, len(axon_inds)), replace=False)
    tree = cKDTree(vertices)
    mesh_radii = []
    fitted_inds = []
    for node_ind in picked_inds:
        node_vertices = vertices[tree.query_ball_point(nodes[node_ind], radius)]
        if len(node_vertices) < 30:
            continue
        centre = node_vertices.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov((node_vertices - centre).T))
        axis = eigenvectors[:, -1]
        sds = np.sqrt(np.maximum(eigenvalues[::-1], 1e-12))
        if sds[0] / sds[1] < min_sd_ratio:
            continue
        rel_vertices = node_vertices - centre
        dist2axis = np.linalg.norm(rel_vertices - np.outer(rel_vertices @ axis, axis), axis=1)
        mesh_radii.append(np.median(dist2axis))
        fitted_inds.append(node_ind)
    if len(mesh_radii) == 0:
        return [cellid, len(axon_inds), 0, np.nan, np.nan, np.nan, np.nan]
    return [cellid, len(axon_inds), len(mesh_radii), np.median(mesh_radii), np.percentile(mesh_radii, 25),
            np.percentile(mesh_radii, 75), np.median(skeleton_radii[fitted_inds])]

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
    n_nodes = 400
    fit_radius = 0.75  # µm
    min_sd_ratio = 2.5
    seed = 0
    int_cts = [9, 10, 11, 12]
    color_key = 'STNGPINTv6'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/261003_j0251{version}_INT_axon_radius_mesh_mcl_%i_n%i_r%.2f_fs%i" % (
        min_comp_len_cell, n_nodes, fit_radius, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('INT_axon_radius_mesh_log', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i for full cells, known mergers excluded = %s, %i nodes per cell, fit radius = %.2f µm, min sd ratio = %.1f, seed = %i" % (
        min_comp_len_cell, exclude_known_mergers, n_nodes, fit_radius, min_sd_ratio, seed))
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    known_mergers = bio_params.load_known_mergers()

    log.info('Step 1/3: Get suitable cellids')
    cell_inputs = []
    cell_cts = {}
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
            cell_inputs.append([cellid, n_nodes, fit_radius, min_sd_ratio, seed])
            cell_cts[cellid] = ct_dict[ct]

    log.info('Step 2/3: Fit cylinders to the mesh around axon nodes')
    outputs = start_multiprocess_imap(get_axon_radius_mesh, cell_inputs)
    radius_df = pd.DataFrame(outputs, columns=['cellid', 'number of axon nodes', 'number of fitted nodes', 'axon median radius mesh',
                                               'axon radius mesh 25th percentile', 'axon radius mesh 75th percentile',
                                               'axon median radius skeleton at fitted nodes']).set_index('cellid')
    radius_df.insert(0, 'celltype', [cell_cts[cellid] for cellid in radius_df.index])
    radius_df.to_csv(f'{f_name}/INT_axon_radius_mesh.csv')

    log.info('Step 3/3: Plot and summarise')
    log.info('Medians per cell type:\n' + radius_df.groupby('celltype')[['number of fitted nodes', 'axon median radius mesh', 'axon median radius skeleton at fitted nodes']].median().round(4).to_string())
    sns.boxplot(data=radius_df, x='celltype', y='axon median radius mesh', palette=ct_palette)
    plt.ylabel('axon median radius (mesh) [µm]', fontsize=fontsize)
    plt.xlabel('cell type', fontsize=fontsize)
    plt.savefig(f'{f_name}/axon_median_radius_mesh_box.png')
    plt.savefig(f'{f_name}/axon_median_radius_mesh_box.svg')
    plt.close()
    log.info('Analysis finished')
