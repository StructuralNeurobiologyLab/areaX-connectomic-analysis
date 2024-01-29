#find syn_ids and syn_coords between two example cellids

from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
import pandas as pd
import numpy as np
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general
from syconn.handler.config import initialize_logging
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
from syconn.proc.meshes import write_meshes2kzip

global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'

cellid1 = 542544908
cellid2 = 32356701
version = 'v6'
bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
ct_dict = bio_params.ct_dict()
cell1 = SuperSegmentationObject(cellid1)
cell2 = SuperSegmentationObject(cellid2)
cell1.load_attr_dict()
cell2.load_attr_dict()
ct1 = ct_dict[cell1.attr_dict['celltype_pts_e3']]
ct2 = ct_dict[cell2.attr_dict['celltype_pts_e3']]
syn_prob_tresh = 0.6
min_syn_size = 0.1
dataset_scaling = [10, 10, 25]
blender_scaling = 10**(-5)

f_name = f'cajal/scratch/users/arother/240129_example_conns_syns/{ct1}_{cellid1}_{ct2}_{cellid2}_example_syns'
log = initialize_logging(f'example cells {ct1} {cellid1}, {ct2} {cellid2} syns', log_dir=f_name + '/logs/')
log.info(f'min syn size = {min_syn_size} µm², syn prob thresh = {syn_prob_tresh},'
         f'dataset scaling = {dataset_scaling}, scaling for blender = {blender_scaling}')

log.info('Get synapse ids and coordinates between two cells')

sd_synssv = SegmentationDataset('syn_ssv')
m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_tresh,
        min_syn_size=min_syn_size)

#filter for cellid1

id1_inds = np.any(np.in1d(m_ssv_partners, cellid1).reshape(len(m_ssv_partners), 2), axis = 1)
id1_cts = m_cts[id1_inds]
id1_ssv_partners = m_ssv_partners[id1_inds]
id1_sizes = m_sizes[id1_inds]
id1_coords = m_rep_coord[id1_inds]
id1_ids = m_ids[id1_inds]
id1_axs = m_axs[id1_inds]
id1_cts = m_cts[id1_inds]

id1_id2_inds = np.any(np.in1d(id1_ssv_partners, cellid2).reshape(len(id1_ssv_partners), 2), axis = 1)
id1_id2_cts = id1_cts[id1_id2_inds]
id1_id2_ssv_partners = id1_ssv_partners[id1_id2_inds]
id1_id2_sizes = id1_sizes[id1_id2_inds]
id1_id2_coords = id1_coords[id1_id2_inds]
id1_id2_ids = id1_ids[id1_id2_inds]
id1_id2_axs = id1_axs[id1_id2_inds]
id1_id2_cts = id1_cts[id1_id2_inds]

if len(id1_id2_sizes) == 0:
    log.info('No synapses between the two cells detected')
    raise ValueError
assert(len(np.unique(id1_id2_ssv_partners)) == 2)

log.info(f'The number of synapses = {len(id1_id2_sizes)}, with a total sum of {np.sum(id1_id2_sizes)} µm² and a mean size of {np.mean(id1_id2_sizes)} µm²')

#write coords, syn_ids, cellid 1, cellid2, axoness 1, axonees 2, syn size, coords for blender, from ct1 into pd Dataframe

columns = ['syn id', 'cellid 1', 'cellid 2', 'syn size', 'axoness 1', 'axoness 2', f'from {ct1}', 'coord x', 'coord y', 'coord z', 'coord x b', 'coord y b', 'coord z b']
conn_df = pd.DataFrame(columns=columns, index = range(len(id1_id2_sizes)))
conn_df['syn id'] = id1_id2_ids
conn_df['cellid 1'] = id1_id2_ssv_partners[:, 0]
conn_df['cellid 2'] = id1_id2_ssv_partners[:, 1]
conn_df['syn size'] = id1_id2_sizes
conn_df['axoness 1'] = id1_id2_axs[:, 0]
conn_df['axoness 2'] = id1_id2_axs[:, 1]
conn_df['coord x'] = id1_id2_coords[:, 0]
conn_df['coord y'] = id1_id2_coords[:, 1]
conn_df['coord z'] = id1_id2_coords[:, 2]
#get coords for blender when loading with blender scaling
conn_df['coord x b'] = id1_id2_coords[:, 0] * dataset_scaling[0] * blender_scaling
conn_df['coord y b'] = id1_id2_coords[:, 1]* dataset_scaling[1] * blender_scaling
conn_df['coord z b'] = id1_id2_coords[:, 2]* dataset_scaling[2] * blender_scaling
testct = np.in1d(id1_id2_axs, cellid1).reshape(len(id1_id2_axs), 2)
testax = np.in1d(id1_id2_axs, 1).reshape(len(id1_id2_axs), 2)
pre_ct_inds = np.any(testct == testax, axis = 1)
conn_df[f'from {ct1}'] = 'No'
conn_df.loc[pre_ct_inds, f'from {ct1}'] = 'Yes'
conn_df.to_csv(f'{f_name}/conn_coords.csv')

log.info('Get meshes for synapses')
syn_color_rgba = np.array([189, 195, 199, 1])
cell1_syns = np.array(cell1.syn_ssv)
cell1_syn_ids = [obj.id for obj in cell1_syns]
conn_syns = cell1_syns[np.in1d(cell1_syn_ids, id1_id2_ids)]
syn_inds = []
syn_verts = []
syn_norms = []
syn_cols = []
for syn in conn_syns:
    indices, vertices, normals = syn.mesh
    syn_inds.append(indices.astype(np.float32))
    syn_verts.append(vertices.astype(np.float32))
    syn_norms.append(normals.astype(np.float32))
    syn_cols.append(syn_color_rgba)
kzip_out = f'{f_name}/{cellid1}_{ct1}_{cellid2}_{ct2}_syn_mesh'
ply_f_names = [f'{cellid1}_{ct1}_{cellid2}_{ct2}_mito_{i}.ply' for i in range(len(syn_inds))]
write_meshes2kzip(kzip_out, syn_inds, syn_verts, syn_norms, syn_cols,
                  ply_f_names)

log.info('Analysis done')


