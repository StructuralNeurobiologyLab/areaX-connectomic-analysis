#create reverse mapping dictionary for cellids and single vesicles

if __name__ == '__main__':
    from syconn import global_params
    import numpy as np
    from syconn.handler.config import initialize_logging
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from tqdm import tqdm
    from syconn.handler.basics import write_obj2pkl

    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    ves_wd = f'{global_params.wd}/single_vesicles/'

    version = 'v6'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    f_name = analysis_params.file_locations
    log = initialize_logging('reverse map single vesicles to cellids',
                             log_dir=f_name + '/logs/')
    log.info('Start loading vesicle arrays')
    log.info(f'wd = {ves_wd}')
    log.info('Load single vesicle data')
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')

    log.info('Get all cellids in the dataset from SegmentationDataset')
    ssd = SuperSegmentationDataset()
    ssv_ids = ssd.ssv_ids

    log.info('Sort arrays according to ssv_mapping')
    sorted_inds = np.argsort(ves_map2ssvids)
    sorted_mapped_ssv_ids = ves_map2ssvids[sorted_inds]
    sorted_coords = single_ves_coords[sorted_inds]
    sorted_ids = single_ves_ids[sorted_inds]

    log.info('Split coords and ids based on mapping')
    mapped_ids, inds = np.unique(sorted_mapped_ssv_ids, return_inverse=True)
    splits = np.cumsum(np.bincount(inds))[:-1]
    split_coordinates = np.split(sorted_coords, splits)
    split_ids = np.split(sorted_ids, splits)
    log.info('Create mapping dictionary with cellids as key and mapped ids and coords as entries')

    rev_mapping_coords = dict(zip(mapped_ids, split_coordinates))
    rev_mapping_ves_ids = dict(zip(mapped_ids, split_ids))

    log.info('Add cellids without any vesicles')
    non_ves_ssv_ids = ssv_ids[np.in1d(ssv_ids, mapped_ids) == False]
    for ssv_id in tqdm(non_ves_ssv_ids):
        rev_mapping_coords[ssv_id] = np.array([], dtype = np.uint16)
        rev_mapping_ves_ids[ssv_id] = np.array([], dtype = np.int64)

    write_obj2pkl(f'{ves_wd}/rev_mapping_dict_coords.pkl', rev_mapping_coords)
    write_obj2pkl(f'{ves_wd}/rev_mapping_dict_ids.pkl', rev_mapping_ves_ids)

    log.info('Analysis done')
