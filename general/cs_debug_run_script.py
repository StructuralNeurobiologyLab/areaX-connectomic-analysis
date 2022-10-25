import numpy as np
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset, SegmentationObject
from syconn.handler.config import initialize_logging
from syconn.extraction.cs_processing_steps import combine_and_split_cs


global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
sd_cs_ssv = SegmentationDataset("cs_ssv", working_dir=global_params.config.working_dir)
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
f_name = "cajal/scratch/users/arother/cs_debugging/221025_excl_by_size_nomem_newwd/"
log = initialize_logging('221025 cs run', log_dir=f_name + '/logs/')
log.info("10000 workers, 1 CPU per task, memory per default, excluded cs ids with size larger than 10**6, no memory tracking")
log.info("loading the filtered cs")
filtered_cs = load_pkl2obj("cajal/nvmescratch/users/arother/cs_debugging/filtered_cs.pkl")

log.info("generating attr_dicts for cs_ssv")
combine_and_split_cs(wd = global_params.wd, ssd_version = ssd.version,
                     cs_version = sd_cs_ssv.version, n_folders_fs=10000, overwrite = True,
                     rel_ssv_with_cs_ids = filtered_cs, save_dir= f_name)