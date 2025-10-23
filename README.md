# General
----

This repository contains scripts to perform analysis on volume electron microscopy datasets that were already processed with SyConn (https://github.com/StructuralNeurobiologyLab/SyConn).  
This assumes that all cells are classified into cell types, have a skeleton representation, and include predictions of which parts are axon, dendrite, or soma.  
Additionally, synapses and other organelles are mapped to the cells.  

This work was developed as part of the PhD project of **Alexandra Rother** at the **Max Planck Institute of Biological Intelligence** with the goal of analysing a connectomics dataset in songbird Area X.  
The more specific scripts in the folders `LMAN_MSN_analysis`, `TAN_DA_axo_analysis`, and `dir_indir_pathway_analysis` are therefore tailored to focus on biological questions for this dataset.  

Scripts and helper files for general use are in the `general` folder. It includes helper functions for analysis, functions to further prepare your data, and functions to get a cell type–based connectivity matrix or different organelle densities.  

---

# How to start
---

To start using it, first adapt the file `general/analysis_params.py` to fit your folder structure and cell types.  
Make sure to specify which cell types are projecting axons or glial cells, as these will be treated differently in the analysis.  
In `file_locations`, a folder should be selected to store preprocessed parameters. It will be used in all scripts and does not need to be inside the repository where you store the output of the SyConn processing.  
If you would like to use a specific color palette, this can be added to `analysis_colors`, but it also includes a few default options.  

Not all cells in a connectomic dataset are completely processed or include all three compartments: axon, dendrite, and soma.  
The analysis scripts are designed to focus on a core set of cells that are as complete as possible. To allow easier filtering for specific criteria (e.g., a minimum skeleton path length),  
a specific set of parameters is extracted from all cells before starting the analysis.  

This is done in the function `analysis_prep.py`. It will calculate parameters such as skeleton path length and mesh surface area for axon and dendrite for each cell that has at least one skeleton  
node predicted for each of those compartments. The function is designed to use multiprocessing, so multiple cores should be available.  

Each script needs specific filtering criteria for cells and synapses. Default settings are: **200 µm skeleton path length** for full cells, **50 µm** for projecting axons, **0.1 µm²** minimum synaptic area, and **0.6 synapse probability**.  

After this is done, the script `connectivity_fraction_per_ct.py` can be run and will result in connectivity matrices on a cell-type basis, as well as export bar plots of fractional in- and output for each cell type.  
The function `ct_morph_analysis.py` compares different cell types based on morphology and includes both a PCA and UMAP to cluster them. To include organelle densities in this analysis, these need to be computed first.  
To compute organelle densities for specific cell types, the function `ct_org_volume_density_cell_volume.py` can be used, which computes the volume density of a specific organelle for the whole cell in relation to the cell’s volume.  
For this, run the function `map_org2cells.py` first to prepare organelle mappings for filtered cell types.  
The functions `ct_organelle_volume_density.py` and `ct_organelle_comp_surface_area_density.py` calculate the organelle density separately in different compartments, either in relation to the skeleton or to the mesh surface area.  
To use this modality, organelles need to be mapped to the different compartments first. For synapses, this is already done in the standard SyConn pipeline. To map other organelles to specific compartments, run `map_org2axoness_fullcells.py`.  

To additionally get the vesicle density and run other scripts in the `single_vesicle_analysis` folder with individual vesicles, you need to have a centre coordinate of each vesicle ready.  
For segmentation of vesicles or other organelles, the **elektronn3** toolkit was previously used in this project (https://github.com/ELEKTRONN/elektronn3). Before running any of the scripts, first run `vesicle_prep.py`.  

---

# Important scripts and functions
---

For an overview of all cell types (see above for which preparation functions need to be run):  
- `general/connectivity_fraction_per_ct.py`: In- and outgoing connectivity per cell type; matrices summarising connectivity on the cell-type level  
- `general/ct_morph_analysis.py`: Quantifies morphological parameters of different cell types and separates them with PCA and UMAP  
- `general/ct_soma_radius.py`: Estimates the soma radii for all cell types independently of the other functions  
- `general/ct_org_volume_density_cell_volume.py`: Calculates organelle volume density of the whole cell in relation to cell volume  
- `general/ct_organelle_volume_density.py`: Calculates organelle volume density of specific compartments in relation to the skeleton path length of that compartment  
- `general/ct_organelle_comp_surface_area_density.py`: Calculates organelle surface area density of specific compartments in relation to the surface area of that compartment  

General functions that are repeatedly used in the analysis are found in the different helper files:  
- `general/analysis_conn_helper.py`: Provides functions for connectivity analysis  
- `general/analysis_morph_helper.py`: Provides functions for morphological analysis  
- `general/vesicle_helper.py`: Provides functions for analysis of individual vesicles  

Further analyses that might be useful for other connectomic datasets:  
- `dir_indir_analysis/CT_comp_conn_analysis.py`: Gives compartment-specific connectivity details of several inputs to one cell type. This includes information on whether a synapse targets the dendritic shaft, spine head, spine neck, or soma of a neuron.  
- `dir_indir_analysis/syn_conn_details.py`: Calculates the number of multi-synapses (synapses with the same pre- and postsynapse) or synapse sizes, either between two cell types or comparing in- and outputs of two cell types to a third one.  
- `dir_indir_analysis/CT_input_syn_distance_analysis.py`: Calculates the distance of synapses to the soma from several cell types to a specific one.  



