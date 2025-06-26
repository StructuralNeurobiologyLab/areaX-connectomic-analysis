General
----

This repository contains scripts to perform analysis on volume electron microscopy datasets that were already processes with SyConn (https://github.com/StructuralNeurobiologyLab/SyConn). 
This assumes that all cells are classified into cell types, they have a skeleton representation and a prediction of what part is axon, dendrite or soma. 
Additionally, synapses and other organelles are mapped to the cells. 

This work was developed as part of the PhD project of Alexandra Rother at the Max-Planck-Institue of Biological Intelligence with the goal of analysing a connectomics dataset in songbird Area X. 
More specific scripts in the folders LMAN_MSN_analysis, TAN_DA_axo_analysis, dir_indir_pathway_analysis are therefore tailored to focus on biological questions for this dataset. 

Scripts and helper files for general use are in the 'general' folder. It includes helper functions for analysis, functions to further prepare your data and functions to get a cell type 
based connectivity matrix or different organelle densities.

How to start
---

To start using it, first adapt the file general/analysis_params.py to fit your folder structure and cell types. 
Make sure to specify which cell types are projecting axons or glia cells as these will be treated differently in analysis.
In file_locations a folder should be selected to store preprocessed parameters. It will be used in all scripts and does not need to be inside the repository where you store the output of the SyConn processing.
If you like to use a specific color palette, this can be added to analysis_colors but it also includes a few default options. 

Not all cells in a connectomic datatset are completely processed or include all three compartments, axon, dendrite and soma. 
The analysis scripts are designed to focus on a core set of cells that are as complete as possible. To allow easier filtering for specific criteria e.g. a minimum skeleton pathlength, 
a specific set of parameters is extracted from all cells before starting the analysis. 

This is done in the function analysis_prep.py. It will calculate parameters such as skeleton pathlength and mesh surface area for axon and dendrite for each cell that has at least one skeleton 
node predicted for each of those compartments. The function is designed to use multiprocessing, so multiple cores should be available. 

Each script needs specific filtering criteria for cells and synapses. Defualt settings are: 200 µm skeleton pathlength for full cells, 50 µm for projecting axons, 0.1 µm² minimum synaptic area and 0.6 synapse probability. 

After this is done the script 'connectivity_fraction_per_ct.py' can be run and will result in connectivity matrices on a cell type basis but also export bar plots of fractional in-and output for each cell type. 
The function 'ct_morph_analysis.py' compares different cell types based on morphology and includes both a PCA and UMAP to cluster them. To include organelle densities in this analysis, these need to be computed first.
'ct_soma_radius.py' estimates the soma radii for all cell types independently of the other functions.
To compute organelle_densities for specific cell types, the function 'ct_org_volume_density_cell_volume.py' can be used which computes the volume density of a specific organelle for the whole cell in relation to the cells volume. 
For this, run the function 'map_org2cells.py' first to have organelle mappings prepared for filtered cell types. 
The functions 'ct_organell_volume_density' and 'ct_organell_comp_surface_area_density' calculate the organelle density separately in different compartments either in relation to the skeleton or to the mesh surface area. 
To use this modality, organelles need to be first mapped to the different compartments. For synapses, this is already done in the standart SyConn pipeline. To map the other organelles to specific compartments run 'map_org2axoness_fullcells.py'

To additionally get the vesicle density and run other scripts in the 'single_vesicle_analysis' folder with individual vesicles you need to have a centre coordinate of each vesicle ready. 
For segmentation of vesicles or other organelles the elektronn3 toolkit was used in this project prviously (https://github.com/ELEKTRONN/elektronn3). Before running any of the scripts first run 'vesicle_prep.py'.

Important functions
---

General functions that are repeatetly used in the analysis are found in the different helper files: 
\'analysis_conn_helper': provides functions for connectivity analysis
\'analysis_morph_helper': provides functions for morphological analysis
\'vesicle_helper': provides functions for analysis of individual vesicles

Further analysis that might useful for other connectomic datasets: 
\'dir_indir_analysis/CT_comp_conn_analysis.py': Gives compartment specific conectivity details of several inputs to one cell type. This includes information of a synapse targets the dendritic shaft, spine head, spine neck or soma of a neuron. 
\'dir_indir_analysis/syn_conn_details.py': Calculates number of multi-synapses (synapses with same pre-and postsynapse) or synapse sizes either between two cell types or compares in-and outputs of two cell type to a third one. 
\'dir_indir_analysis/CT_input_syn_distance_analysis.py': Calcultes the distance of synapses to the soma from several cell types to a specific one. 



