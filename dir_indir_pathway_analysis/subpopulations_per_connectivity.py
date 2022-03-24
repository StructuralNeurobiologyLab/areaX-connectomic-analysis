# function to seperate MSN into groups based on connectivity to GPe/i
# four groups: GPe only, GPi only, GPe/i both and none
# plot amount and sum of synapses, cellids of GPe/i they synapse onto

def sort_by_connectivity(ssd, ct1, ct2, ct3, cellids1, cellids2, cellids3, full_celldict1 = None, full_celldict2 = None, full_celldict3 = None, min_comp_len = 200):
    """
    sort one celltype into 4 groups based on connectivty to two other celltypes. Groups will be only one of them, neither or both.
    Also synapse amount, sum of synaptic area and cellids of the cells they synapsed onto will be looked at.
    :param ssd: super segmentation dataset
    :param ct1: celltype to be sorted
    :param ct2, ct3: celltypes ct1 is connected to
    :param cellids1, cellids2, cellids3: cellids of corresponding celltypes, will be checked for
    minimal compartment length
    :param full_celldict1, full_celldict2, full_celledcit3: dictionaries with parameters of correpsonding celltypes
    :param min_comp_len: minimal compartment length for axon and dendrite
    :return:
    """

#iterate over cells to check their minimum compartment length
#do this via their ct dictionary
# do this for both celltypes



#take cached synapse arrays and sort them into four groups
# save cellids per group as array
# save table and dictionary which GPs they connect to, their amount and sum of synapses
# also save compartment

