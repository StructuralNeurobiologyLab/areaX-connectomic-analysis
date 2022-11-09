#this file sets general analysis values
from syconn.handler.basics import load_pkl2obj

class analysis_params(object):
    '''
    Config object for setting general analysis parameters
    TO DO: base on file
    '''
    def __init__(self, working_dir):
        self._working_dir = working_dir
        self._ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
        self._axoness_dict = {0: 'dendrite', 1:'axon', 2:'soma'}
        self._spiness_dict = {0: 'spine neck', 1: 'spine head', 2:'dendritic shaft', 3:'other'}
        self._axon_cts = [1, 3, 4]
        self._syn_prob_tresh = 0.8
        self._min_syn_size = 0.1
        self._min_comp_length = 200
        self._merger_file_location = "/cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl"
        self._pot_astros_file_location = 'cajal/nvmescratch/users/arother/j0251v4_prep/pot_astro_ids.pkl'
        self._cell_dicts_location = '/cajal/nvmescratch/users/arother/j0251v4_prep/'

    @property
    def working_dir(self):
        return self._working_dir
        
    @property
    def ct_dict(self):
        return self._ct_dict

    @property
    def axoness_dict(self):
        return self.axoness_dict

    @property
    def axon_cts(self):
        return self._axon_cts

    @property
    def syn_prob_thresh(self):
        return self._syn_prob_tresh

    @property
    def min_syn_size(self):
        return self._min_syn_size

    @property
    def min_comp_length(self):
        return self._min_comp_length

    def load_known_mergers(self):
        mergers = load_pkl2obj(self._merger_file_location)
        return mergers

    def load_potential_astros(self):
        potential_astrocytes = load_pkl2obj(self._pot_astros_file_location)
        return potential_astrocytes

    def load_cell_dict(self, celltype):
        if celltype in self._axon_cts:
            cell_dict = load_pkl2obj('%s/ax_%.3s_dict.pkl' % (self._cell_dicts_location, self._ct_dict[celltype]))
        else:
            cell_dict = load_pkl2obj('%s/full_%.3s_dict.pkl' % (self._cell_dicts_location, self._ct_dict[celltype]))
        return cell_dict
        