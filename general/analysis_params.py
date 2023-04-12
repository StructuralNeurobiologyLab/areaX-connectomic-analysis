#this file sets general analysis values
from syconn.handler.basics import load_pkl2obj

class Analysis_Params(object):
    '''
    Config object for setting general analysis parameters
    TO DO: base on file
    '''
    def __init__(self, working_dir, version):
        self._working_dir = working_dir
        self._version = version
        ct_dict = {'v3': {}, 'v4': {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}, 'v5': {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF", 11:"ASTRO", 12:"OLIGO", 13:'MICRO', 14:'FRAG'}}
        self._ct_dict = ct_dict[version]
        if version == 'v5':
            self._glia_cts = [11, 12, 13, 14]
        else:
            self._glia_cts = []
        self._num_cts = len(self._ct_dict.keys())
        self._axoness_dict = {0: 'dendrite', 1:'axon', 2:'soma'}
        self._spiness_dict = {0: 'spine neck', 1: 'spine head', 2:'dendritic shaft', 3:'other'}
        self._axon_cts = [1, 3, 4]
        self._syn_prob_tresh = 0.8
        self._min_syn_size = 0.1
        self._min_comp_length = 200
        self.file_locations = {'v3': '/cajal/nvmescratch/users/arother/j0251v3_prep',
                               'v4': "/cajal/nvmescratch/users/arother/j0251v4_prep",
                               'v5': "/cajal/nvmescratch/users/arother/j0251v5_prep"}
        self._merger_file_location = "/cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl"
        self._pot_astros_file_location = 'cajal/nvmescratch/users/arother/j0251v4_prep/pot_astro_ids.pkl'
        self._cell_dicts_location = '/cajal/nvmescratch/users/arother/j0251v4_prep/'

    def working_dir(self):
        return self._working_dir

    def ct_dict(self, with_glia = False):
        if with_glia:
            return self._ct_dict
        else:
            ct_dict = {i: self._ct_dict[i] for i in range(self._num_cts) if i not in self._glia_cts}
            return self._ct_dict

    def ct_str(self, with_glia = False):
        #return celltype names as list of str
        if with_glia:
            ct_str = [self._ct_dict[i] for i in range(self._num_cts)]
        else:
            ct_str = [self._ct_dict[i] for i in range(self._num_cts) if i not in self._glia_cts]
        return ct_str

    def num_cts(self, with_glia = False):
        if with_glia:
            return self._num_cts
        else:
            return self._num_cts - len(self._glia_cts)

    def axoness_dict(self):
        return self.axoness_dict

    def axon_cts(self):
        return self._axon_cts

    def syn_prob_thresh(self):
        return self._syn_prob_tresh

    def min_syn_size(self):
        return self._min_syn_size

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
        