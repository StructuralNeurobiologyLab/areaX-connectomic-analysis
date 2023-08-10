#this file contains different coloour palettes made by Alexandra Rother
#using either coolors.co or adobe color

import seaborn as sns

class CelltypeColors():
    '''
    Here are colour palettes made to visualize 11 different celltypes
    '''
    def __init__(self):
        self.ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
        self.num_cts = len(self.ct_dict.keys())
        #palette with dark blue, grays, black, red, last, three repeating itself
        c1 = ["#010440", "#010326", "#D98977", "#BF0404", "#D9D9D9", "#8C8C8C", "#404040", "#0D0D0D", "#010440", "#010326", "#D98977"]
        #palette with gray values, also look a bit muddy
        c2 = ['#D0CCD0', '#DBD8DC', '#E6E4E8', '#F1F0F4', '#FBFCFF', '#AEAAAB', '#878181', '#746D6C', '#6A6361', '#605856', '#443D3D']
        # values of blue/green
        c3 = ['#5D737E', '#619595', '#64B6AC', '#92DAD4', '#A9ECE8', '#C0FDFB', '#CDFEF5', '#DAFFEF', '#EBFFF6', '#FCFFFD', '#443D3D']
        # turquoise/pink, very bright
        c4 = ['#0E7C7B', '#139D9B', '#15AEAB', '#17BEBB', '#D4F4DD', '#D5C0B8', '#D58B92', '#D62246', '#912043', '#4B1D3F', '#232121']
        # blue, red, oranges
        c5 = ['#002C42', '#003049', '#1B2F45', '#362E41', '#6B2C39', '#D62828', '#E75414', '#F77F00', '#FCBF49', '#EAE2B7', '#0C0B0B']
        #celltype specific colors: highlighting STN in red, MSN in yellow, GPe in viollett, GPi in green, NGF in black, rest gray
        c6 = ['#790D18', '#707070', '#EAAE34', '#707070', '#707070', '#707070', '#592A87', '#2AC644', '#707070', '#707070', '#232121']
        #celltype specific with ngf subgroups
        c7 = ['#790D18', '#707070', '#EAAE34', '#707070', '#707070', '#707070', '#592A87', '#2AC644', '#707070',
              '#707070', '#232121', '#15AEAB']
        self.colors = {'BlRdGy': c1, 'MudGrays': c2, 'BlGrTe': c3, 'TePkBr': c4, 'BlYw': c5, 'STNGP': c6, 'NGF': c7}
        self.palettes = list(self.colors.keys())

    def ct_palette(self, key, num = False):
        '''
        Creates color palette with celltypes, either str or number as keys.
        :param key: Desired colors
        :param num: if True, use numbers as keys, else str
        :return: color palette
        '''
        if num:
            palette = {i: self.colors[key][i] for i in range(self.num_cts)}
        else:
            palette = {self.ct_dict[i]: self.colors[key][i] for i in range(self.num_cts)}
        return palette

    def ct_palette_add_groups(self, key, ct, add_groups, num=False, ):
        '''
        Creates color palette with celltypes, either str or number as keys.
        :param key: Desired colors
        :param ct: celltype groups are added to (will be counted up from there)
        :param add_groups: list of labels for new names
        :param num: if True, use numbers as keys, else str
        :return: color palette
        '''
        self.num_cts += len(add_groups) - 1
        for i,ag in enumerate(add_groups):
            self.ct_dict[ct + i] = ag
        if num:
            palette = {i: self.colors[key][i] for i in range(self.num_cts)}
        else:
            palette = {self.ct_dict[i]: self.colors[key][i] for i in range(self.num_cts)}
        return palette

class SubCT_Colors(CelltypeColors):
    '''
    For subpopulations within celltypes from CelltypeColors
    '''
    def __init__(self, subct_labels):
        super().__init__()
        self.subct_labels = subct_labels
        self.num_subct = len(subct_labels)
        #colors for MSN subpopulations:, yellow, blues
        c1 = ['#2F86A8', "#EAAE34", "#C86E29", '#707070']
        #turqoise/yellow
        c2 = ["#F0F3BD", "#028090", "#1A5E63", "#00BFB2"]
        # MudGrays as above
        c3 = ['#D0CCD0', '#DBD8DC', '#E6E4E8', '#F1F0F4']
        self.subct_colors = {'MSN': c1, 'TeYw': c2, 'MudGrays': c3}
        self.subct_palettes = self.subct_colors.keys()

    def get_subct_palette_fromct(self, ct, ct_color_key, light = False):
        ct_palette = self.ct_palette(ct_color_key)
        color = ct_palette[ct]
        if light == True:
            colors = sns.light_palette(color, n_colors=self.num_subct)
        else:
            colors = sns.dark_palette(color, n_colors=self.num_subct)
        palette = {self.subct_labels[i]: colors[i] for i in range(self.num_subct)}
        return palette

    def get_subct_palette(self, key):
        palette = {self.subct_labels[i]: self.subct_colors[key][i] for i in range(self.num_subct)}
        return palette


class CompColors():
    '''Colors to use for different compartment visualizations'''

    def __init__(self):
        self.compartments = ['dendrite', 'axon', 'soma']
        self.compartments_deso = ['soma', 'spine neck', 'spine head', 'dendritic shaft']
        self.num_comp = len(self.compartments)
        self.nump_comp_denso = len(self.compartments_deso)
        #MudGrays as above
        c1 = ['#D0CCD0', '#DBD8DC', '#E6E4E8', '#F1F0F4']
        #Gray/Greens
        c2 = ["#C5C5C5", "#4C5B61", "#829191", "#949B96"]
        #turqoise/yellow
        c3 = ["#F0F3BD", "#028090", "#1A5E63", "#00BFB2"]
        #nude/faint with rose
        c4 = ["#C2847A", "#848586", "#BAA898", "#EEE0CB"]
        #blue/gray and red (similar to c1 above)
        c5 = ["#EF233C", "#EDF2F4", "#8D99AE", "#2B2D42"]
        #different shades of turqiouse and black
        c6 = ["#232121", "#1D307D", "#2F86A8" , "#2CBCBF"]
        self.colors = {'MudGrays': c1, 'GreenGrays': c2, 'TeYw': c3, 'NeRe': c4, 'BeRd': c5, 'TeBk': c6}
        self.palettes = list(self.colors.keys())

    def comp_palette(self, key, num = False, denso = False):
        '''
        Creates color palette with celltypes, either str or number as keys.
        :param key: Desired colors
        :param num: if True, use numbers as keys, else str
        :return: color palette
        '''
        if denso:
            num_cats = self.nump_comp_denso
            cats = self.compartments_deso
        else:
            num_cats = self.num_comp
            cats = self.compartments
        if num:
            palette = {i: cats[i] for i in range(num_cats)}
        else:
            palette = {cats[i]: self.colors[key][i] for i in range(num_cats)}
        return palette



