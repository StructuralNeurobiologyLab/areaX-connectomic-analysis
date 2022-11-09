#this file contains different coloour palettes made by Alexandra Rother
#using either coolors.co or adobe color

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
        #celltype specific colors
        c6 = []
        self.colors = {'BlRdGy': c1, 'MudGrays': c2, 'BlGrTe': c3, 'TePkBr': c4, 'BlYw': c5}
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

