#TO DO: write object for results dictionary to standardize plotting of results
#put all relevant function here in this helper file
#make possibility for 2 or more different color schemmes
#boxplot, violinplot, histogram (normed, not normed), scatterplot
#also add flowchart, networx plot
#one for single dictionary and one for comparing results from two dictionaries

import seaborn as sns
import matplotlib.pyplot as plt

class ResultsForPlotting():
    """
    this class contains a dictionary with different results.
    """

    def __init__(self, celltype, filename,dictionary):
        if type(dictionary) != dict:
            raise ValueError("must be dictionary")
        self.celltype = celltype
        self.filename = filename
        self.dictionary = dictionary

    def param_label(self, key, subcell):
        """
        specifies the label for x or y-axis based on the key and subcellular compartment.
        :param key: parameter as key in dictionary
        :param subcell: subcellular compartment
        :return: None
        """
        if "density" in key:
            if "amount" in key:
                self.param_label = "%s density per µm" % subcell
            elif "volume" in key:
                self.param_label = "%s volume density [µm³/µm]" % subcell
            elif "size" in key and subcell == "synapse":
                self.param_label = "%s size density [µm²/µm]" % subcell
            elif "length" in key:
                self.param_label = "%s length density [µm/µm]" % subcell
        else:
            if "amount" in key:
                if "percentage" in key:
                    self.param_label = "percentage of %ss" % subcell
                else:
                    self.param_label = "amount of %ss" % subcell
            elif "size" in key:
                if "percentage" in key:
                    self.param_label = "percentage of %s size" % subcell
                else:
                    if subcell == "synapse":
                        self.param_label = "average %s size [µm²]" % subcell
            elif "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    self.param_label = "% of whole dataset"
                else:
                    self.param_label = " %s volume in µm³" % subcell
            elif "distance" in key:
                self.param_label = "distance in µm"
            elif "median radius" in key:
                self.param_label = "median radius in µm"
            elif "tortuosity" in key:
                self.param_label = "%s tortuosity" % subcell
            else:
                raise ValueError("unknown key description")

    def plot_hist(self, key, subcell, cells = True, color = "steelblue", norm_hist = False, bins = None, xlabel = None, celltype2 = None, outgoing = False):
        """
        plots array given with key in histogram plot
        :param key: key of dictionary that should be plotted
        :param subcell: compartment or subcellular structure that will be plotted
        :param cells: True: cells are plotted, False: subcellular structures are plotted
        :param color: color for plotting
        :param norm_hist: if true: histogram will be normed
        :param bins: amount of bins
        :param xlabel: label of x axis, not needed if clear from key
        :param celltype2: second celltype, if connectivty towards another celltype is tested
        :param outgoing: if connectivity is analysed, if True then self.celltype is presynaptic
        :return:
        """
        if norm_hist:
            sns.distplot(self.dictionary[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": color},
                         kde=False, bins=bins, norm_hist=True)
            if cells:
                plt.ylabel("fraction of cells")
            else:
                plt.ylabel("fraction of %s" % subcell)
        else:
            sns.distplot(self.dictionary[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": color},
                         kde=False, bins=bins)
            if cells:
                plt.ylabel("count of cells")
            else:
                plt.ylabel("count of %s" % subcell)
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(self.param_label(key, subcell))
        if celltype2:
            if outgoing:
                plt.title("%s from %s to %s" % (key, self.celltype, celltype2))
                if norm_hist:
                    plt.savefig("%s/%s_%s2%s_hist_norm.png" % (self.filename, key, self.celltype, celltype2))
                else:
                    plt.savefig("%s/%s_%s2%s_hist.png" % (self.filename, key, self.celltype, celltype2))
            else:
               plt.title("%s from %s to %s" % (key, celltype2, self.celltype))
               if norm_hist:
                   plt.savefig("%s/%s_%s2%s_hist_norm.png" % (self.filename, key, celltype2, self.celltype))
               else:
                   plt.savefig("%s/%s_%s2%s_hist.png" % (self.filename, key, celltype2, self.celltype))
        else:
           plt.title("%s in %s %s" % (key, self.celltype, subcell))
           if norm_hist:
               plt.savefig("%s/%s%_s_%s_hist_norm.png" % (self.filename, subcell, key, self.celltype))
           else:
               plt.savefig("%s/%s_%s_%s_hist.png" % (self.filename, subcell, key, self.celltype))
        plt.close()

class ComparingResultsForPLotting(ResultsForPlotting):
    """
    makes plots from two dictionaries with the same keys to compare their results.
    """
    def __init__(self, celltype1, celltype2, filename, dictionary1, dictionary2, color1 = "mediumorchid", color2 = "springgreen"):
        super().__init__(celltype1, filename, dictionary1)
        self.celltype1 = celltype1
        self.celltype2 = celltype2
        self.dictionary1 = dictionary1
        self.dictionary2 = dictionary2
        self.color1 = color1
        self.color2 = color2

    def color_palette(self, key1, key2):
        self.color_palette = {key1: self.color1, key2: self.color2}

    def plot_hist_comparison(self, key, subcell, cells = True, norm_hist = False, bins = None, xlabel = None, conn_celltype = None, outgoing = False):
        """
                 plots two arrays and compares them in histogram and saves it.
                 :param key: key of dictionary that should be plotted
                 :param subcell: compartment or subcellular structure that will be plotted
                 :param cells: True: cells are plotted, False: subcellular structures are plotted
                 :param norm_hist: if true: histogram will be normed
                 :param bins: amount of bins
                 :param xlabel: label of x axis, not needed if clear from key
                 :param conn_celltype: third celltype if connectivity to other celltype is tested
                 :param outgoing: if connectivity is analysed, if True then self.celltype1 and self.celltype2 are presynaptic
                 :return: None
                 """
        if norm_hist:
            sns.distplot(self.dictionary1[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color1},
                         kde=False, bins=bins, label=self.celltype1, norm_hist=True)
            sns.distplot(self.dictionary2[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color2},
                         kde=False, bins=bins, label=self.celltype2, norm_hist=True)
            if cells:
                plt.ylabel("fraction of cells")
            else:
                plt.ylabel("fraction of %s" % subcell)
        else:
            sns.distplot(self.dictionary1[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color1},
                         kde=False, bins=bins, label = self.celltype1)
            sns.distplot(self.dictionary2[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color2},
                         kde=False, bins=bins, label=self.celltype2)
            if cells:
                plt.ylabel("count of cells")
            else:
                plt.ylabel("count of %s" % subcell)
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(self.param_label(key, subcell))
        if conn_celltype:
            if outgoing:
                plt.title("%s from %s, %s to %s" % (key, self.celltype1, self.celltype2, conn_celltype))
                if norm_hist:
                    plt.savefig("%s/%s_%s_%s2%s_hist_norm.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
                else:
                    plt.savefig("%s/%s_%s_%s2%s_hist.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title("%s from %s to %s, %s" % (key, conn_celltype, self.celltype1, self.celltype2))
                if norm_hist:
                    plt.savefig("%s/%s_%s2%s_%s_hist_norm.png" % (self.filename, key, conn_celltype, self.celltype1, self.celltype2))
                else:
                    plt.savefig("%s/%s_%s2%s_%s_hist.png" % (self.filename, key, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title("%s in %s, %s" % (key, self.celltype1, self.celltype2))
            if norm_hist:
                plt.savefig("%s/%s_%s_%s_hist_norm.png" % (self.filename, key, self.celltype1, self.celltype2))
            else:
                plt.savefig("%s/%s_%s_%s_hist.png" % (self.filename, key, self.celltype1, self.celltype2))
        plt.close()





