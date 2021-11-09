#TO DO: write object for results dictionary to standardize plotting of results
#put all relevant function here in this helper file
#make possibility for 2 or more different color schemmes
#boxplot, violinplot, histogram (normed, not normed), scatterplot
#also add flowchart, networx plot
#one for single dictionary and one for comparing results from two dictionaries

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        :return: param_label
        """
        if "density" in key:
            if "amount" in key:
                param_label = "%s density per µm" % subcell
            elif "volume" in key:
                param_label = "%s volume density [µm³/µm]" % subcell
            elif "size" in key and subcell == "synapse":
                param_label = "%s size density [µm²/µm]" % subcell
            elif "length" in key:
                param_label = "%s length density [µm/µm]" % subcell
        else:
            if "amount" in key:
                if "percentage" in key:
                    param_label = "percentage of %ss" % subcell
                else:
                    param_label = "amount of %ss" % subcell
            elif "size" in key:
                if "percentage" in key:
                    param_label = "percentage of %s size" % subcell
                else:
                    if subcell == "synapse":
                        param_label = "average %s size [µm²]" % subcell
            elif "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    param_label = "% of whole dataset"
                else:
                    param_label = " %s volume in µm³" % subcell
            elif "distance" in key:
                param_label = "distance in µm"
            elif "median radius" in key:
                param_label = "median radius in µm"
            elif "tortuosity" in key:
                param_label = "%s tortuosity" % subcell
            else:
                raise ValueError("unknown key description")
                param_label = 0
            return param_label

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

        def multiple_param_label(labels, ticks):
            self.m_labels = labels
            self.m_ticks = ticks

        def plot_violin_params(key, param_list,subcell, xlabel = None, ticks = None, stripplot = True, celltype2 = None, outgoing = False):
            sns.violinplot(data=param_list, inner="box")
            if stripplot:
                sns.stripplot(data=param_list, color="black", alpha=0.2)
            if xlabel:
                if ticks is None:
                    raise ValueError("need labels and ticks")
                plt.xticks(ticks = ticks, labels= xlabel)
            else:
                plt.xticks(ticks=self.m_ticks, labels=self.m_labels)
            plt.ylabel(self.param_label(key, subcell))
            if celltype2:
                if outgoing:
                    plt.title("%s from %s to %s" % (key, self.celltype, celltype2))
                    plt.savefig("%s/%s_%s_2_%s_violin.png" % (self.filename, key, self.celltype, celltype2))
                else:
                    plt.title("%s from %s to %s" % (key, celltype2, self.celltype))
                    plt.savefig("%s/%s_%s_2_%s_violin.png" % (self.filename, key, celltype2, self.celltype))
            else:
                plt.title("%s in %s %s" % (key, self.celltype, subcell))
                plt.savefig("%s/%s_%s_%s_violin.png" % (self.filename, key, subcell, self.celltype))
            plt.close()

        def plot_box_params(key, param_list,subcell, xlabel = None, ticks = None, stripplot = True, celltype2 = None, outgoing = False):
            sns.violinplot(data=param_list, inner="box")
            if stripplot:
                sns.stripplot(data=param_list, color="black", alpha=0.2)
            if xlabel:
                if ticks is None:
                    raise ValueError("need labels and ticks")
                plt.xticks(ticks = ticks, labels= xlabel)
            else:
                plt.xticks(ticks=self.m_ticks, labels=self.m_labels)
            plt.ylabel(self.param_label(key, subcell))
            if celltype2:
                if outgoing:
                    plt.title("%s from %s to %s" % (key, self.celltype, celltype2))
                    plt.savefig("%s/%s_%s_2_%s_violin.png" % (self.filename, key, self.celltype, celltype2))
                else:
                    plt.title("%s from %s to %s" % (key, celltype2, self.celltype))
                    plt.savefig("%s/%s_%s_2_%s_violin.png" % (self.filename, key, celltype2, self.celltype))
                else:
                plt.title("%s in %s %s" % (key, self.celltype, subcell))
                plt.savefig("%s/%s_%s_%s_violin.png" % (self.filename, key, subcell, self.celltype))
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
        self.max_length_df = np.max(np.array(
            [len(self.dictionary1[self.dictionary1.keys()[0]]), len(self.dictionary1[self.dictionary1.keys()[0]])]))
        self.color_palette = {celltype1: color1, celltype2: color2}

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


    def result_df_perparam(self, key):
        """
        creates pd.Dataframe per parameter for easier plotting
        :param key: parameter to be compared as key in dictionary
        :return: result_df
        """

        results_for_plotting = pd.DataFrame(columns=[self.celltype1, self.celltype2], index=range(self.max_length_df))
        results_for_plotting.loc[0:len(self.dictionary1[key]) - 1, self.celltype1] = self.dictionary1[key]
        results_for_plotting.loc[0:len(self.dictionary2[key]) - 1, self.celltype2] = self.dictionary1[key]
        return results_for_plotting

    def plot_violin(self, key, result_df, subcell, stripplot = True, conn_celltype = None, outgoing = False):
        """
        makes a violinplot of a specific parameter that is compared within two dictionaries.
        :param key: parameter that is compared
        :param result_df: dataframe containing results
        :param subcell: subcellular compartment
        :param stripplot: if true then stripplot will be overlayed
        :param conn_celltype: if connectivity to third celltype tested
        :param outgoing: if True, compared celltypes are presynaptic
        :return: None
        """
        sns.violinplot(data=result_df, inner="box", palette=self.color_palette)
        if stripplot:
            sns.stripplot(data=result_df, color="black", alpha=0.2)
        plt.ylabel(self.param_label(key, subcell))
        if conn_celltype:
            if outgoing:
                plt.title("%s in %s, %s to%s" % (key, self.celltype1, self.celltype2, conn_celltype))
                plt.savefig(
                    "%s/%s_%s_%s_2_%s_violin.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title("%s in %s to %s, %s" % (key, conn_celltype, self.celltype1, self.celltype2))
                plt.savefig("%s/%s_%s_2_%s_%s_violin.png" % (self.filename, key, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title("%s in %s, %s" % (key, self.celltype1, self.celltype2))
            plt.savefig("%s/%s_%s_%s_violin.png" % (self.filename, key, self.celltype1, self.celltype2))
        plt.close()

    def plot_box(self, key, result_df, subcell, stripplot = True, conn_celltype = None, outgoing = False):
        """
        makes a violinplot of a specific parameter that is compared within two dictionaries.
        :param key: parameter that is compared
        :param result_df: dataframe containing results
        :param subcell: subcellular compartment
        :param stripplot: if true then stripplot will be overlayed
        :param conn_celltype: if connectivity to third celltype tested
        :param outgoing: if True, compared celltypes are presynaptic
        :return: None
        """
        sns.boxplot(data=result_df, palette=self.color_palette)
        if stripplot:
            sns.stripplot(data=result_df, color="black", alpha=0.2)
        plt.ylabel(self.param_label(key, subcell))
        if conn_celltype:
            if outgoing:
                plt.title("%s in %s, %s to%s" % (key, self.celltype1, self.celltype2, conn_celltype))
                plt.savefig(
                    "%s/%s_%s_%s_2_%s_box.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title("%s in %s to %s, %s" % (key, conn_celltype, self.celltype1, self.celltype2))
                plt.savefig("%s/%s_%s_2_%s_%s_box.png" % (self.filename, key, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title("%s in %s, %s" % (key, self.celltype1, self.celltype2))
            plt.savefig("%s/%s_%s_%s_box.png" % (self.filename, key, self.celltype1, self.celltype2))
        plt.close()


