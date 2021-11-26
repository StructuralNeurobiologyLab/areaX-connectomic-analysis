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
import networkx as nx
import matplotlib.patches

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
            if "volume" in key:
                param_label = "%s volume density [µm³/µm]" % subcell
            elif "size" in key and subcell == "synapse":
                param_label = "%s size density [µm²/µm]" % subcell
            elif "length" in key:
                param_label = "%s length density [µm/µm]" % subcell
            else:
                param_label = "%s density per µm" % subcell
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
                param_label = "pathlength in µm"
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

    def multiple_param_labels(self, labels, ticks):
        self.m_labels = labels
        self.m_ticks = ticks

    def plot_violin_params(self, key, param_list,subcell, xlabel = None, ticks = None, stripplot = True, celltype2 = None, outgoing = False):
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

    def plot_box_params(self, key, param_list,subcell, xlabel = None, ticks = None, stripplot = True, celltype2 = None, outgoing = False):
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
                plt.savefig("%s/%s_%s_2_%s_box.png" % (self.filename, key, self.celltype, celltype2))
            else:
                plt.title("%s from %s to %s" % (key, celltype2, self.celltype))
                plt.savefig("%s/%s_%s_2_%s_box.png" % (self.filename, key, celltype2, self.celltype))
        else:
            plt.title("%s in %s %s" % (key, self.celltype, subcell))
            plt.savefig("%s/%s_%s_%s_box.png" % (self.filename, key, subcell, self.celltype))
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
        try:
            self.max_length_df = np.max(np.array(
                [len(self.dictionary1[list(self.dictionary1.keys())[0]]), len(self.dictionary2[list(self.dictionary2.keys())[0]])]))
        except TypeError:
            if type(self.dictionary1[list(self.dictionary1.keys())[0]]) == int and type(self.dictionary2[list(self.dictionary2.keys())[0]]) == int:
                self.max_length_df = 1
            else:
                TypeError("unknown dictionary entry")
        self.color_palette = {celltype1: color1, celltype2: color2}

    def plot_hist_comparison(self, key, subcell, cells = True, add_key = None, norm_hist = False, bins = None, xlabel = None, title = None, conn_celltype = None, outgoing = False):
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
            if add_key:
                try:
                    sns.distplot(self.dictionary1[add_key],
                                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                                 kde=False, bins=bins, label=add_key, norm_hist=True)
                except KeyError:
                    sns.distplot(self.dictionary2[add_key],
                                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                                 kde=False, bins=bins, label=add_key, norm_hist=True)
            if cells:
                plt.ylabel("fraction of cells")
            elif "pair" in key:
                plt.ylabel("fraction of %s pairs" % subcell)
            else:
                plt.ylabel("fraction of %s" % subcell)
        else:
            sns.distplot(self.dictionary1[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color1},
                         kde=False, bins=bins, label = self.celltype1)
            sns.distplot(self.dictionary2[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": self.color2},
                         kde=False, bins=bins, label=self.celltype2)
            if add_key:
                try:
                    sns.distplot(self.dictionary1[add_key],
                                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                                 kde=False, bins=bins, label=add_key)
                except KeyError:
                    sns.distplot(self.dictionary2[add_key],
                                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                                 kde=False, bins=bins, label=add_key)
            if cells:
                plt.ylabel("count of cells")
            elif "pair" in key:
                plt.ylabel("count of %s pairs" % subcell)
            else:
                plt.ylabel("count of %s" % subcell)
        plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(self.param_label(key, subcell))
        if conn_celltype:
            if outgoing:
                if title:
                    plt.title(title)
                else:
                    plt.title("%s from %s, %s to %s" % (key, self.celltype1, self.celltype2, conn_celltype))
                if norm_hist:
                    plt.savefig("%s/%s_%s_%s2%s_hist_norm.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
                else:
                    plt.savefig("%s/%s_%s_%s2%s_hist.png" % (self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                if title:
                    plt.title(title)
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


    def result_df_per_param(self, key, key2 = None, column_labels = None):
        """
        creates pd.Dataframe per parameter for easier plotting
        :param key: parameter to be compared as key in dictionary
        :param key2: if more than two groups
        :param column_labels: different column labels than celltypes when more than two groups
        :return: result_df
        """
        max_length = self.max_length_df
        if len(self.dictionary1[key]) > max_length:
            max_length = len(self.dictionary1[key])
        if len(self.dictionary2[key]) > max_length:
            max_length = len(self.dictionary2[key])
        if key2 is None:
            results_for_plotting = pd.DataFrame(columns=[self.celltype1, self.celltype2], index=range(max_length))
            results_for_plotting.loc[0:len(self.dictionary1[key]) - 1, self.celltype1] = self.dictionary1[key]
            results_for_plotting.loc[0:len(self.dictionary2[key]) - 1, self.celltype2] = self.dictionary2[key]
        else:
            try:
                key2_array = self.dictionary1[key2]
            except KeyError:
                key2_array = self.dictionary2[key2]
            key2_length = len(key2_array)
            if key2_length > self.max_length_df:
                max_length = key2_length
            if not column_labels:
                column_labels = [self.celltype1, self.celltype2, key2]
            results_for_plotting = pd.DataFrame(columns=column_labels, index=range(max_length))
            results_for_plotting.loc[0:len(self.dictionary1[key]) - 1, column_labels[0]] = self.dictionary1[key]
            results_for_plotting.loc[0:len(self.dictionary2[key]) - 1, column_labels[1]] = self.dictionary2[key]
            results_for_plotting.loc[0:key2_length - 1, column_labels[2]] = key2_array
        return results_for_plotting

    def result_df_categories(self, label_category):
        """
        creates da dataframe for comparison across keys and two parameters, one category will be a celltype comparison.
        keys should be organized in the way: column label - label e.g. amount synapses - spine head
        :param: keys: list that includes one label
        :param label_category = in column_labels, category corresponding to labels
        :param key_split: if given, where key will be split into columns and labels
        :return: results_df
        """
        column_labels = []
        labels = []
        for ki, key in enumerate(self.dictionary1.keys()):
            if "-" in key:
                key_split = key.split(" - ")
                column_labels.append(key_split[0])
                labels.append(key_split[1])
        if len(column_labels) == 0:
            raise ValueError("keys in dictionary not labelled correctly")
        column_labels = np.hstack([np.unique(column_labels), ["celltype", label_category]])
        labels = np.unique(labels)
        key_example = column_labels[0] + " - " + labels[0]
        len_ct1 = len(self.dictionary1[key_example])
        len_ct2 = len(self.dictionary2[key_example])
        sum_length =  len_ct1 + len_ct2
        result_df = pd.DataFrame(
            columns=column_labels, index=range(sum_length * len(labels)))
        result_df[label_category] = type(labels[0])
        for i, label in enumerate(labels):
            result_df.loc[sum_length * i: sum_length * (i + 1) - 1, label_category] = label
            result_df.loc[sum_length * i: sum_length * i + len_ct1 - 1, "celltype"] = self.celltype1
            result_df.loc[sum_length * i + len_ct1: sum_length * (i + 1) - 1, "celltype"] = self.celltype2
            for ci in range(len(column_labels) - 2):
                result_df.loc[sum_length * i: sum_length * i + len_ct1 - 1, column_labels[ci]] = self.dictionary1[column_labels[ci] + " - " + label]
                result_df.loc[sum_length * i + len_ct1: sum_length * (i + 1) - 1, column_labels[ci]] = self.dictionary2[column_labels[ci] + " - " + label]
        for ci in range(len(column_labels) - 2):
            result_df[column_labels[ci]] = result_df[column_labels[ci]].astype("float64")
        return result_df

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

    def plot_violin_hue(self, key, x, hue, results_df, subcell, stripplot = True, conn_celltype = None, outgoing = False):
        """
        creates violin plot with more than one parameter. Dataframe with results oat least two parameter is required
        :param key: parameter to be plotted on y axis
        :param x: dataframe column on x axis
        :param hue: dataframe column acting as hue
        :param results_df: datafram, suitable one can be created with results_df_two_params
        :param stripplot: if True creates stripplot overlay
        :param conn_celltype: if third celltype connectivty is analysed
        :param outgoing: if true, connected_ct is post_synapse
        :return: None
        """
        if stripplot:
            sns.stripplot(x=x, y=key, data=results_df, hue=hue, color="black", alpha=0.2,
                          dodge=True)
            ax = sns.violinplot(x=x, y=key, data=results_df.reset_index(), inner="box",
                                palette=self.color_palette, hue=hue)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[0:2], labels[0:2])
            plt.ylabel(self.param_label(key, subcell))
        else:
            sns.violinplot(x = x, y= key, data = results_df, inner = "box", palette=self.color_palette, hue=hue)
        if conn_celltype:
            if outgoing:
                plt.title('%s, %s/ %s to %s' % (key, self.celltype1, self.celltype2, conn_celltype))
                plt.savefig("%s/%s_%s_%s_2_%s_multi_violin.png" % (
                    self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title('%s, %s to %s/ %s' % (key, conn_celltype, self.celltype1, self.celltype2))
                plt.savefig("%s/%s_%s_2_%s_%s_multi_violin.png" % (
                    self.filename, key, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title('%s, between %s and %s in different compartments' % (key, self.celltype1, self.celltype2))
            plt.savefig("%s/%s_%s_%s_multi_violin.png" % (self.filename, key, self.celltype1, self.celltype2))
        plt.close()

    def plot_box_hue(self, key, x, hue, results_df, subcell, stripplot = True, conn_celltype = None, outgoing = False):
        """
        creates box plot with more than one parameter. Dataframe with results oat least two parameter is required
        :param key: parameter to be plotted on y axis
        :param x: dataframe column on x axis
        :param hue: dataframe column acting as hue
        :param results_df: datafram, suitable one can be created with results_df_two_params
        :param stripplot: if True creates stripplot overlay
        :param conn_celltype: if third celltype connectivty is analysed
        :param outgoing: if true, connected_ct is post_synapse
        :return: None
        """
        if stripplot:
            sns.stripplot(x=x, y=key, data=results_df, hue=hue, color="black", alpha=0.2,
                          dodge=True)
            ax = sns.boxplot(x=x, y=key, data=results_df.reset_index(),
                                palette=self.color_palette, hue=hue)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[0:2], labels[0:2])
            plt.ylabel(self.param_label(key, subcell))
        else:
            sns.boxplot(x=x, y=key, data=results_df, palette=self.color_palette, hue=hue)
        if conn_celltype:
            if outgoing:
                plt.title('%s, %s/ %s to %s' % (key, self.celltype1, self.celltype2, conn_celltype))
                plt.savefig("%s/%s_%s_%s_2_%s_multi_box.png" % (
                    self.filename, key, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title('%s, %s to %s/ %s' % (key, conn_celltype, self.celltype1, self.celltype2))
                plt.savefig("%s/%s_%s_2_%s_%s_multi_box.png" % (
                    self.filename, key, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title('%s, between %s and %s in different compartments' % (key, self.celltype1, self.celltype2))
            plt.savefig("%s/%s_%s_%s_multi_box.png" % (self.filename, key, self.celltype1, self.celltype2))
        plt.close()

    def plot_bar_hue(self, key, x, hue, results_df, conn_celltype=None, outgoing=False):
        """
        creates box plot with more than one parameter. Dataframe with results oat least two parameter is required
        :param key: parameter to be plotted on y axis
        :param x: dataframe column on x axis
        :param hue: dataframe column acting as hue
        :param results_df: datafram, suitable one can be created with results_df_two_params
        :param stripplot: if True creates stripplot overlay
        :param conn_celltype: if third celltype connectivty is analysed
        :param outgoing: if true, connected_ct is post_synapse
        :return: None
        """
        sns.barplot(x=x, y=key, data=results_df, palette=self.color_palette, hue=hue, orient="h")
        if conn_celltype:
            if outgoing:
                plt.title('%s, %s/ %s to %s' % (key, self.celltype1, self.celltype2, conn_celltype))
                plt.savefig("%s/%s_%s_%s_%s_2_%s_multi_bar.png" % (
                    self.filename, key, x, self.celltype1, self.celltype2, conn_celltype))
            else:
                plt.title('%s, %s to %s/ %s' % (key, conn_celltype, self.celltype1, self.celltype2))
                plt.savefig("%s/%s_%s_%s_2_%s_%s_multi_bar.png" % (
                    self.filename, key, x, conn_celltype, self.celltype1, self.celltype2))
        else:
            plt.title('%s, between %s and %s in different compartments' % (key, self.celltype1, self.celltype2))
            plt.savefig("%s/%s_%s_%s_%s_multi_bar.png" % (self.filename, key,x, self.celltype1, self.celltype2))
        plt.close()


def plot_nx_graph(results_dictionary, filename, title):
    G = nx.DiGraph()
    edges = [[u, v, results_dictionary[(u, v)]] for (u, v) in results_dictionary.keys()]
    G.add_weighted_edges_from(edges)
    weights = [G[u][v]["weight"] / 200 for (u, v) in G.edges()]
    labels = nx.get_edge_attributes(G, "weight")
    labels = {key: int(labels[key]) for key in labels}
    pos = nx.spring_layout(G, seed=7)
    fig = plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G, pos, node_size=3000)
    nx.draw_networkx_labels(G, pos, font_size=16)
    nx.draw_networkx_edges(G, pos, width=weights, arrows=True, connectionstyle="arc3, rad=0.3", arrowstyle= matplotlib.patches.ArrowStyle.Fancy(head_length=3.4, head_width=1.6))
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.2)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.title(title)
    plt.savefig(filename)
    plt.close()
