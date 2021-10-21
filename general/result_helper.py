#TO DO: write object for results dictionary to standardize plotting of results
#put all relevant function here in this helper file
#make possibility for 2 or more different color schemmes
#boxplot, violinplot, histogram (normed, not normed), scatterplot
#also add flowchart, networx plot
#one for single dictionary and one for comparing results from two dictionaries

import seaborn as sns
import matplotlib as plt

class ResultDict:
     '''
     this class is a dictionary with different results.
     '''
     def __init__(self, celltype, filename):
         if type(self) != dict:
             raise ValueError("must be dictionary")
         self.celltype = celltype
         self.filename = filename

     def color_palette(self, key1, key2, color1, color2):
         self.color_palette = {key1: color1, key2: color2}

     def plot_hist(self, key, subcell, cells = True, color = "steelblue", norm_hist = False, bins = 10, xlabel = None):
         if norm_hist == False:
             sns.distplot(self[key],
                          hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "color"},
                          kde=False, bins=bins)
             if cells:
                 plt.ylabel("count of cells")
             else:
                 plt.ylabel("count of %s" % subcell)
         else:
             sns.distplot(self[key],
                          hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "color"},
                          kde=False, bins=bins)
             if cells:
                 plt.ylabel("fraction of cells")
             else:
                 plt.ylabel("fraction of %s" % subcell)
         if xlabel != None:
             plt.xlabel = xlabel
         else:
             if "density" in key:
                    if "amount" in key:
                        plt.xlabel("%s density per µm" % subcell)
                    elif "volume" in key:
                        plt.xlabel("%s volume density [µm³/µm]" % subcell)
                    elif "size" in key and subcell == "synapse":
                        plt.xlabel("%s size density [µm²/µm]" % subcell)
                    elif "length" in key:
                        plt.xlabel("%s length density [µm/µm]" % subcell)
             else:
                 if "amount" in key:
                     if "percentage" in key:
                         plt.xlabel("percentage of %ss" % subcell)
                     else:
                         plt.xlabel("amount of %ss" % subcell)
                 elif "size" in key:
                     if "percentage" in key:
                         plt.xlabel("percentage of %s size" % subcell)
                     else:
                         if subcell == "synapse":
                            plt.xlabel("average %s size [µm²]" % subcell)
                 elif "length" in key:
                     plt.xlabel("pathlength in µm")
                 elif "vol" in key:
                     if "percentage" in key:
                         plt.xlabel("% of whole dataset")
                     else:
                         plt.xlabel(" %s volume in µm³" % subcell)
                 elif "distance" in key:
                     plt.xlabel("distance in µm")
                 else:
                     raise ValueError ("unknown key description")
         plt.title("%s in %s" % (key, self.celltype))
         if norm_hist:
             plt.savefig("%s/%s_%s_hist_norm.png" % (self.filename, key, self.celltype))
         else:
            plt.savefig("%s/%s_%s_hist.png" % (self.filename, key, self.celltype))
         plt.close()

