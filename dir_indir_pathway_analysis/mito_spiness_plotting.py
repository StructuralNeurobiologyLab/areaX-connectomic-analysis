if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    import seaborn as sns
    import os as os
    from scipy.stats import ranksums

    min_comp_len = 200
    celltype = 2
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    folder_dict_name = ("u/arother/test_folder/210618_j0251v3_mito_spiness_%s_mcl%i" % (ct_dict[celltype], min_comp_len))
    axo_synapse_dict = load_pkl2obj("%s/%s_axo_synapse_dict.pkl" % (folder_dict_name, ct_dict[celltype]))
    den_synapse_dict = load_pkl2obj("%s/%s_den_synapse_dict.pkl" % (folder_dict_name, ct_dict[celltype]))
    axo_mito_dict = load_pkl2obj("%s/%s_axo_mito_dict.pkl" % (folder_dict_name, ct_dict[celltype]))
    den_mito_dict = load_pkl2obj("%s/%s_den_mito_dict.pkl" % (folder_dict_name, ct_dict[celltype]))

    spine_density = axo_synapse_dict["spine density"]
    axo_dist_mito_amount_density = axo_mito_dict["dist mito amount density"]
    axo_prox_mito_amount_density = axo_mito_dict["prox mito amount density"]
    axo_synapse_amount_density = axo_synapse_dict["syn amount density"]
    axo_synapse_size_density = axo_synapse_dict["syn size density"]
    axo_mito_amount_density = axo_mito_dict["mito amount density"]

    axo_dist_mito_threshold = 0.8
    axo_prox_mito_threshold = 0.5
    f_name = ("u/arother/test_folder/210621_j0251v3_mito_spiness_%s_mcl%i_dt_%.2f_pt_%.2f" % (ct_dict[celltype], min_comp_len, axo_dist_mito_threshold, axo_prox_mito_threshold))
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    axo_low_dist_mito_am_density_inds = np.where(axo_dist_mito_amount_density <= axo_dist_mito_threshold)[0]
    axo_high_dist_mito_am_density_inds = np.where(axo_dist_mito_amount_density > axo_dist_mito_threshold)[0]

    axo_low_prox_mito_am_density_inds = np.where(axo_prox_mito_amount_density <= axo_prox_mito_threshold)[0]
    axo_high_prox_mito_am_density_inds = np.where(axo_prox_mito_amount_density > axo_prox_mito_threshold)[0]
    axo_mito_amden_median = np.median(axo_mito_amount_density)
    axo_low_mito_am_density_inds = np.where(axo_mito_amount_density <= axo_mito_amden_median)[0]
    axo_high_mito_am_density_inds = np.where(axo_mito_amount_density > axo_mito_amden_median)[0]



    # violin plot with upper and lower half
    high_low_mito_am_den_spiness = pd.DataFrame()
    high_low_mito_am_den_spiness["spiness"] = spine_density
    high_low_mito_am_den_spiness["axo mito amount density distal"] = str()
    high_low_mito_am_den_spiness.loc[axo_low_dist_mito_am_density_inds, "axo mito amount density distal"] = "low mito amount density"
    high_low_mito_am_den_spiness.loc[
        axo_high_dist_mito_am_density_inds, "axo mito amount density distal"] = "high mito amount density"
    high_low_mito_am_den_spiness["axo mito amount density proximal"] = str()
    high_low_mito_am_den_spiness.loc[
        axo_low_prox_mito_am_density_inds, "axo mito amount density proximal"] = "low mito amount density"
    high_low_mito_am_den_spiness.loc[
        axo_high_prox_mito_am_density_inds, "axo mito amount density proximal"] = "high mito amount density"
    high_low_mito_am_den_spiness["axo synapse amount density"]= axo_synapse_amount_density
    high_low_mito_am_den_spiness["axo synapse size density"]= axo_synapse_size_density
    high_low_mito_am_den_spiness.loc[
        axo_high_mito_am_density_inds, "axo mito amount density"] = "high mito amount density"
    high_low_mito_am_den_spiness.loc[
        axo_low_mito_am_density_inds, "axo mito amount density"] = "low mito amount density"
    write_obj2pkl("%s/axo_high_low_mito_am_den_spines_df.pkl" % f_name, high_low_mito_am_den_spiness)
    high_low_mito_am_den_spiness.to_csv("%s/axo_high_low_mito_am_den_spiness_%s.csv" % (f_name, ct_dict[celltype]))


    axo_dist_stat, axo_dist_p_val = ranksums(spine_density[axo_low_dist_mito_am_density_inds], spine_density[axo_high_dist_mito_am_density_inds])
    axo_prox_stat, axo_prox_p_val = ranksums(spine_density[axo_low_prox_mito_am_density_inds],
                                             spine_density[axo_high_prox_mito_am_density_inds])
    axo_dist_synam_stat, axo_dist_synam_p_val = ranksums(axo_synapse_amount_density[axo_low_dist_mito_am_density_inds], axo_synapse_amount_density[axo_high_dist_mito_am_density_inds])
    axo_prox_synam_stat, axo_prox_synam_p_val = ranksums(axo_synapse_amount_density[axo_low_prox_mito_am_density_inds],
                                                         axo_synapse_amount_density[axo_high_prox_mito_am_density_inds])
    axo_dist_synsi_stat, axo_dist_synsi_p_val = ranksums(axo_synapse_size_density[axo_low_dist_mito_am_density_inds],
                                                         axo_synapse_size_density[axo_high_dist_mito_am_density_inds])
    axo_prox_synsi_stat, axo_prox_synsi_p_val = ranksums(axo_synapse_size_density[axo_low_prox_mito_am_density_inds],
                                                         axo_synapse_size_density[axo_high_prox_mito_am_density_inds])
    axo_mito_spiness_stat, axo_mito_spiness_p_val = ranksums(spine_density[axo_low_mito_am_density_inds], spine_density[axo_high_mito_am_density_inds])
    axo_mito_synam_stat, axo_mito_synam_p_val = ranksums(axo_synapse_amount_density[axo_low_mito_am_density_inds],
                                                             axo_synapse_amount_density[axo_high_mito_am_density_inds])
    axo_mito_synsi_stat, axo_mito_synsi_p_val = ranksums(axo_synapse_size_density[axo_low_mito_am_density_inds],
                                                             axo_synapse_size_density[axo_high_mito_am_density_inds])
    ranksum_results = pd.DataFrame(columns=["high vs low spiness distal mitos", "high vs low spiness proximal mitos", "high vs low syn amount proximal mitos",
                                            "high vs low syn amount distal mitos", "high vs low syn size distal mitos",
                                            "high vs low syn size proximal mitos", "high vs low spiness mitos",
                                            "high vs low syn amount mitos", "high vs low syn size mitos"], index=["stats", "p value"])
    ranksum_results["high vs low spiness distal mitos"]["stats"] = axo_dist_stat
    ranksum_results["high vs low spiness distal mitos"]["p value"] = axo_dist_p_val
    ranksum_results["high vs low spiness proximal mitos"]["stats"] = axo_prox_stat
    ranksum_results["high vs low spiness proximal mitos"]["p value"] = axo_prox_p_val
    ranksum_results["high vs low syn amount distal mitos"]["stats"] = axo_dist_synam_stat
    ranksum_results["high vs low syn amount distal mitos"]["p value"] = axo_dist_synam_p_val
    ranksum_results["high vs low syn amount proximal mitos"]["stats"] = axo_prox_synam_stat
    ranksum_results["high vs low syn amount proximal mitos"]["p value"] = axo_prox_synam_p_val
    ranksum_results["high vs low syn size distal mitos"]["stats"] = axo_dist_synsi_stat
    ranksum_results["high vs low syn size distal mitos"]["p value"] = axo_dist_synsi_p_val
    ranksum_results["high vs low syn size proximal mitos"]["stats"] = axo_prox_synsi_stat
    ranksum_results["high vs low syn size proximal mitos"]["p value"] = axo_prox_synsi_p_val
    ranksum_results["high vs low spiness mitos"]["stats"] = axo_mito_spiness_stat
    ranksum_results["high vs low spiness mitos"]["p value"] = axo_mito_spiness_p_val
    ranksum_results["high vs low syn amount mitos"]["stats"] = axo_mito_synam_stat
    ranksum_results["high vs low syn amount mitos"]["p value"] = axo_mito_synam_p_val
    ranksum_results["high vs low syn size mitos"]["stats"] = axo_mito_synsi_stat
    ranksum_results["high vs low syn size mitos"]["p value"] = axo_mito_synsi_p_val
    ranksum_results.to_csv("%s/wilcoxon_ranksum_results.csv" % f_name)

    #violin plots
    spiness_pal = {"high mito amount density": "darkturquoise",
                   "low mito amount density": "gray"}

    sns.violinplot(x="axo mito amount density distal", y="spiness", data=high_low_mito_am_den_spiness, palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density distal", y="spiness", data=high_low_mito_am_den_spiness, color="black", alpha=0.1)
    plt.title('axonal mito amount density vs spiness in %s > 100 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_dist_spine_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density proximal", y="spiness", data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density proximal", y="spiness", data=high_low_mito_am_den_spiness, color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs spiness in %s < 50 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_dist_spine_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density distal", y="axo synapse amount density", data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density distal", y="axo synapse amount density", data=high_low_mito_am_den_spiness, color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse amount density in %s > 100 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_amden_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density proximal", y="axo synapse amount density", data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density proximal", y="axo synapse amount density", data=high_low_mito_am_den_spiness, color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse amount density in %s < 50 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_amden_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density distal", y="axo synapse size density", data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density distal", y= "axo synapse size density", data=high_low_mito_am_den_spiness, color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse size density in %s > 100 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_siden_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density proximal", y="axo synapse size density", data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density proximal", y="axo synapse size density", data=high_low_mito_am_den_spiness, color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse size density in %s < 50 µm from soma' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_siden_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density", y="spiness",
                   data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density", y="spiness", data=high_low_mito_am_den_spiness,
                  color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs spine density in %s' % ct_dict[celltype])
    filename = ("%s/vio_spiness_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density", y="axo synapse amount density",
                   data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density", y="axo synapse amount density", data=high_low_mito_am_den_spiness,
                  color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse amount density in %s' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_amden_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()

    sns.violinplot(x="axo mito amount density", y="axo synapse size density",
                   data=high_low_mito_am_den_spiness,
                   palette=spiness_pal, inner="box")
    sns.stripplot(x="axo mito amount density", y="axo synapse size density", data=high_low_mito_am_den_spiness,
                  color="black",
                  alpha=0.1)
    plt.title('axonal mito amount density vs axo synapse size density in %s' % ct_dict[celltype])
    filename = ("%s/vio_axo_syn_siden_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
    plt.savefig(filename)
    plt.close()



