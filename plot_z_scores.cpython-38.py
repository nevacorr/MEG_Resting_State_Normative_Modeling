# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/plot_z_scores.py
# Compiled at: 2024-01-19 17:52:18
# Size of source mod 2**32: 5708 bytes
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt
import numpy as np, seaborn as sns
import scipy.stats as stats
from helper_functions_MEG import write_list_to_file
import math
from matplotlib import ticker as mtick

def plot_by_gender(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m, binedges, outputdir):
    sig_string_list = []
    for i, r in enumerate(roi_ids):
        zmean_f = np.mean(Z_female[r])
        zmean_m = np.mean(Z_male[r])
        sig_string_list.append(f"{r}: female mean = {zmean_f:.2} p-value = {pvals_corrected_f[i]:.2e}, Mean not 0 is {reject_f[i]}\n{r}: male mean = {zmean_m:.2} p-value = {pvals_corrected_m[i]:.2e}, Mean not 0 is {reject_m[i]}")
    else:
        for i, region in enumerate(roi_ids):
            fig, axs = plt.subplots(2)
            fig.set_size_inches(10, 9)
            plt.subplot(2, 1, 1)
            zmin = min(Z_female[region])
            zmax = max(Z_female[region])
            lf = plt.hist((Z_female[region]), bins=binedges, label="post-covid female", alpha=0.7, color="g")
            lm = plt.hist((Z_male[region]), bins=binedges, label="post-covid male", alpha=0.7, color="b")
            plt.xlim(zmin - 4, zmax + 4)
            plt.xlabel("Z-score")
            plt.legend()
            fig.suptitle("Z-score Distributions Based on Normative Model\n{}".format(sig_string_list[i]))
            ax = plt.subplot(2, 1, 2)
            sns.kdeplot((Z_female[region]), ax=(axs[1]), bw_adjust=0.7, color="g")
            sns.kdeplot((Z_male[region]), ax=(axs[1]), bw_adjust=0.7, color="b")
            plt.xlim(zmin - 4, zmax + 4)
            plt.xlabel("Z-score")
            plt.show(block=False)
            mystop = 1
        else:
            plt.show()


def plot_and_compute_zcores_by_gender(Z_time2, struct_var, roi_ids, outputdir):
    Z_time2["gender"] = Z_time2["participant_id"].apply(lambda x:     if x % 2 == 0:
2 # Avoid dead code: 1)
    gender = Z_time2.pop("gender")
    Z_time2.insert(1, "gender", gender)
    Z_female = Z_time2[Z_time2["gender"] == 2]
    Z_male = Z_time2[Z_time2["gender"] == 1]
    p_values_f = []
    p_values_m = []
    for region in roi_ids:
        zf = Z_female[region].values
        t_statistic_f, p_value_f = stats.ttest_1samp(zf, popmean=0, nan_policy="raise")
        p_values_f.append(p_value_f)
        zm = Z_male[region].values
        t_statistic_m, p_value_m = stats.ttest_1samp(zm, popmean=0, nan_policy="raise")
        p_values_m.append(p_value_m)
    else:
        reject_f, pvals_corrected_f, a1_f, a2_f = smt.multipletests(p_values_f, alpha=0.05, method="fdr_bh")
        reject_m, pvals_corrected_m, a1_m, a2_m = smt.multipletests(p_values_m, alpha=0.05, method="fdr_bh")
        regions_reject_f = [roi_id for roi_id, reject_value in zip(roi_ids, reject_f) if reject_value]
        regions_reject_m = [roi_id for roi_id, reject_value in zip(roi_ids, reject_m) if reject_value]
        filepath = outputdir
        if len(regions_reject_f) > 1:
            write_list_to_file(regions_reject_f, filepath + f"/regions_reject_null_{struct_var}_female.csv")
            write_list_to_file(regions_reject_m, filepath + f"/regions_reject_null_{struct_var}_male.csv")
        maxf = Z_female[roi_ids].max(axis=0).max()
        maxm = Z_male[roi_ids].max(axis=0).max()
        minf = Z_female[roi_ids].min(axis=0).min()
        minm = Z_male[roi_ids].min(axis=0).min()
        binmin = min(minf, minm)
        binmax = max(maxf, maxm)
        binedges = np.linspace(binmin - 0.5, binmax + 0.5, 30)
        plot_by_gender(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m, binedges, outputdir)
