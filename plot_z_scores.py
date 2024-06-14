###
#This program plots post-covid z-scores from the normative model, and makes a separate curve for each not gender
###

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import math
from helper_functions_MEG import write_list_to_file
from matplotlib import ticker as mtick
from matplotlib import rc

def plot_and_compute_zcores_onlykde_specify_region(Z_time2, struct_var, roi_ids, region, ymax):
    # add gender to Z score dataframe
    # females have even subject numbers, males have odd subject numbers
    Z_time2['gender'] = Z_time2['participant_id'].apply(lambda x: 2 if x % 2 == 0 else 1)
    # move the gender column to the front of the dataframe
    gender = Z_time2.pop('gender')
    Z_time2.insert(1, 'gender', gender)

    Z_female = Z_time2[Z_time2['gender'] == 2]
    Z_male = Z_time2[Z_time2['gender'] == 1]
   # region = 'cortthick-rh-precuneus'

    zf = Z_female[region].values
    t_statistic_f, p_value_f = stats.ttest_1samp(zf, popmean=0, nan_policy='raise')

    zm = Z_male[region].values
    t_statistic_m, p_value_m = stats.ttest_1samp(zm, popmean=0, nan_policy='raise')

    reject_f, pvals_corrected_f, a1_f, a2_f = smt.multipletests(p_value_f, alpha=0.05, method='fdr_bh')
    reject_m, pvals_corrected_m, a1_m, a2_m = smt.multipletests(p_value_m, alpha=0.05, method='fdr_bh')

    regions_reject_f = [roi_id for roi_id, reject_value in zip(roi_ids, reject_f) if reject_value]
    regions_reject_m = [roi_id for roi_id, reject_value in zip(roi_ids, reject_m) if reject_value]

    maxf = Z_female[region].max(axis=0).max()
    maxm = Z_male[region].max(axis=0).max()
    minf = Z_female[region].min(axis=0).min()
    minm = Z_male[region].min(axis=0).min()

    binmin = min(minf, minm)
    binmax = max(maxf, maxm)
    binedges = np.linspace(binmin-0.5, binmax+0.5, 24)

    zmax = math.ceil(binedges[-1])
    zmin = math.floor(binedges[0])
    zlim = abs(max(abs(zmin), abs(zmax)))
    zlim=5

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(4.1, 6)

    #ptitle = 'right hemisphere precuneus\nby gender'
    ptitle = region
    one_plot(ax1, ptitle, Z_male[region], Z_female[region], binedges, zlim, 1, 0)

    # sns.kdeplot(Z_time2[region], color='k', ax=ax1)
    ax1.set_xlim(-zlim, zlim)
    ax1.set_ylim(0.0, ymax)
    ax1.set_xlabel('z-score', fontsize=12)
    ax1.set_ylabel('probability density', fontsize=12)
    ax1.set_title(f'{region}\nall subjects', fontsize=12)
    ax1.axvline(x=0, color='dimgray', linestyle='--')

    fmt='%.2f'
    yticks=mtick.FormatStrFormatter(fmt)
    ax1.yaxis.set_major_formatter(yticks)

    plt.tight_layout()
    plt.show(block=False)


def plot_and_compute_zcores(Z_time2, struct_var, roi_ids):

    p_values = []
    for region in roi_ids:
        z2 = Z_time2[region].values
        t_statistic, p_value = stats.ttest_1samp(z2, popmean=0, nan_policy='raise')
        p_values.append(p_value)

    reject, pvals_corrected, a1, a2 = smt.multipletests(p_values ,alpha=0.05, method='fdr_bh')

    sig_string_list=[]
    for i, r in enumerate(roi_ids):
        zmean = np.mean(Z_time2[r])
        sig_string_list.append(f'{r}: mean = {zmean:.2} p-value = {pvals_corrected[i]:.2e}, Mean not 0 is {reject[i]}')

    for i, region in enumerate(roi_ids):
        if reject[i] == True:
            fig, axs = plt.subplots(2)
            fig.set_size_inches(10, 9)
            plt.subplot(2,1,1)
            zmin=min(Z_time2[region])
            zmax=max(Z_time2[region])
            l2=plt.hist(Z_time2[region], label='post-covid')
            plt.xlim(zmin-4, zmax+4)
            plt.xlabel('Z-score')
            plt.legend()
            fig.suptitle('Z-score Distributions Based on Normative Model\n{}'.format(sig_string_list[i]))
            ax = plt.subplot(2,1,2)
            sns.kdeplot(Z_time2[region], ax=axs[1])
            plt.xlim(zmin-4, zmax+4)
            plt.xlabel('Z-score')
            plt.show(block=False)
            plt.savefig(
            '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/data/{}/plots/Z_scores_post_covid_{}'
               .format(struct_var, region))
            # plt.close()
    plt.show()

def plot_by_gender(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m):
    sig_string_list = []
    for i, r in enumerate(roi_ids):
        zmean_f = np.mean(Z_female[r])
        zmean_m = np.mean(Z_male[r])
        sig_string_list.append(
            f'{r}: female mean = {zmean_f:.2} p-value = {pvals_corrected_f[i]:.2e}, Mean not 0 is {reject_f[i]}\n'
            f'{r}: male mean = {zmean_m:.2} p-value = {pvals_corrected_m[i]:.2e}, Mean not 0 is {reject_m[i]}')
    for i, region in enumerate(roi_ids):
        if (reject_f[i] == True) | (reject_m[i] == True):
            fig, axs = plt.subplots(2)
            fig.set_size_inches(10, 9)
            plt.subplot(2, 1, 1)
            zmin = min(Z_female[region])
            zmax = max(Z_female[region])
            lf = plt.hist(Z_female[region], label='post-covid female', alpha=0.8)
            lm = plt.hist(Z_male[region], label='post-covid male', alpha=0.8)
            plt.xlim(zmin - 4, zmax + 4)
            plt.xlabel('Z-score')
            plt.legend()
            fig.suptitle('Z-score Distributions Based on Normative Model\n{}'.format(sig_string_list[i]))
            ax = plt.subplot(2, 1, 2)
            sns.kdeplot(Z_female[region], ax=axs[1], bw_adjust=0.5)
            sns.kdeplot(Z_male[region], ax=axs[1], bw_adjust=0.5)
            plt.xlim(zmin - 4, zmax + 4)
            plt.xlabel('Z-score')
            plt.show(block=False)
            # plt.savefig(
            # '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/data/{}/plots/Z_scores_post_covid_by_gender{}'
            # .format(struct_var, region))
            # plt.close()
            mystop = 1
    plt.show()

def one_plot(ax, ptitle, ptitleB, Z_male_region, Z_female_region, binedges, zlim, yeslegend, nokde):
    if nokde==1:
        ax.hist(Z_male_region, bins=binedges, label='male', alpha=0.4, color='b')
        ax.hist(Z_female_region, bins=binedges, label='female', alpha=0.4, color='g')
        ax.set_ylabel('Number of Subjects', fontsize=14)
    elif nokde==0:
        Z_male_df = pd.Series(Z_male_region, name='male').to_frame()
        Z_female_df = pd.Series(Z_female_region, name='female').to_frame()
        Z_male_df.rename(columns={0: 'male'}, inplace=True)
        Z_female_df.rename(columns={0: 'female'}, inplace=True)
        sns.kdeplot(Z_male_region, ax=ax, color = 'b', bw_adjust=1, label='male')
        sns.kdeplot(Z_female_region, ax=ax, color = 'g', bw_adjust=1, label='female')
        ax.set_ylabel('probability density', fontsize=12)
    ax.axvline(x=0, color='dimgray', linestyle='--')
    ax.set_xlim(-zlim, zlim)
    ax.set_xlabel('Z-score', fontsize=14)
    plt.text(0.5, 1.08, ptitleB, fontsize=14, fontweight='bold', ha = 'center', va='bottom', transform=ax.transAxes)
    plt.text(0.5, 1.01, ptitle, fontsize=14, ha='center', va='bottom', transform=ax.transAxes)
    #ax.set_title(ptitle, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if yeslegend:
        ax.legend(fontsize=14)
    # plt.tight_layout()

def plot_separate_figures_sorted(df, Z_female, Z_male, binedges, zlim, struct_var,f, nokde):
    sig_string_list = []
    bold_string_list = []
    if nokde == 1:
        figstr = 'Z_scores_post_covid_by_gender_nokde'
    elif nokde == 0:
        figstr = 'Z_scores_post_covid_by_gender_withkde'

    for i, r in enumerate(df['roi_ids']):
        zmean_f = np.mean(Z_female[r])
        zmean_m = np.mean(Z_male[r])
        region_string = r.split("-")
        if len(region_string)==1:
            hemi = ''
            region_for_title = r

        else:
            if region_string[1] == 'rh':
                hemi = 'Right hemisphere '
            elif region_string[1] == 'lh':
                hemi = 'Left hemisphere '
            region_for_title = region_string[2]
            if region_for_title == 'bankssts':
                region_for_title = 'banks of STS'
            elif region_for_title == 'frontalpole':
                region_for_title = 'frontal pole'
            elif region_for_title == 'superiortemporal':
                region_for_title = 'superior temporal'
            elif region_for_title == 'lateraloccipital':
                region_for_title = 'lateral occipital'
        bold_string = f'{hemi}{region_for_title}\n'
        not_bold_string = (f'female mean = {zmean_f:.2} p = {df.loc[i, "pfemale"]:.2e}\n '
                           f'male mean = {zmean_m:.2} p = {df.loc[i, "pmale"]:.2e}')
        bold_string_list.append(bold_string)
        sig_string_list.append(not_bold_string)
        #sig_string_list.append(f'{hemi}{region_for_title}\ncortical thickness')

    fignum = f
    for i, region in enumerate(df['roi_ids']):
        a = (i + 1) % 6
        if (df.shape[0] <30 ) & (i == df.shape[0]-1):
            yeslegend = 1
        elif a == 3:
            yeslegend = 1
        else:
            yeslegend = 0

        if a == 1:
            ptitleB = f'{bold_string_list[i]}'
            ptitle = f'{sig_string_list[i]}'
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
            fig.set_size_inches(13.5, 10)
            fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.05, right=0.95)
            one_plot(ax1, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
        elif a == 2:
            ptitle = f'{sig_string_list[i]}'
            ptitleB = f'{bold_string_list[i]}'
            one_plot(ax2, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
        elif a == 3:
            ptitle = f'{sig_string_list[i]}'
            ptitleB = f'{bold_string_list[i]}'
            one_plot(ax3, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
        elif a % 6 == 4:
            ptitle = f'{sig_string_list[i]}'
            ptitleB = f'{bold_string_list[i]}'
            one_plot(ax4, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
        elif a % 6 == 5:
            ptitle = f'{sig_string_list[i]}'
            ptitleB = f'{bold_string_list[i]}'
            one_plot(ax5, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
        elif a % 6 == 0:
            ptitle = f'{sig_string_list[i]}'
            ptitleB = f'{bold_string_list[i]}'
            one_plot(ax6, ptitle, ptitleB, Z_male[region], Z_female[region], binedges, zlim, yeslegend, nokde)
            plt.savefig(
                '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/data/{}/plots/{}_{}'
                .format(struct_var+'_male', figstr, f'fig{fignum}'))
            fignum += 1

        if i == df.shape[0]-1:
            plt.savefig(
                '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/data/{}/plots/{}_{}'
                .format(struct_var+'_male', figstr, f'fig{fignum}'))
            fignum += 1

        plt.show(block=False)
    return fignum

def plot_by_gender_no_kde(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m, binedges, nokde):

    zmax = math.ceil(binedges[-1])
    zmin = math.floor(binedges[0])
    zlim = abs(max(abs(zmin), abs(zmax)))

    #sort roi ids
    #sort for  reject_f ==True and reject_m == True and sort buy pvalue
    #then add strings with reject_f == True and reject_m ==True, sorted by pvalue
    #then add strings with reject_f == False and reject_m ==True, sorted by pvalue
    #then add strings with reject_f == True and reject_m == False and sort by pvalue
    rois_pvals_sig = pd.DataFrame(roi_ids, columns=['roi_ids'])
    rois_pvals_sig['pfemale'] = pvals_corrected_f.tolist()
    rois_pvals_sig['pmale'] = pvals_corrected_m.tolist()
    rois_pvals_sig['rejectf'] =  reject_f.tolist()
    rois_pvals_sig['rejectm'] = reject_m.tolist()
    rois_pvals_sig_femalesigonly = rois_pvals_sig.loc[(rois_pvals_sig['rejectf']==True) & (rois_pvals_sig['rejectm']==False)].copy()
    rois_pvals_sig_allsig = rois_pvals_sig.loc[(rois_pvals_sig['rejectf']==True) & (rois_pvals_sig['rejectm']==True)].copy()
    rois_pvals_sig_malessigonly = rois_pvals_sig[(rois_pvals_sig['rejectf']==False) & (rois_pvals_sig['rejectm']==True)].copy()
    rois_pvals_notsig = rois_pvals_sig[(rois_pvals_sig['rejectf']==False) & (rois_pvals_sig['rejectm']==False)].copy()
    rois_pvals_sig_femalesigonly.sort_values(by=['pfemale'], axis=0, inplace=True, ignore_index=True)
    rois_pvals_sig_allsig.sort_values(by=['pfemale'], axis=0, inplace=True, ignore_index=True)
    rois_pvals_sig_malessigonly.sort_values(by=['pmale'], axis=0, inplace=True, ignore_index=True)
    rois_pvals_notsig.sort_values(by=['pfemale'], axis=0, inplace=True, ignore_index=True)

    #plot separate figures for each category
    fignum=plot_separate_figures_sorted(rois_pvals_sig_femalesigonly, Z_female, Z_male, binedges, zlim, struct_var,0, nokde)
    fignum=plot_separate_figures_sorted(rois_pvals_sig_allsig, Z_female, Z_male, binedges, zlim, struct_var, fignum, nokde)
    fignum=plot_separate_figures_sorted(rois_pvals_sig_malessigonly, Z_female, Z_male, binedges, zlim,struct_var,fignum, nokde)
    fignum=plot_separate_figures_sorted(rois_pvals_notsig, Z_female, Z_male, binedges, zlim,struct_var,fignum, nokde)

    plt.show()
    mystop=1

def plot_by_gender_distsubplots(Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m):
    #there is an error in this function it does not always plot the right label with the distribution
    sig_string_list = []
    for i, r in enumerate(roi_ids):
        zmean_f = np.mean(Z_female[r])
        zmean_m = np.mean(Z_male[r])
        sig_string_list.append(
            f'{r}: female mean = {zmean_f:.2} p = {pvals_corrected_f[i]:.2e}, Mean not 0 is {reject_f[i]}\n'
            f'{r}: male mean = {zmean_m:.2} p = {pvals_corrected_m[i]:.2e}, Mean not 0 is {reject_m[i]}')
    n_rows=3
    n_cols=4
    sns.set(font_scale=0.5)
    num_plots=37
    m=math.ceil(num_plots/n_rows*n_cols)
    last_roi_id=0
    allplotnum=0
    sns.set_style(style='white')
    for row in range(0,m,n_rows*n_cols):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.set_size_inches(16,8)
        fig.suptitle('Cortical Thickness Z-score Distributions Based on Normative Model')
        plotnum=0
        for i, region in enumerate(roi_ids[last_roi_id:]):
            if (reject_f[i] == True) | (reject_m[i] == True):
                zmin = min(Z_female[region])
                zmax = max(Z_female[region])
                ax = axes[plotnum // n_cols, plotnum % n_cols]
                sns.kdeplot(Z_female[region], ax=ax)
                sns.kdeplot(Z_male[region], ax=ax)
                plt.xlim(zmin - 4, zmax + 4)
                ax.set_title(sig_string_list[i])
                ax.set_xlabel('')
              #  plt.legend()
                plotnum+=1
                allplotnum+=1
            if plotnum == n_cols*n_rows:
                last_roi_id = allplotnum
                plt.show()
                break
         # plt.savefig(
        # '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/data/{}/plots/Z_scores_post_covid_by_gender_dist_subplots{}'
        #   .format(struct_var, region))
         #          plt.close()
        mystop=1

def plot_and_compute_zcores_by_gender(struct_var, Z_timepoint2):

    Z_male = Z_timepoint2['male']
    Z_female = Z_timepoint2['female']

    # add gender to Z score dataframe
    Z_male['gender'] = 1
    Z_female['gender'] = 2
    #move the gender column to the front of the dataframes
    gender = Z_male.pop('gender')
    Z_male.insert(1, 'gender', gender)
    gender = Z_female.pop('gender')
    Z_female.insert(1, 'gender', gender)

    #get list of all brain regions
    sinfo = ['participant_id', 'gender']
    roi_ids = [col for col in Z_male.columns if col not in sinfo]

    p_values_f = []
    p_values_m = []
    for region in roi_ids:
        zf = Z_female[region].values
        t_statistic_f, p_value_f = stats.ttest_1samp(zf, popmean=0, nan_policy='raise')
        p_values_f.append(p_value_f)

        zm = Z_male[region].values
        t_statistic_m, p_value_m = stats.ttest_1samp(zm, popmean=0, nan_policy='raise')
        p_values_m.append(p_value_m)

    reject_f, pvals_corrected_f, a1_f, a2_f = smt.multipletests(p_values_f, alpha=0.05, method='fdr_bh')
    reject_m, pvals_corrected_m, a1_m, a2_m = smt.multipletests(p_values_m, alpha=0.05, method='fdr_bh')

    #write regions where reject_f is True to file
    regions_reject_f = [roi_id for roi_id, reject_value in zip(roi_ids, reject_f) if reject_value]
    regions_reject_m = [roi_id for roi_id, reject_value in zip(roi_ids, reject_m) if reject_value]

    filepath = '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate/'
    if len(regions_reject_f) > 1 :
        write_list_to_file(regions_reject_f, filepath + f'regions_reject_null_{struct_var}_female.csv')
        write_list_to_file(regions_reject_m, filepath + f'regions_reject_null_{struct_var}_male.csv')

    maxf = Z_female[roi_ids].max(axis=0).max()
    maxm = Z_male[roi_ids].max(axis=0).max()
    minf = Z_female[roi_ids].min(axis=0).min()
    minm = Z_male[roi_ids].min(axis=0).min()

    binmin = min(minf, minm)
    binmax = max(maxf, maxm)

    binedges = np.linspace(binmin-0.5, binmax+0.5, 24)

    #plot_and_compute_zcores(Z_time2, struct_var, roi_ids)

    #plot_by_gender(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m)
    nokde=1
    plot_by_gender_no_kde(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m, binedges, nokde)

    #there is something wrong the function commented out in next line
    #it doesn't seem to always plot the correct distribution with the corect label
    #plot_by_gender_distsubplots(Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f, pvals_corrected_m)

    mystop=1