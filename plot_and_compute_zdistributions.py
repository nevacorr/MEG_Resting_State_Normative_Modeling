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

def one_plot(ax, ptitle, ptitleB, Z_male_region, Z_female_region, binedges, zlim, yeslegend, nokde):
    if nokde==1:
        ax.hist(Z_male_region, bins=binedges, label='male', alpha=0.4, color='b')
        ax.hist(Z_female_region, bins=binedges, label='female', alpha=0.4, color='crimson')
        ax.set_ylabel('Number of Subjects', fontsize=14)
    elif nokde==0:
        Z_male_df = pd.Series(Z_male_region, name='male').to_frame()
        Z_female_df = pd.Series(Z_female_region, name='female').to_frame()
        Z_male_df.rename(columns={0: 'male'}, inplace=True)
        Z_female_df.rename(columns={0: 'female'}, inplace=True)
        sns.kdeplot(Z_male_region, ax=ax, color = 'b', bw_adjust=1, label='male')
        sns.kdeplot(Z_female_region, ax=ax, color = 'crimson', bw_adjust=1, label='female')
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

def plot_separate_figures_sorted(df, Z_female, Z_male, binedges, zlim, struct_var,f, nokde, working_dir):
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
            plt.savefig('{}/data/{}/plots/{}_{}'.format(working_dir, struct_var+'_male', figstr, f'fig{fignum}'))
            fignum += 1

        if i == df.shape[0]-1:
            plt.savefig(
                '{}/data/{}/plots/{}_{}'.format(working_dir, struct_var+'_male', figstr, f'fig{fignum}'))
            fignum += 1

        plt.show(block=False)
    return fignum

def plot_by_gender_no_kde(struct_var, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f,
                          pvals_corrected_m, binedges, nokde, working_dir):

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
    fignum=plot_separate_figures_sorted(rois_pvals_sig_femalesigonly, Z_female, Z_male, binedges, zlim,
                                        struct_var,0, nokde, working_dir)
    fignum=plot_separate_figures_sorted(rois_pvals_sig_allsig, Z_female, Z_male, binedges, zlim, struct_var,
                                        fignum, nokde, working_dir)
    fignum=plot_separate_figures_sorted(rois_pvals_sig_malessigonly, Z_female, Z_male, binedges, zlim,struct_var,
                                        fignum, nokde, working_dir)
    fignum=plot_separate_figures_sorted(rois_pvals_notsig, Z_female, Z_male, binedges, zlim,struct_var,fignum,
                                        nokde, working_dir)

    plt.show()
    mystop=1

def plot_and_compute_zcores_by_gender(band, Z_timepoint2, working_dir):

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

    filepath = working_dir
    if len(regions_reject_f) > 1 :
        write_list_to_file(regions_reject_f, filepath + f'regions_reject_null_{band}_female.csv')
        write_list_to_file(regions_reject_m, filepath + f'regions_reject_null_{band}_male.csv')

    maxf = Z_female[roi_ids].max(axis=0).max()
    maxm = Z_male[roi_ids].max(axis=0).max()
    minf = Z_female[roi_ids].min(axis=0).min()
    minm = Z_male[roi_ids].min(axis=0).min()

    binmin = min(minf, minm)
    binmax = max(maxf, maxm)

    binedges = np.linspace(binmin-0.5, binmax+0.5, 24)

    nokde=1
    plot_by_gender_no_kde(band, Z_female, Z_male, roi_ids, reject_f, reject_m, pvals_corrected_f,
                          pvals_corrected_m, binedges, nokde, working_dir)

