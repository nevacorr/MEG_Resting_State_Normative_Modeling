###
# This programs evaluates correlations between Z-score values for accelerated cortical thinning at the
# post-COVID time point and Z-scores for resting state MEG power at the post-COVID time point

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.defchararray import capitalize
from scipy.stats import pearsonr
import numpy as np

bands = ['theta', 'alpha', 'beta', 'gamma']
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = os.getcwd()

Z_MEG_time2 = {}

Z_time2_CT = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                         .format(ct_data_dir, 'cortthick'))

correlation_df = pd.DataFrame(columns=bands)
pval_df = pd.DataFrame(columns=bands)

for band in bands:
    Z_time2_MEG_male = pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                               .format(working_dir, 'male', band))
    Z_time2_MEG_male.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    Z_time2_MEG_female = pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                 .format(working_dir, 'female', band))
    Z_time2_MEG_female.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    # Combine male and female rsMEG data
    Z_time2_MEG = pd.concat([Z_time2_MEG_male, Z_time2_MEG_female])

    # Combine CT and rsMEG dataframes
    Z_time2_CT_and_MEG = pd.merge(Z_time2_CT, Z_time2_MEG, how='inner', on='participant_id')

    Z_time2_CT_and_MEG['gender'] = [1 if id % 2 == 1 else 0 for id in Z_time2_CT_and_MEG['participant_id']]

    colnames = Z_time2_MEG.columns.to_list()
    colnames.remove('participant_id')

    pearson_corr = {}
    for c in colnames:
        strs = c.split('-')
        ct_col_str = 'cortthick-' + strs[1] + '-' + strs[0]
        ct_colnumber = Z_time2_CT_and_MEG.columns.get_loc(ct_col_str)
        meg_colnumber = Z_time2_CT_and_MEG.columns.get_loc(c)

        rval, pval = pearsonr(Z_time2_CT_and_MEG.iloc[:, ct_colnumber], (Z_time2_CT_and_MEG.iloc[:,meg_colnumber]))

        correlation_df.loc[c, band] = rval
        pval_df.loc[c, band] = pval

        if pval <0.2:
            pval_df.loc[c, band] = pval
        else:
            pval = np.nan

        if pval <0.05:
            plt.figure()
            sns.regplot(x=Z_time2_CT_and_MEG.iloc[:, ct_colnumber], y=Z_time2_CT_and_MEG.iloc[:, meg_colnumber],  ci=None)
            plt.title(f'Z-score for MEG Power in {capitalize(band)} band vs.\n Z-score for Cortical Thickness in\n {c} r={rval:.2f} uncorrp={pval:.3f}')
            plt.tight_layout()
            plt.show(block=False)

correlation_df = correlation_df.astype(float)

plt.figure(figsize=(12, 15))

plt.imshow(correlation_df, cmap ="RdYlBu", aspect= "auto")
plt.colorbar()
plt.title('Uncorrected Pearson Correlation between Cortical Thickness Z-score and rsMEG Power Z-score')
plt.xticks(range(correlation_df.shape[1]), correlation_df.columns)
plt.yticks(range(correlation_df.shape[0]), correlation_df.index)
plt.xlabel('rsMEG Power Band')
plt.ylabel('Brain Region')
plt.tight_layout()
plt.show(block=False)

mystop = 1
