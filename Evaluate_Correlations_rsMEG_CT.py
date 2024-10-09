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

lobes_only=1
bands = ['theta', 'alpha', 'beta', 'gamma']
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = os.getcwd()

frontal_reg = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
               'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral',
               'frontalpole',
               'rostralanteriorcingulate', 'caudalanteriorcingulate']

parietal_reg = ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',
                'posteriorcingulate',
                'isthmuscingulate']

temporal_reg = ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal',
                'entorhinal', 'temporalpole', 'parahippocampal']

occipital_reg = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']

Z_MEG_time2 = {}

Z_time2_CT = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                         .format(ct_data_dir, 'cortthick'))

# Average Z-scores for regions in each lobe

region_dict = {
    'frontal': frontal_reg,
    'parietal': parietal_reg,
    'temporal': temporal_reg,
    'occipital': occipital_reg
}

Z2_CT_avgreg = pd.DataFrame(index=Z_time2_CT.index)

hemispheres = ['-lh', '-rh']

for region_name, regions in region_dict.items():
    for hemi in hemispheres:
        # Create a pattern to match the columns of interest
        cols_to_avg = [col for col in Z_time2_CT.columns if any(f'cortthick{hemi}-{region}' in col for region in regions)]

        if cols_to_avg:
            # Average the values across columns in the region
            Z2_CT_avgreg[f'cortthick-{region_name}{hemi}'] = Z_time2_CT[cols_to_avg].mean(axis=1)

# Merge the new averaged columns with the original dataframe
# This will overwrite the matching columns but keep the other columns unchanged
Z_time2_CT = Z_time2_CT.drop(columns=[col for col in Z_time2_CT.columns if 'cortthick' in col]) # Remove original region columns

Z_time2_CT = pd.concat([Z_time2_CT, Z2_CT_avgreg], axis=1)

correlation_df = pd.DataFrame(columns=bands)
pval_df = pd.DataFrame(columns=bands)

palette = {1: 'blue', 0: 'crimson'}

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

    Z_time2_CT_and_MEG = Z_time2_CT_and_MEG[Z_time2_CT_and_MEG['gender']==0]

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
            sns.regplot(x=Z_time2_CT_and_MEG.iloc[:, ct_colnumber], y=Z_time2_CT_and_MEG.iloc[:, meg_colnumber],  color='gray', scatter_kws={'color': 'gray'}, ci=None)
            sns.scatterplot(x=Z_time2_CT_and_MEG.columns[ct_colnumber], y=Z_time2_CT_and_MEG.columns[meg_colnumber],
                            hue='gender', data=Z_time2_CT_and_MEG, palette=palette)
            plt.title(f'Z-score for MEG Power in {capitalize(band)} band vs.\n Z-score for Cortical Thickness in\n {c} r={rval:.2f} uncorrp={pval:.3f}')
            plt.tight_layout()
            plt.savefig(f'{working_dir}/plots/MEG_Zscores_{band}_band_vs_CT_Zscores_{c}.png')
            plt.show(block=False)

correlation_df = correlation_df.astype(float)

plt.figure(figsize=(12, 15))

plt.imshow(correlation_df, cmap ="RdYlBu", aspect= "auto")
plt.colorbar()
plt.title('Females Only\nPearson Correlation between Cortical Thickness Z-score and rsMEG Power Z-score')
plt.xticks(range(correlation_df.shape[1]), correlation_df.columns)
plt.yticks(range(correlation_df.shape[0]), correlation_df.index)
plt.xlabel('rsMEG Power Band')
plt.ylabel('Brain Region')
plt.tight_layout()
plt.show(block=False)


mystop = 1
