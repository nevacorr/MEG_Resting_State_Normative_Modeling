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
import pickle
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

bands = ['beta', 'gamma']
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = os.getcwd()

Z_time2_CT = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                         .format(ct_data_dir, 'cortthick'))

with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_male_100_splits.pkl'), 'rb') as f:
    Z2_MEG_male = pickle.load(f)

with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_female_100_splits.pkl'), 'rb') as f:
    Z2_MEG_female = pickle.load(f)

reg_cols = [col for col in Z_time2_CT.columns if col != 'participant_id']
region_list = sorted(set(col.split('-', 2)[2] for col in reg_cols))

for band in bands:
    Z2_MEG_male[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG_female[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    Z2_MEG=pd.concat([Z2_MEG_male[band], Z2_MEG_female[band]], ignore_index=True)

    Z2_CT_MEG = pd.merge(Z_time2_CT, Z2_MEG, on='participant_id', how='inner')

    df = Z2_CT_MEG.copy()

    results = []

    for region in region_list:
        ct_col = f'cortthick-lh-{region}'
        meg_col = f'{region}-lh'

        r, p = pearsonr(df[ct_col], df[meg_col])

        results.append({
            'region': region,
            'r': r,
            'p': p
        })

    results_df = pd.DataFrame(results)
    rejected, pvals_fdr, _, _ = multipletests(results_df['p'], alpha=0.05, method='fdr_bh')
    results_df['p_fdr'] = pvals_fdr
    results_df['significant_fdr'] = rejected

    long_rows = []

    # loop through each participant
    for _, row in Z2_CT_MEG.iterrows():
        subject = row['participant_id']

        # loop through regions
        for region in region_list:  # region_list from CT/MET columns
            for hemi in ['lh', 'rh']:
                ct_col = f'cortthick-{hemi}-{region}'
                meg_col = f'{region}-{hemi}'

                long_rows.append({
                    'subject': subject,
                    'region': f'{region}-{hemi}',
                    'CT_abs': abs(row[ct_col]),
                    'MEG_abs': abs(row[meg_col])
                })

    # make dataframe
    long_df = pd.DataFrame(long_rows)
    #
    # model = smf.mixedlm(
    #     "CT_abs ~ MEG_abs",  # fixed effect
    #     long_df,
    #     groups=long_df["subject"],  # subject random intercept
    #     re_formula="1"  # random intercept only
    # )

    # To also include region as a crossed random effect:
    # statsmodels requires a more manual approach using 'vc_formula'
    model = smf.mixedlm(
        "CT_abs ~ MEG_abs",
        long_df,
        groups=long_df["subject"],
        vc_formula={"region": "0 + C(region)"}  # random intercept for each region
    )

    result = model.fit()

    print(result.summary())

    print("=== Fixed Effects ===")
    print(result.fe_params)  # Intercept and MEG_abs slope
    print("\n=== Fixed Effect p-values ===")
    print(result.pvalues)

    # Subject random effect variance
    try:
        subject_var = result.cov_re.iloc[0, 0]
    except:
        subject_var = "Not shown in cov_re with vc_formula"

    # Region random effect variance
    region_var = result.vcomp[0] if len(result.vcomp) > 0 else "Not available"

    print("\n=== Random Effect Variances ===")
    print(f"Subject random intercept variance: {subject_var}")
    print(f"Region random intercept variance:  {region_var}")


    # Subject random effect variance
    try:
        subject_var = result.cov_re.iloc[0, 0]
    except:
        subject_var = "Not shown in cov_re with vc_formula"

    # Region random effect variance
    region_var = result.vcomp[0] if len(result.vcomp) > 0 else "Not available"

    print("\n=== Random Effect Variances ===")
    print(f"Subject random intercept variance: {subject_var}")
    print(f"Region random intercept variance:  {region_var}")

    # ---------------------------
    # 2. Scatterplot with fixed effect slope
    # ---------------------------
    fixed_intercept = result.fe_params['Intercept']
    fixed_slope = result.fe_params['MEG_abs']

    x_vals = np.linspace(long_df['MEG_abs'].min(), long_df['MEG_abs'].max(), 100)
    y_vals = fixed_intercept + fixed_slope * x_vals

    plt.figure(figsize=(8,6))
    plt.scatter(long_df['MEG_abs'], long_df['CT_abs'], alpha=0.3)
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Fixed effect slope')
    plt.xlabel('|MEG z-score|')
    plt.ylabel('|CT z-score|')
    plt.title('Mixed model: fixed effect of MEG_abs on CT_abs')
    plt.legend()
    plt.show()

    # ---------------------------
    # 3. Bar plot of mean subject random effects
    # ---------------------------
    # Extract random effects per subject
    subject_re = pd.Series({k: v.mean() for k, v in result.random_effects.items()})
    subject_re = subject_re.sort_values()

    plt.figure(figsize=(12,4))
    subject_re.plot(kind='bar', color='skyblue')
    plt.ylabel('Mean subject random effect')
    plt.xlabel('Subject ID')
    plt.title('Random intercepts per subject')
    plt.show()



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
    Z_time2_MEG_male = pd.read_csv('{}/Zscores_post_covid_test_all_bands_male_100_splits.txt'
                               .format(working_dir))
    Z_time2_MEG_male.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    Z_time2_MEG_female = pd.read_csv('{}/Zscores_post_covid_test_all_bands_female_100_splits.txt'
                                 .format(working_dir))
    Z_time2_MEG_female.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    # Combine male and female rsMEG data
    Z_time2_MEG = pd.concat([Z_time2_MEG_male, Z_time2_MEG_female])

    # Combine CT and rsMEG dataframes
    Z_time2_CT_and_MEG = pd.merge(Z_time2_CT, Z_time2_MEG, how='inner', on='participant_id')

    Z_time2_CT_and_MEG['gender'] = [1 if id % 2 == 1 else 0 for id in Z_time2_CT_and_MEG['participant_id']]

    # Keep only female data
    Z_time2_CT_and_MEG = Z_time2_CT_and_MEG[Z_time2_CT_and_MEG['gender']==0]

    colnames = Z_time2_MEG.columns.to_list()
    colnames.remove('participant_id')

    pearson_corr = {}
    for c in colnames:
        strs = c.split('-')
        ct_col_str = 'cortthick-' + strs[0] + '-' + strs[1]
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
