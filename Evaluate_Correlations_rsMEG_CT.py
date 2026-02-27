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

    # Group by region and compute correlation
    region_corr = long_df.groupby('region').apply(
        lambda df: df['MEG_abs'].corr(df['CT_abs'])
    )

    # Sort to see the strongest associations
    region_corr.sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    sns.barplot(x=region_corr.index, y=region_corr.values)
    plt.xticks(rotation=90)
    plt.ylabel('Correlation between MEG_abs & CT_abs')
    plt.title('Per-region MEG-CT co-deviation')
    plt.show()