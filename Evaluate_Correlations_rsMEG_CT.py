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
from scipy import stats

lobes_map = {
    'frontal': ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
               'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral','frontalpole',
               'rostralanteriorcingulate', 'caudalanteriorcingulate'],

    'parietal': ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',
                'posteriorcingulate','isthmuscingulate'],

    'temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal',
                'entorhinal', 'temporalpole', 'parahippocampal'],

    'occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']
}


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

    # ---------------------------
    # Aggregate by lobe + hemisphere
    # ---------------------------
    long_rows = []

    for _, row in Z2_CT_MEG.iterrows():
        subject = row['participant_id']

        for lobe, regions in lobes_map.items():
            for hemi in ['lh', 'rh']:
                # CT and MEG columns for this lobe + hemisphere
                ct_cols = [f'cortthick-{hemi}-{r}' for r in regions if f'cortthick-{hemi}-{r}' in row]
                meg_cols = [f'{r}-{hemi}' for r in regions if f'{r}-{hemi}' in row]

                if ct_cols and meg_cols:
                    ct_avg = row[ct_cols].abs().mean()
                    meg_avg = row[meg_cols].abs().mean()

                    # lobe + hemi combined label
                    lobe_hemi = f"{lobe}_{hemi}"

                    long_rows.append({
                        'subject': subject,
                        'lobe_hemi': lobe_hemi,
                        'CT_abs': ct_avg,
                        'MEG_abs': meg_avg
                    })
    # make dataframe
    lobe_df = pd.DataFrame(long_rows)

    # ---------------------------
    # Within-subject centering of MEG
    # ---------------------------
    lobe_df['MEG_within'] = lobe_df.groupby('subject')['MEG_abs'].transform(lambda x: x - x.mean())

    # # ---------------------------
    # # Fit mixed model
    # # ---------------------------
    # model = smf.mixedlm(
    #     "CT_abs ~ 0 + MEG_within * C(lobe_hemi)",  # fixed effect includes left/right lobes
    #     lobe_df,
    #     groups=lobe_df["subject"],  # random intercept per subject
    # )

    model = smf.mixedlm(
        "CT_abs ~ 0 + MEG_within * C(lobe_hemi)",  # fixed effect includes left/right lobes
        lobe_df,
        groups=lobe_df["subject"],  # random intercept per subject
        re_formula="1+MEG_within"  # add random slopes
    )


    result = model.fit()
    print(f"{capitalize(band)} Band")
    print(result.summary())

    # Fit the model if not already done
    fitted_model = model.fit()

    fe_params = fitted_model.fe_params

    # Get lobe names
    lobe_names = [c.split('[')[1][:-1] for c in fe_params.index if c.startswith('C(lobe_hemi)')]

    # Base slope for reference lobe
    base_slope = fe_params['MEG_within']

    # Build fixed-effect intercepts and slopes per lobe
    fixed_effects = {}
    for lobe in lobe_names:
        intercept = fe_params[f'C(lobe_hemi)[{lobe}]']
        interaction_term = f'MEG_within:C(lobe_hemi)[T.{lobe}]'
        slope = base_slope + fe_params[interaction_term] if interaction_term in fe_params else base_slope
        fixed_effects[lobe] = {'intercept': intercept, 'slope': slope}

    plt.figure(figsize=(10, 6))

    # x-axis range = full MEG_within range in your data
    x = np.linspace(lobe_df['MEG_within'].min(), lobe_df['MEG_within'].max(), 100)

    # Plot fixed-effect lines
    for lobe, params in fixed_effects.items():
        y = params['intercept'] + params['slope'] * x
        plt.plot(x, y, label=lobe, linewidth=2)

    # Overlay individual subject points lightly
    for lobe in lobe_df['lobe_hemi'].unique():
        sub_df = lobe_df[lobe_df['lobe_hemi'] == lobe]
        plt.scatter(sub_df['MEG_within'], sub_df['CT_abs'], alpha=0.3, color='gray')

    plt.xlabel('MEG_within')
    plt.ylabel('CT_abs')
    plt.title(f'{capitalize(band)} band: Fixed-effect slopes per lobe_hemisphere with subject points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    coef_df = result.params.reset_index()
    coef_df.columns = ['term', 'coef']

    # Separate main effect (MEG_within) and interaction terms
    main_meg = coef_df.loc[coef_df['term'] == 'MEG_within', 'coef'].values[0]

    slopes = {}

    for term, coef in coef_df.itertuples(index=False):
        if term.startswith('MEG_within:C(lobe_hemi)'):
            # Extract lobe_hemi name
            lobe = term.split('[')[1].strip(']')
            slopes[lobe] = main_meg + coef
        elif term == 'MEG_within':
            # Reference lobe (alphabetical first)
            ref_lobe = 'reference'
            slopes[ref_lobe] = main_meg

    # 3. Compute z-values and p-values for slopes
    # z = coef / std_err (need to combine main + interaction)
    # We'll approximate by summing variances (main + interaction) for simplicity
    # Get covariance matrix
    cov = result.cov_params()

    slope_stats = []
    for lobe, slope in slopes.items():
        if lobe == 'reference':
            se = np.sqrt(cov.loc['MEG_within', 'MEG_within'])
        else:
            inter_term = f'MEG_within:C(lobe_hemi)[{lobe}]'
            se = np.sqrt(
                cov.loc['MEG_within', 'MEG_within'] +
                cov.loc[inter_term, inter_term] +
                2 * cov.loc['MEG_within', inter_term]
            )
        z = slope / se
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        slope_stats.append({'lobe_hemi': lobe, 'slope': slope, 'SE': se, 'z': z, 'p': p})

    slope_df = pd.DataFrame(slope_stats)
    print(slope_df.sort_values('p'))

    mystop=1

    # print("=== Fixed Effects ===")
    # print(result.fe_params)  # Intercept and MEG_abs slope
    # print("\n=== Fixed Effect p-values ===")
    # print(result.pvalues)
    #
    # # Subject random effect variance
    # try:
    #     subject_var = result.cov_re.iloc[0, 0]
    # except:
    #     subject_var = "Not shown in cov_re with vc_formula"
    #
    # # Region random effect variance
    # region_var = result.vcomp[0] if len(result.vcomp) > 0 else "Not available"
    #
    # print("\n=== Random Effect Variances ===")
    # print(f"Subject random intercept variance: {subject_var}")
    # print(f"Region random intercept variance:  {region_var}")
    #
    #
    # # Subject random effect variance
    # try:
    #     subject_var = result.cov_re.iloc[0, 0]
    # except:
    #     subject_var = "Not shown in cov_re with vc_formula"
    #
    # # Region random effect variance
    # region_var = result.vcomp[0] if len(result.vcomp) > 0 else "Not available"
    #
    # print("\n=== Random Effect Variances ===")
    # print(f"Subject random intercept variance: {subject_var}")
    # print(f"Region random intercept variance:  {region_var}")
    #
    # # ---------------------------
    # # 2. Scatterplot with fixed effect slope
    # # ---------------------------
    # fixed_intercept = result.fe_params['Intercept']
    # fixed_slope = result.fe_params['MEG_abs']
    #
    # x_vals = np.linspace(long_df['MEG_abs'].min(), long_df['MEG_abs'].max(), 100)
    # y_vals = fixed_intercept + fixed_slope * x_vals
    #
    # plt.figure(figsize=(8,6))
    # plt.scatter(long_df['MEG_abs'], long_df['CT_abs'], alpha=0.3)
    # plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Fixed effect slope')
    # plt.xlabel('|MEG z-score|')
    # plt.ylabel('|CT z-score|')
    # plt.title('Mixed model: fixed effect of MEG_abs on CT_abs')
    # plt.legend()
    # plt.show()
    #
    # # ---------------------------
    # # 3. Bar plot of mean subject random effects
    # # ---------------------------
    # # # Extract random effects per subject
    # # subject_re = pd.Series({k: v.mean() for k, v in result.random_effects.items()})
    # # subject_re = subject_re.sort_values()
    # #
    # # plt.figure(figsize=(12,4))
    # # subject_re.plot(kind='bar', color='skyblue')
    # # plt.ylabel('Mean subject random effect')
    # # plt.xlabel('Subject ID')
    # # plt.title('Random intercepts per subject')
    # # plt.show()
    # #
    # # # Group by region and compute correlation
    # # region_corr = long_df.groupby('region').apply(
    # #     lambda df: df['MEG_abs'].corr(df['CT_abs'])
    # # )
    # #
    # # # Sort to see the strongest associations
    # # region_corr.sort_values(ascending=False)
    # #
    # # plt.figure(figsize=(12, 5))
    # # sns.barplot(x=region_corr.index, y=region_corr.values)
    # # plt.xticks(rotation=90)
    # # plt.ylabel('Correlation between MEG_abs & CT_abs')
    # # plt.title('Per-region MEG-CT co-deviation')
    # # plt.show()