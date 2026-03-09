import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pingouin import partial_corr  # pip install pingouin
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

level = "region" #options: "lobe", "region"
bands = ['beta', 'gamma']

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


ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = os.getcwd()

# Load cortical thickness z scores
Z_time2_CT = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                         .format(ct_data_dir, 'cortthick'))

# Load MEG z scores
with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_male_100_splits.pkl'), 'rb') as f:
    Z2_MEG_male = pickle.load(f)
with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_female_100_splits.pkl'), 'rb') as f:
    Z2_MEG_female = pickle.load(f)

reg_cols = [col for col in Z_time2_CT.columns if col != 'participant_id']
region_list = sorted(set(col.split('-', 2)[2] for col in reg_cols))

for band in bands:

    Z2_MEG_male[band]['sex'] = 1
    Z2_MEG_female[band]['sex'] = 0
    # Merge male and female MEG data
    Z2_MEG_male[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG_female[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG=pd.concat([Z2_MEG_male[band], Z2_MEG_female[band]], ignore_index=True)

    # Merge CT and MEG data
    Z2_CT_MEG = pd.merge(Z_time2_CT, Z2_MEG, on='participant_id', how='inner')

    # ---------------------------
    # Reshape/Aggregate
    # ---------------------------
    long_rows = []

    for _, row in Z2_CT_MEG.iterrows():
        subject = row['participant_id']
        sex = row['sex']

        if level == "lobe":
            # Aggregate by lobe + hemisphere
            for lobe, regions in lobes_map.items():
                for hemi in ['lh', 'rh']:
                    # CT and MEG columns for this lobe + hemisphere
                    ct_cols = [f'cortthick-{hemi}-{r}' for r in regions if f'cortthick-{hemi}-{r}' in row]
                    meg_cols = [f'{r}-{hemi}' for r in regions if f'{r}-{hemi}' in row]
                    if ct_cols and meg_cols:
                        ct_avg = row[ct_cols].mean()
                        meg_avg = row[meg_cols].mean()
                        # lobe + hemi combined label
                        lobe_hemi = f"{lobe}_{hemi}"
                        long_rows.append({
                            'subject': subject,
                            'sex': sex,
                            'region': lobe_hemi,
                            'ct_z': ct_avg,
                            'meg_z': meg_avg
                        })
        elif level == "region":
            for col in row.index:
                if col.startswith("cortthick-"):
                    parts = col.split('-')
                    hemi = parts[1]
                    region = parts[2]
                    meg_col = f"{region}-{hemi}"
                    if meg_col in row:
                        long_rows.append({
                            'subject': subject,
                            'sex': sex,
                            'region': f"{region}_{hemi}",
                            'ct_z': row[col],
                            'meg_z': row[meg_col]
                        })

    df_long = pd.DataFrame(long_rows)

    # Assume your DataFrame 'df' has columns: subject, region, ct_z, gamma_z, sex (0/1)
    results = []

    # 1. PER-REGION OLS GLMs
    print("Running per-region OLS GLMs...")
    for region, subdf in df_long.groupby('region'):
        # Design matrix: intercept + ct_z + sex
        X = subdf[['ct_z', 'sex']]
        X = sm.add_constant(X)
        y = subdf['meg_z']

        model = sm.OLS(y, X).fit()

        results.append({
            'region': region,
            'beta_ct_ols': model.params['ct_z'],
            'p_ct_ols': model.pvalues['ct_z'],
            'beta_sex_ols': model.params['sex'],
            'p_sex_ols': model.pvalues['sex'],
            'r2_ols': model.rsquared
        })

    results_df = pd.DataFrame(results)

    # FDR correction for OLS p-values across regions
    _, results_df['p_ct_ols_fdr'], _, _ = multipletests(results_df['p_ct_ols'], method='fdr_bh')

    # 2. PER-REGION PARTIAL CORRELATIONS (CT vs gamma, controlling for sex)
    print("Running partial correlations...")
    partial_results = []
    for region, subdf in df_long.groupby('region'):
        # Partial correlation: ct_z vs gamma_z controlling for sex
        pcorr_result = partial_corr(data=subdf, x='ct_z', y='meg_z', covar='sex')

        partial_results.append({
            'region': region,
            'r_partial': pcorr_result['r'].iloc[0],
            'p_partial': pcorr_result['p-val'].iloc[0]
        })

    partial_df = pd.DataFrame(partial_results)

    # FDR correction for partial correlation p-values
    _, partial_df['p_partial_fdr'], _, _ = multipletests(partial_df['p_partial'], method='fdr_bh')

    # Combine results
    final_results = results_df.merge(partial_df, on='region')

    # Significant by OLS
    ols_sig = final_results[final_results['p_ct_ols_fdr'] < 0.05]

    # Significant by partial correlation
    partial_sig = final_results[final_results['p_partial_fdr'] < 0.05]

    print(f"\n=== {band} OLS Significant Regions (FDR < 0.05) ===")
    if len(ols_sig) > 0:
        print(ols_sig[['region', 'beta_ct_ols', 'p_ct_ols_fdr']].round(4))
    else:
        print("No significant regions")

    print(f"\n=== {band} Partial Correlation Significant Regions (FDR < 0.05) ===")
    if len(partial_sig) > 0:
        print(partial_sig[['region', 'r_partial', 'p_partial_fdr']].round(4))
    else:
        print("No significant regions")

    # Get your significant region
    ols_sig = final_results[final_results['p_ct_ols_fdr'] < 0.05]
    sig_region = ols_sig['region'].iloc[0]
    beta_value = ols_sig['beta_ct_ols'].iloc[0]
    fdr_p = ols_sig['p_ct_ols_fdr'].iloc[0]

    # Get the data for just that region
    sig_data = df_long[df_long['region'] == sig_region]

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=sig_data, x='ct_z', y='gamma_z', hue='sex', size='sex',
                    sizes=(100, 150), alpha=0.7, palette='Set1')

    # Add regression line
    plt.plot([sig_data['ct_z'].min(), sig_data['ct_z'].max()],
             [beta_value * sig_data['ct_z'].min(), beta_value * sig_data['ct_z'].max()],
             'red', linewidth=3, linestyle='--', label=f'Regression line\nβ = {beta_value:.3f}')

    plt.xlabel('Cortical Thickness Z-score', fontsize=14)
    plt.ylabel('Gamma Power Z-score', fontsize=14)
    plt.title(f'{sig_region}\nCT → Gamma association\nFDR-corrected p = {fdr_p:.3f}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

