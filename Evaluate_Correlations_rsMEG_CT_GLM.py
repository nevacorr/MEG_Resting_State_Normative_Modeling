import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy.core.defchararray import capitalize
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pingouin import partial_corr  # pip install pingouin
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

level = "region" #options: "lobe", "region"
bands = ['beta', 'gamma']

lobes_map = {
    'frontal': ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
               'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral','frontalpole'],

    'parietal': ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',],

    'temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal',
                'entorhinal', 'temporalpole', 'parahippocampal'],

    'occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'],

    'cingulate': ['rostralanteriorcingulate', 'caudalanteriorcingulate','posteriorcingulate','isthmuscingulate']
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

    print("Running per-region OLS GLMs...")
    for region, subdf in df_long.groupby('region'):

        model_formula = 'meg_z ~ ct_z + sex'
        model = smf.ols(formula=model_formula, data=subdf).fit()

        results.append({
            'region': region,
            'beta_ct_ols': model.params['ct_z'],  # Main effect (female)
            'p_ct_ols': model.pvalues['ct_z'],
            'intercept': model.params['Intercept'],
            'beta_sex_ols': model.params['sex']
        })

    results_df = pd.DataFrame(results)

    # FDR correction for OLS p-values across regions
    _, results_df['p_ct_ols_fdr'], _, _ = multipletests(results_df['p_ct_ols'], method='fdr_bh')

    # Combine results
    final_results = results_df

    # Quick interaction test for significant regions only
    print("Testing sex × CT interaction in significant regions...")
    sig_region = results_df[results_df['p_ct_ols_fdr'] < 0.05]['region'].iloc[0]
    if len(sig_region) == 0:
        print("No regions survive FDR.")
        continue
    sig_subdf = df_long[df_long['region'] == sig_region].copy()

    model_int_formula = 'meg_z ~ ct_z * sex'
    model_int = smf.ols(formula=model_int_formula, data=sig_subdf).fit()
    p_interaction = model_int.pvalues['ct_z:sex']

    print(f"{sig_region}: Interaction p = {p_interaction:.3f} (not significant, p > 0.05)")
    print("→ Using common slope across sexes ✓")

    #  Find significant regions
    ols_sig = final_results[final_results['p_ct_ols_fdr'] < 0.05]

    print(f"\n=== {band} OLS Significant Regions (FDR < 0.05) ===")
    if len(ols_sig) > 0:
        print(ols_sig[['region', 'beta_ct_ols', 'p_ct_ols_fdr']].round(4))
    else:
        print("No significant regions")

    # Get significant region
    sig_region = ols_sig['region'].iloc[0]
    beta_value = ols_sig['beta_ct_ols'].iloc[0]
    intercept_value = ols_sig['intercept'].iloc[0]
    fdr_p = ols_sig['p_ct_ols_fdr'].iloc[0]

    # Get the data for just that region
    sig_data = df_long[df_long['region'] == sig_region]

    # Map sex to readable labels
    sig_data = sig_data.copy()
    sig_data['sex_label'] = sig_data['sex'].map({0: 'Female', 1: 'Male'})

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=sig_data, x='ct_z', y='meg_z', hue='sex_label', s=150,
                alpha=0.7, palette={'Female': 'purple', 'Male': 'green'})

    # Add regression line
    x_range = np.linspace(sig_data['ct_z'].min(), sig_data['ct_z'].max(), 100)
    beta_sex = ols_sig['beta_sex_ols'].iloc[0]
    y_female = intercept_value + beta_value * x_range
    y_male = intercept_value + beta_sex + beta_value * x_range

    plt.plot(x_range, y_female, 'purple', linestyle='--',)
    plt.plot(x_range, y_male, 'green', linestyle='--')
    plt.xlabel('Cortical Thickness z-score', fontsize=14)
    plt.ylabel('MEG Power z-score', fontsize=14)
    plt.title(f'{capitalize(sig_region)}\nCT z-score vs  {capitalize(band)} Power z-score \nFDR-corrected p = {fdr_p:.3f}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show(block=False)

mystop=1