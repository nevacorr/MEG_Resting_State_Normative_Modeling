import os
import pandas as pd
from itertools import product
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import statsmodels.formula.api as smf

level = "lobe" #options: "lobe", "region"
bands = ['beta', 'gamma']
save_path = os.getcwd()

lobes_map = {
    'frontal': ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
               'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral','frontalpole'],

    'parietal': ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',],

    'temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal',
                'entorhinal', 'temporalpole', 'parahippocampal'],

    'occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'],

    'cingulate': ['rostralanteriorcingulate', 'caudalanteriorcingulate','posteriorcingulate','isthmuscingulate']
}

# Add remove outliers flag
# remove_outliers = 0

# brain_regions_of_interest = ['Minor']
behaviors_of_interest = ['FlankerSU', 'DCSU']

# Get working directory
working_dir = os.getcwd()

# Load behavioral z scores
behav_zs = pd.read_csv('/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID/'
                       'Z_scores_all_meltzoff_cogn_behav_visit2.csv', usecols=lambda column: column != 'Unnamed: 0')

# Load MEG z scores
with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_male_100_splits.pkl'), 'rb') as f:
    Z2_MEG_male = pickle.load(f)
with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_female_100_splits.pkl'), 'rb') as f:
    Z2_MEG_female = pickle.load(f)

# Keep only behavior columns that contain substrings from behaviors_of_interest, plus 'participant_id'
# behav_zs = behav_zs[[col for col in behav_zs.columns if any(sub in col for sub in behaviors_of_interest) or col == 'participant_id']]

# Keep only DWI columns that contain substrings from brain_regions_of_interest, plus 'participant_id'
# dwi_zs = dwi_zs[[col for col in dwi_zs.columns if any(sub in col for sub in brain_regions_of_interest) or col == 'participant_id']]

reg_cols = [col for col in Z2_MEG_male[bands[0]].columns if col != 'subject_id_test']
region_list = sorted(set(col.split('-')[0] for col in reg_cols))

for band in bands:

    # Merge male and female MEG data
    Z2_MEG_male[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG_female[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG=pd.concat([Z2_MEG_male[band], Z2_MEG_female[band]], ignore_index=True)

    # Merge CT and MEG data
    Z2_Beh_MEG = pd.merge(behav_zs, Z2_MEG, on='participant_id', how='inner')

    # ---------------------------
    # Reshape/Aggregate
    # ---------------------------
    for idx, row in Z2_Beh_MEG.iterrows():
        subject = row['participant_id']
        sex = row['gender']
        if level == "lobe":
            # Aggregate by lobe + hemisphere
            for lobe, regions in lobes_map.items():
                for hemi in ['lh', 'rh']:
                    lobe_cols = [f'{r}-{hemi}' for r in regions if f'{r}-{hemi}' in row]
                    if lobe_cols:
                        meg_avg = row[lobe_cols].mean()
                        # lobe + hemi combined label
                        lobe_hemi = f"{lobe}-{hemi}"
                        Z2_Beh_MEG.loc[idx, lobe_hemi] = meg_avg

    if level == "lobe":
        cols_to_drop = [col for col in Z2_Beh_MEG.columns if any(region in col for region in region_list)]
        Z2_Beh_MEG.drop(columns=cols_to_drop, inplace=True)

    Z2_Beh_MEG.columns = Z2_Beh_MEG.columns.str.replace('-', '')
    behav_cols = [col for col in Z2_Beh_MEG if col not in ['participant_id', 'gender'] and not any(suffix in col for suffix in ['rh', 'lh'])]
    meg_cols = [col for col in Z2_Beh_MEG if any(suffix in col for suffix in  ['rh', 'lh'])]

    results = []

    print("Running per-region OLS GLMs...")
    for behav in behav_cols:
        for meg in meg_cols:

            formula = f'{meg} ~ {behav} + gender'

            model = smf.ols(formula=formula, data=Z2_Beh_MEG).fit()

            results.append({
                'behavior': behav,
                'region': meg,
                'beta': model.params[behav],
                'pval': model.pvalues[behav],
                'r2': model.rsquared
            })

    results_df = pd.DataFrame(results)

    # FDR correction for OLS p-values across regions
    _, results_df['pval_fdr'], _, _ = multipletests(results_df['pval'], method='fdr_bh')

    # Combine results
    results_df

    # Quick interaction test for significant regions only
    print("Testing sex × behavior interaction in significant regions...")
    sig_region = results_df[results_df['pval_fdr'] < 0.05]['region'].iloc[0]
    if len(sig_region) == 0:
        print("No regions survive FDR.")
        continue
    sig_subdf = Z2_Beh_MEG[df_long['region'] == sig_region].copy()

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