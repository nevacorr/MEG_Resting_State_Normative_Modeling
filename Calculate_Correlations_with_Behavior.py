import os
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pickle
import statsmodels.formula.api as smf
import seaborn as sns

interaction=True
level = "region" #options: "lobe", "region"
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

# Get brain regions for MEG
reg_cols = [col for col in Z2_MEG_male[bands[0]].columns if col != 'subject_id_test']
region_list = sorted(set(col.split('-')[0] for col in reg_cols))

for band in bands:

    # Merge male and female MEG data
    Z2_MEG_male[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG_female[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG=pd.concat([Z2_MEG_male[band], Z2_MEG_female[band]], ignore_index=True)

    # Merge CT and MEG data
    Z2_Beh_MEG = pd.merge(behav_zs, Z2_MEG, on='participant_id', how='inner')

    # If lobe level, average data across regions in lobe
    if level == "lobe":
        for idx, row in Z2_Beh_MEG.iterrows():
            subject = row['participant_id']
            sex = row['gender']
            for lobe, regions in lobes_map.items():
                for hemi in ['lh', 'rh']:
                    lobe_cols = [f'{r}-{hemi}' for r in regions if f'{r}-{hemi}' in row]
                    if lobe_cols:
                        meg_avg = row[lobe_cols].mean()
                        # lobe + hemi combined label
                        lobe_hemi = f"{lobe}-{hemi}"
                        Z2_Beh_MEG.loc[idx, lobe_hemi] = meg_avg
        # Keep lobe columns and remove individual regions columns
        cols_to_drop = [col for col in Z2_Beh_MEG.columns if any(region in col for region in region_list)]
        Z2_Beh_MEG.drop(columns=cols_to_drop, inplace=True)

    Z2_Beh_MEG.columns = Z2_Beh_MEG.columns.str.replace('-', '')
    behav_cols = [col for col in Z2_Beh_MEG if col not in ['participant_id', 'gender'] and not any(suffix in col for suffix in ['rh', 'lh'])]
    # meg_cols = [col for col in Z2_Beh_MEG if any(suffix in col for suffix in  ['rh', 'lh'])]
    meg_cols = [col for col in Z2_Beh_MEG if any(name in col for name in ['posteriorcingulaterh'])]

    results = []

    print("\nRunning per-region OLS GLMs...")
    for behav in behav_cols:
        for meg in meg_cols:
            if not interaction:
                formula = f'{meg} ~ {behav} + gender'
                model = smf.ols(formula=formula, data=Z2_Beh_MEG).fit()
                results.append({
                    'behavior': behav,
                    'region': meg,
                    'beta': model.params[behav],
                    'pval': model.pvalues[behav],
                })
            else:
                formula = f'{meg} ~ {behav} * gender'
                model = smf.ols(formula=formula, data=Z2_Beh_MEG).fit()
                results.append({
                    'behavior': behav,
                    'region': meg,
                    'beta': model.params[behav],
                    'pval': model.pvalues[behav],
                    'pval_behavxsex': model.pvalues[f'{behav}:gender'],
                })

    results_df = pd.DataFrame(results)

    # FDR correction for OLS p-values across regions
    _, results_df['pval_fdr'], _, _ = multipletests(results_df['pval'], method='fdr_bh')
    #  Find significant regions
    ols_sig = results_df[results_df['pval'] < 0.05]
    print(f"\n=== {band} OLS Significant Slope Regions (p < 0.05) ===")
    if len(ols_sig) > 0:
        print(ols_sig[['behavior', 'region', 'beta', 'pval', 'pval_fdr']].round(4))
    else:
        print("No significant slope in any regions")

    if interaction:
        # FDR Correction for pvalue for interaction
        _, results_df['pval_int_fdr'], _, _ = multipletests(results_df['pval_behavxsex'], method='fdr_bh')
        #  Find significant interactions
        ols_sig_int = results_df[results_df['pval_behavxsex'] < 0.05]
        print(f"\n=== {band} OLS Significant Interaction Regions (p < 0.05) ===")
        if len(ols_sig_int) > 0:
            print(ols_sig_int[['behavior', 'region', 'pval', 'pval_behavxsex', 'pval_int_fdr']].round(4))
        else:
            print("No significant interactions in any regions")

        # Female-only model
        model_f = smf.ols(formula=f'{meg} ~ RSQanxiety', data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 0]).fit()
        print("Female-only model")
        print(model_f.summary())

        # Male-only model
        model_m = smf.ols(formula=f'{meg} ~ RSQanxiety', data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 1]).fit()
        print("Male-only model")
        print(model_m.summary())

        # Plot MEG vs RSQanxiety by sex
        plt.figure(figsize=(6, 5))

        sns.scatterplot(
            x='RSQanxiety',
            y=meg,
            hue='gender',
            style='gender',
            data=Z2_Beh_MEG,
            palette={0: 'purple', 1: 'green'},
            s=60
        )

        # Add regression lines for each sex
        sns.regplot(
            x='RSQanxiety',
            y=meg,
            data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 0],
            scatter=False,
            ci=False,
            color='purple'
        )
        sns.regplot(
            x='RSQanxiety',
            y=meg,
            data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 1],
            scatter=False,
            ci=False,
            color='green'
        )

        plt.xlabel('RSQ Anxiety')
        plt.ylabel(meg)
        plt.title(f'{band} {meg} vs RSQ Anxiety by Sex')
        plt.legend(title='Gender', labels=['Male', 'Female'])
        plt.tight_layout()
        plt.show()
        results_sex = []

        for behav in behav_cols:
            # Female model
            model_f = smf.ols(
                formula=f'{meg} ~ {behav}',
                data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 0]
            ).fit()

            # Male model
            model_m = smf.ols(
                formula=f'{meg} ~ {behav}',
                data=Z2_Beh_MEG[Z2_Beh_MEG['gender'] == 1]
            ).fit()

            results_sex.append({
                'behavior': behav,
                'beta_female': model_f.params[behav],
                'pval_female': model_f.pvalues[behav],
                'beta_male': model_m.params[behav],
                'pval_male': model_m.pvalues[behav]
            })

            results_sex_df = pd.DataFrame(results_sex)

        # Female correction
        _, results_sex_df['pval_female_fdr'], _, _ = multipletests(
            results_sex_df['pval_female'],
            method='fdr_bh'
        )

        # Male correction
        _, results_sex_df['pval_male_fdr'], _, _ = multipletests(
            results_sex_df['pval_male'],
            method='fdr_bh'
        )

        sig_male = results_sex_df[results_sex_df['pval_male'] < 0.05]
        sig_female = results_sex_df[results_sex_df['pval_female'] < 0.05]

        print("Significant behaviors in males:")
        print(sig_male)

        print("Significant behaviors in females:")
        print(sig_female)

        mystop=1