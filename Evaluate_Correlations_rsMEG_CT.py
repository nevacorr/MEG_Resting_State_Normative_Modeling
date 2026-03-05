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

level = "lobe" #options: "lobe", "region"
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
                            'lobe_hemi': lobe_hemi,
                            'CT': ct_avg,
                            'MEG': meg_avg
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
                            'region_hemi': f"{region}_{hemi}",
                            'CT': row[col],
                            'MEG': row[meg_col]
                        })

    df_long = pd.DataFrame(long_rows)

    # ---------------------------
    # Within-subject centering of MEG
    # ---------------------------
    df_long['MEG_within'] = df_long.groupby('subject')['MEG'].transform(lambda x: x - x.mean())

    # # ---------------------------
    # # Fit mixed model
    # # ---------------------------
    # model = smf.mixedlm(
    #     "CT_abs ~ 0 + MEG_within * C(lobe_hemi)",  # fixed effect includes left/right lobes
    #     lobe_df,
    #     groups=lobe_df["subject"],  # random intercept per subject
    # )
    group_col = 'subject'
    if level == "lobe":
            formula = "CT ~ 0 + MEG_within * C(lobe_hemi)"
    else:
            formula = "CT ~ 0 + MEG_within * C(region_hemi)"

    model = smf.mixedlm(
            formula,  # fixed effect
            df_long,
            groups=df_long[group_col],  # random intercept per subject
            re_formula="1+MEG_within"  # add random slopes
        )
    fitted_model = model.fit()
    print(f"\n{capitalize(band)} Band ({capitalize(level)})")
    print(fitted_model.summary())

    # ---------------------------
    # Plot fixed-effect slopes
    # ---------------------------
    fe_params = fitted_model.fe_params
    key_col = 'lobe_hemi' if level == "lobe" else 'region_hemi'

    names = [c.split('[')[1][:-1] for c in fe_params.index if c.startswith(f'C({key_col})')]

    # Base slope for reference lobe
    base_slope = fe_params['MEG_within']

    # Build fixed-effect intercepts and slopes per lobe
    fixed_effects = {}
    for name in names:
        intercept = fe_params[f'C({key_col})[{name}]']
        interaction_term = f'MEG_within:C({key_col})[T.{name}]'
        slope = base_slope + fe_params[interaction_term] if interaction_term in fe_params else base_slope

        # Compute standard error for the slope
        if interaction_term in fitted_model.bse:
            se_slope = np.sqrt(fitted_model.bse['MEG_within'] ** 2 + fitted_model.bse[interaction_term] ** 2)
        else:
            se_slope = fitted_model.bse['MEG_within']

        # z-score and p-value
        z_slope = slope / se_slope
        p_slope = 2 * (1 - stats.norm.cdf(abs(z_slope)))

        fixed_effects[name] = {'intercept': intercept, 'slope': slope, 'pval': p_slope}

    plt.figure(figsize=(10, 6))
    x = np.linspace(df_long['MEG_within'].min(), df_long['MEG_within'].max(), 100)

    # Plot fixed-effect lines
    for name, params in fixed_effects.items():
        if params['pval'] < 0.05:  # only significant lines
            y = params['intercept'] + params['slope'] * x
            plt.plot(x, y, label=name, linewidth=2)

    # Overlay individual subject points lightly
            sub_df = df_long[df_long[key_col] == name]
            plt.scatter(sub_df['MEG_within'], sub_df['CT'], alpha=0.3, color='gray')

    plt.xlabel('Z Resting State MEG power')
    plt.ylabel('Z Cortical Thickness')
    plt.title(f'{capitalize(band)} band: Fixed-effect slopes {level}')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=7)
    # plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    mystop=1

