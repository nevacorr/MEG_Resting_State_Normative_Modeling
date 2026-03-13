import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical
import matplotlib.pyplot as plt
import os
import pickle


# Manual permutation test function (works on any SciPy)
def pls_permutation_test(X, Y, n_components=1, n_perm=1000, random_state=42):
    """Permutation test for PLS canonical correlation"""
    np.random.seed(random_state)

    # Original fit
    pls_orig = PLSCanonical(n_components=n_components).fit(X, Y)
    orig_score = pls_orig.score(X, Y)  # Sum of squared canonical correlations

    # Permutation distribution
    perm_scores = np.zeros(n_perm)
    n_subjects = X.shape[0]

    for i in range(n_perm):
        # Permute subjects (rows) in X only - breaks brain-behavior mapping
        perm_idx = np.random.permutation(n_subjects)
        X_perm = X[perm_idx]

        pls_perm = PLSCanonical(n_components=n_components).fit(X_perm, Y)
        perm_scores[i] = pls_perm.score(X_perm, Y)

    # Two-tailed p-value
    p_val = np.mean(np.abs(perm_scores) >= np.abs(orig_score))
    return orig_score, p_val, perm_scores


# === MAIN ANALYSIS ===
# Load/prepare data
bands = ['beta', 'gamma']
save_path = os.getcwd()

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

# Load cortical thickness z scores
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
Z_time2_CT = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                         .format(ct_data_dir, 'cortthick'))

for band in bands:

    # Merge male and female MEG data
    Z2_MEG_male[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG_female[band].rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
    Z2_MEG=pd.concat([Z2_MEG_male[band], Z2_MEG_female[band]], ignore_index=True)

    # Merge behavior and MEG data
    Z2_Beh_MEG = pd.merge(behav_zs, Z2_MEG, on='participant_id', how='inner')

    Z2_Beh_MEG.columns = Z2_Beh_MEG.columns.str.replace('-', '')
    behavior_cols = [col for col in Z2_Beh_MEG if col not in ['participant_id', 'gender'] and not any(suffix in col for suffix in ['rh', 'lh'])]
    behavior_cols = [col for col in behavior_cols if
                     col not in ['FlankerSU', 'WMemorySU', 'DCSU', 'VocabSU', 'peermindset', 'persmindset', 'needforapproval', 'needforbelonging']]
    # behavior_cols = ['FlankerSU', 'DCSU', 'WMemorySU', 'VocabSU']
    meg_cols = [col for col in Z2_Beh_MEG if any(suffix in col for suffix in  ['rh', 'lh'])]

    # X: 68 MEG regions (beta band)
    X = Z2_Beh_MEG[meg_cols].values

    # Y: behaviors
    Y = Z2_Beh_MEG[behavior_cols].values

    # Center/scale
    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)

    # Fit PLS
    pls = PLSCanonical(n_components=3)
    pls.fit(X, Y)

    # Permutation test LV1
    score, p_val, null_dist = pls_permutation_test(X, Y, n_components=1, n_perm=1000)
    print(f'{band} Band')
    print(f"LV1: score={score:.3f}, p={p_val:.3f}")

    brain_scores = X @ pls.x_weights_[:, 0]
    behav_scores = Y @ pls.y_weights_[:, 0]
    r_val = np.corrcoef(brain_scores, behav_scores)[0, 1]

    print(f"LV1 brain-behavior correlation: r = {r_val:.3f}")
    print(f"Shared variance: r² = {r_val ** 2:.3f}")

    # Results
    print("\nTop regions (LV1 brain weights):")
    brain_weights = np.abs(pls.x_weights_[:, 0])
    top_regions = np.argsort(brain_weights)[-5:][::-1]
    for r in top_regions:
        print(f"  Region {meg_cols[r]}: {brain_weights[r]:.3f}")

    print("\nBehavior weights (LV1):")
    for b, w in enumerate(np.abs(pls.y_weights_[:, 0])):
        print(f"  {behavior_cols[b]}: {w:.3f}")

    if band == 'gamma':
        # Calculate subject scores using original data + weights
        # brain_scores = X @ pls.x_weights_[:, 0]  # (71,68) @ (68,) = (71,)
        # behav_scores = Y @ pls.y_weights_[:, 0]  # (71,11) @ (11,) = (71,)
        # Create the scatterplot
        n = brain_scores.shape[0]
        plt.figure(figsize=(6, 5))
        plt.scatter(brain_scores, behav_scores, alpha=0.7, s=60, color='steelblue')

        slope, intercept = np.polyfit(brain_scores, behav_scores, 1)
        x_line = np.linspace(brain_scores.min(), brain_scores.max(), 100)
        plt.plot(x_line, slope * x_line + intercept, color='grey', lw=2.5)

        # Labels and title
        plt.xlabel('Gamma Power LV1 Brain Scores', fontsize=11)
        plt.ylabel('LV1 Emotion Scores', fontsize=11)
        r = np.corrcoef(brain_scores, behav_scores)[0,1]
        plt.title(f'Gamma PLS LV1: Brain-Behavior Relationship\n(r={r:.3f}, p ={p_val})', fontsize=12, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        # Top 10 brain regions by absolute weight
        top_regions = np.argsort(np.abs(pls.x_weights_[:, 0]))[-10:][::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(range(10), np.abs(pls.x_weights_[top_regions, 0]))
        plt.yticks(range(10), [meg_cols[i] for i in top_regions])
        plt.xlabel('|Brain Weight| (LV1)')
        plt.title('Top Gamma Brain Regions Driving LV1')
        plt.tight_layout()
        plt.show()

        # Sort emotion behaviors by weight
        plt.figure(figsize=(8, 6))
        weights = np.abs(pls.y_weights_[:, 0])
        top_behav = np.argsort(weights)[-8:][::-1]
        plt.barh(range(8), weights[top_behav])
        plt.yticks(range(8), [behavior_cols[i] for i in top_behav])
        plt.xlabel('|Behavior Weight| (LV1)')
        plt.title('Top Emotion Measures Driving LV1')
        plt.tight_layout()
        plt.show()

        # Plot permutation histogram
        plt.figure(figsize=(6, 4))
        plt.hist(null_dist, bins=30, alpha=0.7, color='lightgray')  # null_dist = permuted scores
        plt.axvline(score, color='black', lw=3, label='Observed LV1 = -2.232')  # score = observed
        plt.xlabel('PLS LV1 Covariance Score')
        plt.ylabel('Frequency')
        plt.title('Permutation Test (p=0.069)')
        plt.legend()
        plt.show()


