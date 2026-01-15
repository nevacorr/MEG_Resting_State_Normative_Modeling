import os
import pandas as pd
import matplotlib.pyplot as plt
import myggseg
import pickle
from matplotlib.colors import ListedColormap

from helper_functions_MEG import read_text_list

working_dir = os.getcwd()
save_dir = working_dir + '/plots'

bands = ['theta','alpha', 'beta', 'gamma']
sexes = ['female', 'male']

regions_sig_change_with_age = {}

for sex in sexes:
    regions_sig_change_with_age[sex] = {}
    for band in bands:
        path = os.path.join(working_dir, "output_data_bak", f"{sex}_regions_showing_significant_change_with_age_precovid_{band}_band.txt")
        df = pd.read_csv(path, sep=r"\s+", header=None)
        # Take only the first column (region names)
        region_list = df.iloc[:, 0].tolist()
        regions_sig_change_with_age[sex][band] = region_list

Z2={}

with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_female_100_splits.pkl'), 'rb') as f:
    Z2['female'] = pickle.load(f)

with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_male_100_splits.pkl'), 'rb') as f:
    Z2['male'] = pickle.load(f)

Z2_mean = {}

for sex in Z2:
    mean_dict = {}
    for band, df in Z2[sex].items():
        # Drop 'subject_id_test' if it exists
        if 'subject_id_test' in df.columns:
            df = df.drop(columns='subject_id_test')

        # Compute mean across rows (subjects)
        mean_dict[band] = df.mean(axis=0)

    # Convert to dataframe: rows = bands, columns = brain regions
    Z2_mean[sex] = pd.DataFrame(mean_dict).T

regions_reject_female_file = f'{working_dir}/output_data/regions_reject_null_rsMEG_female.csv'
regions_female = read_text_list(regions_reject_female_file)

regions_reject_male_file = f'{working_dir}/output_data/regions_reject_null_rsMEG_male.csv'
regions_male = read_text_list(regions_reject_male_file)

filtered = {}

for sex in ['female', 'male']:
    original_list = regions_female if sex == 'female' else regions_male
    new_list = []

    for item in original_list:
        # Split into "region-hemi" and "band"
        region_with_hemi, band = item.rsplit("_", 1)  # e.g., "precuneus-lh", "theta"

        # Map hemisphere to match dictionary
        if region_with_hemi.endswith("-lh"):
            region_for_match = region_with_hemi[:-3] + "_left"   # remove "-lh", add "_left"
        elif region_with_hemi.endswith("-rh"):
            region_for_match = region_with_hemi[:-3] + "_right"  # remove "-rh", add "_right"

        # Check if region is in significant-change list for that band
        if region_for_match in regions_sig_change_with_age[sex][band]:
            new_list.append(item)  # keep original format

    filtered[sex] = new_list

regions_female = filtered['female']
regions_male   = filtered['male']

r = {}
female_dict = {}
male_dict = {}

# Define the colors for each integer value
colors = ['steelblue', 'yellow']
cmap = ListedColormap(colors)

for sex in sexes:
    if sex == 'female':
        regions_list = regions_female
        mydict = female_dict
    elif sex == 'male':
        regions_list = regions_male
        mydict = male_dict
    for i, band in enumerate(bands):
        r[band] = [item for item in regions_list if band in item]

        mydict[band] = {}

        for reg in r[band]:
            region = reg.replace(f'_{band}', '')
            region_noband = region
            if '-lh' in region:
                region = region.replace('-lh', '_left')
            else:
                region = region.replace('-rh', '_right')
            if Z2_mean[sex].loc[band, region_noband] > 0:
                mydict[band][region] = 2
            else:
                mydict[band][region] = 1

        dict_to_plot = mydict[band].copy()

        filename = f'{sex.capitalize()} Regions with significantly altered power in post-COVID rsMEG {band} band_sig_change_with_age_only'
        myggseg.plot_dk(dict_to_plot, save_dir, filename, cmap=cmap, background='k', vminmax=[0, 3], edgecolor='w', bordercolor='gray', figsize=(8,8),
                          title=f'{sex.capitalize()} Regions with Significantly Increased Power in\nPost-COVID rsMEG {band.capitalize()} Band')

        plt.show(block=False)
mystop=1