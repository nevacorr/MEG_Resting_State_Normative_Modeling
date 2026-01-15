import os
import pandas as pd
import matplotlib.pyplot as plt
import myggseg
import pickle
from matplotlib.colors import ListedColormap

from helper_functions_MEG import read_text_list

working_dir = os.getcwd()
save_dir = working_dir + '/plots'

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

r = {}
female_dict = {}
male_dict = {}

sexes = ['female', 'male']

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
    for i, band in enumerate(['theta','alpha', 'beta', 'gamma']):
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

        filename = f'{sex.capitalize()} Regions with significantly altered power in post-COVID rsMEG {band} band'
        myggseg.plot_dk(dict_to_plot, save_dir, filename, cmap=cmap, background='k', vminmax=[0, 3], edgecolor='w', bordercolor='gray', figsize=(8,8),
                          title=f'{sex.capitalize()} Regions with Significantly Increased Power in\nPost-COVID rsMEG {band.capitalize()} Band')

        plt.show(block=False)
mystop=1