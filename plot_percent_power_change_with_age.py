# This program loads the (pre-covid) normative model of MEG band power changes for all brain regions between 9 and 17 years of age
# and returns and plots the percent change in power throughout the cerebral cortex.

import os
import numpy as np
import pickle
import pandas as pd
from helper_functions_MEG import create_dummy_design_matrix_one_gender, read_ages_from_file
import myggseg
from matplotlib import pyplot as plt
from bipolar import hotcold

# Set options
plot_model = 0
age_conversion_factor = 365.25
working_dir = os.getcwd()
save_dir = working_dir + '/plots'
data_type  = 'absolute'
struct_var = 'meg'
spline_order = 1
spline_knots = 2
n_splits = 100

if data_type == 'absolute':
    data_dir = 'output_data_rs_abs_24Oct2025'
elif data_type == 'relative':
    data_dir = 'output_data_rs_rel_22Oct2025'

os.makedirs(f'{working_dir}/models_and_histograms', exist_ok=True)

for gender in ['male', 'female']:

    model_slope = pd.read_csv(os.path.join
                              (working_dir, os.path.join(working_dir, data_dir), f'{gender}_{n_splits}_splits_allsplits_slopes.csv'))

    model_ymin = pd.read_csv(os.path.join
                              (working_dir, os.path.join(working_dir, data_dir), f'{gender}_{n_splits}_splits_ymin.csv'))

    if data_type == "absolute":
        model_ymin = model_ymin / 100.0

    model_slope.rename(columns={'Unnamed: 0': 'band'}, inplace=True)
    model_ymin.rename(columns={'Unnamed: 0': 'band'}, inplace=True)

    model_slopes_dict = {band: band_df.drop(columns=['band', 'split']) for band, band_df in model_slope.groupby('band')}
    model_ymin_dict = {band: band_df.drop(columns=['band', 'split']) for band, band_df in model_ymin.groupby('band')}

    df_sig = pd.DataFrame()

    for band in ['theta', 'alpha', 'beta', 'gamma']:

        # Calculate confidence intervals
        for reg in model_slopes_dict[band].columns:
            slopes_reg = model_slopes_dict[band].loc[:, reg]
            lower_bound, upper_bound = np.percentile(slopes_reg.to_numpy(), [2.5, 97.5])

            if lower_bound < 0 < upper_bound:
                df_sig.loc[band, reg] = 0
            else:
                df_sig.loc[band, reg] = 1


    df_sig = df_sig.astype(int)

    bands = df_sig.index.to_list()

    for bandnum, band in enumerate(bands):

        # Get a list of all region names by listing directories in model folder
        all_regions = df_sig.columns.tolist()
        all_regions.sort()
        total_reg_num = len(all_regions)

        # Initialize dictionaries to store percent change values
        change_dict = {}
        # Initialize list to store p value associated with slope
        pvalue_age = []

        for regnum, region in enumerate(all_regions):

            model_path = (f'/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                          f'Modeling/data/{gender}_{band}/ROI_models/{region}/Models/')

            # Read agemin and agemax from file
            agemin, agemax = read_ages_from_file(struct_var, working_dir, gender)

            slope = model_slopes_dict[band][region].mean()

            ymin =  model_ymin_dict[band][region].mean()

            # Create array of ages
            ages = np.array([agemin, agemax])

            # Compute predicted values using the line equation
            y_pred= ymin + slope * (ages - agemin)

            # Plot
            # plt.plot(ages/age_conversion_factor, y_pred, label='Predicted line', color='blue')
            # plt.xlabel('Age')
            # plt.ylabel('Predicted Value')
            # plt.title('Predicted Value vs Age')
            # plt.legend()
            # plt.show()

            pchange = slope/ymin* (agemax-agemin)* 100.0

            if '-lh' in region:
                r = region.replace('-lh', '_left')
            else:
                r = region.replace('-rh', '_right')

            if df_sig.loc[band, region] != 0:
                change_dict[r] = pchange

            if plot_model:
                if gender == 'male':
                    c = 'b'
                else:
                    c = 'crimson'
                # plot model for this brain region
                fig = plt.figure(figsize=(12, 6))
                plt.subplot(1,2,1)
                plt.plot(ages/age_conversion_factor, y_pred, c)
                plt.ylim([0, 40])
                plt.title(f'Regions with Change in MEG power for\n {band} band in region {region}\n{gender} percent change = {pchange:.1f}\nslope = {slope * age_conversion_factor:.3f} sig change={df_sig.loc[band, region]}')
                plt.tight_layout()

                plt.subplot(1, 2, 2)
                slopes_reg = model_slopes_dict[band].loc[:, region]
                lower_bound, upper_bound = np.percentile(slopes_reg.to_numpy(), [2.5, 97.5])
                plt.hist(slopes_reg * age_conversion_factor)
                plt.plot([lower_bound * age_conversion_factor, lower_bound * age_conversion_factor], [0, 20], 'r')
                plt.plot([upper_bound * age_conversion_factor, upper_bound * age_conversion_factor], [0, 20], 'r')
                plt.plot([0, 0], [0, 20], 'k')
                plt.title(
                    f'{band} {gender} {region} sig = {df_sig.loc[band, region]}\nHistogram of slopes\nlower_bound = {lower_bound * age_conversion_factor:.2f} upper_bound = {upper_bound * age_conversion_factor:.2f}')
                plt.tight_layout()
                plt.show()
                # fname = os.path.join(working_dir, 'models_and_histograms', f'{band}_{gender}_{region}_plot_of_model_with_slope_and_histogram_of_slopes.png')
                # plt.savefig(fname)
                # plt.close()

            print(f'{band} {region} {gender} slope = {slope*age_conversion_factor: .3f} percent change = {pchange:.1f} sig change = {df_sig.loc[band, region]}')

        dict_to_plot = change_dict.copy()

        cmap = hotcold(neutral=0.0)

        filename = f'{gender.capitalize()} Regions with significant normative change with age in rsMEG {band} band'
        myggseg.plot_dk(dict_to_plot, save_dir, filename, cmap=cmap, background='k', edgecolor='w', bordercolor='gray', vminmax=[-30, 30], figsize=(8,8),
                      title=f'{gender.capitalize()} Percent {band.capitalize()} Band Power Change in Regions with\nSignificant Normative Change From 9 to 17 Years of Age')

        # Write regions showing significant change with age to file
        with open(f'{os.path.join(working_dir, data_dir)}/{gender}_regions_showing_significant_change_with_age_precovid_{band}_band.txt',
                  'w') as file:
            for key in change_dict.keys():
                file.write(f'{key} {change_dict[key]}\n')

plt.show()

mystop=1

