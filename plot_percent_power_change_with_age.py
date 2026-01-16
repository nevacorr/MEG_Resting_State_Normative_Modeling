# This program loads the (pre-covid) normative model of MEG band power changes for all brain regions between 9 and 17 years of age
# and returns and plots the percent change in power throughout the cerebral cortex.

import os
import numpy as np
import pickle
import pandas as pd
from helper_functions_MEG import create_dummy_design_matrix_one_gender, read_ages_from_file
import myggseg
from helper_functions_plot_percent_power_change import get_model_values, calculate_ypred_percent_change, calculate_sig_of_change
from matplotlib import pyplot as plt
from bipolar import hotcold

# Set options
plot_model = 0
age_conversion_factor = 365.25
working_dir = os.getcwd()
save_dir = working_dir + '/plots'
data_type  = 'relative'
struct_var = 'meg'
spline_order = 1
spline_knots = 2
n_splits = 100

if data_type == 'absolute':
    data_dir = ''
elif data_type == 'relative':
    data_dir = 'output_data_bak'

allbands = ['theta', 'alpha', 'beta', 'gamma']
os.makedirs(f'{working_dir}/models_and_histograms', exist_ok=True)

for gender in ['male', 'female']:
    model_slopes_dict, model_ymin_dict = get_model_values(working_dir, data_dir, gender, n_splits)

    df_sig = pd.DataFrame()
    for band in allbands:
        df_sig = pd.concat([df_sig, calculate_sig_of_change(model_slopes_dict, band)])

    # Create figure with 1 row, 4 columns for bands
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes = axes.flatten()

    # Read agemin and agemax from file
    agemin, agemax = read_ages_from_file(struct_var, working_dir, gender)
    ages = np.array([agemin, agemax])

    for i, band in enumerate(allbands):
        ax = axes[i]
        all_pred_lines = []

        # Get a list of all region names by listing directories in model folder
        all_regions = df_sig.columns.tolist()
        all_regions.sort()
        total_reg_num = len(all_regions)

        # Initialize dictionaries to store percent change values
        change_dict = {}
        # Initialize list to store p value associated with slope
        pvalue_age = []

        for region in all_regions:

            slope, y_pred, all_pred_lines, pchange = (
                calculate_ypred_percent_change(band, region, agemin, agemax, model_slopes_dict, model_ymin_dict, all_pred_lines, ages))

            ax.plot(ages / age_conversion_factor, y_pred, alpha=0.3, linewidth=1, color='gray')

            if '-lh' in region:
                r = region.replace('-lh', '_left')
            else:
                r = region.replace('-rh', '_right')

            if df_sig.loc[band, region] != 0:
                change_dict[r] = pchange

            if plot_model:
                plot_model(gender, ages, age_conversion_factor, y_pred, band, region, pchange, slope, df_sig, model_slopes_dict)

        # Compute average of lines across all regions
        all_pred_lines = np.array(all_pred_lines)
        avg_line = all_pred_lines.mean(axis=0)

        # Plot average line
        ax.plot(ages / age_conversion_factor, avg_line, color='blue', linewidth=2, label='Average')

        # After finishing all regions for this band, finalize the superimposed plot
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Predicted Value')
        ax.set_title(
            f'{gender.capitalize()} — {band.capitalize()} Band')

        dict_to_plot = change_dict.copy()
        cmap = hotcold(neutral=0.0)
        filename = f'{gender.capitalize()} Regions with significant normative change with age in rsMEG {band} band'

        # Plot brain map
        myggseg.plot_dk(dict_to_plot, save_dir, filename, cmap=cmap, background='k', edgecolor='w', bordercolor='gray', vminmax=[-30, 30], figsize=(8,8),
                      title=f'{gender.capitalize()} Percent {band.capitalize()} Band Power Change in Regions with\nSignificant Normative Change From 9 to 17 Years of Age')

        # Write regions showing significant change with age to file
        with open(f'{os.path.join(working_dir, data_dir)}/{gender}_regions_showing_significant_change_with_age_precovid_{band}_band.txt',
                  'w') as file:
            for key in change_dict.keys():
                file.write(f'{key} {change_dict[key]}\n')

    fig.suptitle(f'{gender.capitalize()} - {data_type.capitalize()} Power Change With Age Across All Regions', fontsize=16)
    fig.savefig(os.path.join(save_dir, f'{os.path.join(working_dir, data_dir)}/{gender}_{data_type}_power_change_with_age.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)


mystop=1

