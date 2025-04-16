# This program loads the (pre-covid) normative model of MEG band power changes for all brain regions between 9 and 17 years of age
# and returns and plots the percent change in power throughout the cerebral cortex.

import os
import numpy as np
import pickle
import pandas as pd
from helper_functions_MEG import create_dummy_design_matrix_one_gender, read_ages_from_file
from helper_functions_MEG import fit_regression_model_dummy_data_one_gender
# import ggseg
import myggseg
from joblib import load
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from scipy.stats import pearsonr
from scipy import stats
# from create_custom_colormap import create_custom_colormap
from bipolar import hotcold

# Set options
plot_model = 0
age_conversion_factor = 365.25
working_dir = os.getcwd()
save_dir = working_dir + '/plots'
struct_var = 'meg'
spline_order = 1
spline_knots = 2

for gender in ['male', 'female']:

    df_sig= pd.read_csv(f'{working_dir}/output_data/{gender}_significance of slopes by band and region.csv', index_col=0)
    df_sig = df_sig.astype(int)

    bands = df_sig.index.to_list()

    for bandnum, band in enumerate(bands):

        model_dir_path = f'{working_dir}/data/{gender}_{band}/ROI_models'

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

            # Create dummy covariate matrices with bspline values and save to file
            dummy_cov_file_path = create_dummy_design_matrix_one_gender(agemin, agemax, spline_order, spline_knots, working_dir)
            # Load dummy covariate matrix
            dummy_cov = np.loadtxt(dummy_cov_file_path)

            # remove last row which has erroneous bspline values
            dummy_cov = dummy_cov[:-1]

            # Open model parameter file
            with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as file:
                data = pickle.load(file)

            # Calculate predictions from model based on covariate data
            y_pred = np.dot(dummy_cov, data.blr.m)

            # y_pred = y_pred * minmax_scaler.data_range_[regnum + (bandnum * total_reg_num)] + minmax_scaler.data_min_[regnum + (bandnum * total_reg_num)]

            # calculate percent change with age this brain region
            pchange = (y_pred[-1] - y_pred[0]) / y_pred[0] * 100.00

            if '-lh' in region:
                r = region.replace('-lh', '_left')
            else:
                r = region.replace('-rh', '_right')

            # Convert covariate and y values back to unscaled space
            # dummy_cov[:,0] = dummy_cov[:,0] * minmax_scaler.data_range_[-1] + minmax_scaler.data_min_[-1]

            if df_sig.loc[band, region] != 0:
                change_dict[r] = pchange

            if plot_model:
                plt.figure()
                if gender == 'male':
                    c = 'b'
                else:
                    c = 'crimson'
                # plot model for this brain region
                plt.plot(dummy_cov[:,0]/age_conversion_factor, y_pred, c)
                plt.ylim([0, 40])
                plt.title(f'Regions with Change in MEG power for {band} band in region {region}\n{gender} percent change = {pchange:.1f} sig change={df_sig.loc[band, region]}')
                plt.show()

        dict_to_plot = change_dict.copy()

        cmap = hotcold(neutral=0.0)

        filename = f'{gender.capitalize()} Regions with significant normative change with age in rsMEG {band} band'
        myggseg.plot_dk(dict_to_plot, save_dir, filename, cmap=cmap, background='k', edgecolor='w', bordercolor='gray', vminmax=[-100, 100], figsize=(8,8),
                      title=f'{gender.capitalize()} Percent {band.capitalize()} Band Power Change in Regions with\nSignificant Normative Change From 9 to 17 Years of Age')

        # Write regions showing significant change with age to file
        with open(f'{working_dir}/output_data/{gender}_regions_showing_significant_change_with_age_precovid_{band}_band.txt',
                  'w') as file:
            for key in change_dict.keys():
                file.write(f'{key} {change_dict[key]}\n')

plt.show()
mystop=1

