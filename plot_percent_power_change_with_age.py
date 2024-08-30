# This program loads the (pre-covid) normative model of MEG band power changes for all brain regions between 9 and 17 years of age
# and returns and plots the percent change in power throughout the cerebral cortex.

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from helper_functions_MEG import create_dummy_design_matrix, read_ages_from_file
import ggseg
from joblib import load
from matplotlib import pyplot as plt

age_conversion_factor = 365.25
working_dir = os.getcwd()
bands = ['theta', 'alpha', 'beta', 'gamma']

# Set some options
plot_model = 0
struct_var = 'meg'
spline_order = 1
spline_knots = 2

minmax_scaler = load(f'{working_dir}/minmax_scaler.bin')

for band in bands:

    model_dir_path = f'{working_dir}/data/{band}/ROI_models'

    # Get a list of all region names by listing directories in model folder
    all_regions = [d for d in os.listdir(model_dir_path)
                   if os.path.isdir(os.path.join(model_dir_path, d))]
    all_regions.sort()

    # Initialize dictionaries to store percent change values for males and females
    change_dict_f = {}
    change_dict_m = {}

    for regnum, region in enumerate(all_regions):

        model_path = (f'/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                      f'Modeling/data/{band}/ROI_models/{region}/Models/')

        # Read agemin and agemax from file
        agemin, agemax = read_ages_from_file(struct_var, working_dir)

        # Create dummy covariate matrices with bspline values and save to file
        dummy_cov_file_path_female, dummy_cov_file_path_male = create_dummy_design_matrix(band, agemin, agemax,
                                                                        None, spline_order, spline_knots, working_dir)
        # Load dummy covariate matrices
        dummy_cov_f = np.loadtxt(dummy_cov_file_path_female)
        dummy_cov_m = np.loadtxt(dummy_cov_file_path_male)

        # remove last row which has erroneous bspline values
        dummy_cov_f = dummy_cov_f[:-1]
        dummy_cov_m = dummy_cov_m[:-1]

        # Open model parameter file
        with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as file:
            data = pickle.load(file)

        # Calculate predictions from model based on covariate data
        y_pred_f = np.dot(dummy_cov_f, data.blr.m)
        y_pred_m = np.dot(dummy_cov_m, data.blr.m)

        # Convert covariate and y values back to unscaled space
        dummy_cov_f[:,0] = dummy_cov_f[:,0] * minmax_scaler.data_range_[-1] + minmax_scaler.data_min_[-1]
        dummy_cov_m[:,0] = dummy_cov_m[:,0] * minmax_scaler.data_range_[-1] + minmax_scaler.data_min_[-1]

        y_pred_f = y_pred_f * minmax_scaler.data_range_[regnum] + minmax_scaler.data_min_[regnum]
        y_pred_m = y_pred_m * minmax_scaler.data_range_[regnum] + minmax_scaler.data_min_[regnum]

        # calculate percent change with age this brain region
        pchange_f = (y_pred_f[-1] - y_pred_f[0]) / y_pred_f[0] * 100.00
        pchange_m = (y_pred_m[-1] - y_pred_m[0]) / y_pred_m[0] * 100.00

        region = region.replace('-lh', '_left')
        region = region.replace('-rh', '_right')

        change_dict_f[region] = pchange_f
        change_dict_m[region] = pchange_m

        if plot_model:
            # plot model for this brain region
            plt.plot(dummy_cov_f[:,0]/age_conversion_factor, y_pred_f, 'crimson')
            plt.plot(dummy_cov_m[:,0]/age_conversion_factor, y_pred_m, 'b')
            # plt.ylim([0, 5])
            plt.title(f'Change in MEG power for {band} band in region {region}\nfemale % change = {pchange_f:.2f} '
                      f'male % change = {pchange_m:.2f}')
            plt.show()

    fig_f = ggseg.plot_dk(change_dict_f, cmap='cool', background='k', edgecolor='w', bordercolor='gray', figsize=(8,8),
                  ylabel=f'% Change MEG {band} power', title=f'Female Percent Change in MEG {band} '
                  'power from 9 to 17 years of age')

    fig_m = ggseg.plot_dk(change_dict_m, cmap='cool', background='k', edgecolor='w', bordercolor='gray', figsize=(8,8),
                  ylabel=f'% Change rsMEG {band} power', title=f'Male Percent Change in MEG {band} '
                  'power from 9 to 17 years of age')

mystop=1


