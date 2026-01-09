#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent meg data collected at two time points (before and after the COVID lockdowns).
######
import os
import pickle
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from make_and_apply_normative_model import make_and_apply_normative_model
from helper_functions_MEG import recreate_folder, copy_old_files_to_backup_folder

struct_var = 'meg'
n_splits = 2         # number of train/test splits
show_plots = 0        #set to 1 to show training and test data spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for models
spline_knots = 2        # number of knots in spline to use in models
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
data_type = 'relative' #options: relative, absolute
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
working_dir = os.getcwd()

run_make_norm_model = 1
plot_z_distributions = 1
lobes_only = 0
subjects_to_exclude = [525] #532 was an outlier on original MEG data set but is no longer with updated
bands = ['theta', 'alpha', 'beta', 'gamma']

Z2_all_splits = {}

if data_type == 'relative':
    MEG_resting_state_filename = '/home/toddr/neva/PycharmProjects/data_dir/genz_rs_power_rel_vfix_alln_December2024.csv'
elif data_type == 'absolute':
    MEG_resting_state_filename = '/home/toddr/neva/PycharmProjects/data_dir/genz_rs_power_vfix_alln.csv'

# Create directory for storing subject number bar plots
recreate_folder(os.path.join(working_dir, 'data'))
recreate_folder(os.path.join(working_dir, 'predict_files'))
copy_old_files_to_backup_folder(os.path.join(working_dir, 'output_data'), os.path.join(working_dir, 'output_data_bak'))
recreate_folder(os.path.join(working_dir, 'output_data'))

for gender in ['male', 'female']:

    if run_make_norm_model:

        Z2_all_splits[gender] = make_and_apply_normative_model(gender, struct_var, show_plots, show_nsubject_plots, spline_order,
                                             spline_knots, data_dir, working_dir, ct_data_dir, MEG_resting_state_filename,
                                             subjects_to_exclude, bands, n_splits, lobes_only, data_type)
    # else:
    #
    #     with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_{gender}_{n_splits}_splits.pkl'), 'rb') as f:
    #         Z2_all_splits[gender] = pickle.load(f)

if plot_z_distributions:

    plot_and_compute_zcores_by_gender(Z2_all_splits, working_dir, bands)
