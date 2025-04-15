#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent meg data collected at two time points (before and after the COVID lockdowns).
######
import os

import pandas as pd
from make_time1_normative_model import make_time1_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from make_and_apply_normative_model import make_and_apply_normative_model
from make_time1_normative_model_bootstrap import make_time1_normative_model_bootstrap

struct_var = 'meg'
n_splits = 2           # number of train/test splits
show_plots = 0         #set to 1 to show training and test data spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for models
spline_knots = 2        # number of knots in spline to use in models
ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
MEG_resting_state_filename = '/home/toddr/neva/PycharmProjects/data_dir/genz_rs_power_rel_vfix_alln_December2024.csv'
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
working_dir = os.getcwd()

run_make_norm_model = 1
perform_bootstrap = 0
n_bootstraps = 1
lobes_only = 0
subjects_to_exclude = [525] #532 was an outlier on original MEG data set but is no longer with updated
# bands = ['theta', 'alpha', 'beta', 'gamma']
bands = ['theta', 'alpha']

Z2_all_splits = {}
Z_time2 = {}

for gender in ['male', 'female']:

    if run_make_norm_model:

        Z2_all_splits[gender] = make_and_apply_normative_model(gender, struct_var, show_plots, show_nsubject_plots, spline_order,
                                             spline_knots, data_dir, working_dir, ct_data_dir, MEG_resting_state_filename,
                                             subjects_to_exclude, bands, n_splits, lobes_only)

        # Z_time1[gender], rsd_v1 = make_time1_normative_model(gender, struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
        #                            perform_train_test_split_precovid, working_dir, MEG_resting_state_filename, ct_data_dir,
        #                            subjects_to_exclude, bands, lobes_only)
        #
        # Z_time1[gender].drop(columns=['subject_id_test'], inplace=True)
        #
        # if perform_bootstrap == 0:
        #
        #     make_time1_normative_model_bootstrap(rsd_v1, gender,spline_order, spline_knots,
        #                                                  working_dir, bands, n_bootstraps)


# if run_apply_norm_model:
#
#     for band in bands:
#         Z_time2_male= pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
#                                    .format(working_dir, 'male', band))
#         Z_time2_male.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
#
#         Z_time2_female= pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
#                                    .format(working_dir, 'female', band))
#         Z_time2_female.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
#
#         Z_time2[f'male_{band}'] = Z_time2_male
#         Z_time2[f'female_{band}'] = Z_time2_female
#
#     plot_and_compute_zcores_by_gender(Z_time2, working_dir, bands)
#
# mystop=1