#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent meg data collected at two time points (before and after the COVID lockdowns).
######

import pandas as pd
from make_time1_normative_model import make_time1_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender

struct_var = 'meg'
show_plots = 0          #set to 1 to show training and test data spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0  # flag indicating whether to split training set (pre-covid data) into train and
                                       # validations (test) sets. If this is set to 0, the entire training set is used
                                       # for the model and there is no validation set. Regardless of the value of this
                                       # flag, no post-covid data is used in creating or evaluating the normative model.
run_make_norm_model = 1
run_apply_norm_model = 1
subjects_to_exclude = [525] #532 was an outlier on original data set but is no longer
# bands = ['theta', 'alpha', 'beta', 'gamma']
bands = ['theta']

ct_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = '/home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling'
MEG_resting_state_filename = '/home/toddr/neva/PycharmProjects/data_dir/genz_rs_power_vfix_alln.csv'
Z_time1 = {}
Z_time2 = {}

if run_make_norm_model:

    Z_time1 = make_time1_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, working_dir, MEG_resting_state_filename, ct_data_dir,
                               subjects_to_exclude, bands)

    Z_time1.drop(columns=['subject_id_test'], inplace=True)

if run_apply_norm_model:

    Z_time2, roi_ids = apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                working_dir, MEG_resting_state_filename, ct_data_dir, subjects_to_exclude, bands)

    for band in bands:
        Z_time2= pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                               .format(working_dir, band))
        Z_time2.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)
        plot_and_compute_zcores_by_gender(Z_time2, band, roi_ids, working_dir)

mystop=1