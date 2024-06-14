#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent meg data collected at two time points (before and after the COVID lockdowns).
######

import pandas as pd
from make_time1_normative_model import make_time1_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_z_scores import plot_and_compute_zcores_by_gender

import time

struct_var = 'meg'
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0  # flag indicating whether to split training set (pre-covid data) into train and
                                       # validations (test) sets. If this is set to 0, the entire training set is used
                                       # for the model and there is no validation set. Regardless of the value of this
                                       # flag, no post-covid data is used in creating or evaluating the normative model.
run_make_norm_model = 0
run_apply_norm_model = 0

working_dir = '/home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling'

Z_time1 = {}
Z_time2 = {}

if run_make_norm_model:

        Z_time1 = make_time1_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, working_dir)

        Z_time1.drop(columns=['subject_id_test'], inplace=True)

if run_apply_norm_model:

        Z_time2 = apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                working_dir)


if run_apply_norm_model:
    Z_time2 = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                               .format(working_dir, struct_var))


    Z_time2.to_csv(f'{working_dir}/predict_files/Z_time2.csv', index=False)

    plot_and_compute_zcores_by_gender(struct_var, Z_time2)

mystop=1