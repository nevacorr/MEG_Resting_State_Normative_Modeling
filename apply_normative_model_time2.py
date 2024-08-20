#####
# This program imports the model and Z-scores from the bayesian linear regression normative modeling of the
# training data set (which is the adolescent visit 1 data). It then uses the model to calculate Z-scores for
# the post-covid adolescent (visit 2) data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######
import os
import pandas as pd
from matplotlib import pyplot as plt
from prepare_rsMEG_data import prepare_rsMEG_data
from helper_functions_MEG import plot_num_subjs
from helper_functions_MEG import makenewdir, movefiles, create_dummy_design_matrix
from helper_functions_MEG import plot_data_with_spline, create_design_matrix, read_ages_from_file
import shutil
from normative_edited import predict

def apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                working_dir, MEG_filename, ct_data_dir, subjects_to_exclude):

    bands = ['theta', 'alpha', 'beta', 'gamma']

    # load all rs MEG data
    rsd_v1, rsd_v2 = prepare_rsMEG_data(working_dir, MEG_filename, subjects_to_exclude, ct_data_dir)

    # Replace gender codes 1=male 2=female with binary values (make male=1 and female=0)
    rsd_v1.loc[rsd_v1['gender'] == 2, 'gender'] = 0
    rsd_v2.loc[rsd_v2['gender'] ==2, 'gender'] = 0

    #extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
    subjects_visit1 = rsd_v1['subject']
    rows_in_v2_but_not_v1 = rsd_v2[~rsd_v2['subject'].isin(rsd_v1['subject'])].dropna()
    subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['subject'].copy()
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)
    #only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model 18 and 20 year olds
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

    #only include subjects that were not in the training set
    fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(ct_data_dir, 'cortthick')
    subjects_to_include = pd.read_csv(fname, header=None)
    subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
    rsd_v2 = rsd_v2[rsd_v2['subject'].isin(subjects_to_include[0])]

    #make file diretories for output
    makenewdir('{}/predict_files/'.format(working_dir))
    makenewdir('{}/predict_files/{}'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/plots'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/ROI_models'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/covariate_files'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/response_files'.format(working_dir, struct_var))

    # reset indices
    rsd_v2.reset_index(inplace=True)

    #show number of subjects by gender and age
    if show_nsubject_plots:
        plot_num_subjs(rsd_v2, f'Subjects by Age with Post-COVID MEGrs Data\nEvaluated by Model\n'
                       +' (Total N=' + str(rsd_v2.shape[0]) + ')', struct_var, 'post-covid_allsubj', working_dir)

    # Remove the prefix 't2_' from column names
    rsd_v2.columns = rsd_v2.columns.str.replace(r'^t2_', '', regex=True)

    # read agemin and agemax from file
    agemin, agemax = read_ages_from_file(working_dir, struct_var)

    #specify which columns of dataframe to use as covariates
    rs_covariates = rsd_v2[['agegrp', 'agedays', 'gender']]

    #make a matrix of response variables, one for each brain region
    rscols = [col for col in rsd_v2.columns if col not in ['subject', 'agegrp', 'agedays', 'gender']]

    #specify paths
    training_dir = '{}/data/{}/ROI_models/'.format(working_dir, struct_var)
    out_dir = '{}/predict_files/{}/ROI_models/'.format(working_dir, struct_var)
    #  this path is where ROI_models folders are located
    predict_files_dir = '{}/predict_files/{}/ROI_models/'.format(working_dir, struct_var)

    # loop through each power band separately
    for band in bands:

        rscols_band = [item for item in rscols if band in item]
        rs_features = rsd_v1.loc[:, rscols_band]

        X_test = rs_covariates.copy()
        y_test = rs_features.copy()

        # drop the agegrp column from the test data set because we want to use agedays as a predictor
        X_test.drop(columns=['agegrp'], inplace=True)

        ##########
        # Create output directories for each region and place covariate and response files for that region in  each directory
        ##########
        for c in y_test.columns:
            y_test[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
            X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
            y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in roi_ids:
        roidirname = '{}/predict_files/{}/ROI_models/{}'.format(working_dir, struct_var, i)
        makenewdir(roidirname)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/predict_files/{}/response_files/"
              .format(working_dir, struct_var))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/predict_files/{}/covariate_files/"
              .format(working_dir, struct_var))

    # # Create Design Matrix and add in spline basis and intercept
    # create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['participant_id'] = all_data['participant_id'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    #create design matrices for all regions and save files in respective directories
    create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)

    for roi in roi_ids:
        print('Running ROI:', roi)
        roi_dir = os.path.join(predict_files_dir, roi)
        model_dir = os.path.join(training_dir, roi, 'Models')
        # os.chdir(roi_dir)

        # configure the covariates to use.
        cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        Z_time2[roi] = Z

        #create dummy design matrices
        dummy_cov_file_path= \
            create_dummy_design_matrix_one_gender(struct_var, agemin, agemax, cov_file_te, spline_order, spline_knots,
                                                  working_dir)

        plot_data_with_spline_one_gender(gender, 'Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te,
                                         dummy_cov_file_path, model_dir, roi, show_plots, working_dir)

        mystop=1

    Z_time2.to_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                .format(working_dir, struct_var), index=False)

    plt.show()

    return Z_time2

