#####
# This program imports the model and Z-scores from the bayesian linear regression normative modeling of the
# training data set (which is the adolescent visit 1 data). It then uses the model to calculate Z-scores for
# the post-covid adolescent (visit 2) data.
# Author: Neva M. Corrigan
######
import os
import pandas as pd
from matplotlib import pyplot as plt
from prepare_rsMEG_data import prepare_rsMEG_data
from helper_functions_MEG import plot_num_subjs
from helper_functions_MEG import recreate_folder, movefiles, create_design_matrix_one_gender, recreate_folder
from helper_functions_MEG import plot_data_with_spline_one_gender, create_dummy_design_matrix_one_gender, read_ages_from_file
import shutil
from normative_edited import predict
from joblib import load

def apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                working_dir, rsd_v2, dirdata, dirpredict, sex, band):

    rsd_v2 =rsd_v2[rsd_v2['subject'] < 400]

    # reset indices
    rsd_v2.reset_index(inplace=True, drop=True)

    if show_nsubject_plots:
        plot_num_subjs(sex, rsd_v2, f'{sex.capitalize()} Subjects by Age with Post-COVID MEGrs Data\nEvaluated by Model\n'
                       +' (Total N=' + str(rsd_v2.shape[0]) + ')', struct_var, 'post-covid_allsubj', os.path.join(working_dir, dirdata))

    # read agemin and agemax from file
    agemin, agemax = read_ages_from_file(struct_var, working_dir, sex)

    #make a matrix of response variables, one for each brain region
    rs_covariates = rsd_v2[['agegrp', 'agedays']]
    rscols = [col for col in rsd_v2.columns if col not in ['subject', 'agegrp', 'agedays']]

    # make directories to store band specific files in
    recreate_folder(os.path.join(working_dir, dirpredict, f'{sex}_{band}'))
    recreate_folder(os.path.join(working_dir, dirpredict, f'{sex}_{band}', 'plots'))
    recreate_folder(os.path.join(working_dir, dirpredict, f'{sex}_{band}', 'ROI_models'))
    recreate_folder(os.path.join(working_dir, dirpredict, f'{sex}_{band}', 'covariate_files'))
    recreate_folder(os.path.join(working_dir, dirpredict, f'{sex}_{band}', 'response_files'))

    rscols_band = [item for item in rscols if band in item]
    rs_features = rsd_v2.loc[:, rscols_band]

    X_test = rs_covariates.copy()
    y_test = rs_features.copy()

    # Make a new dataframe with test subject numbers
    s_index_test = X_test.index.values
    subjects_test = rsd_v2.loc[s_index_test, 'subject'].values

    # drop the agegrp column from the test data set because we want to use agedays as a predictor
    X_test.drop(columns=['agegrp'], inplace=True)

    # Get region names from test set feature names
    roi_ids = y_test.columns.str.replace(fr'^{band}-', '', regex=True).tolist()

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########
    for c in y_test.columns:
        roi = c[len(band) + 1:]
        y_test[c].to_csv(f'{working_dir}/resp_te_' + roi + '.txt', header=False, index=False)
        X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in roi_ids:
        roidirname = '{}/{}/{}_{}/ROI_models/{}'.format(working_dir, dirpredict, sex, band, i)
        recreate_folder(roidirname)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/{}/{}_{}/response_files/"
              .format(working_dir, dirpredict, sex, band))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/{}/{}_{}/covariate_files/"
              .format(working_dir, dirpredict, sex, band))

    # specify paths
    training_dir = '{}/{}/{}_{}/ROI_models/'.format(working_dir, dirdata, sex, band)
    out_dir = '{}/{}/{}_{}/ROI_models/'.format(working_dir, dirpredict, sex, band)
    #  this path is where ROI_models folders are located
    predict_files_dir = '{}/{}/{}_{}/ROI_models/'.format(working_dir, dirpredict, sex, band)

    # Create Design Matrix and add in spline basis and intercept
    create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    subjects_test = subjects_test.reshape(-1, 1)
    Z_score_test_matrix = pd.DataFrame(subjects_test, columns=['subject_id_test'])

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    for roi in roi_ids:
        print(f'Running ROI:', roi, 'predict for Band {band}')
        roi_dir = os.path.join(predict_files_dir, roi)
        model_dir = os.path.join(training_dir, roi, 'Models')
        os.chdir(roi_dir)

        # configure the covariates to use.
        cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        #create dummy design matrices
        dummy_cov_file_path= create_dummy_design_matrix_one_gender(agemin, agemax,
                                                    spline_order, spline_knots, working_dir)

        plot_data_with_spline_one_gender(sex, 'Postcovid (Test) Data ', band, cov_file_te, resp_file_te,
                    dummy_cov_file_path, model_dir, roi, show_plots, working_dir, dirdata)
        #
        # plt.show()

        Z_score_test_matrix[roi] = Z

    Z_score_test_matrix.to_csv('{}/{}/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                        .format(working_dir, dirpredict, sex, band), index=False)

    return Z_score_test_matrix

