import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from helper_functions_MEG import plot_num_subjs
from helper_functions_MEG import create_design_matrix, plot_data_with_spline
from helper_functions_MEG import create_dummy_design_matrix
from helper_functions_MEG import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from helper_functions_MEG import write_ages_to_file
from prepare_rsMEG_data import prepare_rsMEG_data
from sklearn.preprocessing import StandardScaler

def make_time1_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, working_dir, MEG_filename, ct_data_dir,
                               subjects_to_exclude):

    bands = ['theta', 'alpha', 'beta', 'gamma']

    # load all rs MEG data
    rsd_v1, rsd_v2 = prepare_rsMEG_data(MEG_filename, subjects_to_exclude, ct_data_dir)

    # Replace gender codes 1=male 2=female with binary values (make male=1 and female=0)
    rsd_v1.loc[rsd_v1['gender'] == 2, 'gender'] = 0
    rsd_v2.loc[rsd_v2['gender'] ==2, 'gender'] = 0

    # Divide all MEG numbers by 150
    columns_to_exclude = ['subject', 'agegrp', 'gender', 'agedays']
    columns_to_modify1 = rsd_v1.columns.difference(columns_to_exclude)
    columns_to_modify2 = rsd_v2.columns.difference(columns_to_exclude)

    rsd_v1[columns_to_modify1] = rsd_v1[columns_to_modify1] / 150
    rsd_v2[columns_to_modify2] = rsd_v2[columns_to_modify2] / 150

    # make directories to store files in
    makenewdir('{}/data/'.format(working_dir))

     # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:
        plot_num_subjs(rsd_v1, f'Subjects by Age with Pre-COVID MEGrs Data\n'
                                 '(Total N=' + str(rsd_v1.shape[0]) + ')', struct_var, 'pre-covid_allsubj',
                                  working_dir)

    ########
    # Use same train test subgroups as was used for cortical thickness analysis
    ########

    # read in file of subjects in test set at ages 9, 11 and 13
    fname = '{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(ct_data_dir, 'cortthick')
    subjects_test = pd.read_csv(fname, header=None)

    # exclude subjects from the training set who are in test set
    rsd_v1 = rsd_v1[~rsd_v1['subject'].isin(subjects_test[0])]
    rsd_v1.reset_index(inplace=True, drop=True)

    # plot number of subjects of each gender by age who are included in training data set
    if show_nsubject_plots:
        plot_num_subjs(rsd_v1, f' Subjects by Age with Pre-COVID MEGrs Data\nUsed to Create Model\n'
                                 '(Total N=' + str(rsd_v1.shape[0]) + ')', struct_var, 'pre-covid_norm_model',
                                  working_dir)

    # Remove the prefix 't1_' from column names
    rsd_v1.columns = rsd_v1.columns.str.replace(r'^t1_', '', regex=True)

    # identify age range in pre-COVID data to be used for modeling
    agemin =rsd_v1['agedays'].min()
    agemax =rsd_v1['agedays'].max()

    # Write ages to file
    write_ages_to_file(agemin, agemax, struct_var, working_dir)

    # separate the brain features (response variables) and predictors (age) in to separate dataframes
    rs_covariates = rsd_v1[['agegrp', 'agedays', 'gender']]
    rscols = [col for col in rsd_v1.columns if col not in ['subject', 'agegrp', 'agedays', 'gender']]

    # loop through all power bands separately
    for band in bands:

        # make directories to store band specific files in
        makenewdir('{}/data/{}'.format(working_dir, band))
        makenewdir('{}/data/{}/plots'.format(working_dir, band))
        makenewdir('{}/data/{}/ROI_models'.format(working_dir, band))
        makenewdir('{}/data/{}/covariate_files'.format(working_dir, band))
        makenewdir('{}/data/{}/response_files'.format(working_dir, band))

        rscols_band = [item for item in rscols if band in item]
        rs_features = rsd_v1.loc[:, rscols_band]

        # # If perform_train_test_split_precovid ==1 , split the training set into training and validation set.
        # # If it is zero, create model based on entire training set
        # if perform_train_test_split_precovid:
        #     # Split training set into training and validation sets. Training set will be used to create models. Performance will be
        #     # evaluated on the validation set. When performing train-test split, stratify by age and gender
        #     X_train, X_test, y_train, y_test = train_test_split(all_data_covariates, all_data_features,
        #                                                         stratify=all_data['age'], test_size=0.2,
        #                                                         random_state=42)
        # else:

        # use entire training set to create models
        X_train = rs_covariates.copy()
        X_test = rs_covariates.copy()
        y_train = rs_features.copy()
        y_test = rs_features.copy()

        # for the first loop iteration, save the subject numbers for the training and validation sets to variables
        if band == 'theta':
            s_index_train = X_train.index.values
            s_index_test = X_test.index.values
            subjects_train = rsd_v1.loc[s_index_train, 'subject'].values
            subjects_test = rsd_v1.loc[s_index_test, 'subject'].values

        # drop the agegrp column from the train and validation data sets because we want to use agedays as a predictor
        X_train.drop(columns=['agegrp'], inplace=True)
        X_test.drop(columns=['agegrp'], inplace=True)

        # change the indices in the train and validation data sets because nan values were dropped above
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # Get region names from training set feature names
        roi_ids = y_train.columns.str.replace(fr'^{band}_', '', regex=True).tolist()

        ##########
        # Set up output directories. Save data for each brain region to its own text file, organized in separate directories,
        # because for each response variable Y (brain region) we fit a separate normative mode
        ##########
        for c in y_train.columns:
            y_train[c].to_csv(f'{working_dir}/resp_tr_' + c + '.txt', header=False, index=False)
            X_train.to_csv(f'{working_dir}/cov_tr.txt', sep='\t', header=False, index=False)
            y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)
        for c in y_test.columns:
            y_test[c].to_csv(f'{working_dir}/resp_te_' + c + '.txt', header=False, index=False)
            X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
            y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/data/{}/ROI_models/{}'.format(working_dir, band, i)
            makenewdir(roidirname)
            resp_tr_filename = "{}/resp_tr_{}_{}.txt".format(working_dir, band, i)
            resp_tr_filepath = roidirname + '/resp_tr.txt'
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)
            resp_te_filename = "{}/resp_te_{}_{}.txt".format(working_dir, band, i)
            resp_te_filepath = roidirname + '/resp_te.txt'
            shutil.copyfile(resp_te_filename, resp_te_filepath)
            cov_tr_filepath = roidirname + '/cov_tr.txt'
            shutil.copyfile("{}/cov_tr.txt".format(working_dir), cov_tr_filepath)
            cov_te_filepath = roidirname + '/cov_te.txt'
            shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/data/{}/response_files/".format(working_dir, band))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/data/{}/covariate_files/".format(working_dir, band))

        #  this path is where ROI_models folders are located
        data_dir = '{}/data/{}/ROI_models/'.format(working_dir, band)

        # Create Design Matrix and add in spline basis and intercept for validation and training data
        create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
        create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

        # Create pandas dataframes with header names to save evaluation metrics
        blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
        blr_site_metrics = pd.DataFrame(
        columns=['ROI', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

        # create dataframe with subject numbers to put the Z scores in. Here 'test' refers to the validation set
        subjects_test = subjects_test.reshape(-1, 1)
        subjects_train = subjects_train.reshape(-1, 1)
        Z_score_test_matrix = pd.DataFrame(subjects_test, columns=['subject_id_test'])
        Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

        # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
        # specific arguments that are worth commenting on:
        # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
        # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
        # ●savemodel=True: do not write out the final estimated model to disk
        # ●saveoutput=False: return the outputs directly rather than writing them to disk
        # ●standardize=False: do not standardize the covariates or response variable

        # Loop through ROIs

        for roi in roi_ids:
            print('Running ROI:', roi)
            roi_dir = os.path.join(data_dir, roi)
            model_dir = os.path.join(data_dir, roi, 'Models')
            os.chdir(roi_dir)

            # configure the covariates to use. Change *_bspline_* to *_int_*
            cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')
            cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

            # load train & test response files
            resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')
            resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

            # calculate a model based on the training data and apply to the validation dataset. If the model is being created
            # from the entire training set, the validation set is simply a copy of the full training set and the purpose of
            # running this function is to creat and save the model, not to evaluate performance. The following are calcualted:
            # the predicted validation set response (yhat_te), the variance of the predicted response (s2_te), the model
            # parameters (nm),the Zscores for the validation data, and other various metrics (metrics_te)
            yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                                                            testcov=cov_file_te, alg='blr', optimizer='powell',
                                                            savemodel=True, saveoutput=False, standardize=False)

            Rho_te = metrics_te['Rho']
            EV_te = metrics_te['EXPV']

            # if show_plots:
            #     # plot y versus y hat for validation data
            #     plot_y_v_yhat(cov_file_te, resp_file_te, yhat_te, 'Validation Data', struct_var, roi,
            #                                            Rho_te, EV_te)

            # create dummy design matrices for visualizing model
            dummy_cov_file_path_female, dummy_cov_file_path_male = create_dummy_design_matrix(band, agemin, agemax,
                                                                cov_file_tr, spline_order, spline_knots, working_dir)

            # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
            plot_data_with_spline('Training Data', band, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                                  dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)

            # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
            plot_data_with_spline('Validation Data', band, cov_file_te, resp_file_te, dummy_cov_file_path_female,
                                  dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)

            # store z score for ROI validation set
            Z_score_test_matrix[roi] = Z_te

            # save validation z scores to file
            Z_score_test_matrix.to_csv('{}/data/{}/Z_scores_by_region_validation_set.txt'.format(working_dir, band),
                                       index=False)

    return Z_score_test_matrix