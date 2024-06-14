import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from helper_functions_MEG import create_design_matrix_one_gender, plot_data_with_spline_one_gender
from helper_functions_MEG import create_dummy_design_matrix_one_gender
from helper_functions_MEG import barplot_performance_values, plot_y_v_yhat_one_gender, makenewdir, movefiles
from helper_functions_MEG import write_ages_to_file_by_gender
from Load_Genz_Data import load_genz_data

def make_time1_normative_model(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, orig_data_dir, working_dir):

    # load visit 1 (pre-COVID) data
    visit = 1
    brain_good, all_data, roi_ids = load_genz_data(orig_struct_var, visit, orig_data_dir)

    if gender == 'male':
        # keep only data for males
        all_data = all_data.loc[all_data['sex'] == 1]
        struct_var = 'cortthick_male'
    else:
        # keep only data for females
        all_data = all_data.loc[all_data['sex'] == 2]
        struct_var = 'cortthick_female'

    #remove sex column
    all_data = all_data.drop(columns=['sex'])

    # make directories to store files
    makenewdir('{}/data/'.format(working_dir))
    makenewdir('{}/data/{}'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/plots'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/ROI_models'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/covariate_files'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/response_files'.format(working_dir, struct_var))

    if gender == 'male':
        # remove subject 525 who has an incidental finding
        brain_good = brain_good[~brain_good['participant_id'].isin([525])]
        all_data = all_data[~all_data['participant_id'].isin([525])]

    # show bar plots with number of subjects per age group in pre-COVID data
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects by Age with Pre-COVID Data\n'
                                 '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_allsubj',
                                  working_dir)

    ########
    # Before any modeling, determine which subjects will be included in the training (pre-COVID) and test (post-COVID) analysis.
    # Save subject numbers to file.
    # Do this only once and then comment the four following lines of code out.
    ########
    # save_test_set_to_file_no_long(struct_var, 9)
    # save_test_set_to_file_no_long(struct_var, 11)
    # save_test_set_to_file_no_long(struct_var, 13)
    # The resulting saved file is named visit1_subjects_excluded_from_normative_model_test_set_{struct_var}_9_11_13.txt

    # read in file of subjects in test set at ages 9, 11 and 13
    fname = '{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(orig_data_dir,
                                                                                              orig_struct_var)
    subjects_test = pd.read_csv(fname, header=None)

    # exclude subjects from the training set who are in test set
  #  brain_good = brain_good[~brain_good['participant_id'].isin(subjects_test[0])]
    all_data = all_data[~all_data['participant_id'].isin(subjects_test[0])]

    # write subject numbers for training set to file
    subjects_training = all_data['participant_id'].tolist()
    fname = '{}/visit1_subjects_used_to_create_normative_model_train_set_{}.txt'.format(working_dir, struct_var)
    file1 = open(fname, "w")
    for subj in subjects_training:
        file1.write(str(subj) + "\n")
    file1.close()

    # plot number of subjects of each gender by age who are included in training data set
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects by Age with Pre-COVID Data\nUsed to Create Model\n'
                                 '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_norm_model',
                                  working_dir)

    # drop rows with any missing values
    all_data = all_data.dropna()
    all_data.reset_index(inplace=True, drop=True)

    # separate the brain features (response variables) and predictors (age) in to separate dataframes
    all_data_features = all_data.loc[:, roi_ids]
    all_data_covariates = all_data[['age', 'agedays']]

    # If perform_train_test_split_precovid ==1 , split the training set into training and validation set.
    # If it is zero, create model based on entire training set
    if perform_train_test_split_precovid:
        # Split training set into training and validation sets. Training set will be used to create models. Performance will be
        # evaluated on the validation set. When performing train-test split, stratify by age and gender
        X_train, X_test, y_train, y_test = train_test_split(all_data_covariates, all_data_features,
                                                            stratify=all_data['age'], test_size=0.2,
                                                            random_state=42)
    else:
        # use entire training set to create models
        X_train = all_data_covariates.copy()
        X_test = all_data_covariates.copy()
        y_train = all_data_features.copy()
        y_test = all_data_features.copy()

    # identify age range in pre-COVID data to be used for modeling
    agemin = X_train['agedays'].min()
    agemax = X_train['agedays'].max()

    write_ages_to_file_by_gender(working_dir, agemin, agemax, struct_var, gender)

    # save the subject numbers for the training and validation sets to variables
    s_index_train = X_train.index.values
    s_index_test = X_test.index.values
    subjects_train = all_data.loc[s_index_train, 'participant_id'].values
    subjects_test = all_data.loc[s_index_test, 'participant_id'].values

    # drop the age column from the train and validation data sets because we want to use agedays as a predictor
    X_train.drop(columns=['age'], inplace=True)
    X_test.drop(columns=['age'], inplace=True)

    # change the indices in the train and validation data sets because nan values were dropped above
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    ##########
    # Set up output directories. Save each brain region to its own text file, organized in separate directories,
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
        roidirname = '{}/data/{}/ROI_models/{}'.format(working_dir, struct_var, i)
        makenewdir(roidirname)
        resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
        resp_tr_filepath = roidirname + '/resp_tr.txt'
        shutil.copyfile(resp_tr_filename, resp_tr_filepath)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_tr_filepath = roidirname + '/cov_tr.txt'
        shutil.copyfile("{}/cov_tr.txt".format(working_dir), cov_tr_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/data/{}/response_files/".format(working_dir, struct_var))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/data/{}/covariate_files/".format(working_dir, struct_var))

    #  this path is where ROI_models folders are located
    data_dir = '{}/data/{}/ROI_models/'.format(working_dir, struct_var)

    # Create Design Matrix and add in spline basis and intercept for validation and training data
    create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
    create_design_matrix_one_gender('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

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

        if show_plots:
            # plot y versus y hat for validation data
            plot_y_v_yhat_one_gender(gender, cov_file_te, resp_file_te, yhat_te, 'Validation Data', struct_var, roi,
                                                   Rho_te, EV_te)

        # create dummy design matrices for visualizing model
        dummy_cov_file_path = \
            (create_dummy_design_matrix_one_gender(struct_var, agemin, agemax, cov_file_tr, spline_order, spline_knots,
                                                   working_dir))

        # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
        plot_data_with_spline_one_gender(gender, 'Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path,
                              model_dir, roi, show_plots, working_dir)

        # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
        plot_data_with_spline_one_gender(gender, 'Validation Data', struct_var, cov_file_te, resp_file_te, dummy_cov_file_path,
                              model_dir, roi, show_plots, working_dir)

        # add a row to the blr_metrics dataframe containing ROI, MSLL, EXPV, SMSE, RMSE, and Rho metrics
        blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0],
                                             metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0],
                                             metrics_te['Rho'][0]]

        # load test (pre-COVID validation) data
        X_te = np.loadtxt(cov_file_te)
        y_te = np.loadtxt(resp_file_te)
        y_te = y_te[:, np.newaxis]  # make sure it is a 2-d array

        y_mean_te = np.mean(y_te)

        y_var_te = np.var(y_te)
        yhat_mean_te = np.mean(yhat_te)
        yhat_var_te = np.var(yhat_te)

        metrics_te = evaluate(y_te, yhat_te, s2_te, y_mean_te, y_var_te)

        blr_site_metrics.loc[len(blr_site_metrics)] = [roi, y_mean_te, y_var_te, yhat_mean_te, yhat_var_te,
                                                       metrics_te['MSLL'][0],
                                                       metrics_te['EXPV'][0], metrics_te['SMSE'][0],
                                                       metrics_te['RMSE'][0],
                                                       metrics_te['Rho'][0]]
        # store z score for ROI validation set
        Z_score_test_matrix[roi] = Z_te

    blr_site_metrics.to_csv('{}/data/{}/blr_metrics_{}.txt'.format(working_dir, struct_var, struct_var), index=False)

    # save validation z scores to file
    Z_score_test_matrix.to_csv('{}/data/{}/Z_scores_by_region_validation_set_{}.txt'.format(working_dir, struct_var,
                                            gender), index=False)

    ##########
    # Display plots of Rho and EV for validation set
    ##########

    blr_metrics.sort_values(by=['Rho'], inplace=True, ignore_index=True)
    barplot_performance_values(struct_var, 'Rho', blr_metrics, spline_order, spline_knots, 'Validation Set',
                               working_dir, gender)
    blr_metrics.sort_values(by=['EV'], inplace=True, ignore_index=True)
    barplot_performance_values(struct_var, 'EV', blr_metrics, spline_order, spline_knots, 'Validation Set', working_dir,
                               gender)
    # plt.show()

    return Z_score_test_matrix