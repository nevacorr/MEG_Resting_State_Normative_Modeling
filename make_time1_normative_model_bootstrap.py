import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from helper_functions_MEG import plot_num_subjs, plot_feature_distributions, makenewdir_deleteold
from helper_functions_MEG import create_design_matrix_one_gender, plot_data_with_spline_one_gender
from helper_functions_MEG import create_dummy_design_matrix_one_gender, remove_outliers_IQR
from helper_functions_MEG import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from helper_functions_MEG import write_ages_to_file_by_gender
from prepare_rsMEG_data import prepare_rsMEG_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump

def make_time1_normative_model_bootstrap(rsd_v1, gender, struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, working_dir, MEG_filename, ct_data_dir,
                               subjects_to_exclude, bands, n_bootstraps):

    # make directories to store files in
    makenewdir('{}/data_bootstrap/'.format(working_dir))

    # identify age range in pre-COVID data to be used for modeling
    agemin =rsd_v1['agedays'].min()
    agemax =rsd_v1['agedays'].max()

    rsd_v1_orig = rsd_v1

    model_slope = {}
    conf_interval = pd.DataFrame()
    sig =  pd.DataFrame()

    # loop through all power bands separately
    for band in bands:

        model_slope[band] = pd.DataFrame()

        # Perform bootstrapping
        for b in range(n_bootstraps):

            print(f'bootstrap {b} of {n_bootstraps}')

            rsd_v1 = rsd_v1_orig.groupby('agegrp', group_keys=False).apply(lambda x: x.sample(frac=1, replace=True, axis=0))

            # separate the brain features (response variables) and predictors (age) in to separate dataframes
            rs_covariates = rsd_v1[['agegrp', 'agedays']]
            rscols = [col for col in rsd_v1.columns if col not in ['subject', 'agegrp', 'agedays']]

            # make directories to store band specific files in
            makenewdir_deleteold('{}/data_bootstrap/{}_{}'.format(working_dir, gender, band))
            makenewdir_deleteold('{}/data_bootstrap/{}_{}/plots'.format(working_dir, gender, band))
            makenewdir_deleteold('{}/data_bootstrap/{}_{}/ROI_models'.format(working_dir, gender, band))
            makenewdir_deleteold('{}/data_bootstrap/{}_{}/covariate_files'.format(working_dir, gender, band))
            makenewdir_deleteold('{}/data_bootstrap/{}_{}/response_files'.format(working_dir, gender, band))

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
                y_train[c].to_csv(f'{working_dir}/resp_tr_' + gender + '_' + c + '.txt', header=False, index=False)
                X_train.to_csv(f'{working_dir}/cov_tr.txt', sep='\t', header=False, index=False)
                y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)
            for c in y_test.columns:
                y_test[c].to_csv(f'{working_dir}/resp_te_' + gender + '_' + c + '.txt', header=False, index=False)
                X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
                y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

            for i in roi_ids:
                roidirname = '{}/data_bootstrap/{}_{}/ROI_models/{}'.format(working_dir, gender, band, i)
                makenewdir(roidirname)
                resp_tr_filename = "{}/resp_tr_{}_{}_{}.txt".format(working_dir, gender, band, i)
                resp_tr_filepath = roidirname + '/resp_tr.txt'
                shutil.copyfile(resp_tr_filename, resp_tr_filepath)
                resp_te_filename = "{}/resp_te_{}_{}_{}.txt".format(working_dir, gender, band, i)
                resp_te_filepath = roidirname + '/resp_te.txt'
                shutil.copyfile(resp_te_filename, resp_te_filepath)
                cov_tr_filepath = roidirname + '/cov_tr.txt'
                shutil.copyfile("{}/cov_tr.txt".format(working_dir), cov_tr_filepath)
                cov_te_filepath = roidirname + '/cov_te.txt'
                shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

            movefiles("{}/resp_*.txt".format(working_dir), "{}/data_bootstrap/{}_{}/response_files/".format(working_dir, gender, band))
            movefiles("{}/cov_t*.txt".format(working_dir), "{}/data_bootstrap/{}_{}/covariate_files/".format(working_dir, gender, band))

            #  this path is where ROI_models folders are located
            data_dir = '{}/data_bootstrap/{}_{}/ROI_models/'.format(working_dir, gender, band)

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

                # Note outscaler = 'standardize' has been added as an argument. This resolves an issue where modeling does
                # not appear to work correctly with numbers at the scale of these MEG numbers (in the hundreds instead of
                # single digits for cortical thickness.
                yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                                                                testcov=cov_file_te, alg='blr', optimizer='powell',
                                                                savemodel=False, saveoutput=False, standardize=True)

                # Create dummy covariate matrices with bspline values and save to file
                dummy_cov_file_path = create_dummy_design_matrix_one_gender(band, agemin, agemax,
                                                                            None, spline_order, spline_knots,
                                                                            working_dir)
                # Load dummy covariate matrix
                dummy_cov = np.loadtxt(dummy_cov_file_path)

                # remove last row which has erroneous bspline values
                dummy_cov = dummy_cov[:-1]

                # Calculate the slope of the line
                index_for_x1 = 0
                index_for_x2 = dummy_cov.shape[0] - 1

                model_slope[band].loc[b,roi] = ((nm.blr.m[0] * (dummy_cov[index_for_x2, 0] - dummy_cov[index_for_x1, 0]) +
                               nm.blr.m[1] * (dummy_cov[index_for_x2, 1] - dummy_cov[index_for_x1, 1]) +
                               nm.blr.m[2] * (dummy_cov[index_for_x2, 2] - dummy_cov[index_for_x1, 2]) +
                               nm.blr.m[3] * (dummy_cov[index_for_x2, 3] - dummy_cov[index_for_x1, 3])) /
                              (dummy_cov[index_for_x2, 0] - dummy_cov[index_for_x1, 0]))

    mystop=1

    # Calculate confidence intervals
    for reg in model_slope[band].columns:
        slopes_reg = model_slope[band].loc[:, reg]
        conf_interval.loc[band, reg] = np.percentile(slopes_reg.to_numpy(), [2.5, 97.5])
        if conf_interval.loc[band, reg][0] < 0 < conf_interval.loc[band,reg][1]:
            sig.loc[band, reg] = 0
        else:
            sig.loc[band, reg] = 1

    mystop=1

    sig.to_csv(f'{gender}_significance of slopes by band and region')

