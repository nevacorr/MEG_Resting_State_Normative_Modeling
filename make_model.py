import pandas as pd
import os
import shutil
import numpy as np
from numpy.core.defchararray import capitalize
from pcntoolkit.normative import estimate, evaluate
import pickle
from helper_functions_MEG import create_design_matrix_one_gender, plot_data_with_spline_one_gender
from helper_functions_MEG import create_dummy_design_matrix_one_gender
from helper_functions_MEG import barplot_performance_values, plot_y_v_yhat, movefiles, plot_num_subjs
from helper_functions_MEG import write_ages_to_file_by_gender, recreate_folder, calc_model_slope
from apply_normative_model_time2 import apply_normative_model_time2
import time
import json
from sklearn.model_selection import train_test_split

def make_model(rsd_v1_orig, rsd_v2_orig, struct_var, n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, sex, bands, evaluate_model_fit):

    dirdata = 'data'
    dirpredict = 'predict_files'

    # Initialize dictionaries for storing Z scores and model slopes
    Z_time2 = {}
    Z2_all_splits_dict = {}
    all_blr_site_metrics_dict = {}
    model_slope = pd.DataFrame()
    ymin = pd.DataFrame()

    for b in bands:
        Z_time2[b] = pd.DataFrame()
        Z2_all_splits_dict[b] = pd.DataFrame()
        all_blr_site_metrics_dict[b] = pd.DataFrame()

    all_split_start = time.time()
    for split in range(n_splits):
        print(f'Running split {split+1}')
        start_time = time.time()

        subjects_train = train_set_array[split, :]
        subjects_test = test_set_array[split, :]

        rsd_v1 = rsd_v1_orig[rsd_v1_orig['subject'].isin(subjects_train)].copy()
        rsd_v2 = rsd_v2_orig[rsd_v2_orig['subject'].isin(subjects_test)].copy()
        rsd_v1.reset_index(drop=True, inplace=True)
        rsd_v2.reset_index(drop=True, inplace=True)

        if show_nsubject_plots:
            plot_num_subjs(sex, rsd_v1, f'{sex.capitalize()} Subjects by Age with Pre-COVID Data used to Train Model Split {split}\n '
                                    f'(Total N=' + str(rsd_v1.shape[0]) + ')', struct_var,'pre-covid_train', os.path.join(working_dir, dirdata))

        # separate the brain features (response variables) and predictors (age) in to separate dataframes
        rs_covariates = rsd_v1[['agegrp', 'agedays']]
        rscols = [col for col in rsd_v1.columns if col not in ['subject', 'agegrp', 'agedays']]

        # loop through all power bands separately
        for band in bands:

            # make directories to store band specific files in
            recreate_folder(os.path.join(working_dir, dirdata, f'{sex}_{band}'))
            recreate_folder(os.path.join(working_dir, dirdata, f'{sex}_{band}','plots'))
            recreate_folder(os.path.join(working_dir, dirdata, f'{sex}_{band}', 'ROI_models'))
            recreate_folder(os.path.join(working_dir, dirdata, f'{sex}_{band}', 'covariate_files'))
            recreate_folder(os.path.join(working_dir, dirdata, f'{sex}_{band}', 'response_files'))

            rscols_band = [item for item in rscols if band in item]
            rs_features = rsd_v1.loc[:, rscols_band]

            if evaluate_model_fit:
                X_train, X_val, y_train, y_val = train_test_split(rs_covariates, rs_features,
                                                    stratify=rs_covariates[['agegrp']], test_size=0.2, random_state=1)
                subjects_val = rsd_v1.loc[X_val.index, 'subject'].values
            else:
                # use entire training set to create models
                X_train = rs_covariates.copy()
                X_val = rs_covariates.copy()
                y_train = rs_features.copy()
                y_val = rs_features.copy()
                subjects_val=rsd_v1.loc[X_val.index, 'subject'].values

            # identify age range in pre-COVID data to be used for modeling
            agemin = X_train['agedays'].min()
            agemax = X_train['agedays'].max()

            write_ages_to_file_by_gender(agemin, agemax, working_dir, sex)

            # drop the agegrp column from the train  and val data set because we want to use agedays as a predictor
            X_train.drop(columns=['agegrp'], inplace=True)
            X_val.drop(columns=['agegrp'], inplace=True)

            # Get region names from training set feature names. Feature names are the same for the validation set
            roi_ids = y_train.columns.str.replace(fr'^{band}-', '', regex=True).tolist()

            ##########
            # Set up output directories. Save data for each brain region to its own text file, organized in separate directories,
            # because for each response variable Y (brain region) we fit a separate normative mode
            ##########
            for c in y_train.columns:
                roi = c[len(band) + 1:]
                y_train[c].to_csv(f'{working_dir}/resp_tr_' + roi + '.txt', header=False, index=False)
                X_train.to_csv(f'{working_dir}/cov_tr.txt', sep='\t', header=False, index=False)
                y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)

            for c in y_val.columns:
                roi = c[len(band) + 1:]
                y_val[c].to_csv(f'{working_dir}/resp_te_' + roi + '.txt', header=False, index=False)
                X_val.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
                y_val.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

            for i in roi_ids:
                roidirname = '{}/{}/{}_{}/ROI_models/{}'.format(working_dir, dirdata, sex, band, i)
                recreate_folder(roidirname)

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

            movefiles("{}/resp_*.txt".format(working_dir), "{}/{}/{}_{}/response_files/".format(working_dir, dirdata, sex, band))
            movefiles("{}/cov_t*.txt".format(working_dir), "{}/{}/{}_{}/covariate_files/".format(working_dir, dirdata, sex, band))

            #  this path is where ROI_models folders are located
            data_dir = '{}/{}/{}_{}/ROI_models/'.format(working_dir, dirdata, sex, band)

            # Create Design Matrix and add in spline basis and intercept for training and validation data
            create_design_matrix_one_gender('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
            create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

            blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'z_mean', 'z_std'])

            subjects_train = subjects_train.reshape(-1, 1)
            subjects_val = subjects_val.reshape(-1, 1)
            # Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

            # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
            # specific arguments that are worth commenting on:
            # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
            # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
            # ●savemodel=True: write out the final estimated model to disk
            # ●saveoutput=False: return the outputs directly rather than writing them to disk
            # ●standardize=False: do not standardize the covariates or response variable

            # Loop through ROIs

            for roi in roi_ids:
                print(f'Running ROI {roi} Split {split+1} Band {band}')
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

                # create dummy design matrices for training set for visualizing model
                dummy_cov_file_path = create_dummy_design_matrix_one_gender(agemin, agemax, spline_order, spline_knots,
                                                                            working_dir)

                # calculate model slope
                model_slope.loc[band, roi], ymin.loc[band, roi] = calc_model_slope(dummy_cov_file_path, nm)

                # compute splines and superimpose on data for training set. Show on screen or save to file depending on show_plots value.
                #plot_data_with_spline_one_gender(sex, 'Training Data', band, cov_file_tr, resp_file_tr, dummy_cov_file_path,
                #                      model_dir, roi, show_plots, working_dir, dirdata)

                # Calculate mean and standard deviation of validation set
                z_mean = np.mean(Z_te)
                z_std = np.std(Z_te)

                # Add row to the blr metrics dataframe
                blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0],
                                                     metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0],
                                                     metrics_te['Rho'][0], z_mean, z_std]

            blr_metrics['split'] = split
            all_blr_site_metrics_dict[band] = pd.concat([all_blr_site_metrics_dict[band], blr_metrics], ignore_index=True)

            if not evaluate_model_fit:
                Z_time2[band] = apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                    working_dir,rsd_v2, dirdata, dirpredict, sex, band, split)

                Z_time2[band]['split'] = split

                Z2_all_splits_dict[band] = pd.concat([Z2_all_splits_dict[band], Z_time2[band]], ignore_index=True)

        if evaluate_model_fit:
            # Write performance statistics to file
            with open(os.path.join(working_dir, f'blr_metrics_{sex}_{n_splits}_splits.pkl'),'wb') as f:
                pickle.dump(all_blr_site_metrics_dict, f)

        else:
            for band, df in Z2_all_splits_dict.items():
                df.to_csv(os.path.join(working_dir, 'output_data', f'Z2_allsplits_{band}_{n_splits}splits.csv'), index=False)

                # Z2_all_splits_dict[band] = {}

            # Save model slopes and ymins to file
            model_slope['split'] = split
            ymin['split'] = split
            slope_file_path = f'{working_dir}/output_data/{sex}_{n_splits}_splits_allsplits_slopes.csv'
            ymin_file = f'{working_dir}/output_data/{sex}_{n_splits}_splits_ymin.csv'
            model_slope.to_csv(slope_file_path, mode='a', index=True, header = not os.path.isfile(slope_file_path))
            ymin.to_csv(ymin_file, mode='a', index=True, header = not os.path.isfile(ymin_file))

        end_time = time.time()
        elapsed = (end_time - start_time)/60.0
        elapsed_all_time = (end_time - all_split_start)/60.0

        print(f'Elapsed time for split{split+1} is {elapsed:.2f} minutes')
        print(f'Elapsed time for program {n_splits} splits is {elapsed_all_time:.2f} minutes')

    if not evaluate_model_fit:
        return Z2_all_splits_dict
