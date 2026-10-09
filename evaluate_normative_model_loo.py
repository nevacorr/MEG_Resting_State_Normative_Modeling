from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import explained_variance_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import shutil
from helper_functions_MEG import recreate_folder, create_dummy_design_matrix, create_design_matrix
from pcn_estimate_myedit_loo import estimate
# from pcntoolkit.normative import estimate

"""
Perform leave-one-out cross validation on the normative training set
"""

def evaluate_normative_model_loo(rs_covariates, rs_features, band, roi_ids, data_dir, spline_order, spline_knots):

    loo = LeaveOneOut()

    blr_metrics = pd.DataFrame(
        columns=['ROI', 'EV', 'RMSE', 'Rho', 'z_mean', 'z_std']
    )

    for roi in roi_ids:

        print(f'ROI: {roi}')

        y_true_all = []
        yhat_all = []
        z_all = []

        # Leave-one-out loop
        for fold, (train_idx, val_idx) in enumerate(loo.split(rs_covariates), start=1):
            print(f'    LOO {fold}/{len(rs_covariates)}')

            # Split covariates and response
            X_train = rs_covariates.iloc[train_idx].copy()
            X_val = rs_covariates.iloc[val_idx].copy()

            y_train = rs_features.iloc[train_idx][[band + '-' + roi]].copy()
            y_val = rs_features.iloc[val_idx][[band + '-' + roi]].copy()

            # Get age range for model
            agemin = rs_covariates['agedays'].min()
            agemax = rs_covariates['agedays'].max()

            # Drop agegrp as covariate
            X_train = X_train.drop(columns=['agegrp'])
            X_val = X_val.drop(columns=['agegrp'])

            # Create ROI directory
            roidirname = os.path.join(data_dir, roi, f'LOOCV_{fold}')
            roi_data_dir = os.path.join(roidirname, roi)
            recreate_folder(roi_data_dir)

            # Write response/covariate files
            resp_tr_filepath = os.path.join(roi_data_dir, 'resp_tr.txt')
            resp_te_filepath = os.path.join(roi_data_dir, 'resp_te.txt')

            cov_tr_filepath = os.path.join(roi_data_dir, 'cov_tr.txt')
            cov_te_filepath = os.path.join(roi_data_dir, 'cov_te.txt')

            y_train.to_csv(resp_tr_filepath, sep='\t', header=False, index=False)
            np.savetxt(resp_te_filepath, y_val.values.reshape(1, -1), delimiter='\t')

            X_train.to_csv(cov_tr_filepath, sep='\t', header=False, index=False)
            np.savetxt(cov_te_filepath, X_val.values.reshape(1, -1), delimiter='\t')

            # Create spline design matrices
            create_design_matrix('train', agemin, agemax, spline_order, spline_knots, [roi], roidirname)
            create_design_matrix('test', agemin, agemax, spline_order, spline_knots, [roi], roidirname)

            # Define paths to spline design matrices
            cov_file_tr = os.path.join(roidirname, roi, 'cov_bspline_tr.txt')
            cov_file_te = os.path.join(roidirname, roi, 'cov_bspline_te.txt')

            # Fit on train set and predict held-out subject
            yhat_te, s2_te, nm, Z_te, metrics_te = (
                estimate(cov_file_tr, resp_tr_filepath, testresp=resp_te_filepath, testcov=cov_file_te, alg='blr',
                         optimizer='powell', savemodel=False, saveoutput=False, standardize=False))

            # Store held-out prediction
            y_true_all.append(float(y_val.iloc[0, 0]))
            yhat_all.append(float(np.asarray(yhat_te).ravel()[0]))
            z_all.append(float(np.asarray(Z_te).ravel()[0]))

        # Combine all LOO predictions
        y_true_all = np.asarray(y_true_all)
        yhat_all = np.asarray(yhat_all)
        z_all = np.asarray(z_all)

        # Calculate performance across ALL held-out sujects
        EV = explained_variance_score(y_true_all, yhat_all)

        RMSE = np.sqrt(mean_squared_error(y_true_all, yhat_all))

        # Correlation
        if np.std(y_true_all) > 0 and np.std(yhat_all) > 0:
            Rho = np.corrcoef(y_true_all, yhat_all)[0, 1]
        else:
            Rho = np.nan

        z_mean = np.mean(z_all)
        z_std = np.std(z_all)

        blr_metrics.loc[len(blr_metrics)] = [roi, EV, RMSE, Rho, z_mean, z_std]

    return blr_metrics