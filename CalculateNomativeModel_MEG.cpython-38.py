# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/CalculateNomativeModel_MEG.py
# Compiled at: 2024-01-12 14:28:33
# Size of source mod 2**32: 10508 bytes
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil
from pcntoolkit.normative import estimate
from helper_functions_MEG import create_design_matrix, plot_data_with_spline, create_dummy_design_matrix
from helper_functions_MEG import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from predict_neva_from_normativepy import predict_neva

def calculate_normative_model_function(X_train, y_train, X_test, y_test, spline_order, spline_knots, show_plots, targetstr, outputdir):
    makenewdir("{}/data/".format(outputdir))
    makenewdir("{}/data/{}".format(outputdir, targetstr))
    makenewdir("{}/data/{}/plots".format(outputdir, targetstr))
    makenewdir("{}/data/{}/ROI_models".format(outputdir, targetstr))
    makenewdir("{}/data/{}/covariate_files".format(outputdir, targetstr))
    makenewdir("{}/data/{}/response_files".format(outputdir, targetstr))
    makenewdir("{}/predict_files/".format(outputdir))
    makenewdir("{}/predict_files/{}".format(outputdir, targetstr))
    makenewdir("{}/predict_files/{}/plots".format(outputdir, targetstr))
    makenewdir("{}/predict_files/{}/ROI_models".format(outputdir, targetstr))
    makenewdir("{}/predict_files/{}/covariate_files".format(outputdir, targetstr))
    makenewdir("{}/predict_files/{}/response_files".format(outputdir, targetstr))
    training_dir = "{}/data/{}/ROI_models/".format(outputdir, targetstr)
    predict_dir = "{}/predict_files/{}/ROI_models/".format(outputdir, targetstr)
    agemin = X_train["agedays"].min()
    agemax = X_train["agedays"].max()
    subjects_train = X_train.loc[(None[:None], "subject")].tolist()
    subjects_test = X_test.loc[(None[:None], "subject")].tolist()
    X_train_cp = X_train.copy()
    X_test_cp = X_test.copy()
    X_train_cp.drop(columns=["subject", "agegrp"], inplace=True)
    X_test_cp.drop(columns=["subject", "agegrp"], inplace=True)
    gender_col_ind = X_train_cp.columns.get_loc("gender")
    agedays_col_ind = X_train_cp.columns.get_loc("agedays")
    target_cols = y_train.columns.tolist()
    for c in target_cols:
        y_train[c].to_csv((f"{outputdir}/resp_tr_" + c + ".txt"), header=False, index=False)
        X_train_cp.to_csv(f"{outputdir}/cov_tr.txt", sep="\t", header=False, index=False)
        y_train.to_csv(f"{outputdir}/resp_tr.txt", sep="\t", header=False, index=False)
    else:
        for i in target_cols:
            roidirname = "{}/data/{}/ROI_models/{}".format(outputdir, targetstr, i)
            makenewdir(roidirname)
            resp_tr_filename = "{}/resp_tr_{}.txt".format(outputdir, i)
            resp_tr_filepath = roidirname + "/resp_tr.txt"
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)
            cov_tr_filepath = roidirname + "/cov_tr.txt"
            shutil.copyfile(f"{outputdir}/cov_tr.txt", cov_tr_filepath)
        else:
            movefiles("resp_*.txt", "{}/data/{}/response_files/".format(outputdir, targetstr))
            movefiles("cov_t*.txt", "{}/data/{}/covariate_files/".format(outputdir, targetstr))
            for c in target_cols:
                y_test[c].to_csv((f"{outputdir}/resp_te_" + c + ".txt"), header=False, index=False)
                X_test_cp.to_csv(f"{outputdir}/cov_te.txt", sep="\t", header=False, index=False)
                y_test.to_csv(f"{outputdir}/resp_te.txt", sep="\t", header=False, index=False)
            else:
                for i in target_cols:
                    roidirname = "{}/predict_files/{}/ROI_models/{}".format(outputdir, targetstr, i)
                    makenewdir(roidirname)
                    resp_te_filename = "{}/resp_te_{}.txt".format(outputdir, i)
                    resp_te_filepath = roidirname + "/resp_te.txt"
                    shutil.copyfile(resp_te_filename, resp_te_filepath)
                    cov_te_filepath = roidirname + "/cov_te.txt"
                    shutil.copyfile(f"{outputdir}/cov_te.txt", cov_te_filepath)
                else:
                    movefiles("resp_*.txt", "{}/predict_files/{}/response_files/".format(outputdir, targetstr))
                    movefiles("cov_t*.txt", "{}/predict_files/{}/covariate_files/".format(outputdir, targetstr))
                    data_dir = "{}/data/{}/ROI_models/".format(outputdir, targetstr)
                    create_design_matrix("train", agemin, agemax, spline_order, spline_knots, target_cols, data_dir)
                    blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
                    Z_score_test_matrix = pd.DataFrame(subjects_test, columns=["subject_id_test"])
                    Z_score_train_matrix = pd.DataFrame(subjects_train, columns=["subject_id_train"])
                    y_test_prediction_error_matrix = pd.DataFrame(subjects_test, columns=["subjects_id_test"])
                    for roi in target_cols:
                        print("Running ROI:", roi)
                        roi_dir = os.path.join(data_dir, roi)
                        model_dir = os.path.join(data_dir, roi, "Models")
                        os.chdir(roi_dir)
                        cov_file_tr = os.path.join(roi_dir, "cov_bspline_tr.txt")
                        resp_file_tr = os.path.join(roi_dir, "resp_tr.txt")
                        yhat_tr, s2_tr, nm_tr, Z_tr, metrics_tr = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_tr, testcov=cov_file_tr,
                          alg="blr",
                          optimizer="powell",
                          savemodel=True,
                          saveoutput=False,
                          standardize=False)
                        Z_score_train_matrix[roi] = Z_tr
                        blr_metrics.loc[len(blr_metrics)] = [
                         roi, metrics_tr["MSLL"][0],
                         metrics_tr["EXPV"][0], metrics_tr["SMSE"][0], metrics_tr["RMSE"][0], metrics_tr["Rho"][0]]
                        if show_plots:
                            Rho_tr = metrics_tr["Rho"]
                            EV_tr = metrics_tr["EXPV"]
                            plot_y_v_yhat(cov_file_tr, resp_file_tr, yhat_tr, "Training Data", targetstr, roi, Rho_tr, EV_tr, gender_col_ind, outputdir)
                            dummy_cov_file_path_female, dummy_cov_file_path_male = create_dummy_design_matrix(targetstr, agemin, agemax, cov_file_tr, spline_order, spline_knots, outputdir)
                            plot_data_with_spline("Training Data", targetstr, cov_file_tr, resp_file_tr, dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi, show_plots, outputdir)
                            plt.show()
                    else:
                        create_design_matrix("test", agemin, agemax, spline_order, spline_knots, target_cols, predict_dir)

            for roi in target_cols:
                print("Running ROI:", roi)
                roi_dir = os.path.join(predict_dir, roi)
                model_dir = os.path.join(training_dir, roi, "Models")
                os.chdir(roi_dir)
                cov_file_te = os.path.join(roi_dir, "cov_bspline_te.txt")
                resp_file_te = os.path.join(roi_dir, "resp_te.txt")
                yhat_te, s2_te, Z = predict_neva(cov_file_te, respfile=resp_file_te, alg="blr", model_path=model_dir)
                Z_score_test_matrix[roi] = Z
                tmp = Z_score_test_matrix.loc[(None[:None], roi)]
                y_test_prediction_error_matrix[roi] = yhat_te - Z_score_test_matrix.loc[(None[:None], roi)].to_numpy().reshape(-1, 1)
                if show_plots:
                    dummy_cov_file_path_female, dummy_cov_file_path_male = create_dummy_design_matrix(targetstr, agemin, agemax, cov_file_te, spline_order, spline_knots, outputdir)
                    plot_data_with_spline("Test Data", targetstr, cov_file_te, resp_file_te, dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi, show_plots, outputdir)
                if show_plots:
                    blr_metrics.sort_values(by=["Rho"], inplace=True, ignore_index=True)
                    barplot_performance_values(targetstr, "Rho", blr_metrics, spline_order, spline_knots, outputdir)
                    blr_metrics.sort_values(by=["EV"], inplace=True, ignore_index=True)
                    barplot_performance_values(targetstr, "EV", blr_metrics, spline_order, spline_knots, outputdir)
                    plt.show()
                return (Z_score_train_matrix, Z_score_test_matrix)
