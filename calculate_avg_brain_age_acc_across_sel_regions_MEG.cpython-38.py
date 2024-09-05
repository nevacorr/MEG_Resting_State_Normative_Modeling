# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/calculate_avg_brain_age_acc_across_sel_regions_MEG.py
# Compiled at: 2024-01-19 16:31:48
# Size of source mod 2**32: 9663 bytes
from helper_functions_MEG import makenewdir, movefiles, create_design_matrix, create_dummy_design_matrix
from helper_functions_MEG import plot_y_v_yhat, plot_data_with_spline
import shutil, pandas as pd, os
from pcntoolkit.normative import estimate
import matplotlib.pyplot as plt
from predict_neva_from_normativepy import predict_neva
from plot_z_scores import plot_and_compute_zcores_by_gender

def calculate_avg_brain_age_acceleration_across_select_regions(band, desc_string, X_train, y_train, X_test, y_test, struct_var, show_plots, spline_order, spline_knots, filepath):
    makenewdir("{}/avgmeg".format(filepath))
    makenewdir("{}/avgmeg/{}".format(filepath, band))
    makenewdir("{}/avgmeg/{}/data".format(filepath, band))
    makenewdir("{}/avgmeg/{}/data/plots".format(filepath, band))
    makenewdir("{}/avgmeg/{}/data/ROI_models".format(filepath, band))
    makenewdir("{}/avgmeg/{}/data/covariate_files".format(filepath, band))
    makenewdir("{}/avgmeg/{}/data/response_files".format(filepath, band))
    agemin = X_train["agedays"].min()
    agemax = X_train["agedays"].max()
    s_index_train = X_train.index.values
    subjects_train = X_train.loc[(s_index_train, "subject")].values
    s_index_test = X_test.index.values
    subjects_test = X_test.loc[(s_index_test, "subject")].values
    X_train.drop(columns=["subject", "agegrp"], inplace=True)
    X_test.drop(columns=["subject", "agegrp"], inplace=True)
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()
    y_train.columns = ["avgmeg"]
    y_test.columns = ["avgmeg"]
    for c in ('avgmeg', ):
        y_train[c].to_csv((f"{filepath}/resp_tr_" + c + ".txt"), header=False, index=False)
        X_train.to_csv(f"{filepath}/cov_tr.txt", sep="\t", header=False, index=False)
        y_train.to_csv(f"{filepath}/resp_tr.txt", sep="\t", header=False, index=False)
    else:
        for i in ('avgmeg', ):
            roidirname = "{}/avgmeg/{}/data/ROI_models/{}".format(filepath, band, i)
            makenewdir(roidirname)
            resp_tr_filename = "{}/resp_tr_{}.txt".format(filepath, i)
            resp_tr_filepath = roidirname + "/resp_tr.txt"
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)
            cov_tr_filepath = roidirname + "/cov_tr.txt"
            shutil.copyfile(f"{filepath}/cov_tr.txt", cov_tr_filepath)
        else:
            movefiles(f"{filepath}/resp_*.txt", "{}/avgmeg/{}/data/response_files/".format(filepath, band))
            movefiles(f"{filepath}/cov_t*.txt", "{}/avgmeg/{}/data/covariate_files/".format(filepath, band))
            data_dir = "{}/avgmeg/{}/data/ROI_models/".format(filepath, band)
            create_design_matrix("train", agemin, agemax, spline_order, spline_knots, [desc_string], data_dir)
            blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
            blr_site_metrics = pd.DataFrame(columns=['ROI', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 
             'SMSE', 'RMSE', 'Rho'])
            subjects_train = subjects_train.reshape(-1, 1)
            Z_score_train_matrix = pd.DataFrame(subjects_train, columns=["subject_id_train"])
            for roi in (
             desc_string,):
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
                Rho_tr = metrics_tr["Rho"]
                EV_tr = metrics_tr["EXPV"]
                dummy_cov_file_path_female, dummy_cov_file_path_male = create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_tr, spline_order, spline_knots, filepath + "/avgmeg/" + band)
                plot_data_with_spline("Training Data", struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi, show_plots, filepath)
            else:
                plt.show()
                makenewdir("{}/avgmeg/{}/predict_files/".format(filepath, band))
                makenewdir("{}/avgmeg/{}/predict_files/".format(filepath, band))
                makenewdir("{}/avgmeg/{}/predict_files/plots".format(filepath, band))
                makenewdir("{}/avgmeg/{}/predict_files/ROI_models".format(filepath, band))
                makenewdir("{}/avgmeg/{}/predict_files/covariate_files".format(filepath, band))
                makenewdir("{}/avgmeg/{}/predict_files/response_files".format(filepath, band))
                training_dir = data_dir
                out_dir = f"{filepath}/avgmeg/{band}/predict_files/ROI_models/"
                predict_files_dir = f"{filepath}/avgmeg/{band}/predict_files/ROI_models/"
                for c in ('avgmeg', ):
                    y_test[c].to_csv((f"{filepath}/resp_te_" + c + ".txt"), header=False, index=False)
                    X_test.to_csv(f"/{filepath}/cov_te.txt", sep="\t", header=False, index=False)
                    y_test.to_csv(f"{filepath}/resp_te.txt", sep="\t", header=False, index=False)
                else:
                    for i in ('avgmeg', ):
                        roidirname = f"{filepath}/avgmeg/{band}/predict_files/ROI_models/{i}"
                        makenewdir(roidirname)
                        resp_te_filename = f"{filepath}/resp_te_{i}.txt"
                        resp_te_filepath = roidirname + "/resp_te.txt"
                        shutil.copyfile(resp_te_filename, resp_te_filepath)
                        cov_te_filepath = roidirname + "/cov_te.txt"
                        shutil.copyfile(f"{filepath}/cov_te.txt", cov_te_filepath)
                    else:
                        movefiles(f"{filepath}/resp_*.txt", f"{filepath}/avgmeg/alpha/predict_files/response_files/")
                        movefiles(f"{filepath}/cov_t*.txt", f"{filepath}/avgmeg/alpha/predict_files/covariate_files/")
                        roi_ids = [
                         "avgmeg"]
                        create_design_matrix("test", agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)
                        subjects_test = subjects_test.reshape(-1, 1)
                        Z_time2 = pd.DataFrame(subjects_test, columns=["subject_id_test"])
                        create_design_matrix("test", agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)
                        for roi in roi_ids:
                            print("Running ROI:", roi)
                            roi_dir = os.path.join(predict_files_dir, roi)
                            model_dir = os.path.join(training_dir, roi, "Models")
                            os.chdir(roi_dir)
                            cov_file_te = os.path.join(roi_dir, "cov_bspline_te.txt")
                            resp_file_te = os.path.join(roi_dir, "resp_te.txt")
                            yhat_te, s2_te, Z = predict_neva(cov_file_te, respfile=resp_file_te, alg="blr", model_path=model_dir)
                            Z_time2[roi] = Z
                        else:
                            Z_time2.to_csv(("{}/avgmeg/{}/predict_files/Z_scores_by_region_postcovid_testset_avgmeg.txt".format(filepath, band)),
                              index=False)
                            Z_time2.rename(columns={'subject_id_test':"participant_id",  'avgmeg':f"avgmeg_{band}"}, inplace=True)
                            plot_and_compute_zcores_by_gender(Z_time2, struct_var, [f"avgmeg_{band}"], filepath)
                            plt.show()
                            return Z_time2
