# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/fit_spline_model.py
# Compiled at: 2024-01-11 16:33:49
# Size of source mod 2**32: 4512 bytes


def FitSplineModel(workingdir, dirtag, spline_order, spline_knots, X_train, y_train, X_test, y_test, agemin, agemax):
    import pandas as pd, os, shutil
    from pcntoolkit.normative import estimate
    from helper_functions_MEG import create_design_matrix
    from helper_functions_MEG import makenewdir, movefiles
    os.chdir(workingdir)
    roi_ids = y_train.columns.to_list()
    for c in y_train.columns:
        y_train[c].to_csv(("resp_tr_" + c + ".txt"), header=False, index=False)
        X_train.to_csv("cov_tr.txt", sep="\t", header=False, index=False)
        y_train.to_csv("resp_tr.txt", sep="\t", header=False, index=False)
    else:
        for c in y_test.columns:
            y_test[c].to_csv(("resp_te_" + c + ".txt"), header=False, index=False)
            X_test.to_csv("cov_te.txt", sep="\t", header=False, index=False)
            y_test.to_csv("resp_te.txt", sep="\t", header=False, index=False)
        else:
            for i in roi_ids:
                roidirname = "{}/data/{}/ROI_models/{}".format(workingdir, dirtag, i)
                makenewdir(roidirname)
                resp_tr_filename = "resp_tr_{}.txt".format(i)
                resp_tr_filepath = roidirname + "/resp_tr.txt"
                shutil.copyfile(resp_tr_filename, resp_tr_filepath)
                resp_te_filename = "resp_te_{}.txt".format(i)
                resp_te_filepath = roidirname + "/resp_te.txt"
                shutil.copyfile(resp_te_filename, resp_te_filepath)
                cov_tr_filepath = roidirname + "/cov_tr.txt"
                shutil.copyfile("cov_tr.txt", cov_tr_filepath)
                cov_te_filepath = roidirname + "/cov_te.txt"
                shutil.copyfile("cov_te.txt", cov_te_filepath)
            else:
                movefiles("resp_*.txt", workingdir + "/data/" + dirtag + "/response_files/")
                movefiles("cov_t*.txt", workingdir + "/data/" + dirtag + "/covariate_files/")
                cross_val_roi_dir = "{}/data/{}/ROI_models/".format(workingdir, dirtag)
                create_design_matrix("test", agemin, agemax, spline_order, spline_knots, roi_ids, cross_val_roi_dir)
                create_design_matrix("train", agemin, agemax, spline_order, spline_knots, roi_ids, cross_val_roi_dir)
                df_metrics_te = pd.DataFrame()
                for roi in roi_ids:
                    print("Running ROI:", roi)
                    roi_dir = os.path.join(cross_val_roi_dir, roi)
                    os.chdir(roi_dir)
                    cov_file_tr = os.path.join(cross_val_roi_dir, roi, "cov_bspline_tr.txt")
                    cov_file_te = os.path.join(cross_val_roi_dir, roi, "cov_bspline_te.txt")
                    resp_file_tr = os.path.join(cross_val_roi_dir, roi, "resp_tr.txt")
                    resp_file_te = os.path.join(cross_val_roi_dir, roi, "resp_te.txt")
                    yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, testcov=cov_file_te,
                      alg="blr",
                      optimizer="powell",
                      savemodel=False,
                      saveoutput=False,
                      standardize=False)
                    metrics_te["RMSE"] = metrics_te["RMSE"][0]
                    metrics_te["Rho"] = metrics_te["Rho"][0]
                    metrics_te["pRho"] = metrics_te["pRho"][0]
                    metrics_te["SMSE"] = metrics_te["SMSE"][0]
                    metrics_te["EXPV"] = metrics_te["EXPV"][0]
                    metrics_te["MSLL"] = metrics_te["MSLL"][0]
                    metrics_te["NLL"] = metrics_te["NLL"].item()
                    metrics_te["BIC"] = metrics_te["BIC"].item()
                    tmp = pd.DataFrame([metrics_te])
                    df_metrics_te = pd.concat([df_metrics_te, pd.DataFrame([metrics_te])])
                    df_metrics_te.reset_index(inplace=True, drop=True)
                else:
                    return df_metrics_te
