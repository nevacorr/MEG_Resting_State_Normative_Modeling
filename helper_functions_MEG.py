# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/helper_functions_MEG.py
# Compiled at: 2024-01-19 12:53:25
# Size of source mod 2**32: 17165 bytes
import os
import numpy as np
import shutil
from matplotlib import pyplot as plt
from pcntoolkit.normative import predict
import pandas as pd, seaborn as sns, shutil, glob
from pcntoolkit.util.utils import create_bspline_basis
import math
from scipy import stats
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

def write_list_to_file(mylist, filepath):
    with open(filepath, "w") as file:
        for item in mylist:
            file.write(item + "\n")

def plot_num_subjs(gender, df, title, struct_var, timept, path):
    sns.set(font_scale=1)
    sns.set_style(style="white")
    if gender == 'female':
        c = 'crimson'
    elif gender == 'male':
        c = 'blue'
    g = sns.catplot(x="agegrp", color=c, data=df, kind="count", legend=False)
    g.fig.suptitle(title, fontsize=10)
    g.fig.subplots_adjust(top=0.85)
    g.ax.set_xlabel("Age", fontsize=10)
    g.ax.set_ylabel("Number of Subjects", fontsize=8)
    g.ax.tick_params(axis="x", labelsize=8)
    g.ax.tick_params(axis="y", labelsize=8)
    g.ax.set(yticks=(np.arange(0, 20, 2)))
    plt.show(block=False)
    plt.savefig("{}/data/{}_NumSubjects_{}".format(path, struct_var, timept))


def makenewdir(path):
    isExist = os.path.exists(path)
    if isExist is False:
        os.mkdir(path)
        print("made directory {}".format(path))

def makenewdir_deleteold(path):
    isExist = os.path.exists(path)
    if isExist is True:
        shutil.rmtree(path)
    os.mkdir(path)
    print("made directory {}". format(path))


def movefiles(pattern, folder):
    files = glob.glob(pattern)
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, folder + file_name)
        print("moved:", file)


def create_design_matrix_one_gender(datatype, agemin, agemax, spline_order, spline_knots, roi_ids, data_dir):
    B = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    for roi in roi_ids:
        print('Creating basis expansion for ROI:', roi)
        roi_dir = os.path.join(data_dir, roi)
        os.chdir(roi_dir)
        # create output dir
        os.makedirs(os.path.join(roi_dir, 'blr'), exist_ok=True)

        # load train & test covariate data matrices
        if datatype == 'train':
            X = np.loadtxt(os.path.join(roi_dir, 'cov_tr.txt'))
        elif datatype == 'test':
            X = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))

        # Add intercept column
        X = np.vstack((X, np.ones(len(X)))).T

        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_int_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_int_te.txt'), X)

        # create Bspline basis set
        # This creates a numpy array called Phi by applying function B to each element of the first column of X_tr
        Phi = np.array([B(i) for i in X[:, 0]])
        X = np.concatenate((X, Phi), axis=1)
        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_te.txt'), X)

def create_dummy_design_matrix_one_gender(band, agemin, agemax, cov_file, spline_order, spline_knots, outputdir):

    # Make dummy test data covariate file starting with a column for age
    dummy_cov = np.linspace(agemin, agemax, num=1000)
    ones = np.ones((1, dummy_cov.shape[0]))

    # Add a column for intercept
    dummy_cov_final = np.vstack((dummy_cov, ones)).T

    # Create spline features and add them to predictor dataframe
    BAll = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    Phidummy = np.array([BAll(i) for i in dummy_cov_final[:, 0]])
    dummy_cov_final = np.concatenate((dummy_cov_final, Phidummy), axis=1)

    # Write these new created predictor variables with spline and response variable to file
    dummy_cov_file_path = os.path.join(outputdir, 'cov_file_dummy.txt')
    np.savetxt(dummy_cov_file_path, dummy_cov_final)
    return dummy_cov_file_path

def plot_data_with_spline_one_gender_rescale(gender, datastr, band, cov_file, resp_file, dummy_cov_file_path, model_dir, roi,
                                     showplots, working_dir, minmax_scaler, regnum, bandnum, total_reg_num):

    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy=output[0]

    # Load real data predictor variables for region
    X = np.loadtxt(cov_file)
    # Load real data response variables for region
    y = np.loadtxt(resp_file)

    # Create dataframes for plotting with seaborn facetgrid objects
    dummy_cov = np.loadtxt(dummy_cov_file_path)

    df_origdata = pd.DataFrame(data=X[:, 0], columns=['Age in Days'])
    df_origdata[band] = y * minmax_scaler.data_range_[regnum + (bandnum * total_reg_num)] + minmax_scaler.data_min_[regnum + (bandnum * total_reg_num)]
    df_origdata['Age in Days'] = (df_origdata['Age in Days'] * minmax_scaler.data_range_[-1] + minmax_scaler.data_min_[-1]) / 365.25

    df_estspline = pd.DataFrame(data=dummy_cov[:, 0].tolist(),columns=['Age in Days'])
    df_estspline['Age in Days'] = (df_estspline['Age in Days'] * minmax_scaler.data_range_[-1] + minmax_scaler.data_min_[-1] ) / 365.25
    tmp = np.array(yhat_predict_dummy.tolist(), dtype=float)
    df_estspline[band] = tmp * minmax_scaler.data_range_[regnum + (bandnum * total_reg_num)] + minmax_scaler.data_min_[regnum + (bandnum * total_reg_num)]
    df_estspline = df_estspline.drop(index=df_estspline.iloc[999].name).reset_index(drop=True)

    # PLot figure
    fig=plt.figure()
    if gender == 'female':
        color = 'crimson'
    else:
        color = 'blue'
    sns.lineplot(data=df_estspline, x='Age in Days', y=band, color=color, legend=False)
    sns.scatterplot(data=df_origdata, x='Age in Days', y=band, color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(datastr +' ' + band +  ' vs. Age\n' + roi.replace(band+'-', ''))
    plt.xlabel('Age')
    plt.ylabel(datastr + band)
    if showplots == 1:
        if datastr == 'Training Data':
            plt.show(block=False)
        else:
            plt.show()
    else:
      plt.savefig('{}/data/{}_{}/plots/{}_{}_{}_vs_age_withsplinefit_{}'
                 .format(working_dir, gender, band, gender, band, roi.replace(band+'-', ''), datastr))
      plt.close(fig)
      if datastr == 'Training Data':
         splinemodel_fname = f'{working_dir}/data/{gender}_{band}/plots/spline_model_{datastr}_{roi}_{gender}.csv'
         origdata_fname = f'{working_dir}/data/{gender}_{band}/plots/datapoints_{datastr}_{roi}_{gender}.csv'
         df_estspline.to_csv(splinemodel_fname)
         df_origdata.to_csv(origdata_fname)

def plot_y_v_yhat(cov_file, resp_file, yhat, typestring, gender, band, roi, Rho, EV, working_dir):
    cov_data = np.loadtxt(cov_file)
    y = np.loadtxt(resp_file).reshape(-1,1)
    dfp = pd.DataFrame()
    y=y.flatten()
    dfp['y'] = y
    dfp['yhat'] = yhat
    print(dfp.dtypes)
    fig = plt.figure()
    if gender == 'female':
        color='green'
    else:
        color='blue'

    sns.scatterplot(data=dfp, x='y', y='yhat', color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(typestring + ' ' + gender + ' ' + band + ' vs. estimate\n'
              + roi +' EV=' + '{:.4}'.format(str(EV.item())) + ' Rho=' + '{:.4}'.format(str(Rho.item())))
    plt.xlabel(typestring + ' ' + band)
    plt.ylabel(band + ' estimate on ' + typestring)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # plots line y = x
    plt.show(block=False)

def barplot_performance_values(struct_var, metric, df, spline_order, spline_knots, outputdir):
    colors = ["blue" if "lh" in x else "green" for x in df.ROI]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=(df[metric]), y=(df["ROI"]), orient="h", palette=colors)
    plt.subplots_adjust(left=0.4)
    plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(bottom=0.05)
    ax.set_title("Test Set " + metric + " for All Brain Regions")
    plt.savefig("{}/data/{}/plots/Test_Set_{}_for_all_regions_splineorder{}, splineknots{}.png".format(outputdir, struct_var, metric, spline_order, spline_knots))


def write_ages_to_file_by_gender(agemin, agemax,outputdir, gender):
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(outputdir, gender), "w") as file:
        file.write(str(agemin) + "\n")
        file.write(str(agemax) + "\n")


def read_ages_from_file(struct_var, outputdir, gender):
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(outputdir,gender), "r") as file:
        lines = file.readlines()
    agemin = float(lines[0].strip())
    agemax = float(lines[1].strip())
    return (agemin, agemax)

def plot_feature_distributions(data_df, column_list):
    df = data_df[column_list].copy()

    #Plot distributions
    n_rows=5
    n_cols=6

    sns.set(font_scale=0.5)
    num_df_cols=df.shape[1]
    m=math.ceil(num_df_cols/30)*30

    for row in range(0,m,30):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.set_size_inches(12, 15)
        for i, column in enumerate(column_list[row:row+30]):
            ax = axes[i//n_cols,i%n_cols]
            sns.histplot(df[column],ax=ax)
        plt.show(block=False)

def remove_outliers_IQR(df, cols):
    # IQR method to remove outliers
    # Initialize an empty list to hold indices of rows to be removed
    outlier_indices = []

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Find outlier indices for the current column
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index
        outlier_indices.extend(outliers)

    # Convert the list to a set to remove duplicates (same row might be an outlier in multiple columns)
    outlier_indices = set(outlier_indices)

    # Drop these outliers from the DataFrame
    df_no_outliers = df.drop(outlier_indices)

    return df_no_outliers

def fit_regression_model_dummy_data_one_gender(model_dir, dummy_cov_file_path):
    # create dummy data to find equation for linear regression fit between age and structvar
    dummy_predictors = pd.read_csv(dummy_cov_file_path, delim_whitespace=True, header=None)
    dummy_ages = dummy_predictors.iloc[:, 0]

    # calculate predicted values for dummy covariates for male and female
    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)
    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy = output[0]

    # remove last element of age and output arrays
    last_index = len(yhat_predict_dummy) - 1
    yhat_predict_dummy = np.delete(yhat_predict_dummy, -1)
    dummy_ages = np.delete(dummy_ages.to_numpy(), -1)

    # find slope and intercept of lines
    slope, intercept, rvalue, pvalue, std_error = stats.linregress(dummy_ages, yhat_predict_dummy)

    return slope, intercept, pvalue


def read_text_list(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    mylist = [line.strip() for line in lines]
    return mylist


# def plot_scatter_with_trendline_corthick_MEGrs_byreg(df, band, cortthick_cols, band_cols, cortthick_str_to_remove, band_str_to_remove, mycolor):
#     all_band_col_names = []
#     allslope = []
#     allintercept = []
#     allr = []
#     allp = []
#     for i, ct_col_name in enumerate(cortthick_cols):
#         cortthick_col_split = ct_col_name.split("-")
#         band_col_name = band + "_" + cortthick_col_split[2] + "-" + cortthick_col_split[1]
#         all_band_col_names.append(band_col_name)
#         slope, intercept, r, p, std_error = linregress(df.loc[(None[:None], band_col_name)], df.loc[(None[:None], ct_col_name)])
#         allslope.append(slope)
#         allintercept.append(intercept)
#         allr.append(r)
#         allp.append(p)
#         mysstop = 1
#     else:
#         output = multipletests(allp, alpha=0.05, method="fdr_bh")
#         correctedp = output[1]
#         rejectho = output[0]
#         rejectho_list = rejectho.tolist()
#         ct_cols_to_plot = [cortthick_cols[i] for i in range(len(cortthick_cols))]
#         ct_cols_to_plot_orig_ind = [i for i, x in enumerate(rejectho_list) if x]
#         p_to_plot = [correctedp[i] for i in ct_cols_to_plot_orig_ind]
#         r_to_plot = [allr[i] for i in ct_cols_to_plot_orig_ind]
#         slope_to_plot = [allslope[i] for i in ct_cols_to_plot_orig_ind]
#         intercept_to_plot = [allintercept[i] for i in ct_cols_to_plot_orig_ind]
#         for i, ct_col_name in enumerate(ct_cols_to_plot):
#             ct_col_split = ct_col_name.split("-")
#             band_col_name = band + "_" + ct_col_split[2] + "-" + ct_col_split[1]
#             plt.scatter((df.loc[(None[:None], band_col_name)]), (df.loc[(None[:None], ct_col_name)]), s=12, color=mycolor)
#             trendline = slope_to_plot[i] * df.loc[(None[:None], band_col_name)] + intercept_to_plot[i]
#             plt.plot((df.loc[(None[:None], band_col_name)]), trendline, color=mycolor, linewidth=1, label="trendline")
#             plt.xlabel("z {band} power", fontsize=8)
#             plt.ylabel("z cortthick", fontsize=8)
#             region = ct_col_name.replace(cortthick_str_to_remove, "")
#             plt.title(f"z corthick vs. z {band} power\n {region} \nr = : {r_to_plot[i]: .2f}, corrp = {p_to_plot[i]:.2f}", fontsize=8)
#             plt.tick_params(axis="x", labelsize=8)
#             plt.tick_params(axis="y", labelsize=8)
#             plt.show()
#             mystop = 1


# def plot_scatter_with_trendline_one_col_against_all_MEGrs(df, strMEGcolsofinterest, cortthick_cols, num_cols, cortthick_str_to_remove, col_int_string, cortthick_str, mycolor):
#     fig, ax = plt.subplots(1, 1)
#     df.reset_index(inplace=True, drop=True)
#     new_df = df.iloc[(None[:None], 2[:None])].copy()
#     melted_df = new_df.melt().drop("variable", axis=1).rename({"value": "z-score"}, axis=1)
#     repeated_puberty = pd.concat(([df.iloc[(None[:None], 1)]] * len(cortthick_cols)), axis=0, ignore_index=True)
#     ax.scatter(repeated_puberty, melted_df, s=12, color=mycolor)
#     slope, intercept, r, p, std_err = linregress(repeated_puberty, melted_df["z-score"])
#     trendline = slope * repeated_puberty + intercept
#     ax.plot(repeated_puberty, trendline, color=mycolor, linewidth=1, label="trendline")
#     ax.set_xlabel(col_int_string)
#     ax.set_ylabel(cortthick_str)
#     ax.set_title(f"{cortthick_str} vs {strMEGcolsofinterest} z across all regions")
#     plt.show()
#     mystop = 1
