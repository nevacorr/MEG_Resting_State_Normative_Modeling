import os
import sys
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.model_selection import KFold
from pathlib import Path

plot_model = 0

model_path = ('/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                             'Modeling/data/alpha/ROI_models/cuneus-lh/Models/')

cov_spline_data = pd.read_csv('/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                             'Modeling/data/alpha/ROI_models/cuneus-lh/cov_bspline_tr.txt', sep=' ', header=None)

# Read in covariate data for all subjects for this region as dataframe
cov_spline_data_f = cov_spline_data[cov_spline_data[1] == 0].sort_values(by=0)
cov_spline_data_m = cov_spline_data[cov_spline_data[1] == 1].sort_values(by=0)

# Remove last data point from female data set (model does not fit end point)
numrows = cov_spline_data_f.shape[0]
cov_spline_data_f = cov_spline_data_f.iloc[0:numrows - 1, :]

# Convert dataframes to numpy array
cov_spline_data_f_array = cov_spline_data_f.to_numpy()
cov_spline_data_m_array = cov_spline_data_m.to_numpy()

# Open model parameter file
with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as file:
    data = pickle.load(file)

# Calculate predictions from model based on covariate data
y_pred_f = np.dot(cov_spline_data_f_array, data.blr.m)
y_pred_m = np.dot(cov_spline_data_m_array, data.blr.m)

if plot_model:
    # plot model for this brain region
    plt.plot(cov_spline_data_f_array[:,0], y_pred_f, 'crimson')
    plt.plot(cov_spline_data_m_array[:,0], y_pred_m, 'b')
    plt.show()

# calculate slope for this brain region
slope_f = (y_pred_f[-3] - y_pred_f[2]) / (cov_spline_data_f_array[-3, 0] - cov_spline_data_f_array[2, 0])

mystop=1


