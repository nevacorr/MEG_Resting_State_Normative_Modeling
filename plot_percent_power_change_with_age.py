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

model_path = ('/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                             'Modeling/data/alpha/ROI_models/cuneus-lh/Models/')

cov_spline_data = pd.read_csv('/home/toddr/neva/PycharmProjects/MEG Resting State Normative '
                             'Modeling/data/alpha/ROI_models/cuneus-lh/cov_bspline_tr.txt', sep=' ', header=None)

cov_spline_data_f = cov_spline_data[cov_spline_data[1] == 0].sort_values(by=0)
cov_spline_data_m = cov_spline_data[cov_spline_data[1] == 1].sort_values(by=0)

numrows = cov_spline_data_f.shape[0]
cov_spline_data_f = cov_spline_data_f.iloc[0:numrows - 1, :]

cov_spline_data_f_array = cov_spline_data_f.to_numpy()
cov_spline_data_m_array = cov_spline_data_m.to_numpy()

# Open the file in binary mode and load the data
with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)

y_pred_f = np.dot(cov_spline_data_f_array, data.blr.m)
y_pred_m = np.dot(cov_spline_data_m_array, data.blr.m)

# plt.plot(cov_spline_data_f_array[:,0], y_pred_f, 'crimson')
# plt.plot(cov_spline_data_m_array[:,0], y_pred_m, 'b')
# plt.show()

slope_f = (y_pred_f[-2] - y_pred_f[1]) / (cov_spline_data_f_array[0, -2] - cov_spline_data_f_array[0, 1])

mystop=1


