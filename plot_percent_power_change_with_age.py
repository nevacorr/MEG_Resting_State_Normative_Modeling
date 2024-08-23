import os
import sys
import numpy as np
import argparse
import pickle
import glob
from sklearn.model_selection import KFold
from pathlib import Path



model_path = '/home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/data/alpha/ROI_models/cuneus-lh/Models/'

if os.path.exists(os.path.join(model_path, 'meta_data.md')):
    with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
        meta_data = pickle.load(file)
    inscaler = meta_data['inscaler']
    outscaler = meta_data['outscaler']
    mY = meta_data['mean_resp']
    sY = meta_data['std_resp']
    scaler_cov = meta_data['scaler_cov']
    scaler_resp = meta_data['scaler_resp']
    meta_data = True
else:
    print("No meta-data file is found!")
    inscaler = 'None'
    outscaler = 'None'
    meta_data = False

mystop=1