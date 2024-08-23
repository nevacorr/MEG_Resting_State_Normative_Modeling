import os
import sys
import numpy as np
import argparse
import pickle
import glob

from sklearn.model_selection import KFold
from pathlib import Path

try:  # run as a package if installed
    from pcntoolkit import configs
    from pcntoolkit.dataio import fileio
    from pcntoolkit.normative_model.norm_utils import norm_init
    from pcntoolkit.util.utils import compute_pearsonr, CustomCV, explained_var
    from pcntoolkit.util.utils import compute_MSLL, scaler, get_package_versions


model_path = kwargs.pop('model_path', 'Models')
job_id = kwargs.pop('job_id', None)
batch_size = kwargs.pop('batch_size', None)
outputsuffix = kwargs.pop('outputsuffix', 'predict')
outputsuffix = "_" + outputsuffix.replace("_", "")
inputsuffix = kwargs.pop('inputsuffix', 'estimate')
inputsuffix = "_" + inputsuffix.replace("_", "")
alg = kwargs.pop('alg')
fold = kwargs.pop('fold', 0)
models = kwargs.pop('models', None)
return_y = kwargs.pop('return_y', False)

if alg == 'gpr':
    raise ValueError("gpr is not supported with predict()")

if respfile is not None and not os.path.exists(respfile):
    print("Response file does not exist. Only returning predictions")
    respfile = None
if not os.path.isdir(model_path):
    print('Models directory does not exist!')
    return
else:
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

if batch_size is not None:
    batch_size = int(batch_size)
    job_id = int(job_id) - 1

# load data
print("Loading data ...")
X = fileio.load(covfile)
if len(X.shape) == 1:
    X = X[:, np.newaxis]

sample_num = X.shape[0]
if models is not None:
    feature_num = len(models)
else:
    feature_num = len(glob.glob(os.path.join(model_path, 'NM_' + str(fold) +
                                             '_*' + inputsuffix + '.pkl')))
    models = range(feature_num)

Yhat = np.zeros([sample_num, feature_num])
S2 = np.zeros([sample_num, feature_num])
Z = np.zeros([sample_num, feature_num])

if inscaler in ['standardize', 'minmax', 'robminmax']:
    Xz = scaler_cov[fold].transform(X)
else:
    Xz = X

# estimate the models for all variabels
# TODO Z-scores adaptation for SHASH HBR
for i, m in enumerate(models):
    print("Prediction by model ", i + 1, "of", feature_num)
    nm = norm_init(Xz)
    nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' +
                              str(m) + inputsuffix + '.pkl'))
    if (alg != 'hbr' or nm.configs['transferred'] == False):
        yhat, s2 = nm.predict(Xz, **kwargs)
    else:
        tsbefile = kwargs.get('tsbefile')
        batch_effects_test = fileio.load(tsbefile)
        yhat, s2 = nm.predict_on_new_sites(Xz, batch_effects_test)

    if outscaler == 'standardize':
        Yhat[:, i] = scaler_resp[fold].inverse_transform(yhat, index=i)
        S2[:, i] = s2.squeeze() * sY[fold][i] ** 2
    elif outscaler in ['minmax', 'robminmax']:
        Yhat[:, i] = scaler_resp[fold].inverse_transform(yhat, index=i)
        S2[:, i] = s2 * (scaler_resp[fold].max[i] -
                         scaler_resp[fold].min[i]) ** 2
    else:
        Yhat[:, i] = yhat.squeeze()
        S2[:, i] = s2.squeeze()

if respfile is None:
    save_results(None, Yhat, S2, None, outputsuffix=outputsuffix)

    return (Yhat, S2)
