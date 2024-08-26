import os
import sys
import numpy as np
import argparse
import pickle
import matplotlib as plt
import glob
from sklearn.model_selection import KFold
from pathlib import Path

model_path = '/home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/data/alpha/ROI_models/cuneus-lh/Models/'

# Open the file in binary mode and load the data
with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)

mystop=1



# if os.path.exists(os.path.join(model_path, 'meta_data.md')):
#     with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
#         meta_data = pickle.load(file)
#     inscaler = meta_data['inscaler']
#     outscaler = meta_data['outscaler']
#     mY = meta_data['mean_resp']
#     sY = meta_data['std_resp']
#     scaler_cov = meta_data['scaler_cov']
#     scaler_resp = meta_data['scaler_resp']
#     meta_data = True

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file



# import numpy as np
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def spline_basis(x, knots):
#     # Extract knots
#     knot1, knot2 = knots
#
#     # Create the basis functions
#     basis = np.vstack([
#         np.ones_like(x),  # Intercept term (constant 1)
#         x,  # Linear term
#         np.maximum(0, x - knot1),  # Piecewise term for x after knot1
#         np.maximum(0, x - knot2)  # Piecewise term for x after knot2
#     ]).T
#     return basis
#
# # Assuming you have the posterior mean `m` and the basis functions `spline_basis`
# def predict(x, model_params):
#     A = model_params.A  # Symmetric covariance matrix
#     m = model_params.m  # Posterior mean of coefficients
#     knots = [model_params.D / 2, model_params.D]  # Example knots
#
#     # Construct the basis for new x values
#     x_basis = spline_basis(x, knots=knots)
#
#     # Predictive mean
#     y_pred = x_basis @ m
#     return y_pred
#
#
# # Example range of x values to predict
# x_new = np.linspace(0, 10, 100)
#

# Make predictions
y_pred = predict(x_new, model_params)

# Plot the results
plt.plot(x_new, y_pred, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Linear Regression with Spline Model (Order 1, 2 Knots)')
plt.legend()
plt.show()
