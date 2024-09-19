#######
# Learn how to create custom color maps
# This follows the code at https://www.geeksforgeeks.org/matplotlib-colors-linearsegmentedcolormap-class-in-python/
#######

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# some dummy data
a = np.arange(0, np.pi, 0.1)
b = np.arange(0, 2 * np.pi, 0.1)
A, B = np.meshgrid(a, b)
X = np.cos(A) * np.sin(B) * 10

# custom segmented color dictionary

cdict3 = {'red': ((0.0, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.8, 1.0),
                  (0.75, 1.0, 1.0),
                  (1.0, 0.4, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.9, 0.9),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
          }


# optional lut kwarg
# plt.register_cmap(name ='BlueRed3', cmap = cdict3)
# plt.register_cmap(name ='BlueRedAlpha', cmap = cdict4)
figure, axes = plt.subplots(figsize =(6, 9))

# Making plot
#----------------------------------------------------
blue_red3 = LinearSegmentedColormap('BlueRed3', cdict3)
plt.register_cmap(cmap=blue_red3)
cmap = plt.get_cmap("BlueRed3")
# set the third cmap as the default.
# plt.rcParams['image.cmap'] = 'BlueRed3'

img3 = plt.imshow(X, interpolation='nearest', cmap=cmap)
figure.colorbar(img3)
plt.title("BlueRed3")
plt.show()
#------------------------------------------------------







mystop=1