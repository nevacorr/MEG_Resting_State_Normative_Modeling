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

cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.1),
                  (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
          }

cdict2 = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 1.0),
                  (1.0, 0.1, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.0, 0.0))
          }

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

# Creating a modified version of cdict3
# with some transparency
# in the center of the range.
cdict4 = {**cdict3,
          'alpha': ((0.0, 1.0, 1.0),
                #   (0.25, 1.0, 1.0),
                    (0.5, 0.3, 0.3),
                #   (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)),
          }

blue_red1 = LinearSegmentedColormap('BlueRed1',
                                    cdict1)
blue_red2 = LinearSegmentedColormap('BlueRed2',
                                    cdict2)

plt.register_cmap(cmap = blue_red2)

# optional lut kwarg
# plt.register_cmap(name ='BlueRed3', cmap = cdict3)
# plt.register_cmap(name ='BlueRedAlpha', cmap = cdict4)
figure, axes = plt.subplots(2, 2,
                            figsize =(6, 9))

figure.subplots_adjust(left=0.02,
                       bottom=0.06,
                       right=0.95,
                       top=0.94,
                       wspace=0.05)

# Making 4 different subplots:
#------------------------------------------------------
img1 = axes[0, 0].imshow(X,
                         interpolation='nearest',
                         cmap=blue_red1)

figure.colorbar(img1, ax=axes[0, 0])
#----------------------------------------------------
cmap = plt.get_cmap('BlueRed2')
img2 = axes[1, 0].imshow(X,
                         interpolation='nearest',
                         cmap=cmap)

figure.colorbar(img2, ax=axes[1, 0])
#------------------------------------------------------
blue_red3 = LinearSegmentedColormap('BlueRed3', cdict3)
plt.register_cmap(cmap=blue_red3)
cmap = plt.get_cmap("BlueRed3")
# set the third cmap as the default.
# plt.rcParams['image.cmap'] = 'BlueRed3'

img3 = axes[0, 1].imshow(X,
                         interpolation='nearest',
                         cmap=cmap)
figure.colorbar(img3, ax=axes[0, 1])
axes[0, 1].set_title("BlueRed3")
#------------------------------------------------------

# Draw a line with low zorder to
# keep it behind the image.
axes[1, 1].plot([0, 10 * np.pi],
                [0, 20 * np.pi],
                color='c',
                lw=19,
                zorder=-1)

img4 = axes[1, 1].imshow(X,
                         interpolation='nearest')
figure.colorbar(img4, ax=axes[1, 1])

# Here it is: changing the colormap
# for the current image and its
# colorbar after they have been plotted.
# img4.set_cmap('BlueRedAlpha')
# axes[1, 1].set_title("Variation in alpha")

figure.subplots_adjust(top=0.8)

plt.show()






mystop=1