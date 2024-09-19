#######
# Create custom color map
# This follows the code at https://www.geeksforgeeks.org/matplotlib-colors-linearsegmentedcolormap-class-in-python/
#######

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap():

    # custom segmented color dictionary

    cdict3 = {'red': ((0.0, 1.0, 1.0), #bright yellow
                      (0.25, 0.8, 0.8), #lighter yellow
                      (0.49, 0.7, 0.7), #dark yellow
                      (0.5, 0.5, 0.5),  #gray
                      (0.51, 0.0, 0.0), #dark green
                      (0.75, 0.0, 0.0), #green
                      (1.0, 0.0, 0.0)), #bright green

              'green': ((0.0, 0.75, 1.0),
                        (0.25, 0.8, 0.8),
                        (0.49, 0.7, 0.7),
                        (0.5, 0.5, 0.5),
                        (0.51, 0.7, 0.7),
                        (0.75, 0.8, 0.8),
                        (1.0, 1.0, 1.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.49, 0.0, 0.0),
                       (0.5, 0.5, 0.5),
                       (0.51, 0.0, 0.0),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }


    custom_colormap = LinearSegmentedColormap('CustomYellowGrayGreen', cdict3)
    plt.register_cmap(cmap=custom_colormap)

    return custom_colormap

