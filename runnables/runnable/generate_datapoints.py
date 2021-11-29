import numpy as np

import mosaic.utils as utils

points = np.random.random((10000, 2)) * 10 - 5
positives =[x for x in points if 1 > x[0] > -1 and 1 > x[1] > -1]
negatives = [x for x in points if not (1 > x[0] > -1 and 1 > x[1] > -1)]
fig = utils.scatter_plot(positives,negatives)
fig.show()