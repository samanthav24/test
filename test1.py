import numpy as np
import numpy.linalg as linalg
from progress.bar import Bar
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from copy import deepcopy
from init import *
import time
from math import sqrt
from numpy import sqrt as sqrt2


# pos = [[1,1], [2,1], [1,3],[2,3], [3,1], [3,2], [2,-1], [4,1]]
# vor = Voronoi(pos)


# # regions = np.array(vor.regions, dtype=object)

# vol = np.zeros(vor.npoints)
# for i, reg_num in enumerate(vor.point_region):
#     indices = vor.regions[reg_num]
#     if -1 in indices: # some regions can be opened
#         vol[i] = np.inf
#     else:
#         vol[i] = ConvexHull(vor.vertices[indices]).volume
# print(vol)

# fig = voronoi_plot_2d(vor)
# plt.show()

# start = time.time()
# for i in range(25000000):
#     n = 5**0.5
# print(time.time() - start)

# start = time.time()
# for i in range(25000000):
#     n = sqrt(5)
# print(time.time() - start)

# start = time.time()
# for i in range(25000000):
#     n = np.sqrt(5)
# print(time.time() - start)

# features = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6,6], [7, 7], [8, 8], [9, 9], [10, 10]]
# # split the features in 10 parts - for each file
# features = np.array_split(features, 10)
# features = np.array_split(features, 2)

# # split into young and old and take the mean
# young_features = np.mean(features[0], axis=0)
# old_features = np.mean(features[1], axis=0)

# print(young_features, old_features)

import math 

v = np.linspace(-10, 10, 50)

label1 = 'g(v) = ' + r"$\frac{1}{1 + e^{-v}}$"
plt.plot(v, 1/(1 + np.exp(-v)), 'r', label=label1)
y = [0.5 for i in range(len(v))]
plt.plot(v, y, 'b--', label='g(v) = 0.5')
plt.xlabel('v')
plt.ylabel('g(v)')
plt.title('Sigmoid function')
plt.legend()
plt.show()