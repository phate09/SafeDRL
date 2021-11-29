import numpy as np
from scipy.spatial import ConvexHull

points = np.random.rand(30, 2)  # 30 random points in 2-D
# points = np.array(((0,1),(1,1),(1,0)))

hull = ConvexHull(points)

import matplotlib.pyplot as plt

plt.plot(points[:, 0], points[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
plt.show()
