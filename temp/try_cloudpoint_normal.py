import random
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import sympy
from sklearn.neighbors import NearestCentroid
from sympy import Point, Line, Float, Eq
from sympy.plotting import plot, plot3d
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull
from sklearn.linear_model import LogisticRegression


def cluster(x, y, deviation=10, n=100):
    source = [x, y]

    points = []
    for _ in range(n):
        newCoords = [source[i] + random.random() * deviation for i in range(2)]
        newPoint = tuple(newCoords)
        points.append(newPoint)
    return np.array(points)


def points_to_classes(list_of_arrays):
    n_classes = len(list_of_arrays)
    X = list_of_arrays[0].copy()
    y = np.array([0] * len(X))
    for i in range(1, n_classes):
        X = np.append(X, list_of_arrays[i], axis=0)
        y = np.append(y, [i] * len(list_of_arrays[i]))
    return X, y


def array_to_points(points):
    result = []
    for point in points:
        result.append(Point(point))
    return result


def point_to_float(point):
    result = []
    for x in point:
        result.append(float(x.evalf()))
    return result


def find_closest_point(points: List[Point], centroid: Point) -> Point:
    result = points[0]
    min_distance = 9999
    for x in points:
        distance = x.distance(centroid)
        if distance < min_distance:
            result = x
            min_distance = distance
    return result


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if (p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax])
    ax.add_line(l)
    return l


# %% plot 2 clusters
points1 = cluster(1, 1, 3)
points2 = cluster(4, 4, 3)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
plt.show()
# %% scatterplot with centroids
list_of_arrays = [points1, points2]
X, y = points_to_classes(list_of_arrays)

clf = NearestCentroid()
clf.fit(X, y)

print(clf.centroids_)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
plt.show()
# %%
centroid1 = Point(clf.centroids_[0])
centroid2 = Point(clf.centroids_[1])
l = Line(centroid1, centroid2)

closest_opposite_point1 = find_closest_point([Point(x) for x in points1], centroid2)
closest_opposite_point2 = find_closest_point([Point(x) for x in points2], centroid1)

perpendicular1 = l.perpendicular_line(closest_opposite_point1)
perpendicular2 = l.perpendicular_line(closest_opposite_point2)
# perpendicular1_points = np.array((point_to_float(perpendicular1.p1), point_to_float(perpendicular1.p2)))
# perpendicular2_points = np.array((point_to_float(perpendicular2.p1), point_to_float(perpendicular2.p2)))
plt.figure(figsize=(7, 7))

plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], c="black")

hull = ConvexHull(points1)
for simplex in hull.simplices:
    plt.plot(points1[simplex, 0], points1[simplex, 1], 'k-')
vertices = hull.vertices.copy()
vertices = sorted(vertices, key=lambda x: Point(hull.points[x]).distance(closest_opposite_point1))

newline(hull.points[vertices[0]], hull.points[vertices[1]])  # hull line
# newline(point_to_float(perpendicular1.p1), point_to_float(perpendicular1.p2)) #perpendicular line
# newline(point_to_float(perpendicular2.p1), point_to_float(perpendicular2.p2))#perpendicular line
plt.show()
# %% logistic regression
clf1 = LogisticRegression(random_state=0).fit(X, y)
coeff = clf1.coef_
intercept = clf1.intercept_
plt.figure(figsize=(7, 7))

plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], c="black")
a = sympy.symbols('x')
b = sympy.symbols('y')
classif_line = Line(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept.item())
# classif_line.arbitrary_point(0),classif_line.arbitrary_point(1)
newline(point_to_float(classif_line.p1), point_to_float(classif_line.p2))
# plot3d(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept.item(),show=False)
plt.show()