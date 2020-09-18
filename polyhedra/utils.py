import random
from typing import List
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from sympy import Point


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