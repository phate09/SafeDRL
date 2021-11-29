import sympy
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sympy import Line

from polyhedra.utils import *

# %% plot 2 clusters
points1 = cluster(1, 1, 3)
points2 = cluster(4, 4, 3)
# points3 = cluster(2, -3, 3)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
# plt.plot(points3[:, 0], points3[:, 1], 'go')
plt.show()
# %% scatterplot with centroids
list_of_arrays = [points1, points2]  # points3
X, y = points_to_classes(list_of_arrays)

clf = NearestCentroid()
clf.fit(X, y)

print(clf.centroids_)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
# plt.plot(points3[:, 0], points3[:, 1], 'go')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
plt.show()
# %%
centroid1 = Point(clf.centroids_[0])
centroid2 = Point(clf.centroids_[1])
# centroid3 = Point(clf.centroids_[2])
l12 = Line(centroid1, centroid2)
# l23 = Line(centroid2, centroid3)
# l31 = Line(centroid1, centroid3)

closest_opposite_point12 = find_closest_point([Point(x) for x in points1], centroid2)
# closest_opposite_point13 = find_closest_point([Point(x) for x in points1], centroid3)
closest_opposite_point21 = find_closest_point([Point(x) for x in points2], centroid1)
# closest_opposite_point23 = find_closest_point([Point(x) for x in points2], centroid3)
# closest_opposite_point31 = find_closest_point([Point(x) for x in points3], centroid1)
# closest_opposite_point32 = find_closest_point([Point(x) for x in points3], centroid2)

perpendicular12 = l12.perpendicular_line(closest_opposite_point12)
# perpendicular13 = l31.perpendicular_line(closest_opposite_point13)
perpendicular21 = l12.perpendicular_line(closest_opposite_point21)
# perpendicular23 = l23.perpendicular_line(closest_opposite_point23)
# perpendicular31 = l31.perpendicular_line(closest_opposite_point31)
# perpendicular32 = l23.perpendicular_line(closest_opposite_point32)
# %%
plt.figure(figsize=(7, 7))
plt.xlim(-2, 10)
plt.ylim(-4, 8)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
# plt.plot(points3[:, 0], points3[:, 1], 'go')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], c="black")
plt.plot(clf.centroids_[[0, -1], :][:, 0], clf.centroids_[[0, -1], :][:, 1], c="black")

# hull = ConvexHull(points1)
# for simplex in hull.simplices:
#     plt.plot(points1[simplex, 0], points1[simplex, 1], 'k-')
# vertices = hull.vertices.copy()
# vertices = sorted(vertices, key=lambda x: Point(hull.points[x]).distance(closest_opposite_point1))

# newline(hull.points[vertices[0]], hull.points[vertices[1]])  # hull line
newline(point_to_float(perpendicular12.p1), point_to_float(perpendicular12.p2))  # perpendicular line
# newline(point_to_float(perpendicular13.p1), point_to_float(perpendicular13.p2)) #perpendicular line
newline(point_to_float(perpendicular21.p1), point_to_float(perpendicular21.p2))  # perpendicular line
# newline(point_to_float(perpendicular23.p1), point_to_float(perpendicular23.p2))#perpendicular line
# newline(point_to_float(perpendicular31.p1), point_to_float(perpendicular31.p2))#perpendicular line
# newline(point_to_float(perpendicular32.p1), point_to_float(perpendicular32.p2))#perpendicular line
# perpendicular1_points = np.array((point_to_float(perpendicular12.p1), point_to_float(perpendicular12.p2)))
# centroid_line = np.array((point_to_float(centroid1), point_to_float(centroid2)))
# plt.plot(perpendicular1_points[:,0],perpendicular1_points[:,1])
# plt.plot(centroid_line[:,0],centroid_line[:,1])
plt.show()
# %% logistic regression
clf1 = LogisticRegression(random_state=0, solver="lbfgs", C=0.01, penalty='l2', max_iter=10000, multi_class='ovr').fit(X, y)
coeff = clf1.coef_
intercept = clf1.intercept_
plt.figure(figsize=(7, 7))
plt.xlim(-2, 10)
plt.ylim(-4, 8)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
# plt.plot(points3[:, 0], points3[:, 1], 'go')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
# plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], c="black")
a = sympy.symbols('x')
b = sympy.symbols('y')
classif_line1 = Line(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept[0].item())
# classif_line2 = Line(coeff[1][0].item() * a + coeff[1][1].item() * b + intercept[1].item())
# classif_line3 = Line(coeff[2][0].item() * a + coeff[2][1].item() * b + intercept[2].item())
# classif_line.arbitrary_point(0),classif_line.arbitrary_point(1)
newline(point_to_float(classif_line1.p1), point_to_float(classif_line1.p2))
# newline(point_to_float(classif_line2.p1), point_to_float(classif_line2.p2))
# newline(point_to_float(classif_line3.p1), point_to_float(classif_line3.p2))
# plot3d(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept.item(),show=False)
plt.show()
# %% support vector machine
# clf2 = svm.SVC(kernel="linear",decision_function_shape='ovo',max_iter=10000)
clf2 = svm.LinearSVC(multi_class='ovr', max_iter=10000)
clf2.fit(X, y)
coeff = clf2.coef_
intercept = clf2.intercept_
plt.figure(figsize=(7, 7))
plt.xlim(-2, 10)
plt.ylim(-4, 8)
plt.plot(points1[:, 0], points1[:, 1], 'o')
plt.plot(points2[:, 0], points2[:, 1], 'ro')
# plt.plot(points3[:, 0], points3[:, 1], 'go')
plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], 'o', c="black", markersize=20)
# plt.plot(clf.centroids_[:, 0], clf.centroids_[:, 1], c="black")
a = sympy.symbols('x')
b = sympy.symbols('y')
classif_line1 = Line(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept[0].item())
classif_line2 = Line(coeff[1][0].item() * a + coeff[1][1].item() * b + intercept[1].item())
# classif_line3 = Line(coeff[2][0].item() * a + coeff[2][1].item() * b + intercept[2].item())
# classif_line.arbitrary_point(0),classif_line.arbitrary_point(1)
newline(point_to_float(classif_line1.p1), point_to_float(classif_line1.p2))
newline(point_to_float(classif_line2.p1), point_to_float(classif_line2.p2))
# newline(point_to_float(classif_line3.p1), point_to_float(classif_line3.p2))
# plot3d(coeff[0][0].item() * a + coeff[0][1].item() * b + intercept.item(),show=False)
plt.show()
