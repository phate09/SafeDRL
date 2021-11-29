#!/usr/bin/env python3

import plotly.graph_objects as go
import pypoman
import sklearn
from ray import remote
from scipy.optimize import linprog
from six.moves import range
from sklearn.linear_model import LogisticRegression

from polyhedra.partitioning import project_to_dimension


def hessian(a, b, x):
    """Return log-barrier Hessian matrix at x."""
    d = (b - a.dot(x))
    s = d ** -2.0
    # s[s == np.inf] = 0
    return a.T.dot(np.diag(s)).dot(a)


def local_norm(h, v):
    """Return the local norm of v based on the given Hessian matrix."""
    return v.T.dot(h).dot(v)


def sample_ellipsoid(e, r):
    """Return a point in the (hyper)ellipsoid uniformly sampled.

    The ellipsoid is defined by the positive definite matrix, ``e``, and
    the radius, ``r``.
    """
    # Generate a point on the sphere surface
    p = np.random.normal(size=e.shape[0])
    p /= np.linalg.norm(p)

    # Scale to a point in the sphere volume
    p *= np.random.uniform() ** (1.0 / e.shape[0])

    # Transform to a point in the ellipsoid
    return np.sqrt(r) * np.linalg.cholesky(np.linalg.pinv(e)).dot(p)


def ellipsoid_axes(e):
    """Return matrix with columns that are the axes of the ellipsoid."""
    w, v = np.linalg.eigh(e)
    return v.dot(np.diag(w ** (-1 / 2.0)))


def dikin_walk(a, b, x0, r=3 / 40):
    """Generate points with Dikin walk."""
    x = x0
    h_x = hessian(a, b, x)

    while True:
        if not (a.dot(x) <= b).all():
            print(a.dot(x) - b)
            raise Exception('Invalid state: {}'.format(x))

        if np.random.uniform() < 0.5:
            yield x
            continue

        z = x + sample_ellipsoid(h_x, r)
        h_z = hessian(a, b, z)

        if local_norm(h_z, x - z) > 1.0:
            yield x
            continue

        p = np.sqrt(np.linalg.det(h_z) / np.linalg.det(h_x))
        if p >= 1 or np.random.uniform() < p:
            x = z
            h_x = h_z

        yield x


def dikin_walk_simplified(a, b, x0, r=3 / 40, n_samples=10000):
    """Generate points with Dikin walk. Using a For loop"""
    x = x0
    h_x = hessian(a, b, x)
    results = []
    for i in range(n_samples):
        if not (a.dot(x) <= b).all():
            print(a.dot(x) - b)
            raise Exception('Invalid state: {}'.format(x))

        if np.random.uniform() < 0.5:
            results.append(x)
            continue

        z = x + sample_ellipsoid(h_x, r)
        h_z = hessian(a, b, z)

        if local_norm(h_z, x - z) > 1.0:
            results.append(x)
            continue

        p = np.sqrt(np.linalg.det(h_z) / np.linalg.det(h_x))
        if p >= 1 or np.random.uniform() < p:
            x = z
            h_x = h_z

        results.append(x)
    return np.array(results)


def hit_and_run(a, b, x0):
    """Generate points with Hit-and-run algorithm."""
    x = x0

    while True:
        if not (a.dot(x) <= b).all():
            print(a.dot(x) - b)
            raise Exception('Invalid state: {}'.format(x))

        # Generate a point on the sphere surface
        d = np.random.normal(size=a.shape[1])
        d /= np.linalg.norm(d)

        # Find closest boundary in the direction
        dist = np.divide(b - a.dot(x), a.dot(d))
        closest = dist[dist > 0].min()

        x += d * closest * np.random.uniform()

        yield x


def chebyshev_center(a, b):
    """Return Chebyshev center of the convex polytope."""
    norm_vector = np.reshape(np.linalg.norm(a, axis=1), (a.shape[0], 1))
    c = np.zeros(a.shape[1] + 1)
    c[-1] = -1
    a_lp = np.hstack((a, norm_vector))
    res = linprog(c, A_ub=a_lp, b_ub=b, bounds=(None, None))
    if not res.success:
        raise Exception('Unable to find Chebyshev center')

    return res.x[:-1]

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from numpy.linalg import det
from scipy.stats import dirichlet


def dist_in_hull(a,b, n):
    points = np.vstack(pypoman.duality.compute_polytope_vertices(a,b))
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size = n, )#p = vols / vols.sum()

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))


def collect_chain(sampler, count, burn, thin, *args, **kwargs):
    """Use the given sampler to collect points from a chain.

    Args:
        count: Number of points to collect.
        burn: Number of points to skip at beginning of chain.
        thin: Number of points to take from sampler for every point. (points to skip?)
    """
    chain = sampler(*args, **kwargs)
    point = next(chain)
    points = np.empty((count, point.shape[0]))

    for i in range(burn - 1):
        next(chain)

    for i in range(count):
        points[i] = next(chain)
        for _ in range(thin - 1):
            next(chain)

    return points


def collect_chain_dikin_walk_simplified(count, burn, thin, *args, **kwargs):
    points = dikin_walk_simplified(*args, **kwargs, n_samples=(count * thin) + burn)
    points = points[burn:]
    results = points[::thin]

    return results


# def collect_chain2(sampler,count,burn,thint,a,b,x0,dikin_radius):
#     x = x0
#     h_x = hessian(a, b, x)


def plot_points_and_prediction(points, prediction: np.ndarray):
    fig = go.Figure()
    # trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    trace1 = go.Scatter(x=points[:, 0], y=points[:, 1], marker=dict(color=prediction, colorscale='bluered', cmax=1, cmin=0), mode='markers')
    fig.add_trace(trace1)
    fig.update_layout(xaxis_title="delta v", yaxis_title="delta x")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def find_dimension_split2(points, predicted_label, template):
    costs = []

    decision_boundaries = []
    proj2ds = []
    classifier_predictions = []
    for dimension in template:
        proj = project_to_dimension(points, dimension)
        proj2d = np.array([[x, predicted_label[i]] for i, x in enumerate(proj)])
        proj2ds.append(proj2d)
        # plot_points_and_prediction(proj2d, predicted_label)
        max_value_x = np.max(proj)
        min_value_x = np.min(proj)
        mid_value_x = (max_value_x + min_value_x) / 2
        max_value_y = np.max(predicted_label)
        min_value_y = np.min(predicted_label)
        mid_value_y = (max_value_y + min_value_y) / 2
        true_label = predicted_label >= 0.5  # mid_value_y
        clf: LogisticRegression = LogisticRegression(random_state=0).fit(proj.reshape(-1, 1), true_label)
        cost = clf.score(proj.reshape(-1, 1), true_label)
        classifier_prediction = clf.predict(proj.reshape(-1, 1))
        classifier_predictions.append(classifier_prediction)
        # plot_points_and_prediction(proj2d, classifier_prediction.astype(int))
        # plot_points_and_prediction(points, classifier_prediction.astype(int))
        # try to find the decision boundary
        num_tests = 300
        X_test = np.linspace(min_value_x, max_value_x, num_tests)
        test_prediction = clf.predict(X_test.reshape(-1, 1))
        decision_boundary = min_value_x
        index_true = list(test_prediction).index(True) if np.any(test_prediction) else 0
        index_false = list(test_prediction).index(False) if not np.all(test_prediction) else 0
        if abs(num_tests / 2 - index_true) < abs(num_tests / 2 - index_false):
            decision_boundary = X_test[index_true]
        else:
            decision_boundary = X_test[index_false]
        costs.append(cost)
        decision_boundaries.append(decision_boundary)
    chosen_dimension = np.argmax(costs)
    # plot_points_and_prediction(points, predicted_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], predicted_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], classifier_predictions[chosen_dimension].astype(int))
    # plot_points_and_prediction(points, classifier_predictions[chosen_dimension].astype(int))

    return chosen_dimension, decision_boundaries[chosen_dimension]







@remote
def remote_find_minimum2(dimension, points, predicted_label):
    proj = project_to_dimension(points, dimension)
    proj2d = np.array([[x, predicted_label[i]] for i, x in enumerate(proj)])
    # plot_points_and_prediction(proj2d, predicted_label)
    max_value_x = np.max(proj)
    min_value_x = np.min(proj)
    mid_value_x = (max_value_x + min_value_x) / 2
    max_value_y = np.max(predicted_label)
    min_value_y = np.min(predicted_label)
    mid_value_y = (max_value_y + min_value_y) / 2
    true_label = predicted_label >= mid_value_y
    # enumerate all points, find ranges of probabilities for each decision boundary
    range_1 = (max_value_y, min_value_y)
    range_2 = (max_value_y, min_value_y)
    decision_boundary = -1
    min_cost = 9999
    sorted_pred = predicted_label[np.argsort(proj)]
    sorted_proj = np.sort(proj)
    assert len(points) > 3
    normalisation_quotient = max_value_y - min_value_y
    costs = []
    # true_label = predicted_label >= min_value_y + (max_value_y - min_value_y) * 0.7
    clf: LogisticRegression = LogisticRegression(random_state=0).fit(proj.astype(float).reshape(-1, 1), true_label)
    cost = clf.score(proj.reshape(-1, 1), true_label)
    num_tests = 300
    X_test = np.linspace(min_value_x, max_value_x, num_tests)
    test_prediction = clf.predict(X_test.reshape(-1, 1))
    decision_value = min_value_x
    index_true = list(test_prediction).index(True) if np.any(test_prediction) else 0
    index_false = list(test_prediction).index(False) if not np.all(test_prediction) else 0
    if abs(num_tests / 2 - index_true) < abs(num_tests / 2 - index_false):
        decision_value = X_test[index_true]
    else:
        decision_value = X_test[index_false]
    decision_index = np.argmin(np.abs(proj - decision_value))
    costs.append(cost)
    min_cost = cost
    # decision_boundaries.append(decision_boundary)
    return decision_index, min_cost, sorted_proj, proj2d, costs

    # for i in range(1, len(sorted_pred)):
    #     range_1_temp = (min(min(sorted_pred[0:i]), range_1[0]), max(max(sorted_pred[0:i]), range_1[1]))
    #     range_2_temp = (min(min(sorted_pred[i:]), range_2[0]), max(max(sorted_pred[i:]), range_2[1]))
    #     range_1_inv = (1 - range_1_temp[1], 1 - range_1_temp[0])
    #     tolerance = (max_value_y - min_value_y) * 0.0
    #     # cost = (max(0, abs(range_1_temp[0] - range_1_temp[1]) - tolerance) / normalisation_quotient) + (
    #     #         max(0, abs(range_2_temp[0] - range_2_temp[1]) - tolerance) / normalisation_quotient)  # todo check
    #     # cost = cost ** 2
    #     # true_label = np.array(range(len(sorted_pred))) > i
    #
    #     if np.all(true_label) or not np.any(true_label):
    #         continue
    #     cost = sklearn.metrics.log_loss(true_label, sorted_pred.astype(float))  # .astype(int)
    #     # eps = 1e-15
    #     # y_pred = np.clip(predicted_label, eps, 1 - eps)  # todo weighted cross entropy loss
    #     # -(transformed_labels * np.log(y_pred)).sum(axis=1)
    #     costs.append(cost)
    #     if cost <= min_cost:
    #         range_1 = range_1_temp
    #         range_2 = range_2_temp
    #         min_cost = cost
    #         decision_boundary = i
    # # plot_list(costs)
    # return decision_boundary, min_cost, sorted_proj, proj2d, costs


def plot_list(costs):
    import plotly.express as px
    fig = px.line(x=range(len(costs)), y=costs)
    fig.show()


def find_dimension_split(points, predicted_label, template):
    costs = []
    true_label_assignment = []
    # plot_points_and_prediction(points, predicted_label)
    for dimension in template:
        proj = project_to_dimension(points, dimension)
        max_value_x = np.max(proj)
        min_value_x = np.min(proj)
        mid_value_x = (max_value_x + min_value_x) / 2
        true_label = proj <= mid_value_x
        true_label_assignment.append(true_label)
        eps = 1e-7
        # y_pred = np.expand_dims(np.clip(predicted_label,eps,1-eps),1)
        # y_pred = np.append(1 - y_pred, y_pred, axis=1)
        # transformed_labels = np.column_stack((np.invert(true_label).astype(int), true_label.astype(int)))
        # loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)
        cost = sklearn.metrics.log_loss(true_label.astype(int), predicted_label.astype(float))
        costs.append(cost)
    chosen_dimension = np.argmin(costs)
    # plot_points_and_prediction(points, true_label_assignment[chosen_dimension].astype(int))

    return chosen_dimension




def sample_polyhedron(a: np.ndarray, b: np.ndarray, count=10000):
    # Initial point to start the chains from.
    # Use the Chebyshev center.
    # x0 = chebyshev_center(a, b)
    x0 = pypoman.polyhedron.compute_chebyshev_center(a, b)
    # print('Chebyshev center: {}'.format(x0))
    # print('A= {}'.format(a))
    # print('b= {}'.format(b))
    # print('x0= {}'.format(x0))
    chain_count = 1
    burn = 1000
    thin = 10
    dikin_radius = 1
    sampler_args = (dikin_radius,)
    # sampler_args = ()
    # chains = collect_chain(dikin_walk, count, burn, thin, a, b, x0, *sampler_args)
    chains = collect_chain_dikin_walk_simplified(count, burn, thin, a, b, x0, *sampler_args)
    return chains


