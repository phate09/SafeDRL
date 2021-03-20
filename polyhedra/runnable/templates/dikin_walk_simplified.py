#!/usr/bin/env python3

import argparse
from concurrent import futures

import ray
import sklearn
import torch
import numpy as np
import pypoman
from scipy.optimize import linprog
from matplotlib import pyplot as plt

from six.moves import range
import plotly.graph_objects as go
from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment


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
    return np.sqrt(r) * np.linalg.cholesky(np.linalg.inv(e)).dot(p)


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


def collect_chain(sampler, count, burn, thin, *args, **kwargs):
    """Use the given sampler to collect points from a chain.

    Args:
        count: Number of points to collect.
        burn: Number of points to skip at beginning of chain.
        thin: Number of points to take from sampler for every point.
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


# def collect_chain2(sampler,count,burn,thint,a,b,x0,dikin_radius):
#     x = x0
#     h_x = hessian(a, b, x)


def main():
    run_sampling_polyhedra()
    template = Experiment.octagon(2)
    boundaries = np.array((1, 2, 1, 1, 2, 1, 1, 1))

    # Polytope parameters
    a = template
    b = boundaries

    chains = sample_polyhedron(a, b)

    fig = go.Figure()
    # trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    trace1 = go.Scatter(x=chains[:, 0], y=chains[:, 1], mode='markers')
    fig.add_trace(trace1)
    fig.update_layout(xaxis_title="delta v", yaxis_title="delta x")
    fig.show()
    print("done")


def get_nn():
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    return sequential_nn


def get_template():
    return Experiment.octagon(2)


def run_sampling_polyhedra():
    ray.init(ignore_reinit_error=True)
    nn = get_nn()
    template = get_template()
    boundaries = np.array((5, -5, 5, 5, 5, 5, 5, 5), dtype=float)
    samples = sample_polyhedron(template, boundaries, 10000)
    samples_ontput = torch.softmax(nn(torch.tensor(samples).float()), 1)

    fig = go.Figure()
    # trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    predicted_label = samples_ontput.detach().numpy()[:, 0]
    trace1 = go.Scatter(x=samples[:, 0], y=samples[:, 1], marker=dict(color=predicted_label, colorscale='bluered'), mode='markers')
    fig.add_trace(trace1)
    fig.update_layout(xaxis_title="delta v", yaxis_title="delta x")
    fig.show()
    points = samples
    chosen_dimension = find_dimension_split(points, predicted_label, template)
    print("done")


def plot_points_and_prediction(points, prediction: np.ndarray):
    fig = go.Figure()
    # trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    trace1 = go.Scatter(x=points[:, 0], y=points[:, 1], marker=dict(color=prediction, colorscale='bluered'), mode='markers')
    fig.add_trace(trace1)
    fig.update_layout(xaxis_title="delta v", yaxis_title="delta x")
    fig.show()


def find_dimension_split2(points, predicted_label, template):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    costs = []

    decision_boundaries = []
    proj2ds = []
    classifier_predictions = []
    for dimension in template:
        proj = project_to_dimension(points, dimension)
        proj2d = np.array([[x, 0] for x in proj])
        proj2ds.append(proj2d)
        # plot_points_and_prediction(proj2d, predicted_label)
        max_value_x = np.max(proj)
        min_value_x = np.min(proj)
        mid_value_x = (max_value_x + min_value_x) / 2
        max_value_y = np.max(predicted_label)
        min_value_y = np.min(predicted_label)
        mid_value_y = (max_value_y + min_value_y) / 2
        true_label = predicted_label >= mid_value_y
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
        index_true = list(test_prediction).index(True)
        index_false = list(test_prediction).index(False)
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


def project_to_dimension(points: np.ndarray, dimension: np.ndarray):
    lengths = np.linalg.norm(dimension)
    projs = np.dot(points, dimension) / lengths
    return projs


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
    chains = collect_chain(dikin_walk, count, burn, thin, a, b, x0, *sampler_args)
    return chains


if __name__ == '__main__':
    main()
