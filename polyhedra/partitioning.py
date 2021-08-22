import torch
import numpy as np
import ray
import polyhedra.runnable.templates.polytope as polytope
from mosaic.utils import PolygonSort, compute_trace_polygons
from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.plot_utils import windowed_projection, show_polygons, project_to_dimension
from polyhedra.runnable.templates.dikin_walk_simplified import plot_points_and_prediction, plot_list
from symbolic import unroll_methods
import gurobi as grb
import scipy
import sklearn
from scipy.optimize import linprog, minimize, minimize_scalar
import plotly.graph_objects as go


def sample_and_split(pre_nn, nn, template, boundaries, env_input_size, template_2d):
    # print("Performing split...", "")
    repeat = True
    samples = None
    while repeat:
        repeat = False
        try:
            samples = polytope.sample(10000, template, boundaries)
            # samples = sample_polyhedron(template, boundaries, 5000)
        except Exception as e:
            print("Warning: error during the sampling")
            repeat = True
    preprocessed = pre_nn(torch.tensor(samples).float())
    samples_ontput = torch.softmax(nn(preprocessed), 1)
    predicted_label = samples_ontput.detach().numpy()[:, 0]
    # template_2d: np.ndarray = np.array([Experiment.e(env_input_size, 2), Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)])
    dimension_lengths = []
    for i, dimension in enumerate(template):
        inverted_dimension = find_inverted_dimension(-dimension, template)
        dimension_length = boundaries[i] + boundaries[inverted_dimension]
        dimension_lengths.append(dimension_length)
    chosen_dimension, decision_point = find_dimension_split3(samples, predicted_label, template, template_2d, dimension_lengths)
    # split3, split4 = split_polyhedron(template, boundaries, chosen_dimension, decision_point)
    split1, split2 = split_polyhedron_milp(template, boundaries, chosen_dimension, decision_point)
    # print("done")
    # plot_points_and_prediction(samples@template_2d.T, predicted_label)
    # show_polygons(template, [split1, split2], template_2d)
    # show_polygons(template, [split3, split4], template_2d)
    return split1, split2


# find corresponding dimension
def pick_longest_dimension(template, boundaries):
    max_dimension = 0
    max_dimension_index = -1
    for dimension, value in enumerate(template):
        inverted_value = -template[dimension]
        inverted_dimension = find_inverted_dimension(inverted_value, template)
        dimension_length = boundaries[dimension] + boundaries[inverted_dimension]
        if dimension_length > max_dimension:
            max_dimension = max(max_dimension, dimension_length)
            max_dimension_index = dimension
    return max_dimension_index


def split_polyhedron(template, boundaries, dimension, decision_point=None):
    # bugged for diagonals, use split_polyhedron_milp
    inverted_value = -template[dimension]
    inverted_dimension = find_inverted_dimension(inverted_value, template)
    # find midpoint
    if decision_point is None:
        midpoint = (boundaries[dimension] - boundaries[inverted_dimension]) / 2
    else:
        midpoint = decision_point
    # create clones
    boundaries1 = list(boundaries)
    boundaries2 = list(boundaries)
    boundaries1[inverted_dimension] = -midpoint
    boundaries2[dimension] = midpoint
    boundaries1 = tuple(boundaries1)
    boundaries2 = tuple(boundaries2)
    return boundaries1, boundaries2


def split_polyhedron_milp(template, boundaries, chosen_dimension, decision_point):
    '''splits the polyhedron in 2 based on the dimension and the decision point using the milp model'''
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
    split_template = template[chosen_dimension]
    input = Experiment.generate_input_region(gurobi_model, template, boundaries, len(split_template))
    gurobi_model.update()
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    gurobi_model.addConstr(sum((split_template[i] * input[i]) for i in range(len(split_template))) >= decision_point)
    split1 = Experiment.optimise(template, gurobi_model, input)

    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
    split_template = template[chosen_dimension]
    input = Experiment.generate_input_region(gurobi_model, template, boundaries, len(split_template))
    gurobi_model.update()
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    gurobi_model.addConstr(sum((split_template[i] * input[i]) for i in range(len(split_template))) <= decision_point)
    split2 = Experiment.optimise(template, gurobi_model, input)
    return split1, split2


def find_inverted_dimension(inverted_value, template):
    inverted_dimension = -1
    for i, val in enumerate(template):
        if np.array_equal(inverted_value, val):
            inverted_dimension = i
            break
    assert inverted_dimension != -1, "Could not find inverted dimension"
    return inverted_dimension


def is_split_range(ranges_probs, max_prob_difference=0.2):
    split_flag = False
    for chosen_action in range(2):
        prob_diff = ranges_probs[chosen_action][1] - ranges_probs[chosen_action][0]
        if prob_diff > max_prob_difference:
            # should split the input
            split_flag = True
            break
    return split_flag


# noinspection PyUnreachableCode
def find_dimension_split3(points, predicted_label, template, template2d, dimension_lengths):
    costs = []
    minimum_length = 0.1
    decision_boundaries = []
    proj2ds = []
    classifier_predictions = []
    indices = []
    proc_ids = []
    for dimension in template:
        proc_id = remote_find_minimum.remote(dimension, points, predicted_label)
        proc_ids.append(proc_id)
    values = ray.get(proc_ids)
    for i, value in enumerate(values):
        decision_boundary, min_cost, sorted_proj, proj2d, plot_costs, ignore = value
        # plot_list(plot_costs)
        dimension_lengths_i_ = dimension_lengths[i]
        if dimension_lengths_i_ > minimum_length and not ignore:  # tries to ignore extremely small dimensionss
            costs.append(min_cost)
            decision_boundaries.append(sorted_proj[decision_boundary] if sorted_proj is not None else None)
            indices.append(decision_boundary)
            proj2ds.append(proj2d)
        else:
            costs.append(float("inf"))
            decision_boundaries.append(None)
            indices.append(None)
            proj2ds.append(None)
    # chosen_dimension = np.argmin([((i - len(points) / 2) / len(points)) ** 2 for i in indices])
    chosen_dimension = np.argmin(costs)
    max_value_y = np.max(predicted_label)
    min_value_y = np.min(predicted_label)
    delta_y = max_value_y - min_value_y
    normalised_label = (predicted_label - min_value_y) / delta_y
    # plot_points_and_prediction(points@template2d.T, normalised_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], normalised_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], [0 if x[0]>decision_boundaries[chosen_dimension] else 1 for x in proj2ds[chosen_dimension]])
    # plot_points_and_prediction(points@template2d.T, [0 if x[0] > decision_boundaries[chosen_dimension] else 1 for x in proj2ds[chosen_dimension]])
    return chosen_dimension, decision_boundaries[chosen_dimension]


@ray.remote
def remote_find_minimum(dimension, points, predicted_label):
    proj = project_to_dimension(points, dimension)
    proj2d = np.array([[x, predicted_label[i]] for i, x in enumerate(proj)])
    # plot_points_and_prediction(proj2d, predicted_label)
    max_value_x = np.max(proj)
    min_value_x = np.min(proj)
    delta_x = max_value_x - min_value_x
    min_absolute_delta_x = 0.0001
    if delta_x < min_absolute_delta_x:
        ignore = True
        return 0, float('inf'), None, None, None, ignore  # aim to give a big negative reward to skip
    mid_value_x = (max_value_x + min_value_x) / 2
    max_value_y = np.max(predicted_label)
    min_value_y = np.min(predicted_label)
    mid_value_y = (max_value_y + min_value_y) / 2
    delta_y = max_value_y - min_value_y
    true_label = predicted_label >= mid_value_y
    normalised_label = (predicted_label - min_value_y) / delta_y
    # enumerate all points, find ranges of probabilities for each decision boundary
    range_1 = (max_value_y, min_value_y)
    range_2 = (max_value_y, min_value_y)
    decision_boundary = -1
    min_cost = 9999
    max_cost = -99999
    sorted_pred = normalised_label[np.argsort(proj)]
    sorted_proj = np.sort(proj)
    assert len(points) > 3
    normalisation_quotient = max_value_y - min_value_y
    costs = []
    # true_label = predicted_label >= min_value_y + (max_value_y - min_value_y) * 0.7
    min_percentage = 0.1  # the minimum relative distance of a slice
    mode = 0

    def f(i):
        true_label = np.array(range(len(sorted_pred))) > i
        if mode == 0:
            cost = sklearn.metrics.log_loss(true_label, sorted_pred.astype(float))  # .astype(int)
        elif mode == 1:
            cost = scipy.stats.entropy(true_label.astype(int), sorted_pred.astype(float))
        else:
            raise Exception("Unvalid choice")
        return cost

    method = "scipy"
    if method == "scipy":
        bounds = max(1, int(min_percentage * len(sorted_pred))), min(len(sorted_pred) - 1, len(sorted_pred) - int(min_percentage * len(sorted_pred)))
        temp_cost = minimize_scalar(f, bounds=bounds, method='bounded')
        decision_boundary = int(temp_cost.x)
        min_cost = temp_cost.fun
        return decision_boundary, min_cost, sorted_proj, proj2d, costs, False  # last boolean is for not ignoring
    else:
        for i in range(max(1, int(min_percentage * len(sorted_pred))), min(len(sorted_pred) - 1, len(sorted_pred) - int(min_percentage * len(sorted_pred)))):
            # range_1_temp = (min(min(sorted_pred[0:i]), range_1[0]), max(max(sorted_pred[0:i]), range_1[1]))
            # range_2_temp = (min(min(sorted_pred[i:]), range_2[0]), max(max(sorted_pred[i:]), range_2[1]))
            # range_1_inv = (1 - range_1_temp[1], 1 - range_1_temp[0])
            # tolerance = (max_value_y - min_value_y) * 0.0
            # cost = (max(0, abs(range_1_temp[0] - range_1_temp[1]) - tolerance) / normalisation_quotient) + (
            #         max(0, abs(range_2_temp[0] - range_2_temp[1]) - tolerance) / normalisation_quotient)  # todo check
            # cost = cost ** 2
            true_label = np.array(range(len(sorted_pred))) > i
            if np.all(true_label) or not np.any(true_label):
                continue
            if mode == 0:
                cost = sklearn.metrics.log_loss(true_label, sorted_pred.astype(float))  # .astype(int)
            elif mode == 1:
                cost = scipy.stats.entropy(true_label.astype(int), sorted_pred.astype(float))
            else:
                raise Exception("Unvalid choice")
            costs.append(cost)
            if cost < min_cost:
                # range_1 = range_1_temp
                # range_2 = range_2_temp
                min_cost = cost
                decision_boundary = i
        # plot_list(costs)
        return decision_boundary, min_cost, sorted_proj, proj2d, costs
