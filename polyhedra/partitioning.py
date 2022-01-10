import gurobi as grb
import numpy as np
import ray
import scipy
import sklearn
import torch
from scipy.optimize import minimize_scalar

import runnables.runnable.templates.polytope as polytope
from polyhedra.milp_methods import generate_input_region, optimise
from polyhedra.plot_utils import project_to_dimension


def sample_and_split(pre_nn, nn, template, boundaries, env_input_size, template_2d, action=0, minimum_length=0.1, use_softmax=True):
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
    samples_ontput = nn(preprocessed)
    if use_softmax:
        samples_ontput = torch.softmax(samples_ontput, 1)
    predicted_label = samples_ontput.detach().numpy()[:, action]
    # template_2d: np.ndarray = np.array([Experiment.e(env_input_size, 2), Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)])
    at_least_one_valid_dimension = False
    dimension_lengths = []
    for i, dimension in enumerate(template):
        inverted_dimension = find_inverted_dimension(-dimension, template)
        dimension_length = boundaries[i] + boundaries[inverted_dimension]
        dimension_lengths.append(dimension_length)
        if dimension_length > minimum_length:
            at_least_one_valid_dimension = True
    if at_least_one_valid_dimension:
        chosen_dimension, decision_point = find_dimension_split3(samples, predicted_label, template, template_2d, dimension_lengths, minimum_length)
        if decision_point is not None:
            split1, split2 = split_polyhedron_milp(template, boundaries, chosen_dimension, decision_point)
            return split1, split2
        else:
            raise Exception("could not find a split that satisfy the minimum length, consider increasing minimum_length parameter")
        # print("done")
        # plot_points_and_prediction(samples@template_2d.T, predicted_label)
        # show_polygons(template, [split1, split2], template_2d)
    else:
        raise Exception("could not find a split that satisfy the minimum length, consider increasing minimum_length parameter")


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
    input = generate_input_region(gurobi_model, template, boundaries, len(split_template))
    gurobi_model.update()
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    gurobi_model.addConstr(sum((split_template[i] * input[i]) for i in range(len(split_template))) >= decision_point)
    split1 = optimise(template, gurobi_model, input)
    assert split1 is not None
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
    split_template = template[chosen_dimension]
    input = generate_input_region(gurobi_model, template, boundaries, len(split_template))
    gurobi_model.update()
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    gurobi_model.addConstr(sum((split_template[i] * input[i]) for i in range(len(split_template))) <= decision_point)
    split2 = optimise(template, gurobi_model, input)
    assert split2 is not None
    return split1, split2


def find_inverted_dimension(inverted_value, template):
    inverted_dimension = -1
    for i, val in enumerate(template):
        if np.array_equal(inverted_value, val):
            inverted_dimension = i
            break
    # assert inverted_dimension != -1, "Could not find inverted dimension"
    return inverted_dimension


def is_split_range(ranges_probs, max_prob_difference=0.2):
    split_flag = False
    for chosen_action in range(len(ranges_probs)):
        prob_diff = ranges_probs[chosen_action][1] - ranges_probs[chosen_action][0]
        if prob_diff > max_prob_difference:
            # should split the input
            split_flag = True
            break
    return split_flag


# noinspection PyUnreachableCode
def find_dimension_split3(points, predicted_label, template, template2d, dimension_lengths, minimum_length=0.1):
    costs = []
    decision_boundaries = []
    proj2ds = []
    proj2dsmax = []
    classifier_predictions = []
    indices = []
    proc_ids = []
    for dimension in template:
        proc_id = remote_find_minimum.remote(dimension, points, predicted_label)
        proc_ids.append(proc_id)
    values = ray.get(proc_ids)
    for i, value in enumerate(values):
        decision_boundary, min_cost, sorted_proj, proj2d, proj2dmax, plot_costs, ignore = value
        # plot_list(plot_costs)
        dimension_lengths_i_ = dimension_lengths[i]
        if dimension_lengths_i_ > minimum_length and not ignore:  # tries to ignore extremely small dimensionss
            costs.append(min_cost)
            decision_boundaries.append(sorted_proj[decision_boundary] if sorted_proj is not None else None)
            indices.append(decision_boundary)
            proj2ds.append(proj2d)
            proj2dsmax.append(proj2dmax)
        else:
            costs.append(float("inf"))
            decision_boundaries.append(None)
            indices.append(None)
            proj2ds.append(None)
            proj2dsmax.append(None)
    # chosen_dimension = np.argmin([((i - len(points) / 2) / len(points)) ** 2 for i in indices])
    # slices_sizes = np.array([(i / len(points)) for i in indices])
    # norm_costs = np.array(costs) / np.max(costs)
    alpha = 1  # hyperparameter to decide the importance of cost vs the size of the slice
    chosen_dimension = np.argmin(np.array(costs))
    # chosen_dimension = np.argmin( - (1 - alpha) * slices_sizes) #norm_costs * alpha
    # chosen_dimension = np.argmin((slices_sizes - 0.5) ** 2)  # try to halve the shape
    max_value_y = np.max(predicted_label)
    min_value_y = np.min(predicted_label)
    delta_y = max_value_y - min_value_y
    normalised_label = (predicted_label - min_value_y) / delta_y
    # plot_points_and_prediction(points@template2d.T, normalised_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], normalised_label)
    # plot_points_and_prediction(proj2ds[chosen_dimension], [1 if x[0]>decision_boundaries[chosen_dimension] else 0 for x in proj2ds[chosen_dimension]])
    # plot_points_and_prediction(proj2dsmax[chosen_dimension], [x[1] for x in proj2dsmax[chosen_dimension]])
    # plot_points_and_prediction(proj2dsmax[chosen_dimension], [1 if x[0]>decision_boundaries[chosen_dimension] else 0 for x in proj2dsmax[chosen_dimension]])
    # plot_points_and_prediction(points@template2d.T, [1 if x>decision_boundaries[chosen_dimension] else 0 for x in points@template[chosen_dimension]])
    return chosen_dimension, decision_boundaries[chosen_dimension]


@ray.remote
def remote_find_minimum(dimension, points, predicted_label):
    normalise = True
    proj = project_to_dimension(points, dimension)
    proj2d = np.array([[x, predicted_label[i]] for i, x in enumerate(proj)])
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
    if normalise and delta_y != 0:
        normalised_label = (predicted_label - min_value_y) / delta_y
    else:
        normalised_label = predicted_label
    # enumerate all points, find ranges of probabilities for each decision boundary
    range_1 = (max_value_y, min_value_y)
    range_2 = (max_value_y, min_value_y)
    decision_boundary = -1
    min_cost = 9999
    max_cost = -99999
    sorted_pred = normalised_label[np.argsort(proj)]
    # sorted_pred = np.array([1.0 if x > 0.5 else 0 for x in sorted_pred])  # simple classifier
    sorted_pred = np.clip(sorted_pred, 1e-7, 1 - 1e-7)
    sorted_proj = np.sort(proj)
    assert len(points) > 3
    normalisation_quotient = max_value_y - min_value_y
    costs = []
    # true_label = predicted_label >= min_value_y + (max_value_y - min_value_y) * 0.7
    min_percentage = 0.05  # the minimum relative distance of a slice
    only_max = []
    max_so_far = 0
    for i, x in enumerate(sorted_pred):
        max_so_far = max(max_so_far, x)
        only_max.append(max_so_far)
    only_max = np.array(only_max)
    proj2dmax = np.array([[x, only_max[i]] for i, x in enumerate(sorted_proj)])
    mode = 0
    back_forth = 0

    # plot_points_and_prediction(proj2d, only_max)

    def f(i):
        n_bins = 100
        true_label = np.array(range(len(only_max))) > i if back_forth == 0 else np.array(range(len(only_max))) < i
        if mode == 0:  # sorted_pred
            histogram = np.histogram(sorted_pred.astype(float), n_bins, range=(0.0, 1.0))
            digitized = np.digitize(sorted_pred.astype(float), np.linspace(0, 1.0, n_bins))
            try:
                cost = sklearn.metrics.log_loss(true_label, sorted_pred.astype(float), sample_weight=1 / np.where(histogram[0] == 0, 1, histogram[0])[digitized])  # .astype(int)
            except:
                print("erro")
        elif mode == 1:  # onlymax
            histogram = np.histogram(only_max.astype(float), n_bins, range=(0.0, 1.0))
            digitized = np.digitize(only_max.astype(float), np.linspace(0, 1.0, n_bins))
            cost = sklearn.metrics.log_loss(true_label, only_max.astype(float), sample_weight=1 / histogram[0][digitized])
        elif mode == 2:
            decision_boundary = int(i)
            group1 = only_max[:decision_boundary]
            group2 = only_max[decision_boundary:]
            cost = (np.max(group1) - np.min(group1)) ** 2 + (np.max(group2) - np.min(group2)) ** 2
        elif mode == 3:
            # filter = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            # true_label = np.array(filter)
            # slice_probabilities = only_max.astype(float)[int(i) - (len(filter) // 2):int(i) + (len(filter) // 2)]
            # cost1 = sklearn.metrics.log_loss(true_label, slice_probabilities)
            filter = [1] * 50 + [0] * 50  # [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            true_label = np.array(filter)
            slice_probabilities = only_max.astype(float)[int(i) - (len(filter) // 2):int(i) + (len(filter) // 2)]
            cost2 = sklearn.metrics.log_loss(true_label, slice_probabilities)
            cost = cost2
        elif mode == 4:
            true_label = np.array(range(len(only_max))) > i
            histogram = np.histogram(sorted_pred.astype(float), n_bins, range=(0.0, 1.0))
            digitized = np.digitize(sorted_pred.astype(float), np.linspace(0, 1.0, n_bins))
            eps = 1e-15
            y_pred = np.clip(true_label.astype(float), eps, 1 - eps)
            # cost = -np.average(sorted_pred.astype(float) * np.log2(y_pred)) #,weights=1 / np.where(histogram[0] == 0, 1, histogram[0])[digitized]
            cost = np.dot(sorted_pred.astype(float) * np.log2(y_pred), 1 / np.where(histogram[0] == 0, 1, histogram[0])[digitized])
            # cost = sklearn.metrics.log_loss(sorted_pred.astype(float), true_label.astype(float), sample_weight=1 / np.where(histogram[0] == 0, 1, histogram[0])[digitized])  # .astype(int)
        else:
            raise Exception("Unvalid choice")
        return cost

    method = "scipy"
    start = max(1, int(min_percentage * len(sorted_pred)))
    stop = min(len(sorted_pred) - 1, len(sorted_pred) - int(min_percentage * len(sorted_pred)))
    if method == "scipy":
        bounds = start, stop
        back_forth = 0
        temp_cost1 = minimize_scalar(f, bounds=bounds, method='bounded')
        decision_boundary1 = int(temp_cost1.x)
        min_cost1 = temp_cost1.fun
        # decision_value = sorted_proj[decision_boundary1]
        back_forth = 1
        temp_cost2 = minimize_scalar(f, bounds=bounds, method='bounded')
        decision_boundary2 = int(temp_cost2.x)
        min_cost2 = temp_cost2.fun
        if min_cost1 < min_cost2:
            return decision_boundary1, min_cost1, sorted_proj, proj2d, proj2dmax, costs, False  # last boolean is for not ignoring
        else:
            return decision_boundary2, min_cost2, sorted_proj, proj2d, proj2dmax, costs, False  # last boolean is for not ignoring
        # min_cost = sklearn.metrics.log_loss(np.array(range(len(only_max))) > decision_boundary, sorted_pred.astype(float))
        # min_cost = min(min_cost, sklearn.metrics.log_loss(np.array(range(len(only_max))) < decision_boundary, sorted_pred.astype(float)))
        # plot_points_and_prediction(proj2d, [1 if x > decision_value else 0 for x in sorted_proj])
        # plot_points_and_prediction(proj2d, only_max)
        # return decision_boundary, min_cost, sorted_proj, proj2d, proj2dmax, costs, False  # last boolean is for not ignoring
    elif method == "custom":
        for i in range(start, stop):  # , (stop - start) // 100
            true_label = np.array(range(len(only_max))) > i
            if np.all(true_label) or not np.any(true_label):
                continue
            if mode == 0:
                cost = sklearn.metrics.log_loss(true_label, sorted_pred.astype(float))  # .astype(int)
            elif mode == 1:
                cost = scipy.stats.entropy(true_label.astype(int), sorted_pred.astype(float))
            elif mode == 2:
                cost = sklearn.metrics.log_loss(true_label, np.array([1.0 if x > 0.5 else 0 for x in sorted_pred]))  # .astype(int)
            elif mode == 3:
                filter = [1] * 50 + [0] * 50  # [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                true_label = np.array(filter)
                slice_probabilities = only_max.astype(float)[int(i) - (len(filter) // 2):int(i) + (len(filter) // 2)]
                cost = sklearn.metrics.log_loss(true_label, slice_probabilities)
            else:
                raise Exception("Unvalid choice")
            costs.append(cost)
            if cost < min_cost:
                # range_1 = range_1_temp
                # range_2 = range_2_temp
                min_cost = cost
                decision_boundary = i
        # plot_list(costs)
        decision_value = sorted_proj[decision_boundary]
        # plot_points_and_prediction(proj2d, [1 if x > decision_value else 0 for x in sorted_proj])
        return decision_boundary, min_cost, sorted_proj, proj2d, costs, False
    else:
        decision_boundary = next(i for i, v in enumerate(sorted_pred) if v > 0.5)
        decision_boundary = max(decision_boundary, start)
        decision_boundary = min(decision_boundary, stop)
        group1 = sorted_pred[:decision_boundary]
        group2 = sorted_pred[decision_boundary:]
        min_cost = (np.max(group1) - np.min(group1)) ** 2 + (np.max(group2) - np.min(group2)) ** 2
        decision_value = sorted_proj[decision_boundary]
        return decision_boundary, min_cost, sorted_proj, proj2d, costs, False


def sigmoid(x):
    ex = np.exp(x)
    return ex / (1 + ex)
