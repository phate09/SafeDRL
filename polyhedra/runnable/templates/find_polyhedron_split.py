from polyhedra.experiments_nn_analysis import Experiment
import numpy as np


def split_polyhedron(template, boundaries, dimension, decision_point=None):
    # todo bugged for diagonals
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


def find_inverted_dimension(inverted_value, template):
    inverted_dimension = -1
    for i, val in enumerate(template):
        if np.array_equal(inverted_value, val):
            inverted_dimension = i
            break
    assert inverted_dimension != -1, "Could not find inverted dimension"
    return inverted_dimension


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


def sample_points(template, boundaries, n_samples: int):
    pass


if __name__ == '__main__':
    template = Experiment.octagon(2)
    boundaries = (1, -1, 1, 1, 1, 1, 1, 1)

    # pick dimension
    dimension = pick_longest_dimension(template, boundaries)
    split_polyhedron(template, boundaries, dimension)
