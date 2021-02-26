from collections import defaultdict
from functools import partial
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from numpy.random import default_rng
import plotly.graph_objects as go
from mosaic.utils import PolygonSort
from polyhedra.experiments_nn_analysis import Experiment, contained
from polyhedra.plot_utils import show_polygon_list3, transform_vertices2, windowed_projection, compute_polygon_trace
from runnables.tests.run_merge_templates import compute_volume, merge_with_volume_analysis


def compute_volume_ga(items: np.ndarray, template, indices: np.ndarray):
    filtered = items[indices.astype(dtype="bool")]
    merged, volume_new, volume_old = merge_with_volume_analysis(template, filtered)
    return (volume_new - volume_old) / volume_old


def run_genetic_merge(items, template):
    algorithm_param = {'max_num_iteration': 3000,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 150}
    f = partial(compute_volume_ga, items, template)
    np.random.seed(0)
    model = ga(function=f,
               dimension=len(items),
               variable_type='bool',
               algorithm_parameters=algorithm_param)
    model.run()

    return model.best_variable.astype(dtype="bool")


def generate_intervals(n):
    rng = default_rng(seed=0)
    centres1 = rng.normal(0, 2, n)
    centres2 = rng.normal(0, 2, n)
    heights = np.abs(rng.normal(0, 1, n))
    widths = np.abs(rng.normal(0, 1, n))
    intervals = []
    for i in range(n):
        interval = (centres1[i] + heights[i] / 2, -(centres1[i] - heights[i] / 2), centres2[i] + widths[i] / 2, -(centres2[i] - widths[i] / 2))
        intervals.append(interval)
    return intervals


def plot_intervals(intervals, template):
    intervals_dict = defaultdict(list)
    for i, interval in enumerate(intervals):
        intervals_dict[i].append(interval)
    template2d = np.array([[1, 0], [0, 1]])
    fig, projected_points = show_polygon_list3(intervals_dict, "x_axis_title", "y_axis_title", template, template2d)
    return fig


def main():
    template = Experiment.box(2)
    intervals = generate_intervals(200)
    items_array = np.array(intervals)
    fig = plot_intervals(intervals, template)
    fig.show()
    # best_variable = run_genetic_merge(items_array, template)
    best_variable = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                              1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
                              0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                              1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                              1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                              0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                              0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
                              0, 0, 1, 1, 0, 1, 1, 1, ], dtype="bool")
    filtered = items_array[best_variable]
    merged, volume_new, volume_old = merge_with_volume_analysis(template, filtered)
    remaining_intervals: np.ndarray = items_array[np.invert(best_variable)]
    merged_intermediate = np.append(remaining_intervals, np.expand_dims(merged, 0), 0)
    fig = plot_intervals(merged_intermediate, template)
    fig.show()
    merged_final = merged_intermediate.copy()
    interval: np.ndarray
    for interval in merged_intermediate:
        merged_final = [x for x in merged_final if np.equal(interval, x).all() or not contained(x, interval)]
    fig = plot_intervals(merged_final, template)
    fig.show()
    print("done")


if __name__ == '__main__':
    main()
