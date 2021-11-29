import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from polyhedra.experiments_nn_analysis import Experiment


def merge_regions(template: np.ndarray, items: np.ndarray):
    """Takes the maximum in each dimension"""
    merged = np.amax(items, 0)
    return merged


def compute_volume(template: np.ndarray, item: np.ndarray):
    vertices = pypoman.compute_polytope_vertices(template, item)
    volume = ConvexHull(vertices).volume
    return volume


def merge_with_volume_analysis(template: np.ndarray, items: np.ndarray):
    merged = merge_regions(template, items)
    volume_new = compute_volume(template, merged)
    volume_old = 0
    for item in items:
        volume = compute_volume(template, item)
        volume_old += volume
    return merged, volume_new, volume_old


def main():
    template = Experiment.box(2)
    item1 = [1, 1, 1, 1]
    item2 = [2, 0, 2, 0]
    merged1 = merge_regions(template, np.array([item1, item2]))
    volume = compute_volume(template, merged1)
    merged, volume_new, volume_old = merge_with_volume_analysis(template, np.array([item1, item2]))


if __name__ == '__main__':
    main()
